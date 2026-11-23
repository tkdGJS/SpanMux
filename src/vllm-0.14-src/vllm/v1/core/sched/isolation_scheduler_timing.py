# SPDX-License-Identifier: Apache-2.0
"""Opt-in, CPU-only schedule-call timing, mirrored in Native and vendor.

This is not GPU time or request queueing time. Only scalar snapshots enter the
writer queue; no Request, SchedulerOutput, tensor, or KV ownership is transferred.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)
SCHEMA = "scheduler_call_v1"


class SchedulerTimingRecorder:
    def __init__(self, path: str, every_n: int = 100, max_pending: int = 4096):
        if every_n < 1 or max_pending < 1:
            raise ValueError("scheduler timing intervals/queue capacity must be positive")
        self.path = path
        self.every_n = every_n
        self.pid = os.getpid()
        self.calls = self.sampled = self.written = self.dropped = self.errors = 0
        self.metadata_errors = 0
        self.close_timed_out = False
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(max_pending)
        self._closing = threading.Event()
        self._failed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="SchedulerTimingWriter")
        self._thread.start()

    def wrap(self, method):
        @functools.wraps(method)
        def measured(*args, **kwargs):
            call_id = self.calls
            self.calls += 1
            if call_id % self.every_n or self._closing.is_set() or self._failed.is_set():
                return method(*args, **kwargs)
            self.sampled += 1
            start = time.monotonic()
            try:
                output = method(*args, **kwargs)
            except BaseException as exc:
                finish = time.monotonic()
                self._record(call_id, start, finish, None, method.__self__,
                             type(exc).__name__)
                raise
            finish = time.monotonic()
            self._record(call_id, start, finish, output, method.__self__, None)
            return output
        return measured

    def _record(self, call_id, start, finish, output, scheduler, error):
        # Metadata extraction and enqueue are outside the measured body.
        record = {
            "schema": SCHEMA, "pid": self.pid, "iteration_id": call_id,
            "sample_every_n": self.every_n,
            "schedule_start_ts": start, "schedule_end_ts": finish,
            "schedule_duration_s": finish - start,
            "schedule_status": "error" if error else "ok",
            "error_type": error,
            "scheduled_phase": "error" if error else "idle",
            "scheduled_num_seqs": 0, "total_num_scheduled_tokens": 0,
        }
        try:
            if output is not None:
                counts = output.num_scheduled_tokens
                record["scheduled_num_seqs"] = len(counts)
                record["total_num_scheduled_tokens"] = int(output.total_num_scheduled_tokens)
                if counts:
                    phase = getattr(output, "scheduled_phase", None)
                    if phase not in ("prefill", "decode"):
                        phases = set()
                        for rid, tokens in counts.items():
                            request = scheduler.requests[rid]
                            # Both schedulers advance computed tokens before
                            # return. This also identifies a 1-token final
                            # Prefill correctly (token count alone does not).
                            before = request.num_computed_tokens - tokens
                            phases.add("prefill" if before < request.num_prompt_tokens
                                       else "decode")
                        phase = phases.pop() if len(phases) == 1 else "mixed"
                    record["scheduled_phase"] = phase
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.metadata_errors += 1
            self.dropped += 1
            if self.metadata_errors == 1:
                logger.exception("Scheduler timing metadata failed; inference is unchanged")

    def stats(self):
        return {
            "schema": SCHEMA, "pid": self.pid, "sample_every_n": self.every_n,
            "calls": self.calls, "sampled": self.sampled, "written": self.written,
            "dropped": self.dropped, "errors": self.errors,
            "metadata_errors": self.metadata_errors,
            "close_timed_out": self.close_timed_out,
        }

    def _run(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            # An old trace must not be silently merged into a new experiment.
            with open(self.path, "x", encoding="utf-8") as handle:
                last_flush = time.monotonic()
                while not self._closing.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        handle.flush()
                        last_flush = time.monotonic()
                        continue
                    handle.write(json.dumps(record) + "\n")
                    self.written += 1
                    now = time.monotonic()
                    if self.written % 64 == 0 or now - last_flush >= 0.25:
                        handle.flush()
                        last_flush = now
        except Exception:
            self.errors += 1
            self._failed.set()
            logger.exception("Scheduler timing writer failed; inference is unchanged")
        finally:
            status = self.stats()
            status["complete"] = bool(self._closing.is_set() and not self.errors
                                      and not self.close_timed_out
                                      and self.sampled == self.written + self.dropped)
            try:
                with open(self.path + ".status.json", "x", encoding="utf-8") as handle:
                    json.dump(status, handle, indent=2)
            except Exception:
                logger.exception("Could not write scheduler timing terminal status")

    def close(self, timeout_s: float = 2.0):
        self._closing.set()
        self._thread.join(timeout=max(0.0, timeout_s))
        if self._thread.is_alive():
            self.close_timed_out = True
            logger.warning("Scheduler timing flush timed out; trace may be incomplete: %s",
                           self.path)
        elif self.dropped or self.errors:
            logger.warning("Scheduler timing final status: %s", self.stats())


def install_scheduler_timing(scheduler) -> None:
    """Wrap the instance's schedule method only when explicitly requested.

    Installed at the end of Scheduler.__init__, so AsyncScheduler inherits the
    same boundary and vendor schedule_pd_phase delegates into it exactly once.
    Default off leaves the original bound method untouched.
    """
    path = os.environ.get("VLLM_ISOLATION_SCHEDULER_TIMING_JSONL", "").strip()
    if not path:
        return
    every_n = int(os.environ.get("VLLM_ITERATION_METRICS_SAMPLE_EVERY_N", "100"))
    recorder = SchedulerTimingRecorder(path, every_n=every_n)
    scheduler._isolation_scheduler_timing = recorder
    scheduler.schedule = recorder.wrap(scheduler.schedule)


def close_scheduler_timing(scheduler) -> None:
    recorder = getattr(scheduler, "_isolation_scheduler_timing", None)
    if recorder is not None:
        recorder.close()
