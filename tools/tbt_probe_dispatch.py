"""Opt-in calibration client control. Never owns runtime Futures, KV or SMs.

The append-only Engine capture is read in bounded chunks on the HTTP client's
event loop. A gate controls only requests that have not yet been sent.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
import re
import time


def load_control(path, observations, workload_path):
    data = json.loads(Path(path).read_text())
    ids = [o.request.request_id for o in observations]
    digest = hashlib.sha256(Path(workload_path).read_bytes()).hexdigest()
    if (data.get("schema_version") != 1 or data.get("workload_sha256") != digest
            or data.get("request_ids") != ids):
        raise ValueError("probe control schema/trace hash/ordered IDs mismatch")
    for key in ("gate_timeout_sec", "no_progress_sec", "poll_sec"):
        value = float(data[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid probe {key}")
    if not .01 <= float(data["poll_sec"]) <= 1:
        raise ValueError("probe poll_sec must be 0.01..1 seconds")
    if not isinstance(data.get("require_completion"), bool):
        raise ValueError("probe require_completion must be boolean")
    if not isinstance(data.get("capture_path"), str) or not data["capture_path"]:
        raise ValueError("probe capture_path required")
    gates = data.get("gates", {})
    if not isinstance(gates, dict) or not set(gates) <= set(ids):
        raise ValueError("probe gates reference unknown request IDs")
    for rid, gate in gates.items():
        earlier = set(ids[:ids.index(rid)])
        dependencies = set(gate.get("tokens_from", [])) | set(gate.get("complete_from", []))
        if gate.get("prefill_from"):
            dependencies.add(gate["prefill_from"])
        if not dependencies or not dependencies <= earlier:
            raise ValueError("probe gate must depend only on earlier requests")
        if int(gate.get("min_tokens", 1)) < 1 or int(gate.get("prefill_tokens", 0)) < 0:
            raise ValueError("invalid probe gate token threshold")
        if gate.get("prefill_mode") not in (None, "native", "ring"):
            raise ValueError("invalid probe gate mode")
        delay = float(gate.get("settle_sec", 0))
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("invalid probe settle_sec")
    return data


class CaptureProgress:
    """Incremental scalar reader; partial final lines wait for the next poll."""

    def __init__(self, path, request_ids, start):
        self.path = Path(path)
        self.request_ids = tuple(request_ids)
        self.start = start
        self.offset = 0
        self.buffer = b""
        self.sequence = 0
        self.timestamp = -math.inf
        self.last_progress = start
        self.outstanding = {}
        self.prefill_progress = {}
        self.modes = {}
        self.events = 0

    def client_id(self, server_id):
        matches = [rid for rid in self.request_ids if server_id == rid or re.fullmatch(
            rf"cmpl-{re.escape(rid)}-0(?:-[0-9a-f]{{8}})?", server_id)]
        return matches[0] if len(matches) == 1 else None

    def accept(self, event):
        sequence, timestamp = int(event["sequence"]), float(event["timestamp"])
        if (sequence != self.sequence + 1 or not math.isfinite(timestamp)
                or timestamp < self.timestamp):
            raise ValueError("probe capture sequence gap/reordered clock")
        if sequence == 1 and (event.get("event") != "start"
                             or event.get("contract") != "tbt-capture-v1"):
            raise ValueError("probe capture contract missing")
        self.sequence, self.timestamp = sequence, timestamp
        if timestamp < self.start:
            return
        self.events += 1
        kind = event["event"]
        for prefill in event.get("prefills", []):
            self.modes[int(prefill["submit_seq"])] = prefill["mode"]
        if kind == "submit":
            self.outstanding[int(event["submit_seq"])] = event
        elif kind == "recover":
            submit = self.outstanding.pop(int(event["submit_seq"]), None)
            if submit and submit["phase"] == "prefill":
                for row in submit["rows"]:
                    rid = self.client_id(row["request_id"])
                    if rid is not None:
                        self.prefill_progress[rid] = max(self.prefill_progress.get(rid, 0),
                                                        int(row["context_tokens"]))
            self.modes.pop(int(event["submit_seq"]), None)
        if kind in {"submit", "recover", "prefill_boundary", "prefill_resume", "transition_commit"}:
            self.last_progress = timestamp

    def poll(self):
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            raise ValueError("probe capture truncated while reading")
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            block = handle.read(1024 * 1024)
        self.offset += len(block)
        self.buffer += block
        lines = self.buffer.split(b"\n")
        self.buffer = lines.pop()
        if len(self.buffer) > 4 * 1024 * 1024:
            raise ValueError("probe capture line exceeds bounded reader")
        for line in lines:
            if line.strip():
                self.accept(json.loads(line))

    def gate(self, gate, observations):
        for rid in gate.get("tokens_from", []):
            obs = observations[rid]
            # Even a completed stream with enough historical tokens cannot
            # establish live interference for an as-yet-unsent request.
            if obs.completion_ts is not None:
                return "failed", f"decode_finished_before_gate:{rid}"
            if obs.output_tokens < int(gate.get("min_tokens", 1)):
                return "wait", f"decode_tokens:{rid}"
        for rid in gate.get("complete_from", []):
            obs = observations[rid]
            if obs.status == "failed":
                return "failed", f"dependency_failed:{rid}"
            if obs.status != "success":
                return "wait", f"request_completion:{rid}"
        rid = gate.get("prefill_from")
        if rid and self.prefill_progress.get(rid, 0) < int(gate["prefill_tokens"]):
            if observations[rid].completion_ts is not None:
                return "failed", f"prefill_finished_before_gate:{rid}"
            return "wait", f"prefill_recovered_progress:{rid}"
        mode = gate.get("prefill_mode")
        if mode and not any(self.modes.get(seq) == mode for seq in self.outstanding):
            return "wait", f"observed_active_prefill_mode:{mode}"
        return "ready", "observed"


async def run_probe(observations, control, *, execute, trace_start, deadline):
    """Return health metadata after cancelling/awaiting only our HTTP tasks."""
    if deadline is None or deadline <= trace_start:
        raise ValueError("event-gated probes require a finite time/cancel window")
    by_id = {o.request.request_id: o for o in observations}
    capture = CaptureProgress(control["capture_path"], by_id, trace_start)
    tasks, opened, records = {}, {}, []
    next_index = 0
    gate_started = trace_start
    last_dispatch = trace_start
    last_report = trace_start
    reason, healthy = "time_limit", True
    last_wait = ""
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            capture.poll()
            if any(o.status == "failed" for o in observations):
                reason, healthy = "request_failed", False
                break
            if any(task.done() and (task.cancelled() or task.exception() is not None
                                   or by_id[rid].status != "success")
                   for rid, task in tasks.items()):
                reason, healthy = "client_task_ended_without_success", False
                break
            while next_index < len(observations):
                obs = observations[next_index]
                rid = obs.request.request_id
                if now < trace_start + obs.effective_offset_s:
                    break
                gate = control.get("gates", {}).get(rid)
                state, detail = capture.gate(gate, by_id) if gate else ("ready", "ungated")
                last_wait = detail
                if state == "failed":
                    reason, healthy = detail, False
                    break
                if state == "wait":
                    if now - gate_started >= float(control["gate_timeout_sec"]):
                        reason, healthy = f"gate_timeout:{rid}:{detail}", False
                    break
                if gate:
                    opened.setdefault(rid, now)
                    if now < opened[rid] + float(gate.get("settle_sec", 0)):
                        break
                # Record the condition time separately from original trace
                # offsets: this is a controlled probe, NOT real trace replay.
                obs.scheduled_ts = now
                records.append(dict(event="dispatch", request_id=rid, timestamp=now,
                                    offset_sec=now-trace_start, condition=detail))
                print(f"[TBT-PROBE] submit={rid} offset={now-trace_start:.3f}s gate={detail}", flush=True)
                tasks[rid] = asyncio.create_task(execute(obs))
                last_dispatch = gate_started = now
                next_index += 1
            if not healthy:
                break
            if next_index == len(observations) and all(t.done() for t in tasks.values()):
                reason = "all_requests_complete"
                break
            active = any(not task.done() for task in tasks.values())
            last_token = max((o.token_rows[-1]["arrival_monotonic_ts"]
                              for o in observations if o.token_rows), default=trace_start)
            last = max(capture.last_progress, last_dispatch, last_token)
            if now-last_report >= 10:
                print("[TBT-PROBE] " + json.dumps(dict(
                    elapsed_sec=round(now-trace_start, 3), submitted=next_index,
                    complete=sum(o.status == "success" for o in observations),
                    waiting_for=last_wait if next_index < len(observations) else "request_completion",
                    model_futures=len(capture.outstanding),
                    prefill_recovered_tokens=capture.prefill_progress,
                    since_progress_sec=round(now-last, 3))), flush=True)
                last_report = now
            if active and not capture.outstanding and now-last >= float(control["no_progress_sec"]):
                reason, healthy = "no_model_future_progress", False
                break
            await asyncio.sleep(float(control["poll_sec"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reason, healthy = f"control_error:{type(exc).__name__}:{exc}", False
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)
        for obs in observations[next_index:]:
            obs.status = "not_submitted_probe_control"
            obs.error_message = f"controlled probe stopped: {reason}"
        for obs in observations[:next_index]:
            if obs.status == "censored_time_limit":
                obs.error_message = f"controlled probe cancelled HTTP stream: {reason}"
    if next_index < len(observations) or (control["require_completion"]
                                         and any(o.status != "success" for o in observations)):
        healthy = False
    if (reason == "all_requests_complete" and any(o.status != "success" for o in observations)
            or any(o.status not in {"success", "censored_time_limit", "failed"}
                   for o in observations[:next_index])):
        reason, healthy = "client_task_ended_without_success", False
    if not capture.events:
        healthy = False
        reason = "no_production_capture:" + reason
    return dict(schema_version=1, protocol="event-gated-v1", healthy=healthy,
                stop_reason=reason, last_gate_wait=last_wait,
                require_completion=control["require_completion"], dispatch_events=records,
                prefill_recovered_tokens=capture.prefill_progress,
                outstanding_submit_seqs=sorted(capture.outstanding),
                capture_events=capture.events, last_model_progress_ts=capture.last_progress,
                note="No-model-Future progress is a diagnostic, not proof that D2H/GPU work is absent.")
