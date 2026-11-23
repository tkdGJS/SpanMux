# SPDX-License-Identifier: Apache-2.0
"""Process-local wakeups for the experimental P/D engine loop.

The P/D phase scheduler and the async UniProc worker run in different host
threads of the same process.  This module lets model-future, request-input,
and tiered-D2H completion paths wake EngineCore without a millisecond polling
loop.  A bounded timeout remains the liveness fallback for external progress
sources that cannot publish a callback (for example, a remote KV connector).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any


_PROGRESS_EVENTS: set[threading.Event] = set()
_PROGRESS_EVENTS_LOCK = threading.Lock()
_PROGRESS_GENERATION = 0
_PD_FUTURE_LIFECYCLE_ATTR = "_vllm_pd_future_lifecycle"


@dataclass(slots=True)
class PDAsyncFutureLifecycle:
    """Scalar-only lifecycle state attached to one async P/D Future.

    The record deliberately owns no SchedulerOutput, tensor, CUDA object, or
    request reference.  It exists only to identify the last host-side stage
    reached when a Future does not publish its terminal callback.
    """

    method: str
    execution_key: str
    created_ts: float = field(default_factory=time.monotonic)
    _stage: str = field(default="created", init=False, repr=False)
    _stage_seq: int = field(default=0, init=False, repr=False)
    _stage_ts: float = field(default=0.0, init=False, repr=False)
    _terminal: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._stage_ts = self.created_ts

    def advance(self, stage: str, *, terminal: bool = False) -> bool:
        """Publish a monotonic stage, rejecting updates after terminal state."""

        now = time.monotonic()
        with self._lock:
            if self._terminal:
                return False
            self._stage = str(stage)
            self._stage_seq += 1
            self._stage_ts = now
            self._terminal = bool(terminal)
            return True

    def snapshot(self) -> dict[str, float | str]:
        now = time.monotonic()
        with self._lock:
            return {
                "pd_future_lifecycle_method": self.method,
                "pd_future_lifecycle_execution_key": self.execution_key,
                "pd_future_lifecycle_stage": self._stage,
                "pd_future_lifecycle_stage_seq": float(self._stage_seq),
                "pd_future_lifecycle_terminal": float(self._terminal),
                "pd_future_lifecycle_age_s": max(
                    0.0, now - self.created_ts
                ),
                "pd_future_lifecycle_stage_age_s": max(
                    0.0, now - self._stage_ts
                ),
            }


def attach_pd_future_lifecycle(
    future: Future[Any],
    *,
    method: str,
    execution_key: str | None,
) -> PDAsyncFutureLifecycle:
    """Attach one lifecycle record and observe its terminal Future state."""

    lifecycle = PDAsyncFutureLifecycle(
        method=str(method),
        execution_key=str(execution_key or ""),
    )
    setattr(future, _PD_FUTURE_LIFECYCLE_ATTR, lifecycle)

    def completed(done_future: Future[Any]) -> None:
        if done_future.cancelled():
            lifecycle.advance("terminal_cancelled", terminal=True)
            return
        try:
            error = done_future.exception()
        except BaseException:
            lifecycle.advance("terminal_error", terminal=True)
            return
        lifecycle.advance(
            "terminal_error" if error is not None else "terminal_success",
            terminal=True,
        )

    future.add_done_callback(completed)
    return lifecycle


def pd_future_lifecycle_snapshot(
    future: Future[Any],
) -> dict[str, float | str]:
    """Return a safe snapshot for tracked and mixed-version Futures."""

    lifecycle = getattr(future, _PD_FUTURE_LIFECYCLE_ATTR, None)
    snapshot = getattr(lifecycle, "snapshot", None)
    if not callable(snapshot):
        return {
            "pd_future_lifecycle_stage": "untracked",
            "pd_future_lifecycle_stage_seq": -1.0,
            "pd_future_lifecycle_terminal": float(future.done()),
            "pd_future_lifecycle_age_s": -1.0,
            "pd_future_lifecycle_stage_age_s": -1.0,
            "pd_future_lifecycle_method": "",
            "pd_future_lifecycle_execution_key": "",
        }
    return snapshot()


def register_pd_engine_progress_event(event: threading.Event) -> None:
    with _PROGRESS_EVENTS_LOCK:
        _PROGRESS_EVENTS.add(event)


def unregister_pd_engine_progress_event(event: threading.Event) -> None:
    with _PROGRESS_EVENTS_LOCK:
        _PROGRESS_EVENTS.discard(event)


def notify_pd_engine_progress() -> None:
    """Wake every in-process EngineCore using the P/D event loop."""

    global _PROGRESS_GENERATION
    with _PROGRESS_EVENTS_LOCK:
        _PROGRESS_GENERATION += 1
        events = tuple(_PROGRESS_EVENTS)
    for event in events:
        event.set()


def pd_engine_progress_generation() -> int:
    with _PROGRESS_EVENTS_LOCK:
        return _PROGRESS_GENERATION


def wait_for_pd_engine_progress(
    event: threading.Event,
    ready: Callable[[], bool],
    *,
    timeout_s: float | None,
    observed_generation: int | None = None,
) -> bool:
    """Wait without losing a completion racing with ``Event.clear``.

    ``ready`` must be a cheap, side-effect-free check such as Future.done().
    The check-clear-check sequence covers a callback that fires immediately
    before the waiter clears a previously consumed notification.
    """

    baseline = (
        pd_engine_progress_generation()
        if observed_generation is None
        else int(observed_generation)
    )
    if ready() or pd_engine_progress_generation() != baseline:
        return True
    event.clear()
    if ready() or pd_engine_progress_generation() != baseline:
        return True
    timeout = None if timeout_s is None else max(0.0, float(timeout_s))
    event.wait(timeout=timeout)
    return ready() or pd_engine_progress_generation() != baseline
