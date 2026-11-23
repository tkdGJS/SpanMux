# SPDX-License-Identifier: Apache-2.0

import threading
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.pd_engine_progress import (
    attach_pd_future_lifecycle,
    notify_pd_engine_progress,
    pd_engine_progress_generation,
    pd_future_lifecycle_snapshot,
    register_pd_engine_progress_event,
    unregister_pd_engine_progress_event,
    wait_for_pd_engine_progress,
)


def test_registered_engine_event_is_notified() -> None:
    event = threading.Event()
    register_pd_engine_progress_event(event)
    try:
        generation = pd_engine_progress_generation()
        notify_pd_engine_progress()
        assert event.is_set()
        assert pd_engine_progress_generation() >= generation + 1
    finally:
        unregister_pd_engine_progress_event(event)


def test_generation_closes_clear_lost_wakeup_window() -> None:
    event = threading.Event()
    observed_generation = pd_engine_progress_generation()
    notify_pd_engine_progress()

    # Simulate a consumed/stale Event bit.  The monotonic generation still
    # proves that an external producer progressed after the engine step began.
    event.clear()
    assert wait_for_pd_engine_progress(
        event,
        lambda: False,
        timeout_s=0.0,
        observed_generation=observed_generation,
    )


def test_ready_predicate_avoids_wait() -> None:
    event = threading.Event()
    assert wait_for_pd_engine_progress(
        event,
        lambda: True,
        timeout_s=1.0,
    )


def test_callback_owned_wait_has_no_periodic_timeout() -> None:
    event = threading.Event()
    ready = threading.Event()
    wait_returned = threading.Event()
    result: list[bool] = []

    def wait() -> None:
        result.append(
            wait_for_pd_engine_progress(
                event,
                ready.is_set,
                timeout_s=None,
            )
        )
        wait_returned.set()

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        # An unbounded callback-owned wait must still be asleep before a
        # producer publishes progress; this distinguishes it from polling.
        assert not wait_returned.wait(timeout=0.02)
        ready.set()
        event.set()
        assert wait_returned.wait(timeout=1.0)
        assert result == [True]
    finally:
        event.set()
        thread.join(timeout=1.0)


def test_bounded_audit_timeout_does_not_publish_progress() -> None:
    event = threading.Event()
    generation = pd_engine_progress_generation()

    assert not wait_for_pd_engine_progress(
        event,
        lambda: False,
        timeout_s=0.0,
        observed_generation=generation,
    )
    assert pd_engine_progress_generation() == generation


def test_future_lifecycle_is_monotonic_and_scalar_only() -> None:
    future: Future[str] = Future()
    lifecycle = attach_pd_future_lifecycle(
        future,
        method="execute_model",
        execution_key="prefill:1",
    )

    created = pd_future_lifecycle_snapshot(future)
    assert created["pd_future_lifecycle_stage"] == "created"
    assert created["pd_future_lifecycle_stage_seq"] == 0.0
    assert created["pd_future_lifecycle_terminal"] == 0.0

    assert lifecycle.advance("worker_queued")
    queued = pd_future_lifecycle_snapshot(future)
    assert queued["pd_future_lifecycle_stage"] == "worker_queued"
    assert queued["pd_future_lifecycle_stage_seq"] == 1.0
    assert queued["pd_future_lifecycle_method"] == "execute_model"
    assert queued["pd_future_lifecycle_execution_key"] == "prefill:1"

    future.set_result("done")
    terminal = pd_future_lifecycle_snapshot(future)
    assert terminal["pd_future_lifecycle_stage"] == "terminal_success"
    assert terminal["pd_future_lifecycle_terminal"] == 1.0
    assert not lifecycle.advance("late_invalid_stage")
    assert (
        pd_future_lifecycle_snapshot(future)["pd_future_lifecycle_stage"]
        == "terminal_success"
    )

    # The public snapshot cannot retain request, scheduler, tensor, CUDA, or
    # Future objects; it is safe to copy into a JSONL record.
    assert all(isinstance(value, str | float) for value in terminal.values())


def test_untracked_future_has_safe_mixed_version_snapshot() -> None:
    future: Future[None] = Future()
    snapshot = pd_future_lifecycle_snapshot(future)
    assert snapshot["pd_future_lifecycle_stage"] == "untracked"
    assert snapshot["pd_future_lifecycle_terminal"] == 0.0


def test_recoverable_ready_follows_phase_policy_not_stale_selection() -> None:
    core = EngineCore.__new__(EngineCore)
    core.pd_dual_continuous_batching_enabled = False

    pending: Future[None] = Future()
    ready: Future[None] = Future()
    ready.set_result(None)
    decode_output = SimpleNamespace(
        pd_phase_submit_seq=1,
        num_scheduled_tokens={"decode-request": 1},
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
    )
    independent_prefill = SimpleNamespace(
        pd_phase_submit_seq=2,
        num_scheduled_tokens={"prefill-request": 16},
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
    )
    phase_queues = {
        "decode": deque([(pending, decode_output, pending)]),
        "prefill": deque([(ready, independent_prefill, ready)]),
        "mixed": deque(),
        "idle": deque(),
    }

    assert core._pd_recoverable_future_ready(phase_queues)

    # A ready result sharing the older request is not recoverable and cannot
    # turn the liveness check into an ownership-order bypass.
    independent_prefill.num_scheduled_tokens = {"decode-request": 16}
    assert not core._pd_recoverable_future_ready(phase_queues)
