# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.core.tiered_errors import TieredBackendEligibilityError
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.pd_capacity_retry import PDPhaseCapacityRetryGate


def test_capacity_retry_waits_for_recorded_minimum() -> None:
    gate = PDPhaseCapacityRetryGate()

    assert gate.block(
        "prefill",
        minimum_free_blocks=786,
        observed_free_blocks=204,
        reason="insufficient_hbm_blocks",
        scheduler_iteration_id=87,
    )
    assert not gate.can_retry("prefill", 204)
    assert not gate.can_retry("prefill", 785)
    assert gate.can_retry("prefill", 786)
    assert gate.can_retry("prefill", 100)

    snapshot = gate.snapshot(786)
    assert snapshot["pd_capacity_retry_gate_active"] == 0
    assert snapshot["pd_capacity_retry_gate_block_count"] == 1
    assert snapshot["pd_capacity_retry_gate_skip_count"] == 2
    assert snapshot["pd_capacity_retry_gate_wake_count"] == 1


def test_capacity_retry_clear_exposes_new_request_ordering() -> None:
    gate = PDPhaseCapacityRetryGate()
    gate.block(
        "prefill",
        minimum_free_blocks=786,
        observed_free_blocks=204,
        reason="insufficient_hbm_blocks",
        scheduler_iteration_id=87,
    )

    gate.clear()

    assert gate.can_retry("prefill", 204)
    assert gate.snapshot(204)["pd_capacity_retry_gate_clear_count"] == 1


def test_capacity_retry_fails_open_when_capacity_is_unavailable() -> None:
    gate = PDPhaseCapacityRetryGate()
    gate.block(
        "prefill",
        minimum_free_blocks=786,
        observed_free_blocks=204,
        reason="insufficient_hbm_blocks",
        scheduler_iteration_id=87,
    )

    assert gate.can_retry("prefill", -1)
    assert gate.is_blocked("prefill", -1) is False


def test_capacity_retry_rejects_stale_non_block() -> None:
    gate = PDPhaseCapacityRetryGate()

    assert not gate.block(
        "prefill",
        minimum_free_blocks=204,
        observed_free_blocks=204,
        reason="insufficient_hbm_blocks",
        scheduler_iteration_id=87,
    )
    assert gate.snapshot(204)["pd_capacity_retry_gate_active"] == 0


def test_no_progress_retry_ignores_nonterminal_decode_progress() -> None:
    gate = PDPhaseCapacityRetryGate()

    assert gate.block(
        "prefill",
        minimum_free_blocks=205,
        observed_free_blocks=204,
        reason="prefill_no_progress:policy",
        scheduler_iteration_id=88,
        wake_on_output_progress=True,
    )
    assert not gate.can_retry("prefill", 204)

    assert gate.notify_output_progress(completed_phase="decode") == 0
    assert not gate.can_retry("prefill", 204)
    snapshot = gate.snapshot(204)
    assert snapshot["pd_capacity_retry_gate_active"] == 1
    assert snapshot["pd_capacity_retry_gate_output_progress_wake_count"] == 0


def test_no_progress_retry_wakes_on_prefill_recovery() -> None:
    gate = PDPhaseCapacityRetryGate()

    assert gate.block(
        "prefill",
        minimum_free_blocks=205,
        observed_free_blocks=204,
        reason="prefill_no_progress:policy",
        scheduler_iteration_id=88,
        wake_on_output_progress=True,
    )

    assert gate.notify_output_progress(completed_phase="prefill") == 1
    assert gate.can_retry("prefill", 204)
    snapshot = gate.snapshot(204)
    assert snapshot["pd_capacity_retry_gate_active"] == 0
    assert snapshot["pd_capacity_retry_gate_wake_count"] == 1
    assert snapshot["pd_capacity_retry_gate_output_progress_wake_count"] == 1


def test_no_progress_retry_wakes_when_decode_finishes_request() -> None:
    gate = PDPhaseCapacityRetryGate()
    gate.block(
        "prefill",
        minimum_free_blocks=205,
        observed_free_blocks=204,
        reason="prefill_no_progress:policy",
        scheduler_iteration_id=89,
        wake_on_output_progress=True,
    )

    assert gate.notify_output_progress(
        completed_phase="decode",
        request_finished=True,
    ) == 1
    snapshot = gate.snapshot(204)
    assert snapshot["pd_capacity_retry_gate_active"] == 0
    assert snapshot["pd_capacity_retry_gate_last_state_change_reason"] == (
        "request_finished"
    )


def test_policy_retry_wakes_after_sm_profile_commit() -> None:
    gate = PDPhaseCapacityRetryGate()
    gate.block(
        "prefill",
        minimum_free_blocks=205,
        observed_free_blocks=204,
        reason="prefill_no_progress:profile",
        scheduler_iteration_id=90,
        wake_on_output_progress=True,
    )

    assert gate.notify_state_change("sm_profile_committed") == 1
    snapshot = gate.snapshot(204)
    assert snapshot["pd_capacity_retry_gate_state_change_generation"] == 1
    assert snapshot["pd_capacity_retry_gate_last_state_change_reason"] == (
        "sm_profile_committed"
    )


def test_exact_capacity_retry_ignores_output_recovery() -> None:
    gate = PDPhaseCapacityRetryGate()

    assert gate.block(
        "prefill",
        minimum_free_blocks=786,
        observed_free_blocks=204,
        reason="insufficient_hbm_blocks",
        scheduler_iteration_id=87,
    )

    assert gate.notify_output_progress(
        completed_phase="decode",
        request_finished=True,
        capacity_increased=True,
    ) == 0
    assert not gate.can_retry("prefill", 204)
    snapshot = gate.snapshot(204)
    assert snapshot["pd_capacity_retry_gate_active"] == 1
    assert snapshot["pd_capacity_retry_gate_output_progress_wake_count"] == 0


def test_tiered_failure_publishes_progress_after_kv_release() -> None:
    """A Future error must wake capacity retry after scheduler cleanup."""

    free_blocks = {"value": 3}
    progress_notifications: list[int] = []

    class _Scheduler:
        @staticmethod
        def fail_scheduled_requests(scheduler_output, message):
            assert "TieredBackendEligibilityError" in message
            free_blocks["value"] = 9
            return {
                0: SimpleNamespace(
                    finished_requests={"request-1"}, outputs=[]
                )
            }

    engine = object.__new__(EngineCore)
    engine.scheduler = _Scheduler()
    engine._pd_capacity_retry_gate = PDPhaseCapacityRetryGate()
    engine._pd_capacity_retry_gate.block(
        "prefill",
        minimum_free_blocks=4,
        observed_free_blocks=3,
        reason="prefill_no_progress:kv_capacity",
        scheduler_iteration_id=10,
        wake_on_output_progress=True,
    )
    engine._pd_capacity_retry_free_blocks = lambda: free_blocks["value"]
    engine._notify_pd_engine_progress = lambda: progress_notifications.append(
        free_blocks["value"]
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-1": 16},
        scheduled_phase="prefill",
        iteration_debug=None,
    )

    outputs = engine._fail_tiered_backend_scheduler_output(
        scheduler_output,
        TieredBackendEligibilityError("short provider tail"),
    )

    assert outputs[0].finished_requests == {"request-1"}
    assert progress_notifications == [9]
    assert scheduler_output.iteration_debug == {
        "tiered_failure_free_blocks_before": 3,
        "tiered_failure_free_blocks_after": 9,
        "tiered_failure_capacity_retry_wake_count": 1,
    }
    assert engine._pd_capacity_retry_gate.can_retry("prefill", 9)
