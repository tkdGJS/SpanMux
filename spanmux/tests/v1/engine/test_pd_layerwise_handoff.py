# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.layers.pd_layerwise_handoff import (
    PDLayerwiseHandoffController,
    layerwise_prefill_only_handoff_decision,
    layerwise_profile_handoff_decision,
    layerwise_source_decode_admission_decision,
)
from vllm.v1.engine.pd_engine_progress import pd_engine_progress_generation


def test_prefill_only_handoff_requires_empty_decode_backlog() -> None:
    eligible, reason = layerwise_prefill_only_handoff_decision(
        armed=True,
        prefill_inflight=True,
        decode_inflight=False,
        decode_ready=True,
    )
    assert not eligible
    assert reason == "decode_backlog_ready"

    eligible, reason = layerwise_prefill_only_handoff_decision(
        armed=True,
        prefill_inflight=True,
        decode_inflight=True,
        decode_ready=False,
    )
    assert not eligible
    assert reason == "decode_inflight"

    eligible, reason = layerwise_prefill_only_handoff_decision(
        armed=True,
        prefill_inflight=True,
        decode_inflight=False,
        decode_ready=False,
    )
    assert eligible
    assert reason == "decode_idle"


def test_layerwise_handoff_lifecycle() -> None:
    controller = PDLayerwiseHandoffController()

    generation = controller.arm(
        source_profile_id="d30_p10",
        source_epoch=7,
    )
    assert generation == 1
    assert (
        controller.begin_execution(
            source_profile_id="d30_p10",
            source_epoch=7,
        )
        == generation
    )
    # Arming reserves controller ownership but must not synchronize every
    # layer before EngineCore has published a real handoff request.
    assert controller.should_bound_submission(generation)
    assert not controller.should_gate(generation)
    wake_event = controller.wake_event_for(generation)
    assert wake_event is not None
    assert not wake_event.is_set()

    controller.note_boundary(generation, sync_ms=2.5)
    assert controller.snapshot()["pd_layerwise_handoff_boundary_count"] == 0.0
    assert controller.target_for(generation) is None
    assert controller.request(
        source_profile_id="d30_p10",
        source_epoch=7,
        target_profile_id="d0_p40",
        target_epoch=8,
    )
    assert wake_event.is_set()
    assert controller.should_gate(generation)
    controller.note_boundary(generation, sync_ms=2.5)
    target = controller.target_for(generation)
    assert target is not None
    assert target.profile_id == "d0_p40"
    assert target.profile_epoch == 8
    resume_event = controller.transition_resume_event_for(generation)
    assert resume_event is not None
    assert not resume_event.is_set()
    assert controller.source_decode_admission(source_decode_sms=30) == (
        True,
        "source_decode_until_boundary",
    )
    progress_generation = pd_engine_progress_generation()
    assert controller.mark_boundary_recorded(generation, layer_index=9)
    assert controller.mark_boundary_completed(
        generation,
        layer_index=9,
        gpu_wait_ms=4.0,
    )
    assert pd_engine_progress_generation() == progress_generation + 1
    assert controller.source_decode_admission(source_decode_sms=30) == (
        False,
        "boundary_completed_drain_source_decode",
    )
    # Submission stays bounded so the same long Prefill future can move again.
    assert controller.should_bound_submission(generation)
    assert not controller.should_gate(generation)

    assert controller.completed_transition_matches(
        source_profile_id="d30_p10",
        source_epoch=7,
        target_profile_id="d0_p40",
        target_epoch=8,
    )
    assert controller.acknowledge_completed_transition(
        source_profile_id="d30_p10",
        source_epoch=7,
        target_profile_id="d0_p40",
        target_epoch=8,
    )
    assert resume_event.is_set()
    assert controller.committed_transition_matches(
        source_profile_id="d30_p10",
        source_epoch=7,
        target_profile_id="d0_p40",
        target_epoch=8,
    )
    snapshot = controller.snapshot()
    assert snapshot["pd_layerwise_handoff_source_profile"] == "d0_p40"
    assert snapshot["pd_layerwise_handoff_source_epoch"] == 8.0
    assert snapshot["pd_layerwise_handoff_committed_count"] == 1.0

    # A later Decode arrival moves the same Prefill continuation back to a
    # concurrent profile instead of waiting for the whole future to finish.
    assert controller.request(
        source_profile_id="d0_p40",
        source_epoch=8,
        target_profile_id="d30_p10",
        target_epoch=9,
    )
    assert controller.mark_boundary_recorded(generation, layer_index=17)
    assert controller.mark_boundary_completed(
        generation,
        layer_index=17,
        gpu_wait_ms=3.0,
    )
    assert controller.acknowledge_completed_transition(
        source_profile_id="d0_p40",
        source_epoch=8,
        target_profile_id="d30_p10",
        target_epoch=9,
    )

    controller.complete_execution(generation)
    snapshot = controller.snapshot()
    assert snapshot["pd_layerwise_handoff_armed"] == 0.0
    assert snapshot["pd_layerwise_handoff_apply_count"] == 2.0
    assert snapshot["pd_layerwise_handoff_commit_count"] == 2.0
    assert snapshot["pd_layerwise_handoff_boundary_count_total"] == 1.0
    assert snapshot["pd_layerwise_handoff_boundary_sync_ms_total_all"] == 2.5
    assert snapshot["pd_layerwise_handoff_boundary_record_count_total"] == 2.0
    assert snapshot["pd_layerwise_handoff_boundary_complete_count_total"] == 2.0
    assert snapshot["pd_layerwise_handoff_boundary_gpu_wait_ms_total"] == 7.0


def test_general_handoff_allows_ready_decode_when_target_has_both_phases() -> None:
    eligible, reason = layerwise_profile_handoff_decision(
        armed=True,
        prefill_inflight=True,
        decode_inflight=False,
        source_prefill_sms=40,
        target_prefill_sms=10,
    )
    assert eligible
    assert reason == "prefill_layer_boundary"

    eligible, reason = layerwise_profile_handoff_decision(
        armed=True,
        prefill_inflight=True,
        decode_inflight=True,
        source_prefill_sms=10,
        target_prefill_sms=20,
    )
    assert not eligible
    assert reason == "decode_inflight"

    eligible, reason = layerwise_profile_handoff_decision(
        armed=True,
        prefill_inflight=True,
        decode_inflight=False,
        source_prefill_sms=10,
        target_prefill_sms=0,
    )
    assert not eligible
    assert reason == "target_has_no_prefill_sms"


def test_source_decode_admission_closes_only_after_boundary_completion() -> None:
    allowed, reason = layerwise_source_decode_admission_decision(
        transition_requested=True,
        boundary_completed=False,
        source_decode_sms=30,
    )
    assert allowed
    assert reason == "source_decode_until_boundary"

    allowed, reason = layerwise_source_decode_admission_decision(
        transition_requested=True,
        boundary_completed=True,
        source_decode_sms=30,
    )
    assert not allowed
    assert reason == "boundary_completed_drain_source_decode"

    allowed, reason = layerwise_source_decode_admission_decision(
        transition_requested=True,
        boundary_completed=False,
        source_decode_sms=0,
    )
    assert not allowed
    assert reason == "source_has_no_decode_sms"


def test_request_is_rejected_before_worker_begins() -> None:
    controller = PDLayerwiseHandoffController()
    controller.arm(source_profile_id="d30_p10", source_epoch=0)

    assert not controller.request(
        source_profile_id="d30_p10",
        source_epoch=0,
        target_profile_id="d0_p40",
        target_epoch=1,
    )
    assert (
        controller.snapshot()[
            "pd_layerwise_handoff_rejected_request_count"
        ]
        == 1.0
    )

    generation = controller.begin_execution(
        source_profile_id="d30_p10",
        source_epoch=0,
    )
    assert generation == 1
    assert controller.request(
        source_profile_id="d30_p10",
        source_epoch=0,
        target_profile_id="d0_p40",
        target_epoch=1,
    )


def test_completed_unrequested_execution_does_not_leave_stale_arm() -> None:
    controller = PDLayerwiseHandoffController()
    generation = controller.arm(
        source_profile_id="d30_p10",
        source_epoch=2,
    )
    controller.begin_execution(
        source_profile_id="d30_p10",
        source_epoch=2,
    )
    controller.complete_execution(generation)

    snapshot = controller.snapshot()
    assert snapshot["pd_layerwise_handoff_armed"] == 0.0
    assert (
        snapshot["pd_layerwise_handoff_completed_without_request_count"]
        == 1.0
    )
    assert snapshot["pd_layerwise_handoff_boundary_count_total"] == 0.0
    assert (
        controller.arm(
            source_profile_id="d20_p20",
            source_epoch=3,
        )
        == 2
    )
