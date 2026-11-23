#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for vendor_v8 feedback-driven TBT SM allocation."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v8"


def _package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_package("vllm", VENDOR / "vllm")
_package("vllm.v1", VENDOR / "vllm/v1")
_package("vllm.v1.core", VENDOR / "vllm/v1/core")
_package("vllm.v1.core.sched", VENDOR / "vllm/v1/core/sched")
_package("vllm.v1.engine", VENDOR / "vllm/v1/engine")
_load("vllm.pd_sm_profiles", VENDOR / "vllm/pd_sm_profiles.py")
_load(
    "vllm.v1.core.sched.decode_scheduler",
    VENDOR / "vllm/v1/core/sched/decode_scheduler.py",
)
_load(
    "vllm.v1.engine.pd_tbt_profile",
    VENDOR / "vllm/v1/engine/pd_tbt_profile.py",
)
_load(
    "vllm.v1.engine.pd_sm_allocator",
    VENDOR / "vllm/v1/engine/pd_sm_allocator.py",
)
_load(
    "vllm.v1.engine.pd_sm_transition",
    VENDOR / "vllm/v1/engine/pd_sm_transition.py",
)

from vllm.pd_sm_profiles import dense_pd_sm_profiles
from vllm.v1.core.sched.decode_scheduler import (
    TBTDecodeScheduler,
    per_token_tbt_samples,
)
from vllm.v1.engine.pd_sm_allocator import (
    PDComputeAllocationController,
    PDSMAllocatorConfig,
)
from vllm.v1.engine.pd_sm_transition import PDSMDrainTransition


@dataclass
class _Request:
    request_id: str = "request-0"
    arrival_time: float = 0.0
    prefill_completed_at: float | None = 0.0
    prefill_completed_iteration_id: int = 0
    tbt_slo_ms: float | None = 125.0
    tbt_last_token_at: float | None = None
    tbt_last_observed_ms: float | None = None
    tbt_recent_gaps_ms: list[float] = field(default_factory=list)
    tbt_service_count: int = 0


def _config(**overrides) -> PDSMAllocatorConfig:
    profiles = dense_pd_sm_profiles(40, 2)
    values = {
        "enabled": True,
        "shadow": False,
        "strict": True,
        "total_sms": 40,
        "alignment": 2,
        "profiles": profiles,
        "initial_profile": next(
            profile for profile in profiles if profile.profile_id == "d20_p20"
        ),
        "ewma_alpha": 0.25,
        "decode_target_ms": 0.0,
        "min_improvement": 0.0,
        "cooldown_decisions": 0,
        "max_step_sms": 8,
        "switch_penalty_ms": 0.0,
        "policy": "tbt_slo_feedback",
        "tbt_default_slo_ms": 125.0,
        "tbt_profile_margin_ms": 0.0,
        "feedback_initial_decode_sms": 30,
        "feedback_step_sms": 2,
        "feedback_window_tokens": 4,
        "feedback_slack_ratio": 0.90,
        "feedback_reprobe_ratio": 0.80,
        "feedback_decrease_windows": 1,
        "feedback_hold_windows": 2,
        "feedback_min_prefill_sms": 2,
        "feedback_context_bucket_tokens": 25000,
    }
    values.update(overrides)
    return PDSMAllocatorConfig(**values)


def _concurrent_plan(
    controller: PDComputeAllocationController,
    *,
    p95_ms: float = 0.0,
    sample_count: int = 0,
    epoch: int = 0,
    inflight: bool = False,
    transition_active: bool = False,
):
    return controller.plan(
        decode_ready=True,
        prefill_ready=True,
        decode_inflight=inflight,
        prefill_inflight=False,
        transition_active=transition_active,
        decode_context_tokens=100000,
        decode_request_count=1,
        prefill_mode="native",
        decode_tbt_slo_ms=125.0,
        decode_tbt_observed_p95_ms=p95_ms,
        decode_tbt_observed_slo_ms=125.0,
        decode_tbt_observed_ratio=p95_ms / 125.0,
        decode_tbt_observation_count=sample_count,
        decode_tbt_observation_epoch=epoch,
    )


def test_feedback_policy_defaults_to_dense_two_sm_profiles() -> None:
    values = {
        "VLLM_PD_DYNAMIC_SM_ENABLED": "1",
        "VLLM_PD_DYNAMIC_SM_POLICY": "tbt_slo_feedback",
        "VLLM_PD_DYNAMIC_SM_TOTAL_SMS": "40",
        "VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS": "20",
        "VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS": "20",
        "VLLM_PD_TBT_DEFAULT_SLO_MS": "125",
    }
    names = {*values, "VLLM_PD_DYNAMIC_SM_PROFILES"}
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(values)
        os.environ.pop("VLLM_PD_DYNAMIC_SM_PROFILES", None)
        config = PDSMAllocatorConfig.from_env()
        assert config.policy == "tbt_slo_feedback"
        assert [profile.decode_sms for profile in config.profiles] == list(
            range(0, 41, 2)
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_decode_observer_reports_and_resets_recent_p95() -> None:
    scheduler = TBTDecodeScheduler()
    request = _Request()
    for served_at in (1.0, 1.1, 1.2, 1.3, 1.4):
        scheduler.mark_tokens_served(
            request, served_at=served_at, num_tokens=1
        )
    snapshot = scheduler.feedback_snapshot(
        [request], default_tbt_slo_ms=0.0, window_tokens=4
    )
    assert 99.9 < snapshot["decode_tbt_observed_p95_ms"] < 100.1
    assert snapshot["decode_tbt_observation_count"] == 4
    assert snapshot["decode_tbt_observation_epoch"] == 5
    scheduler.reset_feedback_window([request])
    assert request.tbt_recent_gaps_ms == []
    assert request.tbt_last_observed_ms is None


def test_chunk_interarrival_is_normalized_per_emitted_token() -> None:
    assert per_token_tbt_samples(
        previous_at=None, served_at=1.0, num_tokens=4
    ) == []
    samples = per_token_tbt_samples(
        previous_at=1.0, served_at=1.1, num_tokens=4
    )
    assert len(samples) == 4
    assert all(24.9 < sample < 25.1 for sample in samples)

    scheduler = TBTDecodeScheduler()
    request = _Request()
    scheduler.mark_tokens_served(request, served_at=1.0, num_tokens=3)
    scheduler.mark_tokens_served(request, served_at=1.1, num_tokens=4)
    scheduler.mark_tokens_served(request, served_at=1.14, num_tokens=2)
    assert len(request.tbt_recent_gaps_ms) == 6
    assert request.tbt_recent_gaps_ms[:4] == samples
    assert all(
        19.9 < sample < 20.1
        for sample in request.tbt_recent_gaps_ms[4:]
    )


def test_feedback_steps_down_then_recovers_and_holds_boundary() -> None:
    controller = PDComputeAllocationController(_config())

    initial = _concurrent_plan(controller)
    assert initial.profile.profile_id == "d30_p10"
    assert initial.reason == "tbt_feedback_initial"

    slack = _concurrent_plan(
        controller, p95_ms=100.0, sample_count=4, epoch=4
    )
    assert slack.profile.profile_id == "d28_p12"
    assert slack.reason == "tbt_feedback_slack_step_down"

    violation = _concurrent_plan(
        controller, p95_ms=130.0, sample_count=4, epoch=8
    )
    assert violation.profile.profile_id == "d30_p10"
    assert violation.reason == "tbt_feedback_violation_step_up"
    assert violation.tbt_slo_emergency
    assert violation.tbt_feedback is not None
    assert violation.tbt_feedback.maximum_infeasible_decode_sms == 28
    assert violation.tbt_feedback.minimum_feasible_decode_sms == 30

    held = _concurrent_plan(
        controller, p95_ms=100.0, sample_count=4, epoch=12
    )
    assert held.profile.profile_id == "d30_p10"
    assert held.reason == "tbt_feedback_boundary_hold"


def test_feedback_pending_target_survives_inflight_hold() -> None:
    controller = PDComputeAllocationController(_config())
    deferred = _concurrent_plan(controller, inflight=True)
    assert deferred.profile.profile_id == "d20_p20"
    assert deferred.target_profile.profile_id == "d30_p10"
    assert deferred.reason == "inflight_profile_hold"

    committed = controller.commit_profile_at_quiescence(
        deferred.target_profile.profile_id,
        expected_epoch=deferred.epoch,
    )
    assert committed.profile_id == "d30_p10"
    assert controller.epoch == deferred.epoch + 1

    applied = _concurrent_plan(controller)
    assert applied.profile.profile_id == "d30_p10"
    assert not applied.switched


def test_active_transition_prevents_direct_epoch_advance() -> None:
    controller = PDComputeAllocationController(_config())
    source_epoch = controller.epoch
    source_profile = controller.current_profile

    deferred = _concurrent_plan(controller, transition_active=True)

    assert deferred.reason == "inflight_profile_hold"
    assert deferred.deferred
    assert deferred.profile == source_profile
    assert deferred.epoch == source_epoch
    assert controller.current_profile == source_profile
    assert controller.epoch == source_epoch

    committed = controller.commit_profile_at_quiescence(
        deferred.target_profile.profile_id,
        expected_epoch=source_epoch,
    )
    assert committed.profile_id == "d30_p10"
    assert controller.epoch == source_epoch + 1


def test_allocator_and_global_queue_drain_commit_one_epoch() -> None:
    controller = PDComputeAllocationController(_config())
    requested = _concurrent_plan(controller, inflight=True)
    transition = PDSMDrainTransition()
    transition.request(
        source_profile_id=requested.profile.profile_id,
        source_epoch=requested.epoch,
        source_decode_sms=requested.profile.decode_sms,
        source_prefill_sms=requested.profile.prefill_sms,
        target_profile_id=requested.target_profile.profile_id,
        target_decode_sms=requested.target_profile.decode_sms,
        target_prefill_sms=requested.target_profile.prefill_sms,
        reason=requested.reason,
        urgent=False,
        now=1.0,
    )

    while_draining = _concurrent_plan(
        controller,
        inflight=False,
        transition_active=transition.active,
    )
    assert while_draining.profile == requested.profile
    assert while_draining.epoch == requested.epoch
    assert controller.epoch == requested.epoch
    assert not transition.advance(
        old_decode_futures=0,
        old_prefill_futures=1,
        global_queue_depth=1,
        now=1.1,
    )
    assert transition.advance(
        old_decode_futures=0,
        old_prefill_futures=0,
        global_queue_depth=0,
        now=1.2,
    )

    controller.commit_profile_at_quiescence(
        transition.target_profile_id,
        expected_epoch=transition.source_epoch,
    )
    transition.finish(now=1.2)
    assert controller.current_profile.profile_id == "d30_p10"
    assert controller.epoch == requested.epoch + 1


def test_feedback_reuses_feasible_boundary_after_solo_phase() -> None:
    controller = PDComputeAllocationController(_config())
    _concurrent_plan(controller)
    _concurrent_plan(controller, p95_ms=100.0, sample_count=4, epoch=4)
    _concurrent_plan(controller, p95_ms=130.0, sample_count=4, epoch=8)

    solo = controller.plan(
        decode_ready=False,
        prefill_ready=True,
        decode_inflight=False,
        prefill_inflight=False,
    )
    assert solo.profile.profile_id == "d0_p40"

    reentered = _concurrent_plan(controller, epoch=8)
    assert reentered.profile.profile_id == "d30_p10"
    assert reentered.reason == "tbt_feedback_cached_feasible"


def test_capacity_serialization_scopes_the_fixed_split_ablation() -> None:
    fixed = PDComputeAllocationController(
        _config(serialized_full_sm=False)
    )
    capacity_plan = fixed.plan(
        decode_ready=True,
        prefill_ready=False,
        decode_inflight=False,
        prefill_inflight=False,
        decode_context_tokens=100000,
        decode_request_count=1,
        prefill_mode="native",
        decode_tbt_slo_ms=125.0,
        capacity_serialized=True,
    )
    assert capacity_plan.profile.profile_id == "d30_p10"
    assert capacity_plan.reason == (
        "capacity_serialized_fixed_split:tbt_feedback_initial"
    )
    assert capacity_plan.capacity_serialized_fixed_split

    natural_solo = fixed.plan(
        decode_ready=True,
        prefill_ready=False,
        decode_inflight=False,
        prefill_inflight=False,
    )
    assert natural_solo.profile.profile_id == "d40_p0"
    assert natural_solo.reason == "decode_only_full_sm"

    full = PDComputeAllocationController(
        _config(serialized_full_sm=True)
    )
    full_plan = full.plan(
        decode_ready=True,
        prefill_ready=False,
        decode_inflight=False,
        prefill_inflight=False,
        capacity_serialized=True,
    )
    assert full_plan.profile.profile_id == "d40_p0"
    assert full_plan.reason == "capacity_serialized_decode_full_sm"
    assert full_plan.serialized_full_sm_targeted


if __name__ == "__main__":
    test_feedback_policy_defaults_to_dense_two_sm_profiles()
    test_decode_observer_reports_and_resets_recent_p95()
    test_chunk_interarrival_is_normalized_per_emitted_token()
    test_feedback_steps_down_then_recovers_and_holds_boundary()
    test_feedback_pending_target_survives_inflight_hold()
    test_active_transition_prevents_direct_epoch_advance()
    test_allocator_and_global_queue_drain_commit_one_epoch()
    test_feedback_reuses_feasible_boundary_after_solo_phase()
    test_capacity_serialization_scopes_the_fixed_split_ablation()
    print("[OK] vendor_v8 feedback TBT SM allocator tests passed")
