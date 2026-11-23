#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Import-free source contract for vendor_v8 policy integration."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v8" / "vllm"


def _read(relative_path: str) -> str:
    return (VENDOR / relative_path).read_text(encoding="utf-8")


def _require(source: str, needle: str, description: str) -> None:
    if needle not in source:
        raise AssertionError(f"missing {description}: {needle}")


def _forbid(source: str, needle: str, description: str) -> None:
    if needle in source:
        raise AssertionError(f"unexpected {description}: {needle}")


def _load_standalone_function(source: str, function_name: str):
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, "<vendor_v8_contract>", "exec"), namespace)
    return namespace[function_name]


def main() -> int:
    config = _read("config/scheduler.py")
    args = _read("engine/arg_utils.py")
    request = _read("v1/request.py")
    scheduler = _read("v1/core/sched/scheduler.py")
    sched_output = _read("v1/core/sched/output.py")
    prefill = _read("v1/core/sched/prefill_chunk_scheduler.py")
    decode = _read("v1/core/sched/decode_scheduler.py")
    residency = _read("v1/core/sched/residency_policy.py")
    core = _read("v1/engine/core.py")
    model_runner = _read("v1/worker/gpu_model_runner.py")
    allocator = _read("v1/engine/pd_sm_allocator.py")
    transition = _read("v1/engine/pd_sm_transition.py")
    tbt_profile = _read("v1/engine/pd_tbt_profile.py")

    _require(
        config,
        "enable_ttft_prefill_chunk_scheduling: bool = False",
        "default-off prefill selector",
    )
    _require(
        config,
        "ttft_prefill_native_chunk_max_tokens: int = Field(default=0",
        "default-off Native Prefill chunk expansion",
    )
    _require(prefill, 'SLO_SRPT = "slo_srpt"', "SLO-guarded SRPT policy")
    _require(
        prefill,
        "ttft_prefill_starvation_bypass_count",
        "prefill starvation metric",
    )
    _require(
        prefill,
        "PrefillChunkServiceProfile",
        "profiled prefill chunk service estimator",
    )
    _require(
        prefill,
        "ttft_prefill_overload_edf_active",
        "explicit EDF overload fallback metric",
    )
    _require(
        prefill,
        "ttft_prefill_starved_by_projected_age",
        "profile-lookahead starvation guard",
    )
    _require(
        scheduler,
        "ttft_prefill_estimated_ms_per_token",
        "scheduler policy configuration",
    )
    _require(
        config,
        "ttft_prefill_service_profile_path",
        "prefill service profile configuration",
    )
    _require(
        args,
        '"--ttft-prefill-service-profile-path"',
        "prefill service profile CLI flag",
    )
    _require(
        args,
        '"--ttft-prefill-starvation-max-bypasses"',
        "starvation CLI flag",
    )
    _require(
        args,
        '"--ttft-prefill-native-chunk-max-tokens"',
        "Native Prefill execution-cap CLI flag",
    )
    _require(
        scheduler,
        "ttft_prefill_native_chunk_expansion_applied",
        "Native Prefill expansion observability",
    )

    _require(request, "self.tbt_slo_ms: float | None", "request TBT SLO")
    _require(request, '"tbt_slo_ms"', "request TBT SLO parser")
    _require(decode, "mark_tokens_served", "decode token-service state")
    _require(
        decode,
        "per_token_tbt_samples",
        "exact multi-token chunk normalization",
    )
    _require(
        scheduler,
        "tbt_slo_observability_enabled",
        "scheduler TBT observability gate",
    )
    _require(
        scheduler,
        "get_tbt_allocation_state",
        "active decode state independent of phase-local output",
    )
    _require(
        core,
        "_pd_sm_output_tbt_state",
        "EngineCore TBT state extraction",
    )
    _require(
        allocator,
        '_TBT_PROFILED_POLICY = "tbt_slo_profiled"',
        "profiled TBT allocator policy",
    )
    _require(
        allocator,
        '_TBT_FEEDBACK_POLICY = "tbt_slo_feedback"',
        "feedback TBT allocator policy",
    )
    _require(
        allocator,
        "tbt_feedback_violation_step_up",
        "feedback fast-up transition",
    )
    _require(
        scheduler,
        "reset_tbt_feedback_window",
        "clean feedback window after SM handoff",
    )
    _require(
        allocator,
        "tbt_slo_overload",
        "explicit TBT overload metric",
    )

    # vendor_v8 must retain vendor_v7's independent ownership boundaries.
    _require(decode, "TBTDecodeSchedulingPolicy.FCFS", "decode FCFS")
    _require(residency, "MIN_RING_CHUNKS", "min-ring-chunks residency")
    _require(residency, "JIT_MIN_COPY", "JIT-min-copy residency")
    _require(allocator, "inflight_profile_hold", "quiescent profile handoff")
    _require(
        allocator,
        "commit_profile_at_quiescence",
        "epoch-validated deferred profile commit",
    )
    _require(
        transition,
        '_DRAIN_PREFILL = "drain_prefill"',
        "Prefill-first bounded drain",
    )
    _require(
        transition,
        '_DRAIN_ALL = "drain_all"',
        "all-phase quiescence fence",
    )
    _require(
        core,
        "_advance_pd_sm_transition",
        "EngineCore transition advancement",
    )
    _require(
        allocator,
        "transition_active: bool = False",
        "allocator epoch freeze during an active transition",
    )
    _require(
        transition,
        "global_queue_depth: int",
        "complete phase-queue commit barrier",
    )
    _require(
        transition,
        "source_decode_sms",
        "source-profile Decode admission guard",
    )
    _require(
        core,
        "pd_sm_transition_resume_submit_delay_ms",
        "commit-to-next-future resume metric",
    )
    _forbid(
        core,
        "dynamic SM transition reached commit with queued futures",
        "fatal nonempty transition queue assertion",
    )
    _require(
        tbt_profile,
        '"unit": "ms_per_output_token"',
        "per-token overlap-profile unit",
    )
    _require(
        tbt_profile,
        "p95_per_token_tbt_ms",
        "unambiguous per-token overlap-profile field",
    )
    _require(core, "pd_sm_profile_epoch", "immutable profile epoch")
    _require(
        scheduler,
        "pd_ttft_handoff_request_ids",
        "final-Prefill TTFT handoff marker",
    )
    _require(
        sched_output,
        "pd_ttft_handoff_request_ids",
        "SchedulerOutput TTFT handoff field",
    )
    _require(
        core,
        "_pd_ttft_handoff_mark_submitted",
        "TTFT handoff fence arm",
    )
    _require(
        core,
        "_pd_ttft_handoff_clear_from_outputs",
        "first-token handoff fence release",
    )
    _require(
        core,
        "pd_ttft_handoff_blocked_prefill_checks",
        "TTFT handoff Prefill admission metric",
    )
    _require(
        scheduler,
        "prev_step_scheduled_phase_by_req",
        "phase-local previous ownership tracking",
    )
    _require(
        scheduler,
        "phase_local_ownership_changed",
        "phase-local token-history refresh condition",
    )
    ownership_changed = _load_standalone_function(
        scheduler, "_pd_phase_local_ownership_changed"
    )
    assert ownership_changed(True, True, "prefill", "decode")
    assert ownership_changed(True, True, "decode", "prefill")
    assert not ownership_changed(True, True, "decode", "decode")
    assert not ownership_changed(True, False, "prefill", "decode")
    assert not ownership_changed(False, True, "prefill", "decode")
    _require(
        scheduler,
        "pd_phase_handoff_token_refresh_request_count",
        "phase-local token-history refresh metric",
    )
    _require(
        scheduler,
        "class ReleasedPrefillVictim:",
        "generation-consistent residency victim result",
    )
    _require(
        scheduler,
        "selected_candidate = released_victim.candidate_snapshot",
        "reclaim-selected snapshot metric bookkeeping",
    )
    _require(
        model_runner,
        "OutputPlaceholderOwnershipError",
        "explicit async placeholder ownership failure",
    )
    _require(
        scheduler,
        "pd_output_placeholder_ledger.resolve(",
        "single-owner async placeholder commit",
    )
    _require(
        scheduler,
        "Cannot preempt a request while a model Future",
        "active output-owner preemption guard",
    )
    _require(
        scheduler,
        "def _pd_preemption_protection_reason",
        "in-flight execution preemption filter",
    )
    _require(
        scheduler,
        'return "execution_future_pending"',
        "non-output Prefill Future preemption protection",
    )
    assert "phase_local_output_token_repair_invalid" not in model_runner, (
        "the removed token repair path must not become a second output producer"
    )
    _require(
        model_runner,
        "phase_handoff_sync_sample_output",
        "final-Prefill-only sample synchronization",
    )
    _require(
        model_runner,
        "out-of-vocabulary token IDs",
        "invalid async placeholder fail-fast error",
    )

    print("[OK] vendor_v8 source integration contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
