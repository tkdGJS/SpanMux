# SPDX-License-Identifier: Apache-2.0
"""Small environment gates for scheduler observability.

Controller state and detailed traces intentionally have separate gates.  A
TBT policy may require token timestamps without requiring a per-request debug
snapshot on every scheduler iteration.
"""

import os
from collections.abc import Mapping

_TRUE = {"1", "true", "yes", "on"}


def _enabled(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in _TRUE


def detailed_iteration_debug_enabled(
    *,
    virtual_context_timing: bool = False,
    env: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if env is None else env
    lightweight_timeline = lightweight_timeline_enabled(values)
    strict_bounded_lightweight = (
        strict_bounded_lightweight_timeline_enabled(values)
    )
    legacy_path_requested = bool(
        values.get("VLLM_ITERATION_METRICS_JSONL")
        or values.get("VLLM_DECODE_TIMELINE_JSONL")
    )
    return bool(
        _enabled(values, "VLLM_FULL_ITERATION_DEBUG")
        # Preserve the historical path-implies-full-debug contract unless the
        # caller explicitly selects the lightweight timeline schema.
        or (legacy_path_requested and not lightweight_timeline)
        or values.get("VLLM_PD_DECODE_GPU_COMPONENT_TRACE_JSONL")
        # Virtual-context timing historically implied the full allocator/page
        # snapshot.  A caller that explicitly selected the lightweight
        # timeline is asking for the bounded hot-path schema instead; retain
        # the timing opt-in only when lightweight mode is not active.  Full
        # debug above remains the explicit override.
        or (virtual_context_timing and not strict_bounded_lightweight)
    )


def lightweight_timeline_enabled(
    env: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if env is None else env
    return _enabled(values, "VLLM_LIGHTWEIGHT_TIMELINE_TRACE")


def strict_bounded_lightweight_timeline_enabled(
    env: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if env is None else env
    return bool(
        lightweight_timeline_enabled(values)
        and _enabled(values, "VLLM_LIGHTWEIGHT_TIMELINE_STRICT_BOUNDED")
    )


def iteration_debug_sample_every_n(
    env: Mapping[str, str] | None = None,
) -> int:
    values = os.environ if env is None else env
    try:
        return max(
            1,
            int(values.get("VLLM_ITERATION_METRICS_SAMPLE_EVERY_N", "100")),
        )
    except ValueError:
        return 100


_LIGHTWEIGHT_WORKER_TIMING_KEYS = frozenset(
    {
        "gpu_forward_elapsed_ms",
        "worker_execute_start_ts",
        "worker_execute_duration_s",
        "worker_input_prep_duration_s",
        "worker_kv_cache_prepare_duration_s",
        "worker_attention_metadata_build_duration_s",
        "worker_forward_duration_s",
        "worker_model_forward_duration_s",
        "worker_postprocess_duration_s",
        "worker_sample_duration_s",
        "worker_sampling_duration_s",
        "worker_bookkeep_duration_s",
        "worker_eplb_duration_s",
        "worker_output_build_duration_s",
        "worker_output_transfer_duration_s",
        "worker_sample_tokens_duration_s",
        "isolation_gpu_allocated_bytes",
        "isolation_gpu_reserved_bytes",
        "isolation_gpu_peak_allocated_bytes",
        "isolation_gpu_peak_reserved_bytes",
        "isolation_gpu_active_bytes",
        "isolation_gpu_inactive_split_bytes",
        "isolation_prefill_workspace_rightsize",
        "isolation_ring_workspace_sharing",
        "pd_prefill_workspace_capacity_tokens",
        "pd_prefill_workspace_arena_bytes",
        "pd_prefill_workspace_retired_bytes",
        "pd_prefill_workspace_growth_count",
        "pd_prefill_workspace_ring_borrow_count",
        "pd_prefill_workspace_ring_borrow_fallback_count",
        "pd_prefill_workspace_ring_borrow_last_reason",
        "pd_prefill_workspace_ring_borrow_peak_bytes",
        # Plan-only layer checkpoint: bounded scalars, never KV address rows.
        "kv_retention_checkpoint_plan_active",
        "kv_retention_checkpoint_request_count",
        "kv_retention_checkpoint_next_layer",
        "kv_retention_checkpoint_end_layer",
        "kv_retention_checkpoint_boundary_sequence",
        "kv_retention_checkpoint_unique_referenced_blocks",
        "kv_retention_checkpoint_layer_token_coverage",
        "kv_retention_checkpoint_release_authorized",
        "kv_retention_offload_copy_ready",
        "kv_retention_offload_bytes",
        "kv_retention_offload_released_blocks",
        "kv_retention_offload_ring_active",
        "kv_retention_offload_published",
        "kv_retention_offload_d2h_wait_s",
        "kv_retention_offload_budget_declined",
        "kv_retention_offload_ring_layers",
        "kv_retention_resume_frame_active",
        "kv_retention_resume_frame_gpu_bytes",
        "kv_retention_resume_frame_copy_bytes",
        "kv_retention_resume_frame_source_gpu_bytes",
        "kv_retention_resume_frame_copy_wait_s",
        "kv_retention_resume_frame_budget_declined",
        "kv_retention_p1_park_prepared",
        "kv_retention_p1_parked_gpu_bytes",
        "kv_retention_p1_sampling_restore_count",
    }
)

_LIGHTWEIGHT_WORKER_TIMING_PREFIXES = (
    # Context-Ring per-forward counters are required to distinguish useful
    # Prefill work from staging/partial-attention overhead.
    "worker_context_ring_",
    # Keep the small execution-authority and phase-transition surface used by
    # the ABCDE timeline and correctness checks.  Large allocator snapshots
    # remain available through VLLM_FULL_ITERATION_DEBUG=1.
    "pd_dynamic_sm_execution_",
    "pd_green_context_",
    "pd_layerwise_",
    "pd_output_owner_",
    "pd_phase_handoff_",
    "pd_phase_sync_",
    "pd_async_",
    "green_",
)


def compact_worker_debug_timings(
    timings: Mapping[str, object],
) -> dict[str, object]:
    """Return the bounded worker schema for lightweight timeline records.

    Worker/controller code may retain a rich in-memory dictionary for its own
    decisions.  Serializing that dictionary on every Decode token is a
    separate concern and can contend for the GIL with phase workers.  This
    filter is deliberately applied only at the JSONL boundary.
    """

    return {
        key: value
        for key, value in timings.items()
        if key in _LIGHTWEIGHT_WORKER_TIMING_KEYS
        or key.startswith(_LIGHTWEIGHT_WORKER_TIMING_PREFIXES)
    }
