# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental tiered blockwise attention executor for virtual context.

This module is deliberately isolated from the normal vLLM attention backend.
It provides a callable data-plane prototype for ZoneAtten 0.5.x:

* materialize tiered K/V blocks into a fixed staging buffer,
* compute partial attention block by block,
* merge partial states with online softmax.

It is not wired into model execution by default.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Literal

VirtualTier = Literal["HBM", "DRAM", "STAGING"]
_DEFAULT_CONTEXT_RING_Q_TILE_TOKENS = 256
_DEFAULT_CONTEXT_RING_COMPUTE_GROUP_TOKENS = 512
_DEFAULT_CONTEXT_RING_STAGING_SLOTS = 2
_DEFAULT_CONTEXT_RING_TRITON_PARTIAL_BLOCK_M = 16
_DEFAULT_CONTEXT_RING_TRITON_PARTIAL_BLOCK_N = 64
_TRUE_VALUES = {"1", "true", "yes", "on"}
logger = logging.getLogger(__name__)


def _isolation_flag(feature: str) -> bool:
    # Keep this numerical module importable by CPU smoke tests without vLLM's
    # torch-dependent package initializer.
    value = os.getenv("VLLM_ISOLATION_" + feature,
                      os.getenv("VLLM_ISOLATION_MEMORY_OPTIMIZATIONS", "0"))
    if value not in {"0", "1"}:
        raise ValueError(f"VLLM_ISOLATION_{feature} must be 0 or 1")
    return value == "1"


def _read_bool_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_VALUES


def _read_non_negative_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(0, value)


def context_ring_q_tile_tokens() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_RING_Q_TILE_TOKENS",
        str(_DEFAULT_CONTEXT_RING_Q_TILE_TOKENS),
    )
    try:
        value = int(raw_value)
    except ValueError:
        return _DEFAULT_CONTEXT_RING_Q_TILE_TOKENS
    return max(0, value)


def context_ring_compute_group_tokens() -> int:
    return _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_COMPUTE_GROUP_TOKENS",
        _DEFAULT_CONTEXT_RING_COMPUTE_GROUP_TOKENS,
    )


def context_ring_streaming_wave_groups() -> int:
    """Return the bounded number of base K/V groups in one streaming wave."""

    value = _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS",
        1,
    )
    return max(1, value)


def context_ring_streaming_wave_tokens(
    compute_group_tokens: int | None = None,
) -> int:
    """Return the maximum K/V tokens represented by one streaming wave."""

    base_tokens = (
        context_ring_compute_group_tokens()
        if compute_group_tokens is None
        else max(0, int(compute_group_tokens))
    )
    if base_tokens <= 0:
        return 0
    return base_tokens * context_ring_streaming_wave_groups()


def context_ring_staging_slots() -> int:
    value = _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_STAGING_SLOTS",
        _DEFAULT_CONTEXT_RING_STAGING_SLOTS,
    )
    return max(1, value)


def context_ring_async_prefetch_enabled() -> bool:
    return _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_ASYNC_PREFETCH")


def context_ring_prefetch_distance() -> int:
    return _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_PREFETCH_DISTANCE",
        1,
    )


def context_ring_cuda_event_timing_enabled() -> bool:
    return _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_CUDA_EVENT_TIMING")


def context_ring_staging_direct_multicopy_enabled() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY"
    )


def context_ring_staging_direct_multicopy_strict() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY_STRICT"
    )


def context_ring_triton_partial_enabled() -> bool:
    return _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_TRITON_PARTIAL")


def context_ring_triton_fused_update_enabled() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE"
    )


def context_ring_native_reduction_enabled() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION"
    )


def context_ring_batched_partial_enabled() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL"
    )


def context_ring_batched_workspace_bytes() -> int:
    return _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_WORKSPACE_BYTES",
        1 << 30,
    )


def context_ring_batched_partial_strict() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL_STRICT"
    )


def context_ring_fully_optimized_strict() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT"
    )


def context_ring_paged_hbm_update_enabled() -> bool:
    return _read_bool_env(
        "VLLM_VIRTUAL_CONTEXT_RING_PAGED_HBM_UPDATE"
    )


def context_ring_triton_partial_block_m() -> int:
    value = _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_M",
        _DEFAULT_CONTEXT_RING_TRITON_PARTIAL_BLOCK_M,
    )
    return max(1, value)


def context_ring_triton_partial_block_n() -> int:
    value = _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_N",
        _DEFAULT_CONTEXT_RING_TRITON_PARTIAL_BLOCK_N,
    )
    return max(1, value)


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tiered blockwise attention requires torch at runtime"
        ) from exc


def _custom_ops_module() -> Any:
    return importlib.import_module("vllm._custom_ops")


def _sum_cuda_event_elapsed_ms(
    event_pairs: list[tuple[Any, Any]],
) -> tuple[float, int]:
    total_ms = 0.0
    count = 0
    for start_event, end_event in event_pairs:
        if start_event is None or end_event is None:
            continue
        end_event.synchronize()
        total_ms += float(start_event.elapsed_time(end_event))
        count += 1
    return total_ms, count


@dataclass(frozen=True)
class TieredKVBlock:
    """One logical K/V block in causal-history order."""

    key: Any
    value: Any
    token_start: int
    tier: VirtualTier = "HBM"
    logical_page: int = -1
    source_leases: tuple[Any, ...] = ()

    @property
    def token_count(self) -> int:
        return int(self.key.shape[0])


@dataclass(frozen=True)
class TieredPagedHBMSegment:
    """A logical HBM range addressed by a tensor of physical block ids."""

    key_cache: Any
    value_cache: Any
    block_ids: Any
    token_start: int
    token_count: int
    page_size: int
    page_count: int
    logical_page: int = -1
    tier: VirtualTier = "HBM"


@dataclass(frozen=True)
class TieredRaggedRequestRow:
    query_start: int
    query_end: int
    sequence_length: int
    kv_blocks: list[TieredKVBlock | TieredPagedHBMSegment]
    request_id: str = ""


@dataclass(frozen=True)
class _TieredStreamingComputePlan:
    groups: list[
        tuple[
            int,
            int,
            list[TieredKVBlock | TieredPagedHBMSegment],
        ]
    ]
    base_group_count: int
    max_wave_tokens: int


def _tiered_block_token_count(
    block: TieredKVBlock | TieredPagedHBMSegment,
) -> int:
    if isinstance(block, TieredPagedHBMSegment):
        return int(block.token_count)
    return int(block.key.shape[0])


def _build_streaming_compute_plan(
    kv_blocks: list[TieredKVBlock | TieredPagedHBMSegment],
    *,
    compute_group_tokens: int,
    streaming_wave_groups: int,
) -> _TieredStreamingComputePlan:
    """Coalesce adjacent DRAM groups into bounded streaming waves.

    HBM objects remain separate so paged HBM updates keep reading the native
    cache directly. DRAM objects may be joined only when their logical token
    ranges are contiguous. With ``streaming_wave_groups == 1`` this is the
    original per-compute-group streaming plan.
    """

    grouping_enabled = compute_group_tokens > 0
    wave_group_count = max(1, int(streaming_wave_groups))
    wave_token_limit = (
        compute_group_tokens * wave_group_count
        if grouping_enabled
        else 0
    )
    groups: list[
        tuple[
            int,
            int,
            list[TieredKVBlock | TieredPagedHBMSegment],
        ]
    ] = []
    base_group_count = 0
    max_wave_tokens = 0
    block_index = 0
    while block_index < len(kv_blocks):
        first_block = kv_blocks[block_index]
        group: list[TieredKVBlock | TieredPagedHBMSegment] = [first_block]
        group_tokens = _tiered_block_token_count(first_block)
        next_block_index = block_index + 1
        if (
            not isinstance(first_block, TieredPagedHBMSegment)
            and grouping_enabled
            and first_block.tier != "HBM"
        ):
            while next_block_index < len(kv_blocks):
                next_block = kv_blocks[next_block_index]
                if (
                    isinstance(next_block, TieredPagedHBMSegment)
                    or next_block.tier == "HBM"
                    or next_block.tier != first_block.tier
                ):
                    break
                next_tokens = _tiered_block_token_count(next_block)
                if next_block.token_start != first_block.token_start + group_tokens:
                    break
                if group_tokens + next_tokens > wave_token_limit:
                    break
                group.append(next_block)
                group_tokens += next_tokens
                next_block_index += 1

        groups.append((block_index, next_block_index, group))
        if grouping_enabled:
            base_group_count += max(
                1,
                (group_tokens + compute_group_tokens - 1)
                // compute_group_tokens,
            )
        else:
            base_group_count += 1
        max_wave_tokens = max(max_wave_tokens, group_tokens)
        block_index = next_block_index

    return _TieredStreamingComputePlan(
        groups=groups,
        base_group_count=base_group_count,
        max_wave_tokens=max_wave_tokens,
    )


@dataclass(frozen=True)
class _PendingSourceLeaseRelease:
    event: Any
    leases: tuple[Any, ...]
    request_ids: frozenset[str]


@dataclass(frozen=True)
class TieredBlockwiseAttentionStats:
    query_tokens: int
    key_tokens: int
    block_count: int
    materialized_blocks: int
    hbm_blocks: int
    dram_blocks: int
    staging_blocks: int
    staging_bytes_copied: int
    online_merge_steps: int
    max_key_tokens_per_block: int
    context_ring_q_tile_tokens: int = 0
    context_ring_q_tile_count: int = 0
    context_ring_compute_group_tokens: int = 0
    context_ring_compute_group_count: int = 0
    context_ring_async_prefetch_enabled: int = 0
    context_ring_prefetch_distance: int = 0
    context_ring_staging_slots: int = 0
    context_ring_prefetch_submits: int = 0
    context_ring_prefetch_waits: int = 0
    context_ring_prefetch_bytes: int = 0
    context_ring_staging_non_blocking: int = 0
    context_ring_staging_materialize_calls: int = 0
    context_ring_staging_slot_waits: int = 0
    context_ring_staging_mark_read_done_count: int = 0
    context_ring_staging_pending_read_events: int = 0
    context_ring_staging_state_transitions: int = 0
    context_ring_staging_peak_busy_slots: int = 0
    context_ring_staging_copying_slots: int = 0
    context_ring_staging_computing_slots: int = 0
    context_ring_staging_read_pending_slots: int = 0
    context_ring_staging_direct_multicopy_attempts: int = 0
    context_ring_staging_direct_multicopy_successes: int = 0
    context_ring_staging_direct_multicopy_fallbacks: int = 0
    context_ring_staging_direct_multicopy_runs: int = 0
    context_ring_staging_direct_multicopy_bytes: int = 0
    context_ring_source_pinned_blocks: int = 0
    context_ring_source_unpinned_blocks: int = 0
    context_ring_prefetch_submit_host_duration_s: float = 0.0
    context_ring_prefetch_wait_host_duration_s: float = 0.0
    context_ring_materialize_host_duration_s: float = 0.0
    context_ring_attention_compute_host_duration_s: float = 0.0
    context_ring_total_host_duration_s: float = 0.0
    context_ring_cuda_event_timing_enabled: int = 0
    context_ring_copy_event_elapsed_ms: float = 0.0
    context_ring_wait_event_elapsed_ms: float = 0.0
    context_ring_compute_event_elapsed_ms: float = 0.0
    context_ring_copy_compute_overlap_ratio: float = 0.0
    context_ring_copy_event_count: int = 0
    context_ring_wait_event_count: int = 0
    context_ring_compute_event_count: int = 0
    context_ring_triton_partial_enabled: int = 0
    context_ring_triton_partial_attempts: int = 0
    context_ring_triton_partial_successes: int = 0
    context_ring_triton_partial_fallbacks: int = 0
    context_ring_triton_partial_block_m: int = 0
    context_ring_triton_partial_block_n: int = 0
    context_ring_triton_partial_host_duration_s: float = 0.0
    context_ring_triton_fused_update_enabled: int = 0
    context_ring_native_reduction_enabled: int = 0
    context_ring_triton_fused_update_attempts: int = 0
    context_ring_triton_fused_update_successes: int = 0
    context_ring_triton_fused_update_failures: int = 0
    context_ring_triton_fused_update_kernel_launches: int = 0
    context_ring_triton_fused_update_host_duration_s: float = 0.0
    context_ring_triton_fused_q_tile_loop_bypassed: int = 0
    context_ring_reference_state_allocated: int = 0
    context_ring_output_buffer_direct: int = 0
    context_ring_lse_skipped: int = 0
    context_ring_paged_hbm_update_enabled: int = 0
    context_ring_paged_hbm_segments: int = 0
    context_ring_paged_hbm_pages: int = 0
    context_ring_paged_hbm_kernel_launches: int = 0
    context_ring_copy_stream_reused: int = 0
    context_ring_fused_state_reused: int = 0
    isolation_ring_state_capacity_tokens: int = 0
    isolation_ring_state_bytes: int = 0
    isolation_ring_state_shared_bytes: int = 0
    isolation_ring_source_ready_waits: int = 0
    isolation_ring_historical_early_groups: int = 0
    context_ring_source_leases_acquired: int = 0
    context_ring_source_lease_releases_deferred: int = 0
    context_ring_source_lease_releases_completed: int = 0
    context_ring_source_lease_releases_pending: int = 0
    context_ring_batched_partial_enabled: int = 0
    context_ring_batched_partial_attempts: int = 0
    context_ring_batched_partial_successes: int = 0
    context_ring_batched_partial_fallbacks: int = 0
    context_ring_batched_partial_stage_a_launches: int = 0
    context_ring_batched_partial_stage_b_launches: int = 0
    context_ring_batched_partial_group_count: int = 0
    context_ring_batched_partial_workspace_bytes: int = 0
    context_ring_host_group_loop_bypassed: int = 0
    context_ring_streaming_wave_groups: int = 1
    context_ring_streaming_wave_count: int = 0
    context_ring_streaming_base_group_count: int = 0
    context_ring_streaming_groups_coalesced: int = 0
    context_ring_streaming_max_wave_tokens: int = 0
    context_ring_streaming_staging_workspace_bytes: int = 0

    def to_debug_fields(self, prefix: str) -> dict[str, int | float]:
        def field_name(name: str) -> str:
            if not prefix:
                return name
            if name.startswith(f"{prefix}_"):
                return name
            return f"{prefix}_{name}"

        return {
            field_name("query_tokens"): self.query_tokens,
            field_name("key_tokens"): self.key_tokens,
            field_name("block_count"): self.block_count,
            field_name("materialized_blocks"): self.materialized_blocks,
            field_name("hbm_blocks"): self.hbm_blocks,
            field_name("dram_blocks"): self.dram_blocks,
            field_name("staging_blocks"): self.staging_blocks,
            field_name("staging_bytes_copied"): self.staging_bytes_copied,
            field_name("online_merge_steps"): self.online_merge_steps,
            field_name("max_key_tokens_per_block"): (
                self.max_key_tokens_per_block
            ),
            field_name("context_ring_q_tile_tokens"): (
                self.context_ring_q_tile_tokens
            ),
            field_name("context_ring_q_tile_count"): (
                self.context_ring_q_tile_count
            ),
            field_name("context_ring_compute_group_tokens"): (
                self.context_ring_compute_group_tokens
            ),
            field_name("context_ring_compute_group_count"): (
                self.context_ring_compute_group_count
            ),
            field_name("context_ring_async_prefetch_enabled"): (
                self.context_ring_async_prefetch_enabled
            ),
            field_name("context_ring_prefetch_distance"): (
                self.context_ring_prefetch_distance
            ),
            field_name("context_ring_staging_slots"): (
                self.context_ring_staging_slots
            ),
            field_name("context_ring_prefetch_submits"): (
                self.context_ring_prefetch_submits
            ),
            field_name("context_ring_prefetch_waits"): (
                self.context_ring_prefetch_waits
            ),
            field_name("context_ring_prefetch_bytes"): (
                self.context_ring_prefetch_bytes
            ),
            field_name("context_ring_staging_non_blocking"): (
                self.context_ring_staging_non_blocking
            ),
            field_name("context_ring_staging_materialize_calls"): (
                self.context_ring_staging_materialize_calls
            ),
            field_name("context_ring_staging_slot_waits"): (
                self.context_ring_staging_slot_waits
            ),
            field_name("context_ring_staging_mark_read_done_count"): (
                self.context_ring_staging_mark_read_done_count
            ),
            field_name("context_ring_staging_pending_read_events"): (
                self.context_ring_staging_pending_read_events
            ),
            field_name("context_ring_staging_state_transitions"): (
                self.context_ring_staging_state_transitions
            ),
            field_name("context_ring_staging_peak_busy_slots"): (
                self.context_ring_staging_peak_busy_slots
            ),
            field_name("context_ring_staging_copying_slots"): (
                self.context_ring_staging_copying_slots
            ),
            field_name("context_ring_staging_computing_slots"): (
                self.context_ring_staging_computing_slots
            ),
            field_name("context_ring_staging_read_pending_slots"): (
                self.context_ring_staging_read_pending_slots
            ),
            field_name("context_ring_staging_direct_multicopy_attempts"): (
                self.context_ring_staging_direct_multicopy_attempts
            ),
            field_name("context_ring_staging_direct_multicopy_successes"): (
                self.context_ring_staging_direct_multicopy_successes
            ),
            field_name("context_ring_staging_direct_multicopy_fallbacks"): (
                self.context_ring_staging_direct_multicopy_fallbacks
            ),
            field_name("context_ring_staging_direct_multicopy_runs"): (
                self.context_ring_staging_direct_multicopy_runs
            ),
            field_name("context_ring_staging_direct_multicopy_bytes"): (
                self.context_ring_staging_direct_multicopy_bytes
            ),
            field_name("context_ring_source_pinned_blocks"): (
                self.context_ring_source_pinned_blocks
            ),
            field_name("context_ring_source_unpinned_blocks"): (
                self.context_ring_source_unpinned_blocks
            ),
            field_name("context_ring_prefetch_submit_host_duration_s"): (
                self.context_ring_prefetch_submit_host_duration_s
            ),
            field_name("context_ring_prefetch_wait_host_duration_s"): (
                self.context_ring_prefetch_wait_host_duration_s
            ),
            field_name("context_ring_materialize_host_duration_s"): (
                self.context_ring_materialize_host_duration_s
            ),
            field_name("context_ring_attention_compute_host_duration_s"): (
                self.context_ring_attention_compute_host_duration_s
            ),
            field_name("context_ring_total_host_duration_s"): (
                self.context_ring_total_host_duration_s
            ),
            field_name("context_ring_cuda_event_timing_enabled"): (
                self.context_ring_cuda_event_timing_enabled
            ),
            field_name("context_ring_copy_event_elapsed_ms"): (
                self.context_ring_copy_event_elapsed_ms
            ),
            field_name("context_ring_wait_event_elapsed_ms"): (
                self.context_ring_wait_event_elapsed_ms
            ),
            field_name("context_ring_compute_event_elapsed_ms"): (
                self.context_ring_compute_event_elapsed_ms
            ),
            field_name("context_ring_copy_compute_overlap_ratio"): (
                self.context_ring_copy_compute_overlap_ratio
            ),
            field_name("context_ring_copy_event_count"): (
                self.context_ring_copy_event_count
            ),
            field_name("context_ring_wait_event_count"): (
                self.context_ring_wait_event_count
            ),
            field_name("context_ring_compute_event_count"): (
                self.context_ring_compute_event_count
            ),
            field_name("context_ring_triton_partial_enabled"): (
                self.context_ring_triton_partial_enabled
            ),
            field_name("context_ring_triton_partial_attempts"): (
                self.context_ring_triton_partial_attempts
            ),
            field_name("context_ring_triton_partial_successes"): (
                self.context_ring_triton_partial_successes
            ),
            field_name("context_ring_triton_partial_fallbacks"): (
                self.context_ring_triton_partial_fallbacks
            ),
            field_name("context_ring_triton_partial_block_m"): (
                self.context_ring_triton_partial_block_m
            ),
            field_name("context_ring_triton_partial_block_n"): (
                self.context_ring_triton_partial_block_n
            ),
            field_name("context_ring_triton_partial_host_duration_s"): (
                self.context_ring_triton_partial_host_duration_s
            ),
            field_name("context_ring_triton_fused_update_enabled"): (
                self.context_ring_triton_fused_update_enabled
            ),
            field_name("context_ring_native_reduction_enabled"): (
                self.context_ring_native_reduction_enabled
            ),
            field_name("context_ring_triton_fused_update_attempts"): (
                self.context_ring_triton_fused_update_attempts
            ),
            field_name("context_ring_triton_fused_update_successes"): (
                self.context_ring_triton_fused_update_successes
            ),
            field_name("context_ring_triton_fused_update_failures"): (
                self.context_ring_triton_fused_update_failures
            ),
            field_name(
                "context_ring_triton_fused_update_kernel_launches"
            ): self.context_ring_triton_fused_update_kernel_launches,
            field_name(
                "context_ring_triton_fused_update_host_duration_s"
            ): self.context_ring_triton_fused_update_host_duration_s,
            field_name(
                "context_ring_triton_fused_q_tile_loop_bypassed"
            ): self.context_ring_triton_fused_q_tile_loop_bypassed,
            field_name("context_ring_reference_state_allocated"): (
                self.context_ring_reference_state_allocated
            ),
            field_name("context_ring_output_buffer_direct"): (
                self.context_ring_output_buffer_direct
            ),
            field_name("context_ring_lse_skipped"): (
                self.context_ring_lse_skipped
            ),
            field_name("context_ring_paged_hbm_update_enabled"): (
                self.context_ring_paged_hbm_update_enabled
            ),
            field_name("context_ring_paged_hbm_segments"): (
                self.context_ring_paged_hbm_segments
            ),
            field_name("context_ring_paged_hbm_pages"): (
                self.context_ring_paged_hbm_pages
            ),
            field_name("context_ring_paged_hbm_kernel_launches"): (
                self.context_ring_paged_hbm_kernel_launches
            ),
            field_name("context_ring_copy_stream_reused"): (
                self.context_ring_copy_stream_reused
            ),
            field_name("context_ring_fused_state_reused"): (
                self.context_ring_fused_state_reused
            ),
            field_name("isolation_ring_state_capacity_tokens"): (
                self.isolation_ring_state_capacity_tokens
            ),
            field_name("isolation_ring_state_bytes"): self.isolation_ring_state_bytes,
            field_name("isolation_ring_state_shared_bytes"): self.isolation_ring_state_shared_bytes,
            field_name("isolation_ring_source_ready_waits"): self.isolation_ring_source_ready_waits,
            field_name("isolation_ring_historical_early_groups"): self.isolation_ring_historical_early_groups,
            field_name("context_ring_source_leases_acquired"): (
                self.context_ring_source_leases_acquired
            ),
            field_name("context_ring_source_lease_releases_deferred"): (
                self.context_ring_source_lease_releases_deferred
            ),
            field_name("context_ring_source_lease_releases_completed"): (
                self.context_ring_source_lease_releases_completed
            ),
            field_name("context_ring_source_lease_releases_pending"): (
                self.context_ring_source_lease_releases_pending
            ),
            field_name("context_ring_batched_partial_enabled"): (
                self.context_ring_batched_partial_enabled
            ),
            field_name("context_ring_batched_partial_attempts"): (
                self.context_ring_batched_partial_attempts
            ),
            field_name("context_ring_batched_partial_successes"): (
                self.context_ring_batched_partial_successes
            ),
            field_name("context_ring_batched_partial_fallbacks"): (
                self.context_ring_batched_partial_fallbacks
            ),
            field_name("context_ring_batched_partial_stage_a_launches"): (
                self.context_ring_batched_partial_stage_a_launches
            ),
            field_name("context_ring_batched_partial_stage_b_launches"): (
                self.context_ring_batched_partial_stage_b_launches
            ),
            field_name("context_ring_batched_partial_group_count"): (
                self.context_ring_batched_partial_group_count
            ),
            field_name("context_ring_batched_partial_workspace_bytes"): (
                self.context_ring_batched_partial_workspace_bytes
            ),
            field_name("context_ring_host_group_loop_bypassed"): (
                self.context_ring_host_group_loop_bypassed
            ),
            field_name("context_ring_streaming_wave_groups"): (
                self.context_ring_streaming_wave_groups
            ),
            field_name("context_ring_streaming_wave_count"): (
                self.context_ring_streaming_wave_count
            ),
            field_name("context_ring_streaming_base_group_count"): (
                self.context_ring_streaming_base_group_count
            ),
            field_name("context_ring_streaming_groups_coalesced"): (
                self.context_ring_streaming_groups_coalesced
            ),
            field_name("context_ring_streaming_max_wave_tokens"): (
                self.context_ring_streaming_max_wave_tokens
            ),
            field_name(
                "context_ring_streaming_staging_workspace_bytes"
            ): self.context_ring_streaming_staging_workspace_bytes,
        }


@dataclass(frozen=True)
class TieredBlockwiseAttentionResult:
    output: Any
    lse: Any | None
    stats: TieredBlockwiseAttentionStats


@dataclass(frozen=True)
class HBMStagingBufferMetrics:
    materialize_calls: int
    slot_waits: int
    mark_read_done_calls: int
    pending_read_events: int
    state_transitions: int
    peak_busy_slots: int
    copying_slots: int
    computing_slots: int
    read_pending_slots: int
    direct_multicopy_attempts: int
    direct_multicopy_successes: int
    direct_multicopy_fallbacks: int
    direct_multicopy_runs: int
    direct_multicopy_bytes: int


class HBMStagingBuffer:
    """Small fixed-slot staging buffer for K/V block materialization.

    The buffer is synchronous in v0.5.5.1. It gives the next integration patch
    a stable place to add explicit copy streams and double buffering.
    """

    def __init__(self, num_slots: int = 2) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        self._key_slots: list[Any | None] = [None] * num_slots
        self._value_slots: list[Any | None] = [None] * num_slots
        self._batched_key_slots: list[Any | None] = [None] * num_slots
        self._batched_value_slots: list[Any | None] = [None] * num_slots
        self._slot_read_events: list[Any | None] = [None] * num_slots
        self._materialize_calls = 0
        self._slot_waits = 0
        self._mark_read_done_calls = 0
        self._slot_states = ["FREE"] * num_slots
        self._state_transitions = 0
        self._peak_busy_slots = 0
        self._direct_multicopy_attempts = 0
        self._direct_multicopy_successes = 0
        self._direct_multicopy_fallbacks = 0
        self._direct_multicopy_runs = 0
        self._direct_multicopy_bytes = 0
        self._direct_multicopy_fallback_warned = False

    @staticmethod
    def _is_inference_tensor(tensor: Any | None) -> bool:
        if tensor is None:
            return False
        is_inference = getattr(tensor, "is_inference", None)
        return bool(is_inference()) if callable(is_inference) else False

    @staticmethod
    def _allocate_normal_tensor(
        factory: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Allocate a staging tensor independent of the caller's mode.

        The shared staging pool is first touched by model forward under
        ``torch.inference_mode()`` but is also mutated by native-reload worker
        threads.  A normal, requires-grad-disabled tensor is valid in both
        execution domains; an inference tensor is not mutable by the worker.
        """

        torch = _torch()
        with torch.inference_mode(False):
            tensor = factory(*args, requires_grad=False, **kwargs)
        if HBMStagingBuffer._is_inference_tensor(tensor):
            raise RuntimeError(
                "shared HBM staging allocation unexpectedly produced an "
                "inference tensor"
            )
        return tensor

    @staticmethod
    def _compatible(slot: Any | None, source: Any, target_device: Any) -> bool:
        return (
            slot is not None
            and not HBMStagingBuffer._is_inference_tensor(slot)
            and tuple(slot.shape) == tuple(source.shape)
            and slot.dtype == source.dtype
            and slot.device == target_device
        )

    @staticmethod
    def _nbytes(tensor: Any) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def _try_direct_multicopy(
        self,
        *,
        source_keys: list[Any],
        source_values: list[Any],
        destination_key: Any,
        destination_value: Any,
        destination_group_ids: list[int],
        destination_token_offsets: list[int],
        target_device: Any,
        bytes_copied: int,
    ) -> bool:
        """Submit one direct C++ call for any Ring staging layout."""

        if (
            not context_ring_staging_direct_multicopy_enabled()
            or getattr(target_device, "type", "") != "cuda"
        ):
            return False
        torch = _torch()
        self._direct_multicopy_attempts += 1
        try:
            copy_runs = int(
                _custom_ops_module().tiered_h2d_staging_multicopy(
                    source_keys,
                    source_values,
                    destination_key,
                    destination_value,
                    torch.tensor(
                        destination_group_ids,
                        dtype=torch.int64,
                        device="cpu",
                    ),
                    torch.tensor(
                        destination_token_offsets,
                        dtype=torch.int64,
                        device="cpu",
                    ),
                )
            )
            self._direct_multicopy_successes += 1
            self._direct_multicopy_runs += copy_runs
            self._direct_multicopy_bytes += bytes_copied
            return True
        except Exception as exc:
            self._direct_multicopy_fallbacks += 1
            if context_ring_staging_direct_multicopy_strict():
                raise RuntimeError(
                    "Context Ring staging direct multicopy failed"
                ) from exc
            if not self._direct_multicopy_fallback_warned:
                logger.warning(
                    "Context Ring staging direct multicopy failed; "
                    "falling back to per-source Tensor.copy_: %s",
                    exc,
                )
                self._direct_multicopy_fallback_warned = True
            return False

    def _set_slot_state(self, slot: int, state: str) -> None:
        if self._slot_states[slot] == state:
            return
        self._slot_states[slot] = state
        self._state_transitions += 1
        busy_slots = sum(
            slot_state != "FREE" for slot_state in self._slot_states
        )
        self._peak_busy_slots = max(self._peak_busy_slots, busy_slots)

    def _wait_until_reusable(self, slot: int, target_device: Any) -> None:
        torch = _torch()
        read_done_event = self._slot_read_events[slot]
        if (
            read_done_event is not None
            and getattr(target_device, "type", "") == "cuda"
        ):
            self._slot_waits += 1
            torch.cuda.current_stream(device=target_device).wait_event(
                read_done_event
            )
            self._slot_read_events[slot] = None
        self._set_slot_state(slot, "FREE")

    @staticmethod
    def _compatible_capacity(
        slot: Any | None,
        *,
        token_count: int,
        heads: int,
        head_dim: int,
        dtype: Any,
        target_device: Any,
    ) -> bool:
        return (
            slot is not None
            and not HBMStagingBuffer._is_inference_tensor(slot)
            and int(slot.shape[0]) >= token_count
            and int(slot.shape[1]) == heads
            and int(slot.shape[2]) == head_dim
            and slot.dtype == dtype
            and slot.device == target_device
        )

    def materialize(
        self,
        key: Any,
        value: Any,
        *,
        target_device: Any,
        slot_index: int,
        non_blocking: bool = True,
    ) -> tuple[Any, Any, int]:
        torch = _torch()
        slot = slot_index % self.num_slots
        self._materialize_calls += 1
        self._wait_until_reusable(slot, target_device)

        if not self._compatible(self._key_slots[slot], key, target_device):
            self._key_slots[slot] = self._allocate_normal_tensor(
                torch.empty_strided,
                tuple(key.shape),
                tuple(key.stride()),
                dtype=key.dtype,
                device=target_device,
            )
        if not self._compatible(self._value_slots[slot], value, target_device):
            self._value_slots[slot] = self._allocate_normal_tensor(
                torch.empty_strided,
                tuple(value.shape),
                tuple(value.stride()),
                dtype=value.dtype,
                device=target_device,
            )

        staged_key = self._key_slots[slot]
        staged_value = self._value_slots[slot]
        self._set_slot_state(slot, "COPYING")
        bytes_copied = self._nbytes(key) + self._nbytes(value)
        with torch.inference_mode():
            direct = self._try_direct_multicopy(
                source_keys=[key],
                source_values=[value],
                destination_key=staged_key.unsqueeze(0),
                destination_value=staged_value.unsqueeze(0),
                destination_group_ids=[0],
                destination_token_offsets=[0],
                target_device=target_device,
                bytes_copied=bytes_copied,
            )
            if not direct:
                staged_key.copy_(key, non_blocking=non_blocking)
                staged_value.copy_(value, non_blocking=non_blocking)
        return (
            staged_key,
            staged_value,
            bytes_copied,
        )

    def materialize_group(
        self,
        sources: list[tuple[Any, Any]],
        *,
        target_device: Any,
        slot_index: int,
        non_blocking: bool = True,
    ) -> tuple[Any, Any, int]:
        if not sources:
            raise ValueError("staging group must contain at least one source")
        torch = _torch()
        slot = slot_index % self.num_slots
        self._materialize_calls += 1
        self._wait_until_reusable(slot, target_device)

        first_key, first_value = sources[0]
        total_tokens = sum(int(key.shape[0]) for key, _ in sources)
        heads = int(first_key.shape[1])
        head_dim = int(first_key.shape[2])
        if not self._compatible_capacity(
            self._key_slots[slot],
            token_count=total_tokens,
            heads=heads,
            head_dim=head_dim,
            dtype=first_key.dtype,
            target_device=target_device,
        ):
            self._key_slots[slot] = self._allocate_normal_tensor(
                torch.empty,
                (total_tokens, heads, head_dim),
                dtype=first_key.dtype,
                device=target_device,
            )
        if not self._compatible_capacity(
            self._value_slots[slot],
            token_count=total_tokens,
            heads=int(first_value.shape[1]),
            head_dim=int(first_value.shape[2]),
            dtype=first_value.dtype,
            target_device=target_device,
        ):
            self._value_slots[slot] = self._allocate_normal_tensor(
                torch.empty,
                (
                    total_tokens,
                    int(first_value.shape[1]),
                    int(first_value.shape[2]),
                ),
                dtype=first_value.dtype,
                device=target_device,
            )

        staged_key = self._key_slots[slot][:total_tokens]
        staged_value = self._value_slots[slot][:total_tokens]
        self._set_slot_state(slot, "COPYING")
        offsets: list[int] = []
        offset = 0
        for key, _ in sources:
            offsets.append(offset)
            offset += int(key.shape[0])
        bytes_copied = sum(
            self._nbytes(key) + self._nbytes(value)
            for key, value in sources
        )
        with torch.inference_mode():
            direct = self._try_direct_multicopy(
                source_keys=[key for key, _ in sources],
                source_values=[value for _, value in sources],
                destination_key=staged_key.unsqueeze(0),
                destination_value=staged_value.unsqueeze(0),
                destination_group_ids=[0] * len(sources),
                destination_token_offsets=offsets,
                target_device=target_device,
                bytes_copied=bytes_copied,
            )
            if not direct:
                for (key, value), offset in zip(sources, offsets):
                    token_count = int(key.shape[0])
                    staged_key[offset:offset + token_count].copy_(
                        key, non_blocking=non_blocking
                    )
                    staged_value[offset:offset + token_count].copy_(
                        value, non_blocking=non_blocking
                    )
        return (
            staged_key,
            staged_value,
            bytes_copied,
        )

    def materialize_group_batch(
        self,
        sources_by_group: list[list[tuple[Any, Any]]],
        *,
        target_device: Any,
        slot_index: int,
        non_blocking: bool = True,
    ) -> tuple[Any, Any, list[int], int]:
        """Pack a layer's K/V groups into one reusable GPU staging slot."""

        if not sources_by_group or any(not group for group in sources_by_group):
            raise ValueError("every batched staging group must be non-empty")
        torch = _torch()
        slot = slot_index % self.num_slots
        self._materialize_calls += 1
        self._wait_until_reusable(slot, target_device)
        token_counts = [
            sum(int(key.shape[0]) for key, _ in group)
            for group in sources_by_group
        ]
        group_count = len(sources_by_group)
        max_tokens = max(token_counts)
        first_key, first_value = sources_by_group[0][0]
        key_shape = (
            group_count,
            max_tokens,
            int(first_key.shape[1]),
            int(first_key.shape[2]),
        )
        value_shape = (
            group_count,
            max_tokens,
            int(first_value.shape[1]),
            int(first_value.shape[2]),
        )
        key_slot = self._batched_key_slots[slot]
        if (
            key_slot is None
            or self._is_inference_tensor(key_slot)
            or tuple(key_slot.shape) != key_shape
            or key_slot.dtype != first_key.dtype
            or key_slot.device != target_device
        ):
            key_slot = self._allocate_normal_tensor(
                torch.empty,
                key_shape,
                dtype=first_key.dtype,
                device=target_device,
            )
            self._batched_key_slots[slot] = key_slot
        value_slot = self._batched_value_slots[slot]
        if (
            value_slot is None
            or self._is_inference_tensor(value_slot)
            or tuple(value_slot.shape) != value_shape
            or value_slot.dtype != first_value.dtype
            or value_slot.device != target_device
        ):
            value_slot = self._allocate_normal_tensor(
                torch.empty,
                value_shape,
                dtype=first_value.dtype,
                device=target_device,
            )
            self._batched_value_slots[slot] = value_slot

        self._set_slot_state(slot, "COPYING")
        bytes_copied = 0
        flat_keys: list[Any] = []
        flat_values: list[Any] = []
        destination_group_ids: list[int] = []
        destination_token_offsets: list[int] = []
        for group_index, group in enumerate(sources_by_group):
            offset = 0
            for key, value in group:
                flat_keys.append(key)
                flat_values.append(value)
                destination_group_ids.append(group_index)
                destination_token_offsets.append(offset)
                bytes_copied += self._nbytes(key) + self._nbytes(value)
                offset += int(key.shape[0])

        with torch.inference_mode():
            direct = self._try_direct_multicopy(
                source_keys=flat_keys,
                source_values=flat_values,
                destination_key=key_slot,
                destination_value=value_slot,
                destination_group_ids=destination_group_ids,
                destination_token_offsets=destination_token_offsets,
                target_device=target_device,
                bytes_copied=bytes_copied,
            )
            if not direct:
                for group_index, group in enumerate(sources_by_group):
                    offset = 0
                    for key, value in group:
                        token_count = int(key.shape[0])
                        key_slot[
                            group_index, offset:offset + token_count
                        ].copy_(key, non_blocking=non_blocking)
                        value_slot[
                            group_index, offset:offset + token_count
                        ].copy_(value, non_blocking=non_blocking)
                        offset += token_count
        return key_slot, value_slot, token_counts, bytes_copied

    def mark_slot_computing(self, slot_index: int) -> None:
        slot = slot_index % self.num_slots
        self._set_slot_state(slot, "COMPUTING")

    def mark_slot_read_done(self, slot_index: int, target_device: Any) -> None:
        if getattr(target_device, "type", "") != "cuda":
            return
        torch = _torch()
        if not torch.cuda.is_available():
            return
        slot = slot_index % self.num_slots
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=target_device))
        self._slot_read_events[slot] = event
        self._set_slot_state(slot, "READ_PENDING")
        self._mark_read_done_calls += 1

    def metrics_snapshot(self) -> HBMStagingBufferMetrics:
        return HBMStagingBufferMetrics(
            materialize_calls=self._materialize_calls,
            slot_waits=self._slot_waits,
            mark_read_done_calls=self._mark_read_done_calls,
            pending_read_events=sum(
                event is not None for event in self._slot_read_events
            ),
            state_transitions=self._state_transitions,
            peak_busy_slots=self._peak_busy_slots,
            copying_slots=sum(
                state == "COPYING" for state in self._slot_states
            ),
            computing_slots=sum(
                state == "COMPUTING" for state in self._slot_states
            ),
            read_pending_slots=sum(
                state == "READ_PENDING" for state in self._slot_states
            ),
            direct_multicopy_attempts=self._direct_multicopy_attempts,
            direct_multicopy_successes=self._direct_multicopy_successes,
            direct_multicopy_fallbacks=self._direct_multicopy_fallbacks,
            direct_multicopy_runs=self._direct_multicopy_runs,
            direct_multicopy_bytes=self._direct_multicopy_bytes,
        )


_SHARED_STAGING_LOCK = Lock()
_SHARED_STAGING_BUFFERS: dict[tuple[str, int], HBMStagingBuffer] = {}
_SHARED_STAGING_CURSORS: dict[tuple[str, int], int] = {}


def get_shared_hbm_staging_buffer(
    target_device: Any,
    num_slots: int | None = None,
) -> HBMStagingBuffer:
    """Return the staging pool shared by context ring and native reload."""

    if num_slots is None:
        num_slots = context_ring_staging_slots()
    cache_key = (str(target_device), int(num_slots))
    with _SHARED_STAGING_LOCK:
        staging = _SHARED_STAGING_BUFFERS.get(cache_key)
        if staging is None:
            staging = HBMStagingBuffer(num_slots)
            _SHARED_STAGING_BUFFERS[cache_key] = staging
            _SHARED_STAGING_CURSORS[cache_key] = 0
        return staging


def next_shared_hbm_staging_slot(
    target_device: Any,
    num_slots: int | None = None,
) -> int:
    if num_slots is None:
        num_slots = context_ring_staging_slots()
    cache_key = (str(target_device), int(num_slots))
    with _SHARED_STAGING_LOCK:
        cursor = _SHARED_STAGING_CURSORS.get(cache_key, 0)
        _SHARED_STAGING_CURSORS[cache_key] = (cursor + 1) % int(num_slots)
        return cursor


@dataclass(frozen=True)
class _PrefetchedKVBlock:
    key: Any
    value: Any
    bytes_copied: int
    event: Any
    copy_start_event: Any | None = None
    copy_done_event: Any | None = None


@dataclass(frozen=True)
class _PrefetchedKVGroup:
    key: Any
    value: Any
    bytes_copied: int
    materialized_count: int
    mark_read_done: int
    event: Any
    staging_slot_index: int
    copy_start_event: Any | None = None
    copy_done_event: Any | None = None


@dataclass(frozen=True)
class _MaterializedKVGroup:
    key: Any
    value: Any
    bytes_copied: int
    materialized_count: int
    mark_read_done: int
    prefetched: bool
    wait_host_duration_s: float
    staging_slot_index: int | None = None
    copy_start_event: Any | None = None
    copy_done_event: Any | None = None
    wait_start_event: Any | None = None
    wait_done_event: Any | None = None


class _ContextRingPrefetchPipeline:
    """Double-buffered CPU/DRAM-to-HBM staging for the reference executor."""

    def __init__(
        self,
        staging_buffer: HBMStagingBuffer,
        target_device: Any,
        *,
        non_blocking: bool,
        cuda_event_timing_enabled: bool,
        copy_stream: Any | None = None,
        source_ready_event: Any | None = None,
    ) -> None:
        torch = _torch()
        self.staging_buffer = staging_buffer
        self.target_device = target_device
        self.non_blocking = non_blocking
        self.cuda_event_timing_enabled = cuda_event_timing_enabled
        self.copy_stream = (
            copy_stream
            if copy_stream is not None
            else torch.cuda.Stream(device=target_device)
        )
        self._pending: dict[int, _PrefetchedKVBlock] = {}
        self._pending_groups: dict[int, _PrefetchedKVGroup] = {}
        self.source_ready_event = source_ready_event
        self.source_ready_waits = 0
        self.historical_early_groups = 0

    def _wait_for_gpu_sources(self, group: list[TieredKVBlock]) -> None:
        if self.source_ready_event is None or self.source_ready_waits:
            return
        if any(getattr(getattr(tensor, "device", None), "type", "") == "cuda"
               for block in group for tensor in (block.key, block.value)):
            self.copy_stream.wait_event(self.source_ready_event)
            self.source_ready_waits += 1
        else:
            self.historical_early_groups += 1

    @staticmethod
    def supported(target_device: Any) -> bool:
        if getattr(target_device, "type", "") != "cuda":
            return False
        torch = _torch()
        return bool(torch.cuda.is_available())

    def submit(self, block_index: int, block: TieredKVBlock) -> int:
        if block_index in self._pending:
            return 0
        self._wait_for_gpu_sources([block])
        torch = _torch()
        with torch.cuda.stream(self.copy_stream):
            copy_start_event = None
            if self.cuda_event_timing_enabled:
                copy_start_event = torch.cuda.Event(enable_timing=True)
                copy_start_event.record(self.copy_stream)
            key, value, bytes_copied = self.staging_buffer.materialize(
                block.key,
                block.value,
                target_device=self.target_device,
                slot_index=block_index,
                non_blocking=self.non_blocking,
            )
            event = torch.cuda.Event(
                enable_timing=self.cuda_event_timing_enabled
            )
            event.record(self.copy_stream)
        self._pending[block_index] = _PrefetchedKVBlock(
            key=key,
            value=value,
            bytes_copied=bytes_copied,
            event=event,
            copy_start_event=copy_start_event,
            copy_done_event=event if self.cuda_event_timing_enabled else None,
        )
        return bytes_copied

    def _needs_materialize(self, block: TieredKVBlock) -> bool:
        return block.tier != "HBM" or block.key.device != self.target_device

    def _materialize_group(
        self,
        first_block_index: int,
        group: list[TieredKVBlock],
        staging_slot_index: int,
    ) -> tuple[Any, Any, int, int, int]:
        if len(group) == 1:
            block = group[0]
            if not self._needs_materialize(block):
                return block.key, block.value, 0, 0, 0
            key, value, bytes_copied = self.staging_buffer.materialize(
                block.key,
                block.value,
                target_device=self.target_device,
                slot_index=staging_slot_index,
                non_blocking=self.non_blocking,
            )
            return key, value, bytes_copied, 1, 1

        group_key, group_value, _ = self.staging_buffer.materialize_group(
            [(block.key, block.value) for block in group],
            target_device=self.target_device,
            slot_index=staging_slot_index,
            non_blocking=self.non_blocking,
        )
        bytes_copied = 0
        materialized_count = 0
        for block in group:
            if self._needs_materialize(block):
                bytes_copied += (
                    HBMStagingBuffer._nbytes(block.key)
                    + HBMStagingBuffer._nbytes(block.value)
                )
                materialized_count += 1
        return group_key, group_value, bytes_copied, materialized_count, 1

    def submit_group(
        self,
        first_block_index: int,
        group: list[TieredKVBlock],
        *,
        staging_slot_index: int,
    ) -> tuple[int, bool]:
        if first_block_index in self._pending_groups:
            return 0, False
        if len(group) == 1 and not self._needs_materialize(group[0]):
            return 0, False
        self._wait_for_gpu_sources(group)
        torch = _torch()
        with torch.cuda.stream(self.copy_stream):
            copy_start_event = None
            if self.cuda_event_timing_enabled:
                copy_start_event = torch.cuda.Event(enable_timing=True)
                copy_start_event.record(self.copy_stream)
            key, value, bytes_copied, materialized_count, mark_read_done = (
                self._materialize_group(
                    first_block_index,
                    group,
                    staging_slot_index,
                )
            )
            event = torch.cuda.Event(
                enable_timing=self.cuda_event_timing_enabled
            )
            event.record(self.copy_stream)
        self._pending_groups[first_block_index] = _PrefetchedKVGroup(
            key=key,
            value=value,
            bytes_copied=bytes_copied,
            materialized_count=materialized_count,
            mark_read_done=mark_read_done,
            event=event,
            staging_slot_index=staging_slot_index,
            copy_start_event=copy_start_event,
            copy_done_event=event if self.cuda_event_timing_enabled else None,
        )
        return bytes_copied, True

    def get_group(
        self,
        first_block_index: int,
        group: list[TieredKVBlock],
        *,
        staging_slot_index: int,
    ) -> _MaterializedKVGroup:
        prefetched = self._pending_groups.pop(first_block_index, None)
        if prefetched is None:
            key, value, bytes_copied, materialized_count, mark_read_done = (
                self._materialize_group(
                    first_block_index,
                    group,
                    staging_slot_index,
                )
            )
            return _MaterializedKVGroup(
                key=key,
                value=value,
                bytes_copied=bytes_copied,
                materialized_count=materialized_count,
                mark_read_done=mark_read_done,
                prefetched=False,
                wait_host_duration_s=0.0,
                staging_slot_index=staging_slot_index,
            )
        torch = _torch()
        stream = torch.cuda.current_stream(device=self.target_device)
        wait_start_event = None
        wait_done_event = None
        wait_started = time.perf_counter()
        if self.cuda_event_timing_enabled:
            wait_start_event = torch.cuda.Event(enable_timing=True)
            wait_done_event = torch.cuda.Event(enable_timing=True)
            wait_start_event.record(stream)
            stream.wait_event(prefetched.event)
            wait_done_event.record(stream)
        else:
            stream.wait_event(prefetched.event)
        wait_host_duration_s = time.perf_counter() - wait_started
        return _MaterializedKVGroup(
            key=prefetched.key,
            value=prefetched.value,
            bytes_copied=prefetched.bytes_copied,
            materialized_count=prefetched.materialized_count,
            mark_read_done=prefetched.mark_read_done,
            prefetched=True,
            wait_host_duration_s=wait_host_duration_s,
            staging_slot_index=prefetched.staging_slot_index,
            copy_start_event=prefetched.copy_start_event,
            copy_done_event=prefetched.copy_done_event,
            wait_start_event=wait_start_event,
            wait_done_event=wait_done_event,
        )

    def get(
        self,
        block_index: int,
        block: TieredKVBlock,
    ) -> _MaterializedKVGroup:
        prefetched = self._pending.pop(block_index, None)
        if prefetched is None:
            key, value, bytes_copied = self.staging_buffer.materialize(
                block.key,
                block.value,
                target_device=self.target_device,
                slot_index=block_index,
                non_blocking=self.non_blocking,
            )
            return _MaterializedKVGroup(
                key=key,
                value=value,
                bytes_copied=bytes_copied,
                materialized_count=1,
                mark_read_done=1,
                prefetched=False,
                wait_host_duration_s=0.0,
                staging_slot_index=block_index,
            )
        torch = _torch()
        stream = torch.cuda.current_stream(device=self.target_device)
        wait_start_event = None
        wait_done_event = None
        wait_started = time.perf_counter()
        if self.cuda_event_timing_enabled:
            wait_start_event = torch.cuda.Event(enable_timing=True)
            wait_done_event = torch.cuda.Event(enable_timing=True)
            wait_start_event.record(stream)
            stream.wait_event(prefetched.event)
            wait_done_event.record(stream)
        else:
            stream.wait_event(prefetched.event)
        wait_host_duration_s = time.perf_counter() - wait_started
        return _MaterializedKVGroup(
            key=prefetched.key,
            value=prefetched.value,
            bytes_copied=prefetched.bytes_copied,
            materialized_count=1,
            mark_read_done=1,
            prefetched=True,
            wait_host_duration_s=wait_host_duration_s,
            staging_slot_index=block_index,
            copy_start_event=prefetched.copy_start_event,
            copy_done_event=prefetched.copy_done_event,
            wait_start_event=wait_start_event,
            wait_done_event=wait_done_event,
        )


class _OnlineSoftmaxState:
    """Online softmax accumulator over K/V blocks.

    State layout follows query-major tensors:

    * max_score: [Q, H]
    * exp_sum: [Q, H]
    * numerator: [Q, H, D]
    """

    def __init__(self, query: Any) -> None:
        torch = _torch()
        q_tokens, q_heads, head_dim = query.shape
        self.max_score = torch.full(
            (q_tokens, q_heads),
            -float("inf"),
            dtype=torch.float32,
            device=query.device,
        )
        self.exp_sum = torch.zeros(
            (q_tokens, q_heads), dtype=torch.float32, device=query.device
        )
        self.numerator = torch.zeros(
            (q_tokens, q_heads, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        self.merge_steps = 0

    def merge(self, partial_max: Any, partial_exp_sum: Any, partial_num: Any) -> None:
        torch = _torch()
        merged_max = torch.maximum(self.max_score, partial_max)
        valid = torch.isfinite(merged_max)
        old_scale = torch.where(
            valid,
            torch.exp(self.max_score - merged_max),
            torch.zeros_like(merged_max),
        )
        new_scale = torch.where(
            valid,
            torch.exp(partial_max - merged_max),
            torch.zeros_like(merged_max),
        )
        self.numerator = (
            self.numerator * old_scale.unsqueeze(-1)
            + partial_num * new_scale.unsqueeze(-1)
        )
        self.exp_sum = self.exp_sum * old_scale + partial_exp_sum * new_scale
        self.max_score = merged_max
        self.merge_steps += 1

    def finalize(self, dtype: Any) -> tuple[Any, Any]:
        torch = _torch()
        has_mass = self.exp_sum > 0
        safe_denominator = torch.where(
            has_mass, self.exp_sum, torch.ones_like(self.exp_sum)
        )
        output = self.numerator / safe_denominator.unsqueeze(-1)
        output = torch.where(
            has_mass.unsqueeze(-1), output, torch.zeros_like(output)
        )
        lse = torch.where(
            has_mass,
            torch.log(safe_denominator) + self.max_score,
            torch.full_like(self.exp_sum, -float("inf")),
        )
        # vLLM merge kernels use [H, Q] LSE layout.
        return output.to(dtype=dtype), lse.transpose(0, 1).contiguous()


class TieredBlockwiseAttentionExecutor:
    """Reference data-plane for ring-like tiered attention.

    Inputs use vLLM's common unpacked attention layout:

    * query: [Q, num_query_heads, head_dim]
    * key/value block: [K, num_kv_heads, head_dim]

    GQA/MQA is supported by repeat-interleaving K/V heads to query heads.
    """

    def __init__(
        self,
        staging_slots: int | None = None,
        staging_buffer: HBMStagingBuffer | None = None,
    ) -> None:
        if staging_slots is None:
            staging_slots = context_ring_staging_slots()
        self.staging_buffer = (
            staging_buffer
            if staging_buffer is not None
            else HBMStagingBuffer(staging_slots)
        )
        self._copy_streams: dict[str, Any] = {}
        self._triton_state_cache: dict[tuple[Any, ...], Any] = {}
        self._isolation_ring_state = None
        self._isolation_state_run_active = False
        self._isolation_ring_state_reuse = _isolation_flag("RING_STATE_REUSE")
        self._isolation_ring_dynamic_kernel = _isolation_flag("RING_DYNAMIC_KERNEL")
        self.isolation_early_prefetch = _isolation_flag("RING_EARLY_PREFETCH")
        self._source_lease_lock = Lock()
        self._pending_source_lease_releases: list[
            _PendingSourceLeaseRelease
        ] = []
        self._source_lease_releases_completed = 0
        self._batched_staging_cursor = 0

    def _reap_source_lease_releases(self) -> int:
        completed = 0
        with self._source_lease_lock:
            pending: list[_PendingSourceLeaseRelease] = []
            for release in self._pending_source_lease_releases:
                try:
                    ready = bool(release.event.query())
                except Exception:
                    ready = False
                if not ready:
                    pending.append(release)
                    continue
                for lease in release.leases:
                    lease.release()
                completed += len(release.leases)
            self._pending_source_lease_releases = pending
            self._source_lease_releases_completed += completed
        return completed

    def flush_source_lease_releases(
        self,
        request_ids: set[str] | None = None,
    ) -> dict[str, int]:
        """Wait for and release selected Context Ring DRAM source leases.

        Normal Ring execution opportunistically reaps completed events at the
        next call. A request that leaves Ring Prefill for native Decode has no
        next Ring call, so its handoff must explicitly close this lifecycle.
        Holding the small bookkeeping lock while synchronizing prevents a
        concurrent executor call from hiding a just-recorded matching event.
        """

        waited_events = 0
        released_leases = 0
        with self._source_lease_lock:
            pending: list[_PendingSourceLeaseRelease] = []
            for release in self._pending_source_lease_releases:
                matches = request_ids is None or bool(
                    release.request_ids.intersection(request_ids)
                )
                if not matches:
                    pending.append(release)
                    continue
                try:
                    release.event.synchronize()
                except Exception:
                    # Keep ownership on failure. Releasing a pool slot before
                    # its CUDA reader has finished would permit data reuse.
                    pending.append(release)
                    continue
                waited_events += 1
                for lease in release.leases:
                    lease.release()
                released_leases += len(release.leases)
            self._pending_source_lease_releases = pending
            self._source_lease_releases_completed += released_leases
            pending_leases = sum(
                len(release.leases)
                for release in pending
                if request_ids is None
                or bool(release.request_ids.intersection(request_ids))
            )
        return {
            "waited_events": waited_events,
            "released_leases": released_leases,
            "pending_leases": pending_leases,
        }

    def _defer_source_lease_releases(
        self,
        leases: tuple[Any, ...],
        target_device: Any,
        request_ids: frozenset[str],
    ) -> int:
        if not leases:
            return 0
        torch = _torch()
        if (
            getattr(target_device, "type", "") == "cuda"
            and bool(torch.cuda.is_available())
        ):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=target_device))
            with self._source_lease_lock:
                self._pending_source_lease_releases.append(
                    _PendingSourceLeaseRelease(
                        event=event,
                        leases=leases,
                        request_ids=request_ids,
                    )
                )
            return len(leases)
        for lease in leases:
            lease.release()
        self._source_lease_releases_completed += len(leases)
        return 0

    @staticmethod
    def _collect_source_leases(
        kv_blocks: list[TieredKVBlock | TieredPagedHBMSegment],
    ) -> tuple[Any, ...]:
        leases: list[Any] = []
        seen: set[int] = set()
        for block in kv_blocks:
            if not isinstance(block, TieredKVBlock):
                continue
            for lease in block.source_leases:
                lease_id = id(lease)
                if lease_id in seen:
                    continue
                seen.add(lease_id)
                leases.append(lease)
        return tuple(leases)

    def _get_copy_stream(self, target_device: Any) -> tuple[Any, bool]:
        device_key = str(target_device)
        copy_stream = self._copy_streams.get(device_key)
        if copy_stream is not None:
            return copy_stream, True
        torch = _torch()
        copy_stream = torch.cuda.Stream(device=target_device)
        self._copy_streams[device_key] = copy_stream
        return copy_stream, False

    @staticmethod
    def _expand_kv_heads(query: Any, key: Any, value: Any) -> tuple[Any, Any]:
        q_heads = int(query.shape[1])
        kv_heads = int(key.shape[1])
        if kv_heads == q_heads:
            return key, value
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"query heads ({q_heads}) must be divisible by KV heads ({kv_heads})"
            )
        repeat = q_heads // kv_heads
        return (
            key.repeat_interleave(repeat, dim=1),
            value.repeat_interleave(repeat, dim=1),
        )

    @staticmethod
    def _partial_attention(
        query: Any,
        key: Any,
        value: Any,
        *,
        query_positions: Any,
        key_start: int,
        causal: bool,
        softmax_scale: float,
    ) -> tuple[Any, Any, Any]:
        torch = _torch()
        key, value = TieredBlockwiseAttentionExecutor._expand_kv_heads(
            query, key, value
        )
        q = query.to(dtype=torch.float32)
        k = key.to(device=query.device, dtype=torch.float32)
        v = value.to(device=query.device, dtype=torch.float32)
        scores = torch.einsum("qhd,khd->qhk", q, k) * softmax_scale

        key_positions = torch.arange(
            key_start,
            key_start + int(key.shape[0]),
            device=query.device,
            dtype=query_positions.dtype,
        )
        if causal:
            mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(1), -float("inf"))

        partial_max = torch.max(scores, dim=-1).values
        valid = torch.isfinite(partial_max)
        safe_max = torch.where(valid, partial_max, torch.zeros_like(partial_max))
        exp_scores = torch.exp(scores - safe_max.unsqueeze(-1))
        exp_scores = torch.where(
            torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores)
        )
        partial_exp_sum = exp_scores.sum(dim=-1)
        partial_num = torch.einsum("qhk,khd->qhd", exp_scores, v)
        return partial_max, partial_exp_sum, partial_num

    def run(
        self,
        query: Any,
        kv_blocks: list[TieredKVBlock | TieredPagedHBMSegment],
        *,
        query_start: int = 0,
        query_positions: Any | None = None,
        causal: bool = True,
        softmax_scale: float | None = None,
        non_blocking_stage: bool = True,
        q_tile_tokens: int | None = None,
        compute_group_tokens_override: int | None = None,
        output_buffer: Any | None = None,
        return_lse: bool = True,
        source_lease_request_ids: tuple[str, ...] = (),
    ) -> TieredBlockwiseAttentionResult:
        if self._isolation_ring_state_reuse and self._isolation_state_run_active:
            raise RuntimeError("concurrent run on a single Ring scratch owner")
        completed = self._reap_source_lease_releases()
        leases = self._collect_source_leases(kv_blocks)
        self._isolation_state_run_active = self._isolation_ring_state_reuse
        try:
            result = self._run_impl(
                query,
                kv_blocks,
                query_start=query_start,
                query_positions=query_positions,
                causal=causal,
                softmax_scale=softmax_scale,
                non_blocking_stage=non_blocking_stage,
                q_tile_tokens=q_tile_tokens,
                compute_group_tokens_override=(
                    compute_group_tokens_override
                ),
                output_buffer=output_buffer,
                return_lse=return_lse,
            )
        except Exception:
            # The error path may already have submitted async copies. Ensure
            # they no longer read provider storage before releasing leases.
            torch = _torch()
            if (
                leases
                and getattr(query.device, "type", "") == "cuda"
                and bool(torch.cuda.is_available())
            ):
                try:
                    torch.cuda.synchronize(query.device)
                except Exception:
                    pass
            for lease in leases:
                lease.release()
            raise
        finally:
            if self._isolation_ring_state is not None:
                self._isolation_ring_state.release()
            self._isolation_state_run_active = False

        deferred = self._defer_source_lease_releases(
            leases,
            query.device,
            frozenset(source_lease_request_ids),
        )
        completed += self._reap_source_lease_releases()
        with self._source_lease_lock:
            pending = sum(
                len(release.leases)
                for release in self._pending_source_lease_releases
            )
        return replace(
            result,
            stats=replace(
                result.stats,
                context_ring_source_leases_acquired=len(leases),
                context_ring_source_lease_releases_deferred=deferred,
                context_ring_source_lease_releases_completed=completed,
                context_ring_source_lease_releases_pending=pending,
            ),
        )

    def run_ragged_batch(
        self,
        query: Any,
        rows: list[TieredRaggedRequestRow],
        *,
        causal: bool = True,
        softmax_scale: float | None = None,
        compute_group_tokens_override: int | None = None,
        output_buffer: Any | None = None,
    ) -> TieredBlockwiseAttentionResult:
        """Run all request rows with one batched partial and one merge launch."""

        completed = self._reap_source_lease_releases()
        all_blocks = [block for row in rows for block in row.kv_blocks]
        leases = self._collect_source_leases(all_blocks)
        try:
            result = self._run_ragged_batch_impl(
                query,
                rows,
                causal=causal,
                softmax_scale=softmax_scale,
                compute_group_tokens_override=compute_group_tokens_override,
                output_buffer=output_buffer,
            )
        except Exception:
            torch = _torch()
            if (
                leases
                and getattr(query.device, "type", "") == "cuda"
                and bool(torch.cuda.is_available())
            ):
                try:
                    torch.cuda.synchronize(query.device)
                except Exception:
                    pass
            for lease in leases:
                lease.release()
            raise
        deferred = self._defer_source_lease_releases(
            leases,
            query.device,
            frozenset(row.request_id for row in rows if row.request_id),
        )
        completed += self._reap_source_lease_releases()
        with self._source_lease_lock:
            pending = sum(
                len(release.leases)
                for release in self._pending_source_lease_releases
            )
        return replace(
            result,
            stats=replace(
                result.stats,
                context_ring_source_leases_acquired=len(leases),
                context_ring_source_lease_releases_deferred=deferred,
                context_ring_source_lease_releases_completed=completed,
                context_ring_source_lease_releases_pending=pending,
            ),
        )

    def _run_ragged_batch_impl(
        self,
        query: Any,
        rows: list[TieredRaggedRequestRow],
        *,
        causal: bool,
        softmax_scale: float | None,
        compute_group_tokens_override: int | None,
        output_buffer: Any | None,
    ) -> TieredBlockwiseAttentionResult:
        started = time.perf_counter()
        torch = _torch()
        if not rows or query.device.type != "cuda":
            raise ValueError("ragged tiered batch requires CUDA request rows")
        if softmax_scale is None:
            softmax_scale = int(query.shape[-1]) ** -0.5
        compute_group_tokens = (
            compute_group_tokens_override
            if compute_group_tokens_override is not None
            and compute_group_tokens_override > 0
            else context_ring_compute_group_tokens()
        )
        compute_group_tokens = max(1, compute_group_tokens)
        staging_before = self.staging_buffer.metrics_snapshot()

        sources_by_group: list[list[tuple[Any, Any]]] = []
        group_key_starts: list[int] = []
        group_query_starts: list[int] = []
        group_query_counts: list[int] = []
        request_query_starts: list[int] = []
        request_query_counts: list[int] = []
        request_group_starts: list[int] = []
        request_group_counts: list[int] = []
        query_position_parts: list[Any] = []
        key_tokens = 0
        hbm_blocks = 0
        dram_blocks = 0
        source_pinned_blocks = 0
        source_unpinned_blocks = 0
        max_group_tokens = 0
        has_gpu_gather = False

        for row in rows:
            query_count = row.query_end - row.query_start
            if query_count <= 0 or row.sequence_length < query_count:
                raise ValueError("invalid ragged request row")
            request_query_starts.append(row.query_start)
            request_query_counts.append(query_count)
            request_group_starts.append(len(sources_by_group))
            query_position_parts.append(
                torch.arange(
                    row.sequence_length - query_count,
                    row.sequence_length,
                    device=query.device,
                    dtype=torch.int32,
                )
            )
            block_index = 0
            row_group_count = 0
            while block_index < len(row.kv_blocks):
                first = row.kv_blocks[block_index]
                group: list[TieredKVBlock | TieredPagedHBMSegment] = [first]
                next_index = block_index + 1
                group_tokens = (
                    first.token_count
                    if isinstance(first, TieredPagedHBMSegment)
                    else int(first.key.shape[0])
                )
                if isinstance(first, TieredKVBlock) and first.tier != "HBM":
                    while next_index < len(row.kv_blocks):
                        candidate = row.kv_blocks[next_index]
                        if (
                            not isinstance(candidate, TieredKVBlock)
                            or candidate.tier == "HBM"
                            or candidate.token_start
                            != first.token_start + group_tokens
                            or group_tokens + int(candidate.key.shape[0])
                            > compute_group_tokens
                        ):
                            break
                        group.append(candidate)
                        group_tokens += int(candidate.key.shape[0])
                        next_index += 1

                sources: list[tuple[Any, Any]] = []
                if isinstance(first, TieredPagedHBMSegment):
                    block_ids = first.block_ids.to(
                        device=query.device,
                        dtype=torch.long,
                    )
                    source_key = first.key_cache.index_select(
                        0, block_ids
                    ).reshape(
                        -1,
                        int(first.key_cache.shape[2]),
                        int(first.key_cache.shape[3]),
                    )[: first.token_count]
                    source_value = first.value_cache.index_select(
                        0, block_ids
                    ).reshape(
                        -1,
                        int(first.value_cache.shape[2]),
                        int(first.value_cache.shape[3]),
                    )[: first.token_count]
                    sources.append((source_key, source_value))
                    hbm_blocks += first.page_count
                    has_gpu_gather = True
                else:
                    for block in group:
                        assert isinstance(block, TieredKVBlock)
                        sources.append((block.key, block.value))
                        if block.tier == "HBM":
                            hbm_blocks += 1
                        else:
                            dram_blocks += 1
                            try:
                                pinned = bool(
                                    block.key.is_pinned()
                                    and block.value.is_pinned()
                                )
                            except RuntimeError:
                                pinned = False
                            if pinned:
                                source_pinned_blocks += 1
                            else:
                                source_unpinned_blocks += 1
                sources_by_group.append(sources)
                group_key_starts.append(first.token_start)
                group_query_starts.append(row.query_start)
                group_query_counts.append(query_count)
                key_tokens += group_tokens
                max_group_tokens = max(max_group_tokens, group_tokens)
                row_group_count += 1
                block_index = next_index
            if row_group_count <= 0 or row_group_count > 128:
                raise ValueError("request group count exceeds ragged merge guard")
            request_group_counts.append(row_group_count)

        group_count = len(sources_by_group)
        max_query_tokens = max(request_query_counts)
        if group_count > 512:
            raise ValueError("ragged batch exceeds 512 staged groups")
        staging_bytes = (
            2
            * group_count
            * max_group_tokens
            * int(query.shape[-1])
            * int(query.element_size())
            * int(sources_by_group[0][0][0].shape[1])
        )
        partial_bytes = (
            group_count
            * max_query_tokens
            * int(query.shape[1])
            * (2 + int(query.shape[-1]))
            * 4
        )
        workspace_bytes = staging_bytes + partial_bytes
        workspace_limit = context_ring_batched_workspace_bytes()
        if workspace_limit > 0 and workspace_bytes > workspace_limit:
            raise RuntimeError(
                "ragged batched workspace exceeds configured limit: "
                f"required={workspace_bytes} limit={workspace_limit}"
            )

        slot_index = self._batched_staging_cursor
        self._batched_staging_cursor = (
            self._batched_staging_cursor + 1
        ) % self.staging_buffer.num_slots
        current_stream = torch.cuda.current_stream(device=query.device)
        source_ready_event = None
        if has_gpu_gather:
            source_ready_event = torch.cuda.Event()
            source_ready_event.record(current_stream)
        copy_stream, copy_stream_reused = self._get_copy_stream(query.device)
        copy_done_event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            if source_ready_event is not None:
                copy_stream.wait_event(source_ready_event)
            staged_keys, staged_values, token_counts, bytes_copied = (
                self.staging_buffer.materialize_group_batch(
                    sources_by_group,
                    target_device=query.device,
                    slot_index=slot_index,
                    non_blocking=True,
                )
            )
            copy_done_event.record(copy_stream)
        current_stream.wait_event(copy_done_event)
        self.staging_buffer.mark_slot_computing(slot_index)

        from vllm.v1.attention.ops.tiered_triton_batched_attention import (
            merge_ragged_batched_tiered_partial_states,
            run_ragged_batched_tiered_partial_attention,
        )

        def device_int(values: list[int]) -> Any:
            return torch.tensor(values, device=query.device, dtype=torch.int32)

        partial_state = run_ragged_batched_tiered_partial_attention(
            query,
            staged_keys,
            staged_values,
            group_key_starts=device_int(group_key_starts),
            group_token_counts=device_int(token_counts),
            group_query_starts=device_int(group_query_starts),
            group_query_counts=device_int(group_query_counts),
            max_query_tokens=max_query_tokens,
            query_positions=torch.cat(query_position_parts),
            causal=causal,
            softmax_scale=softmax_scale,
            block_m=context_ring_triton_partial_block_m(),
            block_n=context_ring_triton_partial_block_n(),
        )
        output = torch.empty_like(query) if output_buffer is None else output_buffer
        merge_ragged_batched_tiered_partial_states(
            partial_state,
            request_query_starts=device_int(request_query_starts),
            request_query_counts=device_int(request_query_counts),
            request_group_starts=device_int(request_group_starts),
            request_group_counts=device_int(request_group_counts),
            max_query_tokens=max_query_tokens,
            output=output,
        )
        self.staging_buffer.mark_slot_read_done(slot_index, query.device)
        staging_after = self.staging_buffer.metrics_snapshot()
        return TieredBlockwiseAttentionResult(
            output=output,
            lse=None,
            stats=TieredBlockwiseAttentionStats(
                query_tokens=int(query.shape[0]),
                key_tokens=key_tokens,
                block_count=sum(len(row.kv_blocks) for row in rows),
                materialized_blocks=dram_blocks,
                hbm_blocks=hbm_blocks,
                dram_blocks=dram_blocks,
                staging_blocks=dram_blocks,
                staging_bytes_copied=bytes_copied,
                online_merge_steps=group_count,
                max_key_tokens_per_block=max_group_tokens,
                context_ring_q_tile_tokens=max_query_tokens,
                context_ring_q_tile_count=len(rows),
                context_ring_compute_group_tokens=compute_group_tokens,
                context_ring_compute_group_count=group_count,
                context_ring_async_prefetch_enabled=1,
                context_ring_staging_slots=self.staging_buffer.num_slots,
                context_ring_prefetch_submits=1,
                context_ring_prefetch_waits=1,
                context_ring_prefetch_bytes=bytes_copied,
                context_ring_staging_non_blocking=1,
                context_ring_staging_materialize_calls=(
                    staging_after.materialize_calls
                    - staging_before.materialize_calls
                ),
                context_ring_staging_slot_waits=(
                    staging_after.slot_waits - staging_before.slot_waits
                ),
                context_ring_staging_mark_read_done_count=(
                    staging_after.mark_read_done_calls
                    - staging_before.mark_read_done_calls
                ),
                context_ring_staging_pending_read_events=(
                    staging_after.pending_read_events
                ),
                context_ring_staging_state_transitions=(
                    staging_after.state_transitions
                    - staging_before.state_transitions
                ),
                context_ring_staging_direct_multicopy_attempts=(
                    staging_after.direct_multicopy_attempts
                    - staging_before.direct_multicopy_attempts
                ),
                context_ring_staging_direct_multicopy_successes=(
                    staging_after.direct_multicopy_successes
                    - staging_before.direct_multicopy_successes
                ),
                context_ring_staging_direct_multicopy_fallbacks=(
                    staging_after.direct_multicopy_fallbacks
                    - staging_before.direct_multicopy_fallbacks
                ),
                context_ring_staging_direct_multicopy_runs=(
                    staging_after.direct_multicopy_runs
                    - staging_before.direct_multicopy_runs
                ),
                context_ring_staging_direct_multicopy_bytes=(
                    staging_after.direct_multicopy_bytes
                    - staging_before.direct_multicopy_bytes
                ),
                context_ring_staging_peak_busy_slots=(
                    staging_after.peak_busy_slots
                ),
                context_ring_source_pinned_blocks=source_pinned_blocks,
                context_ring_source_unpinned_blocks=source_unpinned_blocks,
                context_ring_copy_stream_reused=int(copy_stream_reused),
                context_ring_total_host_duration_s=(
                    time.perf_counter() - started
                ),
                context_ring_output_buffer_direct=int(
                    output_buffer is not None
                ),
                context_ring_lse_skipped=1,
                context_ring_batched_partial_enabled=1,
                context_ring_batched_partial_attempts=1,
                context_ring_batched_partial_successes=1,
                context_ring_batched_partial_stage_a_launches=1,
                context_ring_batched_partial_stage_b_launches=1,
                context_ring_batched_partial_group_count=group_count,
                context_ring_batched_partial_workspace_bytes=(workspace_bytes),
                context_ring_host_group_loop_bypassed=1,
            ),
        )

    def _run_impl(
        self,
        query: Any,
        kv_blocks: list[TieredKVBlock | TieredPagedHBMSegment],
        *,
        query_start: int = 0,
        query_positions: Any | None = None,
        causal: bool = True,
        softmax_scale: float | None = None,
        non_blocking_stage: bool = True,
        q_tile_tokens: int | None = None,
        compute_group_tokens_override: int | None = None,
        output_buffer: Any | None = None,
        return_lse: bool = True,
    ) -> TieredBlockwiseAttentionResult:
        total_started = time.perf_counter()
        torch = _torch()
        staging_metrics_before = self.staging_buffer.metrics_snapshot()
        if len(query.shape) != 3:
            raise ValueError("query must have shape [Q, H, D]")
        query_tokens = int(query.shape[0])
        if query_tokens <= 0:
            raise ValueError("query must contain at least one token")
        if q_tile_tokens is None:
            q_tile_tokens = context_ring_q_tile_tokens()
        if q_tile_tokens <= 0 or q_tile_tokens > query_tokens:
            q_tile_tokens = query_tokens
        q_tile_ranges = [
            (start, min(start + q_tile_tokens, query_tokens))
            for start in range(0, query_tokens, q_tile_tokens)
        ]
        compute_group_tokens = context_ring_compute_group_tokens()
        if (
            compute_group_tokens_override is not None
            and compute_group_tokens_override > 0
        ):
            if compute_group_tokens > 0:
                compute_group_tokens = min(
                    compute_group_tokens,
                    compute_group_tokens_override,
                )
            else:
                compute_group_tokens = compute_group_tokens_override
        compute_grouping_enabled = compute_group_tokens > 0
        streaming_wave_groups = context_ring_streaming_wave_groups()
        cuda_event_timing_enabled = (
            context_ring_cuda_event_timing_enabled()
            and getattr(query.device, "type", "") == "cuda"
            and bool(torch.cuda.is_available())
        )
        async_prefetch_enabled = (
            context_ring_async_prefetch_enabled()
            and _ContextRingPrefetchPipeline.supported(query.device)
            and self.staging_buffer.num_slots >= 2
        )
        triton_partial_enabled = (
            context_ring_triton_partial_enabled()
            and getattr(query.device, "type", "") == "cuda"
            and bool(torch.cuda.is_available())
        )
        triton_fused_update_enabled = (
            context_ring_triton_fused_update_enabled()
            and getattr(query.device, "type", "") == "cuda"
            and bool(torch.cuda.is_available())
        )
        native_reduction_enabled = (
            context_ring_native_reduction_enabled()
            and triton_fused_update_enabled
        )
        paged_hbm_update_enabled = (
            context_ring_paged_hbm_update_enabled()
            and triton_fused_update_enabled
        )
        fully_optimized_strict = context_ring_fully_optimized_strict()
        if (
            streaming_wave_groups > 1
            and context_ring_batched_partial_enabled()
        ):
            raise RuntimeError(
                "streaming Context Ring waves and all-group batched partial "
                "attention are mutually exclusive"
            )
        if streaming_wave_groups > 1:
            if not compute_grouping_enabled:
                raise RuntimeError(
                    "streaming Context Ring waves require positive compute "
                    "group tokens"
                )
            if not async_prefetch_enabled:
                raise RuntimeError(
                    "streaming Context Ring waves require async prefetch with "
                    "at least two HBM staging slots"
                )
            if not triton_fused_update_enabled:
                raise RuntimeError(
                    "streaming Context Ring waves require the fused Triton "
                    "online-softmax update"
                )
        if fully_optimized_strict:
            if not triton_fused_update_enabled:
                raise RuntimeError(
                    "fully optimized context ring requires the fused Triton "
                    "online-softmax update"
                )
            if not paged_hbm_update_enabled:
                raise RuntimeError(
                    "fully optimized context ring requires the paged HBM "
                    "Triton update"
                )
            if not async_prefetch_enabled:
                raise RuntimeError(
                    "fully optimized context ring requires async prefetch "
                    "with at least two HBM staging slots"
                )
            if compute_group_tokens <= 0:
                raise RuntimeError(
                    "fully optimized context ring requires positive compute "
                    "group tokens"
                )
        triton_partial_block_m = context_ring_triton_partial_block_m()
        triton_partial_block_n = context_ring_triton_partial_block_n()
        prefetch_distance = (
            min(
                context_ring_prefetch_distance(),
                max(0, self.staging_buffer.num_slots - 1),
            )
            if async_prefetch_enabled
            else 0
        )
        if not kv_blocks:
            staging_metrics_after = self.staging_buffer.metrics_snapshot()
            output = (
                torch.zeros_like(query)
                if output_buffer is None
                else output_buffer.zero_()
            )
            lse = (
                torch.full(
                    (int(query.shape[1]), int(query.shape[0])),
                    -float("inf"),
                    dtype=torch.float32,
                    device=query.device,
                )
                if return_lse
                else None
            )
            return TieredBlockwiseAttentionResult(
                output=output,
                lse=lse,
                stats=TieredBlockwiseAttentionStats(
                    query_tokens=int(query.shape[0]),
                    key_tokens=0,
                    block_count=0,
                    materialized_blocks=0,
                    hbm_blocks=0,
                    dram_blocks=0,
                    staging_blocks=0,
                    staging_bytes_copied=0,
                    online_merge_steps=0,
                    max_key_tokens_per_block=0,
                    context_ring_q_tile_tokens=q_tile_tokens,
                    context_ring_q_tile_count=len(q_tile_ranges),
                    context_ring_compute_group_tokens=compute_group_tokens,
                    context_ring_compute_group_count=0,
                    context_ring_streaming_wave_groups=(
                        streaming_wave_groups
                    ),
                    context_ring_async_prefetch_enabled=int(
                        async_prefetch_enabled
                    ),
                    context_ring_prefetch_distance=prefetch_distance,
                    context_ring_staging_slots=self.staging_buffer.num_slots,
                    context_ring_staging_non_blocking=int(non_blocking_stage),
                    context_ring_staging_materialize_calls=(
                        staging_metrics_after.materialize_calls
                        - staging_metrics_before.materialize_calls
                    ),
                    context_ring_staging_slot_waits=(
                        staging_metrics_after.slot_waits
                        - staging_metrics_before.slot_waits
                    ),
                    context_ring_staging_mark_read_done_count=(
                        staging_metrics_after.mark_read_done_calls
                        - staging_metrics_before.mark_read_done_calls
                    ),
                    context_ring_staging_pending_read_events=(
                        staging_metrics_after.pending_read_events
                    ),
                    context_ring_cuda_event_timing_enabled=int(
                        cuda_event_timing_enabled
                    ),
                    context_ring_triton_partial_enabled=int(
                        triton_partial_enabled
                    ),
                    context_ring_triton_partial_block_m=(
                        triton_partial_block_m
                    ),
                    context_ring_triton_partial_block_n=(
                        triton_partial_block_n
                    ),
                    context_ring_triton_fused_update_enabled=int(
                        triton_fused_update_enabled
                    ),
                    context_ring_native_reduction_enabled=0,
                    context_ring_output_buffer_direct=int(
                        output_buffer is not None
                    ),
                    context_ring_lse_skipped=int(not return_lse),
                    context_ring_paged_hbm_update_enabled=int(
                        paged_hbm_update_enabled
                    ),
                    context_ring_total_host_duration_s=(
                        time.perf_counter() - total_started
                    ),
                ),
            )

        if query_positions is None:
            query_positions = torch.arange(
                query_start,
                query_start + int(query.shape[0]),
                device=query.device,
                dtype=torch.long,
            )
        else:
            query_positions = query_positions.to(device=query.device)

        if self._isolation_ring_dynamic_kernel and triton_fused_update_enabled:
            # A strided Q must not allocate/repack once per historical group.
            query = query if query.is_contiguous() else query.contiguous()
            query_positions = (
                query_positions if query_positions.is_contiguous()
                else query_positions.contiguous()
            )

        if softmax_scale is None:
            softmax_scale = float(int(query.shape[-1]) ** -0.5)

        states: list[_OnlineSoftmaxState] | None = None

        def ensure_reference_states() -> list[_OnlineSoftmaxState]:
            nonlocal states
            if states is None:
                states = [
                    _OnlineSoftmaxState(query[start:end])
                    for start, end in q_tile_ranges
                ]
            return states
        key_tokens = 0
        materialized_blocks = 0
        hbm_blocks = 0
        dram_blocks = 0
        staging_blocks = 0
        staging_bytes_copied = 0
        max_key_tokens_per_block = 0
        prefetch_submits = 0
        prefetch_waits = 0
        prefetch_bytes = 0
        source_pinned_blocks = 0
        source_unpinned_blocks = 0
        compute_group_count = 0
        prefetch_submit_host_duration_s = 0.0
        prefetch_wait_host_duration_s = 0.0
        materialize_host_duration_s = 0.0
        attention_compute_host_duration_s = 0.0
        triton_partial_attempts = 0
        triton_partial_successes = 0
        triton_partial_fallbacks = 0
        triton_partial_host_duration_s = 0.0
        triton_fused_update_attempts = 0
        triton_fused_update_successes = 0
        triton_fused_update_failures = 0
        triton_fused_update_kernel_launches = 0
        triton_fused_update_host_duration_s = 0.0
        paged_hbm_segments = 0
        paged_hbm_pages = 0
        paged_hbm_kernel_launches = 0
        triton_partial_runner = None
        if triton_partial_enabled:
            try:
                from vllm.v1.attention.ops.tiered_triton_partial_attention import (
                    run_triton_partial_attention,
                )

                triton_partial_runner = run_triton_partial_attention
            except Exception:
                triton_partial_enabled = False
                triton_partial_fallbacks += 1
        triton_fused_update_runner = None
        triton_fused_paged_update_runner = None
        triton_fused_finalize_runner = None
        triton_fused_state = None
        fused_state_reused = False
        if triton_fused_update_enabled:
            try:
                from vllm.v1.attention.ops.tiered_triton_fused_attention import (
                    create_triton_online_softmax_state,
                    finalize_triton_online_softmax_state,
                    update_triton_online_softmax_state,
                    update_triton_online_softmax_state_paged,
                    TritonOnlineSoftmaxState,
                )

                state_cache_key = (
                    str(query.device),
                    int(query.shape[0]),
                    int(query.shape[1]),
                    int(query.shape[2]),
                )
                if self._isolation_ring_state_reuse:
                    if self._isolation_ring_state is None:
                        from vllm.v1.attention.ops.isolation_ring_state import (
                            IsolationRingStateArena,
                        )
                        from vllm.model_executor.layers.pd_prefill_workspace import (
                            active_prefill_workspace_ring_state,
                            prefill_workspace_ring_sharing_requested,
                        )
                        self._isolation_ring_state = IsolationRingStateArena(
                            torch, TritonOnlineSoftmaxState,
                            borrow_factory=(active_prefill_workspace_ring_state
                                            if prefill_workspace_ring_sharing_requested()
                                            else None),
                        )
                    reusable_state, fused_state_reused = (
                        self._isolation_ring_state.acquire(query)
                    )
                else:
                    reusable_state = self._triton_state_cache.get(state_cache_key)
                    fused_state_reused = reusable_state is not None
                triton_fused_state = create_triton_online_softmax_state(
                    query,
                    reusable_state,
                )
                if not self._isolation_ring_state_reuse:
                    self._triton_state_cache[state_cache_key] = triton_fused_state
                triton_fused_update_runner = (
                    update_triton_online_softmax_state
                )
                triton_fused_paged_update_runner = (
                    update_triton_online_softmax_state_paged
                )
                triton_fused_finalize_runner = (
                    finalize_triton_online_softmax_state
                )
            except Exception:
                if (fully_optimized_strict or (
                    self._isolation_ring_state is not None
                    and (self._isolation_ring_state.borrow_failed
                         or self._isolation_ring_state.borrow is not None)
                )):
                    raise
                triton_fused_update_enabled = False
                native_reduction_enabled = False
                paged_hbm_update_enabled = False
                triton_fused_update_failures += 1
        copy_event_pairs: list[tuple[Any, Any]] = []
        wait_event_pairs: list[tuple[Any, Any]] = []
        compute_event_pairs: list[tuple[Any, Any]] = []

        def is_paged_hbm(
            block: TieredKVBlock | TieredPagedHBMSegment,
        ) -> bool:
            return isinstance(block, TieredPagedHBMSegment)

        def needs_materialize(
            block: TieredKVBlock | TieredPagedHBMSegment,
        ) -> bool:
            if is_paged_hbm(block):
                return False
            return block.tier != "HBM" or block.key.device != query.device

        def source_is_pinned(key_tensor: Any, value_tensor: Any) -> bool:
            key_device_type = getattr(
                getattr(key_tensor, "device", None), "type", ""
            )
            value_device_type = getattr(
                getattr(value_tensor, "device", None), "type", ""
            )
            if key_device_type != "cpu" or value_device_type != "cpu":
                return False
            key_is_pinned = getattr(key_tensor, "is_pinned", None)
            value_is_pinned = getattr(value_tensor, "is_pinned", None)
            if key_is_pinned is None or value_is_pinned is None:
                return False
            return bool(key_is_pinned() and value_is_pinned())

        streaming_plan = _build_streaming_compute_plan(
            kv_blocks,
            compute_group_tokens=compute_group_tokens,
            streaming_wave_groups=streaming_wave_groups,
        )
        compute_groups = streaming_plan.groups
        streaming_staging_wave_count = 0
        streaming_max_staging_wave_bytes = 0
        for _, _, group in compute_groups:
            if is_paged_hbm(group[0]) or not any(
                needs_materialize(block) for block in group
            ):
                continue
            streaming_staging_wave_count += 1
            streaming_max_staging_wave_bytes = max(
                streaming_max_staging_wave_bytes,
                sum(
                    HBMStagingBuffer._nbytes(block.key)
                    + HBMStagingBuffer._nbytes(block.value)
                    for block in group
                    if isinstance(block, TieredKVBlock)
                ),
            )
        streaming_staging_workspace_bytes = (
            streaming_max_staging_wave_bytes
            * min(
                self.staging_buffer.num_slots,
                streaming_staging_wave_count,
            )
        )

        batched_partial_enabled = context_ring_batched_partial_enabled()
        batched_partial_attempts = int(batched_partial_enabled)
        batched_partial_fallbacks = 0
        batched_workspace_bytes = 0
        if batched_partial_enabled:
            group_count = len(compute_groups)
            max_group_tokens = max(
                (
                    group[0].token_count
                    if isinstance(group[0], TieredPagedHBMSegment)
                    else sum(int(block.key.shape[0]) for block in group)
                )
                for _, _, group in compute_groups
            )
            staging_workspace_bytes = (
                2
                * group_count
                * max_group_tokens
                * int(kv_blocks[0].key_cache.shape[2]
                      if isinstance(kv_blocks[0], TieredPagedHBMSegment)
                      else kv_blocks[0].key.shape[1])
                * int(query.shape[-1])
                * int(query.element_size())
            )
            partial_workspace_bytes = (
                group_count
                * int(query.shape[0])
                * int(query.shape[1])
                * (2 + int(query.shape[-1]))
                * 4
            )
            batched_workspace_bytes = (
                staging_workspace_bytes + partial_workspace_bytes
            )
            workspace_limit = context_ring_batched_workspace_bytes()
            can_run_batched = (
                query.device.type == "cuda"
                and group_count <= 128
                and group_count * int(query.shape[-1]) <= 16384
                and (
                    workspace_limit <= 0
                    or batched_workspace_bytes <= workspace_limit
                )
            )
            if can_run_batched:
                try:
                    from vllm.v1.attention.ops.tiered_triton_batched_attention import (
                        merge_batched_tiered_partial_states,
                        run_batched_tiered_partial_attention,
                    )

                    sources_by_group: list[list[tuple[Any, Any]]] = []
                    group_key_starts: list[int] = []
                    hbm_page_count = 0
                    dram_page_count = 0
                    source_pinned_count = 0
                    source_unpinned_count = 0
                    for _, _, group in compute_groups:
                        first_block = group[0]
                        group_key_starts.append(first_block.token_start)
                        if isinstance(first_block, TieredPagedHBMSegment):
                            block_ids = first_block.block_ids.to(
                                device=query.device,
                                dtype=torch.long,
                            )
                            staged_key_source = first_block.key_cache.index_select(
                                0, block_ids
                            ).reshape(
                                -1,
                                int(first_block.key_cache.shape[2]),
                                int(first_block.key_cache.shape[3]),
                            )[: first_block.token_count]
                            staged_value_source = (
                                first_block.value_cache.index_select(
                                    0, block_ids
                                ).reshape(
                                    -1,
                                    int(first_block.value_cache.shape[2]),
                                    int(first_block.value_cache.shape[3]),
                                )[: first_block.token_count]
                            )
                            sources_by_group.append([
                                (staged_key_source, staged_value_source)
                            ])
                            hbm_page_count += first_block.page_count
                            continue

                        sources: list[tuple[Any, Any]] = []
                        for grouped_block in group:
                            assert isinstance(grouped_block, TieredKVBlock)
                            sources.append((grouped_block.key, grouped_block.value))
                            if grouped_block.tier == "HBM":
                                hbm_page_count += 1
                            else:
                                dram_page_count += 1
                                if source_is_pinned(
                                    grouped_block.key,
                                    grouped_block.value,
                                ):
                                    source_pinned_count += 1
                                else:
                                    source_unpinned_count += 1
                        sources_by_group.append(sources)

                    copy_stream, copy_stream_reused = self._get_copy_stream(
                        query.device
                    )
                    source_ready_event = torch.cuda.Event()
                    source_ready_event.record(
                        torch.cuda.current_stream(device=query.device)
                    )
                    copy_start_event = None
                    copy_done_event = torch.cuda.Event(
                        enable_timing=cuda_event_timing_enabled
                    )
                    materialize_started = time.perf_counter()
                    slot_index = self._batched_staging_cursor
                    self._batched_staging_cursor = (
                        self._batched_staging_cursor + 1
                    ) % self.staging_buffer.num_slots
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_event(source_ready_event)
                        if cuda_event_timing_enabled:
                            copy_start_event = torch.cuda.Event(
                                enable_timing=True
                            )
                            copy_start_event.record(copy_stream)
                        (
                            staged_keys,
                            staged_values,
                            group_token_counts,
                            bytes_copied,
                        ) = self.staging_buffer.materialize_group_batch(
                            sources_by_group,
                            target_device=query.device,
                            slot_index=slot_index,
                            non_blocking=non_blocking_stage,
                        )
                        copy_done_event.record(copy_stream)
                    current_stream = torch.cuda.current_stream(
                        device=query.device
                    )
                    wait_start_event = None
                    wait_done_event = None
                    if cuda_event_timing_enabled:
                        wait_start_event = torch.cuda.Event(enable_timing=True)
                        wait_done_event = torch.cuda.Event(enable_timing=True)
                        wait_start_event.record(current_stream)
                    current_stream.wait_event(copy_done_event)
                    if wait_done_event is not None:
                        wait_done_event.record(current_stream)
                    self.staging_buffer.mark_slot_computing(slot_index)

                    starts_tensor = torch.tensor(
                        group_key_starts,
                        device=query.device,
                        dtype=torch.int32,
                    )
                    counts_tensor = torch.tensor(
                        group_token_counts,
                        device=query.device,
                        dtype=torch.int32,
                    )
                    compute_start_event = None
                    compute_done_event = None
                    if cuda_event_timing_enabled:
                        compute_start_event = torch.cuda.Event(
                            enable_timing=True
                        )
                        compute_done_event = torch.cuda.Event(
                            enable_timing=True
                        )
                        compute_start_event.record(current_stream)
                    partial_state = run_batched_tiered_partial_attention(
                        query,
                        staged_keys,
                        staged_values,
                        group_key_starts=starts_tensor,
                        group_token_counts=counts_tensor,
                        query_positions=query_positions,
                        causal=causal,
                        softmax_scale=softmax_scale,
                        block_m=triton_partial_block_m,
                        block_n=triton_partial_block_n,
                    )
                    output = (
                        torch.empty_like(query)
                        if output_buffer is None
                        else output_buffer
                    )
                    lse = (
                        torch.empty(
                            (int(query.shape[1]), query_tokens),
                            dtype=torch.float32,
                            device=query.device,
                        )
                        if return_lse
                        else None
                    )
                    merge_batched_tiered_partial_states(
                        partial_state,
                        output=output,
                        lse=lse,
                    )
                    if compute_done_event is not None:
                        compute_done_event.record(current_stream)
                    self.staging_buffer.mark_slot_read_done(
                        slot_index, query.device
                    )
                    staging_metrics_after = self.staging_buffer.metrics_snapshot()

                    copy_ms = 0.0
                    wait_ms = 0.0
                    compute_ms = 0.0
                    if cuda_event_timing_enabled:
                        copy_ms, _ = _sum_cuda_event_elapsed_ms([
                            (copy_start_event, copy_done_event)
                        ])
                        wait_ms, _ = _sum_cuda_event_elapsed_ms([
                            (wait_start_event, wait_done_event)
                        ])
                        compute_ms, _ = _sum_cuda_event_elapsed_ms([
                            (compute_start_event, compute_done_event)
                        ])
                    hidden_ratio = (
                        min(1.0, max(0.0, copy_ms - wait_ms) / copy_ms)
                        if copy_ms > 0.0
                        else 0.0
                    )
                    return TieredBlockwiseAttentionResult(
                        output=output,
                        lse=lse,
                        stats=TieredBlockwiseAttentionStats(
                            query_tokens=int(query.shape[0]),
                            key_tokens=sum(group_token_counts),
                            block_count=len(kv_blocks),
                            materialized_blocks=dram_page_count,
                            hbm_blocks=hbm_page_count,
                            dram_blocks=dram_page_count,
                            staging_blocks=dram_page_count,
                            staging_bytes_copied=bytes_copied,
                            online_merge_steps=group_count,
                            max_key_tokens_per_block=max_group_tokens,
                            context_ring_q_tile_tokens=query_tokens,
                            context_ring_q_tile_count=1,
                            context_ring_compute_group_tokens=(
                                compute_group_tokens
                            ),
                            context_ring_compute_group_count=group_count,
                            context_ring_async_prefetch_enabled=1,
                            context_ring_prefetch_distance=0,
                            context_ring_staging_slots=(
                                self.staging_buffer.num_slots
                            ),
                            context_ring_prefetch_submits=1,
                            context_ring_prefetch_waits=1,
                            context_ring_prefetch_bytes=bytes_copied,
                            context_ring_staging_non_blocking=int(
                                non_blocking_stage
                            ),
                            context_ring_staging_materialize_calls=(
                                staging_metrics_after.materialize_calls
                                - staging_metrics_before.materialize_calls
                            ),
                            context_ring_staging_slot_waits=(
                                staging_metrics_after.slot_waits
                                - staging_metrics_before.slot_waits
                            ),
                            context_ring_staging_mark_read_done_count=(
                                staging_metrics_after.mark_read_done_calls
                                - staging_metrics_before.mark_read_done_calls
                            ),
                            context_ring_staging_pending_read_events=(
                                staging_metrics_after.pending_read_events
                            ),
                            context_ring_staging_state_transitions=(
                                staging_metrics_after.state_transitions
                                - staging_metrics_before.state_transitions
                            ),
                            context_ring_staging_direct_multicopy_attempts=(
                                staging_metrics_after.direct_multicopy_attempts
                                - staging_metrics_before.direct_multicopy_attempts
                            ),
                            context_ring_staging_direct_multicopy_successes=(
                                staging_metrics_after.direct_multicopy_successes
                                - staging_metrics_before.direct_multicopy_successes
                            ),
                            context_ring_staging_direct_multicopy_fallbacks=(
                                staging_metrics_after.direct_multicopy_fallbacks
                                - staging_metrics_before.direct_multicopy_fallbacks
                            ),
                            context_ring_staging_direct_multicopy_runs=(
                                staging_metrics_after.direct_multicopy_runs
                                - staging_metrics_before.direct_multicopy_runs
                            ),
                            context_ring_staging_direct_multicopy_bytes=(
                                staging_metrics_after.direct_multicopy_bytes
                                - staging_metrics_before.direct_multicopy_bytes
                            ),
                            context_ring_staging_peak_busy_slots=(
                                staging_metrics_after.peak_busy_slots
                            ),
                            context_ring_source_pinned_blocks=(
                                source_pinned_count
                            ),
                            context_ring_source_unpinned_blocks=(
                                source_unpinned_count
                            ),
                            context_ring_materialize_host_duration_s=(
                                time.perf_counter() - materialize_started
                            ),
                            context_ring_cuda_event_timing_enabled=int(
                                cuda_event_timing_enabled
                            ),
                            context_ring_copy_event_elapsed_ms=copy_ms,
                            context_ring_wait_event_elapsed_ms=wait_ms,
                            context_ring_compute_event_elapsed_ms=compute_ms,
                            context_ring_copy_compute_overlap_ratio=hidden_ratio,
                            context_ring_copy_event_count=int(copy_ms > 0.0),
                            context_ring_wait_event_count=int(wait_ms > 0.0),
                            context_ring_compute_event_count=int(
                                compute_ms > 0.0
                            ),
                            context_ring_copy_stream_reused=int(
                                copy_stream_reused
                            ),
                            context_ring_output_buffer_direct=int(
                                output_buffer is not None
                            ),
                            context_ring_lse_skipped=int(not return_lse),
                            context_ring_total_host_duration_s=(
                                time.perf_counter() - total_started
                            ),
                            context_ring_batched_partial_enabled=1,
                            context_ring_batched_partial_attempts=1,
                            context_ring_batched_partial_successes=1,
                            context_ring_batched_partial_stage_a_launches=1,
                            context_ring_batched_partial_stage_b_launches=1,
                            context_ring_batched_partial_group_count=(
                                group_count
                            ),
                            context_ring_batched_partial_workspace_bytes=(
                                batched_workspace_bytes
                            ),
                            context_ring_host_group_loop_bypassed=1,
                        ),
                    )
                except Exception:
                    batched_partial_fallbacks = 1
                    if context_ring_batched_partial_strict():
                        raise
            else:
                batched_partial_fallbacks = 1
            if not can_run_batched and context_ring_batched_partial_strict():
                raise RuntimeError(
                    "batched tiered partial attention exceeded its shape or "
                    "workspace guard: "
                    f"groups={group_count} workspace={batched_workspace_bytes} "
                    f"limit={workspace_limit}"
                )

        prefetch_pipeline: _ContextRingPrefetchPipeline | None = None
        copy_stream_reused = False
        prefetched_group_positions: set[int] = set()
        if async_prefetch_enabled and prefetch_distance > 0:
            copy_stream, copy_stream_reused = self._get_copy_stream(
                query.device
            )
            source_ready_event = None
            if self.isolation_early_prefetch:
                source_ready_event = torch.cuda.Event()
                source_ready_event.record(torch.cuda.current_stream(query.device))
            prefetch_pipeline = _ContextRingPrefetchPipeline(
                self.staging_buffer,
                query.device,
                non_blocking=non_blocking_stage,
                cuda_event_timing_enabled=cuda_event_timing_enabled,
                copy_stream=copy_stream,
                source_ready_event=source_ready_event,
            )

        def submit_prefetch_group(group_position: int) -> None:
            nonlocal prefetch_submits, prefetch_bytes
            nonlocal prefetch_submit_host_duration_s
            if prefetch_pipeline is None:
                return
            if group_position < 0 or group_position >= len(compute_groups):
                return
            if group_position in prefetched_group_positions:
                return
            first_block_index, _, group = compute_groups[group_position]
            if is_paged_hbm(group[0]):
                return
            if (
                not compute_grouping_enabled
                and len(group) == 1
                and not needs_materialize(group[0])
            ):
                return
            if compute_grouping_enabled:
                submit_started = time.perf_counter()
                bytes_copied, submitted = prefetch_pipeline.submit_group(
                    first_block_index,
                    group,
                    staging_slot_index=group_position,
                )
                prefetch_submit_host_duration_s += (
                    time.perf_counter() - submit_started
                )
            else:
                submit_started = time.perf_counter()
                bytes_copied = prefetch_pipeline.submit(
                    first_block_index,
                    group[0],
                )
                prefetch_submit_host_duration_s += (
                    time.perf_counter() - submit_started
                )
                submitted = True
            if not submitted:
                return
            prefetched_group_positions.add(group_position)
            prefetch_submits += 1
            prefetch_bytes += bytes_copied

        if prefetch_pipeline is not None:
            for initial_position in range(
                min(len(compute_groups), prefetch_distance + 1)
            ):
                submit_prefetch_group(initial_position)

        def materialize_compute_group(
            group: list[TieredKVBlock | TieredPagedHBMSegment],
            *,
            first_block_index: int,
            group_position: int,
        ) -> _MaterializedKVGroup:
            nonlocal prefetch_waits
            if is_paged_hbm(group[0]):
                raise RuntimeError(
                    "paged HBM segments must bypass the staging materializer"
                )
            if compute_grouping_enabled and prefetch_pipeline is not None:
                materialized = prefetch_pipeline.get_group(
                    first_block_index,
                    group,
                    staging_slot_index=group_position,
                )
                if materialized.prefetched:
                    prefetch_waits += 1
                return materialized
            if len(group) == 1:
                block = group[0]
                key = block.key
                value = block.value
                if not needs_materialize(block):
                    return _MaterializedKVGroup(
                        key=key,
                        value=value,
                        bytes_copied=0,
                        materialized_count=0,
                        mark_read_done=0,
                        prefetched=False,
                        wait_host_duration_s=0.0,
                        staging_slot_index=None,
                    )
                if prefetch_pipeline is not None:
                    materialized = prefetch_pipeline.get(
                        first_block_index, block
                    )
                    if materialized.prefetched:
                        prefetch_waits += 1
                    return materialized
                else:
                    key, value, bytes_copied = (
                        self.staging_buffer.materialize(
                            key,
                            value,
                            target_device=query.device,
                            slot_index=group_position,
                            non_blocking=non_blocking_stage,
                        )
                    )
                return _MaterializedKVGroup(
                    key=key,
                    value=value,
                    bytes_copied=bytes_copied,
                    materialized_count=1,
                    mark_read_done=1,
                    prefetched=False,
                    wait_host_duration_s=0.0,
                    staging_slot_index=group_position,
                )

            group_key, group_value, _ = (
                self.staging_buffer.materialize_group(
                    [(block.key, block.value) for block in group],
                    target_device=query.device,
                    slot_index=group_position,
                    non_blocking=non_blocking_stage,
                )
            )
            bytes_copied = 0
            materialized_count = 0
            for block in group:
                if needs_materialize(block):
                    bytes_copied += (
                        HBMStagingBuffer._nbytes(block.key)
                        + HBMStagingBuffer._nbytes(block.value)
                    )
                    materialized_count += 1
            return _MaterializedKVGroup(
                key=group_key,
                value=group_value,
                bytes_copied=bytes_copied,
                materialized_count=materialized_count,
                mark_read_done=1,
                prefetched=False,
                wait_host_duration_s=0.0,
                staging_slot_index=group_position,
            )

        def run_partial_attention(
            query_tile: Any,
            key_group: Any,
            value_group: Any,
            *,
            query_position_tile: Any,
            key_start: int,
        ) -> tuple[Any, Any, Any]:
            nonlocal triton_partial_runner
            nonlocal triton_partial_attempts, triton_partial_successes
            nonlocal triton_partial_fallbacks, triton_partial_host_duration_s
            if triton_partial_runner is not None:
                triton_partial_attempts += 1
                triton_started = time.perf_counter()
                try:
                    result = triton_partial_runner(
                        query_tile,
                        key_group,
                        value_group,
                        query_positions=query_position_tile,
                        key_start=key_start,
                        causal=causal,
                        softmax_scale=softmax_scale,
                        block_m=triton_partial_block_m,
                        block_n=triton_partial_block_n,
                    )
                    triton_partial_successes += 1
                    triton_partial_host_duration_s += (
                        time.perf_counter() - triton_started
                    )
                    return result
                except Exception:
                    triton_partial_fallbacks += 1
                    triton_partial_runner = None
                    triton_partial_host_duration_s += (
                        time.perf_counter() - triton_started
                    )
            return self._partial_attention(
                query_tile,
                key_group,
                value_group,
                query_positions=query_position_tile,
                key_start=key_start,
                causal=causal,
                softmax_scale=softmax_scale,
            )

        for group_position, (block_index, _next_block_index, group) in enumerate(
            compute_groups
        ):
            compute_group_count += 1
            block = group[0]
            paged_block = (
                block if isinstance(block, TieredPagedHBMSegment) else None
            )
            materialized: _MaterializedKVGroup | None = None
            if paged_block is not None:
                if not paged_hbm_update_enabled:
                    raise RuntimeError(
                        "paged HBM segment requires the fused paged update"
                    )
                if (
                    paged_block.key_cache.dim() != 4
                    or tuple(paged_block.key_cache.shape)
                    != tuple(paged_block.value_cache.shape)
                    or paged_block.block_ids.dim() != 1
                    or paged_block.token_count <= 0
                ):
                    raise ValueError("invalid paged HBM segment")
                key_tokens += paged_block.token_count
                max_key_tokens_per_block = max(
                    max_key_tokens_per_block,
                    paged_block.page_size,
                )
                hbm_blocks += paged_block.page_count
                paged_hbm_segments += 1
                paged_hbm_pages += paged_block.page_count
            else:
                for grouped_block in group:
                    assert isinstance(grouped_block, TieredKVBlock)
                    key = grouped_block.key
                    value = grouped_block.value
                    if len(key.shape) != 3 or len(value.shape) != 3:
                        raise ValueError(
                            "K/V blocks must have shape [K, H, D]"
                        )
                    if tuple(key.shape) != tuple(value.shape):
                        raise ValueError("K/V block shapes must match")
                    if int(key.shape[-1]) != int(query.shape[-1]):
                        raise ValueError(
                            "query and K/V head dimensions must match"
                        )

                    key_tokens += int(key.shape[0])
                    max_key_tokens_per_block = max(
                        max_key_tokens_per_block, int(key.shape[0])
                    )
                    if grouped_block.tier == "HBM":
                        hbm_blocks += 1
                    elif grouped_block.tier == "DRAM":
                        dram_blocks += 1
                    else:
                        staging_blocks += 1

                    if needs_materialize(grouped_block):
                        if source_is_pinned(key, value):
                            source_pinned_blocks += 1
                        else:
                            if fully_optimized_strict:
                                raise RuntimeError(
                                    "fully optimized context ring requires "
                                    "every DRAM source group to use pinned "
                                    "memory"
                                )
                            source_unpinned_blocks += 1

                materialize_started = time.perf_counter()
                materialized = materialize_compute_group(
                    group,
                    first_block_index=block_index,
                    group_position=group_position,
                )
                materialize_host_duration_s += (
                    time.perf_counter() - materialize_started
                )
                key = materialized.key
                value = materialized.value
                prefetch_wait_host_duration_s += (
                    materialized.wait_host_duration_s
                )
                materialized_blocks += materialized.materialized_count
                staging_bytes_copied += materialized.bytes_copied
                if (
                    materialized.copy_start_event is not None
                    and materialized.copy_done_event is not None
                ):
                    copy_event_pairs.append((
                        materialized.copy_start_event,
                        materialized.copy_done_event,
                    ))
                if (
                    materialized.wait_start_event is not None
                    and materialized.wait_done_event is not None
                ):
                    wait_event_pairs.append((
                        materialized.wait_start_event,
                        materialized.wait_done_event,
                    ))
                if materialized.mark_read_done:
                    assert materialized.staging_slot_index is not None
                    self.staging_buffer.mark_slot_computing(
                        materialized.staging_slot_index
                    )

            compute_start_event = None
            compute_done_event = None
            compute_stream = None
            if cuda_event_timing_enabled:
                compute_stream = torch.cuda.current_stream(device=query.device)
                compute_start_event = torch.cuda.Event(enable_timing=True)
                compute_done_event = torch.cuda.Event(enable_timing=True)
                compute_start_event.record(compute_stream)
            fused_group_updated = False
            if (
                triton_fused_update_runner is not None
                and triton_fused_state is not None
            ):
                triton_fused_update_attempts += 1
                fused_started = time.perf_counter()
                try:
                    if paged_block is not None:
                        if triton_fused_paged_update_runner is None:
                            raise RuntimeError(
                                "fused paged HBM update runner is unavailable"
                            )
                        triton_fused_paged_update_runner(
                            query,
                            paged_block.key_cache,
                            paged_block.value_cache,
                            paged_block.block_ids,
                            triton_fused_state,
                            query_positions=query_positions,
                            key_start=paged_block.token_start,
                            key_tokens=paged_block.token_count,
                            causal=causal,
                            softmax_scale=softmax_scale,
                            block_m=triton_partial_block_m,
                            block_n=triton_partial_block_n,
                            native_reduction=native_reduction_enabled,
                        )
                        paged_hbm_kernel_launches += 1
                    else:
                        triton_fused_update_runner(
                            query,
                            key,
                            value,
                            triton_fused_state,
                            query_positions=query_positions,
                            key_start=block.token_start,
                            causal=causal,
                            softmax_scale=softmax_scale,
                            block_m=triton_partial_block_m,
                            block_n=triton_partial_block_n,
                            native_reduction=native_reduction_enabled,
                        )
                except Exception as exc:
                    triton_fused_update_failures += 1
                    if fully_optimized_strict:
                        raise RuntimeError(
                            "fully optimized fused tiered attention failed"
                        ) from exc
                    if triton_fused_update_successes:
                        raise RuntimeError(
                            "fused tiered attention failed after mutating "
                            "the online-softmax state"
                        ) from exc
                    triton_fused_update_runner = None
                    triton_fused_paged_update_runner = None
                    triton_fused_finalize_runner = None
                    triton_fused_state = None
                    triton_fused_update_enabled = False
                    native_reduction_enabled = False
                else:
                    triton_fused_update_successes += 1
                    triton_fused_update_kernel_launches += 1
                    fused_group_updated = True
                finally:
                    fused_duration = time.perf_counter() - fused_started
                    triton_fused_update_host_duration_s += fused_duration
                    attention_compute_host_duration_s += fused_duration

            if not fused_group_updated:
                if paged_block is not None:
                    raise RuntimeError(
                        "paged HBM online-softmax update cannot fall back to "
                        "the object-based reference executor"
                    )
                reference_states = ensure_reference_states()
                for tile_index, (tile_start, tile_end) in enumerate(
                    q_tile_ranges
                ):
                    compute_started = time.perf_counter()
                    partial_max, partial_exp_sum, partial_num = (
                        run_partial_attention(
                            query[tile_start:tile_end],
                            key,
                            value,
                            query_position_tile=query_positions[
                                tile_start:tile_end
                            ],
                            key_start=block.token_start,
                        )
                    )
                    reference_states[tile_index].merge(
                        partial_max,
                        partial_exp_sum,
                        partial_num,
                    )
                    attention_compute_host_duration_s += (
                        time.perf_counter() - compute_started
                    )
            if (
                cuda_event_timing_enabled
                and compute_start_event is not None
                and compute_done_event is not None
                and compute_stream is not None
            ):
                compute_done_event.record(compute_stream)
                compute_event_pairs.append((
                    compute_start_event,
                    compute_done_event,
                ))
            if materialized is not None and materialized.mark_read_done:
                assert materialized.staging_slot_index is not None
                self.staging_buffer.mark_slot_read_done(
                    materialized.staging_slot_index,
                    query.device,
                )
            if prefetch_pipeline is not None:
                submit_prefetch_group(group_position + prefetch_distance + 1)

        if (
            triton_fused_finalize_runner is not None
            and triton_fused_state is not None
            and triton_fused_update_successes == compute_group_count
        ):
            output, lse = triton_fused_finalize_runner(
                query,
                triton_fused_state,
                block_m=triton_partial_block_m,
                output=output_buffer,
                return_lse=return_lse,
            )
            online_merge_steps = triton_fused_update_successes
        else:
            if fully_optimized_strict:
                raise RuntimeError(
                    "fully optimized context ring did not update every "
                    "compute group with the fused Triton kernel"
                )
            reference_states = ensure_reference_states()
            output = (
                torch.empty_like(query)
                if output_buffer is None
                else output_buffer
            )
            lse = (
                torch.empty(
                    (int(query.shape[1]), query_tokens),
                    dtype=torch.float32,
                    device=query.device,
                )
                if return_lse
                else None
            )
            online_merge_steps = 0
            for tile_index, (tile_start, tile_end) in enumerate(q_tile_ranges):
                tile_output, tile_lse = reference_states[tile_index].finalize(
                    query.dtype
                )
                output[tile_start:tile_end] = tile_output
                if lse is not None:
                    lse[:, tile_start:tile_end] = tile_lse
                online_merge_steps += int(
                    reference_states[tile_index].merge_steps
                )

        copy_event_elapsed_ms = 0.0
        wait_event_elapsed_ms = 0.0
        compute_event_elapsed_ms = 0.0
        copy_event_count = 0
        wait_event_count = 0
        compute_event_count = 0
        copy_compute_overlap_ratio = 0.0
        if cuda_event_timing_enabled:
            copy_event_elapsed_ms, copy_event_count = (
                _sum_cuda_event_elapsed_ms(copy_event_pairs)
            )
            wait_event_elapsed_ms, wait_event_count = (
                _sum_cuda_event_elapsed_ms(wait_event_pairs)
            )
            compute_event_elapsed_ms, compute_event_count = (
                _sum_cuda_event_elapsed_ms(compute_event_pairs)
            )
            if copy_event_elapsed_ms > 0:
                hidden_copy_ms = max(
                    0.0, copy_event_elapsed_ms - wait_event_elapsed_ms
                )
                copy_compute_overlap_ratio = min(
                    1.0, hidden_copy_ms / copy_event_elapsed_ms
                )
        staging_metrics_after = self.staging_buffer.metrics_snapshot()

        return TieredBlockwiseAttentionResult(
            output=output,
            lse=lse,
            stats=TieredBlockwiseAttentionStats(
                query_tokens=int(query.shape[0]),
                key_tokens=key_tokens,
                block_count=len(kv_blocks),
                materialized_blocks=materialized_blocks,
                hbm_blocks=hbm_blocks,
                dram_blocks=dram_blocks,
                staging_blocks=staging_blocks,
                staging_bytes_copied=staging_bytes_copied,
                online_merge_steps=online_merge_steps,
                max_key_tokens_per_block=max_key_tokens_per_block,
                context_ring_q_tile_tokens=q_tile_tokens,
                context_ring_q_tile_count=len(q_tile_ranges),
                context_ring_compute_group_tokens=compute_group_tokens,
                context_ring_compute_group_count=compute_group_count,
                context_ring_streaming_wave_groups=streaming_wave_groups,
                context_ring_streaming_wave_count=len(compute_groups),
                context_ring_streaming_base_group_count=(
                    streaming_plan.base_group_count
                ),
                context_ring_streaming_groups_coalesced=max(
                    0,
                    streaming_plan.base_group_count - len(compute_groups),
                ),
                context_ring_streaming_max_wave_tokens=(
                    streaming_plan.max_wave_tokens
                ),
                context_ring_streaming_staging_workspace_bytes=(
                    streaming_staging_workspace_bytes
                ),
                context_ring_async_prefetch_enabled=int(
                    async_prefetch_enabled
                ),
                context_ring_prefetch_distance=prefetch_distance,
                context_ring_staging_slots=self.staging_buffer.num_slots,
                context_ring_prefetch_submits=prefetch_submits,
                context_ring_prefetch_waits=prefetch_waits,
                context_ring_prefetch_bytes=prefetch_bytes,
                context_ring_staging_non_blocking=int(non_blocking_stage),
                context_ring_staging_materialize_calls=(
                    staging_metrics_after.materialize_calls
                    - staging_metrics_before.materialize_calls
                ),
                context_ring_staging_slot_waits=(
                    staging_metrics_after.slot_waits
                    - staging_metrics_before.slot_waits
                ),
                context_ring_staging_mark_read_done_count=(
                    staging_metrics_after.mark_read_done_calls
                    - staging_metrics_before.mark_read_done_calls
                ),
                context_ring_staging_pending_read_events=(
                    staging_metrics_after.pending_read_events
                ),
                context_ring_staging_state_transitions=(
                    staging_metrics_after.state_transitions
                    - staging_metrics_before.state_transitions
                ),
                context_ring_staging_direct_multicopy_attempts=(
                    staging_metrics_after.direct_multicopy_attempts
                    - staging_metrics_before.direct_multicopy_attempts
                ),
                context_ring_staging_direct_multicopy_successes=(
                    staging_metrics_after.direct_multicopy_successes
                    - staging_metrics_before.direct_multicopy_successes
                ),
                context_ring_staging_direct_multicopy_fallbacks=(
                    staging_metrics_after.direct_multicopy_fallbacks
                    - staging_metrics_before.direct_multicopy_fallbacks
                ),
                context_ring_staging_direct_multicopy_runs=(
                    staging_metrics_after.direct_multicopy_runs
                    - staging_metrics_before.direct_multicopy_runs
                ),
                context_ring_staging_direct_multicopy_bytes=(
                    staging_metrics_after.direct_multicopy_bytes
                    - staging_metrics_before.direct_multicopy_bytes
                ),
                context_ring_staging_peak_busy_slots=(
                    staging_metrics_after.peak_busy_slots
                ),
                context_ring_staging_copying_slots=(
                    staging_metrics_after.copying_slots
                ),
                context_ring_staging_computing_slots=(
                    staging_metrics_after.computing_slots
                ),
                context_ring_staging_read_pending_slots=(
                    staging_metrics_after.read_pending_slots
                ),
                context_ring_source_pinned_blocks=source_pinned_blocks,
                context_ring_source_unpinned_blocks=source_unpinned_blocks,
                context_ring_batched_partial_enabled=int(
                    batched_partial_enabled
                ),
                context_ring_batched_partial_attempts=(
                    batched_partial_attempts
                ),
                context_ring_batched_partial_fallbacks=(
                    batched_partial_fallbacks
                ),
                context_ring_batched_partial_workspace_bytes=(
                    batched_workspace_bytes
                ),
                context_ring_prefetch_submit_host_duration_s=(
                    prefetch_submit_host_duration_s
                ),
                context_ring_prefetch_wait_host_duration_s=(
                    prefetch_wait_host_duration_s
                ),
                context_ring_materialize_host_duration_s=(
                    materialize_host_duration_s
                ),
                context_ring_attention_compute_host_duration_s=(
                    attention_compute_host_duration_s
                ),
                context_ring_cuda_event_timing_enabled=int(
                    cuda_event_timing_enabled
                ),
                context_ring_copy_event_elapsed_ms=copy_event_elapsed_ms,
                context_ring_wait_event_elapsed_ms=wait_event_elapsed_ms,
                context_ring_compute_event_elapsed_ms=(
                    compute_event_elapsed_ms
                ),
                context_ring_copy_compute_overlap_ratio=(
                    copy_compute_overlap_ratio
                ),
                context_ring_copy_event_count=copy_event_count,
                context_ring_wait_event_count=wait_event_count,
                context_ring_compute_event_count=compute_event_count,
                context_ring_triton_partial_enabled=int(
                    triton_partial_enabled
                ),
                context_ring_triton_partial_attempts=(
                    triton_partial_attempts
                ),
                context_ring_triton_partial_successes=(
                    triton_partial_successes
                ),
                context_ring_triton_partial_fallbacks=(
                    triton_partial_fallbacks
                ),
                context_ring_triton_partial_block_m=(
                    triton_partial_block_m
                ),
                context_ring_triton_partial_block_n=(
                    triton_partial_block_n
                ),
                context_ring_triton_partial_host_duration_s=(
                    triton_partial_host_duration_s
                ),
                context_ring_triton_fused_update_enabled=int(
                    triton_fused_update_enabled
                ),
                context_ring_native_reduction_enabled=int(
                    native_reduction_enabled
                    and triton_fused_update_successes > 0
                ),
                context_ring_triton_fused_update_attempts=(
                    triton_fused_update_attempts
                ),
                context_ring_triton_fused_update_successes=(
                    triton_fused_update_successes
                ),
                context_ring_triton_fused_update_failures=(
                    triton_fused_update_failures
                ),
                context_ring_triton_fused_update_kernel_launches=(
                    triton_fused_update_kernel_launches
                ),
                context_ring_triton_fused_update_host_duration_s=(
                    triton_fused_update_host_duration_s
                ),
                context_ring_triton_fused_q_tile_loop_bypassed=int(
                    triton_fused_update_successes > 0
                ),
                context_ring_reference_state_allocated=int(states is not None),
                context_ring_output_buffer_direct=int(
                    output_buffer is not None
                ),
                context_ring_lse_skipped=int(not return_lse),
                context_ring_paged_hbm_update_enabled=int(
                    paged_hbm_update_enabled
                ),
                context_ring_paged_hbm_segments=paged_hbm_segments,
                context_ring_paged_hbm_pages=paged_hbm_pages,
                context_ring_paged_hbm_kernel_launches=(
                    paged_hbm_kernel_launches
                ),
                context_ring_copy_stream_reused=int(copy_stream_reused),
                context_ring_fused_state_reused=int(fused_state_reused),
                isolation_ring_state_capacity_tokens=(
                    self._isolation_ring_state.capacity
                    if self._isolation_ring_state is not None else 0
                ),
                isolation_ring_state_bytes=(
                    self._isolation_ring_state.bytes
                    if self._isolation_ring_state is not None else 0
                ),
                isolation_ring_state_shared_bytes=(
                    self._isolation_ring_state.shared_bytes
                    if self._isolation_ring_state is not None else 0
                ),
                isolation_ring_source_ready_waits=(
                    prefetch_pipeline.source_ready_waits
                    if prefetch_pipeline is not None else 0
                ),
                isolation_ring_historical_early_groups=(
                    prefetch_pipeline.historical_early_groups
                    if prefetch_pipeline is not None else 0
                ),
                context_ring_total_host_duration_s=(
                    time.perf_counter() - total_started
                ),
            ),
        )


def run_tiered_blockwise_attention(
    query: Any,
    kv_blocks: list[TieredKVBlock],
    **kwargs: Any,
) -> TieredBlockwiseAttentionResult:
    executor = TieredBlockwiseAttentionExecutor()
    return executor.run(query, kv_blocks, **kwargs)
