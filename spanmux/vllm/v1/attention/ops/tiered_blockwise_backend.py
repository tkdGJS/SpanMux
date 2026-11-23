# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded backend hook for the 0.5.x tiered blockwise executor.

This module is intentionally narrow. It routes only simple single-sequence
TRITON_ATTN batches through the standalone tiered blockwise attention executor.
All other cases fall back to the normal backend unless strict mode is enabled.
"""

from __future__ import annotations

import json
import os
import time
import weakref
from threading import Lock, local
from typing import Any

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.ops.tiered_blockwise_attention import (
    TieredBlockwiseAttentionExecutor,
    TieredKVBlock,
    TieredPagedHBMSegment,
    TieredRaggedRequestRow,
    context_ring_batched_partial_enabled,
    context_ring_compute_group_tokens,
    context_ring_streaming_wave_groups,
    context_ring_streaming_wave_tokens,
    get_shared_hbm_staging_buffer,
    context_ring_paged_hbm_update_enabled,
    context_ring_staging_slots,
)
from vllm.v1.attention.ops.tiered_transfer_priority import (
    acquire_context_ring_priority,
    get_tiered_transfer_priority_stats,
)
from vllm.v1.attention.ops.tiered_residency_provider import (
    TIERED_ATTENTION_PAGE_TIER_DRAM,
    TieredAttentionPagePlanBundle,
    acquire_dram_kv_pool_slots,
    build_dram_kv_page_lookup_plan,
    build_tiered_attention_page_tensor_plan,
    get_dram_kv_contiguous_pool_span,
    get_dram_kv_page_registry_stats,
    tiered_attention_plan_tensors_enabled,
    tiered_dram_page_provider_enabled,
)
from vllm.v1.core.tiered_errors import TieredBackendEligibilityError
from vllm.v1.core.virtual_context import virtual_context_layer_id

_TRUE_VALUES = {"1", "true", "yes", "on"}
_TRACE_LOCK = Lock()
_EXECUTOR_TLS = local()
_EXECUTOR_REGISTRY_LOCK = Lock()
_EXECUTOR_REGISTRY: weakref.WeakSet[TieredBlockwiseAttentionExecutor] = (
    weakref.WeakSet()
)
_RUNTIME_METRICS_TLS = local()


def reset_tiered_context_ring_runtime_metrics() -> None:
    """Reset per-model-forward Context Ring data-plane metrics.

    PD phase workers execute on separate host threads, so thread-local state
    keeps the Prefill measurements isolated without adding a lock to every
    attention layer.
    """

    _RUNTIME_METRICS_TLS.values = {}


def consume_tiered_context_ring_runtime_metrics() -> dict[str, int | float]:
    """Return and reset metrics accumulated by this worker thread."""

    values = dict(getattr(_RUNTIME_METRICS_TLS, "values", {}))
    _RUNTIME_METRICS_TLS.values = {}
    return values


def _record_tiered_context_ring_runtime_metrics(
    stats: Any,
    *,
    backend_host_duration_s: float,
) -> None:
    values = getattr(_RUNTIME_METRICS_TLS, "values", None)
    if values is None:
        values = {}
        _RUNTIME_METRICS_TLS.values = values

    def add(name: str, value: int | float) -> None:
        values[name] = values.get(name, 0) + value

    def keep_max(name: str, value: int | float) -> None:
        values[name] = max(values.get(name, 0), value)

    add("worker_context_ring_calls", 1)
    add(
        "worker_context_ring_backend_host_duration_s",
        backend_host_duration_s,
    )
    keep_max(
        "worker_context_ring_backend_host_duration_max_s",
        backend_host_duration_s,
    )
    add(
        "worker_context_ring_executor_host_duration_s",
        stats.context_ring_total_host_duration_s,
    )
    keep_max(
        "worker_context_ring_executor_host_duration_max_s",
        stats.context_ring_total_host_duration_s,
    )
    add(
        "worker_context_ring_batched_partial_attempts",
        stats.context_ring_batched_partial_attempts,
    )
    add(
        "worker_context_ring_batched_partial_successes",
        stats.context_ring_batched_partial_successes,
    )
    add(
        "worker_context_ring_batched_partial_fallbacks",
        stats.context_ring_batched_partial_fallbacks,
    )
    add(
        "worker_context_ring_host_group_loop_calls",
        int(not stats.context_ring_host_group_loop_bypassed),
    )
    add(
        "worker_context_ring_compute_group_count",
        stats.context_ring_compute_group_count,
    )
    keep_max(
        "worker_context_ring_compute_group_count_max",
        stats.context_ring_compute_group_count,
    )
    add(
        "worker_context_ring_streaming_wave_count",
        stats.context_ring_streaming_wave_count,
    )
    add(
        "worker_context_ring_streaming_base_group_count",
        stats.context_ring_streaming_base_group_count,
    )
    add(
        "worker_context_ring_streaming_groups_coalesced",
        stats.context_ring_streaming_groups_coalesced,
    )
    keep_max(
        "worker_context_ring_streaming_max_wave_tokens",
        stats.context_ring_streaming_max_wave_tokens,
    )
    keep_max(
        "worker_context_ring_streaming_staging_workspace_bytes_max",
        stats.context_ring_streaming_staging_workspace_bytes,
    )
    add(
        "worker_context_ring_batched_partial_stage_a_launches",
        stats.context_ring_batched_partial_stage_a_launches,
    )
    add(
        "worker_context_ring_batched_partial_stage_b_launches",
        stats.context_ring_batched_partial_stage_b_launches,
    )
    keep_max(
        "worker_context_ring_batched_partial_workspace_bytes_max",
        stats.context_ring_batched_partial_workspace_bytes,
    )
    add(
        "worker_context_ring_staging_bytes_copied",
        stats.staging_bytes_copied,
    )


def tiered_blockwise_backend_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND", "").lower()
        in _TRUE_VALUES
    )


def tiered_blockwise_backend_strict() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_STRICT", "").lower()
        in _TRUE_VALUES
    )


def tiered_blockwise_backend_force_all_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL", "")
        .lower()
        in _TRUE_VALUES
    )


def tiered_blockwise_fully_optimized_strict() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def _read_non_negative_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(0, value)


def _trace_path() -> str:
    return os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_TRACE_JSONL", "").strip()


def tiered_blockwise_backend_max_query_tokens() -> int:
    return _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_MAX_QUERY_TOKENS",
        16384,
    )


def tiered_blockwise_backend_max_context_tokens() -> int:
    return _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_MAX_CONTEXT_TOKENS",
        131072,
    )


def tiered_blockwise_backend_max_dram_pages() -> int:
    return _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_MAX_DRAM_PAGES",
        0,
    )


def tiered_blockwise_backend_bulk_copy_tokens() -> int:
    return _read_non_negative_int_env(
        "VLLM_VIRTUAL_CONTEXT_RING_BULK_COPY_TOKENS",
        0,
    )


def _trace_event(event: str, **fields: Any) -> None:
    path = _trace_path()
    if not path:
        return
    payload = {
        "event": event,
        "ts": time.perf_counter(),
        **fields,
    }
    try:
        with _TRACE_LOCK:
            with open(path, "a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        return


def _dram_provider_trace_fields() -> dict[str, Any]:
    if not _trace_path():
        return {}
    from vllm.v1.attention.ops.tiered_layer_reload_prefetch import (
        get_tiered_layer_reload_prefetch_stats,
    )

    return {
        "dram_provider": get_dram_kv_page_registry_stats(),
        "layer_reload_prefetch": (
            get_tiered_layer_reload_prefetch_stats()
        ),
        "transfer_priority": get_tiered_transfer_priority_stats(),
    }


def _get_thread_local_executor(
    *,
    device: torch.device,
    staging_slots: int,
) -> tuple[TieredBlockwiseAttentionExecutor, bool]:
    """Reuse staging allocations without sharing mutable slots across lanes."""

    cache = getattr(_EXECUTOR_TLS, "cache", None)
    if cache is None:
        cache = {}
        _EXECUTOR_TLS.cache = cache
    cache_key = (str(device), int(staging_slots))
    executor = cache.get(cache_key)
    if executor is not None:
        return executor, True
    executor = TieredBlockwiseAttentionExecutor(
        staging_slots=staging_slots,
        staging_buffer=get_shared_hbm_staging_buffer(
            device,
            staging_slots,
        ),
    )
    cache[cache_key] = executor
    with _EXECUTOR_REGISTRY_LOCK:
        _EXECUTOR_REGISTRY.add(executor)
    return executor, False


def flush_tiered_context_ring_source_lease_releases(
    request_ids: list[str] | tuple[str, ...] | set[str] | None,
) -> dict[str, int]:
    """Close final Ring source reads before native Decode H2D reload.

    Executors are thread-local because Prefill and Decode host lanes may run
    concurrently. The registry is only a control-plane index; the executor
    owns and synchronizes its CUDA events and releases only matching requests.
    """

    selected_request_ids = set(request_ids or ())
    if not selected_request_ids:
        return {
            "executor_count": 0,
            "waited_events": 0,
            "released_leases": 0,
            "pending_leases": 0,
        }
    with _EXECUTOR_REGISTRY_LOCK:
        executors = tuple(_EXECUTOR_REGISTRY)
    totals = {
        "executor_count": len(executors),
        "waited_events": 0,
        "released_leases": 0,
        "pending_leases": 0,
    }
    for executor in executors:
        result = executor.flush_source_lease_releases(selected_request_ids)
        for field in ("waited_events", "released_leases", "pending_leases"):
            totals[field] += int(result[field])
    return totals


def _reject(reason: str, *, strict: bool) -> bool:
    _trace_event(
        "tiered_context_ring_reject",
        reason=reason,
        strict=int(strict),
    )
    if strict:
        raise TieredBackendEligibilityError(
            "tiered blockwise backend path is required but ineligible: "
            f"{reason}"
        )
    return False


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _acquire_pool_source_leases(
    slot_versions: list[tuple[int, int]],
    *,
    request_id: str,
    layer_name: str,
) -> tuple[Any, ...]:
    leases = acquire_dram_kv_pool_slots(slot_versions)
    if leases is None:
        raise TieredBackendEligibilityError(
            "DRAM KV source changed before its H2D read lease was acquired: "
            f"request={request_id!r} layer={layer_name!r} "
            f"slots={[slot for slot, _ in slot_versions]}"
        )
    return leases


def _build_hbm_kv_blocks(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    request_id: str,
    layer_name: str,
    page_versions: dict[int, int] | None,
    required_dram_pages: set[int] | None,
    seq_len: int,
    request_numeric_id: int = -1,
    request_key: int | None = None,
    layer_id: int | None = None,
    page_plan: TieredAttentionPagePlanBundle | None = None,
    runtime_bulk_copy_tokens: int = 0,
) -> tuple[
    list[TieredKVBlock | TieredPagedHBMSegment],
    dict[str, Any],
]:
    block_size = int(key_cache.shape[1])
    hbm_page_storage_contiguous = (
        int(key_cache.stride(0))
        == block_size * int(key_cache.stride(1))
        and int(value_cache.stride(0))
        == block_size * int(value_cache.stride(1))
    )
    required_dram_pages = required_dram_pages or set()
    tensor_plan = page_plan.tensor_plan if page_plan is not None else None
    if tensor_plan is not None:
        logical_pages = list(tensor_plan.logical_pages_host)
        page_token_counts = {
            logical_page: tensor_plan.token_counts_host[index]
            for index, logical_page in enumerate(logical_pages)
        }
        hbm_block_ids_by_page = {
            logical_page: tensor_plan.hbm_block_ids_host[index]
            for index, logical_page in enumerate(logical_pages)
        }
        page_tiers_by_page = {
            logical_page: tensor_plan.page_tiers_host[index]
            for index, logical_page in enumerate(logical_pages)
        }
        num_blocks = len(logical_pages)
    else:
        num_blocks = (seq_len + block_size - 1) // block_size
        logical_pages = list(range(num_blocks))
        page_token_counts = {
            logical_page: min(block_size, seq_len - logical_page * block_size)
            for logical_page in logical_pages
        }
        hbm_block_ids_by_page = {}
        page_tiers_by_page = {}
    if page_plan is not None:
        dram_plan = page_plan.dram_plan
    else:
        dram_plan = build_dram_kv_page_lookup_plan(
            request_id=request_id,
            request_key=request_key,
            layer_name=layer_name,
            layer_id=layer_id,
            page_token_counts=page_token_counts,
            page_versions=page_versions,
            strict_pages=required_dram_pages,
        )
    selected_dram_pages = (
        set(required_dram_pages)
        if required_dram_pages
        else set(dram_plan.lookups)
    )

    def selected_dram_lookup(logical_page: int) -> Any | None:
        if logical_page not in selected_dram_pages:
            return None
        return dram_plan.lookups.get(logical_page)

    preacquired_leases = (
        page_plan.pool_leases_by_page
        if page_plan is not None
        else {}
    )
    blocks: list[TieredKVBlock | TieredPagedHBMSegment] = []
    block_ids_by_page: dict[int, int] = {}
    hbm_page_count = 0
    dram_page_count = 0
    bulk_span_count = 0
    bulk_span_page_count = 0
    bulk_span_fallback_count = 0
    hbm_bulk_span_count = 0
    hbm_bulk_span_page_count = 0
    paged_hbm_segment_count = 0
    paged_hbm_page_count = 0
    bulk_copy_tokens = tiered_blockwise_backend_bulk_copy_tokens()
    if runtime_bulk_copy_tokens > 0:
        if bulk_copy_tokens > 0:
            bulk_copy_tokens = min(
                bulk_copy_tokens,
                runtime_bulk_copy_tokens,
            )
        else:
            bulk_copy_tokens = runtime_bulk_copy_tokens
    compute_group_tokens = context_ring_compute_group_tokens()
    if bulk_copy_tokens > 0 and compute_group_tokens > 0:
        bulk_copy_tokens = min(bulk_copy_tokens, compute_group_tokens)
    logical_index = 0
    while logical_index < len(logical_pages):
        logical_page = logical_pages[logical_index]
        token_start = logical_page * block_size
        token_count = page_token_counts[logical_page]
        page_version = (
            page_versions.get(logical_page)
            if page_versions is not None
            else None
        )
        if logical_page in required_dram_pages and page_version is None:
            raise RuntimeError(
                "reload-required logical page is missing a pinned provider "
                "version: "
                f"request_id={request_id!r} layer={layer_name!r} "
                f"logical_page={logical_page}"
            )
        dram_lookup = selected_dram_lookup(logical_page)
        if dram_lookup is not None:
            max_span_pages = (
                bulk_copy_tokens // block_size
                if bulk_copy_tokens >= block_size
                else 0
            )
            if (
                max_span_pages >= 2
                and dram_lookup.source == "pool"
                and token_count == block_size
            ):
                span_pages = [logical_page]
                span_slots = [dram_lookup.slot_id]
                next_index = logical_index + 1
                while (
                    next_index < len(logical_pages)
                    and len(span_pages) < max_span_pages
                ):
                    next_page = logical_pages[next_index]
                    next_lookup = selected_dram_lookup(next_page)
                    next_version = (
                        page_versions.get(next_page)
                        if page_versions is not None
                        else None
                    )
                    if (
                        next_page != span_pages[-1] + 1
                        or page_token_counts[next_page] != block_size
                        or next_lookup is None
                        or next_lookup.source != "pool"
                        or next_lookup.slot_id != span_slots[-1] + 1
                        or (
                            next_page in required_dram_pages
                            and next_version is None
                        )
                    ):
                        break
                    span_pages.append(next_page)
                    span_slots.append(next_lookup.slot_id)
                    next_index += 1
                if len(span_pages) >= 2:
                    span_lookups = [
                        dram_plan.lookups[page] for page in span_pages
                    ]
                    if all(
                        page in preacquired_leases
                        for page in span_pages
                    ):
                        span_leases = tuple(
                            preacquired_leases[page]
                            for page in span_pages
                        )
                    else:
                        span_leases = _acquire_pool_source_leases(
                            [
                                (lookup.slot_id, lookup.version)
                                for lookup in span_lookups
                            ],
                            request_id=request_id,
                            layer_name=layer_name,
                        )
                    span = get_dram_kv_contiguous_pool_span(span_slots)
                    if span is not None:
                        blocks.append(
                            TieredKVBlock(
                                key=span.key,
                                value=span.value,
                                token_start=span.token_start,
                                tier="DRAM",
                                logical_page=logical_page,
                                source_leases=span_leases,
                            )
                        )
                        span_page_count = len(span_pages)
                        dram_page_count += span_page_count
                        bulk_span_count += 1
                        bulk_span_page_count += span_page_count
                        logical_index += span_page_count
                        continue
                    for lease in span_leases:
                        lease.release()
                    bulk_span_fallback_count += 1
            page = dram_lookup.page
            source_leases: tuple[Any, ...] = ()
            if dram_lookup.source == "pool":
                preacquired = preacquired_leases.get(logical_page)
                if preacquired is not None:
                    source_leases = (preacquired,)
                else:
                    source_leases = _acquire_pool_source_leases(
                        [(dram_lookup.slot_id, dram_lookup.version)],
                        request_id=request_id,
                        layer_name=layer_name,
                    )
            blocks.append(
                TieredKVBlock(
                    key=page.key[:token_count],
                    value=page.value[:token_count],
                    token_start=page.token_start,
                    tier="DRAM",
                    logical_page=logical_page,
                    source_leases=source_leases,
                )
            )
            dram_page_count += 1
            logical_index += 1
            continue
        if (
            page_tiers_by_page.get(logical_page)
            == TIERED_ATTENTION_PAGE_TIER_DRAM
        ):
            raise RuntimeError(
                "attention tensor plan marked a DRAM page but the provider "
                "lookup was missing: "
                f"request_id={request_id!r} layer={layer_name!r} "
                f"logical_page={logical_page}"
            )
        if logical_page in required_dram_pages:
            raise RuntimeError(
                "reload-required logical page was not available from the "
                "tiered DRAM provider: "
                f"request_id={request_id!r} layer={layer_name!r} "
                f"logical_page={logical_page}"
            )

        block_id = (
            hbm_block_ids_by_page[logical_page]
            if tensor_plan is not None
            else _as_int(block_table_row[logical_page])
        )
        if block_id < 0 or block_id >= int(key_cache.shape[0]):
            raise RuntimeError(
                f"invalid block id {block_id} at logical page {logical_page}"
            )
        if context_ring_paged_hbm_update_enabled():
            paged_group_tokens = (
                context_ring_streaming_wave_tokens(compute_group_tokens)
                or bulk_copy_tokens
                or compute_group_tokens
                or block_size
            )
            max_paged_pages = max(1, paged_group_tokens // block_size)
            span_pages = [logical_page]
            span_block_ids = [block_id]
            span_token_count = token_count
            next_index = logical_index + 1
            while (
                next_index < len(logical_pages)
                and len(span_pages) < max_paged_pages
            ):
                next_page = logical_pages[next_index]
                if (
                    next_page != span_pages[-1] + 1
                    or next_page in required_dram_pages
                    or selected_dram_lookup(next_page) is not None
                    or page_tiers_by_page.get(next_page)
                    == TIERED_ATTENTION_PAGE_TIER_DRAM
                ):
                    break
                next_block_id = (
                    hbm_block_ids_by_page[next_page]
                    if tensor_plan is not None
                    else _as_int(block_table_row[next_page])
                )
                if (
                    next_block_id < 0
                    or next_block_id >= int(key_cache.shape[0])
                ):
                    break
                span_pages.append(next_page)
                span_block_ids.append(next_block_id)
                span_token_count += page_token_counts[next_page]
                next_index += 1
                if page_token_counts[next_page] < block_size:
                    break
            for page, page_block_id in zip(span_pages, span_block_ids):
                block_ids_by_page[page] = page_block_id
            block_id_tensor = (
                tensor_plan.hbm_block_ids[logical_index:next_index]
                if tensor_plan is not None
                else block_table_row[logical_page:logical_page + len(span_pages)]
            )
            blocks.append(
                TieredPagedHBMSegment(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_ids=block_id_tensor,
                    token_start=token_start,
                    token_count=span_token_count,
                    page_size=block_size,
                    page_count=len(span_pages),
                    logical_page=logical_page,
                )
            )
            hbm_page_count += len(span_pages)
            paged_hbm_segment_count += 1
            paged_hbm_page_count += len(span_pages)
            logical_index = next_index
            continue
        max_hbm_span_pages = (
            bulk_copy_tokens // block_size
            if bulk_copy_tokens >= block_size
            else 0
        )
        if (
            max_hbm_span_pages >= 2
            and token_count == block_size
            and hbm_page_storage_contiguous
        ):
            span_pages = [logical_page]
            span_block_ids = [block_id]
            next_index = logical_index + 1
            while (
                next_index < len(logical_pages)
                and len(span_pages) < max_hbm_span_pages
            ):
                next_page = logical_pages[next_index]
                if (
                    next_page != span_pages[-1] + 1
                    or page_token_counts[next_page] != block_size
                    or next_page in required_dram_pages
                    or selected_dram_lookup(next_page) is not None
                    or page_tiers_by_page.get(next_page)
                    == TIERED_ATTENTION_PAGE_TIER_DRAM
                ):
                    break
                next_block_id = (
                    hbm_block_ids_by_page[next_page]
                    if tensor_plan is not None
                    else _as_int(block_table_row[next_page])
                )
                if next_block_id != span_block_ids[-1] + 1:
                    break
                span_pages.append(next_page)
                span_block_ids.append(next_block_id)
                next_index += 1
            if len(span_pages) >= 2:
                for page, page_block_id in zip(
                    span_pages,
                    span_block_ids,
                ):
                    block_ids_by_page[page] = page_block_id
                span_page_count = len(span_pages)
                token_span = span_page_count * block_size
                blocks.append(
                    TieredKVBlock(
                        key=key_cache[
                            block_id:block_id + span_page_count,
                            :block_size,
                        ].reshape(
                            token_span,
                            int(key_cache.shape[2]),
                            int(key_cache.shape[3]),
                        ),
                        value=value_cache[
                            block_id:block_id + span_page_count,
                            :block_size,
                        ].reshape(
                            token_span,
                            int(value_cache.shape[2]),
                            int(value_cache.shape[3]),
                        ),
                        token_start=token_start,
                        tier="HBM",
                        logical_page=logical_page,
                    )
                )
                hbm_page_count += span_page_count
                hbm_bulk_span_count += 1
                hbm_bulk_span_page_count += span_page_count
                logical_index += span_page_count
                continue

        block_ids_by_page[logical_page] = block_id
        blocks.append(
            TieredKVBlock(
                key=key_cache[block_id, :token_count],
                value=value_cache[block_id, :token_count],
                token_start=token_start,
                tier="HBM",
                logical_page=logical_page,
            )
        )
        hbm_page_count += 1
        logical_index += 1
    missing_required_pages = sorted(
        page for page in required_dram_pages if page < 0 or page >= num_blocks
    )
    if missing_required_pages:
        raise RuntimeError(
            "reload-required logical pages are outside the scheduled sequence: "
            f"request_id={request_id!r} pages={missing_required_pages}"
        )
    plan_trace = {
        "attention_plan_total_pages": num_blocks,
        "attention_plan_hbm_pages": hbm_page_count,
        "attention_plan_dram_pages": dram_page_count,
        "attention_plan_tensor_consumed_by_backend": int(
            tensor_plan is not None
        ),
        "attention_plan_host_view_consumed_by_backend": int(
            tensor_plan is not None
        ),
        "attention_plan_bulk_copy_tokens": bulk_copy_tokens,
        "attention_plan_bulk_span_count": bulk_span_count,
        "attention_plan_bulk_span_pages": bulk_span_page_count,
        "attention_plan_bulk_span_fallback_count": (
            bulk_span_fallback_count
        ),
        "attention_plan_hbm_bulk_span_count": hbm_bulk_span_count,
        "attention_plan_hbm_bulk_span_pages": hbm_bulk_span_page_count,
        "attention_plan_hbm_page_storage_contiguous": int(
            hbm_page_storage_contiguous
        ),
        "attention_plan_paged_hbm_update_enabled": int(
            context_ring_paged_hbm_update_enabled()
        ),
        "attention_plan_paged_hbm_segment_count": (
            paged_hbm_segment_count
        ),
        "attention_plan_paged_hbm_pages": paged_hbm_page_count,
        **(
            page_plan.to_trace_fields()
            if page_plan is not None
            else dram_plan.to_trace_fields()
        ),
    }
    if tiered_attention_plan_tensors_enabled() and page_plan is None:
        tensor_plan = build_tiered_attention_page_tensor_plan(
            request_id=request_id,
            layer_name=layer_name,
            layer_id=dram_plan.layer_id,
            page_token_counts=page_token_counts,
            block_ids_by_page=block_ids_by_page,
            dram_plan=dram_plan,
            required_dram_pages=required_dram_pages,
            block_size=block_size,
            request_numeric_id=request_numeric_id,
            device=block_table_row.device,
        )
        plan_trace.update(tensor_plan.to_trace_fields())
    return blocks, plan_trace


def _required_dram_pages_by_request(
    pages_by_request: dict[str, set[int]] | None,
) -> dict[str, set[int]]:
    if not pages_by_request:
        return {}
    return {
        request_id: set(pages)
        for request_id, pages in pages_by_request.items()
        if pages
    }


def try_run_tiered_blockwise_backend(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata: Any,
    attn_type: AttentionType,
    kv_cache_dtype: str,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    sinks: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    page_versions_by_request: dict[str, dict[int, int]] | None,
    reload_required_pages_by_request: dict[str, set[int]] | None,
    layer_name: str = "",
    reload_to_hbm_native_ready: bool = False,
) -> bool:
    """Try the reference tiered blockwise backend.

    Returns True if the output tensor was written. Returns False when disabled
    or ineligible. This is not a performance path; it is an integration bridge
    for exactness and data-plane validation.

    By default this backend is mandatory only when attention metadata contains
    reload-required DRAM pages. HBM-only calls stay on the optimized upstream
    attention path unless the debug force-all flag is set.
    """

    strict = tiered_blockwise_backend_strict()
    fully_optimized_strict = tiered_blockwise_fully_optimized_strict()
    required_pages_by_request = _required_dram_pages_by_request(
        reload_required_pages_by_request
    )
    forced_by_reload = bool(required_pages_by_request)
    force_all = tiered_blockwise_backend_force_all_enabled()
    if reload_to_hbm_native_ready:
        _trace_event(
            "tiered_context_ring_skip",
            layer_name=layer_name,
            reason="reload_to_hbm_native_ready",
        )
        return False
    _trace_event(
        "tiered_context_ring_consider",
        layer_name=layer_name,
        enabled=int(tiered_blockwise_backend_enabled()),
        strict=int(strict),
        forced_by_reload=int(forced_by_reload),
        force_all=int(force_all),
        required_request_count=len(required_pages_by_request),
        attn_type=str(attn_type),
    )

    if not forced_by_reload and not force_all:
        _trace_event(
            "tiered_context_ring_skip",
            layer_name=layer_name,
            reason="no_reload_required_and_force_all_disabled",
        )
        return False

    if not tiered_blockwise_backend_enabled():
        if forced_by_reload:
            raise TieredBackendEligibilityError(
                "reload-required logical pages cannot fall back to normal HBM "
                "attention. Enable VLLM_VIRTUAL_CONTEXT_RING_BACKEND=1 and "
                "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER=1."
            )
        _trace_event(
            "tiered_context_ring_skip",
            layer_name=layer_name,
            reason="backend_disabled",
        )
        return False
    if forced_by_reload and not tiered_dram_page_provider_enabled():
        raise TieredBackendEligibilityError(
            "reload-required logical pages require the tiered DRAM provider. "
            "Enable VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER=1."
        )
    strict = strict or forced_by_reload

    if attn_type != AttentionType.DECODER:
        return _reject("only decoder attention is supported", strict=strict)
    if kv_cache_dtype.startswith("fp8"):
        return _reject("FP8 KV cache is not supported", strict=strict)
    if alibi_slopes is not None:
        return _reject("ALiBi is not supported", strict=strict)
    if sliding_window != (-1, -1):
        return _reject("sliding-window attention is not supported", strict=strict)
    if logits_soft_cap != 0:
        return _reject("logits soft cap is not supported", strict=strict)
    if sinks is not None:
        return _reject("attention sinks are not supported", strict=strict)
    if output_scale is not None:
        return _reject("fused output scaling is not supported", strict=strict)
    if getattr(attn_metadata, "use_cascade", False):
        return _reject("cascade attention is not supported", strict=strict)
    if getattr(attn_metadata, "mm_prefix_range", None) is not None:
        return _reject("MM prefix ranges are not supported", strict=strict)
    if query.dim() != 3 or output.dim() != 3:
        return _reject("query/output must be [T, H, D]", strict=strict)
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        return _reject("KV cache must be [B, block, H, D]", strict=strict)
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        return _reject("key/value cache shapes differ", strict=strict)
    if int(query.shape[-1]) != int(key_cache.shape[-1]):
        return _reject("query and key head dimensions differ", strict=strict)
    if int(output.shape[-1]) != int(value_cache.shape[-1]):
        return _reject("output and value head dimensions differ", strict=strict)

    query_start_loc = attn_metadata.query_start_loc
    seq_lens = attn_metadata.seq_lens
    query_start_loc_cpu = getattr(
        attn_metadata,
        "query_start_loc_cpu",
        None,
    )
    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    block_table = attn_metadata.block_table
    if int(query_start_loc.numel()) < 2:
        return _reject("query_start_loc must contain at least one sequence", strict=strict)

    request_ids = getattr(attn_metadata, "request_ids", None)
    if forced_by_reload and not request_ids:
        return _reject(
            "reload-required pages require request ids",
            strict=strict,
        )
    padded_num_reqs = int(query_start_loc.numel()) - 1
    # CommonAttentionMetadata can carry padded request rows while request_ids
    # only contains real active requests. Iterate over real requests when that
    # list is available; padded rows have no semantic request id.
    num_reqs = (
        min(len(request_ids), padded_num_reqs)
        if request_ids is not None
        else padded_num_reqs
    )
    if num_reqs <= 0:
        return _reject("no active request rows to run", strict=strict)
    if int(seq_lens.numel()) < num_reqs:
        return _reject("sequence length tensor is shorter than batch", strict=strict)
    if int(block_table.shape[0]) < num_reqs:
        return _reject("block table has fewer rows than batch", strict=strict)
    batch_request_ids = (
        list(request_ids[:num_reqs]) if request_ids is not None else [""] * num_reqs
    )
    request_numeric_ids = getattr(attn_metadata, "request_numeric_ids", None)
    request_numeric_ids_tensor = getattr(
        attn_metadata,
        "request_numeric_ids_tensor",
        None,
    )
    request_provider_keys = getattr(attn_metadata, "request_provider_keys", None)
    request_provider_keys_tensor = getattr(
        attn_metadata,
        "request_provider_keys_tensor",
        None,
    )
    if forced_by_reload:
        missing_request_ids = sorted(
            set(required_pages_by_request) - set(batch_request_ids)
        )
        if missing_request_ids:
            return _reject(
                "reload-required pages do not match the scheduled request ids: "
                f"{missing_request_ids}",
                strict=strict,
            )
    page_plans_by_request = getattr(
        attn_metadata,
        "tiered_attention_page_tensor_plans",
        None,
    )
    page_plans_by_row = getattr(
        attn_metadata,
        "tiered_attention_page_row_plans",
        None,
    )

    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    query_offsets_host = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else query_start_loc
    )
    first_query_offset = _as_int(query_offsets_host[0])
    last_query_offset = _as_int(query_offsets_host[num_reqs])
    if first_query_offset != 0 or last_query_offset > num_actual_tokens:
        return _reject(
            "batched query_start_loc must fit within num_actual_tokens",
            strict=strict,
        )
    if num_actual_tokens != int(query.shape[0]):
        return _reject("query tensor length mismatch", strict=strict)
    if num_actual_tokens != int(output.shape[0]):
        return _reject("output tensor length mismatch", strict=strict)
    if last_query_offset < num_actual_tokens:
        # Padded rows have no request id and no semantic KV ownership. Native
        # backends may still receive padded token slots, so make the reference
        # backend deterministic without pretending those rows are real work.
        output[last_query_offset:num_actual_tokens].zero_()

    max_query_tokens = tiered_blockwise_backend_max_query_tokens()
    max_context_tokens = tiered_blockwise_backend_max_context_tokens()
    max_dram_pages = tiered_blockwise_backend_max_dram_pages()
    block_size = int(key_cache.shape[1])
    admission_selected_tokens = 0
    if (getattr(attn_metadata, "attention_phase", "") or "") == "prefill":
        admission_selected_tokens = int(
            getattr(attn_metadata, "tiered_max_prefill_ring_tokens", 0)
            or 0
        )
    dma_budget_bytes = int(
        getattr(attn_metadata, "tiered_dma_budget_bytes", 0) or 0
    )
    kv_bytes_per_token = (
        2
        * int(key_cache.shape[2])
        * int(key_cache.shape[3])
        * int(key_cache.element_size())
    )
    dma_budget_tokens = (
        dma_budget_bytes // kv_bytes_per_token
        if dma_budget_bytes > 0 and kv_bytes_per_token > 0
        else 0
    )
    staging_budget_blocks = int(
        getattr(attn_metadata, "tiered_staging_budget_blocks", 0) or 0
    )
    staging_budget_tokens = staging_budget_blocks * block_size
    # Q progress/admission and historical-KV traversal are independent axes.
    # Shrinking the scheduled prefill chunk must not turn a 100K context scan
    # into thousands of tiny staging/attention launches. The executor therefore
    # keeps its explicit COMPUTE_GROUP/BULK_COPY setting; scheduler budgets stay
    # visible in trace fields and admission policy only.
    runtime_group_tokens = 0
    executor, executor_cache_hit = _get_thread_local_executor(
        device=query.device,
        staging_slots=context_ring_staging_slots(),
    )
    layer_id = virtual_context_layer_id(layer_name)
    ran_sequences = 0
    batch_started = time.perf_counter()
    total_required_dram_pages = 0
    total_blocks = 0
    ragged_rows: list[TieredRaggedRequestRow] = []
    ragged_row_traces: list[dict[str, Any]] = []
    batched_partial_enabled = context_ring_batched_partial_enabled()
    streaming_wave_groups = context_ring_streaming_wave_groups()
    if batched_partial_enabled and streaming_wave_groups > 1:
        raise TieredBackendEligibilityError(
            "streaming Context Ring waves and all-group batched partial "
            "attention cannot be enabled together"
        )
    use_ragged_batch = batched_partial_enabled and num_reqs > 1
    for req_idx in range(num_reqs):
        request_id = batch_request_ids[req_idx]
        request_numeric_id = -1
        if request_numeric_ids is not None and req_idx < len(request_numeric_ids):
            request_numeric_id = int(request_numeric_ids[req_idx])
        elif (
            request_numeric_ids_tensor is not None
            and req_idx < int(request_numeric_ids_tensor.numel())
        ):
            value = request_numeric_ids_tensor[req_idx]
            request_numeric_id = _as_int(value)
        request_key = None
        if (
            request_provider_keys is not None
            and req_idx < len(request_provider_keys)
        ):
            candidate_key = int(request_provider_keys[req_idx])
            if candidate_key >= 0:
                request_key = candidate_key
        elif (
            request_provider_keys_tensor is not None
            and req_idx < int(request_provider_keys_tensor.numel())
        ):
            value = request_provider_keys_tensor[req_idx]
            candidate_key = _as_int(value)
            if candidate_key >= 0:
                request_key = candidate_key
        q_start = _as_int(query_offsets_host[req_idx])
        q_end = _as_int(query_offsets_host[req_idx + 1])
        q_len = q_end - q_start
        if q_len < 0:
            return _reject("query_start_loc is not monotonic", strict=strict)
        if q_len == 0:
            continue
        if q_end > num_actual_tokens:
            return _reject("query_start_loc exceeds num_actual_tokens", strict=strict)

        seq_len = _as_int(
            seq_lens_cpu[req_idx]
            if seq_lens_cpu is not None
            else seq_lens[req_idx]
        )
        if seq_len < q_len:
            return _reject("sequence length is shorter than query length", strict=strict)
        if max_query_tokens > 0 and q_len > max_query_tokens:
            return _reject(
                "query length exceeds reference context-ring guard: "
                f"request_id={request_id!r} q_len={q_len} max={max_query_tokens}",
                strict=strict,
            )
        if max_context_tokens > 0 and seq_len > max_context_tokens:
            return _reject(
                "sequence length exceeds reference context-ring guard: "
                f"request_id={request_id!r} seq_len={seq_len} "
                f"max={max_context_tokens}",
                strict=strict,
            )

        required_dram_pages = required_pages_by_request.get(request_id, set())
        if max_dram_pages > 0 and len(required_dram_pages) > max_dram_pages:
            return _reject(
                "reload-required DRAM page count exceeds reference "
                "context-ring guard: "
                f"request_id={request_id!r} pages={len(required_dram_pages)} "
                f"max={max_dram_pages}",
                strict=strict,
            )

        num_blocks = (seq_len + block_size - 1) // block_size
        if num_blocks > int(block_table.shape[1]):
            return _reject("block table is shorter than required pages", strict=strict)
        page_versions = (
            page_versions_by_request.get(request_id)
            if page_versions_by_request is not None
            else None
        )
        page_plan = None
        if page_plans_by_row is not None and req_idx < len(page_plans_by_row):
            page_plan = page_plans_by_row[req_idx]
        elif page_plans_by_request is not None:
            page_plan = page_plans_by_request.get(request_id)
        if fully_optimized_strict:
            if page_plan is None or page_plan.tensor_plan is None:
                raise TieredBackendEligibilityError(
                    "fully optimized context ring requires a tensorized "
                    f"attention plan for request {request_id!r}"
                )
            if page_plan.dram_plan.dict_hit_count:
                raise TieredBackendEligibilityError(
                    "fully optimized context ring cannot consume dict-backed "
                    f"DRAM pages for request {request_id!r}"
                )
            if any(
                lookup.source != "pool"
                for lookup in page_plan.dram_plan.lookups.values()
            ):
                raise TieredBackendEligibilityError(
                    "fully optimized context ring requires pool-backed DRAM "
                    f"pages for request {request_id!r}"
                )
            invalid_pool_pages = []
            for logical_page, lookup in page_plan.dram_plan.lookups.items():
                page = lookup.page
                try:
                    pinned = bool(
                        page.key.is_pinned() and page.value.is_pinned()
                    )
                except RuntimeError:
                    pinned = False
                if lookup.slot_id < 0 or not pinned:
                    invalid_pool_pages.append(logical_page)
            if invalid_pool_pages:
                raise TieredBackendEligibilityError(
                    "fully optimized context ring requires pinned, "
                    "slot-addressable DRAM pages; invalid logical pages for "
                    f"request {request_id!r}: {invalid_pool_pages}"
                )
            pool_pages = set(page_plan.dram_plan.lookups)
            missing_lease_pages = sorted(
                pool_pages - set(page_plan.pool_leases_by_page)
            )
            if missing_lease_pages:
                raise TieredBackendEligibilityError(
                    "fully optimized context ring requires pre-acquired "
                    "slot/version read leases; missing logical pages for "
                    f"request {request_id!r}: {missing_lease_pages}"
                )

        trace_started = time.perf_counter()
        _trace_event(
            "tiered_context_ring_submit",
            layer_name=layer_name,
            request_id=request_id,
            request_index=req_idx,
            q_len=q_len,
            seq_len=seq_len,
            num_blocks=num_blocks,
            required_dram_pages=len(required_dram_pages),
            max_query_tokens=max_query_tokens,
            max_context_tokens=max_context_tokens,
            max_dram_pages=max_dram_pages,
        )
        kv_blocks, attention_plan_trace = _build_hbm_kv_blocks(
            key_cache,
            value_cache,
            block_table[req_idx],
            request_id=request_id,
            request_numeric_id=request_numeric_id,
            layer_name=layer_name,
            page_versions=page_versions,
            required_dram_pages=required_dram_pages,
            seq_len=seq_len,
            page_plan=page_plan,
            request_key=request_key,
            layer_id=layer_id,
            runtime_bulk_copy_tokens=runtime_group_tokens,
        )
        _trace_event(
            "tiered_context_ring_attention_plan",
            layer_name=layer_name,
            request_id=request_id,
            request_index=req_idx,
            q_len=q_len,
            seq_len=seq_len,
            **attention_plan_trace,
        )
        if use_ragged_batch:
            ragged_rows.append(
                TieredRaggedRequestRow(
                    query_start=q_start,
                    query_end=q_end,
                    sequence_length=seq_len,
                    kv_blocks=kv_blocks,
                    request_id=request_id,
                )
            )
            ragged_row_traces.append({
                "request_id": request_id,
                "request_index": req_idx,
                "q_len": q_len,
                "seq_len": seq_len,
                "num_blocks": num_blocks,
                "required_dram_pages": len(required_dram_pages),
                "attention_plan": attention_plan_trace,
                "trace_started": trace_started,
            })
        else:
            request_output = output[q_start:q_end]
            with acquire_context_ring_priority():
                result = executor.run(
                    query[q_start:q_end],
                    kv_blocks,
                    query_start=seq_len - q_len,
                    causal=True,
                    softmax_scale=scale,
                    compute_group_tokens_override=runtime_group_tokens,
                    output_buffer=request_output,
                    return_lse=False,
                    source_lease_request_ids=(request_id,),
                )
            backend_host_duration_s = time.perf_counter() - trace_started
            _record_tiered_context_ring_runtime_metrics(
                result.stats,
                backend_host_duration_s=backend_host_duration_s,
            )
            if result.output is not request_output:
                request_output.copy_(result.output)
            _trace_event(
                "tiered_context_ring_done",
                layer_name=layer_name,
                request_id=request_id,
                request_index=req_idx,
                q_len=q_len,
                seq_len=seq_len,
                num_blocks=num_blocks,
                required_dram_pages=len(required_dram_pages),
                elapsed_s=backend_host_duration_s,
                stats=result.stats.to_debug_fields("context_ring"),
                attention_plan=attention_plan_trace,
                **_dram_provider_trace_fields(),
            )
        ran_sequences += 1
        total_required_dram_pages += len(required_dram_pages)
        total_blocks += num_blocks
    ragged_result = None
    if use_ragged_batch and ragged_rows:
        ragged_output = output[:last_query_offset]
        with acquire_context_ring_priority():
            ragged_result = executor.run_ragged_batch(
                query[:last_query_offset],
                ragged_rows,
                causal=True,
                softmax_scale=scale,
                compute_group_tokens_override=runtime_group_tokens,
                output_buffer=ragged_output,
            )
        ragged_backend_host_duration_s = time.perf_counter() - batch_started
        _record_tiered_context_ring_runtime_metrics(
            ragged_result.stats,
            backend_host_duration_s=ragged_backend_host_duration_s,
        )
        if ragged_result.output is not ragged_output:
            ragged_output.copy_(ragged_result.output)
        _trace_event(
            "tiered_context_ring_ragged_done",
            layer_name=layer_name,
            request_count=len(ragged_rows),
            request_rows=ragged_row_traces,
            stats=ragged_result.stats.to_debug_fields("context_ring"),
            elapsed_s=ragged_backend_host_duration_s,
            **_dram_provider_trace_fields(),
        )
    if ran_sequences == 0:
        return _reject("no non-empty query sequences to run", strict=strict)
    _trace_event(
        "tiered_context_ring_batch_done",
        layer_name=layer_name,
        execution_mode=(
            "strict_optimized"
            if fully_optimized_strict
            else "reference_guarded"
        ),
        fully_optimized_strict=int(fully_optimized_strict),
        request_count=ran_sequences,
        request_row_launches=(1 if ragged_result is not None else ran_sequences),
        request_rows_serialized_on_host=int(
            ragged_result is None and ran_sequences > 1
        ),
        request_plan_rows_prepared_on_host=ran_sequences,
        total_blocks=total_blocks,
        total_required_dram_pages=total_required_dram_pages,
        runtime_group_tokens=runtime_group_tokens,
        admission_selected_tokens=admission_selected_tokens,
        dma_budget_bytes=dma_budget_bytes,
        dma_budget_tokens=dma_budget_tokens,
        staging_budget_blocks=staging_budget_blocks,
        staging_budget_tokens=staging_budget_tokens,
        executor_cache_hit=int(executor_cache_hit),
        elapsed_s=time.perf_counter() - batch_started,
    )
    return True
