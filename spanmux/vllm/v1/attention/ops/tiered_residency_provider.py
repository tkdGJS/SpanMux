# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental DRAM page provider for tiered blockwise attention.

This is a deliberately small in-process registry. It lets tests and future
offload integration register CPU-resident K/V pages that the guarded tiered
backend can materialize through its HBM staging buffer.

It is not a production offload manager and is disabled unless explicitly
enabled by environment variable.
"""

from __future__ import annotations

import importlib
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock, local
from typing import Any, Callable

import torch

from vllm.isolation_memory import isolation_execution_enabled
from vllm.logger import init_logger
from vllm.v1.attention.ops.tiered_blockwise_attention import (
    TieredKVBlock,
    context_ring_staging_slots,
    get_shared_hbm_staging_buffer,
    next_shared_hbm_staging_slot,
)
from vllm.v1.attention.ops.tiered_d2h_source_plan import TieredD2HSourcePlan
from vllm.v1.core.tiered_hbm_source_block_lease import (
    TieredHBMSourceBlockLease,
    acquire_tiered_hbm_source_block_lease,
    get_tiered_hbm_source_block_lease_stats,
    tiered_hbm_source_block_lease_enabled,
)
from vllm.v1.core.tiered_errors import TieredBackendEligibilityError
from vllm.v1.core.tiered_required_d2h import required_d2h_progress
from vllm.v1.attention.ops.tiered_transfer_priority import (
    acquire_native_reload_priority,
    get_tiered_transfer_priority_stats,
)
from vllm.v1.core.virtual_context import (
    get_virtual_context_table,
    virtual_context_layer_id,
)
from vllm.v1.engine.pd_engine_progress import notify_pd_engine_progress

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FUSED_KV_SCATTER_MODULE = None
_CUSTOM_OPS_MODULE = None
logger = init_logger(__name__)
TIERED_ATTENTION_PAGE_TIER_HBM = 0
TIERED_ATTENTION_PAGE_TIER_DRAM = 1
TIERED_ATTENTION_PAGE_SOURCE_NONE = 0
TIERED_ATTENTION_PAGE_SOURCE_POOL = 1
TIERED_ATTENTION_PAGE_SOURCE_DICT = 2


def _fused_kv_scatter_module():
    global _FUSED_KV_SCATTER_MODULE
    if _FUSED_KV_SCATTER_MODULE is None:
        _FUSED_KV_SCATTER_MODULE = importlib.import_module(
            "vllm.v1.attention.ops.tiered_triton_kv_scatter"
        )
    return _FUSED_KV_SCATTER_MODULE


def _custom_ops_module():
    global _CUSTOM_OPS_MODULE
    if _CUSTOM_OPS_MODULE is None:
        _CUSTOM_OPS_MODULE = importlib.import_module("vllm._custom_ops")
    return _CUSTOM_OPS_MODULE


def tiered_dram_page_provider_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER", "").lower()
        in _TRUE_VALUES
    )


def tiered_dram_register_from_hbm_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM", "").lower()
        in _TRUE_VALUES
    )


def tiered_dram_register_pressure_only_enabled() -> bool:
    """Return whether native HBM pages are copied only on pressure demand."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_PRESSURE_ONLY",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_register_from_hbm_policy() -> str:
    policy = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM_POLICY",
        "scheduled",
    ).strip().lower()
    if policy not in {"scheduled", "all"}:
        return "scheduled"
    return policy


def tiered_dram_register_max_pages_per_forward() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_MAX_PAGES_PER_FORWARD",
        "1024",
    ).strip()
    if raw_value == "":
        return 1024
    try:
        value = int(raw_value)
    except ValueError:
        return 1024
    return max(0, value)


def tiered_dram_register_async_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC", "").lower()
        in _TRUE_VALUES
    )


def tiered_dram_register_async_workers() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_WORKERS",
        "1",
    ).strip()
    if raw_value == "":
        return 1
    try:
        value = int(raw_value)
    except ValueError:
        return 1
    return max(1, value)


def tiered_dram_register_async_max_pending_pages() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_MAX_PENDING_PAGES",
        "4096",
    ).strip()
    if raw_value == "":
        return 4096
    try:
        value = int(raw_value)
    except ValueError:
        return 4096
    return max(0, value)


def tiered_dram_register_control_dispatcher_enabled() -> bool:
    """Return whether D2H slot reservation/enqueue runs off forward()."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_register_control_dispatcher_workers() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER_WORKERS",
        "1",
    ).strip()
    if raw_value == "":
        return 1
    try:
        value = int(raw_value)
    except ValueError:
        return 1
    return max(1, value)


def tiered_d2h_source_plan_enabled() -> bool:
    """Return whether GPUModelRunner precomputes D2H source pages once."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_PLAN",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_d2h_source_view_dispatch_enabled() -> bool:
    """Return whether concrete HBM views are built in the D2H dispatcher."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_VIEW_DISPATCH",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_provider_pin_memory_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_DRAM_PROVIDER_PIN_MEMORY", "")
        .lower()
        in _TRUE_VALUES
    )


def tiered_dram_page_pool_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL", "").lower()
        in _TRUE_VALUES
    )


def tiered_dram_page_pool_mirror_dict_enabled() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MIRROR_DICT",
            "1",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_page_pool_max_slots() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MAX_SLOTS",
        "0",
    ).strip()
    if raw_value == "":
        return 0
    try:
        value = int(raw_value)
    except ValueError:
        return 0
    return max(0, value)


def tiered_dram_page_pool_contiguous_enabled() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_CONTIGUOUS",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_page_pool_page_tokens() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_PAGE_TOKENS",
        "16",
    ).strip()
    if raw_value == "":
        return 16
    try:
        value = int(raw_value)
    except ValueError:
        return 16
    return max(1, value)


def tiered_dram_page_pool_slab_chunk_slots() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_SLAB_CHUNK_SLOTS",
        "4096",
    ).strip()
    if raw_value == "":
        return 4096
    try:
        value = int(raw_value)
    except ValueError:
        return 4096
    return max(1, value)


def tiered_dram_register_bulk_tokens() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_BULK_TOKENS",
        "16384",
    ).strip()
    if raw_value == "":
        return 16384
    try:
        value = int(raw_value)
    except ValueError:
        return 16384
    return max(0, value)


def tiered_dram_register_direct_multicopy_enabled() -> bool:
    """Batch fragmented native-KV D2H enqueue through one custom-op call."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_DIRECT_MULTICOPY",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_attention_plan_tensors_enabled() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RING_ATTENTION_PLAN_TENSORS",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_reload_to_hbm_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM", "").lower()
        in _TRUE_VALUES
    )


def tiered_dram_reload_bulk_tokens() -> int:
    """Maximum tokens copied by one contiguous DRAM-to-HBM operation.

    Zero preserves the page-wise reference reload path.  The optimized policy
    sets a positive value only when the contiguous pinned page pool is active.
    """

    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_RELOAD_BULK_TOKENS",
        "0",
    ).strip()
    if raw_value == "":
        return 0
    try:
        value = int(raw_value)
    except ValueError:
        return 0
    return max(0, value)


def tiered_dram_reload_staging_scatter_enabled() -> bool:
    """Use bulk H2D staging plus GPU scatter for fragmented HBM blocks."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RELOAD_STAGING_SCATTER",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_reload_direct_multicopy_enabled() -> bool:
    """Copy pinned DRAM spans directly into fragmented native KV blocks."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RELOAD_DIRECT_MULTICOPY",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_direct_multicopy_strict_enabled() -> bool:
    """Fail startup instead of silently benchmarking a Python fallback."""

    return os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_DIRECT_MULTICOPY_STRICT",
        "1",
    ).lower() in _TRUE_VALUES


def validate_tiered_direct_multicopy_ops() -> tuple[str, ...]:
    """Validate the compiled extension against enabled direct-copy paths.

    Python wrappers are shipped separately from ``_C.abi3.so``. Checking the
    wrapper alone therefore cannot prove that the remote worker received the
    matching CUDA extension.
    """

    required_ops: list[str] = []
    if tiered_dram_register_direct_multicopy_enabled():
        required_ops.append("tiered_d2h_multicopy")
    if tiered_dram_reload_direct_multicopy_enabled():
        required_ops.append("tiered_h2d_multicopy")
    if os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY",
        "",
    ).lower() in _TRUE_VALUES:
        required_ops.append("tiered_h2d_staging_multicopy")
    if not required_ops:
        return ()

    # Importing the wrapper loads the extension before checking torch.ops.
    _custom_ops_module()
    namespace = torch.ops._C_cache_ops
    missing = tuple(name for name in required_ops if not hasattr(namespace, name))
    if missing:
        message = (
            "Enabled tiered direct-copy CUDA ops are absent from the loaded "
            f"vllm._C extension: {', '.join(missing)}. Rebuild the extension "
            "from src/vllm-0.14-src and deploy the rebuilt vllm/_C.abi3.so "
            "together with the Python vendor tree."
        )
        if tiered_direct_multicopy_strict_enabled():
            raise RuntimeError(message)
        logger.warning("%s Falling back is allowed by strict=0.", message)
    else:
        logger.info(
            "Validated tiered direct-copy CUDA ops in loaded extension: %s",
            ", ".join(required_ops),
        )
    return missing


def tiered_dram_reload_fused_scatter_enabled() -> bool:
    """Fuse fragmented K/V page scatter into one Triton launch per span."""

    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def tiered_dram_reload_fused_scatter_strict() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER_STRICT",
            "",
        ).lower()
        in _TRUE_VALUES
    )


@dataclass(frozen=True)
class TieredKVPageKey:
    request_key: int
    layer_id: int
    logical_page: int
    version: int
    request_id: str = field(compare=False)
    layer_name: str = field(compare=False)


@dataclass(frozen=True)
class TieredDRAMKVPage:
    key: torch.Tensor
    value: torch.Tensor
    token_start: int
    token_count: int
    version: int


@dataclass(frozen=True)
class TieredDRAMKVLookup:
    page: TieredDRAMKVPage
    slot_id: int
    version: int
    source: str


@dataclass(frozen=True)
class _TieredH2DReloadPage:
    logical_page: int
    block_id: int
    expected_token_count: int
    hbm_token_count: int
    lookup: TieredDRAMKVLookup


def _provider_snapshot_covers_pinned_version(
    *,
    available_version: int,
    available_token_count: int,
    pinned_version: int,
    expected_token_count: int,
) -> bool:
    """Return whether a newer provider page safely replaces a pinned page."""

    return (
        available_version >= pinned_version
        and available_token_count >= expected_token_count
    )


def _coalesce_h2d_source_spans(
    pages: list[_TieredH2DReloadPage],
    *,
    block_size: int,
    max_pages: int,
) -> list[list[_TieredH2DReloadPage]]:
    """Group full pages contiguous in logical and pinned-DRAM space."""

    if not pages:
        return []
    if max_pages < 2:
        return [[page] for page in pages]
    slab_slots = tiered_dram_page_pool_slab_chunk_slots()
    spans: list[list[_TieredH2DReloadPage]] = []
    current: list[_TieredH2DReloadPage] = []
    for page in sorted(pages, key=lambda item: item.logical_page):
        bulk_eligible = (
            page.lookup.source == "pool"
            and page.lookup.slot_id >= 0
            and page.expected_token_count == block_size
        )
        if not current:
            current = [page]
            continue
        previous = current[-1]
        same_slab = (
            page.lookup.slot_id // slab_slots
            == previous.lookup.slot_id // slab_slots
        )
        extends_current = (
            bulk_eligible
            and previous.lookup.source == "pool"
            and previous.expected_token_count == block_size
            and page.logical_page == previous.logical_page + 1
            and page.lookup.slot_id == previous.lookup.slot_id + 1
            and same_slab
            and len(current) < max_pages
        )
        if extends_current:
            current.append(page)
        else:
            spans.append(current)
            current = [page]
    if current:
        spans.append(current)
    return spans


@dataclass(frozen=True)
class TieredDRAMKVLookupPlan:
    request_id: str
    request_key: int
    layer_name: str
    layer_id: int
    logical_pages: tuple[int, ...]
    lookups: dict[int, TieredDRAMKVLookup]
    missing_pages: tuple[int, ...]
    expected_page_count: int
    byte_count: int
    pool_hit_count: int
    dict_hit_count: int

    def to_trace_fields(self) -> dict[str, object]:
        slot_ids = [
            lookup.slot_id
            for lookup in self.lookups.values()
            if lookup.slot_id >= 0
        ]
        versions = [lookup.version for lookup in self.lookups.values()]
        return {
            "dram_plan_expected_pages": self.expected_page_count,
            "dram_plan_request_key": self.request_key,
            "dram_plan_layer_id": self.layer_id,
            "dram_plan_hit_pages": len(self.lookups),
            "dram_plan_missing_pages": len(self.missing_pages),
            "dram_plan_pool_hit_pages": self.pool_hit_count,
            "dram_plan_dict_hit_pages": self.dict_hit_count,
            "dram_plan_byte_count": self.byte_count,
            "dram_plan_slot_id_preview": slot_ids[:16],
            "dram_plan_version_preview": versions[:16],
            "dram_plan_missing_page_preview": list(self.missing_pages[:16]),
        }


@dataclass(frozen=True)
class TieredAttentionPageTensorPlan:
    request_id: str
    request_key: int
    request_numeric_id: int
    layer_name: str
    layer_id: int
    logical_pages: torch.Tensor
    page_tiers: torch.Tensor
    hbm_block_ids: torch.Tensor
    dram_slot_ids: torch.Tensor
    provider_versions: torch.Tensor
    token_starts: torch.Tensor
    token_counts: torch.Tensor
    dram_source_types: torch.Tensor
    logical_pages_host: tuple[int, ...]
    page_tiers_host: tuple[int, ...]
    hbm_block_ids_host: tuple[int, ...]
    dram_slot_ids_host: tuple[int, ...]
    provider_versions_host: tuple[int, ...]
    token_starts_host: tuple[int, ...]
    token_counts_host: tuple[int, ...]
    dram_source_types_host: tuple[int, ...]
    hbm_page_count: int
    dram_page_count: int
    pool_page_count: int
    dict_page_count: int
    missing_required_pages: tuple[int, ...]
    cache_hit: bool = False
    cache_reallocated: bool = False
    cache_delta_pages: int = 0
    cache_copied_pages: int = 0

    def to_trace_fields(self) -> dict[str, object]:
        def preview(values: tuple[int, ...]) -> list[int]:
            return list(values[:16])

        return {
            "attention_plan_tensor_enabled": 1,
            "attention_plan_tensor_device": str(self.logical_pages.device),
            "attention_plan_tensor_request_key": self.request_key,
            "attention_plan_tensor_request_numeric_id": (
                self.request_numeric_id
            ),
            "attention_plan_tensor_layer_id": self.layer_id,
            "attention_plan_tensor_page_count": int(
                self.logical_pages.numel()
            ),
            "attention_plan_tensor_hbm_pages": self.hbm_page_count,
            "attention_plan_tensor_dram_pages": self.dram_page_count,
            "attention_plan_tensor_pool_pages": self.pool_page_count,
            "attention_plan_tensor_dict_pages": self.dict_page_count,
            "attention_plan_tensor_missing_required_pages": len(
                self.missing_required_pages
            ),
            "attention_plan_tensor_cache_hit": int(self.cache_hit),
            "attention_plan_tensor_cache_reallocated": int(
                self.cache_reallocated
            ),
            "attention_plan_tensor_cache_delta_pages": (
                self.cache_delta_pages
            ),
            "attention_plan_tensor_cache_copied_pages": (
                self.cache_copied_pages
            ),
            "attention_plan_tensor_logical_page_preview": preview(
                self.logical_pages_host
            ),
            "attention_plan_tensor_page_tier_preview": preview(
                self.page_tiers_host
            ),
            "attention_plan_tensor_hbm_block_id_preview": preview(
                self.hbm_block_ids_host
            ),
            "attention_plan_tensor_dram_slot_id_preview": preview(
                self.dram_slot_ids_host
            ),
            "attention_plan_tensor_provider_version_preview": preview(
                self.provider_versions_host
            ),
            "attention_plan_tensor_token_count_preview": preview(
                self.token_counts_host
            ),
            "attention_plan_tensor_dram_source_type_preview": preview(
                self.dram_source_types_host
            ),
        }


@dataclass
class _TieredAttentionPageTensorCacheEntry:
    capacity: int
    logical_pages: torch.Tensor
    page_tiers: torch.Tensor
    hbm_block_ids: torch.Tensor
    dram_slot_ids: torch.Tensor
    provider_versions: torch.Tensor
    token_starts: torch.Tensor
    token_counts: torch.Tensor
    dram_source_types: torch.Tensor
    rows_host: tuple[tuple[int, ...], ...]


_ATTENTION_PAGE_TENSOR_CACHE_TLS = local()


def _attention_page_tensor_cache() -> dict[
    tuple[str, int, int], _TieredAttentionPageTensorCacheEntry
]:
    cache = getattr(_ATTENTION_PAGE_TENSOR_CACHE_TLS, "cache", None)
    if cache is None:
        cache = {}
        _ATTENTION_PAGE_TENSOR_CACHE_TLS.cache = cache
    return cache


def _page_plan_capacity(page_count: int) -> int:
    capacity = 16
    while capacity < page_count:
        capacity *= 2
    return capacity


def _allocate_page_plan_tensor(
    capacity: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.empty(capacity, dtype=dtype, device=device)


def _allocate_mutable_cpu_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    pin_memory: bool,
) -> torch.Tensor:
    """Allocate a long-lived CPU buffer with stable mutation semantics.

    The DRAM page pool is initialized lazily. Its first caller may therefore
    be a model thread running under ``torch.inference_mode()`` even though the
    resulting slab is later filled by a background D2H dispatcher. An
    inference tensor cannot be mutated by that dispatcher outside
    InferenceMode. The slab is mutable storage rather than a model output, so
    always create it as a normal tensor independent of the caller context.
    """

    with torch.inference_mode(False):
        tensor = torch.empty(
            shape,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
    if torch.is_inference(tensor):
        raise RuntimeError(
            "mutable tiered-DRAM storage was allocated as an inference tensor"
        )
    return tensor


def _mutable_cpu_tensor_view(
    tensor: torch.Tensor,
    index: Any,
) -> torch.Tensor:
    """Create a mutable slab view independent of the caller's mode.

    ``reserve_from_hbm`` also runs lazily under the model runner's
    InferenceMode. Protecting only the base allocation is insufficient if a
    destination slice created there inherits inference-tensor semantics.
    """

    with torch.inference_mode(False):
        view = tensor[index]
    if torch.is_inference(view):
        raise RuntimeError(
            "mutable tiered-DRAM view was created as an inference tensor"
        )
    return view


def _changed_page_ranges(
    old_rows: tuple[tuple[int, ...], ...],
    new_rows: tuple[tuple[int, ...], ...],
) -> tuple[list[tuple[int, int]], int]:
    changed = [
        index
        for index, row in enumerate(new_rows)
        if index >= len(old_rows) or old_rows[index] != row
    ]
    if not changed:
        return [], 0
    ranges: list[tuple[int, int]] = []
    range_start = changed[0]
    previous = changed[0]
    for index in changed[1:]:
        if index != previous + 1:
            ranges.append((range_start, previous + 1))
            range_start = index
        previous = index
    ranges.append((range_start, previous + 1))
    return ranges, len(changed)


def _cached_page_plan_tensors(
    *,
    cache_key: tuple[str, int, int],
    rows_host: tuple[tuple[int, ...], ...],
    columns: tuple[tuple[int, ...], ...],
    device: torch.device | str,
) -> tuple[tuple[torch.Tensor, ...], bool, bool, int, int]:
    """Return reusable device tensors, copying only changed logical pages."""

    cache = _attention_page_tensor_cache()
    page_count = len(rows_host)
    entry = cache.get(cache_key)
    reallocated = entry is None or entry.capacity < page_count
    if reallocated:
        capacity = _page_plan_capacity(page_count)
        entry = _TieredAttentionPageTensorCacheEntry(
            capacity=capacity,
            logical_pages=_allocate_page_plan_tensor(
                capacity, dtype=torch.int32, device=device
            ),
            page_tiers=_allocate_page_plan_tensor(
                capacity, dtype=torch.int16, device=device
            ),
            hbm_block_ids=_allocate_page_plan_tensor(
                capacity, dtype=torch.int32, device=device
            ),
            dram_slot_ids=_allocate_page_plan_tensor(
                capacity, dtype=torch.int32, device=device
            ),
            provider_versions=_allocate_page_plan_tensor(
                capacity, dtype=torch.int64, device=device
            ),
            token_starts=_allocate_page_plan_tensor(
                capacity, dtype=torch.int32, device=device
            ),
            token_counts=_allocate_page_plan_tensor(
                capacity, dtype=torch.int32, device=device
            ),
            dram_source_types=_allocate_page_plan_tensor(
                capacity, dtype=torch.int16, device=device
            ),
            rows_host=(),
        )
        cache[cache_key] = entry

    assert entry is not None
    changed_ranges, changed_pages = _changed_page_ranges(
        entry.rows_host,
        rows_host,
    )
    tensors = (
        entry.logical_pages,
        entry.page_tiers,
        entry.hbm_block_ids,
        entry.dram_slot_ids,
        entry.provider_versions,
        entry.token_starts,
        entry.token_counts,
        entry.dram_source_types,
    )
    for start, end in changed_ranges:
        for tensor, values in zip(tensors, columns):
            tensor[start:end].copy_(
                torch.as_tensor(
                    values[start:end],
                    dtype=tensor.dtype,
                    device=device,
                )
            )
    entry.rows_host = rows_host
    views = tuple(tensor[:page_count] for tensor in tensors)
    return (
        views,
        not reallocated,
        reallocated,
        changed_pages,
        sum(end - start for start, end in changed_ranges),
    )


@dataclass(frozen=True)
class _PendingHBMRegistrationPage:
    request_id: str
    request_key: int
    layer_name: str
    layer_id: int
    logical_page: int
    key: torch.Tensor
    value: torch.Tensor
    token_start: int
    token_count: int
    version: int
    block_id: int = -1


@dataclass(frozen=True)
class _DeferredHBMRegistrationPage:
    """Source descriptor whose concrete tensor views are built off-forward."""

    request_id: str
    request_key: int
    layer_name: str
    layer_id: int
    logical_page: int
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    token_start: int
    token_count: int
    version: int
    block_id: int

    def materialize(self) -> _PendingHBMRegistrationPage:
        with torch.no_grad():
            key = self.key_cache[self.block_id, : self.token_count].detach()
            value = self.value_cache[self.block_id, : self.token_count].detach()
        return _PendingHBMRegistrationPage(
            request_id=self.request_id,
            request_key=self.request_key,
            layer_name=self.layer_name,
            layer_id=self.layer_id,
            logical_page=self.logical_page,
            key=key,
            value=value,
            token_start=self.token_start,
            token_count=self.token_count,
            version=self.version,
            block_id=self.block_id,
        )


_HBMRegistrationPage = (
    _PendingHBMRegistrationPage | _DeferredHBMRegistrationPage
)


@dataclass(frozen=True)
class TieredAttentionPagePlanBundle:
    dram_plan: TieredDRAMKVLookupPlan
    tensor_plan: TieredAttentionPageTensorPlan | None = None
    pool_leases_by_page: dict[int, object] = field(default_factory=dict)

    def to_trace_fields(self) -> dict[str, object]:
        fields = self.dram_plan.to_trace_fields()
        if self.tensor_plan is not None:
            fields.update(self.tensor_plan.to_trace_fields())
        fields["attention_plan_pool_preacquired_leases"] = len(
            self.pool_leases_by_page
        )
        return fields


@dataclass
class PinnedDRAMKVSlot:
    page_key: TieredKVPageKey
    page: TieredDRAMKVPage
    state: str
    reader_count: int = 0
    retired: bool = False


class TieredDRAMKVLease:
    """Idempotent read lease protecting a pool slot from reuse."""

    def __init__(self, release_callback: Callable[[], None]) -> None:
        self._release_callback = release_callback
        self._lock = Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release_callback()

    def __del__(self) -> None:
        # This is only a leak guard for plans that fail before submitting an
        # H2D copy. The executor releases normal-path leases explicitly after
        # the corresponding CUDA work is ordered.
        try:
            self.release()
        except Exception:
            pass


@dataclass(frozen=True)
class TieredDRAMKVSlotSpan:
    key: torch.Tensor
    value: torch.Tensor
    token_start: int
    token_count: int
    first_slot_id: int
    slot_count: int


@dataclass(frozen=True)
class TieredDRAMKVSlotReservation:
    slot_id: int
    page_key: TieredKVPageKey
    page: TieredDRAMKVPage


@dataclass(frozen=True)
class TieredDirectD2HRegistration:
    """An enqueued GPU-to-pinned-DRAM copy awaiting publication."""

    pages: tuple[_PendingHBMRegistrationPage, ...]
    reservations: tuple[
        tuple[_PendingHBMRegistrationPage, TieredDRAMKVSlotReservation], ...
    ]
    copy_complete: torch.cuda.Event
    started_at: float
    copy_ops: int
    bulk_copy_ops: int
    bulk_pages: int
    direct_multicopy_ops: int
    direct_multicopy_runs: int
    direct_multicopy_pages: int
    direct_multicopy_bytes: int


def _contiguous_tensor_span(
    tensors: list[torch.Tensor],
) -> torch.Tensor | None:
    if not tensors:
        return None
    first = tensors[0]
    if not first.is_contiguous():
        return None
    expected_offset = int(first.storage_offset())
    total_tokens = 0
    storage_ptr = first.untyped_storage().data_ptr()
    for tensor in tensors:
        if (
            not tensor.is_contiguous()
            or tensor.dtype != first.dtype
            or tuple(tensor.shape[1:]) != tuple(first.shape[1:])
            or tensor.untyped_storage().data_ptr() != storage_ptr
            or int(tensor.storage_offset()) != expected_offset
        ):
            return None
        total_tokens += int(tensor.shape[0])
        expected_offset += int(tensor.numel())
    return first.as_strided(
        (total_tokens, *first.shape[1:]),
        first.stride(),
        int(first.storage_offset()),
    )


class PinnedDRAMKVPool:
    """Slot-backed DRAM page pool for provider-backed context ring pages.

    Pages use growable contiguous pinned slabs. A slot lease prevents a newer
    D2H snapshot from reusing slab storage while an async H2D reader still
    references the previous snapshot.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._slots: list[PinnedDRAMKVSlot | None] = []
        self._free_slots: list[int] = []
        self._slot_by_key: dict[TieredKVPageKey, int] = {}
        self._latest_slot: dict[tuple[int, int, int], int] = {}
        self._put_count = 0
        self._put_fail_count = 0
        self._overwrite_count = 0
        self._lookup_count = 0
        self._hit_count = 0
        self._miss_count = 0
        self._latest_lookup_count = 0
        self._pinned_page_count = 0
        self._byte_count = 0
        self._active_slots = 0
        self._peak_slots = 0
        self._key_slabs: list[torch.Tensor] = []
        self._value_slabs: list[torch.Tensor] = []
        self._slab_chunk_slots = 0
        self._slab_page_tokens = 0
        self._slab_kv_heads = 0
        self._slab_head_dim = 0
        self._slab_init_count = 0
        self._slab_init_fail_count = 0
        self._slab_store_count = 0
        self._slab_store_fallback_count = 0
        self._slab_span_lookup_count = 0
        self._slab_span_hit_count = 0
        self._slab_span_miss_count = 0
        self._slab_span_page_count = 0
        self._direct_reserve_count = 0
        self._direct_reserve_fail_count = 0
        self._direct_publish_count = 0
        self._direct_cancel_count = 0
        self._lease_acquire_count = 0
        self._lease_release_count = 0
        self._lease_acquire_fail_count = 0
        self._retired_slot_count = 0

    def _free_slot_locked(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        if slot is not None:
            if slot.reader_count:
                raise RuntimeError("cannot free a leased DRAM KV pool slot")
            if slot.state == "READY":
                self._active_slots = max(0, self._active_slots - 1)
            if slot.state in {"READY", "RETIRED"}:
                self._byte_count = max(
                    0,
                    self._byte_count - self._page_bytes(slot.page),
                )
                if self._page_is_pinned(slot.page):
                    self._pinned_page_count = max(
                        0,
                        self._pinned_page_count - 1,
                    )
            if slot.retired:
                self._retired_slot_count = max(
                    0,
                    self._retired_slot_count - 1,
                )
        self._slots[slot_id] = None
        if slot_id not in self._free_slots:
            self._free_slots.append(slot_id)

    def _release_slot_lease(self, slot_id: int) -> None:
        with self._lock:
            if slot_id < 0 or slot_id >= len(self._slots):
                return
            slot = self._slots[slot_id]
            if slot is None or slot.reader_count <= 0:
                return
            slot.reader_count -= 1
            self._lease_release_count += 1
            if slot.reader_count == 0 and slot.retired:
                self._free_slot_locked(slot_id)

    def acquire_slots(
        self,
        slot_versions: list[tuple[int, int]],
    ) -> tuple[TieredDRAMKVLease, ...] | None:
        """Atomically pin READY slots while an async H2D read is pending."""

        if not slot_versions:
            return ()
        with self._lock:
            slots: list[PinnedDRAMKVSlot] = []
            for slot_id, version in slot_versions:
                if slot_id < 0 or slot_id >= len(self._slots):
                    self._lease_acquire_fail_count += 1
                    return None
                slot = self._slots[slot_id]
                if (
                    slot is None
                    or slot.state not in {"READY", "RETIRED"}
                    or slot.page_key.version != version
                ):
                    self._lease_acquire_fail_count += 1
                    return None
                slots.append(slot)
            for slot in slots:
                slot.reader_count += 1
            self._lease_acquire_count += len(slots)
        return tuple(
            TieredDRAMKVLease(
                lambda slot_id=slot_id: self._release_slot_lease(slot_id)
            )
            for slot_id, _ in slot_versions
        )

    @staticmethod
    def _page_bytes(page: TieredDRAMKVPage) -> int:
        return (
            page.key.numel() * page.key.element_size()
            + page.value.numel() * page.value.element_size()
        )

    @staticmethod
    def _page_is_pinned(page: TieredDRAMKVPage) -> bool:
        try:
            return bool(page.key.is_pinned() and page.value.is_pinned())
        except RuntimeError:
            return False

    def _ensure_contiguous_slab_locked(
        self,
        page: TieredDRAMKVPage,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        if not tiered_dram_page_pool_contiguous_enabled():
            return None
        if self._key_slabs and self._value_slabs:
            compatible = (
                page.key.dtype == self._key_slabs[0].dtype
                and page.value.dtype == self._value_slabs[0].dtype
                and int(page.key.shape[1]) == self._slab_kv_heads
                and int(page.key.shape[2]) == self._slab_head_dim
                and int(page.token_count) <= self._slab_page_tokens
            )
            if not compatible:
                return None

        max_slots = tiered_dram_page_pool_max_slots()
        if max_slots > 0 and slot_id >= max_slots:
            self._slab_init_fail_count += 1
            return None
        page_tokens = tiered_dram_page_pool_page_tokens()
        if int(page.token_count) > page_tokens:
            self._slab_init_fail_count += 1
            return None
        kv_heads = int(page.key.shape[1])
        head_dim = int(page.key.shape[2])
        if not self._key_slabs:
            self._slab_page_tokens = page_tokens
            self._slab_kv_heads = kv_heads
            self._slab_head_dim = head_dim
            self._slab_chunk_slots = (
                tiered_dram_page_pool_slab_chunk_slots()
            )
        chunk_index = slot_id // self._slab_chunk_slots
        local_slot = slot_id % self._slab_chunk_slots
        pin_memory = bool(
            tiered_dram_provider_pin_memory_enabled()
            and torch.cuda.is_available()
        )
        while len(self._key_slabs) <= chunk_index:
            chunk_start = len(self._key_slabs) * self._slab_chunk_slots
            chunk_slots = self._slab_chunk_slots
            if max_slots > 0:
                chunk_slots = min(chunk_slots, max_slots - chunk_start)
            if chunk_slots <= 0:
                self._slab_init_fail_count += 1
                return None
            try:
                key_slab = _allocate_mutable_cpu_tensor(
                    (
                        chunk_slots,
                        self._slab_page_tokens,
                        self._slab_kv_heads,
                        self._slab_head_dim,
                    ),
                    dtype=page.key.dtype,
                    pin_memory=pin_memory,
                )
                value_slab = _allocate_mutable_cpu_tensor(
                    (
                        chunk_slots,
                        self._slab_page_tokens,
                        self._slab_kv_heads,
                        self._slab_head_dim,
                    ),
                    dtype=page.value.dtype,
                    pin_memory=pin_memory,
                )
            except (RuntimeError, MemoryError):
                self._slab_init_fail_count += 1
                return None
            self._key_slabs.append(key_slab)
            self._value_slabs.append(value_slab)
            self._slab_init_count += 1
        key_slab = self._key_slabs[chunk_index]
        value_slab = self._value_slabs[chunk_index]
        if local_slot >= int(key_slab.shape[0]):
            self._slab_init_fail_count += 1
            return None
        return key_slab, value_slab, local_slot

    def _store_in_contiguous_slab_locked(
        self,
        slot_id: int,
        page: TieredDRAMKVPage,
    ) -> TieredDRAMKVPage | None:
        slab_location = self._ensure_contiguous_slab_locked(page, slot_id)
        if slab_location is None:
            self._slab_store_fallback_count += 1
            return None
        key_slab, value_slab, local_slot = slab_location
        token_count = int(page.token_count)
        key_page = _mutable_cpu_tensor_view(
            key_slab,
            (local_slot, slice(0, token_count)),
        )
        value_page = _mutable_cpu_tensor_view(
            value_slab,
            (local_slot, slice(0, token_count)),
        )
        key_page.copy_(page.key[:token_count])
        value_page.copy_(page.value[:token_count])
        self._slab_store_count += 1
        return TieredDRAMKVPage(
            key=key_page,
            value=value_page,
            token_start=page.token_start,
            token_count=token_count,
            version=page.version,
        )

    def put(
        self,
        page_key: TieredKVPageKey,
        page: TieredDRAMKVPage,
    ) -> bool:
        if not tiered_dram_page_pool_enabled():
            return False

        max_slots = tiered_dram_page_pool_max_slots()
        latest_key = (
            page_key.request_key,
            page_key.layer_id,
            page_key.logical_page,
        )
        with self._lock:
            old_slot_id = self._slot_by_key.get(page_key)
            if old_slot_id is not None:
                old_slot = self._slots[old_slot_id]
                if old_slot is not None and old_slot.reader_count == 0:
                    self._byte_count -= self._page_bytes(old_slot.page)
                    if self._page_is_pinned(old_slot.page):
                        self._pinned_page_count -= 1
                    stored_page = (
                        self._store_in_contiguous_slab_locked(
                            old_slot_id, page
                        )
                        or page
                    )
                    self._slots[old_slot_id] = PinnedDRAMKVSlot(
                        page_key=page_key,
                        page=stored_page,
                        state="READY",
                    )
                    self._latest_slot[latest_key] = old_slot_id
                    self._put_count += 1
                    self._overwrite_count += 1
                    self._byte_count += self._page_bytes(stored_page)
                    if self._page_is_pinned(stored_page):
                        self._pinned_page_count += 1
                    return True

            if self._free_slots:
                slot_id = self._free_slots.pop()
            else:
                if max_slots > 0 and len(self._slots) >= max_slots:
                    self._put_fail_count += 1
                    return False
                slot_id = len(self._slots)
                self._slots.append(None)

            stored_page = (
                self._store_in_contiguous_slab_locked(slot_id, page) or page
            )
            self._slots[slot_id] = PinnedDRAMKVSlot(
                page_key=page_key,
                page=stored_page,
                state="READY",
            )
            self._slot_by_key[page_key] = slot_id
            self._latest_slot[latest_key] = slot_id
            if old_slot_id is not None:
                old_slot = self._slots[old_slot_id]
                if old_slot is not None:
                    old_slot.state = "RETIRED"
                    old_slot.retired = True
                    self._active_slots = max(0, self._active_slots - 1)
                    self._retired_slot_count += 1
            self._put_count += 1
            if old_slot_id is not None:
                self._overwrite_count += 1
            self._active_slots += 1
            self._byte_count += self._page_bytes(stored_page)
            if self._page_is_pinned(stored_page):
                self._pinned_page_count += 1
            self._peak_slots = max(self._peak_slots, len(self._slots))
            return True

    def reserve_from_hbm(
        self,
        page_key: TieredKVPageKey,
        source_page: TieredDRAMKVPage,
    ) -> TieredDRAMKVSlotReservation | None:
        """Reserve an unpublished pinned slab slot for direct GPU D2H.

        The slot remains invisible to lookup until ``publish_reservation``.
        A new slot is used even when the same page version already exists so
        readers can continue using the previous READY snapshot while D2H is
        in flight.
        """

        if not (
            tiered_dram_page_pool_enabled()
            and tiered_dram_page_pool_contiguous_enabled()
        ):
            return None
        max_slots = tiered_dram_page_pool_max_slots()
        with self._lock:
            if self._free_slots:
                # Lowest-first reuse preserves contiguous slot runs for a
                # page-ordered registration batch whenever possible.
                slot_id = min(self._free_slots)
                self._free_slots.remove(slot_id)
            else:
                if max_slots > 0 and len(self._slots) >= max_slots:
                    self._direct_reserve_fail_count += 1
                    return None
                slot_id = len(self._slots)
                self._slots.append(None)

            slab_location = self._ensure_contiguous_slab_locked(
                source_page,
                slot_id,
            )
            if slab_location is None:
                self._slots[slot_id] = None
                self._free_slots.append(slot_id)
                self._direct_reserve_fail_count += 1
                return None
            key_slab, value_slab, local_slot = slab_location
            token_count = int(source_page.token_count)
            page = TieredDRAMKVPage(
                key=_mutable_cpu_tensor_view(
                    key_slab,
                    (local_slot, slice(0, token_count)),
                ),
                value=_mutable_cpu_tensor_view(
                    value_slab,
                    (local_slot, slice(0, token_count)),
                ),
                token_start=source_page.token_start,
                token_count=token_count,
                version=source_page.version,
            )
            self._slots[slot_id] = PinnedDRAMKVSlot(
                page_key=page_key,
                page=page,
                state="COPYING_FROM_HBM",
            )
            self._direct_reserve_count += 1
            self._peak_slots = max(self._peak_slots, len(self._slots))
            return TieredDRAMKVSlotReservation(
                slot_id=slot_id,
                page_key=page_key,
                page=page,
            )

    def publish_reservation(
        self,
        reservation: TieredDRAMKVSlotReservation,
    ) -> bool:
        """Atomically expose a completed direct-D2H reservation."""

        latest_key = (
            reservation.page_key.request_key,
            reservation.page_key.layer_id,
            reservation.page_key.logical_page,
        )
        with self._lock:
            slot = self._slots[reservation.slot_id]
            if (
                slot is None
                or slot.page_key != reservation.page_key
                or slot.state != "COPYING_FROM_HBM"
            ):
                return False
            old_slot_id = self._slot_by_key.get(reservation.page_key)
            slot.state = "READY"
            self._slot_by_key[reservation.page_key] = reservation.slot_id
            self._latest_slot[latest_key] = reservation.slot_id
            self._active_slots += 1
            self._byte_count += self._page_bytes(slot.page)
            if self._page_is_pinned(slot.page):
                self._pinned_page_count += 1
            if old_slot_id is not None and old_slot_id != reservation.slot_id:
                old_slot = self._slots[old_slot_id]
                if old_slot is not None:
                    if old_slot.reader_count > 0:
                        old_slot.state = "RETIRED"
                        old_slot.retired = True
                        self._active_slots = max(
                            0, self._active_slots - 1
                        )
                        self._retired_slot_count += 1
                    else:
                        self._free_slot_locked(old_slot_id)
            self._put_count += 1
            self._direct_publish_count += 1
            return True

    def cancel_reservation(
        self,
        reservation: TieredDRAMKVSlotReservation,
    ) -> None:
        with self._lock:
            slot = self._slots[reservation.slot_id]
            if (
                slot is None
                or slot.page_key != reservation.page_key
                or slot.state != "COPYING_FROM_HBM"
            ):
                return
            self._slots[reservation.slot_id] = None
            self._free_slots.append(reservation.slot_id)
            self._direct_cancel_count += 1

    def contiguous_span(
        self,
        slot_ids: list[int],
    ) -> TieredDRAMKVSlotSpan | None:
        """Return a zero-copy CPU view for consecutive full-page slots."""

        if not slot_ids:
            return None
        with self._lock:
            self._slab_span_lookup_count += 1
            if not self._key_slabs or not self._value_slabs:
                self._slab_span_miss_count += 1
                return None
            first_slot = slot_ids[0]
            if first_slot < 0 or slot_ids != list(
                range(first_slot, first_slot + len(slot_ids))
            ):
                self._slab_span_miss_count += 1
                return None
            if first_slot + len(slot_ids) > len(self._slots):
                self._slab_span_miss_count += 1
                return None
            chunk_index = first_slot // self._slab_chunk_slots
            last_chunk_index = slot_ids[-1] // self._slab_chunk_slots
            if chunk_index != last_chunk_index:
                self._slab_span_miss_count += 1
                return None
            local_first_slot = first_slot % self._slab_chunk_slots
            slots = [self._slots[slot_id] for slot_id in slot_ids]
            if any(
                slot is None
                or slot.state not in {"READY", "RETIRED"}
                or (slot.state == "RETIRED" and slot.reader_count <= 0)
                for slot in slots
            ):
                self._slab_span_miss_count += 1
                return None
            ready_slots = [slot for slot in slots if slot is not None]
            if any(
                int(slot.page.token_count) != self._slab_page_tokens
                for slot in ready_slots
            ):
                self._slab_span_miss_count += 1
                return None
            for previous, current in zip(ready_slots, ready_slots[1:]):
                if (
                    current.page.token_start
                    != previous.page.token_start + self._slab_page_tokens
                ):
                    self._slab_span_miss_count += 1
                    return None
            local_last_slot = local_first_slot + len(slot_ids)
            token_count = len(slot_ids) * self._slab_page_tokens
            key = self._key_slabs[chunk_index][
                local_first_slot:local_last_slot
            ].view(
                token_count,
                self._slab_kv_heads,
                self._slab_head_dim,
            )
            value = self._value_slabs[chunk_index][
                local_first_slot:local_last_slot
            ].view(
                token_count,
                self._slab_kv_heads,
                self._slab_head_dim,
            )
            self._slab_span_hit_count += 1
            self._slab_span_page_count += len(slot_ids)
            return TieredDRAMKVSlotSpan(
                key=key,
                value=value,
                token_start=ready_slots[0].page.token_start,
                token_count=token_count,
                first_slot_id=first_slot,
                slot_count=len(slot_ids),
            )

    def lookup(
        self,
        *,
        request_key: int,
        request_id: str,
        layer_name: str,
        layer_id: int,
        logical_page: int,
        version: int | None,
    ) -> TieredDRAMKVLookup | None:
        if not tiered_dram_page_pool_enabled():
            return None

        with self._lock:
            self._lookup_count += 1
            if version is None:
                self._latest_lookup_count += 1
                slot_id = self._latest_slot.get(
                    (request_key, layer_id, logical_page)
                )
            else:
                slot_id = self._slot_by_key.get(
                    TieredKVPageKey(
                        request_key=request_key,
                        request_id=request_id,
                        layer_name=layer_name,
                        layer_id=layer_id,
                        logical_page=logical_page,
                        version=version,
                    )
                )
            if slot_id is None:
                self._miss_count += 1
                return None
            slot = self._slots[slot_id]
            if slot is None or slot.state != "READY":
                self._miss_count += 1
                return None
            self._hit_count += 1
            return TieredDRAMKVLookup(
                page=slot.page,
                slot_id=slot_id,
                version=slot.page_key.version,
                source="pool",
            )

    def get(
        self,
        *,
        request_key: int,
        request_id: str,
        layer_name: str,
        layer_id: int,
        logical_page: int,
        version: int | None,
    ) -> TieredDRAMKVPage | None:
        lookup = self.lookup(
            request_key=request_key,
            request_id=request_id,
            layer_name=layer_name,
            layer_id=layer_id,
            logical_page=logical_page,
            version=version,
        )
        return None if lookup is None else lookup.page

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()
            self._free_slots.clear()
            self._slot_by_key.clear()
            self._latest_slot.clear()
            self._put_count = 0
            self._put_fail_count = 0
            self._overwrite_count = 0
            self._lookup_count = 0
            self._hit_count = 0
            self._miss_count = 0
            self._latest_lookup_count = 0
            self._pinned_page_count = 0
            self._byte_count = 0
            self._active_slots = 0
            self._peak_slots = 0
            self._key_slabs.clear()
            self._value_slabs.clear()
            self._slab_chunk_slots = 0
            self._slab_page_tokens = 0
            self._slab_kv_heads = 0
            self._slab_head_dim = 0
            self._slab_init_count = 0
            self._slab_init_fail_count = 0
            self._slab_store_count = 0
            self._slab_store_fallback_count = 0
            self._slab_span_lookup_count = 0
            self._slab_span_hit_count = 0
            self._slab_span_miss_count = 0
            self._slab_span_page_count = 0
            self._direct_reserve_count = 0
            self._direct_reserve_fail_count = 0
            self._direct_publish_count = 0
            self._direct_cancel_count = 0
            self._lease_acquire_count = 0
            self._lease_release_count = 0
            self._lease_acquire_fail_count = 0
            self._retired_slot_count = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "dram_page_pool_enabled": int(
                    tiered_dram_page_pool_enabled()
                ),
                "dram_page_pool_slots": self._active_slots,
                "dram_page_pool_slot_capacity": len(self._slots),
                "dram_page_pool_peak_slots": self._peak_slots,
                "dram_page_pool_free_slots": len(self._free_slots),
                "dram_page_pool_put_count": self._put_count,
                "dram_page_pool_put_fail_count": self._put_fail_count,
                "dram_page_pool_overwrite_count": self._overwrite_count,
                "dram_page_pool_lookup_count": self._lookup_count,
                "dram_page_pool_hit_count": self._hit_count,
                "dram_page_pool_miss_count": self._miss_count,
                "dram_page_pool_latest_lookup_count": (
                    self._latest_lookup_count
                ),
                "dram_page_pool_pinned_page_count": self._pinned_page_count,
                "dram_page_pool_byte_count": self._byte_count,
                "dram_page_pool_contiguous_enabled": int(
                    tiered_dram_page_pool_contiguous_enabled()
                ),
                "dram_page_pool_slab_ready": int(
                    bool(self._key_slabs) and bool(self._value_slabs)
                ),
                "dram_page_pool_slab_count": len(self._key_slabs),
                "dram_page_pool_slab_slot_capacity": sum(
                    int(slab.shape[0]) for slab in self._key_slabs
                ),
                "dram_page_pool_slab_chunk_slots": self._slab_chunk_slots,
                "dram_page_pool_slab_page_tokens": self._slab_page_tokens,
                "dram_page_pool_slab_init_count": self._slab_init_count,
                "dram_page_pool_slab_init_fail_count": (
                    self._slab_init_fail_count
                ),
                "dram_page_pool_slab_store_count": self._slab_store_count,
                "dram_page_pool_slab_store_fallback_count": (
                    self._slab_store_fallback_count
                ),
                "dram_page_pool_slab_span_lookup_count": (
                    self._slab_span_lookup_count
                ),
                "dram_page_pool_slab_span_hit_count": (
                    self._slab_span_hit_count
                ),
                "dram_page_pool_slab_span_miss_count": (
                    self._slab_span_miss_count
                ),
                "dram_page_pool_slab_span_page_count": (
                    self._slab_span_page_count
                ),
                "dram_page_pool_direct_reserve_count": (
                    self._direct_reserve_count
                ),
                "dram_page_pool_direct_reserve_fail_count": (
                    self._direct_reserve_fail_count
                ),
                "dram_page_pool_direct_publish_count": (
                    self._direct_publish_count
                ),
                "dram_page_pool_direct_cancel_count": (
                    self._direct_cancel_count
                ),
                "dram_page_pool_lease_acquire_count": (
                    self._lease_acquire_count
                ),
                "dram_page_pool_lease_release_count": (
                    self._lease_release_count
                ),
                "dram_page_pool_lease_acquire_fail_count": (
                    self._lease_acquire_fail_count
                ),
                "dram_page_pool_retired_slot_count": (
                    self._retired_slot_count
                ),
                "dram_page_pool_active_reader_count": sum(
                    slot.reader_count
                    for slot in self._slots
                    if slot is not None
                ),
            }


class DRAMKVPageRegistry:
    """Thread-safe debug registry for CPU-resident K/V pages."""

    def __init__(self) -> None:
        self._lock = Lock()
        # Serialize publication and reset of registry-visible page versions.
        # D2H copies can proceed concurrently, but register(), direct publish,
        # and clear() must not race while updating the authoritative mapping.
        self._registration_lock = Lock()
        self._pages: dict[TieredKVPageKey, TieredDRAMKVPage] = {}
        self._latest_versions: dict[tuple[int, int, int], int] = {}
        self._latest_token_counts: dict[tuple[int, int, int], int] = {}
        self._request_key_by_id: dict[str, int] = {}
        self._request_id_by_key: dict[int, str] = {}
        self._next_request_key = 0
        self._pool = PinnedDRAMKVPool()
        self._lookup_count = 0
        self._hit_count = 0
        self._miss_count = 0
        self._pinned_lookup_count = 0
        self._latest_lookup_count = 0
        self._register_count = 0
        self._register_from_hbm_count = 0
        self._register_from_hbm_call_count = 0
        self._register_from_hbm_duration_us_total = 0
        self._register_from_hbm_duration_us_last = 0
        self._register_from_hbm_pages_last = 0
        self._register_from_hbm_skip_count = 0
        self._register_from_hbm_skip_decode_count = 0
        self._register_from_hbm_async_submit_count = 0
        self._register_from_hbm_async_complete_count = 0
        self._register_from_hbm_async_pending_pages = 0
        self._register_from_hbm_async_page_count = 0
        self._register_from_hbm_async_fallback_sync_count = 0
        self._register_from_hbm_async_fail_count = 0
        self._register_from_hbm_async_duration_us_total = 0
        self._register_from_hbm_async_duration_us_last = 0
        self._register_from_hbm_reconcile_verified_pages = 0
        self._register_from_hbm_reconcile_newer_version_pages = 0
        self._register_from_hbm_reconcile_missing_pages = 0
        self._d2h_control_dispatch_submit_count = 0
        self._d2h_control_dispatch_complete_count = 0
        self._d2h_control_dispatch_pending_jobs = 0
        self._d2h_control_dispatch_inline_fallback_count = 0
        self._d2h_control_dispatch_fail_count = 0
        self._d2h_control_dispatch_submit_us_total = 0
        self._d2h_control_dispatch_submit_us_last = 0
        self._d2h_control_dispatch_queue_wait_us_total = 0
        self._d2h_control_dispatch_queue_wait_us_last = 0
        self._d2h_control_dispatch_control_us_total = 0
        self._d2h_control_dispatch_control_us_last = 0
        self._d2h_source_plan_forward_count = 0
        self._d2h_source_plan_page_count = 0
        self._d2h_source_plan_fallback_scan_count = 0
        self._d2h_source_view_dispatch_submit_count = 0
        self._d2h_source_view_dispatch_submit_pages = 0
        self._d2h_source_view_dispatch_materialize_count = 0
        self._d2h_source_view_dispatch_materialize_pages = 0
        self._d2h_source_view_dispatch_materialize_us_total = 0
        self._d2h_source_view_dispatch_materialize_us_last = 0
        self._d2h_source_view_dispatch_fallback_count = 0
        self._register_from_hbm_direct_attempt_pages = 0
        self._register_from_hbm_direct_success_pages = 0
        self._register_from_hbm_direct_fallback_count = 0
        self._register_from_hbm_direct_inference_tensor_error_count = 0
        self._register_from_hbm_direct_byte_count = 0
        self._register_from_hbm_direct_duration_us_total = 0
        self._register_from_hbm_direct_duration_us_last = 0
        self._register_from_hbm_direct_copy_ops = 0
        self._register_from_hbm_direct_bulk_copy_ops = 0
        self._register_from_hbm_direct_bulk_pages = 0
        self._register_from_hbm_direct_multicopy_ops = 0
        self._register_from_hbm_direct_multicopy_runs = 0
        self._register_from_hbm_direct_multicopy_pages = 0
        self._register_from_hbm_direct_multicopy_bytes = 0
        self._register_from_hbm_direct_enqueue_count = 0
        self._register_from_hbm_direct_enqueue_us_total = 0
        self._register_from_hbm_direct_enqueue_us_last = 0
        self._register_from_hbm_direct_completion_wait_us_total = 0
        self._register_from_hbm_direct_completion_wait_us_last = 0
        self._register_from_hbm_direct_fallback_warned = False
        self._reload_to_hbm_count = 0
        self._reload_to_hbm_page_count = 0
        self._reload_to_hbm_byte_count = 0
        self._reload_to_hbm_fail_count = 0
        self._reload_to_hbm_ack_wait_count = 0
        self._reload_to_hbm_ack_wait_us_total = 0
        self._reload_to_hbm_ack_wait_us_last = 0
        self._reload_to_hbm_cpu_block_table_count = 0
        self._reload_to_hbm_plan_us_total = 0
        self._reload_to_hbm_plan_us_last = 0
        self._reload_to_hbm_lookup_us_total = 0
        self._reload_to_hbm_lookup_us_last = 0
        self._reload_to_hbm_enqueue_us_total = 0
        self._reload_to_hbm_enqueue_us_last = 0
        self._reload_to_hbm_copy_ops = 0
        self._reload_to_hbm_page_copy_ops = 0
        self._reload_to_hbm_bulk_copy_ops = 0
        self._reload_to_hbm_bulk_pages = 0
        self._reload_to_hbm_bulk_bytes = 0
        self._reload_to_hbm_direct_multicopy_ops = 0
        self._reload_to_hbm_direct_multicopy_runs = 0
        self._reload_to_hbm_direct_multicopy_pages = 0
        self._reload_to_hbm_direct_multicopy_bytes = 0
        self._reload_to_hbm_staging_scatter_ops = 0
        self._reload_to_hbm_staging_scatter_pages = 0
        self._reload_to_hbm_staging_scatter_bytes = 0
        self._reload_to_hbm_staging_scatter_fallback_pages = 0
        self._reload_to_hbm_request_key_missing_pages = 0
        self._reload_to_hbm_invalid_sequence_pages = 0
        self._reload_to_hbm_invalid_block_table_pages = 0
        self._reload_to_hbm_invalid_token_count_pages = 0
        self._reload_to_hbm_provider_lookup_miss_pages = 0
        self._reload_to_hbm_newer_version_fallback_pages = 0
        self._reload_to_hbm_pool_lease_failure_pages = 0

    def _resolve_request_key_locked(
        self,
        request_id: str,
        *,
        create: bool,
    ) -> int | None:
        request_key = self._request_key_by_id.get(request_id)
        if request_key is not None or not create:
            return request_key
        request_key = self._next_request_key
        self._next_request_key += 1
        self._request_key_by_id[request_id] = request_key
        self._request_id_by_key[request_key] = request_id
        return request_key

    def register(
        self,
        *,
        request_id: str = "",
        request_key: int | None = None,
        layer_name: str,
        layer_id: int | None = None,
        logical_page: int,
        key: torch.Tensor,
        value: torch.Tensor,
        token_start: int,
        token_count: int | None = None,
        version: int = 0,
        copy_to_cpu: bool = True,
    ) -> None:
        if tuple(key.shape) != tuple(value.shape):
            raise ValueError("key/value shapes must match")
        if key.dim() != 3:
            raise ValueError("key/value pages must have shape [T, H, D]")
        if token_count is None:
            token_count = int(key.shape[0])
        if token_count < 0 or token_count > int(key.shape[0]):
            raise ValueError("token_count is outside the key/value page shape")
        resolved_layer_id = (
            virtual_context_layer_id(layer_name)
            if layer_id is None
            else layer_id
        )
        if request_key is None:
            with self._lock:
                request_key = self._resolve_request_key_locked(
                    request_id,
                    create=True,
                )
        assert request_key is not None
        if copy_to_cpu:
            key = key.detach().to(device="cpu").contiguous()
            value = value.detach().to(device="cpu").contiguous()
            if (
                tiered_dram_provider_pin_memory_enabled()
                and torch.cuda.is_available()
            ):
                key = key.pin_memory()
                value = value.pin_memory()
        page_key = TieredKVPageKey(
            request_key=request_key,
            request_id=request_id,
            layer_name=layer_name,
            layer_id=resolved_layer_id,
            logical_page=logical_page,
            version=version,
        )
        page = TieredDRAMKVPage(
            key=key,
            value=value,
            token_start=token_start,
            token_count=token_count,
            version=version,
        )
        latest_key = (request_key, resolved_layer_id, logical_page)
        # Multiple async D2H workers may finish out of order. Serialize only
        # publication (not the copies) so a shorter/older tail snapshot cannot
        # overwrite the provider entry selected by the residency table.
        with self._registration_lock:
            with self._lock:
                latest_version = self._latest_versions.get(latest_key)
                latest_token_count = self._latest_token_counts.get(
                    latest_key, 0
                )
                if latest_version is not None and (
                    version < latest_version
                    or (
                        version == latest_version
                        and token_count < latest_token_count
                    )
                ):
                    return
            pool_stored = self._pool.put(page_key, page)
            mirror_dict = (
                not tiered_dram_page_pool_enabled()
                or tiered_dram_page_pool_mirror_dict_enabled()
                or not pool_stored
            )
            with self._lock:
                if mirror_dict:
                    self._pages[page_key] = page
                else:
                    self._pages.pop(page_key, None)
                self._latest_versions[latest_key] = version
                self._latest_token_counts[latest_key] = token_count
                self._register_count += 1

    def _publish_direct_reservation(
        self,
        reservation: TieredDRAMKVSlotReservation,
    ) -> bool:
        page_key = reservation.page_key
        page = reservation.page
        latest_key = (
            page_key.request_key,
            page_key.layer_id,
            page_key.logical_page,
        )
        with self._registration_lock:
            with self._lock:
                latest_version = self._latest_versions.get(latest_key)
                latest_token_count = self._latest_token_counts.get(
                    latest_key, 0
                )
                stale = latest_version is not None and (
                    page.version < latest_version
                    or (
                        page.version == latest_version
                        and page.token_count < latest_token_count
                    )
                )
            if stale:
                self._pool.cancel_reservation(reservation)
                return False
            if not self._pool.publish_reservation(reservation):
                self._pool.cancel_reservation(reservation)
                return False
            mirror_dict = tiered_dram_page_pool_mirror_dict_enabled()
            with self._lock:
                if mirror_dict:
                    self._pages[page_key] = page
                else:
                    self._pages.pop(page_key, None)
                self._latest_versions[latest_key] = page.version
                self._latest_token_counts[latest_key] = page.token_count
                self._register_count += 1
            return True

    def enqueue_from_hbm_direct(
        self,
        pages: list[_PendingHBMRegistrationPage],
        *,
        source_ready_event: torch.cuda.Event | None,
    ) -> TieredDirectD2HRegistration | None:
        """Reserve pinned slots and enqueue D2H without waiting for a worker."""

        if not pages:
            return None
        eligible = (
            tiered_dram_page_pool_enabled()
            and tiered_dram_page_pool_contiguous_enabled()
            and tiered_dram_provider_pin_memory_enabled()
            and torch.cuda.is_available()
            and all(page.key.is_cuda and page.value.is_cuda for page in pages)
        )
        if not eligible:
            with self._lock:
                self._register_from_hbm_direct_fallback_count += 1
            return None

        device = pages[0].key.device
        if any(
            page.key.device != device or page.value.device != device
            for page in pages
        ):
            with self._lock:
                self._register_from_hbm_direct_fallback_count += 1
            return None

        reservations: list[
            tuple[_PendingHBMRegistrationPage, TieredDRAMKVSlotReservation]
        ] = []
        with self._lock:
            self._register_from_hbm_direct_attempt_pages += len(pages)
        for pending in pages:
            page_key = TieredKVPageKey(
                request_key=pending.request_key,
                request_id=pending.request_id,
                layer_name=pending.layer_name,
                layer_id=pending.layer_id,
                logical_page=pending.logical_page,
                version=pending.version,
            )
            source_page = TieredDRAMKVPage(
                key=pending.key,
                value=pending.value,
                token_start=pending.token_start,
                token_count=pending.token_count,
                version=pending.version,
            )
            reservation = self._pool.reserve_from_hbm(
                page_key,
                source_page,
            )
            if reservation is None:
                for _, previous in reservations:
                    self._pool.cancel_reservation(previous)
                with self._lock:
                    self._register_from_hbm_direct_fallback_count += 1
                return None
            reservations.append((pending, reservation))

        started_at = time.perf_counter()
        copy_ops = 0
        bulk_copy_ops = 0
        bulk_pages = 0
        direct_multicopy_ops = 0
        direct_multicopy_runs = 0
        direct_multicopy_pages = 0
        direct_multicopy_bytes = 0

        copy_stream = None
        copy_enqueued = False
        try:
            copy_stream = _get_async_hbm_d2h_stream(device)
            with torch.cuda.device(device), torch.cuda.stream(copy_stream):
                if source_ready_event is not None:
                    copy_stream.wait_event(source_ready_event)
                bulk_token_limit = tiered_dram_register_bulk_tokens()
                direct_multicopy_enabled = (
                    tiered_dram_register_direct_multicopy_enabled()
                )
                page_index = 0
                while page_index < len(reservations):
                    first_pending, first_reservation = reservations[page_index]
                    run = [reservations[page_index]]
                    run_tokens = first_pending.token_count
                    next_index = page_index + 1
                    while next_index < len(reservations):
                        pending, reservation = reservations[next_index]
                        previous_pending, previous_reservation = run[-1]
                        if (
                            pending.request_key != first_pending.request_key
                            or pending.layer_id != first_pending.layer_id
                            or pending.logical_page
                            != previous_pending.logical_page + 1
                            or (
                                not direct_multicopy_enabled
                                and pending.block_id
                                != previous_pending.block_id + 1
                            )
                            or reservation.slot_id
                            != previous_reservation.slot_id + 1
                            or pending.token_start
                            != previous_pending.token_start
                            + previous_pending.token_count
                            or previous_pending.token_count
                            != tiered_dram_page_pool_page_tokens()
                            or pending.token_count
                            != tiered_dram_page_pool_page_tokens()
                            or (
                                bulk_token_limit > 0
                                and run_tokens + pending.token_count
                                > bulk_token_limit
                            )
                        ):
                            break
                        run.append((pending, reservation))
                        run_tokens += pending.token_count
                        next_index += 1

                    source_keys = [
                        pending.key[: pending.token_count]
                        for pending, _ in run
                    ]
                    source_values = [
                        pending.value[: pending.token_count]
                        for pending, _ in run
                    ]
                    destination_keys = [
                        reservation.page.key for _, reservation in run
                    ]
                    destination_values = [
                        reservation.page.value for _, reservation in run
                    ]
                    source_key_span = _contiguous_tensor_span(source_keys)
                    source_value_span = _contiguous_tensor_span(source_values)
                    destination_key_span = _contiguous_tensor_span(
                        destination_keys
                    )
                    destination_value_span = _contiguous_tensor_span(
                        destination_values
                    )
                    if (
                        len(run) >= 2
                        and source_key_span is not None
                        and source_value_span is not None
                        and destination_key_span is not None
                        and destination_value_span is not None
                    ):
                        copy_enqueued = True
                        destination_key_span.copy_(
                            source_key_span,
                            non_blocking=True,
                        )
                        destination_value_span.copy_(
                            source_value_span,
                            non_blocking=True,
                        )
                        copy_ops += 2
                        bulk_copy_ops += 2
                        bulk_pages += len(run)
                    elif (
                        direct_multicopy_enabled
                        and len(run) >= 2
                        and destination_key_span is not None
                        and destination_value_span is not None
                    ):
                        copy_enqueued = True
                        copy_group_count = int(
                            _custom_ops_module().tiered_d2h_multicopy(
                                source_keys,
                                source_values,
                                destination_key_span,
                                destination_value_span,
                            )
                        )
                        copy_ops += copy_group_count * 2
                        direct_multicopy_ops += copy_group_count * 2
                        direct_multicopy_runs += copy_group_count
                        direct_multicopy_pages += len(run)
                        direct_multicopy_bytes += sum(
                            source_key.numel()
                            * source_key.element_size()
                            + source_value.numel()
                            * source_value.element_size()
                            for source_key, source_value in zip(
                                source_keys,
                                source_values,
                            )
                        )
                    else:
                        for pending, reservation in run:
                            copy_enqueued = True
                            reservation.page.key.copy_(
                                pending.key[: pending.token_count],
                                non_blocking=True,
                            )
                            reservation.page.value.copy_(
                                pending.value[: pending.token_count],
                                non_blocking=True,
                            )
                            copy_ops += 2
                    page_index = next_index
                copy_complete = torch.cuda.Event(enable_timing=False)
                copy_complete.record(copy_stream)
        except Exception as exc:
            # A copy API can fail after earlier operations were already
            # enqueued. Do not return their slab slots to the allocator until
            # those writes have either completed or the CUDA context is known
            # to be unusable. Leaking an unpublished reservation on a poisoned
            # context is safer than allowing a later page to reuse storage
            # that an old DMA may still modify.
            safe_to_reuse = not copy_enqueued
            if copy_enqueued and copy_stream is not None:
                try:
                    copy_stream.synchronize()
                    safe_to_reuse = True
                except Exception:
                    safe_to_reuse = False
            if safe_to_reuse:
                for _, reservation in reservations:
                    self._pool.cancel_reservation(reservation)
            with self._lock:
                self._register_from_hbm_direct_fallback_count += 1
                if "inference tensor" in str(exc).lower():
                    self._register_from_hbm_direct_inference_tensor_error_count += 1
                should_warn = not self._register_from_hbm_direct_fallback_warned
                self._register_from_hbm_direct_fallback_warned = True
            if should_warn:
                logger.warning(
                    "Victim D2H direct path failed; later copies may use the "
                    "Tensor.copy_ fallback: %s",
                    exc,
                )
            return None

        enqueue_us = max(
            0,
            int((time.perf_counter() - started_at) * 1_000_000),
        )
        with self._lock:
            self._register_from_hbm_direct_enqueue_count += 1
            self._register_from_hbm_direct_enqueue_us_total += enqueue_us
            self._register_from_hbm_direct_enqueue_us_last = enqueue_us
        return TieredDirectD2HRegistration(
            pages=tuple(pages),
            reservations=tuple(reservations),
            copy_complete=copy_complete,
            started_at=started_at,
            copy_ops=copy_ops,
            bulk_copy_ops=bulk_copy_ops,
            bulk_pages=bulk_pages,
            direct_multicopy_ops=direct_multicopy_ops,
            direct_multicopy_runs=direct_multicopy_runs,
            direct_multicopy_pages=direct_multicopy_pages,
            direct_multicopy_bytes=direct_multicopy_bytes,
        )

    def complete_from_hbm_direct(
        self,
        registration: TieredDirectD2HRegistration,
    ) -> int:
        """Wait for an enqueued D2H and atomically publish its page slots."""

        wait_started = time.perf_counter()
        try:
            registration.copy_complete.synchronize()
        except Exception:
            for _, reservation in registration.reservations:
                self._pool.cancel_reservation(reservation)
            wait_us = max(
                0,
                int((time.perf_counter() - wait_started) * 1_000_000),
            )
            with self._lock:
                self._register_from_hbm_direct_fallback_count += 1
                self._register_from_hbm_direct_completion_wait_us_total += (
                    wait_us
                )
                self._register_from_hbm_direct_completion_wait_us_last = (
                    wait_us
                )
            return len(registration.pages)

        wait_us = max(
            0,
            int((time.perf_counter() - wait_started) * 1_000_000),
        )

        failed = 0
        byte_count = 0
        for _, reservation in registration.reservations:
            if self._publish_direct_reservation(reservation):
                byte_count += (
                    reservation.page.key.numel()
                    * reservation.page.key.element_size()
                    + reservation.page.value.numel()
                    * reservation.page.value.element_size()
                )
            else:
                failed += 1
        duration_us = max(
            0,
            int(
                (time.perf_counter() - registration.started_at)
                * 1_000_000
            ),
        )
        with self._lock:
            self._register_from_hbm_direct_success_pages += (
                len(registration.pages) - failed
            )
            self._register_from_hbm_direct_byte_count += byte_count
            self._register_from_hbm_direct_duration_us_total += duration_us
            self._register_from_hbm_direct_duration_us_last = duration_us
            self._register_from_hbm_direct_copy_ops += registration.copy_ops
            self._register_from_hbm_direct_bulk_copy_ops += (
                registration.bulk_copy_ops
            )
            self._register_from_hbm_direct_bulk_pages += (
                registration.bulk_pages
            )
            self._register_from_hbm_direct_multicopy_ops += (
                registration.direct_multicopy_ops
            )
            self._register_from_hbm_direct_multicopy_runs += (
                registration.direct_multicopy_runs
            )
            self._register_from_hbm_direct_multicopy_pages += (
                registration.direct_multicopy_pages
            )
            self._register_from_hbm_direct_multicopy_bytes += (
                registration.direct_multicopy_bytes
            )
            self._register_from_hbm_direct_completion_wait_us_total += wait_us
            self._register_from_hbm_direct_completion_wait_us_last = wait_us
        return failed

    def register_from_hbm_direct(
        self,
        pages: list[_PendingHBMRegistrationPage],
        *,
        source_ready_event: torch.cuda.Event | None,
    ) -> tuple[bool, int]:
        """Compatibility wrapper that enqueues and then waits synchronously."""

        registration = self.enqueue_from_hbm_direct(
            pages,
            source_ready_event=source_ready_event,
        )
        if registration is None:
            return False, len(pages)
        return True, self.complete_from_hbm_direct(registration)

    def record_register_from_hbm(
        self,
        page_count: int,
        *,
        duration_s: float = 0.0,
    ) -> None:
        with self._lock:
            self._register_from_hbm_call_count += 1
            self._register_from_hbm_count += page_count
            duration_us = max(0, int(duration_s * 1_000_000))
            self._register_from_hbm_duration_us_total += duration_us
            self._register_from_hbm_duration_us_last = duration_us
            self._register_from_hbm_pages_last = page_count

    def record_register_from_hbm_skip(self, *, phase: str, reason: str) -> None:
        with self._lock:
            self._register_from_hbm_skip_count += 1
            if (
                phase in {"decode", "mixed", "decode_like"}
                or "decode" in reason
            ):
                self._register_from_hbm_skip_decode_count += 1

    def try_reserve_async_register_from_hbm(
        self,
        page_count: int,
        *,
        required: bool = False,
    ) -> bool:
        if page_count <= 0:
            return False
        max_pending = tiered_dram_register_async_max_pending_pages()
        with self._lock:
            if (
                max_pending > 0
                and self._register_from_hbm_async_pending_pages + page_count
                > max_pending
            ):
                if not required:
                    self._register_from_hbm_async_fallback_sync_count += 1
                return False
            self._register_from_hbm_async_submit_count += 1
            self._register_from_hbm_async_pending_pages += page_count
            self._register_from_hbm_async_page_count += page_count
            return True

    def record_async_register_from_hbm_complete(
        self,
        page_count: int,
        *,
        duration_s: float,
        fail_count: int,
    ) -> None:
        duration_us = max(0, int(duration_s * 1_000_000))
        with self._lock:
            self._register_from_hbm_async_complete_count += 1
            self._register_from_hbm_async_pending_pages = max(
                0,
                self._register_from_hbm_async_pending_pages
                - max(0, page_count),
            )
            self._register_from_hbm_async_fail_count += max(0, fail_count)
            self._register_from_hbm_async_duration_us_total += duration_us
            self._register_from_hbm_async_duration_us_last = duration_us
        # Notify outside the registry lock: the required-plan worker checks
        # this budget while holding its own condition, never the reverse.
        required_d2h_progress.budget_released()

    def record_register_from_hbm_reconciliation(
        self,
        *,
        verified_pages: int,
        newer_version_pages: int,
        missing_pages: int,
    ) -> None:
        with self._lock:
            self._register_from_hbm_reconcile_verified_pages += max(
                0, verified_pages
            )
            self._register_from_hbm_reconcile_newer_version_pages += max(
                0, newer_version_pages
            )
            self._register_from_hbm_reconcile_missing_pages += max(
                0, missing_pages
            )

    def cancel_async_register_from_hbm_reservation(
        self,
        page_count: int,
    ) -> None:
        with self._lock:
            self._register_from_hbm_async_pending_pages = max(
                0,
                self._register_from_hbm_async_pending_pages
                - max(0, page_count),
            )
            self._register_from_hbm_async_fallback_sync_count += 1
        required_d2h_progress.budget_released()

    def record_d2h_control_dispatch_submit(self) -> None:
        with self._lock:
            self._d2h_control_dispatch_submit_count += 1
            self._d2h_control_dispatch_pending_jobs += 1

    def record_d2h_control_dispatch_submit_duration(
        self,
        duration_s: float,
    ) -> None:
        duration_us = max(0, int(duration_s * 1_000_000))
        with self._lock:
            self._d2h_control_dispatch_submit_us_total += duration_us
            self._d2h_control_dispatch_submit_us_last = duration_us

    def record_d2h_control_dispatch_start(self, queue_wait_s: float) -> None:
        queue_wait_us = max(0, int(queue_wait_s * 1_000_000))
        with self._lock:
            self._d2h_control_dispatch_queue_wait_us_total += queue_wait_us
            self._d2h_control_dispatch_queue_wait_us_last = queue_wait_us

    def record_d2h_control_dispatch_complete(
        self,
        *,
        duration_s: float,
        failed: bool,
    ) -> None:
        duration_us = max(0, int(duration_s * 1_000_000))
        with self._lock:
            self._d2h_control_dispatch_complete_count += 1
            self._d2h_control_dispatch_pending_jobs = max(
                0,
                self._d2h_control_dispatch_pending_jobs - 1,
            )
            self._d2h_control_dispatch_control_us_total += duration_us
            self._d2h_control_dispatch_control_us_last = duration_us
            if failed:
                self._d2h_control_dispatch_fail_count += 1

    def record_d2h_control_dispatch_inline_fallback(self) -> None:
        with self._lock:
            self._d2h_control_dispatch_inline_fallback_count += 1

    def record_d2h_source_plan(self, *, used: bool, page_count: int) -> None:
        with self._lock:
            if used:
                self._d2h_source_plan_forward_count += 1
                self._d2h_source_plan_page_count += max(0, page_count)
            else:
                self._d2h_source_plan_fallback_scan_count += 1

    def record_d2h_source_view_dispatch_submit(self, page_count: int) -> None:
        with self._lock:
            self._d2h_source_view_dispatch_submit_count += 1
            self._d2h_source_view_dispatch_submit_pages += max(0, page_count)

    def record_d2h_source_view_dispatch_materialize(
        self,
        *,
        page_count: int,
        duration_s: float,
    ) -> None:
        duration_us = max(0, int(duration_s * 1_000_000))
        with self._lock:
            self._d2h_source_view_dispatch_materialize_count += 1
            self._d2h_source_view_dispatch_materialize_pages += max(
                0, page_count
            )
            self._d2h_source_view_dispatch_materialize_us_total += duration_us
            self._d2h_source_view_dispatch_materialize_us_last = duration_us

    def record_d2h_source_view_dispatch_fallback(self) -> None:
        with self._lock:
            self._d2h_source_view_dispatch_fallback_count += 1

    def record_reload_to_hbm(
        self,
        *,
        page_count: int,
        byte_count: int,
        fail_count: int,
        ack_wait_s: float = 0.0,
        cpu_block_table_used: bool = False,
        plan_s: float = 0.0,
        lookup_s: float = 0.0,
        enqueue_s: float = 0.0,
        copy_ops: int = 0,
        page_copy_ops: int = 0,
        bulk_copy_ops: int = 0,
        bulk_pages: int = 0,
        bulk_bytes: int = 0,
        direct_multicopy_ops: int = 0,
        direct_multicopy_runs: int = 0,
        direct_multicopy_pages: int = 0,
        direct_multicopy_bytes: int = 0,
        staging_scatter_ops: int = 0,
        staging_scatter_pages: int = 0,
        staging_scatter_bytes: int = 0,
        staging_scatter_fallback_pages: int = 0,
        request_key_missing_pages: int = 0,
        invalid_sequence_pages: int = 0,
        invalid_block_table_pages: int = 0,
        invalid_token_count_pages: int = 0,
        provider_lookup_miss_pages: int = 0,
        newer_version_fallback_pages: int = 0,
        pool_lease_failure_pages: int = 0,
    ) -> None:
        plan_us = max(0, int(plan_s * 1_000_000))
        lookup_us = max(0, int(lookup_s * 1_000_000))
        enqueue_us = max(0, int(enqueue_s * 1_000_000))
        with self._lock:
            self._reload_to_hbm_count += 1
            self._reload_to_hbm_page_count += page_count
            self._reload_to_hbm_byte_count += byte_count
            self._reload_to_hbm_fail_count += fail_count
            self._reload_to_hbm_cpu_block_table_count += int(
                cpu_block_table_used
            )
            self._reload_to_hbm_plan_us_total += plan_us
            self._reload_to_hbm_plan_us_last = plan_us
            self._reload_to_hbm_lookup_us_total += lookup_us
            self._reload_to_hbm_lookup_us_last = lookup_us
            self._reload_to_hbm_enqueue_us_total += enqueue_us
            self._reload_to_hbm_enqueue_us_last = enqueue_us
            self._reload_to_hbm_copy_ops += max(0, copy_ops)
            self._reload_to_hbm_page_copy_ops += max(0, page_copy_ops)
            self._reload_to_hbm_bulk_copy_ops += max(0, bulk_copy_ops)
            self._reload_to_hbm_bulk_pages += max(0, bulk_pages)
            self._reload_to_hbm_bulk_bytes += max(0, bulk_bytes)
            self._reload_to_hbm_direct_multicopy_ops += max(
                0, direct_multicopy_ops
            )
            self._reload_to_hbm_direct_multicopy_runs += max(
                0, direct_multicopy_runs
            )
            self._reload_to_hbm_direct_multicopy_pages += max(
                0, direct_multicopy_pages
            )
            self._reload_to_hbm_direct_multicopy_bytes += max(
                0, direct_multicopy_bytes
            )
            self._reload_to_hbm_staging_scatter_ops += max(
                0, staging_scatter_ops
            )
            self._reload_to_hbm_staging_scatter_pages += max(
                0, staging_scatter_pages
            )
            self._reload_to_hbm_staging_scatter_bytes += max(
                0, staging_scatter_bytes
            )
            self._reload_to_hbm_staging_scatter_fallback_pages += max(
                0, staging_scatter_fallback_pages
            )
            self._reload_to_hbm_request_key_missing_pages += max(
                0, request_key_missing_pages
            )
            self._reload_to_hbm_invalid_sequence_pages += max(
                0, invalid_sequence_pages
            )
            self._reload_to_hbm_invalid_block_table_pages += max(
                0, invalid_block_table_pages
            )
            self._reload_to_hbm_invalid_token_count_pages += max(
                0, invalid_token_count_pages
            )
            self._reload_to_hbm_provider_lookup_miss_pages += max(
                0, provider_lookup_miss_pages
            )
            self._reload_to_hbm_newer_version_fallback_pages += max(
                0, newer_version_fallback_pages
            )
            self._reload_to_hbm_pool_lease_failure_pages += max(
                0, pool_lease_failure_pages
            )
            if page_count > 0:
                ack_wait_us = max(0, int(ack_wait_s * 1_000_000))
                self._reload_to_hbm_ack_wait_count += 1
                self._reload_to_hbm_ack_wait_us_total += ack_wait_us
                self._reload_to_hbm_ack_wait_us_last = ack_wait_us

    def lookup(
        self,
        *,
        request_id: str = "",
        request_key: int | None = None,
        layer_name: str,
        layer_id: int | None = None,
        logical_page: int,
        expected_token_count: int,
        version: int | None = None,
        strict: bool = True,
        allow_newer_version: bool = False,
    ) -> TieredDRAMKVLookup | None:
        resolved_layer_id = (
            virtual_context_layer_id(layer_name)
            if layer_id is None
            else layer_id
        )
        with self._lock:
            self._lookup_count += 1
            if request_key is None:
                request_key = self._resolve_request_key_locked(
                    request_id,
                    create=False,
                )
            if request_key is None:
                self._miss_count += 1
                return None
        requested_version = version
        lookup = self._pool.lookup(
            request_key=request_key,
            request_id=request_id,
            layer_name=layer_name,
            layer_id=resolved_layer_id,
            logical_page=logical_page,
            version=version,
        )
        if (
            lookup is None
            and requested_version is not None
            and allow_newer_version
        ):
            latest_lookup = self._pool.lookup(
                request_key=request_key,
                request_id=request_id,
                layer_name=layer_name,
                layer_id=resolved_layer_id,
                logical_page=logical_page,
                version=None,
            )
            if (
                latest_lookup is not None
                and _provider_snapshot_covers_pinned_version(
                    available_version=latest_lookup.version,
                    available_token_count=latest_lookup.page.token_count,
                    pinned_version=requested_version,
                    expected_token_count=expected_token_count,
                )
            ):
                lookup = latest_lookup
        if lookup is not None:
            if lookup.page.token_count < expected_token_count:
                with self._lock:
                    self._miss_count += 1
                if strict:
                    raise TieredBackendEligibilityError(
                        "registered DRAM page is shorter than expected: "
                        f"request={request_id} layer={layer_name} "
                        f"logical_page={logical_page} version={version} "
                        f"registered={lookup.page.token_count} "
                        f"expected={expected_token_count}"
                    )
                return None
            with self._lock:
                self._hit_count += 1
                if requested_version is None:
                    self._latest_lookup_count += 1
                else:
                    self._pinned_lookup_count += 1
            return lookup

        with self._lock:
            if requested_version is None:
                self._latest_lookup_count += 1
                version = self._latest_versions.get(
                    (request_key, resolved_layer_id, logical_page)
                )
                if version is None:
                    self._miss_count += 1
                    return None
            else:
                self._pinned_lookup_count += 1
                if allow_newer_version:
                    latest_version = self._latest_versions.get(
                        (request_key, resolved_layer_id, logical_page)
                    )
                    if (
                        latest_version is not None
                        and latest_version >= requested_version
                    ):
                        version = latest_version
            page = self._pages.get(
                TieredKVPageKey(
                    request_key=request_key,
                    request_id=request_id,
                    layer_name=layer_name,
                    layer_id=resolved_layer_id,
                    logical_page=logical_page,
                    version=version,
                )
            )
        if page is None:
            with self._lock:
                self._miss_count += 1
            return None
        if page.token_count < expected_token_count:
            with self._lock:
                self._miss_count += 1
            if strict:
                raise TieredBackendEligibilityError(
                    "registered DRAM page is shorter than expected: "
                    f"request={request_id} layer={layer_name} "
                    f"logical_page={logical_page} version={version} "
                    f"registered={page.token_count} "
                    f"expected={expected_token_count}"
                )
            return None
        with self._lock:
            self._hit_count += 1
        return TieredDRAMKVLookup(
            page=page,
            slot_id=-1,
            version=version,
            source="dict",
        )

    def get(
        self,
        *,
        request_id: str = "",
        request_key: int | None = None,
        layer_name: str,
        layer_id: int | None = None,
        logical_page: int,
        expected_token_count: int,
        version: int | None = None,
        strict: bool = True,
    ) -> TieredDRAMKVPage | None:
        lookup = self.lookup(
            request_id=request_id,
            request_key=request_key,
            layer_name=layer_name,
            layer_id=layer_id,
            logical_page=logical_page,
            expected_token_count=expected_token_count,
            version=version,
            strict=strict,
        )
        return None if lookup is None else lookup.page

    def acquire_pool_slots(
        self,
        slot_versions: list[tuple[int, int]],
    ) -> tuple[TieredDRAMKVLease, ...] | None:
        return self._pool.acquire_slots(slot_versions)

    def contiguous_pool_span(
        self,
        slot_ids: list[int],
    ) -> TieredDRAMKVSlotSpan | None:
        return self._pool.contiguous_span(slot_ids)

    def clear(self) -> None:
        with self._registration_lock:
            self._pool.clear()
            with self._lock:
                self._pages.clear()
                self._latest_versions.clear()
                self._latest_token_counts.clear()
                self._request_key_by_id.clear()
                self._request_id_by_key.clear()
                self._next_request_key = 0
                self._lookup_count = 0
                self._hit_count = 0
                self._miss_count = 0
                self._pinned_lookup_count = 0
                self._latest_lookup_count = 0
                self._register_count = 0
                self._register_from_hbm_count = 0
                self._register_from_hbm_call_count = 0
                self._register_from_hbm_duration_us_total = 0
                self._register_from_hbm_duration_us_last = 0
                self._register_from_hbm_pages_last = 0
                self._register_from_hbm_skip_count = 0
                self._register_from_hbm_skip_decode_count = 0
                self._register_from_hbm_async_submit_count = 0
                self._register_from_hbm_async_complete_count = 0
                self._register_from_hbm_async_pending_pages = 0
                self._register_from_hbm_async_page_count = 0
                self._register_from_hbm_async_fallback_sync_count = 0
                self._register_from_hbm_async_fail_count = 0
                self._register_from_hbm_async_duration_us_total = 0
                self._register_from_hbm_async_duration_us_last = 0
                self._register_from_hbm_reconcile_verified_pages = 0
                self._register_from_hbm_reconcile_newer_version_pages = 0
                self._register_from_hbm_reconcile_missing_pages = 0
                self._d2h_control_dispatch_submit_count = 0
                self._d2h_control_dispatch_complete_count = 0
                self._d2h_control_dispatch_pending_jobs = 0
                self._d2h_control_dispatch_inline_fallback_count = 0
                self._d2h_control_dispatch_fail_count = 0
                self._d2h_control_dispatch_submit_us_total = 0
                self._d2h_control_dispatch_submit_us_last = 0
                self._d2h_control_dispatch_queue_wait_us_total = 0
                self._d2h_control_dispatch_queue_wait_us_last = 0
                self._d2h_control_dispatch_control_us_total = 0
                self._d2h_control_dispatch_control_us_last = 0
                self._d2h_source_plan_forward_count = 0
                self._d2h_source_plan_page_count = 0
                self._d2h_source_plan_fallback_scan_count = 0
                self._d2h_source_view_dispatch_submit_count = 0
                self._d2h_source_view_dispatch_submit_pages = 0
                self._d2h_source_view_dispatch_materialize_count = 0
                self._d2h_source_view_dispatch_materialize_pages = 0
                self._d2h_source_view_dispatch_materialize_us_total = 0
                self._d2h_source_view_dispatch_materialize_us_last = 0
                self._d2h_source_view_dispatch_fallback_count = 0
                self._register_from_hbm_direct_attempt_pages = 0
                self._register_from_hbm_direct_success_pages = 0
                self._register_from_hbm_direct_fallback_count = 0
                self._register_from_hbm_direct_inference_tensor_error_count = 0
                self._register_from_hbm_direct_fallback_warned = False
                self._register_from_hbm_direct_byte_count = 0
                self._register_from_hbm_direct_duration_us_total = 0
                self._register_from_hbm_direct_duration_us_last = 0
                self._register_from_hbm_direct_copy_ops = 0
                self._register_from_hbm_direct_bulk_copy_ops = 0
                self._register_from_hbm_direct_bulk_pages = 0
                self._register_from_hbm_direct_multicopy_ops = 0
                self._register_from_hbm_direct_multicopy_runs = 0
                self._register_from_hbm_direct_multicopy_pages = 0
                self._register_from_hbm_direct_multicopy_bytes = 0
                self._register_from_hbm_direct_enqueue_count = 0
                self._register_from_hbm_direct_enqueue_us_total = 0
                self._register_from_hbm_direct_enqueue_us_last = 0
                self._register_from_hbm_direct_completion_wait_us_total = 0
                self._register_from_hbm_direct_completion_wait_us_last = 0
                self._reload_to_hbm_count = 0
                self._reload_to_hbm_page_count = 0
                self._reload_to_hbm_byte_count = 0
                self._reload_to_hbm_fail_count = 0
                self._reload_to_hbm_ack_wait_count = 0
                self._reload_to_hbm_ack_wait_us_total = 0
                self._reload_to_hbm_ack_wait_us_last = 0
                self._reload_to_hbm_cpu_block_table_count = 0
                self._reload_to_hbm_plan_us_total = 0
                self._reload_to_hbm_plan_us_last = 0
                self._reload_to_hbm_lookup_us_total = 0
                self._reload_to_hbm_lookup_us_last = 0
                self._reload_to_hbm_enqueue_us_total = 0
                self._reload_to_hbm_enqueue_us_last = 0
                self._reload_to_hbm_copy_ops = 0
                self._reload_to_hbm_page_copy_ops = 0
                self._reload_to_hbm_bulk_copy_ops = 0
                self._reload_to_hbm_bulk_pages = 0
                self._reload_to_hbm_bulk_bytes = 0
                self._reload_to_hbm_direct_multicopy_ops = 0
                self._reload_to_hbm_direct_multicopy_runs = 0
                self._reload_to_hbm_direct_multicopy_pages = 0
                self._reload_to_hbm_direct_multicopy_bytes = 0
                self._reload_to_hbm_staging_scatter_ops = 0
                self._reload_to_hbm_staging_scatter_pages = 0
                self._reload_to_hbm_staging_scatter_bytes = 0
                self._reload_to_hbm_staging_scatter_fallback_pages = 0
                self._reload_to_hbm_request_key_missing_pages = 0
                self._reload_to_hbm_invalid_sequence_pages = 0
                self._reload_to_hbm_invalid_block_table_pages = 0
                self._reload_to_hbm_invalid_token_count_pages = 0
                self._reload_to_hbm_provider_lookup_miss_pages = 0
                self._reload_to_hbm_newer_version_fallback_pages = 0
                self._reload_to_hbm_pool_lease_failure_pages = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            stats = {
                "dram_registered_pages": len(self._pages),
                "dram_registered_page_keys": len(self._latest_versions),
                "dram_provider_request_key_count": len(
                    self._request_key_by_id
                ),
                "dram_provider_register_count": self._register_count,
                "dram_provider_register_from_hbm_count": (
                    self._register_from_hbm_count
                ),
                "dram_provider_register_from_hbm_call_count": (
                    self._register_from_hbm_call_count
                ),
                "dram_provider_register_from_hbm_pages_last": (
                    self._register_from_hbm_pages_last
                ),
                "dram_provider_register_from_hbm_duration_us_total": (
                    self._register_from_hbm_duration_us_total
                ),
                "dram_provider_register_from_hbm_duration_us_last": (
                    self._register_from_hbm_duration_us_last
                ),
                "dram_provider_register_from_hbm_skip_count": (
                    self._register_from_hbm_skip_count
                ),
                "dram_provider_register_from_hbm_skip_decode_count": (
                    self._register_from_hbm_skip_decode_count
                ),
                "dram_provider_register_from_hbm_async_submit_count": (
                    self._register_from_hbm_async_submit_count
                ),
                "dram_provider_register_from_hbm_async_complete_count": (
                    self._register_from_hbm_async_complete_count
                ),
                "dram_provider_register_from_hbm_async_pending_pages": (
                    self._register_from_hbm_async_pending_pages
                ),
                "dram_provider_register_from_hbm_async_page_count": (
                    self._register_from_hbm_async_page_count
                ),
                "dram_provider_register_from_hbm_async_fallback_sync_count": (
                    self._register_from_hbm_async_fallback_sync_count
                ),
                "dram_provider_register_from_hbm_async_fail_count": (
                    self._register_from_hbm_async_fail_count
                ),
                "dram_provider_register_from_hbm_async_duration_us_total": (
                    self._register_from_hbm_async_duration_us_total
                ),
                "dram_provider_register_from_hbm_async_duration_us_last": (
                    self._register_from_hbm_async_duration_us_last
                ),
                "dram_provider_register_from_hbm_reconcile_verified_pages": (
                    self._register_from_hbm_reconcile_verified_pages
                ),
                "dram_provider_register_from_hbm_reconcile_newer_version_pages": (
                    self._register_from_hbm_reconcile_newer_version_pages
                ),
                "dram_provider_register_from_hbm_reconcile_missing_pages": (
                    self._register_from_hbm_reconcile_missing_pages
                ),
                "dram_provider_d2h_control_dispatch_enabled": int(
                    tiered_dram_register_control_dispatcher_enabled()
                ),
                "dram_provider_d2h_control_dispatch_submit_count": (
                    self._d2h_control_dispatch_submit_count
                ),
                "dram_provider_d2h_control_dispatch_complete_count": (
                    self._d2h_control_dispatch_complete_count
                ),
                "dram_provider_d2h_control_dispatch_pending_jobs": (
                    self._d2h_control_dispatch_pending_jobs
                ),
                "dram_provider_d2h_control_dispatch_inline_fallback_count": (
                    self._d2h_control_dispatch_inline_fallback_count
                ),
                "dram_provider_d2h_control_dispatch_fail_count": (
                    self._d2h_control_dispatch_fail_count
                ),
                "dram_provider_d2h_control_dispatch_submit_us_total": (
                    self._d2h_control_dispatch_submit_us_total
                ),
                "dram_provider_d2h_control_dispatch_submit_us_last": (
                    self._d2h_control_dispatch_submit_us_last
                ),
                "dram_provider_d2h_control_dispatch_queue_wait_us_total": (
                    self._d2h_control_dispatch_queue_wait_us_total
                ),
                "dram_provider_d2h_control_dispatch_queue_wait_us_last": (
                    self._d2h_control_dispatch_queue_wait_us_last
                ),
                "dram_provider_d2h_control_dispatch_control_us_total": (
                    self._d2h_control_dispatch_control_us_total
                ),
                "dram_provider_d2h_control_dispatch_control_us_last": (
                    self._d2h_control_dispatch_control_us_last
                ),
                "dram_provider_d2h_source_plan_enabled": int(
                    tiered_d2h_source_plan_enabled()
                ),
                "dram_provider_d2h_source_plan_forward_count": (
                    self._d2h_source_plan_forward_count
                ),
                "dram_provider_d2h_source_plan_page_count": (
                    self._d2h_source_plan_page_count
                ),
                "dram_provider_d2h_source_plan_fallback_scan_count": (
                    self._d2h_source_plan_fallback_scan_count
                ),
                "dram_provider_d2h_source_view_dispatch_enabled": int(
                    tiered_d2h_source_view_dispatch_enabled()
                ),
                "dram_provider_d2h_source_view_dispatch_submit_count": (
                    self._d2h_source_view_dispatch_submit_count
                ),
                "dram_provider_d2h_source_view_dispatch_submit_pages": (
                    self._d2h_source_view_dispatch_submit_pages
                ),
                "dram_provider_d2h_source_view_dispatch_materialize_count": (
                    self._d2h_source_view_dispatch_materialize_count
                ),
                "dram_provider_d2h_source_view_dispatch_materialize_pages": (
                    self._d2h_source_view_dispatch_materialize_pages
                ),
                "dram_provider_d2h_source_view_dispatch_materialize_us_total": (
                    self._d2h_source_view_dispatch_materialize_us_total
                ),
                "dram_provider_d2h_source_view_dispatch_materialize_us_last": (
                    self._d2h_source_view_dispatch_materialize_us_last
                ),
                "dram_provider_d2h_source_view_dispatch_fallback_count": (
                    self._d2h_source_view_dispatch_fallback_count
                ),
                **get_tiered_hbm_source_block_lease_stats().to_metrics(
                    "dram_provider_d2h_source_lease"
                ),
                "dram_provider_register_from_hbm_direct_attempt_pages": (
                    self._register_from_hbm_direct_attempt_pages
                ),
                "dram_provider_register_from_hbm_direct_success_pages": (
                    self._register_from_hbm_direct_success_pages
                ),
                "dram_provider_register_from_hbm_direct_fallback_count": (
                    self._register_from_hbm_direct_fallback_count
                ),
                "dram_provider_register_from_hbm_direct_inference_tensor_error_count": (
                    self._register_from_hbm_direct_inference_tensor_error_count
                ),
                "dram_provider_register_from_hbm_direct_byte_count": (
                    self._register_from_hbm_direct_byte_count
                ),
                "dram_provider_register_from_hbm_direct_duration_us_total": (
                    self._register_from_hbm_direct_duration_us_total
                ),
                "dram_provider_register_from_hbm_direct_duration_us_last": (
                    self._register_from_hbm_direct_duration_us_last
                ),
                "dram_provider_register_from_hbm_direct_copy_ops": (
                    self._register_from_hbm_direct_copy_ops
                ),
                "dram_provider_register_from_hbm_direct_bulk_copy_ops": (
                    self._register_from_hbm_direct_bulk_copy_ops
                ),
                "dram_provider_register_from_hbm_direct_bulk_pages": (
                    self._register_from_hbm_direct_bulk_pages
                ),
                "dram_provider_register_from_hbm_direct_multicopy_ops": (
                    self._register_from_hbm_direct_multicopy_ops
                ),
                "dram_provider_register_from_hbm_direct_multicopy_runs": (
                    self._register_from_hbm_direct_multicopy_runs
                ),
                "dram_provider_register_from_hbm_direct_multicopy_pages": (
                    self._register_from_hbm_direct_multicopy_pages
                ),
                "dram_provider_register_from_hbm_direct_multicopy_bytes": (
                    self._register_from_hbm_direct_multicopy_bytes
                ),
                "dram_provider_register_from_hbm_direct_enqueue_count": (
                    self._register_from_hbm_direct_enqueue_count
                ),
                "dram_provider_register_from_hbm_direct_enqueue_us_total": (
                    self._register_from_hbm_direct_enqueue_us_total
                ),
                "dram_provider_register_from_hbm_direct_enqueue_us_last": (
                    self._register_from_hbm_direct_enqueue_us_last
                ),
                "dram_provider_register_from_hbm_direct_completion_wait_us_total": (
                    self._register_from_hbm_direct_completion_wait_us_total
                ),
                "dram_provider_register_from_hbm_direct_completion_wait_us_last": (
                    self._register_from_hbm_direct_completion_wait_us_last
                ),
                "dram_provider_lookup_count": self._lookup_count,
                "dram_provider_hit_count": self._hit_count,
                "dram_provider_miss_count": self._miss_count,
                "dram_provider_pinned_lookup_count": (
                    self._pinned_lookup_count
                ),
                "dram_provider_latest_lookup_count": (
                    self._latest_lookup_count
                ),
                "dram_provider_reload_to_hbm_count": (
                    self._reload_to_hbm_count
                ),
                "dram_provider_reload_to_hbm_page_count": (
                    self._reload_to_hbm_page_count
                ),
                "dram_provider_reload_to_hbm_byte_count": (
                    self._reload_to_hbm_byte_count
                ),
                "dram_provider_reload_to_hbm_fail_count": (
                    self._reload_to_hbm_fail_count
                ),
                "dram_provider_reload_to_hbm_request_key_missing_pages": (
                    self._reload_to_hbm_request_key_missing_pages
                ),
                "dram_provider_reload_to_hbm_invalid_sequence_pages": (
                    self._reload_to_hbm_invalid_sequence_pages
                ),
                "dram_provider_reload_to_hbm_invalid_block_table_pages": (
                    self._reload_to_hbm_invalid_block_table_pages
                ),
                "dram_provider_reload_to_hbm_invalid_token_count_pages": (
                    self._reload_to_hbm_invalid_token_count_pages
                ),
                "dram_provider_reload_to_hbm_provider_lookup_miss_pages": (
                    self._reload_to_hbm_provider_lookup_miss_pages
                ),
                "dram_provider_reload_to_hbm_newer_version_fallback_pages": (
                    self._reload_to_hbm_newer_version_fallback_pages
                ),
                "dram_provider_reload_to_hbm_pool_lease_failure_pages": (
                    self._reload_to_hbm_pool_lease_failure_pages
                ),
                "dram_provider_reload_to_hbm_ack_wait_count": (
                    self._reload_to_hbm_ack_wait_count
                ),
                "dram_provider_reload_to_hbm_ack_wait_us_total": (
                    self._reload_to_hbm_ack_wait_us_total
                ),
                "dram_provider_reload_to_hbm_ack_wait_us_last": (
                    self._reload_to_hbm_ack_wait_us_last
                ),
                "dram_provider_reload_to_hbm_cpu_block_table_count": (
                    self._reload_to_hbm_cpu_block_table_count
                ),
                "dram_provider_reload_to_hbm_plan_us_total": (
                    self._reload_to_hbm_plan_us_total
                ),
                "dram_provider_reload_to_hbm_plan_us_last": (
                    self._reload_to_hbm_plan_us_last
                ),
                "dram_provider_reload_to_hbm_lookup_us_total": (
                    self._reload_to_hbm_lookup_us_total
                ),
                "dram_provider_reload_to_hbm_lookup_us_last": (
                    self._reload_to_hbm_lookup_us_last
                ),
                "dram_provider_reload_to_hbm_enqueue_us_total": (
                    self._reload_to_hbm_enqueue_us_total
                ),
                "dram_provider_reload_to_hbm_enqueue_us_last": (
                    self._reload_to_hbm_enqueue_us_last
                ),
                "dram_provider_reload_to_hbm_copy_ops": (
                    self._reload_to_hbm_copy_ops
                ),
                "dram_provider_reload_to_hbm_page_copy_ops": (
                    self._reload_to_hbm_page_copy_ops
                ),
                "dram_provider_reload_to_hbm_bulk_copy_ops": (
                    self._reload_to_hbm_bulk_copy_ops
                ),
                "dram_provider_reload_to_hbm_bulk_pages": (
                    self._reload_to_hbm_bulk_pages
                ),
                "dram_provider_reload_to_hbm_bulk_bytes": (
                    self._reload_to_hbm_bulk_bytes
                ),
                "dram_provider_reload_to_hbm_direct_multicopy_ops": (
                    self._reload_to_hbm_direct_multicopy_ops
                ),
                "dram_provider_reload_to_hbm_direct_multicopy_runs": (
                    self._reload_to_hbm_direct_multicopy_runs
                ),
                "dram_provider_reload_to_hbm_direct_multicopy_pages": (
                    self._reload_to_hbm_direct_multicopy_pages
                ),
                "dram_provider_reload_to_hbm_direct_multicopy_bytes": (
                    self._reload_to_hbm_direct_multicopy_bytes
                ),
                "dram_provider_reload_to_hbm_staging_scatter_ops": (
                    self._reload_to_hbm_staging_scatter_ops
                ),
                "dram_provider_reload_to_hbm_staging_scatter_pages": (
                    self._reload_to_hbm_staging_scatter_pages
                ),
                "dram_provider_reload_to_hbm_staging_scatter_bytes": (
                    self._reload_to_hbm_staging_scatter_bytes
                ),
                "dram_provider_reload_to_hbm_staging_scatter_fallback_pages": (
                    self._reload_to_hbm_staging_scatter_fallback_pages
                ),
            }
        stats.update(self._pool.stats())
        return stats

    def request_key(self, request_id: str, *, create: bool) -> int | None:
        with self._lock:
            return self._resolve_request_key_locked(
                request_id,
                create=create,
            )


_DRAM_KV_PAGE_REGISTRY = DRAMKVPageRegistry()
_ASYNC_HBM_REGISTER_EXECUTOR: ThreadPoolExecutor | None = None
_ASYNC_HBM_REGISTER_EXECUTOR_LOCK = Lock()
_D2H_CONTROL_DISPATCH_EXECUTOR: ThreadPoolExecutor | None = None
_D2H_CONTROL_DISPATCH_EXECUTOR_LOCK = Lock()
_ASYNC_HBM_D2H_STREAM_LOCAL = local()
_PENDING_HBM_D2H_FENCES_LOCK = Lock()
_PENDING_HBM_D2H_FENCES: dict[
    tuple[str, int, int], Future[None]
] = {}
_HBM_D2H_FENCE_WAIT_COUNT = 0
_HBM_D2H_FENCE_WAIT_PAGE_COUNT = 0
_HBM_D2H_FENCE_WAIT_US_TOTAL = 0
_HBM_D2H_FENCE_WAIT_US_LAST = 0
_DECODE_RELOAD_BARRIER_CHECK_COUNT = 0
_DECODE_RELOAD_BARRIER_WAIT_COUNT = 0
_DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT = 0
_DECODE_RELOAD_BARRIER_WAIT_US_TOTAL = 0
_DECODE_RELOAD_BARRIER_WAIT_US_LAST = 0
_DECODE_RELOAD_BARRIER_COMPLETE_COUNT = 0
_DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT = 0
_DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT = 0
_DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT = 0
_DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT = 0


def has_pending_hbm_d2h_reads(request_ids: list[str] | None) -> bool:
    """Cheaply test whether a request still has an in-flight D2H reader."""

    if not request_ids:
        return False
    request_id_set = set(request_ids)
    with _PENDING_HBM_D2H_FENCES_LOCK:
        return any(
            request_id in request_id_set
            for request_id, _, _ in _PENDING_HBM_D2H_FENCES
        )


def wait_for_pending_hbm_d2h_request_reads(
    request_ids: list[str] | None,
) -> int:
    """Drain every in-flight D2H source read for the selected requests.

    The narrower ``wait_for_pending_hbm_d2h_reads`` fence protects only pages
    that the current forward is about to overwrite. Native decode handoff has a
    stronger requirement: every provider snapshot for the immutable prompt
    must be published before the worker plans DRAM-to-HBM reloads. Snapshot the
    matching futures while holding the registry lock, then wait without the
    lock so completion callbacks can remove their page fences.
    """

    global _DECODE_RELOAD_BARRIER_CHECK_COUNT
    global _DECODE_RELOAD_BARRIER_WAIT_COUNT
    global _DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT
    global _DECODE_RELOAD_BARRIER_WAIT_US_TOTAL
    global _DECODE_RELOAD_BARRIER_WAIT_US_LAST

    if not request_ids:
        return 0
    request_id_set = set(request_ids)
    with _PENDING_HBM_D2H_FENCES_LOCK:
        _DECODE_RELOAD_BARRIER_CHECK_COUNT += 1
        matched_fences = {
            key: future
            for key, future in _PENDING_HBM_D2H_FENCES.items()
            if key[0] in request_id_set and not future.done()
        }
    futures = set(matched_fences.values())
    if not futures:
        return 0

    wait_started = time.perf_counter()
    for future in futures:
        future.result()
    wait_us = max(0, int((time.perf_counter() - wait_started) * 1e6))
    with _PENDING_HBM_D2H_FENCES_LOCK:
        _DECODE_RELOAD_BARRIER_WAIT_COUNT += 1
        _DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT += len(matched_fences)
        _DECODE_RELOAD_BARRIER_WAIT_US_TOTAL += wait_us
        _DECODE_RELOAD_BARRIER_WAIT_US_LAST = wait_us
    return len(futures)


def record_decode_reload_completion_barrier(
    *,
    h2d_retry_count: int,
    h2d_retry_page_count: int,
    remaining_page_count: int,
) -> None:
    """Record the converged native-decode reload barrier outcome."""

    global _DECODE_RELOAD_BARRIER_COMPLETE_COUNT
    global _DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT
    global _DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT
    global _DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT
    global _DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT

    with _PENDING_HBM_D2H_FENCES_LOCK:
        if remaining_page_count > 0:
            _DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT += 1
            _DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT += (
                remaining_page_count
            )
        else:
            _DECODE_RELOAD_BARRIER_COMPLETE_COUNT += 1
        _DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT += max(0, h2d_retry_count)
        _DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT += max(
            0, h2d_retry_page_count
        )


def _hbm_d2h_page_pending(
    request_id: str,
    layer_id: int,
    logical_page: int,
) -> bool:
    with _PENDING_HBM_D2H_FENCES_LOCK:
        future = _PENDING_HBM_D2H_FENCES.get(
            (request_id, layer_id, logical_page)
        )
        return future is not None and not future.done()


def _publish_pending_hbm_d2h_fences(
    pages: list[_HBMRegistrationPage],
    future: Future[None],
) -> None:
    keys = {
        (page.request_id, page.layer_id, page.logical_page)
        for page in pages
    }
    with _PENDING_HBM_D2H_FENCES_LOCK:
        for key in keys:
            previous = _PENDING_HBM_D2H_FENCES.get(key)
            if previous is not None and not previous.done():
                raise RuntimeError(
                    "overlapping HBM-to-DRAM reads were submitted for "
                    f"logical page {key}"
                )
            _PENDING_HBM_D2H_FENCES[key] = future

    def clear_completed(completed: Future[None]) -> None:
        with _PENDING_HBM_D2H_FENCES_LOCK:
            for key in keys:
                if _PENDING_HBM_D2H_FENCES.get(key) is completed:
                    _PENDING_HBM_D2H_FENCES.pop(key, None)
        # Capacity retry can become schedulable only after this provider fence
        # is removed.  Publish that state change instead of waiting for the
        # EngineCore timeout fallback.
        notify_pd_engine_progress()

    future.add_done_callback(clear_completed)


def _release_hbm_source_block_lease(
    source_block_lease: TieredHBMSourceBlockLease | None,
) -> None:
    if source_block_lease is not None:
        source_block_lease.release()


def _get_async_hbm_d2h_stream(device: torch.device) -> torch.cuda.Stream:
    streams = getattr(_ASYNC_HBM_D2H_STREAM_LOCAL, "streams", None)
    if streams is None:
        streams = {}
        _ASYNC_HBM_D2H_STREAM_LOCAL.streams = streams
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = streams.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        streams[device_index] = stream
    return stream


def _get_async_hbm_register_executor() -> ThreadPoolExecutor:
    global _ASYNC_HBM_REGISTER_EXECUTOR
    with _ASYNC_HBM_REGISTER_EXECUTOR_LOCK:
        if _ASYNC_HBM_REGISTER_EXECUTOR is None:
            _ASYNC_HBM_REGISTER_EXECUTOR = ThreadPoolExecutor(
                max_workers=tiered_dram_register_async_workers(),
                thread_name_prefix="tiered-hbm-dram-register",
            )
        return _ASYNC_HBM_REGISTER_EXECUTOR


def _get_d2h_control_dispatch_executor() -> ThreadPoolExecutor:
    global _D2H_CONTROL_DISPATCH_EXECUTOR
    with _D2H_CONTROL_DISPATCH_EXECUTOR_LOCK:
        if _D2H_CONTROL_DISPATCH_EXECUTOR is None:
            _D2H_CONTROL_DISPATCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=tiered_dram_register_control_dispatcher_workers(),
                thread_name_prefix="tiered-d2h-control",
            )
        return _D2H_CONTROL_DISPATCH_EXECUTOR


def _register_pending_hbm_pages_sync(
    pages: list[_PendingHBMRegistrationPage],
) -> int:
    failed = 0
    with torch.no_grad():
        for page in pages:
            try:
                register_dram_kv_page(
                    request_id=page.request_id,
                    request_key=page.request_key,
                    layer_name=page.layer_name,
                    layer_id=page.layer_id,
                    logical_page=page.logical_page,
                    key=page.key,
                    value=page.value,
                    token_start=page.token_start,
                    token_count=page.token_count,
                    version=page.version,
                    copy_to_cpu=True,
                )
            except Exception:
                failed += 1
    return failed


def _materialize_hbm_registration_pages(
    pages: list[_HBMRegistrationPage],
) -> list[_PendingHBMRegistrationPage]:
    return [
        page.materialize()
        if isinstance(page, _DeferredHBMRegistrationPage)
        else page
        for page in pages
    ]


def _mark_hbm_registration_complete(
    *,
    registered_versions_by_request: dict[str, dict[int, int]],
    registered_blocks_by_request: dict[str, dict[int, int]],
    registered_token_counts_by_request: dict[str, dict[int, int]],
    block_size: int,
    layer_id: int,
    pages: list[_PendingHBMRegistrationPage] | None = None,
    require_existing_request: bool = False,
) -> int | None:
    """Publish only provider-verified D2H coverage to the residency sidecar.

    Async copies for the same logical page can complete out of order. In that
    case the registry intentionally rejects an older direct reservation, but a
    newer provider page is already authoritative. Batch-level success/failure
    cannot describe that state: reconcile each logical page against the latest
    provider mapping and publish the version that actually exists.
    """
    virtual_context = get_virtual_context_table()
    if virtual_context is None:
        # Provider-only configurations have no residency sidecar to
        # reconcile. Let the caller preserve the raw copy result.
        return None

    page_versions_by_request = registered_versions_by_request
    page_blocks_by_request = registered_blocks_by_request
    page_token_counts_by_request = registered_token_counts_by_request
    if pages is not None:
        verified_versions: dict[str, dict[int, int]] = defaultdict(dict)
        verified_blocks: dict[str, dict[int, int]] = defaultdict(dict)
        verified_token_counts: dict[str, dict[int, int]] = defaultdict(dict)
        newer_version_pages = 0
        for page in pages:
            lookup = _DRAM_KV_PAGE_REGISTRY.lookup(
                request_id=page.request_id,
                request_key=page.request_key,
                layer_name=page.layer_name,
                layer_id=page.layer_id,
                logical_page=page.logical_page,
                expected_token_count=page.token_count,
                version=None,
                strict=False,
            )
            if lookup is None or not _provider_snapshot_covers_pinned_version(
                available_version=lookup.version,
                available_token_count=lookup.page.token_count,
                pinned_version=page.version,
                expected_token_count=page.token_count,
            ):
                continue
            if lookup.version > page.version:
                newer_version_pages += 1
            verified_versions[page.request_id][page.logical_page] = (
                lookup.version
            )
            verified_blocks[page.request_id][page.logical_page] = page.block_id
            verified_token_counts[page.request_id][page.logical_page] = (
                lookup.page.token_count
            )
        page_versions_by_request = dict(verified_versions)
        page_blocks_by_request = dict(verified_blocks)
        page_token_counts_by_request = dict(verified_token_counts)
        provider_verified_pages = sum(
            len(page_versions)
            for page_versions in page_versions_by_request.values()
        )
        _DRAM_KV_PAGE_REGISTRY.record_register_from_hbm_reconciliation(
            verified_pages=provider_verified_pages,
            newer_version_pages=newer_version_pages,
            missing_pages=max(0, len(pages) - provider_verified_pages),
        )

    verified_page_count = 0
    for request_id, page_versions in page_versions_by_request.items():
        verified_page_count += len(page_versions)
        virtual_context.record_dram_copies_ready(
            request_id=request_id,
            page_versions=page_versions,
            block_size=block_size,
            group_id=0,
            layer_id=layer_id,
            page_to_block=page_blocks_by_request.get(request_id),
            page_token_counts=page_token_counts_by_request.get(
                request_id
            ),
            **({"require_existing_request": True} if require_existing_request else {}),
        )
    return verified_page_count


def _async_hbm_registration_worker(
    *,
    pages: list[_PendingHBMRegistrationPage],
    registered_versions_by_request: dict[str, dict[int, int]],
    registered_blocks_by_request: dict[str, dict[int, int]],
    registered_token_counts_by_request: dict[str, dict[int, int]],
    block_size: int,
    layer_id: int,
    started_at: float,
    source_ready_event: torch.cuda.Event | None,
    direct_registration: TieredDirectD2HRegistration | None,
    require_verified: bool = False,
) -> None:
    failed = len(pages)
    try:
        if direct_registration is not None:
            raw_failed = _DRAM_KV_PAGE_REGISTRY.complete_from_hbm_direct(
                direct_registration
            )
        else:
            if source_ready_event is not None:
                # The compatibility path creates intermediate CPU tensors.
                # Its copies run outside the dedicated D2H stream, so wait
                # for the phase GreenContext producer before reading HBM.
                source_ready_event.synchronize()
            raw_failed = _register_pending_hbm_pages_sync(pages)
        completed = max(0, len(pages) - raw_failed)
        duration_s = time.perf_counter() - started_at
        if completed:
            _DRAM_KV_PAGE_REGISTRY.record_register_from_hbm(
                completed,
                duration_s=duration_s,
            )
        verified = _mark_hbm_registration_complete(
            registered_versions_by_request=registered_versions_by_request,
            registered_blocks_by_request=registered_blocks_by_request,
            registered_token_counts_by_request=(
                registered_token_counts_by_request
            ),
            block_size=block_size,
            layer_id=layer_id,
            pages=pages,
            require_existing_request=require_verified,
        )
        # A rejected stale reservation is not a correctness failure when a
        # provider-verified newer snapshot covers the same logical page.
        failed = (
            raw_failed
            if verified is None
            else max(0, len(pages) - verified)
        )
        if require_verified and failed:
            raise RuntimeError(
                f"required D2H incomplete: layer={layer_id} "
                f"missing={failed}/{len(pages)}"
            )
    finally:
        _DRAM_KV_PAGE_REGISTRY.record_async_register_from_hbm_complete(
            len(pages),
            duration_s=time.perf_counter() - started_at,
            fail_count=failed,
        )


def _propagate_future_result(
    completed: Future[None],
    destination: Future[None],
) -> None:
    if destination.done():
        return
    try:
        completed.result()
    except BaseException as exc:
        destination.set_exception(exc)
    else:
        destination.set_result(None)


def _d2h_control_dispatch_worker(
    *,
    pages: list[_HBMRegistrationPage],
    registered_versions_by_request: dict[str, dict[int, int]],
    registered_blocks_by_request: dict[str, dict[int, int]],
    registered_token_counts_by_request: dict[str, dict[int, int]],
    block_size: int,
    layer_id: int,
    started_at: float,
    source_ready_event: torch.cuda.Event | None,
    dispatch_submitted_at: float,
    completion_future: Future[None],
    require_verified: bool = False,
) -> None:
    """Reserve pinned slots and enqueue D2H outside attention forward()."""

    control_started_at = time.perf_counter()
    _DRAM_KV_PAGE_REGISTRY.record_d2h_control_dispatch_start(
        control_started_at - dispatch_submitted_at
    )
    failed = False
    async_pending_accounted = False
    try:
        materialize_started_at = time.perf_counter()
        materialized_pages = _materialize_hbm_registration_pages(pages)
        deferred_page_count = sum(
            isinstance(page, _DeferredHBMRegistrationPage) for page in pages
        )
        if deferred_page_count:
            _DRAM_KV_PAGE_REGISTRY.record_d2h_source_view_dispatch_materialize(
                page_count=deferred_page_count,
                duration_s=time.perf_counter() - materialize_started_at,
            )
        direct_registration = (
            _DRAM_KV_PAGE_REGISTRY.enqueue_from_hbm_direct(
                materialized_pages,
                source_ready_event=source_ready_event,
            )
        )
        try:
            worker_future = _get_async_hbm_register_executor().submit(
                _async_hbm_registration_worker,
                pages=materialized_pages,
                registered_versions_by_request=(
                    registered_versions_by_request
                ),
                registered_blocks_by_request=registered_blocks_by_request,
                registered_token_counts_by_request=(
                    registered_token_counts_by_request
                ),
                block_size=block_size,
                layer_id=layer_id,
                started_at=started_at,
                source_ready_event=source_ready_event,
                direct_registration=direct_registration,
                require_verified=require_verified,
            )
        except Exception:
            # If the completion pool rejects the job, finish any already
            # enqueued DMA in this dispatcher thread. Source-page fences remain
            # attached to completion_future until publication finishes.
            _DRAM_KV_PAGE_REGISTRY.record_d2h_control_dispatch_inline_fallback()
            async_pending_accounted = True
            _async_hbm_registration_worker(
                pages=materialized_pages,
                registered_versions_by_request=(
                    registered_versions_by_request
                ),
                registered_blocks_by_request=registered_blocks_by_request,
                registered_token_counts_by_request=(
                    registered_token_counts_by_request
                ),
                block_size=block_size,
                layer_id=layer_id,
                started_at=started_at,
                source_ready_event=source_ready_event,
                direct_registration=direct_registration,
                require_verified=require_verified,
            )
            completion_future.set_result(None)
        else:
            async_pending_accounted = True
            worker_future.add_done_callback(
                lambda completed: _propagate_future_result(
                    completed,
                    completion_future,
                )
            )
    except BaseException as exc:
        failed = True
        if not async_pending_accounted:
            _DRAM_KV_PAGE_REGISTRY.record_async_register_from_hbm_complete(
                len(pages),
                duration_s=time.perf_counter() - started_at,
                fail_count=len(pages),
            )
        if not completion_future.done():
            completion_future.set_exception(exc)
    finally:
        _DRAM_KV_PAGE_REGISTRY.record_d2h_control_dispatch_complete(
            duration_s=time.perf_counter() - control_started_at,
            failed=failed,
        )


def _submit_required_hbm_registration(
    *,
    pages: list[_HBMRegistrationPage],
    registered_versions_by_request: dict[str, dict[int, int]],
    registered_blocks_by_request: dict[str, dict[int, int]],
    registered_token_counts_by_request: dict[str, dict[int, int]],
    block_size: int,
    layer_id: int,
    started_at: float,
    source_ready_event: torch.cuda.Event | None,
    source_block_lease: TieredHBMSourceBlockLease | None,
) -> bool:
    """Retain ALL required pages; advance bounded batches off the model lane.

    The Future covers queued as well as enqueued reads. Cancellation of the
    request therefore cannot recycle any source page before this owner drains.
    Budget waits and compatibility copies occur on the independent worker,
    never as an async-capacity fallback inside attention.forward().
    """
    completion: Future[None] = Future()
    completion.set_running_or_notify_cancel()  # Not externally cancellable.
    completion.add_done_callback(
        lambda _done: _release_hbm_source_block_lease(source_block_lease)
    )
    try:
        _publish_pending_hbm_d2h_fences(pages, completion)
    except BaseException as exc:
        completion.set_exception(exc)
        raise
    limits = [value for value in (
        tiered_dram_register_max_pages_per_forward(),
        tiered_dram_register_async_max_pending_pages(),
    ) if value > 0]
    batch_pages = min(limits) if limits else 1024

    def run_plan() -> None:
        for start in range(0, len(pages), batch_pages):
            batch = pages[start:start + batch_pages]
            required_d2h_progress.reserve_batch(
                lambda: _DRAM_KV_PAGE_REGISTRY.try_reserve_async_register_from_hbm(
                    len(batch), required=True,
                )
            )
            batch_completion: Future[None] = Future()
            _DRAM_KV_PAGE_REGISTRY.record_d2h_control_dispatch_submit()
            _d2h_control_dispatch_worker(
                pages=batch,
                registered_versions_by_request=registered_versions_by_request,
                registered_blocks_by_request=registered_blocks_by_request,
                registered_token_counts_by_request=registered_token_counts_by_request,
                block_size=block_size, layer_id=layer_id, started_at=started_at,
                source_ready_event=source_ready_event,
                dispatch_submitted_at=time.perf_counter(),
                completion_future=batch_completion,
                require_verified=True,
            )
            # Only this dedicated control owner waits, not Engine/model/old
            # dispatcher. Completion has its own worker and releases budget.
            batch_completion.result()

    required_d2h_progress.submit(
        run_plan, completion, len(pages), notify_pd_engine_progress,
    )
    return True


def _submit_dispatched_hbm_registration(
    *,
    pages: list[_HBMRegistrationPage],
    registered_versions_by_request: dict[str, dict[int, int]],
    registered_blocks_by_request: dict[str, dict[int, int]],
    registered_token_counts_by_request: dict[str, dict[int, int]],
    block_size: int,
    layer_id: int,
    started_at: float,
    source_ready_event: torch.cuda.Event | None,
    source_block_lease: TieredHBMSourceBlockLease | None,
) -> bool:
    """Publish source fences, then dispatch D2H control work asynchronously."""

    if not _DRAM_KV_PAGE_REGISTRY.try_reserve_async_register_from_hbm(
        len(pages)
    ):
        return False
    submit_started_at = time.perf_counter()
    completion_future: Future[None] = Future()
    completion_future.add_done_callback(
        lambda _completed: _release_hbm_source_block_lease(source_block_lease)
    )
    try:
        _publish_pending_hbm_d2h_fences(pages, completion_future)
    except Exception as exc:
        _DRAM_KV_PAGE_REGISTRY.cancel_async_register_from_hbm_reservation(
            len(pages)
        )
        completion_future.set_exception(exc)
        raise

    dispatch_submitted_at = time.perf_counter()
    _DRAM_KV_PAGE_REGISTRY.record_d2h_control_dispatch_submit()
    kwargs = {
        "pages": pages,
        "registered_versions_by_request": registered_versions_by_request,
        "registered_blocks_by_request": registered_blocks_by_request,
        "registered_token_counts_by_request": (
            registered_token_counts_by_request
        ),
        "block_size": block_size,
        "layer_id": layer_id,
        "started_at": started_at,
        "source_ready_event": source_ready_event,
        "dispatch_submitted_at": dispatch_submitted_at,
        "completion_future": completion_future,
    }
    try:
        _get_d2h_control_dispatch_executor().submit(
            _d2h_control_dispatch_worker,
            **kwargs,
        )
    except Exception:
        # Executor setup failure is rare. Running the same control function
        # inline preserves the already-published overwrite fence and keeps the
        # fallback behavior correct rather than abandoning source HBM pages.
        _DRAM_KV_PAGE_REGISTRY.record_d2h_control_dispatch_inline_fallback()
        _d2h_control_dispatch_worker(**kwargs)
    finally:
        _DRAM_KV_PAGE_REGISTRY.record_d2h_control_dispatch_submit_duration(
            time.perf_counter() - submit_started_at
        )
    return True


def _submit_async_hbm_registration(
    *,
    pages: list[_HBMRegistrationPage],
    registered_versions_by_request: dict[str, dict[int, int]],
    registered_blocks_by_request: dict[str, dict[int, int]],
    registered_token_counts_by_request: dict[str, dict[int, int]],
    block_size: int,
    layer_id: int,
    started_at: float,
    source_ready_event: torch.cuda.Event | None,
    source_block_lease: TieredHBMSourceBlockLease | None,
) -> bool:
    if not pages:
        return False
    if tiered_dram_register_control_dispatcher_enabled():
        return _submit_dispatched_hbm_registration(
            pages=pages,
            registered_versions_by_request=registered_versions_by_request,
            registered_blocks_by_request=registered_blocks_by_request,
            registered_token_counts_by_request=(
                registered_token_counts_by_request
            ),
            block_size=block_size,
            layer_id=layer_id,
            started_at=started_at,
            source_ready_event=source_ready_event,
            source_block_lease=source_block_lease,
        )
    if not _DRAM_KV_PAGE_REGISTRY.try_reserve_async_register_from_hbm(
        len(pages)
    ):
        return False
    try:
        materialized_pages = _materialize_hbm_registration_pages(pages)
        direct_registration = (
            _DRAM_KV_PAGE_REGISTRY.enqueue_from_hbm_direct(
                materialized_pages,
                source_ready_event=source_ready_event,
            )
        )
    except Exception:
        _DRAM_KV_PAGE_REGISTRY.cancel_async_register_from_hbm_reservation(
            len(pages)
        )
        _release_hbm_source_block_lease(source_block_lease)
        raise
    try:
        future = _get_async_hbm_register_executor().submit(
            _async_hbm_registration_worker,
            pages=materialized_pages,
            registered_versions_by_request=registered_versions_by_request,
            registered_blocks_by_request=registered_blocks_by_request,
            registered_token_counts_by_request=(
                registered_token_counts_by_request
            ),
            block_size=block_size,
            layer_id=layer_id,
            started_at=started_at,
            source_ready_event=source_ready_event,
            direct_registration=direct_registration,
        )
    except Exception:
        if direct_registration is None:
            _DRAM_KV_PAGE_REGISTRY.cancel_async_register_from_hbm_reservation(
                len(pages)
            )
            return False
        # Executor creation/submission failure must not abandon an already
        # enqueued DMA or expose its destination before completion. Finish and
        # publish the direct job synchronously as a correctness fallback.
        try:
            _async_hbm_registration_worker(
                pages=materialized_pages,
                registered_versions_by_request=registered_versions_by_request,
                registered_blocks_by_request=registered_blocks_by_request,
                registered_token_counts_by_request=(
                    registered_token_counts_by_request
                ),
                block_size=block_size,
                layer_id=layer_id,
                started_at=started_at,
                source_ready_event=source_ready_event,
                direct_registration=direct_registration,
            )
        finally:
            _release_hbm_source_block_lease(source_block_lease)
        return True
    future.add_done_callback(
        lambda _completed: _release_hbm_source_block_lease(source_block_lease)
    )
    _publish_pending_hbm_d2h_fences(pages, future)
    return True


def register_dram_kv_page(
    *,
    request_id: str = "",
    request_key: int | None = None,
    layer_name: str,
    layer_id: int | None = None,
    logical_page: int,
    key: torch.Tensor,
    value: torch.Tensor,
    token_start: int,
    token_count: int | None = None,
    version: int = 0,
    copy_to_cpu: bool = True,
) -> None:
    _DRAM_KV_PAGE_REGISTRY.register(
        request_id=request_id,
        request_key=request_key,
        layer_name=layer_name,
        layer_id=layer_id,
        logical_page=logical_page,
        key=key,
        value=value,
        token_start=token_start,
        token_count=token_count,
        version=version,
        copy_to_cpu=copy_to_cpu,
    )


def clear_dram_kv_pages() -> None:
    global _HBM_D2H_FENCE_WAIT_COUNT
    global _HBM_D2H_FENCE_WAIT_PAGE_COUNT
    global _HBM_D2H_FENCE_WAIT_US_TOTAL
    global _HBM_D2H_FENCE_WAIT_US_LAST
    global _DECODE_RELOAD_BARRIER_CHECK_COUNT
    global _DECODE_RELOAD_BARRIER_WAIT_COUNT
    global _DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT
    global _DECODE_RELOAD_BARRIER_WAIT_US_TOTAL
    global _DECODE_RELOAD_BARRIER_WAIT_US_LAST
    global _DECODE_RELOAD_BARRIER_COMPLETE_COUNT
    global _DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT
    global _DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT
    global _DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT
    global _DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT

    with _PENDING_HBM_D2H_FENCES_LOCK:
        pending_futures = set(_PENDING_HBM_D2H_FENCES.values())
    for future in pending_futures:
        future.result()
    with _PENDING_HBM_D2H_FENCES_LOCK:
        _PENDING_HBM_D2H_FENCES.clear()
        _HBM_D2H_FENCE_WAIT_COUNT = 0
        _HBM_D2H_FENCE_WAIT_PAGE_COUNT = 0
        _HBM_D2H_FENCE_WAIT_US_TOTAL = 0
        _HBM_D2H_FENCE_WAIT_US_LAST = 0
        _DECODE_RELOAD_BARRIER_CHECK_COUNT = 0
        _DECODE_RELOAD_BARRIER_WAIT_COUNT = 0
        _DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT = 0
        _DECODE_RELOAD_BARRIER_WAIT_US_TOTAL = 0
        _DECODE_RELOAD_BARRIER_WAIT_US_LAST = 0
        _DECODE_RELOAD_BARRIER_COMPLETE_COUNT = 0
        _DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT = 0
        _DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT = 0
        _DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT = 0
        _DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT = 0
    _DRAM_KV_PAGE_REGISTRY.clear()
    try:
        _fused_kv_scatter_module().clear_fused_kv_scatter_state()
    except Exception:
        # The optional Triton helper may be unavailable in CPU-only tests.
        pass
    cache = getattr(_ATTENTION_PAGE_TENSOR_CACHE_TLS, "cache", None)
    if cache is not None:
        cache.clear()


def get_dram_kv_contiguous_pool_span(
    slot_ids: list[int],
) -> TieredDRAMKVSlotSpan | None:
    return _DRAM_KV_PAGE_REGISTRY.contiguous_pool_span(slot_ids)


def acquire_dram_kv_pool_slots(
    slot_versions: list[tuple[int, int]],
) -> tuple[TieredDRAMKVLease, ...] | None:
    return _DRAM_KV_PAGE_REGISTRY.acquire_pool_slots(slot_versions)


def get_dram_kv_request_key(
    request_id: str,
    *,
    create: bool = False,
) -> int | None:
    return _DRAM_KV_PAGE_REGISTRY.request_key(request_id, create=create)


def lookup_dram_kv_page(
    *,
    request_id: str = "",
    request_key: int | None = None,
    layer_name: str,
    layer_id: int | None = None,
    logical_page: int,
    expected_token_count: int,
    version: int | None = None,
    strict: bool = True,
    allow_newer_version: bool = False,
) -> TieredDRAMKVLookup | None:
    if not tiered_dram_page_provider_enabled():
        return None
    return _DRAM_KV_PAGE_REGISTRY.lookup(
        request_id=request_id,
        request_key=request_key,
        layer_name=layer_name,
        layer_id=layer_id,
        logical_page=logical_page,
        expected_token_count=expected_token_count,
        version=version,
        strict=strict,
        allow_newer_version=allow_newer_version,
    )


def build_dram_kv_page_lookup_plan(
    *,
    request_id: str = "",
    request_key: int | None = None,
    layer_name: str,
    layer_id: int | None = None,
    page_token_counts: dict[int, int],
    page_versions: dict[int, int] | None = None,
    strict_pages: set[int] | None = None,
) -> TieredDRAMKVLookupPlan:
    resolved_layer_id = (
        virtual_context_layer_id(layer_name)
        if layer_id is None
        else layer_id
    )
    if request_key is None:
        request_key = get_dram_kv_request_key(request_id, create=False)
    resolved_request_key = -1 if request_key is None else request_key
    if not tiered_dram_page_provider_enabled():
        return TieredDRAMKVLookupPlan(
            request_id=request_id,
            request_key=resolved_request_key,
            layer_name=layer_name,
            layer_id=resolved_layer_id,
            logical_pages=tuple(sorted(page_token_counts)),
            lookups={},
            missing_pages=tuple(sorted(page_token_counts)),
            expected_page_count=len(page_token_counts),
            byte_count=0,
            pool_hit_count=0,
            dict_hit_count=0,
        )

    lookups: dict[int, TieredDRAMKVLookup] = {}
    missing_pages: list[int] = []
    byte_count = 0
    pool_hit_count = 0
    dict_hit_count = 0
    strict_pages = strict_pages or set()

    for logical_page in sorted(page_token_counts):
        expected_token_count = page_token_counts[logical_page]
        version = (
            page_versions.get(logical_page)
            if page_versions is not None
            else None
        )
        lookup = lookup_dram_kv_page(
            request_id=request_id,
            request_key=request_key,
            layer_name=layer_name,
            layer_id=resolved_layer_id,
            logical_page=logical_page,
            expected_token_count=expected_token_count,
            version=version,
            strict=logical_page in strict_pages,
        )
        if lookup is None:
            missing_pages.append(logical_page)
            continue
        lookups[logical_page] = lookup
        byte_count += (
            lookup.page.key[:expected_token_count].numel()
            * lookup.page.key.element_size()
            + lookup.page.value[:expected_token_count].numel()
            * lookup.page.value.element_size()
        )
        if lookup.source == "pool":
            pool_hit_count += 1
        else:
            dict_hit_count += 1

    return TieredDRAMKVLookupPlan(
        request_id=request_id,
        request_key=resolved_request_key,
        layer_name=layer_name,
        layer_id=resolved_layer_id,
        logical_pages=tuple(sorted(page_token_counts)),
        lookups=lookups,
        missing_pages=tuple(missing_pages),
        expected_page_count=len(page_token_counts),
        byte_count=byte_count,
        pool_hit_count=pool_hit_count,
        dict_hit_count=dict_hit_count,
    )


def build_tiered_attention_page_tensor_plan(
    *,
    request_id: str = "",
    layer_name: str,
    layer_id: int | None = None,
    page_token_counts: dict[int, int],
    block_ids_by_page: dict[int, int],
    dram_plan: TieredDRAMKVLookupPlan,
    required_dram_pages: set[int] | None = None,
    block_size: int,
    request_numeric_id: int = -1,
    device: torch.device | str | None = None,
) -> TieredAttentionPageTensorPlan:
    """Build the tensor-shaped page traversal plan for context ring.

    The reference backend still consumes ``TieredKVBlock`` objects. This plan
    is the compatibility boundary for the optimized backend: a future fused
    kernel can consume these numeric tensors without redoing provider lookup in
    the attention loop.
    """
    resolved_layer_id = (
        virtual_context_layer_id(layer_name)
        if layer_id is None
        else layer_id
    )
    logical_pages = tuple(sorted(page_token_counts))
    required_dram_pages = required_dram_pages or set()
    selected_dram_pages = (
        set(required_dram_pages)
        if required_dram_pages
        else set(dram_plan.lookups)
    )
    tensor_device = device if device is not None else torch.device("cpu")

    page_tiers: list[int] = []
    hbm_block_ids: list[int] = []
    dram_slot_ids: list[int] = []
    provider_versions: list[int] = []
    token_starts: list[int] = []
    token_counts: list[int] = []
    dram_source_types: list[int] = []
    hbm_page_count = 0
    dram_page_count = 0
    pool_page_count = 0
    dict_page_count = 0
    missing_required_pages: list[int] = []

    for logical_page in logical_pages:
        token_count = page_token_counts[logical_page]
        token_starts.append(logical_page * block_size)
        token_counts.append(token_count)
        lookup = (
            dram_plan.lookups.get(logical_page)
            if logical_page in selected_dram_pages
            else None
        )
        if lookup is not None:
            page_tiers.append(TIERED_ATTENTION_PAGE_TIER_DRAM)
            hbm_block_ids.append(-1)
            dram_slot_ids.append(lookup.slot_id)
            provider_versions.append(lookup.version)
            if lookup.source == "pool":
                dram_source_types.append(TIERED_ATTENTION_PAGE_SOURCE_POOL)
                pool_page_count += 1
            else:
                dram_source_types.append(TIERED_ATTENTION_PAGE_SOURCE_DICT)
                dict_page_count += 1
            dram_page_count += 1
            continue

        if logical_page in required_dram_pages:
            missing_required_pages.append(logical_page)
        page_tiers.append(TIERED_ATTENTION_PAGE_TIER_HBM)
        hbm_block_ids.append(block_ids_by_page.get(logical_page, -1))
        dram_slot_ids.append(-1)
        provider_versions.append(-1)
        dram_source_types.append(TIERED_ATTENTION_PAGE_SOURCE_NONE)
        hbm_page_count += 1

    logical_pages_host = tuple(logical_pages)
    page_tiers_host = tuple(page_tiers)
    hbm_block_ids_host = tuple(hbm_block_ids)
    dram_slot_ids_host = tuple(dram_slot_ids)
    provider_versions_host = tuple(provider_versions)
    token_starts_host = tuple(token_starts)
    token_counts_host = tuple(token_counts)
    dram_source_types_host = tuple(dram_source_types)
    columns = (
        logical_pages_host,
        page_tiers_host,
        hbm_block_ids_host,
        dram_slot_ids_host,
        provider_versions_host,
        token_starts_host,
        token_counts_host,
        dram_source_types_host,
    )
    rows_host = tuple(zip(*columns))
    cache_key = (
        str(torch.device(tensor_device)),
        int(dram_plan.request_key),
        int(resolved_layer_id),
    )
    (
        plan_tensors,
        cache_hit,
        cache_reallocated,
        cache_delta_pages,
        cache_copied_pages,
    ) = _cached_page_plan_tensors(
        cache_key=cache_key,
        rows_host=rows_host,
        columns=columns,
        device=tensor_device,
    )
    (
        logical_pages_tensor,
        page_tiers_tensor,
        hbm_block_ids_tensor,
        dram_slot_ids_tensor,
        provider_versions_tensor,
        token_starts_tensor,
        token_counts_tensor,
        dram_source_types_tensor,
    ) = plan_tensors

    return TieredAttentionPageTensorPlan(
        request_id=request_id,
        request_key=dram_plan.request_key,
        request_numeric_id=request_numeric_id,
        layer_name=layer_name,
        layer_id=resolved_layer_id,
        logical_pages=logical_pages_tensor,
        page_tiers=page_tiers_tensor,
        hbm_block_ids=hbm_block_ids_tensor,
        dram_slot_ids=dram_slot_ids_tensor,
        provider_versions=provider_versions_tensor,
        token_starts=token_starts_tensor,
        token_counts=token_counts_tensor,
        dram_source_types=dram_source_types_tensor,
        logical_pages_host=logical_pages_host,
        page_tiers_host=page_tiers_host,
        hbm_block_ids_host=hbm_block_ids_host,
        dram_slot_ids_host=dram_slot_ids_host,
        provider_versions_host=provider_versions_host,
        token_starts_host=token_starts_host,
        token_counts_host=token_counts_host,
        dram_source_types_host=dram_source_types_host,
        hbm_page_count=hbm_page_count,
        dram_page_count=dram_page_count,
        pool_page_count=pool_page_count,
        dict_page_count=dict_page_count,
        missing_required_pages=tuple(missing_required_pages),
        cache_hit=cache_hit,
        cache_reallocated=cache_reallocated,
        cache_delta_pages=cache_delta_pages,
        cache_copied_pages=cache_copied_pages,
    )


def get_dram_kv_page_as_block(
    *,
    request_id: str = "",
    request_key: int | None = None,
    layer_name: str,
    layer_id: int | None = None,
    logical_page: int,
    expected_token_count: int,
    version: int | None = None,
) -> TieredKVBlock | None:
    lookup = lookup_dram_kv_page(
        request_id=request_id,
        request_key=request_key,
        layer_name=layer_name,
        layer_id=layer_id,
        logical_page=logical_page,
        expected_token_count=expected_token_count,
        version=version,
    )
    if lookup is None:
        return None
    page = lookup.page
    return TieredKVBlock(
        key=page.key[:expected_token_count],
        value=page.value[:expected_token_count],
        token_start=page.token_start,
        tier="DRAM",
        logical_page=logical_page,
    )


def reload_dram_kv_pages_to_hbm(
    *,
    layer_name: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    request_ids: list[str] | None,
    block_size: int,
    page_versions_by_request: dict[str, dict[int, int]] | None,
    reload_pages_by_request: dict[str, set[int]] | None,
    expected_token_counts_by_request: (
        dict[str, dict[int, int]] | None
    ) = None,
    hbm_token_counts_by_request: dict[str, dict[int, int]] | None = None,
    block_table_cpu: torch.Tensor | None = None,
    seq_lens_cpu: torch.Tensor | None = None,
    query_start_loc_cpu: torch.Tensor | None = None,
    block_table_is_fully_concrete: bool = False,
    diagnostics: dict[str, int | str] | None = None,
) -> dict[str, set[int]]:
    """Copy provider-backed DRAM K/V pages into allocated HBM KV blocks.

    This is the inverse of ``register_hbm_kv_cache_pages`` and is intentionally
    conservative: it only copies when every requested logical page for a
    request maps to a distinct HBM block id. Repeated block ids normally mean
    the scheduler installed virtual/null placeholders, in which case native
    attention is not safe and the caller should keep using the tiered backend.
    ``expected_token_counts_by_request`` freezes the provider-owned prefix of
    a growable partial tail before the current Prefill query extends the HBM
    page. ``hbm_token_counts_by_request`` freezes that distinct post-query
    logical page size from the same authoritative attention metadata.
    ``block_table_is_fully_concrete`` is reserved for explicit native reload
    (Prefill promotion or Decode handoff) after the scheduler has validated
    and replaced every logical null;
    in that state unused row padding must not be interpreted as a null id.
    """
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "reason": "not_started",
            "requested_pages": sum(
                len(pages)
                for pages in (reload_pages_by_request or {}).values()
            ),
            "reloaded_pages": 0,
            "failed_pages": 0,
        })
    if not (
        tiered_dram_page_provider_enabled()
        and tiered_dram_reload_to_hbm_enabled()
    ):
        if diagnostics is not None:
            diagnostics["reason"] = "reload_feature_disabled"
        return {}
    if (
        not layer_name
        or request_ids is None
        or not reload_pages_by_request
    ):
        if diagnostics is not None:
            diagnostics["reason"] = "missing_layer_or_request_metadata"
        return {}
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        if diagnostics is not None:
            diagnostics["reason"] = "invalid_kv_cache_rank"
        return {}
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        if diagnostics is not None:
            diagnostics["reason"] = "kv_cache_shape_mismatch"
        return {}

    layer_id = virtual_context_layer_id(layer_name)
    host_block_table = (
        block_table_cpu if block_table_cpu is not None else block_table
    )
    host_seq_lens = seq_lens_cpu if seq_lens_cpu is not None else seq_lens
    host_query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else query_start_loc
    )
    cpu_block_table_used = not host_block_table.is_cuda
    num_reqs = min(
        len(request_ids),
        int(host_seq_lens.numel()),
        int(host_block_table.shape[0]),
    )
    if num_reqs <= 0:
        if diagnostics is not None:
            diagnostics["reason"] = "empty_request_batch"
        return {}

    cache_block_size = int(key_cache.shape[1])
    if int(block_size) != cache_block_size:
        raise ValueError(
            "tiered KV reload block-size mismatch: "
            f"metadata={block_size}, cache={cache_block_size}"
        )
    block_size = cache_block_size
    reloaded_pages_by_request: dict[str, set[int]] = {}
    reloaded_blocks_by_request: dict[str, dict[int, int]] = {}
    reloaded_token_counts_by_request: dict[str, dict[int, int]] = {}
    reloaded_provider_token_counts_by_request: dict[
        str, dict[int, int]
    ] = {}
    reloaded_provider_versions_by_request: dict[str, dict[int, int]] = {}
    reloaded_pages = 0
    failed_pages = 0
    request_key_missing_pages = 0
    invalid_sequence_pages = 0
    invalid_block_table_pages = 0
    invalid_token_count_pages = 0
    provider_lookup_miss_pages = 0
    newer_version_fallback_pages = 0
    pool_lease_failure_pages = 0
    explicit_token_count_pages = 0
    first_invalid_block_table_detail = ""
    reloaded_bytes = 0
    reload_source_leases: list[TieredDRAMKVLease] = []
    plan_s = 0.0
    lookup_s = 0.0
    enqueue_s = 0.0
    copy_ops = 0
    page_copy_ops = 0
    bulk_copy_ops = 0
    bulk_pages = 0
    bulk_bytes = 0
    direct_multicopy_ops = 0
    direct_multicopy_runs = 0
    direct_multicopy_pages = 0
    direct_multicopy_bytes = 0
    staging_scatter_ops = 0
    staging_scatter_pages = 0
    staging_scatter_bytes = 0
    staging_scatter_fallback_pages = 0
    reload_ack_wait_s = 0.0
    bulk_tokens = tiered_dram_reload_bulk_tokens()
    max_bulk_pages = max(1, bulk_tokens // block_size)
    staging_scatter_enabled = (
        tiered_dram_reload_staging_scatter_enabled()
        and key_cache.is_cuda
        and max_bulk_pages > 1
    )
    direct_multicopy_enabled = (
        tiered_dram_reload_direct_multicopy_enabled()
        and key_cache.is_cuda
        and max_bulk_pages > 1
    )
    fused_scatter_enabled = (
        staging_scatter_enabled
        and tiered_dram_reload_fused_scatter_enabled()
    )

    for req_idx in range(num_reqs):
        request_id = request_ids[req_idx]
        requested_pages = reload_pages_by_request.get(request_id)
        if not requested_pages:
            continue
        request_key = get_dram_kv_request_key(request_id, create=False)
        if request_key is None:
            failed_pages += len(requested_pages)
            request_key_missing_pages += len(requested_pages)
            continue

        plan_started = time.perf_counter()
        seq_len_value = host_seq_lens[req_idx]
        seq_len = int(
            seq_len_value.item()
            if hasattr(seq_len_value, "item")
            else seq_len_value
        )
        if seq_len <= 0:
            failed_pages += len(requested_pages)
            invalid_sequence_pages += len(requested_pages)
            plan_s += time.perf_counter() - plan_started
            continue
        query_len = 0
        if (
            host_query_start_loc is not None
            and int(host_query_start_loc.numel()) > req_idx + 1
        ):
            query_start = host_query_start_loc[req_idx]
            query_end = host_query_start_loc[req_idx + 1]
            query_len = int(
                (query_end - query_start).item()
                if hasattr(query_end, "item")
                else query_end - query_start
            )
        historical_tokens = max(0, seq_len - query_len)

        row = host_block_table[req_idx]
        page_versions = (
            page_versions_by_request.get(request_id)
            if page_versions_by_request is not None
            else None
        )
        explicit_token_counts = (
            expected_token_counts_by_request.get(request_id)
            if expected_token_counts_by_request is not None
            else None
        )
        explicit_hbm_token_counts = (
            hbm_token_counts_by_request.get(request_id)
            if hbm_token_counts_by_request is not None
            else None
        )
        num_valid_pages = min(
            int(row.numel()),
            (seq_len + block_size - 1) // block_size,
        )
        row_value_count = min(
            int(row.numel()),
            num_valid_pages + int(num_valid_pages < int(row.numel())),
        )
        row_values = row[:row_value_count].detach()
        if row_values.is_cuda:
            # Use one metadata transfer on the fallback path instead of one
            # synchronizing CUDA scalar read per logical page.
            row_values = row_values.to(device="cpu")
        row_block_ids = [int(value) for value in row_values.tolist()]
        # An unused block-table cell is storage padding, not an allocator null
        # sentinel. Under explicit native reload the scheduler has already
        # replaced every logical null with a concrete block, so comparing a
        # valid destination against the first padding cell can falsely reject
        # a correct table when that stale/padded value happens to match. Keep
        # the legacy sentinel guard only for callers that have not established
        # the whole-table concrete invariant.
        null_block_id = None
        if not block_table_is_fully_concrete:
            null_block_id = (
                row_block_ids[num_valid_pages]
                if num_valid_pages < len(row_block_ids)
                else None
            )

        page_to_block: dict[int, int] = {}
        duplicate_or_invalid = False
        if block_table_is_fully_concrete:
            concrete_block_ids: set[int] = set()
            for logical_page, block_id in enumerate(
                row_block_ids[:num_valid_pages]
            ):
                if (
                    block_id < 0
                    or block_id >= int(key_cache.shape[0])
                    or block_id in concrete_block_ids
                ):
                    duplicate_or_invalid = True
                    if not first_invalid_block_table_detail:
                        invalid_reason = (
                            "block_range"
                            if block_id < 0
                            or block_id >= int(key_cache.shape[0])
                            else "duplicate_concrete_destination"
                        )
                        first_invalid_block_table_detail = (
                            f"request={request_id!r} page={logical_page} "
                            f"block={block_id} reason={invalid_reason}"
                        )
                    break
                concrete_block_ids.add(block_id)

        seen_block_ids: set[int] = set()
        pages_to_validate = (
            () if duplicate_or_invalid else sorted(requested_pages)
        )
        for logical_page in pages_to_validate:
            if (
                logical_page < 0
                or logical_page >= num_valid_pages
                or logical_page >= len(row_block_ids)
            ):
                duplicate_or_invalid = True
                if not first_invalid_block_table_detail:
                    first_invalid_block_table_detail = (
                        f"request={request_id!r} page={logical_page} "
                        f"valid_pages={num_valid_pages} "
                        f"row_values={len(row_block_ids)} reason=page_range"
                    )
                break
            block_id = row_block_ids[logical_page]
            if (
                block_id < 0
                or block_id >= int(key_cache.shape[0])
                or block_id in seen_block_ids
                or (
                    null_block_id is not None
                    and block_id == null_block_id
                )
            ):
                duplicate_or_invalid = True
                if not first_invalid_block_table_detail:
                    if block_id < 0 or block_id >= int(key_cache.shape[0]):
                        invalid_reason = "block_range"
                    elif block_id in seen_block_ids:
                        invalid_reason = "duplicate_destination"
                    else:
                        invalid_reason = "inferred_null_sentinel"
                    first_invalid_block_table_detail = (
                        f"request={request_id!r} page={logical_page} "
                        f"block={block_id} reason={invalid_reason}"
                    )
                break
            seen_block_ids.add(block_id)
            page_to_block[logical_page] = block_id

        if duplicate_or_invalid:
            failed_pages += len(requested_pages)
            invalid_block_table_pages += len(requested_pages)
            plan_s += time.perf_counter() - plan_started
            continue

        plan_s += time.perf_counter() - plan_started

        reload_entries: list[_TieredH2DReloadPage] = []
        lookup_started = time.perf_counter()
        for logical_page, block_id in page_to_block.items():
            token_start = logical_page * block_size
            explicit_token_count = (
                explicit_token_counts.get(logical_page)
                if explicit_token_counts is not None
                else None
            )
            if explicit_token_count is None:
                expected_token_count = min(
                    block_size,
                    historical_tokens - token_start,
                )
            else:
                expected_token_count = int(explicit_token_count)
                explicit_token_count_pages += 1
            explicit_hbm_token_count = (
                explicit_hbm_token_counts.get(logical_page)
                if explicit_hbm_token_counts is not None
                else None
            )
            hbm_token_count = (
                min(block_size, seq_len - token_start)
                if explicit_hbm_token_count is None
                else int(explicit_hbm_token_count)
            )
            if not (
                0 < expected_token_count <= hbm_token_count <= block_size
            ):
                failed_pages += 1
                invalid_token_count_pages += 1
                continue

            version = None
            if page_versions is not None:
                version = page_versions.get(logical_page)
            lookup = _DRAM_KV_PAGE_REGISTRY.lookup(
                request_id=request_id,
                request_key=request_key,
                layer_name=layer_name,
                layer_id=layer_id,
                logical_page=logical_page,
                expected_token_count=expected_token_count,
                version=version,
                allow_newer_version=True,
            )
            if lookup is None:
                failed_pages += 1
                provider_lookup_miss_pages += 1
                continue
            if version is not None and lookup.version > version:
                newer_version_fallback_pages += 1
            reload_entries.append(
                _TieredH2DReloadPage(
                    logical_page=logical_page,
                    block_id=block_id,
                    expected_token_count=expected_token_count,
                    hbm_token_count=hbm_token_count,
                    lookup=lookup,
                )
            )
        lookup_s += time.perf_counter() - lookup_started

        pool_entries = [
            entry
            for entry in reload_entries
            if entry.lookup.source == "pool"
        ]
        if pool_entries:
            leases = _DRAM_KV_PAGE_REGISTRY.acquire_pool_slots(
                [
                    (entry.lookup.slot_id, entry.lookup.version)
                    for entry in pool_entries
                ]
            )
            if leases is None:
                failed_pages += len(pool_entries)
                pool_lease_failure_pages += len(pool_entries)
                reload_entries = [
                    entry
                    for entry in reload_entries
                    if entry.lookup.source != "pool"
                ]
            else:
                reload_source_leases.extend(leases)

        enqueue_started = time.perf_counter()
        request_ack_wait_s = 0.0
        with acquire_native_reload_priority():
            for span in _coalesce_h2d_source_spans(
                reload_entries,
                block_size=block_size,
                max_pages=max_bulk_pages,
            ):
                copied_as_span = False
                source_span = None
                if len(span) > 1:
                    source_span = (
                        _DRAM_KV_PAGE_REGISTRY.contiguous_pool_span(
                            [entry.lookup.slot_id for entry in span]
                        )
                    )
                destination_contiguous = all(
                    current.block_id == previous.block_id + 1
                    for previous, current in zip(span, span[1:])
                )
                if (
                    source_span is not None
                    and source_span.token_count == len(span) * block_size
                    and destination_contiguous
                ):
                    first_block = span[0].block_id
                    last_block = span[-1].block_id + 1
                    destination_shape = (
                        source_span.token_count,
                        *key_cache.shape[2:],
                    )
                    destination_key = key_cache[
                        first_block:last_block
                    ].reshape(destination_shape)
                    destination_value = value_cache[
                        first_block:last_block
                    ].reshape(destination_shape)
                    with torch.no_grad():
                        destination_key.copy_(
                            source_span.key,
                            non_blocking=True,
                        )
                        destination_value.copy_(
                            source_span.value,
                            non_blocking=True,
                        )
                    span_bytes = (
                        source_span.key.numel()
                        * source_span.key.element_size()
                        + source_span.value.numel()
                        * source_span.value.element_size()
                    )
                    copy_ops += 2
                    bulk_copy_ops += 2
                    bulk_pages += len(span)
                    bulk_bytes += span_bytes
                    copied_as_span = True
                elif (
                    direct_multicopy_enabled
                    and source_span is not None
                    and source_span.token_count == len(span) * block_size
                ):
                    destination_blocks_cpu = torch.tensor(
                        [entry.block_id for entry in span],
                        dtype=torch.int64,
                        device="cpu",
                    )
                    with torch.no_grad():
                        run_count = int(
                            _custom_ops_module().tiered_h2d_multicopy(
                                source_span.key,
                                source_span.value,
                                key_cache,
                                value_cache,
                                destination_blocks_cpu,
                                block_size,
                            )
                        )
                    span_bytes = (
                        source_span.key.numel()
                        * source_span.key.element_size()
                        + source_span.value.numel()
                        * source_span.value.element_size()
                    )
                    multicopy_ops = 2 * run_count
                    copy_ops += multicopy_ops
                    bulk_copy_ops += multicopy_ops
                    bulk_pages += len(span)
                    bulk_bytes += span_bytes
                    direct_multicopy_ops += multicopy_ops
                    direct_multicopy_runs += run_count
                    direct_multicopy_pages += len(span)
                    direct_multicopy_bytes += span_bytes
                    copied_as_span = True
                elif (
                    staging_scatter_enabled
                    and source_span is not None
                    and source_span.token_count == len(span) * block_size
                ):
                    staging_slots = context_ring_staging_slots()
                    staging_buffer = get_shared_hbm_staging_buffer(
                        key_cache.device,
                        staging_slots,
                    )
                    staging_slot = next_shared_hbm_staging_slot(
                        key_cache.device,
                        staging_slots,
                    )
                    destination_block_ids = tuple(
                        entry.block_id for entry in span
                    )
                    with torch.no_grad():
                        staged_key, staged_value, span_bytes = (
                            staging_buffer.materialize_group(
                                [(source_span.key, source_span.value)],
                                target_device=key_cache.device,
                                slot_index=staging_slot,
                                non_blocking=True,
                            )
                        )
                        staging_buffer.mark_slot_computing(staging_slot)
                        fused_scatter_used = False
                        if fused_scatter_enabled:
                            try:
                                scatter_module = _fused_kv_scatter_module()
                                fused_scatter_used = (
                                    scatter_module.fused_scatter_staged_kv_pages(
                                        staged_key=staged_key,
                                        staged_value=staged_value,
                                        key_cache=key_cache,
                                        value_cache=value_cache,
                                        destination_block_ids=(
                                            destination_block_ids
                                        ),
                                    )
                                )
                            except Exception:
                                if tiered_dram_reload_fused_scatter_strict():
                                    raise
                                fused_scatter_used = False
                        if not fused_scatter_used:
                            destination_blocks = torch.tensor(
                                destination_block_ids,
                                dtype=torch.long,
                                device=key_cache.device,
                            )
                            page_shape = (
                                len(span),
                                block_size,
                                *key_cache.shape[2:],
                            )
                            key_cache.index_copy_(
                                0,
                                destination_blocks,
                                staged_key.view(page_shape),
                            )
                            value_cache.index_copy_(
                                0,
                                destination_blocks,
                                staged_value.view(page_shape),
                            )
                        staging_buffer.mark_slot_read_done(
                            staging_slot,
                            key_cache.device,
                        )
                    copy_ops += 3 if fused_scatter_used else 4
                    bulk_copy_ops += 2
                    bulk_pages += len(span)
                    bulk_bytes += span_bytes
                    staging_scatter_ops += 1 if fused_scatter_used else 2
                    staging_scatter_pages += len(span)
                    staging_scatter_bytes += span_bytes
                    copied_as_span = True

                if not copied_as_span:
                    staging_scatter_fallback_pages += len(span)
                    for entry in span:
                        page = entry.lookup.page
                        token_count = entry.expected_token_count
                        with torch.no_grad():
                            key_cache[entry.block_id, :token_count].copy_(
                                page.key[:token_count],
                                non_blocking=True,
                            )
                            value_cache[entry.block_id, :token_count].copy_(
                                page.value[:token_count],
                                non_blocking=True,
                            )
                        copy_ops += 2
                        page_copy_ops += 2

                for entry in span:
                    logical_page = entry.logical_page
                    token_start = logical_page * block_size
                    page = entry.lookup.page
                    token_count = entry.expected_token_count
                    reloaded_pages_by_request.setdefault(
                        request_id, set()
                    ).add(logical_page)
                    reloaded_blocks_by_request.setdefault(request_id, {})[
                        logical_page
                    ] = entry.block_id
                    reloaded_token_counts_by_request.setdefault(
                        request_id, {}
                    )[logical_page] = entry.hbm_token_count
                    reloaded_provider_token_counts_by_request.setdefault(
                        request_id, {}
                    )[logical_page] = token_count
                    reloaded_provider_versions_by_request.setdefault(
                        request_id, {}
                    )[logical_page] = entry.lookup.version
                    reloaded_pages += 1
                    reloaded_bytes += (
                        page.key[:token_count].numel()
                        * page.key.element_size()
                        + page.value[:token_count].numel()
                        * page.value.element_size()
                    )
            if reloaded_blocks_by_request and key_cache.is_cuda:
                reload_complete = torch.cuda.Event(enable_timing=False)
                reload_complete.record(
                    torch.cuda.current_stream(key_cache.device)
                )
                wait_started = time.perf_counter()
                reload_complete.synchronize()
                request_ack_wait_s = time.perf_counter() - wait_started
                reload_ack_wait_s += request_ack_wait_s
        enqueue_s += max(
            0.0,
            time.perf_counter() - enqueue_started - request_ack_wait_s,
        )

    # The copies above are asynchronous.  Native attention in this stream is
    # naturally ordered after them, but the residency table is shared with the
    # scheduler and the other P/D lane.  Do not publish HBM-ready ownership
    # until the copy stream has acknowledged every page for this layer.
    for lease in reload_source_leases:
        lease.release()

    if reloaded_pages or failed_pages:
        _DRAM_KV_PAGE_REGISTRY.record_reload_to_hbm(
            page_count=reloaded_pages,
            byte_count=reloaded_bytes,
            fail_count=failed_pages,
            ack_wait_s=reload_ack_wait_s,
            cpu_block_table_used=cpu_block_table_used,
            plan_s=plan_s,
            lookup_s=lookup_s,
            enqueue_s=enqueue_s,
            copy_ops=copy_ops,
            page_copy_ops=page_copy_ops,
            bulk_copy_ops=bulk_copy_ops,
            bulk_pages=bulk_pages,
            bulk_bytes=bulk_bytes,
            direct_multicopy_ops=direct_multicopy_ops,
            direct_multicopy_runs=direct_multicopy_runs,
            direct_multicopy_pages=direct_multicopy_pages,
            direct_multicopy_bytes=direct_multicopy_bytes,
            staging_scatter_ops=staging_scatter_ops,
            staging_scatter_pages=staging_scatter_pages,
            staging_scatter_bytes=staging_scatter_bytes,
            staging_scatter_fallback_pages=(
                staging_scatter_fallback_pages
            ),
            request_key_missing_pages=request_key_missing_pages,
            invalid_sequence_pages=invalid_sequence_pages,
            invalid_block_table_pages=invalid_block_table_pages,
            invalid_token_count_pages=invalid_token_count_pages,
            provider_lookup_miss_pages=provider_lookup_miss_pages,
            newer_version_fallback_pages=newer_version_fallback_pages,
            pool_lease_failure_pages=pool_lease_failure_pages,
        )
    if reloaded_blocks_by_request:
        virtual_context = get_virtual_context_table()
        if virtual_context is not None:
            for request_id, page_to_block in reloaded_blocks_by_request.items():
                virtual_context.mark_hbm_reload_complete(
                    request_id=request_id,
                    page_to_block=page_to_block,
                    block_size=block_size,
                    group_id=0,
                    layer_id=layer_id,
                    page_token_counts=reloaded_token_counts_by_request.get(
                        request_id
                    ),
                    provider_token_counts=(
                        reloaded_provider_token_counts_by_request.get(
                            request_id
                        )
                    ),
                    provider_versions=(
                        reloaded_provider_versions_by_request.get(request_id)
                    ),
                )
    if diagnostics is not None:
        diagnostics.update({
            "reason": "complete" if failed_pages == 0 else "incomplete",
            "reloaded_pages": reloaded_pages,
            "failed_pages": failed_pages,
            "request_key_missing_pages": request_key_missing_pages,
            "invalid_sequence_pages": invalid_sequence_pages,
            "invalid_block_table_pages": invalid_block_table_pages,
            "invalid_token_count_pages": invalid_token_count_pages,
            "provider_lookup_miss_pages": provider_lookup_miss_pages,
            "explicit_token_count_pages": explicit_token_count_pages,
            "newer_version_fallback_pages": newer_version_fallback_pages,
            "pool_lease_failure_pages": pool_lease_failure_pages,
            "block_table_is_fully_concrete": int(
                block_table_is_fully_concrete
            ),
        })
        if first_invalid_block_table_detail:
            diagnostics["first_invalid_block_table_detail"] = (
                first_invalid_block_table_detail
            )
    return reloaded_pages_by_request


def register_hbm_kv_cache_pages(
    *,
    layer_name: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    request_ids: list[str] | None,
    page_versions_by_request: dict[str, dict[int, int]] | None,
    allowed_request_ids: set[str] | None = None,
    force_all_request_ids: set[str] | None = None,
    exact_force_pages_by_request: dict[str, set[int]] | None = None,
    source_page_plan: TieredD2HSourcePlan | None = None,
) -> int:
    """Register current HBM KV pages in the debug DRAM provider.

    This is an intentionally expensive validation bridge. It copies the
    selected HBM K/V cache pages into the in-process CPU/DRAM registry so the
    guarded tiered backend can later materialize those pages via the provider.
    It does not release HBM blocks or change scheduling state.
    """
    if not (
        tiered_dram_page_provider_enabled()
        and tiered_dram_register_from_hbm_enabled()
    ):
        return 0
    started_at = time.perf_counter()
    if not layer_name or request_ids is None:
        return 0
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        return 0
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        return 0

    layer_id = virtual_context_layer_id(layer_name)
    num_reqs = min(
        len(request_ids),
        int(seq_lens.numel()),
        int(block_table.shape[0]),
    )
    block_size = int(key_cache.shape[1])
    copy_policy = tiered_dram_register_from_hbm_policy()
    max_pages_per_forward = tiered_dram_register_max_pages_per_forward()
    refresh_probe = isolation_execution_enabled("DRAM_REFRESH_PROBE")
    required_progress = bool(exact_force_pages_by_request) and (
        isolation_execution_enabled("D2H_EXACT_PLAN_PROGRESS")
    )
    use_source_page_plan = (
        source_page_plan is not None
        and source_page_plan.matches(
            request_ids[:num_reqs],
            block_size,
            allowed_request_ids,
            force_all_request_ids,
        )
    )
    source_pages_by_request = (
        source_page_plan.group_by_request()
        if use_source_page_plan and source_page_plan is not None
        else None
    )
    if tiered_d2h_source_plan_enabled():
        _DRAM_KV_PAGE_REGISTRY.record_d2h_source_plan(
            used=use_source_page_plan,
            page_count=(
                len(source_page_plan.pages)
                if use_source_page_plan and source_page_plan is not None
                else 0
            ),
        )
    scheduled_pages_by_request: dict[str, set[int]] | None = None
    if copy_policy == "scheduled" and not use_source_page_plan:
        scheduled_pages_by_request = _scheduled_pages_by_request(
            block_table=block_table,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            request_ids=request_ids,
            num_reqs=num_reqs,
            block_size=block_size,
        )
        if not scheduled_pages_by_request and not force_all_request_ids:
            return 0

    registered = 0
    registered_versions_by_request: dict[str, dict[int, int]] = {}
    registered_blocks_by_request: dict[str, dict[int, int]] = {}
    registered_token_counts_by_request: dict[str, dict[int, int]] = {}
    async_register = tiered_dram_register_async_enabled()
    if required_progress and not (
        async_register and tiered_hbm_source_block_lease_enabled()
    ):
        raise RuntimeError(
            "D2H_EXACT_PLAN_PROGRESS requires async registration and source leases"
        )
    pending_pages: list[_HBMRegistrationPage] = []
    source_view_dispatch_requested = tiered_d2h_source_view_dispatch_enabled()
    defer_source_views = (
        source_view_dispatch_requested
        and async_register
        and tiered_dram_register_control_dispatcher_enabled()
        and tiered_hbm_source_block_lease_enabled()
    )
    if source_view_dispatch_requested and not defer_source_views:
        _DRAM_KV_PAGE_REGISTRY.record_d2h_source_view_dispatch_fallback()
    virtual_context = get_virtual_context_table()
    exact_force_pages = exact_force_pages_by_request or {}
    for req_idx in range(num_reqs):
        request_id = request_ids[req_idx]
        if (
            allowed_request_ids is not None
            and request_id not in allowed_request_ids
        ):
            continue
        source_pages = (
            source_pages_by_request.get(request_id, ())
            if source_pages_by_request is not None
            else ()
        )
        if use_source_page_plan:
            if not source_pages:
                continue
            seq_len = max(
                page.token_start + page.token_count for page in source_pages
            )
        else:
            seq_len_value = seq_lens[req_idx]
            seq_len = int(
                seq_len_value.item()
                if hasattr(seq_len_value, "item")
                else seq_len_value
            )
        if seq_len <= 0:
            continue
        force_all_pages = (
            any(page.force_all for page in source_pages)
            if use_source_page_plan
            else (
                force_all_request_ids is not None
                and request_id in force_all_request_ids
            )
        )
        request_exact_force_pages = exact_force_pages.get(request_id)
        required_request = (
            required_progress and force_all_pages
            and request_exact_force_pages is not None
        )
        scheduled_pages = None
        if scheduled_pages_by_request is not None and not force_all_pages:
            scheduled_pages = scheduled_pages_by_request.get(request_id)
        if scheduled_pages_by_request is not None and not scheduled_pages:
            if not force_all_pages:
                continue
        if virtual_context is not None:
            # Register the expected layer synchronously, before an async copy
            # worker can report completion.  HBM release then requires every
            # layer observed during this forward to have a current DRAM copy.
            virtual_context.observe_provider_layer(request_id, layer_id)
        request_key = get_dram_kv_request_key(request_id, create=True)
        assert request_key is not None
        page_versions = (
            page_versions_by_request.get(request_id)
            if page_versions_by_request is not None
            else None
        )
        page_candidates = (
            source_pages
            if use_source_page_plan
            else range((seq_len + block_size - 1) // block_size)
        )
        for page_candidate in page_candidates:
            if (
                max_pages_per_forward > 0
                and registered >= max_pages_per_forward
                and not required_request
            ):
                break
            logical_page = (
                page_candidate.logical_page
                if use_source_page_plan
                else int(page_candidate)
            )
            if (
                force_all_pages
                and request_exact_force_pages is not None
                and logical_page not in request_exact_force_pages
            ):
                continue
            if scheduled_pages is not None and logical_page not in scheduled_pages:
                continue
            if use_source_page_plan:
                block_id = page_candidate.block_id
                token_start = page_candidate.token_start
                token_count = page_candidate.token_count
            else:
                if logical_page >= int(block_table.shape[1]):
                    break
                block_id_value = block_table[req_idx, logical_page]
                block_id = int(
                    block_id_value.item()
                    if hasattr(block_id_value, "item")
                    else block_id_value
                )
                token_start = logical_page * block_size
                token_count = min(block_size, seq_len - token_start)
            if block_id < 0:
                continue
            version = 0
            if page_versions is not None:
                version = page_versions.get(logical_page, 0)
            if force_all_pages:
                if _hbm_d2h_page_pending(
                    request_id,
                    layer_id,
                    logical_page,
                ):
                    continue
                # A producer asks whether it can SKIP a new snapshot. An old
                # partial tail (e.g. 11 tokens before this query, now 16 in
                # HBM) is a freshness miss, not invalid input to attention.
                # Leave strict readers, pending-copy suppression and verified
                # completion publication unchanged. No HBM is released here.
                existing = _DRAM_KV_PAGE_REGISTRY.lookup(
                    request_id=request_id,
                    request_key=request_key,
                    layer_name=layer_name,
                    layer_id=layer_id,
                    logical_page=logical_page,
                    expected_token_count=token_count,
                    version=version,
                    strict=not refresh_probe,
                )
                if existing is not None:
                    continue
            if async_register:
                if defer_source_views:
                    pending_pages.append(
                        _DeferredHBMRegistrationPage(
                            request_id=request_id,
                            request_key=request_key,
                            layer_name=layer_name,
                            layer_id=layer_id,
                            logical_page=logical_page,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            token_start=token_start,
                            token_count=token_count,
                            version=version,
                            block_id=block_id,
                        )
                    )
                else:
                    pending_pages.append(
                        _PendingHBMRegistrationPage(
                            request_id=request_id,
                            request_key=request_key,
                            layer_name=layer_name,
                            layer_id=layer_id,
                            logical_page=logical_page,
                            key=key_cache[block_id, :token_count].detach(),
                            value=value_cache[
                                block_id, :token_count
                            ].detach(),
                            token_start=token_start,
                            token_count=token_count,
                            version=version,
                            block_id=block_id,
                        )
                    )
            else:
                key_page = key_cache[block_id, :token_count].detach()
                value_page = value_cache[block_id, :token_count].detach()
                register_dram_kv_page(
                    request_id=request_id,
                    request_key=request_key,
                    layer_name=layer_name,
                    layer_id=layer_id,
                    logical_page=logical_page,
                    key=key_page,
                    value=value_page,
                    token_start=token_start,
                    token_count=token_count,
                    version=version,
                    copy_to_cpu=True,
                )
            registered_versions_by_request.setdefault(request_id, {})[
                logical_page
            ] = version
            registered_blocks_by_request.setdefault(request_id, {})[
                logical_page
            ] = block_id
            registered_token_counts_by_request.setdefault(request_id, {})[
                logical_page
            ] = token_count
            registered += 1
        if (not required_progress and max_pages_per_forward > 0
                and registered >= max_pages_per_forward):
            break
    if registered:
        provider_verification_pages: list[
            _PendingHBMRegistrationPage
        ] | None = None
        if async_register and pending_pages:
            deferred_page_count = sum(
                isinstance(page, _DeferredHBMRegistrationPage)
                for page in pending_pages
            )
            if deferred_page_count:
                _DRAM_KV_PAGE_REGISTRY.record_d2h_source_view_dispatch_submit(
                    deferred_page_count
                )
            source_ready_event = None
            if key_cache.is_cuda:
                source_ready_event = torch.cuda.Event(enable_timing=False)
                source_ready_event.record(
                    torch.cuda.current_stream(key_cache.device)
                )
            source_block_lease = (
                acquire_tiered_hbm_source_block_lease(
                    page.block_id for page in pending_pages
                )
                if tiered_hbm_source_block_lease_enabled()
                else None
            )
            submit = (
                _submit_required_hbm_registration if required_progress
                else _submit_async_hbm_registration
            )
            submitted = submit(
                pages=pending_pages,
                registered_versions_by_request=registered_versions_by_request,
                registered_blocks_by_request=registered_blocks_by_request,
                registered_token_counts_by_request=(
                    registered_token_counts_by_request
                ),
                block_size=block_size,
                layer_id=layer_id,
                started_at=started_at,
                source_ready_event=source_ready_event,
                source_block_lease=source_block_lease,
            )
            if submitted:
                return registered
            try:
                if deferred_page_count:
                    _DRAM_KV_PAGE_REGISTRY.record_d2h_source_view_dispatch_fallback()
                sync_pages = _materialize_hbm_registration_pages(pending_pages)
                provider_verification_pages = sync_pages
                _register_pending_hbm_pages_sync(sync_pages)
            finally:
                _release_hbm_source_block_lease(source_block_lease)
        _DRAM_KV_PAGE_REGISTRY.record_register_from_hbm(
            registered,
            duration_s=time.perf_counter() - started_at,
        )
        _mark_hbm_registration_complete(
            registered_versions_by_request=registered_versions_by_request,
            registered_blocks_by_request=registered_blocks_by_request,
            registered_token_counts_by_request=(
                registered_token_counts_by_request
            ),
            block_size=block_size,
            layer_id=layer_id,
            pages=provider_verification_pages,
        )
    return registered


def record_hbm_kv_cache_registration_skip(*, phase: str, reason: str) -> None:
    if not (
        tiered_dram_page_provider_enabled()
        and tiered_dram_register_from_hbm_enabled()
    ):
        return
    _DRAM_KV_PAGE_REGISTRY.record_register_from_hbm_skip(
        phase=phase,
        reason=reason,
    )


def _tensor_to_int_list(tensor: torch.Tensor) -> list[int]:
    if tensor.numel() == 0:
        return []
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _scheduled_pages_by_request(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    request_ids: list[str],
    num_reqs: int,
    block_size: int,
) -> dict[str, set[int]]:
    """Infer logical pages written by the current scheduled tokens.

    The debug provider should not scan or copy every resident page on every
    forward. query_start_loc and seq_lens directly identify the appended logical
    token range for each scheduled request, so derive touched logical pages from
    that range instead of reverse-mapping every block-table entry.
    """
    if query_start_loc is None:
        return {}
    if int(query_start_loc.numel()) < num_reqs + 1:
        return {}
    if int(seq_lens.numel()) < num_reqs:
        return {}

    query_starts = _tensor_to_int_list(query_start_loc[: num_reqs + 1])
    seq_lens_list = _tensor_to_int_list(seq_lens[:num_reqs])
    pages_by_request: dict[str, set[int]] = {}

    for req_idx in range(num_reqs):
        request_id = request_ids[req_idx]
        start = query_starts[req_idx]
        end = query_starts[req_idx + 1]
        if end <= start:
            continue
        seq_len = seq_lens_list[req_idx]
        query_len = end - start
        if seq_len <= 0 or query_len <= 0:
            continue
        context_len = max(0, seq_len - query_len)
        first_page = context_len // block_size
        last_page = (seq_len - 1) // block_size
        max_page = min(last_page, int(block_table.shape[1]) - 1)
        if max_page < first_page:
            continue
        pages_by_request[request_id] = set(range(first_page, max_page + 1))

    return pages_by_request


def wait_for_pending_hbm_d2h_reads(
    *,
    layer_name: str,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    request_ids: list[str] | None,
    block_size: int,
    block_table_cpu: torch.Tensor | None = None,
    seq_lens_cpu: torch.Tensor | None = None,
    query_start_loc_cpu: torch.Tensor | None = None,
) -> int:
    """Wait only when this forward will overwrite a pending D2H source.

    Full pages naturally stop appearing in later scheduled write ranges. A
    growing tail page appears again, so it is fenced until its previous async
    HBM-to-DRAM snapshot has completed.
    """

    global _HBM_D2H_FENCE_WAIT_COUNT
    global _HBM_D2H_FENCE_WAIT_PAGE_COUNT
    global _HBM_D2H_FENCE_WAIT_US_TOTAL
    global _HBM_D2H_FENCE_WAIT_US_LAST

    if (
        not tiered_dram_register_async_enabled()
        or not layer_name
        or not request_ids
        or query_start_loc is None
        or block_size <= 0
    ):
        return 0
    selected_block_table = (
        block_table_cpu
        if block_table_cpu is not None
        else block_table
    )
    selected_seq_lens = (
        seq_lens_cpu if seq_lens_cpu is not None else seq_lens
    )
    selected_query_start_loc = (
        query_start_loc_cpu
        if query_start_loc_cpu is not None
        else query_start_loc
    )
    num_reqs = min(
        len(request_ids),
        int(selected_seq_lens.numel()),
        int(selected_block_table.shape[0]),
    )
    if num_reqs <= 0:
        return 0
    pages_by_request = _scheduled_pages_by_request(
        block_table=selected_block_table,
        seq_lens=selected_seq_lens,
        slot_mapping=None,
        query_start_loc=selected_query_start_loc,
        request_ids=request_ids,
        num_reqs=num_reqs,
        block_size=block_size,
    )
    if not pages_by_request:
        return 0
    layer_id = virtual_context_layer_id(layer_name)
    keys = {
        (request_id, layer_id, logical_page)
        for request_id, pages in pages_by_request.items()
        for logical_page in pages
    }
    with _PENDING_HBM_D2H_FENCES_LOCK:
        matched_fences = {
            key: future
            for key in keys
            if (future := _PENDING_HBM_D2H_FENCES.get(key)) is not None
        }
    futures = set(matched_fences.values())
    if not futures:
        return 0
    wait_started = time.perf_counter()
    for future in futures:
        future.result()
    wait_us = max(0, int((time.perf_counter() - wait_started) * 1e6))
    with _PENDING_HBM_D2H_FENCES_LOCK:
        _HBM_D2H_FENCE_WAIT_COUNT += 1
        _HBM_D2H_FENCE_WAIT_PAGE_COUNT += len(matched_fences)
        _HBM_D2H_FENCE_WAIT_US_TOTAL += wait_us
        _HBM_D2H_FENCE_WAIT_US_LAST = wait_us
    return len(futures)


def get_dram_kv_page_registry_stats() -> dict[str, int]:
    stats = _DRAM_KV_PAGE_REGISTRY.stats()
    stats.update(required_d2h_progress.stats())
    stats.update(get_tiered_transfer_priority_stats())
    try:
        stats.update(
            _fused_kv_scatter_module().get_fused_kv_scatter_stats()
        )
    except Exception:
        pass
    with _PENDING_HBM_D2H_FENCES_LOCK:
        stats.update({
            "dram_provider_hbm_d2h_fence_pending_pages": len(
                _PENDING_HBM_D2H_FENCES
            ),
            "dram_provider_hbm_d2h_fence_wait_count": (
                _HBM_D2H_FENCE_WAIT_COUNT
            ),
            "dram_provider_hbm_d2h_fence_wait_page_count": (
                _HBM_D2H_FENCE_WAIT_PAGE_COUNT
            ),
            "dram_provider_hbm_d2h_fence_wait_us_total": (
                _HBM_D2H_FENCE_WAIT_US_TOTAL
            ),
            "dram_provider_hbm_d2h_fence_wait_us_last": (
                _HBM_D2H_FENCE_WAIT_US_LAST
            ),
            "dram_provider_decode_reload_barrier_check_count": (
                _DECODE_RELOAD_BARRIER_CHECK_COUNT
            ),
            "dram_provider_decode_reload_barrier_wait_count": (
                _DECODE_RELOAD_BARRIER_WAIT_COUNT
            ),
            "dram_provider_decode_reload_barrier_wait_page_count": (
                _DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT
            ),
            "dram_provider_decode_reload_barrier_wait_us_total": (
                _DECODE_RELOAD_BARRIER_WAIT_US_TOTAL
            ),
            "dram_provider_decode_reload_barrier_wait_us_last": (
                _DECODE_RELOAD_BARRIER_WAIT_US_LAST
            ),
            "dram_provider_decode_reload_barrier_complete_count": (
                _DECODE_RELOAD_BARRIER_COMPLETE_COUNT
            ),
            "dram_provider_decode_reload_barrier_h2d_retry_count": (
                _DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT
            ),
            "dram_provider_decode_reload_barrier_h2d_retry_page_count": (
                _DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT
            ),
            "dram_provider_decode_reload_barrier_incomplete_count": (
                _DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT
            ),
            "dram_provider_decode_reload_barrier_incomplete_page_count": (
                _DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT
            ),
        })
    return stats
