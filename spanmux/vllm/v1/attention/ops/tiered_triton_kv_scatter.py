# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused staged K/V scatter used by native tiered reload.

The source is one contiguous HBM staging span. Destination vLLM KV blocks
may be fragmented. One Triton launch writes both K and V for every page in
the span, replacing two ``index_copy_`` launches and repeated construction of
the destination-block tensor.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_staged_kv_scatter_kernel(
    staged_key_ptr,
    staged_value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    destination_blocks_ptr,
    PAGE_ELEMENTS: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    page_index = tl.program_id(0)
    element_offsets = (
        tl.program_id(1) * BLOCK_ELEMENTS
        + tl.arange(0, BLOCK_ELEMENTS)
    )
    mask = element_offsets < PAGE_ELEMENTS
    source_offsets = page_index * PAGE_ELEMENTS + element_offsets
    destination_block = tl.load(destination_blocks_ptr + page_index)
    destination_offsets = (
        destination_block * PAGE_ELEMENTS + element_offsets
    )

    key = tl.load(staged_key_ptr + source_offsets, mask=mask)
    value = tl.load(staged_value_ptr + source_offsets, mask=mask)
    tl.store(key_cache_ptr + destination_offsets, key, mask=mask)
    tl.store(value_cache_ptr + destination_offsets, value, mask=mask)


@dataclass(frozen=True)
class _DestinationBlockDescriptor:
    tensor: torch.Tensor
    ready_event: torch.cuda.Event


_DESCRIPTOR_CACHE: OrderedDict[
    tuple[str, tuple[int, ...]], _DestinationBlockDescriptor
] = OrderedDict()
_DESCRIPTOR_CACHE_LOCK = Lock()
_STATS_LOCK = Lock()
_FUSED_SCATTER_LAUNCHES = 0
_FUSED_SCATTER_PAGES = 0
_FUSED_SCATTER_FAILURES = 0
_DESCRIPTOR_CACHE_HITS = 0
_DESCRIPTOR_CACHE_MISSES = 0
_DESCRIPTOR_CACHE_EVICTIONS = 0


def _descriptor_cache_capacity() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_RELOAD_SCATTER_DESCRIPTOR_CACHE_ENTRIES",
        "512",
    ).strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 512


def _destination_block_descriptor(
    destination_block_ids: tuple[int, ...],
    device: torch.device,
) -> tuple[_DestinationBlockDescriptor, bool]:
    global _DESCRIPTOR_CACHE_EVICTIONS
    global _DESCRIPTOR_CACHE_HITS
    global _DESCRIPTOR_CACHE_MISSES

    cache_key = (str(device), destination_block_ids)
    with _DESCRIPTOR_CACHE_LOCK:
        descriptor = _DESCRIPTOR_CACHE.get(cache_key)
        if descriptor is not None:
            _DESCRIPTOR_CACHE.move_to_end(cache_key)
            with _STATS_LOCK:
                _DESCRIPTOR_CACHE_HITS += 1
            cache_hit = True
        else:
            # Phase-local worker threads may run under inference mode. This
            # immutable descriptor is shared across those threads, so create a
            # normal tensor and publish it only after a CUDA ready event.
            with torch.inference_mode(False):
                tensor = torch.tensor(
                    destination_block_ids,
                    dtype=torch.int64,
                    device=device,
                    requires_grad=False,
                )
            ready_event = torch.cuda.Event(enable_timing=False)
            ready_event.record(torch.cuda.current_stream(device))
            descriptor = _DestinationBlockDescriptor(tensor, ready_event)
            _DESCRIPTOR_CACHE[cache_key] = descriptor
            with _STATS_LOCK:
                _DESCRIPTOR_CACHE_MISSES += 1
            cache_hit = False
            while len(_DESCRIPTOR_CACHE) > _descriptor_cache_capacity():
                _DESCRIPTOR_CACHE.popitem(last=False)
                with _STATS_LOCK:
                    _DESCRIPTOR_CACHE_EVICTIONS += 1

    torch.cuda.current_stream(device).wait_event(descriptor.ready_event)
    return descriptor, cache_hit


def fused_scatter_staged_kv_pages(
    *,
    staged_key: torch.Tensor,
    staged_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    destination_block_ids: tuple[int, ...],
) -> bool:
    """Scatter a contiguous staged span into fragmented K/V cache blocks."""

    global _FUSED_SCATTER_FAILURES
    global _FUSED_SCATTER_LAUNCHES
    global _FUSED_SCATTER_PAGES

    num_pages = len(destination_block_ids)
    valid = (
        num_pages > 0
        and staged_key.is_cuda
        and staged_value.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and staged_key.device == key_cache.device
        and staged_value.device == key_cache.device
        and value_cache.device == key_cache.device
        and staged_key.dtype == key_cache.dtype
        and staged_value.dtype == value_cache.dtype
        and staged_key.is_contiguous()
        and staged_value.is_contiguous()
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and tuple(key_cache.shape) == tuple(value_cache.shape)
        and staged_key.numel() == staged_value.numel()
    )
    if not valid:
        with _STATS_LOCK:
            _FUSED_SCATTER_FAILURES += 1
        return False

    page_elements = int(key_cache[0].numel())
    if staged_key.numel() != num_pages * page_elements:
        with _STATS_LOCK:
            _FUSED_SCATTER_FAILURES += 1
        return False

    try:
        descriptor, _ = _destination_block_descriptor(
            destination_block_ids,
            key_cache.device,
        )
        block_elements = 256
        grid = (num_pages, triton.cdiv(page_elements, block_elements))
        _fused_staged_kv_scatter_kernel[grid](
            staged_key,
            staged_value,
            key_cache,
            value_cache,
            descriptor.tensor,
            PAGE_ELEMENTS=page_elements,
            BLOCK_ELEMENTS=block_elements,
        )
        descriptor.tensor.record_stream(
            torch.cuda.current_stream(key_cache.device)
        )
    except Exception:
        with _STATS_LOCK:
            _FUSED_SCATTER_FAILURES += 1
        raise
    with _STATS_LOCK:
        _FUSED_SCATTER_LAUNCHES += 1
        _FUSED_SCATTER_PAGES += num_pages
    return True


def get_fused_kv_scatter_stats() -> dict[str, int]:
    with _DESCRIPTOR_CACHE_LOCK, _STATS_LOCK:
        return {
            "dram_provider_reload_fused_scatter_launches": (
                _FUSED_SCATTER_LAUNCHES
            ),
            "dram_provider_reload_fused_scatter_pages": _FUSED_SCATTER_PAGES,
            "dram_provider_reload_fused_scatter_failures": (
                _FUSED_SCATTER_FAILURES
            ),
            "dram_provider_reload_scatter_descriptor_cache_entries": len(
                _DESCRIPTOR_CACHE
            ),
            "dram_provider_reload_scatter_descriptor_cache_hits": (
                _DESCRIPTOR_CACHE_HITS
            ),
            "dram_provider_reload_scatter_descriptor_cache_misses": (
                _DESCRIPTOR_CACHE_MISSES
            ),
            "dram_provider_reload_scatter_descriptor_cache_evictions": (
                _DESCRIPTOR_CACHE_EVICTIONS
            ),
        }


def clear_fused_kv_scatter_state() -> None:
    global _DESCRIPTOR_CACHE_EVICTIONS
    global _DESCRIPTOR_CACHE_HITS
    global _DESCRIPTOR_CACHE_MISSES
    global _FUSED_SCATTER_FAILURES
    global _FUSED_SCATTER_LAUNCHES
    global _FUSED_SCATTER_PAGES

    with _DESCRIPTOR_CACHE_LOCK, _STATS_LOCK:
        _DESCRIPTOR_CACHE.clear()
        _FUSED_SCATTER_LAUNCHES = 0
        _FUSED_SCATTER_PAGES = 0
        _FUSED_SCATTER_FAILURES = 0
        _DESCRIPTOR_CACHE_HITS = 0
        _DESCRIPTOR_CACHE_MISSES = 0
        _DESCRIPTOR_CACHE_EVICTIONS = 0
