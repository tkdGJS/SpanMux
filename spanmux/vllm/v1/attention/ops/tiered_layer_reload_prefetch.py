# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-ahead DRAM-to-HBM reload pipeline for native attention handoff.

This module is deliberately independent from the scheduler.  Attention layer
``L`` submits provider-backed reload work for the next registered layer on a
dedicated CUDA stream.  Layer ``L+1`` consumes the corresponding future before
it derives its residency state.  The underlying reload helper publishes the
layer ACK only after its CUDA event completes, so another P/D lane cannot see a
premature HBM-ready state.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import torch

from vllm.v1.attention.ops.tiered_residency_provider import (
    reload_dram_kv_pages_to_hbm,
)
from vllm.v1.core.tiered_execution import resolve_tiered_reload_authority
from vllm.v1.core.virtual_context import (
    get_virtual_context_table,
    virtual_context_layer_id,
)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def tiered_layer_reload_prefetch_enabled() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def _worker_count() -> int:
    raw_value = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH_WORKERS",
        "2",
    ).strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 2


@dataclass(frozen=True)
class _LayerCacheEntry:
    layer_name: str
    layer_id: int
    key_cache: torch.Tensor
    value_cache: torch.Tensor


@dataclass(frozen=True)
class _PendingLayerReload:
    layer_id: int
    request_ids: tuple[str, ...]
    future: Future[dict[str, set[int]]]
    submitted_at: float


class _LayerReloadPrefetchCoordinator:
    def __init__(self) -> None:
        self._lock = Lock()
        self._layer_caches: dict[int, _LayerCacheEntry] = {}
        self._pending: dict[
            tuple[int, tuple[str, ...]], _PendingLayerReload
        ] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._copy_streams: dict[tuple[str, int], torch.cuda.Stream] = {}
        self._register_count = 0
        self._submit_count = 0
        self._submit_skip_no_next_layer = 0
        self._submit_skip_no_pages = 0
        self._submit_duplicate_count = 0
        self._consume_count = 0
        self._consume_wait_count = 0
        self._consume_wait_us_total = 0
        self._consume_wait_us_last = 0
        self._failure_count = 0
        self._prefetched_page_count = 0
        self._metadata_snapshot_count = 0
        self._metadata_cpu_snapshot_count = 0

    def register_layer(
        self,
        layer_name: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> int:
        layer_id = virtual_context_layer_id(layer_name)
        if not tiered_layer_reload_prefetch_enabled():
            return layer_id
        with self._lock:
            self._layer_caches[layer_id] = _LayerCacheEntry(
                layer_name=layer_name,
                layer_id=layer_id,
                key_cache=key_cache,
                value_cache=value_cache,
            )
            self._register_count += 1
        return layer_id

    def _next_layer_locked(self, layer_id: int) -> _LayerCacheEntry | None:
        later_ids = [
            candidate
            for candidate in self._layer_caches
            if candidate > layer_id
        ]
        if not later_ids:
            return None
        return self._layer_caches[min(later_ids)]

    def _executor_locked(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=_worker_count(),
                thread_name_prefix="tiered-layer-reload",
            )
        return self._executor

    def _copy_stream(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream:
        device_index = 0 if device.index is None else int(device.index)
        key = (device.type, device_index)
        with self._lock:
            stream = self._copy_streams.get(key)
            if stream is None:
                with torch.cuda.device(device):
                    stream = torch.cuda.Stream(device=device)
                self._copy_streams[key] = stream
            return stream

    def consume(
        self,
        layer_name: str,
        request_ids: list[str] | None,
    ) -> dict[str, set[int]]:
        if not tiered_layer_reload_prefetch_enabled() or not request_ids:
            return {}
        layer_id = virtual_context_layer_id(layer_name)
        request_key = tuple(request_ids)
        pending_key = (layer_id, request_key)
        with self._lock:
            pending = self._pending.pop(pending_key, None)
        if pending is None:
            return {}
        wait_started = time.perf_counter()
        waited = not pending.future.done()
        try:
            result = pending.future.result()
        except Exception:
            with self._lock:
                self._failure_count += 1
            raise
        wait_us = max(0, int((time.perf_counter() - wait_started) * 1e6))
        with self._lock:
            self._consume_count += 1
            if waited:
                self._consume_wait_count += 1
            self._consume_wait_us_total += wait_us
            self._consume_wait_us_last = wait_us
            self._prefetched_page_count += sum(map(len, result.values()))
        return result

    def submit_next(
        self,
        *,
        current_layer_name: str,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        block_table_cpu: torch.Tensor | None,
        seq_lens_cpu: torch.Tensor | None,
        query_start_loc_cpu: torch.Tensor | None,
        request_ids: list[str] | None,
        execution_plans: dict[str, tuple[int, str, int]] | None = None,
        attention_phase: str = "",
    ) -> bool:
        if (
            not tiered_layer_reload_prefetch_enabled()
            or not request_ids
            or not torch.cuda.is_available()
        ):
            return False
        current_layer_id = virtual_context_layer_id(current_layer_name)
        with self._lock:
            next_layer = self._next_layer_locked(current_layer_id)
        if next_layer is None:
            with self._lock:
                self._submit_skip_no_next_layer += 1
            return False

        virtual_context = get_virtual_context_table()
        if virtual_context is None:
            return False
        page_versions = virtual_context.pin_attention_page_versions(
            request_ids,
            group_id=0,
            layer_id=next_layer.layer_id,
        )
        reload_pages = virtual_context.plan_hbm_reload_pages(
            request_ids=set(request_ids),
            layer_id=next_layer.layer_id,
        )
        reload_pages_by_request: dict[str, set[int]] = {}
        for page in reload_pages:
            reload_pages_by_request.setdefault(page.request_id, set()).add(
                page.logical_token_page
            )
        if not reload_pages_by_request:
            with self._lock:
                self._submit_skip_no_pages += 1
            return False

        # Resolve against NEXT layer's rows, not the already-consumed current
        # layer. Freeze the scalar proof alongside this copy's metadata snapshot.
        block_table_is_fully_concrete = resolve_tiered_reload_authority(
            phase=attention_phase,
            reload_required_pages=reload_pages_by_request,
            execution_plans=execution_plans,
        ) == "native"

        request_key = tuple(request_ids)
        pending_key = (next_layer.layer_id, request_key)
        with self._lock:
            if pending_key in self._pending:
                self._submit_duplicate_count += 1
                return False
            executor = self._executor_locked()

        block_table_snapshot = (
            block_table_cpu.detach().clone()
            if block_table_cpu is not None
            else block_table.detach().clone()
        )
        seq_lens_snapshot = (
            seq_lens_cpu.detach().clone()
            if seq_lens_cpu is not None
            else seq_lens.detach().clone()
        )
        query_start_loc_snapshot = None
        if query_start_loc_cpu is not None:
            query_start_loc_snapshot = (
                query_start_loc_cpu.detach().clone()
            )
        elif query_start_loc is not None:
            query_start_loc_snapshot = query_start_loc.detach().clone()
        snapshot_ready_event = None
        snapshot_tensors = [
            block_table_snapshot,
            seq_lens_snapshot,
            query_start_loc_snapshot,
        ]
        cuda_snapshot = next(
            (
                tensor
                for tensor in snapshot_tensors
                if tensor is not None and tensor.is_cuda
            ),
            None,
        )
        if cuda_snapshot is not None:
            snapshot_ready_event = torch.cuda.Event(enable_timing=False)
            snapshot_ready_event.record(
                torch.cuda.current_stream(cuda_snapshot.device)
            )
        with self._lock:
            self._metadata_snapshot_count += 1
            if all(
                tensor is None or not tensor.is_cuda
                for tensor in snapshot_tensors
            ):
                self._metadata_cpu_snapshot_count += 1

        def run_reload() -> dict[str, set[int]]:
            stream = self._copy_stream(next_layer.key_cache.device)
            with torch.cuda.device(next_layer.key_cache.device):
                with torch.cuda.stream(stream):
                    if snapshot_ready_event is not None:
                        stream.wait_event(snapshot_ready_event)
                    # Inference mode is thread-local. The executor worker does
                    # not inherit the model-forward thread's mode, so establish
                    # it explicitly for KV-cache mutation and scatter.
                    with torch.inference_mode():
                        return reload_dram_kv_pages_to_hbm(
                            layer_name=next_layer.layer_name,
                            key_cache=next_layer.key_cache,
                            value_cache=next_layer.value_cache,
                            block_table=block_table_snapshot,
                            seq_lens=seq_lens_snapshot,
                            query_start_loc=query_start_loc_snapshot,
                            request_ids=list(request_ids),
                            block_size=int(next_layer.key_cache.shape[1]),
                            page_versions_by_request=page_versions,
                            reload_pages_by_request=reload_pages_by_request,
                            block_table_is_fully_concrete=(
                                block_table_is_fully_concrete
                            ),
                        )

        future = executor.submit(run_reload)
        pending = _PendingLayerReload(
            layer_id=next_layer.layer_id,
            request_ids=request_key,
            future=future,
            submitted_at=time.perf_counter(),
        )
        with self._lock:
            self._pending[pending_key] = pending
            self._submit_count += 1
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "layer_reload_prefetch_enabled": int(
                    tiered_layer_reload_prefetch_enabled()
                ),
                "layer_reload_prefetch_registered_layers": len(
                    self._layer_caches
                ),
                "layer_reload_prefetch_pending": len(self._pending),
                "layer_reload_prefetch_register_count": self._register_count,
                "layer_reload_prefetch_submit_count": self._submit_count,
                "layer_reload_prefetch_skip_no_next_layer": (
                    self._submit_skip_no_next_layer
                ),
                "layer_reload_prefetch_skip_no_pages": (
                    self._submit_skip_no_pages
                ),
                "layer_reload_prefetch_duplicate_count": (
                    self._submit_duplicate_count
                ),
                "layer_reload_prefetch_consume_count": self._consume_count,
                "layer_reload_prefetch_consume_wait_count": (
                    self._consume_wait_count
                ),
                "layer_reload_prefetch_consume_wait_us_total": (
                    self._consume_wait_us_total
                ),
                "layer_reload_prefetch_consume_wait_us_last": (
                    self._consume_wait_us_last
                ),
                "layer_reload_prefetch_failure_count": self._failure_count,
                "layer_reload_prefetch_page_count": (
                    self._prefetched_page_count
                ),
                "layer_reload_prefetch_metadata_snapshot_count": (
                    self._metadata_snapshot_count
                ),
                "layer_reload_prefetch_metadata_cpu_snapshot_count": (
                    self._metadata_cpu_snapshot_count
                ),
            }


_COORDINATOR = _LayerReloadPrefetchCoordinator()


def register_tiered_layer_kv_cache(
    layer_name: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> int:
    return _COORDINATOR.register_layer(layer_name, key_cache, value_cache)


def consume_tiered_layer_reload_prefetch(
    layer_name: str,
    request_ids: list[str] | None,
) -> dict[str, set[int]]:
    return _COORDINATOR.consume(layer_name, request_ids)


def submit_next_tiered_layer_reload_prefetch(
    *,
    current_layer_name: str,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    request_ids: list[str] | None,
    block_table_cpu: torch.Tensor | None = None,
    seq_lens_cpu: torch.Tensor | None = None,
    query_start_loc_cpu: torch.Tensor | None = None,
    execution_plans: dict[str, tuple[int, str, int]] | None = None,
    attention_phase: str = "",
) -> bool:
    return _COORDINATOR.submit_next(
        current_layer_name=current_layer_name,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        block_table_cpu=block_table_cpu,
        seq_lens_cpu=seq_lens_cpu,
        query_start_loc_cpu=query_start_loc_cpu,
        request_ids=request_ids,
        execution_plans=execution_plans,
        attention_phase=attention_phase,
    )


def get_tiered_layer_reload_prefetch_stats() -> dict[str, int]:
    return _COORDINATOR.stats()
