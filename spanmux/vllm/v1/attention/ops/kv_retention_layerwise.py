# SPDX-License-Identifier: Apache-2.0
"""Private DRAM/staging owner for an offloaded whole Prefill continuation.

No private layer prefix is visible in the ordinary provider until every layer
has computed the same chunk. Attention reuses the production streaming Ring
executor; only source selection/ownership differ from ordinary Context Ring.
"""

from __future__ import annotations

import time

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.kv_retention_layerwise import (
    KVRetentionLayerwiseTransfer,
    reserve_kv_retention_host_bytes,
)
from vllm.model_executor.layers.ttft_layerwise_prefill import TTFTLayerwisePrefillError
from vllm.v1.attention.ops.tiered_blockwise_attention import (
    HBMStagingBuffer,
    TieredBlockwiseAttentionExecutor,
    TieredKVBlock,
    context_ring_batched_partial_enabled,
    context_ring_compute_group_tokens,
    context_ring_streaming_wave_groups,
)
from vllm.v1.attention.ops.tiered_residency_provider import (
    lookup_dram_kv_page,
    register_dram_kv_page,
    wait_for_pending_hbm_d2h_request_reads,
)
from vllm.v1.core.tiered_hbm_source_block_lease import (
    acquire_tiered_hbm_source_block_lease,
)
from vllm.v1.core.virtual_context import (
    get_virtual_context_table,
    virtual_context_layer_id,
)

# A poisoned CUDA context cannot prove DMA completion. Keep storage/leases
# until process teardown, instead of returning them to an otherwise live pool.
_QUARANTINED_OWNERS: list[object] = []


class KVRetentionRingContinuation:
    """Worker-only owner. A receipt does NOT retain this object or its tensors."""

    @classmethod
    def capture(cls, checkpoint, layers):
        """Snapshot after the existing completed-layer boundary.

        Budget exhaustion returns None without mutation. Unsupported execution
        fails before HBM release, not by silently invoking a different kernel.
        """
        if (context_ring_batched_partial_enabled()
                or context_ring_streaming_wave_groups() != 1
                or context_ring_compute_group_tokens() <= 0):
            raise TTFTLayerwisePrefillError(
                "layerwise offload requires bounded streaming Ring: "
                "BATCHED_PARTIAL=0, STREAMING_WAVE_GROUPS=1, positive group tokens"
            )
        caches = {}
        for name in checkpoint.layer_names:
            layer = layers[name]
            impl = layer.impl
            cache = layer.kv_cache[0]
            if (
                type(impl).__name__ != "TritonAttentionImpl"
                or getattr(impl, "alibi_slopes", None) is not None
                or getattr(impl, "sliding_window", None) not in (None, (-1, -1))
                or getattr(impl, "logits_soft_cap", 0) not in (None, 0, 0.0)
                or getattr(impl, "sinks", None) is not None
                or getattr(layer, "kv_sharing_target_layer_name", None) is not None
                or str(getattr(impl, "kv_cache_dtype", "auto")).startswith("fp8")
                or cache.ndim != 5 or cache.shape[1] != 2
                or cache.dtype not in (torch.float16, torch.bfloat16)
                or not cache.is_cuda
            ):
                raise TTFTLayerwisePrefillError("unsupported layerwise Ring attention layout")
            if any(row.block_size != cache.shape[2] for row in checkpoint.rows):
                raise TTFTLayerwisePrefillError("layerwise Ring block geometry changed")
            caches[name] = cache
        # Validate the tensor-free receipt geometry BEFORE allocating/copying.
        ticket = KVRetentionLayerwiseTransfer(checkpoint, copied_bytes=0)
        required = sum(
            (row.start_token + row.scheduled_tokens) * 2 * cache.shape[3]
            * cache.shape[4] * cache.element_size()
            for cache in caches.values() for row in checkpoint.rows
        )
        reservation = reserve_kv_retention_host_bytes(required)
        if reservation is None:
            return None
        owner = cls.__new__(cls)
        owner.ticket = ticket
        owner.reservation = reservation
        owner.buffers = {}
        owner.copy_stream = torch.cuda.Stream(device=next(iter(caches.values())).device)
        owner.copy_event = None
        owner.compute_event = None
        owner.completed_layers = checkpoint.next_layer
        owner.closed = False
        owner.executor = TieredBlockwiseAttentionExecutor(
            staging_buffer=HBMStagingBuffer(num_slots=2)
        )
        owner.group_tokens = context_ring_compute_group_tokens()
        started = time.monotonic()
        lease = acquire_tiered_hbm_source_block_lease(
            block for row in checkpoint.rows for block in row.block_ids
        )
        copy_safe = False
        try:
            wait_for_pending_hbm_d2h_request_reads([row.request_id for row in checkpoint.rows])
            for index, name in enumerate(checkpoint.layer_names):
                cache = caches[name]
                for row in checkpoint.rows:
                    size = row.start_token + row.scheduled_tokens
                    with torch.inference_mode(False):
                        k = torch.empty((size, *cache.shape[3:]), dtype=cache.dtype,
                                        device="cpu", pin_memory=True)
                        v = torch.empty_like(k, pin_memory=True)
                    owner.buffers[name, row.request_id] = (k, v)
                    valid = row.start_token + (row.scheduled_tokens if index < checkpoint.next_layer else 0)
                    if not valid:
                        continue
                    full_pages, tail_tokens = divmod(valid, row.block_size)
                    full_tokens = full_pages * row.block_size
                    with torch.cuda.stream(owner.copy_stream):
                        # The direct API packs equal-shaped pages, not a ragged
                        # list. Keep the exact tail separate: padding it would
                        # read uncomputed KV and overrun the private row span.
                        if full_pages:
                            blocks = row.block_ids[:full_pages]
                            ops.tiered_d2h_multicopy(
                                [cache[block, 0] for block in blocks],
                                [cache[block, 1] for block in blocks],
                                k[:full_tokens], v[:full_tokens],
                            )
                        if tail_tokens:
                            block = row.block_ids[full_pages]
                            ops.tiered_d2h_multicopy(
                                [cache[block, 0, :tail_tokens]],
                                [cache[block, 1, :tail_tokens]],
                                k[full_tokens:valid], v[full_tokens:valid],
                            )
                    # Both calls are covered by the same lease and final event.
                    ticket.copied_bytes += valid * 2 * cache.shape[3] * cache.shape[4] * cache.element_size()
            owner.copy_event = torch.cuda.Event()
            owner.copy_event.record(owner.copy_stream)
            # The existing execute Future completes only after this event.
            # No Engine/global host gate is held; Decode has its own worker.
            owner.copy_event.synchronize()
            copy_safe = True
        except BaseException:
            # Direct API may have enqueued a prefix of the copies before error.
            owner.close()
            copy_safe = True
            raise
        finally:
            if copy_safe:
                lease.release()
            else:
                _QUARANTINED_OWNERS.append((owner, lease))
        owner.snapshot_wait_s = time.monotonic() - started
        return owner

    def attention(self, layer_name, impl, query, key, value, output, metadata):
        self.ticket.require_ring()
        plan = self.ticket.checkpoint
        if (self.completed_layers >= len(plan.layer_names)
                or layer_name != plan.layer_names[self.completed_layers]):
            raise TTFTLayerwisePrefillError("Ring continuation layer order changed")
        rows = sorted(plan.rows, key=lambda row: row.input_row)
        total = sum(row.scheduled_tokens for row in rows)
        if metadata.num_actual_tokens != total or query.shape[0] < total:
            raise TTFTLayerwisePrefillError("Ring continuation query batch changed")
        if tuple(metadata.request_ids or ()) != tuple(row.request_id for row in rows):
            raise TTFTLayerwisePrefillError("Ring continuation input rows changed")
        # Record only after this layer has produced Q/K/V. The copy stream can
        # save current K/V while Ring stages/attends to the historical prefix.
        ready = torch.cuda.Event()
        stream = torch.cuda.current_stream(query.device)
        ready.record(stream)
        self.copy_stream.wait_event(ready)
        ring_copy_stream, _ = self.executor._get_copy_stream(query.device)
        if not getattr(self.executor, "isolation_early_prefetch", False):
            ring_copy_stream.wait_event(ready)
        # With early prefetch the executor fences only groups containing GPU
        # sources. Immutable CPU prefixes do not depend on this layer's QKV.
        cursor = 0
        for row in rows:
            end = cursor + row.scheduled_tokens
            k, v = self.buffers[layer_name, row.request_id]
            with torch.cuda.stream(self.copy_stream):
                k[row.start_token:].copy_(key[cursor:end], non_blocking=True)
                v[row.start_token:].copy_(value[cursor:end], non_blocking=True)
            cursor = end
        self.copy_event = torch.cuda.Event()
        self.copy_event.record(self.copy_stream)
        output_view = output.view(output.shape[0], query.shape[1], query.shape[2])
        cursor = 0
        try:
            for row in rows:
                end = cursor + row.scheduled_tokens
                k, v = self.buffers[layer_name, row.request_id]
                blocks = []
                if row.start_token:
                    blocks.append(TieredKVBlock(k[:row.start_token], v[:row.start_token],
                                               token_start=0, tier="DRAM"))
                blocks.append(TieredKVBlock(key[cursor:end], value[cursor:end],
                                           token_start=row.start_token, tier="HBM"))
                self.executor.run(
                    query[cursor:end], blocks, query_start=row.start_token,
                    causal=True, softmax_scale=impl.scale,
                    compute_group_tokens_override=self.group_tokens,
                    output_buffer=output_view[cursor:end], return_lse=False,
                )
                cursor = end
        finally:
            # GPU dependency, NOT stream.synchronize(): reusable QKV workspace
            # cannot be overwritten by the next layer before its D2H finishes.
            stream.wait_event(self.copy_event)
            self.compute_event = torch.cuda.Event()
            self.compute_event.record(stream)
        self.completed_layers += 1
        return output

    def publish(self) -> None:
        """Publish exact ordinary DRAM pages only after uniform full coverage."""
        self.ticket.require_ring()
        plan = self.ticket.checkpoint
        if self.completed_layers != len(plan.layer_names):
            raise TTFTLayerwisePrefillError("incomplete Ring continuation publication")
        if self.copy_event is not None:
            self.copy_event.synchronize()
        table = get_virtual_context_table()
        if table is None:
            raise TTFTLayerwisePrefillError("lost virtual-context publication owner")
        layer_ids = tuple(virtual_context_layer_id(name) for name in plan.layer_names)
        for row in plan.rows:
            version = table.reserve_layerwise_provider_version(row.request_id)
            total = row.start_token + row.scheduled_tokens
            for name, layer_id in zip(plan.layer_names, layer_ids):
                k, v = self.buffers[name, row.request_id]
                for page in range((total + row.block_size - 1) // row.block_size):
                    start = page * row.block_size
                    count = min(row.block_size, total - start)
                    register_dram_kv_page(
                        request_id=row.request_id, layer_name=name, layer_id=layer_id,
                        logical_page=page, key=k[start:start + count], value=v[start:start + count],
                        token_start=start, token_count=count, version=version, copy_to_cpu=False,
                    )
                    found = lookup_dram_kv_page(
                        request_id=row.request_id, layer_name=name, layer_id=layer_id,
                        logical_page=page, expected_token_count=count, version=version, strict=True,
                    )
                    if found is None:
                        raise TTFTLayerwisePrefillError("layerwise provider rejected final page")
            table.publish_layerwise_dram(
                request_id=row.request_id, logical_tokens=total, block_size=row.block_size,
                released_block_ids=row.block_ids, layer_ids=layer_ids, version=version,
            )
        self.ticket.mark_published(self.completed_layers)

    def close(self) -> None:
        """Ordered cleanup; never recycle storage under unfinished DMA."""
        if self.closed:
            return
        try:
            self.copy_stream.synchronize()
            # A failed run can leave H2D queued without a compute event.
            for stream in self.executor._copy_streams.values():
                stream.synchronize()
            if self.compute_event is not None:
                self.compute_event.synchronize()
        except BaseException:
            _QUARANTINED_OWNERS.append(self)
            raise
        self.buffers.clear()
        self.reservation.release()
        self.closed = True
