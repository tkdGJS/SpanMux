# SPDX-License-Identifier: Apache-2.0
"""Bounded, default-off P1 parking contracts; no CUDA imports or hidden buffers."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

_event_wait_lock = threading.Lock()
_event_wait_pool = None


def kv_retention_bound_prefill_submission(
    events, *, record_event=None, completed_fastpath: bool = False,
) -> None:
    """Keep one executing plus one queued layer group, without stopping Decode.

    The helper owns only an event, never a tensor or a slot. Waiting on its
    Future releases the Prefill thread's GIL; no Engine or Decode stream waits.
    Call only in the opt-in explicit layer loop. The latest event is still
    drained by the runner before it publishes any continuation boundary.
    """
    global _event_wait_pool
    if record_event is None:
        import torch
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
    else:
        event = record_event()
    events.append(event)
    if len(events) < 2:
        return
    if completed_fastpath and events[0].query():
        # Query is a completion proof only for the oldest group. The newest
        # group is still owned by the runner and must be fenced on real yield.
        events.pop(0)
        return
    with _event_wait_lock:
        if _event_wait_pool is None:
            _event_wait_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kv-retention-layer-event")
        waiter = _event_wait_pool.submit(events[0].synchronize)
    waiter.result()  # CUDA errors propagate; they never authorize slot reuse.
    events.pop(0)


def kv_retention_p1_parked_requested() -> bool:
    return os.getenv("VLLM_KV_RETENTION_P1_PARKED_PREFILL", "0").strip().lower() in {
        "1", "true", "yes", "on"}


def kv_retention_p1_hbm_only_requested() -> bool:
    value = os.getenv("VLLM_KV_RETENTION_P1_HBM_ONLY", "0")
    if value not in {"0", "1"}:
        raise ValueError("VLLM_KV_RETENTION_P1_HBM_ONLY must be 0 or 1")
    return value == "1"


def validate_kv_retention_p1_hbm_only(
    *, enabled: bool, parked: bool, offload: bool,
    completion_guard, virtual_context,
) -> None:
    """HBM parking changes slot ownership, never physical KV ownership."""
    if enabled and (not parked or offload or completion_guard is None
                    or virtual_context is not None):
        raise RuntimeError(
            "P1 HBM-only preemption requires P1 parking and the HBM completion "
            "guard, with KV offload and virtual context disabled")


def kv_retention_parked_limits() -> tuple[int, int]:
    batches = int(os.getenv("VLLM_KV_RETENTION_PARKED_MAX_BATCHES", "1"))
    gpu_bytes = int(os.getenv("VLLM_KV_RETENTION_PARKED_GPU_BYTES", str(256 << 20)))
    if batches != 1 or gpu_bytes <= 0:
        raise ValueError("P1 parking requires exactly one parked batch and positive GPU bytes")
    return batches, gpu_bytes


@dataclass(frozen=True)
class KVRetentionParkedTicket:
    """Worker's completed export ACK, not a second token transaction."""
    binding: Any
    next_layer: int
    gpu_bytes: int
    request_ids: tuple[str, ...]

    def validate(self, boundary, output) -> None:
        if (boundary.kv_retention_binding != self.binding
                or boundary.next_layer != self.next_layer
                or tuple(output.num_scheduled_tokens) != self.request_ids
                or self.gpu_bytes < 0):
            raise RuntimeError("parked ticket does not own this completed boundary")
        self.binding.validate_output("prefill", output)


def _request_signature(request):
    return (request.req_id, int(request.num_computed_tokens),
            tuple(request.prompt_token_ids or ()), tuple(request.output_token_ids),
            tuple(tuple(group) for group in request.block_ids),
            int(getattr(request, "output_token_generation", 0)),
            id(request.generator), id(request.sampling_params))


@dataclass
class KVRetentionSamplingFrame:
    """CPU-only row recipe; the original request owns its sampling RNG/history.

    Pending logical ownership prevents Decode or a second chunk from modifying
    these request objects. Validate that proof again before touching slot0.
    Built-in logits processors are reconstructed via normal remove/add updates.
    No InputBatch, GPU sampling metadata or reusable buffer view is retained.
    """
    rows: tuple[Any, ...]
    signatures: tuple[Any, ...]
    computed: tuple[int, ...]
    token_counts: tuple[int, ...]
    buffers: dict[str, Any]
    prompt_logprobs: dict[str, Any]

    @classmethod
    def capture(cls, runner, state):
        batch = runner.input_batch
        ids = tuple(batch.req_ids)
        if (len(ids) != state.num_reqs or set(ids) != set(state.scheduler_output.num_scheduled_tokens)
                or len(set(ids)) != len(ids) or getattr(runner, "_custom_logitsprocs", ())):
            raise RuntimeError("unsupported P1 sampling row/processor ownership")
        rows = tuple(runner.requests[req_id] for req_id in ids)
        if any(getattr(row, "prompt_embeds", None) is not None
               or getattr(row, "mm_features", ()) or getattr(row, "lora_request", None)
               or getattr(row, "pooling_params", None) is not None for row in rows):
            raise RuntimeError("P1 sampling restore supports text-only, non-LoRA generation")
        lengths = {"input_ids": state.num_scheduled_tokens,
                   "positions": state.num_scheduled_tokens,
                   "query_start_loc": len(ids) + 1,
                   "seq_lens": len(ids), "discard_request_mask": len(ids)}
        buffers = {name: getattr(runner, name).np[:length].copy()
                   for name, length in lengths.items()}
        return cls(rows, tuple(map(_request_signature, rows)),
                   tuple(int(v) for v in batch.num_computed_tokens_cpu[:len(ids)]),
                   tuple(int(v) for v in batch.num_tokens_no_spec[:len(ids)]), buffers,
                   {req_id: value for req_id, value in batch.in_progress_prompt_logprobs_cpu.items()
                    if req_id in ids})

    def restore(self, runner) -> None:
        for row, signature in zip(self.rows, self.signatures):
            if (runner.requests.get(row.req_id) is not row
                    or _request_signature(row) != signature):
                raise RuntimeError("parked sampling request changed before import")
        batch = runner.input_batch
        if len(self.rows) > batch.max_num_reqs:
            raise RuntimeError("parked sampling rows exceed active batch budget")
        # Flush removals before adds: built-in processors must not retain C's
        # state at an index that now represents A.
        for req_id in tuple(batch.req_ids):
            batch.remove_request(req_id)
        batch.condense()
        batch.refresh_metadata()
        batch.prev_sampled_token_ids = None
        batch.prev_req_id_to_index = None
        batch.sampled_token_ids_cpu = None
        batch.async_copy_ready_event = None
        for index, row in enumerate(self.rows):
            if batch.add_request(row) != index:
                raise RuntimeError("P1 sampling restore changed immutable row order")
        batch.num_computed_tokens_cpu[:len(self.rows)] = self.computed
        batch.num_tokens_no_spec[:len(self.rows)] = self.token_counts
        batch.in_progress_prompt_logprobs_cpu.update(self.prompt_logprobs)
        batch.refresh_metadata()
        for name, values in self.buffers.items():
            buffer = getattr(runner, name)
            buffer.np[:len(values)] = values
            buffer.copy_to_gpu(len(values))


class KVRetentionParkedStore:
    """One Worker-owned frame; mutation stays on the ordered Prefill worker."""
    def __init__(self, max_gpu_bytes: int):
        self.max_gpu_bytes = max_gpu_bytes
        self.entries: dict[int, tuple[KVRetentionParkedTicket, Any]] = {}

    def put(self, ticket, state):
        if self.entries or ticket.gpu_bytes > self.max_gpu_bytes:
            raise RuntimeError("parked store budget was not reserved")
        self.entries[ticket.binding.work.work_id] = (ticket, state)

    def get(self, binding):
        item = self.entries.get(binding.work.work_id)
        if item is None:
            return None
        if item[0].binding != binding:
            raise RuntimeError("stale parked binding")
        return item

    def take(self, previous, successor):
        item = self.get(previous)
        if item is None:
            raise RuntimeError("parked work lost its frame")
        ticket, state = item
        successor.validate_successor(previous, ticket.next_layer)
        del self.entries[previous.work.work_id]
        return state

    def close(self, binding) -> bool:
        item = self.get(binding)
        if item is None:
            return False
        _, state = item
        if state.kv_offload is not None:
            state.kv_offload.close()
        state.continuation.close()
        del self.entries[binding.work.work_id]
        return True
