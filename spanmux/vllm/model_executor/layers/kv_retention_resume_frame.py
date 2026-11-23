# SPDX-License-Identifier: Apache-2.0
"""Explicit, bounded worker-only Prefill forward state.

This module deliberately has no eager torch/vLLM imports. A forward snapshot
is NOT a release capability for InputBatch, KV, or a phase/global slot. Only
the caller's ordered GPU-complete boundary can authorize capture/disposal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable


def kv_retention_resume_frame_requested() -> bool:
    return os.getenv("VLLM_KV_RETENTION_PREFILL_RESUME_FRAME", "0").strip().lower() in {
        "1", "true", "yes", "on"}


def kv_retention_resume_frame_gpu_bytes() -> int:
    value = int(os.getenv("VLLM_KV_RETENTION_RESUME_FRAME_GPU_BYTES", "268435456"))
    if value <= 0:
        raise ValueError("VLLM_KV_RETENTION_RESUME_FRAME_GPU_BYTES must be positive")
    return value


class KVRetentionFrameBudgetExceeded(RuntimeError):
    """Preflight refusal: no source state or ownership has changed."""


class KVRetentionFrameCopyError(RuntimeError):
    """Ambiguous copy failure retains the candidate; never lends a buffer."""

    def __init__(self, message: str, candidate: Any):
        super().__init__(message)
        self.candidate = candidate


@dataclass
class KVRetentionModelState:
    next_layer: int
    end_layer: int
    positions: Any
    hidden_states: Any
    residual: Any
    aux_hidden_states: list[Any]
    workspace_active: bool


class KVRetentionLayerContinuation:
    """Explicit iterator, with no suspended Python frame or captured inputs.

    The model/descriptor are shared read-only execution references; tensors
    live only in ``state``. The model advances and replaces this state. Once
    exported, only import of that exact frame under a successor binding can
    make the iterator executable again.
    """

    def __init__(self, model: Any, execution: Any, state: KVRetentionModelState):
        self.model = model
        self.execution = execution
        self.state: KVRetentionModelState | None = state
        self.exported: KVRetentionResumeFrame | None = None
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if self.state is None or self.exported is not None:
            raise RuntimeError("Prefill frame is exported, not executable")
        return self.model._kv_retention_advance_prefill(self)

    def export(self, frame: KVRetentionResumeFrame) -> None:
        if self.closed or self.exported is not None or self.state is None:
            raise RuntimeError("duplicate or closed Prefill frame export")
        if frame.next_layer != self.state.next_layer:
            raise RuntimeError("Prefill export layer changed")
        frame.require_ready()
        self.exported = frame
        self.state = None

    def import_frame(self, frame: KVRetentionResumeFrame, binding: Any,
                     *, record_stream: Callable[[Any], None]) -> None:
        if self.closed or self.exported is not frame or self.state is not None:
            raise RuntimeError("Prefill import does not own the exported frame")
        frame.validate_successor(binding)
        frame.record_streams(record_stream)
        self.state = frame.consume()
        self.exported = None

    def close(self) -> None:
        self.execution.cancel_yield_request()
        if self.exported is not None:
            self.exported.close()
        self.state = None
        self.exported = None
        self.closed = True


class _TorchSnapshotBackend:
    def __init__(self):
        import numpy as np
        import torch
        self.np = np
        self.torch = torch

    def tensor(self, value):
        return isinstance(value, self.torch.Tensor)

    def array(self, value):
        return isinstance(value, self.np.ndarray)

    @staticmethod
    def gpu(value):
        return value.device.type == "cuda"

    def clone(self, value):
        if value.layout != self.torch.strided:
            raise RuntimeError("Prefill frame supports dense strided tensors only")
        return value.detach().clone(memory_format=self.torch.contiguous_format)

    def empty(self, value):
        return self.torch.empty_like(value, memory_format=self.torch.contiguous_format)

    @staticmethod
    def compact_bytes(value):
        return value.numel() * value.element_size()

    @staticmethod
    def storage(value):
        storage = value.untyped_storage()
        return (str(value.device), storage.data_ptr()), storage.nbytes()


class _SnapshotTree:
    """Validate then copy a closed data tree; never deepcopy opaque owners."""

    def __init__(self, backend, *, record_only=False):
        self.backend = backend
        self.record_only = record_only
        self.tensors: dict[int, Any] = {}
        self.visited: set[int] = set()
        self.visiting: set[int] = set()
        self.memo: dict[int, Any] = {}
        self.scratch: set[int] = set()
        self.inputs: set[int] = set()

    def inspect(self, value, *, scratch=False):
        if value is None or type(value) in {bool, int, float, str, bytes}:
            return
        key = id(value)
        if (self.record_only
                and type(value).__module__ == "vllm.v1.attention.ops.tiered_residency_provider"
                and type(value).__name__ == "TieredDRAMKVLease"):
            # CPU provider ownership (callback + Lock), not tensor storage.
            # A retained Ring slot keeps this exact owner; only the Ring
            # executor's completed-read event may release it. Never allow
            # opaque owners in a detached frame's clone/export traversal.
            return
        if self.backend.tensor(value):
            (self.scratch if scratch else self.inputs).add(key)
            if key in self.scratch and key in self.inputs:
                raise RuntimeError("Prefill scratch aliases a logical input")
            self.tensors[key] = value
            self.visited.add(key)
            return
        if key in self.visiting:
            raise RuntimeError("cyclic Prefill frame metadata is unsupported")
        if key in self.visited:
            return
        if self.backend.array(value):
            if value.dtype.hasobject:
                raise RuntimeError("object arrays cannot own Prefill frame metadata")
        else:
            self.visiting.add(key)
            if is_dataclass(value) and not isinstance(value, type):
                if any(not f.init for f in fields(value)):
                    raise RuntimeError("Prefill frame dataclass has hidden init state")
                triton_metadata = (
                    type(value).__name__ == "TritonAttentionMetadata"
                    and type(value).__module__ == "vllm.v1.attention.backends.triton_attn")
                for f in fields(value):
                    self.inspect(getattr(value, f.name), scratch=bool(
                        triton_metadata and f.name in {
                            "softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"}))
            elif type(value) is dict:
                for k, v in value.items():
                    if type(k) not in {bool, int, float, str, bytes}:
                        raise RuntimeError("unsupported Prefill metadata key")
                    self.inspect(v)
            elif type(value) in {list, tuple, set, frozenset}:
                for item in value:
                    self.inspect(item)
            else:
                raise RuntimeError(f"unsupported Prefill frame object: {type(value).__name__}")
            self.visiting.remove(key)
        self.visited.add(key)

    def clone(self, value):
        if value is None or type(value) in {bool, int, float, str, bytes}:
            return value
        key = id(value)
        if key in self.memo:
            return self.memo[key]
        if self.backend.tensor(value):
            copied = (self.backend.empty(value) if key in self.scratch
                      else self.backend.clone(value))
        elif self.backend.array(value):
            copied = value.copy()
        elif is_dataclass(value):
            copied = replace(value, **{f.name: self.clone(getattr(value, f.name))
                                       for f in fields(value)})
        elif type(value) is dict:
            copied = {k: self.clone(v) for k, v in value.items()}
        else:
            copied = type(value)(self.clone(v) for v in value)
        self.memo[key] = copied
        return copied


def kv_retention_record_retained_streams(
    *, model_state: KVRetentionModelState, attn_metadata: Any,
    logits_indices: Any, record_stream: Callable[[Any], None], backend: Any = None,
) -> int:
    """Protect target-stream consumers of a slot-retaining SM continuation.

    This only registers allocator lifetime: no clone, frame export, CUDA wait,
    ownership transfer or permission to lend the workspace. The caller must
    already have the completed source-layer boundary and exact slot binding.
    """
    backend = _TorchSnapshotBackend() if backend is None else backend
    tree = _SnapshotTree(backend, record_only=True)
    tree.inspect(dict(model_state=model_state, attn_metadata=attn_metadata,
                      logits_indices=logits_indices))
    tensors = [tensor for tensor in tree.tensors.values() if backend.gpu(tensor)]
    for tensor in tensors:
        record_stream(tensor)
    return len(tensors)


@dataclass
class KVRetentionResumeFrame:
    """Completed forward snapshot, never an allocator or phase-slot receipt."""

    binding: Any
    next_layer: int
    payload: dict[str, Any]
    gpu_storage_bytes: int
    copied_tensor_bytes: int
    source_gpu_storage_bytes: int
    gpu_tensors: tuple[Any, ...] = ()
    ready: bool = False
    target_stream_recorded: bool = False
    consumed: bool = False
    closed: bool = False

    @classmethod
    def capture(cls, *, binding: Any, model_state: KVRetentionModelState,
                attn_metadata: Any, logits_indices: Any,
                num_scheduled_tokens_np: Any, max_gpu_bytes: int,
                gpu_boundary_complete: bool, copy_complete: Callable[[], None],
                backend: Any = None):
        if not gpu_boundary_complete or binding is None:
            raise RuntimeError("Prefill frame capture needs completed owned boundary")
        if not (binding.start_layer < model_state.next_layer < model_state.end_layer):
            raise RuntimeError("Prefill frame layer is outside its segment")
        if max_gpu_bytes <= 0:
            raise ValueError("Prefill frame GPU budget must be positive")
        backend = _TorchSnapshotBackend() if backend is None else backend
        logical_tokens = sum(row[3] for row in binding.work.rows)
        hidden = model_state.hidden_states
        positions = model_state.positions
        residual = model_state.residual
        if (not backend.tensor(hidden) or not backend.tensor(positions)
                or len(hidden.shape) != 2 or len(positions.shape) != 1
                or hidden.shape[0] < logical_tokens
                or positions.shape[0] != hidden.shape[0]
                or (residual is not None and (
                    not backend.tensor(residual) or residual.shape != hidden.shape))
                or len(num_scheduled_tokens_np) != len(binding.work.rows)
                or sum(int(n) for n in num_scheduled_tokens_np) != logical_tokens):
            raise RuntimeError("Prefill frame tensor shape/token ownership mismatch")
        tree = _SnapshotTree(backend)
        payload = dict(model_state=model_state, attn_metadata=attn_metadata,
                       logits_indices=logits_indices,
                       num_scheduled_tokens_np=num_scheduled_tokens_np)
        tree.inspect(payload)  # No allocations or mutation before full validation.
        required = sum(backend.compact_bytes(t) for t in tree.tensors.values()
                       if backend.gpu(t))
        if required > max_gpu_bytes:
            raise KVRetentionFrameBudgetExceeded(
                f"Prefill frame requires {required} GPU bytes, cap={max_gpu_bytes}")
        source_storages = {backend.storage(t)[0]: backend.storage(t)[1]
                           for t in tree.tensors.values() if backend.gpu(t)}
        copied_bytes = sum(backend.compact_bytes(t) for key, t in tree.tensors.items()
                           if key not in tree.scratch)
        try:
            copied = tree.clone(payload)
            cloned_tensors = [tree.memo[key] for key in tree.tensors]
            all_sources = {backend.storage(t)[0] for t in tree.tensors.values()}
            if any(backend.storage(t)[1] and backend.storage(t)[0] in all_sources
                   for t in cloned_tensors):
                raise RuntimeError("Prefill frame clone still aliases a source")
            storages = {backend.storage(t)[0]: backend.storage(t)[1]
                        for t in cloned_tensors if backend.gpu(t)}
            actual = sum(storages.values())
            # A backend which unexpectedly retains larger storage is unsafe;
            # this is not the normal allocation-free budget-refusal path.
            if actual > max_gpu_bytes:
                raise RuntimeError("Prefill frame actual storage exceeds cap")
            candidate = cls(binding, model_state.next_layer, copied, actual,
                            copied_bytes, sum(source_storages.values()),
                            tuple(t for t in cloned_tensors if backend.gpu(t)))
            copy_complete()
            candidate.ready = True
            return candidate
        except BaseException as exc:
            # Keep every successfully allocated copy until caller's fatal
            # cleanup proves completion. Never publish partial capture.
            raise KVRetentionFrameCopyError(
                "Prefill frame copy did not complete", tree.memo) from exc

    def require_ready(self):
        if not self.ready or self.closed or self.consumed:
            raise RuntimeError("Prefill frame is not ready for import")

    def validate_successor(self, binding):
        self.require_ready()
        binding.validate_successor(self.binding, self.next_layer)

    def consume(self):
        self.require_ready()
        if not self.target_stream_recorded:
            raise RuntimeError("Prefill frame import must record target stream")
        self.consumed = True
        return self.payload["model_state"]

    def record_streams(self, record_stream):
        self.require_ready()
        for tensor in self.gpu_tensors:
            record_stream(tensor)
        self.target_stream_recorded = True

    def close(self):
        self.payload.clear()
        self.gpu_tensors = ()
        self.closed = True

    def debug(self):
        return {
            "kv_retention_resume_frame_active": 1.0,
            "kv_retention_resume_frame_gpu_bytes": float(self.gpu_storage_bytes),
            "kv_retention_resume_frame_copy_bytes": float(self.copied_tensor_bytes),
            "kv_retention_resume_frame_source_gpu_bytes": float(self.source_gpu_storage_bytes),
        }
