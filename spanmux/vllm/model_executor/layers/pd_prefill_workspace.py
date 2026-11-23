# SPDX-License-Identifier: Apache-2.0
"""Fixed, phase-slot-scoped scratch storage for eager Prefill execution.

GreenContext profiles use different CUDA streams.  Allocating large eager
Llama intermediates independently on every profile stream leaves one caching
allocator high-water mark per stream.  This module instead owns one physical
arena per bounded Prefill phase slot.  A slot keeps the same storage address
while its GreenContext profile changes, and distinct in-flight Futures never
alias one another.

The feature is intentionally narrow and default-off:

* only phase-pure Prefill uses the arena;
* Decode tensors, KV cache and Context Ring staging never use it;
* configured workspace slots must cover Prefill inflight depth;
* a CUDA event orders reuse of one slot on another profile stream;
* legacy arenas stay fixed; opt-in capacity growth occurs only between leases;
* Ring may borrow dead MLP bytes only inside a same-stream Attention scope.

The model-specific layout is prepared lazily. All profile changes within a
lease keep the same addresses. Opt-in growth between leases retains replaced
storage until completion; it does not shrink or flush the CUDA caching allocator.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

from vllm.isolation_memory import isolation_memory_enabled


class PrefillWorkspaceError(RuntimeError):
    """Raised when fixed-workspace ownership cannot be proven safe."""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def prefill_workspace_requested() -> bool:
    """Return whether the experimental fixed Prefill workspace is enabled."""

    return _env_flag("VLLM_PD_PREFILL_WORKSPACE_ENABLED", False)


def prefill_workspace_strict() -> bool:
    """Fail closed on unsupported layouts instead of using dynamic outputs."""

    return _env_flag("VLLM_PD_PREFILL_WORKSPACE_STRICT", True)


def prefill_workspace_max_tokens() -> int:
    raw_value = os.getenv("VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS", "16384")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise PrefillWorkspaceError(
            "VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS must be an integer"
        ) from exc
    if value <= 0:
        raise PrefillWorkspaceError(
            "VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS must be positive"
        )
    return value


def prefill_workspace_slot_count() -> int:
    """Return the maximum number of concurrently owned Prefill arenas."""

    raw_value = os.getenv("VLLM_PD_PREFILL_WORKSPACE_SLOTS", "1")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise PrefillWorkspaceError(
            "VLLM_PD_PREFILL_WORKSPACE_SLOTS must be an integer"
        ) from exc
    if value <= 0:
        raise PrefillWorkspaceError(
            "VLLM_PD_PREFILL_WORKSPACE_SLOTS must be positive"
        )
    return value


@dataclass(frozen=True)
class LlamaPrefillWorkspaceLayout:
    """Token-major buffer widths held in the common Llama scratch arena."""

    qkv_width: int
    attention_width: int
    gate_up_width: int
    activation_width: int
    hidden_width: int = 0

    @property
    def widths(self) -> dict[str, int]:
        widths = {
            "qkv": self.qkv_width,
            "attention": self.attention_width,
            "gate_up": self.gate_up_width,
            "activation": self.activation_width,
        }
        if self.hidden_width:
            widths.update(norm=self.hidden_width, hidden=self.hidden_width)
        return widths

    def regions(self, tokens: int) -> dict[str, tuple[int, int]]:
        """Offsets encode liveness, not merely differently named tensors."""
        regions = {
            "qkv": (0, tokens * self.qkv_width),
            "attention": (tokens * self.qkv_width, tokens * self.attention_width),
            "gate_up": (0, tokens * self.gate_up_width),
            "activation": (
                tokens * self.gate_up_width, tokens * self.activation_width
            ),
        }
        if self.hidden_width:
            regions["norm"] = (
                tokens * max(
                    self.qkv_width + self.attention_width, self.gate_up_width
                ),
                tokens * self.hidden_width,
            )
            # down_proj consumes activation, after gate_up is dead.
            regions["hidden"] = (0, tokens * self.hidden_width)
        return regions

    def validate(self) -> None:
        if self.hidden_width < 0 or self.hidden_width > self.gate_up_width:
            raise PrefillWorkspaceError("hidden output overlaps live activation")
        invalid = {name: width for name, width in self.widths.items() if width <= 0}
        if invalid:
            raise PrefillWorkspaceError(
                f"Llama Prefill workspace widths must be positive: {invalid}"
            )


@dataclass
class PrefillWorkspaceExecution:
    """Immutable identity plus mutable profile ownership for one Future."""

    generation: int = 0
    phase_slot: int = 0
    handoff_generation: int = 0
    num_tokens: int = 0
    profile_id: str = ""
    profile_epoch: int = 0
    enabled: bool = False
    fallback_reason: str = ""


@dataclass
class _PrefillWorkspaceSlot:
    """Physical arena and asynchronous ownership state for one phase slot."""

    state: str = "free"
    active_generation: int = 0
    active_handoff_generation: int = 0
    active_tokens: int = 0
    active_profile_id: str = ""
    active_profile_epoch: int = 0
    completion_event: Any | None = None
    storage: torch.Tensor | None = None
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    arena_bytes: int = 0
    allocation_reserved_delta_bytes: int = 0
    allocation_allocated_delta_bytes: int = 0
    capacity_tokens: int = 0
    views_exported: bool = False
    retired: list[tuple[Any, Any, int]] = field(default_factory=list)
    attention_stream: Any | None = None
    ring_borrow: Any | None = None


@dataclass
class PrefillRingStateBorrow:
    """Non-cached views, valid until same-stream Ring finalization is submitted."""

    manager: Any
    execution: PrefillWorkspaceExecution
    tensors: tuple[Any, Any, Any]
    stream: Any
    bytes: int

    def release(self) -> None:
        self.manager.release_ring_state(self)


def _workspace_opt_in(feature: str) -> bool:
    # Deliberately independent of the older memory/execution bundles.
    name = "VLLM_ISOLATION_" + feature
    value = os.getenv(name, "0")
    if value not in {"0", "1"}:
        raise PrefillWorkspaceError(f"{name} must be 0 or 1")
    return value == "1"


_ACTIVE_PREFILL_WORKSPACE: ContextVar[
    PrefillWorkspaceExecution | None
] = ContextVar("active_pd_prefill_workspace", default=None)


class PrefillWorkspaceManager:
    """Own one fixed arena per bounded Prefill phase slot."""

    _STATE_FREE = "free"
    _STATE_ACTIVE = "active"
    _STATE_PENDING = "pending"

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        strict: bool | None = None,
        max_tokens: int | None = None,
        slot_count: int | None = None,
        rightsize: bool | None = None,
        sharing: bool | None = None,
    ) -> None:
        self._enabled_override = enabled
        self._strict_override = strict
        self._max_tokens_override = max_tokens
        self._slot_count_override = slot_count
        self._rightsize = (_workspace_opt_in("PREFILL_WORKSPACE_RIGHTSIZE")
                           if rightsize is None else rightsize)
        self.sharing = (_workspace_opt_in("RING_WORKSPACE_SHARING")
                        if sharing is None else sharing)
        self._lock = threading.RLock()
        self._next_generation = 0
        self._slots: dict[int, _PrefillWorkspaceSlot] = {}
        self._last_slot = 0
        self._layout: LlamaPrefillWorkspaceLayout | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        self._acquire_count = 0
        self._reuse_count = 0
        self._handoff_count = 0
        self._release_count = 0
        self._fallback_count = 0
        self._last_fallback_reason = ""
        self._growth_count = 0
        self._ring_borrow_count = 0
        self._ring_borrow_fallback_count = 0
        self._ring_borrow_last_reason = ""
        self._ring_borrow_peak_bytes = 0

    @property
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return prefill_workspace_requested()

    @property
    def strict(self) -> bool:
        if self._strict_override is not None:
            return self._strict_override
        return prefill_workspace_strict()

    @property
    def max_tokens(self) -> int:
        if self._max_tokens_override is not None:
            if self._max_tokens_override <= 0:
                raise PrefillWorkspaceError("max_tokens must be positive")
            return self._max_tokens_override
        return prefill_workspace_max_tokens()

    @property
    def slot_count(self) -> int:
        if self._slot_count_override is not None:
            if self._slot_count_override <= 0:
                raise PrefillWorkspaceError("slot_count must be positive")
            return self._slot_count_override
        return prefill_workspace_slot_count()

    def _fallback_or_raise(self, reason: str) -> bool:
        self._fallback_count += 1
        self._last_fallback_reason = reason
        if self.strict:
            raise PrefillWorkspaceError(reason)
        return False

    def reject(self, reason: str) -> bool:
        """Record one unsupported contract and optionally fail closed."""

        with self._lock:
            return self._fallback_or_raise(reason)

    def validate_runtime_contract(self, phase_slot: int = 0) -> bool:
        """Validate bounded slot ownership before acquisition/allocation."""

        if not self.enabled:
            return False
        raw_depth = os.getenv("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH", "1")
        try:
            prefill_depth = int(raw_depth)
        except ValueError:
            return self._fallback_or_raise(
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH must be an integer"
            )
        if prefill_depth <= 0:
            return self._fallback_or_raise(
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH must be positive; "
                f"got {prefill_depth}"
            )
        if self.slot_count < prefill_depth:
            return self._fallback_or_raise(
                "fixed Prefill workspace slots must cover Prefill inflight "
                f"depth: slots={self.slot_count}, depth={prefill_depth}"
            )
        if phase_slot < 0 or phase_slot >= prefill_depth:
            return self._fallback_or_raise(
                "Prefill workspace phase slot is outside the configured "
                f"inflight depth: slot={phase_slot}, depth={prefill_depth}"
            )
        return True

    def _slot(self, phase_slot: int) -> _PrefillWorkspaceSlot:
        return self._slots.setdefault(phase_slot, _PrefillWorkspaceSlot())

    def acquire(
        self,
        *,
        profile_id: str,
        profile_epoch: int,
        num_tokens: int,
        phase_slot: int = 0,
        handoff_generation: int = 0,
        stream: Any | None = None,
    ) -> PrefillWorkspaceExecution:
        """Acquire the arena for one Prefill Future.

        A pending prior Future in the same phase slot is ordered onto
        ``stream`` with ``wait_event``. Different slots never share storage.
        No host synchronization is introduced.
        """

        with self._lock:
            if not self.validate_runtime_contract(phase_slot):
                return PrefillWorkspaceExecution(
                    fallback_reason=self._last_fallback_reason
                )
            if num_tokens <= 0:
                self._fallback_or_raise(
                    "fixed Prefill workspace requires positive tokens; "
                    f"got {num_tokens}"
                )
                return PrefillWorkspaceExecution(
                    fallback_reason=self._last_fallback_reason
                )
            if num_tokens > self.max_tokens:
                self._fallback_or_raise(
                    "scheduled Prefill exceeds fixed workspace capacity: "
                    f"scheduled={num_tokens}, max={self.max_tokens}"
                )
                return PrefillWorkspaceExecution(
                    fallback_reason=self._last_fallback_reason
                )
            slot = self._slot(phase_slot)
            if slot.state == self._STATE_ACTIVE:
                self._fallback_or_raise(
                    "fixed Prefill workspace slot acquired while another "
                    f"Future is active: slot={phase_slot}"
                )
                return PrefillWorkspaceExecution(
                    fallback_reason=self._last_fallback_reason
                )
            if slot.state == self._STATE_PENDING:
                if slot.completion_event is None:
                    raise PrefillWorkspaceError(
                        "pending Prefill workspace slot has no completion "
                        f"event: slot={phase_slot}"
                    )
                if stream is None or not hasattr(stream, "wait_event"):
                    self._fallback_or_raise(
                        "cross-Future Prefill workspace reuse requires a CUDA stream"
                    )
                    return PrefillWorkspaceExecution(
                        fallback_reason=self._last_fallback_reason
                    )
                stream.wait_event(slot.completion_event)
                self._reuse_count += 1

            # Reap only proven completions. Incomplete old allocations remain
            # bounded by geometric capacity growth, without a host wait.
            slot.retired = [entry for entry in slot.retired if not entry[1].query()]
            if self._rightsize and self._layout is not None:
                # Resume can bypass LlamaModel.forward's initial configuration.
                # Grow here, BEFORE a new generation can export any view.
                self.configure_llama(
                    layout=self._layout, dtype=self._dtype, device=self._device,
                    phase_slot=phase_slot, num_tokens=num_tokens,
                )

            self._next_generation += 1
            slot.active_generation = self._next_generation
            slot.active_handoff_generation = int(handoff_generation)
            slot.active_tokens = int(num_tokens)
            slot.active_profile_id = str(profile_id)
            slot.active_profile_epoch = int(profile_epoch)
            slot.completion_event = None
            slot.views_exported = False
            slot.state = self._STATE_ACTIVE
            self._last_slot = phase_slot
            self._acquire_count += 1
            return PrefillWorkspaceExecution(
                generation=slot.active_generation,
                phase_slot=phase_slot,
                handoff_generation=slot.active_handoff_generation,
                num_tokens=slot.active_tokens,
                profile_id=slot.active_profile_id,
                profile_epoch=slot.active_profile_epoch,
                enabled=True,
            )

    def handoff(
        self,
        execution: PrefillWorkspaceExecution,
        *,
        handoff_generation: int,
        source_profile_id: str,
        source_profile_epoch: int,
        target_profile_id: str,
        target_profile_epoch: int,
    ) -> None:
        """Transfer logical ownership at an already-completed layer event."""

        if not execution.enabled:
            return
        with self._lock:
            slot = self._assert_active_execution(execution)
            if slot.attention_stream is not None or slot.ring_borrow is not None:
                raise PrefillWorkspaceError("workspace handoff inside Attention")
            if (
                execution.handoff_generation > 0
                and execution.handoff_generation != int(handoff_generation)
            ):
                raise PrefillWorkspaceError(
                    "workspace/controller handoff generation mismatch: "
                    f"workspace={execution.handoff_generation}, "
                    f"controller={handoff_generation}"
                )
            if (
                slot.active_profile_id != source_profile_id
                or slot.active_profile_epoch != int(source_profile_epoch)
            ):
                raise PrefillWorkspaceError(
                    "workspace handoff source is not its current owner: "
                    f"owner={slot.active_profile_id}@{slot.active_profile_epoch}, "
                    f"source={source_profile_id}@{source_profile_epoch}"
                )
            slot.active_profile_id = str(target_profile_id)
            slot.active_profile_epoch = int(target_profile_epoch)
            execution.profile_id = slot.active_profile_id
            execution.profile_epoch = slot.active_profile_epoch
            self._handoff_count += 1

    def finish(
        self,
        execution: PrefillWorkspaceExecution,
        *,
        completion_event: Any,
    ) -> None:
        """Release host ownership while retaining an async GPU completion."""

        if not execution.enabled:
            return
        with self._lock:
            slot = self._assert_active_execution(execution)
            if slot.attention_stream is not None or slot.ring_borrow is not None:
                raise PrefillWorkspaceError("workspace finish with live Attention borrow")
            if completion_event is None:
                raise PrefillWorkspaceError(
                    "fixed Prefill workspace release requires a completion event"
                )
            slot.completion_event = completion_event
            slot.state = self._STATE_PENDING
            slot.retired = [entry for entry in slot.retired if not entry[1].query()]
            # Keep the completed Future identity visible while its CUDA event
            # still owns the bytes.  The next acquire replaces these fields
            # only after it enqueues wait_event on the new profile stream.
            self._release_count += 1

    def _assert_active_execution(
        self, execution: PrefillWorkspaceExecution
    ) -> _PrefillWorkspaceSlot:
        slot = self._slots.get(execution.phase_slot)
        if slot is None:
            raise PrefillWorkspaceError(
                "Prefill workspace execution references an unallocated slot: "
                f"slot={execution.phase_slot}"
            )
        if slot.state != self._STATE_ACTIVE:
            raise PrefillWorkspaceError(
                "workspace operation requires active state; "
                f"slot={execution.phase_slot}, got={slot.state}"
            )
        if execution.generation != slot.active_generation:
            raise PrefillWorkspaceError(
                "stale Prefill workspace execution: "
                f"slot={execution.phase_slot}, execution={execution.generation}, "
                f"active={slot.active_generation}"
            )
        return slot

    def configure_llama(
        self,
        *,
        layout: LlamaPrefillWorkspaceLayout,
        dtype: torch.dtype,
        device: torch.device,
        phase_slot: int = 0,
        allocator: Callable[..., torch.Tensor] = torch.empty,
        num_tokens: int | None = None,
    ) -> bool:
        """Allocate stable views; optional growth never moves an active view."""

        with self._lock:
            if not self.validate_runtime_contract(phase_slot):
                return False
            layout.validate()
            normalized_device = torch.device(device)
            if self._layout is not None and (
                self._layout != layout
                or self._dtype != dtype
                or self._device != normalized_device
            ):
                return self._fallback_or_raise(
                    "fixed Prefill workspace cannot change layout after "
                    "allocation"
                )
            slot = self._slot(phase_slot)
            capacity = self.max_tokens
            if self._rightsize:
                requested = num_tokens
                if requested is None and slot.state == self._STATE_ACTIVE:
                    requested = slot.active_tokens
                if requested is None or not 0 < requested <= self.max_tokens:
                    raise PrefillWorkspaceError("rightsize requires valid scheduled tokens")
                capacity = min(self.max_tokens, 1 << (int(requested) - 1).bit_length())
            if slot.storage is not None and slot.capacity_tokens >= capacity:
                return True
            if slot.storage is not None and slot.state == self._STATE_ACTIVE:
                raise PrefillWorkspaceError("cannot grow an active Prefill workspace")
            if slot.attention_stream is not None or slot.ring_borrow is not None:
                raise PrefillWorkspaceError("cannot allocate with live Attention borrow")
            if slot.state == self._STATE_ACTIVE and slot.views_exported:
                raise PrefillWorkspaceError("cannot allocate after exporting a view")
            prior = None
            if slot.storage is not None and slot.state == self._STATE_PENDING:
                if slot.completion_event is None:
                    raise PrefillWorkspaceError("growth lost prior completion owner")
                if not slot.completion_event.query():
                    prior = (slot.storage, slot.completion_event, slot.arena_bytes)

            before_allocated = 0
            before_reserved = 0
            if normalized_device.type == "cuda" and torch.cuda.is_available():
                before_allocated = torch.cuda.memory_allocated(normalized_device)
                before_reserved = torch.cuda.memory_reserved(normalized_device)

            # Attention intermediates are dead before the MLP begins, so both
            # stages deliberately alias the same physical arena.  Buffers
            # within one stage remain disjoint because producer and consumer
            # are concurrently live on the CUDA stream.
            offsets = layout.regions(capacity)
            total_elements = max(offset + size for offset, size in offsets.values())
            storage = allocator(
                total_elements,
                dtype=dtype,
                device=normalized_device,
            )
            buffers = {
                name: storage.narrow(0, offset, elements).view(
                    capacity, layout.widths[name]
                )
                for name, (offset, elements) in offsets.items()
            }
            # Publish only after allocation AND all views succeed. Allocation
            # failure preserves prior storage, completion and geometry.
            if prior is not None:
                slot.retired.append(prior)
            if slot.storage is not None:
                self._growth_count += 1
            slot.storage = storage
            slot.buffers = buffers
            slot.capacity_tokens = capacity
            self._layout = layout
            self._dtype = dtype
            self._device = normalized_device
            slot.arena_bytes = total_elements * storage.element_size()
            self._last_slot = phase_slot

            if normalized_device.type == "cuda" and torch.cuda.is_available():
                slot.allocation_allocated_delta_bytes = max(
                    0,
                    torch.cuda.memory_allocated(normalized_device)
                    - before_allocated,
                )
                slot.allocation_reserved_delta_bytes = max(
                    0,
                    torch.cuda.memory_reserved(normalized_device)
                    - before_reserved,
                )
            return True

    def buffer(
        self,
        execution: PrefillWorkspaceExecution,
        name: str,
        *,
        num_tokens: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return a prefix view owned by the active Prefill Future."""

        if not execution.enabled:
            return None
        with self._lock:
            slot = self._assert_active_execution(execution)
            if slot.ring_borrow is not None:
                raise PrefillWorkspaceError("workspace view requested during Ring borrow")
            if slot.storage is None or self._layout is None:
                self._fallback_or_raise(
                    "Llama fixed Prefill workspace was used before configuration"
                )
                return None
            if num_tokens != execution.num_tokens:
                self._fallback_or_raise(
                    "Prefill workspace token count changed within one Future: "
                    f"lease={execution.num_tokens}, tensor={num_tokens}"
                )
                return None
            expected_width = self._layout.widths.get(name)
            if expected_width != width:
                self._fallback_or_raise(
                    f"Prefill workspace buffer {name!r} width mismatch: "
                    f"layout={expected_width}, requested={width}"
                )
                return None
            if dtype != self._dtype or torch.device(device) != self._device:
                self._fallback_or_raise(
                    f"Prefill workspace buffer {name!r} dtype/device mismatch"
                )
                return None
            if slot.capacity_tokens < num_tokens:
                raise PrefillWorkspaceError("workspace capacity is below active token count")
            slot.views_exported = True
            return slot.buffers[name].narrow(0, 0, num_tokens)

    def begin_attention(self, execution: PrefillWorkspaceExecution, stream: Any) -> None:
        with self._lock:
            slot = self._assert_active_execution(execution)
            if slot.attention_stream is not None or slot.ring_borrow is not None:
                raise PrefillWorkspaceError("nested workspace Attention scope")
            slot.attention_stream = stream

    def end_attention(self, execution: PrefillWorkspaceExecution, stream: Any) -> None:
        with self._lock:
            slot = self._assert_active_execution(execution)
            if slot.ring_borrow is not None or slot.attention_stream != stream:
                raise PrefillWorkspaceError("Attention scope lost its stream/borrow owner")
            slot.attention_stream = None

    def borrow_ring_state(self, execution: PrefillWorkspaceExecution,
                          query: Any, stream: Any) -> PrefillRingStateBorrow | None:
        """Borrow FP32 accumulator space without touching live QKV/output."""
        with self._lock:
            slot = self._assert_active_execution(execution)
            if slot.ring_borrow is not None:
                raise PrefillWorkspaceError("nested Ring workspace borrow")
            if slot.attention_stream is not None and slot.attention_stream != stream:
                raise PrefillWorkspaceError("Ring borrow changed Attention stream")
            reason = ""
            tokens, heads, dim = map(int, query.shape)
            if not self.sharing or slot.attention_stream is None:
                reason = "outside_attention_scope"
            elif slot.storage is None or self._layout is None:
                reason = "workspace_unconfigured"
            elif (torch.device(query.device) != self._device
                  or query.dtype != self._dtype or tokens > execution.num_tokens
                  or min(tokens, heads, dim) <= 0):
                reason = "unsupported_query_geometry"
            if reason:
                self._ring_borrow_fallback_count += 1
                self._ring_borrow_last_reason = reason
                return None
            element_bytes = slot.storage.element_size()
            if element_bytes not in (2, 4):
                self._ring_borrow_fallback_count += 1
                self._ring_borrow_last_reason = "unsupported_element_size"
                return None
            # Norm is dead after QKV projection. QKV and attention output must
            # remain intact; their positions use the arena CAPACITY, not Q size.
            offset = slot.capacity_tokens * (
                self._layout.qkv_width + self._layout.attention_width)
            align = 4 // element_bytes
            offset = ((offset + align - 1) // align) * align
            count = tokens * heads
            state_elements = count * (dim + 2)
            storage_elements = state_elements * align
            if offset + storage_elements > slot.storage.numel():
                self._ring_borrow_fallback_count += 1
                self._ring_borrow_last_reason = "insufficient_dead_mlp_space"
                return None
            state = slot.storage.narrow(0, offset, storage_elements).view(torch.float32)
            tensors = (state.narrow(0, 0, count).view(tokens, heads),
                       state.narrow(0, count, count).view(tokens, heads),
                       state.narrow(0, 2 * count, count * dim).view(tokens, heads, dim))
            borrow = PrefillRingStateBorrow(self, execution, tensors, stream,
                                           state_elements * 4)
            slot.ring_borrow = borrow
            self._ring_borrow_count += 1
            self._ring_borrow_peak_bytes = max(self._ring_borrow_peak_bytes, borrow.bytes)
            self._ring_borrow_last_reason = ""
            return borrow

    def release_ring_state(self, borrow: PrefillRingStateBorrow) -> None:
        with self._lock:
            slot = self._assert_active_execution(borrow.execution)
            if slot.ring_borrow is not borrow or slot.attention_stream != borrow.stream:
                raise PrefillWorkspaceError("stale Ring workspace borrow release")
            slot.ring_borrow = None

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self._fallback_count += 1
            self._last_fallback_reason = reason

    def debug_snapshot(self) -> dict[str, float | int | str]:
        with self._lock:
            configured_slots = [
                slot for slot in self._slots.values() if slot.storage is not None
            ]
            active_slots = [
                phase_slot
                for phase_slot, slot in self._slots.items()
                if slot.state == self._STATE_ACTIVE
            ]
            pending_slots = [
                phase_slot
                for phase_slot, slot in self._slots.items()
                if slot.state == self._STATE_PENDING
            ]
            last_slot = self._slots.get(
                self._last_slot, _PrefillWorkspaceSlot()
            )
            slot_states = ",".join(
                (
                    f"{phase_slot}:"
                    f"{self._slots.get(phase_slot, _PrefillWorkspaceSlot()).state}"
                )
                for phase_slot in range(self.slot_count)
            )
            snapshot: dict[str, float | int | str] = {
                "pd_prefill_workspace_enabled": float(self.enabled),
                "pd_prefill_workspace_strict": float(self.strict),
                "pd_prefill_workspace_configured": float(
                    bool(configured_slots)
                ),
                "pd_prefill_workspace_max_tokens": self.max_tokens,
                "isolation_prefill_workspace_rightsize": int(self._rightsize),
                "isolation_ring_workspace_sharing": int(self.sharing),
                "pd_prefill_workspace_capacity_tokens": last_slot.capacity_tokens,
                "pd_prefill_workspace_growth_count": self._growth_count,
                "pd_prefill_workspace_retired_bytes": sum(
                    entry[2] for slot in self._slots.values() for entry in slot.retired),
                "pd_prefill_workspace_ring_borrow_count": self._ring_borrow_count,
                "pd_prefill_workspace_ring_borrow_fallback_count": self._ring_borrow_fallback_count,
                "pd_prefill_workspace_ring_borrow_last_reason": self._ring_borrow_last_reason,
                "pd_prefill_workspace_ring_borrow_peak_bytes": self._ring_borrow_peak_bytes,
                "pd_prefill_workspace_slot_count": self.slot_count,
                "pd_prefill_workspace_configured_slot_count": len(
                    configured_slots
                ),
                "pd_prefill_workspace_active_slot_count": len(active_slots),
                "pd_prefill_workspace_pending_slot_count": len(pending_slots),
                "pd_prefill_workspace_slot_states": slot_states,
                "pd_prefill_workspace_arena_bytes": sum(
                    slot.arena_bytes for slot in configured_slots
                ),
                "pd_prefill_workspace_state": last_slot.state,
                "pd_prefill_workspace_active_slot": self._last_slot,
                "pd_prefill_workspace_active_generation": (
                    last_slot.active_generation
                ),
                "pd_prefill_workspace_active_tokens": last_slot.active_tokens,
                "pd_prefill_workspace_active_profile_id": (
                    last_slot.active_profile_id
                ),
                "pd_prefill_workspace_active_profile_epoch": (
                    last_slot.active_profile_epoch
                ),
                "pd_prefill_workspace_acquire_count": self._acquire_count,
                "isolation_prefill_extended_workspace": int(
                    self._layout is not None and bool(self._layout.hidden_width)
                ),
                "pd_prefill_workspace_reuse_count": self._reuse_count,
                "pd_prefill_workspace_handoff_count": self._handoff_count,
                "pd_prefill_workspace_release_count": self._release_count,
                "pd_prefill_workspace_fallback_count": self._fallback_count,
                "pd_prefill_workspace_last_fallback_reason": (
                    self._last_fallback_reason
                ),
                "pd_prefill_workspace_allocation_allocated_delta_bytes": (
                    sum(
                        slot.allocation_allocated_delta_bytes
                        for slot in configured_slots
                    )
                ),
                "pd_prefill_workspace_allocation_reserved_delta_bytes": (
                    sum(
                        slot.allocation_reserved_delta_bytes
                        for slot in configured_slots
                    )
                ),
            }
            for phase_slot, slot in sorted(self._slots.items()):
                prefix = f"pd_prefill_workspace_slot_{phase_slot}"
                snapshot[f"{prefix}_configured"] = float(
                    slot.storage is not None
                )
                snapshot[f"{prefix}_state"] = slot.state
                snapshot[f"{prefix}_generation"] = slot.active_generation
                snapshot[f"{prefix}_arena_bytes"] = slot.arena_bytes
                snapshot[f"{prefix}_capacity_tokens"] = slot.capacity_tokens
            if self._device is not None and self._device.type == "cuda":
                snapshot["pd_prefill_workspace_cuda_allocated_bytes"] = (
                    torch.cuda.memory_allocated(self._device)
                )
                snapshot["pd_prefill_workspace_cuda_reserved_bytes"] = (
                    torch.cuda.memory_reserved(self._device)
                )
                snapshot["pd_prefill_workspace_cuda_max_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(self._device)
                )
            return snapshot

PD_PREFILL_WORKSPACE = PrefillWorkspaceManager()


def prefill_workspace_ring_sharing_requested() -> bool:
    """Freeze at Attention construction so disabled forwards keep their path."""
    return PD_PREFILL_WORKSPACE.sharing


@contextmanager
def prefill_workspace_attention() -> Iterator[None]:
    """Only the Attention body may lend scratch, never projection/MLP/yield."""
    execution = _ACTIVE_PREFILL_WORKSPACE.get()
    if not PD_PREFILL_WORKSPACE.sharing or execution is None or not execution.enabled:
        yield
        return
    stream = torch.cuda.current_stream()
    PD_PREFILL_WORKSPACE.begin_attention(execution, stream)
    try:
        yield
    finally:
        PD_PREFILL_WORKSPACE.end_attention(execution, torch.cuda.current_stream())


def active_prefill_workspace_ring_state(query: Any) -> PrefillRingStateBorrow | None:
    execution = _ACTIVE_PREFILL_WORKSPACE.get()
    if (not PD_PREFILL_WORKSPACE.sharing or execution is None or not execution.enabled
            or query.device.type != "cuda" or torch.cuda.is_current_stream_capturing()):
        return None
    return PD_PREFILL_WORKSPACE.borrow_ring_state(
        execution, query, torch.cuda.current_stream(query.device))


@contextmanager
def prefill_workspace_execution(
    phase: str | None,
    *,
    active: bool,
    profile_id: str,
    profile_epoch: int,
    num_tokens: int,
    phase_slot: int = 0,
    handoff_generation: int = 0,
) -> Iterator[PrefillWorkspaceExecution]:
    """Bind one model execution to its process-wide Prefill phase-slot arena."""

    if (
        not prefill_workspace_requested()
        or phase != "prefill"
        or not active
    ):
        yield PrefillWorkspaceExecution()
        return

    stream = torch.cuda.current_stream()
    execution = PD_PREFILL_WORKSPACE.acquire(
        profile_id=profile_id,
        profile_epoch=profile_epoch,
        num_tokens=num_tokens,
        phase_slot=phase_slot,
        handoff_generation=handoff_generation,
        stream=stream,
    )
    token = _ACTIVE_PREFILL_WORKSPACE.set(execution)
    try:
        yield execution
    finally:
        _ACTIVE_PREFILL_WORKSPACE.reset(token)
        if execution.enabled:
            completion_event = torch.cuda.Event()
            completion_event.record(torch.cuda.current_stream())
            PD_PREFILL_WORKSPACE.finish(
                execution,
                completion_event=completion_event,
            )


def prepare_active_llama_prefill_workspace(
    *,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    qkv_width: int,
    dtype: torch.dtype,
    device: torch.device,
    tensor_parallel_size: int,
    quantized: bool,
    lora_enabled: bool,
    opmux_enabled: bool,
    mixed_phase_layer_enabled: bool,
    layer_path_supported: bool,
) -> bool:
    """Configure the supported dense Llama layout for the active Future."""

    execution = _ACTIVE_PREFILL_WORKSPACE.get()
    if execution is None or not execution.enabled:
        return False
    if torch.device(device).type != "cuda":
        return PD_PREFILL_WORKSPACE.reject(
            "fixed Llama Prefill workspace requires a CUDA device"
        )
    if tensor_parallel_size != 1:
        return PD_PREFILL_WORKSPACE.reject(
            "fixed Llama Prefill workspace currently requires tensor parallel size 1"
        )
    if quantized:
        return PD_PREFILL_WORKSPACE.reject(
            "fixed Llama Prefill workspace currently supports unquantized weights"
        )
    if lora_enabled:
        return PD_PREFILL_WORKSPACE.reject(
            "fixed Llama Prefill workspace does not bypass LoRA adapters"
        )
    if opmux_enabled or mixed_phase_layer_enabled:
        return PD_PREFILL_WORKSPACE.reject(
            "fixed Llama Prefill workspace requires phase-pure dense layers"
        )
    if not layer_path_supported:
        return PD_PREFILL_WORKSPACE.reject(
            "model layer does not implement the fixed Prefill workspace path"
        )
    if num_tokens != execution.num_tokens:
        return PD_PREFILL_WORKSPACE.reject(
            "model input tokens do not match the Prefill Future lease: "
            f"model={num_tokens}, lease={execution.num_tokens}"
        )
    layout = LlamaPrefillWorkspaceLayout(
        qkv_width=qkv_width,
        attention_width=hidden_size,
        gate_up_width=2 * intermediate_size,
        activation_width=intermediate_size,
        hidden_width=(
            hidden_size if isolation_memory_enabled("PREFILL_EXTENDED_WORKSPACE")
            else 0
        ),
    )
    return PD_PREFILL_WORKSPACE.configure_llama(
        layout=layout,
        dtype=dtype,
        device=torch.device(device),
        phase_slot=execution.phase_slot,
    )


def active_prefill_workspace_buffer(
    name: str,
    reference: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor | None:
    """Return an arena view matching ``reference``'s token dimension."""

    execution = _ACTIVE_PREFILL_WORKSPACE.get()
    if execution is None or not execution.enabled:
        return None
    return PD_PREFILL_WORKSPACE.buffer(
        execution,
        name,
        num_tokens=int(reference.shape[0]),
        width=width,
        dtype=reference.dtype,
        device=reference.device,
    )


def handoff_active_prefill_workspace(
    *,
    handoff_generation: int,
    source_profile_id: str,
    source_profile_epoch: int,
    target_profile_id: str,
    target_profile_epoch: int,
) -> None:
    """Move arena ownership with the existing layerwise CUDA boundary."""

    execution = _ACTIVE_PREFILL_WORKSPACE.get()
    if execution is None or not execution.enabled:
        return
    PD_PREFILL_WORKSPACE.handoff(
        execution,
        handoff_generation=handoff_generation,
        source_profile_id=source_profile_id,
        source_profile_epoch=source_profile_epoch,
        target_profile_id=target_profile_id,
        target_profile_epoch=target_profile_epoch,
    )


def prefill_workspace_debug_snapshot() -> dict[str, float | int | str]:
    return PD_PREFILL_WORKSPACE.debug_snapshot()
