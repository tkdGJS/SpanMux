# SPDX-License-Identifier: Apache-2.0
"""Default-off execution scope for TTFT-oriented Prefill layer groups.

The default execution scope is tensor-free and only records contiguous decoder
layer groups.  A second, separately gated mode may return a bounded metadata
boundary to EngineCore while the worker-owned continuation retains tensors in
its phase slot.  The boundary itself never owns CUDA or KV storage.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


_ENABLE_ENV = "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION"
_GROUP_LAYERS_ENV = "VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS"
_PREEMPTION_ENV = "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION"
_KV_CHECKPOINT_ENV = "VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN"
_WORK_BINDING_ENV = "VLLM_KV_RETENTION_PREFILL_WORK_BINDING"

_YIELD_REQUEST_LOCK = threading.Lock()
_YIELD_REQUESTS: set[
    tuple[int, int, int, tuple[tuple[str, int], ...]]
] = set()


class TTFTLayerwisePrefillError(RuntimeError):
    """Raised when a layer-group execution contract is inconsistent."""


def kv_retention_exact_phase_slot(execution_key: str) -> tuple[str, int]:
    """Parse one cleanup target; a bare phase is slot zero, never a wildcard."""

    phase, separator, raw_slot = execution_key.partition(":")
    if phase not in {"decode", "prefill"} or (
        separator and (not raw_slot.isascii() or not raw_slot.isdecimal())
    ):
        raise ValueError(f"invalid exact phase slot: {execution_key!r}")
    return phase, int(raw_slot) if separator else 0


@dataclass(frozen=True)
class KVRetentionPhaseOwner:
    """Small process-local cleanup capability, not proof of CUDA completion.

    UniProc retains the original output through execute/continuation/cleanup.
    Object identity also distinguishes legacy outputs without submit counters.
    Never send this identity across processes or retain tensors in this record.
    SM retargeting does not change the output owner.
    """

    execution_key: str
    scheduler_iteration_id: int
    phase_submit_seq: int
    scheduler_output_id: int
    binding: KVRetentionExecutionBinding | None = None

    @classmethod
    def from_output(
        cls, execution_key: str, output: Any,
        binding: KVRetentionExecutionBinding | None = None,
    ) -> KVRetentionPhaseOwner:
        phase, slot = kv_retention_exact_phase_slot(execution_key)
        if int(getattr(output, "pd_phase_slot", 0)) != slot:
            raise RuntimeError("phase cleanup slot does not match SchedulerOutput")
        if getattr(output, "kv_retention_prefill_work", None) is not None:
            if binding is None:
                raise RuntimeError("Prefill work requires an explicit execution binding")
        if binding is not None:
            binding.validate_output(execution_key, output)
        return cls(
            phase if slot == 0 else f"{phase}:{slot}",
            int(getattr(output, "scheduler_iteration_id", -1)),
            int(getattr(output, "pd_phase_submit_seq", -1)),
            id(output),
            binding,
        )

    def validate_key(self, execution_key: str) -> tuple[str, int]:
        phase, slot = kv_retention_exact_phase_slot(execution_key)
        canonical = phase if slot == 0 else f"{phase}:{slot}"
        if self.execution_key != canonical:
            raise RuntimeError("phase cleanup key does not match expected owner")
        return phase, slot


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def ttft_layerwise_prefill_execution_requested() -> bool:
    """Return whether the experimental same-Future group path is requested."""

    return _env_flag(_ENABLE_ENV, False)


def ttft_layerwise_prefill_preemption_requested() -> bool:
    """Return whether Engine-visible layer-group continuation is requested."""

    return _env_flag(_PREEMPTION_ENV, False)


def ttft_prefill_continuation_mode_allowed(
    prefill_mode: str, *, p1_ring_sm_handoff: bool = False,
) -> bool:
    """Shared Engine/Worker eligibility; Ring permits SM yield, not parking.

    Native-origin private Ring keeps mode ``native`` and its existing offload
    contract. Ordinary Ring retains its original slot, metadata and KV owner.
    Unknown/auto execution authority is not a continuation capability.
    """
    return prefill_mode == "native" or (
        p1_ring_sm_handoff and prefill_mode == "ring")


def kv_retention_layerwise_checkpoint_requested() -> bool:
    """Opt in to metadata validation, NOT to offloading or releasing KV."""

    return _env_flag(_KV_CHECKPOINT_ENV, False)


def kv_retention_prefill_work_binding_requested() -> bool:
    """Opt in to scalar work/segment validation, not P1 slot lending."""
    return _env_flag(_WORK_BINDING_ENV, False)


@dataclass(frozen=True)
class KVRetentionPrefillWork:
    """Immutable original token transaction; no tensor or physical KV owner."""

    work_id: int
    scheduler_output_id: int
    scheduler_iteration_id: int
    phase_submit_seq: int
    # request ID, request epoch, pre-chunk position, number of scheduled tokens.
    rows: tuple[tuple[str, int, int, int], ...]
    # request ID, request epoch, claim generation, phase, placeholder count.
    claims: tuple[tuple[str, int, int, str, int], ...]

    @classmethod
    def capture(cls, work_id: int, output: Any, epochs: dict[str, int]):
        if not getattr(output, "pd_output_ownership_enabled", False):
            raise RuntimeError("Prefill work requires output placeholder ownership")
        starts = {row.req_id: int(row.num_computed_tokens)
                  for row in output.scheduled_new_reqs}
        cached = output.scheduled_cached_reqs
        if len(cached.req_ids) != len(cached.num_computed_tokens):
            raise RuntimeError("Prefill work cached row lengths differ")
        for req_id, start in zip(cached.req_ids, cached.num_computed_tokens):
            if req_id in starts:
                raise RuntimeError("Prefill work contains duplicate input rows")
            starts[req_id] = int(start)
        rows = tuple((req_id, int(epochs[req_id]), starts[req_id], int(count))
                     for req_id, count in output.num_scheduled_tokens.items())
        if (not rows or work_id <= 0
                or any(epoch < 0 or start < 0 or count <= 0
                       for _, epoch, start, count in rows)
                or sum(row[3] for row in rows) != output.total_num_scheduled_tokens):
            raise RuntimeError("invalid Prefill logical token ranges")
        claims = []
        owners = output.pd_output_placeholder_owners
        if set(owners).difference(output.num_scheduled_tokens):
            raise RuntimeError("Prefill work contains an unrelated output claim")
        for req_id, epoch, _, _ in rows:
            claim = owners.get(req_id)
            if claim is None:
                continue  # A non-final Prefill chunk does not reserve a token.
            if (len(claim) != 4 or int(claim[0]) != epoch
                    or int(claim[1]) < 0 or claim[2] != "prefill"
                    or int(claim[3]) <= 0):
                raise RuntimeError("invalid Prefill output claim")
            claims.append((req_id, epoch, int(claim[1]), claim[2], int(claim[3])))
        return cls(work_id, id(output), int(output.scheduler_iteration_id),
                   int(output.pd_phase_submit_seq), rows, tuple(claims))

    def validate_output(self, output: Any) -> None:
        if getattr(output, "kv_retention_prefill_work", None) != self:
            raise RuntimeError("Prefill logical work was replaced")
        epochs = {row[0]: row[1] for row in self.rows}
        if self.capture(self.work_id, output, epochs) != self:
            raise RuntimeError("Prefill original token transaction changed")

    def validate_epochs(self, epochs: dict[str, int]) -> None:
        if any(epochs.get(req_id) != epoch for req_id, epoch, _, _ in self.rows):
            raise RuntimeError("Prefill request epoch changed")


@dataclass(frozen=True)
class KVRetentionExecutionBinding:
    """Captured RPC capability for ONE segment, not permission to free storage."""

    work: KVRetentionPrefillWork
    segment_id: int
    slot: int
    slot_generation: int
    profile_id: str
    profile_epoch: int
    start_layer: int

    def validate_output(self, execution_key: str, output: Any) -> None:
        if (kv_retention_exact_phase_slot(execution_key) != ("prefill", self.slot)
                or int(output.pd_phase_slot) != self.slot
                or self.segment_id <= 0 or self.slot_generation <= 0
                or self.start_layer < 0):
            raise RuntimeError("invalid Prefill execution binding")
        self.work.validate_output(output)

    def validate_submission(self, execution_key: str, output: Any) -> None:
        self.validate_output(execution_key, output)
        if (str(output.pd_sm_profile_id), int(output.pd_sm_profile_epoch)) != (
                self.profile_id, self.profile_epoch):
            raise RuntimeError("Prefill binding SM target changed before execution")

    def validate_successor(self, previous: KVRetentionExecutionBinding,
                           next_layer: int) -> None:
        if (self.work != previous.work or self.slot != previous.slot
                or self.segment_id != previous.segment_id + 1
                or self.slot_generation <= previous.slot_generation
                or self.start_layer != next_layer
                or next_layer <= previous.start_layer
                or self.profile_epoch < previous.profile_epoch
                or (self.profile_id != previous.profile_id
                    and self.profile_epoch == previous.profile_epoch)):
            raise RuntimeError("invalid Prefill segment successor")

    def debug(self) -> dict[str, int]:
        return {"kv_retention_work_id": self.work.work_id,
                "kv_retention_segment_id": self.segment_id,
                "kv_retention_slot_generation": self.slot_generation}


@dataclass
class _KVRetentionWorkRecord:
    binding: KVRetentionExecutionBinding
    state: str = "active"
    next_layer: int = 0


class KVRetentionPrefillWorkRegistry:
    """Engine-thread scalar ledger, bounded by active plus opted-in parked work.

    Suspension alone does NOT lend a slot: P1 needs an explicit park ACK.
    No method schedules tokens,
    frees KV, resolves an output claim, waits on CUDA, or owns a tensor/Future.
    Call finish only after the existing terminal action has succeeded.
    """

    def __init__(self, slots: int, *, parked_capacity: int = 0):
        if slots < 1:
            raise ValueError("Prefill work registry needs at least one slot")
        self._generations = [0] * slots
        if parked_capacity not in (0, 1) or (parked_capacity and slots != 1):
            raise ValueError("parked work is supported only with P1 and one parked batch")
        self._parked_capacity = parked_capacity
        self._records: dict[int, _KVRetentionWorkRecord] = {}
        self._next_work_id = 1

    def __len__(self) -> int:
        return len(self._records)

    def _binding(self, work, output, segment, start):
        slot = int(output.pd_phase_slot)
        if not 0 <= slot < len(self._generations):
            raise RuntimeError("Prefill binding exceeds phase slot budget")
        return KVRetentionExecutionBinding(
            work, segment, slot, self._generations[slot] + 1,
            str(output.pd_sm_profile_id), int(output.pd_sm_profile_epoch), start)

    def start(self, output: Any, epochs: dict[str, int]):
        if getattr(output, "kv_retention_prefill_work", None) is not None:
            raise RuntimeError("Prefill work cannot be registered twice")
        work = KVRetentionPrefillWork.capture(self._next_work_id, output, epochs)
        binding = self._binding(work, output, 1, 0)
        ids = {row[0] for row in work.rows}
        if len(self) >= len(self._generations) + self._parked_capacity or any(
                (record.binding.slot == binding.slot and record.state != "parked")
                or ids.intersection(row[0] for row in record.binding.work.rows)
                for record in self._records.values()):
            raise RuntimeError("Prefill work overlaps a retained owner or slot")
        # Publish only after all validation. An RPC failure retains this owner
        # for fatal teardown, never pretends the GPU completed.
        output.kv_retention_prefill_work = work
        self._records[work.work_id] = _KVRetentionWorkRecord(binding)
        self._generations[binding.slot] = binding.slot_generation
        self._next_work_id += 1
        return binding

    def require(self, binding: KVRetentionExecutionBinding, output: Any = None):
        record = self._records.get(binding.work.work_id)
        if record is None or record.binding != binding:
            raise RuntimeError("stale or unknown Prefill execution segment")
        if output is not None:
            binding.work.validate_output(output)
        return record

    def suspend(self, binding, next_layer: int) -> None:
        record = self.require(binding)
        if record.state != "active" or next_layer <= binding.start_layer:
            raise RuntimeError("Prefill segment cannot suspend at this boundary")
        record.state, record.next_layer = "suspended", next_layer

    def park(self, binding) -> None:
        record = self.require(binding)
        if (record.state != "suspended" or not self._parked_capacity
                or sum(r.state == "parked" for r in self._records.values()) >= self._parked_capacity):
            raise RuntimeError("Prefill work cannot lend this execution slot")
        record.state = "parked"

    def resume(self, output: Any, epochs: dict[str, int]):
        work = output.kv_retention_prefill_work
        work.validate_output(output)
        work.validate_epochs(epochs)
        record = self._records.get(work.work_id)
        if record is None or record.state not in {"suspended", "parked"}:
            raise RuntimeError("Prefill resume requires a suspended work")
        if any(other is not record and other.state != "parked"
               and other.binding.slot == int(output.pd_phase_slot)
               for other in self._records.values()):
            raise RuntimeError("Prefill resume slot is still owned by another work")
        binding = self._binding(work, output, record.binding.segment_id + 1,
                                record.next_layer)
        binding.validate_successor(record.binding, record.next_layer)
        record.binding, record.state = binding, "active"
        self._generations[binding.slot] = binding.slot_generation
        return binding

    def sampling(self, binding) -> None:
        record = self.require(binding)
        if record.state != "active":
            raise RuntimeError("Prefill sampling requires final active execute")
        record.state = "sampling"

    def finish(self, binding, *, cancelled: bool = False) -> None:
        record = self.require(binding)
        if record.state in {"suspended", "parked"} and not cancelled:
            raise RuntimeError("suspended Prefill work has no final output")
        del self._records[binding.work.work_id]


def ttft_layerwise_prefill_group_layers() -> int:
    """Return the number of consecutive decoder layers in one group."""

    raw_value = os.getenv(_GROUP_LAYERS_ENV, "2")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise TTFTLayerwisePrefillError(
            f"{_GROUP_LAYERS_ENV} must be an integer"
        ) from exc
    if value <= 0:
        raise TTFTLayerwisePrefillError(
            f"{_GROUP_LAYERS_ENV} must be positive"
        )
    return value


def _normalize_scheduled_tokens(
    *,
    request_ids: tuple[str, ...],
    scheduled_tokens: tuple[tuple[str, int], ...],
    required: bool,
) -> tuple[tuple[str, int], ...]:
    """Return one ordered, immutable whole-batch execution identity."""

    normalized_request_ids = tuple(str(value) for value in request_ids)
    normalized_scheduled_tokens = tuple(
        (str(request_id), int(num_tokens))
        for request_id, num_tokens in scheduled_tokens
    )
    if not normalized_scheduled_tokens:
        if required:
            raise TTFTLayerwisePrefillError(
                "resumable Prefill requires exact per-request scheduled "
                "token counts"
            )
        return ()
    scheduled_request_ids = tuple(
        request_id for request_id, _ in normalized_scheduled_tokens
    )
    if scheduled_request_ids != normalized_request_ids:
        raise TTFTLayerwisePrefillError(
            "Prefill batch request order does not match its scheduled-token "
            f"owner: request_ids={normalized_request_ids!r}, "
            f"scheduled={scheduled_request_ids!r}"
        )
    if len(set(scheduled_request_ids)) != len(scheduled_request_ids):
        raise TTFTLayerwisePrefillError(
            "Prefill batch owner contains duplicate request IDs"
        )
    if any(num_tokens <= 0 for _, num_tokens in normalized_scheduled_tokens):
        raise TTFTLayerwisePrefillError(
            "Prefill batch owner requires a positive token count for every "
            "request"
        )
    return normalized_scheduled_tokens


def should_preempt_ttft_layerwise_prefill_batch(
    *,
    active_srpt_keys: tuple[tuple[int, float, str], ...],
    candidate_srpt_key: tuple[int, float, str] | None,
) -> bool:
    """Return whether an outside SRPT request displaces the active tail.

    The execution substrate can suspend only the whole active batch.  A new
    request therefore triggers a yield when it sorts before at least one
    active member, represented by the batch's largest (tail) SRPT key.  The
    head may temporarily yield with the tail, but no activation row is split
    from the immutable SchedulerOutput transaction.
    """

    if not active_srpt_keys or candidate_srpt_key is None:
        return False
    return candidate_srpt_key < max(active_srpt_keys)


def _yield_owner_key(
    *,
    scheduler_iteration_id: int,
    phase_submit_seq: int,
    phase_slot: int,
    scheduled_tokens: tuple[tuple[str, int], ...],
) -> tuple[int, int, int, tuple[tuple[str, int], ...]]:
    return (
        int(scheduler_iteration_id),
        int(phase_submit_seq),
        int(phase_slot),
        tuple(
            (str(request_id), int(num_tokens))
            for request_id, num_tokens in scheduled_tokens
        ),
    )


def request_ttft_layerwise_prefill_yield(
    *,
    scheduler_iteration_id: int,
    phase_submit_seq: int,
    phase_slot: int,
    scheduled_tokens: tuple[tuple[str, int], ...],
) -> bool:
    """Arm one Engine-requested yield for the exact active Prefill batch."""

    key = _yield_owner_key(
        scheduler_iteration_id=scheduler_iteration_id,
        phase_submit_seq=phase_submit_seq,
        phase_slot=phase_slot,
        scheduled_tokens=scheduled_tokens,
    )
    with _YIELD_REQUEST_LOCK:
        if key in _YIELD_REQUESTS:
            return False
        _YIELD_REQUESTS.add(key)
        return True


def cancel_ttft_layerwise_prefill_yield(
    *,
    scheduler_iteration_id: int,
    phase_submit_seq: int,
    phase_slot: int,
    scheduled_tokens: tuple[tuple[str, int], ...],
) -> bool:
    """Remove a stale signal after completion, abort, or slot cleanup."""

    key = _yield_owner_key(
        scheduler_iteration_id=scheduler_iteration_id,
        phase_submit_seq=phase_submit_seq,
        phase_slot=phase_slot,
        scheduled_tokens=scheduled_tokens,
    )
    with _YIELD_REQUEST_LOCK:
        if key not in _YIELD_REQUESTS:
            return False
        _YIELD_REQUESTS.remove(key)
        return True


def _consume_ttft_layerwise_prefill_yield(
    execution: "TTFTLayerwisePrefillExecution",
) -> bool:
    key = _yield_owner_key(
        scheduler_iteration_id=execution.scheduler_iteration_id,
        phase_submit_seq=execution.phase_submit_seq,
        phase_slot=execution.phase_slot,
        scheduled_tokens=execution.scheduled_tokens,
    )
    with _YIELD_REQUEST_LOCK:
        if key not in _YIELD_REQUESTS:
            return False
        _YIELD_REQUESTS.remove(key)
        return True


@dataclass
class TTFTLayerwisePrefillExecution:
    """Checked, tensor-free metadata for one model-runner invocation."""

    requested: bool = False
    active: bool = False
    phase: str = ""
    group_layers: int = 0
    preemption_requested: bool = False
    preemption_active: bool = False
    phase_slot: int = 0
    request_ids: tuple[str, ...] = ()
    scheduled_tokens: tuple[tuple[str, int], ...] = ()
    profile_id: str = ""
    profile_epoch: int = 0
    prefill_mode: str = "native"
    scheduler_iteration_id: int = -1
    phase_submit_seq: int = -1
    num_tokens: int = 0
    num_requests: int = 0
    group_count: int = 0
    executed_layers: int = 0
    first_layer: int = -1
    last_layer_exclusive: int = -1
    group_host_wall_ms_total: float = 0.0
    group_host_wall_ms_max: float = 0.0
    model_yield_count: int = 0
    scheduler_yield_count: int = 0
    resume_count: int = 0
    yield_check_count: int = 0
    yield_signal_consumed_count: int = 0
    _expected_next_layer: int | None = field(default=None, repr=False)

    def record_group(
        self,
        *,
        start_layer: int,
        end_layer: int,
        host_wall_s: float,
    ) -> None:
        """Record one completed, non-overlapping contiguous layer range."""

        if not self.active:
            raise TTFTLayerwisePrefillError(
                "cannot record a group outside active Prefill execution"
            )
        start_layer = int(start_layer)
        end_layer = int(end_layer)
        if start_layer < 0 or end_layer <= start_layer:
            raise TTFTLayerwisePrefillError(
                "layer group must be a positive half-open range"
            )
        if (
            self._expected_next_layer is not None
            and start_layer != self._expected_next_layer
        ):
            raise TTFTLayerwisePrefillError(
                "layer groups must be contiguous: "
                f"expected {self._expected_next_layer}, got {start_layer}"
            )
        group_width = end_layer - start_layer
        if group_width > self.group_layers:
            raise TTFTLayerwisePrefillError(
                f"layer group width {group_width} exceeds configured "
                f"maximum {self.group_layers}"
            )
        host_wall_ms = max(0.0, float(host_wall_s) * 1000.0)
        if self.group_count == 0:
            self.first_layer = start_layer
        self.group_count += 1
        self.executed_layers += group_width
        self.last_layer_exclusive = end_layer
        self.group_host_wall_ms_total += host_wall_ms
        self.group_host_wall_ms_max = max(
            self.group_host_wall_ms_max,
            host_wall_ms,
        )
        self._expected_next_layer = end_layer

    def as_debug_dict(self) -> dict[str, float | str]:
        """Return bounded scalar fields suitable for iteration tracing."""

        return {
            "ttft_layerwise_prefill_execution_requested": float(
                self.requested
            ),
            "ttft_layerwise_prefill_execution_active": float(self.active),
            "ttft_layerwise_prefill_execution_mode": (
                "resumable_layer_groups"
                if self.preemption_active
                else "same_future_layer_groups"
                if self.active
                else "legacy"
            ),
            "ttft_layerwise_prefill_preemption_requested": float(
                self.preemption_requested
            ),
            "ttft_layerwise_prefill_preemption_active": float(
                self.preemption_active
            ),
            "ttft_layerwise_prefill_phase_slot": float(self.phase_slot),
            "ttft_layerwise_prefill_profile_id": self.profile_id,
            "ttft_layerwise_prefill_profile_epoch": float(
                self.profile_epoch
            ),
            "ttft_layerwise_prefill_prefill_mode": self.prefill_mode,
            "ttft_layerwise_prefill_group_layers": float(self.group_layers),
            "ttft_layerwise_prefill_group_count": float(self.group_count),
            "ttft_layerwise_prefill_executed_layers": float(
                self.executed_layers
            ),
            "ttft_layerwise_prefill_first_layer": float(self.first_layer),
            "ttft_layerwise_prefill_last_layer_exclusive": float(
                self.last_layer_exclusive
            ),
            "ttft_layerwise_prefill_group_host_wall_ms_total": (
                self.group_host_wall_ms_total
            ),
            "ttft_layerwise_prefill_group_host_wall_ms_max": (
                self.group_host_wall_ms_max
            ),
            "ttft_layerwise_prefill_model_yield_count": float(
                self.model_yield_count
            ),
            "ttft_layerwise_prefill_scheduler_yield_count": float(
                self.scheduler_yield_count
            ),
            "ttft_layerwise_prefill_resume_count": float(
                self.resume_count
            ),
            "ttft_layerwise_prefill_yield_check_count": float(
                self.yield_check_count
            ),
            "ttft_layerwise_prefill_yield_signal_consumed_count": float(
                self.yield_signal_consumed_count
            ),
            "ttft_layerwise_prefill_scheduler_iteration_id": float(
                self.scheduler_iteration_id
            ),
            "ttft_layerwise_prefill_phase_submit_seq": float(
                self.phase_submit_seq
            ),
            "ttft_layerwise_prefill_num_tokens": float(self.num_tokens),
            "ttft_layerwise_prefill_num_requests": float(self.num_requests),
            "ttft_layerwise_prefill_batch_width": float(
                len(self.scheduled_tokens)
            ),
        }

    def make_boundary(
        self,
        *,
        next_layer: int,
        end_layer: int,
    ) -> "TTFTLayerwisePrefillBoundary":
        """Create the tensor-free result returned at one resumable boundary."""

        if not self.preemption_active:
            raise TTFTLayerwisePrefillError(
                "cannot yield without active layerwise Prefill preemption"
            )
        next_layer = int(next_layer)
        end_layer = int(end_layer)
        if not (0 <= next_layer < end_layer):
            raise TTFTLayerwisePrefillError(
                "a Prefill boundary must leave at least one decoder layer"
            )
        if next_layer != self.last_layer_exclusive:
            raise TTFTLayerwisePrefillError(
                "Prefill boundary does not match completed layer coverage"
            )
        self.model_yield_count += 1
        return TTFTLayerwisePrefillBoundary(
            scheduler_iteration_id=self.scheduler_iteration_id,
            phase_submit_seq=self.phase_submit_seq,
            phase_slot=self.phase_slot,
            request_ids=self.request_ids,
            scheduled_tokens=self.scheduled_tokens,
            profile_id=self.profile_id,
            profile_epoch=self.profile_epoch,
            prefill_mode=self.prefill_mode,
            next_layer=next_layer,
            end_layer=end_layer,
            model_yield_count=self.model_yield_count,
        )

    def consume_yield_request(self) -> bool:
        """Check one group boundary without forcing an Engine round trip."""

        if not self.preemption_active:
            return False
        self.yield_check_count += 1
        consumed = _consume_ttft_layerwise_prefill_yield(self)
        if consumed:
            self.yield_signal_consumed_count += 1
        return consumed

    def cancel_yield_request(self) -> bool:
        return cancel_ttft_layerwise_prefill_yield(
            scheduler_iteration_id=self.scheduler_iteration_id,
            phase_submit_seq=self.phase_submit_seq,
            phase_slot=self.phase_slot,
            scheduled_tokens=self.scheduled_tokens,
        )

    def validate_resume_owner(
        self,
        *,
        scheduler_iteration_id: int,
        phase_submit_seq: int,
        phase_slot: int,
        request_ids: tuple[str, ...],
        scheduled_tokens: tuple[tuple[str, int], ...],
        profile_id: str,
        profile_epoch: int,
        prefill_mode: str,
    ) -> None:
        """Validate immutable ownership and adopt a newer SM profile.

        A completed layer group is a real GPU quiescent boundary, so EngineCore
        may select a different GreenContext profile before the next group.  The
        request, scheduler output, phase slot, and Prefill mode are immutable;
        the profile epoch may only stay equal or move forward.
        """

        immutable_actual = (
            int(scheduler_iteration_id),
            int(phase_submit_seq),
            int(phase_slot),
            tuple(str(value) for value in request_ids),
            tuple(
                (str(request_id), int(num_tokens))
                for request_id, num_tokens in scheduled_tokens
            ),
            str(prefill_mode),
        )
        immutable_expected = (
            self.scheduler_iteration_id,
            self.phase_submit_seq,
            self.phase_slot,
            self.request_ids,
            self.scheduled_tokens,
            self.prefill_mode,
        )
        if immutable_actual != immutable_expected:
            raise TTFTLayerwisePrefillError(
                "Prefill continuation owner changed: "
                f"expected={immutable_expected!r}, "
                f"actual={immutable_actual!r}"
            )
        next_epoch = int(profile_epoch)
        if next_epoch < self.profile_epoch:
            raise TTFTLayerwisePrefillError(
                "Prefill continuation SM epoch moved backwards: "
                f"current={self.profile_id}@{self.profile_epoch}, "
                f"resume={profile_id}@{next_epoch}"
            )
        if next_epoch == self.profile_epoch and str(profile_id) != self.profile_id:
            raise TTFTLayerwisePrefillError(
                "Prefill continuation changed profile without a new epoch: "
                f"current={self.profile_id}@{self.profile_epoch}, "
                f"resume={profile_id}@{next_epoch}"
            )
        self.profile_id = str(profile_id)
        self.profile_epoch = next_epoch

    def note_scheduler_yield(self) -> None:
        """Acknowledge that EngineCore suspended rather than committed work."""

        if self.scheduler_yield_count >= self.model_yield_count:
            raise TTFTLayerwisePrefillError(
                "scheduler yield has no unmatched model boundary"
            )
        self.scheduler_yield_count += 1

    def note_resume(self) -> None:
        if self.resume_count >= self.model_yield_count:
            raise TTFTLayerwisePrefillError(
                "Prefill continuation resumed without a scheduler yield"
            )
        # A resume RPC is itself the acknowledgement that EngineCore consumed
        # (and did not commit) the preceding model boundary.  Pure state-machine
        # tests may acknowledge it explicitly through note_scheduler_yield().
        if self.scheduler_yield_count == self.resume_count:
            self.scheduler_yield_count += 1
        elif self.scheduler_yield_count != self.resume_count + 1:
            raise TTFTLayerwisePrefillError(
                "Prefill continuation has inconsistent yield ownership"
            )
        self.resume_count += 1


@dataclass
class TTFTLayerwisePrefillBoundary:
    """Tensor-free worker result; activation ownership stays in its slot."""

    scheduler_iteration_id: int
    phase_submit_seq: int
    phase_slot: int
    request_ids: tuple[str, ...]
    scheduled_tokens: tuple[tuple[str, int], ...]
    profile_id: str
    profile_epoch: int
    prefill_mode: str
    next_layer: int
    end_layer: int
    model_yield_count: int
    model_runner_debug_timings: dict[str, Any] = field(default_factory=dict)
    # Tensor-free durable copy receipt, distinct from a checkpoint plan.
    kv_retention_transfer: Any = field(default=None, repr=False)
    kv_retention_binding: KVRetentionExecutionBinding | None = field(
        default=None, repr=False)
    kv_retention_parked_ticket: Any = field(default=None, repr=False)

    def validate_owner(
        self,
        *,
        scheduler_iteration_id: int,
        phase_submit_seq: int,
        phase_slot: int,
        request_ids: tuple[str, ...],
        scheduled_tokens: tuple[tuple[str, int], ...],
        profile_id: str,
        profile_epoch: int,
        prefill_mode: str,
    ) -> None:
        actual = (
            int(scheduler_iteration_id),
            int(phase_submit_seq),
            int(phase_slot),
            tuple(request_ids),
            tuple(
                (str(request_id), int(num_tokens))
                for request_id, num_tokens in scheduled_tokens
            ),
            str(profile_id),
            int(profile_epoch),
            str(prefill_mode),
        )
        expected = (
            self.scheduler_iteration_id,
            self.phase_submit_seq,
            self.phase_slot,
            self.request_ids,
            self.scheduled_tokens,
            self.profile_id,
            self.profile_epoch,
            self.prefill_mode,
        )
        if actual != expected:
            raise TTFTLayerwisePrefillError(
                "Prefill continuation owner changed: "
                f"expected={expected!r}, actual={actual!r}"
            )


@dataclass(frozen=True)
class KVRetentionCheckpointRow:
    """CPU-only mapping of one request in the retained, native input batch.

    ``start_token`` is the worker's position BEFORE this chunk, not the
    scheduler's optimistic position after submission. Blocks are allocator
    blocks, with block zero reserved for the null placeholder.
    """

    request_id: str
    input_row: int
    start_token: int
    scheduled_tokens: int
    block_size: int
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise TTFTLayerwisePrefillError("checkpoint needs a request ID")
        if (
            any(type(value) is not int for value in (
                self.input_row, self.start_token, self.scheduled_tokens,
                self.block_size
            ))
            or self.input_row < 0
            or self.start_token < 0
            or self.scheduled_tokens <= 0
            or self.block_size <= 0
        ):
            raise TTFTLayerwisePrefillError("invalid checkpoint token range")
        if (
            not isinstance(self.block_ids, tuple)
            or any(type(block) is not int or block <= 0
                   for block in self.block_ids)
            or len(set(self.block_ids)) != len(self.block_ids)
        ):
            raise TTFTLayerwisePrefillError(
                "checkpoint requires concrete, distinct native blocks per row"
            )
        required = (
            self.start_token + self.scheduled_tokens + self.block_size - 1
        ) // self.block_size
        if len(self.block_ids) < required:
            raise TTFTLayerwisePrefillError(
                "checkpoint block table does not cover the submitted chunk"
            )


@dataclass(frozen=True)
class KVRetentionCheckpointPage:
    """A valid page range, not a copy-completion or allocator-free receipt."""

    request_id: str
    layer_name: str
    logical_page: int
    physical_block: int
    valid_tokens: int


def _kv_checkpoint_boundary_identity(
    boundary: TTFTLayerwisePrefillBoundary,
) -> tuple[Any, ...]:
    # Profile rebinding is checked by the existing execution owner. A legal
    # SM change does not invalidate these still-owned physical KV addresses.
    return (
        boundary.scheduler_iteration_id,
        boundary.phase_submit_seq,
        boundary.phase_slot,
        tuple(boundary.request_ids),
        tuple(boundary.scheduled_tokens),
        boundary.prefill_mode,
        boundary.next_layer,
        boundary.end_layer,
        boundary.model_yield_count,
    )


@dataclass(frozen=True)
class KVRetentionLayerwiseCheckpoint:
    """Immutable description of still-owned KV at a completed GPU boundary.

    No tensor, Future, provider readiness or allocator capability is retained.
    Shared prefix blocks may occur across rows: unique referenced blocks are
    NOT a reclaimable-capacity estimate. Actual D2H/transfer/Ring resume must
    be a separate transaction; this plan alone never authorizes any of them.
    """

    boundary_identity: tuple[Any, ...]
    layer_names: tuple[str, ...]
    rows: tuple[KVRetentionCheckpointRow, ...]
    next_layer: int

    @property
    def release_authorized(self) -> bool:
        return False

    def iter_layer_pages(
        self, layer_index: int
    ) -> Iterator[KVRetentionCheckpointPage]:
        if not 0 <= layer_index < len(self.layer_names):
            raise TTFTLayerwisePrefillError("checkpoint layer out of range")
        for row in self.rows:
            valid = row.start_token
            if layer_index < self.next_layer:
                valid += row.scheduled_tokens
            for page in range((valid + row.block_size - 1) // row.block_size):
                yield KVRetentionCheckpointPage(
                    request_id=row.request_id,
                    layer_name=self.layer_names[layer_index],
                    logical_page=page,
                    physical_block=row.block_ids[page],
                    valid_tokens=min(row.block_size, valid - page * row.block_size),
                )

    def validate_resume(
        self,
        *,
        boundary: TTFTLayerwisePrefillBoundary,
        layer_names: tuple[str, ...],
        rows: tuple[KVRetentionCheckpointRow, ...],
    ) -> None:
        if _kv_checkpoint_boundary_identity(boundary) != self.boundary_identity:
            raise TTFTLayerwisePrefillError(
                "checkpoint continuation owner or layer boundary changed"
            )
        if tuple(layer_names) != self.layer_names or tuple(rows) != self.rows:
            raise TTFTLayerwisePrefillError(
                "checkpoint KV mapping, token range or layer layout changed"
            )

    def validate_successor(self, successor: KVRetentionLayerwiseCheckpoint) -> None:
        """Do not silently rebaseline a stale mapping at a later boundary."""

        if (
            successor.boundary_identity[:6] != self.boundary_identity[:6]
            or successor.boundary_identity[-2] != self.boundary_identity[-2]
            or successor.boundary_identity[-1] != self.boundary_identity[-1] + 1
            or successor.next_layer <= self.next_layer
            or successor.rows != self.rows
            or successor.layer_names != self.layer_names
        ):
            raise TTFTLayerwisePrefillError(
                "checkpoint successor changed owner/layout or skipped a boundary"
            )

    def as_debug_dict(self) -> dict[str, float]:
        return {
            "kv_retention_checkpoint_plan_active": 1.0,
            "kv_retention_checkpoint_request_count": float(len(self.rows)),
            "kv_retention_checkpoint_next_layer": float(self.next_layer),
            "kv_retention_checkpoint_end_layer": float(len(self.layer_names)),
            "kv_retention_checkpoint_boundary_sequence": float(
                self.boundary_identity[-1]
            ),
            "kv_retention_checkpoint_unique_referenced_blocks": float(len({
                block for row in self.rows for block in row.block_ids
            })),
            "kv_retention_checkpoint_layer_token_coverage": float(sum(
                len(self.layer_names) * row.start_token
                + self.next_layer * row.scheduled_tokens
                for row in self.rows
            )),
            "kv_retention_checkpoint_release_authorized": 0.0,
        }


def build_kv_retention_layerwise_checkpoint(
    *,
    boundary: TTFTLayerwisePrefillBoundary,
    first_layer: int,
    layer_names: tuple[str, ...],
    rows: tuple[KVRetentionCheckpointRow, ...],
    gpu_boundary_complete: bool,
) -> KVRetentionLayerwiseCheckpoint:
    """Capture exact coverage after the runner's existing boundary event wait.

    Layer names come from actual decoder order, never provider numeric IDs
    (whose numbering follows a different, registration-order namespace).
    This function performs no synchronization, GPU operation or allocation.
    """

    if not gpu_boundary_complete:
        raise TTFTLayerwisePrefillError("checkpoint requires a completed boundary")
    if (
        boundary.prefill_mode != "native"
        or first_layer != 0
        or not 0 < boundary.next_layer < boundary.end_layer
        or boundary.end_layer != len(layer_names)
        or boundary.model_yield_count <= 0
        or min(boundary.scheduler_iteration_id, boundary.phase_submit_seq,
               boundary.phase_slot) < 0
        or not all(isinstance(name, str) and name for name in layer_names)
        or len(set(layer_names)) != len(layer_names)
    ):
        raise TTFTLayerwisePrefillError(
            "checkpoint requires an exact full-model native layer boundary"
        )
    tokens = _normalize_scheduled_tokens(
        request_ids=boundary.request_ids,
        scheduled_tokens=boundary.scheduled_tokens,
        required=True,
    )
    if tuple((row.request_id, row.scheduled_tokens) for row in rows) != tokens:
        raise TTFTLayerwisePrefillError("checkpoint rows do not match batch owner")
    if len({row.input_row for row in rows}) != len(rows):
        raise TTFTLayerwisePrefillError("checkpoint input rows must be distinct")
    if len({row.block_size for row in rows}) != 1:
        raise TTFTLayerwisePrefillError("checkpoint requires one block geometry")
    return KVRetentionLayerwiseCheckpoint(
        boundary_identity=_kv_checkpoint_boundary_identity(boundary),
        layer_names=tuple(layer_names),
        rows=tuple(rows),
        next_layer=boundary.next_layer,
    )


@dataclass
class TTFTLayerwisePrefillModelYield:
    """Worker-local model result that owns the Python tensor continuation.

    ``continuation`` captures hidden/residual tensors and therefore must never
    leave GPUModelRunner.  GPUModelRunner stores it in the phase-local slot and
    returns only ``boundary`` to EngineCore.
    """

    boundary: TTFTLayerwisePrefillBoundary
    continuation: Iterator[Any] = field(repr=False)


_ACTIVE_EXECUTION: ContextVar[
    TTFTLayerwisePrefillExecution | None
] = ContextVar("active_ttft_layerwise_prefill_execution", default=None)


def active_ttft_layerwise_prefill_execution(
) -> TTFTLayerwisePrefillExecution | None:
    """Return the active Prefill execution, if the opt-in path owns this call."""

    execution = _ACTIVE_EXECUTION.get()
    if execution is None or not execution.active:
        return None
    return execution


@contextmanager
def ttft_layerwise_prefill_execution(
    phase: str | None,
    *,
    scheduler_iteration_id: int = -1,
    phase_submit_seq: int = -1,
    num_tokens: int = 0,
    num_requests: int = 0,
    phase_slot: int = 0,
    request_ids: tuple[str, ...] = (),
    scheduled_tokens: tuple[tuple[str, int], ...] = (),
    profile_id: str = "",
    profile_epoch: int = 0,
    prefill_mode: str = "native",
    preemption_allowed: bool = False,
    existing_execution: TTFTLayerwisePrefillExecution | None = None,
) -> Iterator[TTFTLayerwisePrefillExecution]:
    """Install one default-off, phase-scoped execution descriptor."""

    requested = ttft_layerwise_prefill_execution_requested()
    preemption_requested = ttft_layerwise_prefill_preemption_requested()
    if kv_retention_layerwise_checkpoint_requested() and not (
        requested and preemption_requested
    ):
        raise TTFTLayerwisePrefillError(
            f"{_KV_CHECKPOINT_ENV}=1 requires {_ENABLE_ENV}=1 and "
            f"{_PREEMPTION_ENV}=1"
        )
    if preemption_requested and not requested:
        raise TTFTLayerwisePrefillError(
            f"{_PREEMPTION_ENV}=1 requires {_ENABLE_ENV}=1"
        )
    normalized_phase = str(phase or "").strip().lower()
    group_layers = ttft_layerwise_prefill_group_layers() if requested else 0
    active = requested and normalized_phase == "prefill"
    normalized_request_ids = tuple(str(value) for value in request_ids)
    normalized_scheduled_tokens = _normalize_scheduled_tokens(
        request_ids=normalized_request_ids,
        scheduled_tokens=scheduled_tokens,
        required=bool(
            active and preemption_requested and preemption_allowed
        ),
    )
    if existing_execution is None:
        execution = TTFTLayerwisePrefillExecution(
            requested=requested,
            active=active,
            phase=normalized_phase,
            group_layers=group_layers,
            preemption_requested=preemption_requested,
            preemption_active=bool(
                active and preemption_requested and preemption_allowed
            ),
            phase_slot=max(0, int(phase_slot)),
            request_ids=normalized_request_ids,
            scheduled_tokens=normalized_scheduled_tokens,
            profile_id=str(profile_id),
            profile_epoch=int(profile_epoch),
            prefill_mode=str(prefill_mode),
            scheduler_iteration_id=int(scheduler_iteration_id),
            phase_submit_seq=int(phase_submit_seq),
            num_tokens=max(0, int(num_tokens)),
            num_requests=max(0, int(num_requests)),
        )
    else:
        execution = existing_execution
        if not execution.preemption_active:
            raise TTFTLayerwisePrefillError(
                "only an active resumable execution may be re-entered"
            )
        execution.validate_resume_owner(
            scheduler_iteration_id=scheduler_iteration_id,
            phase_submit_seq=phase_submit_seq,
            phase_slot=phase_slot,
            request_ids=normalized_request_ids,
            scheduled_tokens=normalized_scheduled_tokens,
            profile_id=profile_id,
            profile_epoch=profile_epoch,
            prefill_mode=prefill_mode,
        )
        execution.note_resume()
    if not active:
        yield execution
        return

    current = _ACTIVE_EXECUTION.get()
    if current is not None and current.active:
        raise TTFTLayerwisePrefillError(
            "nested TTFT layerwise Prefill executions are unsupported"
        )
    token = _ACTIVE_EXECUTION.set(execution)
    try:
        yield execution
    finally:
        _ACTIVE_EXECUTION.reset(token)
