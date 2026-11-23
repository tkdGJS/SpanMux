# SPDX-License-Identifier: Apache-2.0
"""Thread-safe control plane for experimental layer-boundary SM handoff.

The EngineCore and UniProc GPU worker live in one process but use different
host threads.  This module carries a small, process-local state machine between
those threads.  It deliberately contains no CUDA calls; GreenContext routing
applies an accepted request at a real layer completion boundary.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from vllm.v1.engine.pd_engine_progress import notify_pd_engine_progress


_ENABLE_ENV = "VLLM_PD_LAYERWISE_PROFILE_HANDOFF"
_STRICT_ENV = "VLLM_PD_LAYERWISE_PROFILE_HANDOFF_STRICT"


def layerwise_profile_handoff_requested() -> bool:
    return os.environ.get(_ENABLE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def layerwise_profile_handoff_strict() -> bool:
    return os.environ.get(_STRICT_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def layerwise_profile_handoff_decision(
    *,
    armed: bool,
    prefill_inflight: bool,
    decode_inflight: bool,
    source_prefill_sms: int,
    target_prefill_sms: int,
) -> tuple[bool, str]:
    """Decide whether an in-flight Prefill may change SM profile.

    A submitted Decode future must finish on its immutable source profile.
    Scheduler-ready Decode work is not a blocker when the target owns Decode
    SMs: admission remains fenced until the Prefill worker acknowledges a real
    layer boundary and EngineCore commits that target epoch.
    """

    if not armed:
        return False, "not_armed"
    if not prefill_inflight:
        return False, "no_prefill_inflight"
    if decode_inflight:
        return False, "decode_inflight"
    if int(source_prefill_sms) <= 0:
        return False, "source_has_no_prefill_sms"
    if int(target_prefill_sms) <= 0:
        return False, "target_has_no_prefill_sms"
    return True, "prefill_layer_boundary"


def layerwise_source_decode_admission_decision(
    *,
    transition_requested: bool,
    boundary_completed: bool,
    source_decode_sms: int,
) -> tuple[bool, str]:
    """Decide whether source-epoch Decode may run during a layer handoff.

    Recording a CUDA event does not mean that the source Prefill layer has
    finished.  Until that event completes, Decode must continue on the source
    profile when it owns Decode SMs.  Once the boundary completes, admission
    closes so the depth-one source Decode queue can drain before the profile
    epoch is committed.
    """

    if not transition_requested:
        return False, "not_requested"
    if boundary_completed:
        return False, "boundary_completed_drain_source_decode"
    if int(source_decode_sms) <= 0:
        return False, "source_has_no_decode_sms"
    return True, "source_decode_until_boundary"


def layerwise_prefill_only_handoff_decision(
    *,
    armed: bool,
    prefill_inflight: bool,
    decode_inflight: bool,
    decode_ready: bool,
) -> tuple[bool, str]:
    """Compatibility wrapper for the original Decode-idle policy test."""

    if decode_ready:
        return False, "decode_backlog_ready"
    eligible, reason = layerwise_profile_handoff_decision(
        armed=armed,
        prefill_inflight=prefill_inflight,
        decode_inflight=decode_inflight,
        source_prefill_sms=1,
        target_prefill_sms=1,
    )
    return (eligible, "decode_idle" if eligible else reason)


@dataclass(frozen=True)
class PDLayerwiseHandoffTarget:
    generation: int
    profile_id: str
    profile_epoch: int


@dataclass
class _HandoffState:
    generation: int
    source_profile_id: str
    source_epoch: int
    armed_at: float
    execution_started: bool = False
    execution_done: bool = False
    requested: bool = False
    requested_at: float = 0.0
    target_profile_id: str = ""
    target_epoch: int = -1
    boundary_recorded: bool = False
    boundary_recorded_at: float = 0.0
    boundary_completed: bool = False
    boundary_completed_at: float = 0.0
    applied: bool = False
    applied_at: float = 0.0
    applied_layer: int = -1
    committed_handoff_count: int = 0
    last_committed_source_profile_id: str = ""
    last_committed_source_epoch: int = -1
    last_committed_target_profile_id: str = ""
    last_committed_target_epoch: int = -1
    last_committed_layer: int = -1
    boundary_count: int = 0
    boundary_sync_ms_total: float = 0.0
    boundary_sync_ms_max: float = 0.0
    wake_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )
    resume_event: threading.Event | None = field(default=None, repr=False)


class PDLayerwiseHandoffController:
    """Coordinate one phase-pure Prefill future with EngineCore.

    Prefill depth is required to be one when this controller is enabled, so a
    single process-local state is sufficient and avoids request/scheduler state
    mutation from the worker thread.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_generation = 0
        self._state: _HandoffState | None = None
        self._arm_count = 0
        self._request_count = 0
        self._apply_count = 0
        self._commit_count = 0
        self._completed_without_request_count = 0
        self._missed_after_execution_count = 0
        self._rejected_arm_count = 0
        self._rejected_request_count = 0
        self._boundary_count_total = 0
        self._boundary_sync_ms_total = 0.0
        self._boundary_sync_ms_max = 0.0
        self._boundary_record_count_total = 0
        self._boundary_complete_count_total = 0
        self._boundary_gpu_wait_ms_total = 0.0
        self._boundary_gpu_wait_ms_max = 0.0

    def reset(self) -> None:
        """Reset process-local state at engine startup or in unit tests."""

        with self._lock:
            if self._state is not None:
                self._state.wake_event.set()
                if self._state.resume_event is not None:
                    self._state.resume_event.set()
            self._next_generation = 0
            self._state = None
            self._arm_count = 0
            self._request_count = 0
            self._apply_count = 0
            self._commit_count = 0
            self._completed_without_request_count = 0
            self._missed_after_execution_count = 0
            self._rejected_arm_count = 0
            self._rejected_request_count = 0
            self._boundary_count_total = 0
            self._boundary_sync_ms_total = 0.0
            self._boundary_sync_ms_max = 0.0
            self._boundary_record_count_total = 0
            self._boundary_complete_count_total = 0
            self._boundary_gpu_wait_ms_total = 0.0
            self._boundary_gpu_wait_ms_max = 0.0

    def arm(self, *, source_profile_id: str, source_epoch: int) -> int:
        """Arm a not-yet-submitted phase-pure Prefill future."""

        now = time.monotonic()
        with self._lock:
            state = self._state
            if state is not None:
                if (
                    state.source_profile_id == source_profile_id
                    and state.source_epoch == int(source_epoch)
                    and not state.execution_done
                ):
                    return state.generation
                # One controller generation owns one Prefill execution.  Do
                # not let a later future replace live or pending ownership.
                if not state.execution_done or state.requested:
                    self._rejected_arm_count += 1
                    return 0
                # Do not strand a worker waiting on the previous generation's
                # bounded layer window when ownership is replaced.
                state.wake_event.set()
            self._next_generation += 1
            generation = self._next_generation
            self._state = _HandoffState(
                generation=generation,
                source_profile_id=source_profile_id,
                source_epoch=int(source_epoch),
                armed_at=now,
            )
            self._arm_count += 1
            return generation

    def begin_execution(
        self, *, source_profile_id: str, source_epoch: int
    ) -> int:
        with self._lock:
            state = self._state
            if (
                state is None
                or state.source_profile_id != source_profile_id
                or state.source_epoch != int(source_epoch)
                or state.execution_done
            ):
                return 0
            state.execution_started = True
            return state.generation

    def request(
        self,
        *,
        source_profile_id: str,
        source_epoch: int,
        target_profile_id: str,
        target_epoch: int,
    ) -> bool:
        """Request one profile change for the active Prefill continuation."""

        now = time.monotonic()
        with self._lock:
            state = self._state
            if (
                state is None
                or state.source_profile_id != source_profile_id
                or state.source_epoch != int(source_epoch)
                or state.execution_done
                or not state.execution_started
            ):
                self._rejected_request_count += 1
                if state is not None and state.execution_done:
                    self._missed_after_execution_count += 1
                return False
            if state.requested:
                return (
                    state.target_profile_id == target_profile_id
                    and state.target_epoch == int(target_epoch)
                )
            state.requested = True
            state.requested_at = now
            state.target_profile_id = target_profile_id
            state.target_epoch = int(target_epoch)
            state.boundary_recorded = False
            state.boundary_recorded_at = 0.0
            state.boundary_completed = False
            state.boundary_completed_at = 0.0
            state.applied = False
            state.applied_at = 0.0
            state.applied_layer = -1
            state.resume_event = threading.Event()
            self._request_count += 1
            # Wake a worker that is deliberately holding the host submission
            # window at a CUDA-event boundary.  It can then attach the target
            # stream to the latest recorded boundary without waiting for an
            # entire source stream synchronization.
            state.wake_event.set()
            return True

    def should_bound_submission(self, generation: int) -> bool:
        """Return whether host layer submission must remain bounded.

        Unlike :meth:`should_gate`, this is true as soon as execution begins.
        Keeping at most a small number of source-profile layers queued is what
        leaves a real boundary available for a later EngineCore request.
        """

        with self._lock:
            state = self._state
            return bool(
                state is not None
                and state.generation == int(generation)
                and state.execution_started
                and not state.execution_done
            )

    def wake_event_for(self, generation: int) -> threading.Event | None:
        """Return the generation-owned request/cancellation wake event."""

        with self._lock:
            state = self._state
            if (
                state is None
                or state.generation != int(generation)
                or not state.execution_started
                or state.execution_done
            ):
                return None
            return state.wake_event

    def should_gate(self, generation: int) -> bool:
        """Return whether this execution has a handoff to apply.

        Arming only establishes future ownership.  It must not turn every
        transformer-layer boundary into a CUDA synchronization point while
        Decode is still active.  Once EngineCore publishes an actual request,
        the next boundary synchronizes the source stream and applies it.
        """

        with self._lock:
            state = self._state
            return bool(
                state is not None
                and state.generation == int(generation)
                and state.execution_started
                and state.requested
                and not state.execution_done
                and not state.applied
            )

    def target_for(self, generation: int) -> PDLayerwiseHandoffTarget | None:
        with self._lock:
            state = self._state
            if (
                state is None
                or state.generation != int(generation)
                or not state.requested
                or state.applied
                or state.execution_done
            ):
                return None
            return PDLayerwiseHandoffTarget(
                generation=state.generation,
                profile_id=state.target_profile_id,
                profile_epoch=state.target_epoch,
            )

    def transition_resume_event_for(
        self, generation: int
    ) -> threading.Event | None:
        """Return the event released by the Engine's completed commit."""

        with self._lock:
            state = self._state
            if (
                state is None
                or state.generation != int(generation)
                or not state.requested
                or state.execution_done
            ):
                return None
            return state.resume_event

    def source_decode_admission(
        self,
        *,
        source_decode_sms: int,
    ) -> tuple[bool, str]:
        """Read the request/boundary pair atomically for the hot path."""

        with self._lock:
            state = self._state
            return layerwise_source_decode_admission_decision(
                transition_requested=bool(
                    state is not None
                    and state.requested
                    and not state.execution_done
                ),
                boundary_completed=bool(
                    state is not None and state.boundary_completed
                ),
                source_decode_sms=source_decode_sms,
            )

    def mark_boundary_recorded(
        self, generation: int, *, layer_index: int
    ) -> bool:
        """Publish that the source-boundary CUDA event was enqueued.

        This is deliberately not an execution ACK.  The Engine must continue
        source-profile Decode service until :meth:`mark_boundary_completed`
        proves that the source layer actually finished on the GPU.
        """

        now = time.monotonic()
        with self._lock:
            state = self._state
            if (
                state is None
                or state.generation != int(generation)
                or not state.requested
                or state.boundary_recorded
                or state.execution_done
            ):
                return False
            state.boundary_recorded = True
            state.boundary_recorded_at = now
            state.applied_layer = int(layer_index)
            self._boundary_record_count_total += 1
            return True

    def note_boundary(self, generation: int, *, sync_ms: float) -> None:
        with self._lock:
            state = self._state
            if (
                state is None
                or state.generation != int(generation)
                or not state.execution_started
                or not state.requested
                or state.execution_done
                or state.applied
            ):
                return
            state.boundary_count += 1
            bounded_sync_ms = max(0.0, float(sync_ms))
            state.boundary_sync_ms_total += bounded_sync_ms
            state.boundary_sync_ms_max = max(
                state.boundary_sync_ms_max, bounded_sync_ms
            )
            self._boundary_count_total += 1
            self._boundary_sync_ms_total += bounded_sync_ms
            self._boundary_sync_ms_max = max(
                self._boundary_sync_ms_max,
                bounded_sync_ms,
            )

    def mark_boundary_completed(
        self,
        generation: int,
        *,
        layer_index: int,
        gpu_wait_ms: float,
    ) -> bool:
        """Publish the physical source-layer completion ACK.

        The caller has synchronized only the generation-owned CUDA event, not
        the complete GreenContext handle.  This is the earliest point at which
        EngineCore may stop source Decode admission and drain its final Future.
        """

        now = time.monotonic()
        with self._lock:
            state = self._state
            if (
                state is None
                or state.generation != int(generation)
                or not state.requested
                or not state.boundary_recorded
                or state.boundary_completed
                or state.applied
                or state.execution_done
                or state.applied_layer != int(layer_index)
            ):
                return False
            state.boundary_completed = True
            state.boundary_completed_at = now
            state.applied = True
            state.applied_at = now
            state.applied_layer = int(layer_index)
            bounded_wait_ms = max(0.0, float(gpu_wait_ms))
            self._boundary_complete_count_total += 1
            self._boundary_gpu_wait_ms_total += bounded_wait_ms
            self._boundary_gpu_wait_ms_max = max(
                self._boundary_gpu_wait_ms_max,
                bounded_wait_ms,
            )
            self._apply_count += 1
        # EngineCore can be sleeping indefinitely on the still-running Prefill
        # Future. Publish this worker boundary ACK through the same event-driven
        # progress channel as Future and tiered-copy completion.
        notify_pd_engine_progress()
        return True

    def completed_transition_matches(
        self,
        *,
        source_profile_id: str,
        source_epoch: int,
        target_profile_id: str,
        target_epoch: int,
    ) -> bool:
        """Return whether the exact transition reached GPU completion."""

        with self._lock:
            state = self._state
            return bool(
                state is not None
                and state.source_profile_id == source_profile_id
                and state.source_epoch == int(source_epoch)
                and state.requested
                and state.boundary_recorded
                and state.boundary_completed
                and state.applied
                and state.target_profile_id == target_profile_id
                and state.target_epoch == int(target_epoch)
            )

    def complete_execution(self, generation: int) -> None:
        with self._lock:
            state = self._state
            if state is None or state.generation != int(generation):
                return
            state.execution_done = True
            state.wake_event.set()
            if state.resume_event is not None:
                state.resume_event.set()
            if not state.requested:
                self._completed_without_request_count += 1
                self._state = None

    def committed_transition_matches(
        self,
        *,
        source_profile_id: str,
        source_epoch: int,
        target_profile_id: str,
        target_epoch: int,
    ) -> bool:
        """Return whether EngineCore committed this exact transition."""

        with self._lock:
            state = self._state
            return bool(
                state is not None
                and state.last_committed_source_profile_id == source_profile_id
                and state.last_committed_source_epoch == int(source_epoch)
                and state.last_committed_target_profile_id == target_profile_id
                and state.last_committed_target_epoch == int(target_epoch)
            )

    def acknowledge_completed_transition(
        self,
        *,
        source_profile_id: str,
        source_epoch: int,
        target_profile_id: str,
        target_epoch: int,
    ) -> bool:
        """Advance controller ownership after EngineCore commits an ACK.

        The Prefill execution remains armed, so a later Decode arrival or
        completion can request another profile change in the same future.
        """

        with self._lock:
            state = self._state
            if not self.completed_transition_matches(
                source_profile_id=source_profile_id,
                source_epoch=source_epoch,
                target_profile_id=target_profile_id,
                target_epoch=target_epoch,
            ):
                return False
            assert state is not None
            resume_event = state.resume_event
            state.committed_handoff_count += 1
            state.last_committed_source_profile_id = source_profile_id
            state.last_committed_source_epoch = int(source_epoch)
            state.last_committed_target_profile_id = target_profile_id
            state.last_committed_target_epoch = int(target_epoch)
            state.last_committed_layer = state.applied_layer
            state.source_profile_id = target_profile_id
            state.source_epoch = int(target_epoch)
            state.requested = False
            state.requested_at = 0.0
            state.target_profile_id = ""
            state.target_epoch = -1
            state.boundary_recorded = False
            state.boundary_recorded_at = 0.0
            state.boundary_completed = False
            state.boundary_completed_at = 0.0
            state.applied = False
            state.applied_at = 0.0
            state.applied_layer = -1
            state.resume_event = None
            self._commit_count += 1
            if resume_event is not None:
                resume_event.set()
            if state.execution_done:
                state.wake_event.set()
                if state.resume_event is not None:
                    state.resume_event.set()
                self._state = None
            return True

    def finish_engine_transition(
        self, *, source_profile_id: str, source_epoch: int
    ) -> None:
        with self._lock:
            state = self._state
            if (
                state is not None
                and state.source_profile_id == source_profile_id
                and state.source_epoch == int(source_epoch)
            ):
                state.wake_event.set()
                if state.resume_event is not None:
                    state.resume_event.set()
                self._state = None

    def snapshot(self) -> dict[str, int | float | str]:
        with self._lock:
            state = self._state
            result: dict[str, int | float | str] = {
                "pd_layerwise_handoff_armed": float(state is not None),
                "pd_layerwise_handoff_arm_count": float(self._arm_count),
                "pd_layerwise_handoff_request_count": float(
                    self._request_count
                ),
                "pd_layerwise_handoff_apply_count": float(self._apply_count),
                "pd_layerwise_handoff_commit_count": float(self._commit_count),
                "pd_layerwise_handoff_completed_without_request_count": float(
                    self._completed_without_request_count
                ),
                "pd_layerwise_handoff_missed_after_execution_count": float(
                    self._missed_after_execution_count
                ),
                "pd_layerwise_handoff_rejected_arm_count": float(
                    self._rejected_arm_count
                ),
                "pd_layerwise_handoff_rejected_request_count": float(
                    self._rejected_request_count
                ),
                "pd_layerwise_handoff_boundary_count_total": float(
                    self._boundary_count_total
                ),
                "pd_layerwise_handoff_boundary_sync_ms_total_all": (
                    self._boundary_sync_ms_total
                ),
                "pd_layerwise_handoff_boundary_sync_ms_max_all": (
                    self._boundary_sync_ms_max
                ),
                "pd_layerwise_handoff_boundary_record_count_total": float(
                    self._boundary_record_count_total
                ),
                "pd_layerwise_handoff_boundary_complete_count_total": float(
                    self._boundary_complete_count_total
                ),
                "pd_layerwise_handoff_boundary_gpu_wait_ms_total": (
                    self._boundary_gpu_wait_ms_total
                ),
                "pd_layerwise_handoff_boundary_gpu_wait_ms_max": (
                    self._boundary_gpu_wait_ms_max
                ),
            }
            if state is None:
                return result
            result.update(
                {
                    "pd_layerwise_handoff_generation": float(
                        state.generation
                    ),
                    "pd_layerwise_handoff_source_profile": (
                        state.source_profile_id
                    ),
                    "pd_layerwise_handoff_source_epoch": float(
                        state.source_epoch
                    ),
                    "pd_layerwise_handoff_execution_started": float(
                        state.execution_started
                    ),
                    "pd_layerwise_handoff_execution_done": float(
                        state.execution_done
                    ),
                    "pd_layerwise_handoff_requested": float(state.requested),
                    "pd_layerwise_handoff_target_profile": (
                        state.target_profile_id
                    ),
                    "pd_layerwise_handoff_target_epoch": float(
                        state.target_epoch
                    ),
                    "pd_layerwise_handoff_boundary_recorded": float(
                        state.boundary_recorded
                    ),
                    "pd_layerwise_handoff_boundary_completed": float(
                        state.boundary_completed
                    ),
                    "pd_layerwise_handoff_applied": float(state.applied),
                    "pd_layerwise_handoff_applied_layer": float(
                        state.applied_layer
                    ),
                    "pd_layerwise_handoff_committed_count": float(
                        state.committed_handoff_count
                    ),
                    "pd_layerwise_handoff_last_committed_source_profile": (
                        state.last_committed_source_profile_id
                    ),
                    "pd_layerwise_handoff_last_committed_source_epoch": float(
                        state.last_committed_source_epoch
                    ),
                    "pd_layerwise_handoff_last_committed_target_profile": (
                        state.last_committed_target_profile_id
                    ),
                    "pd_layerwise_handoff_last_committed_target_epoch": float(
                        state.last_committed_target_epoch
                    ),
                    "pd_layerwise_handoff_last_committed_layer": float(
                        state.last_committed_layer
                    ),
                    "pd_layerwise_handoff_boundary_count": float(
                        state.boundary_count
                    ),
                    "pd_layerwise_handoff_boundary_sync_ms_total": (
                        state.boundary_sync_ms_total
                    ),
                    "pd_layerwise_handoff_boundary_sync_ms_max": (
                        state.boundary_sync_ms_max
                    ),
                    "pd_layerwise_handoff_request_to_apply_ms": (
                        max(
                            0.0,
                            (state.applied_at - state.requested_at) * 1000.0,
                        )
                        if state.applied and state.requested
                        else 0.0
                    ),
                }
            )
            return result


PD_LAYERWISE_HANDOFF_CONTROLLER = PDLayerwiseHandoffController()
