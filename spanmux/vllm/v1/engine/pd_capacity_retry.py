# SPDX-License-Identifier: Apache-2.0
"""Capacity-change gate for experimental independent P/D phase retries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PDPhaseCapacityBlock:
    """One phase admission failure and its explicit wake contract."""

    minimum_free_blocks: int
    observed_free_blocks: int
    reason: str
    scheduler_iteration_id: int
    cause: str = ""
    wake_on_output_progress: bool = False
    state_change_generation: int = 0


class PDPhaseCapacityRetryGate:
    """Suppress unchanged phase retries without hiding new phase work.

    The scheduler records a block only after it has completed one authoritative
    admission attempt.  The engine may then skip that phase until free HBM
    reaches the recorded minimum. Selected policy/no-progress observations may
    also wake after a typed scheduler-state event. A new request or explicit
    clear invalidates either observation because it can introduce work with a
    different capacity claim.
    """

    def __init__(self) -> None:
        self._blocked: dict[str, PDPhaseCapacityBlock] = {}
        self._state_change_generation = 0
        self._last_state_change_reason = ""
        self.block_count = 0
        self.skip_count = 0
        self.wake_count = 0
        self.output_progress_wake_count = 0
        self.clear_count = 0
        self.wait_count = 0

    def block(
        self,
        phase: str,
        *,
        minimum_free_blocks: int,
        observed_free_blocks: int,
        reason: str,
        scheduler_iteration_id: int,
        cause: str = "",
        wake_on_output_progress: bool = False,
    ) -> bool:
        minimum = max(0, int(minimum_free_blocks))
        observed = max(0, int(observed_free_blocks))
        if minimum <= observed:
            self.clear(phase)
            return False
        state = PDPhaseCapacityBlock(
            minimum_free_blocks=minimum,
            observed_free_blocks=observed,
            reason=str(reason),
            scheduler_iteration_id=int(scheduler_iteration_id),
            cause=str(cause),
            wake_on_output_progress=bool(wake_on_output_progress),
            state_change_generation=self._state_change_generation,
        )
        if self._blocked.get(phase) != state:
            self._blocked[phase] = state
            self.block_count += 1
        return True

    def clear(self, phase: str | None = None) -> None:
        if phase is None:
            if self._blocked:
                self._blocked.clear()
                self.clear_count += 1
            return
        if self._blocked.pop(phase, None) is not None:
            self.clear_count += 1

    def can_retry(self, phase: str, current_free_blocks: int) -> bool:
        state = self._blocked.get(phase)
        if state is None:
            return True
        current_free = int(current_free_blocks)
        if current_free < 0:
            # Unknown capacity must fail open so non-vLLM scheduler mocks and
            # alternate SchedulerInterface implementations keep their legacy
            # behavior.
            return True
        if current_free >= state.minimum_free_blocks:
            self._blocked.pop(phase, None)
            self.wake_count += 1
            return True
        self.skip_count += 1
        return False

    def notify_state_change(self, reason: str) -> int:
        """Wake policy observations after a relevant scheduler-state change.

        A normal non-terminal Decode token does not make a failed Prefill
        admission more feasible and must not call this method.  Callers use
        explicit lifecycle events such as request completion, Prefill future
        recovery, request-set changes, or a committed SM-profile change.
        Exact capacity blocks retain their recorded free-block threshold.
        """

        self._state_change_generation += 1
        self._last_state_change_reason = str(reason)

        phases = tuple(
            phase
            for phase, state in self._blocked.items()
            if state.wake_on_output_progress
        )
        for phase in phases:
            self._blocked.pop(phase, None)
        if phases:
            count = len(phases)
            self.wake_count += count
            self.output_progress_wake_count += count
        return len(phases)

    def notify_output_progress(
        self,
        *,
        completed_phase: str,
        request_finished: bool = False,
        capacity_increased: bool = False,
    ) -> int:
        """Wake only when recovered output can change Prefill feasibility.

        Prefill recovery releases future ownership and changes request-policy
        progress. A finished request or increased free capacity can also make
        a blocked admission feasible. Ordinary Decode token recovery has none
        of those effects (and normally grows KV), so it remains blocked.
        """

        phase = str(completed_phase).strip().lower()
        if not (
            phase == "prefill"
            or bool(request_finished)
            or bool(capacity_increased)
        ):
            return 0
        reasons = []
        if phase == "prefill":
            reasons.append("prefill_recovered")
        if request_finished:
            reasons.append("request_finished")
        if capacity_increased:
            reasons.append("capacity_increased")
        return self.notify_state_change("+".join(reasons))

    def is_blocked(self, phase: str, current_free_blocks: int) -> bool:
        state = self._blocked.get(phase)
        if state is None:
            return False
        current_free = int(current_free_blocks)
        return 0 <= current_free < state.minimum_free_blocks

    def blocking_cause(self, phase: str, current_free_blocks: int) -> str:
        """Return the typed cause only while ``phase`` is actually blocked."""
        state = self._blocked.get(phase)
        if state is None:
            return ""
        current_free = int(current_free_blocks)
        if current_free < 0 or current_free >= state.minimum_free_blocks:
            return ""
        return state.cause

    def note_wait(self) -> None:
        self.wait_count += 1

    def snapshot(self, current_free_blocks: int = -1) -> dict[str, int | str]:
        phases = tuple(sorted(self._blocked))
        minimum = max(
            (
                state.minimum_free_blocks
                for state in self._blocked.values()
            ),
            default=0,
        )
        observed = min(
            (state.observed_free_blocks for state in self._blocked.values()),
            default=0,
        )
        reasons = ",".join(
            sorted({state.reason for state in self._blocked.values() if state.reason})
        )
        causes = ",".join(
            sorted({state.cause for state in self._blocked.values() if state.cause})
        )
        return {
            "pd_capacity_retry_gate_active": int(bool(phases)),
            "pd_capacity_retry_gate_phases": ",".join(phases),
            "pd_capacity_retry_gate_min_free_blocks": minimum,
            "pd_capacity_retry_gate_observed_free_blocks": observed,
            "pd_capacity_retry_gate_current_free_blocks": int(
                current_free_blocks
            ),
            "pd_capacity_retry_gate_reasons": reasons,
            "pd_capacity_retry_gate_causes": causes,
            "pd_capacity_retry_gate_block_count": self.block_count,
            "pd_capacity_retry_gate_skip_count": self.skip_count,
            "pd_capacity_retry_gate_wake_count": self.wake_count,
            "pd_capacity_retry_gate_output_progress_wake_count": (
                self.output_progress_wake_count
            ),
            "pd_capacity_retry_gate_state_change_generation": (
                self._state_change_generation
            ),
            "pd_capacity_retry_gate_last_state_change_reason": (
                self._last_state_change_reason
            ),
            "pd_capacity_retry_gate_clear_count": self.clear_count,
            "pd_capacity_retry_gate_wait_count": self.wait_count,
        }
