# SPDX-License-Identifier: Apache-2.0
"""Explicit ownership for output placeholders produced by model futures."""

from dataclasses import dataclass, field, replace
from itertools import count


_REQUEST_EPOCHS = count(1)


def _next_request_epoch() -> int:
    return next(_REQUEST_EPOCHS)


class OutputPlaceholderOwnershipError(RuntimeError):
    """A model result violated explicit output-placeholder ownership.

    This is a fatal execution-state error. It must not be converted into a
    request-local retry because a stale or missing result can no longer be
    distinguished safely after scheduler token progress has advanced.
    """


@dataclass(frozen=True, slots=True)
class OutputPlaceholderOwnership:
    """One Future's bounded claim over a request's output positions."""

    request_epoch: int
    generation: int
    phase: str
    placeholder_count: int
    cancelled: bool = False


@dataclass(slots=True)
class OutputPlaceholderLedger:
    """Request-local, Future-free output ownership state.

    Only numeric request-epoch, generation, and phase metadata are retained.
    In particular, the ledger never stores a Future, SchedulerOutput, tensor,
    or callback, so ownership metadata cannot extend those objects' lifetimes.
    """

    request_epoch: int = field(default_factory=_next_request_epoch)
    next_generation: int = 0
    active: OutputPlaceholderOwnership | None = None
    last_committed_generation: int = 0
    last_committed_phase: str = ""

    def reserve(
        self,
        *,
        phase: str,
        placeholder_count: int,
    ) -> OutputPlaceholderOwnership:
        normalized_phase = phase.strip().lower()
        if normalized_phase not in {"decode", "prefill"}:
            raise OutputPlaceholderOwnershipError(
                f"Invalid output-placeholder phase: {phase!r}."
            )
        if placeholder_count <= 0:
            raise OutputPlaceholderOwnershipError(
                "Output-placeholder reservations must contain at least one "
                f"position: count={placeholder_count}."
            )
        if self.active is not None:
            raise OutputPlaceholderOwnershipError(
                "A request cannot reserve output positions while an older "
                "Future still owns them: "
                f"active_request_epoch={self.active.request_epoch}, "
                f"active_generation={self.active.generation}, "
                f"active_phase={self.active.phase}, "
                f"active_count={self.active.placeholder_count}, "
                f"new_phase={normalized_phase}, "
                f"new_count={placeholder_count}."
            )

        self.next_generation += 1
        owner = OutputPlaceholderOwnership(
            request_epoch=self.request_epoch,
            generation=self.next_generation,
            phase=normalized_phase,
            placeholder_count=int(placeholder_count),
        )
        self.active = owner
        return owner

    def cancel_active(self) -> OutputPlaceholderOwnership | None:
        owner = self.active
        if owner is None or owner.cancelled:
            return owner
        owner = replace(owner, cancelled=True)
        self.active = owner
        return owner

    def resolve(
        self,
        *,
        request_epoch: int,
        generation: int,
        phase: str,
        generated_count: int,
        rejected_count: int = 0,
    ) -> bool:
        """Resolve one owner, returning false only for an explicit cancel."""

        owner = self.active
        normalized_phase = phase.strip().lower()
        if owner is None:
            raise OutputPlaceholderOwnershipError(
                "A model result has no active output-placeholder owner: "
                f"request_epoch={request_epoch}, generation={generation}, "
                f"phase={normalized_phase}."
            )
        if (
            owner.request_epoch != request_epoch
            or owner.generation != generation
            or owner.phase != normalized_phase
        ):
            raise OutputPlaceholderOwnershipError(
                "A stale, cross-request, or cross-phase model result "
                "attempted to resolve output placeholders: "
                f"active_request_epoch={owner.request_epoch}, "
                f"active_generation={owner.generation}, "
                f"active_phase={owner.phase}, "
                f"result_request_epoch={request_epoch}, "
                f"result_generation={generation}, "
                f"result_phase={normalized_phase}."
            )

        if owner.cancelled:
            self.active = None
            return False
        if generated_count <= 0:
            raise OutputPlaceholderOwnershipError(
                "The owning Future returned no sampled output: "
                f"generation={generation}, phase={normalized_phase}, "
                f"reserved_count={owner.placeholder_count}."
            )
        resolved_count = int(generated_count) + max(0, int(rejected_count))
        if resolved_count != owner.placeholder_count:
            raise OutputPlaceholderOwnershipError(
                "The owning Future resolved a different number of output "
                "positions than it reserved: "
                f"generation={generation}, phase={normalized_phase}, "
                f"reserved_count={owner.placeholder_count}, "
                f"generated_count={generated_count}, "
                f"rejected_count={rejected_count}."
            )

        self.active = None
        self.last_committed_generation = generation
        self.last_committed_phase = normalized_phase
        return True
