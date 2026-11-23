# SPDX-License-Identifier: Apache-2.0
"""Decode-first host admission for independent P/D workers.

This module deliberately contains no CUDA calls.  A gate is held only while
Python is entering ``execute_model`` and submitting work to CUDA.  Decode and
Prefill have independent active lanes: a long Prefill host submission must not
prevent a later Decode submission from entering its phase-local model-runner
context.  Decode tickets that were already accepted still enter before a new
Prefill ticket.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Iterator


@dataclass(frozen=True)
class PDHostEnqueueTicket:
    phase: str | None
    ordered: bool
    decode_registered: bool = False


class DecodeFirstHostEnqueueGate:
    """Order phase-local host entry without serializing Decode behind Prefill.

    Decode tickets are registered by the engine-facing submit thread before
    work is handed to a phase-local executor. A Prefill worker cannot enter
    while one of those Decode submissions has not yet entered. Once admitted,
    Decode and Prefill may submit concurrently, while submissions within one
    phase remain ordered. This relies on ``PDPhaseModelRunnerContext`` routing
    every mutable per-execution buffer away from the shared read-only model.
    """

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._active = {"decode": False, "prefill": False}
        self._pending_decode = 0

    def register(self, phase: str | None, *, ordered: bool) -> PDHostEnqueueTicket:
        normalized = phase if phase in {"decode", "prefill"} else None
        decode_registered = bool(ordered and normalized == "decode")
        if decode_registered:
            with self._condition:
                self._pending_decode += 1
        return PDHostEnqueueTicket(
            phase=normalized,
            ordered=bool(ordered and normalized is not None),
            decode_registered=decode_registered,
        )

    def cancel(self, ticket: PDHostEnqueueTicket) -> None:
        """Release a Decode ticket when executor submission itself fails."""
        if not ticket.decode_registered:
            return
        with self._condition:
            self._pending_decode -= 1
            if self._pending_decode < 0:
                self._pending_decode = 0
                raise RuntimeError("Decode host-enqueue ticket underflow")
            self._condition.notify_all()

    @contextmanager
    def enter(self, ticket: PDHostEnqueueTicket) -> Iterator[None]:
        if not ticket.ordered:
            yield
            return

        entered = False
        try:
            with self._condition:
                if ticket.phase == "prefill":
                    self._condition.wait_for(
                        lambda: (
                            not self._active["prefill"]
                            and self._pending_decode == 0
                        )
                    )
                else:
                    self._condition.wait_for(
                        lambda: not self._active["decode"]
                    )
                assert ticket.phase is not None
                self._active[ticket.phase] = True
                if ticket.decode_registered:
                    # Decode priority is an admission rule, not a whole-forward
                    # exclusion rule. Let Prefill enter as soon as every Decode
                    # ticket accepted ahead of it has reached its own lane.
                    self._pending_decode -= 1
                    if self._pending_decode < 0:
                        self._pending_decode = 0
                        raise RuntimeError(
                            "Decode host-enqueue ticket underflow"
                        )
                    self._condition.notify_all()
                entered = True
            yield
        finally:
            if not entered:
                return
            with self._condition:
                assert ticket.phase is not None
                self._active[ticket.phase] = False
                self._condition.notify_all()

    def snapshot(self) -> tuple[bool, int]:
        """Return state for tests or explicitly enabled executor debug."""
        with self._condition:
            return any(self._active.values()), self._pending_decode

    def phase_snapshot(self) -> tuple[bool, bool, int]:
        """Return Decode-active, Prefill-active, pending-Decode state."""
        with self._condition:
            return (
                self._active["decode"],
                self._active["prefill"],
                self._pending_decode,
            )
