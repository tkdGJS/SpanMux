# SPDX-License-Identifier: Apache-2.0
"""Priority gate for the shared tiered HBM staging data plane."""

from __future__ import annotations

import time
from contextlib import contextmanager
from threading import Condition, Lock
from typing import ContextManager, Iterator


class _TieredTransferPriorityGate:
    """Serialize staging users while giving native reload strict priority.

    An in-flight context-ring operation is allowed to finish. Once a native
    reload starts waiting, no new context-ring operation may enter until all
    waiting native reloads have completed.
    """

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._active_operation: str | None = None
        self._waiting_native = 0
        self._native_acquires = 0
        self._context_acquires = 0
        self._native_wait_count = 0
        self._context_wait_count = 0
        self._native_wait_us_total = 0
        self._context_wait_us_total = 0

    @contextmanager
    def acquire(self, operation: str) -> Iterator[None]:
        if operation not in {"native_reload", "context_ring"}:
            raise ValueError(f"unsupported tiered transfer operation: {operation}")
        native = operation == "native_reload"
        wait_started = time.perf_counter()
        waited = False
        with self._condition:
            if native:
                self._waiting_native += 1
            try:
                while self._active_operation is not None or (
                    not native and self._waiting_native > 0
                ):
                    waited = True
                    self._condition.wait()
                self._active_operation = operation
            finally:
                if native:
                    self._waiting_native -= 1
            wait_us = max(0, int((time.perf_counter() - wait_started) * 1e6))
            if native:
                self._native_acquires += 1
                if waited:
                    self._native_wait_count += 1
                    self._native_wait_us_total += wait_us
            else:
                self._context_acquires += 1
                if waited:
                    self._context_wait_count += 1
                    self._context_wait_us_total += wait_us
        try:
            yield
        finally:
            with self._condition:
                self._active_operation = None
                self._condition.notify_all()

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "tiered_transfer_priority_active_operation": (
                    1
                    if self._active_operation == "native_reload"
                    else 2
                    if self._active_operation == "context_ring"
                    else 0
                ),
                "tiered_transfer_priority_waiting_native": self._waiting_native,
                "tiered_transfer_priority_native_acquires": self._native_acquires,
                "tiered_transfer_priority_context_acquires": self._context_acquires,
                "tiered_transfer_priority_native_wait_count": self._native_wait_count,
                "tiered_transfer_priority_context_wait_count": self._context_wait_count,
                "tiered_transfer_priority_native_wait_us_total": (
                    self._native_wait_us_total
                ),
                "tiered_transfer_priority_context_wait_us_total": (
                    self._context_wait_us_total
                ),
            }


_GATE = _TieredTransferPriorityGate()


def acquire_native_reload_priority() -> ContextManager[None]:
    return _GATE.acquire("native_reload")


def acquire_context_ring_priority() -> ContextManager[None]:
    return _GATE.acquire("context_ring")


def get_tiered_transfer_priority_stats() -> dict[str, int]:
    return _GATE.stats()
