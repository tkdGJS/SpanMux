# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-independent control owner for required (not opportunistic) D2H.

This worker must be separate from the provider's dispatch/completion workers:
queued dispatch jobs may already own the budget for which this worker waits.
No CUDA tensors, scheduler slots, or allocation authority live in this module.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Condition
from typing import Callable


class RequiredD2HProgress:
    def __init__(self) -> None:
        self._condition = Condition()
        self._executor: ThreadPoolExecutor | None = None
        self._error: BaseException | None = None
        self._metrics = dict(
            plans=0, pages=0, pending_plans=0, pending_pages=0,
            completed_plans=0, failed_plans=0, batches=0, budget_waits=0,
        )

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                f"dram_provider_required_d2h_{key}": value
                for key, value in self._metrics.items()
            }

    def raise_if_failed(self) -> None:
        # Single-assignment error latch: keep the healthy scheduler hot path
        # lock-free. The exception remains visible after page fences retire.
        error = self._error
        if error is not None:
            raise RuntimeError("required D2H snapshot failed") from error

    def budget_released(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def reserve_batch(self, reserve: Callable[[], bool]) -> None:
        # Completion updates the provider budget before taking this condition.
        # The condition prevents a lost wake between testing and waiting.
        with self._condition:
            while not reserve():
                self._metrics["budget_waits"] += 1
                self._condition.wait()
            self._metrics["batches"] += 1

    def submit(
        self, run: Callable[[], None], completion: Future[None],
        page_count: int, wake_engine: Callable[[], None],
    ) -> None:
        """Take a finite, already fenced/leased plan; never run it inline."""
        with self._condition:
            self._metrics["plans"] += 1
            self._metrics["pages"] += page_count
            self._metrics["pending_plans"] += 1
            self._metrics["pending_pages"] += page_count
        try:
            with self._condition:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="tiered-required-d2h",
                    )
                executor = self._executor
            executor.submit(self._run, run, completion, page_count, wake_engine)
        except BaseException as exc:
            self._finish(completion, page_count, exc, wake_engine)
            raise

    def _run(self, run, completion, page_count, wake_engine) -> None:
        error = None
        try:
            self.raise_if_failed()
            run()
        except BaseException as exc:
            error = exc
        self._finish(completion, page_count, error, wake_engine)

    def _finish(self, completion, page_count, error, wake_engine) -> None:
        with self._condition:
            if error is not None and self._error is None:
                self._error = error
            self._metrics["pending_plans"] -= 1
            self._metrics["pending_pages"] -= page_count
            self._metrics[
                "completed_plans" if error is None else "failed_plans"
            ] += 1
        try:
            if error is None:
                completion.set_result(None)
            else:
                completion.set_exception(error)
        finally:
            wake_engine()


required_d2h_progress = RequiredD2HProgress()


def raise_if_required_d2h_failed() -> None:
    required_d2h_progress.raise_if_failed()
