# SPDX-License-Identifier: Apache-2.0
"""Experimental request-level scheduling for decode-ready work."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from enum import Enum
from typing import Protocol, TypeVar


class TBTDecodeSchedulingPolicy(str, Enum):
    FCFS = "fcfs"


class DecodeRequest(Protocol):
    request_id: str
    arrival_time: float
    prefill_completed_at: float | None
    prefill_completed_iteration_id: int
    tbt_slo_ms: float | None
    tbt_last_token_at: float | None
    tbt_last_observed_ms: float | None
    tbt_recent_gaps_ms: list[float]
    tbt_service_count: int


DecodeRequestT = TypeVar("DecodeRequestT", bound=DecodeRequest)
TBT_MEASUREMENT_UNIT = "ms_per_output_token"
TBT_MEASUREMENT_SOURCE = "scheduler_output_token_ids"


def per_token_tbt_samples(
    *,
    previous_at: float | None,
    served_at: float,
    num_tokens: int,
) -> list[float]:
    """Convert one streamed output chunk into exact per-token TBT samples.

    The first non-empty chunk is TTFT and therefore contributes no TBT
    sample. For every later chunk, its inter-arrival duration is divided by
    the number of emitted token IDs and repeated once per emitted token.
    """

    if not math.isfinite(served_at):
        raise ValueError("TBT service timestamp must be finite")
    if num_tokens <= 0 or previous_at is None:
        return []
    if not math.isfinite(previous_at):
        raise ValueError("previous TBT service timestamp must be finite")
    gap_ms = max(0.0, (float(served_at) - float(previous_at)) * 1000.0)
    per_token_gap_ms = gap_ms / float(num_tokens)
    return [per_token_gap_ms] * int(num_tokens)


class TBTDecodeScheduler:
    """Order decode-ready requests without owning vLLM queue state.

    FCFS is based on the timestamp at which the first sampled token confirms
    prefill completion. Requests completed in the same model-runner output use
    arrival time and request ID as deterministic tie-breakers. The helper only
    reorders decode candidate slots; it never moves a decode request across a
    prefill slot or limits the active decode set to one request.
    """

    def __init__(
        self,
        policy: str | TBTDecodeSchedulingPolicy = TBTDecodeSchedulingPolicy.FCFS,
    ) -> None:
        self.policy = TBTDecodeSchedulingPolicy(policy)
        self._candidate_order: tuple[str, ...] = ()
        self._candidate_ranks: dict[str, int] = {}
        self._scheduled_turns: dict[str, int] = {}
        self._last_scheduled_request_id = ""
        self._service_observation_epoch = 0

    @staticmethod
    def mark_prefill_completed(
        request: DecodeRequest,
        *,
        completed_at: float,
        iteration_id: int,
    ) -> bool:
        """Record the first confirmed prefill completion exactly once."""

        if request.prefill_completed_at is not None:
            return False
        if not math.isfinite(completed_at):
            raise ValueError("prefill completion timestamp must be finite")
        request.prefill_completed_at = float(completed_at)
        request.prefill_completed_iteration_id = int(iteration_id)
        return True

    def mark_tokens_served(
        self,
        request: DecodeRequest,
        *,
        served_at: float,
        num_tokens: int,
    ) -> None:
        if not math.isfinite(served_at):
            raise ValueError("TBT service timestamp must be finite")
        if num_tokens <= 0:
            return
        previous = request.tbt_last_token_at
        samples = per_token_tbt_samples(
            previous_at=previous,
            served_at=served_at,
            num_tokens=num_tokens,
        )
        if samples:
            request.tbt_last_observed_ms = samples[-1]
            request.tbt_recent_gaps_ms.extend(samples)
            # This is observability state, not an unbounded request trace.
            # Keep enough history for conservative feedback windows while
            # bounding per-request scheduler memory.
            if len(request.tbt_recent_gaps_ms) > 512:
                del request.tbt_recent_gaps_ms[:-512]
        request.tbt_last_token_at = float(served_at)
        request.tbt_service_count = int(
            getattr(request, "tbt_service_count", 0)
        ) + int(num_tokens)
        self._service_observation_epoch += int(num_tokens)

    @staticmethod
    def _nearest_rank_percentile(
        values: list[float], percentile: float
    ) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
        return float(ordered[min(rank - 1, len(ordered) - 1)])

    def feedback_snapshot(
        self,
        requests: Iterable[DecodeRequest],
        *,
        default_tbt_slo_ms: float,
        window_tokens: int,
    ) -> dict[str, int | float]:
        """Return the worst normalized recent request-level TBT window."""

        window = max(1, int(window_tokens))
        worst_ratio = 0.0
        worst_p95_ms = 0.0
        worst_slo_ms = 0.0
        worst_sample_count = 0
        observed_request_count = 0
        for request in requests:
            slo_ms = (
                float(request.tbt_slo_ms)
                if request.tbt_slo_ms is not None
                else max(0.0, float(default_tbt_slo_ms))
            )
            samples = request.tbt_recent_gaps_ms[-window:]
            if slo_ms <= 0.0 or not samples:
                continue
            observed_request_count += 1
            p95_ms = self._nearest_rank_percentile(samples, 95.0)
            ratio = p95_ms / slo_ms
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_p95_ms = p95_ms
                worst_slo_ms = slo_ms
                worst_sample_count = len(samples)
        return {
            "decode_tbt_observed_p95_ms": worst_p95_ms,
            "decode_tbt_observed_slo_ms": worst_slo_ms,
            "decode_tbt_observed_ratio": worst_ratio,
            "decode_tbt_observation_count": worst_sample_count,
            "decode_tbt_observed_request_count": observed_request_count,
            "decode_tbt_observation_epoch": self._service_observation_epoch,
        }

    def reset_feedback_window(
        self, requests: Iterable[DecodeRequest]
    ) -> None:
        """Prevent samples from an old SM partition contaminating a new one."""

        for request in requests:
            request.tbt_recent_gaps_ms.clear()
            request.tbt_last_observed_ms = None

    @staticmethod
    def _fcfs_key(request: DecodeRequest) -> tuple[float, float, str]:
        completed_at = request.prefill_completed_at
        return (
            completed_at if completed_at is not None else math.inf,
            request.arrival_time,
            request.request_id,
        )

    def begin_iteration(
        self,
        requests: Iterable[DecodeRequest],
    ) -> tuple[str, ...]:
        unique_requests: dict[str, DecodeRequest] = {}
        for request in requests:
            if request.prefill_completed_at is None:
                continue
            unique_requests.setdefault(request.request_id, request)
        ordered = sorted(unique_requests.values(), key=self._fcfs_key)
        self._candidate_order = tuple(
            request.request_id for request in ordered
        )
        self._candidate_ranks = {
            request_id: rank
            for rank, request_id in enumerate(self._candidate_order)
        }
        return self._candidate_order

    @property
    def candidate_order(self) -> tuple[str, ...]:
        return self._candidate_order

    def prioritize_candidate_slots(
        self,
        requests: Iterable[DecodeRequestT],
    ) -> list[DecodeRequestT]:
        """Sort decode candidates while preserving every non-decode slot."""

        ordered = list(requests)
        candidate_slots = [
            index
            for index, request in enumerate(ordered)
            if request.request_id in self._candidate_ranks
        ]
        if len(candidate_slots) <= 1:
            return ordered
        candidates = sorted(
            (ordered[index] for index in candidate_slots),
            key=lambda request: self._candidate_ranks[request.request_id],
        )
        for index, request in zip(candidate_slots, candidates):
            ordered[index] = request
        return ordered

    def observe_scheduled_tokens(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            if request_id not in self._candidate_ranks:
                continue
            self._scheduled_turns[request_id] = (
                self._scheduled_turns.get(request_id, 0) + 1
            )
            self._last_scheduled_request_id = request_id

    def forget_request(self, request_id: str) -> None:
        self._candidate_ranks.pop(request_id, None)
        self._scheduled_turns.pop(request_id, None)

    def request_metrics(
        self,
        request: DecodeRequest,
        *,
        now: float | None = None,
    ) -> dict[str, int | float]:
        completed_at = request.prefill_completed_at
        last_token_at = getattr(request, "tbt_last_token_at", None)
        now_monotonic = time.monotonic() if now is None else float(now)
        current_elapsed_ms = (
            max(0.0, (now_monotonic - last_token_at) * 1000.0)
            if last_token_at is not None
            else 0.0
        )
        tbt_slo_ms = getattr(request, "tbt_slo_ms", None)
        return {
            "tbt_decode_policy_rank": self._candidate_ranks.get(
                request.request_id, -1
            ),
            "tbt_decode_scheduled_turns": self._scheduled_turns.get(
                request.request_id, 0
            ),
            "prefill_completed_at": (
                completed_at if completed_at is not None else -1.0
            ),
            "prefill_completed_iteration_id": (
                request.prefill_completed_iteration_id
            ),
            "tbt_slo_ms": tbt_slo_ms if tbt_slo_ms is not None else -1.0,
            "tbt_last_token_at": (
                last_token_at if last_token_at is not None else -1.0
            ),
            "tbt_current_elapsed_ms": current_elapsed_ms,
            "tbt_service_count": int(
                getattr(request, "tbt_service_count", 0)
            ),
            "tbt_last_observed_ms": (
                request.tbt_last_observed_ms
                if request.tbt_last_observed_ms is not None
                else -1.0
            ),
            "tbt_recent_gap_count": len(request.tbt_recent_gaps_ms),
        }

    def iteration_metrics(self) -> dict[str, int | str]:
        return {
            "tbt_decode_scheduling_policy": self.policy.value,
            "tbt_decode_candidate_count": len(self._candidate_order),
            "tbt_decode_candidate_ids": ",".join(self._candidate_order),
            "tbt_decode_last_scheduled_request_id": (
                self._last_scheduled_request_id
            ),
        }
