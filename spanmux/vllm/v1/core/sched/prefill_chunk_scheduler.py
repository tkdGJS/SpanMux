# SPDX-License-Identifier: Apache-2.0
"""Experimental request-level scheduling for chunked prefill work."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol


_PREFILL_PROFILE_MODES = frozenset({"native", "ring", "any"})


@dataclass(frozen=True)
class PrefillChunkServiceProfilePoint:
    """One conservative measured prefill-chunk service observation."""

    profile_id: str
    prefill_mode: str
    max_context_tokens: int
    max_chunk_tokens: int
    p95_prefill_chunk_ms: float

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        index: int,
    ) -> "PrefillChunkServiceProfilePoint":
        try:
            profile_id = str(raw.get("profile_id", "any")).strip()
            prefill_mode = str(raw.get("prefill_mode", "any")).strip().lower()
            max_context_tokens = int(raw["max_context_tokens"])
            max_chunk_tokens = int(raw["max_chunk_tokens"])
            p95_prefill_chunk_ms = float(raw["p95_prefill_chunk_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid prefill service profile point at "
                f"index {index}: {raw!r}"
            ) from exc

        if not profile_id:
            raise ValueError(
                f"prefill service profile point {index} has an empty profile_id"
            )
        if prefill_mode not in _PREFILL_PROFILE_MODES:
            raise ValueError(
                f"invalid prefill_mode={prefill_mode!r} at prefill service "
                f"profile point {index}; expected one of "
                f"{sorted(_PREFILL_PROFILE_MODES)}"
            )
        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens must be positive at prefill service "
                f"profile point {index}"
            )
        if max_chunk_tokens <= 0:
            raise ValueError(
                "max_chunk_tokens must be positive at prefill service "
                f"profile point {index}"
            )
        if (
            not math.isfinite(p95_prefill_chunk_ms)
            or p95_prefill_chunk_ms <= 0
        ):
            raise ValueError(
                "p95_prefill_chunk_ms must be a positive finite number at "
                f"prefill service profile point {index}"
            )
        return cls(
            profile_id=profile_id,
            prefill_mode=prefill_mode,
            max_context_tokens=max_context_tokens,
            max_chunk_tokens=max_chunk_tokens,
            p95_prefill_chunk_ms=p95_prefill_chunk_ms,
        )


@dataclass(frozen=True)
class PrefillChunkServicePrediction:
    """Conservative prediction across every execution variant in a profile."""

    requested_context_tokens: int
    requested_chunk_tokens: int
    bucket_context_tokens: int
    bucket_chunk_tokens: int
    p95_prefill_chunk_ms: float
    profile_id: str
    prefill_mode: str
    covered_variant_count: int


@dataclass(frozen=True)
class PrefillRemainingServicePrediction:
    """Sum of profiled chunk costs for one request's durable remaining work."""

    estimated_remaining_ms: float
    estimated_next_chunk_ms: float
    remaining_chunks: int
    covered_variant_count: int
    profile_covered: bool = True
    first_uncovered_context_tokens: int = 0
    first_uncovered_chunk_tokens: int = 0


class PrefillChunkServiceProfile:
    """Measured chunk-cost lookup that never extrapolates a missing bucket.

    A file may contain multiple ``(profile_id, prefill_mode)`` variants. For
    each requested chunk, the tightest covering bucket is selected within
    every variant and the maximum p95 is returned. This lets the request
    scheduler remain independent from the later P/D profile decision while
    staying conservative when the file covers every allowed execution mode.
    """

    def __init__(
        self,
        points: tuple[PrefillChunkServiceProfilePoint, ...],
        *,
        source: str = "",
    ) -> None:
        if not points:
            raise ValueError(
                "prefill chunk service profile must contain at least one point"
            )
        self.points = points
        self.source = source
        self.variant_keys = tuple(
            sorted({(point.profile_id, point.prefill_mode) for point in points})
        )

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        source: str = "",
    ) -> "PrefillChunkServiceProfile":
        schema_version = raw.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(
                "unsupported prefill service profile schema_version="
                f"{schema_version!r}; expected 1"
            )
        raw_points = raw.get("points")
        if not isinstance(raw_points, list):
            raise ValueError(
                "prefill chunk service profile 'points' must be a list"
            )
        points = tuple(
            PrefillChunkServiceProfilePoint.from_dict(point, index=index)
            for index, point in enumerate(raw_points)
            if isinstance(point, dict)
        )
        if len(points) != len(raw_points):
            raise ValueError(
                "every prefill chunk service profile point must be an object"
            )
        return cls(points, source=source)

    @classmethod
    def from_json_file(
        cls, path: str | Path
    ) -> "PrefillChunkServiceProfile":
        profile_path = Path(path).expanduser()
        try:
            raw = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"failed to load prefill service profile {profile_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(
                "prefill chunk service profile root must be an object"
            )
        return cls.from_dict(raw, source=str(profile_path))

    @lru_cache(maxsize=8192)
    def predict_chunk(
        self,
        *,
        context_tokens: int,
        chunk_tokens: int,
    ) -> PrefillChunkServicePrediction | None:
        requested_context_tokens = max(1, int(context_tokens))
        requested_chunk_tokens = max(1, int(chunk_tokens))
        selected_by_variant: list[PrefillChunkServiceProfilePoint] = []
        for profile_id, prefill_mode in self.variant_keys:
            covering = [
                point
                for point in self.points
                if point.profile_id == profile_id
                and point.prefill_mode == prefill_mode
                and point.max_context_tokens >= requested_context_tokens
                and point.max_chunk_tokens >= requested_chunk_tokens
            ]
            if not covering:
                return None
            best_bucket = min(
                (point.max_context_tokens, point.max_chunk_tokens)
                for point in covering
            )
            bucket_points = [
                point
                for point in covering
                if (point.max_context_tokens, point.max_chunk_tokens)
                == best_bucket
            ]
            selected_by_variant.append(
                max(
                    bucket_points,
                    key=lambda point: point.p95_prefill_chunk_ms,
                )
            )

        selected = max(
            selected_by_variant,
            key=lambda point: (
                point.p95_prefill_chunk_ms,
                point.profile_id,
                point.prefill_mode,
            ),
        )
        return PrefillChunkServicePrediction(
            requested_context_tokens=requested_context_tokens,
            requested_chunk_tokens=requested_chunk_tokens,
            bucket_context_tokens=selected.max_context_tokens,
            bucket_chunk_tokens=selected.max_chunk_tokens,
            p95_prefill_chunk_ms=selected.p95_prefill_chunk_ms,
            profile_id=selected.profile_id,
            prefill_mode=selected.prefill_mode,
            covered_variant_count=len(selected_by_variant),
        )

    @lru_cache(maxsize=4096)
    def estimate_remaining(
        self,
        *,
        logical_progress_tokens: int,
        prompt_tokens: int,
        quantum_tokens: int,
    ) -> PrefillRemainingServicePrediction:
        if quantum_tokens <= 0:
            raise ValueError("quantum_tokens must be positive")
        progress = min(prompt_tokens, max(0, logical_progress_tokens))
        estimated_remaining_ms = 0.0
        estimated_next_chunk_ms = 0.0
        remaining_chunks = 0
        covered_variant_count = 0
        while progress < prompt_tokens:
            chunk_tokens = min(quantum_tokens, prompt_tokens - progress)
            prediction = self.predict_chunk(
                context_tokens=progress + chunk_tokens,
                chunk_tokens=chunk_tokens,
            )
            if prediction is None:
                return PrefillRemainingServicePrediction(
                    estimated_remaining_ms=estimated_remaining_ms,
                    estimated_next_chunk_ms=estimated_next_chunk_ms,
                    remaining_chunks=remaining_chunks,
                    covered_variant_count=covered_variant_count,
                    profile_covered=False,
                    first_uncovered_context_tokens=progress + chunk_tokens,
                    first_uncovered_chunk_tokens=chunk_tokens,
                )
            if remaining_chunks == 0:
                estimated_next_chunk_ms = prediction.p95_prefill_chunk_ms
            estimated_remaining_ms += prediction.p95_prefill_chunk_ms
            remaining_chunks += 1
            covered_variant_count = max(
                covered_variant_count,
                prediction.covered_variant_count,
            )
            progress += chunk_tokens
        return PrefillRemainingServicePrediction(
            estimated_remaining_ms=estimated_remaining_ms,
            estimated_next_chunk_ms=estimated_next_chunk_ms,
            remaining_chunks=remaining_chunks,
            covered_variant_count=covered_variant_count,
        )


class PrefillChunkSchedulingPolicy(str, Enum):
    FCFS = "fcfs"
    EDF = "edf"
    SRPT = "srpt"
    SLO_SRPT = "slo_srpt"
    ROUND_ROBIN = "round_robin"


class PrefillRequest(Protocol):
    request_id: str
    arrival_time: float
    num_prompt_tokens: int
    num_computed_tokens: int
    num_tokens: int
    pd_recompute_frontier_tokens: int
    virtual_context_deferred_logical_computed_tokens: int
    ttft_prefill_deadline_ms: float | None


@dataclass(frozen=True)
class _PrefillServiceEstimate:
    remaining_tokens: int
    remaining_chunks: int
    estimated_remaining_ms: float
    estimated_chunk_ms: float
    source: str
    profile_configured: bool
    profile_covered: bool
    profile_fallback: bool
    profile_variant_count: int
    profile_uncovered_context_tokens: int
    profile_uncovered_chunk_tokens: int


@dataclass
class _SLOSRPTState:
    request: PrefillRequest
    estimate: _PrefillServiceEstimate
    deadline_ts: float
    key: tuple[int, float, float, float, str]
    metrics: dict[str, int | float | str]


@dataclass(frozen=True)
class _EDFFeasibility:
    infeasible: bool
    violation_count: int
    violation_sum_ms: float
    violation_max_ms: float
    predicted_completion_ms: dict[str, float]
    predicted_violation_ms: dict[str, float]


@dataclass(frozen=True)
class PrefillAdmissionCandidate:
    """One policy-ranked request before native scheduler admission."""

    request_id: str
    num_tokens: int
    needs_new_sequence: bool = False
    requires_single_target: bool = False


@dataclass(frozen=True)
class PrefillAdmissionFrontier:
    """Policy order that can fit in one native scheduler iteration."""

    request_ids: tuple[str, ...] = ()
    planned_tokens_by_request: tuple[tuple[str, int], ...] = ()
    token_budget_before: int = 0
    decode_reserved_tokens: int = 0
    token_budget_after: int = 0
    new_sequence_slots_before: int = 0
    new_sequence_slots_after: int = 0
    sequence_blocked_request_ids: tuple[str, ...] = ()
    stop_reason: str = "disabled"


def build_native_prefill_admission_frontier(
    candidates: Iterable[PrefillAdmissionCandidate],
    *,
    token_budget_before: int,
    decode_reserved_tokens: int = 0,
    new_sequence_slots: int,
    force_single_reason: str = "",
    request_limit: int = 0,
) -> PrefillAdmissionFrontier:
    """Fill native budgets in policy order; optional phase row cap defaults off.

    This helper is deliberately allocator-free. It adapts one global Prefill
    policy order to vLLM's separate RUNNING and WAITING scans; the caller's
    normal token calculation and ``allocate_slots()`` stay authoritative.
    """
    if token_budget_before < 0:
        raise ValueError("token_budget_before must be non-negative")
    if decode_reserved_tokens < 0:
        raise ValueError("decode_reserved_tokens must be non-negative")
    if new_sequence_slots < 0:
        raise ValueError("new_sequence_slots must be non-negative")
    if request_limit < 0:
        raise ValueError("request_limit must be non-negative")

    decode_reserved_tokens = min(
        token_budget_before,
        decode_reserved_tokens,
    )
    remaining_token_budget = (
        token_budget_before - decode_reserved_tokens
    )
    remaining_new_sequence_slots = new_sequence_slots
    if remaining_token_budget <= 0:
        return PrefillAdmissionFrontier(
            token_budget_before=token_budget_before,
            decode_reserved_tokens=decode_reserved_tokens,
            token_budget_after=0,
            new_sequence_slots_before=new_sequence_slots,
            new_sequence_slots_after=remaining_new_sequence_slots,
            stop_reason="decode_consumed_token_budget",
        )

    selected_request_ids: list[str] = []
    planned_tokens_by_request: list[tuple[str, int]] = []
    sequence_blocked_request_ids: list[str] = []
    stop_reason = "candidate_order_exhausted"

    for candidate in candidates:
        if candidate.num_tokens < 0:
            raise ValueError("candidate num_tokens must be non-negative")
        if (
            candidate.needs_new_sequence
            and remaining_new_sequence_slots <= 0
        ):
            sequence_blocked_request_ids.append(candidate.request_id)
            if force_single_reason:
                stop_reason = f"{force_single_reason}:max_num_seqs"
                break
            stop_reason = "max_num_seqs_skipped"
            continue
        if candidate.num_tokens <= 0:
            continue
        if candidate.requires_single_target and selected_request_ids:
            stop_reason = "single_target_tiered_prefill"
            break

        planned_tokens = min(
            candidate.num_tokens,
            remaining_token_budget,
        )
        if planned_tokens <= 0:
            break
        selected_request_ids.append(candidate.request_id)
        planned_tokens_by_request.append(
            (candidate.request_id, planned_tokens)
        )
        remaining_token_budget -= planned_tokens
        if candidate.needs_new_sequence:
            remaining_new_sequence_slots -= 1

        if force_single_reason:
            stop_reason = force_single_reason
            break
        if candidate.requires_single_target:
            stop_reason = "single_target_tiered_prefill"
            break
        if request_limit and len(selected_request_ids) >= request_limit:
            stop_reason = "isolation_phase_request_limit"
            break
        if remaining_token_budget <= 0:
            stop_reason = "max_num_batched_tokens"
            break

    return PrefillAdmissionFrontier(
        request_ids=tuple(selected_request_ids),
        planned_tokens_by_request=tuple(planned_tokens_by_request),
        token_budget_before=token_budget_before,
        decode_reserved_tokens=decode_reserved_tokens,
        token_budget_after=remaining_token_budget,
        new_sequence_slots_before=new_sequence_slots,
        new_sequence_slots_after=remaining_new_sequence_slots,
        sequence_blocked_request_ids=tuple(sequence_blocked_request_ids),
        stop_reason=stop_reason,
    )


class PrefillChunkScheduler:
    """Rank requests for the next Prefill scheduler traversal.

    The helper is intentionally independent from vLLM queue and KV ownership.
    It only ranks currently eligible requests. The caller lets the native
    token, sequence, and KV admission path determine how much of that order
    can run, and advances Round Robin state only after chunks were actually
    admitted by the scheduler.
    """

    def __init__(
        self,
        policy: str | PrefillChunkSchedulingPolicy,
        *,
        quantum_tokens: int = 0,
        default_deadline_ms: float = 0.0,
        estimated_ms_per_token: float = 0.0,
        service_profile_path: str = "",
        slo_guard_ms: float = 0.0,
        starvation_threshold_ms: float = 0.0,
        starvation_max_bypasses: int = 0,
    ) -> None:
        self.policy = PrefillChunkSchedulingPolicy(policy)
        if quantum_tokens < 0:
            raise ValueError("quantum_tokens must be non-negative")
        if not math.isfinite(default_deadline_ms) or default_deadline_ms < 0:
            raise ValueError("default_deadline_ms must be finite and non-negative")
        if (
            not math.isfinite(estimated_ms_per_token)
            or estimated_ms_per_token < 0
        ):
            raise ValueError(
                "estimated_ms_per_token must be finite and non-negative"
            )
        if not math.isfinite(slo_guard_ms) or slo_guard_ms < 0:
            raise ValueError("slo_guard_ms must be finite and non-negative")
        if (
            not math.isfinite(starvation_threshold_ms)
            or starvation_threshold_ms < 0
        ):
            raise ValueError(
                "starvation_threshold_ms must be finite and non-negative"
            )
        if starvation_max_bypasses < 0:
            raise ValueError(
                "starvation_max_bypasses must be non-negative"
            )
        self.quantum_tokens = int(quantum_tokens)
        self.default_deadline_ms = float(default_deadline_ms)
        self.estimated_ms_per_token = float(estimated_ms_per_token)
        self.service_profile_path = service_profile_path.strip()
        self.service_profile = (
            PrefillChunkServiceProfile.from_json_file(
                self.service_profile_path
            )
            if self.service_profile_path
            else None
        )
        self.slo_guard_ms = float(slo_guard_ms)
        self.starvation_threshold_ms = float(starvation_threshold_ms)
        self.starvation_max_bypasses = int(starvation_max_bypasses)
        self._rr_queue: list[str] = []
        self._candidate_order: tuple[str, ...] = ()
        self._candidate_ranks: dict[str, int] = {}
        self._chunk_counts: dict[str, int] = {}
        self._candidate_metrics: dict[
            str, dict[str, int | float | str]
        ] = {}
        self._last_chunk_service_at: dict[str, float] = {}
        self._starvation_bypass_counts: dict[str, int] = {}
        self._scheduled_chunk_estimates_ms: dict[str, float] = {}
        self._pending_service_ms = 0.0
        self._pending_prefill_request_count = 0
        self._iteration_now = 0.0
        self._last_scheduled_request_id = ""
        self._switch_count = 0

    @staticmethod
    def logical_prefill_boundary(request: PrefillRequest) -> int:
        """Include output-history replay forced by a Decode reset.

        ``pd_recompute_frontier_tokens`` is zero for the upstream/default
        path. The phase-separated scheduler sets it only after an exceptional
        reset of a live Decode request.
        """

        recompute_frontier = max(
            0,
            int(getattr(request, "pd_recompute_frontier_tokens", 0)),
        )
        request_tokens = max(
            0,
            int(getattr(request, "num_tokens", request.num_prompt_tokens)),
        )
        return max(
            request.num_prompt_tokens,
            min(recompute_frontier, request_tokens),
        )

    @classmethod
    def logical_prefill_progress(cls, request: PrefillRequest) -> int:
        return min(
            cls.logical_prefill_boundary(request),
            max(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            ),
        )

    @classmethod
    def remaining_prefill_tokens(cls, request: PrefillRequest) -> int:
        return max(
            0,
            cls.logical_prefill_boundary(request)
            - cls.logical_prefill_progress(request),
        )

    def deadline_timestamp(self, request: PrefillRequest) -> float:
        relative_ms = request.ttft_prefill_deadline_ms
        if relative_ms is None:
            relative_ms = self.default_deadline_ms
        if relative_ms <= 0:
            return math.inf
        return request.arrival_time + relative_ms / 1000.0

    @staticmethod
    def _fcfs_key(request: PrefillRequest) -> tuple[float, str]:
        return request.arrival_time, request.request_id

    def _estimate_prefill_service(
        self,
        request: PrefillRequest,
    ) -> _PrefillServiceEstimate:
        remaining_tokens = self.remaining_prefill_tokens(request)
        prefill_boundary = self.logical_prefill_boundary(request)
        chunk_tokens = min(remaining_tokens, self.quantum_tokens)
        remaining_chunks = (
            math.ceil(remaining_tokens / self.quantum_tokens)
            if remaining_tokens > 0 and self.quantum_tokens > 0
            else 0
        )
        if self.service_profile is not None and remaining_tokens > 0:
            prediction = self.service_profile.estimate_remaining(
                logical_progress_tokens=self.logical_prefill_progress(request),
                prompt_tokens=prefill_boundary,
                quantum_tokens=self.quantum_tokens,
            )
            if prediction.profile_covered:
                return _PrefillServiceEstimate(
                    remaining_tokens=remaining_tokens,
                    remaining_chunks=prediction.remaining_chunks,
                    estimated_remaining_ms=(
                        prediction.estimated_remaining_ms
                    ),
                    estimated_chunk_ms=(
                        prediction.estimated_next_chunk_ms
                    ),
                    source="prefill_chunk_profile_p95_sum",
                    profile_configured=True,
                    profile_covered=True,
                    profile_fallback=False,
                    profile_variant_count=(
                        prediction.covered_variant_count
                    ),
                    profile_uncovered_context_tokens=0,
                    profile_uncovered_chunk_tokens=0,
                )
            uncovered_context_tokens = (
                prediction.first_uncovered_context_tokens
            )
            uncovered_chunk_tokens = (
                prediction.first_uncovered_chunk_tokens
            )
        else:
            uncovered_context_tokens = 0
            uncovered_chunk_tokens = 0

        profile_configured = self.service_profile is not None
        return _PrefillServiceEstimate(
            remaining_tokens=remaining_tokens,
            remaining_chunks=remaining_chunks,
            estimated_remaining_ms=(
                remaining_tokens * self.estimated_ms_per_token
            ),
            estimated_chunk_ms=(
                chunk_tokens * self.estimated_ms_per_token
            ),
            source=(
                "configured_ms_per_token_profile_fallback"
                if profile_configured
                else "configured_ms_per_token"
            ),
            profile_configured=profile_configured,
            profile_covered=False,
            profile_fallback=profile_configured,
            profile_variant_count=0,
            profile_uncovered_context_tokens=uncovered_context_tokens,
            profile_uncovered_chunk_tokens=uncovered_chunk_tokens,
        )

    def _slo_srpt_state(
        self,
        request: PrefillRequest,
        *,
        now: float,
        estimate: _PrefillServiceEstimate,
        pending_service_ms: float = 0.0,
        projected_peer_wait_ms: float = 0.0,
    ) -> _SLOSRPTState:
        deadline_ts = self.deadline_timestamp(request)
        laxity_ms = (
            (deadline_ts - now) * 1000.0
            - max(0.0, pending_service_ms)
            - estimate.estimated_remaining_ms
            if math.isfinite(deadline_ts)
            else math.inf
        )
        last_service_at = self._last_chunk_service_at.get(
            request.request_id, request.arrival_time
        )
        service_age_ms = max(0.0, (now - last_service_at) * 1000.0)
        bypasses = self._starvation_bypass_counts.get(request.request_id, 0)
        critical = (
            math.isfinite(laxity_ms)
            and laxity_ms
            <= self.slo_guard_ms + estimate.estimated_chunk_ms
        )
        starved_by_current_age = (
            self.starvation_threshold_ms > 0
            and service_age_ms >= self.starvation_threshold_ms
        )
        projected_wait_ms = max(0.0, pending_service_ms) + max(
            0.0,
            projected_peer_wait_ms,
        )
        projected_service_age_ms = service_age_ms + projected_wait_ms
        starved_by_projected_age = (
            self.starvation_threshold_ms > 0
            and projected_wait_ms > 0
            and service_age_ms < self.starvation_threshold_ms
            and projected_service_age_ms >= self.starvation_threshold_ms
        )
        starved_by_age = (
            starved_by_current_age or starved_by_projected_age
        )
        starved_by_bypass = (
            self.starvation_max_bypasses > 0
            and bypasses >= self.starvation_max_bypasses
        )
        starved = starved_by_age or starved_by_bypass

        if critical:
            tier = 0
            tier_name = "slo_critical"
            secondary = laxity_ms
            tertiary = deadline_ts
        elif starved:
            tier = 1
            tier_name = "starved"
            secondary = -service_age_ms
            tertiary = -float(bypasses)
        else:
            tier = 2
            tier_name = "srpt_safe"
            secondary = estimate.estimated_remaining_ms
            tertiary = float(estimate.remaining_tokens)

        metrics: dict[str, int | float | str] = {
            "ttft_prefill_estimated_remaining_ms": (
                estimate.estimated_remaining_ms
            ),
            "ttft_prefill_estimated_chunk_ms": (
                estimate.estimated_chunk_ms
            ),
            "ttft_prefill_estimated_remaining_chunks": (
                estimate.remaining_chunks
            ),
            "ttft_prefill_laxity_ms": (
                laxity_ms if math.isfinite(laxity_ms) else -1.0
            ),
            "ttft_prefill_service_age_ms": service_age_ms,
            "ttft_prefill_projected_peer_wait_ms": max(
                0.0, projected_peer_wait_ms
            ),
            "ttft_prefill_pending_service_ms": max(
                0.0, pending_service_ms
            ),
            "ttft_prefill_projected_service_age_ms": (
                projected_service_age_ms
            ),
            "ttft_prefill_starvation_bypass_count": bypasses,
            "ttft_prefill_slo_critical": int(critical),
            "ttft_prefill_starved": int(starved),
            "ttft_prefill_starved_by_current_age": int(
                starved_by_current_age
            ),
            "ttft_prefill_starved_by_projected_age": int(
                starved_by_projected_age
            ),
            "ttft_prefill_starved_by_bypass": int(starved_by_bypass),
            "ttft_prefill_selection_tier": tier_name,
            "ttft_prefill_estimate_source": estimate.source,
            "ttft_prefill_service_profile_configured": int(
                estimate.profile_configured
            ),
            "ttft_prefill_service_profile_covered": int(
                estimate.profile_covered
            ),
            "ttft_prefill_service_profile_fallback": int(
                estimate.profile_fallback
            ),
            "ttft_prefill_service_profile_variant_count": (
                estimate.profile_variant_count
            ),
            "ttft_prefill_service_profile_uncovered_context_tokens": (
                estimate.profile_uncovered_context_tokens
            ),
            "ttft_prefill_service_profile_uncovered_chunk_tokens": (
                estimate.profile_uncovered_chunk_tokens
            ),
        }
        return _SLOSRPTState(
            request=request,
            estimate=estimate,
            deadline_ts=deadline_ts,
            key=(
                tier,
                secondary,
                tertiary,
                float(request.arrival_time),
                request.request_id,
            ),
            metrics=metrics,
        )

    @staticmethod
    def _evaluate_edf_feasibility(
        states: list[_SLOSRPTState],
        *,
        now: float,
        pending_service_ms: float,
    ) -> _EDFFeasibility:
        """Simulate current finite-deadline work in EDF completion order.

        Every candidate is already released and the scheduler admits at most
        one prefill request per iteration. Under this current-set model, EDF
        minimizes maximum lateness. A miss in this simulation therefore marks
        the current profiled workload as infeasible, subject to estimate error,
        future arrivals, and later KV/residency admission stalls.
        """

        finite_deadline_states = sorted(
            (
                state
                for state in states
                if math.isfinite(state.deadline_ts)
            ),
            key=lambda state: (
                state.deadline_ts,
                state.request.arrival_time,
                state.request.request_id,
            ),
        )
        cursor_ms = now * 1000.0 + max(0.0, pending_service_ms)
        predicted_completion_ms: dict[str, float] = {}
        predicted_violation_ms: dict[str, float] = {}
        for state in finite_deadline_states:
            cursor_ms += state.estimate.estimated_remaining_ms
            request_id = state.request.request_id
            predicted_completion_ms[request_id] = (
                cursor_ms - now * 1000.0
            )
            predicted_violation_ms[request_id] = max(
                0.0,
                cursor_ms - state.deadline_ts * 1000.0,
            )

        violations = [
            violation_ms
            for violation_ms in predicted_violation_ms.values()
            if violation_ms > 0
        ]
        return _EDFFeasibility(
            infeasible=bool(violations),
            violation_count=len(violations),
            violation_sum_ms=sum(violations),
            violation_max_ms=max(violations, default=0.0),
            predicted_completion_ms=predicted_completion_ms,
            predicted_violation_ms=predicted_violation_ms,
        )

    @staticmethod
    def _apply_edf_metrics(
        states: list[_SLOSRPTState],
        feasibility: _EDFFeasibility,
    ) -> None:
        for state in states:
            request_id = state.request.request_id
            state.metrics.update(
                {
                    "ttft_prefill_overload_edf_active": int(
                        feasibility.infeasible
                    ),
                    "ttft_prefill_edf_infeasible_current_set": int(
                        feasibility.infeasible
                    ),
                    "ttft_prefill_edf_predicted_violation_count": (
                        feasibility.violation_count
                    ),
                    "ttft_prefill_edf_predicted_violation_sum_ms": (
                        feasibility.violation_sum_ms
                    ),
                    "ttft_prefill_edf_predicted_violation_max_ms": (
                        feasibility.violation_max_ms
                    ),
                    "ttft_prefill_edf_predicted_completion_from_now_ms": (
                        feasibility.predicted_completion_ms.get(
                            request_id, -1.0
                        )
                    ),
                    "ttft_prefill_edf_predicted_request_violation_ms": (
                        feasibility.predicted_violation_ms.get(
                            request_id, 0.0
                        )
                    ),
                }
            )

    def _order_slo_srpt_requests(
        self,
        requests: list[PrefillRequest],
        *,
        now: float,
    ) -> list[PrefillRequest]:
        estimates = {
            request.request_id: self._estimate_prefill_service(request)
            for request in requests
        }
        initial_states = [
            self._slo_srpt_state(
                request,
                now=now,
                estimate=estimates[request.request_id],
                pending_service_ms=self._pending_service_ms,
            )
            for request in requests
        ]
        if not initial_states:
            self._candidate_metrics = {}
            return []
        feasibility = self._evaluate_edf_feasibility(
            initial_states,
            now=now,
            pending_service_ms=self._pending_service_ms,
        )
        if feasibility.infeasible:
            self._apply_edf_metrics(initial_states, feasibility)
            for state in initial_states:
                state.metrics["ttft_prefill_pre_overload_selection_tier"] = (
                    state.metrics["ttft_prefill_selection_tier"]
                )
                state.metrics["ttft_prefill_selection_tier"] = "overload_edf"
            ordered_states = sorted(
                initial_states,
                key=lambda state: (
                    state.deadline_ts,
                    state.request.arrival_time,
                    state.request.request_id,
                ),
            )
        else:
            provisional = min(initial_states, key=lambda state: state.key)
            projected_peer_wait_ms = (
                provisional.estimate.estimated_chunk_ms
            )
            final_states = [
                self._slo_srpt_state(
                    state.request,
                    now=now,
                    estimate=state.estimate,
                    pending_service_ms=self._pending_service_ms,
                    projected_peer_wait_ms=(
                        0.0
                        if state.request.request_id
                        == provisional.request.request_id
                        else projected_peer_wait_ms
                    ),
                )
                for state in initial_states
            ]
            self._apply_edf_metrics(final_states, feasibility)
            ordered_states = sorted(final_states, key=lambda state: state.key)

        self._candidate_metrics = {
            state.request.request_id: state.metrics
            for state in ordered_states
        }
        return [state.request for state in ordered_states]

    def _ordered_requests(
        self,
        requests: list[PrefillRequest],
        *,
        now: float,
    ) -> list[PrefillRequest]:
        if self.policy == PrefillChunkSchedulingPolicy.FCFS:
            return sorted(requests, key=self._fcfs_key)
        if self.policy == PrefillChunkSchedulingPolicy.EDF:
            return sorted(
                requests,
                key=lambda request: (
                    self.deadline_timestamp(request),
                    request.arrival_time,
                    request.request_id,
                ),
            )
        if self.policy == PrefillChunkSchedulingPolicy.SRPT:
            return sorted(
                requests,
                key=lambda request: (
                    self.remaining_prefill_tokens(request),
                    request.arrival_time,
                    request.request_id,
                ),
            )
        if self.policy == PrefillChunkSchedulingPolicy.SLO_SRPT:
            return self._order_slo_srpt_requests(requests, now=now)

        by_id = {request.request_id: request for request in requests}
        candidate_ids = set(by_id)
        self._rr_queue = [
            request_id
            for request_id in self._rr_queue
            if request_id in candidate_ids
        ]
        queued_ids = set(self._rr_queue)
        new_requests = sorted(
            (
                request
                for request in requests
                if request.request_id not in queued_ids
            ),
            key=self._fcfs_key,
        )
        self._rr_queue.extend(request.request_id for request in new_requests)
        return [by_id[request_id] for request_id in self._rr_queue]

    def begin_iteration(
        self,
        requests: Iterable[PrefillRequest],
        *,
        now: float | None = None,
        pending_request_ids: Iterable[str] = (),
    ) -> tuple[str, ...]:
        self._iteration_now = time.time() if now is None else float(now)
        if not math.isfinite(self._iteration_now):
            raise ValueError("iteration time must be finite")
        pending_prefill_estimates = [
            self._scheduled_chunk_estimates_ms[request_id]
            for request_id in set(pending_request_ids)
            if request_id in self._scheduled_chunk_estimates_ms
        ]
        self._pending_service_ms = sum(pending_prefill_estimates)
        self._pending_prefill_request_count = len(pending_prefill_estimates)
        self._candidate_metrics = {}
        unique_requests: dict[str, PrefillRequest] = {}
        for request in requests:
            if self.remaining_prefill_tokens(request) <= 0:
                continue
            unique_requests.setdefault(request.request_id, request)
        ordered = self._ordered_requests(
            list(unique_requests.values()),
            now=self._iteration_now,
        )
        self._candidate_order = tuple(request.request_id for request in ordered)
        self._candidate_ranks = {
            request_id: rank
            for rank, request_id in enumerate(self._candidate_order)
        }
        return self._candidate_order

    @property
    def preferred_request_id(self) -> str:
        return self._candidate_order[0] if self._candidate_order else ""

    @property
    def candidate_order(self) -> tuple[str, ...]:
        return self._candidate_order

    def prioritize_selected_candidate_slots(
        self,
        requests: Iterable[PrefillRequest],
        *,
        selected_request_ids: Iterable[str],
    ) -> list[PrefillRequest]:
        """Order an admission frontier inside Prefill-only queue slots.

        Decode-ready and otherwise ineligible entries retain their exact
        indices. Frontier Prefills occupy the earliest Prefill candidate slots
        in global policy order; requests outside the frontier follow without
        changing the positions owned by Decode or dependency-blocked work.
        """
        ordered = list(requests)
        selected_ids = tuple(
            dict.fromkeys(
                request_id
                for request_id in selected_request_ids
                if request_id in self._candidate_ranks
            )
        )
        if not selected_ids or not self._candidate_order:
            return ordered
        candidate_indices = [
            index
            for index, request in enumerate(ordered)
            if request.request_id in self._candidate_ranks
        ]
        if not candidate_indices:
            return ordered
        requests_by_id = {
            request.request_id: request
            for request in ordered
            if request.request_id in self._candidate_ranks
        }
        selected_id_set = set(selected_ids)
        selected_candidates = [
            requests_by_id[request_id]
            for request_id in selected_ids
            if request_id in requests_by_id
        ]
        unselected_candidates = [
            request
            for request in ordered
            if request.request_id in self._candidate_ranks
            and request.request_id not in selected_id_set
        ]
        ranked_candidates = selected_candidates + unselected_candidates
        for index, request in zip(candidate_indices, ranked_candidates):
            ordered[index] = request
        return ordered

    def prioritize_preferred_candidate_slot(
        self, requests: Iterable[PrefillRequest]
    ) -> list[PrefillRequest]:
        """Move the preferred request into the first candidate-only slot.

        Entries that are not part of this iteration's prefill candidates keep
        their exact indices. The scheduler uses this to leave decode-ready
        waiting requests untouched while reordering only prefills.
        """
        return self.prioritize_selected_candidate_slots(
            requests,
            selected_request_ids=self._candidate_order[:1],
        )

    def cap_prefill_tokens(
        self,
        request: PrefillRequest,
        num_new_tokens: int,
        *,
        num_computed_tokens: int | None = None,
    ) -> int:
        if self.quantum_tokens <= 0 or num_new_tokens <= 0:
            return num_new_tokens
        progress = (
            self.logical_prefill_progress(request)
            if num_computed_tokens is None
            else min(
                self.logical_prefill_boundary(request),
                max(
                    num_computed_tokens,
                    request.virtual_context_deferred_logical_computed_tokens,
                ),
            )
        )
        prefill_boundary = self.logical_prefill_boundary(request)
        if progress >= prefill_boundary:
            return num_new_tokens
        return min(
            num_new_tokens,
            self.quantum_tokens,
            prefill_boundary - progress,
        )

    def observe_scheduled_chunks(self, request_ids: Iterable[str]) -> None:
        selected_request_ids = tuple(
            dict.fromkeys(
                request_id
                for request_id in request_ids
                if request_id in self._candidate_ranks
            )
        )
        selected_request_id_set = set(selected_request_ids)
        if selected_request_id_set:
            for candidate_id in self._candidate_order:
                if candidate_id in selected_request_id_set:
                    self._starvation_bypass_counts[candidate_id] = 0
                else:
                    self._starvation_bypass_counts[candidate_id] = (
                        self._starvation_bypass_counts.get(candidate_id, 0) + 1
                    )

        for request_id in selected_request_ids:
            if request_id not in self._candidate_ranks:
                continue
            estimated_chunk_ms = self._candidate_metrics.get(
                request_id, {}
            ).get("ttft_prefill_estimated_chunk_ms", 0.0)
            self._scheduled_chunk_estimates_ms[request_id] = float(
                estimated_chunk_ms
            )
            self._last_chunk_service_at[request_id] = self._iteration_now
            self._chunk_counts[request_id] = (
                self._chunk_counts.get(request_id, 0) + 1
            )
            if (
                self._last_scheduled_request_id
                and self._last_scheduled_request_id != request_id
            ):
                self._switch_count += 1
            self._last_scheduled_request_id = request_id
            if (
                self.policy == PrefillChunkSchedulingPolicy.ROUND_ROBIN
                and request_id in self._rr_queue
            ):
                index = self._rr_queue.index(request_id)
                self._rr_queue = (
                    self._rr_queue[index + 1 :] + self._rr_queue[: index + 1]
                )

    def forget_request(self, request_id: str) -> None:
        self._candidate_ranks.pop(request_id, None)
        self._candidate_metrics.pop(request_id, None)
        self._chunk_counts.pop(request_id, None)
        self._last_chunk_service_at.pop(request_id, None)
        self._starvation_bypass_counts.pop(request_id, None)
        self._scheduled_chunk_estimates_ms.pop(request_id, None)
        if request_id in self._rr_queue:
            self._rr_queue.remove(request_id)

    def request_metrics(
        self, request: PrefillRequest
    ) -> dict[str, int | float | str]:
        deadline_ts = self.deadline_timestamp(request)
        metrics: dict[str, int | float | str] = {
            "ttft_prefill_chunk_policy_rank": self._candidate_ranks.get(
                request.request_id, -1
            ),
            "ttft_prefill_policy_chunk_count": self._chunk_counts.get(
                request.request_id, 0
            ),
            "ttft_prefill_remaining_tokens": self.remaining_prefill_tokens(
                request
            ),
            "ttft_prefill_deadline_ms": (
                request.ttft_prefill_deadline_ms
                if request.ttft_prefill_deadline_ms is not None
                else self.default_deadline_ms
            ),
            "ttft_prefill_deadline_ts": (
                deadline_ts if math.isfinite(deadline_ts) else -1.0
            ),
        }
        metrics.update(self._candidate_metrics.get(request.request_id, {}))
        return metrics

    def iteration_metrics(self) -> dict[str, int | float | str]:
        preferred_metrics = self._candidate_metrics.get(
            self.preferred_request_id, {}
        )
        return {
            "ttft_prefill_chunk_scheduling_policy": self.policy.value,
            "ttft_prefill_chunk_quantum_tokens": self.quantum_tokens,
            "ttft_prefill_estimated_ms_per_token": (
                self.estimated_ms_per_token
            ),
            "ttft_prefill_estimate_source": (
                str(
                    preferred_metrics.get(
                        "ttft_prefill_estimate_source",
                        "not_used",
                    )
                )
            ),
            "ttft_prefill_service_profile_path": self.service_profile_path,
            "ttft_prefill_service_profile_configured": int(
                self.service_profile is not None
            ),
            "ttft_prefill_service_profile_fallback_candidate_count": sum(
                int(
                    metrics.get(
                        "ttft_prefill_service_profile_fallback",
                        0,
                    )
                )
                for metrics in self._candidate_metrics.values()
            ),
            "ttft_prefill_pending_service_ms": self._pending_service_ms,
            "ttft_prefill_pending_prefill_request_count": (
                self._pending_prefill_request_count
            ),
            "ttft_prefill_slo_guard_ms": self.slo_guard_ms,
            "ttft_prefill_starvation_threshold_ms": (
                self.starvation_threshold_ms
            ),
            "ttft_prefill_starvation_max_bypasses": (
                self.starvation_max_bypasses
            ),
            "ttft_prefill_chunk_candidate_count": len(self._candidate_order),
            "ttft_prefill_chunk_candidate_ids": ",".join(
                self._candidate_order
            ),
            "ttft_prefill_chunk_preferred_request_id": (
                self.preferred_request_id
            ),
            "ttft_prefill_chunk_last_scheduled_request_id": (
                self._last_scheduled_request_id
            ),
            "ttft_prefill_chunk_switch_count_total": self._switch_count,
            "ttft_prefill_slo_critical_candidate_count": sum(
                int(metrics.get("ttft_prefill_slo_critical", 0))
                for metrics in self._candidate_metrics.values()
            ),
            "ttft_prefill_starved_candidate_count": sum(
                int(metrics.get("ttft_prefill_starved", 0))
                for metrics in self._candidate_metrics.values()
            ),
            "ttft_prefill_preferred_selection_tier": str(
                preferred_metrics.get("ttft_prefill_selection_tier", "")
            ),
            "ttft_prefill_overload_edf_active": int(
                preferred_metrics.get(
                    "ttft_prefill_overload_edf_active",
                    0,
                )
            ),
            "ttft_prefill_edf_predicted_violation_count": int(
                preferred_metrics.get(
                    "ttft_prefill_edf_predicted_violation_count",
                    0,
                )
            ),
            "ttft_prefill_edf_predicted_violation_sum_ms": float(
                preferred_metrics.get(
                    "ttft_prefill_edf_predicted_violation_sum_ms",
                    0.0,
                )
            ),
            "ttft_prefill_edf_predicted_violation_max_ms": float(
                preferred_metrics.get(
                    "ttft_prefill_edf_predicted_violation_max_ms",
                    0.0,
                )
            ),
        }
