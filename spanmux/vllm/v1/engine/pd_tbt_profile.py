# SPDX-License-Identifier: Apache-2.0
"""Measured overlap-profile model for TBT-first P/D SM allocation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PREFILL_MODES = frozenset({"none", "native", "ring", "any"})
_SCHEMA_VERSION = 2
_MEASUREMENT_CONTRACT = {
    "token_count_source": "choice.token_ids",
    "gap_field": "estimated_per_token_gap_ms",
    "unit": "ms_per_output_token",
    "chunk_gap_normalization": (
        "interarrival_ms_divided_by_emitted_token_count"
    ),
    "first_nonempty_chunk": "excluded_as_ttft",
}


@dataclass(frozen=True)
class TBTOverlapProfilePoint:
    """One conservative p95 per-output-token TBT observation."""

    profile_id: str
    max_decode_context_tokens: int
    max_decode_requests: int
    prefill_mode: str
    p95_per_token_tbt_ms: float

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        index: int,
    ) -> "TBTOverlapProfilePoint":
        try:
            profile_id = str(raw["profile_id"]).strip()
            max_context = int(raw["max_decode_context_tokens"])
            max_requests = int(raw["max_decode_requests"])
            prefill_mode = str(raw["prefill_mode"]).strip().lower()
            p95_ms = float(raw["p95_per_token_tbt_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid TBT overlap profile point at index {index}: {raw!r}"
            ) from exc

        if not profile_id:
            raise ValueError(
                f"TBT overlap profile point {index} has an empty profile_id"
            )
        if max_context <= 0:
            raise ValueError(
                "max_decode_context_tokens must be positive at "
                f"TBT overlap profile point {index}"
            )
        if max_requests <= 0:
            raise ValueError(
                "max_decode_requests must be positive at "
                f"TBT overlap profile point {index}"
            )
        if prefill_mode not in _PREFILL_MODES:
            raise ValueError(
                f"invalid prefill_mode={prefill_mode!r} at TBT overlap "
                f"profile point {index}; expected one of "
                f"{sorted(_PREFILL_MODES)}"
            )
        if not math.isfinite(p95_ms) or p95_ms <= 0:
            raise ValueError(
                "p95_per_token_tbt_ms must be a positive finite number at "
                f"TBT overlap profile point {index}"
            )
        return cls(
            profile_id=profile_id,
            max_decode_context_tokens=max_context,
            max_decode_requests=max_requests,
            prefill_mode=prefill_mode,
            p95_per_token_tbt_ms=p95_ms,
        )


@dataclass(frozen=True)
class TBTProfilePrediction:
    profile_id: str
    requested_context_tokens: int
    requested_decode_requests: int
    requested_prefill_mode: str
    bucket_context_tokens: int
    bucket_decode_requests: int
    bucket_prefill_mode: str
    p95_per_token_tbt_ms: float


class TBTOverlapProfile:
    """Lookup model that never extrapolates beyond measured buckets."""

    def __init__(
        self,
        points: tuple[TBTOverlapProfilePoint, ...],
        *,
        source: str = "",
    ) -> None:
        if not points:
            raise ValueError("TBT overlap profile must contain at least one point")
        self.points = points
        self.source = source

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        source: str = "",
    ) -> "TBTOverlapProfile":
        schema_version = raw.get("schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                "unsupported TBT overlap profile schema_version="
                f"{schema_version!r}; expected {_SCHEMA_VERSION}. Regenerate "
                "the profile from exact per-token choice.token_ids data."
            )
        contract = raw.get("measurement_contract")
        if not isinstance(contract, dict):
            raise ValueError(
                "TBT overlap profile measurement_contract must be an object"
            )
        mismatches = {
            key: (contract.get(key), expected)
            for key, expected in _MEASUREMENT_CONTRACT.items()
            if contract.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "TBT overlap profile is not exact per-output-token data: "
                f"measurement_contract mismatches={mismatches!r}"
            )
        raw_points = raw.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("TBT overlap profile 'points' must be a list")
        points = tuple(
            TBTOverlapProfilePoint.from_dict(point, index=index)
            for index, point in enumerate(raw_points)
            if isinstance(point, dict)
        )
        if len(points) != len(raw_points):
            raise ValueError("every TBT overlap profile point must be an object")
        return cls(points, source=source)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TBTOverlapProfile":
        profile_path = Path(path).expanduser()
        try:
            raw = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"failed to load TBT overlap profile {profile_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError("TBT overlap profile root must be an object")
        return cls.from_dict(raw, source=str(profile_path))

    def predict(
        self,
        *,
        profile_id: str,
        decode_context_tokens: int,
        decode_requests: int,
        prefill_mode: str,
    ) -> TBTProfilePrediction | None:
        requested_mode = prefill_mode.strip().lower()
        if requested_mode not in {"none", "native", "ring"}:
            raise ValueError(
                f"invalid requested prefill mode {prefill_mode!r}"
            )
        requested_context = max(1, int(decode_context_tokens))
        requested_requests = max(1, int(decode_requests))
        covering = [
            point
            for point in self.points
            if point.profile_id == profile_id
            and point.max_decode_context_tokens >= requested_context
            and point.max_decode_requests >= requested_requests
            and point.prefill_mode in {requested_mode, "any"}
        ]
        if not covering:
            return None

        # Prefer the tightest measured bucket and a mode-specific observation.
        # Duplicate observations at the same bucket are reduced conservatively
        # to their maximum p95 rather than silently selecting the best run.
        best_bucket = min(
            (
                point.max_decode_context_tokens,
                point.max_decode_requests,
                int(point.prefill_mode == "any"),
            )
            for point in covering
        )
        bucket_points = [
            point
            for point in covering
            if (
                point.max_decode_context_tokens,
                point.max_decode_requests,
                int(point.prefill_mode == "any"),
            )
            == best_bucket
        ]
        selected = max(
            bucket_points,
            key=lambda point: point.p95_per_token_tbt_ms,
        )
        return TBTProfilePrediction(
            profile_id=profile_id,
            requested_context_tokens=requested_context,
            requested_decode_requests=requested_requests,
            requested_prefill_mode=requested_mode,
            bucket_context_tokens=selected.max_decode_context_tokens,
            bucket_decode_requests=selected.max_decode_requests,
            bucket_prefill_mode=selected.prefill_mode,
            p95_per_token_tbt_ms=selected.p95_per_token_tbt_ms,
        )

    def predict_context_envelope(
        self,
        *,
        profile_id: str,
        context_start: int,
        context_end: int,
        decode_requests: int,
        prefill_mode: str,
    ) -> TBTProfilePrediction | None:
        """Worst lookup across a growth interval, without assuming monotonic TBT.

        A faster bucket at the endpoint must not hide a slower intermediate
        bucket. Lookup changes only immediately after a measured context bound.
        All probes, including the endpoint, must have measured coverage.
        """
        start = max(1, int(context_start))
        end = max(start, int(context_end))
        probes = {start, end}
        probes.update(
            point.max_decode_context_tokens + 1
            for point in self.points
            if point.profile_id == profile_id
            and point.prefill_mode in {prefill_mode, "any"}
            and point.max_decode_requests >= max(1, decode_requests)
            and start <= point.max_decode_context_tokens < end
        )
        worst = None
        for context in probes:
            prediction = self.predict(
                profile_id=profile_id,
                decode_context_tokens=context,
                decode_requests=decode_requests,
                prefill_mode=prefill_mode,
            )
            if prediction is None:
                return None
            if worst is None or prediction.p95_per_token_tbt_ms > worst.p95_per_token_tbt_ms:
                worst = prediction
        return worst
