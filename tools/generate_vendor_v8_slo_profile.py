#!/usr/bin/env python3
"""Build regression-based request SLOs from native 40-SM baselines.

The native baseline is a hardware service reference.  The generated request
SLO does not change when the later P/D partition changes; partition profiling
measures whether each split can satisfy that fixed request contract.

One invocation may combine disjoint context buckets measured in separate
completed result directories. This lets expensive exact-context runs be reused
without mechanically merging their CSV files first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from generate_native_profile_slo import (
    build_profile,
    linear_fit,
    read_csv,
    rounded,
)


DEFAULT_CONTEXT_TOKENS = (25_000, 50_000, 75_000, 100_000)

SLO_FIELDNAMES = [
    "context_tokens",
    "ttft_reference_p95_ms",
    "ttft_regression_ms",
    "ttft_one_sided_residual_guard_ms",
    "ttft_slo_ms",
    "tbt_reference_request_p99_p95_ms",
    "tbt_regression_ms",
    "tbt_one_sided_residual_guard_ms",
    "tbt_slo_ms",
    "ttft_slo_multiplier",
    "tbt_slo_multiplier",
]

SPLITWISE_P50_TTFT_MULTIPLIER = 2.0
SPLITWISE_P50_TBT_MULTIPLIER = 1.25
TBT_MEASUREMENT_SOURCE = "choice.token_ids"
TBT_GAP_FIELD = "estimated_per_token_gap_ms"
TBT_UNIT = "ms_per_output_token"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RegressionEnvelope:
    fit_kind: str
    formula: str
    intercept: float
    slope: float
    r_squared: float | None
    one_sided_residual_guard_ms: float
    feature: Callable[[float], float]

    def predict_reference_ms(self, context_tokens: int) -> float:
        predicted = self.intercept + self.slope * self.feature(context_tokens)
        return max(0.001, predicted + self.one_sided_residual_guard_ms)

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "fit_kind": self.fit_kind,
            "formula": self.formula,
            "intercept": self.intercept,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "one_sided_residual_guard_ms": (
                self.one_sided_residual_guard_ms
            ),
        }


@dataclass(frozen=True)
class BaselineSource:
    results_dir: Path
    case_id: str
    model: str
    contexts: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "results_dir": str(self.results_dir),
            "case_id": self.case_id,
            "model": self.model,
            "contexts": list(self.contexts),
        }


_BASELINE_COMPATIBILITY_FIELDS = (
    "model",
    "repeat_count",
    "warmup_count_per_bucket",
    "cooldown_sec",
    "request_max_tokens",
    "request_min_tokens",
    "ignore_eos",
    "submission_mode",
    "completion_boundary",
)


def load_baseline_profiles(
    baseline_results_dirs: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[BaselineSource]]:
    """Load disjoint completed baseline buckets with strict compatibility.

    Duplicate contexts are rejected instead of being silently averaged or
    selected. Cross-run repetition needs an explicit aggregation contract.
    """

    if not baseline_results_dirs:
        raise ValueError("at least one baseline results directory is required")

    combined_profile: list[dict[str, Any]] = []
    sources: list[BaselineSource] = []
    context_sources: dict[int, Path] = {}
    reference_metadata: dict[str, Any] | None = None
    reference_dir: Path | None = None

    for raw_results_dir in baseline_results_dirs:
        results_dir = raw_results_dir.expanduser().resolve()
        raw_path = results_dir / "request_runs.csv"
        summary_path = results_dir / "bucket_summary.csv"
        metadata_path = results_dir / "run_metadata.json"
        for path in (raw_path, summary_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            raise ValueError(
                "native baseline metadata status must be completed: "
                f"{results_dir}"
            )
        if (
            metadata.get("tbt_token_count_source") != TBT_MEASUREMENT_SOURCE
            or metadata.get("tbt_requires_return_token_ids") is not True
        ):
            raise ValueError(
                "native baseline must use direct choice.token_ids TBT "
                f"measurement: {results_dir}; legacy text/BOS archives "
                "must be remeasured before generating a runtime SLO policy"
            )
        if reference_metadata is None:
            reference_metadata = metadata
            reference_dir = results_dir
        else:
            mismatches = [
                field
                for field in _BASELINE_COMPATIBILITY_FIELDS
                if metadata.get(field) != reference_metadata.get(field)
            ]
            if mismatches:
                assert reference_dir is not None
                details = ", ".join(
                    f"{field}={reference_metadata.get(field)!r}/"
                    f"{metadata.get(field)!r}"
                    for field in mismatches
                )
                raise ValueError(
                    "native baseline inputs are not comparable: "
                    f"reference={reference_dir} candidate={results_dir}; "
                    f"mismatches: {details}"
                )

        profile = build_profile(read_csv(raw_path), read_csv(summary_path))
        contexts = tuple(int(row["context_tokens"]) for row in profile)
        for context in contexts:
            previous = context_sources.get(context)
            if previous is not None:
                raise ValueError(
                    "duplicate native baseline context bucket "
                    f"{context}: {previous} and {results_dir}"
                )
            context_sources[context] = results_dir
        combined_profile.extend(profile)
        sources.append(
            BaselineSource(
                results_dir=results_dir,
                case_id=str(metadata.get("case_id") or ""),
                model=str(metadata.get("model") or ""),
                contexts=contexts,
            )
        )

    combined_profile.sort(key=lambda row: int(row["context_tokens"]))
    return combined_profile, sources


def fit_envelope(
    contexts: Sequence[int],
    observations_ms: Sequence[float],
    *,
    feature: Callable[[float], float],
    formula: str,
) -> RegressionEnvelope:
    if len(contexts) != len(observations_ms) or not contexts:
        raise ValueError(
            "SLO envelope requires at least one paired observation"
        )
    if len(contexts) == 1:
        # A quick one-context SM-scaling experiment needs a fixed SLO reference,
        # not an underdetermined cross-context regression. Keep R^2 explicitly
        # undefined so this mode cannot be mistaken for regression evidence.
        return RegressionEnvelope(
            fit_kind="single_point_constant",
            formula=(
                "constant = observed_ms at context_tokens="
                f"{int(contexts[0])}"
            ),
            intercept=float(observations_ms[0]),
            slope=0.0,
            r_squared=None,
            one_sided_residual_guard_ms=0.0,
            feature=feature,
        )
    transformed = [feature(float(context)) for context in contexts]
    fit = linear_fit(transformed, observations_ms)
    residual_guard = max(
        0.0,
        max(
            observation
            - (
                float(fit["intercept"])
                + float(fit["slope"]) * transformed_value
            )
            for observation, transformed_value in zip(
                observations_ms, transformed
            )
        ),
    )
    return RegressionEnvelope(
        fit_kind="ols_with_one_sided_residual_envelope",
        formula=formula,
        intercept=float(fit["intercept"]),
        slope=float(fit["slope"]),
        r_squared=float(fit["r_squared"]),
        one_sided_residual_guard_ms=residual_guard,
        feature=feature,
    )


def require_contexts(
    profile: Sequence[dict[str, Any]],
    required_contexts: Sequence[int],
) -> list[dict[str, Any]]:
    by_context = {
        int(row["context_tokens"]): dict(row)
        for row in profile
    }
    required = list(dict.fromkeys(int(value) for value in required_contexts))
    if not required or any(value <= 0 for value in required):
        raise ValueError("required contexts must contain positive values")
    missing = [value for value in required if value not in by_context]
    if missing:
        raise ValueError(
            "native baseline is missing required context buckets: "
            f"{missing}; available={sorted(by_context)}"
        )
    return [by_context[value] for value in required]


def build_regression_slo(
    profile: Sequence[dict[str, Any]],
    *,
    required_contexts: Sequence[int] = DEFAULT_CONTEXT_TOKENS,
    ttft_slo_multiplier: float = SPLITWISE_P50_TTFT_MULTIPLIER,
    tbt_slo_multiplier: float = SPLITWISE_P50_TBT_MULTIPLIER,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name, multiplier in (
        ("TTFT SLO multiplier", ttft_slo_multiplier),
        ("TBT SLO multiplier", tbt_slo_multiplier),
    ):
        if not math.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError(f"{name} must be finite and at least 1")
    selected = require_contexts(profile, required_contexts)
    contexts = [int(row["context_tokens"]) for row in selected]
    ttft_observations = [
        float(row["ttft_p95_ms"]) for row in selected
    ]
    tbt_observations = [
        float(row["request_p99_tbt_p95_ms"]) for row in selected
    ]
    ttft_envelope = fit_envelope(
        contexts,
        ttft_observations,
        feature=lambda value: value * value,
        formula="intercept + slope * context_tokens^2",
    )
    tbt_envelope = fit_envelope(
        contexts,
        tbt_observations,
        feature=lambda value: value,
        formula="intercept + slope * context_tokens",
    )

    rows: list[dict[str, Any]] = []
    for context, ttft_observed, tbt_observed in zip(
        contexts, ttft_observations, tbt_observations
    ):
        ttft_reference = ttft_envelope.predict_reference_ms(context)
        tbt_reference = tbt_envelope.predict_reference_ms(context)
        rows.append(
            {
                "context_tokens": context,
                "ttft_reference_p95_ms": rounded(ttft_observed),
                "ttft_regression_ms": rounded(
                    ttft_reference
                    - ttft_envelope.one_sided_residual_guard_ms
                ),
                "ttft_one_sided_residual_guard_ms": rounded(
                    ttft_envelope.one_sided_residual_guard_ms
                ),
                "ttft_slo_ms": rounded(
                    ttft_reference * ttft_slo_multiplier
                ),
                "tbt_reference_request_p99_p95_ms": rounded(
                    tbt_observed
                ),
                "tbt_regression_ms": rounded(
                    tbt_reference
                    - tbt_envelope.one_sided_residual_guard_ms
                ),
                "tbt_one_sided_residual_guard_ms": rounded(
                    tbt_envelope.one_sided_residual_guard_ms
                ),
                "tbt_slo_ms": rounded(
                    tbt_reference * tbt_slo_multiplier
                ),
                "ttft_slo_multiplier": rounded(ttft_slo_multiplier),
                "tbt_slo_multiplier": rounded(tbt_slo_multiplier),
            }
        )

    conservative_prefill_ms_per_token = max(
        ttft_envelope.predict_reference_ms(context) / context
        for context in contexts
    )
    policy = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "reference": {
            "hardware": "Tesla T4",
            "total_sms": 40,
            "concurrency": 1,
            "policy": "native_no_contention",
        },
        "required_context_tokens": contexts,
        "calibration_mode": (
            "single_point_constant"
            if len(contexts) == 1
            else "multi_point_regression_envelope"
        ),
        "slo_multipliers": {
            "method": "splitwise_p50_slowdown",
            "ttft": ttft_slo_multiplier,
            "tbt": tbt_slo_multiplier,
        },
        "regression": {
            "ttft": ttft_envelope.as_dict(),
            "tbt": tbt_envelope.as_dict(),
        },
        "slo_contract": {
            "lookup": (
                "smallest max_context_tokens covering request; requests "
                "above the largest measured bucket are rejected"
            ),
            "partition_invariant": True,
            "ttft_definition": (
                "TTFT p95 at the single measured context * "
                "Splitwise P50 TTFT slowdown multiplier "
                f"({ttft_slo_multiplier:g}x)"
                if len(contexts) == 1
                else (
                    "(TTFT p95 regression + maximum positive fit residual) "
                    "* Splitwise P50 TTFT slowdown multiplier "
                    f"({ttft_slo_multiplier:g}x)"
                )
            ),
            "tbt_definition": (
                "request-p99-TBT p95 at the single measured context * "
                "Splitwise P50 TBT slowdown multiplier "
                f"({tbt_slo_multiplier:g}x)"
                if len(contexts) == 1
                else (
                    "(request-p99-TBT p95 regression + maximum positive fit "
                    "residual) * Splitwise P50 TBT slowdown multiplier "
                    f"({tbt_slo_multiplier:g}x)"
                )
            ),
            "tbt_measurement_source": TBT_MEASUREMENT_SOURCE,
            "tbt_gap_field": TBT_GAP_FIELD,
            "tbt_unit": TBT_UNIT,
            "tbt_requires_return_token_ids": True,
            "tbt_computation": (
                "For each streamed response chunk, count new output tokens "
                "from choice.token_ids and divide the chunk inter-arrival "
                "gap by that token count. Request TBT percentiles and token "
                "SLO attainment are computed over those per-token gaps."
            ),
            "ttft_measurement_source": "first streamed chunk with non-empty choice.token_ids",
            "ttft_unit": "ms_from_submit_to_first_output_token_chunk",
        },
        "prefill_fallback_ms_per_token": rounded(
            conservative_prefill_ms_per_token, 8
        ),
        "buckets": [
            {
                "max_context_tokens": int(row["context_tokens"]),
                "ttft_prefill_deadline_ms": float(row["ttft_slo_ms"]),
                "tbt_slo_ms": float(row["tbt_slo_ms"]),
            }
            for row in rows
        ],
        "limitations": [
            (
                "The fit is calibrated only inside the measured "
                f"{min(contexts)}..{max(contexts)} token range."
            ),
            (
                "A single measured bucket defines only a constant local "
                "reference; it provides no cross-context regression or "
                "extrapolation evidence."
                if len(contexts) == 1
                else (
                    f"Summed p95 and a {len(contexts)}-point regression are "
                    "engineering profiles, not a hard real-time guarantee."
                )
            ),
            (
                "P/D partition feasibility is measured separately; the "
                "request SLO is not relaxed for a slower partition."
            ),
        ],
    }
    return policy, rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SLO_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate vendor_v8 regression SLOs from one or more completed "
            "native 40-SM baselines"
        )
    )
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "one or more completed baseline result directories with disjoint "
            "context buckets"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--required-context-tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_TOKENS),
    )
    parser.add_argument(
        "--ttft-slo-multiplier",
        type=float,
        default=SPLITWISE_P50_TTFT_MULTIPLIER,
        help="TTFT slowdown multiplier (Splitwise P50 default: 2.0)",
    )
    parser.add_argument(
        "--tbt-slo-multiplier",
        type=float,
        default=SPLITWISE_P50_TBT_MULTIPLIER,
        help="TBT slowdown multiplier (Splitwise P50 default: 1.25)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile, sources = load_baseline_profiles(args.baseline_results_dir)
    policy, rows = build_regression_slo(
        profile,
        required_contexts=args.required_context_tokens,
        ttft_slo_multiplier=args.ttft_slo_multiplier,
        tbt_slo_multiplier=args.tbt_slo_multiplier,
    )
    policy["source_baselines"] = [source.as_dict() for source in sources]
    policy["source_results_dirs"] = [
        str(source.results_dir) for source in sources
    ]
    policy["source_case_ids"] = [source.case_id for source in sources]
    policy["model"] = sources[0].model
    if len(sources) == 1:
        # Preserve the legacy provenance keys and scalar types.
        policy["source_results_dir"] = str(sources[0].results_dir)
        policy["source_case_id"] = sources[0].case_id

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "request_slo_policy.json"
    csv_path = output_dir / "request_slo_policy.csv"
    policy_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, rows)
    print(f"[DONE] request_slo_policy={policy_path}")
    print(f"[DONE] request_slo_csv={csv_path}")
    ttft_r2 = policy["regression"]["ttft"]["r_squared"]
    tbt_r2 = policy["regression"]["tbt"]["r_squared"]
    print(
        "[FIT] "
        f"mode={policy['calibration_mode']} "
        f"ttft_r2={'n/a' if ttft_r2 is None else f'{ttft_r2:.6f}'} "
        f"tbt_r2={'n/a' if tbt_r2 is None else f'{tbt_r2:.6f}'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
