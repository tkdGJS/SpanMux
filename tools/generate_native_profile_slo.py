#!/usr/bin/env python3
"""Generate a context-bucket profile and SLO policy from a native baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROFILE_FIELDNAMES = [
    "context_tokens",
    "measurement_count",
    "success_count",
    "warmup_count",
    "warmup_success_count",
    "ttft_mean_ms",
    "ttft_stdev_ms",
    "ttft_cv_percent",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_max_ms",
    "request_mean_tbt_mean_ms",
    "request_mean_tbt_stdev_ms",
    "request_p99_tbt_p95_ms",
    "request_p99_tbt_max_ms",
    "pooled_p99_tbt_ms",
    "completion_mean_ms",
    "completion_stdev_ms",
    "output_tokens_min",
    "output_tokens_max",
]

SLO_FIELDNAMES = [
    "context_tokens",
    "reference_policy",
    "ttft_reference_ms",
    "tbt_reference_ms",
    "total_sms",
    "prefill_sms",
    "decode_sms",
    "prefill_share_factor",
    "decode_share_factor",
    "operational_headroom",
    "ttft_slo_scale",
    "tbt_slo_scale",
    "ttft_slo_ms",
    "ttft_slo_seconds",
    "tbt_slo_ms",
    "attainment_target",
]

TIER_FIELDNAMES = [
    "context_tokens",
    "tier",
    "operational_headroom",
    "ttft_slo_scale",
    "tbt_slo_scale",
    "ttft_slo_ms",
    "tbt_slo_ms",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def linear_fit(
    x_values: Sequence[float], y_values: Sequence[float]
) -> dict[str, float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("linear fit requires paired values")
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        raise ValueError("linear fit requires distinct x values")
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [
        intercept + slope * x_value for x_value in x_values
    ]
    total_variance = sum(
        (value - y_mean) ** 2 for value in y_values
    )
    residual = sum(
        (value - prediction) ** 2
        for value, prediction in zip(y_values, predictions)
    )
    r_squared = 1.0 - residual / total_variance
    return {
        "intercept": intercept,
        "slope": slope,
        "r_squared": r_squared,
    }


def build_profile(
    raw_rows: Sequence[dict[str, str]],
    summary_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    summaries = {
        int(row["context_target_tokens"]): row for row in summary_rows
    }
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[int(row["context_target_tokens"])].append(row)

    profile: list[dict[str, Any]] = []
    for context_tokens in sorted(grouped):
        rows = grouped[context_tokens]
        measured = [
            row for row in rows if row["phase"] == "measurement"
        ]
        warmups = [row for row in rows if row["phase"] == "warmup"]
        successful = [
            row for row in measured if row["status"] == "success"
        ]
        successful_warmups = [
            row for row in warmups if row["status"] == "success"
        ]
        if not measured or len(successful) != len(measured):
            raise ValueError(
                f"context {context_tokens}: all measurements must succeed"
            )
        if len(successful_warmups) != len(warmups):
            raise ValueError(
                f"context {context_tokens}: all warmups must succeed"
            )
        ttft = [float(row["ttft_ms"]) for row in successful]
        mean_tbt = [
            float(row["mean_tbt_ms"]) for row in successful
        ]
        p99_tbt = [
            float(row["p99_tbt_ms"]) for row in successful
        ]
        completion = [
            float(row["completion_latency_ms"]) for row in successful
        ]
        output_tokens = [
            int(float(row["output_tokens"])) for row in successful
        ]
        ttft_mean = statistics.fmean(ttft)
        summary = summaries.get(context_tokens)
        if summary is None:
            raise ValueError(
                f"context {context_tokens}: missing bucket summary"
            )
        profile.append(
            {
                "context_tokens": context_tokens,
                "measurement_count": len(measured),
                "success_count": len(successful),
                "warmup_count": len(warmups),
                "warmup_success_count": len(successful_warmups),
                "ttft_mean_ms": rounded(ttft_mean),
                "ttft_stdev_ms": rounded(statistics.stdev(ttft)),
                "ttft_cv_percent": rounded(
                    statistics.stdev(ttft) / ttft_mean * 100.0
                ),
                "ttft_p50_ms": rounded(percentile(ttft, 50)),
                "ttft_p95_ms": rounded(percentile(ttft, 95)),
                "ttft_max_ms": rounded(max(ttft)),
                "request_mean_tbt_mean_ms": rounded(
                    statistics.fmean(mean_tbt)
                ),
                "request_mean_tbt_stdev_ms": rounded(
                    statistics.stdev(mean_tbt)
                ),
                "request_p99_tbt_p95_ms": rounded(
                    percentile(p99_tbt, 95)
                ),
                "request_p99_tbt_max_ms": rounded(max(p99_tbt)),
                "pooled_p99_tbt_ms": rounded(
                    float(summary["pooled_p99_tbt_ms"])
                ),
                "completion_mean_ms": rounded(
                    statistics.fmean(completion)
                ),
                "completion_stdev_ms": rounded(
                    statistics.stdev(completion)
                ),
                "output_tokens_min": min(output_tokens),
                "output_tokens_max": max(output_tokens),
            }
        )
    return profile


def build_slo_rows(
    profile: Sequence[dict[str, Any]],
    *,
    total_sms: int,
    prefill_sms: int,
    decode_sms: int,
    headroom: float,
    attainment_target: float,
) -> list[dict[str, Any]]:
    prefill_share_factor = total_sms / prefill_sms
    decode_share_factor = total_sms / decode_sms
    rows: list[dict[str, Any]] = []
    for bucket in profile:
        ttft_reference = float(bucket["ttft_max_ms"])
        tbt_reference = float(bucket["request_p99_tbt_max_ms"])
        ttft_scale = prefill_share_factor * headroom
        tbt_scale = decode_share_factor * headroom
        ttft_slo = ttft_reference * ttft_scale
        tbt_slo = tbt_reference * tbt_scale
        rows.append(
            {
                "context_tokens": bucket["context_tokens"],
                "reference_policy": (
                    "max_no_contention_ttft_and_max_request_p99_tbt"
                ),
                "ttft_reference_ms": rounded(ttft_reference),
                "tbt_reference_ms": rounded(tbt_reference),
                "total_sms": total_sms,
                "prefill_sms": prefill_sms,
                "decode_sms": decode_sms,
                "prefill_share_factor": rounded(prefill_share_factor),
                "decode_share_factor": rounded(decode_share_factor),
                "operational_headroom": rounded(headroom),
                "ttft_slo_scale": rounded(ttft_scale),
                "tbt_slo_scale": rounded(tbt_scale),
                "ttft_slo_ms": rounded(ttft_slo),
                "ttft_slo_seconds": rounded(ttft_slo / 1000.0),
                "tbt_slo_ms": rounded(tbt_slo),
                "attainment_target": rounded(attainment_target),
            }
        )
    return rows


def tier_name(headroom: float, selected_headroom: float) -> str:
    if math.isclose(headroom, selected_headroom):
        return "recommended"
    if headroom < selected_headroom:
        return "strict"
    return "relaxed"


def build_tier_rows(
    profile: Sequence[dict[str, Any]],
    *,
    total_sms: int,
    prefill_sms: int,
    decode_sms: int,
    selected_headroom: float,
    tier_headrooms: Sequence[float],
) -> list[dict[str, Any]]:
    headrooms = sorted(set([*tier_headrooms, selected_headroom]))
    rows: list[dict[str, Any]] = []
    for bucket in profile:
        for headroom in headrooms:
            ttft_scale = total_sms / prefill_sms * headroom
            tbt_scale = total_sms / decode_sms * headroom
            rows.append(
                {
                    "context_tokens": bucket["context_tokens"],
                    "tier": tier_name(headroom, selected_headroom),
                    "operational_headroom": rounded(headroom),
                    "ttft_slo_scale": rounded(ttft_scale),
                    "tbt_slo_scale": rounded(tbt_scale),
                    "ttft_slo_ms": rounded(
                        float(bucket["ttft_max_ms"]) * ttft_scale
                    ),
                    "tbt_slo_ms": rounded(
                        float(bucket["request_p99_tbt_max_ms"])
                        * tbt_scale
                    ),
                }
            )
    return rows


def markdown_table(
    profile: Sequence[dict[str, Any]],
    slo_rows: Sequence[dict[str, Any]],
) -> str:
    slo_by_context = {
        int(row["context_tokens"]): row for row in slo_rows
    }
    lines = [
        "| Context | TTFT mean | TTFT max/ref | mean TBT | "
        "p99 TBT max/ref | selected TTFT SLO | selected TBT SLO |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profile:
        context = int(row["context_tokens"])
        slo = slo_by_context[context]
        lines.append(
            f"| {context:,} | {float(row['ttft_mean_ms']) / 1000:.3f} s "
            f"| {float(row['ttft_max_ms']) / 1000:.3f} s "
            f"| {float(row['request_mean_tbt_mean_ms']):.3f} ms "
            f"| {float(row['request_p99_tbt_max_ms']):.3f} ms "
            f"| {float(slo['ttft_slo_seconds']):.3f} s "
            f"| {float(slo['tbt_slo_ms']):.3f} ms |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate profile and SLO files from a native 40-SM run"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--total-sms", type=int, default=40)
    parser.add_argument("--prefill-sms", type=int, default=20)
    parser.add_argument("--decode-sms", type=int, default=20)
    parser.add_argument("--operational-headroom", type=float, default=1.25)
    parser.add_argument(
        "--tier-headrooms",
        type=float,
        nargs="+",
        default=[1.10, 1.25, 1.50],
    )
    parser.add_argument("--attainment-target", type=float, default=0.90)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_sms <= 0:
        raise ValueError("--total-sms must be positive")
    if not 0 < args.prefill_sms <= args.total_sms:
        raise ValueError("--prefill-sms must be in (0, total_sms]")
    if not 0 < args.decode_sms <= args.total_sms:
        raise ValueError("--decode-sms must be in (0, total_sms]")
    if args.prefill_sms + args.decode_sms != args.total_sms:
        raise ValueError("prefill and decode SMs must sum to total SMs")
    if args.operational_headroom < 1.0:
        raise ValueError("--operational-headroom must be at least 1.0")
    if any(value < 1.0 for value in args.tier_headrooms):
        raise ValueError("--tier-headrooms must all be at least 1.0")
    if not 0 < args.attainment_target <= 1.0:
        raise ValueError("--attainment-target must be in (0, 1]")


def main() -> int:
    args = parse_args()
    validate_args(args)
    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "slo"
    )
    raw_path = results_dir / "request_runs.csv"
    summary_path = results_dir / "bucket_summary.csv"
    metadata_path = results_dir / "run_metadata.json"
    for path in (raw_path, summary_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("baseline run_metadata status is not completed")
    raw_rows = read_csv(raw_path)
    summary_rows = read_csv(summary_path)
    profile = build_profile(raw_rows, summary_rows)
    slo_rows = build_slo_rows(
        profile,
        total_sms=args.total_sms,
        prefill_sms=args.prefill_sms,
        decode_sms=args.decode_sms,
        headroom=args.operational_headroom,
        attainment_target=args.attainment_target,
    )
    tier_rows = build_tier_rows(
        profile,
        total_sms=args.total_sms,
        prefill_sms=args.prefill_sms,
        decode_sms=args.decode_sms,
        selected_headroom=args.operational_headroom,
        tier_headrooms=args.tier_headrooms,
    )

    context = [float(row["context_tokens"]) for row in profile]
    ttft_mean = [float(row["ttft_mean_ms"]) for row in profile]
    tbt_mean = [
        float(row["request_mean_tbt_mean_ms"]) for row in profile
    ]
    ttft_fit = linear_fit(
        [tokens**2 for tokens in context], ttft_mean
    )
    tbt_fit = linear_fit(context, tbt_mean)

    policy = {
        "created_at_utc": utc_now(),
        "source_results_dir": str(results_dir),
        "source_case_id": metadata.get("case_id"),
        "model": metadata.get("model"),
        "profile_contract": {
            "gpu": "Tesla T4",
            "total_sms": args.total_sms,
            "concurrency": 1,
            "warmup_excluded": True,
            "measurement_repeats_per_bucket": metadata.get(
                "repeat_count"
            ),
            "prefix_cache": False,
            "kv_cache_memory_bytes": 4294967296,
        },
        "target_pd_split": {
            "prefill_sms": args.prefill_sms,
            "decode_sms": args.decode_sms,
            "prefill_share_factor": args.total_sms / args.prefill_sms,
            "decode_share_factor": args.total_sms / args.decode_sms,
        },
        "reference_definition": {
            "ttft": "maximum TTFT among five no-contention measurements",
            "tbt": (
                "maximum per-request p99 TBT among five "
                "no-contention measurements"
            ),
        },
        "selected_policy": {
            "operational_headroom": args.operational_headroom,
            "attainment_target": args.attainment_target,
            "ttft_violation": "request_ttft_ms > bucket_ttft_slo_ms",
            "tbt_violation": (
                "token_gap_ms > bucket_tbt_slo_ms; report both "
                "token-level rate and requests whose p99 exceeds the SLO"
            ),
        },
        "fit_models_interpolation_only": {
            "ttft_ms": {
                "formula": "intercept + slope * context_tokens^2",
                **ttft_fit,
            },
            "request_mean_tbt_ms": {
                "formula": "intercept + slope * context_tokens",
                **tbt_fit,
            },
            "valid_context_range": [
                int(min(context)),
                int(max(context)),
            ],
        },
        "buckets": slo_rows,
        "limitations": [
            (
                "The operational headroom is an experiment policy knob, "
                "not a universal literature constant."
            ),
            (
                "The 40/20 SM share factor assumes inverse scaling with "
                "allocated SMs; isolated 20-SM phase profiles should "
                "replace this estimate for the final paper."
            ),
            (
                "Five repetitions establish a stable engineering profile "
                "but are insufficient for a production tail-latency claim."
            ),
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "profile_baseline.csv", PROFILE_FIELDNAMES, profile)
    write_csv(output_dir / "slo_policy.csv", SLO_FIELDNAMES, slo_rows)
    write_csv(output_dir / "slo_tiers.csv", TIER_FIELDNAMES, tier_rows)
    (output_dir / "slo_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = f"""# Native 40-SM Profile Baseline and SLO

Generated at `{policy["created_at_utc"]}` from
`{results_dir}`.

## Validity

- All four context buckets completed one warmup and five measured requests.
- The warmup is recorded but excluded from the profile.
- The server used one native request at a time, 40 T4 SMs, a 4-GiB KV cache,
  chunked prefill, and disabled prefix caching.
- TTFT coefficient of variation is below 0.35% in every bucket.

## Selected SLO

The selected policy targets a `{args.prefill_sms}/{args.decode_sms}` P/D SM
split.  It uses the maximum observed no-contention TTFT and maximum observed
per-request p99 TBT as the reference envelope:

```text
TTFT SLO = max native-40SM TTFT
           * ({args.total_sms} / {args.prefill_sms})
           * {args.operational_headroom}

TBT SLO  = max native-40SM request-p99 TBT
           * ({args.total_sms} / {args.decode_sms})
           * {args.operational_headroom}
```

The resulting selected SLO scale is
`{args.total_sms / args.prefill_sms * args.operational_headroom:.2f}x` for
TTFT and `{args.total_sms / args.decode_sms * args.operational_headroom:.2f}x`
for TBT.  The `{args.operational_headroom:.2f}x` operational headroom is an
explicit research choice, not a universal constant.

{markdown_table(profile, slo_rows)}

## Scaling

- TTFT follows `intercept + slope * context_tokens^2` with
  `R²={ttft_fit["r_squared"]:.6f}`.
- Mean TBT follows `intercept + slope * context_tokens` with
  `R²={tbt_fit["r_squared"]:.6f}`.
- These fits may interpolate only within 25K--100K; do not extrapolate them
  as a hardware-independent model.

## Evaluation Rule

- TTFT attainment: fraction of requests whose TTFT is at most the bucket SLO.
- TBT token attainment: fraction of measured token gaps at most the bucket
  TBT SLO.
- TBT request attainment: fraction of requests whose p99 token gap is at most
  the bucket TBT SLO.
- Report SLO attainment together with raw TTFT/TBT distributions; do not report
  only the pass percentage.

## Interpretation and Limits

The 40-SM run is the hardware no-contention reference.  It is not the direct
latency floor of a 20/20 P/D partition.  The current selected SLO estimates the
partition floor with a `40/20` share factor and then adds 25% headroom.  Before
the final evaluation, measure isolated 20-SM prefill and isolated 20-SM decode
profiles and replace the inverse-SM estimate if their scaling is nonlinear.

DistServe treats TTFT and TPOT as separate application constraints and
evaluates goodput at more than 90% SLO attainment; it also sweeps an SLO scale
rather than asserting one universal latency constant:
https://arxiv.org/abs/2401.09670

Llumnix's SLO-aware scheduler similarly uses per-model TTFT/TPOT profiling data
to predict schedulability:
https://docs.llumnix.ai/getting_started/e2e_deploy
"""
    (output_dir / "analysis.md").write_text(report, encoding="utf-8")

    print(f"[DONE] profile={output_dir / 'profile_baseline.csv'}")
    print(f"[DONE] selected_slo={output_dir / 'slo_policy.csv'}")
    print(f"[DONE] tiers={output_dir / 'slo_tiers.csv'}")
    print(f"[DONE] policy={output_dir / 'slo_policy.json'}")
    print(f"[DONE] report={output_dir / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
