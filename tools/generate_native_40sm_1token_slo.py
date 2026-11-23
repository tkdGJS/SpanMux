#!/usr/bin/env python3
"""Recompute native 40-SM one-token TBT and Splitwise-style SLOs.

The input may be either a direct ``choice.token_ids`` result or one of the
validated legacy archives that retokenized each SSE text fragment and added
one BOS token per fragment.  Legacy inputs are accepted only when the
reconstructed token count exactly matches every fixed-output request.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


QUANTILES = (50, 90, 99)
SPLITWISE_FACTORS = {
    50: {"ttft": 2.0, "tbt": 1.25, "e2e": 1.25},
    90: {"ttft": 3.0, "tbt": 1.5, "e2e": 1.5},
    99: {"ttft": 6.0, "tbt": 5.0, "e2e": 5.0},
}
EXACT_PROVENANCE = "choice_token_ids"
LEGACY_PROVENANCE = "legacy_archive_bos_reconstruction"

RESULT_FIELDS = (
    "context_tokens",
    "quantile",
    "measurement_request_count",
    "one_token_tbt_gap_count",
    "provenance",
    "profiling_ttft_ms",
    "ttft_slo_factor",
    "ttft_slo_ms",
    "profiling_one_token_tbt_ms",
    "tbt_slo_factor",
    "tbt_slo_ms",
    "profiling_e2e_ms",
    "e2e_slo_factor",
    "e2e_slo_ms",
)
SOURCE_FIELDS = (
    "context_tokens",
    "results_dir",
    "case_id",
    "provenance",
    "measurement_request_count",
    "one_token_tbt_gap_count",
    "requested_output_tokens",
)
REQUEST_FIELDS = (
    "context_tokens",
    "request_id",
    "repeat_index",
    "provenance",
    "output_tokens",
    "ttft_ms",
    "one_token_tbt_mean_ms",
    "one_token_tbt_p50_ms",
    "one_token_tbt_p90_ms",
    "one_token_tbt_p99_ms",
    "e2e_ms",
)
RUNTIME_POLICY_FIELDS = (
    "context_tokens",
    "profiling_p50_ttft_ms",
    "ttft_slo_factor",
    "ttft_prefill_deadline_ms",
    "profiling_p50_one_token_tbt_ms",
    "tbt_slo_factor",
    "tbt_slo_ms",
    "calibration_provenance",
)


@dataclass(frozen=True)
class ContextSamples:
    context_tokens: int
    results_dir: Path
    case_id: str
    provenance: str
    requested_output_tokens: int
    ttft_ms: tuple[float, ...]
    one_token_tbt_ms: tuple[float, ...]
    e2e_ms: tuple[float, ...]
    request_rows: tuple[dict[str, Any], ...]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], quantile: int) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = quantile / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def detect_provenance(metadata: dict[str, Any], results_dir: Path) -> str:
    source = metadata.get("tbt_token_count_source")
    requires_ids = metadata.get("tbt_requires_return_token_ids")
    if source == "choice.token_ids" and requires_ids is True:
        return EXACT_PROVENANCE
    if source is None and requires_ids is None:
        return LEGACY_PROVENANCE
    raise ValueError(
        f"{results_dir}: inconsistent TBT provenance: "
        f"source={source!r}, requires_return_token_ids={requires_ids!r}"
    )


def fragment_model_tokens(recorded: int, provenance: str) -> int:
    if recorded <= 0:
        raise ValueError(f"invalid non-positive chunk token count: {recorded}")
    if provenance == EXACT_PROVENANCE:
        return recorded
    if provenance == LEGACY_PROVENANCE:
        model_tokens = recorded - 1
        if model_tokens <= 0:
            raise ValueError(
                "legacy fragment does not contain a token after BOS removal: "
                f"recorded={recorded}"
            )
        return model_tokens
    raise AssertionError(provenance)


def load_context_samples(results_dir: Path) -> ContextSamples:
    results_dir = results_dir.resolve()
    metadata_path = results_dir / "run_metadata.json"
    request_path = results_dir / "request_runs.csv"
    token_path = results_dir / "token_timeline.csv"
    for path in (metadata_path, request_path, token_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(f"{results_dir}: experiment is not completed")
    provenance = detect_provenance(metadata, results_dir)

    measured_requests = [
        row
        for row in read_csv(request_path)
        if row.get("phase") == "measurement"
    ]
    if not measured_requests:
        raise ValueError(f"{results_dir}: no measurement requests")
    if any(row.get("status") != "success" for row in measured_requests):
        raise ValueError(f"{results_dir}: not every measurement request succeeded")

    contexts = {int(row["context_target_tokens"]) for row in measured_requests}
    if len(contexts) != 1:
        raise ValueError(
            f"{results_dir}: expected one context bucket, found {sorted(contexts)}"
        )
    context_tokens = next(iter(contexts))
    requested_outputs = {
        int(row["requested_max_tokens"]) for row in measured_requests
    }
    if len(requested_outputs) != 1:
        raise ValueError(
            f"{results_dir}: fixed output-token contract is required"
        )
    requested_output_tokens = next(iter(requested_outputs))

    token_rows_by_request: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(token_path):
        if row.get("phase") == "measurement":
            token_rows_by_request[row["request_id"]].append(row)

    all_tbt: list[float] = []
    request_audit: list[dict[str, Any]] = []
    for request in measured_requests:
        request_id = request["request_id"]
        token_rows = sorted(
            token_rows_by_request.get(request_id, []),
            key=lambda row: int(row["chunk_index"]),
        )
        if not token_rows:
            raise ValueError(f"{results_dir}: missing token rows for {request_id}")

        reconstructed_tokens = 0
        request_gaps: list[float] = []
        first_fragment_tokens: int | None = None
        for row in token_rows:
            recorded = int(row.get("chunk_tokens") or 0)
            model_tokens = fragment_model_tokens(recorded, provenance)
            reconstructed_tokens += model_tokens
            if first_fragment_tokens is None:
                first_fragment_tokens = model_tokens
            raw_gap = row.get("chunk_gap_ms")
            if raw_gap in (None, ""):
                continue
            one_token_gap = float(raw_gap) / model_tokens
            request_gaps.extend([one_token_gap] * model_tokens)

        if first_fragment_tokens != 1:
            raise ValueError(
                f"{results_dir}: {request_id} first fragment represents "
                f"{first_fragment_tokens} tokens; exact first-token TBT gaps "
                "cannot be reconstructed"
            )
        if reconstructed_tokens != requested_output_tokens:
            raise ValueError(
                f"{results_dir}: {request_id} reconstructed output tokens "
                f"{reconstructed_tokens} != requested {requested_output_tokens}"
            )
        if len(request_gaps) != requested_output_tokens - 1:
            raise ValueError(
                f"{results_dir}: {request_id} reconstructed TBT gaps "
                f"{len(request_gaps)} != expected {requested_output_tokens - 1}"
            )
        if provenance == LEGACY_PROVENANCE:
            legacy_full_count = int(float(request["output_tokens"]))
            if legacy_full_count != requested_output_tokens + 1:
                raise ValueError(
                    f"{results_dir}: {request_id} legacy full-text count "
                    f"{legacy_full_count} does not equal BOS + requested output"
                )

        all_tbt.extend(request_gaps)
        request_audit.append(
            {
                "context_tokens": context_tokens,
                "request_id": request_id,
                "repeat_index": int(request["repeat_index"]),
                "provenance": provenance,
                "output_tokens": reconstructed_tokens,
                "ttft_ms": round(float(request["ttft_ms"]), 6),
                "one_token_tbt_mean_ms": round(fmean(request_gaps), 6),
                "one_token_tbt_p50_ms": round(percentile(request_gaps, 50), 6),
                "one_token_tbt_p90_ms": round(percentile(request_gaps, 90), 6),
                "one_token_tbt_p99_ms": round(percentile(request_gaps, 99), 6),
                "e2e_ms": round(float(request["completion_latency_ms"]), 6),
            }
        )

    return ContextSamples(
        context_tokens=context_tokens,
        results_dir=results_dir,
        case_id=str(metadata.get("case_id") or ""),
        provenance=provenance,
        requested_output_tokens=requested_output_tokens,
        ttft_ms=tuple(float(row["ttft_ms"]) for row in measured_requests),
        one_token_tbt_ms=tuple(all_tbt),
        e2e_ms=tuple(
            float(row["completion_latency_ms"]) for row in measured_requests
        ),
        request_rows=tuple(request_audit),
    )


def build_result_rows(samples: Sequence[ContextSamples]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: item.context_tokens):
        for quantile in QUANTILES:
            factors = SPLITWISE_FACTORS[quantile]
            ttft = percentile(sample.ttft_ms, quantile)
            tbt = percentile(sample.one_token_tbt_ms, quantile)
            e2e = percentile(sample.e2e_ms, quantile)
            rows.append(
                {
                    "context_tokens": sample.context_tokens,
                    "quantile": f"P{quantile}",
                    "measurement_request_count": len(sample.ttft_ms),
                    "one_token_tbt_gap_count": len(sample.one_token_tbt_ms),
                    "provenance": sample.provenance,
                    "profiling_ttft_ms": round(ttft, 6),
                    "ttft_slo_factor": factors["ttft"],
                    "ttft_slo_ms": round(ttft * factors["ttft"], 6),
                    "profiling_one_token_tbt_ms": round(tbt, 6),
                    "tbt_slo_factor": factors["tbt"],
                    "tbt_slo_ms": round(tbt * factors["tbt"], 6),
                    "profiling_e2e_ms": round(e2e, 6),
                    "e2e_slo_factor": factors["e2e"],
                    "e2e_slo_ms": round(e2e * factors["e2e"], 6),
                }
            )
    return rows


def build_runtime_policy(
    rows: Sequence[dict[str, Any]],
    samples: Sequence[ContextSamples],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the bucket policy consumed by the vendor_v8 benchmark.

    Splitwise factors are quantile matched. The online benchmark uses the P50
    policy contract, while P90/P99 remain evaluation/reporting thresholds.
    Runtime TBT observations are direct ``choice.token_ids`` gaps even when a
    historical calibration archive required validated BOS reconstruction.
    """

    p50_rows = sorted(
        (row for row in rows if row["quantile"] == "P50"),
        key=lambda row: int(row["context_tokens"]),
    )
    if not p50_rows:
        raise ValueError("runtime policy requires at least one P50 row")
    if len(p50_rows) != len(samples):
        raise ValueError("runtime policy P50 rows and source buckets differ")

    sample_by_context = {sample.context_tokens: sample for sample in samples}
    policy_rows: list[dict[str, Any]] = []
    buckets: list[dict[str, Any]] = []
    for row in p50_rows:
        context = int(row["context_tokens"])
        sample = sample_by_context.get(context)
        if sample is None:
            raise ValueError(f"missing source sample for context {context}")
        policy_row = {
            "context_tokens": context,
            "profiling_p50_ttft_ms": float(row["profiling_ttft_ms"]),
            "ttft_slo_factor": float(row["ttft_slo_factor"]),
            "ttft_prefill_deadline_ms": float(row["ttft_slo_ms"]),
            "profiling_p50_one_token_tbt_ms": float(
                row["profiling_one_token_tbt_ms"]
            ),
            "tbt_slo_factor": float(row["tbt_slo_factor"]),
            "tbt_slo_ms": float(row["tbt_slo_ms"]),
            "calibration_provenance": sample.provenance,
        }
        policy_rows.append(policy_row)
        buckets.append(
            {
                "max_context_tokens": context,
                "ttft_prefill_deadline_ms": policy_row[
                    "ttft_prefill_deadline_ms"
                ],
                "tbt_slo_ms": policy_row["tbt_slo_ms"],
            }
        )

    source_rows = [
        {
            "results_dir": str(sample.results_dir),
            "case_id": sample.case_id,
            "contexts": [sample.context_tokens],
            "provenance": sample.provenance,
            "requested_output_tokens": sample.requested_output_tokens,
        }
        for sample in sorted(samples, key=lambda item: item.context_tokens)
    ]
    provenance_values = sorted({sample.provenance for sample in samples})
    fallback_ms_per_token = max(
        float(row["profiling_p50_ttft_ms"]) / int(row["context_tokens"])
        for row in policy_rows
    )
    policy = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "hardware": "Tesla T4",
            "total_sms": 40,
            "concurrency": 1,
            "policy": "native_no_contention",
        },
        "required_context_tokens": [
            int(row["context_tokens"]) for row in p50_rows
        ],
        "calibration_mode": "measured_bucket_lookup",
        "calibration_provenance": {
            "measurement_sources": provenance_values,
            "direct_choice_token_ids": provenance_values == [EXACT_PROVENANCE],
            "note": (
                "legacy_archive_bos_reconstruction is a validated historical "
                "archive reconstruction, not a direct exact-ID calibration"
            ),
        },
        "slo_multipliers": {
            "method": "splitwise_quantile_matched_p50_slowdown",
            "ttft": SPLITWISE_FACTORS[50]["ttft"],
            "tbt": SPLITWISE_FACTORS[50]["tbt"],
        },
        "slo_contract": {
            "lookup": (
                "smallest max_context_tokens covering request; requests "
                "above the largest measured bucket are rejected"
            ),
            "partition_invariant": True,
            "ttft_definition": "native 40-SM P50 TTFT * 2",
            "tbt_definition": (
                "native 40-SM pooled P50 one-output-token TBT * 1.25"
            ),
            "tbt_measurement_source": "choice.token_ids",
            "tbt_gap_field": "estimated_per_token_gap_ms",
            "tbt_unit": "ms_per_output_token",
            "tbt_requires_return_token_ids": True,
            "tbt_computation": (
                "At runtime, divide each streamed chunk inter-arrival gap by "
                "the number of new choice.token_ids; exclude the first token "
                "as TTFT."
            ),
            "ttft_measurement_source": (
                "first streamed chunk with non-empty choice.token_ids"
            ),
            "ttft_unit": "ms_from_submit_to_first_output_token_chunk",
        },
        "prefill_fallback_ms_per_token": round(
            fallback_ms_per_token, 8
        ),
        "buckets": buckets,
        "limitations": [
            "P50 policy thresholds are descriptive Splitwise-style SLOs, "
            "not hard real-time guarantees.",
            "TTFT calibration has three measured requests per context.",
            "Legacy BOS reconstruction calibration must not be reported as "
            "direct choice.token_ids measurement.",
            "P/D partition feasibility and overlap cost are measured separately.",
        ],
        "source_baselines": source_rows,
        "source_results_dirs": [row["results_dir"] for row in source_rows],
        "source_case_ids": [row["case_id"] for row in source_rows],
        "model": "meta-llama/Llama-3.2-1B-Instruct",
    }
    return policy, policy_rows


def build_markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Native 40SM one-token performance and SLO",
        "",
        "TBT is the pooled distribution of individual generated-token gaps. "
        "The first output token is excluded as TTFT.",
        "",
        "| Context | Q | TTFT profile (ms) | TTFT SLO (ms) | "
        "1-token TBT profile (ms) | TBT SLO (ms) | "
        "E2E profile (ms) | E2E SLO (ms) |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{int(row['context_tokens']) // 1000}K | "
            f"{row['quantile']} | "
            f"{float(row['profiling_ttft_ms']):.3f} | "
            f"{float(row['ttft_slo_ms']):.3f} | "
            f"{float(row['profiling_one_token_tbt_ms']):.4f} | "
            f"{float(row['tbt_slo_ms']):.4f} | "
            f"{float(row['profiling_e2e_ms']):.3f} | "
            f"{float(row['e2e_slo_ms']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## SLO multipliers",
            "",
            "| Metric | P50 | P90 | P99 |",
            "|---|---:|---:|---:|",
            "| TTFT | 2x | 3x | 6x |",
            "| TBT | 1.25x | 1.5x | 5x |",
            "| E2E | 1.25x | 1.5x | 5x |",
            "",
            "## Provenance",
            "",
            "`choice_token_ids` is a direct exact-ID measurement. "
            "`legacy_archive_bos_reconstruction` removes one tokenizer-added "
            "BOS per SSE fragment after validating every request's fixed "
            "output-token total. Do not label the latter as a direct exact-ID "
            "measurement.",
            "",
            "## Runtime policy",
            "",
            "`request_slo_policy.json` selects the quantile-matched P50 "
            "contract: TTFT = native P50 × 2 and one-token TBT = native "
            "P50 × 1.25. Runtime TBT remains measured directly from "
            "`choice.token_ids`; the calibration provenance is recorded "
            "separately.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed native_40sm_tbt_sm_scaling_*_results directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = [load_context_samples(path) for path in args.results_dir]
    contexts = [sample.context_tokens for sample in samples]
    if len(contexts) != len(set(contexts)):
        raise ValueError(f"duplicate context buckets: {contexts}")

    result_rows = build_result_rows(samples)
    source_rows = [
        {
            "context_tokens": sample.context_tokens,
            "results_dir": str(sample.results_dir),
            "case_id": sample.case_id,
            "provenance": sample.provenance,
            "measurement_request_count": len(sample.ttft_ms),
            "one_token_tbt_gap_count": len(sample.one_token_tbt_ms),
            "requested_output_tokens": sample.requested_output_tokens,
        }
        for sample in sorted(samples, key=lambda item: item.context_tokens)
    ]
    request_rows = [
        row
        for sample in sorted(samples, key=lambda item: item.context_tokens)
        for row in sample.request_rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "native_40sm_1token_performance_slo.csv",
        RESULT_FIELDS,
        result_rows,
    )
    write_csv(
        args.output_dir / "source_provenance.csv",
        SOURCE_FIELDS,
        source_rows,
    )
    write_csv(
        args.output_dir / "request_samples.csv",
        REQUEST_FIELDS,
        request_rows,
    )
    runtime_policy, runtime_policy_rows = build_runtime_policy(
        result_rows, samples
    )
    write_csv(
        args.output_dir / "request_slo_policy.csv",
        RUNTIME_POLICY_FIELDS,
        runtime_policy_rows,
    )
    (args.output_dir / "request_slo_policy.json").write_text(
        json.dumps(runtime_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        build_markdown(result_rows), encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_contract": {
            "ttft": "submit to first non-empty output-token fragment",
            "tbt": (
                "pooled individual generated-token gaps; first token excluded"
            ),
            "e2e": "submit through streamed completion and [DONE]",
            "quantiles": ["P50", "P90", "P99"],
            "splitwise_factors": SPLITWISE_FACTORS,
        },
        "sources": source_rows,
        "limits": [
            "TTFT and E2E quantiles have only three measured requests per context.",
            "Legacy BOS reconstruction is not a direct choice.token_ids measurement.",
        ],
        "runtime_policy": {
            "path": "request_slo_policy.json",
            "quantile": "P50",
            "ttft_factor": SPLITWISE_FACTORS[50]["ttft"],
            "tbt_factor": SPLITWISE_FACTORS[50]["tbt"],
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "native_40sm_1token_performance_slo.csv")
    print(args.output_dir / "request_slo_policy.json")
    print(args.output_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
