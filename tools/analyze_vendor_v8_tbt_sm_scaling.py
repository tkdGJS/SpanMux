#!/usr/bin/env python3
"""Test the Decode-SM scaling and Prefill-overlap P50 TBT hypothesis.

The experiment uses paired windows from each partition trial:

* Decode-only control: the last fixed number of full token gaps immediately
  before the Prefill probe is submitted.
* Overlap: full token gaps after probe submission and before its first token.

Ratios are normalized to an exact-token-id native 40-SM baseline. Confidence
intervals use request/repeat clusters, rather than treating correlated token
gaps as independent observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-zoneatten")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_FIELDS = (
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "decode_context_tokens",
    "probe_context_tokens",
    "active_decode_requests",
    "cluster_unit",
    "independent_request_clusters",
    "quantile",
    "baseline_repeat_count",
    "partition_repeat_count",
    "baseline_gap_count",
    "control_gap_count",
    "overlap_gap_count",
    "baseline_tbt_ms",
    "p50_slo_factor",
    "p50_slo_ms",
    "ideal_decode_only_tbt_ms",
    "predicted_overlap_tbt_ms",
    "measured_control_tbt_ms",
    "measured_control_ratio",
    "measured_control_ratio_ci95_low",
    "measured_control_ratio_ci95_high",
    "measured_overlap_tbt_ms",
    "measured_overlap_ratio",
    "measured_overlap_ratio_ci95_low",
    "measured_overlap_ratio_ci95_high",
    "measured_overlap_over_control",
    "measured_overlap_over_control_ci95_low",
    "measured_overlap_over_control_ci95_high",
    "point_slo_met",
    "confidence_slo_verdict",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
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


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def expand_gap(row: dict[str, str]) -> list[float]:
    raw = row.get("estimated_per_token_gap_ms")
    if raw in (None, ""):
        return []
    count = max(1, int(row.get("chunk_tokens") or 1))
    return [float(raw)] * count


def require_exact_token_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(f"experiment is not completed: {path.parent}")
    if metadata.get("tbt_token_count_source") != "choice.token_ids":
        raise ValueError(
            f"exact choice.token_ids measurement required: {path.parent}"
        )
    if metadata.get("tbt_requires_return_token_ids") is not True:
        raise ValueError(
            f"return_token_ids contract is absent: {path.parent}"
        )
    return metadata


def load_baseline_clusters(
    baseline_dir: Path,
) -> tuple[dict[int, dict[int, list[float]]], dict[int, str]]:
    metadata = require_exact_token_metadata(
        baseline_dir / "run_metadata.json"
    )
    prompt_hashes = {
        int(context): str(digest)
        for context, digest in (
            metadata.get("prompt_sha256_by_context") or {}
        ).items()
    }
    if not prompt_hashes:
        raise ValueError("baseline prompt fingerprints are required")
    clusters: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in read_csv(baseline_dir / "token_timeline.csv"):
        if row.get("phase") != "measurement":
            continue
        gaps = expand_gap(row)
        if not gaps:
            continue
        context = int(row["context_target_tokens"])
        repeat = int(row["repeat_index"])
        clusters[context][repeat].extend(gaps)
    return (
        {
            context: dict(by_repeat)
            for context, by_repeat in clusters.items()
        },
        prompt_hashes,
    )


PartitionKey = tuple[str, int, int, str, int, int, int]


def load_partition_clusters(
    partition_root: Path,
    *,
    baseline_prompt_hashes: dict[int, str],
    allow_window_clusters: bool,
) -> tuple[
    dict[PartitionKey, dict[str, dict[int, list[float]]]],
    dict[PartitionKey, tuple[str, bool]],
]:
    result: dict[
        PartitionKey, dict[str, dict[int, list[float]]]
    ] = defaultdict(
        lambda: {
            "control": defaultdict(list),
            "overlap": defaultdict(list),
        }
    )
    cluster_info: dict[PartitionKey, tuple[str, bool]] = {}
    for metadata_path in sorted(partition_root.rglob("run_metadata.json")):
        metadata = require_exact_token_metadata(metadata_path)
        if int(metadata.get("decode_only_control_gaps_per_anchor") or 0) <= 0:
            continue
        cluster_unit = str(
            metadata.get("cluster_unit") or "independent_request_repeat"
        )
        independent_clusters = bool(
            metadata.get("independent_request_clusters", True)
        )
        if not independent_clusters and not allow_window_clusters:
            raise ValueError(
                "non-independent probe-window clusters require "
                f"--allow-window-clusters: {metadata_path.parent}"
            )
        case_prompt_hashes = {
            int(context): str(digest)
            for context, digest in (
                metadata.get("prompt_sha256_by_context") or {}
            ).items()
        }
        for context in metadata.get("contexts") or []:
            context = int(context)
            expected = baseline_prompt_hashes.get(context)
            observed = case_prompt_hashes.get(context)
            if expected is None or observed != expected:
                raise ValueError(
                    "baseline/partition prompt mismatch: "
                    f"context={context} baseline={expected} "
                    f"partition={observed} case={metadata_path.parent}"
                )
        case_dir = metadata_path.parent
        trial_rows = {
            (
                int(row["context_tokens"]),
                int(row["active_decode_requests"]),
                int(row["repeat_index"]),
            ): row
            for row in read_csv(case_dir / "trial_summary.csv")
            if row.get("phase") == "measurement"
            and row.get("status") == "success"
        }
        profile_id = str(metadata["profile_id"])
        decode_sms = int(metadata["decode_sms"])
        prefill_sms = int(metadata["prefill_sms"])
        mode = str(metadata["prefill_mode"])
        for row in read_csv(case_dir / "client_token_timeline.csv"):
            if row.get("phase") != "measurement":
                continue
            if row.get("role") != "decode_anchor":
                continue
            gaps = expand_gap(row)
            if not gaps:
                continue
            context = int(row["context_tokens"])
            decode_count = int(row["active_decode_requests"])
            repeat = int(row["repeat_index"])
            trial = trial_rows.get((context, decode_count, repeat))
            if trial is None:
                continue
            probe_context = int(
                trial.get("probe_context_tokens") or context
            )
            key: PartitionKey = (
                profile_id,
                decode_sms,
                prefill_sms,
                mode,
                context,
                probe_context,
                decode_count,
            )
            observed_info = (cluster_unit, independent_clusters)
            if key in cluster_info and cluster_info[key] != observed_info:
                raise ValueError(f"inconsistent cluster metadata for {key}")
            cluster_info[key] = observed_info
            if row.get("before_probe_control") == "1":
                result[key]["control"][repeat].extend(gaps)
            if row.get("during_probe_prefill") == "1":
                result[key]["overlap"][repeat].extend(gaps)
    normalized: dict[
        PartitionKey, dict[str, dict[int, list[float]]]
    ] = {}
    for key, conditions in result.items():
        control = dict(conditions["control"])
        overlap = dict(conditions["overlap"])
        common = sorted(set(control) & set(overlap))
        if common:
            normalized[key] = {
                "control": {repeat: control[repeat] for repeat in common},
                "overlap": {repeat: overlap[repeat] for repeat in common},
            }
    return normalized, {key: cluster_info[key] for key in normalized}


def pooled(clusters: dict[int, list[float]]) -> list[float]:
    return [value for repeat in sorted(clusters) for value in clusters[repeat]]


def stable_seed(base_seed: int, key: object) -> int:
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return base_seed ^ int.from_bytes(digest[:8], "little")


def bootstrap_ratio_ci(
    numerator: dict[int, list[float]],
    denominator: dict[int, list[float]],
    *,
    quantile: float,
    iterations: int,
    seed: int,
    paired: bool,
) -> tuple[float, float]:
    numerator_keys = sorted(numerator)
    denominator_keys = sorted(denominator)
    if not numerator_keys or not denominator_keys:
        raise ValueError("bootstrap requires non-empty clusters")
    if paired and numerator_keys != denominator_keys:
        raise ValueError("paired bootstrap requires identical repeat keys")
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(iterations):
        if paired:
            selected = rng.choices(numerator_keys, k=len(numerator_keys))
            numerator_values = [
                value for key in selected for value in numerator[key]
            ]
            denominator_values = [
                value for key in selected for value in denominator[key]
            ]
        else:
            selected_numerator = rng.choices(
                numerator_keys, k=len(numerator_keys)
            )
            selected_denominator = rng.choices(
                denominator_keys, k=len(denominator_keys)
            )
            numerator_values = [
                value
                for key in selected_numerator
                for value in numerator[key]
            ]
            denominator_values = [
                value
                for key in selected_denominator
                for value in denominator[key]
            ]
        denominator_value = percentile(denominator_values, quantile)
        if denominator_value <= 0:
            raise ValueError("bootstrap denominator must be positive")
        ratios.append(
            percentile(numerator_values, quantile) / denominator_value
        )
    return percentile(ratios, 2.5), percentile(ratios, 97.5)


def confidence_verdict(low: float, high: float, slo_factor: float) -> str:
    if high <= slo_factor:
        return "pass_95ci"
    if low > slo_factor:
        return "fail_95ci"
    return "inconclusive_95ci"


def build_summary(
    baseline: dict[int, dict[int, list[float]]],
    partitions: dict[PartitionKey, dict[str, dict[int, list[float]]]],
    cluster_info: dict[PartitionKey, tuple[str, bool]],
    *,
    total_sms: int,
    quantile: float,
    slo_factor: float,
    assumed_overlap_overhead: float,
    bootstrap_iterations: int,
    seed: int,
    min_repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(partitions):
        (
            profile_id,
            decode_sms,
            prefill_sms,
            mode,
            context,
            probe_context,
            decode_count,
        ) = key
        baseline_clusters = baseline.get(context)
        if not baseline_clusters:
            raise ValueError(f"40-SM baseline is missing context {context}")
        control = partitions[key]["control"]
        overlap = partitions[key]["overlap"]
        cluster_unit, independent_clusters = cluster_info[key]
        if len(baseline_clusters) < min_repeats or len(control) < min_repeats:
            raise ValueError(
                f"insufficient repeat clusters for {key}: "
                f"baseline={len(baseline_clusters)} partition={len(control)} "
                f"required={min_repeats}"
            )
        baseline_value = percentile(pooled(baseline_clusters), quantile)
        control_value = percentile(pooled(control), quantile)
        overlap_value = percentile(pooled(overlap), quantile)
        control_ratio = control_value / baseline_value
        overlap_ratio = overlap_value / baseline_value
        overlap_over_control = overlap_value / control_value
        control_ci = bootstrap_ratio_ci(
            control,
            baseline_clusters,
            quantile=quantile,
            iterations=bootstrap_iterations,
            seed=stable_seed(seed, (key, "control")),
            paired=False,
        )
        overlap_ci = bootstrap_ratio_ci(
            overlap,
            baseline_clusters,
            quantile=quantile,
            iterations=bootstrap_iterations,
            seed=stable_seed(seed, (key, "overlap")),
            paired=False,
        )
        overhead_ci = bootstrap_ratio_ci(
            overlap,
            control,
            quantile=quantile,
            iterations=bootstrap_iterations,
            seed=stable_seed(seed, (key, "paired-overhead")),
            paired=True,
        )
        ideal_decode_only = baseline_value * total_sms / decode_sms
        predicted_overlap = ideal_decode_only * (
            1.0 + assumed_overlap_overhead
        )
        rows.append(
            {
                "profile_id": profile_id,
                "decode_sms": decode_sms,
                "prefill_sms": prefill_sms,
                "prefill_mode": mode,
                "decode_context_tokens": context,
                "probe_context_tokens": probe_context,
                "active_decode_requests": decode_count,
                "cluster_unit": cluster_unit,
                "independent_request_clusters": int(independent_clusters),
                "quantile": f"P{quantile:g}",
                "baseline_repeat_count": len(baseline_clusters),
                "partition_repeat_count": len(control),
                "baseline_gap_count": len(pooled(baseline_clusters)),
                "control_gap_count": len(pooled(control)),
                "overlap_gap_count": len(pooled(overlap)),
                "baseline_tbt_ms": round(baseline_value, 6),
                "p50_slo_factor": slo_factor,
                "p50_slo_ms": round(baseline_value * slo_factor, 6),
                "ideal_decode_only_tbt_ms": round(ideal_decode_only, 6),
                "predicted_overlap_tbt_ms": round(predicted_overlap, 6),
                "measured_control_tbt_ms": round(control_value, 6),
                "measured_control_ratio": round(control_ratio, 6),
                "measured_control_ratio_ci95_low": round(control_ci[0], 6),
                "measured_control_ratio_ci95_high": round(control_ci[1], 6),
                "measured_overlap_tbt_ms": round(overlap_value, 6),
                "measured_overlap_ratio": round(overlap_ratio, 6),
                "measured_overlap_ratio_ci95_low": round(overlap_ci[0], 6),
                "measured_overlap_ratio_ci95_high": round(overlap_ci[1], 6),
                "measured_overlap_over_control": round(
                    overlap_over_control, 6
                ),
                "measured_overlap_over_control_ci95_low": round(
                    overhead_ci[0], 6
                ),
                "measured_overlap_over_control_ci95_high": round(
                    overhead_ci[1], 6
                ),
                "point_slo_met": int(overlap_ratio <= slo_factor),
                "confidence_slo_verdict": (
                    confidence_verdict(
                        overlap_ci[0], overlap_ci[1], slo_factor
                    )
                    if independent_clusters
                    else "descriptive_only_window_clusters"
                ),
            }
        )
    return rows


def plot_summary(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    total_sms: int,
    slo_factor: float,
    assumed_overlap_overhead: float,
    dpi: int,
) -> None:
    modes = sorted({str(row["prefill_mode"]) for row in rows})
    figure, axes = plt.subplots(
        1,
        len(modes),
        figsize=(7.2 * len(modes), 5.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, mode in zip(axes[0], modes):
        selected = sorted(
            (row for row in rows if row["prefill_mode"] == mode),
            key=lambda row: int(row["decode_sms"]),
        )
        xs = [int(row["decode_sms"]) for row in selected]
        ideal = [total_sms / value for value in xs]
        predicted = [
            total_sms / value * (1.0 + assumed_overlap_overhead)
            for value in xs
        ]
        control = [float(row["measured_control_ratio"]) for row in selected]
        overlap = [float(row["measured_overlap_ratio"]) for row in selected]
        control_error = [
            [
                value - float(row["measured_control_ratio_ci95_low"])
                for value, row in zip(control, selected)
            ],
            [
                float(row["measured_control_ratio_ci95_high"]) - value
                for value, row in zip(control, selected)
            ],
        ]
        overlap_error = [
            [
                value - float(row["measured_overlap_ratio_ci95_low"])
                for value, row in zip(overlap, selected)
            ],
            [
                float(row["measured_overlap_ratio_ci95_high"]) - value
                for value, row in zip(overlap, selected)
            ],
        ]
        axis.plot(xs, ideal, "--", color="#9CA3AF", label="Ideal Decode-only")
        axis.plot(
            xs,
            predicted,
            ":",
            color="#6B7280",
            label=f"Ideal + {assumed_overlap_overhead:.0%}",
        )
        axis.errorbar(
            xs,
            control,
            yerr=control_error,
            marker="o",
            capsize=3,
            color="#4C78A8",
            label="Measured Decode-only control",
        )
        axis.errorbar(
            xs,
            overlap,
            yerr=overlap_error,
            marker="s",
            capsize=3,
            color="#F28E2B",
            label="Measured Prefill overlap",
        )
        axis.axhline(
            slo_factor,
            color="#111827",
            linewidth=1.3,
            label=f"P50 SLO ({slo_factor:g}x)",
        )
        axis.set_title(mode)
        axis.set_xlabel("Decode SMs")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("P50 TBT / native 40-SM P50 TBT")
    axes[0][-1].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired Decode-only/Prefill-overlap SM scaling"
    )
    parser.add_argument("--baseline-results-dir", type=Path, required=True)
    parser.add_argument("--partition-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-sms", type=int, default=40)
    parser.add_argument("--quantile", type=float, default=50.0)
    parser.add_argument("--slo-factor", type=float, default=1.25)
    parser.add_argument("--assumed-overlap-overhead", type=float, default=0.10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--allow-window-clusters", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total_sms <= 0 or args.bootstrap_iterations <= 0:
        raise ValueError("SM count and bootstrap iterations must be positive")
    if not 0 < args.quantile < 100:
        raise ValueError("quantile must be between zero and 100")
    if args.slo_factor <= 0 or args.assumed_overlap_overhead < 0:
        raise ValueError("SLO factor must be positive and overhead non-negative")
    if args.min_repeats <= 0 or args.dpi <= 0:
        raise ValueError("minimum repeats and dpi must be positive")
    baseline_dir = args.baseline_results_dir.expanduser().resolve()
    partition_dir = args.partition_results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    baseline, baseline_prompt_hashes = load_baseline_clusters(baseline_dir)
    partitions, cluster_info = load_partition_clusters(
        partition_dir,
        baseline_prompt_hashes=baseline_prompt_hashes,
        allow_window_clusters=args.allow_window_clusters,
    )
    if not partitions:
        raise ValueError(
            "no completed paired-control partition cases were found"
        )
    rows = build_summary(
        baseline,
        partitions,
        cluster_info,
        total_sms=args.total_sms,
        quantile=args.quantile,
        slo_factor=args.slo_factor,
        assumed_overlap_overhead=args.assumed_overlap_overhead,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
        min_repeats=args.min_repeats,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tbt_sm_scaling_summary.csv", rows, SUMMARY_FIELDS)
    plot_summary(
        output_dir / "tbt_sm_scaling_p50.png",
        rows,
        total_sms=args.total_sms,
        slo_factor=args.slo_factor,
        assumed_overlap_overhead=args.assumed_overlap_overhead,
        dpi=args.dpi,
    )
    metadata = {
        "schema_version": 1,
        "baseline_results_dir": str(baseline_dir),
        "partition_results_dir": str(partition_dir),
        "measurement_source": "choice.token_ids",
        "control_window": "last full gaps before Prefill probe submission",
        "overlap_window": "full gaps from probe submission to first token",
        "bootstrap_unit": sorted(
            {unit for unit, _independent in cluster_info.values()}
        ),
        "independent_request_clusters": all(
            independent for _unit, independent in cluster_info.values()
        ),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "slo_factor": args.slo_factor,
        "assumed_overlap_overhead": args.assumed_overlap_overhead,
        "rows": len(rows),
    }
    (output_dir / "tbt_sm_scaling_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[DONE] rows={len(rows)} "
        f"summary={output_dir / 'tbt_sm_scaling_summary.csv'}"
    )
    print(f"[DONE] graph={output_dir / 'tbt_sm_scaling_p50.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
