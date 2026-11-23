#!/usr/bin/env python3
"""Generate Splitwise-aligned T4 SLO and partition-evaluation artifacts.

The reference is a native, full-40-SM, no-contention Tesla T4 run.  Following
Splitwise Table VI, TTFT and pooled token-gap TBT are evaluated at P50, P90,
and P99 with quantile-specific slowdown factors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-zoneatten")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.analyze_vendor_v8_partition_profiles import (  # noqa: E402
    iteration_uses_ring,
    phase_local_prefill_interval,
    read_jsonl,
)
from tools.generate_prior_work_slo_comparison import (  # noqa: E402
    SPLITWISE_FACTORS,
    load_native_samples,
    percentile,
)


QUANTILES = (50, 90, 99)
PROFILE_BAR_COLOR = "#D1D5DB"
SLO_BAR_COLOR = "#4B5563"
BAR_EDGE_COLOR = "#6B7280"
REFERENCE_FIELDS = (
    "context_tokens",
    "quantile",
    "ttft_sample_count",
    "tbt_token_gap_count",
    "e2e_sample_count",
    "native_ttft_ms",
    "ttft_slo_factor",
    "splitwise_ttft_slo_ms",
    "native_pooled_token_gap_tbt_ms",
    "tbt_slo_factor",
    "splitwise_pooled_token_gap_tbt_slo_ms",
    "native_e2e_ms",
    "e2e_slo_factor",
    "splitwise_e2e_slo_ms",
    "provenance",
    "reference_hardware",
    "reference_execution",
)
PARTITION_FIELDS = (
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "context_tokens",
    "active_decode_requests",
    "quantile",
    "measurement_trial_count",
    "overlap_token_gap_count",
    "partition_ttft_ms",
    "native_ttft_ms",
    "ttft_normalized_slowdown",
    "ttft_slo_factor",
    "splitwise_ttft_slo_ms",
    "ttft_slo_met",
    "partition_overlap_tbt_ms",
    "native_pooled_token_gap_tbt_ms",
    "tbt_normalized_slowdown",
    "tbt_slo_factor",
    "splitwise_pooled_token_gap_tbt_slo_ms",
    "tbt_slo_met",
)
SUMMARY_FIELDS = (
    "metric",
    "quantile",
    "completed_partition_cases",
    "expected_partition_cases",
    "evaluated_state_count",
    "slo_met_state_count",
    "state_slo_attainment",
    "mean_normalized_slowdown",
    "max_normalized_slowdown",
    "splitwise_slo_factor",
    "coverage_status",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_reference_rows(
    ttft: dict[int, list[float]],
    pooled_token_tbt: dict[int, list[float]],
    e2e: dict[int, list[float]],
) -> list[dict[str, Any]]:
    if not (set(ttft) == set(pooled_token_tbt) == set(e2e)):
        raise ValueError("TTFT, pooled TBT, and E2E context buckets must match")
    rows: list[dict[str, Any]] = []
    for context in sorted(ttft):
        for quantile in QUANTILES:
            factors = SPLITWISE_FACTORS[quantile]
            ttft_value = percentile(ttft[context], quantile)
            tbt_value = percentile(pooled_token_tbt[context], quantile)
            e2e_value = percentile(e2e[context], quantile)
            rows.append(
                {
                    "context_tokens": context,
                    "quantile": f"P{quantile}",
                    "ttft_sample_count": len(ttft[context]),
                    "tbt_token_gap_count": len(
                        pooled_token_tbt[context]
                    ),
                    "e2e_sample_count": len(e2e[context]),
                    "native_ttft_ms": round(ttft_value, 6),
                    "ttft_slo_factor": factors["ttft"],
                    "splitwise_ttft_slo_ms": round(
                        ttft_value * factors["ttft"], 6
                    ),
                    "native_pooled_token_gap_tbt_ms": round(
                        tbt_value, 6
                    ),
                    "tbt_slo_factor": factors["tbt"],
                    "splitwise_pooled_token_gap_tbt_slo_ms": round(
                        tbt_value * factors["tbt"], 6
                    ),
                    "native_e2e_ms": round(e2e_value, 6),
                    "e2e_slo_factor": factors["e2e"],
                    "splitwise_e2e_slo_ms": round(
                        e2e_value * factors["e2e"], 6
                    ),
                    "provenance": "baseline_results_dir",
                    "reference_hardware": "Tesla T4, 40 SM",
                    "reference_execution": (
                        "native vLLM, one request, no contention"
                    ),
                }
            )
    return rows


def load_precomputed_reference_rows(path: Path) -> list[dict[str, Any]]:
    """Load the audited one-token profile/SLO CSV without re-aggregation."""

    required = {
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
    }
    raw_rows = read_csv(path)
    if not raw_rows:
        raise ValueError(f"precomputed reference CSV is empty: {path}")
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(
            f"precomputed reference CSV is missing columns: {missing}"
        )
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        quantile = str(raw["quantile"])
        if quantile not in {f"P{value}" for value in QUANTILES}:
            raise ValueError(f"unsupported quantile {quantile!r} in {path}")
        rows.append(
            {
                "context_tokens": int(raw["context_tokens"]),
                "quantile": quantile,
                "ttft_sample_count": int(raw["measurement_request_count"]),
                "tbt_token_gap_count": int(raw["one_token_tbt_gap_count"]),
                "e2e_sample_count": int(raw["measurement_request_count"]),
                "native_ttft_ms": float(raw["profiling_ttft_ms"]),
                "ttft_slo_factor": float(raw["ttft_slo_factor"]),
                "splitwise_ttft_slo_ms": float(raw["ttft_slo_ms"]),
                "native_pooled_token_gap_tbt_ms": float(
                    raw["profiling_one_token_tbt_ms"]
                ),
                "tbt_slo_factor": float(raw["tbt_slo_factor"]),
                "splitwise_pooled_token_gap_tbt_slo_ms": float(
                    raw["tbt_slo_ms"]
                ),
                "native_e2e_ms": float(raw["profiling_e2e_ms"]),
                "e2e_slo_factor": float(raw["e2e_slo_factor"]),
                "splitwise_e2e_slo_ms": float(raw["e2e_slo_ms"]),
                "provenance": raw["provenance"],
                "reference_hardware": "Tesla T4, 40 SM",
                "reference_execution": (
                    "native vLLM, one request, no contention"
                ),
            }
        )
    expected = {
        (context, f"P{quantile}")
        for context in {int(row["context_tokens"]) for row in rows}
        for quantile in QUANTILES
    }
    actual = {
        (int(row["context_tokens"]), str(row["quantile"])) for row in rows
    }
    if actual != expected:
        raise ValueError(
            "precomputed reference must contain one P50/P90/P99 row per "
            f"context: missing={sorted(expected - actual)}"
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["context_tokens"]),
            int(str(row["quantile"])[1:]),
        ),
    )


def collect_completed_partition_cases(
    partition_root: Path,
) -> tuple[list[Path], list[Path]]:
    completed: list[Path] = []
    incomplete: list[Path] = []
    for metadata_path in sorted(partition_root.rglob("run_metadata.json")):
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            incomplete.append(metadata_path.parent)
            continue
        if not metadata.get("profile_id") or not metadata.get("prefill_mode"):
            continue
        if metadata.get("status") == "completed":
            completed.append(metadata_path.parent)
        else:
            incomplete.append(metadata_path.parent)
    return completed, incomplete


PartitionKey = tuple[str, int, int, str, int, int]


def load_partition_samples(
    case_dirs: Sequence[Path],
    *,
    chunk_tokens: int,
) -> dict[PartitionKey, dict[str, list[float]]]:
    samples: dict[
        PartitionKey, dict[str, list[float]]
    ] = defaultdict(lambda: {"ttft": [], "tbt": []})

    for case_dir in case_dirs:
        metadata_path = case_dir / "run_metadata.json"
        trial_path = case_dir / "trial_summary.csv"
        token_path = case_dir / "client_token_timeline.csv"
        iteration_path = case_dir / "iteration_metrics.jsonl"
        for path in (
            metadata_path,
            trial_path,
            token_path,
            iteration_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        profile_id = str(metadata["profile_id"])
        decode_sms = int(metadata["decode_sms"])
        prefill_sms = int(metadata["prefill_sms"])
        mode = str(metadata["prefill_mode"])
        iterations = read_jsonl(iteration_path)
        intervals: dict[
            tuple[int, int, int], list[tuple[float, float]]
        ] = defaultdict(list)

        for trial in read_csv(trial_path):
            if (
                trial.get("phase") != "measurement"
                or trial.get("status") != "success"
            ):
                continue
            context = int(trial["context_tokens"])
            decode_count = int(trial["active_decode_requests"])
            repeat_index = int(trial["repeat_index"])
            key = (
                profile_id,
                decode_sms,
                prefill_sms,
                mode,
                context,
                decode_count,
            )
            samples[key]["ttft"].append(float(trial["probe_ttft_ms"]))

            probe_start = float(trial["probe_submit_monotonic_ts"])
            probe_end = float(trial["probe_first_token_monotonic_ts"])
            for iteration in iterations:
                scheduled = int(
                    iteration.get("num_prefill_tokens", 0) or 0
                )
                if scheduled <= 0:
                    continue
                if scheduled > chunk_tokens:
                    raise ValueError(
                        f"{case_dir}: prefill chunk {scheduled} exceeds "
                        f"{chunk_tokens}"
                    )
                iteration_start = float(
                    iteration.get(
                        "schedule_start_ts",
                        iteration.get("scheduled_ts", -1),
                    )
                )
                if not probe_start <= iteration_start <= probe_end:
                    continue
                if (mode == "ring") != iteration_uses_ring(iteration):
                    continue
                interval = phase_local_prefill_interval(iteration)
                if interval is not None:
                    intervals[
                        (context, decode_count, repeat_index)
                    ].append(interval)

        for token in read_csv(token_path):
            if (
                token.get("phase") != "measurement"
                or token.get("role") != "decode_anchor"
                or token.get("during_probe_prefill") != "1"
                or token.get("estimated_per_token_gap_ms") in ("", None)
            ):
                continue
            context = int(token["context_tokens"])
            decode_count = int(token["active_decode_requests"])
            repeat_index = int(token["repeat_index"])
            gap_start_raw = token.get("gap_start_monotonic_ts")
            gap_end_raw = token.get("arrival_monotonic_ts")
            if gap_start_raw in ("", None) or gap_end_raw in ("", None):
                continue
            gap_start = float(gap_start_raw)
            gap_end = float(gap_end_raw)
            if not any(
                gap_start < interval_end and gap_end > interval_start
                for interval_start, interval_end in intervals.get(
                    (context, decode_count, repeat_index), ()
                )
            ):
                continue
            key = (
                profile_id,
                decode_sms,
                prefill_sms,
                mode,
                context,
                decode_count,
            )
            gap = float(token["estimated_per_token_gap_ms"])
            samples[key]["tbt"].extend(
                [gap] * max(1, int(token.get("chunk_tokens", 1) or 1))
            )

    for key, values in samples.items():
        if not values["ttft"] or not values["tbt"]:
            raise ValueError(f"incomplete partition metric samples: {key}")
    return dict(samples)


def build_partition_rows(
    reference_rows: Sequence[dict[str, Any]],
    samples: dict[PartitionKey, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    references = {
        (int(row["context_tokens"]), str(row["quantile"])): row
        for row in reference_rows
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(samples):
        (
            profile_id,
            decode_sms,
            prefill_sms,
            mode,
            context,
            decode_count,
        ) = key
        if not all(
            (context, f"P{quantile}") in references
            for quantile in QUANTILES
        ):
            raise ValueError(
                f"partition context {context} has no native reference"
            )
        for quantile in QUANTILES:
            quantile_name = f"P{quantile}"
            reference = references[(context, quantile_name)]
            partition_ttft = percentile(
                samples[key]["ttft"], quantile
            )
            partition_tbt = percentile(samples[key]["tbt"], quantile)
            native_ttft = float(reference["native_ttft_ms"])
            native_tbt = float(
                reference["native_pooled_token_gap_tbt_ms"]
            )
            ttft_slo = float(reference["splitwise_ttft_slo_ms"])
            tbt_slo = float(
                reference["splitwise_pooled_token_gap_tbt_slo_ms"]
            )
            rows.append(
                {
                    "profile_id": profile_id,
                    "decode_sms": decode_sms,
                    "prefill_sms": prefill_sms,
                    "prefill_mode": mode,
                    "context_tokens": context,
                    "active_decode_requests": decode_count,
                    "quantile": quantile_name,
                    "measurement_trial_count": len(samples[key]["ttft"]),
                    "overlap_token_gap_count": len(samples[key]["tbt"]),
                    "partition_ttft_ms": round(partition_ttft, 6),
                    "native_ttft_ms": round(native_ttft, 6),
                    "ttft_normalized_slowdown": round(
                        partition_ttft / native_ttft, 6
                    ),
                    "ttft_slo_factor": reference["ttft_slo_factor"],
                    "splitwise_ttft_slo_ms": round(ttft_slo, 6),
                    "ttft_slo_met": int(partition_ttft <= ttft_slo),
                    "partition_overlap_tbt_ms": round(partition_tbt, 6),
                    "native_pooled_token_gap_tbt_ms": round(native_tbt, 6),
                    "tbt_normalized_slowdown": round(
                        partition_tbt / native_tbt, 6
                    ),
                    "tbt_slo_factor": reference["tbt_slo_factor"],
                    "splitwise_pooled_token_gap_tbt_slo_ms": round(
                        tbt_slo, 6
                    ),
                    "tbt_slo_met": int(partition_tbt <= tbt_slo),
                }
            )
    return rows


def build_evaluation_summary(
    partition_rows: Sequence[dict[str, Any]],
    *,
    completed_cases: int,
    expected_cases: int,
) -> list[dict[str, Any]]:
    coverage = (
        "complete"
        if completed_cases == expected_cases
        else f"incomplete:{completed_cases}/{expected_cases}"
    )
    rows: list[dict[str, Any]] = []
    for metric in ("ttft", "tbt"):
        for quantile in (f"P{value}" for value in QUANTILES):
            selected = [
                row
                for row in partition_rows
                if row["quantile"] == quantile
            ]
            slowdowns = [
                float(row[f"{metric}_normalized_slowdown"])
                for row in selected
            ]
            met = [int(row[f"{metric}_slo_met"]) for row in selected]
            factor = SPLITWISE_FACTORS[int(quantile[1:])][metric]
            rows.append(
                {
                    "metric": metric.upper(),
                    "quantile": quantile,
                    "completed_partition_cases": completed_cases,
                    "expected_partition_cases": expected_cases,
                    "evaluated_state_count": len(selected),
                    "slo_met_state_count": sum(met),
                    "state_slo_attainment": (
                        round(statistics.fmean(met), 6) if met else ""
                    ),
                    "mean_normalized_slowdown": (
                        round(statistics.fmean(slowdowns), 6)
                        if slowdowns
                        else ""
                    ),
                    "max_normalized_slowdown": (
                        round(max(slowdowns), 6) if slowdowns else ""
                    ),
                    "splitwise_slo_factor": factor,
                    "coverage_status": coverage,
                }
            )
    return rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def native_reference_row_y_limits(
    reference_rows: Sequence[dict[str, Any]],
    *,
    headroom: float = 1.08,
) -> tuple[float, float, float]:
    """Return one common limit for each TTFT, TBT, and E2E row."""
    if not reference_rows:
        raise ValueError("native reference rows must not be empty")
    if headroom <= 1.0:
        raise ValueError("native reference y-axis headroom must exceed 1")
    metric_fields = (
        ("native_ttft_ms", "splitwise_ttft_slo_ms", 1 / 1000.0),
        (
            "native_pooled_token_gap_tbt_ms",
            "splitwise_pooled_token_gap_tbt_slo_ms",
            1.0,
        ),
        ("native_e2e_ms", "splitwise_e2e_slo_ms", 1 / 1000.0),
    )
    limits: list[float] = []
    for profile_field, slo_field, scale in metric_fields:
        maximum = max(
            max(float(row[profile_field]), float(row[slo_field])) * scale
            for row in reference_rows
        )
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError(
                f"native reference metric has invalid maximum: "
                f"{profile_field}={maximum}"
            )
        limits.append(maximum * headroom)
    return limits[0], limits[1], limits[2]


def plot_native_reference(
    reference_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    configure_plot_style()
    contexts = sorted(
        {int(row["context_tokens"]) for row in reference_rows}
    )
    quantiles = tuple(f"P{value}" for value in QUANTILES)
    row_y_limits = native_reference_row_y_limits(reference_rows)
    metric_specs = (
        (
            "native_ttft_ms",
            "splitwise_ttft_slo_ms",
            1 / 1000.0,
            "TTFT (s)",
            PROFILE_BAR_COLOR,
            SLO_BAR_COLOR,
        ),
        (
            "native_pooled_token_gap_tbt_ms",
            "splitwise_pooled_token_gap_tbt_slo_ms",
            1.0,
            "TBT (ms)",
            PROFILE_BAR_COLOR,
            SLO_BAR_COLOR,
        ),
        (
            "native_e2e_ms",
            "splitwise_e2e_slo_ms",
            1 / 1000.0,
            "E2E (s)",
            PROFILE_BAR_COLOR,
            SLO_BAR_COLOR,
        ),
    )
    figure, axes = plt.subplots(
        3,
        len(contexts),
        figsize=(20, 14),
        squeeze=False,
    )
    x_values = np.arange(len(quantiles))
    width = 0.34
    for column, context in enumerate(contexts):
        selected = {
            str(row["quantile"]): row
            for row in reference_rows
            if int(row["context_tokens"]) == context
        }
        for row_index, (
            profile_field,
            slo_field,
            scale,
            ylabel,
            profile_color,
            slo_color,
        ) in enumerate(metric_specs):
            axis = axes[row_index][column]
            profile = [
                float(selected[q][profile_field]) * scale
                for q in quantiles
            ]
            slo = [
                float(selected[q][slo_field]) * scale
                for q in quantiles
            ]
            axis.bar(
                x_values - width / 2,
                profile,
                width,
                color=profile_color,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=0.5,
                label="T4 no-contention profile",
            )
            axis.bar(
                x_values + width / 2,
                slo,
                width,
                color=slo_color,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=0.5,
                label="Splitwise SLO",
            )
            axis.set_xticks(x_values, quantiles)
            metric_name = ("TTFT", "TBT", "E2E")[row_index]
            axis.set_title(f"{context // 1000}K — {metric_name}")
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, row_y_limits[row_index])
            axis.grid(axis="y", color="#CBD5E1", alpha=0.65)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column == 0:
                axis.legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Splitwise SLO on Native 40-SM Tesla T4 "
        "(P50/P90/P99)",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "TTFT factors: 2×/3×/6×; "
        "TBT factors: 1.25×/1.5×/5×. "
        "E2E factors: 1.25×/1.5×/5×. "
        "TBT is computed from pooled decode token gaps.",
        ha="center",
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_partition_slowdown(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    metric: str,
    coverage_label: str,
    dpi: int,
) -> list[Path]:
    configure_plot_style()
    outputs: list[Path] = []
    profile_modes = sorted(
        {
            (str(row["profile_id"]), str(row["prefill_mode"]))
            for row in rows
        }
    )
    for profile_id, mode in profile_modes:
        selected_profile = [
            row
            for row in rows
            if row["profile_id"] == profile_id
            and row["prefill_mode"] == mode
        ]
        states = sorted(
            {
                (
                    int(row["context_tokens"]),
                    int(row["active_decode_requests"]),
                )
                for row in selected_profile
            }
        )
        labels = [
            f"{context // 1000}K/d{decode_count}"
            for context, decode_count in states
        ]
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(18, 11),
            squeeze=False,
        )
        for row_index, quantile_value in enumerate(QUANTILES):
            quantile = f"P{quantile_value}"
            axis = axes[row_index][0]
            by_state = {
                (
                    int(row["context_tokens"]),
                    int(row["active_decode_requests"]),
                ): row
                for row in selected_profile
                if row["quantile"] == quantile
            }
            values = [
                float(by_state[state][f"{metric}_normalized_slowdown"])
                for state in states
            ]
            met = [
                bool(int(by_state[state][f"{metric}_slo_met"]))
                for state in states
            ]
            factor = SPLITWISE_FACTORS[quantile_value][metric]
            colors = [
                "#16A34A" if state_met else "#DC2626"
                for state_met in met
            ]
            x_values = np.arange(len(states))
            axis.bar(x_values, values, color=colors, width=0.7)
            axis.axhline(
                factor,
                color="#111827",
                linestyle="--",
                linewidth=1.5,
                label=f"Splitwise {quantile} limit = {factor:g}×",
            )
            axis.set_xticks(x_values, labels, rotation=35, ha="right")
            axis.set_ylabel("normalized slowdown")
            axis.set_title(f"{metric.upper()} {quantile}")
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#CBD5E1", alpha=0.65)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(frameon=False, loc="upper left")
        figure.suptitle(
            f"{profile_id} / {mode}: {metric.upper()} slowdown "
            f"vs native 40-SM T4 ({coverage_label})",
            fontsize=15,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.01,
            "State label = context / active decode requests; "
            "green meets the quantile SLO, red violates it.",
            ha="center",
            color="#475569",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.95))
        filename = (
            f"partition_{metric}_{safe_filename(profile_id)}_"
            f"{safe_filename(mode)}_p50_p90_p99.png"
        )
        output_path = output_dir / filename
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output_path)
    return outputs


def markdown_table(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "|" + "|".join("---" for _ in fields) + "|"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def write_report(
    output_dir: Path,
    reference_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    *,
    reference_source: Path,
    partition_root: Path | None,
    completed_cases: int,
    expected_cases: int,
    incomplete_dirs: Sequence[Path],
    partition_graphs: Sequence[Path],
) -> None:
    reference_view = [
        {
            "ctx": f"{int(row['context_tokens']) // 1000}K",
            "q": row["quantile"],
            "TTFT profile (s)": round(
                float(row["native_ttft_ms"]) / 1000.0, 3
            ),
            "TTFT SLO (s)": round(
                float(row["splitwise_ttft_slo_ms"]) / 1000.0, 3
            ),
            "TBT profile (ms)": row[
                "native_pooled_token_gap_tbt_ms"
            ],
            "TBT SLO (ms)": row[
                "splitwise_pooled_token_gap_tbt_slo_ms"
            ],
            "E2E profile (s)": round(
                float(row["native_e2e_ms"]) / 1000.0, 3
            ),
            "E2E SLO (s)": round(
                float(row["splitwise_e2e_slo_ms"]) / 1000.0, 3
            ),
        }
        for row in reference_rows
    ]
    coverage = (
        "complete"
        if completed_cases == expected_cases
        else f"INCOMPLETE ({completed_cases}/{expected_cases})"
    )
    report = f"""# Splitwise-aligned Tesla T4 SLO Evaluation

## Reference contract

- Source: `{reference_source}`
- Hardware: Tesla T4, all 40 SMs
- Execution: native vLLM, one request at a time, no P/D contention
- Quantiles: P50, P90, P99
- TTFT slowdown factors: 2x, 3x, 6x
- TBT slowdown factors: 1.25x, 1.5x, 5x
- E2E slowdown factors: 1.25x, 1.5x, 5x
- TBT boundary: pooled decode token gaps, weighted by chunk token count

{markdown_table(reference_view, tuple(reference_view[0]))}

## Partition evaluation

- Source: `{partition_root if partition_root is not None else "not supplied"}`
- Coverage: **{coverage}**
- A state passes when its measured quantile is no greater than the fixed
  context-matched Splitwise SLO derived from the native 40-SM reference.
- Partition TTFT uses measured Prefill-probe TTFT trials.
- Partition TBT uses decode token gaps that intersect an actual phase-local
  Prefill worker interval.

{markdown_table(summary_rows, SUMMARY_FIELDS)}

## Limitations

- Each native context has the sample count recorded in the reference CSV.
  P90/P99 TTFT and E2E values are
  interpolated descriptive estimates, not production-grade tail guarantees.
- Token-level TBT has many observations, but request-level independence is
  bounded by the recorded measurement request count.
- Missing partition cases are not estimated or treated as passing.
- SLO attainment across profiled states is diagnostic coverage, not
  request-weighted production SLO attainment.

## Files

- `splitwise_t4_reference_slo.csv`
- `splitwise_partition_evaluation.csv`
- `splitwise_partition_evaluation_summary.csv`
- `native40_splitwise_ttft_tbt_e2e_p50_p90_p99.png`
"""
    if partition_graphs:
        report += "\n".join(
            f"- `{path.name}`" for path in partition_graphs
        )
        report += "\n"
    if incomplete_dirs:
        report += "\n## Incomplete case directories\n\n"
        report += "\n".join(f"- `{path}`" for path in incomplete_dirs)
        report += "\n"
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Splitwise P50/P90/P99 SLOs from a native T4 "
            "profile and optionally evaluate partition profiles"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline-results-dir", type=Path)
    source.add_argument("--one-token-profile-slo-csv", type=Path)
    parser.add_argument("--partition-results-dir", type=Path)
    parser.add_argument("--expected-partition-cases", type=int, default=10)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_partition_cases < 0:
        raise ValueError("--expected-partition-cases must be non-negative")
    if args.prefill_chunk_tokens <= 0:
        raise ValueError("--prefill-chunk-tokens must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.one_token_profile_slo_csv is not None:
        reference_source = args.one_token_profile_slo_csv.expanduser().resolve()
        reference_rows = load_precomputed_reference_rows(reference_source)
        baseline_dir = None
    else:
        baseline_dir = args.baseline_results_dir.expanduser().resolve()
        reference_source = baseline_dir
        ttft, _, pooled_token_tbt, e2e = load_native_samples(baseline_dir)
        reference_rows = build_reference_rows(ttft, pooled_token_tbt, e2e)
    write_csv(
        output_dir / "splitwise_t4_reference_slo.csv",
        reference_rows,
        REFERENCE_FIELDS,
    )
    plot_native_reference(
        reference_rows,
        output_dir / "native40_splitwise_ttft_tbt_e2e_p50_p90_p99.png",
        dpi=args.dpi,
    )

    partition_root: Path | None = None
    completed_case_dirs: list[Path] = []
    incomplete_case_dirs: list[Path] = []
    partition_rows: list[dict[str, Any]] = []
    if args.partition_results_dir is not None:
        partition_root = args.partition_results_dir.expanduser().resolve()
        completed_case_dirs, incomplete_case_dirs = (
            collect_completed_partition_cases(partition_root)
        )
        samples = load_partition_samples(
            completed_case_dirs,
            chunk_tokens=args.prefill_chunk_tokens,
        )
        partition_rows = build_partition_rows(reference_rows, samples)

    summary_rows = build_evaluation_summary(
        partition_rows,
        completed_cases=len(completed_case_dirs),
        expected_cases=args.expected_partition_cases,
    )
    write_csv(
        output_dir / "splitwise_partition_evaluation.csv",
        partition_rows,
        PARTITION_FIELDS,
    )
    write_csv(
        output_dir / "splitwise_partition_evaluation_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )

    coverage_label = (
        "complete"
        if len(completed_case_dirs) == args.expected_partition_cases
        else (
            f"INCOMPLETE {len(completed_case_dirs)}/"
            f"{args.expected_partition_cases}"
        )
    )
    partition_graphs: list[Path] = []
    if partition_rows:
        partition_graphs.extend(
            plot_partition_slowdown(
                partition_rows,
                output_dir,
                metric="ttft",
                coverage_label=coverage_label,
                dpi=args.dpi,
            )
        )
        partition_graphs.extend(
            plot_partition_slowdown(
                partition_rows,
                output_dir,
                metric="tbt",
                coverage_label=coverage_label,
                dpi=args.dpi,
            )
        )

    (output_dir / "splitwise_slo_evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "Splitwise Table VI normalized slowdown",
                "baseline_results_dir": (
                    str(baseline_dir) if baseline_dir is not None else None
                ),
                "one_token_profile_slo_csv": (
                    str(reference_source)
                    if args.one_token_profile_slo_csv is not None
                    else None
                ),
                "partition_results_dir": (
                    str(partition_root)
                    if partition_root is not None
                    else None
                ),
                "completed_partition_cases": len(completed_case_dirs),
                "expected_partition_cases": (
                    args.expected_partition_cases
                ),
                "reference_rows": reference_rows,
                "partition_rows": partition_rows,
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir,
        reference_rows,
        summary_rows,
        reference_source=reference_source,
        partition_root=partition_root,
        completed_cases=len(completed_case_dirs),
        expected_cases=args.expected_partition_cases,
        incomplete_dirs=incomplete_case_dirs,
        partition_graphs=partition_graphs,
    )
    print(
        "[DONE] "
        f"reference_rows={len(reference_rows)} "
        f"partition_rows={len(partition_rows)} "
        f"partition_cases={len(completed_case_dirs)}/"
        f"{args.expected_partition_cases}"
    )
    print(f"[DONE] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
