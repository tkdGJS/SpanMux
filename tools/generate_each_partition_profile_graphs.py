#!/usr/bin/env python3
"""Generate profiling-only quantile and repeat-mean P/D case graphs.

The figures intentionally contain no SLO thresholds, attainment colors, or
native-normalized slowdown. They show measured Prefill-probe TTFT and Decode
token gaps that overlap actual phase-local Prefill execution. Repeat-mean
figures use common TTFT and TBT y ranges across every measurable case.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
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

from tools.generate_splitwise_slo_evaluation import (
    BAR_EDGE_COLOR,
    PROFILE_BAR_COLOR,
    QUANTILES,
    PartitionKey,
    collect_completed_partition_cases,
    configure_plot_style,
)
from tools.analyze_vendor_v8_partition_profiles import (
    iteration_uses_ring,
    phase_local_prefill_interval,
)
from tools.generate_prior_work_slo_comparison import percentile


DEFAULT_PROFILES = ("d8_p32", "d16_p24", "d20_p20", "d24_p16", "d32_p8")
DEFAULT_MODES = ("native", "ring")
PROFILE_FIELDS = (
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "context_tokens",
    "active_decode_requests",
    "quantile",
    "measurement_trial_count",
    "overlap_token_gap_count",
    "profile_ttft_ms",
    "profile_overlap_tbt_ms",
)
COVERAGE_FIELDS = (
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "status",
    "case_dir",
)
MEAN_FIELDS = (
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "context_tokens",
    "active_decode_requests",
    "ttft_repeat_count",
    "tbt_repeat_count",
    "overlap_token_gap_count",
    "mean_profile_ttft_ms",
    "mean_profile_overlap_tbt_ms",
)
RepeatPartitionKey = tuple[str, int, int, str, int, int, int]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream large iteration traces instead of retaining multi-GB files."""

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if isinstance(row, dict):
                yield row


def parse_profile_id(profile_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"d(\d+)_p(\d+)", profile_id)
    if match is None:
        raise ValueError(f"invalid profile id: {profile_id}")
    return int(match.group(1)), int(match.group(2))


def build_profile_rows(
    samples: dict[PartitionKey, dict[str, list[float]]],
) -> list[dict[str, Any]]:
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
        ttft_samples = samples[key]["ttft"]
        tbt_samples = samples[key]["tbt"]
        for quantile in QUANTILES:
            rows.append(
                {
                    "profile_id": profile_id,
                    "decode_sms": decode_sms,
                    "prefill_sms": prefill_sms,
                    "prefill_mode": mode,
                    "context_tokens": context,
                    "active_decode_requests": decode_count,
                    "quantile": f"P{quantile}",
                    "measurement_trial_count": len(ttft_samples),
                    "overlap_token_gap_count": len(tbt_samples),
                    "profile_ttft_ms": round(
                        percentile(ttft_samples, quantile), 6
                    ),
                    "profile_overlap_tbt_ms": round(
                        percentile(tbt_samples, quantile), 6
                    ),
                }
            )
    return rows


def load_partition_repeat_samples(
    case_dirs: Sequence[Path],
    *,
    chunk_tokens: int,
) -> dict[RepeatPartitionKey, dict[str, list[float]]]:
    """Load TTFT and actual-Prefill-overlap TBT, retaining repeat IDs."""
    samples: dict[RepeatPartitionKey, dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "tbt": []}
    )
    for case_dir in case_dirs:
        metadata = json.loads(
            (case_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        profile_id = str(metadata["profile_id"])
        decode_sms = int(metadata["decode_sms"])
        prefill_sms = int(metadata["prefill_sms"])
        mode = str(metadata["prefill_mode"])
        intervals: dict[
            tuple[int, int, int], list[tuple[float, float]]
        ] = defaultdict(list)
        trial_windows: list[
            tuple[float, float, int, int, int]
        ] = []

        for trial in read_csv(case_dir / "trial_summary.csv"):
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
                repeat_index,
            )
            samples[key]["ttft"].append(float(trial["probe_ttft_ms"]))
            probe_start = float(trial["probe_submit_monotonic_ts"])
            probe_end = float(trial["probe_first_token_monotonic_ts"])
            trial_windows.append(
                (
                    probe_start,
                    probe_end,
                    context,
                    decode_count,
                    repeat_index,
                )
            )

        trial_windows.sort()
        window_starts = [window[0] for window in trial_windows]
        for iteration in iter_jsonl(case_dir / "iteration_metrics.jsonl"):
            scheduled = int(iteration.get("num_prefill_tokens", 0) or 0)
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
            window_index = bisect.bisect_right(
                window_starts, iteration_start
            ) - 1
            if window_index < 0:
                continue
            (
                probe_start,
                probe_end,
                context,
                decode_count,
                repeat_index,
            ) = trial_windows[window_index]
            if not probe_start <= iteration_start <= probe_end:
                continue
            if (mode == "ring") != iteration_uses_ring(iteration):
                continue
            interval = phase_local_prefill_interval(iteration)
            if interval is not None:
                intervals[(context, decode_count, repeat_index)].append(
                    interval
                )

        for token in read_csv(case_dir / "client_token_timeline.csv"):
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
                repeat_index,
            )
            gap = float(token["estimated_per_token_gap_ms"])
            samples[key]["tbt"].extend(
                [gap] * max(1, int(token.get("chunk_tokens", 1) or 1))
            )
    return samples


def flatten_repeat_samples(
    repeat_samples: dict[RepeatPartitionKey, dict[str, list[float]]],
) -> dict[PartitionKey, dict[str, list[float]]]:
    samples: dict[PartitionKey, dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "tbt": []}
    )
    for repeat_key, values in repeat_samples.items():
        key: PartitionKey = repeat_key[:-1]
        samples[key]["ttft"].extend(values["ttft"])
        samples[key]["tbt"].extend(values["tbt"])
    return samples


def build_mean_rows(
    repeat_samples: dict[RepeatPartitionKey, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """Average per-repeat metrics with every successful repeat weighted once."""
    grouped: dict[PartitionKey, dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "tbt": [], "tbt_gap_count": []}
    )
    for repeat_key, values in repeat_samples.items():
        key: PartitionKey = repeat_key[:-1]
        if values["ttft"]:
            grouped[key]["ttft"].append(statistics.fmean(values["ttft"]))
        if values["tbt"]:
            grouped[key]["tbt"].append(statistics.fmean(values["tbt"]))
            grouped[key]["tbt_gap_count"].append(len(values["tbt"]))

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        (
            profile_id,
            decode_sms,
            prefill_sms,
            mode,
            context,
            decode_count,
        ) = key
        values = grouped[key]
        if not values["ttft"] or not values["tbt"]:
            continue
        rows.append(
            {
                "profile_id": profile_id,
                "decode_sms": decode_sms,
                "prefill_sms": prefill_sms,
                "prefill_mode": mode,
                "context_tokens": context,
                "active_decode_requests": decode_count,
                "ttft_repeat_count": len(values["ttft"]),
                "tbt_repeat_count": len(values["tbt"]),
                "overlap_token_gap_count": int(
                    sum(values["tbt_gap_count"])
                ),
                "mean_profile_ttft_ms": round(
                    statistics.fmean(values["ttft"]), 6
                ),
                "mean_profile_overlap_tbt_ms": round(
                    statistics.fmean(values["tbt"]), 6
                ),
            }
        )
    return rows


def build_coverage_rows(
    partition_root: Path,
    profiles: Sequence[str],
    modes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for profile_id in profiles:
            decode_sms, prefill_sms = parse_profile_id(profile_id)
            case_dir = partition_root / "cases" / mode / profile_id
            metadata_path = case_dir / "run_metadata.json"
            status = "missing"
            if case_dir.is_dir():
                status = "directory_without_metadata"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    status = str(metadata.get("status") or "unknown")
                except (OSError, json.JSONDecodeError):
                    status = "invalid_metadata"
            rows.append(
                {
                    "profile_id": profile_id,
                    "decode_sms": decode_sms,
                    "prefill_sms": prefill_sms,
                    "prefill_mode": mode,
                    "status": status,
                    "case_dir": str(case_dir),
                }
            )
    return rows


def collect_measureable_partial_cases(
    case_dirs: Sequence[Path],
) -> list[Path]:
    """Return incomplete cases that still contain successful measurements."""
    measurable: list[Path] = []
    for case_dir in case_dirs:
        required_paths = (
            case_dir / "run_metadata.json",
            case_dir / "trial_summary.csv",
            case_dir / "client_token_timeline.csv",
            case_dir / "iteration_metrics.jsonl",
        )
        if not all(path.is_file() for path in required_paths):
            continue
        if any(
            row.get("phase") == "measurement"
            and row.get("status") == "success"
            for row in read_csv(case_dir / "trial_summary.csv")
        ):
            measurable.append(case_dir)
    return measurable


def _annotate_bars(axis: plt.Axes, bars: Any, *, scale: float) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height / scale:.1f}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )


def plot_partition_profiles(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    coverage_label: str,
    case_statuses: dict[tuple[str, str], str],
    dpi: int,
) -> list[Path]:
    configure_plot_style()
    output_paths: list[Path] = []
    cases = sorted(
        {
            (str(row["profile_id"]), str(row["prefill_mode"]))
            for row in rows
        }
    )
    for profile_id, mode in cases:
        selected = [
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
                for row in selected
            }
        )
        labels = [
            f"{context // 1000}K/d{decode_count}"
            for context, decode_count in states
        ]
        figure, axes = plt.subplots(
            2,
            len(QUANTILES),
            figsize=(20, 10),
            squeeze=False,
        )
        x_values = np.arange(len(states))
        for column, quantile_value in enumerate(QUANTILES):
            quantile = f"P{quantile_value}"
            by_state = {
                (
                    int(row["context_tokens"]),
                    int(row["active_decode_requests"]),
                ): row
                for row in selected
                if row["quantile"] == quantile
            }
            for row_index, (field, scale, ylabel) in enumerate(
                (
                    ("profile_ttft_ms", 1000.0, "Prefill-probe TTFT (s)"),
                    (
                        "profile_overlap_tbt_ms",
                        1.0,
                        "TBT (ms)",
                    ),
                )
            ):
                axis = axes[row_index][column]
                values = [
                    float(by_state[state][field]) / scale
                    for state in states
                ]
                bars = axis.bar(
                    x_values,
                    values,
                    width=0.68,
                    color=PROFILE_BAR_COLOR,
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=0.6,
                    label="Profiling measurement",
                )
                _annotate_bars(axis, bars, scale=1.0)
                axis.set_xticks(
                    x_values,
                    labels,
                    rotation=35,
                    ha="right",
                )
                axis.set_title(
                    f"{'TTFT' if row_index == 0 else 'TBT'} {quantile}"
                )
                axis.set_ylabel(ylabel)
                axis.set_ylim(bottom=0)
                axis.grid(axis="y", color="#E5E7EB", alpha=0.8)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                if row_index == 0 and column == 0:
                    axis.legend(frameon=False, loc="upper left")
        figure.suptitle(
            f"{profile_id} / {mode} "
            f"[{case_statuses.get((profile_id, mode), 'unknown')}]: "
            f"P/D Partition Profiling Only "
            f"({coverage_label})",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.012,
            "State = context / active Decode requests. No SLO threshold, "
            "attainment color, normalization, or missing-case estimate.",
            ha="center",
            color="#4B5563",
        )
        figure.tight_layout(rect=(0, 0.045, 1, 0.95))
        output_path = output_dir / (
            f"partition_profile_{profile_id}_{mode}_p50_p90_p99.png"
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def _annotate_mean_bars(
    axis: plt.Axes,
    bars: Any,
    repeat_counts: Sequence[int],
) -> None:
    for bar, repeat_count in zip(bars, repeat_counts):
        height = float(bar.get_height())
        label = f"{height:.1f}"
        if repeat_count != 3:
            label += f"\n(n={repeat_count})"
        axis.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )


def plot_partition_means(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    coverage_label: str,
    case_statuses: dict[tuple[str, str], str],
    dpi: int,
) -> tuple[list[Path], dict[str, float]]:
    """Plot equally weighted repeat means with metric-global y ranges."""
    configure_plot_style()
    ttft_ymax = max(
        float(row["mean_profile_ttft_ms"]) / 1000.0 for row in rows
    ) * 1.10
    tbt_ymax = max(
        float(row["mean_profile_overlap_tbt_ms"]) for row in rows
    ) * 1.10
    output_paths: list[Path] = []
    cases = sorted(
        {
            (str(row["profile_id"]), str(row["prefill_mode"]))
            for row in rows
        }
    )
    for profile_id, mode in cases:
        selected = sorted(
            (
                row
                for row in rows
                if row["profile_id"] == profile_id
                and row["prefill_mode"] == mode
            ),
            key=lambda row: (
                int(row["context_tokens"]),
                int(row["active_decode_requests"]),
            ),
        )
        labels = [
            f"{int(row['context_tokens']) // 1000}K/d"
            f"{int(row['active_decode_requests'])}"
            for row in selected
        ]
        x_values = np.arange(len(selected))
        figure, axes = plt.subplots(2, 1, figsize=(20, 10), squeeze=False)
        for row_index, (field, count_field, scale, ylabel, ymax) in enumerate(
            (
                (
                    "mean_profile_ttft_ms",
                    "ttft_repeat_count",
                    1000.0,
                    "Mean Prefill-probe TTFT (s)",
                    ttft_ymax,
                ),
                (
                    "mean_profile_overlap_tbt_ms",
                    "tbt_repeat_count",
                    1.0,
                    "Mean TBT (ms)",
                    tbt_ymax,
                ),
            )
        ):
            axis = axes[row_index][0]
            values = [float(row[field]) / scale for row in selected]
            repeat_counts = [int(row[count_field]) for row in selected]
            bars = axis.bar(
                x_values,
                values,
                width=0.68,
                color=PROFILE_BAR_COLOR,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=0.6,
                label="Mean across successful repeats",
            )
            _annotate_mean_bars(axis, bars, repeat_counts)
            axis.set_xticks(x_values, labels, rotation=35, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, ymax)
            axis.set_title("TTFT repeat mean" if row_index == 0 else "TBT repeat mean")
            axis.grid(axis="y", color="#E5E7EB", alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(frameon=False, loc="upper left")
        figure.suptitle(
            f"{profile_id} / {mode} "
            f"[{case_statuses.get((profile_id, mode), 'unknown')}]: "
            f"Mean P/D Partition Profile ({coverage_label})",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.012,
            "Each state gives equal weight to every successful repeat. "
            "TTFT axes share one global range; TBT axes share another. "
            "n<3 is printed below the value.",
            ha="center",
            color="#4B5563",
        )
        figure.tight_layout(rect=(0, 0.045, 1, 0.95))
        output_path = output_dir / (
            f"partition_profile_{profile_id}_{mode}_repeat_mean.png"
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths, {
        "shared_ttft_ymax_seconds": ttft_ymax,
        "shared_tbt_ymax_ms": tbt_ymax,
    }


def write_report(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    mean_rows: Sequence[dict[str, Any]],
    coverage_rows: Sequence[dict[str, Any]],
    graphs: Sequence[Path],
    *,
    partition_root: Path,
) -> None:
    completed = [
        row for row in coverage_rows if row["status"] == "completed"
    ]
    measured_cases = {
        (row["profile_id"], row["prefill_mode"]) for row in rows
    }
    measured_states = {
        (
            row["profile_id"],
            row["prefill_mode"],
            row["context_tokens"],
            row["active_decode_requests"],
        )
        for row in rows
    }
    graph_list = "\n".join(f"- `{path.name}`" for path in graphs)
    missing_list = "\n".join(
        f"- `{row['prefill_mode']}/{row['profile_id']}`: {row['status']}"
        for row in coverage_rows
        if row["status"] != "completed"
    )
    report = f"""# P/D Partition Profiling-Only Graphs

- Source: `{partition_root}`
- Measured cases: {len(measured_cases)}/{len(coverage_rows)}
- Completed expected cases: {len(completed)}/{len(coverage_rows)}
- Measured states: {len(measured_states)}
- Profiling color: light gray `{PROFILE_BAR_COLOR}`
- SLO values are not read, calculated, or drawn in partition figures.
- Repeat-mean figures give each successful repeat equal weight. All TTFT
  figures share one y range and all TBT figures share another y range.

## Measurement boundary

- TTFT: successful measurement-phase Prefill-probe TTFT trials.
- TBT: pooled Decode token gaps that intersect actual phase-local Prefill
  worker execution, weighted by streamed chunk token count.
- P50/P90/P99 are computed independently for every
  profile/mode/context/active-Decode state.
- Mean TTFT is the arithmetic mean of per-repeat TTFT. Mean TBT first averages
  actual-Prefill-overlap token gaps within each repeat, then averages those
  repeat means so a longer repeat does not receive extra weight.

## Graphs

{graph_list}

## Missing expected cases

{missing_list or "- none"}

## Files

- `partition_profile_quantiles.csv`
- `partition_profile_repeat_means.csv`
- `partition_profile_case_coverage.csv`
- `partition_profile_generation.json`
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate partition quantile and repeat-mean graphs without SLO"
        )
    )
    parser.add_argument(
        "--partition-results-dir",
        type=Path,
        default=ROOT / "vendor_v8_each_partition_profile_v1_results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "graphs"
            / "vendor_v8_each_partition_profile_p50_p90_p99"
        ),
    )
    parser.add_argument(
        "--expected-profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
    )
    parser.add_argument(
        "--expected-modes",
        nargs="+",
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--prefill-chunk-tokens", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--include-partial-cases",
        action="store_true",
        help=(
            "include successful measurement trials from failed/running "
            "cases while retaining their incomplete coverage status"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prefill_chunk_tokens <= 0:
        raise ValueError("--prefill-chunk-tokens must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    partition_root = args.partition_results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    completed_dirs, incomplete_dirs = collect_completed_partition_cases(
        partition_root
    )
    selected_dirs = list(completed_dirs)
    measurable_partial_dirs: list[Path] = []
    if args.include_partial_cases:
        measurable_partial_dirs = collect_measureable_partial_cases(
            incomplete_dirs
        )
        selected_dirs.extend(measurable_partial_dirs)
    if not selected_dirs:
        raise ValueError(
            f"no measurable partition cases under {partition_root}"
        )
    repeat_samples = load_partition_repeat_samples(
        selected_dirs,
        chunk_tokens=args.prefill_chunk_tokens,
    )
    samples = flatten_repeat_samples(repeat_samples)
    rows = build_profile_rows(samples)
    mean_rows = build_mean_rows(repeat_samples)
    coverage_rows = build_coverage_rows(
        partition_root,
        args.expected_profiles,
        args.expected_modes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "partition_profile_quantiles.csv",
        rows,
        PROFILE_FIELDS,
    )
    write_csv(
        output_dir / "partition_profile_repeat_means.csv",
        mean_rows,
        MEAN_FIELDS,
    )
    write_csv(
        output_dir / "partition_profile_case_coverage.csv",
        coverage_rows,
        COVERAGE_FIELDS,
    )
    completed_count = sum(
        row["status"] == "completed" for row in coverage_rows
    )
    measured_count = len(
        {
            (row["profile_id"], row["prefill_mode"])
            for row in rows
        }
    )
    coverage_label = (
        f"PARTIAL: {measured_count} measured, "
        f"{completed_count}/{len(coverage_rows)} completed"
    )
    if completed_count == len(coverage_rows):
        coverage_label = "complete"
    case_statuses = {
        (str(row["profile_id"]), str(row["prefill_mode"])): str(
            row["status"]
        )
        for row in coverage_rows
    }
    quantile_graphs = plot_partition_profiles(
        rows,
        output_dir,
        coverage_label=coverage_label,
        case_statuses=case_statuses,
        dpi=args.dpi,
    )
    mean_graphs, shared_mean_y_ranges = plot_partition_means(
        mean_rows,
        output_dir,
        coverage_label=coverage_label,
        case_statuses=case_statuses,
        dpi=args.dpi,
    )
    graphs = [*mean_graphs, *quantile_graphs]
    generation = {
        "schema_version": 1,
        "partition_results_dir": str(partition_root),
        "completed_expected_cases": completed_count,
        "measured_cases": measured_count,
        "expected_cases": len(coverage_rows),
        "discovered_completed_case_dirs": [
            str(path) for path in completed_dirs
        ],
        "discovered_incomplete_case_dirs": [
            str(path) for path in incomplete_dirs
        ],
        "include_partial_cases": args.include_partial_cases,
        "measurable_partial_case_dirs": [
            str(path) for path in measurable_partial_dirs
        ],
        "measurement_boundary": {
            "ttft": "measurement-phase Prefill-probe TTFT",
            "tbt": (
                "pooled chunk-token-weighted Decode token gaps intersecting "
                "actual phase-local Prefill execution"
            ),
        },
        "quantiles": [f"P{value}" for value in QUANTILES],
        "profile_bar_color": PROFILE_BAR_COLOR,
        "partition_slo_included": False,
        "repeat_mean_definition": {
            "ttft": "arithmetic mean across successful repeat TTFT values",
            "tbt": (
                "arithmetic mean across per-repeat means of pooled token "
                "gaps intersecting actual phase-local Prefill execution"
            ),
            "repeat_weighting": "equal",
        },
        "repeat_mean_shared_y_ranges": shared_mean_y_ranges,
        "profile_rows": rows,
        "repeat_mean_rows": mean_rows,
        "coverage_rows": coverage_rows,
    }
    (output_dir / "partition_profile_generation.json").write_text(
        json.dumps(generation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir,
        rows,
        mean_rows,
        coverage_rows,
        graphs,
        partition_root=partition_root,
    )
    print(
        "[DONE] "
        f"completed_cases={completed_count}/{len(coverage_rows)} "
        f"profile_rows={len(rows)} graphs={len(graphs)}"
    )
    print(f"[DONE] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
