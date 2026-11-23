#!/usr/bin/env python3
"""Generate representative A-E timelines for vendor_v8 partition profiles.

One partition-profile case contains many independent trials, so plotting the
entire case as one timeline is misleading.  This tool selects one successful
measurement trial per case: the largest measured context, the largest active
Decode cardinality in that context, and the repeat closest to median TTFT.
All five panels use only that trial's Prefill-probe overlap window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-zoneatten")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PHASE_COLORS = {"decode": "#4C78A8", "prefill": "#F28E2B"}
COMPONENTS = (
    ("submit_gap", "scheduler/submit gap", "#9CA3AF"),
    ("async_queue_wait", "async/future wait", "#E15759"),
    ("worker_forward", "worker forward", "#4C78A8"),
    ("worker_other", "worker non-forward", "#59A14F"),
    ("output_gap", "output gap", "#F28E2B"),
)


@dataclass(frozen=True)
class SelectedTrial:
    case_id: str
    profile_id: str
    prefill_mode: str
    case_status: str
    context_tokens: int
    active_decode_requests: int
    repeat_index: int
    probe_start_ts: float
    probe_end_ts: float
    probe_ttft_ms: float


def fnum(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def select_representative_trial(
    trial_rows: Sequence[dict[str, str]],
    *,
    case_status: str,
) -> SelectedTrial:
    successful = [
        row
        for row in trial_rows
        if row.get("phase") == "measurement"
        and row.get("status") == "success"
    ]
    if not successful:
        raise ValueError("case has no successful measurement trial")
    context = max(int(row["context_tokens"]) for row in successful)
    at_context = [
        row for row in successful if int(row["context_tokens"]) == context
    ]
    decode_count = max(
        int(row["active_decode_requests"]) for row in at_context
    )
    candidates = [
        row
        for row in at_context
        if int(row["active_decode_requests"]) == decode_count
    ]
    median_ttft = statistics.median(
        fnum(row["probe_ttft_ms"]) for row in candidates
    )
    selected = min(
        candidates,
        key=lambda row: (
            abs(fnum(row["probe_ttft_ms"]) - median_ttft),
            int(row["repeat_index"]),
        ),
    )
    start = fnum(selected["probe_submit_monotonic_ts"], -1.0)
    end = fnum(selected["probe_first_token_monotonic_ts"], -1.0)
    if start < 0 or end <= start:
        raise ValueError("selected trial has an invalid probe window")
    return SelectedTrial(
        case_id=str(selected["case_id"]),
        profile_id=str(selected["profile_id"]),
        prefill_mode=str(selected["prefill_mode"]),
        case_status=case_status,
        context_tokens=context,
        active_decode_requests=decode_count,
        repeat_index=int(selected["repeat_index"]),
        probe_start_ts=start,
        probe_end_ts=end,
        probe_ttft_ms=fnum(selected["probe_ttft_ms"]),
    )


def intervals_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    return left_start < right_end and left_end > right_start


def metric_interval(row: dict[str, Any]) -> tuple[float, float]:
    start = fnum(
        row.get("execute_submit_ts"),
        fnum(row.get("schedule_start_ts"), fnum(row.get("scheduled_ts"))),
    )
    end = fnum(
        row.get("model_done_ts"),
        fnum(
            row.get("actual_model_future_done_ts"),
            fnum(row.get("output_process_end_ts"), start),
        ),
    )
    return start, max(start, end)


def load_window_metrics(
    path: Path,
    *,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            interval_start, interval_end = metric_interval(row)
            schedule_ts = fnum(
                row.get("schedule_start_ts"),
                fnum(row.get("scheduled_ts"), interval_start),
            )
            if (
                start <= schedule_ts <= end
                or intervals_overlap(interval_start, interval_end, start, end)
            ):
                rows.append(row)
    return rows


def load_window_events(
    path: Path,
    *,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") != "forward":
                continue
            event_start = fnum(row.get("host_start_event_recorded_ts"), -1.0)
            duration = fnum(row.get("device_elapsed_ms")) / 1000.0
            event_end = event_start + duration
            if event_start >= 0 and duration > 0 and intervals_overlap(
                event_start, event_end, start, end
            ):
                rows.append(row)
    return rows


def load_trial_tokens(
    path: Path,
    trial: SelectedTrial,
) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(path)
        if row.get("phase") == "measurement"
        and int(row.get("context_tokens") or -1) == trial.context_tokens
        and int(row.get("active_decode_requests") or -1)
        == trial.active_decode_requests
        and int(row.get("repeat_index") or -1) == trial.repeat_index
        and row.get("role") == "decode_anchor"
        and row.get("during_probe_prefill") == "1"
    ]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def title_prefix(trial: SelectedTrial) -> str:
    context = f"{trial.context_tokens // 1000}K"
    return (
        f"{trial.prefill_mode}/{trial.profile_id} [{trial.case_status}] "
        f"{context}, d={trial.active_decode_requests}, r={trial.repeat_index}"
    )


def save_empty_plot(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(14.0, 4.8))
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_a_gpu_execution(
    output_path: Path,
    trial: SelectedTrial,
    metrics: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    dpi: int,
) -> str:
    records: dict[str, list[tuple[float, float]]] = defaultdict(list)
    source = "CUDA event device_elapsed_ms"
    for row in events:
        phase = str(row.get("phase") or row.get("phase_context_phase") or "")
        if phase not in PHASE_COLORS:
            continue
        start = fnum(row.get("host_start_event_recorded_ts"))
        end = start + fnum(row.get("device_elapsed_ms")) / 1000.0
        clipped_start = max(start, trial.probe_start_ts)
        clipped_end = min(end, trial.probe_end_ts)
        if clipped_end > clipped_start:
            records[phase].append(
                (clipped_start - trial.probe_start_ts, clipped_end - clipped_start)
            )
    if not records:
        source = "worker forward host proxy"
        for row in metrics:
            phase = str(row.get("scheduled_phase") or "")
            if phase not in PHASE_COLORS:
                continue
            start = fnum(row.get("worker_execute_start_ts"), -1.0)
            duration = fnum(
                row.get("worker_model_forward_duration_s"),
                fnum(row.get("worker_forward_duration_s")),
            )
            clipped_start = max(start, trial.probe_start_ts)
            clipped_end = min(start + duration, trial.probe_end_ts)
            if start >= 0 and clipped_end > clipped_start:
                records[phase].append(
                    (
                        clipped_start - trial.probe_start_ts,
                        clipped_end - clipped_start,
                    )
                )
    if not records:
        save_empty_plot(
            output_path,
            f"A. {title_prefix(trial)} GPU execution",
            "No GPU execution interval in selected window",
        )
        return source
    figure, axis = plt.subplots(figsize=(14.0, 4.8))
    for y, phase in enumerate(("decode", "prefill")):
        if records.get(phase):
            axis.broken_barh(
                records[phase],
                (y - 0.34, 0.68),
                facecolors=PHASE_COLORS[phase],
                edgecolors="none",
                alpha=0.86,
            )
    axis.set_yticks([0, 1], ["Decode", "Prefill"])
    axis.set_xlim(0, (trial.probe_end_ts - trial.probe_start_ts))
    axis.set_xlabel("Time since probe submission (s)")
    axis.set_title(f"A. {title_prefix(trial)} GPU execution ({source})")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return source


def future_step_series(
    intervals: Iterable[tuple[float, float]],
    *,
    start: float,
    end: float,
) -> tuple[list[float], list[int]]:
    deltas: dict[float, int] = defaultdict(int)
    for interval_start, interval_end in intervals:
        clipped_start = max(start, interval_start)
        clipped_end = min(end, interval_end)
        if clipped_end <= clipped_start:
            continue
        deltas[clipped_start] += 1
        deltas[clipped_end] -= 1
    if not deltas:
        return [], []
    xs = [0.0]
    ys = [0]
    outstanding = 0
    for ts in sorted(deltas):
        xs.extend([ts - start, ts - start])
        ys.extend([outstanding, outstanding + deltas[ts]])
        outstanding += deltas[ts]
    xs.append(end - start)
    ys.append(outstanding)
    return xs, ys


def plot_b_future_outstanding(
    output_path: Path,
    trial: SelectedTrial,
    metrics: Sequence[dict[str, Any]],
    *,
    dpi: int,
) -> None:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in metrics:
        phase = str(row.get("submitted_scheduler_output_phase") or row.get("scheduled_phase") or "")
        if phase not in PHASE_COLORS:
            continue
        interval_start, interval_end = metric_interval(row)
        if interval_end > interval_start:
            intervals[phase].append((interval_start, interval_end))
    figure, axis = plt.subplots(figsize=(14.0, 4.8))
    plotted = False
    for phase in ("decode", "prefill"):
        xs, ys = future_step_series(
            intervals.get(phase, ()),
            start=trial.probe_start_ts,
            end=trial.probe_end_ts,
        )
        if xs:
            plotted = True
            axis.plot(xs, ys, color=PHASE_COLORS[phase], linewidth=1.5, label=phase.title())
    if not plotted:
        plt.close(figure)
        save_empty_plot(
            output_path,
            f"B. {title_prefix(trial)} future outstanding",
            "No future interval in selected window",
        )
        return
    axis.set_xlabel("Time since probe submission (s)")
    axis.set_ylabel("Outstanding futures")
    axis.set_title(f"B. {title_prefix(trial)} phase futures")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def anchor_label(request_id: str) -> str:
    match = re.search(r"anchor(\d+)", request_id)
    if match:
        return f"Decode anchor {int(match.group(1)) + 1}"
    return "Decode anchor"


def plot_c_token_service(
    output_path: Path,
    trial: SelectedTrial,
    tokens: Sequence[dict[str, str]],
    *,
    dpi: int,
) -> None:
    points: dict[str, list[tuple[float, int]]] = defaultdict(list)
    cumulative: dict[str, int] = defaultdict(int)
    ordered = sorted(
        tokens,
        key=lambda row: fnum(row.get("arrival_monotonic_ts"), float("inf")),
    )
    for row in ordered:
        arrival = fnum(row.get("arrival_monotonic_ts"), -1.0)
        if not trial.probe_start_ts <= arrival <= trial.probe_end_ts:
            continue
        label = anchor_label(str(row.get("request_id") or ""))
        cumulative[label] += max(1, int(row.get("chunk_tokens") or 1))
        points[label].append((arrival - trial.probe_start_ts, cumulative[label]))
    if not points:
        save_empty_plot(
            output_path,
            f"C. {title_prefix(trial)} token service",
            "No Decode token arrival in selected window",
        )
        return
    figure, axis = plt.subplots(figsize=(14.0, 4.8))
    for index, label in enumerate(sorted(points)):
        values = points[label]
        xs = [0.0] + [value[0] for value in values]
        ys = [0] + [value[1] for value in values]
        axis.step(xs, ys, where="post", linewidth=1.7, label=label)
    axis.set_xlabel("Time since probe submission (s)")
    axis.set_ylabel("Cumulative Decode tokens")
    axis.set_title(f"C. {title_prefix(trial)} token service")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=min(3, len(points)))
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def phase_breakdown(
    metrics: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in metrics:
        phase = str(row.get("scheduled_phase") or "")
        if phase not in PHASE_COLORS:
            continue
        scheduled = fnum(row.get("scheduled_ts"))
        worker_start = fnum(
            row.get("worker_execute_start_ts"),
            fnum(row.get("execute_submit_ts"), scheduled),
        )
        model_done = fnum(
            row.get("model_done_ts"),
            fnum(row.get("actual_model_future_done_ts"), worker_start),
        )
        output_done = fnum(row.get("output_process_end_ts"), model_done)
        worker_execute = fnum(row.get("worker_execute_duration_s"))
        worker_forward = fnum(
            row.get("worker_model_forward_duration_s"),
            fnum(row.get("worker_forward_duration_s")),
        )
        result[phase]["submit_gap"] += max(0.0, worker_start - scheduled)
        result[phase]["worker_forward"] += max(0.0, worker_forward)
        result[phase]["worker_other"] += max(
            0.0, worker_execute - worker_forward
        )
        result[phase]["async_queue_wait"] += max(
            0.0, model_done - worker_start - worker_execute
        )
        result[phase]["output_gap"] += max(0.0, output_done - model_done)
    return result


def plot_d_breakdown(
    output_path: Path,
    csv_path: Path,
    trial: SelectedTrial,
    metrics: Sequence[dict[str, Any]],
    *,
    dpi: int,
) -> None:
    breakdown = phase_breakdown(metrics)
    phases = [phase for phase in ("decode", "prefill") if phase in breakdown]
    if not phases:
        save_empty_plot(
            output_path,
            f"D. {title_prefix(trial)} phase breakdown",
            "No phase breakdown data in selected window",
        )
        return
    figure, axis = plt.subplots(figsize=(14.0, 4.8))
    left = [0.0] * len(phases)
    for key, label, color in COMPONENTS:
        values = [breakdown[phase].get(key, 0.0) for phase in phases]
        axis.barh(phases, values, left=left, color=color, label=label)
        left = [left[index] + values[index] for index in range(len(values))]
    axis.set_xlabel("Summed iteration time (s)")
    axis.set_title(f"D. {title_prefix(trial)} phase latency breakdown")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = ["phase"] + [key for key, _label, _color in COMPONENTS] + ["total"]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for phase in phases:
            row: dict[str, Any] = {"phase": phase}
            total = 0.0
            for key, _label, _color in COMPONENTS:
                value = breakdown[phase].get(key, 0.0)
                row[key] = f"{value:.9f}"
                total += value
            row["total"] = f"{total:.9f}"
            writer.writerow(row)


def plot_e_prefill_progress(
    output_path: Path,
    csv_path: Path,
    trial: SelectedTrial,
    metrics: Sequence[dict[str, Any]],
    *,
    dpi: int,
) -> None:
    chunks: list[tuple[float, int, str]] = []
    for row in metrics:
        tokens = int(row.get("num_prefill_tokens", 0) or 0)
        if tokens <= 0:
            continue
        ts = fnum(row.get("schedule_start_ts"), fnum(row.get("scheduled_ts")))
        if trial.probe_start_ts <= ts <= trial.probe_end_ts:
            ring_active = bool(
                fnum(row.get("prefill_attention_plan_ring_path_active"))
            )
            chunks.append((ts, tokens, "ring" if ring_active else "native"))
    chunks.sort()
    if not chunks:
        save_empty_plot(
            output_path,
            f"E. {title_prefix(trial)} Prefill progress",
            "No Prefill chunk in selected window",
        )
        return
    cumulative = 0
    xs = [0.0]
    ys = [0]
    output_rows: list[dict[str, Any]] = []
    for index, (ts, tokens, path) in enumerate(chunks, 1):
        cumulative += tokens
        xs.append(ts - trial.probe_start_ts)
        ys.append(cumulative)
        output_rows.append(
            {
                "chunk_index": index,
                "time_since_probe_s": f"{ts - trial.probe_start_ts:.9f}",
                "chunk_tokens": tokens,
                "cumulative_prefill_tokens": cumulative,
                "attention_path": path,
            }
        )
    figure, axis = plt.subplots(figsize=(14.0, 4.8))
    axis.step(xs, ys, where="post", color=PHASE_COLORS["prefill"], linewidth=1.8)
    axis.axhline(
        trial.context_tokens,
        color="#374151",
        linestyle="--",
        linewidth=1.1,
        label=f"Prompt target ({trial.context_tokens:,})",
    )
    axis.set_xlim(0, trial.probe_end_ts - trial.probe_start_ts)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Time since probe submission (s)")
    axis.set_ylabel("Cumulative scheduled Prefill tokens")
    axis.set_title(f"E. {title_prefix(trial)} Prefill progress")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def discover_cases(
    partition_root: Path,
    *,
    include_partial_cases: bool,
) -> list[tuple[Path, dict[str, Any], SelectedTrial]]:
    cases: list[tuple[Path, dict[str, Any], SelectedTrial]] = []
    for metadata_path in sorted(partition_root.rglob("run_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("profile_id") or not metadata.get("prefill_mode"):
            continue
        status = str(metadata.get("status") or "unknown")
        if status != "completed" and not include_partial_cases:
            continue
        case_dir = metadata_path.parent
        trial_path = case_dir / "trial_summary.csv"
        if not trial_path.is_file():
            continue
        try:
            selected = select_representative_trial(
                read_csv(trial_path),
                case_status=status,
            )
        except ValueError:
            continue
        cases.append((case_dir, metadata, selected))
    return cases


def write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate representative A-E partition-profile timelines"
    )
    parser.add_argument(
        "--partition-results-dir",
        type=Path,
        default=ROOT / "vendor_v8_each_partition_profile_v1_results",
    )
    parser.add_argument(
        "--partition-artifacts-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "graphs" / "vendor_v8_partition_profile_abcde",
    )
    parser.add_argument("--include-partial-cases", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    partition_root = args.partition_results_dir.expanduser().resolve()
    if args.partition_artifacts_dir is None:
        root_name = partition_root.name
        if not root_name.endswith("_results"):
            raise ValueError(
                "cannot infer artifacts directory; pass --partition-artifacts-dir"
            )
        artifacts_root = partition_root.with_name(
            root_name[: -len("_results")] + "_artifacts"
        )
    else:
        artifacts_root = args.partition_artifacts_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    cases = discover_cases(
        partition_root,
        include_partial_cases=args.include_partial_cases,
    )
    if not cases:
        raise ValueError(f"no measurable cases under {partition_root}")
    manifest: list[dict[str, Any]] = []
    for case_dir, metadata, trial in cases:
        mode = str(metadata["prefill_mode"])
        profile_id = str(metadata["profile_id"])
        case_output = output_root / mode / profile_id
        case_output.mkdir(parents=True, exist_ok=True)
        metrics = load_window_metrics(
            case_dir / "iteration_metrics.jsonl",
            start=trial.probe_start_ts,
            end=trial.probe_end_ts,
        )
        artifact_dir = artifacts_root / mode / profile_id
        events = load_window_events(
            artifact_dir / "pd_cuda_events.jsonl",
            start=trial.probe_start_ts,
            end=trial.probe_end_ts,
        )
        tokens = load_trial_tokens(
            case_dir / "client_token_timeline.csv",
            trial,
        )
        source = plot_a_gpu_execution(
            case_output / "A_gpu_execution_timeline.png",
            trial,
            metrics,
            events,
            dpi=args.dpi,
        )
        plot_b_future_outstanding(
            case_output / "B_future_outstanding_timeline.png",
            trial,
            metrics,
            dpi=args.dpi,
        )
        plot_c_token_service(
            case_output / "C_token_service_timeline.png",
            trial,
            tokens,
            dpi=args.dpi,
        )
        plot_d_breakdown(
            case_output / "D_breakdown_bar.png",
            case_output / "D_breakdown_bar.csv",
            trial,
            metrics,
            dpi=args.dpi,
        )
        plot_e_prefill_progress(
            case_output / "E_prefill_progress_timeline.png",
            case_output / "E_prefill_progress.csv",
            trial,
            metrics,
            dpi=args.dpi,
        )
        selected_payload = asdict(trial) | {
            "selection_policy": (
                "largest successful measurement context, then largest "
                "active Decode count, then repeat nearest median TTFT"
            ),
            "measurement_window": "probe submission to probe first token",
            "metrics_rows": len(metrics),
            "cuda_forward_event_rows": len(events),
            "decode_token_rows": len(tokens),
            "graph_a_source": source,
        }
        (case_output / "selected_trial.json").write_text(
            json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "prefill_mode": mode,
                "profile_id": profile_id,
                "case_status": trial.case_status,
                "context_tokens": trial.context_tokens,
                "active_decode_requests": trial.active_decode_requests,
                "repeat_index": trial.repeat_index,
                "probe_ttft_ms": f"{trial.probe_ttft_ms:.6f}",
                "metrics_rows": len(metrics),
                "cuda_forward_event_rows": len(events),
                "decode_token_rows": len(tokens),
                "output_dir": str(case_output),
            }
        )
        print(
            "[ABCDE] "
            f"{mode}/{profile_id} status={trial.case_status} "
            f"ctx={trial.context_tokens} d={trial.active_decode_requests} "
            f"r={trial.repeat_index}"
        )
    write_manifest(output_root / "selected_trial_manifest.csv", manifest)
    (output_root / "README.md").write_text(
        "# Vendor v8 partition-profile A-E graphs\n\n"
        "Each case uses the largest successful measurement context, the "
        "largest active Decode count at that context, and the repeat nearest "
        "median TTFT. Panels are restricted to the Prefill-probe overlap "
        "window. Partial cases retain their failed/running status.\n",
        encoding="utf-8",
    )
    print(f"[DONE] cases={len(manifest)} output_dir={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
