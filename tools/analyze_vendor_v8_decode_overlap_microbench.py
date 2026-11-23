#!/usr/bin/env python3
"""Compare sampled Decode forward boundaries with and without Prefill overlap.

This analyzer intentionally stops at CUDA-event boundaries.  It does not claim
to split a kernel interval into HBM, L2, SM, or driver causes.  Its job is to
answer the narrower question: did Decode's device interval grow, did host
enqueue grow, and how much device-time overlap with sampled Prefill forwards
was present?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_cuda_forward_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if payload.get("kind") != "forward":
                continue
            phase = str(payload.get("phase", ""))
            if phase not in {"decode", "prefill"}:
                continue
            if (
                payload.get("device_interval_basis")
                != "cuda_event_shared_epoch"
            ):
                raise ValueError(
                    "CUDA trace lacks the shared device epoch required for "
                    "cross-stream overlap classification; rebuild vendor_v8 "
                    "with the boundary-trace patch"
                )
            try:
                start_ms = float(payload["device_start_ms_from_epoch"])
                end_ms = float(payload["device_end_ms_from_epoch"])
                elapsed_ms = float(payload["device_elapsed_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"incomplete CUDA interval at {path}:{line_number}"
                ) from exc
            if end_ms < start_ms:
                raise ValueError(
                    f"negative CUDA interval at {path}:{line_number}"
                )
            records.append(
                {
                    **payload,
                    "phase": phase,
                    "device_start_ms_from_epoch": start_ms,
                    "device_end_ms_from_epoch": end_ms,
                    "device_elapsed_ms": elapsed_ms,
                }
            )
    return records


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_overlap_ms(
    start_ms: float,
    end_ms: float,
    intervals: Iterable[tuple[float, float]],
) -> float:
    return sum(
        max(0.0, min(end_ms, other_end) - max(start_ms, other_start))
        for other_start, other_end in intervals
    )


def classify_decode_records(
    records: list[dict[str, Any]],
    *,
    minimum_overlap_ms: float = 0.001,
) -> list[dict[str, Any]]:
    prefill_by_epoch: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        if record["phase"] != "prefill":
            continue
        prefill_by_epoch[str(record.get("device_epoch_id", ""))].append(
            (
                float(record["device_start_ms_from_epoch"]),
                float(record["device_end_ms_from_epoch"]),
            )
        )
    merged_prefill = {
        epoch: merge_intervals(intervals)
        for epoch, intervals in prefill_by_epoch.items()
    }
    classified: list[dict[str, Any]] = []
    for record in records:
        if record["phase"] != "decode":
            continue
        epoch = str(record.get("device_epoch_id", ""))
        overlap_ms = interval_overlap_ms(
            float(record["device_start_ms_from_epoch"]),
            float(record["device_end_ms_from_epoch"]),
            merged_prefill.get(epoch, ()),
        )
        elapsed_ms = float(record["device_elapsed_ms"])
        try:
            host_enqueue_ms = max(
                0.0,
                (
                    float(record["host_enqueue_end_ts"])
                    - float(record["host_enqueue_start_ts"])
                )
                * 1000.0,
            )
        except (KeyError, TypeError, ValueError):
            host_enqueue_ms = math.nan
        classified.append(
            {
                **record,
                "prefill_device_overlap_ms": overlap_ms,
                "prefill_device_overlap_ratio": (
                    overlap_ms / elapsed_ms if elapsed_ms > 0 else 0.0
                ),
                "prefill_device_overlap": int(overlap_ms >= minimum_overlap_ms),
                "host_enqueue_ms": host_enqueue_ms,
            }
        )
    return classified


def summarize_decode_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for overlap in (0, 1):
        selected = [
            record
            for record in records
            if int(record["prefill_device_overlap"]) == overlap
        ]
        elapsed = [float(record["device_elapsed_ms"]) for record in selected]
        host = [
            float(record["host_enqueue_ms"])
            for record in selected
            if math.isfinite(float(record["host_enqueue_ms"]))
        ]
        overlap_values = [
            float(record["prefill_device_overlap_ms"]) for record in selected
        ]
        ratios = [
            float(record["prefill_device_overlap_ratio"]) for record in selected
        ]
        rows.append(
            {
                "scope": "prefill_overlap" if overlap else "decode_only",
                "sample_count": len(selected),
                "device_forward_mean_ms": (
                    statistics.fmean(elapsed) if elapsed else math.nan
                ),
                "device_forward_p50_ms": percentile(elapsed, 50),
                "device_forward_p90_ms": percentile(elapsed, 90),
                "device_forward_p95_ms": percentile(elapsed, 95),
                "host_enqueue_p50_ms": percentile(host, 50),
                "prefill_overlap_p50_ms": percentile(overlap_values, 50),
                "prefill_overlap_ratio_p50": percentile(ratios, 50),
            }
        )
    return rows


def load_client_window_summary(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("phase") == "measurement"
            and row.get("status") == "success"
        ]
    control = [
        float(row["decode_only_control_p50_tbt_ms"]) for row in rows
    ]
    overlap = [float(row["overlap_decode_p50_tbt_ms"]) for row in rows]
    control_p50 = statistics.median(control) if control else math.nan
    overlap_p50 = statistics.median(overlap) if overlap else math.nan
    return {
        "measurement_windows": len(rows),
        "decode_only_median_window_p50_tbt_ms": control_p50,
        "overlap_median_window_p50_tbt_ms": overlap_p50,
        "overlap_minus_decode_only_tbt_ms": overlap_p50 - control_p50,
        "overlap_over_decode_only_tbt": (
            overlap_p50 / control_p50 if control_p50 > 0 else math.nan
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-trace", type=Path, required=True)
    parser.add_argument("--trial-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-overlap-ms", type=float, default=0.001)
    args = parser.parse_args()

    records = load_cuda_forward_records(args.cuda_trace)
    classified = classify_decode_records(
        records, minimum_overlap_ms=args.minimum_overlap_ms
    )
    if not classified:
        raise ValueError("CUDA trace contains no sampled Decode forward records")
    gpu_summary = summarize_decode_records(classified)
    client = load_client_window_summary(args.trial_csv)
    if any(int(row["sample_count"]) == 0 for row in gpu_summary):
        raise ValueError(
            "both Decode-only and Prefill-overlap sampled Decode forwards "
            "are required; reduce the Decode sampling interval or extend "
            "the probe"
        )
    if int(client["measurement_windows"]) == 0:
        raise ValueError(
            "trial CSV contains no successful measurement windows"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "decode_forward_records.csv", classified)
    write_csv(args.output_dir / "gpu_boundary_summary.csv", gpu_summary)
    with (args.output_dir / "client_tbt_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(client, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    by_scope = {str(row["scope"]): row for row in gpu_summary}
    control = by_scope["decode_only"]
    overlap = by_scope["prefill_overlap"]
    control_gpu = float(control["device_forward_p50_ms"])
    overlap_gpu = float(overlap["device_forward_p50_ms"])
    report = f"""# Decode-only vs Prefill-overlap boundary report

| Metric | Decode-only | Prefill overlap | Delta/ratio |
|---|---:|---:|---:|
| sampled Decode forwards | {control['sample_count']} | {overlap['sample_count']} | - |
| device forward P50 (ms) | {format_value(control_gpu)} | {format_value(overlap_gpu)} | {format_value(overlap_gpu - control_gpu)} ms |
| host enqueue P50 (ms) | {format_value(float(control['host_enqueue_p50_ms']))} | {format_value(float(overlap['host_enqueue_p50_ms']))} | - |
| client per-token TBT, median window P50 (ms) | {format_value(float(client['decode_only_median_window_p50_tbt_ms']))} | {format_value(float(client['overlap_median_window_p50_tbt_ms']))} | {format_value(float(client['overlap_over_decode_only_tbt']))}x |

The GPU rows use non-synchronizing CUDA events on a shared device epoch.
They identify boundary-level device slowdown and actual P/D device overlap,
but do not attribute elapsed time inside a kernel to HBM, L2, or SM causes.
The client TBT rows use `choice.token_ids` and exclude the first token.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"[DONE] output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
