#!/usr/bin/env python3
"""Aggregate fixed-partition measurements into vendor_v8 runtime profiles."""

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

from generate_native_profile_slo import build_profile, read_csv as read_native_csv


SUMMARY_FIELDNAMES = [
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "context_tokens",
    "active_decode_requests",
    "measurement_trials",
    "probe_ttft_p50_ms",
    "probe_ttft_p95_ms",
    "prefill_throughput_mean_tokens_per_s",
    "decode_overlap_gap_count",
    "decode_overlap_tbt_p50_ms",
    "decode_overlap_tbt_p95_ms",
    "decode_overlap_tbt_p99_ms",
    "decode_overlap_throughput_mean_tokens_per_s",
    "ttft_slo_attainment",
    "tbt_token_slo_attainment",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


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


def phase_local_prefill_ms(row: dict[str, Any]) -> float | None:
    for key, scale in (
        ("worker_execute_duration_s", 1000.0),
        ("worker_observed_duration_s", 1000.0),
        ("gpu_forward_elapsed_ms", 1.0),
    ):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        value = float(raw) * scale
        if math.isfinite(value) and value > 0:
            return value
    return None


def phase_local_prefill_interval(
    row: dict[str, Any],
) -> tuple[float, float] | None:
    start: float | None = None
    for key in (
        "worker_execute_start_ts",
        "execute_submit_ts",
        "schedule_start_ts",
        "scheduled_ts",
    ):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        candidate = float(raw)
        if math.isfinite(candidate) and candidate >= 0:
            start = candidate
            break
    if start is None:
        return None
    duration_ms = phase_local_prefill_ms(row)
    if duration_ms is None:
        return None
    end = start + duration_ms / 1000.0
    if not math.isfinite(end) or end <= start:
        return None
    return start, end


def iteration_uses_ring(row: dict[str, Any]) -> bool:
    return any(
        float(row.get(key, 0) or 0) > 0
        for key in (
            "virtual_context_active_decode_ring_prefill_allowed_count",
            "virtual_context_active_decode_ring_prefill_force_all_count",
            "context_ring_executor_calls",
            "context_ring_backend_execute_calls",
            "tiered_context_ring_executor_calls",
        )
    )


def collect_case_dirs(profile_root: Path) -> list[Path]:
    case_dirs: list[Path] = []
    for metadata_path in sorted(profile_root.rglob("run_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("profile_id") and metadata.get("prefill_mode"):
            case_dirs.append(metadata_path.parent)
    return case_dirs


def aggregate_profiles(
    case_dirs: Sequence[Path],
    *,
    chunk_tokens: int,
    allow_incomplete: bool,
    native_baseline_results_dir: Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    summary_groups: dict[
        tuple[str, int, int, str, int, int],
        dict[str, Any],
    ] = {}
    prefill_samples: dict[
        tuple[str, str, int], list[float]
    ] = defaultdict(list)
    tbt_samples: dict[
        tuple[str, str, int, int], list[float]
    ] = defaultdict(list)
    expected_prefill_keys: set[tuple[str, str, int]] = set()
    expected_tbt_keys: set[tuple[str, str, int, int]] = set()
    prefill_intervals: dict[
        tuple[str, str, int, int, int], list[tuple[float, float]]
    ] = defaultdict(list)
    sources: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        metadata_path = case_dir / "run_metadata.json"
        trial_path = case_dir / "trial_summary.csv"
        token_path = case_dir / "client_token_timeline.csv"
        iteration_path = case_dir / "iteration_metrics.jsonl"
        for path in (metadata_path, trial_path, token_path, iteration_path):
            if not path.is_file():
                if allow_incomplete:
                    break
                raise FileNotFoundError(path)
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "completed":
                if allow_incomplete:
                    continue
                raise ValueError(f"incomplete profile case: {case_dir}")
            if (
                metadata.get("tbt_token_count_source")
                != "choice.token_ids"
                or metadata.get("tbt_requires_return_token_ids") is not True
            ):
                raise ValueError(
                    f"{case_dir}: exact choice.token_ids TBT measurement "
                    "contract is required"
                )
            profile_id = str(metadata["profile_id"])
            decode_sms = int(metadata["decode_sms"])
            prefill_sms = int(metadata["prefill_sms"])
            mode = str(metadata["prefill_mode"])
            trials = [
                row
                for row in read_csv(trial_path)
                if row.get("phase") == "measurement"
                and row.get("status") == "success"
            ]
            tokens = [
                row
                for row in read_csv(token_path)
                if row.get("phase") == "measurement"
                and row.get("role") == "decode_anchor"
                and row.get("during_probe_prefill") == "1"
                and row.get("estimated_per_token_gap_ms") not in ("", None)
            ]
            iterations = read_jsonl(iteration_path)
            sources.append(
                {
                    "case_dir": str(case_dir),
                    "profile_id": profile_id,
                    "prefill_mode": mode,
                }
            )

            by_summary_key: dict[
                tuple[int, int], dict[str, list[float]]
            ] = defaultdict(
                lambda: {
                    "ttft": [],
                    "prefill_tps": [],
                    "decode_tps": [],
                    "ttft_attainment": [],
                    "tbt_attainment": [],
                }
            )
            for trial in trials:
                context = int(trial["context_tokens"])
                probe_context = int(
                    trial.get("probe_context_tokens") or context
                )
                decode_count = int(trial["active_decode_requests"])
                expected_prefill_keys.add(
                    (profile_id, mode, probe_context)
                )
                expected_tbt_keys.add(
                    (profile_id, mode, context, decode_count)
                )
                group = by_summary_key[(context, decode_count)]
                group["ttft"].append(float(trial["probe_ttft_ms"]))
                group["prefill_tps"].append(
                    float(trial["probe_effective_prefill_tokens_per_s"])
                )
                if trial.get("overlap_decode_tokens_per_s"):
                    group["decode_tps"].append(
                        float(trial["overlap_decode_tokens_per_s"])
                    )
                group["ttft_attainment"].append(
                    float(trial["ttft_slo_met"])
                )
                if trial.get("tbt_token_slo_attainment"):
                    group["tbt_attainment"].append(
                        float(trial["tbt_token_slo_attainment"])
                    )

                start = float(trial["probe_submit_monotonic_ts"])
                end = float(trial["probe_first_token_monotonic_ts"])
                matching_iterations = [
                    row
                    for row in iterations
                    if int(row.get("num_prefill_tokens", 0) or 0) > 0
                    and start
                    <= float(
                        row.get(
                            "schedule_start_ts",
                            row.get("scheduled_ts", -1),
                        )
                    )
                    <= end
                ]
                for iteration in matching_iterations:
                    scheduled = int(
                        iteration.get("num_prefill_tokens", 0) or 0
                    )
                    if scheduled > chunk_tokens:
                        raise ValueError(
                            f"{case_dir}: scheduled prefill chunk {scheduled} "
                            f"exceeds profile quantum {chunk_tokens}"
                        )
                    actual_ring = iteration_uses_ring(iteration)
                    if (mode == "ring") != actual_ring:
                        continue
                    service_ms = phase_local_prefill_ms(iteration)
                    if service_ms is not None:
                        prefill_samples[
                            (profile_id, mode, probe_context)
                        ].append(service_ms)
                    interval = phase_local_prefill_interval(iteration)
                    if interval is not None:
                        prefill_intervals[
                            (
                                profile_id,
                                mode,
                                context,
                                decode_count,
                                int(trial["repeat_index"]),
                            )
                        ].append(interval)

            by_token_key: dict[tuple[int, int], list[float]] = defaultdict(list)
            for token in tokens:
                context = int(token["context_tokens"])
                decode_count = int(token["active_decode_requests"])
                interval_key = (
                    profile_id,
                    mode,
                    context,
                    decode_count,
                    int(token["repeat_index"]),
                )
                gap_start_raw = token.get("gap_start_monotonic_ts")
                gap_end_raw = token.get("arrival_monotonic_ts")
                if gap_start_raw in ("", None) or gap_end_raw in ("", None):
                    continue
                gap_start = float(gap_start_raw)
                gap_end = float(gap_end_raw)
                if not any(
                    gap_start < interval_end and gap_end > interval_start
                    for interval_start, interval_end in prefill_intervals.get(
                        interval_key, []
                    )
                ):
                    continue
                gap = float(token["estimated_per_token_gap_ms"])
                by_token_key[(context, decode_count)].extend(
                    [gap] * max(1, int(token["chunk_tokens"]))
                )
                tbt_samples[
                    (profile_id, mode, context, decode_count)
                ].extend([gap] * max(1, int(token["chunk_tokens"])))

            for (context, decode_count), values in by_summary_key.items():
                gaps = by_token_key.get((context, decode_count), [])
                key = (
                    profile_id,
                    decode_sms,
                    prefill_sms,
                    mode,
                    context,
                    decode_count,
                )
                summary_groups[key] = {
                    "profile_id": profile_id,
                    "decode_sms": decode_sms,
                    "prefill_sms": prefill_sms,
                    "prefill_mode": mode,
                    "context_tokens": context,
                    "active_decode_requests": decode_count,
                    "measurement_trials": len(values["ttft"]),
                    "probe_ttft_p50_ms": percentile(values["ttft"], 50),
                    "probe_ttft_p95_ms": percentile(values["ttft"], 95),
                    "prefill_throughput_mean_tokens_per_s": (
                        statistics.fmean(values["prefill_tps"])
                    ),
                    "decode_overlap_gap_count": len(gaps),
                    "decode_overlap_tbt_p50_ms": (
                        percentile(gaps, 50) if gaps else ""
                    ),
                    "decode_overlap_tbt_p95_ms": (
                        percentile(gaps, 95) if gaps else ""
                    ),
                    "decode_overlap_tbt_p99_ms": (
                        percentile(gaps, 99) if gaps else ""
                    ),
                    "decode_overlap_throughput_mean_tokens_per_s": (
                        statistics.fmean(values["decode_tps"])
                        if values["decode_tps"]
                        else ""
                    ),
                    "ttft_slo_attainment": statistics.fmean(
                        values["ttft_attainment"]
                    ),
                    "tbt_token_slo_attainment": (
                        statistics.fmean(values["tbt_attainment"])
                        if values["tbt_attainment"]
                        else ""
                    ),
                }

    missing_prefill = sorted(
        key for key in expected_prefill_keys if not prefill_samples.get(key)
    )
    missing_tbt = sorted(
        key for key in expected_tbt_keys if not tbt_samples.get(key)
    )
    if missing_prefill and not allow_incomplete:
        raise ValueError(
            f"missing phase-local prefill samples for {missing_prefill}"
        )
    if missing_tbt and not allow_incomplete:
        raise ValueError(f"missing overlap TBT samples for {missing_tbt}")
    prefill_points = [
        {
            "profile_id": profile_id,
            "prefill_mode": mode,
            "max_context_tokens": context,
            "max_chunk_tokens": chunk_tokens,
            "p95_prefill_chunk_ms": percentile(values, 95),
        }
        for (profile_id, mode, context), values in sorted(
            prefill_samples.items()
        )
        if values
    ]
    tbt_points = [
        {
            "profile_id": profile_id,
            "max_decode_context_tokens": context,
            "max_decode_requests": decode_count,
            "prefill_mode": mode,
            "p95_per_token_tbt_ms": percentile(values, 95),
        }
        for (profile_id, mode, context, decode_count), values in sorted(
            tbt_samples.items()
        )
        if values
    ]
    if native_baseline_results_dir is not None:
        baseline_raw = native_baseline_results_dir / "request_runs.csv"
        baseline_summary = native_baseline_results_dir / "bucket_summary.csv"
        baseline_metadata_path = native_baseline_results_dir / "run_metadata.json"
        for path in (baseline_raw, baseline_summary, baseline_metadata_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        baseline_metadata = json.loads(
            baseline_metadata_path.read_text(encoding="utf-8")
        )
        if (
            baseline_metadata.get("tbt_token_count_source")
            != "choice.token_ids"
            or baseline_metadata.get("tbt_requires_return_token_ids") is not True
        ):
            raise ValueError(
                f"{native_baseline_results_dir}: exact choice.token_ids "
                "native TBT measurement contract is required"
            )
        native_profile = build_profile(
            read_native_csv(baseline_raw),
            read_native_csv(baseline_summary),
        )
        tbt_points.extend(
            {
                "profile_id": "d40_p0",
                "max_decode_context_tokens": int(row["context_tokens"]),
                "max_decode_requests": 1,
                "prefill_mode": "none",
                # A p95 across per-request p99 values is conservative relative
                # to the runtime's per-output-token p95 contract.
                "p95_per_token_tbt_ms": float(
                    row["request_p99_tbt_p95_ms"]
                ),
            }
            for row in native_profile
        )
        sources.append(
            {
                "case_dir": str(native_baseline_results_dir),
                "profile_id": "d40_p0",
                "prefill_mode": "none",
                "source": "native_40sm_request_p99_tbt_p95",
            }
        )
    if not prefill_points and not allow_incomplete:
        raise ValueError("no prefill profile points were produced")
    if not tbt_points and not allow_incomplete:
        raise ValueError("no TBT profile points were produced")
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sources": sources,
        "chunk_tokens": chunk_tokens,
        "measurement_boundary": {
            "prefill": (
                "worker phase-local execute duration; scheduler/future queue "
                "delay excluded"
            ),
            "tbt": (
                "client token gaps inside the all-anchors-active probe "
                "window that intersect an actual phase-local Prefill worker "
                "execution interval"
            ),
        },
    }
    return (
        list(summary_groups.values()),
        {"schema_version": 1, "points": prefill_points},
        {
            "schema_version": 2,
            "measurement_contract": {
                "token_count_source": "choice.token_ids",
                "gap_field": "estimated_per_token_gap_ms",
                "unit": "ms_per_output_token",
                "chunk_gap_normalization": (
                    "interarrival_ms_divided_by_emitted_token_count"
                ),
                "first_nonempty_chunk": "excluded_as_ttft",
            },
            "points": tbt_points,
        },
        metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build vendor_v8 runtime profiles from partition cases"
    )
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--native-baseline-results-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    root = args.profile_root.expanduser().resolve()
    case_dirs = collect_case_dirs(root)
    if not case_dirs:
        raise ValueError(f"no partition profile cases found under {root}")
    summary, prefill_profile, tbt_profile, metadata = aggregate_profiles(
        case_dirs,
        chunk_tokens=args.chunk_tokens,
        allow_incomplete=args.allow_incomplete,
        native_baseline_results_dir=(
            args.native_baseline_results_dir.expanduser().resolve()
            if args.native_baseline_results_dir is not None
            else None
        ),
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "partition_profile_summary.csv", SUMMARY_FIELDNAMES, summary)
    for filename, payload in (
        ("prefill_service_profile.json", prefill_profile),
        ("tbt_overlap_profile.json", tbt_profile),
        ("profile_generation_metadata.json", metadata),
    ):
        (output / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        "[DONE] "
        f"cases={len(case_dirs)} summary_rows={len(summary)} "
        f"prefill_points={len(prefill_profile['points'])} "
        f"tbt_points={len(tbt_profile['points'])}"
    )
    print(f"[DONE] output_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
