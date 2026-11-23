#!/usr/bin/env python3
"""Reconstruct legacy partition-profile TBT without modifying raw results.

The legacy profiler re-tokenized each streamed text delta and therefore
counted one tokenizer-added BOS token in every delta.  Exact
``choice.token_ids`` were not persisted, so this tool performs the strongest
reconstruction supported by the archive: subtract one BOS from each recorded
fragment count and recompute token-weighted gaps from the original monotonic
timestamps.  New experiments must use ``choice.token_ids`` directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


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


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        rank - lower
    )


def fmt(value: float | None) -> str | float:
    return "" if value is None else round(float(value), 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--special-tokens-per-fragment", type=int, default=1)
    return parser.parse_args()


def corrected_tokens(recorded: int, special_tokens: int) -> int:
    return max(1, recorded - special_tokens)


def gap_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["phase"],
        row["context_tokens"],
        row["active_decode_requests"],
        row["repeat_index"],
    )


def reconstruct_case(
    source_case: Path,
    output_case: Path,
    special_tokens: int,
) -> dict[str, Any]:
    token_path = source_case / "client_token_timeline.csv"
    request_path = source_case / "request_runs.csv"
    trial_path = source_case / "trial_summary.csv"
    metadata_path = source_case / "run_metadata.json"
    iteration_path = source_case / "iteration_metrics.jsonl"
    for path in (
        token_path,
        request_path,
        trial_path,
        metadata_path,
        iteration_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    token_rows = read_csv(token_path)
    request_rows = read_csv(request_path)
    trial_rows = read_csv(trial_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    corrected_rows: list[dict[str, Any]] = []
    tokens_by_request: dict[str, int] = defaultdict(int)
    gaps_by_request: dict[str, list[float]] = defaultdict(list)
    overlap_gaps: dict[tuple[str, str, str, str], list[float]] = defaultdict(
        list
    )
    legacy_weighted_gap_count = 0
    corrected_weighted_gap_count = 0

    for row in token_rows:
        recorded = max(1, int(row.get("chunk_tokens") or 1))
        model_tokens = corrected_tokens(recorded, special_tokens)
        request_id = row["request_id"]
        tokens_by_request[request_id] += model_tokens
        corrected = dict(row)
        corrected["chunk_tokens"] = model_tokens

        start_raw = row.get("gap_start_monotonic_ts")
        end_raw = row.get("arrival_monotonic_ts")
        per_token_gap: float | None = None
        if start_raw not in (None, "") and end_raw not in (None, ""):
            chunk_gap_ms = (float(end_raw) - float(start_raw)) * 1000.0
            per_token_gap = chunk_gap_ms / model_tokens
            corrected["estimated_per_token_gap_ms"] = fmt(per_token_gap)
            gaps_by_request[request_id].extend(
                [per_token_gap] * model_tokens
            )
            legacy_weighted_gap_count += recorded
            corrected_weighted_gap_count += model_tokens
            if (
                row.get("role") == "decode_anchor"
                and row.get("during_probe_prefill") == "1"
            ):
                overlap_gaps[gap_key(row)].extend(
                    [per_token_gap] * model_tokens
                )
        else:
            corrected["estimated_per_token_gap_ms"] = ""
        corrected_rows.append(corrected)

    corrected_requests: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in request_rows:
        corrected = dict(row)
        if row.get("status") == "success":
            request_id = row["request_id"]
            gaps = gaps_by_request[request_id]
            output_tokens = tokens_by_request[request_id]
            corrected["output_tokens"] = output_tokens
            corrected["mean_tbt_ms"] = fmt(
                sum(gaps) / len(gaps) if gaps else None
            )
            corrected["p95_tbt_ms"] = fmt(percentile(gaps, 95))
            corrected["p99_tbt_ms"] = fmt(percentile(gaps, 99))
            audit_rows.append(
                {
                    "request_id": request_id,
                    "role": row["role"],
                    "phase": row["phase"],
                    "context_tokens": row["context_tokens"],
                    "active_decode_requests": row[
                        "active_decode_requests"
                    ],
                    "repeat_index": row["repeat_index"],
                    "legacy_output_tokens": row["output_tokens"],
                    "corrected_output_tokens": output_tokens,
                    "legacy_mean_tbt_ms": row["mean_tbt_ms"],
                    "corrected_mean_tbt_ms": corrected["mean_tbt_ms"],
                    "legacy_p95_tbt_ms": row["p95_tbt_ms"],
                    "corrected_p95_tbt_ms": corrected["p95_tbt_ms"],
                    "legacy_p99_tbt_ms": row["p99_tbt_ms"],
                    "corrected_p99_tbt_ms": corrected["p99_tbt_ms"],
                }
            )
        corrected_requests.append(corrected)

    corrected_trials: list[dict[str, Any]] = []
    for row in trial_rows:
        corrected = dict(row)
        gaps = overlap_gaps.get(gap_key(row), [])
        if row.get("status") == "success" and gaps:
            mean_gap = sum(gaps) / len(gaps)
            corrected["overlap_decode_gap_count"] = len(gaps)
            corrected["overlap_decode_mean_tbt_ms"] = fmt(mean_gap)
            corrected["overlap_decode_p95_tbt_ms"] = fmt(
                percentile(gaps, 95)
            )
            corrected["overlap_decode_p99_tbt_ms"] = fmt(
                percentile(gaps, 99)
            )
            corrected["overlap_decode_tokens_per_s"] = fmt(1000.0 / mean_gap)
            slo_raw = row.get("tbt_slo_ms")
            if slo_raw not in (None, ""):
                slo = float(slo_raw)
                corrected["tbt_token_slo_attainment"] = fmt(
                    sum(gap <= slo for gap in gaps) / len(gaps)
                )
        corrected_trials.append(corrected)

    output_case.mkdir(parents=True, exist_ok=True)
    write_csv(output_case / token_path.name, list(token_rows[0]), corrected_rows)
    write_csv(
        output_case / request_path.name,
        list(request_rows[0]),
        corrected_requests,
    )
    write_csv(output_case / trial_path.name, list(trial_rows[0]), corrected_trials)
    write_csv(
        output_case / "tbt_correction_audit.csv",
        list(audit_rows[0]) if audit_rows else (),
        audit_rows,
    )

    linked_iteration = output_case / iteration_path.name
    if linked_iteration.exists() or linked_iteration.is_symlink():
        linked_iteration.unlink()
    linked_iteration.symlink_to(
        os.path.relpath(iteration_path.resolve(), output_case.resolve())
    )
    metadata["tbt_correction"] = {
        "source_case_dir": str(source_case.resolve()),
        "measurement_source": "legacy_archive_bos_reconstruction",
        "exact_choice_token_ids_available": False,
        "special_tokens_removed_per_stream_fragment": special_tokens,
        "method": (
            "(arrival_ts-gap_start_ts)*1000 / "
            "max(1, recorded_chunk_tokens-special_tokens_per_fragment)"
        ),
        "iteration_metrics_reused_by_symlink": True,
        "original_files_modified": False,
    }
    (output_case / metadata_path.name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "profile_id": metadata["profile_id"],
        "prefill_mode": metadata["prefill_mode"],
        "status": metadata.get("status", "unknown"),
        "source_case_dir": str(source_case.resolve()),
        "corrected_case_dir": str(output_case.resolve()),
        "stream_fragment_count": len(token_rows),
        "legacy_weighted_gap_count": legacy_weighted_gap_count,
        "corrected_weighted_gap_count": corrected_weighted_gap_count,
        "exact_choice_token_ids_available": 0,
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must differ")
    if args.special_tokens_per_fragment < 0:
        raise ValueError("--special-tokens-per-fragment must be non-negative")

    case_metadata = sorted((input_dir / "cases").rglob("run_metadata.json"))
    if not case_metadata:
        raise ValueError(f"no partition cases under {input_dir}")
    summaries: list[dict[str, Any]] = []
    for metadata_path in case_metadata:
        source_case = metadata_path.parent
        relative_case = source_case.relative_to(input_dir)
        summaries.append(
            reconstruct_case(
                source_case,
                output_dir / relative_case,
                args.special_tokens_per_fragment,
            )
        )
        print(f"[corrected] {relative_case}", flush=True)

    write_csv(
        output_dir / "tbt_correction_case_summary.csv",
        list(summaries[0]),
        summaries,
    )
    manifest = {
        "schema_version": 1,
        "source_results_dir": str(input_dir),
        "measurement_source": "legacy_archive_bos_reconstruction",
        "exact_choice_token_ids_available": False,
        "special_tokens_removed_per_stream_fragment": (
            args.special_tokens_per_fragment
        ),
        "case_count": len(summaries),
        "original_files_modified": False,
    }
    (output_dir / "tbt_correction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "tbt_correction_case_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
