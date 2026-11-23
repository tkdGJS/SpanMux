#!/usr/bin/env python3
"""Repair archived native-baseline TBT affected by per-fragment BOS counts.

The affected clients called ``/tokenize`` independently for every streamed
text fragment without disabling special tokens.  For this archive, one BOS was
therefore added to every fragment.  The raw chunk-arrival timestamp remains
valid.  This tool removes one BOS from each fragment, reconstructs the number
of model tokens represented by the fragment, and recomputes request and pooled
TBT statistics without modifying the original result directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from measure_native_40sm_profile_baseline import (
    RUN_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    TOKEN_FIELDNAMES,
    fmt,
    mean_or_none,
    percentile,
    summarize_bucket,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--special-tokens-per-fragment", type=int, default=1)
    return parser.parse_args()


def corrected_fragment_tokens(
    recorded_tokens: int,
    special_tokens_per_fragment: int,
) -> int:
    return max(1, recorded_tokens - special_tokens_per_fragment)


def main() -> int:
    args = parse_args()
    if args.special_tokens_per_fragment < 0:
        raise ValueError("--special-tokens-per-fragment must be non-negative")
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("input and output directories must differ")

    request_path = args.input_dir / "request_runs.csv"
    token_path = args.input_dir / "token_timeline.csv"
    summary_path = args.input_dir / "bucket_summary.csv"
    metadata_path = args.input_dir / "run_metadata.json"
    for path in (request_path, token_path, summary_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    request_rows = read_csv(request_path)
    token_rows = read_csv(token_path)
    legacy_summary_rows = read_csv(summary_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("input baseline must have status=completed")

    corrected_gaps_by_request: dict[str, list[float]] = defaultdict(list)
    corrected_tokens_by_request: dict[str, int] = defaultdict(int)
    corrected_token_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in token_rows:
        request_id = row["request_id"]
        recorded_tokens = max(1, int(row.get("chunk_tokens") or 1))
        model_tokens = corrected_fragment_tokens(
            recorded_tokens,
            args.special_tokens_per_fragment,
        )
        corrected_tokens_by_request[request_id] += model_tokens
        gap_value = row.get("chunk_gap_ms")
        corrected_gap: str | float = ""
        if gap_value not in ("", None):
            corrected_gap = float(gap_value) / model_tokens
            corrected_gaps_by_request[request_id].extend(
                [corrected_gap] * model_tokens
            )
        corrected_row = dict(row)
        corrected_row["chunk_tokens"] = model_tokens
        corrected_row["estimated_per_token_gap_ms"] = (
            "" if corrected_gap == "" else fmt(float(corrected_gap))
        )
        corrected_token_rows.append(corrected_row)

    corrected_request_rows: list[dict[str, Any]] = []
    for row in request_rows:
        corrected_row = dict(row)
        if row.get("status") == "success":
            request_id = row["request_id"]
            gaps = corrected_gaps_by_request[request_id]
            output_tokens = corrected_tokens_by_request[request_id]
            requested_tokens = int(row["requested_max_tokens"])
            if output_tokens != requested_tokens:
                raise ValueError(
                    f"{request_id}: corrected output_tokens={output_tokens}, "
                    f"requested_max_tokens={requested_tokens}"
                )
            if len(gaps) != max(0, output_tokens - 1):
                raise ValueError(
                    f"{request_id}: corrected gaps={len(gaps)}, "
                    f"expected={output_tokens - 1}"
                )
            corrected_row["output_tokens"] = output_tokens
            corrected_row["mean_tbt_ms"] = fmt(mean_or_none(gaps))
            corrected_row["p50_tbt_ms"] = fmt(percentile(gaps, 50))
            corrected_row["p90_tbt_ms"] = fmt(percentile(gaps, 90))
            corrected_row["p99_tbt_ms"] = fmt(percentile(gaps, 99))
            audit_rows.append(
                {
                    "request_id": request_id,
                    "phase": row["phase"],
                    "context_target_tokens": row["context_target_tokens"],
                    "repeat_index": row["repeat_index"],
                    "legacy_output_tokens": row["output_tokens"],
                    "corrected_output_tokens": output_tokens,
                    "legacy_p50_tbt_ms": row["p50_tbt_ms"],
                    "corrected_p50_tbt_ms": corrected_row["p50_tbt_ms"],
                    "legacy_p90_tbt_ms": row["p90_tbt_ms"],
                    "corrected_p90_tbt_ms": corrected_row["p90_tbt_ms"],
                    "legacy_p99_tbt_ms": row["p99_tbt_ms"],
                    "corrected_p99_tbt_ms": corrected_row["p99_tbt_ms"],
                }
            )
        corrected_request_rows.append(corrected_row)

    rows_by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pooled_by_bucket: dict[int, list[float]] = defaultdict(list)
    for row in corrected_request_rows:
        rows_by_bucket[int(row["bucket_index"])].append(row)
    for row in corrected_token_rows:
        if row.get("phase") != "measurement":
            continue
        value = row.get("estimated_per_token_gap_ms")
        if value in ("", None):
            continue
        pooled_by_bucket[int(row["bucket_index"])].extend(
            [float(value)] * int(row["chunk_tokens"])
        )

    corrected_summaries: list[dict[str, Any]] = []
    for legacy in legacy_summary_rows:
        bucket_index = int(legacy["bucket_index"])
        corrected_summaries.append(
            summarize_bucket(
                case_id=legacy["case_id"],
                bucket_index=bucket_index,
                target_tokens=int(legacy["context_target_tokens"]),
                actual_input_tokens=int(legacy["actual_input_tokens"]),
                warmup_count=int(legacy["warmup_count"]),
                measured_repeat_count=int(legacy["measured_repeat_count"]),
                request_max_tokens=int(legacy["request_max_tokens"]),
                request_min_tokens=int(legacy["request_min_tokens"]),
                cooldown_sec=float(legacy["cooldown_sec"]),
                run_rows=rows_by_bucket[bucket_index],
                pooled_token_gaps=pooled_by_bucket[bucket_index],
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "request_runs.csv",
        RUN_FIELDNAMES,
        corrected_request_rows,
    )
    write_csv(
        args.output_dir / "token_timeline.csv",
        TOKEN_FIELDNAMES,
        corrected_token_rows,
    )
    write_csv(
        args.output_dir / "bucket_summary.csv",
        SUMMARY_FIELDNAMES,
        corrected_summaries,
    )
    write_csv(
        args.output_dir / "tbt_correction_audit.csv",
        list(audit_rows[0]),
        audit_rows,
    )
    metadata.update(
        {
            "tbt_correction": {
                "source_dir": str(args.input_dir.resolve()),
                "special_tokens_removed_per_stream_fragment": (
                    args.special_tokens_per_fragment
                ),
                "method": (
                    "chunk_gap_ms / max(1, recorded_chunk_tokens - "
                    "special_tokens_per_fragment)"
                ),
                "original_files_modified": False,
            }
        }
    )
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "bucket_summary.csv")
    print(args.output_dir / "tbt_correction_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
