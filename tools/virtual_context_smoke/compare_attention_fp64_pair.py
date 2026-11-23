#!/usr/bin/env python3
"""Compare native and context-ring outputs to FP64 at shared token positions."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path
from typing import Any

from compare_attention_fp64_reference import (
    TraceRecord,
    compare_stream,
    load_manifest,
    output_positions,
    resolve_rank_dir,
    select_bounded_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-trace-root", required=True)
    parser.add_argument("--ring-trace-root", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", action="append", default=[])
    parser.add_argument("--request", action="append", default=[])
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--max-samples-per-stream", type=int, default=4)
    parser.add_argument("--softmax-scale", type=float)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _group_outputs(
    records: list[TraceRecord],
    *,
    execution_path: str,
    phase: str,
    layer_patterns: list[str],
    request_patterns: list[str],
) -> dict[tuple[int, str], list[TraceRecord]]:
    grouped: dict[tuple[int, str], list[TraceRecord]] = {}
    for record in records:
        if (
            record.execution_path != execution_path
            or record.phase != phase
            or record.occurrence != 0
            or not any(
                fnmatch.fnmatchcase(record.layer_name, pattern)
                for pattern in layer_patterns
            )
            or not any(
                fnmatch.fnmatchcase(record.request_id, pattern)
                for pattern in request_patterns
            )
        ):
            continue
        grouped.setdefault(
            (record.request_ordinal, record.layer_name), []
        ).append(record)
    return grouped


def _all_stream_records(
    records: list[TraceRecord], stream_key: tuple[int, str]
) -> list[TraceRecord]:
    request_ordinal, layer_name = stream_key
    return [
        record
        for record in records
        if record.request_ordinal == request_ordinal
        and record.layer_name == layer_name
        and record.occurrence == 0
    ]


def _aggregate(
    stream_pairs: list[dict[str, Any]], side: str
) -> dict[str, Any]:
    results = [pair[side] for pair in stream_pairs]
    squared_error = sum(result["squared_error_sum"] for result in results)
    absolute_error = sum(result["absolute_error_sum"] for result in results)
    element_count = sum(result["element_count"] for result in results)
    return {
        "sample_count": sum(result["sample_count"] for result in results),
        "element_count": element_count,
        "rmse": math.sqrt(squared_error / element_count),
        "mean_abs_error": absolute_error / element_count,
        "max_abs_error": max(result["max_abs_error"] for result in results),
    }


def _strip_accumulators(result: dict[str, Any]) -> None:
    result.pop("squared_error_sum", None)
    result.pop("absolute_error_sum", None)


def main() -> int:
    args = parse_args()
    if args.max_samples_per_stream <= 0:
        raise SystemExit("--max-samples-per-stream must be positive")
    native_rank_dir = resolve_rank_dir(
        Path(args.native_trace_root), args.rank
    )
    ring_rank_dir = resolve_rank_dir(Path(args.ring_trace_root), args.rank)
    native_records = load_manifest(native_rank_dir)
    ring_records = load_manifest(ring_rank_dir)
    layer_patterns = args.layer or ["*"]
    request_patterns = args.request or ["*"]
    native_groups = _group_outputs(
        native_records,
        execution_path="native_hbm",
        phase=args.phase,
        layer_patterns=layer_patterns,
        request_patterns=request_patterns,
    )
    ring_groups = _group_outputs(
        ring_records,
        execution_path="context_ring",
        phase=args.phase,
        layer_patterns=layer_patterns,
        request_patterns=request_patterns,
    )
    shared_streams = sorted(set(native_groups) & set(ring_groups))
    if not shared_streams:
        raise SystemExit(
            "no request-ordinal/layer streams are shared by native and ring"
        )

    stream_pairs: list[dict[str, Any]] = []
    for stream_key in shared_streams:
        native_outputs = sorted(
            native_groups[stream_key], key=lambda item: item.logical_start
        )
        ring_outputs = sorted(
            ring_groups[stream_key], key=lambda item: item.logical_start
        )
        shared_positions = sorted(
            set(output_positions(native_outputs))
            & set(output_positions(ring_outputs))
        )
        sample_positions = select_bounded_positions(
            shared_positions, args.max_samples_per_stream
        )
        if not sample_positions:
            continue
        native_result = compare_stream(
            native_rank_dir,
            _all_stream_records(native_records, stream_key),
            native_outputs,
            max_samples=args.max_samples_per_stream,
            softmax_scale_override=args.softmax_scale,
            sample_positions_override=sample_positions,
        )
        ring_result = compare_stream(
            ring_rank_dir,
            _all_stream_records(ring_records, stream_key),
            ring_outputs,
            max_samples=args.max_samples_per_stream,
            softmax_scale_override=args.softmax_scale,
            sample_positions_override=sample_positions,
        )
        ratio = (
            ring_result["rmse"] / native_result["rmse"]
            if native_result["rmse"]
            else None
        )
        pair = {
            "request_ordinal": stream_key[0],
            "layer_name": stream_key[1],
            "sample_positions": sample_positions,
            "native": native_result,
            "ring": ring_result,
            "ring_minus_native_rmse": (
                ring_result["rmse"] - native_result["rmse"]
            ),
            "ring_to_native_rmse_ratio": ratio,
        }
        stream_pairs.append(pair)
        print(
            f"request={stream_key[0] + 1} layer={stream_key[1]} "
            f"positions={sample_positions} "
            f"E_native_rmse={native_result['rmse']:.12g} "
            f"E_ring_rmse={ring_result['rmse']:.12g} "
            f"ring/native={ratio if ratio is not None else 'undefined'}"
        )

    if not stream_pairs:
        raise SystemExit("shared streams contain no common logical positions")
    native_summary = _aggregate(stream_pairs, "native")
    ring_summary = _aggregate(stream_pairs, "ring")
    aggregate_ratio = (
        ring_summary["rmse"] / native_summary["rmse"]
        if native_summary["rmse"]
        else None
    )
    for pair in stream_pairs:
        _strip_accumulators(pair["native"])
        _strip_accumulators(pair["ring"])
    summary = {
        "schema_version": 1,
        "reference": "dense_causal_fp64",
        "native_trace_root": str(native_rank_dir),
        "ring_trace_root": str(ring_rank_dir),
        "phase": args.phase,
        "stream_count": len(stream_pairs),
        "native": native_summary,
        "ring": ring_summary,
        "ring_minus_native_rmse": (
            ring_summary["rmse"] - native_summary["rmse"]
        ),
        "ring_to_native_rmse_ratio": aggregate_ratio,
        "streams": stream_pairs,
    }
    print(
        f"overall E_native_rmse={native_summary['rmse']:.12g} "
        f"E_ring_rmse={ring_summary['rmse']:.12g} "
        f"ring/native={aggregate_ratio if aggregate_ratio is not None else 'undefined'}"
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
