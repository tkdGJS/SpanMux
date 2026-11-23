#!/usr/bin/env python3
"""Compare recorded attention outputs with a dense FP64 causal reference.

The trace must have been recorded with
VLLM_CONTEXT_RING_CORRECTNESS_CAPTURE_QKV=1. The reference computes every
legal QK score for each sampled query token, but processes GQA KV heads
separately to avoid materializing repeated KV heads.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceRecord:
    request_id: str
    request_ordinal: int
    layer_name: str
    execution_path: str
    phase: str
    logical_start: int
    token_count: int
    occurrence: int
    file: str

    @property
    def logical_end(self) -> int:
        return self.logical_start + self.token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-root",
        required=True,
        help="Attention-correctness root or rank_NNN directory.",
    )
    parser.add_argument(
        "--execution-path",
        required=True,
        choices=("native_hbm", "context_ring"),
    )
    parser.add_argument("--label", default="trace")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", action="append", default=[])
    parser.add_argument("--request", action="append", default=[])
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--max-samples-per-stream", type=int, default=4)
    parser.add_argument("--softmax-scale", type=float)
    parser.add_argument("--output-json")
    return parser.parse_args()


def resolve_rank_dir(trace_root: Path, rank: int) -> Path:
    trace_root = trace_root.resolve()
    if (trace_root / "manifest.jsonl").is_file():
        return trace_root
    rank_dir = trace_root / f"rank_{rank:03d}"
    if not (rank_dir / "manifest.jsonl").is_file():
        raise FileNotFoundError(
            f"manifest.jsonl not found under {trace_root} or {rank_dir}"
        )
    return rank_dir


def load_manifest(rank_dir: Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    with (rank_dir / "manifest.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            if not line.strip():
                continue
            raw = json.loads(line)
            row = raw["row"]
            records.append(
                TraceRecord(
                    request_id=str(row["request_id"]),
                    request_ordinal=int(row["request_ordinal"]),
                    layer_name=str(raw["layer_name"]),
                    execution_path=str(raw["execution_path"]),
                    phase=str(row["phase"]),
                    logical_start=int(raw["logical_query_start"]),
                    token_count=int(raw["query_length"]),
                    occurrence=int(raw["occurrence"]),
                    file=str(raw["file"]),
                )
            )
    return records


def dense_causal_attention_fp64(
    query: Any,
    key: Any,
    value: Any,
    *,
    query_position: int,
    softmax_scale: float,
) -> Any:
    """Compute one dense causal attention row in FP64 without GQA copies."""

    import torch

    query = query.to(dtype=torch.float64, device="cpu")
    key = key[: query_position + 1].to(dtype=torch.float64, device="cpu")
    value = value[: query_position + 1].to(dtype=torch.float64, device="cpu")
    query_heads = int(query.shape[0])
    kv_heads = int(key.shape[1])
    if query_heads % kv_heads:
        raise ValueError(
            f"query heads {query_heads} are not divisible by KV heads {kv_heads}"
        )
    queries_per_kv = query_heads // kv_heads
    output = torch.empty(
        (query_heads, int(query.shape[-1])), dtype=torch.float64
    )
    for kv_head in range(kv_heads):
        head_start = kv_head * queries_per_kv
        head_end = head_start + queries_per_kv
        query_group = query[head_start:head_end]
        key_head = key[:, kv_head, :]
        value_head = value[:, kv_head, :]
        scores = query_group @ key_head.transpose(0, 1)
        scores.mul_(softmax_scale)
        probabilities = torch.softmax(scores, dim=-1)
        output[head_start:head_end] = probabilities @ value_head
    return output


def select_bounded_positions(
    positions: list[int], maximum: int
) -> list[int]:
    positions = sorted(set(positions))
    if not positions or maximum <= 0:
        return []
    if len(positions) <= maximum:
        return positions
    if maximum == 1:
        return [positions[-1]]
    indexes = {
        round(sample * (len(positions) - 1) / (maximum - 1))
        for sample in range(maximum)
    }
    return [positions[index] for index in sorted(indexes)]


def output_positions(records: list[TraceRecord]) -> list[int]:
    positions: list[int] = []
    for record in records:
        positions.extend(range(record.logical_start, record.logical_end))
    return sorted(set(positions))


def _sample_positions(records: list[TraceRecord], maximum: int) -> list[int]:
    return select_bounded_positions(output_positions(records), maximum)


def _payload(rank_dir: Path, record: TraceRecord) -> dict[str, Any]:
    import torch

    payload = torch.load(
        rank_dir / record.file,
        map_location="cpu",
        weights_only=True,
    )
    missing = {"output", "query", "key", "value"} - set(payload)
    if missing:
        raise ValueError(
            f"{record.file} lacks {sorted(missing)}; rerun with "
            "VLLM_CONTEXT_RING_CORRECTNESS_CAPTURE_QKV=1"
        )
    return payload


def _find_record(records: list[TraceRecord], position: int) -> TraceRecord:
    for record in records:
        if record.logical_start <= position < record.logical_end:
            return record
    raise ValueError(f"no output record covers logical token {position}")


def _reconstruct_kv(
    rank_dir: Path,
    records: list[TraceRecord],
    logical_end: int,
) -> tuple[Any, Any]:
    import torch

    unique: dict[tuple[int, int], TraceRecord] = {}
    for record in sorted(
        records,
        key=lambda item: (item.logical_start, item.occurrence),
    ):
        unique.setdefault((record.logical_start, record.token_count), record)

    cursor = 0
    key_parts = []
    value_parts = []
    for record in sorted(unique.values(), key=lambda item: item.logical_start):
        if record.logical_end <= cursor:
            continue
        if record.logical_start > cursor:
            raise ValueError(
                f"Q/K/V trace has a gap [{cursor}, {record.logical_start})"
            )
        payload = _payload(rank_dir, record)
        local_start = cursor - record.logical_start
        local_end = min(record.token_count, logical_end - record.logical_start)
        key_parts.append(payload["key"][local_start:local_end])
        value_parts.append(payload["value"][local_start:local_end])
        cursor = record.logical_start + local_end
        if cursor >= logical_end:
            break
    if cursor < logical_end:
        raise ValueError(f"Q/K/V trace ends at {cursor}, required {logical_end}")
    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)


def compare_stream(
    rank_dir: Path,
    all_records: list[TraceRecord],
    output_records: list[TraceRecord],
    *,
    max_samples: int,
    softmax_scale_override: float | None,
    sample_positions_override: list[int] | None = None,
) -> dict[str, Any]:
    import torch

    sample_positions = (
        sorted(set(sample_positions_override))
        if sample_positions_override is not None
        else _sample_positions(output_records, max_samples)
    )
    if not sample_positions:
        raise ValueError("stream contains no sampleable output positions")
    for position in sample_positions:
        _find_record(output_records, position)
    logical_end = max(sample_positions) + 1
    key, value = _reconstruct_kv(rank_dir, all_records, logical_end)
    key = key.to(dtype=torch.float64, device="cpu")
    value = value.to(dtype=torch.float64, device="cpu")
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    element_count = 0
    max_abs_error = 0.0
    sample_rows = []
    for position in sample_positions:
        record = _find_record(output_records, position)
        payload = _payload(rank_dir, record)
        local_index = position - record.logical_start
        query = payload["query"][local_index]
        actual = payload["output"][local_index].to(dtype=torch.float64)
        scale = (
            softmax_scale_override
            if softmax_scale_override is not None
            else payload.get("softmax_scale")
        )
        if scale is None:
            scale = float(int(query.shape[-1]) ** -0.5)
        reference = dense_causal_attention_fp64(
            query,
            key,
            value,
            query_position=position,
            softmax_scale=float(scale),
        )
        if actual.numel() != reference.numel():
            raise ValueError(
                f"output/query shape mismatch at logical token {position}: "
                f"output={tuple(actual.shape)} reference={tuple(reference.shape)}"
            )
        actual = actual.reshape_as(reference)
        difference = actual - reference
        squared_error = float(torch.sum(difference * difference))
        absolute_error = float(torch.sum(torch.abs(difference)))
        count = difference.numel()
        row_max = float(torch.max(torch.abs(difference)))
        squared_error_sum += squared_error
        absolute_error_sum += absolute_error
        element_count += count
        max_abs_error = max(max_abs_error, row_max)
        sample_rows.append({
            "logical_position": position,
            "rmse": math.sqrt(squared_error / count),
            "max_abs_error": row_max,
        })
    return {
        "request_id": output_records[0].request_id,
        "request_ordinal": output_records[0].request_ordinal,
        "layer_name": output_records[0].layer_name,
        "execution_path": output_records[0].execution_path,
        "sample_count": len(sample_positions),
        "sample_positions": sample_positions,
        "element_count": element_count,
        "rmse": math.sqrt(squared_error_sum / element_count),
        "mean_abs_error": absolute_error_sum / element_count,
        "max_abs_error": max_abs_error,
        "samples": sample_rows,
        "squared_error_sum": squared_error_sum,
        "absolute_error_sum": absolute_error_sum,
    }


def main() -> int:
    args = parse_args()
    if args.max_samples_per_stream <= 0:
        raise SystemExit("--max-samples-per-stream must be positive")
    rank_dir = resolve_rank_dir(Path(args.trace_root), args.rank)
    records = load_manifest(rank_dir)
    layer_patterns = args.layer or ["*"]
    request_patterns = args.request or ["*"]
    selected = [
        record
        for record in records
        if record.execution_path == args.execution_path
        and record.phase == args.phase
        and record.occurrence == 0
        and any(
            fnmatch.fnmatchcase(record.layer_name, pattern)
            for pattern in layer_patterns
        )
        and any(
            fnmatch.fnmatchcase(record.request_id, pattern)
            for pattern in request_patterns
        )
    ]
    grouped: dict[tuple[int, str], list[TraceRecord]] = {}
    for record in selected:
        grouped.setdefault(
            (record.request_ordinal, record.layer_name), []
        ).append(record)
    if not grouped:
        raise SystemExit("no manifest rows matched the requested path/filters")

    stream_results = []
    for stream_key, output_records in sorted(grouped.items()):
        request_ordinal, layer_name = stream_key
        stream_all_records = [
            record
            for record in records
            if record.request_ordinal == request_ordinal
            and record.layer_name == layer_name
            and record.occurrence == 0
        ]
        result = compare_stream(
            rank_dir,
            stream_all_records,
            sorted(output_records, key=lambda item: item.logical_start),
            max_samples=args.max_samples_per_stream,
            softmax_scale_override=args.softmax_scale,
            sample_positions_override=None,
        )
        stream_results.append(result)
        print(
            f"label={args.label} request={request_ordinal + 1} "
            f"layer={layer_name} path={args.execution_path} "
            f"samples={result['sample_count']} rmse={result['rmse']:.12g} "
            f"mean_abs={result['mean_abs_error']:.12g} "
            f"max_abs={result['max_abs_error']:.12g}"
        )

    total_squared_error = sum(
        result.pop("squared_error_sum") for result in stream_results
    )
    total_absolute_error = sum(
        result.pop("absolute_error_sum") for result in stream_results
    )
    total_elements = sum(result["element_count"] for result in stream_results)
    summary = {
        "schema_version": 1,
        "label": args.label,
        "trace_root": str(rank_dir),
        "execution_path": args.execution_path,
        "reference": "dense_causal_fp64",
        "stream_count": len(stream_results),
        "sample_count": sum(
            result["sample_count"] for result in stream_results
        ),
        "element_count": total_elements,
        "rmse": math.sqrt(total_squared_error / total_elements),
        "mean_abs_error": total_absolute_error / total_elements,
        "max_abs_error": max(
            result["max_abs_error"] for result in stream_results
        ),
        "streams": stream_results,
    }
    print(
        f"overall label={args.label} path={args.execution_path} "
        f"samples={summary['sample_count']} rmse={summary['rmse']:.12g} "
        f"mean_abs={summary['mean_abs_error']:.12g} "
        f"max_abs={summary['max_abs_error']:.12g}"
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
