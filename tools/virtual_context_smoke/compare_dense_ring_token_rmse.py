#!/usr/bin/env python3
"""Compare captured dense and context-ring attention outputs token by token.

This is an output-to-output comparator. It does not reconstruct Q/K/V and does
not run a higher-precision attention reference. Both traces must describe the
same prepared requests in the same order so that semantic request identities,
layers, phases, and logical query-token positions have identical meanings.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import fnmatch
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


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
    dtype: str
    shape: tuple[int, ...]

    @property
    def logical_end(self) -> int:
        return self.logical_start + self.token_count


@dataclass(frozen=True)
class StreamKey:
    request_key: str
    layer_name: str


_BENCHMARK_REQUEST_ID = re.compile(
    r"-(foreground|background)-(.+)-(\d+)-(\d+)-(\d+)-([^-]+)$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute token-wise RMSE CDF directly between captured dense "
            "and context-ring attention outputs."
        )
    )
    parser.add_argument("--dense-trace-root", required=True)
    parser.add_argument("--ring-trace-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--dense-execution-path", default="native_hbm")
    parser.add_argument("--ring-execution-path", default="context_ring")
    parser.add_argument(
        "--phase", choices=("prefill", "decode"), default="prefill"
    )
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        help="Layer fnmatch pattern. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--request-index",
        action="append",
        type=int,
        default=[],
        help="One-based request index. Repeat to select multiple requests.",
    )
    parser.add_argument(
        "--request-match",
        choices=("auto", "ordinal", "full-id"),
        default="auto",
        help=(
            "auto matches stable benchmark request-id suffixes and falls "
            "back to capture ordinal"
        ),
    )
    parser.add_argument(
        "--max-tokens-per-stream",
        type=int,
        default=0,
        help="0 compares every common token; otherwise sample evenly.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="RMSE threshold reported as count/fraction. Repeat as needed.",
    )
    parser.add_argument(
        "--title",
        default="Dense vs. context-ring token-wise attention-output RMSE",
    )
    return parser.parse_args(argv)


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
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row = raw.get("row") or {}
            required = {
                "request_id",
                "request_ordinal",
                "phase",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"{rank_dir}/manifest.jsonl:{line_number} row lacks "
                    f"{sorted(missing)}"
                )
            records.append(
                TraceRecord(
                    request_id=str(row["request_id"]),
                    request_ordinal=int(row["request_ordinal"]),
                    layer_name=str(raw["layer_name"]),
                    execution_path=str(raw["execution_path"]),
                    phase=str(row["phase"]),
                    logical_start=int(raw["logical_query_start"]),
                    token_count=int(raw["query_length"]),
                    occurrence=int(raw.get("occurrence", 0)),
                    file=str(raw["file"]),
                    dtype=str(raw.get("dtype", "unknown")),
                    shape=tuple(int(value) for value in raw.get("shape", ())),
                )
            )
    return records


def benchmark_request_identity(request_id: str) -> tuple[str, int] | None:
    match = _BENCHMARK_REQUEST_ID.search(request_id)
    if match is None:
        return None
    role, dataset, repeat_index, request_index, source_index, _nonce = (
        match.groups()
    )
    identity = (
        f"benchmark:{role}:{dataset}:{repeat_index}:"
        f"{request_index}:{source_index}"
    )
    return identity, int(request_index) + 1


def request_match_identity(
    record: TraceRecord, match_mode: str
) -> tuple[str, int]:
    benchmark_identity = benchmark_request_identity(record.request_id)
    request_index = (
        benchmark_identity[1]
        if benchmark_identity is not None
        else record.request_ordinal + 1
    )
    if match_mode == "auto" and benchmark_identity is not None:
        return benchmark_identity
    if match_mode == "full-id":
        return f"request_id:{record.request_id}", request_index
    return f"ordinal:{record.request_ordinal}", request_index


def group_records(
    records: Iterable[TraceRecord],
    *,
    execution_path: str,
    phase: str,
    layer_patterns: Sequence[str],
    request_indexes: set[int],
    request_match: str,
) -> dict[StreamKey, list[TraceRecord]]:
    grouped: dict[StreamKey, list[TraceRecord]] = {}
    for record in records:
        if record.execution_path != execution_path:
            continue
        if record.phase != phase or record.occurrence != 0:
            continue
        request_key, request_index = request_match_identity(
            record, request_match
        )
        if request_indexes and request_index not in request_indexes:
            continue
        if not any(
            fnmatch.fnmatchcase(record.layer_name, pattern)
            for pattern in layer_patterns
        ):
            continue
        key = StreamKey(request_key, record.layer_name)
        grouped.setdefault(key, []).append(record)
    for key, stream_records in grouped.items():
        grouped[key] = validate_stream_records(key, stream_records)
    return grouped


def validate_stream_records(
    key: StreamKey, records: Sequence[TraceRecord]
) -> list[TraceRecord]:
    ordered = sorted(records, key=lambda row: (row.logical_start, row.file))
    previous: TraceRecord | None = None
    for record in ordered:
        if record.token_count <= 0:
            raise ValueError(f"non-positive token count in {record.file}")
        if previous is not None and record.logical_start < previous.logical_end:
            raise ValueError(
                "overlapping occurrence-0 ranges for "
                f"request={key.request_key} layer={key.layer_name}: "
                f"{previous.file} and {record.file}"
            )
        previous = record
    return ordered


def common_intervals(
    dense_records: Sequence[TraceRecord],
    ring_records: Sequence[TraceRecord],
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    dense_index = 0
    ring_index = 0
    while dense_index < len(dense_records) and ring_index < len(ring_records):
        dense = dense_records[dense_index]
        ring = ring_records[ring_index]
        start = max(dense.logical_start, ring.logical_start)
        end = min(dense.logical_end, ring.logical_end)
        if start < end:
            if intervals and intervals[-1][1] == start:
                intervals[-1] = (intervals[-1][0], end)
            else:
                intervals.append((start, end))
        if dense.logical_end <= ring.logical_end:
            dense_index += 1
        if ring.logical_end <= dense.logical_end:
            ring_index += 1
    return intervals


def select_positions(
    intervals: Sequence[tuple[int, int]], maximum: int
) -> list[int]:
    total = sum(end - start for start, end in intervals)
    if total <= 0:
        return []
    if maximum <= 0 or maximum >= total:
        return [
            position
            for start, end in intervals
            for position in range(start, end)
        ]
    if maximum == 1:
        target_indexes = [total - 1]
    else:
        target_indexes = sorted({
            round(sample * (total - 1) / (maximum - 1))
            for sample in range(maximum)
        })
    positions: list[int] = []
    interval_index = 0
    consumed = 0
    for target in target_indexes:
        while True:
            start, end = intervals[interval_index]
            interval_size = end - start
            if target < consumed + interval_size:
                positions.append(start + target - consumed)
                break
            consumed += interval_size
            interval_index += 1
    return positions


class OutputAccessor:
    def __init__(self, rank_dir: Path, records: Sequence[TraceRecord]) -> None:
        self.rank_dir = rank_dir
        self.records = list(records)
        self.starts = [record.logical_start for record in records]
        self._loaded_file = ""
        self._loaded_output: Any | None = None

    def record_at(self, position: int) -> TraceRecord:
        record_index = bisect.bisect_right(self.starts, position) - 1
        if record_index < 0:
            raise ValueError(f"no output record covers logical token {position}")
        record = self.records[record_index]
        if not (record.logical_start <= position < record.logical_end):
            raise ValueError(f"no output record covers logical token {position}")
        return record

    def output_for_record(self, record: TraceRecord) -> Any:
        import torch

        if record.file != self._loaded_file:
            payload = torch.load(
                self.rank_dir / record.file,
                map_location="cpu",
                weights_only=True,
            )
            if "output" not in payload:
                raise ValueError(f"{record.file} lacks output tensor")
            output = payload["output"]
            if int(output.shape[0]) != record.token_count:
                raise ValueError(
                    f"{record.file} output rows={int(output.shape[0])}, "
                    f"manifest query_length={record.token_count}"
                )
            self._loaded_file = record.file
            self._loaded_output = output
        assert self._loaded_output is not None
        return self._loaded_output


def token_error_metrics_batch(
    dense_output: Any, ring_output: Any
) -> dict[str, Any]:
    import torch

    if tuple(dense_output.shape) != tuple(ring_output.shape):
        raise ValueError(
            "output shape mismatch: "
            f"dense={tuple(dense_output.shape)} "
            f"ring={tuple(ring_output.shape)}"
        )
    if dense_output.dtype != ring_output.dtype:
        raise ValueError(
            "output dtype mismatch: "
            f"dense={dense_output.dtype} ring={ring_output.dtype}"
        )
    if dense_output.ndim < 2 or int(dense_output.shape[0]) <= 0:
        raise ValueError(
            "batched output must have shape [tokens, ...], got "
            f"{tuple(dense_output.shape)}"
        )
    # Attention executes in the captured runtime dtype. Only the error metric
    # uses FP32 so squaring small FP16/BF16 differences does not underflow.
    dense = dense_output.detach().to(device="cpu", dtype=torch.float32)
    ring = ring_output.detach().to(device="cpu", dtype=torch.float32)
    dense = dense.reshape(int(dense.shape[0]), -1)
    ring = ring.reshape(int(ring.shape[0]), -1)
    difference = dense - ring
    absolute_difference = torch.abs(difference)
    rmse = torch.sqrt(torch.mean(difference.square(), dim=1))
    mean_abs_error = torch.mean(absolute_difference, dim=1)
    max_abs_error = torch.max(absolute_difference, dim=1).values
    dense_rms = torch.sqrt(torch.mean(dense.square(), dim=1))
    ring_rms = torch.sqrt(torch.mean(ring.square(), dim=1))
    nrmse = rmse / torch.clamp(
        dense_rms, min=torch.finfo(torch.float32).tiny
    )
    dot_product = torch.sum(dense * ring, dim=1)
    dense_norm = torch.linalg.vector_norm(dense, dim=1)
    ring_norm = torch.linalg.vector_norm(ring, dim=1)
    norm_product = dense_norm * ring_norm
    cosine_similarity = dot_product / torch.clamp(
        norm_product, min=torch.finfo(torch.float32).tiny
    )
    both_zero = (dense_norm == 0.0) & (ring_norm == 0.0)
    cosine_similarity = torch.where(
        both_zero,
        torch.ones_like(cosine_similarity),
        cosine_similarity,
    ).clamp(min=-1.0, max=1.0)
    return {
        "element_count": int(difference.shape[1]),
        "rmse": rmse,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "dense_rms": dense_rms,
        "ring_rms": ring_rms,
        "nrmse": nrmse,
        "cosine_similarity": cosine_similarity,
    }


def token_error_metrics(dense_output: Any, ring_output: Any) -> dict[str, float]:
    if tuple(dense_output.shape) != tuple(ring_output.shape):
        raise ValueError(
            "per-token output shape mismatch: "
            f"dense={tuple(dense_output.shape)} "
            f"ring={tuple(ring_output.shape)}"
        )
    if dense_output.dtype != ring_output.dtype:
        raise ValueError(
            "per-token output dtype mismatch: "
            f"dense={dense_output.dtype} ring={ring_output.dtype}"
        )
    metrics = token_error_metrics_batch(
        dense_output.unsqueeze(0), ring_output.unsqueeze(0)
    )
    return {
        "element_count": float(metrics["element_count"]),
        "rmse": float(metrics["rmse"][0]),
        "mean_abs_error": float(metrics["mean_abs_error"][0]),
        "max_abs_error": float(metrics["max_abs_error"][0]),
        "dense_rms": float(metrics["dense_rms"][0]),
        "ring_rms": float(metrics["ring_rms"][0]),
        "nrmse": float(metrics["nrmse"][0]),
        "cosine_similarity": float(metrics["cosine_similarity"][0]),
    }


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"invalid quantile probability {probability}")
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def summarize_values(
    values: Sequence[dict[str, Any]], thresholds: Sequence[float]
) -> dict[str, Any]:
    rmse = sorted(float(row["rmse"]) for row in values)
    nrmse = sorted(float(row["nrmse"]) for row in values)
    cosine = sorted(float(row["cosine_similarity"]) for row in values)
    max_abs = sorted(float(row["max_abs_error"]) for row in values)
    element_count = sum(int(row["element_count"]) for row in values)
    squared_error_sum = sum(
        float(row["rmse"]) ** 2 * int(row["element_count"])
        for row in values
    )
    dense_squared_sum = sum(
        float(row["dense_rms"]) ** 2 * int(row["element_count"])
        for row in values
    )
    ring_squared_sum = sum(
        float(row["ring_rms"]) ** 2 * int(row["element_count"])
        for row in values
    )
    dot_product_sum = sum(
        float(row["cosine_similarity"])
        * float(row["dense_rms"])
        * float(row["ring_rms"])
        * int(row["element_count"])
        for row in values
    )
    absolute_error_sum = sum(
        float(row["mean_abs_error"]) * int(row["element_count"])
        for row in values
    )
    dense_ring_norm_product = math.sqrt(
        dense_squared_sum * ring_squared_sum
    )
    if dense_ring_norm_product > 0.0:
        global_cosine = dot_product_sum / dense_ring_norm_product
        global_cosine = min(1.0, max(-1.0, global_cosine))
    elif dense_squared_sum == 0.0 and ring_squared_sum == 0.0:
        global_cosine = 1.0
    else:
        global_cosine = 0.0
    float32_tiny = 1.1754943508222875e-38
    global_nrmse = math.sqrt(
        squared_error_sum / max(dense_squared_sum, float32_tiny)
    )
    summary: dict[str, Any] = {
        "token_count": len(values),
        "element_count": element_count,
        "rmse_global": math.sqrt(squared_error_sum / element_count),
        "rmse_mean": sum(rmse) / len(rmse),
        "rmse_p50": quantile(rmse, 0.50),
        "rmse_p95": quantile(rmse, 0.95),
        "rmse_p99": quantile(rmse, 0.99),
        "rmse_max": rmse[-1],
        "nrmse_global": global_nrmse,
        "relative_l2_error": global_nrmse,
        "nrmse_mean": sum(nrmse) / len(nrmse),
        "nrmse_p50": quantile(nrmse, 0.50),
        "nrmse_p95": quantile(nrmse, 0.95),
        "nrmse_p99": quantile(nrmse, 0.99),
        "nrmse_max": nrmse[-1],
        "cosine_similarity_global": global_cosine,
        "cosine_similarity_mean": sum(cosine) / len(cosine),
        "cosine_similarity_p01": quantile(cosine, 0.01),
        "cosine_similarity_p05": quantile(cosine, 0.05),
        "cosine_similarity_p50": quantile(cosine, 0.50),
        "cosine_similarity_min": cosine[0],
        "mean_abs_error_global": absolute_error_sum / element_count,
        "max_abs_error_p95": quantile(max_abs, 0.95),
        "max_abs_error_p99": quantile(max_abs, 0.99),
        "max_abs_error": max_abs[-1],
    }
    threshold_rows = []
    for threshold in sorted(set(thresholds)):
        count = sum(value > threshold for value in rmse)
        threshold_rows.append({
            "threshold": threshold,
            "exceed_count": count,
            "exceed_fraction": count / len(rmse),
        })
    summary["thresholds"] = threshold_rows
    return summary


def compare_stream(
    dense_rank_dir: Path,
    ring_rank_dir: Path,
    dense_records: Sequence[TraceRecord],
    ring_records: Sequence[TraceRecord],
    *,
    maximum_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    import torch

    intervals = common_intervals(dense_records, ring_records)
    common_token_count = sum(end - start for start, end in intervals)
    positions = select_positions(intervals, maximum_tokens)
    if not positions:
        return [], common_token_count
    dense_accessor = OutputAccessor(dense_rank_dir, dense_records)
    ring_accessor = OutputAccessor(ring_rank_dir, ring_records)
    rows: list[dict[str, Any]] = []
    position_index = 0
    while position_index < len(positions):
        first_position = positions[position_index]
        dense_record = dense_accessor.record_at(first_position)
        ring_record = ring_accessor.record_at(first_position)
        group_end = position_index + 1
        while group_end < len(positions):
            position = positions[group_end]
            if dense_accessor.record_at(position) != dense_record:
                break
            if ring_accessor.record_at(position) != ring_record:
                break
            group_end += 1
        group_positions = positions[position_index:group_end]
        dense_indexes = torch.tensor(
            [position - dense_record.logical_start for position in group_positions],
            dtype=torch.long,
        )
        ring_indexes = torch.tensor(
            [position - ring_record.logical_start for position in group_positions],
            dtype=torch.long,
        )
        dense_output = dense_accessor.output_for_record(dense_record).index_select(
            0, dense_indexes
        )
        ring_output = ring_accessor.output_for_record(ring_record).index_select(
            0, ring_indexes
        )
        metrics = token_error_metrics_batch(dense_output, ring_output)
        metric_columns = {
            name: tensor.tolist()
            for name, tensor in metrics.items()
            if name != "element_count"
        }
        for local_index, position in enumerate(group_positions):
            rows.append({
                "logical_position": position,
                "dense_request_id": dense_record.request_id,
                "ring_request_id": ring_record.request_id,
                "dense_dtype": dense_record.dtype,
                "ring_dtype": ring_record.dtype,
                "element_count": metrics["element_count"],
                **{
                    name: values[local_index]
                    for name, values in metric_columns.items()
                },
            })
        position_index = group_end
    return rows, common_token_count


def _short_layer_name(layer_name: str) -> str:
    parts = layer_name.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts):
            return f"layer {parts[index + 1]}"
    return layer_name


def write_values_csv(path: Path, stream_results: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "request_index",
        "layer_name",
        "phase",
        "logical_position",
        "dense_request_id",
        "ring_request_id",
        "dense_dtype",
        "ring_dtype",
        "element_count",
        "rmse",
        "mean_abs_error",
        "max_abs_error",
        "dense_rms",
        "ring_rms",
        "nrmse",
        "cosine_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in stream_results:
            for row in result["values"]:
                writer.writerow({
                    "request_index": result["request_index"],
                    "layer_name": result["layer_name"],
                    "phase": result["phase"],
                    **row,
                })


def write_summary_csv(path: Path, stream_results: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "request_index",
        "layer_name",
        "phase",
        "dense_token_count",
        "ring_token_count",
        "common_token_count",
        "compared_token_count",
        "rmse_global",
        "rmse_mean",
        "rmse_p50",
        "rmse_p95",
        "rmse_p99",
        "rmse_max",
        "nrmse_global",
        "relative_l2_error",
        "nrmse_mean",
        "nrmse_p50",
        "nrmse_p95",
        "nrmse_p99",
        "nrmse_max",
        "cosine_similarity_global",
        "cosine_similarity_mean",
        "cosine_similarity_p01",
        "cosine_similarity_p05",
        "cosine_similarity_p50",
        "cosine_similarity_min",
        "mean_abs_error_global",
        "max_abs_error_p95",
        "max_abs_error_p99",
        "max_abs_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in stream_results:
            writer.writerow({
                "request_index": result["request_index"],
                "layer_name": result["layer_name"],
                "phase": result["phase"],
                "dense_token_count": result["dense_token_count"],
                "ring_token_count": result["ring_token_count"],
                "common_token_count": result["common_token_count"],
                "compared_token_count": result["summary"]["token_count"],
                **{
                    key: result["summary"][key]
                    for key in fieldnames
                    if key in result["summary"]
                },
            })


def plot_cdf(
    path: Path, stream_results: Sequence[dict[str, Any]], title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive_values = [
        float(row["rmse"])
        for result in stream_results
        for row in result["values"]
        if float(row["rmse"]) > 0.0
    ]
    zero_plot_value = (
        max(min(positive_values) * 0.5, 1e-12)
        if positive_values
        else 1e-12
    )
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    for result in stream_results:
        exact_values = sorted(float(row["rmse"]) for row in result["values"])
        plot_values = [max(value, zero_plot_value) for value in exact_values]
        cumulative = [
            (index + 1) / len(plot_values) for index in range(len(plot_values))
        ]
        p95 = float(result["summary"]["rmse_p95"])
        label = (
            f"Req{result['request_index']} {_short_layer_name(result['layer_name'])} "
            f"(p95={p95:.3g})"
        )
        (line,) = axis.plot(plot_values, cumulative, linewidth=2.0, label=label)
        axis.scatter(
            [max(p95, zero_plot_value)],
            [0.95],
            color=line.get_color(),
            s=28,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xlim(left=zero_plot_value)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("Token-wise RMSE: dense output vs. context-ring output")
    axis.set_ylabel("Cumulative fraction of tokens")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_tokens_per_stream < 0:
        raise SystemExit("--max-tokens-per-stream must be >= 0")
    if any(index <= 0 for index in args.request_index):
        raise SystemExit("--request-index values are one-based and must be positive")
    if any(threshold < 0 for threshold in args.threshold):
        raise SystemExit("--threshold values must be non-negative")

    dense_rank_dir = resolve_rank_dir(Path(args.dense_trace_root), args.rank)
    ring_rank_dir = resolve_rank_dir(Path(args.ring_trace_root), args.rank)
    dense_records = load_manifest(dense_rank_dir)
    ring_records = load_manifest(ring_rank_dir)
    layer_patterns = args.layer or ["*"]
    request_indexes = set(args.request_index)
    dense_groups = group_records(
        dense_records,
        execution_path=args.dense_execution_path,
        phase=args.phase,
        layer_patterns=layer_patterns,
        request_indexes=request_indexes,
        request_match=args.request_match,
    )
    ring_groups = group_records(
        ring_records,
        execution_path=args.ring_execution_path,
        phase=args.phase,
        layer_patterns=layer_patterns,
        request_indexes=request_indexes,
        request_match=args.request_match,
    )
    shared_keys = sorted(
        set(dense_groups) & set(ring_groups),
        key=lambda key: (key.request_key, key.layer_name),
    )
    if not shared_keys:
        raise SystemExit(
            "no request/layer streams are shared by dense and ring traces"
        )

    thresholds = args.threshold or [1e-3, 1e-2]
    stream_results: list[dict[str, Any]] = []
    for key in shared_keys:
        dense_identity = request_match_identity(
            dense_groups[key][0], args.request_match
        )
        ring_identity = request_match_identity(
            ring_groups[key][0], args.request_match
        )
        if dense_identity[1] != ring_identity[1]:
            raise SystemExit(
                "matched stream has inconsistent one-based request indexes: "
                f"dense={dense_identity[1]} ring={ring_identity[1]} "
                f"key={key.request_key}"
            )
        request_index = dense_identity[1]
        dense_token_count = sum(
            record.token_count for record in dense_groups[key]
        )
        ring_token_count = sum(
            record.token_count for record in ring_groups[key]
        )
        values, common_token_count = compare_stream(
            dense_rank_dir,
            ring_rank_dir,
            dense_groups[key],
            ring_groups[key],
            maximum_tokens=args.max_tokens_per_stream,
        )
        if common_token_count != ring_token_count:
            raise SystemExit(
                "dense trace does not cover every context-ring token for "
                f"request={request_index} layer={key.layer_name}: "
                f"ring={ring_token_count} common={common_token_count}"
            )
        if not values:
            continue
        summary = summarize_values(values, thresholds)
        result = {
            "request_index": request_index,
            "request_match_key": key.request_key,
            "dense_request_ordinal": dense_groups[key][0].request_ordinal,
            "ring_request_ordinal": ring_groups[key][0].request_ordinal,
            "layer_name": key.layer_name,
            "phase": args.phase,
            "dense_token_count": dense_token_count,
            "ring_token_count": ring_token_count,
            "common_token_count": common_token_count,
            "sampling": (
                "all_common_tokens"
                if args.max_tokens_per_stream == 0
                or len(values) == common_token_count
                else "evenly_spaced"
            ),
            "summary": summary,
            "values": values,
        }
        stream_results.append(result)
        print(
            f"request={request_index} layer={key.layer_name} "
            f"common={common_token_count} compared={len(values)} "
            f"p50={summary['rmse_p50']:.12g} "
            f"p95={summary['rmse_p95']:.12g} "
            f"p99={summary['rmse_p99']:.12g} "
            f"max={summary['rmse_max']:.12g} "
            f"cos={summary['cosine_similarity_global']:.12g}"
        )
    if not stream_results:
        raise SystemExit("shared streams contain no common logical token positions")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / "token_rmse_values.csv"
    summary_csv_path = output_dir / "token_rmse_summary.csv"
    summary_json_path = output_dir / "token_rmse_summary.json"
    plot_path = output_dir / "token_rmse_cdf.png"
    write_values_csv(values_path, stream_results)
    write_summary_csv(summary_csv_path, stream_results)
    summary_payload = {
        "schema_version": 1,
        "comparison": "captured_dense_output_vs_captured_context_ring_output",
        "dense_trace_root": str(dense_rank_dir),
        "ring_trace_root": str(ring_rank_dir),
        "dense_execution_path": args.dense_execution_path,
        "ring_execution_path": args.ring_execution_path,
        "phase": args.phase,
        "error_accumulation_dtype": "torch.float32",
        "attention_reexecution": False,
        "stream_count": len(stream_results),
        "streams": [
            {key: value for key, value in result.items() if key != "values"}
            for result in stream_results
        ],
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_cdf(plot_path, stream_results, args.title)
    print(f"wrote {values_path}")
    print(f"wrote {summary_csv_path}")
    print(f"wrote {summary_json_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
