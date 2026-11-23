#!/usr/bin/env python3
"""Audit captured Native/Ring outputs with full-history sampled FP64 references.

This tool does NOT execute a Native or Ring GPU kernel. It reads immutable
captures, compares every common Ring-region output token, and reconstructs a
separate CPU FP64 reference from each run's own rounded QKV. A common-input
claim is made only after exact Q and full causal KV equality is verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compare_attention_fp64_reference import dense_causal_attention_fp64
from compare_dense_ring_token_rmse import (
    StreamKey,
    TraceRecord,
    load_manifest,
    request_match_identity,
    resolve_rank_dir,
    select_positions,
    validate_stream_records,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-trace-root", type=Path, required=True)
    parser.add_argument("--ring-trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=2, help="One-based.")
    parser.add_argument("--layers", default="0-15", help="E.g. 0-15 or 0,7,15.")
    parser.add_argument("--samples-per-layer", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--source-label", default="historical captured implementation")
    return parser.parse_args(argv)


def layer_number(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if not match:
        raise ValueError(f"cannot identify layer number: {name}")
    return int(match.group(1))


def parse_layers(value: str) -> list[int]:
    result: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-"))
            if first < 0 or last < first:
                raise ValueError("invalid layer range")
            result.update(range(first, last + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0:
        raise ValueError("layers must be nonnegative")
    return sorted(result)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def groups_for_request(
    root: Path, request_index: int, layers: list[int]
) -> dict[StreamKey, list[TraceRecord]]:
    groups: dict[StreamKey, list[TraceRecord]] = {}
    for row in load_manifest(root):
        if row.phase != "prefill" or row.occurrence != 0:
            continue
        identity, index = request_match_identity(row, "auto")
        if index != request_index or layer_number(row.layer_name) not in layers:
            continue
        if not identity.startswith("benchmark:"):
            raise ValueError("stable benchmark request identity is required")
        groups.setdefault(StreamKey(identity, row.layer_name), []).append(row)
    for key, rows in groups.items():
        groups[key] = validate_stream_records(key, rows)
    if sorted(layer_number(key.layer_name) for key in groups) != layers:
        raise ValueError(f"requested layer coverage missing or ambiguous under {root}")
    if len({key.request_key for key in groups}) != 1:
        raise ValueError("multiple semantic requests selected")
    return groups


def load_stream(root: Path, rows: list[TraceRecord]) -> dict[str, Any]:
    """Load one layer with complete causal history; never infer absent KV."""
    keys = ("query", "key", "value", "output")
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    ring_parts = []
    cursor = 0
    scale = None
    reference_shapes = None
    reference_dtype = None
    for row in rows:
        if row.logical_start != cursor:
            raise ValueError(f"history gap/overlap at {cursor}: {row.file}")
        path = (root / row.file).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("capture file escapes trace directory")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if any(key not in payload for key in keys):
            raise ValueError(f"missing QKV/output in {path}")
        q, k, v, output = (payload[key] for key in keys)
        if any(not isinstance(t, torch.Tensor) for t in (q, k, v, output)):
            raise ValueError(f"non-tensor payload in {path}")
        if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
            raise ValueError(f"invalid QKV shape in {path}")
        if q.shape[0] != row.token_count or k.shape[0] != row.token_count:
            raise ValueError(f"manifest/payload length mismatch in {path}")
        if q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
            raise ValueError(f"invalid grouped-query geometry in {path}")
        if output.shape[0] != row.token_count or output.numel() != q.numel():
            raise ValueError(f"output/query shape mismatch in {path}")
        output = output.reshape_as(q)
        if len({t.dtype for t in (q, k, v, output)}) != 1:
            raise ValueError(f"inconsistent captured dtype in {path}")
        if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (q, k, v, output)):
            raise ValueError(f"nonfinite or non-floating captured tensor in {path}")
        shapes = (tuple(q.shape[1:]), tuple(k.shape[1:]))
        if reference_shapes is not None and shapes != reference_shapes:
            raise ValueError("head geometry changed within a layer")
        if reference_dtype is not None and q.dtype != reference_dtype:
            raise ValueError("dtype changed within a layer")
        reference_shapes, reference_dtype = shapes, q.dtype
        current_scale = payload.get("softmax_scale")
        if current_scale is None or not math.isfinite(float(current_scale)):
            raise ValueError("recorded finite softmax_scale is required")
        if scale is not None and float(current_scale) != scale:
            raise ValueError("softmax scale changed within a layer")
        scale = float(current_scale)
        for key, tensor in zip(keys, (q, k, v, output)):
            # Captured tensors may be views; clone only the retained view extent.
            parts[key].append(tensor.contiguous())
        if row.execution_path not in ("native_hbm", "context_ring"):
            raise ValueError(f"unsupported captured path: {row.execution_path}")
        ring_parts.append(torch.full((row.token_count,), row.execution_path == "context_ring"))
        cursor = row.logical_end
    if not cursor:
        raise ValueError("empty capture stream")
    result: dict[str, Any] = {key: torch.cat(values) for key, values in parts.items()}
    result.update(ring_mask=torch.cat(ring_parts), softmax_scale=scale, tokens=cursor)
    return result


@dataclass
class ErrorAccumulator:
    count: int = 0
    squared_error: float = 0.0
    reference_squared: float = 0.0
    actual_squared: float = 0.0
    dot: float = 0.0
    absolute_error: float = 0.0
    max_absolute: float = 0.0

    def add(self, reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        if reference.shape != actual.shape:
            raise ValueError("metric shape mismatch")
        ref, act = reference.double(), actual.double()
        if not torch.isfinite(ref).all() or not torch.isfinite(act).all():
            raise ValueError("nonfinite metric input")
        diff = act - ref
        self.count += diff.numel()
        self.squared_error += float(diff.square().sum())
        self.reference_squared += float(ref.square().sum())
        self.actual_squared += float(act.square().sum())
        self.dot += float((ref * act).sum())
        self.absolute_error += float(diff.abs().sum())
        self.max_absolute = max(self.max_absolute, float(diff.abs().max()))
        return diff

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("empty metric population")
        denominator = math.sqrt(self.reference_squared * self.actual_squared)
        cosine = self.dot / denominator if denominator else None
        if self.reference_squared == self.actual_squared == 0:
            cosine = 1.0
        nrmse = math.sqrt(self.squared_error / self.reference_squared) if self.reference_squared else (0.0 if self.squared_error == 0 else None)
        return {
            "element_count": self.count,
            "rmse": math.sqrt(self.squared_error / self.count),
            "nrmse_percent": None if nrmse is None else 100 * nrmse,
            "cosine": None if cosine is None else min(1.0, max(-1.0, cosine)),
            "mean_absolute_error": self.absolute_error / self.count,
            "max_absolute_error": self.max_absolute,
        }


def error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    accumulator = ErrorAccumulator()
    accumulator.add(reference, actual)
    return accumulator.result()


def same_input_mask(
    native: dict[str, Any], ring: dict[str, Any], positions: np.ndarray
) -> np.ndarray:
    """Equality requires the whole causal prefix, not only current-token KV."""
    if native["softmax_scale"] != ring["softmax_scale"]:
        raise ValueError("Native/Ring softmax scale mismatch")
    for key in ("query", "key", "value", "output"):
        if native[key].shape != ring[key].shape or native[key].dtype != ring[key].dtype:
            raise ValueError(f"Native/Ring {key} geometry or dtype mismatch")
    kv_equal = (native["key"] == ring["key"]).flatten(1).all(1)
    kv_equal &= (native["value"] == ring["value"]).flatten(1).all(1)
    prefix_equal = (~kv_equal).cumsum(0) == 0
    indexes = torch.from_numpy(positions.copy())
    query_equal = (native["query"][indexes] == ring["query"][indexes]).flatten(1).all(1)
    return (prefix_equal[indexes] & query_equal).numpy()


def full_output_metrics(
    native: dict[str, Any], ring: dict[str, Any], positions: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, ErrorAccumulator]:
    accumulator = ErrorAccumulator()
    rmse_parts, relative_parts = [], []
    for start in range(0, len(positions), 1024):
        indexes = torch.from_numpy(positions[start:start + 1024].copy())
        ref = native["output"][indexes].flatten(1)
        act = ring["output"][indexes].flatten(1)
        difference = accumulator.add(ref, act)
        rmse = difference.square().mean(1).sqrt()
        ref_rms = ref.double().square().mean(1).sqrt()
        relative = torch.where(ref_rms > 0, rmse / ref_rms, torch.where(rmse == 0, 0.0, float("nan")))
        rmse_parts.append(rmse.numpy())
        relative_parts.append(relative.numpy() * 100)
    rmse_values, relative_values = np.concatenate(rmse_parts), np.concatenate(relative_parts)
    result = accumulator.result()
    result.update(
        compared_tokens=len(positions),
        token_rmse_p50=float(np.quantile(rmse_values, 0.50)),
        token_rmse_p95=float(np.quantile(rmse_values, 0.95)),
        token_rmse_p99=float(np.quantile(rmse_values, 0.99)),
        token_rmse_max=float(rmse_values.max()),
        token_rmse_gt_1e_3_count=int(np.count_nonzero(rmse_values > 1e-3)),
        undefined_token_nrmse_count=int(np.isnan(relative_values).sum()),
    )
    return result, rmse_values, relative_values, accumulator


def sampled_references(
    native: dict[str, Any], ring: dict[str, Any], positions: np.ndarray
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    identical = same_input_mask(native, ring, positions)
    tensors: dict[str, torch.Tensor] = {}
    for label, stream in (("native", native), ("ring", ring)):
        key, value = stream["key"].double(), stream["value"].double()
        references = []
        for position in positions:
            references.append(dense_causal_attention_fp64(
                stream["query"][int(position)], key, value,
                query_position=int(position), softmax_scale=stream["softmax_scale"],
            ))
        tensors[f"{label}_fp64"] = torch.stack(references)
        tensors[f"{label}_output"] = stream["output"][torch.from_numpy(positions.copy())]
        del key, value
    rows = []
    for index, position in enumerate(positions):
        no, ro = tensors["native_output"][index], tensors["ring_output"][index]
        nr, rr = tensors["native_fp64"][index], tensors["ring_fp64"][index]
        nm, rm = error_metrics(nr, no), error_metrics(rr, ro)
        if identical[index] and not torch.equal(nr, rr):
            raise AssertionError("identical input produced different deterministic CPU references")
        rows.append({
            "logical_position": int(position),
            "causal_context_tokens": int(position) + 1,
            "qkv_identical": bool(identical[index]),
            "native_rmse_vs_own_fp64": nm["rmse"],
            "ring_rmse_vs_own_fp64": rm["rmse"],
            "native_nrmse_percent_vs_own_fp64": nm["nrmse_percent"],
            "ring_nrmse_percent_vs_own_fp64": rm["nrmse_percent"],
            "captured_pair_rmse": error_metrics(no, ro)["rmse"],
            "fp64_reference_gap_rmse": error_metrics(nr, rr)["rmse"],
            "local_error_difference_rmse": float(((ro.double() - rr) - (no.double() - nr)).square().mean().sqrt()),
        })
    native_metrics = error_metrics(tensors["native_fp64"], tensors["native_output"])
    ring_metrics = error_metrics(tensors["ring_fp64"], tensors["ring_output"])
    denominator = native_metrics["rmse"]
    summary = {
        "sample_count": len(positions),
        "same_input_samples": int(identical.sum()),
        "native_vs_own_fp64": native_metrics,
        "ring_vs_own_fp64": ring_metrics,
        "ring_to_native_local_rmse_ratio": ring_metrics["rmse"] / denominator if denominator else None,
        "captured_pair": error_metrics(tensors["native_output"], tensors["ring_output"]),
        "reference_gap": error_metrics(tensors["native_fp64"], tensors["ring_fp64"]),
        "local_error_difference_rmse": float(((tensors["ring_output"].double() - tensors["ring_fp64"]) - (tensors["native_output"].double() - tensors["native_fp64"])).square().mean().sqrt()),
    }
    return tensors, rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot publish empty CSV")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numpy_output(tensor: torch.Tensor) -> np.ndarray:
    """NumPy lacks BF16; FP32 represents every captured BF16 value exactly."""
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy()
    return tensor.numpy()


def display_number(value: float | None, spec: str = ".6g") -> str:
    return "undefined (zero reference)" if value is None else format(value, spec)


def plot_results(output: Path, results: list[dict[str, Any]], source_label: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "savefig.dpi": 180})
    blue, orange = "#1f77b4", "#ff7f0e"
    filenames = []

    def save(fig: Any, name: str) -> None:
        fig.text(.5, .01, source_label + " | captured Prefill outputs; CPU FP64 reference", ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, .04, 1, 1))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
        filenames.append(f"{name}.png")

    layers = [row["layer"] for row in results]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(layers, [row["full_output"]["nrmse_percent"] for row in results], "o-", color=orange)
    ax.set(xlabel="Attention layer", ylabel="Global NRMSE vs Native (%)", title="Captured Ring vs Native output — all common Ring-region tokens")
    ax.set_xticks(layers)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    save(fig, "A_layer_output_nrmse")

    last = results[-1]
    with np.load(output / last["token_metrics_file"]) as stored:
        values = np.sort(stored["token_rmse"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    positive = values > 0
    if positive.any():
        ax.plot(values[positive], np.arange(1, len(values) + 1)[positive] / len(values), color=orange)
        ax.set_xscale("log")
    else:
        ax.plot([0], [1], "o", color=orange)
    for probability, marker in ((.95, "o"), (.99, "s")):
        value = float(np.quantile(values, probability))
        ax.plot(value, probability, marker, color="black", label=f"p{int(probability * 100)} = {value:.3g}")
    ax.set(xlabel="Token-wise RMSE vs Native", ylabel="Cumulative fraction of compared tokens", ylim=(0, 1.01), title=f"Layer {last['layer']}: {len(values):,} captured output tokens")
    ax.legend()
    ax.grid(alpha=.25)
    save(fig, "B_final_layer_token_rmse_cdf")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, color in (("native", blue), ("ring", orange)):
        ax.plot(layers, [row["sampled_fp64"][f"{label}_vs_own_fp64"]["rmse"] for row in results], "o-", color=color, label=f"{label.title()} vs its own QKV FP64")
    ax.set(xlabel="Attention layer", ylabel="Sampled output RMSE", yscale="log", title="Local attention error — separate own-input FP64 references")
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(alpha=.25)
    save(fig, "C_layer_own_input_fp64_rmse")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, color in (("native", blue), ("ring", orange)):
        ax.plot([row["causal_context_tokens"] for row in last["samples"]], [row[f"{label}_rmse_vs_own_fp64"] for row in last["samples"]], "o-", color=color, label=label.title())
    ax.set(xlabel="Causal context at sampled query (tokens)", ylabel="RMSE vs own-input FP64", title=f"Layer {last['layer']}: selected queries of one captured prompt")
    ax.legend()
    ax.grid(alpha=.25)
    save(fig, "D_final_layer_fp64_error_by_position")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key, label, color in (("captured_pair", "Captured output difference", orange), ("reference_gap", "Own-input FP64 reference difference", blue)):
        ax.plot(layers, [row["sampled_fp64"][key]["rmse"] for row in results], "o-", label=label, color=color)
    ax.plot(layers, [row["sampled_fp64"]["local_error_difference_rmse"] for row in results], "s--", label="Difference of local rounding errors", color="#2ca02c")
    ax.set(xlabel="Attention layer", ylabel="RMSE on the same sampled positions", title="Output difference and reference-input difference (not additive magnitudes)")
    ax.set_xticks(layers)
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    save(fig, "E_sampled_error_components")

    same = next((row for row in results if row["sampled_fp64"]["same_input_samples"]), None)
    if same is not None:
        samples = [row for row in same["samples"] if row["qkv_identical"]]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for label, color in (("native", blue), ("ring", orange)):
            values = np.sort([row[f"{label}_rmse_vs_own_fp64"] for row in samples])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label.title(), color=color)
        ax.set(xlabel="RMSE vs common-input FP64", ylabel="Cumulative fraction of sampled queries", ylim=(0, 1.01), title=f"Layer {same['layer']}: {len(samples)} verified identical-QKV captured samples")
        ax.legend()
        ax.grid(alpha=.25)
        save(fig, "F_verified_same_input_fp64_cdf")

        with np.load(output / same["sample_outputs_file"]) as stored:
            valid = np.flatnonzero(stored["qkv_identical"])
            index = int(valid[-1])
            position = int(stored["positions"][index])
            native = stored["native_output"][index, 0].astype(np.float64)
            ring = stored["ring_output"][index, 0].astype(np.float64)
            reference = stored["native_fp64"][index, 0]
        write_csv(output / "same_input_output_example.csv", [{
            "layer": same["layer"], "logical_position": position, "head": 0,
            "coordinate": coordinate, "native_output": float(n),
            "ring_output": float(r), "common_fp64": float(f),
            "native_minus_fp64": float(n - f), "ring_minus_fp64": float(r - f),
        } for coordinate, (n, r, f) in enumerate(zip(native, ring, reference))])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(reference, "k--", label="FP64 common reference", linewidth=2)
        ax.plot(native, color=blue, label="Captured Native", alpha=.85)
        ax.plot(ring, color=orange, label="Captured Ring", alpha=.85)
        ax.set(xlabel="Output coordinate within attention head 0", ylabel="Attention output value", title=f"Same-input example: layer {same['layer']}, query {position}, head 0")
        ax.legend()
        ax.grid(alpha=.25)
        save(fig, "G_same_input_output_overlay")

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(native - reference, color=blue, label="Native minus FP64")
        ax.plot(ring - reference, color=orange, label="Ring minus FP64")
        ax.set(xlabel="Output coordinate within attention head 0", ylabel="Signed error", title=f"Same-input example errors: layer {same['layer']}, query {position}, head 0")
        ax.legend()
        ax.grid(alpha=.25)
        save(fig, "H_same_input_output_errors")
    return filenames


def write_html(output: Path, summary: dict[str, Any]) -> None:
    rows = summary["layers"]
    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["layer"], row["full_output"]["compared_tokens"],
            display_number(row['full_output']['nrmse_percent'], ".5f"),
            f"{row['full_output']['token_rmse_p99']:.6g}",
            f"{row['sampled_fp64']['native_vs_own_fp64']['rmse']:.6g}",
            f"{row['sampled_fp64']['ring_vs_own_fp64']['rmse']:.6g}",
            f"{row['sampled_fp64']['same_input_samples']}/{row['sampled_fp64']['sample_count']}",
        )) + "</tr>" for row in rows
    )
    pictures = "".join(f'<figure><a href="{name}"><img src="{name}" alt="{name}"></a><figcaption>{name}</figcaption></figure>' for name in summary["figures"])
    text = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Context Ring attention 출력과 FP64 수치 검증</title>
<style>body{{max-width:1100px;margin:36px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#182331}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ccd2db;text-align:right}}img{{max-width:100%}}figure{{margin:30px 0}}.note{{background:#fff3d7;padding:16px}}code{{overflow-wrap:anywhere}}</style>
<h1>Context Ring attention 출력과 FP64 수치 검증</h1>
<p>대상: {html.escape(summary['source_label'])}. 같은 요청과 토큰 위치의 저장 출력을 비교하고, 각 실행의 저장된 Q/K/V로 CPU FP64 causal attention을 계산했습니다.</p>
<p>전체 출력 비교: layer마다 {rows[0]['full_output']['compared_tokens']:,}개 Ring 구간 토큰. FP64 검증: layer마다 {rows[0]['sampled_fp64']['sample_count']}개 query를 선택하며 각 query의 과거 KV 전체를 사용합니다.</p>
<p class="note">이 결과는 실제 캡처 출력의 수치 검증입니다. 새로운 GPU kernel 실행, 현재 vendor_v8의 검증, Decode 또는 생성 답변 정답률 평가는 아닙니다. Q/K/V가 다른 layer는 각자 FP64 기준을 사용하며, 동일 Q/K/V가 검증된 표본만 공통 기준 비교로 표시합니다. NRMSE 0.3%나 RMSE 0.001은 합격 기준으로 설정하지 않았습니다.</p>
<h2>평가 범위와 결과</h2><table><thead><tr><th>Layer</th><th>전체 비교 토큰</th><th>NRMSE %</th><th>토큰 RMSE p99</th><th>Native 자체 FP64 RMSE</th><th>Ring 자체 FP64 RMSE</th><th>동일 입력 표본</th></tr></thead><tbody>{table}</tbody></table>
<p>NRMSE = ||Ring−Native||₂ / ||Native||₂ × 100. FP64는 캡처 당시 FP16/BF16으로 반올림된 입력에 대한 고정밀 계산이며, 전체 모델을 FP64로 실행한 결과가 아닙니다. E 그래프의 세 RMSE는 벡터 차이의 크기이므로 서로 더하거나 비율로 인과 기여도를 해석할 수 없습니다.</p>
<p><a href="summary.json">완료 및 출처 JSON</a> · <a href="layer_summary.csv">Layer 요약 CSV</a> · <a href="sample_metrics.csv">FP64 표본 CSV</a>. 동일 입력 표본이 있으면 same_input_output_example.csv에 G/H 그래프의 실제 3가지 출력값을 저장합니다. sampled_outputs/의 NPZ에는 native_output, ring_output, native_fp64, ring_fp64, positions, qkv_identical을 저장했습니다. token_metrics/에는 전체 비교 토큰의 오차가 있습니다. BF16 캡처의 NPZ 출력은 값 손실 없이 FP32로 저장하며, 원래 dtype은 요약 JSON에 명시합니다.</p>
<h2>그래프</h2>{pictures}</html>"""
    (output / "report.html").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.samples_per_layer <= 0 or args.threads <= 0 or args.request_index <= 0:
        raise ValueError("sample count, threads and request index must be positive")
    torch.set_num_threads(args.threads)
    layers = parse_layers(args.layers)
    native_root = resolve_rank_dir(args.native_trace_root, args.rank)
    ring_root = resolve_rank_dir(args.ring_trace_root, args.rank)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    native_groups = groups_for_request(native_root, args.request_index, layers)
    ring_groups = groups_for_request(ring_root, args.request_index, layers)
    if set(native_groups) != set(ring_groups):
        raise ValueError("Native/Ring semantic request/layer sets differ")
    output.mkdir(parents=True, exist_ok=True)
    (output / "sampled_outputs").mkdir()
    (output / "token_metrics").mkdir()
    results, csv_rows, sample_csv = [], [], []
    total = ErrorAccumulator()
    for key in sorted(native_groups, key=lambda item: layer_number(item.layer_name)):
        number = layer_number(key.layer_name)
        print(f"[layer {number}] loading captured QKV/output", flush=True)
        native = load_stream(native_root, native_groups[key])
        ring = load_stream(ring_root, ring_groups[key])
        if native["tokens"] != ring["tokens"]:
            raise ValueError("full prompt history endpoints differ")
        if native["ring_mask"].any():
            raise ValueError("Native reference capture contains Ring execution")
        positions = np.flatnonzero(ring["ring_mask"].numpy()).astype(np.int64)
        if not len(positions):
            raise ValueError("candidate has no Context Ring output")
        # Choose uniformly in the compared population, including both endpoints.
        indexes = select_positions([(0, len(positions))], args.samples_per_layer)
        selected = positions[np.asarray(indexes, dtype=np.int64)]
        same_input_mask(native, ring, selected)  # Geometry/scale validation first.
        full, rmses, relatives, accumulator = full_output_metrics(native, ring, positions)
        for field in ("count", "squared_error", "reference_squared", "actual_squared", "dot", "absolute_error"):
            setattr(total, field, getattr(total, field) + getattr(accumulator, field))
        total.max_absolute = max(total.max_absolute, accumulator.max_absolute)
        token_file = f"token_metrics/layer_{number:02d}.npz"
        np.savez_compressed(output / token_file, positions=positions, token_rmse=rmses, token_nrmse_percent=relatives)
        print(f"[layer {number}] {len(positions):,} full-token pairs; {len(selected)} FP64 queries per run", flush=True)
        tensors, samples, fp64 = sampled_references(native, ring, selected)
        sample_file = f"sampled_outputs/layer_{number:02d}.npz"
        np.savez_compressed(output / sample_file, positions=selected, qkv_identical=np.array([row["qkv_identical"] for row in samples]), **{name: numpy_output(tensor) for name, tensor in tensors.items()})
        row = {"layer": number, "layer_name": key.layer_name, "request_key": key.request_key,
               "captured_dtype": str(native["query"].dtype), "history_tokens": native["tokens"],
               "full_output": full, "sampled_fp64": fp64, "samples": samples,
               "token_metrics_file": token_file, "sample_outputs_file": sample_file}
        results.append(row)
        csv_rows.append({"layer": number, **full,
                         "fp64_sample_count": fp64["sample_count"], "same_input_samples": fp64["same_input_samples"],
                         "native_own_fp64_rmse": fp64["native_vs_own_fp64"]["rmse"],
                         "ring_own_fp64_rmse": fp64["ring_vs_own_fp64"]["rmse"],
                         "ring_to_native_local_rmse_ratio": fp64["ring_to_native_local_rmse_ratio"]})
        sample_csv.extend({"layer": number, **sample} for sample in samples)
        print(f"[layer {number}] NRMSE={display_number(full['nrmse_percent'], '.6f')}% local Ring/Native={fp64['ring_to_native_local_rmse_ratio']} same-input={fp64['same_input_samples']}/{len(selected)}", flush=True)
        del native, ring, tensors
    write_csv(output / "layer_summary.csv", csv_rows)
    write_csv(output / "sample_metrics.csv", sample_csv)
    figures = plot_results(output, results, args.source_label)
    summary = {
        "schema_version": 1, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_label": args.source_label, "native_trace_root": str(native_root), "ring_trace_root": str(ring_root),
        "source_manifest_sha256": {"native": sha256(native_root / "manifest.jsonl"), "ring": sha256(ring_root / "manifest.jsonl")},
        "analysis_code_sha256": sha256(Path(__file__)), "python": sys.version, "torch": torch.__version__,
        "reference_device": "cpu", "reference_dtype": "float64", "error_reduction_dtype": "float64",
        "actual_gpu_kernel_reexecution": False, "semantic_answer_accuracy_evaluated": False,
        "sample_selection": "evenly spaced common Ring-region query positions including endpoints; entire causal KV prefix per query",
        "reference_identity": "separate own-QKV reference; common-input subset only when exact Q and full causal KV match",
        "source_payload_hashes_verified": False,
        "aggregate_full_output": total.result(), "layers": results, "figures": figures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_html(output, summary)
    # Completion is published last; errors/cancellation leave no complete marker.
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[DONE] {output / 'report.html'}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
