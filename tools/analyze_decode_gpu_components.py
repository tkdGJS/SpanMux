#!/usr/bin/env python3
"""Summarize vendor_v8 Decode GPU component traces across SM partitions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


COMPONENTS = (
    "decode_stream_pipeline_ms",
    "gpu_model_forward_ms",
    "qkv_proj_ms",
    "attention_ms",
    "o_proj_ms",
    "mlp_gate_up_proj_ms",
    "mlp_activation_ms",
    "mlp_down_proj_ms",
    "attention_stack_ms",
    "mlp_total_ms",
    "gpu_residual_unattributed_ms",
    "logits_postprocess_ms",
    "sampling_ms",
    "decode_stream_interstage_residual_ms",
)
ATOMIC_COMPONENTS = (
    "qkv_proj_ms",
    "attention_ms",
    "o_proj_ms",
    "mlp_gate_up_proj_ms",
    "mlp_activation_ms",
    "mlp_down_proj_ms",
    "gpu_residual_unattributed_ms",
    "logits_postprocess_ms",
    "sampling_ms",
    "decode_stream_interstage_residual_ms",
)
GROUP_FIELDS = (
    "prefill_mode",
    "sm_profile_id",
    "decode_sms",
    "prefill_sms",
    "decode_context_bucket_tokens",
    "decode_request_count",
    "context_ring_decode_request_count",
    "prefill_overlap_candidate",
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def infer_mode_and_profile(path: Path, artifact_root: Path) -> tuple[str, str]:
    relative = path.relative_to(artifact_root)
    parts = relative.parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return "unknown", path.parent.name


def load_records(
    artifact_root: Path,
    *,
    context_bucket_tokens: int,
    include_warmup: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(artifact_root.rglob("decode_gpu_components.jsonl")):
        mode, path_profile = infer_mode_and_profile(path, artifact_root)
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if payload.get("trace_schema") != "pd_decode_gpu_components_v1":
                    continue
                trial_phase = str(payload.get("trial_phase", ""))
                if not include_warmup and trial_phase == "warmup":
                    continue
                context_tokens = integer(payload.get("decode_context_max_tokens"))
                context_bucket = (
                    ((context_tokens + context_bucket_tokens // 2)
                     // context_bucket_tokens)
                    * context_bucket_tokens
                )
                record: dict[str, Any] = {
                    "source_file": str(path),
                    "source_line": line_number,
                    "prefill_mode": mode,
                    "sm_profile_id": str(
                        payload.get("sm_profile_id") or path_profile
                    ),
                    "decode_sms": integer(payload.get("decode_sms")),
                    "prefill_sms": integer(payload.get("prefill_sms")),
                    "iteration_id": integer(payload.get("iteration_id"), -1),
                    "trace_sequence": integer(payload.get("trace_sequence")),
                    "trace_every_n": integer(payload.get("trace_every_n"), 1),
                    "trace_includes_layers": integer(
                        payload.get("trace_includes_layers")
                    ),
                    "trial_phase": trial_phase,
                    "phase_slot": integer(payload.get("phase_slot")),
                    "num_reqs": integer(payload.get("num_reqs")),
                    "num_tokens": integer(payload.get("num_tokens")),
                    "decode_request_count": integer(
                        payload.get("decode_request_count"),
                        integer(payload.get("num_reqs")),
                    ),
                    "decode_context_min_tokens": integer(
                        payload.get("decode_context_min_tokens")
                    ),
                    "decode_context_max_tokens": context_tokens,
                    "decode_context_mean_tokens": finite_float(
                        payload.get("decode_context_mean_tokens")
                    ),
                    "decode_context_bucket_tokens": context_bucket,
                    "context_ring_decode_request_count": integer(
                        payload.get("context_ring_decode_request_count")
                    ),
                    "prefill_inflight_at_decode_submit": integer(
                        payload.get("prefill_inflight_at_decode_submit")
                    ),
                    "prefill_not_ready_at_decode_submit": integer(
                        payload.get("prefill_not_ready_at_decode_submit")
                    ),
                    "paired_prefill_submit": integer(
                        payload.get("paired_prefill_submit")
                    ),
                    "prefill_overlap_candidate": integer(
                        payload.get("prefill_overlap_candidate")
                    ),
                    "green_context_active": finite_float(
                        payload.get("green_context_active")
                    ),
                    "green_sm_count_verified": finite_float(
                        payload.get("green_sm_count_verified")
                    ),
                    "component_trace_forced_eager": finite_float(
                        payload.get("component_trace_forced_eager")
                    ),
                    "matched_modules": integer(payload.get("matched_modules")),
                    "num_layers": integer(payload.get("num_layers")),
                    "num_records": integer(payload.get("num_records")),
                    "decode_request_ids": ",".join(
                        str(item)
                        for item in payload.get("decode_request_ids", [])
                    ),
                }
                for component in COMPONENTS:
                    record[component] = finite_float(payload.get(component))
                records.append(record)
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in GROUP_FIELDS)].append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        row = dict(zip(GROUP_FIELDS, key))
        row["sample_count"] = len(group)
        row["green_context_active_fraction"] = statistics.fmean(
            finite_float(item["green_context_active"]) for item in group
        )
        row["green_sm_count_verified_fraction"] = statistics.fmean(
            finite_float(item["green_sm_count_verified"]) for item in group
        )
        row["matched_modules_min"] = min(
            integer(item["matched_modules"]) for item in group
        )
        row["num_records_mean"] = statistics.fmean(
            integer(item["num_records"]) for item in group
        )
        row["num_layers_mean"] = statistics.fmean(
            integer(item["num_layers"]) for item in group
        )
        for component in COMPONENTS:
            values = [finite_float(item[component]) for item in group]
            row[f"{component}_mean"] = statistics.fmean(values)
            row[f"{component}_p50"] = percentile(values, 0.50)
            row[f"{component}_p90"] = percentile(values, 0.90)
            row[f"{component}_p95"] = percentile(values, 0.95)
            row[f"{component}_p99"] = percentile(values, 0.99)
        forward_mean = finite_float(row["gpu_model_forward_ms_mean"])
        pipeline_mean = finite_float(row["decode_stream_pipeline_ms_mean"])
        for component in ATOMIC_COMPONENTS:
            denominator = (
                pipeline_mean
                if component
                in {
                    "logits_postprocess_ms",
                    "sampling_ms",
                    "decode_stream_interstage_residual_ms",
                }
                else forward_mean
            )
            row[f"{component}_share"] = (
                finite_float(row[f"{component}_mean"]) / denominator
                if denominator > 0.0
                else 0.0
            )
        rows.append(row)
    return rows


def build_partition_deltas(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_fields = (
        "prefill_mode",
        "decode_context_bucket_tokens",
        "decode_request_count",
        "context_ring_decode_request_count",
        "prefill_overlap_candidate",
    )
    states: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        states[tuple(row[field] for field in state_fields)].append(row)

    result: list[dict[str, Any]] = []
    for state, rows in sorted(states.items()):
        reference = max(rows, key=lambda row: integer(row["decode_sms"]))
        for row in sorted(rows, key=lambda item: integer(item["decode_sms"])):
            output = dict(zip(state_fields, state))
            output.update(
                {
                    "sm_profile_id": row["sm_profile_id"],
                    "decode_sms": row["decode_sms"],
                    "prefill_sms": row["prefill_sms"],
                    "sample_count": row["sample_count"],
                    "reference_sm_profile_id": reference["sm_profile_id"],
                    "reference_decode_sms": reference["decode_sms"],
                }
            )
            for component in COMPONENTS:
                value = finite_float(row[f"{component}_mean"])
                reference_value = finite_float(
                    reference[f"{component}_mean"]
                )
                output[f"{component}_mean"] = value
                output[f"{component}_delta_vs_max_decode_sms_ms"] = (
                    value - reference_value
                )
                output[f"{component}_ratio_vs_max_decode_sms"] = (
                    value / reference_value if reference_value > 0.0 else 0.0
                )
            result.append(output)
    return result


def diagnose(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state_fields = (
        "prefill_mode",
        "decode_context_bucket_tokens",
        "decode_request_count",
        "context_ring_decode_request_count",
        "prefill_overlap_candidate",
    )
    states: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        states[tuple(row[field] for field in state_fields)].append(row)

    diagnoses: list[dict[str, Any]] = []
    for state, rows in sorted(states.items()):
        if len({integer(row["decode_sms"]) for row in rows}) < 2:
            continue
        low = min(rows, key=lambda row: integer(row["decode_sms"]))
        high = max(rows, key=lambda row: integer(row["decode_sms"]))
        low_forward = finite_float(low["gpu_model_forward_ms_mean"])
        high_forward = finite_float(high["gpu_model_forward_ms_mean"])
        low_pipeline = finite_float(low["decode_stream_pipeline_ms_mean"])
        high_pipeline = finite_float(high["decode_stream_pipeline_ms_mean"])
        sm_ratio = integer(high["decode_sms"]) / max(1, integer(low["decode_sms"]))
        forward_speedup = low_forward / high_forward if high_forward > 0 else 0.0
        pipeline_speedup = (
            low_pipeline / high_pipeline if high_pipeline > 0 else 0.0
        )
        improvements = {
            component: finite_float(low[f"{component}_mean"])
            - finite_float(high[f"{component}_mean"])
            for component in ATOMIC_COMPONENTS
        }
        dominant = max(
            ATOMIC_COMPONENTS,
            key=lambda component: finite_float(low[f"{component}_mean"]),
        )
        most_changed = max(improvements, key=lambda item: abs(improvements[item]))
        verified = min(
            finite_float(low["green_sm_count_verified_fraction"]),
            finite_float(high["green_sm_count_verified_fraction"]),
        )
        matched_modules = min(
            integer(low["matched_modules_min"]),
            integer(high["matched_modules_min"]),
        )
        num_records = min(
            finite_float(low["num_records_mean"]),
            finite_float(high["num_records_mean"]),
        )
        if matched_modules <= 0 or num_records <= 0:
            hint = (
                "No recognized per-layer modules were recorded; compare the "
                "whole-forward/pipeline metrics only and add architecture-specific "
                "hook names before attributing a subcomponent."
            )
        elif verified < 0.99:
            hint = (
                "GreenContext SM-count verification is incomplete; partition "
                "enforcement must be checked before attributing performance."
            )
        elif (
            pipeline_speedup < 1.05
            and forward_speedup >= 1.10
            and sm_ratio >= 2.0
        ):
            hint = (
                "Model forward improves with Decode SMs but the full Decode "
                "stream pipeline does not; inter-stage host enqueue gaps, logits, "
                "or sampling hide the kernel speedup."
            )
        elif forward_speedup < 1.05 and sm_ratio >= 2.0:
            if dominant == "attention_ms":
                hint = (
                    "Forward is insensitive to added Decode SMs and attention "
                    "dominates; memory/latency saturation is the leading cause."
                )
            elif dominant == "gpu_residual_unattributed_ms":
                hint = (
                    "Forward is insensitive to added Decode SMs and the residual "
                    "dominates; inspect norms/residual kernels, launch gaps, and "
                    "stream interference with Nsight."
                )
            else:
                hint = (
                    "Forward is insensitive to added Decode SMs; small-batch "
                    "kernel occupancy or partition enforcement is the leading "
                    "follow-up check."
                )
        elif most_changed == "attention_ms":
            hint = "Attention accounts for the largest partition-sensitive delta."
        elif most_changed == "gpu_residual_unattributed_ms":
            hint = (
                "Unattributed forward time changes most; inspect RMSNorm, residual "
                "adds, launch gaps, and cross-stream resource contention."
            )
        elif most_changed == "decode_stream_interstage_residual_ms":
            hint = (
                "The execute-to-sample stream gap changes most; inspect worker "
                "enqueue readiness, scheduler recovery, and cross-stream waits."
            )
        elif most_changed in {"logits_postprocess_ms", "sampling_ms"}:
            hint = "Logits/postprocess or sampling is the largest SM-sensitive stage."
        elif most_changed in {
            "qkv_proj_ms",
            "o_proj_ms",
            "mlp_gate_up_proj_ms",
            "mlp_down_proj_ms",
        }:
            hint = "Projection/MLP GEMMs account for the largest SM-sensitive delta."
        else:
            hint = "Activation or another small layer component changes most."

        row = dict(zip(state_fields, state))
        row.update(
            {
                "low_sm_profile_id": low["sm_profile_id"],
                "low_decode_sms": low["decode_sms"],
                "high_sm_profile_id": high["sm_profile_id"],
                "high_decode_sms": high["decode_sms"],
                "decode_sm_ratio": sm_ratio,
                "low_gpu_model_forward_ms_mean": low_forward,
                "high_gpu_model_forward_ms_mean": high_forward,
                "gpu_model_forward_speedup": forward_speedup,
                "low_decode_stream_pipeline_ms_mean": low_pipeline,
                "high_decode_stream_pipeline_ms_mean": high_pipeline,
                "decode_stream_pipeline_speedup": pipeline_speedup,
                "dominant_low_sm_component": dominant,
                "most_partition_sensitive_component": most_changed,
                "most_partition_sensitive_delta_ms": improvements[most_changed],
                "green_sm_count_verified_fraction": verified,
                "matched_modules_min": matched_modules,
                "num_records_min_mean": num_records,
                "diagnosis_hint": hint,
            }
        )
        for component, improvement in improvements.items():
            row[f"{component}_low_minus_high_ms"] = improvement
        diagnoses.append(row)
    return diagnoses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-bucket-tokens", type=int, default=5000)
    parser.add_argument("--include-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context_bucket_tokens <= 0:
        raise ValueError("--context-bucket-tokens must be positive")
    records = load_records(
        args.artifact_root,
        context_bucket_tokens=args.context_bucket_tokens,
        include_warmup=args.include_warmup,
    )
    if not records:
        raise RuntimeError(
            f"no Decode GPU component records found under {args.artifact_root}"
        )
    summary_rows = summarize(records)
    delta_rows = build_partition_deltas(summary_rows)
    diagnosis_rows = diagnose(summary_rows)
    write_csv(args.output_dir / "decode_gpu_component_records.csv", records)
    write_csv(args.output_dir / "decode_gpu_component_summary.csv", summary_rows)
    write_csv(args.output_dir / "decode_gpu_partition_deltas.csv", delta_rows)
    write_csv(args.output_dir / "decode_gpu_component_diagnosis.csv", diagnosis_rows)
    metadata = {
        "artifact_root": str(args.artifact_root),
        "record_count": len(records),
        "summary_group_count": len(summary_rows),
        "diagnosis_count": len(diagnosis_rows),
        "context_bucket_tokens": args.context_bucket_tokens,
        "include_warmup": args.include_warmup,
        "components": list(COMPONENTS),
        "notes": [
            "Component tracing forces eager execution for all compared partitions.",
            "CUDA events are queried asynchronously; only shutdown performs a tail flush.",
            "Diagnosis hints are evidence-directed follow-up checks, not causal proof.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decode_gpu_component_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] records={len(records)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
