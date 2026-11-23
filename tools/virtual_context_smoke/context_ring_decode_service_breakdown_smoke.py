#!/usr/bin/env python3
"""CPU-only smoke for the decode service-gap breakdown generator."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "graph_script"
GENERATOR_PATH = GRAPH_DIR / "generate_context_ring_decode_service_breakdown.py"


def _load_generator():
    sys.path.insert(0, str(GRAPH_DIR))
    spec = importlib.util.spec_from_file_location(
        "context_ring_decode_service_breakdown",
        GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    generator = _load_generator()
    epoch_id = "123:0:1"
    pd_rows = [
        {
            "kind": "forward",
            "phase": "decode",
            "iteration_id": 100,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 10.0,
            "device_end_ms_from_epoch": 90.0,
        },
        {
            "kind": "forward",
            "phase": "decode",
            "iteration_id": 101,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 110.0,
            "device_end_ms_from_epoch": 280.0,
        },
        {
            "kind": "forward",
            "phase": "decode",
            "iteration_id": 102,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 300.0,
            "device_end_ms_from_epoch": 380.0,
        },
        {
            "kind": "forward",
            "phase": "prefill",
            "iteration_id": 500,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 120.0,
            "device_end_ms_from_epoch": 250.0,
        },
    ]
    request_id = "case-foreground-loogle-1-0-abcdef"
    timeline_rows = [
        {
            "phase": "decode",
            "request_id": request_id,
            "iteration_id": iteration_id,
            "timestamp": timestamp,
            "num_prompt_tokens": 100000,
        }
        for iteration_id, timestamp in (
            (100, 1.0),
            (101, 1.1),
            (102, 1.3),
        )
    ]
    ring_rows = [
        {
            "event": "tiered_context_ring_interference",
            "scheduler_iteration_id": 500,
            "ring_execution_id": "123:1:0",
            "request_id": "ring-prefill",
            "cuda_event_timing_deferred": 1,
            "prefetch_bytes": 4096,
            "copy_event_count": 1,
            "wait_event_count": 1,
            "compute_event_count": 1,
        },
        {
            "event": "tiered_context_ring_interference_timing",
            "scheduler_iteration_id": 500,
            "ring_execution_id": "123:1:0",
            "request_id": "ring-prefill",
            "copy_event_elapsed_ms": 4.0,
            "wait_event_elapsed_ms": 1.0,
            "compute_event_elapsed_ms": 18.0,
        },
    ]

    samples, metadata = generator.build_service_gap_samples(
        pd_rows,
        timeline_rows,
        ring_rows,
    )
    assert metadata["interval_basis"] == "cuda_event_shared_epoch"
    assert len(samples) == 2
    assert samples[0]["gap_start_iteration_id"] == 100
    assert samples[0]["gap_end_iteration_id"] == 101
    assert samples[0]["overlap_kind"] == "none"
    assert math.isclose(samples[0]["service_gap_ms"], 100.0)
    assert math.isclose(samples[0]["decode_cuda_forward_ms"], 80.0)
    assert math.isclose(samples[0]["non_forward_residual_ms"], 20.0)
    assert samples[1]["gap_start_iteration_id"] == 101
    assert samples[1]["overlap_kind"] == "context_ring_prefill"
    assert math.isclose(samples[1]["service_gap_ms"], 200.0)
    assert math.isclose(samples[1]["decode_cuda_forward_ms"], 170.0)
    assert math.isclose(samples[1]["non_forward_residual_ms"], 30.0)

    summary = generator.summarize_samples(samples, repeat_index=1)
    by_kind = {row["overlap_kind"]: row for row in summary}
    assert math.isclose(
        by_kind["none"]["service_gap_p50_ms"],
        100.0,
    )
    assert math.isclose(
        by_kind["context_ring_prefill"]["service_gap_p50_ms"],
        200.0,
    )
    iteration_metrics_rows = [
        {
            "iteration_id": 100,
            "schedule_duration_s": 0.004,
            "future_wait_after_poll_s": 0.0,
            "executor_unattributed_wait_duration_s": 0.006,
        },
        {
            "iteration_id": 101,
            "schedule_duration_s": 0.005,
            "future_wait_duration_s": 0.001,
            "executor_unattributed_wait_duration_s": 0.014,
        }
    ]
    tail_samples, tail_summary, tail_metadata = (
        generator.build_tail_residual_breakdown(
            samples,
            iteration_metrics_rows,
            repeat_index=1,
        )
    )
    assert len(tail_samples) == 1
    assert tail_metadata["ring_residual_top_1pct_sample_count"] == 1
    assert math.isclose(
        tail_samples[0]["worker_host_remainder_ms"],
        11.0,
    )
    assert math.isclose(
        tail_summary[0]["executor_unattributed_ms_share_pct"],
        100.0 * 13.0 / 30.0,
    )
    attributed, missing_count = generator.attribute_residual_components(
        samples,
        iteration_metrics_rows,
        repeat_index=1,
    )
    assert missing_count == 0
    quantile_detail = generator.build_quantile_detail_rows(attributed)
    assert len(quantile_detail) == 6
    for row in quantile_detail:
        component_sum = sum(
            float(row[key])
            for key in (
                "decode_cuda_forward_ms",
                "schedule_duration_ms",
                "executor_unattributed_ms",
                "worker_host_remainder_ms",
                "future_wait_after_poll_ms",
            )
        )
        assert math.isclose(
            component_sum,
            float(row["service_gap_ms"]),
        )

    with tempfile.TemporaryDirectory() as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        artifact_dir = temp_dir / "artifacts"
        _write_jsonl(
            artifact_dir
            / "pd_cuda_event_trace"
            / "pd_cuda_event_trace.jsonl",
            pd_rows,
        )
        _write_jsonl(
            artifact_dir / "decode_timeline" / "decode_timeline.jsonl",
            timeline_rows,
        )
        _write_jsonl(
            artifact_dir / "context_ring_trace" / "context_ring_trace.jsonl",
            ring_rows,
        )
        _write_jsonl(
            artifact_dir / "iteration_metrics" / "iteration_metrics.jsonl",
            iteration_metrics_rows,
        )
        output_dir = temp_dir / "output"
        result = generator.generate(
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            repeat_index=1,
            make_plot=True,
        )
        assert math.isclose(result["total_gap_increase_ms"], 100.0)
        assert math.isclose(
            result["decode_cuda_forward_increase_ms"],
            90.0,
        )
        assert math.isclose(
            result["p50_level_residual_increase_ms"],
            10.0,
        )
        assert math.isclose(
            result["decode_cuda_forward_share_pct"],
            90.0,
        )
        assert (output_dir / "decode_service_gap_breakdown.png").is_file()
        assert (
            output_dir / "decode_service_gap_quantile_breakdown.png"
        ).is_file()
        assert (
            output_dir / "decode_p99_non_forward_residual_breakdown.png"
        ).is_file()
        assert (
            output_dir / "decode_service_gap_quantile_detailed_breakdown.png"
        ).is_file()
        assert (output_dir / "decode_service_breakdown_samples.csv").is_file()
        assert (output_dir / "decode_service_breakdown_summary.csv").is_file()
        assert (
            output_dir / "decode_p99_non_forward_residual_samples.csv"
        ).is_file()
        assert (
            output_dir / "decode_p99_non_forward_residual_summary.csv"
        ).is_file()
        assert (
            output_dir / "decode_service_gap_quantile_detailed_breakdown.csv"
        ).is_file()
        assert (output_dir / "analysis_summary.json").is_file()

    print("context-ring decode service breakdown smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
