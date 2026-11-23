#!/usr/bin/env python3
"""CPU-only smoke test for vendor_v4 P/D context-ring interference traces."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = ROOT / "graph_script" / "analyze_pd_context_ring_interference.py"
COPY_ANALYZER_PATH = (
    ROOT / "graph_script" / "analyze_context_ring_copy_overlap.py"
)
ATTENTION_PATH = (
    ROOT
    / "vendor_v4"
    / "vllm"
    / "v1"
    / "attention"
    / "ops"
    / "tiered_blockwise_attention.py"
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _assert_source_contract() -> None:
    runner_source = (
        ROOT / "vendor_v4" / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    ).read_text(encoding="utf-8")
    backend_source = (
        ROOT
        / "vendor_v4"
        / "vllm"
        / "v1"
        / "attention"
        / "ops"
        / "tiered_blockwise_backend.py"
    ).read_text(encoding="utf-8")
    metadata_source = (
        ROOT / "vendor_v4" / "vllm" / "v1" / "attention" / "backend.py"
    ).read_text(encoding="utf-8")
    attention_source = (
        ROOT
        / "vendor_v4"
        / "vllm"
        / "v1"
        / "attention"
        / "ops"
        / "tiered_blockwise_attention.py"
    ).read_text(encoding="utf-8")
    assert "device_start_ms_from_epoch" in runner_source
    assert "cuda_event_shared_epoch" in runner_source
    assert "pd_scheduler_iteration_id" in metadata_source
    assert "tiered_context_ring_interference" in backend_source
    assert "VLLM_CONTEXT_RING_INTERFERENCE_TRACE_ENABLED" in backend_source
    assert "VLLM_CONTEXT_RING_NVTX_TRACE_ENABLED" in backend_source
    assert "TieredContextRingCUDAEventTiming" in attention_source
    assert "ContextRingTiming" in backend_source
    assert "tiered_context_ring_interference_timing" in backend_source
    assert attention_source.count(
        "_sum_cuda_event_elapsed_ms_blocking("
    ) == 4
    run_impl_section = attention_source[
        attention_source.index("    def _run_impl(") :
        attention_source.index("\ndef run_tiered_blockwise_attention(")
    ]
    assert ".synchronize()" not in run_impl_section


def main() -> int:
    _assert_source_contract()
    analyzer = _load_module(
        ANALYZER_PATH,
        "analyze_pd_context_ring_interference",
    )
    copy_analyzer = _load_module(
        COPY_ANALYZER_PATH,
        "analyze_context_ring_copy_overlap",
    )
    attention = _load_module(
        ATTENTION_PATH,
        "tiered_blockwise_attention_deferred_timing_smoke",
    )

    class FakeStartEvent:
        @staticmethod
        def elapsed_time(end_event):
            return end_event.elapsed_ms

    class FakeEndEvent:
        def __init__(self, elapsed_ms):
            self.elapsed_ms = elapsed_ms
            self.synchronize_calls = 0

        def synchronize(self):
            self.synchronize_calls += 1

    copy_end = FakeEndEvent(4.0)
    wait_end = FakeEndEvent(1.0)
    compute_end = FakeEndEvent(18.0)
    timing = attention.TieredContextRingCUDAEventTiming(
        copy_event_pairs=((FakeStartEvent(), copy_end),),
        wait_event_pairs=((FakeStartEvent(), wait_end),),
        compute_event_pairs=((FakeStartEvent(), compute_end),),
    )
    resolved = timing.resolve_blocking()
    assert resolved["copy_event_elapsed_ms"] == 4.0
    assert resolved["wait_event_elapsed_ms"] == 1.0
    assert resolved["compute_event_elapsed_ms"] == 18.0
    assert resolved["copy_compute_overlap_ratio"] == 0.75
    assert copy_end.synchronize_calls == 1
    assert wait_end.synchronize_calls == 1
    assert compute_end.synchronize_calls == 1
    epoch_id = "123:0:1"
    pd_records = [
        {
            "kind": "forward",
            "phase": "decode",
            "iteration_id": 19,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 1.0,
            "device_end_ms_from_epoch": 2.0,
        },
        {
            "kind": "forward",
            "phase": "prefill",
            "iteration_id": 10,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 10.0,
            "device_end_ms_from_epoch": 30.0,
        },
        {
            "kind": "forward",
            "phase": "decode",
            "iteration_id": 20,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 12.0,
            "device_end_ms_from_epoch": 15.0,
        },
        {
            "kind": "forward",
            "phase": "prefill",
            "iteration_id": 11,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 40.0,
            "device_end_ms_from_epoch": 60.0,
        },
        {
            "kind": "forward",
            "phase": "decode",
            "iteration_id": 21,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 45.0,
            "device_end_ms_from_epoch": 50.0,
        },
        {
            "kind": "forward",
            "phase": "decode",
            "iteration_id": 22,
            "device_epoch_id": epoch_id,
            "device_start_ms_from_epoch": 70.0,
            "device_end_ms_from_epoch": 73.0,
        },
    ]
    timeline_records = [
        {
            "phase": "decode",
            "request_id": "req-1",
            "iteration_id": iteration_id,
            "token_iteration_id": token_index,
            "token_request_context_len": 75000 + token_index,
            "timestamp": timestamp,
        }
        for token_index, (iteration_id, timestamp) in enumerate(
            ((19, 1.00), (20, 1.05), (21, 1.12), (22, 1.20))
        )
    ]
    ring_records = [
        {
            "event": "tiered_context_ring_done",
            "scheduler_iteration_id": 10,
            "ring_execution_id": "123:1:0",
            "request_id": "prefill-ring-req",
            "attention_plan": {},
            "stats": {
                "context_ring_prefetch_bytes": 33554432,
                "context_ring_copy_event_elapsed_ms": 0.0,
                "context_ring_copy_event_count": 8,
                "context_ring_wait_event_elapsed_ms": 0.0,
                "context_ring_wait_event_count": 8,
                "context_ring_compute_event_elapsed_ms": 0.0,
                "context_ring_compute_event_count": 8,
                "context_ring_cuda_event_timing_deferred": 1,
            },
        },
        {
            "event": "tiered_context_ring_interference",
            "schema_version": 2,
            "scheduler_iteration_id": 10,
            "ring_execution_id": "123:1:0",
            "request_id": "prefill-ring-req",
            "prefetch_bytes": 33554432,
            "cuda_event_timing_deferred": 1,
            "copy_event_elapsed_ms": 0.0,
            "copy_event_count": 8,
            "wait_event_elapsed_ms": 0.0,
            "wait_event_count": 8,
            "compute_event_elapsed_ms": 0.0,
            "compute_event_count": 8,
            "prefetch_submit_host_duration_s": 0.001,
            "prefetch_wait_host_duration_s": 0.002,
            "attention_compute_host_duration_s": 0.018,
            "total_host_duration_s": 0.021,
            "staging_slot_waits": 2,
        },
        {
            "event": "tiered_context_ring_interference_timing",
            "schema_version": 2,
            "scheduler_iteration_id": 10,
            "ring_execution_id": "123:1:0",
            "request_id": "prefill-ring-req",
            "timing_status": "complete",
            "copy_event_elapsed_ms": 4.0,
            "copy_event_resolved_count": 8,
            "wait_event_elapsed_ms": 1.0,
            "wait_event_resolved_count": 8,
            "compute_event_elapsed_ms": 18.0,
            "compute_event_resolved_count": 8,
            "copy_compute_overlap_ratio": 0.75,
        },
    ]

    samples, ring_iterations, metadata = (
        analyzer.build_decode_overlap_samples(
            pd_records,
            timeline_records,
            ring_records,
        )
    )
    by_iteration = {sample["iteration_id"]: sample for sample in samples}
    assert metadata["interval_basis"] == "cuda_event_shared_epoch"
    assert by_iteration[19]["overlap_kind"] == "none"
    assert by_iteration[20]["overlap_kind"] == "context_ring_prefill"
    assert by_iteration[21]["overlap_kind"] == "native_prefill"
    assert by_iteration[22]["overlap_kind"] == "none"
    assert math.isclose(by_iteration[20]["schedule_gap_ms"], 50.0)
    assert by_iteration[20]["ring_prefetch_bytes"] == 33554432
    assert ring_iterations[10]["copy_event_elapsed_ms"] == 4.0
    assert ring_iterations[10]["copy_event_count"] == 8
    assert ring_iterations[10]["ring_timing_deferred_count"] == 1
    assert ring_iterations[10]["ring_timing_completed_count"] == 1
    assert ring_iterations[10]["ring_timing_pending_count"] == 0
    assert metadata["ring_timing_deferred_count"] == 1
    assert metadata["ring_timing_completed_count"] == 1
    assert metadata["ring_timing_error_count"] == 0
    assert metadata["ring_timing_pending_count"] == 0
    legacy_samples, _, legacy_metadata = (
        analyzer.build_decode_overlap_samples(
            pd_records,
            timeline_records,
            [{"event": "tiered_context_ring_done"}],
        )
    )
    legacy_by_iteration = {
        sample["iteration_id"]: sample for sample in legacy_samples
    }
    assert legacy_metadata["has_unjoinable_ring_executions"]
    assert legacy_by_iteration[20]["overlap_kind"] == "prefill_unattributed"

    with tempfile.TemporaryDirectory() as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        artifact_dir = temp_dir / "artifacts"
        pd_path = artifact_dir / "pd_cuda_event_trace" / "pd_cuda_event_trace.jsonl"
        timeline_path = (
            artifact_dir / "decode_timeline" / "decode_timeline.jsonl"
        )
        ring_path = (
            artifact_dir / "context_ring_trace" / "context_ring_trace.jsonl"
        )
        _write_jsonl(pd_path, pd_records)
        _write_jsonl(timeline_path, timeline_records)
        _write_jsonl(ring_path, ring_records)
        output_dir = temp_dir / "analysis"
        result = analyzer.analyze(
            pd_cuda_path=pd_path,
            decode_timeline_path=timeline_path,
            context_ring_path=ring_path,
            output_dir=output_dir,
            make_plots=False,
        )
        assert result["decode_sample_count"] == 4
        assert (output_dir / "decode_overlap_samples.csv").is_file()
        assert (output_dir / "overlap_summary.csv").is_file()
        assert (output_dir / "ring_iteration_summary.csv").is_file()
        assert (output_dir / "analysis_summary.json").is_file()
        copy_row = copy_analyzer.summarize_dram_to_hbm(ring_path)
        assert copy_row["basis"] == "cuda_event"
        assert math.isclose(copy_row["copy_s"], 0.004)
        assert copy_row["copy_event_count"] == 8
        assert copy_row["timing_deferred_events"] == 1
        assert copy_row["timing_completed_events"] == 1

    print("context-ring interference trace smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
