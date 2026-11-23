#!/usr/bin/env python3
"""Import-free contracts for reversible vendor_v8 throughput paths."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class ThroughputHotPathContractTest(unittest.TestCase):
    def test_fused_phase_rpc_keeps_execute_and_sample_on_one_key(self) -> None:
        worker = read("vendor_v8/vllm/v1/worker/gpu_worker.py")
        executor = read("vendor_v8/vllm/v1/executor/uniproc_executor.py")
        engine = read("vendor_v8/vllm/v1/engine/core.py")

        worker_body = worker.split(
            "def execute_model_and_sample(", 1
        )[1].split("def execute_model(", 1)[0]
        self.assertIn("pd_green_context_phase=pd_green_context_phase", worker_body)
        self.assertIn("return self.sample_tokens(", worker_body)
        self.assertIn('"execute_model_and_sample"', executor)
        self.assertIn("VLLM_PD_FUSED_EXECUTE_SAMPLE", engine)
        self.assertIn("pd_fused_execute_sample_used", engine)

    def test_decode_graph_capture_skips_zero_sm_profiles(self) -> None:
        runner = read("vendor_v8/vllm/v1/worker/gpu_model_runner.py")
        capture = runner.split(
            "def _capture_pd_green_decode_cudagraphs(", 1
        )[1].split("def capture_model(", 1)[0]
        self.assertIn("def has_decode_sms(profile_id: str)", capture)
        self.assertIn(
            "VLLM_PD_GREEN_DECODE_CUDAGRAPH_PROFILE_IDS",
            capture,
        )
        self.assertIn("if has_decode_sms(profile_id)", capture)

    def test_lightweight_trace_filters_worker_snapshot_at_json_boundary(self) -> None:
        observability = read(
            "vendor_v8/vllm/v1/core/sched/observability.py"
        )
        engine = read("vendor_v8/vllm/v1/engine/core.py")
        self.assertIn("def compact_worker_debug_timings(", observability)
        self.assertIn(
            "def strict_bounded_lightweight_timeline_enabled(",
            observability,
        )
        self.assertIn(
            "executor_async_completion_residual_duration_s",
            engine,
        )
        self.assertIn("compact_worker_debug_timings(debug_timings)", engine)

    def test_runner_bundle_is_default_off_and_individually_overridable(self) -> None:
        runner = read("run_vendor_v8_short_validation.sh")
        compare = read("run_vendor_v8_dynamic_hbm_vs_full.sh")
        self.assertIn(
            'FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS="'
            '${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS:-0}"',
            runner,
        )
        self.assertIn("VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE", runner)
        self.assertIn("VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH", runner)
        self.assertIn("VLLM_PD_EVAL_RING_BW_USE_SCHEDULED_HEADROOM", runner)
        self.assertIn("DEFAULT_VIRTUAL_CONTEXT_TIMING=0", runner)
        self.assertIn("DEFAULT_VIRTUAL_CONTEXT_TIMING=1", runner)
        self.assertIn(
            "VLLM_PD_EVAL_STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE",
            runner,
        )
        self.assertIn("VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL", runner)
        self.assertIn(
            'PD_SM_PROFILES="0/40,30/10,32/8,34/6,36/4,38/2,40/0"',
            runner,
        )
        self.assertIn(
            'ENABLE_PD_EXECUTOR_DEBUG="${ENABLE_PD_EXECUTOR_DEBUG:-0}"',
            runner,
        )
        self.assertIn("FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS", compare)

    def test_four_case_serialized_full_sm_ablation_is_explicit(self) -> None:
        runner = read("run_vendor_v8_short_validation.sh")
        compare = read("run_vendor_v8_dynamic_hbm_vs_full.sh")
        allocator = read("vendor_v8/vllm/v1/engine/pd_sm_allocator.py")

        self.assertIn(
            "native_40sm_chunked full_vendor_v8 "
            "dynamic_hbm_serialized_fixed_split "
            "dynamic_hbm_serialized_fullsm",
            compare,
        )
        self.assertIn("serialized_full_sm=0", compare)
        self.assertIn("serialized_full_sm=1", compare)
        self.assertIn(
            'VLLM_PD_SERIALIZED_FULL_SM="${serialized_full_sm}"',
            compare,
        )
        self.assertIn("dynamic_hbm_serialized_fixed_split", runner)
        self.assertIn("dynamic_hbm_serialized_fullsm", runner)
        self.assertIn(
            'VLLM_PD_SERIALIZED_FULL_SM="${PD_SERIALIZED_FULL_SM}"',
            runner,
        )
        self.assertIn('"capacity_serialized_fixed_split:', allocator)
        self.assertIn('"capacity_serialized_decode_full_sm"', allocator)
        self.assertIn('"decode_only_full_sm"', allocator)
        self.assertIn('"prefill_only_full_sm"', allocator)
        self.assertIn(
            '"pd_dynamic_sm_serialized_full_sm_targeted"',
            allocator,
        )
        self.assertIn('"pd_dynamic_sm_capacity_serialized"', allocator)

    def test_four_case_cuda_graph_contract_is_explicit(self) -> None:
        runner = read("run_vendor_v8_short_validation.sh")
        compare = read("run_vendor_v8_dynamic_hbm_vs_full.sh")
        starter = read("start_vllm_causality_microbench.sh")

        self.assertIn('BENCHMARK_CUDAGRAPH="${BENCHMARK_CUDAGRAPH:-1}"', compare)
        self.assertIn(
            'BENCHMARK_CUDAGRAPH_MODE="${BENCHMARK_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"',
            compare,
        )
        self.assertIn(
            'VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH="${green_decode_cudagraph}"',
            compare,
        )
        self.assertIn(
            'VLLM_PD_EVAL_CUDAGRAPH_MODE="${BENCHMARK_CUDAGRAPH_MODE}"',
            compare,
        )
        self.assertIn(
            'VLLM_CUDAGRAPH_MODE="${CUDAGRAPH_MODE}"',
            runner,
        )
        self.assertIn("CUDAGRAPH_CONFIG_ARGS=(", starter)
        self.assertIn(
            '"{\\"cudagraph_mode\\":\\"${CUDAGRAPH_MODE}\\"}"',
            starter,
        )

    def test_inflight_requests_are_not_generic_preemption_victims(self) -> None:
        scheduler = read("vendor_v8/vllm/v1/core/sched/scheduler.py")
        self.assertIn("def _pd_preemption_protection_reason", scheduler)
        self.assertIn('return "execution_future_pending"', scheduler)
        self.assertIn("protected_running = [", scheduler)
        self.assertIn("pd_preemption_future_protection_count", scheduler)

    def test_active_decode_elastic_ring_is_separate_and_default_off(self) -> None:
        scheduler = read("vendor_v8/vllm/v1/core/sched/scheduler.py")
        runner = read("run_vendor_v8_short_validation.sh")
        self.assertIn(
            "VLLM_VIRTUAL_CONTEXT_ACTIVE_DECODE_ELASTIC_PREFILL",
            scheduler,
        )
        self.assertIn(
            'ACTIVE_DECODE_ELASTIC_PREFILL="'
            '${VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL:-0}"',
            runner,
        )
        self.assertIn("elastic_safe_capacity", scheduler)

    def test_future_liveness_audit_is_observational_and_default_off(self) -> None:
        engine = read("vendor_v8/vllm/v1/engine/core.py")
        progress = read(
            "vendor_v8/vllm/v1/engine/pd_engine_progress.py"
        )
        executor = read("vendor_v8/vllm/v1/executor/uniproc_executor.py")

        self.assertIn(
            '"VLLM_PD_FUTURE_LIVENESS_AUDIT", "0"',
            engine,
        )
        self.assertIn(
            "return self._pd_recoverable_future_ready(phase_queues)",
            engine,
        )
        audit_body = engine.split(
            "def _record_pd_future_liveness_audit(", 1
        )[1].split("def _wait_for_pd_progress(", 1)[0]
        self.assertIn("pd_future_lifecycle_snapshot", audit_body)
        self.assertIn(
            '"pd_future_liveness_audit_ownership_preserved": 1.0',
            audit_body,
        )
        self.assertNotIn(".cancel(", audit_body)
        self.assertNotIn("_pop_pd_phase_queue_at", audit_body)
        self.assertIn("class PDAsyncFutureLifecycle", progress)
        self.assertIn("attach_pd_future_lifecycle(", executor)
        self.assertIn(
            "if self.pd_future_liveness_audit_enabled",
            executor,
        )
        self.assertIn('lifecycle.advance("worker_body_started")', executor)
        self.assertIn('lifecycle.advance("output_wait_started")', executor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
