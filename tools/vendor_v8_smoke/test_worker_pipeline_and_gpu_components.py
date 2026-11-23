#!/usr/bin/env python3
"""CPU-only contracts for vendor_v8 worker flow and component analysis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_decode_gpu_components import (  # noqa: E402
    build_partition_deltas,
    diagnose,
    load_records,
    summarize,
)


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class VendorV8WorkerPipelineContractTest(unittest.TestCase):
    def test_phase_plan_uses_cpu_mirror_fast_reject_and_iteration_cache(
        self,
    ) -> None:
        source = read(
            "vendor_v8/vllm/model_executor/layers/"
            "operator_selective_attention.py"
        )
        plan_body = source.split("def get_pd_phase_plan(", 1)[1].split(
            "def run_with_pd_phase_context(", 1
        )[0]
        self.assertIn("_OP_PD_PHASE_PLAN_CACHE", plan_body)
        self.assertIn('getattr(attn_metadata, "query_start_loc_cpu", None)', plan_body)
        self.assertIn('getattr(attn_metadata, "seq_lens_cpu", None)', plan_body)
        self.assertLess(
            plan_body.index("attn_metadata.query_start_loc.shape[0]"),
            plan_body.index('attn_metadata.query_start_loc.detach().to("cpu")'),
        )
        self.assertIn("op_pd_plan_cache_hit_count", source)
        self.assertIn("op_pd_plan_fast_reject_count", source)

    def test_no_mix_disables_operator_selective_selector(self) -> None:
        source = read(
            "vendor_v8/vllm/model_executor/layers/"
            "operator_selective_attention.py"
        )
        launcher = read("start_vllm_causality_microbench.sh")
        enabled_body = source.split(
            "def operator_selective_pd_enabled()", 1
        )[1].split("def muxwise_phase_baseline_enabled()", 1)[0]
        self.assertIn("_PD_DECOUPLED_NO_MIX_ENV", enabled_body)
        self.assertIn("OPERATOR_SELECTIVE_PD_REQUESTED", launcher)
        self.assertIn('OPERATOR_SELECTIVE_PD="0"', launcher)

    def test_decode_first_gate_wraps_model_dispatch_host_entry(self) -> None:
        source = read("vendor_v8/vllm/v1/executor/uniproc_executor.py")
        submit_body = source.split("def _submit_pd_async_rpc(", 1)[1].split(
            "def _pd_async_executor_for_phase(", 1
        )[0]
        run_body = source.split("def _run_pd_async_rpc(", 1)[1].split(
            "def _resolve_pd_async_rpc_result(", 1
        )[0]
        ordered_block = submit_body.split("ordered = (", 1)[1].split(
            "host_enqueue_ticket", 1
        )[0]
        self.assertLess(
            submit_body.index("self._pd_host_enqueue_gate.register("),
            submit_body.index("async_executor.submit("),
        )
        self.assertIn(
            'in {"execute_model", "execute_model_and_sample"}',
            submit_body,
        )
        self.assertNotIn('"sample_tokens"', ordered_block)
        self.assertIn("self._pd_host_enqueue_gate.enter(host_enqueue_ticket)", run_body)
        self.assertNotIn(".get_output()", run_body)

    def test_tbt_control_does_not_force_full_trace_and_writes_are_async(
        self,
    ) -> None:
        observability = read("vendor_v8/vllm/v1/core/sched/observability.py")
        runner = read("vendor_v8/vllm/v1/worker/gpu_model_runner.py")
        writer = read("vendor_v8/vllm/v1/engine/async_jsonl_writer.py")
        detailed_gate = observability.split(
            "def detailed_iteration_debug_enabled(", 1
        )[1].split("def iteration_debug_sample_every_n(", 1)[0]
        self.assertNotIn("TBT_SLO_OBSERVABILITY", detailed_gate)
        self.assertIn("AsyncJSONLWriter", runner)
        self.assertNotIn('open(path, "a"', runner)
        self.assertIn("put_nowait", writer)
        self.assertIn("self._handles", writer)

    def test_lightweight_timeline_avoids_full_scheduler_debug(self) -> None:
        observability = read("vendor_v8/vllm/v1/core/sched/observability.py")
        scheduler = read("vendor_v8/vllm/v1/core/sched/scheduler.py")
        runner = read("run_vendor_v8_short_validation.sh")
        detailed_gate = observability.split(
            "def detailed_iteration_debug_enabled(", 1
        )[1].split("def lightweight_timeline_enabled(", 1)[0]
        self.assertIn("not lightweight_timeline", detailed_gate)
        self.assertIn("_build_lightweight_timeline_debug", scheduler)
        self.assertIn("VLLM_LIGHTWEIGHT_TIMELINE_TRACE", runner)

    def test_reload_admission_count_does_not_materialize_layer_pages(self) -> None:
        scheduler = read("vendor_v8/vllm/v1/core/sched/scheduler.py")
        kv_manager = read("vendor_v8/vllm/v1/core/kv_cache_manager.py")
        count_body = scheduler.split(
            "def _virtual_context_reload_required_page_count(", 1
        )[1].split("\n    def ", 1)[0]
        manager_body = kv_manager.split(
            "def get_virtual_context_hbm_reload_required_page_count(", 1
        )[1].split("\n    def ", 1)[0]
        self.assertIn(
            "get_virtual_context_hbm_reload_required_page_count", count_body
        )
        self.assertNotIn("summarize_reload_guard", count_body)
        self.assertIn("hbm_reload_required_page_count", manager_body)
        self.assertNotIn("plan_hbm_reload_pages(", manager_body)

    def test_async_output_wait_is_off_the_phase_submit_worker(self) -> None:
        source = read("vendor_v8/vllm/v1/executor/uniproc_executor.py")
        run_body = source.split("def _run_pd_async_rpc(", 1)[1].split(
            "def _resolve_pd_async_rpc_result(", 1
        )[0]
        resolve_body = source.split(
            "def _resolve_pd_async_rpc_result(", 1
        )[1].split("def _submit_pd_async_rpc(", 1)[0]
        self.assertNotIn(".get_output()", run_body)
        self.assertIn("result.get_output()", resolve_body)
        self.assertIn("pd_async_uniproc_phase_output_executors", source)
        self.assertIn("pd_async_uniproc_output_offload_enabled", source)
        self.assertIn("args[0].iteration_debug = {}", source)

    def test_phase_depth_disables_shared_pair_and_uses_slot_execution_keys(
        self,
    ) -> None:
        core = read("vendor_v8/vllm/v1/engine/core.py")
        executor = read("vendor_v8/vllm/v1/executor/uniproc_executor.py")
        worker = read("vendor_v8/vllm/v1/worker/gpu_worker.py")
        enabled_expr = core.split(
            "self.pd_paired_phase_launch_enabled = (", 1
        )[1].split(")", 1)[0]
        self.assertIn(
            "self.pd_single_inflight_per_phase_enabled", enabled_expr
        )
        self.assertIn("not self.pd_phase_depth_enabled", enabled_expr)
        self.assertIn(
            "Phase-depth mode must use independent phase", core
        )
        self.assertGreaterEqual(
            core.count("self._assign_pd_phase_submission("), 5
        )
        for function_name in (
            "_submit_pd_phase_pair_to_queues",
            "_submit_pd_phase_pair_decoupled_to_queues",
        ):
            function_body = core.split(f"def {function_name}(", 1)[1].split(
                "\n    def ", 1
            )[0]
            rpc_offset = function_body.index(
                "self.model_executor.execute_model_pd_pair("
            )
            self.assertLess(
                function_body.index(
                    "self._assign_pd_phase_submission(\n"
                    "            phase_queues, decode_scheduler_output"
                ),
                rpc_offset,
            )
            self.assertLess(
                function_body.index(
                    "self._assign_pd_phase_submission(\n"
                    "            phase_queues, prefill_scheduler_output"
                ),
                rpc_offset,
            )
        self.assertIn("pd_green_context_phase=execution_key", core)
        self.assertIn("decode_execution_key", executor)
        self.assertIn("prefill_execution_key", executor)
        self.assertIn("decode_execution_key", worker)
        self.assertIn("prefill_execution_key", worker)

    def test_rpc_debug_keeps_execute_and_sample_timings_separate(self) -> None:
        source = read("vendor_v8/vllm/v1/executor/uniproc_executor.py")
        run_body = source.split("def _run_pd_async_rpc(", 1)[1].split(
            "def _resolve_pd_async_rpc_result(", 1
        )[0]
        self.assertIn("pd_async_{rpc_metric_prefix}_task_queue_wait_s", run_body)
        self.assertIn("pd_async_{rpc_metric_prefix}_lock_wait_s", run_body)
        self.assertIn(
            "pd_async_{rpc_metric_prefix}_worker_call_duration_s", run_body
        )

    def test_ready_decode_is_recovered_before_new_scheduling(self) -> None:
        core = read("vendor_v8/vllm/v1/engine/core.py")
        self.assertIn("ready_recovery_before_submit", core)
        self.assertIn("pd_decode_ready_recovery_before_submit", core)
        self.assertIn("pd_component_prefill_overlap_candidate", core)
        self.assertIn(
            "if not self.full_iteration_debug_enabled:", core
        )
        self.assertIn(
            "scheduler_has_requests and not ready_recovery_before_submit",
            core,
        )

    def test_sampling_does_not_import_unrelated_default_stream_work(self) -> None:
        green = read("vendor_v8/vllm/model_executor/layers/green_context.py")
        runner = read("vendor_v8/vllm/v1/worker/gpu_model_runner.py")
        self.assertIn("wait_for_current_stream: bool = True", green)
        self.assertIn("if wait_for_current_stream:", green)
        sample_body = runner.split("def sample_tokens(", 1)[1].split(
            "def _sample_tokens_impl(", 1
        )[0]
        self.assertIn(
            "wait_for_current_stream=not reuse_ordered_phase_stream",
            sample_body,
        )

    def test_component_capture_is_context_local_and_async(self) -> None:
        runner = read("vendor_v8/vllm/v1/worker/gpu_model_runner.py")
        self.assertIn("contextvars.ContextVar", runner)
        self.assertIn("bool(end_event.query())", runner)
        self.assertIn("pd_decode_gpu_components_v1", runner)
        self.assertIn("gpu_residual_unattributed_ms", runner)
        self.assertIn("VLLM_PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N", runner)
        capture_body = runner.split("def capture_forward(", 1)[1].split(
            "def capture_ready(", 1
        )[0]
        self.assertNotIn("self.matched_modules == 0", capture_body)
        enqueue_body = runner.split(
            "def _enqueue_pd_decode_gpu_component_capture(", 1
        )[1].split("def finalize_pd_decode_gpu_component_trace(", 1)[0]
        self.assertNotIn("synchronize", enqueue_body)

    def test_phase_failure_clears_only_exact_owned_slot(self) -> None:
        runner = read("vendor_v8/vllm/v1/worker/gpu_model_runner.py")
        clear_body = runner.split(
            "def clear_pd_phase_execute_state(", 1
        )[1].split("def _init_model_kwargs(", 1)[0]
        self.assertIn("context.execute_model_state = None", clear_body)
        self.assertIn(
            "context._pd_decode_gpu_component_capture = None", clear_body
        )
        self.assertIn("expected_owner.validate_key(execution_key)", clear_body)
        self.assertIn("context.kv_retention_execute_owner != expected_owner", clear_body)
        self.assertNotIn("self._pd_phase_contexts.items()", clear_body)

    def test_eval_policy_disables_shared_pair_for_phase_depth(self) -> None:
        policy = read("tools/pd_eval_policy_env.sh")
        self.assertIn(
            'VLLM_PD_EVAL_PHASE_SYNC_SAMPLE_OUTPUT:-0', policy
        )
        phase_depth_branch = policy.split(
            'if pd_eval_policy_bool_true "${phase_depth_enabled}"; then', 1
        )[1].split("else", 1)[0]
        self.assertIn("VLLM_PD_PAIRED_PHASE_LAUNCH=0", phase_depth_branch)


class DecodeGPUComponentAnalyzerTest(unittest.TestCase):
    @staticmethod
    def record(decode_sms: int, forward_ms: float, *, warmup: bool) -> dict:
        attention_ms = forward_ms * 0.5
        mlp_ms = forward_ms * 0.2
        return {
            "trace_schema": "pd_decode_gpu_components_v1",
            "trial_phase": "warmup" if warmup else "measurement",
            "sm_profile_id": f"d{decode_sms}_p{40 - decode_sms}",
            "decode_sms": decode_sms,
            "prefill_sms": 40 - decode_sms,
            "decode_context_max_tokens": 100_001,
            "decode_context_min_tokens": 10_000,
            "decode_context_mean_tokens": 40_000,
            "decode_request_count": 3,
            "num_reqs": 3,
            "num_tokens": 3,
            "context_ring_decode_request_count": 0,
            "prefill_inflight_at_decode_submit": 1,
            "prefill_not_ready_at_decode_submit": 1,
            "paired_prefill_submit": 1,
            "prefill_overlap_candidate": 1,
            "green_context_active": 1,
            "green_sm_count_verified": 1,
            "component_trace_forced_eager": 1,
            "matched_modules": 96,
            "decode_stream_pipeline_ms": forward_ms * 1.1,
            "gpu_model_forward_ms": forward_ms,
            "qkv_proj_ms": forward_ms * 0.05,
            "attention_ms": attention_ms,
            "o_proj_ms": forward_ms * 0.05,
            "mlp_gate_up_proj_ms": mlp_ms * 0.4,
            "mlp_activation_ms": mlp_ms * 0.2,
            "mlp_down_proj_ms": mlp_ms * 0.4,
            "attention_stack_ms": forward_ms * 0.6,
            "mlp_total_ms": mlp_ms,
            "gpu_residual_unattributed_ms": forward_ms * 0.2,
            "logits_postprocess_ms": forward_ms * 0.04,
            "sampling_ms": forward_ms * 0.01,
            "decode_stream_interstage_residual_ms": forward_ms * 0.05,
            "num_layers": 16,
            "num_records": 96,
            "decode_request_ids": ["case-measurement-r1-anchor0"],
        }

    def test_summary_delta_and_diagnosis_compare_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for decode_sms, forward_ms in ((8, 20.0), (32, 10.0)):
                path = (
                    root
                    / "native"
                    / f"d{decode_sms}_p{40 - decode_sms}"
                    / "decode_gpu_components.jsonl"
                )
                path.parent.mkdir(parents=True)
                records = [
                    self.record(decode_sms, forward_ms * 2, warmup=True),
                    self.record(decode_sms, forward_ms, warmup=False),
                ]
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
            records = load_records(
                root,
                context_bucket_tokens=5_000,
                include_warmup=False,
            )
            self.assertEqual(len(records), 2)
            summary = summarize(records)
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary[0]["matched_modules_min"], 96)
            deltas = build_partition_deltas(summary)
            low = next(row for row in deltas if row["decode_sms"] == 8)
            self.assertAlmostEqual(
                low["gpu_model_forward_ms_ratio_vs_max_decode_sms"], 2.0
            )
            diagnoses = diagnose(summary)
            self.assertEqual(len(diagnoses), 1)
            self.assertEqual(
                diagnoses[0]["most_partition_sensitive_component"],
                "attention_ms",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
