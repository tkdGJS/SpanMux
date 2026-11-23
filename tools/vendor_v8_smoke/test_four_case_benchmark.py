#!/usr/bin/env python3
"""CPU-only contracts for the vendor_v8 four-case benchmark matrix."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "run_vendor_v8_dynamic_hbm_vs_full.sh"
RUNNER = ROOT / "run_vendor_v8_short_validation.sh"
STARTER = ROOT / "start_vllm_causality_microbench.sh"


def _run(
    script: Path,
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    for name in list(process_env):
        if name.startswith("VLLM_ISOLATION_"):
            process_env.pop(name)
    process_env.pop("ISOLATION_MATCHED_COMPARISON", None)
    process_env.pop("ISOLATION_GUARDED_2K_COMPARISON", None)
    process_env.pop("ISOLATION_BATCH2_PD11_COMPARISON", None)
    process_env.pop("BENCHMARK_VARIANTS", None)
    process_env.pop("SYSTEM_VARIANT", None)
    process_env.pop("PREFILL_SCHEDULING_POLICY", None)
    process_env.pop("DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY", None)
    process_env.pop("VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN", None)
    process_env.pop("VLLM_KV_RETENTION_LAYERWISE_OFFLOAD", None)
    for flag in ("VLLM_KV_RETENTION_P1_PARKED_PREFILL", "VLLM_KV_RETENTION_PREFILL_WORK_BINDING",
                 "VLLM_KV_RETENTION_P1_HBM_ONLY",
                 "VLLM_KV_RETENTION_PREFILL_RESUME_FRAME", "VLLM_KV_RETENTION_PARKED_MAX_BATCHES",
                 "VLLM_KV_RETENTION_PARKED_GPU_BYTES"):
        process_env.pop(flag, None)
    process_env.pop("BENCHMARK_CUDAGRAPH", None)
    process_env.pop("BENCHMARK_CUDAGRAPH_MODE", None)
    process_env.pop("ENABLE_SCHEDULER_TIMING_TRACE", None)
    process_env.pop("VLLM_PD_SERIALIZED_FULL_SM", None)
    process_env.pop("VLLM_PD_EVAL_CUDAGRAPH_MODE", None)
    process_env.pop("VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH", None)
    process_env.pop("VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF", None)
    process_env.pop("VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION", None)
    process_env.pop("VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION", None)
    process_env.pop("VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS", None)
    process_env.pop("VLLM_PD_PREFILL_WORKSPACE_ENABLED", None)
    process_env.pop("VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS", None)
    process_env.pop("VLLM_PD_PREFILL_WORKSPACE_SLOTS", None)
    process_env.pop("VLLM_PD_PREFILL_WORKSPACE_STRICT", None)
    process_env.pop("VLLM_PD_EVAL_NATIVE_PREFILL_MAX_TOKENS", None)
    process_env.pop("VLLM_PD_EVAL_PHASE_DEPTH_ENABLED", None)
    process_env.pop("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH", None)
    process_env.pop("VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH", None)
    process_env.pop("VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH", None)
    process_env.pop("VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE", None)
    process_env.pop("FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS", None)
    process_env.pop("VLLM_MAX_NUM_PARTIAL_PREFILLS", None)
    process_env.pop("VLLM_MAX_LONG_PARTIAL_PREFILLS", None)
    process_env.pop("VLLM_LONG_PREFILL_TOKEN_THRESHOLD", None)
    process_env.pop("VLLM_MAX_NUM_SEQS", None)
    process_env.pop("PD_SM_PROFILES", None)
    process_env.pop("SERVER_SHUTDOWN_TIMEOUT_SEC", None)
    process_env.pop("PORT_RELEASE_TIMEOUT_SEC", None)
    process_env.pop("CASE_RESTART_COOLDOWN_SEC", None)
    process_env.pop("KILL_EXISTING_PORT_OWNER", None)
    process_env.update(env)
    return subprocess.run(
        ["bash", str(script), "--dry-run"],
        cwd=ROOT,
        env=process_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )


class FourCaseBenchmarkTest(unittest.TestCase):
    def test_documented_hbm_layerwise_thesis_command(self):
        section = (ROOT / "docs/09_EXPERIMENT_DESIGN.md").read_text().split(
            "## P/D+SRPT HBM layerwise OFF/ON thesis ablation", 1)[1]
        block = section.split("```bash\n", 1)[1].split("```", 1)[0]
        words = shlex.split(block.replace("\\\n", ""))
        self.assertEqual(words.pop(), "./run_vendor_v8_dynamic_hbm_vs_full.sh")
        env = dict(word.split("=", 1) for word in words)
        result = _run(COMPARE, env=env)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[CASE] completed"), 2)
        self.assertEqual(result.stdout.count("loongserve_termination_mode=requests request_limit=100"), 3)
        self.assertEqual(result.stdout.count("[PLAN] kv_p1_hbm_only=1"), 1)
        self.assertEqual(result.stdout.count("[PLAN] isolation_bounded_prefill_planning=0"), 2)
        self.assertEqual(result.stdout.count("[PLAN] isolation_logical_release_readiness=0"), 2)

    def test_hbm_layerwise_ab_keeps_thesis_budgets_and_no_ring(self):
        result = _run(COMPARE, env={
            "ISOLATION_GUARDED_2K_COMPARISON": "1",
            "BENCHMARK_VARIANTS": "dynamic_hbm_srpt dynamic_hbm_srpt_layerwise",
            "VLLM_MAX_NUM_SEQS": "100", "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
            "KV_CACHE_MEMORY_BYTES": "2147483648", "VLLM_MAX_MODEL_LEN": "65536",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        args = []
        for variant, active in (("dynamic_hbm_srpt", "0"),
                                ("dynamic_hbm_srpt_layerwise", "1")):
            case = result.stdout.split(f"[CASE] begin variant={variant} ", 1)[1]
            case = case.split(f"[CASE] completed variant={variant}", 1)[0]
            self.assertIn("phase_future_depths=1/1/2 (P/D/G)", case)
            self.assertIn("hbm_completion_guard=1", case)
            self.assertIn("max_num_seqs=100 max_num_batched_tokens=16384 long_threshold=2048", case)
            self.assertIn("pd_eval_policy=muxwise_dynamic hbm_only=1", case)
            self.assertIn(f"kv_layerwise_checkpoint_plan={active} kv_layerwise_offload=0", case)
            self.assertIn(f"kv_p1_hbm_only={active}", case)
            self.assertIn(f"preemption={active} group_layers=2 scheduler_yield={active}", case)
            args.append(self.server_args(case))
        self.assertEqual(args[0], args[1])  # env selects execution; CLI budgets equal

    def test_hbm_layerwise_mode_is_case_local_even_with_proposal_selected(self):
        result = _run(COMPARE, env={
            "ISOLATION_GUARDED_2K_COMPARISON": "1",
            "VLLM_KV_RETENTION_P1_HBM_ONLY": "1",  # stale shell must not leak
            "BENCHMARK_VARIANTS": "dynamic_hbm_srpt_layerwise full_vendor_v8 dynamic_hbm_srpt baseline_plus",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[PLAN] kv_p1_hbm_only=1"), 1)
        self.assertEqual(result.stdout.count("kv_layerwise_checkpoint_plan=1 kv_layerwise_offload=1"), 1)
        self.assertEqual(result.stdout.count("[CASE] completed"), 4)

    def test_hbm_layerwise_rejects_missing_guard_or_wrong_depth(self):
        for extra in ({"ISOLATION_MATCHED_COMPARISON": "1",
                       "VLLM_ISOLATION_HBM_COMPLETION_GUARD": "0"},
                      {"ISOLATION_GUARDED_2K_COMPARISON": "1",
                       "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2"}):
            result = _run(COMPARE, env={
                "BENCHMARK_VARIANTS": "dynamic_hbm_srpt_layerwise", **extra})
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("[CASE] completed", result.stdout)

    def test_documented_graph_on_memory_option_block(self):
        section = (ROOT / "docs/09_EXPERIMENT_DESIGN.md").read_text().split(
            "## Graph-ON startup-memory follow-up", 1)[1]
        block = section.split("```bash\n", 1)[1].split("```", 1)[0]
        env = dict(word.split("=", 1) for word in shlex.split(block.replace("\\\n", "")))
        result = _run(COMPARE, env=env)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[CASE] completed"), 5)
        self.assertEqual(result.stdout.count("cudagraph_mode=FULL_AND_PIECEWISE green_decode_cudagraph="), 5)
        self.assertEqual(result.stdout.count("[PLAN] startup_cache_trim=1 startup_memory_audit=1"), 5)

    def test_common_startup_memory_with_graph_on_and_off(self):
        for graph, mode in (("1", "FULL_AND_PIECEWISE"), ("0", "NONE")):
            result = _run(COMPARE, env={
                "ISOLATION_GUARDED_2K_COMPARISON": "1",
                "BENCHMARK_CUDAGRAPH": graph,
                "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
                "VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE": "1",
                "VLLM_ISOLATION_RING_WORKSPACE_SHARING": "1",
                "VLLM_ISOLATION_STARTUP_CACHE_TRIM": "1",
                "VLLM_ISOLATION_STARTUP_MEMORY_AUDIT": "1",
            })
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout.count("[CASE] completed"), 5)
            self.assertEqual(result.stdout.count(
                "[PLAN] startup_cache_trim=1 startup_memory_audit=1 scope=before_capture_only"), 5)
            self.assertEqual(result.stdout.count(
                "[PLAN] prefill_workspace_rightsize=1 ring_workspace_sharing=1"), 3)
            self.assertEqual(result.stdout.count("cudagraph_mode=" + mode + " green_decode_cudagraph="), 5)

    def test_startup_memory_flags_default_off_and_invalid(self):
        result = _run(COMPARE, env={"ISOLATION_GUARDED_2K_COMPARISON": "1",
                                     "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[PLAN] startup_cache_trim=0 startup_memory_audit=0"), 5)
        for script in (COMPARE, RUNNER):
            for flag in ("STARTUP_CACHE_TRIM", "STARTUP_MEMORY_AUDIT"):
                result = _run(script, env={"VLLM_ISOLATION_" + flag: "yes"})
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertNotIn("[CASE] begin", result.stdout)

    def test_startup_memory_preflight_and_settings_before_server_start(self):
        source = RUNNER.read_text()
        start = source.index('ACTIVE_SERVER_PID="$!"')
        self.assertLess(source.index("STARTUP_MEMORY_PROBE="), start)
        for name in ("startup_cache_trim", "startup_memory_audit", "startup_memory_deployment"):
            self.assertLess(source.index('"' + name + "="), start)
            self.assertIn("\n" + name + "=", source)
        self.assertIn('"${ISOLATION_MEMORY_FEATURES[@]}" STARTUP_CACHE_TRIM STARTUP_MEMORY_AUDIT;', source)

    def test_process_lifetime_default_common_pd_and_native_off(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "BENCHMARK_VARIANTS": "native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[PLAN] green_context_process_lifetime=1"), 3)
        self.assertEqual(result.stdout.count("[PLAN] green_context_process_lifetime=0"), 2)

    def test_process_lifetime_rollback_and_invalid(self):
        result = _run(COMPARE, env={
            "BENCHMARK_VARIANTS": "full_vendor_v8",
            "VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME": "0",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[PLAN] green_context_process_lifetime=0", result.stdout)
        for script in (COMPARE, RUNNER):
            result = _run(script, env={"VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME": "bad"})
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("[START]", result.stdout)

    def test_predictive_policy_is_common_pd_only_and_calibration_is_explicit(self):
        result = _run(COMPARE, env={
            "BENCHMARK_VARIANTS": "native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8",
            "PD_DYNAMIC_SM_POLICY": "tbt_slo_predictive",
            "VLLM_PD_TBT_PREDICTIVE_LEAD_MS": "40",
            "VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS": "2",
            "VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS": "10",
            "TBT_OVERLAP_PROFILE": "/synthetic/predictor.json",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("tbt_predictive_applicable=1 lead_ms=40 cutover_ms=2 min_step_ms=10"), 3)
        self.assertEqual(result.stdout.count("tbt_predictive_applicable=0"), 2)
        self.assertEqual(result.stdout.count("[PLAN] tbt_profile=/synthetic/predictor.json"), 3)
        source = RUNNER.read_text()
        self.assertLess(source.index("tbt_predictive_profile_contract=validated"), source.index('ACTIVE_SERVER_PID="$!"'))
        for suffix in ("LEAD_MS", "CUTOVER_MS", "MIN_STEP_MS"):
            self.assertIn(f'VLLM_PD_TBT_PREDICTIVE_{suffix}="${{TBT_PREDICTIVE_{suffix}}}"', source)

    def test_documented_batch2_loongserve_command_resolves_all_five_cases(self):
        doc = (ROOT / "docs/09_EXPERIMENT_DESIGN.md").read_text()
        section = doc.split("## Batch2 versus P/D1+1 guarded2K diagnostic", 1)[1]
        command = section.split("```bash\n", 1)[1].split("```", 1)[0]
        words = shlex.split(command.replace("\\\n", ""))
        self.assertEqual(words[-1], "./run_vendor_v8_dynamic_hbm_vs_full.sh")
        env = dict(word.split("=", 1) for word in words[:-1])
        # Baseline+SRPT validates the directory even in dry-run. Only relocate
        # that remote path for this local fixture; keep every option unchanged.
        env["VLLM_VENDOR_DIR"] = str(ROOT / "vendor_v8")
        result = _run(COMPARE, env=env)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[CASE] completed"), 5)
        self.assertEqual(result.stdout.count("batch2_pd11=1 prefill_batch_max_requests=1 decode_batch_max_requests=1"), 3)
        self.assertEqual(result.stdout.count("batch2_pd11=1 prefill_batch_max_requests=0 decode_batch_max_requests=0"), 2)
        self.assertIn("loongserve_termination_mode=requests request_limit=100", result.stdout)
        self.assertIn("isolation_execution=1 ring_query=0 p1_boundary=1", result.stdout)

    def test_batch2_pd11_five_case_actual_request_caps(self):
        result = _run(COMPARE, env={"ISOLATION_BATCH2_PD11_COMPARISON": "1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("batch2_pd11=1 prefill_batch_max_requests=0 decode_batch_max_requests=0"), 2)
        self.assertEqual(result.stdout.count("batch2_pd11=1 prefill_batch_max_requests=1 decode_batch_max_requests=1"), 3)
        self.assertEqual(result.stdout.count("prefill_admission=native_budget_fill max_num_seqs=2 max_num_batched_tokens=16384 long_threshold=2048"), 5)
        self.assertEqual(result.stdout.count("phase_future_depths=1/1/2"), 3)
        self.assertEqual(result.stdout.count("kv_p1_parked=1 work_binding=1 resume_frame=1"), 1)
        self.assertEqual(result.stdout.count("[CASE] completed"), 5)
        self.assertIn("vendor_v8_guarded_2k_batch2_pd11_", result.stdout)

    def test_batch2_conflict_rejected_before_any_case(self):
        for flag, value in (("VLLM_MAX_NUM_SEQS", "10"),
                            ("ISOLATION_GUARDED_2K_COMPARISON", "0"),
                            ("VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS", "2"),
                            ("VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS", "0")):
            with self.subTest(flag=flag):
                result = _run(COMPARE, env={"ISOLATION_BATCH2_PD11_COMPARISON": "1", flag: value})
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertNotIn("[CASE] begin", result.stdout)

    def test_phase_caps_are_optional_and_do_not_leak_to_native(self):
        result = _run(COMPARE, env={"ISOLATION_GUARDED_2K_COMPARISON": "1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("prefill_batch_max_requests=0 decode_batch_max_requests=0"), 5)
        result = _run(RUNNER, env={"SYSTEM_VARIANT": "baseline_plus",
                                   "VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS": "1"})
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("phase batch limits are P/D-only", result.stdout)

    def test_phase_caps_are_exported_and_snapshotted_before_server_start(self):
        source = RUNNER.read_text()
        self.assertIn('export "${batch_flag}=${batch_value}"', source)
        for name in ("isolation_batch2_pd11_comparison", "prefill_batch_max_requests",
                     "decode_batch_max_requests", "phase_batch_limit_deployment"):
            self.assertIn(f'"{name}=', source)
            self.assertIn(f'\n{name}=', source)
        self.assertLess(source.index("PHASE_BATCH_LIMIT_PROBE="), source.index('ACTIVE_SERVER_PID="$!"'))

    def test_phase_deployment_probe_rejects_stale_runtime(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        probe = RUNNER.read_text().split("PHASE_BATCH_LIMIT_PROBE='", 1)[1].split("'\n  PHASE_BATCH_LIMIT_RESULT=", 1)[0]
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(
                __file__=str(ROOT / "vendor_v8/vllm/__init__.py"))}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(probe, "<phase-deployment-probe>", "exec"), {})
            self.assertIn("frontier_sha256", output.getvalue())
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(
                __file__=str(ROOT / "src/vllm-0.14-src/vllm/__init__.py"))}):
            with self.assertRaisesRegex(SystemExit, "missing phase batch request limits"):
                exec(compile(probe, "<phase-deployment-probe>", "exec"), {})

    def test_guarded_2k_five_case_contract(self):
        result = _run(COMPARE, env={"ISOLATION_GUARDED_2K_COMPARISON": "1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("selected_variants=native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8", result.stdout)
        self.assertEqual(result.stdout.count("prefill_admission=native_budget_fill max_num_seqs=100 max_num_batched_tokens=16384 long_threshold=2048"), 5)
        self.assertEqual(result.stdout.count("guarded_2k=1 hbm_completion_guard=1 decode_output_first=0 next_chunk_fit=0"), 2)
        self.assertEqual(result.stdout.count("guarded_2k=1 hbm_completion_guard=1 decode_output_first=1 next_chunk_fit=0"), 2)
        self.assertEqual(result.stdout.count("guarded_2k=1 hbm_completion_guard=0 decode_output_first=1 next_chunk_fit=1"), 1)
        self.assertEqual(result.stdout.count("native_prefill_max_tokens=2048"), 3)
        self.assertEqual(result.stdout.count("phase_future_depths=1/1/2"), 3)
        self.assertEqual(result.stdout.count("kv_p1_parked=1 work_binding=1 resume_frame=1"), 1)

    def test_guarded_2k_keeps_aggregate_budget_independent(self):
        result = _run(COMPARE, env={"ISOLATION_GUARDED_2K_COMPARISON": "1",
                                   "VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
                                   "VLLM_MAX_NUM_SEQS": "8"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("max_num_seqs=8 max_num_batched_tokens=8192 long_threshold=2048"), 6)
        self.assertEqual(result.stdout.count("native_prefill_max_tokens=2048"), 3)

    def test_guarded_2k_conflicts_rejected_before_cases(self):
        for flag, value in (("PREFILL_CHUNK_TOKENS", "16384"),
                            ("VLLM_LONG_PREFILL_TOKEN_THRESHOLD", "0"),
                            ("VLLM_PD_EVAL_NATIVE_PREFILL_MAX_TOKENS", "16384"),
                            ("VLLM_ISOLATION_RING_QUERY_EXPANSION", "1"),
                            ("VLLM_KV_RETENTION_P1_PARKED_PREFILL", "0")):
            with self.subTest(flag=flag):
                result = _run(COMPARE, env={"ISOLATION_GUARDED_2K_COMPARISON": "1", flag: value})
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertNotIn("[CASE] begin", result.stdout)

    def test_guarded_contract_and_probe_exist_before_start(self):
        source = RUNNER.read_text()
        for key in ("isolation_guarded_2k_comparison", "hbm_completion_guard",
                    "decode_output_first", "prefill_fit_next_chunk"):
            self.assertIn(f'"{key}=', source)
        self.assertLess(source.index("HBM_GUARD_PROBE="), source.index('ACTIVE_SERVER_PID="$!"'))
        self.assertIn('"${VLLM_PYTHON_EXECUTABLE}" -c "${HBM_GUARD_PROBE}"', source)

    def test_guarded_deployment_probe_accepts_both_patched_sources(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        source = RUNNER.read_text()
        probe = source.split("HBM_GUARD_PROBE='", 1)[1].split("'\n  if", 1)[0]
        for root in ("vendor_v8", "src/vllm-0.14-src"):
            with patch.dict(sys.modules, {"vllm": SimpleNamespace(
                    __file__=str(ROOT / root / "vllm/__init__.py"))}):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(compile(probe, "<guard-deployment-probe>", "exec"), {})
                self.assertIn("helper_sha256", output.getvalue())
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(
                __file__=str(ROOT / "tools/not-a-vllm-package/__init__.py"))}):
            with self.assertRaisesRegex(SystemExit, "missing HBM completion guard"):
                exec(compile(probe, "<guard-deployment-probe>", "exec"), {})

    def test_waiting_decode_reclaim_independent_opt_in_and_control_isolation(self):
        key = "VLLM_ISOLATION_WAITING_DECODE_RECLAIM"
        variants = ("native_40sm_chunked", "baseline_plus",
                    "dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt", "full_vendor_v8")
        common = {"ISOLATION_GUARDED_2K_COMPARISON": "1",
                  "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1"}
        for value in (None, "1", "0"):
            env = {**common, "BENCHMARK_VARIANTS": " ".join(variants)}
            if value is not None:
                env[key] = value
            result = _run(COMPARE, env=env)
            self.assertEqual(result.returncode, 0, result.stdout)
            count = 1 if value == "1" else 0
            self.assertEqual(result.stdout.count("[PLAN] isolation_waiting_decode_reclaim=1"), count)
            self.assertEqual(result.stdout.count("[PLAN] isolation_waiting_decode_reclaim=0"), 5 - count)
        for variant in variants:
            result = _run(RUNNER, env={"SYSTEM_VARIANT": variant, key: "1"})
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("isolation_waiting_decode_reclaim=" + str(int(variant == "full_vendor_v8")), result.stdout)
        for script in (COMPARE, RUNNER):
            result = _run(script, env={**common, key: "invalid"})
            self.assertEqual(result.returncode, 2, result.stdout)
        source = RUNNER.read_text()
        self.assertIn("isolation_waiting_decode_reclaim=${" + key + "}", source)
        self.assertIn('done >"${RESULT_DIR}/experiment_snapshot/isolation_execution.txt"', source)

    def test_bounded_prefill_readiness_common_pd_routing_and_rollback(self):
        key = "VLLM_ISOLATION_BOUNDED_PREFILL_PLANNING"
        variants = ("native_40sm_chunked", "baseline_plus",
                    "dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt", "full_vendor_v8")
        common = {"ISOLATION_GUARDED_2K_COMPARISON": "1",
                  "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1"}
        # Unset inherits the bundle; explicit zero restores prior admission.
        for value in (None, "1", "0"):
            with self.subTest(value=value):
                env = {**common, "BENCHMARK_VARIANTS": " ".join(variants)}
                if value is not None:
                    env[key] = value
                result = _run(COMPARE, env=env)
                self.assertEqual(result.returncode, 0, result.stdout)
                enabled = 0 if value == "0" else 3
                self.assertEqual(result.stdout.count(
                    "[PLAN] isolation_bounded_prefill_planning=1"), enabled)
                self.assertEqual(result.stdout.count(
                    "[PLAN] isolation_bounded_prefill_planning=0"), 5 - enabled)
        for script in (COMPARE, RUNNER):
            result = _run(script, env={**common, key: "invalid"})
            self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("isolation_bounded_prefill_planning=${" + key + "}", RUNNER.read_text())

    def test_logical_release_readiness_routing_and_rollback(self):
        key = "VLLM_ISOLATION_LOGICAL_RELEASE_READINESS"
        variants = ("native_40sm_chunked", "baseline_plus",
                    "dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt", "full_vendor_v8")
        common = {"ISOLATION_GUARDED_2K_COMPARISON": "1",
                  "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1"}
        result = _run(COMPARE, env={**common, "BENCHMARK_VARIANTS": " ".join(variants)})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[PLAN] isolation_logical_release_readiness=0"), 2)
        self.assertEqual(result.stdout.count("[PLAN] isolation_logical_release_readiness=1"), 3)
        for variant in variants:
            # As in the guarded routing test below, the direct runner does not
            # expand the comparison wrapper's matched/2K preset itself.
            result = _run(RUNNER, env={**common, "ISOLATION_GUARDED_2K_COMPARISON": "0",
                "VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL": "0",
                "SYSTEM_VARIANT": variant, key: "0"})
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("[PLAN] isolation_logical_release_readiness=0", result.stdout)
        for script in (COMPARE, RUNNER):
            result = _run(script, env={**common, key: "invalid"})
            self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("isolation_logical_release_readiness=${" + key + "}", RUNNER.read_text())

    def test_explicit_proposal_flags_cannot_leak_through_either_launcher(self):
        flags = ("D2H_EXACT_PLAN_PROGRESS", "NATIVE_SOFT_HEADROOM_PROGRESS",
                 "PREFILL_BATCH_KV_BUDGET")
        common = {
            "ISOLATION_GUARDED_2K_COMPARISON": "1",
            "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1",
            **{f"VLLM_ISOLATION_{name}": "1" for name in flags},
        }
        variants = ("native_40sm_chunked", "baseline_plus",
                    "dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt",
                    "full_vendor_v8")
        sweep = _run(COMPARE, env={**common, "BENCHMARK_VARIANTS": " ".join(variants)})
        self.assertEqual(sweep.returncode, 0, sweep.stdout)
        for name in flags:
            key = f"[PLAN] isolation_{name.lower()}="
            self.assertEqual(sweep.stdout.count(key + "0"), 4)
            self.assertEqual(sweep.stdout.count(key + "1"), 1)
        self.assertEqual(sweep.stdout.count("hbm_completion_guard=1"), 4)
        self.assertEqual(sweep.stdout.count("long_threshold=2048"), 6)
        for variant in variants:
            with self.subTest(variant=variant):
                # Guarded/matched presets are expanded by the wrapper, not by
                # this low-level entry point. Test direct option sanitization
                # independently; the guarded sweep above checks its budgets.
                direct = _run(RUNNER, env={
                    **common, "ISOLATION_GUARDED_2K_COMPARISON": "0",
                    "VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL": "0",
                    "SYSTEM_VARIANT": variant,
                })
                self.assertEqual(direct.returncode, 0, direct.stdout)
                for name in flags:
                    expected = int(variant == "full_vendor_v8")
                    self.assertIn(f"[PLAN] isolation_{name.lower()}={expected}",
                                  direct.stdout)

    def test_case_log_preserves_early_failure_and_appends_retry(self):
        source = COMPARE.read_text()
        helper = source.split("run_case_with_log() {", 1)[1].split(
            '\nfor variant in "${benchmark_variants[@]}";', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            harness = f"""
set -uo pipefail
ROOT_DIR={shlex.quote(directory)}
DRY_RUN=0
experiment_name=test_native
variant=native_40sm_chunked
run_case_with_log() {{{helper}
run_case_with_log bash -c 'echo "missing HBM completion guard" >&2; exit 2'
first_status=$?
run_case_with_log bash -c 'echo "[PLAN] native import validated"'
second_status=$?
[[ "$first_status" == 2 && "$second_status" == 0 ]]
"""
            result = subprocess.run(["bash", "-c", harness], text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            log = Path(directory, "test_native_launcher.log").read_text()
            self.assertIn("missing HBM completion guard", log)
            self.assertIn("[PLAN] native import validated", log)
            self.assertIn("exit_status=2 log_status=0", log)
            self.assertEqual(log.count("[CASE] begin"), 2)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["test_native_launcher.log"])
        self.assertIn("if run_case_with_log env", source)

    def test_case_log_preserves_unsafe_cleanup_over_logger_failure(self):
        source = COMPARE.read_text()
        helper = source.split("run_case_with_log() {", 1)[1].split(
            '\nfor variant in "${benchmark_variants[@]}";', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            for child_status, expected in ((86, 86), (2, 2), (0, 2)):
                with self.subTest(child_status=child_status):
                    harness = f"""
set -uo pipefail
ROOT_DIR={shlex.quote(directory)}
DRY_RUN=0
experiment_name=test_child
variant=baseline_plus
run_case_with_log() {{{helper}
tee() {{ command cat; return 1; }}
run_case_with_log bash -c 'echo terminal; exit {child_status}'
"""
                    result = subprocess.run(["bash", "-c", harness], text=True,
                                            capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, expected,
                                     result.stdout + result.stderr)

    def test_case_logging_dry_run_has_no_files_and_bad_log_prevents_launch(self):
        source = COMPARE.read_text()
        helper = source.split("run_case_with_log() {", 1)[1].split(
            '\nfor variant in "${benchmark_variants[@]}";', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            for dry_run, expected in ((1, 7), (0, 2)):
                harness = f"""
set -uo pipefail
ROOT_DIR={shlex.quote(str(Path(directory, 'absent')))}
DRY_RUN={dry_run}
experiment_name=test_child
variant=baseline_plus
run_case_with_log() {{{helper}
run_case_with_log bash -c 'echo child-ran; exit 7'
"""
                result = subprocess.run(["bash", "-c", harness], text=True,
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, expected,
                                 result.stdout + result.stderr)
                self.assertEqual("child-ran" in result.stdout, bool(dry_run))
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_execution_bundle_routes_ring_only_to_proposal_common_scratch_to_pd(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_KV_RETENTION_P1_PARKED_PREFILL": "1",
            "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1",
            "BENCHMARK_VARIANTS": "native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("isolation_execution=0 ring_query=0 p1_boundary=0 prefill_rope=0 text_inputs=0 memory_snapshots=0"), 2)
        self.assertEqual(result.stdout.count("isolation_execution=1 ring_query=0 p1_boundary=0 prefill_rope=1 text_inputs=1 memory_snapshots=1"), 2)
        self.assertEqual(result.stdout.count("isolation_execution=1 ring_query=1 p1_boundary=1 prefill_rope=1 text_inputs=1 memory_snapshots=1"), 1)
        self.assertEqual(result.stdout.count("prefill_admission=native_budget_fill max_num_seqs=100 max_num_batched_tokens=16384"), 5)
        self.assertEqual(result.stdout.count("[PLAN] isolation_p1_ring_sm_handoff=1"), 1)
        self.assertEqual(result.stdout.count("[PLAN] isolation_p1_ring_sm_handoff=0"), 4)
        self.assertEqual(result.stdout.count("[PLAN] isolation_dram_refresh_probe=1"), 1)
        self.assertEqual(result.stdout.count("[PLAN] isolation_dram_refresh_probe=0"), 4)
        self.assertEqual(result.stdout.count("[PLAN] isolation_d2h_exact_plan_progress=1"), 1)
        self.assertEqual(result.stdout.count("[PLAN] isolation_d2h_exact_plan_progress=0"), 4)
        self.assertEqual(result.stdout.count("[PLAN] isolation_native_soft_headroom_progress=1"), 1)
        self.assertEqual(result.stdout.count("[PLAN] isolation_native_soft_headroom_progress=0"), 4)
        self.assertEqual(result.stdout.count("[PLAN] isolation_prefill_batch_kv_budget=1"), 1)
        self.assertEqual(result.stdout.count("[PLAN] isolation_prefill_batch_kv_budget=0"), 4)

    def test_ring_sm_independent_rollback_keeps_other_execution_features(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "BENCHMARK_VARIANTS": "full_vendor_v8",
            "VLLM_KV_RETENTION_P1_PARKED_PREFILL": "1",
            "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1",
            "VLLM_ISOLATION_P1_RING_SM_HANDOFF": "0",
            "VLLM_ISOLATION_DRAM_REFRESH_PROBE": "0",
            "VLLM_ISOLATION_D2H_EXACT_PLAN_PROGRESS": "0",
            "VLLM_ISOLATION_NATIVE_SOFT_HEADROOM_PROGRESS": "0",
            "VLLM_ISOLATION_PREFILL_BATCH_KV_BUDGET": "0",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[PLAN] isolation_p1_ring_sm_handoff=0", result.stdout)
        self.assertIn("[PLAN] isolation_dram_refresh_probe=0", result.stdout)
        self.assertIn("[PLAN] isolation_d2h_exact_plan_progress=0", result.stdout)
        self.assertIn("[PLAN] isolation_native_soft_headroom_progress=0", result.stdout)
        self.assertIn("[PLAN] isolation_prefill_batch_kv_budget=0", result.stdout)
        self.assertIn("isolation_execution=1 ring_query=1 p1_boundary=1", result.stdout)

    def test_execution_bundle_rollback_and_bad_flags(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1", "BENCHMARK_VARIANTS": "full_vendor_v8",
            "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1",
            "VLLM_ISOLATION_RING_QUERY_EXPANSION": "0",
            "VLLM_ISOLATION_PREFILL_FUSED_ROPE": "0",
            "VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL": "0",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("isolation_execution=1 ring_query=0 p1_boundary=1 prefill_rope=0", result.stdout)
        for setting in ({"VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "yes"},
                        {"VLLM_ISOLATION_P1_RING_SM_HANDOFF": "yes"},
                        {"VLLM_ISOLATION_DRAM_REFRESH_PROBE": "yes"},
                        {"VLLM_ISOLATION_D2H_EXACT_PLAN_PROGRESS": "yes"},
                        {"VLLM_ISOLATION_NATIVE_SOFT_HEADROOM_PROGRESS": "yes"},
                        {"VLLM_ISOLATION_TEXT_INPUT_BUFFERS": "2"},
                        {"VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES": "0"}):
            result = _run(RUNNER, env=setting)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("[START]", result.stdout)

    def test_memory_bundle_is_common_vendor_only_and_keeps_default_ceiling(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_KV_RETENTION_P1_PARKED_PREFILL": "1",
            "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
            "VLLM_ISOLATION_RING_STATE_REUSE": "1",
            "BENCHMARK_VARIANTS": "native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count(
            "[PLAN] isolation_memory=1 norm=1 workspace=1 ring_state=1 ring_kernel=1 decode_buffers=1 prefetch=1 graph_audit=1"), 3)
        self.assertEqual(result.stdout.count(
            "[PLAN] isolation_memory=0 norm=0 workspace=0 ring_state=0 ring_kernel=0 decode_buffers=0 prefetch=0 graph_audit=0"), 2)
        self.assertEqual(result.stdout.count(
            "prefill_admission=native_budget_fill max_num_seqs=100 max_num_batched_tokens=16384"), 5)

    def test_memory_independent_rollback_and_validation(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
            "VLLM_ISOLATION_RING_DYNAMIC_KERNEL": "0",
            "BENCHMARK_VARIANTS": "dynamic_hbm_serialized_fullsm full_vendor_v8",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("ring_kernel=0 decode_buffers=1"), 2)
        result = _run(COMPARE, env={"VLLM_ISOLATION_RING_STATE_REUSE": "yes"})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("[CASE] begin", result.stdout)

    def test_workspace_rightsize_and_sharing_common_pd_flags(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
            "VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE": "1",
            "VLLM_ISOLATION_RING_WORKSPACE_SHARING": "1",
            "BENCHMARK_VARIANTS": "native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count(
            "prefill_workspace_rightsize=1 ring_workspace_sharing=1"), 3)
        self.assertEqual(result.stdout.count(
            "prefill_workspace_rightsize=0 ring_workspace_sharing=0"), 2)
        self.assertEqual(result.stdout.count(
            "prefill_admission=native_budget_fill max_num_seqs=100 max_num_batched_tokens=16384"), 5)

    def test_workspace_flags_do_not_inherit_bundle_and_reject_bad_values(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
            "BENCHMARK_VARIANTS": "full_vendor_v8",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("prefill_workspace_rightsize=0 ring_workspace_sharing=0", result.stdout)
        result = _run(COMPARE, env={"VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE": "yes"})
        self.assertNotEqual(result.returncode, 0)
        result = _run(RUNNER, env={
            "VLLM_PD_PREFILL_WORKSPACE_ENABLED": "1",
            "VLLM_ISOLATION_RING_WORKSPACE_SHARING": "1",
            "VLLM_ISOLATION_RING_STATE_REUSE": "0",
        })
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sharing requires VLLM_ISOLATION_RING_STATE_REUSE=1", result.stdout)

    def test_memory_direct_native_controls_clear_inherited_overrides(self):
        for variant in ("native_40sm_chunked", "baseline_plus"):
            result = _run(RUNNER, env={
                "SYSTEM_VARIANT": variant,
                "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
                "VLLM_ISOLATION_FUSED_RMS_NORM": "1",
                "VLLM_ISOLATION_COMPACT_DECODE_BUFFERS": "1",
            })
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "[PLAN] isolation_memory=0 norm=0 workspace=0 ring_state=0 ring_kernel=0 decode_buffers=0 prefetch=0 graph_audit=0",
                result.stdout)

    def test_p1_proposal_isolation_and_common_budgets_for_five_cases(self):
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_KV_RETENTION_P1_PARKED_PREFILL": "1",
            "BENCHMARK_VARIANTS": "native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8",
            "VLLM_MAX_NUM_SEQS": "8", "VLLM_MAX_NUM_BATCHED_TOKENS": "16384"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("[CASE] completed"), 5)
        self.assertEqual(result.stdout.count("kv_p1_parked=1 work_binding=1 resume_frame=1"), 1)
        self.assertEqual(result.stdout.count("kv_p1_parked=0 work_binding=0 resume_frame=0"), 4)
        self.assertEqual(result.stdout.count("prefill_admission=native_budget_fill max_num_seqs=8 max_num_batched_tokens=16384"), 5)
        self.assertEqual(result.stdout.count("phase_future_depths=1/1/2 (P/D/G)"), 3)
        self.assertEqual(result.stdout.count("kv_layerwise_checkpoint_plan=1 kv_layerwise_offload=1"), 1)

    def test_p1_unsafe_parked_budget_rejected_before_server_start(self):
        for env in ({"VLLM_KV_RETENTION_PARKED_MAX_BATCHES": "2"},
                    {"VLLM_KV_RETENTION_PARKED_GPU_BYTES": "0"}):
            result = _run(COMPARE, env={"ISOLATION_MATCHED_COMPARISON": "1",
                "VLLM_KV_RETENTION_P1_PARKED_PREFILL": "1",
                "BENCHMARK_VARIANTS": "full_vendor_v8", **env})
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("[CASE] completed", result.stdout)

    def test_matched_frame_without_p1_bundle_is_not_silently_accepted(self):
        result = _run(COMPARE, env={"ISOLATION_MATCHED_COMPARISON": "1",
            "BENCHMARK_VARIANTS": "full_vendor_v8",
            "VLLM_KV_RETENTION_PREFILL_WORK_BINDING": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("[CASE] completed", result.stdout)

    def test_matched_default_selects_only_requested_three_cases(self) -> None:
        result = _run(COMPARE, env={"ISOLATION_MATCHED_COMPARISON": "1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "selected_variants=native_40sm_chunked dynamic_hbm_serialized_fullsm full_vendor_v8\n",
            result.stdout,
        )
        self.assertEqual(result.stdout.count("[CASE] completed"), 3)

    def test_matched_default100_is_common_to_all_five_cases(self) -> None:
        variants = (
            "native_40sm_chunked", "baseline_plus",
            "dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt",
            "full_vendor_v8",
        )
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_KV_RETENTION_P1_PARKED_PREFILL": "1",
            "BENCHMARK_VARIANTS": " ".join(variants),
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        for variant in variants:
            with self.subTest(variant=variant):
                output = self.case_output(result.stdout, variant)
                self.assertIn(
                    "max_num_seqs=100 max_num_batched_tokens=16384", output)
                self.assertIn(
                    "prefill_batch_max_requests=0 decode_batch_max_requests=0", output)
                args = self.server_args(output)
                self.assertEqual(
                    args[args.index("--cudagraph-capture-sizes") + 1:],
                    [str(n) for n in sorted({1, 2, 4, 100, *range(8, 201, 8)})],
                )
                if variant in {"dynamic_hbm_serialized_fullsm",
                               "dynamic_hbm_srpt", "full_vendor_v8"}:
                    self.assertIn("phase_future_depths=1/1/2", output)

    def test_sequence_default_resolution_in_both_entrypoints(self) -> None:
        # Execute the actual shell assignment independently of GPU/launch
        # prerequisites; explicit overrides remain available for ablations.
        for script in (COMPARE, RUNNER):
            source = script.read_text()
            assignments = [line for line in source.splitlines()
                           if line.startswith('VLLM_MAX_NUM_SEQS=')]
            self.assertEqual(len(assignments), 1)
            block = assignments[0]
            for matched, explicit, expected in (
                ("1", None, "100"), ("0", None, "100"),
                ("1", "", "100"), ("0", "", "100"),
                ("1", "10", "10"), ("1", "7", "7"),
                ("1", "16", "16"), ("0", "16", "16"),
            ):
                with self.subTest(script=script.name, matched=matched, explicit=explicit):
                    env = os.environ.copy()
                    env.pop("VLLM_MAX_NUM_SEQS", None)
                    env["ISOLATION_MATCHED_COMPARISON"] = matched
                    if explicit is not None:
                        env["VLLM_MAX_NUM_SEQS"] = explicit
                    result = subprocess.run(
                        ["bash", "-c", block + '\nprintf "%s" "$VLLM_MAX_NUM_SEQS"'],
                        env=env, text=True, capture_output=True, check=True,
                    )
                    self.assertEqual(result.stdout, expected)

    def test_direct_runner_default100_and_explicit10_use_native_budgets(self) -> None:
        for explicit, expected in ((None, 100), ("10", 10)):
            with self.subTest(explicit=explicit):
                # Direct runner has no wrapper to resolve matched prerequisites.
                env = {"SYSTEM_VARIANT": "dynamic_hbm_srpt"}
                if explicit is not None:
                    env["VLLM_MAX_NUM_SEQS"] = explicit
                result = _run(RUNNER, env=env)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(
                    f"prefill_admission=native_budget_fill max_num_seqs={expected} max_num_batched_tokens=16384",
                    result.stdout)
                self.assertIn(
                    "prefill_batch_max_requests=0 decode_batch_max_requests=0",
                    result.stdout)

    def test_native_frontier_fills_tokens_up_to100_without_phase_quota(self):
        # Exercise the actual CPU-only frontier, not a model of its algorithm.
        # KV allocation remains a separate final check after this token plan.
        from test_slo_prefill_scheduler import MODULE as policy

        for tokens, budget, slots, width in (
            (128, 16384, 100, 100),  # short rows can exceed the archived10
            (2048, 16384, 100, 8),  # full2K chunks exhaust aggregate16K
            (128, 1024, 100, 8),    # lower token budget remains authoritative
            (128, 16384, 10, 10),   # explicit archived ceiling remains valid
        ):
            with self.subTest(tokens=tokens, budget=budget, slots=slots):
                candidates = [policy.PrefillAdmissionCandidate(
                    str(i), tokens, True, False) for i in range(120)]
                frontier = policy.build_native_prefill_admission_frontier(
                    candidates, token_budget_before=budget,
                    new_sequence_slots=slots, request_limit=0)
                self.assertEqual(frontier.request_ids,
                                 tuple(str(i) for i in range(width)))
                self.assertEqual(frontier.token_budget_after,
                                 budget - width * tokens)

    def test_matched_budget_override_applies_to_both_vendor_native_paths(self) -> None:
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "VLLM_MAX_NUM_SEQS": "7",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "12288",
            "VLLM_LONG_PREFILL_TOKEN_THRESHOLD": "2048",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count(
            "[PLAN] prefill_admission=native_budget_fill max_num_seqs=7 max_num_batched_tokens=12288 long_threshold=2048"
        ), 3)
        self.assertEqual(result.stdout.count("[PLAN] prefill_workspace=1 max_tokens=12288 slots=1"), 2)
        self.assertEqual(result.stdout.count("native_prefill_max_tokens=12288"), 2)

    def test_matched_p1_comparison_common_budgets_and_execution_controls(self) -> None:
        variants = (
            "native_40sm_chunked",
            "dynamic_hbm_serialized_fullsm",
            "full_vendor_v8",
        )
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "BENCHMARK_VARIANTS": " ".join(variants),
            "WORKLOAD": "LOONGSERVE",
            "LOONGSERVE_TRACE_JSONL": "/tmp/immutable-trace.jsonl",
            "LOONGSERVE_TERMINATION_MODE": "requests",
            "LOONGSERVE_REQUEST_LIMIT": "100",
            "KV_CACHE_MEMORY_BYTES": "2147483648",
            "VLLM_MAX_NUM_SEQS": "10",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        for variant in variants:
            with self.subTest(variant=variant):
                output = self.case_output(result.stdout, variant)
                args = self.server_args(output)
                for expected in (
                    "isolation_matched_comparison=1",
                    "loongserve_trace=/tmp/immutable-trace.jsonl",
                    "loongserve_termination_mode=requests request_limit=100",
                    "prefill_admission=native_budget_fill max_num_seqs=10 max_num_batched_tokens=16384",
                    "full_iteration_trace=0 sample_every=100",
                    "strict_bounded_lightweight_timeline=1",
                    "virtual_context_timing=0",
                    "fused_execute_sample=0",
                    "ttft_layerwise_prefill_execution=0 preemption=0",
                    "kv_layerwise_checkpoint_plan=0 kv_layerwise_offload=0",
                    "cudagraph_mode=FULL_AND_PIECEWISE",
                ):
                    self.assertIn(expected, output)
                self.assertEqual(
                    args[args.index("--kv-cache-memory-bytes") + 1],
                    "2147483648",
                )
                self.assertEqual(
                    args[args.index("--cudagraph-capture-sizes") + 1:],
                    ["1", "2", "4", "8", "10", "16"],
                )
                if variant == "native_40sm_chunked":
                    self.assertIn("layerwise_profile_handoff=0", output)
                    self.assertIn("prefill_workspace=0", output)
                    self.assertNotIn("--enable-ttft-prefill-chunk-scheduling", args)
                else:
                    self.assertIn("layerwise_profile_handoff=1", output)
                    self.assertIn("prefill_workspace=1 max_tokens=16384 slots=1", output)
                    self.assertIn("phase_future_depths=1/1/2", output)
                    self.assertIn("native_prefill_max_tokens=16384", output)
                    self.assertIn("green_decode_cudagraph=1", output)
                    self.assertIn("--enable-ttft-prefill-chunk-scheduling", args)
                    expected_policy = "srpt" if variant == "full_vendor_v8" else "fcfs"
                    self.assertEqual(
                        args[args.index("--ttft-prefill-chunk-scheduling-policy") + 1],
                        expected_policy,
                    )
                    if expected_policy == "fcfs":
                        self.assertEqual(
                            args[args.index("--ttft-prefill-chunk-quantum-tokens") + 1], "0"
                        )

    def test_matched_capture_ceiling_is_derived_not_hardcoded(self) -> None:
        from types import SimpleNamespace
        from test_green_decode_cudagraph import load_dispatcher_without_cuda

        mode, dispatcher_type = load_dispatcher_without_cuda()
        for max_seqs in (1, 3, 7, 8, 9, 10, 16, 17, 100):
            with self.subTest(max_seqs=max_seqs):
                result = _run(COMPARE, env={
                    "ISOLATION_MATCHED_COMPARISON": "1",
                    "BENCHMARK_VARIANTS": "native_40sm_chunked",
                    "VLLM_MAX_NUM_SEQS": str(max_seqs),
                })
                self.assertEqual(result.returncode, 0, result.stdout)
                args = self.server_args(result.stdout)
                sizes = list(map(int, args[args.index("--cudagraph-capture-sizes") + 1:]))
                self.assertEqual(sizes, sorted(set(sizes)))
                self.assertIn(max_seqs, sizes)
                config = SimpleNamespace(
                    compilation_config=SimpleNamespace(
                        cudagraph_mode=mode.FULL_AND_PIECEWISE,
                        cudagraph_capture_sizes=sizes,
                        max_cudagraph_capture_size=max(sizes),
                        is_attention_compiled_piecewise=lambda: True,
                    ),
                    speculative_config=None, lora_config=None,
                    scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
                    pad_for_cudagraph=lambda n: next(s for s in sizes if s >= n),
                )
                dispatcher = dispatcher_type(config)
                dispatcher.initialize_cudagraph_keys(mode.FULL_AND_PIECEWISE, 1)
                for batch in range(1, max_seqs + 1):
                    self.assertLessEqual(next(x for x in sizes if x >= batch), max_seqs)
                    selected, descriptor = dispatcher.dispatch(batch, True, False)
                    self.assertEqual(selected, mode.FULL)
                    self.assertLessEqual(descriptor.num_reqs, max_seqs)
                    self.assertLessEqual(descriptor.num_tokens, max_seqs)

    def test_matched_conflicts_fail_before_any_case_starts(self) -> None:
        for name, value in (
            ("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH", "2"),
            ("VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH", "3"),
            ("VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF", "0"),
            ("VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION", "1"),
            ("VLLM_KV_RETENTION_LAYERWISE_OFFLOAD", "1"),
            ("VLLM_PD_PREFILL_WORKSPACE_SLOTS", "2"),
            ("ENABLE_FULL_ITERATION_TRACE", "1"),
            ("BENCHMARK_CUDAGRAPH_MODE", "PIECEWISE"),
            ("VLLM_PD_EVAL_NATIVE_PREFILL_MAX_TOKENS", "2048"),
        ):
            with self.subTest(name=name):
                result = _run(COMPARE, env={
                    "ISOLATION_MATCHED_COMPARISON": "1", name: value,
                })
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("ISOLATION_MATCHED_COMPARISON requires", result.stdout)
                self.assertNotIn("[CASE] begin", result.stdout)

    def test_matched_rollback_preserves_legacy_fcfs_gate(self) -> None:
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "0",
            "BENCHMARK_VARIANTS": "dynamic_hbm_serialized_fullsm",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("prefill_admission=legacy_fcfs_single_request_gate", result.stdout)
        self.assertIn("layerwise_profile_handoff=0", result.stdout)
        self.assertIn("max_num_seqs=100 max_num_batched_tokens=16384", result.stdout)
        self.assertNotIn("--enable-ttft-prefill-chunk-scheduling", self.server_args(result.stdout))
        self.assertNotIn("--cudagraph-capture-sizes", self.server_args(result.stdout))

    def test_matched_graph_off_all_five_cases(self) -> None:
        variants = ("native_40sm_chunked", "baseline_plus",
                    "dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt", "full_vendor_v8")
        result = _run(COMPARE, env={
            "ISOLATION_GUARDED_2K_COMPARISON": "1",
            "BENCHMARK_CUDAGRAPH": "0", "BENCHMARK_VARIANTS": " ".join(variants),
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        for variant in variants:
            output = self.case_output(result.stdout, variant)
            self.assertIn("cudagraph_mode=NONE", output)
            self.assertIn("green_decode_cudagraph=0", output)
            self.assertIn("matched_graph_preserves_compile=1", output)
            self.assertNotIn("--cudagraph-capture-sizes", self.server_args(output))
            if variant not in ("native_40sm_chunked", "baseline_plus"):
                self.assertIn("layerwise_profile_handoff=1", output)
                self.assertIn("phase_future_depths=1/1/2", output)

    def test_matched_graph_off_preserves_compile_at_final_launcher(self) -> None:
        for green in ("0", "1"):
            env = os.environ.copy()
            env.update({
                "ISOLATION_MATCHED_COMPARISON": "1", "VLLM_EXECUTABLE": "/bin/echo",
                "VLLM_VENDOR_DIR": "", "VLLM_CUDAGRAPH_MODE": "NONE",
                "VLLM_PD_GREEN_DECODE_CUDAGRAPH": "0", "VLLM_EXTRA_ARGS": "",
                "VLLM_OPERATOR_SELECTIVE_PD": "0",
                "VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT": green,
                "VLLM_MUXWISE_PHASE_BASELINE_GREEN_CONTEXT": green,
            })
            result = subprocess.run(["bash", str(STARTER)], env=env, cwd=ROOT,
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout)
            args = shlex.split(result.stdout.splitlines()[-1])
            self.assertNotIn("--enforce-eager", args)
            # /bin/echo prints argv without shell quoting; shlex would strip
            # JSON's double quotes a second time.
            self.assertIn('--compilation-config {"cudagraph_mode":"NONE"}',
                          result.stdout.splitlines()[-1])

    def test_matched_graph_off_rejects_compile_confound(self) -> None:
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1", "BENCHMARK_CUDAGRAPH": "0",
            "BENCHMARK_VARIANTS": "native_40sm_chunked", "VLLM_EXTRA_ARGS": "--enforce-eager",
        })
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("separate experiment", result.stdout)

    def test_matched_graph_off_keeps_common_budget_validation(self) -> None:
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1", "BENCHMARK_CUDAGRAPH": "0",
            "BENCHMARK_VARIANTS": "native_40sm_chunked", "VLLM_MAX_NUM_SEQS": "100",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "64",
        })
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("positive max-num-seqs <= max-num-batched-tokens", result.stdout)

    def test_scheduler_timing_reaches_all_five_cases(self) -> None:
        result = _run(COMPARE, env={
            "ISOLATION_GUARDED_2K_COMPARISON": "1", "BENCHMARK_CUDAGRAPH": "0",
            "ENABLE_SCHEDULER_TIMING_TRACE": "1", "ITERATION_TRACE_SAMPLE_EVERY_N": "100",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("scheduler_timing_trace=1 sample_every=100 path="), 5)
        self.assertEqual(result.stdout.count("_artifacts/scheduler_metrics.jsonl"), 5)
        source = RUNNER.read_text()
        self.assertIn('VLLM_ISOLATION_SCHEDULER_TIMING_JSONL="${SCHEDULER_METRICS}"', source)
        self.assertLess(source.index("scheduler_timing_deployment=${SCHEDULER_TIMING_RESULT}"),
                        source.index("[LIFECYCLE] server_pid="))

    def test_scheduler_timing_bad_flag_or_sampling_rejected(self) -> None:
        for settings in ({"ENABLE_SCHEDULER_TIMING_TRACE": "yes"},
                         {"ENABLE_SCHEDULER_TIMING_TRACE": "1", "ITERATION_TRACE_SAMPLE_EVERY_N": "0"}):
            result = _run(RUNNER, env=settings)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("[ERROR]", result.stdout)

    def test_matched_instrumentation_period_can_be_changed_equally(self) -> None:
        result = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "BENCHMARK_VARIANTS": "native_40sm_chunked dynamic_hbm_serialized_fullsm full_vendor_v8",
            "ITERATION_TRACE_SAMPLE_EVERY_N": "1",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("full_iteration_trace=0 sample_every=1\n"), 3)

    def test_matched_graph_sizes_reach_real_starter_argv(self) -> None:
        plan = _run(COMPARE, env={
            "ISOLATION_MATCHED_COMPARISON": "1",
            "BENCHMARK_VARIANTS": "native_40sm_chunked",
            "VLLM_MAX_NUM_SEQS": "10",
        })
        self.assertEqual(plan.returncode, 0, plan.stdout)
        process_env = os.environ.copy()
        process_env.update({
            "VLLM_EXECUTABLE": "/bin/echo",
            "VLLM_PD_EVAL_POLICY": "manual",
            "VLLM_CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
            "VLLM_MAX_MODEL_LEN": "65536",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
            "VLLM_MAX_NUM_SEQS": "10",
            "VLLM_NUM_GPU_BLOCKS_OVERRIDE": "",
            "VLLM_EXTRA_ARGS": " ".join(self.server_args(plan.stdout)),
        })
        result = subprocess.run(
            ["bash", str(STARTER), "--model", "fake-model", "--port", "18000"],
            cwd=ROOT, env=process_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        argv = shlex.split(result.stdout.splitlines()[-1])
        self.assertEqual(argv[argv.index("--cudagraph-capture-sizes") + 1:],
                         ["1", "2", "4", "8", "10", "16"])
        self.assertIn("--compilation-config", argv)
        self.assertNotIn("--enforce-eager", argv)

    def test_matched_contract_survives_startup_or_replay_failure(self) -> None:
        source = RUNNER.read_text()
        snapshot = source.index('>"${RESULT_DIR}/experiment_snapshot/matched_comparison.txt"')
        launch = source.index("[LIFECYCLE] server_pid=")
        self.assertLess(snapshot, launch)
        self.assertEqual(source.count("\nisolation_matched_comparison="), 1)
        self.assertEqual(source.count("\ncudagraph_capture_sizes="), 1)

    @staticmethod
    def case_output(output: str, variant: str) -> str:
        start = output.index(f"[CASE] begin variant={variant} ")
        end = output.index(f"[CASE] completed variant={variant}", start)
        return output[start:end]

    @staticmethod
    def server_args(output: str) -> list[str]:
        prefix = "[PLAN] server_base_extra_args="
        lines = [
            line[len(prefix):]
            for line in output.splitlines()
            if line.startswith(prefix)
        ]
        if len(lines) != 1:
            raise AssertionError(f"Expected one actual server argument plan: {lines}")
        return shlex.split(lines[0])

    def test_hbm_srpt_enables_actual_selector_with_native_budgets(self) -> None:
        result = _run(COMPARE, env={
            "BENCHMARK_VARIANTS": "dynamic_hbm_srpt",
            "DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY": "fcfs",
            "VLLM_MAX_NUM_SEQS": "7",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "12288",
            "PREFILL_CHUNK_TOKENS": "2048",
            "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        args = self.server_args(result.stdout)
        self.assertIn("--enable-ttft-prefill-chunk-scheduling", args)
        self.assertEqual(args[args.index("--ttft-prefill-chunk-scheduling-policy") + 1], "srpt")
        self.assertEqual(args[args.index("--ttft-prefill-chunk-quantum-tokens") + 1], "0")
        self.assertEqual(args[args.index("--tbt-decode-scheduling-policy") + 1], "fcfs")
        self.assertNotIn("--ttft-prefill-native-chunk-max-tokens", args)
        self.assertIn("max_num_seqs=7 max_num_batched_tokens=12288", result.stdout)
        self.assertIn("pd_eval_policy=muxwise_dynamic hbm_only=1", result.stdout)
        self.assertIn("prefill_selection_quantum_tokens=0", result.stdout)
        self.assertIn("[PLAN] serialized_full_sm=1", result.stdout)

        # Exercise the final CLI forwarding wrapper without loading vLLM/GPU.
        process_env = os.environ.copy()
        process_env.update({
            "VLLM_EXECUTABLE": "/bin/echo",
            "VLLM_VENDOR_DIR": "",
            "VLLM_PD_EVAL_POLICY": "muxwise_dynamic",
            "VLLM_EXTRA_ARGS": shlex.join(args),
            "VLLM_MAX_MODEL_LEN": "65536",
            "VLLM_MAX_NUM_SEQS": "7",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "12288",
            "VLLM_NUM_GPU_BLOCKS_OVERRIDE": "",
        })
        started = subprocess.run(
            ["bash", str(STARTER), "--model", "fake-model"],
            cwd=ROOT, env=process_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=30,
        )
        self.assertEqual(started.returncode, 0, started.stdout)
        forwarded = shlex.split(started.stdout.splitlines()[-1])
        self.assertEqual(forwarded[-len(args):], args)
        self.assertEqual(forwarded[forwarded.index("--max-num-seqs") + 1], "7")
        self.assertEqual(forwarded[forwarded.index("--max-num-batched-tokens") + 1], "12288")
        self.assertIn("pd_eval_context_mode=none", started.stdout)
        # Check real exported flags, not nonexistent launcher log labels.
        process_env.update({"VLLM_VIRTUAL_CONTEXT": "1",
                            "VLLM_VIRTUAL_CONTEXT_RING_BACKEND": "1"})
        normalized = subprocess.run(
            ["bash", "-c", 'source "$1"; pd_eval_policy_apply; '
             'printf "%s %s\\n" "$VLLM_VIRTUAL_CONTEXT" '
             '"$VLLM_VIRTUAL_CONTEXT_RING_BACKEND"',
             "policy-test", str(ROOT / "tools/pd_eval_policy_env.sh")],
            cwd=ROOT, env=process_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=30,
        )
        self.assertEqual(normalized.returncode, 0, normalized.stdout)
        self.assertEqual(normalized.stdout.strip(), "0 0")

    def test_legacy_dynamic_explicit_srpt_is_not_just_a_plan_label(self) -> None:
        for variant in ("dynamic_pd_hbm", "dynamic_hbm_serialized_fixed_split",
                        "dynamic_hbm_serialized_fullsm"):
            with self.subTest(variant=variant):
                result = _run(COMPARE, env={
                    "BENCHMARK_VARIANTS": variant,
                    "DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY": "srpt",
                    "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
                })
                self.assertEqual(result.returncode, 0, result.stdout)
                args = self.server_args(result.stdout)
                self.assertIn("--enable-ttft-prefill-chunk-scheduling", args)
                self.assertEqual(args[args.index("--ttft-prefill-chunk-quantum-tokens") + 1], "0")
                self.assertIn("pd_eval_policy=muxwise_dynamic hbm_only=1", result.stdout)

    def test_direct_legacy_dynamic_default_stays_fcfs(self) -> None:
        for variant in ("dynamic_pd_hbm", "dynamic_hbm_serialized_fixed_split",
                        "dynamic_hbm_serialized_fullsm"):
            with self.subTest(variant=variant):
                result = _run(RUNNER, env={"SYSTEM_VARIANT": variant})
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("prefill_scheduling_policy=fcfs", result.stdout)
                self.assertNotIn("--enable-ttft-prefill-chunk-scheduling",
                                 self.server_args(result.stdout))

    def test_five_case_sweep_isolates_offload_and_request_preemption(self) -> None:
        variants = ("native_40sm_chunked", "baseline_plus",
                    "dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt",
                    "full_vendor_v8")
        result = _run(COMPARE, env={
            "BENCHMARK_VARIANTS": " ".join(variants),
            "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
            "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION": "1",
            "VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN": "1",
            "VLLM_KV_RETENTION_LAYERWISE_OFFLOAD": "1",
            "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2",
            "VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH": "1",
            "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "3",
            "VLLM_PD_PREFILL_WORKSPACE_ENABLED": "1",
            "VLLM_PD_PREFILL_WORKSPACE_SLOTS": "2",
            "VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE": "0",
            "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
        })
        self.assertEqual(result.returncode, 0, result.stdout)
        for variant in variants:
            part = self.case_output(result.stdout, variant)
            enabled = int(variant == "full_vendor_v8")
            self.assertIn(f"kv_layerwise_checkpoint_plan={enabled} "
                          f"kv_layerwise_offload={enabled}", part)
            self.assertIn(f"ttft_layerwise_prefill_preemption={enabled}", part)
            self.assertIn(f"experiment=vendor_v8_dynamic_hbm_vs_full_smoke_v1_{variant} ", part)
        self.assertIn("pd_eval_policy=manual hbm_only=0",
                      self.case_output(result.stdout, "native_40sm_chunked"))
        baseline = self.case_output(result.stdout, "baseline_plus")
        self.assertIn("pd_eval_policy=mixed_iteration hbm_only=0", baseline)
        self.assertIn("--enable-ttft-prefill-chunk-scheduling", self.server_args(baseline))
        full = self.case_output(result.stdout, "full_vendor_v8")
        self.assertIn("pd_eval_policy=muxwise_dynamic_virtual_ring_optimized", full)

    def test_hbm_srpt_rejects_wrong_policy_and_missing_solo_profiles(self) -> None:
        for extra, message in (
            ({"PREFILL_SCHEDULING_POLICY": "fcfs"}, "requires PREFILL_SCHEDULING_POLICY=srpt"),
            ({"PD_SM_PROFILES": "30/10"}, "requires both 0/40 and 40/0 profiles"),
            ({"VLLM_PD_SERIALIZED_FULL_SM": "0"}, "requires VLLM_PD_SERIALIZED_FULL_SM=1"),
        ):
            with self.subTest(extra=extra):
                result = _run(RUNNER, env={"SYSTEM_VARIANT": "dynamic_hbm_srpt", **extra})
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(message, result.stdout)

    def test_hbm_only_rejects_uncalibrated_slo_srpt(self) -> None:
        result = _run(RUNNER, env={
            "SYSTEM_VARIANT": "dynamic_hbm_serialized_fullsm",
            "PREFILL_SCHEDULING_POLICY": "slo_srpt",
        })
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("HBM-only comparisons support fcfs or srpt", result.stdout)

    def test_controls_reject_direct_checkpoint_offload_or_preemption(self) -> None:
        for variant in ("native_40sm_chunked", "baseline_plus", "dynamic_hbm_srpt",
                        "dynamic_hbm_serialized_fullsm"):
            for flag in ("VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN",
                         "VLLM_KV_RETENTION_LAYERWISE_OFFLOAD",
                         "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION"):
                with self.subTest(variant=variant, flag=flag):
                    result = _run(RUNNER, env={
                        "SYSTEM_VARIANT": variant, flag: "1",
                        "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
                    })
                    self.assertEqual(result.returncode, 2, result.stdout)
                    self.assertIn("controls must keep them disabled", result.stdout)

    def test_full_vendor_offload_checks_prerequisites_before_startup(self) -> None:
        for env, message in (
            (
                {"VLLM_KV_RETENTION_LAYERWISE_OFFLOAD": "1"},
                "requires VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN=1",
            ),
            (
                {"VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN": "1"},
                "requires Stage-2 Prefill preemption",
            ),
            ({"VLLM_KV_RETENTION_LAYERWISE_OFFLOAD": "yes"}, "flags must be 0 or 1"),
        ):
            with self.subTest(env=env):
                result = _run(RUNNER, env={"SYSTEM_VARIANT": "full_vendor_v8", **env})
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(message, result.stdout)

    def test_scheduling_plan_is_resolved_before_dry_run_and_not_rebuilt(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        marker = 'echo "[PLAN] server_base_extra_args=${VLLM_EXTRA_ARGS}"'
        before, after = source.split(marker, 1)
        self.assertIn('--ttft-prefill-chunk-quantum-tokens 0', before)
        self.assertNotIn('--enable-ttft-prefill-chunk-scheduling', after)
        self.assertIn('VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}"', after)
        settings_quantum = [
            line for line in after.splitlines()
            if line.startswith("prefill_selection_quantum_tokens=")
        ]
        self.assertEqual(settings_quantum, [
            "prefill_selection_quantum_tokens=${PREFILL_SELECTOR_QUANTUM_TOKENS}"
        ])
        self.assertIn(
            '--ttft-prefill-estimated-ms-per-token '
            '${PREFILL_FALLBACK_MS_PER_TOKEN}', after,
        )
        for flag, value in (
            ("VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN", "KV_LAYERWISE_CHECKPOINT_PLAN"),
            ("VLLM_KV_RETENTION_LAYERWISE_OFFLOAD", "KV_LAYERWISE_OFFLOAD"),
        ):
            self.assertIn(f'{flag}="${{{value}}}"', after)

    def test_native_budget_fill_prefill_settings_reach_all_cases(
        self,
    ) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "multi_prefill_batch_contract",
                "BENCHMARK_VARIANTS": (
                    "native_40sm_chunked baseline_plus "
                    "dynamic_hbm_serialized_fullsm full_vendor_v8"
                ),
                "VLLM_MAX_NUM_SEQS": "10",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
                "VLLM_LONG_PREFILL_TOKEN_THRESHOLD": "2048",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        budget_plans = [line for line in result.stdout.splitlines()
                        if line.startswith("[PLAN] prefill_admission=")]
        self.assertEqual(len(budget_plans), 4, result.stdout)
        for plan in budget_plans:
            self.assertIn("max_num_seqs=10 max_num_batched_tokens=16384 long_threshold=2048", plan)
        # The rollback path retains its real FCFS gate; do not label that
        # historical control as multi-request merely because its budgets match.
        self.assertEqual(sum("legacy_fcfs_single_request_gate" in p for p in budget_plans), 1)
        starter_text = STARTER.read_text(encoding="utf-8")
        self.assertNotIn("--max-num-partial-prefills", starter_text)
        self.assertNotIn("--max-long-partial-prefills", starter_text)
        self.assertIn("--long-prefill-token-threshold", starter_text)

    def test_unsupported_v1_partial_prefill_width_fails_fast(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "unsupported_partial_width",
                "BENCHMARK_VARIANTS": "baseline_plus",
                "VLLM_MAX_NUM_PARTIAL_PREFILLS": "8",
                "VLLM_MAX_LONG_PARTIAL_PREFILLS": "8",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "unsupported by vLLM V1; omit it",
            result.stdout,
        )

    def test_throughput_bundle_separates_native_and_ring_prefill_sizes(
        self,
    ) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "native_ring_chunk_split_contract",
                "BENCHMARK_VARIANTS": (
                    "full_vendor_v8 dynamic_hbm_serialized_fullsm"
                ),
                "FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS": "1",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
                "PREFILL_CHUNK_TOKENS": "2048",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)

        full_start = result.stdout.index(
            "[CASE] begin variant=full_vendor_v8 "
        )
        full_end = result.stdout.index(
            "[CASE] completed variant=full_vendor_v8",
            full_start,
        )
        full_output = result.stdout[full_start:full_end]
        self.assertIn(
            "prefill_selection_quantum_tokens=2048 "
            "native_prefill_max_tokens=16384",
            full_output,
        )
        self.assertIn(
            "active_decode_native_prefill_tokens=16384 "
            "active_decode_ring_prefill_tokens=2048",
            full_output,
        )

        control_start = result.stdout.index(
            "[CASE] begin variant=dynamic_hbm_serialized_fullsm "
        )
        control_end = result.stdout.index(
            "[CASE] completed variant=dynamic_hbm_serialized_fullsm",
            control_start,
        )
        control_output = result.stdout[control_start:control_end]
        self.assertIn(
            "prefill_selection_quantum_tokens=0 "
            "native_prefill_max_tokens=0",
            control_output,
        )

    def test_default_matrix_selects_four_distinct_cases(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "four_case_cpu_contract",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)

        expected = {
            "native_40sm_chunked": ("1", "upstream", "0"),
            "full_vendor_v8": (
                "1",
                "greencontext_partitioned_decode",
                "1",
            ),
            "dynamic_hbm_serialized_fixed_split": (
                "0",
                "greencontext_partitioned_decode",
                "1",
            ),
            "dynamic_hbm_serialized_fullsm": (
                "1",
                "greencontext_partitioned_decode",
                "1",
            ),
        }
        for variant, (flag, graph_path, green_graph) in expected.items():
            self.assertIn(
                f"[CASE] begin variant={variant} ",
                result.stdout,
            )
            self.assertIn(
                f"system_variant={variant}\n",
                result.stdout,
            )
            case_start = result.stdout.index(
                f"[CASE] begin variant={variant} "
            )
            case_end = result.stdout.index(
                f"[CASE] completed variant={variant}",
                case_start,
            )
            case_output = result.stdout[case_start:case_end]
            self.assertIn(
                f"[PLAN] serialized_full_sm={flag}",
                case_output,
            )
            self.assertIn(
                "[PLAN] cudagraph_mode=FULL_AND_PIECEWISE",
                case_output,
            )
            self.assertIn(
                f"[PLAN] cudagraph_path={graph_path}",
                case_output,
            )
            self.assertIn(
                f"[PLAN] green_decode_cudagraph={green_graph}",
                case_output,
            )
            self.assertIn(
                "[PLAN] layerwise_profile_handoff=0",
                case_output,
            )

    def test_layerwise_opt_in_reaches_all_vendor_cases_only(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "four_case_layerwise_cpu_contract",
                "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF": "1",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)

        for variant, expected in (
            ("native_40sm_chunked", "0"),
            ("full_vendor_v8", "1"),
            ("dynamic_hbm_serialized_fixed_split", "1"),
            ("dynamic_hbm_serialized_fullsm", "1"),
        ):
            case_start = result.stdout.index(
                f"[CASE] begin variant={variant} "
            )
            case_end = result.stdout.index(
                f"[CASE] completed variant={variant}",
                case_start,
            )
            case_output = result.stdout[case_start:case_end]
            self.assertIn(
                f"layerwise_profile_handoff={expected}",
                case_output,
            )
            self.assertIn(
                f"[PLAN] layerwise_profile_handoff={expected}",
                case_output,
            )

    def test_layerwise_rejects_incompatible_phase_depth(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "dynamic_hbm_serialized_fixed_split",
                "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF": "1",
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2",
                "VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH": "1",
                "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "3",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "requires phase depth enabled with P=1/D=1/G>=2",
            result.stdout,
        )

    def test_ttft_layer_group_opt_in_reaches_vendor_cases_only(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "ttft_layer_groups_contract",
                "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
                "VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS": "4",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)

        for variant, expected in (
            ("native_40sm_chunked", "0"),
            ("full_vendor_v8", "1"),
            ("dynamic_hbm_serialized_fixed_split", "1"),
            ("dynamic_hbm_serialized_fullsm", "1"),
        ):
            case_start = result.stdout.index(
                f"[CASE] begin variant={variant} "
            )
            case_end = result.stdout.index(
                f"[CASE] completed variant={variant}",
                case_start,
            )
            case_output = result.stdout[case_start:case_end]
            self.assertIn(
                f"ttft_layerwise_prefill_execution={expected}",
                case_output,
            )
            self.assertIn(
                "preemption=0 group_layers=4 scheduler_yield=0",
                case_output,
            )

    def test_ttft_prefill_preemption_reaches_srpt_vendor_case_only(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "ttft_prefill_preemption_contract",
                "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
                "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION": "1",
                "VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS": "2",
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2",
                "VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH": "1",
                "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "3",
                "VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE": "0",
                "FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS": "0",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)

        for variant, expected in (
            ("native_40sm_chunked", "0"),
            ("full_vendor_v8", "1"),
            ("dynamic_hbm_serialized_fixed_split", "0"),
            ("dynamic_hbm_serialized_fullsm", "0"),
        ):
            case_start = result.stdout.index(
                f"[CASE] begin variant={variant} "
            )
            case_end = result.stdout.index(
                f"[CASE] completed variant={variant}",
                case_start,
            )
            case_output = result.stdout[case_start:case_end]
            self.assertIn(
                f"ttft_layerwise_prefill_preemption={expected}",
                case_output,
            )
            self.assertIn(
                f"preemption={expected} group_layers=2 "
                f"scheduler_yield={expected}",
                case_output,
            )

    def test_ttft_prefill_preemption_requires_base_execution(self) -> None:
        result = _run(
            COMPARE,
            env={
                "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION": "1",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "TTFT layerwise Prefill preemption requires its execution flag",
            result.stdout,
        )

    def test_ttft_prefill_preemption_requires_two_prefill_slots(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "full_vendor_v8",
                "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
                "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION": "1",
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "1",
                "VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH": "1",
                "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "2",
                "VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE": "0",
                "FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS": "0",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "requires phase depth P>=2/D=1/G>=P+D",
            result.stdout,
        )

    def test_ttft_prefill_preemption_rejects_fcfs_direct_use(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "full_vendor_v8",
                "PREFILL_SCHEDULING_POLICY": "fcfs",
                "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
                "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION": "1",
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2",
                "VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH": "1",
                "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "3",
                "VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE": "0",
                "FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS": "0",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "currently requires PREFILL_SCHEDULING_POLICY=srpt",
            result.stdout,
        )

    def test_ttft_layer_group_width_must_be_positive(self) -> None:
        result = _run(
            COMPARE,
            env={
                "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
                "VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS": "0",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS must be positive",
            result.stdout,
        )

    def test_layerwise_requires_boolean_value(self) -> None:
        result = _run(
            COMPARE,
            env={
                "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF": "enabled",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF must be 0 or 1",
            result.stdout,
        )

    def test_native_rejects_direct_layerwise_opt_in(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "native_40sm_chunked",
                "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF": "1",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "cannot enable vendor layerwise profile handoff",
            result.stdout,
        )

    def test_prefill_workspace_opt_in_reaches_vendor_cases_only(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "fixed_prefill_workspace_contract",
                "VLLM_PD_PREFILL_WORKSPACE_ENABLED": "1",
                "VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS": "16384",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)

        for variant, expected in (
            ("native_40sm_chunked", "0"),
            ("full_vendor_v8", "1"),
            ("dynamic_hbm_serialized_fixed_split", "1"),
            ("dynamic_hbm_serialized_fullsm", "1"),
        ):
            case_start = result.stdout.index(
                f"[CASE] begin variant={variant} "
            )
            case_end = result.stdout.index(
                f"[CASE] completed variant={variant}",
                case_start,
            )
            case_output = result.stdout[case_start:case_end]
            self.assertIn(
                f"[PLAN] prefill_workspace={expected} ",
                case_output,
            )

    def test_prefill_workspace_rejects_unsafe_depth_and_capacity(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "dynamic_hbm_serialized_fullsm",
                "VLLM_PD_PREFILL_WORKSPACE_ENABLED": "1",
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "fixed Prefill workspace requires phase depth enabled and "
            "slots >= Prefill depth",
            result.stdout,
        )

        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "dynamic_hbm_serialized_fullsm",
                "VLLM_PD_PREFILL_WORKSPACE_ENABLED": "1",
                "VLLM_PD_PREFILL_WORKSPACE_SLOTS": "2",
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2",
                "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "3",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "[PLAN] prefill_workspace=1 max_tokens=16384 slots=2",
            result.stdout,
        )

        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "dynamic_hbm_serialized_fullsm",
                "VLLM_PD_PREFILL_WORKSPACE_ENABLED": "1",
                "VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS": "1024",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "must cover --max-num-batched-tokens",
            result.stdout,
        )

    def test_prefill_workspace_requires_boolean_value(self) -> None:
        result = _run(
            COMPARE,
            env={"VLLM_PD_PREFILL_WORKSPACE_ENABLED": "enabled"},
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "VLLM_PD_PREFILL_WORKSPACE_ENABLED must be 0 or 1",
            result.stdout,
        )

    def test_prefill_workspace_requires_positive_slot_count(self) -> None:
        result = _run(
            COMPARE,
            env={"VLLM_PD_PREFILL_WORKSPACE_SLOTS": "0"},
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "VLLM_PD_PREFILL_WORKSPACE_SLOTS must be positive",
            result.stdout,
        )

    def test_ttft_prefill_preemption_accepts_two_slot_workspace(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "full_vendor_v8",
                "PREFILL_SCHEDULING_POLICY": "srpt",
                "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": "1",
                "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION": "1",
                "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2",
                "VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH": "1",
                "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "3",
                "VLLM_PD_PREFILL_WORKSPACE_ENABLED": "1",
                "VLLM_PD_PREFILL_WORKSPACE_SLOTS": "2",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "[PLAN] prefill_workspace=1 max_tokens=16384 slots=2",
            result.stdout,
        )

    def test_cudagraph_matrix_has_explicit_rollback(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "four_case_eager_cpu_contract",
                "BENCHMARK_CUDAGRAPH": "0",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[COMPARE] cudagraph=0 mode=NONE", result.stdout)
        self.assertEqual(
            result.stdout.count("[PLAN] cudagraph_mode=NONE"),
            4,
        )
        self.assertEqual(
            result.stdout.count("[PLAN] green_decode_cudagraph=0"),
            4,
        )

    def test_cudagraph_matrix_rejects_piecewise_only_mode(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_CUDAGRAPH_MODE": "PIECEWISE",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("must provide full Decode graphs", result.stdout)

    def test_start_wrapper_passes_cudagraph_json_as_one_argument(self) -> None:
        process_env = os.environ.copy()
        process_env.update(
            {
                "VLLM_EXECUTABLE": "/bin/echo",
                "VLLM_PD_EVAL_POLICY": "manual",
                "VLLM_CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
                "VLLM_MAX_MODEL_LEN": "1024",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "1024",
                "VLLM_MAX_NUM_SEQS": "1",
                "VLLM_NUM_GPU_BLOCKS_OVERRIDE": "",
                "VLLM_EXTRA_ARGS": "",
            }
        )
        result = subprocess.run(
            [
                "bash",
                str(STARTER),
                "--model",
                "fake-model",
                "--port",
                "18000",
            ],
            cwd=ROOT,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            '--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}',
            result.stdout,
        )
        self.assertNotIn("--enforce-eager", result.stdout.splitlines()[-1])

    def test_legacy_dynamic_name_maps_to_full_sm_behavior(self) -> None:
        result = _run(
            COMPARE,
            env={
                "BENCHMARK_MODE": "smoke",
                "BASE_EXPERIMENT_NAME": "legacy_dynamic_cpu_contract",
                "BENCHMARK_VARIANTS": "dynamic_pd_hbm",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("variant=dynamic_pd_hbm", result.stdout)
        self.assertIn("[PLAN] serialized_full_sm=1", result.stdout)

    def test_named_variant_rejects_contradictory_gate(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "dynamic_hbm_serialized_fixed_split",
                "VLLM_PD_SERIALIZED_FULL_SM": "1",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "requires VLLM_PD_SERIALIZED_FULL_SM=0",
            result.stdout,
        )

    def test_full_sm_case_requires_both_solo_profiles(self) -> None:
        result = _run(
            RUNNER,
            env={
                "SYSTEM_VARIANT": "dynamic_hbm_serialized_fullsm",
                "PD_SM_PROFILES": "20/20",
            },
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "requires both 0/40 and 40/0 profiles",
            result.stdout,
        )

    def test_sweep_owns_a_fail_closed_instance_lifecycle(self) -> None:
        compare_text = COMPARE.read_text(encoding="utf-8")
        runner_text = RUNNER.read_text(encoding="utf-8")

        self.assertIn('ACTIVE_SERVER_PGID="${ACTIVE_SERVER_PID}"', runner_text)
        self.assertIn(
            '"${ACTIVE_SERVER_PID}" "${ACTIVE_SERVER_PGID}"',
            runner_text,
        )
        self.assertIn("process_group_has_live_members", runner_text)
        self.assertIn("wait_for_port_release", runner_text)
        self.assertIn("server process group mismatch", runner_text)
        self.assertIn("refusing unsafe restart", runner_text)
        self.assertIn(
            'case_status == INSTANCE_CLEANUP_FAILURE_EXIT_CODE',
            compare_text,
        )
        self.assertIn("wait_for_sweep_port_release", compare_text)
        self.assertIn("CASE_RESTART_COOLDOWN_SEC", compare_text)

        result = _run(
            COMPARE,
            env={
                "BENCHMARK_VARIANTS": (
                    "native_40sm_chunked full_vendor_v8"
                ),
                "CASE_RESTART_COOLDOWN_SEC": "0",
                "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "instance_lifecycle=separate_process_group_per_case",
            result.stdout,
        )
        self.assertEqual(
            result.stdout.count(
                "instance_lifecycle=setsid_group_term_kill_then_port_release"
            ),
            2,
        )

    def test_sweep_rejects_invalid_lifecycle_timeout(self) -> None:
        result = _run(
            COMPARE,
            env={"SERVER_SHUTDOWN_TIMEOUT_SEC": "0"},
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "SERVER_SHUTDOWN_TIMEOUT_SEC must be a positive integer",
            result.stdout,
        )

        result = _run(
            RUNNER,
            env={"KILL_EXISTING_PORT_OWNER": "yes"},
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "KILL_EXISTING_PORT_OWNER must be 0 or 1",
            result.stdout,
        )

    def test_saved_pgid_cleans_worker_after_leader_exits(self) -> None:
        if not all(shutil.which(name) for name in ("setsid", "lsof", "ps", "awk")):
            self.skipTest("process-lifecycle commands are unavailable")
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            self.skipTest("local socket creation is blocked")
        port = int(probe.getsockname()[1])
        probe.close()

        worker_code = """
import os
import socket
import sys
import time

port = int(sys.argv[1])
if os.fork() != 0:
    os._exit(0)
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen()
while True:
    time.sleep(1)
"""
        leader = subprocess.Popen(
            ["setsid", sys.executable, "-c", worker_code, str(port)],
            cwd=ROOT,
        )
        saved_pgid = leader.pid
        try:
            leader.wait(timeout=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                self.fail("orphan worker did not acquire the test port")

            runner_text = RUNNER.read_text(encoding="utf-8")
            lifecycle_start = runner_text.index("cleanup_pid() {")
            lifecycle_end = runner_text.index("cleanup_on_exit() {")
            lifecycle_functions = runner_text[lifecycle_start:lifecycle_end]
            harness = f"""
set -euo pipefail
PORT={port}
SERVER_SHUTDOWN_TIMEOUT_SEC=2
PORT_RELEASE_TIMEOUT_SEC=5
{lifecycle_functions}
ACTIVE_SERVER_PID={leader.pid}
ACTIVE_SERVER_PGID={saved_pgid}
ACTIVE_DMON_PID=""
cleanup_active
"""
            cleaned = subprocess.run(
                ["bash", "-c", harness],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=10,
            )
            self.assertEqual(cleaned.returncode, 0, cleaned.stdout)
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=0.2)
        finally:
            try:
                os.killpg(saved_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_idempotent_exit_cleanup_preserves_workload_failure_status(
        self,
    ) -> None:
        runner_text = RUNNER.read_text(encoding="utf-8")
        lifecycle_start = runner_text.index("cleanup_pid() {")
        lifecycle_end = runner_text.index("for command_name in setsid")
        lifecycle_functions = runner_text[lifecycle_start:lifecycle_end]
        harness = f"""
set -u
PORT=65530
SERVER_SHUTDOWN_TIMEOUT_SEC=1
PORT_RELEASE_TIMEOUT_SEC=1
INSTANCE_CLEANUP_FAILURE_EXIT_CODE=86
{lifecycle_functions}
ACTIVE_SERVER_PID=""
ACTIVE_SERVER_PGID=""
ACTIVE_DMON_PID=""
exit 1
"""
        completed = subprocess.run(
            ["bash", "-c", harness],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 1, completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
