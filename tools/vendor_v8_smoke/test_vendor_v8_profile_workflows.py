#!/usr/bin/env python3
"""CPU-only tests for vendor_v8 profile and short-validation workflows."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from analyze_vendor_v8_partition_profiles import aggregate_profiles
from generate_vendor_v8_slo_profile import (
    DEFAULT_CONTEXT_TOKENS,
    build_regression_slo,
    load_baseline_profiles,
)
from measure_vendor_v8_partition_profile import (
    StreamResult,
    build_profile_state_plan,
    collect_anchor_decode_only_control_gaps,
    collect_all_anchor_overlap_gaps,
    lookup_request_slo,
)
from analyze_vendor_v8_tbt_sm_scaling import build_summary
from measure_vendor_v8_quick_tbt_sm_scaling import (
    ProbeWindow,
    classify_probe_window,
)
from measure_vendor_v8_short_validation import build_validation_plan


class VendorV8SLORegressionTest(unittest.TestCase):
    @staticmethod
    def _write_exact_baseline(
        root: Path,
        *,
        context_tokens: int,
        model: str = "test-model",
    ) -> Path:
        results_dir = root / f"ctx{context_tokens}_results"
        results_dir.mkdir()
        raw_rows = []
        tbt_base_ms = 10.0 + context_tokens / 10_000.0
        for repeat, scale in enumerate((1.0, 1.1, 1.2), start=1):
            raw_rows.append(
                {
                    "context_target_tokens": context_tokens,
                    "phase": "measurement",
                    "status": "success",
                    "ttft_ms": context_tokens * scale,
                    "mean_tbt_ms": tbt_base_ms * scale,
                    "p99_tbt_ms": (tbt_base_ms + 2.0) * scale,
                    "completion_latency_ms": context_tokens * scale + 100.0,
                    "output_tokens": 4096,
                    "repeat_index": repeat,
                }
            )
        with (results_dir / "request_runs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(raw_rows[0]))
            writer.writeheader()
            writer.writerows(raw_rows)
        with (results_dir / "bucket_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["context_target_tokens", "pooled_p99_tbt_ms"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "context_target_tokens": context_tokens,
                    "pooled_p99_tbt_ms": tbt_base_ms + 3.0,
                }
            )
        metadata = {
            "status": "completed",
            "case_id": f"ctx{context_tokens}",
            "model": model,
            "repeat_count": 3,
            "warmup_count_per_bucket": 1,
            "cooldown_sec": 10.0,
            "request_max_tokens": 4096,
            "request_min_tokens": 4096,
            "ignore_eos": True,
            "submission_mode": "strict_synchronous_closed_loop",
            "completion_boundary": "stream_tokens_then_openai_done_marker",
            "tbt_token_count_source": "choice.token_ids",
            "tbt_requires_return_token_ids": True,
        }
        (results_dir / "run_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return results_dir

    def test_default_calibration_temporarily_stops_at_100k(self) -> None:
        self.assertEqual(
            DEFAULT_CONTEXT_TOKENS,
            (25_000, 50_000, 75_000, 100_000),
        )

    def test_five_bucket_regression_builds_partition_invariant_slo(self) -> None:
        contexts = [25_000, 50_000, 75_000, 100_000, 125_000]
        profile = [
            {
                "context_tokens": context,
                "ttft_p95_ms": 10.0 + 0.0001 * context**2,
                "request_p99_tbt_p95_ms": 5.0 + 0.001 * context,
            }
            for context in contexts
        ]
        policy, rows = build_regression_slo(
            profile,
            required_contexts=contexts,
        )
        self.assertEqual(policy["required_context_tokens"], contexts)
        self.assertTrue(policy["slo_contract"]["partition_invariant"])
        self.assertAlmostEqual(
            policy["regression"]["ttft"]["r_squared"], 1.0
        )
        self.assertAlmostEqual(
            policy["regression"]["tbt"]["r_squared"], 1.0
        )
        self.assertAlmostEqual(
            rows[0]["ttft_slo_ms"],
            profile[0]["ttft_p95_ms"] * 2.0,
            places=3,
        )
        self.assertAlmostEqual(
            rows[0]["tbt_slo_ms"],
            profile[0]["request_p99_tbt_p95_ms"] * 1.25,
            places=3,
        )
        self.assertEqual(
            policy["slo_multipliers"],
            {
                "method": "splitwise_p50_slowdown",
                "ttft": 2.0,
                "tbt": 1.25,
            },
        )
        # 10K is conservatively covered by the first measured 25K bucket.
        short_slo = lookup_request_slo(policy, 10_000)
        self.assertEqual(
            short_slo["ttft_prefill_deadline_ms"],
            policy["buckets"][0]["ttft_prefill_deadline_ms"],
        )
        with self.assertRaises(ValueError):
            lookup_request_slo(policy, 130_000)

    def test_single_bucket_builds_local_constant_slo_without_regression(
        self,
    ) -> None:
        policy, rows = build_regression_slo(
            [
                {
                    "context_tokens": 100_000,
                    "ttft_p95_ms": 800_000.0,
                    "request_p99_tbt_p95_ms": 52.0,
                }
            ],
            required_contexts=[100_000],
        )
        self.assertEqual(policy["calibration_mode"], "single_point_constant")
        self.assertEqual(
            policy["regression"]["ttft"]["fit_kind"],
            "single_point_constant",
        )
        self.assertIsNone(policy["regression"]["ttft"]["r_squared"])
        self.assertEqual(rows[0]["ttft_regression_ms"], 800_000.0)
        self.assertEqual(rows[0]["ttft_slo_ms"], 1_600_000.0)
        self.assertEqual(rows[0]["tbt_regression_ms"], 52.0)
        self.assertEqual(rows[0]["tbt_slo_ms"], 65.0)

    def test_separate_exact_context_results_are_combined(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx100k = self._write_exact_baseline(
                root, context_tokens=100_000
            )
            ctx10k = self._write_exact_baseline(root, context_tokens=10_000)
            profile, sources = load_baseline_profiles([ctx100k, ctx10k])

        self.assertEqual(
            [row["context_tokens"] for row in profile], [10_000, 100_000]
        )
        self.assertEqual(
            [source.contexts for source in sources], [(100_000,), (10_000,)]
        )
        policy, _ = build_regression_slo(
            profile,
            required_contexts=[10_000, 100_000],
        )
        self.assertEqual(
            [row["max_context_tokens"] for row in policy["buckets"]],
            [10_000, 100_000],
        )

    def test_combined_results_reject_duplicate_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = self._write_exact_baseline(root, context_tokens=10_000)
            second = root / "duplicate_results"
            first.rename(second)
            replacement = self._write_exact_baseline(
                root, context_tokens=10_000
            )
            with self.assertRaisesRegex(ValueError, "duplicate.*10000"):
                load_baseline_profiles([replacement, second])

    def test_combined_results_reject_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ctx10k = self._write_exact_baseline(root, context_tokens=10_000)
            ctx100k = self._write_exact_baseline(
                root, context_tokens=100_000, model="different-model"
            )
            with self.assertRaisesRegex(ValueError, "not comparable.*model"):
                load_baseline_profiles([ctx10k, ctx100k])

    def test_runtime_slo_rejects_legacy_bos_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = self._write_exact_baseline(
                root, context_tokens=100_000
            )
            metadata_path = legacy / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("tbt_token_count_source")
            metadata.pop("tbt_requires_return_token_ids")
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "direct choice.token_ids"
            ):
                load_baseline_profiles([legacy])


class VendorV8ValidationPlanTest(unittest.TestCase):
    def test_three_completion_gated_long_cycles_have_six_shorts(self) -> None:
        plan = build_validation_plan(
            long_count=3,
            short_per_long=2,
            long_context_tokens=100_000,
            short_context_tokens=10_000,
            short_offsets_sec=[5.0, 10.0],
        )
        self.assertEqual(len(plan), 9)
        for cycle in range(3):
            selected = [item for item in plan if item.cycle_index == cycle]
            self.assertEqual(
                [(item.role, item.context_tokens) for item in selected],
                [
                    ("long", 100_000),
                    ("short", 10_000),
                    ("short", 10_000),
                ],
            )
            self.assertEqual(
                [item.offset_from_long_submit_s for item in selected],
                [0.0, 5.0, 10.0],
            )


class VendorV8AnchorWindowTest(unittest.TestCase):
    def test_paired_control_selects_only_immediate_pre_probe_tail(self) -> None:
        anchor = StreamResult(
            row={"request_id": "anchor-0"},
            token_rows=[
                {
                    "gap_start_monotonic_ts": float(index),
                    "arrival_monotonic_ts": float(index) + 0.5,
                    "estimated_per_token_gap_ms": float(index * 10),
                    "chunk_tokens": 1,
                    "before_probe_control": 0,
                }
                for index in range(1, 5)
            ],
        )
        gaps = collect_anchor_decode_only_control_gaps(
            [anchor], probe_submit=4.0, gaps_per_anchor=2
        )
        self.assertEqual(gaps, [20.0, 30.0])
        self.assertEqual(
            [row["before_probe_control"] for row in anchor.token_rows],
            [0, 1, 1, 0],
        )
        with self.assertRaises(RuntimeError):
            collect_anchor_decode_only_control_gaps(
                [anchor], probe_submit=4.0, gaps_per_anchor=4
            )

    def test_tbt_window_stops_when_first_declared_anchor_exits(self) -> None:
        anchors = [
            StreamResult(
                row={"completion_monotonic_ts": 15.0},
                token_rows=[
                    {
                        "gap_start_monotonic_ts": 11.0,
                        "arrival_monotonic_ts": 14.0,
                        "estimated_per_token_gap_ms": 3.0,
                        "chunk_tokens": 1,
                    },
                    {
                        "gap_start_monotonic_ts": 14.99,
                        "arrival_monotonic_ts": 15.005,
                        "estimated_per_token_gap_ms": 2.0,
                        "chunk_tokens": 1,
                    },
                ],
            ),
            StreamResult(
                row={"completion_monotonic_ts": 20.0},
                token_rows=[
                    {
                        "gap_start_monotonic_ts": 12.0,
                        "arrival_monotonic_ts": 14.5,
                        "estimated_per_token_gap_ms": 2.5,
                        "chunk_tokens": 2,
                    }
                ],
            ),
        ]
        gaps, active_end = collect_all_anchor_overlap_gaps(
            anchors, probe_submit=10.0, probe_first=30.0
        )
        self.assertEqual(active_end, 15.0)
        self.assertEqual(gaps, [3.0, 2.5, 2.5])
        self.assertEqual(
            anchors[0].token_rows[1]["during_probe_prefill"], 0
        )

    def test_native_resident_budget_skips_only_infeasible_states(self) -> None:
        admitted, skipped = build_profile_state_plan(
            [25_000, 50_000, 75_000, 100_000, 125_000],
            [1, 2, 3],
            auxiliary_context_tokens=10_000,
            max_total_input_tokens=122_880,
        )
        self.assertIn((50_000, 3), admitted)
        self.assertNotIn((75_000, 1), admitted)
        self.assertIn(
            {
                "context_tokens": 75_000,
                "active_decode_requests": 1,
                "estimated_total_input_tokens": 150_000,
                "reason": "resident_input_budget",
            },
            skipped,
        )
        ring_admitted, ring_skipped = build_profile_state_plan(
            [125_000],
            [3],
            auxiliary_context_tokens=10_000,
            max_total_input_tokens=0,
        )
        self.assertEqual(ring_admitted, [(125_000, 3)])
        self.assertEqual(ring_skipped, [])

    def test_disabled_resident_prefilter_admits_all_profile_states(
        self,
    ) -> None:
        contexts = [25_000, 50_000, 75_000, 100_000]
        decode_counts = [1, 2, 3]
        admitted, skipped = build_profile_state_plan(
            contexts,
            decode_counts,
            auxiliary_context_tokens=10_000,
            max_total_input_tokens=0,
        )
        self.assertEqual(
            admitted,
            [
                (context, decode_count)
                for context in contexts
                for decode_count in decode_counts
            ],
        )
        self.assertEqual(skipped, [])

    def test_fixed_probe_context_changes_only_capacity_accounting(self) -> None:
        admitted, skipped = build_profile_state_plan(
            [100_000],
            [1],
            auxiliary_context_tokens=10_000,
            max_total_input_tokens=110_000,
            probe_context_tokens=10_000,
        )
        self.assertEqual(admitted, [(100_000, 1)])
        self.assertEqual(skipped, [])


class VendorV8SMScalingProofTest(unittest.TestCase):
    def test_quick_window_uses_pre_probe_tail_and_probe_overlap(self) -> None:
        anchor_rows = [
            {
                "gap_start_monotonic_ts": float(index),
                "arrival_monotonic_ts": float(index) + 0.5,
                "estimated_per_token_gap_ms": float(index * 10),
                "chunk_tokens": 1,
            }
            for index in range(1, 9)
        ]
        window = ProbeWindow(
            phase="measurement",
            repeat_index=1,
            control_lower_bound_ts=1.0,
            result=StreamResult(
                row={
                    "submit_monotonic_ts": 5.0,
                    "first_token_monotonic_ts": 8.0,
                },
                token_rows=[],
            ),
        )
        control, overlap = classify_probe_window(
            anchor_rows, window, control_gap_count=2
        )
        self.assertEqual(
            [row["estimated_per_token_gap_ms"] for row in control],
            [30.0, 40.0],
        )
        self.assertEqual(
            [row["estimated_per_token_gap_ms"] for row in overlap],
            [50.0, 60.0, 70.0],
        )

    def test_clustered_summary_identifies_36sm_p50_pass(self) -> None:
        baseline = {
            100_000: {
                repeat: [100.0, 101.0, 102.0]
                for repeat in range(1, 6)
            }
        }
        partitions = {
            ("d36_p4", 36, 4, "native", 100_000, 10_000, 1): {
                "control": {
                    repeat: [111.0, 112.0, 113.0]
                    for repeat in range(1, 6)
                },
                "overlap": {
                    repeat: [122.0, 123.0, 124.0]
                    for repeat in range(1, 6)
                },
            }
        }
        rows = build_summary(
            baseline,
            partitions,
            {
                (
                    "d36_p4",
                    36,
                    4,
                    "native",
                    100_000,
                    10_000,
                    1,
                ): ("independent_request_repeat", True)
            },
            total_sms=40,
            quantile=50,
            slo_factor=1.25,
            assumed_overlap_overhead=0.10,
            bootstrap_iterations=100,
            seed=7,
            min_repeats=5,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["point_slo_met"], 1)
        self.assertEqual(
            rows[0]["confidence_slo_verdict"], "pass_95ci"
        )

    def test_same_anchor_windows_are_descriptive_only(self) -> None:
        key = ("d36_p4", 36, 4, "ring", 100_000, 2_048, 1)
        baseline = {
            100_000: {
                repeat: [100.0, 101.0, 102.0]
                for repeat in range(1, 6)
            }
        }
        partitions = {
            key: {
                "control": {
                    repeat: [111.0, 112.0, 113.0]
                    for repeat in range(1, 6)
                },
                "overlap": {
                    repeat: [122.0, 123.0, 124.0]
                    for repeat in range(1, 6)
                },
            }
        }
        rows = build_summary(
            baseline,
            partitions,
            {key: ("probe_window_on_single_decode_anchor", False)},
            total_sms=40,
            quantile=50,
            slo_factor=1.25,
            assumed_overlap_overhead=0.10,
            bootstrap_iterations=100,
            seed=7,
            min_repeats=5,
        )
        self.assertEqual(
            rows[0]["confidence_slo_verdict"],
            "descriptive_only_window_clusters",
        )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class VendorV8PartitionAggregationTest(unittest.TestCase):
    def test_phase_local_prefill_and_overlap_tbt_profiles_are_emitted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "cases" / "ring" / "d20_p20"
            case.mkdir(parents=True)
            (case / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "profile_id": "d20_p20",
                        "decode_sms": 20,
                        "prefill_sms": 20,
                        "prefill_mode": "ring",
                        "tbt_token_count_source": "choice.token_ids",
                        "tbt_requires_return_token_ids": True,
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                case / "trial_summary.csv",
                [
                    "phase",
                    "status",
                    "context_tokens",
                    "active_decode_requests",
                    "repeat_index",
                    "probe_ttft_ms",
                    "probe_effective_prefill_tokens_per_s",
                    "overlap_decode_tokens_per_s",
                    "ttft_slo_met",
                    "tbt_token_slo_attainment",
                    "probe_submit_monotonic_ts",
                    "probe_first_token_monotonic_ts",
                ],
                [
                    {
                        "phase": "measurement",
                        "status": "success",
                        "context_tokens": 25_000,
                        "active_decode_requests": 1,
                        "repeat_index": 1,
                        "probe_ttft_ms": 100.0,
                        "probe_effective_prefill_tokens_per_s": 250_000.0,
                        "overlap_decode_tokens_per_s": 100.0,
                        "ttft_slo_met": 1,
                        "tbt_token_slo_attainment": 1.0,
                        "probe_submit_monotonic_ts": 10.0,
                        "probe_first_token_monotonic_ts": 20.0,
                    }
                ],
            )
            write_csv(
                case / "client_token_timeline.csv",
                [
                    "phase",
                    "role",
                    "during_probe_prefill",
                    "estimated_per_token_gap_ms",
                    "context_tokens",
                    "active_decode_requests",
                    "repeat_index",
                    "chunk_tokens",
                    "gap_start_monotonic_ts",
                    "arrival_monotonic_ts",
                ],
                [
                    {
                        "phase": "measurement",
                        "role": "decode_anchor",
                        "during_probe_prefill": 1,
                        "estimated_per_token_gap_ms": 12.0,
                        "context_tokens": 25_000,
                        "active_decode_requests": 1,
                        "repeat_index": 1,
                        "chunk_tokens": 1,
                        "gap_start_monotonic_ts": 14.0,
                        "arrival_monotonic_ts": 16.0,
                    },
                    {
                        "phase": "measurement",
                        "role": "decode_anchor",
                        "during_probe_prefill": 1,
                        "estimated_per_token_gap_ms": 14.0,
                        "context_tokens": 25_000,
                        "active_decode_requests": 1,
                        "repeat_index": 1,
                        "chunk_tokens": 1,
                        "gap_start_monotonic_ts": 15.005,
                        "arrival_monotonic_ts": 15.015,
                    },
                ],
            )
            (case / "iteration_metrics.jsonl").write_text(
                json.dumps(
                    {
                        "schedule_start_ts": 15.0,
                        "worker_execute_start_ts": 15.0,
                        "num_prefill_tokens": 2048,
                        "worker_execute_duration_s": 0.020,
                        "virtual_context_active_decode_ring_prefill_allowed_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary, prefill, tbt, _metadata = aggregate_profiles(
                [case],
                chunk_tokens=2048,
                allow_incomplete=False,
            )
            self.assertEqual(len(summary), 1)
            self.assertEqual(
                prefill["points"][0],
                {
                    "profile_id": "d20_p20",
                    "prefill_mode": "ring",
                    "max_context_tokens": 25_000,
                    "max_chunk_tokens": 2048,
                    "p95_prefill_chunk_ms": 20.0,
                },
            )
            self.assertEqual(tbt["schema_version"], 2)
            self.assertEqual(
                tbt["measurement_contract"]["unit"],
                "ms_per_output_token",
            )
            self.assertEqual(
                tbt["points"][0]["p95_per_token_tbt_ms"], 13.9
            )


class VendorV8ShellDryRunTest(unittest.TestCase):
    def test_mixed_iteration_clears_scheduler_only_baseline_runtime_paths(
        self,
    ) -> None:
        helper = ROOT / "tools/pd_eval_policy_env.sh"
        script = f"""
set -euo pipefail
source {shlex.quote(str(helper))}
VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT=1
VLLM_PD_DECOUPLED_NO_MIX=1
VLLM_PD_INDEPENDENT_PHASE_QUEUES=1
VLLM_PD_ASYNC_UNIPROC_EXECUTOR=1
VLLM_VIRTUAL_CONTEXT=1
VLLM_VIRTUAL_CONTEXT_RING_PREFILL=1
VLLM_VIRTUAL_CONTEXT_TIMING=0
VLLM_PD_EVAL_POLICY=mixed_iteration
pd_eval_policy_apply
printf '%s,%s,%s,%s,%s,%s,%s\n' \
  "$VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT" \
  "$VLLM_PD_DECOUPLED_NO_MIX" \
  "$VLLM_PD_INDEPENDENT_PHASE_QUEUES" \
  "$VLLM_PD_ASYNC_UNIPROC_EXECUTOR" \
  "$VLLM_VIRTUAL_CONTEXT" \
  "$VLLM_VIRTUAL_CONTEXT_RING_PREFILL" \
  "$VLLM_VIRTUAL_CONTEXT_TIMING"
"""
        completed = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "0,0,0,0,0,0,0")

    def test_baseline_plus_dry_run_enforces_scheduler_only_contract(
        self,
    ) -> None:
        runner = ROOT / "run_vendor_v8_dynamic_hbm_vs_full.sh"
        env = {
            **os.environ,
            "BENCHMARK_MODE": "full",
            "BASE_EXPERIMENT_NAME": "baseline_plus_cpu_dry_run",
            "BENCHMARK_VARIANTS": "baseline_plus",
            "WORKLOAD": "C",
            "VLLM_VENDOR_DIR": str(ROOT / "vendor_v8"),
            # A matched sweep normally requests the bundle for full_vendor_v8.
            # The wrapper must scope it away from this scheduler-only case.
            "FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS": "1",
            "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF": "0",
            "VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH": "0",
            "BASELINE_PLUS_PREFILL_QUANTUM_TOKENS": "0",
        }
        completed = subprocess.run(
            ["bash", str(runner), "--dry-run"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"{completed.stdout}\n{completed.stderr}",
        )
        self.assertIn("system_variant=baseline_plus", completed.stdout)
        self.assertIn(
            "execution_profile=native_full_gpu_40sm_scheduler_only",
            completed.stdout,
        )
        self.assertIn(
            "scheduler_only_contract=prefill_srpt_decode_fcfs "
            "quantum_tokens=0",
            completed.stdout,
        )
        self.assertIn(
            "scheduler_only_disabled=pd_partition,dynamic_sm,greencontext,"
            "context_ring,victim,layerwise",
            completed.stdout,
        )
        self.assertIn("throughput_optimizations=0", completed.stdout)

        rejected = subprocess.run(
            ["bash", str(runner), "--dry-run"],
            cwd=ROOT,
            env={
                **env,
                "BASELINE_PLUS_PREFILL_QUANTUM_TOKENS": "2048",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn(
            "requires BASELINE_PLUS_PREFILL_QUANTUM_TOKENS=0",
            rejected.stderr,
        )

    def test_muxwise_capacity_modes_select_wait_or_recompute(self) -> None:
        helper = ROOT / "tools/pd_eval_policy_env.sh"
        script = f"""
set -euo pipefail
source {shlex.quote(str(helper))}
VLLM_PD_EVAL_POLICY=muxwise_dynamic
VLLM_PD_EVAL_NATIVE_PREFILL_CAPACITY_MODE="$1"
pd_eval_policy_apply
printf '%s,%s\\n' \
  "$VLLM_PD_MUXWISE_DYNAMIC_POLICY" \
  "$VLLM_PD_WAIT_ON_PREFILL_KV_CAPACITY"
"""
        expected = {
            "guard_wait": "1,1",
            "legacy_preempt_recompute": "0,0",
        }
        for mode, output in expected.items():
            completed = subprocess.run(
                ["bash", "-c", script, "capacity-mode-test", mode],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), output)

    def test_validation_runner_defaults_to_profile_free_srpt(self) -> None:
        runner = (ROOT / "run_vendor_v8_short_validation.sh").read_text()
        # Assert behavior rather than a single global assignment: legacy
        # Dynamic-HBM now has its own FCFS default, while full vendor stays SRPT.
        env = os.environ.copy()
        env.pop("PREFILL_SCHEDULING_POLICY", None)
        env["SYSTEM_VARIANT"] = "full_vendor_v8"
        completed = subprocess.run(
            ["bash", str(ROOT / "run_vendor_v8_short_validation.sh"), "--dry-run"],
            cwd=ROOT, env=env, text=True, capture_output=True,
            check=False, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("[PLAN] prefill_scheduling_policy=srpt", completed.stdout)
        self.assertIn("[PLAN] prefill_profile=disabled", completed.stdout)
        self.assertIn("--ttft-prefill-chunk-scheduling-policy srpt", completed.stdout)
        self.assertIn(
            '"${PREFILL_SCHEDULING_POLICY}" == "slo_srpt"',
            runner,
        )
        self.assertNotIn(
            '"${REQUEST_SLO_POLICY}" "${PREFILL_SERVICE_PROFILE}"',
            runner,
        )

    def test_partition_and_validation_runners_have_side_effect_free_dry_run(
        self,
    ) -> None:
        commands = [
            ROOT / "run_vendor_v8_partition_profiler.sh",
            ROOT / "run_vendor_v8_quick_tbt_sm_scaling.sh",
            ROOT / "run_vendor_v8_short_validation.sh",
        ]
        for command in commands:
            completed = subprocess.run(
                ["bash", str(command), "--dry-run"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "EXPERIMENT_NAME": "vendor_v8_cpu_dry_run",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{command}\n{completed.stdout}\n{completed.stderr}",
            )
            self.assertIn("[DRY-RUN]", completed.stdout)

    def test_partition_runner_disables_resident_prefilter_by_default(
        self,
    ) -> None:
        env = {
            **os.environ,
            "EXPERIMENT_NAME": "vendor_v8_cpu_dry_run",
        }
        env.pop("NATIVE_PROFILE_MAX_RESIDENT_TOKENS", None)
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "run_vendor_v8_partition_profiler.sh"),
                "--dry-run",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"{completed.stdout}\n{completed.stderr}",
        )
        self.assertIn(
            "[PLAN] native_profile_max_resident_tokens=0",
            completed.stdout,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
