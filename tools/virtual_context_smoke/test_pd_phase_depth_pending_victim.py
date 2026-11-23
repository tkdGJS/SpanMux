#!/usr/bin/env python3
"""CPU-only source contracts for bounded P/D depth and pending victims."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDORS = tuple(f"vendor_v{version}" for version in range(3, 8))


def read(vendor: str, relative_path: str) -> str:
    return (ROOT / vendor / relative_path).read_text(encoding="utf-8")


class PDPhaseDepthContractTest(unittest.TestCase):
    def test_all_vendors_have_bounded_phase_depth_wiring(self) -> None:
        for vendor in VENDORS:
            with self.subTest(vendor=vendor):
                output = read(vendor, "vllm/v1/core/sched/output.py")
                engine = read(vendor, "vllm/v1/engine/core.py")
                executor = read(
                    vendor, "vllm/v1/executor/uniproc_executor.py"
                )
                runner = read(vendor, "vllm/v1/worker/gpu_model_runner.py")

                self.assertIn("pd_phase_slot: int = 0", output)
                self.assertIn("pd_phase_submit_seq: int = -1", output)
                for flag in (
                    "VLLM_PD_PHASE_DEPTH_ENABLED",
                    "VLLM_PD_PREFILL_INFLIGHT_DEPTH",
                    "VLLM_PD_DECODE_INFLIGHT_DEPTH",
                    "VLLM_PD_GLOBAL_INFLIGHT_DEPTH",
                ):
                    self.assertIn(flag, engine)
                self.assertIn(
                    "def _assign_pd_phase_submission(", engine
                )
                self.assertIn(
                    "range(self._pd_phase_inflight_limit(phase))", engine
                )
                self.assertIn(
                    "pd_green_context_phase=execution_key", engine
                )

                self.assertIn(
                    "_pd_async_phase_execute_futures: dict[str, Future[Any]]",
                    executor,
                )
                self.assertIn(
                    "VLLM_PD_GLOBAL_INFLIGHT_DEPTH", executor
                )
                self.assertIn(
                    "getattr(scheduler_output, \"pd_phase_slot\", 0)",
                    executor,
                )

                self.assertIn(
                    "tuple[str, int], PDPhaseModelRunnerContext", runner
                )
                self.assertIn("slot: int", runner)
                self.assertIn(
                    "def _parse_pd_phase_execution_key(", runner
                )

    def test_eval_policy_selects_prefill_1_decode_1_global_2(self) -> None:
        policy = (ROOT / "tools/pd_eval_policy_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH:-1', policy
        )
        self.assertIn(
            'VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH:-1', policy
        )
        self.assertIn(
            'VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH:-2', policy
        )

    def test_vendor_v8_raw_defaults_match_prefill_1_decode_1_global_2(
        self,
    ) -> None:
        engine = read("vendor_v8", "vllm/v1/engine/core.py")
        executor = read("vendor_v8", "vllm/v1/executor/uniproc_executor.py")
        self.assertIn(
            'os.environ.get("VLLM_PD_PREFILL_INFLIGHT_DEPTH", "1")',
            engine,
        )
        self.assertIn(
            'os.environ.get("VLLM_PD_DECODE_INFLIGHT_DEPTH", "1")',
            engine,
        )
        self.assertIn(
            'os.environ.get("VLLM_PD_GLOBAL_INFLIGHT_DEPTH", "2")',
            engine,
        )
        self.assertIn(
            'os.environ.get("VLLM_PD_GLOBAL_INFLIGHT_DEPTH", "2")',
            executor,
        )


class PendingVictimContractTest(unittest.TestCase):
    def test_v6_v7_use_advisory_intent_and_revalidation(self) -> None:
        for vendor in ("vendor_v6", "vendor_v7"):
            with self.subTest(vendor=vendor):
                scheduler = read(
                    vendor, "vllm/v1/core/sched/scheduler.py"
                )
                for symbol in (
                    "VLLM_VIRTUAL_CONTEXT_PENDING_VICTIM_INTENT",
                    "PendingResidencyVictimIntent",
                    "def _record_pending_residency_victim_intent(",
                    "pending_victim_async_d2h",
                    "pending_victim_snapshot_retry",
                    "_virtual_context_pending_victim_revalidated_count",
                    "_virtual_context_pending_victim_committed_count",
                    "pending_intent_snapshot",
                ):
                    self.assertIn(symbol, scheduler)
                self.assertIn(
                    "candidate.request_id in self._pd_pending_request_ids",
                    scheduler,
                )
                self.assertIn(
                    "_try_defer_running_prefill_for_virtual_context(",
                    scheduler,
                )

    def test_pending_intent_is_immutable_observation(self) -> None:
        path = (
            ROOT
            / "vendor_v6/vllm/v1/core/sched/residency_policy.py"
        )
        spec = importlib.util.spec_from_file_location(
            "pending_residency_victim_policy_test", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        intent = module.PendingResidencyVictimIntent(
            request_id="req",
            created_iteration_id=7,
            observed_logical_context_tokens=4096,
            observed_reclaimable_blocks=256,
            required_blocks=128,
            target_free_blocks=512,
        )
        self.assertEqual(intent.request_id, "req")
        with self.assertRaises(Exception):
            intent.required_blocks = 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
