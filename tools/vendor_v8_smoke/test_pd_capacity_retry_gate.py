#!/usr/bin/env python3
"""CPU-only contracts for typed phase-capacity retry causes."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vendor_v8/vllm/v1/engine/pd_capacity_retry.py"
ENGINE_CORE_PATH = ROOT / "vendor_v8/vllm/v1/engine/core.py"
SPEC = importlib.util.spec_from_file_location(
    "vendor_v8_pd_capacity_retry_test_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PDPhaseCapacityRetryGate = MODULE.PDPhaseCapacityRetryGate


class CapacityRetryCauseTest(unittest.TestCase):
    def test_tiered_failure_notifies_after_scheduler_cleanup(self) -> None:
        source = ENGINE_CORE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            item
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "EngineCore"
            for item in node.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "_fail_tiered_backend_scheduler_output"
        )
        method_source = ast.get_source_segment(source, method)
        assert method_source is not None
        free_before = method_source.index("free_blocks_before =")
        failure_cleanup = method_source.index(
            "engine_core_outputs = fail_scheduled_requests("
        )
        retry_publish = method_source.index(
            "retry_wake_count = self._notify_pd_capacity_retry_after_output("
        )
        engine_publish = method_source.index(
            "self._notify_pd_engine_progress()"
        )
        self.assertLess(free_before, failure_cleanup)
        self.assertLess(failure_cleanup, retry_publish)
        self.assertLess(retry_publish, engine_publish)

    def test_typed_cause_is_visible_only_below_recorded_threshold(self) -> None:
        gate = PDPhaseCapacityRetryGate()
        self.assertTrue(
            gate.block(
                "prefill",
                minimum_free_blocks=50,
                observed_free_blocks=20,
                reason="prefill_no_progress:kv_capacity",
                cause="prefill_kv_capacity",
                scheduler_iteration_id=7,
                wake_on_output_progress=True,
            )
        )
        self.assertEqual(
            gate.blocking_cause("prefill", 20),
            "prefill_kv_capacity",
        )
        self.assertEqual(
            gate.blocking_cause("prefill", 49),
            "prefill_kv_capacity",
        )
        self.assertEqual(gate.blocking_cause("prefill", 50), "")
        self.assertEqual(
            gate.snapshot(20)["pd_capacity_retry_gate_causes"],
            "prefill_kv_capacity",
        )

    def test_ordinary_decode_output_does_not_clear_capacity_cause(self) -> None:
        gate = PDPhaseCapacityRetryGate()
        gate.block(
            "prefill",
            minimum_free_blocks=12,
            observed_free_blocks=4,
            reason="prefill_no_progress:kv_capacity",
            cause="prefill_kv_capacity",
            scheduler_iteration_id=1,
            wake_on_output_progress=True,
        )
        self.assertEqual(
            gate.notify_output_progress(completed_phase="decode"),
            0,
        )
        self.assertEqual(
            gate.blocking_cause("prefill", 4),
            "prefill_kv_capacity",
        )
        self.assertEqual(
            gate.notify_output_progress(
                completed_phase="decode",
                request_finished=True,
            ),
            1,
        )
        self.assertEqual(gate.blocking_cause("prefill", 4), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
