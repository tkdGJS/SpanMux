#!/usr/bin/env python3
"""CPU-only checks for fixed-split idle-Prefill timeline rendering."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "graph_script/generate_four_case_sm_phase_timelines.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("four_case_sm_phase_idle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FourCaseSMPhaseIdleTest(unittest.TestCase):
    def test_independent_axis_uses_each_case_e2e(self) -> None:
        source = inspect.getsource(MODULE.main)
        self.assertIn(
            "case.e2e_s if args.independent_x else common_x_max_s",
            source,
        )
        self.assertIn('"x_axis_mode"', source)

    def test_origin_and_long1_lifecycle_have_distinct_semantics(self) -> None:
        source = inspect.getsource(MODULE.load_case)
        self.assertIn("min(request_submit_values, default=-1.0)", source)
        self.assertIn(
            "long1_completion_ts - long1_submit_ts",
            source,
        )

    def test_victim_interval_requires_an_explicit_resume_marker(self) -> None:
        metrics = [
            {"iteration_id": 10, "output_process_end_ts": 12.5},
            {"iteration_id": 11, "output_process_end_ts": 20.5},
            {"iteration_id": 12, "output_process_end_ts": 31.0},
        ]
        # File order may differ from scheduler-time order under async futures.
        timeline = [
            {
                "iteration_id": 12,
                "timestamp": 30.0,
                "request_id": "run-long-2-0-id",
                "scheduled_kind": "resumed",
            },
            {
                "iteration_id": 11,
                "timestamp": 20.0,
                "request_id": "run-short-1-0-id",
                "scheduled_kind": "running",
            },
            {
                "iteration_id": 10,
                "timestamp": 10.0,
                "request_id": "run-long-2-0-id",
                "scheduled_kind": "running",
            },
        ]
        self.assertEqual(
            MODULE.victim_suspension_intervals(metrics, timeline),
            [("Long2", 12.5, 30.0, "resumed")],
        )

    def test_ordinary_unscheduled_gap_is_not_labeled_as_victim(self) -> None:
        metrics = [
            {"iteration_id": 1, "output_process_end_ts": 2.0},
            {"iteration_id": 2, "output_process_end_ts": 101.0},
        ]
        timeline = [
            {
                "iteration_id": 1,
                "timestamp": 1.0,
                "request_id": "run-short-1-0-id",
                "scheduled_kind": "new",
            },
            {
                "iteration_id": 2,
                "timestamp": 100.0,
                "request_id": "run-short-1-0-id",
                "scheduled_kind": "running",
            },
        ]
        self.assertEqual(
            MODULE.victim_suspension_intervals(metrics, timeline),
            [],
        )

    def test_fixed_split_capacity_rows_form_one_merged_idle_interval(self) -> None:
        rows = [
            {
                "scheduled_ts": 1.0,
                "model_done_ts": 1.4,
                "pd_dynamic_sm_capacity_serialized_fixed_split": 0,
            },
            {
                "scheduled_ts": 2.0,
                "model_done_ts": 2.4,
                "pd_dynamic_sm_capacity_serialized_fixed_split": 1,
            },
            {
                "scheduled_ts": 2.5,
                "model_done_ts": 2.9,
                "pd_dynamic_sm_capacity_serialized_fixed_split": 1,
            },
            {
                "scheduled_ts": 3.0,
                "model_done_ts": 3.4,
                "pd_dynamic_sm_capacity_serialized_fixed_split": 0,
            },
        ]
        self.assertEqual(
            MODULE.capacity_serialized_prefill_idle_intervals(rows),
            [(2.0, 3.0)],
        )

    def test_only_prefill_ownership_is_blank_during_idle_interval(self) -> None:
        records = [
            ("Long1", "decode", 1.0, 3.0, 0.0, 30.0),
            ("Long2", "prefill", 1.0, 3.0, 30.0, 10.0),
        ]
        filtered = MODULE.blank_prefill_ownership(
            records,
            idle_intervals=[(2.0, 3.0)],
        )
        self.assertIn(records[0], filtered)
        self.assertIn(("Long2", "prefill", 1.0, 1.0, 30.0, 10.0), filtered)
        self.assertIn(("Long2", "prefill", 3.0, 1.0, 30.0, 10.0), filtered)
        self.assertFalse(
            any(
                phase == "prefill" and start < 3.0 and start + width > 2.0
                for _label, phase, start, width, _y, _height in filtered
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
