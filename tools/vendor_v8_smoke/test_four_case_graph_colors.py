#!/usr/bin/env python3
"""CPU-only contract for stable four-case report colors."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "graph_script/generate_four_case_experiment_comparison.py"
SPEC = importlib.util.spec_from_file_location("four_case_graph_colors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FourCaseGraphColorTest(unittest.TestCase):
    def test_workload_b_origin_is_first_production_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            results_dir = Path(raw_dir)
            (results_dir / "request_results.csv").write_text(
                "role,request_index,status,submit_monotonic_ts,"
                "completion_monotonic_ts,context_tokens,output_tokens\n"
                "short,0,success,100,120,10000,256\n"
                "long,0,success,400,500,50000,8192\n",
                encoding="utf-8",
            )
            (results_dir / "client_token_timeline.csv").write_text(
                "request_id,role,request_index,token_index,"
                "arrival_monotonic_ts,tbt_slo_met\n",
                encoding="utf-8",
            )
            (results_dir / "run_metadata.json").write_text(
                '{"status": "completed"}\n',
                encoding="utf-8",
            )

            case = MODULE.load_case("Workload B", results_dir)

        self.assertEqual(case.production_origin_ts, 100.0)
        self.assertEqual(case.e2e_s, 400.0)

    def test_cli_accepts_available_three_case_comparisons(self) -> None:
        main_source = inspect.getsource(MODULE.main)
        self.assertIn("2 <= len(args.case) <= 4", main_source)
        self.assertIn("len(tbt_timeline_cases) >= 2", main_source)

    def test_retired_version_one_plus_cli_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(MODULE.argparse.ArgumentTypeError, "retired"):
            MODULE.parse_case_spec("Version 1+=unused")

    def test_headline_colors_do_not_depend_on_label_order(self) -> None:
        native_labels = (
            "Native 40SM chunked",
            "native_40sm_chunked",
        )
        vendor_labels = (
            "Full vendor_v8",
            "full_vendor_v8",
        )
        for label in native_labels:
            self.assertEqual(
                MODULE.case_color(label),
                MODULE.NATIVE_40SM_COLOR,
            )
        for label in vendor_labels:
            self.assertEqual(
                MODULE.case_color(label),
                MODULE.FULL_VENDOR_V8_COLOR,
            )
        self.assertEqual(MODULE.NATIVE_40SM_COLOR, "#1F77B4")
        self.assertEqual(MODULE.FULL_VENDOR_V8_COLOR, "#FF7F0E")

    def test_dynamic_cases_have_distinct_stable_colors(self) -> None:
        fixed = MODULE.case_color("Dynamic serialized split")
        full = MODULE.case_color("Dynamic serialized full SM")
        self.assertEqual(fixed, MODULE.DYNAMIC_FIXED_SPLIT_COLOR)
        self.assertEqual(full, MODULE.DYNAMIC_FULL_SM_COLOR)
        self.assertNotEqual(fixed, full)

    def test_presentation_labels_keep_semantic_case_colors(self) -> None:
        expected = {
            "Baseline": MODULE.NATIVE_40SM_COLOR,
            "Version 1": MODULE.DYNAMIC_FULL_SM_COLOR,
            "Version 2": MODULE.FULL_VENDOR_V8_COLOR,
        }
        for label, color in expected.items():
            with self.subTest(label=label):
                self.assertEqual(MODULE.case_color(label), color)

    def test_request_timelines_receive_one_common_x_axis(self) -> None:
        plot_source = inspect.getsource(MODULE.plot_case_timeline)
        main_source = inspect.getsource(MODULE.main)
        self.assertIn("x_limits_s: tuple[float, float] | None", plot_source)
        self.assertIn("lifecycle_axis.set_xlim(*x_limits_s)", plot_source)
        self.assertIn("token_axis.set_xlim(*x_limits_s)", plot_source)
        self.assertIn(
            "common_x_max_s = max(case.e2e_s for case in cases)",
            main_source,
        )
        self.assertIn("x_limits_s=common_x_limits_s", main_source)

    def test_tbt_timeline_uses_semantic_case_order(self) -> None:
        cases = [
            SimpleNamespace(label="Version 2"),
            SimpleNamespace(label="Baseline"),
            SimpleNamespace(label="Version 1"),
            SimpleNamespace(label="Dynamic serialized split (legacy)"),
        ]
        selected = MODULE.select_tbt_timeline_cases(cases)
        self.assertEqual(
            [case.label for case in selected],
            [
                "Baseline",
                "Version 1",
                "Version 2",
                "Dynamic serialized split (legacy)",
            ],
        )

    def test_tbt_points_are_relative_and_exclude_warmup(self) -> None:
        case = SimpleNamespace(
            production_origin_ts=100.0,
            request_rows=[{"request_id": "production"}],
            token_rows=[
                {
                    "request_id": "warmup",
                    "arrival_monotonic_ts": "101.0",
                    "estimated_per_token_gap_ms": "9.0",
                },
                {
                    "request_id": "production",
                    "arrival_monotonic_ts": "102.0",
                    "estimated_per_token_gap_ms": "60.0",
                },
                {
                    "request_id": "production",
                    "arrival_monotonic_ts": "101.0",
                    "estimated_per_token_gap_ms": "",
                },
            ],
        )
        self.assertEqual(MODULE.tbt_timeline_points(case), [(2.0, 60.0)])

    def test_tbt_sum_preserves_chunk_weight_and_spike_split(self) -> None:
        case = SimpleNamespace(
            request_rows=[{"request_id": "production"}],
            token_rows=[
                {
                    "request_id": "production",
                    "estimated_per_token_gap_ms": "50",
                    "chunk_tokens": "2",
                },
                {
                    "request_id": "production",
                    "estimated_per_token_gap_ms": "200",
                    "chunk_tokens": "1",
                },
                {
                    "request_id": "warmup",
                    "estimated_per_token_gap_ms": "900",
                    "chunk_tokens": "1",
                },
            ],
        )
        self.assertEqual(MODULE.tbt_sum_breakdown(case), (0.1, 0.2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
