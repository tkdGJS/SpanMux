#!/usr/bin/env python3
"""CPU-only contracts for the LoongServe three-case report."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "graph_script/generate_loongserve_three_case_comparison.py"
SPEC = importlib.util.spec_from_file_location("loongserve_three_case_graphs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LoongServeThreeCaseGraphTest(unittest.TestCase):
    def test_current_presentation_contract(self) -> None:
        self.assertEqual(
            MODULE.CASE_ORDER,
            ("Baseline", "Version 1", "Version 2"),
        )
        self.assertEqual(MODULE.CASE_COLORS["Baseline"], "#1F77B4")
        self.assertEqual(MODULE.CASE_COLORS["Version 1"], "#D62728")
        self.assertEqual(MODULE.CASE_COLORS["Version 2"], "#FF7F0E")
        self.assertNotIn("Version 1+", MODULE.CASE_COLORS)

    def test_load_case_uses_client_subdirectory_and_time_limit_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            results_dir = root / "example_results"
            client_dir = results_dir / "client"
            client_dir.mkdir(parents=True)
            (client_dir / "request_results.csv").write_text(
                "event_index,status,submit_monotonic_ts,first_token_monotonic_ts,"
                "completion_monotonic_ts,input_tokens,output_tokens,"
                "ttft_slo_met\n"
                "0,success,100,101,102,10,2,1\n"
                "1,censored_time_limit,101,,110,20,0,\n"
                "2,not_submitted_time_limit,,,,30,0,\n",
                encoding="utf-8",
            )
            (client_dir / "workload_summary.csv").write_text(
                "dataset,submitted_count,success_count,failure_count,"
                "censored_count,observed_output_tokens,ttft_p50_ms,"
                "request_p50_tbt_p50_ms,completion_p50_ms,"
                "ttft_slo_attainment,tbt_token_slo_attainment\n"
                "all,2,1,0,1,2,1000,10,2000,1,1\n",
                encoding="utf-8",
            )
            (client_dir / "client_token_timeline.csv").write_text(
                "chunk_tokens,tbt_slo_met\n"
                "1,1\n"
                "1,0\n",
                encoding="utf-8",
            )
            (client_dir / "run_metadata.json").write_text(
                '{"created_at_utc":"2026-08-18T00:00:10+00:00",'
                '"observation_duration_s":10,"trace_start_monotonic_ts":100,'
                '"workload_sha256":"same-trace"}\n',
                encoding="utf-8",
            )

            case = MODULE.load_case("Version 1", results_dir)

        self.assertEqual(case.label, "Version 1")
        self.assertEqual(case.color, "#D62728")
        self.assertEqual(case.submitted_count, 2)
        self.assertEqual(case.success_count, 1)
        self.assertEqual(case.censored_count, 1)
        self.assertEqual(len(case.submitted_rows), 2)
        self.assertAlmostEqual(case.completed_logical_tokens_per_s, 1.2)
        self.assertAlmostEqual(case.ttft_slo_goodput_requests_per_min, 6.0)
        self.assertAlmostEqual(case.tbt_slo_goodput_tokens_per_s, 0.1)

    def test_retired_version_one_plus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "retired"):
            MODULE.load_case("Version 1+", Path("unused"))

    def test_server_interval_throughput_is_time_weighted(self) -> None:
        metadata = {"created_at_utc": "2026-08-18T00:00:20+00:00"}
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "server.log"
            path.write_text(
                "INFO 08-18 00:00:10 [loggers.py:257] Engine 000: "
                "Avg prompt throughput: 100.0 tokens/s, "
                "Avg generation throughput: 10.0 tokens/s\n"
                "INFO 08-18 00:00:20 [loggers.py:257] Engine 000: "
                "Avg prompt throughput: 200.0 tokens/s, "
                "Avg generation throughput: 20.0 tokens/s\n",
                encoding="utf-8",
            )
            throughput = MODULE.load_server_throughput(path, metadata, 20.0)

        self.assertEqual(throughput.records_used, 2)
        self.assertAlmostEqual(throughput.prompt_tokens, 3000.0)
        self.assertAlmostEqual(throughput.generation_tokens, 300.0)
        self.assertAlmostEqual(throughput.prompt_tokens_per_s, 150.0)
        self.assertAlmostEqual(throughput.all_tokens_per_s, 165.0)

    def test_single_panel_writer_creates_exactly_one_axes(self) -> None:
        import matplotlib.figure

        cases = [SimpleNamespace(label="Baseline", color="#1F77B4")]
        axes_counts: list[int] = []

        def capture(figure: object, *_args: object, **_kwargs: object) -> None:
            axes_counts.append(len(figure.axes))

        with tempfile.TemporaryDirectory() as raw_dir:
            with mock.patch.object(
                matplotlib.figure.Figure,
                "savefig",
                autospec=True,
                side_effect=capture,
            ):
                MODULE.plot_one_case_metric(
                    cases,
                    [1.0],
                    Path(raw_dir) / "one.png",
                    title="one",
                    ylabel="tokens/s",
                    fmt=".1f",
                )

        self.assertEqual(axes_counts, [1])

    def test_single_panel_outputs_cover_throughput_and_slo(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        expected = {
            "01_client_observed_output_goodput.png",
            "02_server_decode_token_throughput.png",
            "03_server_prefill_token_throughput.png",
            "04_server_all_token_throughput.png",
            "05_ttft_slo_attainment.png",
            "06_tbt_slo_attainment.png",
        }
        for filename in expected:
            with self.subTest(filename=filename):
                self.assertIn(filename, source)

    def test_observable_service_order_uses_first_token_then_failure(self) -> None:
        case = SimpleNamespace(
            origin_monotonic_ts=100.0,
            duration_s=30.0,
            submitted_rows=[
                {
                    "event_index": "0",
                    "status": "success",
                    "first_token_monotonic_ts": "120",
                },
                {
                    "event_index": "1",
                    "status": "failed",
                    "completion_monotonic_ts": "110",
                },
                {
                    "event_index": "2",
                    "status": "censored_time_limit",
                },
            ],
        )

        rows = MODULE.observable_service_rows(case)

        self.assertEqual([row[0]["event_index"] for row in rows], ["1", "0"])
        self.assertEqual([row[1] for row in rows], [10.0, 20.0])
        self.assertEqual(
            [row[2] for row in rows], ["failed_response", "first_token"]
        )

    def test_request_timeline_outputs_cover_input_and_output(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        expected = {
            "01_offered_dataset_arrival_order.png",
            "02_offered_input_tokens_timeline.png",
            "03_offered_requested_output_tokens_timeline.png",
            "cumulative_output_comparison.png",
            "three_case_processed_input_tokens_linear.png",
            "three_case_observed_output_tokens_linear.png",
            "offered_request_timeline.csv",
            "client_observable_service_order.csv",
        }
        for filename in expected:
            with self.subTest(filename=filename):
                self.assertIn(filename, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
