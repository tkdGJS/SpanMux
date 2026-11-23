#!/usr/bin/env python3
"""CPU-only contracts for production-relative offline metric timelines."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graph_script"))
import generate_loongserve_metric_timelines as m


def csv_file(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root: Path) -> m.Case:
    client = root / "client"
    client.mkdir(parents=True)
    (client / "run_metadata.json").write_text(json.dumps({
        "trace_start_monotonic_ts": 1000, "observation_duration_s": 125,
        "observation_end_monotonic_ts": 1125, "workload_sha256": "same-trace",
    }), encoding="utf-8")
    rows = []
    for index, status, submit, first, complete, inputs, outputs in (
        (0, "success", 1000, 1010, 1120, 20, 4),
        (1, "failed", 1003, 1012, 1065, 40, 3),
    ):
        rows.append({
            "event_index": index, "request_id": f"r{index}", "dataset": "sharegpt",
            "status": status, "submit_monotonic_ts": submit,
            "first_token_monotonic_ts": first, "completion_monotonic_ts": complete,
            "input_tokens": inputs, "output_tokens": outputs,
            "requested_output_tokens": 4 if index == 0 else 10,
            "effective_scheduled_offset_s": submit - 1000,
            "ttft_ms": (first - submit) * 1000, "ttft_slo_ms": 15000,
            "ttft_slo_met": 1, "error_message": "" if index == 0 else "cancelled",
        })
    unsubmitted = {key: "" for key in rows[0]}
    unsubmitted.update(event_index=2, request_id="r2", status="not_submitted_request_limit")
    csv_file(client / "request_results.csv", rows + [unsubmitted])
    tokens = []
    for rid, chunks in (("r0", [(1010, 1), (1060, 2), (1120, 1)]),
                        ("r1", [(1012, 1), (1062, 2)])):
        previous = None
        for chunk_index, (when, count) in enumerate(chunks):
            gap = (when - previous) * 1000 / count if previous is not None else None
            tokens.append({
                "event_index": int(rid[1:]), "request_id": rid,
                "chunk_index": chunk_index, "chunk_tokens": count,
                "arrival_monotonic_ts": when, "arrival_offset_from_trace_start_s": when - 1000,
                "gap_start_monotonic_ts": previous,
                "estimated_per_token_gap_ms": gap, "tbt_slo_ms": 30000,
                "tbt_slo_met": int(gap <= 30000) if gap is not None else "",
            })
            previous = when
    csv_file(client / "client_token_timeline.csv", tokens)
    csv_file(client / "workload_summary.csv", [{
        "dataset": "all", "submitted_count": 2, "success_count": 1,
        "observed_output_tokens": 7, "completed_output_tokens": 4,
    }])
    return m.load_case("fixture", "full_vendor_v8", root)


class MetricTimelinesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = fixture(self.root / "run_results")

    def test_origin_and_unsubmitted_exclusion(self) -> None:
        self.assertEqual(len(self.case.requests), 2)
        self.assertEqual(self.case.requests[0]["first_token_s"], 10)
        self.assertEqual(self.case.tokens[0]["time_s"], 10)
        self.assertEqual(len(self.case.hashes), 4)

    def test_exact_chunk_weight_and_first_token_exclusion(self) -> None:
        result = m.summary(self.case)
        self.assertEqual(result["observed_output_tokens"], 7)
        self.assertEqual(result["tbt_eligible_tokens"], 5)
        self.assertEqual(result["tbt_good_tokens"], 4)
        self.assertEqual(result["ttft_good_requests"], 1)
        self.assertAlmostEqual(result["tbt_slo_goodput_tokens_per_s"], 4 / 125)
        self.assertAlmostEqual(result["ttft_slo_goodput_requests_per_min"], 60 / 125)

    def test_failure_output_remains_observed_but_prompt_not_credited(self) -> None:
        bins = m.aggregate(self.case, 60)
        self.assertEqual(sum(b["prefill_tokens"] for b in bins), 20)
        self.assertEqual(sum(b["decode_tokens"] for b in bins), 7)
        self.assertEqual(sum(b["successful_decode_tokens"] for b in bins), 4)
        self.assertEqual(sum(b["failed_requests"] for b in bins), 1)

    def test_boundaries_final_exposure_and_integrals(self) -> None:
        bins = m.aggregate(self.case, 60)
        self.assertEqual([b["exposure_s"] for b in bins], [60, 60, 5])
        self.assertEqual([b["decode_tokens"] for b in bins], [2, 4, 1])
        self.assertEqual(bins[-1]["decode_tokens_per_s"], 0.2)
        self.assertIsNone(bins[0]["tbt_p50_ms"])
        for count, rate, scale in (("decode_tokens", "decode_tokens_per_s", 1),
                                   ("prefill_tokens", "prefill_tokens_per_s", 1),
                                   ("tbt_good_tokens", "tbt_slo_goodput_tokens_per_s", 1),
                                   ("ttft_good_requests", "ttft_slo_goodput_requests_per_min", 60)):
            self.assertAlmostEqual(sum(b[count] for b in bins),
                                   sum(b[rate] * b["exposure_s"] / scale for b in bins))
        self.case.tokens[-1]["time_s"] = 125
        self.assertEqual(m.aggregate(self.case, 60)[-1]["decode_tokens"], 1)

    def test_weighted_quantiles(self) -> None:
        self.assertEqual(m.weighted_quantile([(10, 9), (100, 1)], 0.5), 10)
        self.assertEqual(m.weighted_quantile([(10, 9), (100, 1)], 0.95), 100)
        self.assertIsNone(m.weighted_quantile([], 0.5))

    def test_invalid_bin_rejected(self) -> None:
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive"):
                m.aggregate(self.case, value)

    def test_attainment_is_weighted_ratio_not_service_rate(self) -> None:
        bins = m.aggregate(self.case, 60)
        self.assertEqual([b["tbt_slo_attainment"] for b in bins], [None, 1, 0])
        self.assertEqual([b["ttft_slo_attainment_successful"] for b in bins], [1, None, None])
        self.assertEqual(bins[1]["tbt_eligible_tokens"], 4)
        self.assertNotEqual(bins[1]["tbt_slo_attainment"], bins[1]["tbt_slo_goodput_tokens_per_s"])
        # One failed request had an observed first token but is not a successful
        # TTFT denominator member. Its observed output still counts for TBT.
        self.assertEqual(bins[0]["first_token_requests"], 2)
        self.assertEqual(bins[0]["successful_first_token_requests"], 1)
        next(t for t in self.case.tokens if t["tbt_ms"] is not None)["tbt_slo_met"] = False
        weighted = m.aggregate(self.case, 60)[1]
        self.assertEqual(weighted["tbt_slo_attainment"], 0.5)

    def test_attainment_plots_percent_and_preserves_missing_bins(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        bins = {self.case.variant: m.aggregate(self.case, 60)}
        expected = {
            "09_tbt_slo_attainment": [math.nan, 100, 0, 0],
            "10_ttft_slo_attainment": [100, math.nan, math.nan, math.nan],
        }
        for metric in m.ATTAINMENT_METRICS:
            def capture(fig, *_args, **_kwargs):
                self.assertEqual(len(fig.axes), 1)
                ax = fig.axes[0]
                self.assertEqual(ax.get_xlim(), (0, 180))
                self.assertEqual(ax.get_ylim(), (-2, 102))
                self.assertEqual(list(ax.get_yticks()), [0, 20, 40, 60, 80, 100])
                line = ax.lines[0]
                self.assertEqual(line.get_color(), self.case.color)
                self.assertEqual(list(line.get_xdata()), [0, 60, 120, 125])
                for actual, wanted in zip(line.get_ydata(), expected[metric[0]]):
                    if math.isnan(wanted):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertEqual(actual, wanted)
            with mock.patch.object(matplotlib.figure.Figure, "savefig", autospec=True,
                                   side_effect=capture):
                m.plot_metric([self.case], bins, metric, self.root / "one.png", 180, 60)
        self.assertEqual(plt.get_fignums(), [])

    def test_attainment_only_selection_report_and_manifest(self) -> None:
        output = self.root / "attainment"
        with mock.patch.object(m, "discover", return_value=[self.case]), \
             mock.patch.object(m, "plot_metric") as plot:
            m.generate([Path("fixture")], output, 60, slo_attainment_only=True)
        selected = [call.args[2][0] for call in plot.call_args_list]
        self.assertEqual(selected, [item[0] for item in m.ATTAINMENT_METRICS] * 2)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["plotted_metrics"], [item[0] for item in m.ATTAINMENT_METRICS])
        report = (output / "index.html").read_text()
        self.assertIn("09_tbt_slo_attainment.png", report)
        self.assertIn("10_ttft_slo_attainment.png", report)
        self.assertNotIn("03_tbt_slo_goodput.png", report)
        self.assertIn("성공 요청 기준", report)
        self.assertIn("undefined", manifest["attainment_empty_bins"])

    def test_mismatch_rejected(self) -> None:
        second = fixture(self.root / "other")
        m.validate_comparison([self.case, second])
        second.requests[0]["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "workload mismatch"):
            m.validate_comparison([self.case, second])
        second.metadata["workload_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "trace hash mismatch"):
            m.validate_comparison([self.case, second])

    def test_before_warmup_zero_and_after_end_cannot_be_silently_clipped(self) -> None:
        for timestamp in (999, 1126):
            with self.assertRaisesRegex(ValueError, "outside production"):
                m.relative_time(timestamp, 1000, 125, "token")

    def test_token_count_corruption_rejected(self) -> None:
        path = self.case.path / "client" / "workload_summary.csv"
        path.write_text(path.read_text().replace("2,1,7,4", "2,1,8,4"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output summary mismatch"):
            m.load_case("fixture", "full_vendor_v8", self.case.path)

    def test_no_overwrite_or_creation_on_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "already exists"):
            m.generate([], self.root, 60)
        target = self.root / "new_output"
        with self.assertRaisesRegex(ValueError, "no cases"):
            m.generate([self.root], target, 60)
        self.assertFalse(target.exists())

    def test_single_axes_and_fixed_colors_for_every_metric(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        import matplotlib.pyplot as plt
        bins = {self.case.variant: m.aggregate(self.case, 60)}
        counts = []
        def capture(fig, *_args, **_kwargs):
            counts.append(len(fig.axes))
            self.assertEqual(fig.axes[0].get_xlim(), (0, 180))
        with mock.patch.object(matplotlib.figure.Figure, "savefig", autospec=True,
                               side_effect=capture):
            for metric in m.METRICS:
                m.plot_metric([self.case], bins, metric, self.root / "one.png", 180, 60)
        self.assertEqual(counts, [1] * 8)
        self.assertEqual(plt.get_fignums(), [])
        self.assertEqual(m.VARIANTS["dynamic_hbm_serialized_fullsm"][1], "#D62728")
        self.assertEqual(m.VARIANTS["full_vendor_v8"][1], "#FF7F0E")


if __name__ == "__main__":
    unittest.main(verbosity=2)
