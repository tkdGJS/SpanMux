"""CPU contracts for latency denominators and missing measurements."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graph_script"))
from generate_e2e_latency_breakdown import (  # noqa: E402
    COLORS, grouped_means, grouped_totals, render, request_components, summarize, total_parts,
)


class E2ELatencyBreakdown(unittest.TestCase):
    def client(self, *, rid="a", index=0, start=100.0, first=130.0, end=140.0):
        return dict(request_id=rid, event_index=index, dataset="sharegpt", input_tokens=100,
                    submit_monotonic_ts=start, first_token_monotonic_ts=first,
                    completion_monotonic_ts=end, ttft_ms=(first - start) * 1000)

    def server(self):
        # Different absolute origin: only durations can be compared.
        return dict(tenant_id="a", prompt_tokens=100, queued_ts=200000.0,
                    scheduled_ts=200010.0, first_token_ts=200022.0,
                    queued_time_s=10.0, prefill_time_s=12.0)

    def test_known_components_conserve_client_e2e_without_cross_clock_subtraction(self):
        r = request_components(self.client(), self.server())
        self.assertEqual(r["initial_wait_s"], 10)
        self.assertEqual(r["prefill_elapsed_s"], 12)
        self.assertEqual(r["decode_elapsed_s"], 10)
        self.assertEqual(r["other_elapsed_s"], 8)
        self.assertEqual(r["ttft_unseparated_s"], 0)
        self.assertEqual(total_parts(r), r["client_e2e_s"])

    def test_missing_server_metric_is_unsplit_ttft_not_zero_waiting(self):
        r = request_components(self.client(), None)
        self.assertIsNone(r["initial_wait_s"])
        self.assertIsNone(r["prefill_elapsed_s"])
        self.assertEqual(r["ttft_unseparated_s"], 30)
        self.assertEqual(total_parts(r), 40)

    def test_overlapping_means_differ_from_makespan_and_last_path_sums_exactly(self):
        rows = [request_components(self.client(end=145), None),
                request_components(self.client(rid="b", index=1, start=140, first=145, end=150), None)]
        mean, path = summarize("test", rows, 100, 60)
        self.assertEqual(mean["mean_request_e2e_s"], 27.5)
        self.assertEqual(total_parts(mean), 27.5)
        self.assertEqual(path["last_request_id"], "b")  # not the longest request
        self.assertEqual(path["last_request_e2e_s"], 10)
        self.assertEqual(path["before_last_submit_s"], 40)
        self.assertEqual(path["observation_tail_s"], 10)
        self.assertEqual(total_parts(path), 60)

    def test_invalid_timings_and_partial_coverage_are_rejected(self):
        with self.assertRaises(ValueError):
            request_components(self.client(first=99), None)
        with self.assertRaises(ValueError):
            request_components(dict(self.client(), ttft_ms=float("nan")), None)
        with self.assertRaises(ValueError):
            request_components(self.client(first=110), self.server())
        rows = [request_components(self.client(), None), request_components(self.client(), self.server())]
        with self.assertRaisesRegex(ValueError, "partial server"):
            summarize("test", rows, 100, 60)
        with self.assertRaisesRegex(ValueError, "outside observation"):
            summarize("test", rows[:1], 100, 20)

    def grouped_fixture(self):
        rows = []
        for case in COLORS:
            for dataset, prefills, ttft_base in (("sharegpt", (8.0, 12.0), 40.0),
                                               ("leval", (20.0, 24.0), 80.0)):
                for index, prefill in enumerate(prefills):
                    rid = f"{dataset}-{index}"
                    first = 100.0 + ttft_base + index * 4
                    client = dict(self.client(rid=rid, index=index, first=first, end=first + 10),
                                  dataset=dataset)
                    server = None if case == "Baseline" else dict(
                        tenant_id=rid, prompt_tokens=100, queued_ts=2000.0,
                        scheduled_ts=2005.0, first_token_ts=2005.0 + prefill,
                        queued_time_s=5.0, prefill_time_s=prefill)
                    rows.append(dict(case=case, **request_components(client, server)))
        return rows

    def test_grouped_proxy_uses_matching_group_mean_and_preserves_raw_missing_values(self):
        raw = self.grouped_fixture()
        before = deepcopy(raw)
        result = grouped_means(raw, baseline_prefill_proxy="srpt-group-mean")
        self.assertEqual(len(result), 10)
        for row in result:
            self.assertEqual(row["request_count"], 2)
            self.assertAlmostEqual(total_parts(row), row["mean_request_e2e_s"])
            self.assertEqual(row["decode_elapsed_s"], 10)
            self.assertEqual(row["ttft_unseparated_s"], 0)
            if row["case"] == "Baseline":
                expected_prefill = 10 if row["group"] == "sharegpt" else 22
                expected_ttft = 42 if row["group"] == "sharegpt" else 82
                self.assertEqual(row["prefill_elapsed_s"], expected_prefill)
                self.assertEqual(row["initial_wait_s"], expected_ttft - expected_prefill)
                self.assertIsNone(row["observed_initial_wait_s"])
                self.assertIsNone(row["observed_prefill_elapsed_s"])
                self.assertEqual(row["observed_ttft_unseparated_s"], expected_ttft)
                self.assertFalse(row["server_metrics_available"])
                self.assertTrue(row["baseline_split_is_hypothetical"])
                self.assertTrue(row["baseline_other_assumed_zero"])
                self.assertEqual(row["proxy_source_group"], row["group"])
            else:
                self.assertFalse(row["baseline_split_is_hypothetical"])
                self.assertEqual(row["initial_wait_s"], row["observed_initial_wait_s"])
        self.assertEqual(raw, before)

    def test_grouped_default_keeps_unsplit_baseline_and_rejects_invalid_proxy(self):
        raw = self.grouped_fixture()
        for row in grouped_means(raw):
            if row["case"] == "Baseline":
                self.assertIsNone(row["initial_wait_s"])
                self.assertIsNone(row["prefill_elapsed_s"])
                self.assertFalse(row["baseline_split_is_hypothetical"])
                self.assertGreater(row["ttft_unseparated_s"], 0)
        with self.assertRaisesRegex(ValueError, "all five"):
            grouped_means([r for r in raw if r["case"] != "Baseline+SRPT"])
        with self.assertRaisesRegex(ValueError, "identities differ"):
            grouped_means(raw[1:])
        for row in raw:
            if row["case"] == "Baseline":
                row.update(first_token_ts=101.0, completion_ts=111.0,
                           client_e2e_s=11.0, ttft_unseparated_s=1.0)
        with self.assertRaisesRegex(ValueError, "negative hypothetical"):
            grouped_means(raw, baseline_prefill_proxy="srpt-group-mean")

    def test_group_figure_has_four_components_and_visible_hypothetical_label(self):
        import matplotlib
        matplotlib.use("Agg")
        rows = [r for r in grouped_means(self.grouped_fixture(), baseline_prefill_proxy="srpt-group-mean")
                if r["group"] == "sharegpt"]
        figures = []
        with patch("matplotlib.figure.Figure.savefig", autospec=True,
                   side_effect=lambda fig, *args, **kwargs: figures.append(fig)):
            render(rows, terminal_path=False, output=Path("unused"), group="sharegpt", baseline_proxy=True)
        self.assertEqual(len(figures), 2)  # PNG + PDF from the same single axes
        fig = figures[0]
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual([t.get_text() for t in fig.legends[0].texts],
                         ["Initial queue wait", "Post-scheduling Prefill", "Decode elapsed", "Other"])
        self.assertEqual(fig.axes[0].get_xticklabels()[0].get_text(), "Baseline*")
        self.assertTrue(any("hypothetical" in t.get_text() for t in fig.texts))

    def test_totals_match_raw_latency_sums_with_unequal_group_counts(self):
        raw = [r for r in self.grouped_fixture() if r["dataset"] == "sharegpt" or r["event_index"] == 0]
        before = deepcopy(raw)
        for proxy in ("none", "srpt-group-mean"):
            means = grouped_means(raw, baseline_prefill_proxy=proxy)
            totals = grouped_totals(raw, baseline_prefill_proxy=proxy)
            for mean, total in zip(means, totals):
                n = 2 if total["group"] == "sharegpt" else 1
                self.assertEqual(total["request_count"], n)
                self.assertEqual(total["total_request_e2e_s"], n * mean["mean_request_e2e_s"])
                selected = [r for r in raw if r["case"] == total["case"] and
                            (r["dataset"] == "sharegpt") == (total["group"] == "sharegpt")]
                self.assertEqual(total["total_request_e2e_s"], sum(r["client_e2e_s"] for r in selected))
                self.assertAlmostEqual(total_parts(total), total["total_request_e2e_s"])
                self.assertNotIn("mean_request_e2e_s", total)
                for key in ("initial_wait_s", "prefill_elapsed_s", "decode_elapsed_s", "other_elapsed_s",
                            "observed_initial_wait_s", "observed_prefill_elapsed_s"):
                    self.assertEqual(total[key], None if mean[key] is None else mean[key] * n)
                self.assertEqual(total["baseline_split_is_hypothetical"], mean["baseline_split_is_hypothetical"])
        self.assertEqual(raw, before)

    def test_total_figure_is_labeled_as_request_sum_not_experiment_elapsed(self):
        import matplotlib
        matplotlib.use("Agg")
        rows = [r for r in grouped_totals(self.grouped_fixture(), baseline_prefill_proxy="srpt-group-mean")
                if r["group"] == "sharegpt"]
        figures = []
        with patch("matplotlib.figure.Figure.savefig", autospec=True,
                   side_effect=lambda fig, *args, **kwargs: figures.append(fig)):
            render(rows, terminal_path=False, output=Path("unused"), group="sharegpt",
                   baseline_proxy=True, summed=True)
        fig = figures[0]
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.legends[0].texts), 4)
        self.assertEqual(fig.axes[0].get_ylabel(), "Summed request latency (s)")
        text = "\n".join(t.get_text() for t in fig.texts)
        self.assertIn("Total request E2E", text)
        self.assertIn("not experiment wall-clock", text)
        self.assertIn("hypothetical", text)
        self.assertNotIn("Totals are request means", text)


if __name__ == "__main__":
    unittest.main()
