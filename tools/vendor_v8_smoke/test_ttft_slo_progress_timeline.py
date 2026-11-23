"""CPU contracts for post-experiment TTFT progress; no vLLM/CUDA imports."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graph_script"))
from generate_ttft_slo_progress_timeline import (  # noqa: E402
    arrival_cohorts,
    cumulative,
    normalize_requests,
    sharegpt_count_series,
)


class TTFTProgress(unittest.TestCase):
    def setUp(self):
        self.events = [
            dict(dataset="sharegpt", scheduled_arrival_s=0, first_token_s=5, ttft_slo_met=1),
            dict(dataset="sharegpt", scheduled_arrival_s=1, first_token_s=5, ttft_slo_met=1),
            dict(dataset="sharegpt", scheduled_arrival_s=2, first_token_s=200, ttft_slo_met=0),
            dict(dataset="leval", scheduled_arrival_s=61, first_token_s=1000, ttft_slo_met=1),
        ]

    def test_service_credit_not_arrival_or_completed_only_denominator(self):
        self.assertEqual(cumulative(self.events, "all", 4)["compliant_requests"], 0)
        at_five = cumulative(self.events, "all", 5)
        self.assertEqual(at_five["compliant_requests"], 2)
        self.assertEqual(at_five["pct_of_full_group"], 50)
        self.assertEqual(cumulative(self.events, "all", 200)["compliant_requests"], 2)
        self.assertEqual(cumulative(self.events, "all", 1000)["compliant_requests"], 3)

    def test_dataset_denominators_and_arrival_outcome(self):
        self.assertAlmostEqual(cumulative(self.events, "sharegpt", 1000)["pct_of_full_group"], 200 / 3)
        self.assertEqual(cumulative(self.events, "loong", 1000)["pct_of_full_group"], 100)
        bins = arrival_cohorts(self.events, "all", 30)
        self.assertEqual(bins[0]["requests"], 3)  # Includes later violation.
        self.assertAlmostEqual(bins[0]["attainment_pct"], 200 / 3)
        self.assertIsNone(bins[1]["attainment_pct"])
        self.assertEqual(bins[2]["attainment_pct"], 100)  # Outcome known at1000s.
        with self.assertRaises(ValueError):
            arrival_cohorts(self.events, "all", 0)

    def test_production_origin_and_reject_invalid_timestamp_or_qualification(self):
        row = dict(request_id="r", event_index="0", dataset="sharegpt", input_tokens="100",
                   submit_monotonic_ts="101", first_token_monotonic_ts="103",
                   effective_scheduled_offset_s="1", ttft_slo_ms="2000",
                   ttft_ms="2000", ttft_slo_met="1")
        event = normalize_requests([row, dict(submit_monotonic_ts="")], 100, 5)[0]
        self.assertEqual(event["first_token_s"], 3)
        for change in (dict(ttft_slo_met="0"), dict(first_token_monotonic_ts="99"),
                       dict(ttft_ms="1999"), dict(ttft_slo_ms="nan")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_requests([dict(row, **change)], 100, 5)
        with self.assertRaises(ValueError):
            normalize_requests([row, row], 100, 5)

    def test_input_order_is_not_completion_order(self):
        events = [
            dict(request_id="late", dataset="sharegpt", event_index=0,
                 submit_s=1, first_token_s=100, ttft_slo_met=0),
            dict(request_id="fast", dataset="sharegpt", event_index=2,
                 submit_s=3, first_token_s=4, ttft_slo_met=1),
            dict(request_id="loong", dataset="leval", event_index=1,
                 submit_s=2, first_token_s=3, ttft_slo_met=1),
        ]
        views = sharegpt_count_series(events)
        self.assertEqual([p["compliant_requests"] for p in views["trace_prefix"]], [0, 0, 1])
        self.assertEqual(views["actual_progress"][-1]["input_count"], 2)
        self.assertEqual(views["actual_progress"][-1]["compliant_requests"], 1)
        self.assertEqual(next(p for p in views["actual_progress"] if p["time_s"] == 3)["compliant_requests"], 0)

    def test_tied_first_tokens_after_last_submission_keep_vertical_growth(self):
        events = [dict(request_id=str(i), dataset="sharegpt", event_index=i,
                       submit_s=i + 1, first_token_s=10, ttft_slo_met=1) for i in range(2)]
        points = sharegpt_count_series(events)["actual_progress"]
        at_last_input = next(p for p in points if p["time_s"] == 2)
        self.assertEqual((at_last_input["input_count"], at_last_input["compliant_requests"]), (2, 0))
        self.assertEqual((points[-1]["input_count"], points[-1]["compliant_requests"]), (2, 2))
        with self.assertRaises(ValueError):
            sharegpt_count_series([])


if __name__ == "__main__":
    unittest.main()
