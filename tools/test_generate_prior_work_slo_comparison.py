#!/usr/bin/env python3
"""CPU-only tests for prior-work SLO semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_prior_work_slo_comparison import (
    build_distserve_rows,
    build_llumnix_rows,
    build_splitwise_rows,
)


class PriorWorkSloComparisonTest(unittest.TestCase):
    def test_splitwise_applies_quantile_specific_factors(self) -> None:
        samples = {
            25_000: [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        rows = build_splitwise_rows(
            ttft=samples,
            request_mean_tbt=samples,
            pooled_token_tbt=samples,
            e2e=samples,
        )
        by_quantile = {row["quantile"]: row for row in rows}

        self.assertEqual(by_quantile["P50"]["profiling_ttft_ms"], 30.0)
        self.assertEqual(by_quantile["P50"]["splitwise_ttft_slo_ms"], 60.0)
        self.assertEqual(
            by_quantile["P50"]["splitwise_request_mean_tbt_slo_ms"],
            37.5,
        )
        self.assertEqual(by_quantile["P50"]["splitwise_e2e_slo_ms"], 37.5)

        self.assertEqual(by_quantile["P90"]["profiling_ttft_ms"], 46.0)
        self.assertEqual(by_quantile["P90"]["splitwise_ttft_slo_ms"], 138.0)
        self.assertEqual(
            by_quantile["P90"]["splitwise_request_mean_tbt_slo_ms"],
            69.0,
        )

        self.assertEqual(by_quantile["P99"]["profiling_ttft_ms"], 49.6)
        self.assertEqual(by_quantile["P99"]["splitwise_ttft_slo_ms"], 297.6)
        self.assertEqual(
            by_quantile["P99"]["splitwise_request_mean_tbt_slo_ms"],
            248.0,
        )

    def test_distserve_scales_reported_targets(self) -> None:
        rows = build_distserve_rows((1.0, 2.0))
        self.assertEqual(len(rows), 10)
        opt13_scale2 = next(
            row
            for row in rows
            if row["model"] == "OPT-13B" and row["slo_scale"] == 2.0
        )
        self.assertEqual(opt13_scale2["scaled_ttft_slo_ms"], 500.0)
        self.assertEqual(opt13_scale2["scaled_tpot_slo_ms"], 200.0)
        self.assertEqual(opt13_scale2["local_t4_recomputed"], 0)

    def test_llumnix_threshold_is_not_latency_multiplier(self) -> None:
        row = build_llumnix_rows()[0]
        self.assertEqual(row["ttft_slo_ms"], 6000.0)
        self.assertEqual(row["tpot_slo_ms"], 50.0)
        self.assertEqual(row["ttft_dispatch_threshold"], 0.9)
        self.assertEqual(row["tpot_dispatch_threshold"], 0.9)
        self.assertNotIn("ttft_dispatch_budget_ms", row)
        self.assertNotIn("tpot_dispatch_budget_ms", row)
        self.assertEqual(row["local_t4_recomputed"], 0)


if __name__ == "__main__":
    unittest.main()
