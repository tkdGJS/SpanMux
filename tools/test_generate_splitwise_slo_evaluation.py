#!/usr/bin/env python3
"""CPU-only tests for Splitwise-aligned SLO evaluation."""

from __future__ import annotations

import unittest
import csv
import tempfile
from pathlib import Path

from tools.generate_splitwise_slo_evaluation import (
    build_evaluation_summary,
    build_partition_rows,
    build_reference_rows,
    load_precomputed_reference_rows,
    native_reference_row_y_limits,
)


class SplitwiseSloEvaluationTest(unittest.TestCase):
    def test_reference_uses_pooled_token_gap_quantiles(self) -> None:
        rows = build_reference_rows(
            {25_000: [10.0, 20.0, 30.0, 40.0, 50.0]},
            {25_000: [1.0, 2.0, 3.0, 4.0, 5.0]},
            {25_000: [100.0, 200.0, 300.0, 400.0, 500.0]},
        )
        by_quantile = {row["quantile"]: row for row in rows}
        self.assertEqual(by_quantile["P50"]["native_ttft_ms"], 30.0)
        self.assertEqual(
            by_quantile["P50"]["splitwise_ttft_slo_ms"], 60.0
        )
        self.assertEqual(
            by_quantile["P90"][
                "native_pooled_token_gap_tbt_ms"
            ],
            4.6,
        )
        self.assertEqual(
            by_quantile["P90"][
                "splitwise_pooled_token_gap_tbt_slo_ms"
            ],
            6.9,
        )
        self.assertEqual(
            by_quantile["P99"][
                "splitwise_pooled_token_gap_tbt_slo_ms"
            ],
            24.8,
        )
        self.assertEqual(by_quantile["P50"]["native_e2e_ms"], 300.0)
        self.assertEqual(
            by_quantile["P50"]["splitwise_e2e_slo_ms"], 375.0
        )

    def test_partition_slowdown_and_summary(self) -> None:
        references = build_reference_rows(
            {25_000: [10.0]},
            {25_000: [2.0]},
            {25_000: [100.0]},
        )
        key = ("d8_p32", 8, 32, "native", 25_000, 1)
        rows = build_partition_rows(
            references,
            {key: {"ttft": [30.0], "tbt": [3.0]}},
        )
        by_quantile = {row["quantile"]: row for row in rows}
        self.assertEqual(
            by_quantile["P50"]["ttft_normalized_slowdown"], 3.0
        )
        self.assertEqual(by_quantile["P50"]["ttft_slo_met"], 0)
        self.assertEqual(
            by_quantile["P50"]["tbt_normalized_slowdown"], 1.5
        )
        self.assertEqual(by_quantile["P50"]["tbt_slo_met"], 0)
        self.assertEqual(by_quantile["P99"]["ttft_slo_met"], 1)
        self.assertEqual(by_quantile["P99"]["tbt_slo_met"], 1)

        summary = build_evaluation_summary(
            rows,
            completed_cases=1,
            expected_cases=10,
        )
        p50_ttft = next(
            row
            for row in summary
            if row["metric"] == "TTFT" and row["quantile"] == "P50"
        )
        self.assertEqual(p50_ttft["state_slo_attainment"], 0.0)
        self.assertEqual(
            p50_ttft["coverage_status"], "incomplete:1/10"
        )

    def test_native_reference_uses_one_limit_per_metric_row(self) -> None:
        rows = build_reference_rows(
            {
                25_000: [10.0],
                100_000: [50.0],
            },
            {
                25_000: [1.0],
                100_000: [5.0],
            },
            {
                25_000: [100.0],
                100_000: [500.0],
            },
        )
        ttft_limit_s, tbt_limit_ms, e2e_limit_s = native_reference_row_y_limits(
            rows,
            headroom=1.08,
        )
        # The global 100K P99 SLO sets every TTFT panel to 0..0.324 s
        # and every TBT panel to 0..27 ms.
        self.assertAlmostEqual(ttft_limit_s, 0.324)
        self.assertAlmostEqual(tbt_limit_ms, 27.0)
        self.assertAlmostEqual(e2e_limit_s, 2.7)

    def test_loads_audited_one_token_csv(self) -> None:
        fields = [
            "context_tokens",
            "quantile",
            "measurement_request_count",
            "one_token_tbt_gap_count",
            "provenance",
            "profiling_ttft_ms",
            "ttft_slo_factor",
            "ttft_slo_ms",
            "profiling_one_token_tbt_ms",
            "tbt_slo_factor",
            "tbt_slo_ms",
            "profiling_e2e_ms",
            "e2e_slo_factor",
            "e2e_slo_ms",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for quantile, factor in (("P50", 1.25), ("P90", 1.5), ("P99", 5.0)):
                    writer.writerow(
                        {
                            "context_tokens": 100_000,
                            "quantile": quantile,
                            "measurement_request_count": 3,
                            "one_token_tbt_gap_count": 12_285,
                            "provenance": "legacy_archive_bos_reconstruction",
                            "profiling_ttft_ms": 10.0,
                            "ttft_slo_factor": 2.0,
                            "ttft_slo_ms": 20.0,
                            "profiling_one_token_tbt_ms": 100.0,
                            "tbt_slo_factor": factor,
                            "tbt_slo_ms": 100.0 * factor,
                            "profiling_e2e_ms": 1_000.0,
                            "e2e_slo_factor": factor,
                            "e2e_slo_ms": 1_000.0 * factor,
                        }
                    )
            rows = load_precomputed_reference_rows(path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["tbt_token_gap_count"], 12_285)
            self.assertEqual(rows[0]["native_e2e_ms"], 1_000.0)


if __name__ == "__main__":
    unittest.main()
