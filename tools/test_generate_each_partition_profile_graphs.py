#!/usr/bin/env python3
"""CPU-only tests for profiling-only partition graph inputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_each_partition_profile_graphs import (
    build_coverage_rows,
    build_mean_rows,
    build_profile_rows,
    collect_measureable_partial_cases,
)


class EachPartitionProfileGraphsTest(unittest.TestCase):
    def test_profile_rows_contain_measurements_without_slo(self) -> None:
        key = ("d8_p32", 8, 32, "native", 25_000, 1)
        rows = build_profile_rows(
            {
                key: {
                    "ttft": [10.0, 20.0, 30.0, 40.0, 50.0],
                    "tbt": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            }
        )
        self.assertEqual([row["quantile"] for row in rows], [
            "P50",
            "P90",
            "P99",
        ])
        self.assertEqual(rows[0]["profile_ttft_ms"], 30.0)
        self.assertEqual(rows[1]["profile_overlap_tbt_ms"], 4.6)
        self.assertEqual(rows[2]["profile_overlap_tbt_ms"], 4.96)
        self.assertFalse(
            any("slo" in field.lower() for field in rows[0])
        )

    def test_repeat_mean_weights_each_repeat_once(self) -> None:
        base = ("d8_p32", 8, 32, "native", 25_000, 1)
        rows = build_mean_rows(
            {
                (*base, 1): {"ttft": [10.0], "tbt": [1.0, 1.0]},
                (*base, 2): {"ttft": [20.0], "tbt": [9.0]},
                (*base, 3): {"ttft": [30.0], "tbt": [5.0, 5.0, 5.0]},
            }
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ttft_repeat_count"], 3)
        self.assertEqual(rows[0]["tbt_repeat_count"], 3)
        self.assertEqual(rows[0]["mean_profile_ttft_ms"], 20.0)
        self.assertEqual(rows[0]["mean_profile_overlap_tbt_ms"], 5.0)

    def test_coverage_keeps_missing_cases_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            completed = root / "cases" / "native" / "d8_p32"
            completed.mkdir(parents=True)
            (completed / "run_metadata.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            empty = root / "cases" / "ring" / "d8_p32"
            empty.mkdir(parents=True)
            rows = build_coverage_rows(
                root,
                ("d8_p32",),
                ("native", "ring"),
            )
        by_mode = {row["prefill_mode"]: row for row in rows}
        self.assertEqual(by_mode["native"]["status"], "completed")
        self.assertEqual(
            by_mode["ring"]["status"],
            "directory_without_metadata",
        )

    def test_partial_case_requires_successful_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            successful = root / "running"
            failed_only = root / "failed"
            for case_dir, trial_status in (
                (successful, "success"),
                (failed_only, "failed"),
            ):
                case_dir.mkdir()
                (case_dir / "run_metadata.json").write_text(
                    json.dumps({"status": case_dir.name}),
                    encoding="utf-8",
                )
                (case_dir / "trial_summary.csv").write_text(
                    "phase,status\nmeasurement," + trial_status + "\n",
                    encoding="utf-8",
                )
                for name in (
                    "client_token_timeline.csv",
                    "iteration_metrics.jsonl",
                ):
                    (case_dir / name).write_text("", encoding="utf-8")
            selected = collect_measureable_partial_cases(
                [successful, failed_only]
            )
        self.assertEqual(selected, [successful])


if __name__ == "__main__":
    unittest.main()
