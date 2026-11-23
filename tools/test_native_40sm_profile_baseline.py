#!/usr/bin/env python3
"""CPU-only unit tests for the sequential native profile driver."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_native_40sm_profile_baseline import (
    PreparedRequest,
    build_trial_plan,
    summarize_bucket,
)


def prepared(tokens: int) -> PreparedRequest:
    return PreparedRequest(
        workload="sharegpt",
        prompt="test",
        input_tokens=tokens,
        max_tokens=4,
        min_tokens=4,
        source_id=f"source-{tokens}",
    )


class NativeProfileBaselineTest(unittest.TestCase):
    def test_trial_plan_is_closed_loop_bucket_order(self) -> None:
        plan = build_trial_plan(
            [prepared(25_000), prepared(50_000)],
            warmup_count=1,
            repeat_count=2,
        )
        self.assertEqual(
            [
                (trial.request.input_tokens, trial.phase, trial.repeat_index)
                for trial in plan
            ],
            [
                (25_000, "warmup", 1),
                (25_000, "measurement", 1),
                (25_000, "measurement", 2),
                (50_000, "warmup", 1),
                (50_000, "measurement", 1),
                (50_000, "measurement", 2),
            ],
        )

    def test_summary_excludes_warmup_from_means(self) -> None:
        rows = [
            {
                "phase": "warmup",
                "status": "success",
                "ttft_ms": "999",
                "mean_tbt_ms": "999",
                "completion_latency_ms": "999",
                "output_tokens": "4",
            },
            {
                "phase": "measurement",
                "status": "success",
                "ttft_ms": "10",
                "mean_tbt_ms": "2",
                "completion_latency_ms": "20",
                "output_tokens": "4",
            },
            {
                "phase": "measurement",
                "status": "success",
                "ttft_ms": "14",
                "mean_tbt_ms": "4",
                "completion_latency_ms": "28",
                "output_tokens": "4",
            },
        ]
        summary = summarize_bucket(
            case_id="test",
            bucket_index=0,
            target_tokens=25_000,
            actual_input_tokens=25_000,
            warmup_count=1,
            measured_repeat_count=2,
            request_max_tokens=4,
            request_min_tokens=4,
            cooldown_sec=0,
            run_rows=rows,
            pooled_token_gaps=[1.0, 2.0, 3.0],
        )
        self.assertEqual(summary["mean_ttft_ms"], "12.0")
        self.assertEqual(summary["mean_request_tbt_ms"], "3.0")
        self.assertEqual(summary["mean_completion_latency_ms"], "24.0")
        self.assertEqual(summary["warmup_successes"], 1)
        self.assertEqual(summary["measurement_successes"], 2)


if __name__ == "__main__":
    unittest.main()
