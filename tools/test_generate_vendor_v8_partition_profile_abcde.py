#!/usr/bin/env python3
"""CPU-only tests for representative partition-profile A-E helpers."""

from __future__ import annotations

import unittest

from tools.generate_vendor_v8_partition_profile_abcde import (
    future_step_series,
    phase_breakdown,
    select_representative_trial,
)


def trial(
    context: int,
    decode_count: int,
    repeat: int,
    ttft_ms: float,
    *,
    status: str = "success",
) -> dict[str, str]:
    return {
        "case_id": "case",
        "profile_id": "d8_p32",
        "prefill_mode": "native",
        "phase": "measurement",
        "status": status,
        "context_tokens": str(context),
        "active_decode_requests": str(decode_count),
        "repeat_index": str(repeat),
        "probe_ttft_ms": str(ttft_ms),
        "probe_submit_monotonic_ts": "10",
        "probe_first_token_monotonic_ts": "20",
    }


class PartitionProfileAbcdeTest(unittest.TestCase):
    def test_selects_largest_state_and_median_repeat(self) -> None:
        selected = select_representative_trial(
            [
                trial(25_000, 3, 1, 10),
                trial(100_000, 2, 1, 40),
                trial(100_000, 3, 1, 50),
                trial(100_000, 3, 2, 30),
                trial(100_000, 3, 3, 40),
                trial(100_000, 3, 4, 45, status="failed"),
            ],
            case_status="failed",
        )
        self.assertEqual(selected.context_tokens, 100_000)
        self.assertEqual(selected.active_decode_requests, 3)
        self.assertEqual(selected.repeat_index, 3)
        self.assertEqual(selected.case_status, "failed")

    def test_future_series_clips_to_probe_window(self) -> None:
        xs, ys = future_step_series(
            [(5.0, 12.0), (11.0, 13.0), (19.0, 25.0)],
            start=10.0,
            end=20.0,
        )
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(xs[-1], 10.0)
        self.assertEqual(max(ys), 2)
        self.assertEqual(ys[-1], 0)

    def test_phase_breakdown_uses_non_overlapping_worker_components(self) -> None:
        values = phase_breakdown(
            [
                {
                    "scheduled_phase": "prefill",
                    "scheduled_ts": 1.0,
                    "worker_execute_start_ts": 2.0,
                    "model_done_ts": 6.0,
                    "output_process_end_ts": 7.0,
                    "worker_execute_duration_s": 3.0,
                    "worker_model_forward_duration_s": 2.0,
                }
            ]
        )["prefill"]
        self.assertEqual(values["submit_gap"], 1.0)
        self.assertEqual(values["worker_forward"], 2.0)
        self.assertEqual(values["worker_other"], 1.0)
        self.assertEqual(values["async_queue_wait"], 1.0)
        self.assertEqual(values["output_gap"], 1.0)


if __name__ == "__main__":
    unittest.main()
