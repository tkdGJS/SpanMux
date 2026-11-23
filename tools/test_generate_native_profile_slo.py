#!/usr/bin/env python3
"""CPU-only tests for native profile and SLO generation."""

from __future__ import annotations

import unittest

from generate_native_profile_slo import build_profile, build_slo_rows


class GenerateNativeProfileSLOTest(unittest.TestCase):
    def test_warmup_is_excluded_and_pd_scale_is_applied(self) -> None:
        raw_rows = [
            {
                "context_target_tokens": "25000",
                "phase": "warmup",
                "status": "success",
                "ttft_ms": "999",
                "mean_tbt_ms": "999",
                "p99_tbt_ms": "999",
                "completion_latency_ms": "999",
                "output_tokens": "4",
            },
            {
                "context_target_tokens": "25000",
                "phase": "measurement",
                "status": "success",
                "ttft_ms": "100",
                "mean_tbt_ms": "10",
                "p99_tbt_ms": "12",
                "completion_latency_ms": "200",
                "output_tokens": "4",
            },
            {
                "context_target_tokens": "25000",
                "phase": "measurement",
                "status": "success",
                "ttft_ms": "120",
                "mean_tbt_ms": "14",
                "p99_tbt_ms": "16",
                "completion_latency_ms": "240",
                "output_tokens": "4",
            },
        ]
        summary_rows = [
            {
                "context_target_tokens": "25000",
                "pooled_p99_tbt_ms": "15",
            }
        ]
        profile = build_profile(raw_rows, summary_rows)
        self.assertEqual(profile[0]["ttft_mean_ms"], 110.0)
        self.assertEqual(profile[0]["ttft_max_ms"], 120.0)
        self.assertEqual(profile[0]["request_p99_tbt_max_ms"], 16.0)
        slo = build_slo_rows(
            profile,
            total_sms=40,
            prefill_sms=20,
            decode_sms=20,
            headroom=1.25,
            attainment_target=0.90,
        )
        self.assertEqual(slo[0]["ttft_slo_scale"], 2.5)
        self.assertEqual(slo[0]["ttft_slo_ms"], 300.0)
        self.assertEqual(slo[0]["tbt_slo_ms"], 40.0)


if __name__ == "__main__":
    unittest.main()
