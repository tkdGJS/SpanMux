#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for vendor_v8 SLO-guarded prefill scheduling."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(os.environ.get("VLLM_VENDOR_DIR", ROOT / "vendor_v8")).resolve()
MODULE_PATH = VENDOR / "vllm/v1/core/sched/prefill_chunk_scheduler.py"
SPEC = importlib.util.spec_from_file_location(
    "vendor_v8_prefill_chunk_scheduler", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PrefillChunkScheduler = MODULE.PrefillChunkScheduler
PrefillChunkServiceProfile = MODULE.PrefillChunkServiceProfile


@dataclass
class FakeRequest:
    request_id: str
    arrival_time: float
    num_prompt_tokens: int
    num_computed_tokens: int = 0
    num_tokens: int = 0
    pd_recompute_frontier_tokens: int = 0
    virtual_context_deferred_logical_computed_tokens: int = 0
    ttft_prefill_deadline_ms: float | None = None


class SLOGuardedSRPTTest(unittest.TestCase):
    @staticmethod
    def scheduler(**overrides) -> PrefillChunkScheduler:
        values = {
            "quantum_tokens": 100,
            "estimated_ms_per_token": 1.0,
            "slo_guard_ms": 100.0,
            "starvation_threshold_ms": 5000.0,
            "starvation_max_bypasses": 3,
        }
        values.update(overrides)
        return PrefillChunkScheduler("slo_srpt", **values)

    def test_safe_tier_uses_estimated_remaining_service(self) -> None:
        scheduler = self.scheduler(starvation_threshold_ms=0.0)
        order = scheduler.begin_iteration(
            [
                FakeRequest("long", 99.5, 1000),
                FakeRequest("short", 99.6, 100),
            ],
            now=100.0,
        )
        self.assertEqual(order, ("short", "long"))
        self.assertEqual(
            scheduler.request_metrics(
                FakeRequest("short", 99.6, 100)
            )["ttft_prefill_selection_tier"],
            "srpt_safe",
        )

    def test_decode_reset_history_remains_prefill_schedulable(self) -> None:
        scheduler = self.scheduler(
            quantum_tokens=100,
            starvation_threshold_ms=0.0,
        )
        replay = FakeRequest(
            "decode-replay",
            99.0,
            500,
            num_computed_tokens=500,
            num_tokens=575,
            pd_recompute_frontier_tokens=574,
        )

        self.assertEqual(scheduler.remaining_prefill_tokens(replay), 74)
        self.assertEqual(
            scheduler.begin_iteration([replay], now=100.0),
            ("decode-replay",),
        )
        self.assertEqual(scheduler.cap_prefill_tokens(replay, 100), 74)
        replay.num_computed_tokens = 574
        self.assertEqual(scheduler.remaining_prefill_tokens(replay), 0)

    def test_critical_laxity_preempts_safe_short_request(self) -> None:
        scheduler = self.scheduler(starvation_threshold_ms=0.0)
        critical = FakeRequest(
            "critical-long",
            99.0,
            1000,
            ttft_prefill_deadline_ms=2150.0,
        )
        safe_short = FakeRequest("safe-short", 99.5, 10)
        order = scheduler.begin_iteration(
            [safe_short, critical],
            now=100.0,
        )
        self.assertEqual(order[0], "critical-long")
        metrics = scheduler.request_metrics(critical)
        self.assertEqual(metrics["ttft_prefill_slo_critical"], 1)
        self.assertEqual(
            metrics["ttft_prefill_selection_tier"], "slo_critical"
        )

    def test_service_age_prevents_long_request_starvation(self) -> None:
        scheduler = self.scheduler(starvation_max_bypasses=0)
        long_waiter = FakeRequest("long-waiter", 90.0, 1000)
        new_short = FakeRequest("new-short", 99.5, 10)
        order = scheduler.begin_iteration(
            [new_short, long_waiter],
            now=100.0,
        )
        self.assertEqual(order[0], "long-waiter")
        self.assertEqual(
            scheduler.request_metrics(long_waiter)[
                "ttft_prefill_selection_tier"
            ],
            "starved",
        )

    def test_projected_peer_chunk_prevents_age_threshold_overshoot(
        self,
    ) -> None:
        scheduler = self.scheduler(
            starvation_threshold_ms=100.0,
            starvation_max_bypasses=0,
        )
        long_waiter = FakeRequest("long", 99.94, 1000)
        short = FakeRequest("short", 99.99, 50)

        order = scheduler.begin_iteration(
            [long_waiter, short],
            now=100.0,
        )

        self.assertEqual(order[0], "long")
        metrics = scheduler.request_metrics(long_waiter)
        self.assertEqual(metrics["ttft_prefill_starved_by_projected_age"], 1)
        self.assertEqual(metrics["ttft_prefill_projected_peer_wait_ms"], 50.0)

    def test_bypass_guard_counts_only_successful_peer_chunks(self) -> None:
        scheduler = self.scheduler(
            starvation_threshold_ms=0.0,
            starvation_max_bypasses=2,
        )
        long_waiter = FakeRequest("long", 1.0, 1000)
        short = FakeRequest("short", 2.0, 10)

        scheduler.begin_iteration([long_waiter, short], now=3.0)
        scheduler.observe_scheduled_chunks([])
        self.assertEqual(
            scheduler.request_metrics(long_waiter)[
                "ttft_prefill_starvation_bypass_count"
            ],
            0,
        )

        scheduler.observe_scheduled_chunks(["short"])
        scheduler.begin_iteration([long_waiter, short], now=4.0)
        scheduler.observe_scheduled_chunks(["short"])
        order = scheduler.begin_iteration(
            [long_waiter, short],
            now=5.0,
        )
        self.assertEqual(order[0], "long")
        self.assertEqual(
            scheduler.request_metrics(long_waiter)[
                "ttft_prefill_starvation_bypass_count"
            ],
            2,
        )

    def test_durable_virtual_progress_is_used(self) -> None:
        scheduler = self.scheduler(starvation_threshold_ms=0.0)
        offloaded = FakeRequest(
            "offloaded",
            2.0,
            1000,
            num_computed_tokens=0,
            virtual_context_deferred_logical_computed_tokens=990,
        )
        resident = FakeRequest("resident", 1.0, 1000, 900)
        self.assertEqual(
            scheduler.begin_iteration(
                [resident, offloaded],
                now=2.1,
            )[0],
            "offloaded",
        )

    def test_infeasible_current_set_switches_to_edf(self) -> None:
        scheduler = self.scheduler(
            quantum_tokens=50,
            slo_guard_ms=0.0,
            starvation_threshold_ms=0.0,
        )
        critical_later_deadline = FakeRequest(
            "later",
            0.0,
            250,
            ttft_prefill_deadline_ms=290.0,
        )
        earlier_deadline = FakeRequest(
            "earlier",
            0.0,
            100,
            ttft_prefill_deadline_ms=200.0,
        )

        order = scheduler.begin_iteration(
            [critical_later_deadline, earlier_deadline],
            now=0.0,
        )

        self.assertEqual(order, ("earlier", "later"))
        metrics = scheduler.request_metrics(earlier_deadline)
        self.assertEqual(metrics["ttft_prefill_overload_edf_active"], 1)
        self.assertEqual(
            metrics["ttft_prefill_edf_predicted_violation_count"],
            1,
        )
        self.assertEqual(
            metrics["ttft_prefill_selection_tier"],
            "overload_edf",
        )

    def test_pending_prefill_chunk_is_charged_before_edf_completion(
        self,
    ) -> None:
        scheduler = self.scheduler(
            quantum_tokens=50,
            slo_guard_ms=0.0,
            starvation_threshold_ms=0.0,
        )
        pending = FakeRequest("pending", 0.0, 100)
        scheduler.begin_iteration([pending], now=0.0)
        scheduler.observe_scheduled_chunks(["pending"])
        candidate = FakeRequest(
            "candidate",
            0.0,
            100,
            ttft_prefill_deadline_ms=120.0,
        )

        scheduler.begin_iteration(
            [candidate],
            now=0.0,
            pending_request_ids=["pending"],
        )

        metrics = scheduler.request_metrics(candidate)
        self.assertEqual(metrics["ttft_prefill_pending_service_ms"], 50.0)
        self.assertEqual(metrics["ttft_prefill_overload_edf_active"], 1)

    def test_profile_sums_future_chunks_and_changes_srpt_order(self) -> None:
        payload = {
            "schema_version": 1,
            "points": [
                {
                    "profile_id": "d20_p20",
                    "prefill_mode": "native",
                    "max_context_tokens": 100,
                    "max_chunk_tokens": 100,
                    "p95_prefill_chunk_ms": 10.0,
                },
                {
                    "profile_id": "d20_p20",
                    "prefill_mode": "native",
                    "max_context_tokens": 200,
                    "max_chunk_tokens": 100,
                    "p95_prefill_chunk_ms": 20.0,
                },
                {
                    "profile_id": "d20_p20",
                    "prefill_mode": "native",
                    "max_context_tokens": 1000,
                    "max_chunk_tokens": 100,
                    "p95_prefill_chunk_ms": 200.0,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "prefill_profile.json"
            profile_path.write_text(json.dumps(payload), encoding="utf-8")
            scheduler = self.scheduler(
                service_profile_path=str(profile_path),
                starvation_threshold_ms=0.0,
            )
            costly_short = FakeRequest(
                "costly-short",
                1.0,
                1000,
                num_computed_tokens=900,
            )
            cheap_longer = FakeRequest("cheap-longer", 2.0, 200)

            order = scheduler.begin_iteration(
                [costly_short, cheap_longer],
                now=2.1,
            )

        self.assertEqual(order[0], "cheap-longer")
        metrics = scheduler.request_metrics(cheap_longer)
        self.assertEqual(
            metrics["ttft_prefill_estimated_remaining_ms"],
            30.0,
        )
        self.assertEqual(
            metrics["ttft_prefill_estimated_remaining_chunks"],
            2,
        )
        self.assertEqual(
            metrics["ttft_prefill_estimate_source"],
            "prefill_chunk_profile_p95_sum",
        )
        self.assertEqual(
            metrics["ttft_prefill_service_profile_covered"],
            1,
        )

    def test_profile_uses_worst_covered_execution_variant(self) -> None:
        profile = PrefillChunkServiceProfile.from_dict(
            {
                "schema_version": 1,
                "points": [
                    {
                        "profile_id": "d20_p20",
                        "prefill_mode": "native",
                        "max_context_tokens": 100,
                        "max_chunk_tokens": 100,
                        "p95_prefill_chunk_ms": 10.0,
                    },
                    {
                        "profile_id": "d20_p20",
                        "prefill_mode": "ring",
                        "max_context_tokens": 100,
                        "max_chunk_tokens": 100,
                        "p95_prefill_chunk_ms": 15.0,
                    },
                    {
                        "profile_id": "d20_p20",
                        "prefill_mode": "native",
                        "max_context_tokens": 200,
                        "max_chunk_tokens": 100,
                        "p95_prefill_chunk_ms": 20.0,
                    },
                    {
                        "profile_id": "d20_p20",
                        "prefill_mode": "ring",
                        "max_context_tokens": 200,
                        "max_chunk_tokens": 100,
                        "p95_prefill_chunk_ms": 30.0,
                    },
                ],
            }
        )

        prediction = profile.estimate_remaining(
            logical_progress_tokens=0,
            prompt_tokens=200,
            quantum_tokens=100,
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.estimated_next_chunk_ms, 15.0)
        self.assertEqual(prediction.estimated_remaining_ms, 45.0)
        self.assertEqual(prediction.covered_variant_count, 2)

    def test_uncovered_profile_uses_observable_scalar_fallback(self) -> None:
        payload = {
            "schema_version": 1,
            "points": [
                {
                    "max_context_tokens": 100,
                    "max_chunk_tokens": 100,
                    "p95_prefill_chunk_ms": 10.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "prefill_profile.json"
            profile_path.write_text(json.dumps(payload), encoding="utf-8")
            scheduler = self.scheduler(
                service_profile_path=str(profile_path),
                starvation_threshold_ms=0.0,
            )
            request = FakeRequest("uncovered", 1.0, 200)
            scheduler.begin_iteration([request], now=1.1)

        metrics = scheduler.request_metrics(request)
        self.assertEqual(
            metrics["ttft_prefill_estimated_remaining_ms"],
            200.0,
        )
        self.assertEqual(
            metrics["ttft_prefill_estimate_source"],
            "configured_ms_per_token_profile_fallback",
        )
        self.assertEqual(
            metrics["ttft_prefill_service_profile_fallback"],
            1,
        )
        self.assertEqual(
            metrics[
                "ttft_prefill_service_profile_uncovered_context_tokens"
            ],
            200,
        )
        self.assertEqual(
            metrics[
                "ttft_prefill_service_profile_uncovered_chunk_tokens"
            ],
            100,
        )


class VendorV8PrefillContractTest(unittest.TestCase):
    def test_default_off_cli_and_request_policy_wiring(self) -> None:
        config = (VENDOR / "vllm/config/scheduler.py").read_text()
        args = (VENDOR / "vllm/engine/arg_utils.py").read_text()
        scheduler = (VENDOR / "vllm/v1/core/sched/scheduler.py").read_text()
        runner = (
            ROOT / "run_sharegpt_randomized_mixed_context_10req_4GB.sh"
        ).read_text()
        self.assertIn(
            "enable_ttft_prefill_chunk_scheduling: bool = False", config
        )
        self.assertIn(
            "ttft_prefill_native_chunk_max_tokens: int = Field(default=0",
            config,
        )
        self.assertIn('"slo_srpt"', config)
        for flag in (
            "--ttft-prefill-native-chunk-max-tokens",
            "--ttft-prefill-estimated-ms-per-token",
            "--ttft-prefill-service-profile-path",
            "--ttft-prefill-slo-guard-ms",
            "--ttft-prefill-starvation-threshold-ms",
            "--ttft-prefill-starvation-max-bypasses",
        ):
            self.assertIn(f'"{flag}"', args)
        self.assertIn("estimated_ms_per_token=(", scheduler)
        self.assertIn("service_profile_path=(", scheduler)
        self.assertIn("starvation_max_bypasses=(", scheduler)
        self.assertIn("def _ttft_prefill_execution_mode(", scheduler)
        self.assertIn('execution_mode="native"', scheduler)
        self.assertIn("slo_srpt)", runner)
        self.assertIn(
            "--ttft-prefill-service-profile-path",
            runner,
        )
        self.assertIn(
            "TTFT_PREFILL_ESTIMATED_MS_PER_TOKEN",
            runner,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
