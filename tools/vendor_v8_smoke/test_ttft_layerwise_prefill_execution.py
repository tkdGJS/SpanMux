#!/usr/bin/env python3
"""CPU-only contracts for the default-off TTFT Prefill layer-group scope."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "vendor_v8/vllm/model_executor/layers/ttft_layerwise_prefill.py"
)
MODULE_NAME = "vendor_v8_ttft_layerwise_prefill_test_target"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)


ENABLE_ENV = "VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION"
PREEMPTION_ENV = "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION"
GROUP_ENV = "VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS"


class TTFTLayerwisePrefillExecutionTest(unittest.TestCase):
    def _clean_environment(self) -> dict[str, str]:
        return {
            key: value
            for key, value in os.environ.items()
            if key not in {ENABLE_ENV, PREEMPTION_ENV, GROUP_ENV}
        }

    def test_default_off_keeps_legacy_execution(self) -> None:
        with patch.dict(os.environ, self._clean_environment(), clear=True):
            with MODULE.ttft_layerwise_prefill_execution(
                "prefill",
                scheduler_iteration_id=7,
                num_tokens=2048,
            ) as execution:
                self.assertFalse(execution.requested)
                self.assertFalse(execution.active)
                self.assertIsNone(
                    MODULE.active_ttft_layerwise_prefill_execution()
                )
                debug = execution.as_debug_dict()
                self.assertEqual(
                    debug["ttft_layerwise_prefill_execution_mode"],
                    "legacy",
                )
                self.assertEqual(
                    debug["ttft_layerwise_prefill_scheduler_yield_count"],
                    0.0,
                )

    def test_enabled_prefill_records_contiguous_groups(self) -> None:
        environment = self._clean_environment()
        environment.update({ENABLE_ENV: "1", GROUP_ENV: "2"})
        with patch.dict(os.environ, environment, clear=True):
            with MODULE.ttft_layerwise_prefill_execution(
                "prefill",
                scheduler_iteration_id=11,
                phase_submit_seq=19,
                num_tokens=4096,
                num_requests=2,
            ) as execution:
                self.assertIs(
                    MODULE.active_ttft_layerwise_prefill_execution(),
                    execution,
                )
                execution.record_group(
                    start_layer=4,
                    end_layer=6,
                    host_wall_s=0.003,
                )
                execution.record_group(
                    start_layer=6,
                    end_layer=7,
                    host_wall_s=0.002,
                )
            self.assertIsNone(
                MODULE.active_ttft_layerwise_prefill_execution()
            )
        debug = execution.as_debug_dict()
        self.assertEqual(
            debug["ttft_layerwise_prefill_execution_mode"],
            "same_future_layer_groups",
        )
        self.assertEqual(debug["ttft_layerwise_prefill_group_count"], 2.0)
        self.assertEqual(debug["ttft_layerwise_prefill_executed_layers"], 3.0)
        self.assertEqual(debug["ttft_layerwise_prefill_first_layer"], 4.0)
        self.assertEqual(
            debug["ttft_layerwise_prefill_last_layer_exclusive"],
            7.0,
        )
        self.assertAlmostEqual(
            debug["ttft_layerwise_prefill_group_host_wall_ms_total"],
            5.0,
        )
        self.assertEqual(
            debug["ttft_layerwise_prefill_scheduler_yield_count"],
            0.0,
        )

    def test_enabled_decode_does_not_install_prefill_scope(self) -> None:
        environment = self._clean_environment()
        environment.update({ENABLE_ENV: "1", GROUP_ENV: "2"})
        with patch.dict(os.environ, environment, clear=True):
            with MODULE.ttft_layerwise_prefill_execution(
                "decode"
            ) as execution:
                self.assertTrue(execution.requested)
                self.assertFalse(execution.active)
                self.assertIsNone(
                    MODULE.active_ttft_layerwise_prefill_execution()
                )

    def test_invalid_group_width_fails_closed(self) -> None:
        for value in ("0", "bad"):
            with self.subTest(value=value):
                environment = self._clean_environment()
                environment.update({ENABLE_ENV: "1", GROUP_ENV: value})
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(
                        MODULE.TTFTLayerwisePrefillError
                    ):
                        with MODULE.ttft_layerwise_prefill_execution(
                            "prefill"
                        ):
                            pass

    def test_noncontiguous_group_is_rejected_and_scope_is_reset(self) -> None:
        environment = self._clean_environment()
        environment.update({ENABLE_ENV: "1", GROUP_ENV: "2"})
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(MODULE.TTFTLayerwisePrefillError):
                with MODULE.ttft_layerwise_prefill_execution(
                    "prefill"
                ) as execution:
                    execution.record_group(
                        start_layer=0,
                        end_layer=2,
                        host_wall_s=0.0,
                    )
                    execution.record_group(
                        start_layer=3,
                        end_layer=4,
                        host_wall_s=0.0,
                    )
            self.assertIsNone(
                MODULE.active_ttft_layerwise_prefill_execution()
            )

    def test_preemption_requires_the_base_execution_flag(self) -> None:
        environment = self._clean_environment()
        environment[PREEMPTION_ENV] = "1"
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(
                MODULE.TTFTLayerwisePrefillError,
                "requires VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION",
            ):
                with MODULE.ttft_layerwise_prefill_execution("prefill"):
                    pass

    def test_resumable_boundary_preserves_owner_and_advances_epoch(self) -> None:
        environment = self._clean_environment()
        environment.update(
            {
                ENABLE_ENV: "1",
                PREEMPTION_ENV: "1",
                GROUP_ENV: "2",
            }
        )
        owner = {
            "scheduler_iteration_id": 11,
            "phase_submit_seq": 19,
            "phase_slot": 1,
            "request_ids": ("short-4",),
            "scheduled_tokens": (("short-4", 2048),),
            "profile_id": "d30_p10",
            "profile_epoch": 3,
            "prefill_mode": "native",
        }
        with patch.dict(os.environ, environment, clear=True):
            with MODULE.ttft_layerwise_prefill_execution(
                "prefill",
                preemption_allowed=True,
                **owner,
            ) as execution:
                self.assertTrue(execution.preemption_active)
                execution.record_group(
                    start_layer=0,
                    end_layer=2,
                    host_wall_s=0.001,
                )
                first_boundary = execution.make_boundary(
                    next_layer=2,
                    end_layer=6,
                )

            first_boundary.validate_owner(**owner)
            self.assertFalse(hasattr(first_boundary, "continuation"))
            resumed_owner = dict(owner)
            resumed_owner.update(
                profile_id="d32_p8",
                profile_epoch=4,
            )
            with MODULE.ttft_layerwise_prefill_execution(
                "prefill",
                preemption_allowed=True,
                existing_execution=execution,
                **resumed_owner,
            ) as resumed:
                self.assertIs(resumed, execution)
                resumed.record_group(
                    start_layer=2,
                    end_layer=4,
                    host_wall_s=0.001,
                )
                second_boundary = resumed.make_boundary(
                    next_layer=4,
                    end_layer=6,
                )

        self.assertEqual(first_boundary.model_yield_count, 1)
        self.assertEqual(second_boundary.model_yield_count, 2)
        self.assertEqual(execution.scheduler_yield_count, 1)
        self.assertEqual(execution.resume_count, 1)
        self.assertEqual(execution.profile_id, "d32_p8")
        self.assertEqual(execution.profile_epoch, 4)

    def test_worker_yields_only_after_an_exact_engine_signal(self) -> None:
        environment = self._clean_environment()
        environment.update(
            {
                ENABLE_ENV: "1",
                PREEMPTION_ENV: "1",
                GROUP_ENV: "2",
            }
        )
        owner = {
            "scheduler_iteration_id": 13,
            "phase_submit_seq": 21,
            "phase_slot": 1,
            "request_ids": ("long-1",),
            "scheduled_tokens": (("long-1", 4096),),
            "profile_id": "d30_p10",
            "profile_epoch": 5,
            "prefill_mode": "native",
        }
        with patch.dict(os.environ, environment, clear=True):
            with MODULE.ttft_layerwise_prefill_execution(
                "prefill",
                preemption_allowed=True,
                **owner,
            ) as execution:
                self.assertFalse(execution.consume_yield_request())
                signal = {
                    name: owner[name]
                    for name in (
                        "scheduler_iteration_id",
                        "phase_submit_seq",
                        "phase_slot",
                        "scheduled_tokens",
                    )
                }
                self.assertTrue(
                    MODULE.request_ttft_layerwise_prefill_yield(**signal)
                )
                self.assertFalse(
                    MODULE.request_ttft_layerwise_prefill_yield(**signal)
                )
                self.assertTrue(execution.consume_yield_request())
                self.assertFalse(execution.consume_yield_request())

        self.assertEqual(execution.yield_check_count, 3)
        self.assertEqual(execution.yield_signal_consumed_count, 1)

    def test_resumable_boundary_rejects_owner_or_epoch_regression(self) -> None:
        environment = self._clean_environment()
        environment.update(
            {
                ENABLE_ENV: "1",
                PREEMPTION_ENV: "1",
                GROUP_ENV: "2",
            }
        )
        owner = {
            "scheduler_iteration_id": 5,
            "phase_submit_seq": 7,
            "phase_slot": 0,
            "request_ids": ("long-1",),
            "scheduled_tokens": (("long-1", 4096),),
            "profile_id": "d30_p10",
            "profile_epoch": 8,
            "prefill_mode": "native",
        }
        with patch.dict(os.environ, environment, clear=True):
            with MODULE.ttft_layerwise_prefill_execution(
                "prefill",
                preemption_allowed=True,
                **owner,
            ) as execution:
                execution.record_group(
                    start_layer=0,
                    end_layer=2,
                    host_wall_s=0.0,
                )
                execution.make_boundary(next_layer=2, end_layer=4)

            wrong_request = dict(owner)
            wrong_request["request_ids"] = ("short-1",)
            wrong_request["scheduled_tokens"] = (("short-1", 4096),)
            with self.assertRaisesRegex(
                MODULE.TTFTLayerwisePrefillError,
                "owner changed",
            ):
                with MODULE.ttft_layerwise_prefill_execution(
                    "prefill",
                    preemption_allowed=True,
                    existing_execution=execution,
                    **wrong_request,
                ):
                    pass

            old_epoch = dict(owner)
            old_epoch["profile_epoch"] = 7
            with self.assertRaisesRegex(
                MODULE.TTFTLayerwisePrefillError,
                "epoch moved backwards",
            ):
                with MODULE.ttft_layerwise_prefill_execution(
                    "prefill",
                    preemption_allowed=True,
                    existing_execution=execution,
                    **old_epoch,
                ):
                    pass

    def test_resumable_multi_request_batch_is_one_exact_owner(self) -> None:
        environment = self._clean_environment()
        environment.update(
            {
                ENABLE_ENV: "1",
                PREEMPTION_ENV: "1",
                GROUP_ENV: "2",
            }
        )
        owner = {
            "scheduler_iteration_id": 31,
            "phase_submit_seq": 41,
            "phase_slot": 0,
            "request_ids": ("short-a", "long-b"),
            "scheduled_tokens": (
                ("short-a", 1024),
                ("long-b", 4096),
            ),
            "profile_id": "d30_p10",
            "profile_epoch": 9,
            "prefill_mode": "native",
        }
        with patch.dict(os.environ, environment, clear=True):
            with MODULE.ttft_layerwise_prefill_execution(
                "prefill",
                preemption_allowed=True,
                **owner,
            ) as execution:
                execution.record_group(
                    start_layer=0,
                    end_layer=2,
                    host_wall_s=0.001,
                )
                boundary = execution.make_boundary(
                    next_layer=2,
                    end_layer=6,
                )

            boundary.validate_owner(**owner)
            self.assertEqual(boundary.request_ids, owner["request_ids"])
            self.assertEqual(
                boundary.scheduled_tokens,
                owner["scheduled_tokens"],
            )
            self.assertEqual(
                execution.as_debug_dict()[
                    "ttft_layerwise_prefill_batch_width"
                ],
                2.0,
            )

            changed_count = dict(owner)
            changed_count["scheduled_tokens"] = (
                ("short-a", 1024),
                ("long-b", 2048),
            )
            with self.assertRaisesRegex(
                MODULE.TTFTLayerwisePrefillError,
                "owner changed",
            ):
                with MODULE.ttft_layerwise_prefill_execution(
                    "prefill",
                    preemption_allowed=True,
                    existing_execution=execution,
                    **changed_count,
                ):
                    pass

    def test_whole_batch_srpt_yields_when_candidate_displaces_tail(self) -> None:
        active_keys = (
            (100, 1.0, "active-head"),
            (1000, 2.0, "active-tail"),
        )
        self.assertTrue(
            MODULE.should_preempt_ttft_layerwise_prefill_batch(
                active_srpt_keys=active_keys,
                candidate_srpt_key=(500, 3.0, "new-short"),
            )
        )
        self.assertFalse(
            MODULE.should_preempt_ttft_layerwise_prefill_batch(
                active_srpt_keys=active_keys,
                candidate_srpt_key=(1200, 3.0, "new-long"),
            )
        )

    def test_llama_runner_and_engine_have_split_ownership(self) -> None:
        llama_source = (
            ROOT / "vendor_v8/vllm/model_executor/models/llama.py"
        ).read_text(encoding="utf-8")
        runner_source = (
            ROOT / "vendor_v8/vllm/v1/worker/gpu_model_runner.py"
        ).read_text(encoding="utf-8")
        engine_source = (
            ROOT / "vendor_v8/vllm/v1/engine/core.py"
        ).read_text(encoding="utf-8")
        scheduler_source = (
            ROOT / "vendor_v8/vllm/v1/core/sched/scheduler.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_forward_decoder_layer_range", llama_source)
        self.assertIn(
            "active_ttft_layerwise_prefill_execution()",
            llama_source,
        )
        self.assertIn(
            "with ttft_layerwise_prefill_execution(",
            runner_source,
        )
        self.assertIn("TTFTLayerwisePrefillModelYield", llama_source)
        self.assertIn("TTFTPrefillContinuationState", runner_source)
        self.assertIn("_suspend_ttft_prefill_boundary", engine_source)
        self.assertIn("_try_resume_ttft_prefill", engine_source)
        self.assertIn(
            "_request_ttft_prefill_preemption_if_needed",
            engine_source,
        )
        self.assertIn('policy_name != "srpt"', engine_source)
        self.assertIn(
            "and execution.consume_yield_request()",
            llama_source,
        )
        self.assertIn(
            "layerwise_prefill_execution_active",
            scheduler_source,
        )
        workspace_call = runner_source.split(
            "with prefill_workspace_execution(", 1
        )[1].split("):", 1)[0]
        self.assertIn("phase_slot=(", workspace_call)
        self.assertIn("phase_context.slot", workspace_call)
        self.assertNotIn(
            'unsupported.append("shared fixed Prefill workspace is enabled")',
            runner_source,
        )
        self.assertIn(
            "ttft_prefill_workspace_compatible",
            engine_source,
        )
        self.assertIn(
            "_ttft_prefill_deferred_abort_ids",
            engine_source,
        )
        self.assertIn(
            "active_tail_request_id",
            engine_source,
        )
        self.assertNotIn(
            "and len(scheduler_output.num_scheduled_tokens) == 1",
            engine_source,
        )
        self.assertNotIn(
            "or len(scheduler_output.num_scheduled_tokens) != 1",
            runner_source,
        )
        self.assertIn(
            "or ttft_layerwise_prefill_execution_requested()",
            engine_source,
        )
        boundary_branch = engine_source.split(
            "if isinstance(model_output, TTFTLayerwisePrefillBoundary):",
            1,
        )[1].split(
            "if (\n            model_output is None",
            1,
        )[0]
        self.assertLess(
            boundary_branch.index("_suspend_ttft_prefill_boundary"),
            boundary_branch.index("_process_aborts_queue"),
        )
        self.assertIn(
            "_flush_ttft_prefill_deferred_aborts(scheduler_output)",
            engine_source,
        )
        self.assertNotIn("scheduler.update_from_output", boundary_branch)


if __name__ == "__main__":
    unittest.main()
