#!/usr/bin/env python3
"""CPU-only correctness contracts for vendor_v9 Prefill depth two."""

from __future__ import annotations

import ast
import threading
import time
import unittest
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # CPU-only source checkouts need not install vLLM.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "vendor_v9/vllm/v1/engine/core.py"
SCHEDULER_PATH = ROOT / "vendor_v9/vllm/v1/core/sched/scheduler.py"
PROVIDER_PATH = (
    ROOT
    / "vendor_v9/vllm/v1/attention/ops/tiered_residency_provider.py"
)
CONTROLLER_PATH = (
    ROOT
    / "vendor_v9/vllm/model_executor/layers/pd_layerwise_handoff.py"
)

_LAYERWISE_SNAPSHOT: dict[str, int | float | str] = {}


@dataclass(frozen=True)
class _PDPrefillChainAdmission:
    request_id: str
    parent_submit_seq: int
    parent_slot: int
    parent_start_token: int
    parent_end_token: int
    parent_chain_depth: int
    parent_mode: str
    parent_profile_id: str
    parent_profile_epoch: int


def _engine_harness(*method_names: str) -> type:
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineCore"
    )
    methods = [
        node
        for node in engine.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    harness = ast.ClassDef(
        name="EngineHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "Future": Future,
        "PDPrefillChainAdmission": _PDPrefillChainAdmission,
        "PD_LAYERWISE_HANDOFF_CONTROLLER": SimpleNamespace(
            snapshot=lambda: dict(_LAYERWISE_SNAPSHOT),
            completed_transition_matches=lambda **_kwargs: True,
        ),
        "RequestStatus": SimpleNamespace(FINISHED_ABORTED="aborted"),
        "SchedulerOutput": object,
    }
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return namespace["EngineHarness"]


def _scheduler_guard_harness() -> type:
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    methods = [
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_pd_prefill_chain_request_native_eligible",
            "blocks_next_schedule",
        }
    ]
    harness = ast.ClassDef(
        name="SchedulerHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, Any] = {"Request": object}
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["SchedulerHarness"]


def _scheduler_methods_harness(*method_names: str) -> type:
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    methods = [
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    harness = ast.ClassDef(
        name="SchedulerHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "PDPrefillChainAdmission": _PDPrefillChainAdmission,
        "ReleasedPrefillVictim": object,
        "Request": object,
        "SchedulerOutput": object,
        "ResidencyVictimPolicy": SimpleNamespace(
            SCHEDULED_PREFILL_FIT="scheduled_prefill_fit"
        ),
    }
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["SchedulerHarness"]


def _provider_function(function_name: str):
    tree = ast.parse(PROVIDER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace: dict[str, Any] = {"Any": Any, "torch": torch}
    exec(compile(module, str(PROVIDER_PATH), "exec"), namespace)
    return namespace[function_name]


def _controller_harness() -> type:
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {
            "PDLayerwiseHandoffTarget",
            "_HandoffState",
            "PDLayerwiseHandoffController",
        }
    ]
    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "dataclass": dataclass,
        "field": field,
        "threading": threading,
        "time": time,
        "notify_pd_engine_progress": lambda: None,
    }
    exec(compile(module, str(CONTROLLER_PATH), "exec"), namespace)
    return namespace["PDLayerwiseHandoffController"]


def _output(
    *,
    request_id: str = "long-1",
    start: int = 0,
    tokens: int = 2048,
    slot: int = 0,
    submit_seq: int = 1,
    parent_seq: int = -1,
    final: bool = False,
    profile: str = "d10_p30",
    epoch: int = 7,
    mode: str = "native",
    tiered_plan: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_phase="prefill",
        num_scheduled_tokens={request_id: tokens},
        total_num_scheduled_tokens=tokens,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[request_id],
            num_computed_tokens=[start],
        ),
        pd_phase_slot=slot,
        pd_phase_submit_seq=submit_seq,
        pd_prefill_chain_parent_submit_seq=parent_seq,
        pd_prefill_chain_parent_slot=(0 if parent_seq >= 0 else -1),
        pd_prefill_chain_dependency_required=parent_seq >= 0,
        pd_prefill_chain_request_id=request_id,
        pd_prefill_chain_start_token=start,
        pd_prefill_chain_end_token=start + tokens,
        pd_prefill_chain_depth=0,
        pd_prefill_chain_parent_mode=(mode if parent_seq >= 0 else ""),
        pd_prefill_chain_parent_start_token=(0 if parent_seq >= 0 else -1),
        pd_prefill_chain_parent_end_token=(start if parent_seq >= 0 else -1),
        pd_prefill_chain_parent_profile_id=(
            profile if parent_seq >= 0 else ""
        ),
        pd_prefill_chain_parent_profile_epoch=(
            epoch if parent_seq >= 0 else -1
        ),
        pd_prefill_chain_mode_transition=False,
        pd_prefill_mode=mode,
        tiered_native_hbm_reload_request_ids=(),
        tiered_request_execution_plans=(
            ((request_id, submit_seq, mode, 1),) if tiered_plan else ()
        ),
        pd_sm_profile_id=profile,
        pd_sm_profile_epoch=epoch,
        pd_sm_decode_sms=10,
        pd_sm_prefill_sms=30,
        pd_ttft_handoff_request_ids=(request_id,) if final else (),
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
        iteration_debug={},
    )


class PrefillDepthTwoTest(unittest.TestCase):
    def test_vendor_v9_defaults_are_isolated_from_vendor_v8(self) -> None:
        v8_engine = (
            ROOT / "vendor_v8/vllm/v1/engine/core.py"
        ).read_text(encoding="utf-8")
        v9_engine = ENGINE_PATH.read_text(encoding="utf-8")
        v9_executor = (
            ROOT / "vendor_v9/vllm/v1/executor/uniproc_executor.py"
        ).read_text(encoding="utf-8")
        wrapper = (
            ROOT / "run_vendor_v9_dynamic_hbm_vs_full.sh"
        ).read_text(encoding="utf-8")
        shared_runner = (
            ROOT / "run_vendor_v8_short_validation.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'os.environ.get("VLLM_PD_PREFILL_INFLIGHT_DEPTH", "1")',
            v8_engine,
        )
        self.assertIn(
            'os.environ.get("VLLM_PD_GLOBAL_INFLIGHT_DEPTH", "2")',
            v8_engine,
        )
        self.assertIn(
            'os.environ.get("VLLM_PD_PREFILL_INFLIGHT_DEPTH", "2")',
            v9_engine,
        )
        self.assertIn(
            'os.environ.get("VLLM_PD_GLOBAL_INFLIGHT_DEPTH", "3")',
            v9_engine,
        )
        self.assertIn(
            'os.environ.get("VLLM_PD_GLOBAL_INFLIGHT_DEPTH", "3")',
            v9_executor,
        )
        self.assertIn("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH:-2", wrapper)
        self.assertIn("VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH:-3", wrapper)
        self.assertIn(
            "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF:-1",
            wrapper,
        )
        self.assertIn(
            "VLLM_VIRTUAL_CONTEXT_DECODE_CONTINUATION_ROLLING_RECLAIM:-1",
            wrapper,
        )
        self.assertIn(
            "FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS:-1",
            wrapper,
        )
        self.assertIn(
            "decode_continuation_rolling_reclaim=",
            shared_runner,
        )

    def test_chain_parent_requires_contiguous_range_and_profile_epoch(self) -> None:
        harness = _engine_harness(
            "_scheduler_output_request_ids",
            "_pd_scheduled_request_token_range",
            "_pd_output_has_tiered_request_authority",
            "_pd_same_request_prefill_chain_parent",
        )
        parent = _output(slot=0, submit_seq=10)
        child = _output(
            start=2048,
            slot=1,
            submit_seq=11,
            parent_seq=10,
        )
        phase_queues = {
            "prefill": [(Future(), parent, Future())],
            "decode": [],
        }

        selected, overlap = harness._pd_same_request_prefill_chain_parent(
            phase_queues,
            child,
            "prefill",
        )
        self.assertIs(selected, parent)
        self.assertTrue(overlap)

        ring_child = _output(
            start=2048,
            slot=1,
            submit_seq=11,
            parent_seq=10,
            mode="ring",
            tiered_plan=True,
        )
        self.assertEqual(
            harness._pd_same_request_prefill_chain_parent(
                phase_queues,
                ring_child,
                "prefill",
            ),
            (None, True),
        )

        tiered_parent = _output(
            slot=0,
            submit_seq=10,
            mode="ring",
            tiered_plan=True,
        )
        tiered_phase_queues = {
            "prefill": [(Future(), tiered_parent, Future())],
            "decode": [],
        }
        self.assertEqual(
            harness._pd_same_request_prefill_chain_parent(
                tiered_phase_queues,
                child,
                "prefill",
            ),
            (None, True),
        )

        gapped = _output(
            start=4096,
            slot=1,
            submit_seq=11,
            parent_seq=10,
        )
        self.assertEqual(
            harness._pd_same_request_prefill_chain_parent(
                phase_queues,
                gapped,
                "prefill",
            ),
            (None, True),
        )
        wrong_epoch = _output(start=2048, parent_seq=10, epoch=8)
        self.assertEqual(
            harness._pd_same_request_prefill_chain_parent(
                phase_queues,
                wrong_epoch,
                "prefill",
            ),
            (None, True),
        )

    def test_child_resolves_exact_parent_future_and_distinct_slot(self) -> None:
        harness = _engine_harness(
            "_scheduler_output_request_ids",
            "_pd_prefill_chain_dependency_future",
        )
        parent_future: Future[Any] = Future()
        parent = _output(slot=0, submit_seq=10)
        child = _output(
            start=2048,
            slot=1,
            submit_seq=11,
            parent_seq=10,
        )
        phase_queues = {
            "prefill": [(parent_future, parent, Future())],
            "decode": [],
        }
        self.assertIs(
            harness._pd_prefill_chain_dependency_future(
                phase_queues,
                child,
            ),
            parent_future,
        )
        child.pd_phase_slot = 0
        with self.assertRaisesRegex(RuntimeError, "ownership changed"):
            harness._pd_prefill_chain_dependency_future(phase_queues, child)

    def test_assignment_rotates_same_request_to_free_slot(self) -> None:
        harness_type = _engine_harness(
            "_scheduler_output_request_ids",
            "_pd_scheduled_request_token_range",
            "_pd_output_has_tiered_request_authority",
            "_pd_same_request_prefill_chain_parent",
            "_assign_pd_phase_submission",
        )
        harness = harness_type()
        harness._pd_phase_queue_submit_seq = 10
        harness.pd_phase_depth_enabled = True
        harness._pd_same_request_prefill_chain_submit_count = 0
        harness._pd_prefill_chain_dependency_submit_count = 0
        harness._record_pd_sm_transition_resume_submission = (
            lambda *_args: None
        )
        harness._pd_phase_inflight_limit = lambda _phase: 2
        harness._bind_pd_phase_execution_key = (
            lambda output, phase: (
                phase
                if output.pd_phase_slot == 0
                else f"{phase}:{output.pd_phase_slot}"
            )
        )
        parent = _output(slot=0, submit_seq=10)
        parent.pd_prefill_chain_depth = 0
        child = _output(start=2048, submit_seq=-1, parent_seq=10)
        child.iteration_debug = {}
        phase_queues = {
            "prefill": deque([(Future(), parent, Future())]),
            "decode": deque(),
        }

        execution_key = harness._assign_pd_phase_submission(
            phase_queues,
            child,
            "prefill",
        )
        self.assertEqual(execution_key, "prefill:1")
        self.assertEqual(child.pd_phase_slot, 1)
        self.assertEqual(child.pd_prefill_chain_parent_slot, 0)
        self.assertEqual(child.pd_prefill_chain_parent_submit_seq, 10)
        self.assertTrue(child.pd_prefill_chain_dependency_required)
        self.assertFalse(child.pd_prefill_chain_mode_transition)

    def test_failed_parent_discards_dependency_children(self) -> None:
        harness_type = _engine_harness(
            "_scheduler_output_request_ids",
            "_discard_pd_prefill_chain_descendants",
        )
        harness = harness_type()
        harness._iteration_debug_timings = {}
        harness._pd_prefill_chain_descendant_cancel_count = 0
        failed = _output(submit_seq=10)
        child_future: Future[Any] = Future()
        child_exec_future: Future[Any] = Future()
        child = _output(
            start=2048,
            slot=1,
            submit_seq=11,
            parent_seq=10,
        )
        phase_queues = {
            "prefill": deque([(child_future, child, child_exec_future)]),
            "decode": deque(),
        }

        discarded = harness._discard_pd_prefill_chain_descendants(
            phase_queues,
            failed,
        )
        self.assertEqual(discarded, 1)
        self.assertFalse(phase_queues["prefill"])
        self.assertTrue(child_future.cancelled())
        self.assertTrue(child_exec_future.cancelled())
        self.assertEqual(harness._pd_prefill_chain_descendant_cancel_count, 1)

    def test_same_request_lookahead_is_bounded(self) -> None:
        harness = _scheduler_guard_harness()()
        harness.enable_ttft_prefill_chunk_scheduling = True
        harness.ttft_prefill_chunk_scheduler = SimpleNamespace(
            quantum_tokens=2048
        )
        harness.pd_prefill_chain_max_lookahead_tokens = 2048
        harness._pd_pending_request_ids = {"long-1"}
        harness._pd_pending_phase_request_ids = {"prefill": {"long-1"}}
        harness._pd_prefill_chain_admissions = {}
        harness._pd_pending_schedule_block_count = 0
        harness._pd_pending_prefill_chain_allow_count = 0
        harness._pd_prefill_chain_admission_missing_block_count = 0
        harness._pd_prefill_chain_admission_progress_block_count = 0
        harness._pd_prefill_chain_admission_tiered_block_count = 0
        harness._pd_pending_schedule_last_reason = ""
        harness._debug_has_prefill_remaining = lambda _request: True
        harness.virtual_context_enabled = False
        request = SimpleNamespace(
            request_id="long-1",
            num_computed_tokens=2048,
            pd_output_placeholder_ledger=SimpleNamespace(active=None),
        )

        self.assertTrue(harness.blocks_next_schedule(request, "prefill"))
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "missing_prefill_chain_precommit_admission",
        )
        harness._pd_prefill_chain_admissions = {
            "long-1": _PDPrefillChainAdmission(
                request_id="long-1",
                parent_submit_seq=10,
                parent_slot=0,
                parent_start_token=0,
                parent_end_token=2048,
                parent_chain_depth=0,
                parent_mode="native",
                parent_profile_id="d10_p30",
                parent_profile_epoch=7,
            )
        }
        self.assertFalse(harness.blocks_next_schedule(request, "prefill"))
        harness._pd_pending_phase_request_ids["prefill_tiered"] = {
            "long-1"
        }
        self.assertTrue(harness.blocks_next_schedule(request, "prefill"))
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "prefill_chain_tiered_authority_fence",
        )
        harness._pd_pending_phase_request_ids.pop("prefill_tiered")
        harness.ttft_prefill_chunk_scheduler.quantum_tokens = 4096
        self.assertTrue(harness.blocks_next_schedule(request, "prefill"))
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "prefill_chain_lookahead_limit",
        )
        harness.ttft_prefill_chunk_scheduler.quantum_tokens = 2048
        harness._pd_pending_phase_request_ids["abort"] = {"long-1"}
        self.assertTrue(harness.blocks_next_schedule(request, "prefill"))
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "deferred_abort_dependency",
        )

    def test_multiround_chain_uses_one_exact_precommit_admission(self) -> None:
        engine_type = _engine_harness(
            "_scheduler_output_request_ids",
            "_pd_scheduled_request_token_range",
            "_pd_output_has_tiered_request_authority",
            "_pd_prefill_chain_execution_profile",
            "_pd_prefill_chain_admissions",
        )
        engine = engine_type()
        engine.pd_phase_depth_enabled = True
        engine._pd_phase_inflight_limit = lambda _phase: 2
        engine.pd_dynamic_sm_enabled = True
        engine.pd_dynamic_sm_enforced = True
        engine.pd_sm_transition = SimpleNamespace(active=False)
        engine.pd_sm_allocator = SimpleNamespace(
            current_profile=SimpleNamespace(profile_id="d10_p30"),
            epoch=7,
        )
        parent = _output(slot=0, submit_seq=10)
        phase_queues = {
            "prefill": deque([(Future(), parent, Future())]),
            "decode": deque(),
        }

        admissions = engine._pd_prefill_chain_admissions(phase_queues)
        admission = admissions["long-1"]
        self.assertEqual(
            (
                admission.parent_submit_seq,
                admission.parent_slot,
                admission.parent_start_token,
                admission.parent_end_token,
                admission.parent_profile_id,
                admission.parent_profile_epoch,
            ),
            (10, 0, 0, 2048, "d10_p30", 7),
        )

        scheduler_type = _scheduler_methods_harness(
            "_pd_output_request_token_range",
            "_apply_pd_prefill_chain_admission",
        )
        scheduler = scheduler_type()
        scheduler._pd_prefill_chain_admissions = admissions
        child = _output(start=2048, slot=0, submit_seq=-1)
        scheduler._apply_pd_prefill_chain_admission(child)
        self.assertTrue(child.pd_prefill_chain_dependency_required)
        self.assertEqual(child.pd_prefill_chain_parent_submit_seq, 10)
        self.assertEqual(child.pd_prefill_chain_parent_end_token, 2048)
        self.assertEqual(child.pd_prefill_chain_start_token, 2048)

        # A profile boundary makes the old parent capability unavailable
        # before Scheduler can advance request/KV state.
        engine.pd_sm_allocator.epoch = 8
        self.assertEqual(
            engine._pd_prefill_chain_admissions(phase_queues),
            {},
        )
        parent.tiered_request_execution_plans = ()
        engine.pd_phase_depth_enabled = False
        self.assertEqual(
            engine._pd_prefill_chain_admissions(phase_queues),
            {},
        )
        engine.pd_phase_depth_enabled = True
        engine.pd_sm_allocator.epoch = 7
        parent.tiered_request_execution_plans = (
            ("long-1", 1, "context_ring", 1),
        )
        self.assertEqual(
            engine._pd_prefill_chain_admissions(phase_queues),
            {},
        )

    def test_dependency_child_cannot_use_paired_rpc(self) -> None:
        harness = _engine_harness("_scheduler_output_can_pair")
        parentless = _output()
        dependency_child = _output(start=2048, parent_seq=10)
        self.assertTrue(harness._scheduler_output_can_pair(parentless))
        self.assertFalse(
            harness._scheduler_output_can_pair(dependency_child)
        )

    def test_tiered_parent_is_published_as_chain_fence(self) -> None:
        harness_type = _engine_harness(
            "_scheduler_output_request_ids",
            "_pd_phase_queue_items",
            "_pd_output_has_tiered_request_authority",
            "_pd_phase_queue_pending_request_ids_by_phase",
        )
        harness = harness_type()
        harness._pd_deferred_abort_request_ids = set()
        parent = _output(
            submit_seq=10,
            mode="ring",
            tiered_plan=True,
        )
        phase_queues = {
            "prefill": deque([(Future(), parent, Future())]),
            "decode": deque(),
            "mixed": deque(),
            "idle": deque(),
        }

        pending = harness._pd_phase_queue_pending_request_ids_by_phase(
            phase_queues
        )
        self.assertEqual(pending["prefill"], {"long-1"})
        self.assertEqual(pending["prefill_tiered"], {"long-1"})

    def test_pending_request_cannot_self_victim_on_chain_miss(self) -> None:
        scheduler_source = SCHEDULER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(scheduler_source)
        scheduler = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
        )
        guarded_methods = {
            node.name: ast.get_source_segment(scheduler_source, node) or ""
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_try_defer_running_prefill_to_tiered_provider",
                "_try_defer_running_prefill_for_virtual_context",
            }
        }
        self.assertEqual(len(guarded_methods), 2)
        for method_source in guarded_methods.values():
            self.assertIn(
                "request.request_id in self._pd_pending_request_ids",
                method_source,
            )
            self.assertIn('"execution_future_pending"', method_source)

    def test_abort_waits_for_last_future_owner(self) -> None:
        harness_type = _engine_harness(
            "_scheduler_output_request_ids",
            "_pd_phase_queue_items",
            "_pd_phase_queue_pending_request_ids",
            "abort_requests",
            "_flush_pd_deferred_aborts",
        )
        harness = harness_type()
        output = _output(submit_seq=10)
        harness.pd_phase_queues = {
            "prefill": deque([(Future(), output, Future())]),
            "decode": deque(),
            "mixed": deque(),
            "idle": deque(),
        }
        harness._pd_recovering_request_ids = set()
        harness._pd_deferred_abort_request_ids = set()
        harness._pd_deferred_abort_count = 0
        harness._pd_capacity_retry_gate = SimpleNamespace(clear=lambda: None)
        finish_calls: list[tuple[set[str], str]] = []
        harness.scheduler = SimpleNamespace(
            finish_requests=lambda request_ids, status: finish_calls.append(
                (set(request_ids), status)
            )
        )

        harness.abort_requests(["long-1"])
        self.assertEqual(
            harness._pd_deferred_abort_request_ids,
            {"long-1"},
        )
        self.assertFalse(finish_calls)
        harness.pd_phase_queues["prefill"].clear()
        flushed = harness._flush_pd_deferred_aborts(
            harness.pd_phase_queues
        )
        self.assertEqual(flushed, {"long-1"})
        self.assertEqual(finish_calls, [({"long-1"}, "aborted")])

    def test_worker_and_executor_have_fail_fast_slot_ownership(self) -> None:
        runner = (
            ROOT / "vendor_v9/vllm/v1/worker/gpu_model_runner.py"
        ).read_text(encoding="utf-8")
        executor = (
            ROOT / "vendor_v9/vllm/v1/executor/uniproc_executor.py"
        ).read_text(encoding="utf-8")
        engine = ENGINE_PATH.read_text(encoding="utf-8")

        self.assertIn("execute_owner_submit_seq: int = -1", runner)
        self.assertIn(
            "A phase-local runner slot received execute_model before",
            runner,
        )
        self.assertIn(
            "its previous execute/sample owner was resolved",
            runner,
        )
        self.assertIn("self._pd_phase_slot_limits", runner)
        self.assertIn(
            "Phase-local runner slot exceeds its configured ownership",
            runner,
        )
        self.assertIn("dependency: Future[Any] | None = None", executor)
        self.assertIn("dependency=dependency", executor)
        self.assertIn("Phase-depth sampling has no slot-local execute owner", executor)
        self.assertIn("execution_dependency =", engine)
        self.assertIn("pd_prefill_chain_success_dependency", engine)
        self.assertIn(
            "vendor-v9 supports only Prefill depth 1 or 2",
            engine,
        )
        self.assertNotIn(
            "scheduler_output.pd_phase_slot = chained_slot",
            engine,
        )

    def test_layerwise_depth_two_rebinds_only_gated_native_child(self) -> None:
        global _LAYERWISE_SNAPSHOT
        _LAYERWISE_SNAPSHOT = {
            "pd_layerwise_handoff_execution_submit_seq": 10.0,
        }
        harness_type = _engine_harness(
            "_scheduler_output_request_ids",
            "_pd_output_has_tiered_request_authority",
            "_pd_phase_queue_total_depth",
            "_pd_layerwise_queued_continuations",
            "_pd_rebind_layerwise_queued_continuations",
            "_pd_layerwise_transition_boundary_ready",
        )
        harness = harness_type()
        harness.pd_layerwise_handoff_enabled = True
        harness._pd_layerwise_handoff_fence_active = True
        harness.pd_sm_transition = SimpleNamespace(
            active=True,
            source_profile_id="d10_p30",
            source_epoch=7,
            source_prefill_sms=30,
            target_profile_id="d0_p40",
            target_prefill_sms=40,
        )
        harness._pd_layerwise_queued_continuation_rebind_count = 0
        harness._pd_layerwise_queued_continuation_rebind_failure_count = 0
        parent = _output(slot=0, submit_seq=10)
        child = _output(
            start=2048,
            slot=1,
            submit_seq=11,
            parent_seq=10,
        )
        parent_future: Future[Any] = Future()
        child_future: Future[Any] = Future()
        child_exec_future: Future[Any] = Future()
        phase_queues = {
            "decode": deque(),
            "prefill": deque(
                [
                    (child_future, child, child_exec_future),
                    (parent_future, parent, Future()),
                ]
            ),
        }

        self.assertEqual(
            harness._pd_layerwise_queued_continuations(phase_queues),
            (child,),
        )
        self.assertTrue(
            harness._pd_layerwise_transition_boundary_ready(
                phase_queues,
                old_decode=0,
                old_prefill=2,
                newer_epoch=0,
            )
        )
        rebound = harness._pd_rebind_layerwise_queued_continuations(
            phase_queues,
            source_profile_id="d10_p30",
            source_epoch=7,
            target_profile_id="d0_p40",
            target_epoch=8,
            target_decode_sms=0,
            target_prefill_sms=40,
        )
        self.assertEqual(rebound, 1)
        self.assertEqual((child.pd_sm_profile_id, child.pd_sm_profile_epoch), (
            "d0_p40",
            8,
        ))
        self.assertEqual((child.pd_sm_decode_sms, child.pd_sm_prefill_sms), (0, 40))
        self.assertEqual(
            harness._pd_layerwise_queued_continuation_rebind_count,
            1,
        )

        independent = _output(
            request_id="long-2",
            slot=1,
            submit_seq=12,
            parent_seq=-1,
        )
        phase_queues["prefill"][0] = (
            Future(),
            independent,
            Future(),
        )
        self.assertIsNone(
            harness._pd_layerwise_queued_continuations(phase_queues)
        )

    def test_layerwise_ownership_is_acquired_at_model_entry(self) -> None:
        engine = ENGINE_PATH.read_text(encoding="utf-8")
        executor = (
            ROOT / "vendor_v9/vllm/v1/executor/uniproc_executor.py"
        ).read_text(encoding="utf-8")
        controller = (
            ROOT
            / "vendor_v9/vllm/model_executor/layers/pd_layerwise_handoff.py"
        ).read_text(encoding="utf-8")
        green_context = (
            ROOT / "vendor_v9/vllm/model_executor/layers/green_context.py"
        ).read_text(encoding="utf-8")

        self.assertIn("self.pd_prefill_inflight_depth in {1, 2}", engine)
        self.assertNotIn(
            "PD_LAYERWISE_HANDOFF_CONTROLLER.arm(\n"
            "                source_profile_id=execution_profile_id",
            engine,
        )
        self.assertNotIn("_arm_pd_layerwise_prefill_execution", executor)
        self.assertNotIn("PD_LAYERWISE_HANDOFF_CONTROLLER", executor)
        self.assertIn(
            "PD_LAYERWISE_HANDOFF_CONTROLLER.acquire_execution(",
            green_context,
        )
        self.assertIn(
            "queued depth-two Futures own no controller state",
            green_context,
        )
        self.assertIn("def acquire_execution(", controller)
        self.assertIn("execution_submit_seq: int", controller)

    def test_model_entry_ownership_preserves_a_queued_independent_prefill(
        self,
    ) -> None:
        controller = _controller_harness()()
        active_generation = controller.acquire_execution(
            source_profile_id="d0_p40",
            source_epoch=3,
            execution_submit_seq=46,
        )
        self.assertEqual(active_generation, 1)
        self.assertEqual(
            controller.acquire_execution(
                source_profile_id="d0_p40",
                source_epoch=3,
                execution_submit_seq=47,
            ),
            0,
        )
        controller.complete_execution(active_generation)
        queued_generation = controller.acquire_execution(
            source_profile_id="d0_p40",
            source_epoch=3,
            execution_submit_seq=47,
        )
        self.assertEqual(queued_generation, 2)
        self.assertEqual(
            controller.snapshot()[
                "pd_layerwise_handoff_execution_submit_seq"
            ],
            47.0,
        )

    def test_final_layer_request_transfers_to_next_prefill_entry(self) -> None:
        controller = _controller_harness()()
        first_generation = controller.acquire_execution(
            source_profile_id="d30_p10",
            source_epoch=7,
            execution_submit_seq=46,
        )
        self.assertTrue(
            controller.request(
                source_profile_id="d30_p10",
                source_epoch=7,
                target_profile_id="d0_p40",
                target_epoch=8,
            )
        )
        controller.complete_execution(first_generation)
        second_generation = controller.acquire_execution(
            source_profile_id="d30_p10",
            source_epoch=7,
            execution_submit_seq=47,
        )
        self.assertEqual(second_generation, 2)
        target = controller.target_for(second_generation)
        self.assertIsNotNone(target)
        self.assertEqual((target.profile_id, target.profile_epoch), (
            "d0_p40",
            8,
        ))
        self.assertEqual(
            controller.snapshot()[
                "pd_layerwise_handoff_request_transfer_count"
            ],
            1.0,
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_mutable_dram_slab_ignores_caller_inference_mode(self) -> None:
        assert torch is not None
        allocate = _provider_function("_allocate_mutable_cpu_tensor")
        mutable_view = _provider_function("_mutable_cpu_tensor_view")
        with torch.inference_mode():
            tensor = allocate(
                (2, 4),
                dtype=torch.float32,
                pin_memory=False,
            )
            view = mutable_view(tensor, (0, slice(None)))
        self.assertFalse(torch.is_inference(tensor))
        self.assertFalse(torch.is_inference(view))
        view.copy_(torch.ones_like(view))
        tensor[1].copy_(torch.ones_like(tensor[1]))
        self.assertEqual(float(tensor.sum()), 8.0)

    def test_decode_continuation_reclaims_only_at_growth_boundary(self) -> None:
        harness = _scheduler_methods_harness(
            "_try_decode_continuation_rolling_reclaim"
        )()
        harness.virtual_context_decode_continuation_rolling_reclaim = True
        harness.virtual_context_residency_victim_policy = "scheduled_prefill_fit"
        harness._debug_has_prefill_remaining = lambda _request: False
        harness._has_pending_prefill_for_ring_progress = lambda: True
        harness.num_lookahead_tokens = 0
        harness.virtual_context_staging_reserved_blocks = 2
        harness.virtual_context_ring_min_free_blocks = 1
        harness.virtual_context_decode_hotset_reserve_blocks = 2
        harness.virtual_context_ring_bw_decode_growth_blocks = 1
        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 5
        harness._decode_continuation_rolling_reclaim_check_count = 0
        harness._decode_continuation_rolling_reclaim_trigger_count = 0
        harness._decode_continuation_rolling_reclaim_success_count = 0
        harness._decode_continuation_rolling_reclaim_reclaimed_blocks = 0
        free_blocks = SimpleNamespace(get_num_free_blocks=lambda: 8)
        harness.kv_cache_manager = SimpleNamespace(
            estimate_num_blocks_to_allocate=lambda *_args, **_kwargs: 1,
            block_pool=free_blocks,
        )
        calls: list[dict[str, Any]] = []
        victim = SimpleNamespace(reclaimed_blocks=5)

        def reclaim(**kwargs):
            calls.append(kwargs)
            return 1, 0, [victim], {"copy_ready"}, 13

        harness._reclaim_prefill_victims_until_hbm_target = reclaim
        request = SimpleNamespace(
            request_id="long-1",
            kv_retention_decode_continuation_lease_active=True,
        )

        result = harness._try_decode_continuation_rolling_reclaim(
            request,
            num_new_tokens=1,
            excluded_request_ids={"scheduled"},
        )
        self.assertIsNotNone(result)
        self.assertEqual(calls[0]["target_free_blocks"], 11)
        self.assertTrue(calls[0]["require_full_target_before_reclaim"])
        self.assertEqual(
            calls[0]["excluded_request_ids"],
            {"scheduled", "long-1"},
        )

        harness.kv_cache_manager.estimate_num_blocks_to_allocate = (
            lambda *_args, **_kwargs: 0
        )
        self.assertIsNone(
            harness._try_decode_continuation_rolling_reclaim(
                request,
                num_new_tokens=1,
                excluded_request_ids=set(),
            )
        )


if __name__ == "__main__":
    unittest.main()
