#!/usr/bin/env python3
"""CPU-only checks for scheduled-prefill-fit residency reclaim."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "vendor_v8/vllm/v1/core/sched/residency_policy.py"
SCHEDULER_PATH = ROOT / "vendor_v8/vllm/v1/core/sched/scheduler.py"
KV_MANAGER_PATH = ROOT / "vendor_v8/vllm/v1/core/kv_cache_manager.py"
SINGLE_MANAGER_PATH = (
    ROOT / "vendor_v8/vllm/v1/core/single_type_kv_cache_manager.py"
)
VIRTUAL_CONTEXT_PATH = ROOT / "vendor_v8/vllm/v1/core/virtual_context.py"
D2H_PLAN_PATH = (
    ROOT
    / "vendor_v8/vllm/v1/attention/ops/tiered_d2h_source_plan.py"
)
PROVIDER_PATH = (
    ROOT
    / "vendor_v8/vllm/v1/attention/ops/tiered_residency_provider.py"
)
DESCRIPTOR_PATH = (
    ROOT
    / "vendor_v8/vllm/v1/worker/kv_retention_residency_descriptor.py"
)

SPEC = importlib.util.spec_from_file_location(
    "vendor_v8_scheduled_prefill_fit_residency_test",
    POLICY_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

D2H_SPEC = importlib.util.spec_from_file_location(
    "vendor_v8_scheduled_prefill_fit_d2h_plan_test",
    D2H_PLAN_PATH,
)
assert D2H_SPEC is not None and D2H_SPEC.loader is not None
D2H_MODULE = importlib.util.module_from_spec(D2H_SPEC)
sys.modules[D2H_SPEC.name] = D2H_MODULE
D2H_SPEC.loader.exec_module(D2H_MODULE)

DESCRIPTOR_SPEC = importlib.util.spec_from_file_location(
    "vendor_v8_kv_retention_residency_descriptor_test",
    DESCRIPTOR_PATH,
)
assert DESCRIPTOR_SPEC is not None and DESCRIPTOR_SPEC.loader is not None
DESCRIPTOR_MODULE = importlib.util.module_from_spec(DESCRIPTOR_SPEC)
sys.modules[DESCRIPTOR_SPEC.name] = DESCRIPTOR_MODULE
DESCRIPTOR_SPEC.loader.exec_module(DESCRIPTOR_MODULE)


def load_release_harness():
    tree = ast.parse(SINGLE_MANAGER_PATH.read_text(encoding="utf-8"))
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SingleTypeKVCacheManager"
    )
    methods = [
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "allocate_hbm_reload_blocks",
            "clear_partial_residency",
            "enter_partial_residency",
            "get_num_hbm_reload_blocks_to_allocate",
            "get_retained_residency_counts",
            "has_partial_residency_pages",
            "release_hbm_pages",
            "resume_partial_residency",
            "validate_hbm_release_pages",
        }
    ]
    harness = ast.ClassDef(
        name="ReleaseHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {
        "KVCacheBlock": object,
        "cdiv": lambda numerator, denominator: (
            numerator + denominator - 1
        )
        // denominator,
    }
    exec(compile(module, str(SINGLE_MANAGER_PATH), "exec"), namespace)
    return namespace["ReleaseHarness"]


def load_preparation_harness():
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_scheduled_prefill_fit_preparation_status"
    )
    active_prefills = ast.parse(
        "def _active_virtual_context_prefill_requests(self):\n"
        "    return tuple(self.running)\n"
    ).body[0]
    harness = ast.ClassDef(
        name="PreparationHarness",
        bases=[],
        keywords=[],
        body=[active_prefills, method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {
        "ScheduledPrefillFitPreparation": object,
    }
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["PreparationHarness"]


def load_snapshot_planner_harness():
    tree = ast.parse(KV_MANAGER_PATH.read_text(encoding="utf-8"))
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager"
    )
    method = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "plan_virtual_context_hbm_snapshot_pages"
    )
    harness = ast.ClassDef(
        name="SnapshotPlannerHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {}
    exec(compile(module, str(KV_MANAGER_PATH), "exec"), namespace)
    return namespace["SnapshotPlannerHarness"]


def load_capacity_claim_harness():
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    claim = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ScheduledPrefillCapacityClaim"
    )
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
            "_prefill_native_safe_before_ring_pressure",
            "_scheduled_prefill_fit_active_decode",
            "_scheduled_prefill_fit_capacity_claim",
            "_scheduled_prefill_fit_headroom_partition",
        }
    ]
    harness = ast.ClassDef(
        name="CapacityClaimHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[claim, harness], type_ignores=[])
    )
    namespace: dict[str, object] = {
        "Request": object,
        "dataclass": dataclass,
        "scheduled_prefill_reclaim_blocks": (
            MODULE.scheduled_prefill_reclaim_blocks
        ),
        "ResidencyVictimPolicy": MODULE.ResidencyVictimPolicy,
        "_read_bool_env": lambda _name: False,
        "virtual_context_residency_victim_context_tokens": (
            lambda computed, deferred: max(computed, deferred)
        ),
    }
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["CapacityClaimHarness"]


def load_ttft_chunk_cap_harness():
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_ttft_cap_selected_prefill_chunk"
    )
    harness = ast.ClassDef(
        name="TTFTChunkCapHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {"Request": object}
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["TTFTChunkCapHarness"]


def load_decode_growth_runway_types():
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {
            "ScheduledPrefillCapacityClaim",
            "ScheduledPrefillFitDecodeGrowthRunway",
        }
    ]
    module = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[])
    )
    namespace: dict[str, object] = {
        "dataclass": dataclass,
        "scheduled_prefill_reclaim_blocks": (
            MODULE.scheduled_prefill_reclaim_blocks
        ),
    }
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return (
        namespace["ScheduledPrefillCapacityClaim"],
        namespace["ScheduledPrefillFitDecodeGrowthRunway"],
    )


def load_proactive_reclaim_harness():
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
            "_scheduled_prefill_fit_active_decode",
            "_scheduled_prefill_fit_headroom_partition",
            "_scheduled_prefill_fit_proactive_snapshot_plan",
            "_scheduled_prefill_fit_ring_fallback_reclaim_requirement",
        }
    ]
    harness = ast.ClassDef(
        name="ProactiveReclaimHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {
        "Request": object,
        "ResidencyVictimPolicy": MODULE.ResidencyVictimPolicy,
        "PrefillChunkSchedulingPolicy": SimpleNamespace(SRPT="srpt"),
        "ProjectedPrefillCandidate": MODULE.ProjectedPrefillCandidate,
        "ResidencyVictimCandidate": MODULE.ResidencyVictimCandidate,
        "projected_srpt_prefill_order": MODULE.projected_srpt_prefill_order,
        "select_best_fit_whole_request_victims": (
            MODULE.select_best_fit_whole_request_victims
        ),
        "virtual_context_residency_victim_context_tokens": (
            lambda computed, deferred: max(computed, deferred)
        ),
    }
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["ProactiveReclaimHarness"]


def load_decode_handoff_requirement_harness():
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_virtual_context_native_decode_handoff_requirement"
    )
    harness = ast.ClassDef(
        name="DecodeHandoffRequirementHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {
        "Request": object,
        "ResidencyVictimPolicy": MODULE.ResidencyVictimPolicy,
        "defaultdict": defaultdict,
        "TieredAllocatorInvariantError": RuntimeError,
        "virtual_context_residency_victim_context_tokens": (
            lambda computed, deferred: max(0, computed, deferred)
        ),
    }
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["DecodeHandoffRequirementHarness"]


def load_control_guard_harness():
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    methods = [
        node
        for node in scheduler.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_cap_pd_phase_tokens",
            "_classify_scheduled_phase",
            "_debug_has_prefill_remaining",
            "_pd_prefill_lane_boundary_tokens",
            "_pd_prefill_lane_remaining_tokens",
            "_pd_preemption_protection_reason",
            "_validate_pd_phase_schedule",
            "blocks_next_schedule",
        }
    ]
    harness = ast.ClassDef(
        name="ControlGuardHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {
        "Request": object,
        "ResidencyVictimPolicy": MODULE.ResidencyVictimPolicy,
    }
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace["ControlGuardHarness"]


def load_engine_prefill_chain_harness():
    engine_path = ROOT / "vendor_v8/vllm/v1/engine/core.py"
    tree = ast.parse(engine_path.read_text(encoding="utf-8"))
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineCore"
    )
    methods = [
        node
        for node in engine.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_scheduler_output_request_ids",
            "_pd_pending_phase_owner_overlap",
        }
    ]
    harness = ast.ClassDef(
        name="EnginePrefillChainHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[])
    )
    namespace: dict[str, object] = {"SchedulerOutput": object}
    exec(compile(module, str(engine_path), "exec"), namespace)
    return namespace["EnginePrefillChainHarness"]


class ScheduledPrefillFitPolicyTest(unittest.TestCase):
    @staticmethod
    def _decode_handoff_requirement(
        *,
        reload_page_count: int,
        retained_counts: tuple[int, int] = (0, 0),
        planned_pages: tuple[SimpleNamespace, ...] = (),
    ):
        retained_calls: list[tuple[str, int]] = []

        def get_retained_residency_counts(
            request_id: str,
            logical_computed_tokens: int,
        ) -> tuple[int, int]:
            retained_calls.append((request_id, logical_computed_tokens))
            if logical_computed_tokens <= 0:
                raise RuntimeError(
                    "retained-residency validation requires positive logical "
                    f"progress: request={request_id} "
                    f"tokens={logical_computed_tokens}"
                )
            return retained_counts

        harness = load_decode_handoff_requirement_harness()()
        harness.kv_cache_manager = SimpleNamespace(
            get_virtual_context_hbm_reload_required_page_count=(
                lambda _request_ids: reload_page_count
            ),
            get_retained_residency_counts=get_retained_residency_counts,
            estimate_num_blocks_to_allocate=lambda request, _tokens, **kw: (
                get_retained_residency_counts(
                    request.request_id,
                    request.num_computed_tokens + kw["num_external_computed_tokens"],
                )[1]
            ),
            block_pool=SimpleNamespace(get_num_free_blocks=lambda: 100),
            plan_virtual_context_hbm_reload_pages=(
                lambda *, request_ids: planned_pages
            ),
        )
        harness.virtual_context_reload_to_hbm_min_free_blocks = 0
        harness.virtual_context_staging_reserved_blocks = 2
        harness.virtual_context_ring_min_free_blocks = 0
        harness.virtual_context_decode_hotset_reserve_blocks = 0
        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 3
        harness.virtual_context_residency_victim_policy = (
            MODULE.ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        )
        harness._has_pending_prefill_for_ring_progress = lambda: False
        harness._virtual_context_active_decode_ring_progress_reserve = (
            lambda: (0, 0, 0)
        )
        return harness, retained_calls

    def test_first_chunk_final_prefill_skips_zero_progress_retained_validation(
        self,
    ) -> None:
        harness, retained_calls = self._decode_handoff_requirement(
            reload_page_count=0,
        )
        request = SimpleNamespace(
            request_id="warmup",
            num_computed_tokens=0,
            virtual_context_deferred_logical_computed_tokens=0,
        )

        self.assertEqual(
            harness._virtual_context_native_decode_handoff_requirement(request),
            (0, {}, 0, 0, False),
        )
        self.assertEqual(retained_calls, [])

    def test_decode_handoff_uses_deferred_durable_progress(self) -> None:
        pages = (
            SimpleNamespace(kv_cache_group_id=0, logical_token_page=0),
            SimpleNamespace(kv_cache_group_id=0, logical_token_page=1),
        )
        harness, retained_calls = self._decode_handoff_requirement(
            reload_page_count=2,
            retained_counts=(2, 2),
            planned_pages=pages,
        )
        request = SimpleNamespace(
            request_id="dram-backed",
            num_computed_tokens=0,
            virtual_context_deferred_logical_computed_tokens=64,
        )

        self.assertEqual(
            harness._virtual_context_native_decode_handoff_requirement(request),
            (2, {0: {0, 1}}, 2, 7, False),
        )
        self.assertEqual(retained_calls, [("dram-backed", 64)])

    def test_reload_pages_with_zero_durable_progress_remain_fatal(self) -> None:
        harness, retained_calls = self._decode_handoff_requirement(
            reload_page_count=1,
        )
        request = SimpleNamespace(
            request_id="invalid-dram-backed",
            num_computed_tokens=0,
            virtual_context_deferred_logical_computed_tokens=0,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "requires positive logical progress",
        ):
            harness._virtual_context_native_decode_handoff_requirement(request)
        self.assertEqual(retained_calls, [])

    @staticmethod
    def _whole_request_planner(*, pending: set[str] | None = None):
        request_specs = (
            ("long-1", 50_000, 34_816, 1.0),
            ("long-2", 50_000, 18_432, 2.0),
            ("short-3", 10_000, 8_192, 3.0),
            ("short-4", 10_000, 4_096, 4.0),
        )
        requests = [
            SimpleNamespace(
                request_id=request_id,
                num_prompt_tokens=prompt_tokens,
                num_computed_tokens=computed_tokens,
                virtual_context_deferred_logical_computed_tokens=0,
                arrival_time=arrival_time,
                has_encoder_inputs=False,
            )
            for (
                request_id,
                prompt_tokens,
                computed_tokens,
                arrival_time,
            ) in request_specs
        ]
        resident_blocks = {
            "long-1": 100,
            "long-2": 50,
            "short-3": 512,
            "short-4": 120,
        }
        kv_manager = SimpleNamespace(
            block_pool=SimpleNamespace(get_num_free_blocks=lambda: 14),
            get_request_resident_block_count=(
                lambda request_id: resident_blocks.get(request_id, 0)
            ),
            get_block_ids=(
                lambda request_id: (
                    list(range(resident_blocks.get(request_id, 0))),
                )
            ),
            plan_virtual_context_hbm_snapshot_pages=(
                lambda *, request_id, logical_tokens, max_pages, group_id: (
                    tuple(
                        range(min(max_pages, resident_blocks.get(request_id, 0)))
                    )
                    if logical_tokens > 0 and group_id == 0
                    else ()
                )
            ),
        )
        harness = load_proactive_reclaim_harness()()
        harness.virtual_context_residency_victim_policy = (
            MODULE.ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        )
        harness.requests = {request.request_id: request for request in requests}
        harness.running = requests
        harness.waiting = []
        harness._pd_pending_request_ids = set(pending or ())
        harness._virtual_context_waiting_decode_ready = lambda _request: False
        harness._active_virtual_context_prefill_requests = lambda: tuple(requests)
        harness.virtual_context_staging_reserved_blocks = 0
        harness.virtual_context_ring_min_free_blocks = 0
        harness.virtual_context_decode_hotset_reserve_blocks = 0
        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 0
        harness._debug_running_phase_counts = lambda: {
            "num_running_decode_reqs": 0
        }
        harness._pd_pending_decode_active = lambda: False
        harness.virtual_context_reload_to_hbm = False
        harness.enable_ttft_prefill_chunk_scheduling = True
        harness.ttft_prefill_native_chunk_max_tokens = 0
        harness.ttft_prefill_chunk_scheduler = SimpleNamespace(
            policy="srpt",
            candidate_order=tuple(request.request_id for request in requests),
        )
        harness._ttft_prefill_execution_mode = lambda _request: "native"
        harness.is_encoder_decoder = False
        harness.block_size = 16
        harness.kv_cache_manager = kv_manager
        harness._scheduled_prefill_fit_protected_victim_skip_count = 0
        harness._scheduled_prefill_fit_capacity_claim = (
            lambda _target, **_kwargs: SimpleNamespace(
                chunk_tokens=2_048,
                raw_required_blocks=128,
                reserved_blocks=0,
                free_blocks=14,
                reclaim_blocks=114,
            )
        )
        harness._virtual_context_reload_required_page_count = (
            lambda _request_ids, _phase="decode": 0
        )
        harness._test_resident_blocks = resident_blocks
        return harness

    def test_whole_request_plan_grows_best_fit_while_protecting_target(self) -> None:
        harness = self._whole_request_planner()

        *_, deficit_blocks, pages = (
            harness._scheduled_prefill_fit_proactive_snapshot_plan(
                {"short-3": 1_808}, ring_token_cap=2_048
            )
        )

        page_request_ids = [entry[0] for entry in pages]
        self.assertEqual(deficit_blocks, 114)
        self.assertEqual(page_request_ids.count("long-1"), 100)
        self.assertEqual(page_request_ids.count("long-2"), 50)
        self.assertNotIn("short-4", page_request_ids)

    def test_whole_request_plan_waits_for_pending_best_fit_victim(self) -> None:
        harness = self._whole_request_planner(pending={"long-1"})

        *_, deficit_blocks, pages = (
            harness._scheduled_prefill_fit_proactive_snapshot_plan(
                {"short-3": 1_808}, ring_token_cap=2_048
            )
        )

        self.assertEqual(deficit_blocks, 114)
        self.assertEqual(pages, ())

    def test_ring_fallback_discards_native_peer_plan(
        self,
    ) -> None:
        harness = self._whole_request_planner()
        harness._scheduled_prefill_fit_capacity_claim = (
            lambda _target, **_kwargs: SimpleNamespace(
                chunk_tokens=2_048,
                raw_required_blocks=1_014,
                reserved_blocks=0,
                free_blocks=14,
                reclaim_blocks=1_000,
            )
        )

        target, victims, *_, deficit_blocks, pages = (
            harness._scheduled_prefill_fit_proactive_snapshot_plan(
                {"short-3": 1_808}, ring_token_cap=2_048
            )
        )

        self.assertEqual(target.request_id, "short-4")
        self.assertEqual(deficit_blocks, 114)
        self.assertEqual([victim.request_id for victim in victims], ["short-4"])
        self.assertEqual(
            {request_id for request_id, _, _ in pages},
            {"short-4"},
        )
        self.assertEqual(harness._scheduled_prefill_fit_last_plan_mode, "ring_fallback")
        self.assertEqual(
            harness._scheduled_prefill_fit_last_native_deficit_blocks,
            1_000,
        )
        self.assertEqual(
            harness._scheduled_prefill_fit_last_ring_fallback_deficit_blocks,
            114,
        )
        self.assertEqual(
            harness._scheduled_prefill_fit_last_avoided_peer_blocks,
            150,
        )

    def test_ring_fallback_adds_only_the_live_ring_deficit_peer(self) -> None:
        harness = self._whole_request_planner()
        harness._test_resident_blocks["short-4"] = 50
        harness._scheduled_prefill_fit_capacity_claim = (
            lambda _target, **_kwargs: SimpleNamespace(
                chunk_tokens=2_048,
                raw_required_blocks=1_014,
                reserved_blocks=0,
                free_blocks=14,
                reclaim_blocks=1_000,
            )
        )

        target, victims, *_, deficit_blocks, _pages = (
            harness._scheduled_prefill_fit_proactive_snapshot_plan(
                {"short-3": 1_808}, ring_token_cap=2_048
            )
        )

        self.assertEqual(target.request_id, "short-4")
        self.assertEqual(deficit_blocks, 114)
        self.assertEqual(
            [victim.request_id for victim in victims],
            ["short-4", "long-1"],
        )
        self.assertEqual(
            harness._scheduled_prefill_fit_last_avoided_peer_blocks,
            50,
        )

    def test_ring_fallback_separates_hard_reserve_from_decode_runway(
        self,
    ) -> None:
        harness = self._whole_request_planner()
        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 41
        target = harness.requests["short-4"]

        self.assertEqual(
            harness._scheduled_prefill_fit_ring_fallback_reclaim_requirement(
                target,
                next_chunk_tokens=2_048,
            ),
            (128, 41, 155, 114),
        )

        harness._debug_running_phase_counts = lambda: {
            "num_running_decode_reqs": 1
        }
        self.assertEqual(
            harness._scheduled_prefill_fit_ring_fallback_reclaim_requirement(
                target,
                next_chunk_tokens=2_048,
            ),
            (128, 41, 155, 155),
        )

    def test_best_fit_grows_set_only_when_no_single_request_covers(self) -> None:
        candidate = MODULE.ResidencyVictimCandidate
        victims = MODULE.select_best_fit_whole_request_victims(
            [
                candidate("a", 1.0, 1000, 500, 60),
                candidate("b", 2.0, 1000, 500, 50),
                candidate("c", 3.0, 1000, 500, 40),
            ],
            required_blocks=100,
        )
        self.assertEqual([victim.request_id for victim in victims], ["a", "c"])

    def test_best_fit_chooses_smallest_cover_not_srpt_tail(self) -> None:
        candidate = MODULE.ResidencyVictimCandidate
        victims = MODULE.select_best_fit_whole_request_victims(
            [
                candidate("large", 1.0, 1000, 500, 256),
                candidate("best", 2.0, 1000, 500, 120),
                candidate("small", 3.0, 1000, 500, 80),
            ],
            required_blocks=114,
        )
        self.assertEqual([victim.request_id for victim in victims], ["best"])

    def test_pd_phase_cap_never_crosses_prompt_boundary(self) -> None:
        harness = load_control_guard_harness()()
        harness.pd_decoupled_no_mix = False
        request = SimpleNamespace(
            request_id="request",
            num_prompt_tokens=100,
            num_tokens=101,
            virtual_context_deferred_state="",
        )
        self.assertEqual(
            harness._cap_pd_phase_tokens(
                request,
                8,
                num_computed_tokens=97,
                selected_phase="prefill",
            ),
            3,
        )
        self.assertEqual(
            harness._cap_pd_phase_tokens(
                request,
                1,
                num_computed_tokens=99,
                selected_phase="decode",
            ),
            0,
        )
        self.assertEqual(
            harness._cap_pd_phase_tokens(
                request,
                1,
                num_computed_tokens=100,
                selected_phase="decode",
            ),
            1,
        )

    def test_decode_reset_history_replays_on_prefill_lane(self) -> None:
        harness = load_control_guard_harness()()
        harness.pd_decoupled_no_mix = True
        harness.pd_phase_local_model_runner_enabled = True
        request = SimpleNamespace(
            request_id="long-2",
            num_prompt_tokens=50_000,
            num_computed_tokens=0,
            num_tokens=57_329,
            num_output_tokens=7_329,
            pd_recompute_frontier_tokens=57_328,
        )
        harness.requests = {request.request_id: request}

        self.assertEqual(
            harness._pd_prefill_lane_remaining_tokens(request),
            57_328,
        )
        self.assertEqual(
            harness._cap_pd_phase_tokens(
                request,
                57_329,
                num_computed_tokens=0,
                selected_phase="prefill",
            ),
            57_328,
        )
        request.num_computed_tokens = 50_000
        self.assertEqual(
            harness._classify_scheduled_phase({request.request_id: 2_048}),
            "prefill",
        )
        self.assertEqual(
            harness._cap_pd_phase_tokens(
                request,
                7_329,
                num_computed_tokens=request.num_computed_tokens,
                selected_phase="decode",
            ),
            0,
        )

        request.num_computed_tokens = 57_328
        self.assertFalse(harness._debug_has_prefill_remaining(request))
        self.assertEqual(
            harness._classify_scheduled_phase({request.request_id: 1}),
            "decode",
        )
        harness._validate_pd_phase_schedule(
            "decode", {request.request_id: 1}
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "exactly one live token",
        ):
            harness._validate_pd_phase_schedule(
                "decode", {request.request_id: 7_330}
            )

    def test_live_decode_residency_is_only_protected_in_pd_mode(self) -> None:
        harness = load_control_guard_harness()()
        harness._pd_pending_request_ids = set()
        request = SimpleNamespace(
            request_id="long-2",
            num_output_tokens=7_329,
            pd_output_placeholder_ledger=SimpleNamespace(active=None),
            kv_retention_decode_continuation_lease_active=False,
            is_finished=lambda: False,
        )

        harness.pd_decoupled_no_mix = True
        self.assertEqual(
            harness._pd_preemption_protection_reason(request),
            "live_decode_residency",
        )
        harness.pd_decoupled_no_mix = False
        self.assertEqual(
            harness._pd_preemption_protection_reason(request),
            "",
        )

    def test_pending_guard_blocks_every_unrecovered_future_owner(self) -> None:
        harness = load_control_guard_harness()()
        harness.enable_ttft_prefill_chunk_scheduling = True
        harness._pd_pending_request_ids = {"request"}
        harness._pd_pending_phase_request_ids = {"prefill": {"request"}}
        harness._pd_pending_schedule_block_count = 0
        harness._pd_pending_prefill_chain_allow_count = 0
        harness._pd_pending_schedule_last_reason = ""
        harness._tiered_execution_single_flight_block_count = 0
        harness._tiered_execution_single_flight_last_request_id = ""
        harness._tiered_execution_single_flight_last_generation = 0
        harness._debug_has_prefill_remaining = lambda _request: True
        request = SimpleNamespace(
            request_id="request",
            pd_output_placeholder_ledger=SimpleNamespace(active=None),
            kv_retention_execution_plan_generation=0,
            kv_retention_execution_plan_acked_generation=0,
        )

        self.assertTrue(harness.blocks_next_schedule(request, "prefill"))
        self.assertEqual(harness._pd_pending_prefill_chain_allow_count, 0)
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "pending_prefill_future_owner",
        )
        self.assertTrue(harness.blocks_next_schedule(request, "decode"))
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "decode_output_dependency",
        )

        harness._pd_pending_phase_request_ids["prefill_final"] = {"request"}
        self.assertTrue(harness.blocks_next_schedule(request, "prefill"))
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "final_prefill_output_dependency",
        )

        harness._pd_pending_phase_request_ids = {"prefill": {"request"}}
        request.kv_retention_execution_plan_generation = 7
        request.kv_retention_execution_plan_acked_generation = 6
        self.assertTrue(harness.blocks_next_schedule(request, "prefill"))
        self.assertEqual(
            harness._pd_pending_schedule_last_reason,
            "tiered_execution_single_flight",
        )
        self.assertEqual(
            harness._tiered_execution_single_flight_block_count,
            1,
        )
        self.assertEqual(
            harness._tiered_execution_single_flight_last_request_id,
            "request",
        )
        self.assertEqual(
            harness._tiered_execution_single_flight_last_generation,
            7,
        )

        request.kv_retention_execution_plan_acked_generation = 7
        harness._pd_pending_request_ids.clear()
        self.assertFalse(harness.blocks_next_schedule(request, "decode"))
        self.assertEqual(harness._pd_pending_schedule_last_reason, "no_pending")

    def test_phase_owner_overlap_never_reuses_a_post_commit_slot(self) -> None:
        harness = load_engine_prefill_chain_harness()()
        harness._ttft_prefill_suspended = {}

        def output(
            *,
            request_id: str = "request",
            slot: int = 0,
            mode: str = "native",
            epoch: int = 7,
            final: bool = False,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                num_scheduled_tokens={request_id: 16},
                pd_phase_slot=slot,
                pd_prefill_mode=mode,
                pd_sm_profile_id="d0_p40",
                pd_sm_profile_epoch=epoch,
                pd_ttft_handoff_request_ids=(request_id,) if final else (),
            )

        queued = output(slot=1)
        phase_queues = {"prefill": [(None, queued, None)], "decode": []}
        self.assertTrue(
            harness._pd_pending_phase_owner_overlap(
                phase_queues,
                output(),
                "prefill",
            )
        )

        harness._ttft_prefill_suspended = {
            (0, 1): (None, output(request_id="suspended", slot=0))
        }
        self.assertTrue(
            harness._pd_pending_phase_owner_overlap(
                {"prefill": [], "decode": []},
                output(request_id="suspended", slot=1),
                "prefill",
            )
        )

        phase_queues["prefill"] = [(None, output(slot=1, final=True), None)]
        self.assertTrue(
            harness._pd_pending_phase_owner_overlap(
                phase_queues,
                output(),
                "prefill",
            )
        )

        phase_queues["prefill"] = [(None, output(slot=1, mode="ring"), None)]
        self.assertTrue(
            harness._pd_pending_phase_owner_overlap(
                phase_queues,
                output(),
                "prefill",
            )
        )

        phase_queues["prefill"] = [(None, output(request_id="other"), None)]
        self.assertFalse(
            harness._pd_pending_phase_owner_overlap(
                phase_queues,
                output(),
                "prefill",
            )
        )

        multi_request_output = output()
        multi_request_output.num_scheduled_tokens["fresh"] = 16
        phase_queues["prefill"] = [(None, queued, None)]
        self.assertTrue(
            harness._pd_pending_phase_owner_overlap(
                phase_queues,
                multi_request_output,
                "prefill",
            )
        )

    def test_deferred_descriptor_survives_batch_removal_until_resume(self) -> None:
        store = DESCRIPTOR_MODULE.KVRetentionResidencyDescriptorStore()
        descriptor = store.capture(
            request_id="long-2",
            block_ids=((10, 11, -1),),
            num_computed_tokens=18_432,
        )

        self.assertEqual(descriptor.block_ids, ((10, 11, -1),))
        self.assertIs(store.lookup("long-2"), descriptor)
        self.assertTrue(store.discard("long-2"))
        self.assertIsNone(store.lookup("long-2"))
        self.assertEqual(
            store.metrics(),
            {
                "worker_deferred_residency_descriptor_active_count": 0,
                "worker_deferred_residency_descriptor_capture_count": 1,
                "worker_deferred_residency_descriptor_discard_count": 1,
                "worker_deferred_residency_descriptor_lookup_count": 2,
                "worker_deferred_residency_descriptor_hit_count": 1,
                "worker_deferred_residency_descriptor_miss_count": 1,
            },
        )

    def test_policy_aliases_and_exact_deficit(self) -> None:
        policy = MODULE.ResidencyVictimPolicy
        self.assertEqual(
            policy.parse("scheduled-prefill-fit"),
            policy.SCHEDULED_PREFILL_FIT,
        )
        self.assertEqual(
            policy.parse("next_prefill_fit"),
            policy.SCHEDULED_PREFILL_FIT,
        )
        self.assertEqual(
            MODULE.scheduled_prefill_reclaim_blocks(
                free_blocks=100,
                scheduled_prefill_required_blocks=8,
                reserved_blocks=96,
            ),
            4,
        )
        self.assertEqual(
            MODULE.scheduled_prefill_capacity_headroom_blocks(4095, 0.99),
            41,
        )
        self.assertEqual(
            MODULE.scheduled_prefill_capacity_headroom_blocks(4095, 0.98),
            82,
        )
        self.assertEqual(
            MODULE.scheduled_prefill_capacity_headroom_blocks(4095, 0.95),
            205,
        )
        self.assertEqual(
            MODULE.scheduled_prefill_capacity_headroom_blocks(8191, 0.99),
            82,
        )

    def test_decode_growth_uses_fraction_complement_once(self) -> None:
        claim_type, runway_type = load_decode_growth_runway_types()
        runway = runway_type(
            target_request_id="short-2",
            target_logical_tokens=6144,
            target_chunk_tokens=2048,
            capacity_headroom_blocks=41,
            committed_reclaim_blocks=128,
            created_iteration_id=10,
        )
        claim = claim_type(
            request_id="short-2",
            logical_tokens=6144,
            chunk_tokens=2048,
            raw_required_blocks=128,
            hard_reserved_blocks=0,
            capacity_headroom_blocks=41,
            reserved_blocks=41,
            free_blocks=167,
            reclaim_blocks=2,
        )

        self.assertEqual(claim.hard_reclaim_blocks, 0)
        self.assertTrue(runway.covers(claim))
        self.assertFalse(
            runway.covers(
                claim_type(
                    **{
                        **claim.__dict__,
                        "free_blocks": 127,
                        "reclaim_blocks": 42,
                    }
                )
            )
        )

    def test_active_or_pending_decode_promotes_headroom_to_hard_reserve(
        self,
    ) -> None:
        harness = load_capacity_claim_harness()()
        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
        harness._debug_running_phase_counts = lambda: {
            "num_running_decode_reqs": 0
        }
        harness._pd_pending_decode_active = lambda: False
        self.assertEqual(
            harness._scheduled_prefill_fit_headroom_partition(),
            (0, 82),
        )

        harness._pd_pending_decode_active = lambda: True
        self.assertEqual(
            harness._scheduled_prefill_fit_headroom_partition(),
            (82, 0),
        )

    def test_new_arrival_recomputes_only_current_target(self) -> None:
        # The old target needed eight blocks. A newly arrived SRPT request may
        # become the next target and need only one; no prior multi-chunk future
        # prediction is carried into the next immutable boundary.
        old_target = MODULE.scheduled_prefill_reclaim_blocks(100, 8, 96)
        new_short_target = MODULE.scheduled_prefill_reclaim_blocks(100, 1, 96)
        self.assertEqual(old_target, 4)
        self.assertEqual(new_short_target, 0)

    def test_projected_srpt_reranks_after_current_chunk(self) -> None:
        candidate = MODULE.ProjectedPrefillCandidate
        order = MODULE.projected_srpt_prefill_order(
            [
                candidate("long", 1.0, 50_000, 32_000),
                candidate("short", 2.0, 10_000, 0),
            ]
        )
        self.assertEqual(order, ("short", "long"))
        self.assertEqual(tuple(reversed(order)), ("long", "short"))

    def test_zero_physical_growth_never_triggers_victim(self) -> None:
        self.assertEqual(
            MODULE.scheduled_prefill_reclaim_blocks(0, 0, 2),
            0,
        )

    def test_physical_release_preserves_non_victim_pages(self) -> None:
        class Block:
            def __init__(self, block_id: int, *, is_null: bool = False):
                self.block_id = block_id
                self.is_null = is_null

        class Pool:
            def __init__(self) -> None:
                self.freed: list[int] = []

            def free_blocks(self, blocks) -> None:
                self.freed.extend(block.block_id for block in blocks)

        harness_type = load_release_harness()
        harness = harness_type()
        harness._null_block = Block(-1, is_null=True)
        harness.block_pool = Pool()
        harness.req_to_blocks = {
            "request": [Block(10), Block(11), Block(12)]
        }

        self.assertFalse(
            harness.validate_hbm_release_pages("request", {0: 999})
        )
        self.assertEqual(
            harness.release_hbm_pages("request", {0: 10}),
            1,
        )
        blocks = harness.req_to_blocks["request"]
        self.assertTrue(blocks[0].is_null)
        self.assertEqual([block.block_id for block in blocks[1:]], [11, 12])
        self.assertEqual(harness.block_pool.freed, [10])

    def test_partial_residency_resume_is_allocator_state(self) -> None:
        class Block:
            def __init__(self, block_id: int, *, is_null: bool = False):
                self.block_id = block_id
                self.is_null = is_null

        harness_type = load_release_harness()
        harness = harness_type()
        harness.block_size = 16
        harness.kv_cache_group_id = 0
        harness.partial_residency_requests = set()
        harness.req_to_blocks = {
            "request": [
                Block(10),
                Block(-1, is_null=True),
                Block(12),
            ]
        }

        harness.enter_partial_residency("request")
        self.assertTrue(harness.has_partial_residency_pages("request"))
        self.assertEqual(
            harness.resume_partial_residency("request", 48),
            (2, 1),
        )
        harness.req_to_blocks["request"][1] = Block(11)
        self.assertEqual(
            harness.resume_partial_residency("request", 48),
            (3, 0),
        )
        harness.clear_partial_residency("request")
        self.assertFalse(harness.has_partial_residency_pages("request"))

    def test_complete_reload_exits_partial_allocator_state(self) -> None:
        class Block:
            def __init__(self, block_id: int, *, is_null: bool = False):
                self.block_id = block_id
                self.is_null = is_null

        class Pool:
            @staticmethod
            def get_num_free_blocks() -> int:
                return 1

            @staticmethod
            def get_new_blocks(count: int) -> list[Block]:
                self.assertEqual(count, 1)
                return [Block(11)]

        harness = load_release_harness()()
        harness.block_size = 16
        harness.kv_cache_group_id = 0
        harness.partial_residency_requests = {"request"}
        harness.req_to_blocks = {
            "request": [Block(10), Block(-1, is_null=True)]
        }
        harness.num_cached_block = {"request": 1}
        harness.block_pool = Pool()

        self.assertEqual(
            harness.allocate_hbm_reload_blocks("request", {1}),
            {1: 11},
        )
        self.assertNotIn("request", harness.partial_residency_requests)
        self.assertEqual(
            harness.get_retained_residency_counts("request", 32),
            (2, 0),
        )

    def test_proactive_d2h_plan_contains_only_exact_victim_pages(self) -> None:
        class BlockTable:
            shape = (1, 4)

            def __getitem__(self, index):
                row, column = index
                self.assert_valid_row(row)
                return (10, 11, 12, 13)[column]

            @staticmethod
            def assert_valid_row(row: int) -> None:
                if row != 0:
                    raise IndexError(row)

        plan = D2H_MODULE.build_tiered_d2h_source_plan(
            request_ids=["request"],
            block_table=BlockTable(),
            seq_lens=[64],
            query_start_loc=[0, 16],
            block_size=16,
            copy_policy="scheduled",
            allowed_request_ids={"request"},
            force_all_request_ids={"request"},
            max_pages=0,
            exact_force_pages_by_request={"request": {0, 2}},
        )
        self.assertEqual(
            [(page.logical_page, page.block_id) for page in plan.pages],
            [(0, 10), (2, 12)],
        )

    def test_snapshot_preserves_mutable_tail_and_selects_oldest_resident_pages(
        self,
    ) -> None:
        block = lambda is_null=False: SimpleNamespace(is_null=is_null)
        harness = load_snapshot_planner_harness()()
        harness.hash_block_size = 16
        harness.coordinator = SimpleNamespace(
            get_blocks=lambda _request_id: [[
                block(),
                block(is_null=True),
                block(),
                block(),
                block(),
            ]]
        )

        self.assertEqual(
            harness.plan_virtual_context_hbm_snapshot_pages(
                request_id="long",
                logical_tokens=80,
                max_pages=2,
                group_id=0,
            ),
            (0, 2),
        )

    def test_capacity_claim_reserves_full_prefill_and_executes_capped_chunk(
        self,
    ) -> None:
        captured: dict[str, int] = {}

        class Manager:
            block_pool = SimpleNamespace(get_num_free_blocks=lambda: 122)

            @staticmethod
            def estimate_num_blocks_to_allocate(
                _request,
                chunk_tokens,
                **kwargs,
            ) -> int:
                captured["allocation_tokens"] = chunk_tokens
                captured.update(kwargs)
                return 64

        harness = load_capacity_claim_harness()()
        harness.max_num_scheduled_tokens = 16_384
        harness.scheduler_config = SimpleNamespace(
            long_prefill_token_threshold=0
        )
        harness.enable_ttft_prefill_chunk_scheduling = True
        harness._ttft_cap_selected_prefill_chunk = (
            lambda _request,
            tokens,
            num_computed_tokens,
            execution_mode: min(tokens, 16_384 if execution_mode == "native" else 1024)
        )
        harness._virtual_context_running_decode_reload_required_page_count = (
            lambda: 0
        )
        harness._virtual_context_reload_required_page_count = (
            lambda _request_ids, _phase="decode": 0
        )
        harness.virtual_context_decode_hotset_first = False
        harness.virtual_context_prefill_ring_tokens_with_decode_cold = 0
        harness.virtual_context_staging_reserved_blocks = 8
        harness.virtual_context_ring_min_free_blocks = 8
        harness.virtual_context_decode_hotset_reserve_blocks = 16
        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 0
        harness._debug_running_phase_counts = lambda: {
            "num_running_decode_reqs": 0
        }
        harness._pd_pending_decode_active = lambda: False
        harness._scheduled_prefill_fit_decode_growth_runway = None
        harness.num_lookahead_tokens = 0
        harness.kv_cache_manager = Manager()
        request = SimpleNamespace(
            request_id="long",
            num_prompt_tokens=50_000,
            num_computed_tokens=0,
            virtual_context_deferred_logical_computed_tokens=18_432,
        )

        claim = harness._scheduled_prefill_fit_capacity_claim(
            request,
            ring_token_cap=1024,
            logical_tokens=19_456,
            num_new_computed_tokens=1024,
        )
        self.assertEqual(claim.chunk_tokens, 16_384)
        self.assertEqual(claim.required_blocks_with_reserve, 96)
        self.assertEqual(claim.reclaim_blocks, 0)
        self.assertEqual(captured["num_new_computed_tokens"], 1024)
        self.assertEqual(captured["num_external_computed_tokens"], 18_432)
        self.assertEqual(captured["allocation_tokens"], 30_544)

        ring_claim = harness._scheduled_prefill_fit_capacity_claim(
            request,
            ring_token_cap=1024,
            logical_tokens=19_456,
            num_new_computed_tokens=1024,
            full_native=False,
        )
        self.assertEqual(ring_claim.chunk_tokens, 1024)
        self.assertEqual(captured["allocation_tokens"], 1024)

        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
        harness._debug_running_phase_counts = lambda: {
            "num_running_decode_reqs": 1
        }
        active_decode_claim = harness._scheduled_prefill_fit_capacity_claim(
            request,
            ring_token_cap=1024,
            logical_tokens=19_456,
            num_new_computed_tokens=1024,
        )
        self.assertEqual(active_decode_claim.hard_reserved_blocks, 114)
        self.assertEqual(active_decode_claim.capacity_headroom_blocks, 0)
        self.assertEqual(active_decode_claim.reserved_blocks, 114)
        self.assertEqual(active_decode_claim.hard_reclaim_blocks, 56)

        harness.virtual_context_scheduled_prefill_fit_headroom_blocks = 0
        harness._debug_running_phase_counts = lambda: {
            "num_running_decode_reqs": 0
        }

        harness.virtual_context_residency_victim_policy = (
            MODULE.ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        )
        harness.virtual_context_decode_hotset_first = True
        harness._debug_has_prefill_remaining = lambda _request: True
        harness._virtual_context_reload_required_page_count = (
            lambda _request_ids, _phase="prefill": 0
        )
        harness._dynamic_virtual_context_ring_prefill_admission_tokens = (
            lambda: 1024
        )
        harness._virtual_context_residency_fixed_ring_chunk_tokens = (
            lambda: 1024
        )
        native_safe, *_, reason = (
            harness._prefill_native_safe_before_ring_pressure(request)
        )
        self.assertTrue(native_safe)
        self.assertEqual(reason, "scheduled_prefill_fit_full_prefill_available")

    def test_native_execution_cap_is_independent_from_ring_quantum(self) -> None:
        class ChunkPolicy:
            quantum_tokens = 2_048

            @staticmethod
            def cap_prefill_tokens(
                request,
                num_new_tokens,
                *,
                num_computed_tokens=None,
            ) -> int:
                progress = (
                    request.num_computed_tokens
                    if num_computed_tokens is None
                    else num_computed_tokens
                )
                return min(
                    num_new_tokens,
                    2_048,
                    request.num_prompt_tokens - progress,
                )

            @staticmethod
            def logical_prefill_progress(request) -> int:
                return max(
                    request.num_computed_tokens,
                    request.virtual_context_deferred_logical_computed_tokens,
                )

            @staticmethod
            def logical_prefill_boundary(request) -> int:
                return request.num_prompt_tokens

        harness = load_ttft_chunk_cap_harness()()
        harness.enable_ttft_prefill_chunk_scheduling = True
        harness.ttft_prefill_native_chunk_max_tokens = 16_384
        harness.ttft_prefill_chunk_scheduler = ChunkPolicy()
        request = SimpleNamespace(
            request_id="long",
            num_prompt_tokens=50_000,
            num_computed_tokens=0,
            virtual_context_deferred_logical_computed_tokens=0,
        )

        self.assertEqual(
            harness._ttft_cap_selected_prefill_chunk(
                request,
                16_384,
                execution_mode="native",
            ),
            16_384,
        )
        self.assertEqual(
            harness._ttft_cap_selected_prefill_chunk(
                request,
                16_384,
                execution_mode="ring",
            ),
            2_048,
        )

    def test_preparation_waits_until_every_exact_page_is_ready(self) -> None:
        harness = load_preparation_harness()()
        harness._scheduled_prefill_fit_preparation = SimpleNamespace(
            target_request_id="short",
            victim_request_id="long",
            victim_request_ids=("long",),
            snapshot_pages=(("long", 0, 0), ("long", 0, 1)),
        )
        harness._scheduled_prefill_fit_preparation_cancelled_count = 0
        harness._scheduled_prefill_fit_preparation_target_mismatch_count = 0
        harness._scheduled_prefill_fit_unprepared_reclaim_count = 0
        victim = SimpleNamespace(request_id="long")
        harness.requests = {"long": victim}
        harness.running = [victim]
        harness._pd_pending_request_ids = set()
        harness._debug_has_prefill_remaining = lambda _request: True
        ready_pages = [0]
        harness.kv_cache_manager = SimpleNamespace(
            plan_virtual_context_hbm_release_candidates=lambda **_kwargs: [
                SimpleNamespace(
                    request_id="long",
                    kv_cache_group_id=0,
                    logical_token_page=page,
                )
                for page in ready_pages
            ]
        )

        _preparation, ready_count, planned_count, ready = (
            harness._scheduled_prefill_fit_preparation_status(
                live_target_request_id="short",
                required_blocks=2,
            )
        )
        self.assertEqual((ready_count, planned_count, ready), (1, 2, False))

        ready_pages.append(1)
        harness._pd_pending_request_ids.add("long")
        _preparation, ready_count, planned_count, ready = (
            harness._scheduled_prefill_fit_preparation_status(
                live_target_request_id="short",
                required_blocks=2,
            )
        )
        self.assertEqual((ready_count, planned_count, ready), (2, 2, False))

        harness._pd_pending_request_ids.clear()
        _preparation, ready_count, planned_count, ready = (
            harness._scheduled_prefill_fit_preparation_status(
                live_target_request_id="short",
                required_blocks=2,
            )
        )
        self.assertEqual((ready_count, planned_count, ready), (2, 2, True))

    def test_insufficient_old_preparation_is_cancelled(self) -> None:
        harness = load_preparation_harness()()
        harness._scheduled_prefill_fit_preparation = SimpleNamespace(
            target_request_id="old",
            victim_request_id="long",
            victim_request_ids=("long",),
            snapshot_pages=(("long", 0, 0),),
        )
        harness._scheduled_prefill_fit_preparation_cancelled_count = 0
        harness._scheduled_prefill_fit_preparation_target_mismatch_count = 0
        harness._scheduled_prefill_fit_unprepared_reclaim_count = 0
        victim = SimpleNamespace(request_id="long")
        harness.requests = {"long": victim}
        harness.running = [victim]
        harness._pd_pending_request_ids = set()
        harness._debug_has_prefill_remaining = lambda _request: True
        harness.kv_cache_manager = SimpleNamespace(
            plan_virtual_context_hbm_release_candidates=lambda **_kwargs: []
        )

        preparation, _ready_count, planned_count, ready = (
            harness._scheduled_prefill_fit_preparation_status(
                live_target_request_id="new",
                required_blocks=2,
            )
        )
        self.assertIsNone(preparation)
        self.assertEqual(planned_count, 1)
        self.assertFalse(ready)
        self.assertIsNone(harness._scheduled_prefill_fit_preparation)
        self.assertEqual(
            harness._scheduled_prefill_fit_preparation_cancelled_count,
            1,
        )

    def test_soft_headroom_drift_keeps_exact_preparation(self) -> None:
        harness = load_preparation_harness()()
        harness._scheduled_prefill_fit_preparation = SimpleNamespace(
            target_request_id="short",
            victim_request_id="long",
            victim_request_ids=("long",),
            snapshot_pages=(("long", 0, 0),),
        )
        harness._scheduled_prefill_fit_preparation_cancelled_count = 0
        harness._scheduled_prefill_fit_preparation_target_mismatch_count = 0
        harness._scheduled_prefill_fit_unprepared_reclaim_count = 0
        victim = SimpleNamespace(request_id="long")
        harness.requests = {"long": victim}
        harness.running = [victim]
        harness._pd_pending_request_ids = set()
        harness._debug_has_prefill_remaining = lambda _request: True
        harness.kv_cache_manager = SimpleNamespace(
            plan_virtual_context_hbm_release_candidates=lambda **_kwargs: [
                SimpleNamespace(
                    request_id="long",
                    kv_cache_group_id=0,
                    logical_token_page=0,
                )
            ]
        )

        preparation, ready_count, planned_count, ready = (
            harness._scheduled_prefill_fit_preparation_status(
                live_target_request_id="short",
                required_blocks=2,
                hard_required_blocks=0,
            )
        )

        self.assertIsNotNone(preparation)
        self.assertEqual((ready_count, planned_count, ready), (1, 1, True))
        self.assertEqual(
            harness._scheduled_prefill_fit_preparation_cancelled_count,
            0,
        )

    def test_target_change_cannot_consume_soft_headroom(self) -> None:
        harness = load_preparation_harness()()
        harness._scheduled_prefill_fit_preparation = SimpleNamespace(
            target_request_id="old-short",
            victim_request_id="long",
            victim_request_ids=("long",),
            snapshot_pages=(("long", 0, 0),),
        )
        harness._scheduled_prefill_fit_preparation_cancelled_count = 0
        harness._scheduled_prefill_fit_preparation_target_mismatch_count = 0
        harness._scheduled_prefill_fit_unprepared_reclaim_count = 0
        victim = SimpleNamespace(request_id="long")
        harness.requests = {"long": victim}
        harness.running = [victim]
        harness._pd_pending_request_ids = set()
        harness._debug_has_prefill_remaining = lambda _request: True
        harness.kv_cache_manager = SimpleNamespace(
            plan_virtual_context_hbm_release_candidates=lambda **_kwargs: []
        )

        preparation, _ready_count, _planned_count, ready = (
            harness._scheduled_prefill_fit_preparation_status(
                live_target_request_id="new-short",
                required_blocks=2,
                hard_required_blocks=0,
            )
        )

        self.assertIsNone(preparation)
        self.assertFalse(ready)
        self.assertEqual(
            harness._scheduled_prefill_fit_preparation_cancelled_count,
            1,
        )


class ScheduledPrefillFitSourceContractTest(unittest.TestCase):
    def test_decode_replay_cannot_claim_a_second_first_token(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        handoff_body = source.split(
            "pd_ttft_handoff_request_ids: list[str] = []", 1
        )[1].split("if self.enable_ttft_prefill_chunk_scheduling:", 1)[0]
        self.assertIn("request.num_output_tokens == 0", handoff_body)
        self.assertIn("pd_recompute_frontier_tokens", source)
        self.assertIn('return "live_decode_residency"', source)
        self.assertIn('"live_decode_growth_wait"', source)
        self.assertIn('"live_decode_growth"', source)
        self.assertIn(
            "pd_capacity_retry_wake_on_output_progress = (", source
        )

    def test_scheduler_recomputes_target_and_requests_whole_commit(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        for contract in (
            "def _scheduled_prefill_fit_target(",
            "def _scheduled_prefill_fit_capacity_claim(",
            "if ring_token_cap > 0 and not full_native:",
            "def _ttft_prefill_execution_mode(",
            "and self.ttft_prefill_native_chunk_max_tokens <= 0",
            "ScheduledPrefillCapacityClaim",
            "def _scheduled_prefill_fit_proactive_snapshot_plan(",
            "self._ttft_begin_prefill_chunk_iteration()",
            "scheduled_prefill_fit_target_reclaim_blocks",
            "scheduled_prefill_capacity_headroom_blocks",
            "hard_required_blocks=",
            "commit_required_blocks = min(",
            "tiered_proactive_snapshot_pages=tuple(",
            "max_reclaim_blocks=(",
            "ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT",
            "released_logical_blocks",
            "retained_virtual_block_table",
            "def _scheduled_prefill_fit_preparation_status(",
            "scheduled_prefill_fit_d2h_wait",
            "projected_srpt_prefill_order(",
            "prepared_request_ids=",
            "select_best_fit_whole_request_victims(",
            "protected_request_ids = set(num_scheduled_tokens)",
            "excluded_request_ids=(",
            "tiered_proactive_snapshot_logical_tokens=tuple(",
            "scheduled_prefill_fit_full_prefill_available",
            "scheduled_prefill_fit_prefill_lane_reclaim",
            "ScheduledPrefillFitDecodeGrowthRunway",
            "scheduled_prefill_fit_decode_growth_runway",
            "def _active_virtual_context_prefill_requests(",
            "request_already_waiting=",
            "def _virtual_context_native_decode_handoff_requirement(",
            "reload_allocator_plan_mismatch",
            "mixed_residency_backend_unavailable",
            "whole-request",
            "max_reclaim_blocks=None",
            "whole_request_reclaimed_blocks",
        ):
            self.assertIn(contract, source)

        tree = ast.parse(source)
        scheduler = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
        )
        methods = {
            node.name: ast.get_source_segment(source, node) or ""
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef)
        }
        proactive = methods["_scheduled_prefill_fit_proactive_snapshot_plan"]
        reclaim_limit = methods["_scheduled_prefill_fit_reclaim_limit"]
        self.assertIn("select_best_fit_whole_request_victims(", proactive)
        self.assertIn("max_pages=logical_page_count", proactive)
        self.assertIn("victim.request_id in self._pd_pending_request_ids", proactive)
        self.assertNotIn("max_pages=remaining_deficit", proactive)
        self.assertIn("return 0", reclaim_limit)

    def test_parked_partial_resume_does_not_depend_on_prefix_cache_state(self) -> None:
        single_source = SINGLE_MANAGER_PATH.read_text(encoding="utf-8")
        coordinator_source = (
            ROOT / "vendor_v8/vllm/v1/core/kv_cache_coordinator.py"
        ).read_text(encoding="utf-8")
        for contract in (
            "self.partial_residency_requests: set[str]",
            "def enter_partial_residency(",
            "def resume_partial_residency(",
            "def get_retained_residency_counts(",
            "def get_num_hbm_reload_blocks_to_allocate(",
            "or request_id in self.partial_residency_requests",
            "self.clear_partial_residency(request_id)",
        ):
            self.assertIn(contract, single_source)

    def test_phase_and_native_promotion_controls_are_wired(self) -> None:
        single_source = SINGLE_MANAGER_PATH.read_text(encoding="utf-8")
        coordinator_source = (
            ROOT / "vendor_v8/vllm/v1/core/kv_cache_coordinator.py"
        ).read_text(encoding="utf-8")
        scheduler_source = SCHEDULER_PATH.read_text(encoding="utf-8")
        output_source = (
            ROOT / "vendor_v8/vllm/v1/core/sched/output.py"
        ).read_text(encoding="utf-8")
        runner_source = (
            ROOT / "vendor_v8/vllm/v1/worker/gpu_model_runner.py"
        ).read_text(encoding="utf-8")
        engine_source = (
            ROOT / "vendor_v8/vllm/v1/engine/core.py"
        ).read_text(encoding="utf-8")
        capacity_retry_source = (
            ROOT / "vendor_v8/vllm/v1/engine/pd_capacity_retry.py"
        ).read_text(encoding="utf-8")
        validation_runner_source = (
            ROOT / "run_vendor_v8_short_validation.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            "def _cap_pd_phase_tokens(",
            "def blocks_next_schedule(",
            "def pd_prefill_no_progress_retry_metadata(",
            "Pending final-Prefill output remains a Prefill owner",
            "pd_phase_classification_corrected",
            "def _prepare_virtual_context_native_prefill_promotion(",
            "target_self_victim",
            '"hbm_reload_pending",',
        ):
            self.assertIn(contract, scheduler_source)
        self.assertIn(
            "tiered_native_hbm_reload_request_ids",
            output_source,
        )
        self.assertIn(
            "pd_capacity_retry_wake_on_output_progress",
            output_source,
        )
        self.assertIn("pd_capacity_retry_cause", output_source)
        self.assertIn(
            "_pd_capacity_retry_gate.notify_output_progress(",
            engine_source,
        )
        self.assertIn("_notify_pd_engine_progress()", engine_source)
        self.assertIn(
            "def notify_output_progress(",
            capacity_retry_source,
        )
        self.assertIn("def blocking_cause(", capacity_retry_source)
        self.assertIn('== "prefill_kv_capacity"', engine_source)
        self.assertIn(
            "capacity_serialized=prefill_capacity_serialized",
            engine_source,
        )
        prefill_filter_index = scheduler_source.index(
            "Pending final-Prefill output remains a Prefill owner"
        )
        pending_dependency_index = scheduler_source.index(
            "if self.blocks_next_schedule(",
            prefill_filter_index,
        )
        self.assertLess(prefill_filter_index, pending_dependency_index)
        self.assertIn(
            '"native_hbm_reload_request_ids"',
            runner_source,
        )
        self.assertNotIn("_cap_prefill_host_lease_tokens", scheduler_source)
        self.assertNotIn("observe_prefill_execution", engine_source)
        self.assertIn("pending_prefill_future_owner", scheduler_source)
        self.assertIn(
            "Detect a post-commit overlap with an existing phase owner",
            engine_source,
        )
        self.assertNotIn(
            "VLLM_PD_PREFILL_MAX_HOST_LEASE_MS",
            validation_runner_source,
        )
        self.assertGreaterEqual(
            single_source.count(
                "or request_id in self.partial_residency_requests"
            ),
            2,
        )
        for contract in (
            "manager.enter_partial_residency(request_id)",
            "manager.resume_partial_residency(",
            "manager.get_retained_residency_counts(",
            "def is_partial_residency(",
        ):
            self.assertIn(contract, coordinator_source)

    def test_parked_partial_release_is_mapping_checked(self) -> None:
        kv_source = KV_MANAGER_PATH.read_text(encoding="utf-8")
        single_source = SINGLE_MANAGER_PATH.read_text(encoding="utf-8")
        virtual_source = VIRTUAL_CONTEXT_PATH.read_text(encoding="utf-8")
        for contract in (
            "def release_virtual_context_hbm_pages(",
            "planned_keys",
            "marked_keys",
            "self.coordinator.release_hbm_pages(",
        ):
            self.assertIn(contract, kv_source)
        for contract in (
            "def validate_hbm_release_pages(",
            "req_blocks[page] = self._null_block",
            "self.block_pool.free_blocks(released_blocks)",
        ):
            self.assertIn(contract, single_source)
        self.assertIn("max_logical_pages: int = 0", virtual_source)
        self.assertIn("selected_page_keys", virtual_source)

    def test_exact_page_filter_reaches_source_plan_and_fallback(self) -> None:
        d2h_source = D2H_PLAN_PATH.read_text(encoding="utf-8")
        provider_source = PROVIDER_PATH.read_text(encoding="utf-8")
        worker_source = (
            ROOT / "vendor_v8/vllm/v1/worker/gpu_model_runner.py"
        ).read_text(encoding="utf-8")
        for source in (d2h_source, provider_source, worker_source):
            self.assertIn("exact_force_pages_by_request", source)
        self.assertIn("tiered_proactive_snapshot_pages", worker_source)
        for contract in (
            "def _register_batch_external_tiered_snapshots(",
            "proactive_snapshot_logical_tokens",
            "block_ids_by_group[group_id]",
            "KVRetentionResidencyDescriptorStore",
            "worker_batch_external_d2h_descriptor_hit_count",
            "register_hbm_kv_cache_pages(",
        ):
            self.assertIn(contract, worker_source)

    def test_tiered_execution_plan_is_request_single_flight(self) -> None:
        scheduler_source = SCHEDULER_PATH.read_text(encoding="utf-8")
        build_body = scheduler_source.split(
            "def _build_tiered_request_execution_plans(", 1
        )[1].split("\n    def ", 1)[0]
        update_body = scheduler_source.split(
            "def update_from_output(", 1
        )[1].split("\n    def ", 1)[0]

        self.assertIn(
            "Tiered execution single-flight rejected a second plan",
            build_body,
        )
        self.assertIn(
            '"tiered_execution_single_flight"',
            scheduler_source,
        )
        self.assertLess(
            update_body.index(
                "request.kv_retention_execution_plan_acked_generation"
            ),
            update_body.index(
                "if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids"
            ),
        )

    def test_metrics_expose_prediction_changes_and_actual_release(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        string_literals = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        for metric in (
            "virtual_context_scheduled_prefill_fit_target_request_id",
            "virtual_context_scheduled_prefill_fit_target_chunk_tokens",
            "virtual_context_scheduled_prefill_fit_target_reclaim_blocks",
            "virtual_context_scheduled_prefill_fit_capacity_fraction",
            "virtual_context_scheduled_prefill_fit_headroom_blocks",
            (
                "virtual_context_scheduled_prefill_fit_"
                "active_decode_hard_headroom"
            ),
            "virtual_context_scheduled_prefill_fit_hard_headroom_blocks",
            "virtual_context_scheduled_prefill_fit_soft_headroom_blocks",
            "virtual_context_scheduled_prefill_fit_target_hard_reclaim_blocks",
            "virtual_context_scheduled_prefill_fit_commit_reclaim_blocks",
            "virtual_context_scheduled_prefill_fit_soft_headroom_shortfall_blocks",
            "virtual_context_scheduled_prefill_fit_partial_release_blocks",
            "virtual_context_scheduled_prefill_fit_whole_request_mode",
            (
                "virtual_context_scheduled_prefill_fit_"
                "whole_request_victim_count"
            ),
            (
                "virtual_context_scheduled_prefill_fit_"
                "whole_request_reclaimed_blocks"
            ),
            "virtual_context_scheduled_prefill_fit_target_change_count",
            "virtual_context_scheduled_prefill_fit_proactive_next_chunk_tokens",
            "virtual_context_scheduled_prefill_fit_proactive_deficit_blocks",
            "virtual_context_scheduled_prefill_fit_proactive_plan_mode",
            (
                "virtual_context_scheduled_prefill_fit_"
                "proactive_native_deficit_blocks"
            ),
            (
                "virtual_context_scheduled_prefill_fit_"
                "proactive_ring_fallback_deficit_blocks"
            ),
            (
                "virtual_context_scheduled_prefill_fit_"
                "proactive_avoided_peer_blocks"
            ),
            "virtual_context_scheduled_prefill_fit_proactive_snapshot_page_count",
            "virtual_context_scheduled_prefill_fit_proactive_victim_request_id",
            "virtual_context_scheduled_prefill_fit_proactive_victim_request_ids",
            "virtual_context_scheduled_prefill_fit_proactive_external_victim_count",
            "virtual_context_scheduled_prefill_fit_proactive_waiting_victim_count",
            "virtual_context_scheduled_prefill_fit_proactive_decode_handoff",
            "virtual_context_scheduled_prefill_fit_preparation_waiting",
            "virtual_context_scheduled_prefill_fit_preparation_victim_request_ids",
            "virtual_context_scheduled_prefill_fit_preparation_ready_page_count",
            "virtual_context_scheduled_prefill_fit_protected_victim_skip_count",
            "virtual_context_scheduled_prefill_fit_allocator_resume_count",
            "virtual_context_scheduled_prefill_fit_decode_growth_runway_active",
            (
                "virtual_context_scheduled_prefill_fit_"
                "decode_growth_runway_absorbed_blocks"
            ),
            (
                "virtual_context_scheduled_prefill_fit_"
                "decode_growth_runway_admission_count"
            ),
        ):
            self.assertIn(metric, string_literals)

    def test_parked_partial_victims_keep_version_reconciled_handoff(
        self,
    ) -> None:
        scheduler_source = SCHEDULER_PATH.read_text(encoding="utf-8")
        provider_source = PROVIDER_PATH.read_text(encoding="utf-8")
        provider_tree = ast.parse(provider_source)
        collections_imports = {
            alias.asname or alias.name
            for node in provider_tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "collections"
            for alias in node.names
        }
        self.assertIn("defaultdict", collections_imports)
        for contract in (
            "_try_defer_running_prefill_to_tiered_provider(",
            "release_virtual_context_hbm_pages(",
            "max_reclaim_blocks=max_reclaim_blocks",
        ):
            self.assertIn(contract, scheduler_source)
        for contract in (
            "def _mark_hbm_registration_complete(",
            "allow_newer_version=True",
            "provider_versions=(",
        ):
            self.assertIn(contract, provider_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
