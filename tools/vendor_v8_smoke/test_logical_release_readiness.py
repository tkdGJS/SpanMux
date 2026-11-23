#!/usr/bin/env python3
"""CPU projection/ownership regressions; optional descriptive query benchmark."""
from __future__ import annotations

import __future__
import ast
import importlib.util
import itertools
import random
import statistics
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCOPED = load("logical_readiness_scoped",
              "tools/vendor_v8_smoke/test_request_scoped_page_planning.py")
V = SCOPED.V
POLICY = load("logical_readiness_policy",
              "vendor_v8/vllm/v1/core/sched/residency_policy.py")


def methods(relative, class_name, names, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == class_name)
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names)
    code = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name="Harness", bases=[], keywords=[], body=body, decorator_list=[])],
        type_ignores=[]))
    exec(compile(code, str(path), "exec", flags=__future__.annotations.compiler_flag),
         namespace)
    return namespace["Harness"]


Manager = methods("vendor_v8/vllm/v1/core/kv_cache_manager.py", "KVCacheManager", {
    "summarize_virtual_context_hbm_release_readiness",
    "plan_virtual_context_hbm_release_candidates",
    "mark_virtual_context_hbm_released_to_dram",
}, {})
Scheduler = methods("vendor_v8/vllm/v1/core/sched/scheduler.py", "Scheduler", {
    "_scheduled_prefill_fit_preparation_status",
    "_virtual_context_prefill_reclaim_candidate_snapshot",
    "_try_defer_running_prefill_to_tiered_provider",
}, {
    "itertools": itertools,
    "ResidencyVictimCandidate": POLICY.ResidencyVictimCandidate,
    "ResidencyVictimPolicy": POLICY.ResidencyVictimPolicy,
    "estimate_jit_min_copy_components": POLICY.estimate_jit_min_copy_components,
    "virtual_context_residency_victim_context_tokens":
        V.virtual_context_residency_victim_context_tokens,
    "resolve_virtual_context_coverage_block_size": V.resolve_virtual_context_coverage_block_size,
    "_read_bool_env": lambda _: True,
    "RequestStatus": NS(PREEMPTED="preempted"),
})


def projection(candidates):
    result = {}
    for c in candidates:
        key = (c.request_id, c.kv_cache_group_id, c.logical_token_page)
        count, tokens = result.get(key, (0, 0))
        result[key] = (count + getattr(c, "candidate_records", 1),
                       max(tokens, c.page_size_tokens))
    return result


def scheduler_fixture(enabled, copied_layers=3):
    t = V.VirtualContextTable()
    SCOPED.populate(t, "A", copied_layers=copied_layers)
    m = Manager()
    m.virtual_context = t
    m.plan_virtual_context_hbm_reload_pages = t.plan_hbm_reload_pages
    m.get_request_resident_block_count = lambda _: 3
    m.get_block_ids = lambda _: ([1, 2, 3],)
    freed = []
    m.free = lambda request, **kwargs: freed.append(request.request_id)
    req = NS(request_id="A", arrival_time=0, num_prompt_tokens=100,
             num_computed_tokens=41, virtual_context_deferred_logical_computed_tokens=0,
             has_encoder_inputs=False, pd_output_placeholder_ledger=NS(active=None))
    # Counters mutated by the real transition method; no mocked readiness.
    for name in (
        "transition_attempt_count", "deferred_prefill_count", "deferred_hbm_released_tokens",
        "tiered_provider_deferred_tokens", "tiered_commit_count", "tiered_commit_released_pages",
        "tiered_commit_existing_dram_pages", "preempt_reset_avoided_tokens",
    ):
        prefix = "virtual_context_tiered_" if name == "transition_attempt_count" else "virtual_context_"
        setattr(req, prefix + name, 0)
    req.validate_external_residency_suspension = lambda **kw: None
    req.suspend_to_external_residency = lambda **kw: None
    h = Scheduler()
    h.isolation_logical_release_readiness = enabled
    h.kv_cache_manager = m
    h._pd_pending_request_ids = set()
    h._debug_has_prefill_remaining = lambda request: True
    h._active_virtual_context_prefill_requests = lambda: tuple(h.requests.values())
    h.requests = {"A": req}
    h.is_encoder_decoder = False
    h.virtual_context_tiered_provider_max_external_tokens = 0
    h.virtual_context_tiered_external_blocks = True
    h.virtual_context_residency_victim_policy = POLICY.ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
    h._virtual_context_residency_fixed_ring_chunk_tokens = lambda: 2048
    h.block_size = 16
    h._iteration_debug_index = 7
    h._consume_scheduled_prefill_fit_preparation = lambda _: None
    h._scheduled_prefill_fit_preparation = NS(target_request_id="T",
        victim_request_ids=("A",), snapshot_pages=(("A", 0, 0), ("A", 0, 1), ("A", 0, 2)))
    h._scheduled_prefill_fit_preparation_cancelled_count = 0
    h._scheduled_prefill_fit_preparation_target_mismatch_count = 0
    h._scheduled_prefill_fit_unprepared_reclaim_count = 0
    return h, req, t, freed


class LogicalReleaseReadinessTests(unittest.TestCase):
    def assert_equivalent(self, table, ids, partial, keys=None):
        expected = projection(table.plan_hbm_release_candidates(
            request_ids=ids, allow_partial_pages=partial))
        if keys is not None:
            expected = {k: v for k, v in expected.items() if k in keys}
        actual = table.summarize_hbm_release_readiness(
            request_ids=ids, allow_partial_pages=partial, page_keys=keys)
        self.assertEqual(projection(actual), expected)
        self.assertEqual(len(actual), len(expected))

    def test_normal_missing_layers_partial_tail_and_growth(self):
        t = V.VirtualContextTable()
        SCOPED.populate(t, "A")
        SCOPED.populate(t, "B", copied_layers=2)
        for tokens in (41, 45, 48):
            t.observe_hbm_blocks("A", ([1, 2, 3],), 16, tokens)
            for ids in (set(), {"A"}, {"B"}, {"missing"}, {"A", "B"}):
                for partial in (False, True):
                    self.assert_equivalent(t, ids, partial)
                    self.assert_equivalent(t, ids, partial, {("A", 0, 2), ("B", 0, 2)})

    def test_random_groups_heads_legacy_orphans_and_unready_states(self):
        rng = random.Random(1103)
        for _ in range(80):
            t = V.VirtualContextTable()
            for rid in ("A", "B"):
                for layer in range(rng.randrange(4)):
                    t.observe_provider_layer(rid, layer)
                with t._lock:
                    for group, page, layer, head in itertools.product(
                        range(2), range(3), (-2, -1, 0, 1, 2), (-1, 0, 1)
                    ):
                        if rng.random() < .2:
                            continue
                        size = rng.choice((0, 7, 16))
                        t._set_page_locked(V.LogicalPageId(rid, group, page, layer, head),
                            V.ResidencyRecord(tier=rng.choice(("HBM", "HBM", "DRAM")),
                                state=rng.choice(("ready", "ready", "loading")), version=2,
                                gpu_block_id=rng.choice((None, 4, 5)), page_size_tokens=size,
                                block_size_tokens=16, dram_copy_version=rng.choice((None, 1, 2)),
                                dram_copy_tokens=rng.choice((7, 16, 20))))
            for partial in (False, True):
                self.assert_equivalent(t, {"A", "B"}, partial)
                self.assert_equivalent(t, {"A"}, partial, {("A", 1, 2), ("B", 0, 1)})

    def test_ready_multiple_groups_heads_and_extra_layer_veto(self):
        t = V.VirtualContextTable()
        t.observe_provider_layer("A", 0)
        with t._lock:
            for group, layer, head in itertools.product(range(2), (-1, 0, 1), (-1, 0, 1)):
                t._set_page_locked(V.LogicalPageId("A", group, 0, layer, head),
                    V.ResidencyRecord(tier="HBM", state="ready", version=2, gpu_block_id=1,
                        page_size_tokens=16, block_size_tokens=16,
                        dram_copy_version=2, dram_copy_tokens=16))
        self.assert_equivalent(t, {"A"}, True)
        self.assertEqual(len(t.summarize_hbm_release_readiness(request_ids={"A"})), 2)
        with t._lock:
            t._pages[V.LogicalPageId("A", 1, 0, 1, 1)].dram_copy_tokens = 7
        self.assert_equivalent(t, {"A"}, True)
        self.assertEqual(len(t.summarize_hbm_release_readiness(request_ids={"A"})), 1)

    def test_explicit_keys_do_not_scan_requests_or_create_release_descriptors(self):
        t = V.VirtualContextTable()
        SCOPED.populate(t, "A", tokens=4096)
        key = ("A", 0, 1)
        original = t._layer_pages_by_aggregate

        class OnlySelected(dict):
            def get(self, candidate, default=None):
                if candidate != key:
                    raise AssertionError("unrelated logical page inspected")
                return super().get(candidate, default)

        t._layer_pages_by_aggregate = OnlySelected(original)
        t._pages = SCOPED.NoGlobalScan(t._pages)
        with patch.object(t, "_request_page_records_locked", side_effect=AssertionError("scan")), \
             patch.object(V, "HBMReleaseCandidate", side_effect=AssertionError("descriptor")):
            result = t.summarize_hbm_release_readiness(request_ids={"A"}, page_keys={key})
            self.assertEqual(projection(result), {key: (3, 16)})
            self.assertEqual(t.summarize_hbm_release_readiness(request_ids={"A"}, page_keys=set()), [])

    def test_growth_invalidates_summary_and_actual_commit_revalidates(self):
        t = V.VirtualContextTable()
        SCOPED.populate(t, "A")
        old = t.summarize_hbm_release_readiness(request_ids={"A"}, allow_partial_pages=True)
        self.assertEqual(len(old), 3)
        t.observe_hbm_blocks("A", ([1, 2, 3],), 16, 45)
        self.assert_equivalent(t, {"A"}, True)
        released = t.mark_hbm_released_to_dram(request_ids={"A"}, allow_partial_pages=True)
        self.assertEqual(set(projection(released)), {("A", 0, 0), ("A", 0, 1)})
        self.assertEqual(len(old), 3)  # immutable observation, not a release grant

    def test_retire_publish_reload_abort_and_request_id_reuse(self):
        t = V.VirtualContextTable()
        SCOPED.populate(t, "A")
        t.retire_layerwise_hbm(["A"])
        self.assert_equivalent(t, {"A"}, True)
        version = t.reserve_layerwise_provider_version("A")
        t.publish_layerwise_dram(request_id="A", logical_tokens=25, block_size=16,
            released_block_ids=(1, 2), layer_ids=(0, 1, 2), version=version)
        for layer in range(3):
            t.mark_hbm_reload_complete("A", {0: 11, 1: 12}, 16, layer_id=layer,
                page_token_counts={0: 16, 1: 9}, provider_versions={0: version, 1: version})
            self.assert_equivalent(t, {"A"}, True)
        t.release_request("A")
        t.record_dram_copies_ready("A", {0: version}, 16, layer_id=0,
            page_to_block={0: 1}, require_existing_request=True)
        self.assert_equivalent(t, {"A"}, True)
        self.assertEqual(t.summarize_hbm_release_readiness(request_ids={"A"}), [])
        SCOPED.populate(t, "A", tokens=16, layers=1)
        self.assert_equivalent(t, {"A"}, True)

    def test_manager_without_virtual_table_is_empty(self):
        m = Manager()
        m.virtual_context = None
        self.assertEqual(m.summarize_virtual_context_hbm_release_readiness(request_ids={"A"}), [])

    def test_scheduler_preparation_status_on_off_equivalence(self):
        for copied, pending, required, target, removed in itertools.product(
            (2, 3), (False, True), (0, 2, 4), ("T", "new"), (False, True)
        ):
            outcomes = []
            for enabled in (False, True):
                h, _, _, _ = scheduler_fixture(enabled, copied)
                if pending:
                    h._pd_pending_request_ids.add("A")
                if removed:
                    h.requests.clear()
                result = h._scheduled_prefill_fit_preparation_status(
                    live_target_request_id=target, required_blocks=required)
                outcomes.append((result, h._scheduled_prefill_fit_preparation_cancelled_count,
                                 h._scheduled_prefill_fit_preparation_target_mismatch_count,
                                 h._scheduled_prefill_fit_unprepared_reclaim_count))
            self.assertEqual(*outcomes)

    def test_scheduler_victim_snapshot_on_off_equivalence(self):
        for copied, pending, policy in itertools.product((2, 3), (False, True),
            (POLICY.ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT, POLICY.ResidencyVictimPolicy.JIT_MIN_COPY)):
            outcomes = []
            for enabled in (False, True):
                h, req, _, _ = scheduler_fixture(enabled, copied)
                h.virtual_context_residency_victim_policy = policy
                if pending:
                    h._pd_pending_request_ids.add("A")
                outcomes.append(h._virtual_context_prefill_reclaim_candidate_snapshot(req))
            self.assertEqual(*outcomes)

    def test_scheduler_transition_keeps_physical_commit_and_counters(self):
        for copied in (2, 3):
            outcomes = []
            for enabled in (False, True):
                h, req, table, freed = scheduler_fixture(enabled, copied)
                with patch.object(table, "mark_hbm_released_to_dram",
                                  wraps=table.mark_hbm_released_to_dram) as commit:
                    result = h._try_defer_running_prefill_to_tiered_provider(
                        req, 41, request_already_waiting=True)
                    self.assertEqual(commit.call_count, int(copied == 3))
                counters = {k: v for k, v in vars(req).items() if k.startswith("virtual_context_")}
                outcomes.append((result, freed, counters, table.get_request_debug_stats("A")))
            self.assertEqual(*outcomes)

    def test_query_exception_releases_lock_without_mutating_pages(self):
        t = V.VirtualContextTable()
        SCOPED.populate(t, "A")
        before = t.plan_hbm_release_candidates(request_ids={"A"})
        with patch.object(t, "_is_hbm_release_candidate", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                t.summarize_hbm_release_readiness(request_ids={"A"})
        self.assertTrue(t._lock.acquire(blocking=False))
        t._lock.release()
        self.assertEqual(t.plan_hbm_release_candidates(request_ids={"A"}), before)


def benchmark():
    t = V.VirtualContextTable()
    SCOPED.populate(t, "A", tokens=50000, layers=16)
    keys = {("A", 0, page) for page in range(128)}
    reference = projection(t.plan_hbm_release_candidates(request_ids={"A"}, allow_partial_pages=True))
    for name, query, expected in (
        ("release_descriptors", lambda: t.plan_hbm_release_candidates(request_ids={"A"},
            allow_partial_pages=True), reference),
        ("logical_summary", lambda: t.summarize_hbm_release_readiness(request_ids={"A"},
            allow_partial_pages=True), reference),
        ("exact_2k_preparation", lambda: t.summarize_hbm_release_readiness(request_ids={"A"},
            allow_partial_pages=True, page_keys=keys), {k: v for k, v in reference.items() if k in keys}),
    ):
        durations = []
        for _ in range(5):
            start = time.perf_counter()
            result = query()
            durations.append((time.perf_counter() - start) * 1000)
            assert projection(result) == expected
        print(f"{name}: median_ms={statistics.median(durations):.3f} result_records={len(result)}")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
