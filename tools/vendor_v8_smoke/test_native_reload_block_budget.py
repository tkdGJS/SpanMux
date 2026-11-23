#!/usr/bin/env python3
"""Real CPU sidecar -> Scheduler claim -> allocator estimate integration.

No torch import, synthetic GPU event or mocked capacity result. Model/queue
construction is replaced by small doubles; the capacity methods are current
source extracted through the existing AST adapter.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from types import SimpleNamespace as NS

import test_exact_phase_cleanup as C
import test_native_reload_admission as N
import test_scheduled_prefill_fit_residency as R


def load_virtual_context():
    spec = importlib.util.spec_from_file_location("reload_budget_vctx", R.VIRTUAL_CONTEXT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = load_virtual_context()
SINGLE = C.adapters("vendor_v8/vllm/v1/core/single_type_kv_cache_manager.py",
    "SingleTypeKVCacheManager", {"get_num_blocks_to_allocate", "get_retained_residency_counts"},
    {"cdiv": lambda a, b: (a + b - 1) // b,
     "count_full_virtual_external_blocks": V.count_full_virtual_external_blocks})
COORD = C.adapters("vendor_v8/vllm/v1/core/kv_cache_coordinator.py", "KVCacheCoordinator",
    {"get_num_blocks_to_allocate", "get_blocks", "get_retained_residency_counts"},
    {"CrossAttentionManager": type("CrossAttentionManager", (), {})})
KM = C.adapters("vendor_v8/vllm/v1/core/kv_cache_manager.py", "KVCacheManager",
    {"estimate_num_blocks_to_allocate", "has_request_block_table", "get_retained_residency_counts",
     "get_virtual_context_hbm_reload_required_page_count"})


def block(null=False):
    return NS(is_null=null, ref_cnt=0 if null else 1)


def allocator(table, groups=(None,), block_sizes=None):
    managers = []
    for i, row in enumerate(groups):
        m = SINGLE()
        m.block_size = (block_sizes or [16] * len(groups))[i]
        m.kv_cache_group_id = i
        m.req_to_blocks = {} if row is None else {"A": list(row)}
        m.num_cached_block = {} if row is None else {"A": len(row)}
        m.partial_residency_requests = set()
        m.get_num_skipped_tokens = lambda _: 0
        managers.append(m)
    c = COORD()
    c.single_type_managers = tuple(managers)
    k = KM()
    k.coordinator = c
    k.max_model_len = 65536
    k.empty_kv_cache_blocks = NS(blocks=tuple(() for _ in groups))
    k.virtual_context = table
    k.block_pool = NS(get_num_free_blocks=lambda: 10)
    return k


def released_table(tokens=32, layers=16):
    t = V.VirtualContextTable()
    pages = (tokens + 15) // 16
    ids = list(range(1, pages + 1))
    t.observe_hbm_blocks("A", (ids,), 16, tokens)
    for layer in range(layers):
        t.observe_provider_layer("A", layer)
        t.record_dram_copies_ready("A", {p: 1 for p in range(pages)}, 16,
            layer_id=layer, page_to_block=dict(enumerate(ids)),
            page_token_counts={p: min(16, tokens - p * 16) for p in range(pages)})
    t.mark_hbm_released_to_dram(request_ids={"A"}, allow_partial_pages=True)
    return t


def claim_harness(k, *, tokens=32, prompt=48, waiting=False):
    s = R.load_capacity_claim_harness()()
    s.kv_cache_manager = k
    s.max_num_scheduled_tokens = 16384
    s.scheduler_config = NS(long_prefill_token_threshold=2048)
    s.enable_ttft_prefill_chunk_scheduling = False
    s._virtual_context_running_decode_reload_required_page_count = lambda: 0
    s._virtual_context_reload_required_page_count = lambda ids, phase="prefill": (
        k.get_virtual_context_hbm_reload_required_page_count(set(ids)))
    s.virtual_context_decode_hotset_first = False
    s.virtual_context_staging_reserved_blocks = 0
    s.virtual_context_ring_min_free_blocks = 0
    s.virtual_context_decode_hotset_reserve_blocks = 0
    s.virtual_context_scheduled_prefill_fit_headroom_blocks = 0
    s._debug_running_phase_counts = lambda: {"num_running_decode_reqs": 0}
    s._pd_pending_decode_active = lambda: False
    s.num_lookahead_tokens = 0
    r = NS(request_id="A", num_prompt_tokens=prompt,
        num_computed_tokens=0 if waiting else tokens,
        virtual_context_deferred_logical_computed_tokens=tokens)
    return s, r


class NativeReloadBlockBudgetTests(unittest.TestCase):
    def claim(self, k, **kwargs):
        s, r = claim_harness(k, **kwargs)
        return s._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048,
                                                       require_full_prompt=True)

    def test_sixteen_layers_need_three_blocks_not_thirty_five(self):
        t = released_table()
        self.assertEqual(t.hbm_reload_required_page_count({"A"}), 34)
        k = allocator(t, ([block(True), block(True)],))
        c = self.claim(k)
        self.assertEqual((c.raw_required_blocks, c.reclaim_blocks), (3, 0))
        self.assertEqual(t.hbm_reload_required_page_count({"A"}), 34,
                         "physical estimation must not clear all-layer ACK records")

    def test_waiting_no_table_full_prefix_and_growth(self):
        k = allocator(released_table())
        self.assertEqual(self.claim(k, waiting=True).raw_required_blocks, 3)
        self.assertEqual(k.coordinator.get_blocks("A"), ([],))

    def test_waiting_partial_tail_is_not_charged_twice(self):
        k = allocator(released_table(tokens=25))
        self.assertEqual(self.claim(k, tokens=25, prompt=41, waiting=True).raw_required_blocks, 3)

    def test_retained_mixed_table_charges_only_holes_and_growth(self):
        k = allocator(released_table(), ([block(), block(True)],))
        self.assertEqual(self.claim(k).raw_required_blocks, 2)

    def test_prepared_concrete_destinations_are_not_charged_again(self):
        k = allocator(released_table(), ([block(), block()],))
        self.assertEqual(self.claim(k).raw_required_blocks, 1)
        self.assertEqual(k.get_virtual_context_hbm_reload_required_page_count({"A"}), 34)

    def test_partial_tail_with_retained_null_is_counted_once(self):
        k = allocator(released_table(tokens=25), ([block(True), block(True)],))
        self.assertEqual(self.claim(k, tokens=25, prompt=41).raw_required_blocks, 3)

    def test_group_geometry_not_layer_multiplier(self):
        k = allocator(released_table(), ([block(True), block()], [block(True)]), (16, 32))
        # group0: 1 hole +1 growth; group1: 1 hole +1 growth.
        self.assertEqual(self.claim(k).raw_required_blocks, 4)

    def test_missing_group_or_short_retained_table_fails_closed(self):
        for groups in [([block()],), ([block(), block()], None)]:
            with self.subTest(groups=groups), self.assertRaises(RuntimeError):
                self.claim(allocator(released_table(), groups))

    def test_ring_estimate_retains_virtual_prefix_credit(self):
        k = allocator(released_table(tokens=25))
        s, r = claim_harness(k, tokens=25, prompt=41, waiting=True)
        c = s._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048, full_native=False)
        self.assertEqual(c.raw_required_blocks, 2)  # partial tail + growth, prefix stays DRAM

    def test_default_estimate_keeps_existing_virtual_allocation_contract(self):
        k = allocator(released_table())
        _, r = claim_harness(k, waiting=True)
        args = dict(num_external_computed_tokens=32,
                    num_virtual_external_computed_tokens=32)
        self.assertEqual(k.estimate_num_blocks_to_allocate(r, 16, **args), 1)
        self.assertEqual(k.estimate_num_blocks_to_allocate(
            r, 16, require_native_hbm=True, **args), 3)

    def test_projected_chunk_is_growth_not_another_historical_hole(self):
        for waiting in (False, True):
            with self.subTest(waiting=waiting):
                k = allocator(released_table(), ([block(True), block(True)],))
                s, r = claim_harness(k, prompt=64, waiting=waiting)
                c = s._scheduled_prefill_fit_capacity_claim(
                    r, ring_token_cap=2048, logical_tokens=48,
                    num_new_computed_tokens=16, require_full_prompt=True)
                self.assertEqual(c.raw_required_blocks, 4)

    def test_reserve_still_blocks_real_shortage(self):
        k = allocator(released_table(), ([block(True), block(True)],))
        s, r = claim_harness(k)
        s.virtual_context_scheduled_prefill_fit_headroom_blocks = 8
        c = s._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048, require_full_prompt=True)
        self.assertEqual((c.raw_required_blocks, c.reserved_blocks, c.reclaim_blocks), (3, 8, 1))

    def test_rejected_capacity_does_not_build_copy_plan(self):
        r = N.request()
        s = N.scheduler(r)
        s._scheduled_prefill_fit_capacity_claim = lambda *a, **k: NS(
            reclaim_blocks=1, required_blocks_with_reserve=101)
        s.kv_cache_manager.plan_virtual_context_hbm_reload_pages = lambda **k: (
            self.fail("blocked promotion materialized a layer-copy plan"))
        result = s._prepare_virtual_context_native_prefill_promotion(r, ring_token_cap=16)
        self.assertEqual(result, (False, 4, 101, "full_prefill_capacity_unavailable"))

    def test_active_or_pending_decode_does_not_probe_reload(self):
        for pending in [False, True]:
            r = N.request()
            s = N.scheduler(r)
            s._debug_running_decode_resident_blocks = lambda: 0 if pending else 1
            s._pd_pending_decode_active = lambda: pending
            s.kv_cache_manager.get_virtual_context_hbm_reload_required_page_count = (
                lambda _: self.fail("Decode should exclude promotion before querying"))
            self.assertFalse(s._prepare_virtual_context_native_prefill_promotion(r, ring_token_cap=16)[0])

    def test_copy_publication_after_abort_does_not_restore_reload_demand(self):
        t = released_table()
        t.release_request("A")
        t.record_dram_copies_ready("A", {0: 2}, 16, layer_id=0,
            page_to_block={0: 1}, require_existing_request=True)
        self.assertEqual(t.hbm_reload_required_page_count({"A"}), 0)


if __name__ == "__main__":
    unittest.main()
