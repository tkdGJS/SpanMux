#!/usr/bin/env python3
"""Real sidecar/allocator estimates across Decode retry and WAITING handoff."""
from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace as NS

import test_exact_phase_cleanup as C
import test_native_reload_admission as N
import test_native_reload_block_budget as B


S = C.adapters(N.PATH, "Scheduler", {
    "_virtual_context_native_decode_handoff_requirement",
    "_should_reload_virtual_context_to_hbm",
    "_prepare_virtual_context_native_decode_handoff",
    "_native_reload_destinations_compatible",
    "_virtual_context_waiting_decode_service_candidate",
}, dict(defaultdict=defaultdict, virtual_context_residency_victim_context_tokens=max,
        ResidencyVictimPolicy=NS(SCHEDULED_PREFILL_FIT="fit"),
        TieredAllocatorInvariantError=RuntimeError))


def harness(*, holes=(False, False), free=82, waiting=False, tokens=32):
    table = B.released_table(tokens=tokens)
    row = None if holes is None else [NS(is_null=h, block_id=0 if h else p + 1,
                                         ref_cnt=0 if h else 1) for p, h in enumerate(holes)]
    k = B.allocator(table, (row,))
    k.block_pool.get_num_free_blocks = lambda: free
    k.get_blocks = lambda rid: NS(blocks=k.coordinator.get_blocks(rid))
    k.plan_virtual_context_hbm_reload_pages = lambda **kw: table.plan_hbm_reload_pages(**kw)
    s, r = S(), N.request(True)
    r.num_prompt_tokens = tokens
    r.num_computed_tokens = 0 if waiting else tokens
    r.virtual_context_deferred_logical_computed_tokens = tokens
    r.kv_retention_decode_continuation_wait_count = 0
    s.kv_cache_manager = k
    s.virtual_context_reload_to_hbm = True
    s.virtual_context_reload_to_hbm_min_free_blocks = 0
    s.virtual_context_staging_reserved_blocks = 0
    s.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
    s.virtual_context_residency_victim_policy = "fit"
    s._has_pending_prefill_for_ring_progress = lambda: False
    s._virtual_context_active_decode_ring_progress_reserve = lambda: (0, 0, 0)
    s._native_decode_handoff_capacity_precheck_count = 0
    s._native_decode_handoff_plan_avoided_page_count = 0
    s._virtual_context_reload_required_page_count = lambda ids, phase: table.hbm_reload_required_page_count(set(ids))
    s._debug_has_prefill_remaining = lambda _: False
    s._iteration_debug_index = 1
    calls = []
    def allocate(rid, plan):
        calls.append((rid, plan))
        for group, pages in plan.items():
            for page in pages:
                b = k.coordinator.single_type_managers[group].req_to_blocks[rid][page]
                b.is_null, b.ref_cnt, b.block_id = False, 1, page + 1
        return k.get_blocks(rid)
    k.allocate_virtual_context_hbm_reload_blocks = allocate
    return s, r, table, calls


class DecodeBudgetTests(unittest.TestCase):
    def test_concrete_retry_uses_reserve_only_not_copy_pages(self):
        s, r, table, calls = harness()
        before = table.hbm_reload_required_page_count({"A"})
        self.assertEqual(before, 34)
        self.assertEqual(s._virtual_context_native_decode_handoff_requirement(r)[2:4], (0, 82))
        for _ in range(2):
            self.assertTrue(s._prepare_virtual_context_native_decode_handoff(r)[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(table.hbm_reload_required_page_count({"A"}), before)
        self.assertEqual(r.kv_retention_execution_plan_generation, 0)

    def test_retained_holes_not_layer_multiplicity(self):
        s, r, _, _ = harness(holes=(False, True), free=83)
        self.assertEqual(s._virtual_context_native_decode_handoff_requirement(r)[2:4], (1, 83))
        self.assertTrue(s._prepare_virtual_context_native_decode_handoff(r)[0])
        self.assertEqual(s._virtual_context_native_decode_handoff_requirement(r)[2:4], (0, 82))

    def test_waiting_reuses_destinations_without_building_plan(self):
        for holes, required in [((False, False), 0), ((False, True), 1), (None, 2)]:
            s, r, _, _ = harness(holes=holes, waiting=True, free=84)
            s.virtual_context_staging_reserved_blocks = 2
            s.kv_cache_manager.plan_virtual_context_hbm_reload_pages = lambda **_: self.fail("WAITING materialized a layer plan")
            self.assertEqual(s._should_reload_virtual_context_to_hbm(r, 32, 84),
                             (True, 34, required, required + 2))

    def test_waiting_partial_tail_allocates_two_pages_not_layers(self):
        s, r, _, _ = harness(holes=None, waiting=True, tokens=25, free=84)
        self.assertEqual(s._should_reload_virtual_context_to_hbm(r, 25, 84)[2:], (2, 2))

    def test_waiting_near_capacity_does_not_acquire_a_new_runway_requirement(self):
        s, r, _, _ = harness(holes=None, waiting=True, free=2)
        self.assertTrue(s._should_reload_virtual_context_to_hbm(r, 32, 2)[0])

    def test_full_reserve_still_required_even_with_no_new_destinations(self):
        s, r, _, calls = harness(free=81)
        s.kv_cache_manager.plan_virtual_context_hbm_reload_pages = lambda **_: self.fail("infeasible copy plan")
        self.assertEqual(s._prepare_virtual_context_native_decode_handoff(r)[-1], "insufficient_hbm_blocks")
        self.assertFalse(calls)
        self.assertEqual(s._native_decode_handoff_capacity_precheck_count, 1)

    def test_invalid_retained_table_still_fails(self):
        s, r, _, _ = harness(holes=(False,))
        with self.assertRaisesRegex(RuntimeError, "does not cover"):
            s._virtual_context_native_decode_handoff_requirement(r)

    def test_shared_destination_not_mistaken_for_safe_ready_content(self):
        s, r, _, calls = harness()
        s.kv_cache_manager.get_blocks("A").blocks[0][0].ref_cnt = 2
        self.assertEqual(s._prepare_virtual_context_native_decode_handoff(r)[-1], "reload_allocator_plan_mismatch")
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
