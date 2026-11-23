#!/usr/bin/env python3
"""Exercise the real shared claim; preserve Native reload's stronger contract."""
import ast
import unittest
from types import SimpleNamespace as NS
import test_scheduled_prefill_fit_residency as R


class ChunkFitTests(unittest.TestCase):
    def harness(self, enabled=True):
        h = R.load_capacity_claim_harness()()
        h.isolation_prefill_fit_next_chunk = enabled
        h.max_num_scheduled_tokens = 16384
        h.scheduler_config = NS(long_prefill_token_threshold=2048)
        h.enable_ttft_prefill_chunk_scheduling = True
        h._ttft_cap_selected_prefill_chunk = lambda r, t, **kw: min(t, 2048)
        h._virtual_context_running_decode_reload_required_page_count = lambda: 0
        h._virtual_context_reload_required_page_count = lambda *a: 0
        h.virtual_context_decode_hotset_first = False
        h.virtual_context_staging_reserved_blocks = 8
        h.virtual_context_ring_min_free_blocks = 8
        h.virtual_context_decode_hotset_reserve_blocks = 0
        h.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
        h._debug_running_phase_counts = lambda: {"num_running_decode_reqs": 1}
        h._pd_pending_decode_active = lambda: True
        h._scheduled_prefill_fit_decode_growth_runway = None
        h.num_lookahead_tokens = 0
        h.captured = {}
        def estimate(r, tokens, **kw):
            h.captured.update(tokens=tokens, **kw)
            # This double models a released prefix, not retained concrete KV.
            # Native reload estimates physical prefix + growth in one call.
            prefix = r.num_computed_tokens if kw.get("require_native_hbm") else 0
            return (tokens + prefix + 15) // 16
        h.kv_cache_manager = NS(block_pool=NS(get_num_free_blocks=lambda: 256),
                                estimate_num_blocks_to_allocate=estimate)
        r = NS(request_id="L", num_prompt_tokens=50000, num_computed_tokens=20480,
               virtual_context_deferred_logical_computed_tokens=0)
        return h, r

    def test_next_native_chunk_fits_but_full_prompt_does_not(self):
        h, r = self.harness()
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual((c.chunk_tokens, c.raw_required_blocks), (2048, 128))
        self.assertEqual(c.hard_reserved_blocks, 98)
        self.assertEqual(c.reclaim_blocks, 0)
        h.isolation_prefill_fit_next_chunk = False
        old = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual(h.captured["tokens"], 29520)
        self.assertGreater(old.reclaim_blocks, 0)

    def test_real_chunk_deficit_and_decode_reserve_are_not_ignored(self):
        h, r = self.harness()
        h.kv_cache_manager.block_pool.get_num_free_blocks = lambda: 200
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual(c.hard_reclaim_blocks, 26)

    def test_projected_claim_preserves_current_chunk_accounting(self):
        h, r = self.harness()
        c = h._scheduled_prefill_fit_capacity_claim(
            r, ring_token_cap=2048, logical_tokens=22528, num_new_computed_tokens=2048)
        self.assertEqual(c.logical_tokens, 22528)
        self.assertEqual(h.captured["num_new_computed_tokens"], 2048)
        self.assertEqual(h.captured["tokens"], 2048)

    def test_full_promotion_remains_full_and_includes_reload(self):
        h, r = self.harness()
        h._virtual_context_reload_required_page_count = lambda *a: 1280 * 17
        c = h._scheduled_prefill_fit_capacity_claim(
            r, ring_token_cap=2048, require_full_prompt=True)
        self.assertEqual(h.captured["tokens"], 29520)
        self.assertTrue(h.captured["require_native_hbm"])
        self.assertEqual(c.raw_required_blocks, 1845 + 1280)
        self.assertGreater(c.reclaim_blocks, 0)

    def test_ring_and_tail_keep_exact_bounded_chunk(self):
        h, r = self.harness()
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=1024, full_native=False)
        self.assertEqual((c.chunk_tokens, h.captured["tokens"]), (1024, 1024))
        r.num_computed_tokens = 49980
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual((c.chunk_tokens, h.captured["tokens"]), (20, 20))

    def test_only_reload_promotion_callers_require_whole_prompt(self):
        tree = ast.parse(R.SCHEDULER_PATH.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        callers = []
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for node in ast.walk(method):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "_scheduled_prefill_fit_capacity_claim" and any(
                        k.arg == "require_full_prompt" and isinstance(k.value, ast.Constant)
                        and k.value.value is True for k in node.keywords):
                        callers.append(method.name)
        self.assertEqual(len(callers), 2)
        self.assertIn("_prepare_virtual_context_native_prefill_promotion", callers)

    def tail_harness(self):
        h, r = self.harness()
        h.isolation_native_soft_headroom_progress = True
        r.num_prompt_tokens, r.num_computed_tokens = 64310, 63488
        h.kv_cache_manager.block_pool.get_num_free_blocks = lambda: 127
        h.virtual_context_staging_reserved_blocks = 0
        h.virtual_context_ring_min_free_blocks = 0
        h._debug_running_phase_counts = lambda: {"num_running_decode_reqs": 0}
        h._pd_pending_decode_active = lambda: False
        return h, r

    def test_decode_free_tail_borrows_only_soft_headroom(self):
        h, r = self.tail_harness()
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual((c.chunk_tokens, c.raw_required_blocks, c.free_blocks), (822, 52, 127))
        self.assertEqual((c.reclaim_blocks, c.hard_reclaim_blocks, c.reserved_blocks), (0, 0, 0))
        self.assertEqual(c.waived_capacity_headroom_blocks, 82)
        h.isolation_native_soft_headroom_progress = False
        old = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual((old.reclaim_blocks, old.waived_capacity_headroom_blocks), (7, 0))

    def test_running_or_pending_decode_restores_hard_growth_reserve(self):
        for pending in (False, True):
            h, r = self.tail_harness()
            h._debug_running_phase_counts = lambda: {"num_running_decode_reqs": int(not pending)}
            h._pd_pending_decode_active = lambda: pending
            c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
            self.assertEqual((c.hard_reserved_blocks, c.hard_reclaim_blocks), (82, 7))
            self.assertEqual(c.waived_capacity_headroom_blocks, 0)

    def test_actual_shortage_and_staging_are_never_waived(self):
        h, r = self.tail_harness()
        h.kv_cache_manager.block_pool.get_num_free_blocks = lambda: 51
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual(c.hard_reclaim_blocks, 1)
        self.assertEqual(c.waived_capacity_headroom_blocks, 0)
        h.kv_cache_manager.block_pool.get_num_free_blocks = lambda: 127
        h.virtual_context_staging_reserved_blocks = 76
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual(c.hard_reclaim_blocks, 1)
        self.assertEqual(c.waived_capacity_headroom_blocks, 0)

    def test_ring_reload_and_full_promotion_keep_old_claim(self):
        for kwargs, reload_pages in (({"full_native": False}, 0), ({"require_full_prompt": True}, 0), ({}, 1)):
            h, r = self.tail_harness()
            h._virtual_context_reload_required_page_count = lambda *a: reload_pages
            c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048, **kwargs)
            self.assertGreater(c.reclaim_blocks, 0)
            self.assertEqual(c.waived_capacity_headroom_blocks, 0)

    def test_projected_tail_and_live_claim_match(self):
        h, r = self.tail_harness()
        live = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        r.num_computed_tokens -= 2048
        projected = h._scheduled_prefill_fit_capacity_claim(
            r, ring_token_cap=2048, logical_tokens=63488, num_new_computed_tokens=2048)
        self.assertEqual(projected, live)
        self.assertEqual(h.captured["num_new_computed_tokens"], 2048)

    def test_obsolete_preparation_clears_without_cancelling_copy(self):
        h, r = self.tail_harness()
        claim = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        prep = R.load_preparation_harness()()
        owned_copy = NS(cancel=lambda: self.fail("must not cancel source copy"))
        prep._scheduled_prefill_fit_preparation = owned_copy
        prep._scheduled_prefill_fit_preparation_cancelled_count = 0
        self.assertEqual(prep._scheduled_prefill_fit_preparation_status(
            live_target_request_id="L", required_blocks=claim.reclaim_blocks), (None, 0, 0, True))
        self.assertIsNone(prep._scheduled_prefill_fit_preparation)

    def test_fraction_complement_remains_decode_reserve(self):
        h, r = self.tail_harness()
        h.virtual_context_scheduled_prefill_fit_headroom_blocks = 205
        self.assertEqual(h._scheduled_prefill_fit_capacity_claim(
            r, ring_token_cap=2048).waived_capacity_headroom_blocks, 205)
        h._pd_pending_decode_active = lambda: True
        c = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=2048)
        self.assertEqual(c.hard_reserved_blocks, 205)
        self.assertGreater(c.reclaim_blocks, 0)


if __name__ == "__main__":
    unittest.main()
