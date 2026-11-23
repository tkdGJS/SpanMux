#!/usr/bin/env python3
"""CPU lifecycle/admission checks for DRAM WAITING Decode service demand."""
from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

import test_exact_phase_cleanup as C
import test_native_reload_admission as N
import test_decode_reload_physical_budget as D
import test_native_reload_block_budget as B
import test_prefill_batch_kv_budget as F


METHODS = {
    "_virtual_context_waiting_decode_service_candidate",
    "_virtual_context_waiting_decode_service_reserve",
    "_virtual_context_waiting_decode_ready",
    "_scheduled_prefill_fit_active_decode",
    "_scheduled_prefill_fit_headroom_partition",
    "_check_virtual_context_prefill_residency_admission",
    "_prepare_virtual_context_native_prefill_promotion",
    "_isolation_prefill_starts_decode",
}
SC = C.adapters(N.PATH, "Scheduler", METHODS,
                dict(RequestStatus=N.STATUS, _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES=N.STATES))
SC._virtual_context_waiting_decode_ready = staticmethod(SC._virtual_context_waiting_decode_ready)


def harness(free=85, holes=None):
    s, r, table, calls = D.harness(holes=holes, free=free, waiting=True)
    for name in METHODS:
        setattr(s, name, getattr(SC, name).__get__(s, type(s)))
    s._virtual_context_waiting_decode_ready = SC._virtual_context_waiting_decode_ready
    s.waiting, s.running, s.requests = [r], [], {"A": r}
    s.pending = set()
    s.isolation_prefill_batch_kv_budget = True
    s._isolation_prefill_batch_budget_debug = {}
    s.virtual_context_residency_admission = False
    s.virtual_context_ring_min_free_blocks = 0
    s.virtual_context_decode_hotset_reserve_blocks = 0
    s._pd_pending_decode_active = lambda: False
    s._debug_running_phase_counts = lambda: {"num_running_decode_reqs": 0}
    s.blocks_next_schedule = lambda req, phase: req.request_id in s.pending
    s._debug_running_decode_resident_blocks = lambda: 0
    s._debug_has_prefill_remaining = lambda req: req.num_output_tokens == 0
    s._debug_record_virtual_context_residency_admission_defer = lambda *a: None
    r.finished = False
    r.is_finished = lambda: r.finished
    r.max_tokens = 10
    return s, r


class WaitingServiceTests(unittest.TestCase):
    def test_dram_waiter_is_active_decode_without_running_or_future(self):
        s, _ = harness()
        self.assertTrue(s._scheduled_prefill_fit_active_decode())
        self.assertEqual(s._scheduled_prefill_fit_headroom_partition(), (82, 0))

    def test_whole_prefix_promotion_cannot_overtake_waiting_decode(self):
        s, _ = harness()
        p = F.request("P", 100)
        self.assertEqual(s._prepare_virtual_context_native_prefill_promotion(p, ring_token_cap=16),
                         (False, 0, 0, "decode_active_or_not_prefill"))

    def test_one_fitting_prefix_plus_next_token_not_all_queue(self):
        s, r = harness()
        another = N.request(True)
        another.request_id, another.is_finished = "B", lambda: False
        s.waiting.append(another)
        self.assertIs(s._virtual_context_waiting_decode_service_candidate(), r)
        self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 3)
        self.assertEqual(s._isolation_prefill_batch_budget_debug["isolation_waiting_decode_claim_fits"], 1)

    def test_pending_or_terminal_owner_never_creates_a_second_claim(self):
        for mode in ("pending", "finished", "removed"):
            s, r = harness()
            if mode == "pending":
                s.pending.add("A")
            elif mode == "finished":
                r.finished = True
            else:
                s.waiting.clear()
            self.assertIsNone(s._virtual_context_waiting_decode_service_candidate())
            self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 0)

    def test_prefill_cannot_take_fitting_decode_capacity(self):
        s, _ = harness(85)
        p = F.request("P", 100)
        p.status = N.STATUS.WAITING
        self.assertFalse(s._check_virtual_context_prefill_residency_admission(p, 1, "prefill")[0])
        s.kv_cache_manager.block_pool.get_num_free_blocks = lambda: 86
        self.assertTrue(s._check_virtual_context_prefill_residency_admission(p, 1, "prefill")[0])

    def test_nonfitting_waiter_does_not_block_bounded_ring_progress(self):
        s, _ = harness(84)
        self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 0)
        p = F.request("P", 100)
        p.status = N.STATUS.WAITING
        self.assertTrue(s._check_virtual_context_prefill_residency_admission(p, 1, "prefill")[0])
        self.assertEqual(s._isolation_prefill_batch_budget_debug["isolation_waiting_decode_claim_fits"], 0)

    def test_projected_claim_and_live_admission_keep_same_service_blocks(self):
        s, _ = harness(91)
        h, _ = B.claim_harness(s.kv_cache_manager)
        # Keep the real capacity-claim function with the live service helpers.
        for key, value in s.__dict__.items():
            if key.startswith("_virtual_context_waiting_decode") or key in (
                    "_scheduled_prefill_fit_active_decode", "_scheduled_prefill_fit_headroom_partition",
                    "_isolation_prefill_starts_decode"):
                continue
            setattr(h, key, value)
        for name in METHODS - {"_virtual_context_waiting_decode_ready"}:
            setattr(h, name, getattr(SC, name).__get__(h, type(h)))
        h._virtual_context_waiting_decode_ready = SC._virtual_context_waiting_decode_ready
        p = F.request("P", 100)
        p.status = N.STATUS.WAITING
        claim = h._scheduled_prefill_fit_capacity_claim(p, ring_token_cap=2048)
        self.assertEqual((claim.raw_required_blocks, claim.hard_reserved_blocks,
                          claim.reclaim_blocks), (7, 85, 1))
        self.assertFalse(h._check_virtual_context_prefill_residency_admission(p, 100, "prefill")[0])

    def test_retry_uses_concrete_destinations_and_completion_removes_claim(self):
        s, _ = harness(85, holes=(False, False))
        self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 1)
        s.waiting.clear()
        self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 0)
        self.assertFalse(s._scheduled_prefill_fit_active_decode())

    def test_existing_optout_and_hbm_only_controls_do_not_change(self):
        for field in ("isolation_prefill_batch_kv_budget", "virtual_context_reload_to_hbm"):
            s, _ = harness()
            setattr(s, field, False)
            self.assertIsNone(s._virtual_context_waiting_decode_service_candidate())
            self.assertFalse(s._scheduled_prefill_fit_active_decode())


if __name__ == "__main__":
    unittest.main()
