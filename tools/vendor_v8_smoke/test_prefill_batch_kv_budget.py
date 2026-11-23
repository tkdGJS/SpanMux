#!/usr/bin/env python3
"""CPU contracts for the real scheduler methods; no GPU/performance claims."""
import ast
import unittest
from types import SimpleNamespace as NS

import test_kv_retention_layerwise_offload as O
import test_scheduled_prefill_fit_residency as R
from test_slo_prefill_scheduler import MODULE as P


S = O.methods(O.BASE / "v1/core/sched/scheduler.py", "Scheduler", {
    "layerwise_offload_needed", "_ttft_plan_prefill_admission_frontier",
    "_check_virtual_context_prefill_residency_admission",
    "_isolation_prefill_starts_decode",
}, dict(logger=NS(info=lambda *a: None),
        PrefillAdmissionFrontier=P.PrefillAdmissionFrontier,
        PrefillAdmissionCandidate=P.PrefillAdmissionCandidate,
        PrefillChunkSchedulingPolicy=P.PrefillChunkSchedulingPolicy,
        build_native_prefill_admission_frontier=P.build_native_prefill_admission_frontier))


class Harness(S, R.load_capacity_claim_harness()):
    pass


def request(rid, prompt, computed=0, outputs=8192):
    r = NS(request_id=rid, num_prompt_tokens=prompt, num_computed_tokens=computed,
           max_tokens=outputs, num_output_tokens=0, virtual_context_deferred_state="",
           virtual_context_deferred_logical_computed_tokens=0, finished=False,
           pd_residency_prefill_skip_count=0)
    r.is_finished = lambda: r.finished
    return r


def harness(free=255):
    h = Harness()
    h.isolation_prefill_batch_kv_budget = True
    h.isolation_native_soft_headroom_progress = True
    h.isolation_prefill_fit_next_chunk = True
    h._isolation_prefill_batch_budget_debug = {}
    h.max_num_scheduled_tokens = 16384
    h.max_num_running_reqs = 100
    h.max_model_len = 65536
    h.num_lookahead_tokens = 0
    h.block_size = 16
    h.scheduler_config = NS(long_prefill_token_threshold=2048)
    h.enable_ttft_prefill_chunk_scheduling = True
    h.pd_decoupled_no_mix = True
    h.isolation_prefill_batch_max_requests = 0
    h.ttft_prefill_chunk_scheduler = NS(policy=P.PrefillChunkSchedulingPolicy.SRPT)
    h._ttft_prefill_execution_mode = lambda r: "native"
    h._pd_prefill_lane_remaining_tokens = lambda r, **kw: r.num_prompt_tokens - kw["num_computed_tokens"]
    h._ttft_cap_selected_prefill_chunk = lambda r, n, **kw: min(n, 2048)
    h.virtual_context_staging_reserved_blocks = 0
    h.virtual_context_ring_min_free_blocks = 0
    h.virtual_context_decode_hotset_reserve_blocks = 0
    h.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
    h.virtual_context_decode_hotset_first = False
    h.virtual_context_residency_admission = False
    h.virtual_context_ring_residency_admission = False
    h._virtual_context_running_decode_reload_required_page_count = lambda: 0
    h._virtual_context_reload_required_page_count = lambda *a: 0
    h._virtual_context_waiting_decode_ready = lambda r: False
    h._debug_running_phase_counts = lambda: {"num_running_decode_reqs": 0}
    h._pd_pending_decode_active = lambda: False
    h._debug_has_prefill_remaining = lambda r: r.num_computed_tokens < r.num_prompt_tokens
    h.blocks_next_schedule = lambda r, phase: False
    h._debug_record_virtual_context_residency_admission_defer = lambda *a: None
    h.free = free
    h.calls = []
    def estimate(r, n, **kw):
        h.calls.append((r.request_id, n, kw))
        return (n + 15) // 16
    h.source_blocks = [NS(block_id=7, ref_cnt=1, is_null=False)]
    h.kv_cache_manager = NS(virtual_context=object(),
        estimate_num_blocks_to_allocate=estimate,
        block_pool=NS(get_num_free_blocks=lambda: h.free),
        get_blocks=lambda rid: NS(blocks=(h.source_blocks,)))
    a, b, long = request("A", 1699), request("B", 1718), request("L", 64310, 61440, 1)
    h.running, h.waiting = [long], [a, b]
    h.requests = {r.request_id: r for r in (a, b, long)}
    out = NS(num_scheduled_tokens={"L": 2048}, pd_prefill_mode="native")
    return h, a, b, long, out


class BatchBudgetTests(unittest.TestCase):
    def test_engine_passes_sorted_nonpending_cohort_before_yield(self):
        h, a, b, long, out = harness()
        pending = request("P", 16)
        for r in (a, b, long, pending):
            r.arrival_time = 0.0
        h.requests = {r.request_id: r for r in (b, pending, long, a)}
        h.ttft_prefill_chunk_scheduler.remaining_prefill_tokens = (
            lambda r: r.num_prompt_tokens - r.num_computed_tokens)
        out.ttft_layerwise_prefill_preemption = True
        out.kv_retention_layerwise_pressure_checked = False
        out.kv_retention_layerwise_transfer = None
        out.kv_retention_p1_park_declined = False
        out.scheduler_iteration_id = 1
        out.pd_phase_submit_seq = 38
        out.pd_phase_slot = 0
        cls = O.methods(O.BASE / "v1/engine/core.py", "EngineCore", {
            "_request_ttft_prefill_preemption_if_needed",
        }, dict(should_preempt_ttft_layerwise_prefill_batch=O.C.M.should_preempt_ttft_layerwise_prefill_batch,
                request_ttft_layerwise_prefill_yield=lambda **kw: True))
        e = cls()
        e.scheduler = h
        e.ttft_layerwise_prefill_preemption_enabled = True
        e.kv_retention_p1_parked_enabled = True
        e.kv_retention_layerwise_offload_enabled = True
        e.pd_sm_transition = NS(active=False)
        e._ttft_prefill_suspended = []
        e._pd_phase_queue_pending_request_ids = lambda queues: {"L", "P"}
        e._ttft_prefill_preemption_signal_count = 0
        e._iteration_debug_timings = {}
        e._write_pd_phase_queue_trace = lambda *a: None
        future = NS(done=lambda: False)
        self.assertTrue(e._request_ttft_prefill_preemption_if_needed(
            {"prefill": [(future, out, future)]}))
        self.assertEqual([c[0] for c in h.calls], ["A", "B"])
        self.assertTrue(out.kv_retention_layerwise_offload_requested)
        self.assertEqual((h.running, h.free), ([long], 255))

    def test_pilot_single_candidate_fits_but_whole_batch_requires_offload(self):
        h, a, b, _, out = harness()
        h.isolation_prefill_batch_kv_budget = False
        self.assertFalse(h.layerwise_offload_needed(out, a))
        h.isolation_prefill_batch_kv_budget = True
        self.assertTrue(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
        self.assertEqual([c[:2] for c in h.calls[-2:]], [("A", 1699), ("B", 1718)])
        self.assertEqual(h.free, 255)  # Advisory: no page allocation/free.

    def test_exact_aggregate_threshold_and_source_ownership(self):
        h, a, b, _, out = harness(297)
        self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
        h.free = 296
        self.assertTrue(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
        h.source_blocks[0].ref_cnt = 2
        self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))

    def test_frontier_obeys_tokens_phase_quota_and_sequence_capacity(self):
        for attr, value in (("max_num_scheduled_tokens", 1699),
                            ("isolation_prefill_batch_max_requests", 1),
                            ("max_num_running_reqs", 1)):
            h, a, b, _, out = harness()
            setattr(h, attr, value)
            self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
            self.assertEqual(len(h.calls), 1)

    def test_parked_source_frees_sequence_slot_only_in_projection(self):
        h, a, b, long, out = harness()
        h.max_num_running_reqs = 2
        self.assertTrue(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
        self.assertEqual(h.running, [long])
        # Active Decode still occupies its slot; only the source is parked.
        h.running.append(request("D", 100, 100))
        self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))

    def test_canceled_blocked_duplicate_and_nonqueued_candidates_not_charged(self):
        h, a, b, _, out = harness()
        b.finished = True
        self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, a, b)))
        self.assertEqual(len(h.calls), 1)
        b.finished = False
        h.blocks_next_schedule = lambda r, phase: r is b
        self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
        h.waiting = []
        h.calls = []
        self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
        self.assertEqual(h.calls, [])

    def test_tiered_single_target_still_stops_native_frontier(self):
        h, a, b, _, out = harness()
        h._ttft_prefill_execution_mode = lambda r: "ring" if r is b else "native"
        self.assertFalse(h.layerwise_offload_needed(out, a, replacement_candidates=(a, b)))
        self.assertEqual(len(h.calls), 1)

    def admit(self, h, r, tokens, scheduled=None, **kwargs):
        return h._check_virtual_context_prefill_residency_admission(
            r, tokens, "test_prefill", scheduled_tokens=scheduled, **kwargs)

    def test_live_budget_rejects_second_row_before_reserve_is_consumed(self):
        h, a, b, _, _ = harness()
        first = self.admit(h, a, 1699, {})
        self.assertTrue(first[0])
        h.free -= first[5]  # Simulate the real allocator's successful debit.
        admitted = {"A": 1699}
        second = self.admit(h, b, 1718, admitted)
        self.assertEqual((h.free, second[0], second[1], second[3]), (148, False, 108, 82))
        self.assertFalse(self.admit(h, b, 1718, admitted)[0])
        self.assertEqual(h.free, 148)  # Rejection/retry creates no shadow debt.

    def test_successful_rows_are_not_debited_twice(self):
        h, a, b, _, _ = harness(400)
        self.assertTrue(self.admit(h, a, 1699, {})[0])
        h.free -= 107
        self.assertTrue(self.admit(h, b, 1718, {"A": 1699})[0])
        self.assertEqual(h.free, 293)  # Second check is read-only too.

    def test_earlier_final_row_protects_headroom_from_later_nonfinal_row(self):
        h, a, b, _, _ = harness(148)
        b.num_prompt_tokens = 10000
        self.assertFalse(self.admit(h, b, 1718, {"A": 1699})[0])
        self.assertTrue(self.admit(h, b, 1718, {})[0])
        # A retracted/cancelled row cannot leave a sticky imminent-decode flag.
        a.finished = True
        self.assertTrue(self.admit(h, b, 1718, {"A": 1699})[0])

    def test_running_or_pending_decode_keeps_fractional_reserve(self):
        for pending in (False, True):
            h, _, b, _, _ = harness(148)
            b.num_prompt_tokens = 10000
            h._debug_running_phase_counts = lambda: {"num_running_decode_reqs": int(not pending)}
            h._pd_pending_decode_active = lambda: pending
            self.assertFalse(self.admit(h, b, 1718)[0])
        h.virtual_context_scheduled_prefill_fit_headroom_blocks = 205
        h.free = 300
        self.assertEqual(self.admit(h, b, 1718)[3], 205)

    def test_final_one_output_tail_keeps_soft_progress_but_multitoken_does_not(self):
        h, _, _, long, _ = harness(127)
        long.num_computed_tokens = 63488
        one = h._scheduled_prefill_fit_capacity_claim(long, ring_token_cap=2048)
        self.assertEqual((one.reclaim_blocks, one.waived_capacity_headroom_blocks), (0, 82))
        long.max_tokens = 8192
        many = h._scheduled_prefill_fit_capacity_claim(long, ring_token_cap=2048)
        self.assertEqual((many.hard_reserved_blocks, many.hard_reclaim_blocks,
                          many.waived_capacity_headroom_blocks), (82, 7, 0))
        self.assertFalse(self.admit(h, long, 822)[0])

    def test_projected_and_live_final_claim_agree(self):
        h, _, _, long, _ = harness(127)
        long.max_tokens = 8192
        projected = h._scheduled_prefill_fit_capacity_claim(
            long, ring_token_cap=2048, logical_tokens=63488, num_new_computed_tokens=2048)
        long.num_computed_tokens = 63488
        live = h._scheduled_prefill_fit_capacity_claim(long, ring_token_cap=2048)
        self.assertEqual(projected, live)

    def test_prefix_and_external_counts_use_exact_allocator_arguments(self):
        h, a, _, _, _ = harness(88)
        blocks = object()
        result = self.admit(h, a, 99, num_new_computed_tokens=800,
            new_computed_blocks=blocks, num_external_computed_tokens=800,
            num_virtual_external_computed_tokens=800,
            num_lookahead_tokens=2, num_encoder_tokens=4)
        self.assertFalse(result[0])  # 99=>7 blocks,88-7<82.
        self.assertEqual(h.calls[-1][2], dict(num_new_computed_tokens=800,
            new_computed_blocks=blocks, num_external_computed_tokens=800,
            num_virtual_external_computed_tokens=800,
            num_lookahead_tokens=2, num_encoder_tokens=4))

    def test_external_prefix_is_not_counted_twice_for_final_detection(self):
        h, a, _, _, _ = harness(10)
        a.virtual_context_deferred_logical_computed_tokens = 800
        # 800+99<1699, not max(800,0)+external800+99.
        self.assertTrue(self.admit(h, a, 99, num_external_computed_tokens=800,
                                  num_virtual_external_computed_tokens=800)[0])

    def test_staging_reserve_and_zero_growth_do_not_bypass_hard_guard(self):
        h, a, _, _, _ = harness(189)
        h.virtual_context_staging_reserved_blocks = 1
        self.assertFalse(self.admit(h, a, 1699)[0])
        h.kv_cache_manager.estimate_num_blocks_to_allocate = lambda *a, **kw: 0
        h.free = 81
        self.assertFalse(self.admit(h, a, 1699)[0])

    def test_decode_handoff_can_spend_headroom(self):
        h, a, _, _, _ = harness(40)
        h.virtual_context_residency_admission = True
        h._virtual_context_waiting_decode_ready = lambda r: True
        self.assertTrue(self.admit(h, a, 16)[0])

    def test_flag_off_restores_legacy_admission(self):
        h, _, b, _, _ = harness(148)
        h.isolation_prefill_batch_kv_budget = False
        self.assertTrue(self.admit(h, b, 1718, {"A": 1699})[0])
        self.assertEqual(h.calls, [])

    def test_estimator_error_creates_no_phantom_reservation(self):
        h, a, _, _, _ = harness()
        def fail(*args, **kw):
            raise RuntimeError("estimate failed")
        h.kv_cache_manager.estimate_num_blocks_to_allocate = fail
        with self.assertRaisesRegex(RuntimeError, "estimate failed"):
            self.admit(h, a, 1699)
        self.assertEqual(h.free, 255)
        self.assertEqual(h._isolation_prefill_batch_budget_debug, {})

    def test_both_real_scans_pass_the_committed_map_before_allocation(self):
        tree = ast.parse(R.SCHEDULER_PATH.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        schedule = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
        calls = [n for n in ast.walk(schedule) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "_check_virtual_context_prefill_residency_admission"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            value = next(k.value for k in call.keywords if k.arg == "scheduled_tokens")
            self.assertEqual(ast.unparse(value), "num_scheduled_tokens")


if __name__ == "__main__":
    unittest.main()
