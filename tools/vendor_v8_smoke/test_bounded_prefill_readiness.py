#!/usr/bin/env python3
"""CPU contracts for bounded readiness and pre-transaction Engine yields.

Load the real table and real Scheduler/Engine methods without importing CUDA.
Readiness observations are compared with the unchanged live release planner;
neither a cached result nor completion of a CPU scan is a physical release.
"""
from __future__ import annotations

import random
import threading
import time
import unittest
from collections import Counter, deque
from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_logical_release_readiness as L


V = L.V
populate = L.SCOPED.populate
projection = L.projection
Manager = L.methods("vendor_v8/vllm/v1/core/kv_cache_manager.py", "KVCacheManager", {
    "warm_virtual_context_hbm_release_readiness",
}, {})
Scheduler = L.methods("vendor_v8/vllm/v1/core/sched/scheduler.py", "Scheduler", {
    "prepare_pd_prefill_admission",
}, {})
Engine = L.methods("vendor_v8/vllm/v1/engine/core.py", "EngineCore", {
    "_submit_pd_scheduler_lanes",
}, {"time": time})
Progress = L.load("bounded_readiness_progress",
                  "vendor_v8/vllm/v1/engine/pd_engine_progress.py")


class ProjectionTests(unittest.TestCase):
    def make_table(self, **kwargs):
        table = V.VirtualContextTable(cache_release_readiness=True)
        populate(table, "A", **kwargs)
        return table

    def assert_exact(self, table, ids=None):
        ids = {"A", "B", "C", "missing"} if ids is None else ids
        for partial in (False, True):
            expected = projection(table.plan_hbm_release_candidates(
                request_ids=ids, allow_partial_pages=partial))
            actual = table.summarize_hbm_release_readiness(
                request_ids=ids, allow_partial_pages=partial)
            self.assertEqual(projection(actual), expected)
            self.assertEqual(len(actual), len(expected))
        expected_index = {}
        for page_id in table._pages:
            counts = expected_index.setdefault(page_id.request_id, Counter())
            counts[(page_id.request_id, page_id.kv_cache_group_id,
                    page_id.logical_token_page)] += 1
        self.assertEqual(table._logical_page_refcounts, expected_index)

    def test_budget_and_repeated_cache_hits(self):
        table = self.make_table(tokens=129 * 16)
        for expected_done, evaluated in ((False, 64), (False, 128), (True, 129)):
            self.assertEqual(table.warm_hbm_release_readiness(request_ids={"A"}),
                             expected_done)
            self.assertEqual(table._readiness_evaluated_pages, evaluated)
        with patch.object(table, "_summarize_release_page_locked",
                          side_effect=AssertionError("warmed query rescanned layers")):
            for _ in range(3):
                self.assertEqual(len(table.summarize_hbm_release_readiness(
                    request_ids={"A"}, allow_partial_pages=True)), 129)
        self.assertEqual(table._readiness_cache_hits, 3 * 129)
        self.assertEqual(table._readiness_warm_yields, 2)
        self.assert_exact(table)

    def test_partial_warm_exact_fallback_and_live_commit(self):
        table = self.make_table(tokens=80 * 16)
        self.assertFalse(table.warm_hbm_release_readiness(request_ids={"A"}))
        self.assertEqual(table._readiness_evaluated_pages, 64)
        self.assertEqual(len(table.summarize_hbm_release_readiness(
            request_ids={"A"}, allow_partial_pages=True)), 80)
        self.assertTrue(table.warm_hbm_release_readiness(request_ids={"A"}))
        self.assertEqual(table._readiness_evaluated_pages, 80)
        with patch.object(table, "_collect_hbm_release_candidates_locked",
                          wraps=table._collect_hbm_release_candidates_locked) as live:
            released = table.mark_hbm_released_to_dram(
                request_ids={"A"}, allow_partial_pages=True)
            live.assert_called_once()
        self.assertEqual(len(projection(released)), 80)
        self.assert_exact(table)

    def test_budget_shared_across_requests_and_empty_ids_not_retained(self):
        table = self.make_table(tokens=40 * 16)
        populate(table, "B", tokens=40 * 16)
        ids = {"A", "B", "missing"}
        self.assertFalse(table.warm_hbm_release_readiness(request_ids=ids))
        self.assertEqual(table._readiness_evaluated_pages, 64)
        self.assertTrue(table.warm_hbm_release_readiness(request_ids=ids))
        self.assertEqual(table._readiness_evaluated_pages, 80)
        self.assertNotIn("missing", table._release_readiness_scans)
        self.assert_exact(table)

    def test_negative_result_invalidated_by_copy_publication(self):
        table = self.make_table(copied_layers=2)
        self.assertTrue(table.warm_hbm_release_readiness(request_ids={"A"}))
        self.assertEqual(table.summarize_hbm_release_readiness(
            request_ids={"A"}, allow_partial_pages=True), [])
        table.record_dram_copies_ready("A", {0: 1, 1: 1, 2: 1}, 16, layer_id=2,
            page_to_block={0: 1, 1: 2, 2: 3},
            page_token_counts={0: 16, 1: 16, 2: 9})
        self.assertNotIn("A", table._release_readiness_scans)
        self.assertEqual(len(table.summarize_hbm_release_readiness(
            request_ids={"A"}, allow_partial_pages=True)), 3)
        self.assert_exact(table)

    def test_partial_tail_growth_rejects_old_copy(self):
        table = self.make_table()
        table.warm_hbm_release_readiness(request_ids={"A"})
        table.observe_hbm_block_delta("A", ([],), new_page_start_by_group=(3,),
                                      block_size=16, logical_tokens=45)
        self.assertNotIn("A", table._release_readiness_scans)
        for layer in range(3):
            table.record_dram_copies_ready("A", {2: 1}, 16, layer_id=layer,
                page_to_block={2: 3}, page_token_counts={2: 9})
        self.assertEqual(set(projection(table.summarize_hbm_release_readiness(
            request_ids={"A"}, allow_partial_pages=True))),
            {("A", 0, 0), ("A", 0, 1)})
        self.assert_exact(table)

    def test_new_expected_layer_invalidates_positive_result(self):
        table = self.make_table()
        table.warm_hbm_release_readiness(request_ids={"A"})
        table.observe_provider_layer("A", 3)
        self.assertNotIn("A", table._release_readiness_scans)
        self.assertEqual(table.summarize_hbm_release_readiness(
            request_ids={"A"}, allow_partial_pages=True), [])
        self.assert_exact(table)

    def test_mutation_resets_only_own_cursor(self):
        table = self.make_table(tokens=80 * 16)
        populate(table, "B")
        table.warm_hbm_release_readiness(request_ids={"B"})
        stable = table._release_readiness_scans["B"]
        self.assertFalse(table.warm_hbm_release_readiness(request_ids={"A"}))
        table.observe_provider_layer("A", 3)
        self.assertNotIn("A", table._release_readiness_scans)
        self.assertIs(table._release_readiness_scans["B"], stable)
        self.assertFalse(table.warm_hbm_release_readiness(request_ids={"A"}))
        self.assertTrue(table.warm_hbm_release_readiness(request_ids={"A"}))
        self.assert_exact(table)

    def test_private_retire_publish_reload_abort_and_id_reuse(self):
        table = self.make_table()
        table.warm_hbm_release_readiness(request_ids={"A"})
        table.retire_layerwise_hbm(["A"])
        self.assertNotIn("A", table._release_readiness_scans)
        version = table.reserve_layerwise_provider_version("A")
        table.publish_layerwise_dram(request_id="A", logical_tokens=25, block_size=16,
            released_block_ids=(1, 2), layer_ids=(0, 1, 2), version=version)
        self.assert_exact(table)
        for layer in range(3):
            table.mark_hbm_reload_complete("A", {0: 11, 1: 12}, 16, layer_id=layer,
                page_token_counts={0: 16, 1: 9}, provider_versions={0: version, 1: version})
            self.assertNotIn("A", table._release_readiness_scans)
            self.assert_exact(table)
        table.release_request("A")
        table.record_dram_copies_ready("A", {0: version}, 16, layer_id=0,
            page_to_block={0: 1}, require_existing_request=True)
        self.assertNotIn("A", table._release_readiness_scans)
        self.assertNotIn("A", table._logical_page_refcounts)
        self.assert_exact(table)
        populate(table, "A", tokens=16, layers=1)
        self.assert_exact(table)

    def test_set_delete_orphan_and_reload_state_refresh(self):
        table = self.make_table()
        table.warm_hbm_release_readiness(request_ids={"A"})
        page_id = V.LogicalPageId("A", 2, 7, 0, 1)
        with table._lock:
            table._set_page_locked(page_id, V.ResidencyRecord(
                tier="HBM", state="ready", version=1, gpu_block_id=99,
                page_size_tokens=16, block_size_tokens=16,
                dram_copy_version=1, dram_copy_tokens=16))
        self.assertNotIn("A", table._release_readiness_scans)
        self.assert_exact(table)
        with table._lock:
            table._pages[page_id].state = "staging"
            table._refresh_reload_index_locked(page_id)
        self.assertNotIn("A", table._release_readiness_scans)
        self.assert_exact(table)
        with table._lock:
            table._delete_page_locked(page_id)
        self.assertNotIn("A", table._release_readiness_scans)
        self.assert_exact(table)

    def test_default_off_and_invalid_budget(self):
        table = V.VirtualContextTable()
        populate(table, "A")
        self.assertTrue(table.warm_hbm_release_readiness(request_ids={"A"}))
        self.assertEqual(table._logical_page_refcounts, {})
        self.assertEqual(table._release_readiness_scans, {})
        self.assertEqual(table._readiness_evaluated_pages, 0)
        for budget in (0, -1):
            with self.assertRaisesRegex(ValueError, "budget must be positive"):
                table.warm_hbm_release_readiness(request_ids={"A"}, max_pages=budget)

    def test_scan_error_releases_lock_without_advancing_cursor(self):
        table = self.make_table()
        with patch.object(table, "_summarize_release_page_locked",
                          side_effect=RuntimeError("coverage failure")):
            with self.assertRaisesRegex(RuntimeError, "coverage failure"):
                table.warm_hbm_release_readiness(request_ids={"A"})
        self.assertEqual(table._release_readiness_scans["A"].cursor, 0)
        self.assertTrue(table._lock.acquire(blocking=False))
        table._lock.release()
        self.assertTrue(table.warm_hbm_release_readiness(request_ids={"A"}))
        self.assert_exact(table)

    def test_random_mutation_sequence_matches_live_release_planner(self):
        table = V.VirtualContextTable(cache_release_readiness=True)
        rng = random.Random(2119)
        for _ in range(150):
            rid = rng.choice(("A", "B", "C"))
            operation = rng.randrange(7)
            if operation == 0:
                table.release_request(rid)
            elif operation == 1:
                populate(table, rid, tokens=rng.randrange(1, 160),
                         copied_layers=rng.randrange(4))
            elif operation == 2:
                table.observe_provider_layer(rid, rng.randrange(5))
            elif operation == 3:
                table.observe_hbm_blocks(rid, ([1, 2, 3],), 16, rng.randrange(1, 49))
            elif operation == 4:
                table.record_dram_copies_ready(rid, {0: rng.randrange(1, 5)}, 16,
                    layer_id=rng.randrange(3), page_to_block={0: 1},
                    page_token_counts={0: rng.randrange(1, 17)})
            elif operation == 5:
                table.mark_hbm_released_to_dram(request_ids={rid}, allow_partial_pages=True)
            else:
                table.mark_hbm_reload_complete(rid, {0: 1}, 16, layer_id=0,
                    page_token_counts={0: 16}, provider_versions={0: 5})
            table.warm_hbm_release_readiness(request_ids={rid}, max_pages=2)
            self.assert_exact(table)


class AdmissionTests(unittest.TestCase):
    def test_scheduler_warms_only_existing_quiescent_preparation_victims(self):
        scheduler = Scheduler()
        scheduler.isolation_bounded_prefill_planning = True
        scheduler.requests = {"A": object(), "B": object(), "unrelated": object()}
        scheduler._pd_pending_request_ids = {"B"}
        scheduler._scheduled_prefill_fit_preparation = NS(
            victim_request_ids=("A", "B", "gone"))
        warm = Mock(return_value=False)
        scheduler.kv_cache_manager = NS(warm_virtual_context_hbm_release_readiness=warm)
        self.assertFalse(scheduler.prepare_pd_prefill_admission())
        warm.assert_called_once_with(request_ids={"A"}, max_pages=64)
        scheduler._scheduled_prefill_fit_preparation = None
        self.assertTrue(scheduler.prepare_pd_prefill_admission())
        scheduler.isolation_bounded_prefill_planning = False
        self.assertTrue(scheduler.prepare_pd_prefill_admission())
        warm.assert_called_once()

    def test_manager_without_virtual_context_is_noop(self):
        manager = Manager()
        manager.virtual_context = None
        self.assertTrue(manager.warm_virtual_context_hbm_release_readiness(
            request_ids={"A"}))

    def engine(self, *, decode=True, enabled=True, warm=None):
        engine = Engine()
        future = Future()
        queues = {"prefill": deque(), "decode": deque()}
        if decode:
            queues["decode"].append((future, NS(request_id="D"), None))
        engine.scheduler = NS(isolation_bounded_prefill_planning=enabled,
                              has_requests=lambda: True,
                              prepare_pd_prefill_admission=Mock(
                                  side_effect=warm, return_value=False))
        engine.batch_queue_size = 2
        engine._pd_phase_queue_total_depth = lambda q: sum(map(len, q.values()))
        engine._select_pd_phase_queue_recovery_item = Mock(return_value=("decode", 0, {}))
        engine._try_submit_pd_paired_phase_launch = Mock(return_value=(False, None, None, False))
        engine._pd_phase_queue_can_submit_phase = lambda q, phase: phase == "prefill"
        engine._pd_capacity_retry_can_submit = Mock(return_value=True)
        engine._pd_phase_queue_pending_request_ids = lambda q: {"D"} if decode else set()
        engine._pd_phase_queue_pending_request_ids_by_phase = lambda q: {
            "decode": {"D"} if decode else set()}
        engine._set_scheduler_pd_pending_request_ids = Mock()
        engine._set_scheduler_pd_pending_phase_request_ids = Mock()
        engine._scheduler_has_pd_phase_work = lambda scheduler, phase: True
        engine._scheduler_schedule_pd_phase = Mock(return_value=NS(
            total_num_scheduled_tokens=0, finished_req_ids=set()))
        engine._record_pd_capacity_retry_block = Mock()
        engine._notify_pd_engine_progress = Mock()
        engine._write_pd_phase_queue_trace = Mock()
        return engine, queues, future

    def submit(self, engine, queues):
        return engine._submit_pd_scheduler_lanes(
            queues, ("decode", "prefill"), allow_multiple_rounds=True)

    def test_incomplete_scan_yields_without_output_or_capacity_gate(self):
        for has_decode in (False, True):
            with self.subTest(decode=has_decode):
                engine, queues, future = self.engine(decode=has_decode)
                self.assertEqual(self.submit(engine, queues), (False, None, None, False))
                engine._scheduler_schedule_pd_phase.assert_not_called()
                engine._record_pd_capacity_retry_block.assert_not_called()
                engine._notify_pd_engine_progress.assert_called_once()
                engine._write_pd_phase_queue_trace.assert_called_once_with(
                    "prefill_readiness_yield", queues)
                engine._set_scheduler_pd_pending_request_ids.assert_called_with(set())
                engine._set_scheduler_pd_pending_phase_request_ids.assert_called_with({})
                self.assertFalse(future.done())
                self.assertEqual(len(queues["decode"]), int(has_decode))

    def test_decode_completes_during_warm_then_no_prefill_transaction(self):
        engine, queues, future = self.engine()

        def warm():
            future.set_result("token")
            return True

        engine.scheduler.prepare_pd_prefill_admission.side_effect = warm
        self.submit(engine, queues)
        engine.scheduler.prepare_pd_prefill_admission.assert_called_once()
        engine._scheduler_schedule_pd_phase.assert_not_called()
        engine._record_pd_capacity_retry_block.assert_not_called()
        self.assertIs(queues["decode"][0][0], future)  # normal recovery still owns it

    def test_ready_decode_shortcut_obeys_dependency_selector(self):
        engine, queues, future = self.engine()
        future.set_result("token")
        self.submit(engine, queues)
        engine.scheduler.prepare_pd_prefill_admission.assert_not_called()
        engine._scheduler_schedule_pd_phase.assert_not_called()
        engine._select_pd_phase_queue_recovery_item.return_value = ("prefill", 0, {})
        self.submit(engine, queues)
        engine.scheduler.prepare_pd_prefill_admission.assert_called_once()
        self.assertIs(queues["decode"][0][0], future)

    def test_flag_off_preserves_original_admission(self):
        engine, queues, future = self.engine(enabled=False)
        future.set_result("token")
        self.submit(engine, queues)
        engine.scheduler.prepare_pd_prefill_admission.assert_not_called()
        engine._scheduler_schedule_pd_phase.assert_called_once_with(engine.scheduler, "prefill")
        engine._record_pd_capacity_retry_block.assert_called_once()
        engine._notify_pd_engine_progress.assert_not_called()

    def test_existing_capacity_gate_is_not_bypassed_or_self_notified(self):
        engine, queues, _ = self.engine()
        engine._pd_capacity_retry_can_submit.return_value = False
        self.submit(engine, queues)
        engine.scheduler.prepare_pd_prefill_admission.assert_not_called()
        engine._scheduler_schedule_pd_phase.assert_not_called()
        engine._notify_pd_engine_progress.assert_not_called()

    def test_real_progress_generation_keeps_cpu_continuation_live(self):
        table = V.VirtualContextTable(cache_release_readiness=True)
        populate(table, "A", tokens=130 * 16, copied_layers=0)
        engine, queues, _ = self.engine(decode=False,
            warm=lambda: table.warm_hbm_release_readiness(request_ids={"A"}))
        engine._notify_pd_engine_progress = Progress.notify_pd_engine_progress
        event = threading.Event()
        Progress.register_pd_engine_progress_event(event)
        try:
            for expected in (64, 128):
                previous = Progress.pd_engine_progress_generation()
                self.submit(engine, queues)
                self.assertEqual(table._readiness_evaluated_pages, expected)
                event.clear()  # generation survives a consumed/lost event bit
                self.assertTrue(Progress.wait_for_pd_engine_progress(
                    event, lambda: False, timeout_s=0, observed_generation=previous))
                engine._scheduler_schedule_pd_phase.assert_not_called()
                engine._record_pd_capacity_retry_block.assert_not_called()
            previous = Progress.pd_engine_progress_generation()
            self.submit(engine, queues)
            engine._scheduler_schedule_pd_phase.assert_called_once()
            self.assertEqual(table._readiness_evaluated_pages, 130)
            self.assertEqual(Progress.pd_engine_progress_generation(), previous)
            self.assertEqual(table.summarize_hbm_release_readiness(
                request_ids={"A"}, allow_partial_pages=True), [])
        finally:
            Progress.unregister_pd_engine_progress_event(event)


if __name__ == "__main__":
    unittest.main()
