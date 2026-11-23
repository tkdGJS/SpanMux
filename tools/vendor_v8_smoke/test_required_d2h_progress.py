#!/usr/bin/env python3
"""Real provider selection/dispatch/completion with CPU DMA and finite events."""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import test_dram_refresh_probe as P

SPEC = importlib.util.spec_from_file_location(
    "required_progress", P.C.ROOT / "vendor_v8/vllm/v1/core/tiered_required_d2h.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
FLAG = "VLLM_ISOLATION_D2H_EXACT_PLAN_PROGRESS"


class Fixture(P.Fixture):
    def __init__(self, count=3968, cap=1024, pending_cap=4096, blocked=False):
        super().__init__(count=None)
        self.count = count
        self.progress = M.RequiredD2HProgress()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.copy_started, self.copy_allowed = Event(), Event()
        self.budget_blocked, self.released_event = Event(), Event()
        if not blocked:
            self.copy_allowed.set()
        self.copies, self.coverage, self.fences, self.wakes = [], {}, {}, []
        self.peak_budget = 0
        self.fail_copy = False
        self.reject_enqueue = False
        self.reject_completion = False
        self.lease_owner = NS(release=lambda: self.released_event.set())
        tree = ast.parse(P.PROVIDER.read_text())
        names = {"_submit_required_hbm_registration", "_d2h_control_dispatch_worker",
                 "_async_hbm_registration_worker", "_propagate_future_result",
                 "_publish_pending_hbm_d2h_fences", "_release_hbm_source_block_lease",
                 "_hbm_d2h_page_pending", "has_pending_hbm_d2h_reads"}
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DRAMKVPageRegistry")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {
            "try_reserve_async_register_from_hbm", "record_async_register_from_hbm_complete",
            "cancel_async_register_from_hbm_reservation"}]
        selected.append(ast.ClassDef(name="Budget", bases=[], keywords=[], decorator_list=[], body=methods))
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(P.PROVIDER), "exec"), self.ns)
        for method in methods:
            setattr(self.reg, method.name, MethodType(getattr(self.ns["Budget"], method.name), self.reg))
        for field in ("pending_pages", "fallback_sync_count", "submit_count", "page_count",
                      "complete_count", "fail_count", "duration_us_total", "duration_us_last"):
            setattr(self.reg, "_register_from_hbm_async_" + field, 0)
        self.reg.lookup = lambda **kw: self.coverage.get((kw["layer_id"], kw["logical_page"]))
        for metric in ("start", "source_view_dispatch_materialize", "inline_fallback", "complete", "submit"):
            setattr(self.reg, "record_d2h_control_dispatch_" + metric, lambda *a, **kw: None)
        self.reg.enqueue_from_hbm_direct = self.enqueue
        self.reg.complete_from_hbm_direct = self.complete
        reserve = self.reg.try_reserve_async_register_from_hbm
        def observed_reserve(n, **kw):
            result = reserve(n, **kw)
            self.peak_budget = max(self.peak_budget, self.reg._register_from_hbm_async_pending_pages)
            if not result:
                self.budget_blocked.set()
            return result
        self.reg.try_reserve_async_register_from_hbm = observed_reserve
        self.ns.update({
            "Future": Future, "required_d2h_progress": self.progress,
            "_PENDING_HBM_D2H_FENCES": self.fences, "_PENDING_HBM_D2H_FENCES_LOCK": Lock(),
            "notify_pd_engine_progress": lambda: self.wakes.append(1),
            "tiered_dram_register_max_pages_per_forward": lambda: cap,
            "tiered_dram_register_async_max_pending_pages": lambda: pending_cap,
            "_get_async_hbm_register_executor": lambda: self.completion_executor(),
            "acquire_tiered_hbm_source_block_lease": lambda ids: self.lease_owner,
        })

    def completion_executor(self):
        if self.reject_completion:
            raise RuntimeError("completion executor rejected")
        return self.pool

    def enqueue(self, pages, **kw):
        if self.reject_enqueue:
            raise RuntimeError("enqueue rejected")
        return pages

    def complete(self, pages):
        self.copy_started.set()
        if not self.copy_allowed.wait(5):
            raise RuntimeError("test DMA gate timeout")
        self.copies.append(len(pages))
        if self.fail_copy:
            return len(pages)
        for page in pages:
            self.coverage[(page.layer_id, page.logical_page)] = NS(
                version=page.version, page=NS(token_count=page.token_count))
        return 0

    def run(self, *, required=True, **kw):
        args = dict(key_cache=P.Tensor(np.zeros((self.count, 16, 1, 2))),
                    value_cache=P.Tensor(np.zeros((self.count, 16, 1, 2))),
                    block_table=P.Tensor([range(self.count)]), seq_lens=P.Tensor([self.count * 16]),
                    exact_force_pages_by_request={"R": set(range(self.count))} if required else None)
        args.update(kw)
        return super().run(**args)

    def wait(self):
        # This barrier is behind the whole plan, not merely its first batch.
        self.progress._executor.submit(lambda: None).result(timeout=5)

    def close(self):
        self.copy_allowed.set()
        self.reg._register_from_hbm_async_pending_pages = 0
        self.progress.budget_released()
        if self.progress._executor is not None:
            self.progress._executor.shutdown(wait=True)
        self.pool.shutdown(wait=True)


class RequiredProgressTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {FLAG: "1", P.FLAG: "1"})
        env.start()
        self.addCleanup(env.stop)

    def fixture(self, **kw):
        f = Fixture(**kw)
        self.addCleanup(f.close)
        return f

    def test_3968_required_pages_finish_without_another_forward(self):
        f = self.fixture()
        self.assertEqual(f.run(), 3968)
        f.wait()
        self.assertEqual(f.copies, [1024, 1024, 1024, 896])
        self.assertEqual(len(f.coverage), 3968)
        self.assertEqual(sum(len(r["page_versions"]) for r in f.ready), 3968)
        self.assertEqual(f.fences, {})
        self.assertTrue(f.released_event.is_set())
        self.assertEqual(f.progress.stats()["dram_provider_required_d2h_pending_pages"], 0)
        self.assertLessEqual(f.peak_budget, 4096)
        self.assertTrue(f.wakes)

    def test_budget_full_returns_then_completion_wakes_without_forward(self):
        f = self.fixture(count=20, cap=8, pending_cap=8)
        f.reg.try_reserve_async_register_from_hbm(8)
        self.assertEqual(f.run(), 20)
        self.assertTrue(f.budget_blocked.wait(2))
        self.assertFalse(f.copy_started.is_set())
        self.assertEqual(len(f.fences), 20)
        self.assertFalse(f.released_event.is_set())
        f.reg.record_async_register_from_hbm_complete(8, duration_s=0, fail_count=0)
        f.wait()
        self.assertEqual(f.copies, [8, 8, 4])
        self.assertEqual(f.reg._register_from_hbm_async_fallback_sync_count, 0)
        self.assertEqual(f.peak_budget, 8)

    def test_queued_sources_stay_fenced_and_request_cannot_cancel_copy(self):
        f = self.fixture(count=20, cap=8, blocked=True)
        f.run()
        self.assertTrue(f.copy_started.wait(2))
        self.assertEqual(len(f.fences), 20)
        owner = next(iter(f.fences.values()))
        self.assertFalse(owner.cancel())
        self.assertTrue(f.ns["_hbm_d2h_page_pending"]("R", 0, 19))
        self.assertFalse(f.ns["has_pending_hbm_d2h_reads"](["unrelated-decode"]))
        self.assertEqual(f.ready, [])
        self.assertFalse(f.released_event.is_set())
        f.copy_allowed.set()
        f.wait()
        owner.result()
        self.assertEqual(f.fences, {})
        self.assertTrue(f.released_event.is_set())

    def test_failure_stays_visible_after_fences_retire(self):
        for failure in ("fail_copy", "reject_enqueue"):
            with self.subTest(failure=failure):
                f = self.fixture(count=20, cap=8)
                setattr(f, failure, True)
                f.run()
                f.wait()
                self.assertEqual(f.ready, [])
                self.assertEqual(f.fences, {})
                self.assertEqual(f.reg._register_from_hbm_async_pending_pages, 0)
                self.assertTrue(f.released_event.is_set())
                with self.assertRaisesRegex(RuntimeError, "required D2H snapshot failed"):
                    f.progress.raise_if_failed()
                self.assertEqual(f.progress.stats()["dram_provider_required_d2h_failed_plans"], 1)

    def test_completion_pool_rejection_drains_off_model_thread(self):
        f = self.fixture(count=20, cap=8)
        f.reject_completion = True
        f.run()
        f.wait()
        f.progress.raise_if_failed()
        self.assertEqual(f.copies, [8, 8, 4])
        self.assertEqual(len(f.coverage), 20)

    def test_selection_flagoff_or_non_required_keeps_legacy_cap(self):
        for required, enabled in ((True, "0"), (False, "1")):
            f = self.fixture()
            with patch.dict(os.environ, {FLAG: enabled}):
                self.assertEqual(f.run(required=required), 1024)
            self.assertEqual(len(f.submissions), 1)
            self.assertIsNone(f.progress._executor)

    def test_required_mode_rejects_unsafe_missing_lease_or_async(self):
        for name in ("tiered_hbm_source_block_lease_enabled", "tiered_dram_register_async_enabled"):
            f = self.fixture(count=2)
            f.ns[name] = lambda: False
            with self.assertRaisesRegex(RuntimeError, "requires async registration and source leases"):
                f.run()
            self.assertEqual(f.fences, {})

    def test_smaller_pending_budget_limits_each_batch(self):
        f = self.fixture(count=20, cap=1024, pending_cap=3)
        f.run()
        f.wait()
        self.assertEqual(f.copies, [3, 3, 3, 3, 3, 3, 2])
        self.assertEqual(f.peak_budget, 3)

    def test_executor_creation_failure_is_terminal_not_inline_copy(self):
        f = self.fixture(count=2)
        with patch.object(M, "ThreadPoolExecutor", side_effect=RuntimeError("setup rejected")):
            with self.assertRaisesRegex(RuntimeError, "setup rejected"):
                f.run()
        self.assertFalse(f.copy_started.is_set())
        self.assertEqual(f.fences, {})
        self.assertTrue(f.released_event.is_set())
        with self.assertRaisesRegex(RuntimeError, "required D2H snapshot failed"):
            f.progress.raise_if_failed()

    def test_cancelled_request_sidecar_is_not_resurrected_on_copy_completion(self):
        spec = importlib.util.spec_from_file_location(
            "required_copy_virtual_context", P.C.ROOT / "vendor_v8/vllm/v1/core/virtual_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        table = module.VirtualContextTable()
        table.observe_hbm_blocks("R", ([0, 1],), block_size=16, logical_tokens=32)
        f = self.fixture(count=2, cap=1, blocked=True)
        f.ns["get_virtual_context_table"] = lambda: table
        f.run()
        self.assertTrue(f.copy_started.wait(2))
        table.release_request("R")
        self.assertFalse(f.released_event.is_set())
        f.copy_allowed.set()
        f.wait()
        f.progress.raise_if_failed()
        self.assertNotIn("R", table._request_tokens)
        self.assertNotIn("R", table._request_versions)
        self.assertFalse(any(key.request_id == "R" for key in table._pages))
        self.assertTrue(f.released_event.is_set())

    def test_live_request_publishes_guarded_sidecar(self):
        spec = importlib.util.spec_from_file_location(
            "required_copy_live_context", P.C.ROOT / "vendor_v8/vllm/v1/core/virtual_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        table = module.VirtualContextTable()
        table.observe_hbm_blocks("R", ([0, 1],), block_size=16, logical_tokens=32)
        f = self.fixture(count=2, cap=1)
        f.ns["get_virtual_context_table"] = lambda: table
        f.run()
        f.wait()
        f.progress.raise_if_failed()
        self.assertEqual(table._request_tokens["R"], 32)
        self.assertEqual(table._dram_copy_ready_calls, 2)

    def test_cancelled_budget_reservation_also_wakes_waiter(self):
        f = self.fixture(count=4, cap=2, pending_cap=2)
        f.reg.try_reserve_async_register_from_hbm(2)
        f.run()
        self.assertTrue(f.budget_blocked.wait(2))
        f.reg.cancel_async_register_from_hbm_reservation(2)
        f.wait()
        self.assertEqual(f.copies, [2, 2])

    def test_exact_filter_and_later_required_request_survive_optional_cap(self):
        f = self.fixture(count=20, cap=2)
        f.run(request_ids=["optional", "R"],
              block_table=P.Tensor([range(20), range(20)]), seq_lens=P.Tensor([320, 320]),
              exact_force_pages_by_request={"R": {3, 4, 5, 19}})
        f.wait()
        required = {page for record in f.ready if record["request_id"] == "R"
                    for page in record["page_versions"]}
        self.assertEqual(required, {3, 4, 5, 19})
        self.assertEqual(sum(f.copies), 6)  # 2 optional + 4 exact, not 20.

    def test_source_plan_path_copies_partial_tail_and_all_required_descriptors(self):
        f = self.fixture(count=3, cap=1)
        pages = [NS(logical_page=i, block_id=i, token_start=i*16,
                    token_count=7 if i == 2 else 16, force_all=True) for i in range(3)]
        plan = NS(matches=lambda *a: True, group_by_request=lambda: {"R": pages})
        f.run(source_page_plan=plan, seq_lens=P.Tensor([39]))
        f.wait()
        self.assertEqual(f.copies, [1, 1, 1])
        self.assertEqual(f.coverage[(0, 2)].page.token_count, 7)
        self.assertEqual(f.ready[-1]["page_token_counts"], {2: 7})

    def test_real_source_lease_protects_all_queued_pages_until_final_completion(self):
        spec = importlib.util.spec_from_file_location(
            "required_copy_source_lease", P.C.ROOT / "vendor_v8/vllm/v1/core/tiered_hbm_source_block_lease.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        f = self.fixture(count=20, cap=8, blocked=True)
        f.ns["acquire_tiered_hbm_source_block_lease"] = module.acquire_tiered_hbm_source_block_lease
        f.run()
        self.assertTrue(f.copy_started.wait(2))
        self.assertEqual(module.get_tiered_hbm_source_leased_block_ids(), frozenset(range(20)))
        f.copy_allowed.set()
        f.wait()
        self.assertEqual(module.get_tiered_hbm_source_leased_block_ids(), frozenset())
        stats = module.get_tiered_hbm_source_block_lease_stats()
        self.assertEqual((stats.acquire_count, stats.release_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
