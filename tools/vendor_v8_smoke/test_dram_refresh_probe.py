#!/usr/bin/env python3
"""Actual producer/registry/completion adapters with CPU storage and DMA doubles."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import time
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import test_exact_phase_cleanup as C


PROVIDER = C.ROOT / "vendor_v8/vllm/v1/attention/ops/tiered_residency_provider.py"
FLAG = "VLLM_ISOLATION_DRAM_REFRESH_PROBE"
SPEC = importlib.util.spec_from_file_location("refresh_isolation_flags", C.ROOT / "vendor_v8/vllm/isolation_memory.py")
FLAGS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FLAGS)


def provider_scope():
    tree = ast.parse(PROVIDER.read_text())
    classes = {"TieredKVPageKey", "TieredDRAMKVPage", "TieredDRAMKVLookup",
               "_PendingHBMRegistrationPage", "_DeferredHBMRegistrationPage"}
    functions = {"register_hbm_kv_cache_pages", "_mark_hbm_registration_complete",
                 "_provider_snapshot_covers_pinned_version",
                 "_register_pending_hbm_pages_sync", "_materialize_hbm_registration_pages"}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "DRAMKVPageRegistry":
            body.append(ast.ClassDef(name=node.name, bases=[], keywords=[], decorator_list=[],
                                    body=[n for n in node.body if isinstance(n, ast.FunctionDef)
                                          and n.name == "lookup"]))
    scope = {"__name__": __name__, "dataclass": dataclass, "field": field,
             "time": time, "defaultdict": defaultdict, "TieredBackendEligibilityError": RuntimeError,
             "isolation_execution_enabled": FLAGS.isolation_execution_enabled,
             "torch": NS(no_grad=contextlib.nullcontext)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(PROVIDER), "exec"), scope)
    return scope


class Tensor:
    is_cuda = False
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
    def dim(self):
        return self.values.ndim
    def numel(self):
        return self.values.size
    def __getitem__(self, index):
        value = self.values[index]
        return Tensor(value) if isinstance(value, np.ndarray) else value
    def detach(self):
        return self


class Fixture:
    def __init__(self, storage="pool", count=11, version=0):
        self.ns = ns = provider_scope()
        self.reg = ns["DRAMKVPageRegistry"]()
        self.storage, self.pending = storage, False
        self.submissions, self.ready, self.leases, self.released = [], [], [], []
        self.reg._lock = Lock()
        for name in ("lookup", "hit", "miss", "latest_lookup", "pinned_lookup"):
            setattr(self.reg, "_" + name + "_count", 0)
        self.reg._pages, self.reg._latest_versions = {}, {}
        self.reg._resolve_request_key_locked = lambda *a, **kw: 1
        self.snapshot = None
        self.reg._pool = NS(lookup=self.pool_lookup)
        self.reg.record_register_from_hbm = lambda *a, **kw: None
        self.reg.record_register_from_hbm_reconciliation = lambda **kw: None
        ns.update({
            "_DRAM_KV_PAGE_REGISTRY": self.reg,
            "virtual_context_layer_id": lambda name: 0,
            "tiered_dram_page_provider_enabled": lambda: True,
            "tiered_dram_register_from_hbm_enabled": lambda: True,
            "tiered_dram_register_from_hbm_policy": lambda: "all",
            "tiered_dram_register_max_pages_per_forward": lambda: 0,
            "tiered_d2h_source_plan_enabled": lambda: False,
            "tiered_dram_register_async_enabled": lambda: True,
            "tiered_d2h_source_view_dispatch_enabled": lambda: False,
            "get_virtual_context_table": lambda: NS(
                observe_provider_layer=lambda *a: None, record_dram_copies_ready=lambda **kw: self.ready.append(kw)),
            "get_dram_kv_request_key": lambda *a, **kw: 1,
            "_hbm_d2h_page_pending": lambda *a: self.pending,
            "tiered_hbm_source_block_lease_enabled": lambda: True,
            "acquire_tiered_hbm_source_block_lease": self.lease,
            "_release_hbm_source_block_lease": self.released.append,
            "_submit_async_hbm_registration": lambda **kw: self.submissions.append(kw) or True,
            "register_dram_kv_page": self.publish,
        })
        if count is not None:
            self.publish(token_count=count, version=version)

    def pool_lookup(self, **kw):
        s = self.snapshot
        return s if self.storage == "pool" and s is not None and (
            kw["version"] is None or s.version == kw["version"]) else None

    def publish(self, *, token_count, version=0, **kw):
        page = self.ns["TieredDRAMKVPage"](None, None, 0, token_count, version)
        self.snapshot = self.ns["TieredDRAMKVLookup"](page, 2, version, self.storage)
        if self.storage == "dict":
            key = self.ns["TieredKVPageKey"](1, 0, 0, version, "R", "layer.0")
            self.reg._pages[key] = page
            self.reg._latest_versions[(1, 0, 0)] = version

    def lease(self, block_ids):
        lease = tuple(block_ids)
        self.leases.append(lease)
        return lease

    def consumer(self, count=16, version=0):
        return self.reg.lookup(request_id="R", request_key=1, layer_name="layer.0",
                               layer_id=0, logical_page=0, expected_token_count=count, version=version)

    def run(self, *, version=0, **overrides):
        args = dict(layer_name="layer.0", key_cache=Tensor(np.zeros((1, 16, 1, 2))),
                    value_cache=Tensor(np.zeros((1, 16, 1, 2))), block_table=Tensor([[0]]),
                    seq_lens=Tensor([16]), request_ids=["R"], page_versions_by_request={"R": {0: version}},
                    force_all_request_ids={"R"})
        args.update(overrides)
        return self.ns["register_hbm_kv_cache_pages"](**args)

    def verify(self):
        submission = self.submissions[-1]
        keys = ("registered_versions_by_request", "registered_blocks_by_request",
                "registered_token_counts_by_request", "block_size", "layer_id", "pages")
        return self.ns["_mark_hbm_registration_complete"](**{k: submission[k] for k in keys})


class RefreshProbeTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {FLAG: "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_old_partial_pool_and_dict_refresh_only_after_complete(self):
        for storage in ("pool", "dict"):
            for count in (9, 11):
                with self.subTest(storage=storage, count=count):
                    f = Fixture(storage, count)
                    with self.assertRaisesRegex(RuntimeError, "shorter than expected"):
                        f.consumer()
                    self.assertEqual(f.run(), 1)
                    self.assertEqual(f.leases, [(0,)])
                    self.assertEqual(f.released, [])
                    self.assertEqual(f.ready, [])
                    self.assertEqual(f.verify(), 0)  # Enqueue/old copy != full-page proof.
                    with self.assertRaises(RuntimeError):
                        f.consumer()
                    f.publish(token_count=16)
                    self.assertEqual(f.verify(), 1)
                    self.assertEqual(f.consumer().page.token_count, 16)
                    self.assertEqual(f.ready[-1]["page_token_counts"], {0: 16})

    def test_pending_short_copy_defers_then_refreshes_on_next_check(self):
        f = Fixture()
        f.pending = True
        self.assertEqual(f.run(), 0)
        self.assertEqual(f.submissions, [])
        f.pending = False
        self.assertEqual(f.run(), 1)
        self.assertEqual(len(f.submissions), 1)

    def test_current_copy_skips_missing_or_new_version_refreshes(self):
        for storage in ("pool", "dict"):
            f = Fixture(storage, 16)
            self.assertEqual(f.run(), 0)
            self.assertEqual(f.run(version=1), 1)
            self.assertEqual(f.submissions[-1]["pages"][0].version, 1)
            missing = Fixture(storage, None)
            self.assertEqual(missing.run(), 1)

    def test_flagoff_reproduces_old_strict_probe_but_current_page_works(self):
        with patch.dict(os.environ, {FLAG: "0"}):
            for storage in ("pool", "dict"):
                f = Fixture(storage)
                with self.assertRaisesRegex(RuntimeError, "shorter than expected"):
                    f.run()
                self.assertEqual(f.submissions, [])
                self.assertEqual(Fixture(storage, 16).run(), 0)

    def test_exact_page_or_request_exclusion_does_not_copy(self):
        for kwargs in ({"exact_force_pages_by_request": {"R": {1}}},
                       {"allowed_request_ids": set()}, {"block_table": Tensor([[-1]])}):
            f = Fixture()
            self.assertEqual(f.run(**kwargs), 0)
            self.assertEqual(f.submissions, [])

    def test_planned_deferred_source_uses_same_refresh_and_lease_path(self):
        f = Fixture()
        page = NS(logical_page=0, block_id=0, token_start=0, token_count=16, force_all=True)
        plan = NS(matches=lambda *a: True, group_by_request=lambda: {"R": (page,)}, pages=(page,))
        f.ns["tiered_d2h_source_plan_enabled"] = lambda: True
        f.reg.record_d2h_source_plan = lambda **kw: None
        f.reg.record_d2h_source_view_dispatch_submit = lambda *a: None
        f.ns["tiered_d2h_source_view_dispatch_enabled"] = lambda: True
        f.ns["tiered_dram_register_control_dispatcher_enabled"] = lambda: True
        self.assertEqual(f.run(source_page_plan=plan), 1)
        deferred = f.submissions[-1]["pages"][0]
        self.assertIsInstance(deferred, f.ns["_DeferredHBMRegistrationPage"])
        self.assertEqual(deferred.materialize().key.shape[0], 16)
        self.assertEqual(f.leases, [(0,)])
        self.assertEqual(f.ready, [])

    def test_newer_complete_version_is_published_not_the_obsolete_plan(self):
        f = Fixture()
        self.assertEqual(f.run(), 1)
        f.publish(token_count=16, version=1)
        self.assertEqual(f.verify(), 1)
        self.assertEqual(f.ready[-1]["page_versions"], {0: 1})

    def test_copy_submission_fallback_failure_never_publishes_ready(self):
        f = Fixture()
        f.ns["_submit_async_hbm_registration"] = lambda **kw: False
        def copy_error(**kw):
            raise RuntimeError("injected copy failure")
        f.ns["register_dram_kv_page"] = copy_error
        self.assertEqual(f.run(), 1)  # Existing count means attempted pages, not readiness.
        self.assertEqual(f.ready, [])
        self.assertEqual(f.released, [(0,)])
        with self.assertRaises(RuntimeError):
            f.consumer()

    def test_bundle_defaults_off_independent_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(FLAGS.isolation_execution_enabled("DRAM_REFRESH_PROBE"))
        with patch.dict(os.environ, {"VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1"}, clear=True):
            self.assertTrue(FLAGS.isolation_execution_enabled("DRAM_REFRESH_PROBE"))
            with patch.dict(os.environ, {FLAG: "0"}):
                self.assertFalse(FLAGS.isolation_execution_enabled("DRAM_REFRESH_PROBE"))


if __name__ == "__main__":
    unittest.main()
