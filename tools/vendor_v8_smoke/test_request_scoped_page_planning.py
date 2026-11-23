#!/usr/bin/env python3
"""CPU query/lifecycle equivalence, with deterministic no-global-scan guards."""
from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("scoped_planning_vctx",
    ROOT / "vendor_v8/vllm/v1/core/virtual_context.py")
V = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V
SPEC.loader.exec_module(V)


class NoGlobalScan(dict):
    def items(self):
        raise AssertionError("scoped query scanned all page records")

    def values(self):
        raise AssertionError("scoped query scanned all page records")

    def __iter__(self):
        raise AssertionError("scoped query scanned all page IDs")


def populate(table, rid, tokens=41, layers=3, copied_layers=None):
    ids = list(range(1, (tokens + 15) // 16 + 1))
    table.observe_hbm_blocks(rid, (ids,), 16, tokens)
    for layer in range(layers):
        table.observe_provider_layer(rid, layer)
    for layer in range(layers if copied_layers is None else copied_layers):
        table.record_dram_copies_ready(rid, {p: 1 for p in range(len(ids))}, 16,
            layer_id=layer, page_to_block=dict(enumerate(ids)),
            page_token_counts={p: min(16, tokens - p * 16) for p in range(len(ids))})


def legacy_release(table, **kwargs):
    # The pre-optimization collector first saw the entire table; it still
    # applies the original requested-ID check and max-logical-page filter.
    with patch.object(table, "_request_page_records_locked", lambda _: table._pages.items()):
        return table.plan_hbm_release_candidates(**kwargs)


class ScopedPagePlanningTests(unittest.TestCase):
    def setUp(self):
        self.t = V.VirtualContextTable()
        populate(self.t, "A")
        populate(self.t, "B", tokens=64)
        populate(self.t, "C", copied_layers=2)

    def assert_index(self, table):
        indexed = {p for ids in table._page_ids_by_request.values() for p in ids}
        self.assertEqual(indexed, set(table._pages))
        for rid, ids in table._page_ids_by_request.items():
            self.assertTrue(ids)
            self.assertTrue(all(p.request_id == rid for p in ids))
        by_layer = table._page_ids_by_request_layer
        self.assertEqual({p for ids in by_layer.values() for p in ids}, indexed)
        for (rid, layer), ids in by_layer.items():
            self.assertTrue(ids)
            self.assertTrue(all((p.request_id, p.layer_id) == (rid, layer) for p in ids))

    def test_scoped_and_global_release_order_match_reference(self):
        for ids in [None, set(), {"A"}, {"missing"}, {"A", "C"}, {"A", "B"}]:
            for partial in [False, True]:
                for max_pages in [0, 1, 4]:
                    for max_logical in [0, 1, 2]:
                        kw = dict(request_ids=ids, allow_partial_pages=partial,
                            max_pages=max_pages, max_logical_pages=max_logical)
                        with self.subTest(**kw):
                            self.assertEqual(self.t.plan_hbm_release_candidates(**kw),
                                             legacy_release(self.t, **kw))

    def test_missing_layer_never_becomes_ready(self):
        self.assertEqual(self.t.plan_hbm_release_candidates(request_ids={"C"},
            allow_partial_pages=True), [])

    def test_partial_tail_needs_current_complete_copy(self):
        t = self.t
        self.assertEqual(len(t.plan_hbm_release_candidates(request_ids={"A"})), 6)
        self.assertEqual(len(t.plan_hbm_release_candidates(request_ids={"A"},
            allow_partial_pages=True)), 9)
        t.observe_hbm_blocks("A", ([1, 2, 3],), 16, 45)
        self.assertEqual(len(t.plan_hbm_release_candidates(request_ids={"A"},
            allow_partial_pages=True)), 6)
        self.assert_index(t)

    def test_version_pinning_matches_naive_scan(self):
        for rid in ["A", "B", "C"]:
            for layer in [-1, 0, 1, 2, 3]:
                expected = {p.logical_token_page: r.version for p, r in self.t._pages.items()
                    if p.request_id == rid and p.kv_cache_group_id == 0
                    and p.layer_id == layer and p.kv_head_group == -1 and r.state == "ready"}
                self.assertEqual(self.t.pin_attention_page_versions([rid], layer_id=layer),
                    {rid: expected} if expected else {})

    def test_scoped_plan_pin_debug_summary_never_iterate_global_records(self):
        t = self.t
        debug = t.get_request_debug_stats("A")
        attention = t.summarize_attention_plan("prefill", ["A"], 16)
        release = t.plan_hbm_release_candidates(request_ids={"A"}, allow_partial_pages=True)
        pinned = t.pin_attention_page_versions(["A"], layer_id=0)
        t._pages = NoGlobalScan(t._pages)
        self.assertEqual(t.get_request_debug_stats("A"), debug)
        self.assertEqual(t.summarize_attention_plan("prefill", ["A"], 16), attention)
        self.assertEqual(t.plan_hbm_release_candidates(request_ids={"A"},
            allow_partial_pages=True), release)
        self.assertEqual(t.pin_attention_page_versions(["A"], layer_id=0), pinned)
        self.assertEqual(t.plan_hbm_release_candidates(request_ids=set()), [])

    def test_release_commit_keeps_same_candidates_and_other_requests(self):
        before = self.t.get_request_debug_stats("B")
        expected = self.t.plan_hbm_release_candidates(request_ids={"A"}, allow_partial_pages=True)
        actual = self.t.mark_hbm_released_to_dram(request_ids={"A"}, allow_partial_pages=True)
        self.assertEqual(actual, expected)
        self.assertEqual(self.t.get_request_debug_stats("B"), before)
        self.assertEqual(self.t.hbm_reload_required_page_count({"A"}), 12)
        self.assert_index(self.t)

    def test_orphan_layer_is_queryable_and_deleted_on_abort(self):
        t = V.VirtualContextTable()
        t.record_dram_copies_ready("orphan", {0: 9}, 16, layer_id=7, page_to_block={0: 42})
        self.assertFalse(t._aggregate_pages_by_request.get("orphan"))
        self.assertEqual(len(t.plan_hbm_release_candidates(request_ids={"orphan"})), 1)
        self.assertEqual(t.pin_attention_page_versions(["orphan"], layer_id=7), {"orphan": {0: 9}})
        t.release_request("orphan")
        self.assert_index(t)
        self.assertEqual(t._page_ids_by_request, {})

    def test_retire_private_publish_and_late_copy_abort(self):
        t = self.t
        t.retire_layerwise_hbm(["A"])
        self.assertFalse(t._page_ids_by_request.get("A"))
        version = t.reserve_layerwise_provider_version("A")
        t.publish_layerwise_dram(request_id="A", logical_tokens=25, block_size=16,
            released_block_ids=(1, 2), layer_ids=(0, 1, 2), version=version)
        self.assertEqual(t.hbm_reload_required_page_count({"A"}), 8)
        self.assert_index(t)
        t.release_request("A")
        t.record_dram_copies_ready("A", {0: 100}, 16, layer_id=0, page_to_block={0: 1},
                                  require_existing_request=True)
        self.assertFalse(t._page_ids_by_request.get("A"))
        with self.assertRaises(RuntimeError):
            t.publish_layerwise_dram(request_id="A", logical_tokens=25, block_size=16,
                released_block_ids=(1, 2), layer_ids=(0, 1, 2), version=version)
        self.assert_index(t)

    def test_id_reuse_and_record_replacement_keep_index_coherent(self):
        t = self.t
        old_order = tuple(t._page_ids_by_request["A"])
        populate(t, "A")
        self.assertEqual(tuple(t._page_ids_by_request["A"]), old_order)
        t.release_request("A")
        populate(t, "A", tokens=16, layers=1)
        self.assertEqual(len(t._page_ids_by_request["A"]), 2)
        self.assert_index(t)

    def test_reload_ack_preserves_index_and_requires_every_layer(self):
        t = self.t
        t.mark_hbm_released_to_dram(request_ids={"A"}, allow_partial_pages=True)
        original_ids = tuple(t._page_ids_by_request["A"])
        for layer in range(3):
            t.mark_hbm_reload_complete("A", {0: 101, 1: 102, 2: 103}, 16,
                layer_id=layer, page_token_counts={0: 16, 1: 16, 2: 9},
                provider_versions={0: 1, 1: 1, 2: 1})
            self.assert_index(t)
            self.assertEqual(tuple(t._page_ids_by_request["A"]), original_ids)
            if layer < 2:
                self.assertGreater(t.hbm_reload_required_page_count({"A"}), 0)
        self.assertEqual(t.hbm_reload_required_page_count({"A"}), 0)

    def test_random_lifecycle_and_candidate_equivalence(self):
        t, rng = self.t, random.Random(23)
        for _ in range(80):
            rid = rng.choice(["X", "Y", "Z"])
            t.release_request(rid)
            if rng.choice([True, False]):
                populate(t, rid, tokens=rng.randint(1, 96), layers=3,
                         copied_layers=rng.randint(0, 3))
            for partial in [False, True]:
                kw = dict(request_ids={rid}, allow_partial_pages=partial, max_logical_pages=2)
                self.assertEqual(t.plan_hbm_release_candidates(**kw), legacy_release(t, **kw))
            self.assert_index(t)

    def test_layer_queries_never_traverse_other_layers(self):
        t = self.t
        t.mark_hbm_released_to_dram(request_ids={"A"}, allow_partial_pages=True)
        expected_pin = t.pin_attention_page_versions(["A"], layer_id=1)
        expected_reload = t.plan_hbm_reload_pages(request_ids={"A"}, layer_id=1)
        t._page_ids_by_request["A"] = NoGlobalScan(t._page_ids_by_request["A"])
        for key, ids in list(t._page_ids_by_request_layer.items()):
            if key != ("A", 1):
                t._page_ids_by_request_layer[key] = NoGlobalScan(ids)
        self.assertEqual(t.pin_attention_page_versions(["A"], layer_id=1), expected_pin)
        self.assertEqual(t.plan_hbm_reload_pages(request_ids={"A"}, layer_id=1), expected_reload)

    def test_layer_reload_filter_matches_global_reference_and_limits(self):
        t = self.t
        t.mark_hbm_released_to_dram(request_ids={"A", "B"}, allow_partial_pages=True)
        all_pages = t.plan_hbm_reload_pages()
        for ids in [set(), {"A"}, {"A", "B"}, {"missing"}]:
            for layer in [-1, 0, 1, 2, 999]:
                for limit in [0, 1, 3]:
                    expected = [p for p in all_pages if p.request_id in ids and p.layer_id == layer]
                    if limit:
                        expected = expected[:limit]
                    self.assertEqual(t.plan_hbm_reload_pages(request_ids=ids, layer_id=layer,
                                                            max_pages=limit), expected)

    def test_layer_ack_immediately_removes_copy_plan_not_identity(self):
        t = self.t
        t.mark_hbm_released_to_dram(request_ids={"A"}, allow_partial_pages=True)
        before = tuple(t._page_ids_by_request_layer[("A", 0)])
        t.mark_hbm_reload_complete("A", {0: 10, 1: 11, 2: 12}, 16,
            layer_id=0, page_token_counts={0: 16, 1: 16, 2: 9},
            provider_versions={0: 1, 1: 1, 2: 1})
        self.assertFalse(t.plan_hbm_reload_pages(request_ids={"A"}, layer_id=0))
        self.assertTrue(t.plan_hbm_reload_pages(request_ids={"A"}, layer_id=1))
        self.assertEqual(tuple(t._page_ids_by_request_layer[("A", 0)]), before)
        self.assert_index(t)

    def test_layer_index_preserves_group_head_and_unready_filters(self):
        t = V.VirtualContextTable()
        with t._lock:
            for group in [0, 1]:
                for head in [-1, 0, 1]:
                    for layer in [-1, 0, 1]:
                        for page in range(3):
                            t._set_page_locked(V.LogicalPageId("X", group, page, layer, head),
                                V.ResidencyRecord(tier="DRAM", state="ready" if page < 2 else "copying",
                                    version=7, page_size_tokens=16, block_size_tokens=16,
                                    dram_copy_version=7, dram_copy_tokens=16, released_gpu_block_id=page + 1))
        for group in [0, 1]:
            for head in [-1, 0, 1]:
                for layer in [-1, 0, 1]:
                    self.assertEqual(t.pin_attention_page_versions(["X"], group_id=group,
                        layer_id=layer, kv_head_group=head), {"X": {0: 7, 1: 7}})
        self.assert_index(t)


def layer_benchmark():
    """Old request-wide pin loop vs indexed pin; identical snapshot results."""
    t = V.VirtualContextTable()
    populate(t, "A", tokens=50000, layers=16)
    def old_pin(layer):
        # Exact pre-patch identity traversal/filter order; no cached readiness.
        versions = {}
        with t._lock:
            for page, record in t._request_page_records_locked({"A"}):
                if page.kv_cache_group_id != 0 or page.layer_id != layer or page.kv_head_group != -1:
                    continue
                if record.state == "ready":
                    versions[page.logical_token_page] = record.version
        return {"A": versions} if versions else {}
    reference = [old_pin(layer) for layer in range(16)]
    for name, call in [("request_wide_pin", old_pin), ("layer_indexed_pin",
            lambda layer: t.pin_attention_page_versions(["A"], layer_id=layer))]:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            values = [call(layer) for layer in range(16)]
            times.append(1000 * (time.perf_counter() - start))
            assert values == reference
        print(f"{name}: 50K_context_16_layers_median_ms={statistics.median(times):.3f}")


def benchmark():
    """Descriptive local CPU medians; never a GPU or E2E performance claim."""
    t = V.VirtualContextTable()
    populate(t, "target", tokens=2048, layers=16)
    for i in range(8):
        rid = f"peer-{i}"
        for layer in range(16):
            t.observe_provider_layer(rid, layer)
        with t._lock:
            for page in range(3125):
                for layer in range(-1, 16):
                    t._set_page_locked(V.LogicalPageId(rid, 0, page, layer),
                        V.ResidencyRecord(tier="DRAM", state="ready", version=1,
                            page_size_tokens=16, block_size_tokens=16, dram_copy_version=1,
                            dram_copy_tokens=16, released_gpu_block_id=1))
    results = {}
    for name, call in [("legacy_global", lambda: legacy_release(t,
            request_ids={"target"}, allow_partial_pages=True)),
            ("request_scoped", lambda: t.plan_hbm_release_candidates(
                request_ids={"target"}, allow_partial_pages=True))]:
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            values = call()
            samples.append(1000 * (time.perf_counter() - start))
        results[name] = values
        print(f"{name}: median_ms={statistics.median(samples):.3f} records={len(values)}")
    assert results["legacy_global"] == results["request_scoped"]


if __name__ == "__main__":
    if "--layer-benchmark" in sys.argv:
        layer_benchmark()
    elif "--benchmark" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--benchmark", action="store_true")
        parser.parse_args()
        benchmark()
    else:
        unittest.main()
