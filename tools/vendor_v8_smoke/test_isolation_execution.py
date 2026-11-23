#!/usr/bin/env python3
"""CPU contracts for bounded Ring expansion, P1 waits and owned Prefill RoPE."""
from __future__ import annotations

import ast
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_isolation_memory import V, flags


def methods(path, class_name, names, namespace=None):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == class_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in names]
    ns = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name="Harness", bases=[], keywords=[], body=selected, decorator_list=[]),
    ], type_ignores=[])), str(path), "exec"), ns)
    return ns["Harness"]


SCHED = V / "v1/core/sched/scheduler.py"
LLAMA = V / "model_executor/models/llama.py"


class RingQueryTests(unittest.TestCase):
    def limit(self, budget, maximum=16384):
        return flags.isolation_ring_query_token_limit(
            max_tokens=maximum, heads=32, head_size=64, element_bytes=2,
            budget_bytes=budget, block_size=16)

    def test_rounding_and_explicit_ceiling(self):
        for budget in (1, 2 << 20, 35 << 20, 80 << 20, 320 << 20):
            tokens = self.limit(budget)
            self.assertEqual(tokens % 16, 0)
            if tokens:
                self.assertLessEqual(flags.isolation_ring_query_bytes(tokens, 32, 64, 2), budget)
            if tokens < 16384:
                self.assertGreater(flags.isolation_ring_query_bytes(tokens + 16, 32, 64, 2), budget)
        self.assertEqual(self.limit(320 << 20), 16384)
        self.assertEqual(self.limit(320 << 20, 4096), 4096)
        with self.assertRaises(ValueError):
            self.limit(0)

    def test_old_memory_bundle_does_not_enable_new_features(self):
        with patch.dict(os.environ, {"VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1"}, clear=True):
            self.assertFalse(flags.isolation_execution_enabled("RING_QUERY_EXPANSION"))
        with patch.dict(os.environ, {"VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1",
                                   "VLLM_ISOLATION_RING_QUERY_EXPANSION": "0"}, clear=True):
            self.assertFalse(flags.isolation_execution_enabled("RING_QUERY_EXPANSION"))
            self.assertTrue(flags.isolation_execution_enabled("PREFILL_FUSED_ROPE"))

    def harness(self):
        h = methods(SCHED, "Scheduler", {"_ttft_cap_selected_prefill_chunk"})()
        h.enable_ttft_prefill_chunk_scheduling = True
        h.isolation_ring_query_expansion = True
        h.isolation_ring_query_max_tokens = 16384
        h.ttft_prefill_native_chunk_max_tokens = 16384
        h.ttft_prefill_chunk_scheduler = NS(
            logical_prefill_progress=lambda r: max(r.num_computed_tokens, r.virtual_context_deferred_logical_computed_tokens),
            logical_prefill_boundary=lambda r: r.boundary,
            cap_prefill_tokens=lambda r, n, **kw: min(n, 2048, r.boundary - max(
                kw.get("num_computed_tokens") or r.num_computed_tokens,
                r.virtual_context_deferred_logical_computed_tokens)))
        return h

    def test_modes_rollback_tail_and_predicted_progress(self):
        h = self.harness()
        r = NS(num_computed_tokens=0, virtual_context_deferred_logical_computed_tokens=0, boundary=50000)
        cap = h._ttft_cap_selected_prefill_chunk
        for mode in ("native", "ring"):
            self.assertEqual(cap(r, 16384, execution_mode=mode), 16384)
            self.assertEqual(cap(r, 16384, execution_mode=mode, num_computed_tokens=49000), 1000)
        r.virtual_context_deferred_logical_computed_tokens = 49984
        self.assertEqual(cap(r, 16384, execution_mode="ring"), 16)
        r.virtual_context_deferred_logical_computed_tokens = 0
        h.isolation_ring_query_max_tokens = 1024
        self.assertEqual(cap(r, 16000, execution_mode="ring"), 1024)
        h.isolation_ring_query_expansion = False
        self.assertEqual(cap(r, 16384, execution_mode="ring"), 2048)
        self.assertEqual(cap(r, 16384, execution_mode="native"), 16384)

    def config(self):
        env = {"VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1",
               "VLLM_VIRTUAL_CONTEXT_RING_BACKEND": "1",
               "VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE": "1"}
        h = methods(SCHED, "Scheduler", {"_configure_isolation_ring_query_budget"}, {
            "os": os, "isolation_execution_enabled": flags.isolation_execution_enabled,
            "isolation_ring_query_token_limit": flags.isolation_ring_query_token_limit,
            "_read_bool_env": lambda name: os.getenv(name, "0") == "1",
            "PrefillChunkSchedulingPolicy": NS(SRPT="srpt"), "logger": Mock(),
        })()
        h.enable_ttft_prefill_chunk_scheduling = True
        h.ttft_prefill_chunk_scheduler = NS(policy="srpt")
        h.virtual_context_ring_bw_dynamic = h.virtual_context_ring_bw_use_scheduled_headroom = True
        h.virtual_context_ring_bw_policy = "safe_capacity"
        h.parallel_config = NS(tensor_parallel_size=1, pipeline_parallel_size=1)
        h.vllm_config = NS(model_config=NS(dtype="torch.float16",
            get_num_attention_heads=lambda _: 32, get_head_size=lambda: 64))
        h.max_num_scheduled_tokens = h.virtual_context_ring_prefill_admission_tokens_max = 16384
        h.block_size = 16
        h.virtual_context_ring_min_progress_tokens = 2048
        return h, env

    def test_config_fail_closed_and_workspace_budget(self):
        h, env = self.config()
        with patch.dict(os.environ, env, clear=True):
            h._configure_isolation_ring_query_budget()
            self.assertEqual(h.isolation_ring_query_max_tokens, 16384)
            h.ttft_prefill_chunk_scheduler.policy = "slo_srpt"
            with self.assertRaises(ValueError):
                h._configure_isolation_ring_query_budget()
        for change in ({"VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES": "1"},
                       {"VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE": "0"},
                       {"VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL": "1"}):
            h, env = self.config()
            with patch.dict(os.environ, {**env, **change}, clear=True), self.assertRaises(ValueError):
                h._configure_isolation_ring_query_budget()
        h, env = self.config()
        env.update(VLLM_PD_PREFILL_WORKSPACE_ENABLED="1", VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS="4096")
        with patch.dict(os.environ, env, clear=True):
            h._configure_isolation_ring_query_budget()
            self.assertEqual(h.isolation_ring_query_max_tokens, 4096)

    def test_actual_final_admission_keeps_zero_capacity_and_decode_cap(self):
        h = methods(SCHED, "Scheduler", {
            "_cap_virtual_context_ring_prefill_admission_tokens",
            "_cap_prefill_admission_with_active_decode",
        }, {"_read_bool_env": lambda _: False})()
        h.virtual_context_ring_prefill_admission_tokens = 2048
        h.virtual_context_ring_min_progress_tokens = 2048
        h.ttft_prefill_native_chunk_max_tokens = 16384
        h._debug_has_prefill_remaining = lambda _: True
        h.virtual_context_decode_hotset_first = True
        h._debug_running_decode_resident_blocks = lambda: 500
        h._virtual_context_reload_required_page_count = lambda *args: 1
        h.virtual_context_active_decode_ring_prefill_tokens = 2048
        h.virtual_context_ring_bw_dynamic = True
        h.virtual_context_ring_bw_policy = "safe_capacity"
        r = NS(request_id="ring")
        for elastic in (False, True):
            h.virtual_context_active_decode_elastic_prefill = elastic
            for safe, expected in ((0, 0), (1024, 0), (4096, 4096 if elastic else 2048),
                                   (16384, 16384 if elastic else 2048)):
                h._dynamic_virtual_context_ring_prefill_admission_tokens = lambda: safe
                self.assertEqual(h._cap_virtual_context_ring_prefill_admission_tokens(
                    r, 16384, execution_mode="ring"), expected)

    def test_predictive_claim_uses_same_expansion_and_cold_decode_cap(self):
        from test_scheduled_prefill_fit_residency import load_capacity_claim_harness
        h = load_capacity_claim_harness()()
        h._ttft_cap_selected_prefill_chunk = self.harness()._ttft_cap_selected_prefill_chunk
        h.enable_ttft_prefill_chunk_scheduling = True
        h.max_num_scheduled_tokens = 16384
        h.scheduler_config = NS(long_prefill_token_threshold=0)
        h._virtual_context_running_decode_reload_required_page_count = lambda: 0
        h._virtual_context_reload_required_page_count = lambda *args: 0
        h.virtual_context_decode_hotset_first = True
        h.virtual_context_prefill_ring_tokens_with_decode_cold = 2048
        h.virtual_context_staging_reserved_blocks = h.virtual_context_ring_min_free_blocks = 8
        h.virtual_context_decode_hotset_reserve_blocks = 16
        h._scheduled_prefill_fit_headroom_partition = lambda: (82, 0)
        captured = []
        h.kv_cache_manager = NS(block_pool=NS(get_num_free_blocks=lambda: 1000),
                               estimate_num_blocks_to_allocate=lambda r, n, **kw: captured.append(n) or (n + 15) // 16)
        h.num_lookahead_tokens = 0
        r = NS(request_id="ring", boundary=50000, num_prompt_tokens=50000,
               num_computed_tokens=0, virtual_context_deferred_logical_computed_tokens=18432)
        for safe, logical, expected in ((16384, 18432, 16384), (4096, 18432, 4096),
                                        (16384, 49000, 1000)):
            claim = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=safe,
                logical_tokens=logical, full_native=False)
            self.assertEqual(claim.chunk_tokens, expected)
            self.assertEqual(captured[-1], expected)
            self.assertEqual(claim.hard_reserved_blocks, 114)
        h._virtual_context_running_decode_reload_required_page_count = lambda: 50
        claim = h._scheduled_prefill_fit_capacity_claim(r, ring_token_cap=16384,
            logical_tokens=18432, full_native=False)
        self.assertEqual(claim.chunk_tokens, 2048)


class BoundaryTests(unittest.TestCase):
    def test_completed_event_does_not_submit_waiter(self):
        from test_p1_parked_prefill import P
        event = NS(query=Mock(return_value=True), synchronize=Mock())
        with patch.object(P, "_event_wait_pool", Mock()) as pool:
            events = [event]
            P.kv_retention_bound_prefill_submission(events, record_event=lambda: event, completed_fastpath=True)
            pool.submit.assert_not_called()
            event.synchronize.assert_not_called()
            self.assertEqual(events, [event])

    def test_unfinished_and_query_error_do_not_grant_completion(self):
        from test_p1_parked_prefill import P
        event = NS(query=Mock(return_value=False), synchronize=Mock())
        events = [event]
        P.kv_retention_bound_prefill_submission(events, record_event=lambda: event, completed_fastpath=True)
        event.synchronize.assert_called_once()
        event.query.side_effect = RuntimeError("CUDA query failed")
        with self.assertRaisesRegex(RuntimeError, "CUDA query failed"):
            P.kv_retention_bound_prefill_submission(events, record_event=lambda: event, completed_fastpath=True)
        self.assertEqual(len(events), 2)

    def test_yield_before_old_group_wait_and_during_wait(self):
        for already_pending in (True, False):
            events = []
            pending = [already_pending]
            def consume():
                result = pending[0]
                pending[0] = False
                return result
            def wait(*args, **kwargs):
                events.append("wait")
                pending[0] = True
            h = methods(LLAMA, "LlamaModel", {"_kv_retention_advance_prefill"}, {
                "time": NS(perf_counter=lambda: 0),
                "kv_retention_bound_prefill_submission": wait,
            })()
            h.end_layer = 16
            h.kv_retention_p1_parked_enabled = h.isolation_p1_boundary_fastpath = True
            h._forward_decoder_layer_range = lambda **kw: events.append("forward") or ("h", "r")
            state = NS(next_layer=0, end_layer=16, positions=None, hidden_states=None,
                       residual=None, aux_hidden_states=[], workspace_active=True)
            execution = NS(group_layers=2, record_group=lambda **kw: None,
                           consume_yield_request=consume, make_boundary=lambda **kw: kw)
            boundary = h._kv_retention_advance_prefill(NS(state=state, execution=execution))
            self.assertEqual(boundary["next_layer"], 2)
            self.assertEqual(events, ["forward"] if already_pending else ["forward", "wait"])


class PrefillRopeTests(unittest.TestCase):
    def test_phase_local_dispatch_no_shared_mutation(self):
        h = methods(LLAMA, "LlamaAttention", {"_apply_rotary_emb"})()
        rope = Mock(return_value="normal")
        rope.forward_cuda.return_value = "inplace"
        h.rotary_emb = rope
        h.isolation_prefill_fused_rope = True
        h.opmux_enabled = False
        self.assertEqual(h._apply_rotary_emb(None, NS(is_cuda=True), None, pd_prefill_workspace=True), "inplace")
        for enabled, workspace, cuda, opmux in ((False, True, True, False), (True, False, True, False),
                                              (True, True, False, False), (True, True, True, True)):
            h.isolation_prefill_fused_rope, h.opmux_enabled = enabled, opmux
            self.assertEqual(h._apply_rotary_emb(None, NS(is_cuda=cuda), None,
                                               pd_prefill_workspace=workspace), "normal")
        self.assertEqual(rope.forward_cuda.call_count, 1)

    def test_rope_is_in_attention_not_mlp_constructor(self):
        tree = ast.parse(LLAMA.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in ("LlamaMLP", "LlamaAttention"):
                text = ast.unparse(node)
                self.assertEqual("isolation_prefill_fused_rope" in text, node.name == "LlamaAttention")
                if node.name == "LlamaMLP":
                    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
                    self.assertIn("self.act_fn = SiluAndMul()", ast.unparse(init))


class TextInputTests(unittest.TestCase):
    def test_only_permanently_unused_staging_is_elided(self):
        opts = dict(supports_mm_inputs=False, prompt_embeds=False, speculative=False,
                    encoder_decoder=False, pooling=False, tensor_parallel_size=1,
                    pipeline_parallel_size=1)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(flags.isolation_unused_embed_staging(**opts))
        with patch.dict(os.environ, {"VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1"}, clear=True):
            self.assertTrue(flags.isolation_unused_embed_staging(**opts))
            for name in opts:
                bad = 2 if name.endswith("parallel_size") else True
                self.assertFalse(flags.isolation_unused_embed_staging(**(opts | {name: bad})), name)

    def test_memory_sampling_calls_only_host_stats_on_sampled_ids(self):
        cuda = Mock()
        cuda.memory_stats.return_value = {"allocated_bytes.all.current": 123,
                                          "reserved_bytes.all.current": 456}
        for i in (-1, 1, 99, 101):
            self.assertEqual(flags.isolation_allocator_snapshot(cuda, 0, iteration=i, sample_every=100), {})
        cuda.memory_stats.assert_not_called()
        stats = flags.isolation_allocator_snapshot(cuda, 0, iteration=100, sample_every=100)
        self.assertEqual(stats["isolation_gpu_allocated_bytes"], 123)
        self.assertEqual(stats["isolation_gpu_reserved_bytes"], 456)
        self.assertEqual(len(stats), 6)
        self.assertEqual(cuda.method_calls, [unittest.mock.call.memory_stats(0)])
        from test_isolation_memory import load_symbols
        path = V / "v1/core/sched/observability.py"
        tree = ast.parse(path.read_text())
        ns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id.startswith("_LIGHTWEIGHT_WORKER_TIMING_")
                    for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        ns = load_symbols(path, {"compact_worker_debug_timings"}, ns)
        self.assertEqual(ns["compact_worker_debug_timings"]({**stats, "large_unknown": []}), stats)


if __name__ == "__main__":
    unittest.main()
