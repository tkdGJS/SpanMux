#!/usr/bin/env python3
"""CPU execution of real offload adapters, with fake CUDA streams/tensors.

CUDA numerical coverage is separate; these tests prove contracts/call ordering,
not device execution or performance. Run directly without pytest or PyTorch.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_kv_retention_layerwise_checkpoint as C

ROOT = C.ROOT
BASE = ROOT / "vendor_v8/vllm"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {"vllm.model_executor.layers.ttft_layerwise_prefill": C.M}):
    M = load_module("kv_offload_contract", BASE / "model_executor/layers/kv_retention_layerwise.py")
V = load_module("kv_offload_virtual_context", BASE / "v1/core/virtual_context.py")


def methods(path, cls_name, wanted, namespace):
    namespace.setdefault("has_native_reload_authority", C.HAS_NATIVE_RELOAD_AUTHORITY)
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(selected) == len(wanted), wanted
    node = ast.ClassDef(name="Adapter", bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Adapter"]


ALLOC = methods(BASE / "v1/core/kv_cache_manager.py", "KVCacheManager", {"release_layerwise_batch"}, {})
SCHED = methods(BASE / "v1/core/sched/scheduler.py", "Scheduler", {
    "layerwise_offload_needed", "commit_layerwise_kv_offload", "complete_layerwise_kv_offload",
}, {"TieredAllocatorInvariantError": RuntimeError, "RequestStatus": NS(PREEMPTED="preempted")})
ENGINE = methods(BASE / "v1/engine/core.py", "EngineCore", {
    "_suspend_ttft_prefill_boundary", "_request_ttft_prefill_preemption_if_needed",
    "_pd_sm_output_tbt_state",
}, {
    "KVRetentionLayerwiseTransfer": M.KVRetentionLayerwiseTransfer,
})


def plan():
    runner, state = C.make_runner_state()
    with patch.dict(os.environ, {C.FLAG: "1"}):
        runner._capture_ttft_kv_checkpoint(state)
    return state.kv_checkpoint


def ticket():
    return M.KVRetentionLayerwiseTransfer(plan(), 128)


def output():
    return NS(scheduler_iteration_id=7, pd_phase_submit_seq=9, pd_phase_slot=1,
              num_scheduled_tokens={"long": 5, "short": 2}, pd_prefill_mode="native",
              pd_sm_profile_id="d30_p10", pd_sm_profile_epoch=1,
              kv_retention_layerwise_transfer=None,
              kv_retention_layerwise_offload_requested=False,
              kv_retention_layerwise_pressure_checked=False,
              ttft_layerwise_prefill_preemption=True)


def allocator(t):
    obj = ALLOC()
    obj.rows = {row.request_id: [NS(block_id=b, ref_cnt=1, is_null=False) for b in row.block_ids]
                for row in t.checkpoint.rows}
    obj.log = []
    obj.virtual_context = NS(retire_layerwise_hbm=lambda ids: obj.log.append(("retire", tuple(ids))))
    obj.block_pool = NS(evict_blocks=lambda blocks: obj.log.append(("evict", set(blocks))))
    obj.get_blocks = lambda rid: NS(blocks=(obj.rows[rid],))
    obj.free = lambda req, **kw: obj.log.append(("free", req.request_id, kw))
    return obj


class TransferTests(unittest.TestCase):
    def test_trace_keeps_receipt_scalars_not_private_owners(self):
        obs = load_module("kv_offload_obs", BASE / "v1/core/sched/observability.py")
        fields = ticket().debug()
        fields.update(kv_retention_offload_d2h_wait_s=0.1,
                      kv_retention_offload_budget_declined=0.0,
                      kv_retention_offload_ring_layers=2.0)
        self.assertEqual(len(fields), 8)
        self.assertEqual(obs.compact_worker_debug_timings(
            dict(fields, kv_retention_offload_ticket=ticket(), kv_retention_offload_buffers=[1])), fields)

    def test_gate_default_off_and_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(M.kv_retention_layerwise_offload_enabled())
        with patch.dict(os.environ, {"VLLM_KV_RETENTION_LAYERWISE_OFFLOAD": "1"}):
            self.assertTrue(M.kv_retention_layerwise_offload_enabled())

    def test_release_exactly_once_and_no_early_ring(self):
        t = ticket()
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.require_ring()
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.commit_release(2)
        t.commit_release(3)
        t.require_ring()
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.commit_release(3)
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.retain_native()

    def test_owner_and_successive_boundary(self):
        t = ticket()
        bad = output()
        bad.pd_phase_submit_seq += 1
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.validate_output(bad)
        t.commit_release(3)
        nxt = C.boundary(next_layer=3, model_yield_count=2, profile_epoch=2, profile_id="d0_p40")
        t.advance(nxt)
        t.validate_boundary(nxt)
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.advance(nxt)
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.mark_published(3)
        t.mark_published(4)
        t.mark_consumed()
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            t.mark_consumed()

    def test_row_geometry_no_shared_physical_pages(self):
        p = plan()
        bad = replace(p, rows=(p.rows[0], replace(p.rows[1], block_ids=(1,))))
        with self.assertRaisesRegex(C.M.TTFTLayerwisePrefillError, "shared"):
            M.KVRetentionLayerwiseTransfer(bad, 1)

    def test_context_restored_on_error_and_retains_authority(self):
        owner = NS(ticket=ticket())
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            with M.kv_retention_ring_scope(owner):
                self.fail("unreleased owner entered Ring")
        owner.ticket.commit_release(3)
        with self.assertRaises(ValueError):
            with M.kv_retention_ring_scope(owner):
                self.assertIs(M.active_kv_retention_ring(), owner)
                raise ValueError("compute failure")
        self.assertIsNone(M.active_kv_retention_ring())

    def test_global_host_budget_and_idempotent_release(self):
        with patch.dict(os.environ, {"VLLM_KV_RETENTION_LAYERWISE_HOST_BYTES": "100"}):
            a = M.reserve_kv_retention_host_bytes(60)
            self.assertIsNone(M.reserve_kv_retention_host_bytes(41))
            b = M.reserve_kv_retention_host_bytes(40)
            a.release()
            a.release()
            self.assertIsNone(M.reserve_kv_retention_host_bytes(61))
            b.release()
            self.assertEqual(M._HOST_BYTES, 0)

    def test_allocator_preflights_every_row_before_mutation(self):
        t = ticket()
        a = allocator(t)
        a.rows["short"][0].block_id = 44
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            a.release_layerwise_batch(t, [NS(request_id=r.request_id) for r in t.checkpoint.rows])
        self.assertEqual(a.log, [])
        self.assertEqual(t.state, "copy_ready")

    def test_allocator_eviction_before_free_preserves_request_counters(self):
        t = ticket()
        a = allocator(t)
        reqs = [NS(request_id=r.request_id, num_computed_tokens=99, ledger=object()) for r in t.checkpoint.rows]
        self.assertTrue(a.release_layerwise_batch(t, reqs))
        self.assertEqual([x[0] for x in a.log], ["evict", "retire", "free", "free"])
        self.assertEqual(a.log[0][1], {1, 2, 4})
        self.assertEqual([r.num_computed_tokens for r in reqs], [99, 99])
        self.assertEqual(t.state, "released")

    def test_shared_reference_declines_without_any_free(self):
        t = ticket()
        a = allocator(t)
        a.rows["long"][0].ref_cnt = 2
        self.assertFalse(a.release_layerwise_batch(t, [NS(request_id=r.request_id) for r in t.checkpoint.rows]))
        self.assertEqual(a.log, [])
        self.assertEqual(t.state, "retained")

    def test_virtual_retirement_preserves_version_and_no_old_hbm(self):
        v = V.VirtualContextTable()
        v.observe_hbm_blocks(request_id="a", block_ids_by_group=([1, 2],), block_size=16, logical_tokens=19)
        old = v._request_versions["a"]
        v.retire_layerwise_hbm(["a"])
        self.assertFalse(v._pages)
        self.assertEqual(v._request_versions["a"], old)
        version = v.reserve_layerwise_provider_version("a")
        v.publish_layerwise_dram(request_id="a", logical_tokens=19, block_size=16,
                                released_block_ids=(1, 2), layer_ids=(4, 7), version=version)
        pages = v.plan_hbm_reload_pages(request_ids=["a"])
        self.assertTrue(pages)
        self.assertTrue(all(r.gpu_block_id is None and r.tier == "DRAM" for r in v._pages.values()))
        tails = [r.page_size_tokens for p, r in v._pages.items() if p.logical_token_page == 1]
        self.assertEqual(tails, [3, 3, 3])
        with self.assertRaises(RuntimeError):
            v.publish_layerwise_dram(request_id="a", logical_tokens=19, block_size=16,
                                    released_block_ids=(1, 2), layer_ids=(4, 7), version=version - 1)


class LifecycleTests(unittest.TestCase):
    def test_sm_controller_uses_effective_ring_without_changing_output_owner(self):
        o, t = output(), ticket()
        o.request_debug = {}
        o.kv_retention_layerwise_transfer = t
        self.assertEqual(ENGINE._pd_sm_output_tbt_state(o)[2], "native")
        t.commit_release(3)
        self.assertEqual(ENGINE._pd_sm_output_tbt_state(o)[2], "ring")
        self.assertEqual(o.pd_prefill_mode, "native")
        t.mark_published(4)
        self.assertEqual(ENGINE._pd_sm_output_tbt_state(o)[2], "ring")

    def test_copy_wait_does_not_repeat_pressure_scan_or_rearm_yield(self):
        e, o = ENGINE(), output()
        e.ttft_layerwise_prefill_preemption_enabled = True
        e.kv_retention_layerwise_offload_enabled = True
        e._pd_owned_phase_depth = lambda *a: 1
        e._pd_phase_inflight_limit = lambda *a: 2
        e.scheduler = NS(ttft_prefill_chunk_scheduler=NS(
            policy="srpt", remaining_prefill_tokens=lambda req: self.fail("repeated scan")))
        o.kv_retention_layerwise_pressure_checked = True
        future = NS(done=lambda: False)
        self.assertFalse(e._request_ttft_prefill_preemption_if_needed(
            {"prefill": [(future, o, future)]}))

    def test_offloaded_runner_still_rejects_changed_cpu_input_rows(self):
        runner, state = C.make_runner_state()
        with patch.dict(os.environ, {C.FLAG: "1"}):
            runner._capture_ttft_kv_checkpoint(state)
        t = M.KVRetentionLayerwiseTransfer(state.kv_checkpoint, 100)
        t.commit_release(3)
        state.kv_offload = NS(ticket=t)
        runner.input_batch.num_computed_tokens_cpu[1] = 18
        with self.assertRaisesRegex(C.M.TTFTLayerwisePrefillError, "layout"):
            runner._validate_ttft_kv_checkpoint(state)

    def test_pressure_check_does_not_offload_when_capacity_is_ample(self):
        s = SCHED()
        s.max_num_scheduled_tokens = 16
        s.virtual_context_scheduled_prefill_fit_headroom_blocks = 2
        a = allocator(ticket())
        a.estimate_num_blocks_to_allocate = lambda req, count: 4
        a.block_pool.get_num_free_blocks = lambda: 6
        s.kv_cache_manager = a
        candidate = NS(virtual_context_deferred_state="", num_prompt_tokens=20, num_computed_tokens=0)
        self.assertFalse(s.layerwise_offload_needed(output(), candidate))
        a.block_pool.get_num_free_blocks = lambda: 5
        self.assertTrue(s.layerwise_offload_needed(output(), candidate))
        a.rows["long"][0].ref_cnt = 2
        self.assertFalse(s.layerwise_offload_needed(output(), candidate))

    def test_real_runner_resumes_offloaded_batch_across_another_yield(self):
        runner, state = C.make_runner_state()
        with patch.dict(os.environ, {C.FLAG: "1"}):
            runner._capture_ttft_kv_checkpoint(state)
        t = M.KVRetentionLayerwiseTransfer(state.kv_checkpoint, 100)
        t.commit_release(3)
        state.kv_offload = NS(ticket=t, completed_layers=2)
        log = []
        state.kv_offload.publish = lambda: t.mark_published(4)
        state.kv_offload.close = lambda: log.append("close")
        result_object = object()
        def generator():
            self.assertIs(M.active_kv_retention_ring(), state.kv_offload)
            state.kv_offload.completed_layers = 3
            yield C.boundary(next_layer=3, model_yield_count=2)
            self.assertIs(M.active_kv_retention_ring(), state.kv_offload)
            state.kv_offload.completed_layers = 4
            return result_object
        state.continuation = generator()
        for name in ("attn_metadata", "num_tokens_across_dp", "cudagraph_mode", "batch_desc",
                     "ubatch_slices_padded", "logits_indices", "spec_decode_metadata",
                     "spec_decode_common_attn_metadata", "ec_connector_output",
                     "cudagraph_stats", "num_scheduled_tokens_np"):
            setattr(state, name, None)
        state.num_tokens_padded = state.num_scheduled_tokens = 7
        state.num_reqs = 2
        runner.vllm_config = None
        runner._pd_green_context_execute_phase = "prefill"
        runner._pd_cuda_phase_trace = lambda *a: contextlib.nullcontext()
        runner._accumulate_ttft_group_debug = lambda *a, **kw: None
        runner._synchronize_ttft_prefill_boundary = lambda: 0.0
        runner._finish_execute_model_forward = lambda **kw: log.append("finish") or result_object
        context = NS(ttft_prefill_continuation=state)
        namespace = {
            "_PD_ACTIVE_MODEL_RUNNER_CONTEXT": NS(get=lambda: context),
            "active_ttft_layerwise_prefill_execution": lambda: state.execution,
            "time": time, "kv_retention_ring_scope": M.kv_retention_ring_scope,
            "set_forward_context": lambda *a, **kw: contextlib.nullcontext(),
            "record_function_or_nullcontext": lambda *a: contextlib.nullcontext(),
        }
        for name in ("reset_operator_selective_pd_metrics", "reset_opmux_metrics",
                     "reset_tiered_context_ring_runtime_metrics"):
            namespace[name] = lambda: None
        with patch.dict(C.RUNNER_GLOBALS, namespace):
            b = runner._resume_ttft_prefill_continuation(state.scheduler_output, state)
            self.assertIs(b.kv_retention_transfer, t)
            self.assertEqual(state.kv_checkpoint.next_layer, 2)  # original source plan, NOT recaptured HBM
            self.assertIs(runner._resume_ttft_prefill_continuation(state.scheduler_output, state), result_object)
        self.assertEqual(log, ["close", "finish"])
        self.assertEqual(t.state, "published")
        self.assertIsNone(context.ttft_prefill_continuation)
        self.assertIsNone(M.active_kv_retention_ring())

    def make_scheduler(self, t):
        s = SCHED()
        s._iteration_debug_index = 100
        s.log = []
        s.requests = {}
        for row in t.checkpoint.rows:
            req = NS(request_id=row.request_id, num_computed_tokens=row.start_token + row.scheduled_tokens,
                     is_finished=lambda: False, virtual_context_deferred_prefill_count=0,
                     virtual_context_deferred_hbm_released_tokens=0,
                     validate_external_residency_suspension=lambda **kw: None)
            req.suspend_to_external_residency = lambda _r=req, **kw: s.log.append(("suspend", _r.request_id, kw))
            s.requests[row.request_id] = req
        s.running = list(s.requests.values())
        s.waiting = NS(prepend_request=lambda req: s.log.append(("wait", req.request_id)))
        s._consume_scheduled_prefill_fit_preparation = lambda ids: s.log.append(("consume", ids))
        return s

    def test_final_recovery_after_ledger_and_skip_cancelled(self):
        t = ticket()
        t.commit_release(3)
        t.mark_published(4)
        s = self.make_scheduler(t)
        del s.requests["short"]
        o = output()
        o.kv_retention_layerwise_transfer = t
        s.complete_layerwise_kv_offload(o)
        self.assertEqual([x[1] for x in s.log if x[0] == "wait"], ["long"])
        self.assertEqual(t.state, "consumed")

    def test_claim_failure_preflights_all_survivors(self):
        t = ticket()
        t.commit_release(3)
        t.mark_published(4)
        s = self.make_scheduler(t)
        def active_claim(**kw):
            raise RuntimeError("active output claim")
        s.requests["short"].validate_external_residency_suspension = active_claim
        o = output()
        o.kv_retention_layerwise_transfer = t
        with self.assertRaisesRegex(RuntimeError, "active output claim"):
            s.complete_layerwise_kv_offload(o)
        self.assertEqual(s.log, [])
        self.assertEqual(len(s.running), 2)

    def test_engine_commits_once_and_wakes_retry_policy(self):
        t, o = ticket(), output()
        b = C.boundary()
        b.kv_retention_transfer = t
        e = ENGINE()
        log = []
        e.scheduler = NS(requests={"long": object(), "short": object()},
                         commit_layerwise_kv_offload=lambda out, transfer: transfer.commit_release(3) or True)
        e._ttft_prefill_execution_key = lambda out: "prefill:1"
        e._scheduler_output_request_ids = lambda out: set(out.num_scheduled_tokens)
        e._ttft_prefill_suspended = {}
        e._iteration_debug_timings = {}
        e._ttft_prefill_boundary_count = 0
        e._pd_capacity_retry_gate = NS(notify_state_change=lambda reason: log.append(reason))
        e._notify_pd_engine_progress = lambda: log.append("engine_wake")
        self.assertTrue(e._suspend_ttft_prefill_boundary(b, o))
        self.assertIs(o.kv_retention_layerwise_transfer, t)
        self.assertEqual(log, ["layerwise_kv_released", "engine_wake"])
        e._ttft_prefill_suspended.clear()
        nxt = C.boundary(next_layer=3, model_yield_count=2)
        t.advance(nxt)
        nxt.kv_retention_transfer = t
        self.assertTrue(e._suspend_ttft_prefill_boundary(nxt, o))
        self.assertEqual(len(log), 2)

    def test_bypass_is_before_cache_write_and_final_after_update(self):
        tree = ast.parse((BASE / "v1/attention/backends/triton_attn.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TritonAttentionImpl")
        f = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        source = ast.unparse(f)
        self.assertLess(source.index("ring_owner.attention("), source.index("kv_cache.unbind("))
        self.assertLess(source.index("ring_owner.attention("), source.index("triton_reshape_and_cache_flash("))
        source = (BASE / "v1/engine/core.py").read_text()
        hook = source.index("self.scheduler.complete_layerwise_kv_offload(scheduler_output)")
        update = source.rindex("self.scheduler.update_from_output(", 0, hook)
        self.assertLess(hook - update, 400)


class FakeTensor:
    def __init__(self, shape, log, *, device="cuda", offset=0):
        self.shape, self.log, self.device = tuple(shape), log, device
        self.ndim, self.dtype, self.is_cuda = len(shape), "bf16", device == "cuda"
        self.offset = offset

    def element_size(self):
        return 2

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        shape = []
        offset = self.offset
        for axis, (index, size) in enumerate(zip(keys, self.shape)):
            stride = 1
            for dim in self.shape[axis + 1:]:
                stride *= dim
            if isinstance(index, slice):
                shape.append(len(range(*index.indices(size))))
                offset += index.indices(size)[0] * stride
            else:
                offset += index * stride
        shape.extend(self.shape[len(keys):])
        return FakeTensor(shape, self.log, device=self.device, offset=offset)

    def view(self, *shape):
        return FakeTensor(shape, self.log, device=self.device)

    def copy_(self, other, **kw):
        self.log.append(("copy", self.shape, other.shape))
        return self


class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log
        self.fail = False

    def wait_event(self, event):
        self.log.append((self.name, "wait", event))

    def synchronize(self):
        self.log.append((self.name, "sync"))
        if self.fail:
            raise RuntimeError("poisoned CUDA")


def gpu_adapter():
    log = []
    compute, copy, ring = [FakeStream(name, log) for name in ("compute", "copy", "ring")]
    class Event:
        def record(self, stream):
            log.append(("record", stream.name, self))
        def synchronize(self):
            log.append(("event_sync", self))
    class Executor:
        def __init__(self, **kw):
            self._copy_streams = {"cuda": ring}
        def _get_copy_stream(self, device):
            return ring, True
        def run(self, query, blocks, **kw):
            log.append(("attention", query.shape, blocks, kw))
    def direct_d2h(keys, values, dst_key, dst_value):
        # Mirror cache_kernels.cu::tiered_d2h_multicopy geometry checks.
        # A logging-only fake hid the original ragged-source CUDA failure.
        assert len(keys) == len(values)
        assert dst_key.device == dst_value.device == "cpu"
        assert dst_key.ndim == dst_value.ndim == 3
        assert dst_key.shape == dst_value.shape
        if not keys:
            return 0
        first = keys[0]
        expected = (len(keys) * first.shape[0], *first.shape[1:])
        if dst_key.shape != expected:
            raise RuntimeError("tiered D2H multicopy destination shape must equal packed source pages")
        for key, value in zip(keys, values):
            assert key.is_cuda and value.is_cuda
            assert key.device == value.device == first.device
            assert key.dtype == value.dtype == dst_key.dtype == dst_value.dtype
            if key.shape != first.shape or value.shape != first.shape:
                raise RuntimeError("tiered D2H multicopy source page shapes must match")
        assert dst_key.offset == dst_value.offset
        log.append(("snapshot", dst_key.shape, [k.shape for k in keys],
                    dst_key.offset, [k.offset for k in keys], [v.offset for v in values]))
        return len(keys)
    fake_torch = NS(float16="fp16", bfloat16="bf16",
                    inference_mode=lambda value: contextlib.nullcontext(),
                    empty=lambda shape, **kw: FakeTensor(shape, log, device="cpu"),
                    empty_like=lambda t, **kw: FakeTensor(t.shape, log, device="cpu"),
                    cuda=NS(Stream=lambda **kw: copy, Event=Event,
                            current_stream=lambda device=None: compute,
                            stream=lambda stream: contextlib.nullcontext()))
    lease = NS(release=lambda: log.append(("lease_release",)))
    namespace = dict(torch=fake_torch, time=time,
                     KVRetentionLayerwiseTransfer=M.KVRetentionLayerwiseTransfer,
                     reserve_kv_retention_host_bytes=M.reserve_kv_retention_host_bytes,
                     TTFTLayerwisePrefillError=C.M.TTFTLayerwisePrefillError,
                     TieredBlockwiseAttentionExecutor=Executor,
                     HBMStagingBuffer=lambda **kw: NS(**kw),
                     context_ring_compute_group_tokens=lambda: 8,
                     context_ring_batched_partial_enabled=lambda: False,
                     context_ring_streaming_wave_groups=lambda: 1,
                     acquire_tiered_hbm_source_block_lease=lambda ids: lease,
                     wait_for_pending_hbm_d2h_request_reads=lambda ids: log.append(("prior_fence",)),
                     ops=NS(tiered_d2h_multicopy=direct_d2h),
                     TieredKVBlock=lambda key, value, **kw: NS(key=key, value=value, **kw),
                     _QUARANTINED_OWNERS=[])
    klass = methods(BASE / "v1/attention/ops/kv_retention_layerwise.py", "KVRetentionRingContinuation",
                    {"capture", "attention", "publish", "close"}, namespace)
    impl = type("TritonAttentionImpl", (), {"scale": 0.5})()
    layers = {name: NS(impl=impl, kv_cache=[FakeTensor((10, 2, 16, 2, 4), log)]) for name in C.NAMES}
    return klass, namespace, layers, log, (compute, copy, ring)


class CopyPipelineTests(unittest.TestCase):
    def capture(self):
        klass, ns, layers, log, streams = gpu_adapter()
        owner = klass.capture(plan(), layers)
        self.addCleanup(owner.close)
        return owner, ns, layers, log, streams

    def test_actual_snapshot_copies_only_valid_layer_ranges(self):
        owner, ns, layers, log, streams = self.capture()
        sizes = [item[1][0] for item in log if item[0] == "snapshot"]
        self.assertEqual(sizes, [16, 8, 2, 16, 8, 2, 16, 3, 16, 3])
        self.assertEqual(owner.ticket.copied_bytes, 90 * 2 * 2 * 4 * 2)
        self.assertLess(next(i for i, e in enumerate(log) if e[0] == "event_sync"),
                        next(i for i, e in enumerate(log) if e[0] == "lease_release"))
        self.assertEqual(owner.ticket.state, "copy_ready")

    def test_direct_copy_fake_rejects_ragged_pages_even_with_padding(self):
        klass, ns, layers, log, streams = gpu_adapter()
        keys = [FakeTensor((n, 2, 4), log) for n in (16, 8)]
        values = [FakeTensor(k.shape, log) for k in keys]
        for length, message in ((24, "packed source pages"), (32, "page shapes")):
            with self.subTest(length=length):
                dst = FakeTensor((length, 2, 4), log, device="cpu")
                with self.assertRaisesRegex(RuntimeError, message):
                    ns["ops"].tiered_d2h_multicopy(keys, values, dst, dst)
        self.assertEqual(log, [])

    def test_snapshot_partition_geometry_and_offsets(self):
        for start, chunk in ((0, 3), (3, 5), (16, 16), (19, 5), (32, 16), (37, 16)):
            with self.subTest(start=start, chunk=chunk):
                klass, ns, layers, log, streams = gpu_adapter()
                blocks = (7, 2, 8, 1)[:(start + chunk + 15) // 16]
                row = C.M.KVRetentionCheckpointRow("a", 0, start, chunk, 16, blocks)
                checkpoint = C.M.build_kv_retention_layerwise_checkpoint(
                    boundary=C.boundary(request_ids=("a",), scheduled_tokens=(("a", chunk),)),
                    first_layer=0, layer_names=C.NAMES, rows=(row,), gpu_boundary_complete=True,
                )
                owner = klass.capture(checkpoint, layers)
                try:
                    expected = []
                    for layer in range(4):
                        valid = start + (chunk if layer < 2 else 0)
                        full, tail = divmod(valid, 16)
                        if full:
                            expected.append(("snapshot", (full * 16, 2, 4),
                                             [(16, 2, 4)] * full, 0,
                                             [b * 256 for b in blocks[:full]],
                                             [b * 256 + 128 for b in blocks[:full]]))
                        if tail:
                            expected.append(("snapshot", (tail, 2, 4), [(tail, 2, 4)],
                                             full * 16 * 8, [blocks[full] * 256],
                                             [blocks[full] * 256 + 128]))
                    self.assertEqual([e for e in log if e[0] == "snapshot"], expected)
                    self.assertEqual(owner.ticket.copied_bytes, (4 * start + 2 * chunk) * 32)
                    self.assertEqual(owner.ticket.state, "copy_ready")
                finally:
                    owner.close()

    def test_tail_copy_failure_drains_full_pages_before_releasing_storage(self):
        for poisoned in (False, True):
            with self.subTest(poisoned=poisoned):
                klass, ns, layers, log, streams = gpu_adapter()
                direct = ns["ops"].tiered_d2h_multicopy
                calls = 0
                def fail_tail(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        streams[1].fail = poisoned
                        raise RuntimeError("tail copy failed after full-page enqueue")
                    return direct(*args)
                ns["ops"].tiered_d2h_multicopy = fail_tail
                host_before = M._HOST_BYTES
                with self.assertRaisesRegex(RuntimeError, "poisoned CUDA" if poisoned else "tail copy failed"):
                    klass.capture(plan(), layers)
                self.assertEqual(calls, 2)
                self.assertEqual(len([e for e in log if e[0] == "snapshot"]), 1)
                self.assertFalse(any(e[0] == "event_sync" for e in log))
                if poisoned:
                    self.assertNotIn(("lease_release",), log)
                    owner, lease = ns["_QUARANTINED_OWNERS"][-1]
                    self.assertTrue(owner.buffers)
                    self.assertFalse(owner.reservation.released)
                    streams[1].fail = False  # release fake test resources only
                    owner.close()
                    lease.release()
                else:
                    self.assertLess(log.index(("copy", "sync")), log.index(("lease_release",)))
                    self.assertLess(log.index(("ring", "sync")), log.index(("lease_release",)))
                self.assertEqual(M._HOST_BYTES, host_before)

    def test_both_copy_streams_wait_producer_and_compute_waits_d2h(self):
        owner, ns, layers, log, streams = self.capture()
        owner.ticket.commit_release(3)
        log.clear()
        tensors = [FakeTensor((7, 2, 4), log) for _ in range(4)]
        owner.attention(C.NAMES[2], layers[C.NAMES[2]].impl, *tensors,
                        NS(num_actual_tokens=7, request_ids=["short", "long"]))
        waits = [(e[0], e[2]) for e in log if len(e) == 3 and e[1] == "wait"]
        self.assertEqual([w[0] for w in waits], ["copy", "ring", "compute"])
        self.assertIs(waits[0][1], waits[1][1])
        self.assertFalse(any(e[0] == "event_sync" or e[-1] == "sync" for e in log))
        calls = [e for e in log if e[0] == "attention"]
        self.assertEqual([c[3]["query_start"] for c in calls], [0, 19])
        self.assertEqual([b.tier for b in calls[1][2]], ["DRAM", "HBM"])
        self.assertEqual(owner.completed_layers, 3)

    def test_wrong_layer_and_rows_fail_before_copy(self):
        owner, ns, layers, log, streams = self.capture()
        owner.ticket.commit_release(3)
        tensors = [FakeTensor((7, 2, 4), log) for _ in range(4)]
        log.clear()
        with self.assertRaisesRegex(C.M.TTFTLayerwisePrefillError, "order"):
            owner.attention(C.NAMES[3], layers[C.NAMES[3]].impl, *tensors,
                            NS(num_actual_tokens=7, request_ids=["short", "long"]))
        self.assertEqual(log, [])

    def test_publication_only_after_uniform_coverage_and_checks_provider(self):
        owner, ns, layers, log, streams = self.capture()
        owner.ticket.commit_release(3)
        with self.assertRaisesRegex(C.M.TTFTLayerwisePrefillError, "incomplete"):
            owner.publish()
        owner.completed_layers = 4
        table = V.VirtualContextTable()
        ns.update(get_virtual_context_table=lambda: table,
                  virtual_context_layer_id=lambda name: C.NAMES.index(name),
                  register_dram_kv_page=lambda **kw: log.append(("register", kw)),
                  lookup_dram_kv_page=lambda **kw: object())
        owner.publish()
        self.assertEqual(owner.ticket.state, "published")
        self.assertEqual(len([e for e in log if e[0] == "register"]), 12)
        self.assertTrue(table.has_hbm_reload_required_pages(["long", "short"]))

    def test_poisoned_cleanup_retains_storage_and_host_budget(self):
        owner, ns, layers, log, streams = self.capture()
        streams[1].fail = True
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            owner.close()
        self.assertIn(owner, ns["_QUARANTINED_OWNERS"])
        self.assertTrue(owner.buffers)
        self.assertFalse(owner.reservation.released)
        streams[1].fail = False  # fake environment cleanup, not a runtime repair

    def test_budget_decline_makes_no_copy_or_lease(self):
        klass, ns, layers, log, streams = gpu_adapter()
        with patch.dict(os.environ, {"VLLM_KV_RETENTION_LAYERWISE_HOST_BYTES": "1"}):
            self.assertIsNone(klass.capture(plan(), layers))
        self.assertEqual(log, [])

    def test_capture_failure_cannot_release_lease_if_cleanup_fails(self):
        klass, ns, layers, log, streams = gpu_adapter()
        def fail_copy(*args):
            streams[1].fail = True
            raise RuntimeError("partially enqueued DMA")
        ns["ops"].tiered_d2h_multicopy = fail_copy
        with self.assertRaises(RuntimeError):
            klass.capture(plan(), layers)
        self.assertNotIn(("lease_release",), log)
        owner, lease = ns["_QUARANTINED_OWNERS"][-1]
        self.assertTrue(owner.buffers)
        self.assertFalse(owner.reservation.released)
        streams[1].fail = False
        owner.close()  # release fake test reservation, not production recovery
        lease.release()

    def test_failed_provider_write_never_marks_ticket_published(self):
        owner, ns, layers, log, streams = self.capture()
        owner.ticket.commit_release(3)
        owner.completed_layers = 4
        table = V.VirtualContextTable()
        ns.update(get_virtual_context_table=lambda: table,
                  virtual_context_layer_id=lambda name: C.NAMES.index(name),
                  register_dram_kv_page=lambda **kw: None,
                  lookup_dram_kv_page=lambda **kw: None)
        with self.assertRaisesRegex(C.M.TTFTLayerwisePrefillError, "rejected"):
            owner.publish()
        self.assertEqual(owner.ticket.state, "released")
        self.assertFalse(table._pages)

    def test_batched_all_group_mode_rejected_before_copy(self):
        klass, ns, layers, log, streams = gpu_adapter()
        ns["context_ring_batched_partial_enabled"] = lambda: True
        with self.assertRaisesRegex(C.M.TTFTLayerwisePrefillError, "streaming"):
            klass.capture(plan(), layers)
        self.assertEqual(log, [])


if __name__ == "__main__":
    unittest.main()
