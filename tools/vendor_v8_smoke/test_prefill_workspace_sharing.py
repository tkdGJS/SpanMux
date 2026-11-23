#!/usr/bin/env python3
"""CPU execution of real workspace/arena methods with tensor/event doubles."""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import threading
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from test_isolation_memory import V, load_symbols


class Device:
    def __init__(self, name):
        self.name = str(name)
        self.type = self.name.split(":")[0]

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == str(other)


class Tensor:
    def __init__(self, data):
        self.data = data
        self.device = Device("cpu")
        self.dtype = data.dtype
        self.shape = data.shape
        self.streams = []

    def narrow(self, axis, offset, count):
        index = [slice(None)] * self.data.ndim
        index[axis] = slice(offset, offset + count)
        return Tensor(self.data[tuple(index)])

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], np.dtype):
            return Tensor(self.data.view(shape[0]))
        return Tensor(self.data.reshape(shape))

    def numel(self):
        return self.data.size

    def element_size(self):
        return self.data.itemsize

    def record_stream(self, stream):
        self.streams.append(stream)

    def __getitem__(self, key):
        return Tensor(self.data[key])


class Event:
    def __init__(self, done=False, fail=False):
        self.done = done
        self.fail = fail

    def query(self):
        return self.done

    def record(self, stream):
        if self.fail:
            raise RuntimeError("injected event failure")
        self.stream = stream


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.stream = NS(waits=[])
        self.stream.wait_event = self.stream.waits.append
        self.allocations = []
        self.fail_allocate = False

        def empty(shape, dtype, device):
            if self.fail_allocate:
                raise RuntimeError("injected allocation failure")
            t = Tensor(np.empty(shape, dtype=dtype))
            self.allocations.append(t)
            return t

        self.torch = NS(empty=empty, float32=np.dtype("float32"), device=Device,
                        cuda=NS(is_available=lambda: False, Event=Event,
                                current_stream=lambda device=None: self.stream))
        names = {"PrefillWorkspaceError", "LlamaPrefillWorkspaceLayout",
                 "PrefillWorkspaceExecution", "_PrefillWorkspaceSlot",
                 "PrefillRingStateBorrow", "PrefillWorkspaceManager", "_env_flag",
                 "_workspace_opt_in", "prefill_workspace_attention",
                 "active_prefill_workspace_ring_state"}
        self.ns = load_symbols(V / "model_executor/layers/pd_prefill_workspace.py", names, {
            "torch": self.torch, "dataclass": dataclasses.dataclass,
            "field": dataclasses.field, "threading": threading, "os": os,
            "contextmanager": contextmanager, "__name__": __name__,
            "_ACTIVE_PREFILL_WORKSPACE": ContextVar("test_workspace", default=None),
        })
        self.manager = self.ns["PrefillWorkspaceManager"](
            enabled=True, strict=True, max_tokens=32, slot_count=1,
            rightsize=True, sharing=True)
        self.ns["PD_PREFILL_WORKSPACE"] = self.manager
        self.layout = self.ns["LlamaPrefillWorkspaceLayout"](6, 4, 16, 8, 4)
        arena = load_symbols(V / "v1/attention/ops/isolation_ring_state.py",
                             {"IsolationRingStateArena"})["IsolationRingStateArena"]
        self.arena_type = arena

    def acquire(self, tokens=7, slot=0):
        return self.manager.acquire(profile_id="d30_p10", profile_epoch=1,
                                    num_tokens=tokens, phase_slot=slot, stream=self.stream)

    def configure(self, slot=0):
        self.manager.configure_llama(layout=self.layout, dtype=np.dtype("float16"),
                                     device="cpu", phase_slot=slot)

    def buffer(self, execution, name):
        return self.manager.buffer(execution, name, num_tokens=execution.num_tokens,
                                   width=self.layout.widths[name],
                                   dtype=np.dtype("float16"), device="cpu")

    def finish(self, execution, done=False):
        event = Event(done=done)
        self.manager.finish(execution, completion_event=event)
        return event

    def query(self, tokens=7, heads=1, dim=4):
        return Tensor(np.zeros((tokens, heads, dim), dtype=np.float16))

    def test_rightsize_growth_event_retirement_and_no_shrink(self):
        x = self.acquire(7)
        self.configure()
        slot = self.manager._slots[0]
        self.assertEqual(slot.capacity_tokens, 8)
        original = slot.storage
        self.buffer(x, "qkv")
        event = self.finish(x)
        y = self.acquire(17)  # Grows even on resume path before model configure.
        self.assertEqual(slot.capacity_tokens, 32)
        self.assertIs(slot.retired[0][0], original)
        self.assertIn(event, self.stream.waits)
        self.assertGreater(self.manager.debug_snapshot()["pd_prefill_workspace_retired_bytes"], 0)
        grown = slot.storage
        self.assertIsNot(grown, original)
        self.finish(y, True)
        event.done = True
        z = self.acquire(2)
        self.assertIs(slot.storage, grown)
        self.assertEqual(slot.retired, [])
        self.assertEqual(self.buffer(z, "qkv").shape, (2, 6))
        self.assertEqual(self.manager._growth_count, 1)

    def test_legacy_still_allocates_maximum(self):
        self.manager._rightsize = False
        x = self.acquire(1)
        self.configure()
        self.assertEqual(self.manager._slots[0].capacity_tokens, 32)
        self.assertEqual(self.buffer(x, "qkv").shape, (1, 6))

    def test_failed_growth_preserves_old_owner(self):
        x = self.acquire()
        self.configure()
        old = self.manager._slots[0].storage
        event = self.finish(x)
        self.fail_allocate = True
        with self.assertRaisesRegex(RuntimeError, "allocation failure"):
            self.acquire(17)
        slot = self.manager._slots[0]
        self.assertIs(slot.storage, old)
        self.assertIs(slot.completion_event, event)
        self.assertEqual(slot.state, "pending")
        self.assertEqual(slot.capacity_tokens, 8)
        self.fail_allocate = False
        self.acquire(17)

    def test_active_growth_and_stale_generation_rejected(self):
        x = self.acquire()
        self.configure()
        self.buffer(x, "qkv")
        with self.assertRaisesRegex(RuntimeError, "active Prefill"):
            self.manager.configure_llama(layout=self.layout, dtype=np.dtype("float16"),
                                         device="cpu", num_tokens=17)
        self.finish(x, True)
        self.acquire()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.buffer(x, "qkv")

    def test_disabled_independent_flags_do_not_inherit_bundle(self):
        with patch.dict(os.environ, {"VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1"}, clear=True):
            m = self.ns["PrefillWorkspaceManager"]()
            self.assertFalse(m._rightsize)
            self.assertFalse(m.sharing)
        with patch.dict(os.environ, {"VLLM_ISOLATION_RING_WORKSPACE_SHARING": "yes"}):
            with self.assertRaisesRegex(RuntimeError, "0 or 1"):
                self.ns["PrefillWorkspaceManager"]()

    def test_shared_state_preserves_qkv_output_and_does_not_allocate(self):
        x = self.acquire()
        self.configure()
        qkv, out = self.buffer(x, "qkv"), self.buffer(x, "attention")
        qkv.data.fill(11)
        out.data.fill(22)
        self.manager.begin_attention(x, self.stream)
        arena = self.arena_type(self.torch, lambda *args: args,
                               lambda q: self.manager.borrow_ring_state(x, q, self.stream))
        state, reused = arena.acquire(self.query())
        self.assertTrue(reused)
        self.assertEqual(len(self.allocations), 1)
        self.assertEqual(arena.bytes, 0)
        self.assertEqual(arena.shared_bytes, 7 * 6 * 4)
        for tensor in state:
            tensor.data.fill(123)
        np.testing.assert_equal(qkv.data, 11)
        np.testing.assert_equal(out.data, 22)
        with self.assertRaisesRegex(RuntimeError, "live Attention"):
            self.finish(x)
        with self.assertRaisesRegex(RuntimeError, "during Ring borrow"):
            self.buffer(x, "gate_up")
        borrow = arena.borrow
        arena.release()
        self.assertIsNone(arena.borrow)
        self.assertIsNone(arena.storage)
        with self.assertRaisesRegex(RuntimeError, "stale Ring"):
            borrow.release()
        self.manager.end_attention(x, self.stream)
        self.buffer(x, "gate_up").data.fill(44)
        self.finish(x)

    def test_scope_and_size_fallback_keep_private_arena(self):
        x = self.acquire()
        self.configure()
        arena = self.arena_type(self.torch, lambda *args: args,
                               lambda q: self.manager.borrow_ring_state(x, q, self.stream))
        _, reused = arena.acquire(self.query())
        self.assertFalse(reused)
        self.assertGreater(arena.bytes, 0)
        arena.release()
        self.manager.begin_attention(x, self.stream)
        self.assertIsNone(self.manager.borrow_ring_state(x, self.query(heads=32), self.stream))
        arena.acquire(self.query())
        self.assertEqual(arena.bytes, 0)  # No permanent duplicate private state.
        arena.release()
        self.manager.end_attention(x, self.stream)

    def test_failed_event_keeps_borrow_and_blocks_finish(self):
        x = self.acquire()
        self.configure()
        self.manager.begin_attention(x, self.stream)
        arena = self.arena_type(self.torch, lambda *args: args,
                               lambda q: self.manager.borrow_ring_state(x, q, self.stream))
        arena.acquire(self.query())
        self.torch.cuda.Event = lambda: Event(fail=True)
        with self.assertRaisesRegex(RuntimeError, "event failure"):
            arena.release()
        self.assertTrue(arena.active)
        with self.assertRaises(RuntimeError):
            self.manager.end_attention(x, self.stream)
        with self.assertRaises(RuntimeError):
            self.finish(x)

    def test_changed_stream_and_nested_borrow_fail_closed(self):
        x = self.acquire()
        self.configure()
        self.manager.begin_attention(x, self.stream)
        other = NS(waits=[])
        with self.assertRaisesRegex(RuntimeError, "changed Attention stream"):
            self.manager.borrow_ring_state(x, self.query(), other)
        borrow = self.manager.borrow_ring_state(x, self.query(), self.stream)
        with self.assertRaisesRegex(RuntimeError, "nested Ring"):
            self.manager.borrow_ring_state(x, self.query(), self.stream)
        self.manager.release_ring_state(borrow)
        self.manager.end_attention(x, self.stream)

    def test_exact_attention_scope_releases_before_projection(self):
        x = self.acquire()
        self.configure()
        token = self.ns["_ACTIVE_PREFILL_WORKSPACE"].set(x)
        try:
            with self.ns["prefill_workspace_attention"]():
                borrow = self.manager.borrow_ring_state(x, self.query(), self.stream)
                borrow.release()
            self.assertIsNone(self.manager._slots[0].attention_stream)
            self.buffer(x, "norm")
        finally:
            self.ns["_ACTIVE_PREFILL_WORKSPACE"].reset(token)

    def test_distinct_phase_slots_never_share(self):
        with patch.dict(os.environ, {"VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "2"}):
            self.manager._slot_count_override = 2
            x = self.acquire(slot=0)
            self.configure(0)
            y = self.acquire(slot=1)
            self.configure(1)
            self.assertFalse(np.shares_memory(self.buffer(x, "qkv").data,
                                             self.buffer(y, "qkv").data))

    def test_lightweight_trace_keeps_bounded_memory_evidence(self):
        spec = importlib.util.spec_from_file_location(
            "workspace_observability", V / "v1/core/sched/observability.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.acquire()
        self.configure()
        trace = module.compact_worker_debug_timings(self.manager.debug_snapshot())
        self.assertEqual(trace["pd_prefill_workspace_capacity_tokens"], 8)
        self.assertIn("pd_prefill_workspace_ring_borrow_count", trace)
        self.assertNotIn("pd_prefill_workspace_slot_states", trace)

    def test_graph_capture_cannot_borrow_eager_arena(self):
        x = self.acquire()
        self.configure()
        q = self.query()
        q.device = Device("cuda:0")
        self.torch.cuda.is_current_stream_capturing = lambda: True
        token = self.ns["_ACTIVE_PREFILL_WORKSPACE"].set(x)
        try:
            self.assertIsNone(self.ns["active_prefill_workspace_ring_state"](q))
            self.assertEqual(self.manager._ring_borrow_count, 0)
            self.assertEqual(self.manager._ring_borrow_fallback_count, 0)
        finally:
            self.ns["_ACTIVE_PREFILL_WORKSPACE"].reset(token)

    def test_borrow_error_poison_cannot_retry_private_fallback(self):
        def bad_borrow(query):
            raise RuntimeError("owner mismatch")
        arena = self.arena_type(self.torch, lambda *args: args, bad_borrow)
        with self.assertRaisesRegex(RuntimeError, "owner mismatch"):
            arena.acquire(self.query())
        arena.borrow_factory = None
        with self.assertRaisesRegex(RuntimeError, "owner failed"):
            arena.acquire(self.query())

    def test_profile_handoff_keeps_pointer_but_is_forbidden_inside_attention(self):
        x = self.acquire()
        self.configure()
        pointer = self.manager._slots[0].storage
        args = dict(handoff_generation=0, source_profile_id="d30_p10",
                    source_profile_epoch=1, target_profile_id="d0_p40", target_profile_epoch=2)
        self.manager.begin_attention(x, self.stream)
        with self.assertRaisesRegex(RuntimeError, "handoff inside Attention"):
            self.manager.handoff(x, **args)
        self.manager.end_attention(x, self.stream)
        self.manager.handoff(x, **args)
        self.assertIs(pointer, self.manager._slots[0].storage)
        self.assertEqual(x.profile_id, "d0_p40")


if __name__ == "__main__":
    unittest.main()
