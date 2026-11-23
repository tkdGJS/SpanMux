#!/usr/bin/env python3
"""Execute real GreenContext lifetime methods with CPU driver/stream doubles."""
from __future__ import annotations

import ast
import ctypes
import gc
import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_isolation_memory import V, load_symbols


class Bridge:
    path = Path("fake-green-bridge")
    supports_sm_count = True

    def __init__(self):
        self.next_id = 1
        self.live = {}
        self.destroyed = []
        self.fail_ptr = False
        self.mismatch = False

    def create(self, **kwargs):
        handle = ctypes.c_void_p(self.next_id)
        self.next_id += 1
        self.live[handle.value] = kwargs
        return handle

    def destroy(self, handle):
        if handle and handle.value:
            self.destroyed.append(handle.value)
            del self.live[handle.value]

    def stream_ptr(self, handle, phase):
        if self.fail_ptr:
            raise RuntimeError("pointer lookup failure before export")
        return 10 * handle.value + phase

    def sm_count(self, handle, phase):
        if self.mismatch:
            return 999
        return self.live[handle.value]["decode_sms" if phase == 0 else "prefill_sms"]


class GreenLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME": "1",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.bridge = Bridge()
        self.export_failure = False
        self.export_observations = []
        self.log = []

        def external(ptr, device):
            self.export_observations.append(len(self.ns["_PROCESS_OWNED_GREEN_HANDLES"]))
            if self.export_failure:
                raise RuntimeError("partially failed ExternalStream export")
            return NS(cuda_stream=ptr, device=device)

        tree = ast.parse((V / "model_executor/layers/green_context.py").read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        constants[name.id] = node.value.value
        profile = NS(profile_id="d30_p10", decode_sms=30, prefill_sms=10)
        names = {"GreenContextError", "_env_bool", "_env_int", "_GreenContextStreams",
                 "_GreenContextProfileBank", "retire_process_owned_green_contexts",
                 "_get_static_streams", "_get_profile_bank", "_dynamic_sm_strict"}
        self.ns = load_symbols(V / "model_executor/layers/green_context.py", names, {
            **constants, "os": os, "ctypes": ctypes, "threading": threading,
            "torch": NS(cuda=NS(is_available=lambda: True, current_device=lambda: 0,
                get_device_properties=lambda d: NS(multi_processor_count=40),
                ExternalStream=external)),
            "logger": NS(info=lambda *args: self.log.append(args), warning_once=lambda *a, **k: None),
            "_BridgeLibrary": lambda p: self.bridge, "_library_path": lambda: Path("fake"),
            "parse_pd_sm_profiles": lambda *a, **k: [profile],
            "profile_by_id": lambda ps, key: next((p for p in ps if p.profile_id == key), None),
            "_INITIALIZATION_LOCK": threading.RLock(),
            "_PROCESS_OWNED_GREEN_HANDLES": [], "_PROCESS_OWNED_GREEN_RETIRED": False,
            "_STREAMS": None, "_PROFILE_BANK": None,
            "_INITIALIZATION_FAILED": False, "_PROFILE_BANK_INITIALIZATION_FAILED": False,
        })
        self.error = self.ns["GreenContextError"]

    def streams(self):
        return self.ns["_GreenContextStreams"](library=self.bridge)

    def test_adopt_before_first_export_only_once(self):
        s = self.streams()
        self.assertEqual(self.export_observations, [1, 1])
        self.assertEqual(len(self.ns["_PROCESS_OWNED_GREEN_HANDLES"]), 1)
        self.assertFalse(s._closed)

    def test_close_and_gc_preserve_late_consumer_context(self):
        s = self.streams()
        handle_id = s.handle.value
        late_stream = s.decode_stream
        s.close()
        s.close()
        del s
        gc.collect()
        self.assertEqual(late_stream.cuda_stream, 10 * handle_id)
        self.assertIn(handle_id, self.bridge.live)
        self.assertEqual(self.bridge.destroyed, [])

    def test_pre_export_failure_cleans_handle(self):
        self.bridge.fail_ptr = True
        with self.assertRaisesRegex(RuntimeError, "before export"):
            self.streams()
        gc.collect()
        self.assertEqual(self.bridge.destroyed, [1])
        self.assertEqual(self.ns["_PROCESS_OWNED_GREEN_HANDLES"], [])

    def test_partial_export_failure_retains_handle(self):
        self.export_failure = True
        with self.assertRaisesRegex(RuntimeError, "partially failed"):
            self.streams()
        gc.collect()
        self.assertEqual(self.bridge.destroyed, [])
        self.assertEqual(len(self.ns["_PROCESS_OWNED_GREEN_HANDLES"]), 1)

    def test_sm_verification_failure_after_export_retains_handle(self):
        self.bridge.mismatch = True
        with self.assertRaisesRegex(self.error, "verification mismatch"):
            self.streams()
        self.assertEqual(self.bridge.destroyed, [])

    def test_legacy_flag_is_unchanged(self):
        with patch.dict(os.environ, {"VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME": "0"}):
            s = self.streams()
        s.close()
        s.close()
        self.assertEqual(self.bridge.destroyed, [1])
        self.assertEqual(self.ns["_PROCESS_OWNED_GREEN_HANDLES"], [])

    def test_flag_is_frozen_at_creation(self):
        s = self.streams()
        with patch.dict(os.environ, {"VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME": "0"}):
            s.close()
        self.assertEqual(self.bridge.destroyed, [])

    def test_retired_stream_cannot_run_or_export_again(self):
        s = self.streams()
        s.close()
        for call in (lambda: s.stream_for_phase(0),
                     lambda: s.run_on_phase_stream(0, lambda: None),
                     lambda: s.run_decode_prefill(lambda: None, lambda: None,
                                                overlap=True, collect_timing=False)):
            with self.assertRaisesRegex(self.error, "retired"):
                call()

    def test_closed_bank_cannot_recreate_get_or_activate(self):
        b = self.ns["_GreenContextProfileBank"]()
        self.assertEqual(b.created_count, 1)
        b.close()
        b.close()
        for call in (lambda: b.get("d30_p10"), lambda: b.activate("d30_p10", 2)):
            with self.assertRaisesRegex(self.error, "retired"):
                call()
        self.assertEqual(len(self.bridge.live), 1)

    def test_global_retirement_blocks_restart_and_is_idempotent(self):
        self.ns["_PROFILE_BANK"] = self.ns["_GreenContextProfileBank"]()
        self.ns["_STREAMS"] = self.streams()
        self.ns["retire_process_owned_green_contexts"]()
        self.ns["retire_process_owned_green_contexts"]()
        for name in ("_get_profile_bank", "_get_static_streams"):
            with self.assertRaisesRegex(self.error, "retired"):
                self.ns[name]()
        self.assertEqual(len(self.bridge.live), 2)
        self.assertEqual(self.bridge.destroyed, [])
        self.assertEqual(sum("serving retired" in x[0] for x in self.log), 1)

    def test_native_noop_does_not_initialize_cuda(self):
        self.ns["retire_process_owned_green_contexts"]()
        self.assertEqual(self.bridge.live, {})
        self.assertFalse(self.ns["_PROCESS_OWNED_GREEN_RETIRED"])

    def test_worker_retires_after_existing_cleanup(self):
        path = V / "v1/worker/gpu_worker.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
        events = []
        ns = {"retire_process_owned_green_contexts": lambda: events.append("retire")}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        worker = NS(model_runner=NS(
            finalize_pd_decode_gpu_component_trace=lambda: events.append("trace"),
            ensure_kv_transfer_shutdown=lambda: events.append("kv")),
            profiler=NS(shutdown=lambda: events.append("profiler")))
        ns["shutdown"](worker)
        self.assertEqual(events, ["trace", "kv", "profiler", "retire"])


if __name__ == "__main__":
    unittest.main()
