# SPDX-License-Identifier: Apache-2.0
"""Opt-in, startup-only allocator cleanup, mirrored in the Native source tree.

No torch import, tensor ownership, worker thread, or serving-path callback.
Emptying an allocator cache is NOT a way to free live KV/workspace/graph tensors.
"""

import ast
import gc
import hashlib
import json
import os
import time
from pathlib import Path

STARTUP_MEMORY_CONTRACT_VERSION = 1


def _enabled(name):
    value = os.getenv("VLLM_ISOLATION_" + name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"VLLM_ISOLATION_{name} must be 0 or 1, got {value!r}")
    return value == "1"


class StartupMemoryAudit:
    """A one-shot warmup -> capture -> sampler boundary, not a memory policy."""

    def __init__(self, *, cuda, device, log, first_startup=True):
        self.trim_requested = _enabled("STARTUP_CACHE_TRIM")
        self.first_startup = first_startup
        self.trim = self.trim_requested and first_startup
        self.audit = _enabled("STARTUP_MEMORY_AUDIT") or self.trim_requested
        self.cuda, self.device, self.log = cuda, device, log
        self._state = "compile_warmup"
        self._started = time.monotonic()
        self._record("before_compile_warmup")

    def _record(self, stage, **extra):
        if not self.audit:
            return {}
        stats = self.cuda.memory_stats(self.device)
        record = {
            "contract_version": STARTUP_MEMORY_CONTRACT_VERSION,
            "stage": stage,
            "device": str(self.device),
            "trim_enabled": self.trim,
            "first_worker_warmup": self.first_startup,
            "trim_skip_reason": (
                "worker_warmup_reentry"
                if self.trim_requested and not self.first_startup else None
            ),
            "elapsed_ms": (time.monotonic() - self._started) * 1000,
        }
        for label, key in (
            ("allocated_bytes", "allocated_bytes.all.current"),
            ("active_bytes", "active_bytes.all.current"),
            ("reserved_bytes", "reserved_bytes.all.current"),
            ("inactive_split_bytes", "inactive_split_bytes.all.current"),
            ("peak_allocated_bytes", "allocated_bytes.all.peak"),
            ("peak_reserved_bytes", "reserved_bytes.all.peak"),
        ):
            record[label] = int(stats.get(key, -1))
        # These counters are not a proof that reserved-minus-allocated is free.
        # Do not reset peaks or synchronize for an audit-only observation.
        record.update(extra)
        self.log("ISOLATION_STARTUP_MEMORY %s", json.dumps(record, sort_keys=True))
        return record

    def before_capture(self):
        if self._state != "compile_warmup":
            raise RuntimeError("startup cache trim is one-shot and pre-capture only")
        # Consume the permission even on failure; never retry a failed barrier.
        self._state = "pre_capture"
        self._record("after_compile_warmup")
        if not self.trim:
            return
        try:
            with self.cuda.device(self.device):
                if self.cuda.is_current_stream_capturing():
                    raise RuntimeError("startup cache trim cannot run during capture")
                before = self._record("before_cache_trim")
                started = time.monotonic()
                gc.collect()
                self.cuda.synchronize(self.device)
                # No live tensor, pending workspace owner, KV page, or context
                # is explicitly released. The allocator decides what is free.
                self.cuda.empty_cache()
                self._record(
                    "after_cache_trim",
                    reserved_before_trim_bytes=before["reserved_bytes"],
                    trim_elapsed_ms=(time.monotonic() - started) * 1000,
                )
        except BaseException:
            self._state = "failed"
            raise

    def after_kernel_warmup(self):
        if self._state != "pre_capture":
            raise RuntimeError("startup kernel warmup boundary is out of order")
        self._state = "capture"
        self._record("after_kernel_warmup")

    def after_capture(self):
        if self._state != "capture":
            raise RuntimeError("startup capture boundary is out of order")
        self._state = "sampler_warmup"
        self._record("after_capture")

    def finish(self):
        if self._state != "sampler_warmup":
            raise RuntimeError("startup sampler boundary is out of order")
        self._state = "finished"
        # Deliberately no flush here: capture/sampler allocations must stay warm.
        self._record("after_sampler_warmup")


def validate_startup_memory_deployment(package_root):
    """Read-only preflight of the actually imported Native/vendor worker."""
    root = Path(package_root)
    worker = root / "v1/worker/gpu_worker.py"
    helper = root / "isolation_startup_memory.py"
    tree = ast.parse(worker.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "Worker")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == "compile_or_warm_up_model")
    required = (
        "StartupMemoryAudit",
        "self.model_runner.maybe_remove_all_loras",
        "startup_memory.before_capture",
        "kernel_warmup",
        "startup_memory.after_kernel_warmup",
        "self.model_runner.capture_model",
        "startup_memory.after_capture",
        "self.model_runner._dummy_sampler_run",
        "startup_memory.finish",
    )
    calls = [ast.unparse(n.func) for n in sorted(
        (n for n in ast.walk(method) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )]
    if any(calls.count(name) != 1 for name in required):
        raise RuntimeError(f"startup memory contract is not wired into {worker}")
    positions = [calls.index(name) for name in required]
    if positions != sorted(positions):
        raise RuntimeError(f"startup memory boundaries are out of order in {worker}")
    audit_call = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == "StartupMemoryAudit")
    markers = [n for n in ast.walk(method) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Constant) and n.value.value is True
               and any(ast.unparse(t) == "self._isolation_startup_memory_seen"
                       for t in n.targets)]
    if (not any(k.arg == "first_startup" for k in audit_call.keywords)
            or not any(n.lineno < audit_call.lineno for n in markers)):
        raise RuntimeError(f"startup worker-lifetime guard is missing in {worker}")
    return {
        "contract_version": STARTUP_MEMORY_CONTRACT_VERSION,
        "worker": str(worker), "helper": str(helper),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
    }
