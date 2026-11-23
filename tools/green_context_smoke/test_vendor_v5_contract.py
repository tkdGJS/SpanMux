#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Import-free source contract for vendor_v5 dynamic P/D SM routing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v5" / "vllm"


def _read(relative_path: str) -> str:
    return (VENDOR / relative_path).read_text(encoding="utf-8")


def _require(source: str, needle: str, description: str) -> None:
    if needle not in source:
        raise AssertionError(f"missing {description}: {needle}")


def main() -> int:
    output = _read("v1/core/sched/output.py")
    core = _read("v1/engine/core.py")
    worker = _read("v1/worker/gpu_worker.py")
    runner = _read("v1/worker/gpu_model_runner.py")
    green = _read("model_executor/layers/green_context.py")
    bridge = _read(
        "model_executor/layers/green_context_lib/green_context_bridge.cc"
    )

    for field in (
        "pd_sm_profile_id",
        "pd_sm_profile_epoch",
        "pd_sm_decode_sms",
        "pd_sm_prefill_sms",
    ):
        _require(output, field, "SchedulerOutput profile field")

    _require(
        core,
        'collective_rpc(\n                "initialize_dynamic_green_context_profiles"',
        "startup profile-bank initialization",
    )
    if core.count("self._assign_pd_sm_plan(") < 3:
        raise AssertionError("all single and paired submission paths need a plan")
    _require(core, "phase_sms == 0", "zero-SM phase admission guard")
    _require(core, "self._observe_pd_sm_plan(", "feedback observation")

    _require(worker, "paired P/D outputs must use one SM profile epoch", "pair guard")
    _require(runner, "_pd_green_context_sample_profile_id", "sample profile state")
    _require(runner, "profile_epoch=profile_epoch", "profile epoch routing")

    _require(green, "class _GreenContextProfileBank", "profile bank")
    _require(
        green,
        "previous.library.synchronize(previous.handle)",
        "quiescent CUDA handoff synchronization",
    )
    _require(green, "requested_epoch < self._active_epoch", "stale epoch guard")
    _require(
        green,
        "dynamic P/D SM execution reached GreenContext routing without",
        "missing-profile fail-fast",
    )

    _require(bridge, "vllm_green_pd_sm_count", "driver SM-count ABI")
    _require(bridge, "cuGreenCtxGetDevResource", "driver SM-count query")
    _require(green, "_sm_count_fn.restype = ctypes.c_int", "signed count ABI")
    _require(bridge, "return -1;", "SM-count query failure sentinel")
    _require(
        bridge,
        "cuCtxSetCurrent(handle->decode_context)",
        "decode-context synchronization",
    )
    _require(
        bridge,
        "cuCtxSetCurrent(handle->prefill_context)",
        "prefill-context synchronization",
    )

    print("[OK] vendor_v5 dynamic P/D SM source contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
