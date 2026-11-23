#!/usr/bin/env python3
"""Smoke test for the vendor_v3 OpMux 0.4.x control plane.

This does not require CUDA. It validates the default-off import path, metric
collection, and the memory-guard policy that suppresses P/D overlap when the
configured residency budget is exceeded.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_opmux_module() -> ModuleType:
    module_path = (
        _repo_root()
        / "vendor_v3"
        / "vllm"
        / "model_executor"
        / "layers"
        / "operator_memory_shaping.py"
    )
    spec = importlib.util.spec_from_file_location("operator_memory_shaping", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["operator_memory_shaping"] = module
    spec.loader.exec_module(module)
    return module


class _Plan:
    num_decode_tokens = 8
    num_prefill_tokens = 1024


def main() -> int:
    opmux = _load_opmux_module()
    consume_opmux_metrics = opmux.consume_opmux_metrics
    filter_opmux_pd_execution = opmux.filter_opmux_pd_execution
    reset_opmux_metrics = opmux.reset_opmux_metrics

    os.environ.pop("VLLM_OPMUX_ENABLED", None)
    reset_opmux_metrics()
    disabled_metrics = consume_opmux_metrics()
    if disabled_metrics:
        raise AssertionError(f"default-off metrics should be empty: {disabled_metrics}")

    os.environ["VLLM_OPMUX_ENABLED"] = "1"
    os.environ["VLLM_OPMUX_POLICY"] = "memory_guard"
    os.environ["VLLM_OPMUX_MEMORY_PRESSURE_MODE"] = "medium"
    os.environ["VLLM_OPMUX_SEGMENTED_KV_COMMIT_TOKENS"] = "256"
    reset_opmux_metrics()
    use_green_context, overlap = filter_opmux_pd_execution(
        kind="phase",
        plan=_Plan(),
        use_green_context=True,
        overlap=True,
    )
    metrics = consume_opmux_metrics()
    if not use_green_context:
        raise AssertionError("memory_guard should keep GreenContext enabled")
    if overlap:
        raise AssertionError("memory_guard should suppress incompatible overlap")
    if metrics.get("opmux_co_run_suppressed_count", 0.0) < 1.0:
        raise AssertionError(f"missing suppression metric: {metrics}")
    if metrics.get("opmux_memory_compatible", 1.0) != 0.0:
        raise AssertionError(f"expected incompatible metric: {metrics}")

    print("[OK] vendor_v3 OpMux smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
