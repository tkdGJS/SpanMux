#!/usr/bin/env python3
"""Validate shared staging across model-forward and reload execution modes."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_blockwise_attention.py"
)


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("[SKIP] torch is unavailable")
        return 0

    spec = importlib.util.spec_from_file_location(
        "staging_cross_thread_test_module",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    staging = module.HBMStagingBuffer(num_slots=1)
    source_key = torch.arange(24, dtype=torch.float32).view(3, 2, 4)
    source_value = source_key + 100

    # Match the first-touch context-ring path under model execution.
    with torch.inference_mode():
        staged_key, staged_value, _ = staging.materialize_group(
            [(source_key, source_value)],
            target_device=torch.device("cpu"),
            slot_index=0,
        )
    if (
        staging._key_slots[0].is_inference()
        or staging._value_slots[0].is_inference()
    ):
        raise AssertionError("shared staging slots must be normal tensors")
    torch.testing.assert_close(staged_key, source_key)
    torch.testing.assert_close(staged_value, source_value)

    # Match the native-reload ThreadPoolExecutor path outside model forward.
    replacement_key = source_key + 10
    replacement_value = source_value + 10
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            staging.materialize_group,
            [(replacement_key, replacement_value)],
            target_device=torch.device("cpu"),
            slot_index=0,
        )
        reloaded_key, reloaded_value, _ = future.result()

    torch.testing.assert_close(reloaded_key, replacement_key)
    torch.testing.assert_close(reloaded_value, replacement_value)
    print("[OK] shared staging survives cross-thread inference-mode reuse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
