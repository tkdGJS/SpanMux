#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the vendor_v3 Green Context Python bridge.

This test assumes `build_vendor_v3_bridge.sh` has already produced
`libvllm_green_context.so`. It verifies that PyTorch can wrap both decode and
prefill Green Context CUstreams and run simple tensor operations on them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendor_root(root: Path) -> Path:
    return Path(os.environ.get("VLLM_VENDOR_DIR", root / "vendor_v3"))


def main() -> int:
    root = _repo_root()
    vendor_root = _vendor_root(root)
    sys.path.insert(0, str(vendor_root))

    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT", "1")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS", "8")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS", "32")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT", "1")

    import torch

    from vllm.model_executor.layers.green_context import (
        green_context_state,
        run_on_decode_green_stream,
        run_on_prefill_green_stream,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(
        int(os.environ.get("VLLM_OPERATOR_SELECTIVE_PD_GREEN_DEVICE", "0"))
    )
    x = torch.arange(4096, device="cuda", dtype=torch.float32)
    y = run_on_decode_green_stream(lambda: x * 2.0 + 1.0)
    z = run_on_prefill_green_stream(lambda: x * 3.0 + 2.0)
    torch.cuda.synchronize()

    x_cpu = x.cpu()
    expected_decode = x_cpu * 2.0 + 1.0
    actual_decode = y.cpu()
    if not torch.equal(actual_decode, expected_decode):
        max_abs = (actual_decode - expected_decode).abs().max().item()
        raise RuntimeError(f"decode output mismatch: max_abs={max_abs}")

    expected_prefill = x_cpu * 3.0 + 2.0
    actual_prefill = z.cpu()
    if not torch.equal(actual_prefill, expected_prefill):
        max_abs = (actual_prefill - expected_prefill).abs().max().item()
        raise RuntimeError(f"prefill output mismatch: max_abs={max_abs}")

    state = green_context_state()
    if state["green_context_active"] != 1.0:
        raise RuntimeError(f"Green Context bridge inactive: {state}")
    if state["green_context_fallback"] != 0.0:
        raise RuntimeError(f"Green Context bridge fell back: {state}")
    print(f"[STATE] {state}")
    print("[OK] vendor_v3 Green Context Python bridge validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
