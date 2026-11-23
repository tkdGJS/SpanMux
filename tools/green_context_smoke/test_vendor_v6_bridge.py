#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU smoke test for the vendor_v6 fixed-split Green Context bridge.

Run ``build_vendor_v6_bridge.sh`` first. This test verifies that vendor_v6 is
the imported vLLM tree and that PyTorch can execute exact tensor operations on
both fixed-split Green Context CUDA streams.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendor_root(root: Path) -> Path:
    return Path(os.environ.get("VLLM_VENDOR_DIR", root / "vendor_v6")).resolve()


def _require_vendor_import(vendor_root: Path, module_file: str) -> None:
    loaded_path = Path(module_file).resolve()
    if vendor_root not in loaded_path.parents:
        raise RuntimeError(
            "vLLM was not imported from vendor_v6: "
            f"expected_root={vendor_root}, loaded={loaded_path}"
        )


def main() -> int:
    root = _repo_root()
    vendor_root = _vendor_root(root)
    bridge_path = (
        vendor_root
        / "vllm/model_executor/layers/green_context_lib"
        / "libvllm_green_context.so"
    )
    if not bridge_path.is_file():
        raise RuntimeError(
            f"vendor_v6 Green Context bridge is missing: {bridge_path}; "
            "run tools/green_context_smoke/build_vendor_v6_bridge.sh first"
        )

    sys.path.insert(0, str(vendor_root))

    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT", "1")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS", "8")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS", "32")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT", "1")

    import torch
    import vllm

    _require_vendor_import(vendor_root, vllm.__file__)

    from vllm.model_executor.layers.green_context import (
        green_context_state,
        run_on_decode_green_stream,
        run_on_prefill_green_stream,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = int(
        os.environ.get("VLLM_OPERATOR_SELECTIVE_PD_GREEN_DEVICE", "0")
    )
    decode_sms = int(
        os.environ["VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS"]
    )
    prefill_sms = int(
        os.environ["VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS"]
    )
    torch.cuda.set_device(device)
    actual_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if decode_sms <= 0 or prefill_sms <= 0:
        raise RuntimeError(
            "bridge smoke requires positive decode and prefill SM counts: "
            f"decode={decode_sms}, prefill={prefill_sms}"
        )
    if decode_sms + prefill_sms > actual_sms:
        raise RuntimeError(
            "configured Green Context split exceeds device SMs: "
            f"decode={decode_sms}, prefill={prefill_sms}, device={actual_sms}"
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
    if state["green_decode_sms"] != decode_sms:
        raise RuntimeError(f"decode SM state mismatch: {state}")
    if state["green_prefill_sms"] != prefill_sms:
        raise RuntimeError(f"prefill SM state mismatch: {state}")

    print(f"[VENDOR] {vendor_root}")
    print(f"[BRIDGE] {bridge_path}")
    print(f"[DEVICE] index={device} sms={actual_sms}")
    print(f"[STATE] {state}")
    print("[OK] vendor_v6 Green Context Python bridge validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
