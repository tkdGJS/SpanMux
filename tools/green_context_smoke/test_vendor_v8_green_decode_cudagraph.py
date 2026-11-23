#!/usr/bin/env python3
"""Capture and replay a CUDA graph on the vendor_v8 Decode GreenContext stream."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v8"
LIBRARY = (
    VENDOR
    / "vllm/model_executor/layers/green_context_lib"
    / "libvllm_green_context.so"
)
sys.path.insert(0, str(VENDOR))

os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT", "1")
os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT", "1")
os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL", "1")
os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS", "32")
os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS", "8")
os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_LIB", str(LIBRARY))

try:
    import torch  # noqa: E402
except ModuleNotFoundError:
    print("[SKIP] PyTorch is unavailable")
    raise SystemExit(77)

from vllm.model_executor.layers.green_context import (  # noqa: E402
    green_context_phase_stream,
    green_context_state,
)


def main() -> int:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is unavailable")
        return 77
    if not LIBRARY.is_file():
        raise FileNotFoundError(f"missing GreenContext bridge: {LIBRARY}")

    device = torch.device("cuda:0")
    source = torch.ones(256, device=device)
    destination = torch.empty_like(source)
    graph = torch.cuda.CUDAGraph()

    with green_context_phase_stream(
        "decode", wait_for_completion=True
    ) as active:
        if not active:
            raise RuntimeError("Decode GreenContext stream was not activated")
        stream = torch.cuda.current_stream(device)
        destination.copy_(source * 2.0)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            destination.copy_(source * 2.0)
        source.fill_(3.0)
        graph.replay()
        stream.synchronize()
        if not torch.equal(destination, torch.full_like(destination, 6.0)):
            raise RuntimeError("GreenContext CUDA graph replay produced bad output")
        stream_pointer = int(stream.cuda_stream)

    state = green_context_state()
    if int(state.get("green_context_active", 0)) != 1:
        raise RuntimeError(f"GreenContext state was not active: {state!r}")
    if int(state.get("green_sm_count_verified", 0)) != 1:
        raise RuntimeError(f"GreenContext SM count was not verified: {state!r}")
    print(
        json.dumps(
            {
                "status": "passed",
                "decode_sms": int(state.get("green_decode_sms", 0)),
                "prefill_sms": int(state.get("green_prefill_sms", 0)),
                "stream_pointer": stream_pointer,
                "result": float(destination[0].item()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
