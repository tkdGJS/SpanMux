#!/usr/bin/env python3
"""Smoke test for the standalone tiered blockwise attention executor.

This script intentionally runs outside the vLLM serving path. It compares the
0.5.x reference data-plane against dense causal attention for a small synthetic
case, including at least one CPU/DRAM K/V block when running on CUDA.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-dir",
        default=os.environ.get("VLLM_VENDOR_DIR"),
        help="Path containing the vllm Python package; defaults to ./vendor_v4.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device for query and HBM-resident blocks.",
    )
    parser.add_argument("--query-tokens", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--block-tokens", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument(
        "--triton-partial",
        action="store_true",
        help=(
            "Enable the experimental Triton partial-attention fast path. "
            "This requires --device cuda and falls back inside the executor "
            "if the path is unsupported."
        ),
    )
    parser.add_argument(
        "--triton-fused-update",
        action="store_true",
        help=(
            "Enable the fused Triton group-attention plus online-softmax "
            "state update path. This requires --device cuda."
        ),
    )
    parser.add_argument(
        "--native-reduction",
        action="store_true",
        help=(
            "Use the opt-in native-shaped QK/PV reduction in the fused "
            "update. Requires --triton-fused-update."
        ),
    )
    parser.add_argument("--triton-block-m", type=int, default=16)
    parser.add_argument("--triton-block-n", type=int, default=64)
    return parser.parse_args()


def resolve_vendor_dir(arg_vendor_dir: str | None) -> Path:
    if arg_vendor_dir:
        return Path(arg_vendor_dir).resolve()
    return Path(__file__).resolve().parents[2] / "vendor_v4"


def dense_reference(
    torch,
    query,
    keys,
    values,
    *,
    query_start: int,
    softmax_scale: float,
):
    key = torch.cat(keys, dim=0).to(device=query.device)
    value = torch.cat(values, dim=0).to(device=query.device)

    q_heads = int(query.shape[1])
    kv_heads = int(key.shape[1])
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError("query-head count must be divisible by kv-head count")
        repeat = q_heads // kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    q = query.to(dtype=torch.float32)
    k = key.to(dtype=torch.float32)
    v = value.to(dtype=torch.float32)
    scores = torch.einsum("qhd,khd->qhk", q, k) * softmax_scale

    query_positions = torch.arange(
        query_start,
        query_start + int(query.shape[0]),
        device=query.device,
        dtype=torch.long,
    )
    key_positions = torch.arange(
        0,
        int(key.shape[0]),
        device=query.device,
        dtype=torch.long,
    )
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~mask.unsqueeze(1), -float("inf"))

    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("qhk,khd->qhd", probs, v).to(dtype=query.dtype)
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous()
    return output, lse


def main() -> int:
    args = parse_args()
    if args.native_reduction and not args.triton_fused_update:
        raise SystemExit(
            "--native-reduction requires --triton-fused-update"
        )
    vendor_dir = resolve_vendor_dir(args.vendor_dir)
    sys.path.insert(0, str(vendor_dir))
    if args.triton_partial:
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_TRITON_PARTIAL"] = "1"
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_M"] = str(
            args.triton_block_m
        )
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_N"] = str(
            args.triton_block_n
        )
    if args.triton_fused_update:
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE"
        ] = "1"
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_M"] = str(
            args.triton_block_m
        )
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_N"] = str(
            args.triton_block_n
        )
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION"] = (
            "1" if args.native_reduction else "0"
        )

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required for this smoke test") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    from vllm.v1.attention.ops.tiered_blockwise_attention import (
        TieredBlockwiseAttentionExecutor,
        TieredKVBlock,
    )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    softmax_scale = float(args.head_dim**-0.5)

    total_key_tokens = args.blocks * args.block_tokens
    query_start = total_key_tokens - args.query_tokens
    query = torch.randn(
        args.query_tokens,
        args.query_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )

    blocks: list[TieredKVBlock] = []
    ref_keys = []
    ref_values = []
    for block_idx in range(args.blocks):
        block_device = device
        tier = "HBM"
        if device.type == "cuda" and block_idx % 2 == 1:
            block_device = torch.device("cpu")
            tier = "DRAM"
        key = torch.randn(
            args.block_tokens,
            args.kv_heads,
            args.head_dim,
            device=block_device,
            dtype=dtype,
        )
        value = torch.randn(
            args.block_tokens,
            args.kv_heads,
            args.head_dim,
            device=block_device,
            dtype=dtype,
        )
        blocks.append(
            TieredKVBlock(
                key=key,
                value=value,
                token_start=block_idx * args.block_tokens,
                tier=tier,
                logical_page=block_idx,
            )
        )
        ref_keys.append(key)
        ref_values.append(value)

    executor = TieredBlockwiseAttentionExecutor(staging_slots=2)
    result = executor.run(
        query,
        blocks,
        query_start=query_start,
        softmax_scale=softmax_scale,
    )
    ref_output, ref_lse = dense_reference(
        torch,
        query,
        ref_keys,
        ref_values,
        query_start=query_start,
        softmax_scale=softmax_scale,
    )

    output_ok = torch.allclose(
        result.output, ref_output, atol=args.atol, rtol=args.rtol
    )
    lse_ok = torch.allclose(result.lse, ref_lse, atol=args.atol, rtol=args.rtol)
    print("[tiered_blockwise_attention_smoke]")
    print(f"vendor_dir={vendor_dir}")
    print(f"device={device} dtype={dtype}")
    print(f"triton_partial={int(args.triton_partial)}")
    print(f"triton_fused_update={int(args.triton_fused_update)}")
    print(f"native_reduction={int(args.native_reduction)}")
    print(f"output_ok={int(output_ok)} lse_ok={int(lse_ok)}")
    print(result.stats)

    if (
        args.native_reduction
        and result.stats.context_ring_native_reduction_enabled != 1
    ):
        print("native-shaped reduction was requested but not active")
        return 1
    if not output_ok or not lse_ok:
        max_output_err = (result.output - ref_output).abs().max().item()
        max_lse_err = (result.lse - ref_lse).abs().max().item()
        print(f"max_output_err={max_output_err}")
        print(f"max_lse_err={max_lse_err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
