#!/usr/bin/env python3
"""Smoke test for the guarded tiered backend DRAM-provider path.

The test does not start vLLM. It directly calls the backend hook with a small
synthetic paged KV cache, registers one logical page as CPU/DRAM-resident, and
compares the hook output with dense causal attention.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-dir",
        default=os.environ.get("VLLM_VENDOR_DIR"),
        help="Path containing the vllm Python package; defaults to ./vendor_v4.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--request-id", default="smoke-request-0")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--layer-name", default="smoke.layer.0.attn")
    parser.add_argument("--page-version", type=int, default=7)
    parser.add_argument(
        "--register-from-hbm",
        action="store_true",
        help="Register provider pages by copying from the synthetic HBM KV cache.",
    )
    parser.add_argument(
        "--contiguous-pool",
        action="store_true",
        help="Use the fixed contiguous pinned DRAM slab and bulk span path.",
    )
    parser.add_argument(
        "--triton-fused-update",
        action="store_true",
        help="Use the fused Triton online-softmax state-update path.",
    )
    parser.add_argument(
        "--fully-optimized",
        action="store_true",
        help=(
            "Require pool-only async direct D2H, tensorized page plans, "
            "pinned staging, and the fused Triton update path."
        ),
    )
    parser.add_argument(
        "--native-reduction",
        action="store_true",
        help=(
            "Use native-shaped QK/PV reductions in fused updates. Requires "
            "--triton-fused-update or --fully-optimized."
        ),
    )
    parser.add_argument(
        "--batched-partial",
        action="store_true",
        help=(
            "Batch all historical groups into one partial-attention launch "
            "and one merge launch."
        ),
    )
    parser.add_argument(
        "--batched-partial-strict",
        action="store_true",
        help="Fail instead of falling back to the historical-group host loop.",
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
    return parser.parse_args()


def resolve_vendor_dir(arg_vendor_dir: str | None) -> Path:
    if arg_vendor_dir:
        return Path(arg_vendor_dir).resolve()
    return Path(__file__).resolve().parents[2] / "vendor_v4"


@dataclass
class FakeTritonAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: object
    max_seq_len: int
    seq_lens: object
    block_table: object
    request_ids: list[str]
    tiered_page_versions: dict[str, dict[int, int]]
    tiered_reload_required_pages: dict[str, set[int]]
    use_cascade: bool = False
    mm_prefix_range: object | None = None


def dense_reference(
    torch,
    query,
    keys,
    values,
    *,
    query_start: int,
    softmax_scale: float,
):
    key = torch.cat([k.to(device=query.device) for k in keys], dim=0)
    value = torch.cat([v.to(device=query.device) for v in values], dim=0)

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
    key_positions = torch.arange(0, int(key.shape[0]), device=query.device)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~mask.unsqueeze(1), -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,khd->qhd", probs, v).to(dtype=query.dtype)


def main() -> int:
    args = parse_args()
    if args.native_reduction and not (
        args.triton_fused_update or args.fully_optimized
    ):
        raise SystemExit(
            "--native-reduction requires --triton-fused-update or "
            "--fully-optimized"
        )
    if args.batched_partial_strict and not args.batched_partial:
        raise SystemExit(
            "--batched-partial-strict requires --batched-partial"
        )
    vendor_dir = resolve_vendor_dir(args.vendor_dir)
    sys.path.insert(0, str(vendor_dir))

    os.environ["VLLM_VIRTUAL_CONTEXT_RING_BACKEND"] = "1"
    os.environ["VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"] = "1"
    if args.contiguous_pool or args.fully_optimized:
        os.environ["VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL"] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_CONTIGUOUS"
        ] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MIRROR_DICT"
        ] = "0"
        os.environ["VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MAX_SLOTS"] = str(
            max(args.blocks * args.request_count, 2)
        )
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_PAGE_TOKENS"
        ] = str(args.block_tokens)
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_BULK_COPY_TOKENS"] = str(
            args.blocks * args.block_tokens
        )
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_COMPUTE_GROUP_TOKENS"] = str(
            args.blocks * args.block_tokens
        )
        if args.device == "cuda":
            os.environ[
                "VLLM_VIRTUAL_CONTEXT_DRAM_PROVIDER_PIN_MEMORY"
            ] = "1"
    if args.triton_fused_update or args.fully_optimized:
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE"
        ] = "1"
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION"] = (
            "1" if args.native_reduction else "0"
        )
    if args.fully_optimized:
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_PAGED_HBM_UPDATE"] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT"
        ] = "1"
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_ATTENTION_PLAN_TENSORS"] = "1"
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_ASYNC_PREFETCH"] = "1"
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_STAGING_SLOTS"] = "2"
        os.environ["VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC"] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER"
        ] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE"
        ] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_VIEW_DISPATCH"
        ] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MIRROR_DICT"
        ] = "0"
    if args.batched_partial:
        os.environ["VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL"] = "1"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL_STRICT"
        ] = "1" if args.batched_partial_strict else "0"
        os.environ[
            "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_WORKSPACE_BYTES"
        ] = str(1024 * 1024 * 1024)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required for this smoke test") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.ops import tiered_blockwise_backend

    try_run_tiered_blockwise_backend = (
        tiered_blockwise_backend.try_run_tiered_blockwise_backend
    )
    reset_tiered_context_ring_runtime_metrics = getattr(
        tiered_blockwise_backend,
        "reset_tiered_context_ring_runtime_metrics",
        lambda: None,
    )
    consume_tiered_context_ring_runtime_metrics = getattr(
        tiered_blockwise_backend,
        "consume_tiered_context_ring_runtime_metrics",
        lambda: {},
    )
    from vllm.v1.attention.ops.tiered_residency_provider import (
        clear_dram_kv_pages,
        build_dram_kv_page_lookup_plan,
        build_tiered_attention_page_tensor_plan,
        acquire_dram_kv_pool_slots,
        get_dram_kv_request_key,
        get_dram_kv_page_registry_stats,
        register_hbm_kv_cache_pages,
        register_dram_kv_page,
    )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    softmax_scale = float(args.head_dim**-0.5)

    if args.request_count <= 0:
        raise ValueError("request-count must be positive")
    if args.query_tokens <= 0:
        raise ValueError("query-tokens must be positive")
    seq_len = args.blocks * args.block_tokens
    if args.query_tokens > seq_len:
        raise ValueError("query-tokens cannot exceed the sequence length")
    query_start = seq_len - args.query_tokens
    request_ids = [
        args.request_id
        if args.request_count == 1
        else f"{args.request_id}-{index}"
        for index in range(args.request_count)
    ]
    total_query_tokens = args.request_count * args.query_tokens
    query = torch.randn(
        total_query_tokens,
        args.query_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    output = torch.empty_like(query)
    kv_cache = torch.empty(
        args.request_count * args.blocks,
        2,
        args.block_tokens,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    block_table = torch.arange(
        args.request_count * args.blocks,
        device=device,
        dtype=torch.int32,
    ).view(args.request_count, args.blocks)

    ref_keys: list[list[object]] = []
    ref_values: list[list[object]] = []
    clear_dram_kv_pages()
    dram_page = 0 if args.register_from_hbm else (
        1 if args.blocks > 1 else 0
    )
    for req_idx, request_id in enumerate(request_ids):
        request_keys = []
        request_values = []
        for logical_page in range(args.blocks):
            key = torch.randn(
                args.block_tokens,
                args.kv_heads,
                args.head_dim,
                device=device,
                dtype=dtype,
            )
            value = torch.randn(
                args.block_tokens,
                args.kv_heads,
                args.head_dim,
                device=device,
                dtype=dtype,
            )
            physical_block = req_idx * args.blocks + logical_page
            kv_cache[physical_block, 0].copy_(key)
            kv_cache[physical_block, 1].copy_(value)
            if logical_page == dram_page and not args.register_from_hbm:
                register_dram_kv_page(
                    request_id=request_id,
                    layer_name=args.layer_name,
                    logical_page=logical_page,
                    key=key,
                    value=value,
                    token_start=logical_page * args.block_tokens,
                    version=args.page_version,
                    copy_to_cpu=True,
                )
            request_keys.append(key)
            request_values.append(value)
        ref_keys.append(request_keys)
        ref_values.append(request_values)

    page_versions = {
        request_id: {
            logical_page: args.page_version
            for logical_page in range(args.blocks)
        }
        for request_id in request_ids
    }
    if args.register_from_hbm:
        os.environ["VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM"] = "1"
        # Register one logical page so the optimized smoke exercises a mixed
        # DRAM-staged + paged-HBM traversal in the same online-softmax state.
        registration_seq_len = args.block_tokens
        registered = register_hbm_kv_cache_pages(
            layer_name=args.layer_name,
            key_cache=kv_cache[:, 0],
            value_cache=kv_cache[:, 1],
            block_table=block_table,
            seq_lens=torch.full(
                (args.request_count,),
                registration_seq_len,
                device=device,
            ),
            query_start_loc=torch.arange(
                0,
                (args.request_count + 1) * registration_seq_len,
                registration_seq_len,
                device=device,
            ),
            request_ids=request_ids,
            page_versions_by_request=page_versions,
        )
        if registered != args.request_count:
            raise RuntimeError(
                "expected to register one HBM page per request, got "
                f"{registered}"
            )
        if args.fully_optimized:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                stats = get_dram_kv_page_registry_stats()
                if (
                    stats.get(
                        "dram_provider_register_from_hbm_async_pending_pages",
                        0,
                    )
                    == 0
                    and stats.get(
                        "dram_provider_register_from_hbm_direct_success_pages",
                        0,
                    )
                    >= args.request_count
                    and stats.get(
                        "dram_provider_register_from_hbm_direct_enqueue_count",
                        0,
                    )
                    > 0
                    and stats.get(
                        "dram_provider_d2h_source_view_dispatch_materialize_pages",
                        0,
                    )
                    >= args.request_count
                    and stats.get(
                        "dram_provider_d2h_source_lease_active_blocks",
                        -1,
                    )
                    == 0
                ):
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError(
                    "timed out waiting for optimized direct D2H registration"
                )
            if stats.get(
                "dram_provider_d2h_source_view_dispatch_fallback_count",
                0,
            ):
                raise RuntimeError("D2H source-view dispatch used fallback")

    metadata = FakeTritonAttentionMetadata(
        num_actual_tokens=total_query_tokens,
        max_query_len=args.query_tokens,
        query_start_loc=torch.arange(
            0,
            total_query_tokens + 1,
            args.query_tokens,
            device=device,
        ),
        max_seq_len=seq_len,
        seq_lens=torch.full(
            (args.request_count,), seq_len, device=device
        ),
        block_table=block_table,
        request_ids=request_ids,
        tiered_page_versions=page_versions,
        tiered_reload_required_pages={
            request_id: {dram_page} for request_id in request_ids
        },
    )
    if args.fully_optimized:
        page_token_counts = {
            logical_page: args.block_tokens
            for logical_page in range(args.blocks)
        }
        from vllm.v1.attention.ops.tiered_residency_provider import (
            TieredAttentionPagePlanBundle,
        )

        page_bundles = {}
        row_plans = []
        for req_idx, request_id in enumerate(request_ids):
            request_key = get_dram_kv_request_key(
                request_id,
                create=False,
            )
            dram_plan = build_dram_kv_page_lookup_plan(
                request_id=request_id,
                request_key=request_key,
                layer_name=args.layer_name,
                page_token_counts=page_token_counts,
                page_versions=page_versions[request_id],
                strict_pages={dram_page},
            )
            tensor_plan = build_tiered_attention_page_tensor_plan(
                request_id=request_id,
                layer_name=args.layer_name,
                page_token_counts=page_token_counts,
                block_ids_by_page={
                    logical_page: req_idx * args.blocks + logical_page
                    for logical_page in range(args.blocks)
                },
                dram_plan=dram_plan,
                required_dram_pages={dram_page},
                block_size=args.block_tokens,
                request_numeric_id=req_idx,
                device=device,
            )
            pool_pages = [
                logical_page
                for logical_page, lookup in dram_plan.lookups.items()
                if lookup.source == "pool"
            ]
            leases = acquire_dram_kv_pool_slots([
                (
                    dram_plan.lookups[logical_page].slot_id,
                    dram_plan.lookups[logical_page].version,
                )
                for logical_page in pool_pages
            ])
            if leases is None:
                raise RuntimeError(
                    "failed to acquire optimized DRAM pool read leases"
                )
            page_bundle = TieredAttentionPagePlanBundle(
                dram_plan=dram_plan,
                tensor_plan=tensor_plan,
                pool_leases_by_page=dict(zip(pool_pages, leases)),
            )
            page_bundles[request_id] = page_bundle
            row_plans.append(page_bundle)
        metadata.tiered_attention_page_tensor_plans = page_bundles
        metadata.tiered_attention_page_row_plans = row_plans

    reset_tiered_context_ring_runtime_metrics()
    did_run = try_run_tiered_blockwise_backend(
        query=query,
        key_cache=kv_cache[:, 0],
        value_cache=kv_cache[:, 1],
        output=output,
        attn_metadata=metadata,
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        scale=softmax_scale,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0,
        sinks=None,
        output_scale=None,
        layer_name=args.layer_name,
    )
    ring_metrics = consume_tiered_context_ring_runtime_metrics()
    ref_output = torch.cat([
        dense_reference(
            torch,
            query[
                req_idx * args.query_tokens:(req_idx + 1)
                * args.query_tokens
            ],
            ref_keys[req_idx],
            ref_values[req_idx],
            query_start=query_start,
            softmax_scale=softmax_scale,
        )
        for req_idx in range(args.request_count)
    ])
    output_ok = torch.allclose(output, ref_output, atol=args.atol, rtol=args.rtol)
    print("[tiered_backend_provider_smoke]")
    print(f"vendor_dir={vendor_dir}")
    print(f"device={device} dtype={dtype}")
    print(f"request_count={args.request_count}")
    print(f"contiguous_pool={int(args.contiguous_pool)}")
    print(f"triton_fused_update={int(args.triton_fused_update)}")
    print(f"fully_optimized={int(args.fully_optimized)}")
    print(f"native_reduction={int(args.native_reduction)}")
    print(f"batched_partial={int(args.batched_partial)}")
    print(f"did_run={int(did_run)} output_ok={int(output_ok)}")
    print(f"ring_metrics={ring_metrics}")
    print(get_dram_kv_page_registry_stats())

    batched_ok = (
        not args.batched_partial
        or (
            ring_metrics.get(
                "worker_context_ring_batched_partial_successes", 0
            )
            > 0
            and ring_metrics.get(
                "worker_context_ring_batched_partial_fallbacks", 0
            )
            == 0
            and ring_metrics.get(
                "worker_context_ring_host_group_loop_calls", 0
            )
            == 0
        )
    )
    if not did_run or not output_ok or not batched_ok:
        max_output_err = (output - ref_output).abs().max().item()
        print(f"max_output_err={max_output_err}")
        print(f"batched_ok={int(batched_ok)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
