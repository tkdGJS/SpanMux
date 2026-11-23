#!/usr/bin/env python3
"""CUDA exactness smoke for the direct fragmented H2D multicopy op."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v4"


def main() -> None:
    try:
        import torch
    except ImportError:
        print("[SKIP] PyTorch is unavailable")
        return

    if not torch.cuda.is_available():
        print("[SKIP] CUDA is unavailable")
        return

    sys.path.insert(0, str(VENDOR))
    from vllm import _custom_ops as ops

    if not hasattr(torch.ops._C_cache_ops, "tiered_h2d_multicopy"):
        print("[SKIP] vLLM was not rebuilt with tiered_h2d_multicopy")
        return

    pages, block_size, heads, head_dim = 6, 4, 2, 8
    destination_ids = (5, 6, 7, 1, 3, 4)
    shape = (pages * block_size, heads, head_dim)
    src_key = torch.arange(
        torch.tensor(shape).prod().item(), dtype=torch.float16
    ).reshape(shape).pin_memory()
    src_value = (src_key + 1000).pin_memory()
    # Match TRITON_ATTN's real allocation: unbind(1) produces block-strided
    # K/V views whose individual blocks are contiguous, but the full tensors
    # are not because K and V blocks are interleaved.
    dst_kv = torch.zeros(
        (9, 2, block_size, heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    dst_key, dst_value = dst_kv.unbind(1)
    assert not dst_key.is_contiguous()
    assert not dst_value.is_contiguous()
    ids = torch.tensor(destination_ids, dtype=torch.int64)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        copy_groups = ops.tiered_h2d_multicopy(
            src_key, src_value, dst_key, dst_value, ids, block_size
        )
        done = torch.cuda.Event()
        done.record()
    done.synchronize()

    # Interleaved K/V storage requires one two-copy group per destination page.
    assert copy_groups == pages, copy_groups
    for page, block_id in enumerate(destination_ids):
        start = page * block_size
        end = start + block_size
        torch.testing.assert_close(dst_key[block_id].cpu(), src_key[start:end])
        torch.testing.assert_close(
            dst_value[block_id].cpu(), src_value[start:end]
        )
    untouched = sorted(set(range(dst_key.shape[0])) - set(destination_ids))
    assert torch.count_nonzero(dst_key[untouched]).item() == 0
    assert torch.count_nonzero(dst_value[untouched]).item() == 0

    duplicate_ids = torch.tensor((1, 1), dtype=torch.int64)
    try:
        ops.tiered_h2d_multicopy(
            src_key[: 2 * block_size],
            src_value[: 2 * block_size],
            dst_key,
            dst_value,
            duplicate_ids,
            block_size,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate destination IDs must be rejected")

    print(
        "[OK] tiered H2D multicopy interleaved-layout exactness and "
        "stream ordering"
    )


if __name__ == "__main__":
    main()
