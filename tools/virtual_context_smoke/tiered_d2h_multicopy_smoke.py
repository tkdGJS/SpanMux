#!/usr/bin/env python3
"""CUDA exactness smoke for fragmented HBM-to-DRAM multicopy."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v6"


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

    if not hasattr(torch.ops._C_cache_ops, "tiered_d2h_multicopy"):
        print("[SKIP] vLLM was not rebuilt with tiered_d2h_multicopy")
        return

    pages, block_size, heads, head_dim = 6, 4, 2, 8
    source_ids = (5, 6, 7, 1, 3, 4)
    source_kv = torch.arange(
        9 * 2 * block_size * heads * head_dim,
        device="cuda",
        dtype=torch.float16,
    ).reshape(9, 2, block_size, heads, head_dim)
    source_key, source_value = source_kv.unbind(1)
    source_keys = [source_key[block_id] for block_id in source_ids]
    source_values = [source_value[block_id] for block_id in source_ids]
    destination_key = torch.empty(
        (pages * block_size, heads, head_dim),
        dtype=torch.float16,
        pin_memory=True,
    )
    destination_value = torch.empty_like(
        destination_key,
        pin_memory=True,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        copy_groups = ops.tiered_d2h_multicopy(
            source_keys,
            source_values,
            destination_key,
            destination_value,
        )
        done = torch.cuda.Event()
        done.record()
    done.synchronize()

    # TRITON_ATTN interleaves K/V, so even consecutive block IDs are not
    # adjacent within one K or V view. The wrapper still submits all page
    # copies from one Python call and preserves the current stream ordering.
    assert copy_groups == pages, copy_groups
    for page, block_id in enumerate(source_ids):
        start = page * block_size
        end = start + block_size
        torch.testing.assert_close(
            destination_key[start:end],
            source_key[block_id].cpu(),
        )
        torch.testing.assert_close(
            destination_value[start:end],
            source_value[block_id].cpu(),
        )

    contiguous_key = torch.arange(
        pages * block_size * heads * head_dim,
        device="cuda",
        dtype=torch.float16,
    ).reshape(pages, block_size, heads, head_dim)
    contiguous_value = contiguous_key + 1000
    with torch.cuda.stream(stream):
        contiguous_groups = ops.tiered_d2h_multicopy(
            list(contiguous_key.unbind(0)),
            list(contiguous_value.unbind(0)),
            destination_key,
            destination_value,
        )
        done.record()
    done.synchronize()
    assert contiguous_groups == 1, contiguous_groups
    torch.testing.assert_close(
        destination_key,
        contiguous_key.reshape_as(destination_key).cpu(),
    )
    torch.testing.assert_close(
        destination_value,
        contiguous_value.reshape_as(destination_value).cpu(),
    )

    print(
        "[OK] tiered D2H multicopy fragmented/contiguous exactness and "
        "stream ordering"
    )


if __name__ == "__main__":
    main()
