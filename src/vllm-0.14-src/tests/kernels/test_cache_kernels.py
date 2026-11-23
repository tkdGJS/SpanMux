# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CUDA kernels in cache_kernels.cu."""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. (pip install -e .)", allow_module_level=True
    )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_oob():
    """
    Tests for OOB read in gather_and_maybe_dequant_cache (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """

    batch_size = 1
    block_size = 64
    entry_size = 128

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # This will result in offset = 128 / block_size = 128 / 64 = 2
    # This will cause the kernel to try to read from
    # block_table[0, 2], but its size is only 2.
    seq_starts = torch.tensor([128], dtype=torch.int32, device="cuda")

    seq_len = 65
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # src_cache: [num_blocks, block_size, entry_size]
    num_blocks = 5
    src_cache = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )

    dst = torch.empty((seq_len, entry_size), dtype=torch.float16, device="cuda")

    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Calling the C++ function gather_and_maybe_dequant_cache
    ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        "auto",  # kv_cache_dtype
        scale,
        seq_starts,
    )

    torch.cuda.synchronize()
    assert True


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_tiered_h2d_staging_multicopy_mixed_sources():
    """CPU-pinned and CUDA historical groups land at exact Ring offsets."""

    shape_tail = (2, 4)
    cpu_key = torch.full(
        (2, *shape_tail), 11, dtype=torch.float16, pin_memory=True
    )
    cpu_value = torch.full(
        (2, *shape_tail), 12, dtype=torch.float16, pin_memory=True
    )
    cuda_key = torch.full(
        (3, *shape_tail), 21, dtype=torch.float16, device="cuda"
    )
    cuda_value = torch.full(
        (3, *shape_tail), 22, dtype=torch.float16, device="cuda"
    )
    tail_key = torch.full(
        (1, *shape_tail), 31, dtype=torch.float16, pin_memory=True
    )
    tail_value = torch.full(
        (1, *shape_tail), 32, dtype=torch.float16, pin_memory=True
    )
    dst_key = torch.full(
        (3, 7, *shape_tail), -1, dtype=torch.float16, device="cuda"
    )
    dst_value = torch.full_like(dst_key, -1)

    copy_runs = ops.tiered_h2d_staging_multicopy(
        [cpu_key, cuda_key, tail_key],
        [cpu_value, cuda_value, tail_value],
        dst_key,
        dst_value,
        torch.tensor([0, 0, 2], dtype=torch.int64),
        torch.tensor([0, 2, 1], dtype=torch.int64),
    )
    torch.cuda.synchronize()

    assert copy_runs >= 1
    torch.testing.assert_close(dst_key[0, :2].cpu(), cpu_key)
    torch.testing.assert_close(dst_value[0, :2].cpu(), cpu_value)
    torch.testing.assert_close(dst_key[0, 2:5], cuda_key)
    torch.testing.assert_close(dst_value[0, 2:5], cuda_value)
    torch.testing.assert_close(dst_key[2, 1:2].cpu(), tail_key)
    torch.testing.assert_close(dst_value[2, 1:2].cpu(), tail_value)
    assert torch.all(dst_key[1] == -1)


if __name__ == "__main__":
    pytest.main([__file__])
