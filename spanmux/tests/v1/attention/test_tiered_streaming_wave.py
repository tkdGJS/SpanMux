# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.ops.tiered_blockwise_attention import (
    TieredKVBlock,
    TieredBlockwiseAttentionExecutor,
    _OnlineSoftmaxState,
    _build_streaming_compute_plan,
    context_ring_streaming_wave_groups,
    context_ring_streaming_wave_tokens,
)


def _block(token_start: int, *, tier: str) -> TieredKVBlock:
    key = torch.zeros((16, 2, 4), dtype=torch.float16)
    return TieredKVBlock(
        key=key,
        value=key.clone(),
        token_start=token_start,
        tier=tier,
    )


def test_streaming_wave_defaults_to_original_single_group(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS",
        raising=False,
    )
    assert context_ring_streaming_wave_groups() == 1
    assert context_ring_streaming_wave_tokens(2048) == 2048


def test_streaming_wave_group_count_is_positive(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS",
        "0",
    )
    assert context_ring_streaming_wave_groups() == 1
    monkeypatch.setenv(
        "VLLM_VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS",
        "4",
    )
    assert context_ring_streaming_wave_groups() == 4
    assert context_ring_streaming_wave_tokens(2048) == 8192


def test_streaming_plan_coalesces_only_contiguous_dram_groups() -> None:
    blocks = [
        *[_block(index * 16, tier="DRAM") for index in range(4)],
        _block(64, tier="HBM"),
        *[_block((index + 5) * 16, tier="DRAM") for index in range(3)],
    ]

    plan = _build_streaming_compute_plan(
        blocks,
        compute_group_tokens=32,
        streaming_wave_groups=2,
    )

    assert len(plan.groups) == 3
    assert [len(group) for _, _, group in plan.groups] == [4, 1, 3]
    assert [group[0].tier for _, _, group in plan.groups] == [
        "DRAM",
        "HBM",
        "DRAM",
    ]
    assert plan.base_group_count == 5
    assert plan.max_wave_tokens == 64


def test_single_group_streaming_preserves_historical_group_loop() -> None:
    blocks = [_block(index * 16, tier="DRAM") for index in range(6)]

    plan = _build_streaming_compute_plan(
        blocks,
        compute_group_tokens=32,
        streaming_wave_groups=1,
    )

    assert [len(group) for _, _, group in plan.groups] == [2, 2, 2]
    assert [group[0].tier for _, _, group in plan.groups] == ["DRAM"] * 3
    assert plan.base_group_count == 3
    assert plan.max_wave_tokens == 32


def test_streaming_plan_does_not_cross_a_logical_gap() -> None:
    blocks = [
        _block(0, tier="DRAM"),
        _block(16, tier="DRAM"),
        _block(64, tier="DRAM"),
    ]

    plan = _build_streaming_compute_plan(
        blocks,
        compute_group_tokens=32,
        streaming_wave_groups=4,
    )

    assert [len(group) for _, _, group in plan.groups] == [2, 1]
    assert plan.base_group_count == 2


def test_streaming_plan_does_not_cross_a_tier_boundary() -> None:
    blocks = [
        _block(0, tier="DRAM"),
        _block(16, tier="STAGING"),
    ]

    plan = _build_streaming_compute_plan(
        blocks,
        compute_group_tokens=32,
        streaming_wave_groups=4,
    )

    assert [len(group) for _, _, group in plan.groups] == [1, 1]


def test_streaming_wave_matches_sequential_online_softmax_updates() -> None:
    torch.manual_seed(7)
    query = torch.randn((4, 2, 4), dtype=torch.float32)
    first_key = torch.randn((16, 2, 4), dtype=torch.float32)
    first_value = torch.randn_like(first_key)
    second_key = torch.randn((16, 2, 4), dtype=torch.float32)
    second_value = torch.randn_like(second_key)
    query_positions = torch.arange(28, 32, dtype=torch.long)
    scale = 4**-0.5

    sequential = _OnlineSoftmaxState(query)
    for key_start, key, value in (
        (0, first_key, first_value),
        (16, second_key, second_value),
    ):
        partial = TieredBlockwiseAttentionExecutor._partial_attention(
            query,
            key,
            value,
            query_positions=query_positions,
            key_start=key_start,
            causal=True,
            softmax_scale=scale,
        )
        sequential.merge(*partial)

    wave = _OnlineSoftmaxState(query)
    wave.merge(
        *TieredBlockwiseAttentionExecutor._partial_attention(
            query,
            torch.cat((first_key, second_key)),
            torch.cat((first_value, second_value)),
            query_positions=query_positions,
            key_start=0,
            causal=True,
            softmax_scale=scale,
        )
    )

    sequential_output, sequential_lse = sequential.finalize(query.dtype)
    wave_output, wave_lse = wave.finalize(query.dtype)
    torch.testing.assert_close(wave_output, sequential_output)
    torch.testing.assert_close(wave_lse, sequential_lse)
