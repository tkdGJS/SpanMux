# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.pd_prefill_workspace import (
    LlamaPrefillWorkspaceLayout,
    PrefillWorkspaceError,
    PrefillWorkspaceManager,
)


class _FakeStream:
    def __init__(self) -> None:
        self.waited_events: list[object] = []

    def wait_event(self, event: object) -> None:
        self.waited_events.append(event)


def _layout() -> LlamaPrefillWorkspaceLayout:
    return LlamaPrefillWorkspaceLayout(
        qkv_width=12,
        attention_width=8,
        gate_up_width=32,
        activation_width=16,
    )


def test_arena_is_allocated_once_and_reused_after_event(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH", "1")
    manager = PrefillWorkspaceManager(
        enabled=True,
        strict=True,
        max_tokens=8,
        slot_count=1,
    )
    assert manager.configure_llama(
        layout=_layout(),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    first_stream = _FakeStream()
    first = manager.acquire(
        profile_id="d30_p10",
        profile_epoch=3,
        num_tokens=4,
        handoff_generation=7,
        stream=first_stream,
    )
    first_qkv = manager.buffer(
        first,
        "qkv",
        num_tokens=4,
        width=12,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert first_qkv is not None
    first_pointer = first_qkv.data_ptr()
    first_gate_up = manager.buffer(
        first,
        "gate_up",
        num_tokens=4,
        width=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert first_gate_up is not None
    # Attention and MLP execute sequentially, so their first regions alias.
    assert first_gate_up.data_ptr() == first_pointer
    completion = object()
    manager.finish(first, completion_event=completion)
    pending = manager.debug_snapshot()
    assert pending["pd_prefill_workspace_state"] == "pending"
    assert pending["pd_prefill_workspace_active_generation"] == first.generation
    assert pending["pd_prefill_workspace_active_profile_id"] == "d30_p10"

    second_stream = _FakeStream()
    second = manager.acquire(
        profile_id="d0_p40",
        profile_epoch=4,
        num_tokens=4,
        stream=second_stream,
    )
    second_qkv = manager.buffer(
        second,
        "qkv",
        num_tokens=4,
        width=12,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert second_qkv is not None
    assert second_qkv.data_ptr() == first_pointer
    assert second_stream.waited_events == [completion]

    snapshot = manager.debug_snapshot()
    assert snapshot["pd_prefill_workspace_acquire_count"] == 2
    assert snapshot["pd_prefill_workspace_reuse_count"] == 1
    assert snapshot["pd_prefill_workspace_arena_bytes"] == (
        8
        * max(12 + 8, 32 + 16)
        * torch.tensor([], dtype=torch.float32).element_size()
    )


def test_profile_handoff_keeps_same_future_owner(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH", "1")
    manager = PrefillWorkspaceManager(
        enabled=True,
        strict=True,
        max_tokens=8,
        slot_count=1,
    )
    execution = manager.acquire(
        profile_id="d30_p10",
        profile_epoch=3,
        num_tokens=4,
        handoff_generation=11,
        stream=_FakeStream(),
    )
    manager.handoff(
        execution,
        handoff_generation=11,
        source_profile_id="d30_p10",
        source_profile_epoch=3,
        target_profile_id="d0_p40",
        target_profile_epoch=4,
    )
    assert execution.profile_id == "d0_p40"
    assert execution.profile_epoch == 4
    assert manager.debug_snapshot()["pd_prefill_workspace_handoff_count"] == 1

    with pytest.raises(PrefillWorkspaceError, match="source is not"):
        manager.handoff(
            execution,
            handoff_generation=11,
            source_profile_id="d30_p10",
            source_profile_epoch=3,
            target_profile_id="d34_p6",
            target_profile_epoch=5,
        )


def test_slot_pool_covers_depth_two_without_cross_future_alias(
    monkeypatch,
) -> None:
    manager = PrefillWorkspaceManager(
        enabled=True,
        strict=True,
        max_tokens=8,
        slot_count=1,
    )
    monkeypatch.setenv("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH", "2")
    with pytest.raises(PrefillWorkspaceError, match="slots must cover"):
        manager.acquire(
            profile_id="d30_p10",
            profile_epoch=0,
            num_tokens=4,
            stream=_FakeStream(),
        )

    manager = PrefillWorkspaceManager(
        enabled=True,
        strict=True,
        max_tokens=8,
        slot_count=2,
    )
    assert manager.configure_llama(
        layout=_layout(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        phase_slot=0,
    )
    assert manager.configure_llama(
        layout=_layout(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        phase_slot=1,
    )
    first = manager.acquire(
        profile_id="d30_p10",
        profile_epoch=0,
        num_tokens=4,
        phase_slot=0,
        stream=_FakeStream(),
    )
    second = manager.acquire(
        profile_id="d34_p6",
        profile_epoch=0,
        num_tokens=4,
        phase_slot=1,
        stream=_FakeStream(),
    )
    first_qkv = manager.buffer(
        first,
        "qkv",
        num_tokens=4,
        width=12,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    second_qkv = manager.buffer(
        second,
        "qkv",
        num_tokens=4,
        width=12,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert first_qkv is not None and second_qkv is not None
    assert first_qkv.data_ptr() != second_qkv.data_ptr()

    with pytest.raises(PrefillWorkspaceError, match="another Future is active"):
        manager.acquire(
            profile_id="d30_p10",
            profile_epoch=0,
            num_tokens=4,
            phase_slot=0,
            stream=_FakeStream(),
        )

    first_done = object()
    manager.finish(first, completion_event=first_done)
    reuse_stream = _FakeStream()
    third = manager.acquire(
        profile_id="d0_p40",
        profile_epoch=1,
        num_tokens=4,
        phase_slot=0,
        stream=reuse_stream,
    )
    third_qkv = manager.buffer(
        third,
        "qkv",
        num_tokens=4,
        width=12,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert third_qkv is not None
    assert third_qkv.data_ptr() == first_qkv.data_ptr()
    assert reuse_stream.waited_events == [first_done]
    snapshot = manager.debug_snapshot()
    assert snapshot["pd_prefill_workspace_slot_count"] == 2
    assert snapshot["pd_prefill_workspace_configured_slot_count"] == 2
    assert snapshot["pd_prefill_workspace_active_slot_count"] == 2


def test_workspace_rejects_schedule_larger_than_capacity(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH", "1")
    manager = PrefillWorkspaceManager(
        enabled=True,
        strict=True,
        max_tokens=8,
        slot_count=1,
    )
    with pytest.raises(PrefillWorkspaceError, match="exceeds.*capacity"):
        manager.acquire(
            profile_id="d30_p10",
            profile_epoch=0,
            num_tokens=9,
            stream=_FakeStream(),
        )


def test_unquantized_linear_writes_into_caller_buffer() -> None:
    method = UnquantizedLinearMethod()
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    layer = SimpleNamespace(weight=weight)
    inputs = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
        dtype=torch.float32,
    )
    output = torch.empty((2, 2), dtype=torch.float32)

    result = method.apply_out(layer, inputs, output)

    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result, inputs @ weight.t())
