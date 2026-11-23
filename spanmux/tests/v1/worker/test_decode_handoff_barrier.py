# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.tiered_blockwise_attention import (
    TieredBlockwiseAttentionExecutor,
)
from vllm.v1.attention.ops.tiered_residency_provider import (
    clear_dram_kv_pages,
    register_dram_kv_page,
    reload_dram_kv_pages_to_hbm,
)
from vllm.v1.worker import gpu_model_runner as runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _FakeVirtualContext:
    def __init__(self) -> None:
        self.pending = {0: {0}, 1: {0}}

    def plan_hbm_reload_pages(self, *, request_ids, layer_id=None):
        assert request_ids == {"long-2"}
        layer_ids = (
            self.pending
            if layer_id is None
            else {layer_id: self.pending[layer_id]}
        )
        return [
            SimpleNamespace(
                request_id="long-2",
                layer_id=current_layer_id,
                logical_token_page=page,
            )
            for current_layer_id, pages in layer_ids.items()
            for page in sorted(pages)
        ]

    def pin_attention_page_versions(self, *args, **kwargs):
        return {}


def _runner_and_metadata():
    runner = object.__new__(GPUModelRunner)
    runner._virtual_context_enabled = True
    runner._pd_green_context_execute_phase = "decode"
    kv_cache_0 = torch.zeros((2, 2, 16, 1, 1))
    kv_cache_1 = torch.zeros_like(kv_cache_0)
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "layer.0": SimpleNamespace(kv_cache=[kv_cache_0]),
            "layer.1": SimpleNamespace(kv_cache=[kv_cache_1]),
        }
    )
    common = dict(
        request_ids=["long-2"],
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens=torch.tensor([16], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        tiered_block_table_cpu=None,
        seq_lens_cpu=None,
        query_start_loc_cpu=None,
    )
    return runner, {
        "layer.0": SimpleNamespace(**common),
        "layer.1": SimpleNamespace(**common),
    }


def test_decode_handoff_reloads_all_layers_before_admission(monkeypatch) -> None:
    runner, metadata = _runner_and_metadata()
    virtual_context = _FakeVirtualContext()
    monkeypatch.setattr(
        runner_module, "get_virtual_context_table", lambda: virtual_context
    )
    monkeypatch.setattr(
        runner_module,
        "virtual_context_layer_id",
        lambda layer_name: int(layer_name.rsplit(".", 1)[1]),
    )
    call_order = []
    monkeypatch.setattr(
        runner_module,
        "flush_tiered_context_ring_source_lease_releases",
        lambda request_ids: call_order.append("ring_flush")
        or {
            "executor_count": 1,
            "waited_events": 1,
            "released_leases": 1,
            "pending_leases": 0,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "wait_for_pending_hbm_d2h_request_reads",
        lambda request_ids: call_order.append("d2h_wait") or 1,
    )

    reloaded_layers = []

    def reload(**kwargs):
        call_order.append("h2d_reload")
        layer_id = int(kwargs["layer_name"].rsplit(".", 1)[1])
        reloaded_layers.append(layer_id)
        virtual_context.pending[layer_id].clear()
        return {"long-2": {0}}

    monkeypatch.setattr(runner_module, "reload_dram_kv_pages_to_hbm", reload)

    metrics = runner._complete_virtual_context_decode_handoff(metadata)

    assert reloaded_layers == [0, 1]
    assert call_order == [
        "ring_flush",
        "d2h_wait",
        "h2d_reload",
        "h2d_reload",
    ]
    assert metrics["worker_decode_handoff_barrier_requested"] == 1
    assert metrics["worker_decode_handoff_ring_flush_released_leases"] == 1
    assert metrics["worker_decode_handoff_barrier_remaining_pages"] == 0


def test_decode_handoff_never_enters_model_with_incomplete_layer(monkeypatch) -> None:
    runner, metadata = _runner_and_metadata()
    virtual_context = _FakeVirtualContext()
    monkeypatch.setattr(
        runner_module, "get_virtual_context_table", lambda: virtual_context
    )
    monkeypatch.setattr(
        runner_module,
        "virtual_context_layer_id",
        lambda layer_name: int(layer_name.rsplit(".", 1)[1]),
    )
    monkeypatch.setattr(
        runner_module,
        "flush_tiered_context_ring_source_lease_releases",
        lambda request_ids: {
            "executor_count": 0,
            "waited_events": 0,
            "released_leases": 0,
            "pending_leases": 0,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "wait_for_pending_hbm_d2h_request_reads",
        lambda request_ids: 0,
    )
    monkeypatch.setattr(
        runner_module,
        "reload_dram_kv_pages_to_hbm",
        lambda **kwargs: {},
    )

    with pytest.raises(RuntimeError, match="blocked before model forward"):
        runner._complete_virtual_context_decode_handoff(metadata)


class _FakeEvent:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


class _FakeLease:
    def __init__(self) -> None:
        self.release_count = 0

    def release(self) -> None:
        self.release_count += 1


def test_context_ring_source_lease_flush_is_request_scoped() -> None:
    executor = TieredBlockwiseAttentionExecutor(staging_slots=1)
    first_event = _FakeEvent()
    second_event = _FakeEvent()
    first_lease = _FakeLease()
    second_lease = _FakeLease()
    executor._pending_source_lease_releases.extend([
        SimpleNamespace(
            event=first_event,
            leases=(first_lease,),
            request_ids=frozenset({"long-2"}),
        ),
        SimpleNamespace(
            event=second_event,
            leases=(second_lease,),
            request_ids=frozenset({"long-3"}),
        ),
    ])

    result = executor.flush_source_lease_releases({"long-2"})

    assert result == {
        "waited_events": 1,
        "released_leases": 1,
        "pending_leases": 0,
    }
    assert first_event.synchronize_count == 1
    assert first_lease.release_count == 1
    assert second_event.synchronize_count == 0
    assert second_lease.release_count == 0


def test_native_handoff_does_not_treat_padding_as_null_block(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER", "1")
    monkeypatch.setenv("VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM", "1")
    clear_dram_kv_pages()
    try:
        source_key = torch.arange(16, dtype=torch.float32).reshape(16, 1, 1)
        source_value = source_key + 100
        register_dram_kv_page(
            request_id="long-2",
            layer_name="layer.padding-collision",
            logical_page=0,
            key=source_key,
            value=source_value,
            token_start=0,
            version=1,
        )
        key_cache = torch.zeros((2, 16, 1, 1), dtype=torch.float32)
        value_cache = torch.zeros_like(key_cache)
        diagnostics: dict[str, int | str] = {}

        reloaded = reload_dram_kv_pages_to_hbm(
            layer_name="layer.padding-collision",
            key_cache=key_cache,
            value_cache=value_cache,
            # The unused padding cell equals valid physical block 0. It is not
            # an allocator null sentinel and must not reject native handoff.
            block_table=torch.tensor([[0, 1, 0]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            request_ids=["long-2"],
            block_size=16,
            page_versions_by_request={"long-2": {0: 1}},
            reload_pages_by_request={"long-2": {0}},
            block_table_is_fully_concrete=True,
            diagnostics=diagnostics,
        )

        assert reloaded == {"long-2": {0}}
        assert diagnostics["reason"] == "complete"
        assert diagnostics["invalid_block_table_pages"] == 0
        torch.testing.assert_close(key_cache[0], source_key)
        torch.testing.assert_close(value_cache[0], source_value)
    finally:
        clear_dram_kv_pages()


def test_prefill_tail_reload_uses_frozen_provider_prefix(
    monkeypatch,
) -> None:
    """Post-query CPU metadata must not enlarge a partial DRAM snapshot."""

    monkeypatch.setenv("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER", "1")
    monkeypatch.setenv("VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM", "1")
    clear_dram_kv_pages()
    try:
        source_key = torch.arange(11, dtype=torch.float32).reshape(11, 1, 1)
        source_value = source_key + 100
        register_dram_kv_page(
            request_id="growing-tail",
            layer_name="layer.growing-tail",
            logical_page=0,
            key=source_key,
            value=source_value,
            token_start=0,
            version=7,
        )
        # Tokens 11..15 represent the current Prefill query, which has already
        # been written into HBM by reshape-and-cache.
        key_cache = torch.full((2, 16, 1, 1), 777.0)
        value_cache = torch.full_like(key_cache, 888.0)
        diagnostics: dict[str, int | str] = {}

        reloaded = reload_dram_kv_pages_to_hbm(
            layer_name="layer.growing-tail",
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([16], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
            request_ids=["growing-tail"],
            block_size=16,
            page_versions_by_request={"growing-tail": {0: 7}},
            reload_pages_by_request={"growing-tail": {0}},
            expected_token_counts_by_request={"growing-tail": {0: 11}},
            hbm_token_counts_by_request={"growing-tail": {0: 16}},
            # Simulate the stale/post-query CPU mirror that previously made
            # the provider lookup demand all 16 tokens from an 11-token page.
            query_start_loc_cpu=torch.tensor([0, 0], dtype=torch.int32),
            diagnostics=diagnostics,
        )

        assert reloaded == {"growing-tail": {0}}
        assert diagnostics["reason"] == "complete"
        assert diagnostics["explicit_token_count_pages"] == 1
        torch.testing.assert_close(key_cache[0, :11], source_key)
        torch.testing.assert_close(value_cache[0, :11], source_value)
        torch.testing.assert_close(
            key_cache[0, 11:], torch.full((5, 1, 1), 777.0)
        )
        torch.testing.assert_close(
            value_cache[0, 11:], torch.full((5, 1, 1), 888.0)
        )
    finally:
        clear_dram_kv_pages()
