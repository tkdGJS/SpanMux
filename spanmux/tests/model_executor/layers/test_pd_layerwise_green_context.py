# SPDX-License-Identifier: Apache-2.0

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

from vllm.model_executor.layers import green_context
from vllm.model_executor.layers.pd_layerwise_handoff import (
    PD_LAYERWISE_HANDOFF_CONTROLLER,
)


class _FakeStream:
    def __init__(self) -> None:
        self.waited_events: list["_FakeEvent"] = []

    def wait_event(self, event) -> None:
        self.waited_events.append(event)


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded_stream = None
        self.synchronize_count = 0

    def record(self, stream) -> None:
        self.recorded_stream = stream

    def synchronize(self) -> None:
        self.synchronize_count += 1


def test_prefill_handoff_applies_only_after_completed_layer(monkeypatch) -> None:
    source_stream = _FakeStream()
    target_stream = _FakeStream()
    target_decode_stream = _FakeStream()
    selected_streams: list[_FakeStream] = []
    boundary_events: list[_FakeEvent] = []
    workspace_handoffs: list[dict[str, object]] = []

    monkeypatch.setenv("VLLM_PD_LAYERWISE_PROFILE_HANDOFF", "1")
    monkeypatch.setattr(
        green_context,
        "green_context_dynamic_sm_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        green_context,
        "_current_torch_stream",
        lambda: source_stream,
    )
    monkeypatch.setattr(
        green_context.torch.cuda,
        "set_stream",
        selected_streams.append,
    )
    monkeypatch.setattr(
        green_context.torch.cuda,
        "Event",
        lambda: boundary_events.append(_FakeEvent()) or boundary_events[-1],
    )
    monkeypatch.setattr(
        green_context,
        "_get_precreated_layerwise_target_streams",
        lambda profile_id: SimpleNamespace(
            stream_for_phase=lambda phase: (
                target_stream if phase == 1 else target_decode_stream
            )
        ),
    )
    monkeypatch.setattr(
        green_context,
        "handoff_active_prefill_workspace",
        lambda **kwargs: workspace_handoffs.append(kwargs),
    )

    PD_LAYERWISE_HANDOFF_CONTROLLER.reset()
    generation = PD_LAYERWISE_HANDOFF_CONTROLLER.arm(
        source_profile_id="d30_p10",
        source_epoch=4,
    )
    with green_context.green_context_layerwise_prefill_handoff(
        "prefill",
        active=True,
        profile_id="d30_p10",
        profile_epoch=4,
    ) as execution:
        green_context.pd_layerwise_handoff_layer_boundary(5)
        # The first queued layer remains asynchronous.  The fixed two-layer
        # window has not reached its bound yet.
        assert boundary_events[0].recorded_stream is source_stream
        assert selected_streams == []

        assert PD_LAYERWISE_HANDOFF_CONTROLLER.request(
            source_profile_id="d30_p10",
            source_epoch=4,
            target_profile_id="d0_p40",
            target_epoch=5,
        )
        def acknowledge_completed_boundary() -> bool:
            for _ in range(1000):
                snapshot = PD_LAYERWISE_HANDOFF_CONTROLLER.snapshot()
                if snapshot.get(
                    "pd_layerwise_handoff_boundary_completed", 0.0
                ):
                    break
                time.sleep(0.001)
            else:
                raise AssertionError("layer boundary did not complete")

            # A physical boundary ACK alone cannot expose the target profile.
            # EngineCore must first commit the bank/allocator epoch and release
            # the generation-owned resume event.
            assert selected_streams == []
            return PD_LAYERWISE_HANDOFF_CONTROLLER.acknowledge_completed_transition(
                source_profile_id="d30_p10",
                source_epoch=4,
                target_profile_id="d0_p40",
                target_epoch=5,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            acknowledgement = executor.submit(acknowledge_completed_boundary)
            # Context-local model execution must stay on this thread.  The
            # helper represents EngineCore, which observes and commits the ACK.
            green_context.pd_layerwise_handoff_layer_boundary(6)
            assert acknowledgement.result(timeout=1.0)
        assert selected_streams == [target_stream]
        assert execution.generation == generation
        assert execution.applied
        assert execution.applied_layer == 6
        assert execution.effective_profile_id == "d0_p40"
        assert execution.effective_profile_epoch == 5
        assert target_stream.waited_events == [boundary_events[1]]
        assert target_decode_stream.waited_events == []
        assert workspace_handoffs == [
            {
                "handoff_generation": generation,
                "source_profile_id": "d30_p10",
                "source_profile_epoch": 4,
                "target_profile_id": "d0_p40",
                "target_profile_epoch": 5,
            }
        ]
        # After EngineCore acknowledges the switch, later callbacks remain
        # bounded so a reverse transition can interrupt the same future.
        green_context.pd_layerwise_handoff_layer_boundary(7)
        assert len(boundary_events) == 3
        assert selected_streams == [target_stream]

    snapshot = PD_LAYERWISE_HANDOFF_CONTROLLER.snapshot()
    assert snapshot["pd_layerwise_handoff_armed"] == 0.0
    assert snapshot["pd_layerwise_handoff_apply_count"] == 1.0
    assert snapshot["pd_layerwise_handoff_commit_count"] == 1.0
    PD_LAYERWISE_HANDOFF_CONTROLLER.reset()


def test_completed_layerwise_bank_commit_skips_handle_synchronize() -> None:
    class _FakeLibrary:
        def __init__(self) -> None:
            self.synchronize_count = 0

        def synchronize(self, _handle) -> None:
            self.synchronize_count += 1

    library = _FakeLibrary()
    source_streams = SimpleNamespace(
        profile_id="d30_p10",
        library=library,
        handle=object(),
    )
    target_streams = SimpleNamespace(
        profile_id="d0_p40",
        library=library,
        handle=object(),
    )
    bank = object.__new__(green_context._GreenContextProfileBank)
    bank._lock = threading.RLock()
    bank._streams = {
        "d30_p10": source_streams,
        "d0_p40": target_streams,
    }
    bank._failed_profiles = {}
    bank._active_profile_id = "d30_p10"
    bank._active_epoch = 4
    bank._handoff_count = 0
    bank._handoff_sync_ms_total = 0.0
    bank._last_handoff_sync_ms = 0.0
    bank._layerwise_commit_count = 0

    PD_LAYERWISE_HANDOFF_CONTROLLER.reset()
    generation = PD_LAYERWISE_HANDOFF_CONTROLLER.arm(
        source_profile_id="d30_p10",
        source_epoch=4,
    )
    assert PD_LAYERWISE_HANDOFF_CONTROLLER.begin_execution(
        source_profile_id="d30_p10",
        source_epoch=4,
    ) == generation
    assert PD_LAYERWISE_HANDOFF_CONTROLLER.request(
        source_profile_id="d30_p10",
        source_epoch=4,
        target_profile_id="d0_p40",
        target_epoch=5,
    )
    assert PD_LAYERWISE_HANDOFF_CONTROLLER.mark_boundary_recorded(
        generation,
        layer_index=6,
    )
    assert PD_LAYERWISE_HANDOFF_CONTROLLER.mark_boundary_completed(
        generation,
        layer_index=6,
        gpu_wait_ms=1.5,
    )

    committed = bank.commit_completed_layerwise_boundary(
        source_profile_id="d30_p10",
        source_epoch=4,
        target_profile_id="d0_p40",
        target_epoch=5,
    )

    assert committed is target_streams
    assert bank.active_profile_id == "d0_p40"
    assert bank.active_epoch == 5
    assert bank.layerwise_commit_count == 1
    assert bank.last_handoff_sync_ms == 0.0
    assert library.synchronize_count == 0
    assert PD_LAYERWISE_HANDOFF_CONTROLLER.acknowledge_completed_transition(
        source_profile_id="d30_p10",
        source_epoch=4,
        target_profile_id="d0_p40",
        target_epoch=5,
    )
    PD_LAYERWISE_HANDOFF_CONTROLLER.complete_execution(generation)
    PD_LAYERWISE_HANDOFF_CONTROLLER.reset()


def test_prefill_submission_window_waits_only_for_oldest_event(
    monkeypatch,
) -> None:
    source_stream = _FakeStream()
    boundary_events: list[_FakeEvent] = []

    monkeypatch.setenv("VLLM_PD_LAYERWISE_PROFILE_HANDOFF", "1")
    monkeypatch.setattr(
        green_context,
        "green_context_dynamic_sm_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        green_context,
        "_current_torch_stream",
        lambda: source_stream,
    )
    monkeypatch.setattr(
        green_context.torch.cuda,
        "Event",
        lambda: boundary_events.append(_FakeEvent()) or boundary_events[-1],
    )

    def completed_boundary_wait(event, wake_event):
        event.synchronize()
        future: Future[None] = Future()
        future.set_result(None)
        wake_event.set()
        return future

    monkeypatch.setattr(
        green_context,
        "_submit_layer_boundary_wait",
        completed_boundary_wait,
    )

    PD_LAYERWISE_HANDOFF_CONTROLLER.reset()
    PD_LAYERWISE_HANDOFF_CONTROLLER.arm(
        source_profile_id="d30_p10",
        source_epoch=4,
    )
    with green_context.green_context_layerwise_prefill_handoff(
        "prefill",
        active=True,
        profile_id="d30_p10",
        profile_epoch=4,
    ):
        green_context.pd_layerwise_handoff_layer_boundary(5)
        assert boundary_events[0].synchronize_count == 0

        green_context.pd_layerwise_handoff_layer_boundary(6)
        assert boundary_events[0].synchronize_count == 1
        assert boundary_events[1].synchronize_count == 0

        green_context.pd_layerwise_handoff_layer_boundary(7)
        assert boundary_events[1].synchronize_count == 1
        assert boundary_events[2].synchronize_count == 0

    PD_LAYERWISE_HANDOFF_CONTROLLER.reset()
