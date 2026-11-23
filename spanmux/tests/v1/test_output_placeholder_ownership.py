# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.output_placeholder import (
    OutputPlaceholderLedger,
    OutputPlaceholderOwnershipError,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _FakeCudaEvent:
    def __init__(self, *, ready: bool) -> None:
        self.ready = ready
        self.query_count = 0
        self.synchronize_count = 0
        self.record_count = 0

    def query(self) -> bool:
        self.query_count += 1
        return self.ready

    def synchronize(self) -> None:
        self.synchronize_count += 1
        self.ready = True

    def record(self) -> None:
        self.record_count += 1
        self.ready = False


def _scheduler_output(
    *,
    phase: str,
    owners: dict[str, tuple[int, int, str, int]],
    spec_tokens: dict[str, list[int]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pd_output_ownership_enabled=True,
        pd_output_placeholder_owners=owners,
        scheduled_phase=phase,
        scheduled_spec_decode_tokens=spec_tokens or {},
    )


def test_output_placeholder_ledger_commits_exact_generation() -> None:
    ledger = OutputPlaceholderLedger()
    owner = ledger.reserve(phase="prefill", placeholder_count=1)

    assert owner.generation == 1
    assert ledger.resolve(
        request_epoch=owner.request_epoch,
        generation=owner.generation,
        phase="prefill",
        generated_count=1,
    )
    assert ledger.active is None
    assert ledger.last_committed_generation == 1
    assert ledger.last_committed_phase == "prefill"

    next_owner = ledger.reserve(phase="decode", placeholder_count=1)
    assert next_owner.generation == 2


def test_missing_owner_output_is_fatal_instead_of_repaired() -> None:
    scheduler = object.__new__(Scheduler)
    ledger = OutputPlaceholderLedger()
    owner = ledger.reserve(phase="prefill", placeholder_count=1)
    request = SimpleNamespace(
        request_id="short-1",
        pd_output_placeholder_ledger=ledger,
        discard_latest_async_tokens=False,
    )
    output = _scheduler_output(
        phase="prefill",
        owners={
            "short-1": (
                owner.request_epoch,
                owner.generation,
                "prefill",
                1,
            )
        },
    )

    with pytest.raises(
        OutputPlaceholderOwnershipError,
        match="returned no sampled output",
    ):
        scheduler._resolve_pd_output_placeholder_ownership(
            request,
            output,
            [],
        )

    # There is deliberately no token-progress rollback/retry repair.
    assert ledger.last_committed_generation == 0


def test_spec_decode_resolves_generated_and_rejected_positions() -> None:
    scheduler = object.__new__(Scheduler)
    ledger = OutputPlaceholderLedger()
    owner = ledger.reserve(phase="decode", placeholder_count=4)
    request = SimpleNamespace(
        request_id="long-1",
        pd_output_placeholder_ledger=ledger,
        discard_latest_async_tokens=False,
    )
    output = _scheduler_output(
        phase="decode",
        owners={
            "long-1": (
                owner.request_epoch,
                owner.generation,
                "decode",
                4,
            )
        },
        spec_tokens={"long-1": [101, 102, 103]},
    )

    # One normal sample + one accepted draft leaves two rejected positions.
    assert scheduler._resolve_pd_output_placeholder_ownership(
        request,
        output,
        [201, 202],
    )
    assert ledger.last_committed_generation == owner.generation


def test_cancelled_owner_discards_only_its_matching_result() -> None:
    scheduler = object.__new__(Scheduler)
    ledger = OutputPlaceholderLedger()
    owner = ledger.reserve(phase="decode", placeholder_count=1)
    ledger.cancel_active()
    request = SimpleNamespace(
        request_id="reset-request",
        pd_output_placeholder_ledger=ledger,
        discard_latest_async_tokens=True,
    )
    output = _scheduler_output(
        phase="decode",
        owners={
            "reset-request": (
                owner.request_epoch,
                owner.generation,
                "decode",
                1,
            )
        },
    )

    assert not scheduler._resolve_pd_output_placeholder_ownership(
        request,
        output,
        [301],
    )
    assert ledger.active is None
    assert ledger.last_committed_generation == 0
    assert not request.discard_latest_async_tokens


def test_stale_generation_cannot_overwrite_newer_owner() -> None:
    scheduler = object.__new__(Scheduler)
    ledger = OutputPlaceholderLedger()
    first_owner = ledger.reserve(phase="prefill", placeholder_count=1)
    assert ledger.resolve(
        request_epoch=first_owner.request_epoch,
        generation=first_owner.generation,
        phase="prefill",
        generated_count=1,
    )
    second_owner = ledger.reserve(phase="decode", placeholder_count=1)
    request = SimpleNamespace(
        request_id="short-2",
        pd_output_placeholder_ledger=ledger,
        discard_latest_async_tokens=False,
    )
    stale_output = _scheduler_output(
        phase="prefill",
        owners={
            "short-2": (
                first_owner.request_epoch,
                first_owner.generation,
                "prefill",
                1,
            )
        },
    )

    with pytest.raises(OutputPlaceholderOwnershipError):
        scheduler._resolve_pd_output_placeholder_ownership(
            request,
            stale_output,
            [401],
        )
    assert ledger.active == second_owner


def test_scheduler_attaches_final_prefill_and_decode_generations() -> None:
    scheduler = object.__new__(Scheduler)
    scheduler.pd_output_placeholder_ownership_enabled = True
    scheduler._pd_output_last_execution_key_by_req = {}
    request = SimpleNamespace(
        request_id="short-3",
        sampling_params=object(),
        pooling_params=None,
        num_output_tokens=0,
        num_output_placeholders=1,
        all_token_ids=[11, 12],
        pd_output_placeholder_ledger=OutputPlaceholderLedger(),
    )
    scheduler.requests = {request.request_id: request}
    prefill_output = SimpleNamespace(
        pd_output_ownership_enabled=False,
        pd_output_placeholder_owners={},
        pd_output_placeholder_expected_counts={},
        pd_output_placeholder_counter_deltas={"short-3": 1},
        pd_output_token_source_generations={},
        scheduled_cached_reqs=SimpleNamespace(all_token_ids={}),
        total_num_scheduled_tokens=128,
        scheduled_phase="prefill",
        pd_ttft_handoff_request_ids=(request.request_id,),
        num_scheduled_tokens={request.request_id: 128},
        scheduled_spec_decode_tokens={},
    )

    scheduler._attach_pd_output_placeholder_ownership(prefill_output)

    request_epoch, generation, phase, count = (
        prefill_output.pd_output_placeholder_owners[request.request_id]
    )
    assert request_epoch == request.pd_output_placeholder_ledger.request_epoch
    assert (generation, phase, count) == (1, "prefill", 1)
    assert request.pd_output_placeholder_ledger.resolve(
        request_epoch=request_epoch,
        generation=generation,
        phase=phase,
        generated_count=1,
    )

    request.num_output_tokens = 1
    request.num_output_placeholders = 1
    request.all_token_ids.append(201)
    decode_output = SimpleNamespace(
        pd_output_ownership_enabled=False,
        pd_output_placeholder_owners={},
        pd_output_placeholder_expected_counts={},
        pd_output_placeholder_counter_deltas={"short-3": 1},
        pd_output_token_source_generations={},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[request.request_id],
            all_token_ids={},
        ),
        total_num_scheduled_tokens=1,
        scheduled_phase="decode",
        pd_ttft_handoff_request_ids=(),
        num_scheduled_tokens={request.request_id: 1},
        scheduled_spec_decode_tokens={},
    )

    scheduler._attach_pd_output_placeholder_ownership(decode_output)
    scheduler._pd_output_last_execution_key_by_req[request.request_id] = (
        "decode:1"
    )
    scheduler.bind_pd_output_execution_context(decode_output, "decode")

    assert decode_output.pd_output_token_source_generations == {
        request.request_id: 1
    }
    assert decode_output.scheduled_cached_reqs.all_token_ids == {
        request.request_id: [11, 12, 201]
    }
    assert decode_output.pd_output_placeholder_owners[request.request_id] == (
        request.pd_output_placeholder_ledger.request_epoch,
        2,
        "decode",
        1,
    )


def test_reused_request_id_cannot_consume_an_older_request_epoch() -> None:
    scheduler = object.__new__(Scheduler)
    old_ledger = OutputPlaceholderLedger()
    old_owner = old_ledger.reserve(phase="decode", placeholder_count=1)
    new_ledger = OutputPlaceholderLedger()
    new_owner = new_ledger.reserve(phase="decode", placeholder_count=1)
    request = SimpleNamespace(
        request_id="reused-id",
        pd_output_placeholder_ledger=new_ledger,
        discard_latest_async_tokens=False,
    )
    stale_output = _scheduler_output(
        phase="decode",
        owners={
            "reused-id": (
                old_owner.request_epoch,
                old_owner.generation,
                "decode",
                1,
            )
        },
    )

    with pytest.raises(
        OutputPlaceholderOwnershipError,
        match="request ledger",
    ):
        scheduler._resolve_pd_output_placeholder_ownership(
            request,
            stale_output,
            [501],
        )
    assert new_ledger.active == new_owner


def test_preemption_rejects_execution_owned_output() -> None:
    scheduler = object.__new__(Scheduler)
    ledger = OutputPlaceholderLedger()
    ledger.reserve(phase="decode", placeholder_count=1)
    request = SimpleNamespace(
        request_id="long-2",
        status=RequestStatus.RUNNING,
        pd_output_placeholder_ledger=ledger,
    )

    with pytest.raises(
        OutputPlaceholderOwnershipError,
        match="Cannot preempt",
    ):
        scheduler._preempt_request(request, 1.0)


def test_external_residency_resume_preserves_logical_progress() -> None:
    request = object.__new__(Request)
    request.request_id = "long-tiered"
    request._all_token_ids = list(range(12_000))
    request.num_computed_tokens = 10_000
    request.spec_token_ids = []
    request.pd_output_placeholder_ledger = OutputPlaceholderLedger()
    request.virtual_context_deferred_logical_computed_tokens = 0
    request.virtual_context_deferred_state = ""
    request.virtual_context_deferred_last_iteration_id = -1
    request.virtual_context_deferred_last_offloaded_tokens = 0

    request.suspend_to_external_residency(
        logical_computed_tokens=10_000,
        deferred_state="deferred_tiered_provider",
        scheduler_iteration_id=42,
    )

    assert request.num_computed_tokens == 0
    assert request.virtual_context_deferred_logical_computed_tokens == 10_000
    request.resume_from_external_residency(logical_computed_tokens=10_000)
    assert request.num_computed_tokens == 10_000


def test_external_residency_suspend_rejects_active_output_owner() -> None:
    request = object.__new__(Request)
    request.request_id = "short-owned"
    request._all_token_ids = list(range(10))
    request.num_computed_tokens = 9
    request.spec_token_ids = []
    request.pd_output_placeholder_ledger = OutputPlaceholderLedger()
    request.pd_output_placeholder_ledger.reserve(
        phase="prefill", placeholder_count=1
    )

    with pytest.raises(OutputPlaceholderOwnershipError, match="Cannot suspend"):
        request.suspend_to_external_residency(
            logical_computed_tokens=9,
            deferred_state="deferred_tiered_provider",
            scheduler_iteration_id=43,
        )


def test_external_residency_preflight_is_mutation_free() -> None:
    request = object.__new__(Request)
    request.request_id = "invalid-tiered"
    request._all_token_ids = list(range(10))
    request.num_computed_tokens = 9
    request.spec_token_ids = [99]
    request.pd_output_placeholder_ledger = OutputPlaceholderLedger()
    request.virtual_context_deferred_logical_computed_tokens = 0
    request.virtual_context_deferred_state = ""

    with pytest.raises(ValueError, match="Invalid external-residency"):
        request.validate_external_residency_suspension(
            logical_computed_tokens=11,
            deferred_state="deferred_tiered_provider",
        )

    assert request.num_computed_tokens == 9
    assert request.spec_token_ids == [99]
    assert request.virtual_context_deferred_logical_computed_tokens == 0
    assert request.virtual_context_deferred_state == ""


def test_final_prefill_owner_blocks_without_progress_rollback() -> None:
    scheduler = object.__new__(Scheduler)
    scheduler._pd_pending_schedule_block_count = 0
    scheduler._pd_pending_schedule_last_reason = ""
    scheduler._pd_pending_request_ids = set()
    scheduler._pd_pending_phase_request_ids = {}
    ledger = OutputPlaceholderLedger()
    ledger.reserve(phase="prefill", placeholder_count=1)
    request = SimpleNamespace(
        request_id="short-4",
        num_prompt_tokens=10_000,
        num_computed_tokens=10_000,
        pd_output_placeholder_ledger=ledger,
    )

    assert scheduler._debug_needs_prefill_lane(request)
    assert scheduler.blocks_next_schedule(request, "prefill")
    assert request.num_computed_tokens == request.num_prompt_tokens
    assert scheduler._pd_pending_schedule_last_reason == (
        "explicit_output_placeholder_owner"
    )


def test_worker_applies_only_committed_generation_in_place() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(vocab_size=1_000)
    runner._pd_phase_handoff_trace_jsonl = ""
    output_ids = [101]
    request_state = SimpleNamespace(
        output_token_ids=output_ids,
        output_token_generation=1,
        num_tokens=3,
    )
    request_data = SimpleNamespace(
        all_token_ids={"short-5": [11, 12, 201, 202]}
    )

    runner._apply_owned_output_token_refresh(
        request_data,
        "short-5",
        request_state,
        num_computed_tokens=4,
        num_output_tokens=2,
        source_generation=2,
    )

    assert request_state.output_token_ids is output_ids
    assert output_ids == [201, 202]
    assert request_state.output_token_generation == 2


def test_worker_never_falls_back_to_cached_tokens_without_history() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(vocab_size=1_000)
    runner._pd_phase_handoff_trace_jsonl = ""
    request_state = SimpleNamespace(
        output_token_ids=[201],
        output_token_generation=1,
        num_tokens=3,
    )
    request_data = SimpleNamespace(all_token_ids={})

    with pytest.raises(
        OutputPlaceholderOwnershipError,
        match="without its token history",
    ):
        runner._apply_owned_output_token_refresh(
            request_data,
            "short-6",
            request_state,
            num_computed_tokens=3,
            num_output_tokens=1,
            source_generation=2,
        )


def test_async_placeholder_counter_matches_future_reservation() -> None:
    scheduler = object.__new__(AsyncScheduler)
    request = SimpleNamespace(
        num_computed_tokens=9,
        num_tokens=10,
        num_output_placeholders=0,
        has_encoder_inputs=False,
        use_structured_output=False,
        spec_token_ids=[],
    )
    scheduler.requests = {"short-7": request}
    scheduler.num_spec_tokens = 0
    scheduler.finished_req_ids = set()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"short-7": 1},
        scheduled_spec_decode_tokens={},
        pd_output_placeholder_expected_counts={"short-7": 1},
        pd_output_placeholder_counter_deltas={},
    )

    scheduler._update_after_schedule(scheduler_output)

    assert request.num_computed_tokens == 10
    assert request.num_output_placeholders == 1
    assert scheduler_output.pd_output_placeholder_counter_deltas == {
        "short-7": 1
    }
    assert not scheduler_output.has_structured_output_requests
    assert not scheduler_output.pending_structured_output_tokens


def test_explicit_output_claim_survives_restored_progress_mismatch() -> None:
    scheduler = object.__new__(AsyncScheduler)
    request = SimpleNamespace(
        # A tiered/preempted request can restore durable progress before the
        # upstream async equality predicate becomes true.
        num_computed_tokens=10_000,
        num_tokens=10_005,
        num_output_placeholders=0,
        has_encoder_inputs=False,
        use_structured_output=False,
        spec_token_ids=[],
    )
    scheduler.requests = {"short-5": request}
    scheduler.num_spec_tokens = 0
    scheduler.finished_req_ids = set()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"short-5": 1},
        scheduled_spec_decode_tokens={},
        pd_output_placeholder_expected_counts={"short-5": 1},
        pd_output_placeholder_counter_deltas={},
    )

    scheduler._update_after_schedule(scheduler_output)

    assert request.num_computed_tokens == 10_001
    assert request.num_output_placeholders == 1
    assert scheduler_output.pd_output_placeholder_counter_deltas == {
        "short-5": 1
    }


def test_input_prep_event_skips_completed_cpu_synchronize() -> None:
    runner = object.__new__(GPUModelRunner)
    event = _FakeCudaEvent(ready=True)
    runner.prepare_inputs_event = event
    runner.prepare_inputs_event_recorded = True
    runner.prepare_inputs_event_fast_path_count = 0
    runner.prepare_inputs_event_wait_count = 0

    with runner.synchronize_input_prep() as waited:
        assert not waited

    assert event.query_count == 1
    assert event.synchronize_count == 0
    assert event.record_count == 1
    assert runner.prepare_inputs_event_fast_path_count == 1


def test_input_prep_event_waits_when_reuse_is_not_ready() -> None:
    runner = object.__new__(GPUModelRunner)
    event = _FakeCudaEvent(ready=False)
    runner.prepare_inputs_event = event
    runner.prepare_inputs_event_recorded = True
    runner.prepare_inputs_event_fast_path_count = 0
    runner.prepare_inputs_event_wait_count = 0

    with runner.synchronize_input_prep() as waited:
        assert waited

    assert event.query_count == 1
    assert event.synchronize_count == 1
    assert event.record_count == 1
    assert runner.prepare_inputs_event_wait_count == 1
