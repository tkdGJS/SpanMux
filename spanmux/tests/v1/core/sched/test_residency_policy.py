# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.prefill_chunk_scheduler import (
    PrefillChunkSchedulingPolicy,
)
from vllm.v1.core.sched.residency_policy import (
    ProjectedPrefillCandidate,
    ResidencyVictimCandidate,
    ResidencyVictimPolicy,
    projected_srpt_prefill_order,
    scheduled_prefill_capacity_headroom_blocks,
    select_best_fit_whole_request_victims,
)
from vllm.v1.core.sched.scheduler import (
    ScheduledPrefillCapacityClaim,
    ScheduledPrefillFitDecodeGrowthRunway,
    ScheduledPrefillFitPreparation,
    Scheduler,
    pd_prefill_no_progress_retry_metadata,
)
from vllm.v1.core.tiered_errors import (
    TieredAllocatorInvariantError,
    TieredBackendEligibilityError,
)
from vllm.v1.output_placeholder import OutputPlaceholderLedger
from vllm.v1.request import RequestStatus


def _request(
    request_id: str,
    *,
    prompt_tokens: int,
    computed_tokens: int,
    arrival_time: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt_tokens,
        num_computed_tokens=computed_tokens,
        virtual_context_deferred_logical_computed_tokens=0,
        num_external_computed_tokens=0,
        virtual_context_hbm_reloaded_tokens=0,
        virtual_context_hbm_reload_planned_tokens=0,
        virtual_context_deferred_state="",
        kv_retention_decode_continuation_lease_active=False,
        kv_retention_decode_continuation_waiting=False,
        kv_retention_decode_continuation_wait_count=0,
        kv_retention_decode_continuation_last_wait_iteration_id=-1,
        kv_retention_execution_plan_generation=0,
        kv_retention_execution_plan_acked_generation=0,
        kv_retention_execution_plan_mode="auto",
        kv_retention_execution_plan_expected_reload_pages=0,
        arrival_time=arrival_time,
        has_encoder_inputs=False,
        pd_output_placeholder_ledger=OutputPlaceholderLedger(),
    )


class _FakeKVCacheManager:
    def __init__(
        self,
        resident_blocks: dict[str, int],
        free_blocks: int,
        required_blocks: int = 128,
    ) -> None:
        self.resident_blocks = resident_blocks
        self.required_blocks = required_blocks
        self.reload_pages: dict[str, int] = {}
        self.reload_plan_calls = 0
        self.allocated_reload_requests: set[str] = set()
        self.block_pool = SimpleNamespace(
            get_num_free_blocks=lambda: free_blocks
        )

    def estimate_num_blocks_to_allocate(self, *args, **kwargs) -> int:
        del args, kwargs
        return self.required_blocks

    def get_virtual_context_hbm_reload_required_page_count(
        self,
        request_ids: set[str],
    ) -> int:
        return sum(self.reload_pages.get(request_id, 0) for request_id in request_ids)

    def get_request_resident_block_count(self, request_id: str) -> int:
        return self.resident_blocks.get(request_id, 0)

    def has_request_block_table(self, request_id: str) -> bool:
        return request_id in self.resident_blocks

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return (list(range(self.resident_blocks.get(request_id, 0))),)

    def plan_virtual_context_hbm_reload_pages(
        self,
        request_ids: set[str],
    ) -> list[SimpleNamespace]:
        self.reload_plan_calls += 1
        return [
            SimpleNamespace(
                request_id=request_id,
                kv_cache_group_id=0,
                logical_token_page=logical_page,
            )
            for request_id in sorted(request_ids)
            for logical_page in range(self.reload_pages.get(request_id, 0))
        ]

    def is_request_partial_residency(self, request_id: str) -> bool:
        return self.reload_pages.get(request_id, 0) > 0

    def resume_partial_residency(
        self,
        request_id: str,
        logical_computed_tokens: int,
    ) -> tuple[int, int]:
        assert logical_computed_tokens > 0
        virtual_blocks = (
            0
            if request_id in self.allocated_reload_requests
            else self.reload_pages.get(request_id, 0)
        )
        return self.resident_blocks.get(request_id, 0), virtual_blocks

    def get_retained_residency_counts(
        self,
        request_id: str,
        logical_computed_tokens: int,
    ) -> tuple[int, int]:
        return self.resume_partial_residency(
            request_id,
            logical_computed_tokens,
        )

    def allocate_virtual_context_hbm_reload_blocks(
        self,
        request_id: str,
        logical_pages_by_group: dict[int, set[int]],
    ) -> SimpleNamespace | None:
        required_blocks = sum(
            len(pages) for pages in logical_pages_by_group.values()
        )
        if required_blocks != self.reload_pages.get(request_id, 0):
            return None
        self.allocated_reload_requests.add(request_id)
        return SimpleNamespace(request_id=request_id)

    def plan_virtual_context_hbm_snapshot_pages(
        self,
        *,
        request_id: str,
        logical_tokens: int,
        max_pages: int,
        group_id: int,
    ) -> tuple[int, ...]:
        assert logical_tokens > 0
        assert group_id == 0
        return tuple(
            range(min(max_pages, self.resident_blocks.get(request_id, 0)))
        )


def _scheduler(
    requests: list[SimpleNamespace],
    *,
    resident_blocks: dict[str, int],
    free_blocks: int = 14,
) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.virtual_context_residency_victim_policy = (
        ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
    )
    scheduler.virtual_context_enabled = True
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.running = list(requests)
    scheduler.waiting = []
    scheduler._pd_pending_request_ids = set()
    scheduler._pd_pending_phase_request_ids = {}
    scheduler._virtual_context_waiting_decode_ready = lambda request: False
    scheduler.virtual_context_staging_reserved_blocks = 0
    scheduler.virtual_context_ring_min_free_blocks = 0
    scheduler.virtual_context_decode_hotset_reserve_blocks = 0
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 0
    scheduler.virtual_context_reload_to_hbm = True
    scheduler.virtual_context_reload_to_hbm_min_free_blocks = 0
    scheduler._scheduled_prefill_fit_decode_growth_runway = None
    scheduler.enable_ttft_prefill_chunk_scheduling = True
    scheduler.ttft_prefill_chunk_scheduler = SimpleNamespace(
        policy=PrefillChunkSchedulingPolicy.SRPT,
        candidate_order=tuple(request.request_id for request in requests),
    )
    scheduler._ttft_cap_selected_prefill_chunk = (
        lambda request, tokens, num_computed_tokens, execution_mode: tokens
    )
    scheduler.max_num_scheduled_tokens = 2048
    scheduler.num_lookahead_tokens = 0
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0
    )
    scheduler.virtual_context_decode_hotset_first = False
    scheduler.virtual_context_active_decode_elastic_prefill = False
    scheduler.virtual_context_native_decode_control_fast_path = False
    scheduler.virtual_context_prefill_ring_tokens_with_decode_cold = 0
    scheduler.is_encoder_decoder = False
    scheduler.block_size = 16
    scheduler._has_pending_prefill_for_ring_progress = lambda: False
    scheduler._native_decode_handoff_capacity_precheck_count = 0
    scheduler._native_decode_handoff_plan_avoided_page_count = 0
    scheduler._native_decode_handoff_reclaim_feasibility_check_count = 0
    scheduler._native_decode_handoff_reclaim_infeasible_count = 0
    scheduler._native_decode_handoff_reclaim_avoided_partial_blocks = 0
    scheduler._native_decode_handoff_reclaim_infeasible_cache_hit_count = 0
    scheduler._native_decode_handoff_reclaim_infeasible_cache = {}
    scheduler._tiered_execution_plan_next_generation = 0
    scheduler._scheduled_prefill_fit_preparation = None
    scheduler._scheduled_prefill_fit_preparation_reclaim_consumed_count = 0
    scheduler._scheduled_prefill_fit_preparation_consumed_victim_count = (
        0
    )
    scheduler._iteration_debug_index = 0
    scheduler.kv_cache_manager = _FakeKVCacheManager(
        resident_blocks,
        free_blocks,
    )
    return scheduler


def test_scheduled_prefill_fit_99_percent_capacity_headroom() -> None:
    assert scheduled_prefill_capacity_headroom_blocks(4095, 0.99) == 41
    assert scheduled_prefill_capacity_headroom_blocks(4095, 0.95) == 205
    assert scheduled_prefill_capacity_headroom_blocks(8191, 0.99) == 82
    assert scheduled_prefill_capacity_headroom_blocks(4095, 1.0) == 0


def test_safe_ring_capacity_reuses_scheduled_prefill_headroom() -> None:
    scheduler = _scheduler([], resident_blocks={}, free_blocks=1000)
    scheduler.virtual_context_ring_bw_safety_blocks = 10
    scheduler.virtual_context_ring_bw_use_scheduled_headroom = True
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 41
    scheduler.virtual_context_decode_hotset_reserve_blocks = 0
    scheduler.virtual_context_ring_bw_decode_growth_blocks = 0
    scheduler.virtual_context_staging_reserved_blocks = 0
    scheduler.virtual_context_ring_min_free_blocks = 0
    scheduler._virtual_context_active_decode_ring_progress_reserve = (
        lambda: (0, 0, 0)
    )
    scheduler._virtual_context_active_decode_for_ring_progress = lambda: False

    selected = scheduler._safe_capacity_virtual_context_ring_prefill_admission_tokens(
        2048,
        16384,
    )

    assert selected == 959 * 16
    assert scheduler._virtual_context_ring_bw_last_safety_blocks == 41


def test_active_decode_ring_cap_can_reuse_safe_elastic_capacity() -> None:
    request = _request(
        "ring-prefill",
        prompt_tokens=50_000,
        computed_tokens=16_384,
        arrival_time=1.0,
    )
    scheduler = _scheduler([request], resident_blocks={}, free_blocks=1000)
    scheduler.virtual_context_decode_hotset_first = True
    scheduler.virtual_context_active_decode_elastic_prefill = True
    scheduler.virtual_context_ring_bw_dynamic = True
    scheduler.virtual_context_ring_bw_policy = "safe_capacity"
    scheduler.virtual_context_active_decode_ring_prefill_tokens = 2_048
    scheduler.virtual_context_active_decode_native_prefill_tokens = 2_048
    scheduler._debug_running_decode_resident_blocks = lambda: 64
    scheduler._pd_pending_decode_active = lambda: True
    scheduler._virtual_context_reload_required_page_count = (
        lambda request_ids, phase: 128
    )

    selected = scheduler._cap_prefill_admission_with_active_decode(
        request,
        16_384,
    )

    assert selected == 16_384
    assert (
        scheduler._virtual_context_ring_bw_last_active_decode_cap_reason
        == "active_decode_ring_required:elastic_safe_capacity"
    )


def test_active_decode_ring_cap_is_legacy_2k_when_elastic_is_off() -> None:
    request = _request(
        "ring-prefill",
        prompt_tokens=50_000,
        computed_tokens=16_384,
        arrival_time=1.0,
    )
    scheduler = _scheduler([request], resident_blocks={}, free_blocks=1000)
    scheduler.virtual_context_decode_hotset_first = True
    scheduler.virtual_context_active_decode_elastic_prefill = False
    scheduler.virtual_context_ring_bw_dynamic = True
    scheduler.virtual_context_ring_bw_policy = "safe_capacity"
    scheduler.virtual_context_active_decode_ring_prefill_tokens = 2_048
    scheduler.virtual_context_active_decode_native_prefill_tokens = 2_048
    scheduler._debug_running_decode_resident_blocks = lambda: 64
    scheduler._pd_pending_decode_active = lambda: True
    scheduler._virtual_context_reload_required_page_count = (
        lambda request_ids, phase: 128
    )

    selected = scheduler._cap_prefill_admission_with_active_decode(
        request,
        16_384,
    )

    assert selected == 2_048


def test_prefill_no_progress_retry_waits_for_decode_output() -> None:
    assert pd_prefill_no_progress_retry_metadata(
        phase_pure=True,
        selected_phase="prefill",
        total_num_scheduled_tokens=0,
        has_logical_prefill_work=True,
        decode_output_pending=True,
        asynchronous_progress_pending=False,
        observed_free_blocks=125,
        defer_reasons={"policy", "decode_hbm_reload_wait"},
    ) == (
        126,
        "prefill_no_progress:decode_hbm_reload_wait,policy",
    )


def test_prefill_no_progress_retry_excludes_async_preparation() -> None:
    assert pd_prefill_no_progress_retry_metadata(
        phase_pure=True,
        selected_phase="prefill",
        total_num_scheduled_tokens=0,
        has_logical_prefill_work=True,
        decode_output_pending=True,
        asynchronous_progress_pending=True,
        observed_free_blocks=125,
        defer_reasons={"scheduled_prefill_fit_d2h_wait"},
    ) is None


def test_prefill_no_progress_retry_excludes_boundary_resume_guard() -> None:
    assert pd_prefill_no_progress_retry_metadata(
        phase_pure=True,
        selected_phase="prefill",
        total_num_scheduled_tokens=0,
        has_logical_prefill_work=True,
        decode_output_pending=True,
        asynchronous_progress_pending=False,
        observed_free_blocks=271,
        defer_reasons={
            "ttft_prefill_chunk_policy",
            "virtual_context_resume_guard",
        },
    ) is None


def test_prefill_no_progress_retry_requires_inflight_decode() -> None:
    assert pd_prefill_no_progress_retry_metadata(
        phase_pure=True,
        selected_phase="prefill",
        total_num_scheduled_tokens=0,
        has_logical_prefill_work=True,
        decode_output_pending=False,
        asynchronous_progress_pending=False,
        observed_free_blocks=125,
        defer_reasons={"policy"},
    ) is None


def test_decode_growth_runway_covers_only_same_hard_safe_boundary() -> None:
    runway = ScheduledPrefillFitDecodeGrowthRunway(
        target_request_id="short-2",
        target_logical_tokens=6144,
        target_chunk_tokens=2048,
        capacity_headroom_blocks=41,
        committed_reclaim_blocks=128,
        created_iteration_id=10,
    )
    decode_drift_claim = ScheduledPrefillCapacityClaim(
        request_id="short-2",
        logical_tokens=6144,
        chunk_tokens=2048,
        raw_required_blocks=128,
        hard_reserved_blocks=0,
        capacity_headroom_blocks=41,
        reserved_blocks=41,
        free_blocks=167,
        reclaim_blocks=2,
    )

    assert decode_drift_claim.hard_reclaim_blocks == 0
    assert runway.covers(decode_drift_claim)
    assert not runway.covers(
        ScheduledPrefillCapacityClaim(
            **{
                **decode_drift_claim.__dict__,
                "request_id": "new-short",
            }
        )
    )
    assert not runway.covers(
        ScheduledPrefillCapacityClaim(
            **{
                **decode_drift_claim.__dict__,
                "free_blocks": 127,
                "reclaim_blocks": 42,
            }
        )
    )


def test_projected_srpt_order_uses_remaining_tokens_and_tiebreaks() -> None:
    candidates = [
        ProjectedPrefillCandidate("long", 1.0, 50_000, 20_000),
        ProjectedPrefillCandidate("short-late", 3.0, 10_000, 4_000),
        ProjectedPrefillCandidate("short-early", 2.0, 10_000, 4_000),
    ]

    assert projected_srpt_prefill_order(candidates) == (
        "short-early",
        "short-late",
        "long",
    )


def test_scheduled_prefill_fit_uses_waiting_best_fit_whole_victim() -> None:
    long_1 = _request(
        "long-1",
        prompt_tokens=50_000,
        computed_tokens=34_816,
        arrival_time=1.0,
    )
    long_2 = _request(
        "long-2",
        prompt_tokens=50_000,
        computed_tokens=18_432,
        arrival_time=2.0,
    )
    short_3 = _request(
        "short-3",
        prompt_tokens=10_000,
        computed_tokens=8_192,
        arrival_time=3.0,
    )
    short_4 = _request(
        "short-4",
        prompt_tokens=10_000,
        computed_tokens=4_096,
        arrival_time=4.0,
    )
    scheduler = _scheduler(
        [long_1, long_2, short_3, short_4],
        resident_blocks={
            "long-1": 2176,
            "long-2": 1152,
            "short-3": 512,
            "short-4": 256,
        },
    )
    scheduler.running.remove(long_2)
    scheduler.waiting.append(long_2)
    # A quiescent prefill waiting for its next SRPT turn remains an active
    # request-granular victim-policy candidate.

    (
        target,
        victims,
        _projected_tokens,
        _next_chunk_tokens,
        _raw_required_blocks,
        _reserved_blocks,
        _free_blocks,
        deficit_blocks,
        snapshot_pages,
    ) = scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
        {"short-3": 1808},
        ring_token_cap=2048,
    )

    assert target is short_4
    assert tuple(victim.request_id for victim in victims) == ("long-2",)
    assert deficit_blocks == 114
    assert len(snapshot_pages) == 1_152
    assert {request_id for request_id, _, _ in snapshot_pages} == {"long-2"}


def test_scheduled_prefill_fit_waits_for_pending_best_fit_victim() -> None:
    long_1 = _request(
        "long-1",
        prompt_tokens=50_000,
        computed_tokens=34_816,
        arrival_time=1.0,
    )
    long_2 = _request(
        "long-2",
        prompt_tokens=50_000,
        computed_tokens=18_432,
        arrival_time=2.0,
    )
    short_3 = _request(
        "short-3",
        prompt_tokens=10_000,
        computed_tokens=8_192,
        arrival_time=3.0,
    )
    short_4 = _request(
        "short-4",
        prompt_tokens=10_000,
        computed_tokens=4_096,
        arrival_time=4.0,
    )
    scheduler = _scheduler(
        [long_1, long_2, short_3, short_4],
        resident_blocks={
            "long-1": 2_176,
            "long-2": 1_152,
            "short-3": 512,
            "short-4": 256,
        },
    )
    scheduler._pd_pending_request_ids = {"long-2"}

    *_, deficit_blocks, snapshot_pages = (
        scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
            {"short-3": 1_808},
            ring_token_cap=2_048,
        )
    )

    assert deficit_blocks == 114
    assert snapshot_pages == ()


def test_scheduled_prefill_fit_grows_best_fit_whole_victim_set() -> None:
    long_1 = _request(
        "long-1",
        prompt_tokens=50_000,
        computed_tokens=34_816,
        arrival_time=1.0,
    )
    long_2 = _request(
        "long-2",
        prompt_tokens=50_000,
        computed_tokens=18_432,
        arrival_time=2.0,
    )
    short_3 = _request(
        "short-3",
        prompt_tokens=10_000,
        computed_tokens=8_192,
        arrival_time=3.0,
    )
    short_4 = _request(
        "short-4",
        prompt_tokens=10_000,
        computed_tokens=4_096,
        arrival_time=4.0,
    )
    scheduler = _scheduler(
        [long_1, long_2, short_3, short_4],
        resident_blocks={
            "long-1": 100,
            "long-2": 50,
            "short-3": 512,
            "short-4": 256,
        },
    )

    *_, deficit_blocks, snapshot_pages = (
        scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
            {"short-3": 1808},
            ring_token_cap=2048,
        )
    )

    assert deficit_blocks == 114
    assert [
        request_id
        for request_id in dict.fromkeys(
            request_id for request_id, _, _ in snapshot_pages
        )
    ] == ["long-1", "long-2"]
    assert sum(
        request_id == "long-2" for request_id, _, _ in snapshot_pages
    ) == 50
    assert sum(
        request_id == "long-1" for request_id, _, _ in snapshot_pages
    ) == 100


def test_scheduled_prefill_fit_self_victims_only_without_peer_capacity() -> None:
    long_1 = _request(
        "long-1",
        prompt_tokens=50_000,
        computed_tokens=18_432,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [long_1],
        resident_blocks={"long-1": 1_152},
    )

    target, victims, *_, deficit_blocks, snapshot_pages = (
        scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
            {"long-1": 1_808},
            ring_token_cap=2_048,
        )
    )

    assert target is long_1
    assert deficit_blocks > 0
    assert victims == (long_1,)
    assert len(snapshot_pages) == 1_152
    assert scheduler._scheduled_prefill_fit_protected_victim_skip_count == 1


def test_scheduled_prefill_fit_discards_native_peers_for_ring_fallback() -> None:
    target = _request(
        "target",
        prompt_tokens=10_000,
        computed_tokens=4_096,
        arrival_time=1.0,
    )
    peer = _request(
        "peer",
        prompt_tokens=50_000,
        computed_tokens=20_000,
        arrival_time=2.0,
    )
    scheduler = _scheduler(
        [target, peer],
        resident_blocks={"target": 256, "peer": 50},
    )
    scheduler.kv_cache_manager.required_blocks = 300

    planned_target, victims, *_, deficit_blocks, snapshot_pages = (
        scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
            {"target": 1_808},
            ring_token_cap=2_048,
        )
    )

    assert planned_target is target
    assert deficit_blocks == 114
    assert victims == (target,)
    assert {request_id for request_id, _, _ in snapshot_pages} == {"target"}
    assert scheduler._scheduled_prefill_fit_last_plan_mode == "ring_fallback"
    assert scheduler._scheduled_prefill_fit_last_native_deficit_blocks == 286
    assert (
        scheduler._scheduled_prefill_fit_last_ring_fallback_deficit_blocks
        == 114
    )
    assert scheduler._scheduled_prefill_fit_last_avoided_peer_blocks == 50


def test_scheduled_prefill_fit_adds_only_ring_deficit_peer_after_target() -> None:
    target = _request(
        "target",
        prompt_tokens=10_000,
        computed_tokens=4_096,
        arrival_time=1.0,
    )
    peer = _request(
        "peer",
        prompt_tokens=50_000,
        computed_tokens=20_000,
        arrival_time=2.0,
    )
    scheduler = _scheduler(
        [target, peer],
        resident_blocks={"target": 50, "peer": 100},
    )
    scheduler.kv_cache_manager.required_blocks = 300

    planned_target, victims, *_, deficit_blocks, snapshot_pages = (
        scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
            {"target": 1_808},
            ring_token_cap=2_048,
        )
    )

    assert planned_target is target
    assert deficit_blocks == 114
    assert victims == (target, peer)
    assert [
        request_id
        for request_id in dict.fromkeys(
            request_id for request_id, _, _ in snapshot_pages
        )
    ] == ["target", "peer"]


def test_scheduled_prefill_fit_never_snapshots_current_compute_request() -> None:
    long_1 = _request(
        "long-1",
        prompt_tokens=50_000,
        computed_tokens=20_000,
        arrival_time=1.0,
    )
    short_2 = _request(
        "short-2",
        prompt_tokens=10_000,
        computed_tokens=8_192,
        arrival_time=2.0,
    )
    scheduler = _scheduler(
        [long_1, short_2],
        resident_blocks={"long-1": 1_250, "short-2": 0},
    )

    target, victims, *_, deficit_blocks, snapshot_pages = (
        scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
            {"long-1": 1_808},
            ring_token_cap=2_048,
        )
    )

    assert target is short_2
    assert deficit_blocks > 0
    assert victims == ()
    assert snapshot_pages == ()
    # Both the currently submitted Long and the live Short target are
    # protected entries in the projected active set.
    assert scheduler._scheduled_prefill_fit_protected_victim_skip_count == 2


def test_scheduled_prefill_fit_prepares_final_prefill_decode_handoff() -> None:
    long_1 = _request(
        "long-1",
        prompt_tokens=50_000,
        computed_tokens=48_192,
        arrival_time=1.0,
    )
    long_2 = _request(
        "long-2",
        prompt_tokens=50_000,
        computed_tokens=0,
        arrival_time=2.0,
    )
    long_2.virtual_context_deferred_logical_computed_tokens = 28_672
    scheduler = _scheduler(
        [long_1, long_2],
        resident_blocks={"long-1": 2_370, "long-2": 1_684},
        free_blocks=41,
    )
    scheduler.running.remove(long_2)
    scheduler.waiting.append(long_2)
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 41
    scheduler.kv_cache_manager.reload_pages["long-1"] = 755

    (
        target,
        victims,
        projected_tokens,
        next_chunk_tokens,
        raw_required_blocks,
        reserved_blocks,
        free_blocks,
        deficit_blocks,
        snapshot_pages,
    ) = scheduler._scheduled_prefill_fit_proactive_snapshot_plan(
        {"long-1": 1_808},
        ring_token_cap=2_048,
    )

    assert target is long_1
    assert projected_tokens == 50_000
    assert next_chunk_tokens == 0
    assert raw_required_blocks == 755
    assert reserved_blocks == 41
    assert free_blocks == 41
    assert deficit_blocks == 755
    assert victims == (long_2,)
    assert len(snapshot_pages) == 1_684
    assert {request_id for request_id, _, _ in snapshot_pages} == {"long-2"}


def test_best_fit_whole_victim_selects_minimum_single_cover() -> None:
    candidates = [
        ResidencyVictimCandidate("large", 1.0, 1000, 500, 256),
        ResidencyVictimCandidate("best", 2.0, 1000, 500, 120),
        ResidencyVictimCandidate("small", 3.0, 1000, 500, 80),
    ]

    selected = select_best_fit_whole_request_victims(
        candidates,
        required_blocks=114,
    )

    assert tuple(candidate.request_id for candidate in selected) == ("best",)


def test_scheduled_prefill_fit_exact_claim_controls_pressure_gate(
    monkeypatch,
) -> None:
    request = _request(
        "short",
        prompt_tokens=10_000,
        computed_tokens=1_024,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"short": 64},
        free_blocks=122,
    )
    scheduler.kv_cache_manager.required_blocks = 64
    scheduler.virtual_context_decode_hotset_first = True
    scheduler.virtual_context_staging_reserved_blocks = 8
    scheduler.virtual_context_ring_min_free_blocks = 8
    scheduler.virtual_context_decode_hotset_reserve_blocks = 16
    scheduler._dynamic_virtual_context_ring_prefill_admission_tokens = (
        lambda: 1024
    )
    scheduler._virtual_context_residency_fixed_ring_chunk_tokens = (
        lambda: 1024
    )
    scheduler._debug_has_prefill_remaining = lambda candidate: True
    scheduler._virtual_context_reload_required_page_count = (
        lambda request_ids, phase="decode": 0
    )
    monkeypatch.delenv("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL", raising=False)

    claim = scheduler._scheduled_prefill_fit_capacity_claim(
        request,
        ring_token_cap=1024,
    )
    assert claim.chunk_tokens == 1024
    assert claim.raw_required_blocks == 64
    assert claim.required_blocks_with_reserve == 96
    assert claim.reclaim_blocks == 0

    (
        native_safe,
        _reload_pages,
        free_blocks,
        threshold_blocks,
        bootstrap_blocks,
        ring_blocks,
        reason,
    ) = scheduler._prefill_native_safe_before_ring_pressure(request)
    assert native_safe
    assert free_blocks == 122
    assert threshold_blocks == 96
    assert bootstrap_blocks == 64
    assert ring_blocks == 0
    assert reason == "scheduled_prefill_fit_full_prefill_available"


def test_scheduled_prefill_fit_headroom_triggers_before_hard_shortfall() -> None:
    request = _request(
        "short",
        prompt_tokens=10_000,
        computed_tokens=1_024,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"short": 64},
        free_blocks=168,
    )
    scheduler.kv_cache_manager.required_blocks = 128
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 41

    claim = scheduler._scheduled_prefill_fit_capacity_claim(
        request,
        ring_token_cap=2048,
    )

    assert claim.raw_required_blocks == 128
    assert claim.capacity_headroom_blocks == 41
    assert claim.reclaim_blocks == 1
    assert claim.hard_reclaim_blocks == 0


def test_native_decode_handoff_requires_complete_allocator_reload_plan() -> None:
    request = _request(
        "decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    request.virtual_context_deferred_state = "tiered_provider_resumed"
    request.virtual_context_hbm_reloaded_tokens = 0
    scheduler = _scheduler(
        [request],
        resident_blocks={"decode": 497},
        free_blocks=177,
    )
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 41
    scheduler.virtual_context_reload_to_hbm_min_free_blocks = 8
    scheduler.kv_cache_manager.reload_pages["decode"] = 128

    ready, pages, required, minimum, reason = (
        scheduler._prepare_virtual_context_native_decode_handoff(request)
    )

    assert ready
    assert pages == 128
    assert required == 128
    assert minimum == 177
    assert reason == "scheduled"
    assert request.virtual_context_hbm_reloaded_tokens == 0
    assert request.virtual_context_hbm_reload_planned_tokens == 2_048
    assert "decode" in scheduler.kv_cache_manager.allocated_reload_requests
    assert (
        scheduler.kv_cache_manager.resume_partial_residency("decode", 10_000)[1]
        == 0
    )


def test_native_decode_handoff_capacity_precheck_avoids_full_page_plan() -> None:
    request = _request(
        "decode-blocked",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"decode-blocked": 113},
        free_blocks=100,
    )
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
    scheduler.kv_cache_manager.reload_pages["decode-blocked"] = 512

    pages, logical_pages, required, minimum, progress_reserved = (
        scheduler._virtual_context_native_decode_handoff_requirement(request)
    )

    assert pages == 512
    assert logical_pages == {}
    assert required == 512
    assert minimum == 594
    assert not progress_reserved
    assert scheduler.kv_cache_manager.reload_plan_calls == 0
    assert scheduler._native_decode_handoff_capacity_precheck_count == 1
    assert scheduler._native_decode_handoff_plan_avoided_page_count == 512


def test_decode_handoff_reclaim_rejects_impossible_aggregate_before_scan(
    monkeypatch,
) -> None:
    decode = _request(
        "short-decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=2.0,
    )
    prefill = _request(
        "long-prefill",
        prompt_tokens=50_000,
        computed_tokens=20_480,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [decode, prefill],
        resident_blocks={"short-decode": 113, "long-prefill": 128},
        free_blocks=81,
    )

    def fail_candidate_scan(**kwargs):
        del kwargs
        raise AssertionError("copy-ready candidate scan must be avoided")

    monkeypatch.setattr(
        scheduler,
        "_ordered_virtual_context_prefill_reclaim_candidates",
        fail_candidate_scan,
    )

    attempts, failures, victims, reasons, final_free = (
        scheduler._reclaim_prefill_victims_until_hbm_target(
            target_free_blocks=786,
            excluded_request_ids={decode.request_id},
            require_full_target_before_reclaim=True,
            reclaim_feasibility_cache_key=decode.request_id,
        )
    )

    assert (attempts, failures, victims, final_free) == (0, 0, [], 81)
    assert reasons == {"aggregate_prefill_reclaim_insufficient"}
    assert scheduler.kv_cache_manager.resident_blocks["long-prefill"] == 128
    assert scheduler._native_decode_handoff_reclaim_feasibility_check_count == 1
    assert scheduler._native_decode_handoff_reclaim_infeasible_count == 1
    assert scheduler._native_decode_handoff_reclaim_avoided_partial_blocks == 128

    def fail_resident_scan(request_id: str) -> int:
        raise AssertionError(f"resident rescan for {request_id}")

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_request_resident_block_count",
        fail_resident_scan,
    )
    cached_result = scheduler._reclaim_prefill_victims_until_hbm_target(
        target_free_blocks=786,
        excluded_request_ids={decode.request_id},
        require_full_target_before_reclaim=True,
        reclaim_feasibility_cache_key=decode.request_id,
    )
    assert cached_result == (
        0,
        0,
        [],
        {"aggregate_prefill_reclaim_insufficient"},
        81,
    )
    assert scheduler._native_decode_handoff_reclaim_feasibility_check_count == 1
    assert scheduler._native_decode_handoff_reclaim_infeasible_cache_hit_count == 1


def test_decode_handoff_reclaim_rejects_incomplete_copy_ready_set(
    monkeypatch,
) -> None:
    decode = _request(
        "short-decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=2.0,
    )
    prefill = _request(
        "long-prefill",
        prompt_tokens=50_000,
        computed_tokens=20_480,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [decode, prefill],
        resident_blocks={"short-decode": 113, "long-prefill": 256},
        free_blocks=100,
    )
    copy_ready_snapshot = ResidencyVictimCandidate(
        request_id=prefill.request_id,
        arrival_time=prefill.arrival_time,
        prompt_tokens=prefill.num_prompt_tokens,
        logical_context_tokens=prefill.num_computed_tokens,
        estimated_reclaimable_blocks=64,
    )
    monkeypatch.setattr(
        scheduler,
        "_ordered_virtual_context_prefill_reclaim_candidates",
        lambda **kwargs: [(prefill, copy_ready_snapshot)],
    )

    attempts, failures, victims, reasons, final_free = (
        scheduler._reclaim_prefill_victims_until_hbm_target(
            target_free_blocks=300,
            excluded_request_ids={decode.request_id},
            require_full_target_before_reclaim=True,
        )
    )

    assert (attempts, failures, victims, final_free) == (0, 0, [], 100)
    assert reasons == {"aggregate_copy_ready_reclaim_insufficient"}
    assert scheduler.kv_cache_manager.resident_blocks["long-prefill"] == 256


def test_decode_handoff_reclaim_fails_fatal_after_partial_commit(
    monkeypatch,
) -> None:
    decode = _request(
        "short-decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=3.0,
    )
    first_victim = _request(
        "long-prefill-1",
        prompt_tokens=50_000,
        computed_tokens=20_480,
        arrival_time=1.0,
    )
    second_victim = _request(
        "long-prefill-2",
        prompt_tokens=50_000,
        computed_tokens=20_480,
        arrival_time=2.0,
    )
    scheduler = _scheduler(
        [decode, first_victim, second_victim],
        resident_blocks={
            decode.request_id: 113,
            first_victim.request_id: 100,
            second_victim.request_id: 100,
        },
        free_blocks=100,
    )
    free_blocks = [100]
    scheduler.kv_cache_manager.block_pool.get_num_free_blocks = (
        lambda: free_blocks[0]
    )
    scheduler.virtual_context_pending_victim_intent_enabled = False
    scheduler._virtual_context_pending_victim_intent = None
    scheduler._virtual_context_pending_victim_cancelled_count = 0
    scheduler._scheduled_prefill_fit_last_release_blocks = 0

    snapshots = {
        request.request_id: ResidencyVictimCandidate(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            prompt_tokens=request.num_prompt_tokens,
            logical_context_tokens=request.num_computed_tokens,
            estimated_reclaimable_blocks=100,
        )
        for request in (first_victim, second_victim)
    }

    def candidates(**kwargs):
        excluded = kwargs["excluded_request_ids"]
        return [
            (request, snapshots[request.request_id])
            for request in (first_victim, second_victim)
            if request.request_id not in excluded
        ]

    monkeypatch.setattr(
        scheduler,
        "_maximum_immediate_prefill_reclaim_blocks",
        lambda **kwargs: 200,
    )
    monkeypatch.setattr(
        scheduler,
        "_ordered_virtual_context_prefill_reclaim_candidates",
        candidates,
    )

    def commit(request, **kwargs):
        del kwargs
        if request is first_victim:
            free_blocks[0] += 100
            return True, "deferred_tiered_provider", 20_480
        return False, "tiered_provider_copy_not_ready", 0

    monkeypatch.setattr(
        scheduler,
        "_try_defer_running_prefill_for_virtual_context",
        commit,
    )

    with pytest.raises(TieredAllocatorInvariantError) as error:
        scheduler._reclaim_prefill_victims_until_hbm_target(
            target_free_blocks=250,
            excluded_request_ids={decode.request_id},
            require_full_target_before_reclaim=True,
        )

    message = str(error.value)
    assert "partially committed" in message
    assert "target_free_blocks=250" in message
    assert "initial_free_blocks=100" in message
    assert "final_free_blocks=200" in message
    assert "missing_free_blocks=50" in message
    assert first_victim not in scheduler.running
    assert second_victim in scheduler.running
    assert not issubclass(
        TieredAllocatorInvariantError,
        TieredBackendEligibilityError,
    )


def test_decode_handoff_reclaim_commit_miss_before_mutation_is_retryable(
    monkeypatch,
) -> None:
    decode = _request(
        "short-decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=2.0,
    )
    victim = _request(
        "long-prefill",
        prompt_tokens=50_000,
        computed_tokens=20_480,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [decode, victim],
        resident_blocks={decode.request_id: 113, victim.request_id: 200},
        free_blocks=100,
    )
    scheduler.virtual_context_pending_victim_intent_enabled = False
    scheduler._virtual_context_pending_victim_intent = None
    scheduler._virtual_context_pending_victim_cancelled_count = 0
    snapshot = ResidencyVictimCandidate(
        request_id=victim.request_id,
        arrival_time=victim.arrival_time,
        prompt_tokens=victim.num_prompt_tokens,
        logical_context_tokens=victim.num_computed_tokens,
        estimated_reclaimable_blocks=200,
    )
    monkeypatch.setattr(
        scheduler,
        "_maximum_immediate_prefill_reclaim_blocks",
        lambda **kwargs: 200,
    )
    monkeypatch.setattr(
        scheduler,
        "_ordered_virtual_context_prefill_reclaim_candidates",
        lambda **kwargs: [(victim, snapshot)],
    )
    monkeypatch.setattr(
        scheduler,
        "_try_defer_running_prefill_for_virtual_context",
        lambda request, **kwargs: (
            False,
            "tiered_provider_copy_not_ready",
            0,
        ),
    )

    attempts, failures, victims, reasons, final_free = (
        scheduler._reclaim_prefill_victims_until_hbm_target(
            target_free_blocks=250,
            excluded_request_ids={decode.request_id},
            require_full_target_before_reclaim=True,
        )
    )

    assert (attempts, failures, victims, final_free) == (1, 1, [], 100)
    assert reasons == {"tiered_provider_copy_not_ready"}
    assert victim in scheduler.running


def test_authoritative_reclaim_consumes_overlapping_preparation() -> None:
    victim = _request(
        "long-prefill",
        prompt_tokens=50_000,
        computed_tokens=20_480,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [victim],
        resident_blocks={victim.request_id: 128},
    )
    scheduler._scheduled_prefill_fit_preparation = (
        ScheduledPrefillFitPreparation(
            target_request_id="short-decode",
            victim_request_ids=(victim.request_id, "peer-prefill"),
            projected_logical_tokens=10_000,
            next_chunk_tokens=0,
            predicted_deficit_blocks=256,
            snapshot_pages=((victim.request_id, 0, 0),),
            created_iteration_id=7,
        )
    )

    consumed = scheduler._consume_scheduled_prefill_fit_preparation(
        {victim.request_id}
    )

    assert consumed
    assert scheduler._scheduled_prefill_fit_preparation is None
    assert scheduler._scheduled_prefill_fit_preparation_reclaim_consumed_count == 1
    assert (
        scheduler._scheduled_prefill_fit_preparation_consumed_victim_count
        == 1
    )


def test_native_decode_handoff_materializes_plan_after_capacity_recovers() -> None:
    request = _request(
        "decode-ready",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"decode-ready": 113},
        free_blocks=594,
    )
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
    scheduler.kv_cache_manager.reload_pages["decode-ready"] = 512

    pages, logical_pages, required, minimum, progress_reserved = (
        scheduler._virtual_context_native_decode_handoff_requirement(request)
    )

    assert pages == 512
    assert logical_pages == {0: set(range(512))}
    assert required == 512
    assert minimum == 594
    assert not progress_reserved
    assert scheduler.kv_cache_manager.reload_plan_calls == 1


def test_native_decode_handoff_revokes_soft_ring_quantum_not_growth_headroom() -> None:
    request = _request(
        "decode-continuation",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"decode-continuation": 113},
        free_blocks=300,
    )
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
    scheduler.kv_cache_manager.reload_pages["decode-continuation"] = 128
    scheduler._has_pending_prefill_for_ring_progress = lambda: True
    scheduler._virtual_context_active_decode_ring_progress_reserve = lambda: (
        2_048,
        192,
        64,
    )

    pages, logical_pages, required, minimum, progress_reserved = (
        scheduler._virtual_context_native_decode_handoff_requirement(request)
    )

    assert pages == 128
    assert logical_pages == {0: set(range(128))}
    assert required == 128
    # 128 reload + 64 hard Decode growth/staging + 82 (2%) headroom.
    # The soft 128-block Ring quantum does not delay Decode continuation.
    assert minimum == 274
    assert progress_reserved


def test_scheduler_execution_plan_freezes_native_reload_generation() -> None:
    request = _request(
        "short-5",
        prompt_tokens=10_000,
        computed_tokens=8_192,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"short-5": 128},
    )
    scheduler.kv_cache_manager.reload_pages["short-5"] = 64

    plans = scheduler._build_tiered_request_execution_plans(
        {"short-5": 1_808},
        {"short-5": ("native_reload", 64)},
    )

    assert plans == (("short-5", 1, "native_reload", 64),)
    assert request.kv_retention_execution_plan_generation == 1
    assert request.kv_retention_execution_plan_mode == "native_reload"


def test_scheduler_execution_plan_marks_provider_prefill_context_ring() -> None:
    request = _request(
        "long-ring",
        prompt_tokens=50_000,
        computed_tokens=20_000,
        arrival_time=1.0,
    )
    request.virtual_context_deferred_logical_computed_tokens = 20_000
    request.virtual_context_deferred_state = "tiered_provider_resumed"
    scheduler = _scheduler(
        [request],
        resident_blocks={"long-ring": 128},
    )
    scheduler.kv_cache_manager.reload_pages["long-ring"] = 64

    plans = scheduler._build_tiered_request_execution_plans(
        {"long-ring": 2_048},
        {},
    )

    assert plans == (("long-ring", 1, "context_ring", 64),)


def test_scheduler_execution_plan_generation_survives_request_id_reuse() -> None:
    old_request = _request(
        "reused-id",
        prompt_tokens=10_000,
        computed_tokens=8_192,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [old_request],
        resident_blocks={"reused-id": 128},
    )
    scheduler.kv_cache_manager.reload_pages["reused-id"] = 64

    old_plans = scheduler._build_tiered_request_execution_plans(
        {"reused-id": 1_808},
        {"reused-id": ("native_reload", 64)},
    )

    new_request = _request(
        "reused-id",
        prompt_tokens=10_000,
        computed_tokens=8_192,
        arrival_time=2.0,
    )
    scheduler.requests["reused-id"] = new_request
    new_plans = scheduler._build_tiered_request_execution_plans(
        {"reused-id": 1_808},
        {"reused-id": ("native_reload", 64)},
    )

    assert old_plans == (("reused-id", 1, "native_reload", 64),)
    assert new_plans == (("reused-id", 2, "native_reload", 64),)
    assert new_request.kv_retention_execution_plan_generation == 2


def test_blocked_native_decode_handoff_activates_continuation_lease() -> None:
    request = _request(
        "decode-blocked-lease",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"decode-blocked-lease": 113},
        free_blocks=100,
    )
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
    scheduler.kv_cache_manager.reload_pages["decode-blocked-lease"] = 128

    ready, pages, required, minimum, reason = (
        scheduler._prepare_virtual_context_native_decode_handoff(request)
    )

    assert not ready
    assert (pages, required, minimum) == (128, 128, 210)
    assert reason == "insufficient_hbm_blocks"
    assert request.kv_retention_decode_continuation_lease_active
    assert request.kv_retention_decode_continuation_waiting
    assert request.kv_retention_decode_continuation_wait_count == 1


def test_decode_continuation_lease_rejects_recompute_preemption() -> None:
    request = _request(
        "protected-decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    request.status = RequestStatus.RUNNING
    request.kv_retention_decode_continuation_lease_active = True
    scheduler = object.__new__(Scheduler)

    with pytest.raises(RuntimeError, match="Decode continuation"):
        scheduler._preempt_request(request, 1.0)


def test_decode_continuation_wait_disables_native_control_fast_path() -> None:
    request = _request(
        "waiting-decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    request.kv_retention_decode_continuation_lease_active = True
    request.kv_retention_decode_continuation_waiting = True
    scheduler = object.__new__(Scheduler)
    scheduler.virtual_context_native_decode_control_fast_path = True
    scheduler._debug_has_prefill_remaining = lambda _request: False

    assert not scheduler._native_decode_runtime_fast_path_eligible(request)


def test_native_decode_handoff_waits_for_model_output_ack() -> None:
    request = _request(
        "decode-pending-ack",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    request.virtual_context_deferred_state = "decode_hbm_reload_pending"
    request.virtual_context_hbm_reload_planned_tokens = 10_000
    scheduler = _scheduler(
        [request],
        resident_blocks={"decode-pending-ack": 625},
        free_blocks=177,
    )
    scheduler.kv_cache_manager.reload_pages["decode-pending-ack"] = 0

    ready, pages, required, minimum, reason = (
        scheduler._prepare_virtual_context_native_decode_handoff(request)
    )

    assert not ready
    assert (pages, required, minimum) == (0, 0, 0)
    assert reason == "reload_completion_unacknowledged"
    assert request.virtual_context_deferred_state == "decode_hbm_reload_pending"
    assert request.virtual_context_hbm_reload_planned_tokens == 10_000


def test_native_prefill_promotion_requires_full_remaining_capacity() -> None:
    request = _request(
        "ring-prefill",
        prompt_tokens=10_000,
        computed_tokens=4_096,
        arrival_time=1.0,
    )
    request.virtual_context_deferred_state = "tiered_provider_resumed"
    request.virtual_context_hbm_reloaded_tokens = 0
    scheduler = _scheduler(
        [request],
        resident_blocks={"ring-prefill": 0},
        free_blocks=1_000,
    )
    scheduler.kv_cache_manager.reload_pages["ring-prefill"] = 256
    scheduler._debug_running_decode_resident_blocks = lambda: 0
    scheduler._pd_pending_decode_active = lambda: False
    scheduler._debug_has_prefill_remaining = lambda candidate: True

    ready, pages, required, reason = (
        scheduler._prepare_virtual_context_native_prefill_promotion(
            request,
            ring_token_cap=2_048,
        )
    )

    assert ready
    assert pages == 256
    assert required == 256
    assert reason == "scheduled"
    assert request.virtual_context_deferred_state == "hbm_reload_pending"
    assert request.virtual_context_hbm_reloaded_tokens == 0
    assert request.virtual_context_hbm_reload_planned_tokens == 4_096


def test_native_decode_handoff_blocks_allocator_plan_mismatch() -> None:
    request = _request(
        "decode",
        prompt_tokens=10_000,
        computed_tokens=10_000,
        arrival_time=1.0,
    )
    request.virtual_context_deferred_state = "tiered_provider_resumed"
    request.virtual_context_hbm_reloaded_tokens = 0
    scheduler = _scheduler(
        [request],
        resident_blocks={"decode": 498},
        free_blocks=177,
    )
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 41
    scheduler.virtual_context_reload_to_hbm_min_free_blocks = 8
    scheduler.kv_cache_manager.reload_pages["decode"] = 128
    scheduler.kv_cache_manager.resume_partial_residency = (
        lambda request_id, logical_tokens: (498, 127)
    )

    ready, pages, required, minimum, reason = (
        scheduler._prepare_virtual_context_native_decode_handoff(request)
    )

    assert not ready
    assert pages == 128
    assert required == 128
    assert minimum == 177
    assert reason == "reload_allocator_plan_mismatch"
    assert not scheduler.kv_cache_manager.allocated_reload_requests


def test_scheduled_prefill_fit_decode_growth_runway_admits_exact_chunk(
    monkeypatch,
) -> None:
    request = _request(
        "short",
        prompt_tokens=10_000,
        computed_tokens=6_144,
        arrival_time=1.0,
    )
    scheduler = _scheduler(
        [request],
        resident_blocks={"short": 384},
        free_blocks=167,
    )
    scheduler.kv_cache_manager.required_blocks = 128
    scheduler.virtual_context_scheduled_prefill_fit_headroom_blocks = 41
    scheduler.virtual_context_decode_hotset_first = True
    scheduler._dynamic_virtual_context_ring_prefill_admission_tokens = (
        lambda: 2048
    )
    scheduler._virtual_context_residency_fixed_ring_chunk_tokens = (
        lambda: 2048
    )
    scheduler._debug_has_prefill_remaining = lambda candidate: True
    scheduler._virtual_context_reload_required_page_count = (
        lambda request_ids, phase="decode": 0
    )
    scheduler._virtual_context_running_decode_reload_required_page_count = (
        lambda: 0
    )
    scheduler._scheduled_prefill_fit_decode_growth_runway = (
        ScheduledPrefillFitDecodeGrowthRunway(
            target_request_id="short",
            target_logical_tokens=6_144,
            target_chunk_tokens=2_048,
            capacity_headroom_blocks=41,
            committed_reclaim_blocks=128,
            created_iteration_id=10,
        )
    )
    monkeypatch.delenv(
        "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL",
        raising=False,
    )

    native_safe, *_rest, reason = (
        scheduler._prefill_native_safe_before_ring_pressure(request)
    )

    assert native_safe
    assert reason == "scheduled_prefill_fit_decode_growth_runway"
