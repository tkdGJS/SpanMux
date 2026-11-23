# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm import envs
from vllm.isolation_memory import (
    isolation_execution_enabled,
    isolation_ring_query_token_limit,
)
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.tiered_required_d2h import raise_if_required_d2h_failed
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.decode_scheduler import TBTDecodeScheduler
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.isolation_hbm_guard import make_isolation_hbm_completion_guard
from vllm.v1.core.sched.isolation_scheduler_timing import (
    close_scheduler_timing,
    install_scheduler_timing,
)
from vllm.v1.core.sched.observability import (
    detailed_iteration_debug_enabled,
    iteration_debug_sample_every_n,
    lightweight_timeline_enabled,
)
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.prefill_chunk_scheduler import (
    PrefillAdmissionCandidate,
    PrefillAdmissionFrontier,
    PrefillChunkScheduler,
    PrefillChunkSchedulingPolicy,
    build_native_prefill_admission_frontier,
)
from vllm.v1.core.sched.residency_policy import (
    PendingResidencyVictimIntent,
    ProjectedPrefillCandidate,
    ResidencyVictimCandidate,
    ResidencyVictimPolicy,
    estimate_jit_min_copy_components,
    fixed_ring_attention_work,
    fixed_ring_chunk_count,
    order_residency_victims,
    projected_srpt_prefill_order,
    required_reclaim_blocks,
    scheduled_prefill_capacity_headroom_blocks,
    scheduled_prefill_reclaim_blocks,
    select_best_fit_whole_request_victims,
)
from vllm.v1.core.tiered_errors import TieredAllocatorInvariantError
from vllm.v1.core.tiered_execution import (
    TIERED_EXECUTION_AUTO,
    TIERED_EXECUTION_CONTEXT_RING,
    TIERED_EXECUTION_NATIVE_RELOAD,
    TIERED_EXECUTION_RING_DECODE,
    TieredExecutionPlanEntry,
    has_native_reload_authority,
    parse_tiered_execution_plan_entries,
)
from vllm.v1.core.virtual_context import (
    AttentionPhase,
    get_virtual_context_table,
    resolve_virtual_context_coverage_block_size,
    virtual_context_active_decode_ring_progress_reserve_blocks,
    virtual_context_enabled,
    virtual_context_prefill_needs_capacity_floor_snapshot,
    virtual_context_prefill_ring_required_mode,
    virtual_context_pressure_native_reload_min_free_blocks,
    virtual_context_residency_victim_context_tokens,
    virtual_context_residency_victim_sort_key,
    virtual_context_select_active_decode_ring_progress_tokens,
    virtual_context_timing_enabled,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.output_placeholder import OutputPlaceholderOwnershipError
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}

_VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES = {
    "deferred_tiered_provider",
    "decode_hbm_reload_wait_dram",
    "tiered_provider_resumed",
}

_VIRTUAL_CONTEXT_REHYDRATE_STATES = {
    "deferred_tiered_provider",
    "deferred_dram",
    "decode_hbm_reload_wait_dram",
    "decode_hbm_reload_pending",
    "hbm_reload_pending",
    "hbm_reload_resumed",
    "reloaded",
    "partial_reloaded",
    "tiered_provider_resumed",
}


def _read_bool_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_VALUES


def _pd_phase_local_ownership_changed(
    enabled: bool,
    scheduled_in_prev_step: bool,
    previous_phase: str | None,
    scheduled_phase: str,
) -> bool:
    return (
        enabled
        and scheduled_in_prev_step
        and previous_phase in {"decode", "prefill"}
        and scheduled_phase in {"decode", "prefill"}
        and previous_phase != scheduled_phase
    )


def pd_prefill_no_progress_retry_metadata(
    *,
    phase_pure: bool,
    selected_phase: str,
    total_num_scheduled_tokens: int,
    has_logical_prefill_work: bool,
    decode_output_pending: bool,
    asynchronous_progress_pending: bool,
    observed_free_blocks: int,
    defer_reasons: Iterable[str] = (),
) -> tuple[int, str] | None:
    """Describe a bounded retry for a zero-work Prefill lane.

    While a Decode output is in flight, immediately rerunning an unchanged
    phase-pure Prefill admission can only repeat the same allocator and policy
    scans.  The caller records a one-block sentinel threshold and explicitly
    marks the gate to wake when the pending output is recovered.  A real HBM
    increase may wake it earlier.  Async preparation waits are excluded because
    their completion is already the progress source.

    ``virtual_context_resume_guard`` is different from a stable admission
    failure.  It protects only the scheduler boundary that publishes or
    recovers a provider-backed request.  The same zero-work schedule call
    clears ``prev_step_scheduled_req_ids`` before returning, so suppressing the
    next Prefill call behind an HBM/output event loses the only retry that can
    resume the request.  Leave that boundary retry to the normal engine loop.
    """

    if not (
        phase_pure
        and selected_phase == "prefill"
        and total_num_scheduled_tokens == 0
        and has_logical_prefill_work
        and decode_output_pending
        and not asynchronous_progress_pending
    ):
        return None
    observed = max(0, int(observed_free_blocks))
    normalized_reasons = sorted(
        {str(reason).strip() for reason in defer_reasons if str(reason).strip()}
    )
    if "virtual_context_resume_guard" in normalized_reasons:
        return None
    reason_suffix = ",".join(normalized_reasons) or "unchanged_scheduler_state"
    return observed + 1, f"prefill_no_progress:{reason_suffix}"


def _read_non_negative_int_env(name: str, default: int = 0) -> int:
    raw_value = os.environ.get(name, "")
    if raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid non-negative integer env %s=%r",
            name,
            raw_value,
        )
        return default
    if value < 0:
        logger.warning("Ignoring negative integer env %s=%r", name, raw_value)
        return default
    return value


def _read_positive_float_env(name: str, default: float = 0.0) -> float:
    raw_value = os.environ.get(name, "")
    if raw_value == "":
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid positive float env %s=%r",
            name,
            raw_value,
        )
        return default
    if value < 0:
        logger.warning("Ignoring negative float env %s=%r", name, raw_value)
        return default
    return value


@dataclass(frozen=True)
class ReleasedPrefillVictim:
    request_id: str
    deferred_tokens: int
    reclaimed_blocks: int
    candidate_snapshot: ResidencyVictimCandidate
    released_logical_blocks: int = 0


@dataclass(frozen=True)
class ScheduledPrefillFitPreparation:
    """Whole-request D2H work for the next Prefill admission boundary."""

    target_request_id: str
    victim_request_ids: tuple[str, ...]
    projected_logical_tokens: int
    next_chunk_tokens: int
    predicted_deficit_blocks: int
    snapshot_pages: tuple[tuple[str, int, int], ...]
    created_iteration_id: int
    target_self_victim: bool = False

    @property
    def victim_request_id(self) -> str:
        """Return the first victim for backward-compatible debug metrics."""

        return self.victim_request_ids[0] if self.victim_request_ids else ""


@dataclass(frozen=True)
class DecodeHandoffReclaimInfeasibleObservation:
    """Stable scheduler state in which Prefill reclaim cannot reach target."""

    target_free_blocks: int
    observed_free_blocks: int
    prefill_state: tuple[tuple[str, int, int, bool], ...]
    prepared_request_ids: tuple[str, ...]
    maximum_reclaim_blocks: int


@dataclass(frozen=True)
class ScheduledPrefillCapacityClaim:
    """One authoritative full-Prefill HBM capacity decision.

    ``chunk_tokens`` remains the next bounded execution quantum.  The physical
    allocation claim covers every remaining prompt token so a target admitted
    as native does not repeatedly enter reclaim between chunks. The
    capacity-fraction complement is soft only without active Decode work; an
    active Decode request/future promotes it into ``hard_reserved_blocks``.
    """

    request_id: str
    logical_tokens: int
    chunk_tokens: int
    raw_required_blocks: int
    hard_reserved_blocks: int
    capacity_headroom_blocks: int
    reserved_blocks: int
    free_blocks: int
    reclaim_blocks: int
    waived_capacity_headroom_blocks: int = 0

    @property
    def required_blocks_with_reserve(self) -> int:
        if self.raw_required_blocks <= 0:
            return 0
        return self.raw_required_blocks + self.reserved_blocks

    @property
    def hard_reclaim_blocks(self) -> int:
        """Return the allocator-safety deficit without soft headroom."""

        return scheduled_prefill_reclaim_blocks(
            self.free_blocks,
            self.raw_required_blocks,
            self.hard_reserved_blocks,
        )


@dataclass(frozen=True)
class ScheduledPrefillFitDecodeGrowthRunway:
    """One-shot soft-headroom credit after an exact reclaim commit.

    The capacity-fraction complement may absorb intervening allocation drift
    only while no Decode work is active. Once Decode is active, the same
    complement is a hard admission reserve and this credit is invalidated.
    Keep the soft credit bound to the immutable target boundary until the
    corresponding Prefill chunk is admitted.
    """

    target_request_id: str
    target_logical_tokens: int
    target_chunk_tokens: int
    capacity_headroom_blocks: int
    committed_reclaim_blocks: int
    created_iteration_id: int

    def covers(self, claim: ScheduledPrefillCapacityClaim) -> bool:
        return (
            self.capacity_headroom_blocks > 0
            and claim.request_id == self.target_request_id
            and claim.logical_tokens == self.target_logical_tokens
            and claim.chunk_tokens == self.target_chunk_tokens
            and claim.hard_reclaim_blocks == 0
            and 0 < claim.reclaim_blocks <= self.capacity_headroom_blocks
        )


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.isolation_bounded_prefill_planning = isolation_execution_enabled(
            "BOUNDED_PREFILL_PLANNING")
        self.isolation_logical_release_readiness = (
            self.isolation_bounded_prefill_planning
            or isolation_execution_enabled("LOGICAL_RELEASE_READINESS")
        )
        self.enable_ttft_prefill_chunk_scheduling = (
            self.scheduler_config.enable_ttft_prefill_chunk_scheduling
        )
        self.ttft_prefill_native_chunk_max_tokens = (
            self.scheduler_config.ttft_prefill_native_chunk_max_tokens
        )
        self.ttft_prefill_chunk_scheduler = PrefillChunkScheduler(
            self.scheduler_config.ttft_prefill_chunk_scheduling_policy,
            quantum_tokens=(
                self.scheduler_config.ttft_prefill_chunk_quantum_tokens
            ),
            default_deadline_ms=(
                self.scheduler_config.ttft_prefill_default_deadline_ms
            ),
            estimated_ms_per_token=(
                self.scheduler_config.ttft_prefill_estimated_ms_per_token
            ),
            service_profile_path=(
                self.scheduler_config.ttft_prefill_service_profile_path
                if (
                    self.enable_ttft_prefill_chunk_scheduling
                    and self.scheduler_config.ttft_prefill_chunk_scheduling_policy
                    == "slo_srpt"
                )
                else ""
            ),
            slo_guard_ms=self.scheduler_config.ttft_prefill_slo_guard_ms,
            starvation_threshold_ms=(
                self.scheduler_config.ttft_prefill_starvation_threshold_ms
            ),
            starvation_max_bypasses=(
                self.scheduler_config.ttft_prefill_starvation_max_bypasses
            ),
        )
        self.enable_tbt_decode_scheduling = (
            self.scheduler_config.enable_tbt_decode_scheduling
        )
        self.tbt_decode_scheduler = TBTDecodeScheduler(
            self.scheduler_config.tbt_decode_scheduling_policy
        )
        self.tbt_slo_observability_enabled = (
            self.enable_tbt_decode_scheduling
            or (
                os.environ.get(
                    "VLLM_PD_DYNAMIC_SM_ENABLED", "0"
                ).strip().lower()
                in ("1", "true", "yes", "on")
                and os.environ.get(
                    "VLLM_PD_DYNAMIC_SM_POLICY", ""
                ).strip().lower()
                in (
                    "tbt",
                    "tbt_profiled",
                    "tbt_slo_profiled",
                    "feedback",
                    "tbt_feedback",
                    "tbt_slo_feedback",
                    "tbt_slo_predictive",
                )
            )
        )
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()
        self.prev_step_scheduled_phase_by_req: dict[str, str] = {}
        self._iteration_debug_index = 0
        self._iteration_debug_enabled = detailed_iteration_debug_enabled(
            virtual_context_timing=virtual_context_timing_enabled()
        )
        self._lightweight_timeline_enabled = lightweight_timeline_enabled()
        self._iteration_debug_sample_every_n = iteration_debug_sample_every_n()
        pd_multiplexing_baseline = os.environ.get(
            "VLLM_PD_MULTIPLEXING_BASELINE", ""
        ).strip().lower()
        if not pd_multiplexing_baseline:
            pd_multiplexing_baseline = os.environ.get(
                "VLLM_OPMUX_BASELINE", ""
            ).strip().lower()
        legacy_pd_no_mix = os.environ.get(
            "VLLM_PD_DECOUPLED_NO_MIX", ""
        ).lower() in ("1", "true", "yes", "on")
        if legacy_pd_no_mix and pd_multiplexing_baseline in {
            "",
            "0",
            "false",
            "no",
            "off",
            "none",
        }:
            pd_multiplexing_baseline = "decode_priority"
        self.pd_phase_separated_baseline_mode = (
            self._normalize_pd_phase_separated_baseline_mode(
                pd_multiplexing_baseline
            )
        )
        self.pd_decoupled_no_mix = (
            self.pd_phase_separated_baseline_mode != "off"
        )
        self._configure_isolation_phase_batch_limits()
        phase_local_model_runner = os.environ.get(
            "VLLM_PD_PHASE_LOCAL_MODEL_RUNNER",
            os.environ.get("VLLM_PD_MUXWISE_EVENT_LOOP", "0"),
        )
        self.pd_phase_local_model_runner_enabled = (
            phase_local_model_runner.lower() in _TRUE_VALUES
        )
        self.pd_wait_on_prefill_kv_capacity = _read_bool_env(
            "VLLM_PD_WAIT_ON_PREFILL_KV_CAPACITY"
        )
        self.isolation_required_d2h_progress = isolation_execution_enabled(
            "D2H_EXACT_PLAN_PROGRESS"
        )
        self.isolation_native_soft_headroom_progress = isolation_execution_enabled(
            "NATIVE_SOFT_HEADROOM_PROGRESS"
        )
        self.isolation_prefill_batch_kv_budget = isolation_execution_enabled(
            "PREFILL_BATCH_KV_BUDGET"
        )
        if self.isolation_native_soft_headroom_progress:
            logger.info("scheduled_prefill_fit may borrow soft headroom only "
                        "for Decode-free, resident Native work; hard reserves unchanged")
        self.isolation_prefill_fit_next_chunk = _read_bool_env(
            "VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK"
        )
        if self.isolation_prefill_fit_next_chunk:
            logger.info("scheduled_prefill_fit Native reclaim scope=next_chunk; "
                        "Native reload/promotion scope=full_prompt")
        self.pd_muxwise_dynamic_policy = _read_bool_env(
            "VLLM_PD_MUXWISE_DYNAMIC_POLICY"
        )
        self.virtual_context_enabled = virtual_context_enabled()
        if self.isolation_prefill_batch_kv_budget:
            if not self.pd_decoupled_no_mix or not self.virtual_context_enabled:
                raise ValueError("Prefill batch KV budget requires phase-pure tiered P/D")
            logger.info("Prefill batch KV budget enabled: shared live allocator "
                        "capacity; final Prefill protects imminent Decode headroom")
        self.virtual_context_defer_prefill = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_DEFER_PREFILL"
        )
        self.virtual_context_residency_admission = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_RESIDENCY_ADMISSION"
        )
        self.virtual_context_tiered_external_blocks = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_TIERED_EXTERNAL_BLOCKS"
        )
        self.virtual_context_staging_reserved_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_STAGING_RESERVED_BLOCKS", 0
            )
        )
        self.virtual_context_decode_hotset_reserve_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_DECODE_HOTSET_RESERVE_BLOCKS", 0
            )
        )
        self.virtual_context_tiered_provider_max_external_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_TIERED_PROVIDER_MAX_EXTERNAL_TOKENS",
                131072,
            )
        )
        self.virtual_context_recompute_on_hbm_relief = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_RECOMPUTE_ON_HBM_RELIEF"
        )
        self.virtual_context_recompute_on_hbm_relief_min_free_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RECOMPUTE_ON_HBM_RELIEF_MIN_FREE_BLOCKS",
                0,
            )
        )
        self.virtual_context_reload_to_hbm = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM"
        )
        self.virtual_context_reload_to_hbm_min_free_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM_MIN_FREE_BLOCKS",
                0,
            )
        )
        self.virtual_context_ring_prefill_admission_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS",
                _read_non_negative_int_env(
                    "VLLM_VIRTUAL_CONTEXT_RING_BW_MIN_TOKENS",
                    0,
                ),
            )
        )
        self.virtual_context_ring_prefill_admission_tokens_max = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS_MAX",
                _read_non_negative_int_env(
                    "VLLM_VIRTUAL_CONTEXT_RING_BW_MAX_TOKENS",
                    0,
                ),
            )
        )
        self.virtual_context_ring_min_progress_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_MIN_PROGRESS_TOKENS",
                1024,
            )
        )
        self.virtual_context_residency_victim_policy = (
            ResidencyVictimPolicy.parse(
                os.environ.get(
                    "VLLM_VIRTUAL_CONTEXT_RESIDENCY_VICTIM_POLICY",
                    ResidencyVictimPolicy.LONGEST_LOGICAL_CONTEXT_FIRST.value,
                )
            )
        )
        scheduled_prefill_fit_capacity_fraction = _read_positive_float_env(
            "VLLM_VIRTUAL_CONTEXT_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION",
            0.98,
        )
        if not 0.0 < scheduled_prefill_fit_capacity_fraction <= 1.0:
            logger.warning(
                "Ignoring scheduled-prefill-fit capacity fraction outside "
                "(0.0, 1.0]: %r; using 0.98",
                scheduled_prefill_fit_capacity_fraction,
            )
            scheduled_prefill_fit_capacity_fraction = 0.98
        self.virtual_context_scheduled_prefill_fit_capacity_fraction = (
            scheduled_prefill_fit_capacity_fraction
        )
        self.virtual_context_pending_victim_intent_enabled = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_PENDING_VICTIM_INTENT"
        )
        self.virtual_context_residency_ring_chunk_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RESIDENCY_RING_CHUNK_TOKENS",
                0,
            )
        )
        self.virtual_context_ring_prefill_admission_dynamic_min_free_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_DYNAMIC_MIN_FREE_BLOCKS",
                _read_non_negative_int_env(
                    "VLLM_VIRTUAL_CONTEXT_RING_BW_LOW_FREE_BLOCKS",
                    0,
                ),
            )
        )
        self.virtual_context_ring_prefill_admission_dynamic_step_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_DYNAMIC_STEP_BLOCKS",
                _read_non_negative_int_env(
                    "VLLM_VIRTUAL_CONTEXT_RING_BW_STEP_BLOCKS",
                    0,
                ),
            )
        )
        self.virtual_context_ring_prefill_admission_dynamic_step_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_DYNAMIC_STEP_TOKENS",
                _read_non_negative_int_env(
                    "VLLM_VIRTUAL_CONTEXT_RING_BW_STEP_TOKENS",
                    0,
                ),
            )
        )
        ring_bw_dynamic_raw = os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RING_BW_DYNAMIC", ""
        ).lower()
        if ring_bw_dynamic_raw:
            self.virtual_context_ring_bw_dynamic = (
                ring_bw_dynamic_raw in _TRUE_VALUES
            )
        else:
            self.virtual_context_ring_bw_dynamic = (
                self.virtual_context_ring_prefill_admission_tokens_max
                > self.virtual_context_ring_prefill_admission_tokens
            )
        ring_bw_policy = os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RING_BW_POLICY", "step"
        ).strip().lower()
        if ring_bw_policy in ("safe", "capacity", "safe_capacity"):
            self.virtual_context_ring_bw_policy = "safe_capacity"
        else:
            self.virtual_context_ring_bw_policy = "step"
        self.virtual_context_ring_bw_safety_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_BW_SAFETY_BLOCKS", 0
            )
        )
        self.virtual_context_ring_bw_use_scheduled_headroom = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_RING_BW_USE_SCHEDULED_HEADROOM"
        )
        self.virtual_context_ring_bw_decode_growth_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_BW_DECODE_GROWTH_BLOCKS", 0
            )
        )
        self._virtual_context_ring_bw_last_reason = "not_evaluated"
        self._virtual_context_ring_bw_last_free_blocks = 0
        self._virtual_context_ring_bw_last_steps = 0
        self._virtual_context_ring_bw_last_safe_blocks = 0
        self._virtual_context_ring_bw_last_safe_tokens = 0
        self._virtual_context_ring_bw_last_reserved_blocks = 0
        self._virtual_context_ring_bw_last_staging_blocks = 0
        self._virtual_context_ring_bw_last_safety_blocks = 0
        self._virtual_context_ring_bw_last_decode_growth_blocks = 0
        ring_residency_admission_raw = os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_RING_RESIDENCY_ADMISSION", ""
        ).lower()
        if ring_residency_admission_raw:
            self.virtual_context_ring_residency_admission = (
                ring_residency_admission_raw in _TRUE_VALUES
            )
        else:
            self.virtual_context_ring_residency_admission = (
                self.virtual_context_ring_prefill_admission_tokens > 0
                and _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND")
            )
        self.virtual_context_ring_min_free_blocks = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_MIN_FREE_BLOCKS", 0
            )
        )
        self.pd_prefill_starvation_max_decode_iters = (
            _read_non_negative_int_env(
                "VLLM_PD_PREFILL_STARVATION_MAX_DECODE_ITERS", 0
            )
        )
        self.pd_ring_min_prefill_tokens_with_decode = (
            _read_non_negative_int_env(
                "VLLM_PD_RING_MIN_PREFILL_TOKENS_WITH_DECODE", 0
            )
        )
        active_decode_prefill_cap_default = (
            self.pd_ring_min_prefill_tokens_with_decode
            or self.virtual_context_ring_prefill_admission_tokens
        )
        self.virtual_context_active_decode_native_prefill_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS",
                active_decode_prefill_cap_default,
            )
        )
        self.virtual_context_active_decode_ring_prefill_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_ACTIVE_DECODE_RING_PREFILL_TOKENS",
                active_decode_prefill_cap_default,
            )
        )
        self.virtual_context_active_decode_elastic_prefill = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_ACTIVE_DECODE_ELASTIC_PREFILL"
        )
        self.virtual_context_ring_pressure_bootstrap_tokens = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_RING_PRESSURE_BOOTSTRAP_TOKENS",
                0,
            )
        )
        self._virtual_context_ring_bw_last_active_decode_cap_tokens = 0
        self._virtual_context_ring_bw_last_active_decode_cap_reason = (
            "not_active_decode"
        )
        self._virtual_context_ring_bw_last_active_decode_reload_pages = 0
        self._virtual_context_ring_bw_last_progress_reserve_applied = 0
        self._virtual_context_ring_bw_last_progress_reserve_tokens = 0
        self._virtual_context_ring_bw_last_progress_reserve_blocks = 0
        self._virtual_context_ring_bw_last_progress_hard_reserved_blocks = 0
        self._virtual_context_ring_bw_last_progress_reserve_reason = (
            "not_evaluated"
        )
        self._virtual_context_ring_pressure_last_free_blocks = 0
        self._virtual_context_ring_pressure_last_threshold_blocks = 0
        self._virtual_context_ring_pressure_last_bootstrap_blocks = 0
        self._virtual_context_ring_pressure_last_ring_blocks = 0
        self._virtual_context_ring_pressure_last_reserved_blocks = 0
        self._virtual_context_ring_pressure_last_native_safe = 0
        self._virtual_context_ring_pressure_last_reason = "not_evaluated"
        self.virtual_context_decode_hotset_first = (
            _read_bool_env("VLLM_PD_DECODE_HOTSET_FIRST")
            or _read_bool_env("VLLM_VIRTUAL_CONTEXT_DECODE_HOTSET_FIRST")
        )
        self.virtual_context_decode_cold_streaming = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_DECODE_COLD_STREAMING"
        )
        self.virtual_context_decode_cold_streaming_on_prefill_done = (
            _read_bool_env(
                "VLLM_VIRTUAL_CONTEXT_DECODE_COLD_STREAMING_ON_PREFILL_DONE"
            )
        )
        self.virtual_context_decode_cold_streaming_slo_ms = (
            _read_positive_float_env(
                "VLLM_VIRTUAL_CONTEXT_DECODE_COLD_STREAMING_SLO_MS",
                _read_positive_float_env("VLLM_PD_TBT_SLO_MS", 200.0),
            )
        )
        self.virtual_context_decode_cold_streaming_min_slack_ms = (
            _read_positive_float_env(
                "VLLM_VIRTUAL_CONTEXT_DECODE_COLD_STREAMING_MIN_SLACK_MS",
                50.0,
            )
        )
        self.virtual_context_prefill_ring_tokens_with_decode_cold = (
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_PREFILL_RING_TOKENS_WITH_DECODE_COLD",
                self.pd_ring_min_prefill_tokens_with_decode,
            )
        )
        self.virtual_context_prefill_ring_required_mode = (
            virtual_context_prefill_ring_required_mode()
        )
        self.virtual_context_dram_register_pressure_only = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_PRESSURE_ONLY"
        )
        self.virtual_context_native_decode_control_fast_path = (
            _read_bool_env(
                "VLLM_VIRTUAL_CONTEXT_NATIVE_DECODE_CONTROL_FAST_PATH"
            )
        )
        self.virtual_context_debug_snapshot_interval = max(
            1,
            _read_non_negative_int_env(
                "VLLM_VIRTUAL_CONTEXT_DEBUG_SNAPSHOT_INTERVAL",
                1,
            ),
        )
        self._virtual_context_debug_stats_cache: dict[str, int | float] = {}
        self._virtual_context_debug_stats_cache_iteration = -1
        self._virtual_context_request_debug_stats_cache: dict[
            str, dict[str, int | float]
        ] = {}
        self._virtual_context_request_debug_stats_cache_iteration: dict[
            str, int
        ] = {}
        self._virtual_context_attention_plan_debug_stats_cache: dict[
            tuple[str, tuple[str, ...]], dict[str, int | float | str]
        ] = {}
        self._virtual_context_attention_plan_debug_stats_cache_iteration: dict[
            tuple[str, tuple[str, ...]], int
        ] = {}
        self._native_decode_reload_guard_fast_path_count = 0
        self._native_decode_reload_guard_full_scan_count = 0
        self._native_decode_handoff_capacity_precheck_count = 0
        self._native_decode_handoff_plan_avoided_page_count = 0
        self._native_decode_handoff_reclaim_feasibility_check_count = 0
        self._native_decode_handoff_reclaim_infeasible_count = 0
        self._native_decode_handoff_reclaim_avoided_partial_blocks = 0
        self._native_decode_handoff_reclaim_infeasible_cache_hit_count = 0
        self._native_decode_handoff_reclaim_infeasible_cache: dict[
            str, DecodeHandoffReclaimInfeasibleObservation
        ] = {}
        # Execution-plan generations must remain unique even when an external
        # request id is reused after the old Request object has been removed.
        # Otherwise a late ModelRunnerOutput from the old lifetime could
        # acknowledge the new request's tiered execution authority.
        self._tiered_execution_plan_next_generation = 0
        self._tiered_execution_single_flight_block_count = 0
        self._tiered_execution_single_flight_last_request_id = ""
        self._tiered_execution_single_flight_last_generation = 0
        self._tiered_execution_plan_ack_count = 0
        self._tiered_execution_plan_failed_load_ack_count = 0
        self._tiered_execution_plan_overwrite_reject_count = 0
        self.pd_request_fcfs = _read_bool_env("VLLM_PD_REQUEST_FCFS")
        # Proposal-only rollback gate. Upstream mixed batching and HBM-only
        # P/D baselines do not create DRAM-backed WAITING Decode entries.
        self.isolation_phase_waiting_scan = bool(
            self.virtual_context_enabled
            and self.pd_decoupled_no_mix
            and _read_bool_env("VLLM_KV_RETENTION_P1_PARKED_PREFILL")
            and os.environ.get("VLLM_ISOLATION_PHASE_WAITING_SCAN", "1").lower()
            in _TRUE_VALUES
        )
        self._isolation_phase_waiting_scan_logged: set[str] = set()
        # Independent rollback of the Prefill-only FCFS guard scope, without
        # undoing the earlier opposite-phase queue reachability fix.
        self.isolation_waiting_decode_fcfs_scope = bool(
            self.isolation_phase_waiting_scan
            and os.environ.get(
                "VLLM_ISOLATION_WAITING_DECODE_FCFS_SCOPE", "1"
            ).lower() in _TRUE_VALUES
        )
        self._isolation_waiting_decode_fcfs_scope_logged = False
        self._configure_isolation_waiting_decode_reclaim()
        if self.isolation_phase_waiting_scan:
            logger.info(
                "Isolation phase WAITING scan enabled for P1 proposal; "
                "opposite-phase rows retain queue order and KV ownership"
            )
            logger.info(
                "Isolation WAITING Decode FCFS scope: enabled=%d; "
                "only older-Prefill ordering is exempted; "
                "resume owners and reload admission remain authoritative",
                int(self.isolation_waiting_decode_fcfs_scope),
            )
        self.pd_output_placeholder_ownership_enabled = (
            self.pd_decoupled_no_mix
            and self.pd_phase_local_model_runner_enabled
        )
        self._pd_output_last_execution_key_by_req: dict[str, str] = {}
        if _read_bool_env("VLLM_PD_FINAL_PREFILL_SAMPLE_REPAIR"):
            logger.warning(
                "VLLM_PD_FINAL_PREFILL_SAMPLE_REPAIR is removed; the "
                "phase-local P/D path now uses explicit Future-generation "
                "output ownership."
            )
        if self.virtual_context_pending_victim_intent_enabled and not (
            self.virtual_context_defer_prefill
            and self.virtual_context_tiered_external_blocks
            and self.virtual_context_ring_residency_admission
            and _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND")
            and _read_bool_env(
                "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"
            )
            and _read_bool_env(
                "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM"
            )
            and _read_bool_env("VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC")
        ):
            raise RuntimeError(
                "Pending residency victims require ring residency, tiered "
                "DRAM-provider deferral, and asynchronous HBM snapshots."
            )
        self._virtual_context_last_decode_schedule_ts = 0.0
        self._pd_phase_separated_last_phase = "idle"
        self._pd_forced_schedule_phase: str | None = None
        self._pd_pending_request_ids: set[str] = set()
        self._pd_pending_phase_request_ids: dict[str, set[str]] = {}
        self._pd_pending_schedule_block_count = 0
        self._pd_pending_prefill_chain_allow_count = 0
        self._pd_pending_schedule_last_reason = "no_pending"
        self._pd_preemption_future_protection_count = 0
        self._pd_preemption_future_protection_last_request_ids = ""
        self._pd_live_decode_preemption_protection_count = 0
        self._pd_live_decode_preemption_protection_last_request_ids = ""
        self._virtual_context_pending_victim_intent: (
            PendingResidencyVictimIntent | None
        ) = None
        self._virtual_context_pending_victim_created_count = 0
        self._virtual_context_pending_victim_wait_count = 0
        self._virtual_context_pending_victim_revalidated_count = 0
        self._virtual_context_pending_victim_committed_count = 0
        self._virtual_context_pending_victim_reranked_count = 0
        self._virtual_context_pending_victim_cancelled_count = 0
        self._scheduled_prefill_fit_last_target_request_id = ""
        self._scheduled_prefill_fit_target_change_count = 0
        self._scheduled_prefill_fit_last_release_blocks = 0
        self._scheduled_prefill_fit_preparation: (
            ScheduledPrefillFitPreparation | None
        ) = None
        self._scheduled_prefill_fit_preparation_created_count = 0
        self._scheduled_prefill_fit_preparation_wait_count = 0
        self._scheduled_prefill_fit_preparation_ready_count = 0
        self._scheduled_prefill_fit_preparation_cancelled_count = 0
        self._scheduled_prefill_fit_preparation_target_mismatch_count = 0
        self._scheduled_prefill_fit_preparation_reclaim_consumed_count = 0
        self._scheduled_prefill_fit_preparation_consumed_victim_count = (
            0
        )
        self._scheduled_prefill_fit_protected_victim_skip_count = 0
        self._scheduled_prefill_fit_unprepared_reclaim_count = 0
        self._scheduled_prefill_fit_soft_headroom_shortfall_count = 0
        self._scheduled_prefill_fit_soft_headroom_shortfall_blocks = 0
        self._scheduled_prefill_fit_last_plan_mode = "none"
        self._scheduled_prefill_fit_last_native_deficit_blocks = 0
        self._scheduled_prefill_fit_last_ring_fallback_deficit_blocks = 0
        self._scheduled_prefill_fit_last_avoided_peer_blocks = 0
        self._scheduled_prefill_fit_decode_growth_runway: (
            ScheduledPrefillFitDecodeGrowthRunway | None
        ) = None
        self._scheduled_prefill_fit_decode_growth_runway_created_count = 0
        self._scheduled_prefill_fit_decode_growth_runway_admission_count = 0
        self._scheduled_prefill_fit_decode_growth_runway_invalidated_count = 0
        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self._configure_isolation_ring_query_budget()
        self.virtual_context_kv_page_size_bytes = max(
            (
                int(group.kv_cache_spec.page_size_bytes)
                for group in kv_cache_config.kv_cache_groups
            ),
            default=0,
        )
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )
        # For encoder-decoder models, allocate the maximum number of tokens for Cross
        # Attn blocks, as for Whisper its input is always padded to the maximum length.
        # TODO (NickLucche): Generalize to models with variable-length encoder inputs.
        self._num_encoder_max_input_tokens = (
            MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(vllm_config.model_config)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        self.virtual_context_scheduled_prefill_fit_total_usable_blocks = max(
            0,
            self.kv_cache_manager.block_pool.num_gpu_blocks - 1,
        )
        self.virtual_context_scheduled_prefill_fit_headroom_blocks = (
            scheduled_prefill_capacity_headroom_blocks(
                self.virtual_context_scheduled_prefill_fit_total_usable_blocks,
                self.virtual_context_scheduled_prefill_fit_capacity_fraction,
            )
        )
        self.virtual_context_scheduled_prefill_fit_capacity_limit_blocks = (
            self.virtual_context_scheduled_prefill_fit_total_usable_blocks
            - self.virtual_context_scheduled_prefill_fit_headroom_blocks
        )
        self.isolation_hbm_guard = make_isolation_hbm_completion_guard(self, logger.info)
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            )
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // len(kv_cache_config.kv_cache_groups) + 1
            ) * self.block_size

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                model_config=self.vllm_config.model_config,
                instance_id=self.vllm_config.instance_id,
            )

        install_scheduler_timing(self)

    @staticmethod
    def _debug_percentile(values: list[int | float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * q
        lower = int(idx)
        upper = min(lower + 1, len(ordered) - 1)
        frac = idx - lower
        return ordered[lower] * (1.0 - frac) + ordered[upper] * frac

    def _build_lightweight_timeline_debug(
        self,
        *,
        scheduled_timestamp: float,
        waiting_queue_len_before: int,
        running_reqs_before: int,
        num_scheduled_tokens: dict[str, int],
        scheduled_new_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        scheduled_residency_rehydrated_req_ids: set[str],
        scheduled_phase: str,
    ) -> tuple[
        dict[str, int | float | str],
        dict[str, dict[str, int | float | str]],
    ]:
        """Build the bounded timeline schema used by normal experiments.

        This intentionally avoids virtual-context page snapshots, allocator
        dumps, per-tenant KV scans, and the large full-debug dictionary. It
        contains only fields required to reconstruct scheduled Prefill/Decode
        intervals and the per-request progress shown in the ABCDE timeline.
        """

        scheduled_kinds = {
            request.request_id: "running" for request in scheduled_running_reqs
        }
        scheduled_kinds.update(
            {request.request_id: "resumed" for request in scheduled_resumed_reqs}
        )
        scheduled_kinds.update(
            {request.request_id: "new" for request in scheduled_new_reqs}
        )
        scheduled_kinds.update(
            {
                request_id: "residency_rehydrated"
                for request_id in scheduled_residency_rehydrated_req_ids
            }
        )

        request_debug: dict[str, dict[str, int | float | str]] = {}
        num_prefill_reqs = 0
        num_decode_reqs = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0
        prefill_context_max = 0
        decode_context_max = 0
        prefill_attention_work_proxy = 0.0
        decode_kv_read_token_proxy = 0.0
        request_rows: list[dict[str, int | float | str]] = []
        for request_id, scheduled_tokens in num_scheduled_tokens.items():
            request = self.requests[request_id]
            num_computed_tokens_before = request.num_computed_tokens
            prefill_remaining = self._pd_prefill_lane_remaining_tokens(
                request,
                num_computed_tokens=num_computed_tokens_before,
            )
            prefill_tokens = min(scheduled_tokens, prefill_remaining)
            decode_tokens = max(0, scheduled_tokens - prefill_tokens)
            if prefill_tokens > 0:
                num_prefill_reqs += 1
                num_prefill_tokens += prefill_tokens
                prefill_context_max = max(
                    prefill_context_max, num_computed_tokens_before
                )
                prefill_attention_work_proxy += (
                    prefill_tokens * num_computed_tokens_before
                    + prefill_tokens * (prefill_tokens + 1) / 2.0
                )
            if decode_tokens > 0:
                num_decode_reqs += 1
                num_decode_tokens += decode_tokens
                decode_context_max = max(
                    decode_context_max, num_computed_tokens_before
                )
                decode_kv_read_token_proxy += (
                    decode_tokens * num_computed_tokens_before
                )
            request_phase = (
                "mixed"
                if prefill_tokens > 0 and decode_tokens > 0
                else "prefill"
                if prefill_tokens > 0
                else "decode"
            )
            request_rows.append(
                {
                    "request_id": request_id,
                    "tenant_id": request.tenant_id,
                    "session_id": request.tenant_session_id,
                    "scheduled_kind": scheduled_kinds.get(
                        request_id, "unknown"
                    ),
                    "phase": request_phase,
                    "scheduled_tokens": scheduled_tokens,
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": decode_tokens,
                    "num_computed_tokens_before": num_computed_tokens_before,
                    "num_prompt_tokens": request.num_prompt_tokens,
                    "num_output_tokens": len(request.output_token_ids),
                    "remaining_prefill_tokens_before_chunk": prefill_remaining,
                    "pd_recompute_frontier_tokens": (
                        request.pd_recompute_frontier_tokens
                    ),
                    "token_iteration_id": self._iteration_debug_index,
                    "token_tenant_id": request.tenant_id,
                    "token_request_context_len": num_computed_tokens_before,
                }
            )

        for row in request_rows:
            row.update(
                {
                    "token_batch_decode_context_max": decode_context_max,
                    "token_batch_prefill_tokens": num_prefill_tokens,
                    "token_batch_prefill_attention_proxy": (
                        prefill_attention_work_proxy
                    ),
                    "token_batch_decode_kv_read_proxy": (
                        decode_kv_read_token_proxy
                    ),
                    "token_batch_has_long_prefill": int(
                        prefill_context_max >= 8192
                    ),
                    "token_batch_has_long_decode": int(
                        decode_context_max >= 8192
                    ),
                }
            )
            request_debug[str(row["request_id"])] = row

        iteration_debug: dict[str, int | float | str] = {
            "iteration_id": self._iteration_debug_index,
            "scheduled_ts": scheduled_timestamp,
            "lightweight_timeline": 1,
            "waiting_queue_len_before": waiting_queue_len_before,
            "waiting_queue_len_after": len(self.waiting),
            "waiting_queue_len": len(self.waiting),
            "running_reqs_before": running_reqs_before,
            "running_reqs_after": len(self.running),
            "running_queue_len": len(self.running),
            "scheduled_phase": scheduled_phase,
            "scheduled_num_seqs": len(num_scheduled_tokens),
            "total_num_scheduled_tokens": sum(num_scheduled_tokens.values()),
            "num_prefill_reqs": num_prefill_reqs,
            "num_decode_reqs": num_decode_reqs,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "scheduled_prefill_tokens": num_prefill_tokens,
            "scheduled_decode_tokens": num_decode_tokens,
            "decode_context_max": decode_context_max,
            "decode_kv_read_token_proxy": decode_kv_read_token_proxy,
            "prefill_attention_work_proxy": prefill_attention_work_proxy,
            "context_ring_bw_reason": (
                self._virtual_context_ring_bw_last_reason
            ),
            "context_ring_bw_free_blocks": (
                self._virtual_context_ring_bw_last_free_blocks
            ),
            "context_ring_bw_safe_tokens": (
                self._virtual_context_ring_bw_last_safe_tokens
            ),
            "context_ring_bw_reserved_blocks": (
                self._virtual_context_ring_bw_last_reserved_blocks
            ),
            "context_ring_bw_safety_blocks": (
                self._virtual_context_ring_bw_last_safety_blocks
            ),
            "context_ring_bw_uses_scheduled_headroom": int(
                self.virtual_context_ring_bw_use_scheduled_headroom
            ),
            "context_ring_bw_active_decode_cap_tokens": (
                self._virtual_context_ring_bw_last_active_decode_cap_tokens
            ),
            "context_ring_bw_active_decode_cap_reason": (
                self._virtual_context_ring_bw_last_active_decode_cap_reason
            ),
            "context_ring_bw_active_decode_elastic_enabled": int(
                self.virtual_context_active_decode_elastic_prefill
            ),
            "pd_preemption_future_protection_count": (
                self._pd_preemption_future_protection_count
            ),
            "pd_preemption_future_protection_last_request_ids": (
                self._pd_preemption_future_protection_last_request_ids
            ),
            "pd_live_decode_preemption_protection_count": (
                self._pd_live_decode_preemption_protection_count
            ),
            "pd_live_decode_preemption_protection_last_request_ids": (
                self._pd_live_decode_preemption_protection_last_request_ids
            ),
        }
        return iteration_debug, request_debug

    def _configure_isolation_phase_batch_limits(self) -> None:
        """Opt-in request rows per phase output, independent of Future depth."""
        for phase in ("prefill", "decode"):
            name = f"VLLM_ISOLATION_{phase.upper()}_BATCH_MAX_REQUESTS"
            raw = os.environ.get(name, "0")
            if not raw.isascii() or not raw.isdecimal():
                raise ValueError(f"{name} must be a non-negative integer")
            limit = int(raw)
            if limit > self.scheduler_config.max_num_seqs:
                raise ValueError(f"{name} exceeds max_num_seqs")
            if limit and not self.pd_decoupled_no_mix:
                raise ValueError(f"{name} requires phase-pure P/D scheduling")
            setattr(self, f"isolation_{phase}_batch_max_requests", limit)
        if (self.isolation_prefill_batch_max_requests
                or self.isolation_decode_batch_max_requests):
            logger.info("isolation phase batch request caps: prefill=%d decode=%d "
                        "(0=unlimited); Future depths and resident owners unchanged",
                        self.isolation_prefill_batch_max_requests,
                        self.isolation_decode_batch_max_requests)

    def _isolation_phase_request_limit(self, phase: str) -> int:
        if not self.pd_decoupled_no_mix or phase not in {"prefill", "decode"}:
            return 0
        return getattr(self, f"isolation_{phase}_batch_max_requests", 0)

    def park_kv_retention_prefill(self, output: SchedulerOutput) -> None:
        """Remove active admission, not logical progress or output ownership."""
        parked = getattr(self, "_kv_retention_parked_requests", {})
        rows = [self.requests[req_id] for req_id in output.num_scheduled_tokens]
        if (parked or not rows or any(request not in self.running
                or request.status != RequestStatus.RUNNING for request in rows)):
            raise RuntimeError("P1 park must move one complete running batch")
        ids = set(output.num_scheduled_tokens)
        self._kv_retention_parked_requests = {r.request_id: r for r in rows}
        self.running = [r for r in self.running if r.request_id not in ids]

    def can_resume_kv_retention_prefill(self, output: SchedulerOutput) -> bool:
        parked = getattr(self, "_kv_retention_parked_requests", {})
        if set(parked) != set(output.num_scheduled_tokens):
            raise RuntimeError("P1 resume lost Scheduler parked ownership")
        if any(self.requests.get(req_id) is not request or request.is_finished()
               for req_id, request in parked.items()):
            raise RuntimeError("P1 parked request incarnation changed")
        return len(self.running) + len(parked) <= self.max_num_running_reqs

    def resume_kv_retention_prefill(self, output: SchedulerOutput) -> None:
        if not self.can_resume_kv_retention_prefill(output):
            raise RuntimeError("P1 resume exceeds active sequence budget")
        self.running.extend(self._kv_retention_parked_requests.values())
        self._kv_retention_parked_requests = {}

    def discard_kv_retention_parked(self, output: SchedulerOutput) -> None:
        """Engine calls only after exact whole-owner Worker cleanup ACK."""
        parked = getattr(self, "_kv_retention_parked_requests", {})
        if set(parked) != set(output.num_scheduled_tokens):
            raise RuntimeError("P1 discard lost complete parked membership")
        self._kv_retention_parked_requests = {}

    def layerwise_offload_needed(
        self, output: SchedulerOutput, candidate: Request, *,
        replacement_candidates: tuple[Request, ...] = (),
    ) -> bool:
        """Conservative replacement pressure check, only on an SRPT yield.

        This is not admission: the normal scheduler still commits the actual
        batch/token budget. Do not copy shared or already-external source KV.
        """
        if has_native_reload_authority(output):
            return False
        manager = self.kv_cache_manager
        if manager.virtual_context is None or candidate.virtual_context_deferred_state:
            return False
        if getattr(self, "isolation_prefill_batch_kv_budget", False):
            queued_ids = {r.request_id for r in (*self.running, *self.waiting)}
            order = tuple(dict.fromkeys(
                r.request_id for r in (replacement_candidates or (candidate,))
                if r.request_id in queued_ids
                and r.request_id not in output.num_scheduled_tokens
                and not r.is_finished()
                and not self.blocks_next_schedule(r, "prefill")
            ))
            frontier = self._ttft_plan_prefill_admission_frontier(
                order, reserve_decode_work=False,
                parked_request_ids=tuple(output.num_scheduled_tokens),
            )
            if not frontier.planned_tokens_by_request:
                return False
            # Advisory, not a second allocator. Prefix hits not yet looked up
            # may make this an upper bound; live admission rechecks exact args.
            needed = sum(
                manager.estimate_num_blocks_to_allocate(
                    self.requests[rid], count,
                    num_lookahead_tokens=(
                        self.num_lookahead_tokens
                        if self.requests[rid].num_computed_tokens > 0 else 0),
                )
                for rid, count in frontier.planned_tokens_by_request
            )
            reserve = (
                self.virtual_context_scheduled_prefill_fit_headroom_blocks
                + self.virtual_context_staging_reserved_blocks
                + self.virtual_context_ring_min_free_blocks
                + self.virtual_context_decode_hotset_reserve_blocks
            )
            logger.info("layerwise replacement KV budget: requests=%d "
                        "required_blocks=%d reserve_blocks=%d free_blocks=%d",
                        len(frontier.request_ids), needed, reserve,
                        manager.block_pool.get_num_free_blocks())
        else:
            tokens = min(self.max_num_scheduled_tokens,
                         max(0, candidate.num_prompt_tokens - candidate.num_computed_tokens))
            if tokens <= 0:
                return False
            needed = manager.estimate_num_blocks_to_allocate(candidate, tokens)
            reserve = self.virtual_context_scheduled_prefill_fit_headroom_blocks
        if manager.block_pool.get_num_free_blocks() >= needed + reserve:
            return False
        seen = set()
        for request_id in output.num_scheduled_tokens:
            groups = manager.get_blocks(request_id).blocks
            if len(groups) != 1 or not groups[0]:
                return False
            for block in groups[0]:
                if block.is_null or block.ref_cnt != 1 or block.block_id in seen:
                    return False
                seen.add(block.block_id)
        return True

    def commit_layerwise_kv_offload(self, output: SchedulerOutput, transfer) -> bool:
        if has_native_reload_authority(output):
            raise TieredAllocatorInvariantError(
                "native_reload Future cannot release KV before all-layer reload ACK"
            )
        transfer.validate_output(output)
        requests = [self.requests[request_id] for request_id in output.num_scheduled_tokens]
        # Output claims, optimistic progress, and status deliberately survive.
        released = self.kv_cache_manager.release_layerwise_batch(transfer, requests)
        if released:
            self._consume_scheduled_prefill_fit_preparation(set(output.num_scheduled_tokens))
        return released

    def complete_layerwise_kv_offload(self, output: SchedulerOutput) -> None:
        """Called AFTER original final-output ledger consumption, never at yield."""
        transfer = output.kv_retention_layerwise_transfer
        if transfer is None:
            return
        transfer.validate_output(output)
        if transfer.state != "published":
            raise TieredAllocatorInvariantError("layerwise result lacks final DRAM publication")
        survivors = []
        for row in transfer.checkpoint.rows:
            request = self.requests.get(row.request_id)
            if request is None or request.is_finished():
                continue
            logical = row.start_token + row.scheduled_tokens
            request.validate_external_residency_suspension(
                logical_computed_tokens=logical, deferred_state="deferred_tiered_provider"
            )
            if request not in self.running or request.num_computed_tokens != logical:
                raise TieredAllocatorInvariantError("layerwise final progress/owner mismatch")
            survivors.append((request, logical))
        # Preflight all survivors before altering any queue.
        for request, logical in survivors:
            self.running.remove(request)
            request.status = RequestStatus.PREEMPTED
            request.suspend_to_external_residency(
                logical_computed_tokens=logical,
                deferred_state="deferred_tiered_provider",
                scheduler_iteration_id=self._iteration_debug_index,
            )
            request.virtual_context_deferred_prefill_count += 1
            request.virtual_context_deferred_hbm_released_tokens += logical
            self.waiting.prepend_request(request)
        self._consume_scheduled_prefill_fit_preparation(set(output.num_scheduled_tokens))
        transfer.mark_consumed()

    def _pd_prefill_lane_boundary_tokens(self, request: Request) -> int:
        """Return the logical boundary that must run on the Prefill lane.

        Ordinary requests stop Prefill at the original prompt boundary. In
        the experimental phase-separated path, a Decode request that suffered
        an allocator reset also has to replay its previously computed output
        history. Treating that history as live Decode would submit thousands
        of query tokens at once and violate the one-token Decode invariant.
        """

        boundary = int(request.num_prompt_tokens)
        if not getattr(self, "pd_decoupled_no_mix", False):
            return boundary
        recompute_frontier = max(
            0,
            int(getattr(request, "pd_recompute_frontier_tokens", 0)),
        )
        request_tokens = max(0, int(request.num_tokens))
        return max(boundary, min(recompute_frontier, request_tokens))

    def _pd_prefill_lane_remaining_tokens(
        self,
        request: Request,
        *,
        num_computed_tokens: int | None = None,
    ) -> int:
        computed_tokens = (
            request.num_computed_tokens
            if num_computed_tokens is None
            else num_computed_tokens
        )
        return max(
            0,
            self._pd_prefill_lane_boundary_tokens(request)
            - int(computed_tokens),
        )

    def _debug_has_prefill_remaining(self, request: Request) -> bool:
        return self._pd_prefill_lane_remaining_tokens(request) > 0

    @staticmethod
    def _virtual_context_waiting_decode_ready(request: Request) -> bool:
        """Return true when a DRAM-backed waiting request is decode-ready.

        Physical ``num_computed_tokens`` is reset when HBM blocks are freed,
        while the durable logical progress remains in the virtual-context
        field.  A request with a sampled output token must re-enter through the
        decode lane and reload its complete context before native attention.
        """
        return (
            request.status == RequestStatus.PREEMPTED
            and request.virtual_context_deferred_logical_computed_tokens
            >= request.num_prompt_tokens
            and request.num_output_tokens > 0
            and request.virtual_context_deferred_state
            in _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES
        )

    def _debug_needs_prefill_lane(self, request: Request) -> bool:
        owner = request.pd_output_placeholder_ledger.active
        return self._debug_has_prefill_remaining(request) or (
            owner is not None and not owner.cancelled and owner.phase == "prefill"
        )

    def _debug_running_phase_counts(self) -> dict[str, int]:
        prefill_reqs = 0
        decode_reqs = 0
        for request in self.running:
            if self._debug_needs_prefill_lane(request):
                prefill_reqs += 1
            else:
                decode_reqs += 1
        return {
            "num_running_prefill_reqs": prefill_reqs,
            "num_running_decode_reqs": decode_reqs,
        }

    def _debug_count_waiting_prefill_reqs(self) -> int:
        return sum(
            1
            for request in self.waiting
            if self._debug_needs_prefill_lane(request)
            and not self._virtual_context_waiting_decode_ready(request)
        )

    def _ttft_begin_prefill_chunk_iteration(self) -> tuple[str, ...]:
        """Rank every currently schedulable prefill request once.

        Running and waiting requests share one policy domain. Queue mutation
        is deliberately separate from ranking so a decode-only P/D iteration
        cannot have its waiting order changed by the prefill policy.
        """
        if not self.enable_ttft_prefill_chunk_scheduling:
            return ()
        candidates = [
            request
            for request in itertools.chain(self.running, self.waiting)
            if not self.blocks_next_schedule(request, "prefill")
            and self._debug_has_prefill_remaining(request)
            and not self._virtual_context_waiting_decode_ready(request)
        ]
        candidate_order = self.ttft_prefill_chunk_scheduler.begin_iteration(
            candidates,
            pending_request_ids=self._pd_pending_request_ids,
        )
        return candidate_order

    def _ttft_decode_admission_reservation(
        self,
        token_budget: int,
    ) -> tuple[int, int]:
        """Conservatively reserve native mixed-batch Decode admission.

        The regular scheduler scans RUNNING Decode requests before WAITING
        Prefills. The SRPT adapter must therefore not promise those tokens or
        sequence slots to Prefill. This helper mirrors only cheap, stable
        admission facts; the normal scheduler still performs handoff, KV, and
        model-length checks authoritatively.
        """
        remaining_token_budget = max(0, token_budget)
        reserved_tokens = 0

        for request in self.running:
            if remaining_token_budget <= 0:
                break
            if (
                request.is_finished()
                or self._debug_needs_prefill_lane(request)
                or self.blocks_next_schedule(request, "decode")
            ):
                continue
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens
                + 2
                - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                continue
            num_new_tokens = max(
                0,
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens,
            )
            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold
            num_new_tokens = min(
                num_new_tokens,
                remaining_token_budget,
                max(
                    0,
                    self.max_model_len
                    - 1
                    - request.num_computed_tokens,
                ),
            )
            reserved_tokens += num_new_tokens
            remaining_token_budget -= num_new_tokens

        available_new_sequence_slots = max(
            0,
            self.max_num_running_reqs - len(self.running),
        )
        reserved_waiting_decode_slots = 0
        for request in self.waiting:
            if (
                available_new_sequence_slots <= 0
                or remaining_token_budget <= 0
            ):
                break
            if (
                request.is_finished()
                or self._debug_needs_prefill_lane(request)
                or self.blocks_next_schedule(request, "decode")
                or not (
                    self._virtual_context_waiting_decode_ready(request)
                    or (
                        request.num_output_tokens > 0
                        and request.prefill_completed_at is not None
                    )
                )
            ):
                continue
            # A resumed Decode request contributes one service token after
            # its durable context is restored. The large reload allocation is
            # deliberately left to allocate_slots()/handoff admission.
            reserved_tokens += 1
            remaining_token_budget -= 1
            available_new_sequence_slots -= 1
            reserved_waiting_decode_slots += 1

        return reserved_tokens, reserved_waiting_decode_slots

    def _ttft_plan_prefill_admission_frontier(
        self,
        candidate_order: tuple[str, ...],
        *,
        reserve_decode_work: bool,
        force_single_reason: str = "",
        parked_request_ids: tuple[str, ...] = (),
    ) -> PrefillAdmissionFrontier:
        """Build the maximum policy-ordered Prefill set native budgets allow.

        There is no speculative KV preflight here. An explicit phase-only
        row quota must also constrain this frontier, or the RUNNING scan may
        displace a higher-priority WAITING request. The frontier bridges RUNNING
        and WAITING in separate loops. Actual token chunking, sequence
        admission, and ``allocate_slots()`` remain the source of truth.

        Context Ring/provider work and an active victim transaction remain
        single-target because one SchedulerOutput owns one immutable tiered
        execution/reclaim plan. ``parked_request_ids`` is advisory sequence
        accounting for a proposed whole-batch yield, never actual queue mutation.
        """
        token_budget_before = self.max_num_scheduled_tokens
        if not self.enable_ttft_prefill_chunk_scheduling:
            return PrefillAdmissionFrontier(
                token_budget_before=token_budget_before,
                token_budget_after=token_budget_before,
            )
        if not candidate_order:
            return PrefillAdmissionFrontier(
                token_budget_before=token_budget_before,
                token_budget_after=token_budget_before,
                stop_reason="empty_candidate_order",
            )

        if (
            not force_single_reason
            and self.ttft_prefill_chunk_scheduler.policy
            == PrefillChunkSchedulingPolicy.SLO_SRPT
        ):
            force_single_reason = "slo_srpt_single_request_model"

        decode_reserved_tokens = 0
        reserved_waiting_decode_slots = 0
        if reserve_decode_work:
            (
                decode_reserved_tokens,
                reserved_waiting_decode_slots,
            ) = self._ttft_decode_admission_reservation(
                token_budget_before
            )
        running_count = len(self.running)
        if parked_request_ids:
            running_count -= sum(r.request_id in parked_request_ids for r in self.running)
        new_sequence_slots_before = max(
            0,
            self.max_num_running_reqs
            - running_count
            - reserved_waiting_decode_slots,
        )
        waiting_request_ids = {
            request.request_id for request in self.waiting
        }
        admission_candidates: list[PrefillAdmissionCandidate] = []

        for request_id in candidate_order:
            if request_id in parked_request_ids:
                continue
            request = self.requests.get(request_id)
            if request is None:
                continue
            request_is_waiting = request_id in waiting_request_ids
            logical_computed_tokens = max(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
            num_new_tokens = self._pd_prefill_lane_remaining_tokens(
                request,
                num_computed_tokens=logical_computed_tokens,
            )
            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold
            execution_mode = self._ttft_prefill_execution_mode(request)
            num_new_tokens = self._ttft_cap_selected_prefill_chunk(
                request,
                num_new_tokens,
                num_computed_tokens=logical_computed_tokens,
                execution_mode=execution_mode,
            )
            num_new_tokens = min(
                num_new_tokens,
                max(
                    0,
                    self.max_model_len
                    - 1
                    - logical_computed_tokens,
                ),
            )
            if num_new_tokens <= 0:
                continue
            admission_candidates.append(
                PrefillAdmissionCandidate(
                    request_id=request_id,
                    num_tokens=num_new_tokens,
                    needs_new_sequence=request_is_waiting,
                    requires_single_target=(execution_mode != "native"),
                )
            )

        return build_native_prefill_admission_frontier(
            admission_candidates,
            token_budget_before=token_budget_before,
            decode_reserved_tokens=decode_reserved_tokens,
            new_sequence_slots=new_sequence_slots_before,
            force_single_reason=force_single_reason,
            request_limit=(
                getattr(self, "isolation_prefill_batch_max_requests", 0)
                if self.pd_decoupled_no_mix else 0
            ),
        )

    def _ttft_prioritize_prefill_frontier(
        self, frontier_request_ids: tuple[str, ...]
    ) -> None:
        """Traverse frontier requests first in RUNNING and WAITING slots.

        Decode-ready entries keep their exact positions. This is important in
        mixed mode: the experimental prefill policy must not silently become
        a decode admission policy. SchedulerConfig also requires the enclosing
        vLLM request queue to use FCFS while this feature is enabled.
        """
        if not frontier_request_ids:
            return
        self.running = (
            self.ttft_prefill_chunk_scheduler.prioritize_selected_candidate_slots(
                self.running,
                selected_request_ids=frontier_request_ids,
            )
        )
        if not self.waiting:
            return
        assert self.policy == SchedulingPolicy.FCFS
        waiting_requests = list(self.waiting)
        reordered_requests = (
            self.ttft_prefill_chunk_scheduler.prioritize_selected_candidate_slots(
                waiting_requests,
                selected_request_ids=frontier_request_ids,
            )
        )
        if reordered_requests == waiting_requests:
            return
        reordered_waiting = create_request_queue(self.policy)
        for request in reordered_requests:
            reordered_waiting.add_request(request)
        self.waiting = reordered_waiting

    def _ttft_prefill_request_is_selected(
        self, request: Request, selected_request_ids: frozenset[str]
    ) -> bool:
        return (
            not self.enable_ttft_prefill_chunk_scheduling
            or not self._debug_has_prefill_remaining(request)
            or request.request_id in selected_request_ids
        )

    def _configure_isolation_ring_query_budget(self) -> None:
        self.isolation_ring_query_expansion = isolation_execution_enabled(
            "RING_QUERY_EXPANSION"
        )
        self.isolation_ring_query_max_tokens = 0
        self.isolation_ring_query_scratch_bytes = 0
        if not self.isolation_ring_query_expansion:
            return
        model = self.vllm_config.model_config
        if (not self.enable_ttft_prefill_chunk_scheduling
                or self.ttft_prefill_chunk_scheduler.policy
                != PrefillChunkSchedulingPolicy.SRPT
                or not self.virtual_context_ring_bw_dynamic
                or self.virtual_context_ring_bw_policy != "safe_capacity"
                or not self.virtual_context_ring_bw_use_scheduled_headroom
                or self.parallel_config.tensor_parallel_size != 1
                or self.parallel_config.pipeline_parallel_size != 1
                or str(model.dtype) not in {"torch.float16", "torch.bfloat16"}
                or not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND")
                or not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE")
                or _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL")):
            raise ValueError(
                "Ring query expansion requires SRPT, single-GPU FP16/BF16 "
                "streaming fused Ring backend and safe-capacity admission "
                "with scheduled headroom"
            )
        budget = int(os.getenv("VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES", str(320 << 20)))
        maximum = min(self.max_num_scheduled_tokens,
                      self.virtual_context_ring_prefill_admission_tokens_max)
        if _read_bool_env("VLLM_PD_PREFILL_WORKSPACE_ENABLED"):
            maximum = min(maximum, int(os.getenv(
                "VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS", "16384")))
        limit = isolation_ring_query_token_limit(
            max_tokens=maximum, heads=model.get_num_attention_heads(self.parallel_config),
            head_size=model.get_head_size(), element_bytes=2,
            budget_bytes=budget, block_size=self.block_size,
        )
        if limit < max(self.block_size, self.virtual_context_ring_min_progress_tokens):
            raise ValueError("Ring query scratch budget cannot fit minimum progress")
        self.isolation_ring_query_scratch_bytes = budget
        self.isolation_ring_query_max_tokens = limit
        logger.info("Isolation Ring query maximum=%d scratch_budget=%d; "
                    "KV, active-Decode and cold-handoff caps still apply", limit, budget)

    def _ttft_cap_selected_prefill_chunk(
        self,
        request: Request,
        num_new_tokens: int,
        *,
        num_computed_tokens: int | None = None,
        execution_mode: str = "ring",
    ) -> int:
        if not self.enable_ttft_prefill_chunk_scheduling:
            return num_new_tokens
        quantum_capped_tokens = (
            self.ttft_prefill_chunk_scheduler.cap_prefill_tokens(
                request,
                num_new_tokens,
                num_computed_tokens=num_computed_tokens,
            )
        )
        ring_expansion = (execution_mode == "ring" and getattr(
            self, "isolation_ring_query_expansion", False))
        execution_max = (self.isolation_ring_query_max_tokens if ring_expansion
                         else self.ttft_prefill_native_chunk_max_tokens)
        if (
            (execution_mode != "native" and not ring_expansion)
            or execution_max <= 0
            or num_new_tokens <= quantum_capped_tokens
        ):
            return (min(quantum_capped_tokens, execution_max) if ring_expansion
                    else quantum_capped_tokens)

        progress = (
            self.ttft_prefill_chunk_scheduler.logical_prefill_progress(
                request
            )
            if num_computed_tokens is None
            else min(
                self.ttft_prefill_chunk_scheduler.logical_prefill_boundary(
                    request
                ),
                max(
                    num_computed_tokens,
                    request.virtual_context_deferred_logical_computed_tokens,
                ),
            )
        )
        remaining_tokens = max(
            0,
            self.ttft_prefill_chunk_scheduler.logical_prefill_boundary(request)
            - progress,
        )
        if remaining_tokens <= 0:
            return quantum_capped_tokens
        if ring_expansion:
            # Same logical boundary for selection, predictive claims and final
            # admission. A scratch ceiling may be smaller than the quantum.
            return min(num_new_tokens, execution_max, remaining_tokens)
        return max(
            quantum_capped_tokens,
            min(
                num_new_tokens,
                execution_max,
                remaining_tokens,
            ),
        )

    def _ttft_prefill_execution_mode(
        self,
        request: Request,
        native_promotion_request_ids: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Classify a selected Prefill before applying its execution cap.

        Request ordering and execution granularity are separate decisions.
        Provider-backed work keeps the Context Ring quantum, while HBM-only
        work may use the independently configured Native cap. A promoted
        request is explicitly Native because its old DRAM state remains
        visible until the matching reload plan executes.
        """

        if not self.virtual_context_enabled:
            return "native"
        if request.request_id in native_promotion_request_ids:
            return "native"
        if (
            request.virtual_context_deferred_logical_computed_tokens > 0
            or request.num_external_computed_tokens > 0
            or request.virtual_context_deferred_state
            in _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES
        ):
            return "ring"
        return "native"

    def _tbt_decode_request_is_candidate(self, request: Request) -> bool:
        return (
            request.prefill_completed_at is not None
            and request.num_output_tokens > 0
            and not request.is_finished()
            and not self._debug_needs_prefill_lane(request)
        )

    def _tbt_begin_decode_iteration(self) -> tuple[str, ...]:
        if not self.enable_tbt_decode_scheduling:
            return ()
        candidates = [
            request
            for request in itertools.chain(self.running, self.waiting)
            if not self.blocks_next_schedule(request, "decode")
            and self._tbt_decode_request_is_candidate(request)
        ]
        return self.tbt_decode_scheduler.begin_iteration(candidates)

    def _tbt_prioritize_decode_candidates(self) -> None:
        """Apply decode FCFS only within existing decode candidate slots."""

        if not self.enable_tbt_decode_scheduling:
            return
        self.running = self.tbt_decode_scheduler.prioritize_candidate_slots(
            self.running
        )
        if not self.waiting:
            return
        assert self.policy == SchedulingPolicy.FCFS
        waiting_requests = list(self.waiting)
        reordered_requests = (
            self.tbt_decode_scheduler.prioritize_candidate_slots(
                waiting_requests
            )
        )
        if reordered_requests == waiting_requests:
            return
        reordered_waiting = create_request_queue(self.policy)
        for request in reordered_requests:
            reordered_waiting.add_request(request)
        self.waiting = reordered_waiting

    def get_tbt_allocation_state(
        self,
        *,
        default_tbt_slo_ms: float = 0.0,
        feedback_window_tokens: int = 32,
        decode_request_ids: set[str] | None = None,
    ) -> dict[str, int | float | str]:
        """Return active-decode state for the EngineCore SM allocator.

        This snapshot includes decode requests that were not present in the
        latest phase-local SchedulerOutput. It avoids treating a prefill-only
        output as if no decode SLO existed while an independent decode lane is
        runnable.
        """

        now = time.monotonic()
        active = [
            request
            for request in self.requests.values()
            if self._tbt_decode_request_is_candidate(request)
            and (decode_request_ids is None or request.request_id in decode_request_ids)
        ]
        tbt_constraints: list[tuple[float, float, str, float]] = []
        for request in active:
            effective_slo_ms = (
                float(request.tbt_slo_ms)
                if request.tbt_slo_ms is not None
                else max(0.0, float(default_tbt_slo_ms))
            )
            if effective_slo_ms <= 0.0:
                continue
            elapsed_ms = (
                max(0.0, (now - request.tbt_last_token_at) * 1000.0)
                if request.tbt_last_token_at is not None
                else 0.0
            )
            tbt_constraints.append(
                (
                    effective_slo_ms,
                    elapsed_ms,
                    request.request_id,
                    float(request.tbt_last_observed_ms or 0.0),
                )
            )
        (
            critical_slo_ms,
            critical_elapsed_ms,
            critical_request_id,
            critical_last_gap_ms,
        ) = (
            min(
                tbt_constraints,
                key=lambda constraint: (
                    constraint[0] - constraint[1],
                    constraint[0],
                    -constraint[1],
                ),
            )
            if tbt_constraints
            else (0.0, 0.0, "", 0.0)
        )
        feedback = self.tbt_decode_scheduler.feedback_snapshot(
            active,
            default_tbt_slo_ms=default_tbt_slo_ms,
            window_tokens=feedback_window_tokens,
        )
        return {
            "timestamp": now,
            "decode_tbt_measurement_unit": "ms_per_output_token",
            "decode_tbt_measurement_source": "scheduler_output_token_ids",
            "decode_tbt_chunk_gap_normalization": (
                "interarrival_ms_divided_by_emitted_token_count"
            ),
            "decode_tbt_first_nonempty_chunk": "excluded_as_ttft",
            "critical_request_id": critical_request_id,
            "decode_last_token_gap_ms": critical_last_gap_ms,
            "decode_context_tokens": max(
                (request.num_computed_tokens for request in active),
                default=0,
            ),
            "decode_context_tokens_sum": sum(
                request.num_computed_tokens for request in active
            ),
            "decode_request_count": len(active),
            "decode_batch_size": len(active),
            "decode_tbt_slo_ms": critical_slo_ms,
            "decode_tbt_elapsed_ms": critical_elapsed_ms,
            "decode_min_tbt_slo_ms": min((item[0] for item in tbt_constraints), default=0.0),
            "decode_tbt_deadline_at": (
                float(self.requests[critical_request_id].tbt_last_token_at
                      if self.requests[critical_request_id].tbt_last_token_at is not None else now)
                + critical_slo_ms / 1000.0 if critical_request_id else 0.0),
            "prefill_inflight": int(
                any(
                    request.num_computed_tokens < request.num_prompt_tokens
                    for request in self.requests.values()
                    if not request.is_finished()
                )
            ),
            **feedback,
        }

    def reset_tbt_feedback_window(self) -> None:
        """Reset active-request feedback after an enforced SM switch."""

        self.tbt_decode_scheduler.reset_feedback_window(
            request
            for request in self.requests.values()
            if self._tbt_decode_request_is_candidate(request)
        )

    @staticmethod
    def _normalize_pd_phase_separated_baseline_mode(mode: str) -> str:
        mode = mode.strip().lower().replace("-", "_")
        if mode in {"", "0", "false", "no", "off", "none"}:
            return "off"
        if mode in {
            "1",
            "true",
            "yes",
            "on",
            "phase_separated",
            "phase_separated_round_robin",
            "round_robin",
            "muxwise_dynamic",
            "muxwise_like_dynamic_lanes",
        }:
            return "round_robin"
        if mode in {
            "decode",
            "decode_priority",
            "phase_separated_decode_priority",
            "no_mix",
        }:
            return "decode_priority"
        logger.warning(
            "Unknown VLLM_PD_MULTIPLEXING_BASELINE=%r; disabling baseline.",
            mode,
        )
        return "off"

    def _select_pd_decoupled_no_mix_phase(self) -> str:
        """Choose a single P/D phase for the current scheduler iteration.

        This is an experimental phase-separated baseline. It deliberately does
        not try to split a single SchedulerOutput into independent futures,
        because the scheduler advances request/KV state for all scheduled work
        together in _update_after_schedule().
        """
        running_phase_counts = self._debug_running_phase_counts()
        has_waiting_decode = any(
            self._virtual_context_waiting_decode_ready(request)
            for request in self.waiting
        )
        has_decode = (
            running_phase_counts["num_running_decode_reqs"] > 0
            or has_waiting_decode
        )
        has_prefill = (
            running_phase_counts["num_running_prefill_reqs"] > 0
            or any(
                not self._virtual_context_waiting_decode_ready(request)
                for request in self.waiting
            )
        )
        if self._pd_forced_schedule_phase is not None:
            forced_phase = self._pd_forced_schedule_phase
            if forced_phase == "decode" and has_decode:
                return "decode"
            if forced_phase == "prefill" and has_prefill:
                return "prefill"
            return "idle"
        if not has_decode and not has_prefill:
            return "idle"
        if has_decode and has_prefill and self._has_starved_prefill_request():
            return "prefill"

        mode = self.pd_phase_separated_baseline_mode
        if mode == "decode_priority":
            if has_decode:
                return "decode"
            return "prefill"

        if mode == "round_robin" and has_decode and has_prefill:
            if self._pd_phase_separated_last_phase == "decode":
                return "prefill"
            return "decode"

        if has_decode:
            return "decode"
        if has_prefill:
            return "prefill"
        return "idle"

    def _has_starved_prefill_request(self) -> bool:
        threshold = self.pd_prefill_starvation_max_decode_iters
        if threshold <= 0:
            return False
        return any(
            self._debug_needs_prefill_lane(request)
            and request.pd_residency_prefill_skip_count >= threshold
            for request in self.running
        )

    def has_pd_phase_work(self, phase: str) -> bool:
        """Return whether a logical P/D scheduler lane has work.

        This is used by the experimental separate-scheduler-lane engine path.
        The scheduler state remains shared; only the scheduling entry point is
        phase-specific.
        """
        running_phase_counts = self._debug_running_phase_counts()
        if phase == "decode":
            return (
                running_phase_counts["num_running_decode_reqs"] > 0
                or any(
                    self._virtual_context_waiting_decode_ready(request)
                    for request in self.waiting
                )
            )
        if phase == "prefill":
            return (
                running_phase_counts["num_running_prefill_reqs"] > 0
                or any(
                    not self._virtual_context_waiting_decode_ready(request)
                    for request in self.waiting
                )
            )
        return self.has_requests()

    def get_pd_capacity_retry_free_blocks(self) -> int:
        """Return the live HBM capacity used by EngineCore's retry gate."""

        return self.kv_cache_manager.block_pool.get_num_free_blocks()

    def prepare_pd_prefill_admission(self) -> bool:
        """Warm only an existing victim plan, BEFORE a Scheduler transaction.

        No plan means no speculative scan of all ordinary Native Prefills.
        Scan completion is not copy readiness; normal admission still decides
        capacity, current order and every physical ownership transition.
        """
        if not getattr(self, "isolation_bounded_prefill_planning", False):
            return True
        preparation = self._scheduled_prefill_fit_preparation
        if preparation is None:
            return True
        request_ids = {
            request_id for request_id in preparation.victim_request_ids
            if request_id in self.requests
            and request_id not in self._pd_pending_request_ids
        }
        return self.kv_cache_manager.warm_virtual_context_hbm_release_readiness(
            request_ids=request_ids, max_pages=64,
        )

    def schedule_pd_phase(self, phase: str) -> SchedulerOutput:
        """Schedule exactly one logical P/D phase under shared scheduler state."""
        normalized_phase = phase.strip().lower()
        if normalized_phase not in {"decode", "prefill"}:
            raise ValueError(f"Invalid P/D scheduler phase: {phase!r}")
        previous_forced_phase = self._pd_forced_schedule_phase
        previous_no_mix = self.pd_decoupled_no_mix
        self._pd_forced_schedule_phase = normalized_phase
        self.pd_decoupled_no_mix = True
        try:
            return self.schedule()
        finally:
            self._pd_forced_schedule_phase = previous_forced_phase
            self.pd_decoupled_no_mix = previous_no_mix

    def set_pd_pending_request_ids(self, request_ids: set[str]) -> None:
        """Track requests that still own an unrecovered phase future."""
        self._pd_pending_request_ids = set(request_ids)

    def set_pd_pending_phase_request_ids(
        self,
        request_ids_by_phase: dict[str, set[str]],
    ) -> None:
        """Track in-flight phase futures for phase-aware admission.

        ``prefill_final`` is a synthetic dependency class populated from
        ``SchedulerOutput.pd_ttft_handoff_request_ids``. Every ordinary,
        final, active-layerwise, or suspended Prefill owner remains exclusive
        until EngineCore recovers or explicitly resumes that owner. Vendor-v8
        has no pre-commit parent capability for a second same-request
        Scheduler transaction.
        """
        self._pd_pending_phase_request_ids = {
            phase: set(request_ids)
            for phase, request_ids in request_ids_by_phase.items()
            if request_ids
        }

    def blocks_next_schedule(self, request: Request, next_phase: str) -> bool:
        """Return whether unresolved work blocks this request's next phase.

        The shared scheduler advances token/KV allocation state at schedule
        time, before EngineCore can validate the final batch membership.
        Therefore every request owned by an unresolved Future is blocked from
        a new Scheduler transaction. Prefill depth two remains available to
        disjoint requests; same-request lookahead requires an exact pre-commit
        parent capability and is intentionally not supported in vendor-v8.

        This method controls scheduling only. Residency victim, release,
        reload, and allocator mutation paths must continue treating every
        request in ``_pd_pending_request_ids`` as execution-owned.
        """

        request_id = request.request_id
        active_output_owner = request.pd_output_placeholder_ledger.active
        if active_output_owner is not None:
            self._pd_pending_schedule_block_count += 1
            self._pd_pending_schedule_last_reason = (
                "explicit_output_placeholder_owner"
            )
            return True
        execution_generation = int(
            getattr(request, "kv_retention_execution_plan_generation", 0)
        )
        acked_execution_generation = int(
            getattr(
                request,
                "kv_retention_execution_plan_acked_generation",
                0,
            )
        )
        if execution_generation > acked_execution_generation:
            # Context Ring/native-reload plans additionally carry mutable
            # allocator authority. Keep the tiered-specific reason and
            # diagnostics, while the general pending-owner rule below also
            # makes native HBM Prefill request-single-flight in vendor-v8.
            self._pd_pending_schedule_block_count += 1
            self._tiered_execution_single_flight_block_count += 1
            self._tiered_execution_single_flight_last_request_id = request_id
            self._tiered_execution_single_flight_last_generation = (
                execution_generation
            )
            self._pd_pending_schedule_last_reason = (
                "tiered_execution_single_flight"
            )
            return True
        if request_id not in self._pd_pending_request_ids:
            self._pd_pending_schedule_last_reason = "no_pending"
            return False

        normalized_phase = next_phase.strip().lower()
        pending_phases = {
            phase
            for phase in ("decode", "prefill")
            if request_id in self._pd_pending_phase_request_ids.get(phase, set())
        }
        final_prefill_pending = request_id in self._pd_pending_phase_request_ids.get(
            "prefill_final", set()
        )
        suspended_prefill_pending = request_id in (
            self._pd_pending_phase_request_ids.get(
                "prefill_suspended", set()
            )
        )
        active_layerwise_prefill_pending = request_id in (
            self._pd_pending_phase_request_ids.get(
                "prefill_layerwise_active", set()
            )
        )
        if active_layerwise_prefill_pending:
            self._pd_pending_schedule_block_count += 1
            self._pd_pending_schedule_last_reason = (
                "layerwise_prefill_execution_active"
            )
            return True
        if suspended_prefill_pending:
            self._pd_pending_schedule_block_count += 1
            self._pd_pending_schedule_last_reason = (
                "layerwise_prefill_continuation_owned"
            )
            return True
        self._pd_pending_schedule_block_count += 1
        if final_prefill_pending:
            reason = "final_prefill_output_dependency"
        elif not pending_phases:
            reason = "missing_pending_phase_metadata"
        elif normalized_phase == "decode":
            reason = "decode_output_dependency"
        elif pending_phases == {"prefill"}:
            reason = "pending_prefill_future_owner"
        else:
            reason = "pending_phase_dependency"
        self._pd_pending_schedule_last_reason = reason
        return True

    def _pd_pending_decode_active(self) -> bool:
        return bool(self._pd_pending_phase_request_ids.get("decode"))

    def _pd_preemption_protection_reason(self, request: Request) -> str:
        """Keep allocator recovery from invalidating in-flight GPU work."""

        if request.request_id in self._pd_pending_request_ids:
            return "execution_future_pending"
        output_owner = request.pd_output_placeholder_ledger.active
        if output_owner is not None and not output_owner.cancelled:
            return "output_placeholder_owner_active"
        if getattr(
            request,
            "kv_retention_decode_continuation_lease_active",
            False,
        ):
            return "decode_continuation_lease_active"
        if (
            getattr(self, "pd_decoupled_no_mix", False)
            and request.num_output_tokens > 0
            and not request.is_finished()
        ):
            # A live Decode request owns an HBM history whose logical output
            # progress is not represented by the prompt-only prefill state.
            # Generic vLLM preemption would free that history and reset
            # num_computed_tokens to zero. Capacity pressure must therefore
            # wait, or use the explicit DRAM-backed suspension path, rather
            # than silently converting Decode into full-history recompute.
            return "live_decode_residency"
        return ""

    def _classify_scheduled_phase(
        self,
        num_scheduled_tokens: dict[str, int],
    ) -> str:
        has_prefill = False
        has_decode = False
        for req_id, scheduled_tokens in num_scheduled_tokens.items():
            if scheduled_tokens <= 0:
                continue
            request = self.requests.get(req_id)
            if request is None:
                continue
            prefill_remaining = self._pd_prefill_lane_remaining_tokens(
                request
            )
            prefill_tokens = min(scheduled_tokens, prefill_remaining)
            decode_tokens = max(0, scheduled_tokens - prefill_tokens)
            has_prefill = has_prefill or prefill_tokens > 0
            has_decode = has_decode or decode_tokens > 0
        if has_prefill and has_decode:
            return "mixed"
        if has_prefill:
            return "prefill"
        if has_decode:
            return "decode"
        return "idle"

    def _cap_pd_phase_tokens(
        self,
        request: Request,
        num_new_tokens: int,
        *,
        num_computed_tokens: int,
        selected_phase: str,
    ) -> int:
        """Prevent one request allocation from crossing a P/D boundary.

        Async output placeholders and resumed external prefixes can make the
        ordinary token calculation span the final prompt token and the first
        Decode token.  A phase-separated future must instead end exactly at
        the prompt boundary; the next scheduler pass owns the Decode token.
        """

        tokens = max(0, int(num_new_tokens))
        prefill_remaining = self._pd_prefill_lane_remaining_tokens(
            request,
            num_computed_tokens=num_computed_tokens,
        )
        if selected_phase == "prefill":
            return min(tokens, prefill_remaining)
        if selected_phase == "decode":
            return tokens if prefill_remaining == 0 else 0
        return 0

    def _validate_pd_phase_schedule(
        self,
        scheduled_phase: str,
        num_scheduled_tokens: dict[str, int],
    ) -> None:
        """Fail before execution if a phase-local Decode is not one-token.

        Independent P/D queues disable speculative decoding, so each live
        Decode request must contribute exactly one query token. The worker's
        strict CUDA Graph check remains a second boundary, but the scheduler
        can report the corrupt request state before launching any model work.
        """

        if not (
            getattr(self, "pd_decoupled_no_mix", False)
            and getattr(self, "pd_phase_local_model_runner_enabled", False)
            and scheduled_phase == "decode"
        ):
            return
        invalid = {
            request_id: scheduled_tokens
            for request_id, scheduled_tokens in num_scheduled_tokens.items()
            if scheduled_tokens != 1
        }
        if invalid:
            details = ", ".join(
                f"{request_id}:{scheduled_tokens}"
                for request_id, scheduled_tokens in sorted(invalid.items())
            )
            raise RuntimeError(
                "Phase-local Decode must schedule exactly one live token per "
                f"request; invalid batches: {details}."
            )

    @staticmethod
    def _debug_record_request_defer(request: Request, reason: str) -> None:
        request.deferred_iterations_count += 1
        if reason == "kv_capacity":
            request.deferred_by_kv_capacity_count += 1
        elif reason == "max_num_batched_tokens":
            request.deferred_by_max_tokens_count += 1
        elif reason == "max_num_seqs":
            request.deferred_by_max_seqs_count += 1
        elif reason == "preemption":
            request.deferred_by_preemption_count += 1
        elif reason == "residency_budget":
            request.deferred_by_residency_budget_count += 1
        else:
            request.deferred_by_policy_count += 1

    def _debug_record_waiting_defer(
        self,
        reason: str,
        reason_counts: dict[str, int],
        excluded_request_ids: set[str] | None = None,
    ) -> None:
        excluded_request_ids = excluded_request_ids or set()
        for request in self.waiting:
            if request.request_id in excluded_request_ids:
                continue
            if not self._debug_has_prefill_remaining(request):
                continue
            reason_counts[reason] += 1
            self._debug_record_request_defer(request, reason)

    def _debug_record_virtual_context_defer_fallback(
        self,
        request: Request,
        reason: str,
    ) -> None:
        request.virtual_context_defer_fallback_count += 1
        request.virtual_context_defer_last_fallback_reason = reason

    def _get_available_offloaded_tokens(
        self,
        request: Request,
        num_computed_tokens: int = 0,
    ) -> int:
        if self.connector is None:
            return 0
        get_available = getattr(
            self.connector, "get_num_available_offloaded_tokens", None
        )
        if get_available is None:
            return 0
        try:
            return int(get_available(request, num_computed_tokens))
        except (AssertionError, ValueError):
            logger.debug(
                "Unable to query offloaded token availability for request %s",
                request.request_id,
                exc_info=True,
            )
            return 0

    def _consume_scheduled_prefill_fit_preparation(
        self,
        reclaimed_request_ids: set[str],
    ) -> bool:
        """Consume a proactive plan touched by an authoritative reclaim.

        A preparation describes one immutable victim/residency snapshot. Once
        any planned victim is actually released, retaining that snapshot lets
        another scheduling path wait for HBM pages that have already become
        DRAM-authoritative. Treat the entire plan as single-use and clear it at
        the common allocator-commit boundary.
        """

        preparation = self._scheduled_prefill_fit_preparation
        if preparation is None or not reclaimed_request_ids:
            return False
        consumed_victim_ids = set(preparation.victim_request_ids).intersection(
            reclaimed_request_ids
        )
        if not consumed_victim_ids:
            return False
        self._scheduled_prefill_fit_preparation = None
        self._scheduled_prefill_fit_preparation_reclaim_consumed_count += 1
        self._scheduled_prefill_fit_preparation_consumed_victim_count += (
            len(consumed_victim_ids)
        )
        return True

    def _try_defer_running_prefill_to_tiered_provider(
        self,
        request: Request,
        logical_computed_tokens: int,
        *,
        deferred_state: str = "deferred_tiered_provider",
        max_reclaim_blocks: int | None = None,
        request_already_waiting: bool = False,
    ) -> tuple[bool, str, int]:
        self._scheduled_prefill_fit_last_release_blocks = 0
        active_owner = request.pd_output_placeholder_ledger.active
        if active_owner is not None:
            return False, "output_placeholder_owner_active", 0
        if max_reclaim_blocks is not None and max_reclaim_blocks <= 0:
            return False, "scheduled_prefill_fit_no_deficit", 0
        if not self.virtual_context_tiered_external_blocks:
            return False, "tiered_external_blocks_disabled", 0
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND"):
            return False, "tiered_ring_backend_disabled", 0
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"):
            return False, "tiered_dram_provider_disabled", 0
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM"):
            return False, "tiered_provider_copy_disabled", 0
        max_external_tokens = (
            self.virtual_context_tiered_provider_max_external_tokens
        )
        if (
            max_external_tokens > 0
            and logical_computed_tokens > max_external_tokens
        ):
            return (
                False,
                "tiered_provider_external_tokens_exceed_guard",
                logical_computed_tokens,
            )

        # Quiescent request: a fully copied partial tail is safe. Summaries are
        # coverage-only; the physical commit below still revalidates all layers.
        release_query = (
            self.kv_cache_manager.summarize_virtual_context_hbm_release_readiness
            if getattr(self, "isolation_logical_release_readiness", False)
            else self.kv_cache_manager.plan_virtual_context_hbm_release_candidates
        )
        release_candidates = release_query(
            request_ids={request.request_id}, allow_partial_pages=True,
        )
        reload_pages = self.kv_cache_manager.plan_virtual_context_hbm_reload_pages(
            request_ids={request.request_id},
        )
        block_size = resolve_virtual_context_coverage_block_size(
            self.block_size,
            itertools.chain(
                (
                    candidate.page_size_tokens
                    for candidate in release_candidates
                    if candidate.page_size_tokens > 0
                ),
                (
                    page.page_size_tokens
                    for page in reload_pages
                    if page.page_size_tokens > 0
                ),
            ),
        )
        required_pages = (
            logical_computed_tokens + block_size - 1
        ) // block_size
        partial_release = max_reclaim_blocks is not None
        release_candidate_pages = {
            candidate.logical_token_page for candidate in release_candidates
        }
        release_candidate_block_keys = {
            (
                candidate.kv_cache_group_id,
                candidate.logical_token_page,
            )
            for candidate in release_candidates
        }
        existing_dram_pages = {
            page.logical_token_page for page in reload_pages
        }
        covered_pages = release_candidate_pages | existing_dram_pages
        if partial_release:
            missing_pages = list(
                range(
                    max(
                        0,
                        int(max_reclaim_blocks or 0)
                        - len(release_candidate_block_keys),
                    )
                )
            )
        else:
            missing_pages = [
                page
                for page in range(required_pages)
                if page not in covered_pages
            ]
        request.virtual_context_tiered_transition_attempt_count += 1
        request.virtual_context_tiered_transition_last_iteration_id = (
            self._iteration_debug_index
        )
        request.virtual_context_tiered_transition_last_logical_tokens = (
            logical_computed_tokens
        )
        request.virtual_context_tiered_transition_last_block_size = block_size
        request.virtual_context_tiered_transition_last_release_candidate_records = sum(
            getattr(candidate, "candidate_records", 1)
            for candidate in release_candidates
        )
        request.virtual_context_tiered_transition_last_release_candidate_pages = len(
            release_candidate_pages
        )
        request.virtual_context_tiered_transition_last_existing_dram_records = len(
            reload_pages
        )
        request.virtual_context_tiered_transition_last_existing_dram_pages = len(
            existing_dram_pages
        )
        request.virtual_context_tiered_transition_last_required_pages = (
            required_pages
        )
        request.virtual_context_tiered_transition_last_covered_pages = len(
            covered_pages
        )
        request.virtual_context_tiered_transition_last_missing_pages = len(
            missing_pages
        )
        if missing_pages:
            contiguous_covered_pages = 0
            while contiguous_covered_pages in covered_pages:
                contiguous_covered_pages += 1
            request.virtual_context_tiered_commit_missing_pages = len(
                missing_pages
            )
            request.virtual_context_tiered_transition_last_reason = (
                "tiered_provider_copy_not_ready"
            )
            return (
                False,
                "tiered_provider_copy_not_ready",
                contiguous_covered_pages * block_size,
            )

        # All request-local validation must finish before the first allocator
        # ownership mutation. The commit call below repeats the validation as
        # a fail-fast invariant, but no scheduler state can change in between.
        request.validate_external_residency_suspension(
            logical_computed_tokens=logical_computed_tokens,
            deferred_state=deferred_state,
        )
        released_blocks = 0
        if partial_release:
            released, released_blocks = (
                self.kv_cache_manager.release_virtual_context_hbm_pages(
                    request_id=request.request_id,
                    max_logical_pages=max_reclaim_blocks,
                    allow_partial_pages=True,
                )
            )
        else:
            released = (
                self.kv_cache_manager.mark_virtual_context_hbm_released_to_dram(
                    request_ids={request.request_id},
                    allow_partial_pages=True,
                )
            )
        if not released and not existing_dram_pages:
            request.virtual_context_tiered_transition_last_reason = (
                "tiered_provider_no_release_candidates"
            )
            return False, "tiered_provider_no_release_candidates", 0

        if partial_release and released_blocks <= 0:
            request.virtual_context_tiered_transition_last_reason = (
                "tiered_provider_partial_release_failed"
            )
            return False, "tiered_provider_partial_release_failed", 0
        if partial_release:
            self._scheduled_prefill_fit_last_release_blocks = released_blocks
        if not partial_release:
            self.kv_cache_manager.free(request, release_virtual_context=False)
        if not request_already_waiting:
            self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.virtual_context_deferred_prefill_count += 1
        request.suspend_to_external_residency(
            logical_computed_tokens=logical_computed_tokens,
            deferred_state=deferred_state,
            scheduler_iteration_id=self._iteration_debug_index,
        )
        request.virtual_context_deferred_hbm_released_tokens += (
            min(logical_computed_tokens, released_blocks * self.block_size)
            if partial_release
            else logical_computed_tokens
        )
        request.virtual_context_tiered_provider_deferred_tokens += (
            logical_computed_tokens
        )
        request.virtual_context_tiered_commit_count += 1
        request.virtual_context_tiered_commit_released_pages += len(released)
        request.virtual_context_tiered_commit_existing_dram_pages += len(
            existing_dram_pages
        )
        request.virtual_context_tiered_commit_missing_pages = 0
        request.virtual_context_tiered_commit_last_iteration_id = (
            self._iteration_debug_index
        )
        request.virtual_context_tiered_commit_last_logical_tokens = (
            logical_computed_tokens
        )
        request.virtual_context_tiered_transition_last_reason = deferred_state
        request.virtual_context_preempt_reset_avoided_tokens += (
            logical_computed_tokens
        )
        if not request_already_waiting:
            self.waiting.prepend_request(request)
        self._consume_scheduled_prefill_fit_preparation({request.request_id})
        return (
            True,
            (
                "deferred_tiered_provider_partial"
                if partial_release
                else "deferred_tiered_provider"
            ),
            logical_computed_tokens,
        )

    def _try_defer_running_prefill_for_virtual_context(
        self,
        request: Request,
        *,
        max_reclaim_blocks: int | None = None,
        request_already_waiting: bool = False,
    ) -> tuple[bool, str, int]:
        if not self.virtual_context_defer_prefill:
            return False, "disabled", 0
        if request.pd_output_placeholder_ledger.active is not None:
            return False, "output_placeholder_owner_active", 0
        if self.pd_decoupled_no_mix and request.num_output_tokens > 0:
            return False, "live_decode_residency", 0
        if not self._debug_has_prefill_remaining(request):
            return False, "not_prefill", 0
        if self.is_encoder_decoder or request.has_encoder_inputs:
            return False, "encoder_unsupported", 0

        logical_computed_tokens = (
            virtual_context_residency_victim_context_tokens(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
        )
        if logical_computed_tokens <= 0:
            return False, "no_progress", 0

        (
            did_tiered_defer,
            tiered_defer_reason,
            tiered_defer_tokens,
        ) = self._try_defer_running_prefill_to_tiered_provider(
            request,
            logical_computed_tokens,
            max_reclaim_blocks=max_reclaim_blocks,
            request_already_waiting=request_already_waiting,
        )
        if did_tiered_defer:
            return True, tiered_defer_reason, tiered_defer_tokens
        if max_reclaim_blocks is not None:
            # The scheduled-prefill-fit policy is page-granular only on the
            # versioned provider path. Never turn a failed partial commit into
            # an all-request connector eviction.
            return False, tiered_defer_reason, tiered_defer_tokens

        if self.connector is None:
            if self.virtual_context_tiered_external_blocks:
                return False, tiered_defer_reason, tiered_defer_tokens
            return False, "connector_missing", 0

        offloaded_tokens = self._get_available_offloaded_tokens(request, 0)
        if offloaded_tokens < logical_computed_tokens:
            if self.virtual_context_tiered_external_blocks:
                return False, tiered_defer_reason, tiered_defer_tokens
            return False, "offload_not_ready", offloaded_tokens

        request.validate_external_residency_suspension(
            logical_computed_tokens=logical_computed_tokens,
            deferred_state="deferred_dram",
        )
        released = self.kv_cache_manager.mark_virtual_context_hbm_released_to_dram(
            request_ids={request.request_id},
            allow_partial_pages=True,
        )
        if not released:
            return False, "external_offload_no_release_candidates", 0
        self.kv_cache_manager.free(request, release_virtual_context=False)
        if not request_already_waiting:
            self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.virtual_context_deferred_prefill_count += 1
        request.suspend_to_external_residency(
            logical_computed_tokens=logical_computed_tokens,
            deferred_state="deferred_dram",
            scheduler_iteration_id=self._iteration_debug_index,
        )
        request.virtual_context_deferred_last_offloaded_tokens = offloaded_tokens
        request.virtual_context_deferred_hbm_released_tokens += (
            logical_computed_tokens
        )
        request.virtual_context_preempt_reset_avoided_tokens += (
            logical_computed_tokens
        )
        if not request_already_waiting:
            self.waiting.prepend_request(request)
        self._consume_scheduled_prefill_fit_preparation({request.request_id})
        return True, "deferred", logical_computed_tokens

    def _record_pending_residency_victim_intent(
        self,
        request: Request,
        *,
        required_blocks: int,
        target_free_blocks: int,
    ) -> None:
        """Record an advisory victim without mutating future-owned KV state."""
        logical_context_tokens = (
            virtual_context_residency_victim_context_tokens(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
        )
        reclaimable_blocks = (
            self.kv_cache_manager.get_request_resident_block_count(
                request.request_id
            )
        )
        previous = self._virtual_context_pending_victim_intent
        if previous is not None and previous.request_id == request.request_id:
            created_iteration_id = previous.created_iteration_id
            self._virtual_context_pending_victim_wait_count += 1
        else:
            if previous is not None:
                self._virtual_context_pending_victim_reranked_count += 1
            created_iteration_id = self._iteration_debug_index
            self._virtual_context_pending_victim_created_count += 1
        self._virtual_context_pending_victim_intent = (
            PendingResidencyVictimIntent(
                request_id=request.request_id,
                created_iteration_id=created_iteration_id,
                observed_logical_context_tokens=logical_context_tokens,
                observed_reclaimable_blocks=reclaimable_blocks,
                required_blocks=max(0, int(required_blocks)),
                target_free_blocks=max(0, int(target_free_blocks)),
            )
        )

    def _pending_residency_victim_debug(self) -> dict[str, int | str]:
        intent = self._virtual_context_pending_victim_intent
        return {
            "virtual_context_pending_victim_intent_enabled": int(
                self.virtual_context_pending_victim_intent_enabled
            ),
            "virtual_context_pending_victim_intent_active": int(
                intent is not None
            ),
            "virtual_context_pending_victim_request_id": (
                intent.request_id if intent is not None else ""
            ),
            "virtual_context_pending_victim_created_iteration_id": (
                intent.created_iteration_id if intent is not None else -1
            ),
            "virtual_context_pending_victim_observed_context_tokens": (
                intent.observed_logical_context_tokens
                if intent is not None
                else 0
            ),
            "virtual_context_pending_victim_observed_reclaimable_blocks": (
                intent.observed_reclaimable_blocks
                if intent is not None
                else 0
            ),
            "virtual_context_pending_victim_required_blocks": (
                intent.required_blocks if intent is not None else 0
            ),
            "virtual_context_pending_victim_target_free_blocks": (
                intent.target_free_blocks if intent is not None else 0
            ),
            "virtual_context_pending_victim_created_count": (
                self._virtual_context_pending_victim_created_count
            ),
            "virtual_context_pending_victim_wait_count": (
                self._virtual_context_pending_victim_wait_count
            ),
            "virtual_context_pending_victim_revalidated_count": (
                self._virtual_context_pending_victim_revalidated_count
            ),
            "virtual_context_pending_victim_committed_count": (
                self._virtual_context_pending_victim_committed_count
            ),
            "virtual_context_pending_victim_reranked_count": (
                self._virtual_context_pending_victim_reranked_count
            ),
            "virtual_context_pending_victim_cancelled_count": (
                self._virtual_context_pending_victim_cancelled_count
            ),
        }

    def _active_virtual_context_prefill_requests(self) -> tuple[Request, ...]:
        """Return unique active prefills across scheduler queue states.

        An active request may be in either RUNNING or WAITING while SRPT moves
        it away from the current compute batch. Both queue states therefore
        remain eligible for request-granular victim selection as long as the
        request still owns HBM blocks.
        """

        active: list[Request] = []
        seen_request_ids: set[str] = set()
        for request in itertools.chain(self.running, self.waiting):
            if request.request_id in seen_request_ids:
                continue
            seen_request_ids.add(request.request_id)
            logical_tokens = virtual_context_residency_victim_context_tokens(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
            if (
                logical_tokens < request.num_prompt_tokens
                and not (
                    self.pd_decoupled_no_mix
                    and request.num_output_tokens > 0
                )
            ):
                active.append(request)
        return tuple(active)

    def _ordered_virtual_context_prefill_reclaim_candidates(
        self,
        *,
        required_blocks: int,
        excluded_request_ids: set[str],
    ) -> list[tuple[Request, ResidencyVictimCandidate]]:
        """Pair ordered active prefills with their immutable snapshots."""

        candidates = [
            request
            for request in self._active_virtual_context_prefill_requests()
            if (
                request.request_id not in excluded_request_ids
                and (
                    self.virtual_context_pending_victim_intent_enabled
                    or request.request_id not in self._pd_pending_request_ids
                )
                and self._debug_has_prefill_remaining(request)
            )
        ]
        snapshots = {
            request.request_id: snapshot
            for request in candidates
            if (
                snapshot
                := self._virtual_context_prefill_reclaim_candidate_snapshot(
                    request
                )
            )
            is not None
        }
        candidates = [
            request
            for request in candidates
            if request.request_id in snapshots
        ]
        if (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.LONGEST_LOGICAL_CONTEXT_FIRST
        ):
            candidates.sort(
                key=lambda request: virtual_context_residency_victim_sort_key(
                    snapshots[request.request_id].logical_context_tokens,
                    request.arrival_time,
                    request.request_id,
                )
            )
            return [
                (request, snapshots[request.request_id])
                for request in candidates
            ]

        if (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            requests_by_id = {
                request.request_id: request for request in candidates
            }
            best_fit = select_best_fit_whole_request_victims(
                snapshots.values(),
                required_blocks=required_blocks,
            )
            return [
                (requests_by_id[snapshot.request_id], snapshot)
                for snapshot in best_fit
            ]

        ordered = order_residency_victims(
            list(snapshots.values()),
            policy=self.virtual_context_residency_victim_policy,
            required_blocks=required_blocks,
            ring_chunk_tokens=(
                self._virtual_context_residency_fixed_ring_chunk_tokens()
            ),
        )
        requests_by_id = {
            request.request_id: request for request in candidates
        }
        return [
            (requests_by_id[snapshot.request_id], snapshot)
            for snapshot in ordered
        ]

    def _virtual_context_prefill_reclaim_candidate_snapshot(
        self,
        request: Request,
    ) -> ResidencyVictimCandidate | None:
        """Return an immutable victim snapshot, or reject an unsafe JIT victim.

        Legacy policies preserve their existing proposal behavior and let the
        authoritative commit path test provider readiness. ``jit_min_copy`` is
        stricter: it ranks only quiescent running prefills whose complete
        logical prefix already has DRAM-provider coverage. This prevents the
        copy-efficiency score from selecting a request that cannot be safely
        virtualized at this scheduler boundary.
        """

        logical_context_tokens = (
            virtual_context_residency_victim_context_tokens(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
        )
        reclaimable_blocks = (
            self.kv_cache_manager.get_request_resident_block_count(
                request.request_id
            )
        )
        base = ResidencyVictimCandidate(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            prompt_tokens=request.num_prompt_tokens,
            logical_context_tokens=logical_context_tokens,
            estimated_reclaimable_blocks=reclaimable_blocks,
        )
        if self.virtual_context_residency_victim_policy not in {
            ResidencyVictimPolicy.JIT_MIN_COPY,
            ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT,
        }:
            return base

        if (
            request.request_id in self._pd_pending_request_ids
            or not self._debug_has_prefill_remaining(request)
            or self.is_encoder_decoder
            or request.has_encoder_inputs
            or logical_context_tokens <= 0
            or reclaimable_blocks <= 0
        ):
            return None
        max_external_tokens = (
            self.virtual_context_tiered_provider_max_external_tokens
        )
        if (
            max_external_tokens > 0
            and logical_context_tokens > max_external_tokens
        ):
            return None

        release_query = (
            self.kv_cache_manager.summarize_virtual_context_hbm_release_readiness
            if getattr(self, "isolation_logical_release_readiness", False)
            else self.kv_cache_manager.plan_virtual_context_hbm_release_candidates
        )
        release_candidates = release_query(
            request_ids={request.request_id}, allow_partial_pages=True,
        )
        reload_pages = (
            self.kv_cache_manager.plan_virtual_context_hbm_reload_pages(
                request_ids={request.request_id},
            )
        )
        coverage_block_size = resolve_virtual_context_coverage_block_size(
            self.block_size,
            itertools.chain(
                (
                    candidate.page_size_tokens
                    for candidate in release_candidates
                    if candidate.page_size_tokens > 0
                ),
                (
                    page.page_size_tokens
                    for page in reload_pages
                    if page.page_size_tokens > 0
                ),
            ),
        )
        required_pages = (
            logical_context_tokens + coverage_block_size - 1
        ) // coverage_block_size
        if (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            covered_keys = {
                (
                    candidate.kv_cache_group_id,
                    candidate.logical_token_page,
                )
                for candidate in release_candidates
            } | {
                (page.kv_cache_group_id, page.logical_token_page)
                for page in reload_pages
            }
            required_keys = {
                (group_id, logical_page)
                for group_id, block_ids in enumerate(
                    self.kv_cache_manager.get_block_ids(request.request_id)
                )
                if block_ids
                for logical_page in range(required_pages)
            }
            # scheduled_prefill_fit deliberately uses the legacy
            # whole-request transition. A victim is commit-ready only after
            # every logical page in its durable prefix has DRAM coverage; a
            # merely sufficient subset must never produce a retained mixed
            # allocator table.
            if not required_keys or not required_keys.issubset(covered_keys):
                return None
            if not release_candidates and not reload_pages:
                return None
        else:
            covered_pages = {
                candidate.logical_token_page
                for candidate in release_candidates
            } | {page.logical_token_page for page in reload_pages}
            if any(page not in covered_pages for page in range(required_pages)):
                return None
            if not release_candidates and not reload_pages:
                return None

        (
            d2h_eviction_tokens,
            ring_staging_h2d_tokens,
            decode_reload_h2d_tokens,
        ) = estimate_jit_min_copy_components(
            logical_context_tokens,
            request.num_prompt_tokens,
            self._virtual_context_residency_fixed_ring_chunk_tokens(),
        )
        return ResidencyVictimCandidate(
            request_id=base.request_id,
            arrival_time=base.arrival_time,
            prompt_tokens=base.prompt_tokens,
            logical_context_tokens=base.logical_context_tokens,
            estimated_reclaimable_blocks=base.estimated_reclaimable_blocks,
            estimated_d2h_eviction_tokens=d2h_eviction_tokens,
            estimated_ring_staging_h2d_tokens=ring_staging_h2d_tokens,
            estimated_decode_reload_h2d_tokens=decode_reload_h2d_tokens,
        )

    def _maximum_immediate_prefill_reclaim_blocks(
        self,
        *,
        excluded_request_ids: set[str],
        prepared_request_ids: tuple[str, ...] = (),
    ) -> int:
        """Return a cheap upper bound for an immediate Prefill reclaim.

        This deliberately does not inspect provider pages. It is the first
        stage of Decode-handoff admission: if even every quiescent resident
        Prefill block cannot satisfy the target, an expensive copy-readiness
        scan and a destructive partial reclaim are both pointless.
        """

        prepared = set(prepared_request_ids)
        return sum(
            max(
                0,
                self.kv_cache_manager.get_request_resident_block_count(
                    request.request_id
                ),
            )
            for request in self._active_virtual_context_prefill_requests()
            if (
                request.request_id not in excluded_request_ids
                and request.request_id not in self._pd_pending_request_ids
                and (not prepared or request.request_id in prepared)
            )
        )

    def _immediate_prefill_reclaim_state(
        self,
        *,
        excluded_request_ids: set[str],
        prepared_request_ids: tuple[str, ...] = (),
    ) -> tuple[tuple[str, int, int, bool], ...]:
        """Return the cheap state key that invalidates an infeasible result."""

        prepared = set(prepared_request_ids)
        return tuple(
            sorted(
                (
                    request.request_id,
                    int(request.num_computed_tokens),
                    int(
                        request.virtual_context_deferred_logical_computed_tokens
                    ),
                    request.request_id in self._pd_pending_request_ids,
                )
                for request in self._active_virtual_context_prefill_requests()
                if (
                    request.request_id not in excluded_request_ids
                    and (not prepared or request.request_id in prepared)
                )
            )
        )

    @staticmethod
    def _restrict_reclaim_candidates_to_preparation(
        candidates: list[tuple[Request, ResidencyVictimCandidate]],
        prepared_request_ids: tuple[str, ...],
    ) -> list[tuple[Request, ResidencyVictimCandidate]]:
        if not prepared_request_ids:
            return candidates
        prepared_rank = {
            request_id: rank
            for rank, request_id in enumerate(prepared_request_ids)
        }
        return sorted(
            (
                candidate
                for candidate in candidates
                if candidate[0].request_id in prepared_rank
            ),
            key=lambda candidate: prepared_rank[candidate[0].request_id],
        )

    def _reclaim_prefill_victims_until_hbm_target(
        self,
        *,
        target_free_blocks: int,
        excluded_request_ids: set[str],
        prepared_request_ids: tuple[str, ...] = (),
        require_full_target_before_reclaim: bool = False,
        reclaim_feasibility_cache_key: str = "",
    ) -> tuple[
        int,
        int,
        list[ReleasedPrefillVictim],
        set[str],
        int,
    ]:
        """Release copy-ready prefill victims until an HBM target is met.

        Ordinary failed candidates are skipped for this immutable scheduling
        boundary. A selected future-owned candidate instead records an intent
        and stops mutation until a later boundary reranks it; a revalidated
        intent with incomplete provider coverage is retained for snapshot
        retry rather than bypassed.
        After every successful release the policy is rerun against the smaller
        shortfall instead of committing a fixed one-victim proposal.
        Decode handoff may require a full-target preflight so an impossible
        admission never destroys useful Prefill residency one victim at a
        time.
        """

        attempts = 0
        failures = 0
        released_victims: list[ReleasedPrefillVictim] = []
        reasons: set[str] = set()
        attempted_request_ids: set[str] = set()
        target = max(0, int(target_free_blocks))
        prechecked_candidates: list[
            tuple[Request, ResidencyVictimCandidate]
        ] | None = None
        validated_transaction_request_ids: tuple[str, ...] = ()

        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        transaction_initial_free_blocks = free_blocks
        if (
            require_full_target_before_reclaim
            and reclaim_feasibility_cache_key
            and free_blocks >= target
        ):
            self._native_decode_handoff_reclaim_infeasible_cache.pop(
                reclaim_feasibility_cache_key,
                None,
            )
        if require_full_target_before_reclaim and free_blocks < target:
            prefill_state = self._immediate_prefill_reclaim_state(
                excluded_request_ids=excluded_request_ids,
                prepared_request_ids=prepared_request_ids,
            )
            cached_infeasible = (
                self._native_decode_handoff_reclaim_infeasible_cache.get(
                    reclaim_feasibility_cache_key
                )
                if reclaim_feasibility_cache_key
                else None
            )
            if (
                cached_infeasible is not None
                and target >= cached_infeasible.target_free_blocks
                and free_blocks <= cached_infeasible.observed_free_blocks
                and prefill_state == cached_infeasible.prefill_state
                and prepared_request_ids
                == cached_infeasible.prepared_request_ids
            ):
                self._native_decode_handoff_reclaim_infeasible_cache_hit_count += (
                    1
                )
                reasons.add("aggregate_prefill_reclaim_insufficient")
                return 0, 0, [], reasons, free_blocks

            self._native_decode_handoff_reclaim_feasibility_check_count += 1
            required_blocks = target - free_blocks
            maximum_reclaim_blocks = (
                self._maximum_immediate_prefill_reclaim_blocks(
                    excluded_request_ids=excluded_request_ids,
                    prepared_request_ids=prepared_request_ids,
                )
            )
            if maximum_reclaim_blocks < required_blocks:
                self._native_decode_handoff_reclaim_infeasible_count += 1
                self._native_decode_handoff_reclaim_avoided_partial_blocks += (
                    maximum_reclaim_blocks
                )
                if reclaim_feasibility_cache_key:
                    self._native_decode_handoff_reclaim_infeasible_cache[
                        reclaim_feasibility_cache_key
                    ] = DecodeHandoffReclaimInfeasibleObservation(
                        target_free_blocks=target,
                        observed_free_blocks=free_blocks,
                        prefill_state=prefill_state,
                        prepared_request_ids=prepared_request_ids,
                        maximum_reclaim_blocks=maximum_reclaim_blocks,
                    )
                reasons.add("aggregate_prefill_reclaim_insufficient")
                return 0, 0, [], reasons, free_blocks

            if reclaim_feasibility_cache_key:
                self._native_decode_handoff_reclaim_infeasible_cache.pop(
                    reclaim_feasibility_cache_key,
                    None,
                )

            # The resident-block upper bound can include pages whose D2H copy
            # is not authoritative yet. Resolve every copy-ready candidate
            # before the first allocator mutation and reject the transaction
            # when that complete set still cannot reach the target.
            prechecked_candidates = (
                self._ordered_virtual_context_prefill_reclaim_candidates(
                    required_blocks=required_blocks,
                    excluded_request_ids=excluded_request_ids,
                )
            )
            prechecked_candidates = (
                self._restrict_reclaim_candidates_to_preparation(
                    prechecked_candidates,
                    prepared_request_ids,
                )
            )
            copy_ready_reclaim_blocks = sum(
                max(0, candidate.estimated_reclaimable_blocks)
                for _, candidate in prechecked_candidates
            )
            if copy_ready_reclaim_blocks < required_blocks:
                self._native_decode_handoff_reclaim_infeasible_count += 1
                self._native_decode_handoff_reclaim_avoided_partial_blocks += (
                    copy_ready_reclaim_blocks
                )
                reasons.add("aggregate_copy_ready_reclaim_insufficient")
                return 0, 0, [], reasons, free_blocks
            validated_transaction_request_ids = tuple(
                request.request_id for request, _ in prechecked_candidates
            )

        while True:
            free_blocks = (
                self.kv_cache_manager.block_pool.get_num_free_blocks()
            )
            if free_blocks >= target:
                if self._virtual_context_pending_victim_intent is not None:
                    self._virtual_context_pending_victim_intent = None
                    self._virtual_context_pending_victim_cancelled_count += 1
                break
            required_blocks = target - free_blocks
            if prechecked_candidates is not None:
                candidates = prechecked_candidates
                prechecked_candidates = None
            else:
                candidates = (
                    self._ordered_virtual_context_prefill_reclaim_candidates(
                        required_blocks=required_blocks,
                        excluded_request_ids=(
                            excluded_request_ids | attempted_request_ids
                        ),
                    )
                )
                candidates = (
                    self._restrict_reclaim_candidates_to_preparation(
                        candidates,
                        validated_transaction_request_ids
                        or prepared_request_ids,
                    )
                )
            if not candidates:
                if self._virtual_context_pending_victim_intent is not None:
                    self._virtual_context_pending_victim_intent = None
                    self._virtual_context_pending_victim_cancelled_count += 1
                break

            released_this_round = False
            for candidate, candidate_snapshot in candidates:
                candidate_in_running = candidate in self.running
                candidate_in_waiting = candidate in self.waiting
                if not candidate_in_running and not candidate_in_waiting:
                    reasons.add("victim_not_active")
                    failures += 1
                    continue
                if (
                    self.virtual_context_pending_victim_intent_enabled
                    and candidate.request_id in self._pd_pending_request_ids
                ):
                    self._record_pending_residency_victim_intent(
                        candidate,
                        required_blocks=required_blocks,
                        target_free_blocks=target,
                    )
                    reasons.add("pending_victim_async_d2h")
                    break

                pending_intent = self._virtual_context_pending_victim_intent
                revalidating_intent = (
                    pending_intent is not None
                    and pending_intent.request_id == candidate.request_id
                )
                if pending_intent is not None and not revalidating_intent:
                    self._virtual_context_pending_victim_intent = None
                    self._virtual_context_pending_victim_reranked_count += 1
                elif revalidating_intent:
                    self._virtual_context_pending_victim_revalidated_count += 1
                attempted_request_ids.add(candidate.request_id)
                attempts += 1
                free_before = (
                    self.kv_cache_manager.block_pool.get_num_free_blocks()
                )
                did_defer, reason, defer_tokens = (
                    self._try_defer_running_prefill_for_virtual_context(
                        candidate,
                        # Reclaim is request-granular. The outer loop stops
                        # after the sum of whole victims satisfies the target,
                        # accepting bounded over-reclaim from the final victim.
                        max_reclaim_blocks=None,
                        request_already_waiting=(
                            candidate_in_waiting and not candidate_in_running
                        ),
                    )
                )
                reasons.add(reason)
                if not did_defer:
                    failures += 1
                    if (
                        revalidating_intent
                        and reason
                        in {
                            "tiered_provider_copy_not_ready",
                            "tiered_provider_no_release_candidates",
                            "offload_not_ready",
                        }
                    ):
                        reasons.add("pending_victim_snapshot_retry")
                        break
                    continue
                if revalidating_intent:
                    self._virtual_context_pending_victim_intent = None
                    self._virtual_context_pending_victim_committed_count += 1
                if candidate_in_running:
                    self.running.remove(candidate)
                elif candidate not in self.waiting:
                    raise RuntimeError(
                        "waiting residency victim left its queue during "
                        f"commit: request={candidate.request_id}"
                    )
                free_after = (
                    self.kv_cache_manager.block_pool.get_num_free_blocks()
                )
                released_victims.append(
                    ReleasedPrefillVictim(
                        request_id=candidate.request_id,
                        deferred_tokens=defer_tokens,
                        reclaimed_blocks=max(0, free_after - free_before),
                        candidate_snapshot=candidate_snapshot,
                        released_logical_blocks=(
                            self._scheduled_prefill_fit_last_release_blocks
                        ),
                    )
                )
                released_this_round = True
                break
            if not released_this_round:
                break

        final_free_blocks = (
            self.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        if (
            require_full_target_before_reclaim
            and validated_transaction_request_ids
            and released_victims
            and final_free_blocks < target
        ):
            released_request_ids = tuple(
                victim.request_id for victim in released_victims
            )
            raise TieredAllocatorInvariantError(
                "Decode-handoff reclaim partially committed after a "
                "successful full-target preflight: "
                f"target_free_blocks={target}, "
                f"initial_free_blocks={transaction_initial_free_blocks}, "
                f"final_free_blocks={final_free_blocks}, "
                f"missing_free_blocks={target - final_free_blocks}, "
                "validated_victims="
                f"{validated_transaction_request_ids}, "
                f"released_victims={released_request_ids}, "
                f"attempted_victims={tuple(sorted(attempted_request_ids))}, "
                f"reasons={tuple(sorted(reasons))}"
            )

        return (
            attempts,
            failures,
            released_victims,
            reasons,
            final_free_blocks,
        )

    def _needs_virtual_context_worker_rehydrate(self, request: Request) -> bool:
        """Return true when a PREEMPTED request lost worker-local state.

        Residency-deferred virtual-context requests are explicitly removed from
        the model runner persistent batch. When such a request is admitted
        again, sending it through scheduled_cached_reqs would make the worker
        look up a deleted CachedRequestState. Route it through NewRequestData
        once so the worker can rebuild request-local state while preserving the
        scheduler's logical prefill progress.
        """
        if not self.virtual_context_defer_prefill:
            return False
        if request.status != RequestStatus.PREEMPTED:
            return False
        if request.virtual_context_deferred_logical_computed_tokens <= 0:
            return False
        return request.virtual_context_deferred_state in (
            _VIRTUAL_CONTEXT_REHYDRATE_STATES
        )

    def _can_resume_with_tiered_virtual_context(
        self,
        request: Request,
        deferred_load_target_tokens: int,
    ) -> bool:
        """Return true when virtual-context state can bypass KVConnector.

        A request may be admitted from the tiered provider path, run one or more
        chunks, and then be deferred again while it still has DRAM-authoritative
        logical pages. In that case the state can be ``tiered_provider_resumed``
        instead of the initial ``deferred_tiered_provider`` marker. Treat both
        as provider-backed resume states so we do not fall back to a missing
        KVConnector. If the marker was lost but the sidecar still has
        reload-required pages, allow the same path; the attention backend will
        either reload to HBM when there is room or use the context-ring path.
        """
        if deferred_load_target_tokens <= 0:
            return False
        if not self.virtual_context_tiered_external_blocks:
            return False
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"):
            return False
        if not (
            _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND")
            or self.virtual_context_reload_to_hbm
        ):
            return False
        if request.virtual_context_deferred_state in (
            _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES
        ):
            return True
        return (
            self._virtual_context_reload_required_page_count(
                [request.request_id],
                "prefill",
            )
            > 0
        )

    def _virtual_context_hbm_relief_min_free_blocks(self) -> int:
        configured = self.virtual_context_recompute_on_hbm_relief_min_free_blocks
        if configured > 0:
            return configured
        token_floor = max(
            self.virtual_context_ring_prefill_admission_tokens,
            self.pd_ring_min_prefill_tokens_with_decode,
            self.block_size,
        )
        return max(1, (token_floor + self.block_size - 1) // self.block_size)

    def _should_reload_virtual_context_to_hbm(
        self,
        request: Request,
        external_tokens: int,
        free_blocks: int,
    ) -> tuple[bool, int, int, int]:
        if not self.virtual_context_reload_to_hbm:
            return False, 0, 0, 0
        if external_tokens <= 0:
            return False, 0, 0, 0
        logical_computed_tokens = max(
            request.num_computed_tokens,
            request.virtual_context_deferred_logical_computed_tokens,
            external_tokens,
        )
        decode_ready = (
            logical_computed_tokens >= request.num_prompt_tokens
            and request.num_output_tokens > 0
        )
        if not self._debug_has_prefill_remaining(request) and not decode_ready:
            return False, 0, 0, 0
        reload_pages = self._virtual_context_reload_required_page_count(
            [request.request_id],
            "prefill",
        )
        if reload_pages <= 0:
            return False, 0, 0, 0
        # A decode-ready request never falls through to virtual-external/ring
        # execution. It either obtains a complete native HBM block table here
        # or remains waiting until enough blocks are available.
        if decode_ready:
            # Share the Native physical-block estimator, not a layer-copy
            # count. Preserve this path's existing reserve: imposing the
            # Running lane's larger runway here can strand a near-capacity
            # WAITING request that previously fit on an otherwise empty GPU.
            required_blocks = (
                self.kv_cache_manager.estimate_num_blocks_to_allocate(
                    request, 0,
                    num_external_computed_tokens=max(
                        0, logical_computed_tokens - request.num_computed_tokens,
                    ),
                    require_native_hbm=True,
                )
            )
            extra_reserve_blocks = (
                self.virtual_context_reload_to_hbm_min_free_blocks
                or self.virtual_context_staging_reserved_blocks
            )
            min_free_blocks = required_blocks + extra_reserve_blocks
            return (
                free_blocks >= min_free_blocks,
                reload_pages,
                required_blocks,
                min_free_blocks,
            )

        # Decode hotset has priority over prefill HBM reload. When decode is
        # active, a provider-backed prefill should keep its existing context as
        # virtual external pages and let the context-ring backend materialize
        # those pages through staging buffers. Reloading the full prefix here
        # would consume the same HBM capacity that the ring path is meant to
        # avoid.
        if self.virtual_context_prefill_ring_required_mode == "always":
            return False, reload_pages, 0, 0
        if (
            self.virtual_context_prefill_ring_required_mode == "pressure"
            and (
                self._debug_running_decode_resident_blocks() > 0
                or self._virtual_context_waiting_decode_service_candidate()
                is not None
            )
        ):
            return False, reload_pages, 0, 0
        required_blocks = max(
            1,
            (external_tokens + self.block_size - 1) // self.block_size,
        )
        if self.virtual_context_prefill_ring_required_mode == "pressure":
            if (
                self.virtual_context_residency_victim_policy
                == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
            ):
                claim = self._scheduled_prefill_fit_capacity_claim(
                    request,
                    ring_token_cap=(
                        self._dynamic_virtual_context_ring_prefill_admission_tokens()
                        or self._virtual_context_residency_fixed_ring_chunk_tokens()
                    ),
                    full_native=True,
                    require_full_prompt=True,
                )
                return (
                    claim.reclaim_blocks <= 0,
                    reload_pages,
                    claim.raw_required_blocks,
                    claim.required_blocks_with_reserve,
                )
            ring_reserved_blocks = (
                self.virtual_context_staging_reserved_blocks
                + self.virtual_context_ring_min_free_blocks
                + max(
                    self.virtual_context_decode_hotset_reserve_blocks,
                    self.virtual_context_ring_bw_decode_growth_blocks,
                )
                + self.virtual_context_ring_bw_safety_blocks
            )
            ring_reserved_blocks = max(
                ring_reserved_blocks,
                self.virtual_context_reload_to_hbm_min_free_blocks,
            )
            min_free_blocks = (
                virtual_context_pressure_native_reload_min_free_blocks(
                    required_blocks,
                    ring_reserved_blocks,
                    self.virtual_context_ring_min_progress_tokens,
                    self.block_size,
                )
            )
            if free_blocks < min_free_blocks:
                return (
                    False,
                    reload_pages,
                    required_blocks,
                    min_free_blocks,
                )
            return True, reload_pages, required_blocks, min_free_blocks
        extra_reserve_blocks = (
            self.virtual_context_reload_to_hbm_min_free_blocks
            or (
                self.virtual_context_decode_hotset_reserve_blocks
                + self.virtual_context_staging_reserved_blocks
            )
        )
        min_free_blocks = required_blocks + extra_reserve_blocks
        if free_blocks < min_free_blocks:
            return False, reload_pages, required_blocks, min_free_blocks
        return True, reload_pages, required_blocks, min_free_blocks

    def _prefill_needs_capacity_floor_snapshot(
        self,
        request: Request,
        scheduled_tokens: int,
    ) -> bool:
        """Request repeated snapshots throughout the pre-floor pressure runway."""
        if (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            # This policy emits an exact page plan after current allocations
            # are known. Never widen it back to a request-wide force-all copy.
            return False
        pending_intent = self._virtual_context_pending_victim_intent
        pending_intent_snapshot = (
            self.virtual_context_pending_victim_intent_enabled
            and pending_intent is not None
            and pending_intent.request_id == request.request_id
        )
        if (
            not self.virtual_context_dram_register_pressure_only
            and not pending_intent_snapshot
        ):
            return False
        if not self.virtual_context_ring_residency_admission:
            return False
        if not self.virtual_context_tiered_external_blocks:
            return False
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND"):
            return False
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"):
            return False
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM"):
            return False
        if not self._debug_has_prefill_remaining(request):
            return False
        if self._virtual_context_reload_required_page_count(
            [request.request_id],
            "prefill",
        ) > 0:
            return False
        (
            _pressure_threshold_blocks,
            pressure_bootstrap_blocks,
            pressure_ring_blocks,
            _pressure_reserved_blocks,
        ) = self._context_ring_pressure_threshold_blocks()
        snapshot_lead_tokens = max(
            self.virtual_context_ring_min_progress_tokens,
            (
                pressure_bootstrap_blocks + pressure_ring_blocks
            )
            * self.block_size,
        )
        return (
            pending_intent_snapshot
            or virtual_context_prefill_needs_capacity_floor_snapshot(
                self._virtual_context_ring_bw_last_safe_tokens,
                scheduled_tokens,
                snapshot_lead_tokens,
            )
        )

    @staticmethod
    def _clear_virtual_context_deferred_state_for_recompute(
        request: Request,
    ) -> None:
        request.virtual_context_deferred_logical_computed_tokens = 0
        request.virtual_context_deferred_state = ""
        request.virtual_context_deferred_last_iteration_id = -1
        request.virtual_context_deferred_last_offloaded_tokens = 0
        request.num_external_computed_tokens = 0
        request.num_cached_tokens = -1

    def _should_recompute_virtual_context_on_hbm_relief(
        self,
        request: Request,
        free_blocks: int,
    ) -> tuple[bool, int, int]:
        if self.virtual_context_reload_to_hbm:
            return False, 0, 0
        if not self.virtual_context_recompute_on_hbm_relief:
            return False, 0, 0
        if not self._debug_has_prefill_remaining(request):
            return False, 0, 0
        if request.virtual_context_deferred_logical_computed_tokens <= 0:
            return False, 0, 0
        reload_pages = self._virtual_context_reload_required_page_count(
            [request.request_id],
            "prefill",
        )
        if reload_pages <= 0:
            return False, 0, 0
        min_free_blocks = self._virtual_context_hbm_relief_min_free_blocks()
        if free_blocks < min_free_blocks:
            return False, reload_pages, min_free_blocks
        return True, reload_pages, min_free_blocks

    def _prepare_waiting_virtual_context_hbm_relief_recompute(
        self,
        request: Request,
    ) -> tuple[bool, int, int, int]:
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        should_recompute, reload_pages, min_free_blocks = (
            self._should_recompute_virtual_context_on_hbm_relief(
                request,
                free_blocks,
            )
        )
        if not should_recompute:
            return False, reload_pages, min_free_blocks, free_blocks

        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        self._clear_virtual_context_deferred_state_for_recompute(request)
        request.status = RequestStatus.WAITING
        request.num_computed_tokens = 0
        request.spec_token_ids.clear()
        return True, reload_pages, min_free_blocks, free_blocks

    def _debug_record_virtual_context_residency_admission_defer(
        self,
        request: Request,
        phase: str,
        required_blocks: int,
        free_blocks: int,
        decode_reserve_blocks: int,
        staging_blocks: int,
    ) -> None:
        request.virtual_context_residency_admission_defer_count += 1
        request.virtual_context_residency_admission_last_iteration_id = (
            self._iteration_debug_index
        )
        request.virtual_context_residency_admission_last_phase = phase
        request.virtual_context_residency_admission_last_required_blocks = (
            required_blocks
        )
        request.virtual_context_residency_admission_last_free_blocks = free_blocks
        request.virtual_context_residency_admission_last_decode_reserve_blocks = (
            decode_reserve_blocks
        )
        request.virtual_context_residency_admission_last_staging_blocks = (
            staging_blocks
        )

    def _check_virtual_context_prefill_residency_admission(
        self,
        request: Request,
        num_new_tokens: int,
        phase: str,
        *,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_virtual_external_computed_tokens: int = 0,
        num_encoder_tokens: int = 0,
        scheduled_tokens: dict[str, int] | None = None,
    ) -> tuple[bool, int, int, int, int, int, int, int]:
        batch_budget = getattr(self, "isolation_prefill_batch_kv_budget", False)
        if not self.virtual_context_residency_admission and not batch_budget:
            return True, 0, 0, 0, 0, 0, 0, 0
        if num_new_tokens <= 0 or not self._debug_has_prefill_remaining(request):
            return True, 0, 0, 0, 0, 0, 0, 0

        raw_required_blocks = self.kv_cache_manager.estimate_num_blocks_to_allocate(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            num_virtual_external_computed_tokens=(
                num_virtual_external_computed_tokens
            ),
            num_encoder_tokens=num_encoder_tokens,
        )
        if raw_required_blocks <= 0 and not batch_budget:
            return True, raw_required_blocks, 0, 0, 0, raw_required_blocks, 0, 0
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if batch_budget and not self._virtual_context_waiting_decode_ready(request):
            logical_tokens = max(
                request.num_computed_tokens + num_new_computed_tokens
                + num_external_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
            imminent_decode = self._isolation_prefill_starts_decode(
                request, num_new_tokens, logical_tokens=logical_tokens,
            ) or any(
                self._isolation_prefill_starts_decode(self.requests[rid], count)
                for rid, count in (scheduled_tokens or {}).items()
                if rid in self.requests and rid != request.request_id
                and not self.requests[rid].is_finished()
            )
            headroom = (
                self.virtual_context_scheduled_prefill_fit_headroom_blocks
                if imminent_decode or self._scheduled_prefill_fit_active_decode()
                else 0
            )
            hard_reserve = (
                self.virtual_context_staging_reserved_blocks
                + self.virtual_context_ring_min_free_blocks
                + self.virtual_context_decode_hotset_reserve_blocks
                + headroom
            )
            if getattr(self, "virtual_context_reload_to_hbm", False):
                hard_reserve += (
                    self._virtual_context_waiting_decode_service_reserve()
                )
            # The allocator already debited preceding successful rows. Never
            # subtract their estimates again, or reserve pages for failed rows.
            feasible = free_blocks - raw_required_blocks >= hard_reserve
            debug = self._isolation_prefill_batch_budget_debug
            debug["isolation_prefill_batch_kv_checks"] = (
                debug.get("isolation_prefill_batch_kv_checks", 0) + 1)
            debug["isolation_prefill_batch_kv_blocked"] = (
                debug.get("isolation_prefill_batch_kv_blocked", 0) + int(not feasible))
            debug["isolation_prefill_batch_imminent_decode"] = max(
                debug.get("isolation_prefill_batch_imminent_decode", 0),
                int(imminent_decode))
            debug["isolation_prefill_batch_kv_last_free_blocks"] = free_blocks
            debug["isolation_prefill_batch_kv_last_required_blocks"] = raw_required_blocks
            debug["isolation_prefill_batch_kv_last_hard_reserve_blocks"] = hard_reserve
            if not feasible:
                self._debug_record_virtual_context_residency_admission_defer(
                    request, phase, raw_required_blocks, free_blocks,
                    headroom + self.virtual_context_decode_hotset_reserve_blocks,
                    self.virtual_context_staging_reserved_blocks
                    + self.virtual_context_ring_min_free_blocks,
                )
                return (False, raw_required_blocks, free_blocks, headroom,
                        hard_reserve - headroom, raw_required_blocks, 0, 0)
        if raw_required_blocks <= 0 or not self.virtual_context_residency_admission:
            return True, raw_required_blocks, free_blocks, 0, 0, raw_required_blocks, 0, 0

        if self._virtual_context_waiting_decode_ready(request):
            staging_blocks = self.virtual_context_staging_reserved_blocks
            feasible = free_blocks - raw_required_blocks >= staging_blocks
            if not feasible:
                self._debug_record_virtual_context_residency_admission_defer(
                    request,
                    phase,
                    raw_required_blocks,
                    free_blocks,
                    0,
                    staging_blocks,
                )
            return (
                feasible,
                raw_required_blocks,
                free_blocks,
                0,
                staging_blocks,
                raw_required_blocks,
                0,
                0,
            )
        decode_used_blocks = self._debug_running_decode_resident_blocks()
        decode_reserve_blocks = max(
            self.virtual_context_decode_hotset_reserve_blocks,
            decode_used_blocks,
        )
        staging_blocks = self.virtual_context_staging_reserved_blocks
        required_blocks = raw_required_blocks
        ring_residency_applied = 0
        if self.virtual_context_ring_residency_admission:
            ring_tokens = (
                self._dynamic_virtual_context_ring_prefill_admission_tokens()
            )
            if (
                self.pd_prefill_starvation_max_decode_iters > 0
                and request.pd_residency_prefill_skip_count
                >= self.pd_prefill_starvation_max_decode_iters
                and self.pd_ring_min_prefill_tokens_with_decode > 0
            ):
                ring_tokens = max(
                    ring_tokens,
                    self.pd_ring_min_prefill_tokens_with_decode,
                )
            if ring_tokens > 0:
                ring_tokens = min(num_new_tokens, ring_tokens)
                ring_required_blocks = max(
                    1,
                    (ring_tokens + self.block_size - 1) // self.block_size,
                )
                required_blocks = min(raw_required_blocks, ring_required_blocks)
                # The running decode hotset is already resident; for the ring
                # path we reserve only an explicit extra free-block guard rather
                # than requiring another decode-sized free region.
                decode_reserve_blocks = (
                    self.virtual_context_decode_hotset_reserve_blocks
                )
                staging_blocks += self.virtual_context_ring_min_free_blocks
                ring_residency_applied = 1
        feasible = (
            free_blocks - required_blocks
            >= decode_reserve_blocks + staging_blocks
        )
        if not feasible:
            self._debug_record_virtual_context_residency_admission_defer(
                request,
                phase,
                required_blocks,
                free_blocks,
                decode_reserve_blocks,
                staging_blocks,
            )
        return (
            feasible,
            required_blocks,
            free_blocks,
            decode_reserve_blocks,
            staging_blocks,
            raw_required_blocks,
            decode_used_blocks,
            ring_residency_applied,
        )

    def _dynamic_virtual_context_ring_prefill_admission_tokens(self) -> int:
        """Return the current scheduler-side context-ring prefill cap.

        The base cap is the conservative bandwidth used under pressure. When
        enough HBM/KV blocks are free, experiments can raise this cap to reduce
        unnecessary ring-path slowdown without disabling the ring path.
        """
        base_cap = self.virtual_context_ring_prefill_admission_tokens
        self._virtual_context_ring_bw_last_free_blocks = (
            self.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        self._virtual_context_ring_bw_last_steps = 0
        self._virtual_context_ring_bw_last_safe_blocks = 0
        self._virtual_context_ring_bw_last_safe_tokens = 0
        self._virtual_context_ring_bw_last_reserved_blocks = 0
        self._virtual_context_ring_bw_last_staging_blocks = 0
        self._virtual_context_ring_bw_last_safety_blocks = 0
        self._virtual_context_ring_bw_last_decode_growth_blocks = 0
        self._virtual_context_ring_bw_last_progress_reserve_applied = 0
        self._virtual_context_ring_bw_last_progress_reserve_tokens = 0
        self._virtual_context_ring_bw_last_progress_reserve_blocks = 0
        self._virtual_context_ring_bw_last_progress_hard_reserved_blocks = 0
        self._virtual_context_ring_bw_last_progress_reserve_reason = (
            "not_evaluated"
        )
        if base_cap <= 0:
            self._virtual_context_ring_bw_last_reason = "disabled"
            self._virtual_context_ring_bw_last_progress_reserve_reason = (
                "ring_admission_disabled"
            )
            return base_cap

        if not self.virtual_context_ring_bw_dynamic:
            self._virtual_context_ring_bw_last_reason = (
                "fixed_dynamic_disabled"
            )
            return base_cap

        max_cap = self.virtual_context_ring_prefill_admission_tokens_max
        if max_cap <= 0 or max_cap <= base_cap:
            self._virtual_context_ring_bw_last_reason = "fixed"
            return base_cap

        if self.virtual_context_ring_bw_policy == "safe_capacity":
            return (
                self._safe_capacity_virtual_context_ring_prefill_admission_tokens(
                    base_cap,
                    max_cap,
                )
            )

        free_blocks = self._virtual_context_ring_bw_last_free_blocks
        min_free_blocks = (
            self.virtual_context_ring_prefill_admission_dynamic_min_free_blocks
        )
        if free_blocks < min_free_blocks:
            self._virtual_context_ring_bw_last_reason = "pressure_low_free_blocks"
            return base_cap

        step_blocks = (
            self.virtual_context_ring_prefill_admission_dynamic_step_blocks
        )
        step_tokens = (
            self.virtual_context_ring_prefill_admission_dynamic_step_tokens
        )
        if step_blocks <= 0 or step_tokens <= 0:
            self._virtual_context_ring_bw_last_reason = "max_without_step"
            self._virtual_context_ring_bw_last_steps = 1
            return max_cap

        steps = (free_blocks - min_free_blocks) // step_blocks
        if steps <= 0:
            self._virtual_context_ring_bw_last_reason = "free_blocks_no_step"
            return base_cap
        self._virtual_context_ring_bw_last_steps = steps
        selected = min(max_cap, base_cap + steps * step_tokens)
        self._virtual_context_ring_bw_last_reason = (
            "max_cap" if selected >= max_cap else "scaled_by_free_blocks"
        )
        return selected

    def _virtual_context_ring_dma_budget_bytes(
        self,
        selected_tokens: int,
    ) -> int:
        """Translate the scheduler token window into a per-layer copy budget."""

        if (
            selected_tokens <= 0
            or self.block_size <= 0
            or self.virtual_context_kv_page_size_bytes <= 0
        ):
            return 0
        selected_pages = (
            selected_tokens + self.block_size - 1
        ) // self.block_size
        return selected_pages * self.virtual_context_kv_page_size_bytes

    def _safe_capacity_virtual_context_ring_prefill_admission_tokens(
        self,
        base_cap: int,
        max_cap: int,
    ) -> int:
        """Choose a ring prefill cap that preserves explicit HBM reserves."""
        free_blocks = self._virtual_context_ring_bw_last_free_blocks
        staging_blocks = (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
        )
        decode_growth_blocks = max(
            self.virtual_context_decode_hotset_reserve_blocks,
            self.virtual_context_ring_bw_decode_growth_blocks,
        )
        safety_blocks = self.virtual_context_ring_bw_safety_blocks
        if getattr(
            self,
            "virtual_context_ring_bw_use_scheduled_headroom",
            False,
        ):
            # Keep one capacity contract across proactive reclaim and elastic
            # Ring admission.  For a 0.98 fit fraction this automatically
            # preserves the matching 2% Decode-growth headroom instead of
            # requiring a second, manually synchronized block constant.
            safety_blocks = max(
                safety_blocks,
                self.virtual_context_scheduled_prefill_fit_headroom_blocks,
            )
        reserved_blocks = (
            staging_blocks + decode_growth_blocks + safety_blocks
        )
        safe_blocks = max(0, free_blocks - reserved_blocks)
        safe_tokens = safe_blocks * self.block_size

        self._virtual_context_ring_bw_last_staging_blocks = staging_blocks
        self._virtual_context_ring_bw_last_decode_growth_blocks = (
            decode_growth_blocks
        )
        self._virtual_context_ring_bw_last_safety_blocks = safety_blocks
        self._virtual_context_ring_bw_last_reserved_blocks = reserved_blocks
        self._virtual_context_ring_bw_last_safe_blocks = safe_blocks
        self._virtual_context_ring_bw_last_safe_tokens = safe_tokens

        (
            progress_tokens,
            progress_reserve_blocks,
            progress_hard_reserved_blocks,
        ) = self._virtual_context_active_decode_ring_progress_reserve()
        self._virtual_context_ring_bw_last_progress_reserve_tokens = (
            progress_tokens
        )
        self._virtual_context_ring_bw_last_progress_reserve_blocks = (
            progress_reserve_blocks
        )
        self._virtual_context_ring_bw_last_progress_hard_reserved_blocks = (
            progress_hard_reserved_blocks
        )
        active_decode = (
            self._virtual_context_active_decode_for_ring_progress()
        )
        if not active_decode:
            self._virtual_context_ring_bw_last_progress_reserve_reason = (
                "not_active_decode"
            )
        elif progress_tokens <= 0:
            self._virtual_context_ring_bw_last_progress_reserve_reason = (
                "progress_reserve_disabled"
            )
        elif safe_tokens < progress_tokens:
            protected_tokens = (
                virtual_context_select_active_decode_ring_progress_tokens(
                    free_blocks,
                    progress_tokens,
                    progress_tokens,
                    self.block_size,
                    staging_blocks,
                    decode_growth_blocks,
                )
            )
            if protected_tokens > 0:
                self._virtual_context_ring_bw_last_progress_reserve_applied = 1
                self._virtual_context_ring_bw_last_progress_reserve_reason = (
                    "active_decode_protected_min_progress"
                )
                self._virtual_context_ring_bw_last_reason = (
                    "active_decode_protected_min_progress"
                )
                return min(max_cap, protected_tokens)
            self._virtual_context_ring_bw_last_progress_reserve_reason = (
                "insufficient_hard_capacity"
            )
        else:
            self._virtual_context_ring_bw_last_progress_reserve_reason = (
                "normal_safe_capacity_sufficient"
            )

        if safe_tokens <= 0:
            self._virtual_context_ring_bw_last_reason = (
                "safe_capacity_no_room"
            )
            return 0

        selected = min(max_cap, safe_tokens)
        selected = (selected // self.block_size) * self.block_size
        if selected <= 0:
            self._virtual_context_ring_bw_last_reason = (
                "safe_capacity_below_block"
            )
            return 0

        self._virtual_context_ring_bw_last_steps = max(
            0,
            (selected - base_cap) // max(1, self.block_size),
        )
        if selected < base_cap:
            self._virtual_context_ring_bw_last_reason = (
                "safe_capacity_below_base"
            )
        elif selected >= max_cap:
            self._virtual_context_ring_bw_last_reason = (
                "safe_capacity_max_cap"
            )
        else:
            self._virtual_context_ring_bw_last_reason = "safe_capacity"
        return selected

    def _cap_virtual_context_ring_prefill_admission_tokens(
        self,
        request: Request,
        num_new_tokens: int,
        *,
        execution_mode: str = "",
    ) -> int:
        """Cap scheduler-side prefill admission for context-ring experiments.

        This intentionally caps the scheduler/admission chunk, not the
        backend-internal context-ring Q tile. It is default-off so upstream vLLM
        behavior is unchanged unless the experiment explicitly sets the flag.
        """
        if num_new_tokens <= 0:
            return num_new_tokens
        if self.virtual_context_ring_prefill_admission_tokens <= 0:
            return num_new_tokens
        if not self._debug_has_prefill_remaining(request):
            return num_new_tokens
        if (
            execution_mode == "native"
            and self.ttft_prefill_native_chunk_max_tokens > 0
        ):
            return self._cap_prefill_admission_with_active_decode(
                request,
                num_new_tokens,
                execution_mode="native",
            )
        minimum_progress_tokens = min(
            num_new_tokens,
            self.virtual_context_ring_min_progress_tokens,
        )
        cap = self._dynamic_virtual_context_ring_prefill_admission_tokens()
        if cap <= 0:
            if minimum_progress_tokens > 0:
                self._virtual_context_ring_bw_last_reason = (
                    "below_min_progress_defer"
                )
            return 0
        cap = self._cap_prefill_admission_with_active_decode(
            request,
            cap,
            execution_mode=execution_mode,
        )
        selected_tokens = min(num_new_tokens, cap)
        if (
            minimum_progress_tokens > 0
            and selected_tokens < minimum_progress_tokens
        ):
            self._virtual_context_ring_bw_last_reason = (
                "below_min_progress_defer"
            )
            return 0
        return selected_tokens

    def _cap_prefill_admission_with_active_decode(
        self,
        request: Request,
        cap: int,
        *,
        execution_mode: str = "",
    ) -> int:
        """Keep active-decode prefill chunks bounded.

        The elastic safe-capacity policy can raise the normal context-ring
        admission window when many KV blocks are free. That is useful when the
        decode lane is idle, but too coarse while decode is already running:
        a first prefill chunk has no DRAM-backed pages yet and would otherwise
        be admitted as a large ordinary HBM-native prefill. Bound both the
        first bootstrap chunk and later ring-required chunks with explicit
        active-decode caps.
        """
        self._virtual_context_ring_bw_last_active_decode_cap_tokens = 0
        self._virtual_context_ring_bw_last_active_decode_cap_reason = (
            "not_active_decode"
        )
        self._virtual_context_ring_bw_last_active_decode_reload_pages = 0

        if cap <= 0:
            return cap
        if not self.virtual_context_decode_hotset_first:
            return cap
        if not self._debug_has_prefill_remaining(request):
            return cap
        if (
            self._debug_running_decode_resident_blocks() <= 0
            and not self._pd_pending_decode_active()
        ):
            return cap

        force_all = _read_bool_env(
            "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL"
        )
        reload_required_pages = self._virtual_context_reload_required_page_count(
            [request.request_id],
            "prefill",
        )
        self._virtual_context_ring_bw_last_active_decode_reload_pages = (
            reload_required_pages
        )

        ring_required = (
            execution_mode == "ring"
            if execution_mode in {"native", "ring"}
            else force_all or reload_required_pages > 0
        )
        if ring_required:
            active_cap = self.virtual_context_active_decode_ring_prefill_tokens
            reason = (
                "active_decode_force_all_ring"
                if force_all
                else "active_decode_ring_required"
            )
        else:
            active_cap = (
                self.virtual_context_active_decode_native_prefill_tokens
            )
            (
                native_safe,
                _reload_required_pages,
                _free_blocks,
                _threshold_blocks,
                _bootstrap_blocks,
                _ring_blocks,
                pressure_reason,
            ) = self._prefill_native_safe_before_ring_pressure(request)
            if native_safe:
                active_cap = max(
                    active_cap,
                    self._virtual_context_ring_pressure_bootstrap_token_budget(),
                )
                reason = f"active_decode_native_safe_bootstrap:{pressure_reason}"
            else:
                reason = f"active_decode_native_bootstrap:{pressure_reason}"

        if (
            ring_required
            and self.virtual_context_active_decode_elastic_prefill
            and self.virtual_context_ring_bw_dynamic
            and self.virtual_context_ring_bw_policy == "safe_capacity"
        ):
            # ``cap`` was already computed after subtracting staging, Decode
            # growth, and the configured safety/headroom reserve.  Reusing it
            # here lets an established Ring request grow beyond the legacy 2K
            # active-Decode cap without admitting a large first native chunk.
            # The switch is experimental and default-off for exact rollback.
            active_cap = max(active_cap, cap)
            reason = f"{reason}:elastic_safe_capacity"

        if active_cap <= 0:
            self._virtual_context_ring_bw_last_active_decode_cap_reason = (
                f"{reason}_uncapped"
            )
            return cap

        selected = min(cap, active_cap)
        self._virtual_context_ring_bw_last_active_decode_cap_tokens = selected
        self._virtual_context_ring_bw_last_active_decode_cap_reason = reason
        return selected

    def _has_older_active_prefill_request(self, request: Request) -> bool:
        """Return whether an older request still owns prefill progress.

        This is used only by the experimental FCFS gate. It prevents a newer
        waiting prefill from being admitted while an older request's prefill is
        still running or has an in-flight prefill future. A request with an
        in-flight decode future has already transferred phase ownership to the
        decode lane, even before its first sampled token is recovered. It must
        not block a newer prefill from the paired prefill lane.
        """
        request_arrival_time = request.arrival_time
        request_arrival_iter = request.arrival_iteration_id
        pending_decode_request_ids = self._pd_pending_phase_request_ids.get(
            "decode",
            set(),
        )

        def is_older_prefill(candidate: Request) -> bool:
            if candidate.request_id == request.request_id:
                return False
            if candidate.request_id in pending_decode_request_ids:
                return False
            if not self._debug_needs_prefill_lane(candidate):
                return False
            if candidate.arrival_time < request_arrival_time:
                return True
            if (
                candidate.arrival_time == request_arrival_time
                and candidate.arrival_iteration_id >= 0
                and request_arrival_iter >= 0
                and candidate.arrival_iteration_id < request_arrival_iter
            ):
                return True
            return False

        for candidate in self.running:
            if is_older_prefill(candidate):
                return True
        for req_id in self._pd_pending_request_ids:
            candidate = self.requests.get(req_id)
            if candidate is not None and is_older_prefill(candidate):
                return True
        return False

    def _debug_request_lifecycle_fields(
        self, request: Request
    ) -> dict[str, int | float | str]:
        output_ledger = request.pd_output_placeholder_ledger
        active_output_owner = output_ledger.active
        return {
            "arrival_iteration_id": request.arrival_iteration_id,
            "first_admitted_iteration_id": request.first_admitted_iteration_id,
            "first_prefill_iteration_id": request.first_prefill_iteration_id,
            "last_prefill_iteration_id": request.last_prefill_iteration_id,
            "first_decode_iteration_id": request.first_decode_iteration_id,
            "prefill_completed_at": (
                request.prefill_completed_at
                if request.prefill_completed_at is not None
                else -1.0
            ),
            "prefill_completed_iteration_id": (
                request.prefill_completed_iteration_id
            ),
            "num_prefill_chunks": request.num_prefill_chunks,
            "deferred_iterations_count": request.deferred_iterations_count,
            "request_deferred_iterations_count": (
                request.deferred_iterations_count
            ),
            "deferred_by_kv_capacity_count": (
                request.deferred_by_kv_capacity_count
            ),
            "deferred_by_max_tokens_count": request.deferred_by_max_tokens_count,
            "deferred_by_max_seqs_count": request.deferred_by_max_seqs_count,
            "deferred_by_policy_count": request.deferred_by_policy_count,
            "deferred_by_preemption_count": request.deferred_by_preemption_count,
            **(
                self.tbt_decode_scheduler.request_metrics(request)
                if self.tbt_slo_observability_enabled
                else {}
            ),
            "deferred_by_residency_budget_count": (
                request.deferred_by_residency_budget_count
            ),
            "running_kv_capacity_miss_count": (
                request.running_kv_capacity_miss_count
            ),
            "running_prefill_kv_capacity_miss_count": (
                request.running_prefill_kv_capacity_miss_count
            ),
            "running_decode_kv_capacity_miss_count": (
                request.running_decode_kv_capacity_miss_count
            ),
            "last_running_kv_capacity_miss_iteration_id": (
                request.last_running_kv_capacity_miss_iteration_id
            ),
            "last_running_kv_capacity_miss_phase": (
                request.last_running_kv_capacity_miss_phase
            ),
            "last_running_kv_capacity_miss_num_computed_tokens": (
                request.last_running_kv_capacity_miss_num_computed_tokens
            ),
            "last_running_kv_capacity_miss_num_tokens": (
                request.last_running_kv_capacity_miss_num_tokens
            ),
            "preempted_reset_count": request.preempted_reset_count,
            "preempted_prefill_reset_count": (
                request.preempted_prefill_reset_count
            ),
            "preempted_decode_reset_count": request.preempted_decode_reset_count,
            "last_preempt_iteration_id": request.last_preempt_iteration_id,
            "last_preempt_phase": request.last_preempt_phase,
            "last_preempt_num_computed_tokens_before_reset": (
                request.last_preempt_num_computed_tokens_before_reset
            ),
            "last_preempt_num_tokens_before_reset": (
                request.last_preempt_num_tokens_before_reset
            ),
            "preempted_tokens_reset_total": (
                request.preempted_tokens_reset_total
            ),
            "pd_recompute_frontier_tokens": (
                request.pd_recompute_frontier_tokens
            ),
            "running_prefill_kv_wait_count": (
                request.running_prefill_kv_wait_count
            ),
            "last_running_prefill_kv_wait_iteration_id": (
                request.last_running_prefill_kv_wait_iteration_id
            ),
            "last_running_prefill_kv_wait_num_computed_tokens": (
                request.last_running_prefill_kv_wait_num_computed_tokens
            ),
            "last_running_prefill_kv_wait_num_tokens": (
                request.last_running_prefill_kv_wait_num_tokens
            ),
            "pd_residency_prefill_skip_count": (
                request.pd_residency_prefill_skip_count
            ),
            "pd_residency_prefill_skip_first_iteration_id": (
                request.pd_residency_prefill_skip_first_iteration_id
            ),
            "pd_residency_prefill_skip_last_iteration_id": (
                request.pd_residency_prefill_skip_last_iteration_id
            ),
            "pd_residency_prefill_unscheduled_duration_s": (
                request.pd_residency_prefill_unscheduled_duration_s
            ),
            "pd_residency_last_prefill_required_blocks": (
                request.pd_residency_last_prefill_required_blocks
            ),
            "pd_residency_last_free_blocks": (
                request.pd_residency_last_free_blocks
            ),
            "pd_residency_last_decode_resident_blocks": (
                request.pd_residency_last_decode_resident_blocks
            ),
            "pd_residency_last_serialized_by_kv_reserve": (
                request.pd_residency_last_serialized_by_kv_reserve
            ),
            "virtual_context_deferred_prefill_count": (
                request.virtual_context_deferred_prefill_count
            ),
            "virtual_context_deferred_logical_computed_tokens": (
                request.virtual_context_deferred_logical_computed_tokens
            ),
            "virtual_context_deferred_state": (
                request.virtual_context_deferred_state
            ),
            "virtual_context_deferred_last_iteration_id": (
                request.virtual_context_deferred_last_iteration_id
            ),
            "virtual_context_deferred_last_offloaded_tokens": (
                request.virtual_context_deferred_last_offloaded_tokens
            ),
            "virtual_context_deferred_hbm_released_tokens": (
                request.virtual_context_deferred_hbm_released_tokens
            ),
            "virtual_context_resume_without_recompute_count": (
                request.virtual_context_resume_without_recompute_count
            ),
            "virtual_context_hbm_reloaded_tokens": (
                request.virtual_context_hbm_reloaded_tokens
            ),
            "virtual_context_hbm_reload_planned_tokens": (
                request.virtual_context_hbm_reload_planned_tokens
            ),
            "kv_retention_decode_continuation_lease_active": int(
                getattr(
                    request,
                    "kv_retention_decode_continuation_lease_active",
                    False,
                )
            ),
            "kv_retention_decode_continuation_waiting": int(
                getattr(
                    request,
                    "kv_retention_decode_continuation_waiting",
                    False,
                )
            ),
            "kv_retention_decode_continuation_wait_count": (
                getattr(
                    request,
                    "kv_retention_decode_continuation_wait_count",
                    0,
                )
            ),
            "kv_retention_decode_continuation_last_wait_iteration_id": (
                getattr(
                    request,
                    "kv_retention_decode_continuation_last_wait_iteration_id",
                    -1,
                )
            ),
            "kv_retention_execution_plan_generation": (
                request.kv_retention_execution_plan_generation
            ),
            "kv_retention_execution_plan_acked_generation": (
                request.kv_retention_execution_plan_acked_generation
            ),
            "kv_retention_execution_plan_outstanding": int(
                request.kv_retention_execution_plan_generation
                > request.kv_retention_execution_plan_acked_generation
            ),
            "kv_retention_execution_plan_mode": (
                request.kv_retention_execution_plan_mode
            ),
            "kv_retention_execution_plan_expected_reload_pages": (
                request.kv_retention_execution_plan_expected_reload_pages
            ),
            "virtual_context_tiered_provider_deferred_tokens": (
                request.virtual_context_tiered_provider_deferred_tokens
            ),
            "virtual_context_tiered_provider_resume_count": (
                request.virtual_context_tiered_provider_resume_count
            ),
            "virtual_context_tiered_provider_virtual_external_tokens": (
                request.virtual_context_tiered_provider_virtual_external_tokens
            ),
            "virtual_context_tiered_commit_count": (
                request.virtual_context_tiered_commit_count
            ),
            "virtual_context_tiered_commit_released_pages": (
                request.virtual_context_tiered_commit_released_pages
            ),
            "virtual_context_tiered_commit_existing_dram_pages": (
                request.virtual_context_tiered_commit_existing_dram_pages
            ),
            "virtual_context_tiered_commit_missing_pages": (
                request.virtual_context_tiered_commit_missing_pages
            ),
            "virtual_context_tiered_commit_last_iteration_id": (
                request.virtual_context_tiered_commit_last_iteration_id
            ),
            "virtual_context_tiered_commit_last_logical_tokens": (
                request.virtual_context_tiered_commit_last_logical_tokens
            ),
            "virtual_context_tiered_transition_attempt_count": (
                request.virtual_context_tiered_transition_attempt_count
            ),
            "virtual_context_tiered_transition_last_iteration_id": (
                request.virtual_context_tiered_transition_last_iteration_id
            ),
            "virtual_context_tiered_transition_last_logical_tokens": (
                request.virtual_context_tiered_transition_last_logical_tokens
            ),
            "virtual_context_tiered_transition_last_block_size": (
                request.virtual_context_tiered_transition_last_block_size
            ),
            "virtual_context_tiered_transition_last_release_candidate_records": (
                request.virtual_context_tiered_transition_last_release_candidate_records
            ),
            "virtual_context_tiered_transition_last_release_candidate_pages": (
                request.virtual_context_tiered_transition_last_release_candidate_pages
            ),
            "virtual_context_tiered_transition_last_existing_dram_records": (
                request.virtual_context_tiered_transition_last_existing_dram_records
            ),
            "virtual_context_tiered_transition_last_existing_dram_pages": (
                request.virtual_context_tiered_transition_last_existing_dram_pages
            ),
            "virtual_context_tiered_transition_last_required_pages": (
                request.virtual_context_tiered_transition_last_required_pages
            ),
            "virtual_context_tiered_transition_last_covered_pages": (
                request.virtual_context_tiered_transition_last_covered_pages
            ),
            "virtual_context_tiered_transition_last_missing_pages": (
                request.virtual_context_tiered_transition_last_missing_pages
            ),
            "virtual_context_tiered_transition_last_reason": (
                request.virtual_context_tiered_transition_last_reason
            ),
            "virtual_context_preempt_reset_avoided_tokens": (
                request.virtual_context_preempt_reset_avoided_tokens
            ),
            "virtual_context_defer_fallback_count": (
                request.virtual_context_defer_fallback_count
            ),
            "virtual_context_defer_last_fallback_reason": (
                request.virtual_context_defer_last_fallback_reason
            ),
            "virtual_context_residency_admission_defer_count": (
                request.virtual_context_residency_admission_defer_count
            ),
            "virtual_context_residency_admission_last_iteration_id": (
                request.virtual_context_residency_admission_last_iteration_id
            ),
            "virtual_context_residency_admission_last_phase": (
                request.virtual_context_residency_admission_last_phase
            ),
            "virtual_context_residency_admission_last_required_blocks": (
                request.virtual_context_residency_admission_last_required_blocks
            ),
            "virtual_context_residency_admission_last_free_blocks": (
                request.virtual_context_residency_admission_last_free_blocks
            ),
            "virtual_context_residency_admission_last_decode_reserve_blocks": (
                request.virtual_context_residency_admission_last_decode_reserve_blocks
            ),
            "virtual_context_residency_admission_last_staging_blocks": (
                request.virtual_context_residency_admission_last_staging_blocks
            ),
            "pd_output_owner_request_epoch": output_ledger.request_epoch,
            "pd_output_owner_active_request_epoch": (
                active_output_owner.request_epoch
                if active_output_owner is not None
                else 0
            ),
            "pd_output_owner_active_generation": (
                active_output_owner.generation
                if active_output_owner is not None
                else 0
            ),
            "pd_output_owner_active_phase": (
                active_output_owner.phase
                if active_output_owner is not None
                else ""
            ),
            "pd_output_owner_active_placeholder_count": (
                active_output_owner.placeholder_count
                if active_output_owner is not None
                else 0
            ),
            "pd_output_owner_active_cancelled": int(
                active_output_owner is not None
                and active_output_owner.cancelled
            ),
            "pd_output_owner_last_committed_generation": (
                output_ledger.last_committed_generation
            ),
            "pd_output_owner_last_committed_phase": (
                output_ledger.last_committed_phase
            ),
            **(
                self.ttft_prefill_chunk_scheduler.request_metrics(request)
                if self.enable_ttft_prefill_chunk_scheduling
                else {}
            ),
        }

    def _native_decode_control_fast_path_eligible(
        self,
        request: Request,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> bool:
        """Return whether decode diagnostics can avoid logical-page scans.

        Historical tiered activity must not permanently disable this path.
        Once prefill is complete and the O(1) residency index reports no page
        requiring HBM reload, native decode can avoid full logical-page scans.
        A pending handoff or any currently DRAM-authoritative page still takes
        the full residency path through the runtime eligibility predicate.
        """
        if not self.virtual_context_native_decode_control_fast_path:
            return False
        if prefill_tokens != 0 or decode_tokens <= 0:
            return False
        return self._native_decode_runtime_fast_path_eligible(request)

    def _native_decode_runtime_fast_path_eligible(
        self,
        request: Request,
    ) -> bool:
        """Return whether native decode can skip tiered residency planning."""
        if not self.virtual_context_native_decode_control_fast_path:
            return False
        if self._debug_has_prefill_remaining(request):
            return False
        if getattr(
            request,
            "kv_retention_decode_continuation_waiting",
            False,
        ):
            return False
        # Let the full handoff path publish its one-time state transition.
        if request.virtual_context_deferred_state == "decode_hbm_reload_pending":
            return False
        return not self.kv_cache_manager.has_virtual_context_hbm_reload_required_pages(
            {request.request_id}
        )

    def _commit_virtual_context_native_reload_admissions(
        self,
        num_scheduled_tokens: dict[str, int],
        explicit_plans: dict[str, tuple[str, int]],
        planned_tokens: dict[str, int],
    ) -> set[str]:
        """Publish pending only after all admission/preemption decisions settle.

        Preparation may allocate destinations, but they remain DRAM-authoritative
        until the worker copies them. An unselected intent owns no Future and
        must not change waiting/decode eligibility or create a reload generation.
        """
        admitted: list[Request] = []
        for request_id, (mode, pages) in explicit_plans.items():
            if mode != TIERED_EXECUTION_NATIVE_RELOAD:
                continue
            if num_scheduled_tokens.get(request_id, 0) <= 0:
                continue
            request = self.requests[request_id]
            if pages <= 0 or planned_tokens.get(request_id, 0) <= 0:
                raise TieredAllocatorInvariantError(
                    f"Native reload admission lacks a copy plan: request={request_id}"
                )
            if (
                request.kv_retention_execution_plan_generation
                > request.kv_retention_execution_plan_acked_generation
            ):
                raise TieredAllocatorInvariantError(
                    f"Native reload admission overlaps an unacknowledged Future: {request_id}"
                )
            admitted.append(request)
        # Preflight the whole set before changing any request.
        for request in admitted:
            request.virtual_context_deferred_state = (
                "hbm_reload_pending"
                if self._debug_has_prefill_remaining(request)
                else "decode_hbm_reload_pending"
            )
            request.virtual_context_hbm_reload_planned_tokens = (
                planned_tokens[request.request_id]
            )
        return {request.request_id for request in admitted}

    def _build_tiered_request_execution_plans(
        self,
        num_scheduled_tokens: dict[str, int],
        explicit_plans: dict[str, tuple[str, int]],
    ) -> tuple[TieredExecutionPlanEntry, ...]:
        """Freeze scheduler execution authority for this exact Future."""

        if not self.virtual_context_enabled:
            return ()
        entries: list[TieredExecutionPlanEntry] = []
        for request_id, scheduled_tokens in num_scheduled_tokens.items():
            if scheduled_tokens <= 0:
                continue
            request = self.requests.get(request_id)
            if request is None:
                continue
            explicit = explicit_plans.get(request_id)
            if (
                request.virtual_context_deferred_state
                in {"hbm_reload_pending", "decode_hbm_reload_pending"}
                and (
                    explicit is None
                    or explicit[0] != TIERED_EXECUTION_NATIVE_RELOAD
                )
            ):
                raise TieredAllocatorInvariantError(
                    "Pending HBM reload requires explicit native_reload authority: "
                    f"request={request_id}"
                )
            if (
                explicit is None
                and request.virtual_context_deferred_logical_computed_tokens
                <= 0
                and request.num_external_computed_tokens <= 0
                and request.virtual_context_deferred_state
                not in _VIRTUAL_CONTEXT_REHYDRATE_STATES
            ):
                continue
            reload_pages = (
                self.kv_cache_manager.get_virtual_context_hbm_reload_required_page_count(
                    {request_id}
                )
            )
            if explicit is not None:
                mode, expected_reload_pages = explicit
                expected_reload_pages = max(
                    int(expected_reload_pages),
                    int(reload_pages),
                )
            else:
                # Native HBM-only iterations need no tiered contract. Avoid
                # adding generation bookkeeping to the per-token Decode hot
                # path when no DRAM authority exists.
                if reload_pages <= 0:
                    continue
                prefill_remaining = self._pd_prefill_lane_remaining_tokens(
                    request,
                )
                prefill_tokens = min(int(scheduled_tokens), prefill_remaining)
                decode_tokens = max(0, int(scheduled_tokens) - prefill_tokens)
                if prefill_tokens > 0 and decode_tokens > 0:
                    mode = TIERED_EXECUTION_AUTO
                elif reload_pages > 0 and prefill_tokens > 0:
                    mode = TIERED_EXECUTION_CONTEXT_RING
                elif reload_pages > 0 and decode_tokens > 0:
                    mode = TIERED_EXECUTION_RING_DECODE
                else:
                    mode = TIERED_EXECUTION_AUTO
                expected_reload_pages = int(reload_pages)

            active_generation = int(
                request.kv_retention_execution_plan_generation
            )
            acked_generation = int(
                request.kv_retention_execution_plan_acked_generation
            )
            if active_generation > acked_generation:
                self._tiered_execution_plan_overwrite_reject_count += 1
                raise TieredAllocatorInvariantError(
                    "Tiered execution single-flight rejected a second plan "
                    "before the prior Future was acknowledged: "
                    f"request={request_id}, active={active_generation}, "
                    f"acked={acked_generation}, next_mode={mode!r}."
                )

            self._tiered_execution_plan_next_generation = (
                int(
                    getattr(
                        self,
                        "_tiered_execution_plan_next_generation",
                        0,
                    )
                )
                + 1
            )
            generation = self._tiered_execution_plan_next_generation
            request.kv_retention_execution_plan_generation = generation
            request.kv_retention_execution_plan_mode = mode
            request.kv_retention_execution_plan_expected_reload_pages = (
                expected_reload_pages
            )
            entries.append(
                (
                    request_id,
                    generation,
                    mode,
                    expected_reload_pages,
                )
            )
        return tuple(sorted(entries))

    @staticmethod
    def _native_decode_request_debug_fields(
        request: Request,
    ) -> dict[str, int]:
        """Return known native-decode facts without traversing logical pages."""
        return {
            "virtual_context_request_debug_fast_path": 1,
            "virtual_context_request_tokens": request.num_computed_tokens,
            # Page totals are intentionally marked unknown. Reporting zero
            # would incorrectly imply that an HBM-native request owns no KV.
            "virtual_context_request_pages": -1,
            "virtual_context_request_hbm_pages": -1,
            "virtual_context_request_dram_pages": 0,
            "virtual_context_request_dram_copy_pages": -1,
            "virtual_context_request_dram_copy_short_pages": -1,
            "virtual_context_request_provider_expected_layers": -1,
            "virtual_context_request_hbm_release_candidate_pages": -1,
            "virtual_context_request_hbm_release_candidate_tokens": -1,
            "virtual_context_request_hbm_release_candidate_with_partial_pages": -1,
            "virtual_context_request_hbm_release_candidate_with_partial_tokens": -1,
            "virtual_context_request_hbm_release_marked_pages": 0,
            "virtual_context_request_hbm_release_marked_tokens": 0,
            "virtual_context_request_hbm_reload_required_pages": 0,
            "virtual_context_request_hbm_reload_required_tokens": 0,
            "virtual_context_request_partial_tail_release_blocked_pages": -1,
            "virtual_context_request_partial_tail_release_blocked_tokens": -1,
            "virtual_context_request_partial_tail_hbm_pinned_pages": -1,
            "virtual_context_request_partial_tail_hbm_pinned_tokens": -1,
        }

    def _virtual_context_debug_snapshot_due(self) -> bool:
        return (
            self.virtual_context_debug_snapshot_interval <= 1
            or self._iteration_debug_index
            % self.virtual_context_debug_snapshot_interval
            == 0
        )

    def _get_virtual_context_request_debug_snapshot(
        self,
        request_id: str,
        *,
        refresh: bool,
    ) -> tuple[dict[str, int | float], bool, float]:
        cached = self._virtual_context_request_debug_stats_cache.get(request_id)
        if refresh or cached is None:
            started = time.perf_counter()
            stats = (
                self.kv_cache_manager.get_virtual_context_request_debug_stats(
                    request_id
                )
            )
            duration_s = time.perf_counter() - started
            self._virtual_context_request_debug_stats_cache[request_id] = dict(
                stats
            )
            self._virtual_context_request_debug_stats_cache_iteration[
                request_id
            ] = self._iteration_debug_index
            result = dict(stats)
            stale = False
        else:
            result = dict(cached)
            duration_s = 0.0
            stale = True

        snapshot_iteration = (
            self._virtual_context_request_debug_stats_cache_iteration.get(
                request_id,
                self._iteration_debug_index,
            )
        )
        result.update(
            {
                "virtual_context_request_debug_snapshot_stale": int(stale),
                "virtual_context_request_debug_snapshot_age_iterations": max(
                    0,
                    self._iteration_debug_index - snapshot_iteration,
                ),
            }
        )
        return result, stale, duration_s

    def _get_virtual_context_attention_plan_debug_snapshot(
        self,
        *,
        phase: AttentionPhase,
        request_ids: list[str],
        query_tokens: int,
        prefix: str,
        staging_budget_blocks: int,
        dma_budget_bytes: int = 0,
        refresh: bool,
    ) -> tuple[dict[str, int | float | str], bool, float]:
        cache_key = (phase, tuple(sorted(request_ids)))
        cached = self._virtual_context_attention_plan_debug_stats_cache.get(
            cache_key
        )
        if refresh or cached is None:
            started = time.perf_counter()
            stats = (
                self.kv_cache_manager.get_virtual_context_attention_plan_debug_stats(
                    phase=phase,
                    request_ids=request_ids,
                    query_tokens=query_tokens,
                    prefix=prefix,
                    staging_budget_blocks=staging_budget_blocks,
                    dma_budget_bytes=dma_budget_bytes,
                )
            )
            duration_s = time.perf_counter() - started
            self._virtual_context_attention_plan_debug_stats_cache[cache_key] = (
                dict(stats)
            )
            self._virtual_context_attention_plan_debug_stats_cache_iteration[
                cache_key
            ] = self._iteration_debug_index
            result = dict(stats)
            stale = False
        else:
            result = dict(cached)
            duration_s = 0.0
            stale = True

        snapshot_iteration = (
            self._virtual_context_attention_plan_debug_stats_cache_iteration.get(
                cache_key,
                self._iteration_debug_index,
            )
        )
        # These fields describe the current scheduler output rather than the
        # cached residency sample and must remain iteration-accurate.
        result.update(
            {
                f"{prefix}_phase": phase,
                f"{prefix}_request_count": len(set(request_ids)),
                f"{prefix}_query_tokens": query_tokens,
                f"{prefix}_staging_budget_blocks": staging_budget_blocks,
                f"{prefix}_dma_budget_bytes": dma_budget_bytes,
                f"{prefix}_debug_snapshot_stale": int(stale),
                f"{prefix}_debug_snapshot_age_iterations": max(
                    0,
                    self._iteration_debug_index - snapshot_iteration,
                ),
            }
        )
        return result, stale, duration_s

    @staticmethod
    def _native_decode_attention_plan_debug_fields(
        prefix: str,
        phase: str,
        request_count: int,
        query_tokens: int,
        staging_budget_blocks: int,
    ) -> dict[str, int | float | str]:
        """Build a no-scan plan summary for an HBM-native decode iteration."""
        return {
            f"{prefix}_phase": phase,
            f"{prefix}_request_count": request_count,
            f"{prefix}_query_tokens": query_tokens,
            f"{prefix}_required_pages": -1,
            f"{prefix}_hbm_resident_pages": -1,
            f"{prefix}_dram_resident_pages": 0,
            f"{prefix}_staging_pages": 0,
            f"{prefix}_pages_to_materialize": 0,
            f"{prefix}_staging_budget_blocks": staging_budget_blocks,
            f"{prefix}_dma_budget_bytes": 0,
            f"{prefix}_tbt_budget_ms": 0.0,
            f"{prefix}_feasible_with_current_staging_budget": 1,
            f"{prefix}_ring_path_requested": 0,
            f"{prefix}_ring_path_active": 0,
            f"{prefix}_ring_traversal_pages": 0,
            f"{prefix}_ring_traversal_segments": 0,
            f"{prefix}_ring_hbm_traversal_pages": 0,
            f"{prefix}_ring_dram_traversal_pages": 0,
            f"{prefix}_ring_staging_traversal_pages": 0,
            f"{prefix}_ring_pages_to_materialize": 0,
            f"{prefix}_ring_online_softmax_merge_required": 0,
            f"{prefix}_ring_executor_available": -1,
            f"{prefix}_ring_executor_enabled": -1,
            f"{prefix}_ring_executor_implemented": -1,
            f"{prefix}_ring_executor_attached_to_vllm_attention": -1,
            f"{prefix}_ring_backend_needed_by_residency": 0,
            f"{prefix}_ring_backend_force_all": 0,
            f"{prefix}_ring_backend_will_attempt": 0,
            f"{prefix}_ring_decode_slo_policy_applied": 0,
        }

    @staticmethod
    def _native_decode_reload_guard_debug_fields(
        prefix: str,
        phase: str,
        request_count: int,
    ) -> dict[str, int | str]:
        return {
            f"{prefix}_phase": phase,
            f"{prefix}_request_count": request_count,
            f"{prefix}_reload_required_pages": 0,
            f"{prefix}_reload_required_tokens": 0,
            f"{prefix}_reload_required_request_count": 0,
            f"{prefix}_tiered_backend_enabled": -1,
            f"{prefix}_dram_provider_enabled": -1,
            f"{prefix}_reload_backend_available": -1,
            f"{prefix}_satisfied_by_tiered_provider": 0,
            f"{prefix}_violation": 0,
        }

    def _debug_estimate_next_prefill_required_blocks(
        self, request: Request
    ) -> int:
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, self.max_num_scheduled_tokens)
        num_new_tokens = min(
            num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
        )
        if num_new_tokens <= 0:
            return 0
        return self.kv_cache_manager.estimate_num_blocks_to_allocate(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )

    def _native_prefill_capacity_feasible_with_active_decode(
        self,
        request: Request,
        num_new_tokens: int,
        *,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_virtual_external_computed_tokens: int = 0,
        num_encoder_tokens: int = 0,
    ) -> tuple[bool, int, int, int, str]:
        """Admit one native Prefill chunk without starving active Decode.

        The old experiment reserved the target request's *entire remaining
        prompt*. That converted ordinary overlap into serialization hundreds
        of blocks before the actual allocator boundary. The authoritative
        claim is now the exact chunk selected by this scheduler pass plus the
        remaining KV growth of Decode requests that already own the Decode
        lane. ``allocate_slots`` remains the final mutating check.
        """
        if not self.pd_muxwise_dynamic_policy:
            return True, 0, 0, 0, "muxwise_dynamic_disabled"
        if self.virtual_context_enabled:
            return True, 0, 0, 0, "virtual_context_enabled"
        if not self._debug_has_prefill_remaining(request):
            return True, 0, 0, 0, "not_prefill"
        if num_new_tokens <= 0:
            return True, 0, 0, 0, "no_selected_prefill_chunk"

        chunk_required_blocks = (
            self.kv_cache_manager.estimate_num_blocks_to_allocate(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                num_virtual_external_computed_tokens=(
                    num_virtual_external_computed_tokens
                ),
                num_encoder_tokens=num_encoder_tokens,
            )
        )
        decode_growth_reserve_blocks = sum(
            self.kv_cache_manager.estimate_num_blocks_to_allocate(
                decode_request,
                max(
                    0,
                    decode_request.max_tokens
                    - decode_request.num_output_tokens,
                ),
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            for decode_request in self.running
            if not self._debug_has_prefill_remaining(decode_request)
            and decode_request.request_id != request.request_id
            and decode_request.max_tokens > decode_request.num_output_tokens
        )
        required_blocks = (
            chunk_required_blocks + decode_growth_reserve_blocks
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        feasible = required_blocks <= free_blocks
        return (
            feasible,
            chunk_required_blocks,
            decode_growth_reserve_blocks,
            free_blocks,
            (
                "native_chunk_plus_decode_reserve_available"
                if feasible
                else "native_chunk_plus_decode_reserve_exceeded"
            ),
        )

    def _debug_running_decode_resident_blocks(self) -> int:
        return sum(
            self.kv_cache_manager.get_request_resident_block_count(
                request.request_id
            )
            for request in self.running
            if not self._debug_has_prefill_remaining(request)
        )

    def _virtual_context_reload_required_page_count(
        self,
        request_ids: Iterable[str],
        phase: str = "decode",
    ) -> int:
        request_id_list = list(request_ids)
        if (
            phase == "decode"
            and self.virtual_context_native_decode_control_fast_path
            and request_id_list
        ):
            requests = [
                self.requests.get(request_id)
                for request_id in request_id_list
            ]
            if all(
                request is not None
                and self._native_decode_control_fast_path_eligible(
                    request,
                    prefill_tokens=0,
                    decode_tokens=1,
                )
                for request in requests
            ):
                self._native_decode_reload_guard_fast_path_count += 1
                return 0
            self._native_decode_reload_guard_full_scan_count += 1
        # Admission only consumes a page count. Do not construct and sort the
        # per-layer HBMReloadPage list for DRAM-backed waiters that cannot fit
        # in HBM yet. A 50K request has tens of thousands of layer-page
        # records, and this path runs once per Decode token.
        return self.kv_cache_manager.get_virtual_context_hbm_reload_required_page_count(
            set(request_id_list)
        )

    def _virtual_context_running_decode_reload_required_page_count(self) -> int:
        return self._virtual_context_reload_required_page_count(
            request.request_id
            for request in self.running
            if not self._debug_has_prefill_remaining(request)
        )

    def _prefill_ring_feasible_with_active_decode(
        self,
        request: Request,
    ) -> tuple[bool, int, int]:
        """Return whether prefill may run while decode is active.

        Ordinary HBM-native prefill remains blocked while decode is active.
        Only provider-backed reload-required prefill, or explicit force-all
        ring experiments, can run in the prefill lane concurrently with decode.
        """
        if not self.virtual_context_decode_hotset_first:
            return False, 0, 0
        if not self._debug_has_prefill_remaining(request):
            return False, 0, 0
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND"):
            return False, 0, 0
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"):
            return False, 0, 0

        force_all = int(
            _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL")
        )
        if force_all:
            return True, 0, force_all

        reload_required_pages = self._virtual_context_reload_required_page_count(
            [request.request_id],
            "prefill",
        )
        return reload_required_pages > 0, reload_required_pages, force_all

    def _virtual_context_ring_pressure_bootstrap_token_budget(self) -> int:
        configured = self.virtual_context_ring_pressure_bootstrap_tokens
        if configured > 0:
            return configured
        candidates = [
            self.virtual_context_active_decode_native_prefill_tokens,
            self.virtual_context_ring_prefill_admission_tokens_max,
            self.max_num_scheduled_tokens,
            self.block_size,
        ]
        return max(token_budget for token_budget in candidates if token_budget > 0)

    def _virtual_context_residency_fixed_ring_chunk_tokens(self) -> int:
        """Return the fixed-quantum assumption used by victim prediction."""

        if self.virtual_context_residency_ring_chunk_tokens > 0:
            return self.virtual_context_residency_ring_chunk_tokens
        candidates = [
            (
                self.ttft_prefill_chunk_scheduler.quantum_tokens
                if self.enable_ttft_prefill_chunk_scheduling
                else 0
            ),
            self.virtual_context_active_decode_ring_prefill_tokens,
            self.virtual_context_ring_prefill_admission_tokens,
            self.virtual_context_ring_min_progress_tokens,
        ]
        positive_candidates = [
            token_budget for token_budget in candidates if token_budget > 0
        ]
        return min(positive_candidates, default=max(1, self.block_size))

    def _virtual_context_residency_reserved_blocks(self) -> int:
        """Return the rolling decode/staging/safety reserve."""

        return (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
            + max(
                self.virtual_context_decode_hotset_reserve_blocks,
                self.virtual_context_ring_bw_decode_growth_blocks,
            )
            + self.virtual_context_ring_bw_safety_blocks
        )

    def _virtual_context_waiting_decode_service_candidate(self) -> Request | None:
        """Next service claimant in the existing queue, never a new policy order."""
        if not (
            getattr(self, "isolation_prefill_batch_kv_budget", False)
            and getattr(self, "virtual_context_reload_to_hbm", False)
        ):
            return None
        return next(
            (request for request in self.waiting
             if self._virtual_context_waiting_decode_ready(request)
             and not request.is_finished()
             and not self.blocks_next_schedule(request, "decode")),
            None,
        )

    def _configure_isolation_waiting_decode_reclaim(self) -> None:
        self.isolation_waiting_decode_reclaim = bool(
            _read_bool_env("VLLM_ISOLATION_WAITING_DECODE_RECLAIM")
            and self.virtual_context_enabled
            and self.pd_decoupled_no_mix
        )
        self._isolation_waiting_decode_reclaim_debug: dict[str, int | str] = {}
        self._isolation_waiting_decode_retirements: set[str] = set()
        if not self.isolation_waiting_decode_reclaim:
            return
        if not (
            self.virtual_context_reload_to_hbm
            and self.isolation_prefill_batch_kv_budget
            and self.isolation_required_d2h_progress
            and self.isolation_phase_waiting_scan
            and self.isolation_waiting_decode_fcfs_scope
            and self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            raise ValueError(
                "WAITING Decode reclaim requires scheduled_prefill_fit, Native "
                "reload, batch KV guard, exact D2H progress and phase WAITING "
                "scan/Decode FCFS scope"
            )
        logger.info(
            "Isolation WAITING Decode reclaim enabled: transactional whole "
            "Prefill victims, independent exact D2H, Future ownership preserved"
        )

    def _isolation_waiting_decode_capacity(self, request: Request) -> tuple[int, int]:
        """Missing prefix + next token, with WAITING's existing reload reserve.

        This is an arithmetic claim, not an allocation or copy-ready receipt.
        Do not import RUNNING's larger Ring runway into near-capacity WAITING
        admission, or use layer-page copy counts as physical block demand.
        """
        logical_tokens = max(
            request.num_computed_tokens,
            request.virtual_context_deferred_logical_computed_tokens,
        )
        required = self.kv_cache_manager.estimate_num_blocks_to_allocate(
            request, 1,
            num_external_computed_tokens=max(
                0, logical_tokens - request.num_computed_tokens,
            ),
            require_native_hbm=True,
        )
        reserve = (
            self.virtual_context_reload_to_hbm_min_free_blocks
            or self.virtual_context_staging_reserved_blocks
        )
        return required, required + reserve

    def _progress_isolation_waiting_decode_reclaim(
        self,
    ) -> tuple[frozenset[str], list[ReleasedPrefillVictim]]:
        """Run before queue traversal; return victims awaiting real D2H/owners.

        Reclaim may prepend victims to WAITING and remove RUNNING entries. It
        must never run after a scan has captured a head/index or allocated a row.
        No capacity retry gate is created: copy completion can authorize a
        release without increasing the free-block count yet.
        """
        debug = self._isolation_waiting_decode_reclaim_debug = {
            "isolation_waiting_decode_reclaim_enabled": 1,
            "isolation_waiting_decode_reclaim_reason": "no_waiter",
        }
        candidate = self._virtual_context_waiting_decode_service_candidate()
        preparation = self._scheduled_prefill_fit_preparation
        if candidate is None:
            if preparation is not None and preparation.next_chunk_tokens == 0:
                target = self.requests.get(preparation.target_request_id)
                if target is None or target.is_finished():
                    self._scheduled_prefill_fit_preparation = None
                    self._scheduled_prefill_fit_preparation_cancelled_count += 1
            return frozenset(), []
        required, target_free = self._isolation_waiting_decode_capacity(candidate)
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        debug.update({
            "isolation_waiting_decode_reclaim_request_id": candidate.request_id,
            "isolation_waiting_decode_reclaim_required_blocks": required,
            "isolation_waiting_decode_reclaim_target_blocks": target_free,
            "isolation_waiting_decode_reclaim_free_blocks": free,
        })
        matching_preparation = bool(
            preparation is not None
            and preparation.target_request_id == candidate.request_id
            and preparation.next_chunk_tokens == 0
        )
        if free >= target_free:
            if matching_preparation:
                # The target fits after an unrelated release. Cancelling the
                # CPU plan does NOT cancel already-owned source DMA/leases.
                self._scheduled_prefill_fit_preparation = None
                self._scheduled_prefill_fit_preparation_cancelled_count += 1
            debug["isolation_waiting_decode_reclaim_reason"] = "capacity_ready"
            return frozenset(), []

        prepared_ids: tuple[str, ...] = ()
        if matching_preparation:
            preparation, ready_pages, planned_pages, ready = (
                self._scheduled_prefill_fit_preparation_status(
                    live_target_request_id=candidate.request_id,
                    required_blocks=target_free - free,
                    hard_required_blocks=target_free - free,
                )
            )
            debug["isolation_waiting_decode_reclaim_ready_pages"] = ready_pages
            debug["isolation_waiting_decode_reclaim_planned_pages"] = planned_pages
            if preparation is not None:
                if not ready:
                    debug["isolation_waiting_decode_reclaim_reason"] = "d2h_or_future_wait"
                    debug["isolation_waiting_decode_reclaim_wait_ids"] = ",".join(
                        preparation.victim_request_ids)
                    return frozenset(preparation.victim_request_ids), []
                prepared_ids = preparation.victim_request_ids

        attempts, failures, released, reasons, final_free = (
            self._reclaim_prefill_victims_until_hbm_target(
                target_free_blocks=target_free,
                excluded_request_ids={candidate.request_id} | self._pd_pending_request_ids,
                prepared_request_ids=prepared_ids,
                require_full_target_before_reclaim=True,
                reclaim_feasibility_cache_key=candidate.request_id,
            )
        )
        debug.update({
            "isolation_waiting_decode_reclaim_attempts": attempts,
            "isolation_waiting_decode_reclaim_failures": failures,
            "isolation_waiting_decode_reclaim_victim_ids": ",".join(
                victim.request_id for victim in released),
            "isolation_waiting_decode_reclaim_released_blocks": sum(
                victim.reclaimed_blocks for victim in released),
            "isolation_waiting_decode_reclaim_final_free_blocks": final_free,
            "isolation_waiting_decode_reclaim_reason": (
                "reclaimed" if released else ",".join(sorted(reasons)) or "no_ready_victim"
            ),
        })
        # A zero-token phase result may never reach the worker. Keep only IDs
        # until a submittable output carries row retirement or rehydration.
        self._isolation_waiting_decode_retirements.update(
            victim.request_id for victim in released)
        return frozenset(), released

    def _commit_isolation_waiting_decode_retirements(
        self, output: SchedulerOutput,
    ) -> None:
        if output.total_num_scheduled_tokens > 0 or output.finished_req_ids:
            self._isolation_waiting_decode_retirements.difference_update(
                output.residency_deferred_req_ids or (),
                output.num_scheduled_tokens,
                output.finished_req_ids,
            )

    def _isolation_waiting_decode_snapshot_plan(
        self, num_scheduled_tokens: dict[str, int],
    ) -> tuple | None:
        """Prepare an exact next-boundary reclaim using a real Prefill producer.

        The current batch may snapshot its own unfinished rows after their KV
        writes. Existing Future owners and rows reaching Decode are excluded.
        No incomplete plan is allowed to stop useful Prefill progress.
        """
        candidate = self._virtual_context_waiting_decode_service_candidate()
        if candidate is None:
            return None
        required, target_free = self._isolation_waiting_decode_capacity(candidate)
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        deficit = max(0, target_free - free)
        if deficit <= 0:
            return None
        empty_plan = (
            candidate, (), candidate.virtual_context_deferred_logical_computed_tokens,
            0, required, target_free - required, free, deficit, (),
        )
        preparation = self._scheduled_prefill_fit_preparation
        if (preparation is not None
                and preparation.target_request_id == candidate.request_id
                and preparation.next_chunk_tokens == 0):
            # Keep the immutable producer/barrier instead of replacing it with
            # a competing SRPT Prefill plan during ordinary Decode recovery.
            return empty_plan
        active = self._active_virtual_context_prefill_requests()
        if not any(num_scheduled_tokens.get(r.request_id, 0) > 0 for r in active):
            return None
        candidates = []
        owners = {}
        for request in active:
            if (request.request_id in self._pd_pending_request_ids
                    or request.is_finished() or self.is_encoder_decoder
                    or request.has_encoder_inputs or request.num_output_tokens > 0):
                continue
            logical = max(request.num_computed_tokens,
                          request.virtual_context_deferred_logical_computed_tokens)
            logical += min(num_scheduled_tokens.get(request.request_id, 0),
                           max(0, request.num_prompt_tokens - logical))
            resident = self.kv_cache_manager.get_request_resident_block_count(
                request.request_id)
            if not (0 < logical < request.num_prompt_tokens and resident > 0):
                continue
            owners[request.request_id] = request
            candidates.append(ResidencyVictimCandidate(
                request_id=request.request_id, arrival_time=request.arrival_time,
                prompt_tokens=request.num_prompt_tokens, logical_context_tokens=logical,
                estimated_reclaimable_blocks=resident,
            ))
        victims = select_best_fit_whole_request_victims(
            candidates, required_blocks=deficit)
        if sum(v.estimated_reclaimable_blocks for v in victims) < deficit:
            return None
        pages = []
        for victim in victims:
            count = (victim.logical_context_tokens + self.block_size - 1) // self.block_size
            for group, blocks in enumerate(self.kv_cache_manager.get_block_ids(victim.request_id)):
                if blocks:
                    pages.extend((victim.request_id, group, page) for page in
                                 self.kv_cache_manager.plan_virtual_context_hbm_snapshot_pages(
                                     request_id=victim.request_id,
                                     logical_tokens=victim.logical_context_tokens,
                                     max_pages=count, group_id=group))
        if len(set(pages)) < deficit:
            return None
        self._scheduled_prefill_fit_last_plan_mode = "waiting_decode_handoff"
        self._scheduled_prefill_fit_last_native_deficit_blocks = deficit
        return (
            candidate, tuple(owners[v.request_id] for v in victims),
            candidate.virtual_context_deferred_logical_computed_tokens,
            0, required, target_free - required, free, deficit, tuple(pages),
        )

    def _virtual_context_waiting_decode_service_reserve(self) -> int:
        """Protect one fitting reload + next-token claim, without owning pages.

        An impossible claim must not stop Ring progress needed to reach a
        future release. This only prevents stealing capacity that is usable by
        the next Decode now; it does not promise admission under overload.
        """
        candidate = self._virtual_context_waiting_decode_service_candidate()
        if candidate is None:
            return 0
        logical_tokens = max(
            candidate.num_computed_tokens,
            candidate.virtual_context_deferred_logical_computed_tokens,
        )
        required = self.kv_cache_manager.estimate_num_blocks_to_allocate(
            candidate, 1,
            num_external_computed_tokens=max(
                0, logical_tokens - candidate.num_computed_tokens,
            ),
            require_native_hbm=True,
        )
        hard_reserve = (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
            + self.virtual_context_decode_hotset_reserve_blocks
            + self.virtual_context_scheduled_prefill_fit_headroom_blocks
        )
        if getattr(self, "isolation_waiting_decode_reclaim", False):
            # Keep a successfully reclaimed WAITING claim reachable even when
            # its legacy reserve is smaller than the Prefill hard headroom.
            # Prefill admission still adds its own hard reserve separately.
            required, minimum = self._isolation_waiting_decode_capacity(candidate)
            hard_reserve = minimum - required
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        reserved = required if required + hard_reserve <= free_blocks else 0
        debug = self._isolation_prefill_batch_budget_debug
        debug["isolation_waiting_decode_candidate"] = 1
        debug["isolation_waiting_decode_required_blocks"] = required
        debug["isolation_waiting_decode_protected_blocks"] = reserved
        debug["isolation_waiting_decode_claim_fits"] = int(
            required + hard_reserve <= free_blocks
        )
        return reserved

    def _scheduled_prefill_fit_active_decode(self) -> bool:
        """Return whether Decode currently owns service or future state."""

        return bool(
            self._debug_running_phase_counts()["num_running_decode_reqs"] > 0
            or self._pd_pending_decode_active()
            or (
                getattr(self, "isolation_prefill_batch_kv_budget", False)
                and getattr(self, "virtual_context_reload_to_hbm", False)
                and self._virtual_context_waiting_decode_service_candidate()
                is not None
            )
        )

    def _isolation_prefill_starts_decode(
        self, request: Request, num_new_tokens: int, *,
        logical_tokens: int | None = None,
    ) -> bool:
        """Final Prefill emits token1; max_tokens=1 leaves no Decode work.

        Pure projection, no request-state/output mutation. For a resumed prompt
        use logical progress, not the number of physical HBM-resident pages.
        """
        if num_new_tokens <= 0 or request.max_tokens <= max(1, request.num_output_tokens):
            return False
        if logical_tokens is None:
            logical_tokens = max(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
        return logical_tokens + num_new_tokens >= request.num_prompt_tokens

    def _scheduled_prefill_fit_headroom_partition(self) -> tuple[int, int]:
        """Return capacity headroom as ``(hard, soft)`` block counts.

        The fraction-derived complement exists to cover Decode KV growth
        while a Prefill allocation or D2H preparation is outstanding. It is
        therefore an admission invariant whenever Decode is active. Without
        active Decode work it remains the legacy one-shot soft runway.
        """

        headroom_blocks = (
            self.virtual_context_scheduled_prefill_fit_headroom_blocks
        )
        if self._scheduled_prefill_fit_active_decode():
            return headroom_blocks, 0
        return 0, headroom_blocks

    def _scheduled_prefill_fit_target(
        self,
        *,
        preferred_request_id: str,
        ring_token_cap: int,
    ) -> tuple[Request | None, int, int, int, int, int]:
        """Snapshot the current next-prefill capacity claim.

        Request arrivals are serialized at scheduler boundaries. This method
        deliberately predicts only the request chosen for this call; a later
        arrival is reranked by the next ``schedule()`` invocation rather than
        causing speculative migration now.
        """

        target: Request | None = None
        if preferred_request_id:
            candidate = self.requests.get(preferred_request_id)
            if (
                candidate is not None
                and not self.blocks_next_schedule(candidate, "prefill")
                and self._debug_has_prefill_remaining(candidate)
                and not self._pd_preemption_protection_reason(candidate)
            ):
                target = candidate
        if target is None:
            target = next(
                (
                    request
                    for request in itertools.chain(self.running, self.waiting)
                    if not self.blocks_next_schedule(request, "prefill")
                    and self._debug_has_prefill_remaining(request)
                    and not self._pd_preemption_protection_reason(request)
                    and not self._virtual_context_waiting_decode_ready(request)
                ),
                None,
            )

        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        hard_headroom_blocks, soft_headroom_blocks = (
            self._scheduled_prefill_fit_headroom_partition()
        )
        hard_reserved_blocks = (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
            + self.virtual_context_decode_hotset_reserve_blocks
            + hard_headroom_blocks
        )
        reserved_blocks = hard_reserved_blocks + soft_headroom_blocks
        if target is None:
            return None, 0, 0, reserved_blocks, free_blocks, 0

        claim = self._scheduled_prefill_fit_capacity_claim(
            target,
            ring_token_cap=ring_token_cap,
            full_native=(
                self._ttft_prefill_execution_mode(target) == "native"
                if self.ttft_prefill_native_chunk_max_tokens > 0
                else self._virtual_context_reload_required_page_count(
                    [target.request_id],
                    "prefill",
                )
                <= 0
            ),
        )
        return (
            target,
            claim.chunk_tokens,
            claim.raw_required_blocks,
            claim.reserved_blocks,
            claim.free_blocks,
            claim.reclaim_blocks,
        )

    def _scheduled_prefill_fit_capacity_claim(
        self,
        target: Request,
        *,
        ring_token_cap: int,
        logical_tokens: int | None = None,
        num_new_computed_tokens: int = 0,
        full_native: bool = True,
        require_full_prompt: bool = False,
    ) -> ScheduledPrefillCapacityClaim:
        """Return the exact capacity claim used by every policy gate.

        ``logical_tokens`` and ``num_new_computed_tokens`` let the proactive
        predictor evaluate the boundary after the current chunk. Admission,
        pressure conversion, and decode-lane diagnostics use the same method
        with the request's durable current progress. ``full_native=False`` is
        reserved for an already provider-backed target that will continue via
        a bounded Context Ring chunk. With next-chunk fit enabled, Native
        reclaim also claims only that chunk. Reload/promotion explicitly sets
        ``require_full_prompt`` to retain its stronger anti-oscillation proof.
        """

        if logical_tokens is None:
            logical_tokens = virtual_context_residency_victim_context_tokens(
                target.num_computed_tokens,
                target.virtual_context_deferred_logical_computed_tokens,
            )
        logical_tokens = min(
            target.num_prompt_tokens,
            max(0, int(logical_tokens)),
        )
        chunk_tokens = min(
            max(0, target.num_prompt_tokens - logical_tokens),
            self.max_num_scheduled_tokens,
        )
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < chunk_tokens:
            chunk_tokens = threshold
        if self.enable_ttft_prefill_chunk_scheduling:
            chunk_tokens = self._ttft_cap_selected_prefill_chunk(
                target,
                chunk_tokens,
                num_computed_tokens=logical_tokens,
                execution_mode=("native" if full_native else "ring"),
            )
        if ring_token_cap > 0 and not full_native:
            chunk_tokens = min(chunk_tokens, ring_token_cap)

        decode_reload_pages = (
            self._virtual_context_running_decode_reload_required_page_count()
        )
        if self.virtual_context_decode_hotset_first and decode_reload_pages > 0:
            cold_cap = self.virtual_context_prefill_ring_tokens_with_decode_cold
            if cold_cap <= 0:
                chunk_tokens = 0
            else:
                chunk_tokens = min(chunk_tokens, cold_cap)
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        hard_headroom_blocks, capacity_headroom_blocks = (
            self._scheduled_prefill_fit_headroom_partition()
        )
        if (getattr(self, "isolation_prefill_batch_kv_budget", False)
                and self._isolation_prefill_starts_decode(
                    target, chunk_tokens, logical_tokens=logical_tokens)):
            # Projected final-Prefill and live admission must agree before the
            # scheduler labels this request Decode or a first token is visible.
            hard_headroom_blocks += capacity_headroom_blocks
            capacity_headroom_blocks = 0
        hard_reserved_blocks = (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
            + self.virtual_context_decode_hotset_reserve_blocks
            + hard_headroom_blocks
        )
        if (
            getattr(self, "isolation_prefill_batch_kv_budget", False)
            and getattr(self, "virtual_context_reload_to_hbm", False)
        ):
            hard_reserved_blocks += (
                self._virtual_context_waiting_decode_service_reserve()
            )
        reserved_blocks = hard_reserved_blocks + capacity_headroom_blocks
        if chunk_tokens <= 0:
            return ScheduledPrefillCapacityClaim(
                request_id=target.request_id,
                logical_tokens=logical_tokens,
                chunk_tokens=0,
                raw_required_blocks=0,
                hard_reserved_blocks=hard_reserved_blocks,
                capacity_headroom_blocks=capacity_headroom_blocks,
                reserved_blocks=reserved_blocks,
                free_blocks=free_blocks,
                reclaim_blocks=0,
            )

        deferred_external_tokens = 0
        durable_logical_tokens = virtual_context_residency_victim_context_tokens(
            target.num_computed_tokens,
            target.virtual_context_deferred_logical_computed_tokens,
        )
        if target.num_computed_tokens == 0 and durable_logical_tokens > 0:
            # A proactive projected claim can include the current chunk in
            # ``logical_tokens``.  Only the already-durable external prefix is
            # passed here; ``num_new_computed_tokens`` accounts for the current
            # chunk separately.
            deferred_external_tokens = durable_logical_tokens
        remaining_prefill_tokens = max(
            0,
            target.num_prompt_tokens - logical_tokens,
        )
        reload_required_pages = self._virtual_context_reload_required_page_count(
            [target.request_id], "prefill"
        )
        raw_required_blocks = (
            self.kv_cache_manager.estimate_num_blocks_to_allocate(
                target,
                (
                    remaining_prefill_tokens
                    if full_native and (
                        require_full_prompt
                        or not getattr(self, "isolation_prefill_fit_next_chunk", False)
                    )
                    else chunk_tokens
                ),
                num_new_computed_tokens=num_new_computed_tokens,
                num_lookahead_tokens=(
                    0
                    if target.num_computed_tokens == 0
                    else self.num_lookahead_tokens
                ),
                num_external_computed_tokens=deferred_external_tokens,
                num_virtual_external_computed_tokens=(
                    deferred_external_tokens
                ),
                require_native_hbm=full_native and reload_required_pages > 0,
            )
        )
        waived_capacity_headroom_blocks = 0
        if (
            getattr(self, "isolation_native_soft_headroom_progress", False)
            and full_native and not require_full_prompt
            and reload_required_pages == 0
            # The shared partition above gives soft capacity headroom only
            # when neither a running Decode nor a pending Decode owns demand.
            and capacity_headroom_blocks > 0
            and raw_required_blocks + hard_reserved_blocks <= free_blocks
        ):
            # Borrow only for a real soft-only shortfall. Keep ordinary claims
            # and full-prompt Native re-promotion's anti-oscillation proof intact.
            if raw_required_blocks + reserved_blocks > free_blocks:
                waived_capacity_headroom_blocks = capacity_headroom_blocks
                capacity_headroom_blocks = 0
                reserved_blocks = hard_reserved_blocks
        reclaim_blocks = scheduled_prefill_reclaim_blocks(
            free_blocks,
            raw_required_blocks,
            reserved_blocks,
        )
        return ScheduledPrefillCapacityClaim(
            request_id=target.request_id,
            logical_tokens=logical_tokens,
            chunk_tokens=chunk_tokens,
            raw_required_blocks=raw_required_blocks,
            hard_reserved_blocks=hard_reserved_blocks,
            capacity_headroom_blocks=capacity_headroom_blocks,
            reserved_blocks=reserved_blocks,
            free_blocks=free_blocks,
            reclaim_blocks=reclaim_blocks,
            waived_capacity_headroom_blocks=waived_capacity_headroom_blocks,
        )

    def _scheduled_prefill_fit_reclaim_limit(
        self,
        request: Request,
        num_new_tokens: int,
    ) -> int | None:
        """Disable page-granular reclaim on scheduled-prefill-fit paths.

        The arguments remain while the experimental partial-residency code is
        parked. Returning ``0`` blocks request-local reactive eviction: the
        proactive best-fit whole-victim path is the only scheduled-prefill-fit capacity
        reclaim authority.
        """

        if (
            self.virtual_context_residency_victim_policy
            != ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            return None
        del request, num_new_tokens
        return 0

    def _scheduled_prefill_fit_ring_fallback_reclaim_requirement(
        self,
        target: Request,
        *,
        next_chunk_tokens: int,
    ) -> tuple[int, int, int, int]:
        """Return Ring raw/reserve and live soft/hard reclaim requirements.

        A native-fit failure changes the capacity objective.  T must be
        released once even when current free HBM already covers one Ring
        chunk, but peer Prefills are needed only when releasing T still leaves
        the bounded Ring chunk and Decode reserves short.  The capacity-
        fraction complement is hard while Decode is active and is otherwise a
        soft Decode-growth runway.
        """

        ring_required_blocks = (
            max(
                1,
                (next_chunk_tokens + self.block_size - 1) // self.block_size,
            )
            if next_chunk_tokens > 0
            else 0
        )
        hard_headroom_blocks, soft_headroom_blocks = (
            self._scheduled_prefill_fit_headroom_partition()
        )
        hard_reserved_blocks = (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
            + self.virtual_context_decode_hotset_reserve_blocks
            + hard_headroom_blocks
        )
        soft_reserved_blocks = hard_reserved_blocks + soft_headroom_blocks
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        target_resident_blocks = (
            self.kv_cache_manager.get_request_resident_block_count(
                target.request_id
            )
        )
        force_target_release = int(target_resident_blocks > 0)
        soft_reclaim_blocks = max(
            force_target_release,
            ring_required_blocks + soft_reserved_blocks - free_blocks,
        )
        hard_reclaim_blocks = max(
            force_target_release,
            ring_required_blocks + hard_reserved_blocks - free_blocks,
        )
        return (
            ring_required_blocks,
            soft_reserved_blocks,
            soft_reclaim_blocks,
            hard_reclaim_blocks,
        )

    def _virtual_context_native_decode_handoff_requirement(
        self,
        request: Request,
    ) -> tuple[int, dict[int, set[int]], int, int, bool]:
        """Return the physical reload and reserve needed before native Decode.

        The capacity-fraction complement is a Decode-growth runway as well as
        a Prefill admission guard.  Applying the same headroom to both the
        proactive prediction and the live handoff prevents a completed
        prefill from consuming the last blocks while its D2H victim plan is in
        flight.
        """

        # Capacity rejection needs only the request-indexed reload count and
        # the allocator's virtual block count. Do not sort and materialize all
        # layer-page descriptors until the complete native handoff can fit.
        # A 10K DRAM-backed request has thousands of layer-pages, and this
        # function can otherwise run once per blocked request per Decode token.
        reload_page_count = (
            self.kv_cache_manager.get_virtual_context_hbm_reload_required_page_count(
                {request.request_id}
            )
        )
        logical_computed_tokens = virtual_context_residency_victim_context_tokens(
            request.num_computed_tokens,
            request.virtual_context_deferred_logical_computed_tokens,
        )
        if reload_page_count <= 0 and logical_computed_tokens <= 0:
            # A newly admitted HBM-native request can finish its whole prompt
            # in the first scheduled Prefill chunk.  The proactive planner
            # sees the projected Prefill-to-Decode boundary before that chunk
            # has executed, so the durable progress counter is still zero.
            # There is no DRAM-authoritative page to reload at this boundary;
            # retained-table validation belongs to the later, recovered
            # request state.  Do not weaken the allocator's positive-progress
            # invariant for real DRAM-backed handoffs.
            return 0, {}, 0, 0, False
        if logical_computed_tokens <= 0:
            raise TieredAllocatorInvariantError(
                "native Decode reload requires positive logical progress: "
                f"request={request.request_id} reload_pages={reload_page_count}"
            )
        estimated_required_blocks = (
            self.kv_cache_manager.estimate_num_blocks_to_allocate(
                request,
                0,
                num_external_computed_tokens=max(
                    0, logical_computed_tokens - request.num_computed_tokens,
                ),
                require_native_hbm=True,
            )
        )
        extra_reserve_blocks = (
            self.virtual_context_reload_to_hbm_min_free_blocks
            or self.virtual_context_staging_reserved_blocks
        )
        progress_reserve_applied = False
        if self._has_pending_prefill_for_ring_progress():
            (
                _progress_tokens,
                progress_reserve_blocks,
                progress_hard_reserved_blocks,
            ) = self._virtual_context_active_decode_ring_progress_reserve()
            if progress_reserve_blocks > 0:
                # A whole 2K Ring quantum is useful throughput headroom, but it
                # is not a correctness reserve. A DRAM-backed Decode
                # continuation may revoke that soft quantum while retaining
                # staging/Decode-growth blocks and fractional headroom.
                extra_reserve_blocks = max(
                    extra_reserve_blocks,
                    progress_hard_reserved_blocks,
                )
                progress_reserve_applied = (
                    progress_hard_reserved_blocks > 0
                )
        if (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            # Fractional headroom absorbs Decode growth *after* preserving the
            # hard native/ring progress reserve; it must not alias staging.
            extra_reserve_blocks += (
                self.virtual_context_scheduled_prefill_fit_headroom_blocks
            )
        estimated_min_free_blocks = (
            estimated_required_blocks + extra_reserve_blocks
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        insufficient_capacity = (
            reload_page_count > 0
            and free_blocks < estimated_min_free_blocks
        )
        if insufficient_capacity:
            self._native_decode_handoff_capacity_precheck_count += 1
            self._native_decode_handoff_plan_avoided_page_count += (
                reload_page_count
            )
        if insufficient_capacity:
            return (
                reload_page_count,
                {},
                estimated_required_blocks,
                estimated_min_free_blocks,
                progress_reserve_applied,
            )

        reload_pages = self.kv_cache_manager.plan_virtual_context_hbm_reload_pages(
            request_ids={request.request_id},
        )
        logical_pages_by_group: dict[int, set[int]] = defaultdict(set)
        for page in reload_pages:
            logical_pages_by_group[page.kv_cache_group_id].add(
                page.logical_token_page
            )
        # Copy every required layer/page, but allocate only missing physical
        # destinations. Existing concrete slots remain DRAM-authoritative until
        # their real copy ACK; reusing them grants no premature content validity.
        required_blocks = estimated_required_blocks
        return (
            len(reload_pages),
            logical_pages_by_group,
            required_blocks,
            required_blocks + extra_reserve_blocks,
            progress_reserve_applied,
        )

    def _scheduled_prefill_fit_proactive_snapshot_plan(
        self,
        num_scheduled_tokens: dict[str, int],
        *,
        ring_token_cap: int,
    ) -> tuple[
        Request | None,
        tuple[Request, ...],
        int,
        int,
        int,
        int,
        int,
        int,
        tuple[tuple[str, int, int], ...],
    ]:
        """Predict the next prefill boundary after this iteration's chunk.

        Allocation for the current SchedulerOutput has already mutated the
        block pool. The returned deficit therefore describes the *following*
        prefill chunk. Every resident page of each selected best-fit request is
        snapshotted during the current forward, but remains HBM-owned until the
        next boundary revalidates the target and commits whole-request reclaim.
        """

        self._scheduled_prefill_fit_last_plan_mode = "none"
        self._scheduled_prefill_fit_last_native_deficit_blocks = 0
        self._scheduled_prefill_fit_last_ring_fallback_deficit_blocks = 0
        self._scheduled_prefill_fit_last_avoided_peer_blocks = 0

        if (
            self.virtual_context_residency_victim_policy
            != ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            return None, (), 0, 0, 0, 0, 0, 0, ()

        if getattr(self, "isolation_waiting_decode_reclaim", False):
            waiting_plan = self._isolation_waiting_decode_snapshot_plan(
                num_scheduled_tokens)
            if waiting_plan is not None:
                return waiting_plan

        scheduled_request: Request | None = None
        scheduled_request_chunk_tokens = 0
        scheduled_prefill_chunk_tokens_by_id: dict[str, int] = {}
        projected_logical_tokens_by_id: dict[str, int] = {}
        for request_id, scheduled_tokens in num_scheduled_tokens.items():
            request = self.requests.get(request_id)
            if request is None or scheduled_tokens <= 0:
                continue
            logical_tokens = virtual_context_residency_victim_context_tokens(
                request.num_computed_tokens,
                request.virtual_context_deferred_logical_computed_tokens,
            )
            prefill_tokens = min(
                scheduled_tokens,
                max(0, request.num_prompt_tokens - logical_tokens),
            )
            if prefill_tokens <= 0:
                continue
            if scheduled_request is None:
                scheduled_request = request
                scheduled_request_chunk_tokens = prefill_tokens
            scheduled_prefill_chunk_tokens_by_id[request_id] = prefill_tokens
            projected_logical_tokens_by_id[request_id] = min(
                request.num_prompt_tokens,
                logical_tokens + prefill_tokens,
            )

        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        reserved_blocks = (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
            + self.virtual_context_decode_hotset_reserve_blocks
            + self.virtual_context_scheduled_prefill_fit_headroom_blocks
        )
        if scheduled_request is None:
            return None, (), 0, 0, 0, reserved_blocks, free_blocks, 0, ()

        active_prefills = self._active_virtual_context_prefill_requests()
        candidates_by_id: dict[str, Request] = {}
        projected_candidates: list[ProjectedPrefillCandidate] = []
        for request in active_prefills:
            if (
                request.request_id in self._pd_pending_request_ids
                or self._virtual_context_waiting_decode_ready(request)
            ):
                continue
            logical_tokens = projected_logical_tokens_by_id.get(
                request.request_id,
                virtual_context_residency_victim_context_tokens(
                    request.num_computed_tokens,
                    request.virtual_context_deferred_logical_computed_tokens,
                ),
            )
            candidate = ProjectedPrefillCandidate(
                request_id=request.request_id,
                arrival_time=request.arrival_time,
                prompt_tokens=request.num_prompt_tokens,
                projected_logical_tokens=logical_tokens,
            )
            if candidate.remaining_prefill_tokens <= 0:
                continue
            candidates_by_id.setdefault(request.request_id, request)
            projected_candidates.append(candidate)

        projected_active_candidates: list[ProjectedPrefillCandidate] = []
        for request in active_prefills:
            if self._virtual_context_waiting_decode_ready(request):
                continue
            logical_tokens = projected_logical_tokens_by_id.get(
                request.request_id,
                virtual_context_residency_victim_context_tokens(
                    request.num_computed_tokens,
                    request.virtual_context_deferred_logical_computed_tokens,
                ),
            )
            candidate = ProjectedPrefillCandidate(
                request_id=request.request_id,
                arrival_time=request.arrival_time,
                prompt_tokens=request.num_prompt_tokens,
                projected_logical_tokens=logical_tokens,
            )
            if candidate.remaining_prefill_tokens > 0:
                projected_active_candidates.append(candidate)

        projected_order: tuple[str, ...] = ()
        if (
            self.enable_ttft_prefill_chunk_scheduling
            and self.ttft_prefill_chunk_scheduler.policy
            == PrefillChunkSchedulingPolicy.SRPT
        ):
            projected_order = projected_srpt_prefill_order(
                projected_candidates
            )
        elif self.enable_ttft_prefill_chunk_scheduling:
            runnable_ids = {
                candidate.request_id for candidate in projected_candidates
            }
            ordered_ids = [
                request_id
                for request_id in self.ttft_prefill_chunk_scheduler.candidate_order
                if request_id in runnable_ids
            ]
            ordered_id_set = set(ordered_ids)
            ordered_ids.extend(
                candidate.request_id
                for candidate in sorted(
                    projected_candidates,
                    key=lambda candidate: (
                        float(candidate.arrival_time),
                        candidate.request_id,
                    ),
                )
                if candidate.request_id not in ordered_id_set
            )
            projected_order = tuple(ordered_ids)
        elif projected_candidates:
            projected_order = tuple(
                candidate.request_id
                for candidate in sorted(
                    projected_candidates,
                    key=lambda candidate: (
                        float(candidate.arrival_time),
                        candidate.request_id,
                    ),
                )
            )
        scheduled_projected_logical_tokens = projected_logical_tokens_by_id[
            scheduled_request.request_id
        ]
        planning_decode_handoff = False
        if (
            getattr(self, "virtual_context_reload_to_hbm", False)
            and scheduled_projected_logical_tokens
            >= scheduled_request.num_prompt_tokens
        ):
            (
                _reload_page_records,
                _reload_pages_by_group,
                handoff_required_blocks,
                handoff_min_free_blocks,
                _progress_reserve_applied,
            ) = self._virtual_context_native_decode_handoff_requirement(
                scheduled_request
            )
            planning_decode_handoff = handoff_required_blocks > 0

        if planning_decode_handoff:
            # The next immutable boundary for this request is native Decode,
            # not another Prefill chunk. Snapshot the best-fit victim pages
            # needed to install the complete HBM decode table while retaining
            # the configured Decode-growth headroom.
            target = scheduled_request
            target_logical_tokens = scheduled_projected_logical_tokens
            next_chunk_tokens = 0
            raw_required_blocks = handoff_required_blocks
            reserved_blocks = max(
                0, handoff_min_free_blocks - handoff_required_blocks
            )
            predicted_deficit_blocks = max(
                0, handoff_min_free_blocks - free_blocks
            )
        else:
            target_request_id = projected_order[0] if projected_order else ""
            target = candidates_by_id.get(target_request_id)
            if target is None:
                return None, (), 0, 0, 0, reserved_blocks, free_blocks, 0, ()

            target_logical_tokens = next(
                candidate.projected_logical_tokens
                for candidate in projected_candidates
                if candidate.request_id == target_request_id
            )
            target_current_chunk_tokens = (
                scheduled_prefill_chunk_tokens_by_id.get(
                    target.request_id,
                    0,
                )
            )
            capacity_claim = self._scheduled_prefill_fit_capacity_claim(
                target,
                ring_token_cap=ring_token_cap,
                logical_tokens=target_logical_tokens,
                num_new_computed_tokens=target_current_chunk_tokens,
                full_native=(
                    self._ttft_prefill_execution_mode(target) == "native"
                    if self.ttft_prefill_native_chunk_max_tokens > 0
                    else self._virtual_context_reload_required_page_count(
                        [target.request_id],
                        "prefill",
                    )
                    <= 0
                ),
            )
            next_chunk_tokens = capacity_claim.chunk_tokens
            raw_required_blocks = capacity_claim.raw_required_blocks
            reserved_blocks = capacity_claim.reserved_blocks
            free_blocks = capacity_claim.free_blocks
            predicted_deficit_blocks = capacity_claim.reclaim_blocks
        self._scheduled_prefill_fit_last_native_deficit_blocks = (
            predicted_deficit_blocks
        )
        if next_chunk_tokens <= 0 and not planning_decode_handoff:
            return (
                target,
                (),
                target_logical_tokens,
                0,
                0,
                reserved_blocks,
                free_blocks,
                0,
                (),
            )
        active_by_id = {
            request.request_id: request for request in active_prefills
        }
        victims: list[Request] = []
        snapshot_pages_list: list[tuple[str, int, int]] = []
        protected_request_ids = set(num_scheduled_tokens)
        protected_request_ids.add(target.request_id)
        victim_candidates: list[ResidencyVictimCandidate] = []
        for candidate in projected_active_candidates:
            victim_request_id = candidate.request_id
            if victim_request_id in protected_request_ids:
                self._scheduled_prefill_fit_protected_victim_skip_count = (
                    getattr(
                        self,
                        "_scheduled_prefill_fit_protected_victim_skip_count",
                        0,
                    )
                    + 1
                )
                continue
            victim = active_by_id.get(victim_request_id)
            if (
                victim is None
                or self.is_encoder_decoder
                or victim.has_encoder_inputs
            ):
                continue
            victim_logical_tokens = projected_logical_tokens_by_id.get(
                victim.request_id,
                virtual_context_residency_victim_context_tokens(
                    victim.num_computed_tokens,
                    victim.virtual_context_deferred_logical_computed_tokens,
                ),
            )
            resident_blocks = (
                self.kv_cache_manager.get_request_resident_block_count(
                    victim.request_id
                )
            )
            if (
                victim_logical_tokens <= 0
                or victim_logical_tokens >= victim.num_prompt_tokens
                or resident_blocks <= 0
            ):
                continue
            victim_candidates.append(
                ResidencyVictimCandidate(
                    request_id=victim.request_id,
                    arrival_time=victim.arrival_time,
                    prompt_tokens=victim.num_prompt_tokens,
                    logical_context_tokens=victim_logical_tokens,
                    estimated_reclaimable_blocks=resident_blocks,
                )
            )

        native_victim_plan = select_best_fit_whole_request_victims(
            victim_candidates,
            required_blocks=predicted_deficit_blocks,
        )

        def collect_peer_snapshot_pages(
            victim_plan: tuple[ResidencyVictimCandidate, ...],
        ) -> tuple[list[Request], list[tuple[str, int, int]]] | None:
            planned_victims: list[Request] = []
            planned_pages: list[tuple[str, int, int]] = []
            for victim_candidate in victim_plan:
                victim = active_by_id[victim_candidate.request_id]
                if victim.request_id in self._pd_pending_request_ids:
                    # Keep the selected best-fit set immutable. A future-owned
                    # peer cannot be replaced only to make D2H preparation
                    # appear ready at this boundary.
                    return None
                victim_logical_tokens = projected_logical_tokens_by_id.get(
                    victim.request_id,
                    virtual_context_residency_victim_context_tokens(
                        victim.num_computed_tokens,
                        victim.virtual_context_deferred_logical_computed_tokens,
                    ),
                )
                victim_snapshot_pages: list[tuple[str, int, int]] = []
                logical_page_count = (
                    victim_logical_tokens + self.block_size - 1
                ) // self.block_size
                for group_id, block_ids in enumerate(
                    self.kv_cache_manager.get_block_ids(victim.request_id)
                ):
                    if not block_ids:
                        continue
                    logical_pages = (
                        self.kv_cache_manager.plan_virtual_context_hbm_snapshot_pages(
                            request_id=victim.request_id,
                            logical_tokens=victim_logical_tokens,
                            max_pages=logical_page_count,
                            group_id=group_id,
                        )
                    )
                    victim_snapshot_pages.extend(
                        (victim.request_id, group_id, logical_page)
                        for logical_page in logical_pages
                    )
                if not victim_snapshot_pages:
                    continue
                planned_victims.append(victim)
                planned_pages.extend(victim_snapshot_pages)
            return planned_victims, planned_pages

        native_snapshot_plan = collect_peer_snapshot_pages(native_victim_plan)
        if native_snapshot_plan is None:
            return (
                target,
                (),
                target_logical_tokens,
                next_chunk_tokens,
                raw_required_blocks,
                reserved_blocks,
                free_blocks,
                predicted_deficit_blocks,
                (),
            )
        victims, snapshot_pages_list = native_snapshot_plan
        native_peer_blocks = sum(
            candidate.estimated_reclaimable_blocks
            for candidate in native_victim_plan
        )
        native_plan_covers_deficit = (
            len(snapshot_pages_list) >= predicted_deficit_blocks
        )
        if native_plan_covers_deficit or planning_decode_handoff:
            self._scheduled_prefill_fit_last_plan_mode = (
                "decode_handoff" if planning_decode_handoff else "native_fit"
            )
            snapshot_pages = tuple(snapshot_pages_list)
            return (
                target,
                tuple(victims),
                target_logical_tokens,
                next_chunk_tokens,
                raw_required_blocks,
                reserved_blocks,
                free_blocks,
                predicted_deficit_blocks,
                snapshot_pages,
            )

        # Peer capacity cannot preserve T's full native Prefill. Discard the
        # native-fit peer plan rather than appending T after it. Recompute the
        # objective for one bounded Context Ring chunk plus Decode reserves.
        if self.is_encoder_decoder or target.has_encoder_inputs:
            return (
                target,
                (),
                target_logical_tokens,
                next_chunk_tokens,
                raw_required_blocks,
                reserved_blocks,
                free_blocks,
                predicted_deficit_blocks,
                (),
            )
        target_snapshot_pages: list[tuple[str, int, int]] = []
        target_page_count = (
            target_logical_tokens + self.block_size - 1
        ) // self.block_size
        for group_id, block_ids in enumerate(
            self.kv_cache_manager.get_block_ids(target.request_id)
        ):
            if not block_ids:
                continue
            logical_pages = (
                self.kv_cache_manager.plan_virtual_context_hbm_snapshot_pages(
                    request_id=target.request_id,
                    logical_tokens=target_logical_tokens,
                    max_pages=target_page_count,
                    group_id=group_id,
                )
            )
            target_snapshot_pages.extend(
                (target.request_id, group_id, logical_page)
                for logical_page in logical_pages
            )
        if not target_snapshot_pages:
            return (
                target,
                (),
                target_logical_tokens,
                next_chunk_tokens,
                raw_required_blocks,
                reserved_blocks,
                free_blocks,
                predicted_deficit_blocks,
                (),
            )

        (
            _ring_required_blocks,
            ring_reserved_blocks,
            ring_reclaim_blocks,
            _ring_hard_reclaim_blocks,
        ) = self._scheduled_prefill_fit_ring_fallback_reclaim_requirement(
            target,
            next_chunk_tokens=next_chunk_tokens,
        )
        target_resident_blocks = (
            self.kv_cache_manager.get_request_resident_block_count(
                target.request_id
            )
        )
        peer_ring_deficit_blocks = max(
            0,
            ring_reclaim_blocks - target_resident_blocks,
        )
        ring_peer_plan = select_best_fit_whole_request_victims(
            victim_candidates,
            required_blocks=peer_ring_deficit_blocks,
        )
        ring_peer_snapshot_plan = collect_peer_snapshot_pages(ring_peer_plan)
        if ring_peer_snapshot_plan is None:
            return (
                target,
                (),
                target_logical_tokens,
                next_chunk_tokens,
                raw_required_blocks,
                ring_reserved_blocks,
                free_blocks,
                ring_reclaim_blocks,
                (),
            )
        ring_peer_victims, ring_peer_snapshot_pages = ring_peer_snapshot_plan
        victims = [target, *ring_peer_victims]
        snapshot_pages_list = [
            *target_snapshot_pages,
            *ring_peer_snapshot_pages,
        ]
        ring_peer_blocks = sum(
            candidate.estimated_reclaimable_blocks
            for candidate in ring_peer_plan
        )
        self._scheduled_prefill_fit_last_plan_mode = "ring_fallback"
        self._scheduled_prefill_fit_last_ring_fallback_deficit_blocks = (
            ring_reclaim_blocks
        )
        self._scheduled_prefill_fit_last_avoided_peer_blocks = max(
            0,
            native_peer_blocks - ring_peer_blocks,
        )
        predicted_deficit_blocks = ring_reclaim_blocks
        reserved_blocks = ring_reserved_blocks
        snapshot_pages = tuple(snapshot_pages_list)
        return (
            target,
            tuple(victims),
            target_logical_tokens,
            next_chunk_tokens,
            raw_required_blocks,
            reserved_blocks,
            free_blocks,
            predicted_deficit_blocks,
            snapshot_pages,
        )

    def _scheduled_prefill_fit_preparation_status(
        self,
        *,
        live_target_request_id: str,
        required_blocks: int,
        hard_required_blocks: int | None = None,
    ) -> tuple[ScheduledPrefillFitPreparation | None, int, int, bool]:
        """Revalidate whether a proactive whole-request D2H plan may commit.

        A prepared plan may consume part of the configured fractional capacity
        headroom when Decode grows while D2H is in flight. It is cancelled
        only when it no longer covers the live allocator-safety deficit.
        """

        preparation = self._scheduled_prefill_fit_preparation
        if required_blocks <= 0:
            if preparation is not None:
                self._scheduled_prefill_fit_preparation = None
                self._scheduled_prefill_fit_preparation_cancelled_count += 1
            return None, 0, 0, True
        if preparation is None:
            self._scheduled_prefill_fit_unprepared_reclaim_count += 1
            return None, 0, 0, False
        target_matches = (
            preparation.target_request_id == live_target_request_id
        )
        if not target_matches:
            self._scheduled_prefill_fit_preparation_target_mismatch_count += 1
        active_prefill_request_ids = {
            request.request_id
            for request in self._active_virtual_context_prefill_requests()
        }
        victims = [
            self.requests.get(request_id)
            for request_id in preparation.victim_request_ids
        ]
        if (
            not victims
            or any(victim is None for victim in victims)
            or any(
                not self._debug_has_prefill_remaining(victim)
                for victim in victims
                if victim is not None
            )
            or any(
                request_id not in active_prefill_request_ids
                for request_id in preparation.victim_request_ids
            )
        ):
            self._scheduled_prefill_fit_preparation = None
            self._scheduled_prefill_fit_preparation_cancelled_count += 1
            return None, 0, 0, False

        planned_keys = set(preparation.snapshot_pages)
        if getattr(self, "isolation_logical_release_readiness", False):
            ready_candidates = (
                self.kv_cache_manager.summarize_virtual_context_hbm_release_readiness(
                    request_ids=set(preparation.victim_request_ids),
                    allow_partial_pages=True,
                    page_keys=planned_keys,
                )
            )
        else:
            ready_candidates = (
                self.kv_cache_manager.plan_virtual_context_hbm_release_candidates(
                    request_ids=set(preparation.victim_request_ids),
                    allow_partial_pages=True,
                )
            )
        ready_keys = {
            (
                candidate.request_id,
                candidate.kv_cache_group_id,
                candidate.logical_token_page,
            )
            for candidate in ready_candidates
        }
        minimum_required_blocks = (
            required_blocks
            if hard_required_blocks is None or not target_matches
            else min(required_blocks, max(0, int(hard_required_blocks)))
        )
        if (
            len(planned_keys) < minimum_required_blocks
            and not getattr(preparation, "target_self_victim", False)
        ):
            # A later SRPT arrival can increase the live capacity claim. The
            # old exact plan is still safe provider work, but it is not a
            # sufficient allocator-safety barrier for the new winner. Fall
            # back to a fresh bounded preparation boundary instead of waiting
            # forever. A soft-headroom-only shortfall is accepted below.
            self._scheduled_prefill_fit_preparation = None
            self._scheduled_prefill_fit_preparation_cancelled_count += 1
            self._scheduled_prefill_fit_unprepared_reclaim_count += 1
            return None, len(planned_keys & ready_keys), len(planned_keys), False
        ready_page_count = len(planned_keys & ready_keys)
        planned_page_count = len(planned_keys)
        victims_quiescent = all(
            request_id not in self._pd_pending_request_ids
            for request_id in preparation.victim_request_ids
        )
        ready = (
            victims_quiescent
            and planned_page_count > 0
            and ready_page_count == planned_page_count
        )
        return preparation, ready_page_count, planned_page_count, ready

    def _scheduled_prefill_fit_default_reclaim_limit(
        self,
        request: Request,
    ) -> int | None:
        if (
            self.virtual_context_residency_victim_policy
            != ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            return None
        del request
        # Partial residency is intentionally parked. ``None`` selects the
        # whole-request provider transition on reactive fallback paths too.
        return None

    def _virtual_context_active_decode_ring_progress_reserve(
        self,
    ) -> tuple[int, int, int]:
        """Return tokens, total blocks, and hard non-progress headroom.

        ``VLLM_PD_RING_MIN_PREFILL_TOKENS_WITH_DECODE`` is default zero and is
        the explicit gate for this liveness reserve. The returned total block
        count includes one bounded prefill quantum plus staging and decode
        growth; the adaptive safety reserve remains a soft admission target.
        """
        if (
            not self.virtual_context_enabled
            or not self.virtual_context_decode_hotset_first
            or not self.virtual_context_ring_residency_admission
            or self.pd_ring_min_prefill_tokens_with_decode <= 0
            or not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND")
            or not _read_bool_env(
                "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"
            )
        ):
            return 0, 0, 0

        progress_tokens = self.pd_ring_min_prefill_tokens_with_decode
        if self.virtual_context_active_decode_ring_prefill_tokens > 0:
            progress_tokens = min(
                progress_tokens,
                self.virtual_context_active_decode_ring_prefill_tokens,
            )
        configured_cap = (
            self.virtual_context_ring_prefill_admission_tokens_max
            or self.virtual_context_ring_prefill_admission_tokens
        )
        if configured_cap > 0:
            progress_tokens = min(progress_tokens, configured_cap)
        progress_tokens = (
            progress_tokens // self.block_size
        ) * self.block_size
        if progress_tokens <= 0:
            return 0, 0, 0

        staging_blocks = (
            self.virtual_context_staging_reserved_blocks
            + self.virtual_context_ring_min_free_blocks
        )
        decode_growth_blocks = max(
            self.virtual_context_decode_hotset_reserve_blocks,
            self.virtual_context_ring_bw_decode_growth_blocks,
        )
        reserve_blocks = (
            virtual_context_active_decode_ring_progress_reserve_blocks(
                progress_tokens,
                self.block_size,
                staging_blocks,
                decode_growth_blocks,
            )
        )
        hard_reserved_blocks = staging_blocks + decode_growth_blocks
        return progress_tokens, reserve_blocks, hard_reserved_blocks

    def _virtual_context_active_decode_for_ring_progress(self) -> bool:
        return (
            self._debug_running_decode_resident_blocks() > 0
            or self._pd_pending_decode_active()
        )

    def _has_pending_prefill_for_ring_progress(self) -> bool:
        return any(
            self._debug_needs_prefill_lane(request)
            for request in self.running
        ) or any(
            not self._virtual_context_waiting_decode_ready(request)
            for request in self.waiting
        )

    def _context_ring_pressure_threshold_blocks(
        self,
    ) -> tuple[int, int, int, int]:
        """Return the HBM-free threshold before forcing context ring.

        The pressure path should not mark a prefill request as ring-required
        while there is enough free HBM for both:
        1. a bounded native bootstrap prefill chunk, and
        2. the maximum context-ring admission window.

        On the T4 4GiB-KV experiments this defaults to 16K bootstrap +
        16K ring window, plus the explicit staging/decode-growth/safety
        reserves. Below this threshold we treat the system as close enough to
        KV pressure to prefer context ring over ordinary HBM-native prefill.
        """
        bootstrap_tokens = (
            self._virtual_context_ring_pressure_bootstrap_token_budget()
        )
        ring_tokens = max(
            self.virtual_context_ring_prefill_admission_tokens_max,
            self.virtual_context_ring_prefill_admission_tokens,
            self.max_num_scheduled_tokens,
            self.block_size,
        )
        bootstrap_blocks = max(
            1,
            (bootstrap_tokens + self.block_size - 1) // self.block_size,
        )
        ring_blocks = max(
            1,
            (ring_tokens + self.block_size - 1) // self.block_size,
        )
        reserved_blocks = self._virtual_context_residency_reserved_blocks()
        threshold_blocks = bootstrap_blocks + ring_blocks + reserved_blocks
        return (
            threshold_blocks,
            bootstrap_blocks,
            ring_blocks,
            reserved_blocks,
        )

    def _prefill_native_safe_before_ring_pressure(
        self,
        request: Request,
    ) -> tuple[bool, int, int, int, int, int, str]:
        """Return whether active-decode prefill can stay HBM-native."""
        if not self.virtual_context_decode_hotset_first:
            return False, 0, 0, 0, 0, 0, "decode_hotset_first_disabled"
        if not self._debug_has_prefill_remaining(request):
            return False, 0, 0, 0, 0, 0, "not_prefill"
        if _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL"):
            return False, 0, 0, 0, 0, 0, "force_all_ring"
        reload_required_pages = self._virtual_context_reload_required_page_count(
            [request.request_id],
            "prefill",
        )
        if reload_required_pages > 0:
            return (
                False,
                reload_required_pages,
                0,
                0,
                0,
                0,
                "reload_required",
            )

        if (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        ):
            # The generic pressure threshold reserves a full bootstrap plus a
            # maximum ring window.  scheduled_prefill_fit instead prepares an
            # an exact Native capacity claim, so using that coarse threshold
            # here can reject a target that the policy has already proved fits and
            # request a zero-page reclaim forever.
            claim = self._scheduled_prefill_fit_capacity_claim(
                request,
                ring_token_cap=(
                    self._dynamic_virtual_context_ring_prefill_admission_tokens()
                    or self._virtual_context_residency_fixed_ring_chunk_tokens()
                ),
                full_native=(
                    self._virtual_context_reload_required_page_count(
                        [request.request_id],
                        "prefill",
                    )
                    <= 0
                ),
            )
            decode_growth_runway = (
                self._scheduled_prefill_fit_decode_growth_runway
            )
            decode_growth_runway_covers = bool(
                decode_growth_runway is not None
                and decode_growth_runway.covers(claim)
            )
            native_safe = (
                claim.reclaim_blocks <= 0 or decode_growth_runway_covers
            )
            threshold_blocks = claim.required_blocks_with_reserve
            reason = (
                "scheduled_prefill_fit_decode_growth_runway"
                if decode_growth_runway_covers
                else (
                    "scheduled_prefill_fit_full_prefill_available"
                    if native_safe
                    else "scheduled_prefill_fit_full_prefill_deficit"
                )
            )
            if getattr(self, "isolation_prefill_fit_next_chunk", False):
                reason = reason.replace("full_prefill", "next_chunk")
            self._virtual_context_ring_pressure_last_free_blocks = (
                claim.free_blocks
            )
            self._virtual_context_ring_pressure_last_threshold_blocks = (
                threshold_blocks
            )
            self._virtual_context_ring_pressure_last_bootstrap_blocks = (
                claim.raw_required_blocks
            )
            self._virtual_context_ring_pressure_last_ring_blocks = 0
            self._virtual_context_ring_pressure_last_reserved_blocks = (
                claim.reserved_blocks
            )
            self._virtual_context_ring_pressure_last_native_safe = int(
                native_safe
            )
            self._virtual_context_ring_pressure_last_reason = reason
            return (
                native_safe,
                reload_required_pages,
                claim.free_blocks,
                threshold_blocks,
                claim.raw_required_blocks,
                0,
                reason,
            )

        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        (
            threshold_blocks,
            bootstrap_blocks,
            ring_blocks,
            reserved_blocks,
        ) = self._context_ring_pressure_threshold_blocks()
        native_safe = free_blocks >= threshold_blocks
        reason = "native_safe_before_ring_pressure" if native_safe else (
            "below_ring_pressure_threshold"
        )

        self._virtual_context_ring_pressure_last_free_blocks = free_blocks
        self._virtual_context_ring_pressure_last_threshold_blocks = (
            threshold_blocks
        )
        self._virtual_context_ring_pressure_last_bootstrap_blocks = (
            bootstrap_blocks
        )
        self._virtual_context_ring_pressure_last_ring_blocks = ring_blocks
        self._virtual_context_ring_pressure_last_reserved_blocks = (
            reserved_blocks
        )
        self._virtual_context_ring_pressure_last_native_safe = int(native_safe)
        self._virtual_context_ring_pressure_last_reason = reason
        return (
            native_safe,
            reload_required_pages,
            free_blocks,
            threshold_blocks,
            bootstrap_blocks,
            ring_blocks,
            reason,
        )

    def _prefill_bootstrap_feasible_with_active_decode(
        self,
        request: Request,
    ) -> bool:
        """Allow the first bounded prefill chunk while decode is active.

        A new prefill request has no completed KV pages that can be released to
        DRAM yet, so it cannot be ring-required on its first scheduling step.
        The bounded first chunk creates provider-backed completed pages; a later
        running-prefill pass can release those pages and resume via context ring.
        """
        if not self.virtual_context_decode_hotset_first:
            return False
        if not self.virtual_context_defer_prefill:
            return False
        if not self.virtual_context_tiered_external_blocks:
            return False
        if not self.virtual_context_ring_residency_admission:
            return False
        if self._dynamic_virtual_context_ring_prefill_admission_tokens() <= 0:
            return False
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND"):
            return False
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"):
            return False
        if not _read_bool_env("VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM"):
            return False
        if not self._debug_has_prefill_remaining(request):
            return False
        if request.num_computed_tokens > 0:
            return False
        if request.virtual_context_deferred_logical_computed_tokens > 0:
            return False
        if request.num_preemptions > 0:
            return False
        return True

    def _try_make_running_prefill_ring_required(
        self,
        request: Request,
    ) -> tuple[bool, str, int]:
        """Release completed prefill KV so the next step must use context ring."""
        if not self.virtual_context_decode_hotset_first:
            return False, "decode_hotset_first_disabled", 0
        if request.num_computed_tokens <= 0:
            return False, "no_progress", 0
        return self._try_defer_running_prefill_for_virtual_context(
            request,
            max_reclaim_blocks=(
                self._scheduled_prefill_fit_default_reclaim_limit(request)
            ),
        )

    def _virtual_context_decode_cold_streaming_slack_ms(
        self, now: float
    ) -> float:
        if self._virtual_context_last_decode_schedule_ts <= 0:
            return self.virtual_context_decode_cold_streaming_slo_ms
        elapsed_ms = max(
            0.0,
            (now - self._virtual_context_last_decode_schedule_ts) * 1000.0,
        )
        return self.virtual_context_decode_cold_streaming_slo_ms - elapsed_ms

    def _virtual_context_decode_cold_streaming_allowed(
        self, now: float
    ) -> tuple[bool, float]:
        if not self.virtual_context_decode_cold_streaming:
            return False, self._virtual_context_decode_cold_streaming_slack_ms(now)
        slack_ms = self._virtual_context_decode_cold_streaming_slack_ms(now)
        return (
            slack_ms
            >= self.virtual_context_decode_cold_streaming_min_slack_ms,
            slack_ms,
        )

    def _virtual_context_decode_after_prefill_cold_streaming_allowed(
        self, request: Request
    ) -> bool:
        if not self.virtual_context_decode_cold_streaming_on_prefill_done:
            return False
        if self._debug_has_prefill_remaining(request):
            return False
        return request.num_computed_tokens >= request.num_prompt_tokens

    def _check_virtual_context_decode_cold_streaming_admission(
        self,
        request: Request,
        now: float,
    ) -> tuple[bool, int, float, bool]:
        if (
            not self.virtual_context_decode_hotset_first
            or self._debug_has_prefill_remaining(request)
        ):
            return True, 0, 0.0, False

        required_pages = self._virtual_context_reload_required_page_count(
            [request.request_id]
        )
        if required_pages <= 0:
            return True, 0, 0.0, False

        if self._virtual_context_decode_after_prefill_cold_streaming_allowed(
            request
        ):
            return (
                True,
                required_pages,
                self._virtual_context_decode_cold_streaming_slack_ms(now),
                True,
            )

        allowed, slack_ms = self._virtual_context_decode_cold_streaming_allowed(
            now
        )
        return allowed, required_pages, slack_ms, False

    def _native_reload_destinations_compatible(
        self, request_id: str, logical_pages_by_group: dict[int, set[int]],
    ) -> bool:
        """Validate writable destinations without treating allocation as data.

        Retry may reuse a concrete destination, but may not overwrite another
        request's shared page. All null slots must belong to this exact reload
        plan; checking just a count can hide a hole at a different position.
        """
        groups = self.kv_cache_manager.get_blocks(request_id).blocks
        null_slots: set[tuple[int, int]] = set()
        physical_ids: set[int] = set()
        for group_id, blocks in enumerate(groups):
            for page, block in enumerate(blocks):
                if block.is_null:
                    null_slots.add((group_id, page))
                elif block.block_id in physical_ids:
                    return False
                else:
                    physical_ids.add(block.block_id)
        planned_null_slots: set[tuple[int, int]] = set()
        for group_id, pages in logical_pages_by_group.items():
            if group_id < 0 or group_id >= len(groups):
                return False
            for page in pages:
                if page < 0 or page >= len(groups[group_id]):
                    return False
                block = groups[group_id][page]
                if block.is_null:
                    planned_null_slots.add((group_id, page))
                elif block.ref_cnt != 1:
                    return False
        return bool(logical_pages_by_group) and null_slots == planned_null_slots

    def _prepare_virtual_context_native_decode_handoff(
        self,
        request: Request,
    ) -> tuple[bool, int, int, int, str]:
        """Prepare a DRAM-resident context for native HBM decode.

        The context-ring path is intentionally prefill-first. Once a request has
        finished prefill, decode should use the normal HBM KV-cache attention
        path whenever capacity allows it. This method installs concrete HBM
        block ids for any DRAM-authoritative logical pages before the worker sees
        the decode step. The worker/provider then copies DRAM K/V into those
        blocks and marks the reload complete inside the attention backend.
        """
        if not self.virtual_context_reload_to_hbm:
            return False, 0, 0, 0, "reload_to_hbm_disabled"
        if self._debug_has_prefill_remaining(request):
            return False, 0, 0, 0, "prefill_remaining"

        (
            reload_page_count,
            logical_pages_by_group,
            required_blocks,
            min_free_blocks,
            progress_reserve_applied,
        ) = self._virtual_context_native_decode_handoff_requirement(request)
        if reload_page_count > 0:
            request.kv_retention_decode_continuation_lease_active = True
            request.kv_retention_decode_continuation_waiting = True
        if reload_page_count <= 0:
            if request.virtual_context_deferred_state == "decode_hbm_reload_pending":
                # The worker has removed every page from the reload plan, but
                # the matching ModelRunnerOutput may still be queued. Do not
                # turn an empty plan into a scheduler-side completion: that
                # would admit the same request again and skip the only ACK
                # which commits the planned reload token count.
                request.kv_retention_decode_continuation_waiting = True
                return False, 0, 0, 0, "reload_completion_unacknowledged"
            request.kv_retention_decode_continuation_waiting = False
            return True, 0, 0, 0, "no_reload_required"

        if not self.kv_cache_manager.has_request_block_table(request.request_id):
            return (
                False,
                reload_page_count,
                required_blocks,
                min_free_blocks,
                "reload_allocator_state_missing",
            )
        # Whole-request victims rebuild an ordinary virtual-external block
        # table on resume and do not carry the experimental partial-residency
        # marker. Validate the concrete/null table without requiring that
        # marker before replacing every DRAM-authoritative slot for Decode.
        _resident_blocks, virtual_blocks = (
            self.kv_cache_manager.get_retained_residency_counts(
                request.request_id, request.num_computed_tokens
            )
        )
        # A declined earlier admission may already own some/all destinations.
        # The idempotent allocator validates page indices; the post-allocation
        # no-null check below still proves full physical coverage, not copy ACK.
        if virtual_blocks > required_blocks:
            return (
                False,
                reload_page_count,
                required_blocks,
                min_free_blocks,
                "reload_allocator_plan_mismatch",
            )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if free_blocks < min_free_blocks:
            request.kv_retention_decode_continuation_waiting = True
            request.kv_retention_decode_continuation_wait_count += 1
            request.kv_retention_decode_continuation_last_wait_iteration_id = (
                self._iteration_debug_index
            )
            return (
                False,
                reload_page_count,
                required_blocks,
                min_free_blocks,
                (
                    "insufficient_hbm_blocks_with_ring_progress_reserve"
                    if progress_reserve_applied
                    else "insufficient_hbm_blocks"
                ),
            )

        if not self._native_reload_destinations_compatible(
            request.request_id, logical_pages_by_group,
        ):
            return (False, reload_page_count, required_blocks, min_free_blocks,
                    "reload_allocator_plan_mismatch")
        hbm_blocks = self.kv_cache_manager.allocate_virtual_context_hbm_reload_blocks(
            request.request_id,
            logical_pages_by_group,
        )
        if hbm_blocks is None:
            return (
                False,
                reload_page_count,
                required_blocks,
                min_free_blocks,
                "block_table_reload_allocation_failed",
            )

        _resident_blocks, remaining_virtual_blocks = (
            self.kv_cache_manager.get_retained_residency_counts(
                request.request_id,
                request.num_computed_tokens,
            )
        )
        if remaining_virtual_blocks > 0:
            raise RuntimeError(
                "native Decode admitted with a mixed HBM/null block table: "
                f"request={request.request_id} "
                f"remaining_virtual_blocks={remaining_virtual_blocks}"
            )

        request.kv_retention_decode_continuation_waiting = False
        # Allocation prepares destinations, not an admitted copy Future.
        return (
            True,
            reload_page_count,
            required_blocks,
            min_free_blocks,
            (
                "scheduled_with_ring_progress_reserve"
                if progress_reserve_applied
                else "scheduled"
            ),
        )

    def _prepare_virtual_context_native_prefill_promotion(
        self,
        request: Request,
        *,
        ring_token_cap: int,
    ) -> tuple[bool, int, int, str]:
        """Promote a whole provider-backed Prefill at a chunk boundary.

        Promotion is all-or-nothing: no Decode work may be active, every
        DRAM-authoritative prefix page receives a concrete HBM block, and the
        free pool must also cover all remaining prompt growth plus configured
        reserves. If any condition changes, the request simply remains on the
        Context Ring path; its current progress is never discarded.
        """

        if (
            not self.virtual_context_reload_to_hbm
            or not self._debug_has_prefill_remaining(request)
            or self._debug_running_decode_resident_blocks() > 0
            or self._pd_pending_decode_active()
            or self._virtual_context_waiting_decode_service_candidate() is not None
        ):
            return False, 0, 0, "decode_active_or_not_prefill"
        reload_page_count = (
            self.kv_cache_manager.get_virtual_context_hbm_reload_required_page_count(
                {request.request_id}
            )
        )
        if reload_page_count <= 0:
            return False, 0, 0, "no_reload_required"
        claim = self._scheduled_prefill_fit_capacity_claim(
            request,
            ring_token_cap=ring_token_cap,
            require_full_prompt=True,
        )
        if claim.reclaim_blocks > 0:
            return (
                False,
                reload_page_count,
                claim.required_blocks_with_reserve,
                "full_prefill_capacity_unavailable",
            )
        # A blocked promotion needs only counts, not a sorted per-layer copy
        # plan. Re-read the actual plan after the allocator capacity proof.
        reload_plan = self.kv_cache_manager.plan_virtual_context_hbm_reload_pages(
            request_ids={request.request_id}
        )
        if not reload_plan:
            return False, 0, 0, "no_reload_required"
        logical_tokens = virtual_context_residency_victim_context_tokens(
            request.num_computed_tokens,
            request.virtual_context_deferred_logical_computed_tokens,
        )
        _resident_blocks, virtual_blocks = (
            self.kv_cache_manager.get_retained_residency_counts(
                request.request_id,
                logical_tokens,
            )
        )
        logical_pages_by_group: dict[int, set[int]] = defaultdict(set)
        for page in reload_plan:
            logical_pages_by_group[page.kv_cache_group_id].add(
                page.logical_token_page
            )
        required_blocks = sum(
            len(pages) for pages in logical_pages_by_group.values()
        )
        if virtual_blocks > required_blocks or not self._native_reload_destinations_compatible(
            request.request_id, logical_pages_by_group,
        ):
            return (
                False,
                len(reload_plan),
                required_blocks,
                "reload_allocator_plan_mismatch",
            )
        hbm_blocks = self.kv_cache_manager.allocate_virtual_context_hbm_reload_blocks(
            request.request_id,
            logical_pages_by_group,
        )
        if hbm_blocks is None:
            return (
                False,
                len(reload_plan),
                required_blocks,
                "block_table_reload_allocation_failed",
            )
        _resident_blocks, remaining_virtual_blocks = (
            self.kv_cache_manager.get_retained_residency_counts(
                request.request_id,
                logical_tokens,
            )
        )
        if remaining_virtual_blocks > 0:
            raise RuntimeError(
                "native Prefill promotion left a mixed HBM/null table: "
                f"request={request.request_id} "
                f"remaining_virtual_blocks={remaining_virtual_blocks}"
            )
        # Already-concrete destinations from a declined admission are reusable.
        # Do not publish pending until positive-token admission is final.
        return True, len(reload_plan), required_blocks, "scheduled"

    def _check_virtual_context_prefill_ring_bandwidth_admission(
        self,
        request: Request,
        num_new_tokens: int,
    ) -> tuple[bool, int, int, int]:
        if (
            not self.virtual_context_decode_hotset_first
            or num_new_tokens <= 0
            or not self._debug_has_prefill_remaining(request)
            or self._virtual_context_waiting_decode_ready(request)
        ):
            return True, num_new_tokens, 0, 0

        decode_reload_pages = (
            self._virtual_context_running_decode_reload_required_page_count()
        )
        if decode_reload_pages <= 0:
            return True, num_new_tokens, 0, 0

        leftover_tokens = self.virtual_context_prefill_ring_tokens_with_decode_cold
        if leftover_tokens <= 0:
            return False, 0, decode_reload_pages, leftover_tokens
        return (
            True,
            min(num_new_tokens, leftover_tokens),
            decode_reload_pages,
            leftover_tokens,
        )

    def _debug_record_pd_residency_prefill_skip(
        self,
        request: Request,
        timestamp: float,
        free_blocks: int,
        decode_resident_blocks: int,
    ) -> int:
        scheduled_prefill_fit = (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        )
        if scheduled_prefill_fit:
            claim = self._scheduled_prefill_fit_capacity_claim(
                request,
                ring_token_cap=(
                    self._dynamic_virtual_context_ring_prefill_admission_tokens()
                    or self._virtual_context_residency_fixed_ring_chunk_tokens()
                ),
            )
            required_blocks = claim.required_blocks_with_reserve
            serialized_by_kv_reserve = claim.reclaim_blocks > 0
        else:
            required_blocks = self._debug_estimate_next_prefill_required_blocks(
                request
            )
            serialized_by_kv_reserve = required_blocks > free_blocks
        request.pd_residency_prefill_skip_count += 1
        if request.pd_residency_prefill_skip_first_iteration_id < 0:
            request.pd_residency_prefill_skip_first_iteration_id = (
                self._iteration_debug_index
            )
        request.pd_residency_prefill_skip_last_iteration_id = (
            self._iteration_debug_index
        )
        if request._pd_residency_prefill_skip_last_ts >= 0:
            request.pd_residency_prefill_unscheduled_duration_s += max(
                0.0,
                timestamp - request._pd_residency_prefill_skip_last_ts,
            )
        request._pd_residency_prefill_skip_last_ts = timestamp
        request.pd_residency_last_prefill_required_blocks = required_blocks
        request.pd_residency_last_free_blocks = free_blocks
        request.pd_residency_last_decode_resident_blocks = decode_resident_blocks
        request.pd_residency_last_serialized_by_kv_reserve = int(
            serialized_by_kv_reserve
        )
        return required_blocks

    @staticmethod
    def _debug_close_pd_residency_prefill_skip(
        request: Request, timestamp: float
    ) -> None:
        if request._pd_residency_prefill_skip_last_ts < 0:
            return
        request.pd_residency_prefill_unscheduled_duration_s += max(
            0.0, timestamp - request._pd_residency_prefill_skip_last_ts
        )
        request._pd_residency_prefill_skip_last_ts = -1.0

    def schedule(self) -> SchedulerOutput:
        # A required snapshot can finish after the last model Future. Its
        # error must remain observable after provider page fences retire.
        if getattr(self, "isolation_required_d2h_progress", False):
            raise_if_required_d2h_failed()
        if getattr(self, "isolation_prefill_batch_kv_budget", False):
            self._isolation_prefill_batch_budget_debug: dict[str, int] = {}
        isolation_waiting_decode_reclaim_wait_ids = frozenset()
        isolation_waiting_decode_released: list[ReleasedPrefillVictim] = []
        if getattr(self, "isolation_waiting_decode_reclaim", False):
            (
                isolation_waiting_decode_reclaim_wait_ids,
                isolation_waiting_decode_released,
            ) = self._progress_isolation_waiting_decode_reclaim()
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        self.kv_cache_manager.reset_virtual_context_schedule_timings()
        schedule_running_scan_duration_s = 0.0
        schedule_native_handoff_duration_s = 0.0
        schedule_decode_cold_guard_duration_s = 0.0
        schedule_allocate_slots_duration_s = 0.0
        native_decode_runtime_fast_path_count = 0
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_residency_rehydrated_req_ids: set[str] = set()
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        waiting_queue_len_before = len(self.waiting)
        running_reqs_before = len(self.running)
        running_phase_counts_before = self._debug_running_phase_counts()
        num_waiting_prefill_reqs_before = self._debug_count_waiting_prefill_reqs()
        prefill_defer_reason_counts: dict[str, int] = defaultdict(int)
        deferred_request_ids: set[str] = set()
        waiting_break_defer_reason: str | None = None
        pd_decoupled_selected_phase = "mixed"
        pd_phase_separated_last_phase_before = (
            self._pd_phase_separated_last_phase
        )
        pd_decoupled_skipped_running_prefill_reqs = 0
        pd_decoupled_skipped_running_decode_reqs = 0
        ttft_prefill_policy_skipped_running_reqs = 0
        ttft_prefill_policy_skipped_waiting_reqs = 0
        ttft_prefill_policy_candidate_order: tuple[str, ...] = ()
        ttft_prefill_policy_selected_request_ids: tuple[str, ...] = ()
        ttft_prefill_policy_selected_request_id_set: frozenset[str] = (
            frozenset()
        )
        ttft_prefill_frontier = PrefillAdmissionFrontier()
        ttft_prefill_policy_scheduled_request_ids: list[str] = []
        ttft_prefill_policy_scheduled_tokens = 0
        tbt_decode_policy_candidate_order: tuple[str, ...] = ()
        tbt_decode_policy_scheduled_request_ids: list[str] = []
        tbt_decode_policy_scheduled_tokens = 0
        tbt_decode_policy_active_this_iteration = False
        pd_separate_scheduler_skipped_pending_reqs = 0
        pd_residency_skipped_running_prefill_request_ids: set[str] = set()
        pd_residency_prefill_required_blocks_by_req: dict[str, int] = {}
        pd_residency_serialized_request_ids: set[str] = set()
        pd_residency_last_free_blocks = 0
        pd_residency_last_decode_resident_blocks = 0
        running_kv_capacity_miss_count = 0
        running_prefill_kv_capacity_miss_count = 0
        running_decode_kv_capacity_miss_count = 0
        running_kv_capacity_miss_request_ids: set[str] = set()
        running_prefill_kv_wait_count = 0
        running_prefill_kv_wait_request_ids: set[str] = set()
        running_prefill_kv_waited = False
        virtual_context_deferred_req_ids: set[str] = set(
            getattr(self, "_isolation_waiting_decode_retirements", ())
        )
        virtual_context_defer_fallback_reasons: set[str] = set()
        prefill_deferred_by_residency_budget = 0
        prefill_defer_residency_fallback_count = 0
        preempt_reset_avoided_tokens = 0
        prefill_dram_resident_tokens = 0
        prefill_hbm_reloaded_tokens = 0
        prefill_resume_without_recompute = 0
        prefill_deferred_by_residency_admission = 0
        residency_admission_deferred_request_ids: set[str] = set()
        residency_admission_required_blocks_sum = 0
        residency_admission_required_blocks_max = 0
        residency_admission_raw_required_blocks_max = 0
        ring_residency_admission_applied_count = 0
        ring_residency_decode_reserve_relaxed_count = 0
        ring_prefill_admission_cap_hit_count = 0
        ring_prefill_admission_cap_running_hit_count = 0
        ring_prefill_admission_cap_waiting_hit_count = 0
        ring_prefill_admission_original_tokens_sum = 0
        ring_prefill_admission_capped_tokens_sum = 0
        ring_prefill_admission_min_progress_defer_count = 0
        ring_prefill_phase_budget_original = 0
        ring_prefill_phase_budget_capped = 0
        ring_prefill_phase_budget_applied = 0
        virtual_context_resume_guard_count = 0
        virtual_context_resume_guard_request_ids: set[str] = set()
        virtual_context_decode_cold_streaming_blocked_count = 0
        virtual_context_decode_cold_streaming_allowed_count = 0
        virtual_context_decode_after_prefill_streaming_allowed_count = 0
        virtual_context_decode_cold_streaming_required_pages = 0
        virtual_context_decode_cold_streaming_last_slack_ms = 0.0
        virtual_context_prefill_ring_bandwidth_deferred_count = 0
        virtual_context_prefill_ring_bandwidth_decode_reload_pages = 0
        virtual_context_prefill_ring_bandwidth_leftover_tokens = (
            self.virtual_context_prefill_ring_tokens_with_decode_cold
        )
        virtual_context_prefill_ring_with_decode_cold_allowed_count = 0
        virtual_context_prefill_ring_with_decode_cold_allowed_tokens = 0
        virtual_context_active_decode_ring_prefill_allowed_count = 0
        virtual_context_active_decode_native_prefill_blocked_count = 0
        virtual_context_active_decode_ring_prefill_reload_pages = 0
        virtual_context_active_decode_ring_prefill_force_all_count = 0
        pd_native_active_decode_prefill_capacity_check_count = 0
        pd_native_active_decode_prefill_capacity_defer_count = 0
        pd_native_active_decode_prefill_required_blocks = 0
        pd_native_prefill_retry_min_blocks = 0
        pd_native_active_decode_prefill_chunk_required_blocks = 0
        pd_native_active_decode_growth_reserve_blocks = 0
        pd_native_active_decode_prefill_free_blocks = 0
        pd_native_active_decode_prefill_capacity_reason = "not_evaluated"
        pd_native_active_decode_prefill_capacity_blocked = False
        virtual_context_residency_pressure_release_attempt_count = 0
        virtual_context_residency_pressure_release_success_count = 0
        virtual_context_residency_pressure_release_failed_count = 0
        virtual_context_residency_pressure_release_tokens = 0
        virtual_context_residency_pressure_release_reasons: set[str] = set()
        virtual_context_pressure_snapshot_request_ids: set[str] = set()
        virtual_context_capacity_floor_snapshot_request_ids: set[str] = set()
        virtual_context_capacity_floor_release_attempt_count = 0
        virtual_context_capacity_floor_release_success_count = 0
        virtual_context_capacity_floor_release_failed_count = 0
        virtual_context_capacity_floor_victim_candidate_ids: list[str] = []
        virtual_context_capacity_floor_victim_candidate_ring_chunks: list[
            str
        ] = []
        virtual_context_capacity_floor_victim_candidate_reclaim_blocks: list[
            str
        ] = []
        virtual_context_capacity_floor_victim_candidate_copy_tokens: list[
            str
        ] = []
        virtual_context_capacity_floor_victim_candidate_copy_scores: list[
            str
        ] = []
        virtual_context_capacity_floor_victim_jit_triggered = 0
        virtual_context_capacity_floor_victim_jit_safe_candidate_count = 0
        virtual_context_capacity_floor_victim_required_reclaim_blocks = 0
        virtual_context_capacity_floor_victim_ring_chunk_tokens = 0
        virtual_context_capacity_floor_victim_selected_request_id = ""
        virtual_context_capacity_floor_victim_selected_context_tokens = 0
        virtual_context_capacity_floor_victim_selected_remaining_tokens = 0
        virtual_context_capacity_floor_victim_selected_ring_chunks = 0
        virtual_context_capacity_floor_victim_selected_reclaim_blocks = 0
        virtual_context_capacity_floor_victim_selected_ring_work = 0
        virtual_context_capacity_floor_victim_selected_d2h_eviction_tokens = 0
        virtual_context_capacity_floor_victim_selected_ring_staging_h2d_tokens = 0
        virtual_context_capacity_floor_victim_selected_decode_reload_h2d_tokens = 0
        virtual_context_capacity_floor_victim_selected_total_copy_tokens = 0
        virtual_context_capacity_floor_victim_selected_request_ids: list[
            str
        ] = []
        virtual_context_capacity_floor_victim_selected_reclaim_blocks_total = 0
        scheduled_prefill_fit_triggered = 0
        scheduled_prefill_fit_target_request_id = ""
        scheduled_prefill_fit_target_logical_tokens = 0
        scheduled_prefill_fit_target_chunk_tokens = 0
        scheduled_prefill_fit_target_raw_required_blocks = 0
        scheduled_prefill_fit_target_hard_reclaim_blocks = 0
        scheduled_prefill_fit_target_waived_headroom_blocks = 0
        scheduled_prefill_fit_target_reserved_blocks = 0
        scheduled_prefill_fit_target_free_blocks = 0
        scheduled_prefill_fit_target_reclaim_blocks = 0
        scheduled_prefill_fit_active_decode_hard_headroom = False
        scheduled_prefill_fit_hard_headroom_blocks = 0
        scheduled_prefill_fit_soft_headroom_blocks = (
            self.virtual_context_scheduled_prefill_fit_headroom_blocks
        )
        scheduled_prefill_fit_soft_headroom_shortfall_blocks = 0
        scheduled_prefill_fit_commit_reclaim_blocks = 0
        scheduled_prefill_fit_safe_candidate_count = 0
        scheduled_prefill_fit_partial_release_blocks = 0
        scheduled_prefill_fit_whole_request_victim_count = 0
        scheduled_prefill_fit_whole_request_reclaimed_blocks = 0
        scheduled_prefill_fit_proactive_target_request_id = ""
        scheduled_prefill_fit_proactive_projected_logical_tokens = 0
        scheduled_prefill_fit_proactive_next_chunk_tokens = 0
        scheduled_prefill_fit_proactive_raw_required_blocks = 0
        scheduled_prefill_fit_proactive_reserved_blocks = 0
        scheduled_prefill_fit_proactive_free_blocks = 0
        scheduled_prefill_fit_proactive_deficit_blocks = 0
        scheduled_prefill_fit_proactive_plan_mode = "none"
        scheduled_prefill_fit_proactive_native_deficit_blocks = 0
        scheduled_prefill_fit_proactive_ring_fallback_deficit_blocks = 0
        scheduled_prefill_fit_proactive_avoided_peer_blocks = 0
        scheduled_prefill_fit_proactive_snapshot_pages: set[
            tuple[str, int, int]
        ] = set()
        scheduled_prefill_fit_proactive_snapshot_logical_tokens: dict[
            str, int
        ] = {}
        scheduled_prefill_fit_proactive_victim_request_id = ""
        scheduled_prefill_fit_proactive_victim_request_ids = ""
        scheduled_prefill_fit_proactive_external_victim_count = 0
        scheduled_prefill_fit_proactive_waiting_victim_count = 0
        scheduled_prefill_fit_proactive_decode_handoff = 0
        scheduled_prefill_fit_preparation_target_request_id = ""
        scheduled_prefill_fit_preparation_victim_request_id = ""
        scheduled_prefill_fit_preparation_victim_request_ids = ""
        scheduled_prefill_fit_preparation_planned_page_count = 0
        scheduled_prefill_fit_preparation_ready_page_count = 0
        scheduled_prefill_fit_preparation_waiting = False
        scheduled_prefill_fit_preparation_ready = False
        scheduled_prefill_fit_decode_growth_runway_active = False
        scheduled_prefill_fit_decode_growth_runway_target_request_id = ""
        scheduled_prefill_fit_decode_growth_runway_absorbed_blocks = 0
        scheduled_prefill_fit_decode_growth_runway_consumed = False
        scheduled_prefill_fit_allocator_resume_count = 0
        scheduled_prefill_fit_allocator_resume_resident_blocks = 0
        scheduled_prefill_fit_allocator_resume_virtual_blocks = 0
        scheduled_prefill_fit_allocator_resume_request_ids: set[str] = set()
        fit_preparation: ScheduledPrefillFitPreparation | None = None
        virtual_context_hbm_relief_recompute_count = 0
        virtual_context_hbm_relief_recompute_tokens = 0
        virtual_context_hbm_relief_recompute_reload_pages = 0
        virtual_context_hbm_relief_recompute_min_free_blocks = (
            self._virtual_context_hbm_relief_min_free_blocks()
        )
        virtual_context_hbm_relief_recompute_free_blocks_max = 0
        virtual_context_hbm_relief_recompute_request_ids: set[str] = set()
        virtual_context_reload_to_hbm_candidate_count = 0
        virtual_context_reload_to_hbm_scheduled_count = 0
        virtual_context_reload_to_hbm_tokens = 0
        virtual_context_reload_to_hbm_pages = 0
        virtual_context_reload_to_hbm_required_blocks = 0
        virtual_context_reload_to_hbm_min_free_blocks = 0
        virtual_context_reload_to_hbm_free_blocks_max = 0
        virtual_context_reload_to_hbm_request_ids: set[str] = set()
        virtual_context_native_prefill_promotion_request_ids: set[str] = set()
        virtual_context_native_prefill_promotion_pages = 0
        virtual_context_native_prefill_promotion_required_blocks = 0
        virtual_context_native_prefill_promotion_reasons: set[str] = set()
        virtual_context_decode_handoff_reload_candidate_count = 0
        virtual_context_decode_handoff_reload_scheduled_count = 0
        virtual_context_decode_handoff_reload_blocked_count = 0
        virtual_context_decode_handoff_reload_pages = 0
        virtual_context_decode_handoff_reload_required_blocks = 0
        virtual_context_decode_handoff_reload_min_free_blocks = 0
        # Minimum capacity at which at least one currently blocked handoff can
        # be reconsidered.  The public debug metric above remains the maximum
        # claim, but an Engine retry gate must not wait for the largest waiter
        # when a smaller request could become schedulable first.
        virtual_context_decode_handoff_retry_min_free_blocks = 0
        virtual_context_decode_handoff_reload_reasons: set[str] = set()
        virtual_context_decode_handoff_reload_request_ids: set[str] = set()
        virtual_context_decode_handoff_reload_full_block_table_req_ids: set[str] = set()
        tiered_explicit_execution_plans: dict[str, tuple[str, int]] = {}
        tiered_native_reload_planned_tokens: dict[str, int] = {}
        virtual_context_decode_handoff_reclaim_target_blocks = 0
        virtual_context_decode_handoff_reclaim_shortfall_blocks = 0
        virtual_context_decode_handoff_reclaim_attempt_count = 0
        virtual_context_decode_handoff_reclaim_success_count = 0
        virtual_context_decode_handoff_reclaim_failed_count = 0
        virtual_context_decode_handoff_reclaimed_blocks = 0
        virtual_context_decode_handoff_reclaim_victim_ids: list[str] = []
        virtual_context_decode_handoff_reclaim_reasons: set[str] = set()
        virtual_context_decode_handoff_reclaim_final_free_blocks = 0
        virtual_context_decode_handoff_reclaim_retry_success_count = 0
        # Physical release also retires worker cached rows through the normal
        # SchedulerOutput path. Do not leave this only in diagnostic counters.
        for released_victim in isolation_waiting_decode_released:
            virtual_context_deferred_req_ids.add(released_victim.request_id)
            prefill_deferred_by_residency_budget += 1
            preempt_reset_avoided_tokens += released_victim.deferred_tokens
            prefill_dram_resident_tokens += released_victim.deferred_tokens
            virtual_context_residency_pressure_release_success_count += 1
            virtual_context_residency_pressure_release_tokens += released_victim.deferred_tokens
            scheduled_prefill_fit_whole_request_victim_count += 1
            scheduled_prefill_fit_whole_request_reclaimed_blocks += released_victim.reclaimed_blocks
            virtual_context_decode_handoff_reclaim_success_count += 1
            virtual_context_decode_handoff_reclaimed_blocks += released_victim.reclaimed_blocks
            virtual_context_decode_handoff_reclaim_victim_ids.append(released_victim.request_id)
        self._virtual_context_ring_pressure_last_free_blocks = 0
        self._virtual_context_ring_pressure_last_threshold_blocks = 0
        self._virtual_context_ring_pressure_last_bootstrap_blocks = 0
        self._virtual_context_ring_pressure_last_ring_blocks = 0
        self._virtual_context_ring_pressure_last_reserved_blocks = 0
        self._virtual_context_ring_pressure_last_native_safe = 0
        self._virtual_context_ring_pressure_last_reason = "not_evaluated"
        virtual_context_ring_prefill_admission_tokens_effective = (
            self._dynamic_virtual_context_ring_prefill_admission_tokens()
        )
        jit_min_copy_capacity_floor_active = (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.JIT_MIN_COPY
        )
        jit_min_copy_ring_chunk_tokens = 0
        jit_min_copy_free_blocks = 0
        jit_min_copy_required_blocks = 0
        if jit_min_copy_capacity_floor_active:
            jit_min_copy_ring_chunk_tokens = (
                self._virtual_context_residency_fixed_ring_chunk_tokens()
            )
            jit_min_copy_free_blocks = (
                self.kv_cache_manager.block_pool.get_num_free_blocks()
            )
            jit_min_copy_required_blocks = required_reclaim_blocks(
                jit_min_copy_free_blocks,
                self._virtual_context_residency_reserved_blocks(),
                jit_min_copy_ring_chunk_tokens,
                self.block_size,
            )
        if self.pd_decoupled_no_mix:
            pd_decoupled_selected_phase = (
                self._select_pd_decoupled_no_mix_phase()
            )
        isolation_phase_request_limit = self._isolation_phase_request_limit(
            pd_decoupled_selected_phase)
        isolation_phase_request_limit_hit = False
        isolation_phase_waiting_skip_count = 0
        isolation_phase_waiting_skipped_ids: list[str] = []
        isolation_waiting_decode_prefill_fcfs_bypass_count = 0
        isolation_waiting_decode_prefill_fcfs_first_id = ""
        isolation_waiting_decode_resume_owner_block_count = 0
        scheduled_prefill_fit_active = (
            self.virtual_context_residency_victim_policy
            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
        )
        if scheduled_prefill_fit_active:
            (
                scheduled_prefill_fit_hard_headroom_blocks,
                scheduled_prefill_fit_soft_headroom_blocks,
            ) = self._scheduled_prefill_fit_headroom_partition()
            scheduled_prefill_fit_active_decode_hard_headroom = bool(
                scheduled_prefill_fit_hard_headroom_blocks > 0
            )
        existing_fit_preparation = self._scheduled_prefill_fit_preparation
        existing_fit_target = (
            self.requests.get(existing_fit_preparation.target_request_id)
            if existing_fit_preparation is not None
            else None
        )
        scheduled_prefill_fit_decode_handoff_prepared = bool(
            existing_fit_target is not None
            and (
                not self._debug_has_prefill_remaining(existing_fit_target)
                or (
                    getattr(self, "isolation_waiting_decode_reclaim", False)
                    and self._virtual_context_waiting_decode_ready(existing_fit_target)
                )
            )
        )
        if scheduled_prefill_fit_active and (
            not self.pd_decoupled_no_mix
            or pd_decoupled_selected_phase == "prefill"
        ):
            if self.enable_ttft_prefill_chunk_scheduling:
                ttft_prefill_policy_candidate_order = (
                    self._ttft_begin_prefill_chunk_iteration()
                )
            early_preferred_request_id = (
                self.ttft_prefill_chunk_scheduler.preferred_request_id
                if self.enable_ttft_prefill_chunk_scheduling
                else ""
            )
            scheduled_prefill_fit_ring_token_cap = (
                virtual_context_ring_prefill_admission_tokens_effective
                or self._virtual_context_residency_fixed_ring_chunk_tokens()
            )
            (
                scheduled_prefill_fit_target,
                scheduled_prefill_fit_target_chunk_tokens,
                scheduled_prefill_fit_target_raw_required_blocks,
                scheduled_prefill_fit_target_reserved_blocks,
                scheduled_prefill_fit_target_free_blocks,
                scheduled_prefill_fit_target_reclaim_blocks,
            ) = self._scheduled_prefill_fit_target(
                preferred_request_id=early_preferred_request_id,
                ring_token_cap=scheduled_prefill_fit_ring_token_cap,
            )
            scheduled_prefill_fit_target_claim: (
                ScheduledPrefillCapacityClaim | None
            ) = None
            if scheduled_prefill_fit_target is not None:
                fit_target = scheduled_prefill_fit_target
                scheduled_prefill_fit_target_claim = (
                    self._scheduled_prefill_fit_capacity_claim(
                        fit_target,
                        ring_token_cap=scheduled_prefill_fit_ring_token_cap,
                        full_native=(
                            self._ttft_prefill_execution_mode(fit_target)
                            == "native"
                            if self.ttft_prefill_native_chunk_max_tokens > 0
                            else self._virtual_context_reload_required_page_count(
                                [fit_target.request_id],
                                "prefill",
                            )
                            <= 0
                        ),
                    )
                )
                scheduled_prefill_fit_target_request_id = (
                    fit_target.request_id
                )
                scheduled_prefill_fit_target_logical_tokens = (
                    scheduled_prefill_fit_target_claim.logical_tokens
                )
                scheduled_prefill_fit_target_hard_reclaim_blocks = (
                    scheduled_prefill_fit_target_claim.hard_reclaim_blocks
                )
                scheduled_prefill_fit_target_waived_headroom_blocks = (
                    scheduled_prefill_fit_target_claim.waived_capacity_headroom_blocks
                )
            decode_growth_runway = (
                self._scheduled_prefill_fit_decode_growth_runway
            )
            if decode_growth_runway is not None:
                runway_boundary_matches = bool(
                    scheduled_prefill_fit_target_claim is not None
                    and scheduled_prefill_fit_target_claim.request_id
                    == decode_growth_runway.target_request_id
                    and scheduled_prefill_fit_target_claim.logical_tokens
                    == decode_growth_runway.target_logical_tokens
                    and scheduled_prefill_fit_target_claim.chunk_tokens
                    == decode_growth_runway.target_chunk_tokens
                    and decode_growth_runway.capacity_headroom_blocks
                    == scheduled_prefill_fit_target_claim.capacity_headroom_blocks
                )
                runway_hard_exhausted = bool(
                    scheduled_prefill_fit_target_claim is not None
                    and (
                        scheduled_prefill_fit_target_claim.hard_reclaim_blocks
                        > 0
                        or scheduled_prefill_fit_target_claim.reclaim_blocks
                        > decode_growth_runway.capacity_headroom_blocks
                    )
                )
                if not runway_boundary_matches or runway_hard_exhausted:
                    self._scheduled_prefill_fit_decode_growth_runway = None
                    (
                        self._scheduled_prefill_fit_decode_growth_runway_invalidated_count
                    ) += 1
                else:
                    scheduled_prefill_fit_decode_growth_runway_active = True
                    scheduled_prefill_fit_decode_growth_runway_target_request_id = (
                        decode_growth_runway.target_request_id
                    )
                    if decode_growth_runway.covers(
                        scheduled_prefill_fit_target_claim
                    ):
                        scheduled_prefill_fit_decode_growth_runway_absorbed_blocks = (
                            scheduled_prefill_fit_target_claim.reclaim_blocks
                        )
            if (
                scheduled_prefill_fit_target_request_id
                != self._scheduled_prefill_fit_last_target_request_id
            ):
                self._scheduled_prefill_fit_target_change_count += 1
                self._scheduled_prefill_fit_last_target_request_id = (
                    scheduled_prefill_fit_target_request_id
                )
            scheduled_prefill_fit_triggered = int(
                scheduled_prefill_fit_target_reclaim_blocks > 0
            )
            if (
                scheduled_prefill_fit_target_reclaim_blocks > 0
                and scheduled_prefill_fit_decode_growth_runway_absorbed_blocks
                <= 0
                and not scheduled_prefill_fit_decode_handoff_prepared
            ):
                preparation_required_blocks = (
                    scheduled_prefill_fit_target_reclaim_blocks
                )
                preparation_hard_required_blocks = (
                    scheduled_prefill_fit_target_hard_reclaim_blocks
                )
                pending_fit_preparation = (
                    self._scheduled_prefill_fit_preparation
                )
                if (
                    pending_fit_preparation is not None
                    and pending_fit_preparation.target_self_victim
                    and scheduled_prefill_fit_target is not None
                    and pending_fit_preparation.target_request_id
                    == scheduled_prefill_fit_target.request_id
                ):
                    (
                        _ring_required_blocks,
                        _ring_reserved_blocks,
                        preparation_required_blocks,
                        preparation_hard_required_blocks,
                    ) = self._scheduled_prefill_fit_ring_fallback_reclaim_requirement(
                        scheduled_prefill_fit_target,
                        next_chunk_tokens=(
                            pending_fit_preparation.next_chunk_tokens
                        ),
                    )
                (
                    fit_preparation,
                    scheduled_prefill_fit_preparation_ready_page_count,
                    scheduled_prefill_fit_preparation_planned_page_count,
                    scheduled_prefill_fit_preparation_ready,
                ) = self._scheduled_prefill_fit_preparation_status(
                    live_target_request_id=(
                        scheduled_prefill_fit_target_request_id
                    ),
                    required_blocks=preparation_required_blocks,
                    hard_required_blocks=preparation_hard_required_blocks,
                )
                if fit_preparation is not None:
                    scheduled_prefill_fit_preparation_target_request_id = (
                        fit_preparation.target_request_id
                    )
                    scheduled_prefill_fit_preparation_victim_request_id = (
                        fit_preparation.victim_request_id
                    )
                    scheduled_prefill_fit_preparation_victim_request_ids = (
                        ",".join(fit_preparation.victim_request_ids)
                    )
                    scheduled_prefill_fit_preparation_waiting = (
                        not scheduled_prefill_fit_preparation_ready
                    )
                    scheduled_prefill_fit_soft_headroom_shortfall_blocks = (
                        max(
                            0,
                            preparation_required_blocks
                            - scheduled_prefill_fit_preparation_planned_page_count,
                        )
                    )
                    if scheduled_prefill_fit_soft_headroom_shortfall_blocks > 0:
                        self._scheduled_prefill_fit_soft_headroom_shortfall_count += 1
                        self._scheduled_prefill_fit_soft_headroom_shortfall_blocks += (
                            scheduled_prefill_fit_soft_headroom_shortfall_blocks
                        )
                    if scheduled_prefill_fit_preparation_waiting:
                        self._scheduled_prefill_fit_preparation_wait_count += 1
                    else:
                        self._scheduled_prefill_fit_preparation_ready_count += 1
            elif (
                self._scheduled_prefill_fit_preparation is not None
                and not scheduled_prefill_fit_decode_handoff_prepared
            ):
                self._scheduled_prefill_fit_preparation_status(
                    live_target_request_id=(
                        scheduled_prefill_fit_target_request_id
                    ),
                    required_blocks=0,
                )
        capacity_floor_triggered = (
            scheduled_prefill_fit_target_reclaim_blocks > 0
            and not scheduled_prefill_fit_preparation_waiting
            and scheduled_prefill_fit_decode_growth_runway_absorbed_blocks <= 0
            if scheduled_prefill_fit_active
            else (
                virtual_context_ring_prefill_admission_tokens_effective <= 0
                or jit_min_copy_required_blocks > 0
            )
        )
        if scheduled_prefill_fit_decode_handoff_prepared:
            capacity_floor_triggered = False
        if (
            capacity_floor_triggered
            and (
                not self.pd_decoupled_no_mix
                or pd_decoupled_selected_phase == "prefill"
            )
            and self.virtual_context_ring_residency_admission
            and self.virtual_context_defer_prefill
        ):
            # No non-reserved KV block remains for another prefill chunk.
            # Retry every immutable active prefill rather than only the
            # request selected by FCFS/RR/EDF/SRPT. Victim ordering is a pure
            # proposal. Legacy mode picks the first copy-ready candidate.
            # Pending-victim mode preserves a top-ranked future owner as an
            # advisory intent and waits for recovery/revalidation.
            capacity_floor_candidates = [
                request
                for request in self._active_virtual_context_prefill_requests()
                if (
                    (
                        self.virtual_context_pending_victim_intent_enabled
                        or request.request_id
                        not in self._pd_pending_request_ids
                    )
                    and self._debug_has_prefill_remaining(request)
                )
            ]
            ring_chunk_tokens = (
                scheduled_prefill_fit_target_chunk_tokens
                if scheduled_prefill_fit_active
                else jit_min_copy_ring_chunk_tokens
                if jit_min_copy_capacity_floor_active
                else self._virtual_context_residency_fixed_ring_chunk_tokens()
            )
            free_blocks = (
                scheduled_prefill_fit_target_free_blocks
                if scheduled_prefill_fit_active
                else jit_min_copy_free_blocks
                if jit_min_copy_capacity_floor_active
                else self.kv_cache_manager.block_pool.get_num_free_blocks()
            )
            required_blocks = (
                scheduled_prefill_fit_target_reclaim_blocks
                if scheduled_prefill_fit_active
                else jit_min_copy_required_blocks
                if jit_min_copy_capacity_floor_active
                else required_reclaim_blocks(
                    free_blocks,
                    self._virtual_context_residency_reserved_blocks(),
                    ring_chunk_tokens,
                    self.block_size,
                )
            )
            if (
                scheduled_prefill_fit_active
                and scheduled_prefill_fit_preparation_ready
                and fit_preparation is not None
                and fit_preparation.target_self_victim
                and scheduled_prefill_fit_target is not None
            ):
                # Ring fallback has a different objective from native fit.
                # Revalidate from the live free-block count so Decode growth
                # may consume headroom, then release T first and only as many
                # prepared peers as the bounded Ring objective still needs.
                (
                    _ring_required_blocks,
                    _ring_reserved_blocks,
                    required_blocks,
                    _ring_hard_required_blocks,
                ) = self._scheduled_prefill_fit_ring_fallback_reclaim_requirement(
                    scheduled_prefill_fit_target,
                    next_chunk_tokens=fit_preparation.next_chunk_tokens,
                )
                free_blocks = (
                    self.kv_cache_manager.block_pool.get_num_free_blocks()
                )
                ring_chunk_tokens = fit_preparation.next_chunk_tokens
            commit_required_blocks = required_blocks
            if (
                scheduled_prefill_fit_active
                and scheduled_prefill_fit_preparation_ready
                and fit_preparation is not None
            ):
                # Commit only from the prepared whole-victim set. Decode may
                # have consumed part of the soft headroom while D2H completed;
                # the reclaim loop may overshoot this block target only by the
                # size of its final request-granular victim.
                commit_required_blocks = min(
                    required_blocks,
                    len(set(fit_preparation.snapshot_pages)),
                )
            if scheduled_prefill_fit_active:
                scheduled_prefill_fit_commit_reclaim_blocks = (
                    commit_required_blocks
                )
            virtual_context_capacity_floor_victim_required_reclaim_blocks = (
                required_blocks
            )
            virtual_context_capacity_floor_victim_ring_chunk_tokens = (
                ring_chunk_tokens
            )
            jit_min_copy_active = (
                self.virtual_context_residency_victim_policy
                == ResidencyVictimPolicy.JIT_MIN_COPY
            )
            virtual_context_capacity_floor_victim_jit_triggered = int(
                jit_min_copy_active and required_blocks > 0
            )
            if jit_min_copy_active and required_blocks <= 0:
                # ``jit`` is literal: do not even rank victims until the current
                # free-block snapshot is below reserve plus one prefill chunk.
                capacity_floor_candidates = []
            candidate_snapshots: dict[str, ResidencyVictimCandidate] = {}
            safe_capacity_floor_candidates: list[Request] = []
            for request in capacity_floor_candidates:
                snapshot = (
                    self._virtual_context_prefill_reclaim_candidate_snapshot(
                        request
                    )
                )
                if snapshot is None:
                    continue
                candidate_snapshots[request.request_id] = snapshot
                safe_capacity_floor_candidates.append(request)
            capacity_floor_candidates = safe_capacity_floor_candidates
            if jit_min_copy_active:
                virtual_context_capacity_floor_victim_jit_safe_candidate_count = len(
                    capacity_floor_candidates
                )
            if scheduled_prefill_fit_active:
                scheduled_prefill_fit_safe_candidate_count = len(
                    capacity_floor_candidates
                )
            if (
                self.virtual_context_residency_victim_policy
                == ResidencyVictimPolicy.LONGEST_LOGICAL_CONTEXT_FIRST
            ):
                # Preserve the previous default path and exact FCFS tie-break.
                capacity_floor_candidates.sort(
                    key=lambda request: virtual_context_residency_victim_sort_key(
                        candidate_snapshots[
                            request.request_id
                        ].logical_context_tokens,
                        request.arrival_time,
                        request.request_id,
                    )
                )
            else:
                ordered_snapshots = order_residency_victims(
                    list(candidate_snapshots.values()),
                    policy=self.virtual_context_residency_victim_policy,
                    required_blocks=required_blocks,
                    ring_chunk_tokens=ring_chunk_tokens,
                )
                requests_by_id = {
                    request.request_id: request
                    for request in capacity_floor_candidates
                }
                capacity_floor_candidates = [
                    requests_by_id[candidate.request_id]
                    for candidate in ordered_snapshots
                ]
            virtual_context_capacity_floor_victim_candidate_ids = [
                request.request_id for request in capacity_floor_candidates
            ]
            for request in capacity_floor_candidates:
                candidate = candidate_snapshots[request.request_id]
                ring_chunks = fixed_ring_chunk_count(
                    candidate.remaining_prefill_tokens,
                    ring_chunk_tokens,
                )
                virtual_context_capacity_floor_victim_candidate_ring_chunks.append(
                    f"{request.request_id}:{ring_chunks}"
                )
                virtual_context_capacity_floor_victim_candidate_reclaim_blocks.append(
                    f"{request.request_id}:"
                    f"{candidate.estimated_reclaimable_blocks}"
                )
                virtual_context_capacity_floor_victim_candidate_copy_tokens.append(
                    f"{request.request_id}:"
                    f"{candidate.estimated_d2h_eviction_tokens}/"
                    f"{candidate.estimated_ring_staging_h2d_tokens}/"
                    f"{candidate.estimated_decode_reload_h2d_tokens}/"
                    f"{candidate.estimated_total_copy_tokens}"
                )
                useful_reclaim = min(
                    max(0, candidate.estimated_reclaimable_blocks),
                    required_blocks,
                )
                virtual_context_capacity_floor_victim_candidate_copy_scores.append(
                    f"{request.request_id}:"
                    f"{candidate.estimated_total_copy_tokens}/"
                    f"{useful_reclaim}"
                )
            (
                capacity_floor_attempts,
                capacity_floor_failures,
                capacity_floor_reclaimed_victims,
                capacity_floor_reasons,
                _capacity_floor_final_free_blocks,
            ) = self._reclaim_prefill_victims_until_hbm_target(
                target_free_blocks=free_blocks + commit_required_blocks,
                excluded_request_ids=(
                    {scheduled_prefill_fit_target_request_id}
                    if (
                        scheduled_prefill_fit_target_request_id
                        and not (
                            fit_preparation is not None
                            and scheduled_prefill_fit_target_request_id
                            in fit_preparation.victim_request_ids
                        )
                    )
                    else set()
                ),
                prepared_request_ids=(
                    fit_preparation.victim_request_ids
                    if (
                        scheduled_prefill_fit_preparation_ready
                        and fit_preparation is not None
                    )
                    else ()
                ),
            )
            virtual_context_capacity_floor_release_attempt_count += (
                capacity_floor_attempts
            )
            virtual_context_capacity_floor_release_failed_count += (
                capacity_floor_failures
            )
            virtual_context_residency_pressure_release_attempt_count += (
                capacity_floor_attempts
            )
            virtual_context_residency_pressure_release_failed_count += (
                capacity_floor_failures
            )
            virtual_context_residency_pressure_release_reasons.update(
                capacity_floor_reasons
            )
            for released_victim in capacity_floor_reclaimed_victims:
                victim_request_id = released_victim.request_id
                defer_tokens = released_victim.deferred_tokens
                virtual_context_deferred_req_ids.add(victim_request_id)
                prefill_deferred_by_residency_budget += 1
                preempt_reset_avoided_tokens += defer_tokens
                prefill_dram_resident_tokens += defer_tokens
                virtual_context_capacity_floor_release_success_count += 1
                virtual_context_residency_pressure_release_success_count += 1
                virtual_context_residency_pressure_release_tokens += (
                    defer_tokens
                )
                virtual_context_capacity_floor_victim_selected_request_ids.append(
                    victim_request_id
                )
                selected_candidate = released_victim.candidate_snapshot
                virtual_context_capacity_floor_victim_selected_reclaim_blocks_total += (
                    selected_candidate.estimated_reclaimable_blocks
                )
                scheduled_prefill_fit_partial_release_blocks += (
                    released_victim.released_logical_blocks
                )
                if scheduled_prefill_fit_active:
                    scheduled_prefill_fit_whole_request_victim_count += 1
                    scheduled_prefill_fit_whole_request_reclaimed_blocks += (
                        released_victim.reclaimed_blocks
                    )
                if not virtual_context_capacity_floor_victim_selected_request_id:
                    virtual_context_capacity_floor_victim_selected_request_id = (
                        victim_request_id
                    )
                    virtual_context_capacity_floor_victim_selected_context_tokens = (
                        defer_tokens
                    )
                    selected_remaining_tokens = (
                        selected_candidate.remaining_prefill_tokens
                    )
                    (
                        virtual_context_capacity_floor_victim_selected_remaining_tokens
                    ) = selected_remaining_tokens
                    (
                        virtual_context_capacity_floor_victim_selected_ring_chunks
                    ) = fixed_ring_chunk_count(
                        selected_remaining_tokens,
                        ring_chunk_tokens,
                    )
                    (
                        virtual_context_capacity_floor_victim_selected_reclaim_blocks
                    ) = selected_candidate.estimated_reclaimable_blocks
                    virtual_context_capacity_floor_victim_selected_ring_work = (
                        fixed_ring_attention_work(
                            selected_candidate.logical_context_tokens,
                            selected_remaining_tokens,
                            ring_chunk_tokens,
                        )
                    )
                    virtual_context_capacity_floor_victim_selected_d2h_eviction_tokens = (
                        selected_candidate.estimated_d2h_eviction_tokens
                    )
                    virtual_context_capacity_floor_victim_selected_ring_staging_h2d_tokens = (
                        selected_candidate.estimated_ring_staging_h2d_tokens
                    )
                    virtual_context_capacity_floor_victim_selected_decode_reload_h2d_tokens = (
                        selected_candidate.estimated_decode_reload_h2d_tokens
                    )
                    virtual_context_capacity_floor_victim_selected_total_copy_tokens = (
                        selected_candidate.estimated_total_copy_tokens
                    )
            if (
                scheduled_prefill_fit_preparation_ready
                and capacity_floor_reclaimed_victims
            ):
                committed_reclaim_blocks = sum(
                    released_victim.released_logical_blocks
                    for released_victim in capacity_floor_reclaimed_victims
                )
                decode_growth_headroom_blocks = (
                    scheduled_prefill_fit_soft_headroom_blocks
                )
                if (
                    committed_reclaim_blocks > 0
                    and scheduled_prefill_fit_target_request_id
                    and scheduled_prefill_fit_target_chunk_tokens > 0
                    and decode_growth_headroom_blocks > 0
                ):
                    self._scheduled_prefill_fit_decode_growth_runway = (
                        ScheduledPrefillFitDecodeGrowthRunway(
                            target_request_id=(
                                scheduled_prefill_fit_target_request_id
                            ),
                            target_logical_tokens=(
                                scheduled_prefill_fit_target_logical_tokens
                            ),
                            target_chunk_tokens=(
                                scheduled_prefill_fit_target_chunk_tokens
                            ),
                            capacity_headroom_blocks=(
                                decode_growth_headroom_blocks
                            ),
                            committed_reclaim_blocks=(
                                committed_reclaim_blocks
                            ),
                            created_iteration_id=(
                                self._iteration_debug_index
                            ),
                        )
                    )
                    self._scheduled_prefill_fit_decode_growth_runway_created_count += (
                        1
                    )
                self._scheduled_prefill_fit_preparation = None
            virtual_context_ring_prefill_admission_tokens_effective = (
                self._dynamic_virtual_context_ring_prefill_admission_tokens()
            )
        tbt_decode_policy_candidate_order = (
            self._tbt_begin_decode_iteration()
        )
        tbt_decode_policy_active_this_iteration = bool(
            self.enable_tbt_decode_scheduling
            and tbt_decode_policy_candidate_order
            and (
                not self.pd_decoupled_no_mix
                or pd_decoupled_selected_phase == "decode"
            )
        )
        if tbt_decode_policy_active_this_iteration:
            self._tbt_prioritize_decode_candidates()
        if not ttft_prefill_policy_candidate_order:
            ttft_prefill_policy_candidate_order = (
                self._ttft_begin_prefill_chunk_iteration()
            )
        ttft_prefill_preferred_request_id = (
            self.ttft_prefill_chunk_scheduler.preferred_request_id
            if self.enable_ttft_prefill_chunk_scheduling
            else ""
        )
        ttft_prefill_force_single_reason = ""
        if (
            scheduled_prefill_fit_active
            and (
                scheduled_prefill_fit_target_reclaim_blocks > 0
                or scheduled_prefill_fit_preparation_waiting
                or scheduled_prefill_fit_preparation_ready
            )
        ):
            ttft_prefill_force_single_reason = (
                "scheduled_prefill_fit_transaction"
            )
        if (
            not self.pd_decoupled_no_mix
            or pd_decoupled_selected_phase == "prefill"
        ):
            ttft_prefill_frontier = (
                self._ttft_plan_prefill_admission_frontier(
                    ttft_prefill_policy_candidate_order,
                    reserve_decode_work=not self.pd_decoupled_no_mix,
                    force_single_reason=ttft_prefill_force_single_reason,
                )
            )
            ttft_prefill_policy_selected_request_ids = (
                ttft_prefill_frontier.request_ids
            )
        else:
            ttft_prefill_frontier = PrefillAdmissionFrontier(
                token_budget_before=self.max_num_scheduled_tokens,
                token_budget_after=self.max_num_scheduled_tokens,
                stop_reason="non_prefill_phase",
            )
        ttft_prefill_policy_selected_request_id_set = frozenset(
            ttft_prefill_policy_selected_request_ids
        )
        ttft_prefill_policy_active_this_iteration = bool(
            self.enable_ttft_prefill_chunk_scheduling
            and ttft_prefill_policy_candidate_order
            and (
                not self.pd_decoupled_no_mix
                or pd_decoupled_selected_phase == "prefill"
            )
        )
        if ttft_prefill_policy_active_this_iteration:
            self._ttft_prioritize_prefill_frontier(
                ttft_prefill_policy_selected_request_ids
            )
        active_decode_during_prefill_phase = (
            self.pd_decoupled_no_mix
            and pd_decoupled_selected_phase == "prefill"
            and (
                running_phase_counts_before["num_running_decode_reqs"] > 0
                or self._pd_pending_decode_active()
            )
        )
        if (
            virtual_context_ring_prefill_admission_tokens_effective > 0
            and self.pd_decoupled_no_mix
            and pd_decoupled_selected_phase == "prefill"
            and self.ttft_prefill_native_chunk_max_tokens <= 0
        ):
            ring_prefill_phase_budget_original = token_budget
            token_budget = min(
                token_budget,
                virtual_context_ring_prefill_admission_tokens_effective,
            )
            ring_prefill_phase_budget_capped = token_budget
            ring_prefill_phase_budget_applied = int(
                token_budget < ring_prefill_phase_budget_original
            )

        # First, schedule the RUNNING requests.
        req_index = 0
        running_scan_start = time.perf_counter()
        while req_index < len(self.running) and token_budget > 0:
            # Count actual admitted rows, not candidates or Futures. This map
            # also removes rows if ordinary preemption rolls back this pass.
            if (isolation_phase_request_limit > 0
                    and len(num_scheduled_tokens) >= isolation_phase_request_limit):
                isolation_phase_request_limit_hit = True
                break
            request = self.running[req_index]

            if (
                self.pd_decoupled_no_mix
                and pd_decoupled_selected_phase == "prefill"
                and not self._debug_needs_prefill_lane(request)
            ):
                # Pending final-Prefill output remains a Prefill owner until
                # its exact generation is recovered. Only a true Decode
                # request is filtered from a forced Prefill scan here.
                pd_decoupled_skipped_running_decode_reqs += 1
                req_index += 1
                continue

            next_request_phase = (
                pd_decoupled_selected_phase
                if pd_decoupled_selected_phase in {"decode", "prefill"}
                else (
                    "prefill"
                    if self._debug_needs_prefill_lane(request)
                    else "decode"
                )
            )
            if self.blocks_next_schedule(request, next_request_phase):
                pd_separate_scheduler_skipped_pending_reqs += 1
                req_index += 1
                continue

            if request.request_id in isolation_waiting_decode_reclaim_wait_ids:
                prefill_defer_reason_counts["scheduled_prefill_fit_d2h_wait"] += 1
                self._debug_record_request_defer(request, "scheduled_prefill_fit_d2h_wait")
                deferred_request_ids.add(request.request_id)
                req_index += 1
                continue

            if (
                ttft_prefill_policy_active_this_iteration
                and not self._ttft_prefill_request_is_selected(
                    request,
                    ttft_prefill_policy_selected_request_id_set,
                )
            ):
                ttft_prefill_policy_skipped_running_reqs += 1
                prefill_defer_reason_counts[
                    "ttft_prefill_chunk_policy"
                ] += 1
                self._debug_record_request_defer(
                    request, "ttft_prefill_chunk_policy"
                )
                deferred_request_ids.add(request.request_id)
                req_index += 1
                continue

            if (
                scheduled_prefill_fit_preparation_waiting
                and request.request_id
                == scheduled_prefill_fit_target_request_id
                and self._debug_has_prefill_remaining(request)
            ):
                prefill_defer_reason_counts[
                    "scheduled_prefill_fit_d2h_wait"
                ] += 1
                self._debug_record_request_defer(
                    request,
                    "scheduled_prefill_fit_d2h_wait",
                )
                deferred_request_ids.add(request.request_id)
                req_index += 1
                continue

            if (
                self.virtual_context_residency_victim_policy
                == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
                and self._debug_has_prefill_remaining(request)
                and self._virtual_context_reload_required_page_count(
                    [request.request_id],
                    "prefill",
                )
                > 0
            ):
                (
                    promoted_to_native,
                    promotion_pages,
                    promotion_required_blocks,
                    promotion_reason,
                ) = self._prepare_virtual_context_native_prefill_promotion(
                    request,
                    ring_token_cap=(
                        virtual_context_ring_prefill_admission_tokens_effective
                        or self._virtual_context_residency_fixed_ring_chunk_tokens()
                    ),
                )
                virtual_context_native_prefill_promotion_pages = max(
                    virtual_context_native_prefill_promotion_pages,
                    promotion_pages,
                )
                virtual_context_native_prefill_promotion_required_blocks = max(
                    virtual_context_native_prefill_promotion_required_blocks,
                    promotion_required_blocks,
                )
                virtual_context_native_prefill_promotion_reasons.add(
                    promotion_reason
                )
                if promoted_to_native:
                    virtual_context_native_prefill_promotion_request_ids.add(
                        request.request_id
                    )
                    tiered_explicit_execution_plans[request.request_id] = (
                        TIERED_EXECUTION_NATIVE_RELOAD,
                        promotion_pages,
                    )
                    tiered_native_reload_planned_tokens[request.request_id] = min(
                        max(
                            request.num_computed_tokens,
                            request.virtual_context_deferred_logical_computed_tokens,
                        ),
                        promotion_required_blocks * self.block_size,
                    )
                    virtual_context_decode_handoff_reload_full_block_table_req_ids.add(
                        request.request_id
                    )

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.pd_decoupled_no_mix:
                request_is_prefill = self._debug_has_prefill_remaining(request)
                if pd_decoupled_selected_phase == "idle":
                    req_index += 1
                    continue
                if (
                    pd_decoupled_selected_phase == "decode"
                    and request_is_prefill
                ):
                    pd_decoupled_skipped_running_prefill_reqs += 1
                    pd_residency_skipped_running_prefill_request_ids.add(
                        request.request_id
                    )
                    pd_residency_last_free_blocks = (
                        self.kv_cache_manager.block_pool.get_num_free_blocks()
                    )
                    pd_residency_last_decode_resident_blocks = (
                        self._debug_running_decode_resident_blocks()
                    )
                    required_blocks = (
                        self._debug_record_pd_residency_prefill_skip(
                            request,
                            scheduled_timestamp,
                            pd_residency_last_free_blocks,
                            pd_residency_last_decode_resident_blocks,
                        )
                    )
                    pd_residency_prefill_required_blocks_by_req[
                        request.request_id
                    ] = required_blocks
                    if required_blocks > pd_residency_last_free_blocks:
                        pd_residency_serialized_request_ids.add(
                            request.request_id
                        )
                        if (
                            self.virtual_context_residency_victim_policy
                            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
                        ):
                            # Decode-lane passes are observational for this
                            # policy.  Exact D2H preparation and allocator
                            # commit belong to a Prefill boundary; retrying a
                            # victim-local reactive reclaim once per Decode
                            # token creates a host-side busy loop and can pick
                            # a request different from the prepared best-fit
                            # victim set.
                            virtual_context_residency_pressure_release_reasons.add(
                                "scheduled_prefill_fit_prefill_lane_reclaim"
                            )
                            req_index += 1
                            continue
                        virtual_context_residency_pressure_release_attempt_count += (
                            1
                        )
                        (
                            did_defer,
                            defer_reason,
                            defer_tokens,
                        ) = self._try_defer_running_prefill_for_virtual_context(
                            request,
                            max_reclaim_blocks=(
                                self._scheduled_prefill_fit_default_reclaim_limit(
                                    request
                                )
                            ),
                        )
                        scheduled_prefill_fit_partial_release_blocks += (
                            self._scheduled_prefill_fit_last_release_blocks
                        )
                        virtual_context_residency_pressure_release_reasons.add(
                            defer_reason
                        )
                        if did_defer:
                            self.running.pop(req_index)
                            virtual_context_deferred_req_ids.add(
                                request.request_id
                            )
                            prefill_deferred_by_residency_budget += 1
                            preempt_reset_avoided_tokens += defer_tokens
                            prefill_dram_resident_tokens += defer_tokens
                            virtual_context_residency_pressure_release_success_count += (
                                1
                            )
                            virtual_context_residency_pressure_release_tokens += (
                                defer_tokens
                            )
                            continue
                        virtual_context_residency_pressure_release_failed_count += 1
                        if self.virtual_context_defer_prefill:
                            prefill_defer_residency_fallback_count += 1
                            virtual_context_defer_fallback_reasons.add(
                                defer_reason
                            )
                            self._debug_record_virtual_context_defer_fallback(
                                request, defer_reason
                            )
                    req_index += 1
                    continue
                if (
                    pd_decoupled_selected_phase == "prefill"
                    and not request_is_prefill
                ):
                    # Defensive classification check. The normal path already
                    # filtered this request before pending-dependency and
                    # residency work.
                    pd_decoupled_skipped_running_decode_reqs += 1
                    req_index += 1
                    continue
                if (
                    pd_decoupled_selected_phase == "prefill"
                    and request_is_prefill
                    and active_decode_during_prefill_phase
                    and self.virtual_context_enabled
                ):
                    (
                        ring_feasible,
                        reload_required_pages,
                        force_all,
                    ) = self._prefill_ring_feasible_with_active_decode(
                        request
                    )
                    if not ring_feasible:
                        (
                            native_safe,
                            _reload_required_pages,
                            _free_blocks,
                            _threshold_blocks,
                            _bootstrap_blocks,
                            _ring_blocks,
                            native_safe_reason,
                        ) = self._prefill_native_safe_before_ring_pressure(
                            request
                        )
                        if native_safe:
                            virtual_context_residency_pressure_release_reasons.add(
                                native_safe_reason
                            )
                            virtual_context_active_decode_ring_prefill_allowed_count += 1
                        else:
                            (
                                did_defer,
                                defer_reason,
                                defer_tokens,
                            ) = self._try_make_running_prefill_ring_required(
                                request
                            )
                            scheduled_prefill_fit_partial_release_blocks += (
                                self._scheduled_prefill_fit_last_release_blocks
                            )
                            virtual_context_residency_pressure_release_attempt_count += (
                                1
                            )
                            virtual_context_residency_pressure_release_reasons.add(
                                defer_reason
                            )
                            if did_defer:
                                self.running.pop(req_index)
                                virtual_context_deferred_req_ids.add(
                                    request.request_id
                                )
                                prefill_deferred_by_residency_budget += 1
                                preempt_reset_avoided_tokens += defer_tokens
                                prefill_dram_resident_tokens += defer_tokens
                                virtual_context_residency_pressure_release_success_count += (
                                    1
                                )
                                virtual_context_residency_pressure_release_tokens += (
                                    defer_tokens
                                )
                                continue
                            virtual_context_residency_pressure_release_failed_count += 1
                            if (
                                self.virtual_context_dram_register_pressure_only
                                and defer_reason in {
                                    "tiered_provider_copy_not_ready",
                                    "tiered_provider_no_release_candidates",
                                }
                                and self.virtual_context_residency_victim_policy
                                != ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
                            ):
                                # Pressure-only registration needs one native
                                # forward to snapshot this request's current HBM
                                # pages before they can become DRAM-authoritative.
                                # Fall through and schedule that bounded prefill
                                # work; the worker receives the request id via
                                # SchedulerOutput control metadata.
                                virtual_context_pressure_snapshot_request_ids.add(
                                    request.request_id
                                )
                            else:
                                virtual_context_active_decode_native_prefill_blocked_count += (
                                    1
                                )
                                req_index += 1
                                continue
                    else:
                        virtual_context_active_decode_ring_prefill_allowed_count += 1
                        virtual_context_active_decode_ring_prefill_reload_pages += (
                            reload_required_pages
                        )
                        virtual_context_active_decode_ring_prefill_force_all_count += (
                            force_all
                        )

            if self._debug_has_prefill_remaining(request):
                recompute_tokens = (
                    request.virtual_context_deferred_logical_computed_tokens
                )
                free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
                (
                    should_recompute,
                    reload_pages,
                    min_free_blocks,
                ) = self._should_recompute_virtual_context_on_hbm_relief(
                    request,
                    free_blocks,
                )
                if should_recompute:
                    request = self.running.pop(req_index)
                    self._preempt_request(request, scheduled_timestamp)
                    self._clear_virtual_context_deferred_state_for_recompute(
                        request
                    )
                    preempted_reqs.append(request)
                    virtual_context_hbm_relief_recompute_count += 1
                    virtual_context_hbm_relief_recompute_tokens += (
                        recompute_tokens
                    )
                    virtual_context_hbm_relief_recompute_reload_pages += (
                        reload_pages
                    )
                    virtual_context_hbm_relief_recompute_min_free_blocks = max(
                        virtual_context_hbm_relief_recompute_min_free_blocks,
                        min_free_blocks,
                    )
                    virtual_context_hbm_relief_recompute_free_blocks_max = max(
                        virtual_context_hbm_relief_recompute_free_blocks_max,
                        free_blocks,
                    )
                    virtual_context_hbm_relief_recompute_request_ids.add(
                        request.request_id
                    )
                    continue

            decode_handoff_preparation_waiting = False
            native_decode_runtime_fast_path = (
                not self._debug_has_prefill_remaining(request)
                and self._native_decode_runtime_fast_path_eligible(request)
            )
            if native_decode_runtime_fast_path:
                native_decode_runtime_fast_path_count += 1
                decode_handoff_ready = True
                decode_handoff_pages = 0
                decode_handoff_required_blocks = 0
                decode_handoff_min_free_blocks = 0
                decode_handoff_reason = "native_hbm_fast_path"
            elif not self._debug_has_prefill_remaining(request):
                native_handoff_start = time.perf_counter()
                try:
                    (
                        decode_handoff_ready,
                        decode_handoff_pages,
                        decode_handoff_required_blocks,
                        decode_handoff_min_free_blocks,
                        decode_handoff_reason,
                    ) = self._prepare_virtual_context_native_decode_handoff(
                        request
                    )
                finally:
                    schedule_native_handoff_duration_s += (
                        time.perf_counter() - native_handoff_start
                    )
                if decode_handoff_pages > 0:
                    virtual_context_decode_handoff_reload_candidate_count += 1
                    virtual_context_decode_handoff_reload_pages += (
                        decode_handoff_pages
                    )
                    virtual_context_decode_handoff_reload_required_blocks = max(
                        virtual_context_decode_handoff_reload_required_blocks,
                        decode_handoff_required_blocks,
                    )
                    virtual_context_decode_handoff_reload_min_free_blocks = max(
                        virtual_context_decode_handoff_reload_min_free_blocks,
                        decode_handoff_min_free_blocks,
                    )
                    virtual_context_decode_handoff_reload_reasons.add(
                        decode_handoff_reason
                    )
                    if (
                        not decode_handoff_ready
                        and decode_handoff_reason
                        in {
                            "insufficient_hbm_blocks",
                            "insufficient_hbm_blocks_with_ring_progress_reserve",
                        }
                        and decode_handoff_min_free_blocks > 0
                    ):
                        reclaim_free_before = (
                            self.kv_cache_manager.block_pool.get_num_free_blocks()
                        )
                        virtual_context_decode_handoff_reclaim_target_blocks = max(
                            virtual_context_decode_handoff_reclaim_target_blocks,
                            decode_handoff_min_free_blocks,
                        )
                        virtual_context_decode_handoff_reclaim_shortfall_blocks = max(
                            virtual_context_decode_handoff_reclaim_shortfall_blocks,
                            max(
                                0,
                                decode_handoff_min_free_blocks
                                - reclaim_free_before,
                            ),
                        )
                        prepared_decode_victim_ids: tuple[str, ...] = ()
                        existing_preparation = (
                            self._scheduled_prefill_fit_preparation
                        )
                        if (
                            self.virtual_context_residency_victim_policy
                            == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
                            and existing_preparation is not None
                            and existing_preparation.target_request_id
                            == request.request_id
                        ):
                            (
                                decode_preparation,
                                _ready_pages,
                                _planned_pages,
                                decode_preparation_ready,
                            ) = self._scheduled_prefill_fit_preparation_status(
                                live_target_request_id=request.request_id,
                                required_blocks=max(
                                    0,
                                    decode_handoff_min_free_blocks
                                    - reclaim_free_before,
                                ),
                                hard_required_blocks=max(
                                    0,
                                    decode_handoff_required_blocks
                                    - reclaim_free_before,
                                ),
                            )
                            if (
                                decode_preparation is not None
                                and decode_preparation_ready
                            ):
                                prepared_decode_victim_ids = (
                                    decode_preparation.victim_request_ids
                                )
                            elif decode_preparation is not None:
                                decode_handoff_preparation_waiting = True
                                virtual_context_decode_handoff_reclaim_reasons.add(
                                    "scheduled_prefill_fit_d2h_wait"
                                )
                        if decode_handoff_preparation_waiting:
                            reclaimed_victims = []
                            reclaim_attempts = 0
                            reclaim_failures = 0
                            reclaim_reasons = {
                                "scheduled_prefill_fit_d2h_wait"
                            }
                            reclaim_final_free_blocks = reclaim_free_before
                        else:
                            (
                                reclaim_attempts,
                                reclaim_failures,
                                reclaimed_victims,
                                reclaim_reasons,
                                reclaim_final_free_blocks,
                            ) = self._reclaim_prefill_victims_until_hbm_target(
                                target_free_blocks=decode_handoff_min_free_blocks,
                                excluded_request_ids={request.request_id},
                                prepared_request_ids=(
                                    prepared_decode_victim_ids
                                ),
                                require_full_target_before_reclaim=(
                                    self.virtual_context_residency_victim_policy
                                    == ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
                                ),
                                reclaim_feasibility_cache_key=request.request_id,
                            )
                        virtual_context_decode_handoff_reclaim_attempt_count += (
                            reclaim_attempts
                        )
                        virtual_context_decode_handoff_reclaim_failed_count += (
                            reclaim_failures
                        )
                        virtual_context_decode_handoff_reclaim_reasons.update(
                            reclaim_reasons
                        )
                        virtual_context_residency_pressure_release_attempt_count += (
                            reclaim_attempts
                        )
                        virtual_context_residency_pressure_release_failed_count += (
                            reclaim_failures
                        )
                        virtual_context_residency_pressure_release_reasons.update(
                            reclaim_reasons
                        )
                        virtual_context_decode_handoff_reclaim_final_free_blocks = max(
                            virtual_context_decode_handoff_reclaim_final_free_blocks,
                            reclaim_final_free_blocks,
                        )
                        for released_victim in reclaimed_victims:
                            victim_request_id = released_victim.request_id
                            victim_tokens = released_victim.deferred_tokens
                            reclaimed_blocks = released_victim.reclaimed_blocks
                            if scheduled_prefill_fit_active:
                                scheduled_prefill_fit_whole_request_victim_count += 1
                                (
                                    scheduled_prefill_fit_whole_request_reclaimed_blocks
                                ) += reclaimed_blocks
                            virtual_context_decode_handoff_reclaim_success_count += (
                                1
                            )
                            virtual_context_decode_handoff_reclaimed_blocks += (
                                reclaimed_blocks
                            )
                            virtual_context_decode_handoff_reclaim_victim_ids.append(
                                victim_request_id
                            )
                            virtual_context_deferred_req_ids.add(
                                victim_request_id
                            )
                            prefill_deferred_by_residency_budget += 1
                            preempt_reset_avoided_tokens += victim_tokens
                            prefill_dram_resident_tokens += victim_tokens
                            virtual_context_residency_pressure_release_success_count += (
                                1
                            )
                            virtual_context_residency_pressure_release_tokens += (
                                victim_tokens
                            )

                        if reclaimed_victims:
                            # Victims can precede the decode request in
                            # self.running, so restore its current index before
                            # retrying the native handoff against the new HBM
                            # state.
                            req_index = self.running.index(request)
                            native_handoff_retry_start = time.perf_counter()
                            try:
                                (
                                    decode_handoff_ready,
                                    decode_handoff_pages,
                                    decode_handoff_required_blocks,
                                    decode_handoff_min_free_blocks,
                                    decode_handoff_reason,
                                ) = self._prepare_virtual_context_native_decode_handoff(
                                    request
                                )
                            finally:
                                schedule_native_handoff_duration_s += (
                                    time.perf_counter()
                                    - native_handoff_retry_start
                                )
                            virtual_context_decode_handoff_reload_required_blocks = max(
                                virtual_context_decode_handoff_reload_required_blocks,
                                decode_handoff_required_blocks,
                            )
                            virtual_context_decode_handoff_reload_min_free_blocks = max(
                                virtual_context_decode_handoff_reload_min_free_blocks,
                                decode_handoff_min_free_blocks,
                            )
                            virtual_context_decode_handoff_reload_reasons.add(
                                decode_handoff_reason
                            )
                            (
                                virtual_context_decode_handoff_reclaim_final_free_blocks
                            ) = max(
                                (
                                    virtual_context_decode_handoff_reclaim_final_free_blocks
                                ),
                                self.kv_cache_manager.block_pool.get_num_free_blocks(),
                            )
                            if decode_handoff_ready:
                                (
                                    virtual_context_decode_handoff_reclaim_retry_success_count
                                ) += 1

                    if decode_handoff_ready:
                        virtual_context_decode_handoff_reload_scheduled_count += 1
                        virtual_context_decode_handoff_reload_request_ids.add(
                            request.request_id
                        )
                        if decode_handoff_pages > 0:
                            tiered_explicit_execution_plans[request.request_id] = (
                                TIERED_EXECUTION_NATIVE_RELOAD,
                                decode_handoff_pages,
                            )
                            tiered_native_reload_planned_tokens[request.request_id] = min(
                                request.num_computed_tokens,
                                decode_handoff_pages * self.block_size,
                            )
                        (
                            virtual_context_decode_handoff_reload_full_block_table_req_ids
                        ).add(request.request_id)
                    else:
                        if decode_handoff_preparation_waiting:
                            virtual_context_decode_handoff_reload_blocked_count += 1
                            prefill_defer_reason_counts[
                                "scheduled_prefill_fit_d2h_wait"
                            ] += 1
                            self._debug_record_request_defer(
                                request,
                                "scheduled_prefill_fit_d2h_wait",
                            )
                            deferred_request_ids.add(request.request_id)
                            req_index += 1
                            continue
                        # A decode-ready request can wait behind another
                        # request's protected decode hotset for an arbitrary
                        # time. Once every currently resident page, including
                        # a partial tail, has a completed provider copy, park
                        # that immutable context in DRAM instead of pinning its
                        # remaining HBM blocks while it waits for a full native
                        # reload opportunity.
                        did_park_decode_context = False
                        park_reason = ""
                        park_tokens = 0
                        if decode_handoff_reason == "insufficient_hbm_blocks":
                            (
                                did_park_decode_context,
                                park_reason,
                                park_tokens,
                            ) = self._try_defer_running_prefill_to_tiered_provider(
                                request,
                                request.num_computed_tokens,
                                deferred_state="decode_hbm_reload_wait_dram",
                            )
                            virtual_context_residency_pressure_release_reasons.add(
                                park_reason
                            )
                        if did_park_decode_context:
                            self.running.pop(req_index)
                            virtual_context_deferred_req_ids.add(
                                request.request_id
                            )
                            virtual_context_residency_pressure_release_success_count += 1
                            virtual_context_residency_pressure_release_tokens += (
                                park_tokens
                            )
                            continue
                        virtual_context_decode_handoff_reload_blocked_count += 1
                        if decode_handoff_min_free_blocks > 0:
                            if (
                                virtual_context_decode_handoff_retry_min_free_blocks
                                <= 0
                            ):
                                virtual_context_decode_handoff_retry_min_free_blocks = (
                                    decode_handoff_min_free_blocks
                                )
                            else:
                                (
                                    virtual_context_decode_handoff_retry_min_free_blocks
                                ) = min(
                                    (
                                        virtual_context_decode_handoff_retry_min_free_blocks
                                    ),
                                    decode_handoff_min_free_blocks,
                                )
                        prefill_defer_reason_counts[
                            "decode_hbm_reload_wait"
                        ] += 1
                        self._debug_record_request_defer(
                            request,
                            "decode_hbm_reload_wait",
                        )
                        deferred_request_ids.add(request.request_id)
                        req_index += 1
                        continue

            if native_decode_runtime_fast_path:
                decode_cold_allowed = True
                decode_cold_required_pages = 0
                decode_cold_slack_ms = 0.0
                decode_cold_after_prefill_allowed = False
            else:
                decode_cold_guard_start = time.perf_counter()
                try:
                    (
                        decode_cold_allowed,
                        decode_cold_required_pages,
                        decode_cold_slack_ms,
                        decode_cold_after_prefill_allowed,
                    ) = self._check_virtual_context_decode_cold_streaming_admission(
                        request,
                        scheduled_timestamp,
                    )
                finally:
                    schedule_decode_cold_guard_duration_s += (
                        time.perf_counter() - decode_cold_guard_start
                    )
            if decode_cold_required_pages > 0:
                virtual_context_decode_cold_streaming_required_pages += (
                    decode_cold_required_pages
                )
                virtual_context_decode_cold_streaming_last_slack_ms = (
                    decode_cold_slack_ms
                )
                if decode_cold_allowed:
                    virtual_context_decode_cold_streaming_allowed_count += 1
                    if decode_cold_after_prefill_allowed:
                        virtual_context_decode_after_prefill_streaming_allowed_count += (
                            1
                        )
                else:
                    virtual_context_decode_cold_streaming_blocked_count += 1
                    req_index += 1
                    continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            prefill_execution_mode = (
                self._ttft_prefill_execution_mode(
                    request,
                    virtual_context_native_prefill_promotion_request_ids,
                )
                if (
                    self.ttft_prefill_native_chunk_max_tokens > 0
                    and self._debug_has_prefill_remaining(request)
                )
                else ""
            )
            if ttft_prefill_policy_active_this_iteration:
                num_new_tokens = self._ttft_cap_selected_prefill_chunk(
                    request,
                    num_new_tokens,
                    execution_mode=prefill_execution_mode,
                )
            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )
            uncapped_num_new_tokens = num_new_tokens
            (
                prefill_ring_allowed,
                num_new_tokens,
                prefill_ring_decode_pages,
                prefill_ring_leftover_tokens,
            ) = self._check_virtual_context_prefill_ring_bandwidth_admission(
                request,
                num_new_tokens,
            )
            if prefill_ring_decode_pages > 0:
                virtual_context_prefill_ring_bandwidth_decode_reload_pages = max(
                    virtual_context_prefill_ring_bandwidth_decode_reload_pages,
                    prefill_ring_decode_pages,
                )
                virtual_context_prefill_ring_bandwidth_leftover_tokens = max(
                    virtual_context_prefill_ring_bandwidth_leftover_tokens,
                    prefill_ring_leftover_tokens,
                )
                if prefill_ring_allowed and num_new_tokens > 0:
                    (
                        virtual_context_prefill_ring_with_decode_cold_allowed_count
                    ) += 1
                    (
                        virtual_context_prefill_ring_with_decode_cold_allowed_tokens
                    ) += num_new_tokens
            if not prefill_ring_allowed:
                virtual_context_prefill_ring_bandwidth_deferred_count += 1
                prefill_defer_reason_counts[
                    "prefill_ring_bandwidth_decode_reserved"
                ] += 1
                self._debug_record_request_defer(
                    request,
                    "prefill_ring_bandwidth_decode_reserved",
                )
                deferred_request_ids.add(request.request_id)
                req_index += 1
                continue

            num_new_tokens = (
                self._cap_virtual_context_ring_prefill_admission_tokens(
                    request,
                    num_new_tokens,
                    execution_mode=prefill_execution_mode,
                )
            )
            if self.pd_decoupled_no_mix:
                num_new_tokens = self._cap_pd_phase_tokens(
                    request,
                    num_new_tokens,
                    num_computed_tokens=request.num_computed_tokens,
                    selected_phase=pd_decoupled_selected_phase,
                )
            if num_new_tokens < uncapped_num_new_tokens:
                ring_prefill_admission_cap_hit_count += 1
                ring_prefill_admission_cap_running_hit_count += 1
                ring_prefill_admission_original_tokens_sum += (
                    uncapped_num_new_tokens
                )
                ring_prefill_admission_capped_tokens_sum += num_new_tokens

            if (
                num_new_tokens == 0
                and uncapped_num_new_tokens > 0
                and self._virtual_context_ring_bw_last_reason
                == "below_min_progress_defer"
            ):
                if (
                    self.virtual_context_ring_residency_admission
                    and self.virtual_context_defer_prefill
                ):
                    virtual_context_capacity_floor_release_attempt_count += 1
                    virtual_context_residency_pressure_release_attempt_count += 1
                    (
                        did_defer,
                        defer_reason,
                        defer_tokens,
                    ) = self._try_defer_running_prefill_for_virtual_context(
                        request,
                        max_reclaim_blocks=(
                            self._scheduled_prefill_fit_reclaim_limit(
                                request,
                                uncapped_num_new_tokens,
                            )
                        ),
                    )
                    scheduled_prefill_fit_partial_release_blocks += (
                        self._scheduled_prefill_fit_last_release_blocks
                    )
                    virtual_context_residency_pressure_release_reasons.add(
                        defer_reason
                    )
                    if did_defer:
                        self.running.pop(req_index)
                        virtual_context_deferred_req_ids.add(
                            request.request_id
                        )
                        prefill_deferred_by_residency_budget += 1
                        preempt_reset_avoided_tokens += defer_tokens
                        prefill_dram_resident_tokens += defer_tokens
                        virtual_context_capacity_floor_release_success_count += 1
                        virtual_context_residency_pressure_release_success_count += (
                            1
                        )
                        virtual_context_residency_pressure_release_tokens += (
                            defer_tokens
                        )
                        continue
                    virtual_context_capacity_floor_release_failed_count += 1
                    virtual_context_residency_pressure_release_failed_count += 1
                ring_prefill_admission_min_progress_defer_count += 1
                prefill_defer_reason_counts[
                    "ring_chunk_below_min_progress"
                ] += 1
                self._debug_record_request_defer(
                    request,
                    "ring_chunk_below_min_progress",
                )
                deferred_request_ids.add(request.request_id)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if (
                num_new_tokens > 0
                and pd_decoupled_selected_phase == "prefill"
                and request_is_prefill
                and active_decode_during_prefill_phase
                and not self.virtual_context_enabled
                and self.pd_muxwise_dynamic_policy
            ):
                (
                    native_capacity_feasible,
                    native_chunk_required_blocks,
                    native_decode_growth_reserve_blocks,
                    native_free_blocks,
                    native_capacity_reason,
                ) = self._native_prefill_capacity_feasible_with_active_decode(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                native_required_blocks = (
                    native_chunk_required_blocks
                    + native_decode_growth_reserve_blocks
                )
                pd_native_active_decode_prefill_capacity_check_count += 1
                pd_native_active_decode_prefill_required_blocks = max(
                    pd_native_active_decode_prefill_required_blocks,
                    native_required_blocks,
                )
                pd_native_active_decode_prefill_chunk_required_blocks = max(
                    pd_native_active_decode_prefill_chunk_required_blocks,
                    native_chunk_required_blocks,
                )
                pd_native_active_decode_growth_reserve_blocks = max(
                    pd_native_active_decode_growth_reserve_blocks,
                    native_decode_growth_reserve_blocks,
                )
                pd_native_active_decode_prefill_free_blocks = native_free_blocks
                pd_native_active_decode_prefill_capacity_reason = (
                    native_capacity_reason
                )
                if not native_capacity_feasible:
                    pd_native_active_decode_prefill_capacity_defer_count += 1
                    pd_native_active_decode_prefill_capacity_blocked = True
                    if (
                        pd_native_prefill_retry_min_blocks
                        <= 0
                    ):
                        pd_native_prefill_retry_min_blocks = (
                            native_required_blocks
                        )
                    else:
                        pd_native_prefill_retry_min_blocks = min(
                            pd_native_prefill_retry_min_blocks,
                            native_required_blocks,
                        )
                    prefill_defer_reason_counts["kv_capacity"] += 1
                    self._debug_record_request_defer(request, "kv_capacity")
                    deferred_request_ids.add(request.request_id)
                    req_index += 1
                    continue

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            (
                residency_admission_allowed,
                residency_required_blocks,
                _residency_free_blocks,
                _residency_decode_reserve_blocks,
                _residency_staging_blocks,
                residency_raw_required_blocks,
                residency_decode_used_blocks,
                residency_ring_applied,
            ) = self._check_virtual_context_prefill_residency_admission(
                request,
                num_new_tokens,
                "running_prefill",
                num_lookahead_tokens=self.num_lookahead_tokens,
                scheduled_tokens=num_scheduled_tokens,
            )
            residency_admission_raw_required_blocks_max = max(
                residency_admission_raw_required_blocks_max,
                residency_raw_required_blocks,
            )
            ring_residency_admission_applied_count += residency_ring_applied
            if (
                residency_ring_applied
                and _residency_decode_reserve_blocks
                < residency_decode_used_blocks
            ):
                ring_residency_decode_reserve_relaxed_count += 1
            if not residency_admission_allowed:
                virtual_context_residency_pressure_release_attempt_count += 1
                (
                    did_defer,
                    defer_reason,
                    defer_tokens,
                ) = self._try_defer_running_prefill_for_virtual_context(
                    request,
                    max_reclaim_blocks=(
                        self._scheduled_prefill_fit_reclaim_limit(
                            request,
                            num_new_tokens,
                        )
                    ),
                )
                scheduled_prefill_fit_partial_release_blocks += (
                    self._scheduled_prefill_fit_last_release_blocks
                )
                virtual_context_residency_pressure_release_reasons.add(
                    defer_reason
                )
                if did_defer:
                    self.running.pop(req_index)
                    virtual_context_deferred_req_ids.add(request.request_id)
                    prefill_deferred_by_residency_budget += 1
                    preempt_reset_avoided_tokens += defer_tokens
                    prefill_dram_resident_tokens += defer_tokens
                    virtual_context_residency_pressure_release_success_count += (
                        1
                    )
                    virtual_context_residency_pressure_release_tokens += (
                        defer_tokens
                    )
                    new_blocks = None
                    continue
                virtual_context_residency_pressure_release_failed_count += 1
                if self.virtual_context_defer_prefill:
                    prefill_defer_residency_fallback_count += 1
                    virtual_context_defer_fallback_reasons.add(defer_reason)
                    self._debug_record_virtual_context_defer_fallback(
                        request, defer_reason
                    )
                prefill_deferred_by_residency_admission += 1
                residency_admission_deferred_request_ids.add(request.request_id)
                residency_admission_required_blocks_sum += (
                    residency_required_blocks
                )
                residency_admission_required_blocks_max = max(
                    residency_admission_required_blocks_max,
                    residency_required_blocks,
                )
                prefill_defer_reason_counts["residency_budget"] += 1
                self._debug_record_request_defer(request, "residency_budget")
                deferred_request_ids.add(request.request_id)
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            new_blocks = None
            deferred_by_virtual_context = False
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    allocate_slots_start = time.perf_counter()
                    try:
                        new_blocks = self.kv_cache_manager.allocate_slots(
                            request,
                            num_new_tokens,
                            num_lookahead_tokens=self.num_lookahead_tokens,
                        )
                    finally:
                        schedule_allocate_slots_duration_s += (
                            time.perf_counter() - allocate_slots_start
                        )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        if (
                            getattr(
                                request,
                                "kv_retention_decode_continuation_lease_active",
                                False,
                            )
                            and not self._debug_has_prefill_remaining(request)
                        ):
                            request.kv_retention_decode_continuation_waiting = (
                                False
                            )
                        break

                    # The request cannot be scheduled.
                    running_kv_capacity_miss_count += 1
                    running_kv_capacity_miss_request_ids.add(
                        request.request_id
                    )
                    request.running_kv_capacity_miss_count += 1
                    request.last_running_kv_capacity_miss_iteration_id = (
                        self._iteration_debug_index
                    )
                    request.last_running_kv_capacity_miss_num_computed_tokens = (
                        request.num_computed_tokens
                    )
                    request.last_running_kv_capacity_miss_num_tokens = (
                        request.num_tokens
                    )
                    if self._debug_has_prefill_remaining(request):
                        request.last_running_kv_capacity_miss_phase = "prefill"
                        request.running_prefill_kv_capacity_miss_count += 1
                        running_prefill_kv_capacity_miss_count += 1
                        if (
                            active_decode_during_prefill_phase
                            and not self.virtual_context_enabled
                            and self.pd_muxwise_dynamic_policy
                        ):
                            pd_native_active_decode_prefill_capacity_blocked = (
                                True
                            )
                        (
                            did_defer,
                            defer_reason,
                            defer_tokens,
                        ) = self._try_defer_running_prefill_for_virtual_context(
                            request,
                            max_reclaim_blocks=(
                                self._scheduled_prefill_fit_reclaim_limit(
                                    request,
                                    num_new_tokens,
                                )
                            ),
                        )
                        scheduled_prefill_fit_partial_release_blocks += (
                            self._scheduled_prefill_fit_last_release_blocks
                        )
                        if did_defer:
                            self.running.pop(req_index)
                            virtual_context_deferred_req_ids.add(
                                request.request_id
                            )
                            prefill_deferred_by_residency_budget += 1
                            preempt_reset_avoided_tokens += defer_tokens
                            prefill_dram_resident_tokens += defer_tokens
                            deferred_by_virtual_context = True
                            new_blocks = None
                            break
                        if self.virtual_context_defer_prefill:
                            prefill_defer_residency_fallback_count += 1
                            virtual_context_defer_fallback_reasons.add(
                                defer_reason
                            )
                            self._debug_record_virtual_context_defer_fallback(
                                request, defer_reason
                            )
                        if self.pd_wait_on_prefill_kv_capacity:
                            # P/D multiplexing should not turn KV infeasibility
                            # into repeated prefill reset/recompute. Preserve
                            # the running prefill's partial KV and let other
                            # phase lanes, especially decode, keep advancing.
                            request.running_prefill_kv_wait_count += 1
                            request.last_running_prefill_kv_wait_iteration_id = (
                                self._iteration_debug_index
                            )
                            request.last_running_prefill_kv_wait_num_computed_tokens = (
                                request.num_computed_tokens
                            )
                            request.last_running_prefill_kv_wait_num_tokens = (
                                request.num_tokens
                            )
                            running_prefill_kv_wait_count += 1
                            running_prefill_kv_wait_request_ids.add(
                                request.request_id
                            )
                            running_prefill_kv_waited = True
                            new_blocks = None
                            break
                    else:
                        request.last_running_kv_capacity_miss_phase = "decode"
                        request.running_decode_kv_capacity_miss_count += 1
                        running_decode_kv_capacity_miss_count += 1
                        if (
                            getattr(
                                request,
                                "kv_retention_decode_continuation_lease_active",
                                False,
                            )
                        ):
                            # This request already owns durable DRAM-backed
                            # prompt progress. Resetting it would turn a
                            # one-block Decode growth miss into a full prompt
                            # recomputation (the Short-4 failure). Preserve the
                            # allocation and wait for a real capacity change.
                            request.kv_retention_decode_continuation_waiting = (
                                True
                            )
                            request.kv_retention_decode_continuation_wait_count += (
                                1
                            )
                            (
                                request.kv_retention_decode_continuation_last_wait_iteration_id
                            ) = self._iteration_debug_index
                            current_free_blocks = (
                                self.kv_cache_manager.block_pool.get_num_free_blocks()
                            )
                            virtual_context_decode_handoff_reload_blocked_count += (
                                1
                            )
                            virtual_context_decode_handoff_reload_reasons.add(
                                "decode_continuation_growth_wait"
                            )
                            continuation_retry_blocks = current_free_blocks + 1
                            if (
                                virtual_context_decode_handoff_retry_min_free_blocks
                                <= 0
                            ):
                                virtual_context_decode_handoff_retry_min_free_blocks = (
                                    continuation_retry_blocks
                                )
                            else:
                                virtual_context_decode_handoff_retry_min_free_blocks = min(
                                    virtual_context_decode_handoff_retry_min_free_blocks,
                                    continuation_retry_blocks,
                                )
                            self._debug_record_request_defer(
                                request,
                                "decode_continuation_growth_wait",
                            )
                            deferred_request_ids.add(request.request_id)
                            new_blocks = None
                            break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    protected_running_reasons = {
                        candidate.request_id: protection_reason
                        for candidate in self.running
                        if (
                            protection_reason
                            := self._pd_preemption_protection_reason(candidate)
                        )
                    }
                    protected_running = [
                        candidate
                        for candidate in self.running
                        if candidate.request_id in protected_running_reasons
                    ]
                    if protected_running:
                        self._pd_preemption_future_protection_count += len(
                            protected_running
                        )
                        self._pd_preemption_future_protection_last_request_ids = (
                            ",".join(
                                sorted(
                                    candidate.request_id
                                    for candidate in protected_running
                                )
                            )
                        )
                    live_decode_protected_request_ids = sorted(
                        request_id
                        for request_id, reason in (
                            protected_running_reasons.items()
                        )
                        if reason == "live_decode_residency"
                    )
                    if live_decode_protected_request_ids:
                        self._pd_live_decode_preemption_protection_count += len(
                            live_decode_protected_request_ids
                        )
                        last_ids = ",".join(
                            live_decode_protected_request_ids
                        )
                        (
                            self._pd_live_decode_preemption_protection_last_request_ids
                        ) = last_ids
                    preemptible_running = [
                        candidate
                        for candidate in self.running
                        if candidate.request_id not in protected_running_reasons
                    ]
                    if not preemptible_running:
                        if (
                            protected_running_reasons.get(request.request_id)
                            == "live_decode_residency"
                        ):
                            # No safe request can release capacity for this
                            # Decode growth boundary. Publish an exact
                            # free-block event instead of resetting Decode or
                            # repeatedly rescanning the unchanged allocator.
                            current_free_blocks = (
                                self.kv_cache_manager.block_pool.get_num_free_blocks()
                            )
                            retry_blocks = current_free_blocks + 1
                            virtual_context_decode_handoff_reload_blocked_count += (
                                1
                            )
                            virtual_context_decode_handoff_reload_reasons.add(
                                "live_decode_growth_wait"
                            )
                            if (
                                virtual_context_decode_handoff_retry_min_free_blocks
                                <= 0
                            ):
                                virtual_context_decode_handoff_retry_min_free_blocks = (
                                    retry_blocks
                                )
                            else:
                                virtual_context_decode_handoff_retry_min_free_blocks = min(
                                    virtual_context_decode_handoff_retry_min_free_blocks,
                                    retry_blocks,
                                )
                            self._debug_record_request_defer(
                                request,
                                "live_decode_growth_wait",
                            )
                            deferred_request_ids.add(request.request_id)
                        new_blocks = None
                        break
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            preemptible_running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id)
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            scheduled_spec_decode_tokens.pop(
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = (
                                scheduled_encoder_inputs.pop(
                                    preempted_req.request_id, None
                                )
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the
                                # preempted request had encoder inputs
                                # scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = preemptible_running[-1]
                        self.running.remove(preempted_req)

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if deferred_by_virtual_context:
                continue

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if (
                request.request_id
                in virtual_context_decode_handoff_reload_full_block_table_req_ids
            ):
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
            else:
                req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if self._prefill_needs_capacity_floor_snapshot(
                request,
                num_new_tokens,
            ):
                virtual_context_pressure_snapshot_request_ids.add(
                    request.request_id
                )
                virtual_context_capacity_floor_snapshot_request_ids.add(
                    request.request_id
                )
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)

        schedule_running_scan_duration_s += (
            time.perf_counter() - running_scan_start
        )
        token_budget_after_running = token_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if (
            self.pd_decoupled_no_mix
            and pd_decoupled_selected_phase == "idle"
        ):
            if self.waiting:
                waiting_break_defer_reason = "policy"
        elif (
            self.pd_decoupled_no_mix
            and pd_decoupled_selected_phase == "decode"
            and not any(
                self._virtual_context_waiting_decode_ready(request)
                for request in self.waiting
            )
        ):
            if self.waiting:
                waiting_break_defer_reason = "policy"
        elif running_prefill_kv_waited:
            if self.waiting:
                waiting_break_defer_reason = "running_prefill_kv_wait"
        elif not preempted_reqs:
            while self.waiting and token_budget > 0:
                # RUNNING and WAITING consume one shared per-output row budget.
                # Do not allocate/reload/reserve a row and trim it afterwards.
                if (isolation_phase_request_limit > 0
                        and len(num_scheduled_tokens) >= isolation_phase_request_limit):
                    isolation_phase_request_limit_hit = True
                    waiting_break_defer_reason = "isolation_phase_request_limit"
                    break
                if len(self.running) == self.max_num_running_reqs:
                    waiting_break_defer_reason = "max_num_seqs"
                    break

                request = self.waiting.peek_request()
                waiting_decode_ready = (
                    self._virtual_context_waiting_decode_ready(request)
                )
                if request.request_id in isolation_waiting_decode_reclaim_wait_ids:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    prefill_defer_reason_counts["scheduled_prefill_fit_d2h_wait"] += 1
                    self._debug_record_request_defer(request, "scheduled_prefill_fit_d2h_wait")
                    deferred_request_ids.add(request.request_id)
                    continue
                if (
                    ttft_prefill_policy_active_this_iteration
                    and not waiting_decode_ready
                    and not self._ttft_prefill_request_is_selected(
                        request,
                        ttft_prefill_policy_selected_request_id_set,
                    )
                ):
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    ttft_prefill_policy_skipped_waiting_reqs += 1
                    prefill_defer_reason_counts[
                        "ttft_prefill_chunk_policy"
                    ] += 1
                    self._debug_record_request_defer(
                        request, "ttft_prefill_chunk_policy"
                    )
                    deferred_request_ids.add(request.request_id)
                    continue
                if (
                    scheduled_prefill_fit_preparation_waiting
                    and request.request_id
                    == scheduled_prefill_fit_target_request_id
                    and not waiting_decode_ready
                ):
                    waiting_break_defer_reason = (
                        "scheduled_prefill_fit_d2h_wait"
                    )
                    deferred_request_ids.add(request.request_id)
                    break
                if self.pd_decoupled_no_mix:
                    if (
                        getattr(self, "isolation_phase_waiting_scan", False)
                        and self._pd_forced_schedule_phase
                        == pd_decoupled_selected_phase
                        and (
                            (
                                pd_decoupled_selected_phase == "decode"
                                and not waiting_decode_ready
                            )
                            or (
                                pd_decoupled_selected_phase == "prefill"
                                and waiting_decode_ready
                            )
                        )
                    ):
                        # The shared queue contains both unfinished Prefills
                        # and sampled DRAM Decode continuations. Phase-slot
                        # sorting deliberately leaves their cross-phase
                        # positions unchanged. Breaking here can make a
                        # protected Decode capacity claim unreachable and
                        # leave both lanes idle. Skip only the other phase;
                        # all same-phase admission checks below still run.
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        isolation_phase_waiting_skip_count += 1
                        isolation_phase_waiting_skipped_ids.append(
                            request.request_id
                        )
                        continue
                    if (
                        pd_decoupled_selected_phase == "decode"
                        and not waiting_decode_ready
                    ):
                        waiting_break_defer_reason = "decode_lane_prefill_wait"
                        break
                    if (
                        pd_decoupled_selected_phase == "prefill"
                        and waiting_decode_ready
                    ):
                        waiting_break_defer_reason = (
                            "prefill_lane_decode_reload_wait"
                        )
                        break
                if (
                    active_decode_during_prefill_phase
                    and self.virtual_context_enabled
                ):
                    (
                        ring_feasible,
                        reload_required_pages,
                        force_all,
                    ) = self._prefill_ring_feasible_with_active_decode(
                        request
                    )
                    if not ring_feasible:
                        (
                            native_safe,
                            _reload_required_pages,
                            _free_blocks,
                            _threshold_blocks,
                            _bootstrap_blocks,
                            _ring_blocks,
                            native_safe_reason,
                        ) = self._prefill_native_safe_before_ring_pressure(
                            request
                        )
                        if native_safe:
                            virtual_context_residency_pressure_release_reasons.add(
                                native_safe_reason
                            )
                        elif not self._prefill_bootstrap_feasible_with_active_decode(
                            request
                        ):
                            virtual_context_active_decode_native_prefill_blocked_count += (
                                1
                            )
                            waiting_break_defer_reason = "policy"
                            break
                    virtual_context_active_decode_ring_prefill_allowed_count += 1
                    virtual_context_active_decode_ring_prefill_reload_pages += (
                        reload_required_pages
                    )
                    virtual_context_active_decode_ring_prefill_force_all_count += (
                        force_all
                    )
                if (
                    self.pd_request_fcfs
                    and not ttft_prefill_policy_active_this_iteration
                    and self._has_older_active_prefill_request(request)
                ):
                    if (
                        self.pd_decoupled_no_mix
                        and getattr(
                            self, "isolation_waiting_decode_fcfs_scope", False
                        )
                        and self._pd_forced_schedule_phase == "decode"
                        and pd_decoupled_selected_phase == "decode"
                        and waiting_decode_ready
                    ):
                        # This row already emitted an output before its KV
                        # moved to DRAM. Prefill ordering cannot stop its
                        # Decode service. This is NOT admission: the separate
                        # resume-owner guard immediately below, plus all KV
                        # and Native reload/ACK checks, still apply.
                        isolation_waiting_decode_prefill_fcfs_bypass_count += 1
                        if not isolation_waiting_decode_prefill_fcfs_first_id:
                            isolation_waiting_decode_prefill_fcfs_first_id = (
                                request.request_id
                            )
                    else:
                        waiting_break_defer_reason = "fcfs_request_order"
                        break

                virtual_context_resume_blocked = (
                    request.status == RequestStatus.PREEMPTED
                    and bool(request.virtual_context_deferred_state)
                    and (
                        request.request_id in virtual_context_deferred_req_ids
                        or request.request_id in self.prev_step_scheduled_req_ids
                        or request.request_id in self._pd_pending_request_ids
                    )
                )
                if virtual_context_resume_blocked:
                    if waiting_decode_ready:
                        isolation_waiting_decode_resume_owner_block_count += 1
                    if (
                        self.pd_request_fcfs
                        and not ttft_prefill_policy_active_this_iteration
                    ):
                        waiting_break_defer_reason = "fcfs_request_order"
                        break

                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    prefill_defer_reason_counts[
                        "virtual_context_resume_guard"
                    ] += 1
                    self._debug_record_request_defer(
                        request, "virtual_context_resume_guard"
                    )
                    deferred_request_ids.add(request.request_id)
                    virtual_context_resume_guard_count += 1
                    virtual_context_resume_guard_request_ids.add(
                        request.request_id
                    )
                    continue

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if (
                            request.num_preemptions
                            or request.virtual_context_deferred_logical_computed_tokens
                            > 0
                        ):
                            # We must be loading for a resumed preemption
                            # or residency-deferred prefill rather than a new
                            # request.
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        prefill_defer_reason_counts["policy"] += 1
                        self._debug_record_request_defer(request, "policy")
                        deferred_request_ids.add(request.request_id)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        prefill_defer_reason_counts["policy"] += 1
                        self._debug_record_request_defer(request, "policy")
                        deferred_request_ids.add(request.request_id)
                        continue

                if (
                    request.status == RequestStatus.PREEMPTED
                    and request.virtual_context_deferred_logical_computed_tokens
                    > 0
                ):
                    recompute_tokens = (
                        request.virtual_context_deferred_logical_computed_tokens
                    )
                    (
                        did_recompute,
                        reload_pages,
                        min_free_blocks,
                        free_blocks,
                    ) = self._prepare_waiting_virtual_context_hbm_relief_recompute(
                        request
                    )
                    if did_recompute:
                        virtual_context_hbm_relief_recompute_count += 1
                        virtual_context_hbm_relief_recompute_tokens += (
                            recompute_tokens
                        )
                        virtual_context_hbm_relief_recompute_reload_pages += (
                            reload_pages
                        )
                        virtual_context_hbm_relief_recompute_min_free_blocks = max(
                            virtual_context_hbm_relief_recompute_min_free_blocks,
                            min_free_blocks,
                        )
                        virtual_context_hbm_relief_recompute_free_blocks_max = max(
                            virtual_context_hbm_relief_recompute_free_blocks_max,
                            free_blocks,
                        )
                        virtual_context_hbm_relief_recompute_request_ids.add(
                            request.request_id
                        )

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    prefill_defer_reason_counts["policy"] += 1
                    self._debug_record_request_defer(request, "policy")
                    deferred_request_ids.add(request.request_id)
                    continue

                if (self.isolation_hbm_guard is not None
                        and not self.isolation_hbm_guard.can_admit(request)):
                    waiting_break_defer_reason = "kv_capacity"
                    prefill_defer_reason_counts["kv_capacity"] += 1
                    pd_native_active_decode_prefill_capacity_blocked = True
                    pd_native_active_decode_prefill_capacity_check_count += 1
                    pd_native_active_decode_prefill_capacity_defer_count += 1
                    pd_native_active_decode_prefill_capacity_reason = "hbm_completion_guard"
                    pd_native_prefill_retry_min_blocks = self.isolation_hbm_guard.last_required_blocks
                    pd_native_active_decode_prefill_required_blocks = self.isolation_hbm_guard.last_required_blocks
                    pd_native_active_decode_prefill_free_blocks = self.isolation_hbm_guard.last_free_blocks
                    break

                num_external_computed_tokens = 0
                num_virtual_external_computed_tokens = 0
                resuming_with_tiered_virtual_context = False
                resumed_virtual_blocks = 0
                load_kv_async = False

                # A RUNNING candidate can have prepared reload destinations,
                # then been deferred/preempted before positive admission. A
                # fresh WAITING choice must not inherit that tentative mode.
                tiered_explicit_execution_plans.pop(request.request_id, None)
                tiered_native_reload_planned_tokens.pop(request.request_id, None)
                virtual_context_native_prefill_promotion_request_ids.discard(
                    request.request_id
                )
                virtual_context_reload_to_hbm_request_ids.discard(request.request_id)
                virtual_context_decode_handoff_reload_request_ids.discard(
                    request.request_id
                )
                virtual_context_decode_handoff_reload_full_block_table_req_ids.discard(
                    request.request_id
                )

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    deferred_load_target_tokens = (
                        request.virtual_context_deferred_logical_computed_tokens
                    )
                    retained_virtual_block_table = (
                        deferred_load_target_tokens > 0
                        and self.kv_cache_manager.has_request_block_table(
                            request.request_id
                        )
                    )
                    if retained_virtual_block_table:
                        # Page-granular victimization keeps non-victim HBM
                        # pages and DRAM/null placeholders in one allocator
                        # table. Rehydrate that exact table instead of adding
                        # prefix-cache hits as if this were a new request.
                        new_computed_blocks = (
                            self.kv_cache_manager.empty_kv_cache_blocks
                        )
                        num_new_local_computed_tokens = 0
                        if self.kv_cache_manager.is_request_partial_residency(
                            request.request_id
                        ):
                            resumed_resident_blocks, resumed_virtual_blocks = (
                                self.kv_cache_manager.resume_partial_residency(
                                    request.request_id, deferred_load_target_tokens,
                                )
                            )
                        else:
                            # A prior declined admission can have installed
                            # concrete reload destinations and cleared the
                            # partial marker. They are not prefix-cache hits:
                            # reuse this exact table and keep DRAM authority.
                            resumed_resident_blocks, resumed_virtual_blocks = (
                                self.kv_cache_manager.get_retained_residency_counts(
                                    request.request_id, deferred_load_target_tokens,
                                )
                            )
                        scheduled_prefill_fit_allocator_resume_count += 1
                        scheduled_prefill_fit_allocator_resume_resident_blocks += (
                            resumed_resident_blocks
                        )
                        scheduled_prefill_fit_allocator_resume_virtual_blocks += (
                            resumed_virtual_blocks
                        )
                        scheduled_prefill_fit_allocator_resume_request_ids.add(
                            request.request_id
                        )
                    else:
                        # Get locally-cached tokens.
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                        ) = self.kv_cache_manager.get_computed_blocks(request)

                    # Get externally-cached tokens if using a KVConnector.
                    if (
                        self._can_resume_with_tiered_virtual_context(
                            request,
                            deferred_load_target_tokens,
                        )
                    ):
                        ext_tokens = max(
                            0,
                            deferred_load_target_tokens
                            - num_new_local_computed_tokens,
                        )
                        resuming_with_tiered_virtual_context = True
                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens
                        (
                            reload_to_hbm,
                            reload_pages,
                            reload_required_blocks,
                            reload_min_free_blocks,
                        ) = self._should_reload_virtual_context_to_hbm(
                            request,
                            ext_tokens,
                            self.kv_cache_manager.block_pool.get_num_free_blocks(),
                        )
                        if reload_pages > 0:
                            virtual_context_reload_to_hbm_candidate_count += 1
                            virtual_context_reload_to_hbm_pages += reload_pages
                            virtual_context_reload_to_hbm_required_blocks = max(
                                virtual_context_reload_to_hbm_required_blocks,
                                reload_required_blocks,
                            )
                            virtual_context_reload_to_hbm_min_free_blocks = max(
                                virtual_context_reload_to_hbm_min_free_blocks,
                                reload_min_free_blocks,
                            )
                        if reload_to_hbm:
                            if retained_virtual_block_table:
                                # Parked partial-residency compatibility path:
                                # replace retained null slots in-place. A
                                # whole-request victim has no retained table;
                                # ordinary allocate_slots() below recreates its
                                # complete HBM table, as in the legacy path.
                                reload_pages_by_group: dict[int, set[int]] = (
                                    defaultdict(set)
                                )
                                kv_cache_manager = self.kv_cache_manager
                                reload_plan = (
                                    kv_cache_manager.plan_virtual_context_hbm_reload_pages(
                                        request_ids={request.request_id}
                                    )
                                )
                                for reload_page in reload_plan:
                                    reload_pages_by_group[
                                        reload_page.kv_cache_group_id
                                    ].add(reload_page.logical_token_page)
                                if not self._native_reload_destinations_compatible(
                                    request.request_id, reload_pages_by_group,
                                ):
                                    raise TieredAllocatorInvariantError(
                                        "WAITING Native reload has unsafe destination ownership: "
                                        f"request={request.request_id}"
                                    )
                                reloaded_blocks = (
                                    kv_cache_manager.allocate_virtual_context_hbm_reload_blocks(
                                        request.request_id,
                                        reload_pages_by_group,
                                    )
                                )
                                if reloaded_blocks is None:
                                    self.waiting.pop_request()
                                    skipped_waiting_requests.prepend_request(
                                        request
                                    )
                                    prefill_defer_reason_counts[
                                        "hbm_reload_allocation_failed"
                                    ] += 1
                                    self._debug_record_request_defer(
                                        request,
                                        "hbm_reload_allocation_failed",
                                    )
                                    deferred_request_ids.add(request.request_id)
                                    continue
                                (
                                    _resumed_resident_blocks,
                                    remaining_virtual_blocks,
                                ) = self.kv_cache_manager.get_retained_residency_counts(
                                    request.request_id,
                                    deferred_load_target_tokens,
                                )
                                if remaining_virtual_blocks > 0:
                                    raise RuntimeError(
                                        "Prefill HBM reload left a mixed "
                                        "HBM/null block table on the native "
                                        f"path: request={request.request_id} "
                                        "remaining_virtual_blocks="
                                        f"{remaining_virtual_blocks}"
                                    )
                            virtual_context_reload_to_hbm_request_ids.add(
                                request.request_id
                            )
                            virtual_context_native_prefill_promotion_request_ids.add(
                                request.request_id
                            )
                            tiered_explicit_execution_plans[request.request_id] = (
                                TIERED_EXECUTION_NATIVE_RELOAD, reload_pages,
                            )
                            tiered_native_reload_planned_tokens[request.request_id] = (
                                ext_tokens
                            )
                            virtual_context_reload_to_hbm_free_blocks_max = max(
                                virtual_context_reload_to_hbm_free_blocks_max,
                                self.kv_cache_manager.block_pool.get_num_free_blocks(),
                            )
                        else:
                            if self._virtual_context_waiting_decode_ready(
                                request
                            ):
                                self.waiting.pop_request()
                                skipped_waiting_requests.prepend_request(
                                    request
                                )
                                prefill_defer_reason_counts[
                                    "decode_hbm_reload_wait"
                                ] += 1
                                self._debug_record_request_defer(
                                    request,
                                    "decode_hbm_reload_wait",
                                )
                                deferred_request_ids.add(request.request_id)
                                continue
                            if resumed_virtual_blocks > 0 and not (
                                _read_bool_env(
                                    "VLLM_VIRTUAL_CONTEXT_RING_BACKEND"
                                )
                                and _read_bool_env(
                                    "VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"
                                )
                            ):
                                self.waiting.pop_request()
                                skipped_waiting_requests.prepend_request(
                                    request
                                )
                                prefill_defer_reason_counts[
                                    "mixed_residency_backend_unavailable"
                                ] += 1
                                self._debug_record_request_defer(
                                    request,
                                    "mixed_residency_backend_unavailable",
                                )
                                deferred_request_ids.add(request.request_id)
                                continue
                            num_virtual_external_computed_tokens = ext_tokens
                            request.virtual_context_tiered_provider_virtual_external_tokens += (
                                ext_tokens
                            )
                    elif self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        if (
                            deferred_load_target_tokens > 0
                            and ext_tokens < deferred_load_target_tokens
                        ):
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            prefill_defer_reason_counts["policy"] += 1
                            self._debug_record_request_defer(request, "policy")
                            self._debug_record_virtual_context_defer_fallback(
                                request, "deferred_load_not_ready"
                            )
                            deferred_request_ids.add(request.request_id)
                            prefill_defer_residency_fallback_count += 1
                            virtual_context_defer_fallback_reasons.add(
                                "deferred_load_not_ready"
                            )
                            continue
                        if deferred_load_target_tokens > 0:
                            ext_tokens = deferred_load_target_tokens
                            load_kv_async = True
                            prefill_hbm_reloaded_tokens += ext_tokens

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    elif (
                        deferred_load_target_tokens > 0
                    ):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        prefill_defer_reason_counts["policy"] += 1
                        self._debug_record_request_defer(request, "policy")
                        self._debug_record_virtual_context_defer_fallback(
                            request, "connector_missing_on_resume"
                        )
                        deferred_request_ids.add(request.request_id)
                        prefill_defer_residency_fallback_count += 1
                        virtual_context_defer_fallback_reasons.add(
                            "connector_missing_on_resume"
                        )
                        continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        waiting_break_defer_reason = "max_num_batched_tokens"
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    prefill_execution_mode = (
                        self._ttft_prefill_execution_mode(
                            request,
                            virtual_context_native_prefill_promotion_request_ids,
                        )
                        if self.ttft_prefill_native_chunk_max_tokens > 0
                        else ""
                    )
                    if ttft_prefill_policy_active_this_iteration:
                        num_new_tokens = self._ttft_cap_selected_prefill_chunk(
                            request,
                            num_new_tokens,
                            num_computed_tokens=num_computed_tokens,
                            execution_mode=prefill_execution_mode,
                        )
                    assert num_new_tokens > 0
                    (
                        prefill_ring_allowed,
                        num_new_tokens,
                        prefill_ring_decode_pages,
                        prefill_ring_leftover_tokens,
                    ) = (
                        self._check_virtual_context_prefill_ring_bandwidth_admission(
                            request,
                            num_new_tokens,
                        )
                    )
                    if prefill_ring_decode_pages > 0:
                        virtual_context_prefill_ring_bandwidth_decode_reload_pages = max(
                            virtual_context_prefill_ring_bandwidth_decode_reload_pages,
                            prefill_ring_decode_pages,
                        )
                        virtual_context_prefill_ring_bandwidth_leftover_tokens = max(
                            virtual_context_prefill_ring_bandwidth_leftover_tokens,
                            prefill_ring_leftover_tokens,
                        )
                        if prefill_ring_allowed and num_new_tokens > 0:
                            (
                                virtual_context_prefill_ring_with_decode_cold_allowed_count
                            ) += 1
                            (
                                virtual_context_prefill_ring_with_decode_cold_allowed_tokens
                            ) += num_new_tokens
                    if not prefill_ring_allowed:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        prefill_defer_reason_counts[
                            "prefill_ring_bandwidth_decode_reserved"
                        ] += 1
                        self._debug_record_request_defer(
                            request,
                            "prefill_ring_bandwidth_decode_reserved",
                        )
                        deferred_request_ids.add(request.request_id)
                        virtual_context_prefill_ring_bandwidth_deferred_count += 1
                        continue

                    uncapped_num_new_tokens = num_new_tokens
                    num_new_tokens = (
                        self._cap_virtual_context_ring_prefill_admission_tokens(
                            request,
                            num_new_tokens,
                            execution_mode=prefill_execution_mode,
                        )
                    )
                    if num_new_tokens < uncapped_num_new_tokens:
                        ring_prefill_admission_cap_hit_count += 1
                        ring_prefill_admission_cap_waiting_hit_count += 1
                        ring_prefill_admission_original_tokens_sum += (
                            uncapped_num_new_tokens
                        )
                        ring_prefill_admission_capped_tokens_sum += (
                            num_new_tokens
                        )

                    if (
                        num_new_tokens == 0
                        and uncapped_num_new_tokens > 0
                        and self._virtual_context_ring_bw_last_reason
                        == "below_min_progress_defer"
                    ):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        ring_prefill_admission_min_progress_defer_count += 1
                        prefill_defer_reason_counts[
                            "ring_chunk_below_min_progress"
                        ] += 1
                        self._debug_record_request_defer(
                            request,
                            "ring_chunk_below_min_progress",
                        )
                        deferred_request_ids.add(request.request_id)
                        continue

                    if self.pd_decoupled_no_mix:
                        num_new_tokens = self._cap_pd_phase_tokens(
                            request,
                            num_new_tokens,
                            num_computed_tokens=num_computed_tokens,
                            selected_phase=pd_decoupled_selected_phase,
                        )
                        if num_new_tokens <= 0:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            prefill_defer_reason_counts[
                                "pd_phase_boundary"
                            ] += 1
                            self._debug_record_request_defer(
                                request,
                                "pd_phase_boundary",
                            )
                            deferred_request_ids.add(request.request_id)
                            continue

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            waiting_break_defer_reason = "policy"
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder and request.has_encoder_inputs
                    else 0
                )

                if (
                    num_new_tokens > 0
                    and active_decode_during_prefill_phase
                    and not waiting_decode_ready
                    and not self.virtual_context_enabled
                    and self.pd_muxwise_dynamic_policy
                ):
                    (
                        native_capacity_feasible,
                        native_chunk_required_blocks,
                        native_decode_growth_reserve_blocks,
                        native_free_blocks,
                        native_capacity_reason,
                    ) = self._native_prefill_capacity_feasible_with_active_decode(
                        request,
                        num_new_tokens,
                        num_new_computed_tokens=(
                            num_new_local_computed_tokens
                        ),
                        new_computed_blocks=new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        num_external_computed_tokens=(
                            num_external_computed_tokens
                        ),
                        num_virtual_external_computed_tokens=(
                            num_virtual_external_computed_tokens
                        ),
                        num_encoder_tokens=num_encoder_tokens,
                    )
                    native_required_blocks = (
                        native_chunk_required_blocks
                        + native_decode_growth_reserve_blocks
                    )
                    pd_native_active_decode_prefill_capacity_check_count += 1
                    pd_native_active_decode_prefill_required_blocks = max(
                        pd_native_active_decode_prefill_required_blocks,
                        native_required_blocks,
                    )
                    pd_native_active_decode_prefill_chunk_required_blocks = max(
                        pd_native_active_decode_prefill_chunk_required_blocks,
                        native_chunk_required_blocks,
                    )
                    pd_native_active_decode_growth_reserve_blocks = max(
                        pd_native_active_decode_growth_reserve_blocks,
                        native_decode_growth_reserve_blocks,
                    )
                    pd_native_active_decode_prefill_free_blocks = (
                        native_free_blocks
                    )
                    pd_native_active_decode_prefill_capacity_reason = (
                        native_capacity_reason
                    )
                    if not native_capacity_feasible:
                        pd_native_active_decode_prefill_capacity_defer_count += 1
                        pd_native_active_decode_prefill_capacity_blocked = True
                        if (
                            pd_native_prefill_retry_min_blocks
                            <= 0
                        ):
                            pd_native_prefill_retry_min_blocks = (
                                native_required_blocks
                            )
                        else:
                            pd_native_prefill_retry_min_blocks = min(
                                pd_native_prefill_retry_min_blocks,
                                native_required_blocks,
                            )
                        prefill_defer_reason_counts["kv_capacity"] += 1
                        self._debug_record_request_defer(
                            request,
                            "kv_capacity",
                        )
                        deferred_request_ids.add(request.request_id)
                        waiting_break_defer_reason = "kv_capacity"
                        break

                (
                    residency_admission_allowed,
                    residency_required_blocks,
                    _residency_free_blocks,
                    _residency_decode_reserve_blocks,
                    _residency_staging_blocks,
                    residency_raw_required_blocks,
                    residency_decode_used_blocks,
                    residency_ring_applied,
                ) = self._check_virtual_context_prefill_residency_admission(
                    request,
                    num_new_tokens,
                    "waiting_prefill",
                    scheduled_tokens=num_scheduled_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    num_virtual_external_computed_tokens=(
                        num_virtual_external_computed_tokens
                    ),
                    num_encoder_tokens=num_encoder_tokens,
                )
                residency_admission_raw_required_blocks_max = max(
                    residency_admission_raw_required_blocks_max,
                    residency_raw_required_blocks,
                )
                ring_residency_admission_applied_count += (
                    residency_ring_applied
                )
                if (
                    residency_ring_applied
                    and _residency_decode_reserve_blocks
                    < residency_decode_used_blocks
                ):
                    ring_residency_decode_reserve_relaxed_count += 1
                if not residency_admission_allowed:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    prefill_defer_reason_counts["residency_budget"] += 1
                    self._debug_record_request_defer(
                        request, "residency_budget"
                    )
                    deferred_request_ids.add(request.request_id)
                    prefill_deferred_by_residency_admission += 1
                    residency_admission_deferred_request_ids.add(
                        request.request_id
                    )
                    residency_admission_required_blocks_sum += (
                        residency_required_blocks
                    )
                    residency_admission_required_blocks_max = max(
                        residency_admission_required_blocks_max,
                        residency_required_blocks,
                    )
                    continue

                allocate_slots_start = time.perf_counter()
                try:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        num_external_computed_tokens=num_external_computed_tokens,
                        num_virtual_external_computed_tokens=(
                            num_virtual_external_computed_tokens
                        ),
                        delay_cache_blocks=load_kv_async,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                finally:
                    schedule_allocate_slots_duration_s += (
                        time.perf_counter() - allocate_slots_start
                    )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if (
                        active_decode_during_prefill_phase
                        and not waiting_decode_ready
                        and not self.virtual_context_enabled
                        and self.pd_muxwise_dynamic_policy
                    ):
                        pd_native_active_decode_prefill_capacity_blocked = True
                    waiting_break_defer_reason = "kv_capacity"
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request.request_id),
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    prefill_defer_reason_counts["policy"] += 1
                    self._debug_record_request_defer(request, "policy")
                    deferred_request_ids.add(request.request_id)
                    continue

                self._update_connector_prefix_cache_stats(request)

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    if self._needs_virtual_context_worker_rehydrate(request):
                        scheduled_new_reqs.append(request)
                        scheduled_residency_rehydrated_req_ids.add(
                            request.request_id
                        )
                    else:
                        scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                if self._prefill_needs_capacity_floor_snapshot(
                    request,
                    num_new_tokens,
                ):
                    virtual_context_pressure_snapshot_request_ids.add(
                        request.request_id
                    )
                    virtual_context_capacity_floor_snapshot_request_ids.add(
                        request.request_id
                )
                request.status = RequestStatus.RUNNING
                if (
                    request.virtual_context_deferred_logical_computed_tokens
                    > 0
                ):
                    request.resume_from_external_residency(
                        logical_computed_tokens=num_computed_tokens
                    )
                else:
                    request.num_computed_tokens = num_computed_tokens
                if request.virtual_context_deferred_state == "reloaded":
                    prefill_resume_without_recompute += 1
                    request.virtual_context_deferred_state = "resumed"
                elif (
                    request.request_id in virtual_context_reload_to_hbm_request_ids
                ):
                    prefill_resume_without_recompute += 1
                    request.virtual_context_resume_without_recompute_count += 1
                    request.virtual_context_tiered_provider_resume_count += 1
                    # The final admitted set below publishes pending + Native
                    # authority. Only the matching worker's all-layer ACK may
                    # complete it in update_from_output().
                elif (
                    resuming_with_tiered_virtual_context
                    or request.virtual_context_deferred_state
                    in _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES
                ):
                    prefill_resume_without_recompute += 1
                    request.virtual_context_resume_without_recompute_count += 1
                    request.virtual_context_tiered_provider_resume_count += 1
                    request.virtual_context_deferred_state = (
                        "tiered_provider_resumed"
                    )
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
        elif self.waiting:
            waiting_break_defer_reason = "preemption"
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)
        if self.waiting and waiting_break_defer_reason is not None:
            self._debug_record_waiting_defer(
                waiting_break_defer_reason,
                prefill_defer_reason_counts,
                deferred_request_ids,
            )

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        if (
            isolation_waiting_decode_prefill_fcfs_bypass_count > 0
            and not self._isolation_waiting_decode_fcfs_scope_logged
        ):
            # Includes zero-token passes, at most once per Scheduler. Passing
            # this ordering check is not an allocation or copy-completion ACK.
            self._isolation_waiting_decode_fcfs_scope_logged = True
            logger.info(
                "Isolation WAITING Decode Prefill-FCFS bypass: count=%d "
                "first_request=%s resume_owner_blocks=%d "
                "admitted_tokens=%d stop=%s",
                isolation_waiting_decode_prefill_fcfs_bypass_count,
                isolation_waiting_decode_prefill_fcfs_first_id,
                isolation_waiting_decode_resume_owner_block_count,
                total_num_scheduled_tokens,
                waiting_break_defer_reason,
            )
        if (
            isolation_phase_waiting_skip_count > 0
            and pd_decoupled_selected_phase
            not in self._isolation_phase_waiting_scan_logged
        ):
            # At most once per phase, including zero-token admissions that
            # normally have no recovered Future/iteration trace of their own.
            self._isolation_phase_waiting_scan_logged.add(
                pd_decoupled_selected_phase
            )
            logger.info(
                "Isolation phase WAITING scan: phase=%s skipped=%d "
                "first_skipped=%s admitted_tokens=%d stop=%s",
                pd_decoupled_selected_phase, isolation_phase_waiting_skip_count,
                ",".join(isolation_phase_waiting_skipped_ids[:8]),
                total_num_scheduled_tokens, waiting_break_defer_reason,
            )
        admitted_native_reload_ids = (
            self._commit_virtual_context_native_reload_admissions(
                num_scheduled_tokens,
                tiered_explicit_execution_plans,
                tiered_native_reload_planned_tokens,
            )
        )
        virtual_context_native_prefill_promotion_request_ids.intersection_update(
            admitted_native_reload_ids
        )
        virtual_context_reload_to_hbm_request_ids.intersection_update(
            admitted_native_reload_ids
        )
        virtual_context_decode_handoff_reload_request_ids.intersection_update(
            num_scheduled_tokens
        )
        virtual_context_decode_handoff_reload_scheduled_count = len(
            virtual_context_decode_handoff_reload_request_ids
        )
        virtual_context_reload_to_hbm_scheduled_count = len(
            virtual_context_reload_to_hbm_request_ids
        )
        virtual_context_reload_to_hbm_tokens = sum(
            tiered_native_reload_planned_tokens[req_id]
            for req_id in virtual_context_reload_to_hbm_request_ids
        )
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        if isolation_phase_request_limit > 0:
            assert len(num_scheduled_tokens) <= isolation_phase_request_limit
        classified_scheduled_phase = self._classify_scheduled_phase(
            num_scheduled_tokens
        )
        pd_phase_classification_corrected = 0
        if (
            self.pd_decoupled_no_mix
            and total_num_scheduled_tokens > 0
            and pd_decoupled_selected_phase in {"decode", "prefill"}
        ):
            # Token admission above is authoritative. The legacy classifier
            # reads Request.num_computed_tokens, which is intentionally stale
            # for a WAITING provider-backed resume until allocation commits.
            scheduled_phase = pd_decoupled_selected_phase
            pd_phase_classification_corrected = int(
                classified_scheduled_phase != scheduled_phase
            )
        else:
            scheduled_phase = classified_scheduled_phase
        self._validate_pd_phase_schedule(
            scheduled_phase,
            num_scheduled_tokens,
        )
        pd_ttft_handoff_request_ids: list[str] = []
        for req_id, scheduled_tokens in num_scheduled_tokens.items():
            request = self.requests[req_id]
            prompt_remaining = max(
                0,
                request.num_prompt_tokens - request.num_computed_tokens,
            )
            prefill_tokens = min(scheduled_tokens, prompt_remaining)
            if (
                prefill_tokens > 0
                and request.num_output_tokens == 0
                and request.num_computed_tokens + prefill_tokens
                >= request.num_prompt_tokens
            ):
                pd_ttft_handoff_request_ids.append(req_id)
        if self.enable_ttft_prefill_chunk_scheduling:
            for req_id, scheduled_tokens in num_scheduled_tokens.items():
                request = self.requests[req_id]
                prefill_tokens = min(
                    scheduled_tokens,
                    self._pd_prefill_lane_remaining_tokens(
                        request,
                    ),
                )
                if prefill_tokens > 0:
                    ttft_prefill_policy_scheduled_request_ids.append(req_id)
                    ttft_prefill_policy_scheduled_tokens += prefill_tokens
            if ttft_prefill_policy_active_this_iteration:
                assert all(
                    req_id in ttft_prefill_policy_selected_request_id_set
                    for req_id in ttft_prefill_policy_scheduled_request_ids
                )
            self.ttft_prefill_chunk_scheduler.observe_scheduled_chunks(
                ttft_prefill_policy_scheduled_request_ids
            )
        if self.enable_tbt_decode_scheduling:
            for req_id, scheduled_tokens in num_scheduled_tokens.items():
                request = self.requests[req_id]
                if not self._tbt_decode_request_is_candidate(request):
                    continue
                tbt_decode_policy_scheduled_request_ids.append(req_id)
                tbt_decode_policy_scheduled_tokens += scheduled_tokens
            self.tbt_decode_scheduler.observe_scheduled_tokens(
                tbt_decode_policy_scheduled_request_ids
            )
        decode_growth_runway = self._scheduled_prefill_fit_decode_growth_runway
        if (
            decode_growth_runway is not None
            and num_scheduled_tokens.get(
                decode_growth_runway.target_request_id,
                0,
            )
            > 0
        ):
            # The one exact target boundary has now consumed its Decode-growth
            # allowance.  Any following chunk must create and validate a new
            # proactive plan using the same capacity-fraction complement.
            self._scheduled_prefill_fit_decode_growth_runway = None
            self._scheduled_prefill_fit_decode_growth_runway_admission_count += 1
            scheduled_prefill_fit_decode_growth_runway_consumed = True

        (
            scheduled_prefill_fit_proactive_target,
            scheduled_prefill_fit_proactive_victims,
            scheduled_prefill_fit_proactive_projected_logical_tokens,
            scheduled_prefill_fit_proactive_next_chunk_tokens,
            scheduled_prefill_fit_proactive_raw_required_blocks,
            scheduled_prefill_fit_proactive_reserved_blocks,
            scheduled_prefill_fit_proactive_free_blocks,
            scheduled_prefill_fit_proactive_deficit_blocks,
            proactive_snapshot_pages,
        ) = self._scheduled_prefill_fit_proactive_snapshot_plan(
            num_scheduled_tokens,
            ring_token_cap=(
                virtual_context_ring_prefill_admission_tokens_effective
                or self._virtual_context_residency_fixed_ring_chunk_tokens()
            ),
        )
        scheduled_prefill_fit_proactive_plan_mode = getattr(
            self,
            "_scheduled_prefill_fit_last_plan_mode",
            "none",
        )
        scheduled_prefill_fit_proactive_native_deficit_blocks = getattr(
            self,
            "_scheduled_prefill_fit_last_native_deficit_blocks",
            0,
        )
        scheduled_prefill_fit_proactive_ring_fallback_deficit_blocks = getattr(
            self,
            "_scheduled_prefill_fit_last_ring_fallback_deficit_blocks",
            0,
        )
        scheduled_prefill_fit_proactive_avoided_peer_blocks = getattr(
            self,
            "_scheduled_prefill_fit_last_avoided_peer_blocks",
            0,
        )
        if scheduled_prefill_fit_proactive_target is not None:
            scheduled_prefill_fit_proactive_target_request_id = (
                scheduled_prefill_fit_proactive_target.request_id
            )
            scheduled_prefill_fit_proactive_decode_handoff = int(
                scheduled_prefill_fit_proactive_next_chunk_tokens == 0
                and scheduled_prefill_fit_proactive_raw_required_blocks > 0
            )
        if scheduled_prefill_fit_proactive_victims:
            scheduled_prefill_fit_proactive_victim_request_id = (
                scheduled_prefill_fit_proactive_victims[0].request_id
            )
            scheduled_prefill_fit_proactive_victim_request_ids = ",".join(
                victim.request_id
                for victim in scheduled_prefill_fit_proactive_victims
            )
            scheduled_request_ids = set(num_scheduled_tokens)
            scheduled_prefill_fit_proactive_external_victim_count = sum(
                victim.request_id not in scheduled_request_ids
                for victim in scheduled_prefill_fit_proactive_victims
            )
            scheduled_prefill_fit_proactive_waiting_victim_count = sum(
                victim in self.waiting
                for victim in scheduled_prefill_fit_proactive_victims
            )
            for victim in scheduled_prefill_fit_proactive_victims:
                logical_tokens = virtual_context_residency_victim_context_tokens(
                    victim.num_computed_tokens,
                    victim.virtual_context_deferred_logical_computed_tokens,
                )
                scheduled_tokens = num_scheduled_tokens.get(
                    victim.request_id,
                    0,
                )
                logical_tokens += min(
                    scheduled_tokens,
                    max(0, victim.num_prompt_tokens - logical_tokens),
                )
                scheduled_prefill_fit_proactive_snapshot_logical_tokens[
                    victim.request_id
                ] = min(victim.num_prompt_tokens, logical_tokens)
        scheduled_prefill_fit_proactive_snapshot_pages.update(
            proactive_snapshot_pages
        )
        if (
            scheduled_prefill_fit_proactive_deficit_blocks > 0
            and (
                len(proactive_snapshot_pages)
                >= scheduled_prefill_fit_proactive_deficit_blocks
                or (
                    scheduled_prefill_fit_proactive_target is not None
                    and any(
                        victim.request_id
                        == scheduled_prefill_fit_proactive_target.request_id
                        for victim in scheduled_prefill_fit_proactive_victims
                    )
                )
            )
            and scheduled_prefill_fit_proactive_target is not None
            and scheduled_prefill_fit_proactive_victims
        ):
            previous_preparation = self._scheduled_prefill_fit_preparation
            if previous_preparation is not None:
                self._scheduled_prefill_fit_preparation_cancelled_count += 1
            self._scheduled_prefill_fit_preparation = (
                ScheduledPrefillFitPreparation(
                    target_request_id=(
                        scheduled_prefill_fit_proactive_target.request_id
                    ),
                    victim_request_ids=tuple(
                        victim.request_id
                        for victim in scheduled_prefill_fit_proactive_victims
                    ),
                    projected_logical_tokens=(
                        scheduled_prefill_fit_proactive_projected_logical_tokens
                    ),
                    next_chunk_tokens=(
                        scheduled_prefill_fit_proactive_next_chunk_tokens
                    ),
                    predicted_deficit_blocks=(
                        scheduled_prefill_fit_proactive_deficit_blocks
                    ),
                    snapshot_pages=tuple(sorted(proactive_snapshot_pages)),
                    created_iteration_id=self._iteration_debug_index,
                    target_self_victim=any(
                        victim.request_id
                        == scheduled_prefill_fit_proactive_target.request_id
                        for victim in scheduled_prefill_fit_proactive_victims
                    ),
                )
            )
            self._scheduled_prefill_fit_preparation_created_count += 1
        if scheduled_prefill_fit_proactive_snapshot_pages:
            proactive_snapshot_request_ids = {
                request_id
                for request_id, _group_id, _logical_page in (
                    scheduled_prefill_fit_proactive_snapshot_pages
                )
            }
            virtual_context_pressure_snapshot_request_ids.update(
                proactive_snapshot_request_ids
            )
            virtual_context_capacity_floor_snapshot_request_ids.update(
                proactive_snapshot_request_ids
            )
        if self.pd_decoupled_no_mix and pd_decoupled_selected_phase in {
            "decode",
            "prefill",
        }:
            self._pd_phase_separated_last_phase = pd_decoupled_selected_phase
        if scheduled_phase == "decode" and total_num_scheduled_tokens > 0:
            self._virtual_context_last_decode_schedule_ts = scheduled_timestamp

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            pd_phase_handoff_token_refresh_request_ids: list[str] = []
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
                scheduled_phase,
                pd_phase_handoff_token_refresh_request_ids,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())
        self.prev_step_scheduled_phase_by_req.clear()
        if scheduled_phase in {"decode", "prefill"}:
            self.prev_step_scheduled_phase_by_req.update(
                dict.fromkeys(num_scheduled_tokens, scheduled_phase)
            )

        iteration_debug: dict[str, int | float | str] | None = None
        request_debug: dict[str, dict[str, int | float | str]] | None = None
        same_step_resumed_deferred_req_ids = (
            virtual_context_deferred_req_ids & set(num_scheduled_tokens)
        )
        residency_deferred_req_ids_for_output = (
            virtual_context_deferred_req_ids - same_step_resumed_deferred_req_ids
        )

        detailed_iteration_debug_sampled = (
            self._iteration_debug_enabled
            and self._iteration_debug_index
            % self._iteration_debug_sample_every_n
            == 0
        )
        # A zero-work admission attempt is control metadata, not a model
        # iteration. Refreshing request/page/attention debug snapshots here can
        # hold the GIL while a Decode worker is preparing its next input. Keep
        # a lightweight record and defer expensive snapshots to useful work.
        collect_iteration_debug = (
            detailed_iteration_debug_sampled
            and total_num_scheduled_tokens > 0
        )
        if collect_iteration_debug:
            virtual_context_debug_block_start = time.perf_counter()
            virtual_context_request_debug_duration_s = 0.0
            virtual_context_attention_plan_debug_duration_s = 0.0
            virtual_context_reload_guard_debug_duration_s = 0.0
            virtual_context_global_debug_duration_s = 0.0
            virtual_context_tenant_kv_debug_duration_s = 0.0
            virtual_context_debug_snapshot_due = (
                self._virtual_context_debug_snapshot_due()
            )
            virtual_context_request_debug_snapshot_hits = 0
            virtual_context_request_debug_snapshot_refreshes = 0
            virtual_context_attention_plan_debug_snapshot_hits = 0
            virtual_context_attention_plan_debug_snapshot_refreshes = 0
            native_decode_fast_path_eligible_request_count = 0
            native_decode_fast_path_saved_request_scan_count = 0
            request_debug = {}
            scheduled_req_kinds: dict[str, str] = {}
            scheduled_req_kinds.update(
                {req.request_id: "running" for req in scheduled_running_reqs}
            )
            scheduled_req_kinds.update(
                {req.request_id: "resumed" for req in scheduled_resumed_reqs}
            )
            scheduled_req_kinds.update(
                {req.request_id: "new" for req in scheduled_new_reqs}
            )
            scheduled_req_kinds.update(
                {
                    req_id: "residency_rehydrated"
                    for req_id in scheduled_residency_rehydrated_req_ids
                }
            )
            num_prefill_reqs = 0
            num_decode_reqs = 0
            num_prefill_tokens = 0
            num_decode_tokens = 0
            num_new_prefill_admitted = 0
            num_prefill_chunks_continued = 0
            num_decode_continued = 0
            token_budget_consumed_by_running_prefill = 0
            token_budget_consumed_by_running_decode = 0
            token_budget_consumed_by_new_prefill = 0
            batch_contexts: list[int] = []
            prefill_contexts: list[int] = []
            decode_contexts: list[int] = []
            scheduled_prefill_plan_req_ids: list[str] = []
            scheduled_decode_plan_req_ids: list[str] = []
            prefill_attention_work_proxy = 0.0
            decode_kv_read_token_proxy = 0.0
            request_records: dict[str, dict[str, int | float | str | bool]] = {}
            for req_id, scheduled_tokens in num_scheduled_tokens.items():
                request = self.requests[req_id]
                scheduled_kind = scheduled_req_kinds.get(req_id, "unknown")
                num_computed_tokens_before = request.num_computed_tokens
                prefill_remaining = self._pd_prefill_lane_remaining_tokens(
                    request,
                    num_computed_tokens=num_computed_tokens_before,
                )
                prefill_tokens = min(scheduled_tokens, prefill_remaining)
                decode_tokens = max(0, scheduled_tokens - prefill_tokens)
                batch_contexts.append(num_computed_tokens_before)
                if prefill_tokens > 0:
                    scheduled_prefill_plan_req_ids.append(req_id)
                    self._debug_close_pd_residency_prefill_skip(
                        request, scheduled_timestamp
                    )
                    num_prefill_reqs += 1
                    num_prefill_tokens += prefill_tokens
                    prefill_contexts.append(num_computed_tokens_before)
                    prefill_attention_work_proxy += (
                        prefill_tokens * num_computed_tokens_before
                        + prefill_tokens * (prefill_tokens + 1) / 2.0
                    )
                    if scheduled_kind == "new":
                        num_new_prefill_admitted += 1
                        token_budget_consumed_by_new_prefill += prefill_tokens
                    else:
                        num_prefill_chunks_continued += 1
                        token_budget_consumed_by_running_prefill += prefill_tokens
                if decode_tokens > 0:
                    scheduled_decode_plan_req_ids.append(req_id)
                    num_decode_reqs += 1
                    num_decode_tokens += decode_tokens
                    decode_contexts.append(num_computed_tokens_before)
                    decode_kv_read_token_proxy += (
                        decode_tokens * num_computed_tokens_before
                    )
                    if scheduled_kind != "new":
                        num_decode_continued += 1
                        token_budget_consumed_by_running_decode += decode_tokens
                if prefill_tokens > 0 and decode_tokens > 0:
                    phase = "mixed"
                elif prefill_tokens > 0:
                    phase = "prefill"
                else:
                    phase = "decode"
                if request.first_admitted_iteration_id < 0:
                    request.first_admitted_iteration_id = self._iteration_debug_index
                if prefill_tokens > 0:
                    if request.first_prefill_iteration_id < 0:
                        request.first_prefill_iteration_id = (
                            self._iteration_debug_index
                        )
                    request.last_prefill_iteration_id = self._iteration_debug_index
                    request.num_prefill_chunks += 1
                if decode_tokens > 0 and request.first_decode_iteration_id < 0:
                    request.first_decode_iteration_id = self._iteration_debug_index
                native_decode_request_fast_path = (
                    self._native_decode_control_fast_path_eligible(
                        request,
                        prefill_tokens,
                        decode_tokens,
                    )
                )
                if native_decode_request_fast_path:
                    native_decode_fast_path_eligible_request_count += 1
                    native_decode_fast_path_saved_request_scan_count += 1
                    virtual_context_request_stats = (
                        self._native_decode_request_debug_fields(request)
                    )
                else:
                    (
                        virtual_context_request_stats,
                        request_debug_snapshot_stale,
                        request_debug_duration_s,
                    ) = self._get_virtual_context_request_debug_snapshot(
                        req_id,
                        refresh=virtual_context_debug_snapshot_due,
                    )
                    virtual_context_request_debug_duration_s += (
                        request_debug_duration_s
                    )
                    if request_debug_snapshot_stale:
                        virtual_context_request_debug_snapshot_hits += 1
                    else:
                        virtual_context_request_debug_snapshot_refreshes += 1
                request_records[req_id] = {
                    "request_id": req_id,
                    "tenant_id": request.tenant_id,
                    "session_id": request.tenant_session_id,
                    "scheduled_kind": scheduled_kind,
                    "phase": phase,
                    "scheduled_tokens": scheduled_tokens,
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": decode_tokens,
                    "num_computed_tokens_before": num_computed_tokens_before,
                    "num_prompt_tokens": request.num_prompt_tokens,
                    "num_output_tokens": len(request.output_token_ids),
                    "remaining_prefill_tokens_before_chunk": prefill_remaining,
                    "pd_recompute_frontier_tokens": (
                        request.pd_recompute_frontier_tokens
                    ),
                    "prefix_hit_tokens_before_chunk": (
                        request.num_cached_tokens
                        if request.num_cached_tokens >= 0
                        else 0
                    ),
                    "recomputed_tokens_before_chunk": max(
                        0,
                        num_computed_tokens_before
                        - (
                            request.num_cached_tokens
                            if request.num_cached_tokens >= 0
                            else 0
                        ),
                    ),
                    "prefill_chunk_index": request.num_prefill_chunks,
                    "token_iteration_id": self._iteration_debug_index,
                    "token_tenant_id": request.tenant_id,
                    "token_request_context_len": num_computed_tokens_before,
                    **self._debug_request_lifecycle_fields(request),
                    **virtual_context_request_stats,
                }

            for request in preempted_reqs:
                if request.request_id in request_records:
                    continue
                request_records[request.request_id] = {
                    "request_id": request.request_id,
                    "tenant_id": request.tenant_id,
                    "session_id": request.tenant_session_id,
                    "scheduled_kind": "preempted",
                    "phase": request.last_preempt_phase or "unknown",
                    "scheduled_tokens": 0,
                    "prefill_tokens": 0,
                    "decode_tokens": 0,
                    "num_computed_tokens_before": (
                        request.last_preempt_num_computed_tokens_before_reset
                    ),
                    "num_prompt_tokens": request.num_prompt_tokens,
                    "num_output_tokens": len(request.output_token_ids),
                    "remaining_prefill_tokens_before_chunk": max(
                        0,
                        request.num_prompt_tokens
                        - request.last_preempt_num_computed_tokens_before_reset,
                    ),
                    "prefix_hit_tokens_before_chunk": (
                        request.num_cached_tokens
                        if request.num_cached_tokens >= 0
                        else 0
                    ),
                    "recomputed_tokens_before_chunk": max(
                        0,
                        request.last_preempt_num_computed_tokens_before_reset
                        - (
                            request.num_cached_tokens
                            if request.num_cached_tokens >= 0
                            else 0
                        ),
                    ),
                    "prefill_chunk_index": request.num_prefill_chunks,
                    "token_iteration_id": self._iteration_debug_index,
                    "token_tenant_id": request.tenant_id,
                    "token_request_context_len": (
                        request.last_preempt_num_computed_tokens_before_reset
                    ),
                    **self._debug_request_lifecycle_fields(request),
                }

            for req_id in pd_residency_skipped_running_prefill_request_ids:
                if req_id in request_records or req_id not in self.requests:
                    continue
                request = self.requests[req_id]
                (
                    virtual_context_request_stats,
                    request_debug_snapshot_stale,
                    request_debug_duration_s,
                ) = self._get_virtual_context_request_debug_snapshot(
                    req_id,
                    refresh=virtual_context_debug_snapshot_due,
                )
                virtual_context_request_debug_duration_s += (
                    request_debug_duration_s
                )
                if request_debug_snapshot_stale:
                    virtual_context_request_debug_snapshot_hits += 1
                else:
                    virtual_context_request_debug_snapshot_refreshes += 1
                request_records[req_id] = {
                    "request_id": req_id,
                    "tenant_id": request.tenant_id,
                    "session_id": request.tenant_session_id,
                    "scheduled_kind": "skipped_running_prefill",
                    "phase": "prefill",
                    "scheduled_tokens": 0,
                    "prefill_tokens": 0,
                    "decode_tokens": 0,
                    "num_computed_tokens_before": request.num_computed_tokens,
                    "num_prompt_tokens": request.num_prompt_tokens,
                    "num_output_tokens": len(request.output_token_ids),
                    "remaining_prefill_tokens_before_chunk": (
                        self._pd_prefill_lane_remaining_tokens(request)
                    ),
                    "prefix_hit_tokens_before_chunk": (
                        request.num_cached_tokens
                        if request.num_cached_tokens >= 0
                        else 0
                    ),
                    "recomputed_tokens_before_chunk": max(
                        0,
                        request.num_computed_tokens
                        - (
                            request.num_cached_tokens
                            if request.num_cached_tokens >= 0
                            else 0
                        ),
                    ),
                    "prefill_chunk_index": request.num_prefill_chunks,
                    "token_iteration_id": self._iteration_debug_index,
                    "token_tenant_id": request.tenant_id,
                    "token_request_context_len": request.num_computed_tokens,
                    **self._debug_request_lifecycle_fields(request),
                    **virtual_context_request_stats,
                }

            batch_context_p50 = self._debug_percentile(batch_contexts, 0.50)
            batch_context_p95 = self._debug_percentile(batch_contexts, 0.95)
            decode_context_p50 = self._debug_percentile(decode_contexts, 0.50)
            decode_context_p95 = self._debug_percentile(decode_contexts, 0.95)
            prefill_context_p95 = self._debug_percentile(prefill_contexts, 0.95)
            batch_context_max = max(batch_contexts) if batch_contexts else 0
            decode_context_max = max(decode_contexts) if decode_contexts else 0
            prefill_context_max = max(prefill_contexts) if prefill_contexts else 0
            batch_context_stats: dict[str, int | float | bool] = {
                "batch_total_context_tokens": sum(batch_contexts),
                "batch_max_context_tokens": batch_context_max,
                "batch_context_p50": batch_context_p50,
                "batch_context_p95": batch_context_p95,
                "prefill_context_before_sum": sum(prefill_contexts),
                "prefill_context_before_max": prefill_context_max,
                "prefill_context_before_p95": prefill_context_p95,
                "decode_context_sum": sum(decode_contexts),
                "decode_context_max": decode_context_max,
                "decode_context_p50": decode_context_p50,
                "decode_context_p95": decode_context_p95,
                "num_long_decode_reqs_4k": sum(
                    1 for value in decode_contexts if value >= 4096
                ),
                "num_long_decode_reqs_8k": sum(
                    1 for value in decode_contexts if value >= 8192
                ),
                "num_long_decode_reqs_16k": sum(
                    1 for value in decode_contexts if value >= 16384
                ),
                "num_long_prefill_reqs_4k": sum(
                    1 for value in prefill_contexts if value >= 4096
                ),
                "num_long_prefill_reqs_8k": sum(
                    1 for value in prefill_contexts if value >= 8192
                ),
                "num_long_prefill_reqs_16k": sum(
                    1 for value in prefill_contexts if value >= 16384
                ),
                "decode_kv_read_token_proxy": decode_kv_read_token_proxy,
                "prefill_attention_work_proxy": prefill_attention_work_proxy,
                "batch_has_long_prefill": prefill_context_max >= 8192,
                "batch_has_long_decode": decode_context_max >= 8192,
            }
            for request_record in request_records.values():
                request_record.update(
                    {
                        "token_batch_decode_context_max": decode_context_max,
                        "token_batch_prefill_tokens": num_prefill_tokens,
                        "token_batch_prefill_attention_proxy": (
                            prefill_attention_work_proxy
                        ),
                        "token_batch_decode_kv_read_proxy": (
                            decode_kv_read_token_proxy
                        ),
                        "token_batch_has_long_prefill": prefill_context_max >= 8192,
                        "token_batch_has_long_decode": decode_context_max >= 8192,
                    }
                )
            request_debug.update(request_records)

            preempted_prefill_reqs = [
                request
                for request in preempted_reqs
                if request.last_preempt_phase == "prefill"
            ]
            preempted_decode_reqs = [
                request
                for request in preempted_reqs
                if request.last_preempt_phase == "decode"
            ]
            preempted_tokens_reset_total = sum(
                request.last_preempt_num_computed_tokens_before_reset
                for request in preempted_reqs
            )
            preempted_prefill_tokens_reset_total = sum(
                request.last_preempt_num_computed_tokens_before_reset
                for request in preempted_prefill_reqs
            )
            preempted_decode_tokens_reset_total = sum(
                request.last_preempt_num_computed_tokens_before_reset
                for request in preempted_decode_reqs
            )
            pd_residency_prefill_required_blocks_sum = sum(
                pd_residency_prefill_required_blocks_by_req.values()
            )
            pd_residency_prefill_required_blocks_max = max(
                pd_residency_prefill_required_blocks_by_req.values(),
                default=0,
            )
            pd_residency_running_prefill_unscheduled_duration_s = sum(
                self.requests[req_id].pd_residency_prefill_unscheduled_duration_s
                for req_id in pd_residency_skipped_running_prefill_request_ids
                if req_id in self.requests
            )
            residency_budget_free_hbm_blocks = (
                pd_residency_last_free_blocks
                if pd_residency_skipped_running_prefill_request_ids
                else self.kv_cache_manager.block_pool.get_num_free_blocks()
            )
            residency_budget_decode_hotset_used_blocks = (
                self._debug_running_decode_resident_blocks()
            )
            residency_budget_decode_hotset_reserved_blocks = (
                max(
                    self.virtual_context_decode_hotset_reserve_blocks,
                    residency_budget_decode_hotset_used_blocks,
                )
            )
            residency_budget_prefill_required_growth_blocks = (
                max(
                    pd_residency_prefill_required_blocks_max,
                    residency_admission_required_blocks_max,
                )
            )
            residency_budget_staging_reserved_blocks = (
                self.virtual_context_staging_reserved_blocks
            )

            running_phase_counts_after = self._debug_running_phase_counts()
            num_waiting_prefill_reqs_after = self._debug_count_waiting_prefill_reqs()
            scheduled_num_seqs = len(num_scheduled_tokens)
            token_budget_consumed_by_running = (
                token_budget_consumed_by_running_prefill
                + token_budget_consumed_by_running_decode
            )
            iteration_debug = {
                "iteration_id": self._iteration_debug_index,
                "scheduled_ts": scheduled_timestamp,
                "waiting_queue_len_before": waiting_queue_len_before,
                "waiting_queue_len_after": len(self.waiting),
                "waiting_queue_len": len(self.waiting),
                "running_reqs_before": running_reqs_before,
                "running_reqs_after": len(self.running),
                "running_queue_len": len(self.running),
                "num_waiting_prefill_reqs": num_waiting_prefill_reqs_before,
                "num_waiting_prefill_reqs_before": (
                    num_waiting_prefill_reqs_before
                ),
                "num_waiting_prefill_reqs_after": (
                    num_waiting_prefill_reqs_after
                ),
                "num_running_decode_reqs": (
                    running_phase_counts_before["num_running_decode_reqs"]
                ),
                "num_running_decode_reqs_before": (
                    running_phase_counts_before["num_running_decode_reqs"]
                ),
                "num_running_decode_reqs_after": (
                    running_phase_counts_after["num_running_decode_reqs"]
                ),
                "num_running_prefill_reqs": (
                    running_phase_counts_before["num_running_prefill_reqs"]
                ),
                "num_running_prefill_reqs_before": (
                    running_phase_counts_before["num_running_prefill_reqs"]
                ),
                "num_running_prefill_reqs_after": (
                    running_phase_counts_after["num_running_prefill_reqs"]
                ),
                "scheduled_new_req_count": len(scheduled_new_reqs),
                "scheduled_running_req_count": len(scheduled_running_reqs),
                "scheduled_resumed_req_count": len(scheduled_resumed_reqs),
                "scheduled_residency_rehydrated_req_count": len(
                    scheduled_residency_rehydrated_req_ids
                ),
                "scheduled_residency_rehydrated_req_ids": ",".join(
                    sorted(scheduled_residency_rehydrated_req_ids)
                ),
                "preempted_req_count": len(preempted_reqs),
                "kv_preempted_reqs_this_iter": len(preempted_reqs),
                "preempted_prefill_req_count": len(preempted_prefill_reqs),
                "preempted_decode_req_count": len(preempted_decode_reqs),
                "preempted_tokens_reset_total": preempted_tokens_reset_total,
                "preempted_prefill_tokens_reset_total": (
                    preempted_prefill_tokens_reset_total
                ),
                "preempted_decode_tokens_reset_total": (
                    preempted_decode_tokens_reset_total
                ),
                "preempted_request_ids": ",".join(
                    request.request_id for request in preempted_reqs
                ),
                "running_kv_capacity_miss_count": (
                    running_kv_capacity_miss_count
                ),
                "running_prefill_kv_capacity_miss_count": (
                    running_prefill_kv_capacity_miss_count
                ),
                "running_decode_kv_capacity_miss_count": (
                    running_decode_kv_capacity_miss_count
                ),
                "running_kv_capacity_miss_req_count": len(
                    running_kv_capacity_miss_request_ids
                ),
                "running_kv_capacity_miss_request_ids": ",".join(
                    sorted(running_kv_capacity_miss_request_ids)
                ),
                "pd_wait_on_prefill_kv_capacity_enabled": int(
                    self.pd_wait_on_prefill_kv_capacity
                ),
                "pd_request_fcfs_enabled": int(self.pd_request_fcfs),
                **(
                    {
                        "ttft_prefill_chunk_scheduling_enabled": 1,
                        "ttft_prefill_chunk_policy_active_this_iteration": int(
                            ttft_prefill_policy_active_this_iteration
                        ),
                        "ttft_prefill_chunk_candidate_ids_at_iteration_start": (
                            ",".join(ttft_prefill_policy_candidate_order)
                        ),
                        "ttft_prefill_chunk_selected_request_ids": ",".join(
                            ttft_prefill_policy_selected_request_ids
                        ),
                        "ttft_prefill_chunk_selected_request_count": len(
                            ttft_prefill_policy_selected_request_ids
                        ),
                        "ttft_prefill_chunk_admission_frontier_tokens": (
                            ",".join(
                                f"{request_id}:{tokens}"
                                for request_id, tokens in (
                                    ttft_prefill_frontier.planned_tokens_by_request
                                )
                            )
                        ),
                        "ttft_prefill_chunk_token_budget_before": (
                            ttft_prefill_frontier.token_budget_before
                        ),
                        "ttft_prefill_chunk_decode_reserved_tokens": (
                            ttft_prefill_frontier.decode_reserved_tokens
                        ),
                        "ttft_prefill_chunk_token_budget_after_frontier": (
                            ttft_prefill_frontier.token_budget_after
                        ),
                        "ttft_prefill_chunk_sequence_slots_before": (
                            ttft_prefill_frontier.new_sequence_slots_before
                        ),
                        "ttft_prefill_chunk_sequence_slots_after": (
                            ttft_prefill_frontier.new_sequence_slots_after
                        ),
                        "ttft_prefill_chunk_sequence_blocked_request_ids": (
                            ",".join(
                                ttft_prefill_frontier.sequence_blocked_request_ids
                            )
                        ),
                        "ttft_prefill_chunk_admission_stop_reason": (
                            ttft_prefill_frontier.stop_reason
                        ),
                        "ttft_prefill_chunk_scheduled_request_ids": ",".join(
                            ttft_prefill_policy_scheduled_request_ids
                        ),
                        "ttft_prefill_chunk_scheduled_tokens": (
                            ttft_prefill_policy_scheduled_tokens
                        ),
                        "ttft_prefill_chunk_skipped_running_reqs": (
                            ttft_prefill_policy_skipped_running_reqs
                        ),
                        "ttft_prefill_chunk_skipped_waiting_reqs": (
                            ttft_prefill_policy_skipped_waiting_reqs
                        ),
                        **self.ttft_prefill_chunk_scheduler.iteration_metrics(),
                    }
                    if self.enable_ttft_prefill_chunk_scheduling
                    else {}
                ),
                **(
                    {
                        "tbt_decode_scheduling_enabled": 1,
                        "tbt_decode_policy_active_this_iteration": int(
                            tbt_decode_policy_active_this_iteration
                        ),
                        "tbt_decode_candidate_ids_at_iteration_start": (
                            ",".join(tbt_decode_policy_candidate_order)
                        ),
                        "tbt_decode_scheduled_request_ids": ",".join(
                            tbt_decode_policy_scheduled_request_ids
                        ),
                        "tbt_decode_scheduled_tokens": (
                            tbt_decode_policy_scheduled_tokens
                        ),
                        **self.tbt_decode_scheduler.iteration_metrics(),
                    }
                    if self.enable_tbt_decode_scheduling
                    else {}
                ),
                "running_prefill_kv_wait_count": running_prefill_kv_wait_count,
                "running_prefill_kv_wait_req_count": len(
                    running_prefill_kv_wait_request_ids
                ),
                "running_prefill_kv_wait_request_ids": ",".join(
                    sorted(running_prefill_kv_wait_request_ids)
                ),
                "virtual_context_defer_prefill_enabled": int(
                    self.virtual_context_defer_prefill
                ),
                "prefill_deferred_by_residency_budget": (
                    prefill_deferred_by_residency_budget
                ),
                "prefill_defer_residency_fallback_count": (
                    prefill_defer_residency_fallback_count
                ),
                "preempt_reset_avoided_tokens": preempt_reset_avoided_tokens,
                "prefill_dram_resident_tokens": prefill_dram_resident_tokens,
                "prefill_hbm_reloaded_tokens": prefill_hbm_reloaded_tokens,
                "prefill_resume_without_recompute": (
                    prefill_resume_without_recompute
                ),
                "virtual_context_tiered_external_blocks_enabled": int(
                    self.virtual_context_tiered_external_blocks
                ),
                "virtual_context_tiered_provider_max_external_tokens": (
                    self.virtual_context_tiered_provider_max_external_tokens
                ),
                "virtual_context_ring_prefill_admission_tokens": (
                    self.virtual_context_ring_prefill_admission_tokens
                ),
                "virtual_context_ring_prefill_admission_tokens_effective": (
                    virtual_context_ring_prefill_admission_tokens_effective
                ),
                "virtual_context_ring_prefill_admission_tokens_max": (
                    self.virtual_context_ring_prefill_admission_tokens_max
                ),
                "virtual_context_ring_min_progress_tokens": (
                    self.virtual_context_ring_min_progress_tokens
                ),
                "virtual_context_ring_prefill_admission_dynamic_min_free_blocks": (
                    self.virtual_context_ring_prefill_admission_dynamic_min_free_blocks
                ),
                "virtual_context_ring_prefill_admission_dynamic_step_blocks": (
                    self.virtual_context_ring_prefill_admission_dynamic_step_blocks
                ),
                "virtual_context_ring_prefill_admission_dynamic_step_tokens": (
                    self.virtual_context_ring_prefill_admission_dynamic_step_tokens
                ),
                "context_ring_bw_dynamic_enabled": int(
                    self.virtual_context_ring_bw_dynamic
                ),
                "context_ring_bw_policy": self.virtual_context_ring_bw_policy,
                "context_ring_bw_selected_tokens": (
                    virtual_context_ring_prefill_admission_tokens_effective
                ),
                "context_ring_bw_page_size_bytes": (
                    self.virtual_context_kv_page_size_bytes
                ),
                "context_ring_bw_selected_dma_budget_bytes": (
                    self._virtual_context_ring_dma_budget_bytes(
                        virtual_context_ring_prefill_admission_tokens_effective
                    )
                ),
                "context_ring_bw_reason": (
                    self._virtual_context_ring_bw_last_reason
                ),
                "context_ring_bw_free_blocks": (
                    self._virtual_context_ring_bw_last_free_blocks
                ),
                "context_ring_bw_steps": (
                    self._virtual_context_ring_bw_last_steps
                ),
                "context_ring_bw_safe_blocks": (
                    self._virtual_context_ring_bw_last_safe_blocks
                ),
                "context_ring_bw_safe_tokens": (
                    self._virtual_context_ring_bw_last_safe_tokens
                ),
                "context_ring_bw_reserved_blocks": (
                    self._virtual_context_ring_bw_last_reserved_blocks
                ),
                "context_ring_bw_staging_blocks": (
                    self._virtual_context_ring_bw_last_staging_blocks
                ),
                "context_ring_bw_safety_blocks": (
                    self._virtual_context_ring_bw_last_safety_blocks
                ),
                "context_ring_bw_uses_scheduled_headroom": int(
                    self.virtual_context_ring_bw_use_scheduled_headroom
                ),
                "context_ring_bw_decode_growth_blocks": (
                    self._virtual_context_ring_bw_last_decode_growth_blocks
                ),
                "context_ring_bw_active_decode_cap_tokens": (
                    self._virtual_context_ring_bw_last_active_decode_cap_tokens
                ),
                "context_ring_bw_active_decode_cap_reason": (
                    self._virtual_context_ring_bw_last_active_decode_cap_reason
                ),
                "context_ring_bw_active_decode_reload_pages": (
                    self._virtual_context_ring_bw_last_active_decode_reload_pages
                ),
                "context_ring_bw_progress_reserve_applied": (
                    self._virtual_context_ring_bw_last_progress_reserve_applied
                ),
                "context_ring_bw_progress_reserve_tokens": (
                    self._virtual_context_ring_bw_last_progress_reserve_tokens
                ),
                "context_ring_bw_progress_reserve_blocks": (
                    self._virtual_context_ring_bw_last_progress_reserve_blocks
                ),
                "context_ring_bw_progress_hard_reserved_blocks": (
                    self._virtual_context_ring_bw_last_progress_hard_reserved_blocks
                ),
                "context_ring_bw_progress_reserve_reason": (
                    self._virtual_context_ring_bw_last_progress_reserve_reason
                ),
                "pd_no_mix_ring_progress_lane_allowed": int(
                    self.pd_decoupled_no_mix
                    and pd_decoupled_selected_phase == "prefill"
                    and active_decode_during_prefill_phase
                    and self._virtual_context_ring_bw_last_progress_reserve_applied
                    and num_prefill_tokens > 0
                ),
                "context_ring_pressure_bootstrap_tokens": (
                    self._virtual_context_ring_pressure_bootstrap_token_budget()
                ),
                "context_ring_pressure_free_blocks": (
                    self._virtual_context_ring_pressure_last_free_blocks
                ),
                "context_ring_pressure_threshold_blocks": (
                    self._virtual_context_ring_pressure_last_threshold_blocks
                ),
                "context_ring_pressure_bootstrap_blocks": (
                    self._virtual_context_ring_pressure_last_bootstrap_blocks
                ),
                "context_ring_pressure_ring_blocks": (
                    self._virtual_context_ring_pressure_last_ring_blocks
                ),
                "context_ring_pressure_reserved_blocks": (
                    self._virtual_context_ring_pressure_last_reserved_blocks
                ),
                "context_ring_pressure_native_safe": (
                    self._virtual_context_ring_pressure_last_native_safe
                ),
                "context_ring_pressure_reason": (
                    self._virtual_context_ring_pressure_last_reason
                ),
                "virtual_context_active_decode_native_prefill_tokens": (
                    self.virtual_context_active_decode_native_prefill_tokens
                ),
                "virtual_context_active_decode_ring_prefill_tokens": (
                    self.virtual_context_active_decode_ring_prefill_tokens
                ),
                "virtual_context_ring_prefill_admission_enabled": int(
                    self.virtual_context_ring_prefill_admission_tokens > 0
                ),
                "virtual_context_ring_residency_admission_enabled": int(
                    self.virtual_context_ring_residency_admission
                ),
                "virtual_context_ring_min_free_blocks": (
                    self.virtual_context_ring_min_free_blocks
                ),
                "ring_residency_admission_applied_count": (
                    ring_residency_admission_applied_count
                ),
                "ring_residency_decode_reserve_relaxed_count": (
                    ring_residency_decode_reserve_relaxed_count
                ),
                "residency_admission_raw_required_blocks_max": (
                    residency_admission_raw_required_blocks_max
                ),
                "virtual_context_decode_hotset_first_enabled": int(
                    self.virtual_context_decode_hotset_first
                ),
                "virtual_context_decode_cold_streaming_enabled": int(
                    self.virtual_context_decode_cold_streaming
                ),
                "virtual_context_decode_cold_streaming_on_prefill_done_enabled": int(
                    self.virtual_context_decode_cold_streaming_on_prefill_done
                ),
                "virtual_context_decode_cold_streaming_slo_ms": (
                    self.virtual_context_decode_cold_streaming_slo_ms
                ),
                "virtual_context_decode_cold_streaming_min_slack_ms": (
                    self.virtual_context_decode_cold_streaming_min_slack_ms
                ),
                "virtual_context_decode_cold_streaming_blocked_count": (
                    virtual_context_decode_cold_streaming_blocked_count
                ),
                "virtual_context_decode_cold_streaming_allowed_count": (
                    virtual_context_decode_cold_streaming_allowed_count
                ),
                "virtual_context_decode_after_prefill_streaming_allowed_count": (
                    virtual_context_decode_after_prefill_streaming_allowed_count
                ),
                "virtual_context_decode_cold_streaming_required_pages": (
                    virtual_context_decode_cold_streaming_required_pages
                ),
                "virtual_context_decode_cold_streaming_last_slack_ms": (
                    virtual_context_decode_cold_streaming_last_slack_ms
                ),
                "virtual_context_prefill_ring_tokens_with_decode_cold": (
                    self.virtual_context_prefill_ring_tokens_with_decode_cold
                ),
                "pd_prefill_starvation_max_decode_iters": (
                    self.pd_prefill_starvation_max_decode_iters
                ),
                "pd_ring_min_prefill_tokens_with_decode": (
                    self.pd_ring_min_prefill_tokens_with_decode
                ),
                "virtual_context_prefill_ring_bandwidth_deferred_count": (
                    virtual_context_prefill_ring_bandwidth_deferred_count
                ),
                "virtual_context_prefill_ring_bandwidth_decode_reload_pages": (
                    virtual_context_prefill_ring_bandwidth_decode_reload_pages
                ),
                "virtual_context_prefill_ring_bandwidth_leftover_tokens": (
                    virtual_context_prefill_ring_bandwidth_leftover_tokens
                ),
                "virtual_context_prefill_ring_with_decode_cold_allowed_count": (
                    virtual_context_prefill_ring_with_decode_cold_allowed_count
                ),
                "virtual_context_prefill_ring_with_decode_cold_allowed_tokens": (
                    virtual_context_prefill_ring_with_decode_cold_allowed_tokens
                ),
                "virtual_context_active_decode_ring_prefill_allowed_count": (
                    virtual_context_active_decode_ring_prefill_allowed_count
                ),
                "virtual_context_active_decode_native_prefill_blocked_count": (
                    virtual_context_active_decode_native_prefill_blocked_count
                ),
                "virtual_context_active_decode_ring_prefill_reload_pages": (
                    virtual_context_active_decode_ring_prefill_reload_pages
                ),
                "virtual_context_active_decode_ring_prefill_force_all_count": (
                    virtual_context_active_decode_ring_prefill_force_all_count
                ),
                "virtual_context_enabled": int(self.virtual_context_enabled),
                "pd_native_active_decode_prefill_admission_enabled": int(
                    active_decode_during_prefill_phase
                    and not self.virtual_context_enabled
                    and self.pd_muxwise_dynamic_policy
                ),
                "pd_native_active_decode_prefill_capacity_check_count": (
                    pd_native_active_decode_prefill_capacity_check_count
                ),
                "pd_native_active_decode_prefill_capacity_defer_count": (
                    pd_native_active_decode_prefill_capacity_defer_count
                ),
                "pd_native_active_decode_prefill_required_blocks": (
                    pd_native_active_decode_prefill_required_blocks
                ),
                "pd_native_active_decode_prefill_retry_min_required_blocks": (
                    pd_native_prefill_retry_min_blocks
                ),
                "pd_native_active_decode_prefill_chunk_required_blocks": (
                    pd_native_active_decode_prefill_chunk_required_blocks
                ),
                "pd_native_active_decode_growth_reserve_blocks": (
                    pd_native_active_decode_growth_reserve_blocks
                ),
                "pd_native_active_decode_prefill_free_blocks": (
                    pd_native_active_decode_prefill_free_blocks
                ),
                "pd_native_active_decode_prefill_capacity_reason": (
                    pd_native_active_decode_prefill_capacity_reason
                ),
                "pd_native_active_decode_prefill_capacity_blocked": int(
                    pd_native_active_decode_prefill_capacity_blocked
                ),
                "virtual_context_residency_pressure_release_attempt_count": (
                    virtual_context_residency_pressure_release_attempt_count
                ),
                "virtual_context_residency_pressure_release_success_count": (
                    virtual_context_residency_pressure_release_success_count
                ),
                "virtual_context_residency_pressure_release_failed_count": (
                    virtual_context_residency_pressure_release_failed_count
                ),
                "virtual_context_residency_pressure_release_tokens": (
                    virtual_context_residency_pressure_release_tokens
                ),
                "virtual_context_residency_pressure_release_reasons": ",".join(
                    sorted(virtual_context_residency_pressure_release_reasons)
                ),
                "virtual_context_pressure_snapshot_request_count": len(
                    virtual_context_pressure_snapshot_request_ids
                ),
                "virtual_context_capacity_floor_snapshot_request_count": len(
                    virtual_context_capacity_floor_snapshot_request_ids
                ),
                "virtual_context_capacity_floor_snapshot_request_ids": ",".join(
                    sorted(
                        virtual_context_capacity_floor_snapshot_request_ids
                    )
                ),
                "virtual_context_capacity_floor_release_attempt_count": (
                    virtual_context_capacity_floor_release_attempt_count
                ),
                "virtual_context_capacity_floor_release_success_count": (
                    virtual_context_capacity_floor_release_success_count
                ),
                "virtual_context_capacity_floor_release_failed_count": (
                    virtual_context_capacity_floor_release_failed_count
                ),
                "virtual_context_capacity_floor_victim_policy": (
                    self.virtual_context_residency_victim_policy.value
                ),
                "virtual_context_capacity_floor_victim_candidate_ids": ",".join(
                    virtual_context_capacity_floor_victim_candidate_ids
                ),
                "virtual_context_capacity_floor_victim_candidate_ring_chunks": (
                    ",".join(
                        virtual_context_capacity_floor_victim_candidate_ring_chunks
                    )
                ),
                "virtual_context_capacity_floor_victim_candidate_reclaim_blocks": (
                    ",".join(
                        virtual_context_capacity_floor_victim_candidate_reclaim_blocks
                    )
                ),
                "virtual_context_capacity_floor_victim_candidate_copy_tokens": (
                    ",".join(
                        virtual_context_capacity_floor_victim_candidate_copy_tokens
                    )
                ),
                "virtual_context_capacity_floor_victim_candidate_copy_scores": (
                    ",".join(
                        virtual_context_capacity_floor_victim_candidate_copy_scores
                    )
                ),
                "virtual_context_capacity_floor_victim_jit_triggered": (
                    virtual_context_capacity_floor_victim_jit_triggered
                ),
                "virtual_context_capacity_floor_victim_jit_safe_candidate_count": (
                    virtual_context_capacity_floor_victim_jit_safe_candidate_count
                ),
                "virtual_context_capacity_floor_victim_required_reclaim_blocks": (
                    virtual_context_capacity_floor_victim_required_reclaim_blocks
                ),
                "virtual_context_capacity_floor_victim_ring_chunk_tokens": (
                    virtual_context_capacity_floor_victim_ring_chunk_tokens
                ),
                "virtual_context_capacity_floor_victim_selected_request_id": (
                    virtual_context_capacity_floor_victim_selected_request_id
                ),
                "virtual_context_capacity_floor_victim_selected_context_tokens": (
                    virtual_context_capacity_floor_victim_selected_context_tokens
                ),
                "virtual_context_capacity_floor_victim_selected_remaining_tokens": (
                    virtual_context_capacity_floor_victim_selected_remaining_tokens
                ),
                "virtual_context_capacity_floor_victim_selected_ring_chunks": (
                    virtual_context_capacity_floor_victim_selected_ring_chunks
                ),
                "virtual_context_capacity_floor_victim_selected_reclaim_blocks": (
                    virtual_context_capacity_floor_victim_selected_reclaim_blocks
                ),
                "virtual_context_capacity_floor_victim_selected_ring_work": (
                    virtual_context_capacity_floor_victim_selected_ring_work
                ),
                "virtual_context_capacity_floor_victim_selected_d2h_eviction_tokens": (
                    virtual_context_capacity_floor_victim_selected_d2h_eviction_tokens
                ),
                "virtual_context_capacity_floor_victim_selected_ring_staging_h2d_tokens": (
                    virtual_context_capacity_floor_victim_selected_ring_staging_h2d_tokens
                ),
                "virtual_context_capacity_floor_victim_selected_decode_reload_h2d_tokens": (
                    virtual_context_capacity_floor_victim_selected_decode_reload_h2d_tokens
                ),
                "virtual_context_capacity_floor_victim_selected_total_copy_tokens": (
                    virtual_context_capacity_floor_victim_selected_total_copy_tokens
                ),
                "virtual_context_capacity_floor_victim_selected_request_ids": (
                    ",".join(
                        virtual_context_capacity_floor_victim_selected_request_ids
                    )
                ),
                "virtual_context_capacity_floor_victim_selected_reclaim_blocks_total": (
                    virtual_context_capacity_floor_victim_selected_reclaim_blocks_total
                ),
                "virtual_context_scheduled_prefill_fit_triggered": (
                    scheduled_prefill_fit_triggered
                ),
                "virtual_context_scheduled_prefill_fit_target_request_id": (
                    scheduled_prefill_fit_target_request_id
                ),
                "virtual_context_scheduled_prefill_fit_target_logical_tokens": (
                    scheduled_prefill_fit_target_logical_tokens
                ),
                "virtual_context_scheduled_prefill_fit_target_chunk_tokens": (
                    scheduled_prefill_fit_target_chunk_tokens
                ),
                "virtual_context_scheduled_prefill_fit_target_raw_required_blocks": (
                    scheduled_prefill_fit_target_raw_required_blocks
                ),
                "virtual_context_scheduled_prefill_fit_target_hard_reclaim_blocks": (
                    scheduled_prefill_fit_target_hard_reclaim_blocks
                ),
                "virtual_context_scheduled_prefill_fit_target_waived_headroom_blocks": (
                    scheduled_prefill_fit_target_waived_headroom_blocks
                ),
                "virtual_context_scheduled_prefill_fit_target_reserved_blocks": (
                    scheduled_prefill_fit_target_reserved_blocks
                ),
                "virtual_context_scheduled_prefill_fit_target_free_blocks": (
                    scheduled_prefill_fit_target_free_blocks
                ),
                "virtual_context_scheduled_prefill_fit_target_reclaim_blocks": (
                    scheduled_prefill_fit_target_reclaim_blocks
                ),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "capacity_fraction"
                ): self.virtual_context_scheduled_prefill_fit_capacity_fraction,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "total_usable_blocks"
                ): self.virtual_context_scheduled_prefill_fit_total_usable_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "capacity_limit_blocks"
                ): self.virtual_context_scheduled_prefill_fit_capacity_limit_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "headroom_blocks"
                ): self.virtual_context_scheduled_prefill_fit_headroom_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "active_decode_hard_headroom"
                ): int(scheduled_prefill_fit_active_decode_hard_headroom),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "hard_headroom_blocks"
                ): scheduled_prefill_fit_hard_headroom_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "soft_headroom_blocks"
                ): scheduled_prefill_fit_soft_headroom_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "soft_headroom_shortfall_blocks"
                ): scheduled_prefill_fit_soft_headroom_shortfall_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "decode_growth_runway_active"
                ): int(scheduled_prefill_fit_decode_growth_runway_active),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "decode_growth_runway_target_request_id"
                ): scheduled_prefill_fit_decode_growth_runway_target_request_id,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "decode_growth_runway_absorbed_blocks"
                ): scheduled_prefill_fit_decode_growth_runway_absorbed_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "decode_growth_runway_consumed"
                ): int(scheduled_prefill_fit_decode_growth_runway_consumed),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "decode_growth_runway_created_count"
                ): self._scheduled_prefill_fit_decode_growth_runway_created_count,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "decode_growth_runway_admission_count"
                ): self._scheduled_prefill_fit_decode_growth_runway_admission_count,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "decode_growth_runway_invalidated_count"
                ): self._scheduled_prefill_fit_decode_growth_runway_invalidated_count,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "commit_reclaim_blocks"
                ): scheduled_prefill_fit_commit_reclaim_blocks,
                "virtual_context_scheduled_prefill_fit_safe_candidate_count": (
                    scheduled_prefill_fit_safe_candidate_count
                ),
                "virtual_context_scheduled_prefill_fit_partial_release_blocks": (
                    scheduled_prefill_fit_partial_release_blocks
                ),
                "virtual_context_scheduled_prefill_fit_whole_request_mode": int(
                    scheduled_prefill_fit_active
                ),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "whole_request_victim_count"
                ): scheduled_prefill_fit_whole_request_victim_count,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "whole_request_reclaimed_blocks"
                ): scheduled_prefill_fit_whole_request_reclaimed_blocks,
                "virtual_context_scheduled_prefill_fit_allocator_resume_count": (
                    scheduled_prefill_fit_allocator_resume_count
                ),
                (
                    "virtual_context_scheduled_prefill_fit_allocator_"
                    "resume_resident_blocks"
                ): scheduled_prefill_fit_allocator_resume_resident_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_allocator_"
                    "resume_virtual_blocks"
                ): scheduled_prefill_fit_allocator_resume_virtual_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_allocator_"
                    "resume_request_ids"
                ): ",".join(
                    sorted(scheduled_prefill_fit_allocator_resume_request_ids)
                ),
                "virtual_context_scheduled_prefill_fit_target_change_count": (
                    self._scheduled_prefill_fit_target_change_count
                ),
                "virtual_context_scheduled_prefill_fit_proactive_target_request_id": (
                    scheduled_prefill_fit_proactive_target_request_id
                ),
                "virtual_context_scheduled_prefill_fit_proactive_victim_request_id": (
                    scheduled_prefill_fit_proactive_victim_request_id
                ),
                "virtual_context_scheduled_prefill_fit_proactive_victim_request_ids": (
                    scheduled_prefill_fit_proactive_victim_request_ids
                ),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "proactive_external_victim_count"
                ): scheduled_prefill_fit_proactive_external_victim_count,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "proactive_waiting_victim_count"
                ): scheduled_prefill_fit_proactive_waiting_victim_count,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "proactive_decode_handoff"
                ): scheduled_prefill_fit_proactive_decode_handoff,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "proactive_projected_logical_tokens"
                ): (
                    scheduled_prefill_fit_proactive_projected_logical_tokens
                ),
                "virtual_context_scheduled_prefill_fit_proactive_next_chunk_tokens": (
                    scheduled_prefill_fit_proactive_next_chunk_tokens
                ),
                "virtual_context_scheduled_prefill_fit_proactive_raw_required_blocks": (
                    scheduled_prefill_fit_proactive_raw_required_blocks
                ),
                "virtual_context_scheduled_prefill_fit_proactive_reserved_blocks": (
                    scheduled_prefill_fit_proactive_reserved_blocks
                ),
                "virtual_context_scheduled_prefill_fit_proactive_free_blocks": (
                    scheduled_prefill_fit_proactive_free_blocks
                ),
                "virtual_context_scheduled_prefill_fit_proactive_deficit_blocks": (
                    scheduled_prefill_fit_proactive_deficit_blocks
                ),
                "virtual_context_scheduled_prefill_fit_proactive_plan_mode": (
                    scheduled_prefill_fit_proactive_plan_mode
                ),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "proactive_native_deficit_blocks"
                ): scheduled_prefill_fit_proactive_native_deficit_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "proactive_ring_fallback_deficit_blocks"
                ): scheduled_prefill_fit_proactive_ring_fallback_deficit_blocks,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "proactive_avoided_peer_blocks"
                ): scheduled_prefill_fit_proactive_avoided_peer_blocks,
                "virtual_context_scheduled_prefill_fit_victim_selection": (
                    "best_fit_whole_request"
                ),
                "virtual_context_scheduled_prefill_fit_proactive_overshoot_blocks": (
                    max(
                        0,
                        len(scheduled_prefill_fit_proactive_snapshot_pages)
                        - scheduled_prefill_fit_proactive_deficit_blocks,
                    )
                ),
                "virtual_context_scheduled_prefill_fit_proactive_self_victim": int(
                    bool(scheduled_prefill_fit_proactive_target_request_id)
                    and scheduled_prefill_fit_proactive_target_request_id
                    in {
                        victim.request_id
                        for victim in scheduled_prefill_fit_proactive_victims
                    }
                ),
                "virtual_context_scheduled_prefill_fit_proactive_snapshot_page_count": (
                    len(scheduled_prefill_fit_proactive_snapshot_pages)
                ),
                "virtual_context_scheduled_prefill_fit_proactive_snapshot_pages": (
                    ",".join(
                        f"{request_id}:{group_id}:{logical_page}"
                        for request_id, group_id, logical_page in sorted(
                            scheduled_prefill_fit_proactive_snapshot_pages
                        )
                    )
                ),
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "target_request_id"
                ): (
                    scheduled_prefill_fit_preparation_target_request_id
                ),
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "victim_request_id"
                ): (
                    scheduled_prefill_fit_preparation_victim_request_id
                ),
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "victim_request_ids"
                ): scheduled_prefill_fit_preparation_victim_request_ids,
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "planned_page_count"
                ): (
                    scheduled_prefill_fit_preparation_planned_page_count
                ),
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "ready_page_count"
                ): (
                    scheduled_prefill_fit_preparation_ready_page_count
                ),
                "virtual_context_scheduled_prefill_fit_preparation_waiting": int(
                    scheduled_prefill_fit_preparation_waiting
                ),
                "virtual_context_scheduled_prefill_fit_preparation_ready": int(
                    scheduled_prefill_fit_preparation_ready
                ),
                "virtual_context_scheduled_prefill_fit_preparation_created_count": (
                    self._scheduled_prefill_fit_preparation_created_count
                ),
                "virtual_context_scheduled_prefill_fit_preparation_wait_count": (
                    self._scheduled_prefill_fit_preparation_wait_count
                ),
                "virtual_context_scheduled_prefill_fit_preparation_ready_count": (
                    self._scheduled_prefill_fit_preparation_ready_count
                ),
                "virtual_context_scheduled_prefill_fit_preparation_cancelled_count": (
                    self._scheduled_prefill_fit_preparation_cancelled_count
                ),
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "target_mismatch_count"
                ): self._scheduled_prefill_fit_preparation_target_mismatch_count,
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "reclaim_consumed_count"
                ): self._scheduled_prefill_fit_preparation_reclaim_consumed_count,
                (
                    "virtual_context_scheduled_prefill_fit_preparation_"
                    "reclaim_consumed_victim_count"
                ): (
                    self._scheduled_prefill_fit_preparation_consumed_victim_count
                ),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "protected_victim_skip_count"
                ): self._scheduled_prefill_fit_protected_victim_skip_count,
                "virtual_context_scheduled_prefill_fit_unprepared_reclaim_count": (
                    self._scheduled_prefill_fit_unprepared_reclaim_count
                ),
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "soft_headroom_shortfall_count"
                ): self._scheduled_prefill_fit_soft_headroom_shortfall_count,
                (
                    "virtual_context_scheduled_prefill_fit_"
                    "soft_headroom_shortfall_blocks_total"
                ): self._scheduled_prefill_fit_soft_headroom_shortfall_blocks,
                "virtual_context_dram_register_pressure_only_enabled": int(
                    self.virtual_context_dram_register_pressure_only
                ),
                "virtual_context_pressure_snapshot_request_ids": ",".join(
                    sorted(virtual_context_pressure_snapshot_request_ids)
                ),
                "virtual_context_hbm_relief_recompute_enabled": int(
                    self.virtual_context_recompute_on_hbm_relief
                ),
                "virtual_context_hbm_relief_recompute_count": (
                    virtual_context_hbm_relief_recompute_count
                ),
                "virtual_context_hbm_relief_recompute_tokens": (
                    virtual_context_hbm_relief_recompute_tokens
                ),
                "virtual_context_hbm_relief_recompute_reload_pages": (
                    virtual_context_hbm_relief_recompute_reload_pages
                ),
                "virtual_context_hbm_relief_recompute_min_free_blocks": (
                    virtual_context_hbm_relief_recompute_min_free_blocks
                ),
                "virtual_context_hbm_relief_recompute_free_blocks_max": (
                    virtual_context_hbm_relief_recompute_free_blocks_max
                ),
                "virtual_context_hbm_relief_recompute_request_ids": ",".join(
                    sorted(virtual_context_hbm_relief_recompute_request_ids)
                ),
                "virtual_context_reload_to_hbm_enabled": int(
                    self.virtual_context_reload_to_hbm
                ),
                "virtual_context_prefill_ring_required_mode": (
                    self.virtual_context_prefill_ring_required_mode
                ),
                "virtual_context_reload_to_hbm_candidate_count": (
                    virtual_context_reload_to_hbm_candidate_count
                ),
                "virtual_context_reload_to_hbm_scheduled_count": (
                    virtual_context_reload_to_hbm_scheduled_count
                ),
                "virtual_context_reload_to_hbm_tokens": (
                    virtual_context_reload_to_hbm_tokens
                ),
                "virtual_context_reload_to_hbm_pages": (
                    virtual_context_reload_to_hbm_pages
                ),
                "virtual_context_reload_to_hbm_required_blocks": (
                    virtual_context_reload_to_hbm_required_blocks
                ),
                "virtual_context_reload_to_hbm_min_free_blocks": (
                    virtual_context_reload_to_hbm_min_free_blocks
                ),
                "virtual_context_reload_to_hbm_free_blocks_max": (
                    virtual_context_reload_to_hbm_free_blocks_max
                ),
                "virtual_context_reload_to_hbm_request_ids": ",".join(
                    sorted(virtual_context_reload_to_hbm_request_ids)
                ),
                "virtual_context_native_prefill_promotion_request_ids": (
                    ",".join(
                        sorted(
                            virtual_context_native_prefill_promotion_request_ids
                        )
                    )
                ),
                "virtual_context_native_prefill_promotion_pages": (
                    virtual_context_native_prefill_promotion_pages
                ),
                "virtual_context_native_prefill_promotion_required_blocks": (
                    virtual_context_native_prefill_promotion_required_blocks
                ),
                "virtual_context_native_prefill_promotion_reasons": ",".join(
                    sorted(virtual_context_native_prefill_promotion_reasons)
                ),
                "virtual_context_decode_handoff_reload_candidate_count": (
                    virtual_context_decode_handoff_reload_candidate_count
                ),
                "virtual_context_decode_handoff_reload_scheduled_count": (
                    virtual_context_decode_handoff_reload_scheduled_count
                ),
                "virtual_context_decode_handoff_reload_blocked_count": (
                    virtual_context_decode_handoff_reload_blocked_count
                ),
                "virtual_context_decode_handoff_reload_pages": (
                    virtual_context_decode_handoff_reload_pages
                ),
                "virtual_context_decode_handoff_reload_required_blocks": (
                    virtual_context_decode_handoff_reload_required_blocks
                ),
                "virtual_context_decode_handoff_reload_min_free_blocks": (
                    virtual_context_decode_handoff_reload_min_free_blocks
                ),
                "virtual_context_decode_handoff_reload_reasons": ",".join(
                    sorted(virtual_context_decode_handoff_reload_reasons)
                ),
                "virtual_context_decode_handoff_reload_request_ids": ",".join(
                    sorted(virtual_context_decode_handoff_reload_request_ids)
                ),
                "virtual_context_decode_handoff_reclaim_target_blocks": (
                    virtual_context_decode_handoff_reclaim_target_blocks
                ),
                "virtual_context_decode_handoff_reclaim_shortfall_blocks": (
                    virtual_context_decode_handoff_reclaim_shortfall_blocks
                ),
                "virtual_context_decode_handoff_reclaim_attempt_count": (
                    virtual_context_decode_handoff_reclaim_attempt_count
                ),
                "virtual_context_decode_handoff_reclaim_success_count": (
                    virtual_context_decode_handoff_reclaim_success_count
                ),
                "virtual_context_decode_handoff_reclaim_failed_count": (
                    virtual_context_decode_handoff_reclaim_failed_count
                ),
                "virtual_context_decode_handoff_reclaimed_blocks": (
                    virtual_context_decode_handoff_reclaimed_blocks
                ),
                "virtual_context_decode_handoff_reclaim_victim_ids": ",".join(
                    virtual_context_decode_handoff_reclaim_victim_ids
                ),
                "virtual_context_decode_handoff_reclaim_reasons": ",".join(
                    sorted(virtual_context_decode_handoff_reclaim_reasons)
                ),
                "virtual_context_decode_handoff_reclaim_final_free_blocks": (
                    virtual_context_decode_handoff_reclaim_final_free_blocks
                ),
                "virtual_context_decode_handoff_reclaim_retry_success_count": (
                    virtual_context_decode_handoff_reclaim_retry_success_count
                ),
                "context_ring_prefill_admission_cap_hit_count": (
                    ring_prefill_admission_cap_hit_count
                ),
                "context_ring_prefill_admission_cap_running_hit_count": (
                    ring_prefill_admission_cap_running_hit_count
                ),
                "context_ring_prefill_admission_cap_waiting_hit_count": (
                    ring_prefill_admission_cap_waiting_hit_count
                ),
                "context_ring_prefill_admission_original_tokens_sum": (
                    ring_prefill_admission_original_tokens_sum
                ),
                "context_ring_prefill_admission_capped_tokens_sum": (
                    ring_prefill_admission_capped_tokens_sum
                ),
                "context_ring_prefill_admission_min_progress_defer_count": (
                    ring_prefill_admission_min_progress_defer_count
                ),
                "context_ring_prefill_admission_tokens_saved": max(
                    0,
                    ring_prefill_admission_original_tokens_sum
                    - ring_prefill_admission_capped_tokens_sum,
                ),
                "context_ring_prefill_phase_budget_original": (
                    ring_prefill_phase_budget_original
                ),
                "context_ring_prefill_phase_budget_capped": (
                    ring_prefill_phase_budget_capped
                ),
                "context_ring_prefill_phase_budget_applied": (
                    ring_prefill_phase_budget_applied
                ),
                "virtual_context_resume_guard_count": (
                    virtual_context_resume_guard_count
                ),
                "virtual_context_resume_guard_request_ids": ",".join(
                    sorted(virtual_context_resume_guard_request_ids)
                ),
                "virtual_context_residency_admission_enabled": int(
                    self.virtual_context_residency_admission
                ),
                "prefill_deferred_by_residency_admission": (
                    prefill_deferred_by_residency_admission
                ),
                "residency_admission_deferred_request_ids": ",".join(
                    sorted(residency_admission_deferred_request_ids)
                ),
                "residency_admission_required_blocks_sum": (
                    residency_admission_required_blocks_sum
                ),
                "residency_admission_required_blocks_max": (
                    residency_admission_required_blocks_max
                ),
                "virtual_context_decode_hotset_reserve_blocks": (
                    self.virtual_context_decode_hotset_reserve_blocks
                ),
                "virtual_context_staging_reserved_blocks": (
                    self.virtual_context_staging_reserved_blocks
                ),
                "virtual_context_deferred_req_ids": ",".join(
                    sorted(virtual_context_deferred_req_ids)
                ),
                "virtual_context_deferred_same_step_resumed_req_ids": ",".join(
                    sorted(same_step_resumed_deferred_req_ids)
                ),
                "virtual_context_deferred_output_req_ids": ",".join(
                    sorted(residency_deferred_req_ids_for_output)
                ),
                "virtual_context_defer_fallback_reasons": ",".join(
                    sorted(virtual_context_defer_fallback_reasons)
                ),
                "scheduled_phase": scheduled_phase,
                "pd_phase_classified_phase": classified_scheduled_phase,
                "pd_phase_classification_corrected": (
                    pd_phase_classification_corrected
                ),
                "pd_phase_handoff_token_refresh_request_count": len(
                    pd_phase_handoff_token_refresh_request_ids
                ),
                "pd_phase_handoff_token_refresh_request_ids": ",".join(
                    sorted(pd_phase_handoff_token_refresh_request_ids)
                ),
                "pd_decoupled_no_mix_enabled": int(self.pd_decoupled_no_mix),
                "pd_decoupled_selected_phase": pd_decoupled_selected_phase,
                "pd_decoupled_skipped_running_prefill_reqs": (
                    pd_decoupled_skipped_running_prefill_reqs
                ),
                "pd_decoupled_skipped_running_decode_reqs": (
                    pd_decoupled_skipped_running_decode_reqs
                ),
                "pd_residency_skipped_running_prefill_reqs": len(
                    pd_residency_skipped_running_prefill_request_ids
                ),
                "pd_residency_skipped_running_prefill_request_ids": ",".join(
                    sorted(pd_residency_skipped_running_prefill_request_ids)
                ),
                "pd_residency_free_blocks": pd_residency_last_free_blocks,
                "pd_residency_decode_kv_reserve_blocks": (
                    pd_residency_last_decode_resident_blocks
                ),
                "pd_residency_prefill_required_blocks": (
                    pd_residency_prefill_required_blocks_max
                ),
                "pd_residency_prefill_required_blocks_max": (
                    pd_residency_prefill_required_blocks_max
                ),
                "pd_residency_prefill_required_blocks_sum": (
                    pd_residency_prefill_required_blocks_sum
                ),
                "prefill_deferred_by_decode_kv_reserve": len(
                    pd_residency_serialized_request_ids
                ),
                "pd_serialized_by_kv_reserve": int(
                    bool(pd_residency_serialized_request_ids)
                ),
                "pd_residency_serialized_request_ids": ",".join(
                    sorted(pd_residency_serialized_request_ids)
                ),
                "running_prefill_unscheduled_duration_s": (
                    pd_residency_running_prefill_unscheduled_duration_s
                ),
                "pd_multiplexing_baseline_mode": (
                    self.pd_phase_separated_baseline_mode
                ),
                "pd_phase_separated_baseline_enabled": int(
                    self.pd_decoupled_no_mix
                ),
                "pd_phase_separated_last_phase_before": (
                    pd_phase_separated_last_phase_before
                ),
                "pd_phase_separated_selected_phase": (
                    pd_decoupled_selected_phase
                ),
                "pd_phase_separated_mixed_violation": int(
                    self.pd_decoupled_no_mix and scheduled_phase == "mixed"
                ),
                "pd_forced_schedule_phase": (
                    self._pd_forced_schedule_phase or ""
                ),
                "pd_separate_scheduler_pending_req_count": len(
                    self._pd_pending_request_ids
                ),
                "pd_pending_decode_request_count": len(
                    self._pd_pending_phase_request_ids.get("decode", set())
                ),
                "pd_pending_prefill_request_count": len(
                    self._pd_pending_phase_request_ids.get("prefill", set())
                ),
                "pd_pending_decode_active": int(
                    self._pd_pending_decode_active()
                ),
                "pd_pending_final_prefill_request_count": len(
                    self._pd_pending_phase_request_ids.get(
                        "prefill_final", set()
                    )
                ),
                "pd_pending_schedule_block_count": (
                    self._pd_pending_schedule_block_count
                ),
                "pd_pending_prefill_chain_allow_count": (
                    self._pd_pending_prefill_chain_allow_count
                ),
                "pd_pending_schedule_last_reason": (
                    self._pd_pending_schedule_last_reason
                ),
                "kv_retention_tiered_single_flight_block_count": (
                    self._tiered_execution_single_flight_block_count
                ),
                "kv_retention_tiered_single_flight_last_request_id": (
                    self._tiered_execution_single_flight_last_request_id
                ),
                "kv_retention_tiered_single_flight_last_generation": (
                    self._tiered_execution_single_flight_last_generation
                ),
                "kv_retention_execution_plan_ack_count": (
                    self._tiered_execution_plan_ack_count
                ),
                "kv_retention_execution_plan_failed_load_ack_count": (
                    self._tiered_execution_plan_failed_load_ack_count
                ),
                "kv_retention_execution_plan_overwrite_reject_count": (
                    self._tiered_execution_plan_overwrite_reject_count
                ),
                "pd_preemption_future_protection_count": (
                    self._pd_preemption_future_protection_count
                ),
                "pd_preemption_future_protection_last_request_ids": (
                    self._pd_preemption_future_protection_last_request_ids
                ),
                "pd_live_decode_preemption_protection_count": (
                    self._pd_live_decode_preemption_protection_count
                ),
                "pd_live_decode_preemption_protection_last_request_ids": (
                    self._pd_live_decode_preemption_protection_last_request_ids
                ),
                "pd_separate_scheduler_skipped_pending_reqs": (
                    pd_separate_scheduler_skipped_pending_reqs
                ),
                "total_num_scheduled_tokens": total_num_scheduled_tokens,
                "scheduled_total_tokens": total_num_scheduled_tokens,
                "num_prefill_reqs": num_prefill_reqs,
                "num_decode_reqs": num_decode_reqs,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "scheduled_prefill_tokens": num_prefill_tokens,
                "scheduled_decode_tokens": num_decode_tokens,
                "scheduled_num_seqs": scheduled_num_seqs,
                "num_new_prefill_admitted": num_new_prefill_admitted,
                "num_prefill_chunks_continued": num_prefill_chunks_continued,
                "num_decode_continued": num_decode_continued,
                "num_prefill_deferred": num_waiting_prefill_reqs_after,
                "prefill_defer_reason_counts": dict(prefill_defer_reason_counts),
                "prefill_defer_reason_kv_capacity": prefill_defer_reason_counts[
                    "kv_capacity"
                ],
                "prefill_defer_reason_max_tokens": prefill_defer_reason_counts[
                    "max_num_batched_tokens"
                ],
                "prefill_defer_reason_max_seqs": prefill_defer_reason_counts[
                    "max_num_seqs"
                ],
                "prefill_defer_reason_policy": prefill_defer_reason_counts[
                    "policy"
                ],
                "prefill_defer_reason_preemption": prefill_defer_reason_counts[
                    "preemption"
                ],
                "prefill_defer_reason_running_prefill_kv_wait": (
                    prefill_defer_reason_counts["running_prefill_kv_wait"]
                ),
                "prefill_defer_reason_residency_budget": (
                    prefill_defer_reason_counts["residency_budget"]
                ),
                "prefill_defer_reason_fcfs_request_order": (
                    prefill_defer_reason_counts["fcfs_request_order"]
                ),
                "prefill_defer_reason_ttft_prefill_chunk_policy": (
                    prefill_defer_reason_counts[
                        "ttft_prefill_chunk_policy"
                    ]
                ),
                "admission_blocked_by_kv_capacity": prefill_defer_reason_counts[
                    "kv_capacity"
                ],
                "admission_blocked_by_max_tokens": prefill_defer_reason_counts[
                    "max_num_batched_tokens"
                ],
                "admission_blocked_by_max_seqs": prefill_defer_reason_counts[
                    "max_num_seqs"
                ],
                "admission_blocked_by_policy": prefill_defer_reason_counts[
                    "policy"
                ],
                "admission_blocked_by_preemption": prefill_defer_reason_counts[
                    "preemption"
                ],
                "admission_blocked_by_running_prefill_kv_wait": (
                    prefill_defer_reason_counts["running_prefill_kv_wait"]
                ),
                "admission_blocked_by_residency_budget": (
                    prefill_defer_reason_counts["residency_budget"]
                ),
                "admission_blocked_by_fcfs_request_order": (
                    prefill_defer_reason_counts["fcfs_request_order"]
                ),
                "token_budget_remaining": token_budget,
                "token_budget_consumed_by_running": token_budget_consumed_by_running,
                "token_budget_consumed_by_running_prefill": (
                    token_budget_consumed_by_running_prefill
                ),
                "token_budget_consumed_by_running_decode": (
                    token_budget_consumed_by_running_decode
                ),
                "token_budget_remaining_after_running": token_budget_after_running,
                "token_budget_remaining_for_new_prefill": (
                    token_budget_after_running
                ),
                "token_budget_consumed_by_new_prefill": (
                    token_budget_consumed_by_new_prefill
                ),
                "remaining_token_budget_after_decode": max(
                    0,
                    self.max_num_scheduled_tokens
                    - token_budget_consumed_by_running_decode,
                ),
                "remaining_token_budget_after_prefill": token_budget,
                "max_num_scheduled_tokens": self.max_num_scheduled_tokens,
                "max_num_running_reqs": self.max_num_running_reqs,
                "max_num_batched_tokens": self.max_num_scheduled_tokens,
                "max_num_seqs": self.max_num_running_reqs,
                "batched_token_utilization": (
                    total_num_scheduled_tokens / self.max_num_scheduled_tokens
                    if self.max_num_scheduled_tokens > 0
                    else 0.0
                ),
                "seq_utilization": (
                    scheduled_num_seqs / self.max_num_running_reqs
                    if self.max_num_running_reqs > 0
                    else 0.0
                ),
                **batch_context_stats,
            }
            iteration_debug.update(
                self.kv_cache_manager.get_virtual_context_schedule_timings()
            )
            iteration_debug.update(self._pending_residency_victim_debug())
            iteration_debug.update(
                {
                    "schedule_running_scan_duration_s": (
                        schedule_running_scan_duration_s
                    ),
                    "schedule_native_handoff_duration_s": (
                        schedule_native_handoff_duration_s
                    ),
                    "schedule_decode_cold_guard_duration_s": (
                        schedule_decode_cold_guard_duration_s
                    ),
                    "schedule_allocate_slots_duration_s": (
                        schedule_allocate_slots_duration_s
                    ),
                    "virtual_context_native_decode_runtime_fast_path_count": (
                        native_decode_runtime_fast_path_count
                    ),
                }
            )
            tenant_kv_debug_start = time.perf_counter()
            iteration_debug.update(
                self.kv_cache_manager.get_tenant_kv_debug_stats()
            )
            virtual_context_tenant_kv_debug_duration_s += (
                time.perf_counter() - tenant_kv_debug_start
            )

            native_decode_control_fast_path_candidate = (
                self.virtual_context_native_decode_control_fast_path
                and bool(num_scheduled_tokens)
                and num_prefill_tokens == 0
                and num_decode_tokens > 0
                and native_decode_fast_path_eligible_request_count
                == len(num_scheduled_tokens)
                and not preempted_reqs
                and not pd_residency_skipped_running_prefill_request_ids
                and not virtual_context_deferred_req_ids
                and not scheduled_residency_rehydrated_req_ids
            )
            native_decode_control_fast_path_used = (
                native_decode_control_fast_path_candidate
                and bool(self._virtual_context_debug_stats_cache)
            )
            native_decode_fast_path_saved_attention_plan_scan_count = 0
            native_decode_fast_path_saved_reload_guard_scan_count = 0

            if native_decode_control_fast_path_used:
                iteration_debug.update(
                    self._native_decode_attention_plan_debug_fields(
                        prefix="prefill_attention_plan",
                        phase="prefill",
                        request_count=0,
                        query_tokens=0,
                        staging_budget_blocks=(
                            self.virtual_context_staging_reserved_blocks
                        ),
                    )
                )
                iteration_debug.update(
                    self._native_decode_reload_guard_debug_fields(
                        prefix="prefill_reload_guard",
                        phase="prefill",
                        request_count=0,
                    )
                )
                iteration_debug.update(
                    self._native_decode_attention_plan_debug_fields(
                        prefix="decode_attention_plan",
                        phase="decode",
                        request_count=len(scheduled_decode_plan_req_ids),
                        query_tokens=num_decode_tokens,
                        staging_budget_blocks=(
                            self.virtual_context_staging_reserved_blocks
                        ),
                    )
                )
                iteration_debug.update(
                    self._native_decode_reload_guard_debug_fields(
                        prefix="decode_reload_guard",
                        phase="decode",
                        request_count=len(scheduled_decode_plan_req_ids),
                    )
                )
                native_decode_fast_path_saved_attention_plan_scan_count = 2
                native_decode_fast_path_saved_reload_guard_scan_count = 2
            else:
                (
                    prefill_attention_plan_stats,
                    prefill_attention_plan_snapshot_stale,
                    prefill_attention_plan_duration_s,
                ) = self._get_virtual_context_attention_plan_debug_snapshot(
                    phase="prefill",
                    request_ids=scheduled_prefill_plan_req_ids,
                    query_tokens=num_prefill_tokens,
                    prefix="prefill_attention_plan",
                    staging_budget_blocks=(
                        self.virtual_context_staging_reserved_blocks
                    ),
                    dma_budget_bytes=(
                        self._virtual_context_ring_dma_budget_bytes(
                            virtual_context_ring_prefill_admission_tokens_effective
                        )
                    ),
                    refresh=virtual_context_debug_snapshot_due,
                )
                iteration_debug.update(prefill_attention_plan_stats)
                (
                    decode_attention_plan_stats,
                    decode_attention_plan_snapshot_stale,
                    decode_attention_plan_duration_s,
                ) = self._get_virtual_context_attention_plan_debug_snapshot(
                    phase="decode",
                    request_ids=scheduled_decode_plan_req_ids,
                    query_tokens=num_decode_tokens,
                    prefix="decode_attention_plan",
                    staging_budget_blocks=(
                        self.virtual_context_staging_reserved_blocks
                    ),
                    refresh=virtual_context_debug_snapshot_due,
                )
                iteration_debug.update(decode_attention_plan_stats)
                virtual_context_attention_plan_debug_duration_s += (
                    prefill_attention_plan_duration_s
                    + decode_attention_plan_duration_s
                )
                virtual_context_attention_plan_debug_snapshot_hits += int(
                    prefill_attention_plan_snapshot_stale
                ) + int(decode_attention_plan_snapshot_stale)
                virtual_context_attention_plan_debug_snapshot_refreshes += int(
                    not prefill_attention_plan_snapshot_stale
                ) + int(not decode_attention_plan_snapshot_stale)

                reload_guard_debug_start = time.perf_counter()
                iteration_debug.update(
                    self.kv_cache_manager.get_virtual_context_reload_guard_debug_stats(
                        phase="prefill",
                        request_ids=scheduled_prefill_plan_req_ids,
                        prefix="prefill_reload_guard",
                    )
                )
                iteration_debug.update(
                    self.kv_cache_manager.get_virtual_context_reload_guard_debug_stats(
                        phase="decode",
                        request_ids=scheduled_decode_plan_req_ids,
                        prefix="decode_reload_guard",
                    )
                )
                virtual_context_reload_guard_debug_duration_s += (
                    time.perf_counter() - reload_guard_debug_start
                )
            iteration_debug.update(
                self.kv_cache_manager.get_virtual_context_residency_budget_debug_stats(
                    free_hbm_blocks=residency_budget_free_hbm_blocks,
                    decode_hotset_reserved_blocks=(
                        residency_budget_decode_hotset_reserved_blocks
                    ),
                    decode_hotset_used_blocks=(
                        residency_budget_decode_hotset_used_blocks
                    ),
                    prefill_required_growth_blocks=(
                        residency_budget_prefill_required_growth_blocks
                    ),
                    staging_reserved_blocks=residency_budget_staging_reserved_blocks,
                )
            )
            global_debug_snapshot_reused = False
            if native_decode_control_fast_path_used:
                iteration_debug.update(self._virtual_context_debug_stats_cache)
                virtual_context_debug_snapshot_age_iterations = (
                    self._iteration_debug_index
                    - self._virtual_context_debug_stats_cache_iteration
                )
                global_debug_snapshot_reused = True
            elif (
                not virtual_context_debug_snapshot_due
                and bool(self._virtual_context_debug_stats_cache)
            ):
                iteration_debug.update(self._virtual_context_debug_stats_cache)
                virtual_context_debug_snapshot_age_iterations = (
                    self._iteration_debug_index
                    - self._virtual_context_debug_stats_cache_iteration
                )
                global_debug_snapshot_reused = True
            else:
                global_debug_start = time.perf_counter()
                virtual_context_debug_stats = (
                    self.kv_cache_manager.get_virtual_context_debug_stats()
                )
                virtual_context_global_debug_duration_s += (
                    time.perf_counter() - global_debug_start
                )
                iteration_debug.update(virtual_context_debug_stats)
                if virtual_context_debug_stats:
                    self._virtual_context_debug_stats_cache = dict(
                        virtual_context_debug_stats
                    )
                    self._virtual_context_debug_stats_cache_iteration = (
                        self._iteration_debug_index
                    )
                virtual_context_debug_snapshot_age_iterations = 0

            iteration_debug.update(
                {
                    "virtual_context_native_decode_control_fast_path_enabled": int(
                        self.virtual_context_native_decode_control_fast_path
                    ),
                    "virtual_context_native_decode_control_fast_path_candidate": int(
                        native_decode_control_fast_path_candidate
                    ),
                    "virtual_context_native_decode_control_fast_path_used": int(
                        native_decode_control_fast_path_used
                    ),
                    "virtual_context_native_decode_reload_guard_fast_path_count": (
                        self._native_decode_reload_guard_fast_path_count
                    ),
                    "virtual_context_native_decode_reload_guard_full_scan_count": (
                        self._native_decode_reload_guard_full_scan_count
                    ),
                    "virtual_context_native_decode_handoff_capacity_precheck_count": (
                        self._native_decode_handoff_capacity_precheck_count
                    ),
                    "virtual_context_native_decode_handoff_plan_avoided_page_count": (
                        self._native_decode_handoff_plan_avoided_page_count
                    ),
                    (
                        "virtual_context_native_decode_handoff_reclaim_"
                        "feasibility_check_count"
                    ): self._native_decode_handoff_reclaim_feasibility_check_count,
                    "virtual_context_native_decode_handoff_reclaim_infeasible_count": (
                        self._native_decode_handoff_reclaim_infeasible_count
                    ),
                    (
                        "virtual_context_native_decode_handoff_reclaim_"
                        "infeasible_cache_hit_count"
                    ): (
                        self._native_decode_handoff_reclaim_infeasible_cache_hit_count
                    ),
                    (
                        "virtual_context_native_decode_handoff_reclaim_"
                        "infeasible_cache_size"
                    ): len(
                        self._native_decode_handoff_reclaim_infeasible_cache
                    ),
                    (
                        "virtual_context_native_decode_handoff_reclaim_"
                        "avoided_partial_blocks"
                    ): self._native_decode_handoff_reclaim_avoided_partial_blocks,
                    "virtual_context_native_decode_control_fast_path_eligible_request_count": (
                        native_decode_fast_path_eligible_request_count
                    ),
                    "virtual_context_native_decode_control_fast_path_saved_request_scan_count": (
                        native_decode_fast_path_saved_request_scan_count
                    ),
                    "virtual_context_native_decode_control_fast_path_saved_attention_plan_scan_count": (
                        native_decode_fast_path_saved_attention_plan_scan_count
                    ),
                    "virtual_context_native_decode_control_fast_path_saved_reload_guard_scan_count": (
                        native_decode_fast_path_saved_reload_guard_scan_count
                    ),
                    "virtual_context_debug_snapshot_stale": int(
                        global_debug_snapshot_reused
                    ),
                    "virtual_context_debug_snapshot_age_iterations": (
                        virtual_context_debug_snapshot_age_iterations
                    ),
                    "virtual_context_request_debug_duration_s": (
                        virtual_context_request_debug_duration_s
                    ),
                    "virtual_context_debug_snapshot_interval": (
                        self.virtual_context_debug_snapshot_interval
                    ),
                    "virtual_context_debug_snapshot_due": int(
                        virtual_context_debug_snapshot_due
                    ),
                    "virtual_context_request_debug_snapshot_hits": (
                        virtual_context_request_debug_snapshot_hits
                    ),
                    "virtual_context_request_debug_snapshot_refreshes": (
                        virtual_context_request_debug_snapshot_refreshes
                    ),
                    "virtual_context_attention_plan_debug_snapshot_hits": (
                        virtual_context_attention_plan_debug_snapshot_hits
                    ),
                    "virtual_context_attention_plan_debug_snapshot_refreshes": (
                        virtual_context_attention_plan_debug_snapshot_refreshes
                    ),
                    "virtual_context_attention_plan_debug_duration_s": (
                        virtual_context_attention_plan_debug_duration_s
                    ),
                    "virtual_context_reload_guard_debug_duration_s": (
                        virtual_context_reload_guard_debug_duration_s
                    ),
                    "virtual_context_global_debug_duration_s": (
                        virtual_context_global_debug_duration_s
                    ),
                    "virtual_context_tenant_kv_debug_duration_s": (
                        virtual_context_tenant_kv_debug_duration_s
                    ),
                    "virtual_context_debug_total_duration_s": (
                        time.perf_counter() - virtual_context_debug_block_start
                    ),
                }
            )
        elif (
            self._lightweight_timeline_enabled
            or detailed_iteration_debug_sampled
        ):
            iteration_debug, request_debug = (
                self._build_lightweight_timeline_debug(
                    scheduled_timestamp=scheduled_timestamp,
                    waiting_queue_len_before=waiting_queue_len_before,
                    running_reqs_before=running_reqs_before,
                    num_scheduled_tokens=num_scheduled_tokens,
                    scheduled_new_reqs=scheduled_new_reqs,
                    scheduled_resumed_reqs=scheduled_resumed_reqs,
                    scheduled_running_reqs=scheduled_running_reqs,
                    scheduled_residency_rehydrated_req_ids=(
                        scheduled_residency_rehydrated_req_ids
                    ),
                    scheduled_phase=scheduled_phase,
                )
            )
            if (
                detailed_iteration_debug_sampled
                and total_num_scheduled_tokens == 0
            ):
                iteration_debug[
                    "detailed_debug_zero_work_snapshot_suppressed"
                ] = 1
        pd_prefill_mode = "native"
        if scheduled_phase == "prefill":
            for req_id, scheduled_tokens in num_scheduled_tokens.items():
                request = self.requests[req_id]
                prefill_remaining = self._pd_prefill_lane_remaining_tokens(
                    request,
                )
                if min(scheduled_tokens, prefill_remaining) <= 0:
                    continue
                if (
                    self._ttft_prefill_execution_mode(
                        request,
                        virtual_context_native_prefill_promotion_request_ids,
                    )
                    == "ring"
                ):
                    pd_prefill_mode = "ring"
                    break

        if iteration_debug is not None:
            ttft_prefill_execution_tokens = (
                total_num_scheduled_tokens
                if scheduled_phase == "prefill"
                else 0
            )
            ttft_prefill_quantum_tokens = (
                self.ttft_prefill_chunk_scheduler.quantum_tokens
                if self.enable_ttft_prefill_chunk_scheduling
                else 0
            )
            ttft_prefill_native_expansion_applied = int(
                self.enable_ttft_prefill_chunk_scheduling
                and self.ttft_prefill_native_chunk_max_tokens > 0
                and pd_prefill_mode == "native"
                and ttft_prefill_execution_tokens
                > ttft_prefill_quantum_tokens
            )
            iteration_debug.update(
                {
                    "ttft_prefill_execution_mode": pd_prefill_mode,
                    "isolation_prefill_batch_max_requests": getattr(
                        self, "isolation_prefill_batch_max_requests", 0),
                    "isolation_decode_batch_max_requests": getattr(
                        self, "isolation_decode_batch_max_requests", 0),
                    "isolation_phase_request_limit": isolation_phase_request_limit,
                    "isolation_phase_waiting_scan": int(getattr(
                        self, "isolation_phase_waiting_scan", False)),
                    "isolation_waiting_decode_fcfs_scope": int(getattr(
                        self, "isolation_waiting_decode_fcfs_scope", False)),
                    "isolation_waiting_decode_prefill_fcfs_bypass_count": (
                        isolation_waiting_decode_prefill_fcfs_bypass_count
                    ),
                    "isolation_waiting_decode_prefill_fcfs_first_id": (
                        isolation_waiting_decode_prefill_fcfs_first_id
                    ),
                    "isolation_waiting_decode_resume_owner_block_count": (
                        isolation_waiting_decode_resume_owner_block_count
                    ),
                    "isolation_phase_waiting_skip_count": (
                        isolation_phase_waiting_skip_count
                    ),
                    "isolation_phase_waiting_skipped_ids": ",".join(
                        isolation_phase_waiting_skipped_ids),
                    "isolation_phase_request_limit_hit": int(
                        isolation_phase_request_limit_hit
                        or ttft_prefill_frontier.stop_reason == "isolation_phase_request_limit"),
                    "isolation_prefill_fit_next_chunk": int(
                        self.isolation_prefill_fit_next_chunk),
                    "isolation_d2h_exact_plan_progress": int(getattr(
                        self, "isolation_required_d2h_progress", False)),
                    "isolation_native_soft_headroom_progress": int(getattr(
                        self, "isolation_native_soft_headroom_progress", False)),
                    "isolation_prefill_batch_kv_budget": int(getattr(
                        self, "isolation_prefill_batch_kv_budget", False)),
                    **getattr(self, "_isolation_prefill_batch_budget_debug", {}),
                    "isolation_waiting_decode_reclaim_enabled": int(getattr(
                        self, "isolation_waiting_decode_reclaim", False)),
                    **getattr(self, "_isolation_waiting_decode_reclaim_debug", {}),
                    "isolation_ring_query_expansion_enabled": int(
                        self.isolation_ring_query_expansion
                    ),
                    "isolation_ring_query_max_tokens": self.isolation_ring_query_max_tokens,
                    "isolation_ring_query_scratch_budget_bytes": self.isolation_ring_query_scratch_bytes,
                    "isolation_ring_query_expansion_applied": int(
                        self.isolation_ring_query_expansion
                        and pd_prefill_mode == "ring"
                        and ttft_prefill_execution_tokens > ttft_prefill_quantum_tokens
                    ),
                    "ttft_prefill_execution_tokens": (
                        ttft_prefill_execution_tokens
                    ),
                    "ttft_prefill_native_chunk_max_tokens": (
                        self.ttft_prefill_native_chunk_max_tokens
                    ),
                    "ttft_prefill_native_chunk_expansion_applied": (
                        ttft_prefill_native_expansion_applied
                    ),
                    "ttft_prefill_native_chunk_expansion_tokens": (
                        max(
                            0,
                            ttft_prefill_execution_tokens
                            - ttft_prefill_quantum_tokens,
                        )
                        if ttft_prefill_native_expansion_applied
                        else 0
                    ),
                }
            )

        pd_capacity_retry_observed_free_blocks = (
            self.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        pd_capacity_retry_min_free_blocks = 0
        pd_capacity_retry_wake_on_output_progress = False
        pd_capacity_retry_cause = ""
        pd_capacity_retry_blocked = bool(
            total_num_scheduled_tokens == 0
            and virtual_context_decode_handoff_reload_blocked_count > 0
            and virtual_context_decode_handoff_retry_min_free_blocks
            > pd_capacity_retry_observed_free_blocks
            and "scheduled_prefill_fit_d2h_wait"
            not in virtual_context_decode_handoff_reclaim_reasons
        )
        pd_capacity_retry_reason = (
            ",".join(sorted(virtual_context_decode_handoff_reload_reasons))
            if pd_capacity_retry_blocked
            else ""
        )
        if pd_capacity_retry_blocked:
            pd_capacity_retry_min_free_blocks = (
                virtual_context_decode_handoff_retry_min_free_blocks
            )
            pd_capacity_retry_wake_on_output_progress = (
                "live_decode_growth_wait"
                in virtual_context_decode_handoff_reload_reasons
            )
            pd_capacity_retry_cause = (
                "live_decode_growth"
                if virtual_context_decode_handoff_reload_reasons
                == {"live_decode_growth_wait"}
                else "decode_handoff_capacity"
            )
        else:
            prefill_retry_defer_reasons = {
                reason
                for reason, count in prefill_defer_reason_counts.items()
                if count > 0
            }
            if waiting_break_defer_reason:
                prefill_retry_defer_reasons.add(waiting_break_defer_reason)
            no_progress_retry = pd_prefill_no_progress_retry_metadata(
                phase_pure=self.pd_decoupled_no_mix,
                selected_phase=pd_decoupled_selected_phase,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                has_logical_prefill_work=bool(
                    running_phase_counts_before["num_running_prefill_reqs"] > 0
                    or num_waiting_prefill_reqs_before > 0
                ),
                decode_output_pending=self._pd_pending_decode_active(),
                asynchronous_progress_pending=bool(
                    scheduled_prefill_fit_preparation_waiting
                    or prefill_defer_reason_counts[
                        "scheduled_prefill_fit_d2h_wait"
                    ]
                    > 0
                    or "scheduled_prefill_fit_d2h_wait"
                    in virtual_context_decode_handoff_reclaim_reasons
                    or waiting_break_defer_reason
                    == "scheduled_prefill_fit_d2h_wait"
                ),
                observed_free_blocks=pd_capacity_retry_observed_free_blocks,
                defer_reasons=prefill_retry_defer_reasons,
            )
            if no_progress_retry is not None:
                (
                    pd_capacity_retry_min_free_blocks,
                    pd_capacity_retry_reason,
                ) = no_progress_retry
                pd_capacity_retry_blocked = True
                pd_capacity_retry_wake_on_output_progress = True
                if pd_native_active_decode_prefill_capacity_blocked:
                    pd_capacity_retry_cause = "prefill_kv_capacity"
                    pd_capacity_retry_min_free_blocks = max(
                        pd_capacity_retry_min_free_blocks,
                        (
                            pd_native_prefill_retry_min_blocks
                            or pd_native_active_decode_prefill_required_blocks
                        ),
                    )
                else:
                    pd_capacity_retry_cause = "prefill_no_progress"

        if iteration_debug is not None:
            iteration_debug.update(
                {
                    "pd_capacity_retry_blocked": int(
                        pd_capacity_retry_blocked
                    ),
                    "pd_capacity_retry_min_free_blocks": (
                        pd_capacity_retry_min_free_blocks
                    ),
                    "pd_capacity_retry_observed_free_blocks": (
                        pd_capacity_retry_observed_free_blocks
                    ),
                    "pd_capacity_retry_reason": pd_capacity_retry_reason,
                    "pd_capacity_retry_cause": pd_capacity_retry_cause,
                }
            )

        tiered_request_execution_plans = (
            self._build_tiered_request_execution_plans(
                num_scheduled_tokens,
                tiered_explicit_execution_plans,
            )
        )
        if iteration_debug is not None:
            iteration_debug.update(
                {
                    "kv_retention_execution_plan_count": len(
                        tiered_request_execution_plans
                    ),
                    "kv_retention_execution_plan_request_ids": ",".join(
                        entry[0] for entry in tiered_request_execution_plans
                    ),
                    "kv_retention_execution_plan_modes": ",".join(
                        entry[2] for entry in tiered_request_execution_plans
                    ),
                    "kv_retention_execution_plan_expected_reload_pages": sum(
                        entry[3] for entry in tiered_request_execution_plans
                    ),
                }
            )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            residency_deferred_req_ids=residency_deferred_req_ids_for_output,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            iteration_debug=iteration_debug,
            request_debug=request_debug,
            scheduler_iteration_id=self._iteration_debug_index,
            tiered_dma_budget_bytes=(
                self._virtual_context_ring_dma_budget_bytes(
                    virtual_context_ring_prefill_admission_tokens_effective
                )
            ),
            tiered_staging_budget_blocks=(
                self.virtual_context_staging_reserved_blocks
            ),
            tiered_max_prefill_ring_tokens=(
                virtual_context_ring_prefill_admission_tokens_effective
            ),
            tiered_decode_tbt_budget_ms=0.0,
            tiered_pressure_snapshot_request_ids=tuple(
                sorted(virtual_context_pressure_snapshot_request_ids)
            ),
            tiered_proactive_snapshot_pages=tuple(
                sorted(scheduled_prefill_fit_proactive_snapshot_pages)
            ),
            tiered_proactive_snapshot_logical_tokens=tuple(
                sorted(
                    scheduled_prefill_fit_proactive_snapshot_logical_tokens.items()
                )
            ),
            tiered_native_hbm_reload_request_ids=tuple(
                sorted(
                    virtual_context_native_prefill_promotion_request_ids
                )
            ),
            tiered_request_execution_plans=(
                tiered_request_execution_plans
            ),
            pd_prefill_mode=pd_prefill_mode,
            scheduled_phase=scheduled_phase,
            pd_ttft_handoff_request_ids=tuple(pd_ttft_handoff_request_ids),
            pd_capacity_retry_blocked=pd_capacity_retry_blocked,
            pd_capacity_retry_min_free_blocks=(
                pd_capacity_retry_min_free_blocks
                if pd_capacity_retry_blocked
                else 0
            ),
            pd_capacity_retry_observed_free_blocks=(
                pd_capacity_retry_observed_free_blocks
                if pd_capacity_retry_blocked
                else 0
            ),
            pd_capacity_retry_reason=pd_capacity_retry_reason,
            pd_capacity_retry_cause=pd_capacity_retry_cause,
            pd_capacity_retry_wake_on_output_progress=(
                pd_capacity_retry_wake_on_output_progress
            ),
        )
        self._prepare_pd_output_placeholder_ownership(scheduler_output)
        self._iteration_debug_index += 1

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        self._commit_pd_output_placeholder_ownership(scheduler_output)
        if getattr(self, "isolation_waiting_decode_reclaim", False):
            self._commit_isolation_waiting_decode_retirements(scheduler_output)
        return scheduler_output

    def _expected_pd_output_placeholder_counts(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, int]:
        if (
            not self.pd_output_placeholder_ownership_enabled
            or scheduler_output.total_num_scheduled_tokens <= 0
        ):
            return {}

        phase = scheduler_output.scheduled_phase
        if phase not in {"decode", "prefill"}:
            raise OutputPlaceholderOwnershipError(
                "The phase-local owner protocol requires a phase-pure "
                f"SchedulerOutput, got phase={phase!r}."
            )

        final_prefill_ids = set(
            scheduler_output.pd_ttft_handoff_request_ids
        )
        expected_counts: dict[str, int] = {}
        for req_id, scheduled_tokens in (
            scheduler_output.num_scheduled_tokens.items()
        ):
            if scheduled_tokens <= 0:
                continue
            request = self.requests[req_id]
            if (
                request.sampling_params is None
                or request.pooling_params is not None
            ):
                continue
            if phase != "decode" and req_id not in final_prefill_ids:
                continue
            expected_counts[req_id] = 1 + len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            )
        if phase == "prefill" and set(expected_counts) != final_prefill_ids:
            raise OutputPlaceholderOwnershipError(
                "Final-Prefill handoff IDs and output claims differ: "
                f"handoff_ids={tuple(sorted(final_prefill_ids))}, "
                f"claim_ids={tuple(sorted(expected_counts))}."
            )
        return expected_counts

    def _prepare_pd_output_placeholder_ownership(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Prepare one atomic async-counter/Future ownership transaction."""

        if not self.pd_output_placeholder_ownership_enabled:
            return

        scheduler_output.pd_output_ownership_enabled = True
        source_generations: dict[str, int] = {}
        for req_id in scheduler_output.scheduled_cached_reqs.all_token_ids:
            request = self.requests.get(req_id)
            if request is None or request.num_output_tokens <= 0:
                continue
            source_generation = (
                request.pd_output_placeholder_ledger.last_committed_generation
            )
            if source_generation <= 0:
                raise OutputPlaceholderOwnershipError(
                    "A phase-local token-history transfer has no committed "
                    f"owner generation: request={req_id}, "
                    f"phase={scheduler_output.scheduled_phase}, "
                    f"output_tokens={request.num_output_tokens}."
                )
            source_generations[req_id] = source_generation
        scheduler_output.pd_output_token_source_generations = source_generations
        expected_counts = self._expected_pd_output_placeholder_counts(
            scheduler_output
        )
        for req_id in expected_counts:
            active_owner = self.requests[
                req_id
            ].pd_output_placeholder_ledger.active
            if active_owner is not None:
                raise OutputPlaceholderOwnershipError(
                    "Cannot prepare output positions while an older Future "
                    "still owns the request: "
                    f"request={req_id}, "
                    f"request_epoch={active_owner.request_epoch}, "
                    f"generation={active_owner.generation}, "
                    f"phase={active_owner.phase}, "
                    f"cancelled={active_owner.cancelled}."
                )
        scheduler_output.pd_output_placeholder_expected_counts = (
            expected_counts
        )

    def _commit_pd_output_placeholder_ownership(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Commit ledgers from the exact counters advanced for this output."""

        if not scheduler_output.pd_output_ownership_enabled:
            return

        expected_counts = (
            scheduler_output.pd_output_placeholder_expected_counts
        )
        counter_deltas = (
            scheduler_output.pd_output_placeholder_counter_deltas
        )
        if counter_deltas != expected_counts:
            raise OutputPlaceholderOwnershipError(
                "Async placeholder accounting differs from the prepared "
                "Future claims: "
                f"expected={expected_counts}, deltas={counter_deltas}, "
                f"phase={scheduler_output.scheduled_phase}."
            )

        owners: dict[str, tuple[int, int, str, int]] = {}
        for req_id, placeholder_count in expected_counts.items():
            request = self.requests[req_id]
            owner = request.pd_output_placeholder_ledger.reserve(
                phase=scheduler_output.scheduled_phase,
                placeholder_count=placeholder_count,
            )
            owners[req_id] = (
                owner.request_epoch,
                owner.generation,
                owner.phase,
                owner.placeholder_count,
            )
        scheduler_output.pd_output_placeholder_owners = owners

    def _attach_pd_output_placeholder_ownership(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Compatibility helper for tests and isolated scheduler callers."""

        self._prepare_pd_output_placeholder_ownership(scheduler_output)
        self._commit_pd_output_placeholder_ownership(scheduler_output)

    def bind_pd_output_execution_context(
        self,
        scheduler_output: SchedulerOutput,
        execution_key: str,
    ) -> None:
        """Attach committed token history when a request changes runner slot."""

        if not scheduler_output.pd_output_ownership_enabled:
            return

        cached_req_ids = set(scheduler_output.scheduled_cached_reqs.req_ids)
        for req_id in scheduler_output.num_scheduled_tokens:
            previous_key = self._pd_output_last_execution_key_by_req.get(req_id)
            if previous_key == execution_key:
                continue
            request = self.requests.get(req_id)
            if (
                request is not None
                and req_id in cached_req_ids
                and request.num_output_tokens > 0
            ):
                output_ledger = request.pd_output_placeholder_ledger
                source_generation = output_ledger.last_committed_generation
                if source_generation <= 0:
                    raise OutputPlaceholderOwnershipError(
                        "A phase-local execution-context transfer has output "
                        "tokens but no committed source generation: "
                        f"request={req_id}, previous_key={previous_key!r}, "
                        f"execution_key={execution_key!r}."
                    )
                scheduler_output.scheduled_cached_reqs.all_token_ids[req_id] = (
                    request.all_token_ids.copy()
                )
                scheduler_output.pd_output_token_source_generations[req_id] = (
                    source_generation
                )
            self._pd_output_last_execution_key_by_req[req_id] = execution_key

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        output_owner = request.pd_output_placeholder_ledger.active
        if output_owner is not None and not output_owner.cancelled:
            raise OutputPlaceholderOwnershipError(
                "Cannot preempt a request while a model Future owns its "
                "output positions. The caller must wait for that Future or "
                "explicitly cancel the ownership before forced preemption: "
                f"request={request.request_id}, "
                f"generation={output_owner.generation}, "
                f"phase={output_owner.phase}, "
                f"placeholder_count={output_owner.placeholder_count}."
            )
        if getattr(
            request,
            "kv_retention_decode_continuation_lease_active",
            False,
        ):
            raise RuntimeError(
                "Cannot reset/recompute a DRAM-backed Decode continuation: "
                f"request={request.request_id}, "
                "state="
                f"{getattr(request, 'virtual_context_deferred_state', '')!r}."
            )
        preempt_phase = (
            "prefill" if self._debug_has_prefill_remaining(request) else "decode"
        )
        num_computed_tokens_before_reset = request.num_computed_tokens
        num_tokens_before_reset = request.num_tokens
        if self.pd_decoupled_no_mix and request.num_output_tokens > 0:
            request.pd_recompute_frontier_tokens = max(
                request.pd_recompute_frontier_tokens,
                num_computed_tokens_before_reset,
            )
        request.preempted_reset_count += 1
        request.last_preempt_iteration_id = self._iteration_debug_index
        request.last_preempt_phase = preempt_phase
        request.last_preempt_num_computed_tokens_before_reset = (
            num_computed_tokens_before_reset
        )
        request.last_preempt_num_tokens_before_reset = num_tokens_before_reset
        request.preempted_tokens_reset_total += num_computed_tokens_before_reset
        if preempt_phase == "prefill":
            request.preempted_prefill_reset_count += 1
        else:
            request.preempted_decode_reset_count += 1
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.spec_token_ids.clear()
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        scheduled_phase: str,
        phase_handoff_token_refresh_req_ids: list[str],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            previous_phase = self.prev_step_scheduled_phase_by_req.get(req_id)
            phase_local_ownership_changed = _pd_phase_local_ownership_changed(
                self.pd_phase_local_model_runner_enabled,
                scheduled_in_prev_step,
                previous_phase,
                scheduled_phase,
            )
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if req.virtual_context_deferred_state in {
                "decode_hbm_reload_pending",
                "hbm_reload_pending",
            }:
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step or phase_local_ownership_changed:
                all_token_ids[req_id] = req.all_token_ids.copy()
            if phase_local_ownership_changed:
                phase_handoff_token_refresh_req_ids.append(req_id)
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # Check remote cache first
        if self.ec_connector is not None:
            remote_cache_has_item = self.ec_connector.has_caches(request)
        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                num_new_tokens = start_pos - num_computed_tokens
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and remote_cache_has_item[i]:
                mm_hashes_to_schedule.add(request.mm_features[i].identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        # PERF: in case of chunked prefill,
        # request might not include any new tokens.
        # Therefore, we might introduce some additional
        # cycle to fill in the bitmask, which could be a big no-op.
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id)) and req.use_structured_output
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        output_update_timestamp = (
            time.monotonic()
            if self.tbt_slo_observability_enabled
            else 0.0
        )
        output_iteration_id = int(scheduler_output.scheduler_iteration_id)
        try:
            tiered_execution_plans = parse_tiered_execution_plan_entries(
                scheduler_output.tiered_request_execution_plans
            )
        except RuntimeError as exc:
            raise TieredAllocatorInvariantError(
                "SchedulerOutput carried an invalid tiered execution plan"
            ) from exc
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            execution_plan = tiered_execution_plans.get(req_id)
            if execution_plan is not None:
                plan_generation, plan_mode, expected_reload_pages = (
                    execution_plan
                )
                if (
                    request.kv_retention_execution_plan_generation
                    != plan_generation
                    or request.kv_retention_execution_plan_mode != plan_mode
                    or request.kv_retention_execution_plan_expected_reload_pages
                    != expected_reload_pages
                ):
                    raise TieredAllocatorInvariantError(
                        "Tiered execution result no longer owns the request "
                        "generation: "
                        f"request={req_id}, output={execution_plan!r}, "
                        "active=("
                        f"{request.kv_retention_execution_plan_generation}, "
                        f"{request.kv_retention_execution_plan_mode!r}, "
                        f"{request.kv_retention_execution_plan_expected_reload_pages}"
                        ")."
                    )
                request.kv_retention_execution_plan_acked_generation = (
                    plan_generation
                )
                self._tiered_execution_plan_ack_count += 1
                if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                    self._tiered_execution_plan_failed_load_ack_count += 1

            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # The Future is resolved, so its tiered lease above must be
                # acknowledged before this request is recomputed/rescheduled.
                continue

            if request.virtual_context_deferred_state in {
                "hbm_reload_pending",
                "decode_hbm_reload_pending",
            }:
                remaining_reload_pages = (
                    self.kv_cache_manager.plan_virtual_context_hbm_reload_pages(
                        request_ids={req_id},
                    )
                )
                if remaining_reload_pages:
                    error_message = (
                        "native decode/prefill output arrived before all "
                        "DRAM-resident KV pages received an HBM reload ACK: "
                        f"request={req_id}, "
                        f"remaining_pages={len(remaining_reload_pages)}"
                    )
                    request.stop_reason = error_message
                    self.finish_requests(
                        req_id, RequestStatus.FINISHED_ERROR
                    )
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=[],
                            finish_reason=request.get_finished_reason(),
                            stop_reason=error_message,
                            events=request.take_events(),
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                        )
                    )
                    debug = scheduler_output.iteration_debug
                    if debug is None:
                        debug = scheduler_output.iteration_debug = {}
                    debug[
                        "virtual_context_reload_ack_incomplete_failure"
                    ] = 1
                    debug[
                        "virtual_context_reload_ack_incomplete_pages"
                    ] = len(remaining_reload_pages)
                    continue

                request.virtual_context_deferred_state = (
                    "decode_hbm_reload_resumed"
                    if request.virtual_context_deferred_state
                    == "decode_hbm_reload_pending"
                    else "hbm_reload_resumed"
                )
                request.virtual_context_hbm_reloaded_tokens += (
                    request.virtual_context_hbm_reload_planned_tokens
                )
                request.virtual_context_hbm_reload_planned_tokens = 0
                request.virtual_context_deferred_logical_computed_tokens = 0
                request.num_external_computed_tokens = 0

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )
            if not self._resolve_pd_output_placeholder_ownership(
                request,
                scheduler_output,
                generated_token_ids,
            ):
                continue
            tbt_decode_output = (
                self.tbt_slo_observability_enabled
                and generated_token_ids
                and request.num_computed_tokens >= request.num_prompt_tokens
            )
            if tbt_decode_output:
                self.tbt_decode_scheduler.mark_prefill_completed(
                    request,
                    completed_at=output_update_timestamp,
                    iteration_id=output_iteration_id,
                )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
                if tbt_decode_output and new_token_ids:
                    self.tbt_decode_scheduler.mark_tokens_served(
                        request,
                        served_at=output_update_timestamp,
                        num_tokens=len(new_token_ids),
                    )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            if stopped:
                if self.vllm_config.model_config.enable_return_routed_experts:
                    kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
                    block_ids = kv_blocks.get_block_ids()[0]
                    num_tokens = request.num_tokens - 1

                    # compute slot mapping
                    block_ids_array = np.array(block_ids, dtype=np.int32)
                    num_blocks = len(block_ids)
                    block_size = self.block_size

                    # generate block offsets
                    block_offsets = np.arange(0, block_size)

                    # compute slot mapping: slot = block_id * block_size + offset
                    slot_mapping = (
                        block_offsets.reshape((1, block_size))
                        + block_ids_array.reshape((num_blocks, 1)) * block_size
                    ).flatten()[:num_tokens]

                    routed_experts = self.routed_experts_reader.get_routed_experts(
                        indices=slot_mapping
                    )
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _resolve_pd_output_placeholder_ownership(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        generated_token_ids: list[int],
    ) -> bool:
        """Commit only the exact Future that reserved these output slots.

        Returning ``False`` means an explicitly cancelled Future completed and
        its result must be discarded. Any other mismatch is an invariant
        violation; retrying it would create two producers for one output slot.
        """
        if not scheduler_output.pd_output_ownership_enabled:
            return True

        owner_record = scheduler_output.pd_output_placeholder_owners.get(
            request.request_id
        )
        if owner_record is None:
            if generated_token_ids:
                raise OutputPlaceholderOwnershipError(
                    "A phase-local Future produced sampled output without "
                    "owning an output position: "
                    f"request={request.request_id}, "
                    f"phase={scheduler_output.scheduled_phase}, "
                    f"generated_count={len(generated_token_ids)}."
                )
            return True

        request_epoch, generation, owner_phase, reserved_count = owner_record
        if owner_phase != scheduler_output.scheduled_phase:
            raise OutputPlaceholderOwnershipError(
                "SchedulerOutput phase differs from its output owner: "
                f"request={request.request_id}, "
                f"request_epoch={request_epoch}, generation={generation}, "
                f"owner_phase={owner_phase}, "
                f"scheduled_phase={scheduler_output.scheduled_phase}."
            )
        active_owner = request.pd_output_placeholder_ledger.active
        if (
            active_owner is None
            or active_owner.request_epoch != request_epoch
            or active_owner.placeholder_count != reserved_count
        ):
            raise OutputPlaceholderOwnershipError(
                "SchedulerOutput reservation differs from the request ledger: "
                f"request={request.request_id}, "
                f"request_epoch={request_epoch}, generation={generation}, "
                f"output_reserved_count={reserved_count}, "
                "ledger_request_epoch="
                f"{active_owner.request_epoch if active_owner else 0}, "
                "ledger_reserved_count="
                f"{active_owner.placeholder_count if active_owner else 0}."
            )

        scheduled_spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(request.request_id)
        )
        num_draft_tokens = len(scheduled_spec_token_ids or ())
        num_accepted = max(0, len(generated_token_ids) - 1)
        num_rejected = max(0, num_draft_tokens - num_accepted)
        commit_result = request.pd_output_placeholder_ledger.resolve(
            request_epoch=request_epoch,
            generation=generation,
            phase=owner_phase,
            generated_count=len(generated_token_ids),
            rejected_count=num_rejected,
        )
        if not commit_result:
            request.discard_latest_async_tokens = False
        return commit_result

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        # A new Prefill can change the aggregate reclaim capacity for every
        # blocked Decode handoff, even before it owns resident blocks.
        self._native_decode_handoff_reclaim_infeasible_cache.clear()
        if request.arrival_iteration_id < 0:
            request.arrival_iteration_id = self._iteration_debug_index
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str], finished_status: RequestStatus
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        if set(request_ids).intersection(getattr(self, "_kv_retention_parked_requests", {})):
            raise RuntimeError("parked Prefill must acknowledge Worker cleanup before free")

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
        # Second pass: set status and free requests
        for request in valid_requests:
            # An external abort/error revokes the exact in-flight generation.
            # If connector cleanup delays request deletion, its late result can
            # still consume this tombstone without committing a token.
            request.pd_output_placeholder_ledger.cancel_active()
            request.status = finished_status
            self._free_request(request)

    def fail_scheduled_requests(
        self,
        scheduler_output: SchedulerOutput,
        error_message: str,
    ) -> dict[int, EngineCoreOutputs]:
        """Fail only requests owned by a rejected model execution.

        ``schedule()`` has already advanced request/KV state before the model
        future is recovered.  An expected tiered-backend eligibility failure
        therefore cannot be handled by merely dropping the future: the
        affected requests must be removed from the scheduler and their KV
        allocations must be released.  This helper mirrors the existing
        request-scoped KV-load failure path while preserving unrelated P/D
        lane requests.

        Unexpected CUDA, sampling, or scheduler errors must not call this
        method; those remain fatal EngineCore errors.
        """
        scheduled_request_ids = tuple(scheduler_output.num_scheduled_tokens)
        failed_requests = [
            request
            for request_id in scheduled_request_ids
            if (request := self.requests.get(request_id)) is not None
            and not request.is_finished()
        ]
        if not failed_requests:
            return {}

        for request in failed_requests:
            request.stop_reason = error_message

        self.finish_requests(
            (request.request_id for request in failed_requests),
            RequestStatus.FINISHED_ERROR,
        )

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        finished_by_client: dict[int, set[str]] = defaultdict(set)
        for request in failed_requests:
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    stop_reason=error_message,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                    num_cached_tokens=request.num_cached_tokens,
                )
            )
            finished_by_client[request.client_index].add(request.request_id)

        # ``finish_requests`` also records these ids for include_finished_set.
        # Attach and consume only the ids handled here so a later successful
        # phase output does not emit a duplicate terminal notification.
        if self.finished_req_ids_dict is not None:
            for client_index, request_ids in finished_by_client.items():
                self.finished_req_ids_dict[client_index].difference_update(
                    request_ids
                )

        debug = scheduler_output.iteration_debug
        if debug is None:
            debug = scheduler_output.iteration_debug = {}
        debug.update(
            {
                "tiered_backend_request_local_failure": 1,
                "tiered_backend_request_local_failed_requests": len(
                    failed_requests
                ),
                "tiered_backend_request_local_error": error_message,
            }
        )
        return {
            client_index: EngineCoreOutputs(
                outputs=client_outputs,
                finished_requests=finished_by_client[client_index],
            )
            for client_index, client_outputs in outputs.items()
        }

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        assert request.is_finished()

        # Natural completion and explicit abort share this authority. Either
        # can change free HBM and the complete Prefill victim set.
        self._native_decode_handoff_reclaim_infeasible_cache.clear()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.ttft_prefill_chunk_scheduler.forget_request(request_id)
        self.tbt_decode_scheduler.forget_request(request_id)
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        request_id = request.request_id
        self._pd_output_last_execution_key_by_req.pop(request_id, None)
        self._virtual_context_request_debug_stats_cache.pop(request_id, None)
        self._virtual_context_request_debug_stats_cache_iteration.pop(
            request_id, None
        )
        stale_plan_keys = [
            cache_key
            for cache_key in self._virtual_context_attention_plan_debug_stats_cache
            if request_id in cache_key[1]
        ]
        for cache_key in stale_plan_keys:
            self._virtual_context_attention_plan_debug_stats_cache.pop(
                cache_key, None
            )
            self._virtual_context_attention_plan_debug_stats_cache_iteration.pop(
                cache_key, None
            )
        del self.requests[request_id]

    def get_num_unfinished_requests(self) -> int:
        return (len(self.waiting) + len(self.running)
                + len(getattr(self, "_kv_retention_parked_requests", {})))

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if getattr(self, "_kv_retention_parked_requests", {}):
            # A reset is not a completed parked-copy/read acknowledgement.
            return False
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                request.pd_output_placeholder_ledger.cancel_active()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_phase_by_req.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        close_scheduler_timing(self)
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def _update_connector_prefix_cache_stats(self, request: Request) -> None:
        if self.connector_prefix_cache_stats is None:
            return

        self.connector_prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=request.num_external_computed_tokens,
            preempted=request.num_preemptions > 0,
        )

    def _make_connector_prefix_cache_stats(self) -> PrefixCacheStats | None:
        if self.connector_prefix_cache_stats is None:
            return None
        stats = self.connector_prefix_cache_stats
        self.connector_prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens
            deferred_target_tokens = (
                request.virtual_context_deferred_logical_computed_tokens
            )
            if deferred_target_tokens > 0:
                loaded_tokens = min(num_computed_tokens, deferred_target_tokens)
                request.virtual_context_hbm_reloaded_tokens += loaded_tokens
                if loaded_tokens >= deferred_target_tokens:
                    request.virtual_context_resume_without_recompute_count += 1
                    request.virtual_context_deferred_state = "reloaded"
                else:
                    request.virtual_context_deferred_state = "partial_reloaded"

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Async loading. If num_computed_tokens is set it implies we
                # already processed some block failures for it in a prior step
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
