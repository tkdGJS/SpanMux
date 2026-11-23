# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import math
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.output_placeholder import (
    OutputPlaceholderLedger,
    OutputPlaceholderOwnershipError,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None
        self.tenant_id = "default"
        self.tenant_session_id = "default"
        self.ttft_prefill_deadline_ms: float | None = None
        self.tbt_slo_ms: float | None = None
        self.tbt_last_token_at: float | None = None
        self.tbt_last_observed_ms: float | None = None
        self.tbt_recent_gaps_ms: list[float] = []
        self.tbt_service_count = 0
        # Experimental per-request scheduler lifecycle counters used only by
        # debug JSONL metrics.
        self.arrival_iteration_id = -1
        self.first_admitted_iteration_id = -1
        self.first_prefill_iteration_id = -1
        self.last_prefill_iteration_id = -1
        self.first_decode_iteration_id = -1
        # Set only after the first sampled token confirms prefill completion.
        # The decode-ready FCFS policy uses this instead of request arrival.
        self.prefill_completed_at: float | None = None
        self.prefill_completed_iteration_id = -1
        self.num_prefill_chunks = 0
        self.deferred_iterations_count = 0
        self.deferred_by_kv_capacity_count = 0
        self.deferred_by_max_tokens_count = 0
        self.deferred_by_max_seqs_count = 0
        self.deferred_by_policy_count = 0
        self.deferred_by_preemption_count = 0
        self.running_kv_capacity_miss_count = 0
        self.running_prefill_kv_capacity_miss_count = 0
        self.running_decode_kv_capacity_miss_count = 0
        self.last_running_kv_capacity_miss_iteration_id = -1
        self.last_running_kv_capacity_miss_phase = ""
        self.last_running_kv_capacity_miss_num_computed_tokens = 0
        self.last_running_kv_capacity_miss_num_tokens = 0
        self.preempted_reset_count = 0
        self.preempted_prefill_reset_count = 0
        self.preempted_decode_reset_count = 0
        self.last_preempt_iteration_id = -1
        self.last_preempt_phase = ""
        self.last_preempt_num_computed_tokens_before_reset = 0
        self.last_preempt_num_tokens_before_reset = 0
        self.preempted_tokens_reset_total = 0
        # A phase-separated Decode request normally has one uncomputed live
        # token at the tail of ``all_token_ids``. If an exceptional allocator
        # reset forces recomputation, every token that had already reached the
        # model must be replayed as Prefill work before that live tail token is
        # admitted to Decode again. Keep the high-water mark across repeated
        # resets so a second reset during replay cannot shorten the boundary.
        self.pd_recompute_frontier_tokens = 0
        self.running_prefill_kv_wait_count = 0
        self.last_running_prefill_kv_wait_iteration_id = -1
        self.last_running_prefill_kv_wait_num_computed_tokens = 0
        self.last_running_prefill_kv_wait_num_tokens = 0
        self.pd_residency_prefill_skip_count = 0
        self.pd_residency_prefill_skip_first_iteration_id = -1
        self.pd_residency_prefill_skip_last_iteration_id = -1
        self.pd_residency_prefill_unscheduled_duration_s = 0.0
        self.pd_residency_last_prefill_required_blocks = 0
        self.pd_residency_last_free_blocks = 0
        self.pd_residency_last_decode_resident_blocks = 0
        self.pd_residency_last_serialized_by_kv_reserve = 0
        self._pd_residency_prefill_skip_last_ts = -1.0
        self.virtual_context_deferred_prefill_count = 0
        self.virtual_context_deferred_logical_computed_tokens = 0
        self.virtual_context_deferred_state = ""
        self.virtual_context_deferred_last_iteration_id = -1
        self.virtual_context_deferred_last_offloaded_tokens = 0
        self.virtual_context_deferred_hbm_released_tokens = 0
        self.virtual_context_resume_without_recompute_count = 0
        self.virtual_context_hbm_reloaded_tokens = 0
        self.virtual_context_hbm_reload_planned_tokens = 0
        # A DRAM-backed request that reached Decode must resume by one native
        # reload and retain that logical progress. Ordinary vLLM preemption
        # would reset the whole prompt, so the scheduler holds this lightweight
        # lease until the request finishes.
        self.kv_retention_decode_continuation_lease_active = False
        self.kv_retention_decode_continuation_waiting = False
        self.kv_retention_decode_continuation_wait_count = 0
        self.kv_retention_decode_continuation_last_wait_iteration_id = -1
        # Scheduler-owned authority for one tiered execution Future. Only
        # small scalar state is retained; GPU tensors and Future references are
        # deliberately excluded from Request lifetime.
        self.kv_retention_execution_plan_generation = 0
        self.kv_retention_execution_plan_acked_generation = 0
        self.kv_retention_execution_plan_mode = "auto"
        self.kv_retention_execution_plan_expected_reload_pages = 0
        self.virtual_context_tiered_provider_deferred_tokens = 0
        self.virtual_context_tiered_provider_resume_count = 0
        self.virtual_context_tiered_provider_virtual_external_tokens = 0
        self.virtual_context_tiered_commit_count = 0
        self.virtual_context_tiered_commit_released_pages = 0
        self.virtual_context_tiered_commit_existing_dram_pages = 0
        self.virtual_context_tiered_commit_missing_pages = 0
        self.virtual_context_tiered_commit_last_iteration_id = -1
        self.virtual_context_tiered_commit_last_logical_tokens = 0
        self.virtual_context_tiered_transition_attempt_count = 0
        self.virtual_context_tiered_transition_last_iteration_id = -1
        self.virtual_context_tiered_transition_last_logical_tokens = 0
        self.virtual_context_tiered_transition_last_block_size = 0
        self.virtual_context_tiered_transition_last_release_candidate_records = 0
        self.virtual_context_tiered_transition_last_release_candidate_pages = 0
        self.virtual_context_tiered_transition_last_existing_dram_records = 0
        self.virtual_context_tiered_transition_last_existing_dram_pages = 0
        self.virtual_context_tiered_transition_last_required_pages = 0
        self.virtual_context_tiered_transition_last_covered_pages = 0
        self.virtual_context_tiered_transition_last_missing_pages = 0
        self.virtual_context_tiered_transition_last_reason = "not_attempted"
        self.virtual_context_preempt_reset_avoided_tokens = 0
        self.virtual_context_defer_fallback_count = 0
        self.virtual_context_defer_last_fallback_reason = ""
        self.deferred_by_residency_budget_count = 0
        self.virtual_context_residency_admission_defer_count = 0
        self.virtual_context_residency_admission_last_iteration_id = -1
        self.virtual_context_residency_admission_last_phase = ""
        self.virtual_context_residency_admission_last_required_blocks = 0
        self.virtual_context_residency_admission_last_free_blocks = 0
        self.virtual_context_residency_admission_last_decode_reserve_blocks = 0
        self.virtual_context_residency_admission_last_staging_blocks = 0
        # Explicit owner of output positions reserved by one phase Future.
        # The ledger stores no Future/tensor references and is enabled by the
        # experimental phase-local P/D scheduler only.
        self.pd_output_placeholder_ledger = OutputPlaceholderLedger()

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                tenant_id = sampling_params.extra_args.get("tenant_id")
                if tenant_id is not None:
                    self.tenant_id = str(tenant_id)
                session_id = sampling_params.extra_args.get("session_id")
                self.tenant_session_id = (
                    str(session_id) if session_id is not None else self.tenant_id
                )
                prefill_deadline_ms = sampling_params.extra_args.get(
                    "ttft_prefill_deadline_ms"
                )
                if prefill_deadline_ms is not None:
                    try:
                        parsed_deadline_ms = float(prefill_deadline_ms)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "vllm_xargs.ttft_prefill_deadline_ms must be a "
                            "positive finite number"
                        ) from error
                    if (
                        not math.isfinite(parsed_deadline_ms)
                        or parsed_deadline_ms <= 0
                    ):
                        raise ValueError(
                            "vllm_xargs.ttft_prefill_deadline_ms must be a "
                            "positive finite number"
                        )
                    self.ttft_prefill_deadline_ms = parsed_deadline_ms
                tbt_slo_ms = sampling_params.extra_args.get("tbt_slo_ms")
                if tbt_slo_ms is not None:
                    try:
                        parsed_tbt_slo_ms = float(tbt_slo_ms)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "vllm_xargs.tbt_slo_ms must be a positive "
                            "finite number"
                        ) from error
                    if (
                        not math.isfinite(parsed_tbt_slo_ms)
                        or parsed_tbt_slo_ms <= 0
                    ):
                        raise ValueError(
                            "vllm_xargs.tbt_slo_ms must be a positive "
                            "finite number"
                        )
                    self.tbt_slo_ms = parsed_tbt_slo_ms
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def validate_external_residency_suspension(
        self,
        *,
        logical_computed_tokens: int,
        deferred_state: str,
    ) -> tuple[int, str]:
        """Validate a tiered suspension before allocator ownership mutates."""
        owner = self.pd_output_placeholder_ledger.active
        if owner is not None:
            raise OutputPlaceholderOwnershipError(
                "Cannot suspend a request to external residency while a "
                "model Future owns output positions: "
                f"request={self.request_id}, "
                f"request_epoch={owner.request_epoch}, "
                f"generation={owner.generation}, "
                f"phase={owner.phase}, cancelled={owner.cancelled}."
            )
        logical_tokens = int(logical_computed_tokens)
        if logical_tokens <= 0 or logical_tokens > self.num_tokens:
            raise ValueError(
                "Invalid external-residency logical progress: "
                f"request={self.request_id}, logical_tokens={logical_tokens}, "
                f"request_tokens={self.num_tokens}."
            )
        state = str(deferred_state).strip()
        if not state:
            raise ValueError("External-residency suspension needs a state")
        return logical_tokens, state

    def suspend_to_external_residency(
        self,
        *,
        logical_computed_tokens: int,
        deferred_state: str,
        scheduler_iteration_id: int,
    ) -> None:
        """Separate durable token progress from released physical residency.

        This is deliberately different from ordinary vLLM preemption, which
        resets progress for recomputation. A tiered victim keeps its logical
        KV in an external provider and may later resume without recomputing it.
        """

        logical_tokens, state = self.validate_external_residency_suspension(
            logical_computed_tokens=logical_computed_tokens,
            deferred_state=deferred_state,
        )
        self.virtual_context_deferred_logical_computed_tokens = logical_tokens
        self.virtual_context_deferred_state = state
        self.virtual_context_deferred_last_iteration_id = int(
            scheduler_iteration_id
        )
        self.virtual_context_deferred_last_offloaded_tokens = logical_tokens
        self.num_computed_tokens = 0
        self.spec_token_ids.clear()

    def resume_from_external_residency(
        self,
        *,
        logical_computed_tokens: int,
    ) -> None:
        """Restore scheduler progress without conflating it with HBM ownership."""

        deferred_tokens = int(
            self.virtual_context_deferred_logical_computed_tokens
        )
        logical_tokens = int(logical_computed_tokens)
        if deferred_tokens <= 0:
            raise ValueError(
                "Cannot resume external residency without durable progress: "
                f"request={self.request_id}."
            )
        if logical_tokens < 0 or logical_tokens > deferred_tokens:
            raise ValueError(
                "External-residency resume exceeded durable progress: "
                f"request={self.request_id}, resume_tokens={logical_tokens}, "
                f"deferred_tokens={deferred_tokens}."
            )
        self.num_computed_tokens = logical_tokens

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
}
