# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

from vllm._bc_linter import bc_linter_include

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@bc_linter_include
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@bc_linter_include
@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@bc_linter_include
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Request IDs whose HBM KV/state was released while logical progress is
    # preserved by a residency/offload path. Workers should drop local runner
    # state and KV connectors should complete pending stores, but these are not
    # scheduler preemptions and must not be counted as reset/recompute events.
    residency_deferred_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None

    # Experimental per-iteration debug payload, populated only when enabled by
    # environment-gated logging in the engine.
    iteration_debug: dict[str, int | float | str] | None = None
    request_debug: dict[str, dict[str, int | float | str]] | None = None

    # Always-cheap scheduler epoch and correctness/control metadata.  These
    # fields must not depend on detailed iteration tracing being enabled.
    scheduler_iteration_id: int = -1
    tiered_dma_budget_bytes: int = 0
    tiered_staging_budget_blocks: int = 0
    tiered_max_prefill_ring_tokens: int = 0
    tiered_decode_tbt_budget_ms: float = 0.0
    tiered_pressure_snapshot_request_ids: tuple[str, ...] = ()
    # Exact scheduler-selected HBM pages to snapshot proactively for a future
    # scheduled-prefill-fit reclaim. Entries are
    # ``(request_id, kv_cache_group_id, logical_page)``. An empty tuple keeps
    # the legacy request-wide pressure snapshot behavior unchanged.
    tiered_proactive_snapshot_pages: tuple[tuple[str, int, int], ...] = ()
    # Durable logical token counts for proactive victims. This lets a worker
    # snapshot an exact partial tail for a quiescent request that is not part
    # of the current compute batch.
    tiered_proactive_snapshot_logical_tokens: tuple[tuple[str, int], ...] = ()
    # Provider-backed Prefill requests whose complete prefix was atomically
    # assigned real HBM blocks at this boundary. The worker may perform native
    # H2D reload for these ids even when the global pressure mode normally
    # prefers Context Ring.
    tiered_native_hbm_reload_request_ids: tuple[str, ...] = ()
    # Authoritative per-request tiered execution contract. Entries are
    # ``(request_id, generation, mode, expected_reload_pages)``. ``auto``
    # preserves legacy heuristics; explicit modes must win over them.
    tiered_request_execution_plans: tuple[
        tuple[str, int, str, int], ...
    ] = ()
    pd_prefill_mode: str = "native"

    # Experimental phase identity used by P/D decoupling prototypes.
    # Values: "prefill", "decode", "mixed", or "idle".
    scheduled_phase: str = "mixed"

    # Experimental phase-local submission identity. Slot zero preserves the
    # legacy one-context-per-phase behavior. EngineCore assigns a non-zero slot
    # only when the explicit P/D phase-depth controller is enabled.
    pd_phase_slot: int = 0
    pd_phase_submit_seq: int = -1
    # Default-off Llama Prefill continuation. When true, execute_model may
    # return a tensor-free layer boundary for this complete immutable batch;
    # no request row may be split or independently committed. Only the final
    # sample result may be committed through Scheduler.update_from_output().
    ttft_layerwise_prefill_preemption: bool = False
    # Immutable process-local logical transaction. Current segment bindings
    # travel on explicit RPC arguments/Futures, never in this mutable output.
    kv_retention_prefill_work: object | None = field(default=None, repr=False)
    # Single-process whole-batch transfer. The Engine arms only the bool;
    # the worker publishes a tensor-free receipt after its D2H event completes.
    kv_retention_layerwise_offload_requested: bool = False
    kv_retention_layerwise_pressure_checked: bool = False
    # P1 request/SM intent is not token work. Only a completed Worker ticket
    # can change physical slot accounting; this boolean records its commit.
    kv_retention_p1_parked: bool = False
    kv_retention_boundary_reason: str = ""
    kv_retention_p1_park_declined: bool = False
    kv_retention_layerwise_transfer: object | None = field(default=None, repr=False)

    # Explicit output-position ownership for the experimental phase-local P/D
    # path. Each value is
    # ``(request_epoch, generation, phase, placeholder_count)``. The scheduler
    # retains the matching request-local ledger and accepts results only from
    # this exact request incarnation and generation.
    pd_output_ownership_enabled: bool = False
    pd_output_placeholder_owners: dict[
        str, tuple[int, int, str, int]
    ] = field(default_factory=dict)
    # Output positions that this SchedulerOutput is expected to produce. The
    # scheduler prepares these claims before advancing async counters, then
    # commits the request ledgers from the exact counter deltas before the
    # output can be dispatched.
    pd_output_placeholder_expected_counts: dict[str, int] = field(
        default_factory=dict
    )
    pd_output_placeholder_counter_deltas: dict[str, int] = field(
        default_factory=dict
    )
    # Committed token-history generation used when a request is inserted into
    # a different phase-local InputBatch. This is a data transfer contract,
    # not a repair fallback.
    pd_output_token_source_generations: dict[str, int] = field(
        default_factory=dict
    )

    # A phase-pure scheduling attempt found logical work but could not admit
    # it. EngineCore uses this authoritative result to suppress identical
    # retries until the recorded capacity threshold or an explicitly selected
    # output-progress boundary is reached. These fields are control metadata
    # and remain available when detailed iteration tracing is disabled.
    pd_capacity_retry_blocked: bool = False
    pd_capacity_retry_min_free_blocks: int = 0
    pd_capacity_retry_observed_free_blocks: int = 0
    pd_capacity_retry_reason: str = ""
    # Stable machine-readable cause. Human-readable ``reason`` may contain
    # multiple policy details and must not drive SM-allocation semantics.
    pd_capacity_retry_cause: str = ""
    # Capacity is not always the only changing input. A zero-work Prefill
    # pass beside an in-flight Decode may be retried once that Decode output
    # advances scheduler/request state, even when free HBM did not increase.
    pd_capacity_retry_wake_on_output_progress: bool = False

    # Experimental vendor_v8 dynamic P/D SM allocation. The engine assigns one
    # immutable profile/epoch before submitting the output. Empty profile id
    # preserves the vendor_v6 fixed GreenContext path exactly.
    pd_sm_profile_id: str = ""
    pd_sm_profile_epoch: int = 0
    pd_sm_decode_sms: int = 0
    pd_sm_prefill_sms: int = 0

    # Requests whose scheduled Prefill chunk reaches the end of the prompt.
    # EngineCore uses this cheap handoff marker to fence unrelated Prefill
    # submissions until the request's first sampled token is emitted. This
    # prevents a following long Prefill future from extending a Short TTFT
    # handoff (including the Prefill-only SM profile case).
    pd_ttft_handoff_request_ids: tuple[str, ...] = ()

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
