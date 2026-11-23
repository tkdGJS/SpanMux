# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextvars
import functools
import gc
import itertools
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field
from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.attention.layer import Attention, MLAAttention
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import (
    CUDAGraphStat,
    CUDAGraphWrapper,
    cudagraph_namespace,
)
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.green_context import (
    green_context_execute_model_enabled,
    green_context_execute_model_no_join_enabled,
    green_context_dynamic_sm_enabled,
    green_context_layerwise_prefill_handoff,
    green_context_phase_stream,
    green_context_state,
    initialize_dynamic_green_context_profiles,
)
from vllm.model_executor.layers.operator_selective_attention import (
    consume_operator_selective_pd_metrics,
    reset_operator_selective_pd_metrics,
)
from vllm.model_executor.layers.operator_memory_shaping import (
    consume_opmux_metrics,
    reset_opmux_metrics,
)
from vllm.model_executor.layers.pd_prefill_workspace import (
    prefill_workspace_debug_snapshot,
    prefill_workspace_execution,
    prefill_workspace_requested,
    prefill_workspace_slot_count,
)
from vllm.model_executor.layers.kv_retention_parked_prefill import (
    KVRetentionParkedStore,
    KVRetentionParkedTicket,
    KVRetentionSamplingFrame,
    kv_retention_p1_hbm_only_requested,
    kv_retention_p1_parked_requested,
    kv_retention_parked_limits,
)
from vllm.model_executor.layers.kv_retention_resume_frame import (
    KVRetentionFrameBudgetExceeded,
    KVRetentionLayerContinuation,
    KVRetentionResumeFrame,
    kv_retention_record_retained_streams,
    kv_retention_resume_frame_gpu_bytes,
)
from vllm.model_executor.layers.ttft_layerwise_prefill import (
    KVRetentionExecutionBinding,
    KVRetentionCheckpointRow,
    KVRetentionLayerwiseCheckpoint,
    KVRetentionPhaseOwner,
    TTFTLayerwisePrefillBoundary,
    TTFTLayerwisePrefillError,
    TTFTLayerwisePrefillExecution,
    TTFTLayerwisePrefillModelYield,
    active_ttft_layerwise_prefill_execution,
    build_kv_retention_layerwise_checkpoint,
    kv_retention_layerwise_checkpoint_requested,
    ttft_layerwise_prefill_execution,
    ttft_layerwise_prefill_preemption_requested,
    ttft_prefill_continuation_mode_allowed,
)
from vllm.model_executor.layers.kv_retention_layerwise import (
    kv_retention_layerwise_offload_enabled,
    kv_retention_ring_scope,
)
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mm_encoder_only,
    supports_mrope,
    supports_multimodal_pruning,
    supports_transcription,
    supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.nvtx_pytorch_hooks import PytHooks
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
    supports_dynamo,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
    split_attn_metadata,
)
from vllm.v1.attention.ops.tiered_blockwise_backend import (
    consume_tiered_context_ring_runtime_metrics,
    flush_tiered_context_ring_source_lease_releases,
    reset_tiered_context_ring_runtime_metrics,
)
from vllm.v1.attention.ops.tiered_d2h_source_plan import (
    build_tiered_d2h_source_plan,
)
from vllm.v1.attention.ops.tiered_residency_provider import (
    get_dram_kv_request_key,
    register_hbm_kv_cache_pages,
    reload_dram_kv_pages_to_hbm,
    tiered_attention_plan_tensors_enabled,
    tiered_d2h_source_plan_enabled,
    tiered_dram_register_from_hbm_enabled,
    tiered_dram_register_from_hbm_policy,
    tiered_dram_register_pressure_only_enabled,
    validate_tiered_direct_multicopy_ops,
    wait_for_pending_hbm_d2h_request_reads,
)
from vllm.v1.core.tiered_execution import (
    TIERED_EXECUTION_CONTEXT_RING,
    TIERED_EXECUTION_NATIVE_RELOAD,
    has_native_reload_authority,
    parse_tiered_execution_plan_entries,
)
from vllm.v1.core.virtual_context import (
    get_virtual_context_table,
    virtual_context_enabled,
    virtual_context_layer_id,
    virtual_context_prefill_ring_required_mode,
    virtual_context_ring_backend_dram_provider_enabled,
    virtual_context_ring_backend_enabled,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.engine.async_jsonl_writer import AsyncJSONLWriter
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.output_placeholder import OutputPlaceholderOwnershipError
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.kv_retention_residency_descriptor import (
    KVRetentionResidencyDescriptorStore,
)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.workspace import lock_workspace

from .utils import (
    AttentionGroup,
    MultiModalBudget,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    sanity_check_mm_encoder_outputs,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict

_PD_CUDA_TRACE_EPOCH_LOCK = threading.Lock()
_PD_CUDA_TRACE_EPOCHS: dict[tuple[int, int], tuple[Any, str]] = {}
_PD_CUDA_TRACE_EPOCH_SEQUENCE = itertools.count(1)


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _get_pd_cuda_trace_epoch(stream: Any) -> tuple[Any, str]:
    """Return a common CUDA-event time origin for all P/D phase streams.

    The first caller records one event on its current stream. Every traced
    stream waits on that already-recorded event before its first sampled
    interval. This creates a comparable device timeline without joining the
    Decode or Prefill stream and without synchronizing the whole device.
    """

    device_index = int(getattr(stream, "device_index", 0) or 0)
    key = (os.getpid(), device_index)
    with _PD_CUDA_TRACE_EPOCH_LOCK:
        cached = _PD_CUDA_TRACE_EPOCHS.get(key)
        if cached is not None:
            return cached
        epoch_event = torch.cuda.Event(enable_timing=True)
        epoch_event.record(stream)
        epoch_id = (
            f"{os.getpid()}:{device_index}:"
            f"{next(_PD_CUDA_TRACE_EPOCH_SEQUENCE)}"
        )
        cached = (epoch_event, epoch_id)
        _PD_CUDA_TRACE_EPOCHS[key] = cached
        return cached


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        return output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensors to avoid them being
        # deallocated until we finish copying it to the host.
        self._raw_pooler_output = raw_pooler_output

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            raw_pooler_output_cpu = json_map_leaves(
                lambda x: None if x is None else x.to("cpu", non_blocking=True),
                self._raw_pooler_output,
            )
            self.async_copy_ready_event.record()
            self._model_runner_output.pooler_output = [
                out if include else None
                for out, include in zip(raw_pooler_output_cpu, finished_mask)
            ]

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        This function blocks until the copy is finished.
        """
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._raw_pooler_output
        return self._model_runner_output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    model_runner_debug_timings: dict[str, Any]


@dataclass
class TTFTPrefillContinuationState:
    """Worker-only state for one immutable whole-batch Prefill continuation."""

    scheduler_output: "SchedulerOutput"
    execution: TTFTLayerwisePrefillExecution
    boundary: TTFTLayerwisePrefillBoundary
    continuation: Iterator[Any] = field(repr=False)
    attn_metadata: Any = field(repr=False)
    num_tokens_padded: int = 0
    num_tokens_across_dp: torch.Tensor | None = field(default=None, repr=False)
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_desc: BatchDescriptor | None = None
    ubatch_slices_padded: Any = field(default=None, repr=False)
    logits_indices: torch.Tensor = field(repr=False, default=None)  # type: ignore[assignment]
    spec_decode_metadata: SpecDecodeMetadata | None = field(
        default=None, repr=False
    )
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None = field(
        default=None, repr=False
    )
    ec_connector_output: ECConnectorOutput | None = field(
        default=None, repr=False
    )
    cudagraph_stats: CUDAGraphStat | None = None
    num_scheduled_tokens: int = 0
    num_scheduled_tokens_np: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    num_reqs: int = 0
    model_runner_debug_timings: dict[str, Any] = field(default_factory=dict)
    forward_host_s_total: float = 0.0
    boundary_wait_s_total: float = 0.0
    kv_checkpoint: KVRetentionLayerwiseCheckpoint | None = None
    kv_offload: Any = field(default=None, repr=False)
    # Forward-only detached state. InputBatch/sampling still owns the slot.
    kv_resume_frame: KVRetentionResumeFrame | None = field(default=None, repr=False)
    kv_sampling_frame: KVRetentionSamplingFrame | None = field(default=None, repr=False)
    kv_sampling_restore_pending: bool = False


@dataclass
class PDPhaseModelRunnerContext:
    """Phase-local runner state for experimental P/D multiplexing.

    vLLM's GPUModelRunner normally reuses persistent input buffers and an
    execute/sample handoff slot across iterations. That is safe for the
    upstream single-lane loop, but not for no-join decode/prefill GreenContext
    lanes. This context gives each bounded phase slot its own input buffers and
    handoff state while keeping request/KV ownership in the existing scheduler.
    """

    phase: str
    slot: int
    input_batch: InputBatch
    input_ids: CpuGpuBuffer
    positions: CpuGpuBuffer
    query_start_loc: CpuGpuBuffer
    seq_lens: CpuGpuBuffer
    encoder_seq_lens: CpuGpuBuffer
    inputs_embeds: CpuGpuBuffer
    is_token_ids: CpuGpuBuffer
    discard_request_mask: CpuGpuBuffer
    num_decode_draft_tokens: CpuGpuBuffer
    num_accepted_tokens: CpuGpuBuffer
    dcp_local_seq_lens: CpuGpuBuffer | None = None
    mrope_positions: CpuGpuBuffer | None = None
    xdrope_positions: CpuGpuBuffer | None = None
    is_mm_embed_buffers: list[CpuGpuBuffer] | None = None
    is_mm_embed_idx: int = 0
    async_output_copy_stream: torch.cuda.Stream | None = None
    prepare_inputs_event: torch.Event | None = None
    prepare_inputs_event_recorded: bool = False
    prepare_inputs_event_fast_path_count: int = 0
    prepare_inputs_event_wait_count: int = 0
    execute_model_state: ExecuteModelState | None = None
    ttft_prefill_continuation: TTFTPrefillContinuationState | None = None
    kv_retention_execute_owner: KVRetentionPhaseOwner | None = None
    # Survives cleanup so a delayed initial RPC cannot resurrect disposed work.
    kv_retention_last_slot_generation: int = 0
    kv_connector_output: KVConnectorOutput | None = None
    _pd_green_context_execute_phase: str | None = None
    _pd_green_context_execute_active: bool = False
    _pd_green_context_execute_profile_id: str = ""
    _pd_green_context_execute_profile_epoch: int = 0
    _tiered_scheduler_budget: dict[str, Any] = field(default_factory=dict)
    _pd_green_context_sample_phase: str | None = None
    _pd_green_context_sample_profile_id: str = ""
    _pd_green_context_sample_profile_epoch: int = 0
    _pd_green_context_current_sample_phase: str | None = None
    _pd_green_context_current_sample_active: bool = False
    _pd_green_context_current_sample_profile_id: str = ""
    _pd_green_context_current_sample_profile_epoch: int = 0
    _pd_decode_gpu_component_capture: "_CudaModuleTimingCapture | None" = None


_PD_ACTIVE_MODEL_RUNNER_CONTEXT: contextvars.ContextVar[
    PDPhaseModelRunnerContext | None
] = contextvars.ContextVar("vllm_pd_active_model_runner_context", default=None)

_PD_PHASE_MODEL_RUNNER_CONTEXT_FIELDS = frozenset(
    {
        "input_batch",
        "input_ids",
        "positions",
        "query_start_loc",
        "seq_lens",
        "encoder_seq_lens",
        "inputs_embeds",
        "is_token_ids",
        "discard_request_mask",
        "num_decode_draft_tokens",
        "num_accepted_tokens",
        "dcp_local_seq_lens",
        "mrope_positions",
        "xdrope_positions",
        "is_mm_embed_buffers",
        "is_mm_embed_idx",
        "async_output_copy_stream",
        "prepare_inputs_event",
        "prepare_inputs_event_recorded",
        "prepare_inputs_event_fast_path_count",
        "prepare_inputs_event_wait_count",
        "execute_model_state",
        "kv_connector_output",
        "_pd_green_context_execute_phase",
        "_pd_green_context_execute_active",
        "_pd_green_context_execute_profile_id",
        "_pd_green_context_execute_profile_epoch",
        "_tiered_scheduler_budget",
        "_pd_green_context_sample_phase",
        "_pd_green_context_sample_profile_id",
        "_pd_green_context_sample_profile_epoch",
        "_pd_green_context_current_sample_phase",
        "_pd_green_context_current_sample_active",
        "_pd_green_context_current_sample_profile_id",
        "_pd_green_context_current_sample_profile_epoch",
        "_pd_decode_gpu_component_capture",
    }
)


class _CudaModuleTimingCapture:
    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self.active_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.metadata = dict(metadata or {})
        self.forward_start_event: torch.cuda.Event | None = None
        self.forward_end_event: torch.cuda.Event | None = None
        self.pipeline_end_event: torch.cuda.Event | None = None
        self.stage_events: dict[
            str, tuple[torch.cuda.Event, torch.cuda.Event]
        ] = {}


class _CudaModuleTimingHooks:
    """Diagnostic CUDA event timing for Llama-like decoder layer modules.

    This is intentionally env-gated by GPUModelRunner. A ContextVar keeps
    concurrent Decode/Prefill host threads from attaching records to the wrong
    forward capture.
    """

    _SUFFIX_TO_CATEGORY = (
        (".self_attn.qkv_proj", "qkv_proj"),
        (".self_attn.attn", "attention"),
        (".self_attn.o_proj", "o_proj"),
        (".mlp.gate_up_proj", "mlp_gate_up_proj"),
        (".mlp.act_fn", "mlp_activation"),
        (".mlp.down_proj", "mlp_down_proj"),
    )

    def __init__(self, model: nn.Module) -> None:
        self._handles: list[Any] = []
        self._active_capture: contextvars.ContextVar[
            _CudaModuleTimingCapture | None
        ] = contextvars.ContextVar(
            f"vllm_cuda_module_timing_capture_{id(self)}", default=None
        )
        self.matched_modules = 0
        self._register_hooks(model)

    @staticmethod
    def _category_for_module_name(name: str) -> str | None:
        for suffix, category in _CudaModuleTimingHooks._SUFFIX_TO_CATEGORY:
            if name.endswith(suffix):
                return category
        return None

    @staticmethod
    def _layer_index_from_module_name(name: str) -> int:
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers":
                try:
                    return int(parts[index + 1])
                except ValueError:
                    return -1
        return -1

    def _register_hooks(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            category = self._category_for_module_name(name)
            if category is None:
                continue
            layer_index = self._layer_index_from_module_name(name)
            self._handles.append(
                module.register_forward_pre_hook(
                    self._make_pre_hook(name, category, layer_index)
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    self._make_post_hook(name, category, layer_index)
                )
            )
            self.matched_modules += 1

    def _make_pre_hook(self, name: str, category: str, layer_index: int):
        def hook(module: nn.Module, inputs: Any) -> None:
            capture = self._active_capture.get()
            if capture is None or not torch.cuda.is_available():
                return
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            record = {
                "name": name,
                "category": category,
                "layer_index": layer_index,
                "start_event": start_event,
                "end_event": end_event,
                "completed": False,
            }
            start_event.record()
            capture.active_records[id(module)].append(record)

        return hook

    def _make_post_hook(self, name: str, category: str, layer_index: int):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            capture = self._active_capture.get()
            if capture is None or not torch.cuda.is_available():
                return
            records = capture.active_records.get(id(module))
            if not records:
                return
            record = records.pop()
            record["end_event"].record()
            record["completed"] = True
            capture.records.append(record)

        return hook

    @contextmanager
    def capture_forward(
        self, metadata: dict[str, Any] | None = None
    ) -> Iterator[_CudaModuleTimingCapture | None]:
        # Retain the whole-forward record even if this model has no recognized
        # Llama-like submodules. A zero-match trace is actionable evidence;
        # silently dropping it would look like a broken writer and could lead
        # to a false partition-scaling diagnosis.
        if not torch.cuda.is_available():
            yield None
            return
        capture = _CudaModuleTimingCapture(metadata)
        capture.forward_start_event = torch.cuda.Event(enable_timing=True)
        capture.forward_end_event = torch.cuda.Event(enable_timing=True)
        capture.forward_start_event.record()
        token = self._active_capture.set(capture)
        try:
            yield capture
        finally:
            capture.forward_end_event.record()
            self._active_capture.reset(token)

    @staticmethod
    def capture_ready(capture: _CudaModuleTimingCapture | None) -> bool:
        if capture is None:
            return True
        end_event = capture.pipeline_end_event
        if end_event is None:
            end_event = capture.forward_end_event
        return end_event is None or bool(end_event.query())

    @staticmethod
    def capture_metrics(
        capture: _CudaModuleTimingCapture,
    ) -> dict[str, Any]:
        """Build component metrics after ``capture_ready`` becomes true."""
        records = [record for record in capture.records if record["completed"]]
        category_ms: dict[str, float] = defaultdict(float)
        layer_category_ms: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for record in records:
            duration_ms = record["start_event"].elapsed_time(record["end_event"])
            category = str(record["category"])
            layer_index = int(record["layer_index"])
            category_ms[category] += duration_ms
            if layer_index >= 0:
                layer_category_ms[layer_index][category] += duration_ms

        component_sum_ms = sum(category_ms.values())
        mlp_total_ms = (
            category_ms["mlp_gate_up_proj"]
            + category_ms["mlp_activation"]
            + category_ms["mlp_down_proj"]
        )
        projection_total_ms = (
            category_ms["qkv_proj"]
            + category_ms["o_proj"]
            + category_ms["mlp_gate_up_proj"]
            + category_ms["mlp_down_proj"]
        )
        attention_stack_ms = (
            category_ms["qkv_proj"]
            + category_ms["attention"]
            + category_ms["o_proj"]
        )
        forward_ms = 0.0
        if (
            capture.forward_start_event is not None
            and capture.forward_end_event is not None
        ):
            forward_ms = capture.forward_start_event.elapsed_time(
                capture.forward_end_event
            )
        stage_ms = {
            name: start_event.elapsed_time(end_event)
            for name, (start_event, end_event) in capture.stage_events.items()
        }
        pipeline_ms = forward_ms
        if (
            capture.forward_start_event is not None
            and capture.pipeline_end_event is not None
        ):
            pipeline_ms = capture.forward_start_event.elapsed_time(
                capture.pipeline_end_event
            )
        logits_postprocess_ms = stage_ms.get("logits_postprocess", 0.0)
        sampling_ms = stage_ms.get("sampling", 0.0)

        layer_details = []
        for layer_index in sorted(layer_category_ms):
            layer_values = layer_category_ms[layer_index]
            layer_mlp_total_ms = (
                layer_values["mlp_gate_up_proj"]
                + layer_values["mlp_activation"]
                + layer_values["mlp_down_proj"]
            )
            layer_attention_stack_ms = (
                layer_values["qkv_proj"]
                + layer_values["attention"]
                + layer_values["o_proj"]
            )
            layer_details.append(
                {
                    "layer_index": layer_index,
                    "qkv_proj_ms": round(layer_values["qkv_proj"], 6),
                    "attention_ms": round(layer_values["attention"], 6),
                    "o_proj_ms": round(layer_values["o_proj"], 6),
                    "mlp_gate_up_proj_ms": round(
                        layer_values["mlp_gate_up_proj"], 6
                    ),
                    "mlp_activation_ms": round(
                        layer_values["mlp_activation"], 6
                    ),
                    "mlp_down_proj_ms": round(
                        layer_values["mlp_down_proj"], 6
                    ),
                    "attention_stack_ms": round(layer_attention_stack_ms, 6),
                    "mlp_total_ms": round(layer_mlp_total_ms, 6),
                    "total_ms": round(sum(layer_values.values()), 6),
                }
            )

        residual_ms = max(0.0, forward_ms - component_sum_ms)
        pipeline_residual_ms = max(
            0.0,
            pipeline_ms - forward_ms - logits_postprocess_ms - sampling_ms,
        )
        return {
            "decode_stream_pipeline_ms": pipeline_ms,
            "gpu_model_forward_ms": forward_ms,
            "gpu_component_sum_ms": component_sum_ms,
            "gpu_residual_unattributed_ms": residual_ms,
            "logits_postprocess_ms": logits_postprocess_ms,
            "sampling_ms": sampling_ms,
            "decode_stream_interstage_residual_ms": pipeline_residual_ms,
            "qkv_proj_ms": category_ms["qkv_proj"],
            "attention_ms": category_ms["attention"],
            "o_proj_ms": category_ms["o_proj"],
            "mlp_gate_up_proj_ms": category_ms["mlp_gate_up_proj"],
            "mlp_activation_ms": category_ms["mlp_activation"],
            "mlp_down_proj_ms": category_ms["mlp_down_proj"],
            "mlp_total_ms": mlp_total_ms,
            "projection_total_ms": projection_total_ms,
            "attention_stack_ms": attention_stack_ms,
            "num_records": len(records),
            "num_layers": len(layer_details),
            "layers": layer_details,
        }

    def summarize_capture(
        self, capture: _CudaModuleTimingCapture | None
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "layer_kernel_timing_enabled": 1.0,
            "layer_kernel_timing_matched_modules": float(self.matched_modules),
            "layer_kernel_timing_num_records": 0.0,
            "layer_kernel_timing_num_layers": 0.0,
            "layer_kernel_timing_total_ms": 0.0,
        }
        if capture is None:
            return summary

        if capture.forward_end_event is not None:
            capture.forward_end_event.synchronize()
        metrics = self.capture_metrics(capture)

        summary.update(
            {
                "layer_kernel_timing_num_records": float(metrics["num_records"]),
                "layer_kernel_timing_num_layers": float(metrics["num_layers"]),
                "layer_kernel_timing_total_ms": metrics["gpu_component_sum_ms"],
                "layer_kernel_timing_qkv_proj_ms": metrics["qkv_proj_ms"],
                "layer_kernel_timing_attention_ms": metrics["attention_ms"],
                "layer_kernel_timing_o_proj_ms": metrics["o_proj_ms"],
                "layer_kernel_timing_mlp_gate_up_proj_ms": metrics[
                    "mlp_gate_up_proj_ms"
                ],
                "layer_kernel_timing_mlp_activation_ms": metrics[
                    "mlp_activation_ms"
                ],
                "layer_kernel_timing_mlp_down_proj_ms": metrics[
                    "mlp_down_proj_ms"
                ],
                "layer_kernel_timing_mlp_total_ms": metrics["mlp_total_ms"],
                "layer_kernel_timing_projection_total_ms": metrics[
                    "projection_total_ms"
                ],
                "layer_kernel_timing_attention_stack_ms": metrics[
                    "attention_stack_ms"
                ],
                "layer_kernel_timing_gpu_model_forward_ms": metrics[
                    "gpu_model_forward_ms"
                ],
                "layer_kernel_timing_residual_unattributed_ms": metrics[
                    "gpu_residual_unattributed_ms"
                ],
                "layer_kernel_timing_layers": metrics["layers"],
            }
        )
        return summary


class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    def __getattribute__(self, name: str) -> Any:
        if name in _PD_PHASE_MODEL_RUNNER_CONTEXT_FIELDS:
            context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
            if context is not None and hasattr(context, name):
                return getattr(context, name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _PD_PHASE_MODEL_RUNNER_CONTEXT_FIELDS:
            context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
            if context is not None and hasattr(context, name):
                setattr(context, name, value)
                return
        super().__setattr__(name, value)

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._pd_prefill_workspace_enabled = prefill_workspace_requested()
        self.layer_kernel_timing_enabled = _env_flag_enabled(
            "VLLM_LAYER_KERNEL_TIMING"
        )
        self._pd_decode_gpu_component_trace_jsonl = os.environ.get(
            "VLLM_PD_DECODE_GPU_COMPONENT_TRACE_JSONL", ""
        ).strip()
        try:
            self._pd_decode_gpu_component_trace_max_pending = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING",
                        "8192",
                    )
                ),
            )
        except ValueError:
            self._pd_decode_gpu_component_trace_max_pending = 8192
        try:
            self._pd_decode_gpu_component_trace_every_n = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N", "1"
                    )
                ),
            )
        except ValueError:
            self._pd_decode_gpu_component_trace_every_n = 1
        self._pd_decode_gpu_component_trace_include_layers = _env_flag_enabled(
            "VLLM_PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS"
        )
        self._pd_decode_gpu_component_trace_seen = 0
        self._pd_decode_gpu_component_trace_pending: list[
            _CudaModuleTimingCapture
        ] = []
        self._pd_decode_gpu_component_trace_lock = threading.Lock()
        self._pd_decode_gpu_component_trace_warned = False
        self._pd_decode_gpu_component_trace_dropped = 0
        self.layer_kernel_timing_hooks: _CudaModuleTimingHooks | None = None

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes

        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )
        from vllm.isolation_memory import (
            isolation_execution_enabled, isolation_unused_embed_staging,
        )
        from vllm.v1.core.sched.observability import iteration_debug_sample_every_n

        self._isolation_unused_embed_staging = isolation_unused_embed_staging(
            supports_mm_inputs=self.supports_mm_inputs,
            prompt_embeds=self.enable_prompt_embeds,
            speculative=bool(self.vllm_config.speculative_config),
            encoder_decoder=model_config.is_encoder_decoder,
            pooling=self.is_pooling_model,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        )
        self._isolation_memory_snapshots = isolation_execution_enabled("MEMORY_SNAPSHOTS")
        self._isolation_p1_ring_sm_handoff = bool(
            kv_retention_p1_parked_requested()
            and ttft_layerwise_prefill_preemption_requested()
            and isolation_execution_enabled("P1_RING_SM_HANDOFF"))
        self._isolation_memory_sample_every = iteration_debug_sample_every_n()
        if self._isolation_unused_embed_staging:
            logger.info("Isolation text-only inputs: unused embeddings staging elided; "
                        "model embedding computation and token budgets unchanged")

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: EplbState | None = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer | SuffixDecodingProposer | EagleProposer | MedusaProposer
            )
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        self._kv_retention_residency_descriptors = (
            KVRetentionResidencyDescriptorStore()
        )
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}
        self.comm_stream = torch.cuda.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        self._custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        self._logitsprocs_need_output_token_ids = bool(self._custom_logitsprocs)
        self._input_batch_block_sizes = [self.cache_config.block_size]
        self._input_batch_kernel_block_sizes = [self.cache_config.block_size]
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=self._build_input_batch_logitsprocs(),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=(
                self._logitsprocs_need_output_token_ids
            ),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=(
                self.parallel_config.cp_kv_cache_interleave_size
            ),
        )

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        self.prepare_inputs_event_recorded = False
        self.prepare_inputs_event_fast_path_count = 0
        self.prepare_inputs_event_wait_count = 0
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            self.prepare_inputs_event = torch.Event()

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            0 if self._isolation_unused_embed_staging else self.max_num_tokens,
            self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            # Double buffer to avoid race condition: previous iteration's async
            # copy may still be reading from CPU while current iteration writes.
            self.is_mm_embed_buffers = [
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
            ]
            self.is_mm_embed_idx = 0

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(
                self.model_config,
                self.scheduler_config,
                self.mm_registry,
            )
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self._pd_green_context_execute_phase: str | None = None
        self._pd_green_context_execute_active = False
        self._pd_green_context_execute_profile_id = ""
        self._pd_green_context_execute_profile_epoch = 0
        self._tiered_scheduler_budget: dict[str, Any] = {}
        self._virtual_context_enabled = virtual_context_enabled()
        if self._virtual_context_enabled:
            validate_tiered_direct_multicopy_ops()
        self._virtual_context_native_decode_worker_fast_path = os.getenv(
            "VLLM_VIRTUAL_CONTEXT_NATIVE_DECODE_CONTROL_FAST_PATH", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._pd_green_context_sample_phase: str | None = None
        self._pd_green_context_sample_profile_id = ""
        self._pd_green_context_sample_profile_epoch = 0
        self._pd_green_context_current_sample_phase: str | None = None
        self._pd_green_context_current_sample_active = False
        self._pd_green_context_current_sample_profile_id = ""
        self._pd_green_context_current_sample_profile_epoch = 0
        self._pd_decode_gpu_component_capture: _CudaModuleTimingCapture | None = None
        self._pd_phase_local_model_runner_enabled = os.environ.get(
            "VLLM_PD_PHASE_LOCAL_MODEL_RUNNER",
            os.environ.get("VLLM_PD_MUXWISE_EVENT_LOOP", "0"),
        ).lower() in ("1", "true", "yes", "on")
        self._pd_green_decode_cudagraph_enabled = _env_flag_enabled(
            "VLLM_PD_GREEN_DECODE_CUDAGRAPH"
        )
        self._pd_green_decode_cudagraph_strict = os.environ.get(
            "VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT", "1"
        ).lower() in ("1", "true", "yes", "on")
        try:
            self._pd_green_decode_cudagraph_slot_count = max(
                1,
                int(os.environ.get("VLLM_PD_DECODE_INFLIGHT_DEPTH", "1")),
            )
        except ValueError:
            self._pd_green_decode_cudagraph_slot_count = 1
        self._pd_green_decode_cudagraph_capture_count = 0
        self._pd_green_decode_cudagraph_namespaces: set[str] = set()
        self._pd_phase_handoff_trace_jsonl = os.environ.get(
            "VLLM_PD_PHASE_HANDOFF_TRACE_JSONL", ""
        )
        self._pd_phase_sync_sample_output_enabled = _env_flag_enabled(
            "VLLM_PD_PHASE_SYNC_SAMPLE_OUTPUT"
        )
        self._pd_phase_sampler_debug_sync_enabled = _env_flag_enabled(
            "VLLM_PD_PHASE_SAMPLER_DEBUG_SYNC"
        )
        self._pd_nvtx_phase_trace_enabled = _env_flag_enabled(
            "VLLM_PD_NVTX_PHASE_TRACE"
        )
        self._pd_cuda_event_trace_jsonl = os.environ.get(
            "VLLM_PD_CUDA_EVENT_TRACE_JSONL", ""
        )
        try:
            self._pd_cuda_event_trace_max_pending = max(
                1,
                int(os.environ.get("VLLM_PD_CUDA_EVENT_TRACE_MAX_PENDING", "8192")),
            )
        except ValueError:
            self._pd_cuda_event_trace_max_pending = 8192
        try:
            self._pd_cuda_event_trace_every_n = max(
                1,
                int(os.environ.get("VLLM_PD_CUDA_EVENT_TRACE_EVERY_N", "1")),
            )
        except ValueError:
            self._pd_cuda_event_trace_every_n = 1
        try:
            self._pd_cuda_event_trace_decode_every_n = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_PD_CUDA_EVENT_TRACE_DECODE_EVERY_N",
                        str(self._pd_cuda_event_trace_every_n),
                    )
                ),
            )
        except ValueError:
            self._pd_cuda_event_trace_decode_every_n = (
                self._pd_cuda_event_trace_every_n
            )
        try:
            self._pd_cuda_event_trace_prefill_every_n = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_PD_CUDA_EVENT_TRACE_PREFILL_EVERY_N",
                        str(self._pd_cuda_event_trace_every_n),
                    )
                ),
            )
        except ValueError:
            self._pd_cuda_event_trace_prefill_every_n = (
                self._pd_cuda_event_trace_every_n
            )
        self._pd_cuda_event_trace_phases = {
            value.strip().lower()
            for value in os.environ.get(
                "VLLM_PD_CUDA_EVENT_TRACE_PHASES", ""
            ).split(",")
            if value.strip()
        }
        self._pd_cuda_event_trace_kinds = {
            value.strip().lower()
            for value in os.environ.get(
                "VLLM_PD_CUDA_EVENT_TRACE_KINDS", ""
            ).split(",")
            if value.strip()
        }
        self._pd_cuda_event_trace_seen: defaultdict[tuple[str, str], int] = (
            defaultdict(int)
        )
        self._pd_cuda_event_trace_pending: list[dict[str, Any]] = []
        self._pd_cuda_event_trace_lock = threading.Lock()
        self._pd_cuda_event_trace_warned = False
        self._pd_cuda_event_trace_dropped = 0
        try:
            pd_debug_trace_queue_size = max(
                1,
                int(os.environ.get("VLLM_DEBUG_TRACE_QUEUE_SIZE", "4096")),
            )
        except ValueError:
            pd_debug_trace_queue_size = 4096
        self._pd_debug_trace_writer = (
            AsyncJSONLWriter(max_pending=pd_debug_trace_queue_size)
            if any(
                (
                    self._pd_decode_gpu_component_trace_jsonl,
                    self._pd_phase_handoff_trace_jsonl,
                    self._pd_cuda_event_trace_jsonl,
                )
            )
            else None
        )
        if (
            self._pd_nvtx_phase_trace_enabled or self._pd_cuda_event_trace_jsonl
        ) and _env_flag_enabled("VLLM_ITERATION_GPU_TIMING"):
            logger.warning(
                "P/D phase CUDA trace is enabled with "
                "VLLM_ITERATION_GPU_TIMING=1. The iteration GPU timing path "
                "synchronizes the forward path and can hide stream overlap."
            )
        self._pd_phase_contexts: dict[
            tuple[str, int], PDPhaseModelRunnerContext
        ] = {}
        self._kv_retention_parked_store = (
            KVRetentionParkedStore(kv_retention_parked_limits()[1])
            if kv_retention_p1_parked_requested() else None)
        self._kv_retention_p1_hbm_only = kv_retention_p1_hbm_only_requested()
        self.kv_connector_output: KVConnectorOutput | None = None
        self.layerwise_nvtx_hooks_registered = False

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        if self.mm_budget:
            self.mm_budget.reset_cache()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """
        Re-initialize the KV cache and FP8 scales after waking from sleep.
        1. Zero out the KV cache tensors to remove garbage data from re-allocation.
        2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
          If these are left at 0.0 (default after wake_up), all KV cache values
          become effectively zero, causing gibberish output.
        """
        if not self.cache_config.cache_dtype.startswith("fp8"):
            return

        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()

        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: Generally, scale is 1.0 if user uses on-the-fly fp8
                # kvcache quant. However, to get better accuracy, compression
                # frameworks like llm-compressors allow users to tune the
                # scale. We may need to restore the specific calibrated scales
                # here in the future.
                k_scale_val, v_scale_val = 1.0, 1.0

                # Processing K Scale
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # Processing V Scale
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions.gpu[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions.gpu[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _build_input_batch_logitsprocs(self) -> LogitsProcessors:
        return build_logitsprocs(
            self.vllm_config,
            self.device,
            self.pin_memory,
            self.is_pooling_model,
            self._custom_logitsprocs,
        )

    def _make_pd_phase_input_batch(self) -> InputBatch:
        # Logits processors keep batch-indexed mutable state (for example
        # min_tokens and logit_bias). P/D phase-local batches have independent
        # request index spaces, so they must not share processor instances.
        return InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=self._input_batch_block_sizes,
            kernel_block_sizes=self._input_batch_kernel_block_sizes,
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=self._build_input_batch_logitsprocs(),
            logitsprocs_need_output_token_ids=(
                self._logitsprocs_need_output_token_ids
            ),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=self.num_spec_tokens,
            cp_kv_cache_interleave_size=(
                self.parallel_config.cp_kv_cache_interleave_size
            ),
        )

    def _make_pd_phase_context(
        self,
        phase: str,
        slot: int,
    ) -> PDPhaseModelRunnerContext:
        from vllm.isolation_memory import isolation_decode_token_capacity

        token_capacity = isolation_decode_token_capacity(
            phase=phase, max_tokens=self.max_num_tokens,
            max_requests=self.max_num_reqs,
            uniform_query_len=self.uniform_decode_query_len,
            supported=(
                not self.supports_mm_inputs and not self.enable_prompt_embeds
                and not self.vllm_config.speculative_config
                and not self.model_config.is_encoder_decoder
                and self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.pipeline_parallel_size == 1
                and not self.uses_mrope and self.uses_xdrope_dim == 0
            ),
        )
        if token_capacity != self.max_num_tokens:
            logger.info(
                "Isolation Decode input capacity=%d (token budget=%d, slot=%d); "
                "request/history and CUDA Graph shape budgets unchanged",
                token_capacity, self.max_num_tokens, slot,
            )
        is_mm_embed_buffers = None
        if self.supports_mm_inputs:
            is_mm_embed_buffers = [
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
            ]

        return PDPhaseModelRunnerContext(
            phase=phase,
            slot=slot,
            input_batch=self._make_pd_phase_input_batch(),
            input_ids=self._make_buffer(token_capacity, dtype=torch.int32),
            positions=self._make_buffer(token_capacity, dtype=torch.int64),
            query_start_loc=self._make_buffer(
                self.max_num_reqs + 1, dtype=torch.int32
            ),
            seq_lens=self._make_buffer(self.max_num_reqs, dtype=torch.int32),
            encoder_seq_lens=self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            ),
            inputs_embeds=self._make_buffer(
                0 if getattr(self, "_isolation_unused_embed_staging", False) else token_capacity,
                self.inputs_embeds_size,
                dtype=self.dtype,
                numpy=False,
            ),
            is_token_ids=self._make_buffer(
                token_capacity, dtype=torch.bool
            ),
            discard_request_mask=self._make_buffer(
                self.max_num_reqs, dtype=torch.bool
            ),
            num_decode_draft_tokens=self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            ),
            num_accepted_tokens=self._make_buffer(
                self.max_num_reqs, dtype=torch.int64
            ),
            dcp_local_seq_lens=(
                self._make_buffer(self.max_num_reqs, dtype=torch.int32)
                if self.dcp_world_size > 1
                else None
            ),
            mrope_positions=(
                self._make_buffer(
                    3, self.max_num_tokens + 1, dtype=torch.int64
                )
                if self.uses_mrope
                else None
            ),
            xdrope_positions=(
                self._make_buffer(
                    self.uses_xdrope_dim,
                    self.max_num_tokens + 1,
                    dtype=torch.int64,
                )
                if self.uses_xdrope_dim > 0
                else None
            ),
            is_mm_embed_buffers=is_mm_embed_buffers,
            async_output_copy_stream=(
                torch.cuda.Stream() if self.use_async_scheduling else None
            ),
            prepare_inputs_event=(
                torch.Event() if self.use_async_scheduling else None
            ),
            prepare_inputs_event_recorded=False,
            prepare_inputs_event_fast_path_count=0,
            prepare_inputs_event_wait_count=0,
        )

    def _get_pd_phase_context(
        self,
        phase: str,
        slot: int,
    ) -> PDPhaseModelRunnerContext:
        key = (phase, slot)
        context = self._pd_phase_contexts.get(key)
        if context is None:
            context = self._make_pd_phase_context(phase, slot)
            self._pd_phase_contexts[key] = context
        return context

    def _pd_phase_input_batch_debug(
        self,
        input_batch: InputBatch,
        req_id: str,
    ) -> dict[str, Any]:
        req_index = input_batch.req_id_to_index.get(req_id)
        return {
            "has_req": float(req_index is not None),
            "req_index": -1 if req_index is None else int(req_index),
            "num_reqs": int(input_batch.num_reqs),
            "prev_has_req": float(
                input_batch.prev_req_id_to_index is not None
                and req_id in input_batch.prev_req_id_to_index
            ),
            "prev_sampled_token_ids_shape": (
                list(input_batch.prev_sampled_token_ids.shape)
                if input_batch.prev_sampled_token_ids is not None
                else []
            ),
            "sampled_token_ids_cpu_set": float(
                getattr(input_batch, "sampled_token_ids_cpu", None) is not None
            ),
            "async_copy_ready_event_set": float(
                getattr(input_batch, "async_copy_ready_event", None) is not None
            ),
        }

    def _active_input_batch_req_ids(self) -> list[str]:
        """Return active request ids in their current batch-index order.

        ``InputBatch.req_ids`` can briefly contain removed entries before
        condensation. Phase-local P/D execution also re-adds virtual-context
        rehydrated requests as ``scheduled_new_reqs``. Use the authoritative
        ``req_id_to_index`` mapping when building per-request arrays so stale
        list entries cannot make the scheduled-token vector longer than the
        active batch.
        """
        return [
            req_id
            for req_id, _ in sorted(
                self.input_batch.req_id_to_index.items(),
                key=lambda item: item[1],
            )
        ]

    def _write_pd_phase_handoff_trace(
        self,
        event: str,
        req_id: str,
        record: dict[str, Any],
    ) -> None:
        if not self._pd_phase_handoff_trace_jsonl:
            return
        path = self._pd_phase_handoff_trace_jsonl
        current_phase = self._pd_green_context_execute_phase or ""
        current = self._pd_phase_input_batch_debug(self.input_batch, req_id)
        phase_contexts: dict[str, dict[str, Any]] = {}
        for (phase, slot), context in self._pd_phase_contexts.items():
            phase_contexts[f"{phase}:{slot}"] = self._pd_phase_input_batch_debug(
                context.input_batch, req_id
            )
        payload = {
            "ts": time.monotonic(),
            "event": event,
            "req_id": req_id,
            "phase": current_phase,
            "phase_local_model_runner_enabled": float(
                self._pd_phase_local_model_runner_enabled
            ),
            "current_input_batch": current,
            "phase_contexts": phase_contexts,
            **record,
        }
        writer = self._pd_debug_trace_writer
        if writer is not None:
            writer.submit(path, payload)

    def _warn_pd_cuda_phase_trace_once(self, message: str, exc: Exception) -> None:
        if self._pd_cuda_event_trace_warned:
            return
        self._pd_cuda_event_trace_warned = True
        logger.warning("%s: %s", message, exc)

    @staticmethod
    def _pd_trace_iteration_id(scheduler_output: Any) -> int:
        explicit_iteration_id = getattr(
            scheduler_output, "scheduler_iteration_id", -1
        )
        if explicit_iteration_id is not None:
            try:
                explicit_iteration_id = int(explicit_iteration_id)
                if explicit_iteration_id >= 0:
                    return explicit_iteration_id
            except (TypeError, ValueError):
                pass
        iteration_debug = getattr(scheduler_output, "iteration_debug", None) or {}
        try:
            return int(iteration_debug.get("iteration_id", -1))
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _tiered_budget_from_scheduler_output(
        scheduler_output: Any,
    ) -> dict[str, Any]:
        def as_int(value: Any) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0

        def as_float(value: Any) -> float:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return 0.0

        request_ids = getattr(
            scheduler_output,
            "tiered_pressure_snapshot_request_ids",
            (),
        )
        proactive_snapshot_pages = getattr(
            scheduler_output,
            "tiered_proactive_snapshot_pages",
            (),
        )
        proactive_snapshot_logical_tokens = getattr(
            scheduler_output,
            "tiered_proactive_snapshot_logical_tokens",
            (),
        )
        native_hbm_reload_request_ids = getattr(
            scheduler_output,
            "tiered_native_hbm_reload_request_ids",
            (),
        )
        execution_plans = parse_tiered_execution_plan_entries(
            getattr(
                scheduler_output,
                "tiered_request_execution_plans",
                (),
            )
        )
        normalized_proactive_pages: set[tuple[str, int, int]] = set()
        for entry in proactive_snapshot_pages:
            try:
                request_id, group_id, logical_page = entry
                request_id = str(request_id)
                group_id = int(group_id)
                logical_page = int(logical_page)
            except (TypeError, ValueError):
                continue
            if request_id and group_id >= 0 and logical_page >= 0:
                normalized_proactive_pages.add(
                    (request_id, group_id, logical_page)
                )
        normalized_proactive_logical_tokens: dict[str, int] = {}
        for entry in proactive_snapshot_logical_tokens:
            try:
                request_id, logical_tokens = entry
                request_id = str(request_id)
                logical_tokens = max(0, int(logical_tokens))
            except (TypeError, ValueError):
                continue
            if request_id and logical_tokens > 0:
                normalized_proactive_logical_tokens[request_id] = logical_tokens

        return {
            "dma_budget_bytes": as_int(
                getattr(scheduler_output, "tiered_dma_budget_bytes", 0)
            ),
            "staging_budget_blocks": as_int(
                getattr(scheduler_output, "tiered_staging_budget_blocks", 0)
            ),
            "max_prefill_ring_tokens": as_int(
                getattr(scheduler_output, "tiered_max_prefill_ring_tokens", 0)
            ),
            "decode_tbt_budget_ms": as_float(
                getattr(scheduler_output, "tiered_decode_tbt_budget_ms", 0.0)
            ),
            "pressure_snapshot_request_ids": {
                str(request_id) for request_id in request_ids if request_id
            },
            "proactive_snapshot_pages": normalized_proactive_pages,
            "proactive_snapshot_logical_tokens": (
                normalized_proactive_logical_tokens
            ),
            "native_hbm_reload_request_ids": {
                str(request_id)
                for request_id in native_hbm_reload_request_ids
                if request_id
            },
            "execution_plans": execution_plans,
        }

    def _register_batch_external_tiered_snapshots(
        self,
        scheduler_output: Any,
    ) -> dict[str, int | float]:
        """Submit exact D2H snapshots for victims outside the compute batch.

        ``scheduled_prefill_fit`` deliberately keeps the SRPT compute batch at
        one request.  A lower-priority running prefill can still be the reclaim
        victim, so use its worker-cached block table to submit provider copies
        without adding that request to the model forward. Prefill slots share
        one ordered CUDA stream, so a future-owned victim's copy is enqueued
        after its earlier write; scheduler-side commit still waits for that
        victim to become quiescent.
        """

        started_at = time.monotonic()
        metrics: dict[str, int | float] = {
            "worker_batch_external_d2h_victim_count": 0,
            "worker_batch_external_d2h_page_count": 0,
            "worker_batch_external_d2h_registered_records": 0,
            "worker_batch_external_d2h_missing_request_count": 0,
            "worker_batch_external_d2h_descriptor_hit_count": 0,
            "worker_batch_external_d2h_unsupported_group_count": 0,
            "worker_batch_external_d2h_duration_s": 0.0,
        }
        metrics.update(self._kv_retention_residency_descriptors.metrics())
        if (
            self._pd_green_context_execute_phase != "prefill"
            or not self._virtual_context_enabled
            or not tiered_dram_register_from_hbm_enabled()
        ):
            return metrics

        scheduled_request_ids = set(scheduler_output.num_scheduled_tokens)
        proactive_entries = {
            (request_id, group_id, logical_page)
            for request_id, group_id, logical_page in self._tiered_scheduler_budget.get(
                "proactive_snapshot_pages",
                set(),
            )
            if request_id not in scheduled_request_ids
        }
        if not proactive_entries:
            return metrics

        logical_tokens_by_request = dict(
            self._tiered_scheduler_budget.get(
                "proactive_snapshot_logical_tokens",
                {},
            )
        )
        pages_by_group: defaultdict[int, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for request_id, group_id, logical_page in proactive_entries:
            pages_by_group[group_id][request_id].add(logical_page)

        virtual_context = get_virtual_context_table()
        if virtual_context is None:
            return metrics

        victim_ids: set[str] = set()
        requested_page_count = 0
        registered_records = 0
        for group_id, requested_pages_by_request in pages_by_group.items():
            if group_id < 0 or group_id >= len(
                self.kv_cache_config.kv_cache_groups
            ):
                metrics["worker_batch_external_d2h_unsupported_group_count"] += 1
                continue
            group = self.kv_cache_config.kv_cache_groups[group_id]
            if not group.layer_names:
                metrics["worker_batch_external_d2h_unsupported_group_count"] += 1
                continue
            kv_cache_spec = group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[
                    group.layer_names[0]
                ]
            if not isinstance(kv_cache_spec, AttentionSpec):
                metrics["worker_batch_external_d2h_unsupported_group_count"] += 1
                continue

            request_ids: list[str] = []
            block_rows: list[list[int]] = []
            seq_lens: list[int] = []
            exact_pages_by_request: dict[str, set[int]] = {}
            for request_id, logical_pages in requested_pages_by_request.items():
                request_state = self.requests.get(request_id)
                descriptor = None
                if request_state is None:
                    descriptor = self._kv_retention_residency_descriptors.lookup(
                        request_id
                    )
                    if descriptor is not None:
                        metrics[
                            "worker_batch_external_d2h_descriptor_hit_count"
                        ] += 1
                logical_tokens = logical_tokens_by_request.get(request_id, 0)
                block_ids_by_group = (
                    request_state.block_ids
                    if request_state is not None
                    else descriptor.block_ids
                    if descriptor is not None
                    else ()
                )
                if (
                    not block_ids_by_group
                    or logical_tokens <= 0
                    or group_id >= len(block_ids_by_group)
                ):
                    metrics[
                        "worker_batch_external_d2h_missing_request_count"
                    ] += 1
                    continue
                block_ids = list(block_ids_by_group[group_id])
                valid_pages = {
                    page
                    for page in logical_pages
                    if 0 <= page < len(block_ids) and block_ids[page] >= 0
                }
                if not valid_pages:
                    continue
                request_ids.append(request_id)
                block_rows.append(block_ids)
                seq_lens.append(logical_tokens)
                exact_pages_by_request[request_id] = valid_pages
                victim_ids.add(request_id)
                requested_page_count += len(valid_pages)

            if not request_ids:
                continue
            table_width = max(len(row) for row in block_rows)
            padded_rows = [
                row + [-1] * (table_width - len(row)) for row in block_rows
            ]
            block_table = torch.tensor(padded_rows, dtype=torch.int32)
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
            query_start_loc = torch.zeros(
                len(request_ids) + 1,
                dtype=torch.int32,
            )
            force_request_ids = set(request_ids)
            source_page_plan = build_tiered_d2h_source_plan(
                request_ids=request_ids,
                block_table=block_table,
                seq_lens=seq_lens_tensor,
                query_start_loc=query_start_loc,
                block_size=kv_cache_spec.block_size,
                copy_policy="all",
                allowed_request_ids=force_request_ids,
                force_all_request_ids=force_request_ids,
                max_pages=0,
                exact_force_pages_by_request=exact_pages_by_request,
            )
            if not source_page_plan.pages:
                continue

            for layer_name in group.layer_names:
                layer = self.compilation_config.static_forward_context.get(
                    layer_name
                )
                layer_kv_caches = getattr(layer, "kv_cache", None)
                if not layer_kv_caches:
                    metrics[
                        "worker_batch_external_d2h_unsupported_group_count"
                    ] += 1
                    continue
                kv_cache = layer_kv_caches[0]
                if (
                    not isinstance(kv_cache, torch.Tensor)
                    or kv_cache.dim() != 5
                    or kv_cache.shape[1] != 2
                ):
                    metrics[
                        "worker_batch_external_d2h_unsupported_group_count"
                    ] += 1
                    continue
                key_cache, value_cache = kv_cache.unbind(1)
                layer_id = virtual_context_layer_id(layer_name)
                page_versions = virtual_context.pin_attention_page_versions(
                    request_ids,
                    group_id=group_id,
                    layer_id=layer_id,
                )
                registered_records += register_hbm_kv_cache_pages(
                    layer_name=layer_name,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table,
                    seq_lens=seq_lens_tensor,
                    request_ids=request_ids,
                    page_versions_by_request=page_versions or None,
                    allowed_request_ids=force_request_ids,
                    force_all_request_ids=force_request_ids,
                    exact_force_pages_by_request=exact_pages_by_request,
                    source_page_plan=source_page_plan,
                )

        metrics["worker_batch_external_d2h_victim_count"] = len(victim_ids)
        metrics["worker_batch_external_d2h_page_count"] = requested_page_count
        metrics["worker_batch_external_d2h_registered_records"] = (
            registered_records
        )
        metrics["worker_batch_external_d2h_duration_s"] = (
            time.monotonic() - started_at
        )
        metrics.update(self._kv_retention_residency_descriptors.metrics())
        return metrics

    def _write_pd_cuda_event_trace(self, payload: dict[str, Any]) -> None:
        if not self._pd_cuda_event_trace_jsonl:
            return
        writer = self._pd_debug_trace_writer
        if writer is not None:
            writer.submit(self._pd_cuda_event_trace_jsonl, payload)

    def _flush_pd_cuda_event_trace_completed(self) -> None:
        if (
            not self._pd_cuda_event_trace_jsonl
            or not self._pd_cuda_event_trace_pending
        ):
            return

        completed: list[dict[str, Any]] = []
        try:
            with self._pd_cuda_event_trace_lock:
                remaining: list[dict[str, Any]] = []
                for record in self._pd_cuda_event_trace_pending:
                    end_event = record["_end_event"]
                    if not end_event.query():
                        remaining.append(record)
                        continue
                    payload = {
                        key: value
                        for key, value in record.items()
                        if not key.startswith("_")
                    }
                    payload["host_completion_observed_ts"] = time.monotonic()
                    payload["device_elapsed_ms"] = record[
                        "_start_event"
                    ].elapsed_time(end_event)
                    epoch_event = record.get("_epoch_event")
                    if epoch_event is not None:
                        payload["device_start_ms_from_epoch"] = (
                            epoch_event.elapsed_time(record["_start_event"])
                        )
                        payload["device_end_ms_from_epoch"] = (
                            epoch_event.elapsed_time(end_event)
                        )
                        payload["device_interval_basis"] = (
                            "cuda_event_shared_epoch"
                        )
                    payload["dropped_records_total"] = (
                        self._pd_cuda_event_trace_dropped
                    )
                    completed.append(payload)
                self._pd_cuda_event_trace_pending = remaining
        except Exception as exc:
            self._warn_pd_cuda_phase_trace_once(
                "Failed to flush P/D CUDA event trace", exc
            )
            return
        for payload in completed:
            self._write_pd_cuda_event_trace(payload)

    def _select_pd_cuda_event_trace(
        self,
        kind: str,
        phase: str | None,
    ) -> tuple[bool, int, int]:
        normalized_kind = kind.strip().lower()
        normalized_phase = (phase or "").strip().lower()
        if (
            self._pd_cuda_event_trace_kinds
            and normalized_kind not in self._pd_cuda_event_trace_kinds
        ):
            return False, 0, 0
        if (
            self._pd_cuda_event_trace_phases
            and normalized_phase not in self._pd_cuda_event_trace_phases
        ):
            return False, 0, 0
        every_n = self._pd_cuda_event_trace_every_n
        if normalized_phase == "decode":
            every_n = self._pd_cuda_event_trace_decode_every_n
        elif normalized_phase == "prefill":
            every_n = self._pd_cuda_event_trace_prefill_every_n
        key = (normalized_phase, normalized_kind)
        with self._pd_cuda_event_trace_lock:
            self._pd_cuda_event_trace_seen[key] += 1
            sequence = self._pd_cuda_event_trace_seen[key]
        return (sequence - 1) % every_n == 0, sequence, every_n

    def _begin_pd_cuda_event_trace(
        self,
        kind: str,
        phase: str | None,
        scheduler_output: Any,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any] | None:
        if not self._pd_cuda_event_trace_jsonl or not torch.cuda.is_available():
            return None

        selected, sequence, every_n = self._select_pd_cuda_event_trace(
            kind, phase
        )
        if not selected:
            return None

        self._flush_pd_cuda_event_trace_completed()
        try:
            stream = torch.cuda.current_stream()
            epoch_event, epoch_id = _get_pd_cuda_trace_epoch(stream)
            stream.wait_event(epoch_event)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            host_enqueue_start_ts = time.monotonic()
            start_event.record(stream)
            return {
                "kind": kind,
                "phase": phase or "",
                "iteration_id": self._pd_trace_iteration_id(scheduler_output),
                "num_reqs": int(num_reqs),
                "num_tokens": int(num_tokens),
                "trace_sequence": int(sequence),
                "trace_every_n": int(every_n),
                "pid": os.getpid(),
                "phase_context_active": float(
                    _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get() is not None
                ),
                "phase_context_phase": (
                    _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get().phase
                    if _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get() is not None
                    else ""
                ),
                "device_index": getattr(stream, "device_index", None),
                "stream_cuda_stream": int(getattr(stream, "cuda_stream", 0)),
                "device_epoch_id": epoch_id,
                "host_enqueue_start_ts": host_enqueue_start_ts,
                "host_start_event_recorded_ts": time.monotonic(),
                "_stream": stream,
                "_epoch_event": epoch_event,
                "_start_event": start_event,
                "_end_event": end_event,
            }
        except Exception as exc:
            self._warn_pd_cuda_phase_trace_once(
                "Failed to start P/D CUDA event trace", exc
            )
            return None

    def _end_pd_cuda_event_trace(self, record: dict[str, Any] | None) -> None:
        if record is None:
            return
        try:
            stream = record.get("_stream")
            if stream is None:
                stream = torch.cuda.current_stream()
            record["host_enqueue_end_ts"] = time.monotonic()
            record["_end_event"].record(stream)
            record["host_end_event_recorded_ts"] = time.monotonic()
            record["end_stream_cuda_stream"] = int(
                getattr(stream, "cuda_stream", 0)
            )
            with self._pd_cuda_event_trace_lock:
                self._pd_cuda_event_trace_pending.append(record)
                overflow = (
                    len(self._pd_cuda_event_trace_pending)
                    - self._pd_cuda_event_trace_max_pending
                )
                if overflow > 0:
                    del self._pd_cuda_event_trace_pending[:overflow]
                    self._pd_cuda_event_trace_dropped += overflow
            self._flush_pd_cuda_event_trace_completed()
        except Exception as exc:
            self._warn_pd_cuda_phase_trace_once(
                "Failed to finish P/D CUDA event trace", exc
            )

    @contextmanager
    def _pd_cuda_phase_trace(
        self,
        kind: str,
        phase: str | None,
        scheduler_output: Any,
        num_reqs: int,
        num_tokens: int,
    ) -> Iterator[None]:
        if (
            not self._pd_nvtx_phase_trace_enabled
            and not self._pd_cuda_event_trace_jsonl
        ):
            yield
            return

        use_cuda = torch.cuda.is_available()
        nvtx_pushed = False
        if self._pd_nvtx_phase_trace_enabled and use_cuda:
            label = (
                f"pd_{kind} phase={phase or 'none'} "
                f"iter={self._pd_trace_iteration_id(scheduler_output)} "
                f"reqs={num_reqs} tokens={num_tokens}"
            )
            try:
                torch.cuda.nvtx.range_push(label)
                nvtx_pushed = True
            except Exception as exc:
                self._warn_pd_cuda_phase_trace_once(
                    "Failed to push P/D NVTX phase trace", exc
                )
        event_record = self._begin_pd_cuda_event_trace(
            kind, phase, scheduler_output, num_reqs, num_tokens
        )
        try:
            yield
        finally:
            self._end_pd_cuda_event_trace(event_record)
            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception as exc:
                    self._warn_pd_cuda_phase_trace_once(
                        "Failed to pop P/D NVTX phase trace", exc
                    )

    def _warn_pd_decode_gpu_component_trace_once(
        self, message: str, exc: Exception
    ) -> None:
        if self._pd_decode_gpu_component_trace_warned:
            return
        self._pd_decode_gpu_component_trace_warned = True
        logger.warning("%s: %s", message, exc)

    def _write_pd_decode_gpu_component_trace(
        self, payload: dict[str, Any]
    ) -> None:
        path = self._pd_decode_gpu_component_trace_jsonl
        if not path:
            return
        try:
            writer = self._pd_debug_trace_writer
            if writer is not None:
                writer.submit(path, payload)
        except Exception as exc:
            self._warn_pd_decode_gpu_component_trace_once(
                "Failed to write Decode GPU component trace", exc
            )

    def _flush_pd_decode_gpu_component_trace_completed(self) -> None:
        if not self._pd_decode_gpu_component_trace_jsonl:
            return
        completed_payloads: list[dict[str, Any]] = []
        try:
            with self._pd_decode_gpu_component_trace_lock:
                if not self._pd_decode_gpu_component_trace_pending:
                    return
                remaining: list[_CudaModuleTimingCapture] = []
                for capture in self._pd_decode_gpu_component_trace_pending:
                    if not _CudaModuleTimingHooks.capture_ready(capture):
                        remaining.append(capture)
                        continue
                    metrics = _CudaModuleTimingHooks.capture_metrics(capture)
                    if not self._pd_decode_gpu_component_trace_include_layers:
                        metrics.pop("layers", None)
                    completed_payloads.append(
                        {
                            **capture.metadata,
                            **metrics,
                            "host_completion_observed_ts": time.monotonic(),
                            "dropped_records_total": (
                                self._pd_decode_gpu_component_trace_dropped
                            ),
                        }
                    )
                self._pd_decode_gpu_component_trace_pending = remaining
        except Exception as exc:
            self._warn_pd_decode_gpu_component_trace_once(
                "Failed to flush Decode GPU component trace", exc
            )
            return
        for payload in completed_payloads:
            self._write_pd_decode_gpu_component_trace(payload)

    def _enqueue_pd_decode_gpu_component_capture(
        self, capture: _CudaModuleTimingCapture | None
    ) -> None:
        if capture is None or not self._pd_decode_gpu_component_trace_jsonl:
            return
        with self._pd_decode_gpu_component_trace_lock:
            self._pd_decode_gpu_component_trace_pending.append(capture)
            overflow = (
                len(self._pd_decode_gpu_component_trace_pending)
                - self._pd_decode_gpu_component_trace_max_pending
            )
            if overflow > 0:
                del self._pd_decode_gpu_component_trace_pending[:overflow]
                self._pd_decode_gpu_component_trace_dropped += overflow
        # query() is non-blocking. Completed captures are serialized only after
        # their end event has fired, so this does not join the Decode stream.
        self._flush_pd_decode_gpu_component_trace_completed()

    def _select_pd_decode_gpu_component_capture(self) -> tuple[bool, int]:
        if (
            not self._pd_decode_gpu_component_trace_jsonl
            or self._pd_green_context_execute_phase != "decode"
        ):
            return False, 0
        with self._pd_decode_gpu_component_trace_lock:
            self._pd_decode_gpu_component_trace_seen += 1
            sequence = self._pd_decode_gpu_component_trace_seen
        selected = (
            (sequence - 1) % self._pd_decode_gpu_component_trace_every_n == 0
        )
        return selected, sequence

    def finalize_pd_decode_gpu_component_trace(self) -> None:
        """Flush all opt-in worker trace tails at server shutdown."""
        try:
            has_cuda_event_tail = bool(
                self._pd_decode_gpu_component_trace_jsonl
                or self._pd_cuda_event_trace_jsonl
            )
            if has_cuda_event_tail and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            if self._pd_decode_gpu_component_trace_jsonl:
                self._flush_pd_decode_gpu_component_trace_completed()
            if self._pd_cuda_event_trace_jsonl:
                self._flush_pd_cuda_event_trace_completed()
        except Exception as exc:
            self._warn_pd_decode_gpu_component_trace_once(
                "Failed to finalize P/D worker traces", exc
            )
        finally:
            writer = self._pd_debug_trace_writer
            if writer is not None:
                writer.close()
                self._pd_debug_trace_writer = None

    def _pd_decode_gpu_component_metadata(
        self,
        scheduler_output: Any,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
        request_debug = getattr(scheduler_output, "request_debug", None) or {}
        decode_request_ids: list[str] = []
        decode_context_tokens: list[int] = []
        ring_request_count = 0
        for request_id, record in request_debug.items():
            if not isinstance(record, dict):
                continue
            try:
                decode_tokens = int(record.get("decode_tokens", 0) or 0)
            except (TypeError, ValueError):
                decode_tokens = 0
            if str(record.get("phase", "")) == "decode" or decode_tokens > 0:
                decode_request_ids.append(str(request_id))
                try:
                    decode_context_tokens.append(
                        max(
                            0,
                            int(
                                record.get("num_computed_tokens_before", 0)
                                or 0
                            ),
                        )
                    )
                except (TypeError, ValueError):
                    decode_context_tokens.append(0)
                try:
                    dram_pages = int(
                        record.get("virtual_context_request_dram_pages", 0)
                        or 0
                    )
                    deferred_tokens = int(
                        record.get(
                            "virtual_context_deferred_logical_computed_tokens",
                            0,
                        )
                        or 0
                    )
                except (TypeError, ValueError):
                    dram_pages = deferred_tokens = 0
                ring_request_count += int(dram_pages > 0 or deferred_tokens > 0)

        iteration_debug = getattr(scheduler_output, "iteration_debug", None) or {}
        green_state = green_context_state()
        current_stream = torch.cuda.current_stream()
        trial_phase = ""
        if any("-measurement-" in request_id for request_id in decode_request_ids):
            trial_phase = "measurement"
        elif any("-warmup-" in request_id for request_id in decode_request_ids):
            trial_phase = "warmup"
        return {
            "trace_schema": "pd_decode_gpu_components_v1",
            "trace_schema_version": 1,
            "host_enqueue_start_ts": time.monotonic(),
            "pid": os.getpid(),
            "device_index": getattr(current_stream, "device_index", None),
            "stream_cuda_stream": int(
                getattr(current_stream, "cuda_stream", 0)
            ),
            "phase": "decode",
            "iteration_id": self._pd_trace_iteration_id(scheduler_output),
            "phase_slot": int(context.slot if context is not None else 0),
            "execution_key": (
                f"decode:{context.slot}"
                if context is not None and context.slot > 0
                else "decode"
            ),
            "num_reqs": int(num_reqs),
            "num_tokens": int(num_tokens),
            "decode_request_count": len(decode_request_ids) or int(num_reqs),
            "decode_request_ids": decode_request_ids,
            "trial_phase": trial_phase,
            "decode_context_min_tokens": (
                min(decode_context_tokens) if decode_context_tokens else 0
            ),
            "decode_context_max_tokens": (
                max(decode_context_tokens) if decode_context_tokens else 0
            ),
            "decode_context_mean_tokens": (
                sum(decode_context_tokens) / len(decode_context_tokens)
                if decode_context_tokens
                else 0.0
            ),
            "context_ring_decode_request_count": ring_request_count,
            "sm_profile_id": str(
                getattr(scheduler_output, "pd_sm_profile_id", "") or ""
            ),
            "sm_profile_epoch": int(
                getattr(scheduler_output, "pd_sm_profile_epoch", 0)
            ),
            "decode_sms": int(
                getattr(scheduler_output, "pd_sm_decode_sms", 0)
                or green_state.get("green_decode_sms", 0)
            ),
            "prefill_sms": int(
                getattr(scheduler_output, "pd_sm_prefill_sms", 0)
                or green_state.get("green_prefill_sms", 0)
            ),
            "green_context_active": float(
                green_state.get("green_context_active", 0.0)
            ),
            "green_sm_count_verified": float(
                green_state.get("green_sm_count_verified", 0.0)
            ),
            "prefill_queue_depth": float(
                iteration_debug.get("pd_phase_queue_prefill_depth", 0.0) or 0.0
            ),
            "decode_queue_depth": float(
                iteration_debug.get("pd_phase_queue_decode_depth", 0.0) or 0.0
            ),
            "prefill_inflight_at_decode_submit": float(
                iteration_debug.get(
                    "pd_component_prefill_inflight_at_submit", 0.0
                )
                or 0.0
            ),
            "prefill_not_ready_at_decode_submit": float(
                iteration_debug.get(
                    "pd_component_prefill_not_ready_at_submit", 0.0
                )
                or 0.0
            ),
            "paired_prefill_submit": float(
                iteration_debug.get(
                    "pd_component_paired_prefill_submit", 0.0
                )
                or 0.0
            ),
            "prefill_overlap_candidate": float(
                iteration_debug.get(
                    "pd_component_prefill_overlap_candidate", 0.0
                )
                or 0.0
            ),
            "component_trace_forced_eager": 1.0,
            "model_dtype": str(self.dtype),
        }

    def _replace_output_token_ids_in_place(
        self,
        req_state: CachedRequestState,
        output_token_ids: Sequence[int],
    ) -> None:
        # InputBatch and SamplingMetadata keep references to this list. Replacing
        # the object can leave another phase with a stale token-history view.
        req_state.output_token_ids[:] = output_token_ids

    def _apply_owned_output_token_refresh(
        self,
        req_data: Any,
        req_id: str,
        req_state: CachedRequestState,
        num_computed_tokens: int,
        num_output_tokens: int,
        source_generation: int,
    ) -> None:
        if source_generation <= 0:
            raise OutputPlaceholderOwnershipError(
                "A phase-local output refresh requires a positive committed "
                f"generation: request={req_id}, generation={source_generation}."
            )
        all_token_ids = req_data.all_token_ids.get(req_id)
        if all_token_ids is None:
            raise OutputPlaceholderOwnershipError(
                "The committed owner generation was provided without its "
                f"token history: request={req_id}, "
                f"generation={source_generation}."
            )
        committed_output_ids = all_token_ids[-num_output_tokens:]
        all_token_ids_len = len(all_token_ids)

        if len(committed_output_ids) != num_output_tokens:
            self._write_pd_phase_handoff_trace(
                "owned_output_token_refresh_failed",
                req_id,
                {
                    "source_generation": int(source_generation),
                    "num_computed_tokens": int(num_computed_tokens),
                    "num_output_tokens": int(num_output_tokens),
                    "req_state_num_tokens": int(req_state.num_tokens),
                    "req_state_output_token_ids_len": int(
                        len(req_state.output_token_ids)
                    ),
                    "all_token_ids_len": int(all_token_ids_len),
                    "committed_output_ids_len": int(
                        len(committed_output_ids)
                    ),
                },
            )
            raise OutputPlaceholderOwnershipError(
                "Cannot apply committed phase-local output tokens for "
                f"{req_id}: generation {source_generation} provided "
                f"{len(committed_output_ids)} tokens, expected "
                f"{num_output_tokens}."
            )

        vocab_size = self.input_batch.vocab_size
        invalid_output_ids = [
            (index, int(token_id))
            for index, token_id in enumerate(committed_output_ids)
            if int(token_id) < 0 or int(token_id) >= vocab_size
        ]
        if invalid_output_ids:
            invalid_preview = ",".join(
                f"{index}:{token_id}"
                for index, token_id in invalid_output_ids[:8]
            )
            self._write_pd_phase_handoff_trace(
                "owned_output_token_refresh_invalid",
                req_id,
                {
                    "source_generation": int(source_generation),
                    "num_computed_tokens": int(num_computed_tokens),
                    "num_output_tokens": int(num_output_tokens),
                    "all_token_ids_len": int(all_token_ids_len),
                    "invalid_output_token_count": len(invalid_output_ids),
                    "invalid_output_token_preview": invalid_preview,
                    "vocab_size": int(vocab_size),
                },
            )
            raise OutputPlaceholderOwnershipError(
                "Cannot apply committed phase-local output tokens for "
                f"{req_id}: generation {source_generation} contains "
                f"out-of-vocabulary token IDs ({invalid_preview}); valid "
                f"range is [0, {vocab_size})."
            )

        current_generation = int(req_state.output_token_generation)
        if current_generation > source_generation:
            raise OutputPlaceholderOwnershipError(
                "A stale phase-local token history attempted to overwrite a "
                "newer worker state: "
                f"request={req_id}, worker_generation={current_generation}, "
                f"source_generation={source_generation}."
            )
        if current_generation == source_generation:
            if list(req_state.output_token_ids) != committed_output_ids:
                raise OutputPlaceholderOwnershipError(
                    "The same committed output generation has different token "
                    f"contents: request={req_id}, "
                    f"generation={source_generation}."
                )
            return

        if self._pd_phase_handoff_trace_jsonl:
            self._write_pd_phase_handoff_trace(
                "owned_output_token_refresh",
                req_id,
                {
                    "source_generation": int(source_generation),
                    "previous_generation": current_generation,
                    "num_computed_tokens": int(num_computed_tokens),
                    "num_output_tokens": int(num_output_tokens),
                    "old_output_token_ids_len": int(len(req_state.output_token_ids)),
                    "all_token_ids_len": int(all_token_ids_len),
                    "refresh_slice_len": int(num_output_tokens),
                },
            )
        self._replace_output_token_ids_in_place(
            req_state,
            committed_output_ids,
        )
        req_state.output_token_generation = source_generation

    @contextmanager
    def _use_pd_phase_model_runner_context(
        self,
        execution_key: str | None,
    ) -> Iterator[PDPhaseModelRunnerContext | None]:
        phase, slot = self._parse_pd_phase_execution_key(execution_key)
        if (
            not self._pd_phase_local_model_runner_enabled
            or phase not in {"decode", "prefill"}
        ):
            yield None
            return

        context = self._get_pd_phase_context(phase, slot)
        token = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.set(context)
        try:
            yield context
        finally:
            _PD_ACTIVE_MODEL_RUNNER_CONTEXT.reset(token)

    def _claim_pd_execute_state(
        self,
        execution_key: str,
        scheduler_output: "SchedulerOutput",
        context: PDPhaseModelRunnerContext,
        binding: KVRetentionExecutionBinding | None = None,
    ) -> None:
        """Stamp ownership before preparation can fail, without replacing live work."""
        owner = KVRetentionPhaseOwner.from_output(
            execution_key, scheduler_output, binding)
        if binding is not None:
            binding.validate_submission(execution_key, scheduler_output)
            if binding.slot_generation <= context.kv_retention_last_slot_generation:
                raise RuntimeError("stale Prefill slot generation")
            if context.execute_model_state is not None:
                raise RuntimeError("Prefill execute cannot replace pending sampling")
            store = getattr(self, "_kv_retention_parked_store", None)
            item = store.entries.get(binding.work.work_id) if store is not None else None
            if item is not None:
                ticket, saved = item
                if context.ttft_prefill_continuation is not None:
                    raise RuntimeError("P1 restore cannot replace another live continuation")
                binding.validate_successor(ticket.binding, ticket.next_layer)
                if saved.scheduler_output is not scheduler_output:
                    raise RuntimeError("P1 restore changed original output identity")
                # The context becomes the owner before any mutable input work.
                # Restoration runs on the selected target stream below.
                saved = store.take(ticket.binding, binding)
                context.ttft_prefill_continuation = saved
                context.kv_retention_execute_owner = KVRetentionPhaseOwner.from_output(
                    execution_key, scheduler_output, ticket.binding)
                saved.kv_sampling_restore_pending = True
            continuation = context.ttft_prefill_continuation
            if continuation is not None:
                previous = context.kv_retention_execute_owner
                if (previous is None or previous.binding is None
                        or continuation.scheduler_output is not scheduler_output
                        or continuation.boundary.kv_retention_binding != previous.binding):
                    raise RuntimeError("Prefill continuation binding changed")
                binding.validate_successor(previous.binding,
                                           continuation.boundary.next_layer)
            elif binding.segment_id != 1 or binding.start_layer != 0:
                raise RuntimeError("Prefill resume has no retained continuation")
            context.kv_retention_execute_owner = owner
            context.kv_retention_last_slot_generation = binding.slot_generation
            return
        for state in (context.execute_model_state, context.ttft_prefill_continuation):
            if state is not None and (
                state.scheduler_output is not scheduler_output
                or context.kv_retention_execute_owner != owner
            ):
                raise RuntimeError(
                    "phase execute attempted to replace a live owner: "
                    f"current={context.kv_retention_execute_owner!r}, new={owner!r}"
                )
        context.kv_retention_execute_owner = owner

    def clear_pd_phase_execute_state(
        self, execution_key: str, expected_owner: KVRetentionPhaseOwner
    ) -> bool:
        """Dispose one exact owner after the caller's GPU-completion proof.

        Native continuations need closing too. A failed close keeps ownership
        visible; a stale cleanup is a no-op, even after the slot is reused.
        """
        phase, slot = expected_owner.validate_key(execution_key)
        store = getattr(self, "_kv_retention_parked_store", None)
        if (store is not None and expected_owner.binding is not None
                and expected_owner.binding.work.work_id in store.entries):
            # A lives outside slot0; C's buffers and generation are untouched.
            return store.close(expected_owner.binding)
        context = self._pd_phase_contexts.get((phase, slot))
        if context is None or context.kv_retention_execute_owner != expected_owner:
            return False
        for state in (context.execute_model_state, context.ttft_prefill_continuation):
            if state is not None and KVRetentionPhaseOwner.from_output(
                execution_key, state.scheduler_output, expected_owner.binding
            ) != expected_owner:
                raise RuntimeError("phase cleanup retained-state ownership mismatch")
        continuation = context.ttft_prefill_continuation
        if continuation is not None:
            if continuation.kv_offload is not None:
                continuation.kv_offload.close()
            continuation.continuation.close()
        context.execute_model_state = None
        context.ttft_prefill_continuation = None
        context.kv_connector_output = None
        context._pd_green_context_sample_phase = None
        context._pd_green_context_sample_profile_id = ""
        context._pd_green_context_sample_profile_epoch = 0
        context._pd_decode_gpu_component_capture = None
        context.kv_retention_execute_owner = None
        return True

    def _validate_kv_retention_sampling(
        self, execution_key: str, context: PDPhaseModelRunnerContext | None,
        binding: KVRetentionExecutionBinding | None,
    ) -> None:
        owner = getattr(context, "kv_retention_execute_owner", None)
        current = getattr(owner, "binding", None)
        if current != binding:
            raise RuntimeError("Prefill sampling does not own the active segment")
        if binding is not None:
            state = context.execute_model_state
            if state is None or context.ttft_prefill_continuation is not None:
                raise RuntimeError("Prefill sampling requires final execute state")
            binding.validate_output(execution_key, state.scheduler_output)

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties"""
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self._kv_retention_residency_descriptors.discard(req_id)
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        for req_id in scheduler_output.residency_deferred_req_ids or ():
            request_state = self.requests.get(req_id)
            if request_state is not None:
                self._kv_retention_residency_descriptors.capture(
                    request_id=req_id,
                    block_ids=request_state.block_ids,
                    num_computed_tokens=request_state.num_computed_tokens,
                )
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
            self.input_batch.remove_request(req_id)

        # Virtual-context rehydration sends a previously seen request through
        # scheduled_new_reqs so the worker can rebuild phase-local state after
        # HBM residency was released. If this phase-local batch still has a row
        # for that request, replace it instead of letting duplicate/stale rows
        # survive into _prepare_inputs().
        scheduled_new_req_ids = {
            new_req_data.req_id
            for new_req_data in scheduler_output.scheduled_new_reqs
        }
        for req_id in scheduled_new_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            self._kv_retention_residency_descriptors.discard(req_id)
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if self._pd_phase_handoff_trace_jsonl:
                other_phase_presence = {
                    f"{phase}:{slot}": float(
                        req_id in context.input_batch.req_id_to_index
                    )
                    for (phase, slot), context in self._pd_phase_contexts.items()
                    if phase != (self._pd_green_context_execute_phase or "")
                }
                if (
                    self._pd_green_context_execute_phase in {"decode", "prefill"}
                    and req_index is None
                    and any(other_phase_presence.values())
                ):
                    self._write_pd_phase_handoff_trace(
                        "phase_local_resume_from_other_phase",
                        req_id,
                        {
                            "num_computed_tokens": int(num_computed_tokens),
                            "num_output_tokens": int(num_output_tokens),
                            "resumed_from_preemption": float(
                                resumed_from_preemption
                            ),
                            "req_state_num_tokens": int(req_state.num_tokens),
                            "req_state_num_computed_tokens": int(
                                req_state.num_computed_tokens
                            ),
                            "req_state_output_token_ids_len": int(
                                len(req_state.output_token_ids)
                            ),
                            "other_phase_presence": other_phase_presence,
                        },
                    )

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # fist step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt lenth),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step does't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption, or the scheduler is
                # forcing a full block-table replacement for a virtual-context
                # DRAM->HBM native decode handoff.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # A phase-local P/D transfer accepts only a scheduler-
                    # committed generation. Upstream/default async scheduling
                    # retains its ordinary full-token-state restoration.
                    if scheduler_output.pd_output_ownership_enabled:
                        source_generation = (
                            scheduler_output
                            .pd_output_token_source_generations.get(req_id)
                        )
                        if source_generation is None:
                            raise OutputPlaceholderOwnershipError(
                                "A resumed phase-local request has output "
                                "tokens but no committed source generation: "
                                f"request={req_id}, "
                                f"num_output_tokens={num_output_tokens}."
                            )
                        self._apply_owned_output_token_refresh(
                            req_data,
                            req_id,
                            req_state,
                            num_computed_tokens,
                            num_output_tokens,
                            source_generation,
                        )
                    else:
                        resumed_token_ids = req_data.all_token_ids[req_id]
                        self._replace_output_token_ids_in_place(
                            req_state,
                            resumed_token_ids[-num_output_tokens:],
                        )

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                if resumed_from_preemption:
                    self.input_batch.block_table.add_row(new_block_ids, req_index)
                else:
                    self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor
    ) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # Find the number of accepted tokens for each sequence.
        num_accepted_tokens = (
            (
                torch.cat(
                    [
                        output_token_ids,
                        torch.full(
                            (output_token_ids.size(0), 1),
                            -1,
                            device=output_token_ids.device,
                        ),
                    ],
                    dim=1,
                )
                == -1
            )
            .int()
            .argmax(-1)
            .cpu()
            .numpy()
        )
        for i, num_tokens in enumerate(num_accepted_tokens):
            self.input_batch.num_accepted_tokens_cpu[i] = num_tokens

    def _init_mrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        assert req_state.prompt_token_ids is not None, (
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_model = cast(SupportsMRoPE, model)

        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                req_state.mm_features,
            )
        )

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        xdrope_model = cast(SupportsXDRoPE, model)
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[MultiModalKwargsItem]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append(feature.data)

        # Input all modalities at once
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_commmon_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0],
                non_blocking=True,
            )
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # Zero out buffer for padding requests that are not actually scheduled (CGs)
        self.encoder_seq_lens.np[:num_reqs] = 0
        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # Get the total number of encoder input tokens for running encoder requests
            # whether encoding is finished or not so that cross-attention knows how
            # many encoder tokens to attend to.
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens

        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """
        :return: tuple[
            logits_indices, spec_decode_metadata,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Calculate XD-RoPE positions.
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        active_req_ids = self._active_input_batch_req_ids()
        num_tokens = [self.requests[r].num_tokens for r in active_req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        self.discard_request_mask.np[:num_reqs] = (
            self.seq_lens.np[:num_reqs] < num_tokens_np
        )
        self._last_pd_output_owner_prefill_sample_unmask_count = 0
        if (
            scheduler_output.pd_output_ownership_enabled
            and self._pd_phase_local_model_runner_enabled
            and self._pd_green_context_execute_phase == "prefill"
        ):
            prefill_owner_ids = {
                req_id
                for req_id, (_, _, owner_phase, owner_count) in (
                    scheduler_output.pd_output_placeholder_owners.items()
                )
                if owner_phase == "prefill" and owner_count > 0
            }
            final_prefill_ids = set(
                scheduler_output.pd_ttft_handoff_request_ids
            )
            if prefill_owner_ids != final_prefill_ids:
                raise OutputPlaceholderOwnershipError(
                    "Worker final-Prefill owners differ from the handoff "
                    "contract: "
                    f"owner_ids={tuple(sorted(prefill_owner_ids))}, "
                    f"handoff_ids={tuple(sorted(final_prefill_ids))}."
                )
            missing_owner_ids = prefill_owner_ids - set(active_req_ids)
            if missing_owner_ids:
                raise OutputPlaceholderOwnershipError(
                    "Final-Prefill output owners are absent from the worker "
                    f"batch: request_ids={tuple(sorted(missing_owner_ids))}."
                )
            owner_mask = np.fromiter(
                (req_id in prefill_owner_ids for req_id in active_req_ids),
                dtype=np.bool_,
                count=num_reqs,
            )
            unmask = self.discard_request_mask.np[:num_reqs] & owner_mask
            unmask_count = int(np.count_nonzero(unmask))
            if unmask_count:
                self.discard_request_mask.np[:num_reqs] &= ~owner_mask
            self._last_pd_output_owner_prefill_sample_unmask_count = (
                unmask_count
            )
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # Hot-Swap lora model
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )

        return (
            logits_indices,
            spec_decode_metadata,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_block_table_and_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
                blk_table_cpu = None
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)
                blk_table_cpu = blk_table.get_cpu_tensor()[:num_reqs_padded]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)

            return blk_table_tensor, slot_mapping, blk_table_cpu

        (
            block_table_gid_0,
            slot_mapping_gid_0,
            block_table_cpu_gid_0,
        ) = _get_block_table_and_slot_mapping(0)
        if self.model_config.enable_return_routed_experts:
            self.slot_mapping = slot_mapping_gid_0[:num_tokens].cpu().numpy()
        active_req_ids = self._active_input_batch_req_ids()
        request_ids = active_req_ids[:num_reqs]
        virtual_context_active = self._virtual_context_enabled
        tiered_scheduler_budget = (
            self._tiered_scheduler_budget if virtual_context_active else {}
        )
        pressure_snapshot_request_ids = (
            set(
                tiered_scheduler_budget.get(
                    "pressure_snapshot_request_ids", set()
                )
            ).intersection(request_ids)
            if virtual_context_active
            else set()
        )
        proactive_snapshot_entries = {
            (request_id, group_id, logical_page)
            for request_id, group_id, logical_page in tiered_scheduler_budget.get(
                "proactive_snapshot_pages", set()
            )
            if request_id in request_ids
        }
        pressure_snapshot_request_ids.update(
            request_id
            for request_id, _group_id, _logical_page in proactive_snapshot_entries
        )

        def proactive_pages_for_group(
            group_id: int,
        ) -> dict[str, set[int]] | None:
            pages_by_request: defaultdict[str, set[int]] = defaultdict(set)
            for request_id, entry_group_id, logical_page in (
                proactive_snapshot_entries
            ):
                if entry_group_id == group_id:
                    pages_by_request[request_id].add(logical_page)
            return dict(pages_by_request) or None

        current_phase = (
            self._pd_green_context_execute_phase or ""
        ).strip().lower()
        virtual_context = (
            get_virtual_context_table() if virtual_context_active else None
        )
        tiered_reload_required_pages: dict[str, set[int]] | None = None
        tiered_disable_native_hbm_reload = False
        tiered_execution_plans = {
            request_id: plan
            for request_id, plan in tiered_scheduler_budget.get(
                "execution_plans", {}
            ).items()
            if request_id in request_ids
        }
        if (
            virtual_context is not None
            and virtual_context.has_hbm_reload_required_pages(
                set(request_ids)
            )
        ):
            reload_pages = virtual_context.plan_hbm_reload_pages(
                request_ids=set(request_ids),
            )
            if reload_pages:
                tiered_reload_required_pages = defaultdict(set)
                for page in reload_pages:
                    tiered_reload_required_pages[page.request_id].add(
                        page.logical_token_page
                    )
                tiered_reload_required_pages = dict(
                    tiered_reload_required_pages
                )
                ring_backend_ready = (
                    virtual_context_ring_backend_enabled()
                    and virtual_context_ring_backend_dram_provider_enabled()
                )
                reload_request_ids = set(tiered_reload_required_pages)
                reload_authority_modes = {
                    tiered_execution_plans[request_id][1]
                    for request_id in reload_request_ids
                    if request_id in tiered_execution_plans
                }
                explicit_native_reload = bool(reload_request_ids) and (
                    reload_authority_modes == {TIERED_EXECUTION_NATIVE_RELOAD}
                    and reload_request_ids.issubset(tiered_execution_plans)
                )
                explicit_context_ring = (
                    TIERED_EXECUTION_CONTEXT_RING in reload_authority_modes
                )
                prefill_requires_ring = current_phase == "prefill" and (
                    explicit_context_ring
                    or (
                        not explicit_native_reload
                        and virtual_context_prefill_ring_required_mode()
                        in {"always", "pressure"}
                        and not reload_request_ids.issubset(
                            tiered_scheduler_budget.get(
                                "native_hbm_reload_request_ids",
                                set(),
                            )
                        )
                    )
                )
                tiered_disable_native_hbm_reload = (
                    ring_backend_ready
                    and prefill_requires_ring
                )
        pressure_only_registration = (
            tiered_dram_register_pressure_only_enabled()
            if virtual_context_active
            else True
        )
        native_hbm_decode_control_bypass = (
            virtual_context_active
            and self._virtual_context_native_decode_worker_fast_path
            and current_phase == "decode"
            and not tiered_reload_required_pages
            and not tiered_disable_native_hbm_reload
        )
        if not virtual_context_active:
            pressure_snapshot_request_ids = set()
            tiered_control_required = False
        elif native_hbm_decode_control_bypass:
            # Decode hotset pages are already HBM-authoritative. A global
            # pressure snapshot is a prefill release signal, not a reason to
            # run provider/page planning in this decode forward.
            pressure_snapshot_request_ids = set()
            tiered_control_required = False
        else:
            tiered_control_required = (
                not pressure_only_registration
                or bool(tiered_reload_required_pages)
                or tiered_disable_native_hbm_reload
                or bool(pressure_snapshot_request_ids)
            )
        tiered_page_versions = (
            virtual_context.pin_attention_page_versions(request_ids, group_id=0)
            if virtual_context is not None and tiered_control_required
            else None
        )
        request_numeric_ids: list[int] | None = None
        request_numeric_ids_tensor: torch.Tensor | None = None
        request_provider_keys: list[int] | None = None
        request_provider_keys_tensor: torch.Tensor | None = None
        if tiered_control_required and tiered_attention_plan_tensors_enabled():
            request_numeric_ids = [
                int(self.input_batch.req_id_to_index[req_id])
                for req_id in request_ids
            ]
            request_numeric_ids_tensor = torch.tensor(
                request_numeric_ids,
                dtype=torch.int32,
                device=block_table_gid_0.device,
            )
            request_provider_keys = []
            for req_id in request_ids:
                request_key = get_dram_kv_request_key(
                    req_id,
                    create=False,
                )
                request_provider_keys.append(
                    -1 if request_key is None else int(request_key)
                )
            request_provider_keys_tensor = torch.tensor(
                request_provider_keys,
                dtype=torch.int64,
                device=block_table_gid_0.device,
            )
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],
            _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
            _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
                :num_reqs_padded
            ],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            request_ids=request_ids,
            request_numeric_ids=request_numeric_ids,
            request_numeric_ids_tensor=request_numeric_ids_tensor,
            request_provider_keys=request_provider_keys,
            request_provider_keys_tensor=request_provider_keys_tensor,
            attention_phase=current_phase,
            tiered_page_versions=tiered_page_versions,
            tiered_reload_required_pages=tiered_reload_required_pages,
            tiered_disable_native_hbm_reload=(
                tiered_disable_native_hbm_reload
            ),
            tiered_execution_plans=tiered_execution_plans or None,
            tiered_pressure_snapshot_request_ids=(
                pressure_snapshot_request_ids or None
            ),
            tiered_proactive_snapshot_pages=proactive_pages_for_group(0),
            tiered_dma_budget_bytes=int(
                tiered_scheduler_budget.get("dma_budget_bytes", 0)
            ),
            tiered_staging_budget_blocks=int(
                tiered_scheduler_budget.get("staging_budget_blocks", 0)
            ),
            tiered_max_prefill_ring_tokens=int(
                tiered_scheduler_budget.get("max_prefill_ring_tokens", 0)
            ),
            tiered_decode_tbt_budget_ms=float(
                tiered_scheduler_budget.get("decode_tbt_budget_ms", 0.0)
            ),
            tiered_block_table_cpu=(
                block_table_cpu_gid_0
                if (
                    tiered_control_required
                    and (
                        tiered_attention_plan_tensors_enabled()
                        or tiered_d2h_source_plan_enabled()
                    )
                )
                else None
            ),
            causal=True,
        )

        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.seq_lens.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # Cache attention metadata builds across hybrid KV-cache groups
        # The only thing that changes between different hybrid KV-cache groups when the
        # same metadata builder and KVCacheSpec is the same is the block table, so we
        # can cache the attention metadata builds and just update the block table using
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # shallow copy

            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )
            if kv_cache_gid > 0:
                (
                    cm.block_table_tensor,
                    cm.slot_mapping,
                    group_block_table_cpu,
                ) = _get_block_table_and_slot_mapping(kv_cache_gid)
                cm.tiered_block_table_cpu = (
                    group_block_table_cpu
                    if (
                        tiered_control_required
                        and (
                            tiered_attention_plan_tensors_enabled()
                            or tiered_d2h_source_plan_enabled()
                        )
                    )
                    else None
                )
                cm.tiered_proactive_snapshot_pages = (
                    proactive_pages_for_group(kv_cache_gid)
                )
            if (
                tiered_control_required
                and tiered_d2h_source_plan_enabled()
                and tiered_dram_register_from_hbm_enabled()
                and not use_spec_decode
                and current_phase not in {"decode", "mixed"}
                and cm.tiered_block_table_cpu is not None
            ):
                allowed_source_request_ids = set(
                    cm.tiered_pressure_snapshot_request_ids or ()
                )
                for request_id, pages in (
                    cm.tiered_reload_required_pages or {}
                ).items():
                    if pages:
                        allowed_source_request_ids.add(request_id)
                if not tiered_dram_register_pressure_only_enabled():
                    allowed_source_request_ids = set(request_ids)
                cm.tiered_d2h_source_plan = build_tiered_d2h_source_plan(
                    request_ids=request_ids,
                    block_table=cm.tiered_block_table_cpu,
                    seq_lens=self.seq_lens.cpu[:num_reqs],
                    query_start_loc=self.query_start_loc.cpu[: num_reqs + 1],
                    block_size=kv_cache_group.kv_cache_spec.block_size,
                    copy_policy=tiered_dram_register_from_hbm_policy(),
                    allowed_request_ids=allowed_source_request_ids,
                    force_all_request_ids=set(
                        cm.tiered_pressure_snapshot_request_ids or ()
                    ),
                    # Keep all candidates. The provider applies the per-forward
                    # copy cap after skipping pages already present in DRAM.
                    max_pages=0,
                    exact_force_pages_by_request=(
                        cm.tiered_proactive_snapshot_pages
                    ),
                )

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )

        return attn_metadata, spec_decode_common_attn_metadata

    def _complete_virtual_context_decode_handoff(
        self,
        attn_metadata: PerLayerAttnMetadata,
    ) -> dict[str, float]:
        """Reload every attention layer before native Decode model forward.

        Layer-local reload inside ``Attention.forward`` is too late for a
        request-level handoff: layer 0 can run before layer N discovers a
        missing DRAM page, leaving the whole model future failed halfway
        through. This barrier performs the same provider-backed H2D copies for
        all layers first and admits the model only after the aggregate virtual
        context table reports no reload-required page for the request.
        """

        metrics = {
            "worker_decode_handoff_barrier_requested": 0.0,
            "worker_decode_handoff_ring_flush_executor_count": 0.0,
            "worker_decode_handoff_ring_flush_waited_events": 0.0,
            "worker_decode_handoff_ring_flush_released_leases": 0.0,
            "worker_decode_handoff_ring_flush_pending_leases": 0.0,
            "worker_decode_handoff_ring_flush_duration_s": 0.0,
            "worker_decode_handoff_barrier_d2h_wait_count": 0.0,
            "worker_decode_handoff_barrier_reload_passes": 0.0,
            "worker_decode_handoff_barrier_reloaded_pages": 0.0,
            "worker_decode_handoff_barrier_remaining_pages": 0.0,
            "worker_decode_handoff_barrier_duration_s": 0.0,
        }
        if (
            not self._virtual_context_enabled
            or self._pd_green_context_execute_phase != "decode"
        ):
            return metrics
        virtual_context = get_virtual_context_table()
        if virtual_context is None:
            return metrics

        metadata_dicts = (
            attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
        )
        request_ids = {
            request_id
            for metadata_by_layer in metadata_dicts
            for metadata in metadata_by_layer.values()
            for request_id in (getattr(metadata, "request_ids", None) or ())
        }
        if not request_ids:
            return metrics
        pending = virtual_context.plan_hbm_reload_pages(
            request_ids=request_ids
        )
        if not pending:
            return metrics

        started = time.monotonic()
        metrics["worker_decode_handoff_barrier_requested"] = 1.0
        ring_flush_started = time.monotonic()
        ring_flush = flush_tiered_context_ring_source_lease_releases(
            sorted(request_ids)
        )
        metrics["worker_decode_handoff_ring_flush_duration_s"] = (
            time.monotonic() - ring_flush_started
        )
        for field in (
            "executor_count",
            "waited_events",
            "released_leases",
            "pending_leases",
        ):
            metrics[f"worker_decode_handoff_ring_flush_{field}"] = float(
                ring_flush[field]
            )
        if ring_flush["pending_leases"]:
            raise RuntimeError(
                "native Decode handoff could not close the final Context Ring "
                "DRAM source reads before H2D reload: "
                f"request_ids={sorted(request_ids)} "
                f"ring_flush={ring_flush}"
            )
        metrics["worker_decode_handoff_barrier_d2h_wait_count"] = float(
            wait_for_pending_hbm_d2h_request_reads(sorted(request_ids))
        )

        unsupported_layers: set[str] = set()
        layer_reload_diagnostics: dict[str, dict[str, int | str]] = {}
        max_passes = 3
        for reload_pass in range(max_passes):
            reloaded_this_pass = 0
            visited: set[tuple[str, int]] = set()
            for metadata_by_layer in metadata_dicts:
                for layer_name, metadata in metadata_by_layer.items():
                    visit_key = (layer_name, id(metadata))
                    if visit_key in visited:
                        continue
                    visited.add(visit_key)
                    layer_request_ids = list(
                        getattr(metadata, "request_ids", None) or ()
                    )
                    if not layer_request_ids:
                        continue
                    layer_id = virtual_context_layer_id(layer_name)
                    layer_pending = virtual_context.plan_hbm_reload_pages(
                        request_ids=set(layer_request_ids),
                        layer_id=layer_id,
                    )
                    if not layer_pending:
                        continue
                    reload_pages_by_request: defaultdict[str, set[int]] = (
                        defaultdict(set)
                    )
                    for page in layer_pending:
                        reload_pages_by_request[page.request_id].add(
                            page.logical_token_page
                        )

                    layer = self.compilation_config.static_forward_context.get(
                        layer_name
                    )
                    layer_kv_caches = getattr(layer, "kv_cache", None)
                    if not layer_kv_caches:
                        unsupported_layers.add(layer_name)
                        continue
                    kv_cache = layer_kv_caches[0]
                    if (
                        not isinstance(kv_cache, torch.Tensor)
                        or kv_cache.dim() != 5
                        or int(kv_cache.shape[1]) != 2
                    ):
                        unsupported_layers.add(layer_name)
                        continue
                    required_metadata = (
                        "block_table",
                        "seq_lens",
                        "query_start_loc",
                    )
                    if any(
                        not hasattr(metadata, field)
                        for field in required_metadata
                    ):
                        unsupported_layers.add(layer_name)
                        continue

                    key_cache, value_cache = kv_cache.unbind(1)
                    page_versions = (
                        virtual_context.pin_attention_page_versions(
                            layer_request_ids,
                            group_id=0,
                            layer_id=layer_id,
                        )
                    )
                    diagnostics: dict[str, int | str] = {}
                    reloaded = reload_dram_kv_pages_to_hbm(
                        layer_name=layer_name,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=metadata.block_table,
                        seq_lens=metadata.seq_lens,
                        query_start_loc=metadata.query_start_loc,
                        request_ids=layer_request_ids,
                        block_size=int(key_cache.shape[1]),
                        page_versions_by_request=page_versions or None,
                        reload_pages_by_request=dict(reload_pages_by_request),
                        block_table_cpu=getattr(
                            metadata, "tiered_block_table_cpu", None
                        ),
                        seq_lens_cpu=getattr(metadata, "seq_lens_cpu", None),
                        query_start_loc_cpu=getattr(
                            metadata, "query_start_loc_cpu", None
                        ),
                        block_table_is_fully_concrete=True,
                        diagnostics=diagnostics,
                    )
                    layer_reload_diagnostics[layer_name] = diagnostics
                    reloaded_this_pass += sum(
                        len(pages) for pages in reloaded.values()
                    )

            metrics["worker_decode_handoff_barrier_reload_passes"] = float(
                reload_pass + 1
            )
            metrics["worker_decode_handoff_barrier_reloaded_pages"] += float(
                reloaded_this_pass
            )
            pending = virtual_context.plan_hbm_reload_pages(
                request_ids=request_ids
            )
            if not pending:
                break
            if reloaded_this_pass == 0:
                break

        metrics["worker_decode_handoff_barrier_remaining_pages"] = float(
            len(pending)
        )
        metrics["worker_decode_handoff_barrier_duration_s"] = (
            time.monotonic() - started
        )
        if pending:
            pending_preview = [
                (
                    page.request_id,
                    page.layer_id,
                    page.logical_token_page,
                )
                for page in pending[:16]
            ]
            failure_diagnostics = {
                layer_name: diagnostics
                for layer_name, diagnostics in layer_reload_diagnostics.items()
                if diagnostics.get("failed_pages", 0)
                or diagnostics.get("reason") != "complete"
            }
            logger.error(
                "Native Decode handoff H2D reload made no complete request-level "
                "progress: request_ids=%s remaining_pages=%d ring_flush=%s "
                "layer_diagnostics=%s",
                sorted(request_ids),
                len(pending),
                ring_flush,
                failure_diagnostics,
            )
            raise RuntimeError(
                "native Decode handoff blocked before model forward because "
                "not every layer acknowledged its DRAM-to-HBM reload: "
                f"remaining_pages={len(pending)} "
                f"unsupported_layers={sorted(unsupported_layers)} "
                f"layer_diagnostics={failure_diagnostics} "
                f"preview={pending_preview}"
            )
        return metrics

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        :return: Optional[cascade_attn_prefix_lens]
            cascade_attn_prefix_lens is 2D: ``[kv_cache_group_id][attn_group_idx]``,
            None if we should not use cascade attention
        """

        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 0 if cascade attention should not be used
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's xdrope_positions are pre-computed
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's xdrope_positions on-the-fly
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                xdrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        if (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and num_logits <= self.cudagraph_batch_sizes[-1]
        ):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_logits_padded = self.vllm_config.pad_for_cudagraph(num_logits)
        else:
            num_logits_padded = num_logits
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[MultiModalKwargsItem],
        list[tuple[str, PlaceholderRange]],
    ]:
        """Batch multimodal inputs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_hashes, mm_kwargs, mm_lora_refs) where:
            - mm_hashes: List of multimodal hashes for each item
            - mm_kwargs: List of multimodal kwargs for each item
            - mm_lora_refs: List of (req_id, placeholder_range) for each item
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], [], []

        mm_hashes = list[str]()
        mm_kwargs = list[MultiModalKwargsItem]()
        # Multimodal LoRA reference info to map each multimodal item
        # back to its request & position
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue

                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append(mm_feature.data)
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        return mm_hashes, mm_kwargs, mm_lora_refs

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)

        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # Build LoRA mappings independently for encoder inputs
            # (encoder batch structure is different from main batch)
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []

            for req_id, pos_info in mm_lora_refs:
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                # Prefer pos_info.get_num_embeds to count precise MM embedding tokens.
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]
                    pos_info.get_num_embeds
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * num_tokens)
                encoder_token_counts.append(num_tokens)

                if lora_id > 0:
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # Set tower adapter mapping
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            if hasattr(self.model, "get_num_mm_connector_tokens"):
                post_op_counts = [
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]
                    for num_tokens in encoder_token_counts
                ]

                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(post_op_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            curr_group_outputs: MultiModalEmbeddings

            # EVS-related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            if (
                self.is_multimodal_pruning_enabled
                and modality == "video"
                and num_items > 1
            ):
                curr_group_outputs_lst = list[torch.Tensor]()
                for video_mm_kwargs_item in filter(
                    lambda item: item.modality == "video", mm_kwargs
                ):
                    _, _, micro_batch_mm_inputs = next(
                        group_mm_kwargs_by_modality(
                            [video_mm_kwargs_item],
                            device=self.device,
                            pin_memory=self.pin_memory,
                        )
                    )

                    micro_batch_outputs = model.embed_multimodal(
                        **micro_batch_mm_inputs
                    )

                    curr_group_outputs_lst.extend(micro_batch_outputs)

                curr_group_outputs = curr_group_outputs_lst
            else:
                # Run the encoder.
                # `curr_group_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.
                curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output
            logger.debug("Finish execute for mm hash %s", mm_hash)
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

        return encoder_outputs

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Swap to the other buffer to avoid race condition with previous
        # iteration's async copy that may still be reading from CPU.
        self.is_mm_embed_idx = 1 - self.is_mm_embed_idx
        is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = is_mm_embed_buf.cpu
        is_mm_embed[:total_num_scheduled_tokens] = False

        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                    True if is_embed is None else is_embed
                )
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        # get raw model out of the cudagraph wrapper.
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            return self.model.unwrap()
        return self.model

    def _get_layer_kernel_timing_hooks(self) -> _CudaModuleTimingHooks | None:
        if not (
            self.layer_kernel_timing_enabled
            or self._pd_decode_gpu_component_trace_jsonl
        ):
            return None
        if not torch.cuda.is_available():
            return None
        if self.layer_kernel_timing_hooks is None:
            self.layer_kernel_timing_hooks = _CudaModuleTimingHooks(self.get_model())
            logger.info_once(
                "CUDA module timing enabled; matched %d Llama-like "
                "projection/attention/MLP modules. Component traces force "
                "eager execution for diagnostic runs.",
                self.layer_kernel_timing_hooks.matched_modules,
            )
        return self.layer_kernel_timing_hooks

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if self.use_async_scheduling:
            return AsyncGPUPoolingModelRunnerOutput(
                model_runner_output=model_runner_output,
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
                async_output_copy_stream=self.async_output_copy_stream,
            )

        raw_pooler_output = json_map_leaves(
            lambda x: None if x is None else x.to("cpu", non_blocking=True),
            raw_pooler_output,
        )
        model_runner_output.pooler_output = [
            out if include else None
            for out, include in zip(raw_pooler_output, finished_mask)
        ]
        self._sync_device()

        return model_runner_output

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_mm_inputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.model.requires_raw_input_tokens:
            input_ids = self.input_ids.gpu[:num_tokens]
        else:
            input_ids = None

        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
        return input_ids, inputs_embeds

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # Update spec_token_ids with real draft tokens from pre step only when
        # output_token_ids is needed (penalties or bad_words are in use).
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        self._update_states_after_model_execute(sampler_output.sampled_token_ids)
        return sampler_output

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
        force_sync_output: bool = False,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        invalid_req_indices_set: set[int] = set()
        logprobs_lists = None
        effective_async_scheduling = (
            self.use_async_scheduling and not force_sync_output
        )
        if not effective_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if effective_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield False
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        event_ready = bool(
            self.prepare_inputs_event_recorded
            and self.prepare_inputs_event.query()
        )
        if event_ready:
            self.prepare_inputs_event_fast_path_count += 1
        else:
            self.prepare_inputs_event_wait_count += 1
            self.prepare_inputs_event.synchronize()
        try:
            yield not event_ready
        finally:
            self.prepare_inputs_event.record()
            self.prepare_inputs_event_recorded = True

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        # Diagnostic module hooks do not fire during CUDA graph replay.
        force_eager = force_eager or self.layer_kernel_timing_enabled or bool(
            self._pd_decode_gpu_component_trace_jsonl
        )
        if (
            self._pd_green_decode_cudagraph_enabled
            and self._pd_green_context_execute_phase != "decode"
        ):
            # Only uniform Decode is captured on GreenContext streams. Prefill
            # remains eager because chunk/ring shapes and staging residency are
            # dynamic and must not enter a Decode graph namespace.
            force_eager = True
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        has_lora = (
            len(self.input_batch.lora_id_to_lora_request) > 0
            if force_has_lora is None
            else force_has_lora
        )

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        dispatch_cudagraph = (
            lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                disable_full=disable_full,
            )
            if not force_eager
            else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
        )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            # Disable DP padding when running eager to avoid excessive padding when
            # running prefills. This lets us set cudagraph_mode="NONE" on the prefiller
            # in a P/D setup and still use CUDA graphs (enabled by this padding) on the
            # decoder.
            allow_dp_padding = (
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            )

            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    allow_dp_padding=allow_dp_padding,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    disable_full=synced_cudagraph_mode <= CUDAGraphMode.PIECEWISE.value,
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _register_layerwise_nvtx_hooks(self) -> None:
        """
        Register layerwise NVTX hooks if --enable-layerwise-nvtx-tracing is enabled
        to trace detailed information of each layer or module in the model.
        """

        if (
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing
            and not self.layerwise_nvtx_hooks_registered
        ):
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when CUDA graph is "
                    "turned off; you may observe part or all of the model "
                    "missing NVTX markers"
                )

            # In STOCK_TORCH_COMPILE mode, after registering hooks here,
            # the __call__ function of nn.module will be recompiled with
            # fullgraph=True. Since nvtx.range_push/pop are not traceable
            # by torch dynamo, we can't register hook functions here
            # because hook functions will also be traced by torch dynamo.
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when "
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "
                    "function hooks registration"
                )
            else:
                pyt_hooks = PytHooks()
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)
                self.layerwise_nvtx_hooks_registered = True

    @staticmethod
    def _parse_pd_phase_execution_key(
        execution_key: str | None,
    ) -> tuple[str | None, int]:
        if execution_key is None:
            return None, 0
        normalized = execution_key.strip().lower()
        phase, separator, slot_text = normalized.partition(":")
        if phase not in {"decode", "prefill"}:
            return None, 0
        if not separator:
            return phase, 0
        try:
            slot = int(slot_text)
        except ValueError:
            return None, 0
        return (phase, slot) if slot >= 0 else (None, 0)

    @staticmethod
    def _normalize_pd_green_context_phase(phase: str | None) -> str | None:
        normalized, _ = GPUModelRunner._parse_pd_phase_execution_key(phase)
        return normalized

    def _pd_green_decode_graph_namespace(
        self,
        execution_key: str | None,
        profile_id: str,
    ) -> str:
        phase, slot = self._parse_pd_phase_execution_key(execution_key)
        if phase != "decode":
            return ""
        stable_profile_id = (
            profile_id if green_context_dynamic_sm_enabled() else "static"
        )
        return f"pd-green-decode:{stable_profile_id}:slot{slot}"

    def _ttft_prefill_preemption_allowed(
        self,
        scheduler_output: "SchedulerOutput",
        *,
        phase: str | None,
        phase_context: PDPhaseModelRunnerContext | None,
        green_context_active: bool,
    ) -> bool:
        """Fail closed unless this call can retain an exact worker slot."""

        if not ttft_layerwise_prefill_preemption_requested():
            return False
        if phase != "prefill":
            return False
        # Ordinary Ring is SM-only: the completed layer keeps the same slot,
        # batch and residency. Native private-Ring still has mode "native".
        if not ttft_prefill_continuation_mode_allowed(
            scheduler_output.pd_prefill_mode,
            p1_ring_sm_handoff=getattr(self, "_isolation_p1_ring_sm_handoff", False),
        ):
            return False
        if (scheduler_output.pd_prefill_mode == "ring"
                and not scheduler_output.ttft_layerwise_prefill_preemption):
            return False
        unsupported: list[str] = []
        if phase_context is None or not self._pd_phase_local_model_runner_enabled:
            unsupported.append("phase-local model-runner slot is unavailable")
        if not green_context_active:
            unsupported.append("Prefill GreenContext is inactive")
        if (
            self._pd_prefill_workspace_enabled
            and phase_context is not None
            and phase_context.slot >= prefill_workspace_slot_count()
        ):
            unsupported.append(
                "fixed Prefill workspace does not cover the phase slot"
            )
        if self.vllm_config.parallel_config.pipeline_parallel_size != 1:
            unsupported.append("pipeline parallelism is enabled")
        if has_kv_transfer_group():
            unsupported.append("KV connector lifecycle is active")
        if has_ec_transfer():
            unsupported.append("EC transfer lifecycle is active")
        if self.use_aux_hidden_state_outputs:
            unsupported.append("auxiliary hidden-state output is enabled")
        if self._pd_decode_gpu_component_trace_jsonl:
            unsupported.append("per-layer Decode component tracing is enabled")
        if os.environ.get("VLLM_ITERATION_GPU_TIMING", "0") == "1":
            unsupported.append("synchronous iteration GPU timing is enabled")
        if unsupported:
            raise RuntimeError(
                "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION cannot retain a "
                "correct continuation: " + "; ".join(unsupported)
            )
        return True

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
        pd_green_context_phase: str | None = None,
        kv_retention_binding: KVRetentionExecutionBinding | None = None,
    ) -> (
        ModelRunnerOutput
        | AsyncModelRunnerOutput
        | IntermediateTensors
        | TTFTLayerwisePrefillBoundary
        | None
    ):
        execution_key = pd_green_context_phase
        phase = self._normalize_pd_green_context_phase(execution_key)
        profile_id = str(getattr(scheduler_output, "pd_sm_profile_id", "") or "")
        profile_epoch = int(getattr(scheduler_output, "pd_sm_profile_epoch", 0))
        graph_namespace = (
            self._pd_green_decode_graph_namespace(execution_key, profile_id)
            if self._pd_green_decode_cudagraph_enabled and phase == "decode"
            else ""
        )
        with self._use_pd_phase_model_runner_context(
            execution_key
        ) as phase_context:
            if (phase_context is not None and phase == "decode"
                    and phase_context.input_ids.gpu.shape[0] < self.max_num_tokens):
                if (scheduler_output.total_num_scheduled_tokens
                        > phase_context.input_ids.gpu.shape[0]
                        or any(n != 1 for n in
                               scheduler_output.num_scheduled_tokens.values())):
                    raise RuntimeError("compact Decode input requires uniform fitting tokens")
            if phase_context is not None:
                assert execution_key is not None
                self._claim_pd_execute_state(
                    execution_key, scheduler_output, phase_context,
                    kv_retention_binding,
                )
            elif kv_retention_binding is not None:
                raise RuntimeError("Prefill binding requires a phase-local context")
            previous_phase = self._pd_green_context_execute_phase
            previous_active = self._pd_green_context_execute_active
            previous_profile_id = self._pd_green_context_execute_profile_id
            previous_profile_epoch = self._pd_green_context_execute_profile_epoch
            previous_sample_phase = self._pd_green_context_sample_phase
            previous_sample_profile_id = self._pd_green_context_sample_profile_id
            previous_sample_profile_epoch = (
                self._pd_green_context_sample_profile_epoch
            )
            previous_tiered_budget = self._tiered_scheduler_budget
            self._pd_green_context_execute_phase = phase
            self._pd_green_context_execute_active = False
            self._pd_green_context_execute_profile_id = profile_id
            self._pd_green_context_execute_profile_epoch = profile_epoch
            self._tiered_scheduler_budget = (
                self._tiered_budget_from_scheduler_output(scheduler_output)
            )
            wait_for_completion = not green_context_execute_model_no_join_enabled()
            try:
                with green_context_phase_stream(
                    phase,
                    wait_for_completion=wait_for_completion,
                    profile_id=profile_id or None,
                    profile_epoch=profile_epoch,
                ) as active:
                    self._pd_green_context_execute_active = active
                    saved = getattr(phase_context, "ttft_prefill_continuation", None)
                    if saved is not None and saved.kv_sampling_restore_pending:
                        saved.kv_sampling_frame.restore(self)
                        saved.kv_sampling_restore_pending = False
                        saved.kv_sampling_frame = None
                        saved.model_runner_debug_timings["kv_retention_p1_sampling_restore_count"] = (
                            saved.model_runner_debug_timings.get("kv_retention_p1_sampling_restore_count", 0) + 1)
                    preemption_allowed = (
                        self._ttft_prefill_preemption_allowed(
                            scheduler_output,
                            phase=phase,
                            phase_context=phase_context,
                            green_context_active=active,
                        )
                    )
                    existing_ttft_execution = (
                        phase_context.ttft_prefill_continuation.execution
                        if phase_context is not None
                        and phase_context.ttft_prefill_continuation is not None
                        else None
                    )
                    with green_context_layerwise_prefill_handoff(
                        phase,
                        active=active and not kv_retention_p1_parked_requested(),
                        profile_id=profile_id,
                        profile_epoch=profile_epoch,
                    ) as layerwise_handoff:
                        with ttft_layerwise_prefill_execution(
                            phase,
                            scheduler_iteration_id=(
                                self._pd_trace_iteration_id(scheduler_output)
                            ),
                            phase_submit_seq=int(
                                getattr(
                                    scheduler_output,
                                    "pd_phase_submit_seq",
                                    -1,
                                )
                            ),
                            num_tokens=(
                                scheduler_output.total_num_scheduled_tokens
                            ),
                            num_requests=len(
                                scheduler_output.num_scheduled_tokens
                            ),
                            phase_slot=(
                                phase_context.slot
                                if phase_context is not None
                                else 0
                            ),
                            request_ids=tuple(
                                scheduler_output.num_scheduled_tokens
                            ),
                            scheduled_tokens=tuple(
                                (
                                    str(request_id),
                                    int(num_scheduled_tokens),
                                )
                                for request_id, num_scheduled_tokens in (
                                    scheduler_output.num_scheduled_tokens.items()
                                )
                            ),
                            profile_id=profile_id,
                            profile_epoch=profile_epoch,
                            prefill_mode=scheduler_output.pd_prefill_mode,
                            preemption_allowed=preemption_allowed,
                            existing_execution=existing_ttft_execution,
                        ) as ttft_layer_execution:
                            if (
                                self._pd_prefill_workspace_enabled
                                and phase == "prefill"
                                and active
                            ):
                                with prefill_workspace_execution(
                                    phase,
                                    active=active,
                                    profile_id=profile_id,
                                    profile_epoch=profile_epoch,
                                    num_tokens=(
                                        scheduler_output.total_num_scheduled_tokens
                                    ),
                                    phase_slot=(
                                        phase_context.slot
                                        if phase_context is not None
                                        else 0
                                    ),
                                    handoff_generation=(
                                        layerwise_handoff.generation
                                    ),
                                ):
                                    with cudagraph_namespace(graph_namespace):
                                        output = self._execute_model_impl(
                                            scheduler_output,
                                            intermediate_tensors,
                                        )
                            else:
                                with cudagraph_namespace(graph_namespace):
                                    output = self._execute_model_impl(
                                        scheduler_output,
                                        intermediate_tensors,
                                    )
                layerwise_debug = layerwise_handoff.as_debug_dict()
                if isinstance(output, TTFTLayerwisePrefillBoundary):
                    output.kv_retention_binding = kv_retention_binding
                    output.model_runner_debug_timings.update(layerwise_debug)
                    output.model_runner_debug_timings.update(
                        ttft_layer_execution.as_debug_dict()
                    )
                elif self.execute_model_state is not None:
                    self.execute_model_state.model_runner_debug_timings.update(
                        layerwise_debug
                    )
                    self.execute_model_state.model_runner_debug_timings.update(
                        ttft_layer_execution.as_debug_dict()
                    )
                    if (
                        self._pd_prefill_workspace_enabled
                        and phase == "prefill"
                    ):
                        self.execute_model_state.model_runner_debug_timings.update(
                            prefill_workspace_debug_snapshot()
                        )
                effective_profile_id = (
                    layerwise_handoff.effective_profile_id or profile_id
                )
                effective_profile_epoch = (
                    layerwise_handoff.effective_profile_epoch
                    if layerwise_handoff.effective_profile_id
                    else profile_epoch
                )
                if isinstance(output, TTFTLayerwisePrefillBoundary):
                    # No logits/sample state exists at an intermediate layer
                    # boundary.  The phase-local continuation owns the slot.
                    self._pd_green_context_sample_phase = None
                    self._pd_green_context_sample_profile_id = ""
                    self._pd_green_context_sample_profile_epoch = 0
                elif output is None and active:
                    self._pd_green_context_sample_phase = phase
                    # If the same Prefill future changed profile at a layer
                    # boundary, sampling/logits postprocessing must remain on
                    # the target stream.  EngineCore fences all other phase
                    # admissions until it commits the matching epoch.
                    self._pd_green_context_sample_profile_id = (
                        effective_profile_id
                    )
                    self._pd_green_context_sample_profile_epoch = (
                        effective_profile_epoch
                    )
                elif output is None:
                    self._pd_green_context_sample_phase = None
                    self._pd_green_context_sample_profile_id = ""
                    self._pd_green_context_sample_profile_epoch = 0
                elif output is not None:
                    self._pd_green_context_sample_phase = previous_sample_phase
                    self._pd_green_context_sample_profile_id = (
                        previous_sample_profile_id
                    )
                    self._pd_green_context_sample_profile_epoch = (
                        previous_sample_profile_epoch
                    )
                return output
            finally:
                self._pd_green_context_execute_phase = previous_phase
                self._pd_green_context_execute_active = previous_active
                self._pd_green_context_execute_profile_id = previous_profile_id
                self._pd_green_context_execute_profile_epoch = previous_profile_epoch
                self._tiered_scheduler_budget = previous_tiered_budget

    def _finish_execute_model_forward(
        self,
        *,
        scheduler_output: "SchedulerOutput",
        model_output: Any,
        model_runner_debug_timings: dict[str, Any],
        logits_indices: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
        spec_decode_common_attn_metadata: CommonAttentionMetadata | None,
        ec_connector_output: ECConnectorOutput | None,
        cudagraph_stats: CUDAGraphStat | None,
        kv_connector_output: KVConnectorOutput | None,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        num_tokens_padded: int,
        layer_kernel_timing_capture: "_CudaModuleTimingCapture | None",
        decode_component_trace_active: bool,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        """Finish logits preparation only after the final decoder layer.

        Intermediate TTFT layer boundaries never call this method; therefore
        they cannot create output placeholders, sample a token, or expose a
        partially computed model result to Scheduler.update_from_output().
        """

        postprocess_start_ts = time.monotonic()
        if getattr(self, "_isolation_memory_snapshots", False):
            from vllm.isolation_memory import isolation_allocator_snapshot
            model_runner_debug_timings.update(isolation_allocator_snapshot(
                torch.cuda, self.device,
                iteration=scheduler_output.scheduler_iteration_id,
                sample_every=self._isolation_memory_sample_every,
            ))
        logits_postprocess_end_event = None
        if layer_kernel_timing_capture is not None and decode_component_trace_active:
            logits_postprocess_start_event = torch.cuda.Event(enable_timing=True)
            logits_postprocess_end_event = torch.cuda.Event(enable_timing=True)
            layer_kernel_timing_capture.stage_events["logits_postprocess"] = (
                logits_postprocess_start_event,
                logits_postprocess_end_event,
            )
            logits_postprocess_start_event.record()
        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                if not get_pp_group().is_last_rank:
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data,
                    src=len(get_pp_group().ranks) - 1,
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]
        if logits_postprocess_end_event is not None:
            logits_postprocess_end_event.record()
        postprocess_end_ts = time.monotonic()
        model_runner_debug_timings["worker_postprocess_duration_s"] = (
            postprocess_end_ts - postprocess_start_ts
        )
        worker_execute_start_ts = float(
            model_runner_debug_timings.get(
                "worker_execute_start_ts", postprocess_start_ts
            )
        )
        model_runner_debug_timings["worker_execute_duration_s"] = (
            postprocess_end_ts - worker_execute_start_ts
        )

        if decode_component_trace_active:
            self._pd_decode_gpu_component_capture = layer_kernel_timing_capture
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            model_runner_debug_timings,
        )
        self.kv_connector_output = kv_connector_output
        return None

    @staticmethod
    def _synchronize_ttft_prefill_boundary() -> float:
        """Wait for exactly the submitted layer group, not the whole device."""

        wait_start_ts = time.monotonic()
        boundary_event = torch.cuda.Event()
        boundary_event.record(torch.cuda.current_stream())
        boundary_event.synchronize()
        return time.monotonic() - wait_start_ts

    def _ttft_kv_checkpoint_inputs(
        self,
        state: TTFTPrefillContinuationState,
    ) -> tuple[tuple[str, ...], tuple[KVRetentionCheckpointRow, ...]]:
        """Read the still-owned CPU input table, never optimistic progress.

        This first checkpoint contract supports full-attention Native batches
        only. Subdivided, windowed, parallel-context or shared-layer layouts
        need their own physical mapping contract before they can be captured.
        """

        groups = self.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise TTFTLayerwisePrefillError(
                "checkpoint requires exactly one full-attention cache group"
            )
        group = groups[0]
        spec = group.kv_cache_spec
        specs = (
            list(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs) else [spec]
        )
        table = self.input_batch.block_table[0]
        if (
            not specs
            or any(not isinstance(item, FullAttentionSpec)
                   or item.sliding_window is not None
                   or item.attention_chunk_size is not None
                   or item.block_size != table.block_size for item in specs)
            or table.use_hybrid_blocks
            or table.blocks_per_kv_block != 1
            or table.pcp_world_size != 1
            or table.dcp_world_size != 1
        ):
            raise TTFTLayerwisePrefillError(
                "checkpoint requires identical full-attention allocator/kernel "
                "blocks without sliding windows or context parallelism"
            )
        decoder = getattr(self.get_model(), "model", None)
        layers = getattr(decoder, "layers", ())
        layer_names = tuple(
            getattr(getattr(getattr(layer, "self_attn", None), "attn", None),
                    "layer_name", "")
            for layer in layers
        )
        if (
            getattr(decoder, "start_layer", None) != 0
            or getattr(decoder, "end_layer", None) != len(layer_names)
            or not layer_names
            or any(not isinstance(name, str) or not name for name in layer_names)
            or len(set(layer_names)) != len(layer_names)
            or len(group.layer_names) != len(layer_names)
            or set(group.layer_names) != set(layer_names)
        ):
            raise TTFTLayerwisePrefillError(
                "checkpoint requires exact full-model decoder attention names"
            )
        scheduled = state.scheduler_output.num_scheduled_tokens
        if set(scheduled) != set(state.boundary.request_ids):
            raise TTFTLayerwisePrefillError("checkpoint scheduled batch changed")
        cpu_blocks = table.get_numpy_array()
        rows = []
        for request_id in state.boundary.request_ids:
            index = self.input_batch.req_id_to_index.get(request_id)
            if (
                index is None
                or not 0 <= index < self.input_batch.num_reqs
                or self.input_batch.req_ids[index] != request_id
            ):
                raise TTFTLayerwisePrefillError(
                    "checkpoint lost its phase-local input row"
                )
            count = int(table.num_blocks_per_row[index])
            if not 0 < count <= cpu_blocks.shape[1]:
                raise TTFTLayerwisePrefillError("invalid checkpoint block count")
            blocks = tuple(int(block) for block in cpu_blocks[index, :count])
            if any(block >= self.kv_cache_config.num_blocks for block in blocks):
                raise TTFTLayerwisePrefillError("checkpoint block outside KV pool")
            rows.append(KVRetentionCheckpointRow(
                request_id=request_id,
                input_row=int(index),
                start_token=int(self.input_batch.num_computed_tokens_cpu[index]),
                scheduled_tokens=int(scheduled[request_id]),
                block_size=int(table.block_size),
                block_ids=blocks,
            ))
        return layer_names, tuple(rows)

    def _capture_ttft_kv_checkpoint(
        self, state: TTFTPrefillContinuationState
    ) -> None:
        # Call ONLY after _synchronize_ttft_prefill_boundary. Plan-only remains
        # observational; the separate offload flag adds a copy-complete receipt.
        if has_native_reload_authority(state.scheduler_output):
            if (state.scheduler_output.kv_retention_boundary_reason != "sm"
                    or state.scheduler_output.kv_retention_layerwise_offload_requested
                    or state.scheduler_output.kv_retention_layerwise_transfer is not None
                    or state.kv_offload is not None or state.kv_checkpoint is not None
                    or state.kv_resume_frame is not None
                    or state.boundary.kv_retention_parked_ticket is not None):
                raise TTFTLayerwisePrefillError(
                    "native_reload SM boundary cannot checkpoint, offload or park"
                )
            state.model_runner_debug_timings[
                "pd_layerwise_native_reload_sm_boundary"
            ] = 1.0
            return
        if state.scheduler_output.pd_prefill_mode == "ring":
            if (not getattr(self, "_isolation_p1_ring_sm_handoff", False)
                    or state.scheduler_output.kv_retention_boundary_reason != "sm"
                    or state.scheduler_output.kv_retention_layerwise_offload_requested
                    or state.scheduler_output.kv_retention_layerwise_transfer is not None
                    or state.kv_offload is not None or state.kv_checkpoint is not None
                    or state.kv_resume_frame is not None
                    or state.boundary.kv_retention_parked_ticket is not None):
                raise TTFTLayerwisePrefillError(
                    "ordinary Ring SM boundary cannot checkpoint, offload or park")
            state.model_runner_debug_timings["pd_layerwise_ring_sm_boundary"] = 1.0
            return
        if state.kv_offload is not None:
            if state.kv_offload.completed_layers != state.boundary.next_layer:
                raise TTFTLayerwisePrefillError("Ring layer coverage disagrees with yield boundary")
            state.kv_offload.ticket.advance(state.boundary)
            state.boundary.kv_retention_transfer = state.kv_offload.ticket
            state.model_runner_debug_timings.update(state.kv_offload.ticket.debug())
            return
        if state.kv_checkpoint is None and not (
            kv_retention_layerwise_checkpoint_requested()
        ):
            return
        names, rows = self._ttft_kv_checkpoint_inputs(state)
        checkpoint = build_kv_retention_layerwise_checkpoint(
            boundary=state.boundary,
            first_layer=state.execution.first_layer,
            layer_names=names,
            rows=rows,
            gpu_boundary_complete=True,
        )
        if state.kv_checkpoint is not None:
            state.kv_checkpoint.validate_successor(checkpoint)
        state.kv_checkpoint = checkpoint
        state.model_runner_debug_timings.update(checkpoint.as_debug_dict())
        if (kv_retention_layerwise_offload_enabled()
                and state.scheduler_output.kv_retention_layerwise_offload_requested):
            from vllm.v1.attention.ops.kv_retention_layerwise import KVRetentionRingContinuation
            state.kv_offload = KVRetentionRingContinuation.capture(
                checkpoint, self.compilation_config.static_forward_context
            )
            if state.kv_offload is not None:
                state.boundary.kv_retention_transfer = state.kv_offload.ticket
                state.model_runner_debug_timings.update(state.kv_offload.ticket.debug())
                state.model_runner_debug_timings["kv_retention_offload_d2h_wait_s"] = (
                    state.kv_offload.snapshot_wait_s
                )
            else:
                state.model_runner_debug_timings["kv_retention_offload_budget_declined"] = 1.0
            state.scheduler_output.kv_retention_layerwise_offload_requested = False

    def _validate_ttft_kv_checkpoint(
        self, state: TTFTPrefillContinuationState
    ) -> None:
        # An in-flight checkpoint cannot be disabled by changing an env var.
        if state.kv_checkpoint is None:
            return
        names, rows = self._ttft_kv_checkpoint_inputs(state)
        if state.kv_offload is not None:
            if names != state.kv_checkpoint.layer_names or rows != state.kv_checkpoint.rows:
                raise TTFTLayerwisePrefillError("offloaded input row layout changed")
            state.kv_offload.ticket.validate_boundary(state.boundary)
            state.kv_offload.ticket.require_ring()
            return
        state.kv_checkpoint.validate_resume(
            boundary=state.boundary, layer_names=names, rows=rows
        )

    def _capture_kv_retention_resume_frame(
        self, state: TTFTPrefillContinuationState,
    ) -> None:
        if has_native_reload_authority(state.scheduler_output):
            if state.scheduler_output.kv_retention_boundary_reason != "sm":
                raise TTFTLayerwisePrefillError("native_reload cannot park for a request")
            return  # Preserve the physical slot and its original reload metadata.
        if state.scheduler_output.pd_prefill_mode == "ring":
            # _capture_ttft_kv_checkpoint has checked the SM-only authority.
            # Ordinary Ring cannot clone/lend slot0 or use native block maps.
            if state.scheduler_output.kv_retention_boundary_reason != "sm":
                raise TTFTLayerwisePrefillError("ordinary Ring cannot park for a request")
            return
        continuation = state.continuation
        if not isinstance(continuation, KVRetentionLayerContinuation):
            return  # Flag-off generator is unchanged.
        store = getattr(self, "_kv_retention_parked_store", None)
        if store is not None and (state.scheduler_output.kv_retention_boundary_reason != "request"
                                  or store.entries):
            # SM-only yield retains the physical slot. No activation copy and
            # no nested request park while another batch is already parked.
            return
        context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
        owner = getattr(context, "kv_retention_execute_owner", None)
        binding = getattr(owner, "binding", None)
        if (context is None or context.ttft_prefill_continuation is not state
                or binding is None or state.kv_resume_frame is not None
                or continuation.state is None
                or state.kv_checkpoint is None
                or state.num_tokens_across_dp is not None
                or state.ubatch_slices_padded is not None
                or state.spec_decode_metadata is not None
                or state.spec_decode_common_attn_metadata is not None
                or state.ec_connector_output is not None
                or state.cudagraph_mode != CUDAGraphMode.NONE):
            raise RuntimeError("unsupported or unowned Prefill forward frame")
        binding.validate_output(owner.execution_key, state.scheduler_output)
        copy_wait_s = 0.0

        def complete_copy():
            nonlocal copy_wait_s
            copy_wait_s = self._synchronize_ttft_prefill_boundary()

        try:
            frame = KVRetentionResumeFrame.capture(
                binding=binding, model_state=continuation.state,
                attn_metadata=state.attn_metadata,
                logits_indices=state.logits_indices,
                num_scheduled_tokens_np=state.num_scheduled_tokens_np,
                max_gpu_bytes=min(kv_retention_resume_frame_gpu_bytes(), store.max_gpu_bytes)
                if store is not None else kv_retention_resume_frame_gpu_bytes(),
                gpu_boundary_complete=True, copy_complete=complete_copy)
        except KVRetentionFrameBudgetExceeded:
            # Retain the physical slot and input views. A declined capture
            # must never become a P1 slot-release ACK.
            state.model_runner_debug_timings[
                "kv_retention_resume_frame_budget_declined"] = 1.0
            if store is not None:
                state.scheduler_output.kv_retention_p1_park_declined = True
            return
        continuation.export(frame)
        state.kv_resume_frame = frame
        state.attn_metadata = frame.payload["attn_metadata"]
        state.logits_indices = frame.payload["logits_indices"]
        state.num_scheduled_tokens_np = frame.payload["num_scheduled_tokens_np"]
        state.model_runner_debug_timings.update(frame.debug())
        state.model_runner_debug_timings[
            "kv_retention_resume_frame_copy_wait_s"] = copy_wait_s
        if store is not None:
            state.kv_sampling_frame = KVRetentionSamplingFrame.capture(self, state)
            ticket = KVRetentionParkedTicket(
                binding, state.boundary.next_layer, frame.gpu_storage_bytes,
                tuple(state.scheduler_output.num_scheduled_tokens))
            state.boundary.kv_retention_binding = binding
            ticket.validate(state.boundary, state.scheduler_output)
            store.put(ticket, state)
            state.boundary.kv_retention_parked_ticket = ticket
            # Clear execution references only after the complete frame/store
            # exists. The Future's completion is the Engine-visible ACK.
            context.ttft_prefill_continuation = None
            context.kv_retention_execute_owner = None
            state.model_runner_debug_timings["kv_retention_p1_park_prepared"] = 1.0
            state.model_runner_debug_timings["kv_retention_p1_parked_gpu_bytes"] = ticket.gpu_bytes

    @torch.inference_mode()
    def offload_parked_prefill(self, expected_owner: KVRetentionPhaseOwner):
        """Ordered late-pressure D2H; it never borrows a live physical slot."""
        if getattr(self, "_kv_retention_p1_hbm_only", False):
            raise RuntimeError("HBM-only parked work cannot offload KV")
        expected_owner.validate_key("prefill")
        store = self._kv_retention_parked_store
        item = store.get(expected_owner.binding) if store is not None else None
        if item is None:
            raise RuntimeError("late-pressure offload lost parked owner")
        _, state = item
        if has_native_reload_authority(state.scheduler_output):
            raise TTFTLayerwisePrefillError("native_reload cannot use parked offload")
        if state.kv_offload is not None:
            return state.kv_offload.ticket
        context = self._get_pd_phase_context("prefill", 0)
        if context.ttft_prefill_continuation is not None or context.execute_model_state is not None:
            raise RuntimeError("late-pressure copy cannot replace active Prefill")
        from vllm.v1.attention.ops.kv_retention_layerwise import KVRetentionRingContinuation
        # The saved checkpoint already proves exact per-layer validity and
        # source GPU completion; this copy does not need slot input buffers.
        state.kv_offload = KVRetentionRingContinuation.capture(
            state.kv_checkpoint, self.compilation_config.static_forward_context)
        if state.kv_offload is None:
            return None
        state.boundary.kv_retention_transfer = state.kv_offload.ticket
        return state.kv_offload.ticket

    def _import_kv_retention_resume_frame(
        self, state: TTFTPrefillContinuationState,
    ) -> None:
        frame = state.kv_resume_frame
        if frame is None:
            if (state.scheduler_output.pd_prefill_mode == "ring"
                    or has_native_reload_authority(state.scheduler_output)):
                continuation = state.continuation
                if (not isinstance(continuation, KVRetentionLayerContinuation)
                        or continuation.state is None):
                    raise TTFTLayerwisePrefillError(
                        "retained Ring/reload SM continuation lost explicit model state"
                    )
                target_stream = torch.cuda.current_stream()
                kv_retention_record_retained_streams(
                    model_state=continuation.state, attn_metadata=state.attn_metadata,
                    logits_indices=state.logits_indices,
                    record_stream=lambda tensor: tensor.record_stream(target_stream))
            return
        context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
        owner = getattr(context, "kv_retention_execute_owner", None)
        binding = getattr(owner, "binding", None)
        if (context is None or context.ttft_prefill_continuation is not state
                or binding is None):
            raise RuntimeError("Prefill frame import lost its execution owner")
        binding.validate_output(owner.execution_key, state.scheduler_output)
        target_stream = torch.cuda.current_stream()
        state.continuation.import_frame(
            frame, binding, record_stream=lambda tensor: tensor.record_stream(target_stream))
        # State and runner now own the copied objects, not the consumed ticket.
        frame.close()
        state.kv_resume_frame = None

    @staticmethod
    def _accumulate_ttft_group_debug(
        state: TTFTPrefillContinuationState,
        *,
        forward_host_s: float,
        boundary_wait_s: float,
    ) -> None:
        state.forward_host_s_total += max(0.0, float(forward_host_s))
        state.boundary_wait_s_total += max(0.0, float(boundary_wait_s))
        state.model_runner_debug_timings.update(
            consume_operator_selective_pd_metrics()
        )
        state.model_runner_debug_timings.update(consume_opmux_metrics())
        state.model_runner_debug_timings.update(
            consume_tiered_context_ring_runtime_metrics()
        )
        state.model_runner_debug_timings.update(
            {
                "ttft_layerwise_prefill_forward_host_s_total": (
                    state.forward_host_s_total
                ),
                "ttft_layerwise_prefill_boundary_wait_s_total": (
                    state.boundary_wait_s_total
                ),
                "worker_forward_duration_s": state.forward_host_s_total,
                "worker_model_forward_duration_s": state.forward_host_s_total,
            }
        )

    def _resume_ttft_prefill_continuation(
        self,
        scheduler_output: "SchedulerOutput",
        state: TTFTPrefillContinuationState,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        """Advance one worker-owned Prefill generator by one layer group."""

        if state.scheduler_output is not scheduler_output:
            raise RuntimeError(
                "TTFT Prefill continuation was resumed with a different "
                "SchedulerOutput object."
            )
        active_context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
        if (
            active_context is None
            or active_context.ttft_prefill_continuation is not state
        ):
            raise RuntimeError(
                "TTFT Prefill continuation lost its phase-local slot owner."
            )
        active_execution = active_ttft_layerwise_prefill_execution()
        if active_execution is not state.execution:
            raise RuntimeError(
                "TTFT Prefill continuation execution descriptor mismatch."
            )

        if state.kv_offload is not None and state.kv_offload.ticket.state == "retained":
            state.kv_offload.close()
            state.kv_offload = None
        self._validate_ttft_kv_checkpoint(state)
        self._import_kv_retention_resume_frame(state)
        reset_operator_selective_pd_metrics()
        reset_opmux_metrics()
        reset_tiered_context_ring_runtime_metrics()
        forward_start_ts = time.monotonic()
        with (
            kv_retention_ring_scope(state.kv_offload),
            set_forward_context(
                state.attn_metadata,
                self.vllm_config,
                num_tokens=state.num_tokens_padded,
                num_tokens_across_dp=state.num_tokens_across_dp,
                cudagraph_runtime_mode=state.cudagraph_mode,
                batch_descriptor=state.batch_desc,
                ubatch_slices=state.ubatch_slices_padded,
            ),
            record_function_or_nullcontext(
                "gpu_model_runner: ttft_prefill_resume"
            ),
            self._pd_cuda_phase_trace(
                "ttft_prefill_resume",
                self._pd_green_context_execute_phase,
                scheduler_output,
                state.num_reqs,
                state.num_scheduled_tokens,
            ),
        ):
            try:
                try:
                    model_output = next(state.continuation)
                except StopIteration as completed:
                    model_output = completed.value
            except BaseException:
                if state.kv_offload is not None:
                    state.kv_offload.close()
                raise
        forward_host_s = time.monotonic() - forward_start_ts

        if isinstance(model_output, TTFTLayerwisePrefillBoundary):
            boundary_wait_s = self._synchronize_ttft_prefill_boundary()
            state.boundary = model_output
            self._capture_ttft_kv_checkpoint(state)
            self._capture_kv_retention_resume_frame(state)
            self._accumulate_ttft_group_debug(
                state,
                forward_host_s=forward_host_s,
                boundary_wait_s=boundary_wait_s,
            )
            model_output.model_runner_debug_timings.update(
                state.model_runner_debug_timings
            )
            return model_output

        if state.kv_offload is not None:
            try:
                state.kv_offload.publish()
                state.model_runner_debug_timings.update(state.kv_offload.ticket.debug())
                state.model_runner_debug_timings["kv_retention_offload_ring_layers"] = float(
                    state.kv_offload.completed_layers - state.kv_checkpoint.next_layer
                )
            finally:
                state.kv_offload.close()
        active_context.ttft_prefill_continuation = None
        self._accumulate_ttft_group_debug(
            state,
            forward_host_s=forward_host_s,
            boundary_wait_s=0.0,
        )
        return self._finish_execute_model_forward(
            scheduler_output=scheduler_output,
            model_output=model_output,
            model_runner_debug_timings=state.model_runner_debug_timings,
            logits_indices=state.logits_indices,
            spec_decode_metadata=state.spec_decode_metadata,
            spec_decode_common_attn_metadata=(
                state.spec_decode_common_attn_metadata
            ),
            ec_connector_output=state.ec_connector_output,
            cudagraph_stats=state.cudagraph_stats,
            kv_connector_output=None,
            num_scheduled_tokens=state.num_scheduled_tokens,
            num_scheduled_tokens_np=state.num_scheduled_tokens_np,
            num_tokens_padded=state.num_tokens_padded,
            layer_kernel_timing_capture=None,
            decode_component_trace_active=False,
        )

    def _execute_model_impl(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> (
        ModelRunnerOutput
        | AsyncModelRunnerOutput
        | IntermediateTensors
        | TTFTLayerwisePrefillBoundary
        | None
    ):
        active_context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
        if (
            active_context is not None
            and active_context.ttft_prefill_continuation is not None
        ):
            return self._resume_ttft_prefill_continuation(
                scheduler_output,
                active_context.ttft_prefill_continuation,
            )

        self._flush_pd_decode_gpu_component_trace_completed()
        decode_component_trace_active, component_trace_sequence = (
            self._select_pd_decode_gpu_component_capture()
        )
        worker_execute_start_ts = time.monotonic()
        model_runner_debug_timings: dict[str, Any] = {
            "worker_execute_start_ts": worker_execute_start_ts,
            "pd_green_context_execute_model_enabled": float(
                green_context_execute_model_enabled()
            ),
            "pd_green_context_execute_model_active": float(
                self._pd_green_context_execute_active
            ),
            "pd_green_context_execute_model_no_join": float(
                green_context_execute_model_no_join_enabled()
            ),
            "pd_green_context_execute_model_phase": (
                self._pd_green_context_execute_phase or ""
            ),
            "pd_green_context_execute_model_profile_id": (
                self._pd_green_context_execute_profile_id
            ),
            "pd_green_context_execute_model_profile_epoch": float(
                self._pd_green_context_execute_profile_epoch
            ),
            "pd_nvtx_phase_trace_enabled": float(
                self._pd_nvtx_phase_trace_enabled
            ),
            "pd_cuda_event_trace_enabled": float(
                bool(self._pd_cuda_event_trace_jsonl)
            ),
            "pd_cuda_event_trace_pending": float(
                len(self._pd_cuda_event_trace_pending)
            ),
            "pd_cuda_event_trace_dropped": float(
                self._pd_cuda_event_trace_dropped
            ),
            "pd_decode_gpu_component_trace_enabled": float(
                bool(self._pd_decode_gpu_component_trace_jsonl)
            ),
            "pd_decode_gpu_component_trace_active": float(
                decode_component_trace_active
            ),
            "pd_decode_gpu_component_trace_forced_eager": float(
                bool(self._pd_decode_gpu_component_trace_jsonl)
            ),
            "pd_decode_gpu_component_trace_pending": float(
                len(self._pd_decode_gpu_component_trace_pending)
            ),
            "pd_decode_gpu_component_trace_dropped": float(
                self._pd_decode_gpu_component_trace_dropped
            ),
            "pd_decode_gpu_component_trace_sequence": float(
                component_trace_sequence
            ),
            "pd_decode_gpu_component_trace_every_n": float(
                self._pd_decode_gpu_component_trace_every_n
            ),
            "pd_phase_local_model_runner_enabled": float(
                self._pd_phase_local_model_runner_enabled
            ),
            "pd_phase_local_model_runner_active": float(
                self._pd_green_context_execute_phase in {"decode", "prefill"}
                and self._pd_phase_local_model_runner_enabled
            ),
            "pd_phase_context_routing_active": float(
                _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get() is not None
            ),
            "pd_phase_context_routing_phase": (
                _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get().phase
                if _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get() is not None
                else ""
            ),
        }
        model_runner_debug_timings.update(green_context_state())
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        kv_store_barrier_req_ids = set(scheduler_output.preempted_req_ids or ())
        kv_store_barrier_req_ids.update(
            scheduler_output.residency_deferred_req_ids or ()
        )
        if kv_store_barrier_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(kv_store_barrier_req_ids)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        input_prep_start_ts = time.monotonic()
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep() as input_prep_event_waited,
        ):
            # Update persistent batch states.
            kv_cache_prepare_start_ts = time.monotonic()
            self._update_states(scheduler_output)
            kv_cache_prepare_end_ts = time.monotonic()
            model_runner_debug_timings[
                "worker_kv_cache_prepare_duration_s"
            ] = kv_cache_prepare_end_ts - kv_cache_prepare_start_ts
            model_runner_debug_timings.update(
                self._register_batch_external_tiered_snapshots(
                    scheduler_output
                )
            )

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
            stale_req_ids = [
                req_id
                for req_id in self._active_input_batch_req_ids()
                if req_id not in scheduled_req_ids
            ]
            if stale_req_ids:
                for req_id in stale_req_ids:
                    self.input_batch.remove_request(req_id)
                self.input_batch.condense()
                self.input_batch.refresh_metadata()

            req_ids = self._active_input_batch_req_ids()
            num_reqs = len(req_ids)
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )
            model_runner_debug_timings[
                "pd_output_owner_prefill_sample_unmask_count"
            ] = float(
                getattr(
                    self,
                    "_last_pd_output_owner_prefill_sample_unmask_count",
                    0,
                )
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            active_pd_context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
            active_slot = (
                active_pd_context.slot if active_pd_context is not None else 0
            )
            green_decode_graph_eligible = bool(
                self._pd_green_decode_cudagraph_enabled
                and self._pd_green_context_execute_phase == "decode"
                and self._pd_green_context_execute_active
            )
            green_decode_uniform_batch = self._is_uniform_decode(
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                uniform_decode_query_len=self.uniform_decode_query_len,
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
            )
            if green_decode_graph_eligible and not green_decode_uniform_batch:
                scheduled_shape = ",".join(
                    str(int(value)) for value in num_scheduled_tokens_np
                )
                raise RuntimeError(
                    "GreenContext Decode received a non-uniform live-token "
                    "batch before CUDA Graph dispatch: "
                    f"scheduled_tokens=[{scheduled_shape}] "
                    f"uniform_query_len={self.uniform_decode_query_len}."
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            graph_execution_key = (
                f"decode:{active_slot}"
                if self._pd_green_context_execute_phase == "decode"
                else None
            )
            green_decode_namespace = self._pd_green_decode_graph_namespace(
                graph_execution_key,
                self._pd_green_context_execute_profile_id,
            )
            green_decode_graph_entry_ready = bool(
                green_decode_graph_eligible
                and isinstance(self.model, CUDAGraphWrapper)
                and self.model.has_cudagraph_entry(
                    green_decode_namespace, batch_desc
                )
            )
            green_decode_graph_active = bool(
                green_decode_graph_eligible
                and cudagraph_mode != CUDAGraphMode.NONE
                and green_decode_namespace
                in self._pd_green_decode_cudagraph_namespaces
                and green_decode_graph_entry_ready
            )
            if (
                green_decode_graph_eligible
                and not green_decode_graph_active
                and self._pd_green_decode_cudagraph_strict
            ):
                raise RuntimeError(
                    "strict GreenContext Decode CUDA graph execution fell "
                    "back to eager or an uncaptured namespace: "
                    f"namespace={green_decode_namespace!r} "
                    f"runtime_mode={cudagraph_mode.name}"
                )
            model_runner_debug_timings.update(
                {
                    "pd_green_decode_cudagraph_enabled": float(
                        self._pd_green_decode_cudagraph_enabled
                    ),
                    "pd_green_decode_cudagraph_eligible": float(
                        green_decode_graph_eligible
                    ),
                    "pd_green_decode_uniform_batch": float(
                        green_decode_uniform_batch
                    ),
                    "pd_green_decode_cudagraph_active": float(
                        green_decode_graph_active
                    ),
                    "pd_green_decode_cudagraph_entry_ready": float(
                        green_decode_graph_entry_ready
                    ),
                    "pd_green_decode_cudagraph_fallback": float(
                        green_decode_graph_eligible
                        and not green_decode_graph_active
                    ),
                    "pd_green_decode_cudagraph_profile_id": (
                        self._pd_green_context_execute_profile_id or "static"
                    ),
                    "pd_green_decode_cudagraph_slot": float(active_slot),
                    "pd_green_decode_cudagraph_namespace": (
                        green_decode_namespace
                    ),
                    "pd_green_decode_cudagraph_runtime_mode": (
                        cudagraph_mode.name
                    ),
                    "pd_green_decode_cudagraph_capture_count": float(
                        self._pd_green_decode_cudagraph_capture_count
                    ),
                }
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            if num_tokens_padded > self.input_ids.gpu.shape[0]:
                raise RuntimeError("padded batch exceeds physical phase input capacity")
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            attention_metadata_start_ts = time.monotonic()
            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                )
            )
            attention_metadata_end_ts = time.monotonic()
            model_runner_debug_timings[
                "worker_attention_metadata_build_duration_s"
            ] = attention_metadata_end_ts - attention_metadata_start_ts
            model_runner_debug_timings.update(
                self._complete_virtual_context_decode_handoff(attn_metadata)
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )
        input_prep_end_ts = time.monotonic()
        model_runner_debug_timings["worker_input_prep_duration_s"] = (
            input_prep_end_ts - input_prep_start_ts
        )
        model_runner_debug_timings.update(
            {
                "worker_input_prep_event_waited": float(
                    input_prep_event_waited
                ),
                "worker_input_prep_event_fast_path_count": float(
                    self.prepare_inputs_event_fast_path_count
                ),
                "worker_input_prep_event_wait_count": float(
                    self.prepare_inputs_event_wait_count
                ),
            }
        )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        forward_start_ts = time.monotonic()
        layer_kernel_timing_hooks = self._get_layer_kernel_timing_hooks()
        layer_kernel_timing_capture: _CudaModuleTimingCapture | None = None
        capture_layer_components = bool(
            layer_kernel_timing_hooks is not None
            and (self.layer_kernel_timing_enabled or decode_component_trace_active)
        )
        component_trace_metadata = (
            self._pd_decode_gpu_component_metadata(
                scheduler_output, num_reqs, num_tokens_unpadded
            )
            if decode_component_trace_active
            else None
        )
        if component_trace_metadata is not None:
            component_trace_metadata.update(
                {
                    "trace_sequence": component_trace_sequence,
                    "trace_every_n": self._pd_decode_gpu_component_trace_every_n,
                    "trace_includes_layers": float(
                        self._pd_decode_gpu_component_trace_include_layers
                    ),
                    "matched_modules": int(
                        layer_kernel_timing_hooks.matched_modules
                        if layer_kernel_timing_hooks is not None
                        else 0
                    ),
                }
            )
        reset_operator_selective_pd_metrics()
        reset_opmux_metrics()
        reset_tiered_context_ring_runtime_metrics()
        gpu_forward_start_event = None
        gpu_forward_end_event = None
        if (
            os.environ.get("VLLM_ITERATION_GPU_TIMING", "0") == "1"
            and torch.cuda.is_available()
        ):
            gpu_forward_start_event = torch.cuda.Event(enable_timing=True)
            gpu_forward_end_event = torch.cuda.Event(enable_timing=True)
            gpu_forward_start_event.record()
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
            self._pd_cuda_phase_trace(
                "forward",
                self._pd_green_context_execute_phase,
                scheduler_output,
                num_reqs,
                num_tokens_unpadded,
            ),
        ):
            if not capture_layer_components:
                model_output = self._model_forward(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
            else:
                assert layer_kernel_timing_hooks is not None
                with layer_kernel_timing_hooks.capture_forward(
                    component_trace_metadata
                ) as capture:
                    layer_kernel_timing_capture = capture
                    model_output = self._model_forward(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )
        if isinstance(model_output, TTFTLayerwisePrefillModelYield):
            if layer_kernel_timing_capture is not None or capture_layer_components:
                raise RuntimeError(
                    "TTFT layerwise Prefill preemption cannot retain a "
                    "per-layer timing-hook capture across Engine yields."
                )
            if kv_connector_output is not None:
                raise RuntimeError(
                    "TTFT layerwise Prefill preemption cannot split a KV "
                    "connector lifecycle across Engine yields."
                )
            phase_context = _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get()
            execution = active_ttft_layerwise_prefill_execution()
            if phase_context is None or execution is None:
                raise RuntimeError(
                    "TTFT layerwise Prefill model yielded without its "
                    "phase-local execution owner."
                )
            if phase_context.ttft_prefill_continuation is not None:
                raise RuntimeError(
                    "TTFT layerwise Prefill slot already owns a continuation."
                )
            boundary_wait_s = self._synchronize_ttft_prefill_boundary()
            forward_host_s = time.monotonic() - forward_start_ts
            continuation_state = TTFTPrefillContinuationState(
                scheduler_output=scheduler_output,
                execution=execution,
                boundary=model_output.boundary,
                continuation=model_output.continuation,
                attn_metadata=attn_metadata,
                num_tokens_padded=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_mode=cudagraph_mode,
                batch_desc=batch_desc,
                ubatch_slices_padded=ubatch_slices_padded,
                logits_indices=logits_indices,
                spec_decode_metadata=spec_decode_metadata,
                spec_decode_common_attn_metadata=(
                    spec_decode_common_attn_metadata
                ),
                ec_connector_output=ec_connector_output,
                cudagraph_stats=cudagraph_stats,
                num_scheduled_tokens=num_scheduled_tokens,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                num_reqs=num_reqs,
                model_runner_debug_timings=model_runner_debug_timings,
            )
            # Publish the exact worker owner before any capture can fail;
            # no tensor/frame is sent to Engine or granted slot-release rights.
            phase_context.ttft_prefill_continuation = continuation_state
            self._capture_ttft_kv_checkpoint(continuation_state)
            self._capture_kv_retention_resume_frame(continuation_state)
            self._accumulate_ttft_group_debug(
                continuation_state,
                forward_host_s=forward_host_s,
                boundary_wait_s=boundary_wait_s,
            )
            model_output.boundary.model_runner_debug_timings.update(
                model_runner_debug_timings
            )
            return model_output.boundary
        if gpu_forward_start_event is not None and gpu_forward_end_event is not None:
            gpu_forward_end_event.record()
            gpu_forward_end_event.synchronize()
            model_runner_debug_timings["gpu_forward_elapsed_ms"] = (
                gpu_forward_start_event.elapsed_time(gpu_forward_end_event)
            )
        if layer_kernel_timing_capture is not None:
            layer_kernel_timing_capture.metadata["host_enqueue_end_ts"] = (
                time.monotonic()
            )
        if self.layer_kernel_timing_enabled and layer_kernel_timing_hooks is not None:
            model_runner_debug_timings.update(
                layer_kernel_timing_hooks.summarize_capture(
                    layer_kernel_timing_capture
                )
            )
        if decode_component_trace_active:
            model_runner_debug_timings[
                "pd_decode_gpu_component_trace_deferred_to_sample"
            ] = float(layer_kernel_timing_capture is not None)
        model_runner_debug_timings.update(consume_operator_selective_pd_metrics())
        model_runner_debug_timings.update(consume_opmux_metrics())
        model_runner_debug_timings.update(
            consume_tiered_context_ring_runtime_metrics()
        )
        forward_end_ts = time.monotonic()
        model_runner_debug_timings["worker_forward_duration_s"] = (
            forward_end_ts - forward_start_ts
        )
        model_runner_debug_timings["worker_model_forward_duration_s"] = (
            forward_end_ts - forward_start_ts
        )

        return self._finish_execute_model_forward(
            scheduler_output=scheduler_output,
            model_output=model_output,
            model_runner_debug_timings=model_runner_debug_timings,
            logits_indices=logits_indices,
            spec_decode_metadata=spec_decode_metadata,
            spec_decode_common_attn_metadata=(
                spec_decode_common_attn_metadata
            ),
            ec_connector_output=ec_connector_output,
            cudagraph_stats=cudagraph_stats,
            kv_connector_output=kv_connector_output,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            num_tokens_padded=num_tokens_padded,
            layer_kernel_timing_capture=layer_kernel_timing_capture,
            decode_component_trace_active=decode_component_trace_active,
        )

    @torch.inference_mode
    def sample_tokens(
        self,
        grammar_output: "GrammarOutput | None",
        pd_green_context_phase: str | None = None,
        kv_retention_binding: KVRetentionExecutionBinding | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        explicit_execution_key = pd_green_context_phase
        explicit_phase = self._normalize_pd_green_context_phase(
            explicit_execution_key
        )
        phase = (
            explicit_phase
            if explicit_phase is not None
            else self._pd_green_context_sample_phase
        )
        execution_key = (
            explicit_execution_key if explicit_phase is not None else phase
        )
        with self._use_pd_phase_model_runner_context(execution_key) as phase_context:
            self._validate_kv_retention_sampling(
                execution_key, phase_context, kv_retention_binding)
            self._pd_green_context_sample_phase = None
            profile_id = self._pd_green_context_sample_profile_id
            profile_epoch = self._pd_green_context_sample_profile_epoch
            self._pd_green_context_sample_profile_id = ""
            self._pd_green_context_sample_profile_epoch = 0
            previous_phase = self._pd_green_context_current_sample_phase
            previous_active = self._pd_green_context_current_sample_active
            previous_profile_id = (
                self._pd_green_context_current_sample_profile_id
            )
            previous_profile_epoch = (
                self._pd_green_context_current_sample_profile_epoch
            )
            self._pd_green_context_current_sample_phase = phase
            self._pd_green_context_current_sample_active = False
            self._pd_green_context_current_sample_profile_id = profile_id
            self._pd_green_context_current_sample_profile_epoch = profile_epoch
            wait_for_completion = not green_context_execute_model_no_join_enabled()
            reuse_ordered_phase_stream = (
                not wait_for_completion
                and self._pd_phase_local_model_runner_enabled
                and explicit_phase in {"decode", "prefill"}
            )
            try:
                with green_context_phase_stream(
                    phase,
                    wait_for_completion=wait_for_completion,
                    # execute_model and sample_tokens for one execution key
                    # are dependency-ordered by UniProcExecutor and run on
                    # the same phase/slot stream. Waiting on the default
                    # stream again here can accidentally import unrelated
                    # work from the other phase into Decode's critical path.
                    wait_for_current_stream=not reuse_ordered_phase_stream,
                    profile_id=profile_id or None,
                    profile_epoch=profile_epoch,
                ) as active:
                    self._pd_green_context_current_sample_active = active
                    return self._sample_tokens_impl(grammar_output)
            finally:
                self._pd_green_context_current_sample_phase = previous_phase
                self._pd_green_context_current_sample_active = previous_active
                self._pd_green_context_current_sample_profile_id = (
                    previous_profile_id
                )
                self._pd_green_context_current_sample_profile_epoch = (
                    previous_profile_epoch
                )

    def _sample_tokens_impl(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            model_runner_debug_timings,
        ) = self.execute_model_state
        decode_component_capture = self._pd_decode_gpu_component_capture
        self._pd_decode_gpu_component_capture = None
        model_runner_debug_timings.update(
            {
                "pd_green_context_sample_phase": (
                    self._pd_green_context_current_sample_phase or ""
                ),
                "pd_green_context_sample_active": float(
                    self._pd_green_context_current_sample_active
                ),
                "pd_phase_local_model_runner_enabled": float(
                    self._pd_phase_local_model_runner_enabled
                ),
                "pd_phase_local_model_runner_sample_active": float(
                    self._pd_green_context_current_sample_phase
                    in {"decode", "prefill"}
                    and self._pd_phase_local_model_runner_enabled
                ),
                "pd_phase_context_routing_sample_active": float(
                    _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get() is not None
                ),
                "pd_phase_context_routing_sample_phase": (
                    _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get().phase
                    if _PD_ACTIVE_MODEL_RUNNER_CONTEXT.get() is not None
                    else ""
                ),
            }
        )
        phase_handoff_sync_sample_output = (
            self._pd_phase_local_model_runner_enabled
            and self._pd_green_context_current_sample_phase == "prefill"
            and bool(scheduler_output.pd_ttft_handoff_request_ids)
        )
        force_sync_sample_output = phase_handoff_sync_sample_output or (
            self._pd_phase_sync_sample_output_enabled
            and self._pd_green_context_current_sample_phase in {"decode", "prefill"}
        )
        model_runner_debug_timings.update(
            {
                "pd_phase_sync_sample_output_enabled": float(
                    self._pd_phase_sync_sample_output_enabled
                ),
                "pd_phase_sync_sample_output_active": float(
                    force_sync_sample_output
                ),
                "pd_phase_handoff_sync_sample_output_active": float(
                    phase_handoff_sync_sample_output
                ),
            }
        )
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        sample_start_ts = time.monotonic()
        sampling_end_event = None
        if decode_component_capture is not None:
            sampling_start_event = torch.cuda.Event(enable_timing=True)
            sampling_end_event = torch.cuda.Event(enable_timing=True)
            decode_component_capture.stage_events["sampling"] = (
                sampling_start_event,
                sampling_end_event,
            )
            decode_component_capture.metadata["host_sample_enqueue_start_ts"] = (
                sample_start_ts
            )
            sampling_start_event.record()
        with (
            record_function_or_nullcontext("gpu_model_runner: sample"),
            self._pd_cuda_phase_trace(
                "sample",
                self._pd_green_context_current_sample_phase,
                scheduler_output,
                self.input_batch.num_reqs,
                scheduler_output.total_num_scheduled_tokens,
            ),
        ):
            sampler_output = self._sample(logits, spec_decode_metadata)
            if (
                self._pd_phase_sampler_debug_sync_enabled
                and self._pd_green_context_current_sample_phase
                in {"decode", "prefill"}
            ):
                torch.cuda.synchronize()
        if sampling_end_event is not None:
            sampling_end_event.record()
            decode_component_capture.pipeline_end_event = torch.cuda.Event(
                enable_timing=True
            )
            decode_component_capture.pipeline_end_event.record()
            decode_component_capture.metadata["host_sample_enqueue_end_ts"] = (
                time.monotonic()
            )
            self._enqueue_pd_decode_gpu_component_capture(
                decode_component_capture
            )
        sample_end_ts = time.monotonic()
        model_runner_debug_timings["worker_sample_duration_s"] = (
            sample_end_ts - sample_start_ts
        )
        model_runner_debug_timings["worker_sampling_duration_s"] = (
            sample_end_ts - sample_start_ts
        )
        model_runner_debug_timings["pd_phase_sampler_debug_sync"] = float(
            self._pd_phase_sampler_debug_sync_enabled
            and self._pd_green_context_current_sample_phase in {"decode", "prefill"}
        )
        model_runner_debug_timings[
            "pd_decode_gpu_component_trace_pending"
        ] = float(len(self._pd_decode_gpu_component_trace_pending))
        model_runner_debug_timings[
            "pd_decode_gpu_component_trace_dropped"
        ] = float(self._pd_decode_gpu_component_trace_dropped)

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            if spec_config.use_eagle() and not spec_config.disable_padded_drafter_batch:
                # EAGLE speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(self.drafter, EagleProposer)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            spec_decode_common_attn_metadata,
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(
                        1, device=self.device, dtype=torch.int32
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        bookkeep_start_ts = time.monotonic()
        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
                force_sync_output=force_sync_sample_output,
            )
        bookkeep_end_ts = time.monotonic()
        model_runner_debug_timings["worker_bookkeep_duration_s"] = (
            bookkeep_end_ts - bookkeep_start_ts
        )

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        eplb_start_ts = time.monotonic()
        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()
        eplb_end_ts = time.monotonic()
        model_runner_debug_timings["worker_eplb_duration_s"] = (
            eplb_end_ts - eplb_start_ts
        )

        output_build_start_ts = time.monotonic()
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.model_config.enable_return_routed_experts:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                model_runner_debug_timings=model_runner_debug_timings,
            )
        output_build_end_ts = time.monotonic()
        model_runner_debug_timings["worker_output_build_duration_s"] = (
            output_build_end_ts - output_build_start_ts
        )
        model_runner_debug_timings["worker_output_transfer_duration_s"] = (
            output_build_end_ts - output_build_start_ts
        )
        model_runner_debug_timings["worker_sample_tokens_duration_s"] = (
            output_build_end_ts - sample_start_ts
        )

        if force_sync_sample_output:
            model_runner_debug_timings["pd_phase_sync_sample_output_returned"] = 1.0
            self.input_batch.sampled_token_ids_cpu = None
            self.input_batch.async_copy_ready_event = None
            self.input_batch.prev_req_id_to_index = None
            return output

        if not self.use_async_scheduling:
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        if spec_config.method == "ngram":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(self.input_batch, sampled_token_ids)
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif spec_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)

            if spec_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    # When padding the batch, token_indices is just a range
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = self._get_positions(total_num_tokens)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            if self.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )
        global_expert_loads, old_global_expert_indices_per_model, rank_mapping = (
            EplbState.get_eep_state(self.parallel_config)
            if eep_scale_up
            else (None, None, None)
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0

        try:
            with DeviceMemoryProfiler() as m:
                time_before_load = time.perf_counter()
                model_loader = get_model_loader(self.load_config)
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    self.drafter.load_model(self.model)
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )

                        global_expert_load = (
                            global_expert_loads[eplb_models]
                            if global_expert_loads
                            else None
                        )
                        old_global_expert_indices = (
                            old_global_expert_indices_per_model[eplb_models]
                            if old_global_expert_indices_per_model
                            else None
                        )
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                            global_expert_load,
                            old_global_expert_indices,
                            rank_mapping,
                        )
                        eplb_models += 1

                if self.use_aux_hidden_state_outputs:
                    if not supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )

                    # Try to get auxiliary layers from speculative config,
                    # otherwise use model's default layers
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = self.model.get_eagle3_aux_hidden_state_layers()

                    self.model.set_aux_hidden_state_layers(aux_layers)
                time_after_load = time.perf_counter()
            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
            scope="local",
        )
        prepare_communication_buffer_for_model(self.model)
        if (drafter := getattr(self, "drafter", None)) and (
            drafter_model := getattr(drafter, "model", None)
        ):
            prepare_communication_buffer_for_model(drafter_model)
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )

        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            global_expert_load = (
                global_expert_loads[eplb_models] if global_expert_loads else None
            )
            old_global_expert_indices = (
                old_global_expert_indices_per_model[eplb_models]
                if old_global_expert_indices_per_model
                else None
            )
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop(rank_mapping=rank_mapping)

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
            and supports_dynamo()
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWraper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, (
            "Cannot reload weights before model is loaded."
        )
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model_loader.load_weights(self.get_model(), model_config=self.model_config)

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_model_len,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data

        # Result in the maximum GPU consumption of the model
        dummy_mm_item = dummy_mm_data[modality][0]
        dummy_mm_items = [dummy_mm_item] * max_items_per_batch

        return next(
            mm_kwargs_group
            for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                dummy_mm_items,
                device=self.device,
                pin_memory=self.pin_memory,
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            activate_lora: If False, dummy_run is performed without LoRAs.
        """
        if supports_mm_encoder_only(self.model):
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=activate_lora,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if num_tokens_padded > self.input_ids.gpu.shape[0]:
                raise RuntimeError("CUDA Graph capture exceeds phase input capacity")
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        if supports_mm_encoder_only(self.model):
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])

        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(task=task, model_config=self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        if supports_mm_encoder_only(self.model):
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])

        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    # NOTE: Currently model is profiled with a single non-text
                    # modality with the max possible input tokens even when
                    # it supports multiple.
                    dummy_modality = mm_budget.get_modality_with_max_tokens()
                    max_mm_items_per_batch = mm_budget.max_items_per_batch_by_modality[
                        dummy_modality
                    ]

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",
                        encoder_budget,
                        max_mm_items_per_batch,
                        dummy_modality,
                    )

                    # Create dummy batch of multimodal inputs.
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,
                        max_mm_items_per_batch,
                    )

                    # Run multimodal encoder.
                    dummy_encoder_outputs = self.model.embed_multimodal(
                        **batched_dummy_mm_inputs
                    )

                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,
                        expected_num_items=max_mm_items_per_batch,
                    )
                    for i, output in enumerate(dummy_encoder_outputs):
                        self.encoder_cache[f"tmp_{i}"] = output

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def _capture_pd_green_decode_cudagraphs(
        self,
        *,
        lora_cases: list[bool],
        cudagraph_mode: CUDAGraphMode,
    ) -> None:
        from vllm.isolation_memory import isolation_memory_enabled

        memory_audit = isolation_memory_enabled("GRAPH_MEMORY_AUDIT")
        if not green_context_execute_model_enabled():
            raise RuntimeError(
                "VLLM_PD_GREEN_DECODE_CUDAGRAPH requires execute-model "
                "GreenContext routing"
            )
        if not self._pd_phase_local_model_runner_enabled:
            raise RuntimeError(
                "VLLM_PD_GREEN_DECODE_CUDAGRAPH requires "
                "VLLM_PD_PHASE_LOCAL_MODEL_RUNNER=1"
            )

        decode_mode = cudagraph_mode.decode_mode()
        if decode_mode != CUDAGraphMode.FULL:
            raise RuntimeError(
                "GreenContext Decode CUDA graph requires FULL Decode graph "
                f"support, got {decode_mode.name}"
            )
        if not isinstance(self.model, CUDAGraphWrapper):
            raise RuntimeError(
                "GreenContext Decode CUDA graph currently requires the "
                "non-ubatched full-graph model wrapper"
            )

        profile_ids = [""]
        if green_context_dynamic_sm_enabled():
            profile_state = initialize_dynamic_green_context_profiles()
            if profile_state.get("enabled") != 1:
                raise RuntimeError(
                    "dynamic GreenContext profile bank was not ready for "
                    f"Decode graph capture: {profile_state!r}"
                )
            profile_ids = [
                value
                for value in str(profile_state.get("profile_ids", "")).split(",")
                if value
            ]
            # A d0_p40 profile has no Decode GreenContext stream and can never
            # execute a Decode batch. Capturing it previously made an
            # otherwise valid profile bank fail strict startup.  An optional
            # allow-list also lets long-running experiments bound graph pool
            # memory to their configured allocator frontier.
            profile_filter = {
                value.strip()
                for value in os.environ.get(
                    "VLLM_PD_GREEN_DECODE_CUDAGRAPH_PROFILE_IDS", ""
                ).split(",")
                if value.strip()
            }

            def has_decode_sms(profile_id: str) -> bool:
                decode_text, separator, _ = profile_id.partition("_p")
                if not separator or not decode_text.startswith("d"):
                    return True
                try:
                    return int(decode_text[1:]) > 0
                except ValueError:
                    return True

            available_profile_ids = set(profile_ids)
            if profile_filter:
                unknown_profiles = profile_filter - available_profile_ids
                if unknown_profiles:
                    raise RuntimeError(
                        "GreenContext Decode CUDA graph profile filter contains "
                        f"unknown profiles: {sorted(unknown_profiles)}"
                    )
            profile_ids = [
                profile_id
                for profile_id in profile_ids
                if has_decode_sms(profile_id)
                and (not profile_filter or profile_id in profile_filter)
            ]
            if not profile_ids:
                raise RuntimeError(
                    "dynamic GreenContext Decode graph capture found no profiles"
                )

        max_num_tokens = (
            self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
        )
        decode_sizes = [
            value
            for value in self.cudagraph_batch_sizes
            if max_num_tokens >= value >= self.uniform_decode_query_len
        ]
        compilation_cases = list(product(reversed(decode_sizes), lora_cases))
        if not compilation_cases:
            raise RuntimeError("no uniform Decode CUDA graph capture sizes")

        for profile_id in profile_ids:
            for slot in range(self._pd_green_decode_cudagraph_slot_count):
                execution_key = f"decode:{slot}"
                namespace = self._pd_green_decode_graph_namespace(
                    execution_key, profile_id
                )
                if memory_audit:
                    allocated_before = torch.cuda.memory_allocated(self.device)
                    reserved_before = torch.cuda.memory_reserved(self.device)
                with self._use_pd_phase_model_runner_context(execution_key):
                    previous_phase = self._pd_green_context_execute_phase
                    previous_profile_id = self._pd_green_context_execute_profile_id
                    previous_profile_epoch = (
                        self._pd_green_context_execute_profile_epoch
                    )
                    self._pd_green_context_execute_phase = "decode"
                    self._pd_green_context_execute_profile_id = profile_id
                    self._pd_green_context_execute_profile_epoch = 0
                    try:
                        with (
                            green_context_phase_stream(
                                "decode",
                                wait_for_completion=True,
                                profile_id=profile_id or None,
                                profile_epoch=0,
                            ) as active,
                            cudagraph_namespace(namespace),
                        ):
                            if not active:
                                raise RuntimeError(
                                    "Decode GreenContext stream was unavailable "
                                    f"during graph capture: profile={profile_id!r}"
                                )
                            self._capture_cudagraphs(
                                compilation_cases=compilation_cases,
                                cudagraph_runtime_mode=decode_mode,
                                uniform_decode=True,
                            )
                    finally:
                        self._pd_green_context_execute_phase = previous_phase
                        self._pd_green_context_execute_profile_id = (
                            previous_profile_id
                        )
                        self._pd_green_context_execute_profile_epoch = (
                            previous_profile_epoch
                        )
                self._pd_green_decode_cudagraph_namespaces.add(namespace)
                if memory_audit:
                    logger.info(
                        "Isolation graph capture scope: namespace=%s "
                        "allocated_delta_bytes=%d reserved_delta_bytes=%d "
                        "pool_policy=independent_profile_slot",
                        namespace,
                        torch.cuda.memory_allocated(self.device) - allocated_before,
                        torch.cuda.memory_reserved(self.device) - reserved_before,
                    )
                self._pd_green_decode_cudagraph_capture_count += len(
                    compilation_cases
                )
                logger.info(
                    "Captured GreenContext Decode CUDA graphs: namespace=%s "
                    "profile=%s slot=%d cases=%d mode=%s",
                    namespace,
                    profile_id or "static",
                    slot,
                    len(compilation_cases),
                    decode_mode.name,
                )

    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None

            if self.lora_config:
                if self.compilation_config.cudagraph_specialize_lora:
                    lora_cases = [True, False]
                else:
                    lora_cases = [True]
            else:
                lora_cases = [False]

            if self._pd_green_decode_cudagraph_enabled:
                try:
                    self._capture_pd_green_decode_cudagraphs(
                        lora_cases=lora_cases,
                        cudagraph_mode=cudagraph_mode,
                    )
                except Exception:
                    if self._pd_green_decode_cudagraph_strict:
                        raise
                    logger.exception(
                        "GreenContext Decode CUDA graph capture failed; "
                        "disabling the experimental path and capturing "
                        "upstream graphs"
                    )
                    self._pd_green_decode_cudagraph_enabled = False

            if (
                not self._pd_green_decode_cudagraph_enabled
                and cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE
            ):
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()
                # make sure we capture the largest batch size first
                compilation_cases = list(
                    product(reversed(self.cudagraph_batch_sizes), lora_cases)
                )
                self._capture_cudagraphs(
                    compilation_cases,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=False,
                )

            # Capture full cudagraph for uniform decode batches if we
            # don't already have full mixed prefill-decode cudagraphs.
            if (
                not self._pd_green_decode_cudagraph_enabled
                and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and cudagraph_mode.separate_routine()
            ):
                max_num_tokens = (
                    self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
                )
                decode_cudagraph_batch_sizes = [
                    x
                    for x in self.cudagraph_batch_sizes
                    if max_num_tokens >= x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    product(reversed(decode_cudagraph_batch_sizes), lora_cases)
                )
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True,
                )

            torch.cuda.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _capture_cudagraphs(
        self,
        compilation_cases: list[tuple[int, bool]],
        cudagraph_runtime_mode: CUDAGraphMode,
        uniform_decode: bool,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.valid_runtime_modes()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens, activate_lora in compilation_cases:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            )

            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
                self._dummy_run(
                    num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=force_attention,
                    uniform_decode=uniform_decode,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    activate_lora=activate_lora,
                )
            self._dummy_run(
                num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=uniform_decode,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                activate_lora=activate_lora,
                is_graph_capturing=True,
            )
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(
            attention_backend_list, kv_cache_config.kv_cache_groups
        )

        # Check if attention backend supports PCP&DCP and related features.
        check_attention_cp_compatibility(self.vllm_config)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_backend_name = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_backend_name = attn_backend.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_backend_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_backend_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # if we have dedicated decode cudagraphs, and spec-decode is enabled,
        # we need to adjust the cudagraph sizes to be a multiple of the uniform
        # decode query length to avoid: https://github.com/vllm-project/vllm/issues/28207
        # temp-fix: https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have separate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
            and self.uniform_decode_query_len > 1
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size
            )
            capture_sizes = self.compilation_config.cudagraph_capture_sizes
            self.cudagraph_batch_sizes = (
                capture_sizes if capture_sizes is not None else []
            )

        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.compilation_config.cudagraph_mode = cudagraph_mode
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    @staticmethod
    def select_common_block_size(
        kv_manager_block_size: int, attn_groups: list[AttentionGroup]
    ) -> int:
        """
        Select a block size that is supported by all backends and is a factor of
        kv_manager_block_size.

        If kv_manager_block_size is supported by all backends, return it directly.
        Otherwise, return the max supported size.

        Args:
            kv_manager_block_size: Block size of KV cache
            attn_groups: List of attention groups

        Returns:
            The selected block size

        Raises:
            ValueError: If no valid block size found
        """

        def block_size_is_supported(
            backends: list[type[AttentionBackend]], block_size: int
        ) -> bool:
            """
            Check if the block size is supported by all backends.
            """
            for backend in backends:
                is_supported = False
                for supported_size in backend.get_supported_kernel_block_sizes():
                    if isinstance(supported_size, int):
                        if block_size == supported_size:
                            is_supported = True
                    elif isinstance(supported_size, MultipleOf):
                        if block_size % supported_size.base == 0:
                            is_supported = True
                    else:
                        raise ValueError(f"Unknown supported size: {supported_size}")
                if not is_supported:
                    return False
            return True

        backends = [group.backend for group in attn_groups]

        # Case 1: if the block_size of kv cache manager is supported by all backends,
        # return it directly
        if block_size_is_supported(backends, kv_manager_block_size):
            return kv_manager_block_size

        # Case 2: otherwise, the block_size must be an `int`-format supported size of
        # at least one backend. Iterate over all `int`-format supported sizes in
        # descending order and return the first one that is supported by all backends.
        # Simple proof:
        # If the supported size b is in MultipleOf(x_i) format for all attention
        # backends i, and b a factor of kv_manager_block_size, then
        # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
        # return kv_manager_block_size in case 1.
        all_int_supported_sizes = set(
            supported_size
            for backend in backends
            for supported_size in backend.get_supported_kernel_block_sizes()
            if isinstance(supported_size, int)
        )

        for supported_size in sorted(all_int_supported_sizes, reverse=True):
            if kv_manager_block_size % supported_size != 0:
                continue
            if block_size_is_supported(backends, supported_size):
                return supported_size
        raise ValueError(f"No common block size for {kv_manager_block_size}. ")

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [
            self.cache_config.block_size
        ]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self._build_input_batch_logitsprocs(),
                logitsprocs_need_output_token_ids=(
                    self._logitsprocs_need_output_token_ids
                ),
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=self.num_spec_tokens,
                cp_kv_cache_interleave_size=(
                    self.parallel_config.cp_kv_cache_interleave_size
                ),
            )
            self._input_batch_block_sizes = block_sizes
            self._input_batch_kernel_block_sizes = kernel_block_sizes
            self._pd_phase_contexts.clear()

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _prepare_kernel_block_sizes(self, kv_cache_config: KVCacheConfig) -> list[int]:
        """
        Generate kernel_block_sizes that matches each block_size.

        For attention backends that support virtual block splitting,
        use the supported block sizes from the backend.
        For other backends (like Mamba), use the same block size (no splitting).

        Args:
            kv_cache_config: The KV cache configuration.

        Returns:
            list[int]: List of kernel block sizes for each cache group.
        """
        kernel_block_sizes = []
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # all backends in the group.
                attn_groups = self.attn_groups[kv_cache_gid]
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
                selected_kernel_size = self.select_common_block_size(
                    kv_manager_block_size, attn_groups
                )
                kernel_block_sizes.append(selected_kernel_size)
            elif isinstance(kv_cache_spec, MambaSpec):
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                kernel_block_sizes.append(kv_cache_spec.block_size)
            else:
                raise NotImplementedError(
                    f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
                )
        return kernel_block_sizes

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)
                        .view(kv_cache_shape)
                        .permute(*inv_order)
                    )
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size
                        )
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
            )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        block_size = self.cache_config.block_size
        self.max_num_kv_tokens = (
            self.kv_cache_config.num_blocks // len(self.kv_cache_config.kv_cache_groups)
            + 1
        ) * block_size

        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            model_config=self.model_config,
            instance_id=self.vllm_config.instance_id,
        )

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()
