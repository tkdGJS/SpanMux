# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

import os
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.kv_retention_layerwise import active_kv_retention_ring
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.context_ring_correctness import (
    attention_correctness_trace_enabled,
    maybe_record_attention_output,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.attention.ops.tiered_d2h_source_plan import TieredD2HSourcePlan
from vllm.v1.attention.ops.tiered_residency_provider import (
    TieredAttentionPagePlanBundle,
    acquire_dram_kv_pool_slots,
    build_dram_kv_page_lookup_plan,
    build_tiered_attention_page_tensor_plan,
    get_dram_kv_request_key,
    has_pending_hbm_d2h_reads,
    record_decode_reload_completion_barrier,
    register_hbm_kv_cache_pages,
    reload_dram_kv_pages_to_hbm,
    record_hbm_kv_cache_registration_skip,
    tiered_attention_plan_tensors_enabled,
    tiered_dram_page_provider_enabled,
    tiered_dram_register_from_hbm_enabled,
    tiered_dram_register_pressure_only_enabled,
    tiered_dram_reload_to_hbm_enabled,
    wait_for_pending_hbm_d2h_reads,
    wait_for_pending_hbm_d2h_request_reads,
)
from vllm.v1.attention.ops.tiered_blockwise_backend import (
    try_run_tiered_blockwise_backend,
)
from vllm.v1.attention.ops.tiered_layer_reload_prefetch import (
    consume_tiered_layer_reload_prefetch,
    register_tiered_layer_kv_cache,
    submit_next_tiered_layer_reload_prefetch,
)
from vllm.v1.core.tiered_errors import TieredBackendEligibilityError
from vllm.v1.core.tiered_execution import (
    TIERED_EXECUTION_AUTO,
    resolve_tiered_reload_authority,
)
from vllm.v1.core.virtual_context import (
    get_virtual_context_table,
    virtual_context_decode_cold_streaming_enabled,
    virtual_context_enabled,
    virtual_context_layer_id,
    virtual_context_prefill_ring_required_mode,
    virtual_context_ring_backend_enabled,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _dram_register_decode_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_DECODE", "")
        .lower()
        in _TRUE_VALUES
    )


def _decode_ring_fallback_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_DECODE_RING_FALLBACK", "")
        .lower()
        in _TRUE_VALUES
    )


def _attention_phase(attn_metadata: "TritonAttentionMetadata") -> str:
    return (getattr(attn_metadata, "attention_phase", "") or "").lower()


def _is_prefill_phase(attn_metadata: "TritonAttentionMetadata") -> bool:
    return _attention_phase(attn_metadata) == "prefill"


def _has_reload_required_pages(
    attn_metadata: "TritonAttentionMetadata",
) -> bool:
    return _has_reload_required_page_map(
        getattr(attn_metadata, "tiered_reload_required_pages", None)
    )


def _has_reload_required_page_map(
    pages_by_request: dict[str, set[int]] | None,
) -> bool:
    if not pages_by_request:
        return False
    return any(bool(pages) for pages in pages_by_request.values())


def _reload_required_page_count(
    pages_by_request: dict[str, set[int]] | None,
) -> int:
    if not pages_by_request:
        return 0
    return sum(len(pages) for pages in pages_by_request.values())


def _layer_specific_tiered_page_state(
    attn_metadata: "TritonAttentionMetadata",
    layer_name: str,
) -> tuple[dict[str, dict[int, int]] | None, dict[str, set[int]] | None]:
    """Return virtual-context page state scoped to the current attention layer.

    CommonAttentionMetadata is created once per model execution and therefore
    cannot naturally carry per-layer reload state. The provider, however, stores
    DRAM pages by layer_name. Narrowing here prevents a reload-required page in
    one layer from forcing every other layer through the same reload/ring path.
    """
    request_ids = getattr(attn_metadata, "request_ids", None)
    if not request_ids or not layer_name or not tiered_dram_page_provider_enabled():
        return (
            attn_metadata.tiered_page_versions,
            attn_metadata.tiered_reload_required_pages,
        )
    virtual_context = get_virtual_context_table()
    if virtual_context is None:
        return (
            attn_metadata.tiered_page_versions,
            attn_metadata.tiered_reload_required_pages,
        )

    layer_id = virtual_context_layer_id(layer_name)
    page_versions = virtual_context.pin_attention_page_versions(
        request_ids,
        group_id=0,
        layer_id=layer_id,
    )
    reload_pages = virtual_context.plan_hbm_reload_pages(
        request_ids=set(request_ids),
        layer_id=layer_id,
    )
    reload_pages_by_request: dict[str, set[int]] = {}
    for page in reload_pages:
        reload_pages_by_request.setdefault(page.request_id, set()).add(
            page.logical_token_page
        )
    return (
        page_versions or None,
        reload_pages_by_request or None,
    )


def _reload_required_prefill_should_use_ring_first(
    attn_metadata: "TritonAttentionMetadata",
    reload_required_pages: dict[str, set[int]] | None,
) -> bool:
    if not _has_reload_required_page_map(reload_required_pages):
        return False
    if not _is_prefill_phase(attn_metadata):
        return False
    if virtual_context_prefill_ring_required_mode() not in {
        "always",
        "pressure",
    }:
        return False
    return (
        virtual_context_ring_backend_enabled()
        and tiered_dram_page_provider_enabled()
    )


def _reload_required_decode_should_use_ring(
    attn_metadata: "TritonAttentionMetadata",
    reload_required_pages: dict[str, set[int]] | None,
) -> bool:
    if _is_prefill_phase(attn_metadata):
        return False
    if not _has_reload_required_page_map(reload_required_pages):
        return False
    if (
        tiered_dram_reload_to_hbm_enabled()
        and os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_DECODE_RING_PREFER_OVER_RELOAD",
            "",
        ).lower()
        not in _TRUE_VALUES
    ):
        return False
    if not (
        virtual_context_decode_cold_streaming_enabled()
        or _decode_ring_fallback_enabled()
    ):
        return False
    return (
        virtual_context_ring_backend_enabled()
        and tiered_dram_page_provider_enabled()
    )


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):
        return int(value.item())  # type: ignore[union-attr]
    return int(value)  # type: ignore[arg-type]


def _prefill_partial_tail_reload_pages(
    attn_metadata: "TritonAttentionMetadata",
    block_size: int,
    reload_required_pages: dict[str, set[int]] | None,
) -> tuple[
    dict[str, set[int]],
    dict[str, dict[int, int]],
    dict[str, dict[int, int]],
]:
    """Select a DRAM prefix page shared with the current prefill query.

    Full historical pages can stay on the context-ring path. If history ends
    in the middle of a page, the current query appends K/V into that same
    logical page. The allocator gives the page a concrete HBM block and this
    helper selects its historical prefix for reload before the combined page
    is registered or consumed.
    """
    if not _is_prefill_phase(attn_metadata) or block_size <= 0:
        return {}, {}, {}
    required_pages = reload_required_pages or {}
    request_ids = attn_metadata.request_ids or []
    query_start_loc = attn_metadata.query_start_loc
    seq_lens = attn_metadata.seq_lens
    num_reqs = min(
        len(request_ids),
        int(seq_lens.numel()),
        max(0, int(query_start_loc.numel()) - 1),
    )
    partial_pages: dict[str, set[int]] = {}
    provider_token_counts: dict[str, dict[int, int]] = {}
    hbm_token_counts: dict[str, dict[int, int]] = {}
    for req_idx in range(num_reqs):
        request_id = request_ids[req_idx]
        request_required_pages = required_pages.get(request_id)
        if not request_required_pages:
            continue
        query_len = _as_int(
            query_start_loc[req_idx + 1] - query_start_loc[req_idx]
        )
        seq_len = _as_int(seq_lens[req_idx])
        historical_tokens = max(0, seq_len - query_len)
        if historical_tokens <= 0 or historical_tokens % block_size == 0:
            continue
        partial_page = historical_tokens // block_size
        if partial_page in request_required_pages:
            partial_pages[request_id] = {partial_page}
            # Freeze the pre-query DRAM prefix before reshape-and-cache grows
            # this logical HBM page.  CPU-side metadata can already describe
            # the post-query length when the reload helper runs, but the
            # provider snapshot still owns only this historical prefix.
            provider_token_counts[request_id] = {
                partial_page: historical_tokens % block_size
            }
            hbm_token_counts[request_id] = {
                partial_page: min(
                    block_size,
                    seq_len - partial_page * block_size,
                )
            }
    return partial_pages, provider_token_counts, hbm_token_counts


def _merge_reloaded_pages(
    destination: dict[str, set[int]],
    source: dict[str, set[int]],
) -> None:
    for request_id, pages in source.items():
        destination.setdefault(request_id, set()).update(pages)


def _prepare_tiered_attention_page_tensor_plans(
    attn_metadata: "TritonAttentionMetadata",
    layer_name: str,
    key_cache: torch.Tensor,
    *,
    page_versions_by_request: dict[str, dict[int, int]] | None,
    required_pages_by_request: dict[str, set[int]] | None,
) -> dict[str, TieredAttentionPagePlanBundle] | None:
    if not tiered_attention_plan_tensors_enabled():
        return None
    request_ids = getattr(attn_metadata, "request_ids", None)
    if not request_ids or not layer_name or not tiered_dram_page_provider_enabled():
        return None
    if key_cache.dim() != 4:
        return None

    block_table = attn_metadata.block_table
    block_table_cpu = attn_metadata.tiered_block_table_cpu
    seq_lens = attn_metadata.seq_lens
    seq_lens_cpu = attn_metadata.seq_lens_cpu
    block_size = int(key_cache.shape[1])
    num_reqs = min(
        len(request_ids),
        int(seq_lens.numel()),
        int(block_table.shape[0]),
    )
    if num_reqs <= 0:
        return None

    required_pages_by_request = required_pages_by_request or {}
    request_numeric_ids = getattr(attn_metadata, "request_numeric_ids", None)
    request_numeric_ids_tensor = getattr(
        attn_metadata,
        "request_numeric_ids_tensor",
        None,
    )
    request_provider_keys = getattr(attn_metadata, "request_provider_keys", None)
    request_provider_keys_tensor = getattr(
        attn_metadata,
        "request_provider_keys_tensor",
        None,
    )
    layer_id = virtual_context_layer_id(layer_name)
    plans: dict[str, TieredAttentionPagePlanBundle] = {}

    for req_idx in range(num_reqs):
        request_id = request_ids[req_idx]
        request_key = None
        if (
            request_provider_keys is not None
            and req_idx < len(request_provider_keys)
        ):
            candidate_key = int(request_provider_keys[req_idx])
            if candidate_key >= 0:
                request_key = candidate_key
        elif (
            request_provider_keys_tensor is not None
            and req_idx < int(request_provider_keys_tensor.numel())
        ):
            candidate_key = _as_int(request_provider_keys_tensor[req_idx])
            if candidate_key >= 0:
                request_key = candidate_key
        if request_key is None:
            request_key = get_dram_kv_request_key(request_id, create=False)
        seq_len = (
            int(seq_lens_cpu[req_idx])
            if seq_lens_cpu is not None
            else _as_int(seq_lens[req_idx])
        )
        if seq_len <= 0:
            continue
        num_blocks = min(
            int(block_table.shape[1]),
            (seq_len + block_size - 1) // block_size,
        )
        if num_blocks <= 0:
            continue

        page_token_counts = {
            logical_page: min(
                block_size,
                seq_len - logical_page * block_size,
            )
            for logical_page in range(num_blocks)
        }
        page_versions = (
            page_versions_by_request.get(request_id)
            if page_versions_by_request is not None
            else None
        )
        required_pages = required_pages_by_request.get(request_id, set())
        provider_page_token_counts = (
            {
                logical_page: page_token_counts[logical_page]
                for logical_page in sorted(required_pages)
                if logical_page in page_token_counts
            }
            if required_pages
            else page_token_counts
        )
        dram_plan = None
        pool_leases_by_page: dict[int, object] = {}
        for _attempt in range(2):
            dram_plan = build_dram_kv_page_lookup_plan(
                request_id=request_id,
                request_key=request_key,
                layer_name=layer_name,
                layer_id=layer_id,
                page_token_counts=provider_page_token_counts,
                page_versions=page_versions,
                strict_pages=required_pages,
            )
            pool_pages = [
                logical_page
                for logical_page in sorted(required_pages)
                if (
                    (lookup := dram_plan.lookups.get(logical_page))
                    is not None
                    and lookup.source == "pool"
                )
            ]
            if not pool_pages:
                break
            leases = acquire_dram_kv_pool_slots([
                (
                    dram_plan.lookups[logical_page].slot_id,
                    dram_plan.lookups[logical_page].version,
                )
                for logical_page in pool_pages
            ])
            if leases is not None:
                pool_leases_by_page = dict(zip(pool_pages, leases))
                break
        else:
            raise TieredBackendEligibilityError(
                "DRAM provider slots changed while building the tiered "
                f"attention plan: request={request_id!r} "
                f"layer={layer_name!r}"
            )
        assert dram_plan is not None
        block_ids_by_page = {
            logical_page: (
                int(block_table_cpu[req_idx, logical_page])
                if block_table_cpu is not None
                else _as_int(block_table[req_idx, logical_page])
            )
            for logical_page in range(num_blocks)
        }
        request_numeric_id = -1
        if (
            request_numeric_ids is not None
            and req_idx < len(request_numeric_ids)
        ):
            request_numeric_id = int(request_numeric_ids[req_idx])
        elif (
            request_numeric_ids_tensor is not None
            and req_idx < int(request_numeric_ids_tensor.numel())
        ):
            request_numeric_id = _as_int(request_numeric_ids_tensor[req_idx])

        tensor_plan = build_tiered_attention_page_tensor_plan(
            request_id=request_id,
            layer_name=layer_name,
            layer_id=layer_id,
            page_token_counts=page_token_counts,
            block_ids_by_page=block_ids_by_page,
            dram_plan=dram_plan,
            required_dram_pages=required_pages_by_request.get(
                request_id,
                set(),
            ),
            block_size=block_size,
            request_numeric_id=request_numeric_id,
            device=block_table.device,
        )
        plans[request_id] = TieredAttentionPagePlanBundle(
            dram_plan=dram_plan,
            tensor_plan=tensor_plan,
            pool_leases_by_page=pool_leases_by_page,
        )
    return plans or None


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor | None
    max_seq_len: int
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    block_table: torch.Tensor
    tiered_block_table_cpu: torch.Tensor | None
    tiered_d2h_source_plan: TieredD2HSourcePlan | None
    slot_mapping: torch.Tensor
    request_ids: list[str] | None
    request_numeric_ids: list[int] | None
    request_numeric_ids_tensor: torch.Tensor | None
    request_provider_keys: list[int] | None
    request_provider_keys_tensor: torch.Tensor | None
    attention_phase: str | None
    tiered_page_versions: dict[str, dict[int, int]] | None
    tiered_reload_required_pages: dict[str, set[int]] | None
    tiered_disable_native_hbm_reload: bool
    tiered_execution_plans: dict[str, tuple[int, str, int]] | None
    tiered_pressure_snapshot_request_ids: set[str] | None
    tiered_proactive_snapshot_pages: dict[str, set[int]] | None
    tiered_dma_budget_bytes: int
    tiered_staging_budget_blocks: int
    tiered_max_prefill_ring_tokens: int
    tiered_decode_tbt_budget_ms: float
    tiered_attention_page_tensor_plans: dict[str, Any] | None
    tiered_attention_page_row_plans: list[Any | None] | None

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None

    @property
    def mm_prefix_range_tensor(self) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        # TODO(Isotr0py): Move to model runner's attention metadata
        # preparation to avoid duplicate computation.
        if self.mm_prefix_range is None:
            return None

        num_seqs = self.seq_lens.shape[0]
        device = self.seq_lens.device

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            self.mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Create 2D tensors with shape (num_ranges, 2) for each sequence
        range_tensors = [
            torch.tensor(r, dtype=torch.int32, device=device).view(-1, 2)
            for r in range_lists
        ]

        return torch.nested.nested_tensor(range_tensors).to_padded_tensor(0)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=(
                common_attn_metadata.query_start_loc_cpu
            ),
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            block_table=block_table_tensor,
            tiered_block_table_cpu=(
                common_attn_metadata.tiered_block_table_cpu
            ),
            tiered_d2h_source_plan=(
                common_attn_metadata.tiered_d2h_source_plan
            ),
            slot_mapping=slot_mapping,
            request_ids=common_attn_metadata.request_ids,
            request_numeric_ids=common_attn_metadata.request_numeric_ids,
            request_numeric_ids_tensor=(
                common_attn_metadata.request_numeric_ids_tensor
            ),
            request_provider_keys=common_attn_metadata.request_provider_keys,
            request_provider_keys_tensor=(
                common_attn_metadata.request_provider_keys_tensor
            ),
            attention_phase=common_attn_metadata.attention_phase,
            tiered_page_versions=common_attn_metadata.tiered_page_versions,
            tiered_reload_required_pages=(
                common_attn_metadata.tiered_reload_required_pages
            ),
            tiered_disable_native_hbm_reload=(
                common_attn_metadata.tiered_disable_native_hbm_reload
            ),
            tiered_execution_plans=(
                common_attn_metadata.tiered_execution_plans
            ),
            tiered_pressure_snapshot_request_ids=(
                common_attn_metadata.tiered_pressure_snapshot_request_ids
            ),
            tiered_proactive_snapshot_pages=(
                common_attn_metadata.tiered_proactive_snapshot_pages
            ),
            tiered_dma_budget_bytes=(
                common_attn_metadata.tiered_dma_budget_bytes
            ),
            tiered_staging_budget_blocks=(
                common_attn_metadata.tiered_staging_budget_blocks
            ),
            tiered_max_prefill_ring_tokens=(
                common_attn_metadata.tiered_max_prefill_ring_tokens
            ),
            tiered_decode_tbt_budget_ms=(
                common_attn_metadata.tiered_decode_tbt_budget_ms
            ),
            tiered_attention_page_tensor_plans=(
                common_attn_metadata.tiered_attention_page_tensor_plans
            ),
            tiered_attention_page_row_plans=(
                common_attn_metadata.tiered_attention_page_row_plans
            ),
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        if include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)

        # (num_blocks, 2, block_size, num_kv_heads, head_size)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()
        # Virtual context is a process-start configuration. Cache the master
        # gate once so the native attention path does not read environment
        # state or enter tiered helpers for every layer.
        self._virtual_context_enabled = virtual_context_enabled()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

        self.supports_quant_query_input = current_platform.is_cuda()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # MUST precede reshape/cache writes and reads of the retained Native
        # block table: this continuation has returned those physical blocks.
        ring_owner = active_kv_retention_ring()
        if ring_owner is not None:
            return ring_owner.attention(
                layer.layer_name, self, query, key, value, output, attn_metadata
            )

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)
        layer_name = getattr(layer, "layer_name", "")
        pressure_snapshot_request_ids = (
            set(
                attn_metadata.tiered_pressure_snapshot_request_ids or ()
            ).intersection(attn_metadata.request_ids or ())
            if self._virtual_context_enabled
            else set()
        )
        pressure_only_registration = (
            tiered_dram_register_pressure_only_enabled()
            if self._virtual_context_enabled
            else True
        )
        tiered_control_required = (
            self._virtual_context_enabled
            and (
                not pressure_only_registration
                or _has_reload_required_pages(attn_metadata)
                or bool(attn_metadata.tiered_disable_native_hbm_reload)
                or bool(pressure_snapshot_request_ids)
            )
        )
        decode_d2h_barrier_requested = (
            tiered_control_required
            and not _is_prefill_phase(attn_metadata)
            and _has_reload_required_pages(attn_metadata)
        )
        if decode_d2h_barrier_requested:
            # Decode handoff is stronger than the overwrite fence below. Every
            # asynchronous D2H provider snapshot for this immutable request
            # must be visible before consuming any layer-ahead H2D prefetch.
            wait_for_pending_hbm_d2h_request_reads(
                attn_metadata.request_ids
            )
        if tiered_control_required:
            register_tiered_layer_kv_cache(
                layer_name,
                key_cache,
                value_cache,
            )
            layer_prefetched_reloaded_pages = (
                consume_tiered_layer_reload_prefetch(
                    layer_name,
                    attn_metadata.request_ids,
                )
            )
        else:
            layer_prefetched_reloaded_pages = {}
        partial_tail_reloaded_pages: dict[str, set[int]] = {}
        layer_page_versions = attn_metadata.tiered_page_versions
        layer_reload_required_pages = (
            attn_metadata.tiered_reload_required_pages
        )
        if tiered_control_required:
            (
                layer_page_versions,
                layer_reload_required_pages,
            ) = _layer_specific_tiered_page_state(attn_metadata, layer_name)

        # Native promotion carries the same allocator proof as Decode handoff.
        # Freeze it before partial-tail copies can consume the last reload page:
        # later layers/prefetch still need the proof even if this layer is ready.
        native_reload_destinations_concrete = (
            resolve_tiered_reload_authority(
                phase=str(attn_metadata.attention_phase or ""),
                reload_required_pages=layer_reload_required_pages,
                execution_plans=attn_metadata.tiered_execution_plans,
            ) == "native"
            if tiered_control_required else False
        )

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            if tiered_control_required or (
                self._virtual_context_enabled
                and has_pending_hbm_d2h_reads(attn_metadata.request_ids)
            ):
                wait_for_pending_hbm_d2h_reads(
                    layer_name=layer_name,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    query_start_loc=attn_metadata.query_start_loc,
                    request_ids=attn_metadata.request_ids,
                    block_size=int(key_cache.shape[1]),
                    block_table_cpu=attn_metadata.tiered_block_table_cpu,
                    seq_lens_cpu=attn_metadata.seq_lens_cpu,
                    query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
                )
            if self.kv_cache_dtype.startswith("fp8"):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
                # triton kernel does not support uint8 kv_cache
                #  (because some explicit casts (e.g. float8_e4m3fnuz)
                #   are not supported)
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            (
                partial_tail_reload_pages,
                partial_tail_provider_token_counts,
                partial_tail_hbm_token_counts,
            ) = (
                _prefill_partial_tail_reload_pages(
                    attn_metadata,
                    int(key_cache.shape[1]),
                    layer_reload_required_pages,
                )
                if self._virtual_context_enabled
                else ({}, {}, {})
            )
            if partial_tail_reload_pages:
                tail_reload_diagnostics: dict[str, int | str] = {}
                partial_tail_reloaded_pages = reload_dram_kv_pages_to_hbm(
                    layer_name=layer_name,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    query_start_loc=attn_metadata.query_start_loc,
                    request_ids=attn_metadata.request_ids,
                    block_size=int(key_cache.shape[1]),
                    page_versions_by_request=layer_page_versions,
                    reload_pages_by_request=partial_tail_reload_pages,
                    expected_token_counts_by_request=(
                        partial_tail_provider_token_counts
                    ),
                    hbm_token_counts_by_request=(
                        partial_tail_hbm_token_counts
                    ),
                    block_table_cpu=attn_metadata.tiered_block_table_cpu,
                    seq_lens_cpu=attn_metadata.seq_lens_cpu,
                    query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
                    block_table_is_fully_concrete=(
                        native_reload_destinations_concrete
                    ),
                    diagnostics=tail_reload_diagnostics,
                )
                missing_partial_pages = {
                    request_id: set(pages)
                    - partial_tail_reloaded_pages.get(request_id, set())
                    for request_id, pages in partial_tail_reload_pages.items()
                }
                missing_partial_pages = {
                    request_id: pages
                    for request_id, pages in missing_partial_pages.items()
                    if pages
                }
                if missing_partial_pages:
                    raise TieredBackendEligibilityError(
                        "prefill cannot append into a DRAM-authoritative "
                        "partial KV page until its historical prefix is "
                        "reloaded into the concrete HBM tail block: "
                        f"missing={missing_partial_pages}, "
                        f"reload_diagnostics={tail_reload_diagnostics}"
                    )
                (
                    layer_page_versions,
                    layer_reload_required_pages,
                ) = _layer_specific_tiered_page_state(
                    attn_metadata,
                    layer_name,
                )
            if self._virtual_context_enabled:
                register_phase = (
                    attn_metadata.attention_phase or ""
                ).strip().lower()
                decode_like_batch = register_phase in {"decode", "mixed"}
                if not register_phase:
                    decode_like_batch = (
                        attn_metadata.max_query_len <= 1
                        and key is not None
                        and int(key.shape[0])
                        <= max(1, len(attn_metadata.request_ids or []))
                    )
                should_register_hbm = tiered_dram_register_from_hbm_enabled()
                allowed_registration_request_ids: set[str] | None = None
                if should_register_hbm and pressure_only_registration:
                    allowed_registration_request_ids = set(
                        pressure_snapshot_request_ids
                    )
                    for request_id, pages in (
                        layer_reload_required_pages or {}
                    ).items():
                        if pages:
                            allowed_registration_request_ids.add(request_id)
                    if not allowed_registration_request_ids:
                        should_register_hbm = False
                if (
                    should_register_hbm
                    and decode_like_batch
                    and not _dram_register_decode_enabled()
                ):
                    record_hbm_kv_cache_registration_skip(
                        phase=register_phase or "decode_like",
                        reason="decode_hotset_native_path",
                    )
                    should_register_hbm = False
                if should_register_hbm:
                    register_hbm_kv_cache_pages(
                        layer_name=layer_name,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=attn_metadata.block_table,
                        seq_lens=attn_metadata.seq_lens,
                        slot_mapping=attn_metadata.slot_mapping,
                        query_start_loc=attn_metadata.query_start_loc,
                        request_ids=attn_metadata.request_ids,
                        page_versions_by_request=layer_page_versions,
                        allowed_request_ids=allowed_registration_request_ids,
                        force_all_request_ids=pressure_snapshot_request_ids,
                        exact_force_pages_by_request=(
                            attn_metadata.tiered_proactive_snapshot_pages
                        ),
                        source_page_plan=attn_metadata.tiered_d2h_source_plan,
                    )

        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        attn_metadata.tiered_attention_page_tensor_plans = None
        attn_metadata.tiered_attention_page_row_plans = None

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        reload_authority = resolve_tiered_reload_authority(
            phase=str(attn_metadata.attention_phase or ""),
            reload_required_pages=layer_reload_required_pages,
            execution_plans=attn_metadata.tiered_execution_plans,
        )
        if reload_authority == "ring":
            decode_should_use_ring = not _is_prefill_phase(attn_metadata)
            prefill_should_use_ring = _is_prefill_phase(attn_metadata)
        elif reload_authority == "native":
            decode_should_use_ring = False
            prefill_should_use_ring = False
        else:
            assert reload_authority == TIERED_EXECUTION_AUTO
            decode_should_use_ring = (
                self._virtual_context_enabled
                and _reload_required_decode_should_use_ring(
                    attn_metadata,
                    layer_reload_required_pages,
                )
            )
            prefill_should_use_ring = (
                self._virtual_context_enabled
                and _is_prefill_phase(attn_metadata)
                and (
                    bool(
                        getattr(
                            attn_metadata,
                            "tiered_disable_native_hbm_reload",
                            False,
                        )
                    )
                    or _reload_required_prefill_should_use_ring_first(
                        attn_metadata,
                        layer_reload_required_pages,
                    )
                )
            )
        disable_native_hbm_reload = (
            prefill_should_use_ring or decode_should_use_ring
        )
        decode_reload_barrier_requested = (
            decode_d2h_barrier_requested
            and not disable_native_hbm_reload
        )
        reloaded_pages: dict[str, set[int]] = {}
        _merge_reloaded_pages(
            reloaded_pages,
            layer_prefetched_reloaded_pages,
        )
        _merge_reloaded_pages(
            reloaded_pages,
            partial_tail_reloaded_pages,
        )
        decode_reload_h2d_retry_count = 0
        decode_reload_h2d_retry_page_count = 0
        native_reload_diagnostics: dict[str, int | str] = {}
        if self._virtual_context_enabled and not disable_native_hbm_reload:
            # Refresh after every acknowledged H2D pass. A decode handoff can
            # otherwise consume metadata built just before the final async D2H
            # provider publication and leave a newly authoritative page out of
            # the one-shot reload plan.
            max_reload_passes = 3 if decode_reload_barrier_requested else 1
            for reload_pass in range(max_reload_passes):
                if not _has_reload_required_page_map(
                    layer_reload_required_pages
                ):
                    break
                if reload_pass > 0:
                    decode_reload_h2d_retry_count += 1
                    decode_reload_h2d_retry_page_count += (
                        _reload_required_page_count(
                            layer_reload_required_pages
                        )
                    )
                native_reloaded_pages = reload_dram_kv_pages_to_hbm(
                    layer_name=layer_name,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table,
                    seq_lens=seqused_k,
                    query_start_loc=attn_metadata.query_start_loc,
                    request_ids=attn_metadata.request_ids,
                    block_size=int(key_cache.shape[1]),
                    page_versions_by_request=layer_page_versions,
                    reload_pages_by_request=layer_reload_required_pages,
                    block_table_cpu=attn_metadata.tiered_block_table_cpu,
                    seq_lens_cpu=attn_metadata.seq_lens_cpu,
                    query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
                    block_table_is_fully_concrete=(
                        native_reload_destinations_concrete
                    ),
                    diagnostics=native_reload_diagnostics,
                )
                _merge_reloaded_pages(
                    reloaded_pages, native_reloaded_pages
                )
                (
                    layer_page_versions,
                    layer_reload_required_pages,
                ) = _layer_specific_tiered_page_state(
                    attn_metadata,
                    layer_name,
                )
        remaining_reload_pages = _reload_required_page_count(
            layer_reload_required_pages
        )
        if decode_reload_barrier_requested:
            record_decode_reload_completion_barrier(
                h2d_retry_count=decode_reload_h2d_retry_count,
                h2d_retry_page_count=decode_reload_h2d_retry_page_count,
                remaining_page_count=remaining_reload_pages,
            )
        reload_to_hbm_native_ready = False
        if decode_reload_barrier_requested:
            reload_to_hbm_native_ready = remaining_reload_pages == 0
        elif reloaded_pages and not _has_reload_required_page_map(
            layer_reload_required_pages
        ):
            reload_to_hbm_native_ready = True
        elif reloaded_pages and layer_reload_required_pages:
            reload_to_hbm_native_ready = True
            for request_id, pages in (
                layer_reload_required_pages.items()
            ):
                if not set(pages).issubset(
                    reloaded_pages.get(request_id, set())
                ):
                    reload_to_hbm_native_ready = False
                    break

        # Build DRAM lookup/page tensor plans only after the native reload
        # attempt. Reload-required pages may be resolvable by copying the
        # provider copy back into HBM; doing the strict provider lookup before
        # that point can turn a recoverable native handoff into a fatal stale
        # partial-tail mismatch.
        if tiered_control_required:
            attn_metadata.tiered_attention_page_tensor_plans = (
                _prepare_tiered_attention_page_tensor_plans(
                    attn_metadata,
                    layer_name,
                    key_cache,
                    page_versions_by_request=layer_page_versions,
                    required_pages_by_request=(
                        layer_reload_required_pages
                    ),
                )
            )
        else:
            attn_metadata.tiered_attention_page_tensor_plans = None
        if (
            attn_metadata.tiered_attention_page_tensor_plans is not None
            and attn_metadata.request_ids is not None
        ):
            attn_metadata.tiered_attention_page_row_plans = [
                attn_metadata.tiered_attention_page_tensor_plans.get(
                    request_id
                )
                for request_id in attn_metadata.request_ids
            ]

        should_attempt_tiered_backend = self._virtual_context_enabled and (
            decode_should_use_ring
            or (
                _is_prefill_phase(attn_metadata)
                and (
                    not pressure_only_registration
                    or prefill_should_use_ring
                )
            )
        )
        if should_attempt_tiered_backend:
            if try_run_tiered_blockwise_backend(
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],
                attn_metadata=attn_metadata,
                attn_type=self.attn_type,
                kv_cache_dtype=self.kv_cache_dtype,
                scale=self.scale,
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                sinks=self.sinks,
                output_scale=output_scale,
                page_versions_by_request=layer_page_versions,
                reload_required_pages_by_request=(
                    layer_reload_required_pages
                ),
                layer_name=layer_name,
                reload_to_hbm_native_ready=reload_to_hbm_native_ready,
            ):
                if attention_correctness_trace_enabled():
                    maybe_record_attention_output(
                        execution_path="context_ring",
                        layer_name=layer_name,
                        attn_metadata=attn_metadata,
                        output=output[:num_actual_tokens],
                        query=query[:num_actual_tokens],
                        key=(
                            key[:num_actual_tokens]
                            if key is not None
                            else None
                        ),
                        value=(
                            value[:num_actual_tokens]
                            if value is not None
                            else None
                        ),
                        softmax_scale=self.scale,
                    )
                return output
        elif (
            self._virtual_context_enabled
            and _has_reload_required_page_map(layer_reload_required_pages)
            and not reload_to_hbm_native_ready
        ):
            raise TieredBackendEligibilityError(
                "Native attention requires all DRAM-resident logical pages "
                "to be reloaded into HBM before execution. "
                f"phase={attn_metadata.attention_phase!r}, "
                f"authority={reload_authority!r}, "
                f"layer={layer_name}, "
                f"remaining_pages={remaining_reload_pages}, "
                f"reload_diagnostics={native_reload_diagnostics}"
            )

        if tiered_control_required:
            submit_next_tiered_layer_reload_prefetch(
                current_layer_name=layer_name,
                block_table=block_table,
                seq_lens=seqused_k,
                query_start_loc=attn_metadata.query_start_loc,
                request_ids=attn_metadata.request_ids,
                block_table_cpu=attn_metadata.tiered_block_table_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu,
                query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
                execution_plans=attn_metadata.tiered_execution_plans,
                attention_phase=str(attn_metadata.attention_phase or ""),
            )

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        if attention_correctness_trace_enabled():
            maybe_record_attention_output(
                execution_path="native_hbm",
                layer_name=layer_name,
                attn_metadata=attn_metadata,
                output=output[:num_actual_tokens],
                query=query[:num_actual_tokens],
                key=(key[:num_actual_tokens] if key is not None else None),
                value=(
                    value[:num_actual_tokens] if value is not None else None
                ),
                softmax_scale=self.scale,
            )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output
