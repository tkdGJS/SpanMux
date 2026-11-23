# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Virtual-context residency state for the experimental ZoneAtten 0.5.x path.

The table is the ownership boundary between vLLM's physical HBM block table
and provider-backed DRAM pages. It records exact per-layer copy coverage,
guards HBM release, and keeps a page DRAM-authoritative until every required
layer has acknowledged a native-HBM reload. The feature remains gated by the
vendor_v4 virtual-context flags; upstream/default vLLM behavior is unchanged.
"""

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Literal

VirtualTier = Literal["HBM", "DRAM", "STAGING"]
VirtualState = Literal["ready", "loading", "storing", "evicting"]
AttentionPhase = Literal["prefill", "decode"]

_TRUE_VALUES = {"1", "true", "yes", "on"}
_PREFILL_RING_REQUIRED_MODES = {"always", "pressure", "never"}
_LAYER_NAME_TO_ID: dict[str, int] = {}
_LAYER_NAME_LOCK = Lock()


def count_full_virtual_external_blocks(
    total_computed_tokens: int,
    num_virtual_external_computed_tokens: int,
    block_size: int,
) -> int:
    """Count block-aligned pages fully covered by virtual external tokens."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_virtual_external_computed_tokens <= 0:
        return 0
    virtual_start = max(
        0,
        total_computed_tokens - num_virtual_external_computed_tokens,
    )
    first_full_page = (virtual_start + block_size - 1) // block_size
    end_full_page = total_computed_tokens // block_size
    return max(0, end_full_page - first_full_page)


def resolve_virtual_context_coverage_block_size(
    default_block_size: int,
    observed_page_sizes: Iterable[int],
) -> int:
    """Resolve page coverage granularity even when no tiered page is ready.

    An empty observed set is expected during the first pressure transition:
    the scheduler must return ``copy_not_ready`` so one bounded native forward
    can create DRAM copies.  Keep the default block size as an explicit list
    member so ``max`` never receives a scalar through empty star expansion.
    """
    if default_block_size <= 0:
        raise ValueError("default_block_size must be positive")
    page_sizes = [default_block_size]
    page_sizes.extend(size for size in observed_page_sizes if size > 0)
    return max(page_sizes)


def virtual_context_prefill_needs_capacity_floor_snapshot(
    safe_capacity_tokens: int,
    scheduled_tokens: int,
    snapshot_lead_tokens: int,
) -> bool:
    """Return whether this prefill must snapshot KV before the HBM floor.

    Pressure-only DRAM registration cannot create a provider copy after the
    scheduler has exhausted every non-reserved KV block: there is no query
    chunk left to submit to the worker. Start repeated snapshots within a
    configured pressure runway so bounded per-forward D2H registration and its
    asynchronous copies can finish before admission reaches zero.
    """
    if safe_capacity_tokens <= 0 or scheduled_tokens <= 0:
        return False
    lead_tokens = max(0, snapshot_lead_tokens)
    return safe_capacity_tokens <= scheduled_tokens + lead_tokens


def virtual_context_pressure_native_reload_min_free_blocks(
    required_blocks: int,
    reserved_blocks: int,
    minimum_progress_tokens: int,
    block_size: int,
) -> int:
    """Return HBM needed for native reload without recreating a zero-progress floor.

    A pressure-deferred prefill may return to the native path only when HBM can
    hold its historical prefix *and* the configured reserve *and* one more
    minimum-progress chunk. Otherwise its DRAM-authoritative prefix must stay
    on the context-ring path.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    progress_blocks = (
        max(0, minimum_progress_tokens) + block_size - 1
    ) // block_size
    return (
        max(0, required_blocks)
        + max(0, reserved_blocks)
        + progress_blocks
    )


def virtual_context_active_decode_ring_progress_reserve_blocks(
    progress_tokens: int,
    block_size: int,
    staging_blocks: int,
    decode_growth_blocks: int,
) -> int:
    """Return hard HBM headroom for one ring-prefill progress quantum.

    The general ring-bandwidth safety reserve is intentionally not included:
    it is adaptive throughput headroom, while this value is the physical
    liveness budget that must survive a native decode hotset reload. The
    caller may still apply the larger safety reserve whenever capacity permits.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    progress_blocks = (
        max(0, progress_tokens) + block_size - 1
    ) // block_size
    return (
        max(0, staging_blocks)
        + max(0, decode_growth_blocks)
        + progress_blocks
    )


def virtual_context_select_active_decode_ring_progress_tokens(
    free_blocks: int,
    progress_tokens: int,
    minimum_progress_tokens: int,
    block_size: int,
    staging_blocks: int,
    decode_growth_blocks: int,
) -> int:
    """Select a protected ring-prefill quantum beside an active decode.

    This is a narrow liveness fallback for the case where the normal
    safe-capacity calculation reaches zero only because its soft safety
    headroom is larger than the remaining free pool. It never spends staging
    or decode-growth headroom and returns zero unless at least the configured
    minimum useful progress can be admitted.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    requested_tokens = max(0, progress_tokens)
    if requested_tokens <= 0:
        return 0
    hard_reserved_blocks = (
        max(0, staging_blocks) + max(0, decode_growth_blocks)
    )
    available_blocks = max(0, free_blocks - hard_reserved_blocks)
    selected_tokens = min(
        requested_tokens,
        available_blocks * block_size,
    )
    selected_tokens = (selected_tokens // block_size) * block_size
    required_minimum = min(
        requested_tokens,
        max(0, minimum_progress_tokens),
    )
    if selected_tokens < required_minimum:
        return 0
    return selected_tokens


def virtual_context_residency_victim_context_tokens(
    num_computed_tokens: int,
    deferred_logical_computed_tokens: int,
) -> int:
    """Return the durable logical context used to rank HBM victims.

    A running request normally reports its resident prefix through
    ``num_computed_tokens``. During a tier transition the durable logical
    prefix can temporarily be larger than that physical counter, so ranking
    must retain the larger value rather than preferring a shorter victim.
    """

    return max(
        0,
        int(num_computed_tokens),
        int(deferred_logical_computed_tokens),
    )


def virtual_context_residency_victim_sort_key(
    logical_context_tokens: int,
    arrival_time: float,
    request_id: str,
) -> tuple[int, float, str]:
    """Rank the longest HBM-resident context first, deterministically."""

    return (
        -max(0, int(logical_context_tokens)),
        float(arrival_time),
        str(request_id),
    )


def virtual_context_enabled() -> bool:
    return os.environ.get("VLLM_VIRTUAL_CONTEXT", "").lower() in _TRUE_VALUES


def virtual_context_timing_enabled() -> bool:
    return os.environ.get("VLLM_VIRTUAL_CONTEXT_TIMING", "").lower() in _TRUE_VALUES


def virtual_context_ring_attention_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_ATTENTION", "").lower()
        in _TRUE_VALUES
    )


def virtual_context_ring_attention_prefill_enabled() -> bool:
    return (
        virtual_context_ring_attention_enabled()
        or os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_PREFILL", "").lower()
        in _TRUE_VALUES
    )


def virtual_context_ring_attention_decode_enabled() -> bool:
    return (
        virtual_context_ring_attention_enabled()
        or os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_DECODE", "").lower()
        in _TRUE_VALUES
    )


def virtual_context_ring_executor_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_EXECUTOR", "").lower()
        in _TRUE_VALUES
        or os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND", "").lower()
        in _TRUE_VALUES
    )


def virtual_context_ring_backend_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND", "").lower()
        in _TRUE_VALUES
    )


def virtual_context_ring_backend_force_all_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL", "")
        .lower()
        in _TRUE_VALUES
    )


def virtual_context_ring_backend_dram_provider_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER", "")
        .lower()
        in _TRUE_VALUES
    )


def virtual_context_prefill_ring_required_enabled() -> bool:
    mode_raw = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED_MODE", ""
    ).lower()
    if mode_raw in _PREFILL_RING_REQUIRED_MODES:
        return mode_raw != "never"
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED", "")
        .lower()
        in _TRUE_VALUES
    )


def virtual_context_prefill_ring_required_mode() -> str:
    mode_raw = os.environ.get(
        "VLLM_VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED_MODE", ""
    ).lower()
    if mode_raw in _PREFILL_RING_REQUIRED_MODES:
        return mode_raw
    if virtual_context_prefill_ring_required_enabled():
        return "always"
    return "never"


def virtual_context_decode_cold_streaming_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_DECODE_COLD_STREAMING", "")
        .lower()
        in _TRUE_VALUES
    )


def virtual_context_decode_cold_streaming_on_prefill_done_enabled() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_DECODE_COLD_STREAMING_ON_PREFILL_DONE",
            "",
        ).lower()
        in _TRUE_VALUES
    )


def virtual_context_reload_to_hbm_enabled() -> bool:
    return (
        os.environ.get("VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM", "").lower()
        in _TRUE_VALUES
    )


def virtual_context_layer_id(layer_name: str | None) -> int:
    """Return a stable in-process id for a model layer name.

    vLLM attention backends naturally identify KV caches by layer name, while
    the virtual context table uses an integer key to keep logical page records
    compact and serializable in debug summaries. Empty layer names keep the
    legacy request/page-only namespace.
    """
    normalized = (layer_name or "").strip()
    if not normalized:
        return -1
    with _LAYER_NAME_LOCK:
        layer_id = _LAYER_NAME_TO_ID.get(normalized)
        if layer_id is None:
            layer_id = len(_LAYER_NAME_TO_ID)
            _LAYER_NAME_TO_ID[normalized] = layer_id
        return layer_id


@dataclass(frozen=True)
class LogicalPageId:
    request_id: str
    # vLLM currently exposes KV-cache groups here, not per-layer pages. The
    # fields below remain explicit so the table can be widened later without
    # changing the metric vocabulary.
    kv_cache_group_id: int
    logical_token_page: int
    layer_id: int = -1
    kv_head_group: int = -1


@dataclass
class ResidencyRecord:
    tier: VirtualTier
    state: VirtualState
    version: int
    gpu_block_id: int | None = None
    cpu_block_id: int | None = None
    page_size_tokens: int = 0
    block_size_tokens: int = 0
    dram_copy_version: int | None = None
    # Number of leading tokens covered by ``dram_copy_version``.  A growing
    # tail can retain an older provider version while its HBM page expands;
    # that copy is not a release candidate until it covers the current page.
    dram_copy_tokens: int = 0
    released_gpu_block_id: int | None = None


@dataclass(frozen=True)
class HBMReleaseCandidate:
    request_id: str
    kv_cache_group_id: int
    logical_token_page: int
    gpu_block_id: int
    dram_copy_version: int
    page_size_tokens: int
    layer_id: int = -1
    kv_head_group: int = -1


@dataclass(frozen=True)
class HBMReleaseReadiness:
    """Logical projection for coverage checks, never a physical release plan."""

    request_id: str
    kv_cache_group_id: int
    logical_token_page: int
    page_size_tokens: int
    candidate_records: int


@dataclass
class _ReleaseReadinessScan:
    """No tensors, physical addresses or authority; lock-owned CPU projection."""

    keys: tuple[tuple[str, int, int], ...]
    values: dict[tuple[str, int, int], HBMReleaseReadiness | None]
    cursor: int = 0


@dataclass(frozen=True)
class HBMReloadPage:
    request_id: str
    kv_cache_group_id: int
    logical_token_page: int
    provider_version: int
    previous_gpu_block_id: int
    page_size_tokens: int
    layer_id: int = -1
    kv_head_group: int = -1


@dataclass(frozen=True)
class AttentionPlanSummary:
    phase: AttentionPhase
    request_count: int
    query_tokens: int
    required_pages: int
    hbm_resident_pages: int
    dram_resident_pages: int
    staging_pages: int
    pages_to_materialize: int
    staging_budget_blocks: int
    dma_budget_bytes: int
    tbt_budget_ms: float | None
    feasible_with_current_staging_budget: bool
    ring_path_requested: bool = False
    ring_path_active: bool = False
    ring_traversal_pages: int = 0
    ring_traversal_segments: int = 0
    ring_hbm_traversal_pages: int = 0
    ring_dram_traversal_pages: int = 0
    ring_staging_traversal_pages: int = 0
    ring_pages_to_materialize: int = 0
    ring_online_softmax_merge_required: bool = False
    ring_executor_available: bool = False
    ring_executor_enabled: bool = False
    ring_executor_implemented: bool = False
    ring_executor_attached_to_vllm_attention: bool = False
    ring_backend_needed_by_residency: bool = False
    ring_backend_force_all: bool = False
    ring_backend_will_attempt: bool = False
    ring_decode_slo_policy_applied: bool = False

    def to_debug_fields(self, prefix: str) -> dict[str, int | float | str]:
        tbt_budget_ms = self.tbt_budget_ms if self.tbt_budget_ms is not None else 0.0
        return {
            f"{prefix}_phase": self.phase,
            f"{prefix}_request_count": self.request_count,
            f"{prefix}_query_tokens": self.query_tokens,
            f"{prefix}_required_pages": self.required_pages,
            f"{prefix}_hbm_resident_pages": self.hbm_resident_pages,
            f"{prefix}_dram_resident_pages": self.dram_resident_pages,
            f"{prefix}_staging_pages": self.staging_pages,
            f"{prefix}_pages_to_materialize": self.pages_to_materialize,
            f"{prefix}_staging_budget_blocks": self.staging_budget_blocks,
            f"{prefix}_dma_budget_bytes": self.dma_budget_bytes,
            f"{prefix}_tbt_budget_ms": tbt_budget_ms,
            f"{prefix}_feasible_with_current_staging_budget": int(
                self.feasible_with_current_staging_budget
            ),
            f"{prefix}_ring_path_requested": int(self.ring_path_requested),
            f"{prefix}_ring_path_active": int(self.ring_path_active),
            f"{prefix}_ring_traversal_pages": self.ring_traversal_pages,
            f"{prefix}_ring_traversal_segments": self.ring_traversal_segments,
            f"{prefix}_ring_hbm_traversal_pages": self.ring_hbm_traversal_pages,
            f"{prefix}_ring_dram_traversal_pages": self.ring_dram_traversal_pages,
            f"{prefix}_ring_staging_traversal_pages": (
                self.ring_staging_traversal_pages
            ),
            f"{prefix}_ring_pages_to_materialize": (
                self.ring_pages_to_materialize
            ),
            f"{prefix}_ring_online_softmax_merge_required": int(
                self.ring_online_softmax_merge_required
            ),
            f"{prefix}_ring_executor_available": int(
                self.ring_executor_available
            ),
            f"{prefix}_ring_executor_enabled": int(self.ring_executor_enabled),
            f"{prefix}_ring_executor_implemented": int(
                self.ring_executor_implemented
            ),
            f"{prefix}_ring_executor_attached_to_vllm_attention": int(
                self.ring_executor_attached_to_vllm_attention
            ),
            f"{prefix}_ring_backend_needed_by_residency": int(
                self.ring_backend_needed_by_residency
            ),
            f"{prefix}_ring_backend_force_all": int(
                self.ring_backend_force_all
            ),
            f"{prefix}_ring_backend_will_attempt": int(
                self.ring_backend_will_attempt
            ),
            f"{prefix}_ring_decode_slo_policy_applied": int(
                self.ring_decode_slo_policy_applied
            ),
        }


@dataclass(frozen=True)
class ResidencyBudgetSnapshot:
    free_hbm_blocks: int
    decode_hotset_reserved_blocks: int
    decode_hotset_used_blocks: int
    prefill_required_growth_blocks: int
    staging_reserved_blocks: int
    available_after_prefill_growth_blocks: int
    prefill_admission_feasible_with_reserve: bool
    serialized_by_residency_budget: bool

    def to_debug_fields(self) -> dict[str, int]:
        return {
            "decode_hotset_reserved_blocks": self.decode_hotset_reserved_blocks,
            "decode_hotset_used_blocks": self.decode_hotset_used_blocks,
            "prefill_required_growth_blocks": (
                self.prefill_required_growth_blocks
            ),
            "staging_reserved_blocks": self.staging_reserved_blocks,
            "residency_free_hbm_blocks": self.free_hbm_blocks,
            "residency_available_after_prefill_growth_blocks": (
                self.available_after_prefill_growth_blocks
            ),
            "prefill_admission_feasible_with_reserve": int(
                self.prefill_admission_feasible_with_reserve
            ),
            "pd_serialized_by_residency_budget": int(
                self.serialized_by_residency_budget
            ),
        }


@dataclass(frozen=True)
class ReloadGuardSummary:
    phase: AttentionPhase
    request_count: int
    reload_required_pages: int
    reload_required_tokens: int
    reload_required_request_count: int
    tiered_backend_enabled: bool
    dram_provider_enabled: bool
    reload_backend_available: bool

    @property
    def satisfied_by_tiered_provider(self) -> bool:
        return self.tiered_backend_enabled and self.dram_provider_enabled

    @property
    def violation(self) -> bool:
        if self.reload_required_pages == 0:
            return False
        return not (
            self.satisfied_by_tiered_provider or self.reload_backend_available
        )

    def to_debug_fields(self, prefix: str) -> dict[str, int | str]:
        return {
            f"{prefix}_phase": self.phase,
            f"{prefix}_request_count": self.request_count,
            f"{prefix}_reload_required_pages": self.reload_required_pages,
            f"{prefix}_reload_required_tokens": self.reload_required_tokens,
            f"{prefix}_reload_required_request_count": (
                self.reload_required_request_count
            ),
            f"{prefix}_tiered_backend_enabled": int(
                self.tiered_backend_enabled
            ),
            f"{prefix}_dram_provider_enabled": int(
                self.dram_provider_enabled
            ),
            f"{prefix}_reload_backend_available": int(
                self.reload_backend_available
            ),
            f"{prefix}_satisfied_by_tiered_provider": int(
                self.satisfied_by_tiered_provider
            ),
            f"{prefix}_violation": int(self.violation),
        }


class VirtualContextTable:
    """A default-off shadow table for logical KV page residency."""

    def __init__(self, *, cache_release_readiness: bool = False) -> None:
        self._lock = Lock()
        self._cache_release_readiness = cache_release_readiness
        # Host-only exact projections. Any coverage mutation drops the entire
        # affected request, including an unfinished scan and negative results.
        self._release_readiness_scans: dict[str, _ReleaseReadinessScan] = {}
        self._logical_page_refcounts: dict[
            str, dict[tuple[str, int, int], int]
        ] = {}
        self._readiness_cache_hits = 0
        self._readiness_evaluated_pages = 0
        self._readiness_invalidations = 0
        self._readiness_warm_yields = 0
        self._pages: dict[LogicalPageId, ResidencyRecord] = {}
        # Ordered IDs only: no tensor, provider lease or second record owner.
        # Include layer-only entries as well as aggregate pages. Retire/publish
        # may temporarily have no aggregate, so that index alone is insufficient.
        self._page_ids_by_request: defaultdict[
            str, dict[LogicalPageId, None]
        ] = defaultdict(dict)
        # Attention asks the same request once per layer. Avoid scanning all
        # L*P records L times; store identity only, never cached readiness.
        self._page_ids_by_request_layer: defaultdict[
            tuple[str, int], dict[LogicalPageId, None]
        ] = defaultdict(dict)
        self._aggregate_pages_by_request: defaultdict[
            str, set[LogicalPageId]
        ] = defaultdict(set)
        self._layer_pages_by_aggregate: defaultdict[
            tuple[str, int, int], set[LogicalPageId]
        ] = defaultdict(set)
        self._reload_required_pages_by_request: defaultdict[
            str, set[LogicalPageId]
        ] = defaultdict(set)
        self._request_tokens: dict[str, int] = {}
        self._request_versions: defaultdict[str, int] = defaultdict(int)
        self._provider_layers_by_request: defaultdict[str, set[int]] = (
            defaultdict(set)
        )
        self._observe_hbm_calls = 0
        self._observe_hbm_delta_calls = 0
        self._observe_hbm_delta_pages = 0
        self._observe_hbm_full_reconcile_pages = 0
        self._observe_dram_load_calls = 0
        self._observe_dram_store_calls = 0
        self._dram_copy_ready_calls = 0
        self._dram_copy_short_rejected_pages = 0
        self._hbm_release_mark_calls = 0
        self._hbm_release_marked_pages = 0
        self._hbm_partial_release_marked_pages = 0
        self._hbm_reload_complete_calls = 0
        self._hbm_reload_completed_pages = 0
        self._hbm_reload_aggregate_completed_pages = 0
        self._released_requests = 0

    def _next_version(self, request_id: str) -> int:
        self._request_versions[request_id] += 1
        return self._request_versions[request_id]

    @staticmethod
    def _has_hbm_release_prereqs(record: ResidencyRecord) -> bool:
        return (
            record.tier == "HBM"
            and record.state == "ready"
            and record.gpu_block_id is not None
            and record.dram_copy_version is not None
            and record.dram_copy_tokens >= record.page_size_tokens
        )

    def observe_provider_layer(self, request_id: str, layer_id: int) -> None:
        """Record a layer that must have a provider copy before HBM release."""
        if not request_id or layer_id < 0:
            return
        with self._lock:
            if layer_id not in self._provider_layers_by_request[request_id]:
                self._invalidate_release_readiness_locked(request_id)
            self._provider_layers_by_request[request_id].add(layer_id)

    @staticmethod
    def _is_closed_full_page(record: ResidencyRecord) -> bool:
        return (
            record.block_size_tokens <= 0
            or record.page_size_tokens >= record.block_size_tokens
        )

    @classmethod
    def _is_hbm_release_candidate(
        cls,
        record: ResidencyRecord,
        *,
        allow_partial_pages: bool = False,
    ) -> bool:
        return cls._has_hbm_release_prereqs(record) and (
            allow_partial_pages or cls._is_closed_full_page(record)
        )

    @classmethod
    def _is_partial_tail_release_blocked(
        cls,
        record: ResidencyRecord,
    ) -> bool:
        return cls._has_hbm_release_prereqs(
            record
        ) and not cls._is_closed_full_page(record)

    @staticmethod
    def _is_hbm_reload_required(record: ResidencyRecord) -> bool:
        return (
            record.tier == "DRAM"
            and record.state == "ready"
            and record.released_gpu_block_id is not None
            and record.dram_copy_version is not None
        )

    @staticmethod
    def _aggregate_page_key(page_id: LogicalPageId) -> tuple[str, int, int]:
        return (
            page_id.request_id,
            page_id.kv_cache_group_id,
            page_id.logical_token_page,
        )

    @staticmethod
    def _is_aggregate_page(page_id: LogicalPageId) -> bool:
        return page_id.layer_id == -1 and page_id.kv_head_group == -1

    def _deindex_page_locked(
        self,
        page_id: LogicalPageId,
        record: ResidencyRecord,
    ) -> None:
        if self._is_aggregate_page(page_id):
            pages = self._aggregate_pages_by_request.get(page_id.request_id)
            if pages is not None:
                pages.discard(page_id)
                if not pages:
                    self._aggregate_pages_by_request.pop(page_id.request_id, None)
        else:
            key = self._aggregate_page_key(page_id)
            pages = self._layer_pages_by_aggregate.get(key)
            if pages is not None:
                pages.discard(page_id)
                if not pages:
                    self._layer_pages_by_aggregate.pop(key, None)
        if self._is_hbm_reload_required(record):
            pages = self._reload_required_pages_by_request.get(
                page_id.request_id
            )
            if pages is not None:
                pages.discard(page_id)
                if not pages:
                    self._reload_required_pages_by_request.pop(
                        page_id.request_id, None
                    )

    def _index_page_locked(
        self,
        page_id: LogicalPageId,
        record: ResidencyRecord,
    ) -> None:
        if self._is_aggregate_page(page_id):
            self._aggregate_pages_by_request[page_id.request_id].add(page_id)
        else:
            self._layer_pages_by_aggregate[
                self._aggregate_page_key(page_id)
            ].add(page_id)
        if self._is_hbm_reload_required(record):
            self._reload_required_pages_by_request[page_id.request_id].add(
                page_id
            )

    def _set_page_locked(
        self,
        page_id: LogicalPageId,
        record: ResidencyRecord,
    ) -> None:
        old = self._pages.get(page_id)
        self._invalidate_release_readiness_locked(page_id.request_id)
        if self._cache_release_readiness and old is None:
            keys = self._logical_page_refcounts.setdefault(page_id.request_id, {})
            key = self._aggregate_page_key(page_id)
            keys[key] = keys.get(key, 0) + 1
        if old is not None:
            self._deindex_page_locked(page_id, old)
        self._pages[page_id] = record
        self._page_ids_by_request[page_id.request_id][page_id] = None
        self._page_ids_by_request_layer[
            (page_id.request_id, page_id.layer_id)
        ][page_id] = None
        self._index_page_locked(page_id, record)

    def _delete_page_locked(self, page_id: LogicalPageId) -> None:
        self._invalidate_release_readiness_locked(page_id.request_id)
        old = self._pages.pop(page_id, None)
        if old is not None:
            if self._cache_release_readiness:
                keys = self._logical_page_refcounts[page_id.request_id]
                key = self._aggregate_page_key(page_id)
                keys[key] -= 1
                if not keys[key]:
                    del keys[key]
                if not keys:
                    del self._logical_page_refcounts[page_id.request_id]
            page_ids = self._page_ids_by_request[page_id.request_id]
            del page_ids[page_id]
            if not page_ids:
                del self._page_ids_by_request[page_id.request_id]
            layer_key = (page_id.request_id, page_id.layer_id)
            layer_ids = self._page_ids_by_request_layer[layer_key]
            del layer_ids[page_id]
            if not layer_ids:
                del self._page_ids_by_request_layer[layer_key]
            self._deindex_page_locked(page_id, old)

    def _request_page_records_locked(
        self, request_ids: set[str] | None,
    ) -> Iterable[tuple[LogicalPageId, ResidencyRecord]]:
        """Iterate selected owners only; caller must hold the table lock.

        Keep per-request insertion order for version/debug consumers; release
        planning applies its existing deterministic sort afterwards. None means
        a global query, while an empty set means no requests.
        """
        if request_ids is None:
            return self._pages.items()
        return (
            (page_id, self._pages[page_id])
            for request_id in sorted(request_ids)
            for page_id in self._page_ids_by_request.get(request_id, ())
        )

    def _refresh_reload_index_locked(self, page_id: LogicalPageId) -> None:
        self._invalidate_release_readiness_locked(page_id.request_id)
        record = self._pages.get(page_id)
        pages = self._reload_required_pages_by_request.get(page_id.request_id)
        if pages is not None:
            pages.discard(page_id)
            if not pages:
                self._reload_required_pages_by_request.pop(
                    page_id.request_id, None
                )
        if record is not None and self._is_hbm_reload_required(record):
            self._reload_required_pages_by_request[page_id.request_id].add(
                page_id
            )

    def _update_layer_page_lengths_locked(
        self,
        page_id: LogicalPageId,
        page_size_tokens: int,
        block_size: int,
    ) -> None:
        self._invalidate_release_readiness_locked(page_id.request_id)
        for candidate_id in tuple(
            self._layer_pages_by_aggregate.get(
                self._aggregate_page_key(page_id), ()
            )
        ):
            candidate_record = self._pages.get(candidate_id)
            if candidate_record is None:
                continue
            candidate_record.page_size_tokens = max(
                candidate_record.page_size_tokens,
                page_size_tokens,
            )
            candidate_record.block_size_tokens = (
                block_size or candidate_record.block_size_tokens
            )

    def _observe_hbm_page_locked(
        self,
        *,
        request_id: str,
        group_id: int,
        page_idx: int,
        block_id: int,
        block_size: int,
        logical_tokens: int,
        version: int,
    ) -> None:
        token_start = page_idx * block_size
        page_size_tokens = min(
            block_size,
            max(0, logical_tokens - token_start),
        )
        if page_size_tokens <= 0:
            return
        page_id = LogicalPageId(
            request_id=request_id,
            kv_cache_group_id=group_id,
            logical_token_page=page_idx,
        )
        old = self._pages.get(page_id)
        if old is not None and self._is_hbm_reload_required(old):
            old.page_size_tokens = page_size_tokens
            old.block_size_tokens = block_size or old.block_size_tokens
        else:
            self._set_page_locked(
                page_id,
                ResidencyRecord(
                    tier="HBM",
                    state="ready",
                    version=version,
                    gpu_block_id=block_id,
                    cpu_block_id=old.cpu_block_id if old else None,
                    page_size_tokens=page_size_tokens,
                    block_size_tokens=block_size,
                    dram_copy_version=(
                        old.dram_copy_version if old else None
                    ),
                    dram_copy_tokens=(old.dram_copy_tokens if old else 0),
                ),
            )
        self._update_layer_page_lengths_locked(
            page_id,
            page_size_tokens,
            block_size,
        )

    def observe_hbm_blocks(
        self,
        request_id: str,
        block_ids_by_group: tuple[list[int], ...],
        block_size: int,
        logical_tokens: int,
        null_block_id: int | None = None,
    ) -> None:
        """Record the currently known GPU KV block table for a request."""
        with self._lock:
            version = self._next_version(request_id)
            self._observe_hbm_calls += 1
            self._request_tokens[request_id] = max(
                self._request_tokens.get(request_id, 0), logical_tokens
            )

            observed_pages: set[LogicalPageId] = set()
            for group_id, block_ids in enumerate(block_ids_by_group):
                for page_idx, block_id in enumerate(block_ids):
                    if block_id is None or block_id < 0:
                        continue
                    if null_block_id is not None and block_id == null_block_id:
                        continue
                    token_start = page_idx * block_size
                    page_size_tokens = min(
                        block_size,
                        max(0, logical_tokens - token_start),
                    )
                    if page_size_tokens <= 0:
                        continue
                    page_id = LogicalPageId(
                        request_id=request_id,
                        kv_cache_group_id=group_id,
                        logical_token_page=page_idx,
                    )
                    observed_pages.add(page_id)
                    self._observe_hbm_page_locked(
                        request_id=request_id,
                        group_id=group_id,
                        page_idx=page_idx,
                        block_id=block_id,
                        block_size=block_size,
                        logical_tokens=logical_tokens,
                        version=version,
                    )
                    self._observe_hbm_full_reconcile_pages += 1

            # HBM block tables are authoritative for currently allocated GPU
            # pages. Drop stale HBM/STAGING records that disappeared from the
            # request block table while preserving DRAM records.
            for page_id in tuple(
                self._aggregate_pages_by_request.get(request_id, ())
            ):
                record = self._pages.get(page_id)
                if record is None:
                    continue
                if record.tier == "DRAM":
                    continue
                if page_id not in observed_pages:
                    self._delete_page_locked(page_id)

    def observe_hbm_block_delta(
        self,
        request_id: str,
        new_block_ids_by_group: tuple[list[int], ...],
        new_page_start_by_group: tuple[int, ...],
        block_size: int,
        logical_tokens: int,
        null_block_id: int | None = None,
    ) -> None:
        """Apply monotonic HBM allocation and growing-tail changes only."""
        with self._lock:
            previous_tokens = self._request_tokens.get(request_id, 0)
            version = self._next_version(request_id)
            self._observe_hbm_delta_calls += 1
            self._request_tokens[request_id] = max(
                previous_tokens, logical_tokens
            )
            touched: set[tuple[int, int]] = set()
            for group_id, block_ids in enumerate(new_block_ids_by_group):
                start_page = new_page_start_by_group[group_id]
                for offset, block_id in enumerate(block_ids):
                    if block_id is None or block_id < 0:
                        continue
                    if null_block_id is not None and block_id == null_block_id:
                        continue
                    page_idx = start_page + offset
                    self._observe_hbm_page_locked(
                        request_id=request_id,
                        group_id=group_id,
                        page_idx=page_idx,
                        block_id=block_id,
                        block_size=block_size,
                        logical_tokens=logical_tokens,
                        version=version,
                    )
                    touched.add((group_id, page_idx))

            tail_pages: set[int] = set()
            if previous_tokens > 0:
                tail_pages.add((previous_tokens - 1) // block_size)
            if logical_tokens > 0:
                tail_pages.add((logical_tokens - 1) // block_size)
            for group_id in range(len(new_block_ids_by_group)):
                for page_idx in tail_pages:
                    if (group_id, page_idx) in touched:
                        continue
                    page_id = LogicalPageId(
                        request_id=request_id,
                        kv_cache_group_id=group_id,
                        logical_token_page=page_idx,
                    )
                    record = self._pages.get(page_id)
                    if record is None or record.gpu_block_id is None:
                        continue
                    self._observe_hbm_page_locked(
                        request_id=request_id,
                        group_id=group_id,
                        page_idx=page_idx,
                        block_id=record.gpu_block_id,
                        block_size=block_size,
                        logical_tokens=logical_tokens,
                        version=version,
                    )
                    touched.add((group_id, page_idx))
            self._observe_hbm_delta_pages += len(touched)

    def observe_dram_pages(
        self,
        request_id: str,
        gpu_page_indices: set[int],
        block_size: int,
        state: VirtualState,
        *,
        group_id: int = 0,
    ) -> None:
        """Record logical pages involved in CPU/DRAM KV load/store.

        The current vLLM offloading connector stores cross-layer blocks as a
        single group, so this starts at group 0. The logical-page key is already
        shaped for future per-layer widening.
        """
        if not gpu_page_indices:
            return
        with self._lock:
            version = self._next_version(request_id)
            if state == "loading":
                self._observe_dram_load_calls += 1
            elif state == "storing":
                self._observe_dram_store_calls += 1
            logical_tokens = (max(gpu_page_indices) + 1) * block_size
            self._request_tokens[request_id] = max(
                self._request_tokens.get(request_id, 0), logical_tokens
            )
            for page_idx in gpu_page_indices:
                page_id = LogicalPageId(
                    request_id=request_id,
                    kv_cache_group_id=group_id,
                    logical_token_page=page_idx,
                )
                old = self._pages.get(page_id)
                self._set_page_locked(
                    page_id,
                    ResidencyRecord(
                        tier="DRAM",
                        state=state,
                        version=version,
                        gpu_block_id=old.gpu_block_id if old else None,
                        cpu_block_id=old.cpu_block_id if old else None,
                        page_size_tokens=block_size,
                        block_size_tokens=block_size,
                        dram_copy_version=(
                            version
                            if state == "storing"
                            else (old.dram_copy_version if old else None)
                        ),
                        dram_copy_tokens=(
                            block_size
                            if state == "storing"
                            else (old.dram_copy_tokens if old else 0)
                        ),
                    ),
                )

    def mark_pages_ready(
        self,
        request_id: str,
        gpu_page_indices: set[int],
        tier: VirtualTier,
        block_size: int,
        *,
        group_id: int = 0,
    ) -> None:
        if not gpu_page_indices:
            return
        with self._lock:
            version = self._next_version(request_id)
            for page_idx in gpu_page_indices:
                page_id = LogicalPageId(
                    request_id=request_id,
                    kv_cache_group_id=group_id,
                    logical_token_page=page_idx,
                )
                old = self._pages.get(page_id)
                self._set_page_locked(
                    page_id,
                    ResidencyRecord(
                        tier=tier,
                        state="ready",
                        version=version,
                        gpu_block_id=old.gpu_block_id if old else None,
                        cpu_block_id=old.cpu_block_id if old else None,
                        page_size_tokens=block_size,
                        block_size_tokens=block_size,
                        dram_copy_version=(
                            version
                            if tier == "DRAM"
                            else (old.dram_copy_version if old else None)
                        ),
                        dram_copy_tokens=(
                            block_size
                            if tier == "DRAM"
                            else (old.dram_copy_tokens if old else 0)
                        ),
                    ),
                )

    def record_dram_copies_ready(
        self,
        request_id: str,
        page_versions: dict[int, int],
        block_size: int,
        *,
        group_id: int = 0,
        layer_id: int = -1,
        kv_head_group: int = -1,
        page_to_block: dict[int, int] | None = None,
        page_token_counts: dict[int, int] | None = None,
        require_existing_request: bool = False,
    ) -> None:
        """Record that DRAM/provider copies exist without releasing HBM.

        This is a dual-residency marker. The authoritative tier remains HBM
        until a later policy explicitly releases the HBM block.
        """
        if not page_versions:
            return
        with self._lock:
            if require_existing_request and request_id not in self._request_versions:
                # A required queued snapshot may outlive request abort. Its
                # source lease still drains normally, but completion must not
                # recreate logical HBM ownership after release_request().
                return
            self._invalidate_release_readiness_locked(request_id)
            self._dram_copy_ready_calls += 1
            logical_tokens = max(
                page_idx * block_size
                + (
                    page_token_counts.get(page_idx, block_size)
                    if page_token_counts is not None
                    else block_size
                )
                for page_idx in page_versions
            )
            self._request_tokens[request_id] = max(
                self._request_tokens.get(request_id, 0), logical_tokens
            )
            for page_idx, copy_version in page_versions.items():
                page_size_tokens = (
                    page_token_counts.get(page_idx, block_size)
                    if page_token_counts is not None
                    else block_size
                )
                page_id = LogicalPageId(
                    request_id=request_id,
                    kv_cache_group_id=group_id,
                    logical_token_page=page_idx,
                    layer_id=layer_id,
                    kv_head_group=kv_head_group,
                )
                old = self._pages.get(page_id)
                aggregate = self._pages.get(
                    LogicalPageId(
                        request_id=request_id,
                        kv_cache_group_id=group_id,
                        logical_token_page=page_idx,
                    )
                )
                expected_page_size_tokens = max(
                    page_size_tokens,
                    old.page_size_tokens if old is not None else 0,
                    aggregate.page_size_tokens
                    if aggregate is not None
                    else 0,
                )
                if page_size_tokens < expected_page_size_tokens:
                    # An async copy may have captured the tail before a later
                    # scheduler step appended more tokens. Never shrink the
                    # layer record to that stale snapshot: doing so would make
                    # the page look releasable even though the provider copy
                    # does not cover the current logical tail.
                    self._dram_copy_short_rejected_pages += 1
                    continue
                block_id = (
                    page_to_block.get(page_idx)
                    if page_to_block is not None
                    else None
                )
                if old is None:
                    self._set_page_locked(
                        page_id,
                        ResidencyRecord(
                            tier="HBM" if block_id is not None else "DRAM",
                            state="ready",
                            version=copy_version,
                            gpu_block_id=block_id,
                            page_size_tokens=page_size_tokens,
                            block_size_tokens=block_size,
                            dram_copy_version=copy_version,
                            dram_copy_tokens=page_size_tokens,
                        ),
                    )
                    continue
                if (
                    old.dram_copy_version is not None
                    and (
                        copy_version < old.dram_copy_version
                        or (
                            copy_version == old.dram_copy_version
                            and page_size_tokens < old.dram_copy_tokens
                        )
                    )
                ):
                    # Async registration workers may complete out of order.
                    # Never let an older or shorter growing-tail snapshot
                    # replace a copy that already covers more logical tokens.
                    continue
                if block_id is not None and not self._is_hbm_reload_required(old):
                    old.gpu_block_id = block_id
                old.dram_copy_version = copy_version
                old.dram_copy_tokens = page_size_tokens
                old.page_size_tokens = page_size_tokens or old.page_size_tokens
                old.block_size_tokens = block_size or old.block_size_tokens

    def mark_hbm_reload_complete(
        self,
        request_id: str,
        page_to_block: dict[int, int],
        block_size: int,
        *,
        group_id: int = 0,
        layer_id: int = -1,
        kv_head_group: int = -1,
        page_token_counts: dict[int, int] | None = None,
        provider_token_counts: dict[int, int] | None = None,
        provider_versions: dict[int, int] | None = None,
    ) -> int:
        """Commit provider-backed reloads after DRAM K/V was copied to HBM.

        Scheduler-side allocation can install HBM block ids before the worker
        has copied the authoritative DRAM pages. This method is the handoff
        boundary: only the worker/provider calls it after successful copies.
        """
        if not page_to_block:
            return 0
        completed = 0
        with self._lock:
            self._invalidate_release_readiness_locked(request_id)
            self._hbm_reload_complete_calls += 1
            logical_tokens = max(
                page_idx * block_size
                + (
                    page_token_counts.get(page_idx, block_size)
                    if page_token_counts is not None
                    else block_size
                )
                for page_idx in page_to_block
            )
            self._request_tokens[request_id] = max(
                self._request_tokens.get(request_id, 0), logical_tokens
            )
            for page_idx, block_id in page_to_block.items():
                if block_id is None or block_id < 0:
                    continue
                provider_version = (
                    provider_versions.get(page_idx)
                    if provider_versions is not None
                    else None
                )
                page_id = LogicalPageId(
                    request_id=request_id,
                    kv_cache_group_id=group_id,
                    logical_token_page=page_idx,
                    layer_id=layer_id,
                    kv_head_group=kv_head_group,
                )
                old = self._pages.get(page_id)
                page_size_tokens = (
                    page_token_counts.get(page_idx, block_size)
                    if page_token_counts is not None
                    else block_size
                )
                provider_token_count = (
                    provider_token_counts.get(page_idx, page_size_tokens)
                    if provider_token_counts is not None
                    else page_size_tokens
                )
                if not 0 <= provider_token_count <= page_size_tokens:
                    raise ValueError(
                        "provider coverage must be within the reloaded HBM "
                        "logical page: "
                        f"request={request_id!r} page={page_idx} "
                        f"provider_tokens={provider_token_count} "
                        f"page_tokens={page_size_tokens}"
                    )
                if provider_version is not None and provider_token_count <= 0:
                    raise ValueError(
                        "provider-backed HBM reload must cover at least one "
                        f"token: request={request_id!r} page={page_idx}"
                    )
                if old is None:
                    self._set_page_locked(
                        page_id,
                        ResidencyRecord(
                            tier="HBM",
                            state="ready",
                            version=self._next_version(request_id),
                            gpu_block_id=block_id,
                            page_size_tokens=page_size_tokens,
                            block_size_tokens=block_size,
                            dram_copy_version=provider_version,
                            dram_copy_tokens=(
                                provider_token_count
                                if provider_version is not None
                                else 0
                            ),
                        ),
                    )
                    completed += 1
                    continue
                if not self._is_hbm_reload_required(old):
                    continue
                old.tier = "HBM"
                old.state = "ready"
                old.gpu_block_id = block_id
                old.released_gpu_block_id = None
                old.page_size_tokens = page_size_tokens or old.page_size_tokens
                old.block_size_tokens = block_size or old.block_size_tokens
                if provider_version is not None and (
                    old.dram_copy_version is None
                    or provider_version >= old.dram_copy_version
                ):
                    # The exact provider version pinned when the scheduler
                    # released HBM may have been superseded by a later async
                    # snapshot before this H2D reload. Keep the dual-residency
                    # marker aligned with the version that was actually copied
                    # so a later pressure release does not pin an evicted
                    # provider version again.
                    old.dram_copy_version = provider_version
                    old.dram_copy_tokens = max(
                        old.dram_copy_tokens,
                        provider_token_count,
                    )
                # Keep dram_copy_version as a valid dual-residency copy marker;
                # the next pressure release can reuse the same provider version.
                completed += 1
                self._refresh_reload_index_locked(page_id)

                # Scheduler allocation is group/page based while provider
                # copies are layer based. Commit the aggregate page only after
                # every layer-specific copy for this logical page has left the
                # DRAM-authoritative state. Until then, the next scheduler step
                # must continue to carry reload metadata.
                if layer_id != -1:
                    layer_reload_pending = any(
                        self._is_hbm_reload_required(self._pages[candidate_id])
                        for candidate_id in self._layer_pages_by_aggregate.get(
                            (request_id, group_id, page_idx), ()
                        )
                        if candidate_id in self._pages
                    )
                    aggregate_id = LogicalPageId(
                        request_id=request_id,
                        kv_cache_group_id=group_id,
                        logical_token_page=page_idx,
                    )
                    aggregate = self._pages.get(aggregate_id)
                    if (
                        not layer_reload_pending
                        and aggregate is not None
                        and self._is_hbm_reload_required(aggregate)
                    ):
                        aggregate.tier = "HBM"
                        aggregate.state = "ready"
                        aggregate.gpu_block_id = block_id
                        aggregate.released_gpu_block_id = None
                        aggregate.page_size_tokens = (
                            page_size_tokens or aggregate.page_size_tokens
                        )
                        aggregate.block_size_tokens = (
                            block_size or aggregate.block_size_tokens
                        )
                        completed_layer_records = [
                            self._pages[candidate_id]
                            for candidate_id in self._layer_pages_by_aggregate.get(
                                (request_id, group_id, page_idx), ()
                            )
                            if candidate_id in self._pages
                        ]
                        completed_provider_versions = [
                            record.dram_copy_version
                            for record in completed_layer_records
                            if record.dram_copy_version is not None
                        ]
                        if completed_provider_versions:
                            aggregate.dram_copy_version = max(
                                completed_provider_versions
                            )
                            # Aggregate eligibility is only as complete as the
                            # shortest layer snapshot.  HBM may already contain
                            # the newly appended query tokens while DRAM still
                            # owns only the historical prefix.
                            completed_provider_coverages = [
                                record.dram_copy_tokens
                                for record in completed_layer_records
                                if record.dram_copy_version is not None
                            ]
                            aggregate.dram_copy_tokens = max(
                                aggregate.dram_copy_tokens,
                                min(completed_provider_coverages),
                            )
                        self._hbm_reload_aggregate_completed_pages += 1
                        self._refresh_reload_index_locked(aggregate_id)
            self._hbm_reload_completed_pages += completed
        return completed

    @staticmethod
    def _candidate_sort_key(
        item: tuple[LogicalPageId, ResidencyRecord],
    ) -> tuple[str, int, int, int, int]:
        page_id, _ = item
        return (
            page_id.request_id,
            page_id.kv_cache_group_id,
            page_id.layer_id,
            page_id.kv_head_group,
            page_id.logical_token_page,
        )

    def _collect_hbm_release_candidates_locked(
        self,
        request_ids: set[str] | None,
        max_pages: int,
        allow_partial_pages: bool,
        max_logical_pages: int = 0,
    ) -> list[tuple[LogicalPageId, ResidencyRecord, HBMReleaseCandidate]]:
        candidates: list[
            tuple[LogicalPageId, ResidencyRecord, HBMReleaseCandidate]
        ] = []
        records = sorted(
            self._request_page_records_locked(request_ids),
            key=self._candidate_sort_key,
        )
        eligible_layer_pages: set[tuple[str, int, int]] = set()
        aggregate_page_sizes: dict[tuple[str, int, int], int] = {}
        layer_records_by_page: defaultdict[
            tuple[str, int, int], dict[int, list[ResidencyRecord]]
        ] = defaultdict(lambda: defaultdict(list))
        for page_id, record in records:
            page_key = (
                page_id.request_id,
                page_id.kv_cache_group_id,
                page_id.logical_token_page,
            )
            if page_id.layer_id < 0:
                aggregate_page_sizes[page_key] = max(
                    aggregate_page_sizes.get(page_key, 0),
                    record.page_size_tokens,
                )
                continue
            layer_records_by_page[page_key][page_id.layer_id].append(record)
        for page_key, records_by_layer in layer_records_by_page.items():
            request_id = page_key[0]
            aggregate_page_size = aggregate_page_sizes.get(page_key, 0)
            expected_layers = self._provider_layers_by_request.get(
                request_id, set()
            )
            if expected_layers and not expected_layers.issubset(records_by_layer):
                continue
            if all(
                all(
                    self._is_hbm_release_candidate(
                        record,
                        allow_partial_pages=allow_partial_pages,
                    )
                    and record.dram_copy_tokens >= aggregate_page_size
                    for record in layer_records
                )
                for layer_records in records_by_layer.values()
            ):
                eligible_layer_pages.add(page_key)

        selected_page_keys: set[tuple[str, int, int]] | None = None
        if max_logical_pages > 0:
            selected_page_keys = set(
                sorted(
                    page_key
                    for page_key in eligible_layer_pages
                    if request_ids is None or page_key[0] in request_ids
                )[:max_logical_pages]
            )

        for page_id, record in records:
            if request_ids is not None and page_id.request_id not in request_ids:
                continue
            page_key = (
                page_id.request_id,
                page_id.kv_cache_group_id,
                page_id.logical_token_page,
            )
            if (
                selected_page_keys is not None
                and page_key not in selected_page_keys
            ):
                continue
            expected_layers = self._provider_layers_by_request.get(
                page_id.request_id, set()
            )
            if expected_layers and page_key not in eligible_layer_pages:
                continue
            if not self._is_hbm_release_candidate(
                record,
                allow_partial_pages=allow_partial_pages,
            ):
                continue
            assert record.gpu_block_id is not None
            assert record.dram_copy_version is not None
            candidate = HBMReleaseCandidate(
                request_id=page_id.request_id,
                kv_cache_group_id=page_id.kv_cache_group_id,
                logical_token_page=page_id.logical_token_page,
                layer_id=page_id.layer_id,
                kv_head_group=page_id.kv_head_group,
                gpu_block_id=record.gpu_block_id,
                dram_copy_version=record.dram_copy_version,
                page_size_tokens=record.page_size_tokens,
            )
            candidates.append((page_id, record, candidate))
            if max_pages > 0 and len(candidates) >= max_pages:
                break
        return candidates

    def _invalidate_release_readiness_locked(self, request_id: str) -> None:
        if self._release_readiness_scans.pop(request_id, None) is not None:
            self._readiness_invalidations += 1

    def _release_readiness_scan_locked(
        self, request_id: str,
    ) -> _ReleaseReadinessScan:
        scan = self._release_readiness_scans.get(request_id)
        if scan is None:
            scan = _ReleaseReadinessScan(
                tuple(self._logical_page_refcounts.get(request_id, ())), {})
            # Do not retain empty/missing request IDs after abort.
            if scan.keys:
                self._release_readiness_scans[request_id] = scan
        return scan

    def _summarize_release_page_locked(
        self, key: tuple[str, int, int], allow_partial_pages: bool,
    ) -> HBMReleaseReadiness | None:
        request_id, group_id, logical_page = key
        aggregate = self._pages.get(
            LogicalPageId(request_id, group_id, logical_page)
        )
        # Includes legacy head-group entries with layer_id < 0.
        records = [
            (page_id, self._pages[page_id])
            for page_id in self._layer_pages_by_aggregate.get(key, ())
        ]
        aggregate_tokens = max(
            (record.page_size_tokens for page_id, record in records
             if page_id.layer_id < 0),
            default=0,
        )
        if aggregate is not None:
            aggregate_tokens = max(aggregate_tokens, aggregate.page_size_tokens)
        expected_layers = self._provider_layers_by_request.get(request_id)
        if expected_layers:
            seen_layers = set()
            for page_id, record in records:
                if page_id.layer_id < 0:
                    continue
                seen_layers.add(page_id.layer_id)
                if not (
                    self._is_hbm_release_candidate(
                        record, allow_partial_pages=allow_partial_pages
                    )
                    and record.dram_copy_tokens >= aggregate_tokens
                ):
                    return None
            if not expected_layers.issubset(seen_layers):
                return None
        count = 0
        max_tokens = 0
        for _, record in records:
            if self._is_hbm_release_candidate(
                record, allow_partial_pages=allow_partial_pages
            ):
                count += 1
                max_tokens = max(max_tokens, record.page_size_tokens)
        if aggregate is not None and self._is_hbm_release_candidate(
            aggregate, allow_partial_pages=allow_partial_pages
        ):
            count += 1
            max_tokens = max(max_tokens, aggregate.page_size_tokens)
        return (
            HBMReleaseReadiness(request_id, group_id, logical_page, max_tokens, count)
            if count else None
        )

    def warm_hbm_release_readiness(
        self, *, request_ids: set[str], max_pages: int = 64,
    ) -> bool:
        """Bound layer-coverage work before admission, without an allocator owner.

        True means the CPU scan is complete, NOT that DRAM copies are ready.
        False means another finite scan slice is needed; it is not a DMA wait.
        Mutators discard the request cursor under the same lock. Final release
        always takes the old full live-validation path.
        """
        if max_pages <= 0:
            raise ValueError("readiness page budget must be positive")
        if not self._cache_release_readiness:
            return True
        with self._lock:
            remaining = max_pages
            for request_id in sorted(request_ids):
                scan = self._release_readiness_scan_locked(request_id)
                while scan.cursor < len(scan.keys):
                    if remaining == 0:
                        self._readiness_warm_yields += 1
                        return False
                    key = scan.keys[scan.cursor]
                    if key not in scan.values:
                        scan.values[key] = self._summarize_release_page_locked(key, True)
                        self._readiness_evaluated_pages += 1
                    scan.cursor += 1
                    remaining -= 1
            return True

    def summarize_hbm_release_readiness(
        self,
        *,
        request_ids: set[str],
        allow_partial_pages: bool = False,
        page_keys: set[tuple[str, int, int]] | None = None,
    ) -> list[HBMReleaseReadiness]:
        """Project an unbounded release query without building layer plans.

        By default every query reads live coverage. Opt-in projections are
        invalidated under the existing lock on growth, copy, abort or reload.
        Explicit keys inspect only a preparation's pages. Results are unordered
        and contain no block IDs or provider versions: callers must still use
        the authoritative release/commit path, which revalidates all records.
        """
        with self._lock:
            if self._cache_release_readiness and allow_partial_pages:
                summaries = []
                for request_id in request_ids:
                    scan = self._release_readiness_scan_locked(request_id)
                    keys = (scan.keys if page_keys is None else
                            (key for key in page_keys if key[0] == request_id))
                    for key in keys:
                        if key in scan.values:
                            self._readiness_cache_hits += 1
                        else:
                            scan.values[key] = self._summarize_release_page_locked(
                                key, True
                            )
                            self._readiness_evaluated_pages += 1
                        value = scan.values[key]
                        if value is not None:
                            summaries.append(value)
                    if page_keys is None:
                        scan.cursor = len(scan.keys)
                return summaries
            keys = (
                {
                    self._aggregate_page_key(page_id)
                    for page_id, _ in self._request_page_records_locked(request_ids)
                }
                if page_keys is None
                else {key for key in page_keys if key[0] in request_ids}
            )
            summaries = []
            for key in keys:
                value = self._summarize_release_page_locked(key, allow_partial_pages)
                if value is not None:
                    summaries.append(value)
            return summaries

    def plan_hbm_release_candidates(
        self,
        *,
        request_ids: set[str] | None = None,
        max_pages: int = 0,
        allow_partial_pages: bool = False,
        max_logical_pages: int = 0,
    ) -> list[HBMReleaseCandidate]:
        """Return deterministic HBM release candidates without releasing them."""
        with self._lock:
            return [
                candidate
                for _, _, candidate in self._collect_hbm_release_candidates_locked(
                    request_ids,
                    max_pages,
                    allow_partial_pages,
                    max_logical_pages,
                )
            ]

    def mark_hbm_released_to_dram(
        self,
        *,
        request_ids: set[str] | None = None,
        max_pages: int = 0,
        allow_partial_pages: bool = False,
        max_logical_pages: int = 0,
    ) -> list[HBMReleaseCandidate]:
        """Mark HBM release candidates as DRAM-authoritative in the sidecar.

        This method changes logical ownership only. The scheduler consumes the
        returned GPU block ids and performs the physical KV-manager free at a
        quiescent request boundary.
        """
        with self._lock:
            candidates = self._collect_hbm_release_candidates_locked(
                request_ids,
                max_pages,
                allow_partial_pages,
                max_logical_pages,
            )
            if not candidates:
                return []
            self._hbm_release_mark_calls += 1
            self._hbm_release_marked_pages += len(candidates)
            partial_page_keys = {
                (
                    page_id.request_id,
                    page_id.kv_cache_group_id,
                    page_id.logical_token_page,
                )
                for page_id, record, _ in candidates
                if not self._is_closed_full_page(record)
            }
            self._hbm_partial_release_marked_pages += len(partial_page_keys)
            for page_id, record, candidate in candidates:
                record.tier = "DRAM"
                record.state = "ready"
                record.version = candidate.dram_copy_version
                record.released_gpu_block_id = candidate.gpu_block_id
                record.gpu_block_id = None
                record.dram_copy_version = candidate.dram_copy_version
                self._refresh_reload_index_locked(page_id)

            # Layer-specific provider records carry the actual K/V copies,
            # while the aggregate record represents the scheduler's physical
            # block table. Once every expected layer is DRAM-authoritative,
            # transition that aggregate record as well. This keeps allocator
            # ownership and layer reload ACK state consistent.
            candidate_page_keys = {
                (
                    page_id.request_id,
                    page_id.kv_cache_group_id,
                    page_id.logical_token_page,
                )
                for page_id, _, _ in candidates
            }
            for request_id, group_id, page_idx in candidate_page_keys:
                expected_layers = self._provider_layers_by_request.get(
                    request_id, set()
                )
                if not expected_layers:
                    continue
                layer_records = [
                    self._pages[page_id]
                    for page_id in self._layer_pages_by_aggregate.get(
                        (request_id, group_id, page_idx), ()
                    )
                    if page_id in self._pages
                    and page_id.layer_id in expected_layers
                ]
                if len(layer_records) < len(expected_layers) or not all(
                    self._is_hbm_reload_required(record)
                    for record in layer_records
                ):
                    continue
                aggregate_id = LogicalPageId(
                    request_id=request_id,
                    kv_cache_group_id=group_id,
                    logical_token_page=page_idx,
                )
                aggregate = self._pages.get(aggregate_id)
                if aggregate is None or self._is_hbm_reload_required(
                    aggregate
                ):
                    continue
                released_block_id = aggregate.gpu_block_id
                if released_block_id is None:
                    released_block_id = layer_records[0].released_gpu_block_id
                if released_block_id is None:
                    continue
                provider_version = max(
                    record.dram_copy_version or 0 for record in layer_records
                )
                aggregate.tier = "DRAM"
                aggregate.state = "ready"
                aggregate.version = provider_version
                aggregate.released_gpu_block_id = released_block_id
                aggregate.gpu_block_id = None
                aggregate.dram_copy_version = provider_version
                aggregate.dram_copy_tokens = aggregate.page_size_tokens
                self._refresh_reload_index_locked(aggregate_id)
            return [candidate for _, _, candidate in candidates]

    def plan_hbm_reload_pages(
        self,
        *,
        request_ids: set[str] | None = None,
        max_pages: int = 0,
        layer_id: int | None = None,
        kv_head_group: int | None = None,
    ) -> list[HBMReloadPage]:
        """Return DRAM-authoritative pages that need a reload/attention plan."""
        with self._lock:
            reload_pages: list[HBMReloadPage] = []
            if request_ids is None:
                candidate_ids = (
                    set().union(*self._reload_required_pages_by_request.values())
                    if self._reload_required_pages_by_request
                    else set()
                )
            elif layer_id is not None:
                # Filter identity before sorting, and consult the live reload
                # index. Layer ACK can remove readiness without changing IDs.
                candidate_ids = {
                    page_id
                    for request_id in request_ids
                    for page_id in self._page_ids_by_request_layer.get(
                        (request_id, layer_id), ()
                    )
                    if page_id in self._reload_required_pages_by_request.get(
                        request_id, ()
                    )
                }
            else:
                candidate_ids = (
                    set().union(
                        *(
                            self._reload_required_pages_by_request.get(
                                request_id, set()
                            )
                            for request_id in request_ids
                        )
                    )
                    if request_ids
                    else set()
                )
            for page_id in sorted(
                candidate_ids,
                key=lambda candidate: (
                    candidate.request_id,
                    candidate.kv_cache_group_id,
                    candidate.layer_id,
                    candidate.kv_head_group,
                    candidate.logical_token_page,
                ),
            ):
                record = self._pages.get(page_id)
                if record is None:
                    continue
                if layer_id is not None and page_id.layer_id != layer_id:
                    continue
                if (
                    kv_head_group is not None
                    and page_id.kv_head_group != kv_head_group
                ):
                    continue
                if not self._is_hbm_reload_required(record):
                    continue
                assert record.released_gpu_block_id is not None
                assert record.dram_copy_version is not None
                reload_pages.append(
                    HBMReloadPage(
                        request_id=page_id.request_id,
                        kv_cache_group_id=page_id.kv_cache_group_id,
                        logical_token_page=page_id.logical_token_page,
                        layer_id=page_id.layer_id,
                        kv_head_group=page_id.kv_head_group,
                        provider_version=record.dram_copy_version,
                        previous_gpu_block_id=record.released_gpu_block_id,
                        page_size_tokens=record.page_size_tokens,
                    )
                )
                if max_pages > 0 and len(reload_pages) >= max_pages:
                    break
            return reload_pages

    def has_hbm_reload_required_pages(
        self,
        request_ids: set[str] | None = None,
    ) -> bool:
        """Return reload presence without scanning or sorting page records."""
        with self._lock:
            if request_ids is None:
                return any(self._reload_required_pages_by_request.values())
            return any(
                self._reload_required_pages_by_request.get(request_id)
                for request_id in request_ids
            )

    def hbm_reload_required_page_count(
        self,
        request_ids: set[str] | None = None,
    ) -> int:
        """Return an O(number of requests) reload-required page count."""
        with self._lock:
            if request_ids is None:
                return sum(
                    len(pages)
                    for pages in self._reload_required_pages_by_request.values()
                )
            return sum(
                len(self._reload_required_pages_by_request.get(request_id, ()))
                for request_id in request_ids
            )

    def pin_attention_page_versions(
        self,
        request_ids: list[str],
        *,
        group_id: int = 0,
        layer_id: int = -1,
        kv_head_group: int = -1,
    ) -> dict[str, dict[int, int]]:
        """Return a stable logical-page -> version snapshot for attention.

        This does not move data or change residency. It is a control-plane
        snapshot that lets a tiered backend avoid racing against "latest"
        provider versions while a request is being migrated.
        """
        request_id_set = set(request_ids)
        versions: dict[str, dict[int, int]] = defaultdict(dict)
        with self._lock:
            selected_ids = (
                page_id
                for request_id in sorted(request_id_set)
                for page_id in self._page_ids_by_request_layer.get(
                    (request_id, layer_id), ()
                )
            )
            for page_id in selected_ids:
                record = self._pages[page_id]
                if page_id.kv_cache_group_id != group_id:
                    continue
                if page_id.layer_id != layer_id:
                    continue
                if page_id.kv_head_group != kv_head_group:
                    continue
                if record.state != "ready":
                    continue
                versions[page_id.request_id][
                    page_id.logical_token_page
                ] = record.version
        return {
            request_id: dict(page_versions)
            for request_id, page_versions in versions.items()
        }

    def summarize_attention_plan(
        self,
        phase: AttentionPhase,
        request_ids: list[str],
        query_tokens: int,
        staging_budget_blocks: int = 0,
        dma_budget_bytes: int = 0,
        tbt_budget_ms: float | None = None,
    ) -> AttentionPlanSummary:
        request_id_set = set(request_ids)
        # Decode correctness is handled by reloading any DRAM-resident pages
        # into HBM and then using the native attention path. The context-ring
        # backend is intentionally a prefill-only execution path in this
        # prototype so decode TBT does not depend on DRAM traversal.
        ring_path_requested = (
            virtual_context_ring_attention_prefill_enabled()
            if phase == "prefill"
            else False
        )
        with self._lock:
            page_records = [
                (page_id, record)
                for page_id, record in self._request_page_records_locked(request_id_set)
            ]
            pages = [record for _, record in page_records]
            hbm_pages = sum(record.tier == "HBM" for record in pages)
            dram_pages = sum(record.tier == "DRAM" for record in pages)
            staging_pages = sum(record.tier == "STAGING" for record in pages)
            ring_traversal_segments = (
                self._count_ring_traversal_segments(page_records)
                if ring_path_requested
                else 0
            )

        if phase == "prefill":
            pages_to_materialize = dram_pages + staging_pages
        else:
            pages_to_materialize = max(0, len(pages) - hbm_pages)

        if pages_to_materialize == 0:
            feasible = True
        elif staging_budget_blocks <= 0:
            feasible = False
        else:
            feasible = pages_to_materialize <= staging_budget_blocks

        ring_executor_enabled = virtual_context_ring_executor_enabled()
        ring_backend_enabled = virtual_context_ring_backend_enabled()
        ring_backend_provider_enabled = (
            virtual_context_ring_backend_dram_provider_enabled()
        )
        ring_backend_force_all = virtual_context_ring_backend_force_all_enabled()
        effective_ring_backend_force_all = (
            ring_backend_force_all and phase == "prefill"
        )
        phase_allows_residency_ring = phase == "prefill"
        ring_backend_needed_by_residency = (
            pages_to_materialize > 0 and phase_allows_residency_ring
        )
        ring_backend_will_attempt = (
            ring_backend_enabled
            and len(pages) > 0
            and query_tokens > 0
            and (
                ring_backend_needed_by_residency
                or effective_ring_backend_force_all
            )
        )
        ring_path_active = (
            (ring_path_requested or ring_backend_will_attempt)
            and len(pages) > 0
            and query_tokens > 0
            and (
                ring_backend_needed_by_residency
                or effective_ring_backend_force_all
            )
        )
        ring_traversal_pages = len(pages) if ring_path_active else 0
        ring_hbm_traversal_pages = hbm_pages if ring_path_active else 0
        ring_dram_traversal_pages = dram_pages if ring_path_active else 0
        ring_staging_traversal_pages = staging_pages if ring_path_active else 0
        ring_pages_to_materialize = (
            pages_to_materialize if ring_path_active else 0
        )
        ring_online_softmax_merge_required = (
            ring_path_active and ring_traversal_pages > 1
        )

        return AttentionPlanSummary(
            phase=phase,
            request_count=len(request_id_set),
            query_tokens=query_tokens,
            required_pages=len(pages),
            hbm_resident_pages=hbm_pages,
            dram_resident_pages=dram_pages,
            staging_pages=staging_pages,
            pages_to_materialize=pages_to_materialize,
            staging_budget_blocks=staging_budget_blocks,
            dma_budget_bytes=dma_budget_bytes,
            tbt_budget_ms=tbt_budget_ms,
            feasible_with_current_staging_budget=feasible,
            ring_path_requested=ring_path_requested,
            ring_path_active=ring_path_active,
            ring_traversal_pages=ring_traversal_pages,
            ring_traversal_segments=ring_traversal_segments,
            ring_hbm_traversal_pages=ring_hbm_traversal_pages,
            ring_dram_traversal_pages=ring_dram_traversal_pages,
            ring_staging_traversal_pages=ring_staging_traversal_pages,
            ring_pages_to_materialize=ring_pages_to_materialize,
            ring_online_softmax_merge_required=(
                ring_online_softmax_merge_required
            ),
            ring_executor_available=True,
            ring_executor_enabled=ring_executor_enabled,
            # v0.5.5.1 adds a callable reference data-plane. v0.5.5.2 can
            # attach it to a narrow TRITON_ATTN backend path behind a flag.
            ring_executor_implemented=ring_executor_enabled,
            ring_executor_attached_to_vllm_attention=ring_backend_will_attempt,
            ring_backend_needed_by_residency=ring_backend_needed_by_residency,
            ring_backend_force_all=effective_ring_backend_force_all,
            ring_backend_will_attempt=ring_backend_will_attempt,
            ring_decode_slo_policy_applied=False,
        )

    @staticmethod
    def _count_ring_traversal_segments(
        page_records: list[tuple[LogicalPageId, ResidencyRecord]],
    ) -> int:
        pages_by_logical_stream: defaultdict[
            tuple[str, int, int, int], list[int]
        ] = defaultdict(list)
        for page_id, _ in page_records:
            stream_key = (
                page_id.request_id,
                page_id.kv_cache_group_id,
                page_id.layer_id,
                page_id.kv_head_group,
            )
            pages_by_logical_stream[stream_key].append(
                page_id.logical_token_page
            )

        segments = 0
        for logical_pages in pages_by_logical_stream.values():
            if not logical_pages:
                continue
            sorted_pages = sorted(set(logical_pages))
            segments += 1
            for prev_page, next_page in zip(sorted_pages, sorted_pages[1:]):
                if next_page != prev_page + 1:
                    segments += 1
        return segments

    @staticmethod
    def summarize_residency_budget(
        free_hbm_blocks: int,
        decode_hotset_reserved_blocks: int,
        decode_hotset_used_blocks: int,
        prefill_required_growth_blocks: int,
        staging_reserved_blocks: int = 0,
    ) -> ResidencyBudgetSnapshot:
        available_after_prefill = (
            free_hbm_blocks - prefill_required_growth_blocks
        )
        required_reserve = decode_hotset_reserved_blocks + staging_reserved_blocks
        feasible = available_after_prefill >= required_reserve
        serialized = prefill_required_growth_blocks > 0 and not feasible
        return ResidencyBudgetSnapshot(
            free_hbm_blocks=free_hbm_blocks,
            decode_hotset_reserved_blocks=decode_hotset_reserved_blocks,
            decode_hotset_used_blocks=decode_hotset_used_blocks,
            prefill_required_growth_blocks=prefill_required_growth_blocks,
            staging_reserved_blocks=staging_reserved_blocks,
            available_after_prefill_growth_blocks=available_after_prefill,
            prefill_admission_feasible_with_reserve=feasible,
            serialized_by_residency_budget=serialized,
        )

    def summarize_reload_guard(
        self,
        phase: AttentionPhase,
        request_ids: list[str],
    ) -> ReloadGuardSummary:
        """Summarize whether scheduled requests need non-HBM KV handling.

        The scheduler consumes this guard to choose native HBM reload or
        provider-backed tiered attention before execution. The returned counts
        include layer-specific residency records; callers that need physical
        HBM blocks deduplicate by KV group and logical page.
        """
        request_id_set = set(request_ids)
        with self._lock:
            candidate_ids = (
                set().union(
                    *(
                        self._reload_required_pages_by_request.get(
                            request_id, set()
                        )
                        for request_id in request_id_set
                    )
                )
                if request_id_set
                else set()
            )
            reload_required_pages = [
                (page_id, self._pages[page_id])
                for page_id in candidate_ids
                if page_id in self._pages
            ]
        reload_required_request_ids = {
            page_id.request_id for page_id, _ in reload_required_pages
        }
        return ReloadGuardSummary(
            phase=phase,
            request_count=len(request_id_set),
            reload_required_pages=len(reload_required_pages),
            reload_required_tokens=sum(
                record.page_size_tokens for _, record in reload_required_pages
            ),
            reload_required_request_count=len(reload_required_request_ids),
            tiered_backend_enabled=virtual_context_ring_backend_enabled(),
            dram_provider_enabled=(
                virtual_context_ring_backend_dram_provider_enabled()
            ),
            reload_backend_available=(
                virtual_context_reload_to_hbm_enabled()
                and virtual_context_ring_backend_dram_provider_enabled()
            ),
        )

    def retire_layerwise_hbm(self, request_ids: Iterable[str]) -> None:
        """Forget freed addresses, but preserve monotonic provider versions.

        Called only by the allocator after validating the durable private
        checkpoint. These requests remain Future-owned, not ordinarily ready.
        """
        with self._lock:
            for request_id in request_ids:
                self._invalidate_release_readiness_locked(request_id)
                for page_id in tuple(self._page_ids_by_request.get(request_id, ())):
                    self._delete_page_locked(page_id)
                self._request_tokens.pop(request_id, None)
                self._provider_layers_by_request.pop(request_id, None)

    def reserve_layerwise_provider_version(self, request_id: str) -> int:
        with self._lock:
            return self._next_version(request_id)

    def publish_layerwise_dram(
        self, *, request_id: str, logical_tokens: int, block_size: int,
        released_block_ids: tuple[int, ...], layer_ids: tuple[int, ...],
        version: int,
    ) -> None:
        """Publish uniform full-model coverage after worker provider writes.

        released_block_ids are provenance only, NEVER readable HBM addresses.
        Their presence activates the existing native-reload safety barrier.
        """
        pages = (logical_tokens + block_size - 1) // block_size if block_size > 0 else 0
        if (logical_tokens <= 0 or pages > len(released_block_ids)
                or not layer_ids or len(set(layer_ids)) != len(layer_ids)
                or any(layer < 0 for layer in layer_ids)):
            raise RuntimeError("invalid final layerwise DRAM coverage")
        with self._lock:
            if self._request_versions.get(request_id) != version:
                raise RuntimeError("stale layerwise provider publication")
            if self._aggregate_pages_by_request.get(request_id):
                raise RuntimeError("layerwise publication cannot replace live HBM")
            self._invalidate_release_readiness_locked(request_id)
            self._provider_layers_by_request[request_id] = set(layer_ids)
            for page in range(pages):
                count = min(block_size, logical_tokens - page * block_size)
                for layer_id in (-1, *layer_ids):
                    self._set_page_locked(
                        LogicalPageId(request_id=request_id, kv_cache_group_id=0,
                                      logical_token_page=page, layer_id=layer_id),
                        ResidencyRecord(
                            tier="DRAM", state="ready", version=version,
                            page_size_tokens=count, block_size_tokens=block_size,
                            dram_copy_version=version, dram_copy_tokens=count,
                            released_gpu_block_id=released_block_ids[page],
                        ),
                    )
            self._request_tokens[request_id] = logical_tokens

    def release_request(self, request_id: str) -> None:
        with self._lock:
            self._released_requests += 1
            self._invalidate_release_readiness_locked(request_id)
            self._request_tokens.pop(request_id, None)
            self._request_versions.pop(request_id, None)
            self._provider_layers_by_request.pop(request_id, None)
            for page_id in tuple(self._page_ids_by_request.get(request_id, ())):
                self._delete_page_locked(page_id)

    def get_request_debug_stats(self, request_id: str) -> dict[str, int | float]:
        with self._lock:
            pages = [
                record
                for _, record in self._request_page_records_locked({request_id})
            ]
            release_candidates = [
                record
                for _, record, _ in self._collect_hbm_release_candidates_locked(
                    {request_id}, 0, False
                )
            ]
            release_candidates_with_partial = [
                record
                for _, record, _ in self._collect_hbm_release_candidates_locked(
                    {request_id}, 0, True
                )
            ]
            release_marked_pages = [
                record
                for record in pages
                if record.tier == "DRAM" and record.released_gpu_block_id is not None
            ]
            reload_required_pages = [
                record for record in pages if self._is_hbm_reload_required(record)
            ]
            partial_tail_release_blocked_pages = [
                record
                for record in pages
                if self._is_partial_tail_release_blocked(record)
            ]
            partial_tail_hbm_pinned_pages = [
                record
                for record in pages
                if record.tier == "HBM"
                and record.state == "ready"
                and not self._is_closed_full_page(record)
            ]
            return {
                "virtual_context_request_tokens": self._request_tokens.get(
                    request_id, 0
                ),
                "virtual_context_request_pages": len(pages),
                "virtual_context_request_hbm_pages": sum(
                    record.tier == "HBM" for record in pages
                ),
                "virtual_context_request_dram_pages": sum(
                    record.tier == "DRAM" for record in pages
                ),
                "virtual_context_request_dram_copy_pages": sum(
                    record.dram_copy_version is not None for record in pages
                ),
                "virtual_context_request_dram_copy_short_pages": sum(
                    record.dram_copy_version is not None
                    and record.dram_copy_tokens < record.page_size_tokens
                    for record in pages
                ),
                "virtual_context_request_provider_expected_layers": len(
                    self._provider_layers_by_request.get(request_id, set())
                ),
                "virtual_context_request_hbm_release_candidate_pages": len(
                    release_candidates
                ),
                "virtual_context_request_hbm_release_candidate_tokens": sum(
                    record.page_size_tokens for record in release_candidates
                ),
                "virtual_context_request_hbm_release_candidate_with_partial_pages": (
                    len(release_candidates_with_partial)
                ),
                "virtual_context_request_hbm_release_candidate_with_partial_tokens": (
                    sum(
                        record.page_size_tokens
                        for record in release_candidates_with_partial
                    )
                ),
                "virtual_context_request_hbm_release_marked_pages": len(
                    release_marked_pages
                ),
                "virtual_context_request_hbm_release_marked_tokens": sum(
                    record.page_size_tokens for record in release_marked_pages
                ),
                "virtual_context_request_hbm_reload_required_pages": len(
                    reload_required_pages
                ),
                "virtual_context_request_hbm_reload_required_tokens": sum(
                    record.page_size_tokens for record in reload_required_pages
                ),
                "virtual_context_request_partial_tail_release_blocked_pages": (
                    len(partial_tail_release_blocked_pages)
                ),
                "virtual_context_request_partial_tail_release_blocked_tokens": (
                    sum(
                        record.page_size_tokens
                        for record in partial_tail_release_blocked_pages
                    )
                ),
                "virtual_context_request_partial_tail_hbm_pinned_pages": len(
                    partial_tail_hbm_pinned_pages
                ),
                "virtual_context_request_partial_tail_hbm_pinned_tokens": sum(
                    record.page_size_tokens
                    for record in partial_tail_hbm_pinned_pages
                ),
            }

    def get_debug_stats(self) -> dict[str, int | float]:
        with self._lock:
            tier_counts = defaultdict[str, int](int)
            state_counts = defaultdict[str, int](int)
            max_version = 0
            dram_copy_pages = 0
            dram_copy_short_pages = 0
            safe_release_candidates = (
                self._collect_hbm_release_candidates_locked(None, 0, False)
            )
            hbm_release_candidates = len(safe_release_candidates)
            hbm_release_candidate_tokens = sum(
                record.page_size_tokens
                for _, record, _ in safe_release_candidates
            )
            hbm_release_candidate_requests = {
                page_id.request_id
                for page_id, _, _ in safe_release_candidates
            }
            hbm_release_marked_pages = 0
            hbm_release_marked_tokens = 0
            hbm_release_marked_requests: set[str] = set()
            hbm_reload_required_pages = 0
            hbm_reload_required_tokens = 0
            hbm_reload_required_requests: set[str] = set()
            partial_tail_release_blocked_pages = 0
            partial_tail_release_blocked_tokens = 0
            partial_tail_release_blocked_requests: set[str] = set()
            partial_tail_hbm_pinned_pages = 0
            partial_tail_hbm_pinned_tokens = 0
            partial_tail_hbm_pinned_requests: set[str] = set()
            for page_id, record in self._pages.items():
                tier_counts[record.tier] += 1
                state_counts[record.state] += 1
                max_version = max(max_version, record.version)
                if record.dram_copy_version is not None:
                    dram_copy_pages += 1
                    if record.dram_copy_tokens < record.page_size_tokens:
                        dram_copy_short_pages += 1
                if (
                    record.tier == "DRAM"
                    and record.released_gpu_block_id is not None
                ):
                    hbm_release_marked_pages += 1
                    hbm_release_marked_tokens += record.page_size_tokens
                    hbm_release_marked_requests.add(page_id.request_id)
                if self._is_hbm_reload_required(record):
                    hbm_reload_required_pages += 1
                    hbm_reload_required_tokens += record.page_size_tokens
                    hbm_reload_required_requests.add(page_id.request_id)
                if self._is_partial_tail_release_blocked(record):
                    partial_tail_release_blocked_pages += 1
                    partial_tail_release_blocked_tokens += (
                        record.page_size_tokens
                    )
                    partial_tail_release_blocked_requests.add(
                        page_id.request_id
                    )
                if (
                    record.tier == "HBM"
                    and record.state == "ready"
                    and not self._is_closed_full_page(record)
                ):
                    partial_tail_hbm_pinned_pages += 1
                    partial_tail_hbm_pinned_tokens += record.page_size_tokens
                    partial_tail_hbm_pinned_requests.add(page_id.request_id)

            return {
                "virtual_context_enabled": 1,
                "virtual_context_readiness_cache_hits": self._readiness_cache_hits,
                "virtual_context_readiness_evaluated_pages": (
                    self._readiness_evaluated_pages
                ),
                "virtual_context_readiness_invalidations": (
                    self._readiness_invalidations
                ),
                "virtual_context_readiness_warm_yields": self._readiness_warm_yields,
                "logical_context_request_count": len(self._request_tokens),
                "logical_context_tokens_total": sum(self._request_tokens.values()),
                "logical_pages_total": len(self._pages),
                "logical_pages_hbm": tier_counts["HBM"],
                "logical_pages_dram": tier_counts["DRAM"],
                "logical_pages_staging": tier_counts["STAGING"],
                "logical_pages_dram_copy_ready": dram_copy_pages,
                "logical_pages_dram_copy_short": dram_copy_short_pages,
                "virtual_context_provider_expected_layer_requests": len(
                    self._provider_layers_by_request
                ),
                "logical_pages_hbm_release_candidates": (
                    hbm_release_candidates
                ),
                "logical_hbm_release_candidate_tokens": (
                    hbm_release_candidate_tokens
                ),
                "logical_hbm_release_candidate_request_count": len(
                    hbm_release_candidate_requests
                ),
                "logical_pages_hbm_release_marked": hbm_release_marked_pages,
                "logical_hbm_release_marked_tokens": hbm_release_marked_tokens,
                "logical_hbm_release_marked_request_count": len(
                    hbm_release_marked_requests
                ),
                "logical_pages_hbm_reload_required": hbm_reload_required_pages,
                "logical_hbm_reload_required_tokens": (
                    hbm_reload_required_tokens
                ),
                "logical_hbm_reload_required_request_count": len(
                    hbm_reload_required_requests
                ),
                "logical_pages_partial_tail_release_blocked": (
                    partial_tail_release_blocked_pages
                ),
                "logical_partial_tail_release_blocked_tokens": (
                    partial_tail_release_blocked_tokens
                ),
                "logical_partial_tail_release_blocked_request_count": len(
                    partial_tail_release_blocked_requests
                ),
                "logical_pages_partial_tail_hbm_pinned": (
                    partial_tail_hbm_pinned_pages
                ),
                "logical_partial_tail_hbm_pinned_tokens": (
                    partial_tail_hbm_pinned_tokens
                ),
                "logical_partial_tail_hbm_pinned_request_count": len(
                    partial_tail_hbm_pinned_requests
                ),
                "logical_pages_ready": state_counts["ready"],
                "logical_pages_loading": state_counts["loading"],
                "logical_pages_storing": state_counts["storing"],
                "logical_pages_evicting": state_counts["evicting"],
                "logical_residency_version_max": max_version,
                "virtual_context_observe_hbm_calls": self._observe_hbm_calls,
                "virtual_context_observe_hbm_delta_calls": (
                    self._observe_hbm_delta_calls
                ),
                "virtual_context_observe_hbm_delta_pages_total": (
                    self._observe_hbm_delta_pages
                ),
                "virtual_context_observe_hbm_full_reconcile_pages_total": (
                    self._observe_hbm_full_reconcile_pages
                ),
                "virtual_context_reload_required_index_pages": sum(
                    len(pages)
                    for pages in self._reload_required_pages_by_request.values()
                ),
                "virtual_context_observe_dram_load_calls": (
                    self._observe_dram_load_calls
                ),
                "virtual_context_observe_dram_store_calls": (
                    self._observe_dram_store_calls
                ),
                "virtual_context_dram_copy_ready_calls": (
                    self._dram_copy_ready_calls
                ),
                "virtual_context_dram_copy_short_rejected_pages_total": (
                    self._dram_copy_short_rejected_pages
                ),
                "virtual_context_hbm_release_mark_calls": (
                    self._hbm_release_mark_calls
                ),
                "virtual_context_hbm_release_marked_pages_total": (
                    self._hbm_release_marked_pages
                ),
                "virtual_context_hbm_partial_release_marked_pages_total": (
                    self._hbm_partial_release_marked_pages
                ),
                "virtual_context_hbm_reload_complete_calls": (
                    self._hbm_reload_complete_calls
                ),
                "virtual_context_hbm_reload_completed_pages_total": (
                    self._hbm_reload_completed_pages
                ),
                "virtual_context_hbm_reload_aggregate_completed_pages_total": (
                    self._hbm_reload_aggregate_completed_pages
                ),
                "virtual_context_released_requests": self._released_requests,
            }


_GLOBAL_TABLE: VirtualContextTable | None = None


def get_virtual_context_table() -> VirtualContextTable | None:
    global _GLOBAL_TABLE
    if not virtual_context_enabled():
        return None
    if _GLOBAL_TABLE is None:
        _GLOBAL_TABLE = VirtualContextTable(
            cache_release_readiness=(os.getenv(
                "VLLM_ISOLATION_BOUNDED_PREFILL_PLANNING",
                os.getenv("VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS", "0"),
            ) == "1")
        )
    return _GLOBAL_TABLE
