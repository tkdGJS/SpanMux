# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.virtual_context import (
    AttentionPhase,
    HBMReleaseCandidate,
    HBMReleaseReadiness,
    HBMReloadPage,
    VirtualContextTable,
    get_virtual_context_table,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len
        self.hash_block_size = hash_block_size

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )
        self.virtual_context: VirtualContextTable | None = (
            get_virtual_context_table()
        )
        self.virtual_context_delta_hbm_observe = os.getenv(
            "VLLM_VIRTUAL_CONTEXT_DELTA_HBM_OBSERVE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._virtual_context_observe_hbm_duration_s = 0.0
        self._virtual_context_observe_hbm_delta_calls = 0
        self._virtual_context_observe_hbm_full_calls = 0

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def reset_virtual_context_schedule_timings(self) -> None:
        self._virtual_context_observe_hbm_duration_s = 0.0
        self._virtual_context_observe_hbm_delta_calls = 0
        self._virtual_context_observe_hbm_full_calls = 0

    def get_virtual_context_schedule_timings(self) -> dict[str, int | float]:
        return {
            "virtual_context_observe_hbm_duration_s": (
                self._virtual_context_observe_hbm_duration_s
            ),
            "virtual_context_observe_hbm_delta_calls": (
                self._virtual_context_observe_hbm_delta_calls
            ),
            "virtual_context_observe_hbm_full_calls": (
                self._virtual_context_observe_hbm_full_calls
            ),
        }

    def has_virtual_context_hbm_reload_required_pages(
        self,
        request_ids: set[str] | None = None,
    ) -> bool:
        if self.virtual_context is None:
            return False
        return self.virtual_context.has_hbm_reload_required_pages(request_ids)

    def get_virtual_context_hbm_reload_required_page_count(
        self,
        request_ids: set[str] | None = None,
    ) -> int:
        """Return reload-required layer-page count without materializing pages.

        Scheduler admission only needs a count/presence check. Building the
        sorted ``HBMReloadPage`` list here turns a blocked long-context decode
        waiter into an O(layer-pages) operation on every generated token.
        ``VirtualContextTable`` maintains a request-indexed reload set, so this
        count is O(number of requested request IDs).
        """

        if self.virtual_context is None:
            return 0
        return self.virtual_context.hbm_reload_required_page_count(request_ids)

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def get_tenant_kv_debug_stats(self) -> dict[str, int | float]:
        """Return experimental tenant KV retention counters."""
        return self.block_pool.get_tenant_kv_debug_stats()

    def get_virtual_context_debug_stats(self) -> dict[str, int | float]:
        stats: dict[str, int | float] = {}
        if self.virtual_context is None:
            return stats
        stats.update(self.virtual_context.get_debug_stats())
        try:
            from vllm.v1.attention.ops.tiered_residency_provider import (
                get_dram_kv_page_registry_stats,
            )

            stats.update(get_dram_kv_page_registry_stats())
        except Exception:
            # Debug metrics must never affect scheduler correctness.
            pass
        return stats

    def get_virtual_context_request_debug_stats(
        self, request_id: str
    ) -> dict[str, int | float]:
        if self.virtual_context is None:
            return {}
        return self.virtual_context.get_request_debug_stats(request_id)

    def get_virtual_context_attention_plan_debug_stats(
        self,
        phase: AttentionPhase,
        request_ids: list[str],
        query_tokens: int,
        prefix: str,
        staging_budget_blocks: int = 0,
        dma_budget_bytes: int = 0,
        tbt_budget_ms: float | None = None,
    ) -> dict[str, int | float | str]:
        if self.virtual_context is None:
            return {}
        return self.virtual_context.summarize_attention_plan(
            phase=phase,
            request_ids=request_ids,
            query_tokens=query_tokens,
            staging_budget_blocks=staging_budget_blocks,
            dma_budget_bytes=dma_budget_bytes,
            tbt_budget_ms=tbt_budget_ms,
        ).to_debug_fields(prefix)

    def get_virtual_context_residency_budget_debug_stats(
        self,
        free_hbm_blocks: int,
        decode_hotset_reserved_blocks: int,
        decode_hotset_used_blocks: int,
        prefill_required_growth_blocks: int,
        staging_reserved_blocks: int = 0,
    ) -> dict[str, int]:
        if self.virtual_context is None:
            return {}
        return self.virtual_context.summarize_residency_budget(
            free_hbm_blocks=free_hbm_blocks,
            decode_hotset_reserved_blocks=decode_hotset_reserved_blocks,
            decode_hotset_used_blocks=decode_hotset_used_blocks,
            prefill_required_growth_blocks=prefill_required_growth_blocks,
            staging_reserved_blocks=staging_reserved_blocks,
        ).to_debug_fields()

    def get_virtual_context_reload_guard_debug_stats(
        self,
        phase: AttentionPhase,
        request_ids: list[str],
        prefix: str,
    ) -> dict[str, int | str]:
        if self.virtual_context is None:
            return {}
        return self.virtual_context.summarize_reload_guard(
            phase=phase,
            request_ids=request_ids,
        ).to_debug_fields(prefix)

    def plan_virtual_context_hbm_release_candidates(
        self,
        request_ids: set[str] | None = None,
        max_pages: int = 0,
        allow_partial_pages: bool = False,
        max_logical_pages: int = 0,
    ) -> list[HBMReleaseCandidate]:
        if self.virtual_context is None:
            return []
        return self.virtual_context.plan_hbm_release_candidates(
            request_ids=request_ids,
            max_pages=max_pages,
            allow_partial_pages=allow_partial_pages,
            max_logical_pages=max_logical_pages,
        )

    def warm_virtual_context_hbm_release_readiness(
        self, *, request_ids: set[str], max_pages: int = 64,
    ) -> bool:
        if self.virtual_context is None:
            return True
        return self.virtual_context.warm_hbm_release_readiness(
            request_ids=request_ids, max_pages=max_pages,
        )

    def summarize_virtual_context_hbm_release_readiness(
        self,
        *,
        request_ids: set[str],
        allow_partial_pages: bool = False,
        page_keys: set[tuple[str, int, int]] | None = None,
    ) -> list[HBMReleaseReadiness]:
        if self.virtual_context is None:
            return []
        return self.virtual_context.summarize_hbm_release_readiness(
            request_ids=request_ids,
            allow_partial_pages=allow_partial_pages,
            page_keys=page_keys,
        )

    def mark_virtual_context_hbm_released_to_dram(
        self,
        request_ids: set[str] | None = None,
        max_pages: int = 0,
        allow_partial_pages: bool = False,
        max_logical_pages: int = 0,
    ) -> list[HBMReleaseCandidate]:
        if self.virtual_context is None:
            return []
        return self.virtual_context.mark_hbm_released_to_dram(
            request_ids=request_ids,
            max_pages=max_pages,
            allow_partial_pages=allow_partial_pages,
            max_logical_pages=max_logical_pages,
        )

    def release_virtual_context_hbm_pages(
        self,
        *,
        request_id: str,
        max_logical_pages: int,
        allow_partial_pages: bool = False,
    ) -> tuple[list[HBMReleaseCandidate], int]:
        """Commit and physically release an exact logical-page prefix.

        The sidecar returns one record per aggregate/layer page, while the
        allocator owns one physical block per cache-group/logical-page pair.
        Validate that complete mapping before changing either ownership view.
        """

        if self.virtual_context is None or max_logical_pages <= 0:
            return [], 0
        planned = self.virtual_context.plan_hbm_release_candidates(
            request_ids={request_id},
            allow_partial_pages=allow_partial_pages,
            max_logical_pages=max_logical_pages,
        )
        page_to_block_by_group: dict[int, dict[int, int]] = {}
        for candidate in planned:
            page_to_block = page_to_block_by_group.setdefault(
                candidate.kv_cache_group_id,
                {},
            )
            previous = page_to_block.setdefault(
                candidate.logical_token_page,
                candidate.gpu_block_id,
            )
            if previous != candidate.gpu_block_id:
                return [], 0
        if not page_to_block_by_group:
            return [], 0

        current_blocks = self.coordinator.get_blocks(request_id)
        for group_id, page_to_block in page_to_block_by_group.items():
            if group_id < 0 or group_id >= len(current_blocks):
                return [], 0
            group_blocks = current_blocks[group_id]
            if any(
                page < 0
                or page >= len(group_blocks)
                or group_blocks[page].is_null
                or group_blocks[page].block_id != block_id
                for page, block_id in page_to_block.items()
            ):
                return [], 0

        marked = self.virtual_context.mark_hbm_released_to_dram(
            request_ids={request_id},
            allow_partial_pages=allow_partial_pages,
            max_logical_pages=max_logical_pages,
        )
        planned_keys = {
            (
                candidate.kv_cache_group_id,
                candidate.logical_token_page,
            )
            for candidate in planned
        }
        marked_keys = {
            (
                candidate.kv_cache_group_id,
                candidate.logical_token_page,
            )
            for candidate in marked
        }
        if marked_keys != planned_keys:
            raise RuntimeError(
                "virtual-context partial release plan changed at commit"
            )
        released_blocks = self.coordinator.release_hbm_pages(
            request_id,
            page_to_block_by_group,
        )
        if released_blocks != len(planned_keys):
            raise RuntimeError(
                "virtual-context partial release allocator mismatch: "
                f"planned={len(planned_keys)} released={released_blocks}"
            )
        return marked, released_blocks

    def plan_virtual_context_hbm_reload_pages(
        self,
        request_ids: set[str] | None = None,
        max_pages: int = 0,
    ) -> list[HBMReloadPage]:
        if self.virtual_context is None:
            return []
        return self.virtual_context.plan_hbm_reload_pages(
            request_ids=request_ids,
            max_pages=max_pages,
        )

    def allocate_virtual_context_hbm_reload_blocks(
        self,
        request_id: str,
        logical_pages_by_group: dict[int, set[int]],
    ) -> KVCacheBlocks | None:
        """Install HBM block-table slots for virtual-context reload pages."""
        allocated = self.coordinator.allocate_hbm_reload_blocks(
            request_id,
            logical_pages_by_group,
        )
        if allocated is None:
            return None
        if self.virtual_context is not None:
            max_page = max(
                (
                    page
                    for pages in logical_pages_by_group.values()
                    for page in pages
                ),
                default=-1,
            )
            self.virtual_context.observe_hbm_blocks(
                request_id=request_id,
                block_ids_by_group=self.get_block_ids(request_id),
                block_size=self.hash_block_size,
                logical_tokens=max(0, max_page + 1) * self.hash_block_size,
                null_block_id=self.block_pool.null_block.block_id,
            )
        # ``get_blocks`` already returns a KVCacheBlocks wrapper.  The reload
        # allocation above mutates the request's block table in-place, so return
        # the refreshed wrapper directly instead of wrapping it a second time.
        return self.get_blocks(request_id)

    def estimate_num_blocks_to_allocate(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_virtual_external_computed_tokens: int = 0,
        num_encoder_tokens: int = 0,
        require_native_hbm: bool = False,
    ) -> int:
        """Estimate additional KV blocks without mutating allocator state.

        Native reload needs physical prefix slots, not per-layer copy records.
        With no retained table, ordinary nonvirtual allocation already counts
        the entire prefix (including a partial tail) and future growth. With a
        retained table it counts growth but treats null slots as existing;
        add those historical holes once. Concrete prepared destinations remain
        owned and are not charged again. Neither estimate proves copy readiness.
        """
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_need_slot = min(
            total_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )
        required_blocks = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=(
                num_local_computed_tokens + num_external_computed_tokens
            ),
            num_virtual_external_computed_tokens=(
                0 if require_native_hbm else num_virtual_external_computed_tokens
            ),
        )
        if require_native_hbm and self.has_request_block_table(request.request_id):
            # Validate retained coverage using the same allocator geometry as
            # reload admission. Do not reconstruct it from layer-copy counts.
            _, historical_null_blocks = self.get_retained_residency_counts(
                request.request_id,
                min(
                    request.num_computed_tokens + num_external_computed_tokens,
                    self.max_model_len,
                ),
            )
            required_blocks += historical_null_blocks
        return required_blocks

    def get_request_resident_block_count(self, request_id: str) -> int:
        """Return currently allocated non-null KV blocks for a request."""
        return sum(
            1
            for group in self.coordinator.get_blocks(request_id)
            for block in group
            if not block.is_null
        )

    def has_request_block_table(self, request_id: str) -> bool:
        """Return whether allocator metadata is retained for a request."""

        return any(
            bool(group) for group in self.coordinator.get_blocks(request_id)
        )

    def is_request_partial_residency(self, request_id: str) -> bool:
        """Return whether the allocator owns a retained mixed block table."""

        return self.coordinator.is_partial_residency(request_id)

    def resume_partial_residency(
        self,
        request_id: str,
        logical_computed_tokens: int,
    ) -> tuple[int, int]:
        """Validate a retained HBM/null table for scheduler readmission."""

        return self.coordinator.resume_partial_residency(
            request_id,
            logical_computed_tokens,
        )

    def get_retained_residency_counts(
        self,
        request_id: str,
        logical_computed_tokens: int,
    ) -> tuple[int, int]:
        """Validate retained allocator slots before native attention."""

        return self.coordinator.get_retained_residency_counts(
            request_id,
            logical_computed_tokens,
        )

    def plan_virtual_context_hbm_snapshot_pages(
        self,
        *,
        request_id: str,
        logical_tokens: int,
        max_pages: int,
        group_id: int = 0,
    ) -> tuple[int, ...]:
        """Select exact resident logical pages for a proactive D2H snapshot.

        Selection is allocator-only and mutation-free. The worker revalidates
        the physical block table when it materializes the source plan, while
        the later release commit independently verifies provider readiness and
        block identity.
        """

        if logical_tokens <= 0 or max_pages <= 0 or group_id < 0:
            return ()
        block_groups = self.coordinator.get_blocks(request_id)
        if group_id >= len(block_groups):
            return ()
        logical_page_count = (
            logical_tokens + self.hash_block_size - 1
        ) // self.hash_block_size
        return tuple(
            page
            for page, block in enumerate(block_groups[group_id])
            if page < logical_page_count and not block.is_null
        )[:max_pages]

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_virtual_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        block_counts_before: tuple[int, ...] | None = None
        if (
            self.virtual_context is not None
            and self.virtual_context_delta_hbm_observe
        ):
            block_counts_before = tuple(
                len(group)
                for group in self.coordinator.get_blocks(request.request_id)
            )

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_need_slot = min(
            total_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_virtual_external_computed_tokens=(
                num_virtual_external_computed_tokens
            ),
        )

        if num_blocks_to_allocate > 0:
            num_free_blocks = self.block_pool.get_num_free_blocks()
        else:
            num_free_blocks = 0
        if num_blocks_to_allocate > 0 and num_blocks_to_allocate > num_free_blocks:
            self.block_pool.record_tenant_kv_allocation_capacity_miss(
                num_blocks_to_allocate, num_free_blocks
            )
            # Cannot allocate new blocks
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                num_virtual_external_computed_tokens=(
                    num_virtual_external_computed_tokens
                ),
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens
        )

        if self.virtual_context is not None:
            observe_start = time.perf_counter()
            blocks_after = self.coordinator.get_blocks(request.request_id)
            block_counts_after = tuple(len(group) for group in blocks_after)
            has_new_computed_blocks = any(
                len(group) > 0 for group in new_computed_block_list
            )
            delta_safe = (
                self.virtual_context_delta_hbm_observe
                and block_counts_before is not None
                and len(block_counts_before) == len(block_counts_after)
                and len(new_blocks) == len(block_counts_after)
                and not has_new_computed_blocks
                and num_external_computed_tokens == 0
                and num_virtual_external_computed_tokens == 0
                and all(
                    after >= before
                    and after - before == len(allocated)
                    for before, after, allocated in zip(
                        block_counts_before,
                        block_counts_after,
                        new_blocks,
                    )
                )
            )
            if delta_safe:
                new_block_ids_by_group = tuple(
                    [block.block_id for block in group]
                    for group in new_blocks
                )
                new_page_start_by_group = tuple(
                    count - len(group)
                    for count, group in zip(
                        block_counts_after, new_blocks
                    )
                )
                self.virtual_context.observe_hbm_block_delta(
                    request_id=request.request_id,
                    new_block_ids_by_group=new_block_ids_by_group,
                    new_page_start_by_group=new_page_start_by_group,
                    block_size=self.hash_block_size,
                    logical_tokens=num_tokens_need_slot,
                    null_block_id=self.block_pool.null_block.block_id,
                )
                self._virtual_context_observe_hbm_delta_calls += 1
            else:
                self.virtual_context.observe_hbm_blocks(
                    request_id=request.request_id,
                    block_ids_by_group=tuple(
                        [block.block_id for block in group]
                        for group in blocks_after
                    ),
                    block_size=self.hash_block_size,
                    logical_tokens=num_tokens_need_slot,
                    null_block_id=self.block_pool.null_block.block_id,
                )
                self._virtual_context_observe_hbm_full_calls += 1
            self._virtual_context_observe_hbm_duration_s += (
                time.perf_counter() - observe_start
            )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def release_layerwise_batch(self, transfer, requests: list[Request]) -> bool:
        """Transfer whole tables, without resetting logical/output state.

        No mutation until EVERY row is validated. A shared-prefix ref acquired
        since the yield signal declines reclaim and keeps Native ownership.
        """
        if transfer.state != "copy_ready" or self.virtual_context is None:
            raise RuntimeError("layerwise release requires a durable private copy")
        rows = transfer.checkpoint.rows
        if tuple(req.request_id for req in requests) != tuple(row.request_id for row in rows):
            raise RuntimeError("layerwise allocator batch owner changed")
        physical = set()
        exclusive = True
        for row in rows:
            groups = self.get_blocks(row.request_id).blocks
            if len(groups) != 1 or tuple(block.block_id for block in groups[0]) != row.block_ids:
                raise RuntimeError("layerwise allocator mapping changed before release")
            for block in groups[0]:
                if block.is_null or block.block_id in physical:
                    raise RuntimeError("layerwise allocator invalid/shared batch mapping")
                physical.add(block.block_id)
                exclusive = exclusive and block.ref_cnt == 1
        if not exclusive:
            transfer.retain_native()
            return False
        # Submitted prefill can already have optimistic prefix hashes. A free
        # alone could let another request hit KV whose later layers never ran.
        self.block_pool.evict_blocks(physical)
        self.virtual_context.retire_layerwise_hbm(req.request_id for req in requests)
        for req in requests:
            self.free(req, release_virtual_context=False)
        transfer.commit_release(len(physical))
        return True

    def free(
        self,
        request: Request,
        release_virtual_context: bool = True,
    ) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
            release_virtual_context: Whether to release the request's virtual
                context sidecar state. Residency-deferred prefill keeps logical
                progress while freeing physical HBM blocks.
        """
        if release_virtual_context and self.virtual_context is not None:
            self.virtual_context.release_request(request.request_id)
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
