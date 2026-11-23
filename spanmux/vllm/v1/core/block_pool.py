# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.tiered_hbm_source_block_lease import (
    get_tiered_hbm_source_block_lease_stats,
    get_tiered_hbm_source_leased_block_ids,
    tiered_hbm_source_block_lease_enabled,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector
        self.tenant_kv_min_blocks = self._read_tenant_kv_min_blocks()
        self.tenant_kv_protection_mode = self._read_tenant_kv_protection_mode()
        self._tenant_kv_access_seq = 0
        self._tenant_prefix_chains: dict[str, list[BlockHashWithGroupId | None]] = {}
        self._tenant_kv_debug_counters: dict[str, float] = {}
        self._protected_free_blocks: dict[int, KVCacheBlock] = {}
        self._protected_position_to_block_id: dict[tuple[str, int], int] = {}
        self._tiered_hbm_source_lease_enabled = (
            tiered_hbm_source_block_lease_enabled()
        )
        self._tiered_hbm_source_lease_deferred_blocks: dict[
            int, KVCacheBlock
        ] = {}
        if self._tiered_hbm_source_lease_enabled:
            logger.info(
                "Process-local async D2H HBM source-block leases enabled"
            )

    def _read_tenant_kv_min_blocks(self) -> int:
        raw_value = os.environ.get("VLLM_TENANT_KV_MIN_BLOCKS", "0")
        try:
            value = int(raw_value)
        except ValueError:
            raise ValueError(
                "VLLM_TENANT_KV_MIN_BLOCKS must be a non-negative integer, "
                f"got {raw_value!r}"
            ) from None
        if value < 0:
            raise ValueError(
                "VLLM_TENANT_KV_MIN_BLOCKS must be a non-negative integer, "
                f"got {raw_value!r}"
            )
        if value > 0:
            logger.info(
                "Tenant KV minimum block protection enabled: "
                "tenant_kv_min_blocks=%d",
                value,
            )
        return value

    def _read_tenant_kv_protection_mode(self) -> str:
        mode = os.environ.get("VLLM_TENANT_KV_PROTECTION_MODE", "scan").lower()
        if mode not in ("scan", "indexed"):
            raise ValueError(
                "VLLM_TENANT_KV_PROTECTION_MODE must be 'scan' or 'indexed', "
                f"got {mode!r}"
            )
        if self.tenant_kv_min_blocks > 0:
            logger.info("Tenant KV protection mode: %s", mode)
        return mode

    def _use_indexed_tenant_kv_protection(self) -> bool:
        return (
            self.tenant_kv_min_blocks > 0
            and self.tenant_kv_protection_mode == "indexed"
        )

    def _record_tenant_access(
        self,
        block: KVCacheBlock,
        tenant_id: str | None = None,
        tenant_block_index: int | None = None,
        tenant_chain_key: str | None = None,
    ) -> None:
        if self.tenant_kv_min_blocks <= 0 or block.is_null:
            return
        if tenant_id is not None:
            block.tenant_id = tenant_id
        if tenant_chain_key is not None:
            block.tenant_chain_key = tenant_chain_key
        if tenant_block_index is not None:
            block.tenant_block_index = tenant_block_index
        self._tenant_kv_access_seq += 1
        block.last_access_seq = self._tenant_kv_access_seq

    def _get_tenant_chain_key(self, request: Request) -> str:
        session_id = getattr(request, "tenant_session_id", request.tenant_id)
        return f"{request.tenant_id}\0{session_id}"

    def _record_tenant_prefix_hash(
        self,
        tenant_chain_key: str,
        tenant_block_index: int,
        block_hash: BlockHashWithGroupId,
    ) -> None:
        if self.tenant_kv_min_blocks <= 0:
            return
        chain = self._tenant_prefix_chains.setdefault(tenant_chain_key, [])
        if len(chain) <= tenant_block_index:
            chain.extend([None] * (tenant_block_index + 1 - len(chain)))
        chain[tenant_block_index] = block_hash

    def _add_tenant_kv_debug_counter(self, key: str, value: float) -> None:
        self._tenant_kv_debug_counters[key] = (
            self._tenant_kv_debug_counters.get(key, 0.0) + value
        )

    def _is_indexed_protection_candidate(self, block: KVCacheBlock) -> bool:
        return (
            self._use_indexed_tenant_kv_protection()
            and block.block_hash is not None
            and block.tenant_chain_key is not None
            and block.tenant_block_index is not None
            and block.tenant_block_index < self.tenant_kv_min_blocks
        )

    def _unprotect_free_block(self, block: KVCacheBlock) -> None:
        if block.block_id not in self._protected_free_blocks:
            return
        self._protected_free_blocks.pop(block.block_id, None)
        if block.tenant_chain_key is not None and block.tenant_block_index is not None:
            position = (block.tenant_chain_key, block.tenant_block_index)
            if self._protected_position_to_block_id.get(position) == block.block_id:
                self._protected_position_to_block_id.pop(position, None)

    def _append_evictable_free_blocks(self, blocks: list[KVCacheBlock]) -> None:
        if blocks:
            self.free_block_queue.append_n(blocks)

    def _append_source_lease_released_blocks(
        self, blocks: list[KVCacheBlock]
    ) -> None:
        if self._use_indexed_tenant_kv_protection():
            for block in blocks:
                self._protect_free_block_or_append_evictable(block)
        else:
            self._append_evictable_free_blocks(blocks)

    def _drain_source_lease_deferred_blocks(self) -> int:
        if (
            not self._tiered_hbm_source_lease_enabled
            or not self._tiered_hbm_source_lease_deferred_blocks
        ):
            return 0
        leased_block_ids = get_tiered_hbm_source_leased_block_ids()
        released_blocks: list[KVCacheBlock] = []
        for block_id, block in tuple(
            self._tiered_hbm_source_lease_deferred_blocks.items()
        ):
            if block_id in leased_block_ids:
                continue
            self._tiered_hbm_source_lease_deferred_blocks.pop(block_id)
            if block.ref_cnt == 0 and not block.is_null:
                released_blocks.append(block)
        self._append_source_lease_released_blocks(released_blocks)
        if released_blocks:
            self._add_tenant_kv_debug_counter(
                "kv_retention_d2h_source_lease_drained_free_blocks",
                float(len(released_blocks)),
            )
        return len(released_blocks)

    def _protect_free_block_or_append_evictable(self, block: KVCacheBlock) -> None:
        if not self._is_indexed_protection_candidate(block):
            self._append_evictable_free_blocks([block])
            return

        assert block.tenant_chain_key is not None
        assert block.tenant_block_index is not None
        position = (block.tenant_chain_key, block.tenant_block_index)
        existing_block_id = self._protected_position_to_block_id.get(position)
        if existing_block_id is None:
            self._protected_position_to_block_id[position] = block.block_id
            self._protected_free_blocks[block.block_id] = block
            return

        existing_block = self._protected_free_blocks.get(existing_block_id)
        if existing_block is None:
            self._protected_position_to_block_id[position] = block.block_id
            self._protected_free_blocks[block.block_id] = block
            return

        if block.last_access_seq > existing_block.last_access_seq:
            self._protected_free_blocks.pop(existing_block.block_id, None)
            self._protected_position_to_block_id[position] = block.block_id
            self._protected_free_blocks[block.block_id] = block
            self._append_evictable_free_blocks([existing_block])
        else:
            self._append_evictable_free_blocks([block])

    def _get_protected_free_block_ids(self, count_debug: bool = True) -> set[int]:
        if self.tenant_kv_min_blocks <= 0:
            return set()

        scan_start_ts = time.monotonic()
        free_cached_blocks_by_hash: dict[BlockHashWithGroupId, list[KVCacheBlock]] = {}
        free_blocks = self.free_block_queue.get_all_free_blocks()
        scanned_free_blocks = 0
        for block in free_blocks:
            scanned_free_blocks += 1
            if block.is_null or block.block_hash is None:
                continue
            free_cached_blocks_by_hash.setdefault(block.block_hash, []).append(block)

        protected_block_ids: set[int] = set()
        scanned_chain_positions = 0
        for chain in self._tenant_prefix_chains.values():
            num_protected_positions = 0
            for block_hash in chain:
                scanned_chain_positions += 1
                if num_protected_positions >= self.tenant_kv_min_blocks:
                    break
                if block_hash is None:
                    continue
                blocks = free_cached_blocks_by_hash.get(block_hash)
                if not blocks:
                    continue
                block = max(blocks, key=lambda candidate: candidate.last_access_seq)
                protected_block_ids.add(block.block_id)
                num_protected_positions += 1
        if count_debug:
            self._add_tenant_kv_debug_counter(
                "tenant_kv_protected_scan_calls", 1.0
            )
            self._add_tenant_kv_debug_counter(
                "tenant_kv_protected_scan_duration_s",
                time.monotonic() - scan_start_ts,
            )
            self._add_tenant_kv_debug_counter(
                "tenant_kv_free_blocks_scanned", float(scanned_free_blocks)
            )
            self._add_tenant_kv_debug_counter(
                "tenant_kv_chain_positions_scanned",
                float(scanned_chain_positions),
            )
            self._add_tenant_kv_debug_counter(
                "tenant_kv_protected_blocks_selected",
                float(len(protected_block_ids)),
            )
        return protected_block_ids

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        tenant_chain_key = self._get_tenant_chain_key(request)
        num_tenant_prefix_blocks = min(num_full_blocks, self.tenant_kv_min_blocks)
        for tenant_block_index in range(num_tenant_prefix_blocks):
            self._record_tenant_prefix_hash(
                tenant_chain_key,
                tenant_block_index,
                make_block_hash_with_group_id(
                    block_hashes[tenant_block_index], kv_cache_group_id
                ),
            )

        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention. We skip null blocks here.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            tenant_block_index = num_cached_blocks + i
            blk.block_hash = block_hash_with_group_id
            self._record_tenant_access(
                blk,
                request.tenant_id,
                tenant_block_index,
                tenant_chain_key,
            )
            if tenant_block_index < self.tenant_kv_min_blocks:
                self._record_tenant_prefix_hash(
                    tenant_chain_key,
                    tenant_block_index,
                    block_hash_with_group_id,
                )
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size : num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=(
                        request.lora_request.adapter_id
                        if request.lora_request
                        else None
                    ),
                    medium=MEDIUM_GPU,
                    lora_name=(
                        request.lora_request.name if request.lora_request else None
                    ),
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        get_new_start_ts = time.monotonic()
        self._drain_source_lease_deferred_blocks()
        self._add_tenant_kv_debug_counter("tenant_kv_get_new_blocks_calls", 1.0)
        self._add_tenant_kv_debug_counter(
            "tenant_kv_get_new_blocks_requested", float(num_blocks)
        )
        available_blocks = self.get_num_free_blocks()
        if num_blocks > available_blocks:
            self._add_tenant_kv_debug_counter(
                "tenant_kv_get_new_blocks_failures", 1.0
            )
            self._add_tenant_kv_debug_counter(
                "tenant_kv_get_new_blocks_shortfall",
                float(num_blocks - available_blocks),
            )
            self._add_tenant_kv_debug_counter(
                "tenant_kv_get_new_blocks_duration_s",
                time.monotonic() - get_new_start_ts,
            )
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        if self.tenant_kv_min_blocks <= 0 or self._use_indexed_tenant_kv_protection():
            ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        else:
            protected_block_ids = self._get_protected_free_block_ids()
            ret = [
                block
                for block in self.free_block_queue.get_all_free_blocks()
                if block.block_id not in protected_block_ids
            ][:num_blocks]
            if len(ret) < num_blocks:
                self._add_tenant_kv_debug_counter(
                    "tenant_kv_get_new_blocks_failures", 1.0
                )
                self._add_tenant_kv_debug_counter(
                    "tenant_kv_get_new_blocks_shortfall",
                    float(num_blocks - len(ret)),
                )
                self._add_tenant_kv_debug_counter(
                    "tenant_kv_get_new_blocks_duration_s",
                    time.monotonic() - get_new_start_ts,
                )
                raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
            for block in ret:
                self.free_block_queue.remove(block)

        evicted_blocks = 0
        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                evicted_blocks += int(self._maybe_evict_cached_block(block))
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        self._add_tenant_kv_debug_counter(
            "tenant_kv_get_new_blocks_allocated", float(len(ret))
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_evicted_on_allocation", float(evicted_blocks)
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_get_new_blocks_duration_s",
            time.monotonic() - get_new_start_ts,
        )
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                if (
                    block.block_id
                    in self._tiered_hbm_source_lease_deferred_blocks
                ):
                    self._tiered_hbm_source_lease_deferred_blocks.pop(
                        block.block_id
                    )
                    self._add_tenant_kv_debug_counter(
                        "kv_retention_d2h_source_lease_deferred_touch_count",
                        1.0,
                    )
                elif block.block_id in self._protected_free_blocks:
                    self._unprotect_free_block(block)
                else:
                    self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)
            self._record_tenant_access(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        free_start_ts = time.monotonic()
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        freed_blocks = [
            block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
        ]
        deferred_blocks: list[KVCacheBlock] = []
        if self._tiered_hbm_source_lease_enabled and freed_blocks:
            leased_block_ids = get_tiered_hbm_source_leased_block_ids()
            immediately_freeable: list[KVCacheBlock] = []
            for block in freed_blocks:
                if block.block_id in leased_block_ids:
                    if (
                        block.block_id
                        in self._tiered_hbm_source_lease_deferred_blocks
                    ):
                        raise RuntimeError(
                            "HBM source block entered deferred free twice: "
                            f"block_id={block.block_id}"
                        )
                    self._tiered_hbm_source_lease_deferred_blocks[
                        block.block_id
                    ] = block
                    deferred_blocks.append(block)
                else:
                    immediately_freeable.append(block)
            freed_blocks = immediately_freeable
        self._append_source_lease_released_blocks(freed_blocks)
        self._add_tenant_kv_debug_counter("tenant_kv_free_blocks_calls", 1.0)
        self._add_tenant_kv_debug_counter(
            "tenant_kv_free_blocks_input", float(len(blocks_list))
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_free_blocks_appended", float(len(freed_blocks))
        )
        self._add_tenant_kv_debug_counter(
            "kv_retention_d2h_source_lease_deferred_free_blocks",
            float(len(deferred_blocks)),
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_free_blocks_duration_s", time.monotonic() - free_start_ts
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        evict_start_ts = time.monotonic()
        evicted_blocks = 0
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            was_protected_free = block.block_id in self._protected_free_blocks
            if was_protected_free:
                self._unprotect_free_block(block)
            evicted_blocks += int(self._maybe_evict_cached_block(block))
            if was_protected_free and block.ref_cnt == 0 and not block.is_null:
                self._append_evictable_free_blocks([block])
        self._add_tenant_kv_debug_counter("tenant_kv_evict_blocks_calls", 1.0)
        self._add_tenant_kv_debug_counter(
            "tenant_kv_evict_blocks_requested", float(len(block_ids))
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_evict_blocks_evicted", float(evicted_blocks)
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_evict_blocks_duration_s", time.monotonic() - evict_start_ts
        )

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        self._drain_source_lease_deferred_blocks()
        if self._use_indexed_tenant_kv_protection():
            num_free_blocks = (
                self.free_block_queue.num_free_blocks
                + len(self._protected_free_blocks)
            )
        else:
            num_free_blocks = self.get_num_free_blocks()
        num_used_blocks = self.num_gpu_blocks - num_free_blocks
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self._tenant_prefix_chains.clear()
        protected_free_blocks = list(self._protected_free_blocks.values())
        self._append_evictable_free_blocks(protected_free_blocks)
        self._protected_free_blocks.clear()
        self._protected_position_to_block_id.clear()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        get_free_start_ts = time.monotonic()
        self._drain_source_lease_deferred_blocks()
        self._add_tenant_kv_debug_counter("tenant_kv_get_num_free_calls", 1.0)
        if self.tenant_kv_min_blocks <= 0 or self._use_indexed_tenant_kv_protection():
            num_free_blocks = self.free_block_queue.num_free_blocks
        else:
            protected_block_ids = self._get_protected_free_block_ids()
            num_free_blocks = sum(
                1
                for block in self.free_block_queue.get_all_free_blocks()
                if block.block_id not in protected_block_ids
            )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_get_num_free_duration_s",
            time.monotonic() - get_free_start_ts,
        )
        return num_free_blocks

    def record_tenant_kv_allocation_capacity_miss(
        self, requested_blocks: int, available_blocks: int
    ) -> None:
        self._add_tenant_kv_debug_counter(
            "tenant_kv_allocation_capacity_misses", 1.0
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_allocation_requested_blocks", float(requested_blocks)
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_allocation_available_blocks", float(available_blocks)
        )
        self._add_tenant_kv_debug_counter(
            "tenant_kv_allocation_shortfall_blocks",
            float(max(0, requested_blocks - available_blocks)),
        )

    def get_tenant_kv_debug_stats(self) -> dict[str, int | float]:
        """Return experimental tenant KV retention counters."""
        self._drain_source_lease_deferred_blocks()
        free_blocks = self.free_block_queue.get_all_free_blocks()
        if self._use_indexed_tenant_kv_protection():
            protected_block_ids = set(self._protected_free_blocks)
        else:
            protected_block_ids = (
                self._get_protected_free_block_ids(count_debug=False)
                if self.tenant_kv_min_blocks > 0
                else set()
            )
        cached_free_blocks = [
            block for block in free_blocks if block.block_hash is not None
        ]
        protected_cached_blocks = [
            block for block in self._protected_free_blocks.values()
            if block.block_hash is not None
        ]
        per_tenant_protected_blocks: dict[str, int] = {}
        for block in protected_cached_blocks:
            tenant_id = block.tenant_id or "unknown"
            per_tenant_protected_blocks[tenant_id] = (
                per_tenant_protected_blocks.get(tenant_id, 0) + 1
            )
        per_tenant_running_blocks: dict[str, int] = {}
        for block in self.blocks:
            if block.ref_cnt <= 0:
                continue
            tenant_id = block.tenant_id or "unknown"
            per_tenant_running_blocks[tenant_id] = (
                per_tenant_running_blocks.get(tenant_id, 0) + 1
            )
        counters = self._tenant_kv_debug_counters
        self._tenant_kv_debug_counters = {}
        num_free_blocks_raw = (
            self.free_block_queue.num_free_blocks + len(self._protected_free_blocks)
        )
        num_free_blocks_effective = (
            self.free_block_queue.num_free_blocks
            if self._use_indexed_tenant_kv_protection()
            else (
                sum(
                    1
                    for block in free_blocks
                    if block.block_id not in protected_block_ids
                )
                if self.tenant_kv_min_blocks > 0
                else self.free_block_queue.num_free_blocks
            )
        )
        num_protected_blocks = len(protected_block_ids)
        num_cached_free_blocks = len(cached_free_blocks) + len(protected_cached_blocks)
        num_evictable_cached_blocks = sum(
            1
            for block in cached_free_blocks
            if block.block_id not in protected_block_ids
        )
        kv_allocation_shortfall_blocks = counters.get(
            "tenant_kv_allocation_shortfall_blocks",
            counters.get("tenant_kv_get_new_blocks_shortfall", 0.0),
        )
        source_lease_stats = get_tiered_hbm_source_block_lease_stats()
        return {
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_free_blocks_raw": num_free_blocks_raw,
            "num_free_blocks_effective": num_free_blocks_effective,
            "num_protected_blocks": num_protected_blocks,
            "num_cached_free_blocks": num_cached_free_blocks,
            "num_evictable_cached_blocks": num_evictable_cached_blocks,
            "kv_total_blocks": self.num_gpu_blocks,
            "kv_free_blocks": num_free_blocks_effective,
            "kv_used_blocks": self.num_gpu_blocks - num_free_blocks_raw,
            "kv_cached_blocks": num_cached_free_blocks,
            "kv_evictable_cached_blocks": num_evictable_cached_blocks,
            "kv_protected_blocks": num_protected_blocks,
            "kv_elastic_blocks": 0,
            "kv_running_pinned_blocks": self.num_gpu_blocks - num_free_blocks_raw,
            "kv_effective_admission_blocks": num_free_blocks_effective,
            "kv_requested_blocks_this_iter": counters.get(
                "tenant_kv_get_new_blocks_requested", 0.0
            ),
            "kv_allocated_blocks_this_iter": counters.get(
                "tenant_kv_get_new_blocks_allocated", 0.0
            ),
            "kv_allocation_shortfall_blocks": kv_allocation_shortfall_blocks,
            "kv_evicted_blocks_this_iter": counters.get(
                "tenant_kv_evicted_on_allocation", 0.0
            ),
            "per_tenant_kv_protected_blocks": per_tenant_protected_blocks,
            "per_tenant_kv_elastic_blocks": {},
            "per_tenant_kv_running_blocks": per_tenant_running_blocks,
            "per_tenant_kv_evicted_blocks": {},
            "per_tenant_kv_reused_blocks": {},
            "tenant_kv_min_blocks": self.tenant_kv_min_blocks,
            "tenant_kv_protection_mode": self.tenant_kv_protection_mode,
            "kv_retention_d2h_source_lease_deferred_blocks": len(
                self._tiered_hbm_source_lease_deferred_blocks
            ),
            **source_lease_stats.to_metrics(
                "kv_retention_d2h_source_lease"
            ),
            **counters,
        }

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
