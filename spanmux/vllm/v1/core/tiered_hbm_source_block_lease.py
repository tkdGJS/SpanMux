# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local leases for HBM blocks used by asynchronous D2H copies.

The tiered KV prototype runs the scheduler and UniProc worker in one process.
An asynchronous HBM-to-DRAM copy can outlive the request that owned its source
block, so the block allocator must not recycle that physical block until the
copy and logical-page publication have completed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock

_TRUE_VALUES = {"1", "true", "yes", "on"}


def tiered_hbm_source_block_lease_enabled() -> bool:
    return (
        os.environ.get(
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE",
            "",
        ).lower()
        in _TRUE_VALUES
    )


@dataclass(frozen=True)
class TieredHBMSourceBlockLeaseStats:
    acquire_count: int
    release_count: int
    acquired_block_refs: int
    released_block_refs: int
    active_blocks: int
    active_block_refs: int
    peak_active_blocks: int
    peak_active_block_refs: int

    def to_metrics(self, prefix: str) -> dict[str, int]:
        return {
            f"{prefix}_enabled": int(tiered_hbm_source_block_lease_enabled()),
            f"{prefix}_acquire_count": self.acquire_count,
            f"{prefix}_release_count": self.release_count,
            f"{prefix}_acquired_block_refs": self.acquired_block_refs,
            f"{prefix}_released_block_refs": self.released_block_refs,
            f"{prefix}_active_blocks": self.active_blocks,
            f"{prefix}_active_block_refs": self.active_block_refs,
            f"{prefix}_peak_active_blocks": self.peak_active_blocks,
            f"{prefix}_peak_active_block_refs": self.peak_active_block_refs,
        }


class TieredHBMSourceBlockLease:
    """Idempotent lease over a unique set of physical HBM block ids."""

    def __init__(
        self,
        registry: _TieredHBMSourceBlockLeaseRegistry,
        block_ids: tuple[int, ...],
    ) -> None:
        self.block_ids = block_ids
        self._registry = registry
        self._lock = Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._registry.release(self.block_ids)

    def __enter__(self) -> TieredHBMSourceBlockLease:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class _TieredHBMSourceBlockLeaseRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._refcounts: dict[int, int] = {}
        self._acquire_count = 0
        self._release_count = 0
        self._acquired_block_refs = 0
        self._released_block_refs = 0
        self._peak_active_blocks = 0
        self._peak_active_block_refs = 0

    @staticmethod
    def _normalize(block_ids: Iterable[int]) -> tuple[int, ...]:
        normalized = sorted({int(block_id) for block_id in block_ids})
        if normalized and normalized[0] < 0:
            raise ValueError("HBM source block ids must be non-negative")
        return tuple(normalized)

    def acquire(self, block_ids: Iterable[int]) -> TieredHBMSourceBlockLease:
        normalized = self._normalize(block_ids)
        with self._lock:
            for block_id in normalized:
                self._refcounts[block_id] = self._refcounts.get(block_id, 0) + 1
            self._acquire_count += 1
            self._acquired_block_refs += len(normalized)
            active_refs = sum(self._refcounts.values())
            self._peak_active_blocks = max(
                self._peak_active_blocks,
                len(self._refcounts),
            )
            self._peak_active_block_refs = max(
                self._peak_active_block_refs,
                active_refs,
            )
        return TieredHBMSourceBlockLease(self, normalized)

    def release(self, block_ids: tuple[int, ...]) -> None:
        with self._lock:
            for block_id in block_ids:
                refcount = self._refcounts.get(block_id)
                if refcount is None:
                    raise RuntimeError(
                        f"HBM source block {block_id} was released without a lease"
                    )
                if refcount == 1:
                    self._refcounts.pop(block_id)
                else:
                    self._refcounts[block_id] = refcount - 1
            self._release_count += 1
            self._released_block_refs += len(block_ids)

    def leased_block_ids(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._refcounts)

    def stats(self) -> TieredHBMSourceBlockLeaseStats:
        with self._lock:
            return TieredHBMSourceBlockLeaseStats(
                acquire_count=self._acquire_count,
                release_count=self._release_count,
                acquired_block_refs=self._acquired_block_refs,
                released_block_refs=self._released_block_refs,
                active_blocks=len(self._refcounts),
                active_block_refs=sum(self._refcounts.values()),
                peak_active_blocks=self._peak_active_blocks,
                peak_active_block_refs=self._peak_active_block_refs,
            )

    def clear_for_test(self) -> None:
        with self._lock:
            if self._refcounts:
                raise RuntimeError("cannot clear active HBM source block leases")
            self._acquire_count = 0
            self._release_count = 0
            self._acquired_block_refs = 0
            self._released_block_refs = 0
            self._peak_active_blocks = 0
            self._peak_active_block_refs = 0


_SOURCE_BLOCK_LEASE_REGISTRY = _TieredHBMSourceBlockLeaseRegistry()


def acquire_tiered_hbm_source_block_lease(
    block_ids: Iterable[int],
) -> TieredHBMSourceBlockLease:
    return _SOURCE_BLOCK_LEASE_REGISTRY.acquire(block_ids)


def get_tiered_hbm_source_leased_block_ids() -> frozenset[int]:
    return _SOURCE_BLOCK_LEASE_REGISTRY.leased_block_ids()


def get_tiered_hbm_source_block_lease_stats() -> TieredHBMSourceBlockLeaseStats:
    return _SOURCE_BLOCK_LEASE_REGISTRY.stats()


def clear_tiered_hbm_source_block_leases_for_test() -> None:
    _SOURCE_BLOCK_LEASE_REGISTRY.clear_for_test()
