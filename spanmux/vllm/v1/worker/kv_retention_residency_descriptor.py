# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side block-table descriptors for residency-deferred requests."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVRetentionResidencyDescriptor:
    """Immutable allocator view retained after model-runner batch removal."""

    request_id: str
    block_ids: tuple[tuple[int, ...], ...]
    num_computed_tokens: int


class KVRetentionResidencyDescriptorStore:
    """Thread-safe descriptor ownership for phase-separated worker lanes.

    Residency deferral removes a request from ``GPUModelRunner.requests`` and
    its persistent input batch. Proactive D2H for that quiescent request still
    needs the last allocator block table, so this store keeps only that small,
    immutable descriptor until the request is rehydrated or finishes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._descriptors: dict[str, KVRetentionResidencyDescriptor] = {}
        self._capture_count = 0
        self._discard_count = 0
        self._lookup_count = 0
        self._hit_count = 0
        self._miss_count = 0

    def capture(
        self,
        *,
        request_id: str,
        block_ids: Sequence[Sequence[int]],
        num_computed_tokens: int,
    ) -> KVRetentionResidencyDescriptor:
        descriptor = KVRetentionResidencyDescriptor(
            request_id=request_id,
            block_ids=tuple(
                tuple(int(block_id) for block_id in group_block_ids)
                for group_block_ids in block_ids
            ),
            num_computed_tokens=max(0, int(num_computed_tokens)),
        )
        with self._lock:
            self._descriptors[request_id] = descriptor
            self._capture_count += 1
        return descriptor

    def lookup(
        self,
        request_id: str,
    ) -> KVRetentionResidencyDescriptor | None:
        with self._lock:
            self._lookup_count += 1
            descriptor = self._descriptors.get(request_id)
            if descriptor is None:
                self._miss_count += 1
            else:
                self._hit_count += 1
            return descriptor

    def discard(self, request_id: str) -> bool:
        with self._lock:
            removed = self._descriptors.pop(request_id, None) is not None
            if removed:
                self._discard_count += 1
            return removed

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "worker_deferred_residency_descriptor_active_count": len(
                    self._descriptors
                ),
                "worker_deferred_residency_descriptor_capture_count": (
                    self._capture_count
                ),
                "worker_deferred_residency_descriptor_discard_count": (
                    self._discard_count
                ),
                "worker_deferred_residency_descriptor_lookup_count": (
                    self._lookup_count
                ),
                "worker_deferred_residency_descriptor_hit_count": self._hit_count,
                "worker_deferred_residency_descriptor_miss_count": self._miss_count,
            }
