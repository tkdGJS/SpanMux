# SPDX-License-Identifier: Apache-2.0
"""Tensor-free authority for a whole-batch layerwise KV transfer.

The checkpoint describes bytes; this receipt proves who may release them.
GPU storage lives only in the worker. This module is also CPU-testable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator

from vllm.model_executor.layers.ttft_layerwise_prefill import (
    KVRetentionLayerwiseCheckpoint,
    TTFTLayerwisePrefillBoundary,
    TTFTLayerwisePrefillError,
    _kv_checkpoint_boundary_identity,
)


def kv_retention_layerwise_offload_enabled() -> bool:
    return os.environ.get("VLLM_KV_RETENTION_LAYERWISE_OFFLOAD", "0").lower() in {
        "1", "true", "yes", "on",
    }


@dataclass
class KVRetentionLayerwiseTransfer:
    """One copy receipt shared by the single-process Engine and worker.

    State changes happen at existing Future completion/submission boundaries.
    No tensor, CUDA event, or callback is retained by this public ticket.
    """

    checkpoint: KVRetentionLayerwiseCheckpoint
    copied_bytes: int
    state: str = field(default="copy_ready", init=False)
    released_blocks: int = field(default=0, init=False)
    current_identity: tuple[Any, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.copied_bytes < 0:
            raise TTFTLayerwisePrefillError("negative layerwise copy size")
        rows = sorted(self.checkpoint.rows, key=lambda row: row.input_row)
        if [row.input_row for row in rows] != list(range(len(rows))):
            raise TTFTLayerwisePrefillError("offload requires a dense whole batch")
        ids = [block for row in rows for block in row.block_ids]
        if len(set(ids)) != len(ids):
            raise TTFTLayerwisePrefillError("offload cannot release shared KV rows")
        self.current_identity = self.checkpoint.boundary_identity

    def validate_output(self, output: Any) -> None:
        actual = (
            output.scheduler_iteration_id, output.pd_phase_submit_seq,
            output.pd_phase_slot, tuple(output.num_scheduled_tokens),
            tuple(output.num_scheduled_tokens.items()), output.pd_prefill_mode,
        )
        if actual != self.checkpoint.boundary_identity[:6]:
            raise TTFTLayerwisePrefillError("layerwise transfer output owner changed")

    def validate_boundary(self, boundary: TTFTLayerwisePrefillBoundary) -> None:
        if _kv_checkpoint_boundary_identity(boundary) != self.current_identity:
            raise TTFTLayerwisePrefillError("layerwise transfer boundary changed")

    def commit_release(self, blocks: int) -> None:
        expected = sum(len(row.block_ids) for row in self.checkpoint.rows)
        if self.state != "copy_ready" or blocks != expected:
            raise TTFTLayerwisePrefillError("duplicate or incomplete KV release")
        self.released_blocks = blocks
        self.state = "released"

    def require_ring(self) -> None:
        if self.state != "released":
            raise TTFTLayerwisePrefillError("Ring resume requires committed KV release")

    def retain_native(self) -> None:
        if self.state != "copy_ready":
            raise TTFTLayerwisePrefillError("cannot retain Native KV after release")
        self.state = "retained"

    def advance(self, boundary: TTFTLayerwisePrefillBoundary) -> None:
        self.require_ring()
        identity = _kv_checkpoint_boundary_identity(boundary)
        old = self.current_identity
        if (identity[:6] != old[:6] or identity[-2] != old[-2]
                or identity[-1] != old[-1] + 1
                or not old[-3] < identity[-3] < identity[-2]):
            raise TTFTLayerwisePrefillError("invalid Ring continuation successor")
        self.current_identity = identity

    def mark_published(self, completed_layers: int) -> None:
        self.require_ring()
        if completed_layers != len(self.checkpoint.layer_names):
            raise TTFTLayerwisePrefillError("cannot publish incomplete layer KV")
        self.state = "published"

    def mark_consumed(self) -> None:
        if self.state != "published":
            raise TTFTLayerwisePrefillError("external resume before final KV publication")
        self.state = "consumed"

    def debug(self) -> dict[str, float]:
        return {
            "kv_retention_offload_copy_ready": 1.0,
            "kv_retention_offload_bytes": float(self.copied_bytes),
            "kv_retention_offload_released_blocks": float(self.released_blocks),
            "kv_retention_offload_ring_active": float(self.state == "released"),
            "kv_retention_offload_published": float(
                self.state in {"published", "consumed"}
            ),
        }


_RING_OWNER: ContextVar[Any | None] = ContextVar("kv_retention_ring_owner", default=None)


def active_kv_retention_ring() -> Any | None:
    return _RING_OWNER.get()


@contextmanager
def kv_retention_ring_scope(owner: Any | None) -> Iterator[None]:
    if owner is not None:
        owner.ticket.require_ring()
    token = _RING_OWNER.set(owner)
    try:
        yield
    finally:
        _RING_OWNER.reset(token)


_HOST_LOCK = Lock()
_HOST_BYTES = 0


@dataclass
class KVRetentionHostReservation:
    size: int
    released: bool = False

    def release(self) -> None:
        global _HOST_BYTES
        with _HOST_LOCK:
            if not self.released:
                _HOST_BYTES -= self.size
                self.released = True


def reserve_kv_retention_host_bytes(size: int) -> KVRetentionHostReservation | None:
    """Bound all simultaneously suspended private buffers, not each request."""
    global _HOST_BYTES
    limit = int(os.environ.get("VLLM_KV_RETENTION_LAYERWISE_HOST_BYTES", 4 << 30))
    if size < 0 or limit <= 0:
        raise TTFTLayerwisePrefillError("invalid layerwise private host budget")
    with _HOST_LOCK:
        if _HOST_BYTES + size > limit:
            return None
        _HOST_BYTES += size
    return KVRetentionHostReservation(size)
