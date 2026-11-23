# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-independent HBM source-page plans for tiered KV snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _as_int(value: Any) -> int:
    return int(value.item() if hasattr(value, "item") else value)


@dataclass(frozen=True)
class TieredD2HSourcePage:
    request_id: str
    request_index: int
    logical_page: int
    block_id: int
    token_start: int
    token_count: int
    force_all: bool


@dataclass(frozen=True)
class TieredD2HSourcePlan:
    """Exact HBM blocks selected once for all attention layers in a group."""

    request_ids: tuple[str, ...]
    eligible_request_ids: tuple[str, ...]
    force_all_request_ids: tuple[str, ...]
    block_size: int
    pages: tuple[TieredD2HSourcePage, ...]

    def for_request_ids(self, request_ids: Sequence[str]) -> "TieredD2HSourcePlan":
        selected = set(request_ids)
        return TieredD2HSourcePlan(
            request_ids=tuple(request_ids),
            eligible_request_ids=tuple(
                request_id
                for request_id in self.eligible_request_ids
                if request_id in selected
            ),
            force_all_request_ids=tuple(
                request_id
                for request_id in self.force_all_request_ids
                if request_id in selected
            ),
            block_size=self.block_size,
            pages=tuple(
                page for page in self.pages if page.request_id in selected
            ),
        )

    def matches(
        self,
        request_ids: Sequence[str],
        block_size: int,
        allowed_request_ids: set[str] | None = None,
        force_all_request_ids: set[str] | None = None,
    ) -> bool:
        expected_eligible = (
            set(request_ids)
            if allowed_request_ids is None
            else set(allowed_request_ids)
        )
        return (
            self.block_size == block_size
            and set(self.request_ids) == set(request_ids)
            and set(self.eligible_request_ids) == expected_eligible
            and set(self.force_all_request_ids)
            == set(force_all_request_ids or ())
        )

    def group_by_request(self) -> dict[str, tuple[TieredD2HSourcePage, ...]]:
        grouped: dict[str, list[TieredD2HSourcePage]] = {}
        for page in self.pages:
            grouped.setdefault(page.request_id, []).append(page)
        return {
            request_id: tuple(pages)
            for request_id, pages in grouped.items()
        }


def build_tiered_d2h_source_plan(
    *,
    request_ids: Sequence[str],
    block_table: Any,
    seq_lens: Sequence[Any],
    query_start_loc: Sequence[Any],
    block_size: int,
    copy_policy: str,
    allowed_request_ids: set[str] | None,
    force_all_request_ids: set[str] | None,
    max_pages: int,
    exact_force_pages_by_request: dict[str, set[int]] | None = None,
) -> TieredD2HSourcePlan:
    """Build a CPU-only page plan without importing torch.

    The plan contains physical HBM block ids but no layer version. Each layer
    resolves its own logical-page version immediately before it captures the
    HBM tensor view and publishes the existing overwrite fence.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    normalized_policy = copy_policy.strip().lower()
    if normalized_policy not in {"scheduled", "all"}:
        normalized_policy = "scheduled"

    num_reqs = min(
        len(request_ids),
        len(seq_lens),
        max(0, len(query_start_loc) - 1),
        int(block_table.shape[0]),
    )
    table_width = int(block_table.shape[1])
    pages: list[TieredD2HSourcePage] = []
    force_ids = force_all_request_ids or set()
    exact_force_pages = exact_force_pages_by_request or {}
    eligible_ids = (
        set(request_ids[:num_reqs])
        if allowed_request_ids is None
        else set(allowed_request_ids).intersection(request_ids[:num_reqs])
    )

    for request_index in range(num_reqs):
        request_id = request_ids[request_index]
        if (
            request_id not in eligible_ids
        ):
            continue
        seq_len = _as_int(seq_lens[request_index])
        if seq_len <= 0:
            continue
        force_all = request_id in force_ids
        if force_all or normalized_policy == "all":
            first_page = 0
        else:
            query_start = _as_int(query_start_loc[request_index])
            query_end = _as_int(query_start_loc[request_index + 1])
            query_len = max(0, query_end - query_start)
            if query_len <= 0:
                continue
            first_page = max(0, seq_len - query_len) // block_size
        last_page = min(
            (seq_len - 1) // block_size,
            table_width - 1,
        )
        if last_page < first_page:
            continue
        for logical_page in range(first_page, last_page + 1):
            if max_pages > 0 and len(pages) >= max_pages:
                break
            request_exact_pages = exact_force_pages.get(request_id)
            if (
                request_id in force_ids
                and request_exact_pages is not None
                and logical_page not in request_exact_pages
            ):
                continue
            block_id = _as_int(block_table[request_index, logical_page])
            if block_id < 0:
                continue
            token_start = logical_page * block_size
            pages.append(
                TieredD2HSourcePage(
                    request_id=request_id,
                    request_index=request_index,
                    logical_page=logical_page,
                    block_id=block_id,
                    token_start=token_start,
                    token_count=min(block_size, seq_len - token_start),
                    force_all=force_all,
                )
            )
        if max_pages > 0 and len(pages) >= max_pages:
            break

    return TieredD2HSourcePlan(
        request_ids=tuple(request_ids[:num_reqs]),
        eligible_request_ids=tuple(
            request_id
            for request_id in request_ids[:num_reqs]
            if request_id in eligible_ids
        ),
        force_all_request_ids=tuple(
            request_id
            for request_id in request_ids[:num_reqs]
            if request_id in force_ids
        ),
        block_size=block_size,
        pages=tuple(pages),
    )
