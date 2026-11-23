# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for experimental virtual-context residency policies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


class ResidencyVictimPolicy(str, Enum):
    """Capacity-floor victim ordering policies."""

    LONGEST_LOGICAL_CONTEXT_FIRST = "longest_logical_context_first"
    MIN_RING_CHUNKS = "min_ring_chunks"
    JIT_MIN_COPY = "jit_min_copy"
    SCHEDULED_PREFILL_FIT = "scheduled_prefill_fit"

    @classmethod
    def parse(cls, value: str) -> "ResidencyVictimPolicy":
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "": cls.LONGEST_LOGICAL_CONTEXT_FIRST,
            "longest": cls.LONGEST_LOGICAL_CONTEXT_FIRST,
            "longest_context": cls.LONGEST_LOGICAL_CONTEXT_FIRST,
            "longest_logical_context": cls.LONGEST_LOGICAL_CONTEXT_FIRST,
            "longest_logical_context_first": (
                cls.LONGEST_LOGICAL_CONTEXT_FIRST
            ),
            "min_ring_chunk": cls.MIN_RING_CHUNKS,
            "min_ring_chunks": cls.MIN_RING_CHUNKS,
            "jit_min_copy": cls.JIT_MIN_COPY,
            "jit_min_copies": cls.JIT_MIN_COPY,
            "scheduled_prefill_fit": cls.SCHEDULED_PREFILL_FIT,
            "next_prefill_fit": cls.SCHEDULED_PREFILL_FIT,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(policy.value for policy in cls)
            raise ValueError(
                f"unknown residency victim policy {value!r}; "
                f"expected one of: {choices}"
            ) from exc


@dataclass(frozen=True)
class ResidencyVictimCandidate:
    """Immutable scheduler snapshot used to rank one residency victim."""

    request_id: str
    arrival_time: float
    prompt_tokens: int
    logical_context_tokens: int
    estimated_reclaimable_blocks: int
    estimated_d2h_eviction_tokens: int = 0
    estimated_ring_staging_h2d_tokens: int = 0
    estimated_decode_reload_h2d_tokens: int = 0

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.logical_context_tokens)

    @property
    def estimated_total_copy_tokens(self) -> int:
        """Return cumulative KV-copy work in token-equivalent units.

        Every KV token has the same byte footprint for one fixed model, dtype,
        and cache layout.  Ranking by copied KV tokens is therefore equivalent
        to ranking by copied bytes without coupling the scheduler policy to a
        worker-specific tensor layout.
        """

        return sum(
            max(0, int(value))
            for value in (
                self.estimated_d2h_eviction_tokens,
                self.estimated_ring_staging_h2d_tokens,
                self.estimated_decode_reload_h2d_tokens,
            )
        )


@dataclass(frozen=True)
class PendingResidencyVictimIntent:
    """Deferred victim decision while the selected request owns a future.

    The scheduler may rank a request whose current GPU submission has not yet
    been recovered, but it must not mutate that request's KV ownership.  This
    immutable intent carries only the observation needed to re-run the policy
    at a later scheduler boundary; it is never permission to commit eviction.
    """

    request_id: str
    created_iteration_id: int
    observed_logical_context_tokens: int
    observed_reclaimable_blocks: int
    required_blocks: int
    target_free_blocks: int


@dataclass(frozen=True)
class ProjectedPrefillCandidate:
    """One candidate evaluated at the next immutable scheduler boundary."""

    request_id: str
    arrival_time: float
    prompt_tokens: int
    projected_logical_tokens: int

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(
            0,
            int(self.prompt_tokens) - int(self.projected_logical_tokens),
        )


def projected_srpt_prefill_order(
    candidates: Iterable[ProjectedPrefillCandidate],
) -> tuple[str, ...]:
    """Return deterministic SRPT order after projected chunk progress."""

    runnable = [
        candidate
        for candidate in candidates
        if candidate.remaining_prefill_tokens > 0
    ]
    return tuple(
        candidate.request_id
        for candidate in sorted(
            runnable,
            key=lambda candidate: (
                candidate.remaining_prefill_tokens,
                float(candidate.arrival_time),
                candidate.request_id,
            ),
        )
    )


def projected_srpt_prefill_target(
    candidates: Iterable[ProjectedPrefillCandidate],
) -> str:
    """Return the deterministic SRPT winner after projected chunk progress."""

    ordered = projected_srpt_prefill_order(candidates)
    return ordered[0] if ordered else ""


def select_best_fit_whole_request_victims(
    candidates: Iterable[ResidencyVictimCandidate],
    *,
    required_blocks: int,
) -> tuple[ResidencyVictimCandidate, ...]:
    """Select whole-request victims with deterministic best-fit growth.

    The current Prefill target and Decode requests must be removed by the
    caller before invoking this pure helper.  For each remaining deficit we
    choose the smallest single request that covers it.  If no request covers
    it, the largest sub-deficit request is selected to make maximal progress,
    then best-fit is evaluated again.  This implements the bounded greedy
    policy ``smallest covering request, otherwise grow the victim set`` while
    avoiding SRPT-tail position as the primary eviction criterion.
    """

    remaining = max(0, int(required_blocks))
    if remaining <= 0:
        return ()
    available = [
        candidate
        for candidate in candidates
        if int(candidate.estimated_reclaimable_blocks) > 0
    ]
    selected: list[ResidencyVictimCandidate] = []
    while remaining > 0 and available:
        covering = [
            candidate
            for candidate in available
            if int(candidate.estimated_reclaimable_blocks) >= remaining
        ]
        if covering:
            victim = min(
                covering,
                key=lambda candidate: (
                    int(candidate.estimated_reclaimable_blocks),
                    float(candidate.arrival_time),
                    candidate.request_id,
                ),
            )
        else:
            victim = min(
                available,
                key=lambda candidate: (
                    -int(candidate.estimated_reclaimable_blocks),
                    float(candidate.arrival_time),
                    candidate.request_id,
                ),
            )
        selected.append(victim)
        available.remove(victim)
        remaining -= int(victim.estimated_reclaimable_blocks)
    return tuple(selected)


def fixed_ring_chunk_count(
    remaining_prefill_tokens: int,
    ring_chunk_tokens: int,
) -> int:
    """Predict future ring invocations under a fixed prefill quantum."""

    if ring_chunk_tokens <= 0:
        raise ValueError("ring_chunk_tokens must be positive")
    remaining = max(0, int(remaining_prefill_tokens))
    return (remaining + ring_chunk_tokens - 1) // ring_chunk_tokens


def fixed_ring_attention_work(
    logical_context_tokens: int,
    remaining_prefill_tokens: int,
    ring_chunk_tokens: int,
) -> int:
    """Return a deterministic q_len * resulting_context work proxy.

    The proxy is only a tie-breaker after predicted invocation count and HBM
    overshoot. It is not a latency model.
    """

    if ring_chunk_tokens <= 0:
        raise ValueError("ring_chunk_tokens must be positive")
    context = max(0, int(logical_context_tokens))
    remaining = max(0, int(remaining_prefill_tokens))
    work = 0
    while remaining > 0:
        query_tokens = min(remaining, ring_chunk_tokens)
        context += query_tokens
        work += query_tokens * context
        remaining -= query_tokens
    return work


def fixed_ring_staging_h2d_tokens(
    logical_context_tokens: int,
    remaining_prefill_tokens: int,
    ring_chunk_tokens: int,
) -> int:
    """Estimate cumulative DRAM-to-staging copy tokens for future chunks.

    Before each fixed-quantum ring invocation, the already-computed logical
    context is the DRAM-backed prefix traversed through the HBM staging slots.
    The current query chunk is produced in HBM and becomes part of the prefix
    only for later invocations.
    """

    if ring_chunk_tokens <= 0:
        raise ValueError("ring_chunk_tokens must be positive")
    context = max(0, int(logical_context_tokens))
    remaining = max(0, int(remaining_prefill_tokens))
    copied_tokens = 0
    while remaining > 0:
        copied_tokens += context
        query_tokens = min(remaining, ring_chunk_tokens)
        context += query_tokens
        remaining -= query_tokens
    return copied_tokens


def estimate_jit_min_copy_components(
    logical_context_tokens: int,
    prompt_tokens: int,
    ring_chunk_tokens: int,
) -> tuple[int, int, int]:
    """Return D2H, ring-staging H2D, and final reload H2D estimates.

    The JIT policy accounts for the complete copy lifecycle created by making
    an in-progress prefill DRAM-authoritative.  Each prompt KV token is copied
    to DRAM once as the request progresses, every future ring invocation reads
    its existing DRAM prefix through staging, and native decode finally reloads
    the completed prompt KV once.
    """

    prompt = max(0, int(prompt_tokens))
    logical = min(prompt, max(0, int(logical_context_tokens)))
    remaining = prompt - logical
    return (
        prompt,
        fixed_ring_staging_h2d_tokens(
            logical,
            remaining,
            ring_chunk_tokens,
        ),
        prompt,
    )


def required_reclaim_blocks(
    free_blocks: int,
    reserved_blocks: int,
    ring_chunk_tokens: int,
    block_size: int,
) -> int:
    """Return HBM blocks missing for one fixed ring-prefill chunk."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if ring_chunk_tokens <= 0:
        raise ValueError("ring_chunk_tokens must be positive")
    chunk_blocks = (ring_chunk_tokens + block_size - 1) // block_size
    return max(
        0,
        max(0, int(reserved_blocks))
        + chunk_blocks
        - max(0, int(free_blocks)),
    )


def scheduled_prefill_reclaim_blocks(
    free_blocks: int,
    scheduled_prefill_required_blocks: int,
    reserved_blocks: int,
) -> int:
    """Return only the deficit for one physical Prefill capacity claim.

    The claim may represent one bounded Ring quantum or all remaining native
    Prefill growth. A later request arrival is intentionally not predicted:
    the next scheduler call recomputes the target from its new candidate set.
    """

    required = max(0, int(scheduled_prefill_required_blocks))
    if required <= 0:
        # Existing allocated tail capacity can make progress without consuming
        # the free pool; residency admission intentionally bypasses reserves in
        # this case as well.
        return 0
    return max(
        0,
        required + max(0, int(reserved_blocks))
        - max(0, int(free_blocks)),
    )


def scheduled_prefill_capacity_headroom_blocks(
    total_usable_blocks: int,
    capacity_fraction: float,
) -> int:
    """Return the HBM headroom above a fractional capacity ceiling.

    ``total_usable_blocks`` excludes the allocator's null block. Flooring the
    usable capacity makes the complementary headroom conservative: a 0.98
    ceiling on 4095 usable blocks permits 4013 and reserves 82 blocks. The
    scheduler decides whether this complement is a hard active-Decode reserve
    or a soft no-Decode runway.
    """

    total = max(0, int(total_usable_blocks))
    fraction = float(capacity_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("capacity_fraction must be in (0.0, 1.0]")
    capacity_limit = int(total * fraction)
    return total - capacity_limit


def min_ring_chunks_sort_key(
    candidate: ResidencyVictimCandidate,
    *,
    required_blocks: int,
    ring_chunk_tokens: int,
) -> tuple[int, int, int, int, int, float, str]:
    """Rank a single victim for incremental min-ring-chunks reclaim.

    A candidate that alone covers the current deficit is preferred. Among
    covering candidates, predicted ring invocation count is primary, followed
    by HBM overshoot and a q_len*context work proxy. If no candidate covers the
    deficit, the first field keeps all insufficient candidates behind covering
    ones and ``estimated_cover_chunks`` favors better incremental progress.
    The scheduler commits one copy-ready victim at a time, re-measures free
    blocks, and may rerun this policy in the same immutable scheduling boundary
    until the current reclaim target is met.
    """

    reclaimable = max(0, int(candidate.estimated_reclaimable_blocks))
    required = max(0, int(required_blocks))
    remaining = candidate.remaining_prefill_tokens
    ring_chunks = fixed_ring_chunk_count(remaining, ring_chunk_tokens)
    ring_work = fixed_ring_attention_work(
        candidate.logical_context_tokens,
        remaining,
        ring_chunk_tokens,
    )
    covers = required == 0 or reclaimable >= required
    if covers:
        return (
            0,
            ring_chunks,
            max(0, reclaimable - required),
            ring_work,
            -reclaimable,
            float(candidate.arrival_time),
            candidate.request_id,
        )

    # Reclaim commits one victim at a time. Estimate how many copies of this
    # candidate's reclaim efficiency would cover the deficit so a tiny,
    # low-chunk candidate cannot defeat all meaningful progress.
    if reclaimable <= 0:
        estimated_cover_chunks = 2**63 - 1
    else:
        victims_needed = (required + reclaimable - 1) // reclaimable
        estimated_cover_chunks = victims_needed * ring_chunks
    return (
        1,
        estimated_cover_chunks,
        ring_chunks,
        ring_work,
        -reclaimable,
        float(candidate.arrival_time),
        candidate.request_id,
    )


def jit_min_copy_sort_key(
    candidate: ResidencyVictimCandidate,
    *,
    required_blocks: int,
) -> tuple[int, Fraction, int, int, float, str]:
    """Rank by total copy per useful block reclaimed for the current deficit.

    ``Fraction`` keeps the ratio comparison exact and deterministic.  Reclaim
    beyond the current deficit receives no credit, which prevents a large
    victim from winning merely by over-evicting.  Zero-progress candidates are
    always ordered last.
    """

    reclaimable = max(0, int(candidate.estimated_reclaimable_blocks))
    required = max(0, int(required_blocks))
    useful_reclaim = min(reclaimable, required)
    total_copy = candidate.estimated_total_copy_tokens
    if useful_reclaim <= 0:
        return (
            1,
            Fraction(0, 1),
            total_copy,
            0,
            float(candidate.arrival_time),
            candidate.request_id,
        )
    return (
        0,
        Fraction(total_copy, useful_reclaim),
        total_copy,
        -useful_reclaim,
        float(candidate.arrival_time),
        candidate.request_id,
    )


def order_residency_victims(
    candidates: list[ResidencyVictimCandidate],
    *,
    policy: ResidencyVictimPolicy,
    required_blocks: int,
    ring_chunk_tokens: int,
) -> list[ResidencyVictimCandidate]:
    """Return a deterministic, mutation-free victim proposal."""

    if policy == ResidencyVictimPolicy.LONGEST_LOGICAL_CONTEXT_FIRST:
        return sorted(
            candidates,
            key=lambda candidate: (
                -max(0, int(candidate.logical_context_tokens)),
                float(candidate.arrival_time),
                candidate.request_id,
            ),
        )
    if policy == ResidencyVictimPolicy.MIN_RING_CHUNKS:
        return sorted(
            candidates,
            key=lambda candidate: min_ring_chunks_sort_key(
                candidate,
                required_blocks=required_blocks,
                ring_chunk_tokens=ring_chunk_tokens,
            ),
        )
    # scheduled_prefill_fit changes the reclaim target and commit granularity,
    # while retaining JIT-Min-Copy's deterministic victim ordering.
    return sorted(
        candidates,
        key=lambda candidate: jit_min_copy_sort_key(
            candidate,
            required_blocks=required_blocks,
        ),
    )
