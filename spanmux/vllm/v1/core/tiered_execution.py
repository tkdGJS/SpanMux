# SPDX-License-Identifier: Apache-2.0
"""Request-scoped execution authority for the experimental tiered KV path."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from vllm.v1.core.tiered_errors import TieredBackendEligibilityError

TIERED_EXECUTION_AUTO = "auto"
TIERED_EXECUTION_NATIVE_RESIDENT = "native_resident"
TIERED_EXECUTION_NATIVE_RELOAD = "native_reload"
TIERED_EXECUTION_CONTEXT_RING = "context_ring"
TIERED_EXECUTION_RING_DECODE = "ring_decode"

TIERED_EXECUTION_MODES = frozenset(
    {
        TIERED_EXECUTION_AUTO,
        TIERED_EXECUTION_NATIVE_RESIDENT,
        TIERED_EXECUTION_NATIVE_RELOAD,
        TIERED_EXECUTION_CONTEXT_RING,
        TIERED_EXECUTION_RING_DECODE,
    }
)

TieredExecutionPlan = tuple[int, str, int]
TieredExecutionPlanEntry = tuple[str, int, str, int]


def has_native_reload_authority(scheduler_output: object) -> bool:
    """An allocated reload destination is not yet valid in every model layer.

    This is a per-Future capability check, not a mutable provider-readiness
    probe. Even after the last layer enqueues its copy, the original Future
    must reach final recovery before a later submission can offload its KV.
    Keep the legacy Native hint conservative for mixed-version outputs.
    """
    return bool(
        getattr(scheduler_output, "tiered_native_hbm_reload_request_ids", ())
        or any(
            entry[2] == TIERED_EXECUTION_NATIVE_RELOAD
            for entry in getattr(scheduler_output, "tiered_request_execution_plans", ())
        )
    )


def parse_tiered_execution_plan_entries(
    entries: Iterable[tuple[object, object, object, object]],
) -> dict[str, TieredExecutionPlan]:
    """Validate the compact SchedulerOutput representation."""

    plans: dict[str, TieredExecutionPlan] = {}
    for entry in entries:
        try:
            request_id, generation, mode, expected_reload_pages = entry
            request_id = str(request_id)
            generation = int(generation)
            mode = str(mode).strip().lower()
            expected_reload_pages = int(expected_reload_pages)
        except (TypeError, ValueError) as exc:
            raise TieredBackendEligibilityError(
                f"invalid tiered execution plan entry: {entry!r}"
            ) from exc
        if (
            not request_id
            or generation <= 0
            or mode not in TIERED_EXECUTION_MODES
            or expected_reload_pages < 0
            or request_id in plans
        ):
            raise TieredBackendEligibilityError(
                f"invalid tiered execution plan entry: {entry!r}"
            )
        plans[request_id] = (generation, mode, expected_reload_pages)
    return plans


def resolve_tiered_reload_authority(
    *,
    phase: str,
    reload_required_pages: Mapping[str, set[int]] | None,
    execution_plans: Mapping[str, TieredExecutionPlan] | None,
) -> str:
    """Resolve reload-required rows to ``native``, ``ring``, or ``auto``.

    Once any request has an explicit authority, every reload-required row in
    that batch must have a compatible explicit authority. This prevents a
    batch-global pressure heuristic from silently overriding one scheduler
    decision, which was the Short-5 correctness failure.
    """

    reload_rows = {
        request_id: pages
        for request_id, pages in (reload_required_pages or {}).items()
        if pages
    }
    if not reload_rows:
        return TIERED_EXECUTION_AUTO

    normalized_phase = str(phase).strip().lower()
    choices: set[str] = set()
    auto_request_ids: list[str] = []
    for request_id, pages in reload_rows.items():
        plan = (execution_plans or {}).get(request_id)
        if plan is None or plan[1] == TIERED_EXECUTION_AUTO:
            auto_request_ids.append(request_id)
            continue
        _generation, mode, expected_reload_pages = plan
        if mode == TIERED_EXECUTION_NATIVE_RESIDENT:
            raise TieredBackendEligibilityError(
                "scheduler marked a request native-resident while DRAM pages "
                f"still require reload: request={request_id}"
            )
        if expected_reload_pages <= 0 or len(pages) > expected_reload_pages:
            raise TieredBackendEligibilityError(
                "tiered execution plan reload-page contract mismatch: "
                f"request={request_id}, expected={expected_reload_pages}, "
                f"layer_pages={len(pages)}"
            )
        if mode == TIERED_EXECUTION_NATIVE_RELOAD:
            choices.add("native")
        elif mode == TIERED_EXECUTION_CONTEXT_RING:
            if normalized_phase != "prefill":
                raise TieredBackendEligibilityError(
                    "context_ring authority was assigned outside Prefill: "
                    f"request={request_id}, phase={normalized_phase!r}"
                )
            choices.add("ring")
        elif mode == TIERED_EXECUTION_RING_DECODE:
            if normalized_phase != "decode":
                raise TieredBackendEligibilityError(
                    "ring_decode authority was assigned outside Decode: "
                    f"request={request_id}, phase={normalized_phase!r}"
                )
            choices.add("ring")

    if choices and auto_request_ids:
        raise TieredBackendEligibilityError(
            "tiered execution authority is incomplete for a reload batch: "
            f"auto_request_ids={tuple(sorted(auto_request_ids))}"
        )
    if len(choices) > 1:
        raise TieredBackendEligibilityError(
            "one attention batch contains incompatible native/ring execution "
            f"authorities: request_ids={tuple(sorted(reload_rows))}"
        )
    return next(iter(choices), TIERED_EXECUTION_AUTO)
