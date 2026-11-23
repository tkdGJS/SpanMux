# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.v1.core.tiered_errors import TieredBackendEligibilityError
from vllm.v1.core.tiered_execution import (
    TIERED_EXECUTION_CONTEXT_RING,
    TIERED_EXECUTION_NATIVE_RELOAD,
    parse_tiered_execution_plan_entries,
    resolve_tiered_reload_authority,
)


def test_native_reload_authority_overrides_pressure_heuristic() -> None:
    plans = parse_tiered_execution_plan_entries(
        (("short-5", 7, TIERED_EXECUTION_NATIVE_RELOAD, 2_176),)
    )

    assert (
        resolve_tiered_reload_authority(
            phase="prefill",
            reload_required_pages={"short-5": {0, 1, 2}},
            execution_plans=plans,
        )
        == "native"
    )


def test_context_ring_authority_is_explicit() -> None:
    plans = parse_tiered_execution_plan_entries(
        (("long-2", 3, TIERED_EXECUTION_CONTEXT_RING, 1_024),)
    )

    assert (
        resolve_tiered_reload_authority(
            phase="prefill",
            reload_required_pages={"long-2": {0, 1}},
            execution_plans=plans,
        )
        == "ring"
    )


def test_mixed_native_and_ring_authority_fails_before_backend_mutation() -> None:
    plans = parse_tiered_execution_plan_entries(
        (
            ("short", 1, TIERED_EXECUTION_NATIVE_RELOAD, 128),
            ("long", 2, TIERED_EXECUTION_CONTEXT_RING, 512),
        )
    )

    with pytest.raises(TieredBackendEligibilityError):
        resolve_tiered_reload_authority(
            phase="prefill",
            reload_required_pages={"short": {0}, "long": {0}},
            execution_plans=plans,
        )


def test_explicit_authority_rejects_unplanned_reload_row() -> None:
    plans = parse_tiered_execution_plan_entries(
        (("short", 1, TIERED_EXECUTION_NATIVE_RELOAD, 128),)
    )

    with pytest.raises(TieredBackendEligibilityError):
        resolve_tiered_reload_authority(
            phase="prefill",
            reload_required_pages={"short": {0}, "stale": {0}},
            execution_plans=plans,
        )
