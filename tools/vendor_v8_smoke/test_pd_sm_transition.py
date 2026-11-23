#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for bounded dynamic P/D SM profile transitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "vendor_v8/vllm/v1/engine/pd_sm_transition.py"
SPEC = importlib.util.spec_from_file_location("pd_sm_transition_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
LOADED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LOADED
SPEC.loader.exec_module(LOADED)
PDSMDrainTransition = LOADED.PDSMDrainTransition


def test_prefill_then_all_phase_drain_reaches_quiescence() -> None:
    transition = PDSMDrainTransition()
    transition.request(
        source_profile_id="d20_p20",
        source_epoch=3,
        source_decode_sms=20,
        source_prefill_sms=20,
        target_profile_id="d30_p10",
        target_decode_sms=30,
        target_prefill_sms=10,
        reason="inflight_profile_hold",
        urgent=True,
        now=10.0,
    )
    assert transition.active
    assert transition.stage == "drain_prefill"
    assert transition.allows_phase("decode")
    assert not transition.allows_phase("prefill")
    assert not transition.allows_phase("mixed")
    transition.note_decode_submission(4)
    transition.note_decode_submission(2)

    assert not transition.advance(
        old_decode_futures=2,
        old_prefill_futures=1,
        global_queue_depth=3,
        now=10.1,
    )
    assert transition.stage == "drain_prefill"
    assert not transition.advance(
        old_decode_futures=2,
        old_prefill_futures=0,
        global_queue_depth=2,
        now=10.2,
    )
    assert transition.stage == "drain_all"
    assert not transition.allows_phase("decode")
    assert transition.advance(
        old_decode_futures=0,
        old_prefill_futures=0,
        global_queue_depth=0,
        now=10.3,
    )
    completed = transition.finish(now=10.4)
    assert 399.9 < completed["pd_sm_transition_wait_ms"] < 400.1
    assert 199.9 < completed["pd_sm_transition_prefill_drain_ms"] < 200.1
    assert 199.9 < completed["pd_sm_transition_all_fence_ms"] < 200.1
    assert (
        completed[
            "pd_sm_transition_decode_submissions_during_prefill_drain"
        ]
        == 2.0
    )
    assert (
        completed["pd_sm_transition_decode_tokens_during_prefill_drain"]
        == 6.0
    )
    assert completed["pd_sm_transition_peak_old_decode_futures"] == 2.0
    assert completed["pd_sm_transition_peak_old_prefill_futures"] == 1.0
    assert not transition.active
    assert transition.stage == "idle"


def test_urgent_pending_target_cannot_be_weakened() -> None:
    transition = PDSMDrainTransition()
    transition.request(
        source_profile_id="d20_p20",
        source_epoch=0,
        source_decode_sms=20,
        source_prefill_sms=20,
        target_profile_id="d30_p10",
        target_decode_sms=30,
        target_prefill_sms=10,
        reason="violation",
        urgent=True,
        now=1.0,
    )
    transition.request(
        source_profile_id="d20_p20",
        source_epoch=0,
        source_decode_sms=20,
        source_prefill_sms=20,
        target_profile_id="d28_p12",
        target_decode_sms=28,
        target_prefill_sms=12,
        reason="transient_slack",
        urgent=False,
        now=1.1,
    )
    assert transition.target_profile_id == "d30_p10"
    assert transition.target_decode_sms == 30


def test_verified_decode_idle_supersedes_urgent_target_with_prefill_only() -> None:
    transition = PDSMDrainTransition()
    transition.request(
        source_profile_id="d30_p10",
        source_epoch=4,
        source_decode_sms=30,
        source_prefill_sms=10,
        target_profile_id="d32_p8",
        target_decode_sms=32,
        target_prefill_sms=8,
        reason="tbt_violation",
        urgent=True,
        now=1.0,
    )
    transition.request(
        source_profile_id="d30_p10",
        source_epoch=4,
        source_decode_sms=30,
        source_prefill_sms=10,
        target_profile_id="d0_p40",
        target_decode_sms=0,
        target_prefill_sms=40,
        reason="layerwise_decode_idle_prefill_only",
        urgent=False,
        decode_backlog_empty=True,
        now=1.1,
    )
    assert transition.target_profile_id == "d0_p40"
    assert transition.target_decode_sms == 0
    assert transition.target_prefill_sms == 40
    assert not transition.urgent
    assert transition.decode_idle_override_count == 1


def test_prefill_only_source_fences_decode_until_commit() -> None:
    transition = PDSMDrainTransition()
    transition.request(
        source_profile_id="d0_p40",
        source_epoch=7,
        source_decode_sms=0,
        source_prefill_sms=40,
        target_profile_id="d30_p10",
        target_decode_sms=30,
        target_prefill_sms=10,
        reason="inflight_profile_hold",
        urgent=True,
        now=20.0,
    )
    assert transition.stage == "drain_prefill"
    assert not transition.allows_phase("decode")
    assert not transition.allows_phase("prefill")
    assert not transition.advance(
        old_decode_futures=0,
        old_prefill_futures=1,
        global_queue_depth=1,
        now=20.1,
    )
    assert transition.advance(
        old_decode_futures=0,
        old_prefill_futures=0,
        global_queue_depth=0,
        now=20.2,
    )


def test_newer_epoch_future_defers_commit_instead_of_failing() -> None:
    transition = PDSMDrainTransition()
    transition.request(
        source_profile_id="d20_p20",
        source_epoch=3,
        source_decode_sms=20,
        source_prefill_sms=20,
        target_profile_id="d30_p10",
        target_decode_sms=30,
        target_prefill_sms=10,
        reason="inflight_profile_hold",
        urgent=False,
        now=30.0,
    )
    assert not transition.advance(
        old_decode_futures=0,
        old_prefill_futures=0,
        global_queue_depth=1,
        newer_epoch_futures=1,
        now=30.1,
    )
    assert transition.stage == "drain_all"
    snapshot = transition.debug_snapshot(now=30.1)
    assert snapshot["pd_sm_transition_peak_newer_epoch_futures"] == 1.0
    assert snapshot["pd_sm_transition_epoch_drift_observation_count"] == 1.0
    assert snapshot["pd_sm_transition_commit_deferred_nonempty_count"] == 1.0
    assert transition.advance(
        old_decode_futures=0,
        old_prefill_futures=0,
        global_queue_depth=0,
        newer_epoch_futures=0,
        now=30.2,
    )


if __name__ == "__main__":
    test_prefill_then_all_phase_drain_reaches_quiescence()
    test_urgent_pending_target_cannot_be_weakened()
    test_verified_decode_idle_supersedes_urgent_target_with_prefill_only()
    test_prefill_only_source_fences_decode_until_commit()
    test_newer_epoch_future_defers_commit_instead_of_failing()
    print("[OK] bounded P/D SM transition tests passed")
