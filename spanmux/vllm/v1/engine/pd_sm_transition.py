# SPDX-License-Identifier: Apache-2.0
"""Bounded drain protocol for whole-forward P/D SM profile changes."""

from __future__ import annotations

import time
from dataclasses import dataclass


_IDLE = "idle"
_DRAIN_PREFILL = "drain_prefill"
_DRAIN_ALL = "drain_all"


@dataclass
class PDSMDrainTransition:
    """Track an old-profile drain without cancelling in-flight work.

    Prefill admission is fenced first while Decode may continue on a source
    profile that owns Decode SMs. Once every old-profile Prefill future is
    gone, all new submissions are fenced so the complete phase queue can drain
    and the profile can be committed at a real quiescent boundary.
    """

    active: bool = False
    stage: str = _IDLE
    source_profile_id: str = ""
    source_epoch: int = -1
    source_decode_sms: int = 0
    source_prefill_sms: int = 0
    target_profile_id: str = ""
    target_decode_sms: int = 0
    target_prefill_sms: int = 0
    reason: str = ""
    urgent: bool = False
    requested_at: float = 0.0
    stage_changed_at: float = 0.0
    drain_all_started_at: float = 0.0
    superseded_count: int = 0
    blocked_prefill_admission_checks: int = 0
    blocked_decode_admission_checks: int = 0
    old_decode_futures: int = 0
    old_prefill_futures: int = 0
    initial_old_decode_futures: int = -1
    initial_old_prefill_futures: int = -1
    peak_old_decode_futures: int = 0
    peak_old_prefill_futures: int = 0
    decode_submissions_during_prefill_drain: int = 0
    decode_tokens_during_prefill_drain: int = 0
    global_queue_depth: int = 0
    peak_global_queue_depth: int = 0
    newer_epoch_futures: int = 0
    peak_newer_epoch_futures: int = 0
    epoch_drift_observation_count: int = 0
    source_mismatch_count: int = 0
    commit_deferred_nonempty_count: int = 0
    decode_idle_override_count: int = 0

    def request(
        self,
        *,
        source_profile_id: str,
        source_epoch: int,
        source_decode_sms: int,
        source_prefill_sms: int,
        target_profile_id: str,
        target_decode_sms: int,
        target_prefill_sms: int,
        reason: str,
        urgent: bool,
        decode_backlog_empty: bool = False,
        now: float | None = None,
    ) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        if not self.active:
            self.active = True
            self.stage = _DRAIN_PREFILL
            self.source_profile_id = source_profile_id
            self.source_epoch = int(source_epoch)
            self.source_decode_sms = max(0, int(source_decode_sms))
            self.source_prefill_sms = max(0, int(source_prefill_sms))
            self.requested_at = timestamp
            self.stage_changed_at = timestamp
            self.drain_all_started_at = 0.0
            self.superseded_count = 0
            self.blocked_prefill_admission_checks = 0
            self.blocked_decode_admission_checks = 0
            self.old_decode_futures = 0
            self.old_prefill_futures = 0
            self.initial_old_decode_futures = -1
            self.initial_old_prefill_futures = -1
            self.peak_old_decode_futures = 0
            self.peak_old_prefill_futures = 0
            self.decode_submissions_during_prefill_drain = 0
            self.decode_tokens_during_prefill_drain = 0
            self.global_queue_depth = 0
            self.peak_global_queue_depth = 0
            self.newer_epoch_futures = 0
            self.peak_newer_epoch_futures = 0
            self.epoch_drift_observation_count = 0
            self.source_mismatch_count = 0
            self.commit_deferred_nonempty_count = 0
            self.decode_idle_override_count = 0
        elif (
            self.source_profile_id != source_profile_id
            or self.source_epoch != int(source_epoch)
            or self.source_decode_sms != max(0, int(source_decode_sms))
            or self.source_prefill_sms != max(0, int(source_prefill_sms))
        ):
            # The active transition owns the source profile and epoch until
            # commit.  A later allocator recommendation may supersede only the
            # target; it must never redefine the source underneath queued
            # futures.
            self.source_mismatch_count += 1
            return
        elif (
            self.target_profile_id != target_profile_id
            or self.target_decode_sms != int(target_decode_sms)
        ):
            self.superseded_count += 1
            decode_idle_override = bool(
                decode_backlog_empty
                and self.source_decode_sms > 0
                and self.source_prefill_sms > 0
                and int(target_decode_sms) == 0
                and int(target_prefill_sms) > self.source_prefill_sms
            )
            # Once an SLO emergency asks for more Decode SMs, do not let a
            # transient slack sample weaken that still-pending transition.
            # A control-plane proof that both submitted and scheduler-ready
            # Decode work are empty is different: the stale Decode emergency
            # no longer owns resources and may be superseded by Prefill-only.
            if (
                self.urgent
                and int(target_decode_sms) < self.target_decode_sms
                and not decode_idle_override
            ):
                return
            if decode_idle_override:
                self.decode_idle_override_count += 1
        self.target_profile_id = target_profile_id
        self.target_decode_sms = int(target_decode_sms)
        self.target_prefill_sms = int(target_prefill_sms)
        self.reason = reason
        if decode_backlog_empty and self.target_decode_sms == 0:
            # No Decode work remains to protect. Do not carry an obsolete
            # urgent bit into the Prefill-only target.
            self.urgent = bool(urgent)
        else:
            self.urgent = self.urgent or bool(urgent)

    def advance(
        self,
        *,
        old_decode_futures: int,
        old_prefill_futures: int,
        global_queue_depth: int,
        newer_epoch_futures: int = 0,
        now: float | None = None,
    ) -> bool:
        """Update drain progress and return whether commit is now safe.

        Source-epoch counts decide when Prefill admission can be fenced.  The
        total phase-queue depth is the sole commit barrier, so an unexpected
        newer-epoch future delays commit instead of crashing the engine.
        """

        if not self.active:
            return False
        self.old_decode_futures = max(0, int(old_decode_futures))
        self.old_prefill_futures = max(0, int(old_prefill_futures))
        self.global_queue_depth = max(0, int(global_queue_depth))
        self.newer_epoch_futures = max(0, int(newer_epoch_futures))
        if self.initial_old_decode_futures < 0:
            self.initial_old_decode_futures = self.old_decode_futures
            self.initial_old_prefill_futures = self.old_prefill_futures
        self.peak_old_decode_futures = max(
            self.peak_old_decode_futures, self.old_decode_futures
        )
        self.peak_old_prefill_futures = max(
            self.peak_old_prefill_futures, self.old_prefill_futures
        )
        self.peak_global_queue_depth = max(
            self.peak_global_queue_depth, self.global_queue_depth
        )
        self.peak_newer_epoch_futures = max(
            self.peak_newer_epoch_futures, self.newer_epoch_futures
        )
        if self.newer_epoch_futures > 0:
            self.epoch_drift_observation_count += 1
        timestamp = time.monotonic() if now is None else float(now)
        if self.stage == _DRAIN_PREFILL and self.old_prefill_futures == 0:
            self.stage = _DRAIN_ALL
            self.stage_changed_at = timestamp
            self.drain_all_started_at = timestamp
        if self.stage == _DRAIN_ALL and self.global_queue_depth > 0:
            self.commit_deferred_nonempty_count += 1
            return False
        return self.stage == _DRAIN_ALL and self.global_queue_depth == 0

    def allows_phase(self, phase: str) -> bool:
        if not self.active:
            return True
        if self.stage == _DRAIN_PREFILL:
            # Mixed/idle scheduler outputs must not slip an old-profile
            # Prefill forward through the drain fence. Decode may continue
            # only if the immutable source profile actually owns Decode SMs.
            return phase == "decode" and self.source_decode_sms > 0
        return False

    def note_blocked(self, phase: str) -> None:
        if phase == "prefill":
            self.blocked_prefill_admission_checks += 1
        elif phase == "decode":
            self.blocked_decode_admission_checks += 1

    def note_decode_submission(self, num_tokens: int) -> None:
        """Account Decode progress preserved while Prefill drains."""

        if self.active and self.stage == _DRAIN_PREFILL:
            self.decode_submissions_during_prefill_drain += 1
            self.decode_tokens_during_prefill_drain += max(0, int(num_tokens))

    def finish(self, *, now: float | None = None) -> dict[str, int | float | str]:
        if not self.active:
            return {}
        timestamp = time.monotonic() if now is None else float(now)
        result = {
            **self.debug_snapshot(now=timestamp),
            "pd_sm_transition_committed_ts": timestamp,
            "pd_sm_transition_wait_ms": max(
                0.0, (timestamp - self.requested_at) * 1000.0
            ),
            "pd_sm_transition_prefill_drain_ms": max(
                0.0,
                (
                    (self.drain_all_started_at or timestamp)
                    - self.requested_at
                )
                * 1000.0,
            ),
            "pd_sm_transition_all_fence_ms": (
                max(
                    0.0,
                    (timestamp - self.drain_all_started_at) * 1000.0,
                )
                if self.drain_all_started_at > 0.0
                else 0.0
            ),
        }
        self.active = False
        self.stage = _IDLE
        self.source_profile_id = ""
        self.source_epoch = -1
        self.source_decode_sms = 0
        self.source_prefill_sms = 0
        self.target_profile_id = ""
        self.target_decode_sms = 0
        self.target_prefill_sms = 0
        self.reason = ""
        self.urgent = False
        self.requested_at = 0.0
        self.stage_changed_at = timestamp
        self.drain_all_started_at = 0.0
        self.old_decode_futures = 0
        self.old_prefill_futures = 0
        return result

    def debug_snapshot(
        self, *, now: float | None = None
    ) -> dict[str, int | float | str]:
        timestamp = time.monotonic() if now is None else float(now)
        return {
            "pd_sm_transition_pending": float(self.active),
            "pd_sm_transition_stage": self.stage,
            "pd_sm_transition_source_profile": self.source_profile_id,
            "pd_sm_transition_source_epoch": float(self.source_epoch),
            "pd_sm_transition_source_decode_sms": float(
                self.source_decode_sms
            ),
            "pd_sm_transition_source_prefill_sms": float(
                self.source_prefill_sms
            ),
            "pd_sm_transition_target_profile": self.target_profile_id,
            "pd_sm_transition_target_decode_sms": float(
                self.target_decode_sms
            ),
            "pd_sm_transition_target_prefill_sms": float(
                self.target_prefill_sms
            ),
            "pd_sm_transition_reason": self.reason,
            "pd_sm_transition_urgent": float(self.urgent),
            "pd_sm_transition_requested_ts": self.requested_at,
            "pd_sm_transition_age_ms": (
                max(0.0, (timestamp - self.requested_at) * 1000.0)
                if self.active
                else 0.0
            ),
            "pd_sm_transition_old_decode_futures": float(
                self.old_decode_futures
            ),
            "pd_sm_transition_old_prefill_futures": float(
                self.old_prefill_futures
            ),
            "pd_sm_transition_initial_old_decode_futures": float(
                max(0, self.initial_old_decode_futures)
            ),
            "pd_sm_transition_initial_old_prefill_futures": float(
                max(0, self.initial_old_prefill_futures)
            ),
            "pd_sm_transition_peak_old_decode_futures": float(
                self.peak_old_decode_futures
            ),
            "pd_sm_transition_peak_old_prefill_futures": float(
                self.peak_old_prefill_futures
            ),
            "pd_sm_transition_decode_submissions_during_prefill_drain": float(
                self.decode_submissions_during_prefill_drain
            ),
            "pd_sm_transition_decode_tokens_during_prefill_drain": float(
                self.decode_tokens_during_prefill_drain
            ),
            "pd_sm_transition_blocked_decode_admission_checks": float(
                self.blocked_decode_admission_checks
            ),
            "pd_sm_transition_blocked_prefill_admission_checks": float(
                self.blocked_prefill_admission_checks
            ),
            "pd_sm_transition_superseded_count": float(
                self.superseded_count
            ),
            "pd_sm_transition_global_queue_depth": float(
                self.global_queue_depth
            ),
            "pd_sm_transition_peak_global_queue_depth": float(
                self.peak_global_queue_depth
            ),
            "pd_sm_transition_newer_epoch_futures": float(
                self.newer_epoch_futures
            ),
            "pd_sm_transition_peak_newer_epoch_futures": float(
                self.peak_newer_epoch_futures
            ),
            "pd_sm_transition_epoch_drift_observation_count": float(
                self.epoch_drift_observation_count
            ),
            "pd_sm_transition_source_mismatch_count": float(
                self.source_mismatch_count
            ),
            "pd_sm_transition_commit_deferred_nonempty_count": float(
                self.commit_deferred_nonempty_count
            ),
            "pd_sm_transition_decode_idle_override_count": float(
                self.decode_idle_override_count
            ),
        }


__all__ = ["PDSMDrainTransition"]
