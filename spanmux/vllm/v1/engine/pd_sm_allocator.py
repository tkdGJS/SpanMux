# SPDX-License-Identifier: Apache-2.0
"""Default-off dynamic SM allocation policy for P/D phase multiplexing."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

from vllm.pd_sm_profiles import (
    PDSMProfile,
    closest_profile,
    parse_pd_sm_profiles,
    profile_by_id,
)
from vllm.v1.engine.pd_tbt_profile import TBTOverlapProfile


_EWMA_POLICY = "ewma_best_fit"
_TBT_PROFILED_POLICY = "tbt_slo_profiled"
_TBT_FEEDBACK_POLICY = "tbt_slo_feedback"
_TBT_PREDICTIVE_POLICY = "tbt_slo_predictive"


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid {name}={value!r}; expected integer") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid {name}={value!r}; expected number") from exc


@dataclass(frozen=True)
class PDSMAllocatorConfig:
    enabled: bool
    shadow: bool
    strict: bool
    total_sms: int
    alignment: int
    profiles: tuple[PDSMProfile, ...]
    initial_profile: PDSMProfile
    ewma_alpha: float
    decode_target_ms: float
    min_improvement: float
    cooldown_decisions: int
    max_step_sms: int
    switch_penalty_ms: float
    policy: str = _EWMA_POLICY
    tbt_profile_path: str = ""
    tbt_default_slo_ms: float = 0.0
    tbt_profile_margin_ms: float = 0.0
    tbt_profile_strict: bool = True
    feedback_initial_decode_sms: int = 30
    feedback_step_sms: int = 2
    feedback_window_tokens: int = 32
    feedback_slack_ratio: float = 0.90
    feedback_reprobe_ratio: float = 0.80
    feedback_decrease_windows: int = 3
    feedback_hold_windows: int = 16
    feedback_min_prefill_sms: int = 2
    feedback_context_bucket_tokens: int = 25000
    serialized_full_sm: bool = True
    # Zero means uncalibrated. Predictive mode requires explicit inputs; a
    # p95 step is NOT a fastest-step bound for projecting context growth.
    tbt_predictive_lead_ms: float = 0.0
    tbt_predictive_cutover_ms: float = -1.0
    tbt_predictive_min_step_ms: float = 0.0

    @classmethod
    def from_env(cls) -> "PDSMAllocatorConfig":
        enabled = _env_bool("VLLM_PD_DYNAMIC_SM_ENABLED")
        shadow = _env_bool("VLLM_PD_DYNAMIC_SM_SHADOW")
        if not enabled:
            # Preserve vendor_v6 startup semantics exactly when the experiment
            # is off: do not validate or otherwise interpret dynamic-SM flags.
            profiles = parse_pd_sm_profiles(
                "auto",
                total_sms=8,
                alignment=2,
            )
            return cls(
                enabled=False,
                shadow=shadow,
                strict=True,
                total_sms=8,
                alignment=2,
                profiles=profiles,
                initial_profile=closest_profile(
                    profiles,
                    decode_sms=4,
                    prefill_sms=4,
                    concurrent_only=True,
                ),
                ewma_alpha=0.25,
                decode_target_ms=0.0,
                min_improvement=0.10,
                cooldown_decisions=2,
                max_step_sms=8,
                switch_penalty_ms=0.05,
            )
        strict = _env_bool("VLLM_PD_DYNAMIC_SM_STRICT", "1")
        policy = os.environ.get(
            "VLLM_PD_DYNAMIC_SM_POLICY", _EWMA_POLICY
        ).strip().lower()
        policy_aliases = {
            "": _EWMA_POLICY,
            "ewma": _EWMA_POLICY,
            "ewma_best_fit": _EWMA_POLICY,
            "tbt": _TBT_PROFILED_POLICY,
            "tbt_profiled": _TBT_PROFILED_POLICY,
            "tbt_slo_profiled": _TBT_PROFILED_POLICY,
            "feedback": _TBT_FEEDBACK_POLICY,
            "tbt_feedback": _TBT_FEEDBACK_POLICY,
            "tbt_slo_feedback": _TBT_FEEDBACK_POLICY,
            "tbt_slo_predictive": _TBT_PREDICTIVE_POLICY,
        }
        if policy not in policy_aliases:
            raise ValueError(
                "invalid VLLM_PD_DYNAMIC_SM_POLICY="
                f"{policy!r}; expected ewma_best_fit, tbt_slo_profiled, "
                "tbt_slo_feedback, or tbt_slo_predictive"
            )
        policy = policy_aliases[policy]
        alignment = max(1, _env_int("VLLM_PD_DYNAMIC_SM_ALIGNMENT", 2))
        static_decode = max(
            0,
            _env_int("VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS", 8),
        )
        static_prefill = max(
            0,
            _env_int("VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS", 0),
        )
        profile_spec = os.environ.get(
            "VLLM_PD_DYNAMIC_SM_PROFILES",
            "auto_dense" if policy in {_TBT_FEEDBACK_POLICY, _TBT_PREDICTIVE_POLICY} else "auto",
        )

        configured_total = _env_int("VLLM_PD_DYNAMIC_SM_TOTAL_SMS", 0)
        if configured_total <= 0 and profile_spec.strip().lower() not in {
            "auto",
            "auto_dense",
            "dense",
        }:
            parsed_totals = []
            for item in profile_spec.split(","):
                normalized = item.strip().lower()
                if "/" in normalized:
                    decode_text, prefill_text = normalized.split("/", 1)
                elif normalized.startswith("d") and "_p" in normalized:
                    decode_text, prefill_text = normalized[1:].split("_p", 1)
                else:
                    continue
                try:
                    parsed_totals.append(int(decode_text) + int(prefill_text))
                except ValueError:
                    pass
            if parsed_totals:
                configured_total = max(parsed_totals)
        total_sms = configured_total or (static_decode + static_prefill)
        if total_sms <= 0:
            if enabled:
                raise ValueError(
                    "dynamic P/D SM allocation needs "
                    "VLLM_PD_DYNAMIC_SM_TOTAL_SMS or a positive static split"
                )
            total_sms = alignment

        profiles = parse_pd_sm_profiles(
            profile_spec,
            total_sms=total_sms,
            alignment=alignment,
        )
        initial_profile = closest_profile(
            profiles,
            decode_sms=static_decode,
            prefill_sms=static_prefill,
            concurrent_only=True,
        )
        alpha = _env_float("VLLM_PD_DYNAMIC_SM_EWMA_ALPHA", 0.25)
        min_improvement = _env_float(
            "VLLM_PD_DYNAMIC_SM_MIN_IMPROVEMENT", 0.10
        )
        tbt_default_slo_ms = _env_float(
            "VLLM_PD_TBT_DEFAULT_SLO_MS", 0.0
        )
        tbt_profile_margin_ms = _env_float(
            "VLLM_PD_TBT_PROFILE_MARGIN_MS", 0.0
        )
        feedback_initial_decode_sms = _env_int(
            "VLLM_PD_FEEDBACK_INITIAL_DECODE_SMS", 30
        )
        feedback_step_sms = _env_int("VLLM_PD_FEEDBACK_STEP_SMS", alignment)
        feedback_window_tokens = _env_int(
            "VLLM_PD_FEEDBACK_WINDOW_TOKENS", 32
        )
        feedback_slack_ratio = _env_float(
            "VLLM_PD_FEEDBACK_SLACK_RATIO", 0.90
        )
        feedback_reprobe_ratio = _env_float(
            "VLLM_PD_FEEDBACK_REPROBE_RATIO", 0.80
        )
        feedback_decrease_windows = _env_int(
            "VLLM_PD_FEEDBACK_DECREASE_WINDOWS", 3
        )
        feedback_hold_windows = _env_int(
            "VLLM_PD_FEEDBACK_HOLD_WINDOWS", 16
        )
        feedback_min_prefill_sms = _env_int(
            "VLLM_PD_FEEDBACK_MIN_PREFILL_SMS", alignment
        )
        feedback_context_bucket_tokens = _env_int(
            "VLLM_PD_FEEDBACK_CONTEXT_BUCKET_TOKENS", 25000
        )
        if (
            not math.isfinite(tbt_default_slo_ms)
            or tbt_default_slo_ms < 0
        ):
            raise ValueError(
                "VLLM_PD_TBT_DEFAULT_SLO_MS must be finite and non-negative"
            )
        if (
            not math.isfinite(tbt_profile_margin_ms)
            or tbt_profile_margin_ms < 0
        ):
            raise ValueError(
                "VLLM_PD_TBT_PROFILE_MARGIN_MS must be finite and "
                "non-negative"
            )
        if feedback_initial_decode_sms <= 0:
            raise ValueError(
                "VLLM_PD_FEEDBACK_INITIAL_DECODE_SMS must be positive"
            )
        if feedback_step_sms <= 0 or feedback_step_sms % alignment != 0:
            raise ValueError(
                "VLLM_PD_FEEDBACK_STEP_SMS must be a positive multiple of "
                "VLLM_PD_DYNAMIC_SM_ALIGNMENT"
            )
        if feedback_window_tokens <= 0:
            raise ValueError(
                "VLLM_PD_FEEDBACK_WINDOW_TOKENS must be positive"
            )
        if not 0.0 < feedback_reprobe_ratio <= feedback_slack_ratio < 1.0:
            raise ValueError(
                "feedback ratios must satisfy 0 < reprobe <= slack < 1"
            )
        if feedback_decrease_windows <= 0 or feedback_hold_windows < 0:
            raise ValueError(
                "feedback decrease windows must be positive and hold "
                "windows non-negative"
            )
        if (
            feedback_min_prefill_sms <= 0
            or feedback_min_prefill_sms % alignment != 0
        ):
            raise ValueError(
                "VLLM_PD_FEEDBACK_MIN_PREFILL_SMS must be a positive "
                "aligned SM count"
            )
        if feedback_context_bucket_tokens <= 0:
            raise ValueError(
                "VLLM_PD_FEEDBACK_CONTEXT_BUCKET_TOKENS must be positive"
            )
        if policy == _TBT_FEEDBACK_POLICY:
            grid_offset = (
                feedback_initial_decode_sms % feedback_step_sms
            )
            feedback_candidates = [
                profile
                for profile in profiles
                if profile.is_concurrent
                and profile.prefill_sms >= feedback_min_prefill_sms
                and profile.decode_sms % feedback_step_sms == grid_offset
            ]
            if not feedback_candidates:
                raise ValueError(
                    "tbt_slo_feedback has no concurrent profile on the "
                    "configured step grid that preserves the Prefill floor"
                )
        return cls(
            enabled=enabled,
            shadow=shadow,
            strict=strict,
            total_sms=total_sms,
            alignment=alignment,
            profiles=profiles,
            initial_profile=initial_profile,
            ewma_alpha=min(1.0, max(0.001, alpha)),
            decode_target_ms=max(
                0.0,
                _env_float("VLLM_PD_DYNAMIC_SM_DECODE_TARGET_MS", 0.0),
            ),
            min_improvement=min(1.0, max(0.0, min_improvement)),
            cooldown_decisions=max(
                0,
                _env_int("VLLM_PD_DYNAMIC_SM_COOLDOWN_DECISIONS", 2),
            ),
            max_step_sms=max(
                0,
                _env_int("VLLM_PD_DYNAMIC_SM_MAX_STEP", 8),
            ),
            switch_penalty_ms=max(
                0.0,
                _env_float("VLLM_PD_DYNAMIC_SM_SWITCH_PENALTY_MS", 0.05),
            ),
            policy=policy,
            tbt_profile_path=os.environ.get(
                "VLLM_PD_TBT_PROFILE_PATH", ""
            ).strip(),
            tbt_default_slo_ms=tbt_default_slo_ms,
            tbt_profile_margin_ms=tbt_profile_margin_ms,
            tbt_profile_strict=_env_bool(
                "VLLM_PD_TBT_PROFILE_STRICT", "1"
            ),
            feedback_initial_decode_sms=feedback_initial_decode_sms,
            feedback_step_sms=feedback_step_sms,
            feedback_window_tokens=feedback_window_tokens,
            feedback_slack_ratio=feedback_slack_ratio,
            feedback_reprobe_ratio=feedback_reprobe_ratio,
            feedback_decrease_windows=feedback_decrease_windows,
            feedback_hold_windows=feedback_hold_windows,
            feedback_min_prefill_sms=feedback_min_prefill_sms,
            feedback_context_bucket_tokens=feedback_context_bucket_tokens,
            serialized_full_sm=_env_bool(
                "VLLM_PD_SERIALIZED_FULL_SM", "1"
            ),
            tbt_predictive_lead_ms=_env_float(
                "VLLM_PD_TBT_PREDICTIVE_LEAD_MS", 0.0),
            tbt_predictive_cutover_ms=_env_float(
                "VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS", -1.0),
            tbt_predictive_min_step_ms=_env_float(
                "VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS", 0.0),
        )

    def with_effective_enabled(self, enabled: bool) -> "PDSMAllocatorConfig":
        return replace(self, enabled=enabled)


@dataclass
class _TBTFeedbackState:
    minimum_feasible_decode_sms: int = 0
    maximum_infeasible_decode_sms: int = 0
    consecutive_slack_windows: int = 0
    hold_windows_remaining: int = 0
    last_evaluation_epoch: int = 0
    last_direction: str = ""
    pending_decode_sms: int = 0
    pending_reason: str = ""


@dataclass(frozen=True)
class _TBTFeedbackSelection:
    profile: PDSMProfile
    reason: str
    state_key: str
    mode: str
    observed_p95_ms: float
    observed_slo_ms: float
    observed_ratio: float
    observation_count: int
    observation_epoch: int
    window_ready: bool
    violation: bool
    minimum_feasible_decode_sms: int
    maximum_infeasible_decode_sms: int


@dataclass(frozen=True)
class PDSMPlan:
    profile: PDSMProfile
    target_profile: PDSMProfile
    epoch: int
    reason: str
    enabled: bool
    shadow: bool
    switched: bool
    deferred: bool
    predicted_decode_ms: float = 0.0
    predicted_prefill_ms: float = 0.0
    predicted_bubble_sm_ms: float = 0.0
    improvement: float = 0.0
    allocator_policy: str = _EWMA_POLICY
    tbt_profile_state_covered: bool = False
    tbt_profile_prefill_mode: str = ""
    tbt_profile_predicted_per_token_ms: float = 0.0
    tbt_profile_bucket_context_tokens: int = 0
    tbt_profile_bucket_decode_requests: int = 0
    tbt_elapsed_ms: float = 0.0
    tbt_predicted_next_ms: float = 0.0
    tbt_slo_ms: float = 0.0
    tbt_slo_feasible: bool = False
    tbt_slo_emergency: bool = False
    tbt_slo_overload: bool = False
    tbt_profile_fallback: bool = False
    tbt_feedback: _TBTFeedbackSelection | None = None
    serialized_full_sm_enabled: bool = True
    serialized_full_sm_targeted: bool = False
    capacity_serialized: bool = False
    capacity_serialized_fixed_split: bool = False
    tbt_predictive: dict[str, int | float | str] | None = None

    @property
    def execution_profile_id(self) -> str:
        return "" if not self.enabled or self.shadow else self.profile.profile_id

    def as_debug_dict(self) -> dict[str, int | float | str]:
        feedback = self.tbt_feedback
        return {
            **(self.tbt_predictive or {}),
            "pd_dynamic_sm_enabled": float(self.enabled),
            "pd_dynamic_sm_shadow": float(self.shadow),
            "pd_dynamic_sm_profile_id": self.execution_profile_id,
            "pd_dynamic_sm_active_profile_id": self.profile.profile_id,
            "pd_dynamic_sm_target_profile_id": self.target_profile.profile_id,
            "pd_dynamic_sm_profile_epoch": float(self.epoch),
            "pd_dynamic_sm_decode_sms": float(self.profile.decode_sms),
            "pd_dynamic_sm_prefill_sms": float(self.profile.prefill_sms),
            "pd_dynamic_sm_target_decode_sms": float(
                self.target_profile.decode_sms
            ),
            "pd_dynamic_sm_target_prefill_sms": float(
                self.target_profile.prefill_sms
            ),
            "pd_dynamic_sm_decision_reason": self.reason,
            "pd_dynamic_sm_switched": float(self.switched),
            "pd_dynamic_sm_switch_deferred": float(self.deferred),
            "pd_dynamic_sm_predicted_decode_ms": self.predicted_decode_ms,
            "pd_dynamic_sm_predicted_prefill_ms": self.predicted_prefill_ms,
            "pd_dynamic_sm_predicted_bubble_sm_ms": (
                self.predicted_bubble_sm_ms
            ),
            "pd_dynamic_sm_predicted_improvement": self.improvement,
            "pd_dynamic_sm_policy": self.allocator_policy,
            "pd_dynamic_sm_serialized_full_sm_enabled": float(
                self.serialized_full_sm_enabled
            ),
            "pd_dynamic_sm_serialized_full_sm_targeted": float(
                self.serialized_full_sm_targeted
            ),
            "pd_dynamic_sm_capacity_serialized": float(
                self.capacity_serialized
            ),
            "pd_dynamic_sm_capacity_serialized_fixed_split": float(
                self.capacity_serialized_fixed_split
            ),
            "pd_tbt_profile_state_covered": float(
                self.tbt_profile_state_covered
            ),
            "pd_tbt_profile_prefill_mode": self.tbt_profile_prefill_mode,
            "pd_tbt_profile_predicted_per_token_ms": (
                self.tbt_profile_predicted_per_token_ms
            ),
            "pd_tbt_profile_bucket_context_tokens": float(
                self.tbt_profile_bucket_context_tokens
            ),
            "pd_tbt_profile_bucket_decode_requests": float(
                self.tbt_profile_bucket_decode_requests
            ),
            "pd_tbt_elapsed_ms": self.tbt_elapsed_ms,
            "pd_tbt_predicted_next_ms": self.tbt_predicted_next_ms,
            "pd_tbt_slo_ms": self.tbt_slo_ms,
            "pd_tbt_slo_feasible": float(self.tbt_slo_feasible),
            "pd_tbt_slo_emergency": float(self.tbt_slo_emergency),
            "pd_tbt_slo_overload": float(self.tbt_slo_overload),
            "pd_tbt_profile_fallback": float(self.tbt_profile_fallback),
            "pd_tbt_feedback_state_key": (
                feedback.state_key if feedback is not None else ""
            ),
            "pd_tbt_feedback_mode": (
                feedback.mode if feedback is not None else "disabled"
            ),
            "pd_tbt_feedback_observed_p95_ms": (
                feedback.observed_p95_ms if feedback is not None else 0.0
            ),
            "pd_tbt_feedback_observed_slo_ms": (
                feedback.observed_slo_ms if feedback is not None else 0.0
            ),
            "pd_tbt_feedback_observed_ratio": (
                feedback.observed_ratio if feedback is not None else 0.0
            ),
            "pd_tbt_feedback_observation_count": float(
                feedback.observation_count if feedback is not None else 0
            ),
            "pd_tbt_feedback_observation_epoch": float(
                feedback.observation_epoch if feedback is not None else 0
            ),
            "pd_tbt_feedback_window_ready": float(
                feedback.window_ready if feedback is not None else False
            ),
            "pd_tbt_feedback_violation": float(
                feedback.violation if feedback is not None else False
            ),
            "pd_tbt_feedback_minimum_feasible_decode_sms": float(
                feedback.minimum_feasible_decode_sms
                if feedback is not None
                else 0
            ),
            "pd_tbt_feedback_maximum_infeasible_decode_sms": float(
                feedback.maximum_infeasible_decode_sms
                if feedback is not None
                else 0
            ),
        }


@dataclass(frozen=True)
class _TBTProfileSelection:
    profile: PDSMProfile
    reason: str
    state_covered: bool
    prefill_mode: str
    predicted_per_token_ms: float
    bucket_context_tokens: int
    bucket_decode_requests: int
    elapsed_ms: float
    predicted_next_ms: float
    slo_ms: float
    feasible: bool
    emergency: bool
    overload: bool
    fallback: bool
    predictive: dict[str, int | float | str] | None = None
    fallback_ewma: tuple[
        PDSMProfile, float, float, float, float, str
    ] | None = None


class PDComputeAllocationController:
    """Recommend profiles and commit them at proven execution boundaries.

    With in-flight work, ``plan`` exposes a deferred target instead of
    mutating the active profile. EngineCore either drains all source Futures or
    proves a GPU-complete layer boundary for the sole Prefill continuation,
    then advances both GreenContext and allocator authority to the same epoch.
    """

    def __init__(self, config: PDSMAllocatorConfig) -> None:
        self.config = config
        self.current_profile = config.initial_profile
        self.epoch = 0
        self.decision_index = 0
        self.last_switch_decision = -config.cooldown_decisions - 1
        self._sm_ms_per_token_ewma: dict[str, float] = {}
        self._observation_count: dict[str, int] = {"decode": 0, "prefill": 0}
        self._tbt_profile: TBTOverlapProfile | None = None
        self._tbt_profile_load_error = ""
        self._tbt_feedback_states: dict[str, _TBTFeedbackState] = {}
        self._tbt_feedback_active_state_key = ""
        self._tbt_predictive_transition_highwater: dict[str, tuple[float, float]] = {}
        self._tbt_predictive_pending_transition: dict[str, int | float | str] | None = None
        self._tbt_predictive_last_decode_at = 0.0
        self._tbt_predictive_last_decode_profile = ""
        self._tbt_predictive_mode = "native"
        self._tbt_predictive_transition_request: tuple[float, str] | None = None
        self._tbt_predictive_deadline_key: tuple[str, float] | None = None
        self._tbt_predictive_wake_at: float | None = None
        if self.predictive_enabled:
            for name, value, allow_zero in (
                ("LEAD_MS", config.tbt_predictive_lead_ms, False),
                ("CUTOVER_MS", config.tbt_predictive_cutover_ms, True),
                ("MIN_STEP_MS", config.tbt_predictive_min_step_ms, False),
            ):
                if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                    raise ValueError(
                        f"VLLM_PD_TBT_PREDICTIVE_{name} needs explicit calibrated "
                        f"{'nonnegative' if allow_zero else 'positive'} milliseconds")
        if config.enabled and config.policy in {_TBT_PROFILED_POLICY, _TBT_PREDICTIVE_POLICY}:
            if not config.tbt_profile_path:
                self._tbt_profile_load_error = (
                    "VLLM_PD_TBT_PROFILE_PATH is required by "
                    f"{config.policy}"
                )
            else:
                try:
                    self._tbt_profile = TBTOverlapProfile.from_json_file(
                        config.tbt_profile_path
                    )
                except ValueError as exc:
                    self._tbt_profile_load_error = str(exc)
            if self._tbt_profile_load_error and (config.tbt_profile_strict or self.predictive_enabled):
                raise ValueError(self._tbt_profile_load_error)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def predictive_enabled(self) -> bool:
        return self.config.enabled and self.config.policy == _TBT_PREDICTIVE_POLICY

    def arm_tbt_deadline(self, *, request_id: str, deadline_at: float) -> None:
        """One control wake per critical token; unrelated events cannot rearm it."""
        key = (request_id, deadline_at)
        if not self.predictive_enabled or not request_id or deadline_at <= 0:
            self._tbt_predictive_deadline_key = None
            self._tbt_predictive_wake_at = None
        elif key != self._tbt_predictive_deadline_key:
            self._tbt_predictive_deadline_key = key
            self._tbt_predictive_wake_at = deadline_at - self.config.tbt_profile_margin_ms / 1000.0

    def tbt_deadline_wait_s(self, now: float) -> float | None:
        wake_at = self._tbt_predictive_wake_at
        return None if wake_at is None else max(0.0, wake_at - now)

    def consume_tbt_deadline(self, now: float) -> bool:
        if self._tbt_predictive_wake_at is None or now < self._tbt_predictive_wake_at:
            return False
        self._tbt_predictive_wake_at = None
        return True

    def note_tbt_decode_completion(self, *, profile_id: str, now: float) -> None:
        if self.predictive_enabled:
            self._tbt_predictive_last_decode_at = now
            self._tbt_predictive_last_decode_profile = profile_id

    def note_tbt_transition_commit(self, debug: dict) -> None:
        """A committed target is not yet a resumed Decode; retain scalars only."""
        if not self.predictive_enabled:
            return
        self._tbt_predictive_pending_transition = None
        if (debug.get("pd_sm_transition_cancelled")
                or float(debug.get("pd_sm_transition_target_decode_sms", 0)) <= 0):
            self._tbt_predictive_transition_request = None
            return
        request = self._tbt_predictive_transition_request
        if (request is None or request[0] != float(debug["pd_sm_transition_requested_ts"])):
            return
        self._tbt_predictive_pending_transition = {
            "source": str(debug["pd_sm_transition_source_profile"]),
            "target": str(debug["pd_sm_transition_target_profile"]),
            "requested_at": float(debug["pd_sm_transition_requested_ts"]),
            "epoch": int(debug["pd_sm_transition_committed_epoch"]),
            "mode": request[1],
        }
        self._tbt_predictive_transition_request = None

    def note_tbt_transition_request(self, *, requested_at: float) -> None:
        request = self._tbt_predictive_transition_request
        if self.predictive_enabled and (request is None or request[0] != requested_at):
            self._tbt_predictive_transition_request = (requested_at, self._tbt_predictive_mode)
            self._tbt_predictive_pending_transition = None

    def note_tbt_decode_submission(self, *, profile_id: str, epoch: int, now: float) -> None:
        pending = self._tbt_predictive_pending_transition
        if (not self.predictive_enabled or pending is None
                or pending["target"] != profile_id or pending["epoch"] != epoch):
            return
        self._tbt_predictive_pending_transition = None
        requested_at = float(pending["requested_at"])
        lead = max(0.0, (now - requested_at) * 1000.0)
        cutover = 0.0
        if (self._tbt_predictive_last_decode_profile == pending["source"]
                and self._tbt_predictive_last_decode_at >= requested_at):
            # Useful old-profile Decode service during preparation is NOT
            # added to the stopped-service gap.
            cutover = max(0.0, (now - self._tbt_predictive_last_decode_at) * 1000.0)
        mode = str(pending["mode"])
        prior = self._tbt_predictive_transition_highwater.get(mode, (0.0, 0.0))
        self._tbt_predictive_transition_highwater[mode] = (max(prior[0], lead), max(prior[1], cutover))

    def commit_profile_at_boundary(
        self,
        profile_id: str,
        *,
        expected_epoch: int,
    ) -> PDSMProfile:
        """Commit a target after an authoritative execution boundary ACK."""

        if int(expected_epoch) != self.epoch:
            raise RuntimeError(
                "dynamic SM boundary commit epoch mismatch: "
                f"expected={expected_epoch} active={self.epoch}"
            )
        target = profile_by_id(self.config.profiles, profile_id)
        if target is None:
            raise RuntimeError(
                f"unknown dynamic SM boundary target {profile_id!r}"
            )
        if target == self.current_profile:
            return self.current_profile
        self.current_profile = target
        self.epoch += 1
        self.last_switch_decision = self.decision_index
        return self.current_profile

    def commit_profile_at_quiescence(
        self,
        profile_id: str,
        *,
        expected_epoch: int,
    ) -> PDSMProfile:
        """Compatibility wrapper for a fully drained future boundary."""

        return self.commit_profile_at_boundary(
            profile_id,
            expected_epoch=expected_epoch,
        )

    def observe(
        self,
        *,
        phase: str,
        num_tokens: int,
        elapsed_ms: float,
        profile_id: str,
    ) -> dict[str, int | float | str]:
        profile = profile_by_id(self.config.profiles, profile_id)
        phase_sms = (
            profile.decode_sms
            if profile is not None and phase == "decode"
            else profile.prefill_sms
            if profile is not None and phase == "prefill"
            else 0
        )
        accepted = (
            self.enabled
            and phase in {"decode", "prefill"}
            and num_tokens > 0
            and elapsed_ms > 0.0
            and math.isfinite(elapsed_ms)
            and phase_sms > 0
        )
        if accepted:
            sample = elapsed_ms * phase_sms / num_tokens
            previous = self._sm_ms_per_token_ewma.get(phase)
            alpha = self.config.ewma_alpha
            self._sm_ms_per_token_ewma[phase] = (
                sample
                if previous is None
                else alpha * sample + (1.0 - alpha) * previous
            )
            self._observation_count[phase] += 1
        return {
            "pd_dynamic_sm_observation_accepted": float(accepted),
            "pd_dynamic_sm_observation_phase": phase,
            "pd_dynamic_sm_observation_elapsed_ms": float(max(0.0, elapsed_ms)),
            "pd_dynamic_sm_observation_tokens": float(max(0, num_tokens)),
            "pd_dynamic_sm_observation_sms": float(max(0, phase_sms)),
            "pd_dynamic_sm_decode_observation_count": float(
                self._observation_count["decode"]
            ),
            "pd_dynamic_sm_prefill_observation_count": float(
                self._observation_count["prefill"]
            ),
        }

    def _predicted_times(
        self,
        profile: PDSMProfile,
        *,
        decode_tokens: int,
        prefill_tokens: int,
    ) -> tuple[float, float] | None:
        decode_cost = self._sm_ms_per_token_ewma.get("decode")
        prefill_cost = self._sm_ms_per_token_ewma.get("prefill")
        if (
            decode_cost is None
            or prefill_cost is None
            or profile.decode_sms <= 0
            or profile.prefill_sms <= 0
        ):
            return None
        decode_ms = decode_cost * max(1, decode_tokens) / profile.decode_sms
        prefill_ms = prefill_cost * max(1, prefill_tokens) / profile.prefill_sms
        return decode_ms, prefill_ms

    def _profile_score(
        self,
        profile: PDSMProfile,
        *,
        decode_tokens: int,
        prefill_tokens: int,
    ) -> tuple[float, float, float, float] | None:
        predictions = self._predicted_times(
            profile,
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
        )
        if predictions is None:
            return None
        decode_ms, prefill_ms = predictions
        horizon_ms = max(decode_ms, prefill_ms)
        bubble_sm_ms = profile.decode_sms * (horizon_ms - decode_ms) + (
            profile.prefill_sms * (horizon_ms - prefill_ms)
        )
        decode_debt_ms = max(0.0, decode_ms - self.config.decode_target_ms)
        decode_penalty = (
            decode_debt_ms * self.config.total_sms * 1000.0
            if self.config.decode_target_ms > 0.0
            else 0.0
        )
        switch_penalty = (
            self.config.switch_penalty_ms * self.config.total_sms
            if profile != self.current_profile
            else 0.0
        )
        return (
            bubble_sm_ms + decode_penalty + switch_penalty,
            decode_ms,
            prefill_ms,
            bubble_sm_ms,
        )

    def _select_concurrent_profile(
        self,
        *,
        decode_tokens: int,
        prefill_tokens: int,
    ) -> tuple[PDSMProfile, float, float, float, float, str]:
        candidates = [
            profile for profile in self.config.profiles if profile.is_concurrent
        ]
        current = self.current_profile
        if self.config.max_step_sms > 0 and current.is_concurrent:
            bounded = [
                profile
                for profile in candidates
                if abs(profile.decode_sms - current.decode_sms)
                <= self.config.max_step_sms
            ]
            if bounded:
                candidates = bounded

        scored = [
            (score, profile)
            for profile in candidates
            if (
                score := self._profile_score(
                    profile,
                    decode_tokens=decode_tokens,
                    prefill_tokens=prefill_tokens,
                )
            )
            is not None
        ]
        if not scored:
            fallback = (
                current
                if current.is_concurrent
                else self.config.initial_profile
            )
            return fallback, 0.0, 0.0, 0.0, 0.0, "paired_cold_start"

        (best_score, decode_ms, prefill_ms, bubble_sm_ms), best = min(
            scored,
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[1].profile_id,
            ),
        )
        current_score_tuple = self._profile_score(
            current if current.is_concurrent else self.config.initial_profile,
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
        )
        current_score = current_score_tuple[0] if current_score_tuple else best_score
        improvement = (
            max(0.0, (current_score - best_score) / current_score)
            if current_score > 0.0
            else 0.0
        )
        if best != current and improvement < self.config.min_improvement:
            held = current if current.is_concurrent else self.config.initial_profile
            held_score = self._profile_score(
                held,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
            )
            if held_score is not None:
                _, decode_ms, prefill_ms, bubble_sm_ms = held_score
            return (
                held,
                decode_ms,
                prefill_ms,
                bubble_sm_ms,
                improvement,
                "paired_hysteresis_hold",
            )
        return (
            best,
            decode_ms,
            prefill_ms,
            bubble_sm_ms,
            improvement,
            "paired_predicted_best_fit",
        )

    def _select_tbt_profiled_concurrent_profile(
        self,
        *,
        decode_tokens: int,
        prefill_tokens: int,
        decode_context_tokens: int,
        decode_request_count: int,
        prefill_mode: str,
        decode_tbt_slo_ms: float,
        decode_tbt_elapsed_ms: float,
    ) -> _TBTProfileSelection:
        mode = prefill_mode.strip().lower()
        if mode not in {"native", "ring"}:
            mode = "native"
        slo_ms = (
            float(decode_tbt_slo_ms)
            if decode_tbt_slo_ms > 0
            else self.config.tbt_default_slo_ms
        )
        elapsed_ms = max(0.0, float(decode_tbt_elapsed_ms))
        candidates = sorted(
            (
                profile
                for profile in self.config.profiles
                if profile.is_concurrent
            ),
            key=lambda profile: (
                profile.decode_sms,
                -profile.prefill_sms,
                profile.profile_id,
            ),
        )
        if not candidates:
            raise RuntimeError(
                "tbt_slo_profiled requires at least one concurrent SM profile"
            )

        if self._tbt_profile is None or slo_ms <= 0:
            if not self.config.tbt_profile_strict:
                fallback = self._select_concurrent_profile(
                    decode_tokens=decode_tokens,
                    prefill_tokens=prefill_tokens,
                )
                return _TBTProfileSelection(
                    profile=fallback[0],
                    reason="tbt_profile_unavailable_ewma_fallback",
                    state_covered=False,
                    prefill_mode=mode,
                    predicted_per_token_ms=0.0,
                    bucket_context_tokens=0,
                    bucket_decode_requests=0,
                    elapsed_ms=elapsed_ms,
                    predicted_next_ms=0.0,
                    slo_ms=slo_ms,
                    feasible=False,
                    emergency=False,
                    overload=False,
                    fallback=True,
                    fallback_ewma=fallback,
                )
            target = max(
                candidates,
                key=lambda profile: (
                    profile.decode_sms,
                    -profile.prefill_sms,
                    profile.profile_id,
                ),
            )
            return _TBTProfileSelection(
                profile=target,
                reason="tbt_profile_unavailable_max_decode_concurrent",
                state_covered=False,
                prefill_mode=mode,
                predicted_per_token_ms=0.0,
                bucket_context_tokens=0,
                bucket_decode_requests=0,
                elapsed_ms=elapsed_ms,
                predicted_next_ms=0.0,
                slo_ms=slo_ms,
                feasible=False,
                emergency=True,
                overload=True,
                fallback=False,
            )

        predictions = {}
        for profile in candidates:
            prediction = self._tbt_profile.predict(
                profile_id=profile.profile_id,
                decode_context_tokens=decode_context_tokens,
                decode_requests=decode_request_count,
                prefill_mode=mode,
            )
            if prediction is not None:
                predictions[profile.profile_id] = prediction

        current_prediction = (
            self._tbt_profile.predict(
                profile_id=self.current_profile.profile_id,
                decode_context_tokens=decode_context_tokens,
                decode_requests=decode_request_count,
                prefill_mode=mode,
            )
            if self.current_profile.is_concurrent
            else None
        )
        current_next_ms = (
            elapsed_ms
            + current_prediction.p95_per_token_tbt_ms
            + self.config.tbt_profile_margin_ms
            if current_prediction is not None
            else math.inf
        )
        emergency = current_next_ms > slo_ms

        considered = candidates
        if (
            not emergency
            and self.config.max_step_sms > 0
            and self.current_profile.is_concurrent
        ):
            bounded = [
                profile
                for profile in candidates
                if abs(profile.decode_sms - self.current_profile.decode_sms)
                <= self.config.max_step_sms
            ]
            if bounded:
                considered = bounded

        feasible = []
        for profile in considered:
            prediction = predictions.get(profile.profile_id)
            if prediction is None:
                continue
            predicted_next_ms = (
                elapsed_ms
                + prediction.p95_per_token_tbt_ms
                + self.config.tbt_profile_margin_ms
            )
            if predicted_next_ms <= slo_ms:
                feasible.append((profile, prediction, predicted_next_ms))

        if feasible:
            target, prediction, predicted_next_ms = min(
                feasible,
                key=lambda item: (
                    item[0].decode_sms,
                    -item[0].prefill_sms,
                    item[2],
                    item[0].profile_id,
                ),
            )
            reason = (
                "tbt_slo_emergency_min_feasible"
                if emergency
                else "tbt_slo_min_decode_sm"
            )
            return _TBTProfileSelection(
                profile=target,
                reason=reason,
                state_covered=True,
                prefill_mode=mode,
                predicted_per_token_ms=prediction.p95_per_token_tbt_ms,
                bucket_context_tokens=prediction.bucket_context_tokens,
                bucket_decode_requests=prediction.bucket_decode_requests,
                elapsed_ms=elapsed_ms,
                predicted_next_ms=predicted_next_ms,
                slo_ms=slo_ms,
                feasible=True,
                emergency=emergency,
                overload=False,
                fallback=False,
            )

        if not predictions and not self.config.tbt_profile_strict:
            fallback = self._select_concurrent_profile(
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
            )
            return _TBTProfileSelection(
                profile=fallback[0],
                reason="tbt_state_uncovered_ewma_fallback",
                state_covered=False,
                prefill_mode=mode,
                predicted_per_token_ms=0.0,
                bucket_context_tokens=0,
                bucket_decode_requests=0,
                elapsed_ms=elapsed_ms,
                predicted_next_ms=0.0,
                slo_ms=slo_ms,
                feasible=False,
                emergency=True,
                overload=False,
                fallback=True,
                fallback_ewma=fallback,
            )

        target = max(
            candidates,
            key=lambda profile: (
                profile.decode_sms,
                -profile.prefill_sms,
                profile.profile_id,
            ),
        )
        prediction = predictions.get(target.profile_id)
        if prediction is not None:
            predicted_per_token_ms = prediction.p95_per_token_tbt_ms
            predicted_next_ms = (
                elapsed_ms
                + predicted_per_token_ms
                + self.config.tbt_profile_margin_ms
            )
            bucket_context_tokens = prediction.bucket_context_tokens
            bucket_decode_requests = prediction.bucket_decode_requests
            state_covered = True
            reason = "tbt_slo_overload_max_decode_concurrent"
        else:
            predicted_per_token_ms = 0.0
            predicted_next_ms = 0.0
            bucket_context_tokens = 0
            bucket_decode_requests = 0
            state_covered = False
            reason = (
                "tbt_state_uncovered_max_decode_concurrent"
                if not predictions
                else "tbt_target_uncovered_max_decode_concurrent"
            )
        return _TBTProfileSelection(
            profile=target,
            reason=reason,
            state_covered=state_covered,
            prefill_mode=mode,
            predicted_per_token_ms=predicted_per_token_ms,
            bucket_context_tokens=bucket_context_tokens,
            bucket_decode_requests=bucket_decode_requests,
            elapsed_ms=elapsed_ms,
            predicted_next_ms=predicted_next_ms,
            slo_ms=slo_ms,
            feasible=False,
            emergency=True,
            overload=True,
            fallback=False,
        )

    def _tbt_feedback_state_key(
        self,
        *,
        decode_context_tokens: int,
        decode_request_count: int,
        prefill_mode: str,
    ) -> str:
        bucket_size = self.config.feedback_context_bucket_tokens
        context = max(0, int(decode_context_tokens))
        context_bucket = (
            int(math.ceil(context / bucket_size)) * bucket_size
            if context > 0
            else 0
        )
        mode = prefill_mode.strip().lower()
        if mode not in {"native", "ring"}:
            mode = "native"
        return f"ctx{context_bucket}_n{max(1, decode_request_count)}_{mode}"

    def _tbt_feedback_candidates(self) -> list[PDSMProfile]:
        grid_offset = (
            self.config.feedback_initial_decode_sms
            % self.config.feedback_step_sms
        )
        candidates = sorted(
            (
                profile
                for profile in self.config.profiles
                if profile.is_concurrent
                and profile.prefill_sms
                >= self.config.feedback_min_prefill_sms
                and profile.decode_sms % self.config.feedback_step_sms
                == grid_offset
            ),
            key=lambda profile: profile.decode_sms,
        )
        if not candidates:
            raise RuntimeError(
                "tbt_slo_feedback requires a concurrent profile that "
                "preserves the configured Prefill SM floor"
            )
        return candidates

    @staticmethod
    def _feedback_profile_at_or_above(
        candidates: list[PDSMProfile], decode_sms: int
    ) -> PDSMProfile:
        return next(
            (
                profile
                for profile in candidates
                if profile.decode_sms >= decode_sms
            ),
            candidates[-1],
        )

    def _feedback_selection(
        self,
        *,
        profile: PDSMProfile,
        reason: str,
        state_key: str,
        mode: str,
        observed_p95_ms: float,
        observed_slo_ms: float,
        observed_ratio: float,
        observation_count: int,
        observation_epoch: int,
        window_ready: bool,
        violation: bool,
        state: _TBTFeedbackState,
    ) -> _TBTFeedbackSelection:
        return _TBTFeedbackSelection(
            profile=profile,
            reason=reason,
            state_key=state_key,
            mode=mode,
            observed_p95_ms=max(0.0, observed_p95_ms),
            observed_slo_ms=max(0.0, observed_slo_ms),
            observed_ratio=max(0.0, observed_ratio),
            observation_count=max(0, observation_count),
            observation_epoch=max(0, observation_epoch),
            window_ready=window_ready,
            violation=violation,
            minimum_feasible_decode_sms=(
                state.minimum_feasible_decode_sms
            ),
            maximum_infeasible_decode_sms=(
                state.maximum_infeasible_decode_sms
            ),
        )

    def _select_tbt_feedback_concurrent_profile(
        self,
        *,
        decode_context_tokens: int,
        decode_request_count: int,
        prefill_mode: str,
        decode_tbt_slo_ms: float,
        decode_tbt_elapsed_ms: float,
        observed_p95_ms: float,
        observed_slo_ms: float,
        observed_ratio: float,
        observation_count: int,
        observation_epoch: int,
    ) -> _TBTFeedbackSelection:
        """Choose a 2-SM feedback step without mutating GPU execution state."""

        candidates = self._tbt_feedback_candidates()
        state_key = self._tbt_feedback_state_key(
            decode_context_tokens=decode_context_tokens,
            decode_request_count=decode_request_count,
            prefill_mode=prefill_mode,
        )
        mode = state_key.rsplit("_", 1)[-1]
        state = self._tbt_feedback_states.setdefault(
            state_key, _TBTFeedbackState()
        )
        effective_slo_ms = (
            observed_slo_ms
            if observed_slo_ms > 0
            else max(0.0, decode_tbt_slo_ms)
        )
        effective_ratio = (
            (observed_p95_ms + self.config.tbt_profile_margin_ms)
            / effective_slo_ms
            if effective_slo_ms > 0 and observed_p95_ms > 0
            else max(0.0, observed_ratio)
        )

        initial_decode_sms = (
            state.minimum_feasible_decode_sms
            or self.config.feedback_initial_decode_sms
        )
        initial = self._feedback_profile_at_or_above(
            candidates, initial_decode_sms
        )
        state_changed = state_key != self._tbt_feedback_active_state_key
        self._tbt_feedback_active_state_key = state_key
        if state_changed or not self.current_profile.is_concurrent:
            state.last_evaluation_epoch = observation_epoch
            state.consecutive_slack_windows = 0
            state.pending_decode_sms = initial.decode_sms
            state.pending_reason = (
                "tbt_feedback_cached_feasible"
                if state.minimum_feasible_decode_sms > 0
                else "tbt_feedback_initial"
            )

        if state.pending_decode_sms > 0:
            pending = self._feedback_profile_at_or_above(
                candidates, state.pending_decode_sms
            )
            if self.current_profile.decode_sms != pending.decode_sms:
                return self._feedback_selection(
                    profile=pending,
                    reason=state.pending_reason,
                    state_key=state_key,
                    mode=mode,
                    observed_p95_ms=observed_p95_ms,
                    observed_slo_ms=effective_slo_ms,
                    observed_ratio=effective_ratio,
                    observation_count=observation_count,
                    observation_epoch=observation_epoch,
                    window_ready=False,
                    violation=False,
                    state=state,
                )
            state.pending_decode_sms = 0
            state.pending_reason = ""

        if effective_slo_ms <= 0.0:
            return self._feedback_selection(
                profile=self.current_profile,
                reason="tbt_feedback_slo_unavailable_hold",
                state_key=state_key,
                mode=mode,
                observed_p95_ms=observed_p95_ms,
                observed_slo_ms=effective_slo_ms,
                observed_ratio=effective_ratio,
                observation_count=observation_count,
                observation_epoch=observation_epoch,
                window_ready=False,
                violation=False,
                state=state,
            )

        try:
            current_index = next(
                index
                for index, profile in enumerate(candidates)
                if profile.profile_id == self.current_profile.profile_id
            )
        except StopIteration:
            return self._feedback_selection(
                profile=initial,
                reason="tbt_feedback_reenter_concurrent",
                state_key=state_key,
                mode=mode,
                observed_p95_ms=observed_p95_ms,
                observed_slo_ms=effective_slo_ms,
                observed_ratio=effective_ratio,
                observation_count=observation_count,
                observation_epoch=observation_epoch,
                window_ready=False,
                violation=False,
                state=state,
            )

        current = candidates[current_index]
        epoch_delta = max(0, observation_epoch - state.last_evaluation_epoch)
        window_ready = (
            observation_count >= self.config.feedback_window_tokens
            and epoch_delta >= self.config.feedback_window_tokens
        )
        elapsed_violation = (
            effective_slo_ms > 0
            and decode_tbt_elapsed_ms
            + self.config.tbt_profile_margin_ms
            >= effective_slo_ms
        )
        if not window_ready and not elapsed_violation:
            return self._feedback_selection(
                profile=current,
                reason="tbt_feedback_collect_window",
                state_key=state_key,
                mode=mode,
                observed_p95_ms=observed_p95_ms,
                observed_slo_ms=effective_slo_ms,
                observed_ratio=effective_ratio,
                observation_count=observation_count,
                observation_epoch=observation_epoch,
                window_ready=False,
                violation=False,
                state=state,
            )

        state.last_evaluation_epoch = observation_epoch
        violation = elapsed_violation or (
            window_ready and effective_slo_ms > 0 and effective_ratio > 1.0
        )
        if violation:
            state.maximum_infeasible_decode_sms = max(
                state.maximum_infeasible_decode_sms,
                current.decode_sms,
            )
            state.consecutive_slack_windows = 0
            if current_index + 1 >= len(candidates):
                state.last_direction = "up"
                return self._feedback_selection(
                    profile=current,
                    reason="tbt_feedback_overload_max_decode",
                    state_key=state_key,
                    mode=mode,
                    observed_p95_ms=observed_p95_ms,
                    observed_slo_ms=effective_slo_ms,
                    observed_ratio=effective_ratio,
                    observation_count=observation_count,
                    observation_epoch=observation_epoch,
                    window_ready=window_ready,
                    violation=True,
                    state=state,
                )
            higher = candidates[current_index + 1]
            if state.minimum_feasible_decode_sms == higher.decode_sms:
                state.hold_windows_remaining = (
                    self.config.feedback_hold_windows
                )
            state.last_direction = "up"
            state.pending_decode_sms = higher.decode_sms
            state.pending_reason = "tbt_feedback_violation_step_up"
            return self._feedback_selection(
                profile=higher,
                reason=state.pending_reason,
                state_key=state_key,
                mode=mode,
                observed_p95_ms=observed_p95_ms,
                observed_slo_ms=effective_slo_ms,
                observed_ratio=effective_ratio,
                observation_count=observation_count,
                observation_epoch=observation_epoch,
                window_ready=window_ready,
                violation=True,
                state=state,
            )

        if effective_slo_ms > 0:
            if (
                state.minimum_feasible_decode_sms <= 0
                or current.decode_sms < state.minimum_feasible_decode_sms
            ):
                state.minimum_feasible_decode_sms = current.decode_sms

        if state.hold_windows_remaining > 0:
            state.hold_windows_remaining -= 1
            state.consecutive_slack_windows = 0
            return self._feedback_selection(
                profile=current,
                reason="tbt_feedback_boundary_hold",
                state_key=state_key,
                mode=mode,
                observed_p95_ms=observed_p95_ms,
                observed_slo_ms=effective_slo_ms,
                observed_ratio=effective_ratio,
                observation_count=observation_count,
                observation_epoch=observation_epoch,
                window_ready=window_ready,
                violation=False,
                state=state,
            )

        if current_index == 0:
            state.consecutive_slack_windows = 0
            return self._feedback_selection(
                profile=current,
                reason="tbt_feedback_min_decode_hold",
                state_key=state_key,
                mode=mode,
                observed_p95_ms=observed_p95_ms,
                observed_slo_ms=effective_slo_ms,
                observed_ratio=effective_ratio,
                observation_count=observation_count,
                observation_epoch=observation_epoch,
                window_ready=window_ready,
                violation=False,
                state=state,
            )

        lower = candidates[current_index - 1]
        adjacent_known_infeasible = (
            state.maximum_infeasible_decode_sms >= lower.decode_sms
        )
        threshold = (
            self.config.feedback_reprobe_ratio
            if adjacent_known_infeasible
            else self.config.feedback_slack_ratio
        )
        if effective_ratio <= threshold:
            state.consecutive_slack_windows += 1
        else:
            state.consecutive_slack_windows = 0
        if (
            state.consecutive_slack_windows
            < self.config.feedback_decrease_windows
        ):
            return self._feedback_selection(
                profile=current,
                reason="tbt_feedback_slo_hold",
                state_key=state_key,
                mode=mode,
                observed_p95_ms=observed_p95_ms,
                observed_slo_ms=effective_slo_ms,
                observed_ratio=effective_ratio,
                observation_count=observation_count,
                observation_epoch=observation_epoch,
                window_ready=window_ready,
                violation=False,
                state=state,
            )

        state.consecutive_slack_windows = 0
        if adjacent_known_infeasible:
            state.maximum_infeasible_decode_sms = min(
                state.maximum_infeasible_decode_sms,
                max(0, lower.decode_sms - self.config.feedback_step_sms),
            )
        state.last_direction = "down"
        state.pending_decode_sms = lower.decode_sms
        state.pending_reason = "tbt_feedback_slack_step_down"
        return self._feedback_selection(
            profile=lower,
            reason=state.pending_reason,
            state_key=state_key,
            mode=mode,
            observed_p95_ms=observed_p95_ms,
            observed_slo_ms=effective_slo_ms,
            observed_ratio=effective_ratio,
            observation_count=observation_count,
            observation_epoch=observation_epoch,
            window_ready=window_ready,
            violation=False,
            state=state,
        )

    def _select_tbt_predictive_profile(
        self, *, candidates: list[PDSMProfile], decode_context_tokens: int,
        decode_request_count: int, prefill_mode: str, decode_tbt_slo_ms: float,
        decode_tbt_elapsed_ms: float, decode_min_tbt_slo_ms: float,
        decode_inflight: bool, transition_age_ms: float,
    ) -> _TBTProfileSelection:
        """Forecast Decode demand independently of Prefill chunk submission.

        Lead is a growth horizon, not a stop-the-world gap. Cutover is charged
        only to a profile change. An already launched Decode cannot be sped up
        retrospectively: its residual GPU service is explicitly unknown.
        """
        mode = prefill_mode
        self._tbt_predictive_mode = mode
        observed_lead, observed_cutover = self._tbt_predictive_transition_highwater.get(mode, (0.0, 0.0))
        lead = max(self.config.tbt_predictive_lead_ms, observed_lead, transition_age_ms)
        cutover = max(self.config.tbt_predictive_cutover_ms, observed_cutover)
        growth = math.ceil(lead / self.config.tbt_predictive_min_step_ms) + 1
        context = max(1, decode_context_tokens)
        horizon_context = context + growth
        slo = decode_tbt_slo_ms if decode_tbt_slo_ms > 0 else self.config.tbt_default_slo_ms
        min_slo = decode_min_tbt_slo_ms if decode_min_tbt_slo_ms > 0 else slo
        margin = self.config.tbt_profile_margin_ms
        elapsed = max(0.0, decode_tbt_elapsed_ms)
        predictions = {}
        feasible = []
        next_gap = {}
        assert self._tbt_profile is not None
        for profile in candidates:
            prediction = self._tbt_profile.predict_context_envelope(
                profile_id=profile.profile_id,
                context_start=context, context_end=horizon_context,
                decode_requests=decode_request_count, prefill_mode=mode)
            if prediction is None:
                continue
            predictions[profile.profile_id] = prediction
            switching = profile != self.current_profile
            pause = cutover if switching else 0.0
            cost = prediction.p95_per_token_tbt_ms + margin + pause
            # With no source Decode partition, preparation really is waiting
            # for Decode. Otherwise old-profile Decode continues during lead.
            next_gap[profile.profile_id] = elapsed + prediction.p95_per_token_tbt_ms + margin + (
                lead if switching and self.current_profile.decode_sms == 0 else pause)
            if (min_slo > 0 and cost <= min_slo
                    and (decode_inflight or next_gap[profile.profile_id] <= slo)):
                feasible.append(profile)
        target = (min(feasible, key=lambda p: p.decode_sms) if feasible else
                  max(candidates, key=lambda p: p.decode_sms))
        prediction = predictions.get(target.profile_id)
        upshift = target.decode_sms > self.current_profile.decode_sms
        deadline_at_risk = slo > 0 and elapsed + margin >= slo
        diagnostics = {
            "pd_tbt_predictive_enabled": 1.0,
            "pd_tbt_predictive_reason": "minimum_forecast_feasible" if feasible else "uncovered_or_infeasible_max_decode",
            "pd_tbt_predictive_lead_ms": lead,
            "pd_tbt_predictive_cutover_ms": cutover,
            "pd_tbt_predictive_observed_lead_ms": observed_lead,
            "pd_tbt_predictive_observed_cutover_ms": observed_cutover,
            "pd_tbt_predictive_growth_tokens": growth,
            "pd_tbt_predictive_context_end": horizon_context,
            "pd_tbt_predictive_min_slo_ms": min_slo,
            "pd_tbt_predictive_deadline_slack_ms": slo - elapsed,
            "pd_tbt_predictive_deadline_at_risk": float(deadline_at_risk),
            "pd_tbt_predictive_current_deadline_evaluated": float(not decode_inflight),
            "pd_tbt_predictive_forecast_feasible": float(bool(feasible)),
            "pd_tbt_predictive_covered_profiles": len(predictions),
            "pd_tbt_predictive_upshift": float(upshift),
        }
        return _TBTProfileSelection(
            profile=target, reason="tbt_predictive_" + str(diagnostics["pd_tbt_predictive_reason"]),
            state_covered=prediction is not None, prefill_mode=mode,
            predicted_per_token_ms=prediction.p95_per_token_tbt_ms if prediction else 0.0,
            bucket_context_tokens=prediction.bucket_context_tokens if prediction else 0,
            bucket_decode_requests=prediction.bucket_decode_requests if prediction else 0,
            elapsed_ms=elapsed,
            # -1 denotes unknown residual of already submitted work, not zero.
            predicted_next_ms=(-1.0 if decode_inflight else next_gap.get(target.profile_id, -1.0)),
            slo_ms=slo, feasible=bool(feasible) and not decode_inflight,
            emergency=upshift or deadline_at_risk, overload=not feasible,
            fallback=False, predictive=diagnostics)

    def plan(
        self,
        *,
        decode_ready: bool,
        prefill_ready: bool,
        decode_inflight: bool,
        prefill_inflight: bool,
        transition_active: bool = False,
        decode_tokens: int = 0,
        prefill_tokens: int = 0,
        decode_context_tokens: int = 0,
        decode_request_count: int = 0,
        prefill_mode: str = "native",
        decode_tbt_slo_ms: float = 0.0,
        decode_tbt_elapsed_ms: float = 0.0,
        decode_tbt_observed_p95_ms: float = 0.0,
        decode_tbt_observed_slo_ms: float = 0.0,
        decode_tbt_observed_ratio: float = 0.0,
        decode_tbt_observation_count: int = 0,
        decode_tbt_observation_epoch: int = 0,
        capacity_serialized: bool = False,
        decode_min_tbt_slo_ms: float = 0.0,
        transition_age_ms: float = 0.0,
    ) -> PDSMPlan:
        self.decision_index += 1
        tbt_profile_state_covered = False
        tbt_profile_prefill_mode = ""
        tbt_profile_predicted_per_token_ms = 0.0
        tbt_profile_bucket_context_tokens = 0
        tbt_profile_bucket_decode_requests = 0
        tbt_elapsed_ms = max(0.0, float(decode_tbt_elapsed_ms))
        tbt_predicted_next_ms = 0.0
        tbt_slo_ms = (
            float(decode_tbt_slo_ms)
            if decode_tbt_slo_ms > 0
            else self.config.tbt_default_slo_ms
        )
        tbt_slo_feasible = False
        tbt_slo_emergency = False
        tbt_slo_overload = False
        tbt_profile_fallback = False
        tbt_feedback_selection: _TBTFeedbackSelection | None = None
        tbt_predictive = None
        serialized_full_sm_targeted = False
        capacity_serialized = bool(capacity_serialized)
        capacity_serialized_fixed_split = bool(
            capacity_serialized
            and decode_ready
            and not prefill_ready
            and not self.config.serialized_full_sm
        )
        # A capacity-serialized Prefill is logical demand that intentionally
        # cannot submit.  Only the fixed-split ablation represents that demand
        # to the concurrent policy so the Prefill share stays idle.  Natural
        # one-lane periods remain work-conserving in both experiment cases.
        allocation_prefill_ready = bool(
            prefill_ready or capacity_serialized_fixed_split
        )
        selection_prefill_tokens = prefill_tokens
        if capacity_serialized_fixed_split and selection_prefill_tokens <= 0:
            selection_prefill_tokens = 1
        if not self.enabled:
            return PDSMPlan(
                profile=self.current_profile,
                target_profile=self.current_profile,
                epoch=self.epoch,
                reason="disabled",
                enabled=False,
                shadow=self.config.shadow,
                switched=False,
                deferred=False,
                allocator_policy=self.config.policy,
                serialized_full_sm_enabled=(
                    self.config.serialized_full_sm
                ),
                capacity_serialized=capacity_serialized,
                capacity_serialized_fixed_split=(
                    capacity_serialized_fixed_split
                ),
            )

        if self.predictive_enabled and decode_ready:
            candidates = [p for p in self.config.profiles if p.is_concurrent] if allocation_prefill_ready else [
                max(self.config.profiles, key=lambda p: p.decode_sms)]
            selection = self._select_tbt_predictive_profile(
                candidates=candidates, decode_context_tokens=decode_context_tokens,
                decode_request_count=decode_request_count,
                prefill_mode=prefill_mode if allocation_prefill_ready else "none",
                decode_tbt_slo_ms=decode_tbt_slo_ms,
                decode_tbt_elapsed_ms=decode_tbt_elapsed_ms,
                decode_min_tbt_slo_ms=decode_min_tbt_slo_ms,
                decode_inflight=decode_inflight, transition_age_ms=transition_age_ms)
            target, reason = selection.profile, selection.reason
            predicted_decode_ms = selection.predicted_per_token_ms
            predicted_prefill_ms = bubble_sm_ms = improvement = 0.0
            tbt_profile_state_covered = selection.state_covered
            tbt_profile_prefill_mode = selection.prefill_mode
            tbt_profile_predicted_per_token_ms = selection.predicted_per_token_ms
            tbt_profile_bucket_context_tokens = selection.bucket_context_tokens
            tbt_profile_bucket_decode_requests = selection.bucket_decode_requests
            tbt_elapsed_ms, tbt_predicted_next_ms, tbt_slo_ms = selection.elapsed_ms, selection.predicted_next_ms, selection.slo_ms
            tbt_slo_feasible, tbt_slo_emergency, tbt_slo_overload = selection.feasible, selection.emergency, selection.overload
            tbt_predictive = selection.predictive
            serialized_full_sm_targeted = bool(capacity_serialized and target.is_decode_only)
        elif decode_ready and not allocation_prefill_ready:
            if self.config.policy == _TBT_FEEDBACK_POLICY:
                self._tbt_feedback_active_state_key = ""
            target = max(
                self.config.profiles,
                key=lambda profile: profile.decode_sms,
            )
            reason = (
                "capacity_serialized_decode_full_sm"
                if capacity_serialized
                else "decode_only_full_sm"
            )
            serialized_full_sm_targeted = bool(
                capacity_serialized
                and self.config.serialized_full_sm
                and target.is_decode_only
            )
            predicted_decode_ms = 0.0
            predicted_prefill_ms = 0.0
            bubble_sm_ms = 0.0
            improvement = 0.0
            if (
                self.config.policy == _TBT_PROFILED_POLICY
                and self._tbt_profile is not None
                and tbt_slo_ms > 0
            ):
                prediction = self._tbt_profile.predict(
                    profile_id=target.profile_id,
                    decode_context_tokens=decode_context_tokens,
                    decode_requests=decode_request_count,
                    prefill_mode="none",
                )
                tbt_profile_prefill_mode = "none"
                if prediction is not None:
                    tbt_profile_state_covered = True
                    tbt_profile_predicted_per_token_ms = (
                        prediction.p95_per_token_tbt_ms
                    )
                    tbt_profile_bucket_context_tokens = (
                        prediction.bucket_context_tokens
                    )
                    tbt_profile_bucket_decode_requests = (
                        prediction.bucket_decode_requests
                    )
                    tbt_predicted_next_ms = (
                        tbt_elapsed_ms
                        + prediction.p95_per_token_tbt_ms
                        + self.config.tbt_profile_margin_ms
                    )
                    tbt_slo_feasible = tbt_predicted_next_ms <= tbt_slo_ms
                    tbt_slo_emergency = not tbt_slo_feasible
                    tbt_slo_overload = not tbt_slo_feasible
                elif self.config.tbt_profile_strict:
                    tbt_slo_emergency = True
                    tbt_slo_overload = True
        elif allocation_prefill_ready and not decode_ready:
            if self.config.policy == _TBT_FEEDBACK_POLICY:
                self._tbt_feedback_active_state_key = ""
            target = max(
                self.config.profiles,
                key=lambda profile: profile.prefill_sms,
            )
            reason = "prefill_only_full_sm"
            predicted_decode_ms = 0.0
            predicted_prefill_ms = 0.0
            bubble_sm_ms = 0.0
            improvement = 0.0
        elif decode_ready and allocation_prefill_ready:
            if self.config.policy == _TBT_PROFILED_POLICY:
                tbt_selection = (
                    self._select_tbt_profiled_concurrent_profile(
                        decode_tokens=decode_tokens,
                        prefill_tokens=selection_prefill_tokens,
                        decode_context_tokens=decode_context_tokens,
                        decode_request_count=decode_request_count,
                        prefill_mode=prefill_mode,
                        decode_tbt_slo_ms=decode_tbt_slo_ms,
                        decode_tbt_elapsed_ms=decode_tbt_elapsed_ms,
                    )
                )
                target = tbt_selection.profile
                reason = tbt_selection.reason
                if tbt_selection.fallback_ewma is not None:
                    (
                        _fallback_profile,
                        predicted_decode_ms,
                        predicted_prefill_ms,
                        bubble_sm_ms,
                        improvement,
                        _fallback_reason,
                    ) = tbt_selection.fallback_ewma
                else:
                    predicted_decode_ms = (
                        tbt_selection.predicted_per_token_ms
                    )
                    predicted_prefill_ms = 0.0
                    bubble_sm_ms = 0.0
                    improvement = 0.0
                tbt_profile_state_covered = tbt_selection.state_covered
                tbt_profile_prefill_mode = tbt_selection.prefill_mode
                tbt_profile_predicted_per_token_ms = (
                    tbt_selection.predicted_per_token_ms
                )
                tbt_profile_bucket_context_tokens = (
                    tbt_selection.bucket_context_tokens
                )
                tbt_profile_bucket_decode_requests = (
                    tbt_selection.bucket_decode_requests
                )
                tbt_elapsed_ms = tbt_selection.elapsed_ms
                tbt_predicted_next_ms = tbt_selection.predicted_next_ms
                tbt_slo_ms = tbt_selection.slo_ms
                tbt_slo_feasible = tbt_selection.feasible
                tbt_slo_emergency = tbt_selection.emergency
                tbt_slo_overload = tbt_selection.overload
                tbt_profile_fallback = tbt_selection.fallback
            elif self.config.policy == _TBT_FEEDBACK_POLICY:
                tbt_feedback_selection = (
                    self._select_tbt_feedback_concurrent_profile(
                        decode_context_tokens=decode_context_tokens,
                        decode_request_count=decode_request_count,
                        prefill_mode=prefill_mode,
                        decode_tbt_slo_ms=decode_tbt_slo_ms,
                        decode_tbt_elapsed_ms=decode_tbt_elapsed_ms,
                        observed_p95_ms=decode_tbt_observed_p95_ms,
                        observed_slo_ms=decode_tbt_observed_slo_ms,
                        observed_ratio=decode_tbt_observed_ratio,
                        observation_count=decode_tbt_observation_count,
                        observation_epoch=decode_tbt_observation_epoch,
                    )
                )
                target = tbt_feedback_selection.profile
                reason = tbt_feedback_selection.reason
                predicted_decode_ms = (
                    tbt_feedback_selection.observed_p95_ms
                )
                predicted_prefill_ms = 0.0
                bubble_sm_ms = 0.0
                improvement = 0.0
                tbt_profile_prefill_mode = tbt_feedback_selection.mode
                tbt_slo_ms = tbt_feedback_selection.observed_slo_ms
                tbt_slo_feasible = (
                    tbt_feedback_selection.window_ready
                    and not tbt_feedback_selection.violation
                )
                tbt_slo_emergency = tbt_feedback_selection.violation
                tbt_slo_overload = (
                    reason == "tbt_feedback_overload_max_decode"
                )
            else:
                (
                    target,
                    predicted_decode_ms,
                    predicted_prefill_ms,
                    bubble_sm_ms,
                    improvement,
                    reason,
                ) = self._select_concurrent_profile(
                    decode_tokens=decode_tokens,
                    prefill_tokens=selection_prefill_tokens,
                )
            if capacity_serialized_fixed_split:
                reason = f"capacity_serialized_fixed_split:{reason}"
        else:
            if self.config.policy == _TBT_FEEDBACK_POLICY:
                self._tbt_feedback_active_state_key = ""
            target = self.current_profile
            reason = "no_runnable_phase"
            predicted_decode_ms = 0.0
            predicted_prefill_ms = 0.0
            bubble_sm_ms = 0.0
            improvement = 0.0

        if self.predictive_enabled and not decode_ready:
            self.arm_tbt_deadline(request_id="", deadline_at=0.0)
            self._tbt_predictive_pending_transition = None
            self._tbt_predictive_transition_request = None

        bypass_cooldown = (
            target.is_decode_only
            or target.is_prefill_only
            or (decode_ready and self.current_profile.decode_sms == 0)
            or (
                allocation_prefill_ready
                and self.current_profile.prefill_sms == 0
            )
            or tbt_slo_emergency
            or tbt_slo_overload
        )
        in_cooldown = (
            self.decision_index - self.last_switch_decision
            <= self.config.cooldown_decisions
        )
        if target != self.current_profile and in_cooldown and not bypass_cooldown:
            return PDSMPlan(
                profile=self.current_profile,
                target_profile=target,
                epoch=self.epoch,
                reason="cooldown_hold",
                enabled=True,
                shadow=self.config.shadow,
                switched=False,
                deferred=True,
                predicted_decode_ms=predicted_decode_ms,
                predicted_prefill_ms=predicted_prefill_ms,
                predicted_bubble_sm_ms=bubble_sm_ms,
                improvement=improvement,
                allocator_policy=self.config.policy,
                tbt_profile_state_covered=tbt_profile_state_covered,
                tbt_profile_prefill_mode=tbt_profile_prefill_mode,
                tbt_profile_predicted_per_token_ms=(
                    tbt_profile_predicted_per_token_ms
                ),
                tbt_profile_bucket_context_tokens=(
                    tbt_profile_bucket_context_tokens
                ),
                tbt_profile_bucket_decode_requests=(
                    tbt_profile_bucket_decode_requests
                ),
                tbt_elapsed_ms=tbt_elapsed_ms,
                tbt_predicted_next_ms=tbt_predicted_next_ms,
                tbt_slo_ms=tbt_slo_ms,
                tbt_slo_feasible=tbt_slo_feasible,
                tbt_slo_emergency=tbt_slo_emergency,
                tbt_slo_overload=tbt_slo_overload,
                tbt_profile_fallback=tbt_profile_fallback,
                tbt_feedback=tbt_feedback_selection,
                tbt_predictive=tbt_predictive,
                serialized_full_sm_enabled=(
                    self.config.serialized_full_sm
                ),
                serialized_full_sm_targeted=(
                    serialized_full_sm_targeted
                ),
                capacity_serialized=capacity_serialized,
                capacity_serialized_fixed_split=(
                    capacity_serialized_fixed_split
                ),
            )

        if decode_inflight or prefill_inflight or transition_active:
            return PDSMPlan(
                profile=self.current_profile,
                target_profile=target,
                epoch=self.epoch,
                reason="inflight_profile_hold",
                enabled=True,
                shadow=self.config.shadow,
                switched=False,
                deferred=target != self.current_profile,
                predicted_decode_ms=predicted_decode_ms,
                predicted_prefill_ms=predicted_prefill_ms,
                predicted_bubble_sm_ms=bubble_sm_ms,
                improvement=improvement,
                allocator_policy=self.config.policy,
                tbt_profile_state_covered=tbt_profile_state_covered,
                tbt_profile_prefill_mode=tbt_profile_prefill_mode,
                tbt_profile_predicted_per_token_ms=(
                    tbt_profile_predicted_per_token_ms
                ),
                tbt_profile_bucket_context_tokens=(
                    tbt_profile_bucket_context_tokens
                ),
                tbt_profile_bucket_decode_requests=(
                    tbt_profile_bucket_decode_requests
                ),
                tbt_elapsed_ms=tbt_elapsed_ms,
                tbt_predicted_next_ms=tbt_predicted_next_ms,
                tbt_slo_ms=tbt_slo_ms,
                tbt_slo_feasible=tbt_slo_feasible,
                tbt_slo_emergency=tbt_slo_emergency,
                tbt_slo_overload=tbt_slo_overload,
                tbt_profile_fallback=tbt_profile_fallback,
                tbt_feedback=tbt_feedback_selection,
                tbt_predictive=tbt_predictive,
                serialized_full_sm_enabled=(
                    self.config.serialized_full_sm
                ),
                serialized_full_sm_targeted=(
                    serialized_full_sm_targeted
                ),
                capacity_serialized=capacity_serialized,
                capacity_serialized_fixed_split=(
                    capacity_serialized_fixed_split
                ),
            )

        switched = target != self.current_profile
        if switched:
            self.current_profile = target
            self.epoch += 1
            self.last_switch_decision = self.decision_index
        return PDSMPlan(
            profile=self.current_profile,
            target_profile=target,
            epoch=self.epoch,
            reason=reason,
            enabled=True,
            shadow=self.config.shadow,
            switched=switched,
            deferred=False,
            predicted_decode_ms=predicted_decode_ms,
            predicted_prefill_ms=predicted_prefill_ms,
            predicted_bubble_sm_ms=bubble_sm_ms,
            improvement=improvement,
            allocator_policy=self.config.policy,
            tbt_profile_state_covered=tbt_profile_state_covered,
            tbt_profile_prefill_mode=tbt_profile_prefill_mode,
            tbt_profile_predicted_per_token_ms=(
                tbt_profile_predicted_per_token_ms
            ),
            tbt_profile_bucket_context_tokens=(
                tbt_profile_bucket_context_tokens
            ),
            tbt_profile_bucket_decode_requests=(
                tbt_profile_bucket_decode_requests
            ),
            tbt_elapsed_ms=tbt_elapsed_ms,
            tbt_predicted_next_ms=tbt_predicted_next_ms,
            tbt_slo_ms=tbt_slo_ms,
            tbt_slo_feasible=tbt_slo_feasible,
            tbt_slo_emergency=tbt_slo_emergency,
            tbt_slo_overload=tbt_slo_overload,
            tbt_profile_fallback=tbt_profile_fallback,
            tbt_feedback=tbt_feedback_selection,
            tbt_predictive=tbt_predictive,
            serialized_full_sm_enabled=self.config.serialized_full_sm,
            serialized_full_sm_targeted=serialized_full_sm_targeted,
            capacity_serialized=capacity_serialized,
            capacity_serialized_fixed_split=(
                capacity_serialized_fixed_split
            ),
        )
