#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for vendor_v8 profiled TBT-first SM allocation."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v8"
MEASUREMENT_CONTRACT = {
    "token_count_source": "choice.token_ids",
    "gap_field": "estimated_per_token_gap_ms",
    "unit": "ms_per_output_token",
    "chunk_gap_normalization": (
        "interarrival_ms_divided_by_emitted_token_count"
    ),
    "first_nonempty_chunk": "excluded_as_ttft",
}


def _package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_package("vllm", VENDOR / "vllm")
_package("vllm.v1", VENDOR / "vllm/v1")
_package("vllm.v1.engine", VENDOR / "vllm/v1/engine")
_load("vllm.pd_sm_profiles", VENDOR / "vllm/pd_sm_profiles.py")
_load(
    "vllm.v1.engine.pd_tbt_profile",
    VENDOR / "vllm/v1/engine/pd_tbt_profile.py",
)
_load(
    "vllm.v1.engine.pd_sm_allocator",
    VENDOR / "vllm/v1/engine/pd_sm_allocator.py",
)

from vllm.pd_sm_profiles import auto_pd_sm_profiles
from vllm.v1.engine.pd_sm_allocator import (
    PDComputeAllocationController,
    PDSMAllocatorConfig,
)
from vllm.v1.engine.pd_tbt_profile import TBTOverlapProfile


def _profile_payload() -> dict:
    points = []
    steps = {
        "d8_p32": 170.0,
        "d16_p24": 135.0,
        "d20_p20": 110.0,
        "d24_p16": 90.0,
        "d32_p8": 70.0,
        "d40_p0": 60.0,
    }
    for profile_id, step_ms in steps.items():
        modes = ("none",) if profile_id == "d40_p0" else ("native", "ring")
        for mode in modes:
            points.append(
                {
                    "profile_id": profile_id,
                    "max_decode_context_tokens": 100000,
                    "max_decode_requests": 4,
                    "prefill_mode": mode,
                    "p95_per_token_tbt_ms": (
                        step_ms + 5.0 if mode == "ring" else step_ms
                    ),
                }
            )
    return {
        "schema_version": 2,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "points": points,
    }


def _config(profile_path: str, **overrides) -> PDSMAllocatorConfig:
    profiles = auto_pd_sm_profiles(40, 2)
    values = {
        "enabled": True,
        "shadow": False,
        "strict": True,
        "total_sms": 40,
        "alignment": 2,
        "profiles": profiles,
        "initial_profile": next(
            profile for profile in profiles if profile.profile_id == "d20_p20"
        ),
        "ewma_alpha": 0.25,
        "decode_target_ms": 0.0,
        "min_improvement": 0.0,
        "cooldown_decisions": 10,
        "max_step_sms": 8,
        "switch_penalty_ms": 0.0,
        "policy": "tbt_slo_profiled",
        "tbt_profile_path": profile_path,
        "tbt_default_slo_ms": 125.0,
        "tbt_profile_margin_ms": 5.0,
        "tbt_profile_strict": True,
    }
    values.update(overrides)
    return PDSMAllocatorConfig(**values)


def _write_profile(directory: str) -> str:
    path = Path(directory) / "profile.json"
    path.write_text(json.dumps(_profile_payload()), encoding="utf-8")
    return str(path)


def test_lookup_uses_tightest_covering_bucket_and_conservative_duplicate() -> None:
    model = TBTOverlapProfile.from_dict(
        {
            "schema_version": 2,
            "measurement_contract": MEASUREMENT_CONTRACT,
            "points": [
                {
                    "profile_id": "d20_p20",
                    "max_decode_context_tokens": 200000,
                    "max_decode_requests": 8,
                    "prefill_mode": "any",
                    "p95_per_token_tbt_ms": 200.0,
                },
                {
                    "profile_id": "d20_p20",
                    "max_decode_context_tokens": 100000,
                    "max_decode_requests": 4,
                    "prefill_mode": "ring",
                    "p95_per_token_tbt_ms": 105.0,
                },
                {
                    "profile_id": "d20_p20",
                    "max_decode_context_tokens": 100000,
                    "max_decode_requests": 4,
                    "prefill_mode": "ring",
                    "p95_per_token_tbt_ms": 110.0,
                },
            ],
        }
    )
    prediction = model.predict(
        profile_id="d20_p20",
        decode_context_tokens=75000,
        decode_requests=3,
        prefill_mode="ring",
    )
    assert prediction is not None
    assert prediction.bucket_context_tokens == 100000
    assert prediction.p95_per_token_tbt_ms == 110.0


def test_legacy_decode_step_profile_is_rejected() -> None:
    try:
        TBTOverlapProfile.from_dict(
            {
                "schema_version": 1,
                "points": [
                    {
                        "profile_id": "d20_p20",
                        "max_decode_context_tokens": 100000,
                        "max_decode_requests": 1,
                        "prefill_mode": "native",
                        "p95_decode_step_ms": 100.0,
                    }
                ],
            }
        )
    except ValueError as exc:
        assert "exact per-token choice.token_ids" in str(exc)
    else:
        raise AssertionError("legacy decode-step profile was accepted")


def test_environment_selects_profiled_policy_and_loads_profile() -> None:
    with tempfile.TemporaryDirectory() as directory:
        values = {
            "VLLM_PD_DYNAMIC_SM_ENABLED": "1",
            "VLLM_PD_DYNAMIC_SM_POLICY": "tbt_slo_profiled",
            "VLLM_PD_DYNAMIC_SM_TOTAL_SMS": "40",
            "VLLM_PD_DYNAMIC_SM_PROFILES": "auto",
            "VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS": "20",
            "VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS": "20",
            "VLLM_PD_TBT_PROFILE_PATH": _write_profile(directory),
            "VLLM_PD_TBT_DEFAULT_SLO_MS": "125",
            "VLLM_PD_TBT_PROFILE_MARGIN_MS": "5",
            "VLLM_PD_TBT_PROFILE_STRICT": "1",
        }
        previous = {name: os.environ.get(name) for name in values}
        try:
            os.environ.update(values)
            config = PDSMAllocatorConfig.from_env()
            assert config.policy == "tbt_slo_profiled"
            assert config.initial_profile.profile_id == "d20_p20"
            controller = PDComputeAllocationController(config)
            assert controller.enabled
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def test_selects_minimum_decode_sms_that_meets_tbt_slo() -> None:
    with tempfile.TemporaryDirectory() as directory:
        controller = PDComputeAllocationController(
            _config(_write_profile(directory))
        )
        plan = controller.plan(
            decode_ready=True,
            prefill_ready=True,
            decode_inflight=False,
            prefill_inflight=False,
            decode_tokens=3,
            prefill_tokens=2048,
            decode_context_tokens=75000,
            decode_request_count=3,
            prefill_mode="native",
            decode_tbt_slo_ms=125.0,
            decode_tbt_elapsed_ms=10.0,
        )
        assert plan.profile.profile_id == "d20_p20"
        assert plan.tbt_slo_feasible
        assert plan.tbt_predicted_next_ms == 125.0
        assert not plan.tbt_slo_overload


def test_emergency_bypasses_step_and_cooldown_at_quiescent_boundary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        controller = PDComputeAllocationController(
            _config(
                _write_profile(directory),
                max_step_sms=2,
                cooldown_decisions=100,
            )
        )
        plan = controller.plan(
            decode_ready=True,
            prefill_ready=True,
            decode_inflight=False,
            prefill_inflight=False,
            decode_context_tokens=90000,
            decode_request_count=4,
            prefill_mode="ring",
            decode_tbt_slo_ms=110.0,
            decode_tbt_elapsed_ms=10.0,
        )
        assert plan.profile.profile_id == "d24_p16"
        assert plan.tbt_slo_emergency
        assert not plan.tbt_slo_overload
        assert plan.switched


def test_inflight_future_defers_even_an_emergency_switch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        controller = PDComputeAllocationController(
            _config(_write_profile(directory))
        )
        plan = controller.plan(
            decode_ready=True,
            prefill_ready=True,
            decode_inflight=True,
            prefill_inflight=False,
            decode_context_tokens=90000,
            decode_request_count=4,
            prefill_mode="ring",
            decode_tbt_slo_ms=110.0,
            decode_tbt_elapsed_ms=10.0,
        )
        assert plan.profile.profile_id == "d20_p20"
        assert plan.target_profile.profile_id == "d24_p16"
        assert plan.reason == "inflight_profile_hold"
        assert plan.deferred
        assert plan.tbt_slo_emergency


def test_uncovered_state_is_explicit_overload_in_strict_mode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        controller = PDComputeAllocationController(
            _config(_write_profile(directory))
        )
        plan = controller.plan(
            decode_ready=True,
            prefill_ready=True,
            decode_inflight=False,
            prefill_inflight=False,
            decode_context_tokens=200000,
            decode_request_count=8,
            prefill_mode="ring",
            decode_tbt_slo_ms=125.0,
        )
        assert plan.profile.profile_id == "d32_p8"
        assert not plan.tbt_profile_state_covered
        assert plan.tbt_slo_overload
        assert not plan.tbt_profile_fallback


def test_missing_profile_strict_rejects_and_nonstrict_falls_back() -> None:
    with tempfile.TemporaryDirectory() as directory:
        missing_path = str(Path(directory) / "missing.json")
        try:
            PDComputeAllocationController(_config(missing_path))
        except ValueError as exc:
            assert "failed to load TBT overlap profile" in str(exc)
        else:
            raise AssertionError(
                "strict profiled policy accepted a missing file"
            )

        controller = PDComputeAllocationController(
            _config(missing_path, tbt_profile_strict=False)
        )
        plan = controller.plan(
            decode_ready=True,
            prefill_ready=True,
            decode_inflight=False,
            prefill_inflight=False,
            decode_tokens=1,
            prefill_tokens=2048,
            decode_tbt_slo_ms=125.0,
        )
        assert plan.tbt_profile_fallback
        assert not plan.tbt_slo_overload


if __name__ == "__main__":
    test_lookup_uses_tightest_covering_bucket_and_conservative_duplicate()
    test_legacy_decode_step_profile_is_rejected()
    test_environment_selects_profiled_policy_and_loads_profile()
    test_selects_minimum_decode_sms_that_meets_tbt_slo()
    test_emergency_bypasses_step_and_cooldown_at_quiescent_boundary()
    test_inflight_future_defers_even_an_emergency_switch()
    test_uncovered_state_is_explicit_overload_in_strict_mode()
    test_missing_profile_strict_rejects_and_nonstrict_falls_back()
    print("[OK] vendor_v8 profiled TBT SM allocator tests passed")
