# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(os.environ.get("VLLM_VENDOR_DIR", ROOT / "vendor_v5")).resolve()


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


# Loading vllm/__init__.py imports torch. Build only the package shells needed
# by these pure-policy tests so they remain runnable on a CPU-only dev host.
_package("vllm", VENDOR / "vllm")
_package("vllm.v1", VENDOR / "vllm" / "v1")
_package("vllm.v1.engine", VENDOR / "vllm" / "v1" / "engine")
_load("vllm.pd_sm_profiles", VENDOR / "vllm" / "pd_sm_profiles.py")
_load(
    "vllm.v1.engine.pd_sm_allocator",
    VENDOR / "vllm" / "v1" / "engine" / "pd_sm_allocator.py",
)

from vllm.pd_sm_profiles import auto_pd_sm_profiles, parse_pd_sm_profiles
from vllm.v1.engine.pd_sm_allocator import (
    PDComputeAllocationController,
    PDSMAllocatorConfig,
)


HAS_SERIALIZED_FULL_SM = (
    "serialized_full_sm" in PDSMAllocatorConfig.__dataclass_fields__
)


def _config(**overrides) -> PDSMAllocatorConfig:
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
        "cooldown_decisions": 2,
        "max_step_sms": 8,
        "switch_penalty_ms": 0.0,
    }
    values.update(overrides)
    return PDSMAllocatorConfig(**values)


def test_t4_auto_profiles_include_both_solo_states() -> None:
    assert [profile.profile_id for profile in auto_pd_sm_profiles(40, 2)] == [
        "d0_p40",
        "d8_p32",
        "d16_p24",
        "d20_p20",
        "d24_p16",
        "d32_p8",
        "d40_p0",
    ]


def test_profile_parser_rejects_partial_device_profile() -> None:
    try:
        parse_pd_sm_profiles("40/0,20/10,0/40", total_sms=40)
    except ValueError as exc:
        assert "expected all 40" in str(exc)
    else:
        raise AssertionError("partial-device profile should be rejected")


def test_disabled_config_ignores_invalid_experimental_values() -> None:
    names = (
        "VLLM_PD_DYNAMIC_SM_ENABLED",
        "VLLM_PD_DYNAMIC_SM_TOTAL_SMS",
        "VLLM_PD_DYNAMIC_SM_EWMA_ALPHA",
        "VLLM_PD_DYNAMIC_SM_PROFILES",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ["VLLM_PD_DYNAMIC_SM_ENABLED"] = "0"
        os.environ["VLLM_PD_DYNAMIC_SM_TOTAL_SMS"] = "invalid"
        os.environ["VLLM_PD_DYNAMIC_SM_EWMA_ALPHA"] = "invalid"
        os.environ["VLLM_PD_DYNAMIC_SM_PROFILES"] = "invalid"
        config = PDSMAllocatorConfig.from_env()
        assert not config.enabled
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_enabled_t4_environment_matches_fixed_initial_split() -> None:
    values = {
        "VLLM_PD_DYNAMIC_SM_ENABLED": "1",
        "VLLM_PD_DYNAMIC_SM_TOTAL_SMS": "40",
        "VLLM_PD_DYNAMIC_SM_PROFILES": "auto",
        "VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS": "20",
        "VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS": "20",
        "VLLM_PD_SERIALIZED_FULL_SM": "0",
    }
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        config = PDSMAllocatorConfig.from_env()
        assert config.enabled
        assert config.total_sms == 40
        assert config.initial_profile.profile_id == "d20_p20"
        if HAS_SERIALIZED_FULL_SM:
            assert not config.serialized_full_sm
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_solo_profile_and_quiescent_paired_handoff() -> None:
    controller = PDComputeAllocationController(_config())
    decode_only = controller.plan(
        decode_ready=True,
        prefill_ready=False,
        decode_inflight=False,
        prefill_inflight=False,
        decode_tokens=1,
    )
    assert decode_only.profile.profile_id == "d40_p0"
    assert decode_only.switched

    held = controller.plan(
        decode_ready=True,
        prefill_ready=True,
        decode_inflight=True,
        prefill_inflight=False,
        decode_tokens=1,
        prefill_tokens=2048,
    )
    assert held.profile.profile_id == "d40_p0"
    assert held.deferred
    assert held.reason == "inflight_profile_hold"

    paired = controller.plan(
        decode_ready=True,
        prefill_ready=True,
        decode_inflight=False,
        prefill_inflight=False,
        decode_tokens=1,
        prefill_tokens=2048,
    )
    assert paired.profile.is_concurrent
    assert paired.profile.prefill_sms > 0

    prefill_only = controller.plan(
        decode_ready=False,
        prefill_ready=True,
        decode_inflight=False,
        prefill_inflight=False,
        prefill_tokens=2048,
    )
    assert prefill_only.profile.profile_id == "d0_p40"
    if HAS_SERIALIZED_FULL_SM:
        assert prefill_only.serialized_full_sm_enabled
        assert not prefill_only.serialized_full_sm_targeted


def test_serialized_ablation_differs_only_at_capacity_boundary() -> None:
    # This file remains usable against the older vendor snapshots selected by
    # VLLM_VENDOR_DIR.  The ablation itself is introduced by vendor_v8.
    if not HAS_SERIALIZED_FULL_SM:
        return
    controller = PDComputeAllocationController(
        _config(serialized_full_sm=False, cooldown_decisions=0)
    )
    controller.commit_profile_at_quiescence(
        "d32_p8",
        expected_epoch=0,
    )

    decode_only = controller.plan(
        decode_ready=True,
        prefill_ready=False,
        decode_inflight=False,
        prefill_inflight=False,
        decode_tokens=1,
    )
    assert decode_only.profile.profile_id == "d40_p0"
    assert decode_only.reason == "decode_only_full_sm"
    assert not decode_only.serialized_full_sm_enabled
    assert not decode_only.serialized_full_sm_targeted
    assert (
        decode_only.as_debug_dict()[
            "pd_dynamic_sm_serialized_full_sm_enabled"
        ]
        == 0.0
    )

    controller.commit_profile_at_quiescence(
        "d32_p8",
        expected_epoch=2,
    )
    capacity_serialized = controller.plan(
        decode_ready=True,
        prefill_ready=False,
        decode_inflight=False,
        prefill_inflight=False,
        decode_tokens=1,
        capacity_serialized=True,
    )
    assert capacity_serialized.profile.is_concurrent
    assert capacity_serialized.profile.prefill_sms > 0
    assert capacity_serialized.reason.startswith(
        "capacity_serialized_fixed_split:"
    )
    assert capacity_serialized.capacity_serialized
    assert capacity_serialized.capacity_serialized_fixed_split

    prefill_only = controller.plan(
        decode_ready=False,
        prefill_ready=True,
        decode_inflight=False,
        prefill_inflight=False,
        prefill_tokens=2048,
    )
    assert prefill_only.profile.profile_id == "d0_p40"
    assert prefill_only.reason == "prefill_only_full_sm"
    assert not prefill_only.serialized_full_sm_targeted


def test_capacity_serialized_full_sm_promotes_decode() -> None:
    if not HAS_SERIALIZED_FULL_SM:
        return
    controller = PDComputeAllocationController(
        _config(serialized_full_sm=True, cooldown_decisions=0)
    )
    plan = controller.plan(
        decode_ready=True,
        prefill_ready=False,
        decode_inflight=False,
        prefill_inflight=False,
        decode_tokens=1,
        capacity_serialized=True,
    )
    assert plan.profile.profile_id == "d40_p0"
    assert plan.reason == "capacity_serialized_decode_full_sm"
    assert plan.serialized_full_sm_targeted
    assert plan.capacity_serialized
    assert not plan.capacity_serialized_fixed_split


def test_feedback_moves_concurrent_split_toward_prefill() -> None:
    controller = PDComputeAllocationController(_config(cooldown_decisions=0))
    controller.observe(
        phase="decode",
        num_tokens=1,
        elapsed_ms=20.0,
        profile_id="d20_p20",
    )
    controller.observe(
        phase="prefill",
        num_tokens=100,
        elapsed_ms=100.0,
        profile_id="d20_p20",
    )
    plan = controller.plan(
        decode_ready=True,
        prefill_ready=True,
        decode_inflight=False,
        prefill_inflight=False,
        decode_tokens=1,
        prefill_tokens=100,
    )
    assert plan.profile.is_concurrent
    assert plan.profile.prefill_sms > plan.profile.decode_sms


if __name__ == "__main__":
    test_t4_auto_profiles_include_both_solo_states()
    test_profile_parser_rejects_partial_device_profile()
    test_disabled_config_ignores_invalid_experimental_values()
    test_enabled_t4_environment_matches_fixed_initial_split()
    test_solo_profile_and_quiescent_paired_handoff()
    test_serialized_ablation_differs_only_at_capacity_boundary()
    test_capacity_serialized_full_sm_promotes_decode()
    test_feedback_moves_concurrent_split_toward_prefill()
    print("[OK] dynamic P/D SM allocator unit tests passed")
