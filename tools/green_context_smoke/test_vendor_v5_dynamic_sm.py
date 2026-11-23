#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU smoke test for vendor_v5 dynamic GreenContext profiles."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    vendor_root = Path(os.environ.get("VLLM_VENDOR_DIR", root / "vendor_v5"))
    sys.path.insert(0, str(vendor_root))

    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT", "1")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL", "1")
    os.environ.setdefault("VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT", "1")
    os.environ.setdefault("VLLM_PD_DYNAMIC_SM_ENABLED", "1")
    os.environ.setdefault("VLLM_PD_DYNAMIC_SM_STRICT", "1")
    os.environ.setdefault("VLLM_PD_DYNAMIC_SM_TOTAL_SMS", "40")
    os.environ.setdefault(
        "VLLM_PD_DYNAMIC_SM_PROFILES",
        "40/0,32/8,24/16,20/20,16/24,8/32,0/40",
    )
    os.environ.setdefault("VLLM_PD_DYNAMIC_SM_EAGER_PROFILES", "1")

    import torch

    from vllm.model_executor.layers.green_context import (
        green_context_phase_stream,
        green_context_state,
        initialize_dynamic_green_context_profiles,
    )
    from vllm.pd_sm_profiles import parse_pd_sm_profiles

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = int(os.environ.get("VLLM_OPERATOR_SELECTIVE_PD_GREEN_DEVICE", "0"))
    torch.cuda.set_device(device)
    actual_sms = torch.cuda.get_device_properties(device).multi_processor_count
    configured_sms = int(os.environ["VLLM_PD_DYNAMIC_SM_TOTAL_SMS"])
    if actual_sms != configured_sms:
        raise RuntimeError(
            f"configured total SMs {configured_sms} != device SMs {actual_sms}"
        )

    profiles = parse_pd_sm_profiles(
        os.environ["VLLM_PD_DYNAMIC_SM_PROFILES"],
        total_sms=actual_sms,
        alignment=2,
    )
    bank_config = initialize_dynamic_green_context_profiles()
    if bank_config.get("enabled") != 1:
        raise RuntimeError(
            f"dynamic GreenContext profile bank was not initialized: {bank_config}"
        )
    if bank_config.get("created_profiles") != len(profiles):
        raise RuntimeError(f"not all profiles were created: {bank_config}")
    if bank_config.get("sm_counts_verified") != 1:
        raise RuntimeError(f"profile SM counts were not verified: {bank_config}")
    source = torch.arange(4096, device="cuda", dtype=torch.float32)
    for epoch, profile in enumerate(profiles, start=1):
        for phase, sms in (
            ("decode", profile.decode_sms),
            ("prefill", profile.prefill_sms),
        ):
            if sms == 0:
                continue
            with green_context_phase_stream(
                phase,
                profile_id=profile.profile_id,
                profile_epoch=epoch,
            ) as active:
                if not active:
                    raise RuntimeError(
                        f"profile {profile.profile_id} phase {phase} inactive"
                    )
                output = source * float(epoch + 1) + float(sms)
                state = green_context_state()
            torch.cuda.synchronize()
            expected = source.cpu() * float(epoch + 1) + float(sms)
            if not torch.equal(output.cpu(), expected):
                raise RuntimeError(
                    f"profile {profile.profile_id} phase {phase} output mismatch"
                )
            if state["pd_dynamic_sm_green_context_profile_id"] != profile.profile_id:
                raise RuntimeError(f"wrong active profile state: {state}")
            if state["green_sm_count_verified"] != 1.0:
                raise RuntimeError(f"SM count was not bridge-verified: {state}")
            if state["green_decode_sms"] != profile.decode_sms:
                raise RuntimeError(f"decode SM mismatch: {state}")
            if state["green_prefill_sms"] != profile.prefill_sms:
                raise RuntimeError(f"prefill SM mismatch: {state}")
            print(f"[OK] {profile.profile_id} {phase}: {state}")

    final_state = green_context_state()
    expected_handoffs = max(0, len(profiles) - 1)
    if final_state["pd_dynamic_sm_green_context_handoff_count"] != float(
        expected_handoffs
    ):
        raise RuntimeError(
            f"expected {expected_handoffs} profile handoffs: {final_state}"
        )
    if (
        final_state["pd_dynamic_sm_green_context_bank_profile_id"]
        != profiles[-1].profile_id
    ):
        raise RuntimeError(f"wrong final bank profile: {final_state}")

    print("[OK] vendor_v5 dynamic P/D SM profile bank validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
