# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.observability import (
    compact_worker_debug_timings,
    detailed_iteration_debug_enabled,
    iteration_debug_sample_every_n,
    lightweight_timeline_enabled,
    strict_bounded_lightweight_timeline_enabled,
)


def test_tbt_controller_does_not_enable_detailed_iteration_debug() -> None:
    env = {
        "VLLM_PD_DYNAMIC_SM_ENABLED": "1",
        "VLLM_PD_DYNAMIC_SM_POLICY": "tbt_slo_feedback",
        "VLLM_TBT_SLO_OBSERVABILITY_ENABLED": "1",
    }
    assert not detailed_iteration_debug_enabled(env=env)


def test_explicit_trace_enables_detailed_iteration_debug() -> None:
    assert detailed_iteration_debug_enabled(
        env={"VLLM_ITERATION_METRICS_JSONL": "/tmp/iteration.jsonl"}
    )
    assert detailed_iteration_debug_enabled(
        env={"VLLM_FULL_ITERATION_DEBUG": "1"}
    )


def test_lightweight_trace_path_does_not_enable_detailed_debug() -> None:
    env = {
        "VLLM_LIGHTWEIGHT_TIMELINE_TRACE": "1",
        "VLLM_ITERATION_METRICS_JSONL": "/tmp/iteration.jsonl",
        "VLLM_DECODE_TIMELINE_JSONL": "/tmp/decode.jsonl",
    }
    assert lightweight_timeline_enabled(env)
    assert not detailed_iteration_debug_enabled(env=env)


def test_full_debug_overrides_lightweight_trace() -> None:
    env = {
        "VLLM_LIGHTWEIGHT_TIMELINE_TRACE": "1",
        "VLLM_ITERATION_METRICS_JSONL": "/tmp/iteration.jsonl",
        "VLLM_FULL_ITERATION_DEBUG": "1",
    }
    assert detailed_iteration_debug_enabled(env=env)


def test_lightweight_trace_overrides_implicit_virtual_context_timing() -> None:
    env = {
        "VLLM_LIGHTWEIGHT_TIMELINE_TRACE": "1",
        "VLLM_LIGHTWEIGHT_TIMELINE_STRICT_BOUNDED": "1",
        "VLLM_ITERATION_METRICS_JSONL": "/tmp/iteration.jsonl",
    }
    assert strict_bounded_lightweight_timeline_enabled(env)
    assert not detailed_iteration_debug_enabled(
        virtual_context_timing=True,
        env=env,
    )


def test_legacy_lightweight_trace_can_restore_virtual_context_debug() -> None:
    env = {
        "VLLM_LIGHTWEIGHT_TIMELINE_TRACE": "1",
        "VLLM_ITERATION_METRICS_JSONL": "/tmp/iteration.jsonl",
    }
    assert not strict_bounded_lightweight_timeline_enabled(env)
    assert detailed_iteration_debug_enabled(
        virtual_context_timing=True,
        env=env,
    )


def test_full_debug_still_overrides_virtual_context_lightweight_trace() -> None:
    env = {
        "VLLM_LIGHTWEIGHT_TIMELINE_TRACE": "1",
        "VLLM_FULL_ITERATION_DEBUG": "1",
    }
    assert detailed_iteration_debug_enabled(
        virtual_context_timing=True,
        env=env,
    )


def test_compact_worker_debug_timings_keeps_hot_path_fields_only() -> None:
    compact = compact_worker_debug_timings(
        {
            "worker_execute_start_ts": 1.0,
            "worker_context_ring_calls": 16,
            "pd_dynamic_sm_execution_decode_sms": 38,
            "pd_output_owner_generation": 4,
            "virtual_context_allocator_full_snapshot": "large",
            "dram_page_registry_cumulative_snapshot": "large",
        }
    )
    assert compact == {
        "worker_execute_start_ts": 1.0,
        "worker_context_ring_calls": 16,
        "pd_dynamic_sm_execution_decode_sms": 38,
        "pd_output_owner_generation": 4,
    }


def test_iteration_debug_sampling_defaults_and_validation() -> None:
    assert iteration_debug_sample_every_n(env={}) == 100
    assert iteration_debug_sample_every_n(
        env={"VLLM_ITERATION_METRICS_SAMPLE_EVERY_N": "17"}
    ) == 17
    assert iteration_debug_sample_every_n(
        env={"VLLM_ITERATION_METRICS_SAMPLE_EVERY_N": "invalid"}
    ) == 100
