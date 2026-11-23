#!/usr/bin/env python3
"""Import-free contracts for vendor_v8 Context Ring execution modes."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / (
    "vendor_v8/vllm/v1/attention/ops/tiered_blockwise_backend.py"
)
EXECUTOR = ROOT / (
    "vendor_v8/vllm/v1/attention/ops/tiered_blockwise_attention.py"
)
MODEL_RUNNER = ROOT / "vendor_v8/vllm/v1/worker/gpu_model_runner.py"
POLICY_HELPER = ROOT / "tools/pd_eval_policy_env.sh"
SHORT_RUNNER = ROOT / "run_vendor_v8_short_validation.sh"


def require_markers(path: Path, markers: tuple[str, ...]) -> str:
    source = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise AssertionError(f"{path}: missing markers {missing}")
    return source


def require_top_level_functions(path: Path, names: tuple[str, ...]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(names) - found)
    if missing:
        raise AssertionError(f"{path}: missing functions {missing}")


def run_dry_plan(variant: str, **overrides: str) -> str:
    env = os.environ.copy()
    for name in (
        "VLLM_PD_EVAL_RING_BATCHED_PARTIAL",
        "VLLM_PD_EVAL_RING_BATCHED_PARTIAL_STRICT",
        "VLLM_PD_EVAL_RING_BATCHED_WORKSPACE_BYTES",
        "VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS",
    ):
        env.pop(name, None)
    env.update(
        {
            "SYSTEM_VARIANT": variant,
            "PD_DYNAMIC_SM_POLICY": "tbt_slo_feedback",
            "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "0",
            **overrides,
        }
    )
    completed = subprocess.run(
        ["bash", str(SHORT_RUNNER), "--dry-run"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def main() -> int:
    backend = require_markers(
        BACKEND,
        (
            "_RUNTIME_METRICS_TLS = local()",
            "_record_tiered_context_ring_runtime_metrics(",
            "worker_context_ring_batched_partial_successes",
            "worker_context_ring_host_group_loop_calls",
        ),
    )
    require_top_level_functions(
        BACKEND,
        (
            "reset_tiered_context_ring_runtime_metrics",
            "consume_tiered_context_ring_runtime_metrics",
            "_record_tiered_context_ring_runtime_metrics",
        ),
    )
    if backend.count("_record_tiered_context_ring_runtime_metrics(") < 3:
        raise AssertionError(
            "backend must define the recorder and call it from both the "
            "single-request and ragged paths"
        )

    require_markers(
        EXECUTOR,
        (
            "materialize_group_batch(",
            "copy_stream.wait_event(source_ready_event)",
            "current_stream.wait_event(copy_done_event)",
            "run_batched_tiered_partial_attention(",
            "merge_batched_tiered_partial_states(",
            "self.staging_buffer.mark_slot_read_done(",
            "context_ring_host_group_loop_bypassed=1",
        ),
    )
    model_runner = require_markers(
        MODEL_RUNNER,
        (
            "reset_tiered_context_ring_runtime_metrics()",
            "consume_tiered_context_ring_runtime_metrics()",
        ),
    )
    # A resumed segment has its own reset/consume path. File-wide first-match
    # order compares DIFFERENT methods after continuation refactors.
    runner_tree = ast.parse(model_runner)
    runner_class = next(node for node in runner_tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner")
    methods = {node.name: ast.get_source_segment(model_runner, node)
               for node in runner_class.body if isinstance(node, ast.FunctionDef)}
    normal_forward = methods["_execute_model_impl"]
    reset_at = normal_forward.index(
        "reset_tiered_context_ring_runtime_metrics()"
    )
    forward_at = normal_forward.index("model_output = self._model_forward(")
    consume_at = normal_forward.index(
        "consume_tiered_context_ring_runtime_metrics()"
    )
    if not reset_at < forward_at < consume_at:
        raise AssertionError(
            "Context Ring metrics must bracket model forward enqueue"
        )
    resumed_forward = methods["_resume_ttft_prefill_continuation"]
    if not (resumed_forward.index("reset_tiered_context_ring_runtime_metrics()")
            < resumed_forward.index("next(state.continuation)")
            < resumed_forward.index("self._accumulate_ttft_group_debug(")):
        raise AssertionError("Context Ring metrics must bracket resumed segments")
    if "consume_tiered_context_ring_runtime_metrics()" not in methods["_accumulate_ttft_group_debug"]:
        raise AssertionError("Resumed segment must consume its Ring counters")

    require_markers(
        POLICY_HELPER,
        (
            'VLLM_PD_EVAL_RING_BATCHED_PARTIAL:-0',
            'VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS:-${_PD_EVAL_INPUT_RING_STREAMING_WAVE_GROUPS:-1}',
        ),
    )
    require_markers(
        SHORT_RUNNER,
        (
            'DEFAULT_RING_BATCHED_PARTIAL=0',
            'DEFAULT_RING_BATCHED_PARTIAL_STRICT=0',
            'DEFAULT_RING_STREAMING_WAVE_GROUPS=1',
            'VLLM_PD_EVAL_RING_BATCHED_PARTIAL="${RING_BATCHED_PARTIAL}"',
            'VLLM_PD_EVAL_RING_BATCHED_PARTIAL_STRICT="${RING_BATCHED_PARTIAL_STRICT}"',
        ),
    )

    full_plan = run_dry_plan("full_vendor_v8")
    if "ring_batched_partial=0 strict=0 workspace_bytes=0" not in full_plan:
        raise AssertionError("full_vendor_v8 enabled all-context batching")
    if "ring_streaming_wave_groups=1 wave_tokens=2048" not in full_plan:
        raise AssertionError("full_vendor_v8 did not restore 2K group streaming")
    if (
        "ring_prefill_query_tokens=2048..2048 active_decode_native_prefill_tokens=2048 "
        "active_decode_ring_prefill_tokens=2048 elastic_active_decode=0 "
        "dynamic=0 historical_group_tokens=2048"
        not in full_plan
    ):
        raise AssertionError("full_vendor_v8 did not fix the Ring query path at 2K")
    if (
        "ring_execution=historical_group_streaming_double_buffered"
        not in full_plan
    ):
        raise AssertionError("full_vendor_v8 did not report the slot pipeline")

    dynamic_plan = run_dry_plan("dynamic_pd_hbm")
    if "ring_batched_partial=0 strict=0" not in dynamic_plan:
        raise AssertionError("dynamic_pd_hbm changed its default Ring policy")

    wave_override_plan = run_dry_plan(
        "full_vendor_v8",
        VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS="4",
    )
    if "ring_streaming_wave_groups=4 wave_tokens=8192" not in wave_override_plan:
        raise AssertionError("explicit streaming-wave ablation was not preserved")

    batched_override_plan = run_dry_plan(
        "full_vendor_v8",
        VLLM_PD_EVAL_RING_BATCHED_PARTIAL="1",
        VLLM_PD_EVAL_RING_BATCHED_PARTIAL_STRICT="1",
        VLLM_PD_EVAL_RING_BATCHED_WORKSPACE_BYTES="1073741824",
    )
    if (
        "ring_batched_partial=1 strict=1 workspace_bytes=1073741824"
        not in batched_override_plan
    ):
        raise AssertionError("explicit all-context ablation was not preserved")
    if "ring_streaming_wave_groups=1 wave_tokens=2048" not in batched_override_plan:
        raise AssertionError("all-context ablation did not retain wave compatibility")

    print("[OK] vendor_v8 Context Ring execution contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
