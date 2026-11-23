#!/usr/bin/env python3
"""Import-free source contract for enqueue-barrier P/D recovery."""

from __future__ import annotations

import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = Path(
    os.environ.get("VLLM_PD_CONTRACT_VENDOR_DIR", ROOT / "vendor_v4")
).resolve()
CORE = VENDOR_DIR / "vllm/v1/engine/core.py"
EXECUTOR = VENDOR_DIR / "vllm/v1/executor/uniproc_executor.py"
MODEL_RUNNER = VENDOR_DIR / "vllm/v1/worker/gpu_model_runner.py"
SCHEDULER = VENDOR_DIR / "vllm/v1/core/sched/scheduler.py"
TRITON_ATTN = VENDOR_DIR / "vllm/v1/attention/backends/triton_attn.py"


def fail(message: str) -> None:
    raise AssertionError(message)


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    fail(f"missing function: {name}")


def call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute):
            names.append(func.attr)
        elif isinstance(func, ast.Name):
            names.append(func.id)
    return names


def main() -> int:
    source = CORE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    required_markers = (
        "VLLM_PD_DECOUPLED_PAIRED_RECOVERY",
        "pd_decoupled_paired_recovery_enabled",
        "pd_decoupled_decode_recovered_with_prefill_not_ready",
        "pd_decoupled_decode_post_recover_resubmit",
        '"submit_paired_decoupled"',
        '"pd_forward_enqueue_barrier_used"',
    )
    for marker in required_markers:
        if marker not in source:
            fail(f"missing source marker: {marker}")

    helper = function_node(
        tree,
        "_submit_pd_phase_pair_decoupled_to_queues",
    )
    helper_calls = call_names(helper)
    if helper_calls.count("execute_model_pd_pair") != 1:
        fail(
            "decoupled pair helper must use one paired worker RPC as the "
            "forward-enqueue barrier"
        )
    if "_submit_to_pd_phase_queue" in helper_calls:
        fail("decoupled pair helper must not start phase sampling before the barrier")
    if "_append_pd_phase_queue_item_after_paired_exec" not in helper_calls:
        fail("decoupled pair helper must create phase-local sample futures")
    helper_source = ast.get_source_segment(source, helper) or ""
    if "decoupled_recovery=True" not in helper_source:
        fail("decoupled phase queue items must be labeled as barrier recovery")

    submit = function_node(tree, "_try_submit_pd_paired_phase_launch")
    submit_source = ast.get_source_segment(source, submit) or ""
    if "self.pd_decoupled_paired_recovery_enabled" not in submit_source:
        fail("paired submit path does not gate the decoupled helper")
    if "_submit_pd_phase_pair_decoupled_to_queues" not in submit_source:
        fail("paired submit path does not invoke the decoupled helper")
    if "_submit_pd_phase_pair_to_queues" not in submit_source:
        fail("legacy shared-pair behavior must remain available when disabled")

    executor_source = EXECUTOR.read_text(encoding="utf-8")
    executor_tree = ast.parse(executor_source)
    pair_execute = function_node(executor_tree, "execute_model_pd_pair")
    pair_execute_source = ast.get_source_segment(executor_source, pair_execute) or ""
    if pair_execute_source.count("_pd_pair_phase_dependency") != 2:
        fail("paired execute future must project decode and prefill dependencies")
    sample = function_node(executor_tree, "sample_tokens")
    sample_source = ast.get_source_segment(executor_source, sample) or ""
    if "_pd_async_phase_execute_futures.pop" not in sample_source:
        fail("phase sampling must consume its projected enqueue dependency")

    runner_source = MODEL_RUNNER.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    context = next(
        (
            node
            for node in ast.walk(runner_tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "PDPhaseModelRunnerContext"
        ),
        None,
    )
    if context is None:
        fail("missing PDPhaseModelRunnerContext")
    context_source = ast.get_source_segment(runner_source, context) or ""
    for field_name in (
        "_pd_green_context_execute_phase",
        "_pd_green_context_execute_active",
        "_tiered_scheduler_budget",
        "_pd_green_context_sample_phase",
        "_pd_green_context_current_sample_phase",
        "_pd_green_context_current_sample_active",
    ):
        if field_name not in context_source:
            fail(f"phase-local runner context is missing {field_name}")

    for function_name in ("execute_model", "sample_tokens"):
        node = function_node(runner_tree, function_name)
        node_source = ast.get_source_segment(runner_source, node) or ""
        phase_context_markers = (
            "with self._use_pd_phase_model_runner_context(phase):",
            "with self._use_pd_phase_model_runner_context(execution_key):",
        )
        if not any(marker in node_source for marker in phase_context_markers):
            fail(f"{function_name} must enter phase context before mutable state access")

    scheduler_source = SCHEDULER.read_text(encoding="utf-8")
    scheduler_tree = ast.parse(scheduler_source)
    fcfs_guard = function_node(
        scheduler_tree,
        "_has_older_active_prefill_request",
    )
    fcfs_guard_source = ast.get_source_segment(
        scheduler_source,
        fcfs_guard,
    ) or ""
    if 'self._pd_pending_phase_request_ids.get(\n            "decode"' not in (
        fcfs_guard_source
    ):
        fail("FCFS guard must read pending decode phase ownership")
    if "candidate.request_id in pending_decode_request_ids" not in (
        fcfs_guard_source
    ):
        fail("pending decode requests must not block paired prefill admission")

    schedule = function_node(scheduler_tree, "schedule")
    schedule_source = ast.get_source_segment(scheduler_source, schedule) or ""
    if schedule_source.count("and self.virtual_context_enabled") < 2:
        fail(
            "native muxwise prefill must bypass both running and waiting "
            "virtual-context admission gates"
        )
    if "pd_native_active_decode_prefill_admission_enabled" not in schedule_source:
        fail("native active-decode prefill admission must be observable")
    if (
        schedule_source.count(
            "self._native_prefill_capacity_feasible_with_active_decode"
        )
        < 2
    ):
        fail("native capacity admission must cover running and waiting prefills")
    native_capacity = function_node(
        scheduler_tree,
        "_native_prefill_capacity_feasible_with_active_decode",
    )
    native_capacity_source = ast.get_source_segment(
        scheduler_source,
        native_capacity,
    ) or ""
    capacity_markers = (
        (
            "num_new_tokens",
            "decode_request.max_tokens",
            "decode_request.num_output_tokens",
            "chunk_required_blocks + decode_growth_reserve_blocks",
            "estimate_num_blocks_to_allocate",
            "required_blocks <= free_blocks",
            "native_chunk_plus_decode_reserve_exceeded",
        )
        if "native_chunk_plus_decode_reserve_exceeded" in native_capacity_source
        else (
            "request.num_prompt_tokens - request.num_computed_tokens",
            "estimate_num_blocks_to_allocate",
            "required_blocks <= free_blocks",
            "native_full_prefill_capacity_exceeded",
        )
    )
    for marker in capacity_markers:
        if marker not in native_capacity_source:
            fail(f"native chunk capacity gate is missing {marker}")
    if "waiting_break_defer_reason = \"kv_capacity\"" not in schedule_source:
        fail("native chunk capacity failure must be labeled kv_capacity")
    if "pd_native_active_decode_prefill_capacity_defer_count" not in (
        schedule_source
    ):
        fail("native whole-prefill capacity decisions must be observable")

    triton_source = TRITON_ATTN.read_text(encoding="utf-8")
    if (
        "self._virtual_context_enabled = virtual_context_enabled()"
        not in triton_source
    ):
        fail("Triton attention must cache the process-start master gate")
    if "self._virtual_context_enabled\n            and (" not in triton_source:
        fail("Triton tiered control must require the virtual-context master gate")
    if (
        "self._virtual_context_enabled\n                and has_pending_hbm_d2h_reads"
        not in triton_source
    ):
        fail("native Triton attention must skip pending tiered-copy checks")
    if (
        "if self._virtual_context_enabled and not disable_native_hbm_reload:"
        not in triton_source
    ):
        fail("native Triton attention must skip empty tiered reload calls")
    if (
        "should_attempt_tiered_backend = self._virtual_context_enabled and ("
        not in triton_source
    ):
        fail("native Triton attention must skip tiered backend attempts")

    if "self._virtual_context_enabled = virtual_context_enabled()" not in (
        runner_source
    ):
        fail("model runner must cache the process-start master gate")
    if "virtual_context_active = self._virtual_context_enabled" not in runner_source:
        fail("model-runner tiered planning must use the master gate")
    if "if not virtual_context_active:" not in runner_source:
        fail("model runner must disable tiered planning in native mode")
    if "tiered_control_required\n                and tiered_d2h_source_plan_enabled()" not in (
        runner_source
    ):
        fail("native model runner must skip tiered D2H source-plan construction")

    print(
        "[OK] enqueue-barrier paired P/D recovery contracts validated: "
        f"{VENDOR_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
