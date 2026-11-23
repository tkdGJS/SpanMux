#!/usr/bin/env python3
"""Import-free behavioral contract for the vendor_v3 P/D FCFS guard."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = ROOT / "vendor_v3/vllm/v1/core/sched/scheduler.py"
POLICY_HELPER = ROOT / "tools/pd_eval_policy_env.sh"


def fail(message: str) -> None:
    raise AssertionError(message)


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    fail(f"missing function: {name}")


def compile_fcfs_predicate(tree: ast.AST) -> type:
    predicate = copy.deepcopy(
        function_node(tree, "_has_older_active_prefill_request")
    )
    for node in ast.walk(predicate):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
        if isinstance(node, ast.arg):
            node.annotation = None
    test_class = ast.ClassDef(
        name="TestScheduler",
        bases=[],
        keywords=[],
        body=[predicate],
        decorator_list=[],
    )
    module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(SCHEDULER), "exec"), namespace)
    return namespace["TestScheduler"]  # type: ignore[return-value]


def request(
    request_id: str,
    *,
    arrival_time: float,
    arrival_iteration_id: int,
    computed_tokens: int = 0,
    prompt_tokens: int = 100,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        arrival_time=arrival_time,
        arrival_iteration_id=arrival_iteration_id,
        num_computed_tokens=computed_tokens,
        num_prompt_tokens=prompt_tokens,
    )


def main() -> int:
    source = SCHEDULER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    schedule_source = ast.get_source_segment(
        source, function_node(tree, "schedule")
    ) or ""

    for marker in (
        '"VLLM_PD_REQUEST_FCFS", ""',
        "self.pd_request_fcfs",
        'waiting_break_defer_reason = "fcfs_request_order"',
        '"pd_request_fcfs_enabled"',
        '"prefill_defer_reason_fcfs_request_order"',
        '"admission_blocked_by_fcfs_request_order"',
    ):
        if marker not in source:
            fail(f"missing vendor_v3 FCFS marker: {marker}")
    if "_has_older_active_prefill_request(request)" not in schedule_source:
        fail("waiting admission does not invoke the request-level FCFS guard")

    scheduler_type = compile_fcfs_predicate(tree)
    scheduler = scheduler_type()
    scheduler.running = []
    scheduler.requests = {}
    scheduler._pd_pending_request_ids = set()
    scheduler._debug_has_prefill_remaining = lambda req: (
        req.num_computed_tokens < req.num_prompt_tokens
    )

    candidate = request(
        "candidate",
        arrival_time=2.0,
        arrival_iteration_id=2,
    )
    older = request("older", arrival_time=1.0, arrival_iteration_id=1)
    scheduler.running = [older]
    if not scheduler._has_older_active_prefill_request(candidate):
        fail("an older running prefill must block a newer prefill")

    scheduler.running = []
    scheduler.requests = {older.request_id: older}
    scheduler._pd_pending_request_ids = {older.request_id}
    if not scheduler._has_older_active_prefill_request(candidate):
        fail("an older in-flight prefill must block a newer prefill")

    scheduler._pd_pending_request_ids = set()
    newer = request("newer", arrival_time=3.0, arrival_iteration_id=3)
    scheduler.running = [newer]
    if scheduler._has_older_active_prefill_request(candidate):
        fail("a newer active prefill must not block an older waiting prefill")

    decode_candidate = request(
        "decode",
        arrival_time=2.0,
        arrival_iteration_id=2,
        computed_tokens=100,
    )
    scheduler.running = [older]
    if scheduler._has_older_active_prefill_request(decode_candidate):
        fail("the request-level prefill guard must not block decode work")

    same_time_older = request(
        "same-time-older",
        arrival_time=2.0,
        arrival_iteration_id=1,
    )
    scheduler.running = [same_time_older]
    if not scheduler._has_older_active_prefill_request(candidate):
        fail("arrival iteration must break equal-timestamp FCFS ties")

    helper_source = POLICY_HELPER.read_text(encoding="utf-8")
    expected_mapping = (
        'VLLM_PD_REQUEST_FCFS="${VLLM_PD_EVAL_REQUEST_FCFS:-1}"'
    )
    if expected_mapping not in helper_source:
        fail("policy helper does not expose the FCFS selector override")

    print("[PASS] vendor_v3 request-level P/D FCFS contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
