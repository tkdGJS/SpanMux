#!/usr/bin/env python3
"""Import-free contracts for the optimized tiered context-ring data plane."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_residency_provider.py"
)
EXECUTOR = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_blockwise_attention.py"
)
BACKEND = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_blockwise_backend.py"
)
ATTENTION_BACKEND = ROOT / "vendor_v4/vllm/v1/attention/backend.py"
TRITON_BACKEND = ROOT / (
    "vendor_v4/vllm/v1/attention/backends/triton_attn.py"
)
LAYER_RELOAD_PREFETCH = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_layer_reload_prefetch.py"
)
SCHEDULER = ROOT / "vendor_v4/vllm/v1/core/sched/scheduler.py"
KV_CACHE_MANAGER = ROOT / "vendor_v4/vllm/v1/core/kv_cache_manager.py"
REQUEST = ROOT / "vendor_v4/vllm/v1/request.py"
BLOCK_POOL = ROOT / "vendor_v4/vllm/v1/core/block_pool.py"
SOURCE_BLOCK_LEASE = ROOT / (
    "vendor_v4/vllm/v1/core/tiered_hbm_source_block_lease.py"
)
MODEL_RUNNER = ROOT / "vendor_v4/vllm/v1/worker/gpu_model_runner.py"
UNIPROC_EXECUTOR = ROOT / "vendor_v4/vllm/v1/executor/uniproc_executor.py"
POLICY_HELPER = ROOT / "tools/pd_eval_policy_env.sh"
BATCHED_ATTENTION = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_triton_batched_attention.py"
)
TRANSFER_PRIORITY = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_transfer_priority.py"
)
FUSED_KV_SCATTER = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_triton_kv_scatter.py"
)
FUSED_ATTENTION = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_triton_fused_attention.py"
)


def require(path: Path, markers: tuple[str, ...]) -> None:
    source = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise AssertionError(f"{path}: missing markers {missing}")


def require_init_attribute(
    path: Path,
    class_name: str,
    attribute_name: str,
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        init = next(
            (
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            ),
            None,
        )
        if init is None:
            break
        for child in ast.walk(init):
            if not isinstance(child, ast.Attribute):
                continue
            if (
                isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and child.attr == attribute_name
            ):
                return
        break
    raise AssertionError(
        f"{path}: {class_name}.__init__ does not initialize "
        f"self.{attribute_name}"
    )


def require_call_keyword(
    path: Path,
    function_name: str,
    keyword_name: str,
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == function_name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == function_name
        )
    ]
    if not calls:
        raise AssertionError(f"{path}: no calls to {function_name}")
    missing_lines = [
        node.lineno
        for node in calls
        if not any(keyword.arg == keyword_name for keyword in node.keywords)
    ]
    if missing_lines:
        raise AssertionError(
            f"{path}: {function_name} calls missing {keyword_name}= at "
            f"lines {missing_lines}"
        )


def require_class_method(
    path: Path,
    class_name: str,
    method_name: str,
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        if any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == method_name
            for item in node.body
        ):
            return
        break
    raise AssertionError(
        f"{path}: {class_name} does not define {method_name}"
    )


def forbid_method_attributes(
    path: Path,
    class_name: str,
    method_name: str,
    attribute_names: tuple[str, ...],
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        method = next(
            (
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == method_name
            ),
            None,
        )
        if method is None:
            break
        accesses = [
            (child.attr, child.lineno)
            for child in ast.walk(method)
            if isinstance(child, ast.Attribute)
            and child.attr in attribute_names
        ]
        if accesses:
            raise AssertionError(
                f"{path}: {class_name}.{method_name} must use current "
                f"residency rather than historical tiered state; found "
                f"{accesses}"
            )
        return
    raise AssertionError(
        f"{path}: {class_name} does not define {method_name}"
    )


def require_type_checking_import(
    path: Path,
    module_name: str,
    imported_name: str,
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    runtime_import_lines: list[int] = []
    type_checking_import_found = False

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            if any(alias.name == imported_name for alias in node.names):
                runtime_import_lines.append(node.lineno)
            continue
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        for child in node.body:
            if not isinstance(child, ast.ImportFrom):
                continue
            if child.module != module_name:
                continue
            if any(alias.name == imported_name for alias in child.names):
                type_checking_import_found = True

    if runtime_import_lines:
        raise AssertionError(
            f"{path}: {imported_name} has runtime imports at "
            f"lines {runtime_import_lines}"
        )
    if not type_checking_import_found:
        raise AssertionError(
            f"{path}: {imported_name} is not imported under TYPE_CHECKING"
        )


def forbid_attribute_store(
    path: Path,
    object_name: str,
    attribute_names: tuple[str, ...],
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stores = [
        (node.attr, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == object_name
        and node.attr in attribute_names
    ]
    if stores:
        raise AssertionError(
            f"{path}: request-level tiered metadata must be immutable inside "
            f"attention forward; found stores {stores}"
        )


def forbid_named_attribute_access(
    path: Path,
    object_name: str,
    attribute_name: str,
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    accesses = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == object_name
        and node.attr == attribute_name
    ]
    if accesses:
        raise AssertionError(
            f"{path}: {object_name} is the coordinator tuple, not "
            f"KVCacheBlocks; invalid .{attribute_name} access at {accesses}"
        )


def main() -> int:
    forbid_method_attributes(
        SCHEDULER,
        "Scheduler",
        "_native_decode_control_fast_path_eligible",
        (
            "virtual_context_deferred_logical_computed_tokens",
            "virtual_context_deferred_hbm_released_tokens",
            "virtual_context_tiered_provider_deferred_tokens",
            "virtual_context_tiered_provider_virtual_external_tokens",
            "virtual_context_tiered_commit_count",
            "virtual_context_tiered_commit_released_pages",
            "virtual_context_tiered_commit_existing_dram_pages",
        ),
    )
    require_type_checking_import(
        ATTENTION_BACKEND,
        "vllm.v1.attention.ops.tiered_d2h_source_plan",
        "TieredD2HSourcePlan",
    )
    require_init_attribute(
        PROVIDER,
        "DRAMKVPageRegistry",
        "_registration_lock",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "reload_dram_kv_pages_to_hbm",
        "block_size",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "reload_dram_kv_pages_to_hbm",
        "block_table_cpu",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "reload_dram_kv_pages_to_hbm",
        "seq_lens_cpu",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "reload_dram_kv_pages_to_hbm",
        "query_start_loc_cpu",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "try_run_tiered_blockwise_backend",
        "page_versions_by_request",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "try_run_tiered_blockwise_backend",
        "reload_required_pages_by_request",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "_prepare_tiered_attention_page_tensor_plans",
        "page_versions_by_request",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "_prepare_tiered_attention_page_tensor_plans",
        "required_pages_by_request",
    )
    forbid_attribute_store(
        TRITON_BACKEND,
        "attn_metadata",
        (
            "tiered_page_versions",
            "tiered_reload_required_pages",
        ),
    )
    forbid_named_attribute_access(
        KV_CACHE_MANAGER,
        "new_blocks",
        "blocks",
    )
    require(
        KV_CACHE_MANAGER,
        (
            "len(new_blocks) == len(block_counts_after)",
            "for group in new_blocks",
        ),
    )
    require_call_keyword(
        LAYER_RELOAD_PREFETCH,
        "reload_dram_kv_pages_to_hbm",
        "block_size",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "register_hbm_kv_cache_pages",
        "allowed_request_ids",
    )
    require_call_keyword(
        TRITON_BACKEND,
        "register_hbm_kv_cache_pages",
        "force_all_request_ids",
    )
    require_class_method(
        UNIPROC_EXECUTOR,
        "UniProcExecutor",
        "_pd_pair_phase_dependency",
    )
    require(
        PROVIDER,
        (
            'state="COPYING_FROM_HBM"',
            'old_slot.state = "RETIRED"',
            "class TieredDRAMKVLease",
            "def acquire_slots(",
            "dram_page_pool_lease_acquire_count",
            "wait_for_pending_hbm_d2h_reads",
            "block_size: int",
            "dram_provider_hbm_d2h_fence_wait_count",
            "copy_complete.synchronize()",
            "self._pool.publish_reservation(reservation)",
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_BULK_TOKENS",
            "VLLM_VIRTUAL_CONTEXT_RELOAD_BULK_TOKENS",
            "VLLM_VIRTUAL_CONTEXT_RELOAD_STAGING_SCATTER",
            "VLLM_VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER",
            "fused_scatter_staged_kv_pages(",
            "def _coalesce_h2d_source_spans(",
            "contiguous_pool_span(",
            "dram_provider_reload_to_hbm_bulk_pages",
            "dram_provider_reload_to_hbm_staging_scatter_pages",
            "key_cache.index_copy_(",
            "dram_provider_reload_to_hbm_plan_us_total",
            "dram_provider_register_from_hbm_direct_bulk_pages",
            "class TieredDirectD2HRegistration",
            "def enqueue_from_hbm_direct(",
            "safe_to_reuse = not copy_enqueued",
            "dram_provider_register_from_hbm_direct_enqueue_count",
            "dram_provider_register_from_hbm_direct_completion_wait_us_total",
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_PRESSURE_ONLY",
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER",
            "VLLM_VIRTUAL_CONTEXT_RELOAD_BULK_TOKENS",
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_PLAN",
            "acquire_tiered_hbm_source_block_lease",
            "source_block_lease: TieredHBMSourceBlockLease | None",
            "_release_hbm_source_block_lease(source_block_lease)",
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_VIEW_DISPATCH",
            "class _DeferredHBMRegistrationPage",
            "def _materialize_hbm_registration_pages(",
            "dram_provider_d2h_source_view_dispatch_materialize_pages",
            "source_page_plan: TieredD2HSourcePlan | None = None",
            "dram_provider_d2h_source_plan_forward_count",
            "dram_provider_d2h_source_plan_fallback_scan_count",
            "def _d2h_control_dispatch_worker(",
            "def _submit_dispatched_hbm_registration(",
            "dram_provider_d2h_control_dispatch_queue_wait_us_total",
            "dram_provider_d2h_control_dispatch_control_us_total",
            "_publish_pending_hbm_d2h_fences(pages, completion_future)",
            "def has_pending_hbm_d2h_reads(",
            "allowed_request_ids: set[str] | None = None",
            "force_all_request_ids: set[str] | None = None",
        ),
    )
    require(
        SCHEDULER,
        (
            "resolve_virtual_context_coverage_block_size(",
            "virtual_context_tiered_transition_last_release_candidate_pages",
            "virtual_context_tiered_transition_last_missing_pages",
            '"tiered_provider_copy_not_ready"',
        ),
    )
    require(
        REQUEST,
        (
            "self.virtual_context_tiered_transition_attempt_count = 0",
            'self.virtual_context_tiered_transition_last_reason = "not_attempted"',
        ),
    )
    require(
        SOURCE_BLOCK_LEASE,
        (
            "class TieredHBMSourceBlockLease",
            "def acquire_tiered_hbm_source_block_lease(",
            "def get_tiered_hbm_source_leased_block_ids(",
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE",
        ),
    )
    require(
        BLOCK_POOL,
        (
            "_tiered_hbm_source_lease_deferred_blocks",
            "def _drain_source_lease_deferred_blocks(",
            "get_tiered_hbm_source_leased_block_ids()",
            "kv_retention_d2h_source_lease_deferred_free_blocks",
        ),
    )
    require(
        EXECUTOR,
        (
            "VLLM_VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT",
            "VLLM_VIRTUAL_CONTEXT_RING_PAGED_HBM_UPDATE",
            "TieredPagedHBMSegment",
            "update_triton_online_softmax_state_paged",
            "requires the paged HBM",
            "def ensure_reference_states()",
            "context_ring_reference_state_allocated",
            "every DRAM source group to use pinned",
            "context_ring_source_leases_acquired",
            "_defer_source_lease_releases",
            "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL",
            "materialize_group_batch",
            "run_batched_tiered_partial_attention",
            "merge_batched_tiered_partial_states",
            "context_ring_host_group_loop_bypassed",
            "def get_shared_hbm_staging_buffer(",
            "def _allocate_normal_tensor(",
            "with torch.inference_mode(False):",
            "shared HBM staging allocation unexpectedly produced an ",
            '"inference tensor"',
        ),
    )
    require(
        LAYER_RELOAD_PREFETCH,
        (
            "Inference mode is thread-local.",
            "with torch.inference_mode():",
        ),
    )
    require(
        TRANSFER_PRIORITY,
        (
            "def acquire_native_reload_priority(",
            "def acquire_context_ring_priority(",
            "tiered_transfer_priority_native_wait_us_total",
            "tiered_transfer_priority_context_wait_us_total",
        ),
    )
    require(
        BATCHED_ATTENTION,
        (
            "_batched_partial_attention_kernel",
            "_hierarchical_online_softmax_merge_kernel",
            "_ragged_batched_partial_attention_kernel",
            "_ragged_hierarchical_online_softmax_merge_kernel",
            "class BatchedTieredPartialState",
            "def run_batched_tiered_partial_attention(",
            "def merge_batched_tiered_partial_states(",
            "def run_ragged_batched_tiered_partial_attention(",
            "def merge_ragged_batched_tiered_partial_states(",
        ),
    )
    require(
        PROVIDER,
        (
            "class _TieredAttentionPageTensorCacheEntry",
            "def _cached_page_plan_tensors(",
            "attention_plan_tensor_cache_hit",
            "attention_plan_tensor_cache_delta_pages",
        ),
    )
    require(
        TRITON_BACKEND,
        (
            "tiered_attention_page_row_plans",
            "_prepare_tiered_attention_page_tensor_plans",
            "wait_for_pending_hbm_d2h_reads(",
            "tiered_dram_register_pressure_only_enabled",
            "tiered_pressure_snapshot_request_ids",
            "should_attempt_tiered_backend",
            "layer_page_versions",
            "layer_reload_required_pages",
        ),
    )
    require(
        BACKEND,
        (
            "tiered_blockwise_fully_optimized_strict",
            "TieredPagedHBMSegment",
            "attention_plan_paged_hbm_segment_count",
            "page_plans_by_row[req_idx]",
            "cannot consume dict-backed",
            "slot-addressable DRAM pages",
            "requires pre-acquired",
            "acquire_dram_kv_pool_slots",
            "source_leases=span_leases",
            'execution_mode=(',
            "request_rows_serialized_on_host",
            "request_plan_rows_prepared_on_host",
            "executor.run_ragged_batch(",
        ),
    )
    require(
        SCHEDULER,
        (
            "virtual_context_kv_page_size_bytes",
            "def _virtual_context_ring_dma_budget_bytes(",
            "def _native_decode_control_fast_path_eligible(",
            "return self._native_decode_runtime_fast_path_eligible(request)",
            "VLLM_VIRTUAL_CONTEXT_RING_MIN_PROGRESS_TOKENS",
            "below_min_progress_defer",
            "ring_chunk_below_min_progress",
            "context_ring_prefill_admission_min_progress_defer_count",
            "self._native_decode_reload_guard_fast_path_count += 1",
            "and not preempted_reqs",
            "virtual_context_native_decode_control_fast_path_used",
            "virtual_context_native_decode_reload_guard_fast_path_count",
            "virtual_context_debug_snapshot_stale",
            "VLLM_VIRTUAL_CONTEXT_DEBUG_SNAPSHOT_INTERVAL",
            "virtual_context_request_debug_snapshot_hits",
            "virtual_context_request_debug_duration_s",
            "prefill_attention_plan",
            "context_ring_bw_selected_dma_budget_bytes",
            "virtual_context_pressure_snapshot_request_ids",
        ),
    )
    require(
        MODEL_RUNNER,
        (
            '"dma_budget_bytes": as_int(',
            "tiered_dma_budget_bytes=int(",
            '"pressure_snapshot_request_ids": as_request_ids(',
            "build_tiered_d2h_source_plan(",
            "tiered_d2h_source_plan_enabled()",
            "cm.tiered_d2h_source_plan =",
            "native_hbm_decode_control_bypass",
        ),
    )
    require(
        POLICY_HELPER,
        (
            "pd_eval_policy_enable_fully_optimized_ring()",
            "muxwise_dynamic_virtual_ring_optimized)",
            "static_pd_virtual_ring_optimized)",
            "VLLM_VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT=1",
            "VLLM_VIRTUAL_CONTEXT_RING_MIN_PROGRESS_TOKENS",
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_PRESSURE_ONLY",
            "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER",
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_PLAN",
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE",
            "VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_VIEW_DISPATCH",
            "VLLM_VIRTUAL_CONTEXT_NATIVE_DECODE_CONTROL_FAST_PATH",
            "VLLM_VIRTUAL_CONTEXT_DELTA_HBM_OBSERVE",
            "VLLM_VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER",
            "VLLM_VIRTUAL_CONTEXT_DEBUG_SNAPSHOT_INTERVAL",
            "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL",
            "VLLM_VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION",
            "VLLM_PD_EVAL_RING_TRITON_BLOCK_M",
            "VLLM_PD_EVAL_RING_TRITON_BLOCK_N",
            "pd_eval_policy_clear_virtual_context",
            "VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL=0",
        ),
    )
    require(
        EXECUTOR,
        (
            "def context_ring_native_reduction_enabled()",
            "native_reduction=native_reduction_enabled",
            "context_ring_native_reduction_enabled",
        ),
    )
    require(
        FUSED_ATTENTION,
        (
            "native_reduction: tl.constexpr",
            "qk = tl.dot(q, k, acc=qk)",
            'input_precision="ieee"',
            "native_reduction: bool = False",
        ),
    )
    require(
        FUSED_KV_SCATTER,
        (
            "def _fused_staged_kv_scatter_kernel(",
            "def fused_scatter_staged_kv_pages(",
            "destination_blocks_ptr",
            "tl.store(key_cache_ptr",
            "tl.store(value_cache_ptr",
            "_DESCRIPTOR_CACHE",
            "ready_event.record",
            "wait_event(descriptor.ready_event)",
            "dram_provider_reload_fused_scatter_launches",
        ),
    )
    print("[OK] fully optimized context-ring source contracts validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
