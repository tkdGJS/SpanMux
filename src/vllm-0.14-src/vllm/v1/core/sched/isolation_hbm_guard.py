# SPDX-License-Identifier: Apache-2.0
"""Conservative opt-in no-recompute admission for HBM-only controls.

Mirrored in source/vendor; no torch, physical reservations, copies or frees.
"""
import os


class IsolationHBMCompletionGuard:
    def __init__(self, scheduler, log):
        self.scheduler = scheduler
        self.log = log
        self.last_blocked = None
        self.blocked_checks = 0
        self.blocked_transitions = 0
        groups = scheduler.kv_cache_config.kv_cache_groups
        parallel = scheduler.parallel_config
        if (len(groups) != 1
                or type(groups[0].kv_cache_spec).__name__ != "FullAttentionSpec"
                or any(getattr(parallel, k, 1) != 1 for k in (
                    "tensor_parallel_size", "pipeline_parallel_size",
                    "decode_context_parallel_size", "prefill_context_parallel_size"))
                or scheduler.vllm_config.speculative_config is not None
                or scheduler.connector is not None
                or scheduler.is_encoder_decoder
                or getattr(scheduler, "virtual_context_enabled", False)):
            raise ValueError("HBM completion guard requires single-GPU, one FullAttentionSpec "
                             "group, no speculation/connector/virtual context/encoder")
        self.block_size = int(groups[0].kv_cache_spec.block_size)
        self.log("isolation HBM completion guard enabled: declared prompt+output "
                 "growth protected; no physical reservation")

    def _blocks(self, request):
        if request.has_encoder_inputs or request.sampling_params is None:
            raise ValueError("HBM completion guard supports text generation only")
        tokens = min(request.num_prompt_tokens + request.max_tokens,
                     self.scheduler.max_model_len)
        return (tokens + self.block_size - 1) // self.block_size

    def can_admit(self, candidate):
        scheduler = self.scheduler
        manager = scheduler.kv_cache_manager
        pool = manager.block_pool
        capacity = pool.num_gpu_blocks - 1  # The null block is never allocatable.
        candidate_blocks = self._blocks(candidate)
        if candidate_blocks > capacity:
            raise ValueError("HBM completion guard: request "
                             f"{candidate.request_id} needs {candidate_blocks} blocks "
                             f"but an empty pool has {capacity}; reduce request length "
                             "or increase KV capacity (waiting cannot fix this)")
        tables = manager.coordinator.single_type_managers[0].req_to_blocks
        # Include WAITING and Future-owned physical tables. Counting only
        # scheduler.running silently drops those growth obligations.
        required = max(0, candidate_blocks - len(tables.get(candidate.request_id, ())))
        owners = []
        for request_id, blocks in tables.items():
            if not blocks or request_id == candidate.request_id:
                continue
            request = scheduler.requests.get(request_id)
            if request is not None:
                required += max(0, self._blocks(request) - len(blocks))
                owners.append(request_id)
        free = pool.get_num_free_blocks()
        self.last_required_blocks = required
        self.last_free_blocks = free
        if required <= free:
            self.last_blocked = None
            return True
        self.blocked_checks += 1
        # Growth consumes free space and the same amount of future budget.
        # Log only changes in obligation, not each Decode token/iteration.
        signature = (candidate.request_id, tuple(sorted(owners)), required - free)
        if signature != self.last_blocked:
            self.blocked_transitions += 1
            self.log("isolation HBM admission wait: request=%s required_blocks=%d "
                     "free_blocks=%d owners=%d", candidate.request_id, required, free, len(owners))
            self.last_blocked = signature
        return False


def make_isolation_hbm_completion_guard(scheduler, log):
    value = os.getenv("VLLM_ISOLATION_HBM_COMPLETION_GUARD", "0")
    if value not in {"0", "1"}:
        raise ValueError("VLLM_ISOLATION_HBM_COMPLETION_GUARD must be 0 or 1")
    return IsolationHBMCompletionGuard(scheduler, log) if value == "1" else None
