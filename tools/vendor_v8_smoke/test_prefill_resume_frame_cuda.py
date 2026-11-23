#!/usr/bin/env python3
"""Opt-in CUDA storage/re-entry acceptance, not a complete vLLM server test.

Uses real CUDA tensors and the actual Llama layer-loop methods with synthetic
layers, overwriting test-owned source buffers before cross-stream re-entry.
No model download or live server mutation. Full-model logits/KV tests remain
separate acceptance requirements.

PYTHONPATH=vendor_v8 VLLM_KV_RETENTION_RUN_CUDA_TESTS=1 python <this-file>
"""
from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace as NS


@unittest.skipUnless(os.environ.get("VLLM_KV_RETENTION_RUN_CUDA_TESTS") == "1",
                     "opt-in CUDA frame acceptance")
class PrefillFrameCUDA(unittest.TestCase):
    parked = False
    def test_two_rows_overwrite_then_cross_stream_resume(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import test_exact_phase_cleanup as C
        import test_prefill_work_binding as W
        from vllm.model_executor.layers.kv_retention_resume_frame import (
            KVRetentionLayerContinuation, KVRetentionModelState, KVRetentionResumeFrame,
        )
        from vllm.model_executor.layers.kv_retention_parked_prefill import (
            kv_retention_bound_prefill_submission,
        )

        model_type = C.adapters(
            "vendor_v8/vllm/model_executor/models/llama.py", "LlamaModel",
            {"_kv_retention_advance_prefill", "_forward_decoder_layer_range"},
            {"time": time, "get_pp_group": lambda: NS(is_last_rank=True),
             "kv_retention_bound_prefill_submission": kv_retention_bound_prefill_submission})
        source, target = torch.cuda.Stream(), torch.cuda.Stream()
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                out = W.output()
                out.num_scheduled_tokens = {"A": 2, "B": 2}
                out.total_num_scheduled_tokens = 4
                registry = W.REGISTRY(1 if self.parked else 2,
                                      parked_capacity=int(self.parked))
                first = registry.start(out, W.epochs(out))
                execution = C.M.TTFTLayerwisePrefillExecution(
                    requested=True, active=True, phase="prefill", group_layers=2,
                    preemption_requested=True, preemption_active=True,
                    request_ids=("A", "B"), scheduled_tokens=(("A", 2), ("B", 2)))
                execution.consume_yield_request = lambda: execution.last_layer_exclusive == 2
                model = model_type()
                model.start_layer, model.end_layer = 0, 6
                model.aux_hidden_state_layers = []
                model.pd_layerwise_profile_handoff_enabled = False
                model.kv_retention_p1_parked_enabled = self.parked
                def layer(positions, hidden, residual):
                    residual = hidden.clone() if residual is None else residual
                    residual.add_(positions[:, None].to(dtype) * 0.01)
                    return hidden * 1.1 + residual, residual
                model.layers = [NS(forward_with_pd_prefill_workspace=layer) for _ in range(6)]
                model.norm = lambda h, r: (h + r, None)
                with torch.cuda.stream(source):
                    positions = torch.tensor([8, 9, 16, 17], device="cuda")
                    hidden = torch.arange(32, device="cuda", dtype=dtype).reshape(4, 8)
                    residual = torch.zeros_like(hidden)
                    expected_h, expected_r = hidden.clone(), residual.clone()
                    for _ in range(6):
                        expected_h, expected_r = layer(positions, expected_h, expected_r)
                    expected = expected_h + expected_r
                    state = KVRetentionModelState(0, 6, positions, hidden, residual, [], True)
                    cont = KVRetentionLayerContinuation(model, execution, state)
                    boundary = next(cont)
                    self.assertEqual(boundary.next_layer, 2)
                    metadata = {"block_table": torch.tensor([[1, 2], [3, 4]], device="cuda"),
                                "positions": positions}
                    def complete():
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream())
                        event.synchronize()
                    complete()
                    frame = KVRetentionResumeFrame.capture(
                        binding=first, model_state=state, attn_metadata=metadata,
                        logits_indices=torch.tensor([1, 3], device="cuda"),
                        num_scheduled_tokens_np=(2, 2), max_gpu_bytes=1024 * 1024,
                        gpu_boundary_complete=True, copy_complete=complete)
                    cont.export(frame)
                    positions.fill_(-999)
                    state.hidden_states.fill_(-999)
                    state.residual.fill_(-999)
                    metadata["block_table"].fill_(0)
                    complete()
                registry.suspend(first, 2)
                if self.parked:
                    from vllm.model_executor.layers.kv_retention_parked_prefill import (
                        KVRetentionParkedStore, KVRetentionParkedTicket,
                    )
                    boundary.kv_retention_binding = first
                    ticket = KVRetentionParkedTicket(first, 2, frame.gpu_storage_bytes, ("A", "B"))
                    boundary.kv_retention_parked_ticket = ticket
                    saved = NS(scheduler_output=out, boundary=boundary,
                               continuation=cont, kv_offload=None,
                               kv_sampling_restore_pending=False)
                    store = KVRetentionParkedStore(1 << 20)
                    store.put(ticket, saved)
                    registry.park(first)
                    other = W.output(ids=("C",))
                    cb = registry.start(other, W.epochs(other))
                    registry.sampling(cb)
                    registry.finish(cb)
                second = registry.resume(out, W.epochs(out))
                if self.parked:
                    runner, ctx = W.RUNNER(), W.context()
                    runner._kv_retention_parked_store = store
                    ctx.kv_retention_last_slot_generation = cb.slot_generation
                    runner._claim_pd_execute_state("prefill", out, ctx, second)
                    self.assertIs(ctx.ttft_prefill_continuation, saved)
                    self.assertEqual(store.entries, {})
                    self.assertEqual(second.slot_generation, 3)
                execution.note_resume()
                with torch.cuda.stream(target):
                    cont.import_frame(frame, second, record_stream=lambda t: t.record_stream(target))
                    copied_metadata = frame.payload["attn_metadata"]
                    frame.close()
                    with self.assertRaises(StopIteration) as done:
                        next(cont)
                    actual = done.exception.value
                target.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(copied_metadata["block_table"].tolist(), [[1, 2], [3, 4]])
                self.assertTrue(cont.closed)
                registry.sampling(second)
                registry.finish(second)
                self.assertEqual(len(registry), 0)


class P1FrameCUDA(PrefillFrameCUDA):
    """Same real tensor overwrite, now through P1 parked store/claim adapters."""
    parked = True

    def test_actual_input_batch_sampling_metadata_restore(self):
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from vllm.sampling_params import SamplingParams
        from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
        from vllm.model_executor.layers.kv_retention_parked_prefill import KVRetentionSamplingFrame

        batch = InputBatch(max_num_reqs=2, max_model_len=32,
            max_num_batched_tokens=8, device=torch.device("cuda"),
            pin_memory=True, vocab_size=64, block_sizes=[4], kernel_block_sizes=[4])
        rows = [CachedRequestState(req_id=name, prompt_token_ids=[1, 2, 3, 4],
            mm_features=[], sampling_params=SamplingParams(temperature=0.7, seed=i),
            generator=torch.Generator(device="cuda").manual_seed(i),
            block_ids=([i + 1, i + 2],), num_computed_tokens=2, output_token_ids=[5])
            for i, name in enumerate(("A", "B", "C"))]
        for row in rows[:2]:
            batch.add_request(row)
        batch.refresh_metadata()
        history = object()
        batch.in_progress_prompt_logprobs_cpu["A"] = history
        runner = NS(input_batch=batch, requests={r.req_id: r for r in rows})
        for name in ("input_ids", "positions", "query_start_loc", "seq_lens", "discard_request_mask"):
            b = NS(np=np.arange(8), gpu=torch.empty(8, dtype=torch.int64, device="cuda"))
            b.copy_to_gpu = lambda n, b=b: b.gpu[:n].copy_(torch.from_numpy(b.np[:n]))
            setattr(runner, name, b)
        frame = KVRetentionSamplingFrame.capture(runner, NS(num_reqs=2,
            num_scheduled_tokens=4, scheduler_output=NS(num_scheduled_tokens={"A": 2, "B": 2})))
        for name in ("A", "B"):
            batch.remove_request(name)
        batch.condense()
        batch.refresh_metadata()
        batch.add_request(rows[2])
        batch.refresh_metadata()
        for name in frame.buffers:
            getattr(runner, name).np.fill(999)
        frame.restore(runner)
        torch.cuda.synchronize()
        self.assertEqual(batch.req_ids, ["A", "B"])
        self.assertIs(batch.sampling_metadata.generators[0], rows[0].generator)
        self.assertIs(batch.sampling_metadata.generators[1], rows[1].generator)
        self.assertIs(batch.in_progress_prompt_logprobs_cpu["A"], history)
        self.assertEqual(batch.token_ids_cpu[0, :5].tolist(), [1, 2, 3, 4, 5])
        for name, values in frame.buffers.items():
            self.assertEqual(getattr(runner, name).gpu[:len(values)].tolist(), values.tolist())


if __name__ == "__main__":
    unittest.main()
