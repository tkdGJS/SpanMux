#!/usr/bin/env python3
"""Opt-in real CUDA layout/order/numerical checks, independent of model weights."""
import os
import unittest


@unittest.skipUnless(os.getenv("VLLM_ISOLATION_RUN_CUDA_TESTS") == "1",
                     "requires CUDA, Torch and vendor Triton kernels")
class WorkspaceSharingCUDA(unittest.TestCase):
    def test_borrowed_ring_matches_private_fp16_bf16_and_survives_overwrite(self):
        import torch
        from unittest.mock import patch
        from vllm.model_executor.layers.pd_prefill_workspace import (
            PrefillWorkspaceManager, LlamaPrefillWorkspaceLayout,
        )
        from vllm.v1.attention.ops.isolation_ring_state import IsolationRingStateArena
        from vllm.v1.attention.ops import tiered_triton_fused_attention as f

        torch.manual_seed(17)
        device = torch.device("cuda", torch.cuda.current_device())
        with patch.dict(os.environ, {"VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "1"}):
            for dtype in (torch.float16, torch.bfloat16):
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    continue
                manager = PrefillWorkspaceManager(enabled=True, strict=True,
                    max_tokens=128, slot_count=1, rightsize=True, sharing=True)
                stream = torch.cuda.current_stream()
                x = manager.acquire(profile_id="test", profile_epoch=0,
                                    num_tokens=37, stream=stream)
                manager.configure_llama(layout=LlamaPrefillWorkspaceLayout(
                    384, 256, 2048, 1024, 256), dtype=dtype, device=device)
                qkv = manager.buffer(x, "qkv", num_tokens=37, width=384,
                                     dtype=dtype, device=device)
                qkv.normal_()
                q = qkv[:, :256].view(37, 4, 64)
                k = torch.randn(97, 2, 64, dtype=dtype, device="cuda")
                v = torch.randn_like(k)
                qkv_ref = qkv.clone()
                manager.begin_attention(x, stream)
                shared = IsolationRingStateArena(torch, f.TritonOnlineSoftmaxState,
                    lambda query: manager.borrow_ring_state(x, query, stream))
                private = IsolationRingStateArena(torch, f.TritonOnlineSoftmaxState)
                results = []
                for arena in (private, shared):
                    state, _ = arena.acquire(q)
                    f.create_triton_online_softmax_state(q, state)
                    for start, stop in ((0, 32), (32, 64), (64, 97)):
                        f.update_triton_online_softmax_state(
                            q, k[start:stop], v[start:stop], state,
                            query_positions=torch.arange(97, 134, device="cuda"),
                            key_start=start, causal=True,
                            softmax_scale=1 / 8, native_reduction=True)
                    results.append(f.finalize_triton_online_softmax_state(q, state))
                    arena.release()
                manager.end_attention(x, stream)
                torch.testing.assert_close(qkv, qkv_ref, atol=0, rtol=0)
                # Overwrite shared state as the real o_proj + MLP would do.
                manager.buffer(x, "gate_up", num_tokens=37, width=2048,
                               dtype=dtype, device=device).fill_(99)
                torch.testing.assert_close(results[0][0], results[1][0])
                torch.testing.assert_close(results[0][1], results[1][1])
                self.assertIsNone(shared.storage)
                self.assertIsNone(shared.borrow)
                event = torch.cuda.Event()
                event.record(stream)
                manager.finish(x, completion_event=event)

    def test_growth_and_reuse_across_streams(self):
        import torch
        from unittest.mock import patch
        from vllm.model_executor.layers.pd_prefill_workspace import (
            PrefillWorkspaceManager, LlamaPrefillWorkspaceLayout,
        )
        with patch.dict(os.environ, {"VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "1"}):
            m = PrefillWorkspaceManager(enabled=True, strict=True, max_tokens=16384,
                                        slot_count=1, rightsize=True)
            layout = LlamaPrefillWorkspaceLayout(384, 256, 2048, 1024, 256)
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            copies = []
            for i, tokens in enumerate((16, 2048, 4096, 16384, 7)):
                with torch.cuda.stream(streams[i % 2]):
                    x = m.acquire(profile_id=str(i), profile_epoch=i, num_tokens=tokens,
                                  stream=torch.cuda.current_stream())
                    m.configure_llama(layout=layout, dtype=torch.float16, device="cuda")
                    qkv = m.buffer(x, "qkv", num_tokens=tokens, width=384,
                                   dtype=torch.float16, device="cuda")
                    qkv.fill_(i + 1)
                    copies.append(qkv.clone())
                    event = torch.cuda.Event()
                    event.record()
                    m.finish(x, completion_event=event)
            torch.cuda.synchronize()
            for i, copy in enumerate(copies):
                self.assertTrue(bool(torch.all(copy == i + 1)))
            self.assertEqual(m._slots[0].capacity_tokens, 16384)


if __name__ == "__main__":
    unittest.main()
