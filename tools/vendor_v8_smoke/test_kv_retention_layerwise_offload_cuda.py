#!/usr/bin/env python3
"""Opt-in CUDA numerical test, including destructive *test-cache* reuse.

PYTHONPATH=vendor_v8 VLLM_KV_RETENTION_RUN_CUDA_TESTS=1 python <this-file>
No live server is used. Only synthetic tensors allocated by this test change.
"""

import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("VLLM_KV_RETENTION_RUN_CUDA_TESTS") == "1",
                     "opt-in CUDA test; requires vendor extension and CUDA GPU")
class LayerwiseOffloadCUDA(unittest.TestCase):
    def test_snapshot_full_pages_and_exact_tails(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from vllm.model_executor.layers.ttft_layerwise_prefill import (
            KVRetentionCheckpointRow, TTFTLayerwisePrefillBoundary,
            build_kv_retention_layerwise_checkpoint,
        )
        from vllm.v1.attention.ops.kv_retention_layerwise import KVRetentionRingContinuation

        names = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
        impl = type("TritonAttentionImpl", (), {"scale": 64 ** -0.5})()
        with patch.dict(os.environ, {
            "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL": "0",
            "VLLM_VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS": "1",
            "VLLM_VIRTUAL_CONTEXT_RING_COMPUTE_GROUP_TOKENS": "16",
        }), torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                for start, chunk in ((0, 3), (3, 5), (16, 16), (19, 5), (32, 16), (37, 16)):
                    with self.subTest(dtype=dtype, start=start, chunk=chunk):
                        blocks = (7, 2, 8, 1)[:(start + chunk + 15) // 16]
                        row = KVRetentionCheckpointRow("a", 0, start, chunk, 16, blocks)
                        boundary = TTFTLayerwisePrefillBoundary(
                            scheduler_iteration_id=1, phase_submit_seq=1, phase_slot=0,
                            request_ids=("a",), scheduled_tokens=(("a", chunk),),
                            profile_id="test", profile_epoch=1, prefill_mode="native",
                            next_layer=1, end_layer=2, model_yield_count=1,
                        )
                        checkpoint = build_kv_retention_layerwise_checkpoint(
                            boundary=boundary, first_layer=0, layer_names=names,
                            rows=(row,), gpu_boundary_complete=True,
                        )
                        caches = [torch.randn((10, 2, 16, 2, 64), dtype=dtype, device="cuda")
                                  for _ in names]
                        expected = {}
                        for index, (name, cache) in enumerate(zip(names, caches)):
                            valid = start + (chunk if index == 0 else 0)
                            for page, block in enumerate(blocks):
                                count = max(0, min(16, valid - page * 16))
                                cache[block, :, count:].fill_(float("nan"))
                            if valid:
                                expected[name] = tuple(torch.cat([
                                    cache[b, kv] for b in blocks
                                ])[:valid].cpu() for kv in (0, 1))
                        torch.cuda.current_stream().synchronize()
                        layers = {name: NS(impl=impl, kv_cache=[cache])
                                  for name, cache in zip(names, caches)}
                        owner = KVRetentionRingContinuation.capture(checkpoint, layers)
                        self.assertIsNotNone(owner)
                        try:
                            for name, pair in expected.items():
                                for actual, reference in zip(owner.buffers[name, "a"], pair):
                                    torch.testing.assert_close(
                                        actual[:reference.shape[0]], reference, rtol=0, atol=0)
                                    self.assertTrue(actual.is_pinned())
                                    self.assertFalse(actual.is_inference())
                            self.assertEqual(owner.ticket.copied_bytes,
                                             (2 * start + chunk) * 2 * 2 * 64 * 2)
                            self.assertEqual(owner.ticket.state, "copy_ready")
                        finally:
                            owner.close()

    def test_two_rows_partial_tail_then_reuse_old_hbm(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        from vllm.model_executor.layers.ttft_layerwise_prefill import (
            KVRetentionCheckpointRow, TTFTLayerwisePrefillBoundary,
            build_kv_retention_layerwise_checkpoint,
        )
        from vllm.v1.attention.ops.kv_retention_layerwise import KVRetentionRingContinuation

        torch.manual_seed(31)
        names = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
        rows = (
            KVRetentionCheckpointRow("a", 0, 19, 5, 16, (1, 2)),
            KVRetentionCheckpointRow("b", 1, 0, 3, 16, (3,)),
        )
        boundary = TTFTLayerwisePrefillBoundary(
            scheduler_iteration_id=1, phase_submit_seq=1, phase_slot=0,
            request_ids=("a", "b"), scheduled_tokens=(("a", 5), ("b", 3)),
            profile_id="test", profile_epoch=1, prefill_mode="native",
            next_layer=1, end_layer=2, model_yield_count=1,
        )
        checkpoint = build_kv_retention_layerwise_checkpoint(
            boundary=boundary, first_layer=0, layer_names=names, rows=rows,
            gpu_boundary_complete=True,
        )
        impl = type("TritonAttentionImpl", (), {"scale": 64 ** -0.5})()
        caches = [torch.randn((5, 2, 16, 2, 64), dtype=torch.float16, device="cuda")
                  for _ in names]
        history_k = torch.cat([caches[1][1, 0], caches[1][2, 0, :3]]).clone()
        history_v = torch.cat([caches[1][1, 1], caches[1][2, 1, :3]]).clone()
        q = torch.randn((8, 4, 64), dtype=torch.float16, device="cuda")
        k = torch.randn((8, 2, 64), dtype=torch.float16, device="cuda")
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        layers = {name: NS(impl=impl, kv_cache=[cache]) for name, cache in zip(names, caches)}
        # capture's caller guarantees an existing completed layer event.
        torch.cuda.current_stream().synchronize()
        with patch.dict(os.environ, {
            "VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL": "0",
            "VLLM_VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS": "1",
            "VLLM_VIRTUAL_CONTEXT_RING_COMPUTE_GROUP_TOKENS": "16",
            "VLLM_VIRTUAL_CONTEXT_RING_ASYNC_PREFETCH": "1",
            "VLLM_VIRTUAL_CONTEXT_RING_PREFETCH_DISTANCE": "1",
            "VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE": "1",
        }):
            owner = KVRetentionRingContinuation.capture(checkpoint, layers)
            self.assertIsNotNone(owner)
            try:
                # Check every valid row/layer prefix, including completed-layer
                # tails, before simulating reuse of the old physical blocks.
                for index, name in enumerate(names):
                    for row in rows:
                        valid = row.start_token + (row.scheduled_tokens if index == 0 else 0)
                        if not valid:
                            continue
                        for kv, saved in enumerate(owner.buffers[name, row.request_id]):
                            expected_prefix = torch.cat([
                                caches[index][block, kv] for block in row.block_ids
                            ])[:valid].cpu()
                            torch.testing.assert_close(saved[:valid], expected_prefix, rtol=0, atol=0)
                self.assertEqual(owner.ticket.copied_bytes, (24 + 3 + 19) * 2 * 2 * 64 * 2)
                owner.ticket.commit_release(3)
                # Simulate allocation to unrelated requests after release.
                for cache in caches:
                    cache.fill_(float("nan"))
                owner.attention(names[1], impl, q, k, v, out,
                                NS(num_actual_tokens=8, request_ids=["a", "b"]))
                owner.copy_event.synchronize()
                expected = []
                for lo, hi, start, hk, hv in (
                    (0, 5, 19, history_k, history_v),
                    (5, 8, 0, k[:0], v[:0]),
                ):
                    keys = torch.cat([hk, k[lo:hi]]).float().repeat_interleave(2, dim=1)
                    vals = torch.cat([hv, v[lo:hi]]).float().repeat_interleave(2, dim=1)
                    scores = torch.einsum("qhd,khd->hqk", q[lo:hi].float(), keys) * impl.scale
                    qp = torch.arange(start, start + hi - lo, device="cuda")
                    kp = torch.arange(keys.shape[0], device="cuda")
                    scores.masked_fill_(kp[None, None, :] > qp[None, :, None], float("-inf"))
                    expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals))
                torch.testing.assert_close(out.float(), torch.cat(expected), rtol=5e-3, atol=5e-3)
                torch.testing.assert_close(owner.buffers[names[1], "a"][0][19:], k[:5].cpu())
                self.assertEqual(owner.completed_layers, 2)
            finally:
                owner.close()


if __name__ == "__main__":
    unittest.main()
