#!/usr/bin/env python3
"""Opt-in CUDA numerical verification; run with PYTHONPATH=vendor_v8."""
import os
import unittest


@unittest.skipUnless(os.getenv("VLLM_ISOLATION_RUN_CUDA_TESTS") == "1",
                     "requires CUDA and compiled vendor ops")
class IsolationMemoryCUDA(unittest.TestCase):
    def test_expanded_ring_query_2k_to_16k(self):
        import torch
        from unittest.mock import patch
        from vllm.v1.attention.ops import tiered_triton_fused_attention as f
        from vllm.v1.attention.ops.isolation_ring_state import IsolationRingStateArena
        from vllm.isolation_memory import isolation_ring_query_bytes
        torch.manual_seed(20261005)
        arena = IsolationRingStateArena(torch, f.TritonOnlineSoftmaxState)
        k = torch.randn(97, 2, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        for tokens in (2048, 8192, 16384, 2048):
            q = torch.randn(tokens, 4, 64, device="cuda", dtype=torch.float16)
            positions = torch.arange(97, 97 + tokens, device="cuda")
            score = torch.einsum("qhd,khd->hqk", q.float(), k.repeat_interleave(2, dim=1).float()) / 8
            ref = torch.einsum("hqk,khd->qhd", score.softmax(-1), v.repeat_interleave(2, dim=1).float())
            with patch.object(f, "_ISOLATION_DYNAMIC_KERNEL", True):
                state, _ = arena.acquire(q)
                try:
                    f.create_triton_online_softmax_state(q, state)
                    for begin, end in ((0, 32), (32, 64), (64, 97)):
                        f.update_triton_online_softmax_state(
                            q, k[begin:end], v[begin:end], state, query_positions=positions,
                            key_start=begin, causal=True, softmax_scale=1 / 8,
                            native_reduction=True)
                    output, _ = f.finalize_triton_online_softmax_state(q, state)
                finally:
                    arena.release()
            self.assertLessEqual(arena.bytes, isolation_ring_query_bytes(16384, 4, 64, 2))
            torch.testing.assert_close(output.float(), ref, atol=6e-3, rtol=6e-3)

    def test_prefill_inplace_rope_strided_qkv_and_untouched_values(self):
        import torch
        from types import SimpleNamespace
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
        from itertools import product
        torch.manual_seed(20261005)
        for dtype, tokens, neox, rotary_dim in product(
            (torch.float16, torch.bfloat16), (7, 2048, 16384), (True, False), (32, 64)
        ):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            head, qwidth, kvwidth = 64, 2048, 512
            packed = torch.randn(tokens, qwidth + 2 * kvwidth, device="cuda", dtype=dtype)
            q, k, v = packed.split((qwidth, kvwidth, kvwidth), dim=-1)
            v_ref = v.clone()
            positions = torch.arange(49999 - tokens, 49999, device="cuda")
            inv = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device="cuda").float() / rotary_dim))
            angles = torch.arange(50000, device="cuda").float()[:, None] * inv[None, :]
            cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
            qref, kref = RotaryEmbedding.forward_static(positions, q, k, head, rotary_dim, cache, neox)
            rope = SimpleNamespace(use_flashinfer=False, head_size=head, cos_sin_cache=cache,
                                   is_neox_style=neox, _match_cos_sin_cache_dtype=lambda _: None)
            qout, kout = RotaryEmbedding.forward_cuda(rope, positions, q, k)
            tol = 3e-2 if dtype == torch.bfloat16 else 4e-3
            torch.testing.assert_close(qout, qref, atol=tol, rtol=tol)
            torch.testing.assert_close(kout, kref, atol=tol, rtol=tol)
            torch.testing.assert_close(v, v_ref, atol=0, rtol=0)
            self.assertEqual(qout.data_ptr(), q.data_ptr())
            self.assertEqual(kout.data_ptr(), k.data_ptr())

    def test_ring_runtime_lengths_paged_tails_and_state_reuse(self):
        import torch
        torch.manual_seed(20261004)
        from vllm.v1.attention.ops import tiered_triton_fused_attention as f
        from vllm.v1.attention.ops.isolation_ring_state import IsolationRingStateArena
        from unittest.mock import patch
        from itertools import product
        arena = IsolationRingStateArena(torch, f.TritonOnlineSoftmaxState)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for dynamic, native_reduction in product((False, True), repeat=2):
            for i, tokens in enumerate((7, 19, 3)):
                q = torch.randn(tokens, 4, 32, device="cuda", dtype=torch.float16)
                k = torch.randn(32, 2, 32, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                positions = torch.arange(28, 28 + tokens, device="cuda")
                ids = torch.arange(2, device="cuda", dtype=torch.int32)
                scores = torch.einsum(
                    "qhd,khd->hqk", q.float(),
                    k[:29].repeat_interleave(2, dim=1).float()
                ) * (32 ** -0.5)
                mask = (11 + torch.arange(29, device="cuda"))[None, :] <= positions[:, None]
                scores.masked_fill_(~mask[None, :, :], -float("inf"))
                ref = torch.einsum(
                    "hqk,khd->qhd", scores.softmax(-1),
                    v[:29].repeat_interleave(2, dim=1).float()
                )
                for paged in (False, True):
                    stream = streams[i % 2]
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream), patch.object(
                        f, "_ISOLATION_DYNAMIC_KERNEL", dynamic
                    ):
                        state, _ = arena.acquire(q)
                        try:
                            f.create_triton_online_softmax_state(q, state)
                            opts = dict(query_positions=positions, key_start=11,
                                        causal=True, softmax_scale=32 ** -0.5,
                                        native_reduction=native_reduction)
                            if paged:
                                f.update_triton_online_softmax_state_paged(
                                    q, k.view(2, 16, 2, 32), v.view(2, 16, 2, 32),
                                    ids, state, key_tokens=29, **opts)
                            else:
                                f.update_triton_online_softmax_state(
                                    q, k[:29], v[:29], state, **opts)
                            out, _ = f.finalize_triton_online_softmax_state(q, state)
                        finally:
                            arena.release()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.testing.assert_close(out.float(), ref, atol=6e-3, rtol=6e-3)

    def test_fused_norm_scratch_and_residual(self):
        import torch
        torch.manual_seed(20261004)
        from vllm.model_executor.layers.layernorm import RMSNorm
        # Avoid constructing CustomOp/config: exercise the actual scratch method.
        from types import SimpleNamespace
        for tokens in (1, 721, 2048, 16384):
            x = torch.randn(tokens, 2048, device="cuda", dtype=torch.float16)
            residual = torch.randn_like(x)
            weight = torch.ones(2048, device="cuda", dtype=x.dtype)
            norm = SimpleNamespace(hidden_size=2048, variance_epsilon=1e-6,
                                   variance_size_override=None, weight=weight)
            out = torch.empty_like(x)
            reference = RMSNorm.forward_static(x, 1e-6, 2048, x.dtype, weight)
            RMSNorm.forward_into(norm, x, out)
            torch.testing.assert_close(out, reference, atol=4e-3, rtol=4e-3)
            ref_out, ref_residual = RMSNorm.forward_static(
                x, 1e-6, 2048, x.dtype, weight, residual.clone()
            )
            RMSNorm.forward_into(norm, x, out, residual)
            torch.testing.assert_close(out, ref_out, atol=4e-3, rtol=4e-3)
            torch.testing.assert_close(residual, ref_residual, atol=4e-3, rtol=4e-3)
            with self.assertRaises(ValueError):
                RMSNorm.forward_into(norm, x, residual, residual)
            backing = torch.empty(tokens * 2048 + 1, device="cuda", dtype=x.dtype)
            with self.assertRaises(ValueError):
                RMSNorm.forward_into(norm, backing[:-1].view(tokens, 2048),
                                     backing[1:].view(tokens, 2048))


if __name__ == "__main__":
    unittest.main()
