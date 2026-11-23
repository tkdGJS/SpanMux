#!/usr/bin/env python3
"""Opt-in CUDA allocator/graph lifetime check; no custom-op rebuild required."""

import json
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
Audit = runpy.run_path(str(ROOT / "vendor_v8/vllm/isolation_startup_memory.py"))[
    "StartupMemoryAudit"]


@unittest.skipUnless(os.getenv("VLLM_ISOLATION_RUN_CUDA_TESTS") == "1",
                     "requires explicit CUDA test opt-in")
class StartupMemoryCUDA(unittest.TestCase):
    def test_live_tensor_survives_completed_nondefault_stream_then_graph_replay(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        records = []
        device = torch.device("cuda", torch.cuda.current_device())
        with patch.dict(os.environ, {"VLLM_ISOLATION_STARTUP_CACHE_TRIM": "1"}):
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                live = torch.ones(1024, device=device)
                live.mul_(3)
                temporary = torch.ones(8 * 1024 * 1024, device=device)
                temporary.mul_(2)
                del temporary
            ptr = live.data_ptr()
            owner = Audit(cuda=torch.cuda, device=device,
                          log=lambda fmt, row: records.append(json.loads(row)))
            owner.before_capture()
            self.assertEqual(live.data_ptr(), ptr)
            torch.testing.assert_close(live, torch.full_like(live, 3))
            after = next(r for r in records if r["stage"] == "after_cache_trim")
            self.assertLessEqual(after["reserved_bytes"], after["reserved_before_trim_bytes"])
            # Real capture AFTER trim must keep stable pointers and replay.
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = live * 2
            stream.synchronize()
            owner.after_kernel_warmup()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = live * 2
            owner.after_capture()
            owner.finish()
            live.fill_(7)
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output, torch.full_like(output, 14))
            with self.assertRaisesRegex(RuntimeError, "one-shot"):
                owner.before_capture()


if __name__ == "__main__":
    unittest.main()
