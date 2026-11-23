#!/usr/bin/env python3
"""Opt-in subprocess test: tensors/graphs may die after GreenContext retirement."""
import os
import subprocess
import sys
import unittest


@unittest.skipUnless(os.getenv("VLLM_ISOLATION_RUN_CUDA_TESTS") == "1",
                     "requires CUDA12.8 GreenContext bridge and Torch")
class GreenLifetimeCUDA(unittest.TestCase):
    def test_late_allocator_cleanup_and_process_exit(self):
        # Fresh process for each run is intentional: retired contexts cannot
        # safely be destroyed/re-created in a process-global caching allocator.
        program = r'''
import gc
import os
os.environ["VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME"] = "1"
import torch
from vllm.model_executor.layers import green_context as g
from vllm.pd_sm_profiles import PDSMProfile
torch.cuda.init()
total = torch.cuda.get_device_properties(0).multi_processor_count
s = g._GreenContextStreams(profile=PDSMProfile(total, 0))
stream = s.decode_stream
with torch.cuda.stream(stream):
    a = torch.ones(262144, device="cuda")
    b = a * 3
    a.record_stream(stream)
    b.record_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        c = b + 2
graph.replay()
stream.synchronize()
assert bool((c == 5).all())
s.close()
g.retire_process_owned_green_contexts()
del s
gc.collect()
# These allocations/graph still refer to the exported external stream.
del a, b, c, graph
gc.collect()
torch.cuda.empty_cache()  # TEST ONLY: force allocator's late event cleanup.
torch.cuda.synchronize()
print("LATE_CUDA_CLEANUP_OK", flush=True)
'''
        for _ in range(2):
            result = subprocess.run([sys.executable, "-c", program], env=os.environ.copy(),
                                    text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("LATE_CUDA_CLEANUP_OK", result.stdout)
            self.assertNotIn("context is destroyed", result.stderr)


if __name__ == "__main__":
    unittest.main()
