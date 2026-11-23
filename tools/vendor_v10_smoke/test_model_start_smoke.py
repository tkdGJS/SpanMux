from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = ROOT / "start_vllm.sh"
SMOKE_SCRIPT = ROOT / "run_vendor_v10_model_start_smoke.sh"


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("VLLM_") or name in {
            "MODEL_NAME",
            "MODEL_NAMES",
            "PORT",
            "REQUIRE_H200",
            "VERIFY_GENERATION",
            "SMOKE_ENFORCE_EAGER",
        }:
            env.pop(name, None)
    return env


class VendorV10ModelStartSmokeContractTest(unittest.TestCase):
    def test_both_qwen_profiles_resolve_in_one_script(self) -> None:
        result = subprocess.run(
            ["bash", str(SMOKE_SCRIPT), "--all", "--dry-run"],
            cwd=ROOT,
            env=clean_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("model=Qwen/Qwen2.5-7B-Instruct-1M", result.stdout)
        self.assertIn("profile=qwen2.5_7b_1m", result.stdout)
        self.assertIn("kv_cache_dtype=auto calculate_kv_scales=0", result.stdout)
        self.assertIn("model=Qwen/Qwen2.5-14B-Instruct-1M", result.stdout)
        self.assertIn("profile=qwen2.5_14b_1m", result.stdout)
        self.assertIn("kv_cache_dtype=fp8 calculate_kv_scales=1", result.stdout)
        self.assertEqual(result.stdout.count("max_model_len=1000000"), 2)
        self.assertEqual(result.stdout.count("num_gpu_blocks_override=auto"), 2)

    def test_explicit_options_override_model_profile(self) -> None:
        env = clean_env()
        env.update(
            {
                "VLLM_MAX_MODEL_LEN": "262144",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
                "VLLM_KV_CACHE_DTYPE": "auto",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.90",
                "SMOKE_ENFORCE_EAGER": "0",
            }
        )
        result = subprocess.run(
            [
                "bash",
                str(SMOKE_SCRIPT),
                "--model",
                "Qwen/Qwen2.5-14B-Instruct-1M",
                "--dry-run",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("max_model_len=262144", result.stdout)
        self.assertIn("max_num_batched_tokens=16384", result.stdout)
        self.assertIn("kv_cache_dtype=auto", result.stdout)
        self.assertIn("gpu_memory_utilization=0.90", result.stdout)
        self.assertIn("[PLAN] extra_args=\n", result.stdout)

    def test_h200_start_profile_emits_capacity_and_parallel_flags(self) -> None:
        env = clean_env()
        env.update(
            {
                "VLLM_HARDWARE_PROFILE": "h200_141gb",
                "VLLM_EXECUTABLE": "/bin/echo",
                "VLLM_VENDOR_DIR": "",
                "VLLM_MAX_MODEL_LEN": "1000000",
                "VLLM_MAX_NUM_SEQS": "32",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "32768",
                "VLLM_KV_CACHE_DTYPE": "fp8",
                "VLLM_CALCULATE_KV_SCALES": "1",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.95",
                "VLLM_TENSOR_PARALLEL_SIZE": "1",
                "VLLM_PIPELINE_PARALLEL_SIZE": "1",
                "VLLM_NUM_GPU_BLOCKS_OVERRIDE": "",
            }
        )
        result = subprocess.run(
            [
                "bash",
                str(START_SCRIPT),
                "--model",
                "Qwen/Qwen2.5-14B-Instruct-1M",
                "--port",
                "8100",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("hardware_profile=h200_141gb", result.stdout)
        self.assertIn("num_gpu_blocks_override=\n", result.stdout)
        self.assertIn("--gpu-memory-utilization 0.95", result.stdout)
        self.assertIn("--tensor-parallel-size 1", result.stdout)
        self.assertIn("--pipeline-parallel-size 1", result.stdout)
        self.assertIn("--calculate-kv-scales", result.stdout)
        self.assertNotIn("--num-gpu-blocks-override", result.stdout)

    def test_legacy_start_defaults_remain_unchanged(self) -> None:
        env = clean_env()
        env.update(
            {
                "VLLM_EXECUTABLE": "/bin/echo",
                "VLLM_VENDOR_DIR": "",
            }
        )
        result = subprocess.run(
            ["bash", str(START_SCRIPT), "--model", "dummy/model"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("hardware_profile=legacy", result.stdout)
        self.assertIn("dtype=half", result.stdout)
        self.assertIn("max_model_len=8192", result.stdout)
        self.assertIn("--num-gpu-blocks-override 512", result.stdout)
        self.assertNotIn("--gpu-memory-utilization", result.stdout)


if __name__ == "__main__":
    unittest.main()
