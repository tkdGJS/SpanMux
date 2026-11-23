#!/usr/bin/env python3
"""CPU-only tests for the fixed-seed 10-request ShareGPT runner."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "run_sharegpt_randomized_mixed_context_10req_4GB.sh"
sys.path.insert(0, str(ROOT))

from prepare_ctx100k_dataset_examples import PromptSource  # noqa: E402
from prepare_mixed_context_sharegpt import (  # noqa: E402
    select_distinct_sources,
)


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[str]:
        del add_special_tokens
        return text.split()


def _source(source_id: str, token_count: int) -> PromptSource:
    return PromptSource(
        workload="sharegpt",
        prefix="",
        body=" ".join([source_id] * token_count),
        suffix=" Assistant:",
        source_ids=(f"{source_id}:0",),
        source_path=Path("sharegpt.json"),
        role_default="foreground",
        stream_default=True,
        construction="single_conversation_prefix_truncated",
        source_faithful=True,
    )


def _dry_run(
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), "--dry-run", *args],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def _input_sequence(output: str) -> list[int]:
    prefix = "[PLAN] input_tokens="
    line = next(line for line in output.splitlines() if line.startswith(prefix))
    return [int(value) for value in line.removeprefix(prefix).split()]


class RandomizedMixedContextRunnerTest(unittest.TestCase):
    def test_fixed_seed_plan_has_two_requests_per_bucket(self) -> None:
        first = _dry_run("--request-gap-sec", "0.25")
        second = _dry_run("--request-gap-sec", "0.25")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)

        sequence = _input_sequence(first.stdout)
        self.assertEqual(
            sequence,
            [
                25000,
                100000,
                100000,
                75000,
                50000,
                125000,
                125000,
                25000,
                50000,
                75000,
            ],
        )
        self.assertEqual(
            Counter(sequence),
            Counter(
                {
                    25000: 2,
                    50000: 2,
                    75000: 2,
                    100000: 2,
                    125000: 2,
                }
            ),
        )
        self.assertIn("[PLAN] seed=20260725", first.stdout)
        self.assertIn("[PLAN] request_gap_sec=0.25", first.stdout)

    def test_request_gap_can_come_from_environment(self) -> None:
        env = dict(os.environ)
        env["REQUEST_GAP_SEC"] = "1.75"
        result = _dry_run(env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[PLAN] request_gap_sec=1.75", result.stdout)

    def test_prefill_policy_sweep_is_visible_in_dry_run(self) -> None:
        env = dict(os.environ)
        env["TTFT_PREFILL_POLICY_SWEEP"] = "fcfs srpt round_robin"
        env["TTFT_PREFILL_CHUNK_QUANTUM_TOKENS"] = "2048"
        result = _dry_run(env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "[PLAN] ttft_prefill_policy_sweep=fcfs srpt round_robin",
            result.stdout,
        )
        self.assertIn(
            "[PLAN] ttft_prefill_chunk_quantum_tokens=2048",
            result.stdout,
        )

    def test_unknown_prefill_policy_is_rejected(self) -> None:
        env = dict(os.environ)
        env["TTFT_PREFILL_POLICY_SWEEP"] = "fcfs mystery"
        result = _dry_run(env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported prefill policy", result.stderr)

    def test_negative_request_gap_is_rejected(self) -> None:
        result = _dry_run("--request-gap-sec", "-0.1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-negative", result.stderr)

    def test_duplicate_targets_still_use_distinct_sources(self) -> None:
        selected = select_distinct_sources(
            _WhitespaceTokenizer(),
            [
                _source("a", 30),
                _source("b", 35),
                _source("c", 55),
                _source("d", 60),
            ],
            [25, 50, 25, 50],
        )
        source_ids = [
            source.source_ids[0] for _target, source, _full_tokens in selected
        ]
        self.assertEqual(len(source_ids), 4)
        self.assertEqual(len(set(source_ids)), 4)
        self.assertEqual(
            [target for target, _source_row, _full_tokens in selected],
            [25, 50, 25, 50],
        )

    def test_runner_builds_scheduler_args_inside_policy_loop(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for required in (
            "for prefill_policy in ${TTFT_PREFILL_POLICY_SWEEP}",
            "--enable-ttft-prefill-chunk-scheduling",
            "--ttft-prefill-chunk-scheduling-policy ${requested_policy}",
            "--ttft-prefill-chunk-quantum-tokens",
            "--enable-tbt-decode-scheduling",
            "prefill_${prefill_policy_id}",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
