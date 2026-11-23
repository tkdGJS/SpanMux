#!/usr/bin/env python3
"""Offline exact-budget and ordering contracts; no model or GPU is required."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import prepare_loongserve_mixed_workload as prep
from measure_loongserve_mixed_workload import load_trace, replay_selection_state


class ByteOffsetTokenizer:
    """Two special tokens; Unicode bytes deliberately have repeated offsets."""

    is_fast = True
    name_or_path = "mock-exact-byte-offset-tokenizer"

    def __call__(self, texts, *, add_special_tokens=True, **kwargs):
        if isinstance(texts, list):
            return {"input_ids": [
                self(text, add_special_tokens=add_special_tokens)["input_ids"]
                for text in texts
            ]}
        ids, offsets = [], []
        for index, char in enumerate(texts):
            encoded = list(char.encode("utf-8"))
            ids.extend(encoded)
            offsets.extend([(index, index + 1)] * len(encoded))
        if add_special_tokens:
            ids = [256, *ids, 257]
        return {"input_ids": ids, "offset_mapping": offsets}


class SeamTokenizer(ByteOffsetTokenizer):
    """A new A/Z seam costs extra tokens, emulating non-additive tokenization."""

    def __call__(self, texts, *, add_special_tokens=True, **kwargs):
        result = super().__call__(texts, add_special_tokens=add_special_tokens, **kwargs)
        if isinstance(texts, str) and "AZ" in texts:
            result["input_ids"].extend([300] * 5)
        return result


class TruncationContracts(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(
            prep.TokenCounter, "_resolve_tokenizer_source", return_value="mock"
        ))
        self.loader = self.stack.enter_context(patch.object(
            prep.TokenCounter, "_load_tokenizer", return_value=ByteOffsetTokenizer()
        ))
        self.counter = self.make_counter()

    @staticmethod
    def make_counter():
        return prep.TokenCounter(
            mode="hf", tokenizer_name="mock", local_files_only=True,
            workers=1, batch_size=2,
        )

    def test_exact_boundary_and_short_inputs_are_unchanged(self):
        for text in ("small", "abcdefgh"):
            self.assertEqual(
                self.counter.truncate_prompt_head_tail(text, 10),
                (text, self.counter.count(text)),
            )

    def test_100000_means_decimal_input_tokens_including_specials(self):
        text = "START" + "x" * 140000 + "ENDQUESTION"
        prompt, count = self.counter.truncate_prompt_head_tail(text, 100000)
        self.assertEqual(count, 100000)
        self.assertTrue(prompt.startswith("START"))
        self.assertTrue(prompt.endswith("ENDQUESTION"))
        self.assertEqual(self.counter.count(prompt), count)

    def test_unicode_character_boundaries_not_lossy_byte_decode(self):
        text = "가나다라마바사아자차카타파하🙂끝"
        prompt, count = self.counter.truncate_prompt_head_tail(text, 23)
        self.assertLessEqual(count, 23)
        self.assertTrue(prompt.startswith("가"))
        self.assertTrue(prompt.endswith("끝"))
        self.assertNotIn("\ufffd", prompt)
        self.assertEqual(count, self.counter.count(prompt))

    def test_join_is_retokenized_and_shrunk_if_seam_expands(self):
        self.loader.return_value = SeamTokenizer()
        counter = self.make_counter()
        prompt, count = counter.truncate_prompt_head_tail(
            "A" * 25 + "B" * 50 + "Z" * 25, 22
        )
        self.assertLessEqual(count, 22)
        self.assertIn("AZ", prompt)
        self.assertEqual(counter.count(prompt), count)

    def test_tiny_budget_and_inexact_tokenizer_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "nonempty prompt"):
            self.counter.truncate_prompt_head_tail("oversized", 2)
        self.counter.is_fast_tokenizer = False
        with self.assertRaisesRegex(ValueError, "exact HF fast"):
            self.counter.truncate_prompt_head_tail("oversized", 10)

    def candidate(self, dataset="leval", index=0, output=8):
        text = f"{dataset}-{index}:" + "x" * (60 + index) + "QUESTION"
        return prep.Candidate(
            dataset=dataset, source_id=f"{dataset}-{index}", source_path="mock",
            prompt=text, reference_output="reference",
            input_tokens=self.counter.count(text), output_tokens=output,
            paper_input_tokens=self.counter.count(text), paper_output_tokens=output,
        )

    def test_audit_and_output_identity(self):
        original = self.candidate()
        effective, audit = prep.truncate_candidate(
            original, self.counter, max_input_tokens=40,
            max_total_tokens=48, max_output_tokens=0,
        )
        self.assertEqual(effective.output_tokens, original.output_tokens)
        self.assertEqual(effective.reference_output, original.reference_output)
        self.assertEqual(effective.source_id, original.source_id)
        self.assertEqual(effective.paper_input_tokens, original.paper_input_tokens)
        self.assertEqual(audit["original_input_tokens"], original.input_tokens)
        self.assertTrue(audit["input_truncated"])
        self.assertEqual(audit["removed_input_tokens"], original.input_tokens - 40)

    def test_total_and_output_caps_never_drop_or_shorten_output(self):
        for total, output in ((47, 0), (0, 7)):
            with self.subTest(total=total, output=output):
                with self.assertRaisesRegex(ValueError, "no request was dropped"):
                    prep.truncate_candidate(
                        self.candidate(), self.counter, max_input_tokens=40,
                        max_total_tokens=total, max_output_tokens=output,
                    )

    def run_prepare(self, target, *, policy, cap=0, total=0, output=0, workers=1):
        argv = [
            "prepare", "--sharegpt-path", "unused", "--leval-path", "unused",
            "--lveval-path", "unused", "--output-path", str(target),
            "--tokenizer", "mock", "--tokenizer-workers", str(workers),
            "--requests-per-dataset", "40", "--request-rate", "0.5",
            "--length-cap-policy", policy, "--max-input-tokens", str(cap),
            "--max-total-tokens", str(total), "--max-output-tokens", str(output),
            "--progress-every", "0",
        ]
        with contextlib.ExitStack() as stack:
            for dataset, factory in (
                ("sharegpt", "sharegpt_candidates"), ("leval", "leval_candidates"),
                ("lveval", "lveval_candidates"),
            ):
                stack.enter_context(patch.object(
                    prep, factory, return_value=[
                        self.candidate(dataset, i) for i in range(50)
                    ]
                ))
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(prep.main(), 0)
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        manifest = json.loads(target.with_suffix(".jsonl.manifest.json").read_text())
        return rows, manifest

    def test_uncapped_identity_ratio_arrivals_and_first100_survive_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, _ = self.run_prepare(root / "original.jsonl", policy="filter_then_sample")
            truncated, manifest = self.run_prepare(
                root / "truncated.jsonl", policy="paper_sample_then_truncate", cap=40,
                total=48,
            )
            repeated, _ = self.run_prepare(
                root / "parallel.jsonl", policy="paper_sample_then_truncate", cap=40,
                total=48, workers=8,
            )
            self.assertEqual(repeated, truncated)
            self.assertEqual(len(original), 120)
            self.assertEqual(len(truncated), len(original))
            self.assertEqual(Counter(r["dataset"] for r in truncated),
                             {"sharegpt": 40, "leval": 40, "lveval": 40})
            for old, new in zip(original, truncated):
                for key in old:
                    if key not in ("prompt", "input_tokens"):
                        self.assertEqual(new[key], old[key], key)
                self.assertLessEqual(new["input_tokens"], 40)
                self.assertEqual(new["original_input_tokens"], old["input_tokens"])
            self.assertEqual(manifest["sampled_before_cap"], manifest["selected_after_cap"])
            self.assertEqual(manifest["prompt_truncation"]["truncated_requests"], 120)
            self.assertEqual(manifest["reproduction_class"], "model_feasible_derivative")
            requests = load_trace(root / "truncated.jsonl")
            selected = [request for index, request in enumerate(requests)
                        if replay_selection_state(
                            position=index, effective_offset_s=request.base_scheduled_offset_s,
                            termination_mode="requests", request_limit=100, time_limit_sec=0,
                        ) == "submit"]
            self.assertEqual(len(selected), 100)
            self.assertEqual([r.source_id for r in selected],
                             [r["source_id"] for r in original[:100]])

    def test_invalid_selected_row_fails_before_publishing_any_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "invalid.jsonl"
            with self.assertRaisesRegex(ValueError, "exceeds output/total cap"):
                self.run_prepare(target, policy="paper_sample_then_truncate", cap=40, total=47)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".jsonl.manifest.json").exists())

    def test_default_cli_and_approx_truncation_rejection(self):
        args = ["prepare", "--sharegpt-path", "x", "--leval-path", "y",
                "--lveval-path", "z", "--output-path", "/unused", "--tokenizer", "mock"]
        with patch.object(sys, "argv", args):
            self.assertEqual(prep.parse_args().length_cap_policy, "filter_then_sample")
        with patch.object(sys, "argv", [*args, "--tokenizer-mode", "approx",
                                       "--length-cap-policy", "paper_sample_then_truncate",
                                       "--max-input-tokens", "100000"]):
            with self.assertRaisesRegex(ValueError, "--tokenizer-mode hf"):
                prep.main()

    def test_wrapper_100_requests_drain_and_truncation_dry_run(self):
        env = dict(os.environ, RUN_MODE="prepare_only", TOKENIZER_MODE="hf",
                   LENGTH_CAP_POLICY="paper_sample_then_truncate",
                   MAX_INPUT_TOKENS="100000", MAX_REQUEST_TOKENS="131072",
                   TERMINATION_MODE="requests", REQUEST_LIMIT="100",
                   TIME_LIMIT_SEC="0", TIME_LIMIT_MODE="drain")
        runner = ROOT / "run_loongserve_mixed_workload.sh"
        result = subprocess.run(["bash", str(runner), "--dry-run"], env=env,
                                text=True, capture_output=True, check=True)
        self.assertIn("termination_mode=requests request_limit=100", result.stdout)
        self.assertIn("time_limit_sec=0 time_limit_mode=drain", result.stdout)
        self.assertIn("max_input_tokens=100000 max_request_tokens=131072", result.stdout)
        self.assertIn('--length-cap-policy "${LENGTH_CAP_POLICY}"', runner.read_text())
        result = subprocess.run(["bash", str(runner), "--dry-run"],
                                env=dict(env, TOKENIZER_MODE="approx"),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
