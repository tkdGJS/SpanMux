#!/usr/bin/env python3
"""Create tiny synthetic RealBench inputs for an end-to-end server smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def content_row(
    benchmark: str,
    task: str,
    prompt: str,
    ground_truth: Any,
    scorer: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": f"{benchmark}-mini-0",
        "source_id": f"synthetic-mini-{benchmark}",
        "benchmark": benchmark,
        "workload": benchmark,
        "task": task,
        "prompt": prompt,
        # This value is only a scheduling annotation in the smoke fixture. The
        # fixture is intentionally excluded from paper evaluation.
        "input_tokens": 128,
        "max_tokens": 8,
        "min_tokens": 8,
        "ground_truth": ground_truth,
        "scorer": scorer,
        "source_path": "synthetic_realbench_mini_fixture",
        "source_row": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_context = " ".join(
        ["The system stores a blue marker in the context."] * 24
    )
    rows = {
        "ruler": content_row(
            "ruler",
            "niah_smoke",
            f"{shared_context}\nFind the marker and answer with its color.",
            ["blue"],
            "ruler_contains",
        ),
        "infinitebench": content_row(
            "infinitebench",
            "longbook_qa_smoke",
            f"Context:\n{shared_context}\n\nQuestion:\nWhat color is the marker?\n\nAnswer:",
            "blue",
            "infinitebench_official_required",
        ),
        "longbench_v2": content_row(
            "longbench_v2",
            "single_document_qa_smoke",
            f"Context:\n{shared_context}\n\nQuestion:\nWhat color is the marker?\n\nA. red\nB. blue\nC. green\nD. black\n\nOutput only the letter of the correct option.",
            "B",
            "longbench_v2_mcq",
        ),
    }
    for benchmark, row in rows.items():
        write_jsonl(args.output_dir / f"{benchmark}.jsonl", [row])
    write_jsonl(args.output_dir / "content_all.jsonl", rows.values())
    write_jsonl(
        args.output_dir / "burstgpt_trace.jsonl",
        [
            {
                "schema_version": 1,
                "trace_index": 0,
                "source_row": 0,
                "timestamp": "synthetic-mini",
                "offset_s": 0.0,
                "session_id": "mini_session_0",
                "model": "synthetic",
                "request_tokens": 128,
                "response_tokens": 8,
                "total_tokens": 136,
                "elapsed_time_s": 0,
                "log_type": "synthetic_smoke_only",
            }
        ],
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "transport smoke only; not benchmark evaluation",
                "benchmarks": list(rows),
                "burstgpt_semantics": "synthetic arrival fixture",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] mini_fixture={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
