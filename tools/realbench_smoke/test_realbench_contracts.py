#!/usr/bin/env python3
"""CPU-only contract smoke test for RealBench preparation and replay plans."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class MockVLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = json.dumps({"data": [{"id": "mock-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        chunks = [
            {"choices": [{"text": "B", "finish_reason": None}]},
            {"choices": [{"text": " ", "finish_reason": None}]},
            {"choices": [{"text": "", "finish_reason": "length"}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="realbench_smoke_") as raw_tmp:
        tmp = Path(raw_tmp)
        ruler = tmp / "ruler.jsonl"
        infinite = tmp / "infinite.jsonl"
        longbench = tmp / "longbench.jsonl"
        burst = tmp / "burst.csv"
        output = tmp / "prepared"
        write_jsonl(
            ruler,
            [{"task": "niah", "input": "Find the hidden needle.", "outputs": ["needle"]}],
        )
        write_jsonl(
            infinite,
            [{"task": "longbook_qa_eng", "context": "A long book.", "input": "What is it?", "answer": "book"}],
        )
        write_jsonl(
            longbench,
            [{
                "_id": "lb-1",
                "domain": "Single-Document QA",
                "context": "The answer is blue.",
                "question": "Which color?",
                "choice_A": "red",
                "choice_B": "blue",
                "choice_C": "green",
                "choice_D": "black",
                "answer": "B",
            }],
        )
        with burst.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "Timestamp",
                    "Session ID",
                    "Model",
                    "Request tokens",
                    "Response tokens",
                    "Total tokens",
                    "Log Type",
                ],
            )
            writer.writeheader()
            writer.writerow({"Timestamp": "2024-01-01T00:00:00Z", "Session ID": "s1", "Model": "gpt", "Request tokens": 10, "Response tokens": 2, "Total tokens": 12, "Log Type": "request"})
            writer.writerow({"Timestamp": "2024-01-01T00:00:02Z", "Session ID": "s2", "Model": "gpt", "Request tokens": 20, "Response tokens": 3, "Total tokens": 23, "Log Type": "request"})

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "prepare_realbench_datasets.py"),
                "--ruler",
                str(ruler),
                "--infinitebench",
                str(infinite),
                "--longbench-v2",
                str(longbench),
                "--burstgpt",
                str(burst),
                "--output-dir",
                str(output),
                "--tokenizer-mode",
                "approx",
                "--length-bins",
                "tiny:1:1024",
            ],
            check=True,
            cwd=ROOT,
        )

        sys.path.insert(0, str(ROOT))
        from replay_realbench import (
            build_replay_plan,
            load_content,
            load_trace,
            score_response,
        )

        for benchmark in ("ruler", "infinitebench", "longbench_v2"):
            records = load_content(output / f"{benchmark}.jsonl")
            assert len(records) == 1
            assert records[0].input_tokens > 0
            assert (output / f"{benchmark}__tiny.jsonl").exists()

        trace = load_trace(output / "burstgpt_trace.jsonl")
        assert [record.offset_s for record in trace] == [0.0, 2.0]
        content = load_content(output / "longbench_v2.jsonl")
        plans = build_replay_plan(
            content=content,
            arrival_mode="burstgpt",
            request_count=2,
            request_gap_sec=1,
            tenant_count=2,
            trace=trace,
            trace_start_index=0,
            trace_window_sec=0,
            arrival_rate_scale=2,
            assignment="cycle",
        )
        assert [plan.offset_s for plan in plans] == [0.0, 1.0]
        assert [plan.tenant_id for plan in plans] == ["s1", "s2"]
        assert score_response("longbench_v2_mcq", "B", "B")[1:] == (1, 1)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["burstgpt_semantics"].startswith("arrival trace only")
        assert manifest["model_context_limit"] == 131072
        assert manifest["reserved_generation_tokens"] == 4096
        assert manifest["input_token_filter"][1] == 126976

        dry_run_env = {
            **os.environ,
            "REALBENCH_DATA_DIR": str(output),
            "REALBENCH_RESULT_DIR": str(tmp / "dry_run_results"),
            "REALBENCH_DRY_RUN": "1",
            "VLLM_VENDOR_DIR": str(ROOT / "vendor_v4"),
            "VLLM_PD_EVAL_POLICY_SWEEP": "static_pd",
            "REALBENCH_CONTENT_BENCHMARKS": (
                "ruler infinitebench longbench_v2"
            ),
            "REALBENCH_ARRIVAL_PROFILES": "fixed burstgpt",
            "REALBENCH_LENGTH_BINS": "all",
            "GREEN_SM_SPLITS": "20/20",
            "REPEAT_COUNT": "1",
        }
        subprocess.run(
            ["bash", str(ROOT / "run_realbench_4bench_4GB.sh")],
            check=True,
            cwd=ROOT,
            env=dry_run_env,
        )

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), MockVLLMHandler)
        except PermissionError:
            server = None
            print("[SKIP] local socket creation is blocked by the sandbox")
        if server is not None:
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            replay_output = tmp / "replay"
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "replay_realbench.py"),
                        "--base-url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--content-jsonl",
                        str(output / "longbench_v2.jsonl"),
                        "--arrival-mode",
                        "fixed",
                        "--request-count",
                        "2",
                        "--request-gap-sec",
                        "0.01",
                        "--fixed-output-tokens",
                        "2",
                        "--min-output-tokens",
                        "2",
                        "--output-token-cap",
                        "2",
                        "--ignore-eos",
                        "--case-id",
                        "mock-realbench",
                        "--output-dir",
                        str(replay_output),
                        "--output-tokenizer-mode",
                        "one_per_chunk",
                        "--fail-if-all-requests-fail",
                    ],
                    check=True,
                    cwd=ROOT,
                )
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)
            with (replay_output / "result_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                replay_summary = next(csv.DictReader(handle))
            assert replay_summary["successes"] == "2"
            assert replay_summary["failures"] == "0"
            assert (replay_output / "responses.jsonl").exists()

        if importlib.util.find_spec("matplotlib") is not None:
            graph_case = (
                tmp
                / "result"
                / "longbench_v2"
                / "all"
                / "fixed"
                / "static_pd"
                / "d20_p20"
                / "repeat_1"
            )
            graph_case.mkdir(parents=True)
            with (graph_case / "result_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "requests",
                        "successes",
                        "p99_request_p99_tbt_ms",
                        "p99_ttft_ms",
                        "output_throughput_tps",
                        "tbt_slo_attainment",
                        "tenant_output_rate_jain",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "requests": 3,
                        "successes": 3,
                        "p99_request_p99_tbt_ms": 10,
                        "p99_ttft_ms": 100,
                        "output_throughput_tps": 20,
                        "tbt_slo_attainment": 1,
                        "tenant_output_rate_jain": 1,
                    }
                )
            (graph_case / "case_metadata.json").write_text(
                json.dumps(
                    {
                        "policy": "static_pd",
                        "benchmark": "longbench_v2",
                        "length_bin": "all",
                        "arrival_source": "fixed",
                        "decode_sms": 20,
                        "prefill_sms": 20,
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "graph_script/generate_realbench_graphs.py"),
                    str(tmp / "result"),
                ],
                check=True,
                cwd=ROOT,
                env={**os.environ, "MPLCONFIGDIR": str(tmp / "mpl")},
            )
            assert list((tmp / "result" / "generated_graphs").rglob("*.png"))

    print("[OK] RealBench preparation and replay contracts validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
