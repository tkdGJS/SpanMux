#!/usr/bin/env python3
"""CPU-only contracts for LoongServe Mixed preparation and replay controls."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW_DATASET_SCRIPT = ROOT / "prepare_loongserve_raw_datasets.sh"
LVEVAL_SUBSETS = (
    "factrecall_en",
    "hotpotwikiqa_mixup",
    "loogle_CR_mixup",
    "loogle_MIR_mixup",
    "loogle_SD_mixup",
    "multifieldqa_en_mixup",
)
LVEVAL_LEVELS = ("16k", "32k", "64k", "128k", "256k")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def raw_dataset_env(target: Path, source: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LOONGSERVE_REPO_ROOT": str(target),
            "LOONGSERVE_SHAREGPT_URL": (source / "sharegpt.json").as_uri(),
            "LOONGSERVE_LEVAL_REPO_URL": (source / "LEval").as_uri(),
            "LOONGSERVE_LVEVAL_BASE_URL": (source / "LVEval").as_uri(),
            "PYTHON_BIN": sys.executable,
        }
    )
    return env


def build_raw_download_sources(source: Path) -> None:
    source.mkdir(parents=True)
    write_json(
        source / "sharegpt.json",
        [
            {
                "id": "sharegpt-0",
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
        ],
    )

    leval = source / "LEval"
    (leval / "LEval-data" / "Open-ended-tasks").mkdir(parents=True)
    write_jsonl(
        leval / "LEval-data" / "Open-ended-tasks" / "sample.jsonl",
        [
            {
                "input": "document",
                "instructions": ["question"],
                "outputs": ["answer"],
            }
        ],
    )
    subprocess.run(["git", "init", "-q", str(leval)], check=True)
    subprocess.run(["git", "-C", str(leval), "add", "."], check=True)
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": "LoongServe test",
            "GIT_AUTHOR_EMAIL": "loongserve-test@example.invalid",
            "GIT_COMMITTER_NAME": "LoongServe test",
            "GIT_COMMITTER_EMAIL": "loongserve-test@example.invalid",
        }
    )
    subprocess.run(
        ["git", "-C", str(leval), "commit", "-q", "-m", "fixture"],
        check=True,
        env=commit_env,
    )

    lveval = source / "LVEval"
    lveval.mkdir()
    for subset in LVEVAL_SUBSETS:
        archive = lveval / f"{subset}.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            for level in LVEVAL_LEVELS:
                row = {
                    "context": f"{subset} {level} context",
                    "input": "question",
                    "answers": ["answer"],
                }
                stream.writestr(
                    f"{subset}/{subset}_{level}.jsonl",
                    json.dumps(row) + "\n",
                )


def test_raw_dataset_bootstrap(tmp: Path) -> None:
    if not all(shutil.which(tool) for tool in ("curl", "git", "unzip")):
        print("[SKIP] raw dataset bootstrap tools are unavailable")
        return

    source = tmp / "raw-sources"
    target = tmp / "raw-target"
    build_raw_download_sources(source)
    env = raw_dataset_env(target, source)

    missing = subprocess.run(
        ["bash", str(RAW_DATASET_SCRIPT), "--check-only"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert missing.returncode == 2
    assert "[SUMMARY] ready=0" in missing.stdout
    assert not target.exists()

    dry_run = subprocess.run(
        ["bash", str(RAW_DATASET_SCRIPT), "--dry-run"],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert "mode=dry_run files_changed=0" in dry_run.stdout
    assert not target.exists()

    prepared = subprocess.run(
        ["bash", str(RAW_DATASET_SCRIPT)],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert "[OK] LoongServe Mixed raw datasets are ready" in prepared.stdout
    assert "downloads=8 extractions=6 ready=1" in prepared.stdout
    assert (target / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json").is_file()
    assert (
        target
        / "data"
        / "raw"
        / "loongserve"
        / "LEval"
        / "LEval-data"
        / "Open-ended-tasks"
        / "sample.jsonl"
    ).is_file()
    for subset in LVEVAL_SUBSETS:
        for level in LVEVAL_LEVELS:
            assert (
                target
                / "data"
                / "raw"
                / "loongserve"
                / "LVEval"
                / subset
                / f"{subset}_{level}.jsonl"
            ).is_file()

    second = subprocess.run(
        ["bash", str(RAW_DATASET_SCRIPT)],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert "downloads=0 extractions=0 ready=1" in second.stdout


class SlowStreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        first = {"choices": [{"text": "a", "token_ids": [1], "finish_reason": None}]}
        try:
            self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
            self.wfile.flush()
            time.sleep(1.0)
            final = {
                "choices": [{"text": "b", "token_ids": [2], "finish_reason": "length"}]
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class TextOnlyStreamHandler(BaseHTTPRequestHandler):
    """Emulate a stale OpenAI server that ignores return_token_ids."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        chunks = [
            {"choices": [{"text": "a", "finish_reason": None}]},
            {"choices": [{"text": "b", "finish_reason": "length"}]},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def build_sources(tmp: Path) -> tuple[Path, Path, Path]:
    sharegpt = tmp / "sharegpt.json"
    leval = tmp / "leval.jsonl"
    lveval = tmp / "lveval.jsonl"
    write_json(
        sharegpt,
        [
            {
                "id": f"sg-{index}",
                "conversations": [
                    {"from": "human", "value": words(f"q{index}-", 24)},
                    {"from": "gpt", "value": words(f"a{index}-", 12)},
                    {"from": "human", "value": words(f"q2-{index}-", 8)},
                    {"from": "gpt", "value": words(f"a2-{index}-", 8)},
                ],
            }
            for index in range(5)
        ],
    )
    write_jsonl(
        leval,
        [
            {
                "id": f"le-{index}",
                "input": words(f"doc{index}-", 36),
                "instructions": [" answer this question"],
                "outputs": [words("answer-", 8)],
            }
            for index in range(5)
        ],
    )
    write_jsonl(
        lveval,
        [
            {
                "id": f"lv-{index}",
                "context": words(f"context{index}-", 38),
                "input": " question",
                "answers": [words("answer-", 8)],
            }
            for index in range(5)
        ],
    )
    return sharegpt, leval, lveval


def test_prepare(tmp: Path) -> Path:
    sharegpt, leval, lveval = build_sources(tmp)
    trace = tmp / "loongserve.jsonl"
    base_command = [
        sys.executable,
        str(ROOT / "prepare_loongserve_mixed_workload.py"),
        "--sharegpt-path",
        str(sharegpt),
        "--leval-path",
        str(leval),
        "--lveval-path",
        str(lveval),
        "--tokenizer",
        "unused-in-approx-mode",
        "--tokenizer-mode",
        "approx",
        "--requests-per-dataset",
        "2",
        "--request-rate",
        "2",
        "--max-total-tokens",
        "100",
        "--progress-every",
        "0",
    ]
    subprocess.run(
        [*base_command, "--output-path", str(trace)],
        check=True,
        cwd=ROOT,
    )
    parallel_trace = tmp / "loongserve-parallel.jsonl"
    subprocess.run(
        [
            *base_command,
            "--output-path",
            str(parallel_trace),
            "--tokenizer-workers",
            "4",
        ],
        check=True,
        cwd=ROOT,
    )
    assert trace.read_bytes() == parallel_trace.read_bytes()
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 6
    assert Counter(row["dataset"] for row in rows) == {
        "sharegpt": 2,
        "leval": 2,
        "lveval": 2,
    }
    assert all(row["input_tokens"] + row["max_tokens"] <= 100 for row in rows)
    offsets = [row["base_scheduled_offset_s"] for row in rows]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0.0
    assert abs(offsets[-1] - 3.0) < 1e-9
    manifest = json.loads(trace.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["reproduction_class"] == "model_feasible_derivative"
    assert manifest["length_caps"]["semantics"].startswith("whole-request")
    parallel_manifest = json.loads(
        parallel_trace.with_suffix(".jsonl.manifest.json").read_text()
    )
    assert manifest["trace_sha256"] == parallel_manifest["trace_sha256"]
    assert manifest["preparation_execution"]["tokenizer_workers"] == 1
    assert parallel_manifest["preparation_execution"]["tokenizer_workers"] == 4
    assert parallel_manifest["preparation_execution"]["tokenizer_batch_size"] == 32
    assert parallel_manifest["preparation_execution"]["ordered_batch_results"]
    assert (
        parallel_manifest["preparation_execution"]["token_count_cache_stats"]["hits"]
        > 0
    )
    return trace


def test_dry_run_scale_and_cutoff(trace: Path, tmp: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "measure_loongserve_mixed_workload.py"),
            "--base-url",
            "http://127.0.0.1:1",
            "--model",
            "mock",
            "--workload-jsonl",
            str(trace),
            "--case-id",
            "dry-run",
            "--output-dir",
            str(tmp / "dry-run"),
            "--arrival-time-scale",
            "2",
            "--time-limit-sec",
            "1",
            "--dry-run",
        ],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert "effective_offset=" in result.stdout
    assert "state=not_submitted_time_limit" in result.stdout


def test_dry_run_request_limit_preserves_trace_order(trace: Path, tmp: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "measure_loongserve_mixed_workload.py"),
            "--base-url",
            "http://127.0.0.1:1",
            "--model",
            "mock",
            "--workload-jsonl",
            str(trace),
            "--case-id",
            "request-limit-dry-run",
            "--output-dir",
            str(tmp / "request-limit-dry-run"),
            "--termination-mode",
            "requests",
            "--request-limit",
            "3",
            "--dry-run",
        ],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("[DRY-RUN]")
    ]
    assert [int(line.split("event=", 1)[1].split()[0]) for line in lines] == list(
        range(6)
    )
    assert [line.rsplit("state=", 1)[1] for line in lines] == [
        "submit",
        "submit",
        "submit",
        "not_submitted_request_limit",
        "not_submitted_request_limit",
        "not_submitted_request_limit",
    ]


def test_request_limit_rejects_mixed_termination(trace: Path, tmp: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "measure_loongserve_mixed_workload.py"),
            "--base-url",
            "http://127.0.0.1:1",
            "--model",
            "mock",
            "--workload-jsonl",
            str(trace),
            "--case-id",
            "mixed-termination",
            "--output-dir",
            str(tmp / "mixed-termination"),
            "--termination-mode",
            "requests",
            "--request-limit",
            "3",
            "--time-limit-sec",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "time-limit-sec must be zero in requests mode" in result.stderr

    too_many = subprocess.run(
        [
            sys.executable,
            str(ROOT / "measure_loongserve_mixed_workload.py"),
            "--base-url",
            "http://127.0.0.1:1",
            "--model",
            "mock",
            "--workload-jsonl",
            str(trace),
            "--case-id",
            "too-many-requests",
            "--output-dir",
            str(tmp / "too-many-requests"),
            "--termination-mode",
            "requests",
            "--request-limit",
            "7",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert too_many.returncode != 0
    assert "request-limit 7 exceeds trace rows 6" in too_many.stderr


def test_request_timeout_contract(trace: Path, tmp: Path) -> None:
    from measure_loongserve_mixed_workload import request_timeout_total_seconds

    assert request_timeout_total_seconds(0) is None
    assert request_timeout_total_seconds(0.25) == 0.25
    assert request_timeout_total_seconds(14400) == 14400
    for value in (-1, float("nan"), float("inf"), float("-inf")):
        try:
            request_timeout_total_seconds(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid timeout accepted: {value}")

    for label, value, enabled in (
        ("default", None, True),
        ("zero", "0", False),
        ("positive", "2.5", True),
        ("negative", "-1", None),
        ("nan", "nan", None),
        ("infinite", "inf", None),
    ):
        output = tmp / f"timeout-{label}"
        command = [
            sys.executable,
            str(ROOT / "measure_loongserve_mixed_workload.py"),
            "--base-url", "http://127.0.0.1:1",
            "--model", "mock",
            "--workload-jsonl", str(trace),
            "--case-id", f"timeout-{label}",
            "--output-dir", str(output),
        ]
        if value is not None:
            command.append(f"--request-timeout-sec={value}")
        if enabled is not None:
            command.append("--dry-run")
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, timeout=15
        )
        assert not output.exists()
        if enabled is None:
            assert result.returncode != 0
            assert "request-timeout-sec must be finite and non-negative" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            state = "enabled" if enabled else "disabled"
            assert f"processing_timeout={state}" in result.stdout
            assert "http_connect_timeout_sec=10" in result.stdout
            if value is None:
                assert "request_timeout_sec=14400" in result.stdout


def test_measurement_contract_helpers(tmp: Path) -> None:
    from measure_loongserve_mixed_workload import (
        IncompleteCompletionError,
        LoongServeRequest,
        build_completion_payload,
        exact_completion_token_ids,
        load_slo_policy,
        lookup_request_slo,
        validate_completion_contract,
    )

    assert exact_completion_token_ids({"token_ids": [1, "2"]}) == [1, 2]
    try:
        exact_completion_token_ids({})
    except IncompleteCompletionError:
        pass
    else:
        raise AssertionError("missing exact token IDs must fail closed")
    validate_completion_contract(
        output_tokens=2,
        requested_min_tokens=2,
        finish_reason="length",
        saw_done=True,
        streaming=True,
    )
    payload = build_completion_payload(
        LoongServeRequest(
            event_index=0,
            request_id="payload-contract",
            tenant_id="tenant",
            dataset="sharegpt",
            role="short",
            prompt="prompt",
            input_tokens=1,
            output_tokens=2,
            base_scheduled_offset_s=0.0,
            source_id="source",
        ),
        model="mock",
        slo={"tbt_slo_ms": 100.0},
        ignore_eos=True,
    )
    assert payload["return_token_ids"] is True
    assert payload["return_prompt_token_ids"] is False
    assert payload["prompt"] == "prompt"
    assert payload["max_tokens"] == payload["min_tokens"] == 2

    policy_path = tmp / "slo-policy.json"
    write_json(
        policy_path,
        {
            "slo_contract": {
                "tbt_measurement_source": "choice.token_ids",
                "tbt_gap_field": "estimated_per_token_gap_ms",
                "tbt_unit": "ms_per_output_token",
                "tbt_requires_return_token_ids": True,
            },
            "buckets": [
                {
                    "max_context_tokens": 65536,
                    "ttft_prefill_deadline_ms": 1000,
                    "tbt_slo_ms": 100,
                }
            ],
        },
    )
    policy = load_slo_policy(policy_path)
    assert lookup_request_slo(policy, 50000) == {
        "ttft_prefill_deadline_ms": 1000.0,
        "tbt_slo_ms": 100.0,
    }


def test_runner_token_contract_wiring() -> None:
    runner = (ROOT / "run_vendor_v8_short_validation.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'PD_PHASE_SYNC_SAMPLE_OUTPUT="'
        '${VLLM_PD_EVAL_PHASE_SYNC_SAMPLE_OUTPUT:-0}"' in runner
    )
    assert 'VLLM_PD_PHASE_SYNC_SAMPLE_OUTPUT="${PD_PHASE_SYNC_SAMPLE_OUTPUT}"' in runner
    assert "loongserve_token_id_contract=" in runner
    assert 'model_fields = getattr(model, "model_fields", None)' in runner
    assert 'getattr(model, "model_fields", getattr(' not in runner
    assert '"return_token_ids" in fields(CompletionRequest)' in runner
    assert '"return_prompt_token_ids" in fields(CompletionRequest)' in runner
    assert '"token_ids" in fields(CompletionResponseStreamChoice)' in runner
    assert '"prompt_token_ids" in fields(CompletionResponseStreamChoice)' in runner
    assert "request.return_prompt_token_ids" in runner
    assert '--termination-mode "${LOONGSERVE_TERMINATION_MODE}"' in runner
    assert '--request-limit "${LOONGSERVE_REQUEST_LIMIT}"' in runner
    assert (
        "LOONGSERVE_REQUEST_LIMIT={request_limit} exceeds trace rows={count}"
        in runner
    )
    assert '--request-timeout-sec "${REQUEST_TIMEOUT_SEC}"' in runner
    standalone = (ROOT / "run_loongserve_mixed_workload.sh").read_text()
    sweep = (ROOT / "run_vendor_v8_dynamic_hbm_vs_full.sh").read_text()
    assert '--request-timeout-sec "${REQUEST_TIMEOUT_SEC}"' in standalone
    assert 'REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC}"' in sweep
    # Shell :- substitutes only for unset/empty values, not the explicit "0".
    for script in (runner, standalone, sweep):
        assert 'REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-' in script


def test_native_and_vendor_prompt_token_id_opt_out_wiring() -> None:
    source_roots = (ROOT / "vendor_v8", ROOT / "src" / "vllm-0.14-src")
    for source_root in source_roots:
        protocol = (
            source_root / "vllm" / "entrypoints" / "openai" / "protocol.py"
        ).read_text(encoding="utf-8")
        serving = (
            source_root
            / "vllm"
            / "entrypoints"
            / "openai"
            / "serving_completion.py"
        ).read_text(encoding="utf-8")
        assert "return_prompt_token_ids: bool | None = Field(" in protocol
        assert "default=None" in protocol
        assert serving.count("request.return_prompt_token_ids is not False") == 3


def test_request_limit_drains_selected_requests(tmp: Path) -> None:
    trace = tmp / "request-limit.jsonl"
    common = {
        "schema_version": 1,
        "tenant_id": "tenant",
        "dataset": "sharegpt",
        "role": "short",
        "prompt": "prompt",
        "input_tokens": 10,
        "max_tokens": 2,
        "min_tokens": 2,
        "source_id": "source",
    }
    write_jsonl(
        trace,
        [
            {
                **common,
                "event_index": index,
                "request_id": f"request-limit-{index}",
                "base_scheduled_offset_s": index * 0.02,
            }
            for index in range(3)
        ],
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowStreamHandler)
    except PermissionError:
        print("[SKIP] local socket creation is blocked by the sandbox")
        return
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp / "request-limit-results"
    bounded_output = tmp / "request-limit-timeout-results"
    try:
        command = [
            sys.executable,
            str(ROOT / "measure_loongserve_mixed_workload.py"),
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--model",
            "mock",
            "--workload-jsonl",
            str(trace),
            "--case-id",
            "request-limit",
            "--termination-mode",
            "requests",
            "--request-limit",
            "2",
        ]
        bounded = subprocess.run(
            [*command, "--output-dir", str(bounded_output),
             "--request-timeout-sec", "0.05"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )
        assert bounded.returncode == 1, bounded.stderr
        bounded_meta = json.loads((bounded_output / "run_metadata.json").read_text())
        assert bounded_meta["request_timeout_enabled"] is True
        assert bounded_meta["summary"]["failure_count"] == 2
        assert bounded_meta["summary"]["success_count"] == 0
        subprocess.run(
            [*command, "--output-dir", str(output), "--request-timeout-sec", "0"],
            check=True,
            cwd=ROOT,
            timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    with (output / "request_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["event_index"] for row in rows] == ["0", "1", "2"]
    assert [row["status"] for row in rows] == [
        "success",
        "success",
        "not_submitted_request_limit",
    ]
    assert [row["submission_status"] for row in rows] == [
        "submitted",
        "submitted",
        "not_submitted",
    ]
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["termination_mode"] == "requests"
    assert metadata["request_limit"] == 2
    assert metadata["effective_request_limit"] == 2
    assert metadata["request_timeout_sec"] == 0
    assert metadata["request_timeout_enabled"] is False
    assert metadata["http_connect_timeout_sec"] == 10
    assert metadata["summary"]["submitted_count"] == 2
    assert metadata["summary"]["success_count"] == 2
    assert metadata["summary"]["not_submitted_count"] == 1


def test_cancel_cutoff(tmp: Path) -> None:
    trace = tmp / "cancel.jsonl"
    common = {
        "schema_version": 1,
        "tenant_id": "tenant",
        "dataset": "sharegpt",
        "role": "short",
        "prompt": "prompt",
        "input_tokens": 10,
        "max_tokens": 2,
        "min_tokens": 2,
        "source_id": "source",
    }
    write_jsonl(
        trace,
        [
            {
                **common,
                "event_index": 0,
                "request_id": "cancel-0",
                "base_scheduled_offset_s": 0.0,
            },
            {
                **common,
                "event_index": 1,
                "request_id": "cancel-1",
                "base_scheduled_offset_s": 1.0,
            },
        ],
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowStreamHandler)
    except PermissionError:
        print("[SKIP] local socket creation is blocked by the sandbox")
        return
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp / "cancel-results"
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "measure_loongserve_mixed_workload.py"),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--model",
                "mock",
                "--workload-jsonl",
                str(trace),
                "--case-id",
                "cancel",
                "--output-dir",
                str(output),
                "--time-limit-sec",
                "0.2",
                "--time-limit-mode",
                "cancel",
                "--request-timeout-sec",
                "0",
            ],
            check=True,
            cwd=ROOT,
            timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    with (output / "request_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == [
        "censored_time_limit",
        "not_submitted_time_limit",
    ]
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["time_limit_mode"] == "cancel"
    assert metadata["request_timeout_enabled"] is False
    assert metadata["summary"]["censored_count"] == 1


def test_missing_token_id_diagnostics(tmp: Path) -> None:
    trace = tmp / "text-only.jsonl"
    write_jsonl(
        trace,
        [
            {
                "schema_version": 1,
                "event_index": 0,
                "request_id": "text-only-0",
                "tenant_id": "tenant",
                "dataset": "sharegpt",
                "role": "short",
                "prompt": "prompt",
                "input_tokens": 10,
                "max_tokens": 2,
                "min_tokens": 2,
                "base_scheduled_offset_s": 0.0,
                "source_id": "source",
            }
        ],
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), TextOnlyStreamHandler)
    except PermissionError:
        print("[SKIP] local socket creation is blocked by the sandbox")
        return
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "measure_loongserve_mixed_workload.py"),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--model",
                "mock",
                "--workload-jsonl",
                str(trace),
                "--case-id",
                "text-only",
                "--output-dir",
                str(tmp / "text-only-results"),
                "--warmup-count",
                "1",
                "--warmup-output-tokens",
                "2",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode != 0
    assert "server emitted text but omitted choice.token_ids" in result.stderr
    assert "choice_chunks=2" in result.stderr
    assert "token_ids_fields=0" in result.stderr
    assert "saw_done=True" in result.stderr


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="loongserve_workload_") as raw:
        tmp = Path(raw)
        test_raw_dataset_bootstrap(tmp)
        trace = test_prepare(tmp)
        test_request_timeout_contract(trace, tmp)
        test_measurement_contract_helpers(tmp)
        test_runner_token_contract_wiring()
        test_native_and_vendor_prompt_token_id_opt_out_wiring()
        test_dry_run_scale_and_cutoff(trace, tmp)
        test_dry_run_request_limit_preserves_trace_order(trace, tmp)
        test_request_limit_rejects_mixed_termination(trace, tmp)
        test_missing_token_id_diagnostics(tmp)
        test_request_limit_drains_selected_requests(tmp)
        test_cancel_cutoff(tmp)
    print("[PASS] LoongServe workload CPU contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
