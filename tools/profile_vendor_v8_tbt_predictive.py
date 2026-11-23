#!/usr/bin/env python3
"""Replay matched P/D cases to calibrate predictive TBT; no GPU needed for --dry-run.

Uses the existing benchmark launcher for server startup, warmup, exact-token
replay, bounded cancellation and owned-process cleanup. Does not regenerate SLOs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from analyze_vendor_v8_tbt_calibration import analyze, analyze_case, sha256
from tbt_calibration_plan import (measurement_configuration, positive_ints,
    prompt_fingerprint, read_trace, replay_rows, steady_recipe, event_probe_recipe,
    liveness_recipe, coverage_for_cases)

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("dynamic_hbm_serialized_fullsm", "dynamic_hbm_srpt", "full_vendor_v8")
DEFAULTS = {
    "BENCHMARK_MODE": "full", "WORKLOAD": "LOONGSERVE",
    "ISOLATION_GUARDED_2K_COMPARISON": "1", "ISOLATION_MATCHED_COMPARISON": "1",
    "ISOLATION_BATCH2_PD11_COMPARISON": "0",
    "VLLM_MAX_NUM_SEQS": "100", "VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
    "KV_CACHE_MEMORY_BYTES": "2147483648", "VLLM_NUM_GPU_BLOCKS_OVERRIDE": "",
    "VLLM_MAX_MODEL_LEN": "65536", "PREFILL_CHUNK_TOKENS": "2048",
    "MODEL_NAME": "meta-llama/Llama-3.2-1B-Instruct",
    "VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS": "0",
    "VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS": "0",
    "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
    "VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS": "1",
    "VLLM_ISOLATION_PREFILL_BATCH_KV_BUDGET": "1",
    "VLLM_ISOLATION_RING_QUERY_EXPANSION": "0",
    "FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS": "1",
    "FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY": "scheduled_prefill_fit",
    "FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION": "0.98",
    "VLLM_PD_EVAL_PHASE_DEPTH_ENABLED": "1",
    "VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH": "1",
    "VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH": "1",
    "VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH": "2",
    "VLLM_PD_EVAL_PHASE_SYNC_SAMPLE_OUTPUT": "0",
    "VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF": "1",
    "VLLM_PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS": "0",
    "VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS": "1",
    "VLLM_PD_EVAL_RING_COMPUTE_GROUP_TOKENS": "2048",
    "VLLM_PD_EVAL_RING_BATCHED_PARTIAL": "0",
    "VLLM_PD_EVAL_RING_BATCHED_PARTIAL_STRICT": "0",
    "VLLM_PD_EVAL_RING_BATCHED_WORKSPACE_BYTES": "0",
    "VLLM_PD_EVAL_RING_BW_DYNAMIC": "1",
    "VLLM_PD_EVAL_RING_BW_MAX_TOKENS": "2048",
    "VLLM_PD_EVAL_RING_BW_USE_SCHEDULED_HEADROOM": "1",
    "VLLM_PD_EVAL_DRAM_REGISTER_DIRECT_MULTICOPY": "1",
    "VLLM_PD_EVAL_RELOAD_DIRECT_MULTICOPY": "1",
    "VLLM_PD_EVAL_RELOAD_STAGING_SCATTER": "0",
    "VLLM_PD_EVAL_RELOAD_FUSED_SCATTER": "0",
    "VLLM_PD_EVAL_RELOAD_FUSED_SCATTER_STRICT": "0",
    "VLLM_PD_EVAL_VERIFY_DIRECT_OPS": "1",
    "VLLM_VIRTUAL_CONTEXT_DECODE_CONTINUATION_ROLLING_RECLAIM": "0",
    "VLLM_KV_RETENTION_P1_PARKED_PREFILL": "1",
    "ITERATION_TRACE_SAMPLE_EVERY_N": "100", "ENABLE_DMON": "1",
    "WARMUP_COUNT": "1", "WARMUP_OUTPUT_TOKENS": "8", "WARMUP_COOLDOWN_SEC": "3",
    "REQUEST_TIMEOUT_SEC": "28800", "SERVER_READY_TIMEOUT_SEC": "900",
    "GENERATE_ABCD": "0", "KILL_EXISTING_PORT_OWNER": "0",
}


def parse_splits(value):
    splits = []
    for item in value.split(","):
        try:
            d, p = map(int, item.split("/"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("splits must be D/P,D/P") from exc
        if min(d, p) <= 0 or d + p != 40 or d % 2 or p % 2:
            raise argparse.ArgumentTypeError("T4 concurrent splits require positive even D/P totaling40")
        if (d, p) in splits:
            raise argparse.ArgumentTypeError("duplicate split")
        splits.append((d, p))
    return splits


def case_environment(base, args, *, case_id, variant, bank, initial):
    env = dict(base)
    env.pop("LOONGSERVE_TBT_PROBE_CONTROL", None)  # never inherit another case's gate
    for key, value in DEFAULTS.items():
        env.setdefault(key, value)
    # Calibration deliberately uses feedback to avoid a circular dependency on
    # the profile it is currently measuring. The matched wrapper validates the
    # remaining inherited flags (e.g. conflicting P2/Stage2) before startup.
    relative = args.output_dir.relative_to(ROOT)
    prefix = str(relative / "cases" / case_id)
    env.update(BASE_EXPERIMENT_NAME=prefix, BENCHMARK_VARIANTS=variant,
        WORKLOAD="LOONGSERVE", WORKLOADS="LOONGSERVE",
        PD_DYNAMIC_SM_POLICY="tbt_slo_feedback", PD_SM_PROFILES=bank,
        INITIAL_GREEN_SM_SPLIT=f"{initial[0]}/{initial[1]}",
        PD_FEEDBACK_INITIAL_DECODE_SMS=str(initial[0]),
        LOONGSERVE_TRACE_JSONL=str(args.trace),
        LOONGSERVE_ARRIVAL_TIME_SCALE=str(args.arrival_scale),
        LOONGSERVE_TERMINATION_MODE="time", LOONGSERVE_REQUEST_LIMIT="0",
        LOONGSERVE_TIME_LIMIT_SEC=str(args.case_seconds),
        LOONGSERVE_TIME_LIMIT_MODE="cancel",
        REQUEST_SLO_POLICY=str(args.slo),
        VLLM_PD_TBT_PROFILING_JSONL=str(args.output_dir / "captures" / f"{case_id}.jsonl"))
    env.setdefault("VLLM_VENDOR_DIR", str(ROOT / "vendor_v8"))
    env.setdefault("PROFILE_RESULTS_DIR", str(args.slo.parent.parent))
    # Local-only defaults; remote invocations normally export their VENV path.
    env.setdefault("VENV_DIR", sys.prefix)
    env.setdefault("VLLM_PYTHON_EXECUTABLE", str(Path(env["VENV_DIR"]) / "bin/python"))
    env.setdefault("VLLM_EXECUTABLE", str(Path(env["VENV_DIR"]) / "bin/vllm"))
    return env, ROOT / f"{prefix}_workload_loongserve_{variant}_results"


def plan_cases(args, base):
    cases = []
    full_bank = "0/40," + ",".join(f"{d}/{p}" for d, p in args.splits) + ",40/0"
    for repeat in range(1, args.repeats + 1):
        for variant in args.variants:
            for d, p in args.splits:
                case_id = f"r{repeat}_{variant}_d{d}_p{p}"
                env, result = case_environment(base, args, case_id=case_id,
                    variant=variant, bank=f"0/40,{d}/{p},40/0", initial=(d, p))
                cases.append(dict(case_id=case_id, kind="restricted_profile", variant=variant,
                                  env=env, results_dir=str(result),
                                  capture_path=env["VLLM_PD_TBT_PROFILING_JSONL"]))
            case_id = f"r{repeat}_{variant}_feedback"
            env, result = case_environment(base, args, case_id=case_id,
                variant=variant, bank=full_bank, initial=args.splits[0])
            cases.append(dict(case_id=case_id, kind="feedback", variant=variant,
                              env=env, results_dir=str(result),
                              capture_path=env["VLLM_PD_TBT_PROFILING_JSONL"]))
    return cases


def recorded_env(env):
    # No credentials/whole-environment dump. Explicit runtime experiment knobs only.
    prefixes = ("VLLM_", "PD_", "ISOLATION_", "FULL_VENDOR_", "DYNAMIC_HBM_",
                "BENCHMARK_", "LOONGSERVE_", "ENABLE_", "PREFILL_", "WARMUP_")
    keys = set(DEFAULTS) | {"BASE_EXPERIMENT_NAME", "REQUEST_SLO_POLICY",
                           "PROFILE_RESULTS_DIR", "MODEL_NAME", "VENV_DIR", "WORKLOAD"}
    secret_suffixes = ("_API_KEY", "_PASSWORD", "_SECRET", "_TOKEN", "_CREDENTIAL")
    return {key: value for key, value in env.items()
            if (key in keys or key.startswith(prefixes)) and not key.endswith(secret_suffixes)}


def select_trace(source, destination, count):
    """Copy the immutable prefix, preserving prompt/output/arrival fields verbatim."""
    selected = 0
    with source.open(encoding="utf-8") as src, destination.open("x", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            if count and selected >= count:
                break
            json.loads(line)  # fail early rather than writing invalid replay
            dst.write(line.rstrip("\n") + "\n")
            selected += 1
    if not selected or (count and selected < count):
        raise ValueError(f"trace has {selected} requests, required {count or 'at least one'}")
    return selected


def load_validation_reference(args, base):
    """Never execute predictive.env as shell code or train on validation rows."""
    if args.calibration_dir is None:
        raise ValueError("validate requires --calibration-dir from a ready controlled calibration")
    directory = args.calibration_dir.resolve()
    coverage = json.loads((directory / "coverage.json").read_text())
    profile_path = directory / "tbt_overlap_profile.json"
    profile = json.loads(profile_path.read_text())
    if not coverage.get("ready") or not coverage.get("declared_coverage_complete"):
        raise ValueError("validation needs a ready declared-grid calibration, not partial/legacy coverage")
    # Canonical sibling first supports archives copied between machines.
    manifest_path = directory.parent / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = Path(profile["manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_mode") != "calibrate":
        raise ValueError("validation reference must be a controlled calibrate run")
    if manifest["measurement_configuration"] != measurement_configuration(base, DEFAULTS):
        raise ValueError("validation model/KV/batch/graph/runtime options differ from calibration")
    if sha256(args.slo) != manifest["slo_sha256"]:
        raise ValueError("validation SLO differs from calibration")
    if (not set(args.variants) <= set(manifest["variants"])
            or [list(p) for p in args.splits] != manifest["splits"]):
        raise ValueError("validation variants/splits not covered by calibration")
    timings = coverage["calibration_ms"]
    for key in ("lead", "cutover", "min_step"):
        value = float(timings[key])
        if not math.isfinite(value) or value < 0 or (key != "cutover" and value == 0):
            raise ValueError(f"invalid calibration timing {key}")
    if profile.get("slo_sha256") != manifest["slo_sha256"]:
        raise ValueError("calibration profile and manifest disagree")
    if coverage.get("profile_sha256") != sha256(profile_path):
        raise ValueError("calibration profile is not the completed coverage artifact")
    return dict(directory=str(directory), profile_path=str(profile_path),
                profile_sha256=sha256(profile_path), manifest=manifest,
                calibration_ms=timings,
                calibration_id=sha256(manifest_path))


def plan_controlled_cases(args, base, source_rows, reference=None):
    """Intentional probes, with exact targets to verify after real execution."""
    config = dict(DEFAULTS, **base)
    full_bank = "0/40," + ",".join(f"{d}/{p}" for d, p in args.splits) + ",40/0"
    specs = []
    event_gated = getattr(args, "probe_protocol", "legacy") == "event-gated-v1"

    def append(case_id, variant, bank, initial, kind, recipe, **extra):
        env, result = case_environment(base, args, case_id=case_id, variant=variant,
                                       bank=bank, initial=initial)
        trace_path = args.output_dir / "traces" / f"{case_id}.jsonl"
        env["LOONGSERVE_TRACE_JSONL"] = str(trace_path)
        # Controlled simultaneous arrivals have seconds specified by their recipe.
        if kind == "steady" or recipe.get("control"):
            env["LOONGSERVE_ARRIVAL_TIME_SCALE"] = "1"
        control_path = None
        if recipe.get("control"):
            control_path = str(args.output_dir / "controls" / f"{case_id}.json")
            env["LOONGSERVE_TBT_PROBE_CONTROL"] = control_path
        if kind == "liveness":
            # A handful of Prefill chunks, not every token of a long Decode.
            env["ITERATION_TRACE_SAMPLE_EVERY_N"] = "1"
            # Otherwise the generic measurer warms up with its sole 64K row
            # BEFORE the bounded production clock and can stall for hours.
            # This preflight measures completion/liveness, not TBT performance.
            env["WARMUP_COUNT"] = "0"
        if extra.get("validation_policy") == "predictive":
            env.update(PD_DYNAMIC_SM_POLICY="tbt_slo_predictive",
                TBT_OVERLAP_PROFILE=reference["profile_path"],
                VLLM_PD_TBT_PREDICTIVE_LEAD_MS=str(reference["calibration_ms"]["lead"]),
                VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS=str(reference["calibration_ms"]["cutover"]),
                VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS=str(reference["calibration_ms"]["min_step"]))
        specs.append(dict(case_id=case_id, kind=kind, variant=variant, bank=bank,
            env=env, results_dir=str(result), trace_path=str(trace_path),
            control_path=control_path,
            capture_path=env["VLLM_PD_TBT_PROFILING_JSONL"], **recipe, **extra))

    for repeat in range(1, args.repeats + 1):
        for variant in args.variants:
            modes = ("native", "ring") if variant == "full_vendor_v8" else ("native",)
            if event_gated and args.mode in {"liveness", "calibrate"}:
                append(f"r{repeat}_{variant}_liveness", variant, full_bank, args.splits[0],
                       "liveness", liveness_recipe(source_rows, args=args, config=config))
            if args.mode in {"steady", "calibrate"}:
                for count in args.decode_batches:
                    for context in args.decode_contexts:
                        for mode in ("none", *modes):
                            recipe = event_probe_recipe(source_rows, args=args, config=config,
                                count=count, context=context, mode=mode) if event_gated else steady_recipe(source_rows, count=count, context=context,
                                window=args.context_window_tokens, output_tokens=args.probe_output_tokens,
                                mode=mode, max_seqs=int(config["VLLM_MAX_NUM_SEQS"]),
                                max_model_len=int(config["VLLM_MAX_MODEL_LEN"]),
                                kv_capacity_tokens=args.kv_capacity_tokens,
                                interference_tokens=args.interference_input_min_tokens,
                                interference_count=args.interference_requests,
                                interference_delay=args.interference_delay_sec)
                            profiles = [(40, 0)] if mode == "none" else args.splits
                            for d, p in profiles:
                                target = dict(recipe["target"], profile_id=f"d{d}_p{p}",
                                              min_decode_steps=args.min_samples)
                                append(f"r{repeat}_{variant}_b{count}_c{context}_{mode}_d{d}_p{p}",
                                    variant, full_bank if mode == "none" else f"0/40,{d}/{p},40/0",
                                    args.splits[0] if mode == "none" else (d, p), "steady",
                                    dict(recipe, target=target))
            if args.mode in {"transitions", "calibrate"}:
                if event_gated:
                    context = args.transition_context_tokens or max(args.decode_contexts or [1024])
                    for mode in modes:
                        recipe = event_probe_recipe(source_rows, args=args, config=config,
                            count=args.transition_base_requests, context=context, mode=mode,
                            transitions=True)
                        append(f"r{repeat}_{variant}_transitions_{mode}", variant, full_bank,
                            args.splits[0], "transitions", recipe, transition_modes=[mode],
                            transition_directions=["increase", "decrease"],
                            concurrent_transitions_only=True)
                else:
                    rows, origin = replay_rows(source_rows, args.request_limit)
                    append(f"r{repeat}_{variant}_transitions", variant, full_bank,
                        args.splits[0], "transitions", dict(rows=rows, plan_status="planned",
                        recipe=dict(kind="source_prefix", arrival_origin_sec=origin)),
                        transition_modes=list(modes), transition_directions=["increase", "decrease"])
            if args.mode == "validate":
                rows, origin = replay_rows(source_rows, args.request_limit,
                    reference["manifest"]["training_prompt_hashes"])
                for policy in ("feedback", "predictive"):
                    append(f"r{repeat}_{variant}_validation_{policy}", variant, full_bank,
                        args.splits[0], "validation", dict(rows=rows, plan_status="planned",
                        recipe=dict(kind="held_out_replay", arrival_origin_sec=origin)),
                        validation_policy=policy)
    return specs


def runtime_hashes(vendor):
    paths = ("vllm/v1/engine/core.py", "vllm/v1/engine/pd_tbt_profiling.py",
             "vllm/v1/engine/pd_sm_allocator.py", "vllm/v1/engine/pd_tbt_profile.py",
             "vllm/v1/core/sched/scheduler.py", "vllm/v1/worker/gpu_model_runner.py",
             "vllm/v1/core/sched/prefill_chunk_scheduler.py",
             "vllm/model_executor/layers/green_context.py",
             "vllm/v1/attention/ops/tiered_residency_provider.py",
             "vllm/v1/core/tiered_required_d2h.py", "vllm/v1/core/virtual_context.py",
             "vllm/v1/attention/backends/triton_attn.py")
    return {p: sha256(vendor / p) for p in paths}


def run_controlled(args, base):
    rows = read_trace(args.trace)
    reference = load_validation_reference(args, base) if args.mode == "validate" else None
    cases = plan_controlled_cases(args, base, rows, reference)
    runnable = [c for c in cases if c["plan_status"] == "planned"]
    print(f"[PROFILE] mode={args.mode} planned={len(cases)} runnable={len(runnable)} "
          f"production_limit_hours={len(runnable)*args.case_seconds/3600:.2f}; "
          "actual cohort/mode/transition coverage is checked, not assumed", flush=True)
    for case in cases:
        print(f"[TARGET] {case['case_id']} {case['plan_status']} "
              f"{case.get('reason', '')} {json.dumps(case.get('target', {}))}", flush=True)
    if args.plan_only:
        return 0
    vendor = Path(base.get("VLLM_VENDOR_DIR", ROOT / "vendor_v8"))
    if not args.dry_run and "self._tbt_capture.recover(" not in (vendor / "vllm/v1/engine/core.py").read_text():
        raise ValueError("deploy profiling observer/hooks first")
    hashes = runtime_hashes(vendor) if not args.dry_run else {}
    if reference and not args.dry_run and hashes != reference["manifest"]["runtime_source_hashes"]:
        raise ValueError("runtime changed since calibration; recollect or explicitly review a new calibration")
    manifest = dict(schema_version=2, experiment_mode=args.mode, splits=[list(p) for p in args.splits],
        variants=args.variants, source_trace=str(args.trace), source_trace_sha256=sha256(args.trace),
        trace_sha256=sha256(args.trace), slo_sha256=sha256(args.slo),
        measurement_configuration=measurement_configuration(base, DEFAULTS),
        runtime_source_hashes=hashes, min_samples=args.min_samples,
        probe_protocol=getattr(args, "probe_protocol", "legacy"),
        min_transition_samples=args.min_transition_samples,
        scope="declared homogeneous cohort probes; not arbitrary context-mixture bounds",
        training_prompt_hashes=sorted({prompt_fingerprint(r) for c in cases for r in c["rows"]})
            if args.mode != "validate" else [],
        cases=[])
    if reference:
        manifest["validation_reference"] = {k: v for k, v in reference.items() if k != "manifest"}
    if not args.dry_run:
        args.output_dir.mkdir(parents=True)
        for name in ("traces", "cases", "captures", "controls"):
            (args.output_dir / name).mkdir()
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            manifest["gpu_inventory"] = gpu.stdout.strip() or gpu.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            manifest["gpu_inventory"] = f"unavailable: {exc}"
        if reference and manifest["gpu_inventory"] != reference["manifest"].get("gpu_inventory"):
            raise ValueError("GPU identity/driver differs from calibration")
    for case in cases:
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in case["rows"])
        case["trace_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if case.get("control"):
            control = dict(case["control"], workload_sha256=case["trace_sha256"],
                           capture_path=case["capture_path"])
            control_payload = json.dumps(control, indent=2) + "\n"
            case["control_sha256"] = hashlib.sha256(control_payload.encode()).hexdigest()
            if not args.dry_run and case["plan_status"] == "planned":
                Path(case["control_path"]).write_text(control_payload)
        if not args.dry_run and case["plan_status"] == "planned":
            Path(case["trace_path"]).write_text(payload, encoding="utf-8")
        record = {k: v for k, v in case.items() if k not in {"env", "rows"}}
        record.update(environment=recorded_env(case["env"]), returncode=None)
        manifest["cases"].append(record)
    manifest_path = args.output_dir / "manifest.json"

    def save():
        if not args.dry_run:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def verify(record, case):
        if args.dry_run:
            return
        try:
            valid = (sha256(args.slo) == manifest["slo_sha256"]
                and sha256(args.trace) == manifest["source_trace_sha256"]
                and sha256(case["trace_path"]) == case["trace_sha256"]
                and (not case.get("control_path") or sha256(case["control_path"]) == case["control_sha256"])
                and runtime_hashes(vendor) == hashes
                and (not reference or sha256(reference["profile_path"]) == reference["profile_sha256"]))
        except OSError:
            valid = False
        if not valid:
            record.update(returncode=2, provenance_error="trace/SLO/runtime/calibration changed")
            save()
            raise ValueError("trace/SLO/runtime/calibration changed; case marked failed, no next server")

    save()
    for case, record in zip(cases, manifest["cases"]):
        if case["plan_status"] != "planned":
            continue
        verify(record, case)
        command = ["bash", str(ROOT / "run_vendor_v8_dynamic_hbm_vs_full.sh")]
        if args.dry_run:
            command.append("--dry-run")
        process = subprocess.Popen(command, env=case["env"], cwd=ROOT, start_new_session=True)
        try:
            status = process.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                print(f"[ERROR] owned wrapper cleanup still pending pgid={process.pid}; stop", file=sys.stderr)
            record["returncode"] = 130
            save()
            return 130
        record["returncode"] = status
        verify(record, case)
        save()
        if status:
            print(f"[ERROR] {case['case_id']} failed status={status}; no next server", file=sys.stderr)
            return status
        if not args.dry_run and case.get("control_path"):
            try:
                report = analyze_case(record, min_samples=args.min_samples,
                                      context_stride=args.context_stride,
                                      manifest_dir=args.output_dir)
                coverage, complete = coverage_for_cases([record], [report],
                    min_transition_samples=args.min_transition_samples)
                record["measurement_status"] = coverage[0]
                record["probe_health"] = report.get("health")
                invalid = report.get("health", {}).get("healthy") is False
            except (OSError, ValueError, KeyError, TypeError) as exc:
                complete = False
                invalid = True
                record["measurement_status"] = dict(status="invalid_probe", reason=str(exc))
            save()
            print("[MEASURED] " + json.dumps(record["measurement_status"]), flush=True)
            if not complete and (invalid or not args.continue_on_unobserved):
                print("[STOP] target/health not established; remaining servers not started", flush=True)
                return analyze(manifest_path, min_samples=args.min_samples,
                               context_stride=args.context_stride)
    if args.dry_run:
        return 0
    return analyze(manifest_path, min_samples=args.min_samples, context_stride=args.context_stride)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--slo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="NEW directory inside this repository; existing outputs are never overwritten")
    parser.add_argument("--splits", type=parse_splits, default=parse_splits("30/10,32/8,34/6,36/4,38/2"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["full_vendor_v8"])
    parser.add_argument("--case-seconds", type=float, default=1800,
                        help="measurement window per server after warmup; cancel at limit")
    parser.add_argument("--arrival-scale", type=float, default=1)
    parser.add_argument("--request-limit", type=int, default=100,
                        help="first N unchanged trace rows in each case; 0=all (time cap still applies)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--context-stride", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="validate launcher routing without GPU or writes")
    parser.add_argument("--mode", choices=("replay", "steady", "transitions", "calibrate", "validate", "liveness"),
                        default="replay", help="legacy replay; declared probes; transitions; both; held-out AB")
    parser.add_argument("--decode-batches", type=positive_ints,
                        help="required for steady/calibrate, e.g. 1,2,4,8; actual measured cohort sizes")
    parser.add_argument("--decode-contexts", type=positive_ints,
                        help="required for steady/calibrate; lower input-length band boundaries")
    parser.add_argument("--context-window-tokens", type=int, default=1024)
    parser.add_argument("--probe-output-tokens", type=int, default=1024)
    parser.add_argument("--kv-capacity-tokens", type=int, default=0,
                        help="measured physical usable KV token ceiling; 0=unknown, do not invent capacity")
    parser.add_argument("--interference-input-min-tokens", type=int, default=32768)
    parser.add_argument("--interference-requests", type=int, default=1)
    parser.add_argument("--interference-delay-sec", type=float, default=30)
    parser.add_argument("--min-transition-samples", type=int, default=2)
    parser.add_argument("--calibration-dir", type=Path, help="ready controlled calibration for --mode validate")
    parser.add_argument("--plan-only", action="store_true", help="print controlled targets; no files/servers")
    parser.add_argument("--probe-protocol", choices=("legacy", "event-gated-v1"), default="legacy",
                        help="Event-gated calibration client; source replay and runtime policies unchanged")
    parser.add_argument("--gate-decode-tokens", type=int, default=8)
    parser.add_argument("--gate-timeout-sec", type=float, default=600)
    parser.add_argument("--no-progress-sec", type=float, default=90,
                        help="Diagnostic stop only if clients pending, model Futures empty and no progress")
    parser.add_argument("--native-interference-max-tokens", type=int, default=49152)
    parser.add_argument("--ring-prime-tokens", type=int, default=0,
                        help="Committed prefix before Decode injection; 0 derives from measured KV capacity")
    parser.add_argument("--transition-base-requests", type=int, default=1)
    parser.add_argument("--transition-extra-requests", type=int, default=2)
    parser.add_argument("--transition-context-tokens", type=int, default=0)
    parser.add_argument("--transition-waves", type=int, default=2)
    parser.add_argument("--transition-burst-output-tokens", type=int, default=256)
    parser.add_argument("--transition-settle-sec", type=float, default=2)
    parser.add_argument("--continue-on-unobserved", action="store_true",
                        help="Continue healthy-cleanup cases to survey missing cells; never bypass readiness")
    args = parser.parse_args(argv)
    args.trace, args.slo, args.output_dir = (p.resolve() for p in (args.trace, args.slo, args.output_dir))
    if (args.repeats < 1 or args.min_samples < 2 or args.context_stride < 1 or args.request_limit < 0
            or not 0 < args.case_seconds < float("inf")
            or not 0 < args.arrival_scale < float("inf")):
        parser.error("positive finite case seconds/arrival scale, repeats>=1, min-samples>=2 required")
    if not args.output_dir.is_relative_to(ROOT) or args.output_dir == ROOT:
        parser.error("output-dir must be a NEW subdirectory inside the repository")
    if args.output_dir.exists():
        parser.error("output-dir already exists; select a new name (no overwrite/resume)")
    base = dict(os.environ)
    if args.probe_protocol == "event-gated-v1":
        if args.mode not in {"liveness", "steady", "transitions", "calibrate"}:
            parser.error("event-gated-v1 is calibration-only, not replay or held-out validation")
        if args.mode in {"transitions", "calibrate"} and len(args.splits) < 2:
            parser.error("Decode-service transition measurement requires at least two concurrent splits")
        if (min(args.gate_decode_tokens, args.transition_base_requests, args.transition_extra_requests,
                args.transition_waves, args.transition_burst_output_tokens) <= 0
                or min(args.ring_prime_tokens, args.transition_context_tokens) < 0
                or args.gate_decode_tokens >= args.probe_output_tokens
                or args.transition_burst_output_tokens > args.probe_output_tokens
                or args.native_interference_max_tokens < args.interference_input_min_tokens
                or not 0 < args.gate_timeout_sec < math.inf
                or not 0 < args.no_progress_sec < math.inf
                or not 0 <= args.transition_settle_sec < math.inf):
            parser.error("invalid event-gated probe lengths, token gates or deadlines")
    elif args.mode == "liveness":
        parser.error("liveness requires --probe-protocol event-gated-v1")
    if args.mode != "replay":
        if args.mode in {"steady", "calibrate"} and (not args.decode_batches or not args.decode_contexts):
            parser.error("steady/calibrate requires --decode-batches and --decode-contexts")
        if len(set(args.variants)) != len(args.variants):
            parser.error("duplicate variants create duplicate case identities")
        if args.mode in {"steady", "calibrate"} and args.probe_output_tokens < args.min_samples + 2:
            parser.error("probe-output-tokens must leave min-samples stable Decode steps after first-token boundaries")
        if (min(args.context_window_tokens, args.probe_output_tokens, args.interference_input_min_tokens,
                args.interference_requests, args.min_transition_samples) <= 0
                or args.kv_capacity_tokens < 0
                or not 0 <= args.interference_delay_sec < args.case_seconds):
            parser.error("invalid controlled probe lengths/counts/capacity/interference delay")
        try:
            return run_controlled(args, base)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
    if args.plan_only or args.calibration_dir or args.decode_batches or args.decode_contexts:
        parser.error("controlled options require --mode steady/transitions/calibrate/validate")
    source_trace = args.trace
    if not args.dry_run:
        for path in (args.trace, args.slo):
            if not path.is_file():
                parser.error(f"missing input: {path}")
        vendor = Path(base.get("VLLM_VENDOR_DIR", ROOT / "vendor_v8"))
        if not (vendor / "vllm/v1/engine/pd_tbt_profiling.py").is_file():
            parser.error("deploy the profiling observer and updated core.py in VLLM_VENDOR_DIR first")
        if "self._tbt_capture.recover(" not in (vendor / "vllm/v1/engine/core.py").read_text():
            parser.error("VLLM_VENDOR_DIR/core.py lacks profiling hooks; deploy both runtime files")
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "captures").mkdir()
        (args.output_dir / "cases").mkdir()
        args.trace = args.output_dir / "selected_trace.jsonl"
        select_trace(source_trace, args.trace, args.request_limit)
    cases = plan_cases(args, base)
    print(f"[PROFILE] {len(cases)} cases, <= {len(cases)*args.case_seconds/3600:.2f} h measurement"
          " + startup/warmup/cleanup; existing SLO unchanged", flush=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {"schema_version": 1, "splits": args.splits,
        "source_trace": str(source_trace), "request_limit": args.request_limit,
        "trace_sha256": sha256(args.trace) if not args.dry_run else "dry-run",
        "slo_sha256": sha256(args.slo) if not args.dry_run else "dry-run",
        "variants": args.variants, "case_seconds": args.case_seconds,
        "scope": "workload-conditioned empirical calibration; restricted banks for per-profile probes",
        "cases": []}
    if not args.dry_run:
        manifest["source_hashes"] = {str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__), ROOT / "tools/analyze_vendor_v8_tbt_calibration.py",
                         ROOT / "run_vendor_v8_dynamic_hbm_vs_full.sh", ROOT / "run_vendor_v8_short_validation.sh")}
        manifest["runtime_source_hashes"] = {str(path): sha256(path)
            for path in (vendor / "vllm/v1/engine/core.py", vendor / "vllm/v1/engine/pd_tbt_profiling.py",
                         vendor / "vllm/v1/engine/pd_sm_allocator.py", vendor / "vllm/v1/core/sched/scheduler.py",
                         vendor / "vllm/v1/attention/ops/tiered_residency_provider.py",
                         vendor / "vllm/v1/core/tiered_required_d2h.py", vendor / "vllm/v1/core/virtual_context.py")}
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            manifest["gpu_inventory"] = gpu.stdout.strip() or gpu.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            manifest["gpu_inventory"] = f"unavailable: {exc}"
    for case in cases:
        env = case.pop("env")
        case.update(trace_sha256=manifest["trace_sha256"], environment=recorded_env(env), returncode=None)
        manifest["cases"].append(case)
        command = ["bash", str(ROOT / "run_vendor_v8_dynamic_hbm_vs_full.sh")]
        if args.dry_run:
            command.append("--dry-run")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[PROFILE] {case['case_id']} bank={env['PD_SM_PROFILES']}", flush=True)
        # Own only this wrapper process group. The existing inner runner owns
        # and cleans its separately launched server process group.
        process = subprocess.Popen(command, env=env, cwd=ROOT, start_new_session=True)
        try:
            status = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                print(f"[ERROR] cleanup still pending for owned wrapper pgid={process.pid}; no next case", file=sys.stderr)
            return 130
        case["returncode"] = status
        if not args.dry_run:
            if sha256(args.slo) != manifest["slo_sha256"] or sha256(args.trace) != manifest["trace_sha256"]:
                raise RuntimeError("trace/SLO changed during profiling; do not use this calibration")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if status:
            print(f"[ERROR] case failed status={status}; stop rather than start another server", file=sys.stderr)
            return status
    if args.dry_run:
        return 0
    return analyze(manifest_path, min_samples=args.min_samples, context_stride=args.context_stride)


if __name__ == "__main__":
    raise SystemExit(main())
