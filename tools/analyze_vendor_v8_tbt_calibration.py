#!/usr/bin/env python3
"""Build workload-conditioned schema-2 TBT profiles; never synthesize coverage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex

from tbt_calibration_plan import coverage_for_cases, steady_matches, transition_direction

CONTRACT = {
    "token_count_source": "choice.token_ids",
    "gap_field": "estimated_per_token_gap_ms",
    "unit": "ms_per_output_token",
    "chunk_gap_normalization": "interarrival_ms_divided_by_emitted_token_count",
    "first_nonempty_chunk": "excluded_as_ttft",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_capture(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if (not records or records[0].get("event") != "start"
            or records[0].get("contract") != "tbt-capture-v1"
            or records[-1].get("event") != "end"):
        raise ValueError("capture has no start/clean-shutdown footer (old code or interrupted server)")
    previous = -math.inf
    for index, record in enumerate(records, 1):
        now = float(record["timestamp"])
        if record.get("sequence") != index or not math.isfinite(now) or now < previous:
            raise ValueError("capture sequence gap/reordered clock: lost records are not calibration")
        previous = now
    return records


def client_id(server_id, known_ids):
    # Completion serving adds cmpl-/prompt index; InputProcessor adds 8 hex.
    candidates = [rid for rid in known_ids if server_id == rid or re.fullmatch(
        rf"cmpl-{re.escape(rid)}-0(?:-[0-9a-f]{{8}})?", server_id)]
    return candidates[0] if len(candidates) == 1 else None


def prefill_signature(record):
    return tuple(sorted((int(p["submit_seq"]), p["mode"], int(p["query_tokens"]))
                        for p in record.get("prefills", [])))


def resolve_case_paths(case, *, manifest_dir=None):
    """Resolve a paired archive, not a basename search across unrelated runs."""
    root = Path(manifest_dir).resolve() if manifest_dir is not None else None

    def resolve_recorded(value):
        path = Path(value)
        if not path.is_absolute() and root is not None:
            resolved = (root / path).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"relative archive path escapes manifest directory: {value}")
            return resolved
        return path.resolve()

    result_dir = resolve_recorded(case["results_dir"])
    capture_path = resolve_recorded(case["capture_path"])
    rebased = False
    # The profiling driver has one canonical cases/captures tree. A copied
    # manifest may retain absolute paths from the original machine. Prefer its
    # local paired subtree when present, even if the old machine is reachable.
    # An incomplete local copy must fail, not borrow the other half remotely.
    case_root = result_dir.parent if result_dir.name == "client" else result_dir
    archive_root = capture_path.parent.parent
    canonical_pair = (
        capture_path.parent.name == "captures"
        and capture_path.name == f"{case['case_id']}.jsonl"
        and case_root.parent == archive_root / "cases"
        and case_root.name.startswith(f"{case['case_id']}_")
        and case_root.name.endswith("_results")
    )
    if root is not None and root != archive_root and canonical_pair:
        local_result = root / result_dir.relative_to(archive_root)
        local_capture = root / capture_path.relative_to(archive_root)
        if (root / "cases" / case_root.name).exists() or local_capture.exists():
            result_dir, capture_path = local_result, local_capture
            rebased = True
    if not result_dir.is_dir():
        raise FileNotFoundError(f"missing case results directory: {result_dir}")
    if not capture_path.is_file():
        raise FileNotFoundError(f"missing paired case capture: {capture_path}")
    return result_dir, capture_path, rebased


def resolve_client_dir(result_dir):
    """Accept flat or runner-owned client/ layout without mixing partial files."""
    required = ("run_metadata.json", "client_token_timeline.csv")
    candidates = [path for path in (result_dir, result_dir / "client")
                  if any((path / name).exists() for name in required)]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous client bundle in {result_dir} and its client/ directory")
    if not candidates:
        raise FileNotFoundError(f"missing client bundle in {result_dir} or its client/ directory")
    client_dir = candidates[0]
    missing = [name for name in required if not (client_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete client bundle in {client_dir}: missing {missing}")
    return client_dir


def probe_diagnostics(events, client_dir, start, end):
    """Separate completed-client idle from unfinished, Future-free tail gaps."""
    request_path = client_dir / "request_results.csv"
    requests = read_csv(request_path) if request_path.is_file() else []
    known = {r["request_id"] for r in requests}
    pending, progress = {}, {}
    commits, overlap = Counter(), Counter()
    last = start
    for event in events:
        kind = event["event"]
        if kind == "submit":
            pending[int(event["submit_seq"])] = event
            if event["phase"] == "decode":
                signature = prefill_signature(event)
                overlap["ring" if any(p[1] == "ring" for p in signature)
                        else "native" if signature else "none"] += 1
        elif kind == "recover":
            submit = pending.pop(int(event["submit_seq"]), None)
            if submit and submit["phase"] == "prefill":
                for row in submit["rows"]:
                    rid = client_id(row["request_id"], known)
                    if rid:
                        progress[rid] = max(progress.get(rid, 0), int(row["context_tokens"]))
        elif kind == "transition_commit":
            debug = event["debug"]
            commits[f"{debug.get('pd_sm_transition_source_profile')}->{debug.get('pd_sm_transition_committed_profile')}"] += 1
        if kind in {"submit", "recover", "prefill_boundary", "prefill_resume", "transition_commit"}:
            last = event["timestamp"]
    incomplete = [r for r in requests if r.get("status") in {"censored_time_limit", "failed", "submitted"}]
    return dict(request_status_counts=dict(Counter(r.get("status") for r in requests)),
                last_model_progress_offset_sec=last-start,
                tail_without_model_events_sec=max(0., end-last),
                unfinished_requests=[dict(request_id=r["request_id"], status=r["status"],
                    input_tokens=int(r["input_tokens"]), recovered_prefill_tokens=progress.get(r["request_id"], 0),
                    output_tokens=int(r.get("output_tokens") or 0)) for r in incomplete],
                outstanding_submit_seqs=sorted(pending),
                unfinished_future_free_tail=bool(incomplete and not pending),
                observed_decode_submissions_by_prefill_mode=dict(overlap),
                recorded_transition_commits=dict(commits))


def analyze_case(case, *, min_samples=32, context_stride=1024, manifest_dir=None):
    result_root, capture_path, rebased = resolve_case_paths(case, manifest_dir=manifest_dir)
    result_dir = resolve_client_dir(result_root)
    metadata = json.loads((result_dir / "run_metadata.json").read_text())
    if (metadata.get("tbt_measurement") != "vllm choice.token_ids"
            or metadata.get("completion_response_contract", {}).get("return_token_ids") is not True):
        raise ValueError("exact choice.token_ids client contract missing")
    if int(metadata.get("summary", {}).get("failure_count", 0)):
        raise ValueError("request failures: failed cases cannot calibrate deployment")
    if metadata.get("workload_sha256") != case["trace_sha256"]:
        raise ValueError("replayed workload hash differs from capture manifest")
    health = metadata.get("tbt_probe_health")
    if case.get("control_path"):
        if (metadata.get("submission_mode") != "controlled_event_gated_probe"
                or metadata.get("tbt_probe_control_sha256") != case.get("control_sha256")
                or not isinstance(health, dict) or not isinstance(health.get("healthy"), bool)):
            raise ValueError("event-gated client control/health contract missing or changed")
    events = read_capture(capture_path)
    start = float(metadata["trace_start_monotonic_ts"])
    end = float(metadata["observation_end_monotonic_ts"])
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("invalid production observation window")
    # Warmup has much smaller context and is not production transition coverage.
    events = [event for event in events if start <= event["timestamp"] <= end]
    clients = read_csv(result_dir / "client_token_timeline.csv")
    known = {r["request_id"] for r in clients}
    starts, previous, qualified = {}, {}, {}
    last_change = 0
    rejected = Counter()
    decode_recoveries = []
    transition_commits, transitions = [], []
    transition_requests = {}
    steps = []
    # Every sample must cross one stable, same-cohort Decode service interval.
    for event in events:
        kind = event["event"]
        if kind in {"prefill_boundary", "prefill_resume"}:
            last_change = event["sequence"]
        elif kind == "transition_request":
            # Retargeting/Prefill drain must not replace the source-mode latch.
            transition_requests.setdefault(float(event["requested_at"]), event)
            last_change = event["sequence"]
        elif kind == "transition_commit":
            transition_commits.append(event)
            last_change = event["sequence"]
        elif kind == "submit":
            starts[int(event["submit_seq"])] = event
            if event["phase"] != "decode":
                last_change = event["sequence"]
                continue
            # First Decode submission, not first Prefill/any-phase resumption.
            for commit in transition_commits:
                debug = commit["debug"]
                if commit.get("used") or (
                    debug.get("pd_sm_transition_committed_profile") != event["profile_id"]
                    or int(debug.get("pd_sm_transition_committed_epoch", -1)) != event["epoch"]
                ):
                    continue
                commit["used"] = True
                requested = float(debug.get("pd_sm_transition_requested_ts", 0))
                source = debug.get("pd_sm_transition_source_profile", "")
                request = transition_requests.get(requested)
                # Direct/solo decisions without a request record are NOT timings.
                if request is None or requested > commit["timestamp"]:
                    continue
                old = [r for r in decode_recoveries if r["profile_id"] == source
                       and requested <= r["timestamp"] <= event["timestamp"]]
                if old:
                    last_service = old[-1]["timestamp"]
                elif source.startswith("d0_"):
                    last_service = requested  # no source Decode lane
                else:
                    rejected["transition_without_source_service"] += 1
                    continue
                transitions.append({"source": source, "target": event["profile_id"],
                    "direction": transition_direction(source, event["profile_id"]),
                    "decode_requests": len(event["rows"]),
                    "decode_context_max": max((int(r["context_tokens"]) for r in event["rows"]), default=0),
                    "epoch": event["epoch"],
                    "source_prefill_mode": (
                        "ring" if any(p[1] == "ring" for p in prefill_signature(request))
                        else "native" if prefill_signature(request) else "none"),
                    "lead_ms": (event["timestamp"] - requested) * 1000,
                    "cutover_ms": (event["timestamp"] - last_service) * 1000,
                    "all_fence_ms": debug.get("pd_sm_transition_all_fence_ms"),
                    "boundary_commit": debug.get("pd_sm_transition_layerwise_boundary_commit"),
                    "source_had_decode_service": bool(old)})
        elif kind == "recover":
            submit = starts.get(int(event["submit_seq"]))
            if submit is None:
                # Resumed Prefill uses its original submit identity. A trace
                # without that original identity is not attributable.
                rejected["unmatched_recovery"] += 1
                last_change = event["sequence"]
                continue
            if submit["phase"] != "decode":
                last_change = event["sequence"]
                continue
            rows = submit["rows"]
            counts = {r["request_id"]: int(r["count"]) for r in event["tokens"]}
            cohort = tuple(sorted(r["request_id"] for r in rows))
            if len(counts) != len(rows) or any(counts.get(rid) != 1 for rid in cohort):
                rejected["cancelled_or_nonuniform_output"] += 1
                last_change = event["sequence"]
                continue
            if any(int(r["scheduled_tokens"]) != 1 or int(r["context_tokens"]) <= 0
                   or int(r["output_start"]) < 1 for r in rows):
                rejected["not_uniform_live_decode"] += 1
                continue
            decode_recoveries.append({"timestamp": event["timestamp"],
                                      "profile_id": submit["profile_id"]})
            signature = prefill_signature(event)
            mode = ("ring" if any(p[1] == "ring" for p in signature)
                    else "native" if signature else "none")
            state = (submit["profile_id"], submit["epoch"], cohort, signature)
            old = previous.get(cohort)
            current = dict(state=state, event=event, submit=submit)
            previous[cohort] = current
            if (old is None or old["state"] != state
                    or prefill_signature(submit) != signature
                    or last_change > old["event"]["sequence"]):
                rejected["changed_cohort_profile_or_prefill"] += 1
                continue
            old_rows = {r["request_id"]: r for r in old["submit"]["rows"]}
            if any(r["output_start"] != old_rows[r["request_id"]]["output_start"] + 1 for r in rows):
                rejected["nonconsecutive_output"] += 1
                continue
            context = max(int(r["context_tokens"]) for r in rows)
            if case.get("kind") == "steady" and not steady_matches(
                    case["target"], submit["profile_id"], mode, rows,
                    lambda rid: client_id(rid, known)):
                rejected["declared_target_not_observed"] += len(rows)
                continue
            # Floor rather than round up: never advertise an unmeasured larger
            # context merely because it shares a histogram bin.
            bucket = max(1, context // context_stride * context_stride)
            key = (submit["profile_id"], mode, len(rows), bucket)
            query_tokens = sum(p[2] for p in signature)
            steps.append((event["timestamp"] - old["event"]["timestamp"]) * 1000)
            for row in rows:
                rid = client_id(row["request_id"], known)
                if rid is not None:
                    qualified[(rid, int(row["output_start"]) + 1)] = (
                        key, query_tokens, context,
                        sum(int(r["context_tokens"]) for r in rows), int(submit["submit_seq"]))

    # SSE may bundle several IDs. All IDs in a chunk must have the same qualified
    # state; normalize by exact ID count, never bytes/words or nominal output len.
    ordinal = defaultdict(int)
    groups, details = defaultdict(list), defaultdict(list)
    slo_samples, slo_attained = Counter(), Counter()
    measured_steps = defaultdict(set)
    for row in clients:
        rid, count = row["request_id"], int(row["chunk_tokens"])
        if count <= 0:
            raise ValueError("non-positive exact client chunk count")
        start = ordinal[rid]
        ordinal[rid] += count
        if start == 0:
            continue
        matches = [qualified.get((rid, n)) for n in range(start + 1, start + count + 1)]
        if any(item is None for item in matches) or len({m[:2] for m in matches if m}) != 1:
            rejected["client_gap_not_attributable"] += count
            continue
        gap = float(row["estimated_per_token_gap_ms"])
        calculated = (float(row["arrival_monotonic_ts"]) - float(row["gap_start_monotonic_ts"])) * 1000 / count
        if not math.isfinite(gap) or gap <= 0 or not math.isclose(gap, calculated, rel_tol=1e-5, abs_tol=0.001):
            raise ValueError("client gap does not satisfy exact-token normalization")
        key, query, context, context_sum, _ = matches[0]
        groups[(key, query)].extend([gap] * count)
        slo_text = row.get("tbt_slo_ms")
        if slo_text not in (None, ""):
            slo = float(slo_text)
            if math.isfinite(slo) and slo > 0:
                slo_samples[(key, query)] += count
                slo_attained[(key, query)] += count * (gap <= slo)
        measured_steps[(key, query)].update(m[4] for m in matches)
        details[key].append((context, context_sum))
    points = []
    for (key, query), gaps in groups.items():
        if len(gaps) < min_samples:
            rejected["too_few_token_samples"] += len(gaps)
            continue
        step_count = len(measured_steps[(key, query)])
        if step_count < case.get("target", {}).get("min_decode_steps", 0):
            rejected["too_few_independent_decode_steps"] += len(gaps)
            continue
        profile, mode, count, context = key
        points.append({"profile_id": profile, "prefill_mode": mode,
            "variant": case.get("variant", "unspecified"),
            "max_decode_requests": count, "max_decode_context_tokens": context,
            "p95_per_token_tbt_ms": percentile(gaps, .95), "sample_count": len(gaps),
            "p50_per_token_tbt_ms": percentile(gaps, .5),
            "p99_per_token_tbt_ms": percentile(gaps, .99), "max_per_token_tbt_ms": max(gaps),
            "tbt_slo_covered_samples": slo_samples[(key, query)],
            "tbt_slo_attainment": (slo_attained[(key, query)] / slo_samples[(key, query)]
                                   if slo_samples[(key, query)] else None),
            "decode_step_count": step_count,
            "prefill_query_tokens": query,
            "observed_context_max": max(d[0] for d in details[key]),
            "observed_context_sum_min": min(d[1] for d in details[key]),
            "observed_context_sum_max": max(d[1] for d in details[key]),
            "case_id": case["case_id"]})
    if health and not health["healthy"]:
        rejected["invalid_probe_health_samples"] += sum(p["sample_count"] for p in points)
        points, transitions = [], []
    return {"case_id": case["case_id"], "points": points,
            "health": health or {},
            "diagnostics": probe_diagnostics(events, result_dir, start, end),
            "transitions": transitions, "min_step_ms": min(steps) if steps else None,
            "rejected": dict(rejected), "qualified_tokens": sum(len(v) for v in groups.values()),
            "input_paths": {"recorded_results_dir": str(case["results_dir"]),
                            "recorded_capture_path": str(case["capture_path"]),
                            "results_dir": str(result_root), "client_dir": str(result_dir),
                            "capture_path": str(capture_path), "archive_rebased": rebased}}


def analyze_validation(manifest, reports, errors, output):
    """Held-out AB: predictions are diagnostics; all client gaps count for SLO."""
    import runpy
    reference = manifest["validation_reference"]
    path = Path(reference["profile_path"])
    if sha256(path) != reference["profile_sha256"]:
        raise ValueError("frozen calibration changed during validation")
    profile_type = runpy.run_path(str(Path(__file__).resolve().parents[1] /
        "vendor_v8/vllm/v1/engine/pd_tbt_profile.py"))["TBTOverlapProfile"]
    model = profile_type.from_json_file(path)
    comparisons, cases = [], []
    planned = {c["case_id"]: c for c in manifest["cases"]}
    for report in reports:
        case = planned[report["case_id"]]
        for point in report["points"]:
            prediction = model.predict(profile_id=point["profile_id"],
                decode_context_tokens=point["observed_context_max"],
                decode_requests=point["max_decode_requests"], prefill_mode=point["prefill_mode"])
            predicted = prediction.p95_per_token_tbt_ms if prediction else None
            comparisons.append(dict(case_id=case["case_id"], variant=case["variant"],
                policy=case["validation_policy"], profile_id=point["profile_id"],
                mode=point["prefill_mode"], count=point["max_decode_requests"],
                context=point["observed_context_max"], samples=point["sample_count"],
                predicted_p95_ms=predicted, observed_p95_ms=point["p95_per_token_tbt_ms"],
                underestimated=point["p95_per_token_tbt_ms"] > predicted if predicted else None))
        client = Path(report["input_paths"]["client_dir"])
        meta = json.loads((client / "run_metadata.json").read_text())
        token_rows = read_csv(client / "client_token_timeline.csv")
        seen, total, attained, slo_count = set(), 0, 0, 0
        for row in token_rows:
            rid = row["request_id"]
            if rid not in seen:
                seen.add(rid)
                continue
            count = int(row["chunk_tokens"])
            total += count
            if row.get("tbt_slo_ms", "") != "":
                slo = float(row["tbt_slo_ms"])
                if math.isfinite(slo) and slo > 0:
                    slo_count += count
                    attained += count * (float(row["estimated_per_token_gap_ms"]) <= slo)
        duration = float(meta["observation_end_monotonic_ts"]) - float(meta["trace_start_monotonic_ts"])
        summary = meta.get("summary", {})
        request_path = client / "request_results.csv"
        request_rows = read_csv(request_path) if request_path.is_file() else []
        prefilled_tokens = sum(int(r["input_tokens"]) for r in request_rows if r.get("ttft_ms", "") != "")
        output_tokens = summary.get("observed_output_tokens")
        cases.append(dict(case_id=case["case_id"], variant=case["variant"], policy=case["validation_policy"],
            trace_sha256=case["trace_sha256"], observation_seconds=duration,
            token_gaps=total, tbt_slo_covered_tokens=slo_count,
            tbt_slo_attainment=attained / slo_count if slo_count else None,
            observed_output_tokens=output_tokens,
            observed_output_token_s=float(output_tokens)/duration if output_tokens is not None else None,
            completed_prefill_input_token_s=prefilled_tokens/duration if request_rows else None,
            ttft_slo_attainment=summary.get("ttft_slo_attainment"),
            ttft_p50_ms=summary.get("ttft_p50_ms"),
            summary=summary))
    result = dict(validation_complete=not errors and len(reports) == len(manifest["cases"]),
        ready=False, training_performed=False, calibration_id=reference["calibration_id"],
        errors=errors, cases=cases, predictions=comparisons,
        uncovered_points=sum(p["predicted_p95_ms"] is None for p in comparisons),
        note="All attributable stable gaps diagnose predictions; all client gaps, including transitions, count for SLO. No hard real-time guarantee.")
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    if cases:
        with (output / "validation_summary.csv").open("w", newline="") as handle:
            columns = [k for k in cases[0] if k != "summary"]
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cases)
    if comparisons:
        with (output / "prediction_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    print(json.dumps({k: result[k] for k in ("validation_complete", "training_performed", "uncovered_points", "errors")}, indent=2))
    return 0 if result["validation_complete"] else 3


def analyze(manifest_path, *, min_samples=32, context_stride=1024,
            transition_margin=1.25, min_step_factor=.5, output_dir=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    output = Path(output_dir).resolve() if output_dir is not None else root / "calibration"
    # Never leave a stale successful env after a failed re-analysis.
    output.mkdir(parents=True, exist_ok=output_dir is None)
    env_path = output / "predictive.env"
    if env_path.exists():
        raise ValueError("calibration/predictive.env already exists; use a new output directory/run")
    reports, errors = [], []
    for case in manifest["cases"]:
        if case.get("plan_status", "planned") != "planned":
            continue
        try:
            if case.get("returncode") != 0:
                raise ValueError(f"case failed or not executed: status={case.get('returncode')}")
            reports.append(analyze_case(case, min_samples=min_samples,
                                        context_stride=context_stride, manifest_dir=root))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"case_id": case["case_id"], "error": str(exc)})
    experiment_mode = manifest.get("experiment_mode", "replay")
    if experiment_mode == "validate":
        return analyze_validation(manifest, reports, errors, output)
    steady_ids = {c["case_id"] for c in manifest["cases"] if c["kind"] == "steady"}
    points = [point for report in reports for point in report["points"]
              if experiment_mode != "calibrate" or report["case_id"] in steady_ids]
    transitions = [t for report in reports for t in report["transitions"]]
    minima = [r["min_step_ms"] for r in reports if r["min_step_ms"] is not None and r["min_step_ms"] > 0]
    modes = (("native", "ring") if "full_vendor_v8" in manifest.get("variants", ["full_vendor_v8"])
             else ("native",))
    expected = {(f"d{d}_p{p}", m) for d, p in manifest["splits"] for m in modes}
    expected.add(("d40_p0", "none"))
    seen = {(p["profile_id"], p["prefill_mode"]) for p in points}
    missing = sorted(expected - seen)
    feedback_ids = {c["case_id"] for c in manifest["cases"] if c["kind"] in {"feedback", "transitions"}}
    feedback_transitions = [t for r in reports if r["case_id"] in feedback_ids for t in r["transitions"]]
    transition_modes = {t["source_prefill_mode"] for t in feedback_transitions}
    required_transition_modes = {mode for _, mode in expected if mode != "none"}
    missing_transition_modes = sorted(required_transition_modes - transition_modes)
    invalid_health = [r["case_id"] for r in reports if r.get("health", {}).get("healthy") is False]
    ready = bool(not errors and not invalid_health and not missing and minima and feedback_transitions
                 and not missing_transition_modes
                 and any(t["source_had_decode_service"] for t in feedback_transitions))
    declared, declared_complete = [], False
    stage_ready = False
    if experiment_mode != "replay":
        declared, declared_complete = coverage_for_cases(manifest["cases"], reports,
            min_transition_samples=manifest["min_transition_samples"])
        stage_ready = declared_complete and not errors and not invalid_health
        ready = ready and stage_ready and experiment_mode == "calibrate"
    profile = {"schema_version": 2, "measurement_contract": CONTRACT, "points": points,
        "scope": "observed workload/cohort envelope, not arbitrary-context-mixture guarantee",
        "manifest": str(manifest_path), "trace_sha256": manifest["trace_sha256"],
        "slo_sha256": manifest["slo_sha256"],
        "coverage_note": "No extrapolation; counts/contexts not observed remain uncalibrated. Duplicate bins retain maximum p95 at runtime."}
    (output / "tbt_overlap_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    report = {"ready": ready, "errors": errors, "missing_profile_modes": missing,
              "invalid_probe_health": invalid_health,
              "profile_sha256": sha256(output / "tbt_overlap_profile.json"),
              "experiment_mode": experiment_mode, "stage_ready": stage_ready,
              "declared_coverage_complete": declared_complete,
              "declared_coverage": declared,
              "cases": reports, "transition_sample_count": len(transitions),
              "feedback_transition_sample_count": len(feedback_transitions),
              "missing_transition_source_modes": missing_transition_modes,
              "empirical_not_hard_bounds": True,
              "transition_margin": transition_margin, "min_step_factor": min_step_factor}
    report["observed_coverage"] = [
        {"profile_id": profile_id, "prefill_mode": mode,
         "actual_decode_counts": sorted({p["max_decode_requests"] for p in points
             if (p["profile_id"], p["prefill_mode"]) == (profile_id, mode)}),
         "context_buckets": sorted({p["max_decode_context_tokens"] for p in points
             if (p["profile_id"], p["prefill_mode"]) == (profile_id, mode)})}
        for profile_id, mode in sorted(seen)]
    if ready:
        lead = max(t["lead_ms"] for t in transitions) * transition_margin
        cutover = max(t["cutover_ms"] for t in transitions) * transition_margin
        minimum = min(minima) * min_step_factor
        report["calibration_ms"] = {"lead": lead, "cutover": cutover, "min_step": minimum}
    (output / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "tbt_points.csv").open("w", newline="") as handle:
        if points:
            writer = csv.DictWriter(handle, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
    with (output / "transition_samples.csv").open("w", newline="") as handle:
        columns = (list(transitions[0]) if transitions else
                   ["source", "target", "source_prefill_mode", "lead_ms", "cutover_ms"])
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(transitions)
    # Publish the deployment environment only after every required report has
    # been written. An interrupted/failed report write must not enable serving.
    if ready:
        env_path.write_text(
            "# Empirical calibration for this manifest/configuration only.\n"
            f"export TBT_OVERLAP_PROFILE={shlex.quote(str(output / 'tbt_overlap_profile.json'))}\n"
            "export PD_DYNAMIC_SM_POLICY=tbt_slo_predictive\n"
            f"export VLLM_PD_TBT_PREDICTIVE_LEAD_MS={lead:.9g}\n"
            f"export VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS={cutover:.9g}\n"
            f"export VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS={minimum:.9g}\n")
    print(json.dumps({"ready": ready, "profile_points": len(points),
                      "stage_ready": stage_ready,
                      "missing_profile_modes": missing,
                      "missing_transition_source_modes": missing_transition_modes,
                      "errors": errors,
                      "output": str(output)}, indent=2))
    return 0 if ready or (experiment_mode in {"steady", "transitions", "liveness"} and stage_ready) else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="NEW directory for reanalysis; default: manifest directory/calibration")
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--context-stride", type=int, default=1024)
    args = parser.parse_args()
    if args.min_samples < 2 or args.context_stride < 1:
        parser.error("min-samples >= 2 and context-stride >= 1 required")
    return analyze(args.manifest, min_samples=args.min_samples,
                   context_stride=args.context_stride, output_dir=args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
