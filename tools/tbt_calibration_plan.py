"""CPU-only experiment recipes and measured-coverage contracts (not policies)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def positive_ints(value):
    values = [int(part) for part in value.split(",")]
    if not values or min(values) <= 0 or len(set(values)) != len(values):
        raise ValueError("expected distinct positive comma-separated integers")
    return values


def prompt_fingerprint(row):
    return hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()


def read_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty trace")
    ids, last = set(), -1.
    for row in rows:
        offset = float(row["base_scheduled_offset_s"])
        if (not isinstance(row["prompt"], str) or not row["prompt"]
                or not isinstance(row.get("request_id"), str) or not row["request_id"]
                or not isinstance(row.get("dataset"), str) or not row["dataset"]
                or int(row["input_tokens"]) <= 0 or int(row["max_tokens"]) <= 0
                or int(row["min_tokens"]) != int(row["max_tokens"])
                or not math.isfinite(offset) or offset < max(0, last)
                or row["request_id"] in ids):
            raise ValueError("invalid trace prompt/length/offset/request identity")
        ids.add(row["request_id"])
        last = offset
    return rows


def replay_rows(rows, limit, excluded=()):
    """Keep source order and relative arrival intervals; select disjoint prompts."""
    excluded = set(excluded)
    selected = [row.copy() for row in rows if prompt_fingerprint(row) not in excluded]
    if limit:
        if len(selected) < limit:
            raise ValueError(f"only {len(selected)} disjoint replay rows, need {limit}")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no disjoint replay rows")
    origin = float(selected[0]["base_scheduled_offset_s"])
    for row in selected:
        row["base_scheduled_offset_s"] = float(row["base_scheduled_offset_s"]) - origin
    return selected, origin


def steady_recipe(rows, *, count, context, window, output_tokens, mode,
                  max_seqs, max_model_len, kv_capacity_tokens,
                  interference_tokens, interference_count, interference_delay):
    """Request a cohort; admission and actual observed mode remain authoritative."""
    target = dict(decode_requests=count, context_min=context,
                  context_max=context + window - 1 + output_tokens,
                  prefill_mode=mode)
    result = dict(target=target, rows=[], plan_status="planned", reason="")
    if count > max_seqs or context + output_tokens > max_model_len:
        return dict(result, plan_status="infeasible", reason="sequence_or_model_limit")
    # Prefix caching is disabled by the reused launcher. This is only a physical
    # lower-bound impossibility proof, not an admission/reserve feasibility claim.
    if kv_capacity_tokens and count * context > kv_capacity_tokens:
        return dict(result, plan_status="infeasible", reason="decode_hotset_exceeds_declared_physical_kv")
    candidates, fingerprints = [], set()
    for row in rows:
        length = int(row["input_tokens"])
        fingerprint = prompt_fingerprint(row)
        if (context <= length < context + window
                and length + output_tokens <= max_model_len and fingerprint not in fingerprints):
            candidates.append(row)
            fingerprints.add(fingerprint)
        if len(candidates) == count:
            break
    if len(candidates) != count:
        return dict(result, plan_status="input_unavailable", reason="not_enough_distinct_prompts_in_length_band")
    if kv_capacity_tokens and sum(int(r["input_tokens"]) for r in candidates) > kv_capacity_tokens:
        return dict(result, plan_status="infeasible", reason="selected_decode_hotset_exceeds_declared_physical_kv")
    background = []
    if mode != "none":
        for row in sorted(rows, key=lambda r: -int(r["input_tokens"])):
            fingerprint = prompt_fingerprint(row)
            if (interference_tokens <= int(row["input_tokens"]) < max_model_len
                    and fingerprint not in fingerprints):
                background.append(row)
                fingerprints.add(fingerprint)
            if len(background) == interference_count:
                break
        if len(background) != interference_count:
            return dict(result, plan_status="input_unavailable", reason="not_enough_interference_prompts")
    probes = []
    for index, row in enumerate(candidates + background):
        probe = row.copy()
        is_decode = index < count
        probe.update(request_id=f"probe_{index}", tenant_id=f"probe_{index}",
                     event_index=index, source_id=str(row.get("source_id", row["request_id"])),
                     role="tbt_decode_probe" if is_decode else "tbt_prefill_interference",
                     base_scheduled_offset_s=0. if is_decode else interference_delay,
                     min_tokens=output_tokens if is_decode else 1,
                     max_tokens=output_tokens if is_decode else 1)
        probes.append(probe)
    target["request_ids"] = [r["request_id"] for r in probes[:count]]
    return dict(result, rows=probes, recipe={
        "kind": "controlled_existing_prompts", "source_prompts_unchanged": True,
        "source_prompt_hashes": [prompt_fingerprint(r) for r in probes],
        "source_input_lengths": [int(r["input_tokens"]) for r in probes],
        "output_tokens": output_tokens, "interference_delay_sec": interference_delay,
        "physical_kv_capacity_tokens": kv_capacity_tokens or None,
        "capacity_note": "lower-bound exclusion only; allocator and actual cohort remain authoritative"})


def transition_direction(source, target):
    old, new = int(source.split("_")[0][1:]), int(target.split("_")[0][1:])
    return "increase" if new > old else "decrease" if new < old else "unchanged"


def event_probe_recipe(rows, *, args, config, count, context, mode, transitions=False):
    """Distinct native and pressure recipes; observation still decides mode.

    Native: live Decode tokens -> bounded background Prefill.
    Ring: committed long-prefix progress -> Decode probes, whose arrival can
    cause a pressure/offload boundary. No synthetic token IDs or force-victim.
    """
    extra = getattr(args, "transition_extra_requests", 1) if transitions else 0
    selected = steady_recipe(rows, count=count+extra, context=context,
        window=args.context_window_tokens, output_tokens=args.probe_output_tokens,
        mode="none", max_seqs=int(config["VLLM_MAX_NUM_SEQS"]),
        max_model_len=int(config["VLLM_MAX_MODEL_LEN"]),
        kv_capacity_tokens=args.kv_capacity_tokens,
        interference_tokens=args.interference_input_min_tokens,
        interference_count=1, interference_delay=0)
    if selected["plan_status"] != "planned":
        selected["target"].update(decode_requests=count, prefill_mode=mode)
        return selected
    cohort, burst = selected["rows"][:count], selected["rows"][count:]
    target = dict(selected["target"], decode_requests=count, prefill_mode=mode,
                  request_ids=[r["request_id"] for r in cohort])
    gates = {}
    control = dict(schema_version=1, gate_timeout_sec=args.gate_timeout_sec,
                   no_progress_sec=args.no_progress_sec, poll_sec=.1,
                   require_completion=mode == "none", gates=gates)
    result = dict(plan_status="planned", reason="", target=target, control=control)
    excluded = {prompt_fingerprint(r) for r in selected["rows"]}
    backgrounds = []
    prime = 0
    if mode != "none":
        if mode == "ring":
            prime = args.ring_prime_tokens
            if not prime:
                if not args.kv_capacity_tokens:
                    return dict(result, rows=[], plan_status="input_unavailable",
                                reason="ring_needs_measured_kv_capacity_or_explicit_prime_tokens")
                quantum = int(config["PREFILL_CHUNK_TOKENS"])
                headroom = sum(int(r["input_tokens"]) for r in cohort) + quantum
                prime = max(quantum, (args.kv_capacity_tokens-headroom)//quantum*quantum)
        candidates = [r for r in rows if prompt_fingerprint(r) not in excluded
                      and args.interference_input_min_tokens <= int(r["input_tokens"])
                      < int(config["VLLM_MAX_MODEL_LEN"])
                      and (int(r["input_tokens"]) <= args.native_interference_max_tokens
                           if mode == "native" else int(r["input_tokens"]) > prime)]
        candidates.sort(key=lambda r: int(r["input_tokens"]), reverse=mode == "ring")
        for row in candidates:
            fingerprint = prompt_fingerprint(row)
            if fingerprint in excluded:
                continue
            excluded.add(fingerprint)
            index = len(backgrounds)
            backgrounds.append(dict(row, request_id=f"background_{index}",
                tenant_id=f"background_{index}", source_id=str(row["request_id"]),
                role="tbt_prefill_interference", min_tokens=1, max_tokens=1,
                base_scheduled_offset_s=0.))
            if len(backgrounds) == args.interference_requests:
                break
        if len(backgrounds) != args.interference_requests:
            return dict(result, rows=[], plan_status="input_unavailable",
                        reason=f"no_distinct_{mode}_interference_in_declared_length_range")
    if mode == "ring":
        # Prime one long prefix before inserting the cohort, rather than
        # allowing a short Decode to finish before any Ring pressure exists.
        ordered = backgrounds + cohort
        for row in cohort:
            gates[row["request_id"]] = dict(prefill_from=backgrounds[0]["request_id"],
                                            prefill_tokens=prime)
    else:
        ordered = cohort + backgrounds
        for row in backgrounds:
            gates[row["request_id"]] = dict(tokens_from=target["request_ids"],
                                            min_tokens=args.gate_decode_tokens)
    if transitions:
        previous = []
        for wave in range(args.transition_waves):
            current = []
            for index, source in enumerate(burst):
                rid = f"burst_{wave}_{index}"
                row = dict(source, request_id=rid, tenant_id=rid,
                           role="tbt_decode_load_burst",
                           min_tokens=args.transition_burst_output_tokens,
                           max_tokens=args.transition_burst_output_tokens)
                ordered.append(row)
                current.append(rid)
                gates[rid] = dict(tokens_from=target["request_ids"],
                    min_tokens=args.gate_decode_tokens, prefill_mode=mode,
                    complete_from=previous, settle_sec=args.transition_settle_sec)
            previous = current
    for index, row in enumerate(ordered):
        row["event_index"] = index
        row["base_scheduled_offset_s"] = 0.
    control["request_ids"] = [r["request_id"] for r in ordered]
    recipe = dict(kind="event_gated_existing_prompts", source_prompts_unchanged=True,
                  source_prompt_hashes=[prompt_fingerprint(r) for r in ordered],
                  source_input_lengths=[r["input_tokens"] for r in ordered],
                  output_tokens=args.probe_output_tokens, ring_prime_tokens=prime,
                  native_interference_max_tokens=args.native_interference_max_tokens,
                  physical_kv_capacity_tokens=args.kv_capacity_tokens or None,
                  mode_note="Pressure priming is an experiment recipe, NOT proof of Ring/admission.")
    return dict(result, rows=ordered, recipe=recipe)


def liveness_recipe(rows, *, args, config):
    candidates = [r for r in rows if int(r["input_tokens"]) >= args.interference_input_min_tokens
                  and int(r["input_tokens"]) < int(config["VLLM_MAX_MODEL_LEN"])]
    if not candidates:
        return dict(rows=[], plan_status="input_unavailable", reason="no_long_preflight_prompt")
    source = max(candidates, key=lambda r: int(r["input_tokens"]))
    row = dict(source, request_id="liveness_tail", tenant_id="liveness_tail", event_index=0,
               source_id=str(source["request_id"]), role="tbt_liveness_preflight",
               min_tokens=1, max_tokens=1, base_scheduled_offset_s=0.)
    return dict(rows=[row], plan_status="planned", reason="",
        recipe=dict(kind="completion_required_long_tail", source_prompts_unchanged=True,
                    source_input_lengths=[row["input_tokens"]]),
        control=dict(schema_version=1, request_ids=[row["request_id"]], gates={},
                     gate_timeout_sec=args.gate_timeout_sec, no_progress_sec=args.no_progress_sec,
                     poll_sec=.1, require_completion=True))


def steady_matches(target, profile, mode, rows, known_id):
    """Do not credit another cohort, a smaller batch or context-max alone."""
    return (profile == target["profile_id"] and mode == target["prefill_mode"]
            and len(rows) == target["decode_requests"]
            and {known_id(r["request_id"]) for r in rows} == set(target["request_ids"])
            and all(target["context_min"] <= int(r["context_tokens"]) <= target["context_max"]
                    for r in rows))


def coverage_for_cases(cases, reports, *, min_transition_samples):
    """Every planned case, including a skipped/missing one, has a visible status."""
    by_id = {r["case_id"]: r for r in reports}
    rows = []
    for case in cases:
        report = by_id.get(case["case_id"])
        status = case.get("plan_status", "planned")
        row = dict(case_id=case["case_id"], kind=case["kind"], status=status,
                   reason=case.get("reason", ""), target=case.get("target"))
        if status == "planned":
            if report is None:
                row.update(status="failed_or_not_run")
            elif report.get("health", {}).get("healthy") is False:
                row.update(status="invalid_probe", reason=report["health"]["stop_reason"])
            elif case["kind"] == "liveness":
                row.update(status="measured" if report.get("health", {}).get("healthy")
                           else "unobserved", reason="completion_required_preflight")
            elif case["kind"] == "steady":
                row.update(status="measured" if report["points"] else "unobserved",
                           sample_count=sum(p["sample_count"] for p in report["points"]))
            elif case["kind"] == "transitions":
                requirements = []
                for mode in case["transition_modes"]:
                    for direction in case["transition_directions"]:
                        samples = [t for t in report["transitions"]
                                   if t["source_prefill_mode"] == mode
                                   and transition_direction(t["source"], t["target"]) == direction
                                   and t["source_had_decode_service"]
                                   and (not case.get("concurrent_transitions_only") or
                                        (int(t["source"].split("_")[1][1:]) > 0
                                         and int(t["target"].split("_")[1][1:]) > 0))]
                        requirements.append(dict(mode=mode, direction=direction,
                            samples=len(samples), required=min_transition_samples,
                            measured_pairs=sorted({(t["source"], t["target"]) for t in samples})))
                row.update(status="measured" if all(r["samples"] >= r["required"] for r in requirements)
                           else "unobserved", transitions=requirements)
            else:
                row.update(status="measured")
        rows.append(row)
    # Infeasible cells are outside the declared scope; unavailable inputs and
    # feasible-but-unobserved cells cannot be silently excluded from readiness.
    complete = bool(rows and any(r["status"] == "measured" for r in rows)
                    and all(r["status"] in {"measured", "infeasible"} for r in rows))
    return rows, complete


def measurement_configuration(base, defaults):
    """Compare actual experiment knobs, not host-specific binary/output paths."""
    env = dict(defaults, **base)
    paths = {"VLLM_VENDOR_DIR", "VLLM_EXECUTABLE", "VLLM_PYTHON_EXECUTABLE",
             "VLLM_PD_TBT_PROFILING_JSONL"}
    prefixes = ("VLLM_", "ISOLATION_", "FULL_VENDOR_", "DYNAMIC_HBM_", "PD_", "BENCHMARK_CUDA")
    return {k: str(v) for k, v in sorted(env.items())
            if (k.startswith(prefixes) or k in {"MODEL_NAME", "KV_CACHE_MEMORY_BYTES", "PREFILL_CHUNK_TOKENS"})
            and k not in paths and "PREDICTIVE_" not in k and k != "PD_DYNAMIC_SM_POLICY"
            and not k.endswith(("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL"))}
