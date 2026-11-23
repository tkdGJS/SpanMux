"""Validation and reporting helpers for prospective patch-effect metrics.

The records deliberately use only the Python standard library.  They measure
what was observed during a patch; they do not turn an observational before/
after comparison into a causal claim.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPACT_DIR = REPO_ROOT / "docs/patch_impacts"
METRICS_DIR = IMPACT_DIR / "metrics"
PROTOCOL_PATH = METRICS_DIR / "protocol.json"
PROTOCOL_ID = "dependency-graph-effectiveness-v1"

STATUS_VALUES = {"in_progress", "complete", "failed", "abandoned"}
MEASUREMENT_SCOPES = {"prospective", "retrospective"}
GRAPH_EXPOSURES = {"used", "not_used", "not_applicable"}
COMPARISON_DESIGNS = {
    "none",
    "matched_observational",
    "paired_replay",
    "randomized_ablation",
}
CHANGE_TYPES = {"runtime", "experiment", "tooling", "documentation", "test", "mixed"}
LEVELS = {"low", "medium", "high"}
COUNT_EVENT_KINDS = {
    "pre_edit_unresolved_dependencies": "pre_edit_unresolved_dependency",
    "implementation_discovered_edges": "implementation_discovered_edge",
    "post_implementation_missing_edges": "post_implementation_missing_edge",
    "secondary_correctness_errors": "secondary_correctness_error",
    "rollbacks": "rollback",
    "reworks": "rework",
}
COHORT_COUNTS = {
    "files_changed",
    "runtime_files_changed",
    "semantic_nodes_touched",
    "stateful_resources_touched",
    "async_boundaries_touched",
}
IGNORED_IMPACT_FILES = {"README.md", "TEMPLATE.md"}
IGNORED_METRIC_FILES = {"protocol.json", "TEMPLATE.json"}


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _timestamp(value: object, label: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty ISO-8601 timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        errors.append(f"{label} is not a valid ISO-8601 timestamp: {value!r}")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(f"{label} must include a UTC offset")
        return None
    return parsed


def _nonnegative_int(value: object, label: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{label} must be a non-negative integer")


def validate_protocol(
    protocol: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    if protocol.get("schema_version") != 1:
        errors.append("protocol.schema_version must be 1")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        errors.append(f"protocol.protocol_id must be {PROTOCOL_ID!r}")
    _timestamp(protocol.get("adopted_at"), "protocol.adopted_at", errors)

    exemptions = protocol.get("legacy_unmeasured_impact_records")
    legacy_paths: set[str] = set()
    if not isinstance(exemptions, list):
        errors.append("protocol.legacy_unmeasured_impact_records must be a list")
        return errors, legacy_paths
    for index, exemption in enumerate(exemptions):
        label = f"protocol.legacy_unmeasured_impact_records[{index}]"
        if not isinstance(exemption, dict):
            errors.append(f"{label} must be an object")
            continue
        path_value = exemption.get("path")
        reason = exemption.get("reason")
        if not _safe_relative_path(path_value):
            errors.append(f"{label}.path must be repository-relative")
            continue
        assert isinstance(path_value, str)
        if path_value in legacy_paths:
            errors.append(f"duplicate legacy exemption: {path_value}")
        legacy_paths.add(path_value)
        if not (repo_root / path_value).is_file():
            errors.append(f"legacy exemption path does not exist: {path_value}")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label}.reason must be non-empty")
    return errors, legacy_paths


def validate_metric_record(
    record: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if record.get("protocol_id") != PROTOCOL_ID:
        errors.append(f"protocol_id must be {PROTOCOL_ID!r}")

    patch_id = record.get("patch_id")
    if not isinstance(patch_id, str) or not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", patch_id or ""
    ):
        errors.append("patch_id must be a lowercase hyphenated identifier")
    elif source_path is not None and source_path.stem != patch_id:
        errors.append(
            f"patch_id {patch_id!r} does not match metrics filename "
            f"{source_path.name!r}"
        )

    impact_record = record.get("impact_record")
    if not _safe_relative_path(impact_record):
        errors.append("impact_record must be a repository-relative path")
    elif not (repo_root / str(impact_record)).is_file():
        errors.append(f"impact_record does not exist: {impact_record}")

    status = record.get("status")
    if status not in STATUS_VALUES:
        errors.append(f"status must be one of {sorted(STATUS_VALUES)}")
    if record.get("measurement_scope") not in MEASUREMENT_SCOPES:
        errors.append(
            f"measurement_scope must be one of {sorted(MEASUREMENT_SCOPES)}"
        )
    if record.get("graph_exposure") not in GRAPH_EXPOSURES:
        errors.append(f"graph_exposure must be one of {sorted(GRAPH_EXPOSURES)}")

    cohort = record.get("cohort")
    if not isinstance(cohort, dict):
        errors.append("cohort must be an object")
    else:
        if not isinstance(cohort.get("subsystem"), str) or not cohort.get(
            "subsystem"
        ):
            errors.append("cohort.subsystem must be non-empty")
        if cohort.get("change_type") not in CHANGE_TYPES:
            errors.append(
                f"cohort.change_type must be one of {sorted(CHANGE_TYPES)}"
            )
        for field in ("risk", "complexity"):
            if cohort.get(field) not in LEVELS:
                errors.append(f"cohort.{field} must be one of {sorted(LEVELS)}")
        for field in sorted(COHORT_COUNTS):
            _nonnegative_int(cohort.get(field), f"cohort.{field}", errors)
        if isinstance(cohort.get("runtime_files_changed"), int) and isinstance(
            cohort.get("files_changed"), int
        ) and cohort["runtime_files_changed"] > cohort["files_changed"]:
            errors.append(
                "cohort.runtime_files_changed cannot exceed cohort.files_changed"
            )

    timing = record.get("timing")
    if not isinstance(timing, dict):
        errors.append("timing must be an object")
    else:
        started = _timestamp(
            timing.get("design_started_at"), "timing.design_started_at", errors
        )
        passed_value = timing.get("first_required_tests_passed_at")
        elapsed = timing.get("elapsed_seconds_to_first_pass")
        turns = timing.get("agent_turns_to_first_pass")
        if status == "complete":
            passed = _timestamp(
                passed_value,
                "timing.first_required_tests_passed_at",
                errors,
            )
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
                errors.append(
                    "timing.elapsed_seconds_to_first_pass must be numeric for "
                    "a complete patch"
                )
            elif elapsed < 0:
                errors.append("timing.elapsed_seconds_to_first_pass cannot be negative")
            if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
                errors.append(
                    "timing.agent_turns_to_first_pass must be a positive integer "
                    "for a complete patch"
                )
            if started is not None and passed is not None:
                actual = (passed - started).total_seconds()
                if actual < 0:
                    errors.append("first test pass precedes design start")
                if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
                    if abs(actual - float(elapsed)) > 1.0:
                        errors.append(
                            "elapsed_seconds_to_first_pass does not match timestamps"
                        )
        elif any(value is not None for value in (passed_value, elapsed, turns)):
            errors.append(
                "first-pass timestamp, elapsed time, and turn count must all be null "
                "until status is complete"
            )

        token_usage = timing.get("token_usage")
        if not isinstance(token_usage, dict):
            errors.append("timing.token_usage must be an object")
        else:
            token_fields = ("input_tokens", "output_tokens", "total_tokens")
            token_values = [token_usage.get(field) for field in token_fields]
            for field, value in zip(token_fields, token_values):
                if value is not None:
                    _nonnegative_int(value, f"timing.token_usage.{field}", errors)
            if any(value is None for value in token_values):
                reason = token_usage.get("unavailable_reason")
                if not isinstance(reason, str) or not reason.strip():
                    errors.append(
                        "missing token counters require token_usage.unavailable_reason"
                    )
            else:
                source = token_usage.get("source")
                if not isinstance(source, str) or not source.strip():
                    errors.append("available token counters require token_usage.source")
                if token_values[0] + token_values[1] != token_values[2]:
                    errors.append("token_usage.total_tokens must equal input + output")

    counts = record.get("counts")
    if not isinstance(counts, dict):
        errors.append("counts must be an object")
        counts = {}
    for field in COUNT_EVENT_KINDS:
        _nonnegative_int(counts.get(field), f"counts.{field}", errors)

    events = record.get("events")
    event_counts: Counter[str] = Counter()
    event_ids: set[str] = set()
    if not isinstance(events, list):
        errors.append("events must be a list")
        events = []
    for index, event in enumerate(events):
        label = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{label} must be an object")
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            errors.append(f"{label}.event_id must be non-empty")
        elif event_id in event_ids:
            errors.append(f"duplicate event_id: {event_id}")
        else:
            event_ids.add(event_id)
        kind = event.get("kind")
        if kind not in COUNT_EVENT_KINDS.values():
            errors.append(f"{label}.kind is unknown: {kind!r}")
        else:
            event_counts[str(kind)] += 1
        _timestamp(event.get("observed_at"), f"{label}.observed_at", errors)
        if not isinstance(event.get("summary"), str) or not event.get("summary"):
            errors.append(f"{label}.summary must be non-empty")
        event_evidence = event.get("evidence")
        if not isinstance(event_evidence, list) or not event_evidence or not all(
            isinstance(value, str) and value.strip() for value in event_evidence
        ):
            errors.append(f"{label}.evidence must be a non-empty string list")

    for count_name, event_kind in COUNT_EVENT_KINDS.items():
        value = counts.get(count_name)
        if isinstance(value, int) and not isinstance(value, bool):
            if value != event_counts[event_kind]:
                errors.append(
                    f"counts.{count_name}={value} but {event_counts[event_kind]} "
                    f"{event_kind!r} events exist"
                )

    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
    else:
        for phase in ("pre_edit", "implementation", "post_edit"):
            values = evidence.get(phase)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                errors.append(f"evidence.{phase} must be a string list")
        if status == "complete" and not evidence.get("post_edit"):
            errors.append("a complete patch requires post_edit evidence")

    comparison = record.get("comparison")
    if not isinstance(comparison, dict):
        errors.append("comparison must be an object")
    else:
        eligible = comparison.get("eligible")
        if not isinstance(eligible, bool):
            errors.append("comparison.eligible must be boolean")
        reason = comparison.get("exclusion_reason")
        candidates = comparison.get("candidate_patch_ids")
        design = comparison.get("design")
        preregistered = comparison.get("pre_registered")
        if not isinstance(candidates, list) or not all(
            isinstance(value, str) and value for value in candidates
        ):
            errors.append("comparison.candidate_patch_ids must be a string list")
        if eligible is False and (not isinstance(reason, str) or not reason.strip()):
            errors.append("an ineligible record requires comparison.exclusion_reason")
        if eligible is True and reason not in (None, ""):
            errors.append("an eligible record cannot have an exclusion reason")
        if design not in COMPARISON_DESIGNS:
            errors.append(
                f"comparison.design must be one of {sorted(COMPARISON_DESIGNS)}"
            )
        if not isinstance(preregistered, bool):
            errors.append("comparison.pre_registered must be boolean")
        if eligible is True and design == "none":
            errors.append("an eligible record requires a comparison design")
        if eligible is False and design != "none":
            errors.append("an ineligible record must use comparison.design='none'")

    return errors


def validate_repository_records(
    *,
    repo_root: Path = REPO_ROOT,
    impact_dir: Path | None = None,
    metrics_dir: Path | None = None,
    protocol_path: Path | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    impact_dir = impact_dir or repo_root / "docs/patch_impacts"
    metrics_dir = metrics_dir or impact_dir / "metrics"
    protocol_path = protocol_path or metrics_dir / "protocol.json"
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        protocol = load_json_object(protocol_path)
    except ValueError as exc:
        return [str(exc)], records
    protocol_errors, legacy_paths = validate_protocol(protocol, repo_root=repo_root)
    errors.extend(protocol_errors)

    impact_paths = {
        path.relative_to(repo_root).as_posix(): path
        for path in impact_dir.glob("*.md")
        if path.name not in IGNORED_IMPACT_FILES
    }
    unknown_exemptions = legacy_paths - impact_paths.keys()
    for path in sorted(unknown_exemptions):
        errors.append(f"legacy exemption is not an impact record: {path}")

    metric_paths = {
        path.stem: path
        for path in metrics_dir.glob("*.json")
        if path.name not in IGNORED_METRIC_FILES
    }
    for impact_relative, impact_path in sorted(impact_paths.items()):
        if impact_relative in legacy_paths:
            continue
        expected_id = impact_path.stem
        metric_path = metric_paths.get(expected_id)
        if metric_path is None:
            errors.append(
                f"prospective impact record has no metrics JSON: {impact_relative}"
            )
            continue
        metric_relative = metric_path.relative_to(repo_root).as_posix()
        text = impact_path.read_text(encoding="utf-8")
        if metric_relative not in text:
            errors.append(
                f"impact record does not link its metrics JSON: {impact_relative}"
            )

    for patch_id, metric_path in sorted(metric_paths.items()):
        try:
            record = load_json_object(metric_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        record_errors = validate_metric_record(
            record, source_path=metric_path, repo_root=repo_root
        )
        errors.extend(f"{metric_path.name}: {error}" for error in record_errors)
        impact_relative = record.get("impact_record")
        if isinstance(impact_relative, str) and impact_relative in legacy_paths:
            errors.append(
                f"metric record {patch_id!r} points to a legacy-unmeasured "
                "impact record"
            )
        records.append(record)
    return errors, records


def summarize_records(records: Sequence[Mapping[str, Any]]) -> str:
    """Return a descriptive summary; never label observational deltas causal."""

    complete = [record for record in records if record.get("status") == "complete"]
    eligible = [
        record
        for record in complete
        if record.get("measurement_scope") == "prospective"
        and record.get("comparison", {}).get("eligible") is True
    ]
    lines = [
        "# Patch effectiveness measurements",
        "",
        f"Complete prospective records: {len(complete)}",
        "",
        "| Patch | Graph | Cohort | Secondary errors | Reworks | Rollbacks | "
        "Turns | Elapsed s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(complete, key=lambda value: str(value.get("patch_id"))):
        cohort = record.get("cohort", {})
        counts = record.get("counts", {})
        timing = record.get("timing", {})
        cohort_label = "/".join(
            str(cohort.get(key, "?"))
            for key in ("subsystem", "change_type", "risk", "complexity")
        )
        lines.append(
            f"| `{record.get('patch_id')}` | {record.get('graph_exposure')} | "
            f"{cohort_label} | {counts.get('secondary_correctness_errors')} | "
            f"{counts.get('reworks')} | {counts.get('rollbacks')} | "
            f"{timing.get('agent_turns_to_first_pass')} | "
            f"{timing.get('elapsed_seconds_to_first_pass')} |"
        )

    lines.extend(["", "## Descriptive aggregates", ""])
    for exposure in ("not_used", "used"):
        group = [
            record
            for record in eligible
            if record.get("graph_exposure") == exposure
        ]
        if not group:
            lines.append(
                f"- `{exposure}`: no complete comparison-eligible records"
            )
            continue
        elapsed_values = [
            float(record["timing"]["elapsed_seconds_to_first_pass"])
            for record in group
        ]
        turns_values = [
            record["timing"]["agent_turns_to_first_pass"] for record in group
        ]
        secondary_values = [
            record["counts"]["secondary_correctness_errors"] for record in group
        ]
        rework_values = [record["counts"]["reworks"] for record in group]
        lines.append(
            f"- `{exposure}`: n={len(group)}, median elapsed="
            f"{median(elapsed_values):.1f}s, "
            f"median turns={median(turns_values):g}, mean secondary errors="
            f"{sum(secondary_values) / len(group):.3f}, mean reworks="
            f"{sum(rework_values) / len(group):.3f}"
        )

    causal_design = [
        record
        for record in eligible
        if record.get("comparison", {}).get("design")
        in {"paired_replay", "randomized_ablation"}
        and record.get("comparison", {}).get("pre_registered") is True
    ]
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            f"Comparison-eligible records: {len(eligible)}.",
            f"Pre-registered paired/ablation records: {len(causal_design)}.",
            "These aggregates are observational. They do not prove that the graph "
            "caused a reduction, nor that a no-graph counterfactual must have failed. "
            "Use matched subsystem/risk/complexity patches or a paired ablation before "
            "reporting a percentage as causal.",
        ]
    )
    return "\n".join(lines) + "\n"
