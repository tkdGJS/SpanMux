"""CPU-only tests for prospective patch-effect metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.dependency_graph.patch_metrics import (
    PROTOCOL_ID,
    summarize_records,
    validate_metric_record,
    validate_repository_records,
)


def valid_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "patch_id": "sample-patch",
        "impact_record": "docs/patch_impacts/sample-patch.md",
        "status": "complete",
        "measurement_scope": "prospective",
        "graph_exposure": "used",
        "cohort": {
            "subsystem": "scheduler",
            "change_type": "runtime",
            "risk": "high",
            "complexity": "medium",
            "files_changed": 3,
            "runtime_files_changed": 1,
            "semantic_nodes_touched": 2,
            "stateful_resources_touched": 1,
            "async_boundaries_touched": 1,
        },
        "timing": {
            "design_started_at": "2026-09-11T00:00:00+09:00",
            "first_required_tests_passed_at": "2026-09-11T00:02:00+09:00",
            "elapsed_seconds_to_first_pass": 120,
            "agent_turns_to_first_pass": 2,
            "token_usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "source": None,
                "unavailable_reason": "authoritative counter unavailable",
            },
        },
        "counts": {
            "pre_edit_unresolved_dependencies": 1,
            "implementation_discovered_edges": 0,
            "post_implementation_missing_edges": 0,
            "secondary_correctness_errors": 0,
            "rollbacks": 0,
            "reworks": 0,
        },
        "events": [
            {
                "event_id": "unresolved-1",
                "kind": "pre_edit_unresolved_dependency",
                "observed_at": "2026-09-11T00:00:30+09:00",
                "summary": "missing output owner",
                "evidence": ["impact command output"],
            }
        ],
        "evidence": {
            "pre_edit": ["impact command"],
            "implementation": ["patch diff"],
            "post_edit": ["unit tests passed"],
        },
        "comparison": {
            "eligible": True,
            "exclusion_reason": None,
            "candidate_patch_ids": [],
            "design": "matched_observational",
            "pre_registered": True,
        },
    }


class PatchMetricsTest(unittest.TestCase):
    def test_valid_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            impact = root / "docs/patch_impacts/sample-patch.md"
            impact.parent.mkdir(parents=True)
            impact.write_text("sample", encoding="utf-8")
            errors = validate_metric_record(valid_record(), repo_root=root)
        self.assertEqual(errors, [])

    def test_count_must_reconcile_with_events(self) -> None:
        record = valid_record()
        record["counts"]["secondary_correctness_errors"] = 1  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            impact = root / "docs/patch_impacts/sample-patch.md"
            impact.parent.mkdir(parents=True)
            impact.write_text("sample", encoding="utf-8")
            errors = validate_metric_record(record, repo_root=root)
        self.assertTrue(
            any("secondary_correctness_errors" in error for error in errors)
        )

    def test_missing_token_counters_require_reason(self) -> None:
        record = valid_record()
        token_usage = record["timing"]["token_usage"]  # type: ignore[index]
        token_usage["unavailable_reason"] = ""  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            impact = root / "docs/patch_impacts/sample-patch.md"
            impact.parent.mkdir(parents=True)
            impact.write_text("sample", encoding="utf-8")
            errors = validate_metric_record(record, repo_root=root)
        self.assertTrue(any("unavailable_reason" in error for error in errors))

    def test_elapsed_time_must_match_timestamps(self) -> None:
        record = valid_record()
        record["timing"]["elapsed_seconds_to_first_pass"] = 12  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            impact = root / "docs/patch_impacts/sample-patch.md"
            impact.parent.mkdir(parents=True)
            impact.write_text("sample", encoding="utf-8")
            errors = validate_metric_record(record, repo_root=root)
        self.assertTrue(any("does not match timestamps" in error for error in errors))

    def test_report_excludes_ineligible_patch_from_aggregate(self) -> None:
        record = valid_record()
        record["comparison"] = {
            "eligible": False,
            "exclusion_reason": "tooling patch",
            "candidate_patch_ids": [],
            "design": "none",
            "pre_registered": False,
        }
        summary = summarize_records([record])
        self.assertIn("Complete prospective records: 1", summary)
        self.assertIn("no complete comparison-eligible records", summary)

    def test_repository_requires_metrics_for_nonlegacy_impact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            impact_dir = root / "docs/patch_impacts"
            metrics_dir = impact_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            impact = impact_dir / "sample-patch.md"
            impact.write_text("no metrics link", encoding="utf-8")
            protocol = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "adopted_at": "2026-09-11T00:00:00+09:00",
                "legacy_unmeasured_impact_records": [],
            }
            protocol_path = metrics_dir / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            errors, _records = validate_repository_records(
                repo_root=root,
                impact_dir=impact_dir,
                metrics_dir=metrics_dir,
                protocol_path=protocol_path,
            )
        self.assertTrue(any("has no metrics JSON" in error for error in errors))

    def test_legacy_exemption_is_explicit_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            impact_dir = root / "docs/patch_impacts"
            metrics_dir = impact_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            impact = impact_dir / "old-patch.md"
            impact.write_text("legacy", encoding="utf-8")
            protocol = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "adopted_at": "2026-09-11T00:00:00+09:00",
                "legacy_unmeasured_impact_records": [
                    {
                        "path": "docs/patch_impacts/old-patch.md",
                        "reason": "completed before adoption",
                    }
                ],
            }
            protocol_path = metrics_dir / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            errors, records = validate_repository_records(
                repo_root=root,
                impact_dir=impact_dir,
                metrics_dir=metrics_dir,
                protocol_path=protocol_path,
            )
        self.assertEqual(errors, [])
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
