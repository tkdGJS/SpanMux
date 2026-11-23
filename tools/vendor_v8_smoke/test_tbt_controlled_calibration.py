#!/usr/bin/env python3
"""No GPU: controlled trace recipes, actual coverage, frozen held-out AB."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import profile_vendor_v8_tbt_predictive as driver
import analyze_vendor_v8_tbt_calibration as analyzer
import tbt_calibration_plan as plan
import test_tbt_calibration as legacy


def source_rows():
    return [dict(request_id=f"source{i}", prompt=f"original prompt {i}", dataset="sharegpt",
                 input_tokens=length, min_tokens=17, max_tokens=17,
                 base_scheduled_offset_s=100. + 3 * i)
            for i, length in enumerate([1024, 1032, 1040, 1056, 8192, 8200, 48000, 49000])]


def args(root, **overrides):
    values = dict(output_dir=root / "output", trace=root / "source.jsonl", slo=root / "slo.json",
        splits=[(30, 10)], variants=["full_vendor_v8"], repeats=1, mode="calibrate",
        case_seconds=600., arrival_scale=1., request_limit=2, decode_batches=[1, 2],
        decode_contexts=[1024], context_window_tokens=1024, probe_output_tokens=256,
        kv_capacity_tokens=0, interference_input_min_tokens=32768, interference_requests=1,
        interference_delay_sec=30., min_samples=2, min_transition_samples=1,
        context_stride=1024, calibration_dir=None, dry_run=False, plan_only=False)
    return NS(**dict(values, **overrides))


class RecipeTests(unittest.TestCase):
    def test_runtime_fingerprints_include_required_copy_progress_and_sidecar(self):
        hashes = driver.runtime_hashes(ROOT / "vendor_v8")
        for path in ("vllm/v1/attention/ops/tiered_residency_provider.py",
                     "vllm/v1/core/tiered_required_d2h.py", "vllm/v1/core/virtual_context.py",
                     "vllm/v1/core/sched/prefill_chunk_scheduler.py"):
            self.assertIn(path, hashes)
            self.assertEqual(len(hashes[path]), 64)

    def test_batch_kv_budget_default_and_rollback_are_fingerprinted(self):
        key = "VLLM_ISOLATION_PREFILL_BATCH_KV_BUDGET"
        configured = plan.measurement_configuration({}, driver.DEFAULTS)
        self.assertEqual(configured[key], "1")
        old_path = plan.measurement_configuration({key: "0"}, driver.DEFAULTS)
        self.assertEqual(old_path[key], "0")
        self.assertNotEqual(configured, old_path)

    def recipe(self, **overrides):
        return plan.steady_recipe(source_rows(), **dict(dict(count=2, context=1024, window=1024,
            output_tokens=256, mode="none", max_seqs=100, max_model_len=65536,
            kv_capacity_tokens=0, interference_tokens=32768, interference_count=1,
            interference_delay=30.), **overrides))

    def test_solo_probe_has_no_background(self):
        result = self.recipe()
        self.assertEqual(result["plan_status"], "planned")
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual([r["base_scheduled_offset_s"] for r in result["rows"]], [0., 0.])
        self.assertEqual([r["max_tokens"] for r in result["rows"]], [256, 256])
        self.assertEqual([r["prompt"] for r in result["rows"]], [r["prompt"] for r in source_rows()[:2]])

    def test_interference_uses_real_distinct_long_prompt_and_declared_delay(self):
        result = self.recipe(mode="ring")
        self.assertEqual(result["target"]["prefill_mode"], "ring")
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["rows"][-1]["input_tokens"], 49000)
        self.assertEqual(result["rows"][-1]["base_scheduled_offset_s"], 30.)
        self.assertEqual(result["rows"][-1]["max_tokens"], 1)
        self.assertEqual(len(set(result["recipe"]["source_prompt_hashes"])), 3)

    def test_capacity_sequence_model_and_missing_inputs_are_distinct(self):
        for changes, status in [({"count": 101}, "infeasible"),
                                ({"context": 65535}, "infeasible"),
                                ({"kv_capacity_tokens": 1500}, "infeasible"),
                                ({"context": 16384}, "input_unavailable"),
                                ({"mode": "ring", "interference_count": 3}, "input_unavailable")]:
            with self.subTest(changes=changes):
                self.assertEqual(self.recipe(**changes)["plan_status"], status)
        self.assertEqual(self.recipe(kv_capacity_tokens=0)["plan_status"], "planned")

    def test_actual_selected_lengths_not_just_lower_band_bound(self):
        self.assertEqual(self.recipe(kv_capacity_tokens=2050)["plan_status"], "infeasible")

    def test_fingerprints_do_not_allow_duplicate_source_id_renaming(self):
        row = source_rows()[0]
        self.assertEqual(plan.prompt_fingerprint(row), plan.prompt_fingerprint(dict(row, request_id="other")))

    def test_heldout_order_and_relative_timeline_unchanged(self):
        source = source_rows()
        excluded = [plan.prompt_fingerprint(source[1])]
        selected, origin = plan.replay_rows(source, 3, excluded)
        self.assertEqual([r["request_id"] for r in selected], ["source0", "source2", "source3"])
        self.assertEqual([r["base_scheduled_offset_s"] for r in selected], [0., 6., 9.])
        self.assertEqual(origin, 100.)
        self.assertEqual(source[0]["base_scheduled_offset_s"], 100.)
        with self.assertRaises(ValueError):
            plan.replay_rows(source, 100)

    def test_positive_grids_fail_closed(self):
        for value in ("", "0", "1,1", "-1", "1,foo"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan.positive_ints(value)
        self.assertEqual(plan.positive_ints("1,2,100"), [1, 2, 100])

    def test_planner_generates_solo_and_mode_specific_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            options = args(Path(temp), variants=list(driver.VARIANTS))
            with patch.object(driver, "ROOT", Path(temp)):
                cases = driver.plan_controlled_cases(options, {}, source_rows())
        self.assertEqual(len(cases), 17)  # 2*(solo+native)*2 +2*(solo+native+ring)+3 transition
        self.assertEqual(sum(c["kind"] == "transitions" for c in cases), 3)
        for case in cases:
            if case["kind"] == "steady":
                self.assertEqual(case["env"]["LOONGSERVE_ARRIVAL_TIME_SCALE"], "1")
                self.assertEqual(case["env"]["VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH"], "1")
                self.assertEqual(case["target"]["min_decode_steps"], 2)
                if case["target"]["prefill_mode"] == "none":
                    self.assertEqual(case["target"]["profile_id"], "d40_p0")
                    self.assertEqual(len(case["rows"]), case["target"]["decode_requests"])
                if case["variant"] != "full_vendor_v8":
                    self.assertNotEqual(case["target"]["prefill_mode"], "ring")

    def test_validation_AB_uses_same_disjoint_trace_and_frozen_timings(self):
        with tempfile.TemporaryDirectory() as temp:
            options = args(Path(temp), mode="validate")
            ref = dict(manifest={"training_prompt_hashes": [plan.prompt_fingerprint(source_rows()[0])]},
                       profile_path="/tmp/frozen.json", calibration_ms=dict(lead=100, cutover=5, min_step=20))
            with patch.object(driver, "ROOT", Path(temp)):
                cases = driver.plan_controlled_cases(options, {}, source_rows(), ref)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["rows"], cases[1]["rows"])
        self.assertEqual(cases[0]["rows"][0]["request_id"], "source1")
        self.assertEqual(cases[0]["env"]["PD_DYNAMIC_SM_POLICY"], "tbt_slo_feedback")
        self.assertEqual(cases[1]["env"]["PD_DYNAMIC_SM_POLICY"], "tbt_slo_predictive")
        self.assertEqual(cases[1]["env"]["VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS"], "5")

    def test_secret_and_machine_paths_are_not_measurement_configuration(self):
        left = plan.measurement_configuration({"VLLM_VENDOR_DIR": "/one", "VLLM_API_KEY": "secret"}, driver.DEFAULTS)
        right = plan.measurement_configuration({"VLLM_VENDOR_DIR": "/two"}, driver.DEFAULTS)
        self.assertEqual(left, right)
        self.assertNotEqual(left, plan.measurement_configuration({"VLLM_MAX_NUM_SEQS": "10"}, driver.DEFAULTS))


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = legacy.AnalyzerTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def steady(self, **changes):
        case = self.fixture.case()
        case.update(kind="steady", plan_status="planned", variant="full_vendor_v8", target=dict(
            profile_id="d30_p10", prefill_mode="native", decode_requests=2,
            request_ids=["q0", "q1"], context_min=1024, context_max=2048, min_decode_steps=2))
        case["target"].update(changes)
        return case

    def test_actual_cohort_qualifies_and_counts_steps_without_multiplying_batch_width(self):
        result = analyzer.analyze_case(self.steady(), min_samples=2)
        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(result["points"][0]["decode_step_count"], 9)
        self.assertEqual(result["points"][0]["sample_count"], 18)

    def test_wrong_batch_mode_profile_context_identity_never_counts(self):
        case = self.steady()
        target = case["target"].copy()
        for change in (dict(decode_requests=3), dict(prefill_mode="none"), dict(profile_id="d40_p0"),
                       dict(context_min=2048), dict(context_max=1100), dict(request_ids=["other0", "other1"]),
                       dict(min_decode_steps=10)):
            with self.subTest(change=change):
                case["target"] = dict(target, **change)
                self.assertEqual(analyzer.analyze_case(case, min_samples=2)["points"], [])

    def test_one_long_one_short_cannot_match_all_long_target(self):
        case = self.steady()
        events = analyzer.read_capture(case["capture_path"])
        for event in events:
            if event["event"] == "submit" and event["phase"] == "decode":
                event["rows"][0]["context_tokens"] = 512
        self.fixture.write_events(Path(case["results_dir"]), events)
        self.assertEqual(analyzer.analyze_case(case, min_samples=2)["points"], [])

    def test_missing_feasible_cell_blocks_scope_but_infeasible_is_explicit(self):
        cases = [dict(case_id="ok", kind="steady", plan_status="planned"),
                 dict(case_id="big", kind="steady", plan_status="infeasible", reason="kv_limit")]
        reports = [dict(case_id="ok", points=[{"sample_count": 32}])]
        rows, complete = plan.coverage_for_cases(cases, reports, min_transition_samples=2)
        self.assertTrue(complete)
        self.assertEqual(rows[-1]["status"], "infeasible")
        cases[-1]["plan_status"] = "input_unavailable"
        self.assertFalse(plan.coverage_for_cases(cases, reports, min_transition_samples=2)[1])
        cases[-1]["plan_status"] = "planned"
        self.assertFalse(plan.coverage_for_cases(cases, reports, min_transition_samples=2)[1])

    def test_transition_modes_directions_and_repetition_are_separate_requirements(self):
        cases = [dict(case_id="t", kind="transitions", transition_modes=["native", "ring"],
                      transition_directions=["increase", "decrease"])]
        transitions = [dict(source="d30_p10", target="d32_p8", source_prefill_mode="native",
                            source_had_decode_service=True)]*4
        reports = [dict(case_id="t", transitions=transitions)]
        self.assertFalse(plan.coverage_for_cases(cases, reports, min_transition_samples=2)[1])
        for mode in ("native", "ring"):
            for source, target in (("d30_p10", "d32_p8"), ("d32_p8", "d30_p10")):
                transitions.extend([dict(source=source, target=target, source_prefill_mode=mode,
                                         source_had_decode_service=True)]*2)
        self.assertTrue(plan.coverage_for_cases(cases, reports, min_transition_samples=2)[1])

    def test_steady_only_is_stage_ready_but_not_deployable(self):
        case = self.steady()
        path = self.root / "manifest.json"
        path.write_text(json.dumps(dict(cases=[case], splits=[[30, 10]], trace_sha256="source",
            slo_sha256="slo", experiment_mode="steady", min_transition_samples=2)))
        self.assertEqual(analyzer.analyze(path, min_samples=2), 0)
        report = json.loads((self.root / "calibration/coverage.json").read_text())
        self.assertTrue(report["stage_ready"])
        self.assertFalse(report["ready"])
        self.assertFalse((self.root / "calibration/predictive.env").exists())

    def test_different_trace_hashes_are_attributed_per_case(self):
        case = self.steady()
        meta = Path(case["results_dir"]) / "run_metadata.json"
        data = json.loads(meta.read_text())
        data["workload_sha256"] = "controlled_case"
        meta.write_text(json.dumps(data))
        case["trace_sha256"] = "controlled_case"
        self.assertTrue(analyzer.analyze_case(case, min_samples=2)["points"])
        case["trace_sha256"] = "source_trace"
        with self.assertRaises(ValueError):
            analyzer.analyze_case(case, min_samples=2)

    def test_calibrate_requires_every_declared_cell_before_export(self):
        cases = []
        for mode, profile in (("native", "d30_p10"), ("ring", "d30_p10"), ("none", "d40_p0")):
            case = self.fixture.case(mode=mode, profile=profile, name=mode)
            case.update(kind="steady", plan_status="planned", variant="full_vendor_v8", target=dict(
                profile_id=profile, prefill_mode=mode, decode_requests=2, request_ids=["q0", "q1"],
                context_min=1024, context_max=2048, min_decode_steps=2))
            cases.append(case)
        for mode in ("native", "ring"):
            case = self.fixture.case(mode=mode, name="transition_"+mode)
            self.fixture.add_servicing_transition(case, mode)
            case.update(kind="transitions", plan_status="planned", variant="full_vendor_v8",
                        transition_modes=[mode], transition_directions=["increase"])
            cases.append(case)
        cases.append(dict(case_id="unavailable_large_batch", kind="steady", plan_status="input_unavailable"))
        manifest = dict(cases=cases, splits=[[30, 10]], variants=["full_vendor_v8"],
                        trace_sha256="source", slo_sha256="slo", experiment_mode="calibrate",
                        min_transition_samples=1)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(analyzer.analyze(path, min_samples=2), 3)
        self.assertFalse((self.root / "calibration/predictive.env").exists())
        report = json.loads((self.root / "calibration/coverage.json").read_text())
        self.assertEqual(report["missing_profile_modes"], [])
        self.assertFalse(report["declared_coverage_complete"])
        cases.pop()
        path.write_text(json.dumps(manifest))
        self.assertEqual(analyzer.analyze(path, min_samples=2, output_dir=self.root / "complete"), 0)
        self.assertTrue((self.root / "complete/predictive.env").is_file())
        profile = json.loads((self.root / "complete/tbt_overlap_profile.json").read_text())
        self.assertEqual({p["case_id"] for p in profile["points"]}, {"native", "ring", "none"})

    def test_validation_does_not_emit_table_or_train_on_holdout(self):
        case = self.fixture.case()
        case.update(kind="validation", variant="full_vendor_v8", validation_policy="feedback")
        profile = self.root / "frozen.json"
        profile.write_text(json.dumps(dict(schema_version=2, measurement_contract=analyzer.CONTRACT,
            points=[dict(profile_id="d30_p10", prefill_mode="native", max_decode_context_tokens=2048,
                         max_decode_requests=2, p95_per_token_tbt_ms=60)])))
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(cases=[case], experiment_mode="validate",
            validation_reference=dict(profile_path=str(profile), profile_sha256=analyzer.sha256(profile),
                                      calibration_id="frozen"))))
        self.assertEqual(analyzer.analyze(manifest, min_samples=2), 0)
        output = self.root / "calibration"
        result = json.loads((output / "validation.json").read_text())
        self.assertFalse(result["training_performed"])
        self.assertFalse((output / "tbt_overlap_profile.json").exists())
        self.assertFalse((output / "predictive.env").exists())
        self.assertEqual(result["uncovered_points"], 0)
        self.assertFalse(result["predictions"][0]["underestimated"])

    def test_changed_frozen_profile_fails_validation(self):
        output = self.root / "output"
        output.mkdir()
        profile = self.root / "profile.json"
        profile.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            analyzer.analyze_validation(dict(validation_reference=dict(profile_path=str(profile),
                profile_sha256="old")), [], [], output)


class CLITests(unittest.TestCase):
    def test_plan_only_and_actual_wrapper_dry_run_do_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace, slo = root / "trace.jsonl", root / "slo.json"
            trace.write_text("".join(json.dumps(r)+"\n" for r in source_rows()))
            slo.write_text("{}")
            output = ROOT / "tbt_controlled_cpu_no_output"
            self.assertFalse(output.exists())
            command = [sys.executable, str(ROOT / "tools/profile_vendor_v8_tbt_predictive.py"),
                "--trace", str(trace), "--slo", str(slo), "--output-dir", str(output),
                "--mode", "calibrate", "--decode-batches", "1", "--decode-contexts", "1024",
                "--splits", "30/10", "--request-limit", "2", "--case-seconds", "60"]
            for mode in ("--plan-only", "--dry-run"):
                result = subprocess.run([*command, mode], cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn("planned=4 runnable=4", result.stdout)
                if mode == "--dry-run":
                    self.assertIn("phase_future_depths=1/1/2", result.stdout)
                self.assertFalse(output.exists())


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = args(self.root, mode="transitions", repeats=2)
        self.options.trace.write_text("".join(json.dumps(r)+"\n" for r in source_rows()))
        self.options.slo.write_text("{}")
        self.base = {"VLLM_VENDOR_DIR": str(ROOT / "vendor_v8")}
        self.enterContext(patch.object(driver, "ROOT", self.root))
        self.enterContext(patch.object(driver, "runtime_hashes", return_value={"core": "version"}))
        self.enterContext(patch.object(driver.subprocess, "run", return_value=NS(stdout="GPU,uuid,driver,15GiB", stderr="")))

    # Python3.10 unittest lacks TestCase.enterContext.
    def enterContext(self, manager):
        result = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return result

    def manifest(self):
        return json.loads((self.options.output_dir / "manifest.json").read_text())

    def test_failed_case_stops_before_next_server(self):
        with patch.object(driver.subprocess, "Popen", return_value=NS(wait=lambda: 9)) as launch:
            self.assertEqual(driver.run_controlled(self.options, self.base), 9)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual([c["returncode"] for c in self.manifest()["cases"]], [9, None])

    def test_mutation_after_success_is_persisted_as_failed_evidence(self):
        def wait():
            self.options.slo.write_text('{"changed":true}')
            return 0
        with patch.object(driver.subprocess, "Popen", return_value=NS(wait=wait)) as launch:
            with self.assertRaisesRegex(ValueError, "marked failed"):
                driver.run_controlled(self.options, self.base)
        self.assertEqual(launch.call_count, 1)
        first = self.manifest()["cases"][0]
        self.assertEqual(first["returncode"], 2)
        self.assertIn("changed", first["provenance_error"])

    def test_interrupt_targets_only_owned_group_and_stops_sweep(self):
        from unittest.mock import Mock
        process = NS(pid=12345, wait=Mock(side_effect=[KeyboardInterrupt, 0]))
        with patch.object(driver.subprocess, "Popen", return_value=process) as launch, \
                patch.object(driver.os, "killpg") as kill:
            self.assertEqual(driver.run_controlled(self.options, self.base), 130)
        self.assertEqual(launch.call_count, 1)
        kill.assert_called_once_with(12345, driver.signal.SIGTERM)
        self.assertEqual(self.manifest()["cases"][0]["returncode"], 130)

    def test_missing_feasible_input_never_launches_or_returns_ready(self):
        self.options.mode = "steady"
        self.options.decode_contexts = [16000]
        with patch.object(driver.subprocess, "Popen") as launch:
            self.assertEqual(driver.run_controlled(self.options, self.base), 3)
        launch.assert_not_called()
        coverage = json.loads((self.options.output_dir / "calibration/coverage.json").read_text())
        self.assertFalse(coverage["ready"])
        self.assertTrue(all(c["status"] == "input_unavailable" for c in coverage["declared_coverage"]))


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = args(self.root, mode="validate", calibration_dir=self.root / "calibration")
        self.options.slo.write_text("{}")
        self.options.calibration_dir.mkdir()
        self.manifest = dict(experiment_mode="calibrate", splits=[[30, 10]], variants=["full_vendor_v8"],
            measurement_configuration=plan.measurement_configuration({}, driver.DEFAULTS),
            slo_sha256=analyzer.sha256(self.options.slo))
        self.profile = dict(slo_sha256=self.manifest["slo_sha256"], manifest=str(self.root / "manifest.json"))
        self.coverage = dict(ready=True, declared_coverage_complete=True,
                            calibration_ms=dict(lead=100, cutover=5, min_step=20))
        self.save()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        path = self.options.calibration_dir / "tbt_overlap_profile.json"
        path.write_text(json.dumps(self.profile))
        self.coverage["profile_sha256"] = analyzer.sha256(path)
        (self.options.calibration_dir / "coverage.json").write_text(json.dumps(self.coverage))

    def test_compatible_reference_returns_exact_frozen_table(self):
        ref = driver.load_validation_reference(self.options, {})
        self.assertEqual(ref["calibration_ms"]["cutover"], 5)
        self.assertEqual(ref["profile_sha256"], self.coverage["profile_sha256"])

    def test_partial_legacy_or_changed_configuration_is_rejected(self):
        self.coverage["ready"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "ready"):
            driver.load_validation_reference(self.options, {})
        self.coverage["ready"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "options differ"):
            driver.load_validation_reference(self.options, {"VLLM_MAX_NUM_SEQS": "10"})
        self.manifest["experiment_mode"] = "replay"
        self.save()
        with self.assertRaisesRegex(ValueError, "controlled"):
            driver.load_validation_reference(self.options, {})

    def test_slo_splits_timings_and_profile_hash_are_guarded(self):
        for key, value in (("lead", 0), ("min_step", -1), ("cutover", float("nan"))):
            old = self.coverage["calibration_ms"][key]
            self.coverage["calibration_ms"][key] = value
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "timing"):
                driver.load_validation_reference(self.options, {})
            self.coverage["calibration_ms"][key] = old
        self.save()
        self.options.splits = [(32, 8)]
        with self.assertRaisesRegex(ValueError, "splits"):
            driver.load_validation_reference(self.options, {})
        self.options.splits = [(30, 10)]
        (self.options.calibration_dir / "tbt_overlap_profile.json").write_text(json.dumps(dict(self.profile, changed=True)))
        with self.assertRaisesRegex(ValueError, "artifact"):
            driver.load_validation_reference(self.options, {})


if __name__ == "__main__":
    unittest.main()
