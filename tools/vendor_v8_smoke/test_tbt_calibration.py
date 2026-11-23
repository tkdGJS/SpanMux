#!/usr/bin/env python3
"""CPU-only observer, exact-token analyzer and real launcher routing tests."""
from __future__ import annotations

import ast
from concurrent.futures import Future
import csv
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_vendor_v8_tbt_calibration as analyzer
import profile_vendor_v8_tbt_predictive as driver

Observer = runpy.run_path(str(ROOT / "vendor_v8/vllm/v1/engine/pd_tbt_profiling.py"))["TBTCaptureObserver"]


class CaptureTests(unittest.TestCase):
    def test_scalar_snapshot_and_private_ring(self):
        records = []
        observer = Observer(records.append)
        future = Future()
        prefill = NS(pd_phase_submit_seq=1, pd_prefill_mode="native",
                     total_num_scheduled_tokens=2048,
                     kv_retention_layerwise_transfer=NS(state="released"))
        output = NS(pd_phase_submit_seq=2, pd_sm_profile_id="d30_p10", pd_sm_profile_epoch=3,
                    total_num_scheduled_tokens=2, num_scheduled_tokens={"A": 1, "B": 1})
        requests = {rid: NS(num_computed_tokens=1234, num_output_tokens=4) for rid in ("A", "B")}
        queues = {"prefill": [(future, prefill, None)]}
        observer.submit(output, "decode", requests, queues)
        observer.recover(output, {0: NS(outputs=[NS(request_id="A", new_token_ids=[42]),
                                                NS(request_id="B", new_token_ids=[])])}, queues)
        requests["A"].num_computed_tokens = 9999
        self.assertEqual(records[1]["rows"][0]["context_tokens"], 1234)
        self.assertEqual(records[1]["prefills"][0]["mode"], "ring")
        self.assertEqual(records[2]["tokens"], [{"request_id": "A", "count": 1}])
        future.set_result(None)
        self.assertEqual(observer.prefills(queues), [])
        observer.close()
        json.dumps(records)  # contains no tensor/Future/request reference
        self.assertEqual([r["sequence"] for r in records], [1, 2, 3, 4])

    def test_zero_work_not_submitted(self):
        records = []
        observer = Observer(records.append)
        observer.submit(NS(total_num_scheduled_tokens=0), "prefill", {}, {})
        self.assertEqual(len(records), 1)

    def test_real_async_writer_flushes_capture_footer(self):
        Writer = runpy.run_path(str(ROOT / "vendor_v8/vllm/v1/engine/async_jsonl_writer.py"))["AsyncJSONLWriter"]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "capture.jsonl")
            writer = Writer()
            observer = Observer(lambda r: writer.submit(path, r))
            observer.write("probe", value=1)
            observer.close()
            writer.close()
            self.assertEqual(len(analyzer.read_capture(path)), 3)

    def test_engine_default_off_and_existing_async_writer(self):
        text = (ROOT / "vendor_v8/vllm/v1/engine/core.py").read_text()
        self.assertIn('os.environ.get("VLLM_PD_TBT_PROFILING_JSONL", "")', text)
        self.assertIn('if tbt_capture_path else None', text)
        tree = ast.parse(text)
        hooks = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr in {"submit", "recover", "write", "close"}
                 and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "_tbt_capture"]
        self.assertEqual(len(hooks), 7)
        self.assertNotIn("synchronize", (ROOT / "vendor_v8/vllm/v1/engine/pd_tbt_profiling.py").read_text())

    def test_actual_engine_submission_keeps_same_owner_with_or_without_observer(self):
        path = ROOT / "vendor_v8/vllm/v1/engine/core.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_assign_pd_phase_submission")
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        scope = {}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        for enabled in (False, True):
            records = []
            engine = NS(_tbt_capture=Observer(records.append) if enabled else None,
                _record_pd_sm_transition_resume_submission=lambda *args: None,
                _pd_phase_queue_submit_seq=0, pd_phase_depth_enabled=False,
                pd_sm_transition=NS(note_decode_submission=lambda count: None),
                scheduler=NS(requests={"A": NS(num_computed_tokens=10, num_output_tokens=2)}),
                _bind_pd_phase_execution_key=lambda out, phase: phase)
            output = NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"A": 1},
                        pd_sm_profile_id="d40_p0", pd_sm_profile_epoch=2)
            self.assertEqual(scope["_assign_pd_phase_submission"](engine, {}, output, "decode"), "decode")
            self.assertEqual((output.pd_phase_submit_seq, output.pd_phase_slot), (1, 0))
            self.assertEqual(len(records), 2 if enabled else 0)


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def case(self, mode="native", count=2, name="case", profile="d30_p10"):
        root = self.root / name
        root.mkdir()
        metadata = {"tbt_measurement": "vllm choice.token_ids",
                    "completion_response_contract": {"return_token_ids": True},
                    "workload_sha256": "abc", "summary": {"failure_count": 0},
                    "trace_start_monotonic_ts": 1., "observation_end_monotonic_ts": 3.}
        (root / "run_metadata.json").write_text(json.dumps(metadata))
        events = [{"event": "start", "timestamp": 1., "contract": "tbt-capture-v1"}]
        signature = [] if mode == "none" else [{"submit_seq": 100, "mode": mode, "query_tokens": 2048}]
        if signature:
            events.append({"event": "submit", "timestamp": 1.1, "submit_seq": 100,
                           "phase": "prefill", "rows": [], "profile_id": profile,
                           "epoch": 0, "prefills": []})
        clients = []
        for rid in range(count):
            clients.append(dict(request_id=f"q{rid}", chunk_tokens=1,
                                arrival_monotonic_ts=1.5, gap_start_monotonic_ts="",
                                estimated_per_token_gap_ms=""))
        for index in range(10):
            submit = 2 + index * .05
            server_ids = [f"cmpl-q{rid}-0-deadbeef" for rid in range(count)]
            events.append(dict(event="submit", timestamp=submit, submit_seq=index + 1,
                phase="decode", profile_id=profile, epoch=0, prefills=signature,
                rows=[dict(request_id=rid, scheduled_tokens=1, context_tokens=1200+index,
                           output_start=index+1) for rid in server_ids]))
            events.append(dict(event="recover", timestamp=submit+.02, submit_seq=index+1,
                tokens=[dict(request_id=rid, count=1) for rid in server_ids], prefills=signature))
            for rid in range(count):
                start = 1.5 if index == 0 else submit-.05+.023
                clients.append(dict(request_id=f"q{rid}", chunk_tokens=1,
                    arrival_monotonic_ts=submit+.023, gap_start_monotonic_ts=start,
                    estimated_per_token_gap_ms=(submit+.023-start)*1000))
        events.append({"event": "end", "timestamp": 3.})
        self.write_events(root, events)
        self.write_clients(root, clients)
        return dict(case_id=name, results_dir=str(root), capture_path=str(root / "capture.jsonl"),
                    trace_sha256="abc", kind="feedback", returncode=0)

    @staticmethod
    def write_events(root, events):
        for i, r in enumerate(events, 1):
            r["sequence"] = i
        (root / "capture.jsonl").write_text("".join(json.dumps(r)+"\n" for r in events))

    @staticmethod
    def write_clients(root, clients):
        with (root / "client_token_timeline.csv").open("w", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=list(clients[0]))
            writer.writeheader()
            writer.writerows(clients)

    def test_actual_batch_floor_context_and_exact_tbt(self):
        case = self.case()
        result = analyzer.analyze_case(case, min_samples=2)
        p = result["points"][0]
        self.assertEqual(p["max_decode_requests"], 2)
        self.assertEqual(p["max_decode_context_tokens"], 1024)
        self.assertEqual(p["sample_count"], 18)
        self.assertAlmostEqual(p["p95_per_token_tbt_ms"], 50)
        self.assertAlmostEqual(result["min_step_ms"], 50)

    @staticmethod
    def move_client_bundle(root):
        client = root / "client"
        client.mkdir()
        for name in ("run_metadata.json", "client_token_timeline.csv"):
            (root / name).rename(client / name)
        return client

    def archive_case(self):
        case = self.case()
        archive = self.root / "archive"
        result = archive / "cases" / "case_workload_loongserve_full_vendor_v8_results"
        result.parent.mkdir(parents=True)
        shutil.copytree(case["results_dir"], result)
        self.move_client_bundle(result)
        capture = archive / "captures" / "case.jsonl"
        capture.parent.mkdir()
        (result / "capture.jsonl").rename(capture)
        case.update(results_dir=str(result), capture_path=str(capture))
        manifest = archive / "manifest.json"
        manifest.write_text(json.dumps(dict(cases=[case], splits=[[30, 10]],
                                           trace_sha256="abc", slo_sha256="slo")))
        return case, archive

    def test_real_client_subdirectory_and_direct_client_path(self):
        case = self.case()
        expected = analyzer.analyze_case(case, min_samples=2)
        client = self.move_client_bundle(Path(case["results_dir"]))
        for result_dir in (client.parent, client):
            with self.subTest(result_dir=result_dir):
                result = analyzer.analyze_case(dict(case, results_dir=str(result_dir)), min_samples=2)
                self.assertEqual(result["points"], expected["points"])
                self.assertEqual(result["input_paths"]["client_dir"], str(client))
                self.assertFalse(result["input_paths"]["archive_rebased"])

    def test_ambiguous_or_split_client_files_are_not_joined(self):
        case = self.case()
        root = Path(case["results_dir"])
        client = root / "client"
        client.mkdir()
        shutil.copy(root / "run_metadata.json", client / "run_metadata.json")
        with self.assertRaisesRegex(ValueError, "ambiguous client bundle"):
            analyzer.analyze_case(case)
        (root / "run_metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "ambiguous client bundle"):
            analyzer.analyze_case(case)

    def test_incomplete_client_bundle_fails_closed(self):
        case = self.case()
        client = self.move_client_bundle(Path(case["results_dir"]))
        (client / "client_token_timeline.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "incomplete client bundle"):
            analyzer.analyze_case(case)

    def test_nested_layout_preserves_measurement_and_failure_guards(self):
        case = self.case()
        path = self.move_client_bundle(Path(case["results_dir"])) / "run_metadata.json"
        original = json.loads(path.read_text())
        variants = [dict(original, workload_sha256="wrong"),
                    dict(original, summary={"failure_count": 1}),
                    dict(original, completion_response_contract={"return_token_ids": False})]
        for metadata in variants:
            path.write_text(json.dumps(metadata))
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                analyzer.analyze_case(case)

    def test_relative_paths_use_manifest_not_process_working_directory(self):
        case = self.case()
        expected = analyzer.analyze_case(case, min_samples=2)
        relative = dict(case, results_dir="case", capture_path="case/capture.jsonl")
        actual = analyzer.analyze_case(relative, manifest_dir=self.root, min_samples=2)
        self.assertEqual(actual["points"], expected["points"])
        with self.assertRaisesRegex(ValueError, "escapes manifest directory"):
            analyzer.analyze_case(dict(relative, capture_path="../capture.jsonl"),
                                  manifest_dir=self.root)

    def test_relocated_archive_rebases_both_paths_without_mutating_manifest(self):
        case, archive = self.archive_case()
        relocated = self.root / "relocated"
        shutil.copytree(archive, relocated)
        manifest = relocated / "manifest.json"
        before = analyzer.sha256(manifest)
        expected = analyzer.analyze_case(case, min_samples=2)
        actual = analyzer.analyze_case(case, manifest_dir=relocated, min_samples=2)
        self.assertEqual(actual["points"], expected["points"])
        paths = actual["input_paths"]
        self.assertTrue(paths["archive_rebased"])
        self.assertTrue(Path(paths["client_dir"]).is_relative_to(relocated))
        self.assertTrue(Path(paths["capture_path"]).is_relative_to(relocated))
        self.assertEqual(analyzer.sha256(manifest), before)

    def test_incomplete_local_copy_does_not_borrow_original_capture(self):
        case, archive = self.archive_case()
        relocated = self.root / "relocated"
        shutil.copytree(archive, relocated)
        (relocated / "captures/case.jsonl").unlink()
        self.assertTrue(Path(case["capture_path"]).exists())
        with self.assertRaisesRegex(FileNotFoundError, "missing paired case capture"):
            analyzer.analyze_case(case, manifest_dir=relocated)

    def test_relocation_does_not_search_unrelated_capture_basenames(self):
        case, archive = self.archive_case()
        case.update(results_dir="/missing-archive/cases/case_workload_loongserve_full_vendor_v8_results",
                    capture_path="/missing-archive/captures/unrelated-case.jsonl")
        with self.assertRaises(FileNotFoundError):
            analyzer.analyze_case(case, manifest_dir=archive)

    def test_fresh_output_preserves_previous_analysis_and_refuses_overwrite(self):
        case, archive = self.archive_case()
        old = archive / "calibration"
        old.mkdir()
        (old / "coverage.json").write_text('{"ready": false, "old": true}')
        before = analyzer.sha256(old / "coverage.json")
        output = archive / "reanalysis"
        self.assertEqual(analyzer.analyze(archive / "manifest.json", min_samples=2,
                                         output_dir=output), 3)
        report = json.loads((output / "coverage.json").read_text())
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["cases"][0]["points"]), 1)
        self.assertFalse((output / "predictive.env").exists())
        self.assertEqual(analyzer.sha256(old / "coverage.json"), before)
        with self.assertRaises(FileExistsError):
            analyzer.analyze(archive / "manifest.json", output_dir=output)

    def test_cli_reanalyzes_relocated_archive_without_server_or_source_writes(self):
        _, archive = self.archive_case()
        relocated = self.root / "relocated"
        shutil.copytree(archive, relocated)
        before = {p: analyzer.sha256(p) for p in relocated.rglob("*") if p.is_file()}
        output = self.root / "fresh_report"
        result = subprocess.run([sys.executable, str(ROOT / "tools/analyze_vendor_v8_tbt_calibration.py"),
            "--manifest", str(relocated / "manifest.json"), "--output-dir", str(output),
            "--min-samples", "2"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        report = json.loads((output / "coverage.json").read_text())
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["cases"][0]["input_paths"]["archive_rebased"])
        self.assertEqual({p: analyzer.sha256(p) for p in before}, before)

    def test_ring_and_none_do_not_infer_from_dram_pages(self):
        for mode in ("ring", "none"):
            with self.subTest(mode=mode):
                case = self.case(mode=mode, name=mode)
                result = analyzer.analyze_case(case, min_samples=2)
                self.assertEqual(result["points"][0]["prefill_mode"], mode)

    def test_missing_footer_and_trace_drops_fail_closed(self):
        case = self.case()
        path = Path(case["capture_path"])
        original = path.read_text().splitlines()
        for lines in (original[:-1], original[:3]+original[4:]):
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaises(ValueError):
                analyzer.read_capture(path)

    def test_cancelled_raw_samples_not_measured(self):
        case = self.case()
        path = Path(case["capture_path"])
        events = analyzer.read_capture(path)
        for e in events:
            if e["event"] == "recover":
                e["tokens"] = e["tokens"][:1]
        self.write_events(path.parent, events)
        result = analyzer.analyze_case(case, min_samples=2)
        self.assertEqual(result["points"], [])
        self.assertEqual(result["rejected"]["cancelled_or_nonuniform_output"], 10)

    def test_client_multitoken_sse_normalized(self):
        case = self.case(count=1)
        root = Path(case["results_dir"])
        rows = analyzer.read_csv(root / "client_token_timeline.csv")
        # Bundle Decode tokens3/4 in one SSE; two eligible IDs, 100ms /2.
        rows[3]["chunk_tokens"] = 2
        rows[3]["gap_start_monotonic_ts"] = rows[2]["gap_start_monotonic_ts"]
        rows[3]["estimated_per_token_gap_ms"] = 50
        del rows[2]
        self.write_clients(root, rows)
        result = analyzer.analyze_case(case, min_samples=2)
        self.assertAlmostEqual(result["points"][0]["p95_per_token_tbt_ms"], 50)
        self.assertEqual(result["points"][0]["sample_count"], 9)

    def test_unknown_or_ambiguous_server_id_not_joined(self):
        self.assertEqual(analyzer.client_id("cmpl-A-0-abcd1234", {"A"}), "A")
        self.assertIsNone(analyzer.client_id("cmpl-A-1-abcd1234", {"A"}))
        self.assertIsNone(analyzer.client_id("cmpl-A-0-abcd1234", {"A", "cmpl-A-0-abcd1234"}))

    def test_phase_or_epoch_change_excludes_boundary_gap(self):
        case = self.case(count=1)
        events = analyzer.read_capture(case["capture_path"])
        for e in events:
            if e["event"] == "submit" and e.get("phase") == "decode" and e["submit_seq"] >= 6:
                e["epoch"] = 1
        self.write_events(Path(case["results_dir"]), events)
        result = analyzer.analyze_case(case, min_samples=2)
        self.assertEqual(sum(p["sample_count"] for p in result["points"]), 8)

    def test_same_owner_prefill_suspend_resume_is_not_stable_overlap(self):
        case = self.case(count=1)
        events = analyzer.read_capture(case["capture_path"])
        events.extend([dict(event="prefill_boundary", timestamp=2.135, submit_seq=100),
                       dict(event="prefill_resume", timestamp=2.14, submit_seq=100)])
        events.sort(key=lambda r: r["timestamp"])
        self.write_events(Path(case["results_dir"]), events)
        result = analyzer.analyze_case(case, min_samples=2)
        self.assertEqual(sum(p["sample_count"] for p in result["points"]), 8)

    def test_warmup_excluded_from_server_minima_and_transitions(self):
        case = self.case(count=1)
        root = Path(case["results_dir"])
        metadata = json.loads((root / "run_metadata.json").read_text())
        metadata["trace_start_monotonic_ts"] = 2.3
        (root / "run_metadata.json").write_text(json.dumps(metadata))
        r = analyzer.analyze_case(case, min_samples=2)
        self.assertEqual(sum(p["sample_count"] for p in r["points"]), 3)
        self.assertEqual(r["transitions"], [])

    def test_missing_coverage_writes_report_not_fake_env(self):
        case = self.case()
        manifest = dict(cases=[case], splits=[[30, 10]], trace_sha256="abc", slo_sha256="slo")
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(analyzer.analyze(path, min_samples=2), 3)
        self.assertFalse((self.root / "calibration/predictive.env").exists())
        report = json.loads((self.root / "calibration/coverage.json").read_text())
        self.assertIn(["d30_p10", "ring"], report["missing_profile_modes"])

    def test_transition_first_decode_not_prefill_epoch_match(self):
        case = self.case(count=1)
        events = analyzer.read_capture(case["capture_path"])
        events.extend([
            dict(event="transition_request", timestamp=1.7, requested_at=1.7,
                 source_profile="d0_p40", target_profile="d30_p10"),
            dict(event="transition_commit", timestamp=1.8, debug={
                "pd_sm_transition_requested_ts": 1.7,
                "pd_sm_transition_source_profile": "d0_p40",
                "pd_sm_transition_committed_profile": "d30_p10",
                "pd_sm_transition_committed_epoch": 0}),
            dict(event="submit", timestamp=1.9, submit_seq=101, phase="prefill",
                 profile_id="d30_p10", epoch=0, prefills=[], rows=[]),
        ])
        events.sort(key=lambda r: r["timestamp"])
        self.write_events(Path(case["results_dir"]), events)
        r = analyzer.analyze_case(case, min_samples=2)
        self.assertEqual(len(r["transitions"]), 1)
        self.assertAlmostEqual(r["transitions"][0]["lead_ms"], 300)
        self.assertAlmostEqual(r["transitions"][0]["cutover_ms"], 300)
        events = analyzer.read_capture(case["capture_path"])
        next(e for e in events if e["event"] == "transition_commit")["debug"]["pd_sm_transition_committed_epoch"] = 99
        self.write_events(Path(case["results_dir"]), events)
        self.assertEqual(analyzer.analyze_case(case, min_samples=2)["transitions"], [])

    def add_servicing_transition(self, case, mode):
        events = analyzer.read_capture(case["capture_path"])
        for e in events:
            if e["event"] == "submit" and e.get("phase") == "decode" and e["submit_seq"] >= 5:
                e["profile_id"], e["epoch"] = "d32_p8", 1
        events.extend([
            dict(event="transition_request", timestamp=2.13, requested_at=2.13,
                 source_profile="d30_p10", target_profile="d32_p8",
                 prefills=[dict(submit_seq=100, mode=mode, query_tokens=2048)]),
            dict(event="transition_request", timestamp=2.175, requested_at=2.13,
                 source_profile="d30_p10", target_profile="d32_p8", prefills=[]),
            dict(event="transition_commit", timestamp=2.18, debug={
                "pd_sm_transition_requested_ts": 2.13,
                "pd_sm_transition_source_profile": "d30_p10",
                "pd_sm_transition_committed_profile": "d32_p8",
                "pd_sm_transition_committed_epoch": 1}),
        ])
        events.sort(key=lambda r: r["timestamp"])
        self.write_events(Path(case["results_dir"]), events)

    def test_ready_requires_native_and_ring_transition_and_exports_real_bounds(self):
        native = self.case(name="native")
        ring = self.case(mode="ring", name="ring")
        solo = self.case(mode="none", profile="d40_p0", name="solo")
        self.add_servicing_transition(native, "native")
        cases = [native, ring, solo]
        path = self.root / "manifest.json"
        path.write_text(json.dumps(dict(cases=cases, splits=[[30,10]], trace_sha256="abc", slo_sha256="slo")))
        self.assertEqual(analyzer.analyze(path, min_samples=2), 3)
        self.assertFalse((self.root / "calibration/predictive.env").exists())
        self.add_servicing_transition(ring, "ring")
        self.assertEqual(analyzer.analyze(path, min_samples=2), 0)
        report = json.loads((self.root / "calibration/coverage.json").read_text())
        self.assertAlmostEqual(report["calibration_ms"]["lead"], 87.5)
        self.assertAlmostEqual(report["calibration_ms"]["cutover"], 37.5)
        self.assertAlmostEqual(report["calibration_ms"]["min_step"], 25)
        env = (self.root / "calibration/predictive.env").read_text()
        self.assertIn("export PD_DYNAMIC_SM_POLICY=tbt_slo_predictive", env)
        # Real runtime schema reader, without importing CUDA/vLLM package.
        profile_type = runpy.run_path(str(ROOT / "vendor_v8/vllm/v1/engine/pd_tbt_profile.py"))["TBTOverlapProfile"]
        profile = profile_type.from_json_file(self.root / "calibration/tbt_overlap_profile.json")
        self.assertIsNotNone(profile.predict(profile_id="d30_p10", decode_context_tokens=1000,
            decode_requests=2, prefill_mode="ring"))
        self.assertIsNone(profile.predict(profile_id="d30_p10", decode_context_tokens=2000,
            decode_requests=100, prefill_mode="ring"))
        with self.assertRaises(ValueError):
            analyzer.analyze(path, min_samples=2)  # cannot leave/overwrite stale env

    def test_report_write_failure_does_not_publish_environment(self):
        native = self.case(name="native")
        ring = self.case(mode="ring", name="ring")
        solo = self.case(mode="none", profile="d40_p0", name="solo")
        self.add_servicing_transition(native, "native")
        self.add_servicing_transition(ring, "ring")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(cases=[native, ring, solo], splits=[[30, 10]],
                                             trace_sha256="abc", slo_sha256="slo")))
        original_write = Path.write_text

        def fail_report(path, *args, **kwargs):
            if path.name == "coverage.json":
                raise OSError("injected report write failure")
            return original_write(path, *args, **kwargs)

        output = self.root / "new_report"
        with patch.object(Path, "write_text", fail_report), self.assertRaises(OSError):
            analyzer.analyze(manifest, min_samples=2, output_dir=output)
        self.assertFalse((output / "predictive.env").exists())


class DriverTests(unittest.TestCase):
    def test_trace_prefix_preserves_order_and_request_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/"src.jsonl", Path(tmp)/"dst.jsonl"
            lines = [json.dumps(dict(request_id=str(i), prompt="untouched", arrival_offset_s=i*3)) for i in range(3)]
            source.write_text("\n".join(lines)+"\n")
            self.assertEqual(driver.select_trace(source, dest, 2), 2)
            self.assertEqual(dest.read_text(), "\n".join(lines[:2])+"\n")
            with self.assertRaises(FileExistsError):
                driver.select_trace(source, dest, 2)

    def test_manifest_does_not_capture_secrets(self):
        self.assertEqual(driver.recorded_env({"VLLM_API_KEY": "secret", "VLLM_AUTH_TOKEN": "secret",
                                             "VLLM_MAX_NUM_BATCHED_TOKENS": "16384", "HF_TOKEN": "secret"}),
                         {"VLLM_MAX_NUM_BATCHED_TOKENS": "16384"})

    def test_invalid_split_and_output_root(self):
        for spec in ("40/0", "30/9", "31/9", "30/10,30/10"):
            with self.assertRaises(Exception):
                driver.parse_splits(spec)
        with self.assertRaises(SystemExit):
            driver.main(["--trace", "none", "--slo", "none", "--output-dir", str(ROOT), "--dry-run"])

    def test_real_wrapper_dry_run_all_pd_variants(self):
        name = "tbt_calibration_test_no_output"
        self.assertFalse((ROOT / name).exists())
        result = subprocess.run([sys.executable, str(ROOT / "tools/profile_vendor_v8_tbt_predictive.py"),
            "--trace", "/tmp/not-required-trace", "--slo", "/tmp/not-required-slo", "--output-dir", name,
            "--splits", "30/10", "--variants", *driver.VARIANTS, "--dry-run"],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("[PROFILE] 6 cases", result.stdout)
        self.assertIn("max_num_seqs=100", result.stdout)
        self.assertIn("phase_future_depths=1/1/2", result.stdout)
        self.assertIn("tbt_profile=disabled_for_tbt_slo_feedback", result.stdout)
        self.assertFalse((ROOT / name).exists())


if __name__ == "__main__":
    unittest.main()
