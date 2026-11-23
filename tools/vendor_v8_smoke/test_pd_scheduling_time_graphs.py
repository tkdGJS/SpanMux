"""CPU-only contracts for sampled scheduling-time plots."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pd_sched_graphs", ROOT / "graph_script/generate_pd_scheduling_time_graphs.py")
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def row(start=11.0, duration=0.002, phase="decode"):
    return {"schedule_start_ts": start, "schedule_end_ts": start + duration,
            "schedule_duration_s": duration, "scheduled_phase": phase, "iteration_id": 100}


def fixture(directory, variant, rows=None):
    results = Path(directory) / f"sample_workload_loongserve_{variant}_results"
    (results / "client").mkdir(parents=True)
    (results / "experiment_snapshot").mkdir()
    (results / "client/run_metadata.json").write_text(json.dumps({
        "trace_start_monotonic_ts": 10, "observation_end_monotonic_ts": 20,
        "model": "test-model", "workload_sha256": "fixture-trace",
        "arrival_time_scale": 1, "termination_mode": "requests", "effective_request_limit": 100,
    }))
    (results / "experiment_snapshot/settings.txt").write_text("iteration_trace_sample_every_n=100\n")
    if rows is not None:
        artifacts = results.with_name(results.name.removesuffix("_results") + "_artifacts")
        artifacts.mkdir()
        (artifacts / "iteration_metrics.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in rows))
    return results


class SchedulingGraphs(unittest.TestCase):
    def call_trace(self, directory, rows=None, status=None):
        results = fixture(directory, "native_40sm_chunked", [row(duration=.090)])
        artifact = results.with_name(results.name.removesuffix("_results") + "_artifacts")
        payload = [{**row(), "schema": "scheduler_call_v1", "pid": 123, "sample_every_n": 100}] if rows is None else rows
        path = artifact / "scheduler_metrics.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in payload))
        if status is not None:
            Path(str(path) + ".status.json").write_text(json.dumps(status))
        return results

    def test_new_call_trace_preferred_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            results = self.call_trace(directory, status={
                "schema": "scheduler_call_v1", "pid": 123, "written": 1,
                "dropped": 0, "complete": True,
            })
            run = G.read_run(Path(directory), G.BASELINE_CASE, results_dir=results)
            self.assertEqual(run["measurement_population"], "schedule_call")
            self.assertEqual(run["summary"][0]["mean_ms"], 2)
            self.assertEqual(run["raw_rows"], 1)
            self.assertEqual(run["trace_health"], "complete")

    def test_missing_close_status_does_not_claim_complete_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            results = self.call_trace(directory)
            run = G.read_run(Path(directory), G.BASELINE_CASE, results_dir=results)
            self.assertEqual(run["trace_health"], "status_missing")
            self.assertIsNone(run["writer_status"])

    def test_new_trace_mismatch_rejected(self):
        for field, value in (("schema", "stale"), ("sample_every_n", 2), ("pid", None)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                payload = {**row(), "schema": "scheduler_call_v1", "pid": 123, "sample_every_n": 100, field: value}
                results = self.call_trace(directory, [payload])
                with self.assertRaises(ValueError):
                    G.read_run(Path(directory), G.BASELINE_CASE, results_dir=results)

    def test_dropped_and_incomplete_are_explicit(self):
        for complete, expected in ((True, "complete_with_drops"), (False, "incomplete")):
            with tempfile.TemporaryDirectory() as directory:
                results = self.call_trace(directory, status={
                    "schema": "scheduler_call_v1", "pid": 123, "written": 1,
                    "complete": complete, "dropped": 4,
                })
                run = G.read_run(Path(directory), G.BASELINE_CASE, results_dir=results)
                self.assertEqual(run["trace_health"], expected)
                self.assertEqual(run["writer_status"]["dropped"], 4)

    def test_population_mixing_is_not_comparable(self):
        a = {"cohort": {"scale": 1}, "sample_every_n": 100, "measurement_population": "schedule_call"}
        b = {**a, "measurement_population": "recovered_iteration"}
        with self.assertRaisesRegex(ValueError, "populations"):
            G.validate_cohort([a, b])
        G.validate_cohort([a, {**b, "data_status": "missing_iteration_trace"}])

    def test_idle_and_error_calls_have_their_own_phase(self):
        samples, _ = G.extract_samples([row(phase="idle"), row(phase="error")], 10, 20, "X")
        self.assertEqual(G.summarize(samples, "X", "idle")["sample_count"], 1)
        self.assertEqual(G.summarize(samples, "X", "decode")["sample_count"], 0)

    def test_enabled_but_missing_trace_does_not_fall_back_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            results = fixture(directory, "native_40sm_chunked", [row()])
            (results / "experiment_snapshot/settings.txt").write_text(
                "iteration_trace_sample_every_n=100\nscheduler_timing_trace=1\n")
            with self.assertRaisesRegex(FileNotFoundError, "was not produced"):
                G.read_run(Path(directory), G.BASELINE_CASE, results_dir=results)

    def test_render_common_call_population_including_baseline(self):
        import importlib.util
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for spec in (G.BASELINE_CASE, G.BASELINE_SRPT_CASE, *G.CASES):
                result = fixture(directory, spec[0], [row()])
                sources[spec[0]] = result
                artifact = result.with_name(result.name.removesuffix("_results") + "_artifacts")
                path = artifact / "scheduler_metrics.jsonl"
                rows = [{**row(phase=p), "schema": "scheduler_call_v1", "pid": 123, "sample_every_n": 100}
                        for p in ("prefill", "decode", "mixed", "idle")]
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = G.render(root, root / "figures",
                                baseline_results_dir=sources[G.BASELINE_CASE[0]],
                                baseline_srpt_results_dir=sources[G.BASELINE_SRPT_CASE[0]])
            self.assertEqual(len(manifest["runs"]), 5)
            self.assertIn("timeline_baseline.png", manifest["images"])
            self.assertTrue(all((root / "figures" / name).stat().st_size > 1000 for name in manifest["images"]))

    def test_window_and_units(self):
        rows, excluded = G.extract_samples([row(9), row(11), row(19.999)], 10, 20, "X")
        self.assertEqual(excluded, {"before_origin": 1, "after_window": 1})
        self.assertEqual(rows[0]["time_s"], 1)
        self.assertEqual(rows[0]["schedule_ms"], 2)

    def test_timestamp_agreement(self):
        value = row()
        value["schedule_duration_s"] = 0.02
        with self.assertRaisesRegex(ValueError, "disagrees"):
            G.extract_samples([value], 10, 20, "X")

    def test_bad_numbers_and_missing(self):
        for value in (None, float("nan"), float("inf"), -1, True):
            with self.subTest(value=value):
                payload = row()
                payload["schedule_duration_s"] = value
                with self.assertRaises(ValueError):
                    G.extract_samples([payload], 10, 20, "X")

    def test_zero_preserved_unknown_phase_rejected(self):
        samples, _ = G.extract_samples([row(duration=0)], 10, 20, "X")
        self.assertEqual(samples[0]["schedule_ms"], 0)
        with self.assertRaises(ValueError):
            G.extract_samples([row(phase="unknown")], 10, 20, "X")

    def test_empty_production_rejected(self):
        with self.assertRaises(ValueError):
            G.extract_samples([row(9)], 10, 20, "X")

    def test_phase_statistics_and_percentiles(self):
        samples, _ = G.extract_samples([row(duration=.001), row(duration=.003),
                                       row(duration=.010, phase="prefill")], 10, 20, "X")
        summary = G.summarize(samples, "X", "decode")
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["mean_ms"], 2)
        self.assertAlmostEqual(summary["p95_ms"], 2.9)
        self.assertIsNone(G.summarize(samples, "X", "mixed")["mean_ms"])

    def test_cohort_and_sampling_mismatch(self):
        a = {"cohort": {"scale": 1}, "sample_every_n": 100}
        G.validate_cohort([a, dict(a)])
        with self.assertRaises(ValueError):
            G.validate_cohort([a, {**a, "cohort": {"scale": 2}}])
        with self.assertRaises(ValueError):
            G.validate_cohort([a, {**a, "sample_every_n": 1}])

    def test_no_overwrite_before_input_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                G.render(Path("/nonexistent/input"), Path(directory))

    def test_explicit_baseline_path_and_color(self):
        with tempfile.TemporaryDirectory() as directory:
            results = fixture(directory, "native_40sm_chunked", [row()])
            run = G.read_run(Path("/not-used"), G.BASELINE_CASE, results_dir=results)
            self.assertEqual(run["label"], "Baseline")
            self.assertEqual(run["color"], "#1F77B4")
            self.assertEqual(run["data_status"], "observed")
            self.assertEqual(len(run["source_hashes"]), 3)
            self.assertEqual(run["summary"][0]["mean_ms"], 2)

    def test_missing_trace_requires_opt_in_and_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            results = fixture(directory, "native_40sm_chunked")
            with self.assertRaises(FileNotFoundError):
                G.read_run(Path(directory), G.BASELINE_CASE, results_dir=results)
            run = G.read_run(Path(directory), G.BASELINE_CASE, results_dir=results,
                             allow_missing_iterations=True)
            self.assertEqual(run["data_status"], "missing_iteration_trace")
            self.assertEqual(run["samples"], [])
            self.assertIsNone(run["raw_rows"])
            self.assertIsNone(run["excluded"])
            self.assertEqual(len(run["missing_inputs"]), 1)
            self.assertEqual(len(run["source_hashes"]), 2)
            for summary in run["summary"]:
                self.assertEqual(summary["sample_count"], 0)
                self.assertIsNone(summary["mean_ms"])
                self.assertIsNone(summary["max_ms"])

    def test_malformed_or_empty_existing_trace_is_not_missing(self):
        for rows in ([], [row(duration=-0.1)], [{**row(), "schedule_duration_s": None}]):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                results = fixture(directory, "baseline_plus", rows)
                with self.assertRaises(ValueError):
                    G.read_run(Path(directory), G.BASELINE_SRPT_CASE, results_dir=results,
                               allow_missing_iterations=True)

    def test_mislabeled_result_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not identify"):
            G.read_run(Path("/not-used"), G.BASELINE_CASE,
                       results_dir=Path("sample_workload_loongserve_baseline_plus_results"),
                       allow_missing_iterations=True)

    def test_missing_required_metadata_is_still_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                G.read_run(Path(directory), G.BASELINE_CASE,
                           results_dir=Path(directory) / "sample_native_40sm_chunked_results",
                           allow_missing_iterations=True)

    def test_mixed_samples_count_once_not_both_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            results = fixture(directory, "baseline_plus", [
                row(duration=.001, phase="prefill"), row(duration=.002, phase="decode"),
                row(duration=.009, phase="mixed"),
            ])
            run = G.read_run(Path(directory), G.BASELINE_SRPT_CASE, results_dir=results)
            stats = {value["phase"]: value for value in run["summary"]}
            self.assertEqual(run["color"], "#2CA02C")
            self.assertEqual(stats["all"]["sample_count"], 3)
            self.assertEqual(stats["all"]["mean_ms"], 4)
            self.assertEqual(stats["prefill"]["sample_count"], 1)
            self.assertEqual(stats["decode"]["sample_count"], 1)
            self.assertEqual(stats["mixed"]["sample_count"], 1)
            self.assertEqual(stats["mixed"]["mean_ms"], 9)

    def test_all_missing_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = {"cohort": {"model": "test"}, "sample_every_n": 100, "samples": []}
            destination = Path(directory) / "output"
            with patch.object(G, "read_run", return_value=missing):
                with self.assertRaisesRegex(ValueError, "No observed"):
                    G.render(Path(directory), destination, allow_missing_iterations=True)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
