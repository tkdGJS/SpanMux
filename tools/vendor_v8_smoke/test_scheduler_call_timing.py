"""CPU-only runtime/launcher contracts for common Native/vendor scheduling time."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
REL = "vllm/v1/core/sched/isolation_scheduler_timing.py"
NATIVE = ROOT / "src/vllm-0.14-src" / REL
VENDOR = ROOT / "vendor_v8" / REL
SPEC = importlib.util.spec_from_file_location("scheduler_timing", NATIVE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class FakeScheduler:
    def __init__(self, counts=None, prompt=8, computed=7):
        self.counts = {"P": 1} if counts is None else counts
        self.requests = {rid: NS(num_prompt_tokens=prompt, num_computed_tokens=computed)
                         for rid in self.counts}
        self.output = NS(num_scheduled_tokens=self.counts,
                         total_num_scheduled_tokens=sum(self.counts.values()))
        self.executed = 0

    def schedule(self):
        self.executed += 1
        for rid, count in self.counts.items():
            self.requests[rid].num_computed_tokens += count
        return self.output

    def schedule_pd_phase(self, phase):
        return self.schedule()


class SchedulerCallTiming(unittest.TestCase):
    def test_mirrored_implementation(self):
        self.assertEqual(NATIVE.read_bytes(), VENDOR.read_bytes())

    def test_default_off_has_no_clock_thread_or_instance_override(self):
        scheduler = FakeScheduler()
        with patch.dict(os.environ, {}, clear=True), patch.object(M, "SchedulerTimingRecorder") as factory:
            M.install_scheduler_timing(scheduler)
            self.assertNotIn("schedule", scheduler.__dict__)
            self.assertIs(scheduler.schedule(), scheduler.output)
            factory.assert_not_called()
            M.close_scheduler_timing(scheduler)

    def test_same_boundary_sampling_and_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheduler_metrics.jsonl"
            scheduler = FakeScheduler()
            with patch.dict(os.environ, {"VLLM_ISOLATION_SCHEDULER_TIMING_JSONL": str(path),
                                        "VLLM_ITERATION_METRICS_SAMPLE_EVERY_N": "2"}):
                M.install_scheduler_timing(scheduler)
            for _ in range(5):
                self.assertIs(scheduler.schedule_pd_phase("prefill"), scheduler.output)
            M.close_scheduler_timing(scheduler)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["iteration_id"] for r in rows], [0, 2, 4])
            self.assertEqual(rows[0]["scheduled_phase"], "prefill")
            self.assertEqual(rows[1]["scheduled_phase"], "decode")
            for r in rows:
                self.assertAlmostEqual(r["schedule_duration_s"], r["schedule_end_ts"] - r["schedule_start_ts"])
                self.assertTrue(all(value is None or isinstance(value, (str, int, float)) for value in r.values()))
            status = json.loads(Path(str(path) + ".status.json").read_text())
            self.assertEqual((status["calls"], status["sampled"], status["written"]), (5, 3, 3))
            self.assertTrue(status["complete"])

    def test_mixed_and_idle_and_vendor_explicit_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = M.SchedulerTimingRecorder(str(Path(directory) / "trace"), every_n=1)
            mixed = FakeScheduler({"P": 1, "D": 2})
            mixed.requests["D"].num_computed_tokens = 20
            recorder.wrap(mixed.schedule)()
            idle = FakeScheduler({})
            recorder.wrap(idle.schedule)()
            vendor = FakeScheduler(computed=100)
            vendor.output.scheduled_phase = "prefill"
            recorder.wrap(vendor.schedule)()
            recorder.close()
            rows = [json.loads(line) for line in Path(recorder.path).read_text().splitlines()]
            self.assertEqual([r["scheduled_phase"] for r in rows], ["mixed", "idle", "prefill"])
            self.assertEqual([r["scheduled_num_seqs"] for r in rows], [2, 0, 1])

    def test_original_exception_propagates_and_is_timed(self):
        class Broken:
            def schedule(self):
                raise ValueError("original")
        with tempfile.TemporaryDirectory() as directory:
            recorder = M.SchedulerTimingRecorder(str(Path(directory) / "trace"), every_n=1)
            with self.assertRaisesRegex(ValueError, "original"):
                recorder.wrap(Broken().schedule)()
            recorder.close()
            row = json.loads(Path(recorder.path).read_text())
            self.assertEqual(row["schedule_status"], "error")
            self.assertEqual(row["scheduled_phase"], "error")
            self.assertEqual(row["error_type"], "ValueError")

    def test_overflow_drops_instead_of_waiting(self):
        # Disable the consumer deterministically; no sleeps/large memory load.
        with patch.object(M.threading.Thread, "start"):
            recorder = M.SchedulerTimingRecorder("unused", every_n=1, max_pending=1)
        method = recorder.wrap(FakeScheduler().schedule)
        method()
        method()
        self.assertEqual(recorder.dropped, 1)
        self.assertEqual(recorder._queue.qsize(), 1)

    def test_io_failure_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing"
            path.write_text("preserve")
            with self.assertLogs(M.logger, level="ERROR"):
                recorder = M.SchedulerTimingRecorder(str(path), every_n=1)
                scheduler = FakeScheduler()
                self.assertIs(recorder.wrap(scheduler.schedule)(), scheduler.output)
                recorder.close()
            self.assertEqual(path.read_text(), "preserve")
            self.assertEqual(recorder.errors, 1)
            status = json.loads(Path(str(path) + ".status.json").read_text())
            self.assertFalse(status["complete"])

    def test_bounded_close(self):
        with patch.object(M.threading.Thread, "start"):
            recorder = M.SchedulerTimingRecorder("unused")
        with patch.object(recorder._thread, "join") as join, patch.object(recorder._thread, "is_alive", return_value=True):
            with self.assertLogs(M.logger, level="WARNING"):
                recorder.close(timeout_s=0.01)
            join.assert_called_once_with(timeout=0.01)
        self.assertTrue(recorder.close_timed_out)

    def test_bad_sampling_rejected_before_thread_start(self):
        with patch.object(M.threading.Thread, "start") as start:
            with self.assertRaises(ValueError):
                M.SchedulerTimingRecorder("unused", every_n=0)
            start.assert_not_called()

    def test_metadata_failure_never_changes_return_value(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = M.SchedulerTimingRecorder(str(Path(directory) / "trace"), every_n=1)
            scheduler = FakeScheduler()
            scheduler.output = object()
            with self.assertLogs(M.logger, level="ERROR"):
                self.assertIs(recorder.wrap(scheduler.schedule)(), scheduler.output)
            recorder.close()
            self.assertEqual(recorder.metadata_errors, 1)
            self.assertEqual(recorder.dropped, 1)

    def test_scheduler_integration_at_constructor_end_and_shutdown_start(self):
        for root in ("src/vllm-0.14-src", "vendor_v8"):
            path = ROOT / root / "vllm/v1/core/sched/scheduler.py"
            cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
            methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            self.assertEqual(ast.unparse(methods["__init__"].body[-1]), "install_scheduler_timing(self)")
            self.assertEqual(ast.unparse(methods["shutdown"].body[0]), "close_scheduler_timing(self)")
            async_path = path.with_name("async_scheduler.py")
            self.assertNotIn("def schedule(", async_path.read_text())

    def test_deployment_probe_both_roots_and_stale_source(self):
        import re
        import sys
        import types
        source = (ROOT / "run_vendor_v8_short_validation.sh").read_text()
        probe = re.search(r"SCHEDULER_TIMING_PROBE='(.*?)'\n", source, re.S).group(1)
        for root in ("src/vllm-0.14-src", "vendor_v8"):
            fake = types.ModuleType("vllm")
            fake.__file__ = str(ROOT / root / "vllm/__init__.py")
            with patch.dict(sys.modules, {"vllm": fake}), patch("builtins.print"):
                exec(probe, {})
        with tempfile.TemporaryDirectory() as directory:
            fake.__file__ = str(Path(directory) / "vllm/__init__.py")
            with patch.dict(sys.modules, {"vllm": fake}), self.assertRaisesRegex(SystemExit, "missing common scheduler"):
                exec(probe, {})


if __name__ == "__main__":
    unittest.main()
