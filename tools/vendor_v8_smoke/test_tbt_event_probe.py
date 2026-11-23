#!/usr/bin/env python3
"""CPU-only event-gated calibration: no inference or production state mutation."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import tbt_probe_dispatch as dispatch
import tbt_calibration_plan as plan
import profile_vendor_v8_tbt_predictive as driver
import analyze_vendor_v8_tbt_calibration as analyzer
import test_tbt_controlled_calibration as old
import test_tbt_calibration as fixtures


def options(root, **overrides):
    args = old.args(root)
    args.__dict__.update(probe_protocol="event-gated-v1", gate_decode_tokens=8,
        gate_timeout_sec=600., no_progress_sec=90., native_interference_max_tokens=49152,
        ring_prime_tokens=0, kv_capacity_tokens=65520, transition_base_requests=1,
        transition_extra_requests=2, transition_context_tokens=1024, transition_waves=2,
        transition_burst_output_tokens=32, transition_settle_sec=2., continue_on_unobserved=False,
        splits=[(30, 10), (32, 8)])
    args.__dict__.update(overrides)
    return args


def sources():
    row = dict(old.source_rows()[-1], request_id="source_tail", input_tokens=64310,
               prompt="different longest real prompt", base_scheduled_offset_s=999.)
    return [*old.source_rows(), row]


def observation(rid):
    return NS(request=NS(request_id=rid), effective_offset_s=0., scheduled_ts=0.,
              completion_ts=None, status="planned", error_message="", output_tokens=0,
              token_rows=[])


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "capture"
        self.reader = dispatch.CaptureProgress(self.path, ["p", "d"], 10.)
        self.reader.accept(dict(event="start", contract="tbt-capture-v1", sequence=1, timestamp=1.))

    def test_prefill_gate_requires_recovery_not_submit(self):
        obs = {"p": observation("p")}
        gate = dict(prefill_from="p", prefill_tokens=4096)
        self.reader.accept(dict(event="submit", sequence=2, timestamp=11., submit_seq=3,
            phase="prefill", rows=[dict(request_id="cmpl-p-0-deadbeef", context_tokens=4096)]))
        self.assertEqual(self.reader.gate(gate, obs)[0], "wait")
        self.reader.accept(dict(event="prefill_boundary", sequence=3, timestamp=12., submit_seq=3))
        self.assertEqual(self.reader.gate(gate, obs)[0], "wait")
        self.reader.accept(dict(event="recover", sequence=4, timestamp=13., submit_seq=3))
        self.assertEqual(self.reader.gate(gate, obs)[0], "ready")
        self.assertFalse(self.reader.outstanding)

    def test_finished_decode_never_establishes_live_overlap(self):
        obs = observation("d")
        obs.output_tokens = 100
        gate = dict(tokens_from=["d"], min_tokens=8)
        self.assertEqual(self.reader.gate(gate, {"d": obs})[0], "ready")
        obs.completion_ts = 12.
        self.assertEqual(self.reader.gate(gate, {"d": obs})[0], "failed")

    def test_partial_lines_wait_and_sequence_loss_fails(self):
        record = json.dumps(dict(event="submit", sequence=2, timestamp=11., submit_seq=3,
                                 phase="decode", rows=[])).encode()
        self.path.write_bytes(record[:20])
        self.reader.poll()
        self.assertEqual(self.reader.events, 0)
        with self.path.open("ab") as handle:
            handle.write(record[20:]+b"\n")
        self.reader.poll()
        self.assertEqual(self.reader.events, 1)
        with self.assertRaisesRegex(ValueError, "sequence"):
            self.reader.accept(dict(event="recover", sequence=4, timestamp=12., submit_seq=3))

    def test_stale_completed_mode_cannot_open_transition_gate(self):
        gate = dict(tokens_from=["d"], prefill_mode="ring")
        obs = observation("d")
        obs.output_tokens = 10
        self.reader.modes[3] = "ring"
        self.assertEqual(self.reader.gate(gate, {"d": obs})[0], "wait")
        self.reader.outstanding[3] = {}
        self.assertEqual(self.reader.gate(gate, {"d": obs})[0], "ready")

    def test_invalid_control_identity_and_forward_dependency_rejected(self):
        trace = self.path.with_name("trace")
        trace.write_text("trace")
        payload = dict(schema_version=1, workload_sha256=hashlib.sha256(b"trace").hexdigest(),
                       request_ids=["d", "p"], gate_timeout_sec=1, no_progress_sec=1,
                       poll_sec=.1, require_completion=False, capture_path=str(self.path), gates={})
        control_path = self.path.with_name("control")
        observations = [observation("d"), observation("p")]
        control_path.write_text(json.dumps(payload))
        self.assertEqual(dispatch.load_control(control_path, observations, trace), payload)
        payload["gates"] = {"d": dict(tokens_from=["p"])}
        control_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "earlier"):
            dispatch.load_control(control_path, observations, trace)
        payload["workload_sha256"] = "changed"
        control_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "hash"):
            dispatch.load_control(control_path, observations, trace)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "capture"
        self.sequence = 0
        self.start = time.monotonic()
        self.emit("start", contract="tbt-capture-v1")
        self.control = dict(capture_path=str(self.path), gate_timeout_sec=.2,
                            no_progress_sec=.2, poll_sec=.01, require_completion=False, gates={})

    def emit(self, kind, **kw):
        self.sequence += 1
        with self.path.open("a") as handle:
            handle.write(json.dumps(dict(event=kind, sequence=self.sequence,
                                         timestamp=time.monotonic(), **kw))+"\n")

    async def test_tokens_gate_and_all_done_return_without_full_window(self):
        obs = [observation("d"), observation("p")]
        self.control["gates"] = {"p": dict(tokens_from=["d"], min_tokens=2)}
        order = []

        async def execute(o):
            o.status = "submitted"
            order.append(o.request.request_id)
            if o.request.request_id == "d":
                self.emit("submit", phase="decode", submit_seq=1, rows=[])
                await asyncio.sleep(.025)
                o.output_tokens = 2
                await asyncio.sleep(.06)
                self.emit("recover", submit_seq=1)
            else:
                self.assertEqual(obs[0].output_tokens, 2)
                self.assertIsNone(obs[0].completion_ts)
                await asyncio.sleep(.01)
            o.status, o.completion_ts = "success", time.monotonic()

        result = await dispatch.run_probe(obs, self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+3)
        self.assertTrue(result["healthy"])
        self.assertEqual(result["stop_reason"], "all_requests_complete")
        self.assertEqual(order, ["d", "p"])
        self.assertLess(time.monotonic()-self.start, 1.)

    async def test_expired_decode_skips_background(self):
        obs = [observation("d"), observation("p")]
        self.control["gates"] = {"p": dict(tokens_from=["d"], min_tokens=8)}

        async def execute(o):
            o.output_tokens = 100
            o.status, o.completion_ts = "success", time.monotonic()

        result = await dispatch.run_probe(obs, self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+1)
        self.assertFalse(result["healthy"])
        self.assertIn("decode_finished", result["stop_reason"])
        self.assertEqual(obs[1].status, "not_submitted_probe_control")

    async def test_future_free_stall_cancels_and_awaits_owned_http(self):
        obs = [observation("p")]
        self.control["no_progress_sec"] = .03
        cancelled = []

        async def execute(o):
            o.status = "submitted"
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                o.status = "censored_time_limit"

        result = await dispatch.run_probe(obs, self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+1)
        self.assertEqual(cancelled, [True])
        self.assertFalse(result["healthy"])
        self.assertEqual(result["stop_reason"], "no_model_future_progress")
        self.assertIn("controlled probe cancelled", obs[0].error_message)

    async def test_inflight_model_is_not_declared_future_free_stall(self):
        obs = [observation("p")]
        self.control["no_progress_sec"] = .01

        async def execute(o):
            o.status = "submitted"
            self.emit("submit", submit_seq=7, phase="prefill", rows=[])
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                o.status = "censored_time_limit"

        result = await dispatch.run_probe(obs, self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+.06)
        self.assertEqual(result["stop_reason"], "time_limit")
        self.assertEqual(result["outstanding_submit_seqs"], [7])
        self.assertTrue(result["healthy"])  # time cap alone is not a runtime failure

    async def test_unexpected_task_exception_is_not_success(self):
        async def execute(o):
            raise RuntimeError("unexpected task exception")
        result = await dispatch.run_probe([observation("d")], self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+1)
        self.assertFalse(result["healthy"])
        self.assertEqual(result["stop_reason"], "client_task_ended_without_success")

    async def test_completion_required_timeout_is_invalid(self):
        self.control["require_completion"] = True

        async def execute(o):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                o.status = "censored_time_limit"

        result = await dispatch.run_probe([observation("p")], self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+.03)
        self.assertFalse(result["healthy"])

    async def test_one_failed_task_cannot_hide_behind_another_live_task(self):
        obs = [observation("d"), observation("p")]
        cancelled = []

        async def execute(o):
            o.status = "submitted"
            if o.request.request_id == "d":
                raise RuntimeError("unexpected client failure")
            self.emit("submit", submit_seq=7, phase="prefill", rows=[])
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                o.status = "censored_time_limit"

        result = await dispatch.run_probe(obs, self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+1)
        self.assertFalse(result["healthy"])
        self.assertEqual(result["stop_reason"], "client_task_ended_without_success")
        self.assertEqual(cancelled, [True])
        self.assertLess(time.monotonic()-self.start, .5)

    async def test_gate_timeout_does_not_free_inflight_model(self):
        obs = [observation("d"), observation("p")]
        self.control.update(gate_timeout_sec=.03, no_progress_sec=1.,
                            gates={"p": dict(tokens_from=["d"], min_tokens=8)})

        async def execute(o):
            self.emit("submit", submit_seq=7, phase="decode", rows=[])
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                o.status = "censored_time_limit"

        result = await dispatch.run_probe(obs, self.control, execute=execute,
                                         trace_start=self.start, deadline=self.start+1)
        self.assertIn("gate_timeout", result["stop_reason"])
        self.assertEqual(result["outstanding_submit_seqs"], [7])
        self.assertEqual(obs[1].status, "not_submitted_probe_control")

    async def test_external_cancellation_awaits_owned_tasks(self):
        entered, cleaned = asyncio.Event(), []

        async def execute(o):
            entered.set()
            try:
                await asyncio.sleep(10)
            finally:
                cleaned.append(True)

        task = asyncio.create_task(dispatch.run_probe([observation("p")], self.control,
            execute=execute, trace_start=self.start, deadline=self.start+10))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(cleaned, [True])

    async def test_measurer_persists_control_contract_without_waiting_full_window(self):
        import measure_loongserve_mixed_workload as measure
        import aiohttp
        root = Path(self.tmp.name)
        trace, control_file, output = root / "trace", root / "control", root / "results"
        output.mkdir()
        row = dict(old.source_rows()[0], schema_version=1, event_index=0,
                   tenant_id="d", role="probe", source_id="original", base_scheduled_offset_s=0.,
                   request_id="d", min_tokens=1, max_tokens=1)
        trace.write_text(json.dumps(row)+"\n")
        control = dict(self.control, schema_version=1, request_ids=["d"], require_completion=True,
                       workload_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        control_file.write_text(json.dumps(control))
        with patch.object(sys, "argv", ["measure", "--base-url", "http://mock", "--model", "mock",
                "--workload-jsonl", str(trace), "--case-id", "probe", "--output-dir", str(output),
                "--time-limit-sec", "5", "--time-limit-mode", "cancel",
                "--tbt-probe-control", str(control_file)]):
            args = measure.parse_args()

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        async def execute(o, **kwargs):
            now = time.monotonic()
            o.status, o.submit_ts, o.dispatch_ts = "submitted", now, now
            self.emit("submit", submit_seq=1, phase="prefill",
                      rows=[dict(request_id="d", context_tokens=1024)])
            await asyncio.sleep(.02)
            self.emit("recover", submit_seq=1)
            o.output_tokens, o.first_token_ts = 1, time.monotonic()
            o.completion_ts, o.status = time.monotonic(), "success"
            return o

        with patch.object(aiohttp, "ClientSession", return_value=Session()), \
                patch.object(aiohttp, "TCPConnector", return_value=None), \
                patch.object(measure, "execute_request", side_effect=execute):
            self.assertEqual(await measure.replay(args), 0)
        meta = json.loads((output / "run_metadata.json").read_text())
        self.assertEqual(meta["submission_mode"], "controlled_event_gated_probe")
        self.assertTrue(meta["tbt_probe_health"]["healthy"])
        self.assertEqual(meta["tbt_probe_control_sha256"], hashlib.sha256(control_file.read_bytes()).hexdigest())
        self.assertLess(meta["observation_duration_s"], 1.)


class PlanningTests(unittest.TestCase):
    def test_native_and_ring_are_different_observed_arrival_recipes(self):
        args = options(Path("/tmp"))
        native = plan.event_probe_recipe(sources(), args=args, config=driver.DEFAULTS,
                                        count=1, context=1024, mode="native")
        self.assertEqual(native["rows"][0]["request_id"], "probe_0")
        self.assertEqual(native["control"]["gates"]["background_0"]["tokens_from"], ["probe_0"])
        self.assertLessEqual(native["rows"][-1]["input_tokens"], 49152)
        ring = plan.event_probe_recipe(sources(), args=args, config=driver.DEFAULTS,
                                      count=1, context=1024, mode="ring")
        self.assertEqual(ring["rows"][0]["request_id"], "background_0")
        self.assertEqual(ring["control"]["gates"]["probe_0"]["prefill_tokens"], 61440)
        self.assertEqual(ring["rows"][0]["input_tokens"], 64310)
        self.assertEqual(ring["target"]["request_ids"], ["probe_0"])

    def test_ring_does_not_guess_unknown_capacity(self):
        args = options(Path("/tmp"), kv_capacity_tokens=0)
        result = plan.event_probe_recipe(sources(), args=args, config=driver.DEFAULTS,
                                        count=1, context=1024, mode="ring")
        self.assertEqual(result["plan_status"], "input_unavailable")

    def test_transition_wave_requires_live_base_observed_mode_and_prior_completion(self):
        result = plan.event_probe_recipe(sources(), args=options(Path("/tmp")),
                                        config=driver.DEFAULTS, count=1, context=1024,
                                        mode="native", transitions=True)
        gates = result["control"]["gates"]
        self.assertEqual(gates["burst_0_0"]["tokens_from"], ["probe_0"])
        self.assertEqual(gates["burst_0_0"]["prefill_mode"], "native")
        self.assertEqual(gates["burst_1_0"]["complete_from"], ["burst_0_0", "burst_0_1"])
        self.assertEqual(len({r["request_id"] for r in result["rows"]}), len(result["rows"]))

    def test_preflight_clock_and_other_warmups_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(driver, "ROOT", root):
                cases = driver.plan_controlled_cases(options(root), {}, sources())
        first = cases[0]
        self.assertEqual(first["kind"], "liveness")
        self.assertEqual(first["env"]["WARMUP_COUNT"], "0")
        self.assertEqual(first["env"]["ITERATION_TRACE_SAMPLE_EVERY_N"], "1")
        self.assertTrue(first["control"]["require_completion"])
        self.assertTrue(all(c["env"]["WARMUP_COUNT"] == "1" for c in cases[1:]))
        self.assertTrue(all(c["concurrent_transitions_only"] for c in cases if c["kind"] == "transitions"))

    def test_cli_plan_and_wrapper_dry_run_create_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace, slo = root / "trace", root / "slo"
            trace.write_text("".join(json.dumps(r)+"\n" for r in sources()))
            slo.write_text("{}")
            output = ROOT / "tbt_event_gated_cpu_no_output"
            command = [sys.executable, str(ROOT / "tools/profile_vendor_v8_tbt_predictive.py"),
                       "--trace", str(trace), "--slo", str(slo), "--output-dir", str(output),
                       "--mode", "calibrate", "--probe-protocol", "event-gated-v1",
                       "--decode-batches", "1", "--decode-contexts", "1024",
                       "--splits", "30/10,32/8", "--kv-capacity-tokens", "65520"]
            for flag in ("--plan-only", "--dry-run"):
                result = subprocess.run([*command, flag], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn("planned=8 runnable=8", result.stdout)
                self.assertFalse(output.exists())

    def test_unhealthy_preflight_stops_before_second_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = options(root, continue_on_unobserved=True)
            args.trace.write_text("".join(json.dumps(r)+"\n" for r in sources()))
            args.slo.write_text("{}")
            with patch.object(driver, "ROOT", root), \
                    patch.object(driver, "runtime_hashes", return_value={"core": "version"}), \
                    patch.object(driver.subprocess, "run", return_value=NS(stdout="GPU", stderr="")), \
                    patch.object(driver.subprocess, "Popen", return_value=NS(wait=lambda: 0)) as launch, \
                    patch.object(driver, "analyze_case", return_value=dict(
                        case_id="r1_full_vendor_v8_liveness", points=[], transitions=[],
                        health=dict(healthy=False, stop_reason="no_model_future_progress"))), \
                    patch.object(driver, "analyze", return_value=3):
                self.assertEqual(driver.run_controlled(args, {"VLLM_VENDOR_DIR": str(ROOT / "vendor_v8")}), 3)
            self.assertEqual(launch.call_count, 1)
            manifest = json.loads((args.output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["cases"][0]["measurement_status"]["status"], "invalid_probe")
            self.assertTrue(all(c["returncode"] is None for c in manifest["cases"][1:]))


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AnalyzerTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_unhealthy_capture_cannot_produce_deployable_points(self):
        case = self.fixture.case()
        case.update(control_path="control.json", control_sha256="control-hash")
        meta = Path(case["results_dir"]) / "run_metadata.json"
        data = json.loads(meta.read_text())
        data.update(submission_mode="controlled_event_gated_probe", tbt_probe_control_sha256="control-hash",
                    tbt_probe_health=dict(healthy=False, stop_reason="no_model_future_progress"))
        meta.write_text(json.dumps(data))
        report = analyzer.analyze_case(case, min_samples=2)
        self.assertFalse(report["points"])
        self.assertFalse(report["transitions"])
        rows, complete = plan.coverage_for_cases([dict(case, kind="steady")], [report],
                                                min_transition_samples=2)
        self.assertFalse(complete)
        self.assertEqual(rows[0]["status"], "invalid_probe")

    def test_missing_new_client_contract_fails_closed(self):
        case = dict(self.fixture.case(), control_path="new.json", control_sha256="hash")
        with self.assertRaisesRegex(ValueError, "health contract"):
            analyzer.analyze_case(case, min_samples=2)

    def test_p95_kept_and_distribution_fields_added(self):
        point = analyzer.analyze_case(self.fixture.case(), min_samples=2)["points"][0]
        self.assertLessEqual(point["p50_per_token_tbt_ms"], point["p95_per_token_tbt_ms"])
        self.assertLessEqual(point["p99_per_token_tbt_ms"], point["max_per_token_tbt_ms"])

    def test_endpoint_transition_cannot_satisfy_concurrent_retune_requirement(self):
        case = dict(case_id="t", kind="transitions", transition_modes=["native"],
                    transition_directions=["increase"], concurrent_transitions_only=True)
        report = dict(case_id="t", transitions=[dict(source="d30_p10", target="d40_p0",
                        source_prefill_mode="native", source_had_decode_service=True)]*2)
        self.assertFalse(plan.coverage_for_cases([case], [report], min_transition_samples=2)[1])
        report["transitions"][0] = dict(report["transitions"][0], target="d32_p8")
        self.assertTrue(plan.coverage_for_cases([case], [report], min_transition_samples=1)[1])


if __name__ == "__main__":
    unittest.main()
