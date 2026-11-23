#!/usr/bin/env python3
"""CPU-only contracts for archived capacity evidence, not runtime behavior."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graph_script"))
import generate_loongserve_capacity_evidence as g


class CapacityEvidenceTest(unittest.TestCase):
    def fixture(self, root: Path, key: str) -> Path:
        stem = "fixture_" + g.VARIANTS[key]
        result = root / (stem + "_results")
        artifacts = root / (stem + "_artifacts")
        logs = root / (stem + "_logs")
        for p in (result / "client", result / "experiment_snapshot", artifacts, logs):
            p.mkdir(parents=True)
        origin = 1000.0 if key == "fcfs" else 2000.0
        meta = {"case_id": stem, "workload_sha256": "same-trace",
                "trace_start_monotonic_ts": origin, "observation_duration_s": 10.0,
                "observation_end_monotonic_ts": origin + 10,
                "summary": {"submitted_count": 2}}
        (result / "client/run_metadata.json").write_text(json.dumps(meta))
        server, clients = [], []
        for index, times in enumerate(((0.1, 1, 2, 4), (0.5, 4.01, 5, 6))):
            q, s, f, e = times
            request_id = f"request-{index + 1}"
            server.append({"session_id": request_id, "queued_ts": origin + q,
                           "scheduled_ts": origin + s, "first_token_ts": origin + f,
                           "last_token_ts": origin + e, "prompt_tokens": 16,
                           "completion_tokens": 3, "queued_time_s": s - q,
                           "prefill_time_s": f - s, "decode_time_s": e - f})
            clients.append({"request_id": request_id, "event_index": index,
                            "dataset": "sharegpt", "source_id": str(index),
                            "input_tokens": 16, "requested_output_tokens": 3,
                            "output_tokens": 3, "submission_status": "submitted",
                            "status": "success", "base_scheduled_offset_s": index / 2,
                            "effective_scheduled_offset_s": index / 2,
                            "ttft_ms": 1000 * (f - q)})
        with (result / "client/request_results.csv").open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(clients[0]))
            w.writeheader()
            w.writerows(clients)
        # Warmup deliberately reuses a submitted session ID.
        warmup = dict(server[0], queued_ts=origin - 1)
        (artifacts / "request_metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in [warmup, *server]) + "\n")
        sample = {"scheduled_ts": origin + 2.5, "iteration_id": 100,
                  "pd_capacity_retry_gate_active": int(key == "fcfs"),
                  "pd_capacity_retry_gate_phases": "prefill",
                  "pd_capacity_retry_gate_causes": "prefill_kv_capacity",
                  "pd_capacity_retry_gate_current_free_blocks": 10,
                  "pd_capacity_retry_gate_min_free_blocks": 20,
                  "pd_dynamic_sm_decode_sms": 40, "pd_dynamic_sm_prefill_sms": 0,
                  "num_waiting_prefill_reqs": 1,
                  "kv_retention_execution_plan_count": int(key == "srpt"),
                  "kv_retention_execution_plan_modes": "context_ring" if key == "srpt" else ""}
        if key == "fcfs":
            sample["virtual_context_enabled"] = 0
        samples = [dict(sample, scheduled_ts=origin - 0.2, iteration_id=0), sample,
                   dict(sample, scheduled_ts=origin + 8, iteration_id=200,
                        pd_capacity_retry_gate_active=0)]
        (artifacts / "iteration_metrics.jsonl").write_text(
            "\n".join(json.dumps(s) for s in samples) + "\n")
        (result / "experiment_snapshot/settings.txt").write_text("iteration_trace_sample_every_n=100\n")
        (logs / "server.log").write_text(
            "pd_eval_context_mode=" + ("none" if key == "fcfs" else "virtual_context_context_ring_optimized")
            + "\n")
        return result

    def test_warmup_join_clock_union_and_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [g.load_case(self.fixture(Path(tmp), key), key) for key in g.VARIANTS]
            g.validate_pair(cases)
            for c in cases:
                self.assertEqual(len(c.requests), 2)
                self.assertEqual(len(c.samples), 2)
                self.assertEqual(c.samples[0]["source_line"], 2)
                self.assertAlmostEqual(c.requests[0]["queue_wait_s"], 0.9)
                self.assertAlmostEqual(c.requests[0]["post_start_prefill_span_s"], 1)
                self.assertEqual(g.summarize(c)["decode_service_span_union_s"], 3)
                self.assertIsNone(g.summarize(c)["stall_duration_s"])
                # Even positive scheduler Ring plans do NOT verify backend execution.
                self.assertIsNone(g.summarize(c)["actual_ring_execution_verified"])
            self.assertEqual(cases[0].samples[0]["virtual_context_enabled"], 0)
            self.assertIsNone(cases[1].samples[0]["virtual_context_enabled"])
            e = g.gate_examples(cases[0])[0]
            self.assertIsNone(e["blocked_duration_s"])
            self.assertIsNone(e["blocked_target_request_id"])
            self.assertAlmostEqual(e["next_schedule_minus_active_decode_end_s"], 0.01)
            self.assertEqual(g.gate_examples(cases[1]), [])
        self.assertEqual(g.union_duration([(3, 6), (1, 4), (8, 9), (2, 3)]), 6)
        self.assertEqual(g.union_duration([]), 0)
        with self.assertRaises(ValueError):
            g.union_duration([(4, 3)])

    def test_bad_join_timestamp_token_count_and_pair_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fcfs = self.fixture(root, "fcfs")
            srpt = self.fixture(root, "srpt")
            server = root / (fcfs.name.removesuffix("_results") + "_artifacts/request_metrics.jsonl")
            original = server.read_text()
            rows = [json.loads(s) for s in original.splitlines()]
            corruptions = [[*rows, rows[1]], rows[:2],
                           [rows[0], dict(rows[1], scheduled_ts=1005), rows[2]],
                           [rows[0], dict(rows[1], completion_tokens=9), rows[2]],
                           [rows[0], dict(rows[1], queued_ts=float("nan")), rows[2]]]
            for corrupt in corruptions:
                server.write_text("\n".join(json.dumps(r) for r in corrupt))
                with self.assertRaises(ValueError):
                    g.generate(fcfs, srpt, root / "new")
                self.assertFalse((root / "new").exists())
            server.write_text(original)
            pair = [g.load_case(fcfs, "fcfs"), g.load_case(srpt, "srpt")]
            pair[1].requests[0]["effective_offset_s"] = 99
            with self.assertRaises(ValueError):
                g.validate_pair(pair)
            pair[1].requests[0]["effective_offset_s"] = 0
            pair[1].metadata["workload_sha256"] = "different"
            with self.assertRaises(ValueError):
                g.validate_pair(pair)
            with self.assertRaises(ValueError):
                g.load_case(root / "full_vendor_v8_op_results", "srpt")

    def test_render_one_axes_sample_only_color_alpha_and_no_overwrite(self):
        import matplotlib.figure
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex

        seen = []

        def capture(figure, path, **kwargs):
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            filename = Path(path).name
            seen.append(filename)
            if "lifecycle" in filename:
                color = g.COLORS["fcfs" if "fcfs" in filename else "srpt"]
                self.assertTrue(any(c.get_alpha() == 0.30 for c in axis.collections))
                faces = [to_hex(c.get_facecolors()[0]).upper() for c in axis.collections]
                self.assertIn(color, faces)
                self.assertEqual(axis.get_xlim(), (0, 10))
            if "gate_samples" in filename:
                self.assertEqual(len(axis.lines), 0)  # no interpolated stall intervals

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fcfs, srpt = [self.fixture(root, key) for key in g.VARIANTS]
            output = root / "plots"
            with mock.patch.object(matplotlib.figure.Figure, "savefig", autospec=True,
                                   side_effect=capture):
                report = g.generate(fcfs, srpt, output)
            self.assertEqual(len(seen), 6)
            self.assertEqual(seen, report["charts"])
            self.assertEqual(plt.get_fignums(), [])
            self.assertTrue(report["source_hashes_unchanged"])
            with (output / "iteration_samples.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["virtual_context_enabled"], "0")
            self.assertEqual(rows[2]["virtual_context_enabled"], "")
            self.assertIn("GPU", (output / "README.md").read_text())
            with self.assertRaises(FileExistsError):
                g.generate(fcfs, srpt, output)
            with self.assertRaises(FileExistsError):
                g.generate(fcfs, srpt, root)


if __name__ == "__main__":
    unittest.main()
