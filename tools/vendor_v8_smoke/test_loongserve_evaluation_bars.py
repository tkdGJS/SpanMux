"""CPU contracts for explicit-cohort historical SLO/throughput reporting."""

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from graph_script import generate_loongserve_evaluation_bars as report


def fixture(path):
    client = path / "client"
    client.mkdir(parents=True)
    (path / "experiment_snapshot").mkdir()
    (path / "experiment_snapshot/request_slo_policy.json").write_text('{}')
    (client / "run_metadata.json").write_text(json.dumps({
        "observation_duration_s": 120, "termination_mode": "requests",
        "effective_request_limit": 3, "workload_sha256": "test-trace",
        "arrival_time_scale": 1,
    }))
    requests, tokens = [], []
    for i, dataset in enumerate(("sharegpt", "leval", "lveval")):
        rid = f"r{i}"
        requests.append({
            "event_index": str(i), "request_id": rid, "dataset": dataset,
            "source_id": str(i), "input_tokens": str(10 * (i + 1)),
            "output_tokens": "2", "requested_output_tokens": "2",
            "ttft_slo_met": "1" if i < 2 else "0", "ttft_slo_ms": "100",
            "tbt_slo_ms": "10", "effective_scheduled_offset_s": str(i),
            "submit_monotonic_ts": str(100 + i), "status": "success",
        })
        tokens.extend([
            {"request_id": rid, "dataset": dataset, "chunk_tokens": "1",
             "tbt_slo_met": "", "estimated_per_token_gap_ms": "", "tbt_slo_ms": "10"},
            {"request_id": rid, "dataset": dataset, "chunk_tokens": "1",
             "tbt_slo_met": "1", "estimated_per_token_gap_ms": "5", "tbt_slo_ms": "10"},
        ])
    for name, rows in (("request_results.csv", requests), ("client_token_timeline.csv", tokens)):
        with (client / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return requests, tokens


class EvaluationBarsTests(unittest.TestCase):
    def test_weighted_pool_first_token_and_full_duration(self):
        rows = [dict(request_id="a", dataset="leval", ttft_slo_met="1", input_tokens="100", output_tokens="2"),
                dict(request_id="b", dataset="lveval", ttft_slo_met="0", input_tokens="200", output_tokens="10"),
                dict(request_id="c", dataset="sharegpt", ttft_slo_met="1", input_tokens="3", output_tokens="2")]
        tokens = [dict(request_id="a", chunk_tokens="1", tbt_slo_met=""),
                  dict(request_id="a", chunk_tokens="1", tbt_slo_met="1"),
                  dict(request_id="b", chunk_tokens="1", tbt_slo_met=""),
                  dict(request_id="b", chunk_tokens="9", tbt_slo_met="0"),
                  dict(request_id="c", chunk_tokens="1", tbt_slo_met="1")]
        result = report.aggregate(rows, tokens, 100, "loong")
        self.assertEqual(result["tbt_slo_attainment_pct"], 10)  # not (100+0)/2
        self.assertEqual(result["ttft_slo_attainment_pct"], 50)
        self.assertEqual(result["prefill_tok_s"], 3)
        self.assertEqual(result["decode_tok_s"], 0.12)
        self.assertEqual(result["tbt_slo_goodput_tok_s"], 0.01)
        self.assertEqual(result["ttft_slo_goodput_req_min"], 0.6)
        self.assertIsNone(report.aggregate([], [], 100, "loong")["tbt_slo_attainment_pct"])

    def test_matching_cohort_and_reject_invalid_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            fixture(path)
            run = report.load_run("Baseline", path)
            report.validate_cohort([run, copy.deepcopy(run)])
            for key in ("signature", "trace_sha256", "slo_sha256", "arrival_scale"):
                bad = copy.deepcopy(run)
                bad[key] = "mismatch"
                with self.assertRaisesRegex(ValueError, key):
                    report.validate_cohort([run, bad])
            raw = path / "client/request_results.csv"
            raw.write_text(raw.read_text().replace("success", "failed", 1))
            with self.assertRaisesRegex(ValueError, "fully completed"):
                report.load_run("Baseline", path)

    def test_manifest_inputs_unchanged_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            fixture(source)
            before = {str(p): report.digest(p) for p in source.rglob("*") if p.is_file()}
            output = root / "plots"
            with mock.patch.object(report, "render_metric") as plot:
                manifest = report.generate(dict.fromkeys(report.COLORS, source), output)
            self.assertEqual(plot.call_count, 16)
            self.assertEqual(len(manifest["charts"]), 16)
            self.assertEqual(len(report.read_csv(output / "comparison_summary.csv")), 15)
            self.assertEqual([c["color"] for c in manifest["cases"]], list(report.COLORS.values()))
            self.assertEqual(before, {str(p): report.digest(p) for p in source.rglob("*") if p.is_file()})
            command = (output / "reproduce_command.txt").read_text()
            self.assertNotIn("\n+", command)
            self.assertEqual(command.count("--case"), 5)
            with self.assertRaises(FileExistsError):
                report.generate(dict.fromkeys(report.COLORS, source), output)

    def test_actual_plot_is_one_axes_and_closed(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            fixture(path)
            run = report.load_run("Baseline", path)
            axes_counts = []
            original = Figure.savefig

            def save(figure, *args, **kwargs):
                axes_counts.append(len(figure.axes))
                return original(figure, *args, **kwargs)

            with mock.patch.object(Figure, "savefig", save):
                report.render_metric([run], "all", report.SLO_METRICS[0], Path(tmp) / "plot.png")
            self.assertEqual(axes_counts, [1])
            self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
