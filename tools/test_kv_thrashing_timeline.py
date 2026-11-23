"""CPU-only evidence/plot contracts for the archived KV-thrashing renderer."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_script"))
import generate_kv_thrashing_timeline as g


class ThrashingTimelineTests(unittest.TestCase):
    def fixture(self):
        metrics, timeline = [], []
        for i, (before, used) in enumerate([(0, 5713), (16384, 6737), (32768, 7761),
                                            (0, 5713), (16384, 6737), (32768, 7761),
                                            (0, 5713)]):
            metrics.append({"iteration_id": i, "scheduled_ts": 100 + i * 10,
                            "worker_execute_start_ts": 100 + i * 10,
                            "model_done_ts": 109 + i * 10, "kv_total_blocks": 8192,
                            "kv_used_blocks": used, "kv_free_blocks": 8192 - used,
                            "max_num_batched_tokens": 16384, "source_line": i + 1})
            timeline.append({"iteration_id": i, "request_id": "case-prefill-r1",
                             "phase": "prefill", "num_prompt_tokens": 75000,
                             "prefill_tokens": 16384, "num_computed_tokens_before": before,
                             "source_line": i + 1})
        return metrics, timeline

    def analyze(self, metrics, timeline):
        return g.analyze(metrics, timeline, kv_bytes=4 * 1024**3, block_size=16)

    def test_reset_and_capacity(self):
        data = self.analyze(*self.fixture())
        self.assertEqual(data["summary"]["reset_count"], 2)
        self.assertEqual(data["summary"]["discarded_tokens"], 98304)
        self.assertEqual(data["summary"]["full_chunk_fit_max_used_mib"], 3584)
        reset = data["resets"][0]
        self.assertEqual(reset["last_chunk_free_mib"], 215.5)
        self.assertEqual(reset["next_chunk_required_mib"], 512)
        self.assertEqual(reset["deficit_blocks"], 593)
        self.assertEqual((reset["window_start_sec"], reset["observed_resume_sec"]), (29, 30))

    def test_missing_and_invalid_memory_never_defaults_to_zero(self):
        for change in ("missing", "conservation", "capacity", "fraction", "nan"):
            ms, ts = self.fixture()
            if change == "missing":
                del ms[1]["kv_free_blocks"]
            elif change == "conservation":
                ms[1]["kv_free_blocks"] += 1
            elif change == "capacity":
                ms[1]["kv_total_blocks"] *= 2
            elif change == "fraction":
                ms[1]["kv_used_blocks"] = 4.5
            else:
                ms[1]["worker_execute_start_ts"] = float("nan")
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                self.analyze(ms, ts)

    def test_exact_identity_and_incomplete_trace(self):
        ms, ts = self.fixture()
        other = copy.deepcopy(ts)
        for row in other:
            row["request_id"] = "case-prefill-r2"
        with self.assertRaises(ValueError):
            self.analyze(ms, ts + other)
        data = g.analyze(ms, ts + other, kv_bytes=4 * 1024**3, block_size=16,
                         request_id="case-prefill-r1")
        self.assertEqual(data["summary"]["reset_count"], 2)
        with self.assertRaises(ValueError):
            self.analyze(ms + [ms[0]], ts)
        with self.assertRaises(ValueError):
            self.analyze(ms, ts[:1] + ts[2:])
        ms[1]["worker_execute_start_ts"] = ms[0]["model_done_ts"] - 1
        with self.assertRaises(ValueError):
            self.analyze(ms, ts)

    def test_page_rounding(self):
        self.assertEqual(g.blocks_for_next_chunk(17, 15, 16), 0)
        self.assertEqual(g.blocks_for_next_chunk(17, 16, 16), 1)
        self.assertEqual(g.blocks_for_next_chunk(49152, 16384, 16), 1024)
        self.assertEqual(g.blocks_for_next_chunk(75000, 0, 16), 0)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            with self.assertRaises(FileExistsError):
                g.generate(target / "missing", target / "missing.txt", target, block_size=16)

    def test_one_panel_shared_time_axis_and_units(self):
        data = self.analyze(*self.fixture())
        for free in (False, True):
            fig = g.plot(data, zoom=(0, 65), free_view=free)
            try:
                self.assertEqual(len(fig.axes), 2)  # One panel with two unit axes.
                self.assertEqual(fig.axes[0].get_position().bounds,
                                 fig.axes[1].get_position().bounds)
                self.assertEqual(fig.axes[0].get_xlim(), fig.axes[1].get_xlim())
                self.assertIn("MiB", fig.axes[0].get_ylabel())
                self.assertIn("tokens", fig.axes[1].get_ylabel())
                self.assertIn("not total GPU VRAM", fig.texts[-1].get_text())
            finally:
                g.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
