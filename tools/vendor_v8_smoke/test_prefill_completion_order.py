"""CPU checks for observed Prefill completion rank, no vLLM/GPU imports."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graph_script"))
from generate_prefill_completion_order import (  # noqa: E402
    INITIAL_WAIT_LABEL, POST_SCHEDULE_LABEL, attach_server_times, generate, rank_requests, render_comparison,
)


class PrefillCompletionOrder(unittest.TestCase):
    def setUp(self):
        self.events = [
            dict(request_id="long-first-input", event_index=0, dataset="leval", first_token_s=100),
            dict(request_id="short-later-input", event_index=1, dataset="sharegpt", first_token_s=10),
            dict(request_id="long-tie", event_index=2, dataset="lveval", first_token_s=100),
            dict(request_id="short-last-input", event_index=3, dataset="sharegpt", first_token_s=20),
        ]
        for event in self.events:
            event["input_tokens"] = 100

    def server_records(self):
        # Deliberately different server/client clock origins and nonmonotonic
        # elapsed times; elapsed duration must never become the ranking key.
        records = []
        for i, (event, elapsed) in enumerate(zip(self.events, (7, 30, 5, 2))):
            queued = 10000 + i * 100
            records.append(dict(tenant_id=event["request_id"], prompt_tokens=100,
                                queued_ts=queued, scheduled_ts=queued + 50,
                                first_token_ts=queued + 50 + elapsed,
                                queued_time_s=50, prefill_time_s=elapsed))
        return records

    def test_server_join_excludes_only_initial_wait_and_preserves_client_order(self):
        records = list(reversed(self.server_records())) + [dict(tenant_id="warmup")]
        joined = attach_server_times(self.events, records)
        self.assertEqual([r["initial_queue_wait_s"] for r in joined], [50] * 4)
        self.assertEqual([r["post_schedule_elapsed_s"] for r in joined], [7, 30, 5, 2])
        self.assertEqual([r["post_schedule_elapsed_s"] for r in rank_requests(joined, "sharegpt")],
                         [30, 2])
        self.assertEqual([r["first_token_s"] for r in joined], [100, 10, 100, 20])
        self.assertNotIn("post_schedule_elapsed_s", self.events[0])

    def test_server_join_rejects_missing_duplicate_and_invalid_metrics(self):
        records = self.server_records()
        cases = [records[:-1], records + [records[0]]]
        for field, value in (("queued_time_s", 49), ("prefill_time_s", 99),
                             ("first_token_ts", float("nan")), ("scheduled_ts", -1),
                             ("prompt_tokens", 101)):
            invalid = [dict(r) for r in records]
            invalid[0][field] = value
            cases.append(invalid)
        for i, invalid in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                attach_server_times(self.events, invalid)

    def test_elapsed_overlay_retains_completion_ranks_and_uses_server_duration(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        runs = [dict(case=label, color=color,
                     events=attach_server_times(self.events, self.server_records()))
                for label, color in (("Baseline+SRPT", "#2CA02C"),
                                      ("P/D+SRPT", "#9467BD"), ("Proposal", "#FF7F0E"))]
        saved = []

        def inspect(figure, filename, **kwargs):
            self.assertEqual(len(figure.axes), 1)
            ax = figure.axes[0]
            self.assertEqual(ax.get_ylabel(), POST_SCHEDULE_LABEL)
            self.assertEqual(len(ax.lines), 3)
            for run, line in zip(runs, ax.lines):
                self.assertEqual(list(line.get_xdata()), [1, 2])
                self.assertEqual(list(line.get_ydata()), [30, 2])
                self.assertEqual(line.get_color(), run["color"])
            self.assertTrue(any("not pure GPU compute" in t.get_text() for t in figure.texts))
            saved.append(Path(filename).suffix)

        with patch.object(Figure, "savefig", inspect):
            render_comparison(runs, "sharegpt", (2, 40), Path("unused"), exclude_initial_wait=True)
        self.assertEqual(saved, [".png", ".pdf"])

    def test_processing_rank_is_completion_not_input_order(self):
        rows = rank_requests(self.events, "all")
        self.assertEqual([r["request_id"] for r in rows],
                         ["short-later-input", "short-last-input", "long-first-input", "long-tie"])
        self.assertEqual([r["input_order_rank"] for r in rows], [2, 4, 1, 3])
        self.assertEqual([r["first_token_s"] for r in rows], [10, 20, 100, 100])

    def test_waiting_overlay_uses_initial_wait_without_reordering_requests(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        records = self.server_records()
        for record, wait in zip(records, (50, 90, 30, 4)):
            record["queued_ts"] = record["scheduled_ts"] - wait
            record["queued_time_s"] = wait
        runs = [dict(case=label, color=color, events=attach_server_times(self.events, records))
                for label, color in (("Baseline+SRPT", "#2CA02C"),
                                      ("P/D+SRPT", "#9467BD"), ("Proposal", "#FF7F0E"))]
        saved = []

        def inspect(figure, filename, **kwargs):
            self.assertEqual(len(figure.axes), 1)
            ax = figure.axes[0]
            self.assertEqual(ax.get_ylabel(), INITIAL_WAIT_LABEL)
            self.assertEqual(len(ax.lines), 3)
            for run, line in zip(runs, ax.lines):
                self.assertEqual(list(line.get_xdata()), [1, 2])
                self.assertEqual(list(line.get_ydata()), [90, 4])
                self.assertEqual(line.get_color(), run["color"])
                self.assertEqual(line.get_label(), run["case"])
            self.assertTrue(any("waiting after scheduling/preemption is not included" in t.get_text()
                                for t in figure.texts))
            saved.append(Path(filename).suffix)

        with patch.object(Figure, "savefig", inspect):
            render_comparison(runs, "sharegpt", (2, 100), Path("unused"), initial_wait=True)
        self.assertEqual(saved, [".png", ".pdf"])

    def test_conflicting_time_options_fail_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            generate(Path("missing_summary.csv"), Path("unused_output"),
                     exclude_initial_wait=True, initial_wait=True)

    def test_group_local_rank_retains_global_identity_and_ties(self):
        rows = rank_requests(list(reversed(self.events)), "loong")
        self.assertEqual([r["completion_rank"] for r in rows], [1, 2])
        self.assertEqual([r["global_completion_rank"] for r in rows], [3, 4])
        self.assertEqual([r["request_id"] for r in rows], ["long-first-input", "long-tie"])
        self.assertEqual(len(rank_requests(self.events, "sharegpt")), 2)
        self.assertTrue(all("completion_rank" not in r for r in self.events))

    def test_overlay_keeps_three_independent_series_on_one_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        runs = [dict(case=label, color=color,
                     events=[dict(r, first_token_s=r["first_token_s"] + offset) for r in self.events])
                for label, color, offset in (("Baseline+SRPT", "#2CA02C", 0),
                                             ("P/D+SRPT", "#9467BD", 1),
                                             ("Proposal", "#FF7F0E", 2))]
        saved = []

        def inspect(figure, filename, **kwargs):
            self.assertEqual(len(figure.axes), 1)
            lines = figure.axes[0].lines
            self.assertEqual(len(lines), 3)
            for i, line in enumerate(lines):
                self.assertEqual(line.get_label(), runs[i]["case"])
                self.assertEqual(line.get_color(), runs[i]["color"])
                self.assertEqual(list(line.get_xdata()), [1, 2])
                self.assertEqual(list(line.get_ydata()), [10 + i, 20 + i])
            saved.append(Path(filename).suffix)

        with patch.object(Figure, "savefig", inspect):
            render_comparison(runs, "sharegpt", (2, 110), Path("unused"))
        self.assertEqual(saved, [".png", ".pdf"])


if __name__ == "__main__":
    unittest.main()
