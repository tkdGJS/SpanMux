#!/usr/bin/env python3
"""CPU-only contracts for frozen four-case summary presentation."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graph_script"))
import regenerate_loongserve_summary_bars as renderer


class SummaryRelabelTest(unittest.TestCase):
    def make_summary(self, root: Path) -> Path:
        fields = ["case", "source_results_dir", "trace_sha256", "submitted_count"]
        fields += list(dict.fromkeys(m.field for m in renderer.METRICS))
        path = root / "summary.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for case in reversed(renderer.CASE_COLORS):
                row = {field: "1.000" for field in fields}
                row.update(case=case, source_results_dir="historical/path", trace_sha256="same",
                           submitted_count="100", success_count="92" if case == "Baseline" else "100")
                writer.writerow(row)
        return path

    def test_identity_colors_and_units(self) -> None:
        self.assertEqual(list(renderer.CASE_COLORS.values()),
                         ["#1F77B4", "#2CA02C", "#D62728", "#FF7F0E"])
        self.assertEqual(renderer.DEFAULT_LABELS["Version 1"], "P/D Mux FCFS")
        self.assertEqual(renderer.DEFAULT_LABELS["Version 2"], "P/D Mux SRPT + Context Ring")
        self.assertEqual(len({m.filename for m in renderer.METRICS}), 15)
        metrics = {m.field: m for m in renderer.METRICS}
        self.assertEqual(metrics["ttft_p50_ms"].scale, 0.001)
        self.assertEqual(metrics["completion_p50_ms"].scale, 0.001)
        self.assertEqual(metrics["mean_framebuffer_mib"].scale, 1 / 1024)
        self.assertEqual(metrics["ttft_slo_attainment"].scale, 100)
        self.assertEqual(metrics["request_p50_tbt_p50_ms"].scale, 1)

    def test_15_single_axes_numeric_preservation_and_no_overwrite(self) -> None:
        import matplotlib.figure
        from matplotlib.colors import to_hex

        seen = []

        def capture(figure, path, **_kwargs):
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            self.assertEqual([to_hex(p.get_facecolor()).upper() for p in axis.patches],
                             list(renderer.CASE_COLORS.values()))
            self.assertIn("P/D Mux SRPT\n+ Context Ring",
                          [tick.get_text() for tick in axis.get_xticklabels()])
            self.assertTrue(any("92/100" in text.get_text() for text in figure.texts))
            seen.append(Path(path).name)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_summary(root)
            original = source.read_bytes()
            fields, rows, digest = renderer.read_summary(source)
            output = root / "new"
            with mock.patch.object(matplotlib.figure.Figure, "savefig", autospec=True,
                                   side_effect=capture):
                renderer.render_summary(source, output)
            self.assertEqual(seen, [m.filename for m in renderer.METRICS])
            self.assertEqual(source.read_bytes(), original)
            with (output / "four_case_summary.csv").open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, fields)
                renamed = list(reader)
            for old, new in zip(rows, renamed):
                self.assertEqual(new.pop("case"), renderer.DEFAULT_LABELS[old["case"]])
                self.assertEqual(new, {k: v for k, v in old.items() if k != "case"})
            manifest = json.loads((output / "plot_manifest.json").read_text())
            self.assertEqual(manifest["source_summary_sha256"], digest)
            self.assertFalse(manifest["metric_values_recomputed"])
            with self.assertRaises(FileExistsError):
                renderer.render_summary(source, output)
            with self.assertRaises(FileExistsError):
                renderer.render_summary(source, root)

    def test_invalid_identity_trace_metric_or_label_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_summary(root)
            original = source.read_text()
            mutations = (
                original.replace("Baseline SRPT", "Baseline"),
                original.replace("same", "different", 1),
                original.replace("1.000", "nan", 1),
                original.replace("observation_duration_s", "missing_field"),
            )
            for data in mutations:
                source.write_text(data)
                with self.assertRaises(ValueError):
                    renderer.render_summary(source, root / "new")
                self.assertFalse((root / "new").exists())
            source.write_text(original)
            labels = dict(renderer.DEFAULT_LABELS, **{"Version 1": "Baseline"})
            with self.assertRaises(ValueError):
                renderer.render_summary(source, root / "new", labels)
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()
