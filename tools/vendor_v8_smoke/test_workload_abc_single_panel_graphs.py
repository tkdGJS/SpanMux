#!/usr/bin/env python3
"""CPU-only contracts for one-axes Workload A/B/C bar images."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "graph_script/generate_workload_abc_combined_bars.py"
SPEC = importlib.util.spec_from_file_location("workload_abc_bars", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WorkloadABCSinglePanelGraphTest(unittest.TestCase):
    def test_one_panel_writer_creates_exactly_one_axes(self) -> None:
        import matplotlib.figure

        workloads = [SimpleNamespace(label="A")]
        axes_counts: list[int] = []

        def capture(figure: object, *_args: object, **_kwargs: object) -> None:
            axes_counts.append(len(figure.axes))

        with tempfile.TemporaryDirectory() as raw_dir:
            output_path = Path(raw_dir) / "one.png"
            with mock.patch.object(
                matplotlib.figure.Figure,
                "savefig",
                autospec=True,
                side_effect=capture,
            ):
                MODULE.plot_one_grouped_panel(
                    workloads,
                    output_path,
                    lambda _workload, _case: 1.0,
                    title="one",
                    ylabel="value",
                    fmt=".1f",
                )

        self.assertEqual(axes_counts, [1])

    def test_single_layout_has_all_throughput_and_attainment_images(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        expected = {
            "01_output_throughput.png",
            "02_logical_throughput.png",
            "03_ttft_slo_attainment_all.png",
            "04_tbt_slo_attainment_all.png",
            "05_ttft_slo_attainment_long.png",
            "06_ttft_slo_attainment_short.png",
            "07_tbt_slo_attainment_long.png",
            "08_tbt_slo_attainment_short.png",
        }
        for filename in expected:
            with self.subTest(filename=filename):
                self.assertIn(filename, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
