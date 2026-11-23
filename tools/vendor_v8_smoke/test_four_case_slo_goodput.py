#!/usr/bin/env python3
"""CPU-only checks for four-case SLO goodput and violation metrics."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "graph_script/generate_four_case_experiment_comparison.py"
SPEC = importlib.util.spec_from_file_location("four_case_slo_goodput", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FourCaseSLOGoodputTest(unittest.TestCase):
    def test_request_goodput_and_conditional_violation_degree(self) -> None:
        result = MODULE.normalized_slo_tradeoff(
            [
                (50.0, 100.0, 1),
                (100.0, 100.0, 1),
                (200.0, 100.0, 1),
            ],
            e2e_s=60.0,
        )
        self.assertEqual(result.eligible_units, 3)
        self.assertEqual(result.compliant_units, 2)
        self.assertTrue(math.isclose(result.goodput_per_s * 60.0, 2.0))
        self.assertTrue(math.isclose(result.violation_rate_pct, 100.0 / 3.0))
        self.assertEqual(result.violation_mean_excess_pct, 100.0)
        self.assertEqual(result.violation_p95_excess_pct, 100.0)

    def test_token_goodput_weights_stream_chunks(self) -> None:
        result = MODULE.normalized_slo_tradeoff(
            [
                (80.0, 100.0, 2),
                (150.0, 100.0, 3),
            ],
            e2e_s=100.0,
        )
        self.assertEqual(result.eligible_units, 5)
        self.assertEqual(result.compliant_units, 2)
        self.assertEqual(result.goodput_per_s, 0.02)
        self.assertEqual(result.violation_rate_pct, 60.0)
        self.assertEqual(result.violation_mean_excess_pct, 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
