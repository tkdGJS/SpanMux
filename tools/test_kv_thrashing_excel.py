"""CPU contracts for chart-ready numeric XLSX export."""

import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_script"))
import export_kv_thrashing_excel as x


class ExcelTests(unittest.TestCase):
    def fixture(self):
        return {
            "summary": {"prompt_tokens": 100000, "end_sec": 40, "kv_capacity_mib": 4096,
                        "full_chunk_fit_max_used_mib": 3584, "full_chunk_required_mib": 512,
                        "request_id": "exact-prefill-id", "reset_count": 1,
                        "discarded_tokens": 16384, "sources_sha256": {"file": "hash"}},
            "samples": [{"time_sec": -1, "used_mib": 3000, "free_mib": 1096},
                        {"time_sec": 10, "used_mib": 3638, "free_mib": 458},
                        {"time_sec": 30, "used_mib": 3639, "free_mib": 457}],
            "chunks": [{"start_sec": 10, "end_sec": 20, "before_tokens": 0, "after_tokens": 16384},
                       {"start_sec": 21, "end_sec": 31, "before_tokens": 0, "after_tokens": 16384}],
            "resets": [{"reset_index": 1, "observed_resume_sec": 21, "discarded_tokens": 16384}],
        }

    def test_common_time_preserves_steps_and_linear_geometry(self):
        d = self.fixture()
        rows = x.chart_rows(d, (5, 25))[1:]
        at10 = [r for r in rows if r[0] == 10]
        self.assertEqual([r[1] for r in at10], [3000, 3638])
        self.assertEqual([r[3] for r in at10], [0, 0])
        self.assertEqual([r[3] for r in rows if r[0] == 25], [6553.6, 6553.6])
        self.assertTrue(all(r[3] is None for r in rows if r[0] < 10 or r[0] > 31))
        self.assertTrue(all(r[1] is None for r in rows if r[0] > 30))
        self.assertTrue(all(math.isclose(r[1] + r[2], 4096) for r in rows if r[1] is not None))
        self.assertEqual(sorted(r[0] for r in rows), [r[0] for r in rows])

    def test_duplicate_progress_timestamp_keeps_both_limits(self):
        self.assertEqual(x.polyline_value([1, 2, 2, 3], [0, 10, 0, 10], 2, left=True), 10)
        self.assertEqual(x.polyline_value([1, 2, 2, 3], [0, 10, 0, 10], 2), 0)

    def test_full_zoom_workbook_typed_cells_and_strings(self):
        sheets = x.workbook_sheets(self.fixture(), (5, 25))
        self.assertEqual(len(sheets), 8)
        self.assertEqual(sheets[2][1][1][0], 5)
        self.assertEqual(sheets[2][1][-1][0], 25)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.xlsx"
            x.write_xlsx(path, sheets + [("Literal", [["header"], ["=1+2"], [None], [3.5]])])
            with ZipFile(path) as z:
                self.assertIsNone(z.testzip())
                for name in z.namelist():
                    if name.endswith((".xml", ".rels")):
                        ET.fromstring(z.read(name))
                root = ET.fromstring(z.read("xl/worksheets/sheet9.xml"))
                ns = {"s": x.SS}
                self.assertEqual(root.find(".//s:c[@r='A2']", ns).get("t"), "inlineStr")
                self.assertIsNone(root.find(".//s:c[@r='A3']", ns))
                self.assertEqual(root.find(".//s:c[@r='A4']/s:v", ns).text, "3.5")
                self.assertIsNone(root.find(".//s:f", ns))
            with self.assertRaises(FileExistsError):
                x.write_xlsx(path, sheets)

    def test_invalid_cells_fail_before_file_creation(self):
        for value in (float("nan"), float("inf"), "bad\x01text"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "data.xlsx"
                with self.assertRaises(ValueError):
                    x.write_xlsx(path, [("Sheet", [["value"], [value]])])
                self.assertFalse(path.exists())

    def test_excel_column_names(self):
        self.assertEqual(x.column_name(1), "A")
        self.assertEqual(x.column_name(26), "Z")
        self.assertEqual(x.column_name(27), "AA")


if __name__ == "__main__":
    unittest.main()
