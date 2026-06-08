import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.config import DEFAULT_WATCHLIST
from twiz.reporting import build_report_bundle, render_markdown, write_report_files
from twiz.storage import ProjectPaths, write_json


class ReportingTests(unittest.TestCase):
    def test_report_bundle_and_files_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ProjectPaths(Path(tmp))
            paths.ensure()
            report_date = "2026-06-06"
            write_json(
                paths.raw / f"market-{report_date}.json",
                {"SPY": [{"date": report_date, "close": 500}]},
            )
            write_json(
                paths.processed / f"technical-{report_date}.json",
                {
                    "SPY": {
                        "regime": "uptrend",
                        "levels": [
                            {"kind": "support", "price": 495, "distance_pct": -1.0},
                            {"kind": "support", "price": 490, "distance_pct": -2.0},
                            {"kind": "support", "price": 475, "distance_pct": -5.0},
                            {"kind": "support", "price": 440, "distance_pct": -12.0},
                            {"kind": "resistance", "price": 505, "distance_pct": 1.0},
                            {"kind": "resistance", "price": 510, "distance_pct": 2.0},
                            {"kind": "resistance", "price": 525, "distance_pct": 5.0},
                        ],
                    }
                },
            )
            write_json(
                paths.processed / f"sentiment-summary-{report_date}.json",
                {"SPY": {"weighted_score": 0.2, "total_weight": 3.0}},
            )
            bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
            markdown = render_markdown(bundle)
            md_path, html_path = write_report_files(paths, bundle)
            self.assertIn("SPY", markdown)
            self.assertIn("Immediate support levels", markdown)
            self.assertIn("Intermediate downside checkpoints", markdown)
            self.assertIn("Deeper downside checkpoints, not immediate support", markdown)
            self.assertIn("Immediate resistance / possible ceiling", markdown)
            self.assertIn("Intermediate upside checkpoints", markdown)
            self.assertIn("495.00", markdown)
            self.assertIn("505.00", markdown)
            self.assertTrue(md_path.exists())
            self.assertTrue(html_path.exists())


if __name__ == "__main__":
    unittest.main()
