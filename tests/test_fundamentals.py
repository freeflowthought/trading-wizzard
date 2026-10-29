import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.fundamentals import score_cash_quality


class FundamentalsTests(unittest.TestCase):
    def test_fortress(self) -> None:
        result = score_cash_quality({
            "free_cash_flow": 50e9, "operating_cash_flow": 100e9, "capex": -25e9,
            "cash_and_equivalents": 60e9, "total_debt": 20e9, "shares_change_pct": -1.5,
        })
        self.assertEqual(result["label"], "fortress")
        self.assertIn("net_cash_fortress", result["flags"])
        self.assertIn("buyback_shrinking_share_count", result["flags"])

    def test_issuance_funded_burner(self) -> None:
        # His ORCL/NVDA sell pattern: negative FCF, capex beyond operating
        # cash flow, and fresh share issuance to fund the burn.
        result = score_cash_quality({
            "free_cash_flow": -5e9, "operating_cash_flow": 10e9, "capex": -18e9,
            "cash_and_equivalents": 5e9, "total_debt": 30e9, "shares_change_pct": 4.0,
        })
        self.assertIn(result["label"], ("burner", "distress"))
        self.assertIn("share_issuance_dilution", result["flags"])
        self.assertIn("capex_exceeds_operating_cash_flow", result["flags"])
        self.assertIn("negative_free_cash_flow", result["flags"])

    def test_missing_data_is_unknown(self) -> None:
        result = score_cash_quality({})
        self.assertIsNone(result["score"])
        self.assertEqual(result["label"], "unknown")
        self.assertEqual(result["flags"], [])

    def test_positive_cash_flow_is_not_labeled_a_cash_burner(self) -> None:
        result = score_cash_quality({
            "free_cash_flow": 20, "operating_cash_flow": 100, "capex": -80,
            "cash_and_equivalents": 10, "total_debt": 50, "shares_change_pct": 1,
        })
        self.assertEqual(result["label"], "pressured")
        self.assertNotIn("negative_free_cash_flow", result["flags"])


if __name__ == "__main__":
    unittest.main()
