import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.events import political_context
from twiz.ipo import build_ipo_snapshot, ipo_events, ipo_stage
from twiz.playbook import build_playbook, leadership_snapshot


def flat_rows(close: float, n: int = 60, drift: float = 0.0):
    rows = []
    value = close
    for index in range(n):
        value = value * (1 + drift)
        rows.append({"date": f"2026-01-{(index % 28) + 1:02d}", "open": value,
                     "high": value * 1.01, "low": value * 0.99, "close": value,
                     "volume": 1000})
    return rows


class IpoTests(unittest.TestCase):
    def test_stage_progression(self) -> None:
        self.assertEqual(ipo_stage(1, 15, 180), "debut_window")
        self.assertEqual(ipo_stage(10, 15, 180), "pre_early_release")
        self.assertEqual(ipo_stage(30, 15, 180), "early_release_open")
        self.assertEqual(ipo_stage(200, 15, 180), "post_lockup")

    def test_spcx_style_snapshot(self) -> None:
        row = {"ticker": "SPCX", "name": "SpaceX", "ipo_date": "2026-06-12",
               "ipo_price": 135, "early_release_days": 15, "lockup_days": 180,
               "verified": True, "source_url": "https://example.com/issuer-prospectus"}
        snap = build_ipo_snapshot(row, close=225.0, as_of=date(2026, 6, 22))
        self.assertEqual(snap["stage"], "pre_early_release")
        self.assertAlmostEqual(snap["vs_ipo_price_pct"], 66.7, places=1)
        self.assertTrue(any("supply shock" in v for v in snap["verdicts"]))

    def test_ipo_events_in_window(self) -> None:
        rows = [{"ticker": "SPCX", "ipo_date": "2026-06-12", "ipo_price": 135,
                 "early_release_days": 15, "verified": True,
                 "source_url": "https://example.com/issuer-prospectus"}]
        events = ipo_events(rows, date(2026, 6, 20), horizon_days=14)
        kinds = [e["kind"] for e in events]
        self.assertIn("ipo_early_release", kinds)


class LeadershipTests(unittest.TestCase):
    def test_leaders_intact(self) -> None:
        data = {"SPY": flat_rows(100, drift=0.001), "MU": flat_rows(100, drift=0.004),
                "INTC": flat_rows(100, drift=0.003)}
        snap = leadership_snapshot(data)
        self.assertEqual(snap["state"], "intact")
        self.assertFalse(snap["leaders"]["MU"]["weakening"])

    def test_leaders_breaking(self) -> None:
        data = {"SPY": flat_rows(100, drift=0.0), "MU": flat_rows(100, drift=-0.006),
                "INTC": flat_rows(100, drift=-0.005), "QCOM": flat_rows(100, drift=-0.004)}
        snap = leadership_snapshot(data)
        self.assertEqual(snap["state"], "broken")


class PoliticalAndGateTests(unittest.TestCase):
    def test_midterm_runup(self) -> None:
        ctx = political_context(date(2026, 10, 27))
        self.assertEqual(ctx["phase"], "election_runup")
        self.assertEqual(ctx["days_away"], 7)
        off = political_context(date(2027, 5, 1))
        self.assertEqual(off["phase"], "off_year")

    def test_trend_gate_on_index(self) -> None:
        rows = flat_rows(100.0)
        technical = {"levels": [
            {"price": 90.0, "kind": "support", "touches": 2, "distance_pct": -10.0,
             "sources": ["pivot"]},
            {"price": 110.0, "kind": "resistance", "touches": 3, "distance_pct": 10.0,
             "sources": ["pivot"]},
        ]}
        pb = build_playbook("SPY", rows, technical, as_of=date(2026, 10, 27))
        self.assertIsNotNone(pb["trend_gate"])
        self.assertEqual(pb["trend_gate"]["breakout_level"], 110.0)
        self.assertEqual(pb["political"]["phase"], "election_runup")
        stock = build_playbook("NVDA", rows, technical, as_of=date(2026, 10, 27))
        self.assertIsNone(stock["trend_gate"])


if __name__ == "__main__":
    unittest.main()
