import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.models import OptionContract
from twiz.options import compute_event_context, summarize_options
from twiz.playbook import (
    build_playbook,
    drawdown_bands,
    macro_gate,
    nearest_round,
    round_levels,
)


def make_rows(peak: float = 100.0, last: float = 92.0, n: int = 260):
    rows = []
    for index in range(n):
        close = 80.0 + index * 0.05
        rows.append(
            {
                "date": f"2026-01-{(index % 28) + 1:02d}",
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000,
            }
        )
    rows[-2]["high"] = peak
    rows[-1]["close"] = last
    rows[-1]["high"] = last + 0.5
    rows[-1]["low"] = last - 0.5
    return rows


def fred_row(series_id: str, latest: float, drift: float = 0.0, points: int = 30):
    history = [
        {"date": f"2026-01-{i + 1:02d}", "value": str(latest - drift + drift * i / (points - 1))}
        for i in range(points)
    ]
    return {
        "source": "FRED",
        "series_id": series_id,
        "name": series_id,
        "date": history[-1]["date"],
        "value": str(latest),
        "history": history,
    }


class PlaybookTests(unittest.TestCase):
    def test_drawdown_bands_from_peak(self) -> None:
        info = drawdown_bands(make_rows(peak=100.0, last=92.0))
        self.assertEqual(info["peak"], 100.0)
        self.assertAlmostEqual(info["drawdown_pct"], -8.0, places=1)
        band8 = [b for b in info["bands"] if b["drawdown_pct"] == 8.0][0]
        self.assertAlmostEqual(band8["price"], 92.0, places=1)
        self.assertEqual(band8["label"], "deep_low_zone")

    def test_round_levels_and_nearest(self) -> None:
        levels = round_levels(7830.0)
        prices = [row["price"] for row in levels]
        self.assertIn(7800.0, prices)
        self.assertIn(8000.0, prices)
        self.assertEqual(nearest_round(7830.0, above=True), 7900.0)
        self.assertEqual(nearest_round(196.0, above=False), 190.0)

    def test_macro_gate_danger_line(self) -> None:
        gate = macro_gate([fred_row("DGS10", 5.05)], as_of=date(2026, 1, 30))
        self.assertEqual(gate["state"], "risk_off")
        calm = macro_gate([fred_row("DGS10", 4.40, drift=-0.1)], as_of=date(2026, 1, 30))
        self.assertEqual(calm["state"], "risk_on")
        rising = macro_gate([fred_row("DGS10", 4.60, drift=0.40)], as_of=date(2026, 1, 30))
        self.assertEqual(rising["state"], "caution")
        self.assertEqual(macro_gate([])["state"], "unknown")

    def test_staged_entries_scale_with_gate(self) -> None:
        rows = make_rows()
        technical = {
            "levels": [
                {"price": 90.0, "kind": "support", "touches": 3, "distance_pct": -2.2,
                 "sources": ["pivot"]},
                {"price": 85.0, "kind": "support", "touches": 2, "distance_pct": -7.6,
                 "sources": ["pivot"]},
                {"price": 95.0, "kind": "resistance", "touches": 2, "distance_pct": 3.3,
                 "sources": ["pivot"]},
            ]
        }
        calm = build_playbook(
            "NVDA", rows, technical,
            gate=macro_gate([fred_row("DGS10", 4.4)], as_of=date(2026, 1, 30)),
            as_of=date(2026, 10, 27),
        )
        scared = build_playbook(
            "NVDA", rows, technical,
            gate=macro_gate([fred_row("DGS10", 5.05)], as_of=date(2026, 1, 30)),
            as_of=date(2026, 10, 27),
        )
        self.assertTrue(calm["staged_entries"])
        self.assertGreater(
            calm["staged_entries"][0]["fraction"],
            scared["staged_entries"][0]["fraction"],
        )
        self.assertIn(95.0, calm["targets"])
        self.assertIsNotNone(calm["sell_put_strike"])
        self.assertLess(calm["sell_put_strike"], 90.0)

    def test_options_event_context(self) -> None:
        contracts = [
            OptionContract("C", "call", 100.0, "2026-11-20", bid=3.0, ask=3.4,
                           last=3.2, open_interest=1000.0, implied_volatility=0.4),
            OptionContract("P", "put", 100.0, "2026-11-20", bid=3.1, ask=3.5,
                           last=3.3, open_interest=2500.0, implied_volatility=0.42),
            OptionContract("P2", "put", 90.0, "2026-11-20", bid=0.9, ask=1.1,
                           last=1.0, open_interest=9000.0, implied_volatility=0.5),
        ]
        context = compute_event_context(contracts, spot=100.0)
        # Straddle mid = 3.2 + 3.3 = 6.5 -> 6.5% implied move ("system sees ~6-7%").
        self.assertAlmostEqual(context["expected_move_pct"], 6.5, places=1)
        self.assertAlmostEqual(context["put_call_oi_ratio"], 11.5, places=1)
        self.assertEqual(context["put_wall"], 90.0)
        snapshot = summarize_options("NVDA", 100.0, contracts, "2026-11-20")
        self.assertAlmostEqual(
            snapshot.skew_summary["expected_move_pct"], 6.5, places=1
        )


if __name__ == "__main__":
    unittest.main()
