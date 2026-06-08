import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.technical import cluster_support_resistance, latest_indicators, rolling_mean


def make_rows(values):
    rows = []
    for index, value in enumerate(values):
        rows.append(
            {
                "date": f"2026-01-{index + 1:02d}",
                "open": value,
                "high": value + (5 if index in {5, 11, 17} else 1),
                "low": value - (5 if index in {3, 9, 15} else 1),
                "close": value,
                "volume": 1000,
            }
        )
    return rows


class TechnicalTests(unittest.TestCase):
    def test_rolling_mean_respects_lookback(self) -> None:
        output = rolling_mean([1, 2, 3, 4], 3)
        self.assertEqual(output[:2], [None, None])
        self.assertEqual(output[2], 2)

    def test_latest_indicators_insufficient_history(self) -> None:
        indicators = latest_indicators(make_rows([10, 11, 12]))
        self.assertIsNone(indicators["sma20"])
        self.assertEqual(indicators["close"], 12)

    def test_support_resistance_clusters_pivots(self) -> None:
        rows = make_rows(
            [100, 102, 101, 98, 101, 106, 102, 101, 100, 97]
            + [101, 106, 102, 101, 100, 97, 101, 106, 103, 102]
        )
        levels = cluster_support_resistance(rows, merge_percent=0.03, bars_for_peak=2)
        self.assertTrue(levels)
        self.assertTrue(any(level["touches"] >= 2 for level in levels))
        self.assertTrue(any(level["kind"] == "support" for level in levels))
        self.assertTrue(any(level["kind"] == "resistance" for level in levels))

    def test_support_resistance_keeps_recent_reaction_levels_separate(self) -> None:
        rows = make_rows([90 + index * 0.2 for index in range(20)])
        rows.extend(
            [
                {
                    "date": "2026-04-01",
                    "open": 109.8,
                    "high": 111.5,
                    "low": 108.2,
                    "close": 110.4,
                    "volume": 1000,
                },
                {
                    "date": "2026-04-02",
                    "open": 110.7,
                    "high": 112.0,
                    "low": 108.6,
                    "close": 111.6,
                    "volume": 1000,
                },
                {
                    "date": "2026-04-03",
                    "open": 112.2,
                    "high": 113.0,
                    "low": 110.8,
                    "close": 112.5,
                    "volume": 1000,
                },
            ]
        )
        levels = cluster_support_resistance(rows)
        supports = [level for level in levels if level["kind"] == "support"]
        self.assertTrue(any(108.0 <= level["price"] <= 109.0 for level in supports))
        self.assertTrue(any(abs(float(level["distance_pct"])) > 6.0 for level in supports))


if __name__ == "__main__":
    unittest.main()
