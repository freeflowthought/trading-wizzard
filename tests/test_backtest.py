import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.backtest import moving_average_crossover_backtest


def rows_from_prices(prices):
    return [
        {
            "date": f"2026-01-{index + 1:02d}",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1000,
        }
        for index, price in enumerate(prices)
    ]


class BacktestTests(unittest.TestCase):
    def test_signal_is_shifted_before_execution(self) -> None:
        rows = rows_from_prices([10, 10, 10, 20, 20, 20])
        with tempfile.TemporaryDirectory() as tmp:
            result = moving_average_crossover_backtest(
                "TEST",
                rows,
                fast=2,
                slow=3,
                fee_bps=0,
                slippage_bps=0,
                output_dir=Path(tmp),
            )
            curve = json.loads(Path(result.equity_curve_path).read_text(encoding="utf-8"))
        self.assertEqual(curve[3]["position"], 0)
        self.assertEqual(curve[4]["position"], 1)
        self.assertEqual(curve[3]["equity"], 1.0)
        self.assertEqual(result.metrics["signals_shifted"], 1.0)

    def test_fees_and_slippage_reduce_return(self) -> None:
        rows = rows_from_prices([10, 10, 10, 20, 20, 22, 23, 21, 20, 25])
        no_cost = moving_average_crossover_backtest(
            "TEST",
            rows,
            fast=2,
            slow=3,
            fee_bps=0,
            slippage_bps=0,
        )
        high_cost = moving_average_crossover_backtest(
            "TEST",
            rows,
            fast=2,
            slow=3,
            fee_bps=100,
            slippage_bps=100,
        )
        self.assertLess(high_cost.metrics["total_return"], no_cost.metrics["total_return"])


if __name__ == "__main__":
    unittest.main()
