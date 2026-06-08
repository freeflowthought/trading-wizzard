import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.options import black_scholes_greeks


class OptionsTests(unittest.TestCase):
    def test_black_scholes_call_delta_known_range(self) -> None:
        greeks = black_scholes_greeks("call", 100, 100, 1.0, 0.01, 0.20)
        self.assertAlmostEqual(greeks["delta"], 0.5596, places=3)
        self.assertGreater(greeks["gamma"], 0)
        self.assertGreater(greeks["vega"], 0)
        self.assertLess(greeks["theta"], 0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            black_scholes_greeks("call", 100, 0, 1.0, 0.01, 0.20)


if __name__ == "__main__":
    unittest.main()
