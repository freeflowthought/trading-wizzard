import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.config import DEFAULT_IGNORE_TERMS, DEFAULT_WATCHLIST
from twiz.text import extract_tickers


class TickerExtractionTests(unittest.TestCase):
    def test_extracts_configured_aliases_and_ignores_ai(self) -> None:
        text = "AI capex is hot, but $NVDA and QQQ are the tickers discussed. Apple also moved."
        tickers = extract_tickers(text, DEFAULT_WATCHLIST, DEFAULT_IGNORE_TERMS)
        self.assertIn("NVDA", tickers)
        self.assertIn("QQQ", tickers)
        self.assertIn("AAPL", tickers)
        self.assertNotIn("AI", tickers)

    def test_plain_lowercase_short_ticker_does_not_match_common_word(self) -> None:
        text = "I use spy stories as a metaphor, not the ETF symbol."
        tickers = extract_tickers(text, DEFAULT_WATCHLIST, DEFAULT_IGNORE_TERMS)
        self.assertNotIn("SPY", tickers)


if __name__ == "__main__":
    unittest.main()
