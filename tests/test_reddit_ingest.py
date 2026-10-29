import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.config import WatchedAsset
from twiz.ingest.reddit import fetch_jina_reddit_search_items, parse_jina_search_results


class RedditIngestTests(unittest.TestCase):
    def test_parse_jina_search_results_text_format(self) -> None:
        payload = """
[1] Title: NVDA investors debate earnings setup
[1] URL Source: https://www.reddit.com/r/stocks/comments/abc/nvda_setup/
[1] Published Time: 2026-06-06T12:00:00Z
[1] Description: Bulls see strong upside, bears worry about valuation.

[2] Title: SOXX semiconductor ETF pullback
[2] URL Source: https://www.reddit.com/r/investing/comments/def/soxx_pullback/
[2] Description: Traders discuss support and weak breadth.
"""
        rows = parse_jina_search_results(payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "NVDA investors debate earnings setup")
        self.assertEqual(
            rows[0]["url"], "https://www.reddit.com/r/stocks/comments/abc/nvda_setup/"
        )
        self.assertIn("strong upside", rows[0]["text"])

    def test_jina_fallback_can_be_disabled(self) -> None:
        old_value = os.environ.get("JINA_SEARCH_ENABLED")
        os.environ["JINA_SEARCH_ENABLED"] = "false"
        try:
            rows, warnings = fetch_jina_reddit_search_items(
                [WatchedAsset("NVDA", "NVIDIA", ["NVDA"], ["NVDA"])],
                [],
                limit=1,
            )
        finally:
            if old_value is None:
                os.environ.pop("JINA_SEARCH_ENABLED", None)
            else:
                os.environ["JINA_SEARCH_ENABLED"] = old_value
        self.assertEqual(rows, [])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
