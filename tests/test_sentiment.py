import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.models import RawItem
from twiz.sentiment import FinancialLexiconScorer, analyze_sentiment_items, engagement_weight


class SentimentTests(unittest.TestCase):
    def test_lexicon_scores_financial_terms(self) -> None:
        label, score, confidence = FinancialLexiconScorer().score(
            "NVDA beat guidance with strong upside"
        )
        self.assertEqual(label, "positive")
        self.assertGreater(score, 0)
        self.assertGreater(confidence, 0.5)

    def test_engagement_weight_decays_over_time(self) -> None:
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        recent = engagement_weight({"score": 100, "num_comments": 25}, now.isoformat(), now=now)
        old = engagement_weight(
            {"score": 100, "num_comments": 25},
            (now - timedelta(days=5)).isoformat(),
            now=now,
        )
        self.assertGreater(recent, old)

    def test_analyze_items_emits_signal_per_ticker(self) -> None:
        item = RawItem(
            source="fixture",
            source_type="reddit",
            url="https://example.test",
            published_at="2026-06-06T12:00:00+00:00",
            ticker_candidates=["NVDA", "QQQ"],
            title="NVDA and QQQ bullish breakout",
            text="Strong upside",
            engagement={"score": 10},
        )
        signals = analyze_sentiment_items([item], [], [], prefer_finbert=False)
        self.assertEqual({signal.ticker for signal in signals}, {"NVDA", "QQQ"})
        self.assertTrue(all(signal.model == "financial-lexicon-fallback" for signal in signals))


if __name__ == "__main__":
    unittest.main()
