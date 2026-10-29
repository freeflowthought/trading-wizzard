import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.models import RawItem
from twiz.sentiment import (
    FinancialLexiconScorer,
    analyze_sentiment_items,
    engagement_weight,
    parse_datetime,
)


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
        signals = analyze_sentiment_items(
            [item], [], [], prefer_finbert=False,
            now=datetime(2026, 6, 6, 13, tzinfo=timezone.utc),
        )
        self.assertEqual({signal.ticker for signal in signals}, {"NVDA", "QQQ"})
        self.assertTrue(all(signal.model == "financial-lexicon-fallback" for signal in signals))

    def test_rss_dates_keep_original_age(self) -> None:
        now = datetime(2026, 10, 27, tzinfo=timezone.utc)
        published = "Sun, 07 Jun 2026 12:00:00 GMT"
        self.assertEqual(parse_datetime(published), datetime(2026, 6, 7, 12, tzinfo=timezone.utc))
        self.assertEqual(engagement_weight({}, published, now=now), 0.1)
        self.assertIsNone(parse_datetime("not a date"))

    def test_stale_future_undated_and_duplicate_news_excluded(self) -> None:
        now = datetime(2026, 10, 27, tzinfo=timezone.utc)
        dates = [
            "2026-10-26T12:00:00Z", "2026-10-26T12:00:00Z",
            "Sun, 07 Jun 2026 12:00:00 GMT", "2026-10-28T12:00:00Z", "invalid",
        ]
        items = [
            RawItem(
                source="fixture", source_type="rss", url=f"https://example.test/{index // 2}",
                published_at=published, ticker_candidates=["SPY"],
                title="SPY bullish", text="",
            )
            for index, published in enumerate(dates)
        ]
        signals = analyze_sentiment_items(items, [], [], prefer_finbert=False, now=now)
        self.assertEqual(len(signals), 1)


if __name__ == "__main__":
    unittest.main()
