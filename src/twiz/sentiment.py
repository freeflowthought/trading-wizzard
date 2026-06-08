import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .config import WatchedAsset
from .models import RawItem, SentimentSignal
from .text import extract_tickers, item_summary_text

POSITIVE_TERMS = {
    "beat",
    "beats",
    "bullish",
    "breakout",
    "growth",
    "upgrade",
    "upgraded",
    "strong",
    "surge",
    "rally",
    "upside",
    "profit",
    "record",
    "buyback",
    "guidance raised",
}

NEGATIVE_TERMS = {
    "miss",
    "misses",
    "bearish",
    "breakdown",
    "downgrade",
    "downgraded",
    "weak",
    "lawsuit",
    "cut guidance",
    "recession",
    "inflation scare",
    "selloff",
    "loss",
    "layoff",
    "probe",
}


def parse_datetime(value: str, now: Optional[datetime] = None) -> datetime:
    if not value:
        return now or datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return now or datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class FinancialLexiconScorer:
    model_name = "financial-lexicon-fallback"

    def score(self, text: str) -> Tuple[str, float, float]:
        lowered = text.lower()
        positive = sum(1 for term in POSITIVE_TERMS if term in lowered)
        negative = sum(1 for term in NEGATIVE_TERMS if term in lowered)
        total = positive + negative
        if total == 0:
            return "neutral", 0.0, 0.50
        score = (positive - negative) / total
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        confidence = min(0.95, 0.55 + abs(score) * 0.4)
        return label, score, confidence


class FinBertScorer:
    model_name = "ProsusAI/finbert"

    def __init__(self) -> None:
        self._pipeline = None
        self._error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._error is None

    def _load(self) -> bool:
        if self._pipeline is not None:
            return True
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text-classification", model=self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - depends on optional local model setup
            self._error = str(exc)
            return False

    def score(self, text: str) -> Tuple[str, float, float]:
        if not self._load():
            return FinancialLexiconScorer().score(text)
        output = self._pipeline(text[:4000], truncation=True)
        first = output[0] if isinstance(output, list) else output
        label = str(first["label"]).lower()
        confidence = float(first["score"])
        if label.startswith("pos"):
            score = confidence
            label = "positive"
        elif label.startswith("neg"):
            score = -confidence
            label = "negative"
        else:
            score = 0.0
            label = "neutral"
        return label, score, confidence


def engagement_weight(
    engagement: Dict[str, float],
    published_at: str,
    now: Optional[datetime] = None,
    half_life_hours: float = 36.0,
) -> float:
    now = now or datetime.now(timezone.utc)
    published = parse_datetime(published_at, now=now)
    age_hours = max((now - published).total_seconds() / 3600, 0.0)
    raw_engagement = (
        float(engagement.get("score", 0))
        + float(engagement.get("upvotes", 0))
        + float(engagement.get("comments", 0)) * 1.5
        + float(engagement.get("num_comments", 0)) * 1.5
    )
    engagement_factor = math.log1p(max(raw_engagement, 0.0))
    time_factor = 0.5 ** (age_hours / half_life_hours)
    return round(max(0.1, (1.0 + engagement_factor) * time_factor), 4)


def analyze_sentiment_items(
    items: Iterable[RawItem],
    watchlist: Iterable[WatchedAsset],
    ignore_terms: Iterable[str],
    prefer_finbert: bool = True,
    now: Optional[datetime] = None,
) -> List[SentimentSignal]:
    scorer = FinBertScorer() if prefer_finbert else FinancialLexiconScorer()
    fallback = FinancialLexiconScorer()
    signals: List[SentimentSignal] = []
    for item in items:
        text = item_summary_text(item.title, item.text)
        tickers = item.ticker_candidates or extract_tickers(text, watchlist, ignore_terms)
        if not tickers:
            continue
        label, score, confidence = scorer.score(text)
        model = getattr(scorer, "model_name", fallback.model_name)
        if isinstance(scorer, FinBertScorer) and not scorer.available:
            model = fallback.model_name
        weight = engagement_weight(item.engagement, item.published_at, now=now)
        for ticker in tickers:
            signals.append(
                SentimentSignal(
                    ticker=ticker,
                    label=label,
                    score=round(score, 4),
                    confidence=round(confidence, 4),
                    weight=weight,
                    model=model,
                    source_refs=[item.url],
                )
            )
    return signals


def aggregate_sentiment(signals: Iterable[SentimentSignal]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for signal in signals:
        bucket = grouped.setdefault(
            signal.ticker,
            {
                "weighted_score": 0.0,
                "total_weight": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
            },
        )
        bucket["weighted_score"] += signal.score * signal.weight
        bucket["total_weight"] += signal.weight
        bucket[signal.label] += 1.0
    for bucket in grouped.values():
        total_weight = bucket["total_weight"]
        bucket["weighted_score"] = (
            round(bucket["weighted_score"] / total_weight, 4) if total_weight else 0.0
        )
        bucket["total_weight"] = round(total_weight, 4)
    return grouped
