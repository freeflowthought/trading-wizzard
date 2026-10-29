from datetime import datetime, timedelta, timezone

import pytest

from twiz.news_events import classify, corroborated_events, parse_rss

NOW = datetime(2025, 4, 3, 21, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("title,expected", [
    ("Trump slaps China with tariffs on up to $60 billion in imports", ("trade_policy", -0.5)),
    ("Trump pauses steeper tariffs for most countries", ("trade_policy", 0.5)),
    ("Russia invades Ukraine on multiple fronts", ("geopolitics", -0.5)),
    ("US and Iran sign ceasefire agreement", ("geopolitics", 0.5)),
    ("WHO declares novel coronavirus outbreak a pandemic", ("public_health", -0.5)),
    ("SVB is largest bank failure since 2008 financial crisis", ("financial_stress", -0.5)),
    ("Fitch downgrades U.S. long-term credit rating to AA+", ("fiscal_policy", -0.5)),
])
def test_factual_headlines_classify(title, expected):
    assert classify(title) == expected


@pytest.mark.parametrize("title", [
    "Trump threatens new tariffs on Europe",
    "Could tariffs cause a recession?",
    "How do tariffs work, and who pays them",
    "D-Day in color: a rare look at the invasion of Normandy",
    "Invasion of Privacy: public relations manager speaks",
    "Mexico declares war on seaweed",
    "Fed holds rates steady",
    "China slaps tariffs on US goods while US pauses tariffs for allies",
    "Coke raising U.S. soda prices, citing tariffs",
    "UN says Libya sides reach permanent ceasefire deal",
    "EU clears hurdle to finalize U.S. trade pact and sidestep Trump tariff hike",
])
def test_threats_explainers_noise_and_mixed_headlines_stay_neutral(title):
    assert classify(title) is None


def item(publisher, title, hours=2):
    return {"publisher": publisher, "title": title, "url": "https://example.test",
            "published_at": NOW - timedelta(hours=hours)}


def test_requires_two_distinct_major_publishers_inside_window():
    title = "Trump unveils sweeping new tariffs on dozens of nations"
    assert corroborated_events([item("Reuters", title)], NOW) == []
    assert corroborated_events([item("Reuters", title), item("Reuters", title)], NOW) == []
    assert corroborated_events([item("Reuters", title), item("Random Blog", title)], NOW) == []
    assert corroborated_events([item("Reuters", title), item("CNBC", title, hours=40)], NOW) == []
    future = item("CNBC", title, hours=-1)
    assert corroborated_events([item("Reuters", title), future], NOW) == []
    rows = corroborated_events([item("Reuters", title), item("AP News", title, hours=5)], NOW)
    assert len(rows) == 1
    row = rows[0]
    assert row["direction"] == -0.5 and row["correlation_family"] == "trade_policy"
    assert row["available_at"] == NOW.isoformat()  # never backdated to publication
    assert row["publisher_count"] == 2 and row["source_refs"][0]["publisher"] == "AP News"
    assert corroborated_events([item("Reuters", title), item("CNBC", title)], NOW, rows) == []


def test_parse_rss_strips_publisher_suffix():
    payload = (b"<rss><channel><item><title>Fed cuts rates - Reuters</title>"
               b"<link>https://n.test/a</link><pubDate>Thu, 03 Apr 2025 07:00:00 GMT</pubDate>"
               b"<source url=\"https://reuters.com\">Reuters</source></item></channel></rss>")
    [row] = parse_rss(payload)
    assert row["title"] == "Fed cuts rates" and row["publisher"] == "Reuters"
    assert row["published_at"] == datetime(2025, 4, 3, 7, tzinfo=timezone.utc)
