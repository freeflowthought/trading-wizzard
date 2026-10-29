from unittest.mock import patch

from twiz.ingest.rss import fetch_rss_items


def test_zero_limit_does_not_fetch():
    with patch("twiz.ingest.rss._fetch_url") as fetch:
        assert fetch_rss_items(
            [{"url": "https://example.test"}], [], [], live=True, limit=0
        ) == ([], [])
        fetch.assert_not_called()


def test_feeds_share_budget_and_undated_news_is_not_assigned_today():
    def entries(prefix):
        return [{"title": "SPY", "url": f"https://example.test/{prefix}/{i}"} for i in range(4)]
    feeds = [{"name": name, "url": f"https://example.test/{name}"} for name in ["fed", "sec"]]
    with (
        patch("twiz.ingest.rss._fetch_url", return_value=b"fixture"),
        patch(
            "twiz.ingest.rss._parse_with_feedparser",
            side_effect=[entries("fed"), entries("sec")],
        ),
    ):
        rows, warnings = fetch_rss_items(feeds, [], [], live=True, limit=4)
    assert not warnings
    assert [row.source for row in rows] == ["fed", "sec", "fed", "sec"]
    assert all(row.published_at == "" for row in rows)
