from datetime import date, datetime, timezone

from twiz.sessions import expected_complete_session, is_us_equity_session


def test_verified_exceptional_equity_closures_are_not_missing_price_sessions():
    for day, previous in (("2018-12-05", "2018-12-04"), ("2025-01-09", "2025-01-08")):
        assert not is_us_equity_session(date.fromisoformat(day))
        now = datetime(2026, 10, 28, tzinfo=timezone.utc)
        assert expected_complete_session(day, now) == previous
    assert is_us_equity_session(date(2018, 12, 6))
    assert is_us_equity_session(date(2025, 1, 10))
