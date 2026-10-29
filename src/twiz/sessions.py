"""Completed regular US equity sessions; intraday quotes are not daily closes."""
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

# Full US equity closures, distinct from shortened futures/fixed-income sessions.
# ICE/NYSE notices: .../2018/New-York-Stock-Exchange-to-Honor-President-George-H-W-Bush/
# and .../2024/The-New-York-Stock-Exchange-Will-Close-Markets-on-January-9-.../
EXCEPTIONAL_CLOSURES = {date(2018, 12, 5), date(2025, 1, 9)}


def _observed(day: date) -> date:
    return day + timedelta(days=1 if day.weekday() == 6 else -1 if day.weekday() == 5 else 0)


def _weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    correction = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * correction) // 451
    value = h + correction - 7 * m + 114
    return date(year, value // 31, value % 31 + 1)


@lru_cache(maxsize=256)
def _regular_holidays(year: int) -> frozenset:
    holidays = {
        _weekday(year, 1, 0, 3), _weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        date(year, 5, 31) - timedelta(days=date(year, 5, 31).weekday()),
        _observed(date(year, 7, 4)), _weekday(year, 9, 0, 1),
        _weekday(year, 11, 3, 4), _observed(date(year, 12, 25)),
    }
    # NYSE does not observe a Saturday New Year's Day on the preceding Friday.
    for candidate_year in (year - 1, year, year + 1):
        new_year = date(candidate_year, 1, 1)
        if new_year.weekday() != 5:
            holidays.add(_observed(new_year))
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    return frozenset(holidays)


def is_us_equity_session(day: date) -> bool:
    return (day.weekday() < 5 and day not in _regular_holidays(day.year)
            and day not in EXCEPTIONAL_CLOSURES)


def expected_complete_session(report_date: str, now: Optional[datetime] = None) -> str:
    """Use regular close plus a 15-minute completion buffer, conservatively on early-close days."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    if now.tzinfo is None:
        raise ValueError("Current market time must include a timezone.")
    local = now.astimezone(ZoneInfo("America/New_York"))
    candidate = min(date.fromisoformat(report_date), local.date())
    if candidate == local.date() and local.time() < time(16, 15):
        candidate -= timedelta(days=1)
    while not is_us_equity_session(candidate):
        candidate -= timedelta(days=1)
    return candidate.isoformat()
