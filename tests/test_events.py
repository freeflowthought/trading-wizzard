import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.events import (
    build_event_calendar,
    earnings_events,
    manual_events,
    rule_based_events,
    seasonal_context,
)


class EventCalendarTests(unittest.TestCase):
    def test_fomc_in_window(self) -> None:
        events = rule_based_events(date(2026, 10, 20), horizon_days=14)
        fomc = [e for e in events if e["kind"] == "fomc"]
        self.assertTrue(fomc)
        self.assertEqual(fomc[0]["date"], "2026-10-28")

    def test_midterm_election_day_2026(self) -> None:
        events = rule_based_events(date(2026, 10, 27), horizon_days=14)
        elections = [e for e in events if e["kind"] == "election"]
        self.assertTrue(elections)
        self.assertEqual(elections[0]["date"], "2026-11-03")
        self.assertIn("midterm", elections[0]["name"])

    def test_opex_is_third_friday(self) -> None:
        events = rule_based_events(date(2026, 9, 1), horizon_days=30)
        opex = [e for e in events if e["kind"] in ("opex", "quad_witching")]
        self.assertTrue(opex)
        # September 2026 third Friday is the 18th — the expiry week the
        # timeline flags for concentrated short positioning.
        self.assertEqual(opex[0]["date"], "2026-09-18")
        self.assertEqual(opex[0]["kind"], "quad_witching")

    def test_earnings_events_filtered_by_window(self) -> None:
        events = earnings_events(
            {"NVDA": "2026-11-19", "AAPL": "2027-03-01"},
            date(2026, 10, 27),
            horizon_days=45,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tickers"], ["NVDA"])

    def test_manual_events_and_seasonality(self) -> None:
        rows = [
            {"date": "2026-11-05", "name": "Lockup expiry", "tickers": ["cbrs"],
             "importance": 4, "note": "Wait for the unlock before batch 2."}
        ]
        events = manual_events(rows, date(2026, 10, 27), horizon_days=45)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tickers"], ["CBRS"])
        sept = seasonal_context(date(2026, 9, 10))
        self.assertEqual(sept["risk"], "high")

    def test_calendar_sorted_with_days_away(self) -> None:
        events = build_event_calendar(date(2026, 10, 27), horizon_days=45)
        self.assertTrue(events)
        dates = [e["date"] for e in events]
        self.assertEqual(dates, sorted(dates))
        self.assertTrue(all("days_away" in e for e in events))


if __name__ == "__main__":
    unittest.main()
