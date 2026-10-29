"""Regression checks for missing inputs and mismatched financial horizons."""
import json
import math
import sys
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from twiz.events import rule_based_events
from twiz.fundamentals import build_fundamental_snapshots, fetch_yfinance_fundamentals
from twiz.ipo import build_ipo_snapshot, ipo_events
from twiz.models import OptionContract
from twiz.options import (
    compute_event_context,
    fetch_yfinance_option_contracts,
    summarize_options,
)
from twiz.playbook import build_playbook, leadership_snapshot, macro_gate


class StatementFrame:
    """Small date-addressable statement fixture, independent of optional pandas."""
    def __init__(self, rows):
        self.rows = rows
        self.index = list(rows)
        self.columns = list(next(iter(rows.values())))
        self.loc = self

    def __getitem__(self, key):
        label, column = key
        return self.rows[label][column]


class SignalQualityTests(unittest.TestCase):
    def test_partial_and_nan_statements_cannot_generate_sell_signal(self):
        for fields in ({"cash_and_equivalents": 100}, {"free_cash_flow": math.nan}):
            with self.subTest(fields=fields), patch(
                "twiz.fundamentals.fetch_yfinance_fundamentals", return_value=fields
            ):
                result = build_fundamental_snapshots(["TEST"], live=True)["TEST"]
                self.assertEqual(result["label"], "unknown")
                self.assertIsNone(result["score"])
                self.assertTrue(result["warnings"])

    def test_statements_use_matching_periods_and_actual_previous_year(self):
        periods = ["2025-12-31", "2024-12-31", "2023-12-31"]
        cf = StatementFrame({
            "Free Cash Flow": dict(zip(periods, [math.nan, 50, 40])),
            "Operating Cash Flow": dict(zip(periods, [100, 80, 60])),
            "Capital Expenditure": dict(zip(periods, [-30, -25, -20])),
        })
        bs = StatementFrame({
            "Ordinary Shares Number": dict(zip(periods, [120, 100, 70])),
            "Cash And Cash Equivalents": dict(zip(periods, [50, 40, 30])),
            "Total Liabilities Net Minority Interest": dict(zip(periods, [900, 800, 700])),
        })
        with patch.dict(sys.modules, {"yfinance": SimpleNamespace(
            Ticker=lambda _: SimpleNamespace(cashflow=cf, balance_sheet=bs)
        )}):
            fin = fetch_yfinance_fundamentals("TEST")
        self.assertEqual(fin["shares_change_pct"], 20)
        self.assertEqual(fin["statement_date"], "2025-12-31")
        self.assertEqual(fin["shares_comparison_date"], "2024-12-31")
        self.assertNotIn("free_cash_flow", fin)  # Do not silently substitute last year's FCF.
        self.assertNotIn("total_debt", fin)  # Liabilities is not debt.

    def test_trend_gate_preserves_observed_uptrend(self):
        rows = [{"close": 100.0, "high": 101.0}] * 30
        technical = {"regime": "uptrend", "levels": [
            {"price": 95.0}, {"price": 110.0},
        ]}
        playbook = build_playbook("SPY", rows, technical, as_of=date(2026, 10, 27))
        self.assertEqual(playbook["trend_gate"]["state"], "uptrend")
        self.assertNotIn("downtrend/correction", playbook["trend_gate"]["rule"])
        self.assertFalse(playbook["staged_entries"])  # Macro gate has no evidence.
        self.assertIsNone(playbook["sell_put_strike"])

    def test_insufficient_leader_history_or_missing_benchmark_is_unknown(self):
        self.assertEqual(leadership_snapshot({"MU": [{"close": 100}]})["state"], "unknown")
        full = [{"close": 100}] * 21
        self.assertEqual(leadership_snapshot({"MU": full, "INTC": full})["state"], "unknown")

    def test_straddle_requires_same_strike_and_expiry(self):
        call = OptionContract("C", "call", 100, "2026-11-20", bid=2, ask=3)
        put = OptionContract("P", "put", 105, "2026-11-20", bid=3, ask=4)
        self.assertIsNone(compute_event_context([call, put], 100)["expected_move_pct"])
        put.strike = 100
        put.expiration = "2026-12-18"
        self.assertIsNone(compute_event_context([call, put], 100)["expected_move_pct"])
        put.expiration = call.expiration
        put.bid = 5  # Crossed quote is not an executable midpoint.
        self.assertIsNone(compute_event_context([call, put], 100)["expected_move_pct"])

    def test_positioning_uses_full_chain_before_display_truncation(self):
        contracts = [OptionContract(str(i), side, 90+i, "2026-11-20", bid=2, ask=3,
                                    open_interest=1)
                     for side in ("call", "put") for i in range(30)]
        contracts.append(OptionContract("far_put", "put", 10, "2026-11-20",
                                        bid=0.1, ask=0.2, open_interest=300))
        snap = summarize_options("TEST", 100, contracts, "2026-11-20")
        self.assertEqual(len(snap.contracts), 20)
        self.assertEqual(snap.skew_summary["put_wall"], 10)
        self.assertEqual(snap.skew_summary["put_call_oi_ratio"], 11)

    def test_pre_earnings_expiry_does_not_claim_earnings_move(self):
        event = {"date": "2026-11-19", "kind": "earnings", "name": "TEST earnings",
                 "tickers": ["TEST"], "importance": 5}
        rows = [{"close": 100, "high": 101, "date": "2026-10-27"}] * 30
        options = {"expiration": "2026-11-06", "skew_summary": {"expected_move_pct": 6.5},
                   "underlying_date": "2026-10-27", "underlying_spot": 100,
                   "fetched_at": "2026-10-28T00:30:00+00:00"}
        pb = build_playbook("TEST", rows, {}, options_snapshot=options, events=[event],
                            as_of=date(2026, 10, 27))
        self.assertNotIn("expected_move_pct", pb["next_event"])
        options["expiration"] = "2026-11-20"
        pb = build_playbook("TEST", rows, {}, options_snapshot=options, events=[event],
                            as_of=date(2026, 10, 27))
        self.assertEqual(pb["next_event"]["expected_move_pct"], 6.5)
        self.assertIn("through expiry", pb["next_event"]["rule"])

    def test_elections_only_occur_in_even_years_and_estimates_are_labeled(self):
        events = rule_based_events(date(2027, 10, 27), horizon_days=750)
        elections = [e for e in events if e["kind"] == "election"]
        self.assertTrue(elections)
        self.assertTrue(all(date.fromisoformat(e["date"]).year % 2 == 0 for e in elections))
        jobs = next(e for e in events if e["kind"] == "jobs_report")
        self.assertEqual(jobs["precision"], "approximate")
        self.assertIn("estimated", jobs["name"])

    def test_unverified_ipo_has_no_price_signals_or_supply_events(self):
        row = {"ticker": "TEST", "ipo_date": "2026-06-12", "ipo_price": 100,
               "early_release_days": 15, "lockup_days": 180}
        snap = build_ipo_snapshot(row, close=250, as_of=date(2026, 6, 20))
        self.assertEqual(snap["stage"], "unverified")
        self.assertFalse(snap["verdicts"])
        self.assertEqual(ipo_events([row], date(2026, 6, 20)), [])
        verified = {**row, "verified": True, "source_url": "https://example.com/prospectus"}
        verified.pop("early_release_days")
        events = ipo_events([verified], date(2026, 6, 20))
        self.assertFalse(any(e["kind"] == "ipo_early_release" for e in events))

    def test_macro_ignores_nan_and_reports_last_valid_observation_date(self):
        self.assertEqual(
            macro_gate([{"series_id": "DGS10", "value": math.nan}])["state"], "unknown"
        )
        gate = macro_gate([{"series_id": "DGS10", "value": ".", "history": [
            {"date": "2026-10-23", "value": "5.1"},
            {"date": "2026-10-26", "value": "."},
        ]}], as_of=date(2026, 10, 27))
        self.assertEqual(gate["state"], "risk_off")
        self.assertEqual(gate["dgs10_date"], "2026-10-23")
        self.assertIn("latest observation missing", gate["reasons"][0])

    def test_provider_missing_or_nan_fields_leave_finite_snapshot(self):
        rows = [
            {"strike": math.nan},
            {"strike": 100, "bid": None, "ask": math.nan, "lastPrice": math.nan,
             "volume": math.nan, "openInterest": None, "impliedVolatility": math.nan},
        ]
        frame = SimpleNamespace(empty=False, iterrows=lambda: iter(enumerate(rows)))
        instrument = SimpleNamespace(
            options=["2026-11-20"],
            option_chain=lambda _: SimpleNamespace(calls=frame, puts=None),
        )
        with patch.dict(sys.modules, {"yfinance": SimpleNamespace(Ticker=lambda _: instrument)}):
            contracts, expiry, warnings = fetch_yfinance_option_contracts("TEST", 100)
        self.assertEqual(len(contracts), 1)
        self.assertTrue(warnings)
        snap = summarize_options("TEST", 100, contracts, expiry)
        self.assertIsNone(snap.greeks["delta"])
        self.assertIsNone(snap.skew_summary["call_iv_atm"])
        self.assertTrue(snap.warnings)
        json.dumps(snap.to_dict(), allow_nan=False)

    def test_stale_macro_cannot_enable_current_position_sizes(self):
        rows = [{"close": 100.0, "high": 101.0}] * 30
        macro = [{"series_id": "DGS10", "date": "2020-01-01", "value": "4.0"}]
        gate = macro_gate(macro, as_of=date(2026, 10, 27))
        self.assertEqual(gate["state"], "unknown")
        pb = build_playbook("SPY", rows, {"levels": [{"price": 95.0}]},
                            macro_rows=macro, as_of=date(2026, 10, 27))
        self.assertFalse(pb["staged_entries"])
        self.assertIsNone(pb["sell_put_strike"])

    def test_playbook_does_not_reuse_options_from_a_different_underlying_snapshot(self):
        rows = [{"close": 100, "high": 101, "date": "2026-10-27"}] * 30
        valid = {"underlying_date": "2026-10-27", "underlying_spot": 100,
                 "fetched_at": "2026-10-28T00:30:00+00:00", "expiration": "2026-11-20",
                 "skew_summary": {"expected_move_pct": 6.5, "put_call_oi_ratio": 1.4,
                                  "put_wall": 90}}
        current = build_playbook("TEST", rows, {}, options_snapshot=valid,
                                 as_of=date(2026, 10, 27))
        self.assertEqual(current["expected_move_pct"], 6.5)
        for change in ({"underlying_date": None}, {"underlying_date": "2026-10-26"},
                       {"underlying_spot": 90}, {"underlying_spot": math.nan},
                       {"fetched_at": "2026-10-27T20:30:00"},
                       {"fetched_at": "2026-10-27T00:30:00+00:00"}, {"fetched_at": None}):
            with self.subTest(change=change):
                pb = build_playbook("TEST", rows, {}, options_snapshot={**valid, **change},
                                    as_of=date(2026, 10, 27))
                self.assertIsNone(pb["expected_move_pct"])
                self.assertIsNone(pb["put_call_oi_ratio"])
                self.assertIsNone(pb["put_wall"])
                self.assertTrue(any("OI withheld" in note for note in pb["notes"]))


if __name__ == "__main__":
    unittest.main()
