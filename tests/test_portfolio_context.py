from datetime import datetime, timedelta, timezone

import pytest
import yaml
from test_macro_rotation import CYCLE, fixture

from twiz import portfolio_context
from twiz.daily_context import BASIS
from twiz.macro_rotation import DEFAULT_PARAMETERS, _clock
from twiz.portfolio_context import refresh_portfolio_context
from twiz.storage import ProjectPaths, read_json, write_json


def project(tmp_path, n=300):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    market, start, end = fixture(n)
    for rows in market.values():
        for row in rows:
            row["price_basis"] = BASIS
    config = {
        "enabled": True,
        "activated_at": "2023-01-01T00:00:00Z",
        "start": start,
        "parameters": {**DEFAULT_PARAMETERS, "use_events": True, "event_threshold": 0.5},
        "cycle_parameters": CYCLE,
        "reference_source": {"recent_max_drawdown_pct": 23},
    }
    (paths.config / "portfolio-strategy.yml").write_text(yaml.safe_dump(config))
    write_json(paths.processed / "portfolio-history.json", market)
    write_json(paths.processed / "observed-event-history.json", [])
    clock = datetime.fromisoformat(_clock(datetime.fromisoformat(end).date())) + timedelta(hours=1)
    current = {ticker: rows[-2:] for ticker, rows in market.items()}
    return paths, market, current, end, clock


def test_dated_paper_plan_matches_prices_sums100_and_contains_source_receipts(tmp_path):
    paths, market, current, end, clock = project(tmp_path)
    result = refresh_portfolio_context(paths, end, current, analysis_clock=clock)
    plan, trace = result["portfolio_plan"], result["portfolio_training"]
    assert plan["status"] == "available"
    assert plan["target_weights"] == {"QQQ": 1, "TQQQ": 0, "cash": 0}
    assert sum(plan["paper_weights"].values()) == pytest.approx(1)
    assert plan["price_closes"]["QQQ"] == market["QQQ"][-1]["close"]
    assert plan["orders_executed"] is False and plan["paper_holdings_only"] is True
    assert plan["reference_source"]["recent_max_drawdown_pct"] == 23
    assert trace["used_price_history_sha256"] and trace["no_model_search_or_refit"]
    assert read_json(paths.processed / f"portfolio-plan-{end}.json") == plan


def test_weekend_news_uses_actual_analysis_clock_without_rewriting_close_equity(tmp_path):
    paths, market, current, end, clock = project(tmp_path)
    while datetime.fromisoformat(end).weekday() != 4:
        for rows in market.values():
            rows.pop()
        end = market["QQQ"][-1]["date"]
    write_json(paths.processed / "portfolio-history.json", market)
    current = {ticker: rows[-2:] for ticker, rows in market.items()}
    friday = datetime.fromisoformat(_clock(datetime.fromisoformat(end).date()))
    saturday = friday + timedelta(days=1, hours=1)
    event = {
        "id": "news-only",
        "date": saturday.date().isoformat(),
        "available_at": (saturday - timedelta(minutes=1)).isoformat(),
        "direction": -0.5,
        "source": "official news",
        "source_refs": ["official receipt"],
    }
    write_json(paths.processed / "observed-event-history.json", [event])
    result = refresh_portfolio_context(
        paths, saturday.date().isoformat(), current, analysis_clock=saturday
    )
    plan = result["portfolio_plan"]
    assert plan["status"] == "available" and plan["cause"] == "cycle_adverse_event"
    assert plan["target_weights"]["QQQ"] == 0.7
    assert plan["decision_at"] == saturday.isoformat()
    assert plan["price_at"] == friday.isoformat()
    assert plan["pending_order"]["known_at"] == saturday.isoformat()
    assert result["portfolio_training"]["replay"]["equity_curve"][-1]["at"] == friday.isoformat()
    assert not any(
        f["signal_at"] == saturday.isoformat()
        for f in result["portfolio_training"]["replay"]["fills"]
    )


def test_prediction_market_rows_never_trigger_news_only_cycle(tmp_path):
    paths, _, current, end, clock = project(tmp_path)
    write_json(
        paths.processed / "observed-event-history.json",
        [
            {
                "id": "pm",
                "available_at": clock.isoformat(),
                "direction": -0.5,
                "source": "Polymarket",
                "source_refs": ["x"],
            },
            {
                "id": "hidden-pm",
                "available_at": clock.isoformat(),
                "direction": -0.5,
                "probabilities": {"hold": 0.9},
                "source_refs": ["x"],
            },
        ],
    )
    result = refresh_portfolio_context(paths, end, current, analysis_clock=clock)
    assert result["portfolio_plan"]["target_weights"]["QQQ"] == 1
    assert len(result["portfolio_training"]["excluded_news"]) == 2
    assert result["portfolio_training"]["used_news"] == []


def test_failed_live_news_refresh_keeps_known_history_and_marks_policy_unknown(
    tmp_path, monkeypatch
):
    paths, _, current, end, clock = project(tmp_path)
    original = (paths.processed / "observed-event-history.json").read_bytes()
    monkeypatch.setattr(
        portfolio_context, "_download_fred", lambda: (_ for _ in ()).throw(OSError())
    )
    result = refresh_portfolio_context(paths, end, current, analysis_clock=clock, live=True)
    assert result["portfolio_plan"]["status"] == "available"
    assert result["portfolio_plan"]["news_freshness"] == "unknown_refresh_failed"
    assert result["warnings"]
    assert (paths.processed / "observed-event-history.json").read_bytes() == original


def test_missing_latest_asset_fails_unknown_instead_of_reusing_old_plan(tmp_path):
    paths, market, current, end, clock = project(tmp_path)
    refresh_portfolio_context(paths, end, current, analysis_clock=clock)
    market["TQQQ"].pop()
    write_json(paths.processed / "portfolio-history.json", market)
    current["TQQQ"].pop()
    result = refresh_portfolio_context(paths, end, current, analysis_clock=clock)
    assert result["portfolio_plan"]["status"] == "unavailable"
    assert result["portfolio_plan"]["target_weights"] is None
    assert "latest_completed_session_missing" in result["portfolio_plan"]["reason"]


def test_already_open_next_session_is_reference_only_not_past_open_execution(tmp_path):
    paths, _, current, end, clock = project(tmp_path)
    # Retain the requested earlier report date; no next-day prices are invented.
    from twiz.macro_rotation import _next

    opened = datetime.fromisoformat(_clock(_next(datetime.fromisoformat(end).date()), True))
    result = refresh_portfolio_context(paths, end, current, analysis_clock=opened)
    assert result["portfolio_plan"]["status"] == "reference_only"
    assert result["portfolio_plan"]["new_tqqq_increment"] == 0
    assert result["portfolio_plan"]["orders_executed"] is False


def test_live_fred_append_only_new_changes_use_retrieval_clock(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    old = [
        {
            "id": "prior",
            "date": "2026-06-01",
            "source": "FRED",
            "series_id": "DFEDTARU",
            "available_at": "2026-06-02T20:15:00Z",
        }
    ]
    body = (
        b"observation_date,DFEDTARU\n2026-05-01,5\n2026-05-02,4.75\n"
        b"2026-06-02,4.75\n2026-06-03,4.5\n"
    )
    fetched = datetime(2026, 6, 8, 22, tzinfo=timezone.utc)
    monkeypatch.setattr(
        portfolio_context, "_download_fred", lambda: (body, fetched, "https://fred.example")
    )
    rows, receipt = portfolio_context._refresh_news(paths, old)
    assert rows[0] == old[0] and len(rows) == 2
    assert rows[-1]["date"] == "2026-06-03"
    assert rows[-1]["available_at"] == fetched.isoformat()
    assert rows[-1]["availability_basis"] == "actual_fetch_first_observed_not_backdated"
    assert list((paths.raw / "portfolio-news").glob("*/response.csv"))
    assert receipt["body_sha256"]


def test_future_news_payload_unread_and_invalid_analysis_clock_rejected(tmp_path):
    class Future(dict):
        def get(self, key, default=None):
            if key in ("direction", "score", "source_refs"):
                raise AssertionError("future news numeric read")
            return super().get(key, default)

    accepted, excluded = portfolio_context._news(
        [Future(available_at="2099-01-01T00:00:00Z")], datetime.now(timezone.utc)
    )
    assert accepted == [] and excluded
    paths, _, current, end, _ = project(tmp_path)
    with pytest.raises(ValueError):
        refresh_portfolio_context(paths, end, current, analysis_clock=datetime(2026, 1, 1))


def test_policy_activation_prevents_historical_as_then_claim(tmp_path):
    paths, _, current, end, clock = project(tmp_path)
    config = yaml.safe_load((paths.config / "portfolio-strategy.yml").read_text())
    config["activated_at"] = (clock + timedelta(days=1)).isoformat()
    (paths.config / "portfolio-strategy.yml").write_text(yaml.safe_dump(config))
    result = refresh_portfolio_context(paths, end, current, analysis_clock=clock)
    assert result["portfolio_plan"]["status"] == "unavailable"
    assert result["portfolio_plan"]["reason"] == "historical_policy_not_activated"


def test_unreadable_news_cache_is_unknown_not_silently_price_only(tmp_path):
    paths, _, current, end, clock = project(tmp_path)
    (paths.processed / "observed-event-history.json").write_text("{invalid")
    result = refresh_portfolio_context(paths, end, current, analysis_clock=clock)
    assert result["portfolio_plan"]["status"] == "unavailable"
    assert result["portfolio_plan"]["reason"] == "observed_news_history_missing_or_unreadable"


def revision_rows():
    from test_macro_rotation import fixture

    market, _, _ = fixture()
    rows = market["SMH"][-30:]
    old = [dict(row, fetched_at="2026-10-28T20:30:00Z") for row in rows]
    new = [dict(row, fetched_at="2026-10-28T21:22:00Z") for row in rows]
    new[-1]["open"] *= 1.000120487
    return old, new


def test_only_newer_small_latest_bar_revision_is_accepted_and_fully_traced():
    from twiz.daily_context import _merge

    old, new = revision_rows()
    with pytest.raises(ValueError):
        _merge(old, new)
    merged, proof = _merge(old, new, allow_recent_revisions=True)
    assert merged[-1]["open"] == new[-1]["open"]
    revision = proof["accepted_recent_provider_revisions"][0]
    assert revision["date"] == new[-1]["date"] and revision["fields"] == ["open"]
    assert revision["old"]["open"] == old[-1]["open"]
    assert revision["new"]["open"] == new[-1]["open"]
    assert revision["new_fetched_at"] > revision["old_fetched_at"]
    assert revision["basis_factor_from_stable_overlap"] == 1


@pytest.mark.parametrize("problem", ["old", "large", "no_clock", "not_newer", "short_overlap"])
def test_bounded_revision_exception_does_not_admit_other_bad_basis(problem):
    from twiz.daily_context import _merge

    old, new = revision_rows()
    if problem == "old":
        new[0]["open"] *= 1.0002
    elif problem == "large":
        new[-1]["open"] *= 1.002
    elif problem == "no_clock":
        new[-1].pop("fetched_at")
    elif problem == "not_newer":
        new[-1]["fetched_at"] = old[-1]["fetched_at"]
    else:
        new = new[-3:]
    with pytest.raises(ValueError):
        _merge(old, new, allow_recent_revisions=True)


def test_unrelated_later_news_does_not_hide_new_fred_target_change(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    original = [
        {
            "id": "white-house",
            "date": "2026-10-27",
            "source": "White House",
            "available_at": "2026-10-27T20:15:00Z",
        }
    ]
    body = b"observation_date,DFEDTARU\n2026-10-23,4.5\n2026-10-26,4.25\n"
    fetched = datetime(2026, 10, 28, 22, tzinfo=timezone.utc)
    monkeypatch.setattr(
        portfolio_context, "_download_fred", lambda: (body, fetched, "https://fred.example")
    )
    rows, _ = portfolio_context._refresh_news(paths, original)
    assert len(rows) == 2 and rows[-1]["date"] == "2026-10-26"
    assert rows[-1]["direction"] == 0.5
    assert rows[-1]["available_at"] == fetched.isoformat()


def test_reconciled_history_is_persisted_for_later_sessions(tmp_path):
    paths, market, current, end, clock = project(tmp_path)
    write_json(paths.processed / "portfolio-history.json",
               {ticker: rows[:-1] for ticker, rows in market.items()})
    result = refresh_portfolio_context(paths, end, current, analysis_clock=clock)
    assert result["portfolio_plan"]["status"] == "available"
    stored = read_json(paths.processed / "portfolio-history.json")
    assert all(rows[-1]["date"] == end for rows in stored.values())
    assert stored["QQQ"][-1]["close"] == market["QQQ"][-1]["close"]
