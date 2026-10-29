from copy import deepcopy
from datetime import datetime, time, timedelta

import pytest
from test_structure import bars

from twiz.structure import _one_structure
from twiz.swing_optimizer import _hash, freeze_candidate_search
from twiz.swing_setups import NY, freeze_weekly_origins, generate_swing_setups


def candidate(**changes):
    value = deepcopy(freeze_candidate_search({"defaults": {"event_variant": "price_only"}})[
        "candidates"][0])
    value["parameters"].update(changes)
    content = {key: value[key] for key in (
        "parameters", "structure_parameters", "event_variant")}
    value["parameters_sha256"] = _hash(content)
    value["id"] = "swing-"+value["parameters_sha256"][:20]
    return value


def opportunity():
    return {"opportunity_id": "QQQ|2026-10-27", "asset": "QQQ",
            "origin_at": "2026-10-27T16:15:00-04:00",
            "watch_expires_at": "2026-11-24T16:15:00-05:00",
            "comparison_available_at": "2027-02-02T16:15:00-05:00",
            "author_explicit_deadline": None}


def test_opportunity_schedule_predeclares_holiday_week_and_maturity_not_returns():
    registry = freeze_weekly_origins("2026-03-30", "2026-04-10")
    origins = registry["origins"]
    assert len(origins) == 4
    assert origins[0]["origin_at"].startswith("2026-04-02T16:15")
    assert origins[2]["origin_at"].startswith("2026-04-10T16:15")
    assert all(row["author_explicit_deadline"] is None for row in origins)
    assert registry == freeze_weekly_origins("2026-03-30", "2026-04-10")
    assert registry["sha256"] == _hash(origins)


def test_incomplete_week_is_not_added_after_seeing_its_price():
    registry = freeze_weekly_origins("2026-10-26", "2026-10-27")
    assert registry["origins"] == []
    assert len(freeze_weekly_origins("2026-03-30", "2026-04-02")["origins"]) == 2


def test_generator_uses_only_predecision_data_and_retains_source_basis():
    rows = bars()
    record = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate())
    assert record["ready_count"] == record["opportunity_count"] == 1
    setup = record["setups"][0]
    assert setup["price_basis"] == "test_adjusted"
    assert setup["data_source"] == "yfinance"
    assert setup["candidate_id"] == candidate()["id"]
    assert setup["parameters_sha256"] == candidate()["parameters_sha256"]
    assert setup["event_context"] is None
    assert setup["author_explicit_deadline"] is None
    future = {**rows[-1], "date": "2026-10-28", "open": 10000,
              "low": 9999, "high": 10001, "close": 10000}
    changed = generate_swing_setups({"QQQ": rows+[future]}, [opportunity()], candidate())
    assert changed == record


def test_lookback_is_used_for_confirmed_pivots_without_shortening_long_ma():
    rows = bars()
    small = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate(lookback_sessions=63))
    large = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate(lookback_sessions=252))
    assert small["policy"]["pivot_lookback_sessions"] == 63
    assert large["policy"]["pivot_lookback_sessions"] == 252
    a = _one_structure("QQQ", rows, "2026-10-27", pivot_lookback_sessions=63)
    b = _one_structure("QQQ", rows, "2026-10-27", pivot_lookback_sessions=252)
    assert a["features"]["sma200"] == b["features"]["sma200"]
    assert len(a["features"]["confirmed_pivots"]) < len(b["features"]["confirmed_pivots"])
    assert (small["setups"][0]["upstream_generation_hash"]
            != large["setups"][0]["upstream_generation_hash"])


def test_missing_history_is_retained_not_dropped_from_opportunity_denominator():
    origins = [opportunity(), {**opportunity(), "opportunity_id": "SPY|2026-10-27", "asset": "SPY"}]
    result = generate_swing_setups({"QQQ": bars()}, origins, candidate())
    assert result["opportunity_count"] == 2 and len(result["dispositions"]) == 2
    assert result["dispositions"][1]["reason"] == "insufficient_predecision_history"


def test_source_forecast_deadline_is_preserved_separately_from_watch_window():
    source = {**opportunity(), "author_explicit_deadline": "2026-12-31"}
    result = generate_swing_setups({"QQQ": bars()}, [source], candidate())
    assert result["setups"][0]["author_explicit_deadline"] == "2026-12-31"


def test_candidate_tampering_and_incomplete_session_are_rejected():
    value = candidate()
    value["parameters"]["lookback_sessions"] = 63
    with pytest.raises(ValueError, match="changed"):
        generate_swing_setups({}, [opportunity()], value)
    morning = {**opportunity(), "origin_at": "2026-10-27T09:30:00-04:00"}
    with pytest.raises(ValueError, match="completed"):
        generate_swing_setups({}, [morning], candidate())
    with pytest.raises(ValueError, match="duplicate"):
        generate_swing_setups({}, [opportunity(), opportunity()], candidate())


@pytest.mark.parametrize("field", ["price_basis", "session_complete"])
def test_legacy_unknown_price_metadata_cannot_become_ready_research_setup(field):
    rows = bars()
    for row in rows:
        del row[field]
    result = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate())
    assert result["ready_count"] == 0


def test_future_event_inputs_are_not_attached_to_a_historical_opportunity():
    context = {opportunity()["opportunity_id"]: {
        "available_at": "2026-10-28T12:00:00Z", "equity_up_probability": .99}}
    result = generate_swing_setups({"QQQ": bars()}, [opportunity()], candidate(),
                                   event_context_by_opportunity=context)
    assert result["ready_count"] == 0
    assert result["dispositions"][0]["reason"] == "unavailable_or_unverifiable_event_context"


@pytest.mark.parametrize("changes", [{"close": float("nan")}, {"volume": -1},
                                     {"price_basis": []}, {"price_basis": "   "},
                                     {"price_basis": "N/A"}])
def test_bad_predecision_record_is_retained_as_unavailable_not_global_exception(changes):
    rows = bars()
    rows[0].update(changes)
    result = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate())
    assert result["opportunity_count"] == len(result["dispositions"]) == 1
    assert result["ready_count"] == 0


def test_generation_hash_binds_structure_output_and_dated_event_inputs(monkeypatch):
    from twiz import swing_setups as module

    rows = bars()
    first = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate())
    original = module._one_structure

    def changed(*args):
        result = original(*args)
        result["zones"]["support"]["upper"] += .1
        return result

    monkeypatch.setattr(module, "_one_structure", changed)
    second = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate())
    assert (first["setups"][0]["upstream_generation_hash"]
            != second["setups"][0]["upstream_generation_hash"])
    context = {opportunity()["opportunity_id"]: {
        "available_at": opportunity()["origin_at"], "equity_up_probability": .7}}
    third = generate_swing_setups({"QQQ": rows}, [opportunity()], candidate(),
                                 event_context_by_opportunity=context)
    assert (second["setups"][0]["upstream_generation_hash"]
            != third["setups"][0]["upstream_generation_hash"])


def test_conservative_calendar_maturity_accounts_for_dst():
    registry = freeze_weekly_origins("2026-10-23", "2026-10-23", watch_sessions=5,
                                   maximum_holding_sessions=10)
    origin = registry["origins"][0]
    parsed = datetime.fromisoformat(origin["comparison_available_at"])
    assert parsed.timetz().hour == 16 and parsed.utcoffset() == timedelta(hours=-5)
    assert parsed.astimezone(NY).time() == time(16, 15)


def test_cached_rule_calendar_still_observes_new_exceptional_closures(monkeypatch):
    from datetime import date

    from twiz import sessions

    day = date(2026, 10, 28)
    assert sessions.is_us_equity_session(day)
    monkeypatch.setattr(sessions, "EXCEPTIONAL_CLOSURES", sessions.EXCEPTIONAL_CLOSURES | {day})
    assert not sessions.is_us_equity_session(day)
