"""Price-window arithmetic and retrospective political labels use synthetic prices only."""

import json
import math
from copy import deepcopy
from statistics import stdev

import pytest

from twiz.election_study import (
    MIDTERM_YEARS,
    _election_date,
    _window_days,
    build_election_study,
)


def rows(year=2022, closes=None):
    closes = closes or [100*1.01**i for i in range(21)]
    return [{"date": day, "close": close, "open": close, "high": close*1.01,
             "low": close*.99, "volume": 1000, "source": "yfinance",
             "session_complete": True, "price_basis": "synthetic_test_price_return"}
            for day, close in zip(_window_days(_election_date(year)), closes)]


def event(year=2022, **changes):
    return {"year": year, "election_date": _election_date(year).isoformat(),
            "president_party": "D", "house_before": "D", "senate_before": "D",
            "house_after": "R", "senate_after": "D", "source_urls": ["https://example.test"],
            "result_effective_date": f"{year+1}-01-03", **changes}


def one(data=None, events=None):
    result = build_election_study({"^GSPC": data or rows()}, events or [event()])
    return result, next(row for row in result["cycles"]
                        if row["year"] == 2022 and row["asset"] == "^GSPC")


def test_all_thirteen_cycles_and_four_assets_remain_even_when_histories_are_missing():
    result, row = one()
    assert MIDTERM_YEARS == tuple(range(1974, 2023, 4)) and len(MIDTERM_YEARS) == 13
    assert len(result["cycles"]) == 52 and result["primary_asset"] == "^GSPC"
    assert result["coverage"]["^GSPC"] == {"declared_cycles": 13, "available_cycles": 1}
    assert result["coverage"]["QQQ"]["available_cycles"] == 0
    assert all(r["pre"] is None and r["post"] is None for r in result["cycles"]
               if r["status"] == "unknown")
    assert row["status"] == "available"
    assert not any(r["year"] == 2026 for r in result["cycles"])


def test_election_dates_and_ten_returns_use_exact_weekday_sessions():
    assert _election_date(1974).isoformat() == "1974-11-05"
    assert _election_date(2018).isoformat() == "2018-11-06"
    assert _election_date(2022).isoformat() == "2022-11-08"
    days = _window_days(_election_date(2022))
    assert len(days) == 21 and days[10] == "2022-11-08"
    assert days[0] == "2022-10-25" and days[-1] == "2022-11-22"
    _, row = one()
    assert row["pre"]["close_count"] == row["post"]["close_count"] == 11
    assert row["pre"]["return_count"] == row["post"]["return_count"] == 10


def test_one_five_ten_returns_log_ols_and_sample_daily_volatility_are_separate():
    _, row = one()
    for side in ("pre", "post"):
        for n in (1, 5, 10):
            assert row[side]["returns_pct"][str(n)] == pytest.approx((1.01**n-1)*100)
        assert row[side]["log_ols_slope_per_session"] == pytest.approx(math.log(1.01))
        assert row[side]["log_ols_r2"] == pytest.approx(1)
        assert row[side]["daily_volatility_sample_pct"] == pytest.approx(0, abs=1e-12)
    prices = rows(closes=[100+i+2*(i % 3) for i in range(21)])
    _, varied = one(prices)
    expected = [(b["close"]/a["close"]-1)*100 for a, b in zip(prices[10:], prices[11:])]
    assert varied["post"]["daily_volatility_sample_pct"] == pytest.approx(stdev(expected))


def test_pre_short_returns_end_at_election_close_and_post_short_returns_start_there():
    values = [100+i*i for i in range(21)]
    _, row = one(rows(closes=values))
    assert row["pre"]["returns_pct"]["5"] == pytest.approx((values[10]/values[5]-1)*100)
    assert row["post"]["returns_pct"]["5"] == pytest.approx((values[15]/values[10]-1)*100)


def test_post_drawdown_uses_ordered_running_peak_not_only_election_close_to_minimum():
    values = [100]*10+[100, 110, 120, 108, 105, 90, 95, 98, 105, 110, 125]
    _, row = one(rows(closes=values))
    assert row["post"]["max_drawdown_pct"] == pytest.approx(-25)
    assert row["post"]["drawdown_peak_date"] == row["window_dates"][12]
    assert row["post"]["drawdown_trough_date"] == row["window_dates"][15]
    assert row["post"]["return_10d_pct"] == pytest.approx(25)


def test_flat_prices_have_zero_slope_and_volatility_but_undefined_r_squared():
    result, row = one(rows(closes=[100]*21))
    assert row["post"]["log_ols_slope_per_session"] == 0
    assert row["post"]["log_ols_r2"] is None
    assert row["post"]["daily_volatility_sample_pct"] == 0
    assert row["post"]["max_drawdown_pct"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"close": float("nan")}, {"close": float("inf")}, {"close": -1}, {"close": 0},
    {"price_basis": "different"}, {"price_basis": None}, {"session_complete": False},
    {"price_basis": {}}, {"price_basis": True},
    {"session_complete": None}, {"source": "sample"}, {"source": "unknown"},
    {"source": " demo "}, {"price_basis": "unknown"},
])
def test_bad_close_basis_completion_or_sample_source_makes_entire_window_unknown(change):
    data = rows()
    data[15].update(change)
    _, row = one(data)
    assert row["status"] == "unknown" and row["pre"] is None and row["post"] is None


def test_missing_and_duplicate_sessions_never_shift_a_window_or_fill_synthetic_prices():
    data = rows()
    _, missing = one(data[:5]+data[6:])
    _, duplicate = one(data+[deepcopy(data[5])])
    assert missing["status"] == duplicate["status"] == "unknown"
    assert "Missing exact" in missing["reasons"][0]
    assert "Duplicate" in duplicate["reasons"][0]
    data[5].pop("session_complete")
    assert one(data)[1]["status"] == "unknown"


def test_chamber_changes_unified_control_and_new_opposition_takeover_are_distinct():
    data = rows(1974)+rows(1994)+rows(2006)+rows(2022)
    events = [event(1974, president_party="R", house_before="D", senate_before="D",
                    house_after="D", senate_after="D"),
              event(1994, house_after="R", senate_after="R"),
              event(2006, president_party="R", house_before="R", senate_before="R",
                    house_after="D", senate_after="D"), event()]
    result = build_election_study({"^GSPC": data}, events)
    political = {r["year"]: r["politics"] for r in result["cycles"] if r["asset"] == "^GSPC"}
    assert political[1974]["chambers_flipped"] == 0
    assert political[1974]["opposition_control"] == "both"
    assert political[1974]["opposition_takeover"] == "neither"
    for year in (1994, 2006):
        assert political[year]["chambers_flipped"] == 2
        assert political[year]["opposition_takeover"] == "both"
        assert political[year]["resulting_control"] == "divided"
    double = [g for g in result["joint_groups"] if g["asset"] == "^GSPC"
              and g["context"]["opposition_takeover"] == "both"]
    assert len(double) == 2  # President's party keeps the 1994 and 2006 contexts separate.
    assert {tuple(g["years"]) for g in double} == {(1994,), (2006,)}
    assert political[2022]["chambers_flipped"] == 1
    assert political[2022]["opposition_takeover"] == "house"
    assert political[2022]["sources"] == ["https://example.test"]
    assert political[2022]["result_effective_date"] == "2023-01-03"


def test_group_summaries_keep_assets_separate_and_zero_case_groups_unknown():
    result = build_election_study({"^GSPC": rows(), "SPY": rows()}, [event()])
    groups = [r for r in result["groups"] if r["dimension"] == "chambers_flipped"
              and r["category"] == 1]
    assert all(r["metrics"]["post_return_10d_pct"]["n"] == 1
               for r in groups if r["asset"] in {"^GSPC", "SPY"})
    assert all(r["metrics"]["post_return_10d_pct"]["n"] == 0
               and r["metrics"]["post_return_10d_pct"]["mean"] is None
               for r in groups if r["asset"] in {"^NDX", "QQQ"})
    assert not any("probability" in key for row in result["cycles"] for key in row)


def test_political_data_timing_mismatch_and_conflicting_duplicates_do_not_change_prices():
    result, row = one(events=[event(election_date="2022-11-09")])
    assert row["status"] == "available"
    assert row["politics"]["metadata_status"] == "event_date_mismatch"
    result, row = one(events=[event(), event(house_after="D")])
    assert row["politics"]["metadata_status"] == "duplicate_event_records"
    assert row["politics"]["chambers_flipped"] == "unknown"
    assert row["control_information_timing"] == "retrospective_after_election_not_known_at_E0"
    assert any("not known at E0" in text for text in result["limitations"])


def test_price_rescaling_cannot_change_returns_volatility_slopes_or_drawdowns():
    data = rows(closes=[100+i+3*(i % 4) for i in range(21)])
    _, original = one(data)
    scaled = [{**row, "close": row["close"]*17} for row in data]
    _, changed = one(scaled)
    for side in ("pre", "post"):
        for key in ("return_10d_pct", "daily_volatility_sample_pct",
                    "log_ols_slope_per_session", "log_ols_r2", "max_drawdown_pct"):
            assert changed[side][key] == pytest.approx(original[side][key])


def test_positive_finite_prices_with_overflowing_ratios_remain_unknown_without_nan_outputs():
    result, row = one(rows(closes=[1e-300]*10+[1e300]*11))
    assert row["status"] == "unknown"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("basis", [True, 123, {}, []])
def test_uniform_nonstring_bases_never_count_as_a_known_price_convention(basis):
    data = rows()
    for row in data:
        row["price_basis"] = basis
    assert one(data)[1]["status"] == "unknown"


def test_real_caller_named_provider_is_allowed_without_forcing_a_yahoo_dependency():
    data = rows()
    for row in data:
        row["source"] = "official_historical_index_archive"
    assert one(data)[1]["status"] == "available"


def test_party_neutral_new_full_opposition_cohort_includes_final_single_chamber_flips():
    events = [
        event(1974, president_party="R", house_before="D", senate_before="D",
              house_after="D", senate_after="D"),
        event(1986, president_party="R", house_before="D", senate_before="R",
              house_after="D", senate_after="D"),
        event(1990, president_party="R", house_before="D", senate_before="D",
              house_after="D", senate_after="D"),
        event(1994, house_after="R", senate_after="R"),
        event(2006, president_party="R", house_before="R", senate_before="R",
              house_after="D", senate_after="D"),
        event(2014, house_before="R", senate_before="D", house_after="R", senate_after="R"),
        event(),
    ]
    data = [row for record in events for row in rows(record["year"])]
    result = build_election_study({"^GSPC": data}, events)
    primary = [row for row in result["cycles"] if row["asset"] == "^GSPC"]
    broad = [row for row in primary if row["politics"]["new_opposition_controls_both"] is True]
    assert [row["year"] for row in broad] == [1986, 1994, 2006, 2014]
    strict = [row for row in broad if row["politics"]["opposition_takeover"] == "both"]
    assert [row["year"] for row in strict] == [1994, 2006]
    assert {row["politics"]["president_party"] for row in broad} == {"Democratic", "Republican"}
    for row in broad:
        expected = "double_flip" if row["year"] in (1994, 2006) else "new_full_opposition_congress"
        assert row["politics"]["power_shift_severity"] == expected
    stable = [row for row in primary if row["year"] in (1974, 1990)]
    assert all(row["politics"]["new_opposition_controls_both"] is False for row in stable)
    assert all(row["politics"]["power_shift_severity"] == "unchanged" for row in stable)
    group = next(g for g in result["groups"] if g["asset"] == "^GSPC"
                 and g["dimension"] == "new_opposition_controls_both" and g["category"] is True)
    assert group["cycles"] == group["metrics"]["post_return_10d_pct"]["n"] == 4
    joint = [g for g in result["joint_groups"] if g["asset"] == "^GSPC"
             and g["context"]["new_opposition_controls_both"] is True]
    assert sorted(year for g in joint for year in g["years"]) == [1986, 1994, 2006, 2014]


def test_new_full_opposition_is_unknown_when_before_control_is_not_supplied():
    _, row = one(events=[event(house_before=None, senate_after="R")])
    assert row["politics"]["opposition_control"] == "both"
    assert row["politics"]["new_opposition_controls_both"] == "unknown"
