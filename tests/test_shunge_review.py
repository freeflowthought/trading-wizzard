"""A dated source scenario cannot become a perpetual license to average down."""

from copy import deepcopy
from datetime import date, timedelta

from twiz.shunge import apply_structure_guard, build_shunge_context

PROFILE = {"case": {
    "ticker": "QQQ", "as_of": "2026-10-21", "review_after": "2026-10-30",
    "source_support": [704, 705], "source_target": 650,
    "confirmation": 725, "reversal": 740, "prior_zone": [645, 655],
    "prior_zone_month": "2026-04", "high_lookback_sessions": 63,
}}


def plan():
    return {
        "ticker": "QQQ", "date": "2026-10-27", "price_date": "2026-10-27",
        "status": "conditional_plan", "buy_now_pct": 0,
        "model_inputs": {"close": 721, "source": "yfinance"},
        "budget": {"max_new_capital_pct": 10},
        "entries": [{"batch": i, "price": p, "account_pct": pct,
                     "currently_enabled_pct": 2 if i == 1 else 0, "trigger": "确认后启用"}
                    for i, p, pct in [(1, 710, 4), (2, 704, 3.5), (3, 701, 2.5)]],
        "exits": [{"stage": 1, "price": 740, "trim_original_dip_pct": 40},
                  {"stage": 2, "price": 750, "trim_original_dip_pct": 30},
                  {"stage": 3, "price": 760, "trim_original_dip_pct": 30}],
        "stop": {"price": 693}, "limitations": [], "current_action": "等待710确认",
    }


def context(**overrides):
    return {"scenario": {
        "active": True, "ticker": "QQQ", "source_date": "2026-10-21",
        "review_after": "2026-10-30", "data_date": "2026-10-27", "close": 721,
        "support_floor": 704, "support_ceiling": 705, "state": "defensive_watch",
        "confirmation": 725, "reversal": 740, "data_source": "yfinance",
        **overrides,
    }}


def bar(day, close):
    return {"date": day, "source": "yfinance", "session_complete": True,
            "close": close, "low": close - 2, "high": close + 2}


def test_guard_stop_is_strictly_below_every_retained_entry():
    guarded = apply_structure_guard(plan(), context())
    assert [entry["price"] for entry in guarded["entries"]] == [710]
    assert all(guarded["stop"]["price"] < entry["price"] for entry in guarded["entries"])
    assert guarded["budget"]["currently_enabled_pct"] <= 2
    assert sum(exit["trim_original_dip_pct"] for exit in guarded["exits"]) == 100


def test_stale_or_expired_context_cannot_rewrite_current_plan():
    original = plan()
    for overrides in ({"data_date": "2026-10-26"}, {"close": 730},
                      {"source_date": "2026-10-28"}, {"review_after": "2026-10-26"}):
        assert apply_structure_guard(deepcopy(original), context(**overrides)) == original
    unknown_source = context()
    unknown_source["scenario"].pop("data_source")
    assert apply_structure_guard(deepcopy(original), unknown_source) == original


def test_recovery_after_a_completed_daily_break_does_not_resurrect_original_campaign():
    rows = [bar("2026-10-21", 725), bar("2026-10-22", 700),
            bar("2026-10-23", 708), bar("2026-10-26", 715), bar("2026-10-27", 721)]
    derived = build_shunge_context("2026-10-27", {"QQQ": rows}, PROFILE)
    guarded = apply_structure_guard(plan(), derived)
    assert guarded["status"] == "blocked"
    assert guarded["entries"] == [] and guarded["exits"] == []
    assert guarded["budget"]["currently_enabled_pct"] == 0


def test_explicitly_incomplete_bar_cannot_establish_a_daily_break():
    rows = [bar("2026-10-21", 725),
            {**bar("2026-10-22", 700), "session_complete": False},
            bar("2026-10-23", 708), bar("2026-10-26", 715), bar("2026-10-27", 721)]
    derived = build_shunge_context("2026-10-27", {"QQQ": rows}, PROFILE)
    guarded = apply_structure_guard(plan(), derived)
    assert guarded["status"] == "conditional_plan"
    assert guarded["entries"] and guarded["budget"]["currently_enabled_pct"] <= 2


def test_empty_above_floor_entry_set_does_not_falsely_report_current_breakdown():
    original = plan()
    original["entries"] = [{**entry, "price": p}
                           for entry, p in zip(original["entries"], [701, 693, 685])]
    guarded = apply_structure_guard(original, context())
    assert guarded["status"] == "blocked"
    assert "当前收盘已低于704" not in guarded["current_action"]


def test_case_projection_is_frozen_before_later_prices_and_preserves_source_mismatch():
    dates = []
    cursor = date(2026, 10, 21)
    while len(dates) < 260:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor -= timedelta(days=1)
    rows = [bar(day.isoformat(), 725) for day in reversed(dates)]
    rows[-10]["high"] = 763
    next(row for row in rows if row["date"] == "2026-10-16")["low"] = 707.23
    before = build_shunge_context("2026-10-27", {"QQQ": rows}, PROFILE)["case"]
    after = build_shunge_context("2026-10-27", {
        "QQQ": rows + [bar("2026-10-22", 5000), bar("2026-10-28", 1)],
    }, PROFILE)["case"]
    assert before == after
    assert before["available"] and before["measured_target"] == 646
    assert before["prior_friday"]["low"] == 707.23
    assert any("不完全吻合" in note for note in before["limitations"])
