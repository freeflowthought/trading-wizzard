"""The author's structural scenario must tighten every part of the same campaign."""
from copy import deepcopy

import pytest

from twiz.shunge import apply_structure_guard, build_shunge_context

DATE = "2026-10-27"
PROFILE = {"case": {
    "ticker": "QQQ", "as_of": "2026-10-21", "review_after": "2026-10-30",
    "source": "data/raw/timelines/october-2026.txt", "source_line": 89,
    "source_support": [704, 705], "source_target": 650, "confirmation": 725, "reversal": 740,
    "prior_zone": [645, 655], "prior_zone_month": "2026-04", "high_lookback_sessions": 63,
}}


def bar(day, close, source="yfinance"):
    return {"date": day, "close": close, "low": close-2, "high": close+2,
            "source": source, "session_complete": True}


def scenario(rows=None, profile=None, report=DATE):
    return build_shunge_context(report, {"QQQ": rows or [bar(DATE, 721.25)]}, profile or PROFILE)


def plan():
    entries = [{"batch": i, "price": price, "account_pct": pct,
                "currently_enabled_pct": pct, "trigger": "旧方案确认条件"}
               for i, price, pct in [(1, 715.53, 4), (2, 706.57, 3.5), (3, 701.03, 2.5)]]
    loss = sum(row["account_pct"]*(1-693.76/row["price"]) for row in entries)
    return {"ticker": "QQQ", "date": DATE, "price_date": DATE, "status": "conditional_plan",
            "buy_now_pct": 0, "model_inputs": {"close": 721.25, "source": "yfinance"},
            "budget": {"max_new_capital_pct": 10, "currently_enabled_pct": 10},
            "entries": entries, "exits": [{"price": price} for price in (739.38, 775, 800)],
            "stop": {"price": 693.76, "modeled_loss_pct": loss}, "limitations": [],
            "current_action": "等待715.53", "entry_reference": "旧715.53成本"}


def test_rebuilds_one_consistent_bounce_trial_and_source_tagged_exit_ladder():
    original = plan()
    guarded = apply_structure_guard(original, scenario())
    assert [row["price"] for row in guarded["entries"]] == [706.57]
    assert guarded["entries"][0]["batch"] == 1 and guarded["entries"][0]["original_batch"] == 2
    assert guarded["entries"][0]["currently_enabled_pct"] == 2
    assert guarded["stop"]["price"] == 704 and guarded["buy_now_pct"] == 0
    assert [row["price"] for row in guarded["exits"]] == [725, 740, 775]
    assert [row["trim_original_dip_pct"] for row in guarded["exits"]] == [40, 30, 30]
    assert [row["reference_kind"] for row in guarded["exits"]] == [
        "author_scenario", "author_scenario", "technical_candidate"]
    assert guarded["exits"][0]["source_date"] == "2026-10-21"
    assert "technical_candidate" in guarded["exits"][2]["source"]
    assert "新观察阻力" in guarded["exits"][0]["status"]
    assert "706.57" in guarded["entry_reference"] and "715.53" not in guarded["current_action"]
    assert original == plan()  # No mutation of the reusable tactical plan.
    for row in guarded["exits"]:
        assert row["reference_return_pct"] == pytest.approx((row["price"]/706.57-1)*100)
    assert guarded["budget"]["estimated_weighted_entry"] == pytest.approx(706.57)
    assert guarded["budget"]["planned_capital_pct"] == 3.5
    assert guarded["budget"]["uncommitted_cash_pct"] == 8
    assert guarded["stop"]["modeled_loss_pct"] == pytest.approx(3.5*(1-704/706.57))
    assert "本系统按日收盘" in guarded["stop"]["trigger"]


def test_first_target_must_offer_at_least_one_r_and_equality_at_stop_is_not_funded():
    original = plan()
    original["entries"][1]["price"] = 704
    guarded = apply_structure_guard(original, scenario())
    assert guarded["status"] == "blocked" and not guarded["entries"]
    assert "当前有效收盘低于" not in guarded["current_action"]


def test_old_macro_block_and_stricter_original_loss_limit_are_not_relaxed():
    original = plan()
    original["status"] = "blocked"
    original["budget"]["currently_enabled_pct"] = 0
    assert apply_structure_guard(original, scenario())["budget"]["currently_enabled_pct"] == 0
    strict = plan()
    strict["stop"]["modeled_loss_pct"] = 0.001
    guarded = apply_structure_guard(strict, scenario())
    assert guarded["stop"]["modeled_loss_pct"] <= 0.001
    assert guarded["budget"]["planned_capital_pct"] < 3.5
    assert guarded["budget"]["currently_enabled_pct"] <= 2


def test_pending_former_second_batch_can_become_first_within_existing_two_percent_authority():
    original = plan()
    original["budget"]["currently_enabled_pct"] = 2
    for row in original["entries"]:
        row["currently_enabled_pct"] = 2 if row["batch"] == 1 else 0
    guarded = apply_structure_guard(original, scenario())
    assert guarded["entries"][0]["original_batch"] == 2
    assert guarded["budget"]["currently_enabled_pct"] == 2


@pytest.mark.parametrize("change", [{"close": 730}, {"data_date": "2026-10-26"},
    {"source_date": "2026-10-28"}, {"data_source": "sample"}, {"close": float("nan")},
    {"confirmation": None}, {"review_after": "2026-10-26"}, {"session_complete": False}])
def test_invalid_stale_or_mismatched_context_cannot_change_a_fresh_plan(change):
    context = scenario()
    context["scenario"].update(change)
    original = plan()
    assert apply_structure_guard(original, context) == original


def test_missing_sample_and_pre_source_cases_do_not_activate_a_source_scenario():
    for context in (build_shunge_context(DATE, {}, {}),
                    scenario([bar(DATE, 721.25, source="sample")]),
                    scenario([bar("2026-10-20", 721.25)], report="2026-10-20")):
        original = plan()
        assert apply_structure_guard(original, context) == original


def test_post_source_daily_break_remains_invalid_after_recovery_and_ignores_future_bars():
    rows = [bar("2026-10-20", 700), bar("2026-10-22", 703), bar(DATE, 721.25),
            bar("2026-10-28", 900)]
    context = scenario(rows)
    assert context["scenario"]["floor_breach"]["date"] == "2026-10-22"
    guarded = apply_structure_guard(plan(), context)
    assert guarded["status"] == "blocked" and guarded["entries"] == guarded["exits"] == []
    assert guarded["budget"]["currently_enabled_pct"] == 0
    assert "即使随后反弹" in guarded["current_action"]
    before = scenario([bar("2026-10-20", 700), bar(DATE, 721.25)])
    after = scenario([bar("2026-10-20", 700), bar(DATE, 721.25), bar("2026-10-28", 600)])
    assert before == after  # Neither a future floor break nor recovery can leak into today's state.
    assert apply_structure_guard(plan(), before)["entries"]


def test_intraday_low_alone_is_not_our_completed_daily_close_invalidation():
    intraday = bar("2026-10-22", 707)
    intraday["low"] = 700
    context = scenario([intraday, bar(DATE, 721.25)])
    assert context["scenario"]["floor_breach"] is None
    assert apply_structure_guard(plan(), context)["entries"]


def test_only_dated_explicit_reassessment_after_latest_break_can_reenable_a_trial():
    rows = [bar("2026-10-22", 703), bar(DATE, 721.25)]
    profile = deepcopy(PROFILE)
    profile["case"]["reassessment"] = {"date": "2026-10-23", "source": "explicit-user-review"}
    assert apply_structure_guard(plan(), scenario(rows, profile))["entries"]
    profile["case"]["reassessment"]["date"] = "2026-10-28"
    assert not apply_structure_guard(plan(), scenario(rows, profile))["entries"]


def test_review_window_expiry_and_missing_higher_candidate_do_not_invent_targets(monkeypatch):
    monkeypatch.setattr("twiz.shunge.expected_complete_session", lambda report: report)
    expired = scenario([bar("2026-11-02", 721.25)], report="2026-11-02")
    assert expired["scenario"]["active"] is False
    original = plan()
    original["exits"] = [{"price": 739}, {"price": 740}]
    guarded = apply_structure_guard(original, scenario())
    assert guarded["exits"][2]["price"] is None
    assert guarded["exits"][2]["trim_original_dip_pct"] == 30
    assert guarded["exits"][2]["reference_return_pct"] is None
