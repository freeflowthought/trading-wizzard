import json
from copy import deepcopy

import pytest
from test_structure import bars

from twiz.swing_forecast import forecast_swing_range


def test_range_forecasts_use_matured_historical_swings_and_label_cutoff():
    rows = bars()
    result = forecast_swing_range(rows, rows[-1]["date"])
    assert result["status"] == "ready"
    assert result["neighbors_used"] == 80
    assert result["training_label_last"] == rows[-6]["date"]
    assert all(row["label_available_date"] <= rows[-6]["date"]
               for row in result["neighbor_proof"])
    prices = result["prices"]
    assert 0 < prices["stress_low"] <= prices["typical_low"] <= prices["low_upper_quantile"]
    assert prices["typical_low"] <= prices["typical_high"]
    assert result["equity_up_probability"] is None
    assert result["raw_frequency_is_calibrated_probability"] is False


def test_future_price_changes_do_not_change_forecast_or_history_proof():
    rows = bars()
    before = deepcopy(rows)
    future = {**rows[-1], "date": "2026-10-28", "close": float("nan")}
    a = forecast_swing_range(rows, rows[-1]["date"])
    b = forecast_swing_range(rows+[future], rows[-1]["date"])
    assert a == b and rows == before


def test_price_unit_change_scales_levels_not_matching_or_raw_direction_frequency():
    rows = bars()
    larger = [{**row, **{key: row[key]*3 for key in ("open", "high", "low", "close")}}
              for row in rows]
    a = forecast_swing_range(rows, rows[-1]["date"])
    b = forecast_swing_range(larger, rows[-1]["date"])
    for key, value in a["prices"].items():
        assert b["prices"][key] == pytest.approx(value*3)
    assert a["raw_neighbor_up_frequency"] == b["raw_neighbor_up_frequency"]
    assert ([row["origin_date"] for row in a["neighbor_proof"]]
            == [row["origin_date"] for row in b["neighbor_proof"]])


def test_missing_session_does_not_shorten_the_forecast_horizon_silently():
    rows = bars()
    result = forecast_swing_range(rows[:100]+rows[101:], rows[-1]["date"])
    assert result["status"] == "unavailable" and result["reason"] == "missing_complete_session"


def test_longer_horizon_retains_longer_maturity_embargo():
    rows = bars()
    result = forecast_swing_range(rows, rows[-1]["date"], horizon_sessions=60)
    assert result["status"] == "ready"
    assert result["training_origin_last"] == rows[-66]["date"]
    assert result["training_label_last"] == rows[-6]["date"]


@pytest.mark.parametrize("params", [{"horizon_sessions": True}, {"embargo_sessions": 0},
                                     {"neighbors": 10}, {"training_sessions": float("nan")},
                                     {"group_weights": {"trend": 1}}])
def test_invalid_research_parameters_fail_explicitly(params):
    with pytest.raises(ValueError):
        forecast_swing_range([], "2026-10-27", **params)


def test_no_forward_target_is_artificially_lifted_over_current_price():
    rows = bars()
    result = forecast_swing_range(rows, rows[-1]["date"])
    target = result["usable_take_profit"]
    assert target is None or target > result["close"]+result["atr"]*.5


def test_unused_noncanonical_vendor_metadata_returns_unavailable():
    rows = bars()
    rows[0]["unused_vendor_field"] = float("nan")
    result = forecast_swing_range(rows, rows[-1]["date"])
    assert result["status"] == "unavailable" and result["reason"] == "noncanonical_source_record"


def test_legacy_unknown_basis_cannot_generate_usable_forecast_levels():
    rows = bars()
    for row in rows:
        row["price_basis"] = "Yahoo缓存未注明复权口径"
    result = forecast_swing_range(rows, rows[-1]["date"])
    assert result["status"] == "unavailable"


def test_extreme_finite_vendor_prices_cannot_produce_unserializable_ready_forecast():
    rows = bars()
    rows[-10].update(open=1e-308, high=1e-308, low=1e-308, close=1e-308)
    result = forecast_swing_range(rows, rows[-1]["date"])
    json.dumps(result, allow_nan=False)
