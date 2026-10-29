import copy
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from twiz import event_bridge_samples as samples
from twiz.event_bridge import fit_event_bridge
from twiz.ingest.historical_event_features import FEATURE_VERSION
from twiz.sessions import is_us_equity_session

NY = ZoneInfo("America/New_York")


def prices(n=300):
    rows, day = [], date(2023, 1, 3)
    while len(rows) < n:
        if is_us_equity_session(day):
            close = 100 + len(rows) * 0.1 + math.sin(len(rows) / 5)
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": close - 0.1,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1000,
                    "source": "yfinance",
                    "price_basis": "test_split_adjusted",
                    "session_complete": True,
                }
            )
        day += timedelta(days=1)
    return rows


def clock(day):
    return datetime.combine(date.fromisoformat(day), time(16, 15), NY).astimezone(timezone.utc)


def proof(origin, identity="1"):
    return {
        "quote_available_at": (origin - timedelta(hours=1)).isoformat(),
        "market_id": identity,
        "yes_token_id": identity + "1",
        "no_token_id": identity + "2",
        "raw_quote_references": [
            {
                "body_file": "raw/source.body",
                "data_index": 0,
                "fetched_at": "2026-10-28T15:00:00+00:00",
            }
        ],
    }


def measurement(origin, family, relation="lt", lower=None, upper=None, p=0.3):
    benchmark = samples.BENCHMARKS[family]
    return {
        "status": "available",
        "family": family,
        "yes_probability": p,
        "no_probability": 1 - p,
        "benchmark": benchmark,
        "specification": {
            "family": family,
            "unit": "jobs" if family == "payroll_jobs" else "percent",
            "relation": relation,
            "lower": lower,
            "upper": upper,
        },
        "source_refs": [proof(origin)],
        "question": "Exact frozen economic question",
    }


class Adapter:
    manifest_sha256 = "1" * 64

    def __init__(self, available=True, transform=None):
        self.available, self.transform, self.calls = available, transform, []

    def lookup(self, origin):
        self.calls.append(origin)
        channels = {
            c: {"status": "missing", "reason_codes": ["not_available"]}
            for c in ("fed", "inflation_cpi", "jobs_growth")
        }
        if self.available:
            channels["fed"] = {
                "status": "available",
                "group_id": "us_fomc:2023-12",
                "declared_event_end": (origin + timedelta(days=10)).isoformat(),
                "probabilities": {"cut": 0.2, "hold": 0.7, "hike": 0.1},
                "source_refs": [proof(origin)],
                "historical_pit_verified": False,
            }
        result = {
            "origin": origin.isoformat(),
            "source_manifest_sha256": self.manifest_sha256,
            "feature_version": FEATURE_VERSION,
            "channels": channels,
        }
        return self.transform(result, origin) if self.transform else result


def build(rows=None, adapter=None, **kwargs):
    rows = rows or prices()
    origin = rows[220]["date"]
    options = {"start": origin, "end": origin, "as_of": datetime(2026, 4, 27, tzinfo=NY)}
    options.update(kwargs)
    return samples.build_event_bridge_samples(
        {"QQQ": rows, "SPY": copy.deepcopy(rows)}, adapter or Adapter(), **options
    )


class NoNumericRead(dict):
    def get(self, key, default=None):
        if key in ("open", "high", "low", "close", "volume", "source_timestamp"):
            raise AssertionError(f"Forbidden numeric read: {key}")
        return super().get(key, default)

    def __getitem__(self, key):
        if key in ("open", "high", "low", "close", "volume", "source_timestamp"):
            raise AssertionError(f"Forbidden numeric read: {key}")
        return super().__getitem__(key)


def test_paired_cohort_39_columns_and_strict_mature_label():
    rows = prices()
    result = build(rows)
    assert len(result["rows"]) == 2 and len(result["feature_names"]) == 39
    assert len(result["baseline_feature_names"]) == 11
    for row in result["rows"]:
        assert row["features"][:11] == row["baseline_features"]
        assert row["features_available_at"] == row["origin_at"]
        assert row["label_available_at"] == row["outcome_available_at"]
        assert row["label_definition"] == "close_20_sessions_ahead_above_origin_completed_close"
        assert row["label"] == int(rows[240]["close"] > rows[220]["close"])
        assert row["label_context"]["future_session_dates"] == [r["date"] for r in rows[221:241]]
        assert row["price_context"]["history_end"] == row["origin_date"]
        assert row["source_policy_sha256"] == result["source_policy_sha256"]
    json.dumps(result, allow_nan=False)


def test_unmatured_window_excluded_before_future_price_or_adapter_access():
    rows, adapter = prices(), Adapter()
    for i in range(221, len(rows)):
        rows[i] = NoNumericRead(rows[i])
    result = build(rows, adapter, as_of=clock(rows[240]["date"]))
    assert result["rows"] == [] and adapter.calls == []
    assert set(result["summary"]["exclusion_reasons"]) == {
        "label_window_unmatured_or_outside_pretest"
    }


def test_testperiod_numeric_rows_never_read_or_hashed():
    rows = prices()
    a = build(rows)
    b = build(rows + [NoNumericRead({"date": "2026-04-27", "close": float("nan")})])
    assert a == b


def test_unused_pretest_future_rows_after_label_window_are_not_read():
    rows = prices()
    expected = build(rows)
    for i in range(241, len(rows)):
        rows[i] = NoNumericRead(rows[i])
    assert build(rows) == expected


def test_price_features_receive_only_preorigin_prefix(monkeypatch):
    seen, original = [], samples._feature

    def feature(rows, index):
        seen.append((rows[-1]["date"], index, len(rows)))
        return original(rows, index)

    monkeypatch.setattr(samples, "_feature", feature)
    rows = prices()
    build(rows)
    assert seen == [(rows[220]["date"], 220, 221)] * 2


def test_future_close_tie_is_label_zero():
    rows = prices()
    close = rows[220]["close"]
    rows[240].update(open=close, close=close, low=close - 1, high=close + 1)
    assert {row["label"] for row in build(rows)["rows"]} == {0}


def test_all_missing_event_cohort_reads_no_price_numbers():
    rows = [NoNumericRead(row) for row in prices()]
    result = build(rows, Adapter(available=False))
    assert result["rows"] == []
    assert result["summary"]["exclusion_reasons"] == {"all_event_channels_missing": 2}


def test_available_event_with_future_availability_excluded_before_prices():
    def future(payload, origin):
        payload["channels"]["fed"]["source_refs"][0]["quote_available_at"] = (
            origin + timedelta(seconds=1)
        ).isoformat()
        return payload

    result = build([NoNumericRead(row) for row in prices()], Adapter(transform=future))
    assert result["rows"] == []
    assert result["summary"]["exclusion_reasons"] == {"event_features_not_available_at_origin": 2}


def test_missing_imputation_and_family_interaction_are_explicit():
    def events(payload, origin):
        payload["channels"]["jobs_growth"] = {
            "status": "available",
            "group_id": "us_employment_situation:2023-11",
            "declared_event_end": (origin + timedelta(days=3)).isoformat(),
            "source_refs": [proof(origin)],
            "measurements": {
                "payroll_jobs": measurement(origin, "payroll_jobs", "interval", 100000, 150000, 0.3)
            },
        }
        return payload

    row = build(adapter=Adapter(transform=events))["rows"][0]
    f = row["feature_values"]
    assert f["payroll_jobs_p_yes"] == 0.3
    assert f["payroll_jobs_threshold_offset"] == -0.25
    assert f["payroll_jobs_interval_width"] == 0.5
    assert f["payroll_jobs_relation_code"] == 2 and f["payroll_jobs_p_yes_relation"] == 0.6
    assert f["payroll_jobs_missing"] == 0
    assert f["cpi_headline_mom_p_yes"] == 0 and f["cpi_headline_mom_missing"] == 1
    assert "ML zero imputation" in build()["source_policy"]["missing_policy"]
    assert row["measurement_specifications"]["payroll_jobs"]["specification"]["upper"] == 150000


def test_lt_interaction_uses_yes_without_converting_no_to_economic_inverse():
    def events(payload, origin):
        payload["channels"]["jobs_growth"] = {
            "status": "available",
            "group_id": "us_employment_situation:2023-11",
            "declared_event_end": (origin + timedelta(days=3)).isoformat(),
            "source_refs": [proof(origin)],
            "measurements": {
                "payroll_jobs": measurement(origin, "payroll_jobs", "lt", None, 150000, 0.8)
            },
        }
        return payload

    f = build(adapter=Adapter(transform=events))["rows"][0]["feature_values"]
    assert f["payroll_jobs_p_yes"] == 0.8 and f["payroll_jobs_p_yes_relation"] == -0.8


def test_episode_primary_nearest_with_stable_channel_tie_and_all_groups():
    def events(payload, origin):
        for c, g in [("jobs_growth", "jobs-id"), ("inflation_cpi", "cpi-id")]:
            payload["channels"][c] = {
                "status": "available",
                "group_id": g,
                "declared_event_end": (origin + timedelta(days=3)).isoformat(),
                "source_refs": [proof(origin)],
                "measurements": {},
            }
        return payload

    row = build(adapter=Adapter(transform=events))["rows"][0]
    assert row["episode_id"] == "cpi-id"
    assert row["underlying_group_ids"] == ["cpi-id", "jobs-id", "us_fomc:2023-12"]


@pytest.mark.parametrize("which", ["prefix", "label"])
def test_missing_session_does_not_shrink_label_or_feature_horizon(which):
    rows = prices()
    origin = rows[220]["date"]
    del rows[230 if which == "label" else 100]
    result = build(rows, start=origin, end=origin)
    assert result["rows"] == []
    assert result["summary"]["exclusion_reasons"] == {"missing_or_duplicate_complete_session": 2}


def test_incomplete_label_bar_rejected_before_its_numeric_values():
    rows = prices()
    rows[230] = NoNumericRead({**rows[230], "session_complete": False})
    result = build(rows)
    assert result["rows"] == []
    assert all(e["detail"] == "price_session_not_confirmed_complete" for e in result["exclusions"])


@pytest.mark.parametrize(
    "change,detail",
    [
        ({"price_basis": "unknown"}, "unknown_price_basis"),
        ({"price_basis": "different_adjustment"}, "price_basis_changed"),
        ({"source": "sample"}, "unsupported_or_sample_price_source"),
        ({"close": float("nan")}, "invalid_ohlcv"),
    ],
)
def test_inconsistent_or_invalid_future_price_cannot_become_training_label(change, detail):
    rows = prices()
    rows[230].update(change)
    result = build(rows)
    assert result["rows"] == [] and all(e["detail"] == detail for e in result["exclusions"])


def test_duplicate_session_rejected():
    rows = prices()
    rows.append(copy.deepcopy(rows[220]))
    result = build(rows, start=rows[220]["date"], end=rows[220]["date"])
    assert result["rows"] == []
    assert result["summary"]["exclusion_reasons"] == {"missing_or_duplicate_complete_session": 2}


def test_postorigin_prices_do_not_change_price_features_or_policy_hash():
    rows = prices()
    a = build(rows)
    rows[240].update(close=200, open=199, low=198, high=201)
    b = build(rows)
    assert a["source_policy_sha256"] == b["source_policy_sha256"]
    assert a["rows"][0]["features"] == b["rows"][0]["features"]
    assert a["rows"][0]["price_context"] == b["rows"][0]["price_context"]
    assert a["rows"][0]["label_context"] != b["rows"][0]["label_context"]


def test_nonprimary_horizon_has_distinct_definition_and_policy():
    row = build(horizon_sessions=5)["rows"][0]
    assert row["label_definition"] == "close_5_sessions_ahead_above_origin_completed_close"
    assert len(row["label_context"]["future_session_dates"]) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"end": "2026-04-27"},
        {"as_of": datetime(2025, 1, 1)},
        {"horizon_sessions": True},
        {"horizon_sessions": 0},
        {"horizon_sessions": 127},
    ],
)
def test_invalid_or_testperiod_requests_fail_before_prices(kwargs):
    with pytest.raises(ValueError):
        build([NoNumericRead(row) for row in prices()], **kwargs)


def test_price_availability_origin_clock_uses_new_york_dst():
    result = build()
    row = result["rows"][0]
    expected = datetime.combine(date.fromisoformat(row["origin_date"]), time(16, 15), NY)
    assert datetime.fromisoformat(row["origin_at"]) == expected


def test_measurement_proof_cannot_hide_future_quote_behind_older_channel_proof():
    def events(payload, origin):
        m = measurement(origin, "payroll_jobs", "lt", None, 150000)
        m["source_refs"][0]["quote_available_at"] = (origin + timedelta(seconds=1)).isoformat()
        payload["channels"]["jobs_growth"] = {
            "status": "available",
            "group_id": "jobs-id",
            "declared_event_end": (origin + timedelta(days=3)).isoformat(),
            "source_refs": [proof(origin)],
            "measurements": {"payroll_jobs": m},
        }
        return payload

    result = build([NoNumericRead(row) for row in prices()], Adapter(transform=events))
    assert result["rows"] == []
    assert result["summary"]["exclusion_reasons"] == {"event_features_not_available_at_origin": 2}


def test_observed_zero_probability_is_distinct_from_missing_imputation():
    def events(payload, origin):
        payload["channels"]["jobs_growth"] = {
            "status": "available",
            "group_id": "jobs-id",
            "declared_event_end": (origin + timedelta(days=3)).isoformat(),
            "source_refs": [proof(origin)],
            "measurements": {
                "payroll_jobs": measurement(origin, "payroll_jobs", "lt", None, 150000, 0)
            },
        }
        return payload

    f = build(adapter=Adapter(transform=events))["rows"][0]["feature_values"]
    assert f["payroll_jobs_p_yes"] == 0 and f["payroll_jobs_missing"] == 0
    assert f["cpi_headline_mom_p_yes"] == 0 and f["cpi_headline_mom_missing"] == 1


def test_source_input_rows_and_adapter_fields_remain_unchanged():
    rows = prices()
    before = copy.deepcopy(rows)
    result = build(rows)
    assert rows == before
    assert result["rows"][0]["raw_channels"]["fed"]["probabilities"] == {
        "cut": 0.2,
        "hold": 0.7,
        "hike": 0.1,
    }


def test_fit_row_adapter_preserves_values_labels_proofs_and_original_bundle():
    bundle = build()
    before = copy.deepcopy(bundle)
    rows = samples.as_event_bridge_fit_rows(bundle)
    assert bundle == before and len(rows) == len(before["rows"])
    for row, original in zip(rows, before["rows"]):
        assert row["feature_vector"] == original["features"]
        assert list(row["features"]) == bundle["feature_names"]
        assert row["features"] == original["feature_values"]
        for key, value in original.items():
            if key != "features":
                assert row[key] == value
    rows[0]["raw_channels"]["fed"]["probabilities"]["cut"] = 0.99
    assert bundle == before


def test_actual_fitter_accepts_converted_synthetic_builder_rows():
    bundle = build()
    rows = samples.as_event_bridge_fit_rows(bundle)
    result = fit_event_bridge(
        rows,
        as_of=datetime(2026, 4, 27, tzinfo=NY),
        feature_names=bundle["feature_names"],
        baseline_feature_names=bundle["baseline_feature_names"],
        label_definition="close_20_sessions_ahead_above_origin_completed_close",
    )
    assert result["accepted_samples"] == len(rows) > 0
    assert result["status"] == "insufficient_history"
    assert not any(row["reason"] == "missing_or_invalid_feature" for row in result["excluded"])


@pytest.mark.parametrize("mutation", ["value", "name", "length", "nan", "bool", "baseline"])
def test_fit_row_adapter_rejects_inconsistent_shape_without_silent_repair(mutation):
    bundle = build()
    row = bundle["rows"][0]
    if mutation == "value":
        row["feature_values"][bundle["feature_names"][0]] += 1
    elif mutation == "name":
        bundle["feature_names"][1] = bundle["feature_names"][0]
    elif mutation == "length":
        row["features"].pop()
    elif mutation == "nan":
        row["features"][0] = float("nan")
    elif mutation == "bool":
        row["features"][0] = True
    else:
        row["baseline_features"][0] += 1
    with pytest.raises(ValueError, match="alignment|inventory|baseline"):
        samples.as_event_bridge_fit_rows(bundle)
