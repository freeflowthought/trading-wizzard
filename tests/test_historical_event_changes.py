import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from test_historical_event_features import (
    GDP_RULES,
    JOB_RULES,
    cpi,
    event,
    fed_events,
    load,
    market,
)

from twiz.ingest.historical_event_changes import (
    HistoricalEventChanges,
    binary_entropy_bits,
    reference_origin,
)

NY = ZoneInfo("America/New_York")
ORIGIN = datetime(2024, 5, 9, 16, 15, tzinfo=NY)
PRIOR = datetime(2024, 5, 2, 16, 15, tzinfo=NY)
END = "2024-05-10T20:00:00+00:00"


def points(current=0.6, previous=0.4, *, prior=PRIOR, origin=ORIGIN):
    return [
        ((prior - timedelta(hours=3)).isoformat(), previous, 1 - previous),
        ((origin - timedelta(hours=3)).isoformat(), current, 1 - current),
    ]


def cpi_event(identity=1, group="cpi-current", **kwargs):
    return event(
        identity, group, "inflation_cpi", [cpi(identity, points=points(), **kwargs)], end=END
    )


def adapter(tmp_path, events=None):
    return HistoricalEventChanges(load(tmp_path, events or [cpi_event()]))


def measurement(result, family="cpi_headline_mom"):
    return result["measurements"][family]


def test_same_binary_question_delta_clocks_entropy_and_raw_proofs(tmp_path):
    changes = adapter(tmp_path)
    before = changes.adapter.lookup(ORIGIN)
    result = changes.lookup(ORIGIN)
    row = measurement(result)
    assert result["reference_origin"] == PRIOR.astimezone(timezone.utc).isoformat()
    assert row["status"] == "available"
    assert row["delta_yes_probability"] == pytest.approx(0.2)
    assert row["current"]["yes_probability"] == 0.6
    assert row["reference"]["yes_probability"] == 0.4
    assert row["reference"]["no_probability"] == 0.6
    assert row["identity"]["yes_token_id"] == "11"
    assert row["identity"]["no_token_id"] == "12"
    assert row["group_id"] == "cpi-current"
    assert row["current"]["binary_entropy_bits"] == pytest.approx(binary_entropy_bits(0.6))
    refs = [row[clock]["source_ref"] for clock in ("current", "reference")]
    assert all(ref["raw_quote_references"] for ref in refs)
    assert refs[0]["quote_available_at"] == result["origin"]
    assert refs[1]["quote_available_at"] == result["reference_origin"]
    assert changes.adapter.lookup(ORIGIN) == before
    json.dumps(result, allow_nan=False)


def test_reference_locks_current_group_not_nearer_old_release(tmp_path, monkeypatch):
    older = cpi_event(2, "nearer-old")
    older["end"] = "2024-05-03T20:00:00+00:00"
    older["markets"][0]["points"] = points()[:1]
    changes = adapter(tmp_path, [cpi_event(), older])
    assert changes.adapter.lookup(PRIOR)["channels"]["inflation_cpi"]["group_id"] == "nearer-old"
    original, calls = changes.adapter.lookup, []

    def counted(origin):
        calls.append(origin)
        return original(origin)

    monkeypatch.setattr(changes.adapter, "lookup", counted)
    row = measurement(changes.lookup(ORIGIN))
    assert row["group_id"] == "cpi-current"
    assert row["reference"]["source_ref"]["event_id"] == "1"
    assert calls == [ORIGIN.astimezone(timezone.utc)]


def test_reference_never_substitutes_older_threshold(tmp_path):
    old = cpi(2, points=points(current=0.8, previous=0.7))
    old["question"] = old["question"].replace("0.2%", "0.3%")
    new = cpi(1, points=points()[1:], created="2024-05-03T00:00:00+00:00")
    changes = adapter(tmp_path, [event(1, "same-group", "inflation_cpi", [new, old], end=END)])
    assert changes.adapter.lookup(PRIOR)["channels"]["inflation_cpi"]["measurements"][
        "cpi_headline_mom"
    ]["yes_probability"] == 0.7
    row = measurement(changes.lookup(ORIGIN))
    assert row["status"] == "missing" and row["delta_yes_probability"] is None
    assert row["reference"] is None
    assert row["reason_codes"] == ["locked_contract_not_created_at_reference_origin"]
    assert row["current"]["yes_probability"] == 0.6
    assert row["current"]["binary_entropy_bits"] > 0


@pytest.mark.parametrize("mode,reason", [
    ("absent", "no_available_quote_at_origin"),
    ("stale", "quote_older_than_24h_or_future"),
    ("incoherent", "incoherent_yes_no_sum"),
    ("future_bucket", "no_available_quote_at_origin"),
])
def test_bad_reference_is_missing_not_asserted_zero(tmp_path, mode, reason):
    values = points()
    if mode == "absent":
        values = values[1:]
    elif mode == "stale":
        values[0] = ((PRIOR - timedelta(hours=25)).isoformat(), 0.4, 0.6)
    elif mode == "incoherent":
        values[0] = (values[0][0], 0.4, 0.9)
    else:
        values[0] = ((PRIOR - timedelta(hours=2, minutes=59)).isoformat(), 0.4, 0.6)
    e = cpi_event()
    e["markets"][0]["points"] = values
    row = measurement(adapter(tmp_path, [e]).lookup(ORIGIN))
    assert row["status"] == "missing" and row["delta_yes_probability"] is None
    assert row["reason_codes"] == [reason]
    assert row["current"]["yes_probability"] == 0.6


def test_unchanged_probability_is_actual_zero_with_two_proofs(tmp_path):
    e = cpi_event()
    e["markets"][0]["points"] = points(0.4, 0.4)
    row = measurement(adapter(tmp_path, [e]).lookup(ORIGIN))
    assert row["status"] == "available" and row["delta_yes_probability"] == 0
    assert row["reference"]["source_ref"] and row["current"]["source_ref"]


def test_selected_farther_event_can_be_more_than_21_days_away_at_reference(tmp_path):
    e = cpi_event()
    e["end"] = "2024-05-29T20:00:00+00:00"
    changes = adapter(tmp_path, [e])
    assert changes.adapter.lookup(PRIOR)["channels"]["inflation_cpi"]["status"] == "missing"
    assert measurement(changes.lookup(ORIGIN))["status"] == "available"


@pytest.mark.parametrize("family,question,rules", [
    ("real_gdp_annualized_qoq", "Will US GDP be above 0% in Q2 2024?", GDP_RULES),
    ("payroll_jobs", "Will the US add more than 150k jobs in April 2024?", JOB_RULES),
    ("unemployment_u3", "Will US unemployment be above 4% in April 2024?", "The U-3 measure."),
])
def test_supported_non_cpi_families_use_exact_yes_question(tmp_path, family, question, rules):
    m = market(1, question, rules, points=points())
    row = measurement(
        adapter(tmp_path, [event(1, "jobs-group", "jobs_growth", [m], end=END)]).lookup(ORIGIN),
        family,
    )
    assert row["specification"]["family"] == family
    assert row["identity"]["question"] == question
    assert row["delta_yes_probability"] == pytest.approx(0.2)


def fed_event(current=(0.3, 0.5, 0.2), previous=(0.2, 0.7, 0.1)):
    e = fed_events(points=[points(c, p) for c, p in zip(current, previous)])
    e["end"] = END
    return e


def test_fed_changes_complete_same_bins_and_per_exact_bin_entropy(tmp_path):
    result = adapter(tmp_path, [fed_event()]).lookup(ORIGIN)["fed"]
    assert result["status"] == "available"
    assert result["delta_probabilities"] == pytest.approx({"cut": 0.1, "hold": -0.2, "hike": 0.1})
    assert result["current"]["normalized"] is False
    hold = next(c for c in result["components"] if c["specification"]["rate_direction"] == "hold")
    assert hold["current"]["yes_probability"] == 0.5
    assert hold["current"]["binary_entropy_bits"] == 1
    assert hold["reference"]["binary_entropy_bits"] == pytest.approx(binary_entropy_bits(0.7))
    assert hold["identity"]["market_id"] == "2"
    assert len(result["current"]["source_refs"]) == len(result["reference"]["source_refs"]) == 3


@pytest.mark.parametrize("mode,reason", [
    ("missing", "no_common_rate_bin_asof"),
    ("created_later", "unsupported_or_not_yet_created_rate_bin"),
    ("incoherent", "rate_bin_sum_outside_0_95_to_1_05"),
])
def test_fed_incomplete_reference_never_emits_aggregate_delta(tmp_path, mode, reason):
    e = fed_event()
    if mode == "missing":
        e["markets"][0]["points"] = e["markets"][0]["points"][1:]
    elif mode == "created_later":
        e["markets"][0]["created"] = "2024-05-03T00:00:00+00:00"
        e["markets"][0]["points"] = e["markets"][0]["points"][1:]
    else:
        e["markets"][0]["points"][0] = (e["markets"][0]["points"][0][0], 0.9, 0.1)
    result = adapter(tmp_path, [e]).lookup(ORIGIN)["fed"]
    assert result["status"] == "missing" and result["delta_probabilities"] is None
    assert result["reason_codes"] == [reason]
    assert len(result["components"]) == 3
    assert all(c["delta_yes_probability"] is None for c in result["components"])


def test_fed_reference_uses_common_bucket_not_individually_newer_bin(tmp_path):
    e = fed_event()
    for m in e["markets"]:
        tick, yes, no = m["points"][0]
        m["points"][0] = ((datetime.fromisoformat(tick) - timedelta(hours=1)).isoformat(), yes, no)
    e["markets"][0]["points"].insert(
        1, ((PRIOR - timedelta(hours=3)).isoformat(), 0.9, 0.1)
    )
    result = adapter(tmp_path, [e]).lookup(ORIGIN)["fed"]
    assert {c["reference"]["source_ref"]["quote_available_at"] for c in result["components"]} == {
        (PRIOR - timedelta(hours=1)).astimezone(timezone.utc).isoformat()
    }
    assert result["components"][0]["reference"]["yes_probability"] == 0.2


def test_current_unavailable_has_no_reference_or_entropy_claim(tmp_path):
    e = cpi_event()
    e["markets"][0]["points"] = points()[:1]
    result = adapter(tmp_path, [e]).lookup(ORIGIN)
    row = measurement(result)
    assert row["current"] is None and row["reference"] is None
    assert row["delta_yes_probability"] is None
    assert result["fed"]["delta_probabilities"] is None


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 0), (0.5, 1), (0.25, 0.8112781244591328)])
def test_binary_entropy_exact_endpoints_and_symmetric_uncertainty(value, expected):
    assert binary_entropy_bits(value) == pytest.approx(expected)
    assert binary_entropy_bits(value) == pytest.approx(binary_entropy_bits(1 - value))


@pytest.mark.parametrize("value", [True, None, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_binary_entropy_invalid_values_rejected(value):
    with pytest.raises(ValueError):
        binary_entropy_bits(value)


@pytest.mark.parametrize("origin,previous", [
    ("2024-03-12T16:15:00-04:00", "2024-03-05T16:15:00-05:00"),
    ("2024-11-05T16:15:00-05:00", "2024-10-29T16:15:00-04:00"),
    ("2026-06-24T16:15:00-04:00", "2026-06-16T16:15:00-04:00"),
    ("2025-01-15T16:15:00-05:00", "2025-01-07T16:15:00-05:00"),
])
def test_reference_counts_sessions_holidays_and_keeps_local_clock(origin, previous):
    assert reference_origin(datetime.fromisoformat(origin)) == datetime.fromisoformat(previous)


@pytest.mark.parametrize("origin", [
    datetime(2024, 5, 9, 16, 15),
    datetime(2024, 5, 9, 16, 14, tzinfo=NY),
    datetime(2024, 5, 11, 16, 15, tzinfo=NY),
    datetime(2026, 6, 19, 16, 15, tzinfo=NY),
])
def test_noncompleted_or_naive_clock_rejected(origin):
    with pytest.raises(ValueError):
        reference_origin(origin)


def test_policy_sealed_before_lookup_includes_source_and_semantics(tmp_path):
    changes = adapter(tmp_path)
    policy = changes.policy
    digest = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == changes.policy_sha256
    assert set(policy["source_sha256"]) == {"changes", "levels", "sessions"}
    assert policy["source_manifest_sha256"] == changes.adapter.manifest_sha256
    policy["lag_us_equity_sessions"] = 99
    assert changes.policy["lag_us_equity_sessions"] == 5
    before = copy.deepcopy(changes.policy)
    assert changes.lookup(ORIGIN)["source_policy_sha256"] == digest
    assert changes.policy == before


@pytest.mark.parametrize("field,value", [
    ("yes_token_id", "999"), ("no_token_id", "999"), ("condition_id", "different"),
    ("question", "different outcome"), ("market_id", "99"),
])
def test_changed_current_identity_cannot_be_used_for_delta(tmp_path, monkeypatch, field, value):
    changes = adapter(tmp_path)
    current = changes.adapter.lookup(ORIGIN)
    current["channels"]["inflation_cpi"]["measurements"]["cpi_headline_mom"]["source_refs"][0][
        field
    ] = value
    monkeypatch.setattr(changes.adapter, "lookup", lambda origin: current)
    with pytest.raises(ValueError):
        changes.lookup(ORIGIN)


def test_unloaded_adapter_is_rejected():
    from twiz.ingest.historical_event_features import HistoricalEventFeatures

    with pytest.raises(ValueError):
        HistoricalEventChanges(HistoricalEventFeatures())
