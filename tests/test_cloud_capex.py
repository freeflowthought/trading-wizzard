import copy
import json
from datetime import date, timedelta

import pytest

from twiz import cloud_capex
from twiz.cloud_capex import CAPEX, ISSUERS, REVENUE, build_cloud_capex_timeline
from twiz.sessions import is_us_equity_session

HASH = "a" * 64


def company(ticker="META"):
    return {"cik": ISSUERS[ticker][0], "entityName": ticker, "facts": {"us-gaap": {}}}


def add(data, tag, start, end, value, filed="2024-07-25", **extra):
    facts = data["facts"]["us-gaap"].setdefault(tag, {"units": {"USD": []}})["units"]["USD"]
    facts.append({"start": start, "end": end, "val": value, "filed": filed,
                  "accn": "0000000000-24-000001", "form": "10-Q", **extra})
    return facts[-1]


def comparable(ticker="META", capex=(100, 150, 300), revenue=(500, 600, 750)):
    data = company(ticker)
    for year, cash, sales in zip((2022, 2023, 2024), capex, revenue):
        start = f"{year - 1}-07-01" if ticker == "MSFT" else f"{year}-01-01"
        for tag, value in ((CAPEX, cash), (REVENUE[0], sales)):
            add(data, tag, start, f"{year}-06-30", value)
    return data


def build(data, days=None):
    return build_cloud_capex_timeline(
        data, days or ["2024-07-26"], dict.fromkeys(data, HASH))


def issuer(result, ticker="META", index=-1):
    return result["rows"][index]["issuers"][ticker]


def test_exact_ytd_growth_intensity_and_same_duration_acceleration():
    result = build({"META": comparable()})
    row = issuer(result)
    assert row["status"] == "available"
    assert row["cash_capex_yoy_pct"] == 100
    assert row["revenue_yoy_pct"] == 25
    assert row["capex_revenue_intensity_pct"] == 40
    assert row["cash_capex_yoy_acceleration_pp"] == 50
    assert row["revenue_yoy_acceleration_pp"] == pytest.approx(5)
    assert row["period"]["cumulative_months"] == 6
    assert result["historical_pit_verified"] is False
    assert "same_YTD_duration" in result["method"]["acceleration"]
    assert result["rows"][0]["available_at"] == "2024-07-26T16:15:00-04:00"


def test_msft_june_year_end_is_not_calendar_ytd_or_filing_fy():
    data = comparable("MSFT")
    for facts in data["facts"]["us-gaap"].values():
        for fact in facts["units"]["USD"]:
            fact.update(fy=2030, fp="Q1")
    row = issuer(build({"MSFT": data}), "MSFT")
    assert row["status"] == "available"
    assert row["period"] == {"start": "2023-07-01", "end": "2024-06-30",
                             "cumulative_months": 12}


def test_filing_date_is_not_available_until_next_actual_session():
    data = comparable()
    for concept in data["facts"]["us-gaap"].values():
        for fact in concept["units"]["USD"]:
            fact["filed"] = "2024-07-03"
    result = build({"META": data}, ["2024-07-03", "2024-07-05"])
    assert issuer(result, index=0)["status"] == "unknown"
    assert issuer(result)["status"] == "available"
    assert all(r["available_at_proxy"] == "2024-07-05T16:15:00-04:00"
               for r in result["source_refs"].values())


def test_future_fact_value_is_never_read_even_with_new_alias_or_period():
    class FutureFact(dict):
        def get(self, key, default=None):
            if key == "val":
                raise AssertionError("future numeric read")
            return super().get(key, default)

    data = comparable()
    concept = data["facts"]["us-gaap"][CAPEX]["units"]["USD"]
    concept.append(FutureFact(start="2024-01-01", end="2024-09-30", filed="2024-10-24",
                              form="10-Q", val=999, accn="later"))
    future = FutureFact(start="2024-01-01", end="2024-06-30", filed="2024-10-24",
                        form="10-Q", val=999, accn="later")
    data["facts"]["us-gaap"][REVENUE[1]] = {"units": {"USD": [future]}}
    assert issuer(build({"META": data}))["cash_capex_yoy_pct"] == 100


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), -1, "300"])
def test_invalid_latest_fact_does_not_fall_back_to_older_valid_value(bad):
    data = comparable()
    add(data, CAPEX, "2024-01-01", "2024-06-30", bad, filed="2024-07-26")
    result = build({"META": data}, ["2024-07-26", "2024-07-29"])
    assert issuer(result, index=0)["status"] == "available"
    row = issuer(result)
    assert row["status"] == "unknown"
    assert row["current"]["capex"]["reason"] == "invalid_latest_fact"
    assert row["cash_capex_yoy_pct"] is None


def test_conflicting_same_day_versions_and_aliases_remain_unknown():
    for tag in (CAPEX, REVENUE[1]):
        data = comparable()
        add(data, tag, "2024-01-01", "2024-06-30", 999)
        row = issuer(build({"META": data}))
        assert row["status"] == "unknown"
        kind = "capex" if tag == CAPEX else "revenue"
        assert row["current"][kind]["reason"] == "conflicting_latest_versions_or_aliases"


def test_legacy_tag_transition_is_chosen_by_known_versions_not_tag_preference():
    data = comparable()
    current = data["facts"]["us-gaap"].pop(REVENUE[0])
    data["facts"]["us-gaap"][REVENUE[2]] = current
    add(data, REVENUE[0], "2024-01-01", "2024-06-30", 900, filed="2024-07-26")
    result = build({"META": data}, ["2024-07-26", "2024-07-29"])
    assert issuer(result, index=0)["current"]["revenue"]["value"] == 750
    assert issuer(result)["current"]["revenue"]["value"] == 900
    refs = issuer(result)["current"]["revenue"]["refs"]
    assert {result["source_refs"][ref]["tag"] for ref in refs} == {REVENUE[0]}


def test_matching_aliases_are_accepted_but_invalid_new_alias_blocks_old():
    data = comparable()
    add(data, REVENUE[1], "2024-01-01", "2024-06-30", 750)
    assert len(issuer(build({"META": data}))["current"]["revenue"]["refs"]) == 2
    add(data, REVENUE[2], "2024-01-01", "2024-06-30", None, filed="2024-07-26")
    row = issuer(build({"META": data}, ["2024-07-29"]))
    assert row["current"]["revenue"]["reason"] == "invalid_latest_fact"


def test_newest_cash_period_missing_revenue_does_not_reuse_older_period():
    data = comparable()
    add(data, CAPEX, "2024-01-01", "2024-09-30", 350, filed="2024-10-24")
    row = issuer(build({"META": data}, ["2024-10-25"]))
    assert row["status"] == "unknown" and row["period"]["end"] == "2024-09-30"
    assert row["current"]["revenue"]["reason"] == "no_available_exact_period_fact"


def test_newer_revenue_period_missing_cash_does_not_present_older_quarter_as_current():
    data = comparable()
    add(data, REVENUE[0], "2024-01-01", "2024-09-30", 900, filed="2024-10-24")
    row = issuer(build({"META": data}, ["2024-10-25"]))
    assert row["status"] == "unknown" and row["period"]["end"] == "2024-09-30"
    assert row["current"]["capex"]["reason"] == "no_available_exact_period_fact"
    assert row["cash_capex_yoy_pct"] is None


def test_prior_duration_must_match_exact_start_and_end_no_quarter_or_leap_tolerance():
    data = comparable()
    for tag in (CAPEX, REVENUE[0]):
        facts = data["facts"]["us-gaap"][tag]["units"]["USD"]
        facts[1]["end"] = "2023-03-31"  # Quarter, not the comparable six-month YTD.
    row = issuer(build({"META": data}))
    assert row["status"] == "partial" and row["cash_capex_yoy_pct"] is None
    assert row["capex_revenue_intensity_pct"] == 40
    data = comparable()
    data["facts"]["us-gaap"][REVENUE[0]]["units"]["USD"][1]["end"] = "2023-06-29"
    assert issuer(build({"META": data}))["status"] == "partial"


def test_zero_capex_is_not_negative_or_artificial_growth_from_zero_base():
    row = issuer(build({"META": comparable(capex=(100, 150, 0))}))
    assert row["status"] == "available" and row["cash_capex_yoy_pct"] == -100
    assert row["capex_revenue_intensity_pct"] == 0
    row = issuer(build({"META": comparable(capex=(100, 0, 300))}))
    assert row["status"] == "partial" and row["cash_capex_yoy_pct"] is None
    row = issuer(build({"META": comparable(revenue=(500, 600, 0))}))
    assert row["status"] == "unknown" and row["capex_revenue_intensity_pct"] is None


def test_cross_issuer_medians_require_two_comparable_issuers_and_never_sum_dollars():
    data = {"MSFT": comparable("MSFT"),
            "META": comparable("META", capex=(100, 200, 300), revenue=(500, 500, 750))}
    result = build(data)
    aggregate = result["rows"][0]["aggregate"]
    assert aggregate["status"] == "available" and aggregate["issuer_count"] == 2
    assert aggregate["median_cash_capex_yoy_pct"] == 75
    assert aggregate["median_revenue_yoy_pct"] == 37.5
    assert aggregate["median_capex_revenue_intensity_pct"] == 40
    assert aggregate["excluded_issuers"]["GOOGL"] == "missing_companyfacts"
    one = build({"MSFT": data["MSFT"]})["rows"][0]["aggregate"]
    assert one["status"] == "insufficient_issuers"
    assert one["median_cash_capex_yoy_pct"] is None


def test_amazon_productive_assets_are_explicitly_excluded_not_a_ppe_substitute():
    data = {"cik": 1018724, "facts": {"us-gaap": {}}}
    add(data, "PaymentsToAcquireProductiveAssets", "2024-01-01", "2024-06-30", 999999)
    result = build({"AMZN": data})
    assert "AMZN" in result["excluded_issuers"]
    assert "AMZN" not in result["rows"][0]["issuers"]
    assert result["rows"][0]["aggregate"]["issuer_count"] == 0


def test_source_pointers_hash_accession_and_availability_survive_without_input_mutation():
    data = {"META": comparable()}
    before = copy.deepcopy(data)
    result = build(data, ["2024-07-26", "2024-07-29"])
    assert data == before
    for ref in result["source_refs"].values():
        assert ref["source_sha256"] == HASH and ref["accn"]
        assert ref["json_pointer"].startswith("/facts/us-gaap/")
        assert ref["available_at_proxy"] == "2024-07-26T16:15:00-04:00"
    assert result["rows"][0]["source_refs"] == result["source_refs"]
    result["rows"][0]["issuers"]["META"]["current"]["capex"]["value"] = -999
    assert issuer(result)["current"]["capex"]["value"] == 300
    assert data == before


@pytest.mark.parametrize("days", [["2024-07-27"], ["2024-07-26", "2024-07-26"],
                                  ["2024-07-29", "2024-07-26"], ["20240726"], ["bad"]])
def test_session_contract_rejects_nontrading_duplicates_unsorted_or_noniso(days):
    with pytest.raises(ValueError):
        build({"META": comparable()}, days)


def test_provenance_and_cik_identity_guards():
    with pytest.raises(ValueError, match="SHA256"):
        build_cloud_capex_timeline({"META": comparable()}, ["2024-07-26"], {})
    data = comparable()
    data["cik"] = ISSUERS["MSFT"][0]
    assert issuer(build({"META": data}))["reason"] == "issuer_cik_mismatch_or_missing"


def test_snapshot_computation_is_cached_between_release_events(monkeypatch):
    snapshots = []
    original = cloud_capex._snapshot

    def counted(*args):
        snapshots.append(args[1])
        return original(*args)

    monkeypatch.setattr(cloud_capex, "_snapshot", counted)
    start = date(2024, 7, 1)
    days = [(start + timedelta(days=i)).isoformat() for i in range(120)
            if is_us_equity_session(start + timedelta(days=i))]
    result = build({"META": comparable()}, days)
    assert len(result["rows"]) > 80
    assert snapshots == ["META", "META"]  # Initial unavailable state and one release date.


def test_empty_timeline_does_not_read_any_fact_values():
    assert build_cloud_capex_timeline({}, [], {})["rows"] == []


def test_nonfinite_derived_ratios_and_median_cannot_escape_as_json_infinity():
    data = comparable(capex=(1, 1e-308, 1e308), revenue=(1, 1, 1))
    result = build({"META": data})
    assert issuer(result)["cash_capex_yoy_pct"] is None
    json.dumps(result, allow_nan=False)
    # Large finite inputs must not overflow while the median adds two constituents.
    full = {name: {"status": "available", "reason": None, **dict.fromkeys(
        cloud_capex.METRICS, 1e308)} for name in ("MSFT", "META")}
    assert cloud_capex._aggregate(full)["median_cash_capex_yoy_pct"] == 1e308


def test_malformed_fact_containers_and_missing_accession_are_unknown_not_substituted():
    data = company()
    data["facts"] = None
    assert issuer(build({"META": data}))["status"] == "unknown"
    data = comparable()
    add(data, CAPEX, "2024-01-01", "2024-06-30", 999, filed="2024-07-26", accn=None)
    assert issuer(build({"META": data}, ["2024-07-29"]))["status"] == "unknown"
