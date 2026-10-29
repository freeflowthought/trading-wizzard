import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from twiz.ingest.historical_event_features import HistoricalEventFeatures, parse_question

ORIGIN = datetime(2024, 5, 2, 12, tzinfo=timezone.utc)
CLOCK = "2026-10-28T15:00:00+00:00"
CREATED = "2024-04-01T00:00:00+00:00"
OBSERVED = "2024-05-01T12:00:00+00:00"
CPI_RULES = "The month-to-month Consumer Price Index for All Urban Consumers (CPI-U)."
FED_RULES = "The upper bound of the target federal funds rate at the FOMC meeting."
JOB_RULES = "The first nonfarm payroll employment release from the BLS."
GDP_RULES = "The BEA advance estimate, annual rate of change in real GDP."


def encode(value):
    return json.dumps(value, sort_keys=True).encode()


def market(identity, question, rules, yes=0.4, **kwargs):
    return {
        "id": str(identity),
        "question": question,
        "rules": rules,
        "points": [(OBSERVED, yes, 1 - yes)],
        **kwargs,
    }


def event(identity, group, channel, markets, end="2024-05-03T00:00:00+00:00", **kwargs):
    return {
        "id": str(identity),
        "group": group,
        "channel": channel,
        "markets": markets,
        "end": end,
        **kwargs,
    }


def cpi(identity=1, yes=0.4, **kwargs):
    return market(
        identity,
        "Will US inflation be more than 0.2% from March to April 2024?",
        CPI_RULES,
        yes,
        **kwargs,
    )


def fed_events(values=(0.2, 0.7, 0.1), points=None, identity=1, offset=0):
    questions = [
        "Will the Fed decrease interest rates by 25 bps after the May 2024 meeting?",
        "No change in Fed interest rates after the May 2024 meeting?",
        "Will the Fed increase interest rates by 25+ bps after the May 2024 meeting?",
    ]
    markets = [
        market(offset + i + 1, q, FED_RULES, p) for i, (q, p) in enumerate(zip(questions, values))
    ]
    if points:
        for m, value_points in zip(markets, points):
            m["points"] = value_points
    return event(identity, "us_fomc:2024-05", "fed", markets)


def archive(tmp_path, events):
    files, rows, parents = {}, [], []

    def put(name, value):
        files[name] = encode(value)
        return hashlib.sha256(files[name]).hexdigest()

    for e in events:
        parent = {
            "id": e["id"],
            "title": e.get("title", e["group"]),
            "slug": f"event-{e['id']}",
            "createdAt": e.get("created", CREATED),
            "startDate": e.get("created", CREATED),
            "endDate": e["end"],
            "negRisk": e.get("negRisk", True),
            "markets": [],
        }
        for m in e["markets"]:
            mid = int(m["id"])
            parent["markets"].append(
                {
                    "id": m["id"],
                    "question": m["question"],
                    "description": m["rules"],
                    "createdAt": m.get("created", e.get("created", CREATED)),
                    "startDate": m.get("created", e.get("created", CREATED)),
                    "endDate": e["end"],
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": json.dumps([str(mid * 10 + 1), str(mid * 10 + 2)]),
                    "conditionId": f"condition-{m['id']}",
                }
            )
            for key in ("closedTime", "umaEndDate", "resolvedAt"):
                if key in m:
                    parent["markets"][-1][key] = m[key]
        parents.append(parent)
    body_name, receipt_name = "inputs/discovery/raw/search.body", "inputs/discovery/raw/search.json"
    body_sha = put(body_name, {"events": parents})
    url = "https://gamma-api.polymarket.com/public-search?q=fixture"
    receipt_sha = put(
        receipt_name,
        {
            "request_url": url,
            "body_file": "raw/search.body",
            "fetched_at": CLOCK,
            "body_sha256": body_sha,
        },
    )
    discovery_sha = put(
        "inputs/discovery/manifest.json",
        {"files": {"raw/search.body": body_sha, "raw/search.json": receipt_sha}},
    )
    groups = {}
    for ei, (e, parent) in enumerate(zip(events, parents)):
        group = groups.setdefault(
            e["group"], {"group_id": e["group"], "channel": e["channel"], "parent_event_ids": []}
        )
        group["parent_event_ids"].append(e["id"])
        source_file = f"sources/{e['id']}.json"
        source_sha = put(
            source_file,
            {
                "request_url": url,
                "fetched_at": CLOCK,
                "response": [parent],
                "canonical_source": {
                    "manifest_sha256": discovery_sha,
                    "body_file": body_name,
                    "body_sha256": body_sha,
                    "event_array_index": ei,
                    "receipt_file": receipt_name,
                },
            },
        )
        for m, raw_market in zip(e["markets"], parent["markets"]):
            for outcome, value_index, token in [
                ("Yes", 1, int(m["id"]) * 10 + 1),
                ("No", 2, int(m["id"]) * 10 + 2),
            ]:
                start = max(
                    datetime.fromisoformat(parent["createdAt"]),
                    datetime.fromisoformat(raw_market["createdAt"]),
                )
                row = {
                    "source_file": source_file,
                    "source_sha256": source_sha,
                    "event_id": e["id"],
                    "market_id": m["id"],
                    "question": m["question"],
                    "channel": e["channel"],
                    "release_group_id": e["group"],
                    "outcome": outcome,
                    "token_id": str(token),
                    "valid_from": start.isoformat(),
                    "source_end_date": e["end"],
                    "cutoff": e["end"],
                    "contract_metadata": {"condition_id": raw_market["conditionId"]},
                    "status": m.get("status", "downloaded"),
                    "raw_path_base": "",
                    "quarantine_reasons": m.get("quarantine", []),
                }
                closures = [
                    datetime.fromisoformat(raw_market[key])
                    for key in ("closedTime", "umaEndDate", "resolvedAt")
                    if key in raw_market
                ]
                aware = [value for value in closures if value.tzinfo is not None]
                row["cutoff"] = min([datetime.fromisoformat(e["end"])] + aware).isoformat()
                if row["status"] == "downloaded":
                    raw_file = f"raw/{m['id']}-{outcome.lower()}.body"
                    raw_points, normalized = [], []
                    for i, values in enumerate(m["points"]):
                        observed = datetime.fromisoformat(values[0])
                        value = values[value_index]
                        raw_points.append(
                            {
                                "timestamp": int(observed.timestamp()),
                                "price": value,
                                "resolution_seconds": 10800,
                            }
                        )
                        normalized.append(
                            {
                                "t": int(observed.timestamp()),
                                "p": value,
                                "resolution_seconds": 10800,
                                "observed_at": observed.isoformat(),
                                "available_at": (observed + timedelta(hours=3)).isoformat(),
                                "raw_references": [{"body_file": raw_file, "data_index": i}],
                            }
                        )
                    put(
                        raw_file,
                        {
                            "data": raw_points,
                            "pagination": {"has_more": False, "next_cursor": None},
                        },
                    )
                    put(
                        raw_file.removesuffix(".body") + ".json",
                        {
                            "request_url": f"https://data-api.polymarket.com/v2/prices-history?token_id={token}",
                            "query": {"token_id": str(token)},
                            "fetched_at": CLOCK,
                        },
                    )
                    row["history_file"] = f"history/{m['id']}-{outcome.lower()}.json"
                    put(
                        row["history_file"],
                        {
                            **row,
                            "response_files": [raw_file],
                            "history": normalized,
                            "pagination_complete": True,
                        },
                    )
                rows.append(row)
    put("index.json", {"contracts": rows, "groups": list(groups.values())})
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return tmp_path, seal(tmp_path)


def seal(path):
    files = {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file() and p != path / "manifest.json"
    }
    body = encode({"files": files, "analysis_clock_utc": CLOCK})
    (path / "manifest.json").write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def load(tmp_path, events):
    path, digest = archive(tmp_path, events)
    return HistoricalEventFeatures.from_archive(path, digest)


def test_lookup_returns_exact_typed_reference_and_original_source(tmp_path):
    s = load(tmp_path, [event(1, "us_cpi:2024-04", "inflation_cpi", [cpi()])])
    result = s.lookup(ORIGIN)
    c = result["channels"]["inflation_cpi"]
    m = c["measurements"]["cpi_headline_mom"]
    assert c["group_id"] == "us_cpi:2024-04" and m["status"] == "available"
    assert m["specification"]["relation"] == "gt"
    assert m["specification"]["lower"] == m["benchmark"] == 0.2
    assert m["benchmark_distance"] == 0 and m["yes_probability"] == 0.4
    assert m["no_probability"] == 0.6 and m["historical_pit_verified"] is False
    assert m["source_refs"][0]["yes_token_id"] == "11"
    assert m["source_refs"][0]["raw_quote_references"][0]["data_index"] == 0
    assert result["data_mode"] == "reconstructed_history"


@pytest.mark.parametrize(
    "origin,reason",
    [
        ("2024-05-01T14:59:59+00:00", "no_available_quote_at_origin"),
        ("2024-05-02T12:00:01+00:00", "quote_older_than_24h_or_future"),
    ],
)
def test_no_future_bucket_or_freshness_via_delayed_availability(tmp_path, origin, reason):
    s = load(tmp_path, [event(1, "us_cpi:2024-04", "inflation_cpi", [cpi()])])
    m = s.lookup(datetime.fromisoformat(origin))["channels"]["inflation_cpi"]["measurements"][
        "cpi_headline_mom"
    ]
    assert m["status"] == "missing" and m["reason_codes"] == [reason]
    assert m["yes_probability"] is None


def test_nearest_missing_group_does_not_fall_forward_to_available_survivor(tmp_path):
    first = event(1, "us_cpi:2024-04", "inflation_cpi", [cpi(status="not_fetched_by_budget")])
    second = event(2, "us_cpi:2024-05", "inflation_cpi", [cpi(2)], end="2024-05-10T00:00:00+00:00")
    s = load(tmp_path, [first, second])
    result = s.lookup(ORIGIN)["channels"]["inflation_cpi"]
    assert result["group_id"] == first["group"] and result["status"] == "missing"


def test_future_created_group_and_beyond_21_day_group_are_not_known_features(tmp_path):
    future = event(
        1,
        "us_cpi:2024-04",
        "inflation_cpi",
        [cpi(status="not_fetched_by_budget")],
        created="2024-05-02T13:00:00+00:00",
    )
    far = event(2, "us_cpi:2024-05", "inflation_cpi", [cpi(2)], end="2024-06-10T00:00:00+00:00")
    result = load(tmp_path, [future, far]).lookup(ORIGIN)["channels"]["inflation_cpi"]
    assert result["status"] == "missing" and "group_id" not in result


def test_fixed_benchmark_selection_does_not_choose_available_farther_threshold(tmp_path):
    missing = cpi(status="not_fetched_by_budget")
    farther = market(2, "Will US inflation be greater than 0.5%?", CPI_RULES, 0.9)
    result = load(tmp_path, [event(1, "us_cpi:2024-04", "inflation_cpi", [missing, farther])])
    m = result.lookup(ORIGIN)["channels"]["inflation_cpi"]["measurements"]["cpi_headline_mom"]
    assert m["status"] == "missing" and m["benchmark_distance"] == 0
    assert m["yes_probability"] is None


def test_fed_aggregates_same_asof_without_normalization(tmp_path):
    result = load(tmp_path, [fed_events((0.2, 0.7, 0.105))]).lookup(ORIGIN)["channels"]["fed"]
    assert result["status"] == "available" and result["normalized"] is False
    assert result["probabilities"] == {"cut": 0.2, "hold": 0.7, "hike": 0.105}
    assert result["observed_sum"] == pytest.approx(1.005)
    assert len({r["quote_available_at"] for r in result["source_refs"]}) == 1


def test_newest_common_incoherent_fed_bucket_cannot_fall_back_to_older_good_sum(tmp_path):
    points = [
        [("2024-04-30T18:00:00+00:00", a, 1 - a), (OBSERVED, b, 1 - b)]
        for a, b in zip((0.2, 0.7, 0.1), (0.3, 0.9, 0.2))
    ]
    result = load(tmp_path, [fed_events(points=points)]).lookup(ORIGIN)["channels"]["fed"]
    assert result["status"] == "incoherent" and result["probabilities"] is None
    assert result["observed_sum"] == pytest.approx(1.4)


def test_fed_aliases_are_deduplicated_without_averaging_their_prices(tmp_path):
    original = fed_events()
    alias = fed_events((0.4, 0.5, 0.1), identity=2, offset=10)
    result = load(tmp_path, [alias, original]).lookup(ORIGIN)["channels"]["fed"]
    assert result["probabilities"] == {"cut": 0.2, "hold": 0.7, "hike": 0.1}
    assert len(result["source_refs"]) == 3


def test_unsupported_other_rate_bin_withholds_vector(tmp_path):
    e = fed_events()
    e["markets"].append(market(4, "Will the FED change rates to another level?", FED_RULES, 0.0))
    result = load(tmp_path, [e]).lookup(ORIGIN)["channels"]["fed"]
    assert result["status"] == "unsupported" and result["probabilities"] is None


def test_binary_incoherence_is_not_repaired(tmp_path):
    m = cpi(points=[(OBSERVED, 0.8, 0.7)])
    result = load(tmp_path, [event(1, "us_cpi:2024-04", "inflation_cpi", [m])]).lookup(ORIGIN)
    measurement = result["channels"]["inflation_cpi"]["measurements"]["cpi_headline_mom"]
    assert measurement["status"] == "missing"
    assert measurement["reason_codes"] == ["incoherent_yes_no_sum"]


@pytest.mark.parametrize(
    "question,rules,family,relation,lower,upper",
    [
        ("Will US GDP grow 0.0% or more?", GDP_RULES, "real_gdp_annualized_qoq", "ge", 0, None),
        ("US GDP declines in Q3 2024?", GDP_RULES, "real_gdp_annualized_qoq", "lt", None, 0),
        ("Will US add less than 150k jobs?", JOB_RULES, "payroll_jobs", "lt", None, 150000),
        (
            "Will US unemployment be 4.1% or lower?",
            "The U-3 unemployment rate.",
            "unemployment_u3",
            "le",
            None,
            4.1,
        ),
        (
            "Will Core CPI YoY be more than 3.0%?",
            "Twelve-month Consumer Price Index excluding food and energy.",
            "cpi_core_yoy",
            "gt",
            3,
            None,
        ),
    ],
)
def test_exact_units_and_relations(question, rules, family, relation, lower, upper):
    spec, reason = parse_question(question, rules)
    assert reason is None and spec.family == family and spec.relation == relation
    assert spec.lower == lower and spec.upper == upper


def test_explicit_interval_endpoints_and_disclosed_distant_reference(tmp_path):
    rules = JOB_RULES + " Between 100,000 (inclusive) and 150,000 (exclusive)."
    m = market(1, "Will US add between 100k and 150k jobs?", rules, 0.25)
    result = load(tmp_path, [event(1, "us_employment_situation:2024-04", "jobs_growth", [m])])
    value = result.lookup(ORIGIN)["channels"]["jobs_growth"]["measurements"]["payroll_jobs"]
    assert value["specification"]["relation"] == "interval"
    assert value["specification"]["upper"] == 150000 and value["benchmark_distance"] == 0
    assert value["specification"]["upper_inclusive"] is False
    assert value["yes_probability"] == 0.25  # not P(jobs>150k), no invented complement


@pytest.mark.parametrize(
    "question,rules",
    [
        ("Will US add ore than 250k jobs?", JOB_RULES),
        ("Will US loose jobs?", JOB_RULES),
        ("Will US add more than 150k jobs?", JOB_RULES + " Fallback to the last available month."),
        ("Will US not add more than 150k jobs?", JOB_RULES),
        ("Will US add more than 150k jobs and GDP fall?", JOB_RULES + GDP_RULES),
        ("Will US add between 150k and 199k jobs?", JOB_RULES + " The higher range bracket wins."),
        ("Will the Fed not increase interest rates by 25 bps?", FED_RULES),
        ("Will the Fed cut interest rates by 25 bps and hike rates by 25 bps?", FED_RULES),
        ("Will US GDP grow more than 150k?", GDP_RULES),
        ("Will US add more than 0.2% jobs?", JOB_RULES),
        ("Will US GDP grow more than 1% less than 2%?", GDP_RULES),
        ("Will US lose jobs by 0.2%?", JOB_RULES),
        ("Will US GDP decline by 2%?", GDP_RULES),
        ("No change in Fed interest rates by 25 bps?", FED_RULES),
    ],
)
def test_ambiguous_typo_negation_and_fallback_are_not_silently_reinterpreted(question, rules):
    spec, reason = parse_question(question, rules)
    assert spec is None and reason


def test_naive_origin_rejected_and_outputs_do_not_mutate_store(tmp_path):
    store = load(tmp_path, [fed_events()])
    with pytest.raises(ValueError, match="aware"):
        store.lookup(datetime(2024, 5, 2))
    first = store.lookup(ORIGIN)
    first["channels"]["fed"]["probabilities"]["cut"] = 0.99
    assert store.lookup(ORIGIN)["channels"]["fed"]["probabilities"]["cut"] == 0.2


def test_external_hash_and_raw_tampering_rejected(tmp_path):
    path, digest = archive(tmp_path, [fed_events()])
    with pytest.raises(ValueError, match="expected hash"):
        HistoricalEventFeatures.from_archive(path, "0" * 64)
    (path / "raw/1-yes.body").write_bytes(b"{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        HistoricalEventFeatures.from_archive(path, digest)


@pytest.mark.parametrize("mutation", ["token", "market", "raw_query"])
def test_resealed_misbindings_still_fail_identity_checks(tmp_path, mutation):
    path, _ = archive(tmp_path, [fed_events()])
    if mutation == "raw_query":
        name = path / "raw/1-yes.json"
        value = json.loads(name.read_text())
        value["request_url"] = value["request_url"].replace("token_id=11", "token_id=999")
    else:
        name = path / "index.json"
        value = json.loads(name.read_text())
        value["contracts"][0]["token_id" if mutation == "token" else "market_id"] = "999"
    name.write_bytes(encode(value))
    with pytest.raises(ValueError, match="binding|membership|reference|receipt"):
        HistoricalEventFeatures.from_archive(path, seal(path))


def test_raw_terminal_caps_later_positive_resolution_quote(tmp_path):
    path, _ = archive(tmp_path, [fed_events()])
    name = path / "raw/1-yes.body"
    body = json.loads(name.read_text())
    body["data"].append(
        {
            "timestamp": int(datetime(2024, 5, 1, 10, tzinfo=timezone.utc).timestamp()),
            "price": 1,
            "resolution_seconds": 0,
        }
    )
    name.write_bytes(encode(body))
    with pytest.raises(ValueError, match="quote/time boundary"):
        HistoricalEventFeatures.from_archive(path, seal(path))


def test_fed_uses_latest_common_bucket_not_mixed_times(tmp_path):
    old = "2024-05-01T00:00:00+00:00"
    points = [
        [(old, p, 1 - p), (OBSERVED, 0.9, 0.1)] if i < 2 else [(old, p, 1 - p)]
        for i, p in enumerate((0.2, 0.7, 0.1))
    ]
    result = load(tmp_path, [fed_events(points=points)]).lookup(
        datetime(2024, 5, 2, tzinfo=timezone.utc)
    )["channels"]["fed"]
    assert result["status"] == "available"
    assert result["probabilities"] == {"cut": 0.2, "hold": 0.7, "hike": 0.1}
    assert {r["quote_available_at"] for r in result["source_refs"]} == {"2024-05-01T03:00:00+00:00"}


def test_future_created_reference_does_not_displace_known_contract(tmp_path):
    future = cpi(
        created="2024-05-02T13:00:00+00:00", points=[("2024-05-02T14:00:00+00:00", 0.8, 0.2)]
    )
    current = market(2, "Will US inflation be more than 0.3%?", CPI_RULES, 0.4)
    value = load(tmp_path, [event(1, "us_cpi:2024-04", "inflation_cpi", [future, current])])
    m = value.lookup(ORIGIN)["channels"]["inflation_cpi"]["measurements"]["cpi_headline_mom"]
    assert m["status"] == "available" and m["benchmark_distance"] == 0.1
    assert m["source_refs"][0]["market_id"] == "2"


def test_core_and_headline_measurements_are_never_averaged(tmp_path):
    headline = cpi(yes=0.3)
    core = market(
        2,
        "Will Core CPI MoM be more than 0.2%?",
        "One-month Consumer Price Index excluding food and energy.",
        0.8,
    )
    s = load(tmp_path, [event(1, "us_cpi:2024-04", "inflation_cpi", [headline, core])])
    ms = s.lookup(ORIGIN)["channels"]["inflation_cpi"]["measurements"]
    assert ms["cpi_headline_mom"]["yes_probability"] == 0.3
    assert ms["cpi_core_mom"]["yes_probability"] == 0.8
    assert ms["cpi_headline_yoy"]["status"] == "missing"


def test_missing_earliest_alias_not_replaced_by_available_later_alias(tmp_path):
    first = event(1, "us_cpi:2024-04", "inflation_cpi", [cpi(status="not_fetched_by_budget")])
    second = event(2, "us_cpi:2024-04", "inflation_cpi", [cpi(2)])
    m = load(tmp_path, [first, second]).lookup(ORIGIN)["channels"]["inflation_cpi"]["measurements"][
        "cpi_headline_mom"
    ]
    assert m["status"] == "missing" and m["yes_probability"] is None


def test_naive_source_start_is_not_treated_as_known_utc(tmp_path):
    e = event(1, "us_cpi:2024-04", "inflation_cpi", [cpi()], created="2024-04-01T00:00:00")
    path, digest = archive(tmp_path, [e])
    with pytest.raises(ValueError, match="creation/start"):
        HistoricalEventFeatures.from_archive(path, digest)


def test_group_at_declared_end_is_expired_and_never_used(tmp_path):
    s = load(tmp_path, [event(1, "us_cpi:2024-04", "inflation_cpi", [cpi()])])
    r = s.lookup(datetime(2024, 5, 3, tzinfo=timezone.utc))["channels"]["inflation_cpi"]
    assert r["status"] == "missing" and "group_id" not in r


def test_quarantined_rate_bin_withholds_partition(tmp_path):
    e = fed_events()
    e["markets"][0]["quarantine"] = ["calendar_conflict"]
    r = load(tmp_path, [e]).lookup(ORIGIN)["channels"]["fed"]
    assert r["status"] == "missing" and r["reason_codes"] == ["quarantined_rate_bins"]


def test_omitted_rare_rate_bin_cannot_make_incomplete_partition_look_coherent(tmp_path):
    path, _ = archive(tmp_path, [fed_events()])
    name = path / "index.json"
    value = json.loads(name.read_text())
    value["contracts"] = [row for row in value["contracts"] if row["market_id"] != "3"]
    name.write_bytes(encode(value))
    with pytest.raises(ValueError, match="Incomplete event market/outcome"):
        HistoricalEventFeatures.from_archive(path, seal(path))


def test_omitted_owned_response_list_cannot_bypass_raw_terminal_check(tmp_path):
    path, _ = archive(tmp_path, [fed_events()])
    name = path / "history/1-yes.json"
    value = json.loads(name.read_text())
    value["response_files"] = []
    name.write_bytes(encode(value))
    with pytest.raises(ValueError, match="response-file inventory"):
        HistoricalEventFeatures.from_archive(path, seal(path))


def test_overlapping_rate_bins_with_coherent_total_are_not_aggregated(tmp_path):
    e = fed_events((0.1, 0.7, 0.1))
    e["markets"][0]["question"] = "Will the Fed cut interest rates by 25+ bps?"
    e["markets"].append(market(4, "Will the Fed cut interest rates by 50 bps?", FED_RULES, 0.1))
    result = load(tmp_path, [e]).lookup(ORIGIN)["channels"]["fed"]
    assert result["status"] == "unsupported" and result["probabilities"] is None
    assert result["reason_codes"] == ["overlapping_rate_bins"]


@pytest.mark.parametrize("key", ["closedTime", "umaEndDate", "resolvedAt"])
def test_present_naive_closure_cannot_extend_market_window(tmp_path, key):
    e = event(1, "us_cpi:2024-04", "inflation_cpi", [cpi(**{key: "2024-05-02T11:00:00"})])
    path, digest = archive(tmp_path, [e])
    with pytest.raises(ValueError, match="closure timestamp"):
        HistoricalEventFeatures.from_archive(path, digest)


def test_closed_contract_quote_not_reused_after_closure(tmp_path):
    e = event(1, "us_cpi:2024-04", "inflation_cpi", [cpi(closedTime="2024-05-02T11:00:00+00:00")])
    m = load(tmp_path, [e]).lookup(ORIGIN)["channels"]["inflation_cpi"]["measurements"][
        "cpi_headline_mom"
    ]
    assert m["status"] == "missing" and m["reason_codes"] == ["market_cutoff_has_passed"]


def test_rate_direction_sum_cannot_exceed_one_even_with_coherent_total(tmp_path):
    e = fed_events((0.60, 0.0, 0.01))
    e["markets"].append(market(4, "Will the Fed cut interest rates by 50+ bps?", FED_RULES, 0.44))
    r = load(tmp_path, [e]).lookup(ORIGIN)["channels"]["fed"]
    assert r["status"] == "incoherent" and r["probabilities"] is None
    assert r["reason_codes"] == ["rate_direction_probability_outside_0_to_1"]
