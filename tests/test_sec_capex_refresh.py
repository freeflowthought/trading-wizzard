import gzip
import json
from datetime import datetime, timezone

from test_cloud_capex import comparable

from twiz.cloud_capex import ISSUERS
from twiz.ingest.sec_capex import (
    COMPANYFACTS,
    SUBMISSIONS,
    cached_latest_filed,
    latest_periodic_filing,
    refresh_cloud_capex,
)
from twiz.ingest.turning_inputs import fed_expectation
from twiz.storage import ProjectPaths

NOW = datetime(2024, 7, 30, 22, tzinfo=timezone.utc)


def submissions(filed, form="10-Q"):
    return {"filings": {"recent": {"form": ["8-K", form], "filingDate": ["2024-07-29", filed],
                                   "accessionNumber": ["a", "b"],
                                   "reportDate": ["", "2024-06-30"]}}}


def fetcher(filed, calls):
    def fetch(url):
        calls.append(url)
        for ticker, (cik, _) in ISSUERS.items():
            if url == SUBMISSIONS.format(cik=cik):
                return json.dumps(submissions(filed[ticker])).encode()
            if url == COMPANYFACTS.format(cik=cik):
                return json.dumps(comparable(ticker)).encode()
        raise AssertionError(url)
    return fetch


def test_latest_periodic_filing_ignores_other_forms():
    assert latest_periodic_filing(submissions("2024-07-25"))["filed"] == "2024-07-25"
    assert latest_periodic_filing({"filings": {"recent": {}}}) is None


def test_no_newer_filing_skips_companyfacts_download(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.processed.mkdir(parents=True)
    history = {"rows": [{"date": "2024-07-26"}], "source_refs": {
        str(i): {"ticker": t, "filed": "2024-07-25"} for i, t in enumerate(ISSUERS)}}
    (paths.processed / "cloud-capex-history.json.gz").write_bytes(
        gzip.compress(json.dumps(history).encode()))
    calls = []
    receipt = refresh_cloud_capex(paths, "2024-07-30",
                                  fetch=fetcher(dict.fromkeys(ISSUERS, "2024-07-25"), calls),
                                  now=NOW)
    assert not receipt["rebuilt"]
    assert all("companyfacts" not in url for url in calls)
    assert {row["status"] for row in receipt["issuers"].values()} == {"current"}


def test_newer_filing_rebuilds_and_flags_facts_not_yet_absorbed(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.processed.mkdir(parents=True)
    paths.raw.mkdir(parents=True)
    filed = {"MSFT": "2024-07-25", "GOOGL": "2024-07-25", "META": "2024-07-31"}
    receipt = refresh_cloud_capex(paths, "2024-07-30", fetch=fetcher(filed, []), now=NOW)
    assert receipt["rebuilt"]
    assert receipt["issuers"]["META"]["status"] == "filing_pending_in_companyfacts"
    assert receipt["issuers"]["MSFT"]["status"] == "refreshed"
    rebuilt = json.loads(gzip.decompress(
        (paths.processed / "cloud-capex-history.json.gz").read_bytes()))
    assert rebuilt["rows"][-1]["date"] == "2024-07-30"
    assert rebuilt["companyfacts_latest_filed"]["MSFT"] == "2024-07-25"
    assert cached_latest_filed(rebuilt)["MSFT"] == "2024-07-25"


def test_fed_expectation_uses_nearest_meeting_and_step_size():
    snapshot = [
        {"question": "Will the Fed increase interest rates by 25 bps after the December 2026 "
                     "meeting?", "yes_probability": 0.6, "end_date": "2026-12-09T23:59:00Z"},
        {"question": "Will the Fed decrease interest rates by 50+ bps after the December 2026 "
                     "meeting?", "yes_probability": 0.1, "end_date": "2026-12-09T23:59:00Z"},
        {"question": "Will the Fed increase interest rates by 25 bps after the January 2027 "
                     "meeting?", "yes_probability": 0.9, "end_date": "2027-01-28T23:59:00Z"},
        {"question": "US x Iran ceasefire continues?", "yes_probability": 0.9},
    ]
    row = fed_expectation(snapshot, NOW)
    assert row["id"].startswith("pm-fed-live|2026-12")
    assert row["probabilities"] == {"hike": 0.6, "cut": 0.2}
    assert fed_expectation(snapshot[3:], NOW) is None
