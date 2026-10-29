"""Keep the cloud cash-PP&E history current with each issuer's newest SEC filing.

The SEC submissions index is checked on every live report. Company facts are only
downloaded again when an issuer has filed a 10-Q/10-K newer than the cached facts,
then the dated timeline is rebuilt with the existing exact-period rules. A filing
that the XBRL facts API has not yet absorbed is reported as pending, never filled
with the previous quarter.
"""

import gzip
import hashlib
import json
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, urlopen

from ..cloud_capex import FORMS, ISSUERS, build_cloud_capex_timeline
from ..sessions import is_us_equity_session
from ..storage import write_json

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MAX_BYTES = 64 * 1024 * 1024
HISTORY = "cloud-capex-history.json.gz"


def _identity():
    return (os.getenv("EDGAR_IDENTITY") or os.getenv("SEC_USER_AGENT")
            or "trading-wizzard/0.1 contact=unknown")


def _get(url, timeout=20):
    request = Request(url, headers={"User-Agent": _identity(), "Accept-Encoding": "identity"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("SEC response exceeds bounded receipt size")
    return body


def latest_periodic_filing(submissions):
    """Newest 10-Q/10-K (or amendment) filing date and accession in the recent index."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    rows = zip(recent.get("form", []), recent.get("filingDate", []),
               recent.get("accessionNumber", []), recent.get("reportDate", []))
    filings = [(filed, form, accession, report) for form, filed, accession, report in rows
               if form in FORMS and filed]
    if not filings:
        return None
    filed, form, accession, report = max(filings)
    return {"filed": filed, "form": form, "accn": accession, "report_date": report}


def cached_latest_filed(history):
    """Latest filed date per issuer among facts the cached timeline already references."""
    latest = {}
    for ref in (history or {}).get("source_refs", {}).values():
        ticker, filed = ref.get("ticker"), ref.get("filed")
        if ticker and filed and filed > latest.get(ticker, ""):
            latest[ticker] = filed
    # Facts already downloaded but not yet available at the last report date count too.
    for ticker, filed in ((history or {}).get("companyfacts_latest_filed") or {}).items():
        if filed and filed > latest.get(ticker, ""):
            latest[ticker] = filed
    return latest


def _facts_latest_filed(company):
    latest = ""
    for concept in ((company.get("facts") or {}).get("us-gaap") or {}).values():
        for fact in ((concept or {}).get("units") or {}).get("USD", []) or []:
            if isinstance(fact, dict) and fact.get("form") in FORMS:
                latest = max(latest, str(fact.get("filed") or ""))
    return latest


def _sessions(first, last):
    day, out = date.fromisoformat(first), []
    while day <= last:
        if is_us_equity_session(day):
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def refresh_cloud_capex(paths, report_date, fetch=_get, now=None):
    """Return a freshness receipt; rewrites the history only when a newer filing exists."""
    now = now or datetime.now(timezone.utc)
    history_path = paths.processed / HISTORY
    history = (json.loads(gzip.decompress(history_path.read_bytes()))
               if history_path.is_file() else {})
    cached = cached_latest_filed(history)
    receipt = {"checked_at": now.isoformat(), "issuers": {}, "rebuilt": False,
               "history_path": str(history_path)}
    stale = {}
    for ticker, (cik, _) in ISSUERS.items():
        try:
            filing = latest_periodic_filing(json.loads(fetch(SUBMISSIONS.format(cik=cik))))
        except Exception as error:  # One issuer failing keeps the others current.
            receipt["issuers"][ticker] = {"status": "submissions_unavailable",
                                          "reason": type(error).__name__}
            continue
        known = cached.get(ticker, "")
        state = {"latest_sec_filing": filing, "cached_facts_filed": known or None}
        if filing and filing["filed"] > known:
            stale[ticker] = filing
            state["status"] = "newer_filing_detected"
        else:
            state["status"] = "current"
        receipt["issuers"][ticker] = state
    if not stale:
        return receipt
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    folder = paths.raw / "sec-capex" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    companies, hashes = {}, {}
    for ticker, (cik, _) in ISSUERS.items():
        try:
            body = fetch(COMPANYFACTS.format(cik=cik))
        except Exception as error:
            receipt["issuers"][ticker].update(status="companyfacts_unavailable",
                                              reason=type(error).__name__)
            continue
        (folder / f"{ticker.lower()}-companyfacts.json").write_bytes(body)
        companies[ticker] = json.loads(body)
        hashes[ticker] = hashlib.sha256(body).hexdigest()
        if ticker in stale:
            absorbed = _facts_latest_filed(companies[ticker]) >= stale[ticker]["filed"]
            # XBRL facts can trail the filing index by hours; never substitute the old quarter.
            receipt["issuers"][ticker]["status"] = (
                "refreshed" if absorbed else "filing_pending_in_companyfacts")
    if len(companies) < 2:
        receipt["status"] = "insufficient_companyfacts_kept_cache"
        return receipt
    first = (history.get("rows") or [{}])[0].get("date") or "2016-01-04"
    sessions = [day for day in _sessions(first, date.fromisoformat(report_date))]
    rebuilt = build_cloud_capex_timeline(companies, sessions, hashes)
    rebuilt.update(source_capture_date=now.date().isoformat(), source_run=str(folder),
                   freshness_receipt=receipt,
                   companyfacts_latest_filed={ticker: _facts_latest_filed(company)
                                              for ticker, company in companies.items()})
    history_path.write_bytes(gzip.compress(json.dumps(rebuilt).encode()))
    receipt["rebuilt"] = True
    write_json(folder / "receipt.json", receipt)
    return receipt
