"""Live inputs for the market-implied turning layer, cached locally between runs.

Yahoo supplies ^VIX, ^VIX3M, HYG and IEF daily closes; FRED's key-free CSV supplies
DGS2, DFEDTARU and ICSA. Polymarket Fed-decision quotes from the daily snapshot are
reduced to an expected next-meeting change and appended to a dated history. Any
failed source keeps the cached rows and is reported; missing values stay neutral.
"""

import csv
import gzip
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, urlopen

from ..storage import write_json

TICKERS = ("^VIX", "^VIX3M", "HYG", "IEF")
FRED_SERIES = ("DGS2", "DFEDTARU", "ICSA")
CACHE = "turning-inputs.json.gz"
PM_HISTORY = "turning-pm-history.json"
FED_QUESTION = re.compile(
    r"\bfed\b.*\b(increase|raise|hike|decrease|cut|lower)\b.*?(\d+)\+?\s*bps?\b.*"
    r"\bafter the ([a-z]+ \d{4}) meeting",
    re.IGNORECASE,
)


def _load(path):
    if not path.is_file():
        return {"market": {}, "fred": {}}
    return json.loads(gzip.decompress(path.read_bytes()))


def _fetch_prices(start, end):
    import yfinance as yf

    frame = yf.download(list(TICKERS), start=start, end=end, progress=False,
                        auto_adjust=False, group_by="ticker")
    out = {}
    for ticker in TICKERS:
        table = frame[ticker].dropna(subset=["Close"])
        out[ticker] = [{"date": index.date().isoformat(), "close": float(row.Close),
                        "source": "yfinance"} for index, row in table.iterrows()]
    return out


def _fetch_fred(series, start):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
    request = Request(url, headers={"User-Agent": "trading-wizzard/turning-inputs"})
    with urlopen(request, timeout=15) as response:
        body = response.read(4_000_001).decode("utf-8-sig")
    rows = []
    for row in csv.DictReader(io.StringIO(body)):
        day = row.get("observation_date") or row.get("DATE")
        try:
            rows.append([date.fromisoformat(day).isoformat(), float(row[series])])
        except (TypeError, ValueError, KeyError):
            continue
    if not rows:
        raise ValueError(series + ": no observations")
    return rows


def _merge(old, new, key):
    merged = {item[key] if isinstance(item, dict) else item[0]: item for item in old}
    merged.update({item[key] if isinstance(item, dict) else item[0]: item for item in new})
    return [merged[day] for day in sorted(merged)]


def fed_expectation(snapshot, observed_at):
    """Expected next-meeting change (pp) from 'Fed increase/decrease N bps' quotes."""
    meetings = {}
    for row in snapshot or []:
        match = FED_QUESTION.search(str(row.get("question") or ""))
        probability = row.get("yes_probability")
        if not match or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            continue
        side = "hike" if match.group(1).lower() in ("increase", "raise", "hike") else "cut"
        meeting = datetime.strptime(match.group(3).title(), "%B %Y").date()
        state = meetings.setdefault(meeting, {"hike": 0.0, "cut": 0.0, "questions": [],
                                              "end_date": row.get("end_date")})
        state[side] += probability * int(match.group(2)) / 25
        state["questions"].append(row.get("question"))
    if not meetings:
        return None
    meeting = min(meetings)
    state = meetings[meeting]
    return {
        "id": f"pm-fed-live|{meeting:%Y-%m}|{observed_at.date().isoformat()}",
        "available_at": observed_at.isoformat(),
        "expires_at": state["end_date"],
        # Expected 25bp steps: "50+ bps" quotes count twice, so values can exceed one.
        "probabilities": {"hike": state["hike"], "cut": state["cut"]},
        "questions": state["questions"],
        "source": "Polymarket daily snapshot",
        "partial_ladder": True,
    }


def refresh_turning_inputs(paths, report_date, live=False, now=None):
    now = now or datetime.now(timezone.utc)
    cache_path = paths.processed / CACHE
    cache = _load(cache_path)
    receipt = {"checked_at": now.isoformat(), "live": live, "sources": {}}
    if live:
        start = (date.fromisoformat(report_date) - timedelta(days=3 * 366)).isoformat()
        end = (date.fromisoformat(report_date) + timedelta(days=1)).isoformat()
        try:
            prices = _fetch_prices(start, end)
            for ticker, rows in prices.items():
                cache["market"][ticker] = _merge(cache["market"].get(ticker, []), rows, "date")
            receipt["sources"]["yahoo"] = "refreshed"
        except Exception as error:
            receipt["sources"]["yahoo"] = "failed_cache_kept:" + type(error).__name__
        for series in FRED_SERIES:
            try:
                rows = _fetch_fred(series, start)
                cache["fred"][series] = _merge(cache["fred"].get(series, []), rows, 0)
                receipt["sources"][series] = "refreshed"
            except Exception as error:
                receipt["sources"][series] = "failed_cache_kept:" + type(error).__name__
        cache["fetched_at"] = now.isoformat()
        cache_path.write_bytes(gzip.compress(json.dumps(cache).encode()))
    pm_path = paths.processed / PM_HISTORY
    pm_rows = json.loads(pm_path.read_text()) if pm_path.is_file() else []
    snapshot_path = paths.processed / f"prediction-markets-{report_date}.json"
    if live and snapshot_path.is_file():
        row = fed_expectation(json.loads(snapshot_path.read_text()), now)
        if row and row["id"] not in {r.get("id") for r in pm_rows}:
            pm_rows.append(row)
            write_json(pm_path, pm_rows)
        receipt["sources"]["polymarket_fed"] = "appended" if row else "no_fed_decision_quote"
    for ticker in TICKERS:
        rows = cache["market"].get(ticker, [])
        receipt["sources"].setdefault(ticker + "_last", rows[-1]["date"] if rows else None)
    cutoff = date.fromisoformat(report_date).isoformat()
    market = {t: [r for r in rows if r["date"] <= cutoff] for t, rows in cache["market"].items()}
    fred = {s: [tuple(r) for r in rows if r[0] <= cutoff] for s, rows in cache["fred"].items()}
    return {"market": market, "fred": fred, "prediction_rows": pm_rows, "receipt": receipt}
