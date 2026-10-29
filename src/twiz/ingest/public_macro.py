"""Bounded, key-free official observations; never infer missing policy rates."""

import csv
import io
import math
import ssl
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TREASURY = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ECB = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"


def _get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "trading-wizzard/0.1 personal research"})
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urlopen(request, timeout=8, context=context) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("Official feed exceeded the response size limit.")
    return data


def _number(value: Any):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def parse_treasury(data: bytes, as_of: str) -> List[Dict[str, Any]]:
    namespaces = {"m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
                  "d": "http://schemas.microsoft.com/ado/2007/08/dataservices"}
    rows = []
    properties = ET.fromstring(data).findall(".//m:properties", namespaces)
    for native, series, title in (("BC_2YEAR", "UST2Y", "美国2年期国债平价收益率"),
                                  ("BC_10YEAR", "UST10Y", "美国10年期国债平价收益率"),
                                  ("BC_30YEAR", "UST30Y", "美国30年期国债平价收益率")):
        history = {}
        for item in properties:
            stamp = item.find("d:NEW_DATE", namespaces)
            node = item.find("d:"+native, namespaces)
            if stamp is None or node is None:
                continue
            try:
                day = date.fromisoformat(str(stamp.text)[:10]).isoformat()
            except ValueError:
                continue
            value = _number(node.text)
            if day <= as_of and value is not None and value >= 0:
                history[day] = {"date": day, "value": value}
        if history:
            ordered = [history[day] for day in sorted(history)]
            rows.append({"series_id": series, "native_series_id": native,
                         "name": title, "source": "US Treasury", "unit": "percent_per_year",
                         "url": TREASURY, "method": "Treasury daily par yield curve",
                         **ordered[-1], "history": ordered})
    return rows


def parse_fred_csv(data: bytes, as_of: str) -> List[Dict[str, Any]]:
    # FRED packages series with different calendars as separate CSVs in a ZIP.
    # Read bounded members in memory; never extract arbitrary archive paths.
    if data.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            csvs = [item for item in archive.infolist() if item.filename.endswith(".csv")]
            if len(csvs) > 4 or sum(item.file_size for item in csvs) > 4_000_000:
                raise ValueError("FRED CSV archive exceeds the data budget.")
            return [row for item in csvs for row in parse_fred_csv(archive.read(item), as_of)]
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    histories = {key: {} for key in ("BAMLH0A0HYM2", "DCOILBRENTEU")}
    for row in reader:
        try:
            stamp = row.get("observation_date") or row.get("DATE")
            day = date.fromisoformat(str(stamp)).isoformat()
        except ValueError:
            continue
        if day > as_of or day < (date.fromisoformat(as_of)-timedelta(days=366)).isoformat():
            continue
        for key, history in histories.items():
            value = _number(row.get(key))
            if value is not None and value >= 0:
                history[day] = {"date": day, "value": value}
    result = []
    for key, history in histories.items():
        if history:
            ordered = [history[day] for day in sorted(history)]
            result.append({"series_id": key, "source": "FRED public CSV",
                           "name": "高收益债信用利差" if key == "BAMLH0A0HYM2" else "Brent原油现货",
                           "unit": "percent" if key == "BAMLH0A0HYM2" else "USD_per_barrel",
                           "url": f"https://fred.stlouisfed.org/series/{key}",
                           **ordered[-1], "history": ordered})
    return result


def parse_ecb(data: bytes, as_of: str) -> List[Dict[str, Any]]:
    histories = {"USDJPY": {}, "USDKRW": {}}
    for node in ET.fromstring(data).iter():
        stamp = node.attrib.get("time")
        if not stamp:
            continue
        try:
            day = date.fromisoformat(stamp).isoformat()
        except ValueError:
            continue
        if day > as_of:
            continue
        values = {child.attrib.get("currency"): _number(child.attrib.get("rate"))
                  for child in node}
        usd = values.get("USD")
        if not usd or usd <= 0:
            continue
        for series, currency in (("USDJPY", "JPY"), ("USDKRW", "KRW")):
            value = values.get(currency)
            if value and value > 0:
                histories[series][day] = {"date": day, "value": value/usd}
    result = []
    for series, history in histories.items():
        if history:
            ordered = [history[day] for day in sorted(history)]
            result.append({"series_id": series, "source": "ECB reference FX",
                           "name": "美元兑日元" if series == "USDJPY" else "美元兑韩元",
                           "unit": "local_currency_per_USD", "url": ECB,
                           "method": "same-date EUR cross rate; daily reference, not live FX",
                           **ordered[-1], "history": ordered})
    return result


def fetch_public_macro(
    report_date: str, live: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not live:
        return [], []
    as_of = date.fromisoformat(report_date)
    query = urlencode({"id": "BAMLH0A0HYM2,DCOILBRENTEU",
                       "cosd": (as_of-timedelta(days=366)).isoformat(), "coed": report_date})
    treasury_query = urlencode({"data": "daily_treasury_yield_curve",
                               "field_tdr_date_value": as_of.year})
    feeds = [
        ("US Treasury", TREASURY+"?"+treasury_query, parse_treasury),
        ("FRED public CSV", FRED_CSV+"?"+query, parse_fred_csv),
        ("ECB reference FX", ECB, parse_ecb),
    ]
    rows, warnings = [], []
    for label, url, parser in feeds:
        try:
            observations = parser(_get(url), report_date)
            if not observations:
                raise ValueError("No valid dated observations.")
            rows.extend({**row, "fetched_at": datetime.now(timezone.utc).isoformat(),
                         "vintage": "latest downloadable history, not publication-time archive"}
                        for row in observations)
        except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile) as exc:
            warnings.append(f"{label}: {type(exc).__name__}; "
                            "source unavailable, no substitute invented.")
    return rows, warnings
