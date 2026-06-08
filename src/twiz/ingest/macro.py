import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch_fred_observations(
    series: Iterable[Dict[str, str]],
    live: bool = False,
    limit: int = 1,
) -> Tuple[List[Dict[str, object]], List[str]]:
    if not live:
        return [], ["FRED live ingest skipped. Pass --live and set FRED_API_KEY."]
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return [], ["FRED_API_KEY is not set; macro series skipped."]

    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    for item in series:
        series_id = str(item.get("id", ""))
        if not series_id:
            continue
        query = urlencode(
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": str(limit),
            }
        )
        url = f"https://api.stlouisfed.org/fred/series/observations?{query}"
        try:
            request = Request(url, headers={"User-Agent": "trading-wizzard/0.1 personal research"})
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, ValueError) as exc:
            warnings.append(f"FRED fetch failed for {series_id}: {exc}")
            continue
        observations = payload.get("observations", [])
        value = observations[0] if observations else {}
        rows.append(
            {
                "source": "FRED",
                "series_id": series_id,
                "name": item.get("name", series_id),
                "date": value.get("date"),
                "value": value.get("value"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows, warnings
