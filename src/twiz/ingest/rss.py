import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ..config import WatchedAsset
from ..models import RawItem
from ..text import extract_tickers, normalize_text


def _user_agent_for_url(url: str) -> str:
    if "sec.gov" in url:
        return (
            os.getenv("EDGAR_IDENTITY")
            or os.getenv("SEC_USER_AGENT")
            or "trading-wizzard/0.1 contact=unknown"
        )
    return os.getenv("TWIZ_USER_AGENT", "trading-wizzard/0.1 personal research")


def _fetch_url(url: str, timeout: int = 10) -> bytes:
    request = Request(url, headers={"User-Agent": _user_agent_for_url(url)})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_with_feedparser(payload: bytes) -> List[Dict[str, str]]:
    try:
        import feedparser
    except Exception:
        return []
    parsed = feedparser.parse(payload)
    rows: List[Dict[str, str]] = []
    for entry in parsed.entries:
        rows.append(
            {
                "title": str(entry.get("title", "")),
                "text": str(entry.get("summary", "")),
                "url": str(entry.get("link", "")),
                "published_at": str(entry.get("published", "")),
            }
        )
    return rows


def _parse_with_xml(payload: bytes) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    root = ElementTree.fromstring(payload)
    for item in root.findall(".//item"):
        rows.append(
            {
                "title": normalize_text(item.findtext("title") or ""),
                "text": normalize_text(item.findtext("description") or ""),
                "url": normalize_text(item.findtext("link") or ""),
                "published_at": normalize_text(item.findtext("pubDate") or ""),
            }
        )
    for item in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        link = item.find("{http://www.w3.org/2005/Atom}link")
        rows.append(
            {
                "title": normalize_text(item.findtext("{http://www.w3.org/2005/Atom}title") or ""),
                "text": normalize_text(item.findtext("{http://www.w3.org/2005/Atom}summary") or ""),
                "url": link.attrib.get("href", "") if link is not None else "",
                "published_at": normalize_text(
                    item.findtext("{http://www.w3.org/2005/Atom}updated") or ""
                ),
            }
        )
    return rows


def fetch_rss_items(
    feeds: Iterable[Dict[str, str]],
    watchlist: Iterable[WatchedAsset],
    ignore_terms: Iterable[str],
    live: bool = False,
    limit: int = 50,
) -> Tuple[List[RawItem], List[str]]:
    if not live:
        return [], ["RSS live ingest skipped. Pass --live to fetch configured feeds."]

    rows: List[RawItem] = []
    warnings: List[str] = []
    for feed in feeds:
        url = str(feed.get("url", ""))
        if not url:
            continue
        if "sec.gov" in url and not (os.getenv("EDGAR_IDENTITY") or os.getenv("SEC_USER_AGENT")):
            warnings.append(
                "SEC EDGAR identity is not set; add EDGAR_IDENTITY='Your Name email@example.com' "
                "to .env to reduce SEC 403 responses."
            )
        try:
            payload = _fetch_url(url)
            entries = _parse_with_feedparser(payload) or _parse_with_xml(payload)
        except (URLError, TimeoutError, ElementTree.ParseError, ValueError) as exc:
            warnings.append(f"RSS ingest failed for {feed.get('name', url)}: {exc}")
            continue
        for entry in entries:
            text = f"{entry.get('title', '')} {entry.get('text', '')}"
            tickers = extract_tickers(text, watchlist, ignore_terms)
            rows.append(
                RawItem(
                    source=str(feed.get("name", url)),
                    source_type=str(feed.get("source_type", "rss")),
                    url=str(entry.get("url", url)),
                    published_at=str(
                        entry.get("published_at") or datetime.now(timezone.utc).isoformat()
                    ),
                    ticker_candidates=tickers,
                    title=str(entry.get("title", "")),
                    text=str(entry.get("text", "")),
                    engagement={},
                )
            )
            if len(rows) >= limit:
                return rows, warnings
    return rows, warnings
