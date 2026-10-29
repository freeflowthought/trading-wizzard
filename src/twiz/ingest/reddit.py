import html
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ..config import WatchedAsset
from ..models import RawItem
from ..text import extract_tickers

TAG_RE = re.compile(r"<[^>]+>")
JINA_FALSE_VALUES = {"0", "false", "no", "off"}
JINA_TITLE_RE = re.compile(r"^(?:\[\d+\]\s*)?Title:\s*(.*)$")
JINA_URL_RE = re.compile(r"^(?:\[\d+\]\s*)?URL Source:\s*(.*)$")
JINA_DATE_RE = re.compile(r"^(?:\[\d+\]\s*)?(?:Published Time|Published):\s*(.*)$")
JINA_FIELD_RE = re.compile(r"^(?:\[\d+\]\s*)?[A-Za-z][A-Za-z ]{1,32}:\s*(.*)$")


def _clean_reddit_html(value: str) -> str:
    return TAG_RE.sub(" ", html.unescape(value or "")).strip()


def _fetch_reddit_rss(url: str, timeout: int = 10) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": os.getenv("REDDIT_USER_AGENT", "trading-wizzard/0.1")},
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _atom_text(entry: ElementTree.Element, tag: str) -> str:
    return _clean_reddit_html(entry.findtext(f"{{http://www.w3.org/2005/Atom}}{tag}") or "")


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in JINA_FALSE_VALUES


def _best_user_agent() -> str:
    return (
        os.getenv("TWIZ_USER_AGENT")
        or os.getenv("REDDIT_USER_AGENT")
        or "trading-wizzard/0.1 personal market report"
    )


def _parse_jina_json(payload: str) -> List[Dict[str, str]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    candidates = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(candidates, list):
        return []
    rows: List[Dict[str, str]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or item.get("source") or "").strip()
        text = str(
            item.get("content") or item.get("description") or item.get("snippet") or ""
        ).strip()
        published_at = str(item.get("publishedTime") or item.get("published_at") or "").strip()
        if title and url:
            rows.append(
                {"title": title, "url": url, "text": text, "published_at": published_at}
            )
    return rows


def parse_jina_search_results(payload: str) -> List[Dict[str, str]]:
    json_rows = _parse_jina_json(payload)
    if json_rows:
        return json_rows

    rows: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    body_lines: List[str] = []

    def flush() -> None:
        nonlocal current, body_lines
        if current and current.get("title") and current.get("url"):
            current["text"] = "\n".join(line for line in body_lines if line).strip()
            rows.append(current)
        current = None
        body_lines = []

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        title_match = JINA_TITLE_RE.match(line)
        if title_match:
            flush()
            current = {
                "title": title_match.group(1).strip(),
                "url": "",
                "text": "",
                "published_at": "",
            }
            continue
        if current is None:
            continue
        url_match = JINA_URL_RE.match(line)
        if url_match:
            current["url"] = url_match.group(1).strip()
            continue
        date_match = JINA_DATE_RE.match(line)
        if date_match:
            current["published_at"] = date_match.group(1).strip()
            continue
        if line.lower() == "markdown content:":
            continue
        field_match = JINA_FIELD_RE.match(line)
        body_lines.append(field_match.group(1).strip() if field_match else line)
    flush()
    return rows


def _fetch_jina_search(query: str, timeout: int = 15) -> str:
    headers = {"User-Agent": _best_user_agent(), "Accept": "text/plain"}
    api_key = os.getenv("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"https://s.jina.ai/{quote(query, safe='')}", headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _asset_search_query(asset: WatchedAsset) -> str:
    terms = " ".join(dict.fromkeys([asset.ticker, asset.name, *asset.reddit_terms[:2]]))
    return f"{terms} stock ETF discussion site:reddit.com"


def _matches_asset(text: str, asset: WatchedAsset) -> bool:
    lowered = text.lower()
    terms = [asset.ticker, asset.name, *asset.aliases, *asset.reddit_terms]
    for term in terms:
        normalized = str(term).strip().lower().lstrip("$")
        if normalized and normalized in lowered:
            return True
    return False


def _reddit_source_from_url(url: str) -> str:
    match = re.search(r"reddit\.com/r/([^/]+)", url, flags=re.IGNORECASE)
    return f"r/{match.group(1)} via Jina" if match else "reddit via Jina"


def _jina_search_enabled() -> bool:
    if "JINA_SEARCH_ENABLED" in os.environ:
        return _env_flag("JINA_SEARCH_ENABLED", default=False)
    if "JINA_ENABLED" in os.environ:
        return _env_flag("JINA_ENABLED", default=False)
    return bool(os.getenv("JINA_API_KEY"))


def fetch_jina_reddit_search_items(
    watchlist: Iterable[WatchedAsset],
    ignore_terms: Iterable[str],
    limit: int = 25,
) -> Tuple[List[RawItem], List[str]]:
    if limit <= 0:
        return [], []
    if not _jina_search_enabled():
        return [], []
    if not os.getenv("JINA_API_KEY"):
        return [], ["Jina Search fallback skipped because JINA_API_KEY is not set."]

    assets = list(watchlist)
    max_queries = _bounded_setting("JINA_MAX_QUERIES", 3, 12)
    rows: List[RawItem] = []
    warnings: List[str] = []
    seen = set()

    for asset in assets[:max_queries]:
        if len(rows) >= limit:
            break
        query = _asset_search_query(asset)
        try:
            results = parse_jina_search_results(_fetch_jina_search(query, timeout=5))
        except Exception as exc:
            warnings.append(f"Jina Search fallback failed for {asset.ticker}: {exc}")
            continue

        for result in results:
            url = result.get("url", "")
            if not url or url in seen or "reddit.com" not in url.lower():
                continue
            seen.add(url)
            title = result.get("title", "")
            body = result.get("text", "")
            combined = f"{title} {body}"
            tickers = extract_tickers(combined, assets, ignore_terms)
            if not tickers and _matches_asset(combined, asset):
                tickers = [asset.ticker]
            if not tickers:
                continue
            rows.append(
                RawItem(
                    source=_reddit_source_from_url(url),
                    source_type="jina_reddit_search",
                    url=url,
                    published_at=result.get("published_at") or "",
                    ticker_candidates=tickers,
                    title=title,
                    text=body,
                    engagement={"score": 0.0},
                )
            )
            if len(rows) >= limit:
                return rows, warnings
    return rows, warnings


def fetch_reddit_fallback_items(
    watchlist: Iterable[WatchedAsset],
    subreddits: Iterable[str],
    ignore_terms: Iterable[str],
    limit: int = 25,
) -> Tuple[List[RawItem], List[str]]:
    rows: List[RawItem] = []
    warnings: List[str] = []
    watch_assets = list(watchlist)

    rss_rows, rss_warnings = fetch_reddit_rss_items(
        watch_assets,
        subreddits,
        ignore_terms,
        limit=limit,
    )
    rows.extend(rss_rows)
    warnings.extend(rss_warnings)

    remaining = max(0, limit - len(rows))
    jina_rows: List[RawItem] = []
    if remaining:
        jina_rows, jina_warnings = fetch_jina_reddit_search_items(
            watch_assets,
            ignore_terms,
            limit=remaining,
        )
        rows.extend(jina_rows)
        warnings.extend(jina_warnings)

    if rss_rows and jina_rows:
        warnings.insert(
            0,
            "Reddit credentials missing; used Reddit RSS plus Jina Search fallback "
            "with limited engagement metadata.",
        )
    elif rss_rows:
        warnings.insert(
            0,
            "Reddit credentials missing; used Reddit RSS fallback with limited metadata.",
        )
    elif jina_rows:
        warnings.insert(
            0,
            "Reddit credentials missing; used Jina Search fallback with limited "
            "engagement metadata.",
        )
    else:
        warnings.insert(
            0,
            "Reddit credentials missing and no fallback returned ticker-matched rows. "
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET for fuller coverage, or set "
            "JINA_SEARCH_ENABLED=true with JINA_API_KEY for web-search fallback.",
        )
    return rows, warnings


def _append_reddit_atom_entries(
    rows: List[RawItem],
    root: ElementTree.Element,
    source: str,
    watchlist: Iterable[WatchedAsset],
    ignore_terms: Iterable[str],
    seen: set,
    limit: int,
) -> None:
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        link = entry.find("{http://www.w3.org/2005/Atom}link")
        entry_url = link.attrib.get("href", "") if link is not None else ""
        if not entry_url or entry_url in seen:
            continue
        seen.add(entry_url)
        title = _atom_text(entry, "title")
        body = _atom_text(entry, "content") or _atom_text(entry, "summary")
        tickers = extract_tickers(f"{title} {body}", watchlist, ignore_terms)
        if not tickers:
            continue
        rows.append(
            RawItem(
                source=source,
                source_type="reddit_rss",
                url=entry_url,
                published_at=_atom_text(entry, "updated") or _atom_text(entry, "published"),
                ticker_candidates=tickers,
                title=title,
                text=body,
                engagement={"score": 1.0},
            )
        )
        if len(rows) >= limit:
            return


def _reddit_listing_rss_url(subreddit_name: str, sort: str) -> str:
    clean_sort = sort.strip().lower()
    if clean_sort in {"", "hot"}:
        return f"https://www.reddit.com/r/{subreddit_name}/.rss"
    if clean_sort in {"new", "top", "rising", "controversial"}:
        return f"https://www.reddit.com/r/{subreddit_name}/{clean_sort}.rss"
    return f"https://www.reddit.com/r/{subreddit_name}/new.rss"


def _bounded_setting(name: str, default: int, maximum: int) -> int:
    try:
        return max(1, min(int(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


def fetch_reddit_rss_items(
    watchlist: Iterable[WatchedAsset],
    subreddits: Iterable[str],
    ignore_terms: Iterable[str],
    limit: int = 25,
) -> Tuple[List[RawItem], List[str]]:
    if limit <= 0:
        return [], []
    rows: List[RawItem] = []
    warnings: List[str] = []
    seen = set()
    watch_assets = list(watchlist)
    subreddit_names = list(dict.fromkeys(subreddits))
    if not watch_assets or not subreddit_names:
        return [], []
    # One combined query per subreddit avoids a request for every ticker alias.
    query = " OR ".join(f'"{asset.ticker}"' for asset in watch_assets)
    requests = []
    if _env_flag("REDDIT_RSS_EXPANDED", default=True):
        sorts = list(dict.fromkeys(
            sort.strip() for sort in os.getenv("REDDIT_RSS_SORTS", "new,hot").split(",")
        ))
        for sort in sorts:
            for name in subreddit_names:
                requests.append((_reddit_listing_rss_url(name, sort), f"r/{name} {sort}"))
    for name in subreddit_names:
        requests.append((
            f"https://www.reddit.com/r/{name}/search.rss"
            f"?q={quote_plus(query)}&restrict_sr=on&sort=new&t=week", f"r/{name} search",
        ))
    maximum = _bounded_setting("REDDIT_RSS_MAX_REQUESTS", 12, 30)
    deadline = time.monotonic() + _bounded_setting("REDDIT_RSS_BUDGET_SECONDS", 30, 120)
    for index, (url, source) in enumerate(dict.fromkeys(requests)):
        remaining = deadline - time.monotonic()
        if index >= maximum or remaining <= 0:
            warnings.append("Reddit RSS request/time budget reached; coverage is partial.")
            break
        try:
            root = ElementTree.fromstring(_fetch_reddit_rss(url, timeout=min(5, remaining)))
        except Exception as exc:
            warnings.append(f"Reddit RSS fallback failed for {source}: {exc}")
            continue
        _append_reddit_atom_entries(
            rows, root, source, watch_assets, ignore_terms, seen, limit,
        )
        if len(rows) >= limit:
            return rows, warnings
    return rows, warnings


def fetch_reddit_items(
    watchlist: Iterable[WatchedAsset],
    subreddits: Iterable[str],
    ignore_terms: Iterable[str],
    limit: int = 25,
    live: bool = False,
) -> Tuple[List[RawItem], List[str]]:
    if limit <= 0:
        return [], []
    if not live:
        return [], ["Reddit live ingest skipped. Pass --live and configure PRAW credentials."]

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "trading-wizzard/0.1 personal market report")
    if not client_id or not client_secret:
        return fetch_reddit_fallback_items(
            watchlist,
            subreddits,
            ignore_terms,
            limit=limit,
        )

    try:
        import praw
    except Exception as exc:
        return [], [f"PRAW unavailable: {exc}"]

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            requestor_kwargs={"timeout": 5},
            ratelimit_seconds=0,
        )
    except Exception as exc:
        return [], [f"Failed to initialize PRAW: {exc}"]

    rows: List[RawItem] = []
    warnings: List[str] = []
    watchlist = list(watchlist)
    terms = [" OR ".join(f'"{asset.ticker}"' for asset in watchlist)]
    if not watchlist:
        return [], []
    deadline = time.monotonic() + 30
    per_query_limit = max(1, min(limit, 10))
    seen = set()

    for subreddit_name in subreddits:
        if time.monotonic() >= deadline:
            warnings.append("Reddit API time budget reached; coverage is partial.")
            break
        try:
            subreddit = reddit.subreddit(str(subreddit_name))
            for term in terms:
                submissions = subreddit.search(
                    term,
                    sort="new",
                    time_filter="week",
                    limit=per_query_limit,
                )
                for submission in submissions:
                    if submission.id in seen:
                        continue
                    seen.add(submission.id)
                    title = getattr(submission, "title", "") or ""
                    body = getattr(submission, "selftext", "") or ""
                    tickers = extract_tickers(f"{title} {body}", watchlist, ignore_terms)
                    if not tickers:
                        continue
                    created = datetime.fromtimestamp(
                        float(getattr(submission, "created_utc", 0) or 0),
                        tz=timezone.utc,
                    )
                    rows.append(
                        RawItem(
                            source=f"r/{subreddit_name}",
                            source_type="reddit",
                            url=f"https://www.reddit.com{getattr(submission, 'permalink', '')}",
                            published_at=created.isoformat(),
                            ticker_candidates=tickers,
                            title=title,
                            text=body,
                            engagement={
                                "score": float(getattr(submission, "score", 0) or 0),
                                "num_comments": float(getattr(submission, "num_comments", 0) or 0),
                            },
                        )
                    )
                    if len(rows) >= limit:
                        return rows, warnings
        except Exception as exc:
            warnings.append(f"Reddit ingest failed for r/{subreddit_name}: {exc}")
    if rows:
        return rows, warnings

    fallback_rows, fallback_warnings = fetch_reddit_fallback_items(
        watchlist,
        subreddits,
        ignore_terms,
        limit=limit,
    )
    if fallback_rows:
        warnings.append(
            "PRAW returned no ticker-matched rows; used fallback social search sources."
        )
    return fallback_rows, warnings + fallback_warnings
