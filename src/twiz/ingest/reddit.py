import html
import os
import re
from datetime import datetime, timezone
from typing import Iterable, List, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ..config import WatchedAsset
from ..models import RawItem
from ..text import extract_tickers

TAG_RE = re.compile(r"<[^>]+>")


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


def fetch_reddit_rss_items(
    watchlist: Iterable[WatchedAsset],
    subreddits: Iterable[str],
    ignore_terms: Iterable[str],
    limit: int = 25,
) -> Tuple[List[RawItem], List[str]]:
    rows: List[RawItem] = []
    warnings: List[str] = []
    seen = set()
    terms = sorted({term for asset in watchlist for term in asset.reddit_terms})
    for subreddit_name in subreddits:
        for term in terms:
            url = (
                f"https://www.reddit.com/r/{subreddit_name}/search.rss"
                f"?q={quote_plus(term)}&restrict_sr=on&sort=new&t=week"
            )
            try:
                root = ElementTree.fromstring(_fetch_reddit_rss(url))
            except Exception as exc:
                warnings.append(f"Reddit RSS fallback failed for r/{subreddit_name} {term}: {exc}")
                continue

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
                        source=f"r/{subreddit_name}",
                        source_type="reddit_rss",
                        url=entry_url,
                        published_at=_atom_text(entry, "updated")
                        or datetime.now(timezone.utc).isoformat(),
                        ticker_candidates=tickers,
                        title=title,
                        text=body,
                        engagement={"score": 1.0},
                    )
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
    if not live:
        return [], ["Reddit live ingest skipped. Pass --live and configure PRAW credentials."]

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "trading-wizzard/0.1 personal market report")
    if not client_id or not client_secret:
        rows, rss_warnings = fetch_reddit_rss_items(
            watchlist,
            subreddits,
            ignore_terms,
            limit=limit,
        )
        if rows:
            return rows, [
                "Reddit credentials missing; used Reddit RSS fallback with limited metadata."
            ] + rss_warnings
        return [], [
            "Reddit credentials missing and RSS fallback returned no ticker-matched rows. "
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET for fuller coverage."
        ] + rss_warnings

    try:
        import praw
    except Exception as exc:
        return [], [f"PRAW unavailable: {exc}"]

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
    except Exception as exc:
        return [], [f"Failed to initialize PRAW: {exc}"]

    rows: List[RawItem] = []
    warnings: List[str] = []
    terms = sorted({term for asset in watchlist for term in asset.reddit_terms})
    per_query_limit = max(1, min(limit, 10))
    seen = set()

    for subreddit_name in subreddits:
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
    return rows, warnings
