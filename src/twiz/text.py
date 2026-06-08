import re
from typing import Dict, Iterable, List, Set, Tuple

from .config import WatchedAsset

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def build_alias_index(watchlist: Iterable[WatchedAsset]) -> List[Tuple[str, str]]:
    aliases = []
    for asset in watchlist:
        for alias in set(asset.aliases + [asset.ticker]):
            aliases.append((alias, asset.ticker))
    return sorted(aliases, key=lambda item: len(item[0]), reverse=True)


def _alias_pattern(alias: str) -> Tuple[re.Pattern, int]:
    if alias.startswith("$"):
        return re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE), 0
    if " " in alias or not alias.isupper():
        return re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE), 0
    return re.compile(r"(?<![A-Za-z0-9$])" + re.escape(alias) + r"(?![A-Za-z0-9])"), 0


def extract_tickers(
    text: str,
    watchlist: Iterable[WatchedAsset],
    ignore_terms: Iterable[str],
) -> List[str]:
    normalized = normalize_text(text)
    ignored: Set[str] = {term.upper() for term in ignore_terms}
    found: Set[str] = set()
    for alias, ticker in build_alias_index(watchlist):
        if alias.upper() in ignored:
            continue
        pattern, _ = _alias_pattern(alias)
        if pattern.search(normalized):
            found.add(ticker)
    return sorted(found)


def item_summary_text(title: str, text: str) -> str:
    title = normalize_text(title)
    text = normalize_text(text)
    if title and text:
        return f"{title}. {text}"
    return title or text


def group_by_ticker(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        for ticker in row.get("ticker_candidates", []) or []:
            grouped.setdefault(str(ticker), []).append(row)
    return grouped
