"""Turn general news headlines into auditable cycle event rows.

Headlines are matched against a small explicit taxonomy (trade policy, geopolitics,
financial stress, fiscal policy, public health). Threats, rumours, questions and
explainers are excluded. A family/direction becomes a trading event only when at
least two distinct publishers report it inside the corroboration window; the row's
availability is the clock at which that corroboration was first observed, never the
earlier publication time. Rows use the same schema as the FRED target-rate events,
so the existing threshold, TTL, family and once-per-episode rules apply unchanged.
"""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

SCHEMA_VERSION = "news-classifier-1"
CORROBORATION_HOURS = 36
MIN_PUBLISHERS = 2
TTL_SESSIONS = 5

# Lookarounds instead of \b: "U.S." ends in a non-word character.
ACTORS = (r"(?<!\w)(us|u\.s\.|united states|america|trump|biden|white house|washington|"
          r"china|beijing|"
          r"eu|european union|mexico|canada|japan|russia|ukraine|taiwan|iran|israel|"
          r"north korea|nato|fed|treasury|congress)(?!\w)")
TARGETS = r"(ukraine|taiwan|israel|iran|poland|the baltics?)"

TAXONOMY = (
    ("trade_policy", -0.5, (
        r"\b(impos|announc|unveil|rais|hik|slap|escalat|expand|hits?\b.{0,30}\bwith)\w*\b.{0,60}\btariffs?\b",
        r"\btariffs?\b.{0,40}\b(take|took|go|went|come|came) into effect\b",
        r"\btariffs? (rise|rises|jump|jumps|hike|hikes)\b",
        r"\btrade war (escalat|deepen|intensif)\w*",
        r"\b(new|sweeping|expanded) export (ban|controls?)\b",
    )),
    ("trade_policy", 0.5, (
        r"\btariffs?\b.{0,40}\b(pause|paused|delay|delayed|suspend|suspended|exempt|lowered|"
        r"roll(ed)? back)\b",
        r"\b(pause|pauses|paused|delay|delays|delayed|suspend|suspends|suspended)\b.{0,40}\btariffs?\b",
        r"\b(sign|signs|signed|reach|reaches|reached|strike|strikes|struck)\b.{0,30}"
        r"\btrade (deal|truce|agreement)\b",
    )),
    ("geopolitics", -0.5, (
        ACTORS + r".{0,40}\b(invades|invaded|launches invasion|begins invasion)\b",
        r"\binvasion of " + TARGETS + r"\b.{0,40}\b(begins|began|launched|underway)\b",
        r"\b(launch|launches|launched|begin|begins|began)\b.{0,25}\binvasion of " + TARGETS + r"\b",
        r"\b(launch|launches|launched|carry out|carries out|carried out)\b.{0,30}"
        r"\b(missile|air) ?strikes?\b",
        ACTORS + r".{0,30}\bdeclares? war\b",
        r"\b(closes|closed|blockade[sd]?)\b.{0,30}\bstrait of hormuz\b",
    )),
    ("geopolitics", 0.5, (
        # Only ceasefires involving a market-relevant actor; local conflicts are context.
        ACTORS + r".{0,40}\b(agree|agrees|agreed|reach|reaches|reached|sign|signs|signed)\b"
        r".{0,30}\b(cease-?fire|peace (deal|agreement|accord))\b",
    )),
    ("financial_stress", -0.5, (
        r"\bbank (run|runs|failure|failures|collapse|collapses)\b",
        r"\b(fdic|regulators?)\b.{0,30}\b(seize|seizes|seized|close|closes|closed|takes? over)\b",
        ACTORS + r".{0,30}\b(defaults|defaulted) on\b.{0,30}\b(debt|bonds?)\b",
        r"\bliquidity crisis\b",
    )),
    ("fiscal_policy", -0.5, (
        r"\b(government|federal) shutdown (begins|began|starts|started)\b",
        r"\b(us |u\.s\. |federal )?government shuts down\b",
        r"\b(downgrades?|downgraded|cuts?|strips?|stripped)\b.{0,40}"
        r"(?<!\w)(us|u\.s\.|united states|america's)(?!\w).{0,40}\b(credit|rating|aaa)\b",
        r"\bdebt ceiling\b.{0,30}\b(breach|breached|default)\b",
    )),
    ("fiscal_policy", 0.5, (
        r"\bshutdown ends\b|\bends? (the )?(government )?shutdown\b",
        r"\bdebt ceiling (deal|agreement)\b.{0,30}\b(pass|passes|passed|reached|signed)\b",
    )),
    ("public_health", -0.5, (
        r"\b(declares?|declared)\b.{0,50}\b(pandemic|public health emergency)\b",
        r"\b(nationwide|national|citywide) lockdowns?\b",
    )),
)

# Conditional, hypothetical, denied or explanatory wording is context, not a fact.
EXCLUDE = re.compile(
    r"\b(threat|threatens?|threatened|considers?|weighs?|mulls?|could|may|might|would|"
    r"if|rumou?rs?|unconfirmed|denies|denied|rules? out|no plans?|explained|explainer|"
    r"what to know|how do|how does|why|opinion|analysis|timeline|q&a|podcast|live updates?|"
    r"d-day|history|historic|anniversary|years (ago|after|later)|review|film|movie|book|"
    r"privacy|robots?|cancer|tumou?r|species|weeds?|seaweed|flowers?|"
    # Company pricing, refunds and avoidance stories are tariff consequences, not policy.
    r"prices?|pricing|refunds?|sidestep|avoid|costs?|plan to|try to|tries to)\b"
    r"|\?",
    re.IGNORECASE,
)
COMPILED = tuple((family, direction, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
                 for family, direction, patterns in TAXONOMY)

# Corroboration counts only established national/wire/financial outlets.
PUBLISHERS = frozenset(name.lower() for name in (
    "Reuters", "AP News", "Associated Press", "Bloomberg", "Bloomberg.com", "CNBC",
    "The Wall Street Journal", "WSJ", "Financial Times", "The New York Times",
    "The Washington Post", "MarketWatch", "Barron's", "Yahoo Finance", "Axios", "CNN",
    "CNN Business", "BBC", "BBC News", "The Guardian", "Politico", "POLITICO", "Fox Business",
    "Fox News", "NPR", "ABC News", "CBS News", "NBC News", "The Hill", "Al Jazeera",
    "The Economist", "Forbes", "Business Insider", "Investopedia", "Nikkei Asia",
    "South China Morning Post", "Kiplinger", "U.S. News & World Report", "PBS",
))

QUERIES = {
    "trade_policy": "tariffs OR \"trade deal\" OR \"export controls\"",
    "geopolitics": "invasion OR \"missile strike\" OR ceasefire OR \"declares war\"",
    "financial_stress": "\"bank failure\" OR \"bank run\" OR \"liquidity crisis\" OR default",
    "fiscal_policy": "shutdown OR \"debt ceiling\" OR downgrade",
    "public_health": "pandemic OR lockdown OR \"public health emergency\"",
}
GOOGLE_NEWS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def classify(title):
    """Return (family, direction) for one headline, or None when not a factual trigger."""
    text = " ".join(str(title or "").split())
    if not text or EXCLUDE.search(text):
        return None
    hits = {(family, direction) for family, direction, patterns in COMPILED
            if any(p.search(text) for p in patterns)}
    families = {family for family, _ in hits}
    # Opposite directions in one family inside one headline stay neutral.
    if len(hits) != 1 or len(families) != 1:
        return None
    return next(iter(hits))


def _utc(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else None
    try:
        return parsedate_to_datetime(str(value)).astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
        except ValueError:
            return None


def parse_rss(payload, query_family=None):
    items = []
    root = ElementTree.fromstring(payload)
    for item in root.iter("item"):
        source = item.find("source")
        title = (item.findtext("title") or "").strip()
        publisher = (source.text or "").strip() if source is not None else ""
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(" - " + publisher)]
        items.append({
            "title": title,
            "publisher": publisher or None,
            "publisher_url": source.get("url") if source is not None else None,
            "url": item.findtext("link"),
            "published_at": _utc(item.findtext("pubDate")),
            "query_family": query_family,
        })
    return items


def fetch_google_news(window="when:2d", timeout=10, opener=urlopen):
    items, failures = [], {}
    for family, query in QUERIES.items():
        url = GOOGLE_NEWS.format(query=quote_plus(f"{query} {window}"))
        try:
            request = Request(url, headers={"User-Agent": "trading-wizzard/news-events"})
            with opener(request, timeout=timeout) as response:
                payload = response.read(4_000_001)
            if len(payload) > 4_000_000:
                raise ValueError("response exceeds bounded size")
            items.extend(parse_rss(payload, family))
        except Exception as error:  # A failing query leaves that family unknown.
            failures[family] = type(error).__name__
    return items, failures


def corroborated_events(items, observed_at, existing_rows=()):
    """Group classified headlines; emit rows only for >=2 publishers within the window."""
    observed_at = _utc(observed_at)
    groups = {}
    for item in items:
        published = _utc(item.get("published_at"))
        publisher = str(item.get("publisher") or "").strip().lower()
        verdict = classify(item.get("title"))
        if (verdict is None or published is None or publisher not in PUBLISHERS
                or published > observed_at):
            continue
        if observed_at - published > timedelta(hours=CORROBORATION_HOURS):
            continue
        groups.setdefault(verdict, []).append({**item, "published_at": published})
    known = {str(row.get("id")) for row in existing_rows if isinstance(row, dict)}
    events = []
    for (family, direction), rows in sorted(groups.items()):
        publishers = {}
        for row in sorted(rows, key=lambda r: r["published_at"]):
            publishers.setdefault(row["publisher"].lower(), row)
        if len(publishers) < MIN_PUBLISHERS:
            continue
        evidence = sorted(publishers.values(), key=lambda r: r["published_at"])
        first = evidence[0]["published_at"]
        year, week, _ = first.isocalendar()
        sign = "adverse" if direction < 0 else "favorable"
        identity = f"news|{family}|{sign}|{year}-W{week:02d}"
        if identity in known:
            continue
        events.append({
            "id": identity,
            "date": first.date().isoformat(),
            "observed_date": observed_at.date().isoformat(),
            "available_at": observed_at.isoformat(),
            "expires_at": (observed_at + timedelta(days=7)).isoformat(),
            "direction": direction,
            "score": direction,
            "episode_id": identity,
            "correlation_family": family,
            "ttl_sessions": TTL_SESSIONS,
            "source": "news classifier v1 (Google News RSS, >=2 publishers)",
            "source_refs": [{
                "publisher": row["publisher"],
                "title": row["title"],
                "url": row.get("url"),
                "published_at": row["published_at"].isoformat(),
                "title_sha256": hashlib.sha256(row["title"].encode()).hexdigest(),
            } for row in evidence[:6]],
            "publisher_count": len(publishers),
            "publication_time_verified": False,
            "availability_basis": "actual_fetch_first_corroborated_not_backdated",
            "classifier": SCHEMA_VERSION,
        })
    return events
