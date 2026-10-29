"""Forward event calendar — the layer Shunge's calls are anchored to.

His timeline is never just price levels: every level call is paired with a
future event and a branch ("if Jackson Hole disappoints, 7750 is the top;
if it passes, the path to 8000 opens"). This module builds the scan he
effectively runs by hand:

- rule-based market events (FOMC, CPI, jobs report, options expiry,
  quad witching, US elections, Jackson Hole window, seasonal danger months)
- per-ticker earnings dates (yfinance when live, manual config otherwise)
- manual events from config/events.yml (product launches, lockup expiries,
  shareholder meetings — the TSLA Semi / CBRS unlock type items)

Every event carries an importance score and a ``playbook_rule`` describing
the standing behaviour around it, so the playbook layer can attach branch
notes instead of treating the calendar as decoration.
"""

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

# FOMC meeting end dates (second day of each meeting), Federal Reserve
# published schedule. Extend as new schedules are announced.
FOMC_DATES = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

# Months where Shunge (and the seasonal record) expects real drawdowns.
SEASONAL_RISK = {
    9: ("high", "September: seasonally weakest month; Shunge budgets a deeper low than August."),
    8: ("medium", "August: usually a shallow dip only; do not over-hedge a normal pullback."),
    10: ("medium", "October: crash anniversaries, but also where midterm rallies launch."),
    2: ("low_medium", "February: common secondary pullback window."),
}


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(days=14)


def _first_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7)


def _us_election_day(year: int) -> date:
    """Tuesday after the first Monday of November (federal election day)."""
    first = date(year, 11, 1)
    first_monday = first + timedelta(days=(0 - first.weekday()) % 7)
    return first_monday + timedelta(days=1)


def _event(
    when: date,
    kind: str,
    name: str,
    importance: int,
    tickers: Optional[List[str]] = None,
    precision: str = "scheduled",
    playbook_rule: str = "",
) -> Dict[str, Any]:
    return {
        "date": when.isoformat(),
        "kind": kind,
        "name": name,
        "importance": importance,
        "tickers": tickers or [],
        "precision": precision,
        "playbook_rule": playbook_rule,
    }


def rule_based_events(start: date, horizon_days: int = 45) -> List[Dict[str, Any]]:
    """Deterministic calendar events in [start, start + horizon_days]."""
    end = start + timedelta(days=horizon_days)
    events: List[Dict[str, Any]] = []

    for raw in FOMC_DATES:
        when = date.fromisoformat(raw)
        if start <= when <= end:
            events.append(_event(
                when, "fomc", "FOMC rate decision", 5,
                playbook_rule=(
                    "Branch point: a calm/dovish outcome opens the path to the next "
                    "round-number target; a hawkish surprise makes the recent high the "
                    "short-term top. No new leveraged bets before the statement."
                ),
            ))

    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        year, month = cursor.year, cursor.month
        opex = _third_friday(year, month)
        if start <= opex <= end:
            quad = month in (3, 6, 9, 12)
            events.append(_event(
                opex,
                "quad_witching" if quad else "opex",
                "Quad witching / options expiry" if quad else "Monthly options expiry",
                4 if quad else 3,
                playbook_rule=(
                    "Expiry weeks concentrate put/call walls; Shunge watches crowded "
                    "put strikes as squeeze fuel and expects outsized moves into expiry."
                ),
            ))
        nfp = _first_friday(year, month)
        if start <= nfp <= end:
            events.append(_event(
                nfp, "jobs_report", "US jobs report (NFP, estimated date)", 4,
                precision="approximate",
                playbook_rule=(
                    "Jobs data moves the 10Y first and equities second; check the macro "
                    "gate before reacting to the headline."
                ),
            ))
        cpi = date(year, month, 12)
        if start <= cpi <= end:
            events.append(_event(
                cpi, "cpi", "US CPI release (approx. date)", 4, precision="approximate",
                playbook_rule=(
                    "Inflation print: a hot number pushes the 10Y toward the danger "
                    "zone; a cool number is fuel for the rally leg."
                ),
            ))
        # Jackson Hole symposium: typically the last full weekend of August.
        if month == 8:
            jh = _third_friday(year, 8) + timedelta(days=7)
            if start <= jh <= end:
                events.append(_event(
                    jh, "jackson_hole", "Jackson Hole symposium (approx.)", 4,
                    precision="approximate",
                    playbook_rule=(
                        "Shunge treats Jackson Hole as a regime branch: trade small "
                        "or not at all until the speech is out."
                    ),
                ))
        cursor = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)

    for year in range(start.year, end.year + 1):
        if year % 2:
            continue
        election = _us_election_day(year)
        if start <= election <= end:
            midterm = year % 4 == 2
            events.append(_event(
                election,
                "election",
                "US midterm election" if midterm else "US presidential election",
                5,
                playbook_rule=(
                    "Standing rule from the timeline: do not short into a midterm — "
                    "shorts are fuel for the pre-election rally to the index target."
                ),
            ))

    return events


def seasonal_context(as_of: date) -> Dict[str, Any]:
    level, note = SEASONAL_RISK.get(
        as_of.month, ("normal", "No strong seasonal signal this month.")
    )
    return {"month": as_of.month, "risk": level, "note": note}


def earnings_events(
    earnings_dates: Dict[str, str],
    start: date,
    horizon_days: int = 45,
) -> List[Dict[str, Any]]:
    end = start + timedelta(days=horizon_days)
    events = []
    for ticker, raw in earnings_dates.items():
        try:
            when = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if start <= when <= end:
            events.append(_event(
                when, "earnings", f"{ticker} earnings", 5, tickers=[ticker.upper()],
                playbook_rule=(
                    "Check an option expiry after the earnings date for event context. "
                    "Heavy positions with profit can "
                    "be trimmed into the print; light positions can wait it out."
                ),
            ))
    return events


def manual_events(
    rows: List[Dict[str, Any]],
    start: date,
    horizon_days: int = 45,
) -> List[Dict[str, Any]]:
    """User-maintained events from config/events.yml.

    Row shape: {date, name, kind?, importance?, tickers?, note?}
    Use this for product launches, lockup expiries, investor days — the
    events with no public machine-readable calendar.
    """
    end = start + timedelta(days=horizon_days)
    events = []
    for row in rows or []:
        try:
            when = date.fromisoformat(str(row.get("date", ""))[:10])
        except ValueError:
            continue
        if not (start <= when <= end):
            continue
        events.append(_event(
            when,
            str(row.get("kind", "custom")),
            str(row.get("name", "Custom event")),
            int(row.get("importance", 3)),
            tickers=[str(t).upper() for t in row.get("tickers", [])],
            playbook_rule=str(row.get("note", "")),
        ))
    return events


def fetch_yfinance_earnings_dates(tickers: List[str]) -> Dict[str, str]:
    """Best-effort next earnings date per ticker via yfinance calendar."""
    dates: Dict[str, str] = {}
    try:
        import yfinance as yf
    except Exception:
        return dates
    for ticker in tickers:
        try:
            cal = yf.Ticker(ticker).calendar
            raw = None
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date")
                if isinstance(raw, (list, tuple)):
                    raw = raw[0] if raw else None
            if raw is not None:
                dates[ticker] = str(raw)[:10]
        except Exception:
            continue
    return dates


def build_event_calendar(
    as_of: date,
    horizon_days: int = 45,
    earnings_dates: Optional[Dict[str, str]] = None,
    manual: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    events = rule_based_events(as_of, horizon_days)
    events.extend(earnings_events(earnings_dates or {}, as_of, horizon_days))
    events.extend(manual_events(manual or [], as_of, horizon_days))
    events.sort(key=lambda row: (row["date"], -int(row["importance"])))
    for row in events:
        row["days_away"] = (date.fromisoformat(row["date"]) - as_of).days
    return events


def next_event_for(
    events: List[Dict[str, Any]],
    ticker: Optional[str] = None,
    min_importance: int = 4,
) -> Optional[Dict[str, Any]]:
    """Nearest market-wide event, plus the ticker's own earnings if sooner."""
    candidates = [
        row for row in events
        if int(row.get("importance", 0)) >= min_importance
        and (not row.get("tickers") or (ticker and ticker.upper() in row["tickers"]))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row["date"])[0]


def political_context(as_of: date) -> Dict[str, Any]:
    """The Trump/political-cycle layer from the June timeline.

    His rules are cycle-based, not headline-based: in a midterm year the
    administration is incentivised to defend asset prices into November
    (deals get signed, confrontations get TACO'd), so shorting the
    run-up means "singing against Trump". After the election the
    political bid fades and the next cycle's risk (he flags 2027-28)
    takes over.
    """
    year = as_of.year
    kind = None
    if year % 4 == 2:
        kind = "midterm"
    elif year % 4 == 0:
        kind = "presidential"
    if kind is None:
        return {
            "election_type": None,
            "election_date": None,
            "days_away": None,
            "phase": "off_year",
            "rule": "No federal election this year; the political bid is weak.",
        }
    election = _us_election_day(year)
    days_away = (election - as_of).days
    if days_away < 0:
        phase = "post_election"
        rule = (
            "Election is behind us; the pre-election political bid fades. "
            "His calendar turns defensive for the following cycle."
        )
    elif days_away <= 300:
        phase = "election_runup"
        rule = (
            f"{kind.capitalize()} run-up ({days_away}d to the vote): the White House "
            "wants new highs into November — do not short this bid; dips are for "
            "the staged ladder, and headline shocks (tariffs, wars) tend to be "
            "walked back (TACO) or settled."
        )
    else:
        phase = "early_cycle"
        rule = "Election year but far from the vote; the political bid is not yet active."
    return {
        "election_type": kind,
        "election_date": election.isoformat(),
        "days_away": days_away,
        "phase": phase,
        "rule": rule,
    }
