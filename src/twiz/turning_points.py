"""Market-implied turning-point layer for the QQQ/TQQQ/cash cycle.

Traded expectation markets (VIX term structure, high-yield credit versus Treasuries,
the two-year yield versus the Fed target, prediction-market Fed odds) and dated macro
releases become one causal feature row per completed session. The cycle simulator
uses them to trim before price-only tops, to enter after volatility capitulation
instead of waiting for an aged base, and optionally to hold a small TQQQ sleeve only
in calm uptrends. Every value carries the clock at which it was usable.
"""

import copy
import math
from bisect import bisect_right
from datetime import date, datetime, time, timedelta
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

from .sessions import is_us_equity_session

NY = ZoneInfo("America/New_York")
SCHEMA_VERSION = "market-implied-turning-1"

DEFAULT_TURNING = {
    # Stress: term-structure inversion, credit drawdown, hawkish repricing, claims.
    "stress_ratio": 1.0,
    "stress_credit_dd": 1.0,
    "fed_shock_pp": 0.35,
    "claims_rise": 0.2,
    # Claims count only while rising above their 26-week average, not off a crisis peak.
    "claims_trend_filter": True,
    "stress_confirm_sma": 20,
    "stress_retained": 0.7,
    # Capitulation bottom: inversion peak inside lookback, then normalisation.
    "capitulation_ratio": 1.05,
    "capitulation_lookback": 15,
    "normalized_ratio": 0.97,
    "capitulation_weight": 0.75,
    # After a capitulation entry, a re-inversion alone cannot trim for this many sessions
    # while QQQ holds the anchor low.
    "capitulation_grace_sessions": 5,
    # Calm-uptrend leverage sleeve; zero disables it.
    "calm_ratio": 0.88,
    "calm_exit_ratio": 0.95,
    "calm_tqqq": 0.0,
    "calm_vol_target": 0.0,
    # Price-only TQQQ bottoms need the long trend or a washout first.
    "leverage_gate": True,
    # Prediction-market Fed odds replace the 2Y proxy when a quote is available.
    "use_prediction_markets": True,
}

CHOICES = {
    "stress_ratio": (0.95, 1.0, 1.05),
    "stress_credit_dd": (0.02, 0.03, 0.04, 1.0),
    "fed_shock_pp": (0.25, 0.35, 0.5, 9.0),
    "claims_rise": (0.15, 0.2, 0.3, 9.0),
    "claims_trend_filter": (False, True),
    "stress_confirm_sma": (10, 20, 50),
    "stress_retained": (0.0, 0.3, 0.5, 0.7, 1.0),
    "capitulation_ratio": (1.0, 1.05, 1.1, 9.0),
    "capitulation_lookback": (10, 15, 20),
    "normalized_ratio": (0.93, 0.95, 0.97, 1.0),
    "capitulation_weight": (0.25, 0.5, 0.75, 1.0),
    "capitulation_grace_sessions": (0, 3, 5, 10),
    "calm_ratio": (0.85, 0.88, 0.9),
    "calm_exit_ratio": (0.92, 0.95, 1.0),
    "calm_tqqq": (0.0, 0.15, 0.25, 0.35, 0.5),
    "calm_vol_target": (0.0, 0.15, 0.2, 0.25),
    "leverage_gate": (False, True),
    "use_prediction_markets": (False, True),
}


def validate_turning(parameters):
    if not isinstance(parameters, dict) or set(parameters) != set(CHOICES):
        raise ValueError("Turning layer requires the exact declared parameters.")
    for key, values in CHOICES.items():
        value = parameters[key]
        if isinstance(value, bool) != isinstance(values[0], bool) or value not in values:
            raise ValueError(f"Turning parameter outside declared choices: {key}")
    return copy.deepcopy(parameters)


def _clock(day, hour=16, minute=15):
    return datetime.combine(day, time(hour, minute), NY)


def _next_session(day):
    day += timedelta(days=1)
    while not is_us_equity_session(day):
        day += timedelta(days=1)
    return day


def _closes(rows):
    return {row["date"]: row["close"] for row in rows if row.get("close") is not None}


def _as_of(series, cutoff):
    """Latest (date, value) whose availability is not after cutoff; series sorted by clock."""
    index = bisect_right([clock for clock, _, _ in series], cutoff) - 1
    return series[index][1:] if index >= 0 else (None, None)


def _fred_daily(rows):
    # H.15/target values for day d are published after that day; use next session close.
    out = []
    for day, value in rows:
        observed = date.fromisoformat(day)
        out.append((_clock(_next_session(observed)), day, value))
    return sorted(out)


def _fred_claims(rows):
    # Week ending Saturday is released the following Thursday 08:30 ET.
    out = []
    for day, value in rows:
        out.append((datetime.combine(date.fromisoformat(day) + timedelta(days=5),
                                     time(8, 30), NY), day, value))
    return sorted(out)


def _pm_series(prediction_rows):
    """Expected next-meeting policy change in percentage points from Fed-odds quotes."""
    out = []
    for row in prediction_rows or []:
        probabilities = row.get("probabilities")
        stamp = row.get("available_at")
        if not isinstance(probabilities, dict) or not stamp:
            continue
        try:
            clock = datetime.fromisoformat(str(stamp))
        except ValueError:
            continue
        cut, hike = probabilities.get("cut"), probabilities.get("hike")
        if not all(isinstance(v, (int, float)) and 0 <= v <= 2 for v in (cut, hike)):
            continue
        try:
            expires = datetime.fromisoformat(str(row.get("expires_at")).replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append((clock, (row.get("id"), expires), 0.25 * (hike - cut)))
    return sorted(out, key=lambda item: item[0])


def build_turning_features(market, fred, prediction_rows=None):
    """One row per QQQ session; inputs are close-of-day prices and dated FRED observations.

    `market` maps QQQ, ^VIX, ^VIX3M, HYG and IEF to dated OHLC rows; `fred` maps
    DGS2, DFEDTARU and ICSA to (date, value) pairs. Missing inputs stay None and
    never fire a signal.
    """
    qqq = sorted(market["QQQ"], key=lambda r: r["date"])
    vix, vix3m = _closes(market.get("^VIX", [])), _closes(market.get("^VIX3M", []))
    hyg, ief = _closes(market.get("HYG", [])), _closes(market.get("IEF", []))
    two, target = _fred_daily(fred.get("DGS2", [])), _fred_daily(fred.get("DFEDTARU", []))
    claims = _fred_claims(fred.get("ICSA", []))
    pm = _pm_series(prediction_rows)
    rows, ratios, credit, gaps, avg4, pms, twos = [], [], [], [], [], [], []
    claim_history, claims_seen = [], 0
    closes = []
    for row in qqq:
        day = row["date"]
        cutoff = _clock(date.fromisoformat(day))
        closes.append(row["close"])
        ratio = vix[day] / vix3m[day] if vix.get(day) and vix3m.get(day) else None
        ratios.append(ratio)
        spread = hyg[day] / ief[day] if hyg.get(day) and ief.get(day) else None
        credit.append(spread)
        recent = [value for value in credit[-63:] if value is not None]
        credit_dd = 1 - spread / max(recent) if spread is not None and recent else None
        credit_ma50 = mean(recent[-50:]) if len(recent) >= 50 else None
        two_day, two_value = _as_of(two, cutoff)
        _, target_value = _as_of(target, cutoff)
        known = two_value is not None and target_value is not None
        gap = two_value - target_value if known else None
        pm_value = None
        if pm:
            identity, pm_change = _as_of(pm, cutoff)
            if identity is not None and identity[1] > cutoff:
                pm_value = pm_change
        gaps.append(gap)
        twos.append(two_value)
        pms.append(pm_value)
        while claims_seen < len(claims) and claims[claims_seen][0] <= cutoff:
            claim_history.append(claims[claims_seen][2])
            claims_seen += 1
            if len(claim_history) >= 4:
                avg4.append(mean(claim_history[-4:]))
        claims_rise = (avg4[-1] / min(avg4[-53:-1]) - 1) if len(avg4) >= 27 else None
        window = [r for r in ratios[-63:] if r is not None]
        returns = [math.log(b / a) for a, b in zip(closes[-21:-1], closes[-20:])]
        realized = pstdev(returns) * math.sqrt(252) if len(returns) >= 19 else None
        rows.append({
            "date": day,
            "available_at": cutoff.isoformat(),
            "vix": vix.get(day),
            "vix3m": vix3m.get(day),
            "vix_ratio": ratio,
            "vix_ratio_max": {n: max((r for r in ratios[-n:] if r is not None), default=None)
                              for n in (10, 15, 20)},
            "vix_ratio_ma63": mean(window) if len(window) >= 40 else None,
            "credit_ratio": spread,
            "credit_dd63": credit_dd,
            "credit_above_ma50": (spread > credit_ma50 if spread is not None
                                  and credit_ma50 is not None else None),
            "two_year_pct": two_value,
            "two_year_date": two_day,
            "fed_upper_pct": target_value,
            "fed_gap_pp": gap,
            "fed_gap_change20_pp": (gap - gaps[-21] if len(gaps) > 20 and gap is not None
                                    and gaps[-21] is not None else None),
            # Hawkish repricing uses the 2Y alone: emergency cuts must not read as tightening.
            "two_year_change20_pp": (two_value - twos[-21] if len(twos) > 20
                                     and two_value is not None and twos[-21] is not None
                                     else None),
            "pm_expected_change_pp": pm_value,
            "pm_change20_pp": (pm_value - pms[-21] if len(pms) > 20 and pm_value is not None
                               and pms[-21] is not None else None),
            "claims_rise": claims_rise,
            "claims_above_trend": avg4[-1] > mean(avg4[-26:]) if claims_rise is not None else None,
            "qqq_realized_vol20": realized,
        })
    return {"schema_version": SCHEMA_VERSION, "rows": rows,
            "limitations": [
                "FRED values are current vintages; ICSA revisions are not ALFRED first prints.",
                "VIX/VIX3M use the 16:15 index close; the decision clock is the same 16:15.",
                "Prediction-market Fed odds exist only since 2025; earlier rows use the 2Y proxy.",
            ]}


def index_features(features):
    return {row["date"]: row for row in (features or {}).get("rows", [])}


def assess(row, q, parameters):
    """Classify one session; returns flags and the evidence used, never a probability."""
    if row is None or row.get("vix_ratio") is None:
        return {"status": "missing_neutral", "stress": False, "capitulation": False,
                "calm": False, "evidence": {}}
    p = parameters
    ratio = row["vix_ratio"]
    credit_dd = row.get("credit_dd63")
    fed_shift = row.get("two_year_change20_pp")
    pm_shift = row.get("pm_change20_pp") if p["use_prediction_markets"] else None
    reasons = []
    if ratio >= p["stress_ratio"]:
        reasons.append("vix_term_inversion")
    if credit_dd is not None and credit_dd >= p["stress_credit_dd"]:
        reasons.append("credit_drawdown")
    if pm_shift is not None:
        # Quoted next-meeting odds supersede the slower two-year proxy when present.
        if pm_shift >= p["fed_shock_pp"] / 2:
            reasons.append("hawkish_repricing_prediction_market")
    elif fed_shift is not None and fed_shift >= p["fed_shock_pp"]:
        reasons.append("hawkish_repricing_2y")
    if (row.get("claims_rise") is not None and row["claims_rise"] >= p["claims_rise"]
            and (not p["claims_trend_filter"] or row.get("claims_above_trend"))):
        reasons.append("jobless_claims_rise")
    below = q["close"] < q.get(f"sma{p['stress_confirm_sma']}", q["sma20"])
    peak = row["vix_ratio_max"].get(p["capitulation_lookback"])
    capitulation = (
        peak is not None
        and peak >= p["capitulation_ratio"]
        and ratio <= p["normalized_ratio"]
        and (credit_dd is None or credit_dd < max(p["stress_credit_dd"], 0.03) * 2)
    )
    calm = (
        ratio <= p["calm_ratio"]
        and q["close"] > q["slow_sma"]
        and q["close"] > q.get("sma50", q["slow_sma"])
        and (credit_dd is None or credit_dd < 0.02)
        and not reasons
    )
    return {
        "status": "available",
        # Leverage is allowed above the long trend, or after a volatility washout.
        "leverage_ok": not p["leverage_gate"]
        or q["close"] > q["slow_sma"]
        or (peak is not None and peak >= p["capitulation_ratio"]),
        "stress": bool(reasons) and below,
        "stress_reasons": reasons,
        "price_confirmed": below,
        "capitulation": capitulation,
        "calm": calm,
        "calm_exit": ratio >= p["calm_exit_ratio"] or bool(reasons)
        or q["close"] < q.get("sma50", q["slow_sma"]),
        "evidence": {k: row.get(k) for k in (
            "available_at", "vix_ratio", "credit_dd63", "two_year_change20_pp",
            "pm_expected_change_pp", "pm_change20_pp", "claims_rise", "claims_above_trend",
            "qqq_realized_vol20")},
        "vix_ratio_peak": peak,
    }


def calm_weight(row, parameters):
    weight = parameters["calm_tqqq"]
    target = parameters["calm_vol_target"]
    vol = (row or {}).get("qqq_realized_vol20")
    if weight and target and vol:
        # TQQQ carries roughly three times QQQ volatility on the sleeve.
        weight = min(weight, max(0.0, (target / vol - 1) / 2))
    return round(weight, 4)
