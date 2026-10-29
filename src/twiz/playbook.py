"""The Shunge playbook layer.

Distilled from the 2026-08-25 → 09-03 prediction timeline (see the
extraction notes kept with the goal). His model, as rules:

1. Index path, not point guesses — anchor to the recent peak and lay out
   drawdown bands: ~-2.5% (shallow low), ~-5% (normal correction),
   ~-8% (his September low zone), ~-12% (systemic break). Targets are the
   next round numbers above (8000 from a 7830 peak).
2. The 10Y Treasury is the master switch — approaching 5% is the crash
   zone, rolling back toward 4.5% is the rally green light. Credit
   spreads (HY OAS) widening is the second tripwire.
3. Supports/targets are prior pivots that coincide with round numbers.
4. Earnings are traded around the options-implied move ("系统目测
   6-7%"), never with fresh options into the print.
5. Crowded puts near a round strike are squeeze fuel — no shorting into
   elections or into crowded put walls.
6. Position management is a farmer's ladder: batch 1 at the first
   support, batch 2 at the deep band, exits trimmed in stages at targets.
   Sell-put strikes belong just under strong support.
"""

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .events import next_event_for, political_context, seasonal_context

DRAWDOWN_BANDS_PCT = [2.5, 5.0, 8.0, 12.0, 20.0]
BAND_LABELS = {
    2.5: "shallow_low_zone",
    5.0: "normal_correction_zone",
    8.0: "deep_low_zone",
    12.0: "systemic_break_zone",
    20.0: "bear_market_zone",
}
INDEX_TICKERS = {"SPY", "QQQ", "SPX", "^GSPC", "DIA", "IWM"}
SEMI_CYCLE_TICKERS = {"SOXL", "SMH", "SOXX", "SOXQ", "MU", "INTC", "QCOM", "NVDA", "AMD", "SNDK"}
HARDWARE_LEADERS = ("MU", "INTC", "QCOM", "NVDA", "AMD", "SNDK")
# His geopolitics chain: de-escalation -> oil down -> cruise/airline up.
OIL_DOWN_BENEFICIARIES = {"RCL", "CCL", "NCLH", "UAL", "DAL", "AAL", "LUV"}


def round_step(price: float) -> float:
    if price >= 1000:
        return 100.0
    if price >= 250:
        return 25.0
    if price >= 100:
        return 10.0
    if price >= 25:
        return 5.0
    if price >= 10:
        return 1.0
    return 0.5


def round_levels(price: float, span_pct: float = 0.30) -> List[Dict[str, Any]]:
    """Psychological round-number levels within +/- span of price."""
    if price <= 0:
        return []
    step = round_step(price)
    low = price * (1.0 - span_pct)
    high = price * (1.0 + span_pct)
    first = math.floor(low / step) * step
    levels = []
    value = first
    while value <= high:
        if value > 0:
            levels.append(
                {
                    "price": round(value, 2),
                    "touches": 1,
                    "source": "round_number",
                }
            )
        value += step
    return levels


def nearest_round(price: float, above: bool) -> float:
    step = round_step(price)
    if above:
        return round((math.floor(price / step) + 1) * step, 2)
    return round(math.floor(price / step) * step, 2)


def drawdown_bands(rows: List[Dict[str, Any]], lookback: int = 252) -> Dict[str, Any]:
    if not rows:
        return {"peak": None, "drawdown_pct": None, "bands": []}
    sample = rows[-lookback:]
    peak = max(float(row["high"]) for row in sample)
    close = float(rows[-1]["close"])
    bands = []
    for pct in DRAWDOWN_BANDS_PCT:
        level = peak * (1.0 - pct / 100.0)
        bands.append(
            {
                "drawdown_pct": pct,
                "price": round(level, 2),
                "label": BAND_LABELS[pct],
                "status": "above_price" if level >= close else "below_price",
            }
        )
    return {
        "peak": round(peak, 2),
        "drawdown_pct": round((close / peak - 1.0) * 100, 2) if peak else None,
        "bands": bands,
    }


def macro_gate(
    macro_rows: List[Dict[str, Any]], as_of: Optional[date] = None
) -> Dict[str, Any]:
    """Shunge's master switch, from FRED rows shaped like ingest/macro.py
    output (one row per series, optional ``history`` list of observations).
    """
    as_of = as_of or date.today()
    series: Dict[str, Dict[str, Any]] = {}
    for row in macro_rows or []:
        series[str(row.get("series_id", ""))] = row
    # Treasury fallback retains its native identity and complete provider history.
    if "DGS10" not in series and "UST10Y" in series:
        series["DGS10"] = series["UST10Y"]

    observations: Dict[str, Dict[str, Any]] = {}
    unavailable: List[str] = []

    def latest(series_id: str) -> Optional[float]:
        row = series.get(series_id) or {}
        candidates = [row] + list(reversed(row.get("history") or []))
        for candidate in candidates:
            try:
                value = float(candidate.get("value"))
                when = date.fromisoformat(str(candidate.get("date"))[:10])
                if math.isfinite(value) and 0 <= (as_of - when).days <= 7:
                    observations[series_id] = {
                        "date": candidate.get("date"), "fallback": candidate is not row,
                    }
                    return value
            except (TypeError, ValueError, AttributeError):
                continue
        if row:
            unavailable.append(
                f"{series_id}: no valid dated observation within seven days of {as_of}."
            )
        return None

    def change(series_id: str, lookback: int) -> Optional[float]:
        row = series.get(series_id)
        history = (row or {}).get("history") or []
        values = []
        for item in history:
            try:
                value = float(item.get("value"))
                when = date.fromisoformat(str(item.get("date"))[:10])
                if math.isfinite(value) and when <= as_of:
                    values.append(value)
            except (TypeError, ValueError, AttributeError):
                continue
        if len(values) <= lookback:
            return None
        # history is stored oldest -> newest
        return round(values[-1] - values[-1 - lookback], 3)

    dgs10 = latest("DGS10")
    hy = latest("BAMLH0A0HYM2")
    dgs10_change_20 = change("DGS10", 20)
    hy_change_20 = change("BAMLH0A0HYM2", 20)

    reasons: List[str] = list(unavailable)
    for series_id, observation in observations.items():
        if observation["fallback"]:
            reasons.append(
                f"{series_id}: latest observation missing; using last numeric "
                f"observation dated {observation['date']}."
            )
    state = "unknown"
    if dgs10 is not None:
        state = "risk_on"
        if dgs10 >= 5.0:
            state = "risk_off"
            reasons.append(
                f"10Y at {dgs10:.2f}% — at/above the configured 5% risk-off threshold."
            )
        elif dgs10 >= 4.8:
            state = "caution"
            reasons.append(f"10Y at {dgs10:.2f}% — approaching the 5% danger line.")
        if dgs10_change_20 is not None and dgs10_change_20 >= 0.25:
            if state == "risk_on":
                state = "caution"
            reasons.append(
                f"10Y up {dgs10_change_20:+.2f} pts in ~20 sessions — rising fast toward the line."
            )
        if dgs10 <= 4.5 and dgs10_change_20 is not None and dgs10_change_20 <= 0:
            reasons.append(f"10Y at {dgs10:.2f}% and not rising — rally green-light zone.")
    if hy is not None:
        if hy >= 5.0:
            state = "risk_off"
            reasons.append(f"HY spreads at {hy:.2f}% — credit is signalling stress.")
        elif hy_change_20 is not None and hy_change_20 >= 0.5:
            if state == "risk_on":
                state = "caution"
            reasons.append(f"HY spreads widened {hy_change_20:+.2f} pts in ~20 sessions.")
    if state == "unknown":
        reasons.append("No FRED macro data available; macro gate cannot be evaluated.")

    return {
        "state": state,
        "dgs10": dgs10,
        "dgs10_date": observations.get("DGS10", {}).get("date"),
        "dgs10_change_20": dgs10_change_20,
        "hy_oas": hy,
        "hy_oas_date": observations.get("BAMLH0A0HYM2", {}).get("date"),
        "hy_change_20": hy_change_20,
        "reasons": reasons,
    }


def leadership_snapshot(
    market_data: Dict[str, List[Dict[str, Any]]],
    leaders: tuple = HARDWARE_LEADERS,
    benchmark: str = "SPY",
    window: int = 20,
) -> Dict[str, Any]:
    """Hardware-leader health — his SOXL regime switch.

    His rule: as long as the hardware leaders (MU first) do not break,
    SOXL cannot really fall and the sell-put/sell-call income mode is
    "no-brainer" mode. When the leaders start blowing up (MU earnings
    miss = his named trigger), the semiconductor/storage cycle is over
    and the whole playbook rotates — to software, in his telling.
    """
    def window_return(rows: List[Dict[str, Any]]) -> Optional[float]:
        closes = [float(r["close"]) for r in rows if float(r["close"]) > 0]
        if len(closes) <= window:
            return None
        return round((closes[-1] / closes[-1 - window] - 1.0) * 100.0, 2)

    def below_ma(rows: List[Dict[str, Any]], ma_window: int = 20) -> Optional[bool]:
        closes = [float(r["close"]) for r in rows if float(r["close"]) > 0]
        if len(closes) < ma_window:
            return None
        return closes[-1] < sum(closes[-ma_window:]) / ma_window

    bench_rows = market_data.get(benchmark, [])
    bench_ret = window_return(bench_rows) if bench_rows else None
    per_leader: Dict[str, Any] = {}
    weakening = 0
    counted = 0
    for ticker in leaders:
        rows = market_data.get(ticker)
        if not rows:
            continue
        ret = window_return(rows)
        under = below_ma(rows)
        eligible = ret is not None and under is not None
        is_weak = bool(ret < -3.0 or under) if eligible else None
        if eligible:
            counted += 1
            weakening += 1 if is_weak else 0
        per_leader[ticker] = {
            "return_20d_pct": ret,
            "vs_benchmark_pct": (
                round(ret - bench_ret, 2) if ret is not None and bench_ret is not None else None
            ),
            "below_20d_mean": under,
            "weakening": is_weak,
            "eligible": eligible,
        }
    if counted < 2 or bench_ret is None:
        state = "unknown"
    elif weakening == 0:
        state = "intact"
    elif weakening >= max(2, counted // 2):
        state = "broken"
    else:
        state = "weakening"
    return {
        "state": state,
        "benchmark": benchmark,
        "benchmark_return_20d_pct": bench_ret,
        "eligible_leaders": counted,
        "required_history": window + 1,
        "leaders": per_leader,
        "rule": (
            "Use leader price strength to confirm semiconductor exposure. When leaders "
            "weaken, wait for recovery before increasing sector leverage; this price "
            "screen does not establish an industry-cycle turning point."
        ),
    }


def _levels_by_side(technical: Dict[str, Any], close: float) -> Dict[str, List[Dict[str, Any]]]:
    supports, resistances = [], []
    for level in technical.get("levels", []) or []:
        price = float(level.get("price", 0))
        row = {
            "price": price,
            "touches": int(level.get("touches", 1)),
            "distance_pct": round((price / close - 1.0) * 100, 2) if close else None,
            "sources": level.get("sources", []),
        }
        (supports if price <= close else resistances).append(row)
    supports.sort(key=lambda r: -r["price"])
    resistances.sort(key=lambda r: r["price"])
    return {"supports": supports, "resistances": resistances}


def build_playbook(
    ticker: str,
    rows: List[Dict[str, Any]],
    technical: Dict[str, Any],
    options_snapshot: Optional[Dict[str, Any]] = None,
    macro_rows: Optional[List[Dict[str, Any]]] = None,
    gate: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    as_of: Optional[date] = None,
    fundamentals: Optional[Dict[str, Any]] = None,
    leadership: Optional[Dict[str, Any]] = None,
    political: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    as_of = as_of or date.today()
    close = float(rows[-1]["close"]) if rows else 0.0
    bands_info = drawdown_bands(rows)
    sides = _levels_by_side(technical, close)
    gate = gate if gate is not None else macro_gate(macro_rows or [], as_of=as_of)
    options_current = False
    if options_snapshot and rows:
        try:
            fetched = datetime.fromisoformat(
                str(options_snapshot.get("fetched_at")).replace("Z", "+00:00")
            )
            underlying_spot = float(options_snapshot.get("underlying_spot"))
            underlying_day = date.fromisoformat(str(options_snapshot.get("underlying_date"))[:10])
            price_day = date.fromisoformat(str(rows[-1].get("date"))[:10])
            options_current = bool(
                fetched.tzinfo is not None
                and fetched.astimezone(ZoneInfo("America/New_York")).date() == as_of
                and underlying_day == price_day and underlying_day <= as_of
                and math.isfinite(underlying_spot) and underlying_spot > 0
                and math.isfinite(close) and abs(underlying_spot-close) <= 0.011
            )
        except (TypeError, ValueError):
            options_current = False
    skew = (options_snapshot or {}).get("skew_summary", {}) if options_current else {}
    skew = skew or {}
    expected_move = skew.get("expected_move_pct")
    pc_ratio = skew.get("put_call_oi_ratio")
    put_wall = skew.get("put_wall")

    # --- staged entries (the farmer's ladder) ---
    base_fractions = [0.40, 0.35, 0.25]
    scale = {"risk_off": 0.5, "caution": 0.75}.get(gate.get("state"), 1.0)
    entries: List[Dict[str, Any]] = []
    entry_prices: List[float] = []
    if sides["supports"]:
        entry_prices.append(sides["supports"][0]["price"])
    deep_band = next(
        (b["price"] for b in bands_info["bands"] if b["drawdown_pct"] == 8.0), None
    )
    if deep_band and (not entry_prices or abs(deep_band - entry_prices[0]) / close > 0.01):
        entry_prices.append(deep_band)
    if len(sides["supports"]) > 1:
        entry_prices.append(sides["supports"][1]["price"])
    entry_prices = sorted({round(p, 2) for p in entry_prices if p < close}, reverse=True)
    for index, price in enumerate(entry_prices[:3] if gate.get("state") != "unknown" else []):
        fraction = base_fractions[index] * (scale if index == 0 else 1.0)
        entries.append(
            {
                "batch": index + 1,
                "price": price,
                "fraction": round(fraction, 2),
                "distance_pct": round((price / close - 1.0) * 100, 2),
            }
        )

    # --- targets: clustered resistance first, then the round number ---
    targets = [row["price"] for row in sides["resistances"][:2]]
    round_above = nearest_round(close, above=True)
    if close > 0 and round_above > close and all(
        abs(round_above - t) / close > 0.01 for t in targets
    ):
        targets.append(round_above)
    targets = sorted(set(targets))[:3]

    # --- sell-put zone: just under the strongest nearby support ---
    sell_put_strike = None
    if sides["supports"] and gate.get("state") != "unknown":
        anchor = sides["supports"][0]["price"]
        step = round_step(anchor)
        sell_put_strike = round(math.floor((anchor * 0.99) / step) * step, 2)
        if sell_put_strike <= 0:
            sell_put_strike = None

    # --- event overlay ---
    upcoming = next_event_for(events or [], ticker=ticker)
    event_note = None
    if upcoming:
        event_note = {
            "date": upcoming["date"],
            "name": upcoming["name"],
            "days_away": upcoming.get("days_away"),
            "rule": upcoming.get("playbook_rule", ""),
            "precision": upcoming.get("precision", "unknown"),
        }
        try:
            option_expiry = date.fromisoformat(str((options_snapshot or {}).get("expiration")))
            earnings_day = date.fromisoformat(str(upcoming.get("date")))
            covers_earnings = option_expiry >= earnings_day and option_expiry >= as_of
        except ValueError:
            covers_earnings = False
        if upcoming.get("kind") == "earnings" and expected_move is not None and covers_earnings:
            event_note["expected_move_pct"] = expected_move
            event_note["rule"] = (
                f"Earnings in {upcoming.get('days_away')}d; the {option_expiry.isoformat()} "
                f"ATM straddle costs {expected_move:.1f}% of spot through expiry "
                "(including this event and other risks). " + event_note["rule"]
            )

    political = political if political is not None else political_context(as_of)

    # --- trend gate (index names): the 7440-7460 style flip level ---
    trend_gate = None
    if close > 0 and (ticker.upper() in INDEX_TICKERS or ticker.upper() == "SOXL"):
        breakout = sides["resistances"][0]["price"] if sides["resistances"] else round_above
        fail_to = sides["supports"][0]["price"] if sides["supports"] else None
        trend_gate = {
            "breakout_level": breakout,
            "failure_target": fail_to,
            "state": technical.get("regime", "unknown"),
            "rule": (
                f"Current technical regime: {technical.get('regime', 'unknown')}. "
                f"Next upside confirmation level is {breakout:g}; "
                f"a failed breakout would put support at {fail_to:g} in focus."
            ) if breakout and fail_to else "",
        }

    notes: List[str] = []
    if options_snapshot and not options_current:
        notes.append(
            "Options snapshot date/underlying price not verified; implied move and OI withheld."
        )
    if gate.get("state") == "unknown":
        notes.append("Macro gate is unknown; staged position sizes and sell-put zone withheld.")
    if trend_gate and trend_gate["rule"]:
        notes.append(f"Trend gate: {trend_gate['rule']}")
    if fundamentals:
        label = fundamentals.get("label")
        if label in ("burner", "distress", "pressured"):
            notes.append(
                "Cash-flow screen: "
                f"{label} (score {fundamentals.get('score')}, "
                f"flags: {', '.join(fundamentals.get('flags', [])) or 'n/a'}). "
                "Review these statement risks before adding exposure; "
                "the heuristic score alone is not a sell signal."
            )
        elif label == "fortress":
            notes.append(
                "Cash-flow screen: high cash-quality score; this does not establish "
                "a price floor or guarantee resilience in a drawdown."
            )
    if (
        leadership
        and ticker.upper() in SEMI_CYCLE_TICKERS
        and leadership.get("state") in ("weakening", "broken")
    ):
        notes.append(
            f"Leadership rotation: hardware leaders are {leadership['state']} "
            "on the 20-session price screen; require sector recovery before increasing "
            "semiconductor leverage. This screen does not date the end of an industry cycle."
        )
    if ticker.upper() in OIL_DOWN_BENEFICIARIES and any(
        row.get("kind") == "geopolitics" for row in (events or [])
    ):
        notes.append(
            "Geopolitics event in the calendar: verify its outcome and the oil-price "
            "reaction before assuming a benefit for transport stocks."
        )
    if ticker.upper() in INDEX_TICKERS and political.get("phase") == "election_runup":
        notes.append(
            f"Political calendar: election in {political.get('days_away')}d. "
            "Election proximity alone does not establish market direction."
        )
    if gate.get("state") == "risk_off":
        notes.append("Macro gate is risk-off: halve batch 1 and wait for the 10Y to stabilise.")
    if pc_ratio is not None and pc_ratio >= 1.2:
        notes.append(
            f"Put/call OI ratio {pc_ratio:.2f} for this expiry. Open interest includes "
            "hedges and spreads; this ratio alone does not establish a squeeze or direction."
        )
    if put_wall:
        notes.append(f"Largest put open interest sits at strike {put_wall:g} (the put wall).")
    seasonal = seasonal_context(as_of)
    if seasonal["risk"] in ("high", "medium"):
        notes.append(f"Seasonal: {seasonal['note']}")
    if ticker.upper() in INDEX_TICKERS and bands_info["drawdown_pct"] is not None:
        notes.append(
            f"Index is {bands_info['drawdown_pct']:+.1f}% from its peak "
            f"({bands_info['peak']:g}); drawdown bands are reference prices, "
            "not independent buy signals."
        )

    return {
        "ticker": ticker.upper(),
        "as_of": as_of.isoformat(),
        "close": round(close, 2),
        "peak": bands_info["peak"],
        "drawdown_pct": bands_info["drawdown_pct"],
        "drawdown_bands": bands_info["bands"],
        "round_below": nearest_round(close, above=False) if close else None,
        "round_above": round_above if close else None,
        "supports": sides["supports"][:4],
        "targets": targets,
        "staged_entries": entries,
        "sell_put_strike": sell_put_strike,
        "expected_move_pct": expected_move,
        "put_call_oi_ratio": pc_ratio,
        "put_wall": put_wall,
        "call_wall": skew.get("call_wall"),
        "macro_gate": gate,
        "seasonal": seasonal,
        "political": political,
        "trend_gate": trend_gate,
        "cash_quality": (
            {
                "label": fundamentals.get("label"),
                "score": fundamentals.get("score"),
                "flags": fundamentals.get("flags", []),
            }
            if fundamentals
            else None
        ),
        "leadership_state": leadership.get("state") if leadership else None,
        "next_event": event_note,
        "notes": notes,
    }


def build_playbooks(
    market_data: Dict[str, List[Dict[str, Any]]],
    technical: Dict[str, Dict[str, Any]],
    options_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
    macro_rows: Optional[List[Dict[str, Any]]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    as_of: Optional[date] = None,
    fundamentals_map: Optional[Dict[str, Dict[str, Any]]] = None,
    leadership: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    gate = macro_gate(macro_rows or [], as_of=as_of)
    options_snapshots = options_snapshots or {}
    fundamentals_map = fundamentals_map or {}
    political = political_context(as_of) if as_of else None
    return {
        ticker: build_playbook(
            ticker,
            rows,
            technical.get(ticker, {}),
            options_snapshot=options_snapshots.get(ticker),
            gate=gate,
            events=events,
            as_of=as_of,
            fundamentals=fundamentals_map.get(ticker),
            leadership=leadership,
            political=political,
        )
        for ticker, rows in market_data.items()
        if rows
    }
