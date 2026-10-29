"""Cash-flow quality screen — the June timeline's sharpest new filter.

His diagnosis of the Mag-7 drawdown: the AI leaders are burning cash and
financing the burn (equity issuance, borrowing) to buy materials, while
the resilient name (AAPL) sits on cash. His trades follow the screen:

- issuance-funded burner -> sell signal (he exited NVDA on "needs
  financing to keep burning", trimmed ORCL after its share issuance)
- fortress balance sheet -> the name that holds up in a drawdown
- when the big customers' cash runs out, the suppliers (MU, SNDK) are
  next — this screen is the fundamental half of that cycle call

Inputs are plain dicts so the scoring is testable offline; the yfinance
fetcher is best-effort and only runs with ``live=True``.
"""

import math
from datetime import date
from typing import Any, Dict, List, Optional


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def score_cash_quality(fin: Dict[str, Any]) -> Dict[str, Any]:
    """Score one company's cash quality from raw statement values.

    Expected keys (any may be missing): free_cash_flow,
    operating_cash_flow, capex, cash_and_equivalents, total_debt,
    shares_change_pct (YoY change in share count, %).
    """
    fcf = _num(fin.get("free_cash_flow"))
    ocf = _num(fin.get("operating_cash_flow"))
    capex = _num(fin.get("capex"))
    cash = _num(fin.get("cash_and_equivalents"))
    debt = _num(fin.get("total_debt"))
    dilution = _num(fin.get("shares_change_pct"))

    flags: List[str] = []
    score = 50

    if fcf is not None:
        if fcf < 0:
            score -= 25
            flags.append("negative_free_cash_flow")
        else:
            score += 15
    if ocf is not None and capex is not None and ocf > 0:
        intensity = abs(capex) / ocf
        if intensity > 1.0:
            score -= 20
            flags.append("capex_exceeds_operating_cash_flow")
        elif intensity > 0.6:
            score -= 8
            flags.append("heavy_capex")
        else:
            score += 5
    if dilution is not None:
        if dilution > 2.0:
            score -= 20
            flags.append("share_issuance_dilution")
        elif dilution > 0.5:
            score -= 8
            flags.append("mild_dilution")
        elif dilution < -0.5:
            score += 10
            flags.append("buyback_shrinking_share_count")
    net_cash = None
    if cash is not None and debt is not None:
        net_cash = cash - debt
        if net_cash > 0:
            score += 15
            flags.append("net_cash_fortress")
        elif debt > 0 and cash is not None and debt > 2.0 * max(cash, 1.0):
            score -= 10
            flags.append("debt_heavy")

    required = {
        "free_cash_flow": fcf, "operating_cash_flow": ocf, "capex": capex,
        "cash_and_equivalents": cash, "total_debt": debt,
    }
    missing = [key for key, value in required.items() if value is None]
    warnings = ["Incomplete statements; cash-flow quality cannot be classified: "
                + ", ".join(missing)] if missing else []
    if dilution is None:
        warnings.append("Comparable prior-year share count unavailable; dilution is unknown.")
    score = max(0, min(100, score))
    if score >= 75:
        label = "fortress"
    elif score >= 55:
        label = "healthy"
    elif fcf is not None and fcf >= 0:
        label = "pressured"
    elif score >= 35:
        label = "burner"
    else:
        label = "distress"
    return {
        "score": None if missing else score,
        "label": "unknown" if missing else label,
        "warnings": warnings,
        "missing_fields": missing,
        "flags": flags,
        "free_cash_flow": fcf,
        "operating_cash_flow": ocf,
        "capex": capex,
        "net_cash": net_cash,
        "shares_change_pct": dilution,
    }


def _column_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _dated_columns(frame: Any) -> List[Any]:
    if frame is None:
        return []
    try:
        return sorted(
            [column for column in frame.columns if _column_date(column) is not None],
            key=lambda column: _column_date(column), reverse=True,
        )
    except (AttributeError, TypeError):
        return []


def _frame_value(frame: Any, labels: List[str], column: Any = None) -> Optional[float]:
    """Read one reporting period; never silently substitute an older year's value."""
    columns = _dated_columns(frame)
    if not columns:
        return None
    column = columns[0] if column is None else column
    try:
        for label in labels:
            if label in frame.index:
                number = _num(frame.loc[label, column])
                if number is not None:
                    return number
    except (KeyError, TypeError, AttributeError):
        return None
    return None


def fetch_yfinance_fundamentals(ticker: str) -> Dict[str, Any]:
    """Best-effort annual statements, with every scored field from the same period."""
    try:
        import yfinance as yf

        instrument = yf.Ticker(ticker)
        cashflow = instrument.cashflow
        balance = instrument.balance_sheet
    except Exception:
        return {}
    cf_columns = _dated_columns(cashflow)
    bs_columns = _dated_columns(balance)
    if not cf_columns:
        return {}
    period = cf_columns[0]
    period_date = _column_date(period)
    balance_period = next((c for c in bs_columns if _column_date(c) == period_date), None)
    fin: Dict[str, Any] = {
        "statement_date": period_date.isoformat(),
        "free_cash_flow": _frame_value(cashflow, ["Free Cash Flow"], period),
        "operating_cash_flow": _frame_value(
            cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"], period
        ),
        "capex": _frame_value(cashflow, ["Capital Expenditure"], period),
    }
    if balance_period is not None:
        fin["cash_and_equivalents"] = _frame_value(
            balance, ["Cash And Cash Equivalents",
                      "Cash Cash Equivalents And Short Term Investments"], balance_period
        )
        # Total liabilities also includes non-debt obligations and is not a substitute.
        fin["total_debt"] = _frame_value(balance, ["Total Debt"], balance_period)
        previous = next((c for c in bs_columns
                         if 330 <= (period_date - _column_date(c)).days <= 400), None)
        share_labels = ["Ordinary Shares Number", "Share Issued"]
        current_shares = _frame_value(balance, share_labels, balance_period)
        prior_shares = (
            _frame_value(balance, share_labels, previous) if previous is not None else None
        )
        if current_shares is not None and prior_shares is not None and prior_shares > 0:
            fin["shares_change_pct"] = round((current_shares / prior_shares - 1.0) * 100, 3)
            fin["shares_comparison_date"] = _column_date(previous).isoformat()
    return {key: value for key, value in fin.items() if value is not None}


def build_fundamental_snapshots(
    tickers: List[str],
    live: bool = False,
) -> Dict[str, Dict[str, Any]]:
    snapshots: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        if not live:
            snapshots[ticker] = {
                "label": "unknown",
                "score": None,
                "flags": [],
                "warnings": ["Live fundamentals skipped. Pass --live to fetch statements."],
            }
            continue
        fin = fetch_yfinance_fundamentals(ticker)
        if not fin:
            snapshots[ticker] = {
                "label": "unknown",
                "score": None,
                "flags": [],
                "warnings": [f"No statement data returned for {ticker}."],
            }
            continue
        result = score_cash_quality(fin)
        result["statement_date"] = fin.get("statement_date")
        result["shares_comparison_date"] = fin.get("shares_comparison_date")
        snapshots[ticker] = result
    return snapshots
