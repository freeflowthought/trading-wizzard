import math
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import OptionContract, OptionSnapshot


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(
    option_type: str,
    spot: float,
    strike: float,
    years_to_expiration: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    if spot <= 0 or strike <= 0 or years_to_expiration <= 0 or volatility <= 0:
        raise ValueError("spot, strike, time, and volatility must be positive")
    flag = option_type.lower()[0]
    sqrt_t = math.sqrt(years_to_expiration)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * years_to_expiration
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount_q = math.exp(-dividend_yield * years_to_expiration)
    discount_r = math.exp(-rate * years_to_expiration)
    gamma = discount_q * _norm_pdf(d1) / (spot * volatility * sqrt_t)
    vega = spot * discount_q * _norm_pdf(d1) * sqrt_t / 100
    if flag == "c":
        delta = discount_q * _norm_cdf(d1)
        theta = (
            -spot * discount_q * _norm_pdf(d1) * volatility / (2 * sqrt_t)
            - rate * strike * discount_r * _norm_cdf(d2)
            + dividend_yield * spot * discount_q * _norm_cdf(d1)
        ) / 365
        rho = strike * years_to_expiration * discount_r * _norm_cdf(d2) / 100
    elif flag == "p":
        delta = discount_q * (_norm_cdf(d1) - 1)
        theta = (
            -spot * discount_q * _norm_pdf(d1) * volatility / (2 * sqrt_t)
            + rate * strike * discount_r * _norm_cdf(-d2)
            - dividend_yield * spot * discount_q * _norm_cdf(-d1)
        ) / 365
        rho = -strike * years_to_expiration * discount_r * _norm_cdf(-d2) / 100
    else:
        raise ValueError("option_type must be call or put")
    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta": round(theta, 6),
        "vega": round(vega, 6),
        "rho": round(rho, 6),
    }


def years_to_expiration(expiration: str, today: Optional[date] = None) -> float:
    today = today or date.today()
    expiry = datetime.fromisoformat(expiration).date()
    return max((expiry - today).days, 1) / 365.0


def fetch_yfinance_option_contracts(
    ticker: str,
    spot: float,
    max_contracts: int = 20,
) -> Tuple[List[OptionContract], Optional[str], List[str]]:
    warnings: List[str] = []
    try:
        import yfinance as yf
    except Exception as exc:
        return [], None, [f"yfinance unavailable for options {ticker}: {exc}"]

    try:
        instrument = yf.Ticker(ticker)
        expirations = list(getattr(instrument, "options", []) or [])
        if not expirations:
            return [], None, [f"No option expirations returned for {ticker}."]
        expiration = expirations[0]
        chain = instrument.option_chain(expiration)
    except Exception as exc:
        return [], None, [f"Option chain fetch failed for {ticker}: {exc}"]

    contracts: List[OptionContract] = []
    for option_type, frame in [("call", chain.calls), ("put", chain.puts)]:
        if frame is None or frame.empty:
            continue
        frame = frame.assign(distance=(frame["strike"] - spot).abs()).sort_values("distance")
        for _, row in frame.head(max_contracts // 2).iterrows():
            contracts.append(
                OptionContract(
                    symbol=str(row.get("contractSymbol", "")),
                    option_type=option_type,
                    strike=float(row.get("strike", 0.0)),
                    expiration=expiration,
                    bid=float(row.get("bid", 0.0)),
                    ask=float(row.get("ask", 0.0)),
                    last=float(row.get("lastPrice", 0.0)),
                    volume=float(row.get("volume", 0.0) or 0.0),
                    open_interest=float(row.get("openInterest", 0.0) or 0.0),
                    implied_volatility=float(row.get("impliedVolatility", 0.0) or 0.0),
                )
            )
    return contracts, expiration, warnings


def enrich_contract_greeks(
    contracts: Iterable[OptionContract],
    spot: float,
    risk_free_rate: float = 0.045,
) -> List[OptionContract]:
    enriched: List[OptionContract] = []
    for contract in contracts:
        volatility = contract.implied_volatility or 0.30
        try:
            greeks = black_scholes_greeks(
                contract.option_type,
                spot,
                contract.strike,
                years_to_expiration(contract.expiration),
                risk_free_rate,
                volatility,
            )
            for key, value in greeks.items():
                setattr(contract, key, value)
        except ValueError:
            pass
        enriched.append(contract)
    return enriched


def summarize_options(
    ticker: str,
    spot: float,
    contracts: List[OptionContract],
    expiration: Optional[str],
) -> OptionSnapshot:
    warnings: List[str] = []
    if not contracts:
        warnings.append("No option chain available; options context is unavailable/stale.")
        return OptionSnapshot(
            ticker=ticker,
            expiration=expiration,
            contracts=[],
            greeks={"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None},
            skew_summary={"call_iv_atm": None, "put_iv_atm": None, "put_call_iv_spread": None},
            payoff_scenarios=[],
            warnings=warnings,
        )

    enriched = enrich_contract_greeks(contracts, spot)
    calls = [contract for contract in enriched if contract.option_type == "call"]
    puts = [contract for contract in enriched if contract.option_type == "put"]
    nearest_call = min(calls, key=lambda row: abs(row.strike - spot)) if calls else None
    nearest_put = min(puts, key=lambda row: abs(row.strike - spot)) if puts else None
    call_iv = nearest_call.implied_volatility if nearest_call else None
    put_iv = nearest_put.implied_volatility if nearest_put else None
    aggregate = {}
    for greek in ["delta", "gamma", "theta", "vega", "rho"]:
        values = [
            getattr(contract, greek)
            for contract in enriched
            if getattr(contract, greek) is not None
        ]
        aggregate[greek] = round(sum(values) / len(values), 6) if values else None
    scenarios = []
    if nearest_call:
        premium = nearest_call.ask or nearest_call.last or nearest_call.bid or 0.0
        for move in [-0.05, 0.0, 0.05]:
            terminal = spot * (1 + move)
            scenarios.append(
                {
                    "strategy": "long_atm_call",
                    "underlying_move_pct": round(move * 100, 2),
                    "estimated_expiry_pl": round(
                        max(terminal - nearest_call.strike, 0.0) - premium,
                        2,
                    ),
                }
            )
    return OptionSnapshot(
        ticker=ticker,
        expiration=expiration,
        contracts=[contract.to_dict() for contract in enriched],
        greeks=aggregate,
        skew_summary={
            "call_iv_atm": call_iv,
            "put_iv_atm": put_iv,
            "put_call_iv_spread": round((put_iv or 0) - (call_iv or 0), 6)
            if call_iv is not None and put_iv is not None
            else None,
        },
        payoff_scenarios=scenarios,
        warnings=warnings,
    )


def build_option_snapshot(ticker: str, spot: float, live: bool = False) -> OptionSnapshot:
    warnings: List[str] = []
    contracts: List[OptionContract] = []
    expiration = None
    if live:
        contracts, expiration, warnings = fetch_yfinance_option_contracts(ticker, spot)
    snapshot = summarize_options(ticker, spot, contracts, expiration)
    snapshot.warnings.extend(warnings)
    if not live:
        snapshot.warnings.append(
            "Live options ingest skipped. Pass --live to attempt yfinance option chain fetch."
        )
    return snapshot


def build_option_snapshots(
    market_data: Dict[str, List[Dict[str, Any]]],
    live: bool = False,
) -> Dict[str, Dict[str, Any]]:
    snapshots: Dict[str, Dict[str, Any]] = {}
    for ticker, rows in market_data.items():
        spot = float(rows[-1]["close"]) if rows else 0.0
        snapshots[ticker] = build_option_snapshot(ticker, spot, live=live).to_dict()
    return snapshots
