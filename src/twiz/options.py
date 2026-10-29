import math
from copy import deepcopy
from datetime import date, datetime, timezone
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
    if (not all(math.isfinite(v) for v in (
        spot, strike, years_to_expiration, rate, volatility, dividend_yield
    )) or spot <= 0 or strike <= 0 or years_to_expiration <= 0 or volatility <= 0):
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


def _valuation_day(value: Optional[date] = None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("valuation_date must be a calendar date")


def years_to_expiration(
    expiration: str, today: Optional[date] = None, *, valuation_date: Optional[date] = None,
) -> float:
    if today is not None and valuation_date is not None and _valuation_day(today) != (
        _valuation_day(valuation_date)
    ):
        raise ValueError("today and valuation_date conflict")
    valuation = _valuation_day(valuation_date if valuation_date is not None else today)
    expiry = datetime.fromisoformat(expiration).date()
    return max((expiry - valuation).days, 1) / 365.0


def _trade_timestamp(value: Any) -> Optional[str]:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).isoformat() if parsed.tzinfo else None
    except (TypeError, ValueError):
        return None


def _finite_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def fetch_yfinance_option_contracts(
    ticker: str,
    spot: float,
    max_contracts: Optional[int] = None,
    *, valuation_date: Optional[date] = None,
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
        if valuation_date is not None:
            future = []
            for candidate in expirations:
                try:
                    if date.fromisoformat(candidate) > valuation_date:
                        future.append(candidate)
                except (TypeError, ValueError):
                    continue
            if not future:
                return [], None, [f"No option expiry after daily valuation {valuation_date} "
                                  f"returned for {ticker}."]
            expiration = min(future)
        else:
            expiration = expirations[0]
        chain = instrument.option_chain(expiration)
    except Exception as exc:
        return [], None, [f"Option chain fetch failed for {ticker}: {exc}"]

    contracts: List[OptionContract] = []
    for option_type, frame in [("call", chain.calls), ("put", chain.puts)]:
        if frame is None or frame.empty:
            continue
        # The default full expiry chain supports positioning totals and walls.
        if max_contracts is not None:
            frame = frame.head(max(0, max_contracts // 2))
        for _, row in frame.iterrows():
            strike = _finite_number(row.get("strike"))
            if not strike:
                if "Skipped option contracts with invalid strikes." not in warnings:
                    warnings.append("Skipped option contracts with invalid strikes.")
                continue
            contracts.append(
                OptionContract(
                    symbol=str(row.get("contractSymbol", "")),
                    option_type=option_type,
                    strike=strike,
                    expiration=expiration,
                    bid=_finite_number(row.get("bid"), None),
                    ask=_finite_number(row.get("ask"), None),
                    last=_finite_number(row.get("lastPrice"), None),
                    volume=_finite_number(row.get("volume")),
                    open_interest=_finite_number(row.get("openInterest")),
                    implied_volatility=_finite_number(row.get("impliedVolatility"), None),
                    last_trade_at=_trade_timestamp(row.get("lastTradeDate")),
                )
            )
    return contracts, expiration, warnings


def _nonnegative(value: Optional[float]) -> float:
    if value is None or not math.isfinite(value) or value < 0:
        return 0.0
    return value


def _contract_mid(contract: OptionContract) -> Optional[float]:
    bid, ask = contract.bid, contract.ask
    if (bid is None or ask is None or not math.isfinite(bid) or not math.isfinite(ask)
            or bid < 0 or ask <= 0 or bid > ask):
        return None
    return (bid + ask) / 2.0


def compute_event_context(
    contracts: List[OptionContract], spot: float
) -> Dict[str, Optional[float]]:
    """Positioning for the supplied chain; an ATM straddle requires one strike/expiry."""
    empty = {
        "expected_move_pct": None, "put_call_oi_ratio": None,
        "put_wall": None, "call_wall": None,
    }
    if not contracts or not math.isfinite(spot) or spot <= 0:
        return empty
    calls = [c for c in contracts if c.option_type == "call"]
    puts = [c for c in contracts if c.option_type == "put"]
    put_by_key = {(c.expiration, c.strike): c for c in puts}
    pairs = [(c, put_by_key[(c.expiration, c.strike)]) for c in calls
             if (c.expiration, c.strike) in put_by_key]
    expected_move = None
    if pairs:
        atm_call, atm_put = min(pairs, key=lambda pair: abs(pair[0].strike - spot))
        call_mid, put_mid = _contract_mid(atm_call), _contract_mid(atm_put)
        if call_mid is not None and put_mid is not None and call_mid + put_mid > 0:
            expected_move = round((call_mid + put_mid) / spot * 100.0, 2)
    call_oi = sum(_nonnegative(c.open_interest) for c in calls)
    put_oi = sum(_nonnegative(c.open_interest) for c in puts)

    def wall(side: List[OptionContract]) -> Optional[float]:
        if not side:
            return None
        best = max(side, key=lambda c: _nonnegative(c.open_interest))
        return best.strike if _nonnegative(best.open_interest) > 0 else None

    return {
        "expected_move_pct": expected_move,
        "put_call_oi_ratio": round(put_oi / call_oi, 3) if call_oi > 0 else None,
        "put_wall": wall(puts), "call_wall": wall(calls),
    }


def enrich_contract_greeks(
    contracts: Iterable[OptionContract],
    spot: float,
    risk_free_rate: float = 0.045,
    valuation_date: Optional[date] = None,
    dividend_yield: float = 0.0,
) -> List[OptionContract]:
    valuation = _valuation_day(valuation_date)
    enriched: List[OptionContract] = []
    for contract in contracts:
        for field in ("bid", "ask", "last", "implied_volatility"):
            setattr(contract, field, _finite_number(getattr(contract, field)))
        for field in ("volume", "open_interest"):
            setattr(contract, field, _finite_number(getattr(contract, field)))
        for field in ("delta", "gamma", "theta", "vega", "rho"):
            setattr(contract, field, None)
        volatility = _finite_number(contract.implied_volatility)
        if not volatility:
            # Missing IV does not justify invented model Greeks.
            enriched.append(contract)
            continue
        try:
            greeks = black_scholes_greeks(
                contract.option_type,
                spot,
                contract.strike,
                years_to_expiration(contract.expiration, valuation_date=valuation),
                risk_free_rate,
                volatility,
                dividend_yield,
            )
            for key, value in greeks.items():
                setattr(contract, key, value)
        except (TypeError, ValueError):
            pass
        enriched.append(contract)
    return enriched


def summarize_options(
    ticker: str,
    spot: float,
    contracts: List[OptionContract],
    expiration: Optional[str],
    valuation_date: Optional[date] = None,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
    chain_metadata: Optional[Dict[str, Any]] = None,
) -> OptionSnapshot:
    valuation = _valuation_day(valuation_date)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in (
        risk_free_rate, dividend_yield
    )):
        raise ValueError("risk-free rate and dividend yield must be finite numeric assumptions")
    # Archive before Greek enrichment or display truncation. Nonfinite numbers are null,
    # so the captured normalized provider records remain valid JSON.
    raw_contracts = []
    for contract in contracts:
        raw = deepcopy(contract.to_dict())
        for key in ("strike", "bid", "ask", "last", "volume", "open_interest",
                    "implied_volatility", "delta", "gamma", "theta", "vega", "rho"):
            value = raw.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raw[key] = None
        raw_contracts.append(raw)
    metadata = {
        "source": "caller_supplied_contracts", "scope": "supplied_single_expiry_contract_records",
        "expiration": expiration, "all_expirations_captured": False,
        "valuation_date": valuation.isoformat(), "time_basis": "ACT/365 calendar days",
        "minimum_maturity_days": 1, "maturity_floor_applies_to": "same-day and expired contracts",
        "greek_model": "European Black-Scholes approximation",
        "model_limitations": "Does not model American early exercise "
                             "or discrete dividend schedules.",
        "risk_free_rate": risk_free_rate, "dividend_yield": dividend_yield,
        "assumption_basis": "caller-supplied or constant defaults; not observed market rates",
        "greek_units": {"delta": "per share", "gamma": "per currency-unit spot change",
                        "theta": "per calendar day", "vega": "per 1 percentage-point volatility",
                        "rho": "per 1 percentage-point rate"},
        "aggregate_greeks": "arithmetic mean of available contract Greeks; not portfolio exposure",
        "positioning_aggregation": "Only known nonnegative open interest is summed; "
                                   "unknown values contribute zero computationally, "
                                   "not as observed zero.",
        "option_chain_fetched_at": None, "quote_timestamp": None, "quote_age_verified": False,
        "underlying_spot": _finite_number(spot), "underlying_date": None,
        "underlying_source": "caller_supplied_unverified", "underlying_fetched_at": None,
        "last_trade_time_meaning": "last_trade_at is last trade time, "
                                   "not a bid/ask quote timestamp",
        "capture_stage": "normalized contract records before model Greeks "
                         "and expiry/display filters",
        "numeric_normalization": "nonfinite provider numbers stored as null; "
                                 "not raw provider frames",
        "rate_convention": "annual continuously compounded model inputs",
        "intraday_expiration_time_modeled": False,
        "returned_contract_count": len(raw_contracts),
    }
    metadata.update({key: value for key, value in (chain_metadata or {}).items() if key in {
        "source", "scope", "underlying_source", "underlying_date", "underlying_fetched_at",
        "underlying_spot", "option_chain_fetched_at", "collection_attempted", "provider_exclusions",
        "expiry_selection", "same_day_expiry_excluded",
    }})
    if any(contract.expiration != expiration for contract in contracts):
        metadata["scope"] = "supplied_contract_records_including_other_expiries"
    warnings: List[str] = []
    contracts = [c for c in contracts
                 if c.expiration == expiration and _finite_number(c.strike)]
    metadata["eligible_field_coverage"] = {
        field: {
            "total": len(contracts),
            "known": sum(_finite_number(getattr(c, field)) is not None for c in contracts),
            "missing": sum(_finite_number(getattr(c, field)) is None for c in contracts),
            "known_zero": sum(_finite_number(getattr(c, field)) == 0 for c in contracts),
        } for field in ("volume", "open_interest")
    }
    if not contracts or not math.isfinite(spot) or spot <= 0:
        warnings.append("No option chain available; options context is unavailable/stale.")
        return OptionSnapshot(
            ticker=ticker,
            expiration=expiration,
            contracts=[],
            greeks={"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None},
            skew_summary={
                "call_iv_atm": None,
                "put_iv_atm": None,
                "put_call_iv_spread": None,
                "expected_move_pct": None,
                "put_call_oi_ratio": None,
                "put_wall": None,
                "call_wall": None,
            },
            payoff_scenarios=[],
            warnings=warnings,
            full_chain_contracts=raw_contracts,
            chain_metadata={**metadata, "eligible_contract_count": 0, "display_contract_count": 0},
        )

    if any(not _finite_number(c.implied_volatility) for c in contracts):
        warnings.append(
            "Some contracts lack valid implied volatility; their Greeks are unavailable."
        )
    enriched = enrich_contract_greeks(
        deepcopy(contracts), spot, risk_free_rate=risk_free_rate,
        valuation_date=valuation, dividend_yield=dividend_yield,
    )
    calls = [contract for contract in enriched if contract.option_type == "call"]
    puts = [contract for contract in enriched if contract.option_type == "put"]
    nearest_call = min(calls, key=lambda row: abs(row.strike - spot)) if calls else None
    nearest_put = min(puts, key=lambda row: abs(row.strike - spot)) if puts else None
    call_iv = _finite_number(nearest_call.implied_volatility) if nearest_call else None
    put_iv = _finite_number(nearest_put.implied_volatility) if nearest_put else None
    call_iv = call_iv if call_iv else None
    put_iv = put_iv if put_iv else None
    aggregate = {}
    for greek in ["delta", "gamma", "theta", "vega", "rho"]:
        values = [
            getattr(contract, greek)
            for contract in enriched
            if getattr(contract, greek) is not None
            and math.isfinite(getattr(contract, greek))
        ]
        aggregate[greek] = round(sum(values) / len(values), 6) if values else None
    scenarios = []
    premium = _finite_number(nearest_call.ask) if nearest_call else None
    if nearest_call and premium:
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
        contracts=[contract.to_dict() for side in (calls, puts)
                   for contract in sorted(side, key=lambda c: abs(c.strike - spot))[:10]],
        greeks=aggregate,
        skew_summary={
            "call_iv_atm": call_iv,
            "put_iv_atm": put_iv,
            "put_call_iv_spread": round((put_iv or 0) - (call_iv or 0), 6)
            if call_iv is not None and put_iv is not None
            else None,
            **compute_event_context(enriched, spot),
        },
        payoff_scenarios=scenarios,
        warnings=warnings,
        full_chain_contracts=raw_contracts,
        chain_metadata={**metadata, "eligible_contract_count": len(enriched),
                        "display_contract_count": min(len(calls), 10)+min(len(puts), 10)},
    )


def build_option_snapshot(
    ticker: str, spot: float, live: bool = False, valuation_date: Optional[date] = None,
    risk_free_rate: float = 0.045, dividend_yield: float = 0.0,
) -> OptionSnapshot:
    valuation = _valuation_day(valuation_date)
    warnings: List[str] = []
    contracts: List[OptionContract] = []
    expiration = None
    collect_current = live and valuation == _valuation_day()
    if collect_current:
        contracts, expiration, warnings = fetch_yfinance_option_contracts(
            ticker, spot, valuation_date=valuation
        )
    elif live:
        warnings.append("Historical/future valuation requested: current option chains were not "
                        "backfilled. Supply a genuinely archived chain for historical valuation.")
    fetched = datetime.now(timezone.utc).isoformat() if collect_current and contracts else None
    snapshot = summarize_options(
        ticker, spot, contracts, expiration, valuation_date=valuation,
        risk_free_rate=risk_free_rate, dividend_yield=dividend_yield,
        chain_metadata={
            "source": "yfinance" if collect_current else "unavailable",
            "scope": "nearest_expiry_normalized_returned_contracts",
            "expiry_selection": "strictly_after_daily_valuation_date",
            "same_day_expiry_excluded": True,
            "collection_attempted": collect_current, "option_chain_fetched_at": fetched,
            "provider_exclusions": "Invalid strikes are skipped by the provider parser and warned.",
            "underlying_spot": _finite_number(spot),
        },
    )
    snapshot.warnings.extend(warnings)
    if not live:
        snapshot.warnings.append(
            "Live options ingest skipped. Pass --live to attempt yfinance option chain fetch."
        )
    return snapshot


def build_option_snapshots(
    market_data: Dict[str, List[Dict[str, Any]]],
    live: bool = False,
    valuation_date: Optional[date] = None,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    valuation = _valuation_day(valuation_date)
    snapshots: Dict[str, Dict[str, Any]] = {}
    for ticker, rows in market_data.items():
        if not rows:
            snapshots[ticker] = summarize_options(
                ticker, 0.0, [], None, valuation_date=valuation,
                risk_free_rate=risk_free_rate, dividend_yield=dividend_yield,
            ).to_dict()
            continue
        spot = float(rows[-1]["close"]) if rows else 0.0
        snapshot = build_option_snapshot(
            ticker, spot, live=live, valuation_date=valuation,
            risk_free_rate=risk_free_rate, dividend_yield=dividend_yield,
        )
        snapshot.chain_metadata.update(
            underlying_spot=spot, underlying_date=rows[-1].get("date"),
            underlying_source=rows[-1].get("source", "unverified"),
            underlying_fetched_at=rows[-1].get("fetched_at"),
        )
        snapshots[ticker] = {
            **snapshot.to_dict(),
            "underlying_spot": spot,
            "underlying_date": rows[-1].get("date"),
            "underlying_source": rows[-1].get("source", "unverified"),
            "fetched_at": snapshot.chain_metadata.get("option_chain_fetched_at"),
        }
    return snapshots
