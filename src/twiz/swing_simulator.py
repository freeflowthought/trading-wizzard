"""Long-only price-based research book: close decisions, following-session open fills.

Setups and their upstream generation are supplied separately. This simulator
does not fit probabilities, invent author deadlines, or place real orders.
"""

import math
from collections import Counter
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from .event_direction import aggregate_event_direction
from .sessions import is_us_equity_session
from .swing_optimizer import (
    CORE_KEYS,
    EXECUTION,
    STRATEGY_KEYS,
    VARIANTS,
    _hash,
    _number,
    _time,
    validate_candidate_parameters,
)

NY = ZoneInfo("America/New_York")
EXECUTION_KEYS = CORE_KEYS - {"lookback_sessions"}
EXECUTION_V2_KEYS = EXECUTION_KEYS | STRATEGY_KEYS
BAD_SOURCES = {"sample", "unknown", "unverified", "synthetic", "demo", "placeholder"}
# Leave microscopic headroom below hard ceilings; never round an actual breach down.
CEILING_BUFFER = 1 - 1e-12
EXIT_VERSION = "asymmetric-close-exit-v1"
EXIT_RETENTION_VERSION = "asymmetric-close-exit-v2"
RETENTION_PEAK_BASIS = "completed_close_realized_net_plus_remaining_net_liquidation"


def _exit_policy(value):
    if value is None:
        return None
    base_keys = {"first_trim_r", "trailing_atr", "resistance_lookback_sessions"}
    if not isinstance(value, dict) or set(value) not in (
            base_keys, base_keys | {"minimum_profit_retention"}):
        raise ValueError("Exit policy requires its three keys and optional profit retention.")
    for key, low, high in (("first_trim_r", .5, 2), ("trailing_atr", 1, 4)):
        if (type(value[key]) not in (int, float) or _number(value[key]) is None
                or not low <= value[key] <= high):
            raise ValueError("Invalid asymmetric exit parameter: " + key)
    lookback = value["resistance_lookback_sessions"]
    if type(lookback) is not int or not 10 <= lookback <= 63:
        raise ValueError("Exit resistance lookback must be 10–63 sessions.")
    result = {"version": EXIT_VERSION, "parameters": {
        "first_trim_r": float(value["first_trim_r"]),
        "trailing_atr": float(value["trailing_atr"]),
        "resistance_lookback_sessions": lookback}}
    if "minimum_profit_retention" in value:
        retention = value["minimum_profit_retention"]
        if (type(retention) not in (int, float) or _number(retention) is None
                or not 0 <= retention <= 1):
            raise ValueError("Minimum profit retention must be finite in [0, 1].")
        result["version"] = EXIT_RETENTION_VERSION
        result["parameters"]["minimum_profit_retention"] = float(retention)
    return result


def _clock(day, opening=False):
    return datetime.combine(day, time(9, 30) if opening else time(16, 15), tzinfo=NY)


def _advance(day, n):
    for _ in range(n):
        day += timedelta(days=1)
        while not is_us_equity_session(day):
            day += timedelta(days=1)
    return day


def _clean_prices(market):
    output = {}
    for asset, rows in market.items():
        clean, duplicate = {}, set()
        for row in rows:
            try:
                day = datetime.fromisoformat(str(row.get("date"))).date()
            except (TypeError, ValueError):
                continue
            values = {key: _number(row.get(key)) for key in ("open", "high", "low", "close")}
            source, basis = row.get("source"), row.get("price_basis")
            if (
                not is_us_equity_session(day)
                or row.get("session_complete") is not True
                or not isinstance(source, str)
                or not source.strip()
                or source.strip().lower() in BAD_SOURCES
                or not isinstance(basis, str)
                or not basis.strip()
                or any(value is None or value <= 0 for value in values.values())
                or not values["low"]
                <= min(values["open"], values["close"])
                <= max(values["open"], values["close"])
                <= values["high"]
            ):
                continue
            if day in clean:
                duplicate.add(day)
            clean[day] = {**row, **values}
        output[asset] = {day: row for day, row in clean.items() if day not in duplicate}
    return output


def _signal(state, bar, reference, lower, stamp, confirm_sessions, rule="retest"):
    """Only completed bars establish a chosen rule's independent batch signal."""
    if rule != "retest":
        touched = (bar["low"] < reference if rule == "sweep_reclaim" else
                   bar["low"] <= reference and bar["high"] >= lower)
        if state["phase"] == "watch" and touched:
            state["phase"] = "touch"
        if state["phase"] == "touch" and bar["close"] >= reference:
            # A sweep and reclaim must share the completed signal bar.
            if rule == "sweep_reclaim" and not bar["low"] < reference:
                return False
            state.update(phase="holding", holds=1)
        elif state["phase"] == "holding":
            if bar["close"] >= reference:
                state["holds"] += 1
            else:
                state.update(phase="watch", holds=0)
        return state["phase"] == "holding" and state["holds"] >= confirm_sessions
    if state["phase"] == "watch":
        if bar["low"] <= reference and bar["high"] >= lower:
            state["phase"] = "touch"
    if state["phase"] == "touch" and bar["close"] >= reference:
        state.update(phase="reclaimed", reclaim_at=stamp, holds=0)
        return False
    if state["phase"] == "reclaimed" and stamp > state["reclaim_at"]:
        if bar["low"] <= reference and bar["high"] >= lower and bar["close"] >= reference:
            state.update(phase="retest", holds=1)
    elif state["phase"] == "retest":
        if bar["close"] >= reference:
            state["holds"] += 1
        else:
            state.update(phase="touch", holds=0)
    return state["phase"] == "retest" and state["holds"] >= confirm_sessions


def _event_gate(setup, variant, params, horizon_end):
    context = setup.get("event_context") or {}
    if variant == "price_only":
        return True, "price_only_ablation"
    if variant == "signed_weighted_event_score":
        result = aggregate_event_direction(
            context.get("components", []),
            as_of=setup["available_at"],
            horizon_end=horizon_end,
            channel_weights=context.get("channel_weights", {}),
            event_threshold=params["event_threshold"],
            main_case_only=params["main_case_only"],
        )
        return result.get("direction_score") is not None and result["direction_score"] > 0, result
    known = _time(context.get("available_at"))
    probability = _number(context.get("equity_up_probability"))
    if (
        not known
        or known > _time(setup["available_at"])
        or context.get("probability_estimate_eligible") is not True
        or probability is None
        or not 0 <= probability <= 1
        or context.get("event_threshold") != params["event_threshold"]
        or context.get("main_case_only") is not params["main_case_only"]
        or any(
            key in context and (not _time(context[key]) or _time(context[key]) > known)
            for key in ("features_available_at", "model_as_of", "model_fit_at")
        )
    ):
        return False, "learned_bridge_unavailable_or_feature_configuration_mismatch"
    return probability >= params["trade_probability_threshold"], {
        "status": "ready",
        "equity_up_probability": probability,
        "available_at": known.isoformat(),
        "model_id": context.get("model_id"),
        "threshold": params["trade_probability_threshold"],
    }


def simulate_swing_book(
    market_data,
    setups,
    *,
    execution_parameters=None,
    candidate=None,
    candidate_id,
    parameters_sha256,
    block_id,
    partition,
    block_start,
    block_end,
    event_variant="price_only",
    corporate_actions=None,
    unavailable_opportunities=None,
    exit_policy=None,
):
    """Return selector-compatible campaigns and actual price-derived daily book NAV.

    Required setup fields: opportunity_id, asset, origin_at, available_at,
    features_available_at, watch_expires_at, zone_lower/upper, atr, price_basis,
    upstream_generation_hash. Optional stop_price, target_prices, event_context,
    comparison_available_at and author_explicit_deadline. lookback/structure
    parameters belong to the generator and are rejected here, rather than ignored.
    zone_width_atr is the TOTAL effective watch-band width about the supplied center.
    """
    treatment = _exit_policy(exit_policy)
    exit_metadata = ({"exit_policy": treatment, "exit_policy_sha256": _hash(treatment),
                      "candidate_parameters_include_exit_policy": False} if treatment else {})
    supplied = execution_parameters or {}
    if set(supplied) - EXECUTION_V2_KEYS:
        raise ValueError(
            "Unsupported simulator parameters; lookback/structure belong to setup generation."
        )
    from .swing_optimizer import _validate_parameter

    if not isinstance(candidate, dict):
        raise ValueError("A full frozen candidate is required to bind actual execution parameters.")
    try:
        content = {
            key: candidate[key] for key in ("parameters", "structure_parameters", "event_variant")
        }
        normalized = validate_candidate_parameters(candidate["parameters"])
        extended = set(normalized) == CORE_KEYS | STRATEGY_KEYS
        execution_keys = EXECUTION_V2_KEYS if extended else EXECUTION_KEYS
        if (
            _hash(content) != candidate["parameters_sha256"]
            or parameters_sha256 != candidate["parameters_sha256"]
            or candidate_id not in {candidate["id"], "baseline"}
            or event_variant != candidate["event_variant"]
        ):
            raise ValueError("Frozen candidate identity, variant or parameter hash mismatch.")
        params = {
            key: normalized[key] for key in execution_keys
        }
        if (set(supplied)-execution_keys or any(
                _validate_parameter(key, value) != params[key]
                for key, value in supplied.items())):
            raise ValueError("Actual execution parameters differ from the frozen candidate.")
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed frozen execution candidate.") from error
    start, end = _time(block_start), _time(block_end)
    if (
        not start
        or not end
        or start >= end
        or event_variant not in VARIANTS
        or partition not in {"train", "validation", "test"}
        or not candidate_id
        or not parameters_sha256
    ):
        raise ValueError("Invalid simulator dates, identity, event variant or partition.")
    fee, slip = EXECUTION["fee_bps_per_side"] / 10000, EXECUTION["slippage_bps_per_side"] / 10000
    cash, active, trace, campaigns, setup_ids = 100.0, None, [], [], set()
    if not isinstance(setups, list) or any(not isinstance(raw, dict) for raw in setups):
        raise ValueError("Setups must be an explicit list of dated setup objects.")
    if any(
        raw.get("candidate_id") != candidate["id"]
        or raw.get("parameters_sha256") != candidate["parameters_sha256"]
        for raw in setups
    ):
        raise ValueError("A setup belongs to a different or unbound frozen candidate.")
    prices = _clean_prices(market_data)
    counts = Counter(
        raw.get("opportunity_id") for raw in setups if isinstance(raw.get("opportunity_id"), str)
    )
    incomplete = False
    for raw in setups:
        identity = raw.get("opportunity_id")
        record = {
            "opportunity_id": identity,
            "asset": raw.get("asset"),
            "status": "unavailable",
            "origin_at": raw.get("origin_at"),
            "outcome_available_at": None,
            "net_account_pnl_pct": 0.0,
            "new_capital_pct": 0.0,
            "modeled_loss_pct": 0.0,
            "entered_at": None,
            "exited_at": None,
            "entry_signal_at": None,
            "exit_signal_at": None,
            "fills": [],
            "reasons": [],
            "author_explicit_deadline": raw.get("author_explicit_deadline"),
            "literal_forecast_status": "not_scored_by_trade_simulator",
            "upstream_generation_hash": raw.get("upstream_generation_hash"),
        }
        campaigns.append(record)
        try:
            origin, available, known, watch = (
                _time(raw.get(key))
                for key in (
                    "origin_at",
                    "available_at",
                    "features_available_at",
                    "watch_expires_at",
                )
            )
            low, high, atr = (_number(raw.get(key)) for key in ("zone_lower", "zone_upper", "atr"))
            if (
                not isinstance(identity, str)
                or not identity
                or identity in setup_ids
                or counts[identity] != 1
                or not all((origin, available, known, watch))
                or not known <= origin <= available < watch
                or not start <= origin <= end
                or not raw.get("upstream_generation_hash")
                or not isinstance(raw.get("price_basis"), str)
                or any(value is None or value <= 0 for value in (low, high, atr))
                or low > high
            ):
                raise ValueError("Invalid setup identity, dated inputs, watch window or zone.")
            setup_ids.add(identity)
            frozen_watch = watch
            if extended:
                watch = min(watch, _clock(_advance(
                    origin.astimezone(NY).date(), params["watch_sessions"])))
            center = (low + high) / 2
            low, high = (
                center - params["zone_width_atr"] * atr / 2,
                center + params["zone_width_atr"] * atr / 2,
            )
            stop = (
                _number(raw.get("stop_price"))
                if raw.get("stop_price") is not None
                else (low - params["stop_atr"] * atr)
            )
            if stop is None or not 0 < stop < low:
                raise ValueError(
                    "Stop must be positive and below all conditional entry references."
                )
            maturity = _clock(
                _advance(frozen_watch.astimezone(NY).date(), params["holding_sessions"] + 1), True
            )
            supplied_maturity = _time(raw.get("comparison_available_at"))
            if supplied_maturity:
                maturity = max(maturity, supplied_maturity)
            record["outcome_available_at"] = maturity.isoformat()
            for action in corporate_actions or []:
                if (
                    action.get("ticker") == raw["asset"]
                    and action.get("type") == "split"
                    and available.date().isoformat()
                    <= str(action.get("effective_date"))
                    <= maturity.date().isoformat()
                ):
                    incomplete = True
                    raise ValueError(
                        "Cross-split campaign requires a verified nominal/share-basis adapter."
                    )
            gate, event_evidence = _event_gate(raw, event_variant, params, maturity.isoformat())
            record["event_evidence"] = event_evidence
            if not gate:
                record["status"] = (
                    "unfilled"
                    if isinstance(event_evidence, dict) and event_evidence.get("status") == "ready"
                    else "unavailable"
                )
                record["reasons"].append(
                    "Event gate unavailable or not supportive; no invented probability."
                )
                continue
            explicit_targets = raw.get("target_prices")
            if explicit_targets is not None and (
                not isinstance(explicit_targets, list)
                or not 1 <= len(explicit_targets) <= 3
                or any(_number(v) is None or v <= high for v in explicit_targets)
                or any(b <= a for a, b in zip(explicit_targets, explicit_targets[1:]))
            ):
                raise ValueError(
                    "Explicit target prices must be finite, ordered and above the entry zone."
                )
            record.update(
                status="watch",
                raw=raw,
                available=available,
                watch=watch,
                lower=low,
                upper=high,
                atr=atr,
                stop=stop,
                refs=(
                    [high+atr, high+atr*.5, high]
                    if extended and params["entry_ladder"] == "atr_approach"
                    else [high, center, low]
                ),
                signal={"phase": "watch"},
                batch=0,
                entry_due=False,
                exit_due=None,
                spent=0.0,
                proceeds=0.0,
                qty=0.0,
                bought_qty=0.0,
                original_qty=None,
                budget=None,
                holding=0,
                first_trim=False,
                next_target=0,
                adds_cancelled=False,
                targets=explicit_targets,
                mae=0.0,
                mfe=0.0,
                stop_basis="explicit" if raw.get("stop_price") is not None else "ATR_proxy",
                target_basis="explicit" if explicit_targets is not None else "R_proxy",
                effective_watch_expires_at=watch.isoformat(),
                original_watch_expires_at=frozen_watch.isoformat(),
            )
            if treatment:
                record["_exit"] = {"original_stop": stop, "highest_close": None,
                                   "highest_at": None, "first_trim_filled_at": None,
                                   "target_revisions": [], "stop_revisions": []}
                if treatment["version"] == EXIT_RETENTION_VERSION:
                    record["_exit"].update(net_profit_peak=0.0, net_profit_peak_at=None)
        except (ValueError, KeyError, TypeError) as error:
            record["reasons"].append(str(error))
            if record["outcome_available_at"] is None:
                record["outcome_available_at"] = end.isoformat()
    nav = [{"at": start.isoformat(), "equity": 100.0}]
    for raw in unavailable_opportunities or []:
        if raw.get("opportunity_id") in setup_ids:
            raise ValueError("Unavailable disposition duplicates an active setup identity.")
        origin = _time(raw.get("origin_at"))
        mature = _time(raw.get("comparison_available_at", raw.get("outcome_available_at")))
        if not origin or not mature or not origin < mature:
            raise ValueError("Unavailable opportunities need dated origin and common maturity.")
        campaigns.append(
            {
                "opportunity_id": raw.get("opportunity_id"),
                "asset": raw.get("asset"),
                "origin_at": origin.isoformat(),
                "outcome_available_at": mature.isoformat(),
                "status": "unavailable",
                "net_account_pnl_pct": 0.0,
                "new_capital_pct": 0.0,
                "modeled_loss_pct": 0.0,
                "entered_at": None,
                "exited_at": None,
                "entry_signal_at": None,
                "exit_signal_at": None,
                "fills": [],
                "reasons": [raw.get("reason", "upstream_generation_unavailable")],
            }
        )

    def lower_first_target(record, fill_cost, day):
        policy, state = treatment["parameters"], record["_exit"]
        risk = fill_cost - state["original_stop"]
        options = [{"price": record["targets"][0], "basis": "existing_first_target",
                    "available_at": (_clock(day, True).isoformat()
                                     if record["target_basis"] == "R_proxy"
                                     else record["ready_at"].isoformat())},
                   {"price": fill_cost + policy["first_trim_r"] * risk,
                    "basis": "first_fill_all_in_cost_plus_policy_R",
                    "available_at": _clock(day, True).isoformat()}]
        if record["target_basis"] == "explicit":
            options = [{**options[0], "basis": "explicit_targets_preserved"}]
        previous = day
        for _ in range(0 if record["target_basis"] == "explicit"
                       else policy["resistance_lookback_sessions"]):
            previous -= timedelta(days=1)
            while not is_us_equity_session(previous):
                previous -= timedelta(days=1)
            known_at = _clock(previous)
            prior = prices.get(record["asset"], {}).get(previous)
            if (prior and prior["price_basis"] == record["raw"]["price_basis"]
                    and known_at <= record["ready_at"]
                    and prior["high"] > fill_cost + .5 * record["atr"]):
                options.append({"price": prior["high"], "basis": "known_prior_completed_high",
                                "source_date": previous.isoformat(),
                                "available_at": known_at.isoformat()})
        chosen = min(options, key=lambda item: (item["price"], item["available_at"]))
        old = list(record["targets"])
        record["targets"][0] = chosen["price"]
        state["target_revisions"].append({
            "at": _clock(day, True).isoformat(), "basis": chosen["basis"],
            "source_date": chosen.get("source_date"),
            "inputs_available_at": _clock(day, True).isoformat(),
            "source_available_at": chosen["available_at"],
            "old_targets": old, "new_targets": list(record["targets"]),
            "first_fill_all_in_cost": fill_cost, "original_risk_per_share": risk})

    def protect_remaining(record, bar, closing):
        state = record["_exit"]
        if state["highest_close"] is None or bar["close"] > state["highest_close"]:
            state.update(highest_close=bar["close"], highest_at=closing.isoformat())
        retention = treatment["parameters"].get("minimum_profit_retention")
        if retention is not None:
            net_profit = (record["proceeds"] + record["qty"] * bar["close"]
                          * (1 - slip) * (1 - fee) - record["spent"])
            if state["net_profit_peak_at"] is None or net_profit > state["net_profit_peak"]:
                state["net_profit_peak"] = max(state["net_profit_peak"], net_profit, 0.0)
                state["net_profit_peak_at"] = closing.isoformat()
        if state["first_trim_filled_at"] is None:
            return
        breakeven = max(0., (record["spent"] - record["proceeds"])
                        / (record["qty"] * (1 - slip) * (1 - fee)))
        trailing = state["highest_close"] - treatment["parameters"]["trailing_atr"] * record["atr"]
        new_stop = max(record["stop"], state["original_stop"], breakeven, trailing)
        retention_trace = {}
        if retention is not None:
            floor = (record["spent"] - record["proceeds"] + retention * state["net_profit_peak"]
                     ) / (record["qty"] * (1 - slip) * (1 - fee))
            new_stop = max(new_stop, floor)
            retention_trace = {"minimum_profit_retention": retention,
                               "profit_retention_floor": floor,
                               "peak_net_liquidation_profit": state["net_profit_peak"],
                               "peak_net_profit_at": state["net_profit_peak_at"],
                               "profit_peak_basis": RETENTION_PEAK_BASIS}
        if new_stop > record["stop"]:
            state["stop_revisions"].append({
                "at": closing.isoformat(), "inputs_available_at": closing.isoformat(),
                "old_stop": record["stop"], "new_stop": new_stop,
                "original_stop": state["original_stop"], "campaign_net_breakeven": breakeven,
                "trailing_price": trailing, "highest_completed_close": state["highest_close"],
                "highest_completed_close_at": state["highest_at"],
                "first_trim_filled_at": state["first_trim_filled_at"], **retention_trace})
            record["stop"] = new_stop

    def sell(record, bar, day):
        nonlocal cash, active
        order = record["exit_due"]
        quantity = min(record["qty"], order["quantity"])
        fill = bar["open"] * (1 - slip)
        proceeds = quantity * fill * (1 - fee)
        cash += proceeds
        record["proceeds"] += proceeds
        record["qty"] -= quantity
        record["fills"].append(
            {
                "side": "sell",
                "at": _clock(day, True).isoformat(),
                "signal_at": order["signal_at"],
                "quantity": quantity,
                "price": fill,
                "fee": quantity * fill * fee,
                "reason": order["reason"],
            }
        )
        record["exit_due"] = None
        if treatment and order["reason"].startswith("close_take_profit"):
            state = record["_exit"]
            if state["first_trim_filled_at"] is None:
                state["first_trim_filled_at"] = _clock(day, True).isoformat()
            record["adds_cancelled"] = True
        if record["qty"] <= max(record["bought_qty"] * 1e-12, 1e-300):
            record.update(
                status="closed",
                exited_at=_clock(day, True).isoformat(),
                exit_signal_at=order["signal_at"],
                qty=0.0,
                net_account_pnl_pct=record["proceeds"] - record["spent"],
            )
            record["mae"] = min(record["mae"], record["net_account_pnl_pct"])
            record["mfe"] = max(record["mfe"], record["net_account_pnl_pct"])
            active = None

    def buy(record, bar, day):
        nonlocal cash, active
        fill_cost = bar["open"] * (1 + slip) * (1 + fee)
        if not math.isfinite(fill_cost):
            raise ValueError("Nonfinite derived entry cost.")
        if (extended and bar["open"] > record["refs"][record["batch"]]
                + params["max_entry_extension_atr"]*record["atr"]):
            record.update(status="open" if record["qty"] else "cancelled", entry_due=False,
                          adds_cancelled=True)
            record["reasons"].append("Next-open extension exceeds this frozen batch's ATR cap.")
            return
        if bar["open"] <= record["stop"]:
            record.update(
                status="open" if record["qty"] else "cancelled",
                entry_due=False,
                adds_cancelled=True,
            )
            record["reasons"].append(
                "Next open is below the structural stop; cancel before funding."
            )
            return
        if record["targets"] and fill_cost >= record["targets"][0]:
            record.update(
                status="open" if record["qty"] else "cancelled",
                entry_due=False,
                adds_cancelled=True,
            )
            record["reasons"].append(
                "Next open has already passed the first target; no chase fill."
            )
            return
        if record["budget"] is None:
            record["starting_equity"] = cash
            risk_prices = [fill_cost] + [
                price * (1 + slip) * (1 + fee) for price in record["refs"][1:]
            ]
            stop_net = record["stop"] * (1 - slip) * (1 - fee)
            modeled = sum(
                10 * w * (1 - stop_net / p) for w, p in zip(params["entry_fractions"], risk_prices)
            )
            scale = min(1.0, 0.5 * CEILING_BUFFER / modeled) if modeled > 0 else 1.0
            record["budget"] = [cash * 0.1 * w * scale for w in params["entry_fractions"]]
            if not record["targets"]:
                r = fill_cost - record["stop"]
                record["targets"] = [
                    fill_cost + (params["take_profit_r"] + n) * r for n in range(3)
                ]
                if not all(math.isfinite(value) for value in record["targets"]):
                    raise ValueError("Nonfinite derived targets.")
        stop_net = record["stop"] * (1 - slip) * (1 - fee)
        risk_used = record["spent"] - record["bought_qty"] * stop_net
        remaining_risk = max(0, record["starting_equity"] * 0.005 * CEILING_BUFFER - risk_used)
        spend = min(
            record["budget"][record["batch"]],
            cash,
            max(0, record["starting_equity"] * 0.1 * CEILING_BUFFER - record["spent"]),
            remaining_risk / max(1 - stop_net / fill_cost, 1e-12),
        )
        if spend <= 1e-10:
            record["entry_due"] = False
            record["adds_cancelled"] = True
            record["reasons"].append("No remaining shared capital/risk authority for this batch.")
            return
        quantity = spend / fill_cost
        cash -= spend
        record["qty"] += quantity
        record["bought_qty"] += quantity
        record["spent"] += spend
        record["fills"].append(
            {
                "side": "buy",
                "at": _clock(day, True).isoformat(),
                "signal_at": record["ready_at"].isoformat(),
                "quantity": quantity,
                "price": bar["open"] * (1 + slip),
                "fee": spend - quantity * bar["open"] * (1 + slip),
                "batch": record["batch"] + 1,
            }
        )
        if record["entered_at"] is None:
            record["entered_at"] = _clock(day, True).isoformat()
            record["entry_signal_at"] = record["ready_at"].isoformat()
            if treatment:
                lower_first_target(record, fill_cost, day)
        record.update(
            status="open", entry_due=False, batch=record["batch"] + 1, signal={"phase": "watch"}
        )
        record["new_capital_pct"] = record["spent"] / record["starting_equity"] * 100
        record["modeled_loss_pct"] = (
            (record["spent"] - record["bought_qty"] * stop_net) / record["starting_equity"] * 100
        )
        if record["new_capital_pct"] > 10 + 1e-9 or record["modeled_loss_pct"] > 0.5 + 1e-9:
            raise ValueError("Shared account capital/risk ceiling exceeded.")
        active = record

    day = start.astimezone(NY).date()
    while _clock(day, True) <= end:
        if incomplete:
            break
        if not is_us_equity_session(day):
            day += timedelta(days=1)
            continue
        opening, closing = _clock(day, True), _clock(day)
        if opening >= start:
            if active and active["exit_due"]:
                bar = prices.get(active["asset"], {}).get(day)
                if bar and bar["price_basis"] == active["raw"]["price_basis"]:
                    sell(active, bar, day)
                else:
                    incomplete = True
                    active["reasons"].append(
                        "Required next-open exit quote missing; no delayed substitute fill."
                    )
                    break
            ready = sorted(
                [r for r in campaigns if r.get("entry_due")],
                key=lambda r: (
                    r["ready_at"],
                    _time(r["origin_at"]),
                    r["asset"],
                    candidate_id,
                    r["opportunity_id"],
                ),
            )
            for record in ready:
                if active is not None and active is not record:
                    record.update(status="blocked", entry_due=False)
                    record["reasons"].append(
                        "One shared funded campaign already occupies the book."
                    )
                    continue
                bar = prices.get(record["asset"], {}).get(day)
                if bar and bar["price_basis"] == record["raw"]["price_basis"]:
                    try:
                        buy(record, bar, day)
                    except ValueError as error:
                        incomplete = True
                        record["reasons"].append(str(error))
                        break
                else:
                    record.update(status="unavailable", entry_due=False)
                    record["reasons"].append("Missing next-open quote or price-basis mismatch.")
                    incomplete = True
                    break
            if incomplete:
                break
            if opening == end:
                mark = prices.get(active["asset"], {}).get(day) if active else None
                if active and mark is None:
                    incomplete = True
                    break
                nav.append(
                    {
                        "at": opening.isoformat(),
                        "equity": cash + active["qty"] * mark["open"] if active else cash,
                    }
                )
        if start <= closing <= end:
            for record in campaigns:
                if (record["status"] not in {"watch", "open"}
                        or (closing if extended else opening) < record["available"]):
                    continue
                if closing > record["watch"]:
                    if record["status"] == "watch":
                        record["status"] = "expired"
                        continue
                    record["adds_cancelled"] = True
                bar = prices.get(record["asset"], {}).get(day)
                if not bar or bar["price_basis"] != record["raw"]["price_basis"]:
                    incomplete = True
                    record["reasons"].append("Missing completed session or mismatched basis.")
                    break
                if record["status"] == "open":
                    if treatment:
                        protect_remaining(record, bar, closing)
                    record["holding"] += 1
                    mark_pnl = record["proceeds"] + record["qty"] * bar["close"] - record["spent"]
                    record["mae"], record["mfe"] = (
                        min(record["mae"], mark_pnl),
                        max(record["mfe"], mark_pnl),
                    )
                    reason, quantity = None, record["qty"]
                    if bar["close"] <= record["stop"]:
                        reason = "close_stop"
                    elif record["holding"] >= params["holding_sessions"]:
                        reason = "holding_deadline_close"
                    elif (
                        record["next_target"] < len(record["targets"])
                        and bar["close"] >= record["targets"][record["next_target"]]
                    ):
                        if not record["first_trim"]:
                            record.update(
                                first_trim=True, original_qty=record["qty"], entry_due=False
                            )
                        stages = []
                        while (
                            record["next_target"] < len(record["targets"])
                            and bar["close"] >= record["targets"][record["next_target"]]
                        ):
                            stages.append(record["next_target"])
                            record["next_target"] += 1
                        quantity = record["original_qty"] * sum(
                            params["trim_fractions"][i] for i in stages
                        )
                        reason = "close_take_profit_" + "_".join(str(i + 1) for i in stages)
                    if reason:
                        record["exit_due"] = {
                            "quantity": min(quantity, record["qty"]),
                            "reason": reason,
                            "signal_at": closing.isoformat(),
                        }
                        record["entry_due"] = False
                        continue
                elif bar["close"] <= record["stop"]:
                    record["status"] = "cancelled"
                    record["reasons"].append(
                        "Structural close failure before entry; old setup cannot resurrect."
                    )
                    continue
                if (
                    record["batch"] < 3
                    and not record["first_trim"]
                    and not record["adds_cancelled"]
                ):
                    reference = record["refs"][record["batch"]]
                    if _signal(
                        record["signal"],
                        bar,
                        reference,
                        record["lower"],
                        closing,
                        params["confirm_sessions"],
                        params.get("entry_rule", "retest"),
                    ):
                        record.update(entry_due=True, ready_at=closing)
                        trace.append(
                            {
                                "opportunity_id": record["opportunity_id"],
                                "at": closing.isoformat(),
                                "event": "next_open_entry_ready",
                                "batch": record["batch"] + 1,
                            }
                        )
            if incomplete:
                break
            if active:
                mark = prices.get(active["asset"], {}).get(day)
                if mark and mark["price_basis"] == active["raw"]["price_basis"]:
                    equity = cash + active["qty"] * mark["close"]
                else:
                    equity = None
                    incomplete = True
            else:
                equity = cash
            if equity is not None:
                if not math.isfinite(equity) or equity <= 0:
                    incomplete = True
                    break
                if _time(nav[-1]["at"]) != closing:
                    nav.append({"at": closing.isoformat(), "equity": equity})
        day += timedelta(days=1)
    for record in campaigns:
        if record["status"] in {"watch", "open"}:
            record["status"] = "pending"
            record["reasons"].append("End of supplied study window; no forced favorable exit.")
        if record["status"] == "pending":
            record["net_account_pnl_pct"] = None
    internal = {
        "raw",
        "available",
        "watch",
        "lower",
        "upper",
        "atr",
        "stop",
        "refs",
        "signal",
        "batch",
        "entry_due",
        "exit_due",
        "spent",
        "proceeds",
        "qty",
        "bought_qty",
        "original_qty",
        "budget",
        "holding",
        "first_trim",
        "next_target",
        "targets",
        "starting_equity",
        "ready_at",
        "_exit",
    }
    cleaned = []
    for original in campaigns:
        record = {key: value for key, value in original.items() if key not in internal}
        record.update(
            original_zone_lower=original.get("raw", {}).get("zone_lower"),
            original_zone_upper=original.get("raw", {}).get("zone_upper"),
            applied_zone_lower=original.get("lower"),
            applied_zone_upper=original.get("upper"),
            applied_stop_price=original.get("stop"),
            applied_target_prices=original.get("targets"),
            applied_entry_references=original.get("refs"),
            remaining_quantity=original.get("qty", 0),
            cash_committed=original.get("spent", 0),
            net_sale_proceeds=original.get("proceeds", 0),
            original_dip_quantity=original.get("original_qty") or original.get("bought_qty", 0),
            excursion_basis="daily_close_account_percentage_points; actual final gap included",
        )
        record["inactive_parameters"] = (
            ["event_threshold", "main_case_only", "trade_probability_threshold"]
            if event_variant == "price_only"
            else ["trade_probability_threshold"]
            if event_variant == "signed_weighted_event_score"
            else []
        )
        if record.get("stop_basis") == "explicit":
            record["inactive_parameters"].append("stop_atr")
        if record.get("target_basis") == "explicit":
            record["inactive_parameters"].append("take_profit_r")
        if event_variant == "signed_weighted_event_score" and not params["main_case_only"]:
            record["inactive_parameters"].append("event_threshold")
        if treatment:
            state = original.get("_exit", {})
            record.update(exit_policy_version=treatment["version"],
                          exit_policy_sha256=exit_metadata["exit_policy_sha256"],
                          original_stop_price=state.get("original_stop"),
                          first_trim_filled_at=state.get("first_trim_filled_at"),
                          target_revisions=state.get("target_revisions", []),
                          stop_revisions=state.get("stop_revisions", []))
            if treatment["version"] == EXIT_RETENTION_VERSION:
                record.update(peak_net_liquidation_profit=state.get("net_profit_peak", 0.0),
                              peak_net_profit_at=state.get("net_profit_peak_at"),
                              profit_peak_basis=RETENTION_PEAK_BASIS)
            if record.get("target_basis") == "explicit":
                record["inactive_parameters"].extend([
                    "exit_policy.first_trim_r", "exit_policy.resistance_lookback_sessions"])
        cleaned.append(record)
    if not nav or _time(nav[-1]["at"]) != end:
        incomplete = True
    return {
        **exit_metadata,
        "candidate_id": candidate_id,
        "frozen_candidate_id": candidate["id"],
        "candidate_binding_verified": True,
        "parameters_sha256": parameters_sha256,
        "execution_policy_id": _hash(EXECUTION),
        "block_id": block_id,
        "partition": partition,
        "block_start": start.isoformat(),
        "block_end": end.isoformat(),
        "outcome_available_at": max(
            [end] + [_time(r["outcome_available_at"]) for r in campaigns]
        ).isoformat(),
        "campaigns": cleaned,
        "equity_curve": [] if incomplete else nav,
        "partial_equity_curve": nav if incomplete else [],
        "trace": trace,
        "status": "incomplete"
        if incomplete
        else "pending"
        if any(r["status"] == "pending" for r in campaigns)
        else "complete",
        "evaluation_basis": "price_based_daily_close_next_open_simulator",
        "execution_parameters_applied": params,
        "zone_band_semantics": "TOTAL_ATR_width_replaces_reference_width_about_its_midpoint",
        "upstream_reference_width_not_used_directly": True,
        "event_variant": event_variant,
        "price_pnl_only": True,
        "dividend_cash_included": False,
        "corporate_action_ledger_supplied": corporate_actions is not None,
        "goal_completion_proven": False,
        "author_comparison_proven": False,
        "limitations": [
            "Gap fills can exceed modeled .5% risk; no leverage or guaranteed stops.",
            "Price PnL excludes unverified dividend cash and other corporate actions.",
            "Cross-split setups require a separate verified unit/basis conversion.",
            "Author deadlines remain unchanged; trade holding is a separate proxy.",
        ],
    }
