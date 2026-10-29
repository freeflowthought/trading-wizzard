"""Daily paper state of the accepted news-only cycle; never broker instructions."""

import copy
import csv
import hashlib
import io
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .config import _read_yaml
from .daily_context import NY, _load, _merge, _normalise, _number, _records, _time
from .ingest.turning_inputs import refresh_turning_inputs
from .macro_rotation import (
    UNIVERSE,
    _clock,
    _next,
    build_macro_rotation_policy,
    simulate_macro_rotation,
)
from .news_events import corroborated_events, fetch_google_news
from .sessions import expected_complete_session
from .storage import write_json
from .swing_optimizer import _hash
from .turning_points import build_turning_features


def _news(rows, cutoff):
    accepted, excluded = [], []
    for row in _records(rows):
        if not isinstance(row, dict):
            continue
        available = _time(row.get("available_at"))
        if available is None or available > cutoff:
            excluded.append({"id": row.get("id"), "reason": "unavailable_at_analysis_clock"})
            continue
        provider = str(row.get("source", "")) + str(row.get("source_refs", ""))
        probability_fields = {
            "probabilities",
            "yes_probability",
            "no_probability",
            "p_cut",
            "p_hike",
        }
        if "polymarket" in provider.lower() or probability_fields.intersection(row):
            excluded.append({"id": row.get("id"), "reason": "prediction_market_excluded_news_only"})
            continue
        if row.get("trigger") is False:
            excluded.append({"id": row.get("id"), "reason": "classified_news_context_only"})
            continue
        direction = _number(row.get("direction", row.get("score")))
        if direction is None or not -1 <= direction <= 1 or not row.get("source_refs"):
            excluded.append({"id": row.get("id"), "reason": "missing_news_score_or_proof"})
            continue
        accepted.append({**copy.deepcopy(row), "direction": direction})
    return accepted, excluded


def _download_fred():
    start = (datetime.now(timezone.utc).date() - timedelta(days=120)).isoformat()
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU&cosd=" + start
    request = Request(url, headers={"User-Agent": "trading-wizzard/portfolio-news"})
    with urlopen(request, timeout=8) as response:
        body = response.read(2_000_001)
    if len(body) > 2_000_000:
        raise ValueError("FRED response exceeds bounded receipt size")
    return body, datetime.now(timezone.utc), url


def _refresh_news(paths, rows):
    """One key-free request; newly detected changes become known only at retrieval."""
    body, fetched, url = _download_fred()
    folder = (
        paths.raw
        / "portfolio-news"
        / (fetched.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8])
    )
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "response.csv").write_bytes(body)
    receipt = {
        "source": "FRED DFEDTARU",
        "url": url,
        "fetched_at": fetched.isoformat(),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "timeout_seconds": 8,
    }
    write_json(folder / "receipt.json", receipt)
    observations = []
    for row in csv.DictReader(io.StringIO(body.decode("utf-8-sig"))):
        day, value = row.get("observation_date", row.get("DATE")), row.get("DFEDTARU")
        try:
            parsed = date.fromisoformat(day)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if _number(numeric) is not None and parsed <= fetched.date():
            observations.append((day, numeric))
    observations.sort()
    if len(observations) < 2:
        raise ValueError("FRED target observations missing")
    known_dates = [
        str(row.get("observed_date", row.get("date", "")))
        for row in rows
        if (
            isinstance(row, dict)
            and row.get("date")
            and "fred" in str(row.get("source", "")).lower()
            and "DFEDTARU"
            in (
                str(row.get("source", ""))
                + str(row.get("series_id", ""))
                + str(row.get("source_refs", ""))
            )
            and _time(row.get("available_at")) is not None
            and _time(row["available_at"]) <= fetched
        )
    ]
    last_known = max(known_dates, default="")
    additions = []
    for (prior_day, prior), (day, value) in zip(observations, observations[1:]):
        if value == prior or day <= last_known:
            continue
        additions.append(
            {
                "id": "fred-dfedtaru-observed-" + day,
                "date": day,
                "observed_date": day,
                "available_at": fetched.isoformat(),
                "direction": 0.5 if value < prior else -0.5,
                "score": 0.5 if value < prior else -0.5,
                "episode_id": "fred-dfedtaru-" + day,
                "correlation_family": "monetary_policy",
                "ttl_sessions": 5,
                "source": "FRED DFEDTARU observed target change",
                "source_refs": [
                    {
                        **receipt,
                        "raw_file": str(folder / "response.csv"),
                        "previous_observed_date": prior_day,
                        "previous_value": prior,
                        "value": value,
                    }
                ],
                "publication_time_verified": False,
                "availability_basis": "actual_fetch_first_observed_not_backdated",
            }
        )
    if additions:
        write_json(paths.processed / "observed-event-history.json", rows + additions)
    return rows + additions, receipt


def _refresh_classified_news(paths, rows, role):
    """Corroborated general-news events; 'context' rows are stored but never trade."""
    fetched = datetime.now(timezone.utc)
    items, failures = fetch_google_news()
    additions = corroborated_events(items, fetched, rows)
    for row in additions:
        row["trigger"] = role == "trigger"
    if additions:
        write_json(paths.processed / "observed-event-history.json", rows + additions)
    receipt = {"source": "Google News RSS", "fetched_at": fetched.isoformat(),
               "items": len(items), "failed_queries": failures,
               "new_events": [row["id"] for row in additions], "role": role}
    return rows + additions, receipt


def _refresh(paths, report_date, market_data, analysis_clock, live):
    clock = analysis_clock or datetime.now(timezone.utc)
    fingerprints, warnings = {}, []
    config_path = paths.config / "portfolio-strategy.yml"
    config = _read_yaml(config_path)
    if not config or config.get("enabled") is not True:
        raise ValueError("accepted_portfolio_strategy_not_enabled")
    activated = _time(config.get("activated_at"))
    if (
        activated is None
        or clock < activated
        or date.fromisoformat(report_date) < activated.astimezone(NY).date()
    ):
        raise ValueError("historical_policy_not_activated")
    fingerprints[config_path.name] = {
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()
    }
    parameters, cycle = config["parameters"], config["cycle_parameters"]
    turning_parameters = config.get("turning_parameters")
    news_role = config.get("news_classifier", "off")
    if parameters["use_events"] is not True or parameters["use_capex"] is not False:
        raise ValueError("daily_portfolio_requires_accepted_news_only_policy")
    if cycle.get("defensive_after_top", False):
        raise ValueError("unaccepted_defensive_prototype_not_enabled_for_delivery")
    frozen = build_macro_rotation_policy(
        parameters, cycle_parameters=cycle, turning_parameters=turning_parameters
    )
    history = _load(
        paths.processed / config.get("portfolio_history", "portfolio-history.json"),
        fingerprints,
        warnings,
    )
    context = _load(paths.raw / f"market-context-{report_date}.json", fingerprints, warnings)
    news_path = paths.processed / config.get(
        "observed_event_history", "observed-event-history.json"
    )
    news_cache = _load(news_path, fingerprints, warnings)
    if not news_path.is_file() or not (
        isinstance(news_cache, list)
        or isinstance(news_cache, dict)
        and isinstance(news_cache.get("rows"), list)
    ):
        raise ValueError("observed_news_history_missing_or_unreadable")
    original_news = _records(news_cache)
    news_freshness, news_receipt = "cached_observations_only", None
    if live:
        try:
            original_news, news_receipt = _refresh_news(paths, original_news)
            news_freshness = "checked_FRED_target_only_other_news_not_exhaustive"
            if analysis_clock is None:
                clock = datetime.now(timezone.utc)
        except Exception as error:
            news_freshness = "unknown_refresh_failed"
            warnings.append(
                "新闻刷新失败；保留已知历史，新政策是否变化未知。 " + type(error).__name__
            )
    classified_receipt = None
    if live and news_role in ("trigger", "context"):
        try:
            original_news, classified_receipt = _refresh_classified_news(
                paths, original_news, news_role
            )
            if news_freshness.startswith("checked_FRED"):
                news_freshness = "checked_FRED_target_and_corroborated_headlines"
        except Exception as error:
            warnings.append("一般新闻分类刷新失败；只使用已知事件。 " + type(error).__name__)
    expected = expected_complete_session(report_date, now=clock)
    news, excluded = _news(original_news, clock)
    merged, reconciliation = {}, {}
    for ticker in UNIVERSE:
        old = _normalise(
            history.get(ticker, []) if isinstance(history, dict) else [], expected, clock
        )
        incoming_rows = market_data.get(ticker, []) or (
            context.get(ticker, []) if isinstance(context, dict) else []
        )
        if incoming_rows:
            incoming = _normalise(incoming_rows, expected, clock)
            rows, proof = _merge(old, incoming, allow_recent_revisions=True)
        else:
            rows, proof = old, {"source": "verified_local_portfolio_history_only"}
        if not rows or rows[-1]["date"] != expected:
            raise ValueError(ticker + ": latest_completed_session_missing")
        merged[ticker], reconciliation[ticker] = rows, proof
    last_close = _time(_clock(date.fromisoformat(expected)))
    next_open = _time(_clock(_next(date.fromisoformat(expected)), True))
    actionable = last_close <= clock < next_open
    tail = clock if actionable else None
    turning, turning_receipt = None, None
    if turning_parameters is not None:
        inputs = refresh_turning_inputs(paths, report_date, live=live)
        turning_receipt = inputs["receipt"]
        turning = {
            "parameters": turning_parameters,
            "features": build_turning_features(
                {"QQQ": merged["QQQ"], **inputs["market"]},
                inputs["fred"],
                inputs["prediction_rows"],
            ),
        }
    replay = simulate_macro_rotation(
        merged,
        config.get("start", "2017-01-12"),
        expected,
        parameters=parameters,
        cycle_parameters=cycle,
        event_rows=news,
        live_tail_clock=tail,
        turning=turning,
    )
    last, point = replay["decisions"][-1], replay["equity_curve"][-1]
    pending = replay["pending_order"]
    weights = {ticker: last["target_weights"][ticker] for ticker in ("QQQ", "TQQQ")}
    weights["cash"] = max(0.0, 1 - sum(weights.values()))
    paper = {ticker: point["weights"][ticker] for ticker in ("QQQ", "TQQQ")}
    paper["cash"] = max(0.0, 1 - sum(paper.values()))
    wac = last["tqqq_entry_wac"]
    status = "available" if actionable else "reference_only"
    plan = {
        "date": report_date,
        "status": status,
        "available_at": clock.isoformat(),
        "price_date": expected,
        "price_dates": {t: expected for t in ("QQQ", "SPY", "TQQQ")},
        "price_closes": {t: merged[t][-1]["close"] for t in ("QQQ", "SPY", "TQQQ")},
        "decision_at": last["at"],
        "price_at": point["at"],
        "next_open_at": next_open.isoformat(),
        "target_weights": weights,
        "paper_weights": paper,
        "cause": last["cause"],
        "qqq_features": last["features"]["QQQ"],
        "structural_stop": last["recovery_stop"],
        "tqqq_entry_wac": wac,
        "tqqq_gain": last["tqqq_gain"],
        "holding_sessions": last["holding_sessions"],
        "profit_target_price": (wac * (1 + cycle["migrate_gain"]) if wac else None),
        "profit_target_gain": cycle["migrate_gain"],
        "time_limit_sessions": cycle["max_holding_sessions"],
        "pending_order": pending,
        "new_tqqq_increment": (
            max(0.0, weights["TQQQ"] - paper["TQQQ"]) if pending and actionable else 0.0
        ),
        "news_only": True,
        "prediction_markets_trigger": False,
        "news_freshness": news_freshness,
        "news_refresh_receipt": news_receipt,
        "classified_news_receipt": classified_receipt,
        "turning_signal": last["auxiliary"].get("turning"),
        "turning_inputs": turning_receipt,
        "paper_holdings_only": True,
        "orders_executed": False,
        "source_fingerprints": fingerprints,
        "policy_sha256": frozen["policy_sha256"],
        "model_source_sha256": replay["source_code_sha256"],
        "reference_source": config.get("reference_source"),
        "reason": (
            "仅为纸面模型下一实际开盘的条件计划；不代表账户持仓或已下单。"
            if actionable
            else "下一开盘已经开始；保留收盘参考，不补记过去开盘成交或提供可执行旧计划。"
        ),
        "limitations": [
            "策略按原始价格单位执行，未加分红；TQQQ使用真实历史。",
            "历史新闻可得时点含保守代理，非完整当时新闻档案。",
            (
                "新闻刷新检查FRED目标利率及≥2家主要媒体交叉印证的标题；仍非完整新闻档案。"
                if news_role in ("trigger", "context")
                else "新闻刷新只检查FRED目标利率，不能保证全部政策新闻已齐。"
            ),
            "预测市场美联储赔率仅2025年后有历史；更早年份由2年期美债变化代理。",
        ],
    }
    # Persist reconciled prices so post-close provider revisions are absorbed while they are
    # still inside the recent-session window, instead of failing on every later day.
    history_path = paths.processed / config.get("portfolio_history", "portfolio-history.json")
    preserved = dict(history) if isinstance(history, dict) else {}
    if not any(str(r.get("date") or "") > expected for rows in preserved.values()
               if isinstance(rows, list) for r in rows if isinstance(r, dict)):
        preserved.update(merged)
        write_json(history_path, preserved)
    trace = {
        "date": report_date,
        "available_at": clock.isoformat(),
        "status": status,
        "policy": replay["policy"],
        "policy_sha256": replay["policy_sha256"],
        "source_fingerprints": fingerprints,
        "provider_reconciliation": reconciliation,
        "used_price_history": merged,
        "used_news": news,
        "excluded_news": excluded,
        "used_price_history_sha256": _hash(merged),
        "used_news_sha256": _hash(news),
        "turning_features_sha256": _hash(turning["features"]) if turning else None,
        "portfolio_context_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay": replay,
        "no_model_search_or_refit": True,
    }
    return plan, trace, warnings


def refresh_portfolio_context(paths, report_date, market_data, analysis_clock=None, live=False):
    """Replace the dated plan with explicit unknown on any cache/source failure."""
    if date.fromisoformat(report_date).isoformat() != report_date:
        raise ValueError("An ISO report date is required.")
    if analysis_clock is not None and (
        not isinstance(analysis_clock, datetime)
        or analysis_clock.tzinfo is None
        or analysis_clock.utcoffset() is None
    ):
        raise ValueError("Analysis clock must be aware.")
    try:
        plan, trace, warnings = _refresh(paths, report_date, market_data, analysis_clock, live)
    except Exception as error:
        clock = analysis_clock or datetime.now(timezone.utc)
        plan = {
            "date": report_date,
            "status": "unavailable",
            "available_at": clock.isoformat(),
            "target_weights": None,
            "paper_weights": None,
            "news_only": True,
            "orders_executed": False,
            "paper_holdings_only": True,
            "reason": str(error) or type(error).__name__,
        }
        trace = {
            "date": report_date,
            "status": "unavailable",
            "error_type": type(error).__name__,
            "reason": plan["reason"],
            "no_model_search_or_refit": True,
        }
        warnings = ["纸面组合上下文不可用：" + plan["reason"]]
    write_json(paths.processed / f"portfolio-plan-{report_date}.json", plan)
    write_json(paths.processed / f"portfolio-training-{report_date}.json", trace)
    return {"portfolio_plan": plan, "portfolio_training": trace, "warnings": warnings}
