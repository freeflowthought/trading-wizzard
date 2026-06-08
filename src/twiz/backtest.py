import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from .models import BacktestResult
from .storage import write_json
from .technical import rolling_mean


def max_drawdown(equity: List[float]) -> float:
    peak = equity[0] if equity else 1.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return worst


def moving_average_crossover_backtest(
    ticker: str,
    rows: List[Dict[str, Any]],
    fast: int = 20,
    slow: int = 50,
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    output_dir: Optional[Path] = None,
) -> BacktestResult:
    warnings: List[str] = []
    if len(rows) <= slow + 2:
        warnings.append("Insufficient history for moving-average crossover backtest.")
    prices = [float(row["close"]) for row in rows]
    fast_ma = rolling_mean(prices, fast)
    slow_ma = rolling_mean(prices, slow)
    signals = [
        bool(f is not None and s is not None and f > s)
        for f, s in zip(fast_ma, slow_ma)
    ]

    cost_per_turnover = (fee_bps + slippage_bps) / 10_000
    equity = [1.0]
    daily_returns: List[float] = []
    positions: List[int] = [0]
    trades = 0
    for index in range(1, len(prices)):
        previous_position = positions[-1]
        shifted_position = 1 if signals[index - 1] else 0
        turnover = abs(shifted_position - previous_position)
        if turnover:
            trades += 1
        raw_return = prices[index] / prices[index - 1] - 1.0 if prices[index - 1] else 0.0
        strategy_return = shifted_position * raw_return - turnover * cost_per_turnover
        daily_returns.append(strategy_return)
        equity.append(equity[-1] * (1.0 + strategy_return))
        positions.append(shifted_position)

    benchmark_return = prices[-1] / prices[0] - 1.0 if prices and prices[0] else 0.0
    total_return = equity[-1] - 1.0 if equity else 0.0
    volatility = pstdev(daily_returns) if len(daily_returns) > 1 else 0.0
    sharpe = mean(daily_returns) / volatility * math.sqrt(252) if volatility else 0.0
    metrics = {
        "total_return": round(total_return, 6),
        "benchmark_return": round(benchmark_return, 6),
        "max_drawdown": round(max_drawdown(equity), 6),
        "sharpe": round(sharpe, 6),
        "trades": float(trades),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "signals_shifted": 1.0,
    }

    equity_curve_path = None
    if output_dir is not None:
        equity_curve_path = str(output_dir / f"{ticker}-ma-crossover-equity.json")
        curve = [
            {"date": row["date"], "equity": round(value, 6), "position": positions[index]}
            for index, (row, value) in enumerate(zip(rows, equity))
        ]
        write_json(Path(equity_curve_path), curve)

    return BacktestResult(
        strategy_id="ma_crossover_20_50",
        ticker=ticker,
        params={"fast": fast, "slow": slow, "fee_bps": fee_bps, "slippage_bps": slippage_bps},
        metrics=metrics,
        equity_curve_path=equity_curve_path,
        warnings=warnings,
    )


def run_backtests(
    market_data: Dict[str, List[Dict[str, Any]]],
    output_dir: Optional[Path] = None,
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    return {
        ticker: moving_average_crossover_backtest(
            ticker,
            rows,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            output_dir=output_dir,
        ).to_dict()
        for ticker, rows in market_data.items()
    }
