# Trading Wizzard

Trading Wizzard is a local, free-first market decision-support tool. It combines Reddit/news sentiment, macro monitoring, technical levels, backtests, and options context into a daily personal report and a local Streamlit dashboard.

This project does not place trades and does not provide financial advice. It is built for personal research workflows.

## Quick Start

```bash
uv sync --dev
uv run twiz report --date today
uv run twiz dashboard
```

The default report path is `reports/YYYY-MM-DD-report.md`.

## Free-First Defaults

- Market data uses a provider adapter. If `yfinance` is unavailable or fails, the tool writes deterministic sample OHLCV so the pipeline still runs.
- Reddit uses PRAW only when credentials are present and `--live` is passed.
- FinBERT is used when `transformers` and model weights are available; otherwise a deterministic financial lexicon scorer is used with a clear model label.
- Options data uses available chain snapshots where possible. Greeks can be calculated locally, but true real-time options flow requires paid OPRA-style data.
- Macro data comes from RSS/Fed/SEC/FRED adapters when live sources are enabled and credentials are available.

## Commands

```bash
twiz ingest --source all --live
twiz analyze --date today
twiz backtest --date today
twiz report --date today
twiz dashboard
```

`--live` is intentionally opt-in. Without it, commands prefer existing local artifacts and offline fallbacks.

## Configuration

Edit `config/watchlist.yml` for tickers, aliases, and Reddit query terms. Edit `config/feeds.yml` for macro/news RSS sources and FRED series IDs.

## Environment Setup

Copy `.env.example` to `.env` and fill only the providers you want to enable:

```bash
cp .env.example .env
```

Required only for fuller Reddit sentiment:
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`

Required only for FRED macro series:
- `FRED_API_KEY`

Recommended for SEC EDGAR feeds:
- `EDGAR_IDENTITY`, formatted as `Your Name your.email@example.com`

Without Reddit credentials, the tool uses Reddit RSS fallback. That is useful for a quick read, but it has weaker coverage and limited engagement metadata.

## Legal Notes

Respect the terms and rate limits of each data source. In particular, yfinance/Yahoo data is intended for personal research use, Reddit API usage must follow Reddit developer terms, and CBOE quote-table pages should not be automated for extraction.

## Troubleshooting

If `uv` warns that `VIRTUAL_ENV` points at `/Users/devacct/Documents/trading-wizzard/.venv`, your terminal still has the old moved virtualenv activated. Run:

```bash
deactivate 2>/dev/null || true
cd /Users/devacct/coding/trading-wizzard
source .venv/bin/activate
```

Or skip manual activation and run commands directly with `uv run ...` from `/Users/devacct/coding/trading-wizzard`.
