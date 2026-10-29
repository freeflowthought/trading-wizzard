# Trading Wizzard

Trading Wizzard is a local, free-first market decision-support tool. It combines Reddit/news sentiment, macro monitoring, technical levels, backtests, and options context into a daily personal report and a local Streamlit dashboard.

This project does not place trades and does not provide financial advice. It is built for personal research workflows.

The Chinese report leads with a simple price-first macro framework: **structural scenarios →
rates and events → sector strength → price levels and actions**. It states a main
view, distinguishes a rebound from a confirmed trend change, and gives upside,
downside, and invalidation conditions. The data tables remain below for checking.
This is an explicit rule framework inspired by the supplied timeline, not a trained
model with an established prediction accuracy. The daily file is a concise decision
brief; complete observations and source checks remain in `DATE-details.md/html`.
Historical author comparisons are available only through the explicit research renderer.

`config/portfolio-strategy.yml` records the accepted news-only QQQ/TQQQ/cash research
strategy. Its daily reference plan is separate from actual account holdings and
does not place orders. Prediction-market quotations remain visible as context;
their additional cycle triggers are not enabled. Direction percentages explicitly
fall back to historical frequencies when conditional models do not improve the
chronological validation baseline.

For the daily live report, use `uv run --extra providers twiz report --date today --live`.
This reloads the accepted strategy and refreshes public market, options, macro,
prediction-market and discussion/news sources; it does not repeat parameter research.
Run after 16:15 New York time to use that day's completed daily bars. Missing or
inconsistent prices make the plan unavailable rather than substituting sample data.
The strategy's automatic observed-event update currently checks FRED target-rate
changes only; RSS stories remain report context, not automatically classified cycle
triggers. The cloud-capex table uses the existing local SEC history with filing and
freshness checks; this command does not yet fetch new company-facts filings.

The legacy **dip-entry / take-profit action card** remains in the research/report
code; it does not control the enabled cycle plan or replace an unavailable cycle plan.
The current user policy permits up to **100% of total account value**, without
leverage. The ceiling is configurable; the default entry split is 40% / 35% / 25%
of that budget. A weak structural hypothesis uses a separate configured provisional
allowance (initially 2%). Initial capital is constrained by macro and
event main cases; immediate buying remains 0% until the observed price confirmation.
Later additions require new higher lows/retests and a new report; their displayed
prices are cost stress references, not descending limit orders. The 40% / 30% / 30% trims refer to the original shares
actually acquired in that dip campaign, excluding existing core holdings. Stop and
target references come from current verified price structure and moving averages.
Historical author targets never enter live position calculations. Structural invalidation can
remove entries or reduce enabled capital further. No live account or execution
system is connected.
Edit `config/allocation.yml` to set the capital ceiling, model-risk budget, and
conditional allocation fractions. The earlier 10% / 0.5% profile is no longer a
global restriction. Gaps and execution delays can exceed a modeled stop loss.

## Quick Start

```bash
uv sync --dev --extra providers
uv run --extra providers twiz report --date today --live
uv run twiz report --date today
uv run twiz dashboard
```

The default report path is `reports/YYYY-MM-DD-report.md`.
Use `--extra providers` with live commands so uv keeps the optional Yahoo/Reddit
dependencies installed. `uv run python -m pytest` runs the test suite.

## Free-First Defaults

- Live market data uses yfinance. Failed requests or invalid bars produce visible warnings and unavailable data; live reports never substitute generated prices. Offline commands reuse local artifacts, with deterministic sample OHLCV only when no cached market artifact exists. Sample prices are labeled demo data and do not produce trading levels, playbooks, or reported backtest returns. Each watchlist row identifies its source and last valid bar date; older artifacts without provenance remain unverified until refreshed.
- Reddit uses PRAW when credentials are present and `--live` is passed. Without Reddit credentials, live ingest uses expanded Reddit RSS search/listing feeds. Optional Jina Search support is available only when a Jina API key is configured.
- FinBERT is used when `transformers` and model weights are available; otherwise a deterministic financial lexicon scorer is used with a clear model label.
- Options data uses available chain snapshots where possible. Greeks can be calculated locally, but true real-time options flow requires paid OPRA-style data.
- Macro data comes from RSS/Fed/SEC/FRED adapters when live sources are enabled and credentials are available.
- Key-free fallbacks now collect Treasury 2/10/30-year yields, FRED public-CSV credit/Brent observations, and ECB JPY/KRW reference cross-rates. Each series keeps its own dated history and source; missing observations are not interpolated. ECB references are daily rather than live FX. Oil observations over three days old do not drive the current scenario.
- Polymarket uses automatic discovery through the public Gamma API. Each live refresh combines bounded active-event scanning and topic searches, then screens stock relevance, transmission channel, deadline and quote quality. The default keeps six events across distinct channels; different deadlines are compared separately. `config/prediction-markets.yml` controls the screen; its old market IDs are legacy study references. The report retains exact questions, rules, timestamps, liquidity and spread. Raw candidates and rejection reasons are saved in `prediction-discovery-DATE.json` and included in report archives. Invalid quotes remain unknown. Event probabilities are not stock-return probabilities or verified geopolitical facts.
- Hyperliquid supplies a read-only BTC/ETH perpetual-market reference for evenings and weekends: mark/oracle prices, provider-day change, raw funding, open interest and volume. Its response does not include a quote timestamp, so retrieval time and unknown quote age are shown separately. These references never substitute for QQQ's regular-session close or change equity allocations automatically.
- On-chain **stock** references are restricted to Lighter's exact SPY and QQQ perpetual contracts (market IDs 128 and 129). BTC/ETH references and all other report sections remain. Crypto and stock-reference sources fail independently; missing quote timestamps stay unknown. Mark, index and last-trade prices are different fields, and none is an official ETF close.

## Commands

```bash
twiz ingest --source all --live
twiz ingest --source predictions --live
twiz ingest --source cross-market --live
twiz analyze --date today
twiz backtest --date today
twiz report --date today
twiz dashboard
```

`--live` is intentionally opt-in. Without it, commands prefer existing local artifacts and offline fallbacks.
Live collection prints its current stage. Reddit RSS uses combined watchlist searches
and a bounded request/time budget instead of one request per ticker alias. Missing
social or macro coverage is reported explicitly.

`--date` bounds market history to that day. Historical reports skip current options,
fundamentals, and earnings fetches; these providers do not supply point-in-time
snapshots. FRED observations are bounded by date but can contain later revisions.
This is a daily research report, not a point-in-time backtesting database.
Sentiment uses dated, unique discussions from the preceding seven days. The default
scorer is a simple financial lexicon; `--finbert` requests the optional FinBERT model.

## Market-data freshness

Actionable equity analysis requires the latest completed US regular session, using
New York time and the regular holiday calendar. After the close plus a 15-minute
buffer, yesterday's data cannot pass as today's data. Early-close days use this
buffer conservatively against 16:00; unscheduled exchange closures are not inferred.
If either SPY or QQQ is incomplete, the report blocks priced entry/exit instructions.
Intraday and after-hours quotes are not treated as completed daily bars.

Yahoo can return a null daily Close even when its regular-session metadata has a
valid closing quote. The adapter uses that metadata only when its timestamp matches
the completed session, the close is inside the day's range, and the completion
buffer has elapsed. The report records `regular_market_metadata` as the source.
Historic adjusted bars and a current closing quote remain explicitly identifiable.
Options context must match the underlying close/date before being used.

For an independent consolidated US equity feed, compare
[Alpaca SIP](https://docs.alpaca.markets/us/docs/about-market-data-api) and
[Massive](https://massive.com/pricing?product=stocks). Alpaca's free live feed is
IEX-only; its FAQ permits historical SIP queries ending at least 15 minutes before
now, which can suit an after-close report. Real-time SIP requires the appropriate
subscription. These services are researched alternatives, not configured providers
in this project yet. A live quote and an official regular-session close must still
be distinguished when changing providers.

[Uniswap](https://developers.uniswap.org/docs/ecosystem/subgraphs/overview),
[PancakeSwap](https://developer.pancakeswap.finance/apis/subgraph) are researched
alternatives; [Lighter](https://apidocs.lighter.xyz/reference/orderbookdetails) is
enabled for the two equity references above. Pool prices require chain/token-contract
identity and index freshness checks; perpetuals require the exact contract definition.
Their prices are not interchangeable with shares on Nasdaq.

## Shunge Playbook Layer (added 2026-10)

The model was extended with the rule system distilled from a followed
trader's published prediction timeline (the "Shunge" calls):

- **Event scan (`twiz events`)** — his calls are event-anchored, so the
  pipeline builds a 45-day forward calendar first: FOMC, CPI, jobs
  reports, options expiry / quad witching, US elections, Jackson Hole,
  per-ticker earnings (live via yfinance), plus manual events from
  `config/events.yml` (product launches, lockup expiries). Each event
  carries the standing behaviour rule for its window.
- **Macro gate** — the 10Y Treasury is the master switch: >= 5.00% is
  risk-off, >= 4.80% or a fast +0.25pt rise in ~20 sessions is caution,
  HY spreads widening is the second tripwire. The gate scales the first
  staged entry; risk-off blocks the action card's new buys. FRED rows keep history so the
  gate can measure drift (needs `FRED_API_KEY` and `--live`).
  Missing or stale observations leave this gate unknown and withhold suggested
  position sizes.
- **Drawdown bands + round numbers** — levels are anchored to the
  recent peak (-2.5% / -5% / -8% / -12% / -20% bands) and pivot clusters
  now merge with psychological round numbers, the confluence his
  supports and targets show.
- **Options event context** — the ATM straddle implied move (his
  pre-earnings "expected swing"), put/call open-interest ratio and the
  put/call walls are computed from the full expiry-chain snapshot.
  Open interest alone does not identify directional demand or a future squeeze.
- **Farmer's ladder output** — every report now carries, per ticker:
  staged entries (batch 1/2/3 at descending supports and the -8% band),
  staged trim targets, a sell-put strike zone just under strong
  support, the next event with its branch rule, and seasonal context
  (September is flagged high-risk).

The comprehensive report also includes a **timeline pattern / extreme-scenario**
section, curated in `config/shunge.yml`. It separates medium-term direction from
the short-term path; structural failure from ordinary pullbacks; price from time
confirmation; macro events from their transmission to prices; sector breadth;
event volatility from direction; and trading inventory from core holdings.
Each observation retains its source date and excerpt location.

The October 21 QQQ case freezes calculations at that date using Yahoo history with
`auto_adjust=False` (Yahoo's historical split handling still applies). A range-height stress
projection from 762.86 through 704–705 gives 646.14, near the author's 650 scenario;
April's 645–655 trading area is a separate historical reference. This is one possible
reconstruction, not proof of the author's formula or a forecast with a measured hit
rate. The 200-day and 50-week averages are intermediate checks, not reasons to jump
directly to 650. The source calls 704 a prior-Friday low, while Yahoo's October 16
low is 707.23; the report preserves that discrepancy.

The dated case is study-only. The live report uses `structure.py` to recompute
QQQ/SPY support, deep and extreme zones from two years of verified OHLCV. It selects
recent repeated price structure, long daily/weekly averages, previous traded areas
and conditional range-height projections. Candidate zones are frozen before a
five-session confirmation window, and pivots need three observed right-hand bars.
Related moving averages count as one evidence family; confluence is not a win rate.
Historical crash regions are separate tail-stress references rather than automatic
dip targets. Fixed lookbacks, ATR buffers and clustering thresholds are disclosed
engineering choices, not fitted author formulas.

`scenarios.py` joins the verified, automatically selected Polymarket contracts with
observed yields, credit, FX and matching-window sector strength. No particular Iran
contract is required. Ambiguous, negated, auxiliary or compound policy questions
remain background observations rather than automatically changing direction.
The user's **65%** threshold selects
Yes or No as an event's main case. Event probabilities are neither averaged nor
relabelled as QQQ-return probabilities. A supportive main case can allow a limited
pre-event probe after price confirmation; an adverse case reduces the allowance to
zero, and macro risk-off still wins. Decision, price zone and account budget are
shown together at the top of the comprehensive report, without charts.

`config/analyst-patterns.yml` compares Eagle and Shunge with dated evidence. Four
Eagle exports contain 163 distinct posts after 31 duplicate removals; four headers
remain undated, and the dated range is May 4–October 24, 2026, rather than a proven
full-year sample. Historical learning replays conservatively use the preceding
completed US session because publication timezones are unverified. MA20/MA60 and a
public nine-bar counter are explicitly proxies. **20MV remains undefined**; the
Eagle-specific low-nine formula and full DeMark implementation are not claimed.
The channel's public SMC description supports sweep/reclaim/retest as a method
clue; unavailable captions and partial machine video summaries do not prove a formula.

Artifacts: `data/processed/events-YYYY-MM-DD.json`,
`macro-gate-YYYY-MM-DD.json`, `playbook-YYYY-MM-DD.json`.

## V2 Layers (June timeline dimensions, added 2026-10)

- **Cash-flow quality screen (`fundamentals.py`)** — his Mag-7
  diagnosis as a score: negative FCF, capex beyond operating cash flow
  and fresh share issuance (the ORCL/NVDA pattern) flag a name as a
  burner/distress. Positive cash-flow names with balance-sheet pressure are labeled
  pressured instead of cash burners; these scores are not automatic sell signals.
  Net-cash buyback names score fortress. Live via yfinance statements
  (`twiz report --live`); artifacts in `fundamentals-YYYY-MM-DD.json`.
- **IPO tracker (`ipo.py`, `config/ipos.yml`)** — tracks verified issuer terms;
  early release and lockup dates must be explicit, with a source. The example
  SPCX terms remain unverified, so they produce no trading signals.
- **Political cycle (`events.political_context`)** — identifies midterm and
  presidential election windows. A calendar date does not establish direction;
  policy-reversal and geopolitical views remain conditional hypotheses. Binary
  questions can use Yes/No market prices, with unknown when no valid data exists.
- **Leadership rotation (`playbook.leadership_snapshot`)** — 20-day
  health of the hardware leaders (MU, INTC, QCOM, NVDA, AMD, SNDK)
  vs SPY. Weakening leaders constrain semiconductor exposure; the price screen
  does not establish when the industry cycle ends.
- **Trend gate** — index playbooks now carry the flip level (his
  7440-7460 rule): nearby breakout resistance and downside support
  are conditional checkpoints alongside the measured technical regime.
- **Geopolitics overlay** — manual events with kind `geopolitics` in
  `config/events.yml` light up his de-escalation chain
  (oil down -> cruise/airlines) on beneficiary tickers.

These playbook rules are research heuristics, with no validated predictive edge.
Calendar dates marked approximate require confirmation. IPO rows require
`verified: true` and a `source_url`; release/lockup windows require explicit terms.
The bundled SPCX row is unverified and produces no supply events or price signals.
Missing fundamental fields produce an unknown screen, and leadership requires
enough price history and a usable benchmark. Options positioning is measured across
one expiry's chain; an implied move is associated with earnings only when that expiry
covers the event. Level confluence counts count inputs, not independent price touches.

The main brief compares sector returns over matching 20-session windows and uses
daily price/SMA evidence for direction. Its horizons are conditional research views;
personal holdings, tax lots and risk limits are not connected. The source's index
targets such as 7640/8000 refer to SPX (`^GSPC`), as confirmed by the user, and must
never be reused as SPY ETF prices. Old target numbers are not current trading levels.

To refresh only event probabilities, run
`uv run --extra providers twiz ingest --source predictions --live`, then rerender with
`uv run --extra providers twiz report --date today`. Full live reports fetch these
contracts automatically. Replace expired contract IDs as events roll forward; the
adapter never silently substitutes a different question or horizon.

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

Without Reddit credentials, the tool uses expanded Reddit RSS fallback. It searches ticker terms and also scans subreddit listing feeds such as `new` and `hot`. This can still produce sentiment counts, but engagement metadata is weak, so those signals are lower quality than PRAW results.

No-key Reddit fallback controls:
- `REDDIT_RSS_EXPANDED=true` scans subreddit listing RSS feeds after ticker search RSS.
- `REDDIT_RSS_SORTS=new,hot` controls which subreddit listing feeds are scanned.

Optional keyed Jina fallback controls:
- `JINA_SEARCH_ENABLED=false` keeps Jina Search off by default.
- `JINA_API_KEY` is required for Jina Search in this environment.
- `JINA_MAX_QUERIES` limits the number of ticker search queries per ingest run.

## Legal Notes

Respect the terms and rate limits of each data source. In particular, yfinance/Yahoo data is intended for personal research use, Reddit API usage must follow Reddit developer terms, and CBOE quote-table pages should not be automated for extraction.

## Troubleshooting

If `uv` warns that `VIRTUAL_ENV` points at `/Users/devacct/Documents/trading-wizzard/.venv`, your terminal still has the old moved virtualenv activated. Moving an environment can also leave installed launchers pointing to its old folder, causing `Failed to spawn: pytest` even when pytest is installed. Run:

```bash
deactivate 2>/dev/null || true
cd /Users/devacct/coding/trading-wizzard
uv sync --dev --extra providers --reinstall
uv run --extra providers python -m pytest
```

Skip manual activation and run commands directly with `uv run ...` from the project
folder. The shell warning disappears after `deactivate` (or `unset VIRTUAL_ENV`);
`--active` would target the obsolete environment and is not the repair.

## Market-implied turning layer (active since 2026-10-29)

`twiz.turning_points` adds an optional layer to the QQQ/TQQQ/cash cycle. It reads
traded expectation markets rather than price alone. Each completed session gets
one dated feature row:

- **VIX/VIX3M term structure**: inversion (≥1.0) marks stress. A capitulation is an
  inversion peak ≥1.05 within 15 sessions that has since normalized to ≤0.97.
- **Two-year Treasury repricing**: a 20-session rise ≥35bp is hawkish stress. When a
  Polymarket Fed-decision quote exists, its 20-session change in expected next-meeting
  move supersedes the proxy. Quotes expire with their meeting.
- **Initial claims**: a 4-week average ≥20% above its 52-week low. Rows become usable
  at the Thursday 08:30 release. FRED daily values are used from the next session.
- **HYG/IEF credit drawdown**: computed and shown, and it guards capitulation entries.
  It is disabled as a trim trigger because it lowered training-window Calmar.

Rules applied to the existing cycle:

- **Stress top**: any stress reason plus a QQQ close below its 20-day average trims to
  70% QQQ and removes TQQQ.
- **Capitulation bottom**: TQQQ 75% / QQQ 25% once the drawdown threshold, a higher
  low and a close above the prior three-day high also hold.
- **Leverage gate**: price-only TQQQ bottoms below the 200-day average need a recent
  washout first.

When `turning` is absent the simulator is byte-identical to the accepted policy. The
golden digests in `tests/test_macro_rotation.py` pin this.

Ten-year replay, 2016-10-27 to 2026-10-27, split-adjusted prices, 1×/2× costs:

| Strategy | CAGR | Max drawdown | Sharpe |
|---|---:|---:|---:|
| QQQ buy & hold | 19.95% | 35.62% | — |
| Accepted news cycle | 23.71% | 31.67% | 0.88 |
| + turning layer (v1) | 25.54% | 25.28% | 0.98 |
| + claims-trend filter and 5-day capitulation grace (live) | 26.75% | 25.28% | 0.98 |
| live, 2× costs | 26.52% | 25.40% | 0.97 |
| 2022–2026 holdout: accepted / live | 16.81% / 22.31% | 31.40% / 24.26% | 0.69 / 0.90 |

How it was selected:

- Principle-based defaults came first.
- A 600-configuration random search on 2016–2021 overfit. Its top 20 had a median
  holdout CAGR of 16.3%. An anchored yearly walk-forward over 2019–2026 found that
  re-optimizing did not beat fixed defaults (27.9% vs 27.3% CAGR, same drawdown).
- Fixed values were then chosen by training-window Calmar.
- A calm-uptrend TQQQ sleeve helped training years but hurt the holdout, so it stays off.
- Release-based CPI and Sahm-rule events lowered returns, so they remain context only.

The six-month window ending 2026-10-27 is unchanged by the layer, with a 23.7%
drawdown. A TQQQ campaign met a pullback during which the term structure never
inverted. Neither a fresh campaign stop nor a de-lever-only stop improved the replay.

### General news and filings

`twiz.news_events` classifies Google News RSS headlines into trade-policy,
geopolitics, financial-stress, fiscal and public-health events:

- Threats, conditionals, explainers and mixed headlines are excluded.
- An event needs at least two established publishers within 36 hours.
- It becomes available at the first fetch that sees that corroboration, never backdated.
- Rows share the FRED-event schema. `news_classifier: context` stores them for the
  report. `trigger` lets them use the existing threshold, TTL and once-per-episode rules.

`twiz.ingest.sec_capex` checks each cloud issuer's SEC submissions index on every live
report. Company facts are downloaded again only when a newer 10-Q/10-K exists. A filing
that the XBRL facts API has not yet absorbed is shown as pending, never replaced by the
prior quarter.

Stability update (same day): claims count only while their 4-week average is above
its 26-week average. This removes the January–February 2021 false alarms, when claims were
rising off a pandemic peak. For five sessions after a capitulation entry, a VIX
re-inversion alone cannot trim while QQQ holds the anchor low. This removes the 2018
buy-sell round trips. Waiting two days to confirm stress, one capitulation per episode,
and a cooldown were rejected: they raised drawdown or skipped the April 2025 bottom.

Activated on 2026-10-29: `config/portfolio-strategy.yml` carries `turning_parameters`
and `news_classifier: context`, and the daily plan reconciles and persists
`portfolio-history.json`. Remove both keys and set a new `activated_at` to return to the
previous policy.

## Reproducible v2 evaluation

With the local datasets prepared, run:

```bash
uv run --extra providers twiz evaluate --protocol config/evaluation-v2.yml --dataset-dir data/processed/evaluation/datasets
uv run --extra providers twiz audit-evaluation --run <printed-experiment-directory> --hash <printed-SHA256>
```

The runner makes a new ignored archive containing the complete project source tree,
protocol, datasets and runtime fingerprints. It selects models using copied pretest
inputs, seals that selection, then evaluates copied full data in a separate process.
It fetches no external data. Results, logs and summaries are hashed; v1 archives remain
unchanged. An interrupted folder is incomplete, does not prove a process is running,
and is never restarted automatically. A completed experiment does not establish the
requested improvement in forecasting accuracy.

## Swing parameters and learned event weights

`config/evaluation-swing.yml` declares the swing research task, bounded parameter
search, fixed account/cost limits and chronological selection. Event thresholds,
zone widths, confirmations, stops, targets, holding periods and allocation splits
are candidate variables. That legacy protocol preserves its original 10% control
for reproducibility; it does not set the current user's allocation ceiling.
Forecast deadlines and scoring definitions stay fixed. A wider forecast zone pays an explicit width
penalty; it cannot earn promotion just by covering more of the future path.

The production structure builder accepts a validated `parameters` mapping and
records the actual values used for each result. Allocation settings now control
entry offsets, stop buffers, target R multiples and both entry/trim fractions.
The default production values remain initial engineering choices until a separate
chronological swing comparison supports promotion.

`config/portfolio-swing-study.yml` defines the newer full-capital comparison.
`twiz.portfolio_swing` permits targets up to 100%, complete exits, retained positions
at intermediate tops, and staged recovery. Each QQQ or SPY account is compared with
its own 100% buy-and-hold account using the same dates and execution costs; the two
accounts are evaluated separately. Raw-price excess return and a conservative
dividend-reinvested benchmark reference are both retained. Execution tests alone
do not establish that timing beats holding.

`twiz.portfolio_structure` separates a confirmed price recovery from the prior
long-trend risk state, and requires a profitable wave plus structural weakening
before trimming. `twiz.sector_rotation` evaluates QQQ/SPY/SMH in one shared account
and records both QQQ holding and an exposure-matched static sector basket. Sector
exposure gains are reported separately from active allocation gains. The user's
current objective prioritizes profit and permits moderately greater drawdown;
`config/sector-profit-priority-study.yml` records that revised research protocol
without changing the earlier strict studies. Its drawdown sensitivity settings
are research variables, not user-approved personal risk limits. None of these
research variants is automatically promoted to the daily action card.

The research helpers compare three approaches: price/macro inputs alone, an
explicit event-impact score, and learned event/market weights. `event_direction`
keeps event levels, probability revisions, horizon alignment, shared-risk groups
and factor contributions auditable. Its score is not an equity probability.
`event_bridge` learns regularized weights, purges immature labels, separates
training/calibration/validation event episodes, and tests against a prior and a
price/macro-only model. Dense quotes from one event share one episode's weight.
Missing history or failed validation yields an unavailable probability; it never
substitutes an event's 65% quote for a stock's 65% chance of rising.

The bridge's auxiliary probability target is the completed close 20 sessions ahead
exceeding the origin close. This definition is recorded with every estimate and is
separate from bottom/top error, net swing profit, drawdown and trade hit rate.
Neither helper is automatically promoted to the report's trading policy.
