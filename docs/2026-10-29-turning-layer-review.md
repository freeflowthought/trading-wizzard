# Turning-point layer: 10-year review (2026-10-29)

**Recommendation:** promote the market-implied turning layer to the daily plan. Keep
general-news classification as report context, not as a trade trigger. Activation is
manual (see the last section).

## What changed

| Problem | Change |
|---|---|
| Prediction markets didn't help call tops/bottoms | New `turning_points.py` layer reads traded expectations: VIX/VIX3M term structure, 2Y Treasury repricing (superseded by Polymarket Fed odds when quoted), initial claims, HYG/IEF credit. Adds a **stress-top** trim and a **volatility-capitulation bottom** entry to the cycle. |
| Only FRED rate changes became trading events | New `news_events.py` classifies Google News headlines (trade, geopolitics, financial stress, fiscal, health), requires ≥2 major publishers in 36h, and stamps availability at first corroborated fetch. Wired into the daily plan as `news_classifier: context`. |
| Capex table relied on cached filings | New `ingest/sec_capex.py` checks each issuer's SEC submissions index on every live report and re-downloads company facts only when a newer 10-Q/10-K exists. Its first run found MSFT's 10-Q filed 2026-10-28, which the cache had missed. |
| Backtest was ~9.8 years | Data re-fetched from 2014-06. Replay covers 2016-10-27 → 2026-10-27, with 260 sessions of warm-up. |

## Ten-year results (split-adjusted prices; 1 bp fee and 2 bp slippage per side)

| Strategy | CAGR | Max drawdown | Sharpe | Calmar |
|---|---:|---:|---:|---:|
| QQQ buy & hold | 19.95% | 35.62% | – | 0.56 |
| Accepted news cycle (current) | 23.71% | 31.67% | 0.88 | 0.75 |
| **+ turning layer** | **25.54%** | **25.28%** | **0.98** | **1.01** |
| + turning layer, costs ×2 | 25.28% | 25.40% | 0.97 | 1.00 |
| + turning layer, dividend-adjusted proxy | 27.33% | 25.28% | 1.05 | 1.08 |

| Window | Accepted CAGR / MDD | Turning CAGR / MDD |
|---|---:|---:|
| Train 2016-10 → 2021 | 30.20% / 29.58% | 28.33% / 22.55% |
| **Holdout 2022 → 2026-10** | 16.81% / 31.40% | **22.31% / 24.26%** |
| Last six months | 27.45% / 23.70% | 27.45% / 23.70% (unchanged) |

The layer gives up about 2pp of CAGR in the 2016–2021 bull market in exchange for 7pp
less drawdown. It earns that back and more in 2022–2026.

### Year by year (raw prices, 1× costs)

| Year | QQQ | Accepted | Turning |
|---|---:|---:|---:|
| 2016 (from 10-27) | +0.8% | +0.6% | -0.3% |
| 2017 | +31.5% | +26.9% | +27.6% |
| 2018 | -1.0% | +3.4% | -5.9% |
| 2019 | +37.8% | +30.9% | +34.5% |
| 2020 | +47.6% | +58.2% | +73.5% |
| 2021 | +26.8% | +43.5% | +30.2% |
| 2022 | -33.1% | -29.5% | -22.1% |
| 2023 | +53.8% | +66.1% | +65.3% |
| 2024 | +24.8% | +26.8% | +31.1% |
| 2025 | +20.2% | +31.6% | +25.0% |
| 2026 (to 10-27) | +17.4% | +9.1% | +26.1% |

It is not better every year. 2018, 2021 and 2025 are worse, mostly from stress trims
that the market quickly reversed. The gain comes from 2020, 2022 and 2026.

## How parameters were chosen (and what was rejected)

1. Defaults set from principle: textbook VIX/VIX3M inversion at 1.0, a 20-day price
   confirmation, and a 70% retained position matching the existing top rule.
2. **Rejected:** a 600-configuration random search on 2016–2021. Its top 20 configs had
   a median holdout CAGR of 16.3%, below baseline. They all leaned on a calm-uptrend TQQQ
   sleeve that only worked in the 2016–2021 bull market.
3. Anchored yearly walk-forward, 2019–2026: re-optimizing each year gave 27.9% CAGR vs
   27.3% for fixed defaults, at the same drawdown. Fixed defaults were kept.
4. Structural choices made by training-window Calmar: capitulation sizing 0.75, and the
   credit-drawdown trigger disabled. Credit is still used as a guard and shown in the report.
5. **Rejected:** full-TQQQ bottoms. CAGR rises to 29.5% but drawdown to 34.7%.
6. **Rejected:** release-based CPI-hot/cool and Sahm-rule events. They cost 0.4–1.3pp
   of CAGR with no drawdown benefit.
7. **Rejected:** fresh per-campaign stops and de-lever-only stops. Both reduced CAGR
   without reducing drawdown.

## Prediction markets: what is and isn't proven

Polymarket Fed odds have history only from 2025-02, and usable 20-session shifts exist
on just 36 sessions, so their backtest contribution is **zero, not proven positive**. In
the 10-year test, the 2-year Treasury yield's repricing plays the same role. The
VIX/VIX3M curve, itself an options-implied forecast, is what drives most stress trims (28 of 37
cite it; claims 9, 2Y repricing 5; reasons can overlap) and every capitulation entry. Live Polymarket Fed quotes are now saved daily
(`turning-pm-history.json`) so their contribution can be tested as history builds up.

## General news as a trigger

A causal archive was built for every one of the 825 sessions where the market had
visibly reacted by the 16:15 decision (|QQQ move| ≥ 1.5% or VIX/VIX3M ≥ 0.95). For each,
date-bounded Google News headlines were fetched and the final classifier was replayed.
It produced 34 corroborated events, including the 2018–2019 tariffs, the WHO pandemic
declaration, the Ukraine invasion, SVB, the Fitch downgrade and Liberation Day.

| Config (10y, raw) | CAGR | Max DD | Holdout CAGR |
|---|---:|---:|---:|
| Turning layer | **25.54%** | 25.28% | **22.31%** |
| + classified news as triggers | 22.92% | 25.28% | 19.26% |
| + adverse-only news | 24.4% | 25.3% | 20.8% |
| + news confirmed by same-day QQQ move | 24.2% | 25.3% | 20.3% |

Headline triggers cost 1–2.6pp of CAGR and never reduced drawdown. By the decision
clock, the market has already priced the headline. The VIX term structure is the cleaner
read of whether news matters. Classified events therefore run as
`news_classifier: context`: fetched and stored automatically, shown in the report, never
traded. Switching to `trigger` is one config value if you want to override this.

The retrospective archive is a Google search run today, not a contemporaneous feed;
ranking and later-published articles can differ from what was visible at the time.

## Stability update (2026-10-29, live)

A diagnosis of the weak years led to two changes:

- **2018:** capitulation buys were trimmed 1–5 days later by VIX re-inversions.
- **2021:** the claims rule fired four times while claims were rising off a pandemic peak.

Fix: `claims_trend_filter` (claims must also exceed their 26-week average) and
`capitulation_grace_sessions: 5` (no VIX-only trim of a fresh entry above its anchor low).

| | 10y CAGR / MDD | Train | Holdout | 2018 | 2021 | 2025 | Fills |
|---|---|---|---|---|---|---|---|
| Turning v1 | 25.54% / 25.28% | 28.33% / 22.55% | 22.31% / 24.26% | −6% | +30% | +25% | 232 |
| **Live** | **26.75% / 25.28%** | **30.73% / 22.02%** | 22.31% / 24.26% | −2% | +35% | +25% | 215 |

Grace periods of 3, 5 and 10 days all improve training; 10 days hurts 2025. Rejected:

- Two-day stress persistence: drawdown rises to 29.5%.
- One capitulation per episode, or a 10-day cooldown: these skip the April 2025 bottom.
- A 1.10 inversion threshold: 2025 improves, but 2018 and the holdout worsen.

2025 (+25% vs +32% for the old model) still suffers from the March 17 first-leg
capitulation. Evidence: `data/processed/evaluation/swing-experiments/turning-10y-20261029T013548Z/`.

## Known limitations

- The June–July 2026 drawdown (23.7%) is untouched. A TQQQ campaign met a pullback during
  which VIX/VIX3M never inverted (it peaked at 0.96). A tighter stop did not help the
  replay.
- FRED values are current vintages, and claims include seasonal revisions. Polymarket
  history is a retrospective download.
- The 2022–2026 holdout was viewed during design, both the exploratory episode review and
  the stop experiments. It is not a pristine out-of-sample test. The walk-forward is the
  more honest estimate.
- TQQQ positions carry roughly 3× exposure. "100% notional" is not 1× market risk.

Evidence: `data/processed/evaluation/swing-experiments/turning-10y-20261029T011024Z/`
(summary.json with input and source hashes).

## Activating

Copy `turning_parameters` and `news_classifier` from
`config/portfolio-strategy-turning-candidate.yml` into `config/portfolio-strategy.yml`,
set a new `activated_at`, and run
`uv run --extra providers twiz report --date today --live` after 16:15 New York time.
