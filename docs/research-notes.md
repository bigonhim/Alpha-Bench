# Alpha Foundry: research notes

These notes are the knowledge base behind Alpha Foundry's template library, Doctor rules and default check thresholds. They were compiled on 2026-09-26 from WorldQuant's public educational content, community write-ups and the *101 Formulaic Alphas* paper.

Anything marked **(verify)** could not be confirmed against an official BRAIN source. The matching value in the app lives in `backend/alphafoundry/catalog/checks.yaml` or `operators.json`, where you can edit it.

---

## 1. WorldQuantCareers YouTube channel

Channel: <https://www.youtube.com/@WorldQuantCareers> (channel id `UCde7Y_tiRThbgWuoaTggasQ`, RSS: `https://www.youtube.com/feeds/videos.xml?channel_id=UCde7Y_tiRThbgWuoaTggasQ`).

The channel has two educational series, both hosted by Nitish Maini (Chief Strategy Officer). Its "WorldQuant BRAIN" playlist is mostly short IQC and sign-up promos.

### Quantcepts (short concept videos)
Titles: What is an Alpha · How to Assess an Alpha · Price Volume Data · What is Market-Neutral Investing · How Quants Can Partner with AI · How Do You Make Risk Neutral Alphas · Why Eliminate Risk Exposure · What are Factor Risks · Diversity · How to Diversify Alphas · What Does a Delay-0 Alpha Look Like · Why Use Delayed Data · Holding Periods · Seasonality · Momentum Alphas · Types of Alpha Ideas · Options Data · Sentiment Data · Company Fundamentals.

### Learn2Quant (11 lessons, "practical examples you can test on BRAIN")
1. A Beginner's Guide to Quantitative Finance Research and Alpha Creation
2. Creating a Quant Alpha
3. How Good is Your Alpha? A Metrics-Based Approach
4. Alpha Examples by Data Category: Part 1
5. Alpha Examples by Data Category: Part 2
6. Alpha Examples by Idea Type. Chapters: variety of ideas, reversion example, momentum example, hypothesis, implementing the alpha, seasonality.
7. Alphas by Holding Frequencies and Delays
8. The Power of Diversity
9. Risk Management
10. Advanced Research Ideas
11. Learn2Quant and BRAIN

### How Alpha Foundry uses the series
The series frames alpha research along a few axes, and Alpha Foundry uses the same axes:
- **Idea types:** reversion, momentum and seasonality, plus value, quality and similar.
- **Data categories:** price-volume, fundamentals, sentiment and options.
- **Holding period and delay.**
- **Risk neutralization.**
- **Diversity.**

Every template and alpha in Alpha Foundry is tagged with `idea`, `category`, `horizon`, `delay` and `neutralization`. The miners enforce diversity across these tags, which is also the practical defence against the SELF_CORRELATION check.

---

## 2. Syntax facts (BRAIN Fast Expression)
- **Variables:** multi-statement expressions are allowed; the last statement is the alpha. Example: `a = rank(x); b = rank(y); a + b`.
- **Conditionals:** `cond ? a : b` works, as does `if_else(cond, a, b)`.
- **Vector fields:** these (news, social) must be reduced with `vec_avg` or `vec_sum` before they can be used as a matrix.
- **`trade_when(trigger, alpha, exit)`:** keeps the previous value while `trigger` is false. When `exit` > 0 the value becomes NaN (the position closes); with `exit = -1` the position never closes.
- **Fitness:** `Sharpe * sqrt(abs(Returns) / max(Turnover, 0.125))`.

---

## 3. Submission checks (delay-1 USA defaults; all editable in `checks.yaml`)
| Check | Pass rule |
|---|---|
| LOW_SHARPE | Sharpe ≥ 1.25 (delay-0: 2.0, **verify**) |
| LOW_FITNESS | Fitness ≥ 1.0 (delay-0: 1.3, **verify**) |
| LOW_TURNOVER / HIGH_TURNOVER | 1% ≤ turnover ≤ 70% |
| CONCENTRATED_WEIGHT | max single-stock weight < 10% |
| LOW_SUB_UNIVERSE_SHARPE | sub-universe Sharpe ≥ 0.75·√(sub/univ)·Sharpe (**verify**) |
| SELF_CORRELATION | PnL correlation < 0.7 with every submitted alpha, or Sharpe ≥ 10% above the correlated alpha |
| (community rule of thumb) | max drawdown < 50% |

Worked examples:
- **Sub-universe:** a TOP3000 alpha with Sharpe 1.6 needs a TOP1000 Sharpe of at least 0.75 × 0.577 × 1.6 ≈ **0.69**.
- **Fitness:** Sharpe 1.5 and returns 10% at turnover 40% gives fitness 0.75 (fail). At turnover 12.5% it gives 1.34 (pass). Turnover below 12.5% brings no further gain because of the floor.

---

## 4. Idea and template catalog

### Generic wrappers
| Purpose | Pattern |
|---|---|
| Rank within peers | `group_rank(ts_rank(<field>/cap, 252), subindustry)` |
| Fundamental z-score | `ts_zscore(ts_backfill(<fundamental>, 120), 252)` |
| Outlier control | `rank(winsorize(S, std=4))` |
| Neutralize | `group_neutralize(S, subindustry)`; `regression_neut(S, <factor>)` |
| Size-bucket neutral | `group_neutralize(S, bucket(rank(cap), range="0.1,1,0.1"))` |
| Smooth / lower turnover | `ts_decay_linear(S, 10)`, `ts_mean(S, 5)`, `hump(S, hump=0.01)` |
| Event gate | `trade_when(volume > adv20, S, -1)` |
| Deviation from average | `ts_av_diff(ts_backfill(<field>, 20), 60)` |
| Combine | `rank(S1) + rank(S2)`, `rank(S1) * rank(S2)`, `rank(C) > 0.5 ? S1 : S2` |

A full improvement chain:
```
raw = -ts_delta(close, 5) / close;
norm = rank(winsorize(raw, std=4));
neutral = group_neutralize(norm, sector);
smooth = ts_decay_linear(neutral, 10);
trade_when(volume > adv20, smooth, -1)
```

### Reversal (short term)
Community IQC reports say this category dominated submissions.
- `(high + low)/2 - close` (TOP3000, market-neutral, turnover about 50%)
- `rank(-ts_delta(close, 5))`
- `group_neutralize(-ts_sum(returns, 3), subindustry)`
- `trade_when(volume > adv20, rank(-ts_delta(close, 3)), -1)`
- `decline_pct = (vwap - close)/close; decline_pct / min(ts_decay_linear(rank(ts_arg_max(close, 30)), 1), 0.15)`

### Momentum
- `rank(ts_delay(ts_sum(returns, 231), 21))` (12-minus-1 month)
- `group_rank(ts_delay(ts_sum(returns, 231), 21), subindustry)` (industry-relative)
- `rank(close / ts_max(high, 252))` (closeness to the 52-week high)

### Value
- `-ts_zscore(enterprise_value/ebitda, 63)` (industry-neutral, turnover about 25%)
- `group_rank(ts_backfill(ebitda, 120)/enterprise_value, industry)`
- `group_rank(ts_rank(ts_backfill(sales, 120)/cap, 252), subindustry)`

### Quality / profitability
- `group_rank(ts_backfill(operating_income,120)/ts_backfill(assets,120), industry)` (ROA)
- `group_rank((ts_backfill(sales,120)-ts_backfill(cogs,120))/ts_backfill(assets,120), industry)` (gross profitability)
- `rank(ts_backfill(cashflow_op,120)/ts_backfill(assets,120))`

### Growth
- `ts_zscore(ts_backfill(sales, 120), 252)`
- `group_rank(ts_delta(ts_backfill(operating_income,120), 252)/ts_backfill(assets,120), industry)`

### Accruals / investment
- `-group_rank((ts_backfill(operating_income,120) - ts_backfill(cashflow_op,120))/ts_backfill(assets,120), industry)`
- `-rank(ts_delta(ts_backfill(assets,120), 252) / ts_delay(ts_backfill(assets,120), 252))` (asset growth)

### Liquidity / volume
- `rank(ts_mean(abs(returns)/(volume*close), 20))` (Amihud illiquidity). This leans toward small caps, so combine it with a cap-bucket neutralization.
- `-ts_corr(rank(close), rank(volume), 10)`
- `-rank(ts_mean(volume, 20)/sharesout)`

### Volatility / low risk
- `-rank(ts_std_dev(returns, 60))`
- `-rank(ts_mean((high-low)/close, 20))`
- `-rank(ts_max(returns, 21))` (lottery / MAX effect)

### Seasonality
- `rank(ts_delay(ts_sum(returns, 21), 231))` (same month last year). Add lags of 483 and 735 days for a multi-year version.

### Size / leverage
- Use size as a neutralization group rather than a signal: `group_rank(S, bucket(rank(cap), range="0.1,1,0.1"))`
- `-group_rank(ts_backfill(debt,120)/ts_backfill(assets,120), industry)`
- `-rank(ts_delta(ts_backfill(liabilities,120)/ts_backfill(assets,120), 252))`

### BRAIN-only data
These can't be backtested locally; test them on BRAIN.
- **News:** `avg_news = vec_avg(nws12_afterhsz_sl); rank(ts_sum(avg_news, 60)) > 0.5 ? 1 : rank(-ts_delta(close, 2))`
- **Social buzz:** `buzz = ts_backfill(-vec_sum(scl12_alltype_buzzvec), 20); ts_av_diff(buzz, 60)`
- **Options:**
  - `rank(implied_volatility_call_120 - implied_volatility_put_120)`
  - `rank(implied_volatility_call_120 / parkinson_volatility_120)`
- **Analyst** (field names vary; import your field catalog):
  - Revisions: `rank(ts_delta(e,21)/abs(ts_delay(e,21)))`
  - Dispersion: `-rank(std/abs(mean))`
  - Surprise: `rank((eps - est)/close)`
- **Event-driven:** `trade_when(days_from_last_change(eps) < 5, rank(s), days_from_last_change(eps) > 63)`

---

## 5. 101 Formulaic Alphas (Kakushadze, 2015, arXiv:1601.00991)

### Operator mapping
| Paper | BRAIN |
|---|---|
| `delay(x,d)` | `ts_delay(x,d)` |
| `delta(x,d)` | `ts_delta(x,d)` |
| `correlation(x,y,d)` | `ts_corr(x,y,d)` |
| `covariance(x,y,d)` | `ts_covariance(x,y,d)` |
| `decay_linear(x,d)` | `ts_decay_linear(x,d)` |
| `SignedPower(x,a)` | `signed_power(x,a)` |
| `Ts_Rank(x,d)` | `ts_rank(x,d)` |
| `ts_argmax/argmin` | `ts_arg_max/ts_arg_min` |
| `sum/product/stddev(x,d)` | `ts_sum/ts_product/ts_std_dev` |
| `min/max(x,d)` (time series) | `ts_min/ts_max` (BRAIN `min/max` are element-wise) |
| `scale(x,a)` | `scale(x, scale=a)` |
| `indneutralize(x, IndClass.g)` | `group_neutralize(x, g)` |
| `adv{d}` | `adv20` for d=20; else `ts_mean(volume*vwap, d)` (the paper uses dollar volume) |

Idea families in the paper:
- Price/volume rank correlation
- Short-term reversal
- Intraday bar shape
- VWAP deviation
- Volume-conditioned signals
- Volatility regimes
- Long-horizon filters
- Industry-neutralized variants

These are short-horizon (0.6–6.4 day holding), high-turnover signals. On BRAIN they usually need decay and neutralization. Because they are widely known, they also tend to be correlated with alphas already submitted.

The translations live in `backend/alphafoundry/catalog/alpha101.yaml`. **Verify them against the paper before relying on them.**

---

## 6. How to fix each failed check
| Symptom | Fixes |
|---|---|
| Turnover too high | decay 4–10; `ts_decay_linear(S, 5–20)`; `ts_mean(S, 3–10)`; `hump(S, 0.01)`; `trade_when(<event>, S, -1)`; longer windows or slower data |
| Turnover too low | use `ts_delta` or `ts_zscore` instead of static levels; shorter windows |
| Fitness too low | cut turnover toward 12.5%; below that, only higher Sharpe or returns help |
| Sharpe too low | rank or zscore, then winsorize; neutralize (scan market → subindustry); condition with `trade_when` or a regime switch; try flipping the sign |
| Sub-universe fails | the signal is driven by small caps: rank instead of raw ratios; neutralize by cap bucket; check TOP1000 and TOP500 locally |
| Concentrated weight | `rank`; `winsorize(S, std=4)`; truncation 0.05–0.08; `ts_backfill(x, 60–120)` or `group_backfill` for sparse data |
| Self-correlation | change dataset → group → horizon → operator family → event gate |
| Overfitting | a few standard windows, shallow nesting, a holdout period; submit only a small fraction of what you test (one community report: 1,103 tested, 28 submitted) |
| Unit mixing | rank each part before adding, e.g. `rank(price_thing) + rank(volume_thing)` |

---

## 7. Sources
- WorldQuantCareers channel: videos, playlists, RSS (links above).
- Learn2Quant lesson 6: <https://www.youtube.com/watch?v=LC6whEo80T0>
- J. Glazar, WorldQuant BRAIN project write-up: <https://jglazar.github.io/projects/wq_project/>
- alexisdpc, *Improving Alphas*: <https://github.com/alexisdpc/WorldQuant-alpha-trading/blob/main/ImprovingAlphas.md>
- Z. Kakushadze, *101 Formulaic Alphas*: <https://arxiv.org/abs/1601.00991>
- Leads found but not opened: `github.com/Uwater1/wq-alpha-research`; the Chinese BRAIN operator list at `zhuanlan.zhihu.com/p/2021324292105728426`.
