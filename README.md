# Alpha Foundry

A workbench for building WorldQuant BRAIN alphas. You write, simulate, mine and refine Fast Expression alphas on your own machine, on free data that approximates BRAIN's USA TOP3000. Every alpha gets a **quality grade** (A = BRAIN-ready) that asks for a safety margin above BRAIN's thresholds, not a bare pass.

The **BRAIN connection** is optional. When you sign in, BRAIN simulates the best candidates and returns the real metrics and checks. You can also import BRAIN's data catalog and your own alphas, and run a mining loop that BRAIN judges. Without it, everything works offline and you paste the exports into BRAIN yourself. Nothing is ever submitted automatically. See [docs/brain-api.md](docs/brain-api.md).

**New here? Read the [User Guide](docs/USER_GUIDE.md)** (also as [PDF](docs/USER_GUIDE.pdf)). It is a step-by-step manual for every page, workflow and setting.

- **Studio:** a Fast Expression editor with autocomplete, signature help and live error checking. Ctrl+Enter runs a local simulation in about 0.1–1.5 s, showing BRAIN-style checks, PnL, robustness tests, Doctor fixes, a settings sweep and a drafted description.
- **Idea Forge:** give it an idea in almost any form and the tool builds the alpha itself.
  - Accepted forms: plain English, a paper or PDF/Word document, a formula in 101-Alphas notation, pandas code, a written formula, JSON, a list of ideas, or Fast Expressions.
  - It drafts dozens of expressions, then combines, repairs, shapes, composes and evolves them.
  - It returns graded champions: a **simple** (one-line) one and a **complex** (multi-statement) one. See [Idea Forge](#idea-forge).
- **Re-engineer:** point it at a weak alpha and it diagnoses what is wrong, then rebuilds it stage by stage (direction, shape, neutralization, horizon, turnover, conditioning, blending, settings). You get a step-by-step recipe and a verdict from the out-of-sample holdout. See [Re-engineer](#re-engineer).
- **Miner:** eleven search methods running as background jobs:
  - Auto-Mine (quality-driven)
  - Complex composites (multi-statement alphas from your best decorrelated alphas)
  - BRAIN mining (BRAIN judges every candidate; needs the BRAIN connection)
  - NSGA-II genetic programming
  - Template grid
  - Typed random grammar
  - 101 Formulaic Alphas
  - Batch list
  - Settings optimizer
  - BRAIN-only ideas
  - Re-engineer (one alpha)
- **Library:** every alpha you have saved or mined, with PnL correlation clustering, a SuperAlpha-style combiner and bulk export.
- **Calibration:** imported or API-fetched BRAIN results refit a pass-likelihood model, map local Sharpe to expected BRAIN Sharpe, and steer the miners' bandit.
- **BRAIN:** sign in, import the data catalog and your alphas, simulate on BRAIN, and run BRAIN mining.

---

## Why earlier alphas failed on BRAIN, and what changed

- **They were mined on the synthetic demo data.** The demo panel contains effects the generator planted (value, quality, reversal, momentum), so a local Sharpe above 2 there means nothing on BRAIN.
  - Mining jobs are now **blocked on demo data** unless you tick "run on demo data anyway".
  - A banner warns while demo data is active.
  - Alphas mined on it are labelled `demo data` in the Library.
- **"Pass" meant scraping past the thresholds locally.** Local results are a proxy, and alphas at the edge usually fail on BRAIN.
  - Every alpha now gets a **quality grade**. **A** needs 1.25× BRAIN's Sharpe and fitness minimums, turnover inside 3–45%, a positive holdout, Sharpe in the latest 5 years (BRAIN's window), window stability, consistent years and a passing sub-universe.
  - The miners optimize this grade and save only B or better by default.
- **Fitness was left on the table.** Promising candidates now go through a **fitness shaper** before saving:
  - it tries decay, smoothing, volume-event gates and hump;
  - it tries peer ranking, size-bucket neutralization, the weight profile and neutralization;
  - a robust settings sweep then picks the best version that keeps the Sharpe.
- **Only one-line alphas.** The **composer** builds readable multi-statement alphas: weighted blends (weights from PnL), volume-gated blends, tilts, volatility regime switches, orthogonalized signals and multi-horizon ensembles.
- **The pool was the S&P 1500.** The **broad pool** (about 3,400 liquid US stocks, universes up to TOP3000) is much closer to BRAIN's TOP3000, and the SEC fields were extended.
- **No ground truth.** The optional BRAIN connection simulates candidates on BRAIN itself and feeds the results back into calibration and the miners.

---

## Quick start

1. **Double-click `start.bat`.** The first run takes a few minutes. It:
   - installs the Python environment with `uv`
   - builds the web UI with `npm`
   - compiles the simulation engine (`numba`) once

   Then it opens `http://127.0.0.1:8765`.
2. The app starts on a **synthetic demo dataset** (300 stocks). You can try everything straight away.
3. Go to **Data**, enter a contact email, choose the pool and click **Build real data**. **Mining is disabled until you do.**
   - **Broad US (recommended):** every listed US common stock from SEC's exchange list, kept when it ranks among the ~3,450 most liquid at any month end.
     - Universes go up to TOP3000, like BRAIN's USA region.
     - Names outside the S&P 1500 get GICS-like groups from a SIC → GICS crosswalk learned on the S&P names.
     - The first build takes about 40–70 minutes and 2–3 GB of disk.
   - **S&P 1500:** the current S&P 500/400/600 constituents with GICS (Wikipedia). About 10–20 minutes and 1 GB.
   - Both use daily prices from Yahoo Finance and point-in-time fundamentals from SEC EDGAR, from 2012 to today. Later updates are incremental.
   - Your email goes only into the User-Agent of SEC and Wikipedia requests, because their access policies require a contact.
4. Optional: open **BRAIN**, sign in with your BRAIN email and password, and import the data catalog. See [docs/brain-api.md](docs/brain-api.md).

**Requirements:** Windows 10/11, [`uv`](https://docs.astral.sh/uv/), and Node.js 20+ (only needed to build the UI). Python 3.13 is fetched by uv if needed.

**Manual start:** `cd backend` then `uv run alphafoundry serve --open`. `start.ps1 -NoBrowser` skips opening the browser.

---

## Suggested workflow

1. **Ideas → Idea Forge or Studio.** Describe the effect in words on **Idea Forge** and let the tool build the alpha, or write an expression in **Studio** and press Ctrl+Enter.
   - If checks fail, click **Doctor**. It proposes and simulates concrete fixes (decay, smoothing, event gates, neutralization, size-bucket neutralization, sign flip, back-filling…) and ranks them by how many checks they fix.
   - **Settings sweep** fills in a decay × neutralization × truncation heatmap. It highlights the *robust* best cell (best neighbourhood), not the lucky peak.
2. **Mining → Miner.**
   - **Auto-Mine** loops for a set time. It uses Thompson sampling across idea families, screens candidates, applies Doctor fixes to near-misses, refines the best material with GP, and keeps only candidates that are mutually decorrelated (below 0.7).
   - **Genetic programming** runs NSGA-II on fitness × novelty × simplicity, with islands and successive halving.
3. **Selection → Library.**
   - Filter to `PASS`, then check OS Sharpe, sub-universe Sharpe and the robustness warnings.
   - Select a few alphas and open **Correlation** to avoid redundant ones, and **Combine** for a SuperAlpha preview.
   - **Export** writes BRAIN simulation payloads (JSON in batches of 10), plain expressions, CSV or Markdown.
4. **Reality check → BRAIN.** Simulate the exports on BRAIN.
5. **Close the loop → BRAIN import.** Paste the results as CSV, JSON or copied text, and mark submitted alphas.
   - Submitted alphas drive the local **SELF_CORRELATION** check.
   - Imported pass/fail results recalibrate the pass-likelihood model and re-weight the miners toward what passes on BRAIN.

---

## Idea Forge

Paste an idea or open a file. Plain English works, for example *"Stocks that drop sharply on heavy volume tend to bounce back within a week"* or *"Profitable companies with low debt outperform their industry peers"*. So does any of these:

| Input | How it is read |
|---|---|
| A paper, notes, a web page (PDF, DOCX, HTML, RTF, notebooks, text) | The text is extracted, the references are dropped, and the ~6 sentences that state the mechanism, direction, data and horizon drive the search |
| 101-Alphas / academic notation, e.g. `(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))` | Translated to BRAIN operators (`delta→ts_delta`, `correlation→ts_corr`, `Ts_Rank`, `IndNeutralize→group_neutralize`, `SignedPower`, `x^y→power`, `adv60→ts_mean(volume, 60)`, fractional windows rounded) |
| pandas / numpy code, e.g. `mom = df['close'].pct_change(252).shift(21); alpha = mom.rank(axis=1)` | Translated through Python's syntax tree (never executed) into a multi-statement Fast Expression. Negative shifts, which look ahead, are refused |
| Written formulas, e.g. `Signal = 12-month return skipping the last month, ranked within industry` or `value = EBITDA / EV` | Compiled into expressions: named ratios (earnings yield, B/M, EV/EBITDA, ROE, gross margin, debt-to-equity, asset growth, FCF yield…), horizons, averages, changes, z-scores, correlations, peer ranks |
| JSON / YAML idea records, bulleted or numbered lists of ideas | Text, expressions, fields and settings are picked out. Several ideas are searched together and composed |
| Settings in the text ("neutralize by industry, decay 4") | Applied to the drafts |
| Fast Expressions, one-line or multi-statement | Used as seeds |

**1. Interpretation (instant, offline, no simulation).** The panel under the text box shows how the idea was read. You can change the mechanisms or the horizon if the reading is off.
- **Mechanism:** reversal, momentum, value, quality, growth, leverage, volatility and others. A text can name several.
- **Direction:** e.g. *long low leverage*. It is marked *as stated* when your text says which way it works, and flagged when that contradicts the usual finding.
- **Data:** the fields you named (solid chips) and typical inputs for the mechanism (dashed chips). Data that only exists on BRAIN (options, news, social, imported fields) is flagged.
- **Horizon and windows:** "within a week" becomes 5 days, "12 months" becomes 252.
- **Peer groups:** "industry peers" leads to ranking within industry.
- **Event conditions:** "on heavy volume" and "after earnings" become `trade_when` gates.
- **Interactions:** "momentum *among* low-volatility stocks".
- **Seeds:** any Fast Expressions pasted into the text.
- **Templates:** the closest library templates.

**2. The forge job.** **Forge alpha** (Ctrl+Enter) runs these stages as a background job. Quick takes up to about 5 minutes, Standard about 12, Deep about 28.

| Stage | What happens |
|---|---|
| Draft | Recipe cores for each mechanism × normalizations (rank, peer rank, size-neutral) × event gates × settings, plus matched templates and your seeds. The top cores are also tested in the **reverse** direction. |
| Combine | Blends and interactions of the best draft of each mechanism, size tilts, event gates, smoothing and neutralization variants of the leaders. |
| Refine | Doctor fixes, the fitness shaper, and a robust decay × neutralization sweep. |
| Compose | Multi-statement alphas from the leaders of different mechanisms: blends, tilts, regime switches, horizon ensembles. |
| Evolve | NSGA-II genetic programming seeded with the leaders, restricted to the idea's data. |
| Polish | A last Doctor pass and sweep on the evolved leaders, then full evaluation of the finalists (sub-universe, stability, every check, quality grade). The best simple and the best complex alpha are always among the finalists. |

**3. Scoring.** Candidates are ranked on in-sample fitness and Sharpe, check failures, stability, pass likelihood and **idea fidelity**.
- Fidelity measures whether the alpha still uses the data you named (close synonyms count, e.g. liabilities for debt) and the mechanism you described.
- It also checks that the alpha adds no unrelated data. For example, a template's value fallback is not allowed into a reversal idea.
- The OS holdout is reported, never selected on.

**4. The result.**
- **Champion:** with its metrics and chart, the steps that built it ("how it was forged") and a **hypothesis check**. The check says whether the data supports the direction you stated or favours the reverse.
- **Most faithful variant:** shown when the champion leaves out something you named.
- **Up to four decorrelated runners-up.**
- **BRAIN-only variants:** for data the local engine does not have.

Everything is saved to the Library with the tags `forge` and `forge-<job id>`. The champion also gets `forge-champion`. Your idea becomes the "Idea" section of the BRAIN description draft.

The interpretation is rule-based: a phrase lexicon plus template matching, with no LLM and no network access. If the reading is loose, name the data (price, volume, earnings, debt, …) and say what should happen next and over what horizon.

---

## Re-engineer

Re-engineer takes an existing weak alpha and rebuilds it into a strong one. It explains what was wrong and records every change it made.

**Where to start it:**
- **Studio:** click **Re-engineer** to work on the expression and settings in the editor.
- **Library:** open an alpha and click **Re-engineer**. The result is linked to the original as its parent.
- **Miner:** choose the **Re-engineer** card and paste an expression.

**How it works:**
1. **Diagnose.** The original is simulated in full. It is checked for a backwards sign, weak edge, churn, concentration, a small-cap-only edge, regime dependence and fragile windows. Its PnL is correlated with plain size, momentum, reversal, volatility and share-turnover factors. Each finding points at the stage that addresses it.
2. **Search, stage by stage.** A beam search (the best 3 versions) runs these stages in order:

   | Stage | What it tries |
   |---|---|
   | Direction | Flip the sign; component surgery (keep, drop or flip each part of a combination); isolate a cleaner sub-signal |
   | Shape | Scale-free ratios (÷ close, cap, assets…); back-fill fundamentals; ranks, winsorized z-scores, peer ranks; time-series anomaly |
   | Neutralize | Other neutralizations; size buckets; regress out the style factors it is most exposed to; `vector_neut` against a correlated submitted alpha |
   | Horizon | Every window, one at a time, on a standard ladder; all windows ×2 or ×½ |
   | Turnover | Decay; `ts_decay_linear`, `hump`, volume-event gates, turnover targeting |
   | Condition | Scale conviction by volume surprise, volatility or liquidity |
   | Blend (optional) | Add one decorrelated companion signal: a textbook factor or a library alpha. At most one per alpha, and the result must keep at least 50% PnL correlation with the pre-blend version |
   | Settings | Decay × neutralization × truncation grid with a robust (neighbourhood) pick |
   | Evolve (optional) | Short GP run around the best versions. It needs twice the usual gain to count, because it tries far more variants |

   A change is kept only if it beats its parent by a clear margin. The score combines fitness, Sharpe and the weaker half of the in-sample period, minus penalties for check violations, deep drawdowns, correlation with submitted alphas and complexity. The search stops at the time budget, when a pass finds nothing, or when a weak alpha reaches the target Sharpe and fitness.
3. **Holdout verdict.** The search only ever sees in-sample data. At the end, the original and up to four finalists are simulated on the OS holdout, with sub-universe and stability tests. The finalists always include the best unblended version. The verdict says whether the holdout confirms the improvement, partly confirms it, or shows it was fitted noise.

The champion, and any finalists that pass every check, are saved to the Library with origin `reengineer`, the tag `reengineered`, and the recipe in their notes. Each recipe step has a **Load** button, so you can stop at any intermediate version.

---

## What the local simulator does

The simulator follows BRAIN's settings and published rules. The result is a **proxy**: use it to rank and filter; BRAIN is the ground truth.

| Step | Local implementation |
|---|---|
| Universe | Monthly-rebalanced liquidity universes TOP100 … TOP3000 over the broad pool (BRAIN TOP3000 → local TOP3000, sub-universe TOP1000). On an S&P 1500 panel, TOP3000 falls back to local TOP1500. |
| Pipeline | pasteurize → evaluate → decay → neutralize (market / sector / industry / sub-industry) → scale → truncate (exact) → book $20M |
| PnL | Positions built from data up to day *s* trade at close *s + delay* and earn the next day's return. |
| Metrics | Sharpe, Returns (annual PnL / half the book, *verify*), Turnover, **Fitness = Sharpe·√(\|Returns\| / max(Turnover, 0.125))**, Drawdown, Margin, yearly table |
| Periods | In-sample (IS), an out-of-sample (OS) holdout of the last 2 years that mining never sees, and a BRAIN-like window of the last 5 IS years |
| Checks | LOW_SHARPE, LOW_FITNESS, LOW/HIGH_TURNOVER, CONCENTRATED_WEIGHT, LOW_SUB_UNIVERSE_SHARPE (re-simulated in the sub-universe), SELF_CORRELATION (against alphas you marked submitted), LOW_2Y_SHARPE (soft) |
| Local gates | OS/IS Sharpe ratio, share of profitable years, window-perturbation stability, drawdown, deflated Sharpe |

**Quality grade** (Library column, Studio card, every job result):

| Grade | Meaning |
|---|---|
| A | BRAIN-ready. Every hard check passes with a margin: Sharpe and fitness ≥ 1.25 × BRAIN's minimum, turnover 3–45%. The robustness evidence holds: holdout Sharpe ≥ 0.6 and ≥ 45% of in-sample, latest-5-year Sharpe above the minimum, window stability ≥ 70%, ≥ 70% profitable years, at most 40 nodes. |
| B | Passes every local check, but with thin margins or one robustness warning. |
| C | Near miss (one check fails narrowly), or it passes locally with weak evidence. |
| D | Fails. |

All thresholds are under `quality:` in `backend/alphafoundry/catalog/checks.yaml`, and editable from Settings.

**Known differences from BRAIN:**
- The pool contains today's listed stocks, so there is survivorship bias (delisted names are missing from Yahoo); BRAIN uses a point-in-time TOP3000.
- `vwap` is approximated by the typical price (H+L+C)/3.
- Fundamentals are trailing-twelve-month SEC values mapped to BRAIN-style names.
- Local delay-0 results are optimistic, because trades use the same close the data comes from.
- Risk-model neutralizations and non-USA regions are export-only.

Thresholds that could not be confirmed publicly are marked *verify* in `docs/research-notes.md`. They are all editable under **Settings**.

**Data honesty:**
- A fundamental value becomes usable only on the trading day **after** its SEC filing date, using the first-filed figure for each period (restatements are ignored).
- Market cap is computed in split-consistent share units.

---

## Where things live

| Path | What |
|---|---|
| `backend/alphafoundry/fastexpr/` | Fast Expression parser, typechecker, canonicalizer, printer, explainer |
| `backend/alphafoundry/engine/` | numba kernels, about 120 locally simulated operators, subexpression cache, memory-mapped panels |
| `backend/alphafoundry/sim/` | BRAIN-like simulator, checks, quality grade (`quality.py`), robustness, correlation index |
| `backend/alphafoundry/gen/` | templates, grammar, GP (NSGA-II), Doctor, bandit, fitness shaper (`optimize.py`), complex-alpha composer (`compose.py`), Idea Forge input formats (`ideaparse.py`) and interpreter (`idea.py`), BRAIN catalog candidates (`brainfields.py`), Re-engineer moves (`reengineer.py`) |
| `backend/alphafoundry/jobs/` | job manager, worker pool, miners (Auto-Mine and others), the Idea Forge job (`forge.py`), the Re-engineer beam search (`reengineer.py`) |
| `backend/alphafoundry/brainio/` | BRAIN export, results import, calibration, the BRAIN API client (`brain_api.py`), sign-in and budget service (`service.py`), credential storage (`credentials.py`) |
| `backend/alphafoundry/data/` | Wikipedia / SEC / Yahoo / SEC EDGAR pipeline (S&P 1500 or broad pool), SIC → GICS crosswalk (`sic.py`), and the demo generator |
| `docs/brain-api.md` | How to get BRAIN API access, sign in, and use the connection |
| `backend/alphafoundry/catalog/` | `operators.py` (catalog), `fields.json`, `templates.yaml` (77 templates), `alpha101.yaml`, `checks.yaml` |
| `frontend/` | React + Vite UI (Monaco editor, ECharts) |
| `runtime/` | your data panels, SQLite library, settings and logs. It is safe to back up; delete it to reset. |
| `docs/research-notes.md` | research behind the templates, checks and Doctor rules (WorldQuantCareers channel, BRAIN docs, 101 Alphas) |

**Customizing:**
- **Templates:** add your own in **Explorer → Templates** (saved to `runtime/templates_user.yaml`).
- **BRAIN data fields:** import your account's field list from **BRAIN import** so the BRAIN-only generator uses real field ids.
- **Check thresholds:** edit them in **Settings** (saved to `runtime/checks_user.yaml`).

---

## Command line

```powershell
cd backend
uv run alphafoundry serve --open          # run the app
uv run alphafoundry warmup                # (re)compile engine kernels
uv run alphafoundry build-data            # download/refresh real data from the terminal
uv run alphafoundry demo                  # rebuild the synthetic demo dataset
uv run pytest                             # backend test suite
uv run python scripts/bench.py            # responsiveness benchmark on the active dataset
cd ../frontend; npm run dev               # UI dev server on :5173 (proxies to :8765)
```

---

## Troubleshooting

- **Slow first simulation or first mining job:** the engine compiles its kernels on first use. `start.bat` does this once up front. Rerun `uv run alphafoundry warmup` after updating the code.
- **OneDrive:** this folder is synced, so large files under `runtime/`, `.venv` and `node_modules` get uploaded. If you see file-lock errors or slowness, pause OneDrive sync during data builds; writes are atomic and retried.
- **Yahoo returns 429 (rate limit):** the downloader backs off and resumes. Run **Update data** again later and cached tickers are skipped.
- **SEC or Wikipedia returns 403:** set a real contact email on the Data page.
- **Port 8765 is busy:** the server picks the next free port and prints it.
