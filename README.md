# Alpha Foundry

An offline workbench for building WorldQuant BRAIN alphas. You write, simulate, mine and refine Fast Expression alphas on your own machine, then export them in BRAIN-ready form. **It never connects to BRAIN.** You paste or run the exports yourself, and you can import BRAIN's results back so the tool learns what actually passes.

- **Studio:** a Fast Expression editor with autocomplete, signature help and live error checking. Ctrl+Enter runs a local simulation in about 0.1–1.5 s, showing BRAIN-style checks, PnL, robustness tests, Doctor fixes, a settings sweep and a drafted description.
- **Idea Forge:** paste an alpha idea in plain English and the tool builds the alpha itself. It interprets the idea, drafts dozens of expressions, then combines, repairs, evolves and polishes them into a champion with BRAIN-ready settings. See [Idea Forge](#idea-forge).
- **Re-engineer:** point it at a weak alpha and it diagnoses what is wrong, then rebuilds it stage by stage (direction, shape, neutralization, horizon, turnover, conditioning, blending, settings). You get a step-by-step recipe and a verdict from the out-of-sample holdout. See [Re-engineer](#re-engineer).
- **Miner:** nine search methods running as background jobs:
  - Auto-Mine
  - NSGA-II genetic programming
  - Template grid
  - Typed random grammar
  - 101 Formulaic Alphas
  - Batch list
  - Settings optimizer
  - BRAIN-only ideas
  - Re-engineer (one alpha)
- **Library:** every alpha you have saved or mined, with PnL correlation clustering, a SuperAlpha-style combiner and bulk export.
- **Calibration:** imported BRAIN results refit a pass-likelihood model, map local Sharpe to expected BRAIN Sharpe, and steer the miners' bandit.

---

## Quick start

1. **Double-click `start.bat`.** The first run takes a few minutes. It:
   - installs the Python environment with `uv`
   - builds the web UI with `npm`
   - compiles the simulation engine (`numba`) once

   Then it opens `http://127.0.0.1:8765`.
2. The app starts on a **synthetic demo dataset** (300 stocks). You can try everything straight away.
3. Go to **Data**, enter a contact email, and click **Build real data**.
   - This downloads the current S&P 500/400/600 constituents with their GICS classes (Wikipedia), daily prices (Yahoo Finance), and point-in-time fundamentals (SEC EDGAR).
   - That is about 1,500 US stocks from 2012 to today. The first build takes 10–20 minutes; later updates are incremental.
   - Your email goes only into the User-Agent of SEC and Wikipedia requests, because their access policies require a contact.

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

Paste an idea in plain English, for example *"Stocks that drop sharply on heavy volume tend to bounce back within a week"* or *"Profitable companies with low debt outperform their industry peers"*. Notes, a paper abstract or Fast Expressions work too.

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

**2. The forge job.** **Forge alpha** (Ctrl+Enter) runs these stages as a background job. Quick takes up to about 4 minutes, Standard about 10, Deep about 25.

| Stage | What happens |
|---|---|
| Draft | Recipe cores for each mechanism × normalizations (rank, peer rank, size-neutral) × event gates × settings, plus matched templates and your seeds. The top cores are also tested in the **reverse** direction. |
| Combine | Blends and interactions of the best draft of each mechanism, size tilts, event gates, smoothing and neutralization variants of the leaders. |
| Refine | Doctor fixes for leaders that fail a check, then a robust decay × neutralization sweep. |
| Evolve | NSGA-II genetic programming seeded with the leaders, restricted to the idea's data. |
| Polish | A last Doctor pass and sweep on the evolved leaders, then full evaluation of the finalists: sub-universe, stability and all checks. |

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
| Universe | Monthly-rebalanced liquidity universes TOP200/500/1000/1500 over the S&P 1500 pool. BRAIN TOP3000 maps to local TOP1500. |
| Pipeline | pasteurize → evaluate → decay → neutralize (market / sector / industry / sub-industry) → scale → truncate (exact) → book $20M |
| PnL | Positions built from data up to day *s* trade at close *s + delay* and earn the next day's return. |
| Metrics | Sharpe, Returns (annual PnL / half the book, *verify*), Turnover, **Fitness = Sharpe·√(\|Returns\| / max(Turnover, 0.125))**, Drawdown, Margin, yearly table |
| Periods | In-sample (IS), an out-of-sample (OS) holdout of the last 2 years that mining never sees, and a BRAIN-like window of the last 5 IS years |
| Checks | LOW_SHARPE, LOW_FITNESS, LOW/HIGH_TURNOVER, CONCENTRATED_WEIGHT, LOW_SUB_UNIVERSE_SHARPE (re-simulated in the sub-universe), SELF_CORRELATION (against alphas you marked submitted), LOW_2Y_SHARPE (soft) |
| Local gates | OS/IS Sharpe ratio, share of profitable years, window-perturbation stability, drawdown, deflated Sharpe |

**Known differences from BRAIN:**
- The pool is today's S&P 1500 constituents, so there is survivorship bias; BRAIN uses a point-in-time TOP3000.
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
| `backend/alphafoundry/sim/` | BRAIN-like simulator, checks, robustness, correlation index |
| `backend/alphafoundry/gen/` | templates, grammar, GP (NSGA-II), Doctor, bandit, Idea Forge interpreter and drafts (`idea.py`), Re-engineer moves, scoring and diagnosis (`reengineer.py`) |
| `backend/alphafoundry/jobs/` | job manager, worker pool, miners (Auto-Mine and others), the Idea Forge job (`forge.py`), the Re-engineer beam search (`reengineer.py`) |
| `backend/alphafoundry/brainio/` | BRAIN export, results import, calibration |
| `backend/alphafoundry/data/` | Wikipedia / Yahoo / SEC EDGAR pipeline and the demo generator |
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
