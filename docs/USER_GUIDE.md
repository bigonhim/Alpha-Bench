# Alpha Foundry User Guide

**How to operate the tool, from first start to alphas that pass on WorldQuant BRAIN**

Version 1.1 · September 2026

---

## Contents

1. [What Alpha Foundry is](#1-what-alpha-foundry-is)
2. [Installing and starting the tool](#2-installing-and-starting-the-tool)
3. [Building real data (do this first)](#3-building-real-data-do-this-first)
4. [Connecting to BRAIN](#4-connecting-to-brain)
5. [Finding your way around](#5-finding-your-way-around)
6. [Reading the results](#6-reading-the-results)
7. [Studio: write and test an alpha](#7-studio-write-and-test-an-alpha)
8. [Idea Forge: from an idea to an alpha](#8-idea-forge-from-an-idea-to-an-alpha)
9. [Miner: automatic search](#9-miner-automatic-search)
10. [Library: select, check and export](#10-library-select-check-and-export)
11. [The BRAIN page in detail](#11-the-brain-page-in-detail)
12. [BRAIN import (without the API)](#12-brain-import-without-the-api)
13. [Explorer](#13-explorer)
14. [Settings](#14-settings)
15. [Recommended workflows](#15-recommended-workflows)
16. [Fixing failed checks: a cheat sheet](#16-fixing-failed-checks-a-cheat-sheet)
17. [Command line](#17-command-line)
18. [Files, backup and reset](#18-files-backup-and-reset)
19. [Troubleshooting](#19-troubleshooting)
20. [Glossary](#20-glossary)
21. [Limits to keep in mind](#21-limits-to-keep-in-mind)

---

## 1. What Alpha Foundry is

Alpha Foundry is a workbench for building WorldQuant BRAIN alphas. You write, generate and refine Fast Expression alphas, and the tool tells you which ones are likely to pass on BRAIN. Everything runs in your browser at `http://127.0.0.1:8765`, served by a program on your own computer.

It has two engines that work together:

| Engine | What it is | What it is for |
|---|---|---|
| **Local simulator** | A BRAIN-like backtester running on free data (Yahoo Finance prices, SEC EDGAR fundamentals) | Fast screening: thousands of candidates per hour, offline, at no BRAIN cost |
| **BRAIN connection** (optional) | Your BRAIN account, reached through BRAIN's API | The ground truth: real metrics, real checks, real self- and production-correlation |

The local simulator is a **proxy**. Its data, universe and in-sample window differ from BRAIN's, so the tool never treats a bare local pass as good enough. Every alpha gets a **quality grade** (A to D). Grade A asks for a clear margin above BRAIN's thresholds plus evidence that the edge is real. The best candidates then go to BRAIN for the final verdict.

**The workflow at a glance**

1. Build real data (once, then update now and then).
2. Generate candidates: write them in the **Studio**, describe an idea in the **Idea Forge**, or let the **Miner** search.
3. Keep the grade A and B alphas.
4. Test them on BRAIN (**Run on BRAIN**), or paste the exports into BRAIN yourself.
5. Submit the ones BRAIN marks ready, on the BRAIN website.
6. Sync the results back so the tool learns what passes.

> **Important.** Alpha Foundry never submits an alpha for you. Submission is always a manual click on the BRAIN website.

---

## 2. Installing and starting the tool

### Requirements

- Windows 10 or 11
- [uv](https://docs.astral.sh/uv/) (Python package manager). Python 3.13 is fetched automatically.
- Node.js 20 or newer. It is only needed to build the user interface, which the start script does for you.
- About 3–4 GB of free disk space for the broad data pool, 1–2 GB for the S&P 1500 pool

### Starting

Double-click **`start.bat`** in the project folder. The script:

1. installs or updates the Python environment (`uv sync`);
2. builds the web interface if its source changed (`npm run build`);
3. on the very first run only, compiles the simulation engine (1–3 minutes);
4. starts the server and opens `http://127.0.0.1:8765` in your browser.

Keep the console window open while you work. Closing it, or pressing **Ctrl+C** in it, stops the tool. Background jobs stop with it, but everything already saved stays saved.

**Options**

- `start.ps1 -NoBrowser` starts without opening a browser tab.
- `start.ps1 -Port 8800` uses another port. If 8765 is busy, the server picks the next free port and prints it.
- Manual start: open a terminal in `backend` and run `uv run alphafoundry serve --open`.

### The first time you open it

The tool starts on a small **synthetic demo dataset** (300 made-up stocks) so the screens are not empty. A warning banner stays at the top while the demo data is active. **Mining is disabled on demo data**, for a good reason: the demo generator plants value, quality, reversal and momentum effects on purpose. Alphas that look excellent there (local Sharpe above 2) do not exist on BRAIN. Your next step is always section 3.

---

## 3. Building real data (do this first)

Open **Data** in the sidebar.

### Step by step

1. **Contact email.** Enter your email and press **Save**. SEC EDGAR and Wikipedia require a contact address in the User-Agent of automated downloads. The email is sent to those two sites only, and only for that purpose.
2. **Stock pool.** Choose one:

| Pool | What it contains | Universes | First build | Disk |
|---|---|---|---|---|
| **Broad US** (recommended) | Every listed US common stock from SEC's exchange list, kept when it ranks among the ~3,450 most liquid at any month end | TOP100 … TOP3000, like BRAIN's USA region | 40–70 min | 2–3 GB |
| **S&P 1500** | Current S&P 500, 400 and 600 constituents | TOP100 … TOP1500 | 10–20 min | ~1 GB |

3. Optional: fill **max tickers** for a quick trial build (for example 200). Leave it empty for the full build.
4. Press **Build real data**. The progress bar shows the phases: universe → prices → classification (broad pool only) → fundamentals → assemble.

When the build finishes, the tool switches to the real data automatically. The header chip changes from *Demo data* to *Real data*, and mining becomes available.

### Good to know

- **Pause OneDrive** while the build runs. The project folder is synced, and a 2–3 GB build can otherwise upload files while they are being written.
- **Updates are incremental.** Press **Update data** (the same button) now and then, for example weekly. Only new prices and changed filings are downloaded.
- **Yahoo rate limits (HTTP 429)** make the downloader wait and retry. If a build stops, run it again; cached tickers are skipped.
- **The broad pool is about twice as slow to simulate** as the S&P 1500 pool, because it holds more than twice as many stocks. It is much closer to what BRAIN tests, which is the point.

### How the data is made honest

- **Point in time.** A fundamental value becomes usable on the trading day **after** its SEC filing date. The first-filed figure is used; later restatements are ignored.
- **Trailing twelve months.** Flow items (sales, income, cash flow…) are TTM sums of quarters.
- **Groups.** S&P names carry GICS sector, industry and sub-industry labels. Other names get GICS-like labels from a SIC → GICS crosswalk learned on the S&P names. The Data page shows which classification the panel uses.
- **Liquidity universes** are rebalanced monthly on 63-day dollar volume.

### The rest of the Data page

- **Active dataset** shows dates, stock and day counts, fields, universes, the in-sample (IS) and out-of-sample (OS) periods and the BRAIN-like window. **Use** switches between real and demo data.
- **Field coverage** shows, per field, the share of stock-days with a value over the last year. Low coverage explains weak or concentrated fundamental alphas.

---

## 4. Connecting to BRAIN

The connection is optional, but it is what turns a local candidate into a verified one.

### What you get and what you do not

| Available | Not available |
|---|---|
| Simulating any alpha on BRAIN: real Sharpe, fitness, turnover, returns, drawdown, margin, and every check | BRAIN's raw data (prices, fundamentals, analyst, news…). BRAIN does not allow downloading it |
| BRAIN's pre-submission check, including self- and production-correlation | Submitting alphas. The tool never submits |
| Your own alphas, their results and daily PnL | |
| The data-field catalog: ids, descriptions, coverage, and how many alphas use each field | |

### How to get access

There is **no API key**. BRAIN's API (`api.worldquantbrain.com`) accepts the same email and password as the BRAIN website. Any account that can sign in on the website can use it. Before automating anything, read the BRAIN terms and your consultant agreement on automated use and simulation limits.

### Signing in

1. Open **BRAIN** in the sidebar.
2. Enter your BRAIN email and password.
3. Optional: tick **Remember in Windows Credential Manager**. The tool then signs in again by itself when the session expires (about every 4 hours). Instead, you can set the environment variables `BRAIN_EMAIL` and `BRAIN_PASSWORD` before starting the tool.
4. Press **Sign in**.
5. **Biometric check.** If your account uses biometric sign-in, a button **Open the verification page** appears. Open it, finish the check in your browser, then press **Complete sign-in**.

The header then shows **BRAIN connected**.

**Where your secrets live.** The session cookie is kept in `%LOCALAPPDATA%\AlphaFoundry\brain_session.json`, outside the OneDrive folder. A remembered password is kept in Windows Credential Manager (entry `AlphaFoundry-BRAIN`). Neither is ever written to the project folder, the database, job settings or logs. **Sign out** ends the session; **Forget saved password** also removes the stored password.

### First things to do once connected

1. **Limits and defaults.** Set *Parallel simulations* to your account's concurrent limit (usually 3) and a *Daily budget* you are comfortable with (default 500).
2. **Data catalog → Import top datasets.** This makes BRAIN's field ids known to the tool.
3. **Your BRAIN alphas → Sync my alphas.** This imports your history, marks submitted alphas for the self-correlation check, and calibrates the local model.

Section 11 describes every part of the BRAIN page.

---

## 5. Finding your way around

### Pages (sidebar)

| Page | Use it to |
|---|---|
| **Dashboard** | See the library at a glance: top candidates by grade, recent jobs, family statistics, coverage of your submitted alphas |
| **Studio** | Write one alpha and test it thoroughly |
| **Idea Forge** | Turn an idea (text, paper, formula, code) into finished alphas |
| **Miner** | Run automatic searches in the background |
| **Library** | Browse, filter, compare, combine, export and send alphas to BRAIN |
| **BRAIN** | Sign in, import catalog and alphas, run BRAIN mining |
| **BRAIN import** | Paste BRAIN results by hand; calibration and bandit statistics |
| **Explorer** | Look up operators, data fields, templates and the 101 Alphas |
| **Data** | Build and inspect the dataset |
| **Settings** | Check thresholds, quality grade, simulation periods, performance, theme |

### Header

- **Page title** and **dataset chip** (source, number of stocks, date range). Hover over the chip to see the IS and OS periods.
- **BRAIN connected / BRAIN offline**: click it to open the BRAIN page.
- **N jobs running**: click it to open the Miner.
- **live / offline**: whether live job updates are streaming.
- **Ctrl+K**: command palette. Jump to any page, open a recent alpha, start a forge, toggle the theme.
- **Sun/moon**: light or dark theme.

### Keyboard shortcuts

| Keys | Where | Action |
|---|---|---|
| Ctrl+Enter | Studio | Simulate |
| Ctrl+Enter | Idea Forge | Forge the idea |
| Ctrl+S | Studio | Save to the Library |
| Ctrl+K | Anywhere | Command palette |

---

## 6. Reading the results

### Metrics

| Metric | Meaning |
|---|---|
| **Sharpe** | Annualized mean over standard deviation of daily PnL |
| **Fitness** | Sharpe × √(\|Returns\| / max(Turnover, 12.5%)), as on BRAIN. Below 12.5% turnover, cutting turnover no longer helps; only higher Sharpe or returns do |
| **Turnover** | Average daily share of the book that is traded |
| **Returns** | Annual PnL divided by half the book (book = $20M) |
| **Drawdown** | Largest peak-to-trough fall of cumulative PnL, as a share of half the book |
| **Margin** | PnL per dollar traded, in basis points |

### Periods

| Period | What it is |
|---|---|
| **IS** (in-sample) | From two years after the data start to two years before today. Selection and optimization use only this period |
| **OS** (out-of-sample holdout) | The most recent two years. It is reported, never optimized on, so it shows whether an edge survives |
| **BRAIN-like window** | The last five IS years, similar to the window BRAIN evaluates |

### Checks

These follow BRAIN's delay-1 rules. All thresholds are editable in Settings.

| Check | Passes when |
|---|---|
| LOW_SHARPE | IS Sharpe ≥ 1.25 (delay 0: 2.0) |
| LOW_FITNESS | IS fitness ≥ 1.0 (delay 0: 1.3) |
| LOW / HIGH_TURNOVER | 1% ≤ turnover ≤ 70% |
| CONCENTRATED_WEIGHT | No stock above 10% of the book |
| LOW_SUB_UNIVERSE_SHARPE | Sharpe in the smaller universe ≥ 0.75 × √(sub size / universe size) × Sharpe |
| SELF_CORRELATION | PnL correlation < 0.7 with every alpha marked submitted, unless the Sharpe is at least 10% higher |
| LOW_2Y_SHARPE (soft) | The last two IS years keep a Sharpe of at least 1.0 |

The tool also reports its own robustness warnings: OS degradation, yearly consistency, parameter stability and drawdown.

### Quality grade

The grade is the single most important number in the tool. It answers the question "will this survive BRAIN?"

| Grade | Meaning | What to do |
|---|---|---|
| **A** (BRAIN-ready) | Every hard check passes **with a margin** (Sharpe and fitness ≥ 1.25 × BRAIN's minimum, turnover 3–45%), **and** the evidence holds (see below) | Send it to BRAIN |
| **B** | Passes every local check, but with thin margins or one robustness warning | Worth a BRAIN test; refine it first if you can |
| **C** | Near miss (one check fails narrowly), or passes locally with weak evidence | Refine it (Doctor, sweep, Re-engineer) |
| **D** | Fails | Drop it or rethink it |

The evidence for grade A:
- the holdout Sharpe is at least 0.6 and at least 45% of the IS Sharpe;
- the BRAIN-like window Sharpe is above the minimum;
- at least 70% of the Sharpe survives when every window moves ±25%;
- at least 70% of IS years are profitable;
- the sub-universe check passes;
- the expression has at most 40 nodes.

Once enough BRAIN results are imported, the calibrated BRAIN Sharpe estimate must also clear the minimum.

The **Quality** card lists the evidence that held and the reasons that cost grades. The grade letter appears in the Library, in job results and on the Forge results.

### Pass likelihood and calibration

**BRAIN pass likelihood** is a logistic model of local metrics that estimates the chance BRAIN's checks pass. It starts from sensible defaults. After about 30 BRAIN results (imported or fetched through the API), it is refitted to your own data. After 5 or more results, the tool also maps local Sharpe to an **expected BRAIN Sharpe**. The fit and its quality are shown on the BRAIN import page.

---

## 7. Studio: write and test an alpha

### The editor

- **Autocomplete** for operators and fields, **signature help** as you type, and **live error checking** with underlines.
- **Multi-statement alphas** are supported: assign variables with `name = expression;` and end with the alpha itself.

```
value = group_rank(ts_backfill(ebitda, 120) / enterprise_value, industry);
reversal = rank(-ts_delta(close, 5));
0.6 * value + 0.4 * reversal
```

- The **settings bar** sets region, universe, delay, decay, neutralization, truncation, pasteurization and NaN handling, as on BRAIN. Only USA can be simulated locally; other regions are for export and for BRAIN simulation.

### Running

Press **Simulate** (Ctrl+Enter). A run takes roughly 0.1–3 seconds depending on the pool. The sub-universe test and window-stability probes follow a moment later; the Quality and Checks cards update when they arrive.

| Panel | Shows |
|---|---|
| Metric strip | Sharpe, fitness, turnover, returns, drawdown, margin (IS, with OS below), BRAIN pass likelihood |
| Charts | PnL (IS and OS shaded), drawdown, rolling one-year Sharpe, turnover, long/short counts, yearly table |
| Quality | Grade, margins, evidence, reasons |
| Checks | Every BRAIN check and robustness gate, with values and limits |
| Robustness | Sharpe of the perturbed-window variants |
| Correlation | Most correlated Library alphas (click one to open it) |
| Attribution | PnL by sector, best and worst names |
| Description | A drafted BRAIN description: idea, data, operators |

### Improving an alpha

| Button | What it does |
|---|---|
| **Doctor** | Proposes and simulates concrete fixes for each failing check: decay, smoothing, event gates, neutralization, size buckets, sign flip, back-filling… It ranks them by how many checks they fix. Click **Apply** to load one |
| **Settings sweep** | A decay × neutralization × truncation heatmap. The highlighted cell is the **robust** best (best neighbourhood), not the lucky peak |
| **Re-engineer** | A staged search that rebuilds a weak alpha: direction, shape, neutralization, horizon, turnover, conditioning, blending, settings. It gives a recipe and a holdout verdict. Set a time budget and targets, then start it; it runs as a background job |

### Saving and sending

- **Save** (Ctrl+S) stores the alpha, its full evaluation and its grade in the Library.
- **Copy** copies the expression. **BRAIN JSON** copies a simulation payload with settings.
- **Run on BRAIN** simulates it on BRAIN (needs the connection). The result appears in the Library.
- **History** lists your recent runs with their Sharpe, fitness and turnover. Click one to rerun it.

---

## 8. Idea Forge: from an idea to an alpha

The Forge turns an idea into finished, graded alphas. You describe **what** should work; it works out **how**.

### What you can give it

| Input | Example | How it is read |
|---|---|---|
| Plain English | *Stocks that drop sharply on heavy volume tend to bounce back within a week.* | Mechanism, direction, data, horizon, peer groups and event conditions are recognised |
| A written formula | *Signal = 12-month return skipping the last month, ranked within industry* | Compiled into `group_rank(ts_delay(ts_sum(returns, 231), 21), industry)` |
| Named ratios | *EV/EBITDA, earnings yield, book-to-market, ROE, gross margin, debt to equity, asset growth, FCF yield, low P/E* | Built from the right fields, with fundamentals back-filled |
| 101-Alphas or paper notation | `(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))` | Translated to BRAIN operators (`delta → ts_delta`, `correlation → ts_corr`, `IndNeutralize → group_neutralize`, `x^y → power`, `adv60 → ts_mean(volume, 60)`) |
| pandas code | `mom = df['close'].pct_change(252).shift(21)` then `alpha = mom.rank(axis=1)` | Translated via Python's syntax tree (never executed) into a multi-statement alpha. A negative `shift`, which looks into the future, is refused |
| Fast Expressions | Any expression or program | Used as seeds |
| A document | PDF, Word, HTML, RTF, notebook, text | Text is extracted, references dropped, and the ~6 sentences that state the idea drive the search |
| JSON or YAML | `{"idea": "...", "fields": [...], "neutralization": "INDUSTRY"}` | Text, expressions, fields and settings are picked out |
| A list of ideas | Bulleted or numbered lines | Searched together; the Compose stage can combine them |
| Settings in the text | *neutralize by industry, decay 4, TOP1000* | Applied to the drafts |

Type or paste into the box, or press **Open a file**. Try the example chips under the box to see how different inputs are read.

### Checking how it read your idea

The **How the forge reads it** panel updates as you type (no simulation yet):

- **Read as:** the detected format, plus any settings found in the text.
- **Formulas built from your description:** the compiled expressions. Click one to open it in the Studio.
- **Key sentences** (documents): the sentences it used.
- **Mechanism and direction:** for example *Short-term reversal: long low recent return (as stated)*. Press **Change mechanisms** to correct it.
- **Data:** solid chips are data you named; dashed chips are typical inputs added for the mechanism; orange chips exist only on BRAIN.
- **Horizon:** short, medium or long. Click to override.
- **Conditions and seeds**, the **first drafts** it will try, and the **closest library templates**.

A *loose reading* means the text named neither the data nor what should happen next. Add both, for example *"… on heavy volume … bounce back within a week"*.

### Running a forge

Choose the effort and press **Forge alpha** (Ctrl+Enter).

| Effort | Time budget | Use for |
|---|---|---|
| Quick | up to ~5 min | Checking an idea |
| Standard | up to ~12 min | Normal use |
| Deep | up to ~28 min | Ideas worth the time; more drafts and a two-island evolution |

The stages run in the background, and the page shows live progress and a leaderboard:

| Stage | What happens |
|---|---|
| Interpret | Reads the idea |
| Draft | Builds and screens recipe drafts, compiled formulas, seeds and matching templates. The top ones are also tested in the reverse direction |
| Combine | Blends and interactions across mechanisms, size tilts, event gates, smoothing, neutralization variants |
| Refine | Doctor fixes, the fitness shaper, and a robust settings sweep |
| Compose | Complex (multi-statement) alphas from the leaders: blends, tilts, regime switches, horizon ensembles |
| Evolve | Genetic programming seeded with the leaders, restricted to the idea's data |
| Polish | Full evaluation of the finalists: sub-universe, stability, every check, quality grade |

You can leave the page; the forge keeps running. Open **BRAIN settings for the search** before starting to change region, universe, delay or other base settings.

### The result

- **Champion alpha:**
  - its grade, status and metrics (IS with OS);
  - the evidence and the reasons behind the grade;
  - how it was forged, step by step;
  - a **hypothesis check** that says whether the data supports the direction you stated or favours the reverse.

  Buttons: **Open in Studio**, **Copy**, **BRAIN JSON**, **Run on BRAIN**, **Library #id**.
- **Best simple (one-line) alpha** and **best complex (multi-statement) alpha**: both are always offered when both exist.
- **Most faithful variant**: shown when the champion leaves out data you named.
- **Runners-up**: decorrelated alternatives (PnL correlation below 0.7).
- **BRAIN-only variants**: drafts on data only BRAIN has, saved unscored. Test them with **Run on BRAIN**.

Everything is saved to the Library with the tags `forge` and `forge-<job id>`. The champion is tagged `forge-champion`.

### Tips for good ideas

- Name the **data** (price, volume, earnings, debt, cash flow…), the **direction** (who outperforms) and the **horizon** (days, weeks, months).
- Say how stocks are compared: "within industry", "size-neutral".
- Mention conditions: "after earnings", "on heavy volume", "among low-volatility stocks".
- Paste formulas as they are; the translators handle paper notation and pandas.

---

## 9. Miner: automatic search

Choose a method card, set its options and press **Start**. Jobs run in the background. The right-hand panel shows live progress and the saved results, and has **Pause**, **Resume** and **Stop** buttons. Stopped jobs keep everything saved so far.

| Method | What it does | Typical use |
|---|---|---|
| **Auto-Mine** | Quality-driven loop: bandit-chosen templates plus random grammar → screen → fitness shaping → Doctor on near misses → periodic genetic programming → periodic complex composites. Saves graded, decorrelated alphas | The default search |
| **Complex composites** | Multi-statement alphas from your best decorrelated Library alphas (or expressions you paste): weighted blends, volume-gated blends, tilts, regime switches, orthogonalized signals, horizon ensembles | After you have several good alphas |
| **Genetic programming** | NSGA-II evolution of expression trees (fitness × novelty × simplicity), with islands and successive halving | Refining a promising family |
| **Template grid** | Expands the idea-tagged template library over fields, windows, groups and settings | Systematic coverage of known ideas |
| **Random grammar** | Typed, unit-aware random expressions | Wide exploration |
| **101 Alphas** | The *101 Formulaic Alphas* as seeds | Price-volume starting points |
| **Batch list** | Evaluates and saves every expression you paste, one per line | Testing a list you already have |
| **Settings optimizer** | Robust decay × neutralization × truncation search for chosen Library alphas | Squeezing fitness out of an existing alpha |
| **BRAIN-only ideas** | Candidates on data that exists only on BRAIN | Feeding BRAIN tests |
| **Re-engineer** | The staged rebuild of one weak alpha | Rescuing an alpha |
| **BRAIN mining** | Opens the BRAIN page, where BRAIN judges every candidate | Verified results |

### Auto-Mine options

| Option | Default | Meaning |
|---|---|---|
| Time limit | 30 min | The job stops at this time … |
| Stop after this many alphas of grade | 10 of grade A | … or when this many alphas of the chosen grade are found |
| Save alphas graded at least | B | Use C to also keep near misses for later refinement |
| Save if Sharpe ≥ | 1.0 | Floor for considering a candidate at all |
| Round size | 24 | Candidates generated per round |
| GP every N rounds | 3 | How often genetic programming refines the best material |
| Idea families | all | Restrict the search to chosen families |
| Build complex composites | on | Every few rounds, compose the good alphas found into multi-statement alphas |
| Also generate BRAIN-only ideas | off | Add BRAIN-only candidates at the end |

The **base BRAIN settings** at the bottom of each method apply to all its candidates.

### How long it takes

On a 2-core laptop, the demo panel evaluates about 30 candidates per second. The real S&P 1500 pool is several times slower, and the broad pool slower still. Plan 30–60 minutes for a meaningful Auto-Mine run on real data. Use **Settings → Performance → Miner worker processes** to match your cores.

---

## 10. Library: select, check and export

Every alpha you save, forge or mine lands here.

### The table

Columns: selection, star, id, local status, **grade**, Sharpe, fitness, turnover, returns, OS Sharpe, sub-universe Sharpe, pass likelihood, highest correlation with another library alpha, idea family, BRAIN status and expression. Click a header to sort; the default sort is by quality. A **demo data** label in the BRAIN column marks alphas mined on the synthetic data. Ignore those.

### Filters

Search (expression, notes, tags) · status · origin · idea · data category · **grade** (A; A or B; A to C) · **BRAIN status** (tested, ready to submit, passed, failed, not tested) · **dataset** (real or demo) · minimum Sharpe · starred · submitted · local only.

### The alpha drawer

Click a row to open it. You see the expression and settings, then:
- **Open in Studio**, **Copy**, **BRAIN JSON**, **Run on BRAIN**, **Star**, **Re-engineer**;
- **Submitted on BRAIN**: marks the alpha so the SELF_CORRELATION check compares new alphas against it;
- notes and tags;
- the full result: metrics, charts, description, quality, checks and robustness;
- **Results on BRAIN**: real metrics, failing checks, self- and production-correlation, and a link to the alpha on the BRAIN website.

### Bulk actions (select rows first)

| Action | What it does |
|---|---|
| **Export** | BRAIN simulation payloads (JSON in batches of 10), plain expressions with settings comments, CSV with local metrics, or Markdown reports. Copy or download them |
| **Correlation** | PnL correlation heatmap of the selection, clustered. Red means redundant; BRAIN's limit is 0.7 |
| **Combine** | A SuperAlpha-style PnL blend preview (equal, inverse-volatility or Sharpe weights) |
| **Optimize settings** | Starts the settings optimizer on the selection |
| **Refine with GP** | Starts genetic programming seeded with the selection |
| **Compose complex** | Builds multi-statement alphas from the selected alphas |
| **Run on BRAIN** | Simulates the selection on BRAIN |
| **Star**, **Mark submitted**, **Unmark**, **Delete** | Bookkeeping |

---

## 11. The BRAIN page in detail

### Connection

Status, session expiry, whether multi-simulation is allowed (up to 10 alphas per request), simulations used today, and the saved sign-in. **Sign out** and **Forget saved password** are here too.

### Limits and defaults

| Setting | Meaning |
|---|---|
| Parallel simulations | Simultaneous BRAIN simulations. Match your account's limit (usually 3) |
| Daily budget | The most simulations this tool may run per day. Jobs stop cleanly when it is reached |
| Multi-simulation | Auto uses it when your permissions allow |
| Region, universe, delay | Defaults for catalog imports and BRAIN mining |
| Run BRAIN's pre-submission check | For alphas that pass, also fetch self- and production-correlation. An alpha that clears everything is marked **ready** |

### Data catalog

Press **Import top datasets**, or tick datasets in the list first and press **Import N dataset(s)**. For each field, the import brings in the id, description, coverage, and how many alphas already use it. Afterwards:
- the Idea Forge finds BRAIN fields your idea describes (for example "analyst revisions" or "news sentiment") and drafts BRAIN-only alphas on them;
- BRAIN mining prefers well-covered fields that few alphas use, because crowded fields tend to fail correlation checks;
- the Explorer lists the fields.

### Your BRAIN alphas

**Sync my alphas** imports your last 300 BRAIN alphas and their results. Submitted ones are marked submitted in the Library, and their PnL drives the local SELF_CORRELATION check. Every result recalibrates the pass model and rewards the idea families that really pass.

### BRAIN mining

A closed loop in which BRAIN judges every candidate:

1. It takes your best untested Library alphas (grade A only, A or B, or A to C) and fresh ideas on imported catalog fields.
2. It simulates them on BRAIN within the budget.
3. It repairs near misses from BRAIN's own failing checks (decay, neutralization, smoothing, ranking…) and simulates the repairs, for up to *Repair rounds* rounds.
4. It records every result in the Library.

| Option | Meaning |
|---|---|
| Simulation budget | Most simulations this run may use |
| Time limit | Wall-clock limit |
| Repair rounds | How many times near misses are repaired and re-simulated |
| Library alphas of grade | Which local alphas qualify |
| Best untested library alphas / ideas on imported catalog fields | The candidate sources |

The **BRAIN jobs** list and panel show progress: simulated, passed, and ready counts.

---

## 12. BRAIN import (without the API)

If you prefer not to connect, or for results you already have:

1. Copy results from BRAIN as CSV, as JSON from your own scripts, or as text copied from the results panel.
2. Paste them into **Import BRAIN results** (or **Open file**), choose the format or leave it on auto-detect, and press **Import**. Tick **Mark all as submitted** for submitted alphas.
3. Rows are matched to Library alphas by canonical expression; new ones are created.

The page also shows:
- **Import BRAIN data-field catalog**: paste a field list (CSV or JSON), for example exported from BRAIN's data explorer;
- **Local vs BRAIN calibration**: local against BRAIN Sharpe, with the fitted mapping;
- **Pass rate by idea family** on BRAIN;
- **Search bandit**: which idea families and templates the miners currently favour, based on local grades and BRAIN outcomes.

---

## 13. Explorer

| Tab | Contents |
|---|---|
| **Operators** | Every Fast Expression operator with signature, description and example. Operators marked BRAIN-only cannot be simulated locally |
| **Data fields** | Local fields and imported BRAIN fields, with descriptions and availability |
| **Templates** | The idea-tagged template library (about 90 templates, including multi-statement composites). **Try in Studio** opens an expansion, **Mine this family** starts the template grid. **Add / update a user template** saves your own to `runtime/templates_user.yaml` |
| **101 Alphas** | The translated formulas from *101 Formulaic Alphas*. Verify them against the paper before relying on them |

---

## 14. Settings

| Card | Settings |
|---|---|
| **Submission check thresholds** | Delay-1 and delay-0 Sharpe and fitness minimums, turnover limits, max weight, sub-universe factor, self-correlation limit, Sharpe improvement and years, soft two-year Sharpe, local robustness gates. **Quality grade A** section: Sharpe and fitness margins, turnover band, holdout Sharpe and ratio, stability, maximum nodes |
| **Simulation** | Years of warm-up before IS, OS holdout length, BRAIN window length, book size, returns basis, history start |
| **Performance** | Miner worker processes, server and worker cache sizes |
| **Appearance** | Theme |

Changes to periods or the dataset reload the data automatically. Check thresholds are saved to `runtime/checks_user.yaml`; the defaults stay in `backend/alphafoundry/catalog/checks.yaml`.

> Tighten the quality margins if too many grade A alphas fail on BRAIN. Loosen them slightly if BRAIN passes alphas the tool rates B.

---

## 15. Recommended workflows

### A. Your first week

1. **Data:** build the Broad US pool (section 3).
2. **BRAIN:** sign in, import the catalog, sync your alphas (section 4).
3. **Miner → Auto-Mine:** 30–60 minutes, target 10 grade A alphas.
4. **Library:** filter *Grade A or B*, sort by quality, select the top 10–20 → **Correlation** to drop near-duplicates → **Run on BRAIN**.
5. **Library:** filter *BRAIN: ready to submit* and submit the best on the BRAIN website.
6. **BRAIN → Sync my alphas**, so self-correlation and calibration include what you submitted.

### B. Turn a paper or a formula into alphas

1. **Idea Forge → Open a file** (PDF or Word), or paste the formula or code.
2. Check the reading: format, compiled formulas, key sentences, mechanism, data. Correct the mechanism or horizon if needed.
3. **Standard** effort → **Forge alpha**.
4. Compare the champion, the best simple and the best complex alpha. Read the hypothesis check.
5. **Run on BRAIN** on the one with the best grade. Test the BRAIN-only variants too.

### C. Fix an alpha that failed on BRAIN

1. Open it in the **Studio**.
2. Press **Doctor**. Apply the fix that clears the check BRAIN failed (see section 16).
3. Run the **Settings sweep** and take the robust cell.
4. If it is still weak, press **Re-engineer** with a 5–10 minute budget.
5. Save the result and **Run on BRAIN**.

### D. Build a complex alpha from your best ones

1. **Library:** filter grade A or B and select 3–8 alphas of different families that are not correlated (check with **Correlation**).
2. **Compose complex**. The job writes blends, tilts, regime switches and horizon ensembles and saves the good ones, tagged `complex`.
3. Compare each composite with its best part: it should add Sharpe or stability, not just complexity.

### E. Stay diverse (avoid SELF_CORRELATION)

- Mark submitted alphas as submitted, or sync them from BRAIN.
- Filter by idea family and mine families you have few alphas in (the Dashboard shows the spread).
- Prefer less-used BRAIN fields (BRAIN mining does this automatically).
- Change the dataset first, then the peer group, then the horizon. These change the PnL path most.

### F. A daily routine

1. **Data → Update data** (a few minutes).
2. Start an **Auto-Mine** or a **BRAIN mining** run in the morning.
3. Review new grade A alphas and BRAIN-ready alphas in the Library at the end of the day.

---

## 16. Fixing failed checks: a cheat sheet

| Failed check | What to try in the tool |
|---|---|
| **LOW_FITNESS** with high turnover | Increase decay (6–16); `ts_decay_linear(x, 5–10)`; a volume-event gate `trade_when(volume > adv20, x, -1)`; `hump(x, 0.02–0.05)`. Doctor and the fitness shaper try all of these |
| **LOW_FITNESS** with turnover already below 12.5% | Only a higher Sharpe or returns help: peer-relative ranking (`group_rank`), a sharper weight profile (winsorized `zscore` instead of `rank`), a better neutralization |
| **LOW_SHARPE** | Rank or z-score and winsorize; test other neutralizations; peer-relative versions; condition on volume or volatility; check the reverse sign (the Forge's hypothesis check does this) |
| **HIGH_TURNOVER** | As for LOW_FITNESS with high turnover, plus longer windows |
| **LOW_TURNOVER** | Decay 0; shorter windows; use changes (`ts_delta`, `ts_zscore`, `ts_rank`) instead of static levels |
| **CONCENTRATED_WEIGHT** | Wrap in `rank()`; `winsorize(x, std=3)`; truncation 0.05; `ts_backfill` sparse fundamentals |
| **LOW_SUB_UNIVERSE_SHARPE** | The edge lives in small caps: rank instead of raw ratios; `group_neutralize(x, bucket(rank(cap), range="0.1,1,0.1"))`; sub-industry neutralization |
| **SELF_CORRELATION** | Different dataset or field; different peer group; double or halve the horizon; different operator family; `regression_neut` against the correlated alpha; a complex composite with a new component |
| **Grade B/C for weak evidence** (holdout, stability) | Standard windows (Doctor → Standard windows); simplify the expression; prefer fewer nodes; horizon ensembles (Compose) |

---

## 17. Command line

Run these from a terminal in the `backend` folder.

```
uv run alphafoundry serve --open             # start the app (options: --host, --port)
uv run alphafoundry warmup                   # recompile the engine kernels (after code updates)
uv run alphafoundry build-data               # build or update real data (pool from Settings)
uv run alphafoundry build-data --max-tickers 200   # quick trial build
uv run alphafoundry demo                     # rebuild the synthetic demo dataset
uv run pytest                                # run the test suite
uv run python scripts/bench.py               # responsiveness benchmark on the active dataset
```

UI development: `cd frontend` then `npm run dev` (serves on port 5173 and forwards to the backend on 8765).

---

## 18. Files, backup and reset

| Path | Contents |
|---|---|
| `runtime/alphafoundry.db` | The Library, results, jobs, BRAIN results, calibration and bandit (SQLite) |
| `runtime/panels/` | The real dataset (large; can be rebuilt) |
| `runtime/panels_demo/` | The demo dataset (rebuilt automatically) |
| `runtime/raw/` | Download caches for Yahoo, SEC and Wikipedia (make updates fast) |
| `runtime/settings.json` | App settings |
| `runtime/checks_user.yaml` | Your check and quality thresholds |
| `runtime/templates_user.yaml` | Your own templates |
| `runtime/fields_user.json` | Imported BRAIN field catalog |
| `runtime/logs/` | Server log |
| `%LOCALAPPDATA%\AlphaFoundry\` | BRAIN session cookie (outside OneDrive) |
| `docs/brain-api.md` | BRAIN connection reference |

- **Back up** `runtime/alphafoundry.db` and the three `*_user.*` files. The datasets can always be rebuilt.
- **Reset everything:** stop the tool and delete the `runtime` folder. The next start recreates it with the demo data.
- **Reset only the Library:** stop the tool and delete `runtime/alphafoundry.db*`.
- **OneDrive:** the project folder is synced. If you see file-lock errors or slowness, pause sync during data builds and long mining runs.

---

## 19. Troubleshooting

| Problem | Solution |
|---|---|
| "The active dataset is the synthetic DEMO panel…" when starting a job | Build real data (section 3). To try the tool anyway, tick *Run on demo data anyway*; the results will not transfer to BRAIN |
| SEC or Wikipedia returns 403 | Enter a real contact email on the Data page |
| Yahoo returns 429 | The downloader waits and retries. Run the build again later; cached tickers are skipped |
| The build stops partway | Run it again. Everything downloaded so far is cached |
| First simulation or job is slow | The engine compiles on first use. `start.bat` does this once; rerun `uv run alphafoundry warmup` after code updates |
| Port 8765 is busy | The server uses the next free port and prints it |
| "BRAIN rejected the email or password" | Check them by signing in on the BRAIN website |
| Biometric link keeps coming back | Finish the check in the browser **before** pressing Complete sign-in; the link expires after a few minutes |
| "BRAIN daily simulation limit reached" | BRAIN's own daily limit. Wait for the reset; results so far are kept |
| "Today's BRAIN simulation budget is used up" | The tool's own cap. Raise *Daily budget* on the BRAIN page |
| A field is unknown on BRAIN | The local name may not exist in that region or universe. Import the catalog and check the field in the Explorer |
| Grade A alphas still fail on BRAIN | Import or sync the results so calibration learns, and raise the quality margins in Settings. Make sure you use the Broad US pool |
| No grade A alphas are found | Run longer, try more families, keep C grades (*Save alphas graded at least: C*) and refine them, or lower the margins slightly |
| A scanned PDF gives no text | The PDF has no text layer. Run OCR on it first, or paste the key passages |

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **Alpha** | A Fast Expression that assigns each stock a weight every day |
| **Book** | The total position size ($20M), half long and half short after neutralization |
| **Decay** | Linear averaging of the alpha over the last *n* days; lowers turnover |
| **Delay** | Days between the data and the trade: 1 on BRAIN's standard setting |
| **Fast Expression** | BRAIN's alpha language |
| **Fitness** | Sharpe × √(\|returns\| / max(turnover, 12.5%)) |
| **Grade** | Alpha Foundry's quality rating: A (BRAIN-ready) to D |
| **IS / OS** | In-sample (used for selection) and out-of-sample holdout (reported only) |
| **Neutralization** | Removing market, sector, industry or sub-industry exposure by demeaning within groups |
| **Pool** | The set of stocks in the local dataset (Broad US or S&P 1500) |
| **Sub-universe** | A smaller, more liquid universe (for example TOP1000 inside TOP3000) used for a robustness check |
| **Truncation** | Cap on any single stock's share of the book |
| **Universe** | The tradable stocks on a given day, chosen by liquidity (TOP3000, TOP1000…) |

---

## 21. Limits to keep in mind

- **Local results are a proxy.** The free data differs from BRAIN's: prices from Yahoo, fundamentals from SEC filings, VWAP approximated by (high + low + close) / 3, and today's listed stocks only, so there is survivorship bias. BRAIN is the ground truth; the quality grade and calibration are there to bridge the gap.
- **Delay-0 results are optimistic** locally, because trades use the same close the data comes from.
- **Only USA can be simulated locally.** Other regions and risk-model neutralizations work through BRAIN (Run on BRAIN, BRAIN mining) and exports.
- **The BRAIN API endpoints** are the ones BRAIN's website and community tools use. They are not an official, versioned API, and BRAIN may change them. If a call starts failing after a BRAIN update, the error names the endpoint.
- **Some thresholds are unverified** against official BRAIN documentation: the delay-0 minimums, the sub-universe factor, the two-year check and the returns basis. They are editable in Settings.
- **Keep submitting selectively.** Test many, submit few, and diversify across data, horizons and ideas.
