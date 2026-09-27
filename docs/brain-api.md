# Connecting Alpha Foundry to WorldQuant BRAIN

This guide covers what the BRAIN connection can and cannot do, how to get access, how to sign in, and how to use it well.

## What you can get from BRAIN, and what you cannot

| Available through the API | Not available |
|---|---|
| Simulating any alpha on BRAIN: real Sharpe, fitness, turnover, returns, drawdown, margin and every submission check | BRAIN's raw data values (prices, fundamentals, analyst, news…). BRAIN does not let anyone download datasets. |
| BRAIN's pre-submission check, including self-correlation and production correlation | Submitting alphas. This tool deliberately never submits; you submit on the website. |
| Your own alphas, their results and their daily PnL | |
| The data-field catalog: field ids, descriptions, coverage, and how many alphas already use each field | |

Local simulation therefore still runs on free data (Yahoo Finance and SEC EDGAR). BRAIN acts as the judge of the candidates the local engine finds.

## How to get API access

1. **There is no separate API key.** BRAIN's REST API (`https://api.worldquantbrain.com`) accepts the same email and password you use on <https://platform.worldquantbrain.com>. Any account that can sign in to the website can use it.
2. **Biometric sign-in.** If your account uses it, the first sign-in from Alpha Foundry returns a verification link. Open it, complete the check in your browser, then press **Complete sign-in** in the app.
3. **Your agreement with BRAIN.** Read the BRAIN terms and your consultant agreement on automated use and simulation limits. Stay within your account's concurrent-simulation limit (usually 3) and your daily limits. Alpha Foundry sets the concurrency to 3 and caps its own use with a daily budget (500 by default) that you can lower on the BRAIN page.

## Signing in

1. Start the app (`start.bat`) and open **BRAIN** in the sidebar.
2. Enter your BRAIN email and password.
   - Tick **Remember** to store them in Windows Credential Manager. The app then signs in again by itself when the session expires (about every 4 hours).
   - Or set `BRAIN_EMAIL` and `BRAIN_PASSWORD` as environment variables before starting the app.
3. If BRAIN asks for a biometric check, open the link, finish it, then press **Complete sign-in**.

**Where secrets live.** The session cookie is kept in `%LOCALAPPDATA%\AlphaFoundry\brain_session.json`, outside the OneDrive-synced project folder. A remembered password goes into Windows Credential Manager (service `AlphaFoundry-BRAIN`). Neither is ever written to the project folder, the database, job configs or logs. **Sign out → Forget saved password** removes both.

## What to do once connected

1. **Import the data catalog** (BRAIN page → Data catalog).
   - Pick datasets, or let the app import the highest-value ones.
   - The Idea Forge then finds BRAIN fields your idea describes ("analyst revisions", "news sentiment", "implied volatility"…) and drafts alphas on them.
   - BRAIN mining prefers well-covered fields that few alphas use, because crowded fields tend to fail self- and production-correlation.
2. **Sync your alphas** (BRAIN page → Your BRAIN alphas).
   - Imports your recent BRAIN alphas and their real results.
   - Submitted alphas are marked as such, so the local SELF_CORRELATION check compares against your real portfolio.
   - Every imported result recalibrates the local pass model and rewards the idea families that actually pass.
3. **Test candidates on BRAIN.**
   - Use **Run on BRAIN** in Studio, in the Idea Forge result, or on a Library selection.
   - The real metrics and checks appear in the Library drawer under *Results on BRAIN*, with a link to the alpha on the website.
   - Passing alphas also get BRAIN's pre-submission check. The status becomes `ready` when BRAIN would accept the submission.
4. **BRAIN mining** (BRAIN page). A closed loop where BRAIN judges every candidate:
   - It sends your best untested local alphas (grade A/B first) plus fresh ideas built on imported catalog fields.
   - Near misses are repaired from BRAIN's own failing checks (decay, neutralization, smoothing, ranking…) and re-simulated, for up to *Repair rounds* rounds.
   - Nothing is ever submitted.

## A good routine

1. **Data → Build real data**, using the *Broad US* pool, which is closest to BRAIN's TOP3000. Never mine on the demo data.
2. **Miner → Auto-Mine** or **Idea Forge**. Keep only grade **A** and **B** alphas.
3. **Library** → filter *Grade A or B* → select → **Run on BRAIN**.
4. **Library** → filter *BRAIN: ready to submit* → submit the best of them on the BRAIN website.
5. **BRAIN → Sync my alphas** after submitting, so self-correlation and calibration stay current.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "BRAIN rejected the email or password" | Check the credentials by signing in on the website first. |
| Biometric link keeps coming back | Finish the check in the browser *before* pressing Complete sign-in; the link expires after a few minutes. |
| "BRAIN daily simulation limit reached" | BRAIN's own daily cap. Wait for the reset. The app stops cleanly and keeps every result so far. |
| "Today's BRAIN simulation budget is used up" | The app's own cap. Raise *Daily budget* on the BRAIN page. |
| Simulations queue slowly | Your account's concurrent-simulation limit. Multi-simulation (10 alphas per request) is used automatically when your permissions include it. |
| A field is "unknown" on BRAIN | The local proxy name may not exist in that region or universe. Import the catalog and check the field in the Explorer. |

## API endpoints used (for reference)

`POST/GET/DELETE /authentication` · `POST /simulations` (single or a list of up to 10) · `GET /simulations/{id}` · `GET /alphas/{id}` · `GET /alphas/{id}/recordsets/pnl` · `GET /alphas/{id}/check` · `GET /alphas/{id}/correlations/{self|prod}` · `GET /data-sets` · `GET /data-fields` · `GET /users/self/alphas`

These are the endpoints the BRAIN website itself uses and that open-source community tools document. BRAIN may change them without notice. If a call starts failing after a BRAIN update, the error message names the endpoint.
