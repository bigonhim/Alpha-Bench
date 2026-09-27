"""Build the real-data panel: universe -> Yahoo prices -> SEC fundamentals -> fields, groups, universes.

Two pools:
  sp1500  current S&P 500/400/600 constituents (GICS from Wikipedia); universes TOP100 ... TOP1500
  broad   every listed US common stock from SEC's exchange file, kept when it ever ranks among the most liquid
          ``broad_pool_size`` x 1.15 names at a month end; universes TOP100 ... TOP3000 like BRAIN's USA region.
          Names outside the S&P 1500 are classified through a SIC -> GICS crosswalk learned from the S&P names.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from ..engine.panel import write_panel
from . import edgar, universe, yahoo
from .demo import liquidity_universes
from .gics import industry_of
from .sic import NON_OPERATING_SIC, Crosswalk

log = logging.getLogger("alphafoundry.data")

Progress = Callable[[str, int, int, str], None]
POOL_HEADROOM = 1.15


def _nop(phase: str, done: int, total: int, msg: str) -> None:  # pragma: no cover
    pass


def forward_fill_series(dates: np.ndarray, series: list, T: int) -> np.ndarray:
    """PIT series [(filed, end, value)] -> (T,) array; value usable strictly after its filing date and only
    if its period end is the latest seen so far."""
    out = np.full(T, np.nan, dtype=np.float64)
    if not series:
        return out
    cur_end = ""
    events: list[tuple[int, float]] = []
    for filed, end, val in series:
        if end < cur_end:
            continue
        cur_end = end
        idx = int(np.searchsorted(dates, np.datetime64(filed, "D"), side="right"))
        events.append((idx, val))
    for k, (idx, val) in enumerate(events):
        if idx >= T:
            break
        nxt = events[k + 1][0] if k + 1 < len(events) else T
        out[idx:min(nxt, T)] = val
    return out


def select_liquid_pool(prices: dict[str, dict], start: str, size: int, headroom: float = POOL_HEADROOM) -> list[str]:
    """Tickers that rank among the ``size * headroom`` most liquid (63-day dollar volume) at any month end.

    Works ticker by ticker so the full candidate matrix (5,000+ names) never has to be held in memory."""
    keep_n = int(math.ceil(size * headroom))
    by_month: dict[np.datetime64, list[tuple[float, str]]] = defaultdict(list)
    first = np.datetime64(start, "M")
    for t, p in prices.items():
        d, c, v = p["dates"], p["close"], p["volume"]
        if len(d) < 63:
            continue
        dv = np.nan_to_num(np.asarray(c, dtype=np.float64) * np.asarray(v, dtype=np.float64))
        cs = np.cumsum(dv)
        roll = (cs[62:] - np.r_[0.0, cs[:-63]]) / 63.0
        months = d[62:].astype("datetime64[M]")
        last = np.r_[months[1:] != months[:-1], True]
        for m, val in zip(months[last], roll[last]):
            if m >= first and val > 0:
                by_month[m].append((float(val), t))
    keep: set[str] = set()
    for lst in by_month.values():
        lst.sort(reverse=True)
        keep.update(t for _, t in lst[:keep_n])
    return sorted(keep)


def classify_pool(uni: pd.DataFrame, meta: dict[int, dict]) -> tuple[pd.DataFrame, dict]:
    """Fill sector/industry/subindustry for names without GICS labels using the learned SIC crosswalk."""
    def num(x) -> int | None:
        return None if x is None or x != x else int(x)

    uni = uni.copy()
    uni["sic"] = [meta.get(num(c), {}).get("sic") if num(c) is not None else None for c in uni["cik"]]
    has = uni["gics"].fillna(False).astype(bool) if "gics" in uni else uni["sector"].notna()
    known = [(num(s), sec, ind, sub) for s, sec, ind, sub, h in
             zip(uni["sic"], uni["sector"], uni["industry"], uni["subindustry"], has) if h and num(s)]
    cw = Crosswalk(known)  # type: ignore[arg-type]
    how_counts: dict[str, int] = defaultdict(int)
    for i in uni.index[~has]:
        c = num(uni.at[i, "cik"])
        desc = meta.get(c, {}).get("sicDescription", "") if c is not None else ""
        (sec, ind, sub), how = cw.classify(num(uni.at[i, "sic"]), desc)
        uni.at[i, "sector"], uni.at[i, "industry"], uni.at[i, "subindustry"] = sec, ind, sub
        how_counts[how] += 1
    uni["sector"] = uni["sector"].fillna("Unknown")
    uni["subindustry"] = uni["subindustry"].fillna("Unknown")
    uni["industry"] = [ind if isinstance(ind, str) and ind else industry_of(sub)
                       for ind, sub in zip(uni["industry"], uni["subindustry"])]
    return uni, dict(how_counts)


def build_real(progress: Progress = _nop, cancelled: Callable[[], bool] | None = None,
               max_tickers: int | None = None, pool: str | None = None) -> dict:
    s = config.load_settings()
    email = (s.get("sec_contact_email") or "").strip()
    if "@" not in email:
        raise ValueError("Set your contact email on the Data page first (SEC EDGAR and Wikipedia require a "
                         "User-Agent with a contact address for automated downloads).")
    sec_ua = f"AlphaFoundry research tool {email}"
    start = s.get("history_start", "2012-01-01")
    end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    raw = config.RAW_DIR
    pool = (pool or s.get("universe_pool") or "broad").lower()
    pool_size = int(s.get("broad_pool_size", 3000) or 3000)

    progress("universe", 0, 2, "Fetching S&P 500/400/600 constituents")
    uni = universe.fetch_universe(email, sec_ua, raw / "universe.json")
    uni["gics"] = True
    if pool == "broad":
        progress("universe", 1, 2, "Fetching every listed US common stock from SEC")
        uni = universe.fetch_broad_candidates(sec_ua, uni, raw / "broad_candidates.json")
    if max_tickers:
        uni = uni.head(max_tickers)
    tickers = list(uni["ticker"])
    progress("universe", 2, 2, f"{len(tickers):,} candidate tickers ({pool} pool)")

    progress("prices", 0, len(tickers), "Downloading daily prices")
    prices = yahoo.download_all(tickers, start, end, raw / "yahoo",
                                progress=lambda d, t, m: progress("prices", d, t, m), cancelled=cancelled)
    if cancelled and cancelled():
        raise RuntimeError("cancelled")
    tickers = [t for t in tickers if t in prices and len(prices[t]["dates"]) > 60]
    if pool == "broad":
        before = len(tickers)
        liquid = set(select_liquid_pool({t: prices[t] for t in tickers}, start, pool_size))
        tickers = [t for t in tickers if t in liquid]
        progress("prices", len(tickers), len(tickers), f"{before:,} priced candidates -> {len(tickers):,} in the "
                                                        f"liquid pool (top {pool_size:,} x {POOL_HEADROOM})")
    uni = uni[uni["ticker"].isin(tickers)].reset_index(drop=True)
    tickers = list(uni["ticker"])
    ciks = [int(c) for c in uni["cik"].dropna().unique()]

    how: dict = {}
    if pool == "broad":
        progress("classification", 0, len(ciks), "Fetching SIC codes (SEC submissions)")
        meta = edgar.fetch_submissions_meta(ciks, raw / "submissions", sec_ua,
                                            progress=lambda d, t, m: progress("classification", d, t, m),
                                            cancelled=cancelled)
        if cancelled and cancelled():
            raise RuntimeError("cancelled")
        drop = {int(c) for c, m in meta.items() if m.get("sic") in NON_OPERATING_SIC}
        uni = uni[[not (c == c and c is not None and int(c) in drop) or bool(g)
                   for c, g in zip(uni["cik"], uni["gics"])]].reset_index(drop=True)
        uni, how = classify_pool(uni, meta)
        tickers = list(uni["ticker"])
        ciks = [int(c) for c in uni["cik"].dropna().unique()]

    progress("fundamentals", 0, len(ciks), "Downloading SEC company facts")
    facts = edgar.download_all(ciks, raw / "edgar", sec_ua,
                               progress=lambda d, t, m: progress("fundamentals", d, t, m), cancelled=cancelled)
    if cancelled and cancelled():
        raise RuntimeError("cancelled")

    progress("assemble", 0, 4, "Aligning calendar")
    all_dates = np.unique(np.concatenate([prices[t]["dates"] for t in tickers]))
    counts = np.zeros(len(all_dates))
    for t in tickers:
        counts[np.searchsorted(all_dates, prices[t]["dates"])] += 1
    dates = all_dates[(counts >= 0.3 * len(tickers)) & (all_dates >= np.datetime64(start))]
    T, N = len(dates), len(tickers)
    est_gb = T * N * 4 * 60 / 1e9
    progress("assemble", 0, 4, f"Aligning {N:,} stocks x {T:,} days (about {est_gb:.1f} GB on disk)")
    fields: dict[str, np.ndarray] = {k: np.full((T, N), np.nan, np.float32)
                                     for k in ("open", "high", "low", "close", "volume", "adjclose")}
    div = np.zeros((T, N), np.float32)
    split = np.ones((T, N), np.float32)
    split_after = np.ones((T, N), np.float64)  # product of split ratios strictly after each date
    exch = []
    for j, t in enumerate(tickers):
        p = prices[t]
        pos = np.searchsorted(dates, p["dates"])
        ok = (pos < T) & (dates[np.minimum(pos, T - 1)] == p["dates"])
        for k in ("open", "high", "low", "close", "volume", "adjclose"):
            fields[k][pos[ok], j] = p[k][ok]
        if len(p.get("div_dates", [])):
            dp = np.searchsorted(dates, p["div_dates"])
            m = dp < T
            div[dp[m], j] = p["div_amount"][m]
        if len(p.get("split_dates", [])):
            for sd, ratio in zip(p["split_dates"], p["split_ratio"]):
                sp = int(np.searchsorted(dates, sd))
                if sp < T and ratio > 0:
                    split[sp, j] = ratio
                    split_after[:sp, j] *= ratio
        exch.append(p.get("exchange", "UNKNOWN"))
    del prices
    close = fields["close"].astype(np.float64)
    adj = fields.pop("adjclose").astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prev = np.vstack([np.full((1, N), np.nan), adj[:-1]])
        ret = adj / prev - 1.0
    del adj, prev
    ret[~np.isfinite(ret)] = np.nan
    ret = np.clip(ret, -0.95, 3.0)
    vol = fields["volume"].astype(np.float64)
    vol[vol <= 0] = np.nan
    csum = np.nancumsum(np.nan_to_num(vol), axis=0)
    cnt = np.cumsum(~np.isnan(vol), axis=0)
    adv20 = np.full((T, N), np.nan)
    num = csum[19:] - np.vstack([np.zeros((1, N)), csum[:-20]])
    den = cnt[19:] - np.vstack([np.zeros((1, N)), cnt[:-20]])
    with np.errstate(divide="ignore", invalid="ignore"):
        adv20[19:] = np.where(den >= 10, num / den, np.nan)
    del csum, cnt, num, den
    dollar = np.nan_to_num(vol * close)
    dcs = np.cumsum(dollar, axis=0)
    del dollar
    adv63d = np.full((T, N), np.nan)
    adv63d[62:] = (dcs[62:] - np.vstack([np.zeros((1, N)), dcs[:-63]])) / 63
    del dcs

    progress("assemble", 1, 4, "Point-in-time fundamentals")
    cik_of = dict(zip(uni["ticker"], uni["cik"]))
    fnames = list(edgar.STOCK) + list(edgar.FLOW) + ["eps", "shares"]
    fund = {k: np.full((T, N), np.nan, np.float32) for k in fnames}
    shares_end = np.full((T, N), np.nan)
    for j, t in enumerate(tickers):
        c = cik_of.get(t)
        if c is None or c != c or int(c) not in facts:
            continue
        ser = facts[int(c)]
        for k in fnames:
            fund[k][:, j] = forward_fill_series(dates, ser.get(k) or [], T)
        # remember which period end the shares figure refers to (for split adjustment)
        sh = ser.get("shares") or []
        if sh:
            ends = forward_fill_series(dates, [(f, e, float(np.datetime64(e, "D").astype(np.int64)))
                                               for f, e, _v in sh], T)
            shares_end[:, j] = ends
    del facts
    # shares in current (split-adjusted) units: multiply by splits after the reporting date
    shares_now = fund.pop("shares").astype(np.float64)
    for j in range(N):
        if not np.isfinite(shares_now[:, j]).any():
            continue
        sp_idx = np.where(split[:, j] != 1)[0]
        if not len(sp_idx):
            continue
        end_days = shares_end[:, j]
        for si in sp_idx:
            sdate = dates[si].astype(np.int64)
            ratio = float(split[si, j])
            # a report whose period ended before the split is in pre-split units
            m = np.isfinite(end_days) & (end_days < sdate)
            shares_now[m, j] *= ratio
    del shares_end
    cap = close * shares_now
    shares_actual = shares_now / split_after
    del shares_now, split_after

    progress("assemble", 2, 4, "Derived fields")
    debt = np.where(np.isnan(fund["debt_lt"]) & np.isnan(fund["debt_st"]), np.nan,
                    np.nan_to_num(fund["debt_lt"]) + np.nan_to_num(fund["debt_st"]))
    liab = np.where(np.isnan(fund["liabilities"]), fund["liab_equity"] - fund["equity"], fund["liabilities"])
    sales = fund["sales"]
    with np.errstate(divide="ignore", invalid="ignore"):
        prev_sales = np.vstack([np.full((252, N), np.nan, np.float32), sales[:-252]]) if T > 252 else \
            np.full_like(sales, np.nan)
        out_fields = {
            "open": fields["open"], "high": fields["high"], "low": fields["low"], "close": fields["close"],
            "volume": vol, "vwap": (fields["high"].astype(np.float64) + fields["low"] + close) / 3.0,
            "returns": ret, "adv20": adv20, "cap": cap, "sharesout": shares_actual, "dividend": div, "split": split,
            "assets": fund["assets"], "assets_curr": fund["assets_curr"], "liabilities": liab,
            "liabilities_curr": fund["liabilities_curr"], "equity": fund["equity"], "cash": fund["cash"],
            "debt": debt, "debt_lt": fund["debt_lt"], "debt_st": fund["debt_st"], "inventory": fund["inventory"],
            "receivable": fund["receivable"], "goodwill": fund["goodwill"],
            "retained_earnings": fund["retained_earnings"], "sales": sales, "revenue": sales,
            "cogs": fund["cogs"], "operating_income": fund["operating_income"], "ebit": fund["operating_income"],
            "ebitda": fund["operating_income"] + np.nan_to_num(fund["da"]), "income": fund["income"],
            "cashflow_op": fund["cashflow_op"], "capex": fund["capex"], "rd_expense": fund["rd_expense"],
            "sga_expense": fund["sga_expense"], "eps": fund["eps"],
            "bookvalue_ps": fund["equity"] / shares_actual, "sales_ps": sales / shares_actual,
            "current_ratio": fund["assets_curr"] / fund["liabilities_curr"],
            "return_equity": fund["income"] / fund["equity"], "return_assets": fund["income"] / fund["assets"],
            "enterprise_value": cap + np.nan_to_num(debt) - np.nan_to_num(fund["cash"]),
            "interest_expense": fund["interest_expense"], "income_tax": fund["income_tax"],
            "pretax_income": fund["pretax_income"], "cashflow_invst": fund["cashflow_invst"],
            "cashflow_fin": fund["cashflow_fin"], "cashflow_dividends": fund["cashflow_dividends"],
            "depre_amort": fund["da"], "buyback": fund["buyback"],
            "working_capital": fund["assets_curr"] - fund["liabilities_curr"],
            "sales_growth": sales / prev_sales - 1.0,
        }
    del prev_sales
    for k in list(out_fields):
        v = np.asarray(out_fields[k], dtype=np.float32)  # float32 in place of float64 copies: keeps memory low
        np.putmask(v, ~np.isfinite(v), np.float32(np.nan))
        out_fields[k] = v
    del fund, close, vol, ret, adv20, cap, shares_actual, debt, liab

    progress("assemble", 3, 4, "Groups and universes")

    def codes(col: str) -> tuple[np.ndarray, list[str]]:
        labels = sorted({str(x) for x in uni[col]})
        m = {lab: i for i, lab in enumerate(labels)}
        return np.array([m[str(x)] for x in uni[col]], dtype=np.int32), labels

    ex_labels = sorted(set(exch))
    groups = {"sector": codes("sector"), "industry": codes("industry"), "subindustry": codes("subindustry"),
              "exchange": (np.array([ex_labels.index(e) for e in exch], dtype=np.int32), ex_labels)}
    dates_s = [str(d) for d in dates]
    if pool == "broad":
        whole = "TOP3000" if N <= 3000 else None
    else:
        whole = "TOP1500"
    universes = liquidity_universes(dates_s, adv63d, N, whole=whole)
    del adv63d
    classification = "GICS (S&P 1500) + SIC crosswalk" if pool == "broad" else "GICS (S&P 1500)"
    write_panel(config.PANELS_DIR, dates_s, tickers, out_fields, groups, universes, source="real",
                extra_meta={"names": list(uni["name"]), "ciks": [None if c != c else int(c) for c in uni["cik"]],
                            "pool": pool, "pool_size": N, "classification": classification,
                            "classified_by": how})
    progress("assemble", 4, 4, "Panel written")
    coverage = {k: round(float(np.isfinite(v[-252:]).mean()), 3) for k, v in out_fields.items()}
    return {"tickers": N, "days": T, "start": dates_s[0], "end": dates_s[-1], "coverage": coverage, "pool": pool,
            "classified_by": how}
