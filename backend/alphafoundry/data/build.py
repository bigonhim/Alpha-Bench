"""Build the real-data panel: universe -> Yahoo prices -> SEC fundamentals -> fields, groups, universes."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Callable

import numpy as np

from .. import config
from ..engine.panel import write_panel
from . import edgar, universe, yahoo
from .demo import liquidity_universes

log = logging.getLogger("alphafoundry.data")

Progress = Callable[[str, int, int, str], None]


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


def build_real(progress: Progress = _nop, cancelled: Callable[[], bool] | None = None,
               max_tickers: int | None = None) -> dict:
    s = config.load_settings()
    email = (s.get("sec_contact_email") or "").strip()
    if "@" not in email:
        raise ValueError("Set your contact email on the Data page first (SEC EDGAR and Wikipedia require a "
                         "User-Agent with a contact address for automated downloads).")
    sec_ua = f"AlphaFoundry research tool {email}"
    start = s.get("history_start", "2012-01-01")
    end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    raw = config.RAW_DIR

    progress("universe", 0, 1, "Fetching S&P 500/400/600 constituents")
    uni = universe.fetch_universe(email, sec_ua, raw / "universe.json")
    if max_tickers:
        uni = uni.head(max_tickers)
    tickers = list(uni["ticker"])
    progress("universe", 1, 1, f"{len(tickers)} tickers")

    progress("prices", 0, len(tickers), "Downloading daily prices")
    prices = yahoo.download_all(tickers, start, end, raw / "yahoo",
                                progress=lambda d, t, m: progress("prices", d, t, m), cancelled=cancelled)
    if cancelled and cancelled():
        raise RuntimeError("cancelled")
    tickers = [t for t in tickers if t in prices and len(prices[t]["dates"]) > 60]
    uni = uni[uni["ticker"].isin(tickers)].reset_index(drop=True)
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
    close = fields["close"].astype(np.float64)
    adj = fields["adjclose"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prev = np.vstack([np.full((1, N), np.nan), adj[:-1]])
        ret = adj / prev - 1.0
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
    dollar = np.nan_to_num(vol * close)
    dcs = np.cumsum(dollar, axis=0)
    adv63d = np.full((T, N), np.nan)
    adv63d[62:] = (dcs[62:] - np.vstack([np.zeros((1, N)), dcs[:-63]])) / 63

    progress("assemble", 1, 4, "Point-in-time fundamentals")
    cik_of = dict(zip(uni["ticker"], uni["cik"]))
    fnames = list(edgar.STOCK) + list(edgar.FLOW) + ["eps", "shares"]
    fund = {k: np.full((T, N), np.nan, np.float32) for k in fnames}
    shares_end = np.full((T, N), np.nan)
    for j, t in enumerate(tickers):
        c = cik_of.get(t)
        if c is None or np.isnan(c) or int(c) not in facts:
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
    # shares in current (split-adjusted) units: multiply by splits after the reporting date
    shares_now = fund["shares"].astype(np.float64)
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
    cap = close * shares_now
    shares_actual = shares_now / split_after

    progress("assemble", 2, 4, "Derived fields")
    debt = np.where(np.isnan(fund["debt_lt"]) & np.isnan(fund["debt_st"]), np.nan,
                    np.nan_to_num(fund["debt_lt"]) + np.nan_to_num(fund["debt_st"]))
    liab = np.where(np.isnan(fund["liabilities"]), fund["liab_equity"] - fund["equity"], fund["liabilities"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out_fields = {
            "open": fields["open"], "high": fields["high"], "low": fields["low"], "close": fields["close"],
            "volume": vol, "vwap": (fields["high"].astype(np.float64) + fields["low"] + close) / 3.0,
            "returns": ret, "adv20": adv20, "cap": cap, "sharesout": shares_actual, "dividend": div, "split": split,
            "assets": fund["assets"], "assets_curr": fund["assets_curr"], "liabilities": liab,
            "liabilities_curr": fund["liabilities_curr"], "equity": fund["equity"], "cash": fund["cash"],
            "debt": debt, "debt_lt": fund["debt_lt"], "debt_st": fund["debt_st"], "inventory": fund["inventory"],
            "receivable": fund["receivable"], "goodwill": fund["goodwill"],
            "retained_earnings": fund["retained_earnings"], "sales": fund["sales"], "revenue": fund["sales"],
            "cogs": fund["cogs"], "operating_income": fund["operating_income"], "ebit": fund["operating_income"],
            "ebitda": fund["operating_income"] + np.nan_to_num(fund["da"]), "income": fund["income"],
            "cashflow_op": fund["cashflow_op"], "capex": fund["capex"], "rd_expense": fund["rd_expense"],
            "sga_expense": fund["sga_expense"], "eps": fund["eps"],
            "bookvalue_ps": fund["equity"] / shares_actual, "sales_ps": fund["sales"] / shares_actual,
            "current_ratio": fund["assets_curr"] / fund["liabilities_curr"],
            "return_equity": fund["income"] / fund["equity"], "return_assets": fund["income"] / fund["assets"],
            "enterprise_value": cap + np.nan_to_num(debt) - np.nan_to_num(fund["cash"]),
        }
    for k, v in out_fields.items():
        v = np.asarray(v, dtype=np.float64)
        v[~np.isfinite(v)] = np.nan
        out_fields[k] = v.astype(np.float32)

    progress("assemble", 3, 4, "Groups and universes")

    def codes(col: str) -> tuple[np.ndarray, list[str]]:
        labels = sorted({str(x) for x in uni[col]})
        m = {lab: i for i, lab in enumerate(labels)}
        return np.array([m[str(x)] for x in uni[col]], dtype=np.int32), labels

    ex_labels = sorted(set(exch))
    groups = {"sector": codes("sector"), "industry": codes("industry"), "subindustry": codes("subindustry"),
              "exchange": (np.array([ex_labels.index(e) for e in exch], dtype=np.int32), ex_labels)}
    dates_s = [str(d) for d in dates]
    universes = liquidity_universes(dates_s, adv63d, N)
    write_panel(config.PANELS_DIR, dates_s, tickers, out_fields, groups, universes, source="real",
                extra_meta={"names": list(uni["name"]), "ciks": [None if c != c else int(c) for c in uni["cik"]]})
    progress("assemble", 4, 4, "Panel written")
    coverage = {k: round(float(np.isfinite(v[-252:]).mean()), 3) for k, v in out_fields.items()}
    return {"tickers": N, "days": T, "start": dates_s[0], "end": dates_s[-1], "coverage": coverage}
