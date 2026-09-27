"""Deterministic synthetic dataset with weak embedded effects.

The demo lets the whole app run before real data is downloaded and gives tests a fixture. Returns
follow a market + sector factor model plus small, known effects that are visible at delay 1:
weekly reversal, 12-1 momentum, value (earnings yield from *reported* fundamentals) and quality
(return on assets). Classic alphas therefore show the expected signs with modest Sharpe ratios.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from ..engine.panel import UNIVERSE_SIZES, write_panel

SECTORS = ["Energy", "Materials", "Industrials", "Consumer Discretionary", "Consumer Staples", "Health Care",
           "Financials", "Information Technology", "Communication Services", "Utilities", "Real Estate"]


def business_days(start: str, n: int) -> list[str]:
    d = dt.date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _z(x: np.ndarray) -> np.ndarray:
    m = np.isfinite(x)
    if m.sum() < 3:
        return np.zeros_like(x)
    v = x[m]
    z = np.zeros_like(x)
    z[m] = (v - v.mean()) / (v.std() + 1e-12)
    return np.clip(z, -3, 3)


def build_demo(root: Path, n_stocks: int = 300, n_days: int = 2000, seed: int = 7, start: str = "2017-01-02") -> Path:
    rng = np.random.default_rng(seed)
    T, N = n_days, n_stocks
    dates = business_days(start, T)
    tickers = [f"DEMO{i:03d}" for i in range(N)]

    n_sub = min(60, max(3, N // 5))
    subind = rng.integers(0, n_sub, N)
    sub_to_ind = np.arange(n_sub) // 2
    n_ind = int(sub_to_ind.max()) + 1
    ind_to_sec = np.arange(n_ind) % len(SECTORS)
    industry = sub_to_ind[subind]
    sector = ind_to_sec[industry]

    size0 = rng.lognormal(mean=22.0, sigma=1.1, size=N)
    beta = rng.normal(1.0, 0.25, N)
    quality = rng.normal(0, 1, N)
    idio_vol = 0.013 + 0.010 * rng.random(N)

    # ---------------------------------------------------------------- fundamentals (quarterly, 45d lag)
    q_every, lag = 63, 45
    asset_turn = np.clip(rng.normal(0.9, 0.3, N), 0.2, 2.5)
    eq_ratio = np.clip(rng.normal(0.45, 0.12, N), 0.1, 0.9)
    base_roa = 0.045 + 0.025 * quality
    assets = np.full((T, N), np.nan)
    income = np.full((T, N), np.nan)
    sales = np.full((T, N), np.nan)
    cfo = np.full((T, N), np.nan)
    lvl = size0 * rng.lognormal(0, 0.4, N)
    for q0 in range(0, T, q_every):
        lvl = lvl * np.exp(rng.normal(0.012, 0.03, N))
        roa_q = base_roa + rng.normal(0, 0.012, N)
        a = q0 + lag
        if a < T:
            assets[a:] = lvl
            income[a:] = lvl * roa_q
            sales[a:] = lvl * asset_turn * np.exp(rng.normal(0, 0.03, N))
            cfo[a:] = lvl * (roa_q + rng.normal(0.01, 0.015, N))
    equity = assets * eq_ratio
    liabilities = assets - equity
    op_inc = income * 1.3
    cogs = sales * np.clip(rng.normal(0.62, 0.1, N), 0.2, 0.9)
    cash = assets * np.clip(rng.normal(0.1, 0.04, N), 0.01, None)
    debt = liabilities * 0.6

    # ---------------------------------------------------------------- prices with embedded effects
    mkt = rng.normal(0.0003, 0.010, T)
    sec_f = rng.normal(0, 0.006, (T, len(SECTORS)))
    noise = rng.standard_t(5, (T, N)) * idio_vol / np.sqrt(5 / 3)
    shares = size0 / (20 + 80 * rng.random(N))
    ret = np.zeros((T, N))
    close = np.zeros((T, N))
    px0 = size0 / shares
    prev_close = px0.copy()
    for t in range(T):
        r = beta * mkt[t] + sec_f[t, sector] + noise[t]
        if t >= 6:
            r += -0.0025 * ret[t - 6:t - 1].sum(axis=0)              # weekly reversal (lags 2..6)
            r += -0.02 * ret[t - 1]                               # 1-day reversal (delay-0 only)
        if t >= 252:
            mom = ret[t - 252:t - 21].sum(axis=0)
            r += 0.00012 * np.tanh(mom / 0.3)                      # 12-1 momentum
        cap_prev = prev_close * shares
        r += 0.00018 * _z(income[t] / cap_prev)                   # value (reported earnings yield)
        r += 0.00010 * _z(income[t] / assets[t])                  # quality (ROA)
        r = np.clip(r, -0.4, 0.4)
        ret[t] = r
        prev_close = prev_close * (1 + r)
        close[t] = prev_close
    gap = rng.normal(0, 0.004, (T, N))
    open_ = np.vstack([px0[None, :], close[:-1]]) * (1 + gap)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (T, N))))
    cap = close * shares
    turn = 0.004 * np.exp(rng.normal(0, 0.35, (T, N))) * (1 + 25 * np.abs(ret))
    volume = shares * turn
    cs = np.cumsum(volume, axis=0)
    adv20 = np.full((T, N), np.nan)
    adv20[19:] = (cs[19:] - np.vstack([np.zeros((1, N)), cs[:-20]])) / 20
    dollar = volume * close
    csd = np.cumsum(dollar, axis=0)
    adv63d = np.full((T, N), np.nan)
    adv63d[62:] = (csd[62:] - np.vstack([np.zeros((1, N)), csd[:-63]])) / 63
    vwap = (hi + lo + close) / 3
    eps = income / shares

    fields = {
        "open": open_, "high": hi, "low": lo, "close": close, "vwap": vwap, "volume": volume, "returns": ret,
        "adv20": adv20, "cap": cap, "sharesout": np.tile(shares, (T, 1)), "assets": assets, "liabilities": liabilities,
        "equity": equity, "sales": sales, "revenue": sales, "income": income, "operating_income": op_inc,
        "ebit": op_inc, "ebitda": op_inc * 1.2, "cashflow_op": cfo, "cogs": cogs, "cash": cash, "debt": debt,
        "eps": eps, "enterprise_value": cap + debt - cash, "bookvalue_ps": equity / shares,
        "return_equity": income / equity, "return_assets": income / assets,
        "current_ratio": np.where(np.isnan(assets), np.nan, 1.5 + 0.2 * quality),
    }
    groups = {
        "sector": (sector.astype(np.int32), SECTORS),
        "industry": (industry.astype(np.int32), [f"Industry {i}" for i in range(n_ind)]),
        "subindustry": (subind.astype(np.int32), [f"Sub-industry {i}" for i in range(n_sub)]),
        "exchange": ((np.arange(N) % 2).astype(np.int32), ["NYSE", "NASDAQ"]),
    }
    universes = liquidity_universes(dates, adv63d, N)
    write_panel(root, dates, tickers, {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}, groups,
                universes, source="demo", extra_meta={"names": [f"Demo Corp {i}" for i in range(N)]})
    return root


def liquidity_universes(dates: list[str], dollar_adv: np.ndarray, N: int,
                        whole: str | None = "TOP1500") -> dict[str, np.ndarray]:
    """Monthly-rebalanced TOPn masks by trailing dollar volume.

    ``whole`` names the universe that is simply the whole pool (TOP1500 for the S&P 1500 pool). The broad pool
    passes None when it is larger than 3000 names, so TOP3000 is a true top-3000 liquidity cut like BRAIN's."""
    T = len(dates)
    liq = np.nan_to_num(dollar_adv, nan=0.0)
    month = np.array([d[:7] for d in dates])
    reb = np.r_[True, month[1:] != month[:-1]]
    universes: dict[str, np.ndarray] = {}
    for uname, size_n in UNIVERSE_SIZES.items():
        if size_n >= N or uname == whole or (whole == "TOP1500" and size_n > 1500):
            continue
        mask = np.zeros((T, N), bool)
        cur = np.zeros(N, bool)
        for t in range(T):
            if reb[t] or not cur.any():
                order = np.argsort(-liq[t])
                cur = np.zeros(N, bool)
                top = order[:size_n]
                cur[top] = liq[t, top] > 0
            mask[t] = cur
        universes[uname] = mask
    if whole:
        universes[whole] = liq > 0
    return universes
