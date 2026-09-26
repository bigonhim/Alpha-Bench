"""Every locally implemented operator vs. a slow reference implementation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphafoundry.engine.ops import minp
from alphafoundry.fastexpr import lower_text


def ev(ctx, text):
    return np.asarray(ctx.evaluate_matrix(lower_text(text)), dtype=np.float64)


def raw(ctx, name):
    return np.asarray(ctx.field(name), dtype=np.float64)


def close(a, b, tol=1e-4):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    na, nb = np.isnan(a), np.isnan(b)
    bad = na != nb
    assert not bad.any(), f"NaN pattern differs at {np.argwhere(bad)[:5].tolist()} ({a[bad][:5]} vs {b[bad][:5]})"
    m = ~na
    np.testing.assert_allclose(a[m], b[m], rtol=tol, atol=tol)


def roll(x, d, fn):
    return pd.DataFrame(x).rolling(d, min_periods=minp(d))


@pytest.mark.parametrize("d", [1, 3, 10])
def test_rolling_moments(ctx, d):
    x = raw(ctx, "x")
    r = pd.DataFrame(x).rolling(d, min_periods=minp(d))
    close(ev(ctx, f"ts_mean(x, {d})"), r.mean().values)
    close(ev(ctx, f"ts_sum(x, {d})"), r.sum().values)
    close(ev(ctx, f"ts_std_dev(x, {d})"), r.std(ddof=0).values)
    sd = r.std(ddof=0).values
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - r.mean().values) / sd
    z[~(sd > 1e-9)] = np.nan
    z[np.isnan(x)] = np.nan
    close(ev(ctx, f"ts_zscore(x, {d})"), z, tol=1e-3)
    av = x - r.mean().values
    close(ev(ctx, f"ts_av_diff(x, {d})"), av)


def test_min_max_arg(ctx):
    x = raw(ctx, "x")
    d = 7
    r = pd.DataFrame(x).rolling(d, min_periods=1)
    close(ev(ctx, f"ts_max(x, {d})"), r.max().values)
    close(ev(ctx, f"ts_min(x, {d})"), r.min().values)
    # days since max, reference loop (ties -> most recent occurrence)
    T, N = x.shape
    ref = np.full((T, N), np.nan)
    for t in range(T):
        for i in range(N):
            w = x[max(0, t - d + 1):t + 1, i]
            if np.isfinite(w).any():
                mx = np.nanmax(w)
                last = max(k for k in range(len(w)) if w[k] == mx)
                ref[t, i] = len(w) - 1 - last
    close(ev(ctx, f"ts_arg_max(x, {d})"), ref)


def test_ts_rank(ctx):
    x = raw(ctx, "x")
    d = 9
    T, N = x.shape
    ref = np.full((T, N), np.nan)
    for t in range(T):
        for i in range(N):
            v = x[t, i]
            w = x[max(0, t - d + 1):t + 1, i]
            w = w[np.isfinite(w)]
            if np.isnan(v) or len(w) < max(2, minp(d)):
                continue
            less = (w < v).sum()
            eq = (w == v).sum() - 1
            ref[t, i] = (less + 0.5 * eq) / (len(w) - 1)
    close(ev(ctx, f"ts_rank(x, {d})"), ref)


def test_delay_delta_backfill(ctx):
    x = raw(ctx, "x")
    close(ev(ctx, "ts_delay(x, 3)"), pd.DataFrame(x).shift(3).values)
    close(ev(ctx, "ts_delta(x, 3)"), x - pd.DataFrame(x).shift(3).values)
    ref = pd.DataFrame(x).ffill(limit=4).values
    close(ev(ctx, "ts_backfill(x, 5)"), ref)


def test_corr_cov_regression(ctx):
    x, y = raw(ctx, "x"), raw(ctx, "y")
    d = 12
    both = np.isfinite(x) & np.isfinite(y)
    xm = np.where(both, x, np.nan)
    ym = np.where(both, y, np.nan)
    T, N = x.shape
    rc = np.full((T, N), np.nan)
    rv = np.full((T, N), np.nan)
    rb = np.full((T, N), np.nan)
    for t in range(T):
        for i in range(N):
            a = xm[max(0, t - d + 1):t + 1, i]
            b = ym[max(0, t - d + 1):t + 1, i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < max(2, minp(d)):
                continue
            a, b = a[m], b[m]
            cov = np.mean(a * b) - a.mean() * b.mean()
            rv[t, i] = cov
            va, vb = a.var(), b.var()
            rc[t, i] = cov / math.sqrt(va * vb)
            if m.sum() >= max(3, minp(d)):
                rb[t, i] = cov / va
    close(ev(ctx, f"ts_corr(x, y, {d})"), rc, tol=1e-3)
    close(ev(ctx, f"ts_covariance(y, x, {d})"), rv, tol=1e-3)
    close(ev(ctx, f"ts_regression(y, x, {d}, rettype=2)"), rb, tol=1e-3)


def test_decay_linear(ctx):
    x = raw(ctx, "x")
    d = 5
    T, N = x.shape
    ref = np.full((T, N), np.nan)
    for t in range(T):
        for i in range(N):
            num = den = 0.0
            for k in range(d):
                u = t - k
                if u >= 0 and np.isfinite(x[u, i]):
                    num += (d - k) * x[u, i]
                    den += d - k
            if den > 0:
                ref[t, i] = num / den
    close(ev(ctx, f"ts_decay_linear(x, {d})"), ref)
    close(ev(ctx, f"ts_decay_linear(x, {d}, dense=true)"),
          np.where(np.isnan(ref), np.nan, _dense(x, d)))


def _dense(x, d):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    full = d * (d + 1) / 2
    for t in range(T):
        for i in range(N):
            num = 0.0
            ok = False
            for k in range(d):
                u = t - k
                if u >= 0 and np.isfinite(x[u, i]):
                    num += (d - k) * x[u, i]
                    ok = True
            if ok:
                out[t, i] = num / full
    return out


def test_median(ctx):
    x = raw(ctx, "x")
    d = 8
    close(ev(ctx, f"ts_median(x, {d})"), pd.DataFrame(x).rolling(d, min_periods=minp(d)).median().values)


def test_cross_sectional(ctx):
    x = raw(ctx, "x")
    close(ev(ctx, "rank(x)"), _row_rank(x))
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    close(ev(ctx, "zscore(x)"), (x - m) / s, tol=1e-3)
    sa = np.nansum(np.abs(x), axis=1, keepdims=True)
    close(ev(ctx, "scale(x)"), x / sa)
    w = np.clip(x, m - 2 * s, m + 2 * s)
    close(ev(ctx, "winsorize(x, std=2)"), w, tol=1e-3)


def _row_rank(x):
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        s = pd.Series(x[t])
        n = s.notna().sum()
        if n == 1:
            out[t] = np.where(s.notna(), 0.5, np.nan)
        elif n > 1:
            out[t] = ((s.rank(method="average") - 1) / (n - 1)).values
    return out


def test_group_ops(ctx, tiny_panel):
    x = raw(ctx, "x")
    codes, _ = tiny_panel.group("subindustry")
    T, N = x.shape
    neut = np.full((T, N), np.nan)
    grank = np.full((T, N), np.nan)
    gmean = np.full((T, N), np.nan)
    for t in range(T):
        df = pd.DataFrame({"v": x[t], "g": codes})
        df = df[(df.g >= 0)]
        mean = df.groupby("g")["v"].transform("mean")
        cnt = df.groupby("g")["v"].transform("count")
        rk = df.groupby("g")["v"].rank(method="average")
        idx = df.index.values
        neut[t, idx] = (df.v - mean).values
        gmean[t, idx] = mean.values
        r = np.where(cnt > 1, (rk - 1) / (cnt - 1), 0.5)
        grank[t, idx] = np.where(df.v.notna(), r, np.nan)
    close(ev(ctx, "group_neutralize(x, subindustry)"), neut)
    close(ev(ctx, "group_mean(x, 1, subindustry)"), gmean)
    close(ev(ctx, "group_rank(x, subindustry)"), grank)


def test_event_ops(ctx):
    s = raw(ctx, "step")
    T, N = s.shape
    ref = np.full((T, N), np.nan)
    for i in range(N):
        last = -1
        prev = None
        for t in range(T):
            if prev is None or s[t, i] != prev:
                last = t
            prev = s[t, i]
            ref[t, i] = t - last
    close(ev(ctx, "days_from_last_change(step)"), ref)
    # trade_when: trigger on even days, exit never
    got = ev(ctx, "trade_when(x > 0, y, -1)")
    x, y = raw(ctx, "x"), raw(ctx, "y")
    ref = np.full((T, N), np.nan)
    for i in range(N):
        prev = np.nan
        for t in range(T):
            if np.isfinite(x[t, i]) and x[t, i] > 0:
                prev = y[t, i]
            ref[t, i] = prev
    close(got, ref)


def test_arithmetic_and_logic(ctx):
    x, y = raw(ctx, "x"), raw(ctx, "y")
    close(ev(ctx, "x + y"), x + y)
    close(ev(ctx, "x / y"), np.where(y == 0, np.nan, x / y))
    close(ev(ctx, "max(x, 0)"), np.maximum(x, 0))
    close(ev(ctx, "signed_power(x, 2)"), np.sign(x) * x ** 2)
    close(ev(ctx, "x > y ? x : y"), np.where(np.isnan(x) | np.isnan(y), np.nan, np.where(x > y, x, y)))
    close(ev(ctx, "abs(x) + sign(y)"), np.abs(x) + np.sign(y))


def test_bucket_groups(ctx):
    got = ev(ctx, 'group_neutralize(x, bucket(rank(y), range="0,1,0.5"))')
    x = raw(ctx, "x")
    ry = _row_rank(raw(ctx, "y"))
    codes = np.where(np.isnan(ry), -1, np.searchsorted(np.array([0.0, 0.5, 1.0]), ry, side="right"))
    ref = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        for g in np.unique(codes[t][codes[t] >= 0]):
            m = (codes[t] == g) & np.isfinite(x[t])
            ref[t, m] = x[t, m] - x[t, m].mean()
    close(got, ref)


def test_cache_does_not_alias(ctx):
    a = ev(ctx, "rank(x)")
    b = ev(ctx, "rank(x) * 2")
    c = ev(ctx, "rank(x)")
    close(a, c)
    close(b, 2 * a)
