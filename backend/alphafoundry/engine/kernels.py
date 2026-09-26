"""Numba kernels for time-series, cross-sectional and group operators.

Conventions
-----------
* Panels are C-contiguous ``float32`` arrays of shape (T, N): time on axis 0, instruments on axis 1.
* NaN means "missing". Rolling statistics use the valid observations in the window and require at
  least ``minp`` of them; otherwise the output is NaN.
* Accumulation happens in float64. Running-sum kernels re-center and recompute exactly every
  ``d`` steps so floating-point drift stays bounded even for large-magnitude inputs (fundamentals).
* Kernels that carry state along time process column chunks in parallel (cache friendly); kernels
  whose rows are independent parallelize over rows.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

CH = 64  # columns per chunk for column-parallel kernels
NAN32 = np.float32(np.nan)


# =========================================================================== rolling moments
# mode: 0 mean, 1 sum, 2 std, 3 zscore, 4 ir, 5 av_diff, 6 count_nans, 7 skew, 8 kurt, 9 var, 10 count

@njit(parallel=True, cache=True, fastmath=False)
def roll_moments(x, d, minp, mode):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    nch = (N + CH - 1) // CH
    for c in prange(nch):
        i0 = c * CH
        i1 = min(N, i0 + CH)
        m = i1 - i0
        cnt = np.zeros(m, np.int64)
        s1 = np.zeros(m)
        s2 = np.zeros(m)
        s3 = np.zeros(m)
        s4 = np.zeros(m)
        K = np.zeros(m)
        for t in range(T):
            if t % d == 0:
                # exact recompute over window [t-d+1, t-1] (current value added below) with new center
                lo = max(0, t - d + 1)
                for j in range(m):
                    i = i0 + j
                    n = 0
                    acc = 0.0
                    for u in range(lo, t + 1):
                        v = x[u, i]
                        if v == v:
                            n += 1
                            acc += v
                    K[j] = acc / n if n > 0 else 0.0
                    cnt[j] = 0
                    s1[j] = 0.0
                    s2[j] = 0.0
                    s3[j] = 0.0
                    s4[j] = 0.0
                    for u in range(lo, t):
                        v = x[u, i]
                        if v == v:
                            z = v - K[j]
                            cnt[j] += 1
                            s1[j] += z
                            z2 = z * z
                            s2[j] += z2
                            s3[j] += z2 * z
                            s4[j] += z2 * z2
            for j in range(m):
                i = i0 + j
                v = x[t, i]
                if v == v:
                    z = v - K[j]
                    cnt[j] += 1
                    s1[j] += z
                    z2 = z * z
                    s2[j] += z2
                    s3[j] += z2 * z
                    s4[j] += z2 * z2
                if t >= d and t % d != 0:
                    u = x[t - d, i]
                    if u == u:
                        z = u - K[j]
                        cnt[j] -= 1
                        s1[j] -= z
                        z2 = z * z
                        s2[j] -= z2
                        s3[j] -= z2 * z
                        s4[j] -= z2 * z2
                n = cnt[j]
                if mode == 6:
                    lo = t - d + 1
                    w = d if lo >= 0 else t + 1
                    out[t, i] = w - n
                    continue
                if mode == 10:
                    out[t, i] = n
                    continue
                if n < minp or n == 0:
                    continue
                mean_z = s1[j] / n
                mean = K[j] + mean_z
                if mode == 0:
                    out[t, i] = mean
                elif mode == 1:
                    out[t, i] = n * K[j] + s1[j]
                elif mode == 5:
                    if v == v:
                        out[t, i] = v - mean
                else:
                    var = s2[j] / n - mean_z * mean_z
                    if var < 0.0:
                        var = 0.0
                    if mode == 9:
                        out[t, i] = var
                        continue
                    sd = math.sqrt(var)
                    if mode == 2:
                        out[t, i] = sd
                    elif mode == 3:
                        if v == v and sd > 1e-12 * (abs(mean) + 1e-30):
                            out[t, i] = (v - mean) / sd
                    elif mode == 4:
                        if sd > 1e-12 * (abs(mean) + 1e-30):
                            out[t, i] = mean / sd
                    elif mode == 7:
                        if sd > 0.0 and n > 2:
                            m3 = s3[j] / n - 3.0 * mean_z * s2[j] / n + 2.0 * mean_z ** 3
                            out[t, i] = m3 / (sd ** 3)
                    elif mode == 8:
                        if sd > 0.0 and n > 3:
                            m4 = (s4[j] / n - 4.0 * mean_z * s3[j] / n + 6.0 * mean_z * mean_z * s2[j] / n
                                  - 3.0 * mean_z ** 4)
                            out[t, i] = m4 / (var * var) - 3.0
    return out


# =========================================================================== rolling min / max / argmin / argmax
# mode: 0 min, 1 max, 2 days since min, 3 days since max

@njit(parallel=True, cache=True)
def roll_minmax(x, d, minp, mode):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    want_max = mode == 1 or mode == 3
    for i in prange(N):
        dq = np.empty(d + 1, np.int64)  # circular deque of time indices
        head = 0
        size = 0
        cnt = 0
        for t in range(T):
            # expire
            old = t - d
            if old >= 0:
                u = x[old, i]
                if u == u:
                    cnt -= 1
                if size > 0 and dq[head] == old:
                    head = (head + 1) % (d + 1)
                    size -= 1
            v = x[t, i]
            if v == v:
                cnt += 1
                # pop dominated from the back
                while size > 0:
                    back = (head + size - 1) % (d + 1)
                    bv = x[dq[back], i]
                    if (want_max and bv <= v) or ((not want_max) and bv >= v):
                        size -= 1
                    else:
                        break
                dq[(head + size) % (d + 1)] = t
                size += 1
            if size > 0 and cnt >= minp:
                idx = dq[head]
                if mode <= 1:
                    out[t, i] = x[idx, i]
                else:
                    out[t, i] = t - idx
    return out


# =========================================================================== rolling rank

@njit(parallel=True, cache=True)
def roll_rank(x, d, minp, constant):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        less = np.zeros(N, np.float64)
        eq = np.zeros(N, np.float64)
        cnt = np.zeros(N, np.int64)
        lo = max(0, t - d + 1)
        for u in range(lo, t + 1):
            for i in range(N):
                a = x[t, i]
                b = x[u, i]
                if a == a and b == b:
                    cnt[i] += 1
                    if b < a:
                        less[i] += 1.0
                    elif b == a:
                        eq[i] += 1.0
        for i in range(N):
            n = cnt[i]
            if n >= minp and n >= 2 and x[t, i] == x[t, i]:
                out[t, i] = (less[i] + 0.5 * (eq[i] - 1.0)) / (n - 1) + constant
    return out


# =========================================================================== rolling order statistics
# percentile with linear interpolation (median: q = 0.5)

@njit(parallel=True, cache=True)
def roll_percentile(x, d, minp, q):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for i in prange(N):
        buf = np.empty(d, np.float64)
        n = 0
        for t in range(T):
            old = t - d
            if old >= 0:
                u = x[old, i]
                if u == u:
                    # remove u from sorted buf
                    lo = 0
                    hi = n
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if buf[mid] < u:
                            lo = mid + 1
                        else:
                            hi = mid
                    for k in range(lo, n - 1):
                        buf[k] = buf[k + 1]
                    n -= 1
            v = x[t, i]
            if v == v:
                lo = 0
                hi = n
                while lo < hi:
                    mid = (lo + hi) // 2
                    if buf[mid] < v:
                        lo = mid + 1
                    else:
                        hi = mid
                for k in range(n, lo, -1):
                    buf[k] = buf[k - 1]
                buf[lo] = v
                n += 1
            if n >= minp and n > 0:
                pos = q * (n - 1)
                k0 = int(math.floor(pos))
                k1 = min(k0 + 1, n - 1)
                fr = pos - k0
                out[t, i] = buf[k0] * (1.0 - fr) + buf[k1] * fr
    return out


# =========================================================================== scan-type rolling kernels
# mode: 0 product, 1 central moment k

@njit(parallel=True, cache=True)
def roll_scan(x, d, minp, mode, k):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        lo = max(0, t - d + 1)
        acc = np.zeros(N)
        cnt = np.zeros(N, np.int64)
        if mode == 0:
            acc[:] = 1.0
            for u in range(lo, t + 1):
                for i in range(N):
                    v = x[u, i]
                    if v == v:
                        acc[i] *= v
                        cnt[i] += 1
            for i in range(N):
                if cnt[i] >= minp:
                    out[t, i] = acc[i]
        else:
            for u in range(lo, t + 1):
                for i in range(N):
                    v = x[u, i]
                    if v == v:
                        acc[i] += v
                        cnt[i] += 1
            mom = np.zeros(N)
            for u in range(lo, t + 1):
                for i in range(N):
                    v = x[u, i]
                    if v == v and cnt[i] > 0:
                        mom[i] += (v - acc[i] / cnt[i]) ** k
            for i in range(N):
                if cnt[i] >= minp and cnt[i] > 0:
                    out[t, i] = mom[i] / cnt[i]
    return out


# =========================================================================== rolling pairwise statistics
# mode: 0 corr, 1 cov(y, x), 2 regression (rettype in ``rt``)

@njit(parallel=True, cache=True)
def roll_pair(y, x, d, minp, mode, rt):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    nch = (N + CH - 1) // CH
    for c in prange(nch):
        i0 = c * CH
        i1 = min(N, i0 + CH)
        m = i1 - i0
        n_ = np.zeros(m, np.int64)
        sx = np.zeros(m)
        sy = np.zeros(m)
        sxx = np.zeros(m)
        syy = np.zeros(m)
        sxy = np.zeros(m)
        kx = np.zeros(m)
        ky = np.zeros(m)
        for t in range(T):
            if t % d == 0:
                lo = max(0, t - d + 1)
                for j in range(m):
                    i = i0 + j
                    n = 0
                    ax = 0.0
                    ay = 0.0
                    for u in range(lo, t + 1):
                        a = x[u, i]
                        b = y[u, i]
                        if a == a and b == b:
                            n += 1
                            ax += a
                            ay += b
                    kx[j] = ax / n if n > 0 else 0.0
                    ky[j] = ay / n if n > 0 else 0.0
                    n_[j] = 0
                    sx[j] = 0.0
                    sy[j] = 0.0
                    sxx[j] = 0.0
                    syy[j] = 0.0
                    sxy[j] = 0.0
                    for u in range(lo, t):
                        a = x[u, i]
                        b = y[u, i]
                        if a == a and b == b:
                            a -= kx[j]
                            b -= ky[j]
                            n_[j] += 1
                            sx[j] += a
                            sy[j] += b
                            sxx[j] += a * a
                            syy[j] += b * b
                            sxy[j] += a * b
            for j in range(m):
                i = i0 + j
                a0 = x[t, i]
                b0 = y[t, i]
                if a0 == a0 and b0 == b0:
                    a = a0 - kx[j]
                    b = b0 - ky[j]
                    n_[j] += 1
                    sx[j] += a
                    sy[j] += b
                    sxx[j] += a * a
                    syy[j] += b * b
                    sxy[j] += a * b
                if t >= d and t % d != 0:
                    a = x[t - d, i]
                    b = y[t - d, i]
                    if a == a and b == b:
                        a -= kx[j]
                        b -= ky[j]
                        n_[j] -= 1
                        sx[j] -= a
                        sy[j] -= b
                        sxx[j] -= a * a
                        syy[j] -= b * b
                        sxy[j] -= a * b
                n = n_[j]
                if n < minp or n < 2:
                    continue
                mx = sx[j] / n
                my = sy[j] / n
                vxx = sxx[j] / n - mx * mx
                vyy = syy[j] / n - my * my
                cxy = sxy[j] / n - mx * my
                if mode == 1:
                    out[t, i] = cxy
                    continue
                if mode == 0:
                    if vxx > 1e-20 and vyy > 1e-20:
                        r = cxy / math.sqrt(vxx * vyy)
                        out[t, i] = max(-1.0, min(1.0, r))
                    continue
                # regression y = alpha + beta x
                if vxx <= 1e-20:
                    continue
                beta = cxy / vxx
                alpha = (my + ky[j]) - beta * (mx + kx[j])
                if rt == 2:
                    out[t, i] = beta
                elif rt == 1:
                    out[t, i] = alpha
                elif rt == 0 or rt == 3:
                    if a0 == a0 and b0 == b0:
                        fit = alpha + beta * a0
                        out[t, i] = (b0 - fit) if rt == 0 else fit
                else:
                    sst = vyy * n
                    r2 = (cxy * cxy) / (vxx * vyy) if vyy > 1e-20 else 0.0
                    sse = sst * (1.0 - r2)
                    if rt == 4:
                        out[t, i] = sse
                    elif rt == 5:
                        out[t, i] = sst
                    elif rt == 6:
                        out[t, i] = r2
                    elif n > 2:
                        mse = sse / (n - 2)
                        if rt == 7:
                            out[t, i] = mse
                        elif rt == 8:
                            out[t, i] = math.sqrt(mse / (vxx * n))
                        elif rt == 9:
                            xm = mx + kx[j]
                            out[t, i] = math.sqrt(mse * (1.0 / n + xm * xm / (vxx * n)))
    return out


# =========================================================================== decays

@njit(parallel=True, cache=True)
def decay_linear(x, d, dense):
    """Weights d (today) ... 1 (d-1 days ago). dense=False renormalizes over valid values."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    full = d * (d + 1) / 2.0
    for i in prange(N):
        for t in range(T):
            w = 0.0
            ws = 0.0
            lo = max(0, t - d + 1)
            any_valid = False
            for u in range(lo, t + 1):
                v = x[u, i]
                wt = d - (t - u)
                if v == v:
                    w += wt * v
                    ws += wt
                    any_valid = True
            if any_valid:
                out[t, i] = w / full if dense else w / ws
    return out


@njit(parallel=True, cache=True)
def decay_linear_fast(x, d):
    """O(1)-per-step linear decay with NaN renormalization (dense=False). Recomputed every d steps."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    nch = (N + CH - 1) // CH
    for c in prange(nch):
        i0 = c * CH
        i1 = min(N, i0 + CH)
        m = i1 - i0
        S = np.zeros(m)   # sum of values in window
        W = np.zeros(m)   # weighted sum
        C = np.zeros(m)   # count of valid
        CW = np.zeros(m)  # weighted count
        for t in range(T):
            if t % d == 0:
                lo = max(0, t - d + 1)
                for j in range(m):
                    i = i0 + j
                    s = 0.0
                    w = 0.0
                    cc = 0.0
                    cw = 0.0
                    for u in range(lo, t + 1):
                        v = x[u, i]
                        if v == v:
                            wt = d - (t - u)
                            s += v
                            w += wt * v
                            cc += 1.0
                            cw += wt
                    S[j] = s
                    W[j] = w
                    C[j] = cc
                    CW[j] = cw
            else:
                for j in range(m):
                    i = i0 + j
                    # every existing weight drops by 1; the value leaving (t-d) had weight 1 -> 0
                    W[j] -= S[j]
                    CW[j] -= C[j]
                    if t - d >= 0:
                        u = x[t - d, i]
                        if u == u:
                            S[j] -= u
                            C[j] -= 1.0
                    v = x[t, i]
                    if v == v:
                        S[j] += v
                        W[j] += d * v
                        C[j] += 1.0
                        CW[j] += d
            for j in range(m):
                if C[j] > 0.5 and CW[j] > 0:
                    out[t, i0 + j] = W[j] / CW[j]
    return out


@njit(parallel=True, cache=True)
def decay_exp(x, d, f):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for i in prange(N):
        for t in range(T):
            w = 0.0
            ws = 0.0
            wt = 1.0
            lo = max(0, t - d + 1)
            for u in range(t, lo - 1, -1):
                v = x[u, i]
                if v == v:
                    w += wt * v
                    ws += wt
                wt *= f
            if ws > 0:
                out[t, i] = w / ws
    return out


@njit(parallel=True, cache=True)
def ema(x, lam):
    """y_t = lam*x_t + (1-lam)*y_{t-1}; NaN input keeps the previous value."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for i in prange(N):
        prev = np.nan
        for t in range(T):
            v = x[t, i]
            if v == v:
                if prev == prev:
                    prev = lam * v + (1.0 - lam) * prev
                else:
                    prev = v
            out[t, i] = prev
    return out


# =========================================================================== fill / event kernels

@njit(parallel=True, cache=True)
def backfill(x, lookback, k):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    for i in prange(N):
        for t in range(T):
            v = x[t, i]
            if v == v and k == 1:
                out[t, i] = v
                continue
            found = 0
            res = np.nan
            lo = max(0, t - lookback + 1)
            for u in range(t, lo - 1, -1):
                w = x[u, i]
                if w == w:
                    found += 1
                    if found == k:
                        res = w
                        break
            out[t, i] = res
    return out


@njit(parallel=True, cache=True)
def ffill_limit(x, limit):
    """Forward fill with a maximum staleness of ``limit`` rows (fast path for ts_backfill k=1)."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    for i in prange(N):
        last = np.nan
        last_t = -10 ** 9
        for t in range(T):
            v = x[t, i]
            if v == v:
                last = v
                last_t = t
                out[t, i] = v
            elif t - last_t < limit:
                out[t, i] = last
            else:
                out[t, i] = np.nan
    return out


@njit(parallel=True, cache=True)
def days_from_last_change(x):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for i in prange(N):
        prev = np.nan
        last_change = -1
        for t in range(T):
            v = x[t, i]
            if v == v:
                if prev != prev or v != prev:
                    last_change = t
                prev = v
            if last_change >= 0:
                out[t, i] = t - last_change
    return out


@njit(parallel=True, cache=True)
def last_diff_value(x, d):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for i in prange(N):
        for t in range(T):
            v = x[t, i]
            if v != v:
                continue
            lo = max(0, t - d)
            for u in range(t - 1, lo - 1, -1):
                w = x[u, i]
                if w == w and w != v:
                    out[t, i] = w
                    break
    return out


@njit(parallel=True, cache=True)
def trade_when(trig, alpha, exit_):
    T, N = alpha.shape
    out = np.empty((T, N), dtype=np.float32)
    for i in prange(N):
        prev = np.nan
        for t in range(T):
            z = exit_[t, i]
            if z == z and z > 0:
                prev = np.nan
            else:
                g = trig[t, i]
                if g == g and g > 0:
                    prev = alpha[t, i]
            out[t, i] = prev
    return out


@njit(cache=True)
def hump(x, h):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in range(T):
        s = 0.0
        n = 0
        for i in range(N):
            v = x[t, i]
            if v == v:
                s += abs(v)
                n += 1
        thr = h * (s / n) if n > 0 else 0.0
        for i in range(N):
            v = x[t, i]
            if t == 0:
                out[t, i] = v
                continue
            p = out[t - 1, i]
            if v != v:
                out[t, i] = np.nan
            elif p == p and abs(v - p) < thr:
                out[t, i] = p
            else:
                out[t, i] = v
    return out


@njit(parallel=True, cache=True)
def kth_element(x, d, k):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for i in prange(N):
        for t in range(T):
            found = 0
            lo = max(0, t - d + 1)
            for u in range(t, lo - 1, -1):
                w = x[u, i]
                if w == w:
                    found += 1
                    if found == k:
                        out[t, i] = w
                        break
    return out


# =========================================================================== cross-sectional (row) kernels

@njit(parallel=True, cache=True)
def cs_rank(x):
    """Average-tie rank in [0, 1] per row; NaN stays NaN; a single valid value gets 0.5."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        row = x[t]
        idx = np.argsort(row)  # NaNs sort last
        n = 0
        for i in range(N):
            if row[i] == row[i]:
                n += 1
        if n == 0:
            continue
        if n == 1:
            for i in range(N):
                if row[i] == row[i]:
                    out[t, i] = 0.5
            continue
        k = 0
        while k < n:
            j = k
            v = row[idx[k]]
            while j + 1 < n and row[idx[j + 1]] == v:
                j += 1
            r = 0.5 * (k + j) / (n - 1)
            for q in range(k, j + 1):
                out[t, idx[q]] = r
            k = j + 1
    return out


@njit(parallel=True, cache=True)
def cs_moments(x):
    """Per-row (mean, std, count, sum_abs, min, max) over valid values."""
    T, N = x.shape
    res = np.empty((T, 6))
    for t in prange(T):
        s = 0.0
        s2 = 0.0
        sa = 0.0
        n = 0
        mn = np.inf
        mx = -np.inf
        for i in range(N):
            v = x[t, i]
            if v == v:
                s += v
                s2 += v * v
                sa += abs(v)
                n += 1
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        if n > 0:
            mean = s / n
            var = s2 / n - mean * mean
            res[t, 0] = mean
            res[t, 1] = math.sqrt(var) if var > 0 else 0.0
        else:
            res[t, 0] = np.nan
            res[t, 1] = np.nan
        res[t, 2] = n
        res[t, 3] = sa
        res[t, 4] = mn if n > 0 else np.nan
        res[t, 5] = mx if n > 0 else np.nan
    return res


@njit(parallel=True, cache=True)
def cs_regress(y, x, mode):
    """Per-row OLS y = a + b x over valid pairs. mode 0 residual, 1 fitted, 2 vector_neut, 3 vector_proj."""
    T, N = y.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        n = 0
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        for i in range(N):
            a = x[t, i]
            b = y[t, i]
            if a == a and b == b:
                n += 1
                sx += a
                sy += b
                sxx += a * a
                sxy += a * b
        if n < 2:
            continue
        if mode >= 2:
            if sxx <= 0:
                continue
            beta = sxy / sxx
            alpha = 0.0
        else:
            mx = sx / n
            my = sy / n
            vxx = sxx / n - mx * mx
            if vxx <= 1e-20:
                continue
            beta = (sxy / n - mx * my) / vxx
            alpha = my - beta * mx
        for i in range(N):
            a = x[t, i]
            b = y[t, i]
            if a == a and b == b:
                fit = alpha + beta * a
                out[t, i] = (b - fit) if (mode == 0 or mode == 2) else fit
    return out


# =========================================================================== group kernels
# gop: 0 mean, 1 sum, 2 count, 3 max, 4 min, 5 std, 6 neutralize, 7 zscore, 8 scale, 9 normalize(abs-sum)

@njit(parallel=True, cache=True)
def group_stat(x, w, codes, G, gop, scale):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        s = np.zeros(G)
        s2 = np.zeros(G)
        sw = np.zeros(G)
        cnt = np.zeros(G, np.int64)
        mx = np.full(G, -np.inf)
        mn = np.full(G, np.inf)
        sa = np.zeros(G)
        for i in range(N):
            g = codes[t, i]
            v = x[t, i]
            if g < 0 or v != v:
                continue
            wt = w[t, i]
            if wt != wt:
                continue
            s[g] += wt * v
            sw[g] += wt
            s2[g] += v * v
            cnt[g] += 1
            sa[g] += abs(v)
            if v > mx[g]:
                mx[g] = v
            if v < mn[g]:
                mn[g] = v
        for i in range(N):
            g = codes[t, i]
            v = x[t, i]
            if g < 0 or cnt[g] == 0:
                continue
            n = cnt[g]
            if gop == 2:
                out[t, i] = n
                continue
            if gop == 3:
                out[t, i] = mx[g]
                continue
            if gop == 4:
                out[t, i] = mn[g]
                continue
            if gop == 1:
                out[t, i] = s[g]
                continue
            mean = s[g] / sw[g] if sw[g] != 0 else np.nan
            if gop == 0:
                out[t, i] = mean
                continue
            # unweighted moments for the rest
            m_u = 0.0
            # recompute unweighted mean from s (w==1 in these calls)
            m_u = mean
            var = s2[g] / n - m_u * m_u
            sd = math.sqrt(var) if var > 0 else 0.0
            if gop == 5:
                out[t, i] = sd
            elif v == v:
                if gop == 6:
                    out[t, i] = v - m_u
                elif gop == 7:
                    if sd > 0:
                        out[t, i] = (v - m_u) / sd
                elif gop == 8:
                    rng = mx[g] - mn[g]
                    if rng > 0:
                        out[t, i] = (v - mn[g]) / rng
                elif gop == 9:
                    if sa[g] > 0:
                        out[t, i] = scale * v / sa[g]
    return out


@njit(parallel=True, cache=True)
def group_rank(x, codes, G):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        row = x[t]
        # key: group code major, value minor -> sort valid entries
        m = 0
        for i in range(N):
            if codes[t, i] >= 0 and row[i] == row[i]:
                m += 1
        if m == 0:
            continue
        ids = np.empty(m, np.int64)
        k = 0
        for i in range(N):
            if codes[t, i] >= 0 and row[i] == row[i]:
                ids[k] = i
                k += 1
        vals = np.empty(m)
        for q in range(m):
            vals[q] = row[ids[q]]
        o1 = np.argsort(vals, kind="mergesort")
        gk = np.empty(m, np.int64)
        for q in range(m):
            gk[q] = codes[t, ids[o1[q]]]
        o2 = np.argsort(gk, kind="mergesort")
        order = np.empty(m, np.int64)
        for q in range(m):
            order[q] = ids[o1[o2[q]]]
        # walk groups
        a = 0
        while a < m:
            g = codes[t, order[a]]
            b = a
            while b + 1 < m and codes[t, order[b + 1]] == g:
                b += 1
            n = b - a + 1
            if n == 1:
                out[t, order[a]] = 0.5
            else:
                k = a
                while k <= b:
                    j = k
                    v = row[order[k]]
                    while j + 1 <= b and row[order[j + 1]] == v:
                        j += 1
                    r = 0.5 * ((k - a) + (j - a)) / (n - 1)
                    for q in range(k, j + 1):
                        out[t, order[q]] = r
                    k = j + 1
            a = b + 1
    return out


@njit(parallel=True, cache=True)
def group_percentile(x, codes, G, q):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        cnt = np.zeros(G, np.int64)
        for i in range(N):
            g = codes[t, i]
            if g >= 0 and x[t, i] == x[t, i]:
                cnt[g] += 1
        start = np.zeros(G + 1, np.int64)
        for g in range(G):
            start[g + 1] = start[g] + cnt[g]
        buf = np.empty(start[G])
        fill = np.zeros(G, np.int64)
        for i in range(N):
            g = codes[t, i]
            if g >= 0 and x[t, i] == x[t, i]:
                buf[start[g] + fill[g]] = x[t, i]
                fill[g] += 1
        res = np.full(G, np.nan)
        for g in range(G):
            n = cnt[g]
            if n == 0:
                continue
            seg = np.sort(buf[start[g]:start[g] + n])
            pos = q * (n - 1)
            k0 = int(math.floor(pos))
            k1 = min(k0 + 1, n - 1)
            fr = pos - k0
            res[g] = seg[k0] * (1 - fr) + seg[k1] * fr
        for i in range(N):
            g = codes[t, i]
            if g >= 0:
                out[t, i] = res[g]
    return out


@njit(parallel=True, cache=True)
def group_vec(x, y, codes, G, proj):
    """Within-group projection of x on y (no intercept); proj=False returns the residual."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        sxy = np.zeros(G)
        syy = np.zeros(G)
        for i in range(N):
            g = codes[t, i]
            a = x[t, i]
            b = y[t, i]
            if g >= 0 and a == a and b == b:
                sxy[g] += a * b
                syy[g] += b * b
        for i in range(N):
            g = codes[t, i]
            a = x[t, i]
            b = y[t, i]
            if g >= 0 and a == a and b == b and syy[g] > 0:
                p = sxy[g] / syy[g] * b
                out[t, i] = p if proj else a - p
    return out


@njit(parallel=True, cache=True)
def group_window_mean(x, codes, G, d):
    """Mean of the group's valid values over the last d days (used by group_backfill)."""
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan
    for t in prange(T):
        s = np.zeros(G)
        c = np.zeros(G)
        lo = max(0, t - d + 1)
        for u in range(lo, t + 1):
            for i in range(N):
                g = codes[t, i]
                v = x[u, i]
                if g >= 0 and v == v:
                    s[g] += v
                    c[g] += 1
        for i in range(N):
            g = codes[t, i]
            if g >= 0 and c[g] > 0:
                out[t, i] = s[g] / c[g]
    return out


# =========================================================================== simulator helpers

@njit(parallel=True, cache=True)
def neutralize_rows(a, codes, G):
    """Subtract the group mean per row (codes<0 -> NaN). Returns float64."""
    T, N = a.shape
    out = np.empty((T, N))
    for t in prange(T):
        s = np.zeros(G)
        c = np.zeros(G)
        for i in range(N):
            g = codes[t, i]
            v = a[t, i]
            if g >= 0 and v == v:
                s[g] += v
                c[g] += 1
        for i in range(N):
            g = codes[t, i]
            v = a[t, i]
            if g >= 0 and v == v and c[g] > 0:
                out[t, i] = v - s[g] / c[g]
            else:
                out[t, i] = np.nan
    return out


@njit(cache=True)
def _norm_trunc_row(vals, ok, trunc, iters, out_row):
    """Row helper: scale valid values to sum|w| = 1 and cap at trunc (water-filling); others -> 0."""
    N = vals.shape[0]
    s = 0.0
    nvalid = 0
    for i in range(N):
        if ok[i] and vals[i] != 0.0:
            s += abs(vals[i])
            nvalid += 1
    for i in range(N):
        out_row[i] = 0.0
    if s <= 0:
        return
    for i in range(N):
        if ok[i]:
            out_row[i] = vals[i] / s
    if trunc <= 0.0 or trunc >= 1.0 or nvalid * trunc <= 1.0:
        return
    clipped = np.zeros(N, np.bool_)
    nclip = 0
    for _ in range(max(iters, 2 * N)):
        budget = 1.0 - trunc * nclip
        free = 0.0
        for i in range(N):
            if ok[i] and not clipped[i]:
                free += abs(vals[i])
        if free <= 0 or budget <= 0:
            break
        sc = budget / free
        new = False
        for i in range(N):
            if not ok[i] or clipped[i]:
                continue
            w = vals[i] * sc
            if w > trunc or w < -trunc:
                clipped[i] = True
                nclip += 1
                new = True
            out_row[i] = w
        if not new:
            break
    for i in range(N):
        if clipped[i]:
            out_row[i] = trunc if vals[i] > 0 else -trunc


@njit(parallel=True, cache=True)
def weights_kernel(A, mask, codes, G, neut, nan_zero, trunc, iters):
    """Fused: universe mask -> (NaN handling) -> group demean -> scale -> truncate. Returns float64 weights."""
    T, N = A.shape
    out = np.zeros((T, N))
    for t in prange(T):
        vals = np.zeros(N)
        ok = np.zeros(N, np.bool_)
        for i in range(N):
            if mask[t, i]:
                v = A[t, i]
                if v == v:
                    vals[i] = v
                    ok[i] = True
                elif nan_zero:
                    ok[i] = True
        if neut:
            s = np.zeros(G)
            c = np.zeros(G)
            for i in range(N):
                if ok[i]:
                    g = codes[t, i]
                    if g >= 0:
                        s[g] += vals[i]
                        c[g] += 1.0
                    else:
                        ok[i] = False
            for i in range(N):
                if ok[i]:
                    vals[i] -= s[codes[t, i]] / c[codes[t, i]]
        _norm_trunc_row(vals, ok, trunc, iters, out[t])
    return out


@njit(parallel=True, cache=True)
def pnl_stats(W, ret, lag, booksize, a, b):
    """Fused PnL pass: daily PnL (total/long), turnover, long/short counts, max weight, per-stock IS PnL."""
    T, N = W.shape
    pnl = np.zeros(T)
    pl = np.zeros(T)
    tvr = np.zeros(T)
    lc = np.zeros(T)
    scnt = np.zeros(T)
    mw = np.zeros(T)
    for t in prange(T):
        s = 0.0
        sl = 0.0
        d = 0.0
        nl = 0
        ns = 0
        m = 0.0
        for i in range(N):
            w = W[t, i]
            if w > 0:
                nl += 1
            elif w < 0:
                ns += 1
            aw = abs(w)
            if aw > m:
                m = aw
            if t >= 1:
                d += abs(w - W[t - 1, i])
            if t >= lag:
                wp = W[t - lag, i]
                if wp != 0.0:
                    r = ret[t, i]
                    if r == r:
                        x = wp * r
                        s += x
                        if wp > 0:
                            sl += x
        pnl[t] = s * booksize
        pl[t] = sl * booksize
        tvr[t] = d
        lc[t] = nl
        scnt[t] = ns
        mw[t] = m
    contrib = np.zeros(N)
    lo = max(a, lag)
    for i in prange(N):
        acc = 0.0
        for t in range(lo, b):
            wp = W[t - lag, i]
            if wp != 0.0:
                r = ret[t, i]
                if r == r:
                    acc += wp * r
        contrib[i] = acc * booksize
    return pnl, pl, tvr, lc, scnt, mw, contrib


@njit(parallel=True, cache=True)
def normalize_truncate(a, trunc, iters):
    """Scale rows to sum|w| = 1 and cap |w| at ``trunc`` exactly (water-filling).

    Capped names are pinned at +/-trunc and the remaining names are rescaled to use the rest of the
    budget, repeating until no free name exceeds the cap. If fewer than 1/trunc names are
    available the cap is infeasible and the proportional weights are kept (the weight-concentration
    check will flag it). ``iters`` bounds the number of passes.
    """
    T, N = a.shape
    out = np.zeros((T, N))
    for t in prange(T):
        s = 0.0
        nvalid = 0
        for i in range(N):
            v = a[t, i]
            if v == v and v != 0.0:
                s += abs(v)
                nvalid += 1
        if s <= 0:
            continue
        for i in range(N):
            v = a[t, i]
            out[t, i] = v / s if v == v else 0.0
        if trunc <= 0.0 or trunc >= 1.0 or nvalid * trunc <= 1.0:
            continue
        clipped = np.zeros(N, np.bool_)
        nclip = 0
        for _ in range(max(iters, 2 * N)):
            budget = 1.0 - trunc * nclip
            free = 0.0
            for i in range(N):
                v = a[t, i]
                if not clipped[i] and v == v:
                    free += abs(v)
            if free <= 0 or budget <= 0:
                break
            sc = budget / free
            new = False
            for i in range(N):
                v = a[t, i]
                if clipped[i] or v != v:
                    continue
                w = v * sc
                if w > trunc or w < -trunc:
                    clipped[i] = True
                    nclip += 1
                    new = True
                out[t, i] = w
            if not new:
                break
        for i in range(N):
            if clipped[i]:
                out[t, i] = trunc if a[t, i] > 0 else -trunc
    return out
