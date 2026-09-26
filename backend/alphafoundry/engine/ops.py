"""Local implementations of BRAIN operators.

Every implementation has the signature ``fn(ctx, args, p) -> ndarray | Groups`` where ``args`` are the
evaluated data arguments (float32 (T, N) arrays, Python floats for constants, or ``Groups``) and
``p`` is the dict of scalar parameters (all explicit). Implementations never mutate their inputs
(results are cached and shared).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from . import kernels as K

REGISTRY: dict[str, Callable[..., Any]] = {}


class Groups:
    __slots__ = ("codes", "G")

    def __init__(self, codes: np.ndarray, G: int):
        self.codes = codes  # int32 (T, N), -1 = no group
        self.G = int(G)

    @property
    def nbytes(self) -> int:
        return self.codes.nbytes


def register(*names: str):
    def deco(fn):
        for n in names:
            REGISTRY[n] = fn
        return fn

    return deco


def local_operator_names() -> set[str]:
    return set(REGISTRY)


# --------------------------------------------------------------------------- helpers


def f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def mat(ctx, a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a if a.dtype == np.float32 and a.flags.c_contiguous else f32(a)
    return np.full((ctx.T, ctx.N), np.float32(a), dtype=np.float32)


def clean(a: Any, ctx=None) -> np.ndarray:
    """Force a float32 (T, N) array and turn +/-inf into NaN (in place on the fresh result)."""
    if not isinstance(a, np.ndarray) or a.ndim == 0:
        return mat(ctx, float(a))
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    bad = ~np.isfinite(a)
    if bad.any():
        a = a.copy() if not a.flags.writeable else a
        a[bad] = np.nan
    return a


def minp(d: int) -> int:
    return max(1, (int(d) + 1) // 2)


def shift(x: np.ndarray, d: int) -> np.ndarray:
    out = np.empty_like(x)
    if d <= 0:
        out[:] = x
        return out
    out[:d] = np.nan
    out[d:] = x[:-d]
    return out


def nanmask(*arrs) -> np.ndarray | bool:
    m = False
    for a in arrs:
        if isinstance(a, np.ndarray):
            m = m | np.isnan(a)
    return m


def _errstate():
    return np.errstate(divide="ignore", invalid="ignore", over="ignore")


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.2e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    lo, hi = 0.02425, 1 - 0.02425
    with _errstate():
        m = (p > 0) & (p < lo)
        q = np.sqrt(-2 * np.log(p[m]))
        out[m] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        m = (p >= lo) & (p <= hi)
        q = p[m] - 0.5
        r = q * q
        out[m] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        m = (p > hi) & (p < 1)
        q = np.sqrt(-2 * np.log(1 - p[m]))
        out[m] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return out


def dist_ppf(p: np.ndarray, driver: str) -> np.ndarray:
    if driver == "uniform":
        return 2.0 * p - 1.0
    if driver == "cauchy":
        with _errstate():
            return np.tan(np.pi * (p - 0.5))
    return norm_ppf(p)


def row_counts(x: np.ndarray) -> np.ndarray:
    return (~np.isnan(x)).sum(axis=1, keepdims=True).astype(np.float64)


# =========================================================================== arithmetic


@register("add")
def _add(ctx, a, p):
    with _errstate():
        if p.get("filter"):
            vals = [np.nan_to_num(v, nan=0.0) if isinstance(v, np.ndarray) else v for v in a]
        else:
            vals = a
        r = vals[0]
        for v in vals[1:]:
            r = r + v
    return clean(r, ctx)


@register("subtract")
def _sub(ctx, a, p):
    x, y = a
    if p.get("filter"):
        x = np.nan_to_num(x, nan=0.0) if isinstance(x, np.ndarray) else x
        y = np.nan_to_num(y, nan=0.0) if isinstance(y, np.ndarray) else y
    with _errstate():
        return clean(x - y, ctx)


@register("multiply")
def _mul(ctx, a, p):
    with _errstate():
        if p.get("filter"):
            vals = [np.nan_to_num(v, nan=1.0) if isinstance(v, np.ndarray) else v for v in a]
        else:
            vals = a
        r = vals[0]
        for v in vals[1:]:
            r = r * v
    return clean(r, ctx)


@register("divide")
def _div(ctx, a, p):
    x, y = a
    with _errstate():
        return clean(np.divide(x, y), ctx)


@register("reverse")
def _rev(ctx, a, p):
    return clean(-a[0], ctx)


@register("inverse")
def _inv(ctx, a, p):
    with _errstate():
        return clean(1.0 / a[0], ctx)


def _unary(fn):
    def impl(ctx, a, p):
        with _errstate():
            return clean(fn(mat(ctx, a[0])), ctx)
    return impl


for _n, _f in {
    "abs": np.abs, "sign": np.sign, "exp": np.exp, "floor": np.floor, "ceiling": np.ceil, "round": np.round,
    "arc_tan": np.arctan, "tanh": np.tanh,
    "log": lambda x: np.where(x > 0, np.log(np.where(x > 0, x, 1)), np.nan),
    "sqrt": lambda x: np.where(x >= 0, np.sqrt(np.abs(x)), np.nan),
    "s_log_1p": lambda x: np.sign(x) * np.log1p(np.abs(x)),
    "arc_cos": lambda x: np.where(np.abs(x) <= 1, np.arccos(np.clip(x, -1, 1)), np.nan),
    "arc_sin": lambda x: np.where(np.abs(x) <= 1, np.arcsin(np.clip(x, -1, 1)), np.nan),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "fraction": lambda x: np.sign(x) * (np.abs(x) - np.floor(np.abs(x))),
    "purify": lambda x: x.copy(),
}.items():
    REGISTRY[_n] = _unary(_f)


@register("round_down")
def _round_down(ctx, a, p):
    f = float(p.get("f", 1.0)) or 1.0
    with _errstate():
        return clean(np.floor(mat(ctx, a[0]) / f) * f, ctx)


@register("power")
def _power(ctx, a, p):
    with _errstate():
        return clean(np.power(mat(ctx, a[0]).astype(np.float64), a[1]), ctx)


@register("signed_power")
def _spower(ctx, a, p):
    x = mat(ctx, a[0])
    with _errstate():
        return clean(np.sign(x) * np.power(np.abs(x).astype(np.float64), a[1]), ctx)


@register("max")
def _max(ctx, a, p):
    r = a[0]
    for v in a[1:]:
        r = np.maximum(r, v)
    return clean(r, ctx)


@register("min")
def _min(ctx, a, p):
    r = a[0]
    for v in a[1:]:
        r = np.minimum(r, v)
    return clean(r, ctx)


@register("to_nan")
def _to_nan(ctx, a, p):
    x = mat(ctx, a[0]).copy()
    v = float(p.get("value", 0.0))
    if p.get("reverse"):
        x[x != v] = np.nan
    else:
        x[x == v] = np.nan
    return x


@register("nan_mask")
def _nan_mask(ctx, a, p):
    x = mat(ctx, a[0]).copy()
    y = mat(ctx, a[1])
    x[y < 0] = np.nan
    return x


@register("nan_out")
def _nan_out(ctx, a, p):
    x = mat(ctx, a[0]).copy()
    lo, hi = float(p.get("lower", 0.0)), float(p.get("upper", 0.0))
    x[(x < lo) | (x > hi)] = np.nan
    return x


@register("pasteurize")
def _pasteurize(ctx, a, p):
    x = clean(mat(ctx, a[0]).copy(), ctx)
    if ctx.univ_mask is not None:
        x[~ctx.univ_mask] = np.nan
    return x


@register("log_diff")
def _log_diff(ctx, a, p):
    x = mat(ctx, a[0])
    with _errstate():
        lx = np.where(x > 0, np.log(np.where(x > 0, x, 1)), np.nan).astype(np.float32)
    return clean(lx - shift(lx, 1), ctx)


# =========================================================================== logical


def _cmp(fn):
    def impl(ctx, a, p):
        x, y = a
        with _errstate():
            r = fn(x, y).astype(np.float32)
        r = mat(ctx, r) if np.ndim(r) == 0 else r
        m = nanmask(x, y)
        if m is not False:
            r = r.copy() if not r.flags.writeable else r
            r[m] = np.nan
        return r
    return impl


for _n, _f in {"less": np.less, "less_equal": np.less_equal, "greater": np.greater,
               "greater_equal": np.greater_equal, "equal": np.equal, "not_equal": np.not_equal}.items():
    REGISTRY[_n] = _cmp(_f)


@register("and")
def _and(ctx, a, p):
    x, y = a
    r = ((np.asarray(x) > 0) & (np.asarray(y) > 0)).astype(np.float32)
    r = mat(ctx, float(r)) if r.ndim == 0 else r
    m = nanmask(x, y)
    if m is not False:
        r[m] = np.nan
    return r


@register("or")
def _or(ctx, a, p):
    x, y = a
    r = ((np.asarray(x) > 0) | (np.asarray(y) > 0)).astype(np.float32)
    r = mat(ctx, float(r)) if r.ndim == 0 else r
    m = nanmask(x, y)
    if m is not False:
        r[m] = np.nan
    return r


@register("not")
def _not(ctx, a, p):
    x = mat(ctx, a[0])
    r = (x <= 0).astype(np.float32)
    r[np.isnan(x)] = np.nan
    return r


@register("is_nan")
def _is_nan(ctx, a, p):
    return np.isnan(mat(ctx, a[0])).astype(np.float32)


@register("if_else")
def _if_else(ctx, a, p):
    c, x, y = a
    c = mat(ctx, c)
    r = np.where(c > 0, x, y).astype(np.float32)
    r = np.ascontiguousarray(np.broadcast_to(r, (ctx.T, ctx.N))).copy() if r.shape != (ctx.T, ctx.N) else r
    r[np.isnan(c)] = np.nan
    return r


# =========================================================================== time series


def _moments(mode: int):
    def impl(ctx, a, p):
        d = int(p["d"])
        return K.roll_moments(mat(ctx, a[0]), d, minp(d), mode)
    return impl


for _n, _m in {"ts_mean": 0, "ts_sum": 1, "ts_std_dev": 2, "ts_zscore": 3, "ts_ir": 4, "ts_av_diff": 5,
               "ts_count_nans": 6, "ts_skewness": 7, "ts_kurtosis": 8}.items():
    REGISTRY[_n] = _moments(_m)


@register("ts_rank")
def _ts_rank(ctx, a, p):
    d = int(p["d"])
    return K.roll_rank(mat(ctx, a[0]), d, max(2, minp(d)), float(p.get("constant", 0.0)))


def _minmax(mode: int):
    def impl(ctx, a, p):
        d = int(p["d"])
        return K.roll_minmax(mat(ctx, a[0]), d, 1, mode)
    return impl


for _n, _m in {"ts_min": 0, "ts_max": 1, "ts_arg_min": 2, "ts_arg_max": 3}.items():
    REGISTRY[_n] = _minmax(_m)


@register("ts_median")
def _ts_median(ctx, a, p):
    d = int(p["d"])
    return K.roll_percentile(mat(ctx, a[0]), d, minp(d), 0.5)


@register("ts_percentage")
def _ts_pct(ctx, a, p):
    d = int(p["d"])
    q = min(1.0, max(0.0, float(p.get("percentage", 0.5))))
    return K.roll_percentile(mat(ctx, a[0]), d, minp(d), q)


@register("ts_product")
def _ts_product(ctx, a, p):
    d = int(p["d"])
    return clean(K.roll_scan(mat(ctx, a[0]), d, minp(d), 0, 1), ctx)


@register("ts_moment")
def _ts_moment(ctx, a, p):
    d = int(p["d"])
    return clean(K.roll_scan(mat(ctx, a[0]), d, minp(d), 1, int(p.get("k", 0))), ctx)


@register("ts_delay")
def _ts_delay(ctx, a, p):
    return shift(mat(ctx, a[0]), int(p["d"]))


@register("ts_delta")
def _ts_delta(ctx, a, p):
    x = mat(ctx, a[0])
    return x - shift(x, int(p["d"]))


@register("ts_returns")
def _ts_returns(ctx, a, p):
    x = mat(ctx, a[0])
    prev = shift(x, int(p["d"]))
    with _errstate():
        if int(p.get("mode", 1)) == 2:
            r = np.log(x / prev)
        else:
            r = (x - prev) / prev
    return clean(r, ctx)


@register("ts_scale")
def _ts_scale(ctx, a, p):
    x = mat(ctx, a[0])
    d = int(p["d"])
    mn = K.roll_minmax(x, d, 1, 0)
    mx = K.roll_minmax(x, d, 1, 1)
    with _errstate():
        return clean((x - mn) / (mx - mn) + float(p.get("constant", 0.0)), ctx)


@register("ts_max_diff")
def _ts_max_diff(ctx, a, p):
    x = mat(ctx, a[0])
    return x - K.roll_minmax(x, int(p["d"]), 1, 1)


@register("ts_min_diff")
def _ts_min_diff(ctx, a, p):
    x = mat(ctx, a[0])
    return x - K.roll_minmax(x, int(p["d"]), 1, 0)


@register("ts_min_max_diff")
def _ts_mm_diff(ctx, a, p):
    x = mat(ctx, a[0])
    d = int(p["d"])
    f = float(p.get("f", 0.5))
    return x - np.float32(f) * (K.roll_minmax(x, d, 1, 0) + K.roll_minmax(x, d, 1, 1))


@register("ts_min_max_cps")
def _ts_mm_cps(ctx, a, p):
    x = mat(ctx, a[0])
    d = int(p["d"])
    f = float(p.get("f", 2.0))
    return (K.roll_minmax(x, d, 1, 0) + K.roll_minmax(x, d, 1, 1)) - np.float32(f) * x


@register("ts_quantile")
def _ts_quantile(ctx, a, p):
    d = int(p["d"])
    r = K.roll_rank(mat(ctx, a[0]), d, max(2, minp(d)), 0.0).astype(np.float64)
    n = float(d)
    shifted = (r * (n - 1) + 0.5) / n
    return clean(dist_ppf(shifted, str(p.get("driver", "gaussian"))), ctx)


@register("ts_decay_linear")
def _ts_decay_linear(ctx, a, p):
    d = int(p["d"])
    x = mat(ctx, a[0])
    if d <= 1:
        return x
    if p.get("dense"):
        return K.decay_linear(x, d, True)
    return K.decay_linear_fast(x, d)


@register("ts_decay_exp_window")
def _ts_decay_exp(ctx, a, p):
    d = int(p["d"])
    f = float(p.get("factor", 0.5))
    return K.decay_exp(mat(ctx, a[0]), d, f)


@register("ts_corr")
def _ts_corr(ctx, a, p):
    d = int(p["d"])
    return K.roll_pair(mat(ctx, a[1]), mat(ctx, a[0]), d, max(2, minp(d)), 0, 0)


@register("ts_covariance")
def _ts_cov(ctx, a, p):
    d = int(p["d"])
    return K.roll_pair(mat(ctx, a[0]), mat(ctx, a[1]), d, max(2, minp(d)), 1, 0)


@register("ts_regression")
def _ts_reg(ctx, a, p):
    d = int(p["d"])
    lag = int(p.get("lag", 0))
    x = mat(ctx, a[1])
    if lag > 0:
        x = shift(x, lag)
    return K.roll_pair(mat(ctx, a[0]), x, d, max(3, minp(d)), 2, int(p.get("rettype", 0)))


@register("ts_vector_neut")
def _ts_vneut(ctx, a, p):
    d = int(p["d"])
    return K.roll_pair(mat(ctx, a[0]), mat(ctx, a[1]), d, max(3, minp(d)), 2, 0)


@register("ts_vector_proj")
def _ts_vproj(ctx, a, p):
    d = int(p["d"])
    return K.roll_pair(mat(ctx, a[0]), mat(ctx, a[1]), d, max(3, minp(d)), 2, 3)


@register("ts_backfill")
def _ts_backfill(ctx, a, p):
    lb = int(p.get("lookback", 252))
    k = int(p.get("k", 1))
    x = mat(ctx, a[0])
    if k <= 1:
        return K.ffill_limit(x, lb)
    return K.backfill(x, lb, k)


@register("kth_element")
def _kth(ctx, a, p):
    return K.kth_element(mat(ctx, a[0]), int(p["d"]), max(1, int(p.get("k", 1))))


@register("ts_step")
def _ts_step(ctx, a, p):
    steps = (np.arange(ctx.T, dtype=np.float32) + ctx.r0 + 1) * np.float32(p.get("n", 1))
    return np.ascontiguousarray(np.broadcast_to(steps[:, None], (ctx.T, ctx.N)))


@register("ts_weighted_decay")
def _ts_wdecay(ctx, a, p):
    x = mat(ctx, a[0])
    k = np.float32(p.get("k", 0.5))
    prev = shift(x, 1)
    r = k * x + (np.float32(1.0) - k) * prev
    r = np.where(np.isnan(prev), x, r).astype(np.float32)
    return r


@register("days_from_last_change")
def _dflc(ctx, a, p):
    return K.days_from_last_change(mat(ctx, a[0]))


@register("last_diff_value")
def _ldv(ctx, a, p):
    return K.last_diff_value(mat(ctx, a[0]), int(p["d"]))


@register("hump")
def _hump(ctx, a, p):
    return K.hump(mat(ctx, a[0]), float(p.get("hump", 0.01)))


@register("jump_decay")
def _jump_decay(ctx, a, p):
    x = mat(ctx, a[0])
    d = int(p["d"])
    sens = np.float32(p.get("sensitivity", 0.5))
    force = np.float32(p.get("force", 0.1))
    prev = shift(x, 1)
    sd = K.roll_moments(x, d, minp(d), 2)
    delta = x - prev
    jump = np.abs(delta) > sens * sd
    r = np.where(jump, prev + delta * force, x).astype(np.float32)
    return r


def _turnover_of(x: np.ndarray) -> float:
    import warnings

    with warnings.catch_warnings(), _errstate():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows are expected (warm-up)
        a = x - np.nanmean(x, axis=1, keepdims=True)
        s = np.nansum(np.abs(a), axis=1, keepdims=True)
        w = np.nan_to_num(a / s)
        return float(np.nanmean(np.abs(np.diff(w, axis=0)).sum(axis=1)))


@register("ts_target_tvr_decay")
def _ts_target_tvr(ctx, a, p):
    """Pick the exponential-smoothing strength whose output turnover is closest to target_tvr."""
    x = mat(ctx, a[0])
    lo = max(1e-3, float(p.get("lambda_min", 0.0)))
    hi = min(1.0, max(lo, float(p.get("lambda_max", 1.0))))
    target = float(p.get("target_tvr", 0.1))
    step = max(1, x.shape[0] // 400)  # estimate turnover on a subsample of days for speed
    best = K.ema(x, hi)
    if _turnover_of(best[::step]) <= target:
        return best
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        y = K.ema(x, mid)
        if _turnover_of(y[::step]) > target:
            hi = mid
        else:
            lo = mid
            best = y
    return best


# =========================================================================== cross-sectional


@register("rank")
def _rank(ctx, a, p):
    return K.cs_rank(mat(ctx, a[0]))


@register("zscore")
def _zscore(ctx, a, p):
    x = mat(ctx, a[0])
    mom = K.cs_moments(x)
    with _errstate():
        r = (x - mom[:, 0:1]) / mom[:, 1:2]
    return clean(r, ctx)


@register("scale")
def _scale(ctx, a, p):
    x = mat(ctx, a[0]).astype(np.float64)
    sc = float(p.get("scale", 1.0))
    ls, ss = float(p.get("longscale", 1.0)), float(p.get("shortscale", 1.0))
    with _errstate():
        if ls == 1.0 and ss == 1.0:
            s = np.nansum(np.abs(x), axis=1, keepdims=True)
            r = x / s * sc
        else:
            pos = np.where(x > 0, x, 0.0)
            neg = np.where(x < 0, x, 0.0)
            ps = np.nansum(pos, axis=1, keepdims=True)
            ns = np.nansum(-neg, axis=1, keepdims=True)
            r = np.where(x > 0, x / ps * ls, np.where(x < 0, x / ns * ss, x))
    return clean(r, ctx)


@register("normalize")
def _normalize(ctx, a, p):
    x = mat(ctx, a[0])
    mom = K.cs_moments(x)
    r = x - mom[:, 0:1]
    with _errstate():
        if p.get("useStd"):
            r = r / mom[:, 1:2]
        lim = float(p.get("limit", 0.0))
        if lim > 0:
            r = np.clip(r, -lim, lim)
    return clean(r, ctx)


@register("quantile")
def _quantile(ctx, a, p):
    x = mat(ctx, a[0])
    r = K.cs_rank(x).astype(np.float64)
    n = row_counts(x)
    with _errstate():
        shifted = (r * (n - 1) + 0.5) / n
    return clean(dist_ppf(shifted, str(p.get("driver", "gaussian"))) * float(p.get("sigma", 1.0)), ctx)


@register("winsorize")
def _winsorize(ctx, a, p):
    x = mat(ctx, a[0])
    mom = K.cs_moments(x)
    k = float(p.get("std", 4.0))
    lo = (mom[:, 0:1] - k * mom[:, 1:2]).astype(np.float32)
    hi = (mom[:, 0:1] + k * mom[:, 1:2]).astype(np.float32)
    return np.clip(x, lo, hi).astype(np.float32)


@register("truncate")
def _truncate(ctx, a, p):
    x = mat(ctx, a[0])
    s = np.nansum(np.abs(x), axis=1, keepdims=True) * float(p.get("maxPercent", 0.01))
    return np.clip(x, -s, s).astype(np.float32)


@register("scale_down")
def _scale_down(ctx, a, p):
    x = mat(ctx, a[0])
    mom = K.cs_moments(x)
    with _errstate():
        r = (x - mom[:, 4:5]) / (mom[:, 5:6] - mom[:, 4:5]) - float(p.get("constant", 0.0))
    return clean(r, ctx)


@register("vector_neut")
def _vneut(ctx, a, p):
    return K.cs_regress(mat(ctx, a[0]), mat(ctx, a[1]), 2)


@register("vector_proj")
def _vproj(ctx, a, p):
    return K.cs_regress(mat(ctx, a[0]), mat(ctx, a[1]), 3)


@register("regression_neut")
def _rneut(ctx, a, p):
    return K.cs_regress(mat(ctx, a[0]), mat(ctx, a[1]), 0)


@register("regression_proj")
def _rproj(ctx, a, p):
    return K.cs_regress(mat(ctx, a[0]), mat(ctx, a[1]), 1)


# =========================================================================== groups


def _codes(ctx, g: Any) -> Groups:
    if isinstance(g, Groups):
        return g
    raise TypeError("expected a group")


def _gstat(gop: int):
    def impl(ctx, a, p):
        x = mat(ctx, a[0])
        g = _codes(ctx, a[-1])
        return K.group_stat(x, ctx.ones(), g.codes, g.G, gop, float(p.get("scale", 1.0)))
    return impl


for _n, _m in {"group_sum": 1, "group_count": 2, "group_max": 3, "group_min": 4, "group_std_dev": 5,
               "group_neutralize": 6, "group_zscore": 7, "group_scale": 8, "group_normalize": 9}.items():
    REGISTRY[_n] = _gstat(_m)


@register("group_mean")
def _gmean(ctx, a, p):
    x = mat(ctx, a[0])
    w = mat(ctx, a[1])
    g = _codes(ctx, a[2])
    return K.group_stat(x, w, g.codes, g.G, 0, 1.0)


@register("group_rank")
def _grank(ctx, a, p):
    g = _codes(ctx, a[1])
    return K.group_rank(mat(ctx, a[0]), g.codes, g.G)


@register("group_median")
def _gmedian(ctx, a, p):
    g = _codes(ctx, a[1])
    return K.group_percentile(mat(ctx, a[0]), g.codes, g.G, 0.5)


@register("group_percentage")
def _gpct(ctx, a, p):
    g = _codes(ctx, a[1])
    return K.group_percentile(mat(ctx, a[0]), g.codes, g.G, min(1.0, max(0.0, float(p.get("percentage", 0.5)))))


@register("group_backfill")
def _gbackfill(ctx, a, p):
    x = mat(ctx, a[0])
    g = _codes(ctx, a[1])
    fill = K.group_window_mean(x, g.codes, g.G, int(p["d"]))
    return np.where(np.isnan(x), fill, x).astype(np.float32)


@register("group_vector_neut")
def _gvneut(ctx, a, p):
    g = _codes(ctx, a[2])
    return K.group_vec(mat(ctx, a[0]), mat(ctx, a[1]), g.codes, g.G, False)


@register("group_vector_proj")
def _gvproj(ctx, a, p):
    g = _codes(ctx, a[2])
    return K.group_vec(mat(ctx, a[0]), mat(ctx, a[1]), g.codes, g.G, True)


def densify_codes(codes: np.ndarray) -> Groups:
    valid = codes >= 0
    if not valid.any():
        return Groups(np.full(codes.shape, -1, np.int32), 1)
    uniq, inv = np.unique(codes[valid], return_inverse=True)
    out = np.full(codes.shape, -1, np.int32)
    out[valid] = inv.astype(np.int32)
    return Groups(out, len(uniq))


@register("densify")
def _densify(ctx, a, p):
    return densify_codes(_codes(ctx, a[0]).codes)


@register("group_cartesian_product")
def _gcart(ctx, a, p):
    g1, g2 = _codes(ctx, a[0]), _codes(ctx, a[1])
    c = np.where((g1.codes >= 0) & (g2.codes >= 0), g1.codes.astype(np.int64) * g2.G + g2.codes, -1)
    return densify_codes(c)


@register("group_coalesce")
def _gcoalesce(ctx, a, p):
    g1, g2 = _codes(ctx, a[0]), _codes(ctx, a[1])
    c = np.where(g1.codes >= 0, g1.codes.astype(np.int64), g2.codes.astype(np.int64) + g1.G)
    c = np.where((g1.codes < 0) & (g2.codes < 0), -1, c)
    return densify_codes(c)


@register("bucket")
def _bucket(ctx, a, p):
    x = mat(ctx, a[0])
    rng = str(p.get("range") or "")
    if rng:
        start, end, step = (float(v) for v in rng.split(","))
        n = int(math.floor((end - start) / step + 1e-9)) + 1
        bounds = start + step * np.arange(n)
    else:
        bounds = np.array(sorted(float(v) for v in str(p.get("buckets") or "").split(",")))
    codes = np.searchsorted(bounds, x, side="right").astype(np.int32)
    nb = len(bounds) + 1
    if p.get("skipBoth"):
        codes = np.clip(codes, 1, nb - 2) - 1
        nb = max(1, nb - 2)
    nanm = np.isnan(x)
    if p.get("NANGroup"):
        codes[nanm] = nb
        nb += 1
    else:
        codes[nanm] = -1
    return Groups(np.ascontiguousarray(codes, dtype=np.int32), nb)


# =========================================================================== transformational


@register("trade_when")
def _trade_when(ctx, a, p):
    trig, alpha, ex = (mat(ctx, v) for v in a)
    return K.trade_when(trig, alpha, ex)


@register("clamp")
def _clamp(ctx, a, p):
    x = mat(ctx, a[0])
    lo, hi = float(p.get("lower", 0.0)), float(p.get("upper", 0.0))
    if p.get("inverse"):
        r = x.copy()
        r[(x > lo) & (x < hi)] = np.float32(p.get("mask", float("nan")))
        return r
    if hi <= lo:
        return x
    return np.clip(x, lo, hi).astype(np.float32)


@register("left_tail")
def _left_tail(ctx, a, p):
    x = mat(ctx, a[0]).copy()
    x[x > float(p.get("maximum", 0.0))] = np.nan
    return x


@register("right_tail")
def _right_tail(ctx, a, p):
    x = mat(ctx, a[0]).copy()
    x[x < float(p.get("minimum", 0.0))] = np.nan
    return x


@register("tail")
def _tail(ctx, a, p):
    x = mat(ctx, a[0]).copy()
    lo, hi = float(p.get("lower", 0.0)), float(p.get("upper", 0.0))
    x[(x > lo) & (x < hi)] = np.float32(p.get("newval", 0.0))
    return x
