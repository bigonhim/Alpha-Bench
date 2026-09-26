"""BRAIN-like simulation of an alpha on the local panel.

Pipeline (per day): pasteurize inputs -> evaluate alpha -> restrict to universe -> linear decay ->
neutralize (group demean) -> scale sum|w| = 1 -> iterative truncation -> positions = w * booksize.
PnL_t = sum_i pos_{t-1-delay, i} * r_{t, i}, i.e. positions built from data through close s are
traded at close s+delay and earn the next day's return. Delay-0 is therefore optimistic locally
(it trades at the same close the data comes from).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..engine import kernels as K
from ..engine.evaluator import EvalContext, EvalError, SubexprCache
from ..engine.panel import UNIVERSE_SIZES, Panel
from ..fastexpr.ast import Node
from ..fastexpr.lower import lookback_of

BRAIN_UNIVERSE_TO_LOCAL = {"TOP3000": "TOP1500", "TOP2000": "TOP1500", "TOP1000": "TOP1000", "TOP500": "TOP500",
                           "TOPSP500": "TOP500", "TOP200": "TOP200", "TOP100": "TOP100"}
SUB_UNIVERSE = {"TOP1500": "TOP500", "TOP1000": "TOP500", "TOP500": "TOP200", "TOP200": "TOP100"}
NEUTRALIZATIONS_LOCAL = ("NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")
NEUTRALIZATIONS_ALL = NEUTRALIZATIONS_LOCAL + ("COUNTRY", "STATISTICAL", "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST",
                                               "REVERSION_AND_MOMENTUM")
REGIONS = ("USA", "GLB", "EUR", "ASI", "CHN", "JPN", "KOR", "TWN", "HKG", "AMR")
UNIVERSES_BRAIN = ("TOP3000", "TOP1000", "TOP500", "TOP200", "TOPSP500")


@dataclass
class SimSettings:
    instrumentType: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    pasteurization: str = "ON"
    unitHandling: str = "VERIFY"
    nanHandling: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "SimSettings":
        s = cls()
        if not d:
            return s
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        s.delay = int(s.delay)
        s.decay = max(0, int(s.decay))
        s.truncation = float(s.truncation)
        s.universe = str(s.universe).upper()
        s.neutralization = str(s.neutralization).upper()
        s.pasteurization = str(s.pasteurization).upper()
        s.nanHandling = str(s.nanHandling).upper()
        s.region = str(s.region).upper()
        return s

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def key(self) -> str:
        return (f"{self.region}|{self.universe}|D{self.delay}|dec{self.decay}|{self.neutralization}|"
                f"tr{self.truncation:g}|p{self.pasteurization}|n{self.nanHandling}")


@dataclass
class Periods:
    is_start: int
    os_start: int
    end: int
    brain_start: int

    def to_dict(self, panel: Panel) -> dict:
        d = panel.dates
        return {"is_start": str(d[self.is_start]), "os_start": str(d[min(self.os_start, panel.T - 1)]),
                "end": str(d[self.end - 1]), "brain_start": str(d[self.brain_start])}


def compute_periods(panel: Panel, cfg: dict) -> Periods:
    d = panel.dates
    warm = int(cfg.get("is_years_warmup", 2))
    os_years = float(cfg.get("os_years", 2))
    bw = float(cfg.get("brain_window_years", 5))
    is_start = panel.index_of(d[0] + np.timedelta64(int(365.25 * warm), "D"))
    os_start = panel.index_of(d[-1] - np.timedelta64(int(365.25 * os_years), "D"))
    is_start = min(is_start, max(0, os_start - 252))
    brain_start = max(is_start, panel.index_of(d[max(0, os_start - 1)] - np.timedelta64(int(365.25 * bw), "D")))
    return Periods(int(is_start), int(os_start), int(panel.T), int(brain_start))


def resolve_universe(panel: Panel, brain_universe: str) -> str:
    want = BRAIN_UNIVERSE_TO_LOCAL.get(brain_universe.upper(), brain_universe.upper())
    if want in panel.universes:
        return want
    avail = sorted(panel.universes, key=lambda u: UNIVERSE_SIZES.get(u, 10 ** 6))
    bigger = [u for u in avail if UNIVERSE_SIZES.get(u, 10 ** 6) >= UNIVERSE_SIZES.get(want, 0)]
    return bigger[0] if bigger else avail[-1]


def resolve_sub_universe(panel: Panel, local_universe: str) -> str | None:
    want = SUB_UNIVERSE.get(local_universe)
    avail = sorted(panel.universes, key=lambda u: UNIVERSE_SIZES.get(u, 10 ** 6))
    if want in avail:
        return want
    size = UNIVERSE_SIZES.get(local_universe, 10 ** 6)
    smaller = [u for u in avail if UNIVERSE_SIZES.get(u, 10 ** 6) < size]
    return smaller[-1] if smaller else None


# --------------------------------------------------------------------------- metrics


def perf_metrics(pnl: np.ndarray, tvr: np.ndarray, longc: np.ndarray, shortc: np.ndarray, maxw: np.ndarray,
                 booksize: float, returns_basis: str = "half_book") -> dict:
    n = len(pnl)
    if n < 2:
        return {"sharpe": 0.0, "fitness": 0.0, "returns": 0.0, "turnover": 0.0, "drawdown": 0.0, "margin_bps": 0.0,
                "pnl": 0.0, "long_count": 0.0, "short_count": 0.0, "max_weight": 0.0, "days": int(n)}
    mean = float(pnl.mean())
    sd = float(pnl.std(ddof=1))
    sharpe = mean / sd * math.sqrt(252) if sd > 0 else 0.0
    base = booksize / 2 if returns_basis == "half_book" else booksize
    ann = mean * 252 / base
    to = float(np.nanmean(tvr)) if len(tvr) else 0.0
    fit = sharpe * math.sqrt(abs(ann) / max(to, 0.125)) if to == to else 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    dd = float(np.max(peak - cum)) / base if n else 0.0
    traded = float(np.nansum(tvr)) * booksize
    margin = float(pnl.sum()) / traded * 1e4 if traded > 0 else 0.0
    return {
        "sharpe": round(sharpe, 4), "fitness": round(fit, 4), "returns": round(ann, 6), "turnover": round(to, 6),
        "drawdown": round(dd, 6), "margin_bps": round(margin, 3), "pnl": round(float(pnl.sum()), 2),
        "long_count": round(float(np.mean(longc)), 1), "short_count": round(float(np.mean(shortc)), 1),
        "max_weight": round(float(np.max(maxw)) if len(maxw) else 0.0, 5), "days": int(n),
    }


@dataclass
class SimResult:
    settings: dict
    local_universe: str
    dates: np.ndarray
    pnl: np.ndarray
    tvr: np.ndarray
    long_count: np.ndarray
    short_count: np.ndarray
    max_weight: np.ndarray
    pnl_long: np.ndarray
    pnl_short: np.ndarray
    idx: dict  # period boundaries within the result window: is, os, brain (start, end)
    metrics: dict = field(default_factory=dict)
    yearly: list = field(default_factory=list)
    sector_pnl: dict = field(default_factory=dict)
    top_names: dict = field(default_factory=dict)
    universe_size: float = 0.0
    elapsed_ms: float = 0.0
    weights: np.ndarray | None = None  # (T_window, N) float64, optional

    def is_pnl(self) -> np.ndarray:
        a, b = self.idx["is"]
        return self.pnl[a:b]

    def series_json(self) -> dict:
        cum = np.cumsum(self.pnl)
        peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
        roll = rolling_sharpe(self.pnl, 252)
        return {
            "dates": [str(d) for d in self.dates],
            "cum_pnl": np.round(cum, 0).tolist(),
            "drawdown": np.round(cum - peak, 0).tolist(),
            "cum_long": np.round(np.cumsum(self.pnl_long), 0).tolist(),
            "cum_short": np.round(np.cumsum(self.pnl_short), 0).tolist(),
            "turnover": np.round(self.tvr, 4).tolist(),
            "rolling_sharpe": [None if not np.isfinite(v) else round(float(v), 3) for v in roll],
            "long_count": self.long_count.astype(int).tolist(),
            "short_count": self.short_count.astype(int).tolist(),
        }


def rolling_sharpe(pnl: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(pnl), np.nan)
    if len(pnl) < w:
        return out
    cs = np.cumsum(np.r_[0.0, pnl])
    cs2 = np.cumsum(np.r_[0.0, pnl * pnl])
    s = cs[w:] - cs[:-w]
    s2 = cs2[w:] - cs2[:-w]
    mean = s / w
    var = np.maximum(s2 / w - mean * mean, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[w - 1:] = mean / np.sqrt(var) * math.sqrt(252)
    return out


# --------------------------------------------------------------------------- simulation


NEUT_GROUP = {"SECTOR": "sector", "INDUSTRY": "industry", "SUBINDUSTRY": "subindustry"}
WARMUP_MIN_ROWS = 520  # fixed warm-up so different expressions share cache keys (rounded to 64 rows)


def eval_start(w_start: int, lookback: int) -> int:
    need = max(int(lookback), WARMUP_MIN_ROWS)
    need = ((need + 63) // 64) * 64
    return max(0, w_start - need)


def neutral_codes(ctx: EvalContext, neutralization: str) -> tuple[np.ndarray, int, bool]:
    """(codes, G, neutralize?) for a BRAIN neutralization; unsupported risk-model options fall back to MARKET."""
    n = neutralization.upper()
    if n == "NONE":
        codes, G = ctx.codes("market")
        return codes, G, False
    name = NEUT_GROUP.get(n, "market")
    try:
        codes, G = ctx.codes(name)
    except (KeyError, EvalError):
        codes, G = ctx.codes("market")
    return codes, G, True


def weights_from_alpha(A: np.ndarray, ctx: EvalContext, s: SimSettings) -> np.ndarray:
    """alpha (T, N) float32 -> weights (T, N) float64 with sum|w| = 1 per day (0 where no position)."""
    univ = ctx.univ_mask
    if s.decay and s.decay > 1:
        masked = np.where(univ, A, np.float32(np.nan)).astype(np.float32)
        A = K.decay_linear_fast(np.ascontiguousarray(masked), int(s.decay))
    codes, G, neut = neutral_codes(ctx, s.neutralization)
    trunc = float(s.truncation) if s.truncation else 0.0
    return K.weights_kernel(np.ascontiguousarray(A, dtype=np.float32), univ, codes, G, neut,
                            s.nanHandling == "ON", trunc, 12)


def postprocess(alpha: np.ndarray, univ_mask: np.ndarray, panel: Panel, s: SimSettings) -> np.ndarray:
    """Standalone variant of the weight pipeline (used by tests and tools without an EvalContext)."""
    A = np.where(univ_mask, alpha, np.float32(np.nan)).astype(np.float32)
    if s.nanHandling == "ON":
        A = np.where(univ_mask & np.isnan(A), np.float32(0.0), A)
    if s.decay and s.decay > 1:
        A = K.decay_linear_fast(np.ascontiguousarray(A), int(s.decay))
        A[~univ_mask] = np.nan
    T = A.shape[0]
    n = s.neutralization.upper()
    if n == "NONE":
        codes, G, neut = np.zeros((T, panel.N), np.int32), 1, False
    else:
        name = NEUT_GROUP.get(n)
        if name is None:
            codes, G = np.zeros((T, panel.N), np.int32), 1
        else:
            c, G = panel.group(name)
            codes = np.ascontiguousarray(np.broadcast_to(c[None, :], (T, panel.N)), dtype=np.int32)
        neut = True
    trunc = float(s.truncation) if s.truncation else 0.0
    return K.weights_kernel(np.ascontiguousarray(A), univ_mask, codes, G, neut, False, trunc, 12)


def simulate(node: Node, settings: SimSettings | dict, panel: Panel, periods: Periods,
             cache: SubexprCache | None = None, span: str = "all", universe_override: str | None = None,
             booksize: float = 20e6, returns_basis: str = "half_book", keep_weights: bool = False,
             alpha: np.ndarray | None = None, rows: tuple[int, int] | None = None) -> SimResult:
    t0 = time.perf_counter()
    s = settings if isinstance(settings, SimSettings) else SimSettings.from_dict(settings)
    local_u = universe_override or resolve_universe(panel, s.universe)
    delay = 0 if int(s.delay) == 0 else 1
    if rows is not None:
        w_start, w_end = rows
    else:
        w_start = periods.is_start
        w_end = periods.end if span == "all" else periods.os_start
    lb = lookback_of(node) + max(0, int(s.decay)) + delay + 6
    r0 = eval_start(w_start, lb)
    r1 = w_end
    ctx = EvalContext(panel, r0, r1, local_u, s.pasteurization != "OFF", cache)
    A = alpha if alpha is not None else ctx.evaluate_matrix(node)
    univ = ctx.univ_mask
    W = weights_from_alpha(A, ctx, s)
    ret = ctx.raw("returns")
    lag = 1 + delay
    T = W.shape[0]
    o0 = w_start - r0
    is_a = max(o0, periods.is_start - r0)
    is_b = max(is_a, min(T, periods.os_start - r0))
    pnl, pnl_long, dW, longc, shortc, maxw, contrib_is = K.pnl_stats(W, ret, lag, float(booksize), is_a, is_b)
    pnl_short = pnl - pnl_long

    o = w_start - r0  # offset of window start in the eval slice
    sl = slice(o, T)
    dates = panel.dates[w_start:w_end]
    res = SimResult(
        settings=s.to_dict(), local_universe=local_u, dates=dates, pnl=pnl[sl], tvr=dW[sl],
        long_count=longc[sl], short_count=shortc[sl], max_weight=maxw[sl], pnl_long=pnl_long[sl],
        pnl_short=pnl_short[sl], idx={},
    )
    n = len(dates)

    def rel(a: int, b: int) -> tuple[int, int]:
        return max(0, min(n, a - w_start)), max(0, min(n, b - w_start))

    res.idx = {"is": rel(periods.is_start, periods.os_start), "os": rel(periods.os_start, periods.end),
               "brain": rel(periods.brain_start, periods.os_start), "all": (0, n)}
    res.universe_size = float(univ[o:].sum(axis=1).mean()) if univ is not None else float(panel.N)
    for name, (a, b) in res.idx.items():
        if b - a >= 2:
            res.metrics[name] = perf_metrics(res.pnl[a:b], res.tvr[a:b], res.long_count[a:b], res.short_count[a:b],
                                             res.max_weight[a:b], booksize, returns_basis)
    # yearly table
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    for y in np.unique(years):
        m = years == y
        if m.sum() < 20:
            continue
        met = perf_metrics(res.pnl[m], res.tvr[m], res.long_count[m], res.short_count[m], res.max_weight[m],
                           booksize, returns_basis)
        met["year"] = int(y)
        a, b = res.idx["os"]
        first = int(np.argmax(m))
        met["period"] = "OS" if first >= a and b > a else "IS"
        res.yearly.append(met)
    # attribution over IS
    a, b = res.idx["is"]
    if b > a:
        per_stock = contrib_is
        try:
            codes, G = panel.group("sector")
            labels = panel.group_labels.get("sector", [])
            valid = codes >= 0
            sums = np.bincount(codes[valid], weights=per_stock[valid], minlength=G)
            res.sector_pnl = {labels[g] if g < len(labels) else str(g): round(float(v), 0) for g, v in enumerate(sums)}
        except KeyError:
            res.sector_pnl = {}
        order = np.argsort(per_stock)
        res.top_names = {
            "best": [[panel.tickers[i], round(float(per_stock[i]), 0)] for i in order[::-1][:8]],
            "worst": [[panel.tickers[i], round(float(per_stock[i]), 0)] for i in order[:8]],
        }
    if keep_weights:
        res.weights = W[sl]
    res.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res


def sweep_cells(node: Node, base: SimSettings, cells: list[dict], panel: Panel, periods: Periods,
                cache: SubexprCache | None, booksize: float = 20e6, returns_basis: str = "half_book"):
    """Yield (cell index, IS metrics) for settings variants of one alpha.

    The alpha is evaluated once; each decay value is applied once and reused across every
    neutralization x truncation combination, so a 128-cell grid costs ~8 decays + 128 fused passes.
    """
    local_u = resolve_universe(panel, base.universe)
    delay = 0 if int(base.delay) == 0 else 1
    max_decay = max(int(c.get("decay", base.decay) or 0) for c in cells) if cells else 0
    w_start, w_end = periods.is_start, periods.os_start
    r0 = eval_start(w_start, lookback_of(node) + max_decay + delay + 6)
    ctx = EvalContext(panel, r0, w_end, local_u, base.pasteurization != "OFF", cache)
    A = np.ascontiguousarray(ctx.evaluate_matrix(node), dtype=np.float32)
    univ = ctx.univ_mask
    ret = ctx.raw("returns")
    lag = 1 + delay
    o = w_start - r0
    T = A.shape[0]
    masked = None
    cur_decay, Ad = None, A
    for k in sorted(range(len(cells)), key=lambda j: int(cells[j].get("decay", base.decay) or 0)):
        c = cells[k]
        d = int(c.get("decay", base.decay) or 0)
        if d != cur_decay:
            if d > 1:
                if masked is None:
                    masked = np.where(univ, A, np.float32(np.nan)).astype(np.float32)
                Ad = K.decay_linear_fast(masked, d)
            else:
                Ad = A
            cur_decay = d
        s = SimSettings.from_dict({**base.to_dict(), **c})
        codes, G, neut = neutral_codes(ctx, s.neutralization)
        W = K.weights_kernel(Ad, univ, codes, G, neut, s.nanHandling == "ON", float(s.truncation or 0.0), 12)
        pnl, _pl, tvr, lc, sc, mw, _c = K.pnl_stats(W, ret, lag, float(booksize), o, T)
        yield k, perf_metrics(pnl[o:], tvr[o:], lc[o:], sc[o:], mw[o:], booksize, returns_basis)


def deflated_sharpe(sharpe_ann: float, n_trials: int, n_days: int, skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Bailey & Lopez de Prado deflated Sharpe ratio (probability the Sharpe beats the best of n null trials)."""
    from ..engine.ops import norm_ppf

    if n_days < 10:
        return {"dsr": 0.0, "expected_max_sharpe": 0.0, "haircut_sharpe": sharpe_ann}
    sr = sharpe_ann / math.sqrt(252)
    n = max(1, int(n_trials))
    g = 0.5772156649
    v = 1.0 / n_days
    if n > 1:
        z1 = float(norm_ppf(np.array([1 - 1.0 / n]))[0])
        z2 = float(norm_ppf(np.array([1 - 1.0 / (n * math.e)]))[0])
        sr0 = math.sqrt(v) * ((1 - g) * z1 + g * z2)
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - sr0) * math.sqrt(n_days - 1) / denom
    dsr = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return {"dsr": round(dsr, 4), "expected_max_sharpe": round(sr0 * math.sqrt(252), 3),
            "haircut_sharpe": round((sr - sr0) * math.sqrt(252), 3)}
