"""Re-engineer: turn a weak alpha into a strong one through a staged, explainable local search.

The search pulls the same levers a researcher would, in a fixed order, and every accepted step is
recorded so the result reads as a recipe rather than a black box:

  direction   sign, and component surgery (keep, drop or flip the parts of a combination)
  shape       scale-free ratios, back-filled fundamentals, peer / cross-sectional / time-series normalization
  neutralize  settings neutralization, size buckets, regression against style factors
  horizon     coordinate search over every lookback window on a standard ladder
  turnover    decay, smoothing, event gates, hump, turnover targeting
  condition   scale conviction by volume surprise, volatility or liquidity
  blend       add one decorrelated companion signal (optional)
  settings    joint decay x neutralization x truncation polish, robust (neighbourhood) pick

This module is pure: moves propose rewrites and ``objective`` scores metric dicts. The job in
``jobs/reengineer.py`` evaluates the proposals and runs the beam search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..catalog import field_map
from ..fastexpr.ast import Const, Field, Node, Op, iter_paths, node_at, replace_at
from ..fastexpr.lower import PHYSICAL_UNITS, lower_text, simplify, unit_of
from ..fastexpr.printer import to_expr
from .build import arg_kind, capbucket, fields_of, mk, node_type
from .doctor import _backfill_fundamentals, _scale_windows
from .grammar import is_degenerate

STAGES: dict[str, tuple[str, str]] = {
    "direction": ("Direction", "Sign and component surgery"),
    "shape": ("Shape", "Scale-free ratios and normalization"),
    "neutralize": ("Neutralize", "Remove group, size and style-factor bets"),
    "horizon": ("Horizon", "Tune every lookback window"),
    "turnover": ("Turnover", "Trade less without losing the edge"),
    "condition": ("Condition", "Bet harder where the edge is stronger"),
    "blend": ("Blend", "Add one decorrelated companion signal"),
    "settings": ("Settings", "Joint decay, neutralization and truncation polish"),
    "evolve": ("Evolve", "Genetic programming around the best versions"),
}
PASS_STAGES = ("shape", "neutralize", "horizon", "turnover", "condition", "blend")

NEUTS = ("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")
NORMALIZERS = {"rank", "zscore", "quantile", "scale", "normalize", "group_rank", "group_zscore", "group_neutralize",
               "winsorize"}
RANKLIKE = {"rank", "group_rank", "quantile"}
COMBINERS = {"add", "subtract", "multiply", "if_else", "max", "min"}
WINDOW_LADDER = (2, 3, 5, 10, 15, 20, 40, 60, 90, 120, 180, 252, 504)
MAX_SIZE = 45
MAX_DEPTH = 10


@dataclass(frozen=True)
class Factor:
    id: str
    title: str
    expr: str
    why: str = ""


STYLE_FACTORS = (
    Factor("size", "Size", "rank(cap)"),
    Factor("momentum", "12-month momentum", "rank(ts_sum(returns, 252))"),
    Factor("reversal", "Short-term reversal", "rank(-ts_sum(returns, 5))"),
    Factor("volatility", "Volatility", "rank(ts_std_dev(returns, 60))"),
    Factor("liquidity", "Share turnover", "rank(ts_mean(volume, 20) / sharesout)"),
)

CONDITIONERS = (
    Factor("volume", "volume surprise", "volume / adv20",
           "Leans into names trading on unusual volume, where new information is being priced."),
    Factor("volatile", "volatility", "ts_std_dev(returns, 20)",
           "Bets harder on volatile names, where mispricings tend to be larger."),
    Factor("calm", "calm names", "-ts_std_dev(returns, 20)",
           "Bets harder on calm names, where the signal is less drowned out by noise."),
    Factor("liquid", "liquidity", "ts_mean(volume * close, 20)",
           "Tilts toward liquid names; this also tends to fix a failing sub-universe check."),
)

COMPANIONS = (
    Factor("reversal", "short-term reversal", "-ts_sum(returns, 5)"),
    Factor("momentum", "12-1 momentum", "ts_delay(ts_sum(returns, 231), 21)"),
    Factor("high52", "closeness to the 52-week high", "close / ts_max(high, 252)"),
    Factor("earnings_yield", "earnings yield", "ts_backfill(income, 120) / cap"),
    Factor("sales_yield", "sales yield", "ts_backfill(sales, 120) / cap"),
    Factor("cf_yield", "cash-flow yield", "ts_backfill(cashflow_op, 120) / cap"),
    Factor("roa", "return on assets", "ts_backfill(operating_income, 120) / ts_backfill(assets, 120)"),
    Factor("gross_prof", "gross profitability",
           "(ts_backfill(sales, 120) - ts_backfill(cogs, 120)) / ts_backfill(assets, 120)"),
    Factor("low_vol", "low volatility", "-ts_std_dev(returns, 60)"),
    Factor("pv_div", "price-volume divergence", "-ts_corr(rank(close), rank(volume), 10)"),
)

SCALE_DENOMS: dict[str, tuple[tuple[str, str], ...]] = {
    "price": (("close", "close"),),
    "per_share": (("close", "close"),),
    "dollar": (("cap", "cap"), ("assets", "ts_backfill(assets, 120)")),
    "shares": (("sharesout", "sharesout"), ("adv20", "adv20")),
}


# --------------------------------------------------------------------------- scoring


def sharpe_of(x: np.ndarray) -> float:
    if len(x) < 10:
        return 0.0
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0


def half_sharpes(pnl: np.ndarray | None) -> tuple[float, float]:
    if pnl is None or len(pnl) < 120:
        return 0.0, 0.0
    h = len(pnl) // 2
    return sharpe_of(pnl[:h]), sharpe_of(pnl[h:])


def pnl_corr(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or len(a) != len(b) or len(a) < 20:
        return None
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0 or sb == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def objective(m: dict, pnl: np.ndarray | None, size: int, cfg: dict, delay: int = 1,
              self_corr: dict | None = None) -> tuple[float, dict]:
    """Single robust score for ranking variants on in-sample data only.

    Fitness carries Sharpe, returns and turnover the way BRAIN weighs them; Sharpe is added again because
    it is a hard check of its own; the weaker half of the IS period rewards edges that hold across time
    rather than in one lucky stretch. Hard-check violations, deep drawdowns, correlation with submitted
    alphas and extra complexity are subtracted.
    """
    sh = float(m.get("sharpe") or 0.0)
    fit = float(m.get("fitness") or 0.0)
    to = float(m.get("turnover") or 0.0)
    mw = float(m.get("max_weight") or 0.0)
    dd = float(m.get("drawdown") or 0.0)
    h1, h2 = half_sharpes(pnl) if pnl is not None and len(pnl) >= 120 else (sh, sh)
    clip = lambda v: max(-3.0, min(6.0, v))  # noqa: E731
    base = clip(fit) + 0.5 * clip(sh) + 0.25 * clip(min(h1, h2))
    pen: dict[str, float] = {}
    if to > cfg["turnover_max"]:
        pen["turnover_high"] = 1.0 + 5.0 * (to - cfg["turnover_max"])
    if to < cfg["turnover_min"]:
        pen["turnover_low"] = 1.0 + 50.0 * (cfg["turnover_min"] - to)
    if mw > cfg["max_weight"] + 1e-9:
        pen["concentration"] = 1.0 + 10.0 * (mw - cfg["max_weight"])
    dd_max = float(cfg.get("local_gates", {}).get("drawdown_max", 0.5))
    if dd > dd_max:
        pen["drawdown"] = 2.0 * (dd - dd_max)
    if size > 12:
        pen["complexity"] = 0.015 * (size - 12)
    if self_corr and self_corr.get("max_corr", 0.0) >= cfg["self_corr_max"]:
        other = float(self_corr.get("alpha_sharpe") or 0.0)
        if sh < other * (1 + cfg["self_corr_sharpe_improvement"]):
            pen["self_correlation"] = 1.0 + 2.0 * (self_corr["max_corr"] - cfg["self_corr_max"])
    score = base - sum(pen.values())
    return round(score, 4), {"base": round(base, 4), "penalties": {k: round(v, 4) for k, v in pen.items()},
                             "halves": [round(h1, 3), round(h2, 3)]}


def meets_target(m: dict, parts: dict, target_sharpe: float, target_fitness: float) -> bool:
    return (float(m.get("sharpe") or 0) >= target_sharpe and float(m.get("fitness") or 0) >= target_fitness
            and not {k for k in parts.get("penalties", {}) if k != "complexity"})


# --------------------------------------------------------------------------- tree helpers


def peel(node: Node) -> tuple[Node, int]:
    """Strip outer normalizers and sign flips. Returns (core, sign)."""
    sign = 1
    n = node
    while isinstance(n, Op):
        if n.name == "reverse":
            sign = -sign
        elif n.name not in NORMALIZERS:
            break
        n = n.args[0]
    return n, sign


def signed(n: Node, sign: int) -> Node:
    return mk("reverse", n) if sign < 0 else n


def ranked(n: Node) -> Node:
    """``n`` if it is already a bounded rank (or a negated one), else rank(n)."""
    inner = n.args[0] if isinstance(n, Op) and n.name == "reverse" else n
    return n if isinstance(inner, Op) and inner.name in RANKLIKE else mk("rank", n)


def has_data(n: Node) -> bool:
    return any(isinstance(x, Field) and node_type(x) == "m" for x in n.walk())


def matrix_sites(n: Node) -> list[tuple[tuple[int, ...], Node]]:
    """Paths of numeric subtrees that sit in a numeric argument slot (safe to replace or extract)."""
    out = []
    for path, sub in iter_paths(n):
        if isinstance(sub, Const) or node_type(sub) != "m":
            continue
        if path:
            parent = node_at(n, path[:-1])
            if isinstance(parent, Op) and arg_kind(parent, path[-1]) != "m":
                continue
        out.append((path, sub))
    return out


def find_combiner(node: Node) -> tuple[tuple[int, ...], Op] | tuple[None, None]:
    """Descend through single-signal wrappers to the first node that combines several signals."""
    path: tuple[int, ...] = ()
    n = node
    while isinstance(n, Op):
        if n.name in COMBINERS:
            return path, n
        if n.name == "trade_when":
            idx = 1
        else:
            data = [i for i, a in enumerate(n.args) if not isinstance(a, Const) and node_type(a) == "m"]
            if len(data) != 1:
                return None, None
            idx = data[0]
        path = path + (idx,)
        n = n.args[idx]
    return None, None


def short(n: Node, width: int = 60) -> str:
    t = to_expr(n)
    return t if len(t) <= width else t[: width - 1] + "…"


def _parse(expr: str) -> Node:
    return lower_text(expr)


# --------------------------------------------------------------------------- moves


@dataclass
class Move:
    stage: str
    label: str
    reason: str
    node: Node
    settings: dict

    @property
    def expr(self) -> str:
        return to_expr(self.node)


@dataclass
class MoveContext:
    local_fields: set[str]
    metrics: dict = field(default_factory=dict)
    exposures: dict[str, float] = field(default_factory=dict)
    allow_structure: bool = True
    orthogonalize_to: Node | None = None
    orthogonalize_label: str = ""

    def has(self, *names: str) -> bool:
        return all(n in self.local_fields for n in names)

    def local(self, n: Node) -> bool:
        return fields_of(n) <= (self.local_fields | {"market"})


def _direction(node: Node, s: dict, ctx: MoveContext) -> list[Move]:
    out = [Move("direction", "Flip the sign", "If the signal points the wrong way, the reversed alpha earns the "
                "opposite PnL.", mk("reverse", node), s)]
    if not ctx.allow_structure:
        return out
    cpath, comb = find_combiner(node)
    if comb is not None:
        parts = [(i, a) for i, a in enumerate(comb.args) if not isinstance(a, Const) and node_type(a) == "m"
                 and has_data(a)]
        if comb.name == "if_else":
            parts = [(i, a) for i, a in parts if i > 0]
        for i, a in parts:
            out.append(Move("direction", f"Keep only {short(a, 40)}",
                            "Tests whether this part carries the edge on its own; the rest may only add noise.",
                            replace_at(node, cpath, a), s))
        if comb.name in ("add", "subtract") and len(parts) == 2:
            for i, a in parts:
                flipped = comb.with_args(tuple(mk("reverse", x) if k == i else x for k, x in enumerate(comb.args)))
                out.append(Move("direction", f"Flip the sign of {short(a, 40)}",
                                "One component may be pointing the wrong way while the other is right.",
                                replace_at(node, cpath, flipped), s))
        if comb.name in ("add", "multiply") and len(comb.args) > 2:
            for i, a in enumerate(comb.args):
                rest = comb.args[:i] + comb.args[i + 1:]
                out.append(Move("direction", f"Drop {short(a, 40)}",
                                "Removing a component that loses money raises the Sharpe of the rest.",
                                replace_at(node, cpath, Op(comb.name, rest, comb.params)), s))
    core_key = peel(node)[0].key
    seen = {core_key}
    subs = [sub for p, sub in matrix_sites(node) if p and isinstance(sub, Op) and sub.size >= 3 and has_data(sub)]
    subs.sort(key=lambda x: -x.size)
    for sub in subs:
        core, _ = peel(sub)
        if core.key in seen or not isinstance(core, Op):
            continue
        seen.add(core.key)
        for sg in (1, -1):
            out.append(Move("direction", f"Isolate {'-' if sg < 0 else ''}{short(core, 40)}",
                            "A sub-signal can be cleaner than the whole expression it is buried in.",
                            signed(mk("rank", core), sg), s))
        if len(seen) > 3:
            break
    return out


def _shape(node: Node, s: dict, ctx: MoveContext) -> list[Move]:
    core, sign = peel(node)
    out: list[Move] = []
    u = unit_of(core, field_map())
    if u in PHYSICAL_UNITS and ctx.allow_structure:
        for name, den in SCALE_DENOMS.get(u, ()):
            dn = _parse(den)
            if not ctx.local(dn) or name in fields_of(core) and name != "close":
                continue
            out.append(Move("shape", f"Scale-free: divide by {name}",
                            f"The signal is measured in {u.replace('_', '-')} units, so large or high-priced stocks "
                            f"dominate it. Dividing by {name} makes stocks comparable.",
                            signed(mk("rank", mk("divide", core, dn)), sign), s))
    bf = _backfill_fundamentals(core)
    if bf.key != core.key:
        out.append(Move("shape", "Back-fill fundamentals",
                        "ts_backfill carries the last reported value forward, so stocks between filings keep a "
                        "position instead of dropping out.", signed(mk("rank", bf), sign), s))
    forms: list[tuple[str, Node, str]] = [
        ("Cross-sectional rank", mk("rank", core), "Ranks are bounded, so outliers cannot dominate the book."),
        ("Winsorized z-score", mk("zscore", mk("winsorize", core, std=3.0)),
         "Keeps the strength of the signal but clips outliers at 3 standard deviations."),
        ("Rank within industry", mk("group_rank", core, Field("industry")),
         "Compares each stock only with its industry peers, removing industry-level bets."),
        ("Rank within sub-industry", mk("group_rank", core, Field("subindustry")),
         "Compares each stock only with its closest peers."),
        ("Rank within sector", mk("group_rank", core, Field("sector")),
         "Compares each stock with its sector, a broader and more stable peer group."),
        ("Emphasize the extremes", mk("signed_power", mk("subtract", mk("rank", core), Const(0.5)), Const(2.0)),
         "Squares the centred rank so the book leans on the strongest signals, where edges usually concentrate."),
    ]
    if not (isinstance(core, Op) and core.name in ("ts_rank", "ts_zscore", "ts_scale")):
        forms += [
            ("Versus its own 1-year history", mk("rank", mk("ts_rank", core, d=252)),
             "Where today's value sits in its own 1-year range: turns a static level into an anomaly signal."),
            ("Versus its own 3-month history", mk("rank", mk("ts_zscore", core, d=60)),
             "How unusual today's value is relative to the last 3 months."),
        ]
    for label, n, why in forms:
        out.append(Move("shape", label, why, signed(n, sign), s))
    return out


def _neutralize(node: Node, s: dict, ctx: MoveContext) -> list[Move]:
    out: list[Move] = []
    cur = str(s.get("neutralization", "")).upper()
    for n in NEUTS:
        if n != cur:
            out.append(Move("neutralize", f"Neutralization: {n.lower()}",
                            "Demeaning within a different grouping changes which group-level bets are removed.",
                            node, {**s, "neutralization": n}))
    if ctx.has("cap") and "bucket" not in {x.name for x in node.walk() if isinstance(x, Op)}:
        out.append(Move("neutralize", "Neutralize within size buckets",
                        "Removes the small-versus-large tilt that often fails the sub-universe check.",
                        mk("group_neutralize", node, capbucket()), s))
    facs = sorted(STYLE_FACTORS, key=lambda f: -abs(ctx.exposures.get(f.id, 0.0)))
    k = 0
    for f in facs:
        fn = _parse(f.expr)
        e = ctx.exposures.get(f.id)
        if not ctx.local(fn) or (e is not None and abs(e) >= 0.8):
            continue  # a factor this close to the alpha *is* the alpha; regressing it out would erase the edge
        why = f"Removes the part of the signal explained by {f.title.lower()}"
        why += f" (PnL correlation {e:+.2f})." if e is not None else "."
        out.append(Move("neutralize", f"Regress out {f.title.lower()}", why, mk("regression_neut", node, fn), s))
        k += 1
        if k >= 3:
            break
    if ctx.orthogonalize_to is not None:
        out.append(Move("neutralize", f"Orthogonalize to {ctx.orthogonalize_label or 'the correlated alpha'}",
                        "vector_neut removes the component shared with a submitted alpha, which is what fails "
                        "SELF_CORRELATION.", mk("vector_neut", node, ctx.orthogonalize_to), s))
    return out


def _nearest(w: int) -> int:
    return min(range(len(WINDOW_LADDER)), key=lambda i: abs(WINDOW_LADDER[i] - w) / WINDOW_LADDER[i])


def _horizon(node: Node, s: dict, ctx: MoveContext) -> list[Move]:
    out: list[Move] = []
    sites = [(p, n) for p, n in iter_paths(node) if isinstance(n, Op) and isinstance(n.param("d"), int)
             and n.param("d") >= 2]
    for path, sub in sites[:6]:
        d = int(sub.param("d"))
        i = _nearest(d)
        cands = {WINDOW_LADDER[j] for j in (i - 1, i + 1) if 0 <= j < len(WINDOW_LADDER)}
        if WINDOW_LADDER[i] != d:
            cands.add(WINDOW_LADDER[i])
        for w in sorted(cands):
            if w == d:
                continue
            out.append(Move("horizon", f"{sub.name} window {d} → {w}",
                            "Longer windows trade slower and average out noise; shorter ones react faster."
                            if w > d else "Shorter windows react faster to new information.",
                            replace_at(node, path, sub.with_param("d", w)), s))
    if len(sites) >= 2:
        out.append(Move("horizon", "All windows ×2", "Slows the whole signal down evenly.",
                        _scale_windows(node, 2.0), s))
        out.append(Move("horizon", "All windows ×½", "Speeds the whole signal up evenly.",
                        _scale_windows(node, 0.5), s))
    return out


def _turnover(node: Node, s: dict, ctx: MoveContext) -> list[Move]:
    out: list[Move] = []
    to = float(ctx.metrics.get("turnover") or 0.0)
    cur = int(s.get("decay", 0) or 0)

    def decay(d: int, why: str) -> None:
        if d != cur and 0 <= d <= 40:
            out.append(Move("turnover", f"Decay {cur} → {d}", why, node, {**s, "decay": d}))

    if to > 0.18:
        for d in sorted({cur + 3, cur + 6, max(10, cur * 2)}):
            decay(d, "Linear decay blends the last few days of positions, cutting turnover; fitness rewards "
                     "turnover down to 12.5%.")
        root = node.name if isinstance(node, Op) else ""
        if root != "ts_decay_linear":
            for d in (5, 10):
                out.append(Move("turnover", f"Smooth with ts_decay_linear(·, {d})",
                                "Smoothing inside the expression lowers turnover before neutralization.",
                                mk("ts_decay_linear", node, d=d), s))
        if root != "hump":
            out.append(Move("turnover", "Ignore small changes (hump)",
                            "hump() keeps yesterday's value unless the signal moves by more than 10% of its "
                            "typical size, so positions only change on real news.", mk("hump", node, hump=0.1), s))
        if root != "trade_when" and ctx.has("volume", "adv20"):
            out.append(Move("turnover", "Trade only on abnormal volume",
                            "trade_when(volume > adv20, …) rebalances on high-volume days and holds otherwise.",
                            mk("trade_when", mk("greater", Field("volume"), Field("adv20")), node, Const(-1.0)), s))
        out.append(Move("turnover", "Target 20% turnover",
                        "ts_target_tvr_decay picks the smoothing strength that lands turnover near 20%.",
                        mk("ts_target_tvr_decay", node, target_tvr=0.2), s))
    elif to < 0.05:
        decay(0, "The signal barely moves; decay only delays it further.")
        decay(cur // 2, "Less decay lets a slow signal react faster.")
    else:
        decay(cur + 2, "A little more decay trims turnover.")
        decay(max(0, cur - 2), "A little less decay reacts faster.")
    return out


def _condition(node: Node, s: dict, ctx: MoveContext) -> list[Move]:
    out: list[Move] = []
    if not ctx.allow_structure:
        return out
    centred = mk("subtract", ranked(node), Const(0.5))
    for c in CONDITIONERS:
        cn = _parse(c.expr)
        if not ctx.local(cn):
            continue
        out.append(Move("condition", f"Scale conviction by {c.title}", c.why,
                        mk("multiply", centred, mk("rank", cn)), s))
    return out


@dataclass
class Partner:
    id: str
    title: str
    node: Node
    sharpe: float
    pnl: np.ndarray
    source: str = "companion"


def blend_moves(node: Node, s: dict, partners: list[tuple[Partner, float]], weights=(0.5, 1.0)) -> list[Move]:
    """``partners`` holds (partner, PnL correlation with ``node``)."""
    out: list[Move] = []
    a = ranked(node)
    for p, corr in partners:
        b = ranked(p.node)
        for w in weights:
            bw = b if w == 1 else mk("multiply", Const(float(w)), b)
            out.append(Move("blend", f"Blend in {p.title} (weight {w:g})",
                            f"{p.title[:1].upper() + p.title[1:]} has IS Sharpe {p.sharpe:.2f} and only {corr:+.2f} "
                            "PnL correlation with this alpha; adding a decorrelated edge raises Sharpe.",
                            mk("add", a, bw), s))
    return out


def blend_candidates(pnl: np.ndarray, sharpe: float, pool: list[Partner], k: int = 4) -> list[tuple[Partner, float]]:
    """Rank partners by the Sharpe an equal-risk blend would have: (s1 + s2) / sqrt(2 + 2 rho)."""
    scored = []
    for p in pool:
        c = pnl_corr(pnl, p.pnl)
        if c is None or p.sharpe <= 0.3 or abs(c) > 0.6:
            continue
        est = (max(sharpe, 0.0) + p.sharpe) / math.sqrt(max(0.2, 2 + 2 * c))
        if est > sharpe + 0.1:
            scored.append((est, p, c))
    scored.sort(key=lambda x: -x[0])
    return [(p, c) for _, p, c in scored[:k]]


def settings_cells(s: dict) -> list[dict]:
    cur = int(s.get("decay", 0) or 0)
    decays = sorted({d for d in (cur - 4, cur - 2, cur, cur + 2, cur + 4, cur + 8) if 0 <= d <= 40})
    return [{"decay": d, "neutralization": n, "truncation": t} for d in decays for n in NEUTS for t in (0.05, 0.08)]


_STAGE_FNS = {"direction": _direction, "shape": _shape, "neutralize": _neutralize, "horizon": _horizon,
              "turnover": _turnover, "condition": _condition}


def valid(n: Node, ctx: MoveContext) -> bool:
    return (n.size <= MAX_SIZE and n.depth <= MAX_DEPTH and not is_degenerate(n) and ctx.local(n)
            and node_type(n) == "m")


def propose(stage: str, node: Node, settings: dict, ctx: MoveContext) -> list[Move]:
    """Candidate rewrites for one stage, simplified, validated and de-duplicated."""
    fn = _STAGE_FNS.get(stage)
    if fn is None:
        return []
    out, seen = [], {(node.key, _skey(settings))}
    for mv in fn(node, settings, ctx):
        try:
            mv.node = simplify(mv.node)
        except (AssertionError, KeyError, TypeError):
            continue
        k = (mv.node.key, _skey(mv.settings))
        if k in seen or not valid(mv.node, ctx):
            continue
        seen.add(k)
        out.append(mv)
    return out


def _skey(s: dict) -> str:
    return f"{s.get('decay')}|{str(s.get('neutralization', '')).upper()}|{s.get('truncation')}"


# --------------------------------------------------------------------------- diagnosis


def diagnose(m_is: dict, m_os: dict | None, halves: tuple[float, float], checks: dict, extras: dict | None,
             exposures: dict[str, float], cfg: dict, delay: int = 1) -> list[dict]:
    """Plain-English reasons an alpha is weak, each pointing at the stage that addresses it."""
    th = cfg["delay0" if int(delay) == 0 else "delay1"]
    sh = float(m_is.get("sharpe") or 0.0)
    fit = float(m_is.get("fitness") or 0.0)
    to = float(m_is.get("turnover") or 0.0)
    mw = float(m_is.get("max_weight") or 0.0)
    dd = float(m_is.get("drawdown") or 0.0)
    items: list[dict] = []

    def add(code: str, severity: str, title: str, detail: str, stage: str) -> None:
        items.append({"code": code, "severity": severity, "title": title, "detail": detail, "stage": stage})

    if sh < -0.3:
        add("backwards", "bad", "The signal points the wrong way",
            f"IS Sharpe is {sh:.2f}. The reversed alpha would earn roughly the opposite PnL.", "direction")
    elif sh < 0.5:
        add("no_edge", "bad", "Little predictive power",
            f"IS Sharpe is {sh:.2f}. The idea needs restructuring, not just tuning.", "shape")
    elif sh < th["sharpe_min"]:
        add("below_bar", "warn", "Edge below the submission bar",
            f"IS Sharpe {sh:.2f} vs the required {th['sharpe_min']}.", "neutralize")
    if to > cfg["turnover_max"]:
        add("churn", "bad", "Trades far too much",
            f"Turnover is {to:.0%} (limit {cfg['turnover_max']:.0%}). Costs would eat the edge.", "turnover")
    elif to > 0.3 and fit < th["fitness_min"] and sh > 0:
        add("turnover_drag", "warn", "Turnover drags fitness down",
            f"At {to:.0%} turnover, fitness is {fit:.2f}. Fitness improves as turnover falls toward 12.5%.",
            "turnover")
    if to < cfg["turnover_min"]:
        add("static", "bad", "Barely trades",
            f"Turnover is {to:.1%} (minimum {cfg['turnover_min']:.0%}).", "turnover")
    if mw > cfg["max_weight"]:
        add("concentrated", "bad", "A few stocks dominate the book",
            f"Largest single-stock weight is {mw:.1%} (limit {cfg['max_weight']:.0%}). Raw, unnormalized values "
            "let outliers take over.", "shape")
    if any(c["name"] == "LOW_SUB_UNIVERSE_SHARPE" and c["result"] == "FAIL" for c in checks.get("checks", [])):
        add("small_caps", "bad", "The edge lives in small, illiquid stocks",
            f"Sub-universe Sharpe {float((extras or {}).get('sub_sharpe') or 0):.2f} is below the bar.",
            "neutralize")
    h1, h2 = halves
    if (h1 > 0.3 and h2 < 0) or (h2 > 0.3 and h1 < 0):
        add("regime", "warn", "Works in only one half of the history",
            f"Sharpe {h1:.2f} in the first half of IS vs {h2:.2f} in the second.", "horizon")
    stab = (extras or {}).get("stability")
    if stab is not None and stab < 0.7 and sh > 0:
        add("fragile", "warn", "Fragile to the exact window lengths",
            f"Only {stab:.0%} of the Sharpe survives ±25% window changes.", "horizon")
    if dd > float(cfg.get("local_gates", {}).get("drawdown_max", 0.5)):
        add("drawdown", "warn", "Deep drawdowns", f"Max drawdown {dd:.0%} of half the book.", "neutralize")
    for fid, c in sorted(exposures.items(), key=lambda kv: -abs(kv[1])):
        if abs(c) >= 0.4:
            f = next((x for x in STYLE_FACTORS if x.id == fid), None)
            if f:
                add(f"exposure_{fid}", "warn" if abs(c) < 0.7 else "info",
                    f"Largely a {f.title.lower()} bet",
                    f"PnL correlation {c:+.2f} with a plain {f.title.lower()} factor. Unless that is the idea, "
                    "regressing it out isolates what is new.", "neutralize")
    if m_os and sh > 0.5:
        os_sh = float(m_os.get("sharpe") or 0.0)
        if os_sh < 0.5 * sh:
            add("os_decay", "info", "Weaker out of sample",
                f"OS Sharpe {os_sh:.2f} vs IS {sh:.2f}. Re-engineering searches on IS only, so the holdout stays "
                "an honest test.", "horizon")
    if not items:
        add("healthy", "info", "No obvious defect",
            "The search will look for incremental gains in shape, horizon and turnover.", "shape")
    return items


def recipe_text(lineage: list[dict]) -> str:
    return " → ".join(step["label"] for step in lineage) if lineage else "no changes"


__all__ = ["STAGES", "PASS_STAGES", "STYLE_FACTORS", "COMPANIONS", "Move", "MoveContext", "Partner", "objective",
           "meets_target", "propose", "blend_moves", "blend_candidates", "settings_cells", "diagnose",
           "half_sharpes", "pnl_corr", "sharpe_of", "recipe_text", "peel"]
