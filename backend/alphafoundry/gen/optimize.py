"""Fitness shaping: rewrites that raise fitness and settle turnover without changing what the alpha says.

BRAIN fitness is Sharpe x sqrt(|returns| / max(turnover, 0.125)). At a given Sharpe it rises when turnover
falls toward 12.5% (decay, smoothing, event gates, hump) or when returns per unit of book rise (a
sharper weight profile). Peer-relative ranking and a different neutralization often lift the Sharpe
itself. ``shaping_variants`` proposes these rewrites for one alpha; the caller simulates them and
``pick_best`` keeps the best one that retains most of the Sharpe and introduces no failing check.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..fastexpr.ast import Const, Field, Node, Op
from ..fastexpr.lower import simplify
from ..sim.quality import quick_score
from .build import capbucket, fields_of, mk

NORMALIZED = {"rank", "zscore", "group_rank", "group_zscore", "quantile", "scale", "normalize", "winsorize",
              "group_neutralize"}
SMOOTHING = {"ts_decay_linear", "ts_mean", "hump", "ts_target_tvr_decay", "trade_when", "ts_decay_exp_window"}
NEUTS = ("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")


@dataclass
class Shape:
    label: str
    node: Node
    settings: dict
    kind: str  # turnover | returns | peers | settings


def _root(n: Node) -> str | None:
    return n.name if isinstance(n, Op) else None


def _is_smoothed(n: Node) -> bool:
    return any(isinstance(x, Op) and x.name in SMOOTHING for x in n.walk())


def shaping_variants(node: Node, settings: dict, m: dict | None, *, has_volume: bool = True,
                     limit: int = 20) -> list[Shape]:
    """Candidate rewrites for one alpha, most promising first (at most ``limit``)."""
    m = m or {}
    s = dict(settings)
    to = float(m.get("turnover") or 0.0)
    decay = int(s.get("decay", 0) or 0)
    neut = str(s.get("neutralization", "SUBINDUSTRY")).upper()
    root = _root(node)
    out: list[Shape] = []

    def add(label: str, n: Node | None, kind: str, **sets) -> None:
        out.append(Shape(label, simplify(n) if n is not None else node, {**s, **sets}, kind))

    # --- turnover: aim for 10-25% (fitness stops improving below 12.5%)
    if to > 0.35:
        for d in (max(6, decay + 4), max(10, decay + 8), 16):
            if d != decay:
                add(f"Decay {d}", None, "turnover", decay=d)
        add("Smooth: ts_decay_linear(x, 10)", mk("ts_decay_linear", node, d=10), "turnover")
        add("Smooth: ts_decay_linear(x, 5)", mk("ts_decay_linear", node, d=5), "turnover")
        if has_volume and "trade_when" not in {x.name for x in node.walk() if isinstance(x, Op)}:
            add("Trade only on abnormal volume", mk("trade_when", mk("greater", Field("volume"), Field("adv20")),
                                                   node, Const(-1.0)), "turnover")
        add("Hump 0.05 (skip small changes)", mk("hump", node, hump=0.05), "turnover")
        add("Target 20% turnover", mk("ts_target_tvr_decay", node, target_tvr=0.2), "turnover")
    elif to > 0.15:
        for d in (max(3, decay + 2), max(6, decay + 4)):
            if d != decay:
                add(f"Decay {d}", None, "turnover", decay=d)
        add("Smooth: ts_decay_linear(x, 3)", mk("ts_decay_linear", node, d=3), "turnover")
        add("Hump 0.02", mk("hump", node, hump=0.02), "turnover")
    elif to < 0.03 and decay > 0:
        add("Decay 0", None, "turnover", decay=0)

    # --- peers and neutralization: often lift the Sharpe itself
    for n in NEUTS:
        if n != neut:
            add(f"Neutralize by {n.lower()}", None, "settings", neutralization=n)
    if root == "rank" and isinstance(node, Op):
        inner = node.args[0]
        add("Rank within sub-industry", mk("group_rank", inner, Field("subindustry")), "peers")
        add("Rank within industry", mk("group_rank", inner, Field("industry")), "peers")
    elif root not in NORMALIZED:
        add("Rank within industry", mk("group_rank", node, Field("industry")), "peers")
    if "bucket" not in {x.name for x in node.walk() if isinstance(x, Op)} and "cap" not in fields_of(node):
        add("Size-bucket neutral", mk("group_neutralize", node, capbucket()), "peers")

    # --- returns: a sharper weight profile at the same ordering
    if root == "rank" and isinstance(node, Op):
        add("Z-score instead of rank (winsorized)", mk("winsorize", mk("zscore", node.args[0]), std=3.0), "returns")
        add("Emphasize the tails", mk("signed_power", mk("subtract", node, Const(0.5)), Const(2.0)), "returns")
    elif root == "group_rank" and isinstance(node, Op):
        add("Group z-score instead of group rank",
            mk("winsorize", mk("group_zscore", node.args[0], node.args[1]), std=3.0), "returns")
    elif root not in NORMALIZED:
        add("Rank the signal", mk("rank", node), "returns")
        add("Winsorized z-score", mk("winsorize", mk("zscore", node), std=3.0), "returns")
    if float(m.get("max_weight") or 0) > 0.07:
        add("Truncation 0.05", None, "settings", truncation=0.05)

    seen, uniq = set(), []
    for sh in out:
        k = (sh.node.key, tuple(sorted((a, str(b)) for a, b in sh.settings.items())))
        if k in seen or (sh.node.key == node.key and sh.settings == settings):
            continue
        seen.add(k)
        uniq.append(sh)
    # interleave kinds so a small limit still covers turnover, peers, returns and settings
    order = {"turnover": 0, "peers": 1, "returns": 2, "settings": 3}
    buckets: dict[int, list[Shape]] = {}
    for sh in uniq:
        buckets.setdefault(order.get(sh.kind, 3), []).append(sh)
    mixed: list[Shape] = []
    while any(buckets.values()):
        for k in sorted(buckets):
            if buckets[k]:
                mixed.append(buckets[k].pop(0))
    return mixed[:limit]


def acceptable(m: dict, base: dict, checks_cfg: dict, keep: float = 0.9) -> bool:
    """No new failing limit, and the Sharpe mostly kept (a shaper must not trade Sharpe for fitness)."""
    if not m:
        return False
    to = float(m.get("turnover") or 0.0)
    if not checks_cfg["turnover_min"] <= to <= checks_cfg["turnover_max"]:
        return False
    if float(m.get("max_weight") or 0.0) > checks_cfg["max_weight"] + 1e-9:
        return False
    bs = float(base.get("sharpe") or 0.0)
    return float(m.get("sharpe") or 0.0) >= keep * bs if bs > 0 else True


def pick_best(base_m: dict, tried: list[tuple[Shape, dict]], checks_cfg: dict, delay: int = 1,
              min_gain: float = 0.05) -> tuple[Shape, dict] | None:
    """The shaped variant with the best quick quality score, if it beats the base by ``min_gain``."""
    base = quick_score(base_m, checks_cfg, delay)
    best, best_s = None, base + min_gain
    for sh, m in tried:
        if not acceptable(m, base_m, checks_cfg):
            continue
        sc = quick_score(m, checks_cfg, delay)
        if sc > best_s:
            best, best_s = (sh, m), sc
    return best


__all__ = ["Shape", "acceptable", "pick_best", "shaping_variants"]
