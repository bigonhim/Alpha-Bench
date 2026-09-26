"""Rule-based Doctor: turn failing checks into concrete, testable rewrites."""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import field_map
from ..fastexpr.ast import Const, Field, Node, Op, iter_paths, replace_at
from ..fastexpr.lower import simplify
from ..fastexpr.printer import to_expr
from ..sim.robustness import snap_window
from .build import GROUP_NAMES, capbucket, mk
from .grammar import FAMILY_OF

NORMALIZED_ROOTS = {"rank", "zscore", "quantile", "group_rank", "group_zscore", "group_neutralize", "scale",
                    "winsorize", "normalize"}


@dataclass
class Fix:
    label: str
    reason: str
    node: Node
    settings: dict
    kind: str  # expression | settings

    def to_json(self) -> dict:
        return {"label": self.label, "reason": self.reason, "expr": to_expr(self.node), "settings": self.settings,
                "kind": self.kind}


def _scale_windows(n: Node, factor: float, only_below: int | None = None) -> Node:
    out = n
    for path, sub in list(iter_paths(n)):
        if isinstance(sub, Op):
            d = sub.param("d")
            if isinstance(d, int) and d >= 2 and (only_below is None or d < only_below):
                cur = out
                from ..fastexpr.ast import node_at
                s2 = node_at(cur, path)
                assert isinstance(s2, Op)
                out = replace_at(cur, path, s2.with_param("d", max(2, snap_window(d * factor))))
    return out


def _round_windows(n: Node) -> Node:
    return _scale_windows(n, 1.0)


def _backfill_fundamentals(n: Node) -> Node:
    fm = field_map()
    out = n
    for path, sub in list(iter_paths(n)):
        if isinstance(sub, Field) and fm.get(sub.name, {}).get("category") == "fundamental":
            parent_is_backfill = False
            if path:
                from ..fastexpr.ast import node_at
                par = node_at(out, path[:-1])
                parent_is_backfill = isinstance(par, Op) and par.name == "ts_backfill"
            if not parent_is_backfill:
                out = replace_at(out, path, mk("ts_backfill", sub, lookback=120))
    return out


def diagnose(node: Node, settings: dict, failed: list[str], metrics: dict | None = None,
             warnings: list[str] | None = None, correlated_node: Node | None = None,
             has_volume: bool = True) -> list[Fix]:
    m = metrics or {}
    warnings = warnings or []
    s = dict(settings)
    fixes: list[Fix] = []
    root_name = node.name if isinstance(node, Op) else None
    sharpe = float(m.get("sharpe", 0.0) or 0.0)
    to = float(m.get("turnover", 0.0) or 0.0)

    def add(label: str, reason: str, n: Node | None = None, **sets):
        nn = simplify(n) if n is not None else node
        ss = {**s, **sets}
        fixes.append(Fix(label, reason, nn, ss, "settings" if n is None else "expression"))

    fset = set(failed)
    if "HIGH_TURNOVER" in fset or ("LOW_FITNESS" in fset and to > 0.125):
        cur = int(s.get("decay", 0) or 0)
        for d in sorted({max(4, cur * 2), max(8, cur + 6), 15}):
            if d != cur:
                add(f"Decay {d}", "Linear decay averages recent alpha values, cutting turnover.", decay=d)
        add("Smooth with ts_decay_linear(x, 5)", "Smooths the signal inside the expression.",
            mk("ts_decay_linear", node, d=5))
        add("Smooth with ts_mean(x, 5)", "Averages the last 5 days of the signal.", mk("ts_mean", node, d=5))
        if has_volume:
            add("Event gate: trade_when(volume > adv20, x, -1)",
                "Only rebalance on abnormal-volume days; hold positions otherwise.",
                mk("trade_when", mk("greater", Field("volume"), Field("adv20")), node, Const(-1.0)))
        add("Target turnover 25%", "ts_target_tvr_decay tunes smoothing to hit a turnover target.",
            mk("ts_target_tvr_decay", node, target_tvr=0.25))
        add("Longer lookbacks (x2)", "Longer windows change the signal more slowly.", _scale_windows(node, 2.0, 30))
    if "LOW_TURNOVER" in fset:
        if int(s.get("decay", 0) or 0) > 0:
            add("Decay 0", "Removing decay lets the signal react faster.", decay=0)
        add("Shorter lookbacks (x0.5)", "Shorter windows update the signal more often.", _scale_windows(node, 0.5))
        add("Use changes: ts_zscore(x, 20)", "Time-series z-score turns a static level into a moving signal.",
            mk("ts_zscore", node, d=20))
    if "LOW_SHARPE" in fset or "LOW_FITNESS" in fset:
        if sharpe < -0.3:
            add("Flip the sign", f"Sharpe is negative ({sharpe:.2f}); the reversed signal earns the opposite PnL.",
                mk("reverse", node))
        for neut in ("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"):
            if neut != str(s.get("neutralization", "")).upper():
                add(f"Neutralize by {neut.lower()}", "Different risk neutralization can remove noisy group bets.",
                    neutralization=neut)
        if root_name not in ("group_rank", "group_neutralize", "group_zscore"):
            add("Rank within sub-industry", "Compare stocks only with close peers.",
                mk("group_rank", node, Field("subindustry")))
        if root_name not in NORMALIZED_ROOTS:
            add("Normalize with rank()", "Ranking removes outliers that dominate PnL.", mk("rank", node))
        add("Volatility-scale", "Dividing by recent volatility equalizes risk across stocks.",
            mk("divide", node, mk("ts_std_dev", Field("returns"), d=20)))
        add("Winsorize at 4 sigma", "Clipping extremes stabilizes the signal.", mk("winsorize", node, std=4.0))
    if "CONCENTRATED_WEIGHT" in fset:
        if root_name not in ("rank", "group_rank"):
            add("Wrap in rank()", "Ranks are bounded, so no single stock can dominate.", mk("rank", node))
        add("Truncation 0.05", "Caps any single stock at 5% of the book.", truncation=0.05)
        add("Winsorize at 3 sigma", "Clips outliers before weighting.", mk("winsorize", node, std=3.0))
        bf = _backfill_fundamentals(node)
        if bf.key != node.key:
            add("Back-fill fundamentals", "ts_backfill fills gaps in quarterly data so more stocks get weight.", bf)
    if "LOW_SUB_UNIVERSE_SHARPE" in fset:
        add("Neutralize within size buckets", "Removes the small-cap tilt that fails the sub-universe test.",
            mk("group_neutralize", node, capbucket()))
        if root_name not in ("rank", "group_rank"):
            add("Rank before weighting", "Ranking reduces the influence of small, extreme names.", mk("rank", node))
        if str(s.get("neutralization", "")).upper() != "SUBINDUSTRY":
            add("Sub-industry neutralization", "Tighter neutralization reduces size/style exposure.",
                neutralization="SUBINDUSTRY")
    if "SELF_CORRELATION" in fset:
        for neut in ("SECTOR", "INDUSTRY", "SUBINDUSTRY", "MARKET"):
            if neut != str(s.get("neutralization", "")).upper():
                add(f"Neutralize by {neut.lower()}", "A different neutralization changes the PnL path.",
                    neutralization=neut)
                break
        add("Double the horizon", "Longer windows decorrelate from the original.", _scale_windows(node, 2.0))
        add("Halve the horizon", "Shorter windows decorrelate from the original.", _scale_windows(node, 0.5))
        for path, sub in iter_paths(node):
            if isinstance(sub, Op) and sub.name in FAMILY_OF:
                alt = [o for o in FAMILY_OF[sub.name] if o != sub.name][0]
                try:
                    add(f"Swap {sub.name} -> {alt}", "Changing the operator family changes the signal shape.",
                        replace_at(node, path, mk(alt, *sub.args, **dict(sub.params))))
                except (KeyError, TypeError):
                    pass
                break
        if correlated_node is not None:
            add("Orthogonalize to the correlated alpha", "vector_neut removes the component shared with it.",
                mk("vector_neut", node, correlated_node))
    if "OS_DEGRADATION" in warnings or "PARAMETER_STABILITY" in warnings:
        rw = _round_windows(node)
        if rw.key != node.key:
            add("Standard windows", "Rounding windows to standard values reduces overfitting.", rw)
        if isinstance(node, Op):
            subs = [sub for p, sub in iter_paths(node) if p and isinstance(sub, Op) and sub.size >= 3]
            if subs:
                add("Simplify (hoist core)", "A simpler expression generalizes better out of sample.",
                    mk("rank", max(subs, key=lambda x: x.size)))
    # de-duplicate
    seen, out = set(), []
    for f in fixes:
        k = (f.node.key, tuple(sorted((k, str(v)) for k, v in f.settings.items())))
        if k in seen or (f.node.key == node.key and f.settings == settings):
            continue
        seen.add(k)
        out.append(f)
    return out[:16]


__all__ = ["Fix", "diagnose", "GROUP_NAMES"]
