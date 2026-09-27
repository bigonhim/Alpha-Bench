"""Complex alphas: readable multi-statement compositions of decorrelated component signals.

BRAIN accepts programs such as ``value = ...; quality = ...; combo = 0.6 * value + 0.4 * quality; combo``.
Two to four signals from different families, each normalized, usually give a higher and steadier Sharpe
than any single part, because their errors partly cancel. The structures here are the ones consultants
use most: weighted blends, blends traded on volume events, tilts (one signal scaled by another), regime
switches (a volatility split chooses the signal) and multi-horizon ensembles of a single idea. Every
program is validated by the Fast Expression analyzer before it is returned.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import numpy as np

from ..catalog import field_map, operator_map
from ..fastexpr import analyze, lower_text, to_expr
from ..fastexpr.ast import Node, Op, node_at, replace_at
from ..sim.robustness import snap_window, window_sites

NORMALIZED_ROOTS = ("rank", "group_rank", "zscore", "group_zscore", "quantile")
VAR_NAMES = {"reversion": "reversal", "momentum": "momentum", "seasonality": "seasonal", "value": "value",
             "quality": "quality", "growth": "growth", "accruals": "accruals", "investment": "investment",
             "leverage": "leverage", "liquidity": "liquidity", "pv_divergence": "pv_diverge",
             "volatility": "low_risk", "sentiment": "sentiment", "options": "options"}
FAST_FAMILIES = {"reversion", "pv_divergence", "liquidity", "sentiment", "options"}


@dataclass
class Component:
    expr: str
    family: str = "other"
    settings: dict | None = None
    sharpe: float = 0.0
    label: str = ""
    alpha_id: int | None = None
    pnl: np.ndarray | None = None
    name: str = field(default="")


@dataclass
class Composite:
    label: str
    text: str
    structure: str
    parts: list[Component]
    weights: list[float]


def _reserved() -> set[str]:
    return set(field_map()) | set(operator_map()) | {"market", "sector", "industry", "subindustry", "exchange",
                                                     "country", "true", "false", "nan"}


def var_name(family: str, used: set[str]) -> str:
    base = VAR_NAMES.get(family, re.sub(r"[^a-z0-9_]", "_", (family or "signal").lower()) or "signal")
    reserved = _reserved()
    name, k = base, 2
    while name in used or name in reserved:
        name, k = f"{base}{k}", k + 1
    used.add(name)
    return name


def single_line(expr: str) -> str:
    """A component as one expression; a program is inlined into a single expression first."""
    e = expr.strip().rstrip(";").strip()
    if ";" not in e:
        return e
    try:
        return to_expr(lower_text(e))
    except Exception:  # noqa: BLE001
        return e


def normalized_expr(c: Component) -> str:
    """Component expression, with its own decay folded in and a cross-sectional normalization on top."""
    e = single_line(c.expr)
    decay = int((c.settings or {}).get("decay", 0) or 0)
    if decay > 1:
        e = f"ts_decay_linear({e}, {decay})"
    try:
        root = lower_text(e)
    except Exception:  # noqa: BLE001
        return f"rank({e})"
    if isinstance(root, Op) and root.name in NORMALIZED_ROOTS:
        return e
    return f"rank({e})"


def weights_for(pnls: np.ndarray | None, k: int, shrink: float = 0.5) -> list[float]:
    """Mean-variance weights on component PnL (shrunk covariance, long-only), rounded to 0.05.

    Falls back to equal weights without PnL. Every kept component gets at least 0.1 so the composite
    still reads as the blend it claims to be."""
    if pnls is None or pnls.shape[0] != k or pnls.shape[1] < 60:
        return [round(1.0 / k, 2)] * k
    x = np.nan_to_num(np.asarray(pnls, dtype=np.float64))
    mu = x.mean(axis=1)
    cov = np.cov(x) if k > 1 else np.array([[x.var()]])
    cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
    try:
        w = np.linalg.solve(cov + 1e-12 * np.eye(k), mu)
    except np.linalg.LinAlgError:
        w = np.ones(k)
    # PnL of each component is in dollars of its own book; scale by its volatility so weights act on the
    # normalized signals rather than on PnL units
    w = np.clip(w * np.sqrt(np.diag(cov)), 0.0, None)
    if w.sum() <= 0:
        w = np.ones(k)
    w = w / w.sum()
    w = np.maximum(w, 0.1)
    w = np.round(w / w.sum() / 0.05) * 0.05
    w[np.argmax(w)] += round(1.0 - float(w.sum()), 2)
    return [round(float(v), 2) for v in w]


def _fmt_w(w: float) -> str:
    return f"{w:.2f}".rstrip("0").rstrip(".") if w != int(w) else str(int(w))


def program(lines: list[tuple[str, str]], final: str) -> str:
    return "\n".join(f"{n} = {e};" for n, e in lines) + "\n" + final


def blend(parts: list[Component], weights: list[float], post: str | None = None, label: str = "") -> Composite:
    used: set[str] = set()
    lines = []
    for c in parts:
        c.name = var_name(c.family, used)
        lines.append((c.name, normalized_expr(c)))
    terms = " + ".join(f"{_fmt_w(w)} * {c.name}" for c, w in zip(parts, weights))
    if post:
        lines.append(("combo", terms))
        final = post.format(x="combo")
    else:
        final = terms
    names = ", ".join(f"{c.family} {_fmt_w(w)}" for c, w in zip(parts, weights))
    return Composite(label or f"Blend of {names}", program(lines, final), "blend" if not post else "blend+post",
                     parts, weights)


def tilt(main: Component, cond: Component) -> Composite:
    """The main signal, scaled up where the conditioning signal is strong (0.5x ... 1.5x)."""
    used: set[str] = set()
    main.name = var_name(main.family, used)
    cond.name = var_name(cond.family, used)
    ce = normalized_expr(cond)
    if not ce.startswith(("rank(", "group_rank(")):
        ce = f"rank({ce})"
    text = program([(main.name, normalized_expr(main)), (cond.name, ce)], f"{main.name} * (0.5 + {cond.name})")
    return Composite(f"{main.family} tilted toward high {cond.family}", text, "tilt", [main, cond], [1.0, 0.0])


def regime(a: Component, b: Component, window: int = 20) -> Composite:
    """Volatile stocks trade signal a, calm stocks signal b (a rank split, rebalanced daily)."""
    used: set[str] = {"vol_rank"}
    a.name = var_name(a.family, used)
    b.name = var_name(b.family, used)
    text = program([(a.name, normalized_expr(a)), (b.name, normalized_expr(b)),
                    ("vol_rank", f"rank(ts_std_dev(returns, {window}))")],
                   f"vol_rank > 0.5 ? {a.name} : {b.name}")
    return Composite(f"{a.family} on volatile stocks, {b.family} on calm stocks", text, "regime", [a, b], [0.5, 0.5])


def orthogonal(main: Component, other: Component) -> Composite:
    """The main signal with its linear exposure to the other removed (cuts correlation with that style)."""
    used: set[str] = set()
    main.name = var_name(main.family, used)
    other.name = var_name(other.family, used)
    text = program([(main.name, normalized_expr(main)), (other.name, normalized_expr(other))],
                   f"regression_neut({main.name}, {other.name})")
    return Composite(f"{main.family} orthogonalized to {other.family}", text, "orthogonal", [main, other], [1.0, 0.0])


def horizon_sites(node: Node) -> list[tuple[tuple[int, ...], str, int]]:
    """Window parameters that set the signal's horizon. Back-fill lookbacks only fill gaps and delays are
    skips or seasonal lags, so neither is scaled."""
    return [(p, k, v) for p, k, v in window_sites(node)
            if k != "lookback" and getattr(node_at(node, p), "name", "") not in ("ts_backfill", "ts_delay")]


def scale_windows(node: Node, factor: float) -> Node:
    cur = node
    for path, key, v in horizon_sites(node):
        sub = node_at(cur, path)
        if isinstance(sub, Op):
            cur = replace_at(cur, path, sub.with_param(key, max(2, snap_window(v * factor))))
    return cur


def horizon_ensemble(c: Component) -> Composite | None:
    """One idea at a short, the original and a long horizon, averaged: robust to the window choice."""
    try:
        node = lower_text(single_line(c.expr))
    except Exception:  # noqa: BLE001
        return None
    if not horizon_sites(node):
        return None
    short, long_ = scale_windows(node, 0.5), scale_windows(node, 2.0)
    if short.key == node.key or long_.key == node.key or short.key == long_.key:
        return None
    parts = [("h_short", to_expr(short)), ("h_mid", to_expr(node)), ("h_long", to_expr(long_))]
    lines = [(n, e if e.startswith(("rank(", "group_rank(")) else f"rank({e})") for n, e in parts]
    text = program(lines, "(h_short + h_mid + h_long) / 3")
    return Composite(f"{c.family} averaged over short, original and long horizons", text, "horizons", [c],
                     [1.0])


def diverse_sets(cands: list[Component], corr: np.ndarray | None, k_max: int = 3, max_corr: float = 0.5,
                 limit: int = 12) -> list[list[int]]:
    """Greedy groups of components that are weakly correlated with each other and ideally of different families."""
    n = len(cands)
    order = sorted(range(n), key=lambda i: -cands[i].sharpe)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for start in order:
        group = [start]
        for j in order:
            if j in group or len(group) >= k_max:
                continue
            if corr is not None and max(abs(float(corr[j, g])) for g in group) >= max_corr:
                continue
            if cands[j].family in {cands[g].family for g in group} and len(order) > k_max:
                continue
            group.append(j)
        for size in range(2, len(group) + 1):
            key = tuple(sorted(group[:size]))
            if key not in seen:
                seen.add(key)
                out.append(list(key))
        if len(out) >= limit:
            break
    if not out and n >= 2:
        out = [list(p) for p in itertools.combinations(order[:4], 2)][:limit]
    return out[:limit]


def composites(cands: list[Component], corr: np.ndarray | None = None, *, local_fields: set[str] | None = None,
               has_volume: bool = True, k_max: int = 3, limit: int = 24) -> list[Composite]:
    """Valid composite programs from the components, the most promising structures first."""
    out: list[Composite] = []
    for group in diverse_sets(cands, corr, k_max=k_max):
        parts = [cands[i] for i in group]
        P = None
        if all(c.pnl is not None for c in parts):
            L = min(len(c.pnl) for c in parts)  # type: ignore[arg-type]
            P = np.vstack([np.asarray(c.pnl[-L:], dtype=np.float64) for c in parts])  # type: ignore[index]
        w = weights_for(P, len(parts))
        fresh = [Component(c.expr, c.family, c.settings, c.sharpe, c.label, c.alpha_id, c.pnl) for c in parts]
        out.append(blend(fresh, w))
        eq = [round(1.0 / len(parts), 2)] * len(parts)
        if any(abs(a - b) > 0.06 for a, b in zip(w, eq)):
            out.append(blend([Component(c.expr, c.family, c.settings, c.sharpe) for c in parts], eq,
                             label="Equal-weight blend of " + ", ".join(c.family for c in parts)))
        fams = {c.family for c in parts}
        if has_volume and fams & FAST_FAMILIES:
            out.append(blend([Component(c.expr, c.family, c.settings, c.sharpe) for c in parts], w,
                             post="trade_when(volume > adv20, {x}, -1)",
                             label="Blend traded on abnormal-volume days"))
        if len(parts) == 2:
            a, b = parts
            out.append(tilt(Component(a.expr, a.family, a.settings, a.sharpe),
                            Component(b.expr, b.family, b.settings, b.sharpe)))
            fast = a if a.family in FAST_FAMILIES else b if b.family in FAST_FAMILIES else None
            if fast is not None:
                slow = b if fast is a else a
                out.append(regime(Component(fast.expr, fast.family, fast.settings, fast.sharpe),
                                  Component(slow.expr, slow.family, slow.settings, slow.sharpe)))
            out.append(orthogonal(Component(a.expr, a.family, a.settings, a.sharpe),
                                  Component(b.expr, b.family, b.settings, b.sharpe)))
    for c in sorted(cands, key=lambda c: -c.sharpe)[:3]:
        h = horizon_ensemble(c)
        if h is not None:
            out.append(h)
    valid, seen = [], set()
    for comp in out:
        an = analyze(comp.text, local_fields=local_fields)
        if not an.ok or (local_fields is not None and not an.local) or an.canon_hash in seen:
            continue
        seen.add(an.canon_hash)
        valid.append(comp)
        if len(valid) >= limit:
            break
    return valid


def is_complex(expr: str) -> bool:
    """Multi-statement program (the user's notion of a 'complex', more-than-one-line alpha)."""
    return ";" in expr.strip().rstrip(";")


__all__ = ["Component", "Composite", "blend", "composites", "diverse_sets", "horizon_ensemble", "is_complex",
           "normalized_expr", "orthogonal", "regime", "tilt", "var_name", "weights_for"]
