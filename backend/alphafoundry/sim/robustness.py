"""Robustness probes: window perturbation, settings sensitivity grid."""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np

from ..fastexpr.ast import Node, Op, iter_paths, replace_at

WINDOW_GRID = (2, 3, 5, 10, 15, 20, 21, 40, 60, 63, 90, 120, 126, 180, 250, 252, 500, 504)


def snap_window(w: float) -> int:
    w = max(1.0, w)
    return int(min(WINDOW_GRID, key=lambda g: abs(g - w) / g))


def window_sites(node: Node) -> list[tuple[tuple[int, ...], str, int]]:
    """(path, param name, value) for every window-like parameter in the tree."""
    sites = []
    for path, n in iter_paths(node):
        if isinstance(n, Op):
            for k, v in n.params:
                if k in ("d", "lookback") and isinstance(v, int) and v >= 2:
                    sites.append((path, k, v))
    return sites


def perturbed_variants(node: Node, factors=(0.75, 1.25), max_variants: int = 6) -> list[Node]:
    """Scale every window by each factor (all windows together), snapped to the standard grid."""
    sites = window_sites(node)
    if not sites:
        return []
    out: list[Node] = []
    for f in factors:
        cur = node
        changed = False
        for path, k, v in sites:
            target = snap_window(v * f)
            if target == v:
                target = max(2, int(round(v * f))) if int(round(v * f)) != v else v + (1 if f > 1 else -1)
                target = max(2, target)
            from ..fastexpr.ast import node_at
            sub = node_at(cur, path)
            assert isinstance(sub, Op)
            cur = replace_at(cur, path, sub.with_param(k, int(target)))
            changed = True
        if changed and cur.key != node.key:
            out.append(cur)
    return out[:max_variants]


def stability_score(base_sharpe: float, variant_sharpes: list[float]) -> float | None:
    if not variant_sharpes or base_sharpe <= 0:
        return None
    return float(np.clip(np.mean(variant_sharpes) / base_sharpe, -1.0, 1.5))


DEFAULT_SWEEP = {
    "decay": [0, 2, 4, 6, 8, 10, 15, 20],
    "neutralization": ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"],
    "truncation": [0.01, 0.05, 0.08, 0.1],
}


def sweep_grid(grid: dict | None = None) -> list[dict]:
    g = {**DEFAULT_SWEEP, **(grid or {})}
    keys = list(g)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(g[k] for k in keys))]


def robust_pick(cells: list[dict], score: Callable[[dict], float]) -> dict | None:
    """Choose the cell with the best neighbourhood-average score (robust optimum, not the peak)."""
    if not cells:
        return None
    by_key = {(c["decay"], c["neutralization"], c["truncation"]): c for c in cells}
    decays = sorted({c["decay"] for c in cells})
    truncs = sorted({c["truncation"] for c in cells})

    def neighbours(c):
        di = decays.index(c["decay"])
        ti = truncs.index(c["truncation"])
        for dd in (-1, 0, 1):
            for dt in (-1, 0, 1):
                a, b = di + dd, ti + dt
                if 0 <= a < len(decays) and 0 <= b < len(truncs):
                    n = by_key.get((decays[a], c["neutralization"], truncs[b]))
                    if n is not None:
                        yield n

    best, best_s = None, -np.inf
    for c in cells:
        own = score(c)
        nb = [score(n) for n in neighbours(c)]
        s = 0.6 * own + 0.4 * (float(np.mean(nb)) if nb else own)
        if s > best_s:
            best, best_s = c, s
    return best
