"""Candidates on BRAIN catalog fields that have no local proxy (analyst, sentiment, options, model data...).

Fields come from a synced BRAIN catalog (``runtime/fields_user.json``). Less-used fields (low alpha count)
with good coverage are preferred: alphas built on crowded fields tend to fail BRAIN's self- and
production-correlation checks. Vector fields are reduced to a matrix first (``vec_avg``/``vec_sum``).
"""

from __future__ import annotations

import math
import random

from ..catalog import field_map
from ..fastexpr import analyze

MATRIX_PATTERNS = [
    ("peer rank", "group_rank({x}, subindustry)"),
    ("1-year z-score", "rank(ts_zscore({x}, 252))"),
    ("quarter change (reversed)", "-rank(ts_delta({x}, 63))"),
    ("quarter change", "rank(ts_delta({x}, 63))"),
    ("rank in own history", "group_rank(ts_rank({x}, 252), industry)"),
    ("deviation from 6-month mean", "group_rank(ts_av_diff({x}, 120), industry)"),
    ("scaled by size", "group_rank({x} / cap, subindustry)"),
]
VECTOR_REDUCERS = ("vec_avg", "vec_sum")


def field_score(f: dict) -> float:
    """Higher for well-covered, rarely used fields."""
    cov = f.get("coverage")
    try:
        cov = float(cov) if cov is not None else 0.6
    except (TypeError, ValueError):
        cov = 0.6
    ac = f.get("alpha_count")
    try:
        ac = float(ac) if ac is not None else 50.0
    except (TypeError, ValueError):
        ac = 50.0
    return min(cov, 1.0) * 2.0 - 0.35 * math.log10(1.0 + max(ac, 0.0))


def brain_fields(local: set[str], categories: set[str] | None = None, min_coverage: float = 0.5) -> list[dict]:
    out = []
    for fid, f in field_map().items():
        if fid in local or f.get("local"):
            continue
        t = str(f.get("type", "MATRIX")).upper()
        if t not in ("MATRIX", "VECTOR"):
            continue
        if categories and str(f.get("category", "")).lower() not in categories:
            continue
        try:
            if f.get("coverage") is not None and float(f["coverage"]) < min_coverage:
                continue
        except (TypeError, ValueError):
            pass
        out.append({**f, "id": fid})
    out.sort(key=lambda f: -field_score(f))
    return out


def field_candidates(local: set[str], n: int = 40, rng: random.Random | None = None,
                     categories: set[str] | None = None, exclude: set[str] | None = None) -> list[tuple[str, str, dict]]:
    """(expression, label, field) for the most promising non-local catalog fields."""
    rng = rng or random.Random(0)
    fields = brain_fields(local, categories)
    out: list[tuple[str, str, dict]] = []
    seen = set(exclude or ())
    pats = list(MATRIX_PATTERNS)
    for k, f in enumerate(fields):
        if len(out) >= n:
            break
        x = f["id"]
        if str(f.get("type", "MATRIX")).upper() == "VECTOR":
            x = f"{rng.choice(VECTOR_REDUCERS)}({x})"
        x = f"ts_backfill({x}, 60)"
        rng.shuffle(pats)
        for label, p in pats[:2 if k < n // 2 else 1]:
            e = p.format(x=x)
            if e in seen:
                continue
            if analyze(e).ok:
                seen.add(e)
                out.append((e, f"{f['id']}: {label}", f))
    return out[:n]


__all__ = ["brain_fields", "field_candidates", "field_score"]
