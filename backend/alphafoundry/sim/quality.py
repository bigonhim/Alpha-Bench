"""Alpha quality: BRAIN checks with safety margins, robustness evidence, a continuous score and a grade.

Local metrics are a proxy. The free data, the universe and the in-sample window all differ from BRAIN, so
an alpha that clears BRAIN's thresholds by a hair locally usually fails there. The grade therefore asks for
a margin above every threshold and for evidence that the edge is real (holdout, recent window,
sub-universe, window perturbation, yearly consistency):

    A  BRAIN-ready: every hard check passes with margin and the robustness evidence holds
    B  passes every hard check locally, but thin margins or one robustness warning
    C  near miss: one hard check fails narrowly (Doctor / optimizer material)
    D  everything else
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_QUALITY: dict[str, Any] = {
    "sharpe_margin": 1.25,          # local IS Sharpe >= margin x BRAIN minimum (1.25 -> 1.56 at delay 1)
    "fitness_margin": 1.25,         # local IS fitness >= margin x BRAIN minimum
    "turnover_band": [0.03, 0.45],  # comfortable band inside BRAIN's 1%..70% limits
    "brain_window_margin": 1.0,     # Sharpe over the BRAIN-like (latest 5y IS) window >= margin x minimum
    "os_sharpe_min": 0.6,           # holdout Sharpe must stay clearly positive ...
    "os_ratio_min": 0.45,           # ... and keep this share of the IS Sharpe
    "stability_min": 0.7,           # Sharpe kept when every window moves +/-25%
    "positive_years_min": 0.7,      # share of profitable IS years
    "max_complexity": 40,           # operator + field nodes; beyond this the alpha is likely overfit
    "near_miss": 0.8,               # grade C: failing value within this share of its limit
}

HARD = ("LOW_SHARPE", "LOW_FITNESS", "LOW_TURNOVER", "HIGH_TURNOVER", "CONCENTRATED_WEIGHT",
        "LOW_SUB_UNIVERSE_SHARPE", "SELF_CORRELATION")
GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


def quality_config(checks_cfg: dict) -> dict:
    q = dict(DEFAULT_QUALITY)
    q.update(checks_cfg.get("quality") or {})
    return q


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def thresholds(checks_cfg: dict, delay: int) -> tuple[float, float]:
    th = checks_cfg["delay0" if int(delay) == 0 else "delay1"]
    return float(th["sharpe_min"]), float(th["fitness_min"])


def quick_score(m: dict | None, checks_cfg: dict, delay: int = 1) -> float:
    """Rank candidates from in-sample metrics alone (screening stage, before any extras exist)."""
    if not m:
        return -99.0
    q = quality_config(checks_cfg)
    sh_req, fit_req = thresholds(checks_cfg, delay)
    sh, fit, to = _num(m.get("sharpe")), _num(m.get("fitness")), _num(m.get("turnover"))
    s = 1.2 * max(-2.0, min(sh / sh_req, 2.2)) + 1.0 * max(-2.0, min(fit / fit_req, 2.2))
    lo, hi = q["turnover_band"]
    if to < checks_cfg["turnover_min"] or to > checks_cfg["turnover_max"]:
        s -= 1.5
    elif to < lo:
        s -= 0.4 * (lo - to) / lo
    elif to > hi:
        s -= 1.5 * (to - hi)
    if _num(m.get("max_weight")) > checks_cfg["max_weight"]:
        s -= 1.0
    return round(s, 4)


def quick_grade(m: dict | None, checks_cfg: dict, delay: int = 1) -> str:
    """Optimistic grade from IS metrics only: which candidates deserve a full evaluation."""
    if not m:
        return "D"
    q = quality_config(checks_cfg)
    sh_req, fit_req = thresholds(checks_cfg, delay)
    sh, fit, to = _num(m.get("sharpe")), _num(m.get("fitness")), _num(m.get("turnover"))
    to_ok = checks_cfg["turnover_min"] <= to <= checks_cfg["turnover_max"]
    w_ok = _num(m.get("max_weight")) <= checks_cfg["max_weight"] + 1e-9
    if to_ok and w_ok and sh >= sh_req * q["sharpe_margin"] and fit >= fit_req * q["fitness_margin"]:
        return "A"
    if to_ok and w_ok and sh >= sh_req and fit >= fit_req:
        return "B"
    nm = q["near_miss"]
    if sh >= sh_req * nm and fit >= fit_req * nm * 0.9:
        return "C"
    return "D"


def assess(metrics: dict, checks: dict, checks_cfg: dict, *, delay: int = 1, extras: dict | None = None,
           yearly: list | None = None, complexity: int | None = None,
           expected_brain_sharpe: float | None = None) -> dict:
    """Grade and score a fully evaluated alpha (IS + OS metrics, extras, checks already computed)."""
    q = quality_config(checks_cfg)
    sh_req, fit_req = thresholds(checks_cfg, delay)
    m = metrics.get("is") or {}
    osm = metrics.get("os") or {}
    bw = metrics.get("brain") or {}
    ex = extras or {}
    sh, fit, to = _num(m.get("sharpe")), _num(m.get("fitness")), _num(m.get("turnover"))
    reasons: list[str] = []
    evidence: list[str] = []

    failed = [c for c in checks.get("checks", []) if c["name"] in HARD and c["result"] == "FAIL"]
    pending = [c for c in checks.get("checks", []) if c["name"] in HARD and c["result"] == "PENDING"]

    margin_ok = True
    if sh < sh_req * q["sharpe_margin"]:
        margin_ok = False
        reasons.append(f"Sharpe {sh:.2f} below the safety target {sh_req * q['sharpe_margin']:.2f}")
    if fit < fit_req * q["fitness_margin"]:
        margin_ok = False
        reasons.append(f"Fitness {fit:.2f} below the safety target {fit_req * q['fitness_margin']:.2f}")
    lo, hi = q["turnover_band"]
    if not lo <= to <= hi:
        margin_ok = False
        reasons.append(f"Turnover {to * 100:.1f}% outside the comfortable {lo * 100:.0f}-{hi * 100:.0f}% band")

    issues_before = len(reasons)
    bw_sh = bw.get("sharpe")
    if bw_sh is not None:
        if _num(bw_sh) < sh_req * q["brain_window_margin"]:
            reasons.append(f"Sharpe over the BRAIN-like latest-5y window is only {_num(bw_sh):.2f}")
        else:
            evidence.append(f"latest-5y Sharpe {_num(bw_sh):.2f}")
    os_sh = osm.get("sharpe")
    os_ratio = None
    if os_sh is not None and sh > 0:
        os_ratio = _num(os_sh) / sh
        if _num(os_sh) < q["os_sharpe_min"] or os_ratio < q["os_ratio_min"]:
            reasons.append(f"Holdout Sharpe {_num(os_sh):.2f} ({os_ratio * 100:.0f}% of IS)")
        else:
            evidence.append(f"holdout Sharpe {_num(os_sh):.2f}")
    stab = ex.get("stability")
    if stab is not None:
        if _num(stab) < q["stability_min"]:
            reasons.append(f"Only {_num(stab) * 100:.0f}% of the Sharpe survives +/-25% window changes")
        else:
            evidence.append(f"window stability {_num(stab) * 100:.0f}%")
    pos_share = None
    if yearly:
        is_years = [y for y in yearly if y.get("period") == "IS"]
        if is_years:
            pos_share = sum(1 for y in is_years if _num(y.get("pnl")) > 0) / len(is_years)
            if pos_share < q["positive_years_min"]:
                reasons.append(f"Only {pos_share * 100:.0f}% of IS years profitable")
    sub_ratio = None
    sub_chk = next((c for c in checks.get("checks", []) if c["name"] == "LOW_SUB_UNIVERSE_SHARPE"), None)
    if sub_chk and sub_chk.get("limit") and _num(sub_chk.get("limit")) > 0:
        sub_ratio = _num(sub_chk.get("value")) / _num(sub_chk.get("limit"))
    if complexity is not None and complexity > q["max_complexity"]:
        reasons.append(f"Complexity {complexity} nodes (limit {q['max_complexity']}) invites overfitting")
    if expected_brain_sharpe is not None and expected_brain_sharpe < sh_req:
        reasons.append(f"Calibrated BRAIN Sharpe estimate {expected_brain_sharpe:.2f} is below {sh_req}")
    robust_issues = len(reasons) - issues_before

    n_fail = len(failed)
    clean = n_fail == 0 and not pending
    if clean and margin_ok and robust_issues == 0:
        grade = "A"
    elif clean and (robust_issues == 0 or (margin_ok and robust_issues == 1)):
        grade = "B"
    elif clean:
        grade = "C"  # passes locally, but the evidence that the edge is real is weak
    elif n_fail == 1 and _near(failed[0], q["near_miss"]):
        grade = "C"
    else:
        grade = "D"
    for c in failed:
        reasons.insert(0, f"{c['name']}: {c.get('message') or c.get('value')}")

    score = 1.2 * max(-2.0, min(sh / sh_req, 2.2)) + 1.0 * max(-2.0, min(fit / fit_req, 2.2))
    score -= 0.8 * n_fail
    if not lo <= to <= hi:
        score -= 0.3 + (1.5 * (to - hi) if to > hi else 0.0)
    if bw_sh is not None:
        score += 0.4 * max(-1.0, min(_num(bw_sh) / sh_req, 1.6))
    score += 0.6 * max(-0.5, min(os_ratio, 1.2)) if os_ratio is not None else 0.3
    score += 0.4 * max(0.0, min(_num(stab), 1.2)) if stab is not None else 0.2
    score += 0.3 * (pos_share if pos_share is not None else 0.6)
    score += 0.3 * max(0.0, min(sub_ratio, 1.6)) if sub_ratio is not None else 0.2
    if complexity is not None:
        score -= 0.02 * max(0, complexity - 20)
    return {"grade": grade, "score": round(score, 4), "reasons": reasons[:6], "evidence": evidence,
            "brain_ready": grade == "A", "margins": {"sharpe": round(sh / sh_req, 3), "fitness": round(fit / fit_req, 3)}}


def _near(c: dict, share: float) -> bool:
    name, v, lim = c["name"], _num(c.get("value")), _num(c.get("limit"))
    if name in ("LOW_SHARPE", "LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE"):
        return lim > 0 and v >= share * lim
    if name == "HIGH_TURNOVER":
        return v <= lim * 1.35
    if name == "LOW_TURNOVER":
        return v >= lim * 0.5
    if name == "CONCENTRATED_WEIGHT":
        return v <= lim * 1.3
    return False


def grade_at_least(grade: str | None, minimum: str) -> bool:
    return GRADE_RANK.get(str(grade or "D").upper(), 3) <= GRADE_RANK.get(minimum.upper(), 3)


__all__ = ["DEFAULT_QUALITY", "GRADE_RANK", "assess", "grade_at_least", "quality_config", "quick_grade",
           "quick_score", "thresholds"]
