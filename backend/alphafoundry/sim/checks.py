"""BRAIN-style submission checks and Alpha Foundry robustness gates."""

from __future__ import annotations

import math
from typing import Any

import yaml

from ..config import CATALOG_DIR, USER_CHECKS_PATH, atomic_write_text

PASS, FAIL, WARNING, PENDING = "PASS", "FAIL", "WARNING", "PENDING"
HARD_CHECKS = ("LOW_SHARPE", "LOW_FITNESS", "LOW_TURNOVER", "HIGH_TURNOVER", "CONCENTRATED_WEIGHT",
               "LOW_SUB_UNIVERSE_SHARPE", "SELF_CORRELATION")


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_checks_config() -> dict:
    base = yaml.safe_load((CATALOG_DIR / "checks.yaml").read_text(encoding="utf-8"))
    if USER_CHECKS_PATH.exists():
        try:
            user = yaml.safe_load(USER_CHECKS_PATH.read_text(encoding="utf-8")) or {}
            base = _merge(base, user)
        except (OSError, yaml.YAMLError):
            pass
    return base


def save_checks_config(cfg: dict) -> dict:
    atomic_write_text(USER_CHECKS_PATH, yaml.safe_dump(cfg, sort_keys=False))
    return load_checks_config()


def _c(name: str, result: str, value: Any, limit: Any, message: str) -> dict:
    return {"name": name, "result": result, "value": value, "limit": limit, "message": message}


def run_checks(metrics: dict, delay: int, cfg: dict, sub: dict | None = None, self_corr: dict | None = None,
               stability: float | None = None, yearly: list | None = None, recent2y_sharpe: float | None = None
               ) -> dict:
    """Evaluate checks. ``metrics`` holds the "is" and optional "os" metric dicts.

    sub:       {"sharpe": float, "sub_size": float, "univ_size": float}
    self_corr: {"max_corr": float, "alpha_id": ..., "alpha_sharpe": float} or None (no submitted alphas)
    """
    m = metrics.get("is") or {}
    th = cfg["delay0" if int(delay) == 0 else "delay1"]
    sh, fit, to = float(m.get("sharpe", 0.0)), float(m.get("fitness", 0.0)), float(m.get("turnover", 0.0))
    out: list[dict] = []
    out.append(_c("LOW_SHARPE", PASS if sh >= th["sharpe_min"] else FAIL, round(sh, 3), th["sharpe_min"],
                  f"IS Sharpe {sh:.2f} vs required {th['sharpe_min']}"))
    out.append(_c("LOW_FITNESS", PASS if fit >= th["fitness_min"] else FAIL, round(fit, 3), th["fitness_min"],
                  f"IS Fitness {fit:.2f} vs required {th['fitness_min']}"))
    out.append(_c("LOW_TURNOVER", PASS if to >= cfg["turnover_min"] else FAIL, round(to, 4), cfg["turnover_min"],
                  f"Turnover {to * 100:.1f}% vs minimum {cfg['turnover_min'] * 100:.0f}%"))
    out.append(_c("HIGH_TURNOVER", PASS if to <= cfg["turnover_max"] else FAIL, round(to, 4), cfg["turnover_max"],
                  f"Turnover {to * 100:.1f}% vs maximum {cfg['turnover_max'] * 100:.0f}%"))
    mw = float(m.get("max_weight", 0.0))
    out.append(_c("CONCENTRATED_WEIGHT", PASS if mw <= cfg["max_weight"] + 1e-9 else FAIL, round(mw, 4),
                  cfg["max_weight"], f"Largest single-stock weight {mw * 100:.1f}% (limit {cfg['max_weight'] * 100:.0f}%)"))
    if sub is None:
        out.append(_c("LOW_SUB_UNIVERSE_SHARPE", PENDING, None, None, "Sub-universe not simulated yet"))
    else:
        ratio = math.sqrt(max(1e-9, sub["sub_size"]) / max(1e-9, sub["univ_size"]))
        lim = cfg["sub_universe_factor"] * ratio * sh
        ok = sub["sharpe"] >= lim
        out.append(_c("LOW_SUB_UNIVERSE_SHARPE", PASS if ok else FAIL, round(sub["sharpe"], 3), round(lim, 3),
                      f"Sub-universe ({sub.get('name', 'sub')}) Sharpe {sub['sharpe']:.2f} vs required {lim:.2f}"))
    if self_corr is None:
        out.append(_c("SELF_CORRELATION", PASS, 0.0, cfg["self_corr_max"],
                      "No submitted alphas to compare with (mark alphas as Submitted to enable)"))
    else:
        mc = float(self_corr.get("max_corr", 0.0))
        other = float(self_corr.get("alpha_sharpe", 0.0) or 0.0)
        ok = mc < cfg["self_corr_max"] or sh >= other * (1 + cfg["self_corr_sharpe_improvement"])
        msg = f"Max PnL correlation {mc:.2f} with alpha #{self_corr.get('alpha_id')}"
        if mc >= cfg["self_corr_max"] and ok:
            msg += f" (allowed: Sharpe {sh:.2f} beats {other:.2f} by 10%+)"
        out.append(_c("SELF_CORRELATION", PASS if ok else FAIL, round(mc, 3), cfg["self_corr_max"], msg))
    if recent2y_sharpe is not None:
        lim = cfg.get("two_year_sharpe_min", 1.0)
        out.append(_c("LOW_2Y_SHARPE", PASS if recent2y_sharpe >= lim else WARNING, round(recent2y_sharpe, 3), lim,
                      f"Most recent 2 IS years Sharpe {recent2y_sharpe:.2f} (soft check, verify threshold)"))
    # ----- local robustness gates (warnings)
    g = cfg.get("local_gates", {})
    osm = metrics.get("os")
    if osm and sh > 0:
        r = float(osm.get("sharpe", 0.0)) / sh
        out.append(_c("OS_DEGRADATION", PASS if r >= g.get("os_is_ratio_min", 0.5) else WARNING, round(r, 3),
                      g.get("os_is_ratio_min", 0.5), f"OS/IS Sharpe ratio {r:.2f} (holdout {osm.get('sharpe', 0):.2f})"))
    if yearly:
        is_years = [y for y in yearly if y.get("period") == "IS"]
        if is_years:
            share = sum(1 for y in is_years if y["pnl"] > 0) / len(is_years)
            out.append(_c("YEARLY_CONSISTENCY", PASS if share >= g.get("positive_years_min", 0.7) else WARNING,
                          round(share, 3), g.get("positive_years_min", 0.7),
                          f"{share * 100:.0f}% of IS years profitable"))
    if stability is not None:
        out.append(_c("PARAMETER_STABILITY", PASS if stability >= g.get("stability_min", 0.7) else WARNING,
                      round(stability, 3), g.get("stability_min", 0.7),
                      f"Sharpe retained under +/-25% window changes: {stability * 100:.0f}%"))
    dd = float(m.get("drawdown", 0.0))
    out.append(_c("DRAWDOWN", PASS if dd <= g.get("drawdown_max", 0.5) else WARNING, round(dd, 4),
                  g.get("drawdown_max", 0.5), f"Max drawdown {dd * 100:.1f}%"))

    hard = [c for c in out if c["name"] in HARD_CHECKS]
    failed = [c["name"] for c in hard if c["result"] == FAIL]
    pending = [c["name"] for c in hard if c["result"] == PENDING]
    warnings = [c["name"] for c in out if c["result"] == WARNING]
    status = "FAIL" if failed else ("PENDING" if pending else "PASS")
    return {"checks": out, "status": status, "failed": failed, "warnings": warnings, "robust": status == "PASS"
            and not warnings}
