"""Quality grading, fitness shaping and complex-alpha composition."""

from __future__ import annotations

import numpy as np
import pytest

from alphafoundry.fastexpr import analyze, lower_text, to_expr
from alphafoundry.gen.compose import Component, composites, horizon_ensemble, is_complex, weights_for
from alphafoundry.gen.optimize import acceptable, pick_best, shaping_variants
from alphafoundry.sim.checks import load_checks_config, run_checks
from alphafoundry.sim.quality import assess, grade_at_least, quick_grade, quick_score

LOCAL = {"open", "high", "low", "close", "vwap", "volume", "returns", "adv20", "cap", "assets", "equity", "sales",
         "income", "operating_income", "ebitda", "cashflow_op", "enterprise_value", "sector", "industry",
         "subindustry", "market"}
CFG = load_checks_config()


def _metrics(sh, fit, to, os_sh=None, bw=None, mw=0.02):
    m = {"is": {"sharpe": sh, "fitness": fit, "turnover": to, "returns": 0.08, "max_weight": mw, "drawdown": 0.1}}
    if os_sh is not None:
        m["os"] = {"sharpe": os_sh}
    if bw is not None:
        m["brain"] = {"sharpe": bw}
    return m


def _grade(sh, fit, to, os_sh, bw, stability=0.9, sub=(1.2, 500, 1500)):
    metrics = _metrics(sh, fit, to, os_sh, bw)
    checks = run_checks(metrics, 1, CFG, sub={"sharpe": sub[0], "sub_size": sub[1], "univ_size": sub[2]},
                        stability=stability)
    yearly = [{"period": "IS", "pnl": 1.0}] * 8 + [{"period": "IS", "pnl": -1.0}]
    return assess(metrics, checks, CFG, delay=1, extras={"stability": stability}, yearly=yearly, complexity=10)


def test_grades_reward_margin_and_evidence():
    assert _grade(2.1, 1.7, 0.15, 1.5, 2.0)["grade"] == "A"
    thin = _grade(1.3, 1.05, 0.15, 1.1, 1.3)
    assert thin["grade"] == "B" and any("safety target" in r for r in thin["reasons"])
    overfit = _grade(2.1, 1.7, 0.15, -0.4, 0.9, stability=0.3)
    assert overfit["grade"] == "C"  # passes locally, weak evidence
    near = _grade(1.15, 0.9, 0.15, 1.0, 1.1)
    assert near["grade"] in ("C", "D") and not near["brain_ready"]
    assert _grade(0.4, 0.2, 0.9, 0.1, 0.2)["grade"] == "D"
    assert _grade(2.1, 1.7, 0.15, 1.5, 2.0)["score"] > thin["score"] > _grade(0.4, 0.2, 0.9, 0.1, 0.2)["score"]


def test_quick_grade_and_score():
    assert quick_grade({"sharpe": 2.0, "fitness": 1.6, "turnover": 0.2}, CFG) == "A"
    assert quick_grade({"sharpe": 1.3, "fitness": 1.05, "turnover": 0.2}, CFG) == "B"
    assert quick_grade({"sharpe": 1.3, "fitness": 0.9, "turnover": 0.2}, CFG) == "C"
    assert quick_grade({"sharpe": 2.0, "fitness": 1.6, "turnover": 0.9}, CFG) != "A"
    assert quick_score({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.2}, CFG) > \
        quick_score({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.8}, CFG)
    assert grade_at_least("A", "B") and grade_at_least("B", "B") and not grade_at_least("C", "B")


def test_shaping_variants_are_valid_and_target_the_problem():
    node = lower_text("rank(-ts_delta(close, 5))")
    shapes = shaping_variants(node, {"decay": 0, "neutralization": "SUBINDUSTRY"}, {"turnover": 0.6, "sharpe": 1.4})
    assert shapes and len(shapes) <= 20
    kinds = {s.kind for s in shapes}
    assert {"turnover", "peers", "returns", "settings"} <= kinds
    assert any(s.settings.get("decay", 0) >= 6 for s in shapes)
    for s in shapes:
        an = analyze(to_expr(s.node), local_fields=LOCAL)
        assert an.ok and an.local, to_expr(s.node)


def test_pick_best_keeps_sharpe():
    base = {"sharpe": 1.4, "fitness": 0.8, "turnover": 0.6, "max_weight": 0.02}
    shapes = shaping_variants(lower_text("rank(-ts_delta(close, 5))"), {"decay": 0}, base)
    worse_sharpe = {"sharpe": 0.9, "fitness": 1.3, "turnover": 0.1, "max_weight": 0.02}
    better = {"sharpe": 1.35, "fitness": 1.2, "turnover": 0.2, "max_weight": 0.02}
    assert not acceptable(worse_sharpe, base, CFG)
    best = pick_best(base, [(shapes[0], worse_sharpe), (shapes[1], better)], CFG)
    assert best is not None and best[1] is better
    assert pick_best(base, [(shapes[0], worse_sharpe)], CFG) is None


def test_composites_are_valid_multi_statement_programs():
    rng = np.random.default_rng(1)
    comps = [Component("group_rank(ts_backfill(ebitda, 120) / enterprise_value, industry)", "value", {"decay": 4},
                       1.8, pnl=rng.normal(1, 5, 600)),
             Component("rank(-ts_delta(close, 5))", "reversion", {"decay": 0}, 1.4, pnl=rng.normal(1, 6, 600)),
             Component("group_rank(ts_backfill(cashflow_op, 120) / ts_backfill(assets, 120), subindustry)",
                       "quality", {}, 1.5, pnl=rng.normal(0.8, 4, 600))]
    out = composites(comps, np.eye(3), local_fields=LOCAL)
    structures = {c.structure for c in out}
    assert {"blend", "tilt", "regime", "orthogonal"} <= structures
    for c in out:
        an = analyze(c.text, local_fields=LOCAL)
        assert an.ok and an.local and is_complex(c.text), c.text
    blend = next(c for c in out if c.structure == "blend")
    assert abs(sum(blend.weights) - 1.0) < 1e-6 and min(blend.weights) >= 0.1 - 1e-9


def test_horizon_ensemble_keeps_skips_and_backfills():
    h = horizon_ensemble(Component("rank(ts_delay(ts_sum(returns, 126), 21))", "momentum"))
    assert h is not None and "ts_sum(returns, 63), 21)" in h.text and "ts_sum(returns, 252), 21)" in h.text
    assert horizon_ensemble(Component("group_rank(ts_backfill(ebitda, 120) / cap, industry)", "value")) is None


def test_weights_for_prefers_better_component():
    rng = np.random.default_rng(2)
    good = rng.normal(2.0, 5, 800)
    weak = rng.normal(0.2, 5, 800)
    w = weights_for(np.vstack([good, weak]), 2)
    assert w[0] > w[1] and abs(sum(w) - 1) < 1e-6
    assert weights_for(None, 3) == [0.33, 0.33, 0.33]


@pytest.mark.parametrize("expr", ["a = rank(close); b = rank(volume); a - b", "rank(close)"])
def test_is_complex(expr):
    assert is_complex(expr) == (";" in expr)
