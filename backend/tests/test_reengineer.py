"""Re-engineer: every proposed rewrite must be valid Fast Expression; scoring and diagnosis behave sensibly."""

from __future__ import annotations

import numpy as np
import pytest

from alphafoundry.fastexpr import analyze, lower_text, to_expr
from alphafoundry.gen import reengineer as R
from alphafoundry.sim.checks import load_checks_config

LOCAL = {"open", "high", "low", "close", "vwap", "volume", "returns", "adv20", "cap", "sharesout", "assets",
         "liabilities", "equity", "sales", "revenue", "income", "operating_income", "ebit", "ebitda", "cashflow_op",
         "cogs", "cash", "debt", "eps", "enterprise_value", "bookvalue_ps", "return_equity", "return_assets",
         "current_ratio", "sector", "industry", "subindustry", "market", "exchange", "country"}
SETTINGS = {"decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08}
SEEDS = [
    "ts_delta(close, 5)",
    "sales",
    "rank(ts_delta(close, 5)) + rank(volume)",
    "-group_rank(ts_mean(returns, 20), sector)",
    "trade_when(volume > adv20, rank(-ts_delta(close, 3)), -1)",
    "ts_decay_linear(rank(ts_backfill(income, 120) / cap), 10) * rank(ts_std_dev(returns, 20))",
    "volume > adv20 ? rank(-returns) : rank(ts_zscore(close, 60))",
]


@pytest.mark.parametrize("expr", SEEDS)
def test_every_move_is_valid_and_new(expr):
    node = lower_text(expr)
    ctx = R.MoveContext(local_fields=LOCAL, metrics={"turnover": 0.6, "sharpe": 0.4},
                        exposures={"reversal": -0.5, "size": 0.3})
    total = 0
    for stage in ("direction", "shape", "neutralize", "horizon", "turnover", "condition"):
        moves = R.propose(stage, node, SETTINGS, ctx)
        keys = set()
        for mv in moves:
            text = to_expr(mv.node)
            an = analyze(text, local_fields=LOCAL)
            assert an.ok, (stage, mv.label, text, [d.message for d in an.diagnostics])
            assert an.local, (stage, mv.label, text, an.brain_only_reasons)
            assert lower_text(text).key == mv.node.key  # printed form round-trips
            k = (mv.node.key, R._skey(mv.settings))
            assert k not in keys and k != (node.key, R._skey(SETTINGS))
            assert mv.stage == stage and mv.label and mv.reason
            keys.add(k)
        total += len(moves)
    assert total >= 15


def test_turnover_moves_follow_turnover_level():
    node = lower_text("rank(-ts_delta(close, 5))")
    hi = R.propose("turnover", node, SETTINGS, R.MoveContext(LOCAL, metrics={"turnover": 0.65}))
    lo = R.propose("turnover", node, {**SETTINGS, "decay": 8}, R.MoveContext(LOCAL, metrics={"turnover": 0.02}))
    assert any("trade_when" in m.expr for m in hi) and any(m.settings["decay"] > 0 for m in hi)
    assert lo and all(m.settings["decay"] < 8 for m in lo)


def test_structure_toggle_limits_moves():
    node = lower_text("rank(ts_delta(close, 5)) + rank(volume)")
    ctx = R.MoveContext(LOCAL, allow_structure=False)
    assert [m.label for m in R.propose("direction", node, SETTINGS, ctx)] == ["Flip the sign"]
    assert R.propose("condition", node, SETTINGS, ctx) == []


def test_peel_and_combiner():
    core, sign = R.peel(lower_text("-group_rank(ts_mean(returns, 20), sector)"))
    assert to_expr(core) == "ts_mean(returns, 20)" and sign == -1
    path, comb = R.find_combiner(lower_text("rank(ts_decay_linear(rank(close) - rank(volume), 5))"))
    assert comb is not None and comb.name == "subtract" and path == (0, 0)
    assert R.find_combiner(lower_text("rank(close)")) == (None, None)
    assert to_expr(R.ranked(lower_text("-rank(close)"))) == "-rank(close)"


def test_objective_rewards_sharpe_and_penalizes_violations():
    cfg = load_checks_config()
    rng = np.random.default_rng(0)
    pnl = rng.normal(1000, 10000, 1500)
    good, _ = R.objective({"sharpe": 2.0, "fitness": 1.6, "turnover": 0.2, "max_weight": 0.02}, pnl, 8, cfg)
    weak, _ = R.objective({"sharpe": 0.8, "fitness": 0.5, "turnover": 0.2, "max_weight": 0.02}, pnl, 8, cfg)
    churn, parts = R.objective({"sharpe": 2.0, "fitness": 1.6, "turnover": 0.9, "max_weight": 0.02}, pnl, 8, cfg)
    heavy, parts2 = R.objective({"sharpe": 2.0, "fitness": 1.6, "turnover": 0.2, "max_weight": 0.3}, pnl, 30, cfg)
    assert good > weak and good > churn and good > heavy
    assert "turnover_high" in parts["penalties"]
    assert {"concentration", "complexity"} <= set(parts2["penalties"])
    assert R.meets_target({"sharpe": 2.1, "fitness": 1.6}, {"penalties": {"complexity": 0.1}}, 2.0, 1.5)
    assert not R.meets_target({"sharpe": 2.1, "fitness": 1.6}, parts, 2.0, 1.5)


def test_blend_candidates_prefer_decorrelated_edges():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 1000)
    twin = base + rng.normal(0, 0.1, 1000)
    other = rng.normal(0, 1, 1000)
    pool = [R.Partner("twin", "twin", lower_text("rank(close)"), 2.0, twin),
            R.Partner("other", "other", lower_text("rank(volume)"), 1.5, other),
            R.Partner("loser", "loser", lower_text("rank(cap)"), 0.1, other)]
    picks = R.blend_candidates(base, 1.2, pool)
    assert [p.id for p, _ in picks] == ["other"]
    moves = R.blend_moves(lower_text("rank(-returns)"), SETTINGS, picks)
    assert len(moves) == 2 and all(analyze(m.expr, local_fields=LOCAL).ok for m in moves)


def test_diagnosis_names_the_defects():
    cfg = load_checks_config()
    checks = {"checks": [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}]}
    items = R.diagnose({"sharpe": -1.4, "fitness": -0.5, "turnover": 0.9, "max_weight": 0.2}, {"sharpe": -1.0},
                       (-1.0, -1.8), checks, {"sub_sharpe": -1.1}, {"reversal": -0.72}, cfg)
    codes = {i["code"] for i in items}
    assert {"backwards", "churn", "concentrated", "small_caps", "exposure_reversal"} <= codes
    assert all(i["stage"] in R.STAGES for i in items)
    healthy = R.diagnose({"sharpe": 2.0, "fitness": 1.5, "turnover": 0.2, "max_weight": 0.02}, None, (1.9, 2.1),
                         {"checks": []}, None, {}, cfg)
    assert [i["code"] for i in healthy] == ["healthy"]
