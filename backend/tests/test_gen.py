"""Generators, Doctor, BRAIN import/export: everything they emit must be valid Fast Expression."""

from __future__ import annotations

import json
import random

import pytest

from alphafoundry.brainio import export as bexport
from alphafoundry.brainio import importer
from alphafoundry.fastexpr import analyze, lower_text, to_expr
from alphafoundry.gen.doctor import diagnose
from alphafoundry.gen.gp import Individual, Variation, nondominated_sort
from alphafoundry.gen.grammar import Grammar, is_degenerate
from alphafoundry.gen.templates import alpha101_candidates, expand, load_alpha101, load_templates

LOCAL = {"open", "high", "low", "close", "vwap", "volume", "returns", "adv20", "cap", "sharesout", "assets",
         "liabilities", "equity", "sales", "revenue", "income", "operating_income", "ebit", "ebitda", "cashflow_op",
         "cogs", "cash", "debt", "eps", "enterprise_value", "bookvalue_ps", "return_equity", "return_assets",
         "current_ratio", "sector", "industry", "subindustry", "market", "exchange", "country"}


def test_every_template_expansion_is_valid():
    tmpls = load_templates()
    assert len(tmpls) >= 70
    n = 0
    for t in tmpls:
        for e in expand(t, local_fields=None, validate=False):
            an = analyze(e, local_ops=None)
            assert an.ok, (t.id, e, [d.message for d in an.diagnostics])
            n += 1
    assert n >= 280  # 77 templates -> 284 distinct expressions before settings grids


def test_local_expansions_and_brain_only_split():
    tmpls = load_templates()
    local_ids = {t.id for t in tmpls if expand(t, local_fields=LOCAL, limit=1)}
    assert "rev_delta" in local_ids and "val_yield" in local_ids
    assert "brain_opt_skew" not in local_ids  # option data is BRAIN-only


def test_alpha101_parse():
    rows = load_alpha101()
    assert len(rows) >= 25
    for a in rows:
        an = analyze(a["expr"], local_ops=None)
        assert an.ok, (a["id"], [d.message for d in an.diagnostics])
    assert len(alpha101_candidates({"universe": "TOP3000"}, LOCAL)) >= 40


def test_grammar_trees_valid_and_roundtrip():
    g = Grammar(LOCAL, random.Random(1))
    good = 0
    for _ in range(300):
        n = g.tree()
        if is_degenerate(n):
            continue
        text = to_expr(n)
        an = analyze(text, local_fields=LOCAL)
        assert an.ok, (text, [d.message for d in an.diagnostics])
        assert lower_text(text).key == n.key
        assert an.local, (text, an.brain_only_reasons)
        good += 1
    assert good > 200


def test_gp_variation_stays_valid():
    rng = random.Random(7)
    g = Grammar(LOCAL, rng)
    var = Variation(g, rng, max_depth=6, max_nodes=30)
    pop = [Individual(g.tree(), {"decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08}) for _ in range(20)]
    for _ in range(300):
        a, b = rng.choice(pop), rng.choice(pop)
        child = var.mate(a, b) if rng.random() < 0.5 else var.mutate(a)
        text = to_expr(child.node)
        an = analyze(text, local_fields=LOCAL)
        assert an.ok, (text, [d.message for d in an.diagnostics])
        assert child.node.depth <= 6 + 1 and child.node.size <= 31
        pop.append(child)


def test_nsga_sort_orders_by_dominance():
    def ind(f, nov, size, viol=0.0):
        i = Individual(lower_text("rank(close)"), {})
        i.objectives = (f, nov, -size)
        i.violation = viol
        return i
    a, b, c, d = ind(2.0, 0.8, 5), ind(1.0, 0.5, 9), ind(1.5, 0.9, 12), ind(3.0, 0.9, 3, viol=1.0)
    fronts = nondominated_sort([a, b, c, d])
    assert a in fronts[0] and c in fronts[0]
    assert b not in fronts[0]
    assert d in fronts[-1]  # infeasible individuals are dominated by feasible ones


def test_doctor_rewrites_are_valid():
    node = lower_text("rank(-returns)")
    fixes = diagnose(node, {"decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08},
                     ["HIGH_TURNOVER", "LOW_SHARPE", "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"],
                     {"sharpe": -0.8, "turnover": 1.2}, ["OS_DEGRADATION"])
    assert len(fixes) >= 8
    labels = {f.label for f in fixes}
    assert "Flip the sign" in labels and any(l.startswith("Decay") for l in labels)
    for f in fixes:
        an = analyze(to_expr(f.node), local_fields=LOCAL)
        assert an.ok, (f.label, to_expr(f.node))


def test_import_formats():
    csv_txt = ("code,sharpe,fitness,turnover,returns,status\n"
               "\"rank(-ts_delta(close, 5))\",1.4,1.1,35.2%,8.1%,PASS\n"
               "\"rank(close)\",0.3,0.1,12,2.0,FAIL\n")
    rows = importer.parse_any(csv_txt)
    assert len(rows) == 2
    assert rows[0]["metrics"]["turnover"] == pytest.approx(0.352)
    assert rows[1]["metrics"]["turnover"] == pytest.approx(0.12)  # bare "12" means 12%
    assert rows[0]["metrics"]["passed"] is True and rows[1]["metrics"]["passed"] is False
    js = json.dumps([{"regular": {"code": "rank(volume)"}, "settings": {"decay": 4, "neutralization": "SECTOR"},
                      "is": {"sharpe": 1.3, "fitness": 1.05, "turnover": 0.2, "returns": 0.07, "margin": 0.0006,
                             "checks": [{"name": "LOW_SHARPE", "result": "PASS"}]}}])
    r = importer.parse_any(js)[0]
    assert r["expr"] == "rank(volume)" and r["settings"]["decay"] == 4
    assert r["metrics"]["margin"] == pytest.approx(6.0) and r["metrics"]["passed"] is True
    text = "rank(-ts_delta(close, 5))\nSharpe 1.52 Turnover 23.1% Fitness 1.21 Returns 9.4%\n\nrank(close)\nSharpe 0.4"
    t = importer.parse_any(text)
    assert [x["expr"] for x in t] == ["rank(-ts_delta(close, 5))", "rank(close)"]
    assert t[0]["metrics"]["turnover"] == pytest.approx(0.231)


def test_export_payload_schema():
    alphas = [{"id": i, "expr": "rank(-returns)", "settings": {"decay": i}} for i in range(12)]
    data = json.loads(bexport.to_json(alphas, batch=10))
    batches = data["multi_simulations"]
    assert [len(b) for b in batches] == [10, 2]
    p = batches[0][3]
    assert p["type"] == "REGULAR" and p["regular"] == "rank(-returns)"
    assert set(p["settings"]) == set(bexport.BRAIN_SETTING_KEYS)
    assert p["settings"]["decay"] == 3 and p["settings"]["language"] == "FASTEXPR"
