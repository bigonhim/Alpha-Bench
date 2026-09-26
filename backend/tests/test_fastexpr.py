from __future__ import annotations

import pytest

from alphafoundry.fastexpr import analyze, lower_text, to_expr
from alphafoundry.fastexpr.explain import classify, description
from alphafoundry.fastexpr.parser import ParseError, parse

ROUNDTRIP = [
    "rank(-ts_delta(close, 5))",
    "group_rank(ts_rank(ts_backfill(sales, 120) / cap, 252), subindustry)",
    "a = rank(close); b = rank(volume); a + b",
    "volume > adv20 ? -returns : 0",
    "trade_when(volume > adv20, rank(-ts_delta(close, 3)), -1)",
    'group_neutralize(rank(close), bucket(rank(cap), range="0.1,1,0.1"))',
    "x - -5 * (a - (b - c))",
    "-(close - open) / (high - low + 0.001)",
    "ts_decay_exp_window(close, 20, factor=0.8)",
    "quantile(returns, driver=cauchy, sigma=2)",
    "if_else(close > open, 1, -1) * rank(volume)",
    "add(close, open, filter=true)",
    "ts_regression(returns, ts_delay(returns, 1), 60, rettype=2)",
    "!(close > open) && volume < adv20 || returns == 0",
    "scale(rank(close), longscale=2)",
]


@pytest.mark.parametrize("text", ROUNDTRIP)
def test_roundtrip(text):
    node = lower_text(text)
    printed = to_expr(node)
    assert lower_text(printed).key == node.key, printed
    assert lower_text(to_expr(node, compact=True)).key == node.key


def test_canonical_equivalence():
    assert lower_text("a + b").key == lower_text("add(b, a)").key
    assert lower_text("rank(x)").key == lower_text("rank(x, rate=2)").key
    assert lower_text("a + b + c").key == lower_text("c + (b + a)").key
    assert lower_text("-1 * x").key == lower_text("-x").key
    assert lower_text("v = ts_mean(close, 5); rank(v)").key == lower_text("rank(ts_mean(close, 5))").key
    assert lower_text("c ? a : b").key == lower_text("if_else(c, a, b)").key
    assert lower_text("a - b").key != lower_text("b - a").key


@pytest.mark.parametrize("text,msg", [
    ("ts_mean(close)", "missing argument 'd'"),
    ("rank(sector)", "group"),
    ("foo(close)", "Unknown operator"),
    ("ts_mean(close, 2.5)", "whole number"),
    ("ts_mean(close, 0)", "positive"),
    ("rank(close, bogus=1)", "no option"),
    ("group_rank(close, close)", "Expected a group"),
    ("vec_avg(close)", "vector"),
    ("rank(nws12_afterhsz_sl)", "reduced"),
    ('bucket(rank(cap), range="1,0,0.1")', "range must look like"),
    ("a = 1;", "last statement"),
    ("rank(close", "')'"),
    ("close ^ 2", "power"),
    ("Close", "case-sensitive"),
])
def test_errors(text, msg):
    an = analyze(text, local_ops=set())
    assert not an.ok
    assert any(msg in d.message for d in an.diagnostics if d.severity == "error"), [d.message for d in an.diagnostics]


def test_error_spans():
    an = analyze("rank(close) + ts_mean(volume)", local_ops=set())
    err = [d for d in an.diagnostics if d.severity == "error"][0]
    assert "ts_mean(volume)" in "rank(close) + ts_mean(volume)"[err.start:err.end]
    with pytest.raises(ParseError) as e:
        parse("rank(close) $ 3")
    assert e.value.start == 12


def test_warnings_and_info():
    an = analyze("close + volume", local_ops={"add"})
    assert an.ok
    assert any("mixes units" in d.message for d in an.diagnostics)
    an = analyze("rank(unknown_field_xyz)", local_ops={"rank"})
    assert an.ok and not an.local
    assert any("Unknown data field" in d.message for d in an.diagnostics)


def test_analysis_facts():
    an = analyze("group_rank(ts_rank(ts_backfill(sales,120)/cap, 252), subindustry)")
    assert an.ok
    assert an.fields == ["cap", "sales", "subindustry"]
    assert an.lookback == 372
    assert an.local
    assert an.categories == ["fundamental", "pv"]


def test_classify_and_description():
    assert classify(lower_text("rank(-ts_delta(close, 5))"))["idea"] == "reversion"
    assert classify(lower_text("group_rank(income/cap, subindustry)"))["idea"] == "value"
    d = description(lower_text("group_rank(ts_mean(returns, 20), sector)"))
    assert d["summary"] and d["operators"]
