"""Idea input understanding: every format produces analyzer-valid Fast Expressions."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from alphafoundry.fastexpr import analyze
from alphafoundry.gen import ideaparse
from alphafoundry.gen.ideaparse import (compile_phrase, extract_document_text, key_sentences, parse_input,
                                        search_fields, translate_paper, translate_python)


def ok(e: str | None) -> bool:
    return e is not None and analyze(e).ok


@pytest.mark.parametrize("paper, expect", [
    ("(-1 * correlation(open, volume, 10))", "-ts_corr(open, volume, 10)"),
    ("(sign(delta(volume, 1)) * (-1 * delta(close, 1)))", "ts_delta(volume, 1)"),
    ("((close - open) / ((high - low) + .001))", "0.001"),
    ("(-1 * Ts_Rank(rank(low), 9))", "-ts_rank(rank(low), 9)"),
    ("IndNeutralize(correlation(rank(close), rank(adv20), 9.91), IndClass.industry)",
     "group_neutralize(ts_corr(rank(close), rank(adv20), 10), industry)"),
    ("((rank(delta(close, 7)) ^ 2) * -1)", "power(rank(ts_delta(close, 7)), 2)"),
    ("(SignedPower(Ts_ArgMax(returns, 5), 2) - 0.5)", "signed_power(ts_arg_max(returns, 5), 2)"),
    ("rank(min(delay(close,1), 5))", "ts_min(ts_delay(close, 1), 5)"),
    ("rank(adv60)", "ts_mean(volume, 60)"),
])
def test_paper_notation(paper, expect):
    out = translate_paper(paper)
    assert ok(out) and expect in out, out


def test_paper_ternary():
    out = translate_paper("((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2) ? (-1 * 1) : 1")
    assert ok(out) and "if_else" in out


@pytest.mark.parametrize("code, expect", [
    ("mom = df['close'].pct_change(252).shift(21)\nalpha = mom.rank(axis=1, pct=True)", "ts_delay("),
    ("def alpha(df):\n    return -df['close'].diff(5)", "-ts_delta(close, 5)"),
    ("ret = df['close'].pct_change()\nvol = ret.rolling(20).std()\nsignal = -(ret.rolling(5).sum() / vol)",
     "ts_std_dev(ret, 20)"),
    ("x = (df.close - df.close.rolling(20).mean()) / df.close.rolling(20).std()\n"
     "alpha = np.where(df.volume > df.volume.rolling(20).mean(), -x, 0)", "?"),
    ("s = df['volume'].rolling(10).corr(df['close'])\nalpha = -s", "ts_corr(volume, close, 10)"),
    ("ema = df['close'].ewm(span=10).mean()\nalpha = (df['close'] / ema - 1).rank(axis=1)", "ts_decay_exp_window"),
    ("z = (df['close'] - df['close'].mean(axis=1)) / df['close'].std(axis=1)", "zscore(close)"),
    ("alpha = df['close'].groupby(sector).rank()", "group_rank(close, sector)"),
])
def test_python(code, expect):
    out, _warns = translate_python(code)
    assert ok(out) and expect in out, out


def test_python_rejects_lookahead():
    out, warns = translate_python("alpha = df['close'].shift(-1)")
    assert out is None and any("future" in w for w in warns)


@pytest.mark.parametrize("phrase, expect", [
    ("12-month return skipping the last month", "ts_delay(ts_sum(returns, 231), 21)"),
    ("the 20-day average of volume", "ts_mean(volume, 20)"),
    ("5-day change in close", "ts_delta(close, 5)"),
    ("ratio of EBITDA to enterprise value", "ebitda"),
    ("book-to-market", "equity"),
    ("EV/EBITDA", "enterprise_value / ts_backfill(ebitda, 120)"),
    ("gross margin", "cogs"),
    ("debt to equity", "debt"),
    ("asset growth", "ts_delta(ts_backfill(assets, 120), 252)"),
    ("free cash flow yield", "capex"),
    ("z-score of volume over 60 days", "ts_zscore(volume, 60)"),
    ("EBITDA to enterprise value ranked within industry", "group_rank("),
    ("correlation between close and volume over 10 days", "ts_corr(close, volume, 10)"),
    ("volatility over 20 days", "ts_std_dev(returns, 20)"),
    ("distance from the 52-week high", "ts_max(high, 252)"),
    ("low P/E", "ts_backfill(eps, 120) / close"),
    ("Signal = rank(EBITDA / EV)", "rank(ts_backfill(ebitda, 120) / enterprise_value)"),
    ("Gross profitability is revenue minus cost of goods sold, scaled by total assets", "cogs"),
])
def test_described_computations(phrase, expect):
    out = compile_phrase(phrase)
    assert out, phrase
    assert all(ok(e) for _, e in out)
    assert any(expect in e for _, e in out), out


def test_bare_fields_are_not_compiled():
    assert compile_phrase("using operating_income, debt") == []


def test_parse_json_record():
    raw = json.dumps({"idea": "Profitable firms with low debt outperform", "fields": ["operating_income", "debt"],
                      "neutralization": "INDUSTRY",
                      "expression": "group_rank(ts_backfill(operating_income, 120) / ts_backfill(assets, 120), industry)"})
    r = parse_input(raw)
    assert "json" in r.formats and r.settings["neutralization"] == "INDUSTRY"
    assert r.seeds and ok(r.seeds[0]) and "Profitable firms" in r.text


def test_parse_list_of_ideas():
    r = parse_input("- Stocks that drop on heavy volume bounce back within a week\n"
                    "- Cheap stocks by cash-flow yield outperform\n"
                    "1. Low volatility stocks earn higher risk-adjusted returns")
    assert r.format == "list" and len(r.sub_ideas) == 3


def test_parse_multistatement_fastexpr_without_semicolons():
    r = parse_input("value = group_rank(ts_backfill(ebitda, 120) / enterprise_value, industry)\n"
                    "rev = rank(-ts_delta(close, 5))\n0.6 * value + 0.4 * rev")
    assert r.format == "fastexpr" and ";" in r.seeds[0] and ok(r.seeds[0])


def test_parse_mixed_prose_code_and_settings():
    r = parse_input("Short-term reversal works best on volatile names. Neutralize by subindustry, decay 4.\n"
                    "```\nrank(-ts_delta(close, 3)) * rank(ts_std_dev(returns, 20))\n```")
    assert r.format == "mixed" and r.seeds and r.settings == {"neutralization": "SUBINDUSTRY", "decay": 4}
    assert "reversal" in r.text


def test_parse_python_and_paper_inputs():
    assert parse_input("import pandas as pd\nret = df['close'].pct_change()\nalpha = -ret.rolling(5).sum()"
                       ".rank(axis=1)").format == "python"
    r = parse_input("Alpha#6: (-1 * correlation(open, volume, 10))")
    assert r.format == "paper" and r.seeds == ["-ts_corr(open, volume, 10)"]


PAPER = ("Abstract. We document that firms with high gross profitability relative to their assets earn "
         "significantly higher average returns than unprofitable firms. The effect is robust within industries.\n"
         "1. Introduction. Asset pricing research has long studied expected returns. "
         + "This paragraph discusses prior work in detail and is not about the mechanism itself. " * 30 +
         "\n2. Data. Gross profitability is revenue minus cost of goods sold, scaled by total assets.\n"
         "3. Results. Table 2 reports t-stats and standard errors. We find that a long-short portfolio of profitable "
         "minus unprofitable stocks earns 0.31% per month over the following year.\n"
         "References\nFama, E. and French, K. (1993). Common risk factors. Profitability value momentum returns.\n")


def test_document_key_sentences_and_formula():
    r = parse_input(PAPER)
    assert r.format == "document" and 2 <= len(r.key_sentences) <= 6
    assert any("gross profitability" in s.lower() for s in r.key_sentences)
    assert not any("Fama" in s for s in r.key_sentences)
    assert any("cogs" in e and "assets" in e for _, e in r.compiled)
    assert key_sentences(PAPER, k=2)


def test_extract_docx_html_rtf():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><w:body><w:p><w:r><w:t>Stocks that drop sharply</w:t></w:r>"
                                        "</w:p><w:p><w:r><w:t>bounce back &amp; recover.</w:t></w:r></w:p></w:body>"
                                        "</w:document>")
    assert extract_document_text("idea.docx", buf.getvalue()) == "Stocks that drop sharply\nbounce back & recover."
    html = b"<html><style>x{}</style><body><h1>Idea</h1><p>Momentum among calm stocks.</p><script>1</script></body>"
    t = extract_document_text("p.html", html)
    assert "Momentum among calm stocks." in t and "x{}" not in t and "script" not in t
    assert "reversal" in extract_document_text("n.rtf", b"{\\rtf1\\ansi reversal idea\\par}")
    assert extract_document_text("n.txt", "café idea".encode()) == "café idea"


def test_search_fields_ranks_catalog(monkeypatch):
    fake = {"anl4_eps_rev": {"type": "MATRIX", "description": "Analyst EPS estimate revisions, 1 month",
                             "dataset": "analyst4", "category": "analyst"},
            "nws_sent": {"type": "VECTOR", "description": "News sentiment score", "dataset": "news12"},
            "close": {"type": "MATRIX", "description": "Daily close price", "category": "pv"}}
    monkeypatch.setattr(ideaparse, "field_map", lambda: fake)
    ideaparse._FIELD_INDEX["key"] = None
    hits = search_fields("analyst estimate revisions", k=2)
    assert hits[0][0] == "anl4_eps_rev"
    assert search_fields("news sentiment")[0][0] == "nws_sent"
    ideaparse._FIELD_INDEX["key"] = None
