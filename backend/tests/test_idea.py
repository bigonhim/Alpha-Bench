"""Idea Forge: interpretation, draft synthesis (always valid, local, on-idea), API and an end-to-end forge job."""

from __future__ import annotations

import random
import time

import pytest

from alphafoundry.fastexpr import analyze
from alphafoundry.gen.idea import (apply_overrides, brain_only_drafts, fidelity, foreign_fields, interpret, neg,
                                   preview, synthesize)

LOCAL = {"open", "high", "low", "close", "vwap", "volume", "returns", "adv20", "cap", "sharesout", "assets",
         "liabilities", "equity", "sales", "revenue", "income", "operating_income", "ebit", "ebitda", "cashflow_op",
         "cogs", "cash", "debt", "eps", "enterprise_value", "bookvalue_ps", "return_equity", "return_assets",
         "current_ratio", "sector", "industry", "subindustry", "market", "exchange", "country"}
BASE = {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY",
        "truncation": 0.08, "pasteurization": "ON", "nanHandling": "OFF"}


def test_negation_does_not_stack_minus_signs():
    assert neg("ts_delta(close, 5)") == "-ts_delta(close, 5)"
    assert neg("-ts_delta(close, 5)") == "ts_delta(close, 5)"
    assert neg("-ts_mean(volume, 20) / sharesout") == "ts_mean(volume, 20) / sharesout"
    assert neg("a - b") == "-(a - b)"


@pytest.mark.parametrize("idea, fam, direction, stated", [
    ("Stocks that drop sharply on heavy volume tend to bounce back within a week.", "reversion", -1, True),
    ("Stocks that rallied hard on no news tend to give back their gains", "reversion", -1, True),
    ("Winners keep winning over the next 12 months, skipping the most recent month.", "momentum", 1, True),
    ("High-volatility stocks outperform.", "volatility", 1, True),       # against the usual low-vol finding
    ("Low volatility stocks earn better risk-adjusted returns", "volatility", -1, True),
    ("Firms with rising R&D spending relative to sales earn higher returns", "investment", -1, True),
])
def test_mechanism_and_direction(idea, fam, direction, stated):
    s = interpret(idea, LOCAL)
    assert s.local_families[0] == fam, s.families
    assert s.direction[fam] == direction
    assert s.stated[fam] is stated


def test_multi_concept_idea_reads_both_parts():
    s = interpret("Profitable companies with low debt outperform their industry peers.", LOCAL)
    assert s.families[:2] == ["quality", "leverage"]
    assert s.direction == {"quality": 1, "leverage": -1}
    assert "debt" in s.mentioned and "industry" in s.groups
    assert "operating_income" in s.fields  # typical input added for the quality mechanism


def test_conditions_horizon_windows_and_interaction():
    s = interpret("Stocks that drop sharply on heavy volume tend to bounce back within a week.", LOCAL)
    assert {"volume_event", "extreme"} <= set(s.conditions)
    assert s.horizon == "short" and s.windows == [5]
    s = interpret("After earnings announcements, stocks with improving EPS keep drifting up for a month", LOCAL)
    assert "earnings_event" in s.conditions and "eps" in s.mentioned and s.windows == [21]
    s = interpret("Momentum works better among low-volatility stocks", LOCAL)
    assert s.interaction and s.subset_families == ["volatility"] and s.families[0] == "momentum"
    s = interpret("stocks that sell off on no news", LOCAL)
    assert "sentiment" not in s.families


def test_seeds_and_brain_only_data():
    s = interpret("Try `group_rank(ts_backfill(operating_income, 120) / cap, subindustry)` with less turnover", LOCAL)
    assert s.seeds == ["group_rank(ts_backfill(operating_income, 120) / cap, subindustry)"]
    s = interpret("implied volatility skew predicts returns", LOCAL)
    assert "options" in s.families and "implied_volatility_call_120" in s.brain_fields
    bo = brain_only_drafts(s, LOCAL, BASE)
    assert bo and all(analyze(d.expr).ok and not analyze(d.expr, local_fields=LOCAL).local for d in bo)
    vague = interpret("something about banana markets", LOCAL)
    assert vague.warnings and len(vague.local_families) >= 3


@pytest.mark.parametrize("idea", [
    "Stocks that drop sharply on heavy volume tend to bounce back within a week.",
    "Profitable companies with low debt outperform their industry peers.",
    "Cheap stocks by free cash flow yield that are starting to trend up",
    "Momentum works better among small caps, relative to sector peers, with low turnover",
    "After earnings announcements, stocks with improving EPS keep drifting up for a month",
    "rank(-ts_delta(close, 5))",
    "something about banana markets",
])
def test_drafts_are_valid_local_unique_and_on_idea(idea):
    s = interpret(idea, LOCAL)
    drafts = synthesize(s, LOCAL, BASE, budget=80, rng=random.Random(0))
    assert len(drafts) >= 20
    keys = set()
    for d in drafts:
        an = analyze(d.expr, local_fields=LOCAL)
        assert an.ok and an.local, (d.expr, [x.message for x in an.diagnostics])
        assert not foreign_fields(an.node, s), (d.expr, foreign_fields(an.node, s))
        keys.add((an.canon_hash, tuple(sorted((k, str(v)) for k, v in d.settings.items()))))
        assert 0 <= fidelity(an.node, s, d.family) <= 1
    assert len(keys) == len(drafts)
    assert any(d.sign == -1 for d in drafts)  # the reverse direction is always tested
    assert preview(s, LOCAL, BASE)


def test_overrides_replace_mechanism_and_horizon():
    s = apply_overrides(interpret("stocks with heavy volume", LOCAL), LOCAL, ["value", "quality"], "long")
    assert s.families[:2] == ["value", "quality"] and s.horizon == "long" and s.windows == [126, 252]
    assert "income" in s.fields
    drafts = synthesize(s, LOCAL, BASE, budget=30)
    assert {d.family for d in drafts} >= {"value", "quality"}


# --------------------------------------------------------------------------- API + job


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from alphafoundry import config
    from alphafoundry.data.demo import build_demo

    config.ensure_dirs()
    if not (config.DEMO_PANELS_DIR / "meta.json").exists():
        build_demo(config.DEMO_PANELS_DIR, n_stocks=120, n_days=1300, seed=3)
    config.save_settings({"workers": 0, "active_dataset": "demo"})
    from alphafoundry.api.app import app

    with TestClient(app) as c:
        yield c


def test_interpret_endpoint(client):
    r = client.post("/api/forge/interpret", json={"text": "Profitable companies with low debt outperform peers"}).json()
    assert [f["id"] for f in r["families"]][:2] == ["quality", "leverage"]
    assert r["preview"] and r["templates"] and r["all_families"]
    assert any(f["id"] == "debt" and f["mentioned"] for f in r["fields"])
    r2 = client.post("/api/forge/interpret", json={"text": "heavy volume", "families": ["momentum"],
                                                   "horizon": "long"}).json()
    assert r2["families"][0]["id"] == "momentum" and r2["horizon"] == "long"
    assert client.post("/api/forge/interpret", json={"text": ""}).json()["preview"] == []


def test_forge_job_produces_a_saved_champion(client):
    cfg = {"idea": "Profitable companies with low debt outperform their industry peers.", "effort": "quick",
           "draft": 16, "combine": 6, "refine_top": 2, "fixes_per": 2, "sweep_top": 1, "gp_population": 8,
           "gp_generations": 1, "finalists": 3, "time_limit_min": 4, "seed": 1}
    jid = client.post("/api/jobs", json={"kind": "forge", "config": cfg}).json()["id"]
    for _ in range(1500):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    p = st["progress"]
    assert p["spec"]["families"][0]["id"] == "quality"
    assert all(p["stages"][s]["status"] in ("done", "skipped") for s in ("draft", "combine", "refine", "evolve", "polish"))
    f = p["forge"]
    champ = f["champion"]
    assert champ and champ["id"] and champ["lineage"] and champ["sharpe"] is not None
    assert f["hypothesis"]["text"]
    alpha = client.get(f"/api/alphas/{champ['id']}").json()["alpha"]
    assert alpha["origin"] == "forge" and "forge" in alpha["tags"] and "forge-champion" in alpha["tags"]
    assert "Profitable companies" in alpha["description"]["idea"]  # the idea becomes the BRAIN description
    listed = client.get("/api/alphas", params={"tag": f"forge-{jid}"}).json()
    assert listed["total"] >= 1
