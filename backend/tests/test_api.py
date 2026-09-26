"""End-to-end API tests on the synthetic demo dataset (in-process evaluation, no worker pool)."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from alphafoundry import config
    from alphafoundry.data.demo import build_demo

    config.ensure_dirs()
    build_demo(config.DEMO_PANELS_DIR, n_stocks=120, n_days=1300, seed=3)
    config.save_settings({"workers": 0, "active_dataset": "demo"})
    from alphafoundry.api.app import app

    with TestClient(app) as c:
        yield c


def test_status_catalog_parse(client):
    s = client.get("/api/status").json()
    assert s["data"]["source"] == "demo" and s["data"]["N"] == 120
    c = client.get("/api/catalog").json()
    names = {o["name"] for o in c["operators"]}
    assert {"ts_rank", "group_neutralize", "trade_when", "vec_avg"} <= names
    assert any(o["local"] for o in c["operators"]) and any(not o["local"] for o in c["operators"])
    p = client.post("/api/parse", json={"text": "rank(-ts_delta(close, 5))"}).json()
    assert p["ok"] and p["local"] and p["tags"]["idea"] == "reversion"
    bad = client.post("/api/parse", json={"text": "rank(close"}).json()
    assert not bad["ok"] and bad["diagnostics"][0]["severity"] == "error"


def test_simulate_extras_doctor_sweep(client):
    body = {"text": "rank(-ts_delta(close, 5))", "settings": {"decay": 0, "neutralization": "SUBINDUSTRY"},
            "extras": True}
    r = client.post("/api/simulate", json=body).json()
    assert r["ok"] and not r["brain_only"]
    assert {"is", "os", "all"} <= set(r["metrics"])
    names = {c["name"] for c in r["checks"]["checks"]}
    assert {"LOW_SHARPE", "LOW_FITNESS", "HIGH_TURNOVER", "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE",
            "SELF_CORRELATION"} <= names
    assert len(r["series"]["dates"]) == len(r["series"]["cum_pnl"])
    assert 0 <= r["pass_prob"] <= 1
    bo = client.post("/api/simulate", json={"text": "rank(implied_volatility_call_120)"}).json()
    assert bo["ok"] and bo["brain_only"]
    d = client.post("/api/doctor", json={"text": "rank(-returns)", "settings": {}}).json()
    assert d["ok"] and d["fixes"], d
    assert all("metrics" in f for f in d["fixes"])
    with client.stream("POST", "/api/sweep", json={"text": "rank(-ts_delta(close, 5))",
                                                   "grid": {"decay": [0, 4], "neutralization": ["MARKET", "SECTOR"],
                                                            "truncation": [0.08]}}) as resp:
        events = [json.loads(line) for line in resp.iter_lines() if line]
    assert sum(e["type"] == "cell" for e in events) == 4 and events[-1]["type"] == "best"


def test_save_library_export_import(client):
    r = client.post("/api/alphas", json={"text": "group_rank(income/cap, subindustry)", "settings": {},
                                          "tags": ["value"]}).json()
    assert r["ok"]
    aid = r["id"]
    lst = client.get("/api/alphas").json()
    assert lst["total"] >= 1 and any(a["id"] == aid for a in lst["rows"])
    det = client.get(f"/api/alphas/{aid}").json()
    assert det["alpha"]["id"] == aid and det["result"]["ok"]
    client.patch(f"/api/alphas/{aid}", json={"notes": "hello", "starred": True})
    assert client.get(f"/api/alphas/{aid}").json()["alpha"]["starred"] == 1
    js = client.post("/api/export", json={"ids": [aid], "format": "json"}).text
    payload = json.loads(js)
    sim = payload["multi_simulations"][0][0]
    assert sim["type"] == "REGULAR" and sim["settings"]["language"] == "FASTEXPR" and "group_rank" in sim["regular"]
    csv_txt = client.post("/api/export", json={"ids": [aid], "format": "csv"}).text
    assert csv_txt.startswith("id,expr")
    imp = client.post("/api/import", json={"text": "code,sharpe,fitness,turnover,status\n"
                                                    "\"group_rank(income / cap, subindustry)\",1.6,1.2,3.1%,PASS\n"
                                                    "\"rank(-ts_delta(close, 3))\",0.9,0.4,55%,FAIL\n"}).json()
    assert imp["parsed"] == 2 and imp["matched"] >= 1 and imp["created"] >= 1
    cal = client.get("/api/calibration").json()
    assert len(cal["points"]) >= 1
    client.post("/api/alphas/submitted", json={"ids": [aid], "submitted": True})
    r2 = client.post("/api/simulate", json={"text": "group_rank(income/cap, subindustry)", "settings": {}}).json()
    sc = next(c for c in r2["checks"]["checks"] if c["name"] == "SELF_CORRELATION")
    assert sc["value"] > 0.95  # identical alpha is fully correlated with the submitted one
    corr = client.post("/api/correlation", json={"ids": [aid]}).json()
    assert corr["ids"] == [aid]


def test_jobs_templates_and_gp(client):
    j = client.post("/api/jobs", json={"kind": "templates", "config": {"per_template": 1, "settings_per_expr": 1,
                                                                        "max_candidates": 12, "save_min_sharpe": -9,
                                                                        "families": ["reversion", "value"]}}).json()
    jid = j["id"]
    for _ in range(600):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    assert st["stats"]["evaluated"] >= 5
    j = client.post("/api/jobs", json={"kind": "gp", "config": {"population": 8, "generations": 2, "islands": 1,
                                                                 "halving": False, "min_sharpe": -9,
                                                                 "min_fitness": -9, "time_limit_min": 3}}).json()
    jid = j["id"]
    for _ in range(1500):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    assert st["stats"]["evaluated"] >= 8


def test_reengineer_job_turns_a_weak_alpha_around(client):
    j = client.post("/api/jobs", json={"kind": "reengineer", "config": {
        "expr": "ts_delta(close, 5)", "settings": {"decay": 0, "neutralization": "SUBINDUSTRY"},
        "time_limit_min": 1.0, "evolve": False, "max_passes": 1}}).json()
    jid = j["id"]
    for _ in range(1500):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    rep = client.get(f"/api/reengineer/{jid}").json()
    assert rep["status"] == "done" and rep["diagnosis"] and rep["stages"]
    assert rep["diagnosis"][0]["code"] == "backwards"  # the raw 5-day change is a reversal signal with the wrong sign
    final = rep["final"]
    champ = final["champion"]
    assert champ is not None and champ["alpha_id"]
    assert champ["is"]["sharpe"] > rep["original"]["is"]["sharpe"] + 1.0
    assert champ["lineage"] and champ["lineage"][0]["label"] == "Flip the sign"
    for step in champ["lineage"]:
        assert client.post("/api/parse", json={"text": step["expr"]}).json()["ok"]
    assert final["verdict"]["level"] in ("good", "warn", "bad")
    assert len(final["series"]["dates"]) == len(final["series"]["champion"]) == len(final["series"]["original"])
    saved = client.get(f"/api/alphas/{champ['alpha_id']}").json()["alpha"]
    assert saved["origin"] == "reengineer" and "reengineered" in saved["tags"]
    assert client.get("/api/reengineer/999999").status_code == 404
    bad = client.post("/api/jobs", json={"kind": "reengineer", "config": {"expr": "rank(implied_volatility_call_120)",
                                                                        "time_limit_min": 0.2}}).json()
    for _ in range(100):
        st = client.get(f"/api/jobs/{bad['id']}").json()
        if st["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.1)
    assert st["status"] == "error" and "locally" in st["error"]
