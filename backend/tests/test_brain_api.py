"""BRAIN API client, service and routes against a mocked BRAIN (httpx.MockTransport, no network)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from alphafoundry.brainio import credentials
from alphafoundry.brainio.brain_api import (BrainAuthError, BrainClient, DailyLimitReached, PersonaRequired,
                                            summarize_alpha)

AUTH_OK = {"user": {"id": "UX1"}, "token": {"expiry": 14400}, "permissions": ["TEST"]}


def alpha_json(aid="a1", code="rank(-ts_delta(close, 5))", checks=("PASS", "PASS")):
    return {"id": aid, "type": "REGULAR", "status": "UNSUBMITTED", "stage": "IS", "dateCreated": "2026-09-01",
            "settings": {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 4},
            "regular": {"code": code, "operatorCount": 2},
            "is": {"sharpe": 1.61, "fitness": 1.12, "turnover": 0.21, "returns": 0.087, "drawdown": 0.05,
                   "margin": 0.00083, "longCount": 1500, "shortCount": 1490,
                   "checks": [{"name": "LOW_SHARPE", "result": checks[0], "limit": 1.25, "value": 1.61},
                              {"name": "LOW_FITNESS", "result": checks[1], "limit": 1.0, "value": 1.12}]}}


class FakeBrain:
    """A tiny stateful BRAIN: auth, simulations with progress polling, alphas, record sets, checks, catalog."""

    def __init__(self, persona=False, multi=False, rate_limit_once=False):
        self.persona = persona
        self.persona_done = False
        self.multi = multi
        self.rate_limit_once = rate_limit_once
        self.polls: dict[str, int] = {}
        self.sims: dict[str, dict] = {}
        self.calls: list[str] = []
        self.logged_in = False

    def __call__(self, req: httpx.Request) -> httpx.Response:
        p, m = req.url.path, req.method
        self.calls.append(f"{m} {p}")
        if p == "/authentication":
            if m == "POST":
                if self.persona and not self.persona_done:
                    return httpx.Response(401, headers={"WWW-Authenticate": "persona",
                                                        "Location": "/authentication/persona?inquiry=inq_1"})
                if req.headers.get("authorization", "").startswith("Basic ") and "bad" in \
                        __import__("base64").b64decode(req.headers["authorization"][6:]).decode():
                    return httpx.Response(401)
                self.logged_in = True
                return httpx.Response(201, json=AUTH_OK, headers={"set-cookie": "t=tok123; Path=/"})
            if m == "GET":
                return httpx.Response(200, json=AUTH_OK) if self.logged_in else httpx.Response(401)
            if m == "DELETE":
                self.logged_in = False
                return httpx.Response(204)
        if p == "/authentication/persona" and m == "POST":
            self.persona_done = True
            self.logged_in = True
            return httpx.Response(201, json=AUTH_OK)
        if p == "/simulations" and m == "POST":
            if self.rate_limit_once:
                self.rate_limit_once = False
                return httpx.Response(429, headers={"Retry-After": "1"})
            body = json.loads(req.content)
            if isinstance(body, list):
                kids = []
                for k, b in enumerate(body):
                    sid = f"c{len(self.sims)}"
                    self.sims[sid] = {"status": "COMPLETE", "alpha": f"A{len(self.sims)}", "code": b["regular"]}
                    kids.append(sid)
                self.sims["parent"] = {"status": "COMPLETE", "children": kids}
                return httpx.Response(201, headers={"Location": "https://api.worldquantbrain.com/simulations/parent"})
            sid = f"s{len(self.sims)}"
            bad = "bad_field" in body["regular"]
            self.sims[sid] = {"status": "ERROR" if bad else "COMPLETE", "alpha": None if bad else f"A{len(self.sims)}",
                              "message": "unknown field" if bad else None}
            return httpx.Response(201, headers={"Location": f"https://api.worldquantbrain.com/simulations/{sid}"})
        if p.startswith("/simulations/") and m == "GET":
            sid = p.rsplit("/", 1)[1]
            n = self.polls.get(sid, 0)
            self.polls[sid] = n + 1
            if n < 1:
                return httpx.Response(200, json={"progress": 0.4}, headers={"Retry-After": "2"})
            return httpx.Response(200, json=self.sims[sid])
        if p.startswith("/alphas/") and p.endswith("/recordsets/pnl"):
            return httpx.Response(200, json={"schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
                                             "records": [["2020-01-02", 100.0], ["2020-01-03", 150.0],
                                                         ["2020-01-06", 120.0]]})
        if p.startswith("/alphas/") and p.endswith("/check"):
            return httpx.Response(200, json={"is": {"checks": [
                {"name": "LOW_SHARPE", "result": "PASS", "value": 1.6, "limit": 1.25},
                {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.41, "limit": 0.7},
                {"name": "PROD_CORRELATION", "result": "PASS", "value": 0.33, "limit": 0.7}]}})
        if p.startswith("/alphas/"):
            aid = p.rsplit("/", 1)[1]
            return httpx.Response(200, json=alpha_json(aid))
        if p == "/data-fields":
            off = int(req.url.params.get("offset", 0))
            lim = int(req.url.params.get("limit", 50))
            allf = [{"id": f"anl4_f{i}", "description": f"analyst field {i}", "type": "MATRIX", "coverage": 0.8,
                     "alphaCount": i, "dataset": {"id": "analyst4"}, "category": {"id": "analyst"}} for i in range(73)]
            return httpx.Response(200, json={"count": 73, "results": allf[off:off + lim]})
        if p == "/users/self/alphas":
            return httpx.Response(200, json={"count": 1, "results": [dict(alpha_json("S9"), stage="OS",
                                                                          status="ACTIVE")]})
        return httpx.Response(404)


def client(fake: FakeBrain, tmp_path) -> BrainClient:
    return BrainClient(tmp_path / "session.json", transport=httpx.MockTransport(fake), sleep=lambda s: None)


def test_login_persists_cookie_and_status(tmp_path):
    fake = FakeBrain()
    c = client(fake, tmp_path)
    info = c.login("me@x.com", "pw")
    assert info["user"]["id"] == "UX1"
    saved = json.loads((tmp_path / "session.json").read_text())
    assert any(ck["name"] == "t" and ck["value"] == "tok123" for ck in saved)
    assert "pw" not in (tmp_path / "session.json").read_text()
    assert c.auth_info()["token"]["expiry"] == 14400
    with pytest.raises(BrainAuthError):
        c.login("me@x.com", "bad")


def test_persona_flow(tmp_path):
    fake = FakeBrain(persona=True)
    c = client(fake, tmp_path)
    with pytest.raises(PersonaRequired) as e:
        c.login("me@x.com", "pw")
    assert e.value.url == "https://api.worldquantbrain.com/authentication/persona?inquiry=inq_1"
    info = c.complete_persona(e.value.url)
    assert info["user"]["id"] == "UX1"


def test_simulation_polls_and_retries_rate_limit(tmp_path):
    fake = FakeBrain(rate_limit_once=True)
    c = client(fake, tmp_path)
    c.login("me@x.com", "pw")
    seen = []
    out = c.simulate([{"expr": "rank(-ts_delta(close, 5))", "settings": {"decay": 4}},
                      {"expr": "rank(bad_field)", "settings": {}}], concurrency=1, multi=False,
                     on_result=lambda i, r: seen.append(i))
    assert out[0]["ok"] and out[0]["alpha_id"].startswith("A") and out[0]["status"] == "COMPLETE"
    assert not out[1]["ok"] and out[1]["status"] == "ERROR" and "unknown field" in out[1]["message"]
    assert sorted(seen) == [0, 1]
    posts = [x for x in fake.calls if x == "POST /simulations"]
    assert len(posts) == 3  # one 429 retry


def test_multi_simulation_children(tmp_path):
    fake = FakeBrain()
    c = client(fake, tmp_path)
    c.login("me@x.com", "pw")
    items = [{"expr": f"rank(ts_delta(close, {w}))", "settings": {}} for w in (3, 5, 10)]
    out = c.simulate(items, multi=True)
    assert [r["ok"] for r in out] == [True, True, True]
    assert len({r["alpha_id"] for r in out}) == 3


def test_daily_limit_stops_cleanly(tmp_path):
    def handler(req):
        if req.url.path == "/simulations":
            return httpx.Response(429, headers={"X-RateLimit-Remaining": "0", "Retry-After": "1"})
        return httpx.Response(200, json=AUTH_OK)
    c = BrainClient(None, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(DailyLimitReached):
        c.start_simulation({"regular": "rank(close)"})
    out = c.simulate([{"expr": "rank(close)", "settings": {}}] * 2, multi=False, concurrency=1)
    assert all(r["status"] in ("LIMIT", "SKIPPED") for r in out)


def test_summarize_alpha_and_pnl_and_check(tmp_path):
    s = summarize_alpha(alpha_json())
    assert s["expr"] == "rank(-ts_delta(close, 5))" and s["passed"] is True and s["is"]["margin"] == 8.3
    s2 = summarize_alpha(dict(alpha_json(checks=("FAIL", "PASS")), regular="rank(close)"))
    assert s2["expr"] == "rank(close)" and s2["passed"] is False and s2["failed"] == ["LOW_SHARPE"]
    assert summarize_alpha({"id": "x", "regular": "a", "is": {}})["passed"] is None
    fake = FakeBrain()
    c = client(fake, tmp_path)
    c.login("me@x.com", "pw")
    dates, daily = c.get_pnl("A1")
    assert dates == ["2020-01-02", "2020-01-03", "2020-01-06"] and np.allclose(daily, [100, 50, -30])
    chk = c.check("A1")
    assert chk["can_submit"] and chk["self_corr"] == 0.41 and chk["prod_corr"] == 0.33


def test_paginated_catalog(tmp_path):
    c = client(FakeBrain(), tmp_path)
    c.login("me@x.com", "pw")
    rows = c.datafields("USA", 1, "TOP3000", dataset_id="analyst4")
    assert len(rows) == 73 and rows[-1]["id"] == "anl4_f72"
    assert len(c.datafields(dataset_id="analyst4", limit=10)) == 10


def test_env_credentials(monkeypatch):
    monkeypatch.setenv("BRAIN_EMAIL", "a@b.c")
    monkeypatch.setenv("BRAIN_PASSWORD", "secret")
    assert credentials.load_credentials() == ("a@b.c", "secret")
    monkeypatch.delenv("BRAIN_EMAIL")
    monkeypatch.setenv("WQB_EMAIL", "w@b.c")
    monkeypatch.setenv("WQB_PASSWORD", "s2")
    assert credentials.env_credentials() == ("w@b.c", "s2")


def test_service_records_results_against_library(tmp_path, monkeypatch):
    """End to end through the service on a real Workspace (demo panel), with the fake BRAIN."""
    from alphafoundry import config
    from alphafoundry.brainio.service import BrainService
    from alphafoundry.data.demo import build_demo
    from alphafoundry.jobs.manager import JobHandle
    from alphafoundry.workspace import get_workspace

    config.ensure_dirs()
    if not (config.DEMO_PANELS_DIR / "meta.json").exists():
        build_demo(config.DEMO_PANELS_DIR, n_stocks=120, n_days=1300, seed=3)
    for k in ("BRAIN_EMAIL", "BRAIN_PASSWORD", "WQB_EMAIL", "WQB_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    ws = get_workspace()
    fake = FakeBrain()
    svc = BrainService(ws, transport=httpx.MockTransport(fake), sleep=lambda s: None)
    svc.client.login("me@x.com", "pw")

    class Mgr:
        bc = type("B", (), {"publish": lambda self, e: None})()
    h = JobHandle(Mgr(), 0, "brain_sim", {})
    h.emit = lambda force=False: None
    rows = svc.simulate_alphas(h, [{"alpha_id": None, "expr": "rank(-ts_delta(close, 5))",
                                    "settings": {"decay": 4}}], check=True)
    assert rows[0]["passed"] is True and rows[0]["status_brain"] == "ready" and rows[0]["self_corr"] == 0.41
    a = ws.store.get_alpha(rows[0]["id"])
    assert a["status_brain"] == "ready" and a["brain_alpha_id"].startswith("A")
    assert svc.usage_today() == 1
    br = svc.brain_rows(rows[0]["id"])
    assert br and br[0]["metrics"]["is"]["sharpe"] == 1.61
    assert any(r["sharpe"] == 1.61 for r in ws.store.brain_results())
