"""FastAPI application: REST API, WebSocket events, and the built React UI."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import orjson
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .. import config
from ..catalog import OPERATORS, catalog_json, import_user_fields
from ..fastexpr import lower_text
from ..fastexpr.explain import classify, description
from ..gen.bandit import Bandit
from ..gen.templates import (delete_user_template, load_alpha101, save_user_template, templates_summary)
from ..jobs.manager import Broadcaster, JobManager
from ..sim.checks import load_checks_config, save_checks_config
from ..sim.simulator import NEUTRALIZATIONS_ALL, NEUTRALIZATIONS_LOCAL, REGIONS, UNIVERSES_BRAIN, SimSettings
from ..workspace import Workspace, get_workspace, universes_for

log = logging.getLogger("alphafoundry.api")


class ORJSON(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS, default=str)


STATE: dict[str, Any] = {}


def ws_() -> Workspace:
    return STATE["ws"]


def mgr() -> JobManager:
    return STATE["mgr"]


def _warmup(ws: Workspace) -> None:
    """Compile/load numba kernels in the background so the first interactive run is fast."""
    try:
        for e in ("rank(-ts_delta(close, 5))", "group_rank(ts_mean(returns, 20), sector)",
                  "ts_corr(rank(close), rank(volume), 10)", "ts_decay_linear(zscore(returns), 5)"):
            ws.simulate(e, None, extras=False)
        STATE["warm"] = True
    except Exception:  # noqa: BLE001
        log.exception("warmup failed")
        STATE["warm"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    ws = get_workspace()
    bc = Broadcaster()
    bc.bind(asyncio.get_running_loop())
    STATE.update(ws=ws, bc=bc, mgr=JobManager(ws, bc), warm=False)
    threading.Thread(target=_warmup, args=(ws,), daemon=True).start()
    yield
    STATE["mgr"].shutdown()


app = FastAPI(title="Alpha Foundry", version="1.0.0", default_response_class=ORJSON, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- models


class TextIn(BaseModel):
    text: str


class SimIn(BaseModel):
    text: str
    settings: dict | None = None
    extras: bool = False
    rationale: str | None = None


class SweepIn(BaseModel):
    text: str
    settings: dict | None = None
    grid: dict | None = None


class SaveIn(BaseModel):
    text: str
    settings: dict | None = None
    origin: str = "manual"
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    rationale: str | None = None


class IdsIn(BaseModel):
    ids: list[int]


class SubmittedIn(BaseModel):
    ids: list[int]
    submitted: bool = True


class CombineIn(BaseModel):
    ids: list[int]
    method: str = "equal"


class ExportIn(BaseModel):
    ids: list[int]
    format: str = "json"
    batch: int = 10


class ImportIn(BaseModel):
    text: str
    format: str = "auto"
    mark_submitted: bool = False


class JobIn(BaseModel):
    kind: str
    config: dict = Field(default_factory=dict)


class FieldsImportIn(BaseModel):
    text: str | None = None
    rows: list[dict] | None = None


# --------------------------------------------------------------------------- meta


@app.get("/api/status")
def status() -> dict:
    ws = ws_()
    s = ws.status()
    s["warm"] = STATE.get("warm", False)
    s["jobs_running"] = mgr().running()
    s["defaults"] = SimSettings().to_dict()
    s["universe_map"] = universes_for(ws.panel)
    return s


@app.get("/api/catalog")
def catalog() -> dict:
    ws = ws_()
    c = catalog_json()
    local = ws.local_fields()
    from ..engine.ops import local_operator_names
    lops = local_operator_names()
    for o in c["operators"]:
        o["local"] = o["name"] in lops
    for f in c["fields"]:
        f["available"] = f["id"] in local
    c["settings_options"] = {
        "regions": list(REGIONS), "universes": list(UNIVERSES_BRAIN), "delays": [0, 1],
        "neutralizations": list(NEUTRALIZATIONS_ALL), "neutralizations_local": list(NEUTRALIZATIONS_LOCAL),
        "pasteurization": ["ON", "OFF"], "nanHandling": ["OFF", "ON"], "unitHandling": ["VERIFY"],
    }
    return c


@app.post("/api/parse")
def parse_ep(body: TextIn) -> dict:
    return ws_().analyze(body.text)


@app.post("/api/explain")
def explain_ep(body: TextIn) -> dict:
    try:
        node = lower_text(body.text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from None
    return {"description": description(node), "tags": classify(node)}


# --------------------------------------------------------------------------- simulation


@app.post("/api/simulate")
def simulate_ep(body: SimIn) -> dict:
    return ws_().simulate(body.text, body.settings, extras=body.extras, rationale=body.rationale)


@app.post("/api/extras")
def extras_ep(body: SimIn) -> dict:
    return ws_().extras(body.text, body.settings)


@app.post("/api/sweep")
def sweep_ep(body: SweepIn):
    ws = ws_()

    def gen():
        for ev in ws.sweep(body.text, body.settings, body.grid):
            yield orjson.dumps(ev, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/doctor")
def doctor_ep(body: SimIn) -> dict:
    return ws_().doctor(body.text, body.settings)


# --------------------------------------------------------------------------- library


@app.get("/api/alphas")
def list_alphas(search: str = "", status: str = "", origin: str = "", family: str = "", category: str = "",
                submitted: bool | None = None, starred: bool | None = None, local: bool | None = None,
                min_sharpe: float | None = None, min_fitness: float | None = None, tag: str = "",
                job_id: int | None = None, sort: str = "fitness", desc: bool = True, limit: int = 200,
                offset: int = 0) -> dict:
    return ws_().store.list_alphas(search=search, status=status, origin=origin, family=family, category=category,
                                   submitted=submitted, starred=starred, local=local, min_sharpe=min_sharpe,
                                   min_fitness=min_fitness, tag=tag, job_id=job_id, sort=sort, desc=desc,
                                   limit=min(limit, 5000), offset=offset)


@app.get("/api/alphas/facets")
def facets() -> dict:
    return ws_().store.facet_counts()


@app.get("/api/alphas/{alpha_id}")
def alpha_detail(alpha_id: int) -> dict:
    d = ws_().alpha_detail(alpha_id)
    if d is None:
        raise HTTPException(404, "alpha not found")
    return d


@app.post("/api/alphas")
def save_alpha(body: SaveIn) -> dict:
    return ws_().save_alpha(body.text, body.settings, origin=body.origin, tags=body.tags, notes=body.notes,
                            rationale=body.rationale)


@app.patch("/api/alphas/{alpha_id}")
def patch_alpha(alpha_id: int, fields: dict) -> dict:
    ws = ws_()
    if "submitted" in fields:
        ws.set_submitted(alpha_id, bool(fields.pop("submitted")))
    ws.store.update_alpha(alpha_id, fields)
    return {"ok": True, "alpha": ws.store.get_alpha(alpha_id)}


@app.post("/api/alphas/delete")
def delete_alphas(body: IdsIn) -> dict:
    ws = ws_()
    n = ws.store.delete_alphas(body.ids)
    for i in body.ids:
        ws.corr_lib.remove(i)
        ws.corr_sub.remove(i)
    return {"ok": True, "deleted": n}


@app.post("/api/alphas/submitted")
def mark_submitted(body: SubmittedIn) -> dict:
    ws = ws_()
    for i in body.ids:
        ws.set_submitted(i, body.submitted)
    return {"ok": True}


@app.post("/api/correlation")
def correlation(body: IdsIn) -> dict:
    return ws_().correlation_matrix(body.ids)


@app.post("/api/combine")
def combine(body: CombineIn) -> dict:
    return ws_().combine(body.ids, body.method)


@app.post("/api/export")
def export(body: ExportIn):
    txt = ws_().export(body.ids, body.format, body.batch)
    return PlainTextResponse(txt)


@app.post("/api/import")
def import_ep(body: ImportIn) -> dict:
    return ws_().import_results(body.text, body.format, body.mark_submitted)


@app.get("/api/calibration")
def calibration() -> dict:
    ws = ws_()
    out = ws.calibration()
    out["bandit"] = Bandit(ws.store).table()
    return out


# --------------------------------------------------------------------------- catalog editing


@app.get("/api/templates")
def templates() -> list:
    return templates_summary(ws_().local_fields())


@app.post("/api/templates")
def save_template(t: dict) -> dict:
    try:
        return save_user_template(t)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from None


@app.delete("/api/templates/{tid}")
def del_template(tid: str) -> dict:
    return {"ok": delete_user_template(tid)}


@app.get("/api/alpha101")
def alpha101() -> list:
    return load_alpha101()


@app.post("/api/catalog/fields/import")
def fields_import(body: FieldsImportIn) -> dict:
    rows = body.rows or []
    if body.text:
        t = body.text.strip()
        if t.startswith("[") or t.startswith("{"):
            data = orjson.loads(t)
            rows = data if isinstance(data, list) else data.get("fields") or data.get("results") or []
        else:
            import csv
            import io
            reader = csv.DictReader(io.StringIO(t), dialect=csv.Sniffer().sniff(t[:2000], delimiters=",;\t"))
            for r in reader:
                low = {str(k).strip().lower(): v for k, v in r.items()}
                rows.append({"id": low.get("id") or low.get("field") or low.get("name") or low.get("field id"),
                             "description": low.get("description"), "dataset": low.get("dataset") or
                             low.get("dataset id"), "type": low.get("type"), "category": low.get("category"),
                             "coverage": low.get("coverage")})
    n = import_user_fields(rows)
    return {"ok": True, "imported": n}


@app.get("/api/checks")
def get_checks() -> dict:
    return load_checks_config()


@app.post("/api/checks")
def set_checks(cfg: dict) -> dict:
    out = save_checks_config(cfg)
    ws_().reload_checks()
    return out


@app.get("/api/settings")
def get_settings() -> dict:
    return ws_().settings


@app.post("/api/settings")
def set_settings(updates: dict) -> dict:
    ws = ws_()
    out = ws.update_settings(updates)
    if {"workers", "worker_cache_mb", "active_dataset", "is_years_warmup", "os_years"} & set(updates):
        mgr().reset_pool()
    return out


# --------------------------------------------------------------------------- idea forge


class IdeaIn(BaseModel):
    text: str
    families: list[str] | None = None
    horizon: str | None = None
    settings: dict | None = None


@app.post("/api/forge/interpret")
def forge_interpret(body: IdeaIn) -> dict:
    """Instant, offline reading of an idea (no simulation): hypothesis, matched templates, first drafts."""
    from ..gen.idea import FAMILY_LABEL, apply_overrides, interpret, preview
    from ..gen.templates import load_templates
    from ..jobs.miners import BASE_SETTINGS

    ws = ws_()
    local = ws.local_fields()
    spec = apply_overrides(interpret(body.text, local), local, body.families, body.horizon)
    out = spec.to_json()
    base = {**BASE_SETTINGS, **(body.settings or {})}
    out["preview"] = preview(spec, local, base) if body.text.strip() else []
    by_id = {t.id: t for t in load_templates()}
    out["templates"] = [{"id": tid, "idea": by_id[tid].idea, "rationale": by_id[tid].rationale, "score": s}
                        for tid, s in spec.templates[:5] if tid in by_id]
    out["all_families"] = [{"id": k, "label": v} for k, v in FAMILY_LABEL.items()]
    return out


# --------------------------------------------------------------------------- jobs / data


@app.post("/api/jobs")
def start_job(body: JobIn) -> dict:
    try:
        jid = mgr().start(body.kind, body.config)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"ok": True, "id": jid}


@app.get("/api/jobs")
def list_jobs() -> dict:
    running = {j["id"]: j for j in mgr().running()}
    rows = ws_().store.list_jobs(100)
    for r in rows:
        if r["id"] in running:
            r.update(running[r["id"]])
    return {"jobs": rows}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int) -> dict:
    j = ws_().store.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    for r in mgr().running():
        if r["id"] == job_id:
            j.update(r)
    return j


@app.post("/api/jobs/{job_id}/{action}")
def job_action(job_id: int, action: str) -> dict:
    return {"ok": mgr().control(job_id, action)}


@app.get("/api/reengineer/{job_id}")
def reengineer_report(job_id: int) -> dict:
    """Latest report of a re-engineer job (diagnosis, stage log, lineage, final holdout comparison)."""
    rep = ws_().store.kv_get(f"reengineer:{job_id}")
    if rep is None:
        raise HTTPException(404, "no re-engineer report for this job")
    return rep


@app.get("/api/bandit")
def bandit() -> list:
    return Bandit(ws_().store).table()


@app.get("/api/dashboard")
def dashboard() -> dict:
    ws = ws_()
    st = ws.store
    top = st.list_alphas(status="PASS", sort="pass_prob", limit=60)["rows"]
    top.sort(key=lambda a: (-(a.get("robust") or 0), -(a.get("pass_prob") or 0)))
    top = top[:12]
    recent = st.list_alphas(sort="created_at", limit=8)["rows"]
    sub = st.list_alphas(submitted=True, limit=5000)["rows"]
    pyramid: dict[str, int] = {}
    for a in sub:
        s = a.get("settings") or {}
        k = f"{s.get('region', 'USA')}|D{s.get('delay', 1)}|{a.get('category') or 'pv'}"
        pyramid[k] = pyramid.get(k, 0) + 1
    ideas = st.q("SELECT idea, COUNT(*) n, SUM(status_local='PASS') p, AVG(sharpe) s FROM alphas GROUP BY idea")
    return {
        "stats": st.stats(), "top_candidates": top, "recent": recent, "jobs": st.list_jobs(8),
        "pyramid": pyramid, "ideas": [dict(r) for r in ideas], "data": ws.panel.info(),
        "periods": ws.periods.to_dict(ws.panel), "cache": ws.cache.stats(), "bandit": Bandit(st).table(),
    }


@app.post("/api/data/build")
def data_build(cfg: dict | None = None) -> dict:
    jid = mgr().start("data_build", cfg or {})
    return {"ok": True, "id": jid}


@app.get("/api/data/coverage")
def data_coverage() -> dict:
    import numpy as np
    p = ws_().panel
    out = {}
    for f in p.field_names():
        a = p.field(f)
        tail = np.asarray(a[-min(252, p.T):])
        out[f] = round(float(np.isfinite(tail).mean()), 4)
    return {"coverage": out, "info": p.info()}


# --------------------------------------------------------------------------- websocket


@app.websocket("/ws")
async def ws_events(sock: WebSocket):
    await sock.accept()
    bc: Broadcaster = STATE["bc"]
    q = bc.subscribe()
    try:
        await sock.send_text(orjson.dumps({"type": "hello", "jobs": mgr().running()},
                                          option=orjson.OPT_SERIALIZE_NUMPY).decode())
        while True:
            ev = await q.get()
            await sock.send_text(orjson.dumps(ev, option=orjson.OPT_SERIALIZE_NUMPY, default=str).decode())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        bc.unsubscribe(q)


# --------------------------------------------------------------------------- static UI

DIST = config.FRONTEND_DIST


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str, request: Request):
    if path.startswith("api/") or path == "ws":
        raise HTTPException(404)
    f = (DIST / path).resolve()
    if path and f.is_file() and str(f).startswith(str(DIST.resolve())):
        return FileResponse(f)
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return PlainTextResponse("Alpha Foundry API is running. Build the UI with `npm run build` in frontend/ "
                             "(start.bat does this automatically).", status_code=200)


__all__ = ["app", "OPERATORS"]
