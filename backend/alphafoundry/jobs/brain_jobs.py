"""Jobs that talk to BRAIN: simulate alphas, sync the data catalog, import your alphas, and a closed mining loop
with BRAIN as the judge. Nothing here ever submits an alpha."""

from __future__ import annotations

import random
import time

from ..brainio.brain_api import BrainError, summarize_alpha
from ..brainio.service import get_service
from ..catalog import field_map, import_user_fields
from ..fastexpr import analyze, lower_text, to_expr
from ..gen.brainfields import field_candidates
from ..gen.doctor import diagnose
from ..sim.quality import grade_at_least
from .manager import JobHandle, JobManager
from .miners import base_settings

BRAIN_CHECKS = {"LOW_SHARPE", "LOW_FITNESS", "HIGH_TURNOVER", "LOW_TURNOVER", "CONCENTRATED_WEIGHT",
                "LOW_SUB_UNIVERSE_SHARPE", "SELF_CORRELATION"}


def _scope(svc, config: dict) -> tuple[str, int, str]:
    s = svc.settings
    return (str(config.get("region") or s["region"]), int(config.get("delay", s["delay"])),
            str(config.get("universe") or s["universe"]))


def job_brain_sim(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Simulate library alphas and/or pasted expressions on BRAIN and store the real results."""
    svc = get_service(ws)
    items = []
    for aid in config.get("alpha_ids") or []:
        a = ws.store.get_alpha(int(aid))
        if a:
            items.append({"alpha_id": a["id"], "expr": a["expr"], "settings": a.get("settings") or {}})
    bs = base_settings(config)
    for e in config.get("exprs") or []:
        e = str(e).strip()
        if e and not e.startswith("#"):
            items.append({"alpha_id": None, "expr": e, "settings": bs})
    if not items:
        raise ValueError("Nothing to simulate: pass alpha ids or expressions.")
    svc.simulate_alphas(h, items, check=config.get("check"))


def job_brain_fields(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Import BRAIN's data catalog (datasets + fields with coverage and usage counts) for one scope."""
    svc = get_service(ws)
    svc.ensure()
    region, delay, universe = _scope(svc, config)
    h.update(phase=f"listing datasets ({region} {universe} D{delay})")
    sets = svc.client.datasets(region, delay, universe)
    slim = [{"id": d.get("id"), "name": d.get("name"), "category": (d.get("category") or {}).get("id")
             if isinstance(d.get("category"), dict) else d.get("category"),
             "description": (d.get("description") or "")[:300], "coverage": d.get("coverage"),
             "value_score": d.get("valueScore"), "alpha_count": d.get("alphaCount"),
             "user_count": d.get("userCount"), "field_count": d.get("fieldCount")} for d in sets]
    ws.store.kv_set("brain_datasets", {"scope": [region, delay, universe], "datasets": slim,
                                       "synced": time.strftime("%Y-%m-%d %H:%M")})
    wanted = [str(x) for x in (config.get("datasets") or [])]
    if not wanted or config.get("auto"):
        ranked = sorted(slim, key=lambda d: -(d.get("value_score") or 0))
        wanted += [d["id"] for d in ranked[: int(config.get("auto_n", 12))]]
        wanted += [d for d in ("pv1", "fundamental6") if any(s["id"] == d for s in slim)]
    wanted = list(dict.fromkeys(w for w in wanted if w))
    cap = int(config.get("max_fields_per_dataset", 600))
    names = {d["id"]: d.get("name") for d in slim}
    total = 0
    for k, ds in enumerate(wanted):
        h.check()
        h.update(phase=f"fields of {ds}", done=k, total=len(wanted))
        try:
            rows = svc.client.datafields(region, delay, universe, dataset_id=ds, limit=cap)
        except BrainError as e:
            h.bump("errors")
            h.event("warning", {"message": f"{ds}: {e}"})
            continue
        batch = []
        for f in rows:
            cat = f.get("category")
            batch.append({"id": f.get("id"), "description": f.get("description"),
                          "dataset": (f.get("dataset") or {}).get("id") or ds, "dataset_name": names.get(ds),
                          "category": cat.get("id") if isinstance(cat, dict) else cat,
                          "subcategory": (f.get("subcategory") or {}).get("id")
                          if isinstance(f.get("subcategory"), dict) else f.get("subcategory"),
                          "type": f.get("type"), "coverage": f.get("coverage"), "alpha_count": f.get("alphaCount"),
                          "user_count": f.get("userCount"), "region": region, "delay": delay, "universe": universe,
                          "source_brain": True})
        total += import_user_fields(batch)
        h.stats["fields"] = total
    h.update(phase="done", done=len(wanted), total=len(wanted))


def job_brain_sync(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Import your BRAIN alphas and results; submitted ones feed the local SELF_CORRELATION check."""
    svc = get_service(ws)
    svc.ensure()
    params = {k: config[k] for k in ("stage", "status") if config.get(k)}
    h.update(phase="listing your BRAIN alphas")
    alphas = svc.client.user_alphas(params, limit=int(config.get("limit", 300)))
    h.update(phase=f"importing {len(alphas)} alphas", done=0, total=len(alphas))
    for k, a in enumerate(alphas):
        h.check()
        s = summarize_alpha(a)
        if not s["expr"]:
            continue
        aid = svc.local_alpha_for(s["expr"], s["settings"], origin="brain")
        if aid is None:
            h.bump("errors")
            continue
        submitted = str(s.get("stage") or "").upper() in ("OS", "PROD") or str(s.get("status")).upper() == "ACTIVE"
        pnl = None
        if submitted and config.get("pnl", True):
            try:
                pnl = svc.client.get_pnl(s["brain_id"])
            except BrainError:
                pnl = None
        row = svc.record_result(aid, s, pnl)
        if submitted:
            ws.set_submitted(aid, True)
            ws.store.update_alpha(aid, {"status_brain": "submitted"})
            h.bump("submitted")
        h.bump("saved")
        h.add_result(row)
        h.update(done=k + 1)
    ws.refit_calibration()


def _failed_checks(row: dict) -> list[str]:
    return [c for c in row.get("failed") or [] if c in BRAIN_CHECKS]


def job_brain_mine(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Closed loop with BRAIN as the judge: the best untested local alphas and catalog-field ideas go to BRAIN;
    near misses are repaired from BRAIN's own failing checks and re-simulated."""
    svc = get_service(ws)
    svc.ensure()
    rng = random.Random(config.get("seed", int(time.time())))
    budget = min(int(config.get("budget", 60)), svc.budget_left())
    if budget <= 0:
        raise BrainError("No BRAIN simulations left in today's budget.")
    deadline = time.time() + float(config.get("time_limit_min", 60)) * 60
    sources = set(config.get("sources") or ["library", "brain_fields"])
    bs = base_settings(config)
    near = {"sharpe": 1.0, "fitness": 0.7, **(config.get("near_miss") or {})}
    queue: list[dict] = []
    if "library" in sources:
        min_grade = str(config.get("min_grade", "B"))
        rows = ws.store.list_alphas(local=True, sort="quality", limit=400)["rows"]
        for r in rows:
            if r.get("status_brain") not in (None, "", "untested") or r.get("data_source") == "demo":
                continue
            ok = grade_at_least(r["grade"], min_grade) if r.get("grade") else r.get("status_local") == "PASS"
            if ok:
                queue.append({"alpha_id": r["id"], "expr": r["expr"], "settings": r.get("settings") or bs})
        queue = queue[: max(1, int(budget * 0.6))] if "brain_fields" in sources else queue
    if "brain_fields" in sources:
        taken = {q["expr"] for q in queue}
        n = max(0, budget - len(queue))
        for e, label, f in field_candidates(ws.local_fields(), n=n, rng=rng, exclude=taken):
            queue.append({"alpha_id": None, "expr": e, "settings": {**bs, "decay": 4}, "tags": ["brain-field"],
                          "notes": label})
        if not any(q.get("tags") for q in queue) and not field_map_has_brain():
            h.event("warning", {"message": "No BRAIN catalog fields imported yet: run 'Sync data catalog' first."})
    queue = queue[:budget]
    used = 0
    rounds = int(config.get("refine_rounds", 2))
    for rnd in range(rounds + 1):
        if not queue or time.time() > deadline or used >= budget:
            break
        h.update(phase=f"round {rnd + 1}: {len(queue)} alphas to BRAIN")
        rows = svc.simulate_alphas(h, queue[: budget - used], check=config.get("check"))
        used += len(rows)
        if rnd == rounds:
            break
        nxt: list[dict] = []
        for q, row in zip(queue, rows):
            if row.get("passed") or row.get("sharpe") is None:
                continue
            if (row.get("sharpe") or 0) < near["sharpe"] and (row.get("fitness") or 0) < near["fitness"]:
                continue
            try:
                node = lower_text(q["expr"])
            except Exception:  # noqa: BLE001
                continue
            m = {"sharpe": row.get("sharpe"), "fitness": row.get("fitness"), "turnover": row.get("turnover")}
            for f in diagnose(node, q["settings"], _failed_checks(row) or ["LOW_FITNESS"], m)[:4]:
                e = to_expr(f.node)
                if analyze(e).ok:
                    nxt.append({"alpha_id": None, "expr": e, "settings": f.settings,
                                "tags": ["brain-refined", f"refined-from-{row.get('id')}"],
                                "notes": f"BRAIN refinement of #{row.get('id')}: {f.label}"})
        rng.shuffle(nxt)
        queue = nxt
    h.update(phase="done")


def field_map_has_brain() -> bool:
    return any(f.get("source_brain") for f in field_map().values())


BRAIN_JOB_KINDS = {"brain_sim": job_brain_sim, "brain_fields": job_brain_fields, "brain_sync": job_brain_sync,
                   "brain_mine": job_brain_mine}

__all__ = ["BRAIN_JOB_KINDS"]
