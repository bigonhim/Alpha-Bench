"""Quality refinement shared by the miners and the forge: fitness shaping and complex-alpha composition."""

from __future__ import annotations

import time

import numpy as np

from ..fastexpr import analyze, lower_text, to_expr
from ..fastexpr.explain import classify
from ..gen.compose import Component, composites
from ..gen.optimize import pick_best, shaping_variants
from ..gen.templates import Candidate
from ..sim.correlation import CorrelationIndex
from ..sim.quality import grade_at_least, quick_grade, quick_score
from ..sim.robustness import robust_pick
from .manager import JobHandle, JobManager

SWEEP = {"decay": [0, 2, 4, 6, 8, 12], "neutralization": ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]}


def _is(r: dict) -> dict | None:
    return (r.get("metrics") or {}).get("is") if r.get("ok") else None


def optimize(h: JobHandle, ws, mgr: JobManager, items: list[tuple[Candidate, dict]], *, max_shapes: int = 14,
             sweep: bool = True, label: str = "optimizing") -> list[tuple[Candidate, dict]]:
    """Shape each (candidate, IS metrics): one round of rewrites, then a robust decay x neutralization sweep.

    Every candidate comes back, improved or unchanged, with the IS metrics of the version returned."""
    if not items:
        return []
    cfg = ws.checks_cfg
    has_volume = {"volume", "adv20"} <= ws.local_fields()
    local = ws.local_fields()
    jobs: list[tuple[int, object, dict]] = []
    for k, (c, m) in enumerate(items):
        try:
            node = lower_text(c.expr)
        except Exception:  # noqa: BLE001
            continue
        for sh in shaping_variants(node, c.settings, m, has_volume=has_volume, limit=max_shapes):
            e = to_expr(sh.node)
            an = analyze(e, local_fields=local)
            if an.ok and an.local:
                jobs.append((k, sh, {"expr": e, "settings": sh.settings}))
    h.update(phase=f"{label}: {len(jobs)} shaped variants", done=0, total=len(jobs))
    res = mgr.evaluate(h, [p for _, _, p in jobs], "is",
                       on_result=lambda i, r: (h.bump("evaluated"), h.update(done=h.progress.get("done", 0) + 1)))
    tried: dict[int, list] = {}
    for (k, sh, p), r in zip(jobs, res):
        m = _is(r)
        if m:
            tried.setdefault(k, []).append((sh, m, p))
    out: list[tuple[Candidate, dict]] = []
    delay = 1
    for k, (c, m) in enumerate(items):
        delay = int(c.settings.get("delay", 1) or 1)
        best = pick_best(m, [(sh, mm) for sh, mm, _ in tried.get(k, [])], cfg, delay)
        if best is None:
            out.append((c, m))
            continue
        sh, mm = best
        p = next(p for s2, m2, p in tried[k] if s2 is sh)
        out.append((Candidate(expr=p["expr"], settings=p["settings"], template_id=c.template_id, idea=c.idea,
                              category=c.category, horizon=c.horizon, rationale=c.rationale, origin=c.origin,
                              parents=c.parents), mm))
        h.bump("optimized")
    if not sweep:
        return out
    # robust settings sweep on each (possibly reshaped) candidate
    cells = [{"decay": d, "neutralization": n} for d in SWEEP["decay"] for n in SWEEP["neutralization"]]
    payloads, owners = [], []
    for k, (c, _m) in enumerate(out):
        for cell in cells:
            payloads.append({"expr": c.expr, "settings": {**c.settings, **cell}})
            owners.append((k, cell))
    h.update(phase=f"{label}: settings sweep ({len(payloads)} cells)", done=0, total=len(payloads))
    res = mgr.evaluate(h, payloads, "is",
                       on_result=lambda i, r: (h.bump("evaluated"), h.update(done=h.progress.get("done", 0) + 1)))
    by: dict[int, list[dict]] = {}
    for (k, cell), r in zip(owners, res):
        m = _is(r)
        if m:
            by.setdefault(k, []).append({**cell, "truncation": 0.0, "_m": m})
    final = []
    for k, (c, m) in enumerate(out):
        cells_k = by.get(k) or []
        pick = robust_pick(cells_k, lambda x: quick_score(x["_m"], cfg, delay)) if cells_k else None
        if pick and quick_score(pick["_m"], cfg, delay) > quick_score(m, cfg, delay) + 0.03:
            s = {**c.settings, "decay": pick["decay"], "neutralization": pick["neutralization"]}
            final.append((Candidate(expr=c.expr, settings=s, template_id=c.template_id, idea=c.idea,
                                    category=c.category, horizon=c.horizon, rationale=c.rationale, origin=c.origin,
                                    parents=c.parents), pick["_m"]))
        else:
            final.append((c, m))
    return final


# --------------------------------------------------------------------------- compose job


def library_components(ws, config: dict) -> list[Component]:
    """Decorrelation-ready components from the library: local alphas simulated on the active dataset."""
    ids = [int(i) for i in (config.get("alpha_ids") or [])]
    if ids:
        rows = ws.store.list_alphas(ids=ids, limit=len(ids))["rows"]
    else:
        rows = ws.store.list_alphas(local=True, min_sharpe=float(config.get("min_sharpe", 1.0)), sort="quality",
                                    limit=int(config.get("max_library", 40)))["rows"]
        min_grade = str(config.get("min_grade", "C"))
        rows = [r for r in rows if r.get("grade") is None or grade_at_least(r.get("grade"), min_grade)]
    out = []
    per = ws.periods
    for r in rows:
        if not r.get("local"):
            continue
        res = ws.store.get_result(r["id"])
        if not res or res["data_version"] != ws.panel.version:
            continue
        start = ws.panel.index_of(res["dates_start"])
        a = max(0, per.is_start - start)
        b = max(a, per.os_start - start)
        pnl = np.asarray(res["pnl"][a:b], dtype=np.float64)
        out.append(Component(r["expr"], r.get("idea") or r.get("family") or "other", r.get("settings") or {},
                             float(r.get("sharpe") or 0.0), f"#{r['id']}", r["id"], pnl))
    return out


def component_corr(ws, comps: list[Component]) -> np.ndarray:
    n = len(comps)
    C = np.eye(n)
    L = min((len(c.pnl) for c in comps if c.pnl is not None), default=0)
    if L < 60:
        return C
    P = np.vstack([np.asarray(c.pnl[-L:], dtype=np.float64) for c in comps])  # type: ignore[index]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.corrcoef(P))
    return C


def job_compose(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Build complex (multi-statement) alphas from decorrelated library alphas or pasted components."""
    from .miners import base_settings, screen_and_save

    t0 = time.time()
    comps = library_components(ws, config)
    for e in config.get("exprs") or []:
        e = str(e).strip()
        if e:
            an = analyze(e, local_fields=ws.local_fields())
            if an.ok and an.local:
                comps.append(Component(e, classify(an.node)["idea"], dict(base_settings(config)), 1.0))
    missing = [c for c in comps if c.pnl is None]
    if missing:
        h.update(phase=f"simulating {len(missing)} pasted components")
        res = mgr.evaluate(h, [{"expr": c.expr, "settings": c.settings} for c in missing], "is")
        for c, r in zip(missing, res):
            m = _is(r)
            if m:
                c.sharpe = float(m.get("sharpe") or 0.0)
                c.pnl = np.frombuffer(r["pnl_is"], dtype=np.float32).astype(np.float64) if r.get("pnl_is") else None
        comps = [c for c in comps if c.pnl is not None and c.sharpe > 0]
    if len(comps) < 2:
        raise ValueError("Composition needs at least two simulated components: mine or save some alphas first, "
                         "or paste component expressions.")
    corr = component_corr(ws, comps)
    h.update(phase=f"composing from {len(comps)} components")
    progs = composites(comps, corr, local_fields=ws.local_fields(), has_volume={"volume", "adv20"} <= ws.local_fields(),
                       k_max=int(config.get("max_components", 3)), limit=int(config.get("n", 30)))
    bs = base_settings(config)
    cands = []
    for p in progs:
        neut = [str((c.settings or {}).get("neutralization", bs["neutralization"])).upper() for c in p.parts]
        s = {**bs, "decay": 0, "neutralization": max(set(neut), key=neut.count) if neut else bs["neutralization"]}
        fam = p.parts[0].family if p.parts else "other"
        cands.append(Candidate(expr=p.text, settings=s, idea=fam, rationale=p.label, origin="compose",
                               parents=[c.alpha_id for c in p.parts if c.alpha_id]))
    h.stats["components"] = len(comps)
    screen_and_save(h, ws, mgr, cands, {**config, "save_min_sharpe": float(config.get("save_min_sharpe", 1.0)),
                                        "halving": False, "tags": ["complex", "compose"]},
                    job_index=CorrelationIndex(ws._corr_dates()), phase="composites")
    h.stats["elapsed_s"] = round(time.time() - t0, 1)


__all__ = ["component_corr", "job_compose", "library_components", "optimize", "quick_grade"]
