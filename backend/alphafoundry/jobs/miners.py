"""Job kinds: batch, templates, alpha101, grammar, gp, settings_opt, brain_only, automine, forge."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from ..fastexpr import analyze, lower_text, to_expr
from ..fastexpr.explain import classify, description
from ..gen.bandit import Bandit
from ..gen.doctor import diagnose
from ..gen.gp import GPConfig, GPEngine, Individual
from ..gen.grammar import Grammar, GrammarConfig, is_degenerate
from ..gen.templates import Candidate, alpha101_candidates, expand, load_templates, settings_grid, template_candidates
from ..sim.checks import FAIL, HARD_CHECKS, run_checks
from ..sim.correlation import CorrelationIndex
from ..sim.quality import assess, grade_at_least, quick_grade, quick_score
from ..sim.simulator import SimSettings
from .manager import JobHandle, JobManager

BASE_SETTINGS = {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY",
                 "truncation": 0.08, "pasteurization": "ON", "nanHandling": "OFF"}


def base_settings(config: dict) -> dict:
    return {**BASE_SETTINGS, **(config.get("settings") or {})}


# --------------------------------------------------------------------------- helpers


def _pnl(res: dict, key: str) -> np.ndarray:
    b = res.get(key)
    return np.frombuffer(b, dtype=np.float32) if b else np.zeros(0, np.float32)


def _dates(ws, start: str | None, n: int) -> np.ndarray:
    if not start or n == 0:
        return np.zeros(0, dtype="datetime64[D]")
    i = ws.panel.index_of(start)
    return ws.panel.dates[i:i + n]


def is_gate(m: dict | None, ws, relaxed: float = 1.0) -> bool:
    if not m:
        return False
    cfg = ws.checks_cfg
    th = cfg["delay1"]
    return (m.get("sharpe", 0) >= th["sharpe_min"] * relaxed and m.get("fitness", 0) >= th["fitness_min"] * relaxed
            and cfg["turnover_min"] <= m.get("turnover", 0) <= cfg["turnover_max"]
            and m.get("max_weight", 0) <= cfg["max_weight"] + 1e-9)


def score_result(ws, res: dict) -> tuple[dict, float, int]:
    """BRAIN-style checks, calibrated pass likelihood and hard-failure count for a full evaluation."""
    metrics = res["metrics"]
    m_is = metrics.get("is") or {}
    pnl_is = _pnl(res, "pnl_is")
    d_is = _dates(ws, res.get("pnl_is_start"), len(pnl_is))
    ex = res.get("extras") or {}
    sub = None
    if ex.get("sub_sharpe") is not None:
        sub = {"sharpe": ex["sub_sharpe"], "sub_size": ex["sub_size"], "univ_size": ex["univ_size"],
               "name": ex.get("sub_universe")}
    self_corr = ws.corr_sub.max_corr(d_is, pnl_is) if ws.corr_sub is not None and len(ws.corr_sub) else None
    s = SimSettings.from_dict(res["settings"])
    checks = run_checks(metrics, s.delay, ws.checks_cfg, sub=sub, self_corr=self_corr, stability=ex.get("stability"),
                        yearly=res.get("yearly"), recent2y_sharpe=res.get("recent2y"))
    failed_hard = sum(1 for c in checks["checks"] if c["name"] in HARD_CHECKS and c["result"] == FAIL)
    feats = {**m_is, "os_sharpe": (metrics.get("os") or {}).get("sharpe"), "sub_sharpe": ex.get("sub_sharpe"),
             "complexity": res.get("size")}
    return checks, ws.cal.predict(feats, failed_hard), failed_hard


def quality_of(ws, res: dict, checks: dict) -> dict:
    """Grade (A-D) and quality score of a full evaluation."""
    m_is = (res.get("metrics") or {}).get("is") or {}
    s = SimSettings.from_dict(res["settings"])
    return assess(res.get("metrics") or {}, checks, ws.checks_cfg, delay=s.delay, extras=res.get("extras"),
                  yearly=res.get("yearly"), complexity=res.get("size"),
                  expected_brain_sharpe=ws.cal.trusted_brain_sharpe(m_is.get("sharpe") or 0.0))


def persist(h: JobHandle, ws, cand: Candidate, res: dict, job_index: CorrelationIndex | None = None,
            corr_cap: float | None = None, tags: list[str] | None = None, min_grade: str | None = None) -> dict | None:
    """Turn a full evaluation into checks + a saved alpha. Returns a summary row or None if rejected.

    ``min_grade`` drops alphas below that quality grade (A best ... D) instead of saving them."""
    if not res.get("ok"):
        return None
    metrics = res["metrics"]
    m_is = metrics.get("is") or {}
    pnl_is = _pnl(res, "pnl_is")
    d_is = _dates(ws, res.get("pnl_is_start"), len(pnl_is))
    if job_index is not None and corr_cap is not None and len(job_index):
        mc = job_index.max_corr(d_is, pnl_is)
        if mc and mc["max_corr"] >= corr_cap:
            h.bump("rejected_corr")
            return None
    ex = res.get("extras") or {}
    s = SimSettings.from_dict(res["settings"])
    checks, pass_prob, _ = score_result(ws, res)
    qual = quality_of(ws, res, checks)
    if min_grade and not grade_at_least(qual["grade"], min_grade):
        h.bump("rejected_quality")
        return None
    node = lower_text(cand.expr)
    tg = classify(node)
    top = ws.corr_lib.top(d_is, pnl_is, n=1) if ws.corr_lib is not None else []
    rec = {
        "expr": cand.expr, "canon": res["canonical"], "canon_hash": res["canon_hash"], "settings": s.to_dict(),
        "settings_key": s.key(), "origin": cand.origin, "family": cand.idea or tg["idea"],
        "idea": cand.idea if cand.idea and cand.idea != "alpha101" else tg["idea"],
        "category": cand.category or tg["category"], "horizon": cand.horizon or tg["horizon"],
        "template_id": cand.template_id, "parents": cand.parents, "local": 1, "complexity": res.get("size"),
        "description": description(node, cand.rationale), "job_id": h.id,
        "sharpe": m_is.get("sharpe"), "fitness": m_is.get("fitness"), "turnover": m_is.get("turnover"),
        "returns": m_is.get("returns"), "drawdown": m_is.get("drawdown"), "margin": m_is.get("margin_bps"),
        "os_sharpe": (metrics.get("os") or {}).get("sharpe"), "sub_sharpe": ex.get("sub_sharpe"),
        "max_corr": top[0]["corr"] if top else None, "pass_prob": pass_prob, "status_local": checks["status"],
        "robust": int(bool(checks.get("robust"))), "failed": checks["failed"],
        "quality": qual["score"], "grade": qual["grade"], "data_source": ws.panel.source,
    }
    tags = list(tags or [])
    if ";" in cand.expr.strip().rstrip(";") and "complex" not in tags:
        tags.append("complex")
    if tags:
        rec["tags"] = ",".join(tags)
    payload = {"metrics": metrics, "yearly": res.get("yearly"), "checks": checks, "extras": ex,
               "sector_pnl": res.get("sector_pnl"), "top_names": res.get("top_names"), "pass_prob": pass_prob,
               "local_universe": res.get("local_universe"), "universe_size": res.get("universe_size"),
               "quality": qual}
    pnl = _pnl(res, "pnl")
    aid = ws.save_result_record(rec, payload, pnl, res.get("dates_start"))
    if job_index is not None:
        job_index.add(aid, d_is, pnl_is, m_is.get("sharpe", 0.0))
    h.bump("saved")
    if checks["status"] == "PASS":
        h.bump("passed")
    h.bump(f"grade_{qual['grade'].lower()}")
    row = {"id": aid, "expr": cand.expr, "sharpe": m_is.get("sharpe"), "fitness": m_is.get("fitness"),
           "turnover": m_is.get("turnover"), "returns": m_is.get("returns"), "status": checks["status"],
           "pass_prob": pass_prob, "os_sharpe": rec["os_sharpe"], "idea": rec["idea"], "origin": cand.origin,
           "settings": s.to_dict(), "failed": checks["failed"], "grade": qual["grade"], "quality": qual["score"],
           "reasons": qual["reasons"][:3]}
    h.add_result(row)
    return row


GRADE_REWARD = {"A": 1.0, "B": 0.8, "C": 0.3, "D": 0.0}


def _reward(bandit: Bandit | None, c: Candidate, grade: str) -> None:
    if bandit is None:
        return
    for arm in (f"idea:{c.idea or 'other'}", f"tmpl:{c.template_id}" if c.template_id else None):
        if arm:
            bandit.update(arm, GRADE_REWARD.get(grade, 0.0))


def screen_and_save(h: JobHandle, ws, mgr: JobManager, cands: list[Candidate], config: dict,
                    job_index: CorrelationIndex | None = None, span: str = "is", phase: str = "screening",
                    bandit: Bandit | None = None) -> list[tuple[Candidate, dict]]:
    """Screen in-sample, shape the promising ones for fitness, fully evaluate the best and save by grade.

    Config keys: save_min_sharpe (IS floor to consider), optimize (bool), optimize_top, full_max,
    save_min_grade (A..D; default B = passes every local check), tags, dedup_corr/corr_cap, halving."""
    from .refine import optimize

    cfg = ws.checks_cfg
    save_min = float(config.get("save_min_sharpe", 1.0))
    min_grade = str(config.get("save_min_grade", "B")).upper()
    do_opt = bool(config.get("optimize", True))
    corr_cap = float(config.get("corr_cap", 0.7)) if config.get("dedup_corr", True) else None
    delay = int((cands[0].settings if cands else {}).get("delay", 1) or 1)
    if config.get("halving", True) and len(cands) >= 24 and span == "is":
        # successive halving: cheap 3-year pre-screen, then full IS only for the better half
        h.update(phase=f"{phase} (3-year pre-screen)", done=0, total=len(cands))

        def on0(i, r):
            h.bump("evaluated")
            h.update(done=h.progress.get("done", 0) + 1)

        res0 = mgr.evaluate(h, [{"expr": c.expr, "settings": c.settings} for c in cands], "screen", on_result=on0)
        scored = []
        for c, r in zip(cands, res0):
            m = (r.get("metrics") or {}).get("is") or {} if r.get("ok") else None
            if m is None:
                h.bump("brain_only" if r.get("brain_only") else "errors")
                continue
            scored.append((c, m))
        floor = max(0.3, 0.5 * save_min)
        scored.sort(key=lambda cm: -quick_score(cm[1], cfg, delay))
        keep = [c for c, m in scored if (m.get("sharpe") or -9) >= floor]
        keep = keep or [c for c, _ in scored[: max(1, len(scored) // 4)]]
        kept = {id(c) for c in keep}
        for c, _m in scored:
            if id(c) not in kept:
                _reward(bandit, c, "D")
        cands = keep
    payloads = [{"expr": c.expr, "settings": c.settings} for c in cands]
    h.update(phase=phase, done=0, total=len(payloads))
    stage1: list[tuple[Candidate, dict]] = []

    def on1(i, r):
        h.bump("evaluated")
        if not r.get("ok"):
            h.bump("brain_only" if r.get("brain_only") else "errors")
        h.update(done=h.progress.get("done", 0) + 1)

    res1 = mgr.evaluate(h, payloads, span, on_result=on1)
    for c, r in zip(cands, res1):
        if r.get("ok"):
            stage1.append((c, r))
            _reward(bandit, c, quick_grade((r.get("metrics") or {}).get("is"), cfg, delay))
    promote = "D" if min_grade == "D" else ("C" if do_opt or min_grade == "C" else "B")
    final_floor = "B" if min_grade in ("A", "B") else min_grade
    pool = []
    for c, r in stage1:
        m = (r.get("metrics") or {}).get("is") or {}
        if (m.get("sharpe") or -9) >= save_min and grade_at_least(quick_grade(m, cfg, delay), promote):
            pool.append((c, m))
    if not pool:
        return stage1
    pool.sort(key=lambda cm: -quick_score(cm[1], cfg, delay))
    if do_opt:
        n_opt = int(config.get("optimize_top", 6))
        shaped = optimize(h, ws, mgr, pool[:n_opt], max_shapes=int(config.get("optimize_shapes", 14)),
                          sweep=bool(config.get("optimize_sweep", True)), label=f"{phase}: shaping")
        pool = shaped + pool[n_opt:]
    pool = [(c, m) for c, m in pool if grade_at_least(quick_grade(m, cfg, delay), final_floor)]
    seen, promising = set(), []
    for c, m in sorted(pool, key=lambda cm: -quick_score(cm[1], cfg, delay)):
        k = c.expr + str(sorted(c.settings.items()))
        if k not in seen:
            seen.add(k)
            promising.append(c)
    promising = promising[: int(config.get("full_max", 12))]
    if not promising:
        return stage1
    h.update(phase="full evaluation", done=0, total=len(promising))
    res2 = mgr.evaluate(h, [{"expr": c.expr, "settings": c.settings, "extras": config.get("final_extras", True)}
                            for c in promising], "full",
                        on_result=lambda i, r: h.update(done=h.progress.get("done", 0) + 1))
    order = sorted(range(len(promising)), key=lambda k: -quick_score(((res2[k].get("metrics") or {}).get("is")), cfg,
                                                                    delay) if res2[k].get("ok") else 9)
    for k in order:
        h.check()
        persist(h, ws, promising[k], res2[k], job_index, corr_cap, tags=config.get("tags"), min_grade=min_grade)
    return stage1


# --------------------------------------------------------------------------- job kinds


def job_batch(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    exprs = config.get("exprs") or []
    if isinstance(exprs, str):
        exprs = exprs.splitlines()
    exprs = [e.strip() for e in exprs if e.strip() and not e.strip().startswith("#")]
    bs = base_settings(config)
    cands = [Candidate(expr=e, settings=bs, origin="batch") for e in exprs]
    config = {**config, "save_min_sharpe": config.get("save_min_sharpe", -99), "dedup_corr": False,
              "save_min_grade": config.get("save_min_grade", "D"), "optimize": config.get("optimize", False),
              "full_max": max(len(exprs), 1)}
    brain_only = []
    local = []
    for c in cands:
        an = analyze(c.expr, local_fields=ws.local_fields())
        if an.ok and not an.local:
            brain_only.append(c)
        elif an.ok:
            local.append(c)
        else:
            h.bump("errors")
    for c in brain_only:
        ws.save_alpha(c.expr, c.settings, origin="batch", job_id=h.id, extras=False)
        h.bump("brain_only")
        h.bump("saved")
    screen_and_save(h, ws, mgr, local, config, span="is")


def job_templates(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    rng = random.Random(config.get("seed", int(time.time())))
    tmpls = load_templates()
    ids = set(config.get("template_ids") or [])
    fams = set(config.get("families") or [])
    if ids:
        tmpls = [t for t in tmpls if t.id in ids]
    if fams:
        tmpls = [t for t in tmpls if t.idea in fams or t.category in fams]
    h.update(phase="expanding templates")
    cands = template_candidates(tmpls, base_settings(config), ws.local_fields(),
                                per_template=int(config.get("per_template", 12)),
                                settings_per_expr=int(config.get("settings_per_expr", 2)), rng=rng,
                                settings_override=config.get("settings_grid"))
    rng.shuffle(cands)
    cands = cands[: int(config.get("max_candidates", 400))]
    screen_and_save(h, ws, mgr, cands, config, job_index=CorrelationIndex(ws._corr_dates()), bandit=Bandit(ws.store))


def job_alpha101(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    cands = alpha101_candidates(base_settings(config), ws.local_fields(), config.get("variants"))
    screen_and_save(h, ws, mgr, cands, {**config, "save_min_sharpe": config.get("save_min_sharpe", 0.5),
                                        "save_min_grade": config.get("save_min_grade", "C")},
                    job_index=CorrelationIndex(ws._corr_dates()))


def _grammar(ws, config: dict, rng: random.Random) -> Grammar:
    gc = GrammarConfig(max_depth=int(config.get("max_depth", 4)),
                       category_weights=config.get("category_weights") or {"pv": 0.6, "fundamental": 0.4},
                       allow_trade_when=bool(config.get("allow_trade_when", True)))
    return Grammar(ws.local_fields(), rng, gc, fields_whitelist=config.get("fields") or None)


def job_grammar(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    rng = random.Random(config.get("seed", int(time.time())))
    g = _grammar(ws, config, rng)
    n = int(config.get("n", 300))
    seen: set[str] = set()
    cands = []
    tries = 0
    bs = base_settings(config)
    while len(cands) < n and tries < n * 20:
        tries += 1
        node = g.tree()
        if is_degenerate(node) or node.key in seen:
            continue
        seen.add(node.key)
        s = dict(bs)
        if config.get("randomize_settings", True):
            s["decay"] = rng.choice([0, 2, 4, 6])
            s["neutralization"] = rng.choice(["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"])
        cands.append(Candidate(expr=to_expr(node), settings=s, origin="grammar"))
    screen_and_save(h, ws, mgr, cands, config, job_index=CorrelationIndex(ws._corr_dates()))


def _novelty_fn(ws, extra_index: CorrelationIndex):
    def fn(inds: list[Individual]) -> None:
        for i in inds:
            pnl = getattr(i, "_pnl", None)
            dates = getattr(i, "_dates", None)
            if pnl is None or not len(pnl):
                i.novelty = 0.0
                continue
            best = 0.0
            for idx in (ws.corr_lib, extra_index):
                if idx is not None and len(idx):
                    c = idx.correlations(dates, pnl)
                    if len(c):
                        best = max(best, float(np.max(np.abs(c))))
            i.novelty = 1.0 - best
    return fn


def run_gp(h: JobHandle, ws, mgr: JobManager, config: dict, seeds: list[Individual] | None = None,
           job_index: CorrelationIndex | None = None) -> list[Individual]:
    rng = random.Random(config.get("seed", int(time.time())))
    g = _grammar(ws, config, rng)
    cfg = GPConfig(population=int(config.get("population", 60)), generations=int(config.get("generations", 15)),
                   islands=int(config.get("islands", 2)), max_depth=int(config.get("max_depth", 6)),
                   max_nodes=int(config.get("max_nodes", 30)), min_sharpe=float(config.get("min_sharpe", 1.25)),
                   min_fitness=float(config.get("min_fitness", 1.0)), halving=bool(config.get("halving", True)),
                   seed=rng.randint(0, 10 ** 9))
    hof_index = job_index or CorrelationIndex(ws._corr_dates())
    gates = {"turnover_min": ws.checks_cfg["turnover_min"], "turnover_max": ws.checks_cfg["turnover_max"],
             "max_weight": ws.checks_cfg["max_weight"]}
    deadline = time.time() + float(config.get("time_limit_min", 30)) * 60

    def evaluate(inds: list[Individual], span: str) -> None:
        payloads = [{"expr": i.expr, "settings": i.settings} for i in inds]
        res = mgr.evaluate(h, payloads, span, on_result=lambda k, r: h.bump("evaluated"))
        for i, r in zip(inds, res):
            if r.get("ok"):
                i.metrics = (r.get("metrics") or {}).get("is")
                i._pnl = _pnl(r, "pnl_is")  # type: ignore[attr-defined]
                i._dates = _dates(ws, r.get("pnl_is_start"), len(i._pnl))  # type: ignore[attr-defined]
            else:
                i.metrics = None
                h.bump("errors")

    def on_gen(stat: dict) -> None:
        h.update(phase=f"generation {stat['generation']}/{cfg.generations}", done=stat["generation"],
                 total=cfg.generations, best_fitness=stat["best_fitness"], hof=stat["hof"])
        h.event("gp_generation", {"stat": stat})

    def on_hof(ind: Individual) -> None:
        pnl = getattr(ind, "_pnl", None)
        if pnl is not None and len(pnl):
            hof_index.add(-len(hof_index) - 1, ind._dates, pnl, (ind.metrics or {}).get("sharpe", 0))  # type: ignore

    eng = GPEngine(cfg, g, evaluate, _novelty_fn(ws, hof_index), gates, on_gen,
                   stop=lambda: h.cancelled() or time.time() > deadline, on_hof=on_hof)
    h.update(phase="initial population", done=0, total=cfg.generations)
    hof = eng.run(seeds or [])
    return hof


def _library_seeds(ws, config: dict, n: int) -> list[Individual]:
    rows = ws.store.list_alphas(local=True, min_sharpe=float(config.get("seed_min_sharpe", 0.8)), sort="fitness",
                                limit=n)["rows"]
    out = []
    for r in rows:
        try:
            out.append(Individual(lower_text(r["expr"]), r["settings"], origin="library", alpha_id=r["id"]))
        except Exception:  # noqa: BLE001
            continue
    return out


def job_gp(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    seeds = []
    if config.get("seed_from_library", True):
        seeds += _library_seeds(ws, config, int(config.get("library_seeds", 10)))
    if config.get("seed_exprs"):
        for e in config["seed_exprs"]:
            try:
                seeds.append(Individual(lower_text(e), base_settings(config), origin="seed"))
            except Exception:  # noqa: BLE001
                h.bump("errors")
    for s in seeds:
        s.settings = {**base_settings(config), **(s.settings or {})}
    job_index = CorrelationIndex(ws._corr_dates())
    hof = run_gp(h, ws, mgr, config, seeds)
    finalize_individuals(h, ws, mgr, hof, config, job_index)


def finalize_individuals(h: JobHandle, ws, mgr: JobManager, inds: list[Individual], config: dict,
                         job_index: CorrelationIndex) -> None:
    if not inds:
        return
    cands = [Candidate(expr=i.expr, settings=i.settings, origin=i.origin if i.origin != "library" else "gp",
                       parents=i.parents) for i in inds]
    h.update(phase="final evaluation", done=0, total=len(cands))
    res = mgr.evaluate(h, [{"expr": c.expr, "settings": c.settings, "extras": True} for c in cands], "full",
                       on_result=lambda k, r: h.update(done=h.progress.get("done", 0) + 1))
    for c, r in sorted(zip(cands, res), key=lambda cr: -((cr[1].get("metrics") or {}).get("is") or {}).get("fitness", -9)
                       if cr[1].get("ok") else 9):
        h.check()
        persist(h, ws, c, r, job_index, float(config.get("corr_cap", 0.7)),
                min_grade=str(config.get("save_min_grade", "C")))


def job_settings_opt(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    ids = config.get("alpha_ids") or []
    grid = config.get("grid") or {"decay": [0, 2, 4, 8, 12], "neutralization": ["MARKET", "SECTOR", "INDUSTRY",
                                                                                   "SUBINDUSTRY"],
                                  "truncation": [0.05, 0.08]}
    from ..sim.robustness import robust_pick, sweep_grid
    cells = sweep_grid(grid)
    job_index = CorrelationIndex(ws._corr_dates())
    for k, aid in enumerate(ids):
        h.check()
        a = ws.store.get_alpha(int(aid))
        if not a or not a.get("local"):
            continue
        h.update(phase=f"alpha #{aid} ({k + 1}/{len(ids)})", done=k, total=len(ids))
        payloads = [{"expr": a["expr"], "settings": {**a["settings"], **c}} for c in cells]
        res = mgr.evaluate(h, payloads, "is", on_result=lambda i, r: h.bump("evaluated"))
        scored = []
        for c, r in zip(cells, res):
            if r.get("ok"):
                m = (r.get("metrics") or {}).get("is") or {}
                scored.append({**c, **{k2: m.get(k2) for k2 in ("sharpe", "fitness", "turnover")}})
        cfg = ws.checks_cfg
        best = robust_pick(scored, lambda c: (c["fitness"] or -5) - (0 if cfg["turnover_min"] <= (c["turnover"] or 0)
                                                                     <= cfg["turnover_max"] else 2))
        if best:
            s = {**a["settings"], **{kk: best[kk] for kk in ("decay", "neutralization", "truncation")}}
            cand = Candidate(expr=a["expr"], settings=s, origin="settings_opt", parents=[a["id"]],
                             idea=a.get("idea"), category=a.get("category"))
            r = mgr.evaluate(h, [{"expr": cand.expr, "settings": s, "extras": True}], "full")[0]
            persist(h, ws, cand, r, job_index, None)
            h.event("sweep_result", {"alpha_id": aid, "cells": scored, "best": best})


def job_brain_only(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Generate BRAIN-only candidates (fields not available locally) ranked by bandit priors."""
    rng = random.Random(config.get("seed", int(time.time())))
    local = ws.local_fields()
    bandit = Bandit(ws.store, rng)
    tmpls = load_templates()
    per = int(config.get("per_template", 6))
    made = 0
    h.update(phase="generating BRAIN-only candidates", done=0, total=len(tmpls))
    for k, t in enumerate(tmpls):
        h.check()
        all_exprs = expand(t, local_fields=None, limit=per * 3, rng=rng)
        for e in all_exprs:
            an = analyze(e, local_fields=local)
            if not an.ok or an.local:
                continue
            for s in settings_grid(t, base_settings(config))[:2]:
                r = ws.save_alpha(e, s, origin="brain_only", template_id=t.id, rationale=t.rationale, job_id=h.id,
                                  extras=False)
                if r.get("ok"):
                    made += 1
                    h.bump("saved")
                    h.add_result({"id": r["id"], "expr": e, "status": "UNSCORED", "idea": t.idea,
                                  "origin": "brain_only", "settings": s})
        h.update(done=k + 1)
    # Grammar-built BRAIN-only candidates from user-imported (non-local) fields
    from ..catalog import field_map
    remote = [fid for fid, f in field_map().items() if fid not in local and str(f.get("type")) == "MATRIX"]
    n_extra = int(config.get("n_field_candidates", 40))
    for _ in range(min(n_extra, len(remote) * 4)):
        h.check()
        fid = rng.choice(remote)
        w = rng.choice([5, 20, 60, 120])
        pattern = rng.choice([
            f"rank(ts_zscore(ts_backfill({fid}, 20), {w}))",
            f"group_rank(ts_backfill({fid}, 20), subindustry)",
            f"-rank(ts_delta(ts_backfill({fid}, 20), {w}))",
            f"ts_av_diff(ts_backfill({fid}, 20), {w})",
            f"group_rank(ts_rank(ts_backfill({fid}, 20), {w}), industry)",
        ])
        r = ws.save_alpha(pattern, base_settings(config), origin="brain_only", job_id=h.id, extras=False)
        if r.get("ok"):
            made += 1
            h.bump("saved")
            h.add_result({"id": r["id"], "expr": pattern, "status": "UNSCORED", "origin": "brain_only"})
    h.update(phase="done", made=made)
    del bandit


def _grade_count(h: JobHandle, target_grade: str) -> int:
    n = h.stats.get("grade_a", 0)
    if target_grade.upper() in ("B", "C", "D"):
        n += h.stats.get("grade_b", 0)
    return int(n)


def job_automine(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Quality-driven loop: bandit-chosen templates + grammar -> screen -> shape -> Doctor near misses ->
    periodic GP refinement and complex composites -> save alphas graded B or better (A = BRAIN-ready)."""
    from ..gen.compose import composites
    from .refine import component_corr, library_components

    rng = random.Random(config.get("seed", int(time.time())))
    deadline = time.time() + float(config.get("time_limit_min", 20)) * 60
    target = int(config.get("target_candidates", 10))
    target_grade = str(config.get("target_grade", "A")).upper()
    bandit = Bandit(ws.store, rng)
    job_index = CorrelationIndex(ws._corr_dates())
    bs = base_settings(config)
    local = ws.local_fields()
    has_volume = {"volume", "adv20"} <= local
    tmpls = [t for t in load_templates() if expand(t, local_fields=local, limit=1, validate=True)]
    fams = sorted({t.idea for t in tmpls})
    if config.get("families"):
        fams = [f for f in fams if f in set(config["families"])] or fams
    g = _grammar(ws, config, rng)
    seen: set[str] = set()
    rnd = 0
    pool_promising: list[tuple[Candidate, dict]] = []
    cfg = ws.checks_cfg
    save_cfg = {**config, "save_min_sharpe": float(config.get("save_min_sharpe", 1.0)),
                "save_min_grade": str(config.get("save_min_grade", "B"))}
    while time.time() < deadline and _grade_count(h, target_grade) < target:
        h.check()
        rnd += 1
        # 1) choose families by Thompson sampling, expand fresh candidates
        weights = bandit.weights([f"idea:{f}" for f in fams])
        cands: list[Candidate] = []
        batch = int(config.get("round_size", 24))
        grammar_share = float(config.get("grammar_share", 0.2))
        for _ in range(batch * 3):
            if len(cands) >= int(batch * (1 - grammar_share)):
                break
            fam = rng.choices(fams, weights=[weights[f"idea:{f}"] + 0.02 for f in fams])[0]
            t = rng.choice([t for t in tmpls if t.idea == fam])
            exprs = expand(t, local_fields=local, limit=3, rng=rng)
            if not exprs:
                continue
            e = rng.choice(exprs)
            s = rng.choice(settings_grid(t, bs))
            key = e + str(sorted(s.items()))
            if key in seen:
                continue
            seen.add(key)
            cands.append(Candidate(expr=e, settings=s, template_id=t.id, idea=t.idea, category=t.category,
                                   horizon=t.horizon, rationale=t.rationale, origin="automine"))
        tries = 0
        while len(cands) < batch and tries < batch * 10:
            tries += 1
            node = g.tree()
            if is_degenerate(node):
                continue
            e = to_expr(node)
            if e in seen:
                continue
            seen.add(e)
            cands.append(Candidate(expr=e, settings=dict(bs, decay=rng.choice([0, 3, 6])), origin="automine",
                                   idea=classify(node)["idea"]))
        h.update(phase=f"round {rnd}: screening", round=rnd)
        stage1 = screen_and_save(h, ws, mgr, cands, save_cfg, job_index, span="is", phase=f"round {rnd}: screening",
                                 bandit=bandit)
        # 2) near misses -> Doctor fixes (the shaper already handled the fitness/turnover side)
        near = []
        for c, r in stage1:
            m = (r.get("metrics") or {}).get("is") or {}
            if (m.get("sharpe") or 0) >= 0.7 and quick_grade(m, cfg, int(c.settings.get("delay", 1))) in ("C", "D"):
                near.append((c, r))
        near.sort(key=lambda cr: -quick_score((cr[1].get("metrics") or {}).get("is"), cfg))
        pool_promising.extend(near[:6])
        fixes: list[Candidate] = []
        for c, r in near[:6]:
            m = (r.get("metrics") or {}).get("is") or {}
            failed = []
            if m.get("turnover", 0) > cfg["turnover_max"]:
                failed.append("HIGH_TURNOVER")
            if m.get("turnover", 0) < cfg["turnover_min"]:
                failed.append("LOW_TURNOVER")
            if m.get("sharpe", 0) < cfg["delay1"]["sharpe_min"]:
                failed.append("LOW_SHARPE")
            if m.get("fitness", 0) < cfg["delay1"]["fitness_min"]:
                failed.append("LOW_FITNESS")
            if m.get("max_weight", 0) > cfg["max_weight"]:
                failed.append("CONCENTRATED_WEIGHT")
            try:
                node = lower_text(c.expr)
            except Exception:  # noqa: BLE001
                continue
            for f in diagnose(node, c.settings, failed, m, has_volume=has_volume)[:5]:
                e = to_expr(f.node)
                key = e + str(sorted(f.settings.items()))
                if key in seen:
                    continue
                seen.add(key)
                fixes.append(Candidate(expr=e, settings=f.settings, template_id=c.template_id, idea=c.idea,
                                       category=c.category, horizon=c.horizon, rationale=c.rationale,
                                       origin="doctor"))
        if fixes and time.time() < deadline:
            screen_and_save(h, ws, mgr, fixes, {**save_cfg, "optimize_top": 3}, job_index, span="is",
                            phase=f"round {rnd}: doctor fixes")
        # 3) GP refinement every few rounds from the best material found so far
        if rnd % int(config.get("gp_every", 3)) == 0 and time.time() < deadline - 60:
            seeds = []
            for c, r in sorted(pool_promising, key=lambda cr: -((cr[1].get("metrics") or {}).get("is") or {}).get(
                    "fitness", 0))[:10]:
                try:
                    seeds.append(Individual(lower_text(c.expr), c.settings, origin="automine"))
                except Exception:  # noqa: BLE001
                    pass
            seeds += _library_seeds(ws, {"seed_min_sharpe": 1.0}, 6)
            gp_cfg = {**config, "population": int(config.get("gp_population", 30)),
                      "generations": int(config.get("gp_generations", 5)), "islands": 1,
                      "time_limit_min": max(1.0, (deadline - time.time()) / 60 / 2)}
            hof = run_gp(h, ws, mgr, gp_cfg, seeds, job_index)
            finalize_individuals(h, ws, mgr, hof, {**config, "save_min_grade": save_cfg["save_min_grade"]}, job_index)
        # 4) complex alphas: compose decorrelated good alphas into multi-statement programs
        compose_due = rnd % int(config.get("compose_every", 4)) == 0
        if config.get("compose", True) and compose_due and time.time() < deadline - 60:
            comps = library_components(ws, {"max_library": 24, "min_grade": "B", "min_sharpe": 1.0})
            if len(comps) >= 2:
                progs = composites(comps, component_corr(ws, comps), local_fields=local, has_volume=has_volume,
                                   limit=int(config.get("compose_n", 8)))
                ccands = [Candidate(expr=p.text, settings={**bs, "decay": 0}, idea=p.parts[0].family,
                                    rationale=p.label, origin="compose",
                                    parents=[c.alpha_id for c in p.parts if c.alpha_id]) for p in progs
                          if p.text not in seen]
                seen.update(c.expr for c in ccands)
                if ccands:
                    screen_and_save(h, ws, mgr, ccands, {**save_cfg, "halving": False, "tags": ["complex"],
                                                         "optimize_top": 3}, job_index, span="is",
                                    phase=f"round {rnd}: complex composites")
        h.update(phase=f"round {rnd} done", passed=h.stats.get("passed", 0), grade_a=h.stats.get("grade_a", 0))
    if config.get("include_brain_only"):
        job_brain_only(h, ws, mgr, {**config, "per_template": 2, "n_field_candidates": 10})


JOB_KINDS: dict[str, Any] = {
    "batch": job_batch,
    "templates": job_templates,
    "alpha101": job_alpha101,
    "grammar": job_grammar,
    "gp": job_gp,
    "settings_opt": job_settings_opt,
    "brain_only": job_brain_only,
    "automine": job_automine,
}


def job_data_build(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Download/refresh free data (Wikipedia, Yahoo, SEC) and rebuild the real panel."""
    from ..data.build import build_real

    def prog(phase: str, done: int, total: int, msg: str) -> None:
        h.update(phase=f"{phase}: {msg}", done=done, total=total, stage=phase)

    info = build_real(prog, cancelled=h.cancelled, max_tickers=config.get("max_tickers"), pool=config.get("pool"))
    h.stats.update({"tickers": info["tickers"], "days": info["days"], "pool": info.get("pool")})
    ws.update_settings({"active_dataset": "auto"})
    ws.load_panel()
    mgr.reset_pool()
    h.event("data_ready", {"info": ws.panel.info(), "coverage": info.get("coverage")})


def job_demo_build(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    from .. import config as cfgmod
    from ..data.demo import build_demo

    h.update(phase="building demo dataset", done=0, total=1)
    build_demo(cfgmod.DEMO_PANELS_DIR, n_stocks=int(config.get("n_stocks", 300)),
               n_days=int(config.get("n_days", 2000)), seed=int(config.get("seed", 7)))
    ws.load_panel()
    mgr.reset_pool()
    h.update(done=1)
    h.event("data_ready", {"info": ws.panel.info()})


def job_forge(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Idea Forge: turn a plain-English idea into a champion alpha (see jobs/forge.py)."""
    from .forge import run_forge

    run_forge(h, ws, mgr, config)


def job_reengineer(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Re-engineer: staged beam search from a weak alpha to a strong one (see jobs/reengineer.py)."""
    from .reengineer import Reengineer

    Reengineer(h, ws, mgr, config).run()


JOB_KINDS["data_build"] = job_data_build
JOB_KINDS["demo_build"] = job_demo_build
JOB_KINDS["reengineer"] = job_reengineer
JOB_KINDS["forge"] = job_forge


def job_compose(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    """Complex (multi-statement) alphas from decorrelated library alphas (see jobs/refine.py)."""
    from .refine import job_compose as run

    run(h, ws, mgr, config)


JOB_KINDS["compose"] = job_compose


def _register_brain_jobs() -> None:
    from .brain_jobs import BRAIN_JOB_KINDS

    JOB_KINDS.update(BRAIN_JOB_KINDS)


_register_brain_jobs()
