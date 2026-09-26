"""Re-engineer job: a staged, explainable beam search that turns a weak alpha into a strong one.

The search scores variants on in-sample data only. The out-of-sample holdout is simulated once, at the
end, for the original and the finalists, so it stays an honest test of whether the improvement is real.
Every accepted step is kept as a lineage (the "recipe") and streamed to the UI as ``reengineer`` events.
"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..fastexpr import analyze, lower_text, to_expr
from ..fastexpr.ast import Node
from ..fastexpr.lower import simplify
from ..gen import reengineer as R
from ..gen.gp import GPConfig, GPEngine, Individual, Variation
from ..gen.templates import Candidate
from ..sim.correlation import CorrelationIndex
from ..sim.robustness import robust_pick
from ..sim.simulator import SimSettings
from .manager import JobCancelled, JobHandle, JobManager

METRIC_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "max_weight", "margin_bps")
SETTING_KEYS = ("decay", "neutralization", "truncation", "universe", "delay", "region")


def _slim(m: dict | None) -> dict:
    return {k: (None if (m or {}).get(k) is None else round(float(m[k]), 4)) for k in METRIC_KEYS} if m else {}


def _settings(s: dict) -> dict:
    return {k: s.get(k) for k in SETTING_KEYS if k in s}


@dataclass
class State:
    node: Node
    settings: dict
    expr: str
    metrics: dict
    score: float
    parts: dict
    pnl: np.ndarray
    size: int
    key: str
    lineage: list = field(default_factory=list)
    parent: "State | None" = None

    def summary(self) -> dict:
        return {"expr": self.expr, "settings": _settings(self.settings), "metrics": _slim(self.metrics),
                "score": self.score, "parts": self.parts, "steps": len(self.lineage)}


class Reengineer:
    def __init__(self, h: JobHandle, ws, mgr: JobManager, config: dict):
        from .miners import base_settings

        self.h, self.ws, self.mgr, self.cfg = h, ws, mgr, config
        self.t0 = time.time()
        budget = float(config.get("time_limit_min", 6)) * 60
        self.deadline = self.t0 + budget
        self.reserve = max(20.0, 0.12 * budget)  # time kept for the final holdout evaluation
        self.base = {**base_settings(config)}
        self.delay = int(self.base.get("delay", 1))
        self.local = ws.local_fields()
        self.beam_width = max(1, int(config.get("beam", 3)))
        self.min_gain = float(config.get("min_gain", 0.05))
        self.max_passes = max(1, int(config.get("max_passes", 3)))
        self.target_sharpe = float(config.get("target_sharpe", 2.0))
        self.target_fitness = float(config.get("target_fitness", 1.5))
        self.allow_blend = bool(config.get("allow_blend", True))
        self.allow_structure = bool(config.get("allow_structure", True))
        self.evolve = bool(config.get("evolve", True))
        self.min_kinship = float(config.get("min_kinship", 0.5))
        self.alpha_id = config.get("alpha_id")
        self.seed = int(config.get("seed", int(time.time())))
        self.by_key: dict[str, State] = {}
        self.states: list[State] = []
        self.beam: list[State] = []
        self.factor_pnl: dict[str, np.ndarray] = {}
        self.partners: list[R.Partner] | None = None
        self.n_evals = 0
        self.finishing = False
        self.stopped = False
        self.plan: list[str] = []
        self.done_steps = 0
        self.report: dict = {"job_id": h.id, "status": "running", "stages": [], "trajectory": [], "final": None,
                             "config": {k: config.get(k) for k in ("time_limit_min", "target_sharpe", "target_fitness",
                                                                   "allow_blend", "allow_structure", "evolve", "beam")}}

    # ------------------------------------------------------------------ evaluation
    @property
    def best(self) -> State:
        return self.beam[0]

    def _key(self, expr: str, s: dict) -> str:
        return f"{expr}|{SimSettings.from_dict(s).key()}"

    def _eval(self, payloads: list[dict], span: str) -> list[dict]:
        def on(i: int, r: dict) -> None:
            self.n_evals += 1
            self.h.bump("evaluated")
            if not r.get("ok"):
                self.h.bump("brain_only" if r.get("brain_only") else "errors")
        return self.mgr.evaluate(None if self.finishing else self.h, payloads, span, on_result=on)

    def _self_corr(self, pnl: np.ndarray, start: str | None) -> dict | None:
        from .miners import _dates

        idx = self.ws.corr_sub
        if idx is None or not len(idx) or not len(pnl):
            return None
        return idx.max_corr(_dates(self.ws, start, len(pnl)), pnl)

    def _state(self, r: dict, node: Node, s: dict, parent: State | None = None) -> State:
        from .miners import _pnl

        m = (r.get("metrics") or {}).get("is") or {}
        pnl = _pnl(r, "pnl_is").astype(np.float64)
        size = int(r.get("size") or node.size)
        score, parts = R.objective(m, pnl, size, self.ws.checks_cfg, self.delay,
                                   self._self_corr(pnl, r.get("pnl_is_start")))
        expr = to_expr(node)
        return State(node, dict(s), expr, m, score, parts, pnl, size, self._key(expr, s), parent=parent)

    def _step(self, mv: R.Move, st: State, parent: State) -> dict:
        pm, m = parent.metrics, st.metrics
        return {"stage": mv.stage, "stage_title": R.STAGES[mv.stage][0], "label": mv.label, "reason": mv.reason,
                "expr": st.expr, "settings": _settings(st.settings), "metrics": _slim(m), "score": st.score,
                "score_delta": round(st.score - parent.score, 4),
                "delta": {k: round(float(m.get(k) or 0) - float(pm.get(k) or 0), 4)
                          for k in ("sharpe", "fitness", "turnover", "returns")}}

    def run_moves(self, pairs: list[tuple[State, R.Move]], phase: str, halving: bool = True) -> list[State]:
        """Evaluate proposals on IS (3-year pre-screen first when there are many); reuse earlier evaluations."""
        out: list[State] = []
        fresh: list[tuple[State, R.Move, dict, str]] = []
        queued: set[str] = set()
        for parent, mv in pairs:
            s = {**parent.settings, **mv.settings}
            expr = mv.expr
            k = self._key(expr, s)
            if k in self.by_key:
                old = self.by_key[k]
                st = dataclasses.replace(old, parent=parent, lineage=[])
                st.lineage = parent.lineage + [self._step(mv, st, parent)]
                out.append(st)
            elif k not in queued:
                queued.add(k)
                fresh.append((parent, mv, s, expr))
        if not fresh:
            return out
        self.h.update(phase=f"{phase} · {len(fresh)} variants")
        if halving and len(fresh) > 16:
            res0 = self._eval([{"expr": e, "settings": s} for _, _, s, e in fresh], "screen")
            scored = [(self._state(r, f[1].node, f[2]).score, i) for i, (f, r) in enumerate(zip(fresh, res0))
                      if r.get("ok")]
            scored.sort(key=lambda x: -x[0])
            keep = {i for _, i in scored[: max(8, int(len(fresh) * 0.4))]}
            fresh = [f for i, f in enumerate(fresh) if i in keep]
        res = self._eval([{"expr": e, "settings": s} for _, _, s, e in fresh], "is")
        for (parent, mv, s, _e), r in zip(fresh, res):
            if not r.get("ok"):
                continue
            st = self._state(r, mv.node, s, parent)
            st.lineage = parent.lineage + [self._step(mv, st, parent)]
            self.by_key[st.key] = st
            self.states.append(st)
            out.append(st)
        return out

    def select(self, pool: list[State]) -> list[State]:
        """Top states by score, skipping near-duplicates (PnL correlation > 0.985) to keep the beam broad."""
        out: list[State] = []
        for st in sorted(pool, key=lambda x: -x.score):
            if any(st.key == o.key for o in out):
                continue
            if any((c := R.pnl_corr(st.pnl, o.pnl)) is not None and c > 0.985 for o in out):
                continue
            out.append(st)
            if len(out) >= self.beam_width:
                break
        return out

    # ------------------------------------------------------------------ context
    def exposures(self, pnl: np.ndarray) -> dict[str, float]:
        out = {}
        for fid, fp in self.factor_pnl.items():
            c = R.pnl_corr(pnl, fp)
            if c is not None:
                out[fid] = round(c, 3)
        return out

    def ctx(self, st: State) -> R.MoveContext:
        ctx = R.MoveContext(local_fields=self.local, metrics=st.metrics, exposures=self.exposures(st.pnl),
                            allow_structure=self.allow_structure)
        if "self_correlation" in st.parts.get("penalties", {}):
            sc = self._self_corr_for(st)
            if sc:
                a = self.ws.store.get_alpha(sc["alpha_id"])
                if a and a.get("local"):
                    try:
                        ctx.orthogonalize_to = lower_text(a["expr"])
                        ctx.orthogonalize_label = f"submitted alpha #{a['id']}"
                    except Exception:  # noqa: BLE001 - an unparsable library row just skips this move
                        pass
        return ctx

    def _self_corr_for(self, st: State) -> dict | None:
        start = str(self.ws.panel.dates[self.ws.periods.is_start])
        return self._self_corr(st.pnl, start)

    # ------------------------------------------------------------------ stages
    def partner_pool(self) -> list[R.Partner]:
        if self.partners is not None:
            return self.partners
        cands: list[tuple[str, str, Node, str]] = []
        for c in R.COMPANIONS:
            n = lower_text(c.expr)
            if {f for f in _fields(n)} <= self.local | {"market"}:
                cands.append((c.id, c.title, n, "companion"))
        orig_key = self.original.node.key
        rows = self.ws.store.list_alphas(local=True, min_sharpe=1.0, sort="fitness", limit=12)["rows"]
        for a in rows:
            try:
                n = lower_text(a["expr"])
            except Exception:  # noqa: BLE001
                continue
            if n.size <= 15 and n.key != orig_key and a["id"] != self.alpha_id:
                cands.append((f"alpha{a['id']}", f"library alpha #{a['id']}", n, "library"))
        s = self.best.settings
        res = self._eval([{"expr": to_expr(R.ranked(n)), "settings": s} for _, _, n, _ in cands], "is")
        from .miners import _pnl

        self.partners = []
        for (pid, title, n, src), r in zip(cands, res):
            if r.get("ok"):
                m = (r.get("metrics") or {}).get("is") or {}
                self.partners.append(R.Partner(pid, title, n, float(m.get("sharpe") or 0.0),
                                               _pnl(r, "pnl_is").astype(np.float64), src))
        return self.partners

    def run_stage(self, stage: str, pass_no: int) -> None:
        title = R.STAGES[stage][0]
        phase = f"pass {pass_no} · {title}"
        before = self.best
        pairs: list[tuple[State, R.Move]] = []
        if stage == "settings":
            children, winners = self._settings_stage(phase)
        else:
            if stage == "blend":
                pool = self.partner_pool()
                for st in self.beam:
                    if any(step["stage"] == "blend" for step in st.lineage):
                        continue  # one companion per lineage keeps the result recognisably the same alpha
                    ctx = self.ctx(st)
                    picks = R.blend_candidates(st.pnl, float(st.metrics.get("sharpe") or 0), pool, k=3)
                    for mv in R.blend_moves(st.node, st.settings, picks):
                        mv.node = simplify(mv.node)
                        if R.valid(mv.node, ctx):
                            pairs.append((st, mv))
            else:
                for st in self.beam:
                    pairs += [(st, mv) for mv in R.propose(stage, st.node, st.settings, self.ctx(st))]
            children = self.run_moves(pairs, phase)
            if stage == "blend":
                kept = []
                for c in children:
                    k = R.pnl_corr(c.pnl, c.parent.pnl if c.parent else None)
                    if k is not None and k >= self.min_kinship:
                        kept.append(c)
                    else:
                        self.h.bump("rejected_kinship")
                children = kept
            winners = [c for c in children if c.parent is not None and c.score >= c.parent.score + self.min_gain]
            self.beam = self.select(self.beam + winners)
        moved = self.best is not before and self.best.lineage
        self.report["stages"].append({
            "pass": pass_no, "stage": stage, "title": title, "tried": len(children), "accepted": len(winners),
            "best_label": self.best.lineage[-1]["label"] if moved else None,
            "gain": round(self.best.score - before.score, 4), "elapsed_s": round(time.time() - self.t0, 1)})
        self._tick(f"P{pass_no} {title}", stage, pass_no)

    def _settings_stage(self, phase: str) -> tuple[list[State], list[State]]:
        st = self.best
        cells = R.settings_cells(st.settings)
        pairs = [(st, R.Move("settings", f"Settings: decay {c['decay']}, {c['neutralization'].lower()}, truncation "
                                         f"{c['truncation']:g}",
                             "Robust pick from a decay × neutralization × truncation grid: the best neighbourhood, "
                             "not the lucky peak.", st.node, {**st.settings, **c}))
                 for c in cells]
        children = self.run_moves(pairs, phase, halving=False)
        by_cell: dict[tuple, State] = {}
        for c in [st] + children:
            s = c.settings
            by_cell[(int(s.get("decay") or 0), str(s.get("neutralization")).upper(), float(s.get("truncation")))] = c
        rows = [{"decay": k[0], "neutralization": k[1], "truncation": k[2], "score": v.score}
                for k, v in by_cell.items()]
        pick = robust_pick(rows, lambda r: r["score"])
        winners = []
        if pick:
            chosen = by_cell[(pick["decay"], pick["neutralization"], pick["truncation"])]
            if chosen is not st and chosen.score >= st.score + self.min_gain:
                winners = [chosen]
                self.beam = self.select(self.beam + winners)
        return children, winners

    def run_evolve(self) -> None:
        from .miners import _dates, _grammar, _pnl

        remaining = self.deadline - self.reserve - time.time()
        if remaining < 45:
            return
        rng = random.Random(self.seed)
        best = self.best
        bm = best.metrics
        cfg = GPConfig(population=int(self.cfg.get("gp_population", 24)),
                       generations=int(self.cfg.get("gp_generations", 6)), islands=1, max_depth=R.MAX_DEPTH,
                       max_nodes=R.MAX_SIZE - 5, min_sharpe=float(bm.get("sharpe") or 0),
                       min_fitness=float(bm.get("fitness") or 0), corr_cap=1.0, halving=True,
                       seed=rng.randint(0, 10 ** 9))
        g = _grammar(self.ws, {"max_depth": 4}, rng)
        var = Variation(g, rng, cfg.max_depth, cfg.max_nodes)
        ctx = self.ctx(best)
        seeds = [Individual(s.node, dict(s.settings), origin="reengineer") for s in self.beam]
        tries = 0
        while len(seeds) < cfg.population and tries < cfg.population * 10:
            tries += 1
            par = rng.choice(self.beam)
            child = var.mutate(Individual(par.node, dict(par.settings), origin="reengineer"))
            if R.valid(child.node, ctx):
                seeds.append(child)
        stop_at = time.time() + remaining * 0.8
        n_is = len(best.pnl)

        def evaluate(inds: list[Individual], span: str) -> None:
            res = self._eval([{"expr": i.expr, "settings": i.settings} for i in inds], span)
            for i, r in zip(inds, res):
                if r.get("ok"):
                    i.metrics = (r.get("metrics") or {}).get("is")
                    i._pnl = _pnl(r, "pnl_is").astype(np.float64)  # type: ignore[attr-defined]
                    i._dates = _dates(self.ws, r.get("pnl_is_start"), len(i._pnl))  # type: ignore[attr-defined]
                    i._res = r  # type: ignore[attr-defined]
                else:
                    i.metrics = None

        def novelty(inds: list[Individual]) -> None:
            for i in inds:
                c = R.pnl_corr(getattr(i, "_pnl", None), best.pnl)
                i.novelty = 1.0 - abs(c) if c is not None else 0.5

        def on_gen(stat: dict) -> None:
            self.h.update(phase=f"Evolve · generation {stat['generation']}/{cfg.generations}")
            self.h.event("gp_generation", {"stat": stat})

        gates = {"turnover_min": self.ws.checks_cfg["turnover_min"], "turnover_max": self.ws.checks_cfg["turnover_max"],
                 "max_weight": self.ws.checks_cfg["max_weight"]}
        eng = GPEngine(cfg, g, evaluate, novelty, gates, on_gen,
                       stop=lambda: self.h.cancelled() or time.time() > stop_at)
        before = self.best
        hof = eng.run(seeds)
        winners = []
        for ind in hof:
            r = getattr(ind, "_res", None)
            if r is None or len(getattr(ind, "_pnl", [])) != n_is:
                continue
            parent = max(self.beam, key=lambda b: R.pnl_corr(ind._pnl, b.pnl) or -1)  # type: ignore[attr-defined]
            st = self._state(r, simplify(ind.node), ind.settings, parent)
            if st.key in self.by_key:
                continue
            mv = R.Move("evolve", "GP rewrite", "Genetic programming found a better neighbour of the best versions "
                        "(mutation and crossover of their expression trees).", st.node, st.settings)
            st.lineage = parent.lineage + [self._step(mv, st, parent)]
            self.by_key[st.key] = st
            self.states.append(st)
            if st.score >= parent.score + 2 * self.min_gain:  # GP tries far more variants: demand a larger gain
                winners.append(st)
        self.beam = self.select(self.beam + winners)
        self.report["stages"].append({
            "pass": 0, "stage": "evolve", "title": "Evolve", "tried": eng.evals, "accepted": len(winners),
            "best_label": "GP rewrite" if winners else None, "gain": round(self.best.score - before.score, 4),
            "elapsed_s": round(time.time() - self.t0, 1)})
        self._tick("Evolve", "evolve", 0)

    # ------------------------------------------------------------------ reporting
    def _tick(self, label: str, stage: str, pass_no: int) -> None:
        b = self.best
        self.report["trajectory"].append({"label": label, "stage": stage, "pass": pass_no, "score": b.score,
                                          "sharpe": _slim(b.metrics).get("sharpe"),
                                          "fitness": _slim(b.metrics).get("fitness"),
                                          "turnover": _slim(b.metrics).get("turnover"), "evals": self.n_evals})
        self.done_steps += 1
        self.h.update(done=min(self.done_steps, len(self.plan)), total=len(self.plan), best_score=b.score,
                      best_sharpe=b.metrics.get("sharpe"), best_fitness=b.metrics.get("fitness"))
        self.emit()

    def emit(self, store: bool = True) -> None:
        b = self.best
        self.report.update({
            "best": {**b.summary(), "lineage": b.lineage},
            "beam": [s.summary() for s in self.beam],
            "n_evaluated": self.n_evals, "elapsed_s": round(time.time() - self.t0, 1),
            "improvement": round(b.score - self.original.score, 4),
        })
        self.h.event("reengineer", {"report": self.report})
        if store:
            self.ws.store.kv_set(f"reengineer:{self.h.id}", self.report)

    def time_left(self) -> bool:
        return time.time() < self.deadline - self.reserve and not self.h.cancelled()

    # ------------------------------------------------------------------ main
    def baseline(self) -> None:
        from .miners import score_result

        expr = str(self.cfg.get("expr") or "").strip()
        an = analyze(expr, local_fields=self.local)
        if not an.ok:
            raise ValueError(an.diagnostics[-1].message if an.diagnostics else "invalid expression")
        if not an.local:
            raise ValueError("Re-engineering needs an alpha that simulates locally: "
                             + "; ".join(an.brain_only_reasons))
        self.h.update(phase="Diagnosing the original", done=0, total=len(self.plan))
        r0 = self._eval([{"expr": expr, "settings": self.base, "extras": True}], "full")[0]
        if not r0.get("ok"):
            raise ValueError(r0.get("error") or "the original alpha could not be simulated")
        self.r0 = r0
        self.checks0, self.pp0, _ = score_result(self.ws, r0)
        self.original = self._state(r0, an.node, self.base)
        self.by_key[self.original.key] = self.original
        self.beam = [self.original]
        facs = [f for f in R.STYLE_FACTORS if _fields(lower_text(f.expr)) <= self.local | {"market"}]
        from .miners import _pnl

        for f, r in zip(facs, self._eval([{"expr": f.expr, "settings": self.base} for f in facs], "is")):
            if r.get("ok"):
                self.factor_pnl[f.id] = _pnl(r, "pnl_is").astype(np.float64)
        m0 = r0["metrics"]
        expo = self.exposures(self.original.pnl)
        self.report["original"] = {
            "expr": expr, "settings": _settings(self.base), "is": _slim(m0.get("is")), "os": _slim(m0.get("os")),
            "score": self.original.score, "parts": self.original.parts, "status": self.checks0["status"],
            "failed": self.checks0["failed"], "warnings": self.checks0["warnings"],
            "sub_sharpe": (r0.get("extras") or {}).get("sub_sharpe"),
            "stability": (r0.get("extras") or {}).get("stability"), "alpha_id": self.alpha_id}
        self.report["exposures"] = expo
        self.report["diagnosis"] = R.diagnose(m0.get("is") or {}, m0.get("os"),
                                              tuple(self.original.parts.get("halves", (0, 0))), self.checks0,
                                              r0.get("extras"), expo, self.ws.checks_cfg, self.delay)
        self._tick("Original", "baseline", 0)

    def search(self) -> None:
        for p in range(1, self.max_passes + 1):
            start = self.best.score
            stages = (["direction"] if p == 1 else []) + [s for s in R.PASS_STAGES if s != "blend" or self.allow_blend]
            stages.append("settings")
            for stage in stages:
                if not self.time_left():
                    return
                self.run_stage(stage, p)
                if self.reached_target():
                    return
            if self.best.score < start + self.min_gain:
                break  # a full pass found nothing: more passes of the same moves would only fit noise
        if self.evolve and self.time_left():
            self.run_evolve()

    def reached_target(self) -> bool:
        """Stop early once a weak alpha reaches the target (fewer tries, less overfitting). An alpha that already
        met the target is only polished, so the target never cuts its search short."""
        b = self.best
        o = self.original
        if R.meets_target(o.metrics, o.parts, self.target_sharpe, self.target_fitness):
            return False
        return b is not o and R.meets_target(b.metrics, b.parts, self.target_sharpe, self.target_fitness)

    def finalists(self, k: int = 4) -> list[State]:
        """Best-scoring distinct versions, plus the best one without a blend step so the pure re-engineering
        of the original idea is always on the table next to the blended one."""
        pool = [s for s in self.states if s.score >= self.original.score + self.min_gain]
        ranked = sorted(pool, key=lambda x: -x.score)
        pure = next((s for s in ranked if not any(st["stage"] == "blend" for st in s.lineage)), None)
        head = [self.best] + ([pure] if pure is not None else [])
        out: list[State] = []
        for st in head + ranked:
            if st is self.original or any(st.key == o.key for o in out):
                continue
            if st.score < self.original.score + self.min_gain:
                continue
            if any((c := R.pnl_corr(st.pnl, o.pnl)) is not None and c > 0.95 for o in out):
                continue
            out.append(st)
            if len(out) >= k:
                break
        return out

    def finalize(self) -> None:
        from .miners import _pnl, persist, score_result

        self.finishing = True
        self.h.update(phase="Holdout check of the finalists")
        fins = self.finalists()
        res = self._eval([{"expr": s.expr, "settings": s.settings, "extras": True} for s in fins], "full")
        rows = []
        for st, r in zip(fins, res):
            if not r.get("ok"):
                continue
            checks, pp, n_fail = score_result(self.ws, r)
            stab = (r.get("extras") or {}).get("stability")
            final = st.score - 0.75 * n_fail - (0.3 if stab is not None and stab < 0.7 else 0.0)
            rows.append({"final": round(final, 4), "state": st, "res": r, "checks": checks, "pass_prob": pp})
        rows.sort(key=lambda x: -x["final"])
        orig_expr = self.report["original"]["expr"]
        job_index = CorrelationIndex(self.ws._corr_dates())
        for i, row in enumerate(rows):
            if i > 0 and row["checks"]["status"] != "PASS":
                continue  # alternatives are listed for comparison, but only the champion and passing ones are saved
            st = row["state"]
            recipe = R.recipe_text(st.lineage)
            cand = Candidate(expr=st.expr, settings=st.settings, origin="reengineer",
                             parents=[int(self.alpha_id)] if self.alpha_id else [],
                             rationale=f"Re-engineered from {orig_expr}: {recipe}.")
            out = persist(self.h, self.ws, cand, row["res"], job_index, 0.97, tags=["reengineered"])
            if out:
                row["alpha_id"] = out["id"]
                a = self.ws.store.get_alpha(out["id"])
                if a is not None and not a.get("notes"):
                    self.ws.store.update_alpha(out["id"], {"notes": f"Re-engineered by job #{self.h.id} from: "
                                                                    f"{orig_expr}\nRecipe: {recipe}"})
        final = {"n_evaluated": self.n_evals, "elapsed_s": round(time.time() - self.t0, 1), "champion": None,
                 "alternatives": [], "series": None}
        orig_row = {**self.report["original"], "checks": _check_rows(self.checks0), "pass_prob": self.pp0,
                    "final": self.original.score}
        final["original"] = orig_row
        if rows:
            champ = rows[0]
            final["champion"] = self._final_row(champ)
            final["alternatives"] = [self._final_row(r) for r in rows[1:]]
            final["series"] = self._series(self.r0, champ["res"])
            final["exposures_after"] = self.exposures(champ["state"].pnl)
            kin = R.pnl_corr(champ["state"].pnl, self.original.pnl)
            final["kinship"] = None if kin is None else round(abs(kin), 3)
            final["kinship_flipped"] = kin is not None and kin < 0
            final["verdict"] = self._verdict(final["champion"], orig_row)
        else:
            final["verdict"] = {"level": "warn", "title": "No robust improvement found",
                                "text": "No variant beat the original by a meaningful margin within the budget. Try a "
                                        "longer budget, allow blending, or rethink the idea itself."}
        self.report["final"] = final

    def _final_row(self, row: dict) -> dict:
        st: State = row["state"]
        r = row["res"]
        ex = r.get("extras") or {}
        return {"expr": st.expr, "settings": _settings(st.settings), "is": _slim(r["metrics"].get("is")),
                "os": _slim(r["metrics"].get("os")), "sub_sharpe": ex.get("sub_sharpe"),
                "stability": ex.get("stability"), "status": row["checks"]["status"], "failed": row["checks"]["failed"],
                "warnings": row["checks"]["warnings"], "checks": _check_rows(row["checks"]),
                "pass_prob": row["pass_prob"], "score": st.score, "final": row["final"],
                "alpha_id": row.get("alpha_id"), "recipe": R.recipe_text(st.lineage), "lineage": st.lineage,
                "size": st.size}

    def _series(self, r0: dict, r1: dict, max_points: int = 420) -> dict | None:
        from .miners import _pnl

        p = self.ws.panel
        a, b = _pnl(r0, "pnl").astype(np.float64), _pnl(r1, "pnl").astype(np.float64)
        if not len(a) or not len(b) or not r0.get("dates_start") or not r1.get("dates_start"):
            return None
        i0, i1 = p.index_of(r0["dates_start"]), p.index_of(r1["dates_start"])
        start, end = max(i0, i1), min(i0 + len(a), i1 + len(b))
        if end - start < 2:
            return None
        ca = np.cumsum(a[start - i0:end - i0])
        cb = np.cumsum(b[start - i1:end - i1])
        step = max(1, math.ceil((end - start) / max_points))
        os_i = self.ws.periods.os_start
        keep = [end - start - 1] + ([os_i - start] if start <= os_i < end else [])  # the OS start must be a category
        idx = np.unique(np.r_[np.arange(0, end - start, step), keep])
        dates = [str(d) for d in p.dates[start:end][idx]]
        return {"dates": dates, "original": np.round(ca[idx], 0).tolist(), "champion": np.round(cb[idx], 0).tolist(),
                "os_start": str(p.dates[os_i]) if start <= os_i < end else None}

    def _verdict(self, ch: dict, orig: dict) -> dict:
        """Judge the champion on the OS holdout, which the search never saw."""
        blended = any(step["stage"] == "blend" for step in ch.get("lineage") or [])
        bar = float(self.ws.checks_cfg["delay0" if self.delay == 0 else "delay1"]["sharpe_min"])
        is_sh = float(ch["is"].get("sharpe") or 0)
        os_sh = float((ch["os"] or {}).get("sharpe") or 0)
        o_is = float((orig.get("is") or {}).get("sharpe") or 0)
        o_os = float((orig.get("os") or {}).get("sharpe") or 0)
        ratio = os_sh / is_sh if is_sh > 0 else 0.0
        change = f"IS Sharpe {o_is:.2f} → {is_sh:.2f}; holdout (OS) Sharpe {o_os:.2f} → {os_sh:.2f}."
        if os_sh >= 0.8 * bar and ratio >= 0.5 and os_sh > o_os:
            if ch["status"] == "PASS":
                level, title = "good", "The holdout confirms the improvement"
            else:
                level, title = "warn", "The holdout confirms the edge, but some checks still fail"
            text = f"{change} The search never saw the holdout, so this is an honest test."
        elif os_sh > max(0.3, o_os + 0.3):
            level, title = "warn", "Better out of sample, but below the in-sample level"
            which = "especially the unblended one" if blended else "simpler ones often hold up better"
            text = (f"{change} The edge is real but part of the in-sample gain is probably fitted noise. Compare "
                    f"the alternatives ({which}) before submitting.")
        elif os_sh > o_os:
            level, title = "bad", "Most of the gain does not hold out of sample"
            text = (f"{change} The holdout barely moved while in-sample jumped, so most of the gain is fitted "
                    "noise. Do not submit this version as is.")
        else:
            level, title = "bad", "The holdout does not confirm the improvement"
            text = f"{change} Treat the in-sample gain as overfitting and do not submit this version as is."
        return {"level": level, "title": title, "text": text, "os_is_ratio": round(ratio, 3)}

    def run(self) -> None:
        stages_per_pass = len(R.PASS_STAGES) - (0 if self.allow_blend else 1) + 1
        self.plan = ["baseline"] + ["stage"] * (1 + stages_per_pass * self.max_passes) + \
            (["evolve"] if self.evolve else [])
        self.baseline()
        try:
            self.search()
        except JobCancelled:
            self.stopped = True
        self.finalize()
        self.report["status"] = "stopped" if self.stopped else "done"
        self.h.update(phase="stopped: kept the best version found" if self.stopped else "done",
                      done=len(self.plan), total=len(self.plan))
        self.emit()
        if self.stopped:
            raise JobCancelled()


def _fields(n: Node) -> set[str]:
    from ..fastexpr.ast import Field

    return {x.name for x in n.walk() if isinstance(x, Field)}


def _check_rows(checks: dict) -> list[dict]:
    return [{k: c.get(k) for k in ("name", "result", "value", "limit", "message")} for c in checks.get("checks", [])]

