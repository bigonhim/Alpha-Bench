"""Idea Forge, part 2: the background job that turns an interpreted idea into a champion alpha.

Stages (selection uses the in-sample period only; the OS holdout is reported, never selected on):
  interpret - read the idea (gen/idea.py)
  draft     - recipe, template and seed drafts built from the hypothesis
  combine   - cross-family blends, subset interactions, event gates, smoothing and size neutrality on the leaders
  refine    - Doctor fixes for near-misses, fitness shaping, then a robust decay x neutralization sweep on the leaders
  compose   - complex (multi-statement) alphas: blends, tilts, regime switches and horizon ensembles of the leaders
  evolve    - NSGA-II genetic programming seeded with the leaders and restricted to the idea's data
  polish    - full evaluation (sub-universe, stability, checks, quality grade) of the finalists; champion, the best
              simple and the best complex alpha, and decorrelated runners-up
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..fastexpr import analyze, lower_text, to_expr
from ..fastexpr.explain import classify
from ..gen.build import CAPBUCKET_TEXT
from ..gen.compose import Component, composites, is_complex
from ..gen.doctor import diagnose
from ..gen.gp import Individual
from ..gen.idea import (FAMILY_LABEL, FUNDAMENTAL_FAMILIES, IdeaSpec, allowed_fields, apply_overrides, as_rank,
                        brain_only_drafts, fidelity, foreign_fields, interpret, missing_fields, synthesize)
from ..gen.optimize import acceptable, shaping_variants
from ..gen.templates import Candidate
from ..sim.correlation import CorrelationIndex
from ..sim.quality import quick_score as quality_quick_score
from ..sim.robustness import robust_pick
from ..sim.simulator import SimSettings
from .manager import JobHandle, JobManager
from .miners import _dates, _pnl, base_settings, persist, quality_of, run_gp, score_result

STAGES = ("interpret", "draft", "combine", "refine", "compose", "evolve", "polish")

EFFORT: dict[str, dict] = {
    "quick": dict(draft=48, combine=16, refine_top=4, fixes_per=4, sweep_top=1, shape_top=2, compose=8,
                  gp_population=24, gp_generations=4, gp_minutes=1.5, finalists=6, time_limit_min=5),
    "standard": dict(draft=120, combine=36, refine_top=6, fixes_per=5, sweep_top=2, shape_top=3, compose=16,
                     gp_population=40, gp_generations=8, gp_minutes=4.0, finalists=10, time_limit_min=12),
    "deep": dict(draft=240, combine=64, refine_top=8, fixes_per=6, sweep_top=3, shape_top=5, compose=28,
                 gp_population=64, gp_generations=16, gp_minutes=10.0, finalists=14, time_limit_min=28),
}
SWEEP_DECAYS = [0, 2, 4, 6, 8, 12]
SWEEP_NEUTS = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
RUNNERS = 4
CORR_CAP = 0.7


@dataclass
class Entry:
    expr: str
    settings: dict
    family: str
    stage: str
    lineage: list[str]
    sign: int = 0
    origin: str = "recipe"
    template_id: str | None = None
    category: str = "pv"
    horizon: str = "short"
    canon: str = ""
    metrics: dict | None = None
    fidelity: float = 0.0
    score: float = -99.0
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.canon}|{SimSettings.from_dict(self.settings).key()}"

    def child(self, expr: str, step: str, stage: str, settings: dict | None = None, origin: str | None = None,
              family: str | None = None) -> "Entry":
        return Entry(expr, dict(settings or self.settings), family or self.family, stage, self.lineage + [step],
                     self.sign, origin or "variant", self.template_id, self.category, self.horizon)

    def row(self) -> dict:
        m = self.metrics or {}
        return {"expr": self.expr, "settings": self.settings, "family": self.family, "stage": self.stage,
                "sharpe": m.get("sharpe"), "fitness": m.get("fitness"), "turnover": m.get("turnover"),
                "returns": m.get("returns"), "fidelity": self.fidelity, "score": round(self.score, 3)}


def _clip(x, lo: float, hi: float) -> float:
    try:
        return float(min(hi, max(lo, float(x))))
    except (TypeError, ValueError):
        return lo


class Forge:
    def __init__(self, h: JobHandle, ws, mgr: JobManager, config: dict):
        self.h, self.ws, self.mgr = h, ws, mgr
        effort = str(config.get("effort", "standard"))
        self.effort = effort if effort in EFFORT else "standard"
        self.cfg = {**EFFORT[self.effort], **{k: v for k, v in config.items() if k in EFFORT["quick"]}}
        self.base = base_settings(config)
        self.local = ws.local_fields()
        self.rng = random.Random(config.get("seed", int(time.time())))
        self.started = time.time()
        self.deadline = self.started + float(self.cfg["time_limit_min"]) * 60
        self.reserve = max(20.0, 0.08 * float(self.cfg["time_limit_min"]) * 60)
        self.idea = str(config.get("idea") or "").strip()
        self.spec: IdeaSpec = interpret(self.idea, self.local)
        apply_overrides(self.spec, self.local, config.get("families") or None, config.get("horizon") or None)
        self.pool: dict[str, Entry] = {}
        self.tried: set[str] = set()
        self.stages: dict[str, dict] = {s: {"status": "pending"} for s in STAGES}
        self.has_volume = {"volume", "adv20"} <= self.local

    # ------------------------------------------------------------------ bookkeeping
    def time_left(self) -> float:
        return self.deadline - time.time()

    def can_run(self) -> bool:
        return self.time_left() > self.reserve

    def leaderboard(self, n: int = 6) -> list[dict]:
        out, seen = [], set()
        for e in sorted(self.pool.values(), key=lambda e: -e.score):
            if e.canon in seen:
                continue
            seen.add(e.canon)
            out.append(e.row())
            if len(out) >= n:
                break
        return out

    def begin(self, stage: str) -> None:
        self.stages[stage] = {"status": "running", "started": round(time.time() - self.started, 1)}
        self.h.update(stage=stage, stage_idx=STAGES.index(stage), stages=self.stages, phase=f"{stage}")

    def end(self, stage: str, entries: list[Entry] | None = None, status: str = "done", note: str = "") -> None:
        st = self.stages[stage]
        st["status"] = status
        st["seconds"] = max(0.0, round(time.time() - self.started - float(st.get("started", 0.0)), 1))
        if entries is not None:
            st["candidates"] = len(entries)
            ok = [e for e in entries if e.metrics]
            if ok:
                b = max(ok, key=lambda e: e.score)
                st["best"] = b.row()
        if note:
            st["note"] = note
        self.h.update(stages=self.stages, leaderboard=self.leaderboard())

    def skip(self, stage: str, why: str) -> None:
        self.stages[stage] = {"status": "skipped", "note": why}
        self.h.update(stages=self.stages)

    def quick_score(self, m: dict | None, fid: float) -> float:
        if not m:
            return -99.0
        cfg = self.ws.checks_cfg
        to = float(m.get("turnover", 0.0) or 0.0)
        mw = float(m.get("max_weight", 0.0) or 0.0)
        pen = 0.0
        if to > cfg["turnover_max"]:
            pen += 0.5 + 4 * (to - cfg["turnover_max"])
        if to < cfg["turnover_min"]:
            pen += 1.0
        if mw > cfg["max_weight"]:
            pen += 0.5 + 10 * (mw - cfg["max_weight"])
        return _clip(m.get("fitness"), -3, 5) + 0.25 * _clip(m.get("sharpe"), -3, 6) - pen + 0.5 * fid

    def _canon(self, expr: str) -> str | None:
        an = analyze(expr, local_fields=self.local)
        return an.canon_hash if an.ok and an.local else None

    def fresh(self, entries: list[Entry], limit: int | None = None) -> list[Entry]:
        """Drop invalid, non-local and already-tried (expression, settings) pairs."""
        out = []
        for e in entries:
            c = self._canon(e.expr)
            if c is None:
                continue
            e.canon = c
            if e.key in self.tried:
                continue
            self.tried.add(e.key)
            out.append(e)
            if limit and len(out) >= limit:
                break
        return out

    def _on_result(self, i: int, r: dict) -> None:
        self.h.bump("evaluated")
        if not r.get("ok"):
            self.h.bump("errors")
        self.h.update(done=self.h.progress.get("done", 0) + 1)

    def admit(self, e: Entry, metrics: dict | None) -> bool:
        if not metrics:
            return False
        e.metrics = metrics
        try:
            node = lower_text(e.expr)
        except Exception:  # noqa: BLE001
            return False
        e.fidelity = fidelity(node, self.spec, e.family if e.origin != "gp" else None)
        if e.sign < 0:
            e.fidelity = round(e.fidelity * 0.6, 3)  # a reverse-direction test is, by design, not the idea
        e.extra["foreign"] = sorted(foreign_fields(node, self.spec))
        e.score = self.quick_score(metrics, e.fidelity)
        old = self.pool.get(e.key)
        if old is None or old.score < e.score:
            self.pool[e.key] = e
        return True

    def screen(self, entries: list[Entry], label: str) -> list[Entry]:
        if not entries:
            return []
        self.h.update(phase=f"{label}: simulating {len(entries)} candidates", done=0, total=len(entries))
        res = self.mgr.evaluate(self.h, [{"expr": e.expr, "settings": e.settings} for e in entries], "is",
                                on_result=self._on_result)
        out = []
        for e, r in zip(entries, res):
            if r.get("ok") and self.admit(e, (r.get("metrics") or {}).get("is")):
                if r.get("pnl_is"):
                    e.extra["pnl"] = _pnl(r, "pnl_is")  # kept for composing complex alphas
                out.append(e)
        self.h.update(leaderboard=self.leaderboard())
        return out

    def leaders(self, n: int, min_fid: float = 0.0, on_idea: bool = False) -> list[Entry]:
        out, seen = [], set()
        for e in sorted(self.pool.values(), key=lambda e: -e.score):
            if e.canon in seen or e.fidelity < min_fid or (on_idea and e.extra.get("foreign")):
                continue
            seen.add(e.canon)
            out.append(e)
            if len(out) >= n:
                break
        return out

    def failing(self, m: dict) -> list[str]:
        cfg = self.ws.checks_cfg
        th = cfg["delay1"] if int(self.base.get("delay", 1)) == 1 else cfg["delay0"]
        f = []
        if m.get("turnover", 0) > cfg["turnover_max"]:
            f.append("HIGH_TURNOVER")
        if m.get("turnover", 0) < cfg["turnover_min"]:
            f.append("LOW_TURNOVER")
        if m.get("sharpe", 0) < th["sharpe_min"]:
            f.append("LOW_SHARPE")
        if m.get("fitness", 0) < th["fitness_min"]:
            f.append("LOW_FITNESS")
        if m.get("max_weight", 0) > cfg["max_weight"]:
            f.append("CONCENTRATED_WEIGHT")
        return f

    # ------------------------------------------------------------------ stages
    def stage_draft(self) -> None:
        self.begin("draft")
        drafts = synthesize(self.spec, self.local, self.base, budget=int(self.cfg["draft"]), rng=self.rng)
        entries = [Entry(d.expr, d.settings, d.family, "draft",
                         [("Your expression" if d.origin == "seed" else
                           d.label[0].upper() + d.label[1:] if d.origin != "template" else
                           "Library " + d.label)],
                         d.sign, d.origin, d.template_id, d.category, d.horizon) for d in drafts]
        out = self.screen(self.fresh(entries), "Draft")
        self.end("draft", out)

    def stage_combine(self) -> None:
        self.begin("combine")
        spec = self.spec
        best: dict[str, Entry] = {}
        for e in sorted(self.pool.values(), key=lambda e: -e.score):
            if e.sign >= 0 and e.family not in best:
                best[e.family] = e
        fams = [f for f in spec.local_families if f in best] or list(best)[:1]
        cands: list[Entry] = []
        if fams:
            A = best[fams[0]]
            a = as_rank(A.expr)
            for f in fams[1:3]:
                B = best[f]
                b = as_rank(B.expr)
                lab = FAMILY_LABEL.get(f, f).lower()
                if f in spec.subset_families:
                    cands.append(A.child(f"{a} * {b}", f"Applied within high-{lab} stocks (interaction)", "combine",
                                         origin="combo"))
                cands.append(A.child(f"{a} + {b}", f"Blended equally with {lab}: {B.expr}", "combine", origin="combo"))
                cands.append(A.child(f"{a} + 0.5 * {b}", f"Blended with half-weight {lab}", "combine", origin="combo"))
                cands.append(A.child(f"{a} * {b}", f"Interacted with {lab} (both must agree)", "combine",
                                     origin="combo"))
            if spec.subset_size:
                sz = "-cap" if spec.subset_size < 0 else "cap"
                word = "small" if spec.subset_size < 0 else "large"
                cands.append(A.child(f"{a} * rank({sz})", f"Tilted toward {word} caps", "combine", origin="combo"))
        gates = []
        if self.has_volume and "volume_event" in spec.conditions:
            gates.append(("volume > 1.5 * adv20", "Traded only after abnormal-volume days"))
        if "earnings_event" in spec.conditions:
            f = next((f for f in ("eps", "income", "sales") if f in self.local), None)
            if f:
                gates.append((f"days_from_last_change({f}) < 10", "Updated only after new filings"))
        for e in self.leaders(4):
            m = e.metrics or {}
            if (m.get("turnover") or 0) > 0.35:
                cands.append(e.child(f"ts_decay_linear({e.expr}, 5)", "Smoothed with ts_decay_linear(x, 5)",
                                     "combine"))
                cands.append(e.child(e.expr, "Decay raised to cut turnover", "combine",
                                     settings={**e.settings, "decay": int(e.settings.get("decay", 0) or 0) + 6}))
            if "bucket(" not in e.expr:
                cands.append(e.child(f"group_neutralize({e.expr}, {CAPBUCKET_TEXT})", "Neutralized within size buckets",
                                     "combine"))
            if "trade_when" not in e.expr:
                for cond, lab in gates:
                    cands.append(e.child(f"trade_when({cond}, {e.expr}, -1)", lab, "combine"))
            if "subindustry" in e.expr:
                cands.append(e.child(e.expr.replace("subindustry", "industry"), "Peer group widened to industry",
                                     "combine"))
            elif "industry" in e.expr:
                cands.append(e.child(e.expr.replace("industry", "subindustry"), "Peer group narrowed to sub-industry",
                                     "combine"))
            neut = str(e.settings.get("neutralization", "SUBINDUSTRY")).upper()
            alt = "INDUSTRY" if neut == "SUBINDUSTRY" else "SUBINDUSTRY"
            cands.append(e.child(e.expr, f"Neutralization {alt.lower()}", "combine",
                                 settings={**e.settings, "neutralization": alt}))
        out = self.screen(self.fresh(cands, int(self.cfg["combine"])), "Combine")
        self.end("combine", out)

    def doctor(self, entries: list[Entry], stage: str, label: str) -> list[Entry]:
        """Screen the Doctor's rewrites for entries that fail an IS check (each entry is treated once)."""
        fixes: list[Entry] = []
        for e in entries:
            m = e.metrics or {}
            failed = self.failing(m)
            if not failed or e.extra.get("doctored"):
                continue
            e.extra["doctored"] = True
            try:
                node = lower_text(e.expr)
            except Exception:  # noqa: BLE001
                continue
            for f in diagnose(node, e.settings, failed, m, has_volume=self.has_volume)[: int(self.cfg["fixes_per"])]:
                fixes.append(e.child(to_expr(f.node), f"Doctor: {f.label}", stage, settings=f.settings,
                                     origin="doctor"))
        return self.screen(self.fresh(fixes), label)

    def sweep(self, entries: list[Entry], stage: str, label: str) -> list[Entry]:
        """Robust decay x neutralization sweep; the neighbourhood-best cell joins the pool."""
        out = []
        cfg = self.ws.checks_cfg
        for e in entries:
            if not self.can_run() or e.extra.get("swept"):
                continue
            self.h.check()
            e.extra["swept"] = True
            cells = [{"decay": d, "neutralization": n, "truncation": float(e.settings.get("truncation", 0.08))}
                     for d in SWEEP_DECAYS for n in SWEEP_NEUTS]
            self.h.update(phase=f"{label}: settings sweep ({len(cells)} cells)", done=0, total=len(cells))
            res = self.mgr.evaluate(self.h, [{"expr": e.expr, "settings": {**e.settings, **c}} for c in cells], "is",
                                    on_result=self._on_result)
            scored = []
            for c, r in zip(cells, res):
                if r.get("ok"):
                    m = (r.get("metrics") or {}).get("is") or {}
                    scored.append({**c, "sharpe": m.get("sharpe"), "fitness": m.get("fitness"),
                                   "turnover": m.get("turnover"), "_m": m})
            pick = robust_pick(scored, lambda c: (c["fitness"] or -5) - (0 if cfg["turnover_min"] <= (c["turnover"] or 0)
                                                                         <= cfg["turnover_max"] else 2))
            if not pick:
                continue
            s = {**e.settings, "decay": pick["decay"], "neutralization": pick["neutralization"]}
            ch = e.child(e.expr, f"Settings sweep: decay {pick['decay']}, {pick['neutralization'].lower()} "
                                 f"neutralization (robust pick of {len(scored)} cells)", stage, settings=s,
                         origin=e.origin)
            ch.canon = e.canon
            ch.extra["swept"] = True
            if ch.key == e.key:
                continue  # the current settings already are the robust pick
            self.tried.add(ch.key)
            if self.admit(ch, pick["_m"]):
                out.append(ch)
        self.h.update(leaderboard=self.leaderboard())
        return out

    def shape(self, entries: list[Entry], stage: str, label: str) -> list[Entry]:
        """Fitness shaping (turnover, peer ranking, weight profile, neutralization) of the leaders."""
        cands: list[Entry] = []
        for e in entries:
            if e.extra.get("shaped") or not e.metrics:
                continue
            e.extra["shaped"] = True
            try:
                node = lower_text(e.expr)
            except Exception:  # noqa: BLE001
                continue
            for sh in shaping_variants(node, e.settings, e.metrics, has_volume=self.has_volume, limit=12):
                cands.append(e.child(to_expr(sh.node), f"Shaped: {sh.label}", stage, settings=sh.settings))
        out = self.screen(self.fresh(cands), label)
        cfg = self.ws.checks_cfg
        # a shaped variant counts only if it keeps the Sharpe and adds no failing limit
        for c in out:
            parent = next((e for e in entries if c.lineage[:-1] == e.lineage), None)
            if parent is not None and parent.metrics and not acceptable(c.metrics or {}, parent.metrics, cfg):
                c.score -= 0.5
        return out

    def stage_refine(self) -> None:
        self.begin("refine")
        out = self.doctor(self.leaders(int(self.cfg["refine_top"])), "refine", "Refine (Doctor)")
        if self.can_run():
            out += self.shape(self.leaders(int(self.cfg["shape_top"])), "refine", "Refine (fitness shaping)")
        out += self.sweep(self.leaders(int(self.cfg["sweep_top"])), "refine", "Refine")
        self.end("refine", out)

    def stage_compose(self) -> None:
        """Complex alphas: multi-statement programs built from the leaders of different mechanisms/cores."""
        self.begin("compose")
        picks: list[Entry] = []
        fams_seen: set[str] = set()
        for e in sorted(self.pool.values(), key=lambda e: -e.score):
            if e.sign < 0 or e.extra.get("foreign") or e.extra.get("pnl") is None or is_complex(e.expr):
                continue
            if e.origin in ("combo", "compose", "gp") or (analyze(e.expr).size or 0) > 16:
                continue  # compose single, readable signals; blends of blends overfit and read badly
            if e.family not in fams_seen or (len(picks) < 4 and all(p.canon != e.canon for p in picks)
                                             and len([p for p in picks if p.family == e.family]) < 2):
                fams_seen.add(e.family)
                picks.append(e)
            if len(picks) >= 5:
                break
        comps = [Component(e.expr, e.family, e.settings, float((e.metrics or {}).get("sharpe") or 0.0),
                           pnl=np.asarray(e.extra["pnl"], dtype=np.float64)) for e in picks]
        if not comps:
            self.skip("compose", "no simulated leaders to compose")
            return
        L = min(len(c.pnl) for c in comps)  # type: ignore[arg-type]
        C = np.eye(len(comps))
        if len(comps) > 1 and L >= 60:
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.nan_to_num(np.corrcoef(np.vstack([c.pnl[-L:] for c in comps])))  # type: ignore[index]
        progs = composites(comps, C, local_fields=self.local, has_volume=self.has_volume,
                           limit=int(self.cfg["compose"]))
        cands: list[Entry] = []
        for p in progs:
            head = next((e for e in picks if e.expr == p.parts[0].expr), picks[0])
            neut = str(head.settings.get("neutralization", self.base.get("neutralization", "SUBINDUSTRY")))
            cands.append(head.child(p.text, f"Composed ({p.structure}): {p.label}", "compose",
                                    settings={**head.settings, "decay": 0, "neutralization": neut}, origin="compose"))
        out = self.screen(self.fresh(cands), "Compose")
        if out and self.can_run():
            out += self.sweep(sorted(out, key=lambda e: -e.score)[:1], "compose", "Compose")
        self.end("compose", out)

    def stage_evolve(self) -> None:
        self.begin("evolve")
        seeds = []
        for e in self.leaders(10):
            try:
                seeds.append(Individual(lower_text(e.expr), dict(e.settings), origin="forge"))
            except Exception:  # noqa: BLE001
                continue
        spec = self.spec
        allowed = allowed_fields(spec) & self.local
        fund = (spec.local_families or ["reversion"])[0] in FUNDAMENTAL_FAMILIES
        minutes = max(0.5, min(float(self.cfg["gp_minutes"]), (self.time_left() - self.reserve) / 60))
        pop = int(self.cfg["gp_population"])
        # hall-of-fame bar relative to what the drafts reached, so GP can report improvements on hard ideas too
        best = [e.metrics for e in self.leaders(3) if e.metrics]
        bs = max((m.get("sharpe") or 0 for m in best), default=1.0)
        bf = max((m.get("fitness") or 0 for m in best), default=0.8)
        min_sharpe = min(1.0, max(0.3, 0.7 * bs))
        min_fitness = min(0.8, max(0.1, 0.7 * bf))
        gp_cfg = {"settings": self.base, "population": pop, "generations": int(self.cfg["gp_generations"]),
                  "islands": 2 if self.effort == "deep" else 1, "max_depth": 6, "max_nodes": 28,
                  "min_sharpe": min_sharpe, "min_fitness": min_fitness, "halving": pop >= 30,
                  "time_limit_min": minutes,
                  "fields": sorted(allowed), "seed": self.rng.randint(0, 10 ** 9),
                  "category_weights": {"fundamental": 0.7, "pv": 0.3} if fund else {"pv": 0.8, "fundamental": 0.2},
                  "allow_trade_when": self.has_volume}
        hof = run_gp(self.h, self.ws, self.mgr, gp_cfg, seeds, CorrelationIndex(self.ws._corr_dates()))
        out = []
        gens = self.cfg["gp_generations"]
        for ind in hof:
            fam = classify(ind.node)["idea"]
            e = Entry(ind.expr, dict(ind.settings), fam if fam in spec.families else spec.families[0] if spec.families
                      else fam, "evolve", [f"Evolved by genetic programming (up to {gens} generations) "
                                           f"from the leading candidates"], 0, "gp")
            e.canon = ind.node.digest
            if e.key in self.tried:
                continue
            self.tried.add(e.key)
            if self.admit(e, ind.metrics):
                out.append(e)
        self.h.update(stage="evolve", leaderboard=self.leaderboard())
        self.end("evolve", out)

    def stage_polish(self) -> dict:
        self.begin("polish")
        n = int(self.cfg["finalists"])
        min_fid = 0.3 if self.spec.key_fields else 0.0
        # the evolved leaders have not been tuned yet: one more Doctor pass and settings sweep
        if self.time_left() > self.reserve / 2:
            self.doctor(self.leaders(3, min_fid, on_idea=True), "polish", "Polish (Doctor)")
            self.sweep(self.leaders(int(self.cfg["sweep_top"]), min_fid, on_idea=True), "polish", "Polish")
        finalists = self.leaders(n, min_fid, on_idea=True)
        if len(finalists) < 2:
            finalists += [e for e in self.leaders(n) if e not in finalists][: n - len(finalists)]
        # always include the best candidate that uses everything the idea names, so the result can show it
        on_dir = [e for e in self.pool.values() if e.sign >= 0 and not e.extra.get("foreign")]
        top_fid = max((e.fidelity for e in on_dir), default=0.0)
        faithful = max((e for e in on_dir if e.fidelity >= top_fid - 1e-9), key=lambda e: e.score, default=None)
        if faithful is not None and faithful not in finalists:
            finalists.append(faithful)
        # the result always offers a simple (one-line) and a complex (multi-statement) alpha when both exist
        for want_complex in (False, True):
            if not any(is_complex(e.expr) == want_complex for e in finalists):
                best = max((e for e in self.pool.values() if is_complex(e.expr) == want_complex and e.sign >= 0
                            and e.fidelity >= min_fid), key=lambda e: e.score, default=None)
                if best is not None:
                    finalists.append(best)
        self.h.update(phase=f"Polish: full evaluation of {len(finalists)} finalists", done=0, total=len(finalists))
        res = self.mgr.evaluate(self.h, [{"expr": e.expr, "settings": e.settings, "extras": True} for e in finalists],
                                "full", on_result=lambda i, r: self.h.update(done=self.h.progress.get("done", 0) + 1))
        scored = []
        for e, r in zip(finalists, res):
            if not r.get("ok"):
                continue
            checks, pass_prob, failed_hard = score_result(self.ws, r)
            qual = quality_of(self.ws, r, checks)
            e.extra["quality"] = qual
            # quality (margins + robustness evidence) decides; fidelity keeps the result on the idea
            final = qual["score"] + {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5}[qual["grade"]]
            final += 0.8 * e.fidelity + 0.2 * pass_prob
            scored.append((final, e, r, checks, pass_prob))
        scored.sort(key=lambda t: -t[0])
        if not scored:
            self.end("polish", [], status="done", note="No finalist could be evaluated")
            return {}
        champ = scored[0][1]
        faith = None
        if champ.fidelity < top_fid - 1e-9 or champ.sign < 0:
            faith = next((t for t in scored if t[1].fidelity >= top_fid - 1e-9 and t[1].sign >= 0
                          and not t[1].extra.get("foreign")), None)
        idx = CorrelationIndex(self.ws._corr_dates())
        chosen = []
        for t in scored:
            _, e, r, _, _ = t
            pnl = _pnl(r, "pnl_is")
            d = _dates(self.ws, r.get("pnl_is_start"), len(pnl))
            if chosen:
                mc = idx.max_corr(d, pnl)
                if mc and mc["max_corr"] >= CORR_CAP:
                    continue
            idx.add(len(chosen), d, pnl, ((r.get("metrics") or {}).get("is") or {}).get("sharpe", 0.0))
            chosen.append(t)
            if len(chosen) > RUNNERS:
                break
        rationale = self.spec.rationale()

        def save(t: tuple, tags: list[str]) -> dict:
            final, e, r, checks, pp = t
            cand = Candidate(expr=e.expr, settings=e.settings, template_id=e.template_id, idea=e.family,
                             category=e.category, horizon=e.horizon, rationale=rationale, origin="forge")
            row = persist(self.h, self.ws, cand, r, None, None, tags=["forge", f"forge-{self.h.id}"] + tags)
            return self._summary(final, e, r, checks, pp, row)

        summaries = [save(t, ["forge-champion"] if k == 0 else []) for k, t in enumerate(chosen)]
        faithful_summary = None
        if faith is not None:
            faithful_summary = next((s for s, t in zip(summaries, chosen) if t is faith), None) or save(faith, [])
        kinds: dict[str, dict | None] = {}
        for label, want_complex in (("simple", False), ("complex", True)):
            t = next((t for t in scored if is_complex(t[1].expr) == want_complex), None)
            if t is None:
                kinds[label] = None
                continue
            kinds[label] = next((s for s, c in zip(summaries, chosen) if c is t), None) or \
                save(t, [f"forge-{label}"])
        self.end("polish", [t[1] for t in chosen])
        return {"champion": summaries[0] if summaries else None, "runners": summaries[1:],
                "faithful": faithful_summary, "simple": kinds["simple"], "complex": kinds["complex"]}

    def _summary(self, final: float, e: Entry, r: dict, checks: dict, pp: float, row: dict | None) -> dict:
        m = (r.get("metrics") or {}).get("is") or {}
        os_ = (r.get("metrics") or {}).get("os") or {}
        ex = r.get("extras") or {}
        return {
            "id": (row or {}).get("id"), "expr": e.expr, "settings": SimSettings.from_dict(e.settings).to_dict(),
            "family": e.family, "family_label": FAMILY_LABEL.get(e.family, e.family), "stage": e.stage,
            "lineage": e.lineage, "fidelity": e.fidelity, "score": round(final, 3),
            "sharpe": m.get("sharpe"), "fitness": m.get("fitness"), "turnover": m.get("turnover"),
            "returns": m.get("returns"), "drawdown": m.get("drawdown"), "margin": m.get("margin_bps"),
            "os_sharpe": os_.get("sharpe"), "os_fitness": os_.get("fitness"), "sub_sharpe": ex.get("sub_sharpe"),
            "stability": ex.get("stability"), "status": checks["status"], "failed": checks["failed"],
            "warnings": checks["warnings"], "pass_prob": pp,
            "expected_brain_sharpe": self.ws.cal.expected_brain_sharpe(m.get("sharpe") or 0.0),
            "complexity": r.get("size"), "sign": e.sign, "foreign": e.extra.get("foreign", []),
            "missing": missing_fields(set(re.findall(r"[a-z_][a-z_0-9]*", e.expr)), self.spec),
            "grade": (e.extra.get("quality") or {}).get("grade"),
            "quality": (e.extra.get("quality") or {}).get("score"),
            "quality_reasons": (e.extra.get("quality") or {}).get("reasons", []),
            "quality_evidence": (e.extra.get("quality") or {}).get("evidence", []),
            "complex": is_complex(e.expr),
        }

    def hypothesis(self) -> dict:
        fam = (self.spec.local_families or [None])[0]
        hyp = [e for e in self.pool.values() if e.sign == 1 and e.origin == "recipe" and e.family == fam]
        rev = [e for e in self.pool.values() if e.sign == -1 and e.family == fam]
        if not fam or not hyp or not rev:
            return {"holds": None, "text": ""}
        bh = max((e.metrics or {}).get("sharpe", -9) for e in hyp)
        br = max((e.metrics or {}).get("sharpe", -9) for e in rev)
        d = self.spec.direction.get(fam, 1)
        amount = {"reversion": "recent return", "momentum": "past return"}.get(fam, "")
        stated = "the direction in your idea" if self.spec.stated.get(fam) else "the conventional direction"
        if bh >= br - 0.1:
            txt = (f"The data supports {stated} for {FAMILY_LABEL[fam].lower()}: best in-sample Sharpe {bh:.2f} "
                   f"versus {br:.2f} for the reversed signal.")
            holds = True
        else:
            txt = (f"The data favours the opposite of {stated} for {FAMILY_LABEL[fam].lower()}: the reversed signal "
                   f"reached in-sample Sharpe {br:.2f} versus {bh:.2f}. Consider whether the mechanism runs the other "
                   f"way round.")
            holds = False
        return {"holds": holds, "text": txt, "sharpe_hyp": round(bh, 3), "sharpe_rev": round(br, 3),
                "family": fam, "direction": d, "amount": amount}

    def brain_only(self) -> list[dict]:
        out = []
        for d in brain_only_drafts(self.spec, self.local, self.base):
            r = self.ws.save_alpha(d.expr, d.settings, origin="forge", rationale=self.spec.rationale(),
                                   job_id=self.h.id, extras=False, tags=["forge", f"forge-{self.h.id}", "brain-only"])
            if r.get("ok"):
                self.h.bump("brain_only")
                self.h.bump("saved")
                out.append({"id": r["id"], "expr": d.expr, "label": d.label, "settings": d.settings})
        return out

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        h = self.h
        self.begin("interpret")
        spec_json = self.spec.to_json()
        self.end("interpret", note=f"{len(self.spec.local_families)} mechanism(s), {len(self.spec.fields)} field(s)")
        h.update(spec=spec_json, effort=self.effort)
        if not self.idea:
            raise ValueError("Paste an idea first")
        self.stage_draft()
        if not self.pool:
            h.update(forge={"champion": None, "runners": [], "message": "None of the drafts could be simulated "
                            "locally. Check the Data page, or name price/volume/fundamental data in the idea."})
            return
        for name, fn in (("combine", self.stage_combine), ("refine", self.stage_refine),
                         ("compose", self.stage_compose), ("evolve", self.stage_evolve)):
            h.check()
            if self.can_run():
                fn()
            else:
                self.skip(name, "time budget reached")
        h.check()
        result = self.stage_polish()
        champ = result.get("champion")
        notes = []
        if champ is None:
            notes.append("No finalist survived the full evaluation.")
        else:
            if champ["status"] != "PASS":
                notes.append("The best candidate does not yet pass every local BRAIN check "
                             f"({', '.join(champ['failed']) or 'see warnings'}). Try the Deep effort, add detail to "
                             "the idea, or open it in Studio and run Doctor.")
            elif champ.get("grade") not in ("A", None):
                why = "; ".join(champ.get("quality_reasons", [])[:2])
                notes.append(f"Quality grade {champ['grade']}: it passes locally but not with the safety margin "
                             f"that usually survives BRAIN ({why}). Verify it on BRAIN before relying on it.")
            if self.ws.panel.source == "demo":
                notes.append("These results come from the synthetic DEMO data and will not transfer to BRAIN.")
            if champ["sign"] < 0:
                notes.append("The champion trades the reverse of the direction in your idea because the data "
                             "favours it (see the hypothesis check).")
            if champ["missing"] and result.get("faithful"):
                notes.append(f"The champion leaves out {', '.join(champ['missing'])} from your idea because variants "
                             "using it scored lower here; the most faithful variant is shown separately.")
        msg = " ".join(notes)
        result.update({
            "message": msg, "hypothesis": self.hypothesis(), "brain_only": self.brain_only(),
            "evaluated": h.stats.get("evaluated", 0), "pool": len(self.pool),
            "seconds": round(time.time() - self.started, 1), "idea": self.idea,
        })
        h.update(forge=result, stage="done", stage_idx=len(STAGES), leaderboard=self.leaderboard(),
                 phase="done")
        h.event("forge_done", {"forge": result})


def run_forge(h: JobHandle, ws, mgr: JobManager, config: dict) -> None:
    Forge(h, ws, mgr, config).run()


__all__ = ["run_forge", "Forge", "EFFORT", "STAGES"]
