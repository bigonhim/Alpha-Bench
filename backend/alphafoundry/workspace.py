"""Workspace: the application's runtime state and high-level operations.

Holds the active panel, periods, subexpression cache, SQLite store, correlation indexes and the
calibration model, and implements simulate / extras / sweep / doctor / save / import / export used by
the API and background jobs. Engine calls in the API process are serialized with a lock because
numba's default threading layer does not support concurrent parallel launches.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Iterator

import numpy as np

from . import config
from .brainio import export as bexport
from .brainio import importer as bimport
from .brainio.calibration import Calibrator, summary as cal_summary
from .catalog import field_map
from .engine.evaluator import EvalError, SubexprCache
from .engine.panel import Panel
from .fastexpr import analyze, lower_text, to_expr
from .fastexpr.ast import Node
from .fastexpr.explain import classify, description
from .fastexpr.lower import node_info
from .gen.doctor import diagnose
from .sim.checks import FAIL, HARD_CHECKS, load_checks_config, run_checks
from .sim.correlation import CorrelationIndex, cluster_order
from .sim.robustness import perturbed_variants, robust_pick, stability_score, sweep_grid
from .sim.simulator import (Periods, SimResult, SimSettings, compute_periods, deflated_sharpe, resolve_sub_universe,
                            resolve_universe, simulate, sweep_cells)
from .store.db import Store

log = logging.getLogger("alphafoundry")


class Workspace:
    def __init__(self) -> None:
        config.ensure_dirs()
        self.settings = config.load_settings()
        self.checks_cfg = load_checks_config()
        self.lock = threading.RLock()
        self.store = Store(config.DB_PATH)
        self.cal = Calibrator(self.store.kv_get("calibration"))
        self.cache = SubexprCache(self.settings.get("api_cache_mb", 600))
        self._recent: OrderedDict[str, SimResult] = OrderedDict()
        self.panel: Panel | None = None
        self.periods: Periods | None = None
        self.corr_lib: CorrelationIndex | None = None
        self.corr_sub: CorrelationIndex | None = None
        self.load_panel()

    # ------------------------------------------------------------------ dataset
    def panel_path(self):
        pref = self.settings.get("active_dataset", "auto")
        real_ok = (config.PANELS_DIR / "meta.json").exists()
        if pref == "real" and real_ok:
            return config.PANELS_DIR
        if pref == "demo" or not real_ok:
            if not (config.DEMO_PANELS_DIR / "meta.json").exists():
                from .data.demo import build_demo
                build_demo(config.DEMO_PANELS_DIR)
            return config.DEMO_PANELS_DIR
        return config.PANELS_DIR

    def load_panel(self) -> None:
        with self.lock:
            self.panel = Panel(self.panel_path())
            self.periods = compute_periods(self.panel, self.settings)
            self.cache.clear()
            self._recent.clear()
            self._build_corr_indexes()

    def local_fields(self) -> set[str]:
        assert self.panel is not None
        return set(self.panel.field_names()) | set(self.panel.group_labels) | {"market", "country"}

    def _corr_dates(self) -> np.ndarray:
        p, per = self.panel, self.periods
        years = float(self.checks_cfg.get("self_corr_years", 4))
        start = max(per.is_start, p.index_of(p.dates[max(0, per.os_start - 1)] - np.timedelta64(int(365.25 * years), "D")))
        return p.dates[start:per.os_start]

    def _build_corr_indexes(self) -> None:
        dates = self._corr_dates()
        self.corr_lib = CorrelationIndex(dates)
        self.corr_sub = CorrelationIndex(dates)
        for r in self.store.pnl_rows(limit=4000):
            self._index_row(r)

    def _index_row(self, r: dict) -> None:
        if r["data_version"] != self.panel.version or not len(r["pnl"]):
            return
        start = self.panel.index_of(r["dates_start"])
        dates = self.panel.dates[start:start + len(r["pnl"])]
        n = min(len(dates), len(r["pnl"]))
        self.corr_lib.add(r["id"], dates[:n], r["pnl"][:n], r["sharpe"])
        if r["submitted"]:
            self.corr_sub.add(r["id"], dates[:n], r["pnl"][:n], r["sharpe"])

    def status(self) -> dict:
        p = self.panel
        return {
            "data": p.info() if p else None,
            "periods": self.periods.to_dict(p) if p else None,
            "cache": self.cache.stats(),
            "db": self.store.stats(),
            "real_data_available": (config.PANELS_DIR / "meta.json").exists(),
            "settings": self.settings,
            "library_indexed": len(self.corr_lib or []),
            "submitted_indexed": len(self.corr_sub or []),
        }

    # ------------------------------------------------------------------ analysis
    def analyze(self, text: str) -> dict:
        an = analyze(text, local_fields=self.local_fields())
        d = an.to_json()
        if an.ok and an.node is not None:
            d["tags"] = classify(an.node)
        return d

    # ------------------------------------------------------------------ simulation
    def _sim(self, node: Node, s: SimSettings, **kw) -> SimResult:
        return simulate(node, s, self.panel, self.periods, self.cache,
                        booksize=float(self.settings.get("booksize", 20e6)),
                        returns_basis=self.settings.get("returns_basis", "half_book"), **kw)

    def _recent_key(self, node: Node, s: SimSettings) -> str:
        return node.key + "||" + s.key()

    def _remember(self, key: str, res: SimResult) -> None:
        self._recent[key] = res
        self._recent.move_to_end(key)
        while len(self._recent) > 24:
            self._recent.popitem(last=False)

    def _recent2y(self, res: SimResult) -> float | None:
        a, b = res.idx["is"]
        seg = res.pnl[max(a, b - 504):b]
        if len(seg) < 100 or seg.std() == 0:
            return None
        return float(seg.mean() / seg.std(ddof=1) * np.sqrt(252))

    def _self_corr(self, res: SimResult, exclude: set[int] | None = None) -> dict | None:
        if not self.corr_sub or not len(self.corr_sub):
            return None
        a, b = res.idx["is"]
        return self.corr_sub.max_corr(res.dates[a:b], res.pnl[a:b], exclude=exclude)

    def _pack(self, node: Node, an_json: dict, s: SimSettings, res: SimResult, checks: dict, extras: dict | None,
              rationale: str | None = None, n_trials: int = 1) -> dict:
        m_is = res.metrics.get("is", {})
        feats = {**m_is, "os_sharpe": res.metrics.get("os", {}).get("sharpe"),
                 "sub_sharpe": (extras or {}).get("sub_sharpe"), "complexity": an_json.get("size")}
        failed_hard = sum(1 for c in checks["checks"] if c["name"] in HARD_CHECKS and c["result"] == FAIL)
        a, b = res.idx["is"]
        top = self.corr_lib.top(res.dates[a:b], res.pnl[a:b], n=6) if self.corr_lib else []
        pnl_is = res.pnl[a:b]
        skew = float(((pnl_is - pnl_is.mean()) ** 3).mean() / (pnl_is.std() ** 3 + 1e-12)) if len(pnl_is) > 10 else 0.0
        kurt = float(((pnl_is - pnl_is.mean()) ** 4).mean() / (pnl_is.var() ** 2 + 1e-12)) if len(pnl_is) > 10 else 3.0
        return {
            "ok": True,
            "brain_only": False,
            "analysis": an_json,
            "settings": s.to_dict(),
            "local_universe": res.local_universe,
            "universe_size": round(res.universe_size, 1),
            "metrics": res.metrics,
            "yearly": res.yearly,
            "series": res.series_json(),
            "periods": {k: [int(v[0]), int(v[1])] for k, v in res.idx.items()},
            "checks": checks,
            "extras": extras,
            "sector_pnl": res.sector_pnl,
            "top_names": res.top_names,
            "correlation": {"top": top},
            "pass_prob": self.cal.predict(feats, failed_hard),
            "expected_brain_sharpe": self.cal.expected_brain_sharpe(m_is.get("sharpe", 0.0)),
            "dsr": deflated_sharpe(m_is.get("sharpe", 0.0), n_trials, len(pnl_is), skew, kurt),
            "description": description(node, rationale),
            "elapsed_ms": res.elapsed_ms,
            "data": {"source": self.panel.source, "version": self.panel.version},
        }

    def simulate(self, text: str, settings: dict | None = None, extras: bool = False,
                 rationale: str | None = None) -> dict:
        t0 = time.perf_counter()
        an = analyze(text, local_fields=self.local_fields())
        an_json = an.to_json()
        s = SimSettings.from_dict(settings)
        if not an.ok:
            return {"ok": False, "analysis": an_json, "settings": s.to_dict()}
        node = an.node
        an_json["tags"] = classify(node)
        if not an.local:
            return {"ok": True, "brain_only": True, "analysis": an_json, "settings": s.to_dict(),
                    "description": description(node, rationale),
                    "message": "This alpha uses data or operators that are not available locally. "
                               "Export it and simulate it on BRAIN."}
        try:
            with self.lock:
                res = self._sim(node, s, span="all")
                self._remember(self._recent_key(node, s), res)
                ex = self._extras(node, s, res) if extras else None
        except EvalError as e:
            return {"ok": False, "analysis": an_json, "settings": s.to_dict(), "error": str(e)}
        checks = self._checks(res, s, ex)
        out = self._pack(node, an_json, s, res, checks, ex, rationale)
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def _checks(self, res: SimResult, s: SimSettings, ex: dict | None, exclude: set[int] | None = None) -> dict:
        sub = None
        stability = None
        if ex:
            if ex.get("sub_sharpe") is not None:
                sub = {"sharpe": ex["sub_sharpe"], "sub_size": ex["sub_size"], "univ_size": ex["univ_size"],
                       "name": ex.get("sub_universe")}
            stability = ex.get("stability")
        return run_checks(res.metrics, s.delay, self.checks_cfg, sub=sub, self_corr=self._self_corr(res, exclude),
                          stability=stability, yearly=res.yearly, recent2y_sharpe=self._recent2y(res))

    def _extras(self, node: Node, s: SimSettings, res: SimResult) -> dict:
        out: dict[str, Any] = {}
        sub_u = resolve_sub_universe(self.panel, res.local_universe)
        if sub_u:
            sres = self._sim(node, s, span="is", universe_override=sub_u)
            out.update({"sub_universe": sub_u, "sub_sharpe": sres.metrics.get("is", {}).get("sharpe", 0.0),
                        "sub_size": sres.universe_size, "univ_size": res.universe_size,
                        "sub_metrics": sres.metrics.get("is")})
        base = res.metrics.get("is", {}).get("sharpe", 0.0)
        variants = perturbed_variants(node)
        vs = []
        for v in variants:
            try:
                vr = self._sim(v, s, span="is")
                vs.append({"expr": to_expr(v), "sharpe": vr.metrics.get("is", {}).get("sharpe", 0.0),
                           "fitness": vr.metrics.get("is", {}).get("fitness", 0.0)})
            except EvalError:
                continue
        out["variants"] = vs
        out["stability"] = stability_score(base, [v["sharpe"] for v in vs])
        return out

    def extras(self, text: str, settings: dict | None = None) -> dict:
        an = analyze(text, local_fields=self.local_fields())
        if not an.ok or not an.local:
            return {"ok": False}
        s = SimSettings.from_dict(settings)
        node = an.node
        with self.lock:
            res = self._recent.get(self._recent_key(node, s)) or self._sim(node, s, span="all")
            ex = self._extras(node, s, res)
        checks = self._checks(res, s, ex)
        m_is = res.metrics.get("is", {})
        failed_hard = sum(1 for c in checks["checks"] if c["name"] in HARD_CHECKS and c["result"] == FAIL)
        feats = {**m_is, "os_sharpe": res.metrics.get("os", {}).get("sharpe"), "sub_sharpe": ex.get("sub_sharpe"),
                 "complexity": node.size}
        return {"ok": True, "extras": ex, "checks": checks, "pass_prob": self.cal.predict(feats, failed_hard)}

    # ------------------------------------------------------------------ sweep / doctor
    def sweep(self, text: str, settings: dict | None = None, grid: dict | None = None) -> Iterator[dict]:
        an = analyze(text, local_fields=self.local_fields())
        if not an.ok or not an.local:
            return
        base = SimSettings.from_dict(settings)
        node = an.node
        cells = sweep_grid(grid)
        results = []
        bs = float(self.settings.get("booksize", 20e6))
        rb = self.settings.get("returns_basis", "half_book")
        gen = sweep_cells(node, base, cells, self.panel, self.periods, self.cache, bs, rb)
        while True:
            with self.lock:
                nxt = next(gen, None)
            if nxt is None:
                break
            k, m = nxt
            c = cells[k]
            cell = {**c, "sharpe": m.get("sharpe"), "fitness": m.get("fitness"), "turnover": m.get("turnover"),
                    "returns": m.get("returns"), "max_weight": m.get("max_weight")}
            results.append(cell)
            yield {"type": "cell", "cell": cell}
        cfg = self.checks_cfg

        def score(c):
            ok_to = cfg["turnover_min"] <= (c["turnover"] or 0) <= cfg["turnover_max"]
            return (c["fitness"] or -5) - (0 if ok_to else 2.0)
        best = robust_pick(results, score)
        yield {"type": "best", "cell": best}

    def doctor(self, text: str, settings: dict | None = None, max_fixes: int = 12) -> dict:
        base = self.simulate(text, settings, extras=True)
        if not base.get("ok") or base.get("brain_only"):
            return {"ok": False, "base": base}
        node = lower_text(text)
        s = SimSettings.from_dict(settings)
        failed = base["checks"]["failed"]
        warnings = base["checks"]["warnings"]
        corr_node = None
        sc = next((c for c in base["checks"]["checks"] if c["name"] == "SELF_CORRELATION"), None)
        if sc and sc["result"] == FAIL:
            m = self._corr_partner_expr(base)
            if m:
                try:
                    corr_node = lower_text(m)
                except Exception:  # noqa: BLE001
                    corr_node = None
        fixes = diagnose(node, s.to_dict(), failed, base["metrics"].get("is"), warnings, corr_node,
                         has_volume={"volume", "adv20"} <= self.local_fields())
        if not failed and not warnings:
            fixes = fixes[:0]
        out = []
        base_is = base["metrics"].get("is", {})
        for f in fixes[:max_fixes]:
            info = node_info(f.node, local_fields=self.local_fields())
            if not info["local"]:
                continue
            fs = SimSettings.from_dict(f.settings)
            try:
                with self.lock:
                    r = self._sim(f.node, fs, span="all")
            except EvalError:
                continue
            chk = self._checks(r, fs, None)
            m = r.metrics.get("is", {})
            n_fail = len(chk["failed"])
            out.append({**f.to_json(), "metrics": m, "os": r.metrics.get("os"), "failed": chk["failed"],
                        "n_failed": n_fail,
                        "delta": {k: round((m.get(k) or 0) - (base_is.get(k) or 0), 4)
                                  for k in ("sharpe", "fitness", "turnover", "returns")}})
        out.sort(key=lambda x: (x["n_failed"], -(x["metrics"].get("fitness") or -9)))
        return {"ok": True, "base_failed": failed, "base_warnings": warnings, "fixes": out,
                "base_metrics": base_is}

    def _corr_partner_expr(self, base: dict) -> str | None:
        top = (base.get("correlation") or {}).get("top") or []
        for t in top:
            a = self.store.get_alpha(t["alpha_id"])
            if a and a.get("submitted"):
                return a["expr"]
        return None

    # ------------------------------------------------------------------ persistence
    def save_alpha(self, text: str, settings: dict | None, origin: str = "manual", *, template_id: str | None = None,
                   rationale: str | None = None, parents: list | None = None, job_id: int | None = None,
                   tags: list[str] | None = None, notes: str = "", result: dict | None = None,
                   extras: bool = True) -> dict:
        res = result or self.simulate(text, settings, extras=extras, rationale=rationale)
        if not res.get("ok"):
            return {"ok": False, "result": res}
        an = res["analysis"]
        s = SimSettings.from_dict(res.get("settings") or settings)
        tags_ = res["analysis"].get("tags") or {}
        rec = {
            "expr": text.strip(), "canon": an["canonical"], "canon_hash": an["canon_hash"],
            "settings": s.to_dict(), "settings_key": s.key(), "origin": origin, "family": tags_.get("idea"),
            "idea": tags_.get("idea"), "category": tags_.get("category"), "horizon": tags_.get("horizon"),
            "template_id": template_id, "parents": parents or [], "local": int(not res.get("brain_only")),
            "brain_only_reasons": an.get("brain_only_reasons"), "complexity": an.get("size"),
            "description": res.get("description"), "job_id": job_id,
        }
        if tags:
            rec["tags"] = ",".join(tags)
        if notes:
            rec["notes"] = notes
        if res.get("brain_only"):
            rec.update({"status_local": "UNSCORED"})
            aid = self.store.upsert_alpha(rec)
            return {"ok": True, "id": aid, "brain_only": True}
        m = res["metrics"].get("is", {})
        ex = res.get("extras") or {}
        rec.update({
            "sharpe": m.get("sharpe"), "fitness": m.get("fitness"), "turnover": m.get("turnover"),
            "returns": m.get("returns"), "drawdown": m.get("drawdown"), "margin": m.get("margin_bps"),
            "os_sharpe": (res["metrics"].get("os") or {}).get("sharpe"), "sub_sharpe": ex.get("sub_sharpe"),
            "max_corr": (res["correlation"]["top"][0]["corr"] if res["correlation"]["top"] else None),
            "pass_prob": res.get("pass_prob"), "status_local": res["checks"]["status"],
            "robust": int(bool(res["checks"].get("robust"))), "failed": res["checks"]["failed"],
        })
        aid = self.store.upsert_alpha(rec)
        payload = {k: res.get(k) for k in ("metrics", "yearly", "checks", "extras", "sector_pnl", "top_names",
                                           "dsr", "pass_prob", "local_universe", "universe_size")}
        pnl = np.asarray(res["series"]["cum_pnl"], dtype=np.float64)
        daily = np.diff(np.r_[0.0, pnl])
        dates_start = res["series"]["dates"][0]
        self.store.save_result(aid, self.panel.version, payload, daily, dates_start)
        self._index_row({"id": aid, "sharpe": m.get("sharpe") or 0.0, "submitted": False, "pnl": daily.astype(np.float32),
                         "dates_start": dates_start, "data_version": self.panel.version})
        return {"ok": True, "id": aid, "status": res["checks"]["status"]}

    def save_result_record(self, rec: dict, payload: dict, pnl: np.ndarray, dates_start: str) -> int:
        """Persist a result computed elsewhere (e.g. by a miner worker)."""
        aid = self.store.upsert_alpha(rec)
        self.store.save_result(aid, self.panel.version, payload, pnl, dates_start)
        self._index_row({"id": aid, "sharpe": rec.get("sharpe") or 0.0, "submitted": False, "pnl": pnl,
                         "dates_start": dates_start, "data_version": self.panel.version})
        return aid

    def set_submitted(self, alpha_id: int, submitted: bool) -> None:
        self.store.update_alpha(alpha_id, {"submitted": submitted, "status_brain": "submitted" if submitted else None})
        if submitted:
            r = self.store.pnl_rows("a.id=?", (alpha_id,), limit=1)
            if r and r[0]["data_version"] == self.panel.version:
                start = self.panel.index_of(r[0]["dates_start"])
                dates = self.panel.dates[start:start + len(r[0]["pnl"])]
                n = min(len(dates), len(r[0]["pnl"]))
                self.corr_sub.add(alpha_id, dates[:n], r[0]["pnl"][:n], r[0]["sharpe"])
        else:
            self.corr_sub.remove(alpha_id)

    def alpha_detail(self, alpha_id: int) -> dict | None:
        a = self.store.get_alpha(alpha_id)
        if not a:
            return None
        out = {"alpha": a, "result": None, "brain": [b for b in self.store.brain_results() if b["alpha_id"] == alpha_id]}
        if a.get("local"):
            res = self.simulate(a["expr"], a.get("settings"), extras=False)
            if res.get("ok") and not res.get("brain_only"):
                stored = self.store.get_result(alpha_id)
                if stored:
                    res["extras"] = stored["payload"].get("extras")
                    res["checks"] = stored["payload"].get("checks") or res["checks"]
                if isinstance(a.get("description"), dict):
                    res["description"] = a["description"]  # keeps the rationale it was saved with
                out["result"] = res
        return out

    # ------------------------------------------------------------------ correlation / combine
    def correlation_matrix(self, ids: list[int]) -> dict:
        ids_in, C = self.corr_lib.matrix(ids)
        order = cluster_order(C) if len(ids_in) > 2 else list(range(len(ids_in)))
        ids_o = [ids_in[k] for k in order]
        C = C[np.ix_(order, order)] if len(order) else C
        missing = [i for i in ids if i not in ids_in]
        return {"ids": ids_o, "matrix": np.round(C, 3).tolist(), "missing": missing}

    def combine(self, ids: list[int], method: str = "equal") -> dict:
        rows = [r for r in self.store.pnl_rows(f"a.id IN ({','.join('?' for _ in ids)})", ids, limit=len(ids))
                if r["data_version"] == self.panel.version]
        if not rows:
            return {"ok": False, "error": "No simulated alphas selected"}
        start = min(self.panel.index_of(r["dates_start"]) for r in rows)
        end = max(self.panel.index_of(r["dates_start"]) + len(r["pnl"]) for r in rows)
        T = end - start
        M = np.zeros((len(rows), T))
        for k, r in enumerate(rows):
            s0 = self.panel.index_of(r["dates_start"]) - start
            M[k, s0:s0 + len(r["pnl"])] = r["pnl"]
        a, b = self.periods.is_start - start, self.periods.os_start - start
        a = max(0, a)
        vol = M[:, a:b].std(axis=1) + 1e-9
        sh = M[:, a:b].mean(axis=1) / vol
        if method == "inverse_vol":
            w = 1 / vol
        elif method == "sharpe":
            w = np.clip(sh, 0, None)
            if w.sum() <= 0:
                w = np.ones(len(rows))
        else:
            w = np.ones(len(rows))
        w = w / w.sum()
        comb = w @ M
        dates = self.panel.dates[start:end]

        def stats(x):
            if len(x) < 2 or x.std() == 0:
                return {"sharpe": 0.0}
            return {"sharpe": round(float(x.mean() / x.std(ddof=1) * np.sqrt(252)), 3),
                    "returns": round(float(x.mean() * 252 / (self.settings.get("booksize", 20e6) / 2)), 5)}
        C = np.corrcoef(M[:, a:b]) if len(rows) > 1 else np.ones((1, 1))
        avg_corr = float((C.sum() - len(rows)) / max(1, len(rows) * (len(rows) - 1))) if len(rows) > 1 else 1.0
        return {"ok": True, "ids": [r["id"] for r in rows], "weights": np.round(w, 4).tolist(),
                "dates": [str(d) for d in dates], "cum_pnl": np.round(np.cumsum(comb), 0).tolist(),
                "is": stats(comb[a:b]), "os": stats(comb[b:]), "avg_pairwise_corr": round(avg_corr, 3),
                "components": [{"id": r["id"], "is_sharpe": round(float(sh[k] * np.sqrt(252)), 3)}
                               for k, r in enumerate(rows)]}

    # ------------------------------------------------------------------ import / export / calibration
    def import_results(self, text: str, fmt: str = "auto", mark_submitted: bool = False) -> dict:
        rows = bimport.parse_any(text, fmt)
        summary = {"parsed": len(rows), "matched": 0, "created": 0, "errors": [], "items": []}
        from .gen.bandit import Bandit
        bandit = Bandit(self.store)
        for r in rows:
            an = analyze(r["expr"], local_fields=self.local_fields())
            if not an.ok:
                summary["errors"].append({"expr": r["expr"][:120], "error": an.diagnostics[-1].message})
                continue
            s = SimSettings.from_dict(r["settings"] or {})
            existing = self.store.find_alpha(an.canon_hash, s.key() if r["settings"] else None)
            if existing:
                aid = existing["id"]
                summary["matched"] += 1
            else:
                saved = self.save_alpha(r["expr"], s.to_dict(), origin="import", extras=False)
                if not saved.get("ok"):
                    summary["errors"].append({"expr": r["expr"][:120], "error": "could not save"})
                    continue
                aid = saved["id"]
                summary["created"] += 1
            m = dict(r["metrics"])
            m["passed"] = bimport.infer_passed(m, int(s.delay))
            self.store.add_brain_result(aid, m, r["raw"])
            status = "passed" if m["passed"] else ("failed" if m["passed"] is False else "tested")
            if r.get("submitted") or mark_submitted:
                self.set_submitted(aid, True)
                status = "submitted"
            self.store.update_alpha(aid, {"status_brain": status})
            a = self.store.get_alpha(aid) or {}
            if m["passed"] is not None:
                for arm in (f"idea:{a.get('idea') or 'other'}", f"cat:{a.get('category') or 'pv'}"):
                    bandit.update(arm, 1.0 if m["passed"] else 0.0, weight=5.0)
            summary["items"].append({"id": aid, "expr": r["expr"][:160], "passed": m["passed"],
                                     "brain_sharpe": m.get("sharpe"), "local_sharpe": a.get("sharpe")})
        self.refit_calibration()
        return summary

    def refit_calibration(self) -> dict:
        rows = self.store.brain_results()
        st = self.cal.fit(rows)
        self.store.kv_set("calibration", st)
        return st

    def calibration(self) -> dict:
        return cal_summary(self.store.brain_results(), self.cal)

    def export(self, ids: list[int], fmt: str = "json", batch: int = 10) -> str:
        alphas = self.store.list_alphas(ids=ids, limit=max(1, len(ids)))["rows"]
        order = {i: k for k, i in enumerate(ids)}
        alphas.sort(key=lambda a: order.get(a["id"], 0))
        if fmt == "csv":
            return bexport.to_csv(alphas)
        if fmt == "text":
            return bexport.to_text(alphas, with_settings=True)
        if fmt == "markdown":
            parts = []
            for a in alphas:
                r = self.store.get_result(a["id"])
                parts.append(bexport.to_markdown(a, (r or {}).get("payload")))
            return "\n\n---\n\n".join(parts)
        return bexport.to_json(alphas, batch=batch)

    # ------------------------------------------------------------------ settings
    def update_settings(self, updates: dict) -> dict:
        reload = any(k in updates for k in ("is_years_warmup", "os_years", "brain_window_years", "active_dataset",
                                            "returns_basis", "booksize"))
        self.settings = config.save_settings(updates)
        if "api_cache_mb" in updates:
            self.cache.max_bytes = int(float(self.settings["api_cache_mb"]) * 1048576)
        if reload:
            self.load_panel()
        return self.settings

    def reload_checks(self) -> None:
        self.checks_cfg = load_checks_config()
        self._build_corr_indexes()


_ws: Workspace | None = None
_ws_lock = threading.Lock()


def get_workspace() -> Workspace:
    global _ws
    with _ws_lock:
        if _ws is None:
            _ws = Workspace()
        return _ws


def default_settings() -> dict:
    return SimSettings().to_dict()


def universes_for(panel: Panel) -> dict:
    return {u: resolve_universe(panel, u) for u in ("TOP3000", "TOP1000", "TOP500", "TOP200", "TOPSP500")}


__all__ = ["Workspace", "get_workspace", "default_settings", "field_map", "universes_for"]
