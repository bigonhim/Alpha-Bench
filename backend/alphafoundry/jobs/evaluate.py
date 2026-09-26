"""Evaluation core shared by the process-pool workers and the in-process fallback."""

from __future__ import annotations

import traceback

import numpy as np

from ..engine.evaluator import EvalError, SubexprCache
from ..engine.panel import Panel
from ..fastexpr import analyze
from ..sim.robustness import perturbed_variants, stability_score
from ..sim.simulator import Periods, SimSettings, resolve_sub_universe, simulate


def screen_rows(panel: Panel, periods: Periods, years: float = 3.0) -> tuple[int, int]:
    start = panel.index_of(panel.dates[max(0, periods.os_start - 1)] - np.timedelta64(int(365.25 * years), "D"))
    return max(periods.is_start, start), periods.os_start


def evaluate_one(panel: Panel, periods: Periods, cache: SubexprCache, payload: dict, span: str, cfg: dict,
                 local_fields: set[str]) -> dict:
    """span: 'screen' (last 3 IS years), 'is' (full IS), 'full' (IS+OS with sub-universe & stability)."""
    expr = payload["expr"]
    try:
        an = analyze(expr, local_fields=local_fields)
        if not an.ok:
            return {"ok": False, "error": an.diagnostics[-1].message if an.diagnostics else "invalid"}
        if not an.local:
            return {"ok": False, "brain_only": True, "error": "; ".join(an.brain_only_reasons)}
        node = an.node
        s = SimSettings.from_dict(payload.get("settings"))
        kw = dict(booksize=float(cfg.get("booksize", 20e6)), returns_basis=cfg.get("returns_basis", "half_book"))
        if span == "screen":
            rows = screen_rows(panel, periods)
            res = simulate(node, s, panel, periods, cache, rows=rows, **kw)
        elif span == "is":
            res = simulate(node, s, panel, periods, cache, span="is", **kw)
        else:
            res = simulate(node, s, panel, periods, cache, span="all", **kw)
        m_all = res.metrics
        out = {
            "ok": True, "canonical": an.canonical, "canon_hash": an.canon_hash, "size": an.size,
            "fields": an.fields, "operators": an.operators, "categories": an.categories,
            "settings": s.to_dict(), "local_universe": res.local_universe,
        }
        if span == "screen":
            out["metrics"] = {"is": m_all.get("all") or m_all.get("is") or {}}
        else:
            out["metrics"] = m_all
        a, b = res.idx["is"] if span != "screen" else res.idx["all"]
        out["pnl_is"] = np.asarray(res.pnl[a:b], dtype=np.float32).tobytes()
        out["pnl_is_start"] = str(res.dates[a]) if b > a else None
        if span == "full":
            out["yearly"] = res.yearly
            out["sector_pnl"] = res.sector_pnl
            out["top_names"] = res.top_names
            out["universe_size"] = res.universe_size
            out["pnl"] = np.asarray(res.pnl, dtype=np.float32).tobytes()
            out["dates_start"] = str(res.dates[0])
            ia, ib = res.idx["is"]
            seg = res.pnl[max(ia, ib - 504):ib]
            out["recent2y"] = float(seg.mean() / seg.std(ddof=1) * np.sqrt(252)) if len(seg) > 100 and seg.std() > 0 else None
            ex: dict = {}
            sub_u = resolve_sub_universe(panel, res.local_universe)
            if sub_u and payload.get("extras", True):
                sres = simulate(node, s, panel, periods, cache, span="is", universe_override=sub_u, **kw)
                ex.update({"sub_universe": sub_u, "sub_sharpe": sres.metrics.get("is", {}).get("sharpe", 0.0),
                           "sub_size": sres.universe_size, "univ_size": res.universe_size})
            if payload.get("extras", True):
                base = res.metrics.get("is", {}).get("sharpe", 0.0)
                vs = []
                for v in perturbed_variants(node, max_variants=2):
                    try:
                        vr = simulate(v, s, panel, periods, cache, span="is", **kw)
                        vs.append(vr.metrics.get("is", {}).get("sharpe", 0.0))
                    except EvalError:
                        pass
                ex["stability"] = stability_score(base, vs)
            out["extras"] = ex
        return out
    except EvalError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001 - report any evaluation failure to the job
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
