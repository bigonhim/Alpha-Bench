"""Responsiveness benchmark on the active dataset.

Usage:  uv run python scripts/bench.py [--synthetic N_STOCKS N_DAYS]

With --synthetic, a deterministic panel of the given size is built in an isolated runtime folder
(set ALPHAFOUNDRY_RUNTIME yourself to keep it), so the targets can be checked at production scale
without downloading data.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time


def timed(fn, reps=3):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", nargs=2, type=int, metavar=("N_STOCKS", "N_DAYS"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.synthetic:
        os.environ.setdefault("ALPHAFOUNDRY_RUNTIME", tempfile.mkdtemp(prefix="af_bench_"))
    from alphafoundry import config

    if args.synthetic:
        from alphafoundry.data.demo import build_demo

        t0 = time.perf_counter()
        config.ensure_dirs()
        build_demo(config.DEMO_PANELS_DIR, n_stocks=args.synthetic[0], n_days=args.synthetic[1])
        config.save_settings({"active_dataset": "demo", "workers": args.workers})
        print(f"built synthetic panel {args.synthetic[0]}x{args.synthetic[1]} in {time.perf_counter() - t0:.1f}s")
    from alphafoundry.workspace import get_workspace

    ws = get_workspace()
    p = ws.panel
    print(f"panel: {p.source} {p.N} stocks x {p.T} days; IS {ws.periods.to_dict(p)}")
    exprs = ["rank(-ts_delta(close, 5))", "group_rank(ts_backfill(operating_income, 120) / cap, subindustry)",
             "ts_corr(rank(close), rank(volume), 10)", "group_neutralize(ts_rank(ts_mean(returns, 20), 60), sector)",
             "trade_when(volume > adv20, rank(-returns), -1)"]
    parse_ms = timed(lambda: ws.analyze(exprs[1]), reps=20)
    print(f"parse/typecheck: median {statistics.median(parse_ms):.1f} ms")
    for e in exprs:
        ws.cache.clear()
        cold = timed(lambda: ws.simulate(e, {"neutralization": "SUBINDUSTRY"}), reps=1)[0]
        warm = statistics.median(timed(lambda: ws.simulate(e, {"neutralization": "SUBINDUSTRY"}), reps=3))
        chg = statistics.median(timed(lambda: ws.simulate(e, {"neutralization": "SECTOR", "decay": 6}), reps=3))
        print(f"simulate cold {cold:7.0f} ms | warm {warm:6.0f} ms | settings change {chg:6.0f} ms | {e}")
    t0 = time.perf_counter()
    n = sum(1 for ev in ws.sweep(exprs[0], None) if ev["type"] == "cell")
    print(f"settings sweep: {n} cells in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    ws.simulate(exprs[0], None, extras=True)
    print(f"simulate + extras (sub-universe, stability): {(time.perf_counter() - t0) * 1000:.0f} ms")
    if args.workers > 0:
        import random

        from alphafoundry.gen.templates import load_templates, template_candidates
        from alphafoundry.jobs.manager import Broadcaster, JobManager

        mgr = JobManager(ws, Broadcaster())
        t0 = time.perf_counter()
        mgr.pool()
        print(f"worker pool ready in {time.perf_counter() - t0:.1f}s")
        cands = template_candidates(load_templates(), {}, ws.local_fields(), per_template=2, settings_per_expr=1,
                                    rng=random.Random(1))[:80]
        payloads = [{"expr": c.expr, "settings": c.settings} for c in cands]
        for span, label in (("screen", "3-year pre-screen"), ("is", "full in-sample")):
            t0 = time.perf_counter()
            res = mgr.evaluate(None, payloads, span)
            dt = time.perf_counter() - t0
            print(f"mining throughput ({label}, {args.workers} workers): {len(res) / dt:.1f} alphas/s "
                  f"({sum(r.get('ok', False) for r in res)}/{len(res)} ok)")
        mgr.shutdown()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
