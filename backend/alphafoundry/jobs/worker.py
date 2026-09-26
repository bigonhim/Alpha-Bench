"""Process-pool worker. Each worker memory-maps the same panel files (shared via the OS page cache)
and keeps its own subexpression cache. numba threads per worker = logical cores / workers (set by the
parent through AF_WORKER_THREADS before the pool starts, and applied before numba is imported), so the
pool uses every core without oversubscribing it."""

from __future__ import annotations

import os

os.environ["NUMBA_NUM_THREADS"] = os.environ.get("AF_WORKER_THREADS", "1")

_state: dict = {}


def init_worker(panel_path: str, cfg: dict, cache_mb: float) -> None:
    from ..engine.evaluator import SubexprCache
    from ..engine.panel import Panel
    from ..sim.simulator import compute_periods

    panel = Panel(panel_path)
    _state["panel"] = panel
    _state["periods"] = compute_periods(panel, cfg)
    _state["cache"] = SubexprCache(cache_mb)
    _state["cfg"] = cfg
    _state["local_fields"] = set(panel.field_names()) | set(panel.group_labels) | {"market", "country"}


def warmup() -> bool:
    """Touch the most common kernels so the first real task is fast."""
    run_batch([{"expr": "rank(-ts_delta(close, 5))", "settings": {}}], "screen")
    return True


def run_batch(payloads: list[dict], span: str) -> list[dict]:
    from .evaluate import evaluate_one

    st = _state
    return [evaluate_one(st["panel"], st["periods"], st["cache"], p, span, st["cfg"], st["local_fields"])
            for p in payloads]
