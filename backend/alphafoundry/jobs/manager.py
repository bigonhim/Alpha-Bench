"""Background job manager, process pool and WebSocket event broadcaster."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

log = logging.getLogger("alphafoundry.jobs")


class Broadcaster:
    """Fan-out of JSON events to connected WebSocket clients; safe to call from any thread."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queues: set[asyncio.Queue] = set()
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        with self._lock:
            self.queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self.queues.discard(q)

    def publish(self, event: dict) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            qs = list(self.queues)
        for q in qs:
            def put(q=q, e=event):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(e)
            try:
                loop.call_soon_threadsafe(put)
            except RuntimeError:
                pass


class JobCancelled(Exception):
    pass


# jobs that generate or tune alphas from the local panel (blocked on the synthetic demo data by default)
MINING_KINDS = {"automine", "gp", "templates", "grammar", "alpha101", "forge", "reengineer", "compose",
                "settings_opt"}


class JobHandle:
    def __init__(self, manager: "JobManager", job_id: int, kind: str, config: dict):
        self.m = manager
        self.id = job_id
        self.kind = kind
        self.config = config
        self.stop_event = threading.Event()
        self.run_event = threading.Event()
        self.run_event.set()
        self.progress: dict[str, Any] = {"done": 0, "total": 0, "phase": "starting"}
        self.stats: dict[str, Any] = {"evaluated": 0, "saved": 0, "passed": 0, "errors": 0, "brain_only": 0}
        self.status = "running"
        self.started = time.time()
        self._last_emit = 0.0
        self._pending_results: list[dict] = []
        self._lock = threading.Lock()

    # ---- control
    def check(self) -> None:
        """Honor pause/cancel; call frequently from job code."""
        if self.stop_event.is_set():
            raise JobCancelled()
        if not self.run_event.is_set():
            self.set_status("paused")
            while not self.run_event.wait(0.25):
                if self.stop_event.is_set():
                    raise JobCancelled()
            self.set_status("running")

    def cancelled(self) -> bool:
        return self.stop_event.is_set()

    def set_status(self, status: str) -> None:
        self.status = status
        self.m.ws.store.update_job(self.id, status=status)
        self.emit(force=True)

    # ---- progress / events
    def update(self, **kw) -> None:
        self.progress.update(kw)
        self.emit()

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + n

    def add_result(self, row: dict) -> None:
        with self._lock:
            self._pending_results.append(row)
        self.emit()

    def event(self, etype: str, data: dict) -> None:
        self.m.bc.publish({"type": etype, "job_id": self.id, **data})

    def snapshot(self) -> dict:
        el = time.time() - self.started
        return {"id": self.id, "kind": self.kind, "status": self.status, "progress": self.progress,
                "stats": {**self.stats, "elapsed_s": round(el, 1),
                          "rate_per_s": round(self.stats.get("evaluated", 0) / el, 2) if el > 0 else 0.0},
                "config": self.config}

    def emit(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_emit < 0.25:
            return
        self._last_emit = now
        with self._lock:
            results, self._pending_results = self._pending_results, []
        snap = self.snapshot()
        self.m.bc.publish({"type": "job", "job": snap, "results": results})
        self.m.ws.store.update_job(self.id, progress=snap["progress"], stats=snap["stats"])


class JobManager:
    def __init__(self, ws, bc: Broadcaster):
        self.ws = ws
        self.bc = bc
        self.jobs: dict[int, JobHandle] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._pool_key: tuple | None = None
        self._pool_lock = threading.Lock()
        self._pool_failed = False

    # ------------------------------------------------------------------ pool
    def pool(self) -> ProcessPoolExecutor | None:
        n = int(self.ws.settings.get("workers", 2) or 0)
        if n <= 0 or self._pool_failed:
            return None
        key = (str(self.ws.panel.root), self.ws.panel.version, n)
        with self._pool_lock:
            if self._pool is not None and self._pool_key == key:
                return self._pool
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
            try:
                import os

                from .worker import init_worker, warmup
                os.environ["AF_WORKER_THREADS"] = str(max(1, (os.cpu_count() or 2) // n))
                cfg = dict(self.ws.settings)
                self._pool = ProcessPoolExecutor(max_workers=n, initializer=init_worker,
                                                 initargs=(str(self.ws.panel.root), cfg,
                                                           float(cfg.get("worker_cache_mb", 300))))
                self._pool_key = key
                for f in [self._pool.submit(warmup) for _ in range(n)]:
                    f.result(timeout=600)
            except Exception:  # noqa: BLE001
                log.exception("process pool unavailable; falling back to in-process evaluation")
                self._pool_failed = True
                self._pool = None
            return self._pool

    def shutdown(self) -> None:
        for h in list(self.jobs.values()):
            h.stop_event.set()
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None

    def reset_pool(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            self._pool_key = None
            self._pool_failed = False

    def evaluate(self, handle: JobHandle | None, payloads: list[dict], span: str,
                 on_result: Callable[[int, dict], None] | None = None, chunk: int = 3) -> list[dict]:
        """Evaluate payloads (in order) on the pool, or in-process when no pool is available."""
        results: list[dict | None] = [None] * len(payloads)
        if not payloads:
            return []
        pool = self.pool()
        if pool is None:
            from .evaluate import evaluate_one
            ws = self.ws
            lf = ws.local_fields()
            for i, p in enumerate(payloads):
                if handle:
                    handle.check()
                with ws.lock:
                    r = evaluate_one(ws.panel, ws.periods, ws.cache, p, span, ws.settings, lf)
                results[i] = r
                if on_result:
                    on_result(i, r)
            return results  # type: ignore[return-value]
        from .worker import run_batch
        futs = {}
        for start in range(0, len(payloads), chunk):
            futs[pool.submit(run_batch, payloads[start:start + chunk], span)] = start
        try:
            for f in as_completed(futs):
                if handle and handle.cancelled():
                    raise JobCancelled()
                start = futs[f]
                try:
                    batch = f.result()
                except Exception as e:  # noqa: BLE001 - worker crash
                    batch = [{"ok": False, "error": f"worker error: {e}"}] * min(chunk, len(payloads) - start)
                for k, r in enumerate(batch):
                    results[start + k] = r
                    if on_result:
                        on_result(start + k, r)
                if handle:
                    handle.check()
        except JobCancelled:
            for f in futs:
                f.cancel()
            raise
        return [r if r is not None else {"ok": False, "error": "not evaluated"} for r in results]

    # ------------------------------------------------------------------ jobs
    def demo_guard(self, kind: str, config: dict) -> None:
        """Refuse to mine on the synthetic demo panel: its planted effects do not exist on BRAIN."""
        import os

        if kind not in MINING_KINDS or getattr(self.ws.panel, "source", "real") != "demo":
            return
        if config.get("allow_demo") or self.ws.settings.get("allow_demo_mining") \
                or os.environ.get("ALPHAFOUNDRY_ALLOW_DEMO_MINING") == "1":
            return
        raise ValueError("The active dataset is the synthetic DEMO panel. Alphas mined on it come from effects "
                         "planted in the demo generator and will not pass on BRAIN. Build real data on the Data "
                         "page first (or tick 'run on demo data anyway' to try the tool).")

    def start(self, kind: str, config: dict) -> int:
        from . import miners

        fn = miners.JOB_KINDS.get(kind)
        if fn is None:
            raise ValueError(f"unknown job kind '{kind}'")
        self.demo_guard(kind, config)
        job_id = self.ws.store.create_job(kind, config)
        h = JobHandle(self, job_id, kind, config)
        self.jobs[job_id] = h

        def runner():
            self.ws.store.update_job(job_id, status="running", started_at=dt.datetime.now().isoformat(timespec="seconds"))
            h.emit(force=True)
            status, error = "done", None
            try:
                fn(h, self.ws, self, config)
            except JobCancelled:
                status = "cancelled"
            except Exception as e:  # noqa: BLE001
                status, error = "error", f"{type(e).__name__}: {e}"
                log.error("job %s failed: %s", job_id, traceback.format_exc())
            h.status = status
            h.progress["phase"] = status
            self.ws.store.update_job(job_id, status=status, error=error,
                                     finished_at=dt.datetime.now().isoformat(timespec="seconds"),
                                     progress=h.progress, stats=h.snapshot()["stats"])
            h.emit(force=True)
            if error:
                self.bc.publish({"type": "toast", "level": "error", "message": f"Job #{job_id} failed: {error}"})
            else:
                self.bc.publish({"type": "toast", "level": "success" if status == "done" else "info",
                                 "message": f"Job #{job_id} ({kind}) {status}: {h.stats.get('saved', 0)} alphas saved"})
            self.jobs.pop(job_id, None)

        threading.Thread(target=runner, name=f"job-{job_id}", daemon=True).start()
        return job_id

    def control(self, job_id: int, action: str) -> bool:
        h = self.jobs.get(job_id)
        if h is None:
            return False
        if action == "cancel":
            h.stop_event.set()
            h.run_event.set()
        elif action == "pause":
            h.run_event.clear()
        elif action == "resume":
            h.run_event.set()
        else:
            return False
        return True

    def running(self) -> list[dict]:
        return [h.snapshot() for h in self.jobs.values()]
