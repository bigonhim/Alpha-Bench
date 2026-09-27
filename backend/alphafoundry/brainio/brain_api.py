"""Client for the WorldQuant BRAIN REST API (https://api.worldquantbrain.com).

There is no API key: BRAIN accepts the website email and password over HTTP Basic auth and returns a
session cookie. Accounts with biometric sign-in answer ``401`` with ``WWW-Authenticate: persona``; the
user opens the returned link, finishes the check in a browser, and the client then POSTs the same link.

Everything asynchronous on BRAIN (simulations, record sets, checks, correlations) is polled while the
server sends a ``Retry-After`` header. ``429`` responses are retried after ``Retry-After``; a ``401`` in
the middle of a session triggers one re-authentication with the stored credentials.

Alpha submission is deliberately not implemented: submitting stays a manual action on the website.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

import httpx
import numpy as np

from .export import simulation_payload

log = logging.getLogger("alphafoundry.brain")

BASE_URL = "https://api.worldquantbrain.com"
PLATFORM_URL = "https://platform.worldquantbrain.com"
V2 = {"Accept": "application/json;version=2.0"}
FINAL = {"COMPLETE", "WARNING", "ERROR", "FAIL", "FAILED", "CANCELLED", "TIMEOUT"}
OK = {"COMPLETE", "WARNING"}


class BrainError(Exception):
    def __init__(self, message: str, status: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class BrainAuthError(BrainError):
    pass


class PersonaRequired(BrainAuthError):
    """Biometric sign-in: open ``url`` in a browser, finish the check, then call ``complete_persona(url)``."""

    def __init__(self, url: str):
        super().__init__("Biometric check required: open the link, complete it in the browser, then click "
                         "Complete sign-in.", 401)
        self.url = url


class DailyLimitReached(BrainError):
    pass


def _retry_after(r: httpx.Response, fallback: float) -> float:
    try:
        return max(0.5, min(float(r.headers.get("Retry-After")), 120.0))
    except (TypeError, ValueError):
        return fallback


def _json(r: httpx.Response) -> Any:
    if not r.content:
        return {}
    try:
        return r.json()
    except ValueError:
        return r.text


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


class BrainClient:
    def __init__(self, session_file: Path | None = None, base_url: str = BASE_URL, timeout: float = 60.0,
                 transport: httpx.BaseTransport | None = None,
                 credentials: Callable[[], tuple[str, str] | None] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self.session_file = session_file
        self.credentials = credentials
        self.sleep = sleep
        self.last_ratelimit: dict[str, str] = {}
        self.info: dict | None = None
        self._auth_lock = threading.RLock()
        self.http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False,
                                 headers={"User-Agent": "AlphaFoundry/1.1", "Accept": "application/json"})
        self._load_cookies()

    # ------------------------------------------------------------------ cookies
    def _load_cookies(self) -> None:
        if not self.session_file or not self.session_file.exists():
            return
        try:
            for c in json.loads(self.session_file.read_text(encoding="utf-8")):
                self.http.cookies.set(c["name"], c["value"], domain=c.get("domain") or "", path=c.get("path") or "/")
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _save_cookies(self) -> None:
        if not self.session_file:
            return
        items = [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path} for c in self.http.cookies.jar]
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            self.session_file.write_text(json.dumps(items), encoding="utf-8")
        except OSError:
            pass

    def clear_session(self) -> None:
        self.http.cookies.clear()
        self.info = None
        if self.session_file and self.session_file.exists():
            try:
                self.session_file.unlink()
            except OSError:
                pass

    def url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path.replace("http://api.worldquantbrain.com", "https://api.worldquantbrain.com")
        return f"{self.base_url}/{path.lstrip('/')}"

    # ------------------------------------------------------------------ authentication
    def auth_info(self) -> dict | None:
        """Current session (``{user:{id}, token:{expiry}, permissions}``) or None when not signed in."""
        try:
            r = self.http.get(self.url("/authentication"))
        except httpx.HTTPError:
            return None
        if r.status_code == 200:
            data = _json(r)
            if isinstance(data, dict) and data.get("user"):
                self.info = data
                return data
        return None

    def login(self, email: str, password: str) -> dict:
        with self._auth_lock:
            for attempt in range(4):
                try:
                    r = self.http.post(self.url("/authentication"), auth=(email, password))
                except httpx.HTTPError as e:
                    if attempt == 3:
                        raise BrainError(f"Could not reach BRAIN: {e}") from None
                    self.sleep(2.0 + 2 * attempt)
                    continue
                if r.status_code in (200, 201):
                    data = _json(r)
                    self.info = data if isinstance(data, dict) else {}
                    self._save_cookies()
                    return self.info
                if r.status_code == 401 and r.headers.get("WWW-Authenticate", "").lower() == "persona":
                    raise PersonaRequired(urljoin(str(r.url), r.headers.get("Location", "")))
                if r.status_code == 401:
                    raise BrainAuthError("BRAIN rejected the email or password.", 401)
                if r.status_code == 429 or r.status_code >= 500:
                    self.sleep(_retry_after(r, 3.0 + 3 * attempt))
                    continue
                raise BrainAuthError(f"Sign-in failed (HTTP {r.status_code}).", r.status_code)
            raise BrainAuthError("Sign-in failed after several attempts.")

    def complete_persona(self, url: str) -> dict:
        r = self.http.post(self.url(url))
        if r.status_code not in (200, 201):
            raise BrainAuthError(f"The biometric check is not complete yet (HTTP {r.status_code}). Finish it in the "
                                 "browser, then try again.", r.status_code)
        self._save_cookies()
        return self.auth_info() or {}

    def logout(self) -> None:
        try:
            self.http.delete(self.url("/authentication"))
        except httpx.HTTPError:
            pass
        self.clear_session()

    def reauthenticate(self) -> bool:
        creds = self.credentials() if self.credentials else None
        if not creds:
            return False
        try:
            self.login(*creds)
            return True
        except PersonaRequired:
            raise
        except BrainError:
            return False

    # ------------------------------------------------------------------ requests
    def request(self, method: str, path: str, *, params: Any = None, json_body: Any = None,
                headers: dict | None = None, max_retries: int = 6) -> httpx.Response:
        reauthed = False
        last: httpx.Response | None = None
        for attempt in range(max_retries):
            try:
                r = self.http.request(method, self.url(path), params=params, json=json_body, headers=headers)
            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    raise BrainError(f"{method} {path}: network error {e}") from None
                self.sleep(min(2.0 + 2 * attempt, 20.0))
                continue
            last = r
            rl = {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
            if rl:
                self.last_ratelimit = rl
            if r.status_code == 401 and not reauthed:
                reauthed = True
                if self.reauthenticate():
                    continue
                raise BrainAuthError("The BRAIN session has expired. Sign in again on the BRAIN page.", 401)
            if r.status_code == 429:
                if method.upper() == "POST" and rl.get("x-ratelimit-remaining") == "0":
                    raise DailyLimitReached("BRAIN daily simulation limit reached.", 429, _json(r))
                self.sleep(_retry_after(r, 3.0 + 2 * attempt))
                continue
            if r.status_code >= 500 and attempt < max_retries - 1:
                self.sleep(min(2.0 + 3 * attempt, 20.0))
                continue
            return r
        assert last is not None
        return last

    def get(self, path: str, params: Any = None, headers: dict | None = None) -> Any:
        r = self.request("GET", path, params=params, headers=headers)
        if r.status_code >= 400:
            raise BrainError(f"GET {path} -> HTTP {r.status_code}: {r.text[:300]}", r.status_code, _json(r))
        return _json(r)

    def poll(self, path: str, *, timeout: float = 1800, headers: dict | None = None,
             on_progress: Callable[[Any], None] | None = None, cancelled: Callable[[], bool] | None = None
             ) -> httpx.Response:
        """GET repeatedly while BRAIN answers with a Retry-After header (the resource is still being built)."""
        deadline = time.monotonic() + timeout
        while True:
            r = self.request("GET", path, headers=headers)
            ra = _f(r.headers.get("Retry-After"))
            if r.status_code < 400 and ra is not None and ra > 0:
                if on_progress:
                    on_progress(_json(r))
                if time.monotonic() > deadline:
                    raise BrainError(f"Timed out waiting for {path}", 408)
                if cancelled and cancelled():
                    raise BrainError("cancelled", 499)
                self.sleep(max(1.0, min(ra, 30.0)))
                continue
            return r

    def paginate(self, path: str, params: dict | None = None, *, limit: int | None = None,
                 page: int = 50) -> list[dict]:
        out: list[dict] = []
        offset = 0
        base = {k: v for k, v in (params or {}).items() if v is not None}
        while True:
            size = page if limit is None else max(1, min(page, limit - len(out)))
            data = self.get(path, {**base, "limit": size, "offset": offset})
            items = data.get("results", []) if isinstance(data, dict) else (data or [])
            out.extend(items)
            offset += len(items)
            total = data.get("count") if isinstance(data, dict) else None
            if not items or (limit is not None and len(out) >= limit) or (total is not None and offset >= total):
                break
        return out if limit is None else out[:limit]

    # ------------------------------------------------------------------ simulation
    @property
    def permissions(self) -> list[str]:
        return list((self.info or {}).get("permissions") or [])

    def start_simulation(self, payload: dict | list[dict], max_wait: float = 900.0) -> str:
        """POST /simulations (one payload, or 2-10 for a multi-simulation). Returns the progress URL."""
        waited = 0.0
        while True:
            r = self.request("POST", "/simulations", json_body=payload, max_retries=2)
            if r.status_code in (200, 201, 202) and r.headers.get("Location"):
                return r.headers["Location"]
            if r.status_code == 429:  # concurrent-simulation limit: wait for a slot
                wait = _retry_after(r, 10.0)
                if waited > max_wait:
                    raise BrainError("BRAIN concurrent-simulation limit: no free slot.", 429, _json(r))
                self.sleep(wait)
                waited += wait
                continue
            detail = _json(r)
            msg = detail.get("message") if isinstance(detail, dict) else None
            if isinstance(detail, dict) and not msg:
                msg = "; ".join(f"{k}: {v}" for k, v in detail.items())[:300]
            raise BrainError(f"BRAIN rejected the simulation: {msg or r.text[:300]}", r.status_code, detail)

    def wait_simulation(self, location: str, timeout: float = 1800,
                        cancelled: Callable[[], bool] | None = None) -> dict:
        r = self.poll(location, timeout=timeout, cancelled=cancelled)
        data = _json(r)
        if r.status_code >= 400 or not isinstance(data, dict):
            return {"status": "ERROR", "message": f"HTTP {r.status_code}: {str(data)[:200]}"}
        return data

    def _single(self, item: dict, timeout: float, cancelled: Callable[[], bool] | None) -> dict:
        payload = simulation_payload(item["expr"], item.get("settings") or {})
        try:
            loc = self.start_simulation(payload)
            sim = self.wait_simulation(loc, timeout, cancelled)
        except DailyLimitReached:
            raise
        except BrainError as e:
            return {**item, "ok": False, "status": "SUBMIT_ERROR", "alpha_id": None, "message": str(e)}
        return self._result(item, sim)

    @staticmethod
    def _result(item: dict, sim: dict) -> dict:
        status = str(sim.get("status") or "").upper() or "UNKNOWN"
        return {**item, "ok": status in OK and bool(sim.get("alpha")), "status": status, "alpha_id": sim.get("alpha"),
                "message": sim.get("message")}

    def _multi(self, items: list[dict], timeout: float, cancelled: Callable[[], bool] | None) -> list[dict]:
        payloads = [simulation_payload(it["expr"], it.get("settings") or {}) for it in items]
        try:
            loc = self.start_simulation(payloads)
            parent = self.wait_simulation(loc, timeout, cancelled)
        except DailyLimitReached:
            raise
        except BrainError as e:
            parent = {"status": "ERROR", "message": str(e)}
        children = parent.get("children") or []
        if not children:  # one bad member can sink a multi-simulation: run them one by one
            return [self._single(it, timeout, cancelled) for it in items]
        out = []
        for it, child in zip(items, children):
            sim = self.wait_simulation(f"/simulations/{child}", timeout, cancelled)
            res = self._result(it, sim)
            if not res["ok"] and not res["message"] and str(parent.get("status")).upper() in ("ERROR", "FAIL"):
                res = self._single(it, timeout, cancelled)
            out.append(res)
        for it in items[len(children):]:
            out.append({**it, "ok": False, "status": "ERROR", "alpha_id": None, "message": "missing child"})
        return out

    def simulate(self, items: list[dict], *, concurrency: int = 3, multi: bool | None = None,
                 on_result: Callable[[int, dict], None] | None = None, cancelled: Callable[[], bool] | None = None,
                 timeout: float = 1800) -> list[dict]:
        """Simulate ``[{"expr", "settings", ...}]`` on BRAIN. Results keep the input order.

        Multi-simulation (up to 10 alphas per request) is used when the account has the permission and
        the items share region, universe and delay."""
        if multi is None:
            multi = "MULTI_SIMULATION" in self.permissions
        results: list[dict | None] = [None] * len(items)
        groups: list[list[int]] = []
        if multi:
            by: dict[tuple, list[int]] = {}
            for i, it in enumerate(items):
                s = it.get("settings") or {}
                by.setdefault((s.get("region", "USA"), s.get("universe", "TOP3000"), int(s.get("delay", 1))),
                              []).append(i)
            for idx in by.values():
                groups += [idx[k:k + 10] for k in range(0, len(idx), 10)]
        else:
            groups = [[i] for i in range(len(items))]
        stop = threading.Event()
        lock = threading.Lock()

        def finish(i: int, res: dict) -> None:
            with lock:
                results[i] = res
            if on_result:
                on_result(i, res)

        def run(group: list[int]) -> None:
            if stop.is_set() or (cancelled and cancelled()):
                return
            try:
                if len(group) == 1:
                    finish(group[0], self._single(items[group[0]], timeout, cancelled))
                else:
                    for i, res in zip(group, self._multi([items[i] for i in group], timeout, cancelled)):
                        finish(i, res)
            except DailyLimitReached as e:
                stop.set()
                for i in group:
                    finish(i, {**items[i], "ok": False, "status": "LIMIT", "alpha_id": None, "message": str(e)})

        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
            list(pool.map(run, groups))
        return [r if r is not None else {**it, "ok": False, "status": "SKIPPED", "alpha_id": None,
                                         "message": "not simulated (cancelled or daily limit)"}
                for r, it in zip(results, items)]

    # ------------------------------------------------------------------ alphas
    def get_alpha(self, alpha_id: str) -> dict:
        r = self.poll(f"/alphas/{alpha_id}", timeout=300)
        if r.status_code >= 400:
            raise BrainError(f"GET /alphas/{alpha_id} -> HTTP {r.status_code}", r.status_code)
        return _json(r)

    def _recordset(self, alpha_id: str, name: str) -> dict:
        r = self.poll(f"/alphas/{alpha_id}/recordsets/{name}", timeout=600)
        if r.status_code >= 400:
            raise BrainError(f"record set {name} of {alpha_id} -> HTTP {r.status_code}", r.status_code)
        data = _json(r)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def recordset_rows(data: dict) -> list[dict]:
        cols = [p.get("name") for p in ((data.get("schema") or {}).get("properties") or [])]
        return [dict(zip(cols, rec)) for rec in data.get("records") or []]

    def get_pnl(self, alpha_id: str) -> tuple[list[str], np.ndarray]:
        """(dates, DAILY PnL) from the cumulative ``pnl`` record set."""
        rows = [r for r in self.recordset_rows(self._recordset(alpha_id, "pnl")) if r.get("pnl") is not None]
        dates = [str(r.get("date"))[:10] for r in rows]
        cum = np.array([float(r["pnl"]) for r in rows], dtype=np.float64)
        return dates, np.diff(np.r_[0.0, cum]) if len(cum) else cum

    def yearly_stats(self, alpha_id: str) -> list[dict]:
        return self.recordset_rows(self._recordset(alpha_id, "yearly-stats"))

    def check(self, alpha_id: str, timeout: float = 900) -> dict:
        """Pre-submission test suite (self/prod correlation, sub-universe, ...). Never submits."""
        r = self.poll(f"/alphas/{alpha_id}/check", timeout=timeout, headers=V2)
        data = _json(r)
        if r.status_code >= 400 or not isinstance(data, dict):
            return {"checks": [], "can_submit": False, "self_corr": None, "prod_corr": None, "failed": [],
                    "pending": [], "error": str(data)[:300]}
        checks = (data.get("is") or {}).get("checks") or data.get("checks") or []
        res = {str(c.get("result", "")).upper() for c in checks}

        def val(name: str) -> float | None:
            c = next((c for c in checks if c.get("name") == name), None)
            return _f(c.get("value")) if c else None
        return {"checks": checks, "can_submit": bool(checks) and not ({"FAIL", "ERROR", "PENDING"} & res),
                "self_corr": val("SELF_CORRELATION"), "prod_corr": val("PROD_CORRELATION"),
                "failed": [c.get("name") for c in checks if str(c.get("result")).upper() in ("FAIL", "ERROR")],
                "pending": [c.get("name") for c in checks if str(c.get("result")).upper() == "PENDING"]}

    def correlations(self, alpha_id: str, kind: str = "self") -> dict:
        r = self.poll(f"/alphas/{alpha_id}/correlations/{kind}", timeout=600, headers=V2)
        data = _json(r)
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ catalog / user
    @staticmethod
    def _scope(region: str, delay: int, universe: str, instrument_type: str) -> dict:
        return {"instrumentType": instrument_type, "region": region, "delay": int(delay), "universe": universe}

    def datasets(self, region: str = "USA", delay: int = 1, universe: str = "TOP3000", *,
                 instrument_type: str = "EQUITY", category: str | None = None, search: str | None = None,
                 limit: int | None = None) -> list[dict]:
        p = {**self._scope(region, delay, universe, instrument_type), "category": category, "search": search}
        return self.paginate("/data-sets", p, limit=limit)

    def datafields(self, region: str = "USA", delay: int = 1, universe: str = "TOP3000", *,
                   dataset_id: str | None = None, search: str | None = None, instrument_type: str = "EQUITY",
                   limit: int | None = None) -> list[dict]:
        p = {**self._scope(region, delay, universe, instrument_type), "dataset.id": dataset_id, "search": search}
        return self.paginate("/data-fields", p, limit=limit)

    def operators(self) -> list[dict]:
        data = self.get("/operators")
        return data if isinstance(data, list) else (data or {}).get("results", [])

    def user_alphas(self, params: dict | None = None, limit: int | None = 100) -> list[dict]:
        return self.paginate("/users/self/alphas", {"order": "-dateCreated", **(params or {})}, limit=limit, page=100)


def summarize_alpha(a: dict) -> dict:
    """Compact, DB-ready view of an alpha JSON (margin converted to basis points)."""
    code = a.get("regular")
    expr = code.get("code") if isinstance(code, dict) else code

    def metrics(block: Any) -> dict | None:
        if not isinstance(block, dict) or not block:
            return None
        margin = _f(block.get("margin"))
        return {"sharpe": _f(block.get("sharpe")), "fitness": _f(block.get("fitness")),
                "turnover": _f(block.get("turnover")), "returns": _f(block.get("returns")),
                "drawdown": _f(block.get("drawdown")), "margin": None if margin is None else round(margin * 1e4, 3),
                "long_count": _f(block.get("longCount")), "short_count": _f(block.get("shortCount"))}
    is_block = a.get("is") or {}
    checks = [{"name": c.get("name"), "result": str(c.get("result", "")).upper(), "value": c.get("value"),
               "limit": c.get("limit")} for c in (is_block.get("checks") or []) if isinstance(c, dict)]
    failed = [c["name"] for c in checks if c["result"] in ("FAIL", "ERROR")]
    return {"brain_id": a.get("id"), "expr": (expr or "").strip(), "settings": a.get("settings") or {},
            "status": a.get("status"), "stage": a.get("stage"), "grade": a.get("grade"),
            "date_created": a.get("dateCreated"), "is": metrics(is_block), "os": metrics(a.get("os")),
            "checks": checks, "failed": failed, "passed": (not failed) if checks else None,
            "url": f"{PLATFORM_URL}/alpha/{a.get('id')}" if a.get("id") else None}


__all__ = ["BASE_URL", "BrainAuthError", "BrainClient", "BrainError", "DailyLimitReached", "PersonaRequired",
           "summarize_alpha"]
