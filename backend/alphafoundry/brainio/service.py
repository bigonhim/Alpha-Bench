"""BRAIN connection bound to a Workspace: sign-in state, usage budget, and recording BRAIN results locally.

BRAIN's own simulation is the ground truth. Results are stored next to the local ones (``brain_results``
and ``brain_alphas``), they recalibrate the local pass model, and they reward the miners' bandit arms
for the idea families that actually pass on BRAIN.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from typing import Any

from ..fastexpr import analyze
from ..store.db import now, pack_pnl, unpack_pnl
from . import credentials as creds
from .brain_api import BrainAuthError, BrainClient, BrainError, PersonaRequired, summarize_alpha

log = logging.getLogger("alphafoundry.brain")

DEFAULT_BRAIN_SETTINGS: dict[str, Any] = {"concurrency": 3, "multi": "auto", "daily_budget": 500,
                                          "check_passing": True, "region": "USA", "universe": "TOP3000", "delay": 1}


class BrainService:
    def __init__(self, ws, transport=None, sleep=None):
        self.ws = ws
        kw: dict[str, Any] = {"transport": transport, "credentials": creds.load_credentials}
        if sleep is not None:
            kw["sleep"] = sleep
        self.client = BrainClient(creds.session_dir() / "brain_session.json", **kw)
        self.persona_url: str | None = None
        self._pending: tuple[str, str, bool] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ settings / usage
    @property
    def settings(self) -> dict:
        return {**DEFAULT_BRAIN_SETTINGS, **(self.ws.store.kv_get("brain_settings") or {})}

    def update_settings(self, updates: dict) -> dict:
        s = self.settings
        for k, v in updates.items():
            if k in DEFAULT_BRAIN_SETTINGS:
                s[k] = v
        s["concurrency"] = max(1, min(10, int(s["concurrency"])))
        s["daily_budget"] = max(0, int(s["daily_budget"]))
        self.ws.store.kv_set("brain_settings", s)
        return s

    def usage_today(self) -> int:
        return int((self.ws.store.kv_get("brain_usage") or {}).get(dt.date.today().isoformat(), 0))

    def budget_left(self) -> int:
        return max(0, int(self.settings["daily_budget"]) - self.usage_today())

    def count_usage(self, n: int) -> None:
        with self._lock:
            u = self.ws.store.kv_get("brain_usage") or {}
            day = dt.date.today().isoformat()
            u = {k: v for k, v in u.items() if k >= (dt.date.today() - dt.timedelta(days=30)).isoformat()}
            u[day] = int(u.get(day, 0)) + int(n)
            self.ws.store.kv_set("brain_usage", u)

    # ------------------------------------------------------------------ connection
    def connect(self, email: str, password: str, remember: bool = False) -> dict:
        email = (email or "").strip()
        if not email or not password:
            if creds.load_credentials():
                email, password = creds.load_credentials()  # type: ignore[misc]
            else:
                raise BrainAuthError("Enter your BRAIN email and password.")
        try:
            self.client.login(email, password)
        except PersonaRequired as e:
            self.persona_url = e.url
            self._pending = (email, password, remember) if remember else None
            return {**self.status(), "persona_url": e.url}
        self.persona_url = None
        if remember:
            creds.save_credentials(email, password)
        return self.status()

    def complete_persona(self) -> dict:
        if not self.persona_url:
            raise BrainAuthError("No biometric check is pending. Sign in first.")
        self.client.complete_persona(self.persona_url)
        self.persona_url = None
        if self._pending:
            creds.save_credentials(self._pending[0], self._pending[1])
            self._pending = None
        return self.status()

    def disconnect(self, forget: bool = False) -> dict:
        self.client.logout()
        if forget:
            creds.forget_credentials()
        return self.status()

    def ensure(self) -> dict:
        """A live session, re-signing in with stored credentials when the token expired."""
        info = self.client.auth_info()
        if info:
            return info
        if self.client.reauthenticate():
            info = self.client.auth_info() or self.client.info
            if info:
                return info
        raise BrainAuthError("Not signed in to BRAIN. Open the BRAIN page and sign in.")

    def status(self) -> dict:
        # no session cookie and nothing to sign in with: stay off the network entirely
        has_session = len(self.client.http.cookies.jar) > 0 or self.client.info is not None
        info = self.client.auth_info() if has_session else None
        token = (info or {}).get("token") or {}
        perms = list((info or {}).get("permissions") or [])
        return {
            "connected": bool(info), "user_id": ((info or {}).get("user") or {}).get("id"),
            "expiry_s": token.get("expiry"), "permissions": perms, "multi_allowed": "MULTI_SIMULATION" in perms,
            "usage_today": self.usage_today(), "budget": int(self.settings["daily_budget"]),
            "budget_left": self.budget_left(), "settings": self.settings,
            "keyring_available": creds.keyring_available(), "credentials_saved": creds.credentials_saved(),
            "saved_email": creds.saved_email() or (creds.env_credentials() or (None,))[0],
            "persona_url": self.persona_url, "ratelimit": self.client.last_ratelimit,
        }

    # ------------------------------------------------------------------ recording
    def record_result(self, alpha_id: int, summary: dict, pnl: tuple[list[str], Any] | None = None,
                      check: dict | None = None) -> dict:
        """Store a BRAIN result against a local alpha; update calibration inputs and the bandit."""
        store = self.ws.store
        m = dict(summary.get("is") or {})
        passed = summary.get("passed")
        m.update({"passed": passed, "checks": summary.get("checks") or []})
        raw = {k: summary.get(k) for k in ("brain_id", "status", "stage", "grade", "date_created", "url", "os")}
        if check:
            raw["check"] = {k: check.get(k) for k in ("can_submit", "self_corr", "prod_corr", "failed", "pending")}
        store.add_brain_result(alpha_id, m, raw)
        pnl_blob, pnl_start = None, None
        if pnl is not None and len(pnl[0]):
            pnl_blob, pnl_start = pack_pnl(pnl[1]), pnl[0][0]
        store.x("INSERT INTO brain_alphas (brain_id, alpha_id, expr, settings, status, stage, metrics, checks, pnl, "
                "pnl_start, submitted, date_created, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(brain_id) DO UPDATE SET alpha_id=excluded.alpha_id, status=excluded.status, "
                "stage=excluded.stage, metrics=excluded.metrics, checks=excluded.checks, "
                "pnl=COALESCE(excluded.pnl, brain_alphas.pnl), pnl_start=COALESCE(excluded.pnl_start, "
                "brain_alphas.pnl_start), submitted=excluded.submitted, fetched_at=excluded.fetched_at",
                (summary.get("brain_id"), alpha_id, summary.get("expr"), json.dumps(summary.get("settings") or {}),
                 summary.get("status"), summary.get("stage"),
                 json.dumps({"is": summary.get("is"), "os": summary.get("os"), "check": raw.get("check")}),
                 json.dumps(summary.get("checks") or []), pnl_blob, pnl_start,
                 int(str(summary.get("stage") or "").upper() in ("OS", "PROD")), summary.get("date_created"), now()))
        if check and check.get("can_submit"):
            status = "ready"
        else:
            status = "passed" if passed else ("failed" if passed is False else "tested")
        store.update_alpha(alpha_id, {"brain_alpha_id": summary.get("brain_id"), "status_brain": status})
        a = store.get_alpha(alpha_id) or {}
        if passed is not None:
            from ..gen.bandit import Bandit
            bandit = Bandit(store)
            for arm in (f"idea:{a.get('idea') or 'other'}", f"cat:{a.get('category') or 'pv'}"):
                bandit.update(arm, 1.0 if passed else 0.0, weight=5.0)
        return {"id": alpha_id, "brain_id": summary.get("brain_id"), "expr": a.get("expr") or summary.get("expr"),
                "status": "PASS" if passed else ("FAIL" if passed is False else "PENDING"),
                "settings": a.get("settings") or summary.get("settings"),
                "sharpe": m.get("sharpe"), "fitness": m.get("fitness"), "turnover": m.get("turnover"),
                "returns": m.get("returns"), "passed": passed, "failed": summary.get("failed") or [],
                "status_brain": status, "url": summary.get("url"),
                "self_corr": (check or {}).get("self_corr"), "prod_corr": (check or {}).get("prod_corr"),
                "local_sharpe": a.get("sharpe"), "grade": a.get("grade")}

    def local_alpha_for(self, expr: str, settings: dict, origin: str = "brain", tags: list[str] | None = None,
                        notes: str = "") -> int | None:
        an = analyze(expr, local_fields=self.ws.local_fields())
        if not an.ok:
            return None
        from ..sim.simulator import SimSettings
        s = SimSettings.from_dict(settings)
        found = self.ws.store.find_alpha(an.canon_hash, s.key())
        if found:
            return int(found["id"])
        r = self.ws.save_alpha(expr, s.to_dict(), origin=origin, extras=False, tags=tags, notes=notes)
        return int(r["id"]) if r.get("ok") else None

    # ------------------------------------------------------------------ simulation
    def simulate_alphas(self, h, items: list[dict], *, check: bool | None = None) -> list[dict]:
        """Simulate ``[{alpha_id?, expr, settings, tags?, notes?}]`` on BRAIN and record every result."""
        self.ensure()
        st = self.settings
        check = st["check_passing"] if check is None else check
        left = self.budget_left()
        if left <= 0:
            raise BrainError(f"Today's BRAIN simulation budget ({st['daily_budget']}) is used up. Raise it on the "
                             "BRAIN page.")
        if len(items) > left:
            h.update(note=f"budget allows {left} of {len(items)} simulations today")
            items = items[:left]
        for it in items:
            if not it.get("alpha_id"):
                it["alpha_id"] = self.local_alpha_for(it["expr"], it.get("settings") or {},
                                                      tags=it.get("tags"), notes=it.get("notes", ""))
        multi = {"auto": None, "on": True, "off": False}.get(str(st["multi"]), None)
        rows: list[dict] = []
        h.update(phase=f"simulating {len(items)} alphas on BRAIN", done=0, total=len(items))

        def on_result(i: int, res: dict) -> None:
            h.bump("brain_simulated")
            self.count_usage(1)
            h.update(done=h.progress.get("done", 0) + 1)

        results = self.client.simulate([{"expr": it["expr"], "settings": it.get("settings") or {}} for it in items],
                                       concurrency=int(st["concurrency"]), multi=multi, on_result=on_result,
                                       cancelled=h.cancelled)
        h.update(phase="fetching BRAIN results", done=0, total=len(items))
        for it, res in zip(items, results):
            h.check()
            if not res.get("ok") or not res.get("alpha_id"):
                h.bump("brain_errors")
                row = {"id": it.get("alpha_id"), "expr": it["expr"], "passed": None, "status": "FAIL", "status_brain": "error",
                       "message": res.get("message") or res.get("status")}
                if it.get("alpha_id"):
                    self.ws.store.update_alpha(it["alpha_id"], {"status_brain": "error"})
                rows.append(row)
                h.add_result(row)
                continue
            try:
                summary = summarize_alpha(self.client.get_alpha(res["alpha_id"]))
                chk = self.client.check(res["alpha_id"]) if check and summary.get("passed") else None
            except BrainError as e:
                rows.append({"id": it.get("alpha_id"), "expr": it["expr"], "passed": None, "status": "FAIL", "status_brain": "error",
                             "message": str(e)})
                continue
            if it.get("alpha_id") is None:
                m = summary.get("is") or {}
                rows.append({"id": None, "brain_id": summary.get("brain_id"), "expr": it["expr"],
                             "sharpe": m.get("sharpe"), "fitness": m.get("fitness"), "turnover": m.get("turnover"),
                             "passed": summary.get("passed"), "failed": summary.get("failed") or [],
                             "status_brain": "tested", "status": "PASS" if summary.get("passed") else "FAIL",
                             "url": summary.get("url")})
                continue
            row = self.record_result(int(it["alpha_id"]), summary, check=chk)
            h.bump("brain_passed" if summary.get("passed") else "brain_failed")
            if chk and chk.get("can_submit"):
                h.bump("brain_ready")
            rows.append(row)
            h.add_result(row)
            h.update(done=h.progress.get("done", 0) + 1)
        self.ws.refit_calibration()
        return rows

    # ------------------------------------------------------------------ queries
    def brain_rows(self, alpha_id: int) -> list[dict]:
        out = []
        for r in self.ws.store.q("SELECT * FROM brain_alphas WHERE alpha_id=? ORDER BY fetched_at DESC", (alpha_id,)):
            d = dict(r)
            for k in ("settings", "metrics", "checks"):
                try:
                    d[k] = json.loads(d.get(k) or "null")
                except ValueError:
                    pass
            blob = d.pop("pnl", None)
            if blob:
                import numpy as np
                daily = unpack_pnl(blob).astype(float)
                d["pnl"] = {"start": d.get("pnl_start"), "cum_pnl": np.round(np.cumsum(daily), 0).tolist()}
            out.append(d)
        return out


def get_service(ws) -> BrainService:
    svc = getattr(ws, "_brain_service", None)
    if svc is None:
        svc = BrainService(ws)
        ws._brain_service = svc
    return svc


__all__ = ["BrainService", "DEFAULT_BRAIN_SETTINGS", "get_service"]
