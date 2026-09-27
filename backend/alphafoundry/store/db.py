"""SQLite persistence (WAL mode, single writer connection guarded by a lock)."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS alphas (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  expr TEXT NOT NULL,
  canon TEXT NOT NULL,
  canon_hash TEXT NOT NULL,
  settings TEXT NOT NULL,
  settings_key TEXT NOT NULL,
  origin TEXT NOT NULL DEFAULT 'manual',
  family TEXT, idea TEXT, category TEXT, horizon TEXT,
  template_id TEXT, parents TEXT, tags TEXT DEFAULT '', notes TEXT DEFAULT '',
  description TEXT,
  local INTEGER DEFAULT 1, brain_only_reasons TEXT,
  status_local TEXT DEFAULT 'PENDING', robust INTEGER DEFAULT 0,
  status_brain TEXT DEFAULT 'untested', submitted INTEGER DEFAULT 0, starred INTEGER DEFAULT 0,
  job_id INTEGER,
  sharpe REAL, fitness REAL, turnover REAL, returns REAL, drawdown REAL, margin REAL,
  os_sharpe REAL, sub_sharpe REAL, max_corr REAL, pass_prob REAL, complexity INTEGER,
  failed TEXT DEFAULT '',
  created_at TEXT, updated_at TEXT,
  UNIQUE(canon_hash, settings_key)
);
CREATE INDEX IF NOT EXISTS ix_alphas_fitness ON alphas(fitness);
CREATE INDEX IF NOT EXISTS ix_alphas_status ON alphas(status_local);
CREATE INDEX IF NOT EXISTS ix_alphas_created ON alphas(created_at);
CREATE INDEX IF NOT EXISTS ix_alphas_canon ON alphas(canon_hash);
CREATE TABLE IF NOT EXISTS results (
  alpha_id INTEGER PRIMARY KEY REFERENCES alphas(id) ON DELETE CASCADE,
  data_version TEXT, payload TEXT, pnl BLOB, dates_start TEXT, n_days INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS brain_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  alpha_id INTEGER REFERENCES alphas(id) ON DELETE CASCADE,
  sharpe REAL, fitness REAL, turnover REAL, returns REAL, drawdown REAL, margin REAL,
  passed INTEGER, checks TEXT, raw TEXT, imported_at TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, config TEXT, status TEXT, progress TEXT, stats TEXT,
  error TEXT, created_at TEXT, started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS bandit (arm TEXT PRIMARY KEY, a REAL, b REAL, n INTEGER);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS brain_alphas (
  brain_id TEXT PRIMARY KEY,
  alpha_id INTEGER REFERENCES alphas(id) ON DELETE SET NULL,
  expr TEXT, settings TEXT, status TEXT, stage TEXT, metrics TEXT, checks TEXT,
  pnl BLOB, pnl_start TEXT, submitted INTEGER DEFAULT 0, date_created TEXT, fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_brain_alphas_alpha ON brain_alphas(alpha_id);
"""

# Columns added after the first release; created on start-up when an older database is opened.
MIGRATIONS = {
    "alphas": [("brain_alpha_id", "TEXT"), ("quality", "REAL"), ("grade", "TEXT"), ("data_source", "TEXT")],
}

ALPHA_METRIC_COLS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin", "os_sharpe", "sub_sharpe",
                     "max_corr", "pass_prob", "complexity", "quality")
SORTABLE = set(ALPHA_METRIC_COLS) | {"id", "created_at", "updated_at", "status_local", "origin", "family", "grade",
                                     "status_brain"}


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def pack_pnl(pnl: np.ndarray) -> bytes:
    return zlib.compress(np.asarray(pnl, dtype=np.float32).tobytes(), 3)


def unpack_pnl(b: bytes | None) -> np.ndarray:
    if not b:
        return np.zeros(0, np.float32)
    return np.frombuffer(zlib.decompress(b), dtype=np.float32)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=10000")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, decl in cols:
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_alphas_grade ON alphas(grade)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_alphas_brain ON alphas(brain_alpha_id)")

    # ------------------------------------------------------------------ low level
    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, tuple(args)).fetchall()

    def x(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(args))

    def tx(self, fn):
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                r = fn(self.conn)
                self.conn.execute("COMMIT")
                return r
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ alphas
    def upsert_alpha(self, rec: dict) -> int:
        """Insert or update an alpha keyed by (canon_hash, settings_key). Returns id."""
        rec = dict(rec)
        rec.setdefault("created_at", now())
        rec["updated_at"] = now()
        for k in ("settings", "parents", "brain_only_reasons", "description"):
            if k in rec and not isinstance(rec[k], (str, type(None))):
                rec[k] = json.dumps(rec[k])
        if isinstance(rec.get("failed"), list):
            rec["failed"] = ",".join(rec["failed"])
        cols = [c for c in rec if c != "id"]
        placeholders = ",".join("?" for _ in cols)
        updatable = [c for c in cols if c not in ("created_at", "canon_hash", "settings_key", "origin", "tags",
                                                  "notes", "submitted", "starred", "status_brain", "parents",
                                                  "template_id", "job_id")]
        sql = (f"INSERT INTO alphas ({','.join(cols)}) VALUES ({placeholders}) "
               f"ON CONFLICT(canon_hash, settings_key) DO UPDATE SET "
               + ",".join(f"{c}=excluded.{c}" for c in updatable))
        with self._lock:
            self.conn.execute(sql, [rec[c] for c in cols])
            row = self.conn.execute("SELECT id FROM alphas WHERE canon_hash=? AND settings_key=?",
                                    (rec["canon_hash"], rec["settings_key"])).fetchone()
        return int(row["id"])

    def find_alpha(self, canon_hash: str, settings_key: str | None = None) -> dict | None:
        if settings_key:
            rows = self.q("SELECT * FROM alphas WHERE canon_hash=? AND settings_key=?", (canon_hash, settings_key))
        else:
            rows = self.q("SELECT * FROM alphas WHERE canon_hash=? ORDER BY id DESC LIMIT 1", (canon_hash,))
        return self._row(rows[0]) if rows else None

    def save_result(self, alpha_id: int, data_version: str, payload: dict, pnl: np.ndarray, dates_start: str) -> None:
        self.x("INSERT OR REPLACE INTO results (alpha_id, data_version, payload, pnl, dates_start, n_days, created_at)"
               " VALUES (?,?,?,?,?,?,?)",
               (alpha_id, data_version, json.dumps(payload), pack_pnl(pnl), dates_start, int(len(pnl)), now()))

    def get_result(self, alpha_id: int) -> dict | None:
        rows = self.q("SELECT * FROM results WHERE alpha_id=?", (alpha_id,))
        if not rows:
            return None
        r = rows[0]
        return {"data_version": r["data_version"], "payload": json.loads(r["payload"] or "{}"),
                "pnl": unpack_pnl(r["pnl"]), "dates_start": r["dates_start"], "n_days": r["n_days"]}

    def pnl_rows(self, where: str = "1=1", args: Iterable[Any] = (), limit: int = 4000) -> list[dict]:
        rows = self.q(f"SELECT a.id, a.sharpe, a.submitted, r.pnl, r.dates_start, r.data_version FROM alphas a "
                      f"JOIN results r ON r.alpha_id=a.id WHERE {where} ORDER BY a.fitness DESC LIMIT ?",
                      (*args, limit))
        return [{"id": r["id"], "sharpe": r["sharpe"] or 0.0, "submitted": bool(r["submitted"]),
                 "pnl": unpack_pnl(r["pnl"]), "dates_start": r["dates_start"], "data_version": r["data_version"]}
                for r in rows]

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("settings", "parents", "brain_only_reasons", "description"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
        d["failed"] = [f for f in (d.get("failed") or "").split(",") if f]
        d["tags"] = [t for t in (d.get("tags") or "").split(",") if t]
        return d

    def get_alpha(self, alpha_id: int) -> dict | None:
        rows = self.q("SELECT * FROM alphas WHERE id=?", (alpha_id,))
        return self._row(rows[0]) if rows else None

    def list_alphas(self, *, search: str = "", status: str = "", origin: str = "", family: str = "",
                    category: str = "", submitted: bool | None = None, starred: bool | None = None,
                    local: bool | None = None, min_sharpe: float | None = None, min_fitness: float | None = None,
                    tag: str = "", job_id: int | None = None, ids: list[int] | None = None,
                    grade: str = "", brain: str = "", data_source: str = "",
                    sort: str = "fitness", desc: bool = True, limit: int = 200, offset: int = 0) -> dict:
        where, args = ["1=1"], []
        if grade:
            gs = [g.strip().upper() for g in grade.split(",") if g.strip()]
            where.append(f"grade IN ({','.join('?' for _ in gs)})")
            args += gs
        if brain:
            if brain == "tested":
                where.append("status_brain IS NOT NULL AND status_brain NOT IN ('untested', '')")
            else:
                where.append("status_brain=?")
                args.append(brain)
        if data_source:
            where.append("COALESCE(data_source, '')=?")
            args.append(data_source)
        if search:
            where.append("(expr LIKE ? OR notes LIKE ? OR tags LIKE ?)")
            args += [f"%{search}%"] * 3
        if status:
            where.append("status_local=?")
            args.append(status)
        if origin:
            where.append("origin=?")
            args.append(origin)
        if family:
            where.append("(family=? OR idea=?)")
            args += [family, family]
        if category:
            where.append("category=?")
            args.append(category)
        if submitted is not None:
            where.append("submitted=?")
            args.append(int(submitted))
        if starred is not None:
            where.append("starred=?")
            args.append(int(starred))
        if local is not None:
            where.append("local=?")
            args.append(int(local))
        if min_sharpe is not None:
            where.append("sharpe>=?")
            args.append(min_sharpe)
        if min_fitness is not None:
            where.append("fitness>=?")
            args.append(min_fitness)
        if tag:
            where.append("(','||tags||',') LIKE ?")
            args.append(f"%,{tag},%")
        if job_id is not None:
            where.append("job_id=?")
            args.append(job_id)
        if ids:
            where.append(f"id IN ({','.join('?' for _ in ids)})")
            args += list(ids)
        col = sort if sort in SORTABLE else "fitness"
        order = f"{col} IS NULL, {col} {'DESC' if desc else 'ASC'}, id DESC"
        w = " AND ".join(where)
        total = self.q(f"SELECT COUNT(*) AS n FROM alphas WHERE {w}", args)[0]["n"]
        rows = self.q(f"SELECT * FROM alphas WHERE {w} ORDER BY {order} LIMIT ? OFFSET ?", (*args, limit, offset))
        return {"total": total, "rows": [self._row(r) for r in rows]}

    def update_alpha(self, alpha_id: int, fields: dict) -> None:
        allowed = {"tags", "notes", "submitted", "starred", "status_brain", "family", "description", "expr",
                   "brain_alpha_id", "grade", "quality", "data_source"}
        sets, args = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "tags" and isinstance(v, list):
                v = ",".join(sorted({t.strip() for t in v if t.strip()}))
            if k == "description" and not isinstance(v, (str, type(None))):
                v = json.dumps(v)
            if k in ("submitted", "starred"):
                v = int(bool(v))
            sets.append(f"{k}=?")
            args.append(v)
        if not sets:
            return
        sets.append("updated_at=?")
        args.append(now())
        self.x(f"UPDATE alphas SET {','.join(sets)} WHERE id=?", (*args, alpha_id))

    def delete_alphas(self, ids: list[int]) -> int:
        if not ids:
            return 0
        cur = self.x(f"DELETE FROM alphas WHERE id IN ({','.join('?' for _ in ids)})", ids)
        return cur.rowcount

    def stats(self) -> dict:
        r = self.q("SELECT COUNT(*) n, SUM(status_local='PASS') pass, SUM(submitted) sub, SUM(local=0) brain_only, "
                   "SUM(starred) starred, MAX(fitness) best_fitness, MAX(sharpe) best_sharpe, "
                   "SUM(grade='A') grade_a, SUM(grade='B') grade_b, SUM(data_source='demo') demo FROM alphas")[0]
        b = self.q("SELECT COUNT(*) n, SUM(passed) pass FROM brain_results")[0]
        return {"alphas": r["n"] or 0, "local_pass": r["pass"] or 0, "submitted": r["sub"] or 0,
                "brain_only": r["brain_only"] or 0, "starred": r["starred"] or 0,
                "best_fitness": r["best_fitness"], "best_sharpe": r["best_sharpe"],
                "grade_a": r["grade_a"] or 0, "grade_b": r["grade_b"] or 0, "demo_mined": r["demo"] or 0,
                "brain_imported": b["n"] or 0, "brain_passed": b["pass"] or 0}

    def facet_counts(self) -> dict:
        out = {}
        for col in ("origin", "status_local", "idea", "category", "status_brain", "grade", "data_source"):
            out[col] = {(r[0] or "none"): r[1] for r in self.q(f"SELECT {col}, COUNT(*) FROM alphas GROUP BY {col}")}
        return out

    # ------------------------------------------------------------------ brain results
    def add_brain_result(self, alpha_id: int, m: dict, raw: Any) -> None:
        self.x("INSERT INTO brain_results (alpha_id, sharpe, fitness, turnover, returns, drawdown, margin, passed, "
               "checks, raw, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
               (alpha_id, m.get("sharpe"), m.get("fitness"), m.get("turnover"), m.get("returns"), m.get("drawdown"),
                m.get("margin"), None if m.get("passed") is None else int(bool(m.get("passed"))),
                json.dumps(m.get("checks") or []), json.dumps(raw, default=str), now()))

    def brain_results(self, limit: int = 5000) -> list[dict]:
        rows = self.q("SELECT b.*, a.expr, a.sharpe AS local_sharpe, a.fitness AS local_fitness, "
                      "a.turnover AS local_turnover, a.returns AS local_returns, a.os_sharpe, a.sub_sharpe, "
                      "a.complexity, a.idea, a.category, a.family, a.origin FROM brain_results b "
                      "JOIN alphas a ON a.id=b.alpha_id ORDER BY b.id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ jobs
    def create_job(self, kind: str, config: dict) -> int:
        cur = self.x("INSERT INTO jobs (kind, config, status, progress, stats, created_at) VALUES (?,?,?,?,?,?)",
                     (kind, json.dumps(config), "queued", "{}", "{}", now()))
        return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields) -> None:
        sets, args = [], []
        for k, v in fields.items():
            if k in ("progress", "stats", "config") and not isinstance(v, str):
                v = json.dumps(v, default=float)
            sets.append(f"{k}=?")
            args.append(v)
        if sets:
            self.x(f"UPDATE jobs SET {','.join(sets)} WHERE id=?", (*args, job_id))

    def get_job(self, job_id: int) -> dict | None:
        rows = self.q("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._job(rows[0]) if rows else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        return [self._job(r) for r in self.q("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    @staticmethod
    def _job(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("config", "progress", "stats"):
            try:
                d[k] = json.loads(d.get(k) or "{}")
            except ValueError:
                d[k] = {}
        return d

    # ------------------------------------------------------------------ bandit / kv
    def bandit_all(self) -> dict[str, tuple[float, float, int]]:
        return {r["arm"]: (r["a"], r["b"], r["n"]) for r in self.q("SELECT * FROM bandit")}

    def bandit_update(self, arm: str, reward: float, weight: float = 1.0) -> None:
        with self._lock:
            row = self.conn.execute("SELECT a, b, n FROM bandit WHERE arm=?", (arm,)).fetchone()
            a, b, n = (row["a"], row["b"], row["n"]) if row else (1.0, 1.0, 0)
            a += weight * reward
            b += weight * (1.0 - reward)
            self.conn.execute("INSERT OR REPLACE INTO bandit (arm, a, b, n) VALUES (?,?,?,?)", (arm, a, b, n + 1))

    def kv_get(self, k: str, default: Any = None) -> Any:
        rows = self.q("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(rows[0]["v"]) if rows else default

    def kv_set(self, k: str, v: Any) -> None:
        self.x("INSERT OR REPLACE INTO kv (k, v) VALUES (?,?)", (k, json.dumps(v)))
