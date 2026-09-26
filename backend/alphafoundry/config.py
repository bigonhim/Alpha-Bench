"""Paths and persisted application settings."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
CATALOG_DIR = PACKAGE_DIR / "catalog"
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

RUNTIME_DIR = Path(os.environ.get("ALPHAFOUNDRY_RUNTIME", PROJECT_DIR / "runtime")).resolve()
PANELS_DIR = RUNTIME_DIR / "panels"
DEMO_PANELS_DIR = RUNTIME_DIR / "panels_demo"
RAW_DIR = RUNTIME_DIR / "raw"
DB_PATH = RUNTIME_DIR / "alphafoundry.db"
SETTINGS_PATH = RUNTIME_DIR / "settings.json"
USER_FIELDS_PATH = RUNTIME_DIR / "fields_user.json"
USER_CHECKS_PATH = RUNTIME_DIR / "checks_user.yaml"
LOG_DIR = RUNTIME_DIR / "logs"

DEFAULT_SETTINGS: dict[str, Any] = {
    "sec_contact_email": "",          # required by SEC fair-access policy for EDGAR downloads
    "history_start": "2012-01-01",    # first date downloaded
    "is_years_warmup": 2,             # IS starts this many years after history_start
    "os_years": 2,                    # most recent years held out as OS
    "brain_window_years": 5,          # BRAIN-like window = last N IS years
    "workers": 2,                     # miner process-pool size
    "api_cache_mb": 600,
    "worker_cache_mb": 300,
    "returns_basis": "half_book",     # half_book (verify) | full_book
    "booksize": 20_000_000.0,
    "theme": "dark",
    "active_dataset": "auto",         # auto | real | demo
}

_lock = threading.Lock()


def ensure_dirs() -> None:
    for d in (RUNTIME_DIR, PANELS_DIR, DEMO_PANELS_DIR, RAW_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    ensure_dirs()
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return s


def save_settings(updates: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        s = load_settings()
        for k, v in updates.items():
            if k in DEFAULT_SETTINGS:
                s[k] = v
        atomic_write_text(SETTINGS_PATH, json.dumps(s, indent=2))
        return s


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via temp file + rename, retrying briefly when OneDrive/AV holds a lock."""
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    os.replace(tmp, path)
