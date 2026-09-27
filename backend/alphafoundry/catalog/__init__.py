"""Catalog access: operators (code), data fields (JSON seed + user imports), checks, templates."""

from __future__ import annotations

import json
import threading
from typing import Any

from ..config import CATALOG_DIR, USER_FIELDS_PATH, atomic_write_text
from .operators import OPERATORS, OpSpec, Param, operator_map  # noqa: F401

_fields_lock = threading.Lock()
_fields_cache: dict[str, dict] | None = None

GROUP_FIELDS_LOCAL = ("market", "sector", "industry", "subindustry", "exchange", "country")


def _load_seed_fields() -> list[dict]:
    data = json.loads((CATALOG_DIR / "fields.json").read_text(encoding="utf-8"))
    return data["fields"]


def _load_user_fields() -> list[dict]:
    if not USER_FIELDS_PATH.exists():
        return []
    try:
        return json.loads(USER_FIELDS_PATH.read_text(encoding="utf-8")).get("fields", [])
    except (OSError, ValueError):
        return []


def field_map() -> dict[str, dict]:
    """All known fields keyed by id. User-imported fields override/extend the seed catalog."""
    global _fields_cache
    with _fields_lock:
        if _fields_cache is None:
            m: dict[str, dict] = {}
            for f in _load_seed_fields():
                m[f["id"]] = dict(f, source="seed")
            for f in _load_user_fields():
                fid = f.get("id")
                if not fid:
                    continue
                base = m.get(fid, {})
                if base:
                    # a seed field keeps its local definition; BRAIN adds usage statistics and confirms the id
                    merged = {**f, **base, "verified": True, "on_brain": True}
                    for k in ("coverage", "alpha_count", "user_count", "dataset_name", "subcategory"):
                        if f.get(k) is not None:
                            merged[k] = f[k]
                else:
                    merged = {**f, "source": "user"}
                merged.setdefault("local", base.get("local", False))
                merged.setdefault("type", "MATRIX")
                merged.setdefault("category", "other")
                merged.setdefault("unit", "unknown")
                m[fid] = merged
            _fields_cache = m
        return _fields_cache


def invalidate_fields() -> None:
    global _fields_cache
    with _fields_lock:
        _fields_cache = None


def import_user_fields(rows: list[dict[str, Any]]) -> int:
    """Merge user-provided BRAIN field definitions into runtime/fields_user.json."""
    existing = {f["id"]: f for f in _load_user_fields() if f.get("id")}
    n = 0
    for r in rows:
        fid = str(r.get("id") or "").strip()
        if not fid:
            continue
        t = str(r.get("type") or "MATRIX").upper()
        if t not in ("MATRIX", "VECTOR", "GROUP"):
            t = "MATRIX"
        existing[fid] = {
            "id": fid,
            "dataset": str(r.get("dataset") or r.get("dataset_id") or "user"),
            "category": str(r.get("category") or "other").lower(),
            "type": t,
            "unit": str(r.get("unit") or "unknown"),
            "description": str(r.get("description") or ""),
            "local": False,
            "verified": True,
            "coverage": r.get("coverage"),
        }
        for k in ("dataset_name", "subcategory", "alpha_count", "user_count", "region", "delay", "universe",
                  "source_brain"):
            if r.get(k) is not None:
                existing[fid][k] = r[k]
        n += 1
    atomic_write_text(USER_FIELDS_PATH, json.dumps({"fields": list(existing.values())}, indent=1))
    invalidate_fields()
    return n


def catalog_json() -> dict:
    return {
        "operators": [o.to_json() for o in OPERATORS],
        "fields": list(field_map().values()),
    }
