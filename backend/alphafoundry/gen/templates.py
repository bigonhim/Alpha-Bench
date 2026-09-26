"""Template library loading and expansion."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..catalog import field_map
from ..config import CATALOG_DIR, RUNTIME_DIR, atomic_write_text
from ..fastexpr import analyze
from .build import CAPBUCKET_TEXT

USER_TEMPLATES = RUNTIME_DIR / "templates_user.yaml"


@dataclass
class Template:
    id: str
    idea: str
    category: str
    horizon: str
    expr: str
    slots: dict
    settings: dict
    rationale: str = ""
    source: str = "builtin"

    def to_json(self) -> dict:
        return {"id": self.id, "idea": self.idea, "category": self.category, "horizon": self.horizon,
                "expr": self.expr, "slots": self.slots, "settings": self.settings, "rationale": self.rationale,
                "source": self.source}


@dataclass
class Candidate:
    expr: str
    settings: dict
    template_id: str | None = None
    idea: str | None = None
    category: str | None = None
    horizon: str | None = None
    rationale: str | None = None
    origin: str = "template"
    parents: list = field(default_factory=list)


def _load_file(path, source: str) -> tuple[dict, list[Template]]:
    if not path.exists():
        return {}, []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults", {}).get("settings", {})
    out = []
    for t in data.get("templates", []) or []:
        out.append(Template(id=str(t["id"]), idea=t.get("idea", "other"), category=t.get("category", "pv"),
                            horizon=t.get("horizon", "medium"), expr=t["expr"], slots=t.get("slots") or {},
                            settings={**defaults, **(t.get("settings") or {})}, rationale=t.get("rationale", ""),
                            source=source))
    return defaults, out


def load_templates() -> list[Template]:
    _, builtin = _load_file(CATALOG_DIR / "templates.yaml", "builtin")
    _, user = _load_file(USER_TEMPLATES, "user")
    by_id = {t.id: t for t in builtin}
    for t in user:
        by_id[t.id] = t
    return list(by_id.values())


def save_user_template(t: dict) -> dict:
    """Validate and persist a user template (runtime/templates_user.yaml)."""
    tmpl = Template(id=str(t["id"]), idea=t.get("idea", "other"), category=t.get("category", "pv"),
                    horizon=t.get("horizon", "medium"), expr=t["expr"], slots=t.get("slots") or {},
                    settings=t.get("settings") or {}, rationale=t.get("rationale", ""), source="user")
    sample = next(iter(expand(tmpl, local_fields=None, limit=1)), None)
    if sample is None:
        raise ValueError("Template does not expand to any valid expression")
    existing = []
    if USER_TEMPLATES.exists():
        existing = (yaml.safe_load(USER_TEMPLATES.read_text(encoding="utf-8")) or {}).get("templates", []) or []
    existing = [e for e in existing if str(e.get("id")) != tmpl.id]
    d = tmpl.to_json()
    d.pop("source")
    existing.append(d)
    atomic_write_text(USER_TEMPLATES, yaml.safe_dump({"templates": existing}, sort_keys=False))
    return tmpl.to_json()


def delete_user_template(tid: str) -> bool:
    if not USER_TEMPLATES.exists():
        return False
    existing = (yaml.safe_load(USER_TEMPLATES.read_text(encoding="utf-8")) or {}).get("templates", []) or []
    new = [e for e in existing if str(e.get("id")) != tid]
    atomic_write_text(USER_TEMPLATES, yaml.safe_dump({"templates": new}, sort_keys=False))
    return len(new) != len(existing)


def slot_values(spec: dict, local_fields: set[str] | None) -> list[str]:
    if "fields" in spec:
        vals = [str(v) for v in spec["fields"]]
        return [v for v in vals if local_fields is None or v in local_fields]
    if "select" in spec:
        sel = spec["select"] or {}
        out = []
        for fid, f in field_map().items():
            if all(str(f.get(k)) == str(v) for k, v in sel.items()) and str(f.get("type", "MATRIX")) == "MATRIX":
                if local_fields is None or fid in local_fields:
                    out.append(fid)
        return sorted(out)
    if "windows" in spec:
        return [str(int(w)) for w in spec["windows"]]
    if "groups" in spec:
        out = []
        for g in spec["groups"]:
            out.append(CAPBUCKET_TEXT if g == "capbucket" else str(g))
        return out
    if "values" in spec:
        return [str(v) for v in spec["values"]]
    return []


def settings_grid(t: Template, base: dict | None = None, override: dict | None = None) -> list[dict]:
    g = dict(t.settings)
    if override:
        g.update({k: v for k, v in override.items() if v})
    keys = [k for k in ("decay", "neutralization", "truncation") if k in g]
    vals = [g[k] if isinstance(g[k], list) else [g[k]] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        s = dict(base or {})
        s.update(dict(zip(keys, combo)))
        out.append(s)
    return out or [dict(base or {})]


def expand(t: Template, local_fields: set[str] | None = None, limit: int | None = None,
           rng: random.Random | None = None, validate: bool = True) -> list[str]:
    names = list(t.slots)
    value_lists = [slot_values(t.slots[n] or {}, local_fields) for n in names]
    if any(not v for v in value_lists):
        return []
    combos = list(itertools.product(*value_lists)) if names else [()]
    if rng is not None:
        rng.shuffle(combos)
    out: list[str] = []
    seen: set[str] = set()
    for combo in combos:
        text = t.expr
        for n, v in zip(names, combo):
            text = text.replace("{" + n + "}", v)
        if validate:
            an = analyze(text, local_fields=local_fields)
            if not an.ok or an.canonical in seen:
                continue
            if local_fields is not None and not an.local:
                continue
            seen.add(an.canonical)
        out.append(text)
        if limit and len(out) >= limit:
            break
    return out


def template_candidates(templates: list[Template], base_settings: dict, local_fields: set[str] | None,
                        per_template: int = 50, settings_per_expr: int = 2, rng: random.Random | None = None,
                        settings_override: dict | None = None) -> list[Candidate]:
    rng = rng or random.Random(0)
    out: list[Candidate] = []
    for t in templates:
        exprs = expand(t, local_fields=local_fields, limit=per_template, rng=rng)
        grid = settings_grid(t, base_settings, settings_override)
        for e in exprs:
            for s in (rng.sample(grid, min(settings_per_expr, len(grid))) if len(grid) > settings_per_expr else grid):
                out.append(Candidate(expr=e, settings=s, template_id=t.id, idea=t.idea, category=t.category,
                                     horizon=t.horizon, rationale=t.rationale, origin="template"))
    return out


def load_alpha101() -> list[dict]:
    data = yaml.safe_load((CATALOG_DIR / "alpha101.yaml").read_text(encoding="utf-8")) or {}
    return data.get("alphas", [])


def alpha101_candidates(base_settings: dict, local_fields: set[str] | None,
                        settings_variants: list[dict] | None = None) -> list[Candidate]:
    variants = settings_variants or [{"decay": 4, "neutralization": "SUBINDUSTRY"},
                                     {"decay": 8, "neutralization": "INDUSTRY"}]
    out = []
    for a in load_alpha101():
        an = analyze(a["expr"], local_fields=local_fields)
        if not an.ok or (local_fields is not None and not an.local):
            continue
        for v in variants:
            out.append(Candidate(expr=a["expr"], settings={**base_settings, **v}, template_id=a["id"],
                                 idea="alpha101", category="pv", horizon="short",
                                 rationale=f"101 Formulaic Alphas {a['id']} (Kakushadze 2015)", origin="alpha101"))
    return out


def templates_summary(local_fields: set[str] | None) -> list[dict[str, Any]]:
    out = []
    for t in load_templates():
        n_local = len(expand(t, local_fields=local_fields, validate=False)) if local_fields is not None else None
        d = t.to_json()
        d["n_expansions"] = len(expand(t, local_fields=None, validate=False))
        d["n_local"] = n_local
        d["brain_only"] = n_local == 0
        out.append(d)
    return out
