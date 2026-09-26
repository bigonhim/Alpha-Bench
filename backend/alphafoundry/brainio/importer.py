"""Parse BRAIN results pasted/uploaded by the user (CSV, BRAIN-API-style JSON, or copied text)."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from ..fastexpr import analyze

EXPR_KEYS = ("regular", "code", "expr", "expression", "alpha", "formula", "fast_expression")
METRICS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin")
SETTING_KEYS = {"region": "region", "universe": "universe", "delay": "delay", "decay": "decay",
                "neutralization": "neutralization", "truncation": "truncation", "pasteurization": "pasteurization",
                "nanhandling": "nanHandling", "nan_handling": "nanHandling", "unithandling": "unitHandling"}
PCT_METRICS = ("turnover", "returns", "drawdown")


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s or s.lower() in ("nan", "none", "-", "n/a"):
        return None
    pct = s.endswith("%")
    bps = s.endswith("‱") or s.lower().endswith("bps")
    s = re.sub(r"[%‱a-zA-Z\s]", "", s)
    try:
        x = float(s)
    except ValueError:
        return None
    if pct:
        return x / 100.0
    if bps:
        return x
    return x


def _norm_metric(key: str, v: float | None, raw: Any) -> float | None:
    if v is None:
        return None
    if key in PCT_METRICS and not (isinstance(raw, str) and raw.strip().endswith("%")):
        if abs(v) > 2.0:  # e.g. "12.3" meaning 12.3 %
            return v / 100.0
    if key == "margin" and abs(v) < 0.05 and v != 0:  # fraction -> bps
        return v * 1e4
    return v


def _passed_from(obj: dict) -> bool | None:
    for k in ("passed", "pass", "status", "result", "is_passed"):
        if k in obj and obj[k] not in (None, ""):
            v = str(obj[k]).strip().upper()
            if v in ("TRUE", "1", "PASS", "PASSED", "YES", "SUBMITTED", "ACTIVE", "OK"):
                return True
            if v in ("FALSE", "0", "FAIL", "FAILED", "NO", "REJECTED"):
                return False
    checks = obj.get("checks")
    if isinstance(checks, list) and checks:
        res = [str(c.get("result", "")).upper() for c in checks if isinstance(c, dict)]
        if any(r == "FAIL" for r in res):
            return False
        if res and all(r in ("PASS", "WARNING", "PENDING") for r in res):
            return True
    return None


def _row_from_dict(d: dict) -> dict | None:
    low = {str(k).strip().lower(): v for k, v in d.items()}
    expr = None
    reg = low.get("regular")
    if isinstance(reg, dict):
        expr = reg.get("code")
    for k in EXPR_KEYS:
        if expr:
            break
        v = low.get(k)
        if isinstance(v, str) and v.strip():
            expr = v
    if not expr:
        return None
    settings = {}
    sd = low.get("settings") if isinstance(low.get("settings"), dict) else low
    for k, v in (sd or {}).items():
        kk = SETTING_KEYS.get(str(k).lower())
        if kk and v not in (None, ""):
            settings[kk] = v
    metrics_src = low.get("is") if isinstance(low.get("is"), dict) else low
    ms = {str(k).lower(): v for k, v in metrics_src.items()}
    m: dict[str, Any] = {}
    for k in METRICS:
        raw = ms.get(k)
        if raw is None:
            for alt in (f"is_{k}", f"is {k}", f"{k} (is)"):
                if alt in ms:
                    raw = ms[alt]
                    break
        m[k] = _norm_metric(k, _num(raw), raw)
    checks = metrics_src.get("checks") if isinstance(metrics_src, dict) else None
    if isinstance(checks, list):
        m["checks"] = [{"name": c.get("name"), "result": c.get("result"), "value": c.get("value"),
                        "limit": c.get("limit")} for c in checks if isinstance(c, dict)]
    p = _passed_from({**low, **({"checks": checks} if checks else {})})
    m["passed"] = p
    submitted = str(low.get("status", "")).upper() in ("ACTIVE", "SUBMITTED")
    return {"expr": expr.strip(), "settings": settings, "metrics": m, "submitted": submitted, "raw": d}


def parse_csv(text: str) -> list[dict]:
    dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;\t") if text.strip() else csv.excel
    rows = []
    for d in csv.DictReader(io.StringIO(text), dialect=dialect):
        r = _row_from_dict(d)
        if r:
            rows.append(r)
    return rows


def parse_json(text: str) -> list[dict]:
    data = json.loads(text)
    if isinstance(data, dict):
        for k in ("results", "alphas", "data", "items"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            data = [data]
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict):
            r = _row_from_dict(d)
            if r:
                out.append(r)
    return out


_METRIC_RE = {k: re.compile(rf"{k}\s*[:=]?\s*(-?\d+(?:\.\d+)?\s*(?:%|‱|bps)?)", re.I) for k in METRICS}


def parse_text(text: str) -> list[dict]:
    """Blocks separated by blank lines; the first line(s) that parse as an expression are the alpha."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        expr = None
        for ln in lines:
            cand = ln.split("\t")[0]
            if any(k in cand.lower() for k in ("sharpe", "fitness", "turnover")) and not re.search(r"\(", cand):
                continue
            an = analyze(cand, local_ops=set())
            if an.ok and "(" in cand or (an.ok and len(cand.split()) <= 3):
                expr = cand
                break
        if not expr:
            continue
        m: dict[str, Any] = {}
        for k, rx in _METRIC_RE.items():
            mm = rx.search(block)
            raw = mm.group(1) if mm else None
            m[k] = _norm_metric(k, _num(raw), raw)
        up = block.upper()
        m["passed"] = True if re.search(r"\bPASS(ED)?\b", up) and "FAIL" not in up else (
            False if "FAIL" in up else None)
        out.append({"expr": expr, "settings": {}, "metrics": m, "submitted": "SUBMITTED" in up, "raw": block})
    return out


def parse_any(text: str, fmt: str = "auto") -> list[dict]:
    t = text.strip()
    if not t:
        return []
    if fmt == "json" or (fmt == "auto" and t[:1] in "[{"):
        return parse_json(t)
    if fmt == "csv" or (fmt == "auto" and "\n" in t and ("," in t.splitlines()[0] or "\t" in t.splitlines()[0])
                        and any(k in t.splitlines()[0].lower() for k in EXPR_KEYS)):
        return parse_csv(t)
    return parse_text(t)


def infer_passed(m: dict, delay: int = 1) -> bool | None:
    if m.get("passed") is not None:
        return m["passed"]
    sh, fit, to = m.get("sharpe"), m.get("fitness"), m.get("turnover")
    if sh is None or fit is None:
        return None
    ok = sh >= (2.0 if delay == 0 else 1.25) and fit >= (1.3 if delay == 0 else 1.0)
    if to is not None:
        ok = ok and 0.01 <= to <= 0.7
    return ok
