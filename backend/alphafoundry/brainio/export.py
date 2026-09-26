"""Export alphas in BRAIN-ready formats (no network calls: the user pastes/runs these themselves)."""

from __future__ import annotations

import csv
import io
import json

BRAIN_SETTING_KEYS = ("instrumentType", "region", "universe", "delay", "decay", "neutralization", "truncation",
                      "pasteurization", "unitHandling", "nanHandling", "language", "visualization")


def brain_settings(s: dict) -> dict:
    out = {
        "instrumentType": s.get("instrumentType", "EQUITY"),
        "region": s.get("region", "USA"),
        "universe": s.get("universe", "TOP3000"),
        "delay": int(s.get("delay", 1)),
        "decay": int(s.get("decay", 0)),
        "neutralization": s.get("neutralization", "SUBINDUSTRY"),
        "truncation": float(s.get("truncation", 0.08)),
        "pasteurization": s.get("pasteurization", "ON"),
        "unitHandling": s.get("unitHandling", "VERIFY"),
        "nanHandling": s.get("nanHandling", "OFF"),
        "language": "FASTEXPR",
        "visualization": False,
    }
    return out


def simulation_payload(expr: str, settings: dict) -> dict:
    return {"type": "REGULAR", "settings": brain_settings(settings), "regular": expr}


def to_json(alphas: list[dict], batch: int = 10) -> str:
    payloads = [simulation_payload(a["expr"], a.get("settings") or {}) for a in alphas]
    if batch and batch > 1:
        batches = [payloads[i:i + batch] for i in range(0, len(payloads), batch)]
        return json.dumps({"multi_simulations": batches, "count": len(payloads)}, indent=2)
    return json.dumps(payloads, indent=2)


def to_text(alphas: list[dict], with_settings: bool = False) -> str:
    lines = []
    for a in alphas:
        if with_settings:
            s = brain_settings(a.get("settings") or {})
            lines.append(f"# {s['region']} {s['universe']} D{s['delay']} decay={s['decay']} "
                         f"neut={s['neutralization']} trunc={s['truncation']}")
        lines.append(a["expr"])
    return "\n".join(lines) + "\n"


CSV_COLS = ("id", "expr", "region", "universe", "delay", "decay", "neutralization", "truncation", "pasteurization",
            "nanHandling", "sharpe", "fitness", "turnover", "returns", "drawdown", "margin", "os_sharpe",
            "sub_sharpe", "pass_prob", "status_local", "status_brain", "origin", "idea", "category", "horizon",
            "tags", "notes")


def to_csv(alphas: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLS)
    for a in alphas:
        s = brain_settings(a.get("settings") or {})
        row = []
        for c in CSV_COLS:
            if c in s:
                row.append(s[c])
            elif c == "tags":
                row.append(",".join(a.get("tags") or []))
            else:
                v = a.get(c)
                row.append("" if v is None else v)
        w.writerow(row)
    return buf.getvalue()


def to_markdown(a: dict, result: dict | None = None) -> str:
    s = brain_settings(a.get("settings") or {})
    desc = a.get("description") or {}
    lines = [f"# Alpha #{a.get('id')}", "", "```", a["expr"], "```", "",
             f"**Settings:** {s['region']} · {s['universe']} · delay {s['delay']} · decay {s['decay']} · "
             f"{s['neutralization']} · truncation {s['truncation']} · pasteurization {s['pasteurization']}", ""]
    if a.get("sharpe") is not None:
        lines += ["| Metric | Local IS | Local OS |", "|---|---|---|",
                  f"| Sharpe | {a.get('sharpe'):.2f} | {a.get('os_sharpe') or 0:.2f} |",
                  f"| Fitness | {a.get('fitness') or 0:.2f} | |",
                  f"| Turnover | {(a.get('turnover') or 0) * 100:.1f}% | |",
                  f"| Returns | {(a.get('returns') or 0) * 100:.2f}% | |",
                  f"| Drawdown | {(a.get('drawdown') or 0) * 100:.2f}% | |",
                  f"| Margin | {a.get('margin') or 0:.1f} bps | |", ""]
    if isinstance(desc, dict) and desc:
        lines += ["## Idea", desc.get("idea", ""), "", "## Data", desc.get("data", ""), "", "## Operators"]
        lines += [f"- {o}" for o in desc.get("operators", [])]
    if result and result.get("checks"):
        lines += ["", "## Local checks"] + [f"- {c['name']}: {c['result']} — {c['message']}"
                                            for c in result["checks"]]
    lines += ["", "_Local metrics come from Alpha Foundry's proxy simulator (free data); confirm on BRAIN._"]
    return "\n".join(lines)
