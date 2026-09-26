"""Pass-likelihood model and local-vs-BRAIN calibration.

A logistic model maps local proxy metrics to the probability that BRAIN's checks pass. It starts
from hand-set prior coefficients and is refit (ridge-regularized IRLS toward the prior) once enough
BRAIN results have been imported. A linear map from local to BRAIN Sharpe is also maintained.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATURES = ("bias", "sharpe", "fitness", "turnover", "os_ratio", "sub_ratio", "complexity")
PRIOR = np.array([-4.0, 1.6, 1.4, -1.5, 0.8, 0.6, -0.15])
MIN_FIT_ROWS = 30


def features(m: dict) -> np.ndarray:
    sh = float(m.get("sharpe") or 0.0)
    os_sh = m.get("os_sharpe")
    sub_sh = m.get("sub_sharpe")
    os_ratio = float(np.clip((os_sh / sh) if (os_sh is not None and sh > 0) else 0.6, -1.0, 1.5))
    sub_ratio = float(np.clip((sub_sh / sh) if (sub_sh is not None and sh > 0) else 0.7, -1.0, 1.5))
    return np.array([1.0, sh, float(m.get("fitness") or 0.0), min(1.5, float(m.get("turnover") or 0.0)),
                     os_ratio, sub_ratio, float(m.get("complexity") or 8) / 10.0])


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class Calibrator:
    def __init__(self, state: dict | None = None):
        state = state or {}
        self.coef = np.array(state.get("coef", PRIOR.tolist()), dtype=float)
        self.n_fit = int(state.get("n_fit", 0))
        self.sharpe_map = state.get("sharpe_map")  # {"slope","intercept","r2","n"}
        self.ratios = state.get("ratios", {})

    def state(self) -> dict:
        return {"coef": self.coef.tolist(), "n_fit": self.n_fit, "sharpe_map": self.sharpe_map,
                "ratios": self.ratios}

    def predict(self, m: dict, failed_hard: int = 0) -> float:
        z = float(features(m) @ self.coef) - 2.0 * failed_hard
        return round(_sigmoid(z), 4)

    def expected_brain_sharpe(self, local_sharpe: float) -> float | None:
        if not self.sharpe_map:
            return None
        return round(self.sharpe_map["slope"] * local_sharpe + self.sharpe_map["intercept"], 3)

    def fit(self, rows: list[dict]) -> dict:
        """rows: brain_results joined with local metrics (keys local_sharpe, sharpe (brain), passed, ...)."""
        pairs = [(r.get("local_sharpe"), r.get("sharpe")) for r in rows
                 if r.get("local_sharpe") is not None and r.get("sharpe") is not None]
        if len(pairs) >= 5:
            x = np.array([p[0] for p in pairs], float)
            y = np.array([p[1] for p in pairs], float)
            A = np.vstack([x, np.ones_like(x)]).T
            (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ np.array([slope, icpt])
            ss = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else 0.0
            self.sharpe_map = {"slope": round(float(slope), 4), "intercept": round(float(icpt), 4),
                               "r2": round(r2, 4), "n": len(pairs)}
        for key, lk in (("turnover", "local_turnover"), ("returns", "local_returns"), ("fitness", "local_fitness")):
            ratios = [r[key] / r[lk] for r in rows if r.get(key) and r.get(lk) and abs(r[lk]) > 1e-9]
            if len(ratios) >= 3:
                self.ratios[key] = round(float(np.median(ratios)), 4)
        labeled = [r for r in rows if r.get("passed") is not None and r.get("local_sharpe") is not None]
        if len(labeled) >= MIN_FIT_ROWS:
            X = np.vstack([features({"sharpe": r["local_sharpe"], "fitness": r.get("local_fitness"),
                                     "turnover": r.get("local_turnover"), "os_sharpe": r.get("os_sharpe"),
                                     "sub_sharpe": r.get("sub_sharpe"), "complexity": r.get("complexity")})
                           for r in labeled])
            y = np.array([1.0 if r["passed"] else 0.0 for r in labeled])
            w = PRIOR.copy()
            lam = 2.0
            for _ in range(25):
                z = X @ w
                p = 1 / (1 + np.exp(-z))
                W = p * (1 - p) + 1e-6
                grad = X.T @ (y - p) - lam * (w - PRIOR)
                H = (X.T * W) @ X + lam * np.eye(len(w))
                step = np.linalg.solve(H, grad)
                w = w + step
                if np.abs(step).max() < 1e-6:
                    break
            self.coef = w
            self.n_fit = len(labeled)
        return self.state()


def summary(rows: list[dict], cal: Calibrator) -> dict[str, Any]:
    pts = [{"local": r.get("local_sharpe"), "brain": r.get("sharpe"),
            "passed": None if r.get("passed") is None else bool(r.get("passed")),
            "alpha_id": r.get("alpha_id"), "family": r.get("idea") or r.get("family")} for r in rows
           if r.get("local_sharpe") is not None and r.get("sharpe") is not None]
    fam: dict[str, list] = {}
    for r in rows:
        k = r.get("idea") or r.get("family") or "other"
        fam.setdefault(k, []).append(1 if r.get("passed") else 0)
    return {
        "points": pts,
        "sharpe_map": cal.sharpe_map,
        "ratios": cal.ratios,
        "n_fit": cal.n_fit,
        "coef": dict(zip(FEATURES, [round(float(c), 3) for c in cal.coef])),
        "families": {k: {"n": len(v), "pass_rate": round(sum(v) / len(v), 3)} for k, v in fam.items()},
        "min_rows_for_fit": MIN_FIT_ROWS,
    }
