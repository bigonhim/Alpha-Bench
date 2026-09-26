"""PnL correlation index: fast max-correlation lookups and clustered correlation matrices."""

from __future__ import annotations

import threading

import numpy as np


class CorrelationIndex:
    """Standardized daily PnL vectors aligned on a fixed date index.

    ``max_corr`` is a single matrix-vector product, so checking a candidate against thousands of
    stored alphas costs about a millisecond.
    """

    def __init__(self, dates: np.ndarray):
        self.dates = np.asarray(dates, dtype="datetime64[D]")
        self._pos = {d: i for i, d in enumerate(self.dates.tolist())}
        self.ids: list[int] = []
        self.sharpes: list[float] = []
        self._rows: list[np.ndarray] = []
        self._mat: np.ndarray | None = None
        self._lock = threading.Lock()

    def _align(self, dates: np.ndarray, pnl: np.ndarray) -> np.ndarray:
        v = np.zeros(len(self.dates))
        idx = np.fromiter((self._pos.get(d, -1) for d in np.asarray(dates, dtype="datetime64[D]").tolist()),
                          dtype=np.int64, count=len(dates))
        ok = idx >= 0
        v[idx[ok]] = np.asarray(pnl, dtype=np.float64)[ok]
        return v

    @staticmethod
    def _std(v: np.ndarray) -> np.ndarray | None:
        v = v - v.mean()
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def add(self, alpha_id: int, dates: np.ndarray, pnl: np.ndarray, sharpe: float = 0.0) -> None:
        z = self._std(self._align(dates, pnl))
        if z is None:
            return
        with self._lock:
            if alpha_id in self.ids:
                k = self.ids.index(alpha_id)
                self._rows[k] = z
                self.sharpes[k] = sharpe
            else:
                self.ids.append(alpha_id)
                self._rows.append(z)
                self.sharpes.append(sharpe)
            self._mat = None

    def remove(self, alpha_id: int) -> None:
        with self._lock:
            if alpha_id in self.ids:
                k = self.ids.index(alpha_id)
                del self.ids[k], self._rows[k], self.sharpes[k]
                self._mat = None

    def __len__(self) -> int:
        return len(self.ids)

    def _matrix(self) -> np.ndarray:
        with self._lock:
            if self._mat is None:
                self._mat = np.vstack(self._rows) if self._rows else np.zeros((0, len(self.dates)))
            return self._mat

    def correlations(self, dates: np.ndarray, pnl: np.ndarray) -> np.ndarray:
        z = self._std(self._align(dates, pnl))
        M = self._matrix()
        if z is None or not len(M):
            return np.zeros(len(M))
        return M @ z

    def max_corr(self, dates: np.ndarray, pnl: np.ndarray, exclude: set[int] | None = None) -> dict | None:
        if not self.ids:
            return None
        c = self.correlations(dates, pnl)
        if exclude:
            for k, i in enumerate(self.ids):
                if i in exclude:
                    c[k] = -np.inf
        if not len(c) or not np.isfinite(c).any():
            return None
        k = int(np.argmax(c))
        return {"max_corr": float(c[k]), "alpha_id": self.ids[k], "alpha_sharpe": self.sharpes[k]}

    def top(self, dates: np.ndarray, pnl: np.ndarray, n: int = 5, exclude: set[int] | None = None) -> list[dict]:
        if not self.ids:
            return []
        c = self.correlations(dates, pnl)
        order = np.argsort(-np.abs(c))
        out = []
        for k in order:
            if exclude and self.ids[k] in exclude:
                continue
            out.append({"alpha_id": self.ids[k], "corr": round(float(c[k]), 4)})
            if len(out) >= n:
                break
        return out

    def matrix(self, ids: list[int] | None = None) -> tuple[list[int], np.ndarray]:
        M = self._matrix()
        sel = list(range(len(self.ids))) if ids is None else [self.ids.index(i) for i in ids if i in self.ids]
        sub = M[sel]
        return [self.ids[k] for k in sel], sub @ sub.T


def cluster_order(C: np.ndarray) -> list[int]:
    """Average-linkage agglomerative ordering on distance 1 - |corr| (leaf order for a heatmap)."""
    n = C.shape[0]
    if n <= 2:
        return list(range(n))
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, np.inf)
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    size = {i: 1 for i in range(n)}
    active = list(range(n))
    Dm = D.copy()
    nxt = n
    idx_of = {i: i for i in range(n)}
    big = np.full((2 * n, 2 * n), np.inf)
    big[:n, :n] = Dm
    while len(active) > 1:
        sub = big[np.ix_(active, active)]
        k = int(np.argmin(sub))
        a, b = active[k // len(active)], active[k % len(active)]
        if a == b:
            break
        new = nxt
        nxt += 1
        clusters[new] = clusters.pop(a) + clusters.pop(b)
        size[new] = size[a] + size[b]
        for c in active:
            if c in (a, b):
                continue
            d = (big[a, c] * size[a] + big[b, c] * size[b]) / size[new]
            big[new, c] = big[c, new] = d
        active = [c for c in active if c not in (a, b)] + [new]
    del idx_of
    return clusters[active[0]] if active else list(range(n))
