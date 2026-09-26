"""Evaluate normalized expression trees on a panel slice, with a byte-budgeted subexpression cache."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from ..fastexpr.ast import Const, Field, Node, Op
from .ops import REGISTRY, Groups
from .panel import Panel


class EvalError(Exception):
    pass


class SubexprCache:
    """Thread-safe LRU keyed by (subtree key, context signature), bounded by total array bytes."""

    def __init__(self, max_mb: float = 600):
        self.max_bytes = int(max_mb * 1024 * 1024)
        self._d: OrderedDict[tuple, Any] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _size(v: Any) -> int:
        if isinstance(v, np.ndarray):
            return v.nbytes
        if isinstance(v, Groups):
            return v.nbytes
        return 64

    def get(self, key: tuple) -> Any:
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key: tuple, v: Any) -> None:
        sz = self._size(v)
        if sz > self.max_bytes // 4:
            return
        with self._lock:
            if key in self._d:
                return
            self._d[key] = v
            self._bytes += sz
            while self._bytes > self.max_bytes and self._d:
                _, old = self._d.popitem(last=False)
                self._bytes -= self._size(old)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._d), "mb": round(self._bytes / 1048576, 1), "hits": self.hits,
                "misses": self.misses, "hit_rate": round(self.hits / total, 3) if total else 0.0}


class EvalContext:
    def __init__(self, panel: Panel, r0: int, r1: int, universe: str | None, pasteurize: bool,
                 cache: SubexprCache | None = None):
        self.panel = panel
        self.r0, self.r1 = int(r0), int(r1)
        self.T = self.r1 - self.r0
        self.N = panel.N
        self.universe = universe
        self.pasteurize = pasteurize
        self.cache = cache
        self.sig = (panel.version, self.r0, self.r1, universe, bool(pasteurize))
        self.rsig = (panel.version, self.r0, self.r1)  # signature for universe-independent slices
        self.univ_mask = (self._cached_raw(("$u", universe),
                                           lambda: np.ascontiguousarray(panel.universe(universe)[self.r0:self.r1],
                                                                        dtype=np.bool_))
                          if universe else None)
        self._ones: np.ndarray | None = None

    def _cached_raw(self, key: tuple, fn):
        if self.cache is None:
            return fn()
        k = (key, self.rsig)
        v = self.cache.get(k)
        if v is None:
            v = fn()
            self.cache.put(k, v)
        return v

    def raw(self, name: str) -> np.ndarray:
        """Unmasked float32 slice of a field (e.g. returns for PnL), cached."""
        return self._cached_raw(("$raw", name), lambda: np.array(self.panel.field(name)[self.r0:self.r1],
                                                                 dtype=np.float32))

    def codes(self, name: str) -> tuple[np.ndarray, int]:
        """Group codes broadcast to (T, N) int32 plus the number of groups, cached."""
        def load():
            if name in ("market", "country"):
                return Groups(np.zeros((self.T, self.N), np.int32), 1)
            c, G = self.panel.group(name)
            return Groups(np.ascontiguousarray(np.broadcast_to(c[None, :], (self.T, self.N)), dtype=np.int32), G)
        g = self._cached_raw(("$codes", name), load)
        return g.codes, g.G

    def ones(self) -> np.ndarray:
        if self._ones is None:
            self._ones = np.ones((self.T, self.N), np.float32)
        return self._ones

    def _cached(self, key: str, fn):
        if self.cache is None:
            return fn()
        k = (key, self.sig)
        v = self.cache.get(k)
        if v is None:
            v = fn()
            self.cache.put(k, v)
        return v

    def field(self, name: str) -> np.ndarray:
        def load():
            try:
                mm = self.panel.field(name)
            except KeyError:
                raise EvalError(f"Field '{name}' is not available in the local dataset") from None
            a = np.array(mm[self.r0:self.r1], dtype=np.float32)
            if self.pasteurize and self.univ_mask is not None:
                a[~self.univ_mask] = np.nan
            return a
        return self._cached("$f:" + name, load)

    def group(self, name: str) -> Groups:
        def load():
            try:
                codes, G = self.panel.group(name)
            except KeyError:
                raise EvalError(f"Group '{name}' is not available in the local dataset") from None
            c = np.ascontiguousarray(np.broadcast_to(codes[None, :], (self.T, self.N)), dtype=np.int32)
            return Groups(c, G)
        return self._cached("$g:" + name, load)

    def evaluate(self, node: Node) -> Any:
        if isinstance(node, Const):
            return node.value
        if isinstance(node, Field):
            if node.name in self.panel.group_labels or node.name in ("market", "country"):
                return self.group(node.name)
            return self.field(node.name)
        assert isinstance(node, Op)
        if self.cache is not None:
            k = (node.key, self.sig)
            hit = self.cache.get(k)
            if hit is not None:
                return hit
        fn = REGISTRY.get(node.name)
        if fn is None:
            raise EvalError(f"Operator {node.name}() has no local implementation (BRAIN-only)")
        args = [self.evaluate(a) for a in node.args]
        res = fn(self, args, dict(node.params))
        if isinstance(res, np.ndarray) and res.shape != (self.T, self.N):
            res = np.ascontiguousarray(np.broadcast_to(res, (self.T, self.N)), dtype=np.float32)
        if self.cache is not None:
            self.cache.put((node.key, self.sig), res)
        return res

    def evaluate_matrix(self, node: Node) -> np.ndarray:
        r = self.evaluate(node)
        if isinstance(r, Groups):
            raise EvalError("The alpha evaluates to a group, not a numeric signal")
        if not isinstance(r, np.ndarray):
            return np.full((self.T, self.N), np.float32(r), np.float32)
        return r
