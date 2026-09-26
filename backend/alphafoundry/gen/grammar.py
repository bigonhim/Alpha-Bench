"""Typed, unit-aware random expression generator (the search space for grammar mining and GP)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..catalog import field_map
from ..fastexpr.ast import Const, Field, Node, Op
from ..fastexpr.lower import PHYSICAL_UNITS, unit_of
from .build import GROUP_NAMES, capbucket, mk

WINDOWS = (2, 3, 5, 10, 20, 40, 60, 120, 252)
SHORT_WINDOWS = (2, 3, 5, 10, 20)
LONG_WINDOWS = (20, 40, 60, 120, 252)

TS_UNARY = ("ts_mean", "ts_sum", "ts_std_dev", "ts_zscore", "ts_rank", "ts_delta", "ts_delay", "ts_decay_linear",
            "ts_min", "ts_max", "ts_arg_max", "ts_arg_min", "ts_av_diff", "ts_ir", "ts_skewness", "ts_median",
            "ts_scale")
TS_BINARY = ("ts_corr", "ts_covariance")
CS_UNARY = ("rank", "zscore", "quantile")
GROUP_OPS = ("group_rank", "group_neutralize", "group_zscore")
ROOT_NORMALIZERS = ("rank", "group_rank", "zscore", "group_zscore", "quantile", "group_neutralize")
SCALE_FIELDS = ("cap", "assets", "equity", "sales", "enterprise_value")

# operator families for point mutation (same signature)
FAMILIES = [
    ("ts_mean", "ts_median", "ts_decay_linear", "ts_sum"),
    ("ts_zscore", "ts_rank", "ts_scale", "ts_av_diff"),
    ("ts_std_dev", "ts_skewness", "ts_ir"),
    ("ts_min", "ts_max"),
    ("ts_arg_max", "ts_arg_min"),
    ("ts_corr", "ts_covariance"),
    ("rank", "zscore", "quantile"),
    ("group_rank", "group_zscore", "group_neutralize"),
    ("add", "subtract"),
    ("abs", "sign", "reverse"),
]
FAMILY_OF = {op: fam for fam in FAMILIES for op in fam}


@dataclass
class GrammarConfig:
    max_depth: int = 4
    root_normalize_prob: float = 0.85
    fundamental_ratio_prob: float = 0.6
    category_weights: dict = field(default_factory=lambda: {"pv": 0.6, "fundamental": 0.4})
    allow_trade_when: bool = True


class Grammar:
    def __init__(self, local_fields: set[str] | None, rng: random.Random, cfg: GrammarConfig | None = None,
                 fields_whitelist: list[str] | None = None):
        self.rng = rng
        self.cfg = cfg or GrammarConfig()
        fm = field_map()
        self.fields = fm
        pool = []
        for fid, f in fm.items():
            if str(f.get("type", "MATRIX")) != "MATRIX":
                continue
            if local_fields is not None and fid not in local_fields:
                continue
            if fields_whitelist and fid not in fields_whitelist:
                continue
            if fid in ("split", "dividend"):
                continue
            pool.append((fid, str(f.get("category", "other")), str(f.get("unit", "unknown"))))
        self.pool = pool
        self.by_cat: dict[str, list[tuple[str, str, str]]] = {}
        for p in pool:
            self.by_cat.setdefault(p[1], []).append(p)
        self.groups = [g for g in GROUP_NAMES if local_fields is None or g in local_fields or g == "market"]
        self.local_fields = local_fields

    # ------------------------------------------------------------------ leaves
    def pick_category(self) -> str:
        cats = [c for c in self.cfg.category_weights if c in self.by_cat]
        if not cats:
            cats = list(self.by_cat)
        w = [self.cfg.category_weights.get(c, 0.1) for c in cats]
        return self.rng.choices(cats, weights=w)[0]

    def field_leaf(self, category: str | None = None) -> Node:
        cat = category or self.pick_category()
        choices = self.by_cat.get(cat) or self.pool
        fid, cat, unit = self.rng.choice(choices)
        node: Node = Field(fid)
        if cat == "fundamental":
            if self.rng.random() < 0.5:
                node = mk("ts_backfill", node, lookback=120)
            if unit == "dollar" and self.rng.random() < self.cfg.fundamental_ratio_prob:
                scales = [s for s in SCALE_FIELDS if s != fid and (self.local_fields is None or s in self.local_fields)]
                if scales:
                    den = self.rng.choice(scales)
                    dnode: Node = Field(den)
                    if self.fields.get(den, {}).get("category") == "fundamental":
                        dnode = mk("ts_backfill", dnode, lookback=120)
                    node = mk("divide", node, dnode)
        return node

    def group(self) -> Node:
        if self.rng.random() < 0.12 and (self.local_fields is None or "cap" in self.local_fields):
            return capbucket()
        return Field(self.rng.choice(self.groups or ["market"]))

    def window(self, short: bool | None = None) -> int:
        if short is True:
            return self.rng.choice(SHORT_WINDOWS)
        if short is False:
            return self.rng.choice(LONG_WINDOWS)
        return self.rng.choice(WINDOWS)

    # ------------------------------------------------------------------ trees
    def matrix(self, depth: int) -> Node:
        r = self.rng.random()
        if depth <= 1 or r < 0.18:
            return self.field_leaf()
        if r < 0.62:
            op = self.rng.choice(TS_UNARY)
            x = self.matrix(depth - 1)
            if op in ("ts_delay",):
                return mk("rank", mk(op, x, d=self.window()))
            return mk(op, x, d=self.window())
        if r < 0.70:
            op = self.rng.choice(TS_BINARY)
            a = mk("rank", self.matrix(depth - 2)) if depth > 2 else self.field_leaf("pv")
            b = mk("rank", self.matrix(depth - 2)) if depth > 2 else self.field_leaf("pv")
            if a.key == b.key:
                b = mk("rank", Field("volume")) if self.local_fields is None or "volume" in self.local_fields else b
            return mk(op, a, b, d=self.rng.choice((5, 10, 20, 60)))
        if r < 0.85:
            op = self.rng.choice(("add", "subtract", "multiply", "divide"))
            a = self.matrix(depth - 1)
            b = self.matrix(depth - 1)
            if op in ("add", "subtract"):
                a, b = self._unit_safe(a, b)
            if op == "divide":
                b = self._safe_denominator(b)
            return mk(op, a, b)
        if r < 0.93:
            op = self.rng.choice(GROUP_OPS)
            return mk(op, self.matrix(depth - 1), self.group())
        if r < 0.97 and self.cfg.allow_trade_when and (self.local_fields is None or {"volume", "adv20"} <= self.local_fields):
            cond = mk("greater", Field("volume"), mk("multiply", Const(self.rng.choice((1.0, 1.5, 2.0))), Field("adv20")))
            return mk("trade_when", cond, self.matrix(depth - 1), Const(-1.0))
        op = self.rng.choice(("abs", "sign", "reverse", "signed_power"))
        x = self.matrix(depth - 1)
        if op == "signed_power":
            return mk(op, mk("rank", x) if self.rng.random() < 0.5 else x, Const(self.rng.choice((0.5, 2.0))))
        return mk(op, x)

    def _unit_safe(self, a: Node, b: Node) -> tuple[Node, Node]:
        ua, ub = unit_of(a, self.fields), unit_of(b, self.fields)
        if ua in PHYSICAL_UNITS and ub in PHYSICAL_UNITS and ua != ub:
            return mk("rank", a), mk("rank", b)
        return a, b

    def _safe_denominator(self, b: Node) -> Node:
        if isinstance(b, Op) and b.name in ("rank", "ts_rank", "ts_scale"):
            return mk("add", b, Const(0.01))
        if isinstance(b, Op) and b.name in ("ts_delta", "ts_zscore", "zscore", "reverse", "subtract", "ts_av_diff"):
            return mk("add", mk("abs", b), Const(0.01))
        return b

    def tree(self, max_depth: int | None = None) -> Node:
        d = max_depth or self.cfg.max_depth
        body = self.matrix(self.rng.randint(2, max(2, d - 1)))
        if self.rng.random() < self.cfg.root_normalize_prob:
            root = self.rng.choice(ROOT_NORMALIZERS)
            if root.startswith("group_"):
                body = mk(root, body, self.group())
            else:
                body = mk(root, body)
        if self.rng.random() < 0.5:
            body = mk("reverse", body)
        from ..fastexpr.lower import simplify
        return simplify(body)


def is_degenerate(n: Node) -> bool:
    """Reject trees without data, constant outputs, or silly nesting."""
    fields = [x for x in n.walk() if isinstance(x, Field) and x.name not in GROUP_NAMES]
    if not fields:
        return True
    for x in n.walk():
        if isinstance(x, Op):
            if x.name in ("rank", "zscore", "quantile") and isinstance(x.args[0], Op) and \
                    x.args[0].name in ("rank", "zscore", "quantile"):
                return True
            if x.name == "ts_delay" and isinstance(x.args[0], Op) and x.args[0].name == "ts_delay":
                return True
            if x.name in ("ts_corr", "ts_covariance", "subtract", "divide") and len(x.args) == 2 and \
                    x.args[0].key == x.args[1].key:
                return True
    return False
