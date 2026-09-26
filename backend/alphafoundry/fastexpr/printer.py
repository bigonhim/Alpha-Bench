"""Render normalized trees back to BRAIN Fast Expression text.

The output always re-parses to the same canonical tree (round-trip property). Arithmetic and
logical operators print as infix with minimal parentheses; optional scalar parameters are printed
as keywords only when they differ from their defaults.
"""

from __future__ import annotations

import math
from typing import Any

from ..catalog import operator_map
from ..catalog.operators import OP_TO_INFIX
from .ast import Const, Field, Node, Op, fmt_num

_PREC = {"||": 2, "&&": 3, "==": 4, "!=": 4, "<": 5, "<=": 5, ">": 5, ">=": 5, "+": 6, "-": 6, "*": 7, "/": 7}
_ASSOCIATIVE = {"+", "*", "&&", "||"}
_UNARY_PREC = 8
_ATOM = 100
# optional scalar params that are conventionally written positionally on BRAIN
_POSITIONAL_OPTIONAL = {("ts_backfill", "lookback")}


def _same_default(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _fmt_param(kind: str, v: Any) -> str:
    if kind == "b":
        return "true" if v else "false"
    if kind in ("w", "i"):
        return str(int(v))
    if kind == "n":
        return fmt_num(float(v))
    if kind == "e":
        return str(v)
    return '"' + str(v) + '"'


def _infix_ok(node: Op) -> str | None:
    sym = OP_TO_INFIX.get(node.name)
    if sym is None:
        return None
    for k, v in node.params:
        if k == "filter" and v:
            return None
    if sym in ("+", "*") and len(node.args) < 2:
        return None
    if sym not in ("+", "*", "&&", "||") and len(node.args) != 2:
        return None
    return sym


class _Printer:
    def __init__(self, spaced: bool = True):
        self.sp = spaced
        self.ops = operator_map()

    def render(self, node: Node) -> tuple[str, int]:
        """Return (text, precedence) for ``node``."""
        if isinstance(node, Const):
            s = fmt_num(node.value)
            return s, (_UNARY_PREC if s.startswith("-") else _ATOM)
        if isinstance(node, Field):
            return node.name, _ATOM
        assert isinstance(node, Op)
        if node.name == "reverse":
            inner, p = self.render(node.args[0])
            if p < _ATOM or inner.startswith("-"):
                inner = f"({inner})"
            return "-" + inner, _UNARY_PREC
        sym = _infix_ok(node)
        if sym is not None:
            prec = _PREC[sym]
            parts = []
            for i, a in enumerate(node.args):
                t, p = self.render(a)
                need = p < prec or (i > 0 and p == prec) or (i > 0 and t.startswith("-"))
                if sym in _ASSOCIATIVE and i > 0 and p == prec and isinstance(a, Op) and a.name == node.name:
                    need = True
                parts.append(f"({t})" if need else t)
            joiner = f" {sym} " if self.sp else sym
            return joiner.join(parts), prec
        return self.call(node), _ATOM

    def call(self, node: Op) -> str:
        spec = self.ops.get(node.name)
        args = [self.render(a)[0] for a in node.args]
        if spec is None:
            return f"{node.name}({', '.join(args)})"
        rendered: list[str] = []
        data_iter = iter(args)
        pv = dict(node.params)
        positional = True
        n_data_decl = len(spec.data_params)
        data_seen = 0
        for p in spec.params:
            if p.is_data:
                rendered.append(next(data_iter))
                data_seen += 1
                if spec.variadic and data_seen == n_data_decl:
                    rendered.extend(data_iter)  # variadic tail right after declared data params
                continue
            v = pv.get(p.name, p.default)
            if not p.has_default:
                if positional:
                    rendered.append(_fmt_param(p.kind, v))
                else:
                    rendered.append(f"{p.name}={_fmt_param(p.kind, v)}")
                continue
            if _same_default(v, p.default):
                positional = False
                continue
            if positional and (node.name, p.name) in _POSITIONAL_OPTIONAL:
                rendered.append(_fmt_param(p.kind, v))
            else:
                positional = False
                rendered.append(f"{p.name}={_fmt_param(p.kind, v)}")
        sep = ", " if self.sp else ","
        return f"{node.name}({sep.join(rendered)})"


def to_expr(node: Node, compact: bool = False) -> str:
    return _Printer(spaced=not compact).render(node)[0]
