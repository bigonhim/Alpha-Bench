"""Lowering, type checking, canonicalization and static analysis of Fast Expressions."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..catalog import field_map, operator_map
from ..catalog.operators import INFIX_TO_OP, OpSpec, Param
from .ast import (Const, Field, Node, Op, RBinary, RBool, RCall, RIdent, RNode, RNum, RProgram, RStr, RTernary,
                  RUnary)
from .parser import ParseError, parse

TYPE_NAMES = {"m": "numeric", "g": "group", "v": "vector"}


@dataclass
class Diagnostic:
    severity: str  # error | warning | info
    message: str
    start: int
    end: int

    def to_json(self) -> dict:
        return {"severity": self.severity, "message": self.message, "start": self.start, "end": self.end}


class LowerError(Exception):
    def __init__(self, message: str, start: int, end: int):
        super().__init__(message)
        self.message, self.start, self.end = message, start, max(end, start + 1)


@dataclass
class Analysis:
    text: str
    ok: bool
    node: Node | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    lookback: int = 0
    depth: int = 0
    size: int = 0
    op_count: int = 0
    local: bool = False
    brain_only_reasons: list[str] = field(default_factory=list)
    canonical: str = ""
    canon_hash: str = ""
    pretty: str = ""

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "diagnostics": [d.to_json() for d in self.diagnostics],
            "fields": self.fields,
            "operators": self.operators,
            "categories": self.categories,
            "lookback": self.lookback,
            "depth": self.depth,
            "size": self.size,
            "op_count": self.op_count,
            "local": self.local,
            "brain_only_reasons": self.brain_only_reasons,
            "canonical": self.canonical,
            "canon_hash": self.canon_hash,
            "pretty": self.pretty,
        }


# --------------------------------------------------------------------------- lowering


class _Lowerer:
    def __init__(self, text: str):
        self.text = text
        self.ops = operator_map()
        self.fields = field_map()
        self.env: dict[str, tuple[Node, str]] = {}
        self.diags: list[Diagnostic] = []
        self.spans: dict[str, tuple[int, int]] = {}

    def warn(self, msg: str, n: RNode, severity: str = "warning") -> None:
        self.diags.append(Diagnostic(severity, msg, n.start, n.end))

    def program(self, prog: RProgram) -> Node:
        for a in prog.assigns:
            if a.name in self.ops:
                raise LowerError(f"'{a.name}' is an operator name and cannot be used as a variable", a.start,
                                 a.start + len(a.name))
            if a.name in self.fields:
                self.warn(f"Variable '{a.name}' shadows the data field of the same name", a)
            self.env[a.name] = self.expr(a.value)
        node, typ = self.expr(prog.result)
        if typ != "m":
            raise LowerError(f"The alpha must be numeric, but this expression is a {TYPE_NAMES[typ]}",
                             prog.result.start, prog.result.end)
        return node

    def remember(self, node: Node, n: RNode) -> Node:
        self.spans.setdefault(node.key, (n.start, n.end))
        return node

    def expr(self, n: RNode) -> tuple[Node, str]:
        if isinstance(n, RNum):
            return Const(n.value), "m"
        if isinstance(n, RBool):
            return Const(1.0 if n.value else 0.0), "m"
        if isinstance(n, RStr):
            raise LowerError('Text literals are only allowed as operator options (e.g. range="0,1,0.1")', n.start, n.end)
        if isinstance(n, RIdent):
            return self.ident(n)
        if isinstance(n, RUnary):
            x, t = self.expr(n.operand)
            self.need("m", t, n.operand, "operand of unary " + n.op)
            if n.op == "-":
                if isinstance(x, Const):
                    return Const(-x.value), "m"
                return self.remember(self.make("reverse", (x,)), n), "m"
            return self.remember(self.make("not", (x,)), n), "m"
        if isinstance(n, RBinary):
            name = INFIX_TO_OP[n.op]
            left, lt = self.expr(n.left)
            right, rt = self.expr(n.right)
            self.need("m", lt, n.left, f"left side of '{n.op}'")
            self.need("m", rt, n.right, f"right side of '{n.op}'")
            return self.remember(self.make(name, (left, right)), n), "m"
        if isinstance(n, RTernary):
            c, ct = self.expr(n.cond)
            a, at = self.expr(n.a)
            b, bt = self.expr(n.b)
            self.need("m", ct, n.cond, "condition")
            self.need("m", at, n.a, "true branch")
            self.need("m", bt, n.b, "false branch")
            return self.remember(self.make("if_else", (c, a, b)), n), "m"
        if isinstance(n, RCall):
            return self.call(n)
        raise LowerError("Unsupported syntax", n.start, n.end)

    def make(self, name: str, args: tuple[Node, ...]) -> Op:
        spec = self.ops[name]
        params = tuple((p.name, p.default) for p in spec.scalar_params)
        return Op(name, args, params)

    def ident(self, n: RIdent) -> tuple[Node, str]:
        if n.name in self.env:
            return self.env[n.name]
        if n.name.lower() == "nan":
            return Const(float("nan")), "m"
        if n.name in self.ops:
            raise LowerError(f"'{n.name}' is an operator; call it like {self.ops[n.name].signature}", n.start, n.end)
        f = self.fields.get(n.name)
        if f is None:
            lower = {k.lower(): k for k in self.fields}
            if n.name.lower() in lower:
                real = lower[n.name.lower()]
                raise LowerError(f"Unknown field '{n.name}'. Did you mean '{real}'? (names are case-sensitive)",
                                 n.start, n.end)
            sugg = difflib.get_close_matches(n.name, list(self.fields) + list(self.env), n=3, cutoff=0.75)
            hint = f" Did you mean {', '.join(repr(s) for s in sugg)}?" if sugg else ""
            self.warn(f"Unknown data field '{n.name}' (not in the catalog). It may exist on BRAIN; "
                      f"the alpha cannot be simulated locally.{hint}", n)
            return Field(n.name), "m"
        t = {"MATRIX": "m", "GROUP": "g", "VECTOR": "v"}.get(str(f.get("type", "MATRIX")).upper(), "m")
        return Field(n.name), t

    def need(self, want: str, got: str, n: RNode, what: str) -> None:
        if want == got:
            return
        if want == "m" and got == "g":
            raise LowerError(f"A group ({self._txt(n)}) cannot be used as the {what}; groups go in the group "
                             "argument of group_* operators or group_neutralize", n.start, n.end)
        if want == "m" and got == "v":
            raise LowerError(f"Vector field {self._txt(n)} must be reduced first, e.g. vec_avg({self._txt(n)})",
                             n.start, n.end)
        if want == "g":
            raise LowerError(f"Expected a group for the {what} (e.g. sector, industry, subindustry, market or "
                             f'bucket(rank(cap), range="0.1,1,0.1")), got a {TYPE_NAMES[got]} expression',
                             n.start, n.end)
        if want == "v":
            raise LowerError(f"Expected a vector field for the {what}", n.start, n.end)
        raise LowerError(f"Type mismatch for the {what}", n.start, n.end)

    def _txt(self, n: RNode) -> str:
        s = self.text[n.start:n.end]
        return f"'{s}'" if len(s) <= 40 else "this expression"

    def call(self, n: RCall) -> tuple[Node, str]:
        spec = self.ops.get(n.name)
        if spec is None:
            lower = {k.lower(): k for k in self.ops}
            if n.name.lower() in lower:
                raise LowerError(f"Unknown operator '{n.name}'. Did you mean '{lower[n.name.lower()]}'?",
                                 n.start, n.name_end)
            sugg = difflib.get_close_matches(n.name, list(self.ops), n=3, cutoff=0.6)
            hint = f" Did you mean {', '.join(sugg)}?" if sugg else ""
            raise LowerError(f"Unknown operator '{n.name}'.{hint}", n.start, n.name_end)

        positional = [a for a in n.args if a.name is None]
        keywords = [a for a in n.args if a.name is not None]
        seen_kw = False
        for a in n.args:
            if a.name is not None:
                seen_kw = True
            elif seen_kw:
                raise LowerError("Positional arguments must come before keyword arguments", a.start, a.end)

        assigned: dict[str, Any] = {}
        extra = []
        if spec.variadic:
            dps = spec.data_params
            for i, a in enumerate(positional):
                if i < len(dps):
                    assigned[dps[i].name] = a
                else:
                    extra.append(a)
        else:
            if len(positional) > len(spec.params):
                a = positional[len(spec.params)]
                raise LowerError(f"{spec.name}() takes at most {len(spec.params)} argument(s): {spec.signature}",
                                 a.start, a.end)
            for i, a in enumerate(positional):
                assigned[spec.params[i].name] = a
        by_name = {p.name: p for p in spec.params}
        by_lower = {p.name.lower(): p for p in spec.params}
        for a in keywords:
            p = by_name.get(a.name) or by_lower.get(a.name.lower())
            if p is None:
                valid = ", ".join(p.name for p in spec.params if not p.is_data) or "none"
                raise LowerError(f"{spec.name}() has no option '{a.name}' (options: {valid})", a.start, a.end)
            if p.name in assigned:
                raise LowerError(f"{spec.name}(): '{p.name}' is given twice", a.start, a.end)
            assigned[p.name] = a

        args: list[Node] = []
        params: list[tuple[str, Any]] = []
        for p in spec.params:
            a = assigned.get(p.name)
            if p.is_data:
                if a is None:
                    raise LowerError(f"{spec.name}() is missing argument '{p.name}': {spec.signature}",
                                     n.start, n.end)
                node, t = self.expr(a.value)
                self.need(p.kind, t, a.value, f"'{p.name}' argument of {spec.name}")
                args.append(node)
            else:
                if a is None:
                    if not p.has_default:
                        raise LowerError(f"{spec.name}() is missing argument '{p.name}': {spec.signature}",
                                         n.start, n.end)
                    params.append((p.name, p.default))
                else:
                    params.append((p.name, self.scalar(spec, p, a.value)))
        for a in extra:
            node, t = self.expr(a.value)
            self.need("m", t, a.value, f"argument of {spec.name}")
            args.append(node)
        self._validate(spec, params, n)
        node = Op(spec.name, tuple(args), tuple(params))
        return self.remember(node, n), spec.returns

    def scalar(self, spec: OpSpec, p: Param, v: RNode) -> Any:
        where = f"{spec.name}(): '{p.name}'"
        if isinstance(v, RIdent) and v.name in self.env and isinstance(self.env[v.name][0], Const):
            v = RNum(v.start, v.end, self.env[v.name][0].value, str(self.env[v.name][0].value))
        k = p.kind
        if k in ("w", "i"):
            if not isinstance(v, RNum):
                raise LowerError(f"{where} must be an integer literal", v.start, v.end)
            if v.value != int(v.value):
                raise LowerError(f"{where} must be a whole number (got {v.text})", v.start, v.end)
            iv = int(v.value)
            if k == "w" and iv < 1:
                raise LowerError(f"{where} must be a positive number of days", v.start, v.end)
            if k == "w" and iv > 2520:
                self.warn(f"{where}: window {iv} is longer than the available history", v)
            return iv
        if k == "n":
            if isinstance(v, RNum):
                return float(v.value)
            if isinstance(v, RBool):
                return 1.0 if v.value else 0.0
            if isinstance(v, RIdent) and v.name.lower() == "nan":
                return float("nan")
            raise LowerError(f"{where} must be a number", v.start, v.end)
        if k == "b":
            if isinstance(v, RBool):
                return v.value
            if isinstance(v, RNum) and v.value in (0.0, 1.0):
                return bool(v.value)
            raise LowerError(f"{where} must be true or false", v.start, v.end)
        if k == "s":
            if isinstance(v, RStr):
                return v.value
            if isinstance(v, RIdent):
                return v.name
            if isinstance(v, RNum):
                return v.text
            raise LowerError(f'{where} must be text, e.g. "0,1,0.1"', v.start, v.end)
        if k == "e":
            s = v.value if isinstance(v, RStr) else v.name if isinstance(v, RIdent) else None
            if s is None:
                raise LowerError(f"{where} must be one of: {', '.join(p.choices)}", v.start, v.end)
            for c in p.choices:
                if c.lower() == s.lower():
                    return c
            raise LowerError(f"{where} must be one of: {', '.join(p.choices)}", v.start, v.end)
        raise LowerError(f"{where}: unsupported option kind", v.start, v.end)

    def _validate(self, spec: OpSpec, params: list[tuple[str, Any]], n: RCall) -> None:
        pv = dict(params)
        if spec.name == "bucket":
            rng, bks = str(pv.get("range") or ""), str(pv.get("buckets") or "")
            if not rng and not bks:
                raise LowerError('bucket() needs range="start,end,step" or buckets="b1,b2,..."', n.start, n.end)
            try:
                if rng:
                    parts = [float(x) for x in rng.split(",")]
                    if len(parts) != 3 or parts[2] <= 0 or parts[1] <= parts[0]:
                        raise ValueError
                else:
                    vals = [float(x) for x in bks.split(",")]
                    if len(vals) < 1:
                        raise ValueError
            except ValueError:
                raise LowerError('bucket(): range must look like "0.1,1,0.1" (start,end,step) and buckets like '
                                 '"0.2,0.5,0.8"', n.start, n.end) from None
        elif spec.name == "ts_regression":
            if int(pv.get("rettype", 0)) not in range(10):
                raise LowerError("ts_regression(): rettype must be between 0 and 9", n.start, n.end)
        elif spec.name == "truncate":
            if not 0 < float(pv.get("maxPercent", 0.01)) <= 1:
                raise LowerError("truncate(): maxPercent must be in (0, 1]", n.start, n.end)


# --------------------------------------------------------------------------- canonical simplification

_FOLD = {
    "add": lambda *a: sum(a),
    "subtract": lambda a, b: a - b,
    "multiply": lambda *a: math.prod(a),
    "divide": lambda a, b: a / b if b != 0 else float("nan"),
    "reverse": lambda a: -a,
    "abs": abs,
    "inverse": lambda a: 1.0 / a if a != 0 else float("nan"),
    "power": lambda a, b: a ** b,
    "sqrt": lambda a: math.sqrt(a) if a >= 0 else float("nan"),
    "log": lambda a: math.log(a) if a > 0 else float("nan"),
}


def simplify(node: Node) -> Node:
    if not isinstance(node, Op):
        return node
    args = tuple(simplify(a) for a in node.args)
    name, params = node.name, node.params
    spec = operator_map()[name]
    if name in _FOLD and args and all(isinstance(a, Const) for a in args) and all(
            not v for k, v in params if k == "filter"):
        try:
            v = _FOLD[name](*[a.value for a in args])
            if isinstance(v, (int, float)) and not isinstance(v, complex):
                return Const(float(v))
        except (OverflowError, ValueError, ZeroDivisionError):
            pass
    if name == "reverse":
        (x,) = args
        if isinstance(x, Op) and x.name == "reverse":
            return x.args[0]
    if name in ("add", "multiply"):
        flat: list[Node] = []
        for a in args:
            if isinstance(a, Op) and a.name == name and a.params == params:
                flat.extend(a.args)
            else:
                flat.append(a)
        args = tuple(flat)
        if name == "multiply" and len(args) == 2:
            for i in (0, 1):
                if isinstance(args[i], Const) and args[i].value == -1.0:
                    return simplify(Op("reverse", (args[1 - i],), ()))
                if isinstance(args[i], Const) and args[i].value == 1.0 and not dict(params).get("filter"):
                    return args[1 - i]
    if name == "subtract" and isinstance(args[0], Const) and args[0].value == 0.0 and not dict(params).get("filter"):
        return simplify(Op("reverse", (args[1],), ()))
    if spec.commutative:
        args = tuple(sorted(args, key=lambda a: a.key))
    return Op(name, args, params)


# --------------------------------------------------------------------------- analysis

_WINDOW_PARAMS = {"d", "lookback", "period"}
PHYSICAL_UNITS = {"price", "shares", "dollar", "per_share"}
_SCORE_OPS = {
    "rank", "zscore", "quantile", "ts_rank", "ts_zscore", "ts_quantile", "group_rank", "group_zscore", "scale",
    "ts_scale", "group_scale", "sign", "ts_corr", "ts_ir", "ts_skewness", "ts_kurtosis", "ts_arg_max", "ts_arg_min",
    "days_from_last_change", "ts_count_nans", "less", "less_equal", "greater", "greater_equal", "equal", "not_equal",
    "and", "or", "not", "is_nan", "ts_step", "log", "s_log_1p", "tanh", "sigmoid", "arc_tan", "arc_sin", "arc_cos",
    "scale_down", "normalize", "group_normalize", "rank_by_side", "generalized_rank", "vector_neut", "regression_neut",
    "ts_returns", "log_diff", "ts_entropy", "ts_partial_corr", "ts_triple_corr", "ts_co_skewness", "ts_co_kurtosis",
    "group_count", "ts_moment", "rank_gmean_amean_diff", "inst_tvr",
}


def lookback_of(node: Node) -> int:
    if not isinstance(node, Op):
        return 0
    inner = max((lookback_of(a) for a in node.args), default=0)
    w = 0
    for k, v in node.params:
        if k in _WINDOW_PARAMS and isinstance(v, int):
            w += v
    if node.name == "ts_regression":
        w += int(node.param("lag", 0) or 0)
    return inner + w


def unit_of(node: Node, fields: dict[str, dict], warn: list[tuple[str, str]] | None = None) -> str:
    if isinstance(node, Const):
        return "const"
    if isinstance(node, Field):
        return str(fields.get(node.name, {}).get("unit", "unknown"))
    assert isinstance(node, Op)
    units = [unit_of(a, fields, warn) for a in node.args]
    name = node.name
    if name in _SCORE_OPS:
        return "score"
    if name in ("add", "subtract", "max", "min"):
        phys = [u for u in units if u in PHYSICAL_UNITS]
        if len(set(phys)) > 1 and warn is not None:
            warn.append((node.key, f"{name}() mixes units ({' vs '.join(sorted(set(phys)))}); "
                                   "rank or normalize each part before combining"))
        known = [u for u in units if u not in ("const", "score")]
        return known[0] if len(set(known)) == 1 else ("mixed" if known else "score")
    if name == "divide":
        a, b = units
        if a == b and a in PHYSICAL_UNITS:
            return "ratio"
        if b in ("const", "score"):
            return a
        return "mixed"
    if name == "multiply":
        known = [u for u in units if u not in ("const", "score")]
        return known[0] if len(known) == 1 else ("score" if not known else "mixed")
    if name in ("if_else",):
        return units[1] if units[1] not in ("const",) else units[2]
    if name == "trade_when":
        return units[1]
    if name in ("group_mean",):
        return units[0]
    return units[0] if units else "score"


def node_info(node: Node, local_ops: Iterable[str] | None = None, local_fields: Iterable[str] | None = None) -> dict:
    """Static facts about a normalized tree (used by miners and the analyzer)."""
    fields = field_map()
    if local_ops is None:
        from ..engine.ops import local_operator_names

        local_ops = local_operator_names()
    local_ops = set(local_ops)
    fnames: set[str] = set()
    onames: set[str] = set()
    op_count = 0
    for n in node.walk():
        if isinstance(n, Field):
            fnames.add(n.name)
        elif isinstance(n, Op):
            onames.add(n.name)
            op_count += 1
    reasons: list[str] = []
    for o in sorted(onames):
        if o not in local_ops:
            reasons.append(f"operator {o}() has no local implementation")
    for f in sorted(fnames):
        meta = fields.get(f)
        if meta is None:
            reasons.append(f"field '{f}' is not in the catalog")
        elif local_fields is not None and f not in set(local_fields):
            reasons.append(f"field '{f}' is not in the local dataset")
        elif local_fields is None and not meta.get("local"):
            reasons.append(f"field '{f}' is BRAIN-only data")
    cats = sorted({str(fields.get(f, {}).get("category", "other")) for f in fnames} - {"group"})
    return {
        "fields": sorted(fnames),
        "operators": sorted(onames),
        "categories": cats,
        "lookback": lookback_of(node),
        "depth": node.depth,
        "size": node.size,
        "op_count": op_count,
        "local": not reasons,
        "brain_only_reasons": reasons,
    }


def lower_text(text: str) -> Node:
    """Parse + lower + simplify; raises ``LowerError``/``ParseError`` on the first error."""
    lw = _Lowerer(text)
    return simplify(lw.program(parse(text)))


def analyze(text: str, local_ops: Iterable[str] | None = None,
            local_fields: Iterable[str] | None = None) -> Analysis:
    from .printer import to_expr

    an = Analysis(text=text, ok=False)
    if not text or not text.strip():
        an.diagnostics.append(Diagnostic("error", "Empty expression", 0, 1))
        return an
    lw = _Lowerer(text)
    try:
        prog = parse(text)
        node = simplify(lw.program(prog))
    except (ParseError, LowerError) as e:
        an.diagnostics = lw.diags + [Diagnostic("error", e.message, e.start, e.end)]
        return an
    an.node = node
    an.diagnostics = list(lw.diags)
    info = node_info(node, local_ops, local_fields)
    for k, v in info.items():
        setattr(an, k, v)
    warns: list[tuple[str, str]] = []
    unit_of(node, lw.fields, warns)
    for key, msg in warns:
        s, e = lw.spans.get(key, (0, len(text)))
        an.diagnostics.append(Diagnostic("warning", msg, s, e))
    for n in node.walk():
        if isinstance(n, Op) and n.name in ("rank", "zscore", "quantile") and isinstance(n.args[0], Op) \
                and n.args[0].name in ("rank", "zscore", "quantile"):
            s, e = lw.spans.get(n.key, (0, len(text)))
            an.diagnostics.append(Diagnostic("info", f"Nested {n.name}({n.args[0].name}(...)) is redundant", s, e))
    if isinstance(node, (Field, Const)):
        an.diagnostics.append(Diagnostic("info", "Tip: a raw field is rarely a good alpha; normalize with rank() "
                                                 "or group_rank() and add a time-series operator", 0, len(text)))
    an.ok = True
    an.canonical = node.key
    an.canon_hash = node.digest
    an.pretty = to_expr(node)
    return an
