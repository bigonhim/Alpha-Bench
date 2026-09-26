"""Helpers for building and editing normalized trees in a type-safe, canonical way."""

from __future__ import annotations

from typing import Any

from ..catalog import field_map, operator_map
from ..fastexpr.ast import Const, Field, Node, Op
from ..fastexpr.lower import simplify

CAPBUCKET_TEXT = 'bucket(rank(cap), range="0.1,1,0.1")'
GROUP_NAMES = ("market", "sector", "industry", "subindustry")


def mk(name: str, *args: Node | float, **params: Any) -> Op:
    """Build an Op with catalog defaults filled (canonical parameter order)."""
    spec = operator_map()[name]
    nodes = tuple(a if isinstance(a, Node) else Const(float(a)) for a in args)
    ps = []
    for p in spec.scalar_params:
        v = params.get(p.name, p.default)
        if p.kind in ("w", "i") and v is not None:
            v = int(v)
        elif p.kind == "n" and v is not None:
            v = float(v)
        ps.append((p.name, v))
    return Op(name, nodes, tuple(ps))


def capbucket() -> Op:
    return mk("bucket", mk("rank", Field("cap")), range="0.1,1,0.1")


def group_node(name: str) -> Node:
    return capbucket() if name == "capbucket" else Field(name)


def node_type(n: Node) -> str:
    """'m' matrix, 'g' group, 'v' vector."""
    if isinstance(n, Const):
        return "m"
    if isinstance(n, Field):
        t = str(field_map().get(n.name, {}).get("type", "MATRIX")).upper()
        return {"GROUP": "g", "VECTOR": "v"}.get(t, "m")
    assert isinstance(n, Op)
    return operator_map()[n.name].returns


def arg_kind(parent: Op, index: int) -> str:
    spec = operator_map()[parent.name]
    dps = spec.data_params
    if index < len(dps):
        return dps[index].kind
    return "m"  # variadic tail


def canon(n: Node) -> Node:
    return simplify(n)


def fields_of(n: Node) -> set[str]:
    return {x.name for x in n.walk() if isinstance(x, Field)}
