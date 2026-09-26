"""AST definitions.

Two layers:

* Surface nodes (``R*``) mirror the text exactly and carry source spans for diagnostics.
* Normalized nodes (``Const``, ``Field``, ``Op``) are what everything else works on: variables are
  inlined, infix operators become calls, keyword/positional arguments are resolved against the
  operator catalog (all scalar parameters explicit), and commutative arguments are ordered. Each
  normalized node carries a precomputed canonical ``key`` so structural equality, hashing, caching
  and deduplication are cheap.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterator

# --------------------------------------------------------------------------- surface syntax


@dataclass(frozen=True, slots=True)
class RNode:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RNum(RNode):
    value: float
    text: str


@dataclass(frozen=True, slots=True)
class RStr(RNode):
    value: str


@dataclass(frozen=True, slots=True)
class RBool(RNode):
    value: bool


@dataclass(frozen=True, slots=True)
class RIdent(RNode):
    name: str


@dataclass(frozen=True, slots=True)
class RArg(RNode):
    name: str | None
    value: RNode


@dataclass(frozen=True, slots=True)
class RCall(RNode):
    name: str
    name_end: int
    args: tuple[RArg, ...]


@dataclass(frozen=True, slots=True)
class RUnary(RNode):
    op: str
    operand: RNode


@dataclass(frozen=True, slots=True)
class RBinary(RNode):
    op: str
    left: RNode
    right: RNode


@dataclass(frozen=True, slots=True)
class RTernary(RNode):
    cond: RNode
    a: RNode
    b: RNode


@dataclass(frozen=True, slots=True)
class RAssign(RNode):
    name: str
    value: RNode


@dataclass(frozen=True, slots=True)
class RProgram(RNode):
    assigns: tuple[RAssign, ...]
    result: RNode


# --------------------------------------------------------------------------- normalized tree


def fmt_num(v: float) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    r = repr(float(v))
    return r


def fmt_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return fmt_num(v)
    return '"' + str(v) + '"'


class Node:
    __slots__ = ("key", "_h")
    key: str

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return self._h

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and other.key == self.key

    def __repr__(self) -> str:
        return self.key

    @property
    def children(self) -> tuple["Node", ...]:
        return ()

    def walk(self) -> Iterator["Node"]:
        yield self
        for c in self.children:
            yield from c.walk()

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    @property
    def depth(self) -> int:
        ch = self.children
        return 1 + (max(c.depth for c in ch) if ch else 0)

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.key.encode("utf-8")).hexdigest()[:16]


class Const(Node):
    __slots__ = ("value",)

    def __init__(self, value: float):
        self.value = float(value)
        self.key = fmt_num(self.value)
        self._h = hash(self.key)


class Field(Node):
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name
        self.key = name
        self._h = hash(self.key)


class Op(Node):
    """Operator application. ``args`` are data arguments (matrix/group/vector expressions);
    ``params`` are scalar literals in catalog order, always explicit (defaults filled)."""

    __slots__ = ("name", "args", "params")

    def __init__(self, name: str, args: tuple[Node, ...], params: tuple[tuple[str, Any], ...] = ()):
        self.name = name
        self.args = tuple(args)
        self.params = tuple(params)
        parts = [a.key for a in self.args]
        parts.extend(f"{k}={fmt_scalar(v)}" for k, v in self.params)
        self.key = f"{name}({','.join(parts)})"
        self._h = hash(self.key)

    @property
    def children(self) -> tuple[Node, ...]:
        return self.args

    def param(self, name: str, default: Any = None) -> Any:
        for k, v in self.params:
            if k == name:
                return v
        return default

    def with_args(self, args: tuple[Node, ...]) -> "Op":
        return Op(self.name, args, self.params)

    def with_param(self, name: str, value: Any) -> "Op":
        return Op(self.name, self.args, tuple((k, value if k == name else v) for k, v in self.params))


def replace_at(root: Node, path: tuple[int, ...], new: Node) -> Node:
    """Return a copy of ``root`` with the subtree at ``path`` (child indices) replaced."""
    if not path:
        return new
    assert isinstance(root, Op)
    i = path[0]
    args = list(root.args)
    args[i] = replace_at(args[i], path[1:], new)
    return root.with_args(tuple(args))


def iter_paths(root: Node, path: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], Node]]:
    yield path, root
    if isinstance(root, Op):
        for i, a in enumerate(root.args):
            yield from iter_paths(a, path + (i,))


def node_at(root: Node, path: tuple[int, ...]) -> Node:
    n = root
    for i in path:
        assert isinstance(n, Op)
        n = n.args[i]
    return n
