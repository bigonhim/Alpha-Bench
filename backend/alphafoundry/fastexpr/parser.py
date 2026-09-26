"""Lexer and Pratt parser for BRAIN Fast Expressions.

Grammar (informal)::

    program   := stmt (';' stmt)* ';'?          -- the last statement is the alpha
    stmt      := IDENT '=' expr | expr
    expr      := ternary
    ternary   := or ('?' expr ':' ternary)?
    binary    := || < && < (== !=) < (< <= > >=) < (+ -) < (* /)   (left associative)
    unary     := ('-' | '!' | '+') unary | postfix
    primary   := NUMBER | STRING | true | false | IDENT | IDENT '(' args ')' | '(' expr ')'
    args      := (arg (',' arg)*)?
    arg       := IDENT '=' expr | expr
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ast import (RArg, RAssign, RBinary, RBool, RCall, RIdent, RNode, RNum, RProgram, RStr, RTernary,
                  RUnary)


class ParseError(Exception):
    def __init__(self, message: str, start: int, end: int):
        super().__init__(message)
        self.message = message
        self.start = start
        self.end = max(end, start + 1)


@dataclass(slots=True)
class Tok:
    kind: str  # NUM STR IDENT OP LP RP COMMA SEMI EOF
    value: str
    start: int
    end: int


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<str>"[^"\n]*"|'[^'\n]*')
  | (?P<ident>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<op>&&|\|\||==|!=|<=|>=|[-+*/<>!?:=^])
  | (?P<lp>\()
  | (?P<rp>\))
  | (?P<comma>,)
  | (?P<semi>;)
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[Tok]:
    toks: list[Tok] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if not m:
            ch = text[pos]
            if ch in "\"'":
                raise ParseError("Unterminated string literal", pos, n)
            raise ParseError(f"Unexpected character '{ch}'", pos, pos + 1)
        kind = m.lastgroup
        s, e = m.start(), m.end()
        if kind == "ws":
            pass
        elif kind == "num":
            toks.append(Tok("NUM", m.group(), s, e))
        elif kind == "str":
            toks.append(Tok("STR", m.group()[1:-1], s, e))
        elif kind == "ident":
            ident = m.group()
            if ident.endswith("."):
                raise ParseError(f"Invalid identifier '{ident}'", s, e)
            toks.append(Tok("IDENT", ident, s, e))
        elif kind == "op":
            if m.group() == "^":
                raise ParseError("'^' is not supported; use power(x, y) or signed_power(x, y)", s, e)
            toks.append(Tok("OP", m.group(), s, e))
        elif kind == "lp":
            toks.append(Tok("LP", "(", s, e))
        elif kind == "rp":
            toks.append(Tok("RP", ")", s, e))
        elif kind == "comma":
            toks.append(Tok("COMMA", ",", s, e))
        elif kind == "semi":
            toks.append(Tok("SEMI", ";", s, e))
        pos = e
    toks.append(Tok("EOF", "", n, n))
    return toks


BINARY_PREC = {
    "||": 2, "&&": 3,
    "==": 4, "!=": 4,
    "<": 5, "<=": 5, ">": 5, ">=": 5,
    "+": 6, "-": 6,
    "*": 7, "/": 7,
}
TERNARY_PREC = 1
UNARY_PREC = 8


class Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0

    # -- token helpers
    def peek(self, k: int = 0) -> Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i = min(self.i + 1, len(self.toks) - 1)
        return t

    def expect(self, kind: str, value: str | None = None, what: str = "") -> Tok:
        t = self.peek()
        if t.kind != kind or (value is not None and t.value != value):
            exp = what or (value or kind)
            got = "end of input" if t.kind == "EOF" else f"'{t.value}'"
            raise ParseError(f"Expected {exp} but found {got}", t.start, t.end)
        return self.next()

    # -- program
    def parse_program(self) -> RProgram:
        if self.peek().kind == "EOF":
            raise ParseError("Empty expression", 0, 1)
        assigns: list[RAssign] = []
        result: RNode | None = None
        while True:
            t = self.peek()
            if t.kind == "EOF":
                break
            if t.kind == "IDENT" and self.peek(1).kind == "OP" and self.peek(1).value == "=":
                name_tok = self.next()
                self.next()  # '='
                value = self.parse_expr(0)
                assigns.append(RAssign(name_tok.start, value.end, name_tok.value, value))
                result = None
            else:
                result = self.parse_expr(0)
            t = self.peek()
            if t.kind == "SEMI":
                self.next()
                if result is not None and self.peek().kind != "EOF":
                    raise ParseError("Only the last statement may be a bare expression; "
                                     "earlier statements must be assignments (name = expr)", result.start, result.end)
                continue
            if t.kind != "EOF":
                got = f"'{t.value}'"
                raise ParseError(f"Unexpected {got}; separate statements with ';'", t.start, t.end)
        if result is None:
            end = len(self.text)
            raise ParseError("The last statement must be an expression (the alpha), not an assignment",
                             max(0, end - 1), end)
        return RProgram(0, len(self.text), tuple(assigns), result)

    # -- expressions
    def parse_expr(self, min_prec: int) -> RNode:
        left = self.parse_unary()
        while True:
            t = self.peek()
            if t.kind != "OP":
                break
            if t.value == "?":
                if TERNARY_PREC < min_prec:
                    break
                self.next()
                a = self.parse_expr(0)
                self.expect("OP", ":", "':' of the conditional (c ? a : b)")
                b = self.parse_expr(TERNARY_PREC)
                left = RTernary(left.start, b.end, left, a, b)
                continue
            prec = BINARY_PREC.get(t.value)
            if prec is None or prec < min_prec:
                break
            self.next()
            right = self.parse_expr(prec + 1)
            left = RBinary(left.start, right.end, t.value, left, right)
        return left

    def parse_unary(self) -> RNode:
        t = self.peek()
        if t.kind == "OP" and t.value in ("-", "!", "+"):
            self.next()
            operand = self.parse_expr(UNARY_PREC)
            if t.value == "+":
                return operand
            if t.value == "-" and isinstance(operand, RNum):
                return RNum(t.start, operand.end, -operand.value, "-" + operand.text)
            return RUnary(t.start, operand.end, t.value, operand)
        return self.parse_primary()

    def parse_primary(self) -> RNode:
        t = self.next()
        if t.kind == "NUM":
            return RNum(t.start, t.end, float(t.value), t.value)
        if t.kind == "STR":
            return RStr(t.start, t.end, t.value)
        if t.kind == "LP":
            inner = self.parse_expr(0)
            self.expect("RP", what="')'")
            return inner
        if t.kind == "IDENT":
            if self.peek().kind == "LP":
                return self.parse_call(t)
            low = t.value.lower()
            if low in ("true", "false"):
                return RBool(t.start, t.end, low == "true")
            return RIdent(t.start, t.end, t.value)
        if t.kind == "EOF":
            raise ParseError("Unexpected end of input", t.start, t.end)
        raise ParseError(f"Unexpected '{t.value}'", t.start, t.end)

    def parse_call(self, name_tok: Tok) -> RCall:
        self.expect("LP")
        args: list[RArg] = []
        if self.peek().kind != "RP":
            while True:
                t = self.peek()
                if t.kind == "IDENT" and self.peek(1).kind == "OP" and self.peek(1).value == "=":
                    self.next()
                    self.next()
                    value = self.parse_expr(0)
                    args.append(RArg(t.start, value.end, t.value, value))
                else:
                    value = self.parse_expr(0)
                    args.append(RArg(value.start, value.end, None, value))
                if self.peek().kind == "COMMA":
                    self.next()
                    continue
                break
        rp = self.expect("RP", what=f"')' to close {name_tok.value}(")
        return RCall(name_tok.start, rp.end, name_tok.value, name_tok.end, tuple(args))


def parse(text: str) -> RProgram:
    return Parser(text).parse_program()
