"""Idea Forge input understanding: read an idea in whatever form it arrives and extract what can be built.

Accepted forms (any mix of them):

* plain English (the lexicon in ``gen/idea.py`` reads it);
* Fast Expressions, single-line or multi-statement, in code fences or not;
* formulas in the *101 Formulaic Alphas* / academic notation (``correlation``, ``delta``, ``Ts_Rank``,
  ``IndNeutralize``, ``SignedPower``, ``x^y``, ``adv20``...), translated to BRAIN operators;
* pandas / numpy code (``df['close'].pct_change(252).shift(21)``, ``.rolling(n).mean()``,
  ``.rank(axis=1)``, ``np.where``...), translated through Python's ``ast`` without executing anything;
* described computations and written formulas ("12-month return skipping the last month",
  "EBITDA to enterprise value ranked within industry", "Signal = rank(EBITDA / EV)");
* JSON / YAML idea records, bulleted or numbered lists of several ideas;
* long documents (papers, notes, web pages): the key sentences are extracted first.

Everything is deterministic and offline. Every expression returned has passed the Fast Expression analyzer.
"""

from __future__ import annotations

import ast
import html as _html
import io
import json
import math
import re
import textwrap
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache

from ..catalog import field_map
from ..fastexpr import analyze, lower_text, to_expr

MAX_DOC_CHARS = 400_000


@dataclass
class IdeaInput:
    format: str = "english"
    text: str = ""
    seeds: list[str] = field(default_factory=list)
    compiled: list[tuple[str, str]] = field(default_factory=list)
    sub_ideas: list[str] = field(default_factory=list)
    key_sentences: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"format": self.format, "formats": self.formats, "text": self.text, "seeds": self.seeds,
                "compiled": [{"label": lab, "expr": e} for lab, e in self.compiled], "sub_ideas": self.sub_ideas,
                "key_sentences": self.key_sentences, "settings": self.settings, "notes": self.notes,
                "warnings": self.warnings}


def _valid(expr: str, local_fields: set[str] | None = None) -> str | None:
    """The expression in clean printed form if the analyzer accepts it (programs keep their statements)."""
    e = expr.strip().strip(";").strip()
    if not e:
        return None
    an = analyze(e, local_fields=local_fields)
    if not an.ok or an.node is None:
        return None
    if ";" in e:
        return "\n".join(s.strip() + ";" for s in e.split(";")[:-1] if s.strip()) + "\n" + e.split(";")[-1].strip()
    try:
        return to_expr(lower_text(e))
    except Exception:  # noqa: BLE001
        return e


# =========================================================================== field vocabulary

FUNDAMENTAL_CATS = {"fundamental"}
WORD_FIELDS: list[tuple[str, str]] = [
    # multi-word first (longest match wins)
    (r"free cash ?flows?|\bfcf\b", "(cashflow_op - capex)"),
    (r"operating cash ?flows?|cash ?flows? from operations|\bcfo\b|\bocf\b", "cashflow_op"),
    (r"cash ?flows?", "cashflow_op"),
    (r"enterprise value|\bev\b", "enterprise_value"),
    (r"market cap(?:itali[sz]ation)?|market value|\bmcap\b|\bmarket_cap\b|\bsize\b", "cap"),
    (r"net income|net earnings|\bearnings\b|\bprofits?\b|bottom line", "income"),
    (r"operating income|operating profits?|\bebit\b", "operating_income"),
    (r"\bebitda\b", "ebitda"),
    (r"\bsales\b|\brevenues?\b|top line|turnover of the company", "sales"),
    (r"cost of (?:goods|sales|revenue)(?: sold)?|\bcogs\b", "cogs"),
    (r"book value(?: of equity)?|shareholders'? equity|\bbook\b|\bequity\b", "equity"),
    (r"total assets|\bassets\b", "assets"),
    (r"total liabilities|\bliabilities\b", "liabilities"),
    (r"total debt|\bdebt\b|\bborrowings?\b", "debt"),
    (r"\bcash\b(?! ?flow)", "cash"),
    (r"capital expenditures?|\bcapex\b", "capex"),
    (r"research and development|\br&d\b", "rd_expense"),
    (r"\bsg&a\b|selling,? general and administrative", "sga_expense"),
    (r"earnings per share|\beps\b", "eps"),
    (r"dividends?(?: per share)?", "dividend"),
    (r"shares outstanding|share count|\bsharesout\b", "sharesout"),
    (r"inventor(?:y|ies)", "inventory"),
    (r"receivables?", "receivable"),
    (r"\broe\b|return on equity", "return_equity"),
    (r"\broa\b|return on assets", "return_assets"),
    (r"current ratio", "current_ratio"),
    (r"average daily volume|\badv\b|\badv20\b", "adv20"),
    (r"\bvwap\b|volume[- ]weighted (?:average )?price", "vwap"),
    (r"trading volume|\bvolumes?\b|shares traded", "volume"),
    (r"\bopen(?:ing)?(?: price)?\b", "open"),
    (r"\bhighs?\b|intraday high|daily high", "high"),
    (r"\blows?\b|intraday low|daily low", "low"),
    (r"\breturns?\b", "returns"),
    (r"closing prices?|\bclose\b|\bprices?\b|share price|stock price", "close"),
]
_WORD_FIELD_RE = [(re.compile(p), f) for p, f in WORD_FIELDS]

NAMED: list[tuple[str, str, str]] = [
    # (pattern, expression, label) - named ratios; fundamentals are back-filled when emitted
    (r"earnings[- ]to[- ]price|\be/p\b|earnings yield", "{income} / cap", "earnings yield"),
    (r"book[- ]to[- ]market|book[- ]to[- ]price|\bb/m\b|\bb/p\b", "{equity} / cap", "book-to-market"),
    (r"low (?:price[- ]to[- ]earnings|p/e|pe)(?: ratios?)?", "{eps} / close", "earnings-to-price (low P/E)"),
    (r"price[- ]to[- ]earnings|\bp/e\b|\bpe ratio\b|\bp/e ratio\b", "close / {eps}", "price-to-earnings"),
    (r"price[- ]to[- ]book|\bp/b\b", "cap / {equity}", "price-to-book"),
    (r"price[- ]to[- ]sales|\bp/s\b", "cap / {sales}", "price-to-sales"),
    (r"sales[- ]to[- ]price|sales yield", "{sales} / cap", "sales yield"),
    (r"ev ?/ ?ebitda|ev[- ]to[- ]ebitda|enterprise multiple", "enterprise_value / {ebitda}", "EV/EBITDA"),
    (r"ebitda[- ]to[- ]ev|ebitda ?/ ?ev|ebitda yield", "{ebitda} / enterprise_value", "EBITDA/EV"),
    (r"free cash ?flow yield|\bfcf yield\b", "({cashflow_op} - {capex}) / cap", "free-cash-flow yield"),
    (r"cash ?flow yield|cfo yield", "{cashflow_op} / cap", "cash-flow yield"),
    (r"dividend yield", "ts_sum(dividend, 252) / close", "dividend yield"),
    (r"gross (?:profit )?margin", "({sales} - {cogs}) / {sales}", "gross margin"),
    (r"gross profitability|gross profits?[- ]to[- ]assets", "({sales} - {cogs}) / {assets}", "gross profitability"),
    (r"operating margin", "{operating_income} / {sales}", "operating margin"),
    (r"net margin|profit margin", "{income} / {sales}", "net margin"),
    (r"debt[- ]to[- ]equity|\bd/e\b", "{debt} / {equity}", "debt-to-equity"),
    (r"debt[- ]to[- ]assets|leverage ratio", "{debt} / {assets}", "debt-to-assets"),
    (r"asset growth", "ts_delta({assets}, 252) / ts_delay({assets}, 252)", "asset growth"),
    (r"(?:sales|revenue) growth", "ts_delta({sales}, 252) / ts_delay({sales}, 252)", "sales growth"),
    (r"earnings growth", "ts_delta({income}, 252) / abs(ts_delay({income}, 252))", "earnings growth"),
    (r"total accruals|\baccruals\b", "({income} - {cashflow_op}) / {assets}", "accruals"),
    (r"return on equity|\broe\b", "{income} / {equity}", "return on equity"),
    (r"return on assets|\broa\b", "{income} / {assets}", "return on assets"),
    (r"cash[- ]to[- ]assets", "{cash} / {assets}", "cash-to-assets"),
    (r"share turnover", "volume / sharesout", "share turnover"),
    (r"amihud|illiquidity", "ts_mean(abs(returns) / (volume * close), 20)", "Amihud illiquidity"),
    (r"idiosyncratic volatility", "ts_std_dev(returns - group_mean(returns, 1, industry), 60)",
     "idiosyncratic volatility"),
    (r"intraday (?:return|move)", "(close - open) / open", "intraday return"),
    (r"overnight (?:return|gap)", "open / ts_delay(close, 1) - 1", "overnight return"),
    (r"distance (?:from|to) (?:the )?52[- ]?week high|close(?:ness)? to (?:the )?52[- ]?week high|"
     r"52[- ]?week high ratio", "close / ts_max(high, 252)", "closeness to the 52-week high"),
    (r"52[- ]?week high", "ts_max(high, 252)", "52-week high"),
    (r"52[- ]?week low", "ts_min(low, 252)", "52-week low"),
    (r"abnormal volume|volume surprise|relative volume", "volume / adv20", "abnormal volume"),
]
_NAMED_RE = [(re.compile(p), e, lab) for p, e, lab in NAMED]

UNIT_DAYS = {"day": 1, "days": 1, "trading day": 1, "trading days": 1, "d": 1, "week": 5, "weeks": 5, "w": 5,
             "month": 21, "months": 21, "m": 21, "mo": 21, "quarter": 63, "quarters": 63, "q": 63, "year": 252,
             "years": 252, "y": 252, "yr": 252, "yrs": 252}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60,
             "a": 1, "an": 1}
_PERIOD = r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|sixty|a|an)" \
          r"[- ]?(trading[- ]days?|days?|weeks?|months?|quarters?|years?|yrs?|d|w|mo|m|q|y)\b"


def period_days(n: str, unit: str) -> int:
    k = NUM_WORDS.get(n.lower()) if not n.isdigit() else int(n)
    d = int(k or 1) * UNIT_DAYS.get(unit.lower().replace("-", " "), UNIT_DAYS.get(unit.lower().rstrip("s"), 1))
    return {260: 252, 250: 252, 126: 126, 105: 105}.get(d, d)


NO_BACKFILL = {"enterprise_value", "cap"}  # recomputed every day from the price


def _fund(f: str) -> str:
    cat = str(field_map().get(f, {}).get("category", ""))
    return f"ts_backfill({f}, 120)" if cat in FUNDAMENTAL_CATS and f not in NO_BACKFILL else f


def _fill(template: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: _fund(m.group(1)), template)


def field_of(phrase: str) -> str | None:
    """A single data field (or small field expression) named by a noun phrase."""
    p = phrase.strip().lower()
    fm = field_map()
    tok = re.sub(r"[^a-z0-9_]", "", p)
    if tok in fm and str(fm[tok].get("type", "MATRIX")).upper() == "MATRIX":
        return _fund(tok)
    for rx, f in _WORD_FIELD_RE:
        if rx.search(p):
            if f.startswith("("):  # a small field expression such as free cash flow
                return re.sub(r"[a-z_]+", lambda m: _fund(m.group(0)), f)
            return _fund(f)
    return None


# =========================================================================== English formula compiler

_FILLER = re.compile(r"^(?:the|a|an|its|their|each stock'?s|a stock'?s|stock'?s|company'?s|firm'?s)\s+")
_GROUPS = {"sub-industry": "subindustry", "subindustry": "subindustry", "sub industry": "subindustry",
           "industry": "industry", "sector": "sector", "market": "market", "peers": "industry",
           "peer group": "industry"}


def _clean(s: str) -> str:
    s = s.strip().strip(".,;:!?\"'()[]").strip()
    while True:
        t = _FILLER.sub("", s)
        if t == s:
            return s
        s = t


def compile_one(phrase: str, depth: int = 0) -> tuple[str, str] | None:
    """(expression, label) for one described computation, or None (memoized: the grammar is recursive)."""
    p = _clean(phrase.lower())
    if not p or depth > 6 or len(p.split()) > 24:
        return None
    return _compile_cached(p, min(depth, 6))


@lru_cache(maxsize=8192)
def _compile_cached(p: str, depth: int) -> tuple[str, str] | None:
    # "gross profitability is (defined as) revenue minus ..." - compile the definition
    m = re.match(r"^([\w /&'-]{2,40}?) (?:is|are|equals|is defined as|is computed as|is measured as|is calculated as"
                 r"|defined as|computed as|measured as)\s+(.+)$", p)
    if m and len(m.group(1).split()) <= 4:
        return compile_one(m.group(2), depth + 1)
    for rx, tmpl, lab in _NAMED_RE:  # an exact named ratio ("low P/E") wins over the generic rules below
        if rx.fullmatch(p):
            return _fill(tmpl), lab
    # --- cross-sectional wrappers
    m = re.match(r"^(?:rank of |ranking of |percentile of )?(.+?),? ranked (?:with)?in (?:its |their |the )?"
                 r"(sub[- ]?industry|industry|sector|market|peers|peer group)\b", p) or \
        re.match(r"^(?:rank|percentile) of (.+?) (?:with)?in (?:its |their |the )?"
                 r"(sub[- ]?industry|industry|sector|market|peers|peer group)\b", p)
    if m:
        inner = compile_one(m.group(1), depth + 1)
        g = _GROUPS.get(m.group(2).replace("-", " ") if "sub" in m.group(2) else m.group(2), "industry")
        g = "subindustry" if "sub" in m.group(2) else g
        if inner:
            e = f"rank({inner[0]})" if g == "market" else f"group_rank({inner[0]}, {g})"
            return e, f"{inner[1]}, ranked within {g}"
    m = re.match(r"^(?:cross[- ]sectional )?(?:rank|percentile|ranking) of (.+)$", p) or \
        re.match(r"^(.+?),? (?:cross[- ]sectionally )?ranked$", p)
    if m:
        inner = compile_one(m.group(1), depth + 1)
        if inner:
            return f"rank({inner[0]})", f"rank of {inner[1]}"
    m = re.match(r"^cross[- ]sectional z[- ]?score of (.+)$", p)
    if m:
        inner = compile_one(m.group(1), depth + 1)
        if inner:
            return f"zscore({inner[0]})", f"cross-sectional z-score of {inner[1]}"
    # --- sign words
    m = re.match(r"^(?:negative of|minus|inverse of|the opposite of|short) (.+)$", p) or \
        re.match(r"^(?:low|lower|lowest|cheap|small|smaller|smallest) (.+)$", p)
    if m:
        inner = compile_one(m.group(1), depth + 1)
        if inner:
            return f"-({inner[0]})" if re.search(r"[-+*/ ]", inner[0]) else f"-{inner[0]}", f"low {inner[1]}"
    m = re.match(r"^(?:high|higher|highest|large|larger|strong) (.+)$", p)
    if m:
        inner = compile_one(m.group(1), depth + 1)
        if inner:
            return inner
    # --- returns / momentum over a horizon, optionally skipping the most recent period
    m = re.match(rf"^(?:(?:{_PERIOD}) )?(?:total |cumulative |past |trailing |price )?(?:returns?|performance|momentum)"
                 rf"(?: over| during| in)?(?: the)?(?: past| last| previous| prior| trailing)?(?: {_PERIOD})?"
                 rf"(?:,? (?:skipping|excluding|ignoring|except|lagged by|without) (?:the )?(?:most recent |last )?"
                 rf"(?:{_PERIOD}|month|week))?$", p)
    if m and (m.group(1) or m.group(3)):
        n = period_days(m.group(1), m.group(2)) if m.group(1) else period_days(m.group(3), m.group(4))
        skip = 0
        if m.group(5):
            skip = period_days(m.group(5), m.group(6))
        elif re.search(r"(?:skipping|excluding|ignoring|except|without) (?:the )?(?:most recent |last )?month", p):
            skip = 21
        elif re.search(r"(?:skipping|excluding|ignoring|except|without) (?:the )?(?:most recent |last )?week", p):
            skip = 5
        if skip and n > skip:
            return f"ts_delay(ts_sum(returns, {n - skip}), {skip})", f"{n}-day return skipping the last {skip} days"
        return f"ts_sum(returns, {n})", f"{n}-day return"
    # --- time-series operators with a horizon
    ts_ops = [
        (r"(?:moving )?average|mean|sma", "ts_mean", "average"),
        (r"standard deviation|std(?:ev)?|volatility", "ts_std_dev", "volatility"),
        (r"(?:percent(?:age)? )?change|difference|delta", "ts_delta", "change"),
        (r"(?:time[- ]series )?z[- ]?score", "ts_zscore", "z-score"),
        (r"(?:time[- ]series )?rank|percentile", "ts_rank", "rank in own history"),
        (r"maximum|max|highest(?: value)?", "ts_max", "maximum"),
        (r"minimum|min|lowest(?: value)?", "ts_min", "minimum"),
        (r"sum|total", "ts_sum", "sum"),
        (r"skew(?:ness)?", "ts_skewness", "skewness"),
    ]
    for word, op, lab in ts_ops:
        m = re.match(rf"^{_PERIOD} (?:{word}) (?:of|in) (.+)$", p) or \
            re.match(rf"^(?:{word}) (?:of|in) (.+?) (?:over|during|in) (?:the )?(?:past |last |previous |trailing )?{_PERIOD}$", p)
        if not m:
            m2 = re.match(rf"^(.+?)'?s? {_PERIOD} (?:{word})$", p)
            if m2:
                inner_txt, n = m2.group(1), period_days(m2.group(2), m2.group(3))
            else:
                continue
        else:
            if m.re.pattern.startswith("^" + _PERIOD):
                n, inner_txt = period_days(m.group(1), m.group(2)), m.group(3)
            else:
                inner_txt, n = m.group(1), period_days(m.group(2), m.group(3))
        if op == "ts_std_dev" and re.fullmatch(r"(?:daily )?(?:stock )?returns?|prices?|the stock", inner_txt.strip()):
            return f"ts_std_dev(returns, {max(n, 5)})", f"{n}-day volatility"
        inner = compile_one(inner_txt, depth + 1)
        if not inner:
            continue
        if op == "ts_delta" and "percent" in p:
            return f"{inner[0]} / ts_delay({inner[0]}, {n}) - 1", f"{n}-day percent change in {inner[1]}"
        return f"{op}({inner[0]}, {max(n, 2)})", f"{n}-day {lab} of {inner[1]}"
    m = re.match(rf"^{_PERIOD} volatility$", p) or re.match(rf"^volatility (?:over|of) (?:the )?(?:past |last )?{_PERIOD}$", p)
    if m:
        n = period_days(m.group(1), m.group(2))
        return f"ts_std_dev(returns, {max(n, 5)})", f"{n}-day volatility"
    m = re.match(rf"^(?:the )?correlation (?:between|of) (.+?) (?:and|with) (.+?)(?: over (?:the )?(?:past |last )?{_PERIOD})?$", p)
    if m:
        a, b = compile_one(m.group(1), depth + 1), compile_one(m.group(2), depth + 1)
        if a and b:
            n = period_days(m.group(3), m.group(4)) if m.group(3) else 20
            return f"ts_corr({a[0]}, {b[0]}, {max(n, 5)})", f"{n}-day correlation of {a[1]} and {b[1]}"
    # --- named ratios
    for rx, tmpl, lab in _NAMED_RE:
        if rx.fullmatch(p) or (rx.search(p) and len(p.split()) <= 4):
            return _fill(tmpl), lab
    # --- arithmetic between two phrases
    for pat, op, word in ((r"^ratio of (.+?) to (.+)$", "/", "to"), (r"^(.+?) divided by (.+)$", "/", "/"),
                          (r"^(.+?) (?:per|relative to|scaled by|over) (.+)$", "/", "/"),
                          (r"^(.+?)[- ]to[- ](.+?)(?: ratio)?$", "/", "to"), (r"^(.+?) / (.+)$", "/", "/"),
                          (r"^(.+?) minus (.+)$", "-", "minus"), (r"^(.+?) less (.+)$", "-", "minus"),
                          (r"^(.+?) plus (.+)$", "+", "plus"), (r"^(.+?) times (.+)$", "*", "times")):
        m = re.match(pat, p)
        if m and len(m.group(1).split()) <= 8 and len(m.group(2).split()) <= 8:
            a, b = compile_one(m.group(1), depth + 1), compile_one(m.group(2), depth + 1)
            if a and b:
                return f"{_paren(a[0])} {op} {_paren(b[0])}", f"{a[1]} {word} {b[1]}"
    # --- a single field
    f = field_of(p)
    if f and len(p.split()) <= 5:
        return f, p
    return None


def _paren(e: str) -> str:
    return f"({e})" if re.search(r"\s[-+]\s", e) and not (e.startswith("(") and e.endswith(")")) else e


def compile_phrase(text: str, local_fields: set[str] | None = None, limit: int = 8) -> list[tuple[str, str]]:
    """(label, expression) for the computations a text describes. Sentences are split into clauses and each
    clause is compiled on its own; only analyzer-valid expressions are returned."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    clauses: list[str] = []
    for line in re.split(r"[\n;]+", text):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*[\w .\-/()&']{1,40}?\s*(?:=|:=|:)\s*(.+)$", line)
        if m and not re.match(r"^\s*https?:", line):
            clauses.append(m.group(1))  # a definition: only its right-hand side describes the computation
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            clauses += [c for c in re.split(r",\s+(?:and |or |then |while |but )|\s+(?:and then|whereas|while)\s+",
                                            sent) if c]
    for c in clauses:
        c = c.strip()
        if not c or len(c) > 160:
            continue
        cand = _try_expression_like(c)
        res = cand or compile_one(c)
        if not res:
            # "long stocks with high X" / "buy X" / "go long X" / "X predicts returns": compile the object
            m = re.search(r"(?:long|buy|overweight|favou?r|prefer|rank(?:ed)? by|sort(?:ed)? by|based on|using)\s+"
                          r"(?:stocks|companies|firms|names)?\s*(?:with |that have |having )?(.+)$", c.lower())
            if m:
                res = compile_one(m.group(1))
        if not res:
            continue
        e, lab = res
        v = _valid(e, local_fields)
        if v and re.fullmatch(r"(?:ts_backfill\()?[a-z_0-9]+(?:, \d+\))?", v):
            continue  # a bare field is not a computation; the mechanism lexicon handles plain data mentions
        if v and v not in seen and len(v) < 400:
            seen.add(v)
            out.append((lab, v))
        if len(out) >= limit:
            break
    return out


def _try_expression_like(rhs: str) -> tuple[str, str] | None:
    """Formula text with words for data, e.g. 'rank(EBITDA / EV)' or 'net income / total assets'."""
    if not re.search(r"[()/*+\-]", rhs):
        return None
    s = rhs.strip().rstrip(".")
    placeholders: list[str] = []

    def sub(m: re.Match) -> str:
        f = field_of(m.group(0))
        if f is None:
            return m.group(0)
        placeholders.append(f)
        return f"__F{len(placeholders) - 1}__"
    lowered = s.lower()
    for rx, tmpl, _lab in _NAMED_RE:
        if rx.search(lowered):
            placeholders.append(_fill(tmpl))
            lowered = rx.sub(f"__F{len(placeholders) - 1}__", lowered)
    for rx, _f in _WORD_FIELD_RE:
        lowered = rx.sub(sub, lowered)
    expr = re.sub(r"__F(\d+)__", lambda m: f"({placeholders[int(m.group(1))]})"
                  if re.search(r"[-+*/ ]", placeholders[int(m.group(1))]) else placeholders[int(m.group(1))], lowered)
    expr = re.sub(r"\bln\(", "log(", expr)
    if not placeholders or not analyze(expr).ok:
        return None
    return expr, f"formula: {rhs.strip()[:80]}"


# =========================================================================== 101-Alphas / paper notation

PAPER_FUNCS = {"delay": "ts_delay", "delta": "ts_delta", "correlation": "ts_corr", "corr": "ts_corr",
               "covariance": "ts_covariance", "cov": "ts_covariance", "ts_rank": "ts_rank", "tsrank": "ts_rank",
               "ts_argmax": "ts_arg_max", "ts_argmin": "ts_arg_min", "ts_min": "ts_min", "ts_max": "ts_max",
               "sum": "ts_sum", "product": "ts_product", "stddev": "ts_std_dev", "std": "ts_std_dev",
               "decay_linear": "ts_decay_linear", "decaylinear": "ts_decay_linear", "signedpower": "signed_power",
               "scale": "scale", "indneutralize": "group_neutralize", "rank": "rank", "log": "log", "abs": "abs",
               "sign": "sign", "exp": "exp", "sqrt": "sqrt", "ts_mean": "ts_mean", "mean": "ts_mean",
               "ts_sum": "ts_sum", "ts_stddev": "ts_std_dev", "ts_delta": "ts_delta", "ts_delay": "ts_delay",
               "ts_corr": "ts_corr", "ts_covariance": "ts_covariance", "ts_decay_linear": "ts_decay_linear",
               "signed_power": "signed_power", "ts_arg_max": "ts_arg_max", "ts_arg_min": "ts_arg_min",
               "power": "power", "if_else": "if_else", "min": "min", "max": "max", "ts_product": "ts_product"}
PAPER_WINDOWED = {"ts_delay", "ts_delta", "ts_corr", "ts_covariance", "ts_rank", "ts_arg_max", "ts_arg_min", "ts_min",
                  "ts_max", "ts_sum", "ts_product", "ts_std_dev", "ts_decay_linear", "ts_mean"}
PAPER_MARKERS = re.compile(r"\b(?:correlation|covariance|delta|delay|Ts_Rank|Ts_ArgMax|Ts_ArgMin|IndNeutralize|"
                           r"SignedPower|decay_linear|stddev|IndClass|adv\d+)\b|\^", re.I)
_PTOK = re.compile(r"\s*(?:(?P<num>\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)|(?P<id>[A-Za-z_][A-Za-z0-9_.]*)|"
                   r"(?P<op>\|\||&&|==|!=|<=|>=|[-+*/^<>?:(),!]))")


class _PaperParser:
    """Recursive-descent parser for paper notation that re-emits BRAIN Fast Expression text."""

    PREC = {"?": 1, "||": 2, "&&": 3, "==": 4, "!=": 4, "<": 5, "<=": 5, ">": 5, ">=": 5, "+": 6, "-": 6, "*": 7,
            "/": 7, "^": 9}

    def __init__(self, text: str):
        self.toks: list[tuple[str, str]] = []
        pos = 0
        text = text.strip()
        while pos < len(text):
            m = _PTOK.match(text, pos)
            if not m or m.end() == pos:
                if text[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"unexpected {text[pos]!r}")
            kind = m.lastgroup or "op"
            self.toks.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, val: str | None = None) -> tuple[str, str]:
        t = self.peek()
        if t is None or (val is not None and t[1] != val):
            raise ValueError(f"expected {val}")
        self.i += 1
        return t

    def expr(self, min_prec: int = 0) -> str:
        left = self.unary()
        while True:
            t = self.peek()
            if t is None or t[0] != "op" or t[1] not in self.PREC or self.PREC[t[1]] < min_prec:
                return left
            op = t[1]
            self.i += 1
            if op == "?":
                a = self.expr(0)
                self.take(":")
                b = self.expr(1)
                left = f"(({left}) ? ({a}) : ({b}))"
                continue
            right = self.expr(self.PREC[op] + (0 if op == "^" else 1))
            left = f"power({left}, {right})" if op == "^" else f"({left} {op} {right})"

    def unary(self) -> str:
        t = self.peek()
        if t and t[0] == "op" and t[1] in ("-", "+", "!"):
            self.i += 1
            operand = self.expr(8)
            return operand if t[1] == "+" else f"({t[1]}{operand})"
        return self.primary()

    def primary(self) -> str:
        kind, val = self.take()
        if kind == "num":
            return val
        if kind == "op" and val == "(":
            e = self.expr(0)
            self.take(")")
            return f"({e})"
        if kind == "id":
            nxt = self.peek()
            if nxt and nxt[1] == "(":
                self.take("(")
                args: list[str] = []
                if not (self.peek() and self.peek()[1] == ")"):  # type: ignore[index]
                    while True:
                        args.append(self.expr(0))
                        if self.peek() and self.peek()[1] == ",":  # type: ignore[index]
                            self.i += 1
                            continue
                        break
                self.take(")")
                return self.call(val, args)
            return self.ident(val)
        raise ValueError(f"unexpected {val}")

    @staticmethod
    def ident(name: str) -> str:
        low = name.lower()
        if low.startswith("indclass."):
            return {"subindustry": "subindustry", "industry": "industry", "sector": "sector"}.get(low[9:], "industry")
        m = re.fullmatch(r"adv(\d+)", low)
        if m:
            return "adv20" if m.group(1) == "20" else f"ts_mean(volume, {int(m.group(1))})"
        return {"cap": "cap", "returns": "returns", "vwap": "vwap", "volume": "volume", "open": "open",
                "close": "close", "high": "high", "low": "low"}.get(low, name)

    @staticmethod
    def _int(a: str) -> str:
        s = a.strip("() ")
        try:
            return str(max(1, int(round(float(s)))))
        except ValueError:
            return a

    def call(self, name: str, args: list[str]) -> str:
        low = name.lower()
        if low in ("min", "max") and len(args) == 2 and re.fullmatch(r"\(?\s*\d+\.?\d*\s*\)?", args[1]):
            return f"ts_{low}({args[0]}, {self._int(args[1])})"
        fn = PAPER_FUNCS.get(low, low)
        if fn in PAPER_WINDOWED and len(args) >= 2:
            args = args[:-1] + [self._int(args[-1])]
        if fn == "group_neutralize" and len(args) == 2:
            args = [args[0], args[1].strip("() ")]
        if fn == "scale" and len(args) == 2:
            return f"scale({args[0]}, scale={args[1]})"
        return f"{fn}({', '.join(args)})"


def translate_paper(expr: str) -> str | None:
    """101-Alphas / academic notation -> Fast Expression (None when it cannot be translated safely)."""
    s = expr.strip().rstrip(".;")
    s = re.sub(r"^\s*(?:alpha\s*#?\s*\d+\s*[:=]\s*)", "", s, flags=re.I)
    try:
        p = _PaperParser(s)
        out = p.expr(0)
        if p.peek() is not None:
            return None
    except (ValueError, IndexError, RecursionError):
        return None
    return _valid(out)


# =========================================================================== pandas / numpy code

PY_FIELDS = {"close": "close", "adj_close": "close", "adjclose": "close", "adj close": "close", "price": "close",
             "prices": "close", "px": "close", "px_last": "close", "last": "close", "open": "open", "high": "high",
             "low": "low", "volume": "volume", "vwap": "vwap", "returns": "returns", "ret": "returns",
             "rets": "returns", "return": "returns", "daily_returns": "returns", "market_cap": "cap", "mcap": "cap",
             "marketcap": "cap", "cap": "cap", "adv20": "adv20", "shares": "sharesout", "sharesout": "sharesout",
             "shares_outstanding": "sharesout", "sector": "sector", "industry": "industry",
             "subindustry": "subindustry"}
DF_NAMES = {"df", "data", "dfs", "panel", "d", "frame", "stocks", "universe", "px_df", "prices_df"}
PY_MARKERS = re.compile(r"^\s*(?:import |from \w+ import |def |return\b)|\bdf\[|\.rolling\(|\.shift\(|\.pct_change\(|"
                        r"\bnp\.|\bpd\.|\.groupby\(|\.ewm\(|\.diff\(", re.M)
ROLLING = {"mean": "ts_mean", "std": "ts_std_dev", "sum": "ts_sum", "min": "ts_min", "max": "ts_max",
           "median": "ts_median", "skew": "ts_skewness", "kurt": "ts_kurtosis", "rank": "ts_rank", "prod": "ts_product",
           "var": None}
NP_FUNCS = {"log": "log", "abs": "abs", "sign": "sign", "sqrt": "sqrt", "exp": "exp", "tanh": "tanh",
            "log1p": "s_log_1p", "fabs": "abs"}


class _PyTranslator:
    def __init__(self) -> None:
        self.vars: dict[str, str] = {}
        self.lines: list[tuple[str, str]] = []
        self.warnings: list[str] = []
        self.reserved = set(field_map()) | {"market", "sector", "industry", "subindustry"}

    def var_name(self, name: str) -> str:
        n = re.sub(r"[^A-Za-z0-9_]", "_", name)
        return f"{n}_v" if n.lower() in self.reserved or n.lower() in PAPER_FUNCS else n

    def field(self, key: str) -> str | None:
        k = key.strip().lower().replace(" ", "_")
        if k in PY_FIELDS:
            return PY_FIELDS[k]
        k2 = k.replace("_", " ")
        if k2 in PY_FIELDS:
            return PY_FIELDS[k2]
        return _fund(k) if k in field_map() else None

    @staticmethod
    def _num(n: ast.AST) -> float | None:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = _PyTranslator._num(n.operand)
            return -v if v is not None else None
        return None

    def _kw(self, call: ast.Call, name: str, pos: int | None = None) -> ast.AST | None:
        for k in call.keywords:
            if k.arg == name:
                return k.value
        if pos is not None and len(call.args) > pos:
            return call.args[pos]
        return None

    def _window(self, call: ast.Call, pos: int = 0, name: str = "window", default: int | None = None) -> int | None:
        n = self._kw(call, name, pos)
        v = self._num(n) if n is not None else None
        if v is None:
            return default
        return int(round(v))

    def tx(self, n: ast.AST) -> str | None:  # noqa: C901 - one dispatch over node kinds
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                return "1" if n.value else "0"
            if isinstance(n.value, (int, float)):
                return repr(n.value) if isinstance(n.value, float) else str(n.value)
            return None
        if isinstance(n, ast.Name):
            if n.id in self.vars:
                return self.vars[n.id]
            return self.field(n.id)
        if isinstance(n, ast.Subscript):
            base = n.value
            key = n.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if isinstance(base, ast.Name) and (base.id in DF_NAMES or base.id not in self.vars):
                    return self.field(key.value)
            return None
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and (n.value.id in DF_NAMES):
                return self.field(n.attr)
            if n.attr in ("values", "T"):
                return self.tx(n.value)
            return None
        if isinstance(n, ast.BinOp):
            zs = self._zscore_pattern(n)
            if zs:
                return zs
            a, b = self.tx(n.left), self.tx(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Pow):
                return f"power({a}, {b})"
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(n.op))
            return f"({a} {op} {b})" if op else None
        if isinstance(n, ast.UnaryOp):
            v = self.tx(n.operand)
            if v is None:
                return None
            if isinstance(n.op, ast.USub):
                return f"(-{v})"
            if isinstance(n.op, (ast.Not, ast.Invert)):
                return f"(!{v})"
            return v
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = self.tx(n.left), self.tx(n.comparators[0])
            op = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}.get(
                type(n.ops[0]))
            return f"({a} {op} {b})" if a and b and op else None
        if isinstance(n, ast.BoolOp):
            parts = [self.tx(v) for v in n.values]
            if any(p is None for p in parts):
                return None
            return "(" + (" && " if isinstance(n.op, ast.And) else " || ").join(parts) + ")"  # type: ignore[arg-type]
        if isinstance(n, ast.IfExp):
            c, a, b = self.tx(n.test), self.tx(n.body), self.tx(n.orelse)
            return f"({c} ? {a} : {b})" if c and a and b else None
        if isinstance(n, ast.Call):
            return self.call(n)
        return None

    def _zscore_pattern(self, n: ast.BinOp) -> str | None:
        """(x - x.mean(axis=1)) / x.std(axis=1) -> zscore(x)."""
        if not (isinstance(n.op, ast.Div) and isinstance(n.left, ast.BinOp) and isinstance(n.left.op, ast.Sub)):
            return None
        x, mean = n.left.left, n.left.right
        std = n.right

        def is_cs(call: ast.AST, meth: str) -> bool:
            return isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == meth \
                and ast.dump(call.func.value) == ast.dump(x)
        if is_cs(mean, "mean") and is_cs(std, "std"):
            v = self.tx(x)
            return f"zscore({v})" if v else None
        return None

    def call(self, n: ast.Call) -> str | None:  # noqa: C901
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy", "math"):
            args = [self.tx(a) for a in n.args]
            if any(a is None for a in args):
                return None
            if f.attr in NP_FUNCS and args:
                return f"{NP_FUNCS[f.attr]}({args[0]})"
            if f.attr == "where" and len(args) == 3:
                return f"({args[0]} ? {args[1]} : {args[2]})"
            if f.attr in ("maximum", "fmax") and len(args) == 2:
                return f"max({args[0]}, {args[1]})"
            if f.attr in ("minimum", "fmin") and len(args) == 2:
                return f"min({args[0]}, {args[1]})"
            if f.attr in ("power",) and len(args) == 2:
                return f"power({args[0]}, {args[1]})"
            return None
        if isinstance(f, ast.Name):
            args = [self.tx(a) for a in n.args]
            if any(a is None for a in args):
                return None
            if f.id in ("abs", "log", "sqrt", "exp", "sign") and args:
                return f"{f.id}({args[0]})"
            if f.id in ("max", "min") and len(args) == 2:
                return f"{f.id}({args[0]}, {args[1]})"
            return None
        if not isinstance(f, ast.Attribute):
            return None
        meth, recv = f.attr, f.value
        # rolling(n).<agg>() / rolling(n).corr(y)
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr in ("rolling",
                                                                                                    "expanding"):
            if recv.func.attr == "expanding":
                return None
            w = self._window(recv, 0, "window")
            x = self.tx(recv.func.value)
            if w is None or x is None:
                return None
            w = max(2, w)
            if meth in ("corr", "cov") and n.args:
                y = self.tx(n.args[0])
                return f"ts_{'corr' if meth == 'corr' else 'covariance'}({x}, {y}, {w})" if y else None
            op = ROLLING.get(meth)
            if meth == "var":
                return f"power(ts_std_dev({x}, {w}), 2)"
            return f"{op}({x}, {w})" if op else None
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr == "ewm" \
                and meth == "mean":
            x = self.tx(recv.func.value)
            span = self._num(self._kw(recv, "span") or ast.Constant(None))
            com = self._num(self._kw(recv, "com") or ast.Constant(None))
            hl = self._num(self._kw(recv, "halflife") or ast.Constant(None))
            alpha = self._num(self._kw(recv, "alpha") or ast.Constant(None))
            if span:
                alpha = 2.0 / (span + 1)
            elif com is not None:
                alpha = 1.0 / (com + 1)
            elif hl:
                alpha = 1 - math.exp(math.log(0.5) / hl)
            if x is None or not alpha:
                return None
            factor = round(1 - alpha, 4)
            d = int(min(252, max(5, round(math.log(0.01) / math.log(max(factor, 1e-6))))))
            return f"ts_decay_exp_window({x}, {d}, factor={factor})"
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr == "groupby":
            g = recv.args[0] if recv.args else self._kw(recv, "by")
            gname = None
            if isinstance(g, ast.Name):
                gname = self.field(g.id)
            elif isinstance(g, ast.Constant) and isinstance(g.value, str):
                gname = self.field(g.value)
            x = self.tx(recv.func.value)
            if gname not in ("sector", "industry", "subindustry") or x is None:
                return None
            if meth == "rank":
                return f"group_rank({x}, {gname})"
            if meth == "transform" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "mean":
                return f"group_mean({x}, 1, {gname})"
            return None
        x = self.tx(recv)
        if x is None:
            return None
        if meth == "pct_change":
            k = self._window(n, 0, "periods", 1) or 1
            return "returns" if x == "close" and k == 1 else f"({x} / ts_delay({x}, {k}) - 1)"
        if meth == "diff":
            k = self._window(n, 0, "periods", 1) or 1
            return f"ts_delta({x}, {k})"
        if meth == "shift":
            k = self._window(n, 0, "periods", 1)
            if k is None or k < 0:
                self.warnings.append("A negative shift looks into the future; it was dropped.")
                return None
            return x if k == 0 else f"ts_delay({x}, {k})"
        if meth == "rank":
            ax = self._kw(n, "axis")
            if ax is not None and self._num(ax) == 0:
                self.warnings.append("rank(axis=0) ranks through time; read as a cross-sectional rank.")
            return f"rank({x})"
        if meth in ("sub", "subtract", "add", "mul", "multiply", "div", "divide", "truediv") and n.args:
            y = self.tx(n.args[0])
            op = {"sub": "-", "subtract": "-", "add": "+", "mul": "*", "multiply": "*", "div": "/", "divide": "/",
                  "truediv": "/"}[meth]
            return f"({x} {op} {y})" if y else None
        if meth == "mean" and self._num(self._kw(n, "axis") or ast.Constant(None)) == 1:
            return f"group_mean({x}, 1, market)"
        if meth in ("abs",):
            return f"abs({x})"
        if meth in ("ffill", "bfill") or (meth == "fillna" and any(k.arg == "method" for k in n.keywords)):
            return f"ts_backfill({x}, 20)"
        if meth in ("fillna", "astype", "copy", "dropna", "replace"):
            return x
        if meth == "clip":
            lo, hi = self._kw(n, "lower", 0), self._kw(n, "upper", 1)
            out = x
            if hi is not None and self.tx(hi):
                out = f"min({out}, {self.tx(hi)})"
            if lo is not None and self.tx(lo):
                out = f"max({out}, {self.tx(lo)})"
            return out
        if meth == "apply" and n.args and isinstance(n.args[0], ast.Attribute) and \
                isinstance(n.args[0].value, ast.Name) and n.args[0].value.id in ("np", "numpy"):
            fn = NP_FUNCS.get(n.args[0].attr)
            return f"{fn}({x})" if fn else None
        return None

    def run(self, code: str) -> str | None:
        tree = ast.parse(textwrap.dedent(code))
        body: list[ast.stmt] = list(tree.body)
        # a function definition: translate its body (the last one defined wins)
        funcs = [s for s in body if isinstance(s, ast.FunctionDef)]
        if funcs and not any(isinstance(s, (ast.Assign, ast.Expr)) for s in body if s not in funcs):
            body = list(funcs[-1].body)
        result: str | None = None
        result_name: str | None = None
        for st in body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                name = st.targets[0].id
                v = self.tx(st.value)
                if v is None:
                    if name.lower() not in DF_NAMES:
                        self.warnings.append(f"Could not translate '{name} = {ast.unparse(st.value)[:60]}'.")
                    continue
                vn = self.var_name(name)
                self.lines.append((vn, v))
                self.vars[name] = vn
                result, result_name = vn, name
            elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Subscript):
                tgt = st.targets[0]
                if isinstance(tgt.slice, ast.Constant) and isinstance(tgt.slice.value, str):
                    v = self.tx(st.value)
                    if v is not None:
                        vn = self.var_name(tgt.slice.value)
                        self.lines.append((vn, v))
                        self.vars[tgt.slice.value] = vn
                        result, result_name = vn, tgt.slice.value
            elif isinstance(st, ast.Return) and st.value is not None:
                result = self.tx(st.value)
                result_name = None
            elif isinstance(st, ast.Expr):
                v = self.tx(st.value)
                if v is not None:
                    result, result_name = v, None
        if result is None:
            return None
        named = [k for k in self.vars if k.lower() in ("alpha", "signal", "factor", "score", "weights", "position",
                                                       "positions")]
        if named:
            result = self.vars[named[-1]]
            result_name = named[-1]
        # inline a trailing variable as the final statement
        lines = list(self.lines)
        if result_name is not None and lines and lines[-1][0] == self.vars.get(result_name):
            final = lines.pop()[1]
        else:
            final = result
        used = set(re.findall(r"[A-Za-z_]\w*", final))
        for n_, e_ in reversed(lines):
            if n_ in used:
                used |= set(re.findall(r"[A-Za-z_]\w*", e_))
        lines = [(n_, e_) for n_, e_ in lines if n_ in used]
        text = "".join(f"{n_} = {e_};\n" for n_, e_ in lines) + final
        return _valid(text)


def translate_python(code: str) -> tuple[str | None, list[str]]:
    """pandas/numpy code -> (Fast Expression program or None, warnings). The code is parsed, never executed."""
    t = _PyTranslator()
    try:
        return t.run(code), t.warnings
    except (SyntaxError, ValueError, RecursionError):
        return None, t.warnings


# =========================================================================== documents

def extract_document_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    text = ""
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            parts = []
            for page in reader.pages[:200]:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001 - one bad page must not lose the rest
                    continue
            text = "\n".join(parts)
        except Exception:  # noqa: BLE001
            text = ""
    elif name.endswith((".docx", ".docm")):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                xml = z.read("word/document.xml").decode("utf-8", "replace")
            xml = re.sub(r"</w:p>", "\n", xml)
            xml = re.sub(r"<w:tab/>", "\t", xml)
            text = _html.unescape(re.sub(r"<[^>]+>", "", xml))
        except (zipfile.BadZipFile, KeyError):
            text = ""
    elif name.endswith(".ipynb"):
        try:
            nb = json.loads(data.decode("utf-8", "replace"))
            text = "\n\n".join("".join(c.get("source") or []) for c in nb.get("cells", []))
        except ValueError:
            text = ""
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", "replace")
        if name.endswith((".html", ".htm")) or re.search(r"<html|<body|<p>", text[:2000], re.I):
            text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
            text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", text)
            text = _html.unescape(re.sub(r"<[^>]+>", " ", text))
        elif name.endswith(".rtf") or text.startswith("{\\rtf"):
            text = re.sub(r"\\par[d]?", "\n", text)
            text = re.sub(r"\\'[0-9a-f]{2}|\\[a-z]+-?\d* ?|[{}]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_DOC_CHARS]


def _sentences(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", t)
    return [p.strip() for p in parts if p.strip()]


SECTION_BONUS = re.compile(r"\b(?:abstract|hypothes[ie]s|we (?:find|show|document|propose|argue)|our results?|"
                           r"results? (?:show|suggest|indicate)|conclu(?:de|sion)|in summary|the strategy|"
                           r"long[- ]short|portfolio)\b", re.I)
STOP_SECTION = re.compile(r"^\s*(?:references|bibliography|acknowledg(?:e)?ments?|appendix)\b", re.I | re.M)


def key_sentences(text: str, k: int = 6) -> list[str]:
    """The sentences that carry the idea: mechanism, direction, data, horizon (references are ignored)."""
    from .idea import FAMILY_PATTERNS, FIELD_PATTERNS, NEG_OUT, POS_OUT

    stop = STOP_SECTION.search(text)
    body = text[: stop.start()] if stop and stop.start() > len(text) * 0.3 else text
    sents = _sentences(body)
    fam_rx = [re.compile(p) for pats in FAMILY_PATTERNS.values() for p, _ in pats]
    out_rx = [re.compile(p) for p in POS_OUT + NEG_OUT]
    fld_rx = [re.compile(p) for p, _ in FIELD_PATTERNS]
    scored = []
    for i, s in enumerate(sents):
        low = s.lower()
        words = len(low.split())
        if words < 5 or words > 80 or re.search(r"\bet al\.|\(\d{4}\)|doi:|http", low) and words < 12:
            continue
        sc = 2.0 * sum(1 for r in fam_rx if r.search(low))
        sc += 1.5 * min(2, sum(1 for r in out_rx if r.search(low)))
        sc += 1.0 * min(3, sum(1 for r in fld_rx if r.search(low)))
        sc += 0.5 * len(re.findall(_PERIOD, low))
        if SECTION_BONUS.search(low):
            sc += 1.5
        if i < max(8, len(sents) // 10):
            sc += 0.8  # abstracts and introductions state the idea
        if re.search(r"\b(?:table|figure|panel [a-z]|t-stat|standard errors?)\b", low):
            sc -= 1.5
        if sc > 0:
            scored.append((sc, i, s))
    best = sorted(scored, key=lambda t: -t[0])[:k]
    return [s for _, _, s in sorted(best, key=lambda t: t[1])]


# =========================================================================== BRAIN catalog search

_FIELD_INDEX: dict = {"key": None}


def _ftokens(s: str) -> list[str]:
    from .idea import _stem

    return [_stem(w) for w in re.findall(r"[a-z][a-z0-9]+", s.lower()) if len(w) > 2 and w not in _STOP]


_STOP = set("the and for with from that this are was per value values data field fields based daily total "
            "number average of in on at to by as is be".split())


def search_fields(text: str, k: int = 12, exclude: set[str] | None = None) -> list[tuple[str, float]]:
    """BRAIN catalog fields relevant to the text (TF-IDF over field ids, descriptions and datasets)."""
    fm = field_map()
    key = (len(fm), id(fm))
    if _FIELD_INDEX["key"] != key:
        docs = {}
        df: dict[str, int] = {}
        for fid, f in fm.items():
            if str(f.get("type", "MATRIX")).upper() == "GROUP":
                continue
            toks = _ftokens(" ".join([fid.replace("_", " "), str(f.get("description", "")),
                                      str(f.get("dataset_name") or f.get("dataset") or ""),
                                      str(f.get("category", ""))]))
            if not toks:
                continue
            docs[fid] = toks
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        _FIELD_INDEX.update(key=key, docs=docs, df=df, n=max(1, len(docs)))
    docs, df, n = _FIELD_INDEX["docs"], _FIELD_INDEX["df"], _FIELD_INDEX["n"]
    q = _ftokens(text)
    if not q:
        return []
    qs = set(q)
    out = []
    for fid, toks in docs.items():
        if exclude and fid in exclude:
            continue
        tset = set(toks)
        hit = qs & tset
        if not hit:
            continue
        sc = sum(math.log(1 + n / (1 + df[t])) for t in hit) / math.sqrt(len(tset) + 3)
        out.append((fid, round(sc, 4)))
    out.sort(key=lambda t: -t[1])
    return out[:k]


# =========================================================================== orchestration

FENCE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.S)
LIST_ITEM = re.compile(r"^\s*(?:[-*•▪◦]|\d{1,2}[.)]|\(?[a-h]\))\s+(.+)$")
SETTING_HINTS = [
    (re.compile(r"neutrali[sz](?:e|ed|ation|ing)?\s*(?:by|to|on|:|=)?\s*(?:the\s+)?(sub[- ]?industry|industry|sector|"
                r"market|none)\b", re.I), "neutralization"),
    (re.compile(r"\bdecay\s*(?:of|=|:)?\s*(\d{1,2})\b", re.I), "decay"),
    (re.compile(r"\btruncation\s*(?:of|=|:)?\s*(0?\.\d+)\b", re.I), "truncation"),
    (re.compile(r"\b(TOP(?:3000|2000|1000|500|200|SP500))\b", re.I), "universe"),
    (re.compile(r"\bdelay[- ]?(0|1)\b|\bdelay\s*(?:=|:)\s*(0|1)\b", re.I), "delay"),
    (re.compile(r"\bregion\s*(?:=|:)?\s*(USA|GLB|EUR|ASI|CHN|JPN|KOR|TWN|HKG|AMR)\b", re.I), "region"),
]


def _settings_hints(text: str) -> dict:
    out: dict = {}
    for rx, key in SETTING_HINTS:
        m = rx.search(text)
        if not m:
            continue
        v = next(g for g in m.groups() if g is not None)
        if key == "neutralization":
            out[key] = v.upper().replace("-", "").replace(" ", "")
        elif key in ("decay", "delay"):
            out[key] = int(v)
        elif key == "truncation":
            out[key] = float(v)
        else:
            out[key] = v.upper()
    return out


def _strip_comments(code: str) -> str:
    lines = []
    for ln in code.splitlines():
        ln = re.sub(r"(?<![:\"'])(?://|#).*$", "", ln) if not re.match(r"^\s*#\s*\w+\s*=", ln) else ""
        if ln.strip():
            lines.append(ln.rstrip())
    return "\n".join(lines)


def as_fastexpr(chunk: str, local_fields: set[str] | None = None) -> str | None:
    """A chunk that already is a Fast Expression (newline-separated statements get their ';')."""
    c = _strip_comments(chunk).strip()
    if not c or "(" not in c and "=" not in c:
        return None
    if re.search(r"\b(?:import|def|return|df\[|np\.|pd\.)\b|\.rolling\(|\.shift\(", c):
        return None
    for cand in (c, ";\n".join(s.strip().rstrip(";") for s in c.splitlines() if s.strip())):
        an = analyze(cand, local_fields=None)
        if an.ok and an.node is not None and (an.op_count >= 1 or ";" in cand):
            return _valid(cand)
    return None


def _json_idea(obj, inp: IdeaInput) -> None:
    if isinstance(obj, list):
        for x in obj[:20]:
            _json_idea(x, inp)
        return
    if not isinstance(obj, dict):
        if isinstance(obj, str):
            inp.text += " " + obj
        return
    for k, v in obj.items():
        kl = str(k).lower()
        if kl in ("expression", "expressions", "alpha", "alphas", "code", "regular", "fastexpr", "formula"):
            for e in (v if isinstance(v, list) else [v]):
                if isinstance(e, dict):
                    e = e.get("code") or e.get("expr")
                if isinstance(e, str):
                    fe = as_fastexpr(e) or translate_paper(e)
                    if fe:
                        inp.seeds.append(fe)
        elif kl in ("settings",) and isinstance(v, dict):
            inp.settings.update({kk: vv for kk, vv in v.items() if kk in ("neutralization", "decay", "truncation",
                                                                         "universe", "region", "delay")})
        elif kl in ("neutralization", "decay", "truncation", "universe", "region", "delay"):
            inp.settings[kl] = v
        elif kl in ("fields", "data", "datafields", "inputs") and isinstance(v, list):
            inp.text += " using " + ", ".join(str(x) for x in v) + "."
        elif isinstance(v, (str, list, dict)):
            if isinstance(v, str):
                inp.text += " " + v.strip().rstrip(".") + "."
            else:
                _json_idea(v, inp)


def parse_input(raw: str, local_fields: set[str] | None = None) -> IdeaInput:
    """Detect the format(s) of an idea and extract seeds, compiled formulas, sub-ideas and settings hints."""
    inp = IdeaInput()
    raw = (raw or "").replace("\r\n", "\n").replace("’", "'").replace("–", "-").replace("—", " - ")
    s = raw.strip()
    if not s:
        return inp
    formats: list[str] = []

    # --- structured records
    if s[:1] in "{[":
        try:
            obj = json.loads(s)
            _json_idea(obj, inp)
            formats.append("json")
            s_text = inp.text.strip()
        except ValueError:
            s_text = s
    else:
        s_text = s
        if re.match(r"^\s*[\w ]{2,30}:\s*\S", s) and len(re.findall(r"^\s*[\w ]{2,30}:\s", s, re.M)) >= 2:
            try:
                import yaml

                obj = yaml.safe_load(s)
                if isinstance(obj, dict) and any(str(k).lower() in ("idea", "hypothesis", "description", "fields",
                                                                     "expression", "alpha", "horizon", "signal")
                                                 for k in obj):
                    inp.text = ""
                    _json_idea(obj, inp)
                    formats.append("yaml")
                    s_text = inp.text.strip()
            except Exception:  # noqa: BLE001 - not YAML after all
                pass
    inp.text = ""

    # --- code blocks
    prose = s_text
    for lang, block in FENCE.findall(s_text):
        prose = prose.replace(block, " ")
        fe = as_fastexpr(block, local_fields)
        if fe and lang.lower() not in ("python", "py"):
            inp.seeds.append(fe)
            formats.append("fastexpr")
            continue
        if lang.lower() in ("python", "py", "") and PY_MARKERS.search(block):
            e, warns = translate_python(block)
            inp.warnings += warns
            if e:
                inp.seeds.append(e)
                formats.append("python")
                continue
        for ln in block.splitlines():
            p = translate_paper(ln) if PAPER_MARKERS.search(ln) else None
            if p:
                inp.seeds.append(p)
                formats.append("paper")
    prose = re.sub(r"```[\w+-]*", " ", prose)

    # --- the whole text is code
    whole = as_fastexpr(prose, local_fields) if "\n\n" not in prose.strip() else None
    if whole and len(prose.strip()) < 4000:
        inp.seeds.append(whole)
        formats.append("fastexpr")
        prose = ""
    elif PY_MARKERS.search(prose) and len(re.findall(r"^\s*\w+\s*=", prose, re.M)) + \
            len(re.findall(r"^\s*(?:def|return|import)\b", prose, re.M)) >= 1:
        code_lines = [ln for ln in prose.splitlines() if re.match(r"^\s*(?:\w+\s*=|def |return|import|from |\s{2,})", ln)
                      or PY_MARKERS.search(ln)]
        e, warns = translate_python("\n".join(code_lines))
        if e:
            inp.seeds.append(e)
            inp.warnings += warns
            formats.append("python")
            prose = "\n".join(ln for ln in prose.splitlines() if ln not in code_lines)

    # --- formula lines in paper notation or plain Fast Expression inside prose
    kept = []
    for ln in prose.splitlines():
        cand = re.sub(r"^\s*(?:alpha\s*#?\s*\d+\s*[:=]|[-*•]\s*)", "", ln, flags=re.I).strip()
        done = False
        if "(" in cand and len(cand) < 600:
            fe = as_fastexpr(cand, local_fields)
            if fe:
                inp.seeds.append(fe)
                formats.append("fastexpr")
                done = True
            elif PAPER_MARKERS.search(cand) or re.match(r"^\(?\s*-?1?\s*\*?\s*\(?\s*(?:rank|correlation|delta|sum)\(",
                                                         cand, re.I):
                p = translate_paper(cand)
                if p:
                    inp.seeds.append(p)
                    formats.append("paper")
                    done = True
        if not done:
            kept.append(ln)
    prose = "\n".join(kept).strip()

    # --- lists of several ideas
    lines = [ln for ln in prose.splitlines() if ln.strip()]
    items = [m.group(1).strip() for m in (LIST_ITEM.match(ln) for ln in lines) if m]
    short = [i for i in items if 4 <= len(i.split()) <= 45]
    if len(short) >= 2 and len(items) >= 0.6 * len(lines):  # numbered paper sections are not a list of ideas
        inp.sub_ideas = short[:8]
        formats.append("list")

    # --- long documents: keep the sentences that carry the idea
    n_sent = len(_sentences(prose))
    if len(prose) > 1200 or n_sent > 12:
        inp.key_sentences = key_sentences(prose)
        formats.append("document")
        text_for_lexicon = " ".join(inp.key_sentences) or prose[:2000]
    else:
        text_for_lexicon = prose
    inp.text = text_for_lexicon.strip()

    # --- described computations and written formulas
    comp_src = "\n".join(inp.key_sentences) if inp.key_sentences else prose
    formula_lines = [ln for ln in prose.splitlines() if re.match(r"^\s*[\w .\-/()&']{1,40}?\s*(?:=|:=)\s*\S", ln)]
    comp = compile_phrase("\n".join(formula_lines + [comp_src]), local_fields)
    if comp:
        inp.compiled = comp
        formats.append("formula" if formula_lines else "described")
    inp.settings.update(_settings_hints(s))
    # --- de-duplicate seeds
    seen, uniq = set(), []
    for e in inp.seeds:
        an = analyze(e)
        if an.ok and an.canon_hash not in seen:
            seen.add(an.canon_hash)
            uniq.append(e)
    inp.seeds = uniq[:8]
    fmts = list(dict.fromkeys(formats))
    kinds = [f for f in fmts if f != "described"]
    code = {"fastexpr", "python", "paper"} & set(kinds)
    prose_words = len(prose.split()) if prose else 0
    if code and (len(set(kinds)) > 1 or prose_words > 6):
        inp.format = "mixed"  # code or formulas together with prose
    elif kinds:
        inp.format = kinds[0]
    else:
        inp.format = "formula" if inp.compiled else "english"
    inp.formats = fmts or ["english"]
    if inp.seeds:
        inp.notes.append(f"Read {len(inp.seeds)} complete alpha formula(s) from the {', '.join(f for f in fmts if f in ('fastexpr', 'paper', 'python', 'json', 'yaml')) or 'text'}.")
    if inp.compiled:
        inp.notes.append("Built " + ", ".join(lab for lab, _ in inp.compiled[:4]) + " from the described computation.")
    if inp.key_sentences:
        inp.notes.append(f"Long text: used the {len(inp.key_sentences)} sentences that state the idea.")
    if inp.sub_ideas:
        inp.notes.append(f"{len(inp.sub_ideas)} separate ideas found; they are combined into one search.")
    return inp


__all__ = ["IdeaInput", "compile_one", "compile_phrase", "extract_document_text", "key_sentences", "parse_input",
           "search_fields", "translate_paper", "translate_python"]
