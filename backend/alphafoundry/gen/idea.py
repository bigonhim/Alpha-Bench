"""Idea Forge, part 1: read a plain-English alpha idea and draft Fast Expressions from it.

``interpret`` is rule-based and fully offline. A phrase lexicon maps the text to idea families
(mechanisms), data fields, the stated direction, horizon/windows, peer groups and event conditions;
Fast Expressions pasted in the text become seeds, and library templates are matched on their
rationale text. ``synthesize`` turns that hypothesis into validated drafts (recipe cores x signs x
normalizations x event gates x settings) for the forge job to screen, combine, repair and evolve.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from ..catalog import field_map
from ..fastexpr import analyze
from ..fastexpr.ast import Field, Node
from ..fastexpr.explain import classify
from ..sim.robustness import snap_window
from .build import CAPBUCKET_TEXT
from .templates import Template, expand, load_templates, settings_grid

# --------------------------------------------------------------------------- lexicon

FAMILY_LABEL = {
    "reversion": "Short-term reversal", "momentum": "Momentum / trend", "seasonality": "Seasonality",
    "value": "Value", "quality": "Quality / profitability", "growth": "Fundamental growth",
    "accruals": "Accruals / earnings quality", "investment": "Investment / asset growth",
    "leverage": "Leverage / balance-sheet risk", "liquidity": "Liquidity / attention",
    "pv_divergence": "Price-volume divergence", "volatility": "Low risk / volatility",
    "sentiment": "News & sentiment (BRAIN data)", "options": "Option-implied (BRAIN data)",
}

# What the family's "amount" measures and which sign is conventional (the direction the literature
# usually finds). The text can override the sign, e.g. "high-volatility stocks outperform".
AMOUNT = {
    "reversion": "recent return", "momentum": "past return", "seasonality": "same-month return last year",
    "value": "cheapness", "quality": "profitability", "growth": "fundamental growth", "accruals": "accruals",
    "investment": "asset growth / investment", "leverage": "leverage", "liquidity": "illiquidity",
    "pv_divergence": "price-volume co-movement", "volatility": "volatility", "sentiment": "sentiment",
    "options": "option-implied signal",
}
CANONICAL = {"reversion": -1, "momentum": 1, "seasonality": 1, "value": 1, "quality": 1, "growth": 1,
             "accruals": -1, "investment": -1, "leverage": -1, "liquidity": 1, "pv_divergence": -1,
             "volatility": -1, "sentiment": 1, "options": 1}
BRAIN_ONLY_FAMILIES = {"sentiment", "options"}
FUNDAMENTAL_FAMILIES = {"value", "quality", "growth", "accruals", "investment", "leverage"}

# (pattern, intrinsic amount sign). A negative sign means the phrase names the opposite end of the
# family's amount ("expensive" is low cheapness, "calm" is low volatility).
FAMILY_PATTERNS: dict[str, list[tuple[str, int]]] = {
    "reversion": [(r"revers(?:al|als|e|es|ion|ing)\b", 1), (r"mean[- ]?revert\w*", 1), (r"\brevert\w*", 1),
                  (r"over[- ]?react\w*", 1), (r"\bbounce\w*", 1), (r"rebound\w*", 1), (r"over[- ]?sold", 1),
                  (r"over[- ]?bought", 1), (r"snap[- ]?back", 1), (r"give (?:back|up) (?:their |the )?gains", 1),
                  (r"pull[- ]?backs?", 1), (r"contrarian", 1), (r"\bfad(?:e|es|ing)\b", 1), (r"exhaust\w*", 1),
                  (r"capitulat\w*", 1), (r"\bdips?\b", 1), (r"sell[- ]?offs?\b|sold off", 1),
                  (r"short[- ]term losers?", 1), (r"overshoot\w*", 1), (r"liquidity provi\w*", 1)],
    "momentum": [(r"momentum", 1), (r"\btrend\w*", 1), (r"\bcontinu(?:e|es|ation|ing)\b", 1),
                 (r"\bpersist\w*", 1), (r"\bwinners?\b", 1), (r"keep (?:rising|going up|outperforming|winning)", 1),
                 (r"break[- ]?outs?", 1), (r"52[- ]?week high", 1), (r"new highs?", 1), (r"\bdrift\w*", 1),
                 (r"follow[- ]?through", 1), (r"relative strength", 1), (r"under[- ]?react\w*", 1),
                 (r"slow(?:ly)? (?:diffus|incorporat|react|adjust)\w*", 1)],
    "seasonality": [(r"season\w*", 1), (r"same (?:calendar )?month", 1), (r"\bcalendar\b", 1),
                    (r"\bjanuary\b", 1), (r"annual (?:pattern|cycle)", 1), (r"\brecurr\w*", 1),
                    (r"turn of the (?:month|year)", 1), (r"same time (?:last|each|every) year", 1)],
    "value": [(r"\bvalue stocks?\b|\bvalue (?:factor|premium|investing)\b", 1), (r"\bcheap\w*", 1),
              (r"under[- ]?valued", 1), (r"inexpensive", 1), (r"bargains?", 1),
              (r"low (?:p/?e|pe|multiples?|valuations?|price[- ]to[- ]\w+)", 1), (r"\byields?\b", 1),
              (r"book[- ]to[- ]market", 1), (r"price[- ]to[- ](?:book|earnings|sales)", 1),
              (r"\bp/[ebs]\b", 1), (r"ev/?ebitda", 1), (r"\bmultiples?\b", 1), (r"\bvaluations?\b", 1),
              (r"over[- ]?valued", -1), (r"\bexpensive\b", -1), (r"rich(?:ly)? valued", -1)],
    "quality": [(r"\bquality\b", 1), (r"\bprofitab\w*", 1), (r"\bunprofitab\w*", -1), (r"\bmargins?\b", 1),
                (r"\broe\b", 1), (r"\broa\b", 1), (r"return on (?:equity|assets|capital|invested)", 1),
                (r"efficien\w*", 1), (r"competitive advantage", 1), (r"\bmoats?\b", 1), (r"\bdurable\b", 1),
                (r"(?:stable|steady|consistent) (?:earnings|profits?|margins?)", 1), (r"gross profit\w*", 1),
                (r"pricing power", 1), (r"\bjunk\b", -1)],
    "growth": [(r"\bgrowth\b", 1), (r"\bgrow(?:ing|s)?\b", 1), (r"\bimprov\w*", 1), (r"accelerat\w*", 1),
               (r"(?:rising|increasing|higher|growing) (?:sales|revenues?|earnings|profits?|income|margins?|eps)", 1),
               (r"(?:falling|declining|shrinking|lower) (?:sales|revenues?|earnings|profits?|income|margins?|eps)", -1),
               (r"(?:earnings )?surprises?", 1), (r"\bbeats?\b(?! the market)", 1), (r"upgrades?", 1),
               (r"fundamental momentum", 1), (r"deteriorat\w*", -1)],
    "accruals": [(r"accruals?", 1), (r"earnings quality", -1), (r"cash[- ]backed", -1),
                 (r"not backed by cash", 1), (r"working capital", 1)],
    "investment": [(r"asset growth", 1), (r"\bcapex\b", 1), (r"capital (?:expenditures?|spending)", 1),
                   (r"over[- ]?invest\w*", 1), (r"empire[- ]?build\w*", 1),
                   (r"(?:expanding|growing|bloated) (?:balance sheets?|assets)", 1),
                   (r"investment intensity", 1), (r"aggressive (?:spending|investment|expansion)", 1),
                   (r"\br&d\b|research and development", -1)],
    "leverage": [(r"\bleverag\w*", 1), (r"deleverag\w*", -1), (r"\bdebts?\b", 1), (r"indebted", 1),
                 (r"\bdistress\w*", 1), (r"bankrupt\w*", 1), (r"\bsolven\w*", -1), (r"balance[- ]sheet risk", 1),
                 (r"\bliabilit\w*", 1), (r"\bborrow\w*", 1), (r"credit risk", 1)],
    "liquidity": [(r"\billiquid\w*", 1), (r"\bliquidity\b(?! provi)", 1), (r"amihud", 1), (r"share turnover", -1),
                  (r"turnover of shares", -1), (r"\battention\b", -1), (r"heavily traded", -1),
                  (r"trading activity", -1), (r"crowded", -1)],
    "pv_divergence": [(r"divergen\w*", 1), (r"unconfirmed", 1), (r"not confirmed", 1), (r"price[- ]volume", 1),
                      (r"confirm\w* by volume", 1), (r"volume confirm\w*", 1),
                      (r"correlation between (?:price|returns) and volume", 1)],
    "volatility": [(r"\bvolatil\w*", 1), (r"low[- ]risk", -1), (r"\brisky\b", 1), (r"lottery", 1),
                   (r"\bskew\w*", 1), (r"kurtosis", 1), (r"\bbeta\b", 1), (r"idiosyncratic", 1), (r"\bcalm\w*", -1),
                   (r"jumpy", 1), (r"max(?:imum)? (?:daily )?returns?", 1), (r"tail risk", 1), (r"\bvariance\b", 1),
                   (r"(?:high|low)[- ]vol\b", 1), (r"\bdefensive\b", -1)],
    "sentiment": [(r"(?<!no )(?<!without )(?<!any )\bnews\b", 1), (r"sentiment", 1), (r"\bsocial\b", 1), (r"twitter", 1), (r"\bbuzz\w*", 1),
                  (r"headlines?", 1), (r"media coverage", 1), (r"reddit", 1)],
    "options": [(r"\boptions?\b(?! to)", 1), (r"implied vol\w*", 1), (r"\biv\b", 1), (r"put[- ]call", 1),
                (r"option[- ]implied", 1)],
}

FIELD_PATTERNS: list[tuple[str, list[str]]] = [
    (r"\bvwap\b|volume[- ]weighted", ["vwap"]),
    (r"\bopen(?:ing)?(?: price)?\b(?! interest)|overnight|\bgaps?\b", ["open"]),
    (r"(?:daily|intraday|day'?s|52[- ]?week|recent|the) (?:high|low)s?\b|high[- ]low|trading range|\brange\b",
     ["high", "low"]),
    (r"\bprices?\b|\bclos(?:e|ing)\b", ["close"]),
    (r"\bvolumes?\b|shares traded|trading activity|heavily traded", ["volume", "adv20"]),
    (r"\breturns?\b|\bperformance\b|\bgains?\b|\blosses\b|\bmoves?\b|\bmoved\b", ["returns"]),
    (r"market cap\w*|\bsize\b|small[- ]?caps?|large[- ]?caps?|mega[- ]?caps?|market value|"
     r"(?:small|large|big|tiny) (?:stocks|companies|firms)", ["cap"]),
    (r"shares outstanding|\bdilut\w*|buy[- ]?backs?|repurchas\w*|issuance|share count", ["sharesout"]),
    (r"\bdividends?\b", ["dividend"]),
    (r"\beps\b|earnings per share", ["eps"]),
    (r"\bearnings\b(?! per share)|net income|\bprofits?\b|bottom line", ["income"]),
    (r"operating (?:income|profits?|margins?)|\bebit\b", ["operating_income"]),
    (r"\bebitda\b", ["ebitda"]),
    (r"\bsales\b|\brevenues?\b|top line", ["sales"]),
    (r"free cash ?flows?|\bfcf\b", ["cashflow_op", "capex"]),
    (r"cash ?flows?|\bcfo\b|operating cash", ["cashflow_op"]),
    (r"book value|\bbook\b|\bequity\b(?! research)|net worth", ["equity", "bookvalue_ps"]),
    (r"\bassets\b|asset growth|balance sheets?", ["assets"]),
    (r"\bdebts?\b|\bborrow\w*|\bleverag\w*", ["debt"]),
    (r"\bliabilit\w*", ["liabilities"]),
    (r"\bcash\b(?![- ]?flow)(?![- ]backed)", ["cash"]),
    (r"inventor(?:y|ies)", ["inventory"]),
    (r"receivables?", ["receivable"]),
    (r"goodwill", ["goodwill"]),
    (r"\bcapex\b|capital (?:expenditures?|spending)", ["capex"]),
    (r"\br&d\b|research and development", ["rd_expense"]),
    (r"\bsg&a\b|overhead|selling,? general", ["sga_expense"]),
    (r"gross (?:profit|margin)\w*", ["sales", "cogs"]),
    (r"\bcogs\b|cost of (?:goods|sales|revenue)", ["cogs"]),
    (r"\broe\b|return on equity", ["return_equity"]),
    (r"\broa\b|return on assets", ["return_assets"]),
    (r"current ratio", ["current_ratio"]),
    (r"enterprise value|\bev\b", ["enterprise_value"]),
    (r"retained earnings", ["retained_earnings"]),
    (r"implied vol\w*|\biv\b|\boptions?\b(?! to)", ["implied_volatility_call_120", "implied_volatility_put_120"]),
    (r"(?<!no )(?<!without )(?<!any )\bnews\b|headlines?", ["nws12_afterhsz_sl"]),
    (r"\bsocial\b|twitter|\bbuzz\w*|reddit", ["scl12_alltype_buzzvec"]),
]

GENERIC_FIELDS = {"returns", "close"}
PV_FIELDS = {"open", "high", "low", "close", "vwap", "volume", "returns", "adv20", "cap", "sharesout"}
SCALE_FIELDS = {"cap", "assets", "equity", "sales", "enterprise_value"}

FAMILY_DEFAULT_FIELDS = {
    "reversion": ["close", "vwap", "returns"], "momentum": ["returns", "close"], "seasonality": ["returns"],
    "value": ["income", "cashflow_op", "sales", "ebitda", "equity"],
    "quality": ["operating_income", "income", "cashflow_op", "return_equity"],
    "growth": ["sales", "income", "operating_income", "eps"],
    "accruals": ["operating_income", "cashflow_op", "assets"], "investment": ["assets", "capex"],
    "leverage": ["debt", "liabilities", "equity"], "liquidity": ["volume", "close", "returns"],
    "pv_divergence": ["close", "volume"], "volatility": ["returns", "high", "low"],
    "sentiment": ["volume", "returns"], "options": ["returns", "high", "low"],
}

HIGH_Q = (r"\b(?:high|higher|highest|large|larger|largest|big|bigger|biggest|more|most|rising|increasing|strong|"
          r"stronger|heavy|heavier|top|excessive|elevated|greater|hot)\b")
LOW_Q = (r"\b(?:low|lower|lowest|small|smaller|smallest|less|least|falling|declining|decreasing|weak|weaker|"
         r"little|few|fewer|bottom|minimal|reduced|cold)\b")

MARKER = r"(?:tend|tends|tended|likely|will|would|to|then|later|subsequently|afterwards|next|going forward|keep|continue)"
POS_OUT = [r"outperform\w*", r"beat(?:s|ing)? (?:the )?(?:market|peers|index|benchmark)", r"do(?:es)? better",
           r"(?:higher|better|superior|stronger|greater) (?:future |subsequent |forward |risk[- ]adjusted )?returns",
           r"earn\w* (?:a |an )?(?:premium|excess|higher|better|positive|superior)",
           r"\brall(?:y|ies)\b", rf"{MARKER}\W+(?:\w+\W+){{0,2}}(?:rise|go up|appreciate|gain|climb|recover|rally)",
           r"\bbounce\w*", r"rebound\w*", r"\brecover\w*", r"under[- ]?priced", r"under[- ]?valued",
           r"keep (?:winning|rising|going up|outperforming)", r"continue to (?:rise|outperform|win|gain|go up)",
           r"\bgo long\b", r"\bbuy\b", r"\breward\w*", r"positive (?:future |subsequent )?returns"]
NEG_OUT = [r"underperform\w*", r"do(?:es)? worse",
           r"(?:lower|worse|weaker|poorer) (?:future |subsequent |forward |risk[- ]adjusted )?returns", r"\blag(?:s|ging)?\b",
           r"disappoint\w*", r"over[- ]?priced", r"over[- ]?valued",
           rf"{MARKER}\W+(?:\w+\W+){{0,2}}(?:fall|drop|decline|go down|crash|lose|slump|sink)",
           r"keep (?:losing|falling|dropping|underperforming)", r"continue to (?:fall|drop|decline|underperform|lose)",
           r"give (?:back|up)", r"\bavoid\b", r"\bgo short\b", r"\bsell\b", r"\bpunish\w*",
           r"negative (?:future |subsequent )?returns"]
PAST_UP = (r"\b(?:rose|risen|rises|rising|rallied|gained|jumped|jumps|surged|surges|soared|spiked|spikes|climbed|"
           r"went up|winners?|outperformed|overbought|run[- ]?ups?|big gains?)\b")
PAST_DOWN = (r"\b(?:fell|fallen|falls|drop(?:s|ped|ping)?|declined|declines|plunged|plunges|sold off|sell[- ]?offs?|"
             r"crashed|tanked|slumped|losers?|went down|underperformed|oversold|big losses)\b")

HORIZON_PATTERNS = [
    ("short", r"short[- ]term|intraday|overnight|next (?:day|few days|week)|within (?:a |one )?(?:day|week)|"
              r"few days|\bdaily\b|\bweekly\b|days? later|this week"),
    ("medium", r"medium[- ]term|(?:next|over|within) (?:a |the next |the )?(?:few |several )?(?:weeks|month|months|quarter)|"
               r"\bmonthly\b|\bquarterly\b"),
    ("long", r"long[- ]term|(?:next|over|past|prior|last) (?:the )?(?:year|12 months|twelve months|several years)|"
             r"\bannual(?:ly)?\b|multi[- ]year|\byears\b"),
]
UNIT_DAYS = {"d": 1, "day": 1, "days": 1, "trading day": 1, "trading days": 1, "w": 5, "wk": 5, "wks": 5,
             "week": 5, "weeks": 5, "m": 21, "mo": 21, "mos": 21, "month": 21, "months": 21, "q": 63,
             "quarter": 63, "quarters": 63, "y": 252, "yr": 252, "yrs": 252, "year": 252, "years": 252}
WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "nine": 9,
            "ten": 10, "twelve": 12}
DEFAULT_WINDOWS = {"short": [3, 5, 10], "medium": [20, 40, 60], "long": [126, 252]}
FAMILY_HORIZON = {"reversion": "short", "momentum": "long", "seasonality": "long", "value": "long",
                  "quality": "long", "growth": "long", "accruals": "long", "investment": "long", "leverage": "long",
                  "liquidity": "medium", "pv_divergence": "short", "volatility": "medium", "sentiment": "short",
                  "options": "short"}

GROUP_PATTERNS = [
    ("subindustry", r"sub[- ]?industr\w*|close(?:st)? peers|direct competitors|competitors"),
    ("industry", r"(?<!sub-)(?<!sub)\bindustr(?:y|ies)\b"),
    ("sector", r"\bsectors?\b"),
    ("peers", r"\bpeers?\b|peer group|within (?:its |their )?group|cross[- ]section\w*"),
    ("capbucket", r"\bsize\b|small[- ]?caps?|large[- ]?caps?|market cap|size[- ]neutral"),
    ("market", r"market[- ]neutral|relative to the market|vs\.? (?:the )?market"),
]

CONDITION_PATTERNS = [
    ("volume_event", r"(?:on|after|with|during|amid|accompanied by) (?:very |extremely |an? )?(?:high|heavy|abnormal|unusual|"
                     r"big|large|huge|strong|elevated|surging|spiking|massive|record) (?:trading )?volume|"
                     r"volume (?:spikes?|surges?|shocks?|bursts?|jumps?)|abnormal volume|unusual volume|"
                     r"high[- ]volume (?:days?|moves?|events?|sessions?)"),
    ("low_volume", r"(?:on|with|during) (?:low|light|thin|weak|quiet) (?:trading )?volume|low[- ]volume (?:days?|moves?)"),
    ("earnings_event", r"(?:after|around|following|post)[- ]?(?:the |an? )?earnings|earnings (?:announcements?|releases?|"
                       r"reports?|season|calls?|dates?)|(?:new|latest|recent) (?:filings?|quarterly results)|"
                       r"quarterly (?:results|reports?)"),
    ("high_vol", r"(?:in|during|when) (?:volatile|turbulent|stressed|high[- ]volatility) (?:markets?|periods?|times?|regimes?)|"
                 r"when volatility (?:is )?(?:high|elevated|spikes?)|volatility regimes?"),
    ("extreme", r"\bsharp(?:ly)?\b|\bextreme\w*|\b(?:big|large|huge|massive|outsized) (?:moves?|drops?|gains?|declines?|rallies)"),
]
CONDITION_LABEL = {
    "volume_event": "Only trades after abnormal-volume days (trade_when volume > 1.5 x adv20)",
    "low_volume": "Only trades after low-volume days",
    "earnings_event": "Only updates right after new fundamental filings (days_from_last_change)",
    "high_vol": "Only updates when a stock's short-term volatility exceeds its long-term level",
    "extreme": "Focuses on unusually large moves (volatility-scaled signals and big-move gates)",
}

SMOOTH_PAT = r"low[- ]turnover|slow[- ]moving|\bsmooth\w*|stable (?:signal|positions?)|\bpatient\b|long holding"
INTERACTION_PAT = (r"\b(?:among|within|for|in)\s+(?:the\s+)?(?:\w+[- ]?){0,3}?(?:stocks|companies|firms|names|shares|"
                   r"small[- ]?caps|large[- ]?caps)\b")

STOPWORDS = set("""a an the and or of to in on at for with by from that this these those is are was were be been being
it its their them they as than then which who whom whose when where while if but not no so such into over under about
after before during between more most less least very can could will would should may might do does did done have has
had tend tends tended stock stocks company companies firm firms share shares market alpha idea signal i we you our my
also just only other same each every any some""".split())


def _stem(w: str) -> str:
    for suf, rep in (("ations", ""), ("ation", ""), ("ings", ""), ("ing", ""), ("ies", "y"), ("ied", "y"),
                     ("ers", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)] + rep
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z][a-z0-9&]+", text.lower()) if w not in STOPWORDS and len(w) > 2]


# --------------------------------------------------------------------------- spec


@dataclass
class IdeaSpec:
    text: str
    families: list[str] = field(default_factory=list)
    family_scores: dict[str, float] = field(default_factory=dict)
    direction: dict[str, int] = field(default_factory=dict)
    stated: dict[str, bool] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)        # local fields: named in the text, then typical inputs
    mentioned: list[str] = field(default_factory=list)     # local fields named in the text
    recipe_fields: set[str] = field(default_factory=set)   # inputs of the mechanisms' recipes (set by synthesize)
    brain_fields: list[str] = field(default_factory=list)
    horizon: str = "short"
    horizon_stated: bool = False
    windows: list[int] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    smooth: bool = False
    interaction: bool = False
    subset_families: list[str] = field(default_factory=list)
    subset_size: int = 0
    seeds: list[str] = field(default_factory=list)
    templates: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def local_families(self) -> list[str]:
        return [f for f in self.families if f not in BRAIN_ONLY_FAMILIES]

    @property
    def key_fields(self) -> list[str]:
        """Named fields that make an expression recognisably 'this idea' (generic price/returns excluded)."""
        return [f for f in self.mentioned if f not in GENERIC_FIELDS and f != "adv20"]

    def rationale(self, limit: int = 420) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= limit else t[: limit - 1].rsplit(" ", 1)[0] + "…"

    def to_json(self) -> dict:
        fm = field_map()
        return {
            "text": self.text,
            "families": [{"id": f, "label": FAMILY_LABEL.get(f, f), "score": round(self.family_scores.get(f, 0), 2),
                          "direction": self.direction.get(f, CANONICAL.get(f, 1)),
                          "flipped": self.direction.get(f, CANONICAL.get(f, 1)) != CANONICAL.get(f, 1),
                          "stated": bool(self.stated.get(f)), "amount": AMOUNT.get(f, ""),
                          "brain_only": f in BRAIN_ONLY_FAMILIES} for f in self.families],
            "fields": [{"id": f, "description": fm.get(f, {}).get("description", ""), "local": True,
                        "mentioned": f in self.mentioned} for f in self.fields] +
                      [{"id": f, "description": fm.get(f, {}).get("description", ""), "local": False,
                        "mentioned": True} for f in self.brain_fields],
            "horizon": self.horizon, "horizon_stated": self.horizon_stated, "windows": self.windows,
            "groups": self.groups,
            "conditions": [{"id": c, "label": CONDITION_LABEL.get(c, c)} for c in self.conditions],
            "smooth": self.smooth, "interaction": self.interaction, "subset_families": self.subset_families,
            "seeds": self.seeds, "notes": self.notes, "warnings": self.warnings,
            "confidence": round(self.confidence, 2),
        }


def _find_all(pattern: str, text: str) -> list[re.Match]:
    return list(re.finditer(pattern, text))


def _sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"[.!?;]\s+|\n+", text):
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(text)))
    return [s for s in spans if s[1] > s[0]]


def _sentence_of(pos: int, sents: list[tuple[int, int]]) -> tuple[int, int]:
    for a, b in sents:
        if a <= pos < b:
            return a, b
    return 0, 0


def _outcome_sign(text: str) -> int:
    """+1 if the text says the stocks do well, -1 if badly, 0 if it doesn't say (first mention wins)."""
    best_pos, sign = None, 0
    for pats, s in ((POS_OUT, 1), (NEG_OUT, -1)):
        for p in pats:
            m = re.search(p, text)
            if m and (best_pos is None or m.start() < best_pos):
                best_pos, sign = m.start(), s
    return sign


def _past_move_sign(text: str) -> int:
    """Sign of the recent move the idea conditions on ('stocks that fell' -> -1)."""
    for m in sorted(_find_all(PAST_UP, text) + _find_all(PAST_DOWN, text), key=lambda m: m.start()):
        before = text[max(0, m.start() - 30):m.start()]
        if re.search(r"\b" + MARKER + r"\W+(?:\w+\W+){0,1}$", before):
            continue  # "tend to fall" is an outcome, not the conditioning move
        return 1 if re.fullmatch(PAST_UP, m.group(0)) else -1
    return 0


def _qualifier_sign(text: str, start: int) -> int:
    before = text[max(0, start - 28):start]
    hi = [m.end() for m in re.finditer(HIGH_Q, before)]
    lo = [m.end() for m in re.finditer(LOW_Q, before)]
    if not hi and not lo:
        return 0
    return 1 if (max(hi) if hi else -1) > (max(lo) if lo else -1) else -1


def _find_seeds(text: str, local_fields: set[str] | None) -> list[str]:
    out: list[str] = []
    chunks = [text.strip()] + re.findall(r"`([^`]+)`", text) + [ln.strip() for ln in text.splitlines()]
    for c in chunks:
        c = c.strip().strip(".;")
        if "(" not in c or len(c) > 2000:
            continue
        an = analyze(c, local_fields=local_fields)
        if an.ok and an.node is not None and an.op_count >= 1 and c not in out:
            out.append(c)
    return out[:6]


def _explicit_windows(t: str) -> list[int]:
    ws: list[int] = []
    unit_re = r"(trading[- ]days?|days?|weeks?|wks?|months?|mos?|quarters?|years?|yrs?|d|w|m|q|y)\b"
    for m in re.finditer(r"(\d{1,3})\s*[- ]?\s*" + unit_re, t):
        n, unit = int(m.group(1)), m.group(2).replace("-", " ")
        ws.append(n * UNIT_DAYS.get(unit, UNIT_DAYS.get(unit.rstrip("s"), 1)))
    for m in re.finditer(r"\b(a|an|one|two|three|four|five|six|nine|ten|twelve)[- ](day|week|month|quarter|year)s?\b", t):
        ws.append(WORD_NUM[m.group(1)] * UNIT_DAYS[m.group(2)])
    out = []
    for w in ws:
        if 2 <= w <= 756:
            s = snap_window(w)
            if s not in out:
                out.append(s)
    return out


def match_templates(spec: IdeaSpec, local_fields: set[str] | None, limit: int = 8) -> list[tuple[Template, float]]:
    """Rank library templates by TF-IDF overlap with the idea plus family/field agreement."""
    tmpls = load_templates()
    docs = []
    for t in tmpls:
        words = f"{t.rationale} {t.idea.replace('_', ' ')} {t.id.replace('_', ' ')} {t.category}"
        docs.append(Counter(_tokens(words)))
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    q = Counter(_tokens(spec.text))

    def vec(c: Counter) -> dict[str, float]:
        return {w: (1 + math.log(k)) * math.log(1 + n / (1 + df.get(w, 0))) for w, k in c.items()}

    qv = vec(q)
    qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    fam_w = {f: (1.0 if i == 0 else 0.6) for i, f in enumerate(spec.families)}
    out = []
    for t, d in zip(tmpls, docs):
        dv = vec(d)
        dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        cos = sum(qv[w] * dv.get(w, 0.0) for w in qv) / (qn * dn)
        s = 2.0 * cos + fam_w.get(t.idea, 0.0)
        used = set(re.findall(r"[a-z_0-9]+", t.expr + " " + str(t.slots)))
        s += 0.25 * len(used & set(spec.key_fields))
        if s > 0.3:
            out.append((t, s))
    out.sort(key=lambda ts: -ts[1])
    return out[:limit]


def interpret(text: str, local_fields: set[str] | None = None) -> IdeaSpec:
    raw = text or ""
    t = raw.lower().replace("’", "'").replace("–", "-").replace("—", " - ")
    spec = IdeaSpec(text=raw.strip())
    fm = field_map()
    sents = _sentences(t)

    # ---- seeds: Fast Expressions pasted in the idea
    spec.seeds = _find_seeds(raw, local_fields)

    # ---- families
    first_pos: dict[str, int] = {}
    hits: dict[str, list[tuple[re.Match, int]]] = {}
    for fam, pats in FAMILY_PATTERNS.items():
        for p, intrinsic in pats:
            ms = _find_all(p, t)
            if ms:
                hits.setdefault(fam, []).extend((m, intrinsic) for m in ms)
                spec.family_scores[fam] = spec.family_scores.get(fam, 0.0) + 1.0 + 0.25 * (len(ms) - 1)
                first_pos[fam] = min(first_pos.get(fam, 10 ** 9), ms[0].start())
    fams = sorted(spec.family_scores, key=lambda f: (-spec.family_scores[f], first_pos.get(f, 0)))

    # interaction: "momentum among low-volatility stocks" -> the family inside the phrase conditions the other
    im = re.search(INTERACTION_PAT, t)
    if im and len(fams) >= 1:
        inside = [f for f in fams if im.start() <= first_pos.get(f, -1) < im.end()]
        outside = [f for f in fams if f not in inside]
        if re.search(r"small[- ]?caps?|small (?:stocks|companies|firms)|smaller", im.group(0)):
            spec.subset_size = -1
        elif re.search(r"large[- ]?caps?|large (?:stocks|companies|firms)|bigger|larger", im.group(0)):
            spec.subset_size = 1
        if outside and (inside or spec.subset_size):
            spec.interaction = True
            spec.subset_families = inside
            fams = outside + inside
    spec.families = fams

    # ---- direction per family
    for fam in fams:
        canon = CANONICAL.get(fam, 1)
        text_sign = 0
        for m, intrinsic in hits.get(fam, []):
            a, b = _sentence_of(m.start(), sents)
            sent = t[a:b] if b > a else t
            if fam in ("reversion", "momentum"):
                amount = _past_move_sign(sent) or _past_move_sign(t)
            else:
                q = _qualifier_sign(t, m.start())
                amount = intrinsic * (q if q else 1)
            outcome = _outcome_sign(sent) or _outcome_sign(t)
            if fam == "reversion" and not outcome and amount:
                outcome = -amount  # "stocks that spiked revert": the move itself is what reverses
            if amount and outcome:
                text_sign = amount * outcome
                break
        spec.direction[fam] = text_sign or canon
        spec.stated[fam] = bool(text_sign)

    # ---- fields
    local = local_fields
    mentions: list[tuple[int, str]] = []
    for p, fids in FIELD_PATTERNS:
        m = re.search(p, t)
        if m:
            mentions += [(m.start(), fid) for fid in fids]
    for m in re.finditer(r"[a-z][a-z0-9_]{2,}", t):
        if "_" in m.group(0) and m.group(0) in fm:
            mentions.append((m.start(), m.group(0)))  # exact field ids pasted from BRAIN's data explorer
    seen: list[str] = []
    for _, fid in sorted(mentions, key=lambda pm: pm[0]):
        if fid not in seen:
            seen.append(fid)
    for fid in seen:
        if fid not in fm:
            continue
        if local is None or fid in local:
            spec.fields.append(fid)
        elif str(fm[fid].get("type", "MATRIX")) != "GROUP":
            spec.brain_fields.append(fid)

    # ---- horizon and windows
    spec.windows = _explicit_windows(t)
    found = [(re.search(p, t), h) for h, p in HORIZON_PATTERNS]
    found = [(m.start(), h) for m, h in found if m]
    if found:
        spec.horizon = min(found)[1]
        spec.horizon_stated = True
    elif spec.windows:
        mx = max(spec.windows)
        spec.horizon = "short" if mx <= 10 else ("medium" if mx <= 63 else "long")
        spec.horizon_stated = True
    elif spec.local_families:
        spec.horizon = FAMILY_HORIZON.get(spec.local_families[0], "short")

    # ---- peer groups, conditions, smoothing
    for g, p in GROUP_PATTERNS:
        if re.search(p, t):
            if g == "peers":
                for gg in ("subindustry", "industry"):
                    if gg not in spec.groups:
                        spec.groups.append(gg)
            elif g not in spec.groups:
                spec.groups.append(g)
    for c, p in CONDITION_PATTERNS:
        if re.search(p, t):
            spec.conditions.append(c)
    spec.smooth = bool(re.search(SMOOTH_PAT, t))

    # ---- fallbacks when the text names data but no mechanism, or nothing at all
    if not spec.local_families:
        inferred = []
        f = set(spec.fields)
        if f & {"volume", "adv20"} and f & {"close", "returns", "vwap"}:
            inferred.append("pv_divergence")
        if f & {"income", "sales", "cashflow_op", "ebitda", "operating_income", "eps", "equity", "bookvalue_ps"}:
            inferred += ["value", "growth"]
        if f & {"debt", "liabilities"}:
            inferred.append("leverage")
        if f & {"assets", "capex"}:
            inferred.append("investment")
        if "sentiment" in spec.families:
            inferred.append("liquidity")
            spec.notes.append("News and social data only exist on BRAIN; locally, attention is proxied by abnormal volume.")
        if "options" in spec.families:
            inferred.append("volatility")
            spec.notes.append("Option data only exists on BRAIN; locally, realized volatility stands in for it.")
        for sd in spec.seeds[:1]:
            fam = classify(analyze(sd).node)["idea"]  # type: ignore[arg-type]
            if fam in FAMILY_LABEL and fam not in BRAIN_ONLY_FAMILIES:
                inferred.append(fam)
        if not inferred and not spec.seeds:
            inferred = ["reversion", "momentum", "value", "quality"]
            spec.warnings.append("No specific mechanism recognised; exploring broad families. Naming the data "
                                 "(price, volume, earnings, debt...) and what should happen next sharpens the search.")
        for fam in inferred:
            if fam not in spec.families:
                spec.families.append(fam)
                spec.family_scores[fam] = 0.5
                spec.direction[fam] = CANONICAL[fam]
                spec.stated[fam] = False
        if not spec.horizon_stated and spec.local_families:
            spec.horizon = FAMILY_HORIZON.get(spec.local_families[0], "short")

    spec.mentioned = list(spec.fields)
    for fam in spec.local_families[:3]:
        defaults = [f for f in FAMILY_DEFAULT_FIELDS.get(fam, []) if local is None or f in local]
        if not set(defaults) & set(spec.mentioned):
            spec.fields += [f for f in defaults if f not in spec.fields]
    if not spec.windows:
        spec.windows = list(DEFAULT_WINDOWS[spec.horizon])

    # ---- templates
    spec.templates = [(tt.id, round(s, 3)) for tt, s in match_templates(spec, local_fields)]

    # ---- notes, warnings, confidence
    lf = spec.local_families
    if lf:
        main = FAMILY_LABEL[lf[0]]
        extra = ", ".join(FAMILY_LABEL[f] for f in lf[1:3])
        spec.notes.insert(0, f"Mechanism: {main}" + (f", combined with {extra}" if extra else ""))
        for fam in lf[:3]:
            d = spec.direction[fam]
            hi_lo = "high" if d > 0 else "low"
            how = "as your idea states" if spec.stated[fam] else "the usual direction; the reverse is tested too"
            if spec.stated[fam] and d != CANONICAL[fam]:
                how = "as your idea states, which is the opposite of the usual finding"
            spec.notes.append(f"{FAMILY_LABEL[fam]}: long {hi_lo} {AMOUNT[fam]} ({how})")
    if spec.interaction and (spec.subset_families or spec.subset_size):
        subset = ", ".join(FAMILY_LABEL[f] for f in spec.subset_families) or \
            ("small caps" if spec.subset_size < 0 else "large caps")
        spec.notes.append(f"Applies the main signal within a subset ({subset}), not just alongside it")
    typical = [f for f in spec.fields if f not in spec.mentioned]
    if spec.mentioned:
        spec.notes.append("Data you named: " + ", ".join(spec.mentioned[:8]) +
                          (f"; typical inputs added: {', '.join(typical[:6])}" if typical else ""))
    elif typical:
        spec.notes.append("Data (typical inputs for this mechanism): " + ", ".join(typical[:8]))
    hz = {"short": "short (days)", "medium": "medium (weeks to months)", "long": "long (months to a year)"}[spec.horizon]
    spec.notes.append(f"Horizon: {hz}; windows {', '.join(str(w) for w in spec.windows)}")
    if spec.groups:
        spec.notes.append("Compared within: " + ", ".join(g.replace("capbucket", "size buckets") for g in spec.groups))
    for c in spec.conditions:
        spec.notes.append(CONDITION_LABEL[c])
    if spec.smooth:
        spec.notes.append("Prefers slow, low-turnover positions (higher decay)")
    if spec.seeds:
        spec.notes.append(f"Found {len(spec.seeds)} Fast Expression{'s' if len(spec.seeds) > 1 else ''} in your text; "
                          f"they seed the search")
    if spec.brain_fields:
        spec.warnings.append("Not available locally (BRAIN-only): " + ", ".join(spec.brain_fields[:6]) +
                             ". BRAIN-only variants are saved for you to simulate on BRAIN.")
    conf = 0.0
    conf += 0.4 if any(spec.stated.values()) or spec.family_scores.get(lf[0] if lf else "", 0) >= 1 else 0.15
    conf += 0.25 if spec.key_fields or spec.fields else 0.0
    conf += 0.15 if spec.horizon_stated else 0.0
    conf += 0.1 if spec.groups or spec.conditions else 0.0
    conf += 0.3 if spec.seeds else 0.0
    spec.confidence = min(1.0, conf)
    return spec


def apply_overrides(spec: IdeaSpec, local_fields: set[str] | None, families: list[str] | None = None,
                    horizon: str | None = None) -> IdeaSpec:
    """Let the user correct the reading: pick the mechanisms (in order) and/or the horizon."""
    fams = [f for f in (families or []) if f in FAMILY_LABEL]
    if fams:
        spec.families = fams + [f for f in spec.families if f in BRAIN_ONLY_FAMILIES and f not in fams]
        for f in fams:
            spec.direction.setdefault(f, CANONICAL[f])
            spec.stated.setdefault(f, False)
            spec.family_scores[f] = max(spec.family_scores.get(f, 0.0), 1.0)
            defaults = [x for x in FAMILY_DEFAULT_FIELDS.get(f, []) if local_fields is None or x in local_fields]
            if not set(defaults) & set(spec.mentioned):
                spec.fields += [x for x in defaults if x not in spec.fields]
        spec.warnings = [w for w in spec.warnings if not w.startswith("No specific mechanism")]
        spec.notes = [n for n in spec.notes if not n.startswith("Mechanism:")]
        spec.notes.insert(0, "Mechanism (your choice): " + ", ".join(FAMILY_LABEL[f] for f in fams))
        spec.templates = [(t.id, round(s, 3)) for t, s in match_templates(spec, local_fields)]
    if horizon in DEFAULT_WINDOWS and horizon != spec.horizon:
        spec.horizon = horizon
        spec.horizon_stated = True
        spec.windows = list(DEFAULT_WINDOWS[horizon])
        spec.notes = [n for n in spec.notes if not n.startswith("Horizon:")]
        spec.notes.append(f"Horizon (your choice): {horizon}; windows {', '.join(str(w) for w in spec.windows)}")
    return spec


# --------------------------------------------------------------------------- synthesis


@dataclass
class Draft:
    expr: str
    settings: dict
    family: str
    label: str
    sign: int = 1               # +1 follows the hypothesis direction, -1 tests the reverse, 0 makes no claim
    origin: str = "recipe"      # recipe | template | seed | combo | variant
    template_id: str | None = None
    category: str = "pv"
    horizon: str = "short"


def _top_ops(e: str) -> set[str]:
    """Binary operator characters at parenthesis depth 0 (a leading unary minus is ignored)."""
    ops, depth, prev = set(), 0, ""
    for i, ch in enumerate(e):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and i > 0 and ch in "+-*/<>?:=&|,!" and prev not in "+-*/<>?:=&|,!(":
            ops.add(ch)
        if not ch.isspace():
            prev = ch
    return ops


def _atomic(e: str) -> bool:
    return not _top_ops(e)


def neg(e: str) -> str:
    """Negate an expression text without piling up minus signs. Unary minus binds tighter than * and /,
    so '-a / b' already means -(a / b)."""
    e = e.strip()
    ops = _top_ops(e)
    if e.startswith("-") and ops <= {"*", "/"} and not e[1:].lstrip().startswith("-"):
        return e[1:].lstrip()
    return f"-{e}" if ops <= {"*", "/"} else f"-({e})"


def signed(e: str, s: int) -> str:
    return e if s > 0 else neg(e)


def normalized(e: str) -> bool:
    return bool(re.match(r"^(?:rank|group_rank|zscore|group_zscore|quantile|group_neutralize)\(", e)) and _atomic(e)


def as_rank(e: str) -> str:
    return e if normalized(e) else f"rank({e})"


class _Ctx:
    def __init__(self, spec: IdeaSpec, local: set[str]):
        self.spec = spec
        self.local = local
        self.fm = field_map()

    def has(self, *fs: str) -> bool:
        return all(f in self.local for f in fs)

    def F(self, f: str) -> str:
        return f"ts_backfill({f}, 120)" if self.fm.get(f, {}).get("category") == "fundamental" else f

    def unit(self, f: str) -> str:
        return str(self.fm.get(f, {}).get("unit", "unknown"))

    def fund(self, candidates: list[str], default: list[str]) -> list[str]:
        mine = [f for f in self.spec.fields if f in candidates and f in self.local]
        return mine or [f for f in default if f in self.local]

    def price_fields(self) -> list[str]:
        mine = [f for f in self.spec.fields if f in ("close", "vwap", "open") and f in self.local]
        return (mine or ["close"])[:2]

    def windows(self, lo: int, hi: int, default: list[int]) -> list[int]:
        ws = [w for w in self.spec.windows if lo <= w <= hi]
        return ws or default

    def yield_of(self, f: str) -> str | None:
        u = self.unit(f)
        if u == "dollar":
            return f"{self.F(f)} / cap"
        if u == "per_share":
            return f"{self.F(f)} / close"
        if u == "ratio":
            return self.F(f)
        return None

    def intensity(self, f: str) -> str | None:
        u = self.unit(f)
        if u == "dollar" and self.has("assets"):
            return f"{self.F(f)} / {self.F('assets')}" if f != "assets" else f"{self.F(f)} / cap"
        if u == "per_share":
            return f"{self.F(f)} / close"
        if u == "ratio":
            return self.F(f)
        return None


FLOWS = ["income", "operating_income", "ebitda", "ebit", "sales", "revenue", "cashflow_op", "eps", "equity",
         "bookvalue_ps", "sales_ps", "dividend"]
PROFIT = ["operating_income", "income", "cashflow_op", "ebitda", "return_equity", "return_assets", "sales"]


def _cores(fam: str, c: _Ctx) -> list[tuple[str, str]]:
    """(label, amount expression) pairs: the amount rises with the family's concept (canonical sign NOT applied)."""
    s = c.spec
    out: list[tuple[str, str]] = []
    g = (s.groups[0] if s.groups and s.groups[0] not in ("capbucket", "market") else "industry")
    if fam == "reversion":
        for p in c.price_fields():
            for w in c.windows(2, 30, [3, 5, 10])[:3]:
                out.append((f"{w}-day change in {p}", f"ts_delta({p}, {w})"))
                out.append((f"{p} vs its {w}-day average", f"{p} / ts_mean({p}, {w}) - 1"))
        w = c.windows(2, 30, [5])[0]
        out.append((f"{w}-day peer-relative return", f"ts_sum(returns - group_mean(returns, 1, {g}), {w})"))
        out.append((f"{w}-day volatility-scaled move", f"ts_delta(close, {w}) / (ts_std_dev(returns, 20) * close)"))
        out.append((f"{w}-day z-score of close", f"ts_zscore(close, {max(w, 5)})"))
        if c.has("volume", "adv20") and ("volume" in s.fields or "volume_event" in s.conditions):
            out.append(("volume-weighted daily return", "returns * (volume / adv20)"))
            out.append((f"{w}-day volume-weighted return", f"ts_sum(returns * volume / adv20, {w})"))
        if c.has("open") and "open" in s.fields:
            out.append(("overnight gap", "(open - ts_delay(close, 1)) / ts_delay(close, 1)"))
            out.append(("intraday move", "(close - open) / open"))
        if c.has("vwap"):
            out.append((f"{w}-day close above VWAP", f"ts_mean(close - vwap, {w}) / close"))
        if c.has("high", "low") and ("high" in s.fields or "low" in s.fields):
            ww = c.windows(5, 60, [10, 20])[0]
            out.append((f"position in the {ww}-day range",
                        f"(close - ts_min(low, {ww})) / (ts_max(high, {ww}) - ts_min(low, {ww}) + 0.001)"))
    elif fam == "momentum":
        longs = c.windows(60, 504, [126, 231])
        for L in longs[:2]:
            skip = 21 if L >= 120 else 5
            out.append((f"{L}-day return skipping the last {skip} days", f"ts_delay(ts_sum(returns, {L}), {skip})"))
        L = longs[0]
        out.append((f"{L}-day peer-relative momentum",
                    f"ts_delay(ts_sum(returns - group_mean(returns, 1, {g}), {L}), 21)"))
        out.append((f"{L}-day return / volatility", f"ts_ir(returns, {L})"))
        for p in c.price_fields()[:1]:
            out.append((f"{p} rank in its {L}-day history", f"ts_rank({p}, {L})"))
        if c.has("high"):
            out.append(("closeness to the 52-week high", "close / ts_max(high, 252)"))
        out.append((f"{g} momentum", f"group_mean(ts_sum(returns, {min(L, 126)}), 1, {g})"))
        short = [w for w in s.windows if w <= 60]
        for w in short[:1]:
            out.append((f"{w}-day trend", f"ts_sum(returns, {w})"))
    elif fam == "seasonality":
        out.append(("same month last year", "ts_delay(ts_sum(returns, 21), 231)"))
        out.append(("same month over the last 3 years",
                    "ts_delay(ts_sum(returns, 21), 231) + ts_delay(ts_sum(returns, 21), 483) + "
                    "ts_delay(ts_sum(returns, 21), 735)"))
    elif fam == "value":
        for f in c.fund(FLOWS, ["income", "cashflow_op", "sales", "ebitda"])[:4]:
            y = c.yield_of(f)
            if y:
                out.append((f"{f} yield", y))
            if c.unit(f) == "dollar" and c.has("enterprise_value") and f != "equity":
                out.append((f"{f} / enterprise value", f"{c.F(f)} / enterprise_value"))
            if y and c.unit(f) == "dollar":
                out.append((f"{f} yield vs its own 1-year history", f"ts_rank({y}, 252)"))
        if c.has("bookvalue_ps") and ("bookvalue_ps" in s.fields or not out):
            out.append(("book-to-price", f"{c.F('bookvalue_ps')} / close"))
    elif fam == "quality":
        for f in c.fund(PROFIT, ["operating_income", "income", "cashflow_op"])[:3]:
            i = c.intensity(f)
            if i:
                out.append((f"{f} / assets" if c.unit(f) == "dollar" else f, i))
            if c.unit(f) == "dollar" and c.has("equity") and f in ("income", "operating_income"):
                out.append((f"{f} / equity", f"{c.F(f)} / {c.F('equity')}"))
        if c.has("operating_income", "sales"):
            out.append(("operating margin", f"{c.F('operating_income')} / {c.F('sales')}"))
        if c.has("sales", "cogs", "assets"):
            out.append(("gross profitability", f"({c.F('sales')} - {c.F('cogs')}) / {c.F('assets')}"))
        if c.has("cashflow_op", "income", "assets"):
            out.append(("cash-backed earnings",
                        f"{c.F('cashflow_op')} / (abs({c.F('income')}) + 0.01 * {c.F('assets')})"))
        if c.has("income", "assets"):
            out.append(("earnings stability", f"-ts_std_dev({c.F('income')} / {c.F('assets')}, 252)"))
    elif fam == "growth":
        for f in c.fund(FLOWS + ["operating_income"], ["sales", "income", "operating_income"])[:3]:
            if c.unit(f) == "dollar" and c.has("assets"):
                out.append((f"1-year change in {f} / assets", f"ts_delta({c.F(f)}, 252) / {c.F('assets')}"))
            elif c.unit(f) == "per_share":
                out.append((f"1-year change in {f} / price", f"ts_delta({c.F(f)}, 252) / close"))
            out.append((f"{f} vs its 1-year history", f"ts_zscore({c.F(f)}, 252)"))
            out.append((f"quarterly acceleration of {f}",
                        f"ts_delta({c.F(f)}, 63) / (abs(ts_delay({c.F(f)}, 63)) + 0.001)"))
        if c.has("eps"):
            out.append(("earnings-surprise proxy", f"ts_delta({c.F('eps')}, 63) / close"))
    elif fam == "accruals":
        if c.has("operating_income", "cashflow_op", "assets"):
            out.append(("operating accruals", f"({c.F('operating_income')} - {c.F('cashflow_op')}) / {c.F('assets')}"))
        if c.has("income", "cashflow_op", "assets"):
            out.append(("net income accruals", f"({c.F('income')} - {c.F('cashflow_op')}) / {c.F('assets')}"))
        if c.has("assets_curr", "liabilities_curr", "assets"):
            out.append(("working-capital build-up",
                        f"ts_delta({c.F('assets_curr')} - {c.F('liabilities_curr')}, 252) / {c.F('assets')}"))
    elif fam == "investment":
        if c.has("assets"):
            out.append(("1-year asset growth", f"ts_delta({c.F('assets')}, 252) / ts_delay({c.F('assets')}, 252)"))
            out.append(("asset growth vs history", f"ts_zscore({c.F('assets')}, 252)"))
        if c.has("capex", "assets"):
            out.append(("capex intensity", f"{c.F('capex')} / {c.F('assets')}"))
        if c.has("rd_expense") and "rd_expense" in s.fields:
            # negative amount: R&D-heavy firms tend to earn more, unlike other investment
            out.insert(0, ("R&D intensity", f"-{c.F('rd_expense')} / cap"))
            if c.has("sales"):
                out.insert(1, ("R&D / sales", f"-{c.F('rd_expense')} / {c.F('sales')}"))
                out.insert(2, ("1-year change in R&D / sales",
                               f"-ts_delta({c.F('rd_expense')} / {c.F('sales')}, 252)"))
    elif fam == "leverage":
        for num in c.fund(["debt", "liabilities"], ["debt", "liabilities"])[:2]:
            if c.has("assets"):
                out.append((f"{num} / assets", f"{c.F(num)} / {c.F('assets')}"))
            out.append((f"{num} / market cap", f"{c.F(num)} / cap"))
        if c.has("liabilities", "assets"):
            out.append(("1-year change in leverage", f"ts_delta({c.F('liabilities')} / {c.F('assets')}, 252)"))
        if c.has("equity", "assets"):
            out.append(("equity / assets (inverse leverage)", f"-{c.F('equity')} / {c.F('assets')}"))
    elif fam == "liquidity":
        ws = c.windows(5, 126, [20, 60])
        for w in ws[:2]:
            out.append((f"{w}-day Amihud illiquidity", f"ts_mean(abs(returns) / (volume * close), {w})"))
        w = ws[0]
        if c.has("sharesout"):
            out.append((f"{w}-day share turnover (reversed)", f"-ts_mean(volume, {w}) / sharesout"))
        out.append((f"{w}-day abnormal volume (reversed)", f"-ts_zscore(volume, {w})"))
        out.append(("volume surge (reversed)", f"-ts_delta(ts_mean(volume, 5), {w}) / adv20"))
    elif fam == "pv_divergence":
        for w in c.windows(3, 60, [5, 10, 20])[:3]:
            out.append((f"{w}-day price-volume rank correlation", f"ts_corr(rank(close), rank(volume), {w})"))
        w = c.windows(3, 60, [10])[0]
        out.append((f"{w}-day price-volume covariance", f"ts_covariance(rank(close), rank(volume), {w})"))
        if c.has("high"):
            out.append((f"{w}-day high-volume correlation", f"ts_corr(high, rank(volume), {w})"))
        out.append((f"{w}-day return-volume correlation", f"ts_corr(returns, volume / adv20, {w})"))
    elif fam == "volatility":
        for w in c.windows(5, 252, [20, 60])[:2]:
            out.append((f"{w}-day volatility", f"ts_std_dev(returns, {w})"))
        w = c.windows(5, 252, [60])[0]
        if c.has("high", "low"):
            out.append((f"{w}-day high-low range", f"ts_mean((high - low) / close, {w})"))
        out.append(("max daily return (lottery)", f"ts_max(returns, {min(w, 21)})"))
        out.append((f"{max(w, 60)}-day skewness", f"ts_skewness(returns, {max(w, 60)})"))
        out.append((f"{w}-day idiosyncratic volatility", f"ts_std_dev(returns - group_mean(returns, 1, {g}), {w})"))
        out.append(("market beta", f"ts_regression(returns, group_mean(returns, 1, market), {max(w, 120)}, rettype=2)"))
    return out


def _generic_cores(c: _Ctx, skip: set[str]) -> list[tuple[str, str, str]]:
    """(label, core, category) for fields the idea names that no mechanism recipe uses; both signs get tested."""
    out = []
    for f in c.spec.key_fields[:4]:
        if f not in c.local or f in skip or f in ("adv20", "cap", "sharesout"):
            continue
        cat = str(c.fm.get(f, {}).get("category", "pv"))
        if cat == "fundamental":
            y = c.yield_of(f) or c.F(f)
            out.append((f"{f} level", y, cat))
            out.append((f"1-year change in {f}", f"ts_zscore({c.F(f)}, 252)", cat))
        elif f == "volume":
            out.append(("abnormal volume", "volume / adv20", "pv"))
            out.append(("20-day volume trend", "ts_zscore(volume, 20)", "pv"))
        else:
            w = c.windows(2, 252, [20])[0]
            out.append((f"{w}-day change in {f}", f"ts_delta({f}, {w})", "pv"))
            out.append((f"{f} vs its {w}-day history", f"ts_rank({f}, {w})", "pv"))
    return out


def _wrappers(fam: str, c: _Ctx) -> list[tuple[str, str]]:
    """Normalizations, as ('label', format string with {x})."""
    s = c.spec
    groups = [g for g in s.groups if g in ("subindustry", "industry", "sector")]
    if not groups:
        groups = ["industry", "subindustry"] if fam in FUNDAMENTAL_FAMILIES else ["subindustry"]
    out = [("ranked", "rank({x})")]
    for g in groups[:2]:
        out.append((f"ranked within {g}", f"group_rank({{x}}, {g})"))
    if any(g in ("subindustry", "industry", "sector") for g in s.groups):
        out = out[1:] + out[:1]  # the idea compares with peers: peer-relative rank is the primary form
    if "capbucket" in s.groups or fam == "liquidity":
        out.insert(1, ("size-neutral rank", f"group_neutralize(rank({{x}}), {CAPBUCKET_TEXT})"))
    return out


def _gate(cond: str, c: _Ctx) -> tuple[str, str] | None:
    if cond == "volume_event" and c.has("volume", "adv20"):
        return "on abnormal-volume days", "trade_when(volume > 1.5 * adv20, {x}, -1)"
    if cond == "low_volume" and c.has("volume", "adv20"):
        return "on low-volume days", "trade_when(volume < adv20, {x}, -1)"
    if cond == "earnings_event":
        f = next((f for f in ("eps", "income", "sales") if f in c.local), None)
        if f:
            return "right after new filings", f"trade_when(days_from_last_change({f}) < 10, {{x}}, -1)"
    if cond == "high_vol":
        return "when volatility is elevated", "trade_when(ts_std_dev(returns, 20) > ts_std_dev(returns, 120), {x}, -1)"
    if cond == "extreme":
        return "after unusually large moves", "trade_when(abs(returns) > 2 * ts_std_dev(returns, 20), {x}, -1)"
    return None


def settings_for(fam: str, spec: IdeaSpec, base: dict) -> list[dict]:
    hz = spec.horizon if spec.horizon_stated else FAMILY_HORIZON.get(fam, spec.horizon)
    decays = {"short": [0, 3, 6], "medium": [0, 4, 8], "long": [0, 4]}[hz]
    if spec.smooth:
        decays = sorted({d + 6 for d in decays} | {decays[-1]})
    neuts: list[str] = []
    for g in spec.groups:
        n = {"subindustry": "SUBINDUSTRY", "industry": "INDUSTRY", "sector": "SECTOR", "market": "MARKET"}.get(g)
        if n and n not in neuts:
            neuts.append(n)
    for n in (["INDUSTRY", "SUBINDUSTRY"] if fam in FUNDAMENTAL_FAMILIES else ["SUBINDUSTRY", "INDUSTRY", "MARKET"]):
        if n not in neuts:
            neuts.append(n)
    out = []
    for n in neuts[:3]:
        for d in decays:
            out.append({**base, "decay": d, "neutralization": n})
    return out


def _horizon_of(fam: str, spec: IdeaSpec) -> str:
    return spec.horizon if spec.horizon_stated else FAMILY_HORIZON.get(fam, spec.horizon)


def synthesize(spec: IdeaSpec, local_fields: set[str], base: dict, budget: int = 120,
               rng: random.Random | None = None) -> list[Draft]:
    """Validated, locally simulable drafts in priority order (seeds, hypothesis cores, reverse tests,
    matched templates), deduplicated by canonical form and settings."""
    rng = rng or random.Random(0)
    c = _Ctx(spec, local_fields)
    per_family: list[list[Draft]] = []
    fams = spec.local_families or ["reversion"]
    weights = []
    covered: set[str] = set()
    for k, fam in enumerate(fams[:4]):
        drafts: list[Draft] = []
        sign = spec.direction.get(fam, CANONICAL.get(fam, 1))
        cores = _cores(fam, c)
        wraps = _wrappers(fam, c)
        gates = [g for g in (_gate(cd, c) for cd in spec.conditions
                             if cd != "extreme" or fam not in FUNDAMENTAL_FAMILIES) if g]
        sets = settings_for(fam, spec, base)
        cat = "fundamental" if fam in FUNDAMENTAL_FAMILIES else "pv"
        hz = _horizon_of(fam, spec)
        # round-robin so every core gets its primary form before any core gets a second one
        variants: list[list[Draft]] = []
        reverse: list[Draft] = []
        for i, (label, core) in enumerate(cores):
            covered |= set(re.findall(r"[a-z_][a-z_0-9]*", core))
            x = signed(core, sign)
            vs = []
            for j, (wl, wf) in enumerate(wraps):
                e = wf.format(x=x)
                st = sets[(i + j) % len(sets)]
                vs.append(Draft(e, st, fam, f"{label}, {wl}", 1, "recipe", None, cat, hz))
                for gl, gf in gates:
                    vs.append(Draft(gf.format(x=e), sets[(i + j + 1) % len(sets)], fam,
                                    f"{label}, {wl}, {gl}", 1, "recipe", None, cat, hz))
            if i < 2:  # does the data agree with the stated direction? test the reverse of the top cores
                reverse.append(Draft(wraps[0][1].format(x=signed(core, -sign)), sets[0], fam,
                                     f"{label}, reversed (tests the opposite direction)", -1, "recipe", None, cat, hz))
            variants.append(vs)
        depth = max((len(v) for v in variants), default=0)
        for r in range(depth):
            for vs in variants:
                if r < len(vs):
                    drafts.append(vs[r])
        drafts[3:3] = reverse  # early, so even small budgets test the direction
        per_family.append(drafts)
        weights.append(spec.family_scores.get(fam, 0.5) * (1.6 if k == 0 else 1.0))

    fm = field_map()
    spec.recipe_fields = {w for w in covered if w in fm}
    for sd in spec.seeds:
        spec.recipe_fields |= set(re.findall(r"[a-z_][a-z_0-9]*", sd)) & set(fm)
    generic = _generic_cores(c, skip=covered)
    if generic:
        fam0 = fams[0]
        gd = []
        for label, core, cat in generic:
            for sg in (1, -1):
                gd.append(Draft(f"rank({signed(core, sg)})", settings_for(fam0, spec, base)[0], fam0,
                                f"{label} ({'high' if sg > 0 else 'low'} is long)", 0, "recipe", None, cat,
                                _horizon_of(fam0, spec)))
        per_family.append(gd)
        weights.append(0.6)

    tmpl_drafts: list[Draft] = []
    by_id = {t.id: t for t in load_templates()}
    for tid, _score in spec.templates:
        t = by_id.get(tid)
        if t is None:
            continue
        exprs = expand(t, local_fields=local_fields, limit=24, rng=rng)
        # a template may bolt on another mechanism's data (e.g. a value fallback); keep only on-idea expansions
        exprs = [e for e in exprs if not foreign_fields(analyze(e).node, spec)]  # type: ignore[arg-type]
        key = set(spec.key_fields)
        exprs.sort(key=lambda e: -len(key & set(re.findall(r"[a-z_0-9]+", e))))
        grid = settings_grid(t, base)
        for e in exprs[:4]:
            tmpl_drafts.append(Draft(e, rng.choice(grid), t.idea, f"template {t.id}: {t.rationale}",
                                     0, "template", t.id, t.category, t.horizon))

    seed_drafts: list[Draft] = []
    fam0 = fams[0]
    for sd in spec.seeds:
        for st in settings_for(fam0, spec, base)[:3]:
            seed_drafts.append(Draft(sd, st, fam0, "your expression", 1, "seed", None, "pv", _horizon_of(fam0, spec)))
        if not normalized(sd):
            seed_drafts.append(Draft(f"rank({sd})", settings_for(fam0, spec, base)[0], fam0,
                                     "your expression, ranked", 1, "seed", None, "pv", _horizon_of(fam0, spec)))

    # allocate: seeds first, then families by weight, templates get ~20%
    out: list[Draft] = []
    seen: set[str] = set()

    def take(d: Draft) -> bool:
        an = analyze(d.expr, local_fields=local_fields)
        if not an.ok or not an.local:
            return False
        k = an.canon_hash + "|" + str(sorted((k, str(v)) for k, v in d.settings.items()))
        if k in seen:
            return False
        seen.add(k)
        out.append(d)
        return True

    for d in seed_drafts:
        if len(out) >= budget:
            break
        take(d)
    remaining = budget - len(out)
    t_quota = min(len(tmpl_drafts), max(2, int(remaining * 0.2)))
    f_quota = remaining - t_quota
    tot = sum(weights) or 1.0
    quotas = [max(4, int(round(f_quota * w / tot))) for w in weights]
    for q, drafts in zip(quotas, per_family):
        n = 0
        for d in drafts:
            if n >= q or len(out) >= budget:
                break
            n += take(d)
    for d in tmpl_drafts:
        if len(out) >= budget:
            break
        take(d)
    # top up from any family list if validation rejected some
    for drafts in per_family:
        for d in drafts:
            if len(out) >= budget:
                break
            take(d)
    return out


def brain_only_drafts(spec: IdeaSpec, local_fields: set[str], base: dict, limit: int = 4) -> list[Draft]:
    """Expressions on data only BRAIN has (options, news, social, imported fields) for the user to test there."""
    fm = field_map()
    out: list[Draft] = []
    by_id = {t.id: t for t in load_templates()}
    for tid, _ in spec.templates:
        t = by_id.get(tid)
        if t is None or expand(t, local_fields=local_fields, limit=1):
            continue
        for e in expand(t, local_fields=None, limit=2):
            out.append(Draft(e, settings_grid(t, base)[0], t.idea, f"template {t.id}: {t.rationale}", 1,
                             "template", t.id, t.category, t.horizon))
    w = (spec.windows or [20])[0]
    for fid in spec.brain_fields:
        f = fm.get(fid, {})
        ftype = str(f.get("type", "MATRIX"))
        x = f"vec_avg({fid})" if ftype == "VECTOR" else fid
        x = f"ts_backfill({x}, 20)"
        fam = spec.families[0] if spec.families else "other"
        for label, e in ((f"{fid}: {w}-day z-score", f"rank(ts_zscore({x}, {max(w, 5)}))"),
                         (f"{fid} within sub-industry", f"group_rank({x}, subindustry)"),
                         (f"{fid}: {w}-day change (reversed)", f"-rank(ts_delta({x}, {max(w, 5)}))")):
            out.append(Draft(e, {**base, "decay": 4}, fam, label, 1, "brain_only", None,
                             str(f.get("category", "other")), spec.horizon))
    valid = []
    seen = set()
    for d in out:
        an = analyze(d.expr, local_fields=local_fields)
        if an.ok and not an.local and an.canon_hash not in seen:
            seen.add(an.canon_hash)
            valid.append(d)
    return valid[:limit]


def allowed_fields(spec: IdeaSpec) -> set[str]:
    """Data an expression may use and still be 'this idea': what the idea names, its mechanisms' recipe
    inputs, price-volume basics, and the usual size denominators for fundamental ideas."""
    a = set(spec.fields) | set(spec.recipe_fields) | PV_FIELDS | set(spec.brain_fields)
    if any(f in FUNDAMENTAL_FAMILIES for f in spec.local_families):
        a |= SCALE_FIELDS
    return a


def foreign_fields(node: Node, spec: IdeaSpec) -> set[str]:
    """Data fields in the expression that the idea never asked for (e.g. a value fallback in a reversal idea)."""
    fm = field_map()
    names = {x.name for x in node.walk() if isinstance(x, Field)}
    return {n for n in names - allowed_fields(spec) if str(fm.get(n, {}).get("category", "group")) != "group"}


SYNONYMS = [
    {"debt", "debt_lt", "debt_st", "liabilities", "liabilities_curr"},
    {"income", "operating_income", "ebit", "ebitda", "eps", "return_equity", "return_assets"},
    {"sales", "revenue", "sales_ps"},
    {"equity", "bookvalue_ps"},
    {"volume", "adv20"},
    {"close", "vwap", "open"},
    {"assets", "assets_curr"},
]


def _covers(f: str, names: set[str]) -> bool:
    return f in names or any(f in g and g & names for g in SYNONYMS)


def missing_fields(names: set[str], spec: IdeaSpec) -> list[str]:
    """Named key fields that the expression expresses neither directly nor through a close synonym."""
    return [f for f in spec.key_fields if not _covers(f, names)]


def fidelity(node: Node, spec: IdeaSpec, family: str | None = None) -> float:
    """0..1: how recognisably the expression still expresses the idea (named data, mechanism, no foreign data)."""
    names = {x.name for x in node.walk() if isinstance(x, Field)}
    key = set(spec.key_fields)
    if key:
        f_part = min(1.0, (len(key) - len(missing_fields(names, spec))) / min(len(key), 2))
    else:
        f_part = 1.0 if names & (set(spec.fields) | GENERIC_FIELDS) else 0.5
    fam = family if family in spec.families else classify(node)["idea"]
    m_part = 1.0 if fam in spec.families else (0.5 if fam in FAMILY_LABEL else 0.3)
    score = 0.6 * f_part + 0.4 * m_part
    if foreign_fields(node, spec):
        score *= 0.5
    return round(score, 3)


def preview(spec: IdeaSpec, local_fields: set[str], base: dict, n: int = 5) -> list[dict]:
    """A few representative first drafts for the interpretation card: the user's seeds, one per mechanism,
    one per event condition and one per named field, then the rest in priority order."""
    drafts = [d for d in synthesize(spec, local_fields, base, budget=max(n * 12, 48), rng=random.Random(1))
              if d.sign >= 0]
    picks: list[Draft] = [d for d in drafts if d.origin == "seed"][:2]
    for fam in spec.local_families:
        picks += [d for d in drafts if d.family == fam and d.origin == "recipe"][:1]
    if spec.conditions:
        picks += [d for d in drafts if "trade_when" in d.expr][:1]
    for f in spec.key_fields:
        picks += [d for d in drafts if re.search(rf"\b{re.escape(f)}\b", d.expr)][:1]
    picks += drafts
    out: list[dict] = []
    for d in picks:
        if not any(o["expr"] == d.expr for o in out):
            out.append({"expr": d.expr, "label": d.label, "family": d.family, "origin": d.origin})
        if len(out) >= n:
            break
    return out


__all__ = ["IdeaSpec", "Draft", "interpret", "apply_overrides", "synthesize", "brain_only_drafts", "fidelity", "preview",
           "allowed_fields", "foreign_fields", "missing_fields",
           "match_templates", "settings_for", "as_rank", "neg", "signed", "normalized", "FAMILY_LABEL",
           "CANONICAL"]
