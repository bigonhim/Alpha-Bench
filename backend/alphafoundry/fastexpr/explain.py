"""Rule-based explanations: English description, idea classification and BRAIN description drafts."""

from __future__ import annotations

from ..catalog import field_map, operator_map
from .ast import Const, Field, Node, Op, fmt_num

FRIENDLY = {
    "close": "the close price", "open": "the open price", "high": "the daily high", "low": "the daily low",
    "vwap": "VWAP", "volume": "share volume", "returns": "daily returns", "adv20": "20-day average dollar volume",
    "cap": "market cap", "sharesout": "shares outstanding", "sales": "sales", "revenue": "revenue",
    "assets": "total assets", "liabilities": "total liabilities", "equity": "book equity", "income": "net income",
    "operating_income": "operating income", "cashflow_op": "operating cash flow", "ebitda": "EBITDA",
    "enterprise_value": "enterprise value", "eps": "EPS", "debt": "total debt", "cogs": "cost of goods sold",
    "market": "the whole market", "sector": "sector", "industry": "industry", "subindustry": "sub-industry",
}

WHY = {
    "rank": "rank() maps values to [0,1] cross-sectionally, removing outliers and limiting weight concentration",
    "zscore": "zscore() standardizes the signal across stocks each day",
    "group_rank": "group_rank() compares stocks only with their peers, removing group-level bets",
    "group_neutralize": "group_neutralize() removes the group average so the alpha carries no group exposure",
    "group_zscore": "group_zscore() standardizes within peer groups",
    "ts_rank": "ts_rank() measures where today's value sits relative to its own history",
    "ts_zscore": "ts_zscore() measures how unusual today's value is versus its own history",
    "ts_mean": "ts_mean() smooths noise and lowers turnover",
    "ts_decay_linear": "ts_decay_linear() smooths the signal with linearly decaying weights, lowering turnover",
    "ts_delta": "ts_delta() captures recent change (momentum/reversal)",
    "ts_delay": "ts_delay() lags the input to skip the most recent, noisy period",
    "ts_backfill": "ts_backfill() fills gaps in sparse (e.g. quarterly) data so more stocks carry a signal",
    "ts_std_dev": "ts_std_dev() measures volatility",
    "ts_corr": "ts_corr() measures co-movement between two series",
    "trade_when": "trade_when() only updates positions on events, cutting turnover",
    "hump": "hump() ignores small signal changes, cutting turnover",
    "winsorize": "winsorize() clips extreme values",
    "bucket": "bucket() builds groups (e.g. size buckets) for neutralization",
    "ts_av_diff": "ts_av_diff() measures deviation from the recent average",
    "signed_power": "signed_power() reshapes the signal while keeping its sign",
    "vec_avg": "vec_avg() turns a vector field into one value per stock",
    "vec_sum": "vec_sum() turns a vector field into one value per stock",
    "regression_neut": "regression_neut() removes the part explained by another factor",
    "vector_neut": "vector_neut() orthogonalizes the signal to another factor",
    "scale": "scale() normalizes total exposure",
    "if_else": "if_else() switches between signals depending on a condition",
}


def fname(n: Field) -> str:
    return FRIENDLY.get(n.name, n.name.replace("_", " "))


def describe(node: Node, depth: int = 0) -> str:
    if isinstance(node, Const):
        return fmt_num(node.value)
    if isinstance(node, Field):
        return fname(node)
    assert isinstance(node, Op)
    a = [describe(x, depth + 1) for x in node.args]
    d = node.param("d")
    g = a[-1] if node.args and isinstance(node.args[-1], Field) else (a[-1] if a else "")
    n = node.name
    table = {
        "rank": lambda: f"the cross-sectional rank of {a[0]}",
        "zscore": lambda: f"the cross-sectional z-score of {a[0]}",
        "quantile": lambda: f"the Gaussianized rank of {a[0]}",
        "group_rank": lambda: f"the rank of {a[0]} within each {g}",
        "group_zscore": lambda: f"the z-score of {a[0]} within each {g}",
        "group_neutralize": lambda: f"{a[0]} relative to its {g} average",
        "group_mean": lambda: f"the {a[-1]} average of {a[0]}",
        "ts_rank": lambda: f"the {d}-day time-series rank of {a[0]}",
        "ts_zscore": lambda: f"the {d}-day z-score of {a[0]}",
        "ts_mean": lambda: f"the {d}-day average of {a[0]}",
        "ts_sum": lambda: f"the {d}-day sum of {a[0]}",
        "ts_std_dev": lambda: f"the {d}-day volatility of {a[0]}",
        "ts_delta": lambda: f"the {d}-day change in {a[0]}",
        "ts_delay": lambda: f"{a[0]} from {d} days ago",
        "ts_decay_linear": lambda: f"a {d}-day linearly-decayed average of {a[0]}",
        "ts_decay_exp_window": lambda: f"a {d}-day exponentially-decayed average of {a[0]}",
        "ts_backfill": lambda: f"{a[0]} (gaps back-filled)" if depth else f"{a[0]} with gaps back-filled",
        "ts_corr": lambda: f"the {d}-day correlation between {a[0]} and {a[1]}",
        "ts_covariance": lambda: f"the {d}-day covariance between {a[0]} and {a[1]}",
        "ts_max": lambda: f"the {d}-day maximum of {a[0]}",
        "ts_min": lambda: f"the {d}-day minimum of {a[0]}",
        "ts_arg_max": lambda: f"days since the {d}-day high of {a[0]}",
        "ts_arg_min": lambda: f"days since the {d}-day low of {a[0]}",
        "ts_av_diff": lambda: f"{a[0]} minus its {d}-day average",
        "ts_scale": lambda: f"where {a[0]} sits in its {d}-day range",
        "reverse": lambda: f"the negative of {a[0]}",
        "abs": lambda: f"the absolute value of {a[0]}",
        "log": lambda: f"the log of {a[0]}",
        "sign": lambda: f"the sign of {a[0]}",
        "divide": lambda: f"{a[0]} divided by {a[1]}",
        "multiply": lambda: " times ".join(a),
        "add": lambda: " plus ".join(a),
        "subtract": lambda: f"{a[0]} minus {a[1]}",
        "greater": lambda: f"{a[0]} > {a[1]}",
        "less": lambda: f"{a[0]} < {a[1]}",
        "trade_when": lambda: f"{a[1]}, updated only when {a[0]}",
        "if_else": lambda: f"{a[1]} when {a[0]}, otherwise {a[2]}",
        "winsorize": lambda: f"{a[0]} (winsorized)",
        "hump": lambda: f"{a[0]} (small changes ignored)",
        "vec_avg": lambda: f"the average of {a[0]}",
        "vec_sum": lambda: f"the sum of {a[0]}",
        "bucket": lambda: f"buckets of {a[0]}",
        "signed_power": lambda: f"{a[0]} raised to a signed power",
    }
    f = table.get(n)
    if f is not None:
        try:
            return f()
        except (IndexError, TypeError):
            pass
    return f"{n}({', '.join(a)})"


def _windows(node: Node) -> list[int]:
    out = []
    for n in node.walk():
        if isinstance(n, Op):
            for k, v in n.params:
                if k in ("d", "lookback") and isinstance(v, int):
                    out.append(v)
    return out


def horizon_of(node: Node) -> str:
    ws = [w for w in _windows(node)]
    signal_ws = [w for n in node.walk() if isinstance(n, Op) and n.name != "ts_backfill"
                 for k, w in n.params if k == "d" and isinstance(w, int)]
    ws = signal_ws or ws
    if not ws:
        return "short"
    m = max(ws)
    if m <= 10:
        return "short"
    if m <= 63:
        return "medium"
    return "long"


def classify(node: Node) -> dict:
    """Heuristic tags: idea type, data category, horizon."""
    fields = field_map()
    fnames = {n.name for n in node.walk() if isinstance(n, Field)}
    onames = {n.name for n in node.walk() if isinstance(n, Op)}
    cats = {str(fields.get(f, {}).get("category", "other")) for f in fnames} - {"group"}
    category = "pv" if cats <= {"pv"} else (sorted(cats - {"pv"})[0] if cats - {"pv"} else "pv")
    horizon = horizon_of(node)
    price = {"close", "open", "high", "low", "vwap", "returns"}
    fundamental_value = {"sales", "revenue", "income", "operating_income", "ebitda", "ebit", "cashflow_op", "eps",
                         "bookvalue_ps", "equity"}
    idea = "other"
    neg = isinstance(node, Op) and node.name == "reverse" or any(
        isinstance(n, Op) and n.name == "reverse" for n in node.walk())
    if category == "fundamental":
        if "cap" in fnames or "enterprise_value" in fnames or "close" in fnames:
            idea = "value" if fnames & fundamental_value else "value"
        elif {"ts_delta", "ts_zscore"} & onames:
            idea = "growth"
        elif "assets" in fnames and fnames & fundamental_value:
            idea = "quality"
        elif {"debt", "liabilities"} & fnames:
            idea = "leverage"
        else:
            idea = "quality"
        if "cashflow_op" in fnames and "operating_income" in fnames:
            idea = "accruals"
    elif category in ("news", "sentiment", "social"):
        idea = "sentiment"
    elif category == "option":
        idea = "options"
    elif category == "analyst":
        idea = "analyst"
    else:
        if {"ts_std_dev", "ts_max"} & onames and "returns" in fnames and neg:
            idea = "volatility"
        elif "volume" in fnames or "adv20" in fnames:
            idea = "liquidity" if not (fnames & price) else ("reversion" if neg else "pv_divergence")
            if "ts_corr" in onames:
                idea = "pv_divergence"
        if fnames & price and idea in ("other", "reversion", "liquidity"):
            ws = _windows(node)
            longest = max(ws) if ws else 1
            if "ts_delay" in onames and longest >= 200 and "ts_sum" in onames:
                idea = "seasonality" if any(isinstance(n, Op) and n.name == "ts_delay" and (n.param("d") or 0) > 200
                                            for n in node.walk()) else "momentum"
            elif longest >= 120 and not neg:
                idea = "momentum"
            elif neg or horizon == "short":
                idea = "reversion"
            else:
                idea = "momentum"
    return {"idea": idea, "category": category, "horizon": horizon}


def description(node: Node, rationale: str | None = None) -> dict:
    """Draft of the three BRAIN description sections: idea, data rationale, operator rationale."""
    fields = field_map()
    ops = operator_map()
    tags = classify(node)
    fnames = sorted({n.name for n in node.walk() if isinstance(n, Field)})
    onames = []
    for n in node.walk():
        if isinstance(n, Op) and n.name not in onames:
            onames.append(n.name)
    idea_txt = rationale or {
        "reversion": "Short-term overreaction: stocks that moved sharply tend to partially revert.",
        "momentum": "Trend persistence: medium/long-term winners tend to keep outperforming.",
        "seasonality": "Seasonality: returns in the same calendar period tend to recur year over year.",
        "value": "Value: stocks that are cheap relative to fundamentals tend to outperform.",
        "quality": "Quality/profitability: more profitable, efficient firms tend to outperform.",
        "growth": "Improving fundamentals: accelerating fundamentals are under-reacted to by the market.",
        "accruals": "Earnings quality: earnings not backed by cash flow tend to disappoint.",
        "leverage": "Balance-sheet risk: highly levered firms tend to underperform.",
        "liquidity": "Liquidity/attention: abnormal trading activity carries information about future returns.",
        "pv_divergence": "Price-volume divergence: price moves unconfirmed by volume tend to revert.",
        "volatility": "Low-risk anomaly: lower-volatility stocks earn higher risk-adjusted returns.",
        "sentiment": "Sentiment/attention: news and social activity predict short-term returns.",
        "options": "Option-implied information: option markets lead the stock market.",
        "analyst": "Analyst expectations: revisions and surprises are incorporated slowly.",
    }.get(tags["idea"], "A cross-sectional signal built from the fields below.")
    data_txt = "; ".join(
        f"{f} ({fields.get(f, {}).get('description', 'data field')})" for f in fnames) or "none"
    op_lines = [WHY.get(o) or f"{o}(): {ops[o].doc}" for o in onames if o in ops]
    return {
        "summary": describe(node)[0].upper() + describe(node)[1:],
        "idea": idea_txt,
        "data": data_txt,
        "operators": op_lines,
        "tags": tags,
    }
