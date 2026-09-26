"""BRAIN Fast Expression operator catalog (consultant set).

Each operator is declared with a compact parameter spec so the list stays readable and editable:

    "x:m"                       matrix argument (a numeric constant is also accepted)
    "g:g"                       group argument (group field or group-valued expression such as bucket(...))
    "v:v"                       vector field argument
    "d:w"                       lookback window (positive integer literal)
    "k:i=1"                     integer literal with default
    "rate:n=2"                  numeric literal with default
    "range:s=0,1,0.1"           string literal with default
    "filter:b=false"            boolean literal with default
    "driver:e(gaussian|uniform|cauchy)=gaussian"   enum literal (bare identifier or string)
    "*:m"                       variadic tail: any number of extra matrix arguments

Scalar parameters (w/i/n/s/b/e) must be literals, as on BRAIN. Data parameters (m/g/v) may be any
expression. Whether an operator can be evaluated by the local engine is decided at runtime by
the engine's registry (see ``alphafoundry.engine.ops``); operators without a local implementation
are still valid syntax and mark an alpha as "BRAIN-only".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

DATA_KINDS = {"m", "g", "v"}
SCALAR_KINDS = {"w", "i", "n", "s", "b", "e"}


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    default: Any = None
    has_default: bool = False
    choices: tuple[str, ...] = ()

    @property
    def is_data(self) -> bool:
        return self.kind in DATA_KINDS

    def to_json(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.has_default:
            d["default"] = self.default
        if self.choices:
            d["choices"] = list(self.choices)
        return d


@dataclass(frozen=True)
class OpSpec:
    name: str
    category: str
    params: tuple[Param, ...]
    returns: str
    doc: str
    example: str = ""
    level: str = "base"  # base | consultant
    variadic: bool = False
    commutative: bool = False
    infix: str | None = None  # printed as infix operator when set
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def data_params(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.is_data)

    @property
    def scalar_params(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if not p.is_data)

    @property
    def signature(self) -> str:
        parts = []
        for p in self.params:
            if p.has_default:
                dv = p.default
                if isinstance(dv, bool):
                    dv = "true" if dv else "false"
                elif p.kind == "s":
                    dv = f'"{dv}"'
                parts.append(f"{p.name}={dv}")
            else:
                parts.append(p.name)
        if self.variadic:
            parts.insert(len(self.data_params), "...")
        return f"{self.name}({', '.join(parts)})"

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "params": [p.to_json() for p in self.params],
            "returns": self.returns,
            "doc": self.doc,
            "example": self.example,
            "level": self.level,
            "variadic": self.variadic,
            "commutative": self.commutative,
            "infix": self.infix,
            "signature": self.signature,
        }


def _parse_param(spec: str) -> Param:
    name, _, rest = spec.partition(":")
    kind, eq, default_s = rest.partition("=")
    choices: tuple[str, ...] = ()
    if kind.startswith("e(") and kind.endswith(")"):
        choices = tuple(kind[2:-1].split("|"))
        kind = "e"
    if not eq:
        return Param(name, kind, None, False, choices)
    default: Any
    if kind in ("w", "i"):
        default = int(default_s)
    elif kind == "n":
        default = float("nan") if default_s == "nan" else float(default_s)
    elif kind == "b":
        default = default_s.lower() == "true"
    else:
        default = default_s
    return Param(name, kind, default, True, choices)


def op(name: str, category: str, params: list[str], returns: str, doc: str, example: str = "",
       level: str = "base", commutative: bool = False, infix: str | None = None) -> OpSpec:
    variadic = False
    parsed: list[Param] = []
    for s in params:
        if s.startswith("*:"):
            variadic = True
            continue
        parsed.append(_parse_param(s))
    return OpSpec(name, category, tuple(parsed), returns, doc, example, level, variadic, commutative, infix)


A, L, TS, CS, V, T, G, S = ("Arithmetic", "Logical", "Time Series", "Cross Sectional", "Vector",
                            "Transformational", "Group", "Special")
C = "consultant"

OPERATORS: list[OpSpec] = [
    # ------------------------------------------------------------------ Arithmetic
    op("abs", A, ["x:m"], "m", "Absolute value of x.", "abs(returns)"),
    op("add", A, ["x:m", "y:m", "*:m", "filter:b=false"], "m",
       "x + y (+ ...). With filter=true, NaNs are treated as 0.", "add(rank(x), rank(y))", commutative=True, infix="+"),
    op("subtract", A, ["x:m", "y:m", "filter:b=false"], "m", "x - y. With filter=true, NaNs are treated as 0.",
       "subtract(close, open)", infix="-"),
    op("multiply", A, ["x:m", "y:m", "*:m", "filter:b=false"], "m", "x * y (* ...). With filter=true, NaNs are treated as 1.",
       "multiply(rank(x), rank(y))", commutative=True, infix="*"),
    op("divide", A, ["x:m", "y:m"], "m", "x / y. Division by zero gives NaN.", "divide(close, open)", infix="/"),
    op("reverse", A, ["x:m"], "m", "-x.", "reverse(returns)"),
    op("inverse", A, ["x:m"], "m", "1 / x.", "inverse(close)"),
    op("sign", A, ["x:m"], "m", "Sign of x: -1, 0 or +1 (NaN stays NaN).", "sign(ts_delta(close, 1))"),
    op("log", A, ["x:m"], "m", "Natural logarithm (NaN for x <= 0).", "log(volume)"),
    op("exp", A, ["x:m"], "m", "Natural exponential.", "exp(returns)"),
    op("sqrt", A, ["x:m"], "m", "Square root (NaN for x < 0).", "sqrt(volume)"),
    op("power", A, ["x:m", "y:m"], "m", "x ^ y.", "power(returns, 2)"),
    op("signed_power", A, ["x:m", "y:m"], "m", "sign(x) * abs(x) ^ y; keeps the sign while reshaping magnitudes.",
       "signed_power(rank(x) - 0.5, 2)"),
    op("s_log_1p", A, ["x:m"], "m", "sign(x) * log(1 + abs(x)); compresses outliers symmetrically.", "s_log_1p(ts_delta(volume, 5))"),
    op("max", A, ["x:m", "y:m", "*:m"], "m", "Element-wise maximum.", "max(close - open, 0)", commutative=True),
    op("min", A, ["x:m", "y:m", "*:m"], "m", "Element-wise minimum.", "min(returns, 0)", commutative=True),
    op("floor", A, ["x:m"], "m", "Round down to integer.", "floor(x)"),
    op("ceiling", A, ["x:m"], "m", "Round up to integer.", "ceiling(x)"),
    op("round", A, ["x:m"], "m", "Round to nearest integer.", "round(x)"),
    op("round_down", A, ["x:m", "f:n=1"], "m", "Round x down to the nearest multiple of f.", "round_down(x, f=0.1)"),
    op("fraction", A, ["x:m"], "m", "Fractional part: sign(x) * (abs(x) - floor(abs(x))).", "fraction(close)"),
    op("densify", A, ["x:g"], "g", "Re-encodes a grouping field into consecutive bucket ids (only buckets present).",
       "densify(subindustry)"),
    op("to_nan", A, ["x:m", "value:n=0", "reverse:b=false"], "m", "Converts value to NaN (or everything but value if reverse).",
       "to_nan(volume, value=0)"),
    op("nan_mask", A, ["x:m", "y:m"], "m", "NaN where y < 0, else x.", "nan_mask(x, y)"),
    op("nan_out", A, ["x:m", "lower:n=0", "upper:n=0"], "m", "NaN where x < lower or x > upper.", "nan_out(returns, lower=-0.1, upper=0.1)"),
    op("purify", A, ["x:m"], "m", "Replaces +/-inf with NaN.", "purify(x)"),
    op("pasteurize", A, ["x:m"], "m", "NaN for instruments outside the universe (and inf -> NaN).", "pasteurize(x)"),
    op("arc_cos", A, ["x:m"], "m", "Inverse cosine.", "arc_cos(x)"),
    op("arc_sin", A, ["x:m"], "m", "Inverse sine.", "arc_sin(x)"),
    op("arc_tan", A, ["x:m"], "m", "Inverse tangent.", "arc_tan(x)"),
    op("tanh", A, ["x:m"], "m", "Hyperbolic tangent.", "tanh(zscore(x))"),
    op("sigmoid", A, ["x:m"], "m", "1 / (1 + exp(-x)).", "sigmoid(zscore(x))"),
    op("log_diff", A, ["x:m"], "m", "log(x) - log(ts_delay(x, 1)).", "log_diff(close)"),

    # ------------------------------------------------------------------ Logical
    op("if_else", L, ["cond:m", "x:m", "y:m"], "m", "x where cond is true, else y. Same as cond ? x : y.",
       "if_else(volume > adv20, -returns, 0)"),
    op("and", L, ["x:m", "y:m"], "m", "Logical AND (1/0). Same as x && y.", "and(x > 0, y > 0)", commutative=True, infix="&&"),
    op("or", L, ["x:m", "y:m"], "m", "Logical OR (1/0). Same as x || y.", "or(x > 0, y > 0)", commutative=True, infix="||"),
    op("not", L, ["x:m"], "m", "Logical NOT (1/0).", "not(is_nan(x))"),
    op("is_nan", L, ["x:m"], "m", "1 where x is NaN, else 0.", "is_nan(eps)"),
    op("less", L, ["x:m", "y:m"], "m", "x < y (1/0).", "close < open", infix="<"),
    op("less_equal", L, ["x:m", "y:m"], "m", "x <= y (1/0).", "close <= open", infix="<="),
    op("greater", L, ["x:m", "y:m"], "m", "x > y (1/0).", "volume > adv20", infix=">"),
    op("greater_equal", L, ["x:m", "y:m"], "m", "x >= y (1/0).", "close >= open", infix=">="),
    op("equal", L, ["x:m", "y:m"], "m", "x == y (1/0).", "sign(x) == 1", commutative=True, infix="=="),
    op("not_equal", L, ["x:m", "y:m"], "m", "x != y (1/0).", "x != 0", commutative=True, infix="!="),

    # ------------------------------------------------------------------ Time Series
    op("ts_mean", TS, ["x:m", "d:w"], "m", "Average of x over the past d days.", "ts_mean(returns, 20)"),
    op("ts_sum", TS, ["x:m", "d:w"], "m", "Sum of x over the past d days.", "ts_sum(returns, 5)"),
    op("ts_std_dev", TS, ["x:m", "d:w"], "m", "Standard deviation of x over the past d days.", "ts_std_dev(returns, 20)"),
    op("ts_zscore", TS, ["x:m", "d:w"], "m", "(x - ts_mean(x, d)) / ts_std_dev(x, d).", "ts_zscore(volume, 60)"),
    op("ts_rank", TS, ["x:m", "d:w", "constant:n=0"], "m", "Rank of today's x within its past d values, in [0,1] (+ constant).",
       "ts_rank(close, 20)"),
    op("ts_min", TS, ["x:m", "d:w"], "m", "Minimum of x over the past d days.", "ts_min(low, 20)"),
    op("ts_max", TS, ["x:m", "d:w"], "m", "Maximum of x over the past d days.", "ts_max(high, 20)"),
    op("ts_median", TS, ["x:m", "d:w"], "m", "Median of x over the past d days.", "ts_median(returns, 20)"),
    op("ts_product", TS, ["x:m", "d:w"], "m", "Product of x over the past d days.", "ts_product(1 + returns, 5)"),
    op("ts_skewness", TS, ["x:m", "d:w"], "m", "Skewness of x over the past d days.", "ts_skewness(returns, 60)"),
    op("ts_kurtosis", TS, ["x:m", "d:w"], "m", "Excess kurtosis of x over the past d days.", "ts_kurtosis(returns, 60)"),
    op("ts_moment", TS, ["x:m", "d:w", "k:i=0"], "m", "k-th central moment of x over the past d days.", "ts_moment(returns, 60, k=3)", level=C),
    op("ts_arg_max", TS, ["x:m", "d:w"], "m", "Days since the maximum of x in the past d days (0 = today).", "ts_arg_max(close, 10)"),
    op("ts_arg_min", TS, ["x:m", "d:w"], "m", "Days since the minimum of x in the past d days (0 = today).", "ts_arg_min(close, 10)"),
    op("ts_delta", TS, ["x:m", "d:w"], "m", "x - ts_delay(x, d).", "ts_delta(close, 5)"),
    op("ts_delay", TS, ["x:m", "d:w"], "m", "Value of x d days ago.", "ts_delay(close, 1)"),
    op("ts_av_diff", TS, ["x:m", "d:w"], "m", "x - ts_mean(x, d), ignoring NaNs.", "ts_av_diff(volume, 20)"),
    op("ts_scale", TS, ["x:m", "d:w", "constant:n=0"], "m", "(x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant.",
       "ts_scale(close, 20)"),
    op("ts_quantile", TS, ["x:m", "d:w", "driver:e(gaussian|uniform|cauchy)=gaussian"], "m",
       "ts_rank followed by the inverse CDF of the chosen distribution.", "ts_quantile(volume, 20)"),
    op("ts_decay_linear", TS, ["x:m", "d:w", "dense:b=false"], "m", "Linearly decayed average over d days (weights d, d-1, ..., 1).",
       "ts_decay_linear(rank(x), 10)"),
    op("ts_decay_exp_window", TS, ["x:m", "d:w", "factor:n=0.5"], "m", "Exponentially decayed average over d days (weights factor^k).",
       "ts_decay_exp_window(returns, 20, factor=0.8)"),
    op("ts_corr", TS, ["x:m", "y:m", "d:w"], "m", "Pearson correlation of x and y over the past d days.", "ts_corr(close, volume, 20)"),
    op("ts_covariance", TS, ["y:m", "x:m", "d:w"], "m", "Covariance of y and x over the past d days.", "ts_covariance(returns, volume, 20)"),
    op("ts_regression", TS, ["y:m", "x:m", "d:w", "lag:i=0", "rettype:i=0"], "m",
       "Rolling OLS of y on x (lagged). rettype 0=residual, 1=intercept, 2=slope, 3=fitted, 6=R^2.",
       "ts_regression(returns, ts_delay(returns, 1), 60, rettype=2)"),
    op("ts_ir", TS, ["x:m", "d:w"], "m", "ts_mean(x, d) / ts_std_dev(x, d) (information ratio).", "ts_ir(returns, 60)"),
    op("ts_backfill", TS, ["x:m", "lookback:w=252", "k:i=1", "ignore:s=NAN"], "m",
       "Replaces NaN with the k-th most recent non-NaN value within lookback days. Essential for sparse fundamentals.",
       "ts_backfill(eps, 120)"),
    op("ts_count_nans", TS, ["x:m", "d:w"], "m", "Number of NaNs in the past d days.", "ts_count_nans(eps, 60)"),
    op("ts_step", TS, ["n:i=1"], "m", "Day counter (1, 2, 3, ...).", "ts_step(1)"),
    op("ts_returns", TS, ["x:m", "d:w", "mode:i=1"], "m", "mode 1: (x - x[t-d]) / x[t-d]; mode 2: log(x / x[t-d]).",
       "ts_returns(close, 5)"),
    op("ts_max_diff", TS, ["x:m", "d:w"], "m", "x - ts_max(x, d).", "ts_max_diff(close, 20)"),
    op("ts_min_diff", TS, ["x:m", "d:w"], "m", "x - ts_min(x, d).", "ts_min_diff(close, 20)"),
    op("ts_min_max_diff", TS, ["x:m", "d:w", "f:n=0.5"], "m", "x - f * (ts_min(x, d) + ts_max(x, d)).", "ts_min_max_diff(close, 20)"),
    op("ts_min_max_cps", TS, ["x:m", "d:w", "f:n=2"], "m", "(ts_min(x, d) + ts_max(x, d)) - f * x.", "ts_min_max_cps(close, 20)"),
    op("ts_percentage", TS, ["x:m", "d:w", "percentage:n=0.5"], "m", "Percentile value of x over the past d days.",
       "ts_percentage(returns, 60, percentage=0.9)"),
    op("ts_weighted_decay", TS, ["x:m", "k:n=0.5"], "m", "k * x + (1 - k) * ts_delay(x, 1).", "ts_weighted_decay(rank(x), k=0.5)"),
    op("days_from_last_change", TS, ["x:m"], "m", "Days since x last changed value.", "days_from_last_change(eps)"),
    op("last_diff_value", TS, ["x:m", "d:w"], "m", "Most recent value of x (within d days) that differs from today's.",
       "last_diff_value(eps, 90)"),
    op("kth_element", TS, ["x:m", "d:w", "k:i=1"], "m", "k-th most recent non-NaN value within d days (k=1 is the latest).",
       "kth_element(eps, 120, k=1)"),
    op("hump", TS, ["x:m", "hump:n=0.01"], "m",
       "Holds yesterday's value unless the change exceeds hump * mean(|x|); cuts turnover.", "hump(rank(x), hump=0.01)"),
    op("jump_decay", TS, ["x:m", "d:w", "sensitivity:n=0.5", "force:n=0.1"], "m",
       "Damps sudden jumps: if |x - x[t-1]| > sensitivity * ts_std_dev(x, d), move only a force fraction.",
       "jump_decay(rank(x), 20)", level=C),
    op("ts_target_tvr_decay", TS, ["x:m", "lambda_min:n=0", "lambda_max:n=1", "target_tvr:n=0.1"], "m",
       "Exponential smoothing tuned so the output's turnover approaches target_tvr.", "ts_target_tvr_decay(rank(x), target_tvr=0.1)", level=C),
    op("ts_target_tvr_delta_limit", TS, ["x:m", "y:m", "lambda_min:n=0", "lambda_max:n=1", "target_tvr:n=0.1"], "m",
       "Delta-limits x (scaled by y) to reach target turnover.", "ts_target_tvr_delta_limit(x, adv20, target_tvr=0.1)", level=C),
    op("ts_target_tvr_hump", TS, ["x:m", "lambda_min:n=0", "lambda_max:n=1", "target_tvr:n=0.1"], "m",
       "hump() with the threshold tuned to reach target turnover.", "ts_target_tvr_hump(x, target_tvr=0.1)", level=C),
    op("ts_delta_limit", TS, ["x:m", "y:m", "limit_volume:n=0.1"], "m", "Limits daily change of x to limit_volume * y.",
       "ts_delta_limit(x, adv20, limit_volume=0.1)", level=C),
    op("ts_entropy", TS, ["x:m", "d:w", "buckets:i=10"], "m", "Information entropy of x over the past d days.", "ts_entropy(returns, 60)", level=C),
    op("ts_co_kurtosis", TS, ["y:m", "x:m", "d:w"], "m", "Co-kurtosis of y and x over d days.", "ts_co_kurtosis(returns, volume, 60)", level=C),
    op("ts_co_skewness", TS, ["y:m", "x:m", "d:w"], "m", "Co-skewness of y and x over d days.", "ts_co_skewness(returns, volume, 60)", level=C),
    op("ts_partial_corr", TS, ["x:m", "y:m", "z:m", "d:w"], "m", "Partial correlation of x and y controlling for z.",
       "ts_partial_corr(returns, volume, close, 60)", level=C),
    op("ts_triple_corr", TS, ["x:m", "y:m", "z:m", "d:w"], "m", "Triple correlation of x, y, z over d days.",
       "ts_triple_corr(returns, volume, close, 60)", level=C),
    op("ts_theilsen", TS, ["x:m", "y:m", "d:w"], "m", "Theil-Sen robust slope of x on y over d days.", "ts_theilsen(returns, volume, 60)", level=C),
    op("ts_poly_regression", TS, ["y:m", "x:m", "d:w", "k:i=1"], "m", "Residual of polynomial regression of y on x (degree k).",
       "ts_poly_regression(returns, volume, 60, k=2)", level=C),
    op("ts_vector_neut", TS, ["x:m", "y:m", "d:w"], "m", "Removes from x its rolling projection on y (time-series residual).",
       "ts_vector_neut(returns, ts_mean(returns, 5), 60)", level=C),
    op("ts_vector_proj", TS, ["x:m", "y:m", "d:w"], "m", "Rolling projection of x on y.", "ts_vector_proj(returns, volume, 60)", level=C),
    op("inst_tvr", TS, ["x:m", "d:w"], "m", "Instrument-level turnover of x over d days.", "inst_tvr(rank(x), 20)", level=C),
    op("hump_decay", TS, ["x:m", "p:n=0"], "m", "Ignores changes smaller than p relative to yesterday.", "hump_decay(x, p=0.1)", level=C),

    # ------------------------------------------------------------------ Cross Sectional
    op("rank", CS, ["x:m", "rate:n=2"], "m", "Cross-sectional rank in [0,1] (ties averaged).", "rank(-returns)"),
    op("zscore", CS, ["x:m"], "m", "Cross-sectional (x - mean) / std.", "zscore(returns)"),
    op("scale", CS, ["x:m", "scale:n=1", "longscale:n=1", "shortscale:n=1"], "m",
       "Scales x so that sum(|x|) = scale (long/short legs optionally scaled separately).", "scale(rank(x))"),
    op("normalize", CS, ["x:m", "useStd:b=false", "limit:n=0"], "m",
       "x - mean(x) (divided by std if useStd), optionally clipped to +/-limit.", "normalize(x, useStd=true, limit=3)"),
    op("quantile", CS, ["x:m", "driver:e(gaussian|uniform|cauchy)=gaussian", "sigma:n=1"], "m",
       "Rank then inverse CDF of the driver distribution (Gaussianizes the signal).", "quantile(x, driver=gaussian)"),
    op("winsorize", CS, ["x:m", "std:n=4"], "m", "Clips x to mean +/- std standard deviations.", "winsorize(x, std=4)"),
    op("truncate", CS, ["x:m", "maxPercent:n=0.01"], "m", "Caps each value at maxPercent of sum(|x|).", "truncate(x, maxPercent=0.05)"),
    op("scale_down", CS, ["x:m", "constant:n=0"], "m", "(x - min) / (max - min) - constant.", "scale_down(x)"),
    op("vector_neut", CS, ["x:m", "y:m"], "m", "x minus its cross-sectional projection on y (orthogonalizes x to y).",
       "vector_neut(rank(x), rank(cap))"),
    op("vector_proj", CS, ["x:m", "y:m"], "m", "Cross-sectional projection of x on y.", "vector_proj(x, y)"),
    op("regression_neut", CS, ["y:m", "x:m"], "m", "Residual of the cross-sectional OLS of y on x (with intercept).",
       "regression_neut(rank(x), log(cap))"),
    op("regression_proj", CS, ["y:m", "x:m"], "m", "Fitted value of the cross-sectional OLS of y on x.", "regression_proj(y, x)"),
    op("rank_by_side", CS, ["x:m", "rate:n=2", "scale:n=1"], "m", "Ranks positive and negative values separately.", "rank_by_side(x)", level=C),
    op("generalized_rank", CS, ["x:m", "m:n=1"], "m", "Generalized (power) rank.", "generalized_rank(x, m=2)", level=C),
    op("one_side", CS, ["x:m", "side:e(long|short)=long"], "m", "Shifts x to be all long or all short.", "one_side(x, side=long)", level=C),
    op("rank_gmean_amean_diff", CS, ["x:m", "y:m", "*:m"], "m", "Geometric minus arithmetic mean of the input ranks.",
       "rank_gmean_amean_diff(x, y, z)", level=C),

    # ------------------------------------------------------------------ Vector (vector fields -> matrix)
    op("vec_avg", V, ["x:v"], "m", "Mean of the vector field for each instrument/day.", "vec_avg(nws12_afterhsz_sl)"),
    op("vec_sum", V, ["x:v"], "m", "Sum of the vector field.", "vec_sum(scl12_alltype_buzzvec)"),
    op("vec_max", V, ["x:v"], "m", "Max of the vector field.", "vec_max(x)"),
    op("vec_min", V, ["x:v"], "m", "Min of the vector field.", "vec_min(x)"),
    op("vec_count", V, ["x:v"], "m", "Number of elements in the vector field.", "vec_count(x)"),
    op("vec_stddev", V, ["x:v"], "m", "Standard deviation of the vector field.", "vec_stddev(x)"),
    op("vec_range", V, ["x:v"], "m", "max - min of the vector field.", "vec_range(x)"),
    op("vec_ir", V, ["x:v"], "m", "mean / std of the vector field.", "vec_ir(x)"),
    op("vec_skewness", V, ["x:v"], "m", "Skewness of the vector field.", "vec_skewness(x)"),
    op("vec_kurtosis", V, ["x:v"], "m", "Kurtosis of the vector field.", "vec_kurtosis(x)"),
    op("vec_norm", V, ["x:v"], "m", "Sum of absolute values of the vector field.", "vec_norm(x)"),
    op("vec_percentage", V, ["x:v", "percentage:n=0.5"], "m", "Percentile of the vector field.", "vec_percentage(x, percentage=0.5)"),
    op("vec_powersum", V, ["x:v", "constant:n=2"], "m", "Sum of |x|^constant over the vector field.", "vec_powersum(x, constant=2)"),
    op("vec_choose", V, ["x:v", "nth:i=0"], "m", "n-th element of the vector field.", "vec_choose(x, nth=0)"),

    # ------------------------------------------------------------------ Transformational
    op("trade_when", T, ["x:m", "y:m", "z:m"], "m",
       "Event gate: y where trigger x > 0; NaN (exit) where z > 0; otherwise keep yesterday's value. z = -1 never exits.",
       "trade_when(volume > adv20, rank(-returns), -1)"),
    op("bucket", T, ["x:m", "range:s=", "buckets:s=", "skipBoth:b=false", "NANGroup:b=false"], "g",
       'Turns x into group ids: range="start,end,step" or buckets="b1,b2,...". Commonly bucket(rank(cap), range="0.1,1,0.1").',
       'bucket(rank(cap), range="0.1,1,0.1")'),
    op("clamp", T, ["x:m", "lower:n=0", "upper:n=0", "inverse:b=false", "mask:n=nan"], "m",
       "Clips x to [lower, upper] (inverse: values inside the range are replaced by mask).", "clamp(x, lower=-3, upper=3)"),
    op("left_tail", T, ["x:m", "maximum:n=0"], "m", "NaN where x > maximum.", "left_tail(rank(x), maximum=0.2)"),
    op("right_tail", T, ["x:m", "minimum:n=0"], "m", "NaN where x < minimum.", "right_tail(rank(x), minimum=0.8)"),
    op("tail", T, ["x:m", "lower:n=0", "upper:n=0", "newval:n=0"], "m", "Values within [lower, upper] are replaced by newval.",
       "tail(zscore(x), lower=-1, upper=1, newval=0)"),
    op("filter", T, ["x:m", "h:s=1,2,3,4", "t:s=0.5"], "m", "Linear filter with coefficients h over past values.",
       'filter(x, h="1,2,3,4", t="0.5")', level=C),
    op("keep", T, ["x:m", "f:m", "period:w=5"], "m", "Keeps x for period days after condition f fires.", "keep(x, volume > adv20, period=5)", level=C),

    # ------------------------------------------------------------------ Group
    op("group_neutralize", G, ["x:m", "group:g"], "m", "x minus its group mean (group-neutral).", "group_neutralize(rank(x), subindustry)"),
    op("group_rank", G, ["x:m", "group:g"], "m", "Rank of x within its group, in [0,1].", "group_rank(x, subindustry)"),
    op("group_zscore", G, ["x:m", "group:g"], "m", "z-score of x within its group.", "group_zscore(x, industry)"),
    op("group_scale", G, ["x:m", "group:g"], "m", "(x - group_min) / (group_max - group_min).", "group_scale(x, sector)"),
    op("group_normalize", G, ["x:m", "group:g", "constantCheck:b=false", "tolerance:n=0.01", "scale:n=1"], "m",
       "Scales x so sum(|x|) within each group equals scale.", "group_normalize(x, industry)"),
    op("group_mean", G, ["x:m", "weight:m", "group:g"], "m", "Weighted mean of x within its group.", "group_mean(returns, 1, subindustry)"),
    op("group_median", G, ["x:m", "group:g"], "m", "Median of x within its group.", "group_median(x, industry)"),
    op("group_sum", G, ["x:m", "group:g"], "m", "Sum of x within its group.", "group_sum(x, sector)"),
    op("group_count", G, ["x:m", "group:g"], "m", "Number of non-NaN x within its group.", "group_count(x, industry)"),
    op("group_max", G, ["x:m", "group:g"], "m", "Max of x within its group.", "group_max(x, sector)"),
    op("group_min", G, ["x:m", "group:g"], "m", "Min of x within its group.", "group_min(x, sector)"),
    op("group_std_dev", G, ["x:m", "group:g"], "m", "Standard deviation of x within its group.", "group_std_dev(x, industry)"),
    op("group_percentage", G, ["x:m", "group:g", "percentage:n=0.5"], "m", "Percentile of x within its group.",
       "group_percentage(x, industry, percentage=0.5)"),
    op("group_backfill", G, ["x:m", "group:g", "d:w", "std:n=4"], "m",
       "Fills NaN with the winsorized mean of the group's recent values.", "group_backfill(eps, industry, 60)"),
    op("group_vector_neut", G, ["x:m", "y:m", "group:g"], "m", "vector_neut applied within each group.", "group_vector_neut(x, y, industry)"),
    op("group_vector_proj", G, ["x:m", "y:m", "group:g"], "m", "vector_proj applied within each group.", "group_vector_proj(x, y, industry)"),
    op("group_cartesian_product", G, ["g1:g", "g2:g"], "g", "Combines two groupings into one (every pair is a group).",
       "group_cartesian_product(sector, bucket(rank(cap), range=\"0,1,0.5\"))"),
    op("group_coalesce", G, ["g1:g", "g2:g"], "g", "Uses g1, falling back to g2 where g1 is missing.", "group_coalesce(subindustry, industry)", level=C),
    op("group_extra", G, ["x:m", "weight:m", "group:g"], "m", "Fills NaN with the group mean.", "group_extra(x, 1, industry)", level=C),
    op("group_multi_regression", G, ["y:m", "x:m", "group:g"], "m", "Residual of within-group regression of y on x.",
       "group_multi_regression(y, x, industry)", level=C),

    # ------------------------------------------------------------------ Special
    op("inst_pnl", S, ["x:m"], "m", "Per-instrument PnL of alpha x.", "inst_pnl(rank(x))", level=C),
    op("self_corr", S, ["x:m"], "m", "Self-correlation helper (BRAIN only).", "self_corr(x)", level=C),
    op("convert", S, ["x:m", "mode:s=dollar2share"], "m", "Unit conversion helper.", 'convert(x, mode="dollar2share")', level=C),
]


@lru_cache(maxsize=1)
def operator_map() -> dict[str, OpSpec]:
    return {o.name: o for o in OPERATORS}


INFIX_TO_OP = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide", "<": "less", "<=": "less_equal",
               ">": "greater", ">=": "greater_equal", "==": "equal", "!=": "not_equal", "&&": "and", "||": "or"}
OP_TO_INFIX = {v: k for k, v in INFIX_TO_OP.items()}
