from __future__ import annotations

import os
import tempfile

# isolate all runtime state (panels, DB, settings) for the test session
os.environ.setdefault("ALPHAFOUNDRY_RUNTIME", tempfile.mkdtemp(prefix="af_test_runtime_"))
os.environ.setdefault("ALPHAFOUNDRY_SESSION_DIR", tempfile.mkdtemp(prefix="af_test_session_"))
# the tests exercise the miners on the synthetic demo panel on purpose
os.environ.setdefault("ALPHAFOUNDRY_ALLOW_DEMO_MINING", "1")

import numpy as np
import pytest

from alphafoundry.engine.evaluator import EvalContext, SubexprCache
from alphafoundry.engine.panel import Panel, write_panel


@pytest.fixture(scope="session")
def tiny_panel(tmp_path_factory) -> Panel:
    """Small random panel with ~6% NaNs, 3 sectors / 5 subindustries, all-in universe."""
    rng = np.random.default_rng(123)
    T, N = 160, 23
    dates = [str(np.datetime64("2020-01-01") + np.timedelta64(i, "D")) for i in range(T)]
    tickers = [f"T{i}" for i in range(N)]

    def noisy(scale=1.0, loc=0.0):
        a = rng.normal(loc, scale, (T, N))
        a[rng.random((T, N)) < 0.06] = np.nan
        return a

    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    fields = {
        "close": close, "open": close * (1 + rng.normal(0, 0.01, (T, N))), "volume": np.abs(noisy(1e5, 1e6)),
        "returns": noisy(0.02), "x": noisy(), "y": noisy(), "cap": close * 1e6, "adv20": np.full((T, N), 1e6),
        "sales": noisy(10, 100), "high": close * 1.01, "low": close * 0.99, "vwap": close,
    }
    # step-like series for days_from_last_change / last_diff_value
    step = np.repeat(rng.integers(0, 4, (T // 10 + 1, N)), 10, axis=0)[:T].astype(float)
    fields["step"] = step
    sector = (np.arange(N) % 3).astype(np.int32)
    sub = (np.arange(N) % 5).astype(np.int32)
    sub[4] = -1  # one instrument without classification
    groups = {"sector": (sector, ["A", "B", "C"]), "subindustry": (sub, ["s0", "s1", "s2", "s3", "s4"]),
              "industry": (sector, ["A", "B", "C"])}
    univ = {"TOP1500": np.ones((T, N), bool), "TOP100": np.ones((T, N), bool)}
    root = tmp_path_factory.mktemp("tiny_panel")
    write_panel(root, dates, tickers, fields, groups, univ, source="test")
    return Panel(root)


@pytest.fixture()
def ctx(tiny_panel) -> EvalContext:
    return EvalContext(tiny_panel, 0, tiny_panel.T, "TOP1500", False, SubexprCache(64))
