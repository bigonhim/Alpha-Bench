"""Simulator correctness on hand-checkable cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphafoundry.engine import kernels as K
from alphafoundry.engine.panel import Panel, write_panel
from alphafoundry.fastexpr import lower_text
from alphafoundry.sim.simulator import Periods, SimSettings, perf_metrics, postprocess, simulate


@pytest.fixture(scope="module")
def toy(tmp_path_factory) -> Panel:
    """3 stocks x 12 days with known returns; signal field 's' chosen by hand."""
    T, N = 12, 3
    dates = [f"2021-01-{i + 1:02d}" for i in range(T)]
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, (T, N))
    s = rng.normal(0, 1, (T, N))
    fields = {"returns": ret, "s": s, "close": 100 * np.cumprod(1 + ret, axis=0)}
    groups = {"sector": (np.array([0, 0, 1], np.int32), ["a", "b"]),
              "industry": (np.array([0, 0, 1], np.int32), ["a", "b"]),
              "subindustry": (np.array([0, 0, 1], np.int32), ["a", "b"])}
    root = tmp_path_factory.mktemp("toy")
    write_panel(root, dates, [f"S{i}" for i in range(N)], fields, groups, {"TOP1500": np.ones((T, N), bool)}, "test")
    return Panel(root)


def test_pnl_alignment_and_metrics(toy):
    per = Periods(is_start=0, os_start=toy.T, end=toy.T, brain_start=0)
    s = SimSettings(neutralization="NONE", truncation=0.0, delay=1, decay=0)
    res = simulate(lower_text("s"), s, toy, per, keep_weights=True)
    sig = np.asarray(toy.field("s"), dtype=np.float64)
    w = sig / np.abs(sig).sum(axis=1, keepdims=True)
    ret = np.asarray(toy.field("returns"), dtype=np.float64)
    book = 20e6
    exp_pnl = np.zeros(toy.T)
    exp_pnl[2:] = (w[:-2] * ret[2:]).sum(axis=1) * book  # delay 1: weights from day t-2 earn day t
    np.testing.assert_allclose(res.pnl, exp_pnl, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(res.weights, w, rtol=1e-6)
    tvr = np.r_[0.0, np.abs(np.diff(w, axis=0)).sum(axis=1)]
    np.testing.assert_allclose(res.tvr, tvr, rtol=1e-6)
    m = res.metrics["is"]
    sharpe = exp_pnl.mean() / exp_pnl.std(ddof=1) * math.sqrt(252)
    assert m["sharpe"] == pytest.approx(sharpe, rel=1e-3)
    ann = exp_pnl.mean() * 252 / (book / 2)
    assert m["returns"] == pytest.approx(ann, rel=1e-3)
    fit = sharpe * math.sqrt(abs(ann) / max(tvr.mean(), 0.125))
    assert m["fitness"] == pytest.approx(fit, rel=1e-3)
    # delay 0 uses weights from t-1
    res0 = simulate(lower_text("s"), SimSettings(neutralization="NONE", truncation=0.0, delay=0), toy, per)
    exp0 = np.zeros(toy.T)
    exp0[1:] = (w[:-1] * ret[1:]).sum(axis=1) * book
    np.testing.assert_allclose(res0.pnl, exp0, rtol=1e-9, atol=1e-6)


def test_sign_flip_negates_pnl(toy):
    per = Periods(0, toy.T, toy.T, 0)
    a = simulate(lower_text("s"), SimSettings(neutralization="MARKET"), toy, per)
    b = simulate(lower_text("-s"), SimSettings(neutralization="MARKET"), toy, per)
    np.testing.assert_allclose(a.pnl, -b.pnl, atol=1e-6)


def test_neutralization_and_truncation_invariants():
    rng = np.random.default_rng(3)
    T, N = 50, 40
    a = rng.normal(0, 1, (T, N)).astype(np.float32)
    a[rng.random((T, N)) < 0.1] = np.nan
    codes = np.ascontiguousarray(np.tile(np.arange(N) % 4, (T, 1)), dtype=np.int32)
    neu = K.neutralize_rows(a, codes, 4)
    for t in range(T):
        for g in range(4):
            m = (codes[t] == g) & np.isfinite(neu[t])
            assert abs(neu[t, m].sum()) < 1e-6
    w = K.normalize_truncate(np.ascontiguousarray(neu), 0.05, 12)
    assert np.allclose(np.abs(w).sum(axis=1), 1.0)
    assert np.abs(w).max() <= 0.05 + 1e-9
    w1 = K.normalize_truncate(np.ascontiguousarray(neu), 0.0, 12)
    assert np.allclose(w1.sum(axis=1), 0.0, atol=1e-9)  # neutral without truncation


def test_postprocess_decay_masks_universe(toy):
    s = SimSettings(neutralization="NONE", decay=3, truncation=0.0)
    A = np.asarray(toy.field("s"), dtype=np.float32).copy()
    univ = np.ones_like(A, dtype=bool)
    univ[:, 2] = False
    W = postprocess(A, univ, toy, s)
    assert np.all(W[:, 2] == 0)
    assert np.allclose(np.abs(W).sum(axis=1), 1.0)


def test_perf_metrics_floor():
    pnl = np.array([1.0, 2.0, -1.0, 3.0])
    m = perf_metrics(pnl, np.array([0.05] * 4), np.ones(4), np.ones(4), np.ones(4) * 0.1, 20e6)
    ann = pnl.mean() * 252 / 10e6
    sh = pnl.mean() / pnl.std(ddof=1) * math.sqrt(252)
    assert m["fitness"] == pytest.approx(sh * math.sqrt(abs(ann) / 0.125), rel=1e-3)
