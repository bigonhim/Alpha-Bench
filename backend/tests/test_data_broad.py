"""Broad (TOP3000-like) universe: candidate filtering, SIC crosswalk, liquidity pool, universes, new fundamentals."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from alphafoundry.data import build, edgar, universe
from alphafoundry.data.demo import business_days, liquidity_universes
from alphafoundry.data.sic import Crosswalk, sector_of_sic
from alphafoundry.sim.simulator import resolve_sub_universe, resolve_universe


def test_ticker_and_name_filters():
    js = {"fields": ["cik", "name", "ticker", "exchange"], "data": [
        [320193, "Apple Inc.", "AAPL", "Nasdaq"], [1067983, "Berkshire Hathaway", "BRK-B", "NYSE"],
        [1, "Some Warrant Co", "ABCDW", "Nasdaq"], [2, "Bank Pref", "BAC-PL", "NYSE"], [3, "OTC Co", "OTCX", "OTC"],
        [4, "iShares Core ETF", "IVV", "NYSE"], [5, "Alpha Acquisition Corp", "AAC", "NYSE"],
        [6, "Realty Income Trust", "O", "NYSE"], [7, "Units Co", "XYZ-U", "NYSE"], [8, "No Exchange", "NEX", ""]]}
    df = universe.parse_exchange_file(js)
    assert list(df["ticker"]) == ["AAPL", "BRK-B", "O"]
    assert universe.is_common_ticker("BF-B", "NYSE") and not universe.is_common_ticker("ABC-WT", "NYSE")


def test_merge_broad_keeps_gics_for_sp_names():
    sp = pd.DataFrame([{"ticker": "AAPL", "name": "Apple", "sector": "Information Technology",
                        "industry": "Technology Hardware, Storage & Peripherals",
                        "subindustry": "Technology Hardware, Storage & Peripherals", "cik": 320193, "index": "SP500",
                        "gics": True}])
    listed = pd.DataFrame([{"ticker": "AAPL", "name": "Apple", "cik": 320193, "exchange": "Nasdaq"},
                           {"ticker": "SMAL", "name": "Small Co", "cik": 9, "exchange": "Nasdaq"}])
    df = universe.merge_broad(listed, sp)
    assert len(df) == 2
    assert df.set_index("ticker").loc["AAPL", "sector"] == "Information Technology"
    assert df.set_index("ticker").loc["SMAL", "gics"] is False or not df.set_index("ticker").loc["SMAL", "gics"]


def test_sic_crosswalk_majority_and_fallbacks():
    known = [(3674, "Information Technology", "Semiconductors & Semiconductor Equipment", "Semiconductors")] * 3 + \
            [(3674, "Industrials", "Machinery", "Industrial Machinery")] + \
            [(2834, "Health Care", "Pharmaceuticals", "Pharmaceuticals")] * 2
    cw = Crosswalk(known)
    assert cw.classify(3674)[0][2] == "Semiconductors" and cw.classify(3674)[1] == "sic4"
    assert cw.classify(3672)[1] == "sic3" and cw.classify(3672)[0][0] == "Information Technology"
    (sec, ind, sub), how = cw.classify(6798, "Real Estate Investment Trusts")
    assert how == "division" and sec == "Real Estate" and sub.startswith("SIC 6798")
    assert sector_of_sic(1311) == "Energy" and sector_of_sic(6022) == "Financials" and sector_of_sic(None) == "Unknown"


def _prices(n: int, days: int = 400, seed: int = 0) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    dates = np.array(business_days("2020-01-01", days), dtype="datetime64[D]")
    out = {}
    for i in range(n):
        # liquidity ranks are stable: neighbours differ by 30% in volume, far more than the price noise moves
        close = 20 * np.exp(np.cumsum(rng.normal(0, 0.003, days)))
        vol = np.full(days, 1e3 * 1.3 ** i) * rng.uniform(0.97, 1.03, days)
        out[f"T{i:03d}"] = {"dates": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                            "volume": vol, "adjclose": close, "exchange": "NMS",
                            "div_dates": np.array([], "datetime64[D]"), "div_amount": np.array([]),
                            "split_dates": np.array([], "datetime64[D]"), "split_ratio": np.array([])}
    return out


def test_select_liquid_pool_keeps_most_liquid_with_headroom():
    p = _prices(50)
    keep = build.select_liquid_pool(p, "2020-01-01", 20)
    assert len(keep) == 23  # ceil(20 x 1.15)
    assert "T049" in keep and "T000" not in keep


def test_liquidity_universes_broad_sizes():
    T, N = 30, 3200
    dates = business_days("2021-01-01", T)
    liq = np.tile(np.arange(N, 0, -1, dtype=float), (T, 1))
    u = liquidity_universes(dates, liq, N, whole=None)
    assert {"TOP3000", "TOP2000", "TOP1500", "TOP1000", "TOP500", "TOP200", "TOP100"} <= set(u)
    assert u["TOP3000"][5].sum() == 3000 and u["TOP1000"][5].sum() == 1000
    old = liquidity_universes(dates, liq[:, :1500], 1500)
    assert "TOP3000" not in old and old["TOP1500"].all()


class _P:
    def __init__(self, universes):
        self.universes = universes


def test_universe_mapping_new_and_old_panels():
    broad = _P(["TOP100", "TOP200", "TOP500", "TOP1000", "TOP1500", "TOP2000", "TOP3000"])
    old = _P(["TOP100", "TOP200", "TOP500", "TOP1000", "TOP1500"])
    assert resolve_universe(broad, "TOP3000") == "TOP3000" and resolve_sub_universe(broad, "TOP3000") == "TOP1000"
    assert resolve_universe(old, "TOP3000") == "TOP1500" and resolve_sub_universe(old, "TOP1500") == "TOP500"
    assert resolve_universe(broad, "TOPSP500") == "TOP500"


def test_new_ttm_concepts_extracted():
    def q(start, end, val, filed):
        return {"start": start, "end": end, "val": val, "filed": filed, "form": "10-Q"}
    quarters = [("2021-01-01", "2021-03-31", "2021-05-01"), ("2021-04-01", "2021-06-30", "2021-08-01"),
                ("2021-07-01", "2021-09-30", "2021-11-01"), ("2021-10-01", "2021-12-31", "2022-02-01")]
    facts = {"facts": {"us-gaap": {
        "InterestExpense": {"units": {"USD": [q(s, e, 10.0, f) for s, e, f in quarters]}},
        "PaymentsForRepurchaseOfCommonStock": {"units": {"USD": [q(s, e, 5.0, f) for s, e, f in quarters]}},
    }}}
    out = edgar.extract(facts)
    assert out["interest_expense"][-1][2] == 40.0 and out["buyback"][-1][2] == 20.0
    assert edgar.submissions_meta({"sic": "3674", "sicDescription": "Semiconductors", "entityType": "operating"}) \
        ["sic"] == 3674


def test_forward_fill_unchanged():
    dates = np.array(business_days("2021-01-04", 10), dtype="datetime64[D]")
    s = build.forward_fill_series(dates, [("2021-01-05", "2020-12-31", 1.0), ("2021-01-08", "2021-03-31", 2.0)], 10)
    assert np.isnan(s[1]) and s[2] == 1.0 and s[5] == 2.0


def test_build_real_broad_end_to_end(tmp_path, monkeypatch):
    """The whole broad build with every network call replaced by fixtures."""
    from alphafoundry import config
    from alphafoundry.engine.panel import Panel

    monkeypatch.setattr(config, "PANELS_DIR", tmp_path / "panels")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "load_settings", lambda: {"sec_contact_email": "me@example.com",
                                                          "history_start": "2020-01-01", "universe_pool": "broad",
                                                          "broad_pool_size": 20})
    n = 40
    prices = _prices(n, days=420)
    sp = pd.DataFrame([{"ticker": f"T{i:03d}", "name": f"Co {i}", "sector": "Information Technology",
                        "industry": "Software", "subindustry": "Application Software", "cik": 1000 + i,
                        "index": "SP600"} for i in range(0, n, 2)])
    monkeypatch.setattr(universe, "fetch_universe", lambda *a, **k: sp.copy())
    listed = pd.DataFrame([{"ticker": f"T{i:03d}", "name": f"Co {i}", "cik": 1000 + i, "exchange": "Nasdaq"}
                           for i in range(n)])
    monkeypatch.setattr(universe, "fetch_broad_candidates", lambda ua, s, cache=None: universe.merge_broad(listed, s))
    monkeypatch.setattr(build.yahoo, "download_all", lambda tickers, *a, **k: {t: prices[t] for t in tickers})
    meta = {1000 + i: {"sic": 7372 if i % 2 == 0 else (7371 if i % 3 else 6770), "sicDescription": "Software"}
            for i in range(n)}
    monkeypatch.setattr(edgar, "fetch_submissions_meta", lambda ciks, *a, **k: {c: meta[c] for c in ciks})
    ends = [(dt.date(2019, 3, 31) + dt.timedelta(days=91 * k)) for k in range(8)]
    series = {"sales": [(str(e + dt.timedelta(days=40)), str(e), 100.0 + k) for k, e in enumerate(ends)],
              "assets": [(str(e + dt.timedelta(days=40)), str(e), 500.0) for e in ends],
              "shares": [(str(e + dt.timedelta(days=40)), str(e), 1e6) for e in ends]}
    monkeypatch.setattr(edgar, "download_all", lambda ciks, *a, **k: {c: series for c in ciks})
    info = build.build_real(pool="broad")
    p = Panel(tmp_path / "panels")
    assert info["pool"] == "broad" and p.meta["pool"] == "broad" and p.meta["classification"].startswith("GICS")
    # the liquid pool is the top 20 x 1.15 = 23 names (T017..T039); the 4 SPAC-coded (SIC 6770) non-S&P names go
    assert p.N == 19 and all(int(t[1:]) >= 17 for t in p.tickers)
    assert not {"T021", "T027", "T033", "T039"} & set(p.tickers)
    assert "TOP3000" in p.universes
    for f in ("working_capital", "sales_growth", "buyback", "interest_expense", "depre_amort"):
        assert f in p.field_names()
    sec_codes, _ = p.group("sector")
    assert (sec_codes >= 0).all()
