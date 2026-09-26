"""Point-in-time fundamentals from SEC EDGAR XBRL ``companyfacts``.

Rules (no look-ahead):
* For every period, use the FIRST-filed value; later restatements are ignored.
* A value becomes usable on the first trading day strictly after its ``filed`` date.
* Balance-sheet items are point-in-time levels. Flow items are exposed as trailing-twelve-month sums
  of quarterly values; Q4 is derived as FY minus the 9-month year-to-date value.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from pathlib import Path
from typing import Callable

import httpx

COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT", "20-F", "40-F", "8-K"}

STOCK = {
    "assets": ["Assets"],
    "assets_curr": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "liabilities_curr": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "Cash"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "debt_st": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "inventory": ["InventoryNet"],
    "receivable": ["AccountsReceivableNetCurrent"],
    "goodwill": ["Goodwill"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "liab_equity": ["LiabilitiesAndStockholdersEquity"],
}
FLOW = {
    "sales": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
              "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "operating_income": ["OperatingIncomeLoss"],
    "income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "cashflow_op": ["NetCashProvidedByUsedInOperatingActivities",
                    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"],
}
EPS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
SHARES_DEI = ["EntityCommonStockSharesOutstanding"]
SHARES_GAAP = ["CommonStockSharesOutstanding"]


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _first_filed(tax: dict, concepts: list[str], unit: str, duration: bool) -> dict:
    """{key: (filed, val, priority, start)} where key is end (instant) or (start, end) (duration)."""
    best: dict = {}
    for pri, c in enumerate(concepts):
        node = tax.get(c)
        if not node:
            continue
        for f in node.get("units", {}).get(unit, []):
            if f.get("form") not in FORMS or "filed" not in f or "end" not in f:
                continue
            if duration != ("start" in f):
                continue
            key = (f["start"], f["end"]) if duration else f["end"]
            cur = best.get(key)
            cand = (f["filed"], float(f["val"]), pri, f.get("start"))
            if cur is None or pri < cur[2] or (pri == cur[2] and f["filed"] < cur[0]):
                best[key] = cand
    return best


def pit_instant(tax: dict, concepts: list[str], unit: str = "USD") -> list[tuple[str, str, float]]:
    best = _first_filed(tax, concepts, unit, duration=False)
    return sorted((v[0], end, v[1]) for end, v in best.items())


def pit_ttm(tax: dict, concepts: list[str], unit: str = "USD") -> list[tuple[str, str, float]]:
    best = _first_filed(tax, concepts, unit, duration=True)
    quarters: dict[str, tuple[str, float]] = {}
    annual: dict[str, tuple[str, float, str]] = {}
    ytd9: dict[str, tuple[str, float, str]] = {}
    for (start, end), (filed, val, _pri, _s) in best.items():
        days = (_d(end) - _d(start)).days
        if 75 <= days <= 105:
            if end not in quarters or filed < quarters[end][0]:
                quarters[end] = (filed, val)
        elif 340 <= days <= 390:
            if end not in annual or filed < annual[end][0]:
                annual[end] = (filed, val, start)
        elif 255 <= days <= 290:
            if end not in ytd9 or filed < ytd9[end][0]:
                ytd9[end] = (filed, val, start)
    for end, (filed, val, start) in annual.items():
        if end in quarters:
            continue
        cands = [(e, v) for e, v in ytd9.items() if v[2] == start and e < end]
        if cands:
            e9, (f9, v9, _s9) = max(cands)
            quarters[end] = (max(filed, f9), val - v9)
    ends = sorted(quarters)
    series: list[tuple[str, str, float]] = []
    have_end = set()
    for k in range(3, len(ends)):
        win = ends[k - 3:k + 1]
        ok = all(60 <= (_d(win[j + 1]) - _d(win[j])).days <= 120 for j in range(3))
        if not ok:
            continue
        ttm = sum(quarters[e][1] for e in win)
        avail = max(quarters[e][0] for e in win)
        series.append((avail, ends[k], ttm))
        have_end.add(ends[k])
    for end, (filed, val, _start) in annual.items():
        if end not in have_end:
            series.append((filed, end, val))
    series.sort()
    return series


def extract(facts: dict) -> dict[str, list]:
    f = facts.get("facts", {})
    gaap, dei = f.get("us-gaap", {}), f.get("dei", {})
    out: dict[str, list] = {}
    for field, concepts in STOCK.items():
        out[field] = pit_instant(gaap, concepts)
    for field, concepts in FLOW.items():
        out[field] = pit_ttm(gaap, concepts)
    out["eps"] = pit_ttm(gaap, EPS, unit="USD/shares")
    sh = pit_instant(dei, SHARES_DEI, unit="shares")
    if not sh:
        sh = pit_instant(gaap, SHARES_GAAP, unit="shares")
    out["shares"] = sh
    return out


def download_all(ciks: list[int], raw_dir: Path, user_agent: str, concurrency: int = 4, rate_per_s: float = 5.0,
                 max_age_days: int = 7, progress: Callable[[int, int, str], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> dict[int, dict]:
    """Fetch companyfacts for each CIK (respecting SEC fair access) and cache the extracted series."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, dict] = {}
    today = dt.date.today()
    min_gap = 1.0 / rate_per_s

    async def run():
        sem = asyncio.Semaphore(concurrency)
        last = [0.0]
        lock = asyncio.Lock()
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
        async with httpx.AsyncClient(timeout=60, headers=headers) as client:
            done = 0

            async def one(cik: int):
                nonlocal done
                if cancelled and cancelled():
                    return
                path = raw_dir / f"{cik}.json"
                data = None
                if path.exists():
                    try:
                        cached = json.loads(path.read_text(encoding="utf-8"))
                        if (today - dt.date.fromisoformat(cached.get("fetched", "1970-01-01"))).days <= max_age_days:
                            data = cached
                    except (OSError, ValueError):
                        data = None
                if data is None:
                    facts = None
                    delay = 1.0
                    for _ in range(5):
                        async with sem:
                            async with lock:
                                wait = last[0] + min_gap - time.monotonic()
                                if wait > 0:
                                    await asyncio.sleep(wait)
                                last[0] = time.monotonic()
                            try:
                                r = await client.get(COMPANYFACTS.format(cik=cik))
                            except httpx.HTTPError:
                                r = None
                        if r is not None and r.status_code == 200:
                            try:
                                facts = r.json()
                            except ValueError:
                                facts = None
                            break
                        if r is not None and r.status_code == 404:
                            break
                        await asyncio.sleep(delay)
                        delay = min(20.0, delay * 2)
                    if facts is not None:
                        data = {"fetched": today.isoformat(), "series": extract(facts)}
                        path.write_text(json.dumps(data), encoding="utf-8")
                    elif path.exists():
                        try:
                            data = json.loads(path.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            data = None
                if data is not None:
                    out[cik] = data["series"]
                done += 1
                if progress:
                    progress(done, len(ciks), str(cik))

            await asyncio.gather(*(one(c) for c in ciks))

    asyncio.run(run())
    return out
