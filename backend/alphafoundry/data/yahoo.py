"""Daily OHLCV from Yahoo Finance's v8 chart endpoint (no API key), with retries and a per-ticker cache."""

from __future__ import annotations

import asyncio
import datetime as dt
import random
from pathlib import Path
from typing import Callable

import httpx
import numpy as np

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
CHART_ALT = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def _to_epoch(d: str) -> int:
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())


def parse_chart(js: dict) -> dict | None:
    res = (js.get("chart") or {}).get("result") or []
    if not res:
        return None
    r = res[0]
    ts = r.get("timestamp") or []
    if not ts:
        return None
    meta = r.get("meta") or {}
    off = int(meta.get("gmtoffset") or 0)
    dates = np.array([dt.datetime.fromtimestamp(t + off, dt.timezone.utc).date().isoformat() for t in ts],
                     dtype="datetime64[D]")
    q = (r.get("indicators") or {}).get("quote", [{}])[0]
    adj = ((r.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")

    def arr(v):
        return np.array([np.nan if x is None else float(x) for x in (v or [None] * len(ts))], dtype=np.float64)

    out = {"dates": dates, "open": arr(q.get("open")), "high": arr(q.get("high")), "low": arr(q.get("low")),
           "close": arr(q.get("close")), "volume": arr(q.get("volume")),
           "adjclose": arr(adj) if adj else arr(q.get("close")),
           "exchange": str(meta.get("exchangeName") or meta.get("fullExchangeName") or "UNKNOWN")}
    ev = r.get("events") or {}
    divs = ev.get("dividends") or {}
    spl = ev.get("splits") or {}
    out["div_dates"] = np.array([dt.datetime.fromtimestamp(int(v["date"]) + off, dt.timezone.utc).date().isoformat()
                                 for v in divs.values()], dtype="datetime64[D]")
    out["div_amount"] = np.array([float(v.get("amount") or 0) for v in divs.values()], dtype=np.float64)
    out["split_dates"] = np.array([dt.datetime.fromtimestamp(int(v["date"]) + off, dt.timezone.utc).date().isoformat()
                                   for v in spl.values()], dtype="datetime64[D]")
    out["split_ratio"] = np.array([float(v.get("numerator") or 1) / float(v.get("denominator") or 1)
                                   for v in spl.values()], dtype=np.float64)
    # de-duplicate dates (Yahoo sometimes repeats the live bar)
    _, idx = np.unique(out["dates"], return_index=True)
    for k in ("dates", "open", "high", "low", "close", "volume", "adjclose"):
        out[k] = out[k][idx]
    return out


async def _fetch(client: httpx.AsyncClient, sym: str, start: str, end: str, sem: asyncio.Semaphore,
                 retries: int = 6) -> dict | None:
    params = {"period1": _to_epoch(start), "period2": _to_epoch(end), "interval": "1d", "events": "div,splits",
              "includeAdjustedClose": "true"}
    delay = 1.0
    for attempt in range(retries):
        url = (CHART if attempt % 2 == 0 else CHART_ALT).format(sym=sym)
        async with sem:
            try:
                r = await client.get(url, params=params)
            except httpx.HTTPError:
                r = None
        if r is not None and r.status_code == 200:
            try:
                return parse_chart(r.json())
            except ValueError:
                return None
        if r is not None and r.status_code == 404:
            return None
        await asyncio.sleep(delay + random.random())
        delay = min(30.0, delay * 2)
    return None


def save_raw(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in d.items() if k != "exchange"},
                        exchange=np.array([d.get("exchange", "UNKNOWN")]))


def load_raw(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["exchange"] = str(d["exchange"][0]) if "exchange" in d else "UNKNOWN"
        return d
    except (OSError, ValueError, KeyError):
        return None


def merge(old: dict, new: dict) -> dict:
    """Append new bars (new wins on overlap); events are unioned."""
    keep = old["dates"] < new["dates"][0]
    out = {}
    for k in ("dates", "open", "high", "low", "close", "volume", "adjclose"):
        out[k] = np.concatenate([old[k][keep], new[k]])
    for dk, vk in (("div_dates", "div_amount"), ("split_dates", "split_ratio")):
        dd = np.concatenate([old.get(dk, np.array([], "datetime64[D]")), new.get(dk, np.array([], "datetime64[D]"))])
        vv = np.concatenate([old.get(vk, np.array([])), new.get(vk, np.array([]))])
        _, idx = np.unique(dd, return_index=True)
        out[dk], out[vk] = dd[idx], vv[idx]
    out["exchange"] = new.get("exchange") or old.get("exchange", "UNKNOWN")
    # a new split changes the whole adjusted history -> caller should refetch fully
    return out


def download_all(tickers: list[str], start: str, end: str, raw_dir: Path, concurrency: int = 4,
                 progress: Callable[[int, int, str], None] | None = None, refresh_days: int = 1,
                 cancelled: Callable[[], bool] | None = None) -> dict[str, dict]:
    """Fetch/update every ticker; returns {ticker: data}. Incremental: only bars after the cached last date."""
    out: dict[str, dict] = {}
    today = dt.date.today()

    async def run():
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": UA}, follow_redirects=True) as client:
            done = 0

            async def one(t: str):
                nonlocal done
                if cancelled and cancelled():
                    return
                path = raw_dir / f"{t}.npz"
                old = load_raw(path)
                data = None
                if old is not None and len(old["dates"]):
                    last = old["dates"][-1].astype(dt.date)
                    if (today - last).days <= refresh_days:
                        data = old
                    else:
                        new = await _fetch(client, t, (last - dt.timedelta(days=7)).isoformat(), end, sem)
                        if new is not None and len(new["dates"]):
                            splits_new = new["split_dates"][new["split_dates"] > np.datetime64(last)] \
                                if len(new["split_dates"]) else []
                            if len(splits_new):
                                full = await _fetch(client, t, start, end, sem)
                                data = full or merge(old, new)
                            else:
                                data = merge(old, new)
                            save_raw(path, data)
                        else:
                            data = old
                else:
                    data = await _fetch(client, t, start, end, sem)
                    if data is not None:
                        save_raw(path, data)
                if data is not None and len(data["dates"]):
                    out[t] = data
                done += 1
                if progress:
                    progress(done, len(tickers), t)

            await asyncio.gather(*(one(t) for t in tickers))

    asyncio.run(run())
    return out
