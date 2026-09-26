"""Universe pool: S&P 500 + 400 + 600 constituents (Wikipedia) with GICS labels and SEC CIKs."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pandas as pd

from .gics import industry_of

WIKI = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
# Wikimedia's robot policy rejects agents without contact details (HTTP 403), so the tool identifies
# itself and includes the same contact address the user provides for SEC EDGAR.
WIKI_UA = "AlphaFoundry/1.0 (personal quantitative research tool; contact: {email}) python-httpx"


def norm_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-").replace(" ", "")


def _pick_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c or "ticker" in c for c in cols) and any("gics sector" in c or c == "sector" for c in cols):
            return t
    raise ValueError("constituents table not found")


def _col(df: pd.DataFrame, *names: str) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        for n in names:
            if n in lc:
                return c
    return None


def fetch_universe(contact_email: str, sec_user_agent: str, cache: Path | None = None) -> pd.DataFrame:
    rows = []
    wiki_ua = WIKI_UA.format(email=contact_email)
    with httpx.Client(timeout=30, headers={"User-Agent": wiki_ua}, follow_redirects=True) as c:
        for index, url in WIKI.items():
            resp = c.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Wikipedia returned HTTP {resp.status_code} for {index} constituents")
            html = resp.text
            df = _pick_table(html)
            sym = _col(df, "symbol", "ticker")
            name = _col(df, "security", "company")
            sec = _col(df, "gics sector", "sector")
            sub = _col(df, "gics sub-industry", "sub-industry", "sub industry")
            cik = _col(df, "cik")
            for _, r in df.iterrows():
                t = norm_ticker(r[sym])
                if not t or t == "NAN":
                    continue
                rows.append({
                    "ticker": t, "name": str(r[name]) if name else t, "sector": str(r[sec]) if sec else "Unknown",
                    "subindustry": str(r[sub]) if sub else "Unknown",
                    "cik": int(r[cik]) if cik and pd.notna(r[cik]) and str(r[cik]).strip().isdigit() else None,
                    "index": index,
                })
        sr = c.get(SEC_TICKERS, headers={"User-Agent": sec_user_agent})
        if sr.status_code != 200:
            raise RuntimeError(f"SEC returned HTTP {sr.status_code} for the ticker map (check the contact email)")
        sec_map = sr.json()
    by_ticker = {norm_ticker(v["ticker"]): int(v["cik_str"]) for v in sec_map.values()}
    df = pd.DataFrame(rows).drop_duplicates("ticker", keep="first")
    # Wikipedia only lists CIKs for the S&P 500; missing values arrive as NaN (truthy!), so test explicitly
    df["cik"] = [int(c) if (c is not None and c == c) else by_ticker.get(t)
                 for t, c in zip(df["ticker"], df["cik"])]
    df["industry"] = [industry_of(s) for s in df["subindustry"]]
    df = df.sort_values("ticker").reset_index(drop=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(df.to_json(orient="records"), encoding="utf-8")
    return df


def load_cached_universe(cache: Path) -> pd.DataFrame | None:
    if not cache.exists():
        return None
    return pd.DataFrame(json.loads(cache.read_text(encoding="utf-8")))
