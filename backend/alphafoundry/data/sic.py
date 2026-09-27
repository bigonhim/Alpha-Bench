"""SIC -> GICS-like classification for stocks outside the S&P 1500.

The S&P 1500 names carry GICS labels (Wikipedia) and SIC codes (SEC). The crosswalk is learned from them:
each 4-digit SIC code maps to the GICS (sector, industry, sub-industry) most S&P names with that code
have; unseen codes fall back to the 3-digit, then the 2-digit majority, and finally to a static SIC
division -> GICS sector table, with SIC-based industry and sub-industry labels so those names still form
their own peer groups.
"""

from __future__ import annotations

from collections import Counter

# (first SIC code, last SIC code, GICS sector) - checked in order, the first match wins
SECTOR_RANGES: list[tuple[int, int, str]] = [
    (2833, 2836, "Health Care"), (3841, 3851, "Health Care"), (5047, 5047, "Health Care"), (5122, 5122, "Health Care"),
    (8000, 8099, "Health Care"), (8731, 8734, "Health Care"),
    (3570, 3579, "Information Technology"), (3600, 3699, "Information Technology"),
    (3820, 3829, "Information Technology"), (7370, 7379, "Information Technology"), (5045, 5045, "Information Technology"),
    (3711, 3716, "Consumer Discretionary"), (3751, 3751, "Consumer Discretionary"), (1531, 1531, "Consumer Discretionary"),
    (2200, 2399, "Consumer Discretionary"), (2500, 2599, "Consumer Discretionary"), (3100, 3199, "Consumer Discretionary"),
    (3900, 3999, "Consumer Discretionary"), (5200, 5399, "Consumer Discretionary"), (5500, 5799, "Consumer Discretionary"),
    (5900, 5911, "Consumer Discretionary"), (5913, 5999, "Consumer Discretionary"), (7000, 7099, "Consumer Discretionary"),
    (7200, 7299, "Consumer Discretionary"), (7500, 7599, "Consumer Discretionary"), (7900, 7999, "Consumer Discretionary"),
    (8200, 8299, "Consumer Discretionary"),
    (100, 999, "Consumer Staples"), (2000, 2199, "Consumer Staples"), (5140, 5149, "Consumer Staples"),
    (5400, 5499, "Consumer Staples"), (5912, 5912, "Consumer Staples"),
    (1200, 1399, "Energy"), (2900, 2999, "Energy"), (4610, 4619, "Energy"), (4920, 4923, "Energy"),
    (1000, 1099, "Materials"), (1400, 1499, "Materials"), (2400, 2499, "Materials"), (2600, 2699, "Materials"),
    (2800, 2832, "Materials"), (2837, 2899, "Materials"), (3000, 3099, "Materials"), (3200, 3399, "Materials"),
    (2700, 2799, "Communication Services"), (4800, 4899, "Communication Services"), (7800, 7899, "Communication Services"),
    (4900, 4919, "Utilities"), (4924, 4952, "Utilities"), (4954, 4999, "Utilities"),
    (6500, 6599, "Real Estate"), (6798, 6798, "Real Estate"),
    (6000, 6799, "Financials"),
    (1500, 1799, "Industrials"), (3400, 3569, "Industrials"), (3580, 3599, "Industrials"), (3700, 3799, "Industrials"),
    (3800, 3819, "Industrials"), (3830, 3840, "Industrials"), (4000, 4799, "Industrials"), (4953, 4953, "Industrials"),
    (5000, 5199, "Industrials"), (7300, 7369, "Industrials"), (7380, 7399, "Industrials"), (8700, 8999, "Industrials"),
    (9000, 9999, "Industrials"),
]

# SIC codes that are not operating companies (blank checks, funds, trusts)
NON_OPERATING_SIC = {6770, 6722, 6726, 6221, 6795, 9995}


def sector_of_sic(sic: int | None) -> str:
    if sic is None:
        return "Unknown"
    for lo, hi, sector in SECTOR_RANGES:
        if lo <= sic <= hi:
            return sector
    return "Unknown"


def _majority(pairs: list[tuple[str, tuple[str, str, str]]]) -> dict[str, tuple[str, str, str]]:
    by: dict[str, Counter] = {}
    for key, gics in pairs:
        by.setdefault(key, Counter())[gics] += 1
    # ties resolve to the alphabetically first label so the build is deterministic
    return {k: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, c in by.items()}


class Crosswalk:
    """Learned SIC -> (sector, industry, subindustry)."""

    def __init__(self, known: list[tuple[int, str, str, str]]):
        """known: (sic, sector, industry, subindustry) for names that have both classifications."""
        rows = [(int(s), (sec, ind, sub)) for s, sec, ind, sub in known if s and sec and sec != "Unknown"]
        self.by4 = _majority([(f"{s:04d}", g) for s, g in rows])
        self.by3 = _majority([(f"{s:04d}"[:3], g) for s, g in rows])
        self.by2 = _majority([(f"{s:04d}"[:2], g) for s, g in rows])

    def classify(self, sic: int | None, description: str = "") -> tuple[tuple[str, str, str], str]:
        """((sector, industry, subindustry), how) where how is sic4 | sic3 | sic2 | division | none."""
        if sic is None:
            return ("Unknown", "Unknown", "Unknown"), "none"
        k = f"{int(sic):04d}"
        if k in self.by4:
            return self.by4[k], "sic4"
        if k[:3] in self.by3:
            return self.by3[k[:3]], "sic3"
        sector = sector_of_sic(int(sic))
        if k[:2] in self.by2 and self.by2[k[:2]][0] == sector:
            return self.by2[k[:2]], "sic2"
        label = f"SIC {k} {description}".strip()
        return (sector, f"SIC {k[:2]}xx ({sector})", label), "division"


__all__ = ["Crosswalk", "NON_OPERATING_SIC", "SECTOR_RANGES", "sector_of_sic"]
