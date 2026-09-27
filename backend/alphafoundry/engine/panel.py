"""Panel store: one memory-mapped float32 .npy per field (dates x instruments) plus metadata.

Layout of a panel directory::

    meta.json                 dates, tickers, groups labels, universes, version, source
    <field>.npy               float32 (T, N)
    group_<name>.npy          int32 (N,) static codes (-1 = unknown)
    univ_<name>.npy           uint8 (T, N) membership masks (TOP100 ... TOP1500)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

UNIVERSE_SIZES = {"TOP100": 100, "TOP200": 200, "TOP500": 500, "TOP1000": 1000, "TOP1500": 1500, "TOP2000": 2000,
                  "TOP3000": 3000}


class Panel:
    def __init__(self, root: Path):
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.version: str = meta["version"]
        self.source: str = meta.get("source", "unknown")
        self.dates = np.array(meta["dates"], dtype="datetime64[D]")
        self.tickers: list[str] = list(meta["tickers"])
        self.names: list[str] = list(meta.get("names", self.tickers))
        self.T = len(self.dates)
        self.N = len(self.tickers)
        self.group_labels: dict[str, list[str]] = meta.get("groups", {})
        self.universes: list[str] = list(meta.get("universes", []))
        self._mm: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._field_names = sorted(p.stem for p in self.root.glob("*.npy")
                                   if not p.stem.startswith(("group_", "univ_")))

    # ------------------------------------------------------------------ access
    def field_names(self) -> list[str]:
        return list(self._field_names)

    def has_field(self, name: str) -> bool:
        return name in self._field_names or name in self.group_labels or name in ("market", "country")

    def _load(self, fname: str) -> np.ndarray:
        with self._lock:
            arr = self._mm.get(fname)
            if arr is None:
                arr = np.load(self.root / f"{fname}.npy", mmap_mode="r")
                self._mm[fname] = arr
            return arr

    def field(self, name: str) -> np.ndarray:
        if name not in self._field_names:
            raise KeyError(name)
        return self._load(name)

    def group(self, name: str) -> tuple[np.ndarray, int]:
        """Static codes (N,) and number of groups."""
        if name in ("market", "country"):
            return np.zeros(self.N, np.int32), 1
        labels = self.group_labels.get(name)
        if labels is None:
            raise KeyError(name)
        return np.asarray(self._load(f"group_{name}"), dtype=np.int32), max(1, len(labels))

    def universe(self, name: str) -> np.ndarray:
        if name not in self.universes:
            raise KeyError(f"universe {name} not available (have {self.universes})")
        return self._load(f"univ_{name}")

    # ------------------------------------------------------------------ dates
    def index_of(self, date: str | np.datetime64, side: str = "left") -> int:
        return int(np.searchsorted(self.dates, np.datetime64(date, "D"), side=side))

    def info(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "start": str(self.dates[0]) if self.T else None,
            "end": str(self.dates[-1]) if self.T else None,
            "T": self.T,
            "N": self.N,
            "fields": self.field_names(),
            "groups": {k: len(v) for k, v in self.group_labels.items()},
            "universes": self.universes,
            "built": self.meta.get("built"),
            "pool": self.meta.get("pool") or ("sp1500" if self.source == "real" else self.source),
            "classification": self.meta.get("classification"),
        }


def write_panel(root: Path, dates: list[str], tickers: list[str], fields: dict[str, np.ndarray],
                groups: dict[str, tuple[np.ndarray, list[str]]], universes: dict[str, np.ndarray],
                source: str, extra_meta: dict | None = None) -> None:
    """Write a complete panel atomically-ish: files first, meta.json last (it defines validity)."""
    import datetime as _dt
    import hashlib

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta_path.unlink()
    for old in root.glob("*.npy"):
        try:
            old.unlink()
        except PermissionError:
            pass
    h = hashlib.sha1()
    h.update(("|".join(dates[:1] + dates[-1:]) + str(len(dates)) + "|".join(tickers)).encode())
    for name, arr in sorted(fields.items()):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        assert a.shape == (len(dates), len(tickers)), (name, a.shape)
        np.save(root / f"{name}.npy", a)
        h.update(name.encode())
        h.update(a[::97, ::13].tobytes())
    glabels = {}
    for name, (codes, labels) in groups.items():
        np.save(root / f"group_{name}.npy", np.ascontiguousarray(codes, dtype=np.int32))
        glabels[name] = labels
    for name, mask in universes.items():
        np.save(root / f"univ_{name}.npy", np.ascontiguousarray(mask, dtype=np.uint8))
    meta = {
        "version": h.hexdigest()[:12],
        "source": source,
        "dates": dates,
        "tickers": tickers,
        "groups": glabels,
        "universes": sorted(universes, key=lambda u: UNIVERSE_SIZES.get(u, 10 ** 6)),
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
