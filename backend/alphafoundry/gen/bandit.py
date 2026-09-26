"""Thompson-sampling bandit over search arms (template families, data categories, GP islands)."""

from __future__ import annotations

import random

from ..store.db import Store


class Bandit:
    def __init__(self, store: Store, rng: random.Random | None = None):
        self.store = store
        self.rng = rng or random.Random()

    def sample(self, arms: list[str]) -> str:
        stats = self.store.bandit_all()
        best, best_v = arms[0], -1.0
        for a in arms:
            al, be, _ = stats.get(a, (1.0, 1.0, 0))
            v = self.rng.betavariate(max(al, 1e-3), max(be, 1e-3))
            if v > best_v:
                best, best_v = a, v
        return best

    def weights(self, arms: list[str], draws: int = 200) -> dict[str, float]:
        """Share of Thompson draws each arm would win (used to split a budget)."""
        wins = {a: 0 for a in arms}
        for _ in range(draws):
            wins[self.sample(arms)] += 1
        return {a: w / draws for a, w in wins.items()}

    def update(self, arm: str, reward: float, weight: float = 1.0) -> None:
        self.store.bandit_update(arm, max(0.0, min(1.0, reward)), weight)

    def table(self) -> list[dict]:
        out = []
        for arm, (a, b, n) in sorted(self.store.bandit_all().items()):
            out.append({"arm": arm, "alpha": round(a, 2), "beta": round(b, 2), "n": n,
                        "mean": round(a / (a + b), 3)})
        return out
