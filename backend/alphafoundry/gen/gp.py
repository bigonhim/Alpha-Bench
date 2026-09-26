"""Genetic programming for alpha expressions: variation operators and NSGA-II selection.

The GP core is independent of how candidates are evaluated: callers pass ``evaluate(batch, span)``
which returns metric dicts (sharpe, fitness, turnover, max_weight, pnl ...). This keeps the loop
usable both in-process and with the process pool.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from ..fastexpr.ast import Const, Field, Node, Op, iter_paths, node_at, replace_at
from ..fastexpr.lower import simplify
from ..fastexpr.printer import to_expr
from .build import GROUP_NAMES, arg_kind, capbucket, mk, node_type
from .grammar import FAMILY_OF, WINDOWS, Grammar, is_degenerate

NEUTS = ("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")
DECAYS = (0, 2, 3, 4, 5, 6, 8, 10, 15, 20)
TRUNCS = (0.01, 0.05, 0.08, 0.1)


@dataclass
class Individual:
    node: Node
    settings: dict
    origin: str = "gp"
    parents: list = field(default_factory=list)
    metrics: dict | None = None
    screened: bool = False
    novelty: float = 1.0
    objectives: tuple = ()
    violation: float = 0.0
    rank: int = 0
    crowd: float = 0.0
    alpha_id: int | None = None
    island: int = 0

    @property
    def key(self) -> str:
        s = self.settings
        return (f"{self.node.key}|{s.get('decay')}|{s.get('neutralization')}|{s.get('truncation')}|"
                f"{s.get('delay')}|{s.get('universe')}")

    @property
    def expr(self) -> str:
        return to_expr(self.node)


# --------------------------------------------------------------------------- variation


class Variation:
    def __init__(self, grammar: Grammar, rng: random.Random, max_depth: int = 6, max_nodes: int = 30):
        self.g = grammar
        self.rng = rng
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def ok(self, n: Node) -> bool:
        return n.depth <= self.max_depth and n.size <= self.max_nodes and not is_degenerate(n)

    def _matrix_sites(self, n: Node) -> list[tuple[tuple[int, ...], Node]]:
        sites = []
        for path, sub in iter_paths(n):
            if node_type(sub) != "m" or isinstance(sub, Const):
                continue
            if path:
                parent = node_at(n, path[:-1])
                if isinstance(parent, Op) and arg_kind(parent, path[-1]) != "m":
                    continue
            sites.append((path, sub))
        return sites

    def point(self, n: Node) -> Node:
        ops = [(p, s) for p, s in iter_paths(n) if isinstance(s, Op) and s.name in FAMILY_OF]
        if not ops:
            return n
        path, sub = self.rng.choice(ops)
        alts = [o for o in FAMILY_OF[sub.name] if o != sub.name]
        new_name = self.rng.choice(alts)
        try:
            params = {k: v for k, v in sub.params}
            new = mk(new_name, *sub.args, **params)
        except (KeyError, TypeError):
            return n
        return replace_at(n, path, new)

    def window(self, n: Node) -> Node:
        sites = [(p, s) for p, s in iter_paths(n) if isinstance(s, Op) and any(k == "d" for k, _ in s.params)]
        if not sites:
            return n
        path, sub = self.rng.choice(sites)
        cur = int(sub.param("d"))
        factor = self.rng.choice((0.5, 0.67, 1.5, 2.0))
        target = min(WINDOWS, key=lambda w: abs(w - cur * factor))
        if target == cur:
            idx = WINDOWS.index(target) if target in WINDOWS else 0
            target = WINDOWS[min(len(WINDOWS) - 1, max(0, idx + self.rng.choice((-1, 1))))]
        return replace_at(n, path, sub.with_param("d", int(target)))

    def field_swap(self, n: Node) -> Node:
        sites = [(p, s) for p, s in iter_paths(n) if isinstance(s, Field) and s.name not in GROUP_NAMES]
        if not sites:
            return n
        path, sub = self.rng.choice(sites)
        meta = self.g.fields.get(sub.name, {})
        same = [f for f in self.g.pool if f[1] == meta.get("category") and f[2] == meta.get("unit") and f[0] != sub.name]
        if not same:
            same = [f for f in self.g.pool if f[1] == meta.get("category") and f[0] != sub.name]
        if not same:
            return n
        return replace_at(n, path, Field(self.rng.choice(same)[0]))

    def group_swap(self, n: Node) -> Node:
        sites = []
        for path, sub in iter_paths(n):
            if path and node_type(sub) == "g":
                sites.append((path, sub))
        if not sites:
            return n
        path, sub = self.rng.choice(sites)
        return replace_at(n, path, self.g.group())

    def subtree(self, n: Node) -> Node:
        sites = self._matrix_sites(n)
        if not sites:
            return n
        path, _ = self.rng.choice(sites)
        return replace_at(n, path, self.g.matrix(self.rng.randint(1, 3)))

    def hoist(self, n: Node) -> Node:
        sites = [(p, s) for p, s in self._matrix_sites(n) if p and isinstance(s, Op)]
        if not sites:
            return n
        _, sub = self.rng.choice(sites)
        return mk("rank", sub) if not (isinstance(sub, Op) and sub.name in ("rank", "group_rank")) else sub

    def wrap(self, n: Node) -> Node:
        choice = self.rng.random()
        if choice < 0.3:
            return mk("ts_decay_linear", n, d=self.rng.choice((3, 5, 10)))
        if choice < 0.55:
            return mk("group_neutralize", n, self.g.group())
        if choice < 0.75 and self.g.cfg.allow_trade_when:
            cond = mk("greater", Field("volume"), Field("adv20"))
            return mk("trade_when", cond, n, Const(-1.0))
        if choice < 0.9:
            return mk("ts_mean", n, d=self.rng.choice((3, 5)))
        return mk("rank", n) if not (isinstance(n, Op) and n.name == "rank") else mk("zscore", n)

    def crossover(self, a: Node, b: Node) -> Node:
        sa = self._matrix_sites(a)
        sb = [s for _, s in self._matrix_sites(b)]
        if not sa or not sb:
            return a
        path, _ = self.rng.choice(sa)
        return replace_at(a, path, self.rng.choice(sb))

    def settings(self, s: dict) -> dict:
        s = dict(s)
        r = self.rng.random()
        if r < 0.45:
            s["decay"] = self.rng.choice(DECAYS)
        elif r < 0.85:
            s["neutralization"] = self.rng.choice(NEUTS)
        else:
            s["truncation"] = self.rng.choice(TRUNCS)
        return s

    def mutate(self, ind: Individual) -> Individual:
        ops = [(self.point, 0.22), (self.window, 0.22), (self.field_swap, 0.14), (self.group_swap, 0.07),
               (self.subtree, 0.12), (self.hoist, 0.05), (self.wrap, 0.08)]
        settings = ind.settings
        node = ind.node
        for _ in range(6):
            if self.rng.random() < 0.15:
                settings = self.settings(ind.settings)
                cand = node
            else:
                fn = self.rng.choices([o for o, _ in ops], weights=[w for _, w in ops])[0]
                try:
                    cand = simplify(fn(node))
                except (AssertionError, KeyError, IndexError, TypeError):
                    continue
            if (cand.key != node.key or settings != ind.settings) and self.ok(cand):
                return Individual(cand, settings, origin="gp", parents=[ind.alpha_id] if ind.alpha_id else [],
                                  island=ind.island)
        return Individual(node, self.settings(ind.settings), origin="gp", island=ind.island,
                          parents=[ind.alpha_id] if ind.alpha_id else [])

    def mate(self, a: Individual, b: Individual) -> Individual:
        for _ in range(5):
            try:
                child = simplify(self.crossover(a.node, b.node))
            except (AssertionError, KeyError, IndexError, TypeError):
                continue
            if child.key not in (a.node.key, b.node.key) and self.ok(child):
                s = dict(a.settings if self.rng.random() < 0.5 else b.settings)
                return Individual(child, s, origin="gp",
                                  parents=[p for p in (a.alpha_id, b.alpha_id) if p], island=a.island)
        return self.mutate(a)


# --------------------------------------------------------------------------- NSGA-II


def dominates(a: Individual, b: Individual) -> bool:
    if a.violation < b.violation:
        return True
    if a.violation > b.violation:
        return False
    better = False
    for x, y in zip(a.objectives, b.objectives):
        if x < y:
            return False
        if x > y:
            better = True
    return better


def nondominated_sort(pop: list[Individual]) -> list[list[Individual]]:
    S: dict[int, list[int]] = {i: [] for i in range(len(pop))}
    n = [0] * len(pop)
    fronts: list[list[int]] = [[]]
    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j:
                continue
            if dominates(p, q):
                S[i].append(j)
            elif dominates(q, p):
                n[i] += 1
        if n[i] == 0:
            p.rank = 0
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    pop[j].rank = k + 1
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [[pop[i] for i in f] for f in fronts if f]


def assign_crowding(front: list[Individual]) -> None:
    if not front:
        return
    for p in front:
        p.crowd = 0.0
    m = len(front[0].objectives)
    for k in range(m):
        front.sort(key=lambda p: p.objectives[k])
        front[0].crowd = front[-1].crowd = math.inf
        lo, hi = front[0].objectives[k], front[-1].objectives[k]
        if hi - lo <= 1e-12:
            continue
        for i in range(1, len(front) - 1):
            front[i].crowd += (front[i + 1].objectives[k] - front[i - 1].objectives[k]) / (hi - lo)


def select_survivors(pop: list[Individual], n: int) -> list[Individual]:
    out: list[Individual] = []
    for front in nondominated_sort(pop):
        assign_crowding(front)
        if len(out) + len(front) <= n:
            out.extend(front)
        else:
            front.sort(key=lambda p: -p.crowd)
            out.extend(front[: n - len(out)])
            break
    return out


def tournament(pop: list[Individual], rng: random.Random, k: int = 3) -> Individual:
    cand = rng.sample(pop, min(k, len(pop)))
    return min(cand, key=lambda p: (p.rank, -p.crowd))


def set_objectives(ind: Individual, gates: dict) -> None:
    m = ind.metrics or {}
    fit = float(m.get("fitness", -5.0) or -5.0)
    sh = float(m.get("sharpe", -5.0) or -5.0)
    if ind.screened:
        fit *= 0.7
    to = float(m.get("turnover", 0.0) or 0.0)
    viol = 0.0
    if not m:
        viol += 10
    if to > gates.get("turnover_max", 0.7):
        viol += (to - gates.get("turnover_max", 0.7)) * 5
    if to < gates.get("turnover_min", 0.01):
        viol += (gates.get("turnover_min", 0.01) - to) * 50
    mw = float(m.get("max_weight", 0.0) or 0.0)
    if mw > gates.get("max_weight", 0.1):
        viol += (mw - gates.get("max_weight", 0.1)) * 10
    ind.violation = round(viol, 6)
    ind.objectives = (round(fit + 0.25 * max(-2.0, min(sh, 4.0)), 6), round(ind.novelty, 6), -ind.node.size / 30.0)


@dataclass
class GPConfig:
    population: int = 60
    generations: int = 20
    islands: int = 2
    migrate_every: int = 5
    crossover_prob: float = 0.45
    elite_hof: int = 50
    halving: bool = True
    promote_frac: float = 0.4
    max_depth: int = 6
    max_nodes: int = 30
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    corr_cap: float = 0.7
    seed: int = 0


class GPEngine:
    """Island-model NSGA-II. ``evaluate(list[Individual], span)`` fills ``ind.metrics`` in place."""

    def __init__(self, cfg: GPConfig, grammar: Grammar, evaluate: Callable[[list[Individual], str], None],
                 novelty: Callable[[list[Individual]], None], gates: dict,
                 on_generation: Callable[[dict], None] | None = None, stop: Callable[[], bool] | None = None,
                 on_hof: Callable[[Individual], None] | None = None):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.grammar = grammar
        self.var = Variation(grammar, self.rng, cfg.max_depth, cfg.max_nodes)
        self.evaluate = evaluate
        self.novelty = novelty
        self.gates = gates
        self.on_generation = on_generation
        self.stop = stop or (lambda: False)
        self.on_hof = on_hof
        self.seen: set[str] = set()
        self.hof: list[Individual] = []
        self.evals = 0
        self.history: list[dict] = []

    def _fresh(self, inds: list[Individual]) -> list[Individual]:
        out = []
        for i in inds:
            if i.key in self.seen:
                continue
            self.seen.add(i.key)
            out.append(i)
        return out

    def _score(self, inds: list[Individual]) -> None:
        if not inds:
            return
        if self.cfg.halving and len(inds) >= 6:
            self.evaluate(inds, "screen")
            for i in inds:
                i.screened = True
            ranked = sorted(inds, key=lambda i: -(i.metrics or {}).get("fitness", -9) if i.metrics else 9)
            k = max(1, int(len(inds) * self.cfg.promote_frac))
            promote = [i for i in ranked[:k] if i.metrics]
            self.evaluate(promote, "is")
            for i in promote:
                i.screened = False
        else:
            self.evaluate(inds, "is")
        self.evals += len(inds)
        self.novelty(inds)
        for i in inds:
            set_objectives(i, self.gates)
            self._consider_hof(i)

    def _consider_hof(self, ind: Individual) -> None:
        m = ind.metrics or {}
        if ind.screened or not m:
            return
        if m.get("sharpe", 0) < self.cfg.min_sharpe or m.get("fitness", 0) < self.cfg.min_fitness:
            return
        if ind.violation > 0:
            return
        if ind.novelty < 1 - self.cfg.corr_cap:
            return
        self.hof.append(ind)
        self.hof.sort(key=lambda i: -(i.metrics or {}).get("fitness", 0))
        self.hof = self.hof[: self.cfg.elite_hof]
        if self.on_hof:
            self.on_hof(ind)

    def run(self, seeds: list[Individual]) -> list[Individual]:
        n_isl = max(1, self.cfg.islands)
        per = max(4, self.cfg.population // n_isl)
        islands: list[list[Individual]] = [[] for _ in range(n_isl)]
        seeds = self._fresh(seeds)
        for k, s in enumerate(seeds):
            s.island = k % n_isl
            islands[s.island].append(s)
        for isl in range(n_isl):
            tries = 0
            while len(islands[isl]) < per and tries < per * 20:
                tries += 1
                node = self.grammar.tree()
                if is_degenerate(node):
                    continue
                s = dict(seeds[0].settings if seeds else {})
                s = self.var.settings(s) if self.rng.random() < 0.5 else s
                ind = Individual(node, s, origin="gp", island=isl)
                if ind.key in self.seen:
                    continue
                self.seen.add(ind.key)
                islands[isl].append(ind)
        self._score([i for isl in islands for i in isl])
        islands = [select_survivors(isl, per) for isl in islands]
        for gen in range(self.cfg.generations):
            if self.stop():
                break
            offspring_all: list[Individual] = []
            for isl_idx, isl in enumerate(islands):
                off: list[Individual] = []
                tries = 0
                while len(off) < per and tries < per * 6:
                    tries += 1
                    a = tournament(isl, self.rng)
                    if self.rng.random() < self.cfg.crossover_prob and len(isl) > 1:
                        b = tournament(isl, self.rng)
                        child = self.var.mate(a, b)
                    else:
                        child = self.var.mutate(a)
                    child.island = isl_idx
                    if child.key in self.seen:
                        continue
                    self.seen.add(child.key)
                    off.append(child)
                offspring_all.extend(off)
            self._score(offspring_all)
            for isl_idx in range(n_isl):
                pool = islands[isl_idx] + [o for o in offspring_all if o.island == isl_idx]
                islands[isl_idx] = select_survivors(pool, per)
            if n_isl > 1 and (gen + 1) % self.cfg.migrate_every == 0:
                bests = [sorted(isl, key=lambda p: (p.rank, -p.objectives[0] if p.objectives else 0))[:2]
                         for isl in islands]
                for k in range(n_isl):
                    migrants = bests[(k - 1) % n_isl]
                    for m in migrants:
                        clone = Individual(m.node, m.settings, m.origin, m.parents, m.metrics, m.screened,
                                           m.novelty, m.objectives, m.violation, alpha_id=m.alpha_id, island=k)
                        islands[k].append(clone)
                    islands[k] = select_survivors(islands[k], per)
            allpop = [i for isl in islands for i in isl if i.metrics]
            fits = sorted([(i.metrics or {}).get("fitness", 0.0) for i in allpop if not i.screened], reverse=True)
            stat = {"generation": gen + 1, "evaluations": self.evals, "hof": len(self.hof),
                    "best_fitness": round(fits[0], 3) if fits else None,
                    "median_fitness": round(fits[len(fits) // 2], 3) if fits else None,
                    "front": [{"sharpe": (i.metrics or {}).get("sharpe"), "turnover": (i.metrics or {}).get("turnover"),
                               "fitness": (i.metrics or {}).get("fitness"), "novelty": round(i.novelty, 3),
                               "expr": i.expr[:120]}
                              for i in allpop if i.rank == 0][:60]}
            self.history.append({k: v for k, v in stat.items() if k != "front"})
            if self.on_generation:
                self.on_generation(stat)
        return self.hof
