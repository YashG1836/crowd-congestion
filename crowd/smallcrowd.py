"""Small crowds solved exactly: identical people, a few slots, every number exact.

Why this is exact. With n identical people, relabelling who is who turns a correlated
equilibrium (CE) into another CE with the same cost, and the average of all relabellings
is again a CE (the CE conditions are linear). That average is *exchangeable*: it draws a
head-count vector m = (m_0, .., m_{T-1}) and then hands the m_t "go to t" messages out at
random. So the best CE is a linear program over head-count vectors (C(n+T-1, T-1) of them,
1891 for 60 people and 3 slots) instead of over T^n profiles. A person told t is in a
head-count m with probability proportional to p(m) m_t, so obedience reads

    sum_m p(m) m_t [ c_t(m) - c_t'(m - e_t + e_t') ]  <=  0      for every t != t'.

The same holds for the Bayes CE of a bad-day game (one such LP per state, obedience pooled
over states). `full_profile_ce` solves the textbook LP over all T^n profiles for tiny n,
which the tests use to confirm the reduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import lgamma

import numpy as np
from scipy.optimize import linprog

from .game import FluidQueue


def head_counts(n: int, T: int) -> np.ndarray:
    """Every way to split n identical people over T slots, as rows of an (M, T) array."""
    if T == 1:
        return np.array([[n]])
    rows = [np.concatenate([[k], rest]) for k in range(n, -1, -1) for rest in head_counts(n - k, T - 1)]
    return np.array(rows, dtype=int)


@dataclass
class SmallCrowd:
    """n identical people, T slots of `slot_minutes`, all preferring slot `pref`.

    The bottleneck serves mu = n / (load * T * slot_minutes) people per minute, so `load` is
    the share of the window's capacity that the crowd needs (the same at every n).
    `capacity_factor` < 1 models a bad day.
    """

    n: int
    T: int = 3
    slot_minutes: float = 10.0
    pref: int = 1
    beta: float = 2.0
    load: float = 0.8
    capacity_factor: float = 1.0
    M: np.ndarray = field(init=False, repr=False)  # head-count vectors (M, T)
    cost: np.ndarray = field(init=False, repr=False)  # (M, T): cost of a person in slot t under head-count m
    dev: np.ndarray = field(init=False, repr=False)  # (M, T, T): cost after moving alone from t to t'

    def __post_init__(self) -> None:
        mu = self.n / (self.load * self.T * self.slot_minutes) * self.capacity_factor
        self.queue = FluidQueue(mu, self.T, self.slot_minutes)
        self.M = head_counts(self.n, self.T)
        sched = self.beta * np.abs(np.arange(self.T) - self.pref)
        self.sched = sched
        self.cost = self.queue.wait(self.M.astype(float)) + sched[None, :]
        E = np.eye(self.T)
        moved = self.M[:, None, None, :] - E[None, :, None, :] + E[None, None, :, :]  # (M, t, t', T)
        W = self.queue.wait(np.maximum(moved, 0).astype(float))  # waits after the move
        tt = np.arange(self.T)
        self.dev = W[:, tt[:, None], tt[None, :], tt[None, :]] + sched[None, None, :]

    # ------------------------------------------------------------ benchmarks
    def social(self) -> np.ndarray:
        """Average cost per person of every head-count."""
        return (self.M * self.cost).sum(axis=1) / self.n

    def gains(self) -> np.ndarray:
        """(M, T, T): what a person in t gains by moving alone to t' (NaN if nobody is in t)."""
        g = self.cost[:, :, None] - self.dev
        return np.where(self.M[:, :, None] > 0, g, np.nan)

    def pure_nash(self, tol: float = 1e-9) -> np.ndarray:
        """Indices of the head-counts that are pure Nash equilibria."""
        g = np.nan_to_num(self.gains(), nan=-np.inf)
        return np.flatnonzero(g.max(axis=(1, 2)) <= tol)

    def optimum(self) -> tuple[float, np.ndarray]:
        s = self.social()
        j = int(np.argmin(s))
        return float(s[j]), self.M[j]

    def best_ce(self, coarse: bool = False) -> tuple[float, np.ndarray]:
        """Exact best (coarse) correlated equilibrium: (cost per person, p over head-counts)."""
        A_ub = self._obedience_rows(coarse)
        res = linprog(self.social(), A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=np.ones((1, len(self.M))),
                      b_eq=[1.0], bounds=[(0, None)] * len(self.M), method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        return float(res.fun), res.x

    def _obedience_rows(self, coarse: bool = False) -> np.ndarray:
        g = np.nan_to_num(self.gains(), nan=0.0) * self.M[:, :, None]  # (M, t, t') weighted by m_t
        if coarse:  # decide before the message: one row per fixed alternative t'
            return g.sum(axis=1).T
        off = ~np.eye(self.T, dtype=bool)
        return g[:, off].T

    def mixed_nash(self, grid: int = 200) -> tuple[float, np.ndarray]:
        """Symmetric mixed equilibrium (everyone randomises the same way), by grid + refinement."""
        others = head_counts(self.n - 1, self.T)
        idx = {tuple(r): i for i, r in enumerate(self.M)}
        # cost of joining slot t when the others' head-count is m'
        join = np.array([[self.cost[idx[tuple(r + np.eye(self.T, dtype=int)[t])], t] for t in range(self.T)]
                         for r in others])
        logc = np.array([lgamma(self.n) - sum(lgamma(k + 1) for k in r) for r in others])

        def expected(q):
            q = np.clip(q, 1e-300, 1)
            logp = logc + (others * np.log(q)[None, :]).sum(axis=1)
            return np.exp(logp) @ join  # (T,)

        def gap(q):
            e = expected(q)
            return float(q @ e - e.min()), e

        best = (np.inf, None)
        pts = [np.array(p) / grid for p in product(range(grid + 1), repeat=self.T - 1) if sum(p) <= grid]
        for p in pts:
            q = np.append(p, 1 - p.sum())
            gp, _ = gap(q)
            if gp < best[0]:
                best = (gp, q)
        q = best[1]
        step = 1.0 / grid
        for _ in range(60):  # local refinement on a shrinking grid
            improved = False
            for d in product((-1, 0, 1), repeat=self.T - 1):
                cand = q.copy()
                cand[:-1] += step * np.array(d)
                cand[-1] = 1 - cand[:-1].sum()
                if cand.min() < 0:
                    continue
                gp, _ = gap(cand)
                if gp < best[0] - 1e-15:
                    best, q, improved = (gp, cand), cand, True
            if not improved:
                step /= 2
            if step < 1e-7:
                break
        e = expected(q)
        self.mixed_gap = float(q @ e - e.min())  # 0 at an exact symmetric equilibrium
        return float(e.min()), q


# ============================================================================
# Bad days in a small crowd: who should be told what, exactly.
# ============================================================================
@dataclass
class SmallBadDay:
    tell_nobody: float  # best CE when nobody learns the state (advice cannot depend on it)
    announce: float  # best CE of each state, averaged
    best_hint: float  # best public hint, each message followed by the best CE of its belief
    private: float  # best Bayes correlated equilibrium (advice may depend on the state)
    beliefs: np.ndarray
    value: np.ndarray  # best CE cost when everyone believes "bad" with probability q

    @property
    def privacy_gain(self) -> float:
        """What private, state-dependent advice saves over the best public hint (share)."""
        return 1 - self.private / self.best_hint


def small_bad_day(n: int, bad_factor: float = 0.7, p_bad: float = 0.3, n_beliefs: int = 41, **kw) -> SmallBadDay:
    good = SmallCrowd(n, **kw)
    bad = SmallCrowd(n, capacity_factor=bad_factor, **kw)

    def belief_ce(q: float) -> float:
        mix = SmallCrowd(n, **kw)
        mix.cost = (1 - q) * good.cost + q * bad.cost
        mix.dev = (1 - q) * good.dev + q * bad.dev
        return mix.best_ce()[0]

    q = np.unique(np.r_[np.linspace(0, 1, n_beliefs), p_bad])
    V = np.array([belief_ce(x) for x in q])
    ip = int(np.flatnonzero(np.isclose(q, p_bad))[0])
    best = V[ip]
    for i in range(ip + 1):  # best chord through p (two messages suffice with two states)
        for j in range(ip, len(q)):
            if i != j:
                lam = (q[j] - p_bad) / (q[j] - q[i])
                best = min(best, lam * V[i] + (1 - lam) * V[j])
    # private advice: one distribution over head-counts per state, obedience pooled over states
    Mn = len(good.M)
    c = np.r_[(1 - p_bad) * good.social(), p_bad * bad.social()]
    A_ub = np.hstack([(1 - p_bad) * good._obedience_rows(), p_bad * bad._obedience_rows()])
    A_eq = np.zeros((2, 2 * Mn))
    A_eq[0, :Mn] = 1
    A_eq[1, Mn:] = 1
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=[1, 1],
                  bounds=[(0, None)] * (2 * Mn), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    announce = (1 - p_bad) * good.best_ce()[0] + p_bad * bad.best_ce()[0]
    return SmallBadDay(float(V[ip]), float(announce), float(best), float(res.fun), q, V)


# ============================================================================
# The textbook CE linear program over all T^n profiles (only for tiny n: a check).
# ============================================================================
def full_profile_ce(crowd: SmallCrowd) -> float:
    n, T = crowd.n, crowd.T
    profiles = np.array(list(product(range(T), repeat=n)))  # (T^n, n)
    counts = np.stack([(profiles == t).sum(axis=1) for t in range(T)], axis=1)
    idx = {tuple(r): i for i, r in enumerate(crowd.M)}
    rows_m = np.array([idx[tuple(r)] for r in counts])
    social = crowd.social()[rows_m]
    A = []
    for i in range(n):
        for t in range(T):
            for t2 in range(T):
                if t2 == t:
                    continue
                told = profiles[:, i] == t
                g = np.zeros(len(profiles))
                g[told] = crowd.cost[rows_m[told], t] - crowd.dev[rows_m[told], t, t2]
                A.append(g)
    A = np.array(A)
    res = linprog(social, A_ub=A, b_ub=np.zeros(len(A)), A_eq=np.ones((1, len(profiles))), b_eq=[1.0],
                  bounds=[(0, None)] * len(profiles), method="highs")
    return float(res.fun)


def large_crowd_bad_day(bad_factor: float = 0.7, p_bad: float = 0.3, grid: int = 100, n_beliefs: int = 41,
                        tol: float = 1e-4, T: int = 3, slot_minutes: float = 10.0, pref: int = 1, beta: float = 2.0,
                        load: float = 0.8) -> SmallBadDay:
    """The same bad-day game when the crowd is large (every person negligible).

    A schedule is a split of the crowd over the slots. The menu holds every split on a 1/grid lattice
    plus the exact (Wardrop) equilibrium of every belief, so silence, announcing and every public hint
    are feasible; a person who deviates does not change the loads, so obedience compares slot costs at
    the same loads (with `tol` minutes of slack for numerical error). Refining the grid shows convergence.
    """
    from .game import Expected, SlotGame, large_crowd_equilibrium

    N = 1000  # per-person results do not depend on N because the service rate scales with it
    mu = N / (load * T * slot_minutes)
    good, bad = FluidQueue(mu, T, slot_minutes), FluidQueue(mu * bad_factor, T, slot_minutes)
    sched = beta * np.abs(np.arange(T) - pref)
    q = np.unique(np.r_[np.linspace(0, 1, n_beliefs), p_bad])

    def equilibrium(b):
        cm = good if b <= 1e-12 else bad if b >= 1 - 1e-12 else Expected([good, bad], np.array([1 - b, b]))
        x, _ = large_crowd_equilibrium(SlotGame(cm, [pref], [beta], [N]), tol=1e-7, iters=20000)
        return x[0] / N

    menu = np.vstack([head_counts(grid, T) / grid] + [equilibrium(b)[None, :] for b in q])
    Cg = good.wait(menu * N) + sched[None, :]
    Cb = bad.wait(menu * N) + sched[None, :]
    off = ~np.eye(T, dtype=bool)

    def rows(C):  # obedience: share told t times (C_t - C_t' - tol) for every t != t'
        return (menu[:, :, None] * (C[:, :, None] - C[:, None, :] - tol))[:, off].T

    def best(C):
        res = linprog((menu * C).sum(1), A_ub=rows(C), b_ub=np.zeros(T * (T - 1)), A_eq=np.ones((1, len(C))),
                      b_eq=[1.0], bounds=[(0, None)] * len(C), method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        return float(res.fun)

    V = np.array([best((1 - x) * Cg + x * Cb) for x in q])
    ip = int(np.flatnonzero(np.isclose(q, p_bad))[0])
    hint = V[ip]
    for i in range(ip + 1):
        for j in range(ip, len(q)):
            if i != j:
                lam = (q[j] - p_bad) / (q[j] - q[i])
                hint = min(hint, lam * V[i] + (1 - lam) * V[j])
    Mn = len(menu)
    c = np.r_[(1 - p_bad) * (menu * Cg).sum(1), p_bad * (menu * Cb).sum(1)]
    A_ub = np.hstack([(1 - p_bad) * rows(Cg), p_bad * rows(Cb)])
    A_eq = np.zeros((2, 2 * Mn))
    A_eq[0, :Mn] = 1
    A_eq[1, Mn:] = 1
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=[1, 1],
                  bounds=[(0, None)] * (2 * Mn), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    announce = (1 - p_bad) * V[0] + p_bad * V[-1]
    return SmallBadDay(float(V[ip]), float(announce), float(hint), float(res.fun), q, V)


# ============================================================================
# Crowds of a few types (e.g. flexible and rigid people), still exact.
#
# Relabelling people *within* a type keeps a CE a CE, so the exact best CE is a linear program
# over per-type head-counts m[k, t]: obedience for a type-k person told t reads
#     sum_m p(m) m[k, t] ( c_k,t(m) - c_k,t'(m - e_t + e_t') )  <=  0.
# ============================================================================
@dataclass
class TypedCrowd:
    """Types given as (count, favourite slot, beta). Same queue as SmallCrowd (load = share of capacity used)."""

    types: list[tuple[int, int, float]]
    T: int = 3
    slot_minutes: float = 10.0
    load: float = 0.8

    def __post_init__(self) -> None:
        self.n = sum(c for c, _, _ in self.types)
        mu = self.n / (self.load * self.T * self.slot_minutes)
        self.queue = FluidQueue(mu, self.T, self.slot_minutes)
        per_type = [head_counts(c, self.T) for c, _, _ in self.types]
        grids = np.meshgrid(*[np.arange(len(h)) for h in per_type], indexing="ij")
        idx = np.stack([g.ravel() for g in grids], axis=1)
        self.M = np.stack([per_type[k][idx[:, k]] for k in range(len(self.types))], axis=1)  # (S, K, T)
        self.sched = np.array([[b * abs(t - pref) for t in range(self.T)] for _, pref, b in self.types])  # (K, T)
        A = self.M.sum(axis=1).astype(float)  # (S, T)
        W = self.queue.wait(A)
        self.cost = W[:, None, :] + self.sched[None, :, :]  # (S, K, T)
        E = np.eye(self.T)
        moved = A[:, None, None, :] - E[None, :, None, :] + E[None, None, :, :]  # (S, t, t', T)
        Wm = self.queue.wait(np.maximum(moved, 0))
        tt = np.arange(self.T)
        dest = Wm[:, tt[:, None], tt[None, :], tt[None, :]]  # (S, t, t'): wait at t' after one person moves t -> t'
        self.dev = dest[:, None, :, :] + self.sched[None, :, None, :]  # (S, K, t, t')

    def social(self) -> np.ndarray:
        return (self.M * self.cost).sum(axis=(1, 2)) / self.n

    def pure_nash(self, tol: float = 1e-9) -> np.ndarray:
        g = self.cost[:, :, :, None] - self.dev
        g = np.where(self.M[:, :, :, None] > 0, g, -np.inf)
        return np.flatnonzero(g.max(axis=(1, 2, 3)) <= tol)

    def best_ce(self) -> float:
        g = (self.cost[:, :, :, None] - self.dev) * self.M[:, :, :, None]  # (S, K, t, t')
        off = ~np.eye(self.T, dtype=bool)
        rows = g[:, :, off].reshape(len(self.M), -1).T
        res = linprog(self.social(), A_ub=rows, b_ub=np.zeros(rows.shape[0]), A_eq=np.ones((1, len(self.M))),
                      b_eq=[1.0], bounds=[(0, None)] * len(self.M), method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        return float(res.fun)


def crowd_shapes(sizes=(4, 6, 8, 10, 12, 16, 20, 30, 40)) -> list[dict]:
    """The exact small-crowd comparison for several crowd shapes."""
    shapes = {
        "base: 3 slots, load 0.8, beta 2": dict(),
        "lighter load (0.6)": dict(load=0.6),
        "full load (1.0)": dict(load=1.0),
        "more flexible (beta 1)": dict(beta=1.0),
        "more rigid (beta 4)": dict(beta=4.0),
        "4 slots": dict(T=4),
    }
    out = []
    for name, kw in shapes.items():
        for n in sizes:
            c = SmallCrowd(n, **kw)
            ne = c.pure_nash()
            ce = c.best_ce()[0]
            mixed = c.mixed_nash(grid=60 if c.T == 3 else 24)[0]
            ok = c.mixed_gap < 1e-4  # keep the mixed equilibrium only when it was found exactly
            out.append({"shape": name, "n": n, "pure_nash": float(c.social()[ne].min()) if len(ne) else None,
                        "best_ce": ce, "mixed": mixed if ok else None})
    two = {"2 types: flexible and rigid": lambda n: [(n // 2, 1, 1.0), (n - n // 2, 1, 4.0)],
           "2 groups: middle and late": lambda n: [(n // 2, 1, 2.0), (n - n // 2, 2, 2.0)]}
    for name, make in two.items():
        for n in sizes:
            c = TypedCrowd(make(n))
            ne = c.pure_nash()
            out.append({"shape": name, "n": n, "pure_nash": float(c.social()[ne].min()) if len(ne) else None,
                        "best_ce": c.best_ce(), "mixed": None})
    return out
