"""Layer 1 - exact game theory on small games (normal form, CE/CCE linear programs, toy games, learning, delayed feedback)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog, minimize_scalar


# ============================================================================
# Small normal-form games: Nash, correlated and coarse correlated equilibria.
#
# Everything here is *exact* and meant for games small enough to list every
# action profile (a few players, a few actions each). We work with COSTS
# (lower is better) because congestion problems are about waiting time.
#
# Solution concepts
# -----------------
# Pure Nash equilibrium (NE)
#     No player can lower their own cost by changing only their own action.
# Mixed NE (two players)
#     Found by support enumeration: guess which actions each player uses, make
#     the opponent indifferent over them, keep the guess if nothing outside the
#     support is better.
# Correlated equilibrium (CE)  [Aumann 1974]
#     A mediator draws a joint profile s ~ p and privately tells each player i
#     only s_i. p is a CE if, for every player i and every pair of actions
#     (a, a'), being told a and obeying is no worse than switching to a':
#
#         sum_{s : s_i = a} p(s) * ( c_i(s) - c_i(a', s_-i) )  <=  0.
#
#     These are linear inequalities in p, so the set of CE is a polytope and
#     the best CE is a linear program (LP).
# Coarse correlated equilibrium (CCE)
#     Weaker: a player must decide to obey *before* seeing the recommendation.
#     For every i and every fixed action a':
#
#         sum_s p(s) * ( c_i(s) - c_i(a', s_-i) )  <=  0.
#
#     NE  is a subset of  CE  is a subset of  CCE.
# ============================================================================

@dataclass
class NormalFormGame:
    """A finite game given by a cost tensor.

    costs[i][s] is the cost of player i at action profile s, so ``costs`` has
    shape (n_players, m_1, ..., m_n).
    """

    costs: np.ndarray
    action_names: list[list[str]] | None = None
    player_names: list[str] | None = None
    name: str = "game"
    profiles: list[tuple[int, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.costs = np.asarray(self.costs, dtype=float)
        if self.costs.shape[0] != self.costs.ndim - 1:
            raise ValueError("costs must have shape (n_players, m_1, ..., m_n)")
        if self.action_names is None:
            self.action_names = [[f"a{j}" for j in range(m)] for m in self.shape]
        if self.player_names is None:
            self.player_names = [f"P{i + 1}" for i in range(self.n_players)]
        self.profiles = list(itertools.product(*[range(m) for m in self.shape]))

    # ---------------------------------------------------------------- basics
    @property
    def n_players(self) -> int:
        return self.costs.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.costs.shape[1:]

    def cost(self, i: int, profile: tuple[int, ...]) -> float:
        return float(self.costs[(i, *profile)])

    def total_cost(self, profile: tuple[int, ...]) -> float:
        return float(sum(self.cost(i, profile) for i in range(self.n_players)))

    def profile_name(self, profile: tuple[int, ...]) -> str:
        return "(" + ", ".join(self.action_names[i][a] for i, a in enumerate(profile)) + ")"

    def _deviate(self, profile: tuple[int, ...], i: int, a: int) -> tuple[int, ...]:
        s = list(profile)
        s[i] = a
        return tuple(s)

    # ------------------------------------------------------------ pure NE
    def pure_nash(self, tol: float = 1e-9) -> list[tuple[int, ...]]:
        out = []
        for s in self.profiles:
            stable = all(
                self.cost(i, s) <= self.cost(i, self._deviate(s, i, a)) + tol
                for i in range(self.n_players)
                for a in range(self.shape[i])
            )
            if stable:
                out.append(s)
        return out

    # ----------------------------------------------------- mixed NE (2 players)
    def mixed_nash_2p(self, tol: float = 1e-9) -> list[tuple[np.ndarray, np.ndarray]]:
        """All Nash equilibria of a two-player game by support enumeration.

        Returns a list of (x, y): x is player 1's mixed strategy, y is player 2's.
        Works for non-degenerate small games (every game in this project).
        """
        if self.n_players != 2:
            raise ValueError("support enumeration is implemented for 2 players")
        A, B = self.costs[0], self.costs[1]  # A[a, b] cost of P1, B[a, b] cost of P2
        m, k = A.shape
        found: list[tuple[np.ndarray, np.ndarray]] = []
        for size in range(1, min(m, k) + 1):
            for S1 in itertools.combinations(range(m), size):
                for S2 in itertools.combinations(range(k), size):
                    # y on S2 makes P1 indifferent on S1; x on S1 makes P2 indifferent on S2.
                    y = _indifference(A[np.ix_(S1, S2)], size)
                    x = _indifference(B[np.ix_(S1, S2)].T, size)
                    if x is None or y is None:
                        continue
                    xf = np.zeros(m)
                    yf = np.zeros(k)
                    xf[list(S1)] = x
                    yf[list(S2)] = y
                    c1 = A @ yf  # P1's cost of each pure action
                    c2 = xf @ B  # P2's cost of each pure action
                    if c1.min() < c1[list(S1)].max() - tol or c2.min() < c2[list(S2)].max() - tol:
                        continue
                    if not any(np.allclose(xf, u) and np.allclose(yf, v) for u, v in found):
                        found.append((xf, yf))
        return found

    # -------------------------------------------- correlated distributions
    def product_distribution(self, mixed: list[np.ndarray]) -> np.ndarray:
        """Joint distribution over profiles when players mix independently."""
        p = np.ones(len(self.profiles))
        for idx, s in enumerate(self.profiles):
            for i, a in enumerate(s):
                p[idx] *= mixed[i][a]
        return p

    def expected_costs(self, p: np.ndarray) -> np.ndarray:
        return np.array(
            [sum(p[j] * self.cost(i, s) for j, s in enumerate(self.profiles)) for i in range(self.n_players)]
        )

    def _ce_rows(self) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """One row per (player, told action a, deviation a'): sum_s p(s) * gain <= 0."""
        rows, labels = [], []
        for i in range(self.n_players):
            for a in range(self.shape[i]):
                for b in range(self.shape[i]):
                    if a == b:
                        continue
                    row = np.zeros(len(self.profiles))
                    for j, s in enumerate(self.profiles):
                        if s[i] == a:
                            row[j] = self.cost(i, s) - self.cost(i, self._deviate(s, i, b))
                    rows.append(row)
                    labels.append((i, a, b))
        return np.array(rows), labels

    def _cce_rows(self) -> np.ndarray:
        rows = []
        for i in range(self.n_players):
            for b in range(self.shape[i]):
                rows.append(
                    [self.cost(i, s) - self.cost(i, self._deviate(s, i, b)) for s in self.profiles]
                )
        return np.array(rows)

    def obedience_violation(self, p: np.ndarray) -> float:
        """Largest expected gain from disobeying a recommendation (<= 0 means p is a CE)."""
        rows, _ = self._ce_rows()
        return float(np.max(rows @ p)) if len(rows) else 0.0

    def cce_violation(self, p: np.ndarray) -> float:
        return float(np.max(self._cce_rows() @ p))

    def correlated_equilibrium(self, objective: str = "best", coarse: bool = False) -> "EquilibriumLP":
        """Solve the CE (or CCE) linear program.

        objective = "best"  -> minimise expected total cost (best for society)
        objective = "worst" -> maximise expected total cost (worst CE)
        objective = "fair"  -> minimise the largest player's expected cost
        """
        n_prof = len(self.profiles)
        total = np.array([self.total_cost(s) for s in self.profiles])
        A_ub = self._cce_rows() if coarse else self._ce_rows()[0]
        b_ub = np.zeros(A_ub.shape[0])
        A_eq = np.ones((1, n_prof))
        b_eq = np.array([1.0])
        bounds = [(0, None)] * n_prof

        if objective in ("best", "worst"):
            c = total if objective == "best" else -total
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            p = res.x
        elif objective == "fair":
            # variables (p, z): minimise z s.t. E[c_i] <= z for every player
            per_player = np.array([[self.cost(i, s) for s in self.profiles] for i in range(self.n_players)])
            c = np.r_[np.zeros(n_prof), 1.0]
            A_fair = np.c_[per_player, -np.ones(self.n_players)]
            A_all = np.r_[np.c_[A_ub, np.zeros(A_ub.shape[0])], A_fair]
            b_all = np.r_[b_ub, np.zeros(self.n_players)]
            res = linprog(c, A_ub=A_all, b_ub=b_all, A_eq=np.c_[A_eq, [[0.0]]], b_eq=b_eq,
                          bounds=bounds + [(None, None)], method="highs")
            p = res.x[:n_prof]
        else:
            raise ValueError(f"unknown objective {objective!r}")
        if not res.success:
            raise RuntimeError(f"CE LP failed: {res.message}")
        p = np.clip(p, 0, None)
        p /= p.sum()
        return EquilibriumLP(game=self, p=p, coarse=coarse)

    def social_optimum(self) -> tuple[tuple[int, ...], float]:
        best = min(self.profiles, key=self.total_cost)
        return best, self.total_cost(best)


@dataclass
class EquilibriumLP:
    """Result of a CE / CCE linear program."""

    game: NormalFormGame
    p: np.ndarray
    coarse: bool = False

    @property
    def expected_costs(self) -> np.ndarray:
        return self.game.expected_costs(self.p)

    @property
    def total_cost(self) -> float:
        return float(self.expected_costs.sum())

    def support(self, tol: float = 1e-7) -> dict[str, float]:
        return {
            self.game.profile_name(s): float(self.p[j])
            for j, s in enumerate(self.game.profiles)
            if self.p[j] > tol
        }


def _indifference(M: np.ndarray, size: int) -> np.ndarray | None:
    """Probabilities q >= 0, sum q = 1, with M @ q constant (rows indifferent)."""
    lhs = np.r_[np.c_[M, -np.ones(size)], np.c_[np.ones((1, size)), [[0.0]]]]
    rhs = np.r_[np.zeros(size), 1.0]
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    q = sol[:size]
    if np.any(q < -1e-9):
        return None
    return np.clip(q, 0, None)


# ============================================================================
# The three small games that carry the whole story.
#
# 1. Pigou's network       -> selfish choices waste resources (price of anarchy).
# 2. Two-student dinner     -> a private, correlated signal reaches the optimum,
#                              a shared counter does worst.
# 3. Intersection (Chicken) -> a correlated equilibrium can beat *every* Nash
#                              equilibrium and every mixture of them.
# ============================================================================

def pigou_poa(p: float) -> float:
    """Closed-form price of anarchy when the peak road costs x**p and the other costs 1.

    Selfish: everyone takes the peak road (it never costs more than 1) -> cost 1.
    Planner: send x* = (p+1)^(-1/p) to the peak road.
    PoA(p) = [1 - p (p+1)^(-(p+1)/p)]^(-1);  PoA(1) = 4/3.
    """
    return 1.0 / (1.0 - p * (p + 1.0) ** (-(p + 1.0) / p))


def pigou_numeric(p: float) -> dict[str, float]:
    """Same quantities found numerically (used to check the formula)."""
    social = lambda x: x * x**p + (1.0 - x) * 1.0  # noqa: E731
    res = minimize_scalar(social, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10})
    return {"selfish_cost": 1.0, "optimal_cost": float(res.fun), "optimal_peak_share": float(res.x),
            "poa": 1.0 / float(res.fun)}


# --------------------------------------------------------------------------
# 2. The two-student dinner game (from our project brief)
# --------------------------------------------------------------------------
def dinner_game(late_cost: float = 1.5) -> NormalFormGame:
    """Two students choose Peak or Late.

    Peak costs the number of students at Peak (1 or 2). Late always costs
    ``late_cost`` (no queue, but eating late). Lower is better.
    """
    P, L = 0, 1
    costs = np.zeros((2, 2, 2))
    for a in (P, L):
        for b in (P, L):
            at_peak = (a == P) + (b == P)
            costs[0, a, b] = at_peak if a == P else late_cost
            costs[1, a, b] = at_peak if b == P else late_cost
    return NormalFormGame(costs, action_names=[["Peak", "Late"], ["Peak", "Late"]],
                          player_names=["Student 1", "Student 2"], name="Two-student dinner game")


def shared_counter_dynamics(game: NormalFormGame, start: tuple[int, int], days: int = 20) -> list[tuple[int, int]]:
    """Both students best-respond to *yesterday's* public outcome (a live counter).

    Identical students read the same counter and react identically, so the
    crowd jumps between slots (the El Farol effect).
    """
    history = [start]
    for _ in range(days - 1):
        a, b = history[-1]
        new_a = int(np.argmin([game.cost(0, (x, b)) for x in range(game.shape[0])]))
        new_b = int(np.argmin([game.cost(1, (a, y)) for y in range(game.shape[1])]))
        history.append((new_a, new_b))
    return history


# --------------------------------------------------------------------------
# 3. The intersection game (Aumann's Chicken)
# --------------------------------------------------------------------------
def intersection_game() -> NormalFormGame:
    """Two drivers reach a crossing: Stop or Go.

    Utilities (Aumann 1974): (Stop, Stop) = 6/6, (Stop, Go) = 2/7,
    (Go, Stop) = 7/2, (Go, Go) = 0/0 (crash). We store costs = -utility.
    A traffic light is the correlated device.
    """
    util = np.array([[[6, 2], [7, 0]], [[6, 7], [2, 0]]], dtype=float)
    return NormalFormGame(-util, action_names=[["Stop", "Go"], ["Stop", "Go"]],
                          player_names=["Driver 1", "Driver 2"], name="Intersection (Chicken)")


# --------------------------------------------------------------------------
# A slightly bigger mess game for the CE LP (3 students, 3 slots)
# --------------------------------------------------------------------------
def three_student_game(capacity: int = 1, wait_per_extra: float = 2.0,
                       shift_cost: float = 0.8, preferred: int = 1) -> NormalFormGame:
    """Three students, slots Early / Peak / Late, all prefer ``preferred``.

    Cost = waiting (wait_per_extra per student above capacity in the slot)
           + shift_cost * |slot - preferred|.
    """
    n, T = 3, 3
    costs = np.zeros((n, T, T, T))
    for s in np.ndindex(T, T, T):
        loads = np.bincount(s, minlength=T)
        for i in range(n):
            t = s[i]
            costs[(i, *s)] = wait_per_extra * max(0, loads[t] - capacity) + shift_cost * abs(t - preferred)
    names = ["Early", "Peak", "Late"]
    return NormalFormGame(costs, action_names=[names] * n,
                          player_names=[f"Student {i + 1}" for i in range(n)], name="Three-student game")


# ============================================================================
# Learning dynamics that *find* equilibria without a mediator.
#
# Two classic results connect online learning to equilibrium:
#
# * If every player uses a no-EXTERNAL-regret rule (e.g. Hedge), the empirical
#   distribution of joint play converges to the set of COARSE correlated
#   equilibria.
# * If every player uses a no-SWAP-regret rule, e.g. Hart & Mas-Colell's
#   regret matching (2000), it converges to the set of CORRELATED equilibria.
#
# So a correlated equilibrium is not only something a mediator can impose: it
# is where simple adaptive players end up on their own. We measure this by
# tracking how far the empirical distribution is from satisfying the CE / CCE
# inequalities.
# ============================================================================

@dataclass
class LearningRun:
    profiles: np.ndarray  # (rounds, n_players) actions played
    empirical: np.ndarray  # final empirical joint distribution over game.profiles
    ce_violation: np.ndarray  # violation of CE inequalities by the running empirical distribution
    cce_violation: np.ndarray
    rounds_logged: np.ndarray


def _profile_index(game: NormalFormGame) -> dict[tuple[int, ...], int]:
    return {s: j for j, s in enumerate(game.profiles)}


def _counterfactual_costs(game: NormalFormGame, i: int, profile: tuple[int, ...]) -> np.ndarray:
    """Cost player i would have had with each of its actions, others fixed."""
    s = list(profile)
    out = np.empty(game.shape[i])
    for a in range(game.shape[i]):
        s[i] = a
        out[a] = game.cost(i, tuple(s))
    return out


def regret_matching(game: NormalFormGame, rounds: int = 5000, seed: int = 0, log_every: int = 50) -> LearningRun:
    """Hart & Mas-Colell (2000) conditional regret matching.

    D[i][j, k] = average over past rounds where i played j of
                 (cost actually paid) - (cost if i had played k instead).
    Next round, from last action j, switch to k with probability D[j, k]^+ / mu,
    otherwise repeat j. The empirical joint play converges to the CE set.
    """
    rng = np.random.default_rng(seed)
    n = game.n_players
    D = [np.zeros((m, m)) for m in game.shape]
    cost_range = float(game.costs.max() - game.costs.min()) or 1.0
    mu = [2.0 * (m - 1) * cost_range + 1e-9 for m in game.shape]
    idx = _profile_index(game)
    counts = np.zeros(len(game.profiles))
    current = tuple(int(rng.integers(m)) for m in game.shape)
    played, logged, ce_v, cce_v = [], [], [], []
    for t in range(1, rounds + 1):
        played.append(current)
        counts[idx[current]] += 1
        for i in range(n):
            j = current[i]
            cf = _counterfactual_costs(game, i, current)
            D[i][j] += cf[j] - cf  # regret of not having played k when j was played
        if t % log_every == 0 or t == rounds:
            p = counts / counts.sum()
            logged.append(t)
            ce_v.append(game.obedience_violation(p))
            cce_v.append(game.cce_violation(p))
        nxt = []
        for i in range(n):
            j = current[i]
            avg = np.maximum(D[i][j] / t, 0.0)
            avg[j] = 0.0
            probs = avg / mu[i]
            probs[j] = 1.0 - probs.sum()
            nxt.append(int(rng.choice(game.shape[i], p=probs)))
        current = tuple(nxt)
    return LearningRun(np.array(played), counts / counts.sum(), np.array(ce_v), np.array(cce_v), np.array(logged))


def hedge(game: NormalFormGame, rounds: int = 5000, eta: float | None = None, seed: int = 0,
          log_every: int = 50) -> LearningRun:
    """Every player runs Hedge (multiplicative weights) on full-information costs.

    Hedge has no external regret, so the empirical distribution approaches
    the CCE set (not necessarily the smaller CE set).
    """
    rng = np.random.default_rng(seed)
    n = game.n_players
    cost_range = float(game.costs.max() - game.costs.min()) or 1.0
    eta = eta if eta is not None else np.sqrt(8 * np.log(max(game.shape)) / rounds) / cost_range
    cum = [np.zeros(m) for m in game.shape]
    idx = _profile_index(game)
    counts = np.zeros(len(game.profiles))
    played, logged, ce_v, cce_v = [], [], [], []
    for t in range(1, rounds + 1):
        current = []
        for i in range(n):
            w = np.exp(-eta * (cum[i] - cum[i].min()))
            current.append(int(rng.choice(game.shape[i], p=w / w.sum())))
        current = tuple(current)
        played.append(current)
        counts[idx[current]] += 1
        for i in range(n):
            cum[i] += _counterfactual_costs(game, i, current)
        if t % log_every == 0 or t == rounds:
            p = counts / counts.sum()
            logged.append(t)
            ce_v.append(game.obedience_violation(p))
            cce_v.append(game.cce_violation(p))
    return LearningRun(np.array(played), counts / counts.sum(), np.array(ce_v), np.array(cce_v), np.array(logged))


# ============================================================================
# Why a delayed crowd display makes the crowd swing (Hypothesis H2).
#
# Linear model: y_t is how far the crowd in a slot is above its balanced level.
# A fraction of students react to a display that is d steps old, with overall
# responsiveness k (share who react x how strongly they react):
#
#         y_{t+1} = y_t - k * y_{t-d}.
#
# Characteristic polynomial: z^{d+1} - z^d + k = 0. The crowd settles iff every
# root lies strictly inside the unit circle, which holds iff
#
#         0 < k < k*(d) = 2 sin( pi / (4d + 2) ).
#
# k*(0) = 2, k*(1) = 1, k*(2) ~ 0.618, ... so older information tolerates less
# reaction before it starts to oscillate.
# ============================================================================

def critical_responsiveness(d: int) -> float:
    """Closed-form stability boundary k*(d)."""
    return 2.0 * np.sin(np.pi / (4 * d + 2))


def spectral_radius(k: float, d: int) -> float:
    """Largest |root| of z^{d+1} - z^d + k (stable iff < 1)."""
    coeffs = np.zeros(d + 2)  # coeffs[j] multiplies z^(d+1-j)
    coeffs[0] += 1.0  # z^{d+1}
    coeffs[1] -= 1.0  # -z^d
    coeffs[d + 1] += k  # +k   (for d = 0 this shares a slot with -z^0)
    return float(np.max(np.abs(np.roots(coeffs))))


def numeric_boundary(d: int, k_max: float = 2.5, tol: float = 1e-10) -> float:
    """Bisection for the largest k with spectral radius < 1 (checks the formula)."""
    lo, hi = 1e-9, k_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if spectral_radius(mid, d) < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def simulate(k: float, d: int, steps: int = 40, y0: float = 1.0) -> np.ndarray:
    y = np.zeros(steps)
    y[0] = y0
    for t in range(steps - 1):
        y[t + 1] = y[t] - k * y[max(t - d, 0)]
    return y
