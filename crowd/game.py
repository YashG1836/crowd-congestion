"""Layer 2 - the slot-choice congestion game for a whole population: congestion models, the game, Nash / optimum / price of anarchy."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


# ============================================================================
# How crowded a slot is -> how much each arriving person pays.
#
# A *congestion model* maps the load vector A (A[t] = number of people who
# arrive in slot t) to a congestion cost g[t] in minutes: what one person
# arriving in slot t pays because of the crowd. Every model is vectorised:
# A can have shape (..., T) and g has the same shape.
#
# FluidQueue      the main model: one service point serving `service_rate`
#                 people per minute, arrivals spread evenly inside each slot,
#                 first come first served, leftover queue carries into the
#                 next slot. Optional scarcity: a popular item with `stock`
#                 portions; people served after it runs out pay `scarcity_penalty`.
# Affine          g[t] = a[t] * A[t] + b[t]. Separable and linear: the textbook
#                 congestion game (Pigou's network is the case T = 2).
# Expected        average of several models with probabilities: the expected
#                 cost when the state of the night is uncertain (Bayesian game).
# ============================================================================

class CongestionModel:
    T: int

    def cost(self, A: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def wait(self, A: np.ndarray) -> np.ndarray:
        """Waiting part of the cost (minutes). Defaults to the full cost."""
        return self.cost(A)


@dataclass
class FluidQueue(CongestionModel):
    """Deterministic fluid queue with carry-over between slots.

    In slot t (length dt minutes) people arrive at rate A[t]/dt and are served
    at rate mu. Starting from backlog q0, the queue after u minutes is
    max(0, q0 + r*u) with r = (A[t] - mu*dt)/dt. A person who arrives when
    the queue is q waits exactly q/mu minutes (FIFO). Averaging over the slot:

        r >= 0:  W = (q0 + (A - mu*dt)/2) / mu
        r <  0:  queue drains after s = min(dt, q0/(-r)) minutes,
                 W = (q0*s + r*s^2/2) / (dt*mu)

    and the backlog passed on is max(0, q0 + A - mu*dt). Total waiting
    sum_t A[t]*W[t] equals the area under the queue curve (Little's law).
    """

    service_rate: float  # people served per minute (mu)
    T: int = 12
    slot_minutes: float = 10.0
    scarcity_penalty: float = 0.0  # minutes-equivalent cost of missing the special item
    stock: float | None = None  # portions of the special item (None = unlimited)

    @property
    def capacity(self) -> float:
        """People that can be served in one slot."""
        return self.service_rate * self.slot_minutes

    def _run(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = np.asarray(A, dtype=float)
        dt, mu = self.slot_minutes, self.service_rate
        W = np.empty_like(A)
        Q0 = np.empty_like(A)  # backlog at the start of each slot
        q0 = np.zeros(A.shape[:-1])
        for t in range(A.shape[-1]):
            Q0[..., t] = q0
            net = A[..., t] - mu * dt
            r = net / dt
            draining = r < 0
            safe_r = np.where(draining, -r, 1.0)
            s = np.where(draining, np.minimum(dt, q0 / safe_r), dt)
            area = q0 * s + 0.5 * r * s**2
            W[..., t] = area / (dt * mu)
            q0 = np.maximum(0.0, q0 + net)
        return W, Q0

    def wait(self, A: np.ndarray) -> np.ndarray:
        return self._run(A)[0]

    def backlog(self, A: np.ndarray) -> np.ndarray:
        """Queue length at the start of each slot."""
        return self._run(A)[1]

    def miss_fraction(self, A: np.ndarray) -> np.ndarray:
        """Share of slot-t arrivals who find the special item gone (FIFO service order)."""
        A = np.asarray(A, dtype=float)
        if self.stock is None:
            return np.zeros_like(A)
        cum = np.cumsum(A, axis=-1)
        before = cum - A
        # people in slot t occupy service positions (before, cum]; those beyond `stock` miss out.
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(A > 0, (cum - self.stock) / np.where(A > 0, A, 1.0),
                            (before >= self.stock).astype(float))
        return np.clip(frac, 0.0, 1.0)

    def cost(self, A: np.ndarray) -> np.ndarray:
        g = self.wait(A)
        if self.stock is not None and self.scarcity_penalty > 0:
            g = g + self.scarcity_penalty * self.miss_fraction(A)
        return g

    def wait_at(self, A: np.ndarray, slot: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Exact wait of individuals arriving u minutes into `slot` (for synthetic scan logs)."""
        A = np.asarray(A, dtype=float)
        Q0 = self.backlog(A)
        r = (A[slot] - self.capacity) / self.slot_minutes
        q = np.maximum(0.0, Q0[slot] + r * u)
        return q / self.service_rate


@dataclass
class Affine(CongestionModel):
    """Separable linear congestion cost g[t] = a[t] * A[t] + b[t]."""

    a: np.ndarray
    b: np.ndarray
    T: int = field(init=False)

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.T = len(self.a)

    def cost(self, A: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(A, dtype=float) + self.b


@dataclass
class Expected(CongestionModel):
    """Expected congestion cost over uncertain states: sum_w P(w) g_w(A)."""

    models: list[CongestionModel]
    probs: np.ndarray
    T: int = field(init=False)

    def __post_init__(self) -> None:
        self.probs = np.asarray(self.probs, dtype=float)
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError("probabilities must sum to 1")
        self.T = self.models[0].T

    def cost(self, A: np.ndarray) -> np.ndarray:
        return sum(p * m.cost(A) for p, m in zip(self.probs, self.models))

    def wait(self, A: np.ndarray) -> np.ndarray:
        return sum(p * m.wait(A) for p, m in zip(self.probs, self.models))


# ============================================================================
# The slot-choice congestion game for a whole population.
#
# Players  n people who share one facility (mess, gym, library, shuttle...).
# Actions  T time slots.
# Types    people with the same preferred slot tau and flexibility beta are
#          interchangeable, so we group them into K types with counts n_k.
# State    X[k, t] = number of type-k people who use slot t. Loads A = X.sum(0).
# Cost     a type-k person in slot t pays
#
#              c_k(t; A) = g_t(A)  +  beta_k * |t - tau_k|
#                          crowd      schedule inconvenience
#
#          where g is any congestion model (see queue.py). beta_k is in
#          "minutes of waiting I would accept to move one slot".
#
# The game is *atomic*: when one person switches from t to t', the loads change
# to A - e_t + e_t'. All equilibrium checks use these exact one-person moves.
# ============================================================================

@dataclass
class SlotGame:
    congestion: CongestionModel
    tau: np.ndarray  # (K,) preferred slot of each type
    beta: np.ndarray  # (K,) flexibility cost per slot of shift (minutes)
    counts: np.ndarray  # (K,) number of people of each type
    slot_labels: list[str] | None = None
    name: str = "slot game"
    _sched: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tau = np.asarray(self.tau, dtype=int)
        self.beta = np.asarray(self.beta, dtype=float)
        self.counts = np.asarray(self.counts, dtype=int)
        if not (len(self.tau) == len(self.beta) == len(self.counts)):
            raise ValueError("tau, beta and counts must have the same length")
        if self.tau.min() < 0 or self.tau.max() >= self.T:
            raise ValueError("preferred slots must lie in 0..T-1")
        slots = np.arange(self.T)
        self._sched = self.beta[:, None] * np.abs(slots[None, :] - self.tau[:, None])
        if self.slot_labels is None:
            self.slot_labels = [f"slot {t}" for t in range(self.T)]

    # ------------------------------------------------------------ basics
    @property
    def T(self) -> int:
        return self.congestion.T

    @property
    def K(self) -> int:
        return len(self.tau)

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    @property
    def schedule_cost(self) -> np.ndarray:
        """S[k, t] = beta_k * |t - tau_k|."""
        return self._sched

    def with_congestion(self, model: CongestionModel, name: str | None = None) -> "SlotGame":
        return replace(self, congestion=model, name=name or self.name)

    def check(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.shape != (self.K, self.T):
            raise ValueError(f"schedule must have shape {(self.K, self.T)}")
        if np.any(X.sum(1) != self.counts):
            raise ValueError("each type's row must sum to its count")
        return X

    @staticmethod
    def loads(X: np.ndarray) -> np.ndarray:
        return np.asarray(X).sum(axis=0)

    def selfish_start(self) -> np.ndarray:
        """Everyone at their preferred slot (what happens with no thought about the crowd)."""
        X = np.zeros((self.K, self.T), dtype=int)
        X[np.arange(self.K), self.tau] = self.counts
        return X

    # ------------------------------------------------------------ costs
    def costs(self, X: np.ndarray) -> np.ndarray:
        """C[k, t]: cost of a type-k person who is in slot t (at the current loads)."""
        g = self.congestion.cost(self.loads(X))
        return g[None, :] + self._sched

    def _moved_loads(self, A: np.ndarray) -> np.ndarray:
        """M[t, t2, :] = A - e_t + e_t2 (one person moves from t to t2)."""
        E = np.eye(self.T)
        return A[None, None, :] - E[:, None, :] + E[None, :, :]

    def deviation_costs(self, X: np.ndarray) -> np.ndarray:
        """D[k, t, t2]: cost a type-k person now in t would pay after moving alone to t2."""
        A = self.loads(X).astype(float)
        g_moved = self.congestion.cost(self._moved_loads(A))  # (T, T, T)
        t_idx = np.arange(self.T)
        g_dest = g_moved[t_idx[:, None], t_idx[None, :], t_idx[None, :]]  # (T, T): g at t2 after move
        return g_dest[None, :, :] + self._sched[:, None, :]

    def gains(self, X: np.ndarray) -> np.ndarray:
        """G[k, t, t2] = cost now - cost after moving (positive = profitable move). -inf if nobody to move."""
        D = self.deviation_costs(X)
        t_idx = np.arange(self.T)
        current = D[:, t_idx, t_idx]  # moving to your own slot changes nothing
        G = current[:, :, None] - D
        G[np.asarray(X) <= 0] = -np.inf
        return G

    def max_regret(self, X: np.ndarray) -> float:
        """Largest gain any single person could get by moving (<= 0 at a Nash equilibrium)."""
        return float(np.max(self.gains(X)))

    def is_nash(self, X: np.ndarray, tol: float = 1e-9) -> bool:
        return self.max_regret(X) <= tol

    # ------------------------------------------------------------ welfare
    def congestion_total(self, A: np.ndarray) -> np.ndarray:
        """sum_t A_t g_t(A) (vectorised over leading axes)."""
        A = np.asarray(A, dtype=float)
        return (A * self.congestion.cost(A)).sum(axis=-1)

    def social_cost(self, X: np.ndarray) -> float:
        X = np.asarray(X)
        return float(self.congestion_total(self.loads(X)) + (X * self._sched).sum())

    def breakdown(self, X: np.ndarray) -> dict[str, float]:
        """Total and per-person cost split into waiting, scarcity and schedule parts."""
        X = np.asarray(X)
        A = self.loads(X).astype(float)
        wait = float((A * self.congestion.wait(A)).sum())
        total_cong = float(self.congestion_total(A))
        sched = float((X * self._sched).sum())
        n = max(self.n, 1)
        return {
            "social_cost": total_cong + sched,
            "per_person": (total_cong + sched) / n,
            "wait_per_person": wait / n,
            "scarcity_per_person": (total_cong - wait) / n,
            "schedule_per_person": sched / n,
            "peak_load": float(A.max()),
        }

    # ------------------------------------------------------------ building
    @classmethod
    def from_individuals(cls, congestion: CongestionModel, tau: np.ndarray, beta: np.ndarray,
                         slot_labels: list[str] | None = None, name: str = "slot game"
                         ) -> tuple["SlotGame", np.ndarray]:
        """Group individuals with equal (tau, beta) into types.

        Returns the game and ``type_of`` (the type index of every individual).
        """
        tau = np.asarray(tau, dtype=int)
        beta = np.asarray(beta, dtype=float)
        keys = np.stack([tau, beta], axis=1)
        uniq, type_of, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        game = cls(congestion, uniq[:, 0].astype(int), uniq[:, 1], counts, slot_labels, name)
        return game, type_of.ravel()


# ============================================================================
# Benchmarks: Nash equilibrium, social optimum and the price of anarchy.
#
# Nash equilibrium (selfish outcome)
#     Best-response dynamics: repeatedly let the person with the most
#     profitable one-person move make it. For separable congestion costs this
#     is Rosenthal's potential game, so it always stops at a pure NE. With
#     queue carry-over there is no potential guarantee, so we report whether
#     it converged and verify the result (max regret <= 0).
#
# Social optimum (planner)
#     Minimise total cost. For the fluid queue the continuous relaxation is a
#     linear program (queue lengths as variables, Lindley's recursion as
#     constraints), which gives an excellent starting point. We round it to
#     integers and polish with exact local search (one-person moves and
#     two-person swaps). Local search is also started from the NE, so the
#     optimum we report is never worse than the NE.
#
# Price of anarchy
#     PoA = SC(worst NE found) / SC(optimum).  1.0 = selfishness costs nothing.
# ============================================================================

@dataclass
class Solution:
    X: np.ndarray
    social_cost: float
    converged: bool = True
    iterations: int = 0
    method: str = ""
    epsilon: float = 0.0  # largest gain any one person could still get by moving (minutes)


# ---------------------------------------------------------------- Nash
def nash_equilibrium(game: SlotGame, X0: np.ndarray | None = None, max_iter: int | None = None,
                     tol: float = 1e-9) -> Solution:
    """Best-response dynamics with one-person moves (largest gain first).

    Stops at an exact pure NE when no move gains more than ``tol``. With queue
    carry-over there is no potential function, so the dynamics can enter a
    cycle. We remember the state with the smallest remaining gain seen so far
    and return it when a cycle is detected (or after ``max_iter`` moves): an
    epsilon-Nash equilibrium, with epsilon reported in ``Solution.epsilon``.
    """
    X = (game.selfish_start() if X0 is None else np.array(X0, dtype=int)).copy()
    game.check(X)
    max_iter = max_iter or 5 * game.n + 100
    seen: set[bytes] = set()
    best_X, best_gain = X.copy(), np.inf
    for it in range(max_iter):
        G = game.gains(X)
        k, t, t2 = np.unravel_index(np.argmax(G), G.shape)
        gain = float(G[k, t, t2])
        if gain <= tol:
            return Solution(X, game.social_cost(X), True, it, "best-response dynamics (exact NE)", max(gain, 0.0))
        if gain < best_gain:
            best_X, best_gain = X.copy(), gain
        key = X.tobytes()
        if key in seen:
            return Solution(best_X, game.social_cost(best_X), True, it, "best-response dynamics (epsilon-NE)",
                            best_gain)
        seen.add(key)
        X[k, t] -= 1
        X[k, t2] += 1
    return Solution(best_X, game.social_cost(best_X), False, max_iter,
                    "best-response dynamics (epsilon-NE, iteration cap)", best_gain)


def wardrop_equilibrium(game: SlotGame, iters: int = 2000) -> np.ndarray:
    """Equilibrium of the large-population (non-atomic) version, by the method of successive averages.

    Each round every type's best slot gets weight 1/(r+1): x <- x + (BR(x) - x)/(r+1).
    This is fictitious play, the standard traffic-assignment algorithm. Returns
    fractional counts; round with ``round_schedule``.
    """
    x = game.selfish_start().astype(float)
    rows = np.arange(game.K)
    for r in range(1, iters + 1):
        C = game.costs(x)
        br = np.zeros_like(x)
        br[rows, C.argmin(axis=1)] = game.counts
        x += (br - x) / (r + 1)
    return x


def wardrop_gap(game: SlotGame, x: np.ndarray) -> float:
    """Average (over people) excess cost above the cheapest slot for their type; 0 at a Wardrop equilibrium."""
    C = game.costs(x)
    return float((x * (C - C.min(axis=1, keepdims=True))).sum() / max(game.n, 1))


def nash_from_several_starts(game: SlotGame, seed: int = 0, n_random: int = 2,
                             extra_starts: list[np.ndarray] | None = None) -> list[Solution]:
    """Run best-response dynamics from different starting points (to find several NE)."""
    rng = np.random.default_rng(seed)
    starts = [game.selfish_start(), round_schedule(game, wardrop_equilibrium(game))]
    for _ in range(n_random):
        X = np.zeros((game.K, game.T), dtype=int)
        for k in range(game.K):
            X[k] = rng.multinomial(game.counts[k], np.full(game.T, 1.0 / game.T))
        starts.append(X)
    starts += list(extra_starts or [])
    return [nash_equilibrium(game, X0) for X0 in starts]


# ---------------------------------------------------------------- Optimum
def _lp_relaxation(game: SlotGame) -> np.ndarray | None:
    """Continuous optimum for the fluid queue (scarcity ignored), on a 1-minute grid.

    Variables x[k,t] >= 0 and q[m] >= 0 (queue at the end of minute m).
    q[m] >= q[m-1] + a[m] - mu,  a[m] = A[slot(m)] / slot_minutes.
    Minimise sum_m (q[m-1] + q[m]) / 2   (person-minutes waited, Little's law)
             + sum x[k,t] * beta_k |t - tau_k|.
    """
    cm = game.congestion
    if not isinstance(cm, FluidQueue):
        return None
    K, T = game.K, game.T
    dt, mu = cm.slot_minutes, cm.service_rate
    steps = int(round(dt))
    if not np.isclose(steps, dt) or steps < 1:
        return None
    window = T * steps
    M = window + int(np.ceil(game.n / mu)) + 2
    nx = K * T
    # objective
    c_x = game.schedule_cost.ravel()
    c_q = np.ones(M)
    c_q[-1] = 0.5  # trapezoid weights: q[1..M-1] count fully, q[M] half (q[0] = 0)
    c = np.r_[c_x, c_q]
    # Lindley constraints: -q[m] + q[m-1] + a[m] <= mu
    rows, cols, vals = [], [], []
    for m in range(M):
        rows.append(m), cols.append(nx + m), vals.append(-1.0)
        if m > 0:
            rows.append(m), cols.append(nx + m - 1), vals.append(1.0)
        if m < window:
            t = m // steps
            for k in range(K):
                rows.append(m), cols.append(k * T + t), vals.append(1.0 / dt)
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(M, nx + M))
    b_ub = np.full(M, mu)
    # each type assigned exactly once
    rows, cols = [], []
    for k in range(K):
        for t in range(T):
            rows.append(k), cols.append(k * T + t)
    A_eq = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(K, nx + M))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=game.counts.astype(float),
                  bounds=[(0, None)] * (nx + M), method="highs")
    if not res.success:
        return None
    return res.x[:nx].reshape(K, T)


def round_schedule(game: SlotGame, Xf: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding that keeps every row sum equal to the type count."""
    X = np.floor(Xf + 1e-9).astype(int)
    for k in range(game.K):
        short = game.counts[k] - X[k].sum()
        if short > 0:
            order = np.argsort(-(Xf[k] - X[k]))
            X[k, order[:short]] += 1
        elif short < 0:
            order = np.argsort(Xf[k] - X[k])
            for t in order:
                take = min(-short, X[k, t])
                X[k, t] -= take
                short += take
                if short == 0:
                    break
    return X


def local_search(game: SlotGame, X0: np.ndarray, max_iter: int = 200_000, tol: float = 1e-9) -> Solution:
    """Exact improvement steps on total cost: one-person moves, then two-person swaps."""
    X = np.array(X0, dtype=int).copy()
    S = game.schedule_cost
    T = game.T
    for it in range(max_iter):
        A = game.loads(X).astype(float)
        tc_now = game.congestion_total(A)
        tc_moved = game.congestion_total(game._moved_loads(A))  # (T, T)
        delta = (tc_moved - tc_now)[None, :, :] + S[:, None, :] - S[:, :, None]  # (K, T, T)
        delta[X <= 0] = np.inf
        delta[:, np.arange(T), np.arange(T)] = np.inf
        k, t, t2 = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[k, t, t2] < -tol:
            X[k, t] -= 1
            X[k, t2] += 1
            continue
        # swaps keep the loads fixed and only change who goes where
        # swap[k, t, k2, t2] = change in schedule cost when one type-k person in t and one
        # type-k2 person in t2 trade places = S[k,t2] - S[k,t] + S[k2,t] - S[k2,t2]
        swap = (S[:, None, None, :] - S[:, :, None, None]
                + S.T[None, :, :, None] - S[None, None, :, :])
        occupied = X > 0
        swap[~occupied] = np.inf
        swap[:, :, ~occupied] = np.inf
        k, t, k2, t2 = np.unravel_index(np.argmin(swap), swap.shape)
        if swap[k, t, k2, t2] < -tol:
            X[k, t] -= 1
            X[k, t2] += 1
            X[k2, t2] -= 1
            X[k2, t] += 1
            continue
        return Solution(X, game.social_cost(X), True, it, "local search")
    return Solution(X, game.social_cost(X), False, max_iter, "local search (not converged)")


def social_optimum(game: SlotGame, ne: Solution | None = None) -> Solution:
    """Best schedule a planner could impose (minimum total cost)."""
    candidates = []
    Xf = _lp_relaxation(game)
    if Xf is not None:
        candidates.append(local_search(game, round_schedule(game, Xf)))
        candidates[-1].method = "LP relaxation + local search"
    candidates.append(local_search(game, game.selfish_start()))
    if ne is not None:
        candidates.append(local_search(game, ne.X))
    return min(candidates, key=lambda s: s.social_cost)


# ---------------------------------------------------------------- PoA
@dataclass
class Benchmarks:
    game: SlotGame
    nash: Solution  # worst NE found
    best_nash: Solution
    optimum: Solution
    all_nash: list[Solution]

    @property
    def epsilon(self) -> float:
        """Largest gain (minutes) any single person could still get at the reported equilibria."""
        return max(s.epsilon for s in self.all_nash)

    @staticmethod
    def _ratio(a: float, b: float) -> float:
        if b <= 1e-12:
            return 1.0 if a <= 1e-12 else float("inf")  # no congestion at all -> no inefficiency
        return a / b

    @property
    def poa(self) -> float:
        return self._ratio(self.nash.social_cost, self.optimum.social_cost)

    @property
    def pos(self) -> float:
        """Price of stability: best NE / optimum."""
        return self._ratio(self.best_nash.social_cost, self.optimum.social_cost)


def benchmarks(game: SlotGame, seed: int = 0, n_random: int = 2, eps_margin: float = 0.1) -> Benchmarks:
    """NE from several starts, the optimum, and PoA / PoS.

    Only equilibria whose epsilon is within ``eps_margin`` minutes (6 s) of the
    most stable one found are compared, so PoA is not driven by a poor
    approximation. ``Benchmarks.epsilon`` is the largest epsilon among them.
    """
    nes = nash_from_several_starts(game, seed=seed, n_random=n_random)
    opt = social_optimum(game, ne=min(nes, key=lambda s: s.social_cost))
    nes.append(nash_equilibrium(game, opt.X))  # selection matters: try starting from the optimum too
    floor = min(s.epsilon for s in nes)
    accepted = [s for s in nes if s.epsilon <= floor + eps_margin]
    worst = max(accepted, key=lambda s: s.social_cost)
    best = min(accepted, key=lambda s: s.social_cost)
    return Benchmarks(game, worst, best, opt, accepted)
