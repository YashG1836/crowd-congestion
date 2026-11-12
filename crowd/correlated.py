"""Layer 2 - correlated equilibrium for a population (the smart display) and the Bayesian (incomplete-information) analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .game import CongestionModel, Expected, SlotGame, benchmarks, round_schedule


# ============================================================================
# Correlated equilibrium for a whole population: the "smart display".
#
# A mediator (the mess app) picks a schedule X and privately tells each person
# a slot. Inside each type the slots are handed out by a random shuffle, so a
# type-k person is told slot t with probability X[k, t] / n_k and does not know
# who else was told what. The recommendation is a correlated equilibrium (CE)
# when nobody wants to disobey:
#
#     for every type k, told slot t, and alternative t':
#         E[ cost of obeying  -  cost of going to t'  |  told t ]  <=  0.
#
# Three things live here.
#
# 1. Fair recommender (``recommend``). Shuffle a Nash schedule. Since every
#    shuffle is a Nash equilibrium, the mixture is a CE: obeying is a best
#    response. It is the traffic light of the mess: it coordinates identical
#    people who could not agree on their own, it is stable from day one, and
#    because the shuffle changes daily the inconvenient slots rotate.
#
# 2. Obedience check and the price of obedience (``obedience``,
#    ``obedience_incentive``). The planner's optimum is NOT a CE: people sent to
#    an off-peak slot would rather go to the peak. We measure how much they
#    would gain and the smallest reward (e.g. a priority token worth b minutes)
#    that makes following the optimum a best response.
#
# 3. Best CE over a menu of schedules (``best_ce``). A CE may randomise over
#    several schedules; a person told t then averages over which schedule
#    was drawn. Choosing the probabilities is a linear program. In a large
#    congestion game with monotone costs, theory says the best CE costs the
#    same as Nash (see DESIGN.md, Proposition 2); this LP checks it on our model.
# ============================================================================

def recommend(rng: np.random.Generator, type_of: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Give every individual a slot so that type k gets exactly the slot counts X[k].

    type_of[i] is individual i's type. Members of a type are shuffled, then
    filled into slots in order, so each member's slot is uniformly random.
    """
    X = np.asarray(X, dtype=int)
    out = np.empty(len(type_of), dtype=int)
    for k in range(X.shape[0]):
        members = np.flatnonzero(type_of == k)
        if len(members) != X[k].sum():
            raise ValueError(f"type {k}: {len(members)} members but schedule has {X[k].sum()}")
        slots = np.repeat(np.arange(X.shape[1]), X[k])
        out[rng.permutation(members)] = slots
    return out


# ----------------------------------------------------------------- 2. obedience
@dataclass
class ObedienceReport:
    max_gain: float  # largest gain (minutes) any person could get by disobeying
    share_tempted: float  # share of people with a positive gain (> tol)
    mean_gain: float  # average gain over everyone (0 for those happy to obey)
    gain_by_cell: np.ndarray  # (K, T) best gain for a type-k person told slot t
    is_ce: bool


def obedience(game: SlotGame, X: np.ndarray, tol: float = 1e-9) -> ObedienceReport:
    """How tempted are people to ignore a shuffled recommendation drawn from schedule X?"""
    X = np.asarray(X)
    G = game.gains(X)  # (K, T, T); -inf where nobody is told t
    best = np.max(G, axis=2)
    best = np.where(X > 0, np.maximum(best, 0.0), 0.0)
    n = max(game.n, 1)
    return ObedienceReport(
        max_gain=float(best.max()),
        share_tempted=float((X * (best > tol)).sum() / n),
        mean_gain=float((X * best).sum() / n),
        gain_by_cell=best,
        is_ce=bool(best.max() <= tol),
    )


@dataclass
class IncentiveReport:
    bonus_by_cell: np.ndarray  # (K, T) reward needed for a type-k person told slot t
    total_per_meal: float  # sum of rewards (minutes-equivalent)
    per_person: float
    share_rewarded: float
    welfare_gain: float  # minutes saved vs the reference schedule (e.g. Nash)


def obedience_incentive(game: SlotGame, X: np.ndarray, reference_cost: float | None = None,
                        tol: float = 1e-9) -> IncentiveReport:
    """Smallest per-cell reward that makes following schedule X a best response.

    A type-k person told t would gain at most gain_by_cell[k, t] by moving, so a
    reward of exactly that much for showing up in the recommended slot removes
    the temptation. (Rewards are transfers such as a priority token; they are
    not a waste, but someone must provide them.)
    """
    rep = obedience(game, X, tol)
    bonus = rep.gain_by_cell
    total = float((np.asarray(X) * bonus).sum())
    n = max(game.n, 1)
    welfare = (reference_cost - game.social_cost(X)) if reference_cost is not None else float("nan")
    return IncentiveReport(bonus, total, total / n, float((np.asarray(X) * (bonus > tol)).sum() / n), welfare)


# ----------------------------------------------------------------- 3. best CE
@dataclass
class CEResult:
    probs: np.ndarray  # probability of each candidate schedule
    expected_cost: float
    schedules: list[np.ndarray]
    labels: list[str]
    max_violation: float

    def support(self, tol: float = 1e-6) -> dict[str, float]:
        return {lab: float(p) for lab, p in zip(self.labels, self.probs) if p > tol}


def best_ce(game: SlotGame, schedules: list[np.ndarray], labels: list[str] | None = None,
            coarse: bool = False, slack: float = 0.0) -> CEResult:
    """Lowest-cost (coarse) correlated equilibrium supported on the given schedules.

    Variables: p_j >= 0, sum p_j = 1. For CE, for each type k, told slot t and
    alternative t':     sum_j p_j X_j[k,t] (C_j[k,t] - D_j[k,t,t'] - eps)  <=  0
    i.e. E[gain from disobeying | told t] <= eps. For CCE (decide before the
    message), for each k and fixed t':
                         sum_j p_j sum_t X_j[k,t] (C_j[k,t] - D_j[k,t,t'] - eps)  <=  0
    ``slack`` = eps allows an epsilon-CE (needed when the Nash schedules are
    themselves epsilon-NE because of one-person granularity).
    """
    labels = labels or [f"schedule {j}" for j in range(len(schedules))]
    K, T = game.K, game.T
    cols, costs = [], []
    t_idx = np.arange(T)
    for X in schedules:
        X = game.check(np.asarray(X))
        D = game.deviation_costs(X)  # (K, T, T)
        C = D[:, t_idx, t_idx]
        G = X[:, :, None] * (C[:, :, None] - D - slack)  # (K, T, T) weighted gain
        cols.append(G.sum(axis=1) if coarse else G)
        costs.append(game.social_cost(X))
    Gs = np.stack(cols, axis=-1)  # (K, T, T, J) or (K, T, J)
    if coarse:
        A_ub = Gs.reshape(K * T, -1)
    else:
        mask = ~np.eye(T, dtype=bool)
        A_ub = Gs[:, mask, :].reshape(K * T * (T - 1), -1)
    b_ub = np.zeros(A_ub.shape[0])
    J = len(schedules)
    res = linprog(np.array(costs), A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, J)), b_eq=[1.0],
                  bounds=[(0, None)] * J, method="highs")
    if not res.success:
        raise RuntimeError(f"best-CE LP infeasible: {res.message}")
    p = np.clip(res.x, 0, None)
    p /= p.sum()
    viol = float(np.max(A_ub @ p - b_ub)) if len(A_ub) else 0.0
    return CEResult(p, float(np.dot(p, costs)), list(schedules), labels, viol)


def interpolated_schedules(game: SlotGame, X_from: np.ndarray, X_to: np.ndarray, steps: int = 10
                           ) -> tuple[list[np.ndarray], list[str]]:
    """Schedules on the straight line between two schedules (rounded to whole people)."""
    out, labels = [], []
    for a in np.linspace(0, 1, steps + 1):
        out.append(round_schedule(game, (1 - a) * np.asarray(X_from) + a * np.asarray(X_to)))
        labels.append(f"{a:.0%} of the way to the optimum")
    return out, labels


# ============================================================================
# Complete vs incomplete information: what is it worth to know tonight's state?
#
# Some nights are different (a counter is closed, a special item is slow to
# make, more people show up). Students do not know the state w in advance; they
# only know the prior P(w). This is a Bayesian game with a common unknown state.
#
# * Incomplete information (no signal). A student's choice cannot depend on w,
#   so a risk-neutral student minimises EXPECTED cost. The Bayesian Nash
#   equilibrium (BNE) is therefore the Nash equilibrium of the game whose
#   congestion cost is the average sum_w P(w) g_w(A).
# * Complete information (the app announces w, or a mediator recommends a slot
#   that depends on w -- a Bayes correlated equilibrium). Each night people
#   play the Nash equilibrium of that night's game.
#
# Value of information = E[cost | play BNE] - E[cost | play NE of the true night].
# It can be positive or negative: in congestion games extra information is not
# automatically good (that is the point of the project).
# ============================================================================

@dataclass
class BayesianReport:
    state_names: list[str]
    prior: np.ndarray
    bne_cost: float  # expected social cost when nobody knows the state
    informed_ne_cost: float  # expected social cost when the state is announced
    uninformed_opt_cost: float  # best a planner could do without knowing the state
    informed_opt_cost: float  # best a planner could do knowing the state
    per_state: dict[str, dict[str, float]]
    n: int

    @property
    def value_of_information(self) -> float:
        """Minutes per person saved by announcing the state (negative = information hurts)."""
        return (self.bne_cost - self.informed_ne_cost) / self.n


def bayesian_analysis(game: SlotGame, states: dict[str, CongestionModel], prior: list[float]) -> BayesianReport:
    names = list(states)
    prior = np.asarray(prior, dtype=float)
    models = [states[s] for s in names]
    games = {s: game.with_congestion(states[s], name=f"{game.name} [{s}]") for s in names}
    avg_game = game.with_congestion(Expected(models, prior), name=f"{game.name} [expected]")

    avg = benchmarks(avg_game)
    X_bne, X_uopt = avg.nash.X, avg.optimum.X
    per_state, bne, inf_ne, uopt, iopt = {}, 0.0, 0.0, 0.0, 0.0
    for p, s in zip(prior, names):
        b = benchmarks(games[s])
        row = {
            "bne": games[s].social_cost(X_bne),
            "informed_ne": b.nash.social_cost,
            "uninformed_opt": games[s].social_cost(X_uopt),
            "informed_opt": b.optimum.social_cost,
        }
        per_state[s] = {k: v / game.n for k, v in row.items()}
        bne += p * row["bne"]
        inf_ne += p * row["informed_ne"]
        uopt += p * row["uninformed_opt"]
        iopt += p * row["informed_opt"]
    return BayesianReport(names, prior, bne, inf_ne, uopt, iopt, per_state, game.n)
