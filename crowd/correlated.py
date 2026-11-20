"""Layer 2 - correlated equilibrium for a population: recommendations, obedience, rewards, and information design."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .game import (Expected, FluidQueue, SlotGame, continuous_optimum, large_crowd_equilibrium, large_crowd_gap,
                   round_schedule)


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
# Who gains from coordination? A Pareto check.
#
# Compare every type's expected cost under the Nash schedule, the planner's
# optimum, and the optimum plus the obedience reward of Proposition 3 (the
# reward is subtracted: it is what the person receives for following advice).
# ============================================================================
@dataclass
class WhoGains:
    nash: np.ndarray  # (K,) expected cost per person of each type at the Nash schedule
    optimum: np.ndarray  # (K,) at the optimum, no reward
    with_reward: np.ndarray  # (K,) at the optimum, minus the obedience reward received
    budget: float  # total reward per visit (minutes-equivalent)
    counts: np.ndarray

    def worse_off(self, which: str, tol: float = 1e-9) -> int:
        """Number of people whose cost under `which` is higher than under Nash."""
        return int(((getattr(self, which) > self.nash + tol) * self.counts).sum())

    def max_loss(self, which: str) -> float:
        return float(max((getattr(self, which) - self.nash).max(), 0.0))

    def group_mean(self, which: str, mask: np.ndarray) -> float:
        v, w = getattr(self, which), self.counts
        return float((v[mask] * w[mask]).sum() / w[mask].sum())


def who_gains(game: SlotGame, X_nash: np.ndarray, X_opt: np.ndarray) -> WhoGains:
    def per_type(X):
        X = np.asarray(X)
        return (X * game.costs(X)).sum(axis=1) / game.counts

    gains = obedience(game, X_opt).gain_by_cell
    reward = (np.asarray(X_opt) * gains).sum(axis=1) / game.counts
    return WhoGains(per_type(X_nash), per_type(X_opt), per_type(X_opt) - reward,
                    float((np.asarray(X_opt) * gains).sum()), game.counts)


# ============================================================================
# Correlated equilibrium in a LARGE crowd (every person negligible).
#
# Here a person who switches slot does not change the loads, so the obedience
# condition for a type-k person told slot t is simply
#     sum over (state w, schedule j):  p(w, j) * x_j[k, t] * ( C_wj[k, t] - C_wj[k, t'] )  <=  0
# for every alternative t'. Equilibria are exact (Wardrop, gap below a second),
# so no tolerance is needed to make Nash feasible -- the comparison is clean.
# ============================================================================
def large_crowd_ce(states: dict[str, SlotGame], prior: dict[str, float], menu: list[np.ndarray],
                   tol: float = 0.001) -> tuple[float, np.ndarray]:
    """Lowest expected cost of a (Bayes) correlated equilibrium supported on ``menu``.

    states: the game in each state of the world; prior: their probabilities. The mediator
    knows the state and draws a schedule from p(w, .), then privately tells each person a slot.
    With a single state this is the ordinary correlated equilibrium. tol (minutes) absorbs
    numerical error. Returns (expected social cost, p as an array [state, schedule]).
    """
    names = list(states)
    T = states[names[0]].T
    off = ~np.eye(T, dtype=bool)
    cols, costs = [], []
    for w in names:
        for x in menu:
            C = states[w].costs(x)
            gain = np.asarray(x)[:, :, None] * (C[:, :, None] - C[:, None, :] - tol)
            cols.append(gain[:, off].ravel())
            costs.append(states[w].social_cost(x))
    A_ub = np.stack(cols, axis=1)
    J = len(menu)
    A_eq = np.zeros((len(names), len(names) * J))
    for i in range(len(names)):
        A_eq[i, i * J:(i + 1) * J] = 1.0
    res = linprog(np.array(costs), A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq,
                  b_eq=[prior[w] for w in names], bounds=[(0, None)] * (len(names) * J), method="highs")
    if not res.success:
        raise RuntimeError(f"large-crowd CE LP failed: {res.message}")
    return float(res.fun), res.x.reshape(len(names), J)


def _segment(a: np.ndarray, b: np.ndarray, steps: int) -> list[np.ndarray]:
    return [(1 - s) * a + s * b for s in np.linspace(0, 1, steps + 1)[1:-1]]


@dataclass
class LargeCrowdCE:
    nash: float  # cost per person of the (Wardrop) equilibrium
    best_ce: float  # cost per person of the best correlated equilibrium found
    optimum: float  # planner's optimum per person
    gap_sec: float  # accuracy of the equilibrium (seconds)


def large_crowd_check(game: SlotGame, steps: int = 20, tol: float = 0.001) -> LargeCrowdCE:
    """Proposition 2 in numbers: with complete information the best CE costs what Nash costs.

    The menu holds the equilibrium, the optimum and ``steps`` schedules in between.
    """
    x_ne, gap = large_crowd_equilibrium(game)
    x_opt = continuous_optimum(game)
    menu = [x_ne, x_opt] + _segment(x_ne, x_opt, steps)
    cost, _ = large_crowd_ce({"only": game}, {"only": 1.0}, menu, tol)
    n = game.n
    return LargeCrowdCE(game.social_cost(x_ne) / n, cost / n, game.social_cost(x_opt) / n, 60 * gap)


# ============================================================================
# Information design under uncertainty: who should be told what?
#
# The operator knows tonight's state (normal, or "bad": the bottleneck serves only
# a share of its usual rate); the crowd only knows the probability p of a bad night.
# If everyone believes "bad with probability q", they play the large-crowd equilibrium
# x_q of that belief, and the expected cost per person is
#     V(q) = (1 - q) * cost_normal(x_q) + q * cost_bad(x_q).
# Five ways to use the operator's knowledge:
#   tell nobody          everyone keeps the prior:            V(p)
#   announce the state   beliefs become 0 or 1:               (1 - p) V(0) + p V(1)
#   best public hint     any public message moves beliefs to some q, on average p; the best
#                        one gives the lower convex envelope of V at p (Bayesian persuasion,
#                        "concavification"); with two states two messages are enough
#   private advice       the best Bayes correlated equilibrium: privately recommended slots,
#                        obedient on average over the nights on which that advice is given
#                        (linear program over a menu of schedules)
#   planner, informed    the optimum of each night (needs rewards to be followed)
# Every public-information outcome is a Bayes correlated equilibrium, so private advice is
# never worse than the best public hint; the question is how much better it is.
# ============================================================================
@dataclass
class InfoDesign:
    prior: dict[str, float]
    expected: dict[str, float]  # policy -> expected cost per person
    per_state: dict[str, dict[str, float]]  # policy -> state -> cost per person
    beliefs: list[float]  # q grid
    value: list[float]  # V(q) per person
    envelope: list[float]  # lower convex envelope of V
    hint: dict[str, float]  # best public hint: P(say "busy" | bad), P(say "busy" | normal), the two beliefs
    private_support: dict[str, list[tuple[str, float]]]  # state -> [(schedule label, probability)]
    max_gap_sec: float  # accuracy of every equilibrium used
    tol_sec: float  # obedience tolerance used by the private-advice LP (>= max gap)
    menu_size: int


def _lower_envelope(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Lower convex envelope of points (q_i, v_i), evaluated at every q_i."""
    hull: list[int] = []
    for i in range(len(q)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            if (v[i2] - v[i1]) * (q[i] - q[i1]) >= (v[i] - v[i1]) * (q[i2] - q[i1]):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(q, q[hull], v[hull])


def information_design(game: SlotGame, bad_factor: float = 0.7, p_bad: float = 0.3, n_beliefs: int = 21,
                       steps: int = 8, tol: float = 0.001) -> InfoDesign:
    cm = game.congestion
    if not isinstance(cm, FluidQueue):
        raise ValueError("information_design needs a fluid-queue game")
    normal = FluidQueue(cm.service_rate, cm.T, cm.slot_minutes)
    bad = FluidQueue(cm.service_rate * bad_factor, cm.T, cm.slot_minutes)
    S = {"normal": game.with_congestion(normal), "bad": game.with_congestion(bad)}
    prior = {"normal": 1 - p_bad, "bad": p_bad}
    n = game.n

    def belief_game(q):  # q = probability of a bad night given what people know
        if q <= 1e-12:
            return S["normal"]
        if q >= 1 - 1e-12:
            return S["bad"]
        return game.with_congestion(Expected([normal, bad], np.array([1 - q, q])))

    q = np.unique(np.r_[np.linspace(0, 1, n_beliefs), p_bad])
    xs, gaps, x_prev = [], [], None
    for qi in q:  # continuation: each belief warm-starts from the previous one
        x, gp = large_crowd_equilibrium(belief_game(qi), x0=x_prev)
        xs.append(x)
        gaps.append(gp)
        x_prev = x
    V = np.array([((1 - qi) * S["normal"].social_cost(x) + qi * S["bad"].social_cost(x)) / n for qi, x in zip(q, xs)])
    env = _lower_envelope(q, V)
    ip = int(np.flatnonzero(np.isclose(q, p_bad))[0])

    per, exp_ = {}, {}
    xp = xs[ip]
    per["tell nobody"] = {w: S[w].social_cost(xp) / n for w in S}
    per["announce the state"] = {"normal": S["normal"].social_cost(xs[0]) / n, "bad": S["bad"].social_cost(xs[-1]) / n}
    # best public hint: the pair of beliefs (q_i <= p <= q_j) whose chord is lowest at p
    best = (V[ip], ip, ip, 1.0)
    for i in range(ip + 1):
        for j in range(ip, len(q)):
            if j == i:
                continue
            lam = (q[j] - p_bad) / (q[j] - q[i])  # weight on message i
            val = lam * V[i] + (1 - lam) * V[j]
            if val < best[0] - 1e-12:
                best = (val, i, j, lam)
    _, i, j, lam = best
    msgs = [(lam, i), (1 - lam, j)] if i != j else [(1.0, i)]
    per["best public hint"] = {
        "normal": sum(m * (1 - q[k]) / (1 - p_bad) * S["normal"].social_cost(xs[k]) for m, k in msgs) / n,
        "bad": sum(m * q[k] / p_bad * S["bad"].social_cost(xs[k]) for m, k in msgs) / n}
    hint = {"P(busy | bad)": float((1 - lam) * q[j] / p_bad) if i != j else 0.0,
            "P(busy | normal)": float((1 - lam) * (1 - q[j]) / (1 - p_bad)) if i != j else 0.0,
            "belief after quiet": float(q[i]), "belief after busy": float(q[j])}

    o_n, o_b = continuous_optimum(S["normal"]), continuous_optimum(S["bad"])
    o_e = continuous_optimum(belief_game(p_bad))
    per["planner, informed"] = {"normal": S["normal"].social_cost(o_n) / n, "bad": S["bad"].social_cost(o_b) / n}

    base = {"Nash, normal": xs[0], "Nash, bad": xs[-1], "Nash, unsure": xp,
            "optimum, normal": o_n, "optimum, bad": o_b, "optimum, unsure": o_e}
    menu, labels = list(base.values()), list(base)
    for k, qi in enumerate(q):
        if k not in (0, len(q) - 1, ip):
            menu.append(xs[k])
            labels.append(f"Nash, belief {qi:.2f}")
    keys = list(base)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            for k, x in enumerate(_segment(base[keys[a]], base[keys[b]], steps)):
                menu.append(x)
                labels.append(f"{k + 1}/{steps} from {keys[a]} to {keys[b]}")
    lp_tol = max(tol, max(gaps) + 1e-6)  # every equilibrium used above must be feasible
    cost, P = large_crowd_ce(S, prior, menu, lp_tol)
    per["private advice"] = {w: sum(P[r, c] * S[w].social_cost(menu[c]) for c in range(len(menu))) / prior[w] / n
                             for r, w in enumerate(S)}
    support = {w: [(labels[c], float(P[r, c] / prior[w])) for c in np.argsort(-P[r]) if P[r, c] > 1e-6]
               for r, w in enumerate(S)}
    order = ["tell nobody", "announce the state", "best public hint", "private advice", "planner, informed"]
    for k in order:
        exp_[k] = prior["normal"] * per[k]["normal"] + prior["bad"] * per[k]["bad"]
    return InfoDesign(prior, {k: float(exp_[k]) for k in order}, {k: per[k] for k in order}, q.tolist(), V.tolist(),
                      env.tolist(), hint, support, 60 * max(gaps), 60 * lp_tol, len(menu))


# ============================================================================
# Can ANY correlated equilibrium of a large crowd beat Nash? A search that is close to exhaustive.
#
# Lemma (the queue is monotone). For any two arrival profiles A, B (same total),
#     <A - B, W(A) - W(B)>  >=  0.
# Proof: in continuous time the queue is Q = phi + R with phi(u) = N(u) - mu u (arrivals minus
# service) and R the cumulative idle time, which only grows while Q = 0 (Skorokhod reflection).
# Then sum_t (A_t - B_t)(W_t(A) - W_t(B)) = (1/mu) int (a - b)(Q_A - Q_B) du
#   = (1/mu) [ (Q_A(L) - Q_B(L))^2 / 2  -  int (Q_A - Q_B) dR_A  +  int (Q_A - Q_B) dR_B ],
# and both integrals have the right sign because dR_A lives where Q_A = 0 (so Q_A - Q_B <= 0 there)
# and dR_B where Q_B = 0. The queue is monotone but not strictly, so Proposition 2's argument gives:
# every (coarse) correlated equilibrium only uses schedules that send each type to slots that are
# already optimal for it at Nash -- the "best-response face". We then search that face:
#   1. vertices of the face plus thousands of random splits, and the CE linear program over them;
#   2. column generation: the LP's dual prices define a pricing problem whose solution is the face
#      schedule that would lower the cost most; we add it and repeat;
#   3. the obedience tolerance is shrunk tenfold twice. A real CE cheaper than Nash would keep its
#      saving; numerical slack shrinks in proportion to the tolerance.
# ============================================================================
@dataclass
class FaceCheck:
    nash: float  # Nash (Wardrop) cost per person
    cheapest_on_face: float  # cheapest schedule on the face (not necessarily obeyed)
    face_dim: int  # dimension of the best-response face
    menu: int  # schedules in the final LP
    saving_sec: dict[float, float]  # obedience tolerance (minutes) -> best CE saving vs Nash (seconds per person)
    cg_rounds: int


def best_response_face(game: SlotGame, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """(K, T) mask of the slots that are (eps-)optimal for each type at schedule x."""
    C = game.costs(x)
    return C <= C.min(axis=1, keepdims=True) + eps


def _face_project(v: np.ndarray, mask: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v)
    for k in range(v.shape[0]):
        idx = np.flatnonzero(mask[k])
        u = np.sort(v[k, idx])[::-1]
        css = np.cumsum(u) - z[k]
        rho = np.nonzero(u - css / np.arange(1, len(u) + 1) > 0)[0][-1]
        out[k, idx] = np.maximum(v[k, idx] - css[rho] / (rho + 1), 0.0)
    return out


def _pricing(game: SlotGame, x: np.ndarray, lam: np.ndarray, tol: float) -> tuple[float, np.ndarray]:
    """Value and gradient of  SC(x) + sum lam[k,t,t'] x[k,t] (C[k,t] - C[k,t'] - tol)  (the reduced cost + const)."""
    cm, S = game.congestion, game.schedule_cost
    A = x.sum(0)
    W = cm.wait(A)
    C = W[None, :] + S
    lout = lam.sum(2)
    H = lout * C - np.einsum("ktu,ku->kt", lam, C) - tol * lout
    E = np.eye(game.T) * 1e-3
    J = ((cm.wait(A[None, :] + E) - cm.wait(np.maximum(A[None, :] - E, 0))) / 2e-3).T  # J[t1, t] = dW_t1/dA_t
    u, v = (x * lout).sum(0), np.einsum("kt,ktu->u", x, lam)
    return float((x * (C + H)).sum()), (W + J.T @ A + J.T @ (u - v))[None, :] + S + H


def _ce_master(game: SlotGame, menu: list[np.ndarray], tol: float):
    T = game.T
    off = ~np.eye(T, dtype=bool)
    cols, costs = [], []
    for x in menu:
        C = game.costs(x)
        cols.append((np.asarray(x)[:, :, None] * (C[:, :, None] - C[:, None, :] - tol))[:, off].ravel())
        costs.append(game.social_cost(x))
    A_ub = np.stack(cols, axis=1)
    res = linprog(np.array(costs), A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=np.ones((1, len(menu))),
                  b_eq=[1.0], bounds=[(0, None)] * len(menu), method="highs")
    lam = np.zeros((game.K, T, T))
    lam[:, off] = (-res.ineqlin.marginals).reshape(game.K, -1)
    return float(res.fun), res.x, lam, float(res.eqlin.marginals[0])


def face_ce_check(game: SlotGame, n_samples: int = 3000, cg_rounds: int = 15, tols=(1e-4, 1e-5, 1e-6),
                  seed: int = 0) -> FaceCheck:
    rng = np.random.default_rng(seed)
    x, _ = large_crowd_equilibrium(game, tol=1e-7, iters=40000)
    mask = best_response_face(game, x)
    z = game.counts.astype(float)
    multi = [k for k in range(game.K) if mask[k].sum() > 1]
    menu = [x]
    for k in multi:  # vertices: one type moves entirely to one of its optimal slots
        for t in np.flatnonzero(mask[k]):
            y = x.copy()
            y[k] = 0
            y[k, t] = z[k]
            menu.append(y)

    def random_face_point():
        y = x.copy()
        conc = rng.choice([0.1, 0.5, 1.0, 5.0])
        for k in multi:
            if rng.random() < 0.7:
                idx = np.flatnonzero(mask[k])
                y[k] = 0
                y[k, idx] = rng.dirichlet(np.full(len(idx), conc)) * z[k]
        return y

    menu += [random_face_point() for _ in range(n_samples)]
    tol0 = tols[0]
    nash = game.social_cost(x)
    for _ in range(cg_rounds):  # column generation on the face
        cost, p, lam, v0 = _ce_master(game, menu, tol0)
        starts = [menu[j] for j in np.flatnonzero(p > 1e-9)] + [random_face_point() for _ in range(8)]
        best_rc, best_x = 0.0, None
        for y in starts:
            y = _face_project(np.asarray(y, float), mask, z)
            F, G = _pricing(game, y, lam, tol0)
            step = 0.02 * game.n / max(np.abs(G).max(), 1e-9)
            for _ in range(200):
                while True:
                    y2 = _face_project(y - step * G, mask, z)
                    F2, G2 = _pricing(game, y2, lam, tol0)
                    if F2 <= F - 1e-4 * (G * (y - y2)).sum() or step < 1e-14:
                        break
                    step *= 0.5
                done = F - F2 < 1e-12 * abs(F)
                y, F, G, step = y2, F2, G2, step * 1.5
                if done:
                    break
            if F - v0 < best_rc:
                best_rc, best_x = F - v0, y
        if best_x is None or best_rc > -1e-9 * nash:
            break
        menu.append(best_x)
    saving = {}
    for tol in tols:
        cost, *_ = _ce_master(game, menu, tol)
        saving[tol] = 60 * (nash - cost) / game.n
    cheapest = min(game.social_cost(m) for m in menu) / game.n
    return FaceCheck(nash / game.n, cheapest, int((mask.sum(1) - 1).sum()), len(menu), saving, cg_rounds)
