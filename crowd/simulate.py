"""Layer 3 - the day-to-day simulator: information policies (pi0-pi8), behaviour models, and the simulation loop."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .correlated import recommend
from .game import CongestionModel


# ============================================================================
# Information policies: what the display tells people before they choose.
#
# Each policy produces, for every person, an *estimate* est[i, t] of tonight's
# congestion cost (minutes) in each slot. Recommendation policies additionally
# hand some people a personal slot.
#
#     pi0  NoInfo        nothing; people use their own memory of past visits
#     pi1  Live          the exact crowd cost of the most recent meal, for all
#     pi2  Live(delay=d) the same, but d meals old
#     pi3  Noisy         recent cost + private random error (each person sees a different number)
#     pi4  Coarse        only K levels (e.g. Low / Medium / High)
#     pi5  Partial       one of the above, but only a share of people see it
#     pi6  Forecast      a smoothed average of past meals (exponential moving average)
#     pi7  Recommend     a private slot for each person drawn from a correlated equilibrium
# ============================================================================

class Policy:
    name = "policy"

    def setup(self, n: int, T: int, rng: np.random.Generator) -> None:
        self.n, self.T = n, T

    def estimate(self, history: list[np.ndarray], memory: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return memory

    def recommendation(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
        """(slot for each person, whether each person follows it) or None."""
        return None


@dataclass
class NoInfo(Policy):
    name: str = "pi0 No information"


@dataclass
class Live(Policy):
    delay: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or ("pi1 Exact live" if self.delay == 1 else f"pi2 Delayed ({self.delay} meals)")

    def estimate(self, history, memory, rng):
        if len(history) < self.delay:
            return memory
        return np.broadcast_to(history[-self.delay], memory.shape)


@dataclass
class Noisy(Policy):
    sigma: float = 3.0  # minutes of private noise
    delay: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"pi3 Noisy (sd {self.sigma:g} min)"

    def estimate(self, history, memory, rng):
        if len(history) < self.delay:
            return memory
        return np.maximum(0.0, history[-self.delay][None, :] + rng.normal(0, self.sigma, memory.shape))


@dataclass
class Coarse(Policy):
    """Show only a level (Low / Medium / High ...) for each slot.

    Levels are equal-width bins on [0, top] minutes (everything above `top`
    is the top level). People read a level as the average cost of all past
    slots that showed that level -- they learn what "Medium" usually means.
    """

    levels: int = 3
    top: float = 15.0
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"pi4 Coarse ({self.levels} levels)"
        edges = np.linspace(0, self.top, self.levels + 1)
        self._edges = edges[1:-1]
        self._prior = 0.5 * (edges[:-1] + edges[1:])  # before any data: the bin midpoint

    def estimate(self, history, memory, rng):
        if not history:
            return memory
        past = np.concatenate(history)
        past_level = np.digitize(past, self._edges)
        meaning = np.array([past[past_level == j].mean() if np.any(past_level == j) else self._prior[j]
                            for j in range(self.levels)])
        level = np.digitize(history[-1], self._edges)
        return np.broadcast_to(meaning[level], memory.shape)


@dataclass
class Partial(Policy):
    inner: Policy = field(default_factory=Live)
    share: float = 0.3  # fraction of people who see the display
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"pi5 Partial ({self.share:.0%} see {self.inner.name.split(' ', 1)[1]})"

    def setup(self, n, T, rng):
        super().setup(n, T, rng)
        self.inner.setup(n, T, rng)
        self.sees = rng.random(n) < self.share

    def estimate(self, history, memory, rng):
        shown = np.asarray(self.inner.estimate(history, memory, rng))
        return np.where(self.sees[:, None], shown, memory)


@dataclass
class Forecast(Policy):
    alpha: float = 0.3  # weight on the newest meal
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"pi6 Forecast (EMA {self.alpha:g})"

    def estimate(self, history, memory, rng):
        if not history:
            return memory
        f = history[0].copy()
        for g in history[1:]:
            f = (1 - self.alpha) * f + self.alpha * g
        return np.broadcast_to(f, memory.shape)


@dataclass
class Recommend(Policy):
    """pi7: private slot recommendations drawn by shuffling a target schedule inside each type.

    target       X[k, t]: how many type-k people to send to slot t (e.g. a Nash
                 equilibrium -> a correlated equilibrium; or the optimum).
    type_of      type index of every individual.
    compliance   share of people who use the app and consider the advice.
    rational     if True, an app user disobeys when disobeying would save more
                 than `tolerance` minutes (given the published policy); they
                 then go to their best alternative. If False they always obey.
    gains        (K, T) best gain from disobeying (from congestion.correlated.obedience)
    best_alt     (K, T) the slot a tempted person would switch to.
    bonus        (K, T) reward for showing up in the recommended slot (minutes-
                 equivalent, e.g. a priority token). A person is tempted only if
                 gains - bonus > tolerance. With bonus = the obedience incentive
                 the recommendation is a CE of the rewarded game (pi8).
    fallback     what non-users look at (default: their own memory).
    """

    target: np.ndarray = None
    type_of: np.ndarray = None
    compliance: float = 1.0
    rational: bool = False
    gains: np.ndarray | None = None
    best_alt: np.ndarray | None = None
    bonus: np.ndarray | None = None
    tolerance: float = 0.1
    fallback: Policy = field(default_factory=NoInfo)
    name: str = "pi7 CE recommendation"

    def setup(self, n, T, rng):
        super().setup(n, T, rng)
        self.fallback.setup(n, T, rng)
        self.users = rng.random(n) < self.compliance

    def estimate(self, history, memory, rng):
        return self.fallback.estimate(history, memory, rng)

    def recommendation(self, rng):
        rec = recommend(rng, self.type_of, self.target)
        follow = self.users.copy()
        if self.rational and self.gains is not None:
            k = self.type_of
            net = self.gains - (self.bonus if self.bonus is not None else 0.0)
            tempted = net[k, rec] > self.tolerance
            rec = np.where(follow & tempted, self.best_alt[k, rec], rec)
        return rec, follow


# ============================================================================
# How people choose a slot each day.
#
# Signal-response behaviour (uses the display's estimate):
#     Logit(lam, revise)   with probability `revise` a person reconsiders today;
#                          if so they pick slot t with probability
#                          proportional to exp(-lam * (est[t] + beta*|t - tau|))
#                          (quantal / logit response). lam = inf is exact best
#                          response. Otherwise they repeat their last slot (habit).
#
# Learning behaviour (ignores the display's estimate, learns from feedback):
#     Hedge(eta)           full information: after each meal the app publishes
#                          every slot's cost, so a person can update all slots.
#                          No external regret  -> play approaches a coarse CE.
#     EXP3(eta, explore)   bandit feedback: a person only knows their own cost.
#     RegretMatching()     Hart & Mas-Colell style: switch from your usual slot
#                          to t with probability proportional to the regret of
#                          not having gone to t on the days you went to your
#                          usual slot. No swap regret -> play approaches a CE.
# ============================================================================

class Behaviour:
    name = "behaviour"
    uses_estimate = True

    def setup(self, tau, beta, T, rng) -> None:
        self.tau, self.beta, self.T = tau, beta, T
        self.sched = beta[:, None] * np.abs(np.arange(T)[None, :] - tau[:, None])

    def choose(self, est, prev, rng) -> np.ndarray:
        raise NotImplementedError

    def update(self, slots, cost_vectors, attended, rng) -> None:
        """cost_vectors[i, t]: cost person i would have paid in slot t today."""


@dataclass
class Logit(Behaviour):
    lam: float = 1.0
    revise: float = 0.3
    name: str = ""

    def __post_init__(self) -> None:
        lam = "inf" if np.isinf(self.lam) else f"{self.lam:g}"
        self.name = self.name or f"logit (lambda={lam}, revise={self.revise:g})"

    def choose(self, est, prev, rng):
        cost = np.asarray(est) + self.sched
        if np.isinf(self.lam):
            score = -cost + 1e-9 * rng.random(cost.shape)  # random tie-breaking
        else:
            score = -self.lam * cost + rng.gumbel(size=cost.shape)  # Gumbel-max = logit sampling
        new = np.argmax(score, axis=1)
        keep = rng.random(len(prev)) >= self.revise
        return np.where(keep, prev, new)


@dataclass
class Hedge(Behaviour):
    eta: float = 0.05
    name: str = ""
    uses_estimate = False

    def __post_init__(self) -> None:
        self.name = self.name or f"Hedge (eta={self.eta:g}, full feedback)"

    def setup(self, tau, beta, T, rng):
        super().setup(tau, beta, T, rng)
        self.cum = self.sched.copy() * 0.0

    def choose(self, est, prev, rng):
        z = -self.eta * (self.cum - self.cum.min(axis=1, keepdims=True))
        return np.argmax(z + rng.gumbel(size=z.shape), axis=1)

    def update(self, slots, cost_vectors, attended, rng):
        self.cum[attended] += cost_vectors[attended]


@dataclass
class EXP3(Behaviour):
    eta: float = 0.05
    explore: float = 0.05
    scale: float = 30.0  # costs are divided by this to lie roughly in [0, 1]
    name: str = ""
    uses_estimate = False

    def __post_init__(self) -> None:
        self.name = self.name or f"EXP3 (eta={self.eta:g}, own feedback only)"

    def setup(self, tau, beta, T, rng):
        super().setup(tau, beta, T, rng)
        self.cum = np.zeros((len(tau), T))
        self.p = np.full((len(tau), T), 1.0 / T)

    def choose(self, est, prev, rng):
        z = -self.eta * (self.cum - self.cum.min(axis=1, keepdims=True))  # cum is on the [0, 1] scale
        w = np.exp(z)
        self.p = (1 - self.explore) * w / w.sum(axis=1, keepdims=True) + self.explore / self.T
        u = rng.random(len(self.p))[:, None]
        return (self.p.cumsum(axis=1) < u).sum(axis=1).clip(0, self.T - 1)

    def update(self, slots, cost_vectors, attended, rng):
        idx = np.flatnonzero(attended)
        own = np.clip(cost_vectors[idx, slots[idx]] / self.scale, 0, 1)
        self.cum[idx, slots[idx]] += own / self.p[idx, slots[idx]]  # importance-weighted estimate


@dataclass
class RegretMatching(Behaviour):
    inertia: float = 0.5  # always stay with at least this probability
    name: str = ""
    uses_estimate = False

    def __post_init__(self) -> None:
        self.name = self.name or "Regret matching (full feedback)"

    def setup(self, tau, beta, T, rng):
        super().setup(tau, beta, T, rng)
        n = len(tau)
        self.R = np.zeros((n, T, T))  # R[i, j, k]: total regret for k on days i played j
        self.days = np.zeros(n)

    def choose(self, est, prev, rng):
        n = len(prev)
        r = np.maximum(self.R[np.arange(n), prev, :], 0.0) / np.maximum(self.days, 1)[:, None]
        r[np.arange(n), prev] = 0.0
        total = r.sum(axis=1, keepdims=True)
        room = 1.0 - self.inertia
        q = np.where(total > room, r * room / np.where(total > 0, total, 1), r)
        q[np.arange(n), prev] = 1.0 - q.sum(axis=1)
        u = rng.random(n)[:, None]
        return (q.cumsum(axis=1) < u).sum(axis=1).clip(0, self.T - 1)

    def update(self, slots, cost_vectors, attended, rng):
        idx = np.flatnonzero(attended)
        j = slots[idx]
        c = cost_vectors[idx]
        self.R[idx, j, :] += c[np.arange(len(idx)), j][:, None] - c
        self.days[idx] += 1


# ============================================================================
# Day-to-day simulator: the same people choose a slot every day.
#
# One simulated "day" is one meal (one dinner, say). Each day:
#   1. the policy shows everyone an estimate of each slot's crowd cost (and
#      maybe a private recommendation);
#   2. every person picks a slot (behaviour model);
#   3. loads, queues and costs are computed with the congestion model;
#   4. people update their memory / learning state; the meal's crowd costs
#      become part of the history the display can show tomorrow.
#
# Metrics (after a burn-in period):
#   mean_cost     average cost per person per day (wait + schedule + scarcity), minutes
#   mean_wait     average waiting time per person per day, minutes
#   p90_wait      90th percentile of individual waits (bad-day experience)
#   swing         share of the crowd whose slot load moved vs. the previous day
#                 (0.5 * |A_d - A_{d-1}|_1 / n): how much the rush jumps around
#   fairness      Jain's index of each person's average cost (1 = perfectly equal)
#   ext_regret    average external regret per day (best fixed slot in hindsight)
#   swap_regret   average swap regret per day (best "whenever I went to j, go to k" rule)
# ============================================================================

@dataclass
class SimResult:
    loads: np.ndarray  # (days, T)
    crowd_cost: np.ndarray  # (days, T) congestion cost g per slot
    slots: np.ndarray  # (days, n)
    cost: np.ndarray  # (days, n) NaN when absent
    wait: np.ndarray  # (days, n) NaN when absent
    ext_regret_curve: np.ndarray  # (days,) population-average external regret per day so far
    swap_regret_curve: np.ndarray  # (days,)
    burn_in: int
    policy: str
    behaviour: str
    metrics: dict[str, float] = field(default_factory=dict)


def jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.all(x == 0):
        return 1.0
    return float(x.sum() ** 2 / (len(x) * (x**2).sum()))


def simulate(congestion: CongestionModel, tau: np.ndarray, beta: np.ndarray, policy: Policy | None = None,
             behaviour: Behaviour | None = None, days: int = 150, burn_in: int = 50, seed: int = 0,
             attend_prob: float | np.ndarray = 1.0, memory_rate: float = 0.3) -> SimResult:
    rng = np.random.default_rng(seed)
    tau = np.asarray(tau, dtype=int)
    beta = np.asarray(beta, dtype=float)
    n, T = len(tau), congestion.T
    policy = policy or NoInfo()
    behaviour = behaviour or Logit()
    policy.setup(n, T, rng)
    behaviour.setup(tau, beta, T, rng)
    sched = beta[:, None] * np.abs(np.arange(T)[None, :] - tau[:, None])
    attend_prob = np.asarray(attend_prob, dtype=float)  # scalar, (n,) or (days, n)
    attend_prob = np.broadcast_to(attend_prob, (days, n) if attend_prob.ndim == 2 else (n,))

    memory = np.zeros((n, T))  # personal estimate of each slot's crowd cost (optimistic start)
    prev = tau.copy()
    history: list[np.ndarray] = []
    loads = np.zeros((days, T))
    crowd = np.zeros((days, T))
    slots = np.zeros((days, n), dtype=int)
    cost = np.full((days, n), np.nan)
    wait = np.full((days, n), np.nan)
    cum_actual = np.zeros(n)
    cum_cf = np.zeros((n, T))
    swap = np.zeros((n, T, T))
    attended_days = np.zeros(n)
    ext_curve = np.zeros(days)
    swap_curve = np.zeros(days)

    for d in range(days):
        here = rng.random(n) < (attend_prob[d] if attend_prob.ndim == 2 else attend_prob)
        est = policy.estimate(history, memory, rng) if behaviour.uses_estimate else None
        s = behaviour.choose(est, prev, rng)
        rec = policy.recommendation(rng)
        if rec is not None:
            rec_slot, follow = rec
            s = np.where(follow, rec_slot, s)

        A = np.bincount(s[here], minlength=T).astype(float)
        g = congestion.cost(A)
        w = congestion.wait(A)
        cv = g[None, :] + sched  # what each person would pay in each slot today
        loads[d], crowd[d], slots[d] = A, g, s
        cost[d, here] = cv[here, s[here]]
        wait[d, here] = w[s[here]]

        # memory: people remember the crowd they experienced themselves
        idx = np.flatnonzero(here)
        memory[idx, s[idx]] = (1 - memory_rate) * memory[idx, s[idx]] + memory_rate * g[s[idx]]
        behaviour.update(s, cv, here, rng)
        prev = np.where(here, s, prev)
        history.append(g)

        # regret bookkeeping (counterfactuals use today's loads: one person is negligible)
        cum_actual[idx] += cv[idx, s[idx]]
        cum_cf[idx] += cv[idx]
        swap[idx, s[idx], :] += cv[idx, s[idx]][:, None] - cv[idx]
        attended_days[idx] += 1
        denom = np.maximum(attended_days, 1)
        ext_curve[d] = np.mean((cum_actual - cum_cf.min(axis=1)) / denom)
        swap_curve[d] = np.mean(np.maximum(swap, 0).max(axis=2).sum(axis=1) / denom)

    res = SimResult(loads, crowd, slots, cost, wait, ext_curve, swap_curve, burn_in, policy.name, behaviour.name)
    res.metrics = summarise(res, congestion)
    return res


def summarise(res: SimResult, congestion: CongestionModel) -> dict[str, float]:
    b = res.burn_in
    c, w, A = res.cost[b:], res.wait[b:], res.loads[b:]
    # day d (d >= 1) moved 0.5 * |A_d - A_{d-1}|_1 / (people present on day d)
    change = 0.5 * np.abs(np.diff(res.loads, axis=0)).sum(axis=1) / np.maximum(res.loads[1:].sum(axis=1), 1)
    moved = change[max(b, 1) - 1:]
    per_person = np.nanmean(c, axis=0)
    return {
        "mean_cost": float(np.nanmean(c)),
        "mean_wait": float(np.nanmean(w)),
        "p90_wait": float(np.nanpercentile(w, 90)),
        "swing": float(np.mean(moved)) if len(moved) else 0.0,
        "peak_load": float(A.max(axis=1).mean()),
        "fairness": jain_index(per_person),
        "ext_regret": float(res.ext_regret_curve[-1]),
        "swap_regret": float(res.swap_regret_curve[-1]),
    }
