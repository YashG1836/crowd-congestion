"""Run every experiment; writes results/figures/*.png, results/RESULTS.md and the website's docs/data.js.

    python experiments.py            # full run (about 10 minutes)
    python experiments.py --quick    # 5 seeds instead of 30 (for a fast check)

General part (five shared facilities: road, network, restaurant, gym, mess):
    E1  small games      Pigou, the two-student dinner game, a traffic light, learning to a CE
    E2  selfish timing   Nash equilibrium vs the planner's optimum (price of anarchy)
    E3  public display   what a shared crowd display does, under three ways of reacting
    E4  CE advice        private recommendations: obedience, best-CE LP, how many must follow, rewards
    E5  uncertainty      who should be told what when the operator knows tonight's state
    E6  learning         students who learn from experience end near a correlated equilibrium
    E8  small crowds     identical people solved exactly: when correlation beats Nash, and when privacy pays
Case study:
    E7  the IITGN mess   synthetic QR-scan logs, calibration, every meal, the dinner day by day
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from crowd import viz
from crowd.correlated import (best_ce, face_ce_check, information_design, interpolated_schedules, large_crowd_check,
                              obedience, obedience_incentive, who_gains)
from crowd.domains import DOMAINS, build
from crowd.game import (FluidQueue, benchmarks, booking, continuous_optimum, local_search, random_queue_cost,
                        round_schedule)
from crowd.smallcrowd import SmallCrowd, crowd_shapes, full_profile_ce, large_crowd_bad_day, small_bad_day
from crowd.mess import (MEALS, MESSES, arrival_heatmap, estimate_preferred_slots, estimate_service_rate, generate,
                        make_roster, meal_instance, nowcast_errors, preference_accuracy, roster_truth,
                        scans_per_slot)
from crowd.simulate import (EXP3, Coarse, Forecast, Hedge, Live, Logit, Noisy, NoInfo, Partial, Recommend,
                            RegretMatching, simulate)
from crowd.theory import (critical_responsiveness, dinner_game, hedge, intersection_game, pigou_poa, regret_matching,
                          shared_counter_dynamics)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = OUT / "figures"
SUMMARY: dict = {}
SITE: dict = {}
PER_SEED: dict = {}  # (domain, regime, policy) -> mean cost of every seed (for paired tests)

REGIMES = {
    "calm": Logit(3.0, 0.3),       # 30% of people reconsider each day
    "reactive": Logit(3.0, 1.0),   # everyone reconsiders each day
    "noisy": Logit(1.0, 0.3),      # low rationality
}
DOMAIN_ORDER = ["road", "network", "restaurant", "gym", "mess"]
QUEUE_RUNS = 200      # simulated runs of the random-arrival queue (E2)
FACE_SAMPLES = 3000   # random schedules on the best-response face (E4)
FACE_ROUNDS = 15      # column-generation rounds on the face (E4)
SHORT = {"road": "Road", "network": "Network", "restaurant": "Restaurant", "gym": "Gym", "mess": "Mess"}
POLICY_COLORS = {"no information": viz.MUTED, "public display": viz.NE, "display for 30%": viz.SERIES[3],
                 "CE advice": viz.CE, "optimum as advice": viz.SERIES[4], "CE advice + reward": viz.OPT}


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    viz.plt.close(fig)


def plain_log(ax, axis: str, ticks) -> None:
    """Log axis with ordinary numbers (2, 5, 10) instead of 10^x labels."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(lambda v, _: f"{v:g}")
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + viz.SEQ_BLUE)


# ------------------------------------------------------------------ shared set-up (computed once)
@lru_cache(maxsize=None)
def domain(key: str):
    """Instance, equilibrium/optimum benchmarks, and what a tempted person would do, for one domain."""
    inst = build(DOMAINS[key])
    b = benchmarks(inst.game)
    ob_opt = obedience(inst.game, b.optimum.X)
    ob_ne = obedience(inst.game, b.best_nash.X)
    g = inst.game
    alt_opt = g.deviation_costs(b.optimum.X).argmin(axis=2)
    alt_ne = g.deviation_costs(b.best_nash.X).argmin(axis=2)
    return inst, b, ob_ne, ob_opt, alt_ne, alt_opt


@lru_cache(maxsize=None)
def jaiswal(day: str = "normal"):
    inst = meal_instance("Jaiswal", "dinner", day)
    b = benchmarks(inst.game)
    g = inst.game
    return (inst, b, obedience(g, b.best_nash.X), obedience(g, b.optimum.X),
            g.deviation_costs(b.best_nash.X).argmin(axis=2), g.deviation_costs(b.optimum.X).argmin(axis=2))


def core_policies(setup) -> dict:
    """The six information policies compared in every domain."""
    inst, b, ob_ne, ob_opt, alt_ne, alt_opt = setup
    t = inst.type_of
    tol = max(0.1, b.epsilon) + 1e-9  # disobey only if it saves more than 6 s (or than the schedule's own accuracy)
    return {
        "no information": NoInfo(),
        "public display": Live(),
        "display for 30%": Partial(Live(), 0.3),
        "CE advice": Recommend(target=b.best_nash.X, type_of=t, rational=True, gains=ob_ne.gain_by_cell,
                               best_alt=alt_ne, tolerance=tol, name="CE advice"),
        "optimum as advice": Recommend(target=b.optimum.X, type_of=t, rational=True, gains=ob_opt.gain_by_cell,
                                       best_alt=alt_opt, name="optimum as advice"),
        "CE advice + reward": Recommend(target=b.optimum.X, type_of=t, rational=True, gains=ob_opt.gain_by_cell,
                                        best_alt=alt_opt, bonus=ob_opt.gain_by_cell, name="CE advice + reward"),
    }


def run(inst, policy, regime: str, seed: int, days: int = 150, burn: int = 50):
    g = inst.game
    return simulate(g.congestion, inst.tau, inst.beta, policy, REGIMES[regime], days=days, burn_in=burn, seed=seed)


# =============================================================== E1 small games
def e1_small_games() -> None:
    plt = viz.plt
    g = dinner_game()
    mixed = [m for m in g.mixed_nash_2p() if 0 < m[0][0] < 1][0]
    counter = float(np.mean([g.total_cost(s) for s in shared_counter_dynamics(g, (1, 1), 20)]))
    mixed_cost = float(g.expected_costs(g.product_distribution(list(mixed))).sum())
    ce = g.correlated_equilibrium("fair")
    ig = intersection_game()
    best = ig.correlated_equilibrium("best")
    nash_payoffs = [-ig.expected_costs(ig.product_distribution([x, y])) for x, y in ig.mixed_nash_2p()]
    rm = regret_matching(ig, rounds=6000, seed=0, log_every=25)
    hd = hedge(ig, rounds=6000, seed=0, log_every=25)
    SUMMARY["e1"] = {
        "dinner": {"shared counter": counter, "no information (mixed Nash)": mixed_cost,
                   "private coin (CE)": ce.total_cost, "optimum": g.social_optimum()[1], "mixed_peak_prob": float(mixed[0][0])},
        "intersection": {"best_ce_each": float(-best.expected_costs[0]), "best_symmetric_nash": float(max(min(u) for u in nash_payoffs)),
                         "support": best.support()},
        "pigou": {str(p): pigou_poa(p) for p in (1, 2, 4, 8)},
        "learning": {"regret_matching_ce_gap": float(rm.ce_violation[-1]), "hedge_cce_gap": float(hd.cce_violation[-1])},
        "delay": {str(d): critical_responsiveness(d) for d in range(5)},
    }
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.5))
    vals = [counter, mixed_cost, ce.total_cost]
    ax[0].bar(range(3), vals, 0.6, color=[viz.NE, viz.MUTED, viz.CE])
    ax[0].axhline(g.social_optimum()[1], color=viz.OPT, ls="--", lw=1.4)
    ax[0].text(2.45, g.social_optimum()[1] + 0.06, "optimum", color=viz.OPT, ha="right", fontsize=9)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.06, f"{v:.1f}", ha="center", fontsize=10, color=viz.TEXT_2)
    ax[0].set_xticks(range(3), ["shared counter", "no information\n(mixed Nash)", "private coin\n(correlated eq.)"])
    ax[0].set(title="Two students, Peak or Late: total cost", ylim=(0, 4.2))
    outcomes = ["both stop", "one goes", "crash"]
    ce_p = [best.p[0], best.p[1] + best.p[2], best.p[3]]
    mix = [m for m in ig.mixed_nash_2p() if 0 < m[0][0] < 1][0]
    ne_p = ig.product_distribution(list(mix))
    ne_p = [ne_p[0], ne_p[1] + ne_p[2], ne_p[3]]
    x = np.arange(3)
    ax[1].bar(x - 0.2, ce_p, 0.38, color=viz.CE, label=f"correlated eq. ({-best.expected_costs[0]:.2f} each)")
    ax[1].bar(x + 0.2, ne_p, 0.38, color=viz.NE, label=f"mixed Nash ({14 / 3:.2f} each)")
    ax[1].set_xticks(x, outcomes)
    ax[1].set(title="A traffic light: how often each outcome happens", ylabel="probability", ylim=(0, 0.75))
    ax[1].legend(fontsize=8.5)
    ax[2].plot(rm.rounds_logged, rm.ce_violation, color=viz.CE, label="regret matching (CE gap)")
    ax[2].plot(hd.rounds_logged, hd.cce_violation, color=viz.SERIES[3], label="Hedge (coarse-CE gap)")
    ax[2].set(title="Players who learn find a correlated equilibrium", xlabel="rounds played", yscale="log",
              ylabel="largest gain from deviating")
    ax[2].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig1_small_games")


# =============================================================== E2 selfish timing
def e2_selfish() -> None:
    plt = viz.plt
    rows = []
    for key in DOMAIN_ORDER:
        inst, b, *_ = domain(key)
        g, d = inst.game, DOMAINS[key]
        ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
        rows.append({"domain": key, "title": d.title, "people": g.n, "unit": d.unit, "slot_min": d.slot_minutes,
                     "service_rate": d.service_rate, "nash": ne["per_person"], "nash_wait": ne["wait_per_person"],
                     "opt": op["per_person"], "opt_wait": op["wait_per_person"], "poa": b.poa,
                     "eps_sec": 60 * b.epsilon})
    # --- is the fluid queue too kind to the planner? Same schedules, random arrival times inside each slot
    robust = []
    for key in DOMAIN_ORDER:
        inst, b, *_ = domain(key)
        g = inst.game
        ne, op = b.nash.X, b.optimum.X
        r_ne, r_op = random_queue_cost(g, ne, QUEUE_RUNS), random_queue_cost(g, op, QUEUE_RUNS)
        s_ne = random_queue_cost(g, ne, QUEUE_RUNS, random_service=True)
        s_op = random_queue_cost(g, op, QUEUE_RUNS, random_service=True)
        tuned, factor = r_op, 1.0  # the planner plans for a slightly slower counter to leave a buffer
        for f in (0.95, 0.9, 0.85, 0.8):
            g2 = g.with_congestion(FluidQueue(g.congestion.service_rate * f, g.T, g.congestion.slot_minutes))
            x = local_search(g2, round_schedule(g2, continuous_optimum(g2))).X
            c = random_queue_cost(g, x, QUEUE_RUNS)
            if c < tuned:
                tuned, factor = c, f
        books = [booking(g, 1.0, seed) for seed in range(5)]
        robust.append({"domain": key, "nash_fluid": g.social_cost(ne) / g.n, "opt_fluid": g.social_cost(op) / g.n,
                       "nash_random": r_ne, "opt_random": r_op, "nash_random_service": s_ne,
                       "opt_random_service": s_op, "opt_tuned": tuned, "tuned_factor": factor,
                       "booking_fluid": float(np.mean([g.social_cost(x) / g.n for x in books])),
                       "booking_random": float(np.mean([random_queue_cost(g, x, 100) for x in books]))})
    SUMMARY["e2"] = {"domains": rows, "robust": robust}
    fig_selfish()


def fig_selfish() -> None:
    """The commute under selfish timing and the planner, and the price of anarchy under three queue models."""
    plt = viz.plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    inst, b, *_ = domain("road")
    g = inst.game
    xs = np.arange(g.T)
    ax = axes[0]
    ax.bar(xs - 0.2, g.loads(b.nash.X), 0.38, color=viz.NE, label="selfish (Nash)")
    ax.bar(xs + 0.2, g.loads(b.optimum.X), 0.38, color=viz.OPT, label="planner (optimum)")
    ax.axhline(g.congestion.capacity, color=viz.MUTED, ls="--", lw=1.2)
    ax.text(-0.45, g.congestion.capacity + 6, "bridge capacity per slot", ha="left", color=viz.TEXT_2, fontsize=9)
    ax.set_xticks(xs[::2], g.slot_labels[::2])
    ax.set(title="Morning commute: cars arriving per 15 minutes", ylabel="cars", ylim=(0, 400))
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    rob = pd.DataFrame(SUMMARY["e2"]["robust"])
    ax = axes[1]
    series = [("fluid queue (model)", rob.nash_fluid / rob.opt_fluid, viz.NE),
              ("random arrivals", rob.nash_random / rob.opt_random, viz.SERIES[3]),
              ("random arrivals and service", rob.nash_random_service / rob.opt_random_service, viz.MUTED),
              ("random arrivals, planner keeps a buffer", rob.nash_random / rob.opt_tuned, viz.CE)]
    w = 0.2
    for j, (name, vals, col) in enumerate(series):
        ax.bar(np.arange(len(rob)) + (j - 1.5) * w, vals, w * 0.92, color=col, label=name)
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    ax.set_xticks(range(len(rob)), [SHORT[k] for k in rob.domain])
    ax.set(title="Price of anarchy (selfish / planner) under three queue models", ylabel="price of anarchy")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_ylim(0, 10)
    fig.tight_layout()
    save(fig, "fig2_selfish_vs_planner")


# =============================================================== E3 public display
def e3_display(seeds: int) -> None:
    plt = viz.plt
    rows = []
    for key in DOMAIN_ORDER:
        setup = domain(key)
        inst, b = setup[0], setup[1]
        nash = b.best_nash.social_cost / inst.game.n
        for regime in REGIMES:
            for name, pol in core_policies(setup).items():
                ms = [run(inst, pol, regime, s).metrics for s in range(seeds)]
                PER_SEED[(key, regime, name)] = [m["mean_cost"] for m in ms]
                c, h = ci([m["mean_cost"] for m in ms])
                rows.append({"domain": key, "regime": regime, "policy": name, "cost": c, "ci": h, "vs_nash": c / nash,
                             "swing": float(np.mean([m["swing"] for m in ms])),
                             "regret": float(np.mean([m["swap_regret"] for m in ms])),
                             "p90": float(np.mean([m["p90_wait"] for m in ms]))})
    SUMMARY["e3"] = rows
    df = pd.DataFrame(rows)
    pols = list(POLICY_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    w = 0.13
    for ax, regime in zip(axes, REGIMES):
        sub = df[df.regime == regime]
        for j, pol in enumerate(pols):
            cells = [sub[(sub.domain == k) & (sub.policy == pol)].iloc[0] for k in DOMAIN_ORDER]
            vals = [c.vs_nash for c in cells]
            errs = [c.ci / c.cost * c.vs_nash for c in cells]  # 95% CI, same relative width
            ax.bar(np.arange(5) + (j - 2.5) * w, vals, w * 0.92, color=POLICY_COLORS[pol], label=pol, yerr=errs,
                   error_kw=dict(elinewidth=0.8, capsize=1.5, ecolor=viz.TEXT_2))
        ax.axhline(1, color=viz.TEXT_2, lw=1)
        ax.set_xticks(range(5), [SHORT[k] for k in DOMAIN_ORDER])
        ax.set_title({"calm": "Calm crowd (30% reconsider daily)", "reactive": "Reactive crowd (everyone reconsiders)",
                      "noisy": "Noisy crowd (low rationality)"}[regime])
        plain_log(ax, "y", [0.1, 0.3, 1, 3, 10])
    axes[0].set_ylabel("cost relative to Nash (log)")
    axes[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig3_public_display")


# =============================================================== E4 correlated advice at scale
def e4_ce(seeds: int) -> None:
    plt = viz.plt
    table, comp = [], []
    shares = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    for key in DOMAIN_ORDER:
        setup = domain(key)
        inst, b, ob_ne, ob_opt, alt_ne, alt_opt = setup
        g = inst.game
        n = g.n
        sched, labels = interpolated_schedules(g, b.best_nash.X, b.optimum.X, steps=20)
        sched += [s.X for s in b.all_nash]
        atomic = best_ce(g, sched, labels, slack=b.epsilon)
        lc = large_crowd_check(g)
        inc = obedience_incentive(g, b.optimum.X, b.best_nash.social_cost)
        wg = who_gains(g, b.best_nash.X, b.optimum.X)
        table.append({"domain": key, "nash": b.best_nash.social_cost / n, "opt": b.optimum.social_cost / n,
                      "ce_max_gain_sec": 60 * ob_ne.max_gain, "opt_tempted": ob_opt.share_tempted,
                      "opt_max_gain_min": ob_opt.max_gain, "best_ce_atomic": atomic.expected_cost / n,
                      "eps_sec": 60 * b.epsilon, "large_nash": lc.nash, "large_best_ce": lc.best_ce,
                      "large_gap_sec": lc.gap_sec, "reward_pp": inc.per_person, "rewarded": inc.share_rewarded,
                      "reward_per_rewarded": inc.per_person / max(inc.share_rewarded, 1e-9),
                      "saving_pp": inc.welfare_gain / n, "worse_off_opt": wg.worse_off("optimum"),
                      "worse_off_reward": wg.worse_off("with_reward"), "max_loss_opt_sec": 60 * wg.max_loss("optimum")})
        nash = b.best_nash.social_cost / n
        for sh in shares:
            pol = Recommend(target=b.best_nash.X, type_of=inst.type_of, compliance=sh, rational=True,
                            gains=ob_ne.gain_by_cell, best_alt=alt_ne, fallback=Live(),
                            tolerance=max(0.1, b.epsilon) + 1e-9)
            c, h = ci([run(inst, pol, "reactive", s).metrics["mean_cost"] for s in range(max(3, seeds // 3))])
            comp.append({"domain": key, "share": sh, "cost": c, "ci": h, "vs_nash": c / nash})
    # --- can ANY correlated equilibrium beat Nash in a large crowd? Search the best-response face
    face = []
    for key in DOMAIN_ORDER:
        r = face_ce_check(domain(key)[0].game, n_samples=FACE_SAMPLES, cg_rounds=FACE_ROUNDS)
        face.append({"domain": key, "nash": r.nash, "cheapest_on_face": r.cheapest_on_face, "face_dim": r.face_dim,
                     "menu": r.menu, **{f"saving_sec_tol_{60 * t:g}s": v for t, v in r.saving_sec.items()}})
    # --- app users who are noisy (logit) instead of rational; a reward margin on top of Proposition 3
    noisy = []
    for key in DOMAIN_ORDER:
        inst, b, ob_ne, ob_opt, *_ = domain(key)
        g = inst.game
        nash = b.best_nash.social_cost / g.n
        D_ne, D_op = g.deviation_costs(b.best_nash.X), g.deviation_costs(b.optimum.X)
        for lam in (3.0, 1.0):
            pols = {"CE advice": Recommend(target=b.best_nash.X, type_of=inst.type_of, noise=lam, alt_costs=D_ne)}
            for m in (0.0, 1.0, 2.0):
                pols[f"CE advice + reward, margin {m:g}"] = Recommend(target=b.optimum.X, type_of=inst.type_of, noise=lam,
                                                                      alt_costs=D_op, bonus=ob_opt.gain_by_cell, margin=m)
            for name, pol in pols.items():
                vals = [run(inst, pol, "calm", sd).metrics["mean_cost"] for sd in range(max(3, seeds // 3))]
                PER_SEED[(key, f"noisy {lam:g}", name)] = vals
                c, h = ci(vals)
                noisy.append({"domain": key, "lambda": lam, "policy": name, "vs_nash": c / nash, "ci": h / nash})
    SUMMARY["e4"] = {"table": table, "compliance": comp, "face": face, "noisy": noisy}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.7))
    ax = axes[0]
    cdf = pd.DataFrame(comp)
    for key, col in zip(DOMAIN_ORDER, [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.SERIES[3], viz.SERIES[6]]):
        sub = cdf[cdf.domain == key]
        ax.errorbar(sub.share, sub.vs_nash, yerr=sub.ci / sub.cost * sub.vs_nash, marker="o", ms=4.5, color=col,
                    label=SHORT[key], capsize=2, elinewidth=0.8)
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    plain_log(ax, "y", [1, 2, 5, 10, 20])
    ax.set(title="Reactive crowd: share of people using private advice", xlabel="share using the app (others see the display)",
           ylabel="cost relative to Nash (log)")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    t = pd.DataFrame(table)
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.reward_pp, 0.38, color=viz.SERIES[3], label="reward paid (average per person)")
    ax.bar(x + 0.2, t.saving_pp, 0.38, color=viz.OPT, label="time saved vs Nash (per person)")
    for xi, r in zip(x, table):
        ax.text(xi - 0.2, r["reward_pp"] * 1.15, f"{r['rewarded']:.0%}\npaid", ha="center", fontsize=7.5, color=viz.TEXT_2)
    ax.set_xticks(x, [SHORT[k] for k in t.domain])
    plain_log(ax, "y", [0.1, 0.3, 1, 3, 10])
    ax.set(title="Price of obedience: reward needed vs time saved", ylabel="minutes per person (log)", ylim=(0.08, 30))
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_ce_advice")


# =============================================================== E5 uncertainty / information design
def e5_information() -> None:
    plt = viz.plt
    out = {}
    for key in DOMAIN_ORDER:
        inst, *_ = domain(key)
        r = information_design(inst.game)
        out[key] = {"expected": r.expected, "per_state": r.per_state, "hint": r.hint, "support": r.private_support,
                    "beliefs": r.beliefs, "value": r.value, "envelope": r.envelope, "max_gap_sec": r.max_gap_sec,
                    "tol_sec": r.tol_sec, "menu": r.menu_size, "bad_state": DOMAINS[key].bad_state}
    # --- how bad, and how often? (fewer belief points: the sweep only needs the policies' costs)
    sweep = []
    for key in DOMAIN_ORDER:
        g = domain(key)[0].game
        settings = [(f, 0.3) for f in (0.5, 0.6, 0.85)] + [(0.7, p) for p in (0.1, 0.5)]
        for f, p in settings:
            r = information_design(g, bad_factor=f, p_bad=p, n_beliefs=11, steps=4)
            sweep.append({"domain": key, "capacity": f, "p_bad": p, **r.expected, "max_gap_sec": r.max_gap_sec})
        e = out[key]["expected"]
        sweep.append({"domain": key, "capacity": 0.7, "p_bad": 0.3, **e, "max_gap_sec": out[key]["max_gap_sec"]})
    SUMMARY["e5"] = out
    SUMMARY["e5_sweep"] = sweep
    fig_bad_days()


def fig_bad_days() -> None:
    """How bad a bad day must be for announcing to backfire, and the road's value of beliefs."""
    plt = viz.plt
    out = SUMMARY["e5"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    sw = pd.DataFrame(SUMMARY["e5_sweep"])
    sw = sw[np.isclose(sw.p_bad, 0.3)].sort_values("capacity")
    for key, col in zip(DOMAIN_ORDER, [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.SERIES[3], viz.SERIES[6]]):
        sub = sw[sw.domain == key]
        ax.plot(100 * sub.capacity, sub["announce the state"] / sub["tell nobody"], marker="o", ms=4.5, color=col,
                label=SHORT[key])
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    ax.text(60.5, 1.02, "above the line: announcing backfires", fontsize=8.5, color=viz.TEXT_2)
    ax.set(title="Announcing a bad day vs telling nobody (probability 0.3)", xlabel="capacity left on a bad day (%)",
           ylabel="announce / tell nobody")
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax = axes[1]
    r = out["road"]
    q, V, E = np.array(r["beliefs"]), np.array(r["value"]), np.array(r["envelope"])
    p0 = 0.3
    ax.plot(q, V, color=viz.TEXT_2, lw=1.8, label="cost if everyone believes q")
    ax.plot(q, E, color=viz.SERIES[3], lw=2, ls="--", label="best public hint (convex envelope)")
    ax.plot([0, 1], [V[0], V[-1]], color=viz.NE, lw=1.2, ls=":", label="announce the state")
    ax.scatter([p0], [r["expected"]["tell nobody"]], color=viz.MUTED, s=40, zorder=3)
    ax.scatter([p0], [r["expected"]["announce the state"]], color=viz.NE, s=40, zorder=3)
    ax.scatter([p0], [r["expected"]["best public hint"]], color=viz.SERIES[3], s=40, zorder=3)
    ax.axvline(p0, color=viz.GRID, lw=1)
    ax.set(title="Road (70% capacity): cost as a function of the crowd's belief",
           xlabel="belief q that today is a bad day", ylabel="expected minutes per car")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_uncertainty")


# =============================================================== E6 learning
def e6_learning(seeds: int) -> None:
    rows = []
    learners = {"Hedge (sees every slot's wait)": lambda: Hedge(0.05),
                "Regret matching (sees every slot's wait)": lambda: RegretMatching(),
                "EXP3 (sees only its own wait)": lambda: EXP3(0.05)}
    for key in DOMAIN_ORDER:
        inst, b, *_ = domain(key)
        g = inst.game
        nash = b.best_nash.social_cost / g.n
        for name, make in learners.items():
            ms = [simulate(g.congestion, inst.tau, inst.beta, Live(), make(), days=300, burn_in=100, seed=s).metrics
                  for s in range(max(3, seeds // 10))]
            rows.append({"domain": key, "learner": name, "vs_nash": float(np.mean([m["mean_cost"] for m in ms])) / nash,
                         "swap_regret": float(np.mean([m["swap_regret"] for m in ms])),
                         "swing": float(np.mean([m["swing"] for m in ms]))})
    SUMMARY["e6"] = rows


# =============================================================== E7 the IITGN mess (case study)
def e7_mess(seeds: int) -> None:
    plt = viz.plt
    # --- data and calibration
    scans, truth = generate(weeks=8, seed=0)
    rt = roster_truth(0).set_index("student_id")
    calib = []
    for meal in MEALS:
        for menu in ("normal", "special"):
            true_mu = MEALS[meal].service_rate * (0.8 if menu == "special" else 1.0)
            est = estimate_service_rate(scans, "Jaiswal", meal, menu)
            w = truth[(truth.meal == meal) & (truth.menu == menu) & (truth.mess == "Jaiswal")].wait_min
            calib.append({"meal": meal, "menu": menu, "true_mu": true_mu, "estimated_mu": round(est, 2),
                          "queue_forms": bool(w.max() > 1)})
    prefs = {meal: preference_accuracy(estimate_preferred_slots(scans, meal), rt[f"tau_{meal}"]) for meal in MEALS}
    now = []
    for meal in MEALS:
        for menu in ("normal", "special"):
            tables = [scans_per_slot(scans, m, meal, menu) for m in MESSES]
            errs = [nowcast_errors(t) for t in tables]
            now.append({"meal": meal, "menu": menu, **{k: float(np.mean([e[k] for e in errs])) for k in errs[0]},
                        "mean_scans_per_slot": float(np.mean([t.to_numpy().mean() for t in tables]))})
    # --- every mess, meal and day
    roster = make_roster(0)
    meals = []
    for mess in MESSES:
        for meal in MEALS:
            for day in ("normal", "special"):
                inst = meal_instance(mess, meal, day, roster=roster)
                g = inst.game
                b = benchmarks(g)
                ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
                meals.append({"mess": mess, "meal": meal, "day": day, "diners": g.n, "nash": ne["per_person"],
                              "opt": op["per_person"], "lost": ne["per_person"] - op["per_person"],
                              "poa": b.poa if ne["per_person"] >= 0.1 else None, "eps_sec": 60 * b.epsilon})
    mdf = pd.DataFrame(meals)
    hours = float((mdf.lost * mdf.diners)[mdf.day == "normal"].sum() / 60)
    agg = mdf.groupby(["meal", "day"])[["nash", "opt", "lost"]].mean().reset_index()
    agg = agg.sort_values(["meal", "day"], key=lambda c: c.map({m: i for i, m in enumerate(MEALS)}) if c.name == "meal" else c)
    agg = agg.reset_index(drop=True)  # breakfast, lunch, dinner; normal before special
    # --- Jaiswal dinner: all display policies (the brief's list) x three crowds
    setup = jaiswal("normal")
    inst, b, ob_ne, ob_opt, alt_ne, alt_opt = setup
    g = inst.game
    nash = b.best_nash.social_cost / g.n
    extra = {"display 3 dinners old": Live(delay=3), "display + private noise": Noisy(3.0),
             "display in 3 levels": Coarse(3), "moving-average forecast": Forecast(0.3)}
    pols = {**core_policies(setup), **extra}
    jrows = []
    for regime in REGIMES:
        for name, pol in pols.items():
            ms = [run(inst, pol, regime, s).metrics for s in range(seeds)]
            c, h = ci([m["mean_cost"] for m in ms])
            jrows.append({"regime": regime, "policy": name, "cost": c, "ci": h, "swing": float(np.mean([m["swing"] for m in ms])),
                          "p90": float(np.mean([m["p90_wait"] for m in ms])),
                          "regret": float(np.mean([m["swap_regret"] for m in ms]))})
    # --- sweeps (reactive crowd) and delay x reactivity
    sweeps = []
    for name, items in {"noise": [(v, Noisy(v)) for v in (0, 1, 2, 3, 5, 8, 12)],
                        "share": [(v, Partial(Live(), v)) for v in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)],
                        "levels": [(v, Coarse(v)) for v in (2, 3, 5, 10, 30)]}.items():
        for val, pol in items:
            c, h = ci([run(inst, pol, "reactive", s).metrics["mean_cost"] for s in range(seeds)])
            sweeps.append({"sweep": name, "value": val, "cost": c, "ci": h})
    revise = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    delays = [1, 2, 3, 4, 5]
    swing = np.zeros((len(revise), len(delays)))
    for i, r in enumerate(revise):
        for j, d in enumerate(delays):
            swing[i, j] = np.mean([simulate(g.congestion, inst.tau, inst.beta, Live(delay=d), Logit(3.0, r), days=150,
                                            burn_in=50, seed=s).metrics["swing"] for s in range(max(3, seeds // 3))])
    info = information_design(g)
    wg = who_gains(g, b.best_nash.X, b.optimum.X)
    inc_j = obedience_incentive(g, b.optimum.X, b.best_nash.social_cost)
    peak = (g.tau >= 3) & (g.tau <= 6)
    groups = {"flexible students": g.beta == 1.5, "average": g.beta == 3.0, "rigid students": g.beta == 6.0,
              "prefer 20:00-20:30": peak, "prefer other times": ~peak}
    gains = [{"group": nm, "nash": wg.group_mean("nash", m), "optimum": wg.group_mean("optimum", m),
              "with_reward": wg.group_mean("with_reward", m)} for nm, m in groups.items()]
    SUMMARY["e7"] = {"rows": int(len(scans)), "students": int(scans.student_id.nunique()), "calibration": calib,
                     "preferences": prefs, "nowcast": now, "meals": meals, "by_meal": agg.to_dict("records"),
                     "hours_lost_per_day": hours, "jaiswal": {"nash": nash, "opt": b.optimum.social_cost / g.n,
                                                             "policies": jrows}, "sweeps": sweeps,
                     "swing": {"revise": revise, "delays": delays, "values": swing.round(4).tolist()},
                     "information": {"expected": info.expected, "per_state": info.per_state, "hint": info.hint},
                     "who_gains": gains, "worse_off": {"optimum": wg.worse_off("optimum"),
                                                       "with_reward": wg.worse_off("with_reward")},
                     "max_loss_sec": 60 * wg.max_loss("optimum"), "budget_pp": wg.budget / g.n,
                     "reward": {"share": inc_j.share_rewarded, "per_person": inc_j.per_person,
                                "per_rewarded": inc_j.per_person / max(inc_j.share_rewarded, 1e-9),
                                "tempted": ob_opt.share_tempted}}

    # --- figure: data and every meal
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    H = arrival_heatmap(scans, "Jaiswal", "dinner").iloc[:, :12]
    SITE["heatmap"] = {"days": list(H.index), "times": list(H.columns), "values": H.values.round(1).tolist()}
    im = axes[0].imshow(H.values, aspect="auto", cmap=seq_cmap(), vmin=0)
    axes[0].set_xticks(range(0, 12, 2), H.columns[::2])
    axes[0].set_yticks(range(len(H.index)), H.index)
    axes[0].grid(False)
    axes[0].set_title("Jaiswal dinner: QR scans per 10 minutes (8 weeks)")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="scans")
    ax = axes[1]
    labels = [f"{r['meal']}\n{r['day']}" for r in agg.to_dict("records")]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, agg.nash, 0.38, color=viz.NE, label="selfish (Nash)")
    ax.bar(x + 0.2, agg.opt, 0.38, color=viz.OPT, label="planner")
    for xi, v in zip(x, agg.nash):
        ax.text(xi - 0.2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8, color=viz.TEXT_2)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set(title="Cost per student per meal (average of 4 messes)", ylabel="minutes")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig6_mess_data")

    # --- figure: the dinner day by day (reactive crowd)
    show = {"no information": pols["no information"], "public display": pols["public display"],
            "CE advice": pols["CE advice"]}
    cap = g.congestion.capacity
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 4.6), gridspec_kw={"height_ratios": [2.2, 1]})
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
    for j, (name, pol) in enumerate(show.items()):
        res = run(inst, pol, "reactive", 0, days=90, burn=50)
        A = res.loads[50:90] / cap
        im = axes[0, j].imshow(A, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        axes[0, j].set_xticks(range(0, 12, 3), g.slot_labels[::3])
        axes[0, j].set_yticks([0, 10, 20, 30, 39], [51, 61, 71, 81, 90])
        axes[0, j].grid(False)
        axes[0, j].set_title(f"{name}: arrivals / capacity", fontsize=10.5)
        axes[1, j].plot(np.arange(51, 91), np.nanmean(res.cost[50:90], axis=1), color=POLICY_COLORS[name], lw=1.8)
        axes[1, j].set(ylim=(0, 16), xlabel="dinner")
        axes[1, j].text(52, 13.5, f"swing {res.metrics['swing']:.0%}", fontsize=9, color=viz.TEXT_2)
    axes[0, 0].set_ylabel("dinner")
    axes[1, 0].set_ylabel("cost (min)")
    fig.colorbar(im, ax=axes[0, :], shrink=0.9, label="arrivals / capacity (red = overload)")
    save(fig, "fig7_mess_dinners")

    # --- figure: how to show information at the mess
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    sdf = pd.DataFrame(sweeps)
    ax = axes[0]
    sub = sdf[sdf.sweep == "share"]
    ax.errorbar(sub.value, sub.cost, yerr=sub.ci, color=viz.CE, marker="o", ms=5, capsize=2, label="share who see the display")
    ax.axhline(nash, color=viz.NE, ls="--", lw=1.2)
    ax.text(0.01, nash * 1.08, "Nash", color=viz.NE, fontsize=9)
    ax.set(title="Reactive crowd: who should see the public display?", xlabel="share of students who see it",
           ylabel="minutes per student")
    ax = axes[1]
    im = ax.imshow(swing, origin="lower", aspect="auto", cmap=seq_cmap())
    ax.set_xticks(range(len(delays)), delays)
    ax.set_yticks(range(len(revise)), revise)
    ax.grid(False)
    for i in range(len(revise)):
        for j in range(len(delays)):
            ax.text(j, i, f"{swing[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if swing[i, j] > swing.max() * 0.55 else viz.TEXT)
    ax.set(xlabel="display delay (dinners)", ylabel="share reacting each day", title="Crowd swing: share switching slot daily")
    save(fig, "fig8_mess_display")


# =============================================================== E8 small crowds, solved exactly
def e8_small_crowds() -> None:
    plt = viz.plt
    rows = []
    for n in range(2, 61):
        c = SmallCrowd(n)
        soc = c.social()
        ne = c.pure_nash()
        ce, p = c.best_ce()
        opt, _ = c.optimum()
        mixed, q = c.mixed_nash(grid=120)
        rows.append({"n": n, "pure_nash": int(len(ne)), "best_pure_nash": float(soc[ne].min()) if len(ne) else None,
                     "best_ce": ce, "optimum": opt, "mixed_nash": mixed,
                     "support": [(c.M[j].tolist(), float(p[j])) for j in np.argsort(-p)[:3] if p[j] > 1e-4]})
    privacy = []
    for f in (0.7, 0.5):
        for n in (4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 120):
            r = small_bad_day(n, bad_factor=f)
            privacy.append({"n": n, "capacity": f, "tell_nobody": r.tell_nobody, "announce": r.announce,
                            "best_hint": r.best_hint, "private": r.private, "privacy_gain": r.privacy_gain})
        r = large_crowd_bad_day(f, grid=120)
        privacy.append({"n": "large", "capacity": f, "tell_nobody": r.tell_nobody, "announce": r.announce,
                        "best_hint": r.best_hint, "private": r.private, "privacy_gain": r.privacy_gain})
    c10 = SmallCrowd(10)  # why the best CE beats Nash at 10 people (slots: early, peak, late)
    _, p10 = c10.best_ce()
    top = np.argsort(-p10)[:2]
    late, peak_slot = 2, c10.pref
    mech = {"advice": [c10.M[j].tolist() for j in top], "prob": [float(p10[j]) for j in top],
            "late_to_peak_gain_min": [float(c10.cost[j, late] - c10.dev[j, late, peak_slot]) for j in top],
            "nash": c10.M[c10.pure_nash()[0]].tolist()}
    SUMMARY["e8"] = {"crowds": rows, "privacy": privacy, "shapes": crowd_shapes(), "mechanism": mech}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    df = pd.DataFrame(rows)
    ax = axes[0]
    ax.plot(df.n, df.mixed_nash, color=viz.MUTED, lw=1.6, label="mixed Nash (no coordination)")
    ax.plot(df.n, df.best_pure_nash.astype(float), color=viz.NE, lw=1.6, marker="o", ms=3,
            label="best pure Nash (gaps: none exists)")
    ax.plot(df.n, df.best_ce, color=viz.CE, lw=2, marker="o", ms=3, label="best correlated equilibrium")
    ax.plot(df.n, df.optimum, color=viz.OPT, lw=1.6, ls="--", label="planner")
    ax.set(title="Identical people, 3 slots: every number exact", xlabel="crowd size n (same load at every n)",
           ylabel="minutes per person", ylim=(0.9, 3.6))
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax = axes[1]
    pdf = pd.DataFrame([r for r in privacy if r["n"] != "large"])
    for f, col in ((0.7, viz.CE), (0.5, viz.SERIES[4])):
        sub = pdf[pdf.capacity == f]
        ax.plot(sub.n, 100 * sub.privacy_gain, color=col, marker="o", ms=4, label=f"bad day at {f:.0%} capacity")
        lim = [r for r in privacy if r["n"] == "large" and r["capacity"] == f][0]
        ax.axhline(100 * lim["privacy_gain"], color=col, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([4, 10, 30, 120], ["4", "10", "30", "120"])
    ax.set(title="Bad days: what private advice saves over the best public hint", xlabel="crowd size n (log)",
           ylabel="saving (%)")
    ax.text(4.2, 100 * [r for r in privacy if r["n"] == "large" and r["capacity"] == 0.5][0]["privacy_gain"] + 0.4,
            "dashed: same game, large-crowd limit", fontsize=8, color=viz.TEXT_2)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig9_small_crowds")


# =============================================================== paired tests (reads PER_SEED)
def paired_tests() -> None:
    """Two-sided sign-flip permutation tests on paired per-seed differences, Holm-corrected within each family."""
    rng = np.random.default_rng(11)

    def signflip(d, draws=200_000):
        d = np.asarray(d, dtype=float)
        if np.allclose(d, 0):
            return 1.0
        signs = rng.choice([-1.0, 1.0], size=(draws, len(d)))
        return float((np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12).mean() + 1 / draws)

    families = {
        "public display worse than no information (reactive)": ("reactive", "public display", "no information"),
        "display for 30% better than public display (reactive)": ("reactive", "display for 30%", "public display"),
        "CE advice better than public display (reactive)": ("reactive", "CE advice", "public display"),
        "reward margin 2 better than margin 0 (noisy users)": ("noisy 3", "CE advice + reward, margin 2",
                                                              "CE advice + reward, margin 0"),
    }
    rows = []
    for fam, (regime, a, b) in families.items():
        res = []
        for key in DOMAIN_ORDER:
            d = np.array(PER_SEED[(key, regime, a)]) - np.array(PER_SEED[(key, regime, b)])
            res.append((key, float(d.mean()), signflip(d), len(d)))
        order = np.argsort([r[2] for r in res])
        adj, running = np.empty(len(res)), 0.0
        for rank, i in enumerate(order):  # Holm: step-down, monotone
            running = max(running, min(1.0, (len(res) - rank) * res[i][2]))
            adj[i] = running
        for (key, diff, pval, n), pa in zip(res, adj):
            rows.append({"family": fam, "domain": key, "mean_difference": diff, "p": pval, "p_holm": float(pa),
                         "seeds": n})
    SUMMARY["tests"] = rows


# =============================================================== verification checks (for the report's table)
def checks() -> None:
    rng = np.random.default_rng(7)
    mono = []
    for key in DOMAIN_ORDER:
        g = domain(key)[0].game
        cm, n, T = g.congestion, g.n, g.T
        worst = np.inf
        for _ in range(20000):
            A = rng.dirichlet(np.ones(T) * rng.uniform(0.2, 3)) * n
            B = rng.dirichlet(np.ones(T) * rng.uniform(0.2, 3)) * n
            worst = min(worst, float((A - B) @ (cm.wait(A) - cm.wait(B))))
        mono.append(worst)
    full = [abs(SmallCrowd(n).best_ce()[0] - full_profile_ce(SmallCrowd(n))) for n in (3, 4, 5, 6)]
    # the random queue with evenly spaced arrivals must reproduce the fluid queue (to one service time)
    g = domain("mess")[0].game
    b = domain("mess")[1]
    A = g.loads(b.nash.X)
    even_gap = []
    mu, dt = g.congestion.service_rate, g.congestion.slot_minutes
    arr = np.concatenate([t * dt + (np.arange(a) + 0.5) * dt / a for t, a in enumerate(A) if a > 0])
    lab = np.concatenate([np.full(a, t) for t, a in enumerate(A) if a > 0])
    free, waits = 0.0, np.empty(len(arr))
    for i, a in enumerate(arr):
        start = max(a, free)
        waits[i] = start - a
        free = start + 1 / mu
    sim = np.array([waits[lab == t].mean() if (lab == t).any() else 0.0 for t in range(g.T)])
    even_gap = float(np.abs(sim - g.congestion.wait(A.astype(float)))[A > 0].max())
    SUMMARY["checks"] = {"monotone_min": float(min(mono)), "monotone_samples": 20000 * len(DOMAIN_ORDER),
                         "exact_vs_full_lp": float(max(full)), "even_arrivals_vs_fluid_min": even_gap,
                         "service_time_min": 1 / mu}


# =============================================================== website data
REPO = "https://github.com/YashG1836/crowd-congestion"
APP = "https://crowd-congestion.streamlit.app/"
SITE_URL = "https://yashg1836.github.io/crowd-congestion/"


def export_site() -> None:
    """Write docs/data.js: every number and series the website shows, produced by this code."""
    playback = {}
    for key in DOMAIN_ORDER:
        setup = domain(key)
        inst = setup[0]
        g = inst.game
        playback[key] = {"labels": g.slot_labels, "capacity": g.congestion.capacity, "unit": DOMAINS[key].unit,
                         "policies": {}}
        for name in ("no information", "public display", "display for 30%", "CE advice", "CE advice + reward"):
            res = run(inst, core_policies(setup)[name], "reactive", 0, days=120, burn=40)
            playback[key]["policies"][name] = {"loads": res.loads.astype(int).tolist(),
                                               "cost": np.nanmean(res.cost, axis=1).round(3).tolist(),
                                               "swing": res.metrics["swing"], "mean": res.metrics["mean_cost"]}
    jsetup = jaiswal("normal")
    inst, b, ob_ne, ob_opt, *_ = jsetup
    g = inst.game
    playback["jaiswal"] = {"labels": g.slot_labels, "capacity": g.congestion.capacity, "unit": "students",
                           "policies": {}}
    for name in ("no information", "public display", "display for 30%", "CE advice", "CE advice + reward"):
        res = run(inst, core_policies(jsetup)[name], "reactive", 0, days=120, burn=40)
        playback["jaiswal"]["policies"][name] = {"loads": res.loads.astype(int).tolist(),
                                                 "cost": np.nanmean(res.cost, axis=1).round(3).tolist(),
                                                 "swing": res.metrics["swing"], "mean": res.metrics["mean_cost"]}

    def phone(X, k, t, bonus=0.0):
        D = g.deviation_costs(X)[k, t].copy()  # this student's cost in every slot if everyone else obeys
        D[t] -= bonus
        return {"tau": int(g.tau[k]), "rec": int(t), "costs": D.round(3).tolist(), "beta": float(g.beta[k])}

    k_opt, t_opt = map(int, np.unravel_index(np.argmax(ob_opt.gain_by_cell), ob_opt.gain_by_cell.shape))
    k_ne = int(np.argmax(g.counts * (g.tau == 4)))
    t_ne = int(np.argmax(b.best_nash.X[k_ne]))
    phones = {"nash": phone(b.best_nash.X, k_ne, t_ne), "opt": phone(b.optimum.X, k_opt, t_opt),
              "reward": phone(b.optimum.X, k_opt, t_opt, float(ob_opt.gain_by_cell[k_opt, t_opt]))}
    mess_sched = {"labels": g.slot_labels, "capacity": g.congestion.capacity,
                  "nash": g.loads(b.nash.X).astype(int).tolist(), "opt": g.loads(b.optimum.X).astype(int).tolist(),
                  "wait_nash": g.congestion.wait(g.loads(b.nash.X).astype(float)).round(2).tolist(),
                  "wait_opt": g.congestion.wait(g.loads(b.optimum.X).astype(float)).round(2).tolist()}
    dom_sched = {}
    for key in DOMAIN_ORDER:
        di, db, *_ = domain(key)
        G = di.game
        dom_sched[key] = {"labels": G.slot_labels, "capacity": G.congestion.capacity,
                          "nash": G.loads(db.nash.X).astype(int).tolist(), "opt": G.loads(db.optimum.X).astype(int).tolist(),
                          "story": DOMAINS[key].story, "title": DOMAINS[key].title, "unit": DOMAINS[key].unit,
                          "bad_state": DOMAINS[key].bad_state}
    data = {"links": {"repo": REPO, "app": APP, "site": SITE_URL, "report": REPO + "/blob/main/report/main.pdf"},
            "summary": SUMMARY, "playback": playback, "phones": phones, "mess_schedule": mess_sched,
            "domains": dom_sched, "order": DOMAIN_ORDER, **SITE}
    (ROOT / "docs").mkdir(exist_ok=True)
    text = json.dumps(data, default=_jsonable, separators=(",", ":"))
    (ROOT / "docs" / "data.js").write_text("// generated by `python experiments.py` -- do not edit by hand\n"
                                           "window.DATA = " + text + ";\n")


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# =============================================================== the report's numbers and tables
# Every number quoted in report/main.tex is a macro defined in report/generated/numbers.tex, and the
# report's main tables are report/generated/tab_*.tex, both written here from SUMMARY. So
#     python experiments.py && (cd report && tectonic main.tex)
# reproduces the paper exactly from the code.
REPORT_GEN = ROOT / "report" / "generated"
LONG = {"road": "Road: bridge commute", "network": "Network: night backups", "restaurant": "Food court: lunch",
        "gym": "Gym: after work", "mess": "Mess dinner (generic)"}
NAME = {"road": "Road", "network": "Network", "restaurant": "Food court", "gym": "Gym", "mess": "Mess"}
BADDAY = {"road": "a lane closed", "network": "link degraded", "restaurant": "a counter closed",
          "gym": "machines down", "mess": "staff short"}


def _f(v, d: int = 1) -> str:
    return f"{float(v):.{d}f}"


def _range(vals, d: int = 1, d_hi: int | None = None) -> str:
    vals = [float(v) for v in vals]
    lo, hi = _f(min(vals), d), _f(max(vals), d if d_hi is None else d_hi)
    return lo if lo == hi else f"{lo}--{hi}"


def _pct(vals, d: int = 0) -> str:
    return _range([100 * float(v) for v in vals], d)


def _p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def _num(x: float) -> str:  # a cost ratio or minutes: 2 decimals below 10, else 1
    return f"{x:.2f}" if abs(x) < 10 else f"{x:.1f}"


def write_report_tex() -> None:
    s = SUMMARY
    REPORT_GEN.mkdir(parents=True, exist_ok=True)
    M: dict[str, str] = {}
    cfg = s["config"]
    M["Rseeds"], M["RnoisySeeds"], M["RqueueRuns"] = str(cfg["seeds"]), str(cfg["noisy_seeds"]), str(cfg["queue_runs"])
    M["RfaceSamples"], M["RfaceRounds"] = f"{cfg['face_samples']:,}", str(cfg["face_rounds"])
    # ---- E1 small games
    e1 = s["e1"]
    M["RdinCounter"] = _f(e1["dinner"]["shared counter"])
    M["RdinMixed"] = _f(e1["dinner"]["no information (mixed Nash)"])
    M["RdinCE"] = _f(e1["dinner"]["private coin (CE)"])
    M["RlightCE"] = _f(e1["intersection"]["best_ce_each"], 2)
    M["RlightNash"] = _f(e1["intersection"]["best_symmetric_nash"], 2)
    M["RrmGap"] = _f(e1["learning"]["regret_matching_ce_gap"], 4)
    # ---- E2 selfish timing and robustness
    dom = {r["domain"]: r for r in s["e2"]["domains"]}
    rob = {r["domain"]: r for r in s["e2"]["robust"]}
    M["RpoaRange"] = _range([dom[k]["poa"] for k in DOMAIN_ORDER])
    M["RroadNash"], M["RroadOpt"] = _f(dom["road"]["nash"]), _f(dom["road"]["opt"])
    M["RroadWait"] = _f(dom["road"]["nash_wait"])
    M["RmessNash"], M["RmessOpt"] = _f(dom["mess"]["nash"]), _f(dom["mess"]["opt"], 2)
    M["RepsMax"] = _f(max(r["eps_sec"] for r in dom.values()), 0)
    M["RpoaRandRange"] = _range([rob[k]["nash_random"] / rob[k]["opt_random"] for k in DOMAIN_ORDER])
    M["RpoaRandServRange"] = _range([rob[k]["nash_random_service"] / rob[k]["opt_random_service"] for k in DOMAIN_ORDER])
    M["RpoaBufRange"] = _range([rob[k]["nash_random"] / rob[k]["opt_tuned"] for k in DOMAIN_ORDER])
    M["RbufMax"] = _f(100 * max(1 - rob[k]["tuned_factor"] for k in DOMAIN_ORDER), 0)
    M["RbookOptRange"] = _range([rob[k]["booking_fluid"] / rob[k]["opt_fluid"] for k in DOMAIN_ORDER])
    M["RbookNashRange"] = _pct([rob[k]["booking_fluid"] / rob[k]["nash_fluid"] for k in DOMAIN_ORDER])
    M["RroadBook"] = _f(rob["road"]["booking_fluid"])
    # ---- E3 public display (and the paired tests)
    e3 = pd.DataFrame(s["e3"])
    sel = lambda reg, pol: e3[(e3.regime == reg) & (e3.policy == pol)]
    disp = sel("reactive", "public display")
    M["RdispRange"] = f"{disp.vs_nash.min():.1f}--{disp.vs_nash.max():.0f}"
    M["RdispMax"] = f"{disp.vs_nash.max():.0f}"
    M["RswingRange"] = _pct(disp.swing)
    M["RthirtyRange"] = _range(sel("reactive", "display for 30%").vs_nash, 2)
    M["RcalmDispMax"] = _f(sel("calm", "public display").vs_nash.max())
    noinfo = e3[e3.policy == "no information"]
    M["RnoInfoRange"] = _range(noinfo.vs_nash, 2)
    M["RnoInfoRegret"] = _f(noinfo.regret.max())
    M["RciPct"] = _f(100 * (e3.ci / e3.cost).max(), 0)
    M["RceRegret"] = _f(e3[e3.policy == "CE advice"].regret.max(), 2)
    M["RoptAdvRange"] = _range(e3[e3.policy == "optimum as advice"].vs_nash)
    tests = pd.DataFrame(s["tests"])
    dtests = tests[~tests.family.str.contains("margin")]
    M["RtestN"], M["RtestP"] = str(len(dtests)), _p(dtests.p_holm.max())
    M["RtestMarginP"] = _p(tests[tests.family.str.contains("margin")].p_holm.max())
    # ---- E4 correlated advice
    t4 = {r["domain"]: r for r in s["e4"]["table"]}
    comp = pd.DataFrame(s["e4"]["compliance"])
    enough = {k: comp[(comp.domain == k) & (comp.vs_nash <= 1.1)].share.min() for k in DOMAIN_ORDER}
    M["RcomplRange"], M["RcomplMess"] = _pct(enough.values()), _f(100 * enough["mess"], 0)
    M["RceGainRange"] = _range([t4[k]["ce_max_gain_sec"] for k in DOMAIN_ORDER], 0)
    M["RtemptRange"] = _pct([t4[k]["opt_tempted"] for k in DOMAIN_ORDER])
    gains = [t4[k]["opt_max_gain_min"] for k in DOMAIN_ORDER]
    M["RoptGainRange"] = f"{min(gains):.1f}--{max(gains):.0f}"
    M["RrewardRange"] = _range([t4[k]["reward_pp"] for k in DOMAIN_ORDER], 2)
    M["RsavingRange"] = _range([t4[k]["saving_pp"] for k in DOMAIN_ORDER])
    M["RbuyRange"] = _range([t4[k]["saving_pp"] / t4[k]["reward_pp"] for k in DOMAIN_ORDER])
    face = {r["domain"]: r for r in s["e4"]["face"]}
    tkeys = sorted([c for c in face["road"] if c.startswith("saving_sec_tol_")],
                   key=lambda c: -float(c[len("saving_sec_tol_"):-1]))
    M["RwardGap"] = _f(max(t4[k]["large_gap_sec"] for k in DOMAIN_ORDER), 2)
    M["RfaceDimRange"] = _range([face[k]["face_dim"] for k in DOMAIN_ORDER], 0)
    M["RfaceMax"] = _f(max(face[k][tkeys[0]] for k in DOMAIN_ORDER), 2)
    M["RfaceCheapRange"] = _pct([1 - face[k]["cheapest_on_face"] / face[k]["nash"] for k in DOMAIN_ORDER])
    noisy = {(r["domain"], float(r["lambda"]), r["policy"]): r["vs_nash"] for r in s["e4"]["noisy"]}
    nz = lambda lam, pol: [noisy[(k, lam, pol)] for k in DOMAIN_ORDER]
    M["RnoisyCE"] = _range(nz(3.0, "CE advice"), 2)
    M["RnoisyRzero"] = _range(nz(3.0, "CE advice + reward, margin 0"), 2)
    M["RnoisyRtwo"] = _range(nz(3.0, "CE advice + reward, margin 2"), 2)
    M["RnoisyLoneRtwo"] = _range(nz(1.0, "CE advice + reward, margin 2"), 2)
    M["RoptRel"] = _range([t4[k]["opt"] / t4[k]["nash"] for k in DOMAIN_ORDER], 2)
    # ---- E5 bad days
    e5 = s["e5"]
    ex = {k: e5[k]["expected"] for k in DOMAIN_ORDER}
    M["RannounceHelp"] = _pct([1 - ex[k]["announce the state"] / ex[k]["tell nobody"] for k in DOMAIN_ORDER
                               if ex[k]["announce the state"] < ex[k]["tell nobody"]])
    M["RroadUp"] = _f(100 * (ex["road"]["announce the state"] / ex["road"]["tell nobody"] - 1), 0)
    M["RroadSilence"], M["RroadAnnounce"] = _f(ex["road"]["tell nobody"]), _f(ex["road"]["announce the state"])
    ps = e5["road"]["per_state"]
    M["RroadNormalSilence"] = _f(ps["tell nobody"]["normal"])
    M["RroadNormalAnnounce"] = _f(ps["announce the state"]["normal"])
    h = e5["road"]["hint"]
    M["RhintBad"], M["RhintNormal"] = _f(100 * h["P(busy | bad)"], 0), _f(100 * h["P(busy | normal)"], 0)
    M["RbeliefLo"], M["RbeliefHi"] = _f(h["belief after quiet"], 2), _f(h["belief after busy"], 2)
    M["RbadGap"] = _f(max(e5[k]["max_gap_sec"] for k in DOMAIN_ORDER), 0)
    M["RmenuSize"] = str(e5["road"]["menu"])
    M["RprivDefault"] = _f(100 * max(1 - ex[k]["private advice"] / ex[k]["best public hint"] for k in DOMAIN_ORDER), 1)
    sw = pd.DataFrame(s["e5_sweep"])
    M["RprivFacMax"] = _f(100 * (1 - sw["private advice"] / sw["best public hint"]).max(), 1)

    def cell(k, cap, p):
        r = sw[(sw.domain == k) & np.isclose(sw.capacity, cap) & np.isclose(sw.p_bad, p)].iloc[0]
        return r["announce the state"] / r["tell nobody"]

    mild = [cell(k, 0.85, 0.3) for k in DOMAIN_ORDER]
    severe = {k: cell(k, 0.5, 0.3) for k in DOMAIN_ORDER}
    back = [k for k in DOMAIN_ORDER if severe[k] > 1]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "all five"}
    mild_txt = ("on a mild bad day (85\\% capacity) announcing helps in every facility" if max(mild) < 1 else
                f"on a mild bad day (85\\% capacity) announcing helps in {sum(m < 1 for m in mild)} of five facilities")
    sev_txt = (f"on a severe one (50\\%) it backfires in {words.get(len(back), len(back))} of five ("
               + ", ".join(f"{NAME[k].lower()} {100 * (severe[k] - 1):+.0f}\\%" for k in back) + ")") if back else \
        "on a severe one (50\\%) it never backfires"
    road_p = [(p, cell("road", 0.7, p)) for p in (0.1, 0.3, 0.5)]
    every = all(v > 1 for _, v in road_p)
    road_txt = (("on the road it backfires at every probability we tried (" if every else "on the road (")
                + ", ".join(f"probability {p:g}: {100 * (v - 1):+.0f}\\%" for p, v in road_p) + ")")
    M["RsweepText"] = f"{mild_txt}; {sev_txt}; and {road_txt}"
    hint_ok = bool(((sw["private advice"] <= sw["best public hint"] + 1e-6) &
                    (sw["best public hint"] <= np.minimum(sw["tell nobody"], sw["announce the state"]) + 1e-6)).all())
    M["RbadSettings"], M["RbadOrderOK"] = str(len(sw)), ("holds" if hint_ok else "FAILS")
    m50 = sw[(sw.domain == "mess") & np.isclose(sw.capacity, 0.5) & np.isclose(sw.p_bad, 0.3)].iloc[0]
    M["RmessSevSilence"], M["RmessSevAnnounce"] = _f(m50["tell nobody"]), _f(m50["announce the state"])
    M["RmessSevHint"] = _f(m50["best public hint"])
    # ---- E6 learning
    e6 = pd.DataFrame(s["e6"])
    rm = e6[e6.learner.str.startswith("Regret")]
    M["RrmRange"], M["RrmRegretMax"] = _range(rm.vs_nash, 2), _f(rm.swap_regret.max(), 2)
    M["RexpRegret"] = _range(e6[e6.learner.str.startswith("EXP3")].swap_regret)
    # ---- E7 the mess
    e7 = s["e7"]
    M["Rscans"], M["RhoursLost"] = f"{e7['rows']:,}", _f(e7["hours_lost_per_day"], 0)
    cal = [r for r in e7["calibration"] if r["queue_forms"]]
    M["RmuErr"] = _range([100 * abs(r["estimated_mu"] / r["true_mu"] - 1) for r in cal], 0)
    pref = e7["preferences"]
    M["RprefExact"] = _pct([pref[m]["exact"] for m in MEALS])
    M["RprefWithin"] = _pct([pref[m]["within_1_slot"] for m in MEALS])
    rel = [r[k] / r["mean_scans_per_slot"] for r in e7["nowcast"] for k in ("N0 historical mean", "N1 ratio scaling")]
    M["RforecastPct"] = _pct(rel)
    bm = {(r["meal"], r["day"]): r for r in e7["by_meal"]}
    M["RdinnerLost"], M["RspecialLost"] = _f(bm[("dinner", "normal")]["lost"]), _f(bm[("dinner", "special")]["lost"])
    jp = {(r["regime"], r["policy"]): r for r in e7["jaiswal"]["policies"]}
    jc = lambda pol, reg="reactive": _f(jp[(reg, pol)]["cost"], 2)
    M["RjNash"], M["RjOpt"] = _f(e7["jaiswal"]["nash"], 2), _f(e7["jaiswal"]["opt"], 2)
    M["RjDisp"], M["RjOld"], M["RjLevels"] = jc("public display"), jc("display 3 dinners old"), jc("display in 3 levels")
    M["RjNoise"], M["RjForecast"] = jc("display + private noise"), jc("moving-average forecast")
    M["RjCE"], M["RjReward"] = jc("CE advice"), jc("CE advice + reward")
    M["RjSwing"] = _f(100 * jp[("reactive", "public display")]["swing"], 0)
    half = [r["cost"] for r in e7["sweeps"] if r["sweep"] == "share" and r["value"] <= 0.5]
    M["RjHalf"] = _range(half)
    swv = e7["swing"]
    i3, i1, j4 = swv["revise"].index(0.3), swv["revise"].index(1.0), swv["delays"].index(4)
    M["RjSwingLo"], M["RjSwingHi"] = _f(100 * swv["values"][i3][0], 0), _f(100 * swv["values"][i3][j4], 0)
    M["RswingAllLo"], M["RswingAllHi"] = _f(100 * swv["values"][i1][0], 0), _f(100 * swv["values"][i1][-1], 0)
    wgr = {r["group"]: r for r in e7["who_gains"]}
    M["RjFlexible"] = _f(wgr["flexible students"]["with_reward"], 2)
    M["RjWorse"], M["RjMaxLoss"] = str(e7["worse_off"]["optimum"]), _f(e7["max_loss_sec"], 0)
    M["RjWorseReward"], M["RjBudget"] = str(e7["worse_off"]["with_reward"]), _f(e7["budget_pp"], 2)
    M["RjRewardShare"], M["RjRewardEach"] = _f(100 * e7["reward"]["share"], 0), _f(e7["reward"]["per_rewarded"])
    inf = e7["information"]
    M["RjInfoSilence"], M["RjInfoAnnounce"] = _f(inf["expected"]["tell nobody"], 2), _f(inf["expected"]["announce the state"], 2)
    M["RjNormalSilence"] = _f(inf["per_state"]["tell nobody"]["normal"], 2)
    M["RjNormalAnnounce"] = _f(inf["per_state"]["announce the state"]["normal"], 2)
    # ---- E8 small crowds, solved exactly
    e8 = s["e8"]
    cr = {int(r["n"]): r for r in e8["crowds"]}
    has = [r for r in e8["crowds"] if r["best_pure_nash"] is not None]
    M["RceBeats"] = str(sum(r["best_ce"] < r["best_pure_nash"] - 1e-9 for r in has))
    M["RceSizes"], M["RnMax"] = str(len(has)), str(max(cr))
    t10, t60 = cr[10], cr[max(cr)]
    M["RtenCE"], M["RtenNash"], M["RtenOpt"] = _f(t10["best_ce"], 2), _f(t10["best_pure_nash"], 2), _f(t10["optimum"], 2)
    gap = lambda r: (r["best_pure_nash"] - r["best_ce"]) / (r["best_pure_nash"] - r["optimum"])
    M["RgapTen"], M["RgapSixty"] = _f(100 * gap(t10), 0), _f(100 * gap(t60), 0)
    none = [str(int(r["n"])) for r in e8["crowds"] if r["best_pure_nash"] is None]
    M["RnoPure"] = (", ".join(none[:-1]) + " and " + none[-1]) if len(none) > 1 else (none[0] if none else "no")
    M["RmixTen"], M["RmixSixtyRatio"] = _f(t10["mixed_nash"], 2), _f(t60["mixed_nash"] / t60["best_ce"], 2)
    mech = e8["mechanism"]
    M["RmechA"], M["RmechB"] = "/".join(map(str, mech["advice"][0])), "/".join(map(str, mech["advice"][1]))
    M["RmechAp"], M["RmechBp"] = _f(100 * mech["prob"][0], 0), _f(100 * mech["prob"][1], 0)
    M["RmechNash"] = "/".join(map(str, mech["nash"]))
    M["RmechGain"], M["RmechLoss"] = _f(60 * mech["late_to_peak_gain_min"][0], 0), _f(-mech["late_to_peak_gain_min"][1], 1)
    shp = pd.DataFrame(e8["shapes"])
    beat, worse, mix = 0, 0, []
    for _, sub in shp.groupby("shape", sort=False):
        hh = sub.dropna(subset=["pure_nash"])
        beat += int((hh.best_ce < hh.pure_nash - 1e-9).any())
        worse += int((hh.best_ce > hh.pure_nash + 1e-9).sum())
        vm = sub.dropna(subset=["mixed"])
        mix += list(1 - vm.best_ce / vm.mixed)
    M["RshapesN"], M["RshapesBeat"], M["RshapesWorse"] = str(shp["shape"].nunique()), str(beat), str(worse)
    M["RshapesMix"] = _pct(mix)
    pr = e8["privacy"]
    small = lambda cap: [r["privacy_gain"] for r in pr if r["n"] != "large" and np.isclose(r["capacity"], cap)]
    lim = {round(float(r["capacity"]), 2): r["privacy_gain"] for r in pr if r["n"] == "large"}
    M["RprivSmallSeventy"], M["RprivSmallFifty"] = _pct(small(0.7)), _pct(small(0.5))
    M["RlimSeventy"], M["RlimFifty"] = _f(100 * lim[0.7]), _f(100 * lim[0.5])
    # ---- verification checks
    ch = s["checks"]
    M["RmonoN"], M["RmonoMin"] = f"{ch['monotone_samples']:,}", f"{ch['monotone_min']:.1g}"
    M["RevenGap"], M["RexactDiff"] = _f(ch["even_arrivals_vs_fluid_min"], 3), f"{ch['exact_vs_full_lp']:.1g}"

    out = ["% generated by `python experiments.py` -- do not edit by hand"]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    (REPORT_GEN / "numbers.tex").write_text("\n".join(out) + "\n")

    def table(name, spec, header, rows, extra=""):
        body = "\n".join(rows)
        (REPORT_GEN / f"{name}.tex").write_text(
            f"% generated by `python experiments.py`\n\\begin{{tabular}}{{{spec}}}\\toprule\n{header}\\\\\\midrule\n"
            f"{body}\n\\bottomrule{extra}\n\\end{{tabular}}\n")

    table("tab_domains", "@{}lrrrrrrrl@{}",
          "Facility & People & Slot & $\\mu$/min & Selfish & Planner & PoA & PoA, random & Bad day",
          [f"{LONG[k]} & {dom[k]['people']} {dom[k]['unit']} & {dom[k]['slot_min']:g}\\,min & "
           f"{dom[k]['service_rate']:g} & {dom[k]['nash']:.2f} & {dom[k]['opt']:.2f} & {dom[k]['poa']:.1f} & "
           f"{rob[k]['nash_random'] / rob[k]['opt_tuned']:.1f} & {BADDAY[k]}\\\\" for k in DOMAIN_ORDER])
    tl = [f"{1000 * float(c[len('saving_sec_tol_'):-1]):g}\\,ms" for c in tkeys]
    ms = lambda v: f"{1000 * v:.2f}" if 1000 * v >= 0.1 else f"{1000 * v:.3f}"
    table("tab_ce", "@{}lrrrrrrrr@{}",
          " & \\multicolumn{2}{c}{real game} & \\multicolumn{3}{c}{large crowd: best-response face} & "
          "\\multicolumn{3}{c}{best-CE saving (ms) at tolerance}\\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(l){7-9}\n"
          "Facility & Nash & best CE & Nash & dimension & cheapest & " + " & ".join(tl),
          [f"{NAME[k]} & {t4[k]['nash']:.2f} & {t4[k]['best_ce_atomic']:.2f} & {face[k]['nash']:.2f} & "
           f"{face[k]['face_dim']} & {face[k]['cheapest_on_face']:.2f} & " + " & ".join(ms(face[k][c]) for c in tkeys)
           + "\\\\" for k in DOMAIN_ORDER])
    table("tab_advice", "@{}lrrrrrrrr@{}",
          " & \\multicolumn{4}{c}{price of obedience (rational users)} & "
          "\\multicolumn{4}{c}{noisy users ($\\lambda=3$), cost / Nash}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n"
          "Facility & tempted & reward & saving & worse off & CE advice & + reward & + reward, margin 2 & planner",
          [f"{NAME[k]} & {100 * t4[k]['opt_tempted']:.0f}\\% & {t4[k]['reward_pp']:.2f} & {t4[k]['saving_pp']:.2f} & "
           f"{t4[k]['worse_off_opt']} / {t4[k]['worse_off_reward']} & {noisy[(k, 3.0, 'CE advice')]:.2f} & "
           f"{noisy[(k, 3.0, 'CE advice + reward, margin 0')]:.2f} & "
           f"{noisy[(k, 3.0, 'CE advice + reward, margin 2')]:.2f} & {t4[k]['opt'] / t4[k]['nash']:.2f}\\\\"
           for k in DOMAIN_ORDER])
    rows = []
    for k in DOMAIN_ORDER:
        e = ex[k]
        ann = f"{e['announce the state']:.2f}"
        if e["announce the state"] > e["tell nobody"]:
            ann = f"\\textcolor{{cNE}}{{{ann}}}"
        hk = e5[k]["hint"]
        rows.append(f"{NAME[k]} & {e['tell nobody']:.2f} & {ann} & {e['best public hint']:.2f} & "
                    f"{e['private advice']:.2f} & {100 * hk['P(busy | bad)']:.0f}\\% / {100 * hk['P(busy | normal)']:.0f}\\% & "
                    f"{e['planner, informed']:.2f}\\\\")
    table("tab_bad", "@{}lrrrrrr@{}", "Facility & Tell nobody & Announce & Best public hint & Private advice & "
          "``Busy'' on bad / normal days & Planner", rows)
    calib = {(r["meal"], r["menu"]): r for r in e7["calibration"]}
    nowc = {(r["meal"], r["menu"]): r for r in e7["nowcast"]}
    est = lambda r: f"{r['estimated_mu']:.1f}" + ("$^{\\dagger}$" if not r["queue_forms"] else "")
    best = lambda r: min(r["N0 historical mean"], r["N1 ratio scaling"])
    table("tab_data", "@{}lrrrrr@{}",
          "Meal (normal / special) & true $\\mu$ & estimated $\\mu$ & favourite slot: exact / within 1 & "
          "scans per slot & forecast error",
          [f"{m.capitalize()} & {calib[(m, 'normal')]['true_mu']:.1f} / {calib[(m, 'special')]['true_mu']:.1f} & "
           f"{est(calib[(m, 'normal')])} / {est(calib[(m, 'special')])} & "
           f"{100 * pref[m]['exact']:.1f}\\% / {100 * pref[m]['within_1_slot']:.1f}\\% & "
           f"{nowc[(m, 'normal')]['mean_scans_per_slot']:.1f} / {nowc[(m, 'special')]['mean_scans_per_slot']:.1f} & "
           f"{best(nowc[(m, 'normal')]):.1f} / {best(nowc[(m, 'special')]):.1f}\\\\" for m in MEALS],
          "\n\\multicolumn{6}{@{}l}{\\footnotesize $^{\\dagger}$No queue forms, so the scans give only a lower bound.}")


# =============================================================== RESULTS.md
def md(df: pd.DataFrame, digits: int = 3) -> str:
    return df.round(digits).to_markdown(index=False)


def write_markdown() -> None:
    s = SUMMARY
    L = ["# Results (generated by `python experiments.py`)", "",
         "Costs are minutes per person per visit (waiting + inconvenience of going away from one's preferred time). "
         "Synthetic data. Intervals are 95% confidence half-widths over random seeds.", ""]
    e1 = s["e1"]
    L += ["## E1 Small games", "", md(pd.DataFrame([e1["dinner"]])), "",
          f"Traffic light: best correlated equilibrium gives each driver {e1['intersection']['best_ce_each']:.2f}; the best "
          f"symmetric Nash outcome gives {e1['intersection']['best_symmetric_nash']:.2f}. Learning: regret matching ends "
          f"{e1['learning']['regret_matching_ce_gap']:.4f} from the CE set, Hedge {e1['learning']['hedge_cce_gap']:.4f} "
          "from the coarse-CE set.", ""]
    L += ["## E2 Selfish timing vs the planner", "", md(pd.DataFrame(s["e2"]["domains"]).drop(columns=["title"])), "",
          "Random arrival times inside each slot (same schedules; 200 simulated runs), a planner re-tuned for "
          "randomness (optimum of a slower counter), and slot booking (random order, nearest free slot):", "",
          md(pd.DataFrame(s["e2"]["robust"])), ""]
    e3 = pd.DataFrame(s["e3"])
    piv = e3.pivot_table(index=["regime", "policy"], columns="domain", values="vs_nash").reindex(columns=DOMAIN_ORDER)
    L += ["## E3 Public display: cost relative to Nash", "", piv.round(2).to_markdown(), "",
          "Crowd swing (share of people switching slot each day), reactive crowd:", "",
          e3[e3.regime == "reactive"].pivot_table(index="policy", columns="domain", values="swing")
          .reindex(columns=DOMAIN_ORDER).round(3).to_markdown(), ""]
    L += ["## E4 Correlated advice at scale", "", md(pd.DataFrame(s["e4"]["table"])), "",
          "Compliance (reactive crowd, non-users see the display), cost relative to Nash:", "",
          pd.DataFrame(s["e4"]["compliance"]).pivot_table(index="share", columns="domain", values="vs_nash")
          .reindex(columns=DOMAIN_ORDER).round(2).to_markdown(), ""]
    L += ["Best-response face check (large crowd): best CE saving over Nash, seconds per person, at three obedience "
          "tolerances (3000 sampled face schedules + column generation):", "", md(pd.DataFrame(s["e4"]["face"]), 5), "",
          "Noisy app users (logit lambda), cost relative to Nash (calm crowd for non-users):", "",
          pd.DataFrame(s["e4"]["noisy"]).pivot_table(index=["lambda", "policy"], columns="domain", values="vs_nash")
          .reindex(columns=DOMAIN_ORDER).round(2).to_markdown(), ""]
    L += ["## E5 Who should be told what (bad day: capacity 70%, probability 0.3; large-crowd limit)", ""]
    rows = []
    for k, v in s["e5"].items():
        rows.append({"domain": k, **{p: round(c, 3) for p, c in v["expected"].items()},
                     "P(busy|bad)": round(v["hint"]["P(busy | bad)"], 2),
                     "P(busy|normal)": round(v["hint"]["P(busy | normal)"], 2),
                     "tolerance (s)": round(v["tol_sec"], 2)})
    L += [md(pd.DataFrame(rows)), "", "Per-state costs (road):", "",
          pd.DataFrame(s["e5"]["road"]["per_state"]).round(2).to_markdown(), ""]
    sw = pd.DataFrame(s["e5_sweep"])
    sw["announce/silence"] = sw["announce the state"] / sw["tell nobody"]
    sw["hint/silence"] = sw["best public hint"] / sw["tell nobody"]
    sw["privacy gain %"] = 100 * (1 - sw["private advice"] / sw["best public hint"])
    L += ["How bad and how often (11 belief points for the sweep):", "",
          md(sw[["domain", "capacity", "p_bad", "announce/silence", "hint/silence", "privacy gain %", "max_gap_sec"]]), ""]
    L += ["## E6 Learning", "", md(pd.DataFrame(s["e6"])), ""]
    e7 = s["e7"]
    L += ["## E7 The IITGN mess", "",
          f"{e7['rows']:,} QR scans for {e7['students']} students over 8 weeks; {e7['hours_lost_per_day']:.0f} "
          "student-hours lost to selfish timing on a normal day (4 messes, 3 meals).", "",
          md(pd.DataFrame(e7["calibration"])), "", pd.DataFrame(e7["preferences"]).T.round(3).to_markdown(), "",
          md(pd.DataFrame(e7["by_meal"])), "", "Jaiswal dinner, every display policy:", "",
          pd.DataFrame(e7["jaiswal"]["policies"]).pivot_table(index="policy", columns="regime", values="cost")
          .round(2).to_markdown(), "", md(pd.DataFrame(e7["sweeps"])), "",
          "Information design at Jaiswal dinner:", "", md(pd.DataFrame([e7["information"]["expected"]])), "",
          "Who gains:", "", md(pd.DataFrame(e7["who_gains"])), "",
          f"Worse off than Nash: optimum without reward {e7['worse_off']['optimum']} students (by at most "
          f"{e7['max_loss_sec']:.1f} s), with reward {e7['worse_off']['with_reward']}. Reward budget "
          f"{e7['budget_pp']:.2f} min per student.", "", "Nowcasting MAE (scans per slot):", "", md(pd.DataFrame(e7["nowcast"])), ""]
    e8 = s["e8"]
    cr = pd.DataFrame(e8["crowds"]).drop(columns=["support"])
    L += ["## E8 Small crowds solved exactly (identical people, 3 slots, load 0.8)", "", md(cr), "",
          "Support of the best CE at n = 10: " + str(next(r["support"] for r in e8["crowds"] if r["n"] == 10)), "",
          "Bad days in small crowds (best public hint followed by the best CE of its belief):", "",
          md(pd.DataFrame(e8["privacy"])), ""]
    sh = pd.DataFrame(e8["shapes"])
    L += ["Crowd shapes (exact):", "", md(sh), ""]
    L += ["## Paired tests (sign-flip permutation, Holm within each family)", "", md(pd.DataFrame(s["tests"]), 5), ""]
    L += ["## Verification checks", "", "```", json.dumps(s["checks"], indent=1), "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 seeds instead of 30")
    seeds = 5 if ap.parse_args().quick else 30
    SUMMARY["config"] = {"seeds": seeds, "noisy_seeds": max(3, seeds // 3), "queue_runs": QUEUE_RUNS,
                         "face_samples": FACE_SAMPLES, "face_rounds": FACE_ROUNDS}
    FIG.mkdir(parents=True, exist_ok=True)
    viz.style()
    steps = [("E1 small games", e1_small_games), ("E2 selfish timing", e2_selfish),
             ("E3 public display", lambda: e3_display(seeds)), ("E4 CE advice", lambda: e4_ce(seeds)),
             ("E5 uncertainty", e5_information), ("E6 learning", lambda: e6_learning(seeds)),
             ("E7 the mess", lambda: e7_mess(seeds)), ("E8 small crowds", e8_small_crowds), ("paired tests", paired_tests),
             ("checks", checks),
             ("website data", export_site)]
    for name, fn in steps:
        t0 = time.time()
        fn()
        print(f"{name:20s} done in {time.time() - t0:6.1f}s", flush=True)
    write_markdown()
    write_report_tex()
    print("wrote results/, docs/data.js and report/generated/")


if __name__ == "__main__":
    main()
