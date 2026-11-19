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
from crowd.correlated import (best_ce, information_design, interpolated_schedules, large_crowd_check, obedience,
                              obedience_incentive, who_gains)
from crowd.domains import DOMAINS, build
from crowd.game import benchmarks
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

REGIMES = {
    "calm": Logit(3.0, 0.3),       # 30% of people reconsider each day
    "reactive": Logit(3.0, 1.0),   # everyone reconsiders each day
    "noisy": Logit(1.0, 0.3),      # low rationality
}
DOMAIN_ORDER = ["road", "network", "restaurant", "gym", "mess"]
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
    SUMMARY["e2"] = {"domains": rows}
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.nash, 0.38, color=viz.NE, label="selfish (Nash)")
    ax.bar(x + 0.2, df.opt, 0.38, color=viz.OPT, label="planner (optimum)")
    for xi, r in zip(x, rows):
        ax.text(xi, r["nash"] * 1.12, f"{r['poa']:.1f}x", ha="center", fontsize=9, color=viz.TEXT_2)
    ax.set_xticks(x, [SHORT[k] for k in df.domain])
    plain_log(ax, "y", [0.3, 1, 3, 10, 30])
    ax.set(title="Minutes lost per person, and the price of anarchy", ylabel="minutes per person (log)", ylim=(0.2, 40))
    ax.legend(loc="upper right", fontsize=8.5)
    inst, b, *_ = domain("road")
    g = inst.game
    xs = np.arange(g.T)
    ax = axes[1]
    ax.bar(xs - 0.2, g.loads(b.nash.X), 0.38, color=viz.NE, label="selfish (Nash)")
    ax.bar(xs + 0.2, g.loads(b.optimum.X), 0.38, color=viz.OPT, label="planner (optimum)")
    ax.axhline(g.congestion.capacity, color=viz.MUTED, ls="--", lw=1.2)
    ax.text(g.T - 0.5, g.congestion.capacity + 6, "bridge capacity per slot", ha="right", color=viz.TEXT_2, fontsize=9)
    ax.set_xticks(xs[::2], g.slot_labels[::2])
    ax.set(title="Morning commute: cars arriving per 15 minutes", ylabel="cars")
    ax.legend(loc="upper left", fontsize=8.5)
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
            vals = [sub[(sub.domain == k) & (sub.policy == pol)].vs_nash.iloc[0] for k in DOMAIN_ORDER]
            ax.bar(np.arange(5) + (j - 2.5) * w, vals, w * 0.92, color=POLICY_COLORS[pol], label=pol)
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
    SUMMARY["e4"] = {"table": table, "compliance": comp}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.7))
    ax = axes[0]
    cdf = pd.DataFrame(comp)
    for key, col in zip(DOMAIN_ORDER, [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.SERIES[3], viz.SERIES[6]]):
        sub = cdf[cdf.domain == key]
        ax.plot(sub.share, sub.vs_nash, marker="o", ms=4.5, color=col, label=SHORT[key])
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
    SUMMARY["e5"] = out
    pols = ["tell nobody", "announce the state", "best public hint", "private advice"]
    cols = [viz.MUTED, viz.NE, viz.SERIES[3], viz.CE]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    w = 0.19
    for j, (pol, c) in enumerate(zip(pols, cols)):
        vals = [out[k]["expected"][pol] / out[k]["expected"]["tell nobody"] for k in DOMAIN_ORDER]
        ax.bar(np.arange(5) + (j - 1.5) * w, vals, w * 0.92, color=c, label=pol)
    ax.axhline(1, color=viz.TEXT_2, lw=1)
    ax.set_xticks(range(5), [SHORT[k] for k in DOMAIN_ORDER])
    ax.set(title="Expected cost relative to telling nobody", ylabel="relative cost", ylim=(0, 1.42))
    ax.legend(fontsize=8, ncol=2, loc="upper left")
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
    ax.set(title="Road: cost as a function of the crowd's belief", xlabel="belief q that today is a bad day",
           ylabel="expected minutes per car")
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
            errs = [nowcast_errors(scans_per_slot(scans, m, meal, menu)) for m in MESSES]
            now.append({"meal": meal, "menu": menu, **{k: float(np.mean([e[k] for e in errs])) for k in errs[0]}})
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
                     "max_loss_sec": 60 * wg.max_loss("optimum"), "budget_pp": wg.budget / g.n}

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
    L += ["## E2 Selfish timing vs the planner", "", md(pd.DataFrame(s["e2"]["domains"]).drop(columns=["title"])), ""]
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
    L += ["## E5 Who should be told what (bad day: capacity 70%, probability 0.3; large-crowd limit)", ""]
    rows = []
    for k, v in s["e5"].items():
        rows.append({"domain": k, **{p: round(c, 3) for p, c in v["expected"].items()},
                     "P(busy|bad)": round(v["hint"]["P(busy | bad)"], 2),
                     "P(busy|normal)": round(v["hint"]["P(busy | normal)"], 2),
                     "tolerance (s)": round(v["tol_sec"], 2)})
    L += [md(pd.DataFrame(rows)), "", "Per-state costs (road):", "",
          pd.DataFrame(s["e5"]["road"]["per_state"]).round(2).to_markdown(), ""]
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
    (OUT / "RESULTS.md").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 seeds instead of 30")
    seeds = 5 if ap.parse_args().quick else 30
    FIG.mkdir(parents=True, exist_ok=True)
    viz.style()
    steps = [("E1 small games", e1_small_games), ("E2 selfish timing", e2_selfish),
             ("E3 public display", lambda: e3_display(seeds)), ("E4 CE advice", lambda: e4_ce(seeds)),
             ("E5 uncertainty", e5_information), ("E6 learning", lambda: e6_learning(seeds)),
             ("E7 the mess", lambda: e7_mess(seeds)), ("website data", export_site)]
    for name, fn in steps:
        t0 = time.time()
        fn()
        print(f"{name:20s} done in {time.time() - t0:6.1f}s", flush=True)
    write_markdown()
    print(f"wrote {OUT} and docs/data.js")


if __name__ == "__main__":
    main()
