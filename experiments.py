"""Run every experiment; writes results/figures/*.png and results/RESULTS.md.

    python experiments.py            # full run (about 3 minutes)
    python experiments.py --quick    # 5 seeds instead of 30 (for a fast check)

Experiments (each answers one question from DESIGN.md):
    E0  theory      toy games: Pigou, dinner game, intersection, learning, delay stability
    E1  data        synthetic QR logs: arrival heatmap + calibration (can we recover mu, tau?)
    E2  poa         RQ1: how much does selfish timing cost? (24 mess instances + 5 domains)
    E3  policies    RQ2: which display policy works, under three behaviour regimes? (H1)
    E4  sweeps      RQ2: noise, share of viewers, coarseness (is partial information best?)
    E5  delay       RQ2: delay x responsiveness -> crowd swings (H2)
    E6  ce          RQ3: correlated recommendations, compliance, price of obedience (H3)
    E7  learning    Hedge vs EXP3 vs regret matching (does publishing data help learning?)
    E8  bayes       complete vs incomplete information (value of announcing a slow night)
    E9  nowcast     RQ4: predict the next 30 minutes of scans from tonight's scans so far
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crowd import viz
from crowd.correlated import bayesian_analysis, best_ce, interpolated_schedules, obedience, obedience_incentive
from crowd.domains import DOMAINS, build
from crowd.game import FluidQueue, benchmarks
from crowd.mess import (MEALS, MESSES, arrival_heatmap, estimate_preferred_slots, estimate_service_rate, generate,
                        make_roster, meal_instance, nowcast_errors, preference_accuracy, roster_truth,
                        scans_per_slot)
from crowd.simulate import EXP3, Coarse, Forecast, Hedge, Live, Logit, Noisy, NoInfo, Partial, Recommend, \
    RegretMatching, simulate
from crowd.theory import (critical_responsiveness, dinner_game, hedge, intersection_game, pigou_poa, regret_matching,
                          shared_counter_dynamics)
from crowd.theory import simulate as delay_sim

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = OUT / "figures"
SUMMARY: dict = {}

REGIMES = {
    "calm (30% reconsider daily)": Logit(3.0, 0.3),
    "reactive (everyone reconsiders daily)": Logit(3.0, 1.0),
    "noisy (low rationality)": Logit(1.0, 0.3),
}


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


def standard_instance():
    inst = meal_instance("Jaiswal", "dinner", "normal")
    return inst, benchmarks(inst.game)


def policy_list(inst, b, ob_opt, best_alt):
    """pi0-pi8. Recommendation users are rational: they disobey whenever it pays (> 6 s)."""
    common = dict(type_of=inst.type_of, rational=True, best_alt=best_alt)
    ob_ne = obedience(inst.game, b.best_nash.X)
    alt_ne = inst.game.deviation_costs(b.best_nash.X).argmin(axis=2)
    return [
        NoInfo(), Live(), Live(delay=3), Noisy(3.0), Coarse(3), Partial(Live(), 0.3), Forecast(0.3),
        Recommend(target=b.best_nash.X, type_of=inst.type_of, rational=True, gains=ob_ne.gain_by_cell,
                  best_alt=alt_ne, name="pi7 CE recommendation"),
        Recommend(target=b.optimum.X, gains=ob_opt.gain_by_cell, name="Optimum as advice, no reward", **common),
        Recommend(target=b.optimum.X, gains=ob_opt.gain_by_cell, bonus=ob_opt.gain_by_cell,
                  name="pi8 Optimum as advice + obedience reward", **common),
    ]


# =============================================================== E0 theory
def e0_theory() -> None:
    plt = viz.plt
    g = dinner_game()
    mixed = [m for m in g.mixed_nash_2p() if 0 < m[0][0] < 1][0]
    path = shared_counter_dynamics(g, (1, 1), 20)
    rows = [
        ("Shared live counter (both react to yesterday)", np.mean([g.total_cost(s) for s in path])),
        ("No information (symmetric mixed Nash)", g.expected_costs(g.product_distribution(list(mixed))).sum()),
        ("Private recommendation (correlated equilibrium)", g.correlated_equilibrium("best").total_cost),
    ]
    opt = g.social_optimum()[1]
    dinner = pd.DataFrame([(a, c, c / opt) for a, c in rows], columns=["policy", "total cost", "ratio to optimum"])

    ig = intersection_game()
    ce = ig.correlated_equilibrium("best")
    SUMMARY["e0"] = {
        "dinner": dinner.to_dict("records"),
        "intersection_best_ce_payoff": float(-ce.expected_costs[0]),
        "intersection_best_ce_support": ce.support(),
        "intersection_mixed_nash_payoff": 14 / 3,
        "pigou_poa": {p: pigou_poa(p) for p in (1, 2, 4, 8)},
    }

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    p = np.linspace(1, 12, 200)
    ax[0].plot(p, [pigou_poa(v) for v in p], color=viz.NE)
    ax[0].scatter([1], [4 / 3], color=viz.NE, s=40, zorder=3)
    ax[0].annotate("linear cost: 4/3", (1, 4 / 3), (2.2, 1.25), color=viz.TEXT_2, fontsize=9)
    ax[0].set(title="Pigou: steeper congestion, bigger waste", xlabel="steepness p of peak cost x^p",
              ylabel="price of anarchy")

    rm = regret_matching(ig, rounds=6000, seed=0, log_every=25)
    hd = hedge(ig, rounds=6000, seed=0, log_every=25)
    ax[1].plot(rm.rounds_logged, rm.ce_violation, color=viz.CE, label="regret matching (CE gap)")
    ax[1].plot(hd.rounds_logged, hd.ce_violation, color=viz.SERIES[3], label="Hedge (CE gap)")
    ax[1].set(title="Learning finds correlated equilibria", xlabel="rounds played", yscale="log",
              ylabel="largest gain from disobeying")
    ax[1].legend()

    d = np.arange(0, 7)
    ax[2].fill_between(d, 0, [critical_responsiveness(x) for x in d], color=viz.SEQ_BLUE[0], step=None)
    ax[2].plot(d, [critical_responsiveness(x) for x in d], color=viz.CE, marker="o", ms=5)
    ax[2].text(0.3, 0.25, "crowd settles", color=viz.TEXT_2)
    ax[2].text(3.2, 1.3, "crowd swings", color=viz.TEXT_2)
    ax[2].set(title="Old information tolerates less reaction", xlabel="display delay d",
              ylabel="max responsiveness k*(d)", ylim=(0, 2.1))
    fig.tight_layout()
    save(fig, "e0_theory")

    fig, ax = plt.subplots(figsize=(6, 3))
    for (k, dd), c in zip([(0.3, 0), (0.5, 2), (0.75, 2)], [viz.OPT, viz.CE, viz.NE]):
        ax.plot(delay_sim(k, dd, 25), color=c, label=f"k={k}, delay={dd}")
    ax.axhline(0, color=viz.MUTED, lw=0.8)
    ax.set(title="Delayed feedback: y(t+1) = y(t) - k y(t-d)", xlabel="day", ylabel="excess crowd")
    ax.legend()
    save(fig, "e0_delay_paths")


# =============================================================== E1 data
def e1_data() -> None:
    plt = viz.plt
    scans, truth = generate(weeks=8, seed=0)
    rt = roster_truth(0).set_index("student_id")
    rows = []
    for meal in MEALS:
        for menu in ("normal", "special"):
            true_mu = MEALS[meal].service_rate * (0.8 if menu == "special" else 1.0)
            est = estimate_service_rate(scans, "Jaiswal", meal, menu)
            queued = truth[(truth.meal == meal) & (truth.menu == menu) & (truth.mess == "Jaiswal")].wait_min
            rows.append({"meal": meal, "menu": menu, "true_mu": true_mu, "estimated_mu": round(est, 2),
                         "mean_wait_min": round(float(queued.mean()), 2),
                         "note": "queue forms: mu identifiable" if queued.max() > 1 else "no queue: estimate is a lower bound"})
    calib = pd.DataFrame(rows)
    acc = {meal: preference_accuracy(estimate_preferred_slots(scans, meal), rt[f"tau_{meal}"]) for meal in MEALS}
    SUMMARY["e1"] = {"rows": int(len(scans)), "students": int(scans.student_id.nunique()),
                     "service_rate": calib.to_dict("records"), "preferences": acc}

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
    for ax, meal in zip(axes, ("lunch", "dinner")):
        H = arrival_heatmap(scans, "Jaiswal", meal)
        H = H.iloc[:, :12]
        im = ax.imshow(H.values, aspect="auto", cmap=matplotlib_seq(), vmin=0)
        ax.set_xticks(range(0, 12, 2), H.columns[::2])
        ax.set_yticks(range(len(H.index)), H.index)
        ax.grid(False)
        ax.set_title(f"Jaiswal {meal}: QR scans per 10 min")
        fig.colorbar(im, ax=ax, shrink=0.8, label="scans")
    fig.tight_layout()
    save(fig, "e1_heatmap")


def matplotlib_seq():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + viz.SEQ_BLUE)


# =============================================================== E2 price of anarchy
def e2_poa() -> None:
    plt = viz.plt
    roster = make_roster(0)
    rows = []
    for mess in MESSES:
        for meal in MEALS:
            for day in ("normal", "special"):
                inst = meal_instance(mess, meal, day, roster=roster)
                g = inst.game
                b = benchmarks(g)
                ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
                rows.append({"mess": mess, "meal": meal, "day": day, "diners": g.n,
                             "nash_cost": ne["per_person"], "nash_wait": ne["wait_per_person"],
                             "opt_cost": op["per_person"], "opt_wait": op["wait_per_person"],
                             "minutes_lost": ne["per_person"] - op["per_person"],
                             # a ratio of two tiny numbers means nothing: report PoA only if Nash costs >= 0.1 min
                             "poa": b.poa if ne["per_person"] >= 0.1 else np.nan, "epsilon_sec": 60 * b.epsilon})
    df = pd.DataFrame(rows)

    dom = []
    for key, d in DOMAINS.items():
        inst = build(d)
        b = benchmarks(inst.game)
        g = inst.game
        ob = obedience(g, b.optimum.X)
        inc = obedience_incentive(g, b.optimum.X, b.nash.social_cost)
        dom.append({"domain": d.title, "players": f"{g.n} {d.unit}", "nash_cost": b.nash.social_cost / g.n,
                    "opt_cost": b.optimum.social_cost / g.n, "poa": b.poa,
                    "tempted_by_optimum": ob.share_tempted, "reward_per_person": inc.per_person,
                    "saving_per_person": inc.welfare_gain / g.n})
    dd = pd.DataFrame(dom)
    agg = df.groupby(["meal", "day"])[["nash_cost", "opt_cost", "minutes_lost", "poa"]].mean().round(2)
    SUMMARY["e2"] = {"by_meal_day": agg.reset_index().to_dict("records"), "domains": dd.round(3).to_dict("records"),
                     "max_epsilon_sec": float(df.epsilon_sec.max()),
                     "person_hours_lost_per_day_all_messes": float(
                         (df.minutes_lost * df.diners)[df.day == "normal"].sum() / 60)}

    inst, b = standard_instance()
    sp = meal_instance("Jaiswal", "dinner", "special")
    bs = benchmarks(sp.game)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, (I, B, title) in zip(axes, [(inst, b, "normal night"), (sp, bs, "special-menu night")]):
        g = I.game
        x = np.arange(g.T)
        w = 0.4
        ax.bar(x - w / 2, g.loads(B.nash.X), w - 0.04, color=viz.NE, label="selfish (Nash)")
        ax.bar(x + w / 2, g.loads(B.optimum.X), w - 0.04, color=viz.OPT, label="planner (optimum)")
        ax.axhline(g.congestion.capacity, color=viz.MUTED, ls="--", lw=1.2)
        ax.text(g.T - 0.5, g.congestion.capacity + 2, "counter capacity per slot", ha="right", color=viz.TEXT_2,
                fontsize=9)
        ax.set_xticks(x[::2], g.slot_labels[::2])
        ax.set(title=f"Jaiswal dinner, {title}: arrivals per slot", ylabel="students arriving")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.92))
    fig.tight_layout()
    save(fig, "e2_nash_vs_optimum")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [f"{m}\n{d}" for m, d in agg.index]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, agg.nash_cost, 0.36, color=viz.NE, label="selfish (Nash)")
    ax.bar(x + 0.2, agg.opt_cost, 0.36, color=viz.OPT, label="planner (optimum)")
    for xi, v in zip(x, agg.nash_cost):
        ax.text(xi - 0.2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8, color=viz.TEXT_2)
    ax.set_xticks(x, labels)
    ax.set(title="Cost per student per meal (average of 4 messes)", ylabel="minutes (wait + inconvenience)")
    ax.legend()
    save(fig, "e2_cost_by_meal")


# =============================================================== E3 policies
def e3_policies(seeds: int) -> None:
    plt = viz.plt
    inst, b = standard_instance()
    g = inst.game
    ob = obedience(g, b.optimum.X)
    alt = g.deviation_costs(b.optimum.X).argmin(axis=2)
    cm, tau, beta = g.congestion, inst.tau, inst.beta
    rows = []
    for regime, beh in REGIMES.items():
        for pol in policy_list(inst, b, ob, alt):
            ms = [simulate(cm, tau, beta, pol, beh, days=150, burn_in=50, seed=s).metrics for s in range(seeds)]
            row = {"regime": regime, "policy": pol.name}
            for key in ("mean_cost", "mean_wait", "p90_wait", "swing", "swap_regret", "fairness"):
                row[key], row[key + "_ci"] = ci([m[key] for m in ms])
            rows.append(row)
    df = pd.DataFrame(rows)
    ne, opt = b.best_nash.social_cost / g.n, b.optimum.social_cost / g.n
    worst = df.groupby("policy", sort=False)["mean_cost"].max()
    SUMMARY["e3"] = {"nash": ne, "opt": opt, "table": df.round(3).to_dict("records"),
                     "worst_case_by_policy": worst.round(3).to_dict()}

    policies = list(dict.fromkeys(df.policy))
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(policies))
    for j, (regime, c, mk) in enumerate(zip(REGIMES, [viz.SERIES[0], viz.SERIES[3], viz.SERIES[6]], "osD")):
        sub = df[df.regime == regime].set_index("policy").loc[policies]
        ax.errorbar(sub.mean_cost, y + (j - 1) * 0.22, xerr=sub.mean_cost_ci, fmt=mk, color=c, ms=7,
                    capsize=2, lw=1, label=regime)
    ax.axvline(ne, color=viz.NE, ls="--", lw=1.4)
    ax.axvline(opt, color=viz.OPT, ls="--", lw=1.4)
    ax.text(ne, len(policies) - 0.35, " Nash", color=viz.NE, fontsize=9)
    ax.text(opt, len(policies) - 0.35, " optimum", color=viz.OPT, fontsize=9)
    ax.set_yticks(y, policies)
    ax.invert_yaxis()
    plain_log(ax, "x", [0.3, 0.5, 1, 2, 5, 10, 15])
    ax.set(title="What the display shows vs. average cost per student (Jaiswal dinner, 30 seeds)",
           xlabel="minutes per student per meal (log scale)")
    ax.legend(loc="lower right")
    save(fig, "e3_policies")


# =============================================================== E4 sweeps
def e4_sweeps(seeds: int) -> None:
    plt = viz.plt
    inst, b = standard_instance()
    cm, tau, beta, g = inst.game.congestion, inst.tau, inst.beta, inst.game
    beh = REGIMES["reactive (everyone reconsiders daily)"]
    sweeps = {
        "noise sd (min)": [(s, Noisy(s)) for s in (0.0, 1, 2, 3, 5, 8, 12)],
        "share who see the live display": [(p, Partial(Live(), p)) for p in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)],
        "display levels": [(k, Coarse(k)) for k in (2, 3, 5, 10, 30)],
    }
    out, rows = {}, []
    for name, items in sweeps.items():
        for val, pol in items:
            m, h = ci([simulate(cm, tau, beta, pol, beh, days=150, burn_in=50, seed=s).metrics["mean_cost"]
                       for s in range(seeds)])
            rows.append({"sweep": name, "value": val, "mean_cost": m, "ci": h})
    df = pd.DataFrame(rows)
    for name in sweeps:
        sub = df[df.sweep == name]
        out[name] = sub[["value", "mean_cost"]].round(3).to_dict("records")
    SUMMARY["e4"] = out

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    ne = b.best_nash.social_cost / g.n
    for ax, name in zip(axes, sweeps):
        sub = df[df.sweep == name]
        ax.errorbar(sub.value, sub.mean_cost, yerr=sub.ci, color=viz.CE, marker="o", ms=6, capsize=2)
        ax.axhline(ne, color=viz.NE, ls="--", lw=1.2)
        ax.set(xlabel=name, title=f"varying the {name.split(' (')[0]}")
        if name == "display levels":
            plain_log(ax, "x", [2, 3, 5, 10, 30])
    axes[0].set_ylabel("minutes per student")
    axes[0].text(0, ne * 1.08, "Nash", color=viz.NE, fontsize=9)
    fig.suptitle("Reactive crowd: less (or less shared) information is better", y=1.03, fontweight="bold")
    fig.tight_layout()
    save(fig, "e4_sweeps")


# =============================================================== E5 delay x responsiveness
def e5_delay(seeds: int) -> None:
    plt = viz.plt
    inst, _ = standard_instance()
    cm, tau, beta = inst.game.congestion, inst.tau, inst.beta
    revise = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    delays = [1, 2, 3, 4, 5]
    S = np.zeros((len(revise), len(delays)))
    for i, r in enumerate(revise):
        for j, d in enumerate(delays):
            S[i, j] = np.mean([simulate(cm, tau, beta, Live(delay=d), Logit(3.0, r), days=150, burn_in=50,
                                        seed=s).metrics["swing"] for s in range(max(3, seeds // 3))])
    SUMMARY["e5"] = {"revise": revise, "delays": delays, "swing": S.round(4).tolist()}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(S, origin="lower", aspect="auto", cmap=matplotlib_seq())
    ax.set_xticks(range(len(delays)), delays)
    ax.set_yticks(range(len(revise)), revise)
    ax.grid(False)
    for i in range(len(revise)):
        for j in range(len(delays)):
            ax.text(j, i, f"{S[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if S[i, j] > S.max() * 0.55 else viz.TEXT)
    ax.set(xlabel="display delay (meals)", ylabel="share of people who react each day",
           title="Crowd swing: share of students switching slot daily")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "e5_delay_swing")


# =============================================================== E6 correlated equilibrium
def e6_ce(seeds: int) -> None:
    plt = viz.plt
    inst, b = standard_instance()
    g = inst.game
    cm, tau, beta = g.congestion, inst.tau, inst.beta
    n = g.n
    ob_ne = obedience(g, b.best_nash.X, tol=b.epsilon + 1e-9)
    ob_opt = obedience(g, b.optimum.X)
    inc = obedience_incentive(g, b.optimum.X, b.best_nash.social_cost)
    sched, labels = interpolated_schedules(g, b.best_nash.X, b.optimum.X, steps=20)
    sched += [s.X for s in b.all_nash]
    labels += [f"Nash #{i}" for i in range(len(b.all_nash))]
    ce = best_ce(g, sched, labels, slack=b.epsilon)
    cce = best_ce(g, sched, labels, coarse=True, slack=b.epsilon)

    comp = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    rows = []
    for regime in ("calm (30% reconsider daily)", "reactive (everyone reconsiders daily)"):
        for rho in comp:
            pol = Recommend(target=b.best_nash.X, type_of=inst.type_of, compliance=rho, fallback=Live())
            m, h = ci([simulate(cm, tau, beta, pol, REGIMES[regime], days=150, burn_in=50, seed=s).metrics["mean_cost"]
                       for s in range(seeds)])
            rows.append({"regime": regime + " (non-users see live display)", "compliance": rho, "mean_cost": m, "ci": h})
    df = pd.DataFrame(rows)

    SUMMARY["e6"] = {
        "nash_cost": b.best_nash.social_cost / n, "opt_cost": b.optimum.social_cost / n,
        "nash_rec_max_gain_sec": 60 * ob_ne.max_gain, "nash_rec_is_ce": ob_ne.is_ce,
        "opt_rec_share_tempted": ob_opt.share_tempted, "opt_rec_max_gain_min": ob_opt.max_gain,
        "reward_per_person": inc.per_person, "share_rewarded": inc.share_rewarded,
        "reward_per_rewarded_person": inc.per_person / max(inc.share_rewarded, 1e-9),
        "saving_per_person": inc.welfare_gain / n,
        "best_ce_cost": ce.expected_cost / n, "best_cce_cost": cce.expected_cost / n,
        "best_ce_support": ce.support(), "best_cce_support": cce.support(),
        "compliance": df.round(3).to_dict("records"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    ax = axes[0]
    for (regime, sub), c in zip(df.groupby("regime", sort=False), [viz.CE, viz.SERIES[3]]):
        ax.errorbar(sub.compliance, sub.mean_cost, yerr=sub.ci, color=c, marker="o", ms=6, capsize=2,
                    label=regime.split(" (")[0] + " crowd")
    ax.axhline(b.best_nash.social_cost / n, color=viz.NE, ls="--", lw=1.2)
    ax.text(0.01, b.best_nash.social_cost / n * 1.05, "Nash", color=viz.NE, fontsize=9)
    plain_log(ax, "y", [2, 3, 5, 8, 12])
    ax.set(title="Share of students using CE recommendations", xlabel="compliance (share using the app)",
           ylabel="minutes per student (log)")
    ax.legend()
    ax = axes[1]
    bars = [("selfish\n(Nash)", b.best_nash.social_cost / n, viz.NE),
            ("best CE\n(advice only)", ce.expected_cost / n, viz.CE),
            ("best coarse CE", cce.expected_cost / n, viz.SERIES[6]),
            ("optimum +\nobedience reward", b.optimum.social_cost / n, viz.OPT)]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, 0.6, color=c)
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=9, color=viz.TEXT_2)
    ax.set_xticks(range(len(bars)), [x[0] for x in bars])
    ax.set(title="Advice alone cannot beat Nash; a small reward can", ylabel="minutes per student")
    fig.tight_layout()
    save(fig, "e6_correlated")


# =============================================================== E7 learning
def e7_learning(seeds: int) -> None:
    plt = viz.plt
    inst, b = standard_instance()
    cm, tau, beta, g = inst.game.congestion, inst.tau, inst.beta, inst.game
    learners = {"Hedge (app publishes every slot's wait)": Hedge(0.05),
                "Regret matching (app publishes every slot's wait)": RegretMatching(),
                "EXP3 (no app: only your own wait)": EXP3(0.05)}
    curves, rows = {}, []
    days = 300
    for name, beh in learners.items():
        runs = [simulate(cm, tau, beta, Live(), beh, days=days, burn_in=100, seed=s) for s in range(max(3, seeds // 3))]
        cost = np.mean([np.nanmean(r.cost, axis=1) for r in runs], axis=0)
        curves[name] = (cost, np.mean([r.swap_regret_curve for r in runs], axis=0))
        rows.append({"learner": name, **{k: np.mean([r.metrics[k] for r in runs]) for k in
                                         ("mean_cost", "mean_wait", "swing", "ext_regret", "swap_regret")}})
    df = pd.DataFrame(rows)
    SUMMARY["e7"] = {"table": df.round(3).to_dict("records"), "nash": b.best_nash.social_cost / g.n}
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for (name, (cost, reg)), c in zip(curves.items(), [viz.SERIES[3], viz.CE, viz.SERIES[4]]):
        k = 7
        smooth = np.convolve(cost, np.ones(k) / k, mode="valid")
        axes[0].plot(np.arange(len(smooth)) + k // 2, smooth, color=c, label=name)
        axes[1].plot(reg, color=c, label=name)
    axes[0].axhline(b.best_nash.social_cost / g.n, color=viz.NE, ls="--", lw=1.2)
    axes[0].set(title="Average cost per student (7-day average)", xlabel="day", ylabel="minutes")
    plain_log(axes[0], "y", [1, 2, 3, 5, 8])
    axes[1].set(title="Swap regret -> 0 means play is a correlated equilibrium", xlabel="day",
                ylabel="avg swap regret per day (min)")
    plain_log(axes[1], "y", [0.01, 0.1, 1, 10])
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "e7_learning")


# =============================================================== E8 Bayesian
def e8_bayes() -> None:
    inst, _ = standard_instance()
    g = inst.game
    mu = g.congestion.service_rate
    states = {"normal night": FluidQueue(mu), "slow night (counter at 70%)": FluidQueue(0.7 * mu)}
    rep = bayesian_analysis(g, states, [0.7, 0.3])
    per = pd.DataFrame(rep.per_state).T
    SUMMARY["e8"] = {"prior": [0.7, 0.3], "bne": rep.bne_cost / g.n, "informed_ne": rep.informed_ne_cost / g.n,
                     "uninformed_opt": rep.uninformed_opt_cost / g.n, "informed_opt": rep.informed_opt_cost / g.n,
                     "value_of_information": rep.value_of_information, "per_state": per.round(3).to_dict()}


# =============================================================== E9 nowcasting
def e9_nowcast() -> None:
    scans, _ = generate(weeks=8, seed=0)
    rows = []
    for meal in MEALS:
        for menu in ("normal", "special"):
            errs = [nowcast_errors(scans_per_slot(scans, mess, meal, menu)) for mess in MESSES]
            table = scans_per_slot(scans, "Jaiswal", meal, menu)
            rows.append({"meal": meal, "menu": menu, "mean scans per slot": table.to_numpy().mean(),
                         **{k: float(np.mean([e[k] for e in errs])) for k in errs[0]}})
    SUMMARY["e9"] = pd.DataFrame(rows).round(2).to_dict("records")


def write_markdown() -> None:
    s = SUMMARY
    lines = ["# Results (generated by `python experiments.py`)", ""]
    lines += ["All costs are minutes per student per meal (waiting + inconvenience of eating away from your"
              " preferred time). Synthetic data; 95% confidence intervals over random seeds.", ""]
    lines += ["## E0 Toy games", "", pd.DataFrame(s["e0"]["dinner"]).round(3).to_markdown(index=False), "",
              f"Intersection: best CE gives each driver {s['e0']['intersection_best_ce_payoff']:.2f}"
              f" vs 4.67 in the best symmetric Nash outcome.", ""]
    lines += ["## E1 Synthetic logs and calibration", "",
              f"{s['e1']['rows']:,} scan records for {s['e1']['students']} students over 8 weeks.", "",
              pd.DataFrame(s["e1"]["service_rate"]).to_markdown(index=False), "",
              pd.DataFrame(s["e1"]["preferences"]).T.round(3).to_markdown(), ""]
    lines += ["## E2 Price of anarchy", "", pd.DataFrame(s["e2"]["by_meal_day"]).to_markdown(index=False), "",
              f"Largest epsilon of any reported equilibrium: {s['e2']['max_epsilon_sec']:.1f} seconds.", "",
              f"On a normal day the four messes lose {s['e2']['person_hours_lost_per_day_all_messes']:.0f}"
              " student-hours to selfish timing (all three meals).", "",
              pd.DataFrame(s["e2"]["domains"]).to_markdown(index=False), ""]
    t = pd.DataFrame(s["e3"]["table"])[["regime", "policy", "mean_cost", "mean_cost_ci", "p90_wait", "swing",
                                        "swap_regret"]]
    lines += ["## E3 Information policies", "", f"Nash = {s['e3']['nash']:.2f}, optimum = {s['e3']['opt']:.2f}.",
              "", t.round(2).to_markdown(index=False), "", "Worst case over the three behaviour regimes:", "",
              pd.Series(s["e3"]["worst_case_by_policy"]).rename("worst mean cost").to_markdown(), ""]
    lines += ["## E4 Sweeps (reactive crowd)", ""]
    for k, v in s["e4"].items():
        lines += [f"**{k}**", "", pd.DataFrame(v).to_markdown(index=False), ""]
    e5 = pd.DataFrame(s["e5"]["swing"], index=s["e5"]["revise"], columns=s["e5"]["delays"])
    lines += ["## E5 Crowd swing (rows: share reacting daily, columns: delay in meals)", "",
              e5.round(3).to_markdown(), ""]
    e6 = s["e6"]
    lines += ["## E6 Correlated recommendations", "",
              f"- Nash-based recommendation: max gain from disobeying {e6['nash_rec_max_gain_sec']:.1f} s"
              " (an epsilon-CE).",
              f"- Optimum-based recommendation: {e6['opt_rec_share_tempted']:.0%} of students tempted,"
              f" up to {e6['opt_rec_max_gain_min']:.1f} min.",
              f"- Reward that makes the optimum obedient: {e6['reward_per_person']:.2f} min per student on average"
              f" ({e6['share_rewarded']:.0%} of students get {e6['reward_per_rewarded_person']:.1f} min),"
              f" saving {e6['saving_per_person']:.2f} min per student.",
              f"- Best CE over 26 schedules: {e6['best_ce_cost']:.3f}; best coarse CE: {e6['best_cce_cost']:.3f};"
              f" Nash {e6['nash_cost']:.3f}; optimum {e6['opt_cost']:.3f}.", "",
              pd.DataFrame(e6["compliance"]).to_markdown(index=False), ""]
    lines += ["## E7 Learning", "", pd.DataFrame(s["e7"]["table"]).round(3).to_markdown(index=False), ""]
    e8 = s["e8"]
    lines += ["## E8 Complete vs incomplete information", "",
              f"Expected cost: no announcement (Bayesian Nash) {e8['bne']:.2f}; slow night announced"
              f" {e8['informed_ne']:.2f}; value of information {e8['value_of_information']:.2f} min per student.", "",
              pd.DataFrame(e8["per_state"]).to_markdown(), ""]
    lines += ["## E9 Nowcasting (mean absolute error, scans per 10-min slot, next 3 slots; 4 messes)", "",
              pd.DataFrame(s["e9"]).to_markdown(index=False), ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 seeds instead of 30")
    args = ap.parse_args()
    seeds = 5 if args.quick else 30
    FIG.mkdir(parents=True, exist_ok=True)
    viz.style()
    steps = [("E0 theory", e0_theory), ("E1 data", e1_data), ("E2 price of anarchy", e2_poa),
             ("E3 policies", lambda: e3_policies(seeds)), ("E4 sweeps", lambda: e4_sweeps(seeds)),
             ("E5 delay", lambda: e5_delay(seeds)), ("E6 correlated eq.", lambda: e6_ce(seeds)),
             ("E7 learning", lambda: e7_learning(seeds)), ("E8 Bayesian", e8_bayes), ("E9 nowcasting", e9_nowcast)]
    for name, fn in steps:
        t0 = time.time()
        fn()
        print(f"{name:22s} done in {time.time() - t0:5.1f}s")
    write_markdown()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
