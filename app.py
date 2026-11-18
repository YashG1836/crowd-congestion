"""Interactive lab:  streamlit run app.py

Every page starts with three lines: what you see, why it matters, what to try.
Pages follow the story: start here -> small games -> selfish vs planner ->
the display -> recommendations -> learning & uncertainty -> mess data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crowd import viz
from crowd.correlated import (bayesian_analysis, best_ce, fair_scheme, interpolated_schedules, obedience,
                              obedience_incentive, queue_cost_game, recommend)
from crowd.domains import DOMAINS, build
from crowd.game import FluidQueue, benchmarks
from crowd.mess import (MEALS, MESSES, arrival_heatmap, estimate_preferred_slots, estimate_service_rate, generate,
                        meal_instance, nowcast_errors, preference_accuracy, roster_truth, scans_per_slot)
from crowd.simulate import (EXP3, UCB, Coarse, EpsilonGreedy, Forecast, Hedge, Live, Logit, Noisy, NoInfo, Partial,
                            Recommend, RegretMatching, evening, simulate, solve_go_wait)
from crowd.theory import (backward_induction, dinner_game, hedge, intersection_game, iterated_elimination, pigou_poa,
                          regret_matching, sequential_dinner_game, sequential_dinner_normal_form,
                          shared_counter_dynamics)

st.set_page_config(page_title="Coordinating Crowds: interactive lab", layout="wide")

LINKS = {"website": "https://yashg1836.github.io/crowd-congestion/",
         "report": "https://github.com/YashG1836/crowd-congestion/blob/main/report/main.pdf",
         "code": "https://github.com/YashG1836/crowd-congestion"}
BEHAVIOURS = {
    "Calm: 30% of students reconsider each day": (3.0, 0.3),
    "Reactive: everyone reconsiders each day": (3.0, 1.0),
    "Noisy: low rationality": (1.0, 0.3),
}


# ------------------------------------------------------------------ cached computations
@st.cache_resource(show_spinner="Solving the game (Nash equilibrium and optimum)...")
def solve_mess(mess: str, meal: str, day: str, mu: float | None):
    inst = meal_instance(mess, meal, day, service_rate=mu)
    return inst, benchmarks(inst.game)


@st.cache_resource(show_spinner="Solving the game...")
def solve_domain(key: str, mu: float | None):
    inst = build(DOMAINS[key], service_rate=mu)
    return inst, benchmarks(inst.game)


@st.cache_resource(show_spinner="Generating 8 weeks of synthetic QR scans...")
def logs():
    return generate(weeks=8, seed=0)


# ------------------------------------------------------------------ small helpers
def intro(what: str, why: str, try_: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**What you see** · {what}  \n**Why it matters** · {why}" + (f"  \n**Try this** · {try_}" if try_ else ""))


def fig_layout(fig: go.Figure, title: str, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(title=title, height=height, **{**viz.PLOTLY_LAYOUT, **kw})
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")


def loads_chart(game, schedules: dict[str, tuple[np.ndarray, str]], title: str) -> go.Figure:
    fig = go.Figure()
    for name, (X, color) in schedules.items():
        fig.add_bar(x=game.slot_labels, y=game.loads(X), name=name, marker_color=color)
    fig.add_hline(y=game.congestion.capacity, line_dash="dash", line_color=viz.MUTED,
                  annotation_text="capacity per slot", annotation_position="top left")
    return fig_layout(fig, title, barmode="group", yaxis_title="arrivals", hovermode="x")


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    for col, (label, value, help_) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_)


SEQ = [[0, "#f4f8fd"]] + [[(i + 1) / 7, c] for i, c in enumerate(viz.SEQ_BLUE)]


def pick_instance(key: str):
    """Choose the mess case study or another domain; returns (instance, benchmarks)."""
    mode = st.radio("Facility", ["Mess case study", "Other domain"], horizontal=True, key=key + "mode")
    if mode == "Mess case study":
        c = st.columns(4)
        mess = c[0].selectbox("Mess", list(MESSES), key=key + "mess")
        meal = c[1].selectbox("Meal", list(MEALS), index=2, key=key + "meal")
        day = c[2].selectbox("Day", ["normal", "special", "special_scarce"], key=key + "day",
                             format_func=lambda d: {"normal": "normal menu", "special": "special menu (dosa night)",
                                                    "special_scarce": "special + item can run out"}[d])
        default_mu = MEALS[meal].service_rate * (1.0 if day == "normal" else 0.8)
        mu = c[3].slider("serving rate (students/min)", 4.0, 14.0, float(default_mu), 0.5, key=key + "mu")
        return solve_mess(mess, meal, day, None if np.isclose(mu, default_mu) else mu)
    c = st.columns(2)
    dom = c[0].selectbox("Domain", [k for k in DOMAINS if k != "mess"], format_func=lambda k: DOMAINS[k].title,
                         key=key + "dom")
    d = DOMAINS[dom]
    mu = c[1].slider(f"service rate ({d.unit}/min)", d.service_rate * 0.5, d.service_rate * 1.6,
                     float(d.service_rate), d.service_rate / 20, key=key + "dmu")
    st.caption(d.story)
    return solve_domain(dom, None if np.isclose(mu, d.service_rate) else mu)


# ================================================================== pages
def page_start() -> None:
    st.title("Coordinating crowds")
    st.markdown("Many people share one bottleneck and each chooses **when** to use it: cars at a bridge, backups on a "
                "network link, diners at a food court, students at the hostel mess. Everyone wants a convenient time "
                "and a short queue; every person who joins a queue delays everyone behind them. **It is a "
                "coordination problem, not a capacity problem.** This lab lets you run every experiment of the "
                "project yourself.")
    c = st.columns(3)
    c[0].link_button("Website (the short story)", LINKS["website"], width="stretch")
    c[1].link_button("Report (PDF)", LINKS["report"], width="stretch")
    c[2].link_button("Source code", LINKS["code"], width="stretch")
    inst, b = solve_mess("Jaiswal", "dinner", "normal", None)
    g = inst.game
    live = simulate(g.congestion, inst.tau, inst.beta, Live(), Logit(3.0, 1.0), days=150, burn_in=50, seed=0)
    st.markdown("#### The story in four steps (Jaiswal mess, a normal dinner)")
    s = st.columns(4)
    s[0].metric("1 · Selfish timing", f"{b.nash.social_cost / g.n:.2f} min",
                help="Nash equilibrium cost per student (waiting + eating away from your favourite time)")
    s[1].metric("2 · Planner's schedule", f"{b.optimum.social_cost / g.n:.2f} min", help="the social optimum")
    s[2].metric("3 · Everyone reacts to a live counter", f"{live.metrics['mean_cost']:.2f} min",
                help="the rush jumps from slot to slot (El Farol)")
    s[3].metric("4 · CE advice + small reward", f"{b.optimum.social_cost / g.n:.2f} min",
                help="private recommendations people want to follow, made self-enforcing by a reward")
    st.markdown("#### How to use this lab")
    st.markdown(
        "Use the menu on the left, top to bottom:\n"
        "1. **Small games**: the ideas in games you can solve by hand (dominance, Nash, backward induction, correlated equilibrium).\n"
        "2. **Selfish vs planner**: how much selfish timing costs, and who causes the queue (Shapley value).\n"
        "3. **The display**: what happens day by day under each display policy, and the live counter within an evening (MDP).\n"
        "4. **Recommendations**: correlated equilibrium, rewards, a student's phone, fairness.\n"
        "5. **Learning & uncertainty**: students who learn (bandits, RL) and nights nobody announces (Bayesian game).\n"
        "6. **Mess data**: the synthetic QR-scan logs and what we can learn from them.")
    st.markdown("#### The same game everywhere")
    rows = []
    for d in DOMAINS.values():
        _, bd = solve_domain(d.key, None)
        rows.append({"facility": d.title, "who": f"{d.n} {d.unit}", "selfish (min)": round(bd.nash.social_cost / d.n, 2),
                     "planner (min)": round(bd.optimum.social_cost / d.n, 2), "price of anarchy": round(bd.poa, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("All data are synthetic. Costs are minutes per person per visit.")


def page_games() -> None:
    st.header("1 · Small games")
    intro("Games small enough to solve exactly.",
          "They carry the whole story: selfishness wastes, shared information can backfire, and private "
          "correlated advice can beat every Nash equilibrium.",
          "move the sliders and watch which equilibrium concept gives which outcome.")
    t1, t2, t3, t4 = st.tabs(["Dominance & Pigou", "Dinner game: simultaneous vs sequential", "Traffic light (CE)",
                              "Learning finds equilibria"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pigou's network.** Road A costs 1; road B costs x^p when a share x uses it. Selfishly "
                        "everyone takes B; a planner splits the traffic.")
            p = st.slider("steepness p", 1.0, 12.0, 1.0, 0.5)
            metric_row([("price of anarchy", f"{pigou_poa(p):.3f}", "4/3 for linear costs"),
                        ("planner's share on B", f"{(p + 1) ** (-1 / p):.0%}", None)])
            ps = np.linspace(1, 12, 100)
            fig = go.Figure(go.Scatter(x=ps, y=[pigou_poa(v) for v in ps], line_color=viz.NE))
            fig.add_scatter(x=[p], y=[pigou_poa(p)], mode="markers", marker=dict(size=11, color=viz.NE))
            show(fig_layout(fig, "Steeper congestion, bigger waste", 280, xaxis_title="p", showlegend=False))
        with c2:
            st.markdown("**Dominance.** Two students: Peak costs the number at Peak (1 or 2); Late costs L. If L > 2, "
                        "Late costs more than Peak whatever the other does: it is *strictly dominated*.")
            L = st.slider("cost of going late, L", 1.1, 2.9, 2.5, 0.1, key="domL")
            alive, log = iterated_elimination(dinner_game(L))
            g = dinner_game(L)
            st.dataframe(pd.DataFrame([[f"({g.costs[0, a, b]:g}, {g.costs[1, a, b]:g})" for b in (0, 1)] for a in (0, 1)],
                                      index=["S1: Peak", "S1: Late"], columns=["S2: Peak", "S2: Late"]))
            if log:
                st.warning("Iterated elimination: " + "; ".join(log) + ". Only (Peak, Peak) survives: everyone "
                           "rushes, as in Pigou's network, and no advice can change that.")
            else:
                st.info("Nothing is dominated, so dominance does not predict the outcome: we need Nash or correlated "
                        "equilibrium.")
    with t2:
        L = st.slider("cost of going late, L", 1.05, 1.95, 1.5, 0.05, key="dinL")
        g = dinner_game(L)
        mixed = [m for m in g.mixed_nash_2p() if 0 < m[0][0] < 1][0]
        path = shared_counter_dynamics(g, (1, 1), 20)
        ce = g.correlated_equilibrium("fair")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Simultaneous** (nobody sees the others)")
            res = pd.DataFrame({"what students do": ["shared counter: both react to yesterday", "no information: mixed Nash",
                                                     "private fair coin: a correlated equilibrium"],
                                "total cost": [np.mean([g.total_cost(s) for s in path]),
                                               g.expected_costs(g.product_distribution(list(mixed))).sum(), ce.total_cost]})
            res["vs optimum"] = (res["total cost"] / g.social_optimum()[1]).round(2)
            st.dataframe(res.round(3), hide_index=True, width="stretch")
            st.caption(f"Mixed Nash: each goes Peak with probability {mixed[0][0]:.2f}. The coin tells one student Peak and "
                       "the other Late; both obey, and each is told Peak half the time.")
        with c2:
            st.markdown("**Sequential** (a counter during the evening shows student 1's choice to student 2)")
            value, strategy, sg_path = backward_induction(sequential_dinner_game(L))
            nf = sequential_dinner_normal_form(L)
            st.dataframe(pd.DataFrame({"decision": list(strategy), "backward induction": list(strategy.values())}),
                         hide_index=True, width="stretch")
            st.success(f"Subgame-perfect outcome {tuple(sg_path)}, costs {value}: efficient (total {sum(value):.2f}) "
                       "but the first mover gets the better slot.")
            st.caption("Nash equilibria of the same game in normal form: " +
                       ", ".join(nf.profile_name(x) for x in nf.pure_nash()) +
                       ". (Late, always Peak) rests on a threat student 2 would not carry out, so it is not subgame perfect.")
    with t3:
        ig = intersection_game()
        st.markdown("Two drivers: **Stop** or **Go**. Payoffs (higher is better): both stop 6/6, one goes 7 vs 2, "
                    "both go = crash 0/0. Nash equilibria: (Stop, Go), (Go, Stop), and a mixed one worth 4.67 each.")
        best = ig.correlated_equilibrium("best")
        metric_row([("best symmetric Nash mixture", "4.67 each", None),
                    ("best correlated equilibrium (LP)", f"{-best.expected_costs[0]:.2f} each", None)])
        st.dataframe(pd.DataFrame({"traffic light says": list(best.support()), "probability": list(best.support().values())}),
                     hide_index=True)
    with t4:
        ig = intersection_game()
        rounds = st.select_slider("rounds", [1000, 3000, 6000, 12000], 6000)
        rm = regret_matching(ig, rounds=rounds, seed=0, log_every=25)
        hd = hedge(ig, rounds=rounds, seed=0, log_every=25)
        fig = go.Figure()
        fig.add_scatter(x=rm.rounds_logged, y=rm.ce_violation, name="regret matching -> CE", line_color=viz.CE)
        fig.add_scatter(x=hd.rounds_logged, y=hd.cce_violation, name="Hedge -> coarse CE", line_color=viz.SERIES[3])
        show(fig_layout(fig, "Gap to equilibrium of the empirical joint play", 320, yaxis_type="log",
                        xaxis_title="round", yaxis_title="largest gain from deviating"))
        st.caption("No-swap-regret learners end up at a correlated equilibrium, no-external-regret learners at a coarse "
                   "one. A mediator gets there on day one.")


def page_poa() -> None:
    st.header("2 · Selfish vs planner")
    intro("Arrivals per slot when everyone picks selfishly (Nash equilibrium) vs the planner's schedule.",
          "Each person in a queue delays everyone behind them but feels only their own wait, so selfish timing wastes time.",
          "switch to a special-menu night, change the serving rate, or pick another domain (road, network, ...).")
    inst, b = pick_instance("poa")
    g = inst.game
    t1, t2 = st.tabs(["Price of anarchy", "Who causes the queue? (Shapley value, core)"])
    with t1:
        ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
        metric_row([("people", f"{g.n}", None),
                    ("selfish cost / person", f"{ne['per_person']:.2f} min", f"waiting {ne['wait_per_person']:.2f}"),
                    ("planner cost / person", f"{op['per_person']:.2f} min", f"waiting {op['wait_per_person']:.2f}"),
                    ("price of anarchy", f"{b.poa:.2f}" if ne["per_person"] >= 0.1 else "n/a (no queue)", None),
                    ("equilibrium accuracy", f"{60 * b.epsilon:.1f} s", "largest gain any one person could still get")])
        show(loads_chart(g, {"selfish (Nash)": (b.nash.X, viz.NE), "planner (optimum)": (b.optimum.X, viz.OPT)},
                         "When people arrive"))
        fig = go.Figure()
        fig.add_scatter(x=g.slot_labels, y=g.congestion.wait(g.loads(b.nash.X).astype(float)), name="selfish", line_color=viz.NE)
        fig.add_scatter(x=g.slot_labels, y=g.congestion.wait(g.loads(b.optimum.X).astype(float)), name="planner", line_color=viz.OPT)
        show(fig_layout(fig, "Wait of someone arriving in each slot (minutes)", 280))
        st.caption("The planner feeds arrivals at exactly the serving rate, so the queue never builds. The bumpy Nash "
                   "profile is real: the slot right after a rush inherits its queue.")
    with t2:
        r = queue_cost_game(g)
        st.markdown("Players are the **arrival groups** (everyone who prefers the same slot). The cost of a coalition is "
                    "the total waiting if only those groups come at their preferred times. The **Shapley value** is a "
                    "group's average extra queue over all orders in which groups could arrive.")
        fig = go.Figure()
        fig.add_bar(x=r.groups, y=r.shapley_per_member, name="queue caused (Shapley), per person", marker_color=viz.NE)
        fig.add_bar(x=r.groups, y=r.own_wait_per_member, name="queue suffered, per person", marker_color=viz.CE)
        show(fig_layout(fig, "Caused vs suffered, by preferred time", 320, barmode="group", yaxis_title="minutes"))
        metric_row([("total waiting", f"{r.total:.0f} min", "= sum of Shapley shares"),
                    ("core", "empty" if r.core_empty else "non-empty", "LP: no split of the waiting satisfies every coalition"),
                    ("least-core epsilon", f"{r.least_core_eps:.0f} min", None),
                    ("cost game supermodular", "yes" if r.supermodular else "no", "queues make costs grow faster than linearly")])
        st.caption("Students at the start of a rush cause more queue than they suffer; later arrivals suffer more than "
                   "they cause. Because waiting is a negative externality, no coalition-proof way to split it exists.")


def run_policy(inst, policy, lam, revise, days, seed=0):
    return simulate(inst.game.congestion, inst.tau, inst.beta, policy, Logit(lam, revise), days=days,
                    burn_in=min(50, days // 3), seed=seed)


def make_policy(name: str, inst, b, params: dict):
    g = inst.game
    simple = {"pi0 No information": NoInfo(), "pi1 Exact live": Live(), "pi2 Delayed": Live(delay=params.get("delay", 3)),
              "pi3 Noisy": Noisy(params.get("sigma", 3.0)), "pi4 Coarse": Coarse(params.get("levels", 3)),
              "pi5 Partial": Partial(Live(), params.get("share", 0.3)), "pi6 Forecast": Forecast(0.3)}
    if name in simple:
        return simple[name]
    X = b.best_nash.X if name.startswith("pi7") else b.optimum.X
    ob = obedience(g, X)
    common = dict(target=X, type_of=inst.type_of, rational=True, gains=ob.gain_by_cell,
                  best_alt=g.deviation_costs(X).argmin(2), compliance=params.get("compliance", 1.0), fallback=Live())
    if name.startswith("pi8"):
        return Recommend(bonus=ob.gain_by_cell, name=name, **common)
    return Recommend(name=name, **common)


POLICIES = ["pi0 No information", "pi1 Exact live", "pi2 Delayed", "pi3 Noisy", "pi4 Coarse", "pi5 Partial",
            "pi6 Forecast", "pi7 CE recommendation", "Optimum as advice, no reward",
            "pi8 Optimum as advice + obedience reward"]


def page_display() -> None:
    st.header("3 · What should the display show?")
    intro("The same students choose a slot every day; the display shows them something first.",
          "Information that everyone reacts to the same way can make the rush jump around (El Farol).",
          "pick 'Reactive' behaviour and 'pi1 Exact live', then compare with 'pi7 CE recommendation'.")
    inst, b = pick_instance("pol")
    g = inst.game
    t1, t2 = st.tabs(["Day by day (display policies)", "Within one evening: go now or wait? (MDP)"])
    with t1:
        c = st.columns(3)
        lam, revise = BEHAVIOURS[c[0].selectbox("How students behave", list(BEHAVIOURS), index=1)]
        days = c[1].slider("days simulated", 60, 300, 150, 30)
        policy = c[2].selectbox("Display policy", POLICIES, index=1)
        params = {}
        if policy == "pi2 Delayed":
            params["delay"] = st.slider("delay (meals)", 1, 6, 3)
        if policy == "pi3 Noisy":
            params["sigma"] = st.slider("private noise (sd, minutes)", 0.0, 12.0, 3.0, 0.5)
        if policy == "pi4 Coarse":
            params["levels"] = st.slider("number of levels", 2, 10, 3)
        if policy == "pi5 Partial":
            params["share"] = st.slider("share of students who see the live display", 0.0, 1.0, 0.3, 0.05)
        if policy.startswith(("pi7", "pi8", "Optimum")):
            params["compliance"] = st.slider("share of students using the app (others see the live display)", 0.0, 1.0, 1.0, 0.05)
        res = run_policy(inst, make_policy(policy, inst, b, params), lam, revise, days)
        m = res.metrics
        nash, opt = b.best_nash.social_cost / g.n, b.optimum.social_cost / g.n
        metric_row([("avg cost / person", f"{m['mean_cost']:.2f} min", "after the first third of the days"),
                    ("vs Nash", f"{m['mean_cost'] / nash:.2f}x", f"Nash = {nash:.2f} min"),
                    ("90th-pct wait", f"{m['p90_wait']:.1f} min", None),
                    ("crowd swing", f"{m['swing']:.0%}", "share of the crowd that changes slot vs. the previous day"),
                    ("regret", f"{m['swap_regret']:.2f} min", "0 = nobody would gain by changing their habit")])
        fig = go.Figure(go.Heatmap(z=res.loads.T, x=np.arange(days), y=g.slot_labels, colorscale=SEQ,
                                   colorbar=dict(title="arrivals"),
                                   hovertemplate="day %{x}<br>%{y}: %{z} arrivals<extra></extra>"))
        show(fig_layout(fig, "Arrivals per slot, day by day (stripes = the rush jumping around)", 360,
                        xaxis_title="day", hovermode="closest"))
        fig = go.Figure(go.Scatter(x=np.arange(days), y=np.nanmean(res.cost, axis=1), line_color=viz.CE))
        fig.add_hline(y=nash, line_dash="dash", line_color=viz.NE, annotation_text="Nash")
        fig.add_hline(y=opt, line_dash="dash", line_color=viz.OPT, annotation_text="optimum")
        show(fig_layout(fig, "Average cost per student each day", 260, yaxis_title="minutes", showlegend=False))
        if st.button("Compare all policies for this behaviour (10 seeds)"):
            rows, prog = [], st.progress(0.0)
            for i, name in enumerate(POLICIES):
                ms = [run_policy(inst, make_policy(name, inst, b, {}), lam, revise, 150, seed=s).metrics for s in range(10)]
                rows.append({"policy": name, **{k: np.mean([x[k] for x in ms]) for k in ("mean_cost", "p90_wait", "swing", "swap_regret")}})
                prog.progress((i + 1) / len(POLICIES))
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    with t2:
        st.markdown("A ready student sees the live queue every minute and chooses **go now** (wait queue/μ) or **wait** "
                    "(pay a delay cost β per slot). Solving this Markov decision process by value iteration gives a "
                    "threshold: go if the queue is below q*(t). The catch: the MDP assumes the crowd does not react.")
        c = st.columns(2)
        share = c[0].slider("share of students using the live counter", 0.0, 1.0, 1.0, 0.05)
        delay = c[1].slider("the counter shows the queue from ... minutes ago", 0, 15, 0)
        rng = np.random.default_rng(0)
        dt = g.congestion.slot_minutes
        minutes = int(g.T * dt)
        tau_i, beta_i = inst.tau, inst.beta
        ready = (tau_i * dt + rng.uniform(0, dt, len(tau_i))).astype(int)
        lam = np.bincount(ready, minlength=minutes)[:minutes].astype(float)
        dcost = beta_i / dt
        pols = {d: solve_go_wait(lam, g.congestion.service_rate, d) for d in np.unique(dcost)}
        use = rng.random(len(ready)) < share
        base = evening(ready, dcost, np.zeros(len(ready), bool), g.congestion.service_rate, pols, minutes)
        res = evening(ready, dcost, use, g.congestion.service_rate, pols, minutes, display_delay=delay)
        metric_row([("no display", f"{base.cost.mean():.2f} min", "everyone goes when ready"),
                    ("with this counter", f"{res.cost.mean():.2f} min", "wait + delay cost, everyone"),
                    ("counter users", f"{res.cost[use].mean():.2f} min" if use.any() else "–", None),
                    ("longest queue", f"{res.queue.max():.0f}", f"no display: {base.queue.max():.0f}")])
        fig = go.Figure()
        fig.add_scatter(y=base.queue[: minutes + 30], name="no display", line_color=viz.MUTED)
        fig.add_scatter(y=res.queue[: minutes + 30], name="with this counter", line_color=viz.CE)
        show(fig_layout(fig, "Queue during the evening", 300, xaxis_title="minutes after opening", yaxis_title="people"))
        fig = go.Figure()
        for (d, pol), col in zip(sorted(pols.items()), [viz.OPT, viz.CE, viz.NE]):
            fig.add_scatter(y=np.minimum(pol.thresholds, 200), name=f"delay cost {d:g} min/min", line_color=col)
        show(fig_layout(fig, "MDP policy: go if the queue is below this", 260, xaxis_title="minute", yaxis_title="queue"))
        st.caption("A fresh counter helps: students wait out the peak. A stale one makes them all go together when the "
                   "old number looked small, so the rush comes back.")


def page_ce() -> None:
    st.header("4 · Recommendations people want to follow")
    intro("The app privately tells each student a slot (shuffled every day inside groups of similar students).",
          "Advice is a correlated equilibrium when nobody gains by disobeying. Then it is stable without any police.",
          "open the phone below and switch between the Nash schedule, the optimum, and the optimum with a reward.")
    inst, b = pick_instance("ce")
    g = inst.game
    n = g.n
    ob_ne = obedience(g, b.best_nash.X, tol=b.epsilon + 1e-9)
    ob_opt = obedience(g, b.optimum.X)
    inc = obedience_incentive(g, b.optimum.X, b.best_nash.social_cost)
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("##### Shuffle a Nash schedule")
        st.metric("cost / student", f"{b.best_nash.social_cost / n:.2f} min")
        st.metric("largest gain from disobeying", f"{60 * ob_ne.max_gain:.1f} s")
        st.success("Obeyed: a correlated equilibrium.")
    with c2, st.container(border=True):
        st.markdown("##### Shuffle the optimum")
        st.metric("cost / student if obeyed", f"{b.optimum.social_cost / n:.2f} min")
        st.metric("students tempted to disobey", f"{ob_opt.share_tempted:.0%}", f"up to {ob_opt.max_gain:.1f} min gain",
                  delta_color="off")
        st.error("Not a CE: off-peak students would rather go at the peak.")
    with c3, st.container(border=True):
        st.markdown("##### Optimum + obedience reward")
        st.metric("average reward", f"{inc.per_person:.2f} min / student", f"{inc.share_rewarded:.0%} get one", delta_color="off")
        st.metric("time saved vs Nash", f"{inc.welfare_gain / n:.2f} min / student")
        st.success("With the reward, obeying is a best response again.")

    st.markdown("#### A student's phone")
    c1, c2 = st.columns([1, 2])
    which = c1.radio("schedule the app uses", ["Nash (CE)", "Optimum, no reward", "Optimum + reward"])
    day = c1.number_input("day", 1, 365, 1)
    X = b.best_nash.X if which.startswith("Nash") else b.optimum.X
    rec = recommend(np.random.default_rng(day), inst.type_of, X)
    ids = getattr(inst, "ids", None)
    who = c1.selectbox("student", list(range(min(n, 400))), format_func=lambda i: ids[i] if ids is not None else f"person {i}")
    k, t = inst.type_of[who], rec[who]
    D = g.deviation_costs(X)[k, t].copy()
    if which == "Optimum + reward":
        D[t] -= ob_opt.gain_by_cell[k, t]
    c1.markdown(f"**Your slot today: {g.slot_labels[t]}**  \nyou like {g.slot_labels[g.tau[k]]}")
    ok = D[t] <= D.min() + max(b.epsilon, 1e-9) + 1e-9
    fig = go.Figure(go.Bar(x=g.slot_labels, y=D, marker_color=[viz.CE if s == t else viz.MUTED for s in range(g.T)],
                           hovertemplate="%{x}: %{y:.2f} min<extra></extra>"))
    verdict = "obeying is your best choice" if ok else f"you would rather go at {g.slot_labels[int(np.argmin(D))]}"
    c2.plotly_chart(fig_layout(fig, f"Your cost in each slot if everyone else follows the app: {verdict}", 330,
                               yaxis_title="minutes", showlegend=False), width="stretch")

    t1, t2 = st.tabs(["Fairness: who carries the cost?", "Can a cleverer CE beat Nash? (LP)"])
    with t1:
        f = fair_scheme(g, b.best_nash.X, b.optimum.X)
        groups = {"flexible (beta 1.5)": g.beta == 1.5, "average (beta 3)": g.beta == 3.0, "rigid (beta 6)": g.beta == 6.0,
                  "everyone": np.ones(g.K, bool)}
        cols = {"nash": "selfish (Nash)", "optimum": "optimum, no reward", "with_reward": "+ obedience reward",
                "with_fee": "+ Shapley-proportional fee"}
        st.dataframe(pd.DataFrame([{"group": nm, **{lab: round(f.group_mean(kk, m), 3) for kk, lab in cols.items()}}
                                   for nm, m in groups.items() if m.any()]), hide_index=True, width="stretch")
        st.caption(f"Rewards are paid for by a fee proportional to the queue each arrival group causes (its Shapley "
                   f"share), so the scheme pays for itself. People worse off than under selfish play: without reward "
                   f"{f.worse_off('optimum')}, with reward {f.worse_off('with_reward')}, with reward and fee "
                   f"{f.worse_off('with_fee')} (by at most {60 * max(f.max_loss('with_fee'), 0):.1f} s).")
    with t2:
        if st.button("Solve the best-CE linear program over 26 candidate schedules"):
            sched, labels = interpolated_schedules(g, b.best_nash.X, b.optimum.X, steps=20)
            sched += [s.X for s in b.all_nash]
            labels += [f"Nash #{i}" for i in range(len(b.all_nash))]
            ce = best_ce(g, sched, labels, slack=b.epsilon)
            cce = best_ce(g, sched, labels, coarse=True, slack=b.epsilon)
            metric_row([("Nash", f"{b.best_nash.social_cost / n:.3f}", None), ("best CE", f"{ce.expected_cost / n:.3f}", None),
                        ("best coarse CE", f"{cce.expected_cost / n:.3f}", None), ("optimum", f"{b.optimum.social_cost / n:.3f}", None)])
        st.caption("Theory (report, Proposition 2): in a large crowd with monotone congestion costs every (coarse) "
                   "correlated equilibrium costs the same as Nash. Advice coordinates; rewards are needed to do better.")


def page_learning() -> None:
    st.header("5 · Learning and uncertainty")
    intro("Students who learn from experience instead of reacting to a display, and nights nobody announces.",
          "Learning rules from the course (bandits, TD, no-regret) end up near equilibrium, but how fast depends on "
          "what the app publishes.", "compare bandit learners (own wait only) with full-information learners.")
    inst, b = solve_mess("Jaiswal", "dinner", "normal", None)
    g = inst.game
    t1, t2 = st.tabs(["Students who learn", "Should the mess announce slow nights?"])
    with t1:
        options = {"epsilon-greedy, Monte Carlo average (bandit)": lambda: EpsilonGreedy(0.05, None),
                   "epsilon-greedy, TD step 0.1 (bandit)": lambda: EpsilonGreedy(0.05, 0.1),
                   "UCB1 (bandit)": lambda: UCB(2.0), "EXP3 (bandit)": lambda: EXP3(0.05),
                   "Hedge (app publishes all waits)": lambda: Hedge(0.05),
                   "Regret matching (app publishes all waits)": lambda: RegretMatching()}
        chosen = st.multiselect("learners", list(options), default=list(options)[:2] + list(options)[4:])
        days = st.slider("days", 100, 400, 250, 50)
        fig1, fig2, rows = go.Figure(), go.Figure(), []
        for name, col in zip(chosen, viz.SERIES):
            r = simulate(g.congestion, inst.tau, inst.beta, Live(), options[name](), days=days, burn_in=days // 3, seed=0)
            fig1.add_scatter(y=pd.Series(np.nanmean(r.cost, axis=1)).rolling(7, min_periods=1).mean(), name=name, line_color=col)
            fig2.add_scatter(y=r.swap_regret_curve, name=name, line_color=col)
            rows.append({"learner": name, **{k: round(r.metrics[k], 3) for k in ("mean_cost", "swing", "swap_regret")}})
        fig1.add_hline(y=b.best_nash.social_cost / g.n, line_dash="dash", line_color=viz.NE, annotation_text="Nash")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_layout(fig1, "Cost per student (7-day average)", 340, yaxis_type="log", xaxis_title="day"), width="stretch")
        c2.plotly_chart(fig_layout(fig2, "Swap regret (0 = correlated equilibrium)", 340, yaxis_type="log", xaxis_title="day"), width="stretch")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("UCB is deterministic, so identical students make identical choices and herd onto their favourite slot.")
    with t2:
        c = st.columns(2)
        p = c[0].slider("probability of a slow night", 0.05, 0.6, 0.3, 0.05)
        f = c[1].slider("slow-night serving rate (share of normal)", 0.5, 0.95, 0.7, 0.05)
        rep = bayesian_cached(p, f, g.congestion.service_rate)
        metric_row([("no announcement (Bayesian Nash)", f"{rep['bne']:.2f} min", None),
                    ("night announced", f"{rep['informed']:.2f} min", None),
                    ("value of the announcement", f"{rep['voi']:+.2f} min / student", "positive = it helps")])
        st.dataframe(pd.DataFrame(rep["per_state"]).T.rename(columns={
            "bne": "no announcement", "informed_ne": "announced", "uninformed_opt": "planner, no info",
            "informed_opt": "planner, informed"}).round(2), width="stretch")
        st.caption("On normal nights not knowing is better: the fear of a slow night keeps people spread out.")


@st.cache_data(show_spinner="Solving three games...")
def bayesian_cached(p: float, f: float, mu: float) -> dict:
    inst, _ = solve_mess("Jaiswal", "dinner", "normal", None)
    rep = bayesian_analysis(inst.game, {"normal night": FluidQueue(mu), "slow night": FluidQueue(f * mu)}, [1 - p, p])
    return {"bne": rep.bne_cost / inst.game.n, "informed": rep.informed_ne_cost / inst.game.n,
            "voi": rep.value_of_information, "per_state": rep.per_state}


def page_data() -> None:
    st.header("6 · The mess data (synthetic)")
    intro("8 weeks of QR scans for 4 messes and 3000 students: exactly what the mess office already records.",
          "A scan happens at plate pick-up, so waiting is not observed directly; we still recover the serving rate, "
          "preferred times, and forecast the next 30 minutes.", "pick a mess and a meal.")
    scans, _ = logs()
    c = st.columns(2)
    mess = c[0].selectbox("Mess", list(MESSES), key="dmess")
    meal = c[1].selectbox("Meal", list(MEALS), index=2, key="dmeal")
    H = arrival_heatmap(scans, mess, meal).iloc[:, :14]
    fig = go.Figure(go.Heatmap(z=H.values, x=H.columns, y=H.index, colorscale=SEQ, colorbar=dict(title="scans"),
                               hovertemplate="%{y} %{x}: %{z:.0f} scans<extra></extra>"))
    show(fig_layout(fig, f"{mess} {meal}: average scans per 10 minutes (special menus: Wed & Sun dinner, Sun lunch)",
                    330, hovermode="closest"))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Serving rate from scans**")
        rows = [{"menu": menu, "true": MEALS[meal].service_rate * (0.8 if menu == "special" else 1.0),
                 "estimated": round(estimate_service_rate(scans, mess, meal, menu), 2)} for menu in ("normal", "special")]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("While a queue exists plates leave at the serving rate. No queue: only a lower bound.")
    with c2:
        st.markdown("**Preferred times from scans**")
        acc = preference_accuracy(estimate_preferred_slots(scans, meal), roster_truth().set_index("student_id")[f"tau_{meal}"])
        st.metric("recovered exactly", f"{acc['exact']:.0%}")
        st.metric("within one slot", f"{acc['within_1_slot']:.0%}")
    with c3:
        st.markdown("**Nowcasting the next 30 minutes**")
        rows = []
        for menu in ("normal", "special"):
            table = scans_per_slot(scans, mess, meal, menu)
            rows.append({"menu": menu, "mean scans/slot": round(table.to_numpy().mean(), 1),
                         **{kk: round(v, 2) for kk, v in nowcast_errors(table).items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Mean absolute error per slot; each day is predicted from earlier days only.")
    st.dataframe(scans.head(200), hide_index=True, width="stretch")
    st.download_button("Download all scans (CSV)", scans.to_csv(index=False).encode(), "synthetic_scans.csv", "text/csv")


PAGES = {
    "Start here": page_start,
    "1 · Small games": page_games,
    "2 · Selfish vs planner": page_poa,
    "3 · The display": page_display,
    "4 · Recommendations": page_ce,
    "5 · Learning & uncertainty": page_learning,
    "6 · Mess data": page_data,
}
with st.sidebar:
    st.markdown("### Coordinating crowds")
    st.caption("Foundations of AI · IIT Gandhinagar")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown(f"[Website]({LINKS['website']}) · [Report]({LINKS['report']}) · [Code]({LINKS['code']})")
    st.caption("All data are synthetic. Costs are minutes per person per visit.")
PAGES[page]()
