"""Interactive demo:  streamlit run app.py

Pages follow the story of the project:
  Overview -> toy games -> selfish vs planner -> what the display should show
  -> correlated recommendations -> learning & uncertainty -> mess data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crowd import viz
from crowd.correlated import (bayesian_analysis, best_ce, interpolated_schedules, obedience, obedience_incentive,
                              recommend)
from crowd.domains import DOMAINS, build
from crowd.game import FluidQueue, benchmarks
from crowd.mess import (MEALS, MESSES, arrival_heatmap, estimate_preferred_slots, estimate_service_rate, generate,
                        meal_instance, nowcast_errors, preference_accuracy, roster_truth,
                        scans_per_slot)
from crowd.simulate import EXP3, Coarse, Forecast, Hedge, Live, Logit, Noisy, NoInfo, Partial, Recommend, \
    RegretMatching, simulate
from crowd.theory import dinner_game, hedge, intersection_game, pigou_poa, regret_matching, shared_counter_dynamics

st.set_page_config(page_title="Crowd congestion and correlated equilibrium", layout="wide")

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


def fig_layout(fig: go.Figure, title: str, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(title=title, height=height, **{**viz.PLOTLY_LAYOUT, **kw})
    return fig


def loads_chart(game, schedules: dict[str, tuple[np.ndarray, str]], title: str) -> go.Figure:
    fig = go.Figure()
    for name, (X, color) in schedules.items():
        fig.add_bar(x=game.slot_labels, y=game.loads(X), name=name, marker_color=color)
    cap = game.congestion.capacity
    fig.add_hline(y=cap, line_dash="dash", line_color=viz.MUTED,
                  annotation_text="capacity per slot", annotation_position="top left")
    return fig_layout(fig, title, barmode="group", yaxis_title="arrivals", hovermode="x")


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        col.metric(label, value, help=help_)


# ------------------------------------------------------------------ pages
def page_overview() -> None:
    st.title("Crowds, information and correlated equilibrium")
    st.markdown(
        "**The problem.** Many people share one bottleneck and each chooses *when* to use it: cars on a bridge, "
        "backups on a network link, diners at a food court, students at the hostel mess. Each person wants a "
        "convenient time and a short queue, and every person who joins the queue makes it longer for everyone "
        "behind them. The facility usually has enough capacity *over the whole window*; the trouble is that "
        "everyone comes at once. **It is a coordination problem, not a capacity problem.**")
    st.markdown(
        "**The question.** A display (an app, a screen at the entrance) can tell people about the crowd. "
        "*What should it show, and to whom,* so that people's own choices lead to short waits?")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### What we find")
        st.markdown(
            "1. **Selfish timing wastes a lot.** In every domain the Nash equilibrium costs 3-8x the planner's optimum.\n"
            "2. **More information can make things worse.** A live crowd counter that everyone reacts to makes the "
            "rush jump from slot to slot (the El Farol effect): in a reactive crowd average cost rises about 5x.\n"
            "3. **Private recommendations that people *want* to follow are correlated equilibria.** A recommendation "
            "drawn from a Nash schedule is obeyed by rational people, is stable from day one and removes herding, "
            "even when only half the crowd uses it.\n"
            "4. **Advice alone cannot beat Nash in a large crowd** (we prove it, and our LP confirms it). But the "
            "correlated-equilibrium constraints tell us the *smallest reward* that makes the optimum self-enforcing: "
            "a token worth ~3 minutes for the ~13% sent off-peak cuts the average cost about 6x.")
    with c2:
        st.markdown("#### Concepts from Foundations of AI used")
        st.markdown(
            "- Congestion games, Nash equilibrium, potential games\n"
            "- Price of anarchy (Pigou's network)\n"
            "- **Correlated & coarse correlated equilibrium (linear programming)**\n"
            "- Mediators, obedience, information design\n"
            "- Bayesian games (incomplete information)\n"
            "- Logit / quantal response (bounded rationality)\n"
            "- No-regret learning: Hedge, EXP3, regret matching")
    st.markdown("#### One model, many shared facilities")
    rows = []
    for d in DOMAINS.values():
        _, b = solve_domain(d.key, None)
        rows.append({"facility": d.title, "who": f"{d.n} {d.unit}", "story": d.story,
                     "selfish cost (min)": round(b.nash.social_cost / d.n, 2),
                     "optimum (min)": round(b.optimum.social_cost / d.n, 2), "price of anarchy": round(b.poa, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Cost = minutes per person: waiting in the queue + inconvenience of using a slot away from your "
               "preferred time. All data are synthetic.")


def page_toy() -> None:
    st.header("1 · Three small games that carry the whole story")
    t1, t2, t3, t4 = st.tabs(["Pigou's network", "Two-student dinner game", "Intersection (traffic light)",
                              "Learning finds equilibria"])
    with t1:
        st.markdown("A unit of traffic goes from s to t. The **upper road** always costs 1. The **lower road** costs "
                    "x^p where x is the share of traffic on it. Selfishly everyone takes the lower road (it never "
                    "costs more than 1). A planner sends only part of the traffic there.")
        p = st.slider("steepness p of the lower road's cost (dosa-night queues are steep)", 1.0, 12.0, 1.0, 0.5)
        x_opt = (p + 1) ** (-1 / p)
        metric_row([("selfish cost", "1.000", None), ("optimal cost", f"{1 / pigou_poa(p):.3f}", None),
                    ("price of anarchy", f"{pigou_poa(p):.3f}", "1 = selfishness costs nothing"),
                    ("planner's share on the lower road", f"{x_opt:.0%}", None)])
        ps = np.linspace(1, 12, 100)
        fig = go.Figure(go.Scatter(x=ps, y=[pigou_poa(v) for v in ps], line_color=viz.NE, name="PoA(p)"))
        fig.add_scatter(x=[p], y=[pigou_poa(p)], mode="markers", marker=dict(size=11, color=viz.NE), name="you")
        st.plotly_chart(fig_layout(fig, "Price of anarchy grows with steepness", 300, xaxis_title="p",
                                   showlegend=False), width="stretch")
    with t2:
        st.markdown("Two students choose **Peak** or **Late**. Peak costs the number of students there (1 or 2). "
                    "Late always costs L (no queue, but eating late).")
        L = st.slider("cost of going late, L", 1.05, 1.95, 1.5, 0.05)
        g = dinner_game(L)
        mixed = [m for m in g.mixed_nash_2p() if 0 < m[0][0] < 1][0]
        path = shared_counter_dynamics(g, (1, 1), 20)
        ce = g.correlated_equilibrium("fair")
        table = pd.DataFrame(
            [[f"({g.costs[0, a, b]:g}, {g.costs[1, a, b]:g})" for b in (0, 1)] for a in (0, 1)],
            index=["S1: Peak", "S1: Late"], columns=["S2: Peak", "S2: Late"])
        c1, c2 = st.columns([1, 2])
        c1.markdown("**Costs (student 1, student 2)**")
        c1.dataframe(table)
        res = pd.DataFrame({
            "what students do": ["Shared live counter: both react to yesterday", "No information: mixed Nash",
                                 "Private coin: one told Peak, other Late (CE)"],
            "total cost": [np.mean([g.total_cost(s) for s in path]),
                           g.expected_costs(g.product_distribution(list(mixed))).sum(), ce.total_cost]})
        res["vs optimum"] = (res["total cost"] / g.social_optimum()[1]).round(2)
        c2.dataframe(res.round(3), hide_index=True, width="stretch")
        st.info(f"Mixed Nash: each goes to Peak with probability {mixed[0][0]:.2f}. The correlated equilibrium is "
                "a traffic light: whoever is told *Late* knows the other is at Peak (moving would cost 2 > L); "
                "whoever is told *Peak* knows the other is Late (moving would cost L > 1). Both obey, and each is "
                f"told Peak half the time, so it is fair: expected cost {ce.expected_costs[0]:.2f} each.")
    with t3:
        ig = intersection_game()
        st.markdown("Two drivers reach a crossing: **Stop** or **Go**. Payoffs (higher is better): both stop 6/6; "
                    "one goes 7 vs 2; both go = crash 0/0. Nash equilibria: (Stop, Go), (Go, Stop), and a mixed one "
                    "worth 4.67 each.")
        best = ig.correlated_equilibrium("best")
        c1, c2 = st.columns(2)
        c1.metric("best payoff with any Nash mixture (symmetric)", "4.67")
        c2.metric("best correlated equilibrium", f"{-best.expected_costs[0]:.2f} each")
        st.dataframe(pd.DataFrame({"signal": list(best.support()), "probability": list(best.support().values())}),
                     hide_index=True)
        st.caption("The CE is found by a linear program with one 'obedience' constraint per (player, told action, "
                   "alternative). It never recommends a crash, which no Nash mixture can avoid: correlation creates value.")
    with t4:
        ig = intersection_game()
        rounds = st.select_slider("rounds", [1000, 3000, 6000, 12000], 6000)
        rm = regret_matching(ig, rounds=rounds, seed=0, log_every=25)
        hd = hedge(ig, rounds=rounds, seed=0, log_every=25)
        fig = go.Figure()
        fig.add_scatter(x=rm.rounds_logged, y=rm.ce_violation, name="regret matching -> CE", line_color=viz.CE)
        fig.add_scatter(x=hd.rounds_logged, y=hd.cce_violation, name="Hedge -> coarse CE", line_color=viz.SERIES[3])
        st.plotly_chart(fig_layout(fig, "Gap to equilibrium of the empirical joint play", 320, yaxis_type="log",
                                   xaxis_title="round", yaxis_title="largest gain from deviating"),
                        width="stretch")
        st.markdown("No-swap-regret learners (regret matching) end up at a **correlated equilibrium**; "
                    "no-external-regret learners (Hedge) at a **coarse** correlated equilibrium. A mediator just "
                    "gets there on day one.")


def pick_instance(key_prefix: str):
    mode = st.radio("Facility", ["Mess case study", "Other domain"], horizontal=True, key=key_prefix + "mode")
    if mode == "Mess case study":
        c = st.columns(4)
        mess = c[0].selectbox("Mess", list(MESSES), key=key_prefix + "mess")
        meal = c[1].selectbox("Meal", list(MEALS), index=2, key=key_prefix + "meal")
        day = c[2].selectbox("Day", ["normal", "special", "special_scarce"], key=key_prefix + "day",
                             format_func=lambda d: {"normal": "normal menu", "special": "special menu (dosa night)",
                                                    "special_scarce": "special + item can run out"}[d])
        default_mu = MEALS[meal].service_rate * (1.0 if day == "normal" else 0.8)
        mu = c[3].slider("serving rate (students/min)", 4.0, 14.0, float(default_mu), 0.5, key=key_prefix + "mu")
        return solve_mess(mess, meal, day, None if np.isclose(mu, default_mu) else mu)
    c = st.columns(2)
    key = c[0].selectbox("Domain", [k for k in DOMAINS if k != "mess"], format_func=lambda k: DOMAINS[k].title,
                         key=key_prefix + "dom")
    d = DOMAINS[key]
    mu = c[1].slider(f"service rate ({d.unit}/min)", d.service_rate * 0.5, d.service_rate * 1.6,
                     float(d.service_rate), d.service_rate / 20, key=key_prefix + "dmu")
    st.caption(d.story)
    return solve_domain(key, None if np.isclose(mu, d.service_rate) else mu)


def page_poa() -> None:
    st.header("2 · Selfish timing vs. the planner (price of anarchy)")
    st.markdown("**Nash equilibrium**: nobody can lower their own cost by moving alone (found by best-response "
                "dynamics). **Optimum**: the schedule a planner would impose to minimise total cost (linear "
                "programming + exact local search).")
    inst, b = pick_instance("poa")
    g = inst.game
    ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
    metric_row([("people", f"{g.n}", None),
                ("selfish cost / person", f"{ne['per_person']:.2f} min", f"waiting {ne['wait_per_person']:.2f}"),
                ("optimal cost / person", f"{op['per_person']:.2f} min", f"waiting {op['wait_per_person']:.2f}"),
                ("price of anarchy", f"{b.poa:.2f}" if ne["per_person"] >= 0.1 else "n/a (no congestion)", None),
                ("equilibrium accuracy", f"{60 * b.epsilon:.1f} s", "largest gain any one person could still get")])
    st.plotly_chart(loads_chart(g, {"selfish (Nash)": (b.nash.X, viz.NE), "planner (optimum)": (b.optimum.X, viz.OPT)},
                                "When people arrive"), width="stretch")
    W_ne = g.congestion.wait(g.loads(b.nash.X).astype(float))
    W_op = g.congestion.wait(g.loads(b.optimum.X).astype(float))
    fig = go.Figure()
    fig.add_scatter(x=g.slot_labels, y=W_ne, name="selfish (Nash)", line_color=viz.NE)
    fig.add_scatter(x=g.slot_labels, y=W_op, name="planner (optimum)", line_color=viz.OPT)
    st.plotly_chart(fig_layout(fig, "Average wait of someone arriving in each slot (minutes)", 300),
                    width="stretch")
    st.info("Why is the optimum so much better? Each person who joins a queue delays *everyone behind them* until "
            "the queue clears, but only feels their own wait. The planner feeds arrivals at exactly the serving "
            "rate, so the queue never builds; the price is that some people eat a slot or two away from their "
            "favourite time. (The bumpy Nash profile is real: the slot right after a rush inherits its queue.)")


def run_policy(inst, policy, lam, revise, days, seed=0):
    g = inst.game
    return simulate(g.congestion, inst.tau, inst.beta, policy, Logit(lam, revise), days=days,
                    burn_in=min(50, days // 3), seed=seed)


def make_policy(name: str, inst, b, params: dict):
    g = inst.game
    if name == "pi0 No information":
        return NoInfo()
    if name == "pi1 Exact live":
        return Live()
    if name == "pi2 Delayed":
        return Live(delay=params.get("delay", 3))
    if name == "pi3 Noisy":
        return Noisy(params.get("sigma", 3.0))
    if name == "pi4 Coarse":
        return Coarse(params.get("levels", 3))
    if name == "pi5 Partial":
        return Partial(Live(), params.get("share", 0.3))
    if name == "pi6 Forecast":
        return Forecast(0.3)
    if name == "pi7 CE recommendation":
        ob = obedience(g, b.best_nash.X)
        return Recommend(target=b.best_nash.X, type_of=inst.type_of, compliance=params.get("compliance", 1.0),
                         rational=True, gains=ob.gain_by_cell, best_alt=g.deviation_costs(b.best_nash.X).argmin(2),
                         fallback=Live(), name=name)
    ob = obedience(g, b.optimum.X)
    common = dict(target=b.optimum.X, type_of=inst.type_of, rational=True, gains=ob.gain_by_cell,
                  best_alt=g.deviation_costs(b.optimum.X).argmin(2), compliance=params.get("compliance", 1.0),
                  fallback=Live())
    if name == "Optimum as advice, no reward":
        return Recommend(name=name, **common)
    return Recommend(bonus=ob.gain_by_cell, name=name, **common)


POLICIES = ["pi0 No information", "pi1 Exact live", "pi2 Delayed", "pi3 Noisy", "pi4 Coarse", "pi5 Partial",
            "pi6 Forecast", "pi7 CE recommendation", "Optimum as advice, no reward",
            "pi8 Optimum as advice + obedience reward"]


def page_policies() -> None:
    st.header("3 · What should the display show?")
    st.markdown("The same students choose a slot every day. The display shows them something before they choose; "
                "they respond with a logit rule (better slots are chosen more often, not always) and habits.")
    inst, b = pick_instance("pol")
    g = inst.game
    c = st.columns(3)
    beh = c[0].selectbox("How students behave", list(BEHAVIOURS))
    lam, revise = BEHAVIOURS[beh]
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
    if policy.startswith("pi7") or policy.startswith("Optimum") or policy.startswith("pi8"):
        params["compliance"] = st.slider("share of students using the app (others see the live display)",
                                         0.0, 1.0, 1.0, 0.05)
    res = run_policy(inst, make_policy(policy, inst, b, params), lam, revise, days)
    m = res.metrics
    nash, opt = b.best_nash.social_cost / g.n, b.optimum.social_cost / g.n
    metric_row([("avg cost / person", f"{m['mean_cost']:.2f} min", "after the first third of the days"),
                ("vs Nash", f"{m['mean_cost'] / nash:.2f}x", f"Nash = {nash:.2f} min"),
                ("vs optimum", f"{m['mean_cost'] / opt:.2f}x" if opt > 0 else "n/a", f"optimum = {opt:.2f} min"),
                ("90th-pct wait", f"{m['p90_wait']:.1f} min", None),
                ("crowd swing", f"{m['swing']:.0%}", "share of the crowd that changes slot vs. the previous day"),
                ("regret", f"{m['swap_regret']:.2f} min", "0 = nobody would gain by changing their habit")])
    fig = go.Figure(go.Heatmap(z=res.loads.T, x=np.arange(days), y=g.slot_labels,
                               colorscale=[[0, "#f4f8fd"]] + [[(i + 1) / 7, c] for i, c in enumerate(viz.SEQ_BLUE)],
                               colorbar=dict(title="arrivals"), hovertemplate="day %{x}<br>%{y}: %{z} arrivals<extra></extra>"))
    st.plotly_chart(fig_layout(fig, "Arrivals per slot, day by day (stripes = the rush jumping around)", 380,
                               xaxis_title="day", hovermode="closest"), width="stretch")
    daily = np.nanmean(res.cost, axis=1)
    fig = go.Figure(go.Scatter(x=np.arange(days), y=daily, line_color=viz.CE, name="this policy"))
    fig.add_hline(y=nash, line_dash="dash", line_color=viz.NE, annotation_text="Nash")
    fig.add_hline(y=opt, line_dash="dash", line_color=viz.OPT, annotation_text="optimum")
    st.plotly_chart(fig_layout(fig, "Average cost per student each day", 280, yaxis_title="minutes",
                               showlegend=False), width="stretch")
    if st.button("Compare all policies for this behaviour (10 seeds)"):
        rows = []
        prog = st.progress(0.0)
        for i, name in enumerate(POLICIES):
            ms = [run_policy(inst, make_policy(name, inst, b, {}), lam, revise, 150, seed=s).metrics for s in range(10)]
            rows.append({"policy": name, "avg cost (min)": np.mean([x["mean_cost"] for x in ms]),
                         "90th-pct wait": np.mean([x["p90_wait"] for x in ms]),
                         "crowd swing": np.mean([x["swing"] for x in ms]),
                         "regret (min)": np.mean([x["swap_regret"] for x in ms])})
            prog.progress((i + 1) / len(POLICIES))
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")


def page_ce() -> None:
    st.header("4 · Correlated recommendations: advice people want to follow")
    st.markdown(
        "The app privately tells each student a slot. Inside each group of similar students the slots are "
        "shuffled every day, so nobody knows what others were told and inconvenient slots rotate. "
        "**A recommendation is a correlated equilibrium (CE) when no student can gain by disobeying, given "
        "that the others obey.**")
    inst, b = pick_instance("ce")
    g = inst.game
    n = g.n
    ob_ne = obedience(g, b.best_nash.X, tol=b.epsilon + 1e-9)
    ob_opt = obedience(g, b.optimum.X)
    inc = obedience_incentive(g, b.optimum.X, b.best_nash.social_cost)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Shuffle a Nash schedule")
        st.metric("cost / student", f"{b.best_nash.social_cost / n:.2f} min")
        st.metric("largest gain from disobeying", f"{60 * ob_ne.max_gain:.1f} s")
        st.success("Obeyed by rational students: a correlated equilibrium.")
    with c2:
        st.markdown("##### Shuffle the optimum")
        st.metric("cost / student if obeyed", f"{b.optimum.social_cost / n:.2f} min")
        st.metric("students tempted to disobey", f"{ob_opt.share_tempted:.0%}",
                  f"up to {ob_opt.max_gain:.1f} min gain")
        st.error("Not a CE: those sent off-peak would rather go to the peak.")
    with c3:
        st.markdown("##### Optimum + obedience reward")
        st.metric("average reward needed", f"{inc.per_person:.2f} min / student",
                  f"{inc.share_rewarded:.0%} of students get a reward")
        st.metric("time saved vs Nash", f"{inc.welfare_gain / n:.2f} min / student")
        st.success("With the reward (e.g. a priority token) obeying is a best response again.")

    with st.expander("Can a cleverer correlated equilibrium beat Nash without rewards? (LP check)"):
        if st.button("Solve the best-CE linear program over 26 candidate schedules"):
            sched, labels = interpolated_schedules(g, b.best_nash.X, b.optimum.X, steps=20)
            sched += [s.X for s in b.all_nash]
            labels += [f"Nash #{i}" for i in range(len(b.all_nash))]
            ce = best_ce(g, sched, labels, slack=b.epsilon)
            cce = best_ce(g, sched, labels, coarse=True, slack=b.epsilon)
            metric_row([("Nash", f"{b.best_nash.social_cost / n:.3f}", None),
                        ("best CE", f"{ce.expected_cost / n:.3f}", None),
                        ("best coarse CE", f"{cce.expected_cost / n:.3f}", None),
                        ("optimum", f"{b.optimum.social_cost / n:.3f}", None)])
            st.write("Best CE puts its weight on:", ce.support())
        st.markdown("Theory (DESIGN.md, Proposition 2): in a large population with monotone congestion costs, "
                    "*every* (coarse) correlated equilibrium has the same total cost as the Nash equilibrium. "
                    "Advice coordinates; it cannot make people accept a worse slot for others' benefit.")

    st.markdown("#### A student's phone")
    c1, c2 = st.columns([1, 2])
    day = c1.number_input("day", 1, 365, 1)
    who = c1.selectbox("student", list(range(min(n, 400))), format_func=lambda i: inst.ids[i] if hasattr(inst, "ids")
                       else f"person {i}")
    which = c1.radio("schedule the app uses", ["Nash (CE)", "Optimum, no reward", "Optimum + reward"])
    X = b.best_nash.X if which.startswith("Nash") else b.optimum.X
    rec = recommend(np.random.default_rng(day), inst.type_of, X)
    k, t = inst.type_of[who], rec[who]
    D = g.deviation_costs(X)[k, t]  # cost of each slot for this student if everyone else obeys
    bonus = ob_opt.gain_by_cell[k, t] if which == "Optimum + reward" else 0.0
    shown = D.copy()
    shown[t] -= bonus
    c1.markdown(f"**Your slot today: {g.slot_labels[t]}**  \nyour favourite: {g.slot_labels[g.tau[k]]}")
    if bonus > 0:
        c1.markdown(f"Reward for coming at your slot: **{bonus:.1f} min** of priority")
    colors = [viz.CE if s == t else viz.MUTED for s in range(g.T)]
    fig = go.Figure(go.Bar(x=g.slot_labels, y=shown, marker_color=colors,
                           hovertemplate="%{x}: %{y:.2f} min<extra></extra>"))
    best = int(np.argmin(shown))
    verdict = "obeying is your best choice" if shown[t] <= shown.min() + max(b.epsilon, 1e-9) + 1e-9 else \
        f"you would rather go at {g.slot_labels[best]}"
    c2.plotly_chart(fig_layout(fig, f"Your cost in each slot if everyone else follows the app: {verdict}", 330,
                               yaxis_title="minutes", showlegend=False), width="stretch")


def page_learning() -> None:
    st.header("5 · Learning and uncertainty")
    t1, t2 = st.tabs(["Learning over a semester", "Complete vs incomplete information"])
    inst, b = solve_mess("Jaiswal", "dinner", "normal", None)
    g = inst.game
    with t1:
        st.markdown("Instead of reacting to the display, students *learn*. **Hedge** and **regret matching** need "
                    "the app to publish every slot's wait after each meal (full information); **EXP3** only knows "
                    "the student's own wait (no app).")
        days = st.slider("days", 100, 400, 250, 50)
        fig1, fig2 = go.Figure(), go.Figure()
        rows = []
        for (name, beh), c in zip({"Hedge (app publishes waits)": Hedge(0.05),
                                   "Regret matching (app publishes waits)": RegretMatching(),
                                   "EXP3 (own wait only)": EXP3(0.05)}.items(), [viz.SERIES[3], viz.CE, viz.SERIES[4]]):
            r = simulate(g.congestion, inst.tau, inst.beta, Live(), beh, days=days, burn_in=days // 3, seed=0)
            daily = pd.Series(np.nanmean(r.cost, axis=1)).rolling(7, min_periods=1).mean()
            fig1.add_scatter(y=daily, name=name, line_color=c)
            fig2.add_scatter(y=r.swap_regret_curve, name=name, line_color=c)
            rows.append({"learner": name, **{k: round(r.metrics[k], 3) for k in ("mean_cost", "swing", "swap_regret")}})
        fig1.add_hline(y=b.best_nash.social_cost / g.n, line_dash="dash", line_color=viz.NE, annotation_text="Nash")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_layout(fig1, "Cost per student (7-day average)", 330, yaxis_type="log",
                                   xaxis_title="day"), width="stretch")
        c2.plotly_chart(fig_layout(fig2, "Swap regret (0 = correlated equilibrium)", 330, yaxis_type="log",
                                   xaxis_title="day"), width="stretch")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with t2:
        st.markdown("Some nights the counter is slow (a staff member missing, a slow special dish). Students don't "
                    "know in advance. Without an announcement they play a **Bayesian Nash equilibrium** (one plan "
                    "for all nights); with one, they play the Nash equilibrium of the actual night.")
        c = st.columns(2)
        p = c[0].slider("probability of a slow night", 0.05, 0.6, 0.3, 0.05)
        f = c[1].slider("slow-night serving rate (share of normal)", 0.5, 0.95, 0.7, 0.05)
        mu = g.congestion.service_rate
        rep = bayesian_cached(p, f, mu)
        metric_row([("no announcement (BNE)", f"{rep['bne']:.2f} min", None),
                    ("night announced", f"{rep['informed']:.2f} min", None),
                    ("value of the announcement", f"{rep['voi']:+.2f} min / student", "positive = it helps")])
        st.dataframe(pd.DataFrame(rep["per_state"]).T.rename(columns={
            "bne": "no announcement", "informed_ne": "announced", "uninformed_opt": "planner, no info",
            "informed_opt": "planner, informed"}).round(2), width="stretch")
        st.caption("Notice the normal-night row: the fear of a slow night keeps people spread out, so on normal "
                   "nights *not* knowing is better. Information helps on average but not everywhere.")


@st.cache_data(show_spinner="Solving three games...")
def bayesian_cached(p: float, f: float, mu: float) -> dict:
    inst, _ = solve_mess("Jaiswal", "dinner", "normal", None)
    rep = bayesian_analysis(inst.game, {"normal night": FluidQueue(mu), "slow night": FluidQueue(f * mu)}, [1 - p, p])
    return {"bne": rep.bne_cost / inst.game.n, "informed": rep.informed_ne_cost / inst.game.n,
            "voi": rep.value_of_information, "per_state": rep.per_state}


def page_data() -> None:
    st.header("6 · The mess data (synthetic)")
    st.markdown("4 messes, 3000 students, 3 meals of 2 hours, 8 weeks. Each record is a QR scan at plate pick-up, "
                "exactly what the mess office already collects. Scan time = arrival + wait, so waiting time is "
                "*not* directly observed.")
    scans, truth = logs()
    c = st.columns(2)
    mess = c[0].selectbox("Mess", list(MESSES), key="dmess")
    meal = c[1].selectbox("Meal", list(MEALS), index=2, key="dmeal")
    H = arrival_heatmap(scans, mess, meal).iloc[:, :14]
    fig = go.Figure(go.Heatmap(z=H.values, x=H.columns, y=H.index,
                               colorscale=[[0, "#f4f8fd"]] + [[(i + 1) / 7, c] for i, c in enumerate(viz.SEQ_BLUE)],
                               colorbar=dict(title="scans"), hovertemplate="%{y} %{x}: %{z:.0f} scans<extra></extra>"))
    st.plotly_chart(fig_layout(fig, f"{mess} {meal}: average scans per 10 minutes (Wed & Sun dinners are special)",
                               330, hovermode="closest"), width="stretch")
    st.markdown("#### Learning the unknowns from scans alone")
    rows = []
    for menu in ("normal", "special"):
        true_mu = MEALS[meal].service_rate * (0.8 if menu == "special" else 1.0)
        est = estimate_service_rate(scans, mess, meal, menu)
        rows.append({"menu": menu, "true serving rate": true_mu, "estimated from scans": round(est, 2)})
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame(rows), hide_index=True)
    c1.caption("While a queue exists, plates leave at the serving rate, so the busiest stretch of scans reveals it. "
               "With no queue (e.g. breakfast) the estimate is only a lower bound.")
    acc = preference_accuracy(estimate_preferred_slots(scans, meal), roster_truth().set_index("student_id")[f"tau_{meal}"])
    c2.metric("preferred slot recovered exactly", f"{acc['exact']:.0%}")
    c2.metric("within one slot", f"{acc['within_1_slot']:.0%}")
    st.markdown("#### Nowcasting: the next 30 minutes from tonight's scans so far")
    rows = []
    for menu in ("normal", "special"):
        table = scans_per_slot(scans, mess, meal, menu)
        rows.append({"menu": menu, "mean scans per slot": round(table.to_numpy().mean(), 1),
                     **{k: round(v, 2) for k, v in nowcast_errors(table).items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Mean absolute error in scans per 10-minute slot (rolling origin: each day predicted from earlier "
               "days only). N1 rescales the usual profile by how busy tonight has been so far.")
    st.markdown("#### First rows of the log")
    st.dataframe(scans.head(200), hide_index=True, width="stretch")
    st.download_button("Download all scans (CSV)", scans.to_csv(index=False).encode(), "synthetic_scans.csv",
                       "text/csv")


PAGES = {
    "Overview": page_overview,
    "1 · Toy games": page_toy,
    "2 · Selfish vs planner": page_poa,
    "3 · What should the display show?": page_policies,
    "4 · Correlated recommendations": page_ce,
    "5 · Learning & uncertainty": page_learning,
    "6 · Mess data": page_data,
}
with st.sidebar:
    st.markdown("### Crowd congestion")
    st.caption("Foundations of AI · IIT Gandhinagar")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.caption("All data are synthetic. Costs are minutes per person per visit.")
PAGES[page]()
