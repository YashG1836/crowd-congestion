"""Interactive lab:  streamlit run app.py

Every page starts with three lines: what you see, why it matters, what to try.
Pages follow the story: start here -> small games -> selfish vs planner -> public display
-> correlated advice -> mess case study -> team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crowd import viz
from crowd.correlated import (best_ce, information_design, interpolated_schedules, large_crowd_check, obedience,
                              obedience_incentive, recommend, who_gains)
from crowd.domains import DOMAINS, build
from crowd.game import benchmarks
from crowd.mess import (MEALS, MESSES, arrival_heatmap, estimate_preferred_slots, estimate_service_rate, generate,
                        meal_instance, nowcast_errors, preference_accuracy, roster_truth, scans_per_slot)
from crowd.simulate import EXP3, Coarse, Forecast, Hedge, Live, Logit, Noisy, NoInfo, Partial, Recommend, \
    RegretMatching, simulate
from crowd.smallcrowd import SmallCrowd
from crowd.theory import dinner_game, hedge, intersection_game, pigou_poa, regret_matching, shared_counter_dynamics

st.set_page_config(page_title="Coordinating Crowds: interactive lab", layout="wide")

LINKS = {"website": "https://yashg1836.github.io/crowd-congestion/",
         "report": "https://github.com/YashG1836/crowd-congestion/blob/main/report/main.pdf",
         "code": "https://github.com/YashG1836/crowd-congestion"}
BEHAVIOURS = {
    "Calm: 30% of people reconsider each day": (3.0, 0.3),
    "Reactive: everyone reconsiders each day": (3.0, 1.0),
    "Noisy: low rationality": (1.0, 0.3),
}
TEAM = [
    ("Harsh Kumar Jha", "24110130", "Artificial Intelligence", "harsh.jha@iitgn.ac.in",
     "Mess case study (synthetic QR-scan data, calibration, forecasting); small crowds solved exactly, noisy users, robustness checks"),
    ("Yash Goyal", "24110399", "Computer Science and Engineering", "yash.g@iitgn.ac.in",
     "Project lead and integration: solvers, queue lemma, information design, learning; report, website and lab"),
    ("Piyush Kumar Keshri", "24110260", "Chemical Engineering", "piyush.keshri@iitgn.ac.in",
     "Equilibria: Nash and optimum solvers, correlated-equilibrium LPs, rewards"),
    ("Garv Singhal", "24110119", "Computer Science and Engineering", "garv.singhal@iitgn.ac.in",
     "Day-to-day simulator, display policies, cross-facility experiments"),
]


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


@st.cache_resource(show_spinner="Comparing what to tell the crowd (exact large-crowd equilibria; one to two minutes)...")
def info_design(_game, key: str, bad_factor: float, p_bad: float):
    """`key` identifies the game for the cache (Streamlit does not hash arguments starting with _)."""
    return information_design(_game, bad_factor=bad_factor, p_bad=p_bad)


# ------------------------------------------------------------------ helpers
def intro(what: str, why: str, try_: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**What you see** · {what}  \n**Why it matters** · {why}" + (f"  \n**Try this** · {try_}" if try_ else ""))


def fig_layout(fig: go.Figure, title: str, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(title=title, height=height, **{**viz.PLOTLY_LAYOUT, **kw})
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    for col, (label, value, help_) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_)


SEQ = [[0, "#f4f8fd"]] + [[(i + 1) / 7, c] for i, c in enumerate(viz.SEQ_BLUE)]


def pick_instance(key: str):
    """Choose a facility (or the mess case study); returns (instance, benchmarks, domain key or None)."""
    mode = st.radio("Facility", ["Any shared facility", "IITGN mess (case study)"], horizontal=True, key=key + "mode")
    if mode == "Any shared facility":
        c = st.columns(2)
        dom = c[0].selectbox("Facility", list(DOMAINS), format_func=lambda k: DOMAINS[k].title, key=key + "dom")
        d = DOMAINS[dom]
        mu = c[1].slider(f"service rate ({d.unit} per minute)", d.service_rate * 0.5, d.service_rate * 1.6,
                         float(d.service_rate), d.service_rate / 20, key=key + "dmu")
        st.caption(d.story)
        inst, b = solve_domain(dom, None if np.isclose(mu, d.service_rate) else mu)
        return inst, b, dom
    c = st.columns(4)
    mess = c[0].selectbox("Mess", list(MESSES), key=key + "mess")
    meal = c[1].selectbox("Meal", list(MEALS), index=2, key=key + "meal")
    day = c[2].selectbox("Day", ["normal", "special"], key=key + "day",
                         format_func=lambda d: {"normal": "normal menu", "special": "special menu (dosa night)"}[d])
    default_mu = MEALS[meal].service_rate * (1.0 if day == "normal" else 0.8)
    mu = c[3].slider("serving rate (students per minute)", 4.0, 14.0, float(default_mu), 0.5, key=key + "mu")
    inst, b = solve_mess(mess, meal, day, None if np.isclose(mu, default_mu) else mu)
    return inst, b, None


def loads_chart(game, schedules: dict[str, tuple[np.ndarray, str]], title: str) -> go.Figure:
    fig = go.Figure()
    for name, (X, color) in schedules.items():
        fig.add_bar(x=game.slot_labels, y=game.loads(X), name=name, marker_color=color)
    fig.add_hline(y=game.congestion.capacity, line_dash="dash", line_color=viz.MUTED,
                  annotation_text="capacity per slot", annotation_position="top left")
    return fig_layout(fig, title, barmode="group", yaxis_title="arrivals", hovermode="x")


def make_policy(name: str, inst, b, params: dict):
    g = inst.game
    simple = {"No information": NoInfo(), "Public display (yesterday)": Live(),
              "Public display, 3 days old": Live(delay=params.get("delay", 3)),
              "Public display + private noise": Noisy(params.get("sigma", 3.0)),
              "Public display in levels": Coarse(params.get("levels", 3)),
              "Display for a share of people": Partial(Live(), params.get("share", 0.3)),
              "Moving-average forecast": Forecast(0.3)}
    if name in simple:
        return simple[name]
    X = b.best_nash.X if name == "CE advice (Nash schedule)" else b.optimum.X
    ob = obedience(g, X)
    common = dict(target=X, type_of=inst.type_of, rational=True, gains=ob.gain_by_cell, best_alt=g.deviation_costs(X).argmin(2),
                  compliance=params.get("compliance", 1.0), fallback=Live(), tolerance=max(0.1, b.epsilon) + 1e-9)
    if name == "CE advice + reward":
        return Recommend(bonus=ob.gain_by_cell, name=name, **common)
    return Recommend(name=name, **common)


POLICIES = ["No information", "Public display (yesterday)", "Display for a share of people", "CE advice (Nash schedule)",
            "Optimum as advice (no reward)", "CE advice + reward", "Public display, 3 days old",
            "Public display + private noise", "Public display in levels", "Moving-average forecast"]


def run_policy(inst, policy, lam, revise, days, seed=0):
    return simulate(inst.game.congestion, inst.tau, inst.beta, policy, Logit(lam, revise), days=days,
                    burn_in=min(50, days // 3), seed=seed)


# ================================================================== pages
def page_start() -> None:
    st.title("Coordinating crowds")
    st.markdown("Many people share one bottleneck and each chooses **when** to use it: commuters at a bridge, backups on "
                "a network link, diners at a food court, students at the hostel mess. Everyone wants a convenient time "
                "and a short queue; every person who joins a queue delays everyone behind them. **It is a coordination "
                "problem, not a capacity problem.** This lab runs every experiment of the project live.")
    c = st.columns(3)
    c[0].link_button("Website (the short story)", LINKS["website"], width="stretch")
    c[1].link_button("Report (PDF)", LINKS["report"], width="stretch")
    c[2].link_button("Source code", LINKS["code"], width="stretch")
    st.markdown("#### The story in four numbers (a hostel-mess dinner, 680 students)")
    inst, b = solve_domain("mess", None)
    g = inst.game
    live = simulate(g.congestion, inst.tau, inst.beta, Live(), Logit(3.0, 1.0), days=150, burn_in=50, seed=0)
    s = st.columns(4)
    s[0].metric("1 · Selfish timing (Nash)", f"{b.best_nash.social_cost / g.n:.2f} min", help="wait + inconvenience per person")
    s[1].metric("2 · Planner's schedule", f"{b.optimum.social_cost / g.n:.2f} min")
    s[2].metric("3 · Everyone follows a public display", f"{live.metrics['mean_cost']:.2f} min",
                help="reactive crowd: the rush jumps from slot to slot")
    s[3].metric("4 · Private CE advice + small reward", f"{b.optimum.social_cost / g.n:.2f} min",
                help="advice people want to follow, made self-enforcing by a reward")
    st.markdown("#### How to use this lab")
    st.markdown(
        "Use the menu on the left, top to bottom:\n"
        "1. **Small games**: the ideas in games you can solve by hand (Nash, correlated equilibrium, learning).\n"
        "2. **Selfish vs planner**: how much selfish timing costs, for any facility.\n"
        "3. **Public display**: what happens day by day under each way of showing the crowd.\n"
        "4. **Correlated advice**: private recommendations, rewards, a student's phone, and who should be told what.\n"
        "5. **Mess case study**: the four IITGN messes and their synthetic QR-scan data.\n"
        "6. **Team**.")
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
          "They show the three effects behind everything else: selfishness wastes, shared information can backfire, "
          "and private correlated advice can beat every Nash equilibrium.", "move the sliders.")
    t1, t2, t3, t5, t4 = st.tabs(["Pigou's two roads", "Two students: Peak or Late", "A traffic light",
                                  "Small crowds, solved exactly", "Learning finds equilibria"])
    with t1:
        st.markdown("Road A costs 1; road B costs x^p when a share x uses it. Selfishly everyone takes B; a planner splits.")
        p = st.slider("steepness p", 1.0, 12.0, 1.0, 0.5)
        metric_row([("price of anarchy", f"{pigou_poa(p):.3f}", "4/3 for linear costs"),
                    ("planner's share on B", f"{(p + 1) ** (-1 / p):.0%}", None)])
        ps = np.linspace(1, 12, 100)
        fig = go.Figure(go.Scatter(x=ps, y=[pigou_poa(v) for v in ps], line_color=viz.NE))
        fig.add_scatter(x=[p], y=[pigou_poa(p)], mode="markers", marker=dict(size=11, color=viz.NE))
        show(fig_layout(fig, "Steeper congestion, bigger waste", 300, xaxis_title="p", showlegend=False))
    with t2:
        L = st.slider("cost of going late, L", 1.05, 1.95, 1.5, 0.05)
        g = dinner_game(L)
        mixed = [m for m in g.mixed_nash_2p() if 0 < m[0][0] < 1][0]
        ce = g.correlated_equilibrium("fair")
        res = pd.DataFrame({"what students do": ["shared counter: both react to yesterday", "no information: mixed Nash",
                                                 "private fair coin: a correlated equilibrium"],
                            "total cost": [np.mean([g.total_cost(s) for s in shared_counter_dynamics(g, (1, 1), 20)]),
                                           g.expected_costs(g.product_distribution(list(mixed))).sum(), ce.total_cost]})
        res["vs optimum"] = (res["total cost"] / g.social_optimum()[1]).round(2)
        st.dataframe(pd.DataFrame([[f"({g.costs[0, a, b]:g}, {g.costs[1, a, b]:g})" for b in (0, 1)] for a in (0, 1)],
                                  index=["S1: Peak", "S1: Late"], columns=["S2: Peak", "S2: Late"]))
        st.dataframe(res.round(3), hide_index=True, width="stretch")
        st.caption(f"Mixed Nash: each goes Peak with probability {mixed[0][0]:.2f}. The coin tells one student Peak and the "
                   "other Late; both obey (switching costs more), and each is told Peak half the time.")
    with t3:
        ig = intersection_game()
        best = ig.correlated_equilibrium("best")
        st.markdown("Two drivers: **Stop** or **Go**. Both stop 6/6, one goes 7 vs 2, both go = crash 0/0 (higher is better).")
        metric_row([("best symmetric Nash mixture", "4.67 each", None),
                    ("best correlated equilibrium (LP)", f"{-best.expected_costs[0]:.2f} each", None)])
        st.dataframe(pd.DataFrame({"traffic light says": list(best.support()), "probability": list(best.support().values())}),
                     hide_index=True)
    with t5:
        st.markdown("n identical people, three 10-minute slots (early / peak / late), everyone prefers the middle one, "
                    "and the counter always needs 80% of its capacity. Every number is exact: a linear program over "
                    "head-counts (who goes where is shuffled).")
        n = st.slider("crowd size n", 2, 40, 10)
        crowd = SmallCrowd(n)
        soc = crowd.social()
        ne = crowd.pure_nash()
        ce, p = crowd.best_ce()
        opt, mo = crowd.optimum()
        metric_row([("best pure Nash", f"{soc[ne].min():.2f} min" if len(ne) else "none exists", None),
                    ("best correlated equilibrium", f"{ce:.2f} min", None),
                    ("planner", f"{opt:.2f} min", f"{tuple(int(v) for v in mo)}"),
                    ("gap closed by correlation", f"{(soc[ne].min() - ce) / max(soc[ne].min() - opt, 1e-9):.0%}"
                     if len(ne) else "n/a", None)])
        st.dataframe(pd.DataFrame([{"advice (early / peak / late)": " / ".join(str(int(v)) for v in crowd.M[j]),
                                    "how often": f"{p[j]:.1%}", "cost per person": round(soc[j], 3)}
                                   for j in np.argsort(-p) if p[j] > 1e-4]), hide_index=True)
        st.caption("At 10 people the advice is 4/5/1 on 99% of days and 8/0/2 on the rest: the rare crowded day makes "
                   "the late slot worth obeying. As n grows the advantage fades (Proposition 2 in the report).")
    with t4:
        ig = intersection_game()
        rounds = st.select_slider("rounds", [1000, 3000, 6000, 12000], 6000)
        rm = regret_matching(ig, rounds=rounds, seed=0, log_every=25)
        hd = hedge(ig, rounds=rounds, seed=0, log_every=25)
        fig = go.Figure()
        fig.add_scatter(x=rm.rounds_logged, y=rm.ce_violation, name="regret matching -> CE", line_color=viz.CE)
        fig.add_scatter(x=hd.rounds_logged, y=hd.cce_violation, name="Hedge -> coarse CE", line_color=viz.SERIES[3])
        show(fig_layout(fig, "Gap to equilibrium of the empirical joint play", 320, yaxis_type="log", xaxis_title="round",
                        yaxis_title="largest gain from deviating"))
        st.caption("Players who move away from actions they regret end up at a correlated equilibrium. A mediator gets "
                   "there on day one.")


def page_poa() -> None:
    st.header("2 · Selfish timing vs the planner")
    intro("Arrivals per slot when everyone picks selfishly (Nash equilibrium) vs the planner's schedule.",
          "Each person in a queue delays everyone behind them but feels only their own wait, so selfish timing wastes time.",
          "pick another facility, or the mess on a special-menu night, and change the service rate.")
    inst, b, _ = pick_instance("poa")
    g = inst.game
    ne, op = g.breakdown(b.nash.X), g.breakdown(b.optimum.X)
    metric_row([("people", f"{g.n}", None),
                ("selfish cost / person", f"{ne['per_person']:.2f} min", f"waiting {ne['wait_per_person']:.2f}"),
                ("planner cost / person", f"{op['per_person']:.2f} min", f"waiting {op['wait_per_person']:.2f}"),
                ("price of anarchy", f"{b.poa:.2f}" if ne["per_person"] >= 0.1 else "n/a (no queue)", None),
                ("equilibrium accuracy", f"{60 * b.epsilon:.1f} s", "largest gain any one person could still get")])
    show(loads_chart(g, {"selfish (Nash)": (b.nash.X, viz.NE), "planner (optimum)": (b.optimum.X, viz.OPT)}, "When people arrive"))
    fig = go.Figure()
    fig.add_scatter(x=g.slot_labels, y=g.congestion.wait(g.loads(b.nash.X).astype(float)), name="selfish", line_color=viz.NE)
    fig.add_scatter(x=g.slot_labels, y=g.congestion.wait(g.loads(b.optimum.X).astype(float)), name="planner", line_color=viz.OPT)
    show(fig_layout(fig, "Wait of someone arriving in each slot (minutes)", 280))
    st.caption("The planner feeds arrivals at exactly the service rate, so no queue builds. The bumpy Nash profile is "
               "real: the slot right after a rush inherits its queue.")


def page_display() -> None:
    st.header("3 · What does a public display do?")
    intro("The same people choose a slot every day; a display shows them something first.",
          "Information that everyone reacts to in the same way can make the rush jump around.",
          "keep 'Reactive', compare 'Public display (yesterday)' with 'CE advice (Nash schedule)'.")
    inst, b, _ = pick_instance("pol")
    g = inst.game
    c = st.columns(3)
    lam, revise = BEHAVIOURS[c[0].selectbox("How people behave", list(BEHAVIOURS), index=1)]
    days = c[1].slider("days simulated", 60, 300, 150, 30)
    policy = c[2].selectbox("Display policy", POLICIES, index=1)
    params = {}
    if policy == "Public display, 3 days old":
        params["delay"] = st.slider("delay (days)", 1, 6, 3)
    if policy == "Public display + private noise":
        params["sigma"] = st.slider("private noise (sd, minutes)", 0.0, 12.0, 3.0, 0.5)
    if policy == "Public display in levels":
        params["levels"] = st.slider("number of levels", 2, 10, 3)
    if policy == "Display for a share of people":
        params["share"] = st.slider("share of people who see the display", 0.0, 1.0, 0.3, 0.05)
    if policy in ("CE advice (Nash schedule)", "Optimum as advice (no reward)", "CE advice + reward"):
        params["compliance"] = st.slider("share of people using the app (others see the display)", 0.0, 1.0, 1.0, 0.05)
    res = run_policy(inst, make_policy(policy, inst, b, params), lam, revise, days)
    m = res.metrics
    nash, opt = b.best_nash.social_cost / g.n, b.optimum.social_cost / g.n
    metric_row([("average cost / person", f"{m['mean_cost']:.2f} min", "after the first third of the days"),
                ("vs Nash", f"{m['mean_cost'] / nash:.2f}x", f"Nash = {nash:.2f} min"),
                ("90th-percentile wait", f"{m['p90_wait']:.1f} min", None),
                ("crowd swing", f"{m['swing']:.0%}", "share of the crowd that changes slot vs the previous day"),
                ("regret", f"{m['swap_regret']:.2f} min", "0 = nobody would gain by changing their habit")])
    fig = go.Figure(go.Heatmap(z=res.loads.T, x=np.arange(days), y=g.slot_labels, colorscale=SEQ, colorbar=dict(title="arrivals"),
                               hovertemplate="day %{x}<br>%{y}: %{z} arrivals<extra></extra>"))
    show(fig_layout(fig, "Arrivals per slot, day by day (stripes = the rush jumping around)", 360, xaxis_title="day",
                    hovermode="closest"))
    fig = go.Figure(go.Scatter(x=np.arange(days), y=np.nanmean(res.cost, axis=1), line_color=viz.CE))
    fig.add_hline(y=nash, line_dash="dash", line_color=viz.NE, annotation_text="Nash")
    fig.add_hline(y=opt, line_dash="dash", line_color=viz.OPT, annotation_text="planner")
    show(fig_layout(fig, "Average cost per person each day", 260, yaxis_title="minutes", showlegend=False))
    if st.button("Compare all policies for this behaviour (10 seeds)"):
        rows, prog = [], st.progress(0.0)
        for i, name in enumerate(POLICIES):
            ms = [run_policy(inst, make_policy(name, inst, b, {}), lam, revise, 150, seed=s).metrics for s in range(10)]
            rows.append({"policy": name, **{k: np.mean([x[k] for x in ms]) for k in ("mean_cost", "p90_wait", "swing", "swap_regret")}})
            prog.progress((i + 1) / len(POLICIES))
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")


def page_advice() -> None:
    st.header("4 · Correlated advice: recommendations people want to follow")
    intro("The app privately tells each person a slot (shuffled every day inside groups of similar people).",
          "Advice is a correlated equilibrium when nobody gains by disobeying; then it is stable without enforcement.",
          "open the phone and switch between the Nash schedule, the planner's optimum and the optimum with a reward.")
    inst, b, dom = pick_instance("ce")
    g = inst.game
    n = g.n
    ob_ne = obedience(g, b.best_nash.X, tol=b.epsilon + 1e-9)
    ob_opt = obedience(g, b.optimum.X)
    inc = obedience_incentive(g, b.optimum.X, b.best_nash.social_cost)
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("##### Shuffle a Nash schedule")
        st.metric("cost / person", f"{b.best_nash.social_cost / n:.2f} min")
        st.metric("largest gain from disobeying", f"{60 * ob_ne.max_gain:.1f} s")
        st.success("Obeyed: a correlated equilibrium.")
    with c2, st.container(border=True):
        st.markdown("##### Shuffle the planner's optimum")
        st.metric("cost / person if obeyed", f"{b.optimum.social_cost / n:.2f} min")
        st.metric("people tempted to disobey", f"{ob_opt.share_tempted:.0%}", f"up to {ob_opt.max_gain:.1f} min gain",
                  delta_color="off")
        st.error("Not a correlated equilibrium: those sent off-peak would rather go at the peak.")
    with c3, st.container(border=True):
        st.markdown("##### Optimum + obedience reward")
        st.metric("average reward", f"{inc.per_person:.2f} min / person", f"{inc.share_rewarded:.0%} get one", delta_color="off")
        st.metric("time saved vs Nash", f"{inc.welfare_gain / n:.2f} min / person")
        st.success("With the reward, obeying is a best response again.")

    st.markdown("#### A phone")
    c1, c2 = st.columns([1, 2])
    which = c1.radio("schedule the app uses", ["Nash (CE)", "Optimum, no reward", "Optimum + reward"])
    day = c1.number_input("day", 1, 365, 1)
    X = b.best_nash.X if which.startswith("Nash") else b.optimum.X
    rec = recommend(np.random.default_rng(day), inst.type_of, X)
    ids = getattr(inst, "ids", None)
    who = c1.selectbox("person", list(range(min(n, 400))), format_func=lambda i: ids[i] if ids is not None else f"person {i}")
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

    t1, t2, t3, t4 = st.tabs(["Can correlation beat Nash? (LP)", "Who should be told what?", "Who gains?",
                              "Learning reaches CE"])
    with t1:
        if st.button("Solve the best correlated equilibrium (linear programs)"):
            sched, labels = interpolated_schedules(g, b.best_nash.X, b.optimum.X, steps=20)
            sched += [s.X for s in b.all_nash]
            atomic = best_ce(g, sched, labels, slack=b.epsilon)
            lc = large_crowd_check(g)
            metric_row([("Nash", f"{b.best_nash.social_cost / n:.3f}", None),
                        ("best CE (atomic, with tolerance)", f"{atomic.expected_cost / n:.3f}", None),
                        ("best CE (large crowd, exact)", f"{lc.best_ce:.3f}", f"large-crowd Nash {lc.nash:.3f}"),
                        ("planner", f"{b.optimum.social_cost / n:.3f}", None)])
        st.caption("In a large crowd no correlated equilibrium beats Nash (report, Section 5.4: the queue is monotone, "
                   "and a search of the remaining candidates finds none). In small crowds it can (see 'Small crowds, "
                   "solved exactly' on page 1). Advice coordinates; a reward is needed to do better.")
    with t2:
        key = f"{g.name}|{g.congestion.service_rate:.4f}|{g.n}"  # this facility or meal, at this service rate
        c = st.columns(2)
        p = c[0].slider("probability of a bad day", 0.1, 0.5, 0.3, 0.05)
        f = c[1].slider("bad-day capacity (share of normal)", 0.5, 0.9, 0.7, 0.05)
        if st.button("Compare: tell nobody, announce, best hint, private advice"):
            r = info_design(g, key, f, p)
            st.dataframe(pd.DataFrame([{"policy": pol, "expected cost": round(v, 3),
                                        "normal day": round(r.per_state[pol]["normal"], 3),
                                        "bad day": round(r.per_state[pol]["bad"], 3)} for pol, v in r.expected.items()]),
                         hide_index=True, width="stretch")
            fig = go.Figure()
            fig.add_scatter(x=r.beliefs, y=r.value, name="cost if everyone believes q", line_color=viz.TEXT_2)
            fig.add_scatter(x=r.beliefs, y=r.envelope, name="best public hint (convex envelope)", line=dict(color=viz.SERIES[3], dash="dash"))
            show(fig_layout(fig, "Cost as a function of the crowd's belief that today is a bad day", 300,
                            xaxis_title="belief q", yaxis_title="minutes"))
            st.caption(f"Best hint: say 'busy' with probability {r.hint['P(busy | bad)']:.2f} on bad days and "
                       f"{r.hint['P(busy | normal)']:.2f} on normal ones. Equilibria accurate to {r.max_gap_sec:.1f} s.")
    with t3:
        w = who_gains(g, b.best_nash.X, b.optimum.X)
        groups = {"flexible (beta 1.5)": g.beta == np.min(g.beta), "rigid (largest beta)": g.beta == np.max(g.beta),
                  "everyone": np.ones(g.K, bool)}
        st.dataframe(pd.DataFrame([{"group": nm, "selfish (Nash)": round(w.group_mean("nash", m), 3),
                                    "optimum, no reward": round(w.group_mean("optimum", m), 3),
                                    "optimum + reward": round(w.group_mean("with_reward", m), 3)}
                                   for nm, m in groups.items() if m.any()]), hide_index=True, width="stretch")
        st.caption(f"Worse off than under selfish timing: {w.worse_off('optimum')} people with the optimum alone (by at "
                   f"most {60 * w.max_loss('optimum'):.1f} s), {w.worse_off('with_reward')} with the reward.")
    with t4:
        days = st.slider("days", 100, 400, 250, 50)
        fig1, fig2, rows = go.Figure(), go.Figure(), []
        for (name, make), col in zip({"Hedge (sees every slot's wait)": lambda: Hedge(0.05),
                                      "Regret matching (sees every slot's wait)": lambda: RegretMatching(),
                                      "EXP3 (sees only its own wait)": lambda: EXP3(0.05)}.items(), [viz.SERIES[3], viz.CE, viz.SERIES[4]]):
            r = simulate(g.congestion, inst.tau, inst.beta, Live(), make(), days=days, burn_in=days // 3, seed=0)
            fig1.add_scatter(y=pd.Series(np.nanmean(r.cost, axis=1)).rolling(7, min_periods=1).mean(), name=name, line_color=col)
            fig2.add_scatter(y=r.swap_regret_curve, name=name, line_color=col)
            rows.append({"learner": name, **{kk: round(r.metrics[kk], 3) for kk in ("mean_cost", "swing", "swap_regret")}})
        fig1.add_hline(y=b.best_nash.social_cost / n, line_dash="dash", line_color=viz.NE, annotation_text="Nash")
        a, b2 = st.columns(2)
        a.plotly_chart(fig_layout(fig1, "Cost per person (7-day average)", 320, yaxis_type="log", xaxis_title="day"), width="stretch")
        b2.plotly_chart(fig_layout(fig2, "Swap regret (0 = correlated equilibrium)", 320, yaxis_type="log", xaxis_title="day"), width="stretch")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def page_mess() -> None:
    st.header("5 · Case study: the IITGN mess")
    intro("Eight weeks of synthetic QR scans for four messes and 3000 students: what the mess office already records.",
          "A scan happens at plate pick-up, so waiting is not observed directly; we still recover the serving rate and "
          "students' preferred times, which is what an advice app needs.", "pick a mess and a meal.")
    scans, _ = logs()
    c = st.columns(2)
    mess = c[0].selectbox("Mess", list(MESSES), key="dmess")
    meal = c[1].selectbox("Meal", list(MEALS), index=2, key="dmeal")
    H = arrival_heatmap(scans, mess, meal).iloc[:, :14]
    fig = go.Figure(go.Heatmap(z=H.values, x=H.columns, y=H.index, colorscale=SEQ, colorbar=dict(title="scans"),
                               hovertemplate="%{y} %{x}: %{z:.0f} scans<extra></extra>"))
    show(fig_layout(fig, f"{mess} {meal}: average scans per 10 minutes (special menus: Wed & Sun dinner, Sun lunch)", 330,
                    hovermode="closest"))
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
        st.markdown("**Forecasting the next 30 minutes**")
        rows = []
        for menu in ("normal", "special"):
            table = scans_per_slot(scans, mess, meal, menu)
            rows.append({"menu": menu, "mean scans/slot": round(table.to_numpy().mean(), 1),
                         **{kk: round(v, 2) for kk, v in nowcast_errors(table).items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Mean absolute error per slot; each day is predicted from earlier days only.")
    st.markdown("#### This meal as a game")
    inst, b = solve_mess(mess, meal, "normal", None)
    g = inst.game
    metric_row([("diners", f"{g.n}", None), ("selfish (Nash)", f"{b.nash.social_cost / g.n:.2f} min", None),
                ("planner", f"{b.optimum.social_cost / g.n:.2f} min", None),
                ("price of anarchy", f"{b.poa:.2f}" if b.nash.social_cost / g.n >= 0.1 else "n/a (no queue)", None)])
    show(loads_chart(g, {"selfish (Nash)": (b.nash.X, viz.NE), "planner (optimum)": (b.optimum.X, viz.OPT)},
                     f"{mess} {meal}, normal day: when students arrive"))
    st.dataframe(scans.head(200), hide_index=True, width="stretch")
    st.download_button("Download all scans (CSV)", scans.to_csv(index=False).encode(), "synthetic_scans.csv", "text/csv")


def page_team() -> None:
    st.header("Team")
    st.caption("Course project, Foundations of AI (Multi-Agent Systems), IIT Gandhinagar")
    cols = st.columns(4)
    for col, (name, roll, branch, email, role) in zip(cols, TEAM):
        with col, st.container(border=True):
            st.markdown(f"#### {name}")
            st.markdown(f"{roll} · {branch}  \n[{email}](mailto:{email})")
            st.caption(role)
    st.markdown("#### Reproduce everything")
    st.code("pip install -r requirements.txt\npytest -q\npython experiments.py\nstreamlit run app.py", language="bash")
    st.caption("Photos on the website: Wikimedia Commons (credits there). All data are synthetic.")


PAGES = {
    "Start here": page_start,
    "1 · Small games": page_games,
    "2 · Selfish vs planner": page_poa,
    "3 · Public display": page_display,
    "4 · Correlated advice": page_advice,
    "5 · Mess case study": page_mess,
    "Team": page_team,
}
with st.sidebar:
    st.markdown("### Coordinating crowds")
    st.caption("Foundations of AI · IIT Gandhinagar")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown(f"[Website]({LINKS['website']}) · [Report]({LINKS['report']}) · [Code]({LINKS['code']})")
    st.caption("All data are synthetic. Costs are minutes per person per visit.")
PAGES[page]()
