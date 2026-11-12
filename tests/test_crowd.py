"""Tests for every layer. Each number quoted in README / DESIGN is checked here.

    pytest -q
"""
import numpy as np
import pytest

from crowd.correlated import (bayesian_analysis, best_ce, interpolated_schedules, obedience, obedience_incentive,
                              recommend)
from crowd.domains import DOMAINS, build
from crowd.game import (Affine, Expected, FluidQueue, SlotGame, benchmarks, local_search, wardrop_equilibrium,
                        wardrop_gap)
from crowd.mess import (MEALS, MESSES, arrival_heatmap, estimate_preferred_slots, estimate_service_rate, generate,
                        make_roster, meal_instance, nowcast_errors, preference_accuracy, roster_truth,
                        scans_per_slot)
from crowd.simulate import (EXP3, Coarse, Forecast, Hedge, Live, Logit, Noisy, NoInfo, Partial, Recommend,
                            RegretMatching, jain_index, simulate)
from crowd.theory import (critical_responsiveness, dinner_game, hedge, intersection_game, numeric_boundary,
                          pigou_numeric, pigou_poa, regret_matching, shared_counter_dynamics, three_student_game)


@pytest.fixture(scope="module")
def dinner():
    """The generic mess-dinner domain with its Nash equilibria and optimum (solved once)."""
    inst = build(DOMAINS["mess"])
    return inst, benchmarks(inst.game)


# ############################################################################
# Layer 1: toy games (theory.py)
# ############################################################################
# ------------------------------------------------------------------ Pigou
def test_pigou_linear_poa_is_four_thirds():
    assert pigou_poa(1) == pytest.approx(4 / 3)


@pytest.mark.parametrize("p", [1, 2, 4, 8])
def test_pigou_formula_matches_numeric(p):
    assert pigou_poa(p) == pytest.approx(pigou_numeric(p)["poa"], rel=1e-6)


def test_pigou_poa_grows_with_steepness():
    vals = [pigou_poa(p) for p in (1, 2, 4, 8, 16)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


# ------------------------------------------------------------------ dinner game
def test_dinner_game_equilibria_and_costs():
    g = dinner_game()
    assert sorted(g.profile_name(s) for s in g.pure_nash()) == ["(Late, Peak)", "(Peak, Late)"]
    mixed = [(x, y) for x, y in g.mixed_nash_2p() if 0 < x[0] < 1]
    assert len(mixed) == 1
    x, y = mixed[0]
    assert x[0] == pytest.approx(0.5) and y[0] == pytest.approx(0.5)
    assert g.expected_costs(g.product_distribution([x, y])).sum() == pytest.approx(3.0)


def test_dinner_best_ce_reaches_optimum_and_is_fair():
    g = dinner_game()
    assert g.correlated_equilibrium("best").total_cost == pytest.approx(2.5)
    assert g.social_optimum()[1] == pytest.approx(2.5)
    fair = g.correlated_equilibrium("fair")
    assert fair.expected_costs == pytest.approx([1.25, 1.25])
    assert fair.support() == pytest.approx({"(Peak, Late)": 0.5, "(Late, Peak)": 0.5})


def test_shared_counter_makes_crowd_alternate():
    g = dinner_game()
    path = shared_counter_dynamics(g, (1, 1), days=20)
    assert set(path) == {(0, 0), (1, 1)}  # both Peak, both Late, ...
    assert np.mean([g.total_cost(s) for s in path]) == pytest.approx(3.5)


def test_every_nash_is_a_correlated_equilibrium():
    for g in (dinner_game(), intersection_game()):
        for x, y in g.mixed_nash_2p():
            p = g.product_distribution([x, y])
            assert g.obedience_violation(p) <= 1e-9
            assert g.cce_violation(p) <= 1e-9


# ------------------------------------------------------------------ intersection (Chicken)
def test_intersection_ce_beats_every_nash_mixture():
    g = intersection_game()
    best = g.correlated_equilibrium("best")
    assert -best.expected_costs == pytest.approx([5.25, 5.25])
    nash_payoffs = [-g.expected_costs(g.product_distribution([x, y])) for x, y in g.mixed_nash_2p()]
    # best symmetric payoff reachable by mixing Nash outcomes is 14/3 < 5.25
    assert max(min(u) for u in nash_payoffs) == pytest.approx(14 / 3)
    assert best.support() == pytest.approx({"(Stop, Stop)": 0.5, "(Stop, Go)": 0.25, "(Go, Stop)": 0.25})


def test_ce_set_inside_cce_set():
    g = intersection_game()
    for obj in ("best", "worst"):
        ce = g.correlated_equilibrium(obj)
        assert g.cce_violation(ce.p) <= 1e-9
    assert g.correlated_equilibrium("best", coarse=True).total_cost <= g.correlated_equilibrium("best").total_cost + 1e-9


def test_three_student_game_ce_equals_optimum():
    g = three_student_game()
    assert g.correlated_equilibrium("best").total_cost == pytest.approx(g.social_optimum()[1])


# ------------------------------------------------------------------ learning
def test_regret_matching_converges_to_ce():
    run = regret_matching(intersection_game(), rounds=20000, seed=1)
    assert run.ce_violation[-1] < 0.02
    assert run.cce_violation[-1] < 0.02  # CE is inside CCE


def test_hedge_converges_to_cce():
    run = hedge(dinner_game(), rounds=20000, seed=1)
    assert run.cce_violation[-1] < 0.05


# ------------------------------------------------------------------ delayed feedback
@pytest.mark.parametrize("d", range(7))
def test_delay_stability_boundary(d):
    assert numeric_boundary(d) == pytest.approx(critical_responsiveness(d), abs=1e-6)


def test_known_boundary_values():
    assert critical_responsiveness(0) == pytest.approx(2.0)
    assert critical_responsiveness(1) == pytest.approx(1.0)
    assert critical_responsiveness(2) == pytest.approx((np.sqrt(5) - 1) / 2)

# ############################################################################
# Layer 2: queue, solvers, correlated equilibrium, Bayesian (game.py, correlated.py)
# ############################################################################
# ------------------------------------------------------------------ queue
def test_no_wait_below_capacity():
    q = FluidQueue(service_rate=5, T=4, slot_minutes=10)
    assert np.allclose(q.wait(np.array([10, 49, 50, 0])), 0)


def test_overload_wait_formula():
    q = FluidQueue(service_rate=5, T=2, slot_minutes=10)
    W = q.wait(np.array([70.0, 0.0]))
    assert W[0] == pytest.approx((70 - 50) / 2 / 5)  # queue grows from 0 to 20: average 10 people -> 2 min
    # second slot: backlog 20 drains at 5/min in 4 min; average over 10 min of queue/mu
    assert W[1] == pytest.approx((20 * 4 - 0.5 * 5 * 16) / (10 * 5))


def test_littles_law_total_wait_equals_queue_area():
    rng = np.random.default_rng(0)
    q = FluidQueue(service_rate=4, T=6, slot_minutes=10)
    A = rng.integers(0, 90, size=6).astype(float)
    total = (A * q.wait(A)).sum()
    # integrate the queue on a fine grid, including the tail after the window
    dt = 0.001
    qq, area = 0.0, 0.0
    for t in np.arange(0, 6 * 10 + A.sum() / 4 + 1, dt):
        rate = A[int(t // 10)] / 10 if t < 60 else 0.0
        qq = max(0.0, qq + (rate - 4) * dt)
        area += qq * dt
    assert total == pytest.approx(area, rel=1e-3)


def test_waits_increase_with_load():
    q = FluidQueue(service_rate=6, T=3, slot_minutes=10)
    base = np.array([80.0, 60.0, 40.0])
    for t in range(3):
        more = base.copy()
        more[t] += 10
        assert np.all(q.wait(more) >= q.wait(base) - 1e-12)


def test_scarcity_share():
    q = FluidQueue(service_rate=100, T=3, slot_minutes=1, scarcity_penalty=10, stock=50)
    miss = q.miss_fraction(np.array([30.0, 40.0, 10.0]))
    assert miss == pytest.approx([0.0, 0.5, 1.0])


def test_expected_model_is_average():
    a, b = FluidQueue(5, T=2), FluidQueue(3, T=2)
    e = Expected([a, b], [0.25, 0.75])
    A = np.array([80.0, 20.0])
    assert e.cost(A) == pytest.approx(0.25 * a.cost(A) + 0.75 * b.cost(A))


# ------------------------------------------------------------------ Pigou inside the slot game
def test_atomic_pigou_poa_close_to_four_thirds():
    n = 400
    game = SlotGame(Affine([1.0 / n, 0.0], [0.0, 1.0]), tau=[0], beta=[0.0], counts=[n])
    b = benchmarks(game)
    assert game.loads(b.nash.X)[0] == n  # selfish: everyone on the variable road
    assert game.loads(b.optimum.X)[0] == n // 2  # planner: half and half
    assert b.poa == pytest.approx(4 / 3, rel=0.01)


# ------------------------------------------------------------------ solvers on a realistic instance
def test_nash_is_epsilon_equilibrium(dinner):
    inst, b = dinner
    mu = inst.game.congestion.service_rate
    for s in b.all_nash:
        assert inst.game.max_regret(s.X) == pytest.approx(s.epsilon, abs=1e-9)
        assert s.epsilon <= 1.0 / mu  # nobody can save more than one person's service time


def test_optimum_beats_nash_and_is_locally_optimal(dinner):
    inst, b = dinner
    assert b.optimum.social_cost <= b.best_nash.social_cost
    again = local_search(inst.game, b.optimum.X)
    assert again.social_cost == pytest.approx(b.optimum.social_cost)
    assert b.poa >= b.pos >= 1.0


def test_wardrop_gap_small(dinner):
    inst, _ = dinner
    x = wardrop_equilibrium(inst.game)
    assert wardrop_gap(inst.game, x) < 0.05


def test_schedules_keep_type_counts(dinner):
    inst, b = dinner
    for X in (b.nash.X, b.optimum.X):
        assert np.array_equal(X.sum(1), inst.game.counts)
        assert X.min() >= 0


# ------------------------------------------------------------------ correlated equilibrium
def test_recommendation_respects_schedule(dinner):
    inst, b = dinner
    rec = recommend(np.random.default_rng(0), inst.type_of, b.nash.X)
    assert np.array_equal(np.bincount(rec, minlength=inst.game.T), inst.game.loads(b.nash.X))
    for k in range(inst.game.K):
        assert np.array_equal(np.bincount(rec[inst.type_of == k], minlength=inst.game.T), b.nash.X[k])


def test_nash_recommendation_is_obeyed_optimum_is_not(dinner):
    inst, b = dinner
    ne = obedience(inst.game, b.nash.X, tol=b.epsilon + 1e-9)
    opt = obedience(inst.game, b.optimum.X)
    assert ne.is_ce
    assert not opt.is_ce and opt.share_tempted > 0.05 and opt.max_gain > 1.0


def test_incentive_makes_optimum_obedient(dinner):
    inst, b = dinner
    inc = obedience_incentive(inst.game, b.optimum.X, b.nash.social_cost)
    g = inst.game
    gains = g.gains(b.optimum.X)  # gain from disobeying
    # after paying bonus[k,t] for showing up in the recommended slot nobody gains
    assert np.all(gains - inc.bonus_by_cell[:, :, None] <= 1e-9)
    assert 0 < inc.per_person < inc.welfare_gain / g.n


def test_best_ce_between_optimum_and_nash(dinner):
    inst, b = dinner
    sched, labels = interpolated_schedules(inst.game, b.best_nash.X, b.optimum.X, steps=10)
    ce = best_ce(inst.game, sched, labels, slack=b.epsilon)
    cce = best_ce(inst.game, sched, labels, coarse=True, slack=b.epsilon)
    assert b.optimum.social_cost - 1e-6 <= cce.expected_cost <= ce.expected_cost + 1e-6
    assert ce.expected_cost <= b.best_nash.social_cost + 1e-6
    assert ce.max_violation <= 1e-6


# ------------------------------------------------------------------ Bayesian
def test_bayesian_single_state_matches_nash(dinner):
    inst, b = dinner
    rep = bayesian_analysis(inst.game, {"only": inst.game.congestion}, [1.0])
    assert rep.value_of_information == pytest.approx(0.0, abs=1e-9)
    assert rep.informed_opt_cost == pytest.approx(b.optimum.social_cost, rel=1e-6)


def test_all_domains_build_and_solve():
    for d in DOMAINS.values():
        inst = build(d)
        b = benchmarks(inst.game)
        assert b.optimum.social_cost <= b.best_nash.social_cost
        assert inst.game.n == d.n

# ############################################################################
# Layer 3-4: simulator and mess case study (simulate.py, mess.py)
# ############################################################################
def test_simulation_is_reproducible(dinner):
    inst, _ = dinner
    a = simulate(inst.game.congestion, inst.tau, inst.beta, Live(), Logit(), days=40, burn_in=10, seed=3)
    b = simulate(inst.game.congestion, inst.tau, inst.beta, Live(), Logit(), days=40, burn_in=10, seed=3)
    assert np.array_equal(a.slots, b.slots)
    assert a.metrics == b.metrics


def test_full_compliance_ce_recommendation_reproduces_nash_every_day(dinner):
    inst, b = dinner
    pol = Recommend(target=b.nash.X, type_of=inst.type_of)
    res = simulate(inst.game.congestion, inst.tau, inst.beta, pol, Logit(), days=30, burn_in=5, seed=0)
    assert np.allclose(res.loads, inst.game.loads(b.nash.X))
    assert res.metrics["mean_cost"] == pytest.approx(b.nash.social_cost / inst.game.n)
    assert res.metrics["swing"] == 0.0


def test_rational_people_abandon_optimum_recommendation(dinner):
    inst, b = dinner
    ob = obedience(inst.game, b.optimum.X)
    alt = inst.game.deviation_costs(b.optimum.X).argmin(axis=2)
    pol = Recommend(target=b.optimum.X, type_of=inst.type_of, rational=True, gains=ob.gain_by_cell, best_alt=alt)
    res = simulate(inst.game.congestion, inst.tau, inst.beta, pol, Logit(), days=20, burn_in=5, seed=0)
    assert res.metrics["mean_cost"] > b.optimum.social_cost / inst.game.n + 1.0


def test_obedience_reward_makes_rational_people_follow_optimum(dinner):
    inst, b = dinner
    ob = obedience(inst.game, b.optimum.X)
    alt = inst.game.deviation_costs(b.optimum.X).argmin(axis=2)
    pol = Recommend(target=b.optimum.X, type_of=inst.type_of, rational=True, gains=ob.gain_by_cell,
                    best_alt=alt, bonus=ob.gain_by_cell)
    res = simulate(inst.game.congestion, inst.tau, inst.beta, pol, Logit(), days=20, burn_in=5, seed=0)
    assert res.metrics["mean_cost"] == pytest.approx(b.optimum.social_cost / inst.game.n)


def test_every_policy_and_behaviour_runs(dinner):
    inst, b = dinner
    cm, tau, beta = inst.game.congestion, inst.tau, inst.beta
    for pol in (NoInfo(), Live(), Live(delay=3), Noisy(3), Coarse(3), Partial(Live(), 0.3), Forecast(0.3)):
        m = simulate(cm, tau, beta, pol, Logit(2.0, 0.3), days=30, burn_in=10, seed=1).metrics
        assert np.isfinite(m["mean_cost"]) and 0 <= m["swing"] <= 1 and 0 < m["fairness"] <= 1
    for beh in (Hedge(), EXP3(), RegretMatching()):
        m = simulate(cm, tau, beta, Live(), beh, days=30, burn_in=10, seed=1).metrics
        assert np.isfinite(m["mean_cost"]) and m["ext_regret"] <= m["swap_regret"] + 1e-9


def test_regret_matching_regret_shrinks(dinner):
    inst, _ = dinner
    res = simulate(inst.game.congestion, inst.tau, inst.beta, Live(), RegretMatching(), days=200, burn_in=50, seed=0)
    assert res.swap_regret_curve[-1] < 0.5 * res.swap_regret_curve[10]


def test_jain_index():
    assert jain_index(np.ones(5)) == pytest.approx(1.0)
    assert jain_index(np.array([1.0, 0, 0, 0])) == pytest.approx(0.25)


# ------------------------------------------------------------------ mess case study
def test_roster_has_3000_students_in_4_messes():
    r = make_roster(0)
    assert len(r.ids) == 3000 == sum(MESSES.values())
    assert set(r.mess) == set(MESSES)
    assert len(set(r.ids)) == 3000


def test_meal_instance_attendance():
    inst = meal_instance("Bhopal", "dinner", "normal")
    assert 0.8 * 745 < inst.game.n < 0.97 * 745


@pytest.fixture(scope="module")
def logs():
    return generate(weeks=2, seed=0)


def test_logs_look_like_scan_records(logs):
    scans, truth = logs
    assert list(scans.columns) == ["date", "weekday", "meal", "mess", "menu", "student_id", "scan_time"]
    assert len(scans) == len(truth) > 10_000
    assert (truth["scan_min"] >= truth["arrival_min"]).all()
    assert scans.groupby(["date", "meal"])["student_id"].apply(lambda s: s.is_unique).all()


def test_service_rate_recovered_on_congested_meals(logs):
    scans, _ = logs
    for meal in ("lunch", "dinner"):
        est = estimate_service_rate(scans, "Jaiswal", meal, "normal")
        assert est == pytest.approx(MEALS[meal].service_rate, rel=0.08)


def test_preferences_recovered(logs):
    scans, _ = logs
    truth = roster_truth().set_index("student_id")
    acc = preference_accuracy(estimate_preferred_slots(scans, "dinner"), truth["tau_dinner"])
    assert acc["within_1_slot"] > 0.9


def test_heatmap_shape(logs):
    scans, _ = logs
    table = arrival_heatmap(scans, "Mohani", "lunch")
    assert table.shape[0] == 7 and table.shape[1] >= 12


def test_nowcasting_runs_and_is_reasonable(logs):
    scans, _ = logs
    table = scans_per_slot(scans, "Jaiswal", "dinner", "normal")
    assert table.shape[1] == MEALS["dinner"].T + 4
    errs = nowcast_errors(table, min_history=2)
    mean_load = table.to_numpy().mean()
    assert all(0 < e < mean_load for e in errs.values())  # better than predicting zero
