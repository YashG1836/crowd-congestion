# Design: crowd congestion, information and correlated equilibrium

This document is the complete plan of the project: the problem, the model, the
solution concepts and algorithms, the case study, the experiments and how the
code is organised. Results are in [results/RESULTS.md](results/RESULTS.md);
a short tour is in the [README](README.md).

---

## 1. The problem

Many people share one bottleneck and each one chooses **when** to use it.

| Facility | Players | Bottleneck | What congestion costs |
|---|---|---|---|
| Road | commuters | a bridge or toll plaza | queueing at the bottleneck |
| Network | backup jobs / downloads | a shared link or server | transfer delay |
| Restaurant | diners | food counters | queue for food |
| Gym | members | machines | wait for equipment |
| **Hostel mess** (case study) | students | serving counter | queue for the plate |

Everyone wants a convenient time and a short queue. Every person who joins the
queue delays **everyone behind them**, but feels only their own wait, so people
crowd into the same slots. Usually the facility has enough capacity over the
whole window. The rush is a **coordination** problem, not a capacity problem.

A display (an app, a screen) can inform people. If everyone reacts to the same
number, though, the rush just moves (the El Farol bar problem). **Central
question: what should the display show, how fresh and how accurate should it
be, and who should see it, so that people's own choices lead to short waits?**
Our main tool for the "who sees what" part is the **correlated equilibrium**: a
private recommendation that each person *wants* to follow.

The plan goes from general to specific:

1. **Theory on small games** (exact): what selfishness costs, what correlation buys.
2. **A general slot-congestion model** that covers all the facilities above.
3. **A detailed case study**: the four IITGN messes (synthetic data, since the
   real scan logs are not required to make the point).

---

## 2. Concepts from Foundations of AI and where they appear

| Concept | Used for | Code | Result |
|---|---|---|---|
| Normal-form games, pure & mixed Nash equilibrium | dinner game, intersection | `theory.py` | E0 |
| Congestion games, Rosenthal potential, best-response dynamics | the selfish outcome at scale | `game.py` | E2 |
| Wardrop equilibrium, method of successive averages (fictitious play) | large-population equilibrium | `game.py` | E2 |
| Social optimum (linear programming), price of anarchy / stability, Pigou | what selfishness costs | `theory.py`, `game.py` | E0, E2 |
| **Correlated & coarse correlated equilibrium (LP)** | mediator, traffic light, recommendations | `theory.py`, `correlated.py` | E0, E6 |
| Mediators, obedience constraints, information design | the "smart display" | `correlated.py`, `simulate.py` | E3, E6 |
| Incentive compatibility (price of obedience) | making the optimum self-enforcing | `correlated.py` | E6 |
| Bayesian games, Bayesian Nash equilibrium, value of information | slow nights nobody announces | `correlated.py` | E8 |
| Bounded rationality: logit / quantal response | how students react | `simulate.py` | E3-E5 |
| No-regret learning: Hedge, EXP3; regret matching (no swap regret) | learning to an equilibrium | `theory.py`, `simulate.py` | E0, E7 |
| Linear dynamical systems / stability | why delayed displays oscillate | `theory.py` | E0, E5 |
| Estimation from data (learning unknown parameters), forecasting | serving rate, preferences, nowcasts from scans | `mess.py` | E1, E9 |

---

## 3. Layer 1: three small games (exact)

**Pigou's network (roads).** A unit of traffic goes from s to t. The upper road
costs 1; the lower costs x^p when a share x uses it. Selfishly everyone takes the
lower road (cost 1). The planner sends x* = (p+1)^(-1/p) there, so

    PoA(p) = [1 - p (p+1)^(-(p+1)/p)]^(-1),   PoA(1) = 4/3,  PoA(4) = 2.15.

Steep congestion (a full queue on a dosa night) wastes more.

**Two-student dinner game (restaurant / mess).** Peak costs the number of
students at Peak (1 or 2); Late always costs 1.5.

| | Peak | Late |
|---|---|---|
| **Peak** | (2, 2) | (1, 1.5) |
| **Late** | (1.5, 1) | (1.5, 1.5) |

The pure equilibria (P,L), (L,P) are optimal (total 2.5), but identical students
cannot agree on who goes late. The mixed equilibrium (each goes Peak with
probability 1/2) costs 3.0. A shared counter that both react to makes them
alternate between (P,P) and (L,L): 3.5. A fair private coin ("you Peak, you
Late") is a **correlated equilibrium**: whoever is told Late knows the other is at
Peak (switching costs 2 > 1.5), and whoever is told Peak knows the other is Late
(switching costs 1.5 > 1). Cost 2.5, the optimum, and fair (1.25 each).
*Shared information can be worse than none; private coordinated advice is best.*

**Intersection / Chicken (traffic light).** Stop/Go with utilities 6/6, 2/7, 7/2,
0/0 (crash). The best mixture of Nash outcomes gives 4.67 each; the best CE (LP)
puts 1/2 on (Stop, Stop) and 1/4 on each (Stop, Go) and (Go, Stop), giving
**5.25 each**: correlation can beat *every* Nash equilibrium.

**CE and CCE as linear programs.** For a joint distribution p over profiles s:

    CE :  for every player i, told action a, alternative a':
          sum_{s: s_i = a} p(s) [ c_i(s) - c_i(a', s_-i) ] <= 0
    CCE:  for every player i and fixed alternative a':
          sum_s p(s) [ c_i(s) - c_i(a', s_-i) ] <= 0

with p >= 0 and sum p = 1. Best CE = minimise expected total cost; a "fair" CE
minimises the largest player's cost. NE ⊂ CE ⊂ CCE.

**Learning.** Hart & Mas-Colell regret matching (no swap regret) makes the
empirical joint play converge to the **CE** set; Hedge (no external regret) to
the **CCE** set. A mediator gets there on day one.

**Delayed feedback.** With deviation y_t from the balanced crowd and a display
that is d steps old, y_{t+1} = y_t - k y_{t-d}. The crowd settles iff
0 < k < k*(d) = 2 sin(π/(4d+2)) (k*(0)=2, k*(1)=1, k*(2)=0.618...). Older
information tolerates less reaction.

---

## 4. Layer 2: the general slot-congestion game

### 4.1 Model

* **Slots** t = 0..T-1 of length Δ minutes (mess: T = 12 slots of 10 minutes = 2 hours).
* **People** i = 1..n with a private **preferred slot** τ_i and **flexibility**
  β_i (minutes of waiting they would accept to avoid moving one slot).
* **Types.** People with equal (τ, β) are interchangeable, so the state is a
  matrix X[k, t] = number of type-k people in slot t; loads A_t = Σ_k X[k, t].
* **Cost** of a type-k person in slot t:

      c_k(t; A) = g_t(A) + β_k |t - τ_k|
                  crowd     schedule inconvenience

* **Congestion model g** (`game.py`):
  * `FluidQueue` (main): one service point with rate μ people/minute; arrivals
    spread evenly inside each slot; first come, first served; leftover queue
    carries into the next slot. With backlog q₀ at the start of a slot and net
    rate r = (A_t − μΔ)/Δ, the queue after u minutes is max(0, q₀ + r u) and a
    person arriving then waits that queue divided by μ. Averaging over the slot:
    - r ≥ 0: W_t = (q₀ + (A_t − μΔ)/2)/μ
    - r < 0: the queue drains after s = min(Δ, q₀/(−r)); W_t = (q₀ s + r s²/2)/(Δ μ)

    and the backlog passed on is max(0, q₀ + A_t − μΔ). Total waiting Σ A_t W_t
    equals the area under the queue curve (Little's law, tested). This is
    Vickrey's bottleneck model for roads and a deterministic link buffer for networks.
  * Optional **scarcity**: a special item with limited stock; people served after
    it runs out pay a penalty. (Off by default: it makes the game "lumpy", see §9.)
  * `Affine`: g_t = a_t A_t + b_t, the textbook separable game (Pigou = T = 2).
  * `Expected`: Σ_w P(w) g_w(A), the expected cost under an unknown state.
* **Social cost** SC(X) = Σ_t A_t g_t(A) + Σ_{k,t} X[k,t] β_k |t − τ_k|.
* The game is **atomic**: a person moving from t to t' changes the loads to
  A − e_t + e_t'. Every equilibrium check uses exact one-person moves.

### 4.2 Algorithms

| Quantity | Method | Guarantee / check |
|---|---|---|
| Nash equilibrium | best-response dynamics (largest gain first), started from: everyone at τ; the rounded Wardrop solution; random; the optimum | exact NE when no move gains; queue carry-over has no potential, so we detect cycles and return the state with the smallest remaining gain, an **ε-Nash equilibrium**. The worst ε in all 24 mess instances is **14 seconds**. |
| Wardrop (large-population) equilibrium | method of successive averages, x ← x + (BR(x) − x)/(r+1) | average excess cost (gap) < 0.05 min (tested) |
| Social optimum | LP relaxation on a 1-minute grid (queue lengths as variables, Lindley's recursion as constraints), rounded, then exact local search with one-person moves and two-person swaps; also local search from the NE | never worse than the NE; no improving move remains (tested) |
| Price of anarchy / stability | PoA = SC(worst NE found)/SC(opt), PoS = SC(best NE found)/SC(opt), comparing only equilibria with ε within 6 s of the best | Pigou in this framework gives 4/3 (tested) |

### 4.3 Correlated equilibrium at population scale (`correlated.py`)

The mediator (the app) picks a schedule X and, **inside each type, shuffles who
gets which slot**, then tells each person only their own slot. A type-k person is
told slot t with probability X[k,t]/n_k. The recommendation is a CE iff for every
type k, told slot t, and alternative t':
E[c_k(t) − c_k(t') | told t] ≤ 0 (with the atomic move A − e_t + e_t').

**Proposition 1 (shuffled Nash is a CE).** If X is a Nash equilibrium, the
shuffled recommendation is a CE. *Proof:* every realised assignment has the
same type-level schedule X, which is an NE; a person told t knows the loads are
exactly A(X) and that t is a best response for their type. ∎ It is the traffic
light of the mess: it coordinates identical people, it is stable from the first
day, and the shuffle rotates the inconvenient slots.

**Proposition 2 (advice alone cannot beat Nash in a large crowd).** In the
non-atomic limit with monotone congestion costs, every coarse correlated
equilibrium (and so every CE) has the same social cost as the Nash (Wardrop)
equilibrium. *Proof sketch:* let a CCE draw flows x^ω with probability p_ω. A
person of type k may deviate to any fixed slot, so summing the CCE inequalities
with weights given by the Wardrop flow x* gives Σ_ω p_ω ⟨x^ω − x*, c(x^ω)⟩ ≤ 0.
Each term splits as ⟨x^ω − x*, c(x^ω) − c(x*)⟩ (≥ 0 by monotonicity) plus
⟨x^ω − x*, c(x*)⟩ (≥ 0 since x* solves the variational inequality). So both are
0 for every ω in the support, the loads equal the Wardrop loads, and the cost
equals the NE cost. ∎ Our queue has carry-over, so the theorem does not apply
exactly. The LP below checks it numerically: **the best CE over 26 candidate
schedules costs exactly the Nash cost** (E6).

**Best CE over a menu of schedules (LP).** Given candidate schedules X_1..X_J
(the NE, the optimum, 20 schedules in between, other NE), choose probabilities p_j
to minimise Σ p_j SC(X_j) subject to, for every (k, t, t'),
Σ_j p_j X_j[k,t] (C_j[k,t] − D_j[k,t,t'] − ε) ≤ 0. The CCE version sums over t.

**Proposition 3 (price of obedience).** The optimum X° is not a CE: someone told
an off-peak slot gains G[k,t] = max_t' (C[k,t] − D[k,t,t']) by moving. A reward
b[k,t] = G[k,t] for showing up in the recommended slot (e.g. a priority token)
makes obeying a best response, so the rewarded recommendation is a CE of the
rewarded game. The total reward is Σ X°[k,t] G[k,t]; it is a transfer, not a
waste. (Tested: after the reward no one gains by moving.)

### 4.4 Incomplete information (`correlated.py`, `bayesian_analysis`)

Some nights are slow (state w with prior P(w), e.g. the counter serves 70% of
normal). Without an announcement the strategy cannot depend on w, so a
risk-neutral person minimises expected cost: the **Bayesian Nash equilibrium** is
the NE of the game with g = Σ_w P(w) g_w. With an announcement (or a mediator
who knows w, a Bayes correlated equilibrium) people play the NE of the true
night. Value of information = E[SC | BNE] − E[SC | informed NE]. (Uncertainty
about *other people's* preferences barely matters in a large crowd, because the
law of large numbers makes the loads predictable. Uncertainty about the *night*
does matter.)

---

## 5. Layer 3: the day-to-day simulator (`simulate.py`)

The same people choose a slot every day (one simulated day = one meal).

1. The policy shows each person an estimate est[i, t] of each slot's crowd cost
   (and maybe a private slot).
2. Each person chooses (behaviour model), loads and costs are computed, people
   update their memory or learning state, and the meal joins the display's history.

**Information policies.**

| | Policy | What people see |
|---|---|---|
| π0 | No information | their own memory of slots they visited (a moving average) |
| π1 | Exact live | the exact crowd cost of every slot at the last meal |
| π2 | Delayed | the same, d meals old |
| π3 | Noisy | last meal + private random error (each person sees a different number, like differential privacy) |
| π4 | Coarse | only K levels (Low / Medium / High); people read a level as its historical average |
| π5 | Partial | a live display seen by only a share of people |
| π6 | Forecast | a smoothed (exponential moving) average of past meals |
| π7 | **CE recommendation** | a private slot from a shuffled Nash schedule |
| – | Optimum as advice | a private slot from the shuffled optimum (not a CE) |
| π8 | **Optimum + obedience reward** | the same, plus the reward of Proposition 3 |

Recommendation users are **rational**: they disobey whenever disobeying saves
more than 6 seconds, given the published policy. Non-users follow their behaviour model.

**Behaviour models.**
* Logit(λ, revise): with probability `revise` a person reconsiders today and
  picks slot t with probability ∝ exp(−λ(est_t + β|t−τ|)); otherwise they keep
  their habit. Three regimes: *calm* (λ=3, revise 0.3), *reactive* (λ=3,
  revise 1), *noisy* (λ=1, revise 0.3).
* Hedge (full information: the app publishes all slots' waits), EXP3 (bandit:
  own wait only), regret matching (no swap regret).

**Metrics** (after a 50-day burn-in, 150 days, 30 seeds, 95% CI): mean cost per
person, mean and 90th-percentile wait, **swing** (share of the crowd that
changes slot versus the previous day), Jain fairness index, external and swap
regret (swap regret 0 = a correlated equilibrium).

---

## 6. Layer 4: the IITGN mess case study (`mess.py`)

**Given:** 4 messes, ~3000 students split about equally (760/750/745/745),
breakfast, lunch and dinner, each about 2 hours.

**Assumed (all in one place, `mess.py`):**

| Item | Value |
|---|---|
| Slots | 12 × 10 min; breakfast 07:30, lunch 12:00, dinner 19:30 |
| Attendance | breakfast 60%, lunch 85%, dinner 90% (each student has their own probability; ±10% whole-day shocks) |
| Preferred times | lunch rush ~13:05 after classes, dinner rush ~20:10, plus smaller groups |
| Flexibility β | 1.5 / 3 / 6 minutes per slot (30% / 45% / 25%) |
| Serving rate | 8 (breakfast), 9 (lunch, dinner) students/min per mess |
| Special menu (Sunday lunch & breakfast, Wednesday & Sunday dinner) | +5 points attendance, counter ×0.8, preferences 10 min earlier and tighter |
| Optional scarcity | portions for 80% of diners; missing out costs 10 minutes |

**Synthetic QR-scan logs.** The status quo (no display, logit students with
memory and habits) is simulated for 8 weeks for every mess, meal and kind of
day. Within a slot arrivals are spread evenly (as in the fluid model), and the
exact queue wait is added, because **a scan happens at plate pick-up, not when
a student joins the queue**. Output: about 394,000 rows of
`date, weekday, meal, mess, menu, student_id, scan_time`, plus a hidden
ground-truth table (slot, arrival, wait) used only to score our estimators.

**Calibration (learning unknowns from scans).**
* Serving rate: while a queue exists, plates leave at rate μ, so the busiest
  5-minute scan rate estimates μ (within 2-3% whenever queues form; only a lower
  bound when they never do, e.g. normal breakfasts).
* Nowcasting (S6): predict the scans of the next three slots from tonight's scans so far with N0 (historical
  mean profile) or N1 (N0 × tonight-so-far / history-so-far), rolling-origin validation.
* Preferred slot: each student's median scan slot on normal days (dinner: 80%
  exact, 97% within one slot; crowding pushes people slightly off their favourite).

---

## 7. Questions, hypotheses and experiments

| | Question | Experiment | Answer (see RESULTS.md) |
|---|---|---|---|
| RQ1 | How much does selfish timing cost? | E2 | Nash wastes 1.2-1.8 min per student at a normal lunch/dinner and 3.6-7.6 min at a special one (PoA 3.9-6.7; normal breakfasts have no queue at all); **132 student-hours per normal day** across the 4 messes. The same 3-8× appears for roads, networks, restaurants, gyms. |
| RQ2 / H1 | Is partial information best? | E3, E4 | *Partly.* For a reactive crowd, showing everyone the exact counter is a disaster (11.6 min vs 2.2 Nash, 76% of students switch slot every day). Showing it to ≤ 50% costs nothing extra, and private noise helps (11.6 → 3.2). Coarse levels do **not** help. For calm crowds no information is best, but only because people hold pessimistic beliefs (positive regret). |
| RQ2 / H2 | Do delays make the crowd swing? | E0, E5 | Yes, for moderate reactivity: with 30% reacting, the swing grows from 2% (delay 1) to 11% (delay 4), matching k*(d) shrinking with d. |
| RQ3 / H3 | Do personal recommendations close the NE-optimum gap? | E6 | **Refined.** CE advice gives exactly the Nash cost, with zero swing and zero regret in every behaviour regime, and removes herding once ≥ 50-70% use it. It cannot beat Nash (Proposition 2, confirmed by the LP). With the obedience reward (Proposition 3: 13% of students get ~2.7 min worth of priority) the optimum becomes self-enforcing: **2.24 → 0.36 min**. |
| – | Does publishing data help learning? | E7 | Yes. Hedge/regret matching with published waits settle near Nash (2.29-2.31 min) with swap regret down to 0.19 / 0.03 min per day (close to a CE); EXP3 with only one's own wait stays at 5 min regret after 300 days. |
| – | Complete vs incomplete information | E8 | Announcing slow nights saves 1.1 min per student on average, but makes *normal* nights worse (0.88 → 2.24): the fear of a slow night was spreading people out. |
| – | Does calibration work? | E1 | μ within 2-3% when queues form; preferences 97% within one slot at dinner. |
| RQ4 | Can we forecast the next 30 minutes from tonight's scans? | E9 | N0 (historical mean) and N1 (scaled by tonight's scans so far) are within 8-14% of the load; N1 helps when an evening starts unusually (normal breakfasts −13%, special dinners −17%), elsewhere they are within 5%. |

Every experiment is one function in `experiments.py`; figures go to
`results/figures/`, all numbers to `results/RESULTS.md`.

---

## 8. Code organisation

```
crowd-congestion/
├── crowd/                 the library, one file per layer
│   ├── theory.py          Layer 1: normal-form games, NE, CE/CCE LPs, Pigou, toy games, learning, delay stability
│   ├── game.py            Layer 2: congestion models, SlotGame, Nash / Wardrop / optimum / PoA
│   ├── correlated.py      Layer 2: recommendations, obedience, price of obedience, best-CE LP, Bayesian analysis
│   ├── simulate.py        Layer 3: policies pi0-pi8, behaviour models, day-to-day simulator
│   ├── domains.py         generic domains (road, network, restaurant, gym, mess) + synthetic populations
│   ├── mess.py            Layer 4: IITGN scenario, synthetic scan logs, calibration, nowcasting
│   └── viz.py             one colour language for figures and the app
├── app.py                 Streamlit demo (7 pages)
├── experiments.py         E0-E9 -> results/
├── results/               figures/*.png and RESULTS.md (generated)
├── tests/test_crowd.py    54 checks of every claim
├── report/                final LaTeX report (main.tex -> main.pdf, 8 pages)
├── DESIGN.md              this document
└── README.md              tour, results, how to run and host
```

Data flow: `mess.py` / `domains.py` build a population → `game.py` builds the
`SlotGame` and solves NE and the optimum → `correlated.py` turns schedules into
recommendations and checks obedience → `simulate.py` runs policies day by day →
`experiments.py` / `app.py` report.

---

## 9. Limitations (honest list)

* **Synthetic data.** Every mess number is synthetic. The conclusions are about
  mechanisms (herding, obedience, the price of obedience), not exact IITGN minutes.
* **Fluid queue.** Arrivals inside a slot are spread evenly, so a perfectly
  smoothed schedule has zero wait. Real arrivals are random, so the optimum's
  wait (and hence PoA) is optimistic. We report minutes lost, not only ratios.
* **Atomic, non-potential game.** With carry-over, pure NE may not exist
  exactly; we report ε-NE (worst 14 s). The scarcity option makes jumps larger
  (ε up to ~2 min), so it is off by default.
* **Behaviour models are assumptions.** We show three regimes and learning
  rules and say which conclusions change across them (E3).
* **Rewards** are modelled as minutes-equivalent transfers; their real design
  (tokens, priority lanes) and budget are future work.
* No live test with students this term.

## 10. Future work

A Telegram bot or web page for one mess (A/B test of the peak-to-average
ratio); dosa tokens or pre-booked slots as a non-monetary Pigouvian toll;
choice *between* messes (a routing game with possible Braess paradoxes);
differentially private displays (our π3 suggests private noise can help);
other campus resources (library, gym, laundry, shuttle); a second scan at the
queue entrance to measure waits directly.

## 11. Work split (from the project brief)

| Member | Package | Files |
|---|---|---|
| Harsh Kumar Jha | data, serving rate and waits, forecasting | `mess.py` (logs, calibration, nowcasting), E1, E9 |
| Yash Goyal | preferences, Bayesian model, Hedge/EXP3, statistics | `correlated.py` (Bayesian), `simulate.py` (learners), E7, E8 |
| Piyush Kumar Keshri | optimum and NE solvers, PoA, Pigou, recommendations | `theory.py`, `game.py`, `correlated.py`, E0, E2, E6 |
| Garv Singhal | simulator, policies, swings, robustness | `simulate.py`, E3-E5 |
