# Design: who should be told what? Coordinating crowds with correlated equilibrium

This document is the plan of the project: the problem, the model, the two solution concepts and the
algorithms, the case study, the experiments and how the code is organised. Results are in
[results/RESULTS.md](results/RESULTS.md); the story is told in the [report](report/main.pdf) and on the
[website](https://yashg1836.github.io/crowd-congestion/).

---

## 1. The problem

Many people share one bottleneck and each one chooses **when** to use it.

| Facility | Players | Bottleneck | What congestion costs |
|---|---|---|---|
| Road | commuters | a bridge or toll plaza | queueing at the bottleneck |
| Network | backup jobs | a shared uplink | transfer delay |
| Food court | diners | food counters | queue for food |
| Gym | members | machines | wait for equipment |
| **Hostel mess** (case study) | students | serving counter | queue for the plate |

Everyone wants a convenient time and a short queue. Every person who joins the queue delays **everyone
behind them**, but feels only their own wait, so people crowd into the same slots although the facility has
enough capacity over the whole window. The rush is a **coordination** problem, not a capacity problem.

A display (an app, a screen) can inform people, but if everyone reacts to the same number the rush just
moves (the El Farol bar problem). **Central question: what should be told to whom, so that people's own
choices lead to short queues?**

The project rests on two ideas:

* **Nash equilibrium**: nobody can do better by changing only their own choice. It is where selfish timing
  settles, and it tells us how much time is lost.
* **Correlated equilibrium (CE)**: a mediator privately recommends an action to each person, and nobody
  gains by disobeying given what their own advice reveals. A traffic light is the everyday example. Its
  incomplete-information version, the **Bayes correlated equilibrium**, covers days on which the operator
  knows more than the crowd.

The plan goes from general to specific: crowd congestion in general on five facilities (about two thirds of
the report), then the IITGN mess as a detailed case study with synthetic data.

---

## 2. Small games (exact; `theory.py`)

**Two students, Peak or Late.** Peak costs the number of students at Peak (1 or 2); Late always costs 1.5.

| | Peak | Late |
|---|---|---|
| **Peak** | (2, 2) | (1, 1.5) |
| **Late** | (1.5, 1) | (1.5, 1.5) |

The pure equilibria (P,L), (L,P) are optimal (total 2.5), but identical students cannot agree on who goes
late. The mixed equilibrium (each goes Peak with probability 1/2) costs 3.0. A shared counter that both react
to makes them alternate between (P,P) and (L,L): 3.5. A fair private coin ("you Peak, you Late") is a
**correlated equilibrium**: whoever is told Late knows the other is at Peak (switching costs 2 > 1.5). Cost
2.5, the optimum, and fair. *Shared information can be worse than none; private coordinated advice is best.*

**Traffic light (Chicken).** Stop/Go with utilities 6/6, 2/7, 7/2, 0/0 (crash). The best symmetric Nash
outcome gives 4.67 each; the best CE (linear program) puts 1/2 on (Stop, Stop) and 1/4 on each (Stop, Go),
(Go, Stop), giving **5.25 each**: correlation can beat *every* Nash equilibrium.

**CE as a linear program.** For a joint distribution p over profiles s: for every player i, told action a,
alternative a': `sum_{s: s_i = a} p(s) [c_i(s) - c_i(a', s_-i)] <= 0`, with p >= 0 and sum p = 1. The coarse
CE (CCE) fixes the alternative before the message. NE ⊂ CE ⊂ CCE.

**Learning.** Regret matching (no swap regret) makes the empirical joint play converge to the **CE** set;
Hedge (no external regret) to the **CCE** set. A mediator gets there on day one.

(Pigou's two roads, PoA 4/3 for linear costs, is kept as a check of the solvers and on the website.)

**Small crowds, solved exactly (`smallcrowd.py`).** With n identical people the CE linear program over all T^n profiles
is unnecessary: relabelling who is who maps a CE to a CE of the same cost, and averaging over all relabellings gives a
CE of the form "draw a head-count per slot, then shuffle". So an LP over head-counts (1,891 of them for 60 people and
3 slots) gives the exact best CE; it matches the full LP for n ≤ 6 (tested). Setup: three 10-minute slots, everyone
prefers the middle one (β = 2), and the counter's speed grows with n so the crowd always needs 80% of its capacity.
Result: the best CE beats every pure Nash equilibrium in 52 of the 56 sizes that have one; at 10 people it closes 60%
of the gap (1.56 vs 2.12, planner 1.20) by mixing 4/5/1 (99%) with a rare 8/0/2 that makes "late" worth obeying; at 60
people only 1% of the gap. No pure NE exists at 7, 9, 11 people. The same LP with two states gives the exact Bayes CE
of a bad day, and a lattice version gives its large-crowd limit.

---

## 3. The slot-congestion game (`game.py`)

### 3.1 Model

* **Slots** t = 0..T-1 of length Δ minutes (T = 12).
* **People** i = 1..n with a private **favourite slot** τ_i and **flexibility** β_i (minutes of waiting they
  would accept to avoid moving one slot; three levels).
* **Types.** People with equal (τ, β) are interchangeable, so a schedule is a matrix X[k, t] = number of
  type-k people in slot t (at most 36 types); loads A_t = Σ_k X[k, t].
* **Cost** of a person of type k in slot t: `c_k(t) = W_t(A) + β_k |t − τ_k|` (waiting + inconvenience).
* **Fluid queue** (`FluidQueue`): one server with rate μ people/minute, first come first served, arrivals
  spread evenly inside each slot, someone who finds q people ahead waits q/μ, and the queue left at the end of
  a slot carries over: `Q_{t+1} = max(0, Q_t + A_t − μΔ)`. Total waiting equals the area under the queue
  (Little's law, tested). This is Vickrey's bottleneck model for roads and a link buffer for networks.
  `Expected` mixes several queues (a state that is unknown to the crowd); `Affine` gives textbook linear
  costs (Pigou).
* **Social cost** SC = sum of everyone's cost; reported per person.
* The game is **atomic**: a person moving from t to t' changes the loads to A − e_t + e_t'.

### 3.2 Algorithms

| Quantity | Method | Guarantee / check |
|---|---|---|
| Nash equilibrium | best-response dynamics (largest gain first) from several starts | carry-over removes the potential function, so cycles are detected and the best **ε-equilibrium** is returned (ε ≤ 7 s in the five facilities) |
| Planner's optimum | LP on a 1-minute grid (arrivals and queue lengths as variables, the queue recursion as constraints), one queue chain per state, rounded, then exact local search (moves and swaps) | never worse than the NE; no improving move remains (tested) |
| Large-crowd (Wardrop) equilibrium | projected extragradient on the type-by-slot flows (projection onto each type's simplex), step sizes tried from large to small, warm starts | largest excess cost of a used slot ≤ 0.02 s in the five facilities (tested < 0.6 s) |
| Price of anarchy | SC(worst NE found) / SC(optimum), comparing equilibria within 6 s of the best | Pigou gives 4/3 (tested) |

---

## 4. Correlated advice at scale (`correlated.py`)

**Shuffled recommender.** The app picks a schedule X and, inside each type, shuffles who gets which slot
(every day, so inconvenient slots rotate). A type-k person is told t with probability X[k,t]/n_k.

**Proposition 1 (shuffled Nash is a CE).** If X is a Nash equilibrium, the shuffled recommendation is a CE.
*Proof:* every realised assignment has type-level schedule X; a person told t knows the loads exactly and t is
a best response for their type. ∎

**Proposition 2 (advice alone cannot beat Nash in a large crowd).** With complete information and congestion
costs strictly increasing in the load, every CCE (so every CE) of the large-crowd game has the loads and cost
of the Wardrop equilibrium x*. *Proof sketch:* weighting the CCE inequalities by x* gives
Σ_s p(s) ⟨x^s − x*, c(x^s)⟩ ≤ 0; each term is ⟨x^s − x*, c(x^s) − c(x*)⟩ ≥ 0 (monotone costs) plus
⟨x^s − x*, c(x*)⟩ ≥ 0 (x* is an equilibrium), so both vanish. ∎ Our queue carries over, so we check it
numerically with two LPs:

* `best_ce`: atomic game, randomisation over the Nash schedules, the optimum and 20 schedules in between, with
  obedience `Σ_j p_j X_j[k,t] (C_j[k,t] − D_j[k,t,t'] − ε) ≤ 0`.
* `large_crowd_ce` / `large_crowd_check`: large-crowd game, exact equilibria, the same menu idea.

In all five facilities the best CE over these menus costs exactly the Nash cost.

**Lemma (the queue is monotone).** For any two arrival profiles A, B with the same total,
Σ_t (A_t − B_t)(W_t(A) − W_t(B)) ≥ 0. Proof: in continuous time Q = N − μu + R, where R (idle time) grows only while
Q = 0; then (1/μ)∫(a − b)(Q_A − Q_B)du = (1/μ)[½(Q_A − Q_B)²|end − ∫(Q_A − Q_B)dR_A + ∫(Q_A − Q_B)dR_B] ≥ 0 because
dR_A lives where Q_A = 0 and dR_B where Q_B = 0. Checked on 100,000 random pairs (tested). The queue is monotone but
not strictly, so Proposition 2's proof shows only that every CCE/CE of the large crowd lives on the **best-response
face** (each type only in slots that are optimal for it at Nash; 3–32 dimensions here). `face_ce_check` searches it:
the CE LP over all vertices and 3,000 random face points, then 15 rounds of column generation (the LP's duals define
a pricing problem over the face, solved by projected gradient from many starts), at three obedience tolerances. The
best CE saves at most 0.03 s per person and the saving shrinks tenfold with each tenfold smaller tolerance: numerical
slack, not a better equilibrium.

**Proposition 3 (price of obedience).** For a schedule X let G[k,t] be the most that a type-k person told t can
gain by moving alone. A reward G[k,t] for showing up in the recommended slot makes the shuffled X a CE of the
rewarded game, and no smaller per-slot reward does; the total is Σ X[k,t] G[k,t] (a transfer, not a waste).
`who_gains` checks that with the reward nobody is worse off than under the Nash equilibrium.

---

## 5. Bad days: who should be told what? (`correlated.information_design`)

On a bad day (probability p = 0.3) the bottleneck runs at 70% of its capacity. The operator knows the state;
the crowd knows only p. If everyone believes "bad with probability q", they play the large-crowd equilibrium
x_q of the expected costs, and `V(q) = (1−q) SC_normal(x_q) + q SC_bad(x_q)`.

| Policy | Cost |
|---|---|
| Tell nobody (Bayesian Nash) | V(p) |
| Announce the state | (1−p) V(0) + p V(1) |
| Best public hint (Bayesian persuasion) | lower convex envelope of V at p, over a grid of 21 beliefs; two messages suffice |
| Private advice (best Bayes correlated equilibrium) | LP over 130 schedules (equilibria at every belief, optima, mixtures); obedience on average over the days a message is sent |
| Planner, informed | optimum of each state (needs rewards) |

Every public outcome is a Bayes CE, so private ≤ best hint ≤ min(silence, announce) (tested). Every equilibrium
used is accurate to 5 s or better, and the LP tolerance is at least that accuracy.

**How bad and how often.** The comparison is repeated for bad-day capacity 50%, 60%, 70%, 85% (probability 0.3) and
probability 0.1, 0.3, 0.5 (capacity 70%). **When privacy pays** is measured exactly in small crowds (`small_bad_day`:
a public hint is followed by the best CE of the belief it creates; private advice is the best Bayes CE) and in their
large-crowd limit (`large_crowd_bad_day`: a lattice of splits plus the exact equilibrium of every belief).

---

## 6. The day-to-day simulator (`simulate.py`)

The same people choose a slot every day (150 days, first 50 discarded, 30 seeds, 95% CIs).

| Policy | What people see |
|---|---|
| No information | their own memory of slots they visited |
| Public display | yesterday's wait in every slot (also: d days old, private noise, K levels, moving-average forecast) |
| Display for a share | the display, seen by only a share of people |
| CE advice | a private slot from a shuffled Nash schedule (Prop. 1) |
| Optimum as advice | a private slot from the shuffled optimum (not a CE) |
| CE advice + reward | the optimum plus the reward of Prop. 3 |

App users are **rational**: they disobey whenever disobeying saves more than 6 seconds (or the schedule's own
accuracy). Or **noisy** (`Recommend(noise=λ)`): told r, a user picks t with probability ∝ exp(−λ c(t)), where c(t) is
what t would cost if everyone else obeyed (minus the reward, plus an optional margin). Behaviour: logit choice with habit, `Logit(λ, revise)`; three crowds: calm (λ=3, 30% reconsider
daily), reactive (λ=3, everyone), noisy (λ=1, 30%). Learners: Hedge and regret matching (all waits
published), EXP3 (own wait only). Metrics: mean cost, 90th-percentile wait, **swing** (share of people who
change slot each day), swap regret (0 = a correlated equilibrium).

---

## 6b. Model checks (`game.py`)

* `random_queue_waits` / `random_queue_cost`: keep a schedule's head-counts but draw arrival times uniformly inside
  each slot, serve people one at a time (deterministic or exponential service), and measure waits by simulation
  (200 runs). With evenly spaced arrivals it reproduces the fluid queue to within one service time (checked).
* The planner may keep a buffer: optimum of a counter 5–20% slower, evaluated with random arrivals.
* `booking`: slots reserved one by one in random order, each person taking the free slot nearest their favourite.

## 7. The IITGN mess case study (`mess.py`)

| Item | Value |
|---|---|
| Messes | Jaiswal 760, Mohani 750, Argoras 745, Bhopal 745 students |
| Slots | 12 × 10 min; breakfast 07:30, lunch 12:00, dinner 19:30 |
| Attendance | breakfast 60%, lunch 85%, dinner 90% (personal probabilities; ±10% whole-day shocks) |
| Preferred times | lunch rush ~13:05 after classes, dinner rush ~20:10, plus smaller groups |
| Flexibility β | 1.5 / 3 / 6 minutes per slot (30% / 45% / 25%) |
| Serving rate | 8 (breakfast), 9 (lunch, dinner) students per minute |
| Special menu (Sun breakfast & lunch, Wed & Sun dinner) | +5 points attendance, counter ×0.8, preferences 10 min earlier and tighter |

**Synthetic QR-scan logs.** The status quo (no display; logit students with memory) is simulated for 8 weeks.
A scan happens at plate pick-up, so scan time = arrival + exact queue wait: 393,747 rows of
`date, weekday, meal, mess, menu, student_id, scan_time`, plus a hidden ground-truth table used only to score
the estimators.

**Calibration.** Serving rate = busiest 5-minute scan rate while a queue exists (within 2–3%; only a lower
bound when no queue forms). Preferred slot = each student's median scan slot on normal days. Forecast of the
next 30 minutes from tonight's scans so far: historical profile, or rescaled by tonight-so-far.

---

## 8. Experiments (`experiments.py`)

| | Question | Answer (numbers in RESULTS.md) |
|---|---|---|
| E1 | Small games: when does correlation help? | Dinner game: CE 2.5 = optimum vs mixed Nash 3.0 vs shared counter 3.5. Traffic light: CE 5.25 vs 4.67. Regret matching reaches the CE set. |
| E2 | What does selfish timing cost? | PoA 3.3–8.0 in all five facilities; almost all of it is waiting. With random arrivals 1.3–3.2 (1.7–3.4 with a buffered planner); booking ≈ 2× the planner. |
| E3 | What does a public display do? | Reactive crowd: 3.7–17× Nash, 75–89% switch daily. Display for 30%: 0.82–1.35×. CE advice: exactly Nash, no swing, every crowd. |
| E4 | Correlated advice at scale | Best CE = Nash in every facility (atomic menu; large crowd: face search with column generation). 50–90% app users stop the herding. Reward: 1 minute buys 2.4–6.9 minutes; nobody worse off. Noisy users: CE advice drifts from Nash; a 1–2 minute margin restores the optimum. |
| E5 | Who should be told what on bad days? | Announcing helps 16–49% except on the road (+20%); best hint = vague on the road; private advice adds little in large crowds. On severe days announcing backfires in most facilities. |
| E6 | Do learners find a CE? | Regret matching with published waits: 0.99–1.08× Nash (1.36× gym), swap regret ≤ 0.45 min; EXP3 keeps 2.7–9.0 min of regret. |
| E8 | Small crowds solved exactly | Best CE beats every pure NE in 52/56 sizes (60% of the gap at 10 people), fading with n; privacy on bad days worth several % in small crowds. |
| E7 | The IITGN mess | 393,747 scans; μ within 2–3%; 132 student-hours lost per normal day; Jaiswal dinner: display 11.55, CE 2.24, CE + reward 0.36; who gains; short-staffed evenings. |

Figures go to `results/figures/fig1…fig9`, all numbers to `results/RESULTS.md`, the website's data to
`docs/data.js`, and the report's numbers and tables to `report/generated/` (`numbers.tex`: one macro per number
quoted in the report; `tab_*.tex`: its tables), so the 8-page report is rebuilt exactly from a run.

---

## 9. Code organisation

```
crowd-congestion/
├── crowd/                 the library
│   ├── theory.py          small games: NE, CE/CCE LPs, Pigou, regret matching, Hedge
│   ├── game.py            queue, SlotGame, Nash / large-crowd / optimum solvers, PoA
│   ├── correlated.py      recommender, obedience, best-CE LPs, reward, who gains, information design, face search
│   ├── smallcrowd.py      small crowds of identical people solved exactly (CE, Nash, bad days)
│   ├── simulate.py        policies, behaviour, learners, simulator
│   ├── domains.py         road, network, food court, gym, mess + synthetic populations
│   ├── mess.py            IITGN scenario, synthetic scan logs, calibration, forecasting
│   └── viz.py             one colour language for figures, the app and the website
├── app.py                 Streamlit lab (7 pages), https://crowd-congestion.streamlit.app/
├── docs/                  website (GitHub Pages; one page per step, guided demo), https://yashg1836.github.io/crowd-congestion/
├── experiments.py         E1–E8 -> results/ and docs/data.js
├── results/               figures/*.png and RESULTS.md (generated)
├── tests/test_crowd.py    66 checks of every claim
├── report/                LaTeX report, PRIMEarxiv (main.tex -> main.pdf, 8 pages; generated/ from experiments.py); draft1/
├── DESIGN.md              this document
└── README.md              tour, results, how to run
```

Data flow: `mess.py` / `domains.py` build a population → `game.py` builds the `SlotGame` and solves the NE and
the optimum → `correlated.py` turns schedules into recommendations and checks obedience → `simulate.py` runs
policies day by day → `experiments.py` / `app.py` / `docs/` report.

---

## 10. Limitations

* **Synthetic data.** Conclusions are about mechanisms (herding, obedience, the price of obedience, the value
  of a hint), not exact IITGN minutes.
* **Fluid queue.** Arrivals inside a slot are spread evenly, so a perfectly smoothed schedule has zero wait and
  the PoA is optimistic. We report minutes lost, not only ratios.
* **Atomic, non-potential game.** We report ε-equilibria (≤ 7 s) and use exact large-crowd equilibria where it
  matters (Proposition 2 check, bad days). The face search is near-exhaustive, not a proof; the five-facility
  private-advice LP uses a menu, so its gain is a lower bound. With random arrivals we re-evaluate schedules but do
  not re-solve the equilibria.
* **Behaviour models are assumptions.** Three crowds and three learners; we say which conclusions change.
* **Bad days**: two states, one prior; rewards are minutes-equivalent transfers.

## 11. Future work

A Telegram bot for one mess (A/B test of the peak-to-average ratio); priority tokens or pre-booked slots as a
non-monetary reward; choice *between* messes (a routing game with possible Braess effects); other campus
resources (library, laundry, shuttle); a second scan at the queue entrance to measure waits directly.

## 12. Work split (listed by size of contribution)

| Member | Roll no. | Branch | Package | Files | Experiments |
|---|---|---|---|---|---|
| Yash Goyal | 24110399 | CSE | Lead and integration: model and solvers, large-crowd equilibrium, queue lemma and face search, information design (bad days), learners, booking baseline, experiments pipeline, report, website, lab, repository | `game.py`, `correlated.py`, `experiments.py`, `app.py`, `docs/` | E2, E4, E5, E6 |
| Harsh Kumar Jha | 24110130 | AI | Mess case study (scenario, synthetic scan logs, calibration, forecasting); small crowds solved exactly, noisy app users, random-arrival robustness, bad-day sweeps, paired tests and verification checks | `mess.py`, `smallcrowd.py` | E7, E8 |
| Piyush Kumar Keshri | 24110260 | Chemical | Equilibria and correlated equilibrium: small games, Nash and optimum solvers, CE LPs, recommender, Propositions 1–3, obedience reward, who gains | `theory.py`, `game.py`, `correlated.py` | E1, E2, E4 |
| Garv Singhal | 24110119 | CSE | Simulator and displays: day-to-day simulator, display policies, behaviour models, sweeps, delay and swing, the five facilities | `simulate.py`, `domains.py` | E3, E7 |
