"""crowd: congestion games, information design and correlated equilibrium.

Layers (read in this order):
    theory.py      small games: dominance, Nash, CE/CCE LPs, backward induction, alpha-beta, Pigou, learning
    game.py        the slot-choice congestion game for a population: queue model, Nash, optimum, PoA
    correlated.py  correlated recommendations, price of obedience, best-CE LP, Shapley/core, fair scheme, Bayesian
    simulate.py    day-to-day simulator (policies pi0-pi8; logit, bandit, no-regret students) and the go-or-wait MDP
    domains.py     the same model for roads, networks, restaurants, gyms, the mess
    mess.py        IITGN mess case study: scenario, synthetic QR-scan logs, calibration, nowcasting
    viz.py         shared colours for figures and the app
"""
