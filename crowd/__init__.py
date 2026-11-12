"""crowd: congestion games, information design and correlated equilibrium.

Layers (read in this order):
    theory.py      exact small games: Nash, correlated equilibrium LPs, Pigou, learning, delay stability
    game.py        the slot-choice congestion game for a population: queue model, Nash, optimum, PoA
    correlated.py  correlated recommendations at scale, price of obedience, best-CE LP, Bayesian analysis
    simulate.py    day-to-day simulator: display policies pi0-pi8 and behaviour models
    domains.py     the same model for roads, networks, restaurants, gyms, the mess
    mess.py        IITGN mess case study: scenario, synthetic QR-scan logs, calibration
    viz.py         shared colours for figures and the app
"""
