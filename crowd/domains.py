"""Generic congestion domains (road, network, restaurant, gym, mess) and synthetic populations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .game import FluidQueue, SlotGame


# ============================================================================
# Synthetic populations: who wants to use the facility when, and how flexible they are.
# ============================================================================

@dataclass
class Peak:
    """One group of people with a common preferred time."""

    weight: float  # share of people in this group
    mean_min: float  # preferred time, minutes after the window opens
    sd_min: float  # spread of preferred times, minutes


def sample_preferences(rng: np.random.Generator, n: int, peaks: list[Peak], T: int, slot_minutes: float,
                       shift_min: float = 0.0, squeeze: float = 1.0) -> np.ndarray:
    """Preferred slot of each person from a mixture of normal 'peaks'.

    shift_min moves everyone (e.g. -10 on special nights: people come earlier),
    squeeze < 1 concentrates preferences (the rush gets tighter).
    """
    w = np.array([p.weight for p in peaks], dtype=float)
    group = rng.choice(len(peaks), size=n, p=w / w.sum())
    mean = np.array([p.mean_min for p in peaks])[group]
    sd = np.array([p.sd_min for p in peaks])[group]
    minutes = mean + shift_min + squeeze * sd * rng.standard_normal(n)
    return np.clip(np.floor(minutes / slot_minutes), 0, T - 1).astype(int)


def sample_flexibility(rng: np.random.Generator, n: int, levels=(1.5, 3.0, 6.0),
                       probs=(0.30, 0.45, 0.25)) -> np.ndarray:
    """beta: minutes of waiting a person would accept to avoid moving one slot.

    Three levels keep the number of types small (people with equal (tau, beta)
    are interchangeable): flexible, average, rigid.
    """
    return rng.choice(np.asarray(levels, dtype=float), size=n, p=np.asarray(probs) / np.sum(probs))


def clock_labels(start: str, T: int, slot_minutes: float) -> list[str]:
    h, m = map(int, start.split(":"))
    base = h * 60 + m
    return [f"{int((base + t * slot_minutes) // 60) % 24:02d}:{int((base + t * slot_minutes) % 60):02d}"
            for t in range(T)]


# ============================================================================
# Congestion is everywhere: one model, many shared facilities.
#
# Every example below is the same game: n people choose a time slot to use a
# shared bottleneck that serves `service_rate` people (cars, jobs, diners...)
# per minute. Arriving when it is busy means waiting; arriving away from your
# preferred time is inconvenient. Only the numbers change.
#
#     domain       players      bottleneck            slot      congestion cost
#     ---------    ----------   -------------------   -------   -------------------
#     road         commuters    bridge / toll plaza   15 min    queue at the bottleneck
#     network      backup jobs  shared uplink         10 min    transfer queue delay
#     restaurant   diners       food-court counters   10 min    queue for food
#     gym          members      equipment             20 min    wait for a machine
#     mess         students     serving counter       10 min    queue for the plate  (case study)
#
# (The fluid queue is exactly Vickrey's bottleneck model of road traffic and a
# deterministic link buffer in networking, so one congestion model covers all.)
# ============================================================================

@dataclass
class Domain:
    key: str
    title: str
    story: str
    players: str
    bottleneck: str
    n: int
    start: str
    T: int
    slot_minutes: float
    service_rate: float  # people per minute
    peaks: list[Peak]
    beta_levels: tuple[float, float, float] = (1.5, 3.0, 6.0)
    beta_probs: tuple[float, float, float] = (0.30, 0.45, 0.25)
    unit: str = "people"


@dataclass
class Instance:
    domain: Domain
    game: SlotGame
    type_of: np.ndarray  # type of each individual
    tau: np.ndarray  # preferred slot of each individual
    beta: np.ndarray  # flexibility of each individual
    labels: list[str] = field(default_factory=list)


def build(domain: Domain, seed: int = 0, service_rate: float | None = None, n: int | None = None) -> Instance:
    rng = np.random.default_rng(seed)
    n = n or domain.n
    tau = sample_preferences(rng, n, domain.peaks, domain.T, domain.slot_minutes)
    beta = sample_flexibility(rng, n, domain.beta_levels, domain.beta_probs)
    labels = clock_labels(domain.start, domain.T, domain.slot_minutes)
    cm = FluidQueue(service_rate or domain.service_rate, T=domain.T, slot_minutes=domain.slot_minutes)
    game, type_of = SlotGame.from_individuals(cm, tau, beta, labels, name=domain.title)
    return Instance(domain, game, type_of, tau, beta, labels)


DOMAINS: dict[str, Domain] = {
    "road": Domain(
        key="road", title="Morning commute over a bridge",
        story="Office starts at 9:00; everyone wants to cross around 8:30 and the bridge clears 15 cars a minute.",
        players="commuters", bottleneck="bridge", unit="cars",
        n=2400, start="06:30", T=12, slot_minutes=15, service_rate=15,
        peaks=[Peak(0.75, 120, 20), Peak(0.25, 75, 25)], beta_levels=(3.0, 6.0, 12.0)),
    "network": Domain(
        key="network", title="Night-time backups on a shared link",
        story="Laptops are set to back up right after 22:00; the uplink moves 7 backups a minute.",
        players="backup jobs", bottleneck="uplink", unit="jobs",
        n=600, start="22:00", T=12, slot_minutes=10, service_rate=7,
        peaks=[Peak(0.65, 10, 10), Peak(0.35, 50, 25)], beta_levels=(0.5, 1.0, 2.0)),
    "restaurant": Domain(
        key="restaurant", title="Food-court lunch",
        story="Offices break at 13:00; six counters serve 6 diners a minute between them.",
        players="diners", bottleneck="counters", unit="diners",
        n=450, start="12:00", T=12, slot_minutes=10, service_rate=6,
        peaks=[Peak(0.70, 65, 12), Peak(0.30, 30, 15)]),
    "gym": Domain(
        key="gym", title="Evening gym",
        story="Members come after work (about 18:30); machines free up for 1.5 members a minute.",
        players="members", bottleneck="machines", unit="members",
        n=240, start="17:00", T=12, slot_minutes=20, service_rate=1.5,
        peaks=[Peak(0.60, 90, 25), Peak(0.40, 170, 30)], beta_levels=(2.0, 4.0, 8.0)),
    "mess": Domain(
        key="mess", title="Hostel mess dinner (case study)",
        story="About 680 students eat dinner between 19:30 and 21:30; the counter serves 9 a minute.",
        players="students", bottleneck="serving counter", unit="students",
        n=680, start="19:30", T=12, slot_minutes=10, service_rate=9,
        peaks=[Peak(0.70, 40, 15), Peak(0.30, 80, 20)]),
}
