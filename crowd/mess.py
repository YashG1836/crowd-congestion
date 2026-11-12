"""Layer 4 - the IITGN mess case study: scenario, synthetic QR-scan logs, calibration."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .domains import Peak, clock_labels, sample_flexibility, sample_preferences
from .game import FluidQueue, SlotGame
from .simulate import Logit, NoInfo, simulate


# ============================================================================
# The IITGN mess case study (all numbers are synthetic but realistic).
#
# Facts we were given
#     4 messes, about 3000 students in total, split roughly equally.
#     Breakfast, lunch and dinner are each served for about 2 hours.
# Assumptions we make (all editable here, in one place)
#     * 12 ten-minute slots per meal.
#     * Attendance: breakfast 60%, lunch 85%, dinner 90% of a mess's students.
#     * Preferred times: a main rush after classes / at the usual dinner time,
#       plus a smaller second group (see MEALS).
#     * Serving rate: about 9 students per minute per mess on a normal day.
#     * Special menu (e.g. dosa night): more students come (+5 points), the
#       counter is slower (x0.8), preferences move 10 minutes earlier and
#       tighten (fear of the item running out), and there are portions for
#       only 80% of diners -- missing out costs 10 minutes-equivalent.
# ============================================================================

MESSES: dict[str, int] = {"Jaiswal": 760, "Mohani": 750, "Argoras": 745, "Bhopal": 745}  # 3000 students
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class Meal:
    name: str
    start: str
    attendance: float
    service_rate: float
    peaks: tuple[Peak, ...]
    special_days: tuple[int, ...] = ()  # weekday numbers (Mon = 0) with a special menu
    T: int = 12
    slot_minutes: float = 10.0

    @property
    def labels(self) -> list[str]:
        return clock_labels(self.start, self.T, self.slot_minutes)


MEALS: dict[str, Meal] = {
    "breakfast": Meal("breakfast", "07:30", 0.60, 8.0,
                      (Peak(0.55, 35, 15), Peak(0.45, 85, 20)), special_days=(6,)),
    "lunch": Meal("lunch", "12:00", 0.85, 9.0,
                  (Peak(0.25, 20, 15), Peak(0.60, 65, 12), Peak(0.15, 95, 15)), special_days=(6,)),
    "dinner": Meal("dinner", "19:30", 0.90, 9.0,
                   (Peak(0.70, 40, 15), Peak(0.30, 80, 20)), special_days=(2, 6)),
}


@dataclass(frozen=True)
class DayType:
    name: str
    extra_attendance: float = 0.0
    service_factor: float = 1.0
    shift_min: float = 0.0
    squeeze: float = 1.0
    stock_share: float | None = None  # portions of the special item / diners
    scarcity_penalty: float = 0.0


DAY_TYPES: dict[str, DayType] = {
    "normal": DayType("normal"),
    "special": DayType("special", extra_attendance=0.05, service_factor=0.8, shift_min=-10, squeeze=0.8),
    "special_scarce": DayType("special_scarce", extra_attendance=0.05, service_factor=0.8, shift_min=-10, squeeze=0.8,
                       stock_share=0.8, scarcity_penalty=10.0),
}


@dataclass
class Student:
    """The roster: one row per student with private, fixed preferences (assumption A3)."""

    mess: np.ndarray  # mess name per student
    ids: np.ndarray  # anonymised id
    beta: np.ndarray  # flexibility (same for all meals)
    tau: dict[str, np.ndarray] = field(default_factory=dict)  # preferred slot per meal (normal day)
    attend: dict[str, np.ndarray] = field(default_factory=dict)  # personal attendance probability per meal


def make_roster(seed: int = 0) -> Student:
    rng = np.random.default_rng(seed)
    mess = np.concatenate([[m] * n for m, n in MESSES.items()])
    n = len(mess)
    ids = np.array([f"S{h:05d}" for h in rng.permutation(n)])
    beta = sample_flexibility(rng, n)
    roster = Student(mess, ids, beta)
    for name, meal in MEALS.items():
        roster.tau[name] = sample_preferences(rng, n, list(meal.peaks), meal.T, meal.slot_minutes)
        a = meal.attendance
        roster.attend[name] = rng.beta(a * 8, (1 - a) * 8, size=n)  # personal habits, mean = a
    return roster


def special_preferences(tau: np.ndarray, meal: Meal, day: DayType, rng: np.random.Generator) -> np.ndarray:
    """Preferred slots on a special day: earlier and tighter around the usual peak."""
    if day.shift_min == 0 and day.squeeze == 1.0:
        return tau
    centre = np.median(tau)
    shifted = centre + day.squeeze * (tau - centre) + day.shift_min / meal.slot_minutes
    shifted = shifted + rng.uniform(-0.5, 0.5, size=len(tau))
    return np.clip(np.round(shifted), 0, meal.T - 1).astype(int)


def congestion_model(meal: Meal, day: DayType, diners: int) -> FluidQueue:
    stock = None if day.stock_share is None else day.stock_share * diners
    return FluidQueue(meal.service_rate * day.service_factor, T=meal.T, slot_minutes=meal.slot_minutes,
                      scarcity_penalty=day.scarcity_penalty, stock=stock)


@dataclass
class MealInstance:
    """One mess, one meal, one kind of day, with the students who come."""

    mess: str
    meal: Meal
    day: DayType
    game: SlotGame
    type_of: np.ndarray
    tau: np.ndarray
    beta: np.ndarray
    ids: np.ndarray


def meal_instance(mess: str = "Jaiswal", meal: str = "dinner", day: str = "normal", seed: int = 0,
                  roster: Student | None = None, service_rate: float | None = None) -> MealInstance:
    """Build the game for a typical day: each student comes with their personal probability."""
    roster = roster or make_roster(seed)
    m, d = MEALS[meal], DAY_TYPES[day]
    rng = np.random.default_rng(seed + 1000 * (list(MEALS).index(meal) + 1) + (day == "special"))
    in_mess = roster.mess == mess
    p = np.clip(roster.attend[meal][in_mess] + d.extra_attendance, 0, 1)
    comes = rng.random(in_mess.sum()) < p
    tau = roster.tau[meal][in_mess][comes]
    tau = special_preferences(tau, m, d, rng)
    beta = roster.beta[in_mess][comes]
    ids = roster.ids[in_mess][comes]
    cm = congestion_model(m, d, len(tau))
    if service_rate is not None:
        cm = FluidQueue(service_rate, cm.T, cm.slot_minutes, cm.scarcity_penalty, cm.stock)
    game, type_of = SlotGame.from_individuals(cm, tau, beta, m.labels, name=f"{mess} {meal} ({day})")
    return MealInstance(mess, m, d, game, type_of, tau, beta, ids)


# ============================================================================
# Synthetic QR-scan logs for a semester (what Student Affairs would hand us).
#
# How the logs are made
#     1. A roster of 3000 students with private preferences (scenario.make_roster).
#     2. For every mess x meal x kind of day, the "status quo" is simulated
#        with the day-to-day simulator: no display, students remember their
#        own past waits and respond with a logit rule (they are not perfectly
#        rational and have habits). Each student comes with their own
#        attendance probability.
#     3. Inside its slot, each arrival time is uniform; the exact fluid-queue
#        wait is added, because a QR scan happens when the plate is picked up
#        (assumption A2), not when the student joins the queue.
#
# Two tables are returned:
#     scans         date, weekday, meal, mess, menu, student_id, scan_time   <- the "real" log
#     ground_truth  the same rows plus slot, arrival time, wait, got_item     <- hidden truth,
#                   used only to check that our estimators recover it.
# ============================================================================

SEMESTER_START = dt.date(2026, 8, 3)  # a Monday
LOGGED_DAY_TYPES = ("normal", "special")  # menus that appear in the logs (scarcity is an analysis option)


def _minutes_to_clock(start: str, minutes: np.ndarray) -> np.ndarray:
    h, m = map(int, start.split(":"))
    secs = np.round((h * 60 + m + minutes) * 60).astype(int)
    return np.array([f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}" for s in secs])


def _spread_within_slots(rng: np.random.Generator, slot: np.ndarray, slot_minutes: float) -> np.ndarray:
    """Arrival offsets inside each slot, evenly spread with jitter (the fluid model's assumption).

    The j-th of the A arrivals of a slot (in random order) comes at (j + U)/A of the slot.
    """
    u = np.empty(len(slot))
    for t in np.unique(slot):
        members = np.flatnonzero(slot == t)
        order = rng.permutation(len(members))
        u[members] = (order + rng.random(len(members))) / len(members) * slot_minutes
    return u


def generate(weeks: int = 8, seed: int = 0, warmup_days: int = 30,
             lam: float = 2.0, revise: float = 0.3, day_shock: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    roster = make_roster(seed)
    rng = np.random.default_rng(seed + 7)
    dates = [SEMESTER_START + dt.timedelta(days=i) for i in range(7 * weeks)]
    frames = []
    for mess in MESSES:
        in_mess = roster.mess == mess
        ids = roster.ids[in_mess]
        beta = roster.beta[in_mess]
        for meal_name, meal in MEALS.items():
            for day_name in LOGGED_DAY_TYPES:
                day = DAY_TYPES[day_name]
                special = day_name == "special"
                day_dates = [d for d in dates if (d.weekday() in meal.special_days) == special]
                if not day_dates:
                    continue
                p = np.clip(roster.attend[meal_name][in_mess] + day.extra_attendance, 0, 1)
                # whole-day swings (exams, holidays, weather): attendance x N(1, day_shock), per day
                n_days = warmup_days + len(day_dates)
                shock = np.clip(rng.normal(1.0, day_shock, size=(n_days, 1)), 0.5, 1.3)
                tau = special_preferences(roster.tau[meal_name][in_mess], meal, day, rng)
                cm = congestion_model(meal, day, int(round(p.sum())))
                sim = simulate(cm, tau, beta, NoInfo(), Logit(lam, revise), days=warmup_days + len(day_dates),
                               burn_in=0, seed=int(rng.integers(1 << 31)),
                               attend_prob=np.clip(shock * p[None, :], 0, 1))
                for j, date in enumerate(day_dates):
                    d = warmup_days + j
                    here = np.isfinite(sim.cost[d])
                    slot = sim.slots[d][here]
                    A = sim.loads[d]
                    u = _spread_within_slots(rng, slot, meal.slot_minutes)
                    wait = cm.wait_at(A, slot, u)
                    order_rank = np.argsort(np.argsort(slot * meal.slot_minutes + u))  # FIFO service order
                    got = np.ones(len(slot), bool) if cm.stock is None else order_rank < cm.stock
                    arrival = slot * meal.slot_minutes + u
                    frames.append(pd.DataFrame({
                        "date": date.isoformat(), "weekday": WEEKDAYS[date.weekday()], "meal": meal_name,
                        "mess": mess, "menu": day_name, "student_id": ids[here],
                        "scan_min": arrival + wait,  # minutes after the meal opens
                        "slot": slot, "arrival_min": arrival, "wait_min": wait, "got_item": got,
                    }))
    truth = pd.concat(frames, ignore_index=True)
    truth["scan_time"] = ""
    for meal_name, meal in MEALS.items():
        rows = truth["meal"] == meal_name
        truth.loc[rows, "scan_time"] = _minutes_to_clock(meal.start, truth.loc[rows, "scan_min"].to_numpy())
    truth = truth.sort_values(["date", "meal", "mess", "scan_min"], ignore_index=True)
    scans = truth[["date", "weekday", "meal", "mess", "menu", "student_id", "scan_time"]].copy()
    return scans, truth


def roster_truth(seed: int = 0) -> pd.DataFrame:
    """Hidden preferences of every student (for checking estimators)."""
    r = make_roster(seed)
    df = pd.DataFrame({"student_id": r.ids, "mess": r.mess, "beta": r.beta})
    for meal in MEALS:
        df[f"tau_{meal}"] = r.tau[meal]
        df[f"attend_{meal}"] = r.attend[meal]
    return df


# ============================================================================
# Learning the unknowns from the scan logs alone.
#
# 1. Serving rate mu. While a queue exists, plates leave the counter at the
#    serving rate, so the busiest stretch of scans (highest 5-minute average
#    scan rate) estimates mu. We take the median over days.
# 2. Preferred time tau_i. A student's usual slot on normal days (median scan
#    slot) estimates when they like to eat. Crowding pushes people a little
#    away from their favourite slot, so this is a slightly biased estimate --
#    we report how often it is exactly right or within one slot.
# 3. Arrival profile. Scans per 10 minutes, averaged over days, by weekday:
#    the heatmap the mess office would look at first.
# Because the data are synthetic we know the truth and can score each estimate.
# ============================================================================

def _scan_minutes(scans: pd.DataFrame) -> pd.Series:
    t = scans["scan_time"].str.split(":", expand=True).astype(int)
    clock = t[0] * 60 + t[1] + t[2] / 60
    start = scans["meal"].map({k: int(m.start[:2]) * 60 + int(m.start[3:]) for k, m in MEALS.items()})
    return clock - start


def estimate_service_rate(scans: pd.DataFrame, mess: str, meal: str, menu: str = "normal",
                          window: int = 5) -> float:
    df = scans[(scans["mess"] == mess) & (scans["meal"] == meal) & (scans["menu"] == menu)]
    rates = []
    for _, day in df.groupby("date"):
        minute = np.floor(_scan_minutes(day)).astype(int)
        counts = np.bincount(minute.clip(lower=0), minlength=200)
        rates.append(np.convolve(counts, np.ones(window) / window, mode="valid").max())
    return float(np.median(rates)) if rates else float("nan")


def estimate_preferred_slots(scans: pd.DataFrame, meal: str) -> pd.Series:
    """Median scan slot per student on normal days."""
    m = MEALS[meal]
    df = scans[(scans["meal"] == meal) & (scans["menu"] == "normal")]
    slot = np.clip(np.floor(_scan_minutes(df) / m.slot_minutes), 0, m.T - 1)
    return slot.groupby(df["student_id"]).median().round().astype(int)


def preference_accuracy(estimated: pd.Series, true_tau: pd.Series) -> dict[str, float]:
    joined = pd.concat([estimated.rename("est"), true_tau.rename("true")], axis=1, join="inner")
    err = (joined["est"] - joined["true"]).abs()
    return {"students": int(len(joined)), "exact": float((err == 0).mean()), "within_1_slot": float((err <= 1).mean()),
            "mean_abs_error_slots": float(err.mean())}


def arrival_heatmap(scans: pd.DataFrame, mess: str, meal: str) -> pd.DataFrame:
    """Average scans per 10-minute bin, rows = weekday, columns = clock time."""
    m = MEALS[meal]
    df = scans[(scans["mess"] == mess) & (scans["meal"] == meal)].copy()
    bins = np.floor(_scan_minutes(df) / m.slot_minutes).clip(0, m.T + 3).astype(int)
    df["bin"] = bins
    counts = df.groupby(["date", "weekday", "bin"]).size().rename("scans").reset_index()
    table = counts.groupby(["weekday", "bin"])["scans"].mean().unstack(fill_value=0.0)
    table = table.reindex([d for d in WEEKDAYS if d in table.index])
    h, mi = map(int, m.start.split(":"))
    table.columns = [f"{(h * 60 + mi + 10 * c) // 60:02d}:{(h * 60 + mi + 10 * c) % 60:02d}" for c in table.columns]
    return table


# ============================================================================
# Nowcasting tonight's crowd (S6): predict the next 30 minutes of scans from
# the scans seen so far tonight.
#   N0  historical mean profile of earlier days with the same menu
#   N1  N0 scaled by (tonight's scans so far) / (historical scans by this time)
# Rolling-origin validation: every day is predicted using only earlier days.
# ============================================================================
def scans_per_slot(scans: pd.DataFrame, mess: str, meal: str, menu: str) -> pd.DataFrame:
    """Rows = dates (in order), columns = slots: number of scans in each 10-minute slot."""
    m = MEALS[meal]
    df = scans[(scans["mess"] == mess) & (scans["meal"] == meal) & (scans["menu"] == menu)]
    slot = np.clip(np.floor(_scan_minutes(df) / m.slot_minutes), 0, m.T + 3).astype(int)
    table = df.assign(slot=slot).groupby(["date", "slot"]).size().unstack(fill_value=0)
    return table.reindex(columns=range(m.T + 4), fill_value=0).sort_index()


def nowcast_errors(table: pd.DataFrame, horizon: int = 3, min_history: int = 3) -> dict[str, float]:
    """Mean absolute error (scans per slot) of N0 and N1 over all days and cut-off slots."""
    counts = table.to_numpy(dtype=float)
    err = {"N0 historical mean": [], "N1 ratio scaling": []}
    for d in range(min_history, len(counts)):
        hist = counts[:d].mean(axis=0)
        for cut in range(1, counts.shape[1] - horizon):
            truth = counts[d, cut:cut + horizon]
            n0 = hist[cut:cut + horizon]
            seen, expected = counts[d, :cut].sum(), hist[:cut].sum()
            n1 = n0 * (seen / expected if expected > 0 else 1.0)
            err["N0 historical mean"].append(np.abs(n0 - truth).mean())
            err["N1 ratio scaling"].append(np.abs(n1 - truth).mean())
    return {k: float(np.mean(v)) for k, v in err.items()}
