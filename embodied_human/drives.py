"""
Homeostatic drives: what the body *wants*.

A drive is a scalar that rises as an internal variable departs from its
viability range and falls when it is satisfied.  Drives are the bridge between
interoception and behaviour: they become the **preferred observations** that
active inference minimises expected free energy against, and they are what
makes the agent's goals *its own* rather than assigned.

Each drive reports:

* ``level``      normalised 0 (satisfied) .. 1+ (urgent)
* ``setpoint``   the value the body is trying to hold
* ``value``      the current interoceptive reading
* ``urgency``    level scaled by how fast it is getting worse and by
                 allostatic load (a stressed body prioritises differently)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from .affect import NM, AffectSystem
from .config import SimConfig
from .interoception import IDX, InteroceptiveSystem

DRIVES = (
    "hunger", "thirst", "sleepiness", "thermal_cold", "thermal_heat",
    "pain", "air_hunger", "nausea", "bladder", "fatigue",
    "itch", "social_need", "safety", "curiosity", "comfort", "restlessness",
    # added with the internal world (see inner_world.EXTRA_DRIVES)
    "ocular_comfort", "muscle_soreness", "gut_discomfort", "mental_fatigue", "shift_urge",
)
N_DRIVES = len(DRIVES)
D = {n: i for i, n in enumerate(DRIVES)}


@dataclass
class DriveFrame:
    t: float = 0.0
    level: np.ndarray = field(default_factory=lambda: np.zeros(N_DRIVES))
    urgency: np.ndarray = field(default_factory=lambda: np.zeros(N_DRIVES))
    setpoint: np.ndarray = field(default_factory=lambda: np.zeros(N_DRIVES))
    value: np.ndarray = field(default_factory=lambda: np.zeros(N_DRIVES))
    slope: np.ndarray = field(default_factory=lambda: np.zeros(N_DRIVES))
    most_urgent: str = ""
    most_urgent_level: float = 0.0
    total_pressure: float = 0.0
    satisfaction: float = 1.0

    def flat(self) -> np.ndarray:
        return np.concatenate([self.level, self.urgency, self.slope])

    def as_dict(self) -> dict:
        return {n: float(self.level[i]) for i, n in enumerate(DRIVES)}

    def top(self, k: int = 4) -> list[tuple[str, float]]:
        order = np.argsort(-self.level)[:k]
        return [(DRIVES[i], float(self.level[i])) for i in order]


SETPOINTS = {
    "hunger": 0.35, "thirst": 1.0, "sleepiness": 0.2, "core_temp": 36.8,
    "pain": 0.0, "air_hunger": 0.05, "nausea": 0.0, "bladder": 0.3,
    "fatigue": 0.1, "itch": 0.0, "social_need": 0.2, "safety": 1.0,
    "curiosity": 0.5, "comfort": 0.0,
}


class DriveSystem:
    """Turns the internal milieu plus affect into motivational pressure."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.prev = np.zeros(N_DRIVES)
        self.level = np.zeros(N_DRIVES)
        self.slope = np.zeros(N_DRIVES)
        self.social_satiation = 0.2
        self.curiosity_satiation = 0.5
        self.safety = 1.0
        self._init = False

    # ------------------------------------------------------------------
    def update(self, dt: float, intero: InteroceptiveSystem,
               affect, afferent_pain: float = 0.0,
               afferent_itch: float = 0.0, social_contact: float = 0.0,
               novelty: float = 0.0, balance_error: float = 0.0,
               fallen: float = 0.0, extra: np.ndarray | None = None) -> DriveFrame:
        s = intero.s
        g = IDX

        # ---- interoceptive drives -------------------------------------
        # hunger: gastric emptiness + ghrelin, resisted by leptin
        hunger = fclip(0.55 * (1.0 - s[g["gastric_fullness"]])
                         + 0.75 * s[g["ghrelin"]]
                         - 0.45 * s[g["leptin"]]
                         + 0.25 * (1.0 - s[g["glucose"]] / 5.2), 0.0, 1.4)
        # thirst from osmolality and hypovolaemia
        thirst = fclip(0.9 * (s[g["osmolality"]] / 290.0 - 1.0) * 4.0
                         + 0.8 * (1.0 - s[g["hydration"]]), 0.0, 1.4)
        sleepiness = float(fclip(s[g["sleepiness"]], 0.0, 1.3))
        # thermal drives are directional
        cold = fclip((36.6 - s[g["core_temp"]]) * 3.5
                       + (33.0 - s[g["skin_temp_mean"]]) * 0.12, 0.0, 1.4)
        heat = fclip((s[g["core_temp"]] - 37.1) * 3.5
                       + (s[g["skin_temp_mean"]] - 35.0) * 0.10, 0.0, 1.4)
        # pain: central integration already applied the gate
        pain = float(fclip(1.15 * s[g["pain_intensity"]] + 0.35 * afferent_pain,
                             0.0, 1.5))
        air = float(fclip(s[g["air_hunger"]], 0.0, 1.3))
        nausea = float(fclip(s[g["nausea"]], 0.0, 1.3))
        bladder = float(fclip((s[g["bladder_fullness"]] - 0.25) / 0.6, 0.0, 1.3))
        fatigue = float(fclip(0.7 * s[g["peripheral_fatigue"]]
                                + 0.6 * s[g["central_fatigue"]], 0.0, 1.3))
        itch = float(fclip(1.2 * afferent_itch
                             + 0.05 * affect.neuromodulator("histamine"),
                             0.0, 1.3))

        # ---- psychological drives -------------------------------------
        # social need: satiated by affectionate touch and contact
        self.social_satiation += dt * (
            -0.05 * social_contact + 0.0012 * (1.0 - self.social_satiation))
        self.social_satiation = float(fclip(self.social_satiation, 0.0, 1.2))
        social_need = float(fclip(
            self.social_satiation + 0.5 * affect.emotion("loneliness")
            - 0.3 * (1.0 - affect.stress), 0.0, 1.4))

        # safety: threat, pain, instability, and the memory of a fall
        safety_target = float(fclip(
            1.0 - 0.7 * affect.emotion("fear") - 0.4 * affect.emotion("anxiety")
            - 0.6 * pain - 1.0 * fallen - 0.8 * balance_error, 0.0, 1.2))
        self.safety += (dt / 1.0) * (safety_target - self.safety)

        # curiosity: novelty-seeking, damped by fear and fatigue, and it
        # satiates when the agent is actually learning something
        self.curiosity_satiation += dt * (
            -1.6 * novelty + 0.06 * (1.0 - self.curiosity_satiation))
        self.curiosity_satiation = float(fclip(self.curiosity_satiation, 0.0, 1.2))
        curiosity = float(fclip(
            self.curiosity_satiation
            * (1.0 - 0.5 * affect.emotion("fear"))
            * (1.0 - 0.3 * s[g["central_fatigue"]]), 0.0, 1.3))

        # comfort: the sum of mild physical discomforts
        comfort = float(fclip(
            0.5 * s[g["thermal_discomfort"]] + 0.35 * nausea
            + 0.3 * itch + 0.25 * s[g["sickness_behavior"]], 0.0, 1.3))

        # restlessness: arousal with nowhere to go
        restlessness = float(fclip(
            affect.arousal * 0.8 - 0.6 * affect.dominance
            + 0.4 * affect.emotion("boredom"), 0.0, 1.4))

        levels = np.array([hunger, thirst, sleepiness, cold, heat, pain, air,
                           nausea, bladder, fatigue, itch, social_need,
                           1.0 - self.safety, curiosity, comfort, restlessness])
        n_extra = N_DRIVES - len(levels)
        ex = np.zeros(n_extra) if extra is None else np.asarray(extra, float)[:n_extra]
        if len(ex) < n_extra:
            ex = np.concatenate([ex, np.zeros(n_extra - len(ex))])
        levels = np.concatenate([levels, ex])
        levels = np.nan_to_num(fclip(levels, 0.0, 1.5))

        if not self._init:
            self.prev = levels.copy()
            self._init = True
        # rate of change: a drive that is getting worse feels more urgent
        raw_slope = (levels - self.prev) / max(dt, 1e-6)
        self.slope += (dt / 3.0) * (fclip(raw_slope, -2.0, 2.0) - self.slope)
        self.prev = levels.copy()

        # allostatic load and stress amplify urgency (a stressed body
        # prioritises its needs more sharply)
        amplif = 1.0 + 0.5 * affect.allostatic_load + 0.35 * affect.stress
        urgency = fclip(levels * amplif + 0.6 * fclip(self.slope, 0, 2.0),
                          0.0, 2.0)
        self.level += (dt / 0.5) * (levels - self.level)

        sp = np.array([SETPOINTS.get(n, 0.0) for n in DRIVES])
        val = np.array([
            s[g["gastric_fullness"]], s[g["hydration"]], s[g["sleepiness"]],
            s[g["core_temp"]], s[g["core_temp"]], s[g["pain_intensity"]],
            s[g["air_hunger"]], s[g["nausea"]], s[g["bladder_fullness"]],
            s[g["peripheral_fatigue"]], afferent_itch, self.social_satiation,
            self.safety, self.curiosity_satiation, s[g["thermal_discomfort"]],
            affect.arousal,
        ])
        val = np.concatenate([val, levels[len(val):]])

        order = int(np.argmax(levels))
        total = float(np.sum(fclip(levels, 0, 1) ** 2)) ** 0.5

        return DriveFrame(
            level=levels,
            urgency=urgency,
            setpoint=sp,
            value=val,
            slope=self.slope.copy(),
            most_urgent=DRIVES[order],
            most_urgent_level=float(levels[order]),
            total_pressure=total,
            satisfaction=float(fclip(1.0 - np.mean(fclip(levels, 0, 1)), 0, 1)),
        )

    def describe(self) -> dict:
        return {
            "n_drives": N_DRIVES,
            "drives": list(DRIVES),
            "setpoints": SETPOINTS,
        }
