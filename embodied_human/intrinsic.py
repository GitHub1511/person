"""
Intrinsic motivation: the reward decomposition.

There is no task and no external reward.  Everything the agent "wants" is
generated internally, which is what makes its behaviour look motivated rather
than instructed.  The terms are the ones that make an embodied agent prefer
its own body over random flailing:

============  =========================================================
term          what it rewards
============  =========================================================
jerk          smoothness of the motor command (no twitching)
energy        metabolic economy
stability     staying upright and near the equilibrium point
novelty       curiosity: prediction error that is reducible
tactile       informative, exploratory contact
comfort       moving the internal milieu towards its setpoints
affective     positive valence and low tension (the agent likes feeling well)
empowerment   maintaining control over one's own sensory consequences
pain          a large, hard penalty on nociceptive input
contact       soft, gradual contact rather than hard impacts
============  =========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from .config import SimConfig

REWARD_TERMS = ("jerk", "energy", "stability", "novelty", "tactile", "comfort",
                "affective", "empowerment", "pain", "contact", "total")


@dataclass
class RewardFrame:
    t: float = 0.0
    terms: np.ndarray = field(default_factory=lambda: np.zeros(len(REWARD_TERMS)))
    total: float = 0.0
    rpe: float = 0.0

    def as_dict(self) -> dict:
        return {n: float(self.terms[i]) for i, n in enumerate(REWARD_TERMS)}


class IntrinsicMotivation:
    """Combines the reward terms into one scalar with a prediction-error signal."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.W = cfg.intrinsic
        self.prev_total = 0.0
        self.baseline = 0.0
        self.value_estimate = 0.0
        self.rpe_ema = 0.0
        self._init = False

    # ------------------------------------------------------------------
    def compute(self, *, jerk: float, effort: float, power: float,
                balance_error: float, free_energy: float,
                surprise: float, body_ownership: np.ndarray | None,
                touch_intensity: float, contact_rate: float,
                pain: float, drive_pressure: float, valence: float,
                tension: float, empowerment: float) -> RewardFrame:
        W = self.W

        # ---- individual terms ---------------------------------------
        r_jerk = -W.w_jerk * float(fclip(jerk * 50.0, 0, 10))
        r_energy = -W.w_energy * float(fclip(abs(power) / 200.0 + effort, 0, 5))
        r_stability = -W.w_stability * float(fclip(balance_error * 8.0, 0, 10))
        # novelty: reward prediction error that is *reducible* -- we use the
        # free energy rather than raw surprise so irreducible noise is ignored
        r_novelty = W.w_novelty * float(fclip(np.tanh(free_energy * 0.15), 0, 1)) \
            * (1.0 if free_energy > 0.05 else 0.0)
        r_tactile = W.w_tactile * float(fclip(touch_intensity / 5.0, 0, 2))
        r_comfort = -W.w_comfort * float(fclip(drive_pressure, 0, 3))
        r_affect = W.w_affective_balance * float(fclip(valence, -1, 1)) \
            - 0.4 * W.w_affective_balance * float(fclip(tension, 0, 1.5))
        r_empower = W.w_empowerment * float(fclip(empowerment, 0, 5)) / 5.0
        r_pain = -W.w_pain * float(fclip(pain, 0, 1.5))
        r_contact = -W.w_smoothness_contact * float(fclip(contact_rate, 0, 1))

        terms = np.array([r_jerk, r_energy, r_stability, r_novelty, r_tactile,
                          r_comfort, r_affect, r_empower, r_pain, r_contact])
        total = float(terms.sum())

        # ---- reward prediction error --------------------------------
        if not self._init:
            self.value_estimate = total
            self.baseline = total
            self._init = True
        self.value_estimate += 0.05 * (total - self.value_estimate)
        self.baseline += 0.001 * (total - self.baseline)
        rpe = total - self.value_estimate
        self.rpe_ema += 0.05 * (rpe - self.rpe_ema)

        out = np.concatenate([terms, [total]])
        return RewardFrame(terms=out, total=total, rpe=float(rpe))

    def reset(self) -> None:
        self._init = False
        self.value_estimate = 0.0
        self.baseline = 0.0
        self.rpe_ema = 0.0

    def describe(self) -> dict:
        return {
            "terms": list(REWARD_TERMS),
            "weights": {
                "jerk": self.W.w_jerk, "energy": self.W.w_energy,
                "stability": self.W.w_stability, "novelty": self.W.w_novelty,
                "tactile": self.W.w_tactile, "comfort": self.W.w_comfort,
                "affective": self.W.w_affective_balance,
                "empowerment": self.W.w_empowerment, "pain": self.W.w_pain,
                "contact": self.W.w_smoothness_contact,
            },
        }
