"""
Active inference: choosing what to do by minimising expected free energy.

The agent does not maximise a reward.  It minimises the difference between what
it expects to feel and what it *wants* to feel, where "wants" are supplied by
the interoceptive setpoints and the affective state.  That single change is
what turns optimisation into motivation.

Two quantities are computed:

**Variational free energy** ``F = inaccuracy + complexity``
    How badly the current sensory stream fits the generative model.  This is
    computed in :mod:`embodied_human.predictive` and passed in.

**Expected free energy** ``G(pi)`` for each candidate policy ``pi``

.. math::
    G(\\pi) = \\underbrace{D_{KL}[q(o|\\pi)\\,\\|\\,p(o)]}_{risk}
            + \\underbrace{H[q(o|\\pi)]}_{ambiguity}
            - \\underbrace{I(o;\\theta|\\pi)}_{epistemic value}

The *risk* term is where emotion enters: ``p(o)`` -- the preferred observation
-- is assembled from the homeostatic drives and the current affective state, so
a frightened agent literally prefers different sensations than a curious one.

Policies are equilibrium-point motor programs (see
:mod:`embodied_human.motor`), evaluated by rolling the *learned forward model*
forward.  Policy precision is a softmax whose temperature is set by
noradrenaline: aroused agents explore more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig
from .skeleton import nominal_posture


@dataclass
class Policy:
    """A named motor program: an equilibrium point plus optional overlays."""
    name: str
    overrides: dict = field(default_factory=dict)   # joint name -> absolute angle
    cpg: float = 0.0                                # locomotion drive 0..1
    duration: float = 1.0
    category: str = "posture"
    gaze: tuple = (0.0, 0.0)                        # eye yaw, pitch
    jaw: float = 0.0
    description: str = ""


# ==========================================================================
# The policy library
# ==========================================================================
def default_policies() -> list[Policy]:
    P: list[Policy] = []

    def add(name, ov, cat="posture", dur=1.0, gaze=(0.0, 0.0), jaw=0.0, desc=""):
        P.append(Policy(name=name, overrides=dict(ov), category=cat,
                        duration=dur, gaze=gaze, jaw=jaw, description=desc))

    # ---- do nothing / maintain ------------------------------------------
    add("hold", {}, desc="maintain the current equilibrium point")
    add("stand_tall", {"knee_l": 0.02, "knee_r": 0.02, "spine_bend": -0.02,
                       "chest_bend": 0.0}, desc="extend the legs, lift the chest")
    add("crouch", {"knee_l": 0.75, "knee_r": 0.75, "hip_l_flex": -0.55,
                   "hip_r_flex": -0.55, "ankle_l_flex": -0.30,
                   "ankle_r_flex": -0.30, "spine_bend": 0.18},
        desc="lower the centre of mass")

    # ---- balance shifting ------------------------------------------------
    add("weight_left", {"ankle_l_inv": -0.06, "ankle_r_inv": 0.10,
                        "hip_l_abd": -0.12, "hip_r_abd": 0.02,
                        "spine_side": 0.06}, cat="balance", dur=1.2,
        desc="shift weight onto the left foot")
    add("weight_right", {"ankle_l_inv": -0.10, "ankle_r_inv": 0.06,
                         "hip_l_abd": -0.02, "hip_r_abd": 0.12,
                         "spine_side": -0.06}, cat="balance", dur=1.2,
        desc="shift weight onto the right foot")
    add("lean_forward", {"ankle_l_flex": 0.10, "ankle_r_flex": 0.10,
                         "spine_bend": 0.10, "hip_l_flex": 0.04,
                         "hip_r_flex": 0.04}, cat="balance",
        desc="lean the body forwards")
    add("lean_back", {"ankle_l_flex": -0.16, "ankle_r_flex": -0.16,
                      "spine_bend": -0.04, "toe_l": -0.25, "toe_r": -0.25},
        cat="balance", desc="lean back onto the heels")
    add("lean_left", {"ankle_l_inv": -0.12, "ankle_r_inv": -0.12,
                      "spine_side": 0.12, "hip_l_abd": -0.05,
                      "hip_r_abd": 0.05}, cat="balance", desc="lean to the left")
    add("lean_right", {"ankle_l_inv": 0.12, "ankle_r_inv": 0.12,
                       "spine_side": -0.12, "hip_l_abd": -0.05,
                       "hip_r_abd": 0.05}, cat="balance", desc="lean to the right")

    # ---- reaching / arm movement ----------------------------------------
    for s in ("l", "r"):
        sx = 1.0 if s == "l" else -1.0
        add(f"reach_forward_{s}",
            {f"sh_{s}_flex": -1.05, f"sh_{s}_abd": -0.10 * sx, f"elbow_{s}": -0.55,
             f"wrist_{s}_flex": -0.10}, cat="reach", dur=1.4,
            desc=f"reach forward with the {'left' if s == 'l' else 'right'} arm")
        add(f"reach_up_{s}",
            {f"sh_{s}_flex": -2.30, f"sh_{s}_abd": -0.35 * sx, f"elbow_{s}": -0.25},
            cat="reach", dur=1.6, desc="raise the arm overhead")
        add(f"reach_down_{s}",
            {f"sh_{s}_flex": -0.30, f"sh_{s}_abd": 0.05 * sx,
             f"elbow_{s}": -0.20}, cat="reach", dur=1.2,
            desc="reach down towards the table")
        add(f"reach_across_{s}",
            {f"sh_{s}_flex": -1.20, f"sh_{s}_abd": 0.55 * sx,
             f"sh_{s}_rot": 0.9 * sx, f"elbow_{s}": -1.30}, cat="reach", dur=1.6,
            desc="reach across the body")
        add(f"hand_to_face_{s}",
            {f"sh_{s}_flex": -1.35, f"sh_{s}_abd": 0.30 * sx,
             f"elbow_{s}": -2.05, f"wrist_{s}_flex": 0.25}, cat="self_touch",
            dur=2.0, desc="bring the hand to the face")
        add(f"hand_to_chest_{s}",
            {f"sh_{s}_flex": -0.75, f"sh_{s}_abd": 0.30 * sx,
             f"elbow_{s}": -1.85}, cat="self_touch", dur=1.6,
            desc="place the hand on the chest")
        add(f"rub_forearm_{s}",
            {f"sh_{s}_flex": -0.95, f"sh_{s}_abd": 0.15 * sx,
             f"elbow_{s}": -1.95, f"wrist_{s}_flex": 0.15}, cat="self_touch",
            dur=2.2, desc="stroke the opposite forearm (self-touch)")
        add(f"grasp_{s}",
            {f"{f}_{s}_mcp": 0.95 * sx for f in ("thumb", "index", "fingers")}
            | {f"{f}_{s}_pip": 0.95 * sx for f in ("thumb", "index", "fingers")},
            cat="hand", dur=1.2, desc="close the hand")
        add(f"open_hand_{s}",
            {f"{f}_{s}_mcp": -0.10 * sx for f in ("thumb", "index", "fingers")}
            | {f"{f}_{s}_pip": -0.05 * sx for f in ("thumb", "index", "fingers")},
            cat="hand", dur=1.0, desc="open the hand")
        add(f"tap_fingers_{s}",
            {f"index_{s}_mcp": 0.55 * sx, f"index_{s}_pip": 0.60 * sx},
            cat="hand", dur=0.8, desc="tap the index finger")

    # ---- gaze ------------------------------------------------------------
    add("gaze_front", {}, cat="gaze", dur=0.8, gaze=(0.0, 0.0))
    add("gaze_left", {}, cat="gaze", dur=0.8, gaze=(-0.55, 0.0))
    add("gaze_right", {}, cat="gaze", dur=0.8, gaze=(0.55, 0.0))
    add("gaze_up", {}, cat="gaze", dur=0.8, gaze=(0.0, -0.40))
    add("gaze_down", {}, cat="gaze", dur=0.8, gaze=(0.0, 0.40))
    add("gaze_hand_l", {"neck_twist": 0.45, "neck_bend": 0.35}, cat="gaze",
        dur=1.2, gaze=(0.3, 0.4), desc="look at the left hand")
    add("gaze_hand_r", {"neck_twist": -0.45, "neck_bend": 0.35}, cat="gaze",
        dur=1.2, gaze=(-0.3, 0.4), desc="look at the right hand")

    # ---- head / face -----------------------------------------------------
    add("nod", {"neck_bend": 0.30}, cat="head", dur=0.7)
    add("look_up", {"neck_bend": -0.28}, cat="head", dur=0.8)
    add("shake_head", {"neck_twist": 0.55}, cat="head", dur=0.7)
    add("turn_head_left", {"neck_twist": 0.75}, cat="head", dur=0.9)
    add("turn_head_right", {"neck_twist": -0.75}, cat="head", dur=0.9)
    add("open_mouth", {}, cat="face", dur=0.6, jaw=0.35)
    add("close_mouth", {}, cat="face", dur=0.5, jaw=0.0)

    # ---- defensive -------------------------------------------------------
    add("flinch", {"sh_l_abd": -0.55, "sh_r_abd": 0.55,
                   "sh_l_flex": -0.45, "sh_r_flex": -0.45,
                   "elbow_l": -1.30, "elbow_r": -1.30,
                   "spine_bend": 0.22, "neck_bend": 0.30,
                   "knee_l": 0.35, "knee_r": 0.35}, cat="defensive", dur=1.0,
        desc="withdraw and protect the head and trunk")
    add("freeze", {"knee_l": 0.22, "knee_r": 0.22, "spine_bend": 0.06},
        cat="defensive", dur=1.5, desc="stiffen and stop moving")
    add("brace", {"knee_l": 0.30, "knee_r": 0.30, "hip_l_flex": -0.20,
                  "hip_r_flex": -0.20, "ankle_l_flex": -0.18,
                  "ankle_r_flex": -0.18, "spine_bend": 0.12},
        cat="defensive", dur=1.5, desc="brace for impact")
    add("step_back", {"hip_l_flex": -0.45, "knee_r": 0.50, "knee_l": 0.30},
        cat="defensive", dur=1.2, desc="recoil backwards")

    # ---- locomotion ------------------------------------------------------
    add("walk_in_place", {}, cat="locomotion", dur=2.0)
    P[-1].cpg = 1.0
    add("walk_forward", {}, cat="locomotion", dur=2.0)
    P[-1].cpg = 1.0
    add("march", {}, cat="locomotion", dur=1.6)
    P[-1].cpg = 0.7

    # ---- exploratory -----------------------------------------------------
    add("explore_left", {"neck_twist": 0.35, "spine_twist": 0.15},
        cat="explore", dur=1.2, gaze=(0.2, 0.0))
    add("explore_right", {"neck_twist": -0.35, "spine_twist": -0.15},
        cat="explore", dur=1.2, gaze=(-0.2, 0.0))
    add("explore_hands",
        {f"{f}_{s}_mcp": 0.30 * (1 if s == "l" else -1)
         for s in ("l", "r") for f in ("thumb", "index", "fingers")},
        cat="explore", dur=1.6, desc="open and close the hands to feel the air")
    add("press_palm_l", {"sh_l_flex": -0.55, "sh_l_abd": -0.10, "elbow_l": -0.85},
        cat="explore", dur=1.5, desc="press the left palm onto a surface")
    add("press_palm_r", {"sh_r_flex": -0.55, "sh_r_abd": 0.10, "elbow_r": -0.85},
        cat="explore", dur=1.5, desc="press the right palm onto a surface")

    return P


class ActiveInference:
    """Evaluates policies by expected free energy and picks one."""

    def __init__(self, cfg: SimConfig, meta, policies: list[Policy] | None = None):
        self.cfg = cfg
        self.I = cfg.inference
        self.meta = meta
        self.names = [n for _, n, _ in meta.joint_order]
        self.idx = {n: i for i, n in enumerate(self.names)}
        nominal = nominal_posture()
        self.q_nom = np.array([nominal.get(n, 0.0) for n in self.names])
        self.policies = policies if policies is not None else default_policies()
        self.targets = np.stack([self._target(p) for p in self.policies])
        self.current: Policy = self.policies[0]
        self.hold_timer = 0.0
        self.last_G = np.zeros(len(self.policies))
        self.last_probs = np.zeros(len(self.policies))
        self.n_switches = 0
        self.rng = np.random.default_rng(cfg.seed + 71)
        self.history: list = []

    # ------------------------------------------------------------------
    def _target(self, p: Policy) -> np.ndarray:
        t = self.q_nom.copy()
        for k, v in p.overrides.items():
            if k in self.idx:
                t[self.idx[k]] = v
        return t

    # ------------------------------------------------------------------
    def evaluate(self, latent: np.ndarray, preferred: np.ndarray,
                 predict, precision: np.ndarray, *,
                 q_slice: slice | None = None,
                 policy_temperature: float = 1.0,
                 valence: float = 0.0, arousal: float = 0.3,
                 fear: float = 0.0, pain: float = 0.0,
                 curiosity: float = 0.0, social_need: float = 0.0,
                 balance_error: float = 0.0,
                 locomotor_drive: float = 0.0,
                 drives_level: np.ndarray | None = None,
                 action_tendency: np.ndarray | None = None,
                 current_action: np.ndarray | None = None,
                 ) -> tuple[Policy, np.ndarray, dict]:
        """Return (chosen policy, G for every policy, diagnostics).

        Risk is computed against a *belief* about the sensory consequence of
        adopting each equilibrium point: the proprioceptive block is expected
        to move most of the way to the policy's target, and once the learned
        forward model has seen enough data its prediction is blended in.  This
        keeps policy selection meaningful from the very first tick, which
        matters because a body has to act before it has learned anything.
        """
        I = self.I
        n_pol = len(self.policies)
        G = np.zeros(n_pol)
        risk_all = np.zeros(n_pol)
        bias_all = np.zeros(n_pol)
        if q_slice is None:
            q_slice = slice(0, self.n)

        fwd = predict.forward if predict is not None else None
        err_var = np.maximum(fwd.err_var, 1e-4) if fwd is not None else np.ones(1)
        log_amb = 0.5 * float(np.sum(np.log1p(err_var)))
        model_ready = fwd is not None and fwd.n_updates > 40
        if drives_level is not None and len(drives_level) > 13:
            curiosity = max(curiosity, float(drives_level[13]))

        for i, pol in enumerate(self.policies):
            a = self.targets[i]
            # belief about where the body will be once it adopts this target
            pred = latent.copy()
            pred[q_slice] = 0.65 * a + 0.35 * latent[q_slice]
            if model_ready:
                learned = predict.imagine(latent, a)
                pred = 0.45 * pred + 0.55 * learned
            d = pred - preferred
            risk = 0.5 * float(np.sum((d ** 2) * precision))
            risk_all[i] = risk

            ambiguity = log_amb * (1.0 - 0.05 * min(pol.duration, 2.0))
            if current_action is not None:
                motor_cost = 0.0015 * float(np.sum((a - current_action) ** 2))
            else:
                motor_cost = 0.0

            bias = self._category_bias(pol, fear, pain, valence, arousal,
                                       curiosity, social_need, balance_error,
                                       action_tendency, locomotor_drive)
            bias_all[i] = bias
            G[i] = (I.risk_weight * risk + I.ambiguity_weight * ambiguity
                    + motor_cost + bias)

        G = G - G.min()
        temp = max(0.05, policy_temperature * (1.0 + 0.6 * arousal)
                   * self.cfg.inference.temperature)
        logits = -G / temp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        # A committed motor program is normally held to completion, but a
        # postural threat interrupts it -- the same override a person shows
        # when a reach is cut short because the ground moves under them.
        interrupted = balance_error > 0.06
        if self.hold_timer <= 0.0 or interrupted:
            idx = int(self.rng.choice(n_pol, p=probs))
        else:
            idx = self.policies.index(self.current)
        chosen = self.policies[idx]

        self.last_G = G
        self.last_probs = probs
        info = {
            "risk": risk_all,
            "bias": bias_all,
            "ambiguity": float(log_amb),
            "g_mean": float(G.mean()),
            "g_max": float(G.max()),
            "entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
            "chosen_prob": float(probs[idx]),
            "n_policies": n_pol,
            "model_ready": bool(model_ready),
        }
        return chosen, G, info

    # ------------------------------------------------------------------
    @staticmethod
    def _category_bias(pol, fear: float, pain: float, valence: float,
                       arousal: float, curiosity: float, social_need: float,
                       balance_error: float,
                       tendency: np.ndarray | None,
                       locomotor_drive: float = 0.0) -> float:
        """Emotional action tendencies bias which motor programs are considered.

        This is deliberately *not* part of the free-energy calculation: an
        action tendency is a low-level disposition (freeze, withdraw, approach)
        rather than a deliberative prediction.  Keeping them separate is what
        lets fear produce a flinch in a fraction of a second.
        """
        approach = avoid = freeze = attack = withdraw = explore = 0.0
        if tendency is not None and len(tendency) >= 6:
            approach, avoid, freeze, attack, withdraw, explore = [float(x)
                                                                  for x in tendency[:6]]
        cat = pol.category
        # Exploration and self-touch are suppressed when the body is unstable:
        # nobody investigates an interesting object while falling over.
        steady = float(np.clip(1.0 - balance_error * 8.0, 0.0, 1.0))
        b = 0.0
        if cat == "defensive":
            b -= 6.0 * (0.8 * fear + 0.6 * pain) + 3.0 * avoid + 2.5 * freeze + 1.5 * withdraw
        elif cat == "balance":
            b -= 3.0 * balance_error
        elif cat == "explore":
            b -= steady * (1.5 * curiosity * (1.0 - fear) + 0.8 * explore) - 1.5 * fear
        elif cat == "self_touch":
            b -= steady * (1.4 * social_need * (1.0 - fear) + 0.4 * approach) - 1.5 * fear
        elif cat in ("reach", "hand"):
            b -= steady * (0.7 * curiosity + 0.4 * approach) - 1.3 * fear
        elif cat == "locomotion":
            # Walking is only available when something actually wants to move.
            # The CPG oscillates the legs without a balance coupling, so
            # engaging it while standing is a guaranteed fall.
            b -= steady * (0.9 * explore + 0.5 * approach) - 2.5 * fear
            b += 8.0 * (1.0 - float(np.clip(locomotor_drive, 0.0, 1.0)))
            b += 6.0 * float(np.clip(balance_error * 6.0, 0.0, 1.0))
        elif cat == "gaze":
            b -= 0.5 * curiosity * steady - 0.2 * fear
        elif cat == "face":
            b -= 0.25 * max(0.0, valence)
        elif cat == "head":
            b -= 0.2 * curiosity * steady
        if pol.name == "hold":
            # a small metabolic preference for the rest configuration
            b -= 0.45 * steady + 0.4 * float(np.clip(1.0 - valence, 0, 2))
        return b

    # ------------------------------------------------------------------
    def commit(self, policy: Policy, dt: float) -> np.ndarray:
        if policy is not self.current:
            self.n_switches += 1
            self.current = policy
            self.hold_timer = policy.duration
        self.hold_timer = max(0.0, self.hold_timer - dt)
        return self.targets[self.policies.index(policy)]

    def gaze_command(self) -> tuple:
        return self.current.gaze

    def jaw_command(self) -> float:
        return self.current.jaw

    def cpg_command(self) -> float:
        return self.current.cpg

    def reset(self) -> None:
        self.current = self.policies[0]
        self.hold_timer = 0.0
        self.n_switches = 0

    def describe(self) -> dict:
        cats: dict[str, int] = {}
        for p in self.policies:
            cats[p.category] = cats.get(p.category, 0) + 1
        return {
            "n_policies": len(self.policies),
            "categories": cats,
            "policy_names": [p.name for p in self.policies],
            "horizon": self.I.horizon,
            "risk_weight": self.I.risk_weight,
            "ambiguity_weight": self.I.ambiguity_weight,
        }
