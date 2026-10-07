"""
Choosing and executing behaviours from the generative space.

``BehaviorSelector``  turns the person's condition (drives, affect, the state
                      of the eyes ...) into a *desire* over the affinity tags of
                      the behaviour space, samples candidate descriptors from the
                      structured prior this gives, and scores each candidate by
                      expected free energy using the learned forward model --
                      the same machinery the 55 original policies used, now over
                      an open-ended set.
``BehaviorExecutor``  runs the chosen behaviour at 50 Hz: joint targets with
                      speed / amplitude / tremor / rhythm, gaze, lids and blink
                      patterns, jaw and vocalisation, weight shift and crouch,
                      and -- through the skill layer -- reaching, pointing,
                      grasping and touching *itself* (rubbing the eyes, scratching,
                      hugging the arms for warmth).
``BehaviorStats``     measures how much of the space is actually used.

It runs in two modes.  *Autonomous*: the person is its own mind and these
behaviours are everything it does.  *Ambient*: something else (the language-model
mind) is in charge of deliberate action, and this layer only supplies the
involuntary, expressive and housekeeping behaviour a person produces anyway --
the glances, the shifting, the blinking, the hand that goes to a dry eye -- on the
channels the mind is not using.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ._fast import fclip
from .behavior_space import (ARM_SCHEMAS, ARM_VAR, BLINK_PATTERNS, GAZE_PITCH, GAZE_YAW,
                             HAND_SHAPES, HAND_VAR, HEAD_PITCH, HEAD_TILT, HEAD_YAW,
                             HOLD_SECONDS, INTENT_MODES, JAW_LEVELS, LID_APERTURE,
                             LID_APERTURE_NAMES, NT, OBJECTS, SELF_ACTIONS, SELF_REGIONS,
                             STANCE_FORWARD, STANCE_HEIGHT, STANCE_LATERAL, STYLE_AMP,
                             STYLE_RHYTHM, STYLE_SPEED, STYLE_TREMOR, TAGS, TI, TORSO_BEND,
                             TORSO_SIDE, TORSO_TWIST, VOCALS, Behavior, BehaviorSpace)

# channel groups the sampler treats as units ---------------------------------
GROUPS = {
    "head": ["head"], "eyes": ["eyes"], "lids": ["lids"], "mouth": ["mouth"],
    "torso": ["torso"],
    "l_arm": ["l_schema"] + [f"l_arm_{j}" for j in range(6)],
    "r_arm": ["r_schema"] + [f"r_arm_{j}" for j in range(6)],
    "l_hand": ["l_hand"] + [f"l_hand_{j}" for j in range(3)],
    "r_hand": ["r_hand"] + [f"r_hand_{j}" for j in range(3)],
    "stance": ["stance"], "style": ["style"], "hold": ["hold"],
    "intent": ["intent"], "touch": ["touch"], "walk": ["walk"],
}
# which channels carry an affinity score
SCORED = ("head", "eyes", "lids", "mouth", "torso", "l_schema", "r_schema", "l_hand",
          "r_hand", "stance", "style", "intent", "touch", "walk")
BASE_P = {"head": 0.55, "eyes": 0.65, "lids": 0.35, "mouth": 0.22, "torso": 0.35,
          "l_arm": 0.45, "r_arm": 0.45, "l_hand": 0.38, "r_hand": 0.38, "stance": 0.16,
          "style": 0.5, "hold": 1.0, "intent": 0.12, "touch": 0.20, "walk": 0.0}
AMBIENT_P = {"head": 0.30, "eyes": 0.45, "lids": 0.40, "mouth": 0.08, "torso": 0.18,
             "l_arm": 0.12, "r_arm": 0.12, "l_hand": 0.30, "r_hand": 0.30, "stance": 0.05,
             "style": 0.3, "hold": 1.0, "intent": 0.0, "touch": 0.10, "walk": 0.0}
# joints the behaviour layer writes (legs are the whole-body controller's)
LEG_PREFIX = ("hip_", "knee_", "ankle_", "toe_")


def _gumbel(rng, shape):
    return -np.log(-np.log(rng.random(shape) + 1e-12) + 1e-12)


# ==========================================================================
class BehaviorStats:
    """How much of the behaviour space has been used."""

    def __init__(self, space: BehaviorSpace):
        self.space = space
        self.keys: set[bytes] = set()
        self.n = 0
        self.per_channel = [Counter() for _ in range(space.n_channels)]
        self.recent: list[str] = []

    def add(self, b: Behavior) -> None:
        self.n += 1
        self.keys.add(b.key())
        for i, v in enumerate(b.desc):
            self.per_channel[i][int(v)] += 1
        self.recent.append(b.name)
        self.recent = self.recent[-12:]

    def summary(self) -> dict:
        ent = []
        for (nm, size), c in zip(self.space.layout, self.per_channel):
            tot = sum(c.values())
            if tot == 0:
                continue
            p = np.array(list(c.values()), float) / tot
            ent.append(float(-(p * np.log(p + 1e-12)).sum() / max(math.log(int(size)), 1e-9)))
        return {"decisions": self.n, "distinct": len(self.keys),
                "mean_channel_entropy": float(np.mean(ent)) if ent else 0.0}


# ==========================================================================
class BehaviorSelector:
    def __init__(self, space: BehaviorSpace, seed: int = 0):
        self.space = space
        self.rng = np.random.default_rng(seed + 8811)
        self.layout_idx = space.channel_index
        self.gain = 2.4

    # ------------------------------------------------------------------
    def scores(self, desire: np.ndarray) -> dict[str, np.ndarray]:
        s = {}
        tm = self.space.tagm
        for ch in ("head", "eyes", "lids", "mouth", "torso", "stance", "style", "intent",
                   "touch", "walk"):
            s[ch] = tm[ch] @ desire
        s["l_schema"] = s["r_schema"] = tm["arm_schema"] @ desire
        s["l_hand"] = s["r_hand"] = tm["hand_shape"] @ desire
        return s

    def sample(self, desire: np.ndarray, current: np.ndarray, n: int, *,
               ambient: bool, allow_walk: bool, tau: float, objects_ok: bool = True,
               ) -> list[np.ndarray]:
        sp = self.space
        rng = self.rng
        sc = self.scores(desire)
        ci = self.layout_idx
        P = AMBIENT_P if ambient else BASE_P
        mean_d = float(np.mean(np.clip(desire, 0, 1.5)))
        neutral = sp.neutral()
        out = []
        for _ in range(n):
            d = (current if (rng.random() < 0.30 and not ambient) else neutral).copy()
            for g, chans in GROUPS.items():
                p = P[g] * (0.55 + 1.6 * mean_d) if g not in ("hold", "style") else P[g]
                if g == "walk" and not allow_walk:
                    p = 0.0
                if g == "intent" and not objects_ok:
                    p = 0.0
                if rng.random() > min(p, 0.97):
                    continue
                main = chans[0]
                size = int(sp.layout[ci[main]][1])
                if g == "hold":
                    w = np.array([0.8, 1.2, 1.5, 1.2, 0.7, 0.4]) * (1.0 - 0.5 * np.clip(desire[TI["energetic"]], 0, 1)
                                                                    * np.array([-1, -0.3, 0, 0.3, 0.6, 1.0]))
                    d[ci[main]] = int(rng.choice(size, p=w / w.sum()))
                    continue
                key = main if main in sc else (g if g in sc else None)
                if g in ("l_arm", "r_arm", "l_hand", "r_hand"):
                    key = main
                logit = (self.gain * sc[key]) / max(tau, 0.1) + _gumbel(rng, size) \
                    if key is not None else _gumbel(rng, size)
                d[ci[main]] = int(np.argmax(logit))
                if g in ("l_arm", "r_arm"):
                    spread = 0.25 + 0.35 * float(np.clip(desire[TI["expressive"]] + desire[TI["energetic"]], 0, 1.5))
                    pv = np.array([0.08, 0.22, 0.40, 0.22, 0.08]) * (1 - spread) + spread / ARM_VAR
                    for j in range(6):
                        d[ci[chans[1 + j]]] = int(rng.choice(ARM_VAR, p=pv / pv.sum()))
                elif g in ("l_hand", "r_hand"):
                    pv = np.array([0.25, 0.5, 0.25])
                    for j in range(3):
                        d[ci[chans[1 + j]]] = int(rng.choice(HAND_VAR, p=pv))
            if ambient:
                # an ambient behaviour never goes for big, loud things
                d[ci["style"]] = np.ravel_multi_index(
                    (int(rng.integers(0, 3)), int(rng.integers(0, 2)), int(rng.integers(0, 2)),
                     int(rng.integers(0, 2))),
                    (len(STYLE_SPEED), len(STYLE_AMP), len(STYLE_TREMOR), len(STYLE_RHYTHM)))
            out.append(d)
        return out

    def affinity(self, d: np.ndarray, sc: dict) -> float:
        ci = self.layout_idx
        a = 0.0
        for ch in SCORED:
            a += float(sc[ch][int(d[ci[ch]])])
        return a


# ==========================================================================
class BehaviorExecutor:
    """Decides (at ~2 Hz) and executes (at 50 Hz) behaviours."""

    def __init__(self, agent, *, ambient_only: bool = False, seed: int = 0):
        self.ag = agent
        meta = agent.meta
        names = [n for _, n, _ in meta.joint_order]
        lo = np.array([j.lo for _, _, j in meta.joint_order])
        hi = np.array([j.hi for _, _, j in meta.joint_order])
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.q_nom = np.array(agent.motor.q_nom, float)
        self.space = BehaviorSpace(names, lo, hi, self.q_nom)
        self.selector = BehaviorSelector(self.space, seed)
        self.stats = BehaviorStats(self.space)
        self.rng = np.random.default_rng(seed + 4417)
        self.enabled = True
        self.ambient = bool(ambient_only)     # set by the agent from `autonomous`
        self.allow_walk = False
        self.n_candidates = 40
        # --- state of the running behaviour -------------------------------
        self.current: Behavior = self.space.compile(self.space.neutral())
        self.cur_desc = self.space.neutral()
        self.t_start = 0.0
        self.t_next = 0.5
        self.cmd = self.q_nom.copy()          # smoothed joint command
        self.cur_gaze = np.zeros(2)
        self.tremor_state = np.zeros(len(names))
        self.last_desire = np.zeros(NT)
        self.last_G = np.zeros(0)
        self.last_entropy = 0.0
        self._skill_started = False
        self._vocal_t = 0.0
        self._stand_h0: float | None = None
        self.owns_eyes = True
        self.jaw_active = False
        self.voice = 0.0
        self.owned_joints = np.array([not n.startswith(LEG_PREFIX) and not n.startswith("eye_")
                                      and n != "jaw_open" for n in names])
        self.history: list[tuple[float, str]] = []
        self._pending: Behavior | None = None

    # ------------------------------------------------------------------
    # desire: what the person's condition makes attractive
    # ------------------------------------------------------------------
    def desire(self) -> np.ndarray:
        ag = self.ag
        d = np.zeros(NT)
        a = ag.affect_frame
        dr = ag.drive_frame
        if a is None or dr is None:
            return d
        lv = dr.level
        em = a.emotion
        oc = ag.ocular.out if getattr(ag, "ocular", None) is not None else None
        fear = em("fear"); anx = em("anxiety"); sad = em("sadness"); joy = em("joy")
        bored = em("boredom"); curious = em("curiosity")
        social_pulse = getattr(ag.skills, "social_pulse", 0.0)
        d[TI["vigilance"]] = 1.2 * fear + 0.6 * anx + 0.3 * a.arousal
        d[TI["defensive"]] = 1.3 * fear + 0.9 * lv[5]
        d[TI["explore"]] = 0.9 * lv[13] + 0.6 * curious + 0.3 * bored
        d[TI["social"]] = 0.8 * lv[11] + 1.2 * social_pulse + 0.3 * joy
        d[TI["comfort"]] = lv[14] + 0.5 * lv[7]
        d[TI["fatigue"]] = 0.9 * lv[2] + 0.8 * lv[9]
        if oc is not None:
            d[TI["ocular"]] = 1.3 * oc.discomfort + 1.0 * oc.rub_urge + 0.6 * oc.blur
        d[TI["pain"]] = lv[5]
        d[TI["itch"]] = lv[10]
        d[TI["cold"]] = lv[3]
        d[TI["heat"]] = lv[4]
        d[TI["hunger"]] = 0.8 * lv[0] + 0.3 * lv[1]
        d[TI["boredom"]] = 0.8 * bored + 0.7 * lv[15] + 0.3 * (1.0 - a.arousal)
        d[TI["energetic"]] = max(0.0, a.arousal * (0.4 + a.valence + joy))
        d[TI["expressive"]] = 0.6 * joy + 0.6 * social_pulse + 0.3 * a.arousal
        d[TI["soothe"]] = 0.9 * anx + 0.8 * sad + 0.6 * a.stress
        d[TI["object"]] = 0.7 * lv[13] + 0.4 * lv[0]
        d[TI["relax"]] = max(0.0, 0.9 * (1.0 - a.arousal) * (0.4 + em("calm")))
        return np.clip(d, 0.0, 1.6)

    # ------------------------------------------------------------------
    def decide(self, now: float) -> Behavior:
        ag = self.ag
        desire = self.desire()
        self.last_desire = desire
        sel = self.selector
        a = ag.affect_frame
        arousal = float(a.arousal) if a is not None else 0.3
        tau = 0.55 + 0.7 * arousal + 0.3 * float(desire[TI["boredom"]])
        cands = sel.sample(desire, self.cur_desc, self.n_candidates, ambient=self.ambient,
                           allow_walk=self.allow_walk, tau=tau)
        # keep a few simple "do nothing special" options in the pool: stillness is a choice
        for _ in range(3):
            cands.append(self.space.neutral())
        sc = sel.scores(desire)
        latent, pref, prec = ag.latent, ag.preferred, ag.precision
        pred_mod = ag.predict
        qs = ag.latent_spec.offset["q"]
        n_j = ag.latent_spec.n_joints
        fwd = pred_mod.forward if pred_mod is not None else None
        ready = fwd is not None and fwd.n_updates > 40
        G = np.zeros(len(cands))
        compiled: list[Behavior] = []
        for i, d in enumerate(cands):
            b = self.space.compile(d)
            compiled.append(b)
            tgt = self.q_nom.copy()
            for nm, v in b.joints.items():
                tgt[self.idx[nm]] = v
            pred = latent.copy()
            pred[qs:qs + n_j] = 0.65 * tgt + 0.35 * latent[qs:qs + n_j]
            if ready:
                learned = pred_mod.imagine(latent, tgt)
                pred = 0.45 * pred + 0.55 * learned
            risk = 0.5 * float(np.sum(((pred - pref) ** 2) * prec))
            motor = 0.0015 * float(np.sum((tgt - self.cmd) ** 2)) * b.speed
            aff = sel.affinity(d, sc) * (0.6 if self.ambient else 1.0)
            # tremor / big amplitude cost a little; large hold means commitment
            G[i] = 0.55 * risk + motor + 0.08 * b.hold - sel.gain * aff \
                + 0.5 * b.tremor * 10
        G = G - G.min()
        temp = max(0.15, 0.9 * (0.6 + arousal))
        z = -G / temp
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        k = int(self.rng.choice(len(cands), p=p))
        self.last_G = G
        self.last_entropy = float(-(p * np.log(p + 1e-12)).sum())
        return compiled[k]

    # ------------------------------------------------------------------
    def begin(self, b: Behavior, now: float) -> None:
        self.current = b
        self.cur_desc = b.desc.copy()
        self.t_start = now
        self.t_next = now + b.hold
        self._skill_started = False
        self._vocal_t = 0.0
        self.stats.add(b)
        self.history.append((now, b.name))
        self.history = self.history[-50:]
        ag = self.ag
        oc = getattr(ag, "ocular", None)
        if oc is not None:
            if b.blink == "double":
                oc.blink_now(1.0, 1.3); oc.blink_now(0.9, 1.3)
            elif b.blink == "slow":
                oc.blink_now(1.0, 0.45)
            elif b.blink == "flutter":
                for _ in range(5):
                    oc.blink_now(0.7, 2.2)
            elif b.blink == "wink_left":
                oc.wink(0, 1.0)
            elif b.blink == "wink_right":
                oc.wink(1, 1.0)
        # skill-layer parts: they run as queued actions and own the arm meanwhile
        sk = ag.skills
        if b.touch is not None and not sk.busy:
            region, hand, action = b.touch
            sk.api_touch_self(region, hand, action)
            self._skill_started = True
        elif b.intent is not None and not sk.busy:
            mode, obj = b.intent
            try:
                if mode == "look":
                    sk.api_look_at(obj)
                elif mode == "point":
                    sk.api_point_at(obj)
                elif mode == "reach":
                    sk.api_reach(obj)
                elif mode == "grasp":
                    sk.api_grab(obj)
                self._skill_started = mode != "look"
            except Exception:
                pass
        if b.walk is not None and self.allow_walk and not sk.busy:
            ag.gait.walk(*b.walk)

    # ------------------------------------------------------------------
    def _style_envelope(self, b: Behavior, t: float) -> float:
        if b.rhythm == "steady":
            return 1.0
        if b.rhythm == "pulse":
            return 0.55 + 0.45 * math.tanh(3.0 * math.sin(2 * math.pi * 1.1 * t))
        if b.rhythm == "sway_slow":
            return 1.0 + 0.30 * math.sin(2 * math.pi * 0.45 * t)
        return 1.0 + 0.14 * math.sin(2 * math.pi * 2.6 * t)

    def step(self, dt: float, now: float) -> None:
        if not self.enabled:
            return
        ag = self.ag
        sk = ag.skills
        # ---- when to decide ------------------------------------------------
        if now >= self.t_next:
            busy_skill = sk.busy and self._skill_started
            if not busy_skill or now - self.t_start > self.current.hold + 12.0:
                if self._skill_started and not sk.busy:
                    self._finish_skill()
                self.begin(self.decide(now), now)
        elif self._skill_started and not sk.busy and now - self.t_start > 0.5:
            self._finish_skill()
            self._skill_started = False
        b = self.current
        t = now - self.t_start
        env = self._style_envelope(b, t)

        # ---- joint command ------------------------------------------------------
        owned = self.owned_joints.copy()
        for s in "lr":
            if sk.arm_busy(s) or sk.arm[s].active:
                for nm in self.names:
                    if nm.endswith(f"_{s}") or f"_{s}_" in nm:
                        if nm.startswith(("sh_", "elbow_", "wrist_")):
                            owned[self.idx[nm]] = False
            if sk.hands[s].owned:
                for nm in self.names:
                    if nm.startswith(("thumb_", "index_", "fingers_")) and nm.endswith(f"_{s}_mcp") \
                            or nm.startswith(("thumb_", "index_", "fingers_")) and nm.endswith(f"_{s}_pip"):
                        owned[self.idx[nm]] = False
        if sk.gaze is not None or sk.gesture_active:
            for nm in ("neck_twist", "neck_bend"):
                owned[self.idx[nm]] = False
        if sk.gesture_active:
            owned[:] = np.where(self.owned_joints & ~np.array(
                [nm in sk.gesture_targets for nm in self.names]), owned, False)
        amp = b.amp * (0.5 if self.ambient else 1.0)
        tgt = self.q_nom.copy()
        for nm, v in b.joints.items():
            i = self.idx[nm]
            tgt[i] = self.q_nom[i] + amp * env * (v - self.q_nom[i])
        # tremor: band-limited noise on the arms, hands and head
        if b.tremor > 0:
            self.tremor_state += (-self.tremor_state * dt * 18.0
                                  + self.rng.standard_normal(len(self.names)) * math.sqrt(dt) * 14.0 * b.tremor)
            mask = np.array([nm.startswith(("sh_", "elbow_", "wrist_", "thumb_", "index_", "fingers_", "neck_"))
                             for nm in self.names])
            tgt = tgt + self.tremor_state * mask
        if b.rhythm in ("sway_slow", "sway_fast"):
            w = 2 * math.pi * (0.45 if b.rhythm == "sway_slow" else 2.6)
            for nm in self.names:
                if nm.startswith(("neck_side", "spine_side", "chest_side")):
                    tgt[self.idx[nm]] += 0.05 * amp * math.sin(w * t)
        vmax = 1.3 * b.speed * (0.7 if self.ambient else 1.0)
        step = np.clip(tgt - self.cmd, -vmax * dt, vmax * dt)
        self.cmd = self.cmd + step
        hands = np.array([nm.startswith(("thumb_", "index_", "fingers_")) for nm in self.names])
        self.cmd[hands] = self.cmd[hands] + np.clip(tgt[hands] - self.cmd[hands], -2.5 * dt, 2.5 * dt) * 0 \
            + np.clip(tgt[hands] - self.cmd[hands], -vmax * 2.0 * dt, vmax * 2.0 * dt) * 0
        vt = ag.voluntary_target
        vt[owned] = np.clip(self.cmd[owned], ag.motor.q_lo[owned], ag.motor.q_hi[owned])

        # ---- eyes, jaw, voice ----------------------------------------------------
        gaze = np.array(b.gaze) * (0.6 if self.ambient else 1.0)
        # a fast saccade-like move, then hold, plus small fixational drift
        self.cur_gaze += np.clip(gaze - self.cur_gaze, -3.2 * dt, 3.2 * dt)
        self.eye_target = self.cur_gaze + 0.01 * self.rng.standard_normal(2)
        if (ag.autonomous or not sk.busy) and sk.gaze is None:
            for s in "lr":
                a_y = ag.meta.qpos_addr.get(f"eye_{s}_yaw")
                a_p = ag.meta.qpos_addr.get(f"eye_{s}_pitch")
                if a_y is not None:
                    ag.data.qpos[a_y] = float(self.eye_target[0])
                    ag.data.qpos[a_p] = float(self.eye_target[1])
        jaw, voice = self._vocal(b, t, dt)
        self.voice = voice
        ag.receptors.self_voice = max(float(sk.speech.jaw) * 0.0, voice)
        self.jaw_active = (jaw > 1e-3) and not sk.speech.speaking
        if self.jaw_active and sk.jaw_adr is not None:
            ag.data.qpos[sk.jaw_adr] = float(np.clip(jaw, 0.0, 0.42))
            ag.data.qvel[sk.jaw_dof] = 0.0

        # ---- stance ------------------------------------------------------------------
        g = ag.gait
        if g.hold_stance and not g.walking and not sk.busy:
            if self._stand_h0 is None:
                self._stand_h0 = float(g.stand_height or (g.diag.com[2] if g.diag.com is not None else 0.86) or 0.86)
            drop, lat, fwdb = b.stance
            drop *= (0.5 if self.ambient else 1.0)
            g.set_stand_height(self._stand_h0 + drop)
            g.stand_bias += np.clip(np.array([lat, fwdb]) - g.stand_bias, -0.02 * dt * 10, 0.02 * dt * 10)

    # ------------------------------------------------------------------
    def _finish_skill(self) -> None:
        sk = self.ag.skills
        for s in "lr":
            if sk.arm[s].active and sk.held[s] is None:
                sk.arm[s].begin_retract()
            if sk.hands[s].owned and sk.held[s] is None:
                sk.hands[s].owned = False
        if sk.gaze is not None and self.current.intent is not None and self.current.intent[0] == "look":
            sk.gaze = None

    def _vocal(self, b: Behavior, t: float, dt: float) -> tuple[float, float]:
        """(jaw angle, loudness) of the current vocalisation at time ``t``."""
        v = b.vocal
        jaw = b.jaw
        loud = 0.0
        if v == "none":
            return jaw, 0.0
        if v == "hum":
            jaw = max(jaw, 0.05 + 0.03 * math.sin(2 * math.pi * 2.0 * t)); loud = 0.35
        elif v == "sigh":
            e = math.exp(-t / 1.2); jaw = max(jaw, 0.20 * e); loud = 0.45 * e
        elif v == "cough":
            burst = max(0.0, math.sin(2 * math.pi * 3.0 * t)) ** 4 if t < 1.0 else 0.0
            jaw = max(jaw, 0.18 * burst); loud = 0.9 * burst
        elif v == "gasp":
            e = math.exp(-t / 0.5); jaw = max(jaw, 0.38 * e); loud = 0.5 * e
        elif v == "laugh":
            on = max(0.0, math.sin(2 * math.pi * 5.0 * t)) if t < 2.2 else 0.0
            jaw = max(jaw, 0.22 * on); loud = 0.8 * on
        elif v == "yawn":
            env = math.sin(math.pi * min(t / 3.2, 1.0)) if t < 3.2 else 0.0
            jaw = max(jaw, 0.42 * env); loud = 0.25 * env
            if env > 0.3 and getattr(self.ag, "ocular", None) is not None:
                self.ag.ocular.blink_now(1.0, 0.35) if self._vocal_t < 0.01 else None
                self._vocal_t = 1.0
        elif v == "murmur":
            jaw = max(jaw, 0.08 + 0.05 * abs(math.sin(2 * math.pi * 3.0 * t))); loud = 0.4
        elif v == "whistle":
            jaw = max(jaw, 0.04); loud = 0.5
        return jaw, loud

    # ------------------------------------------------------------------
    def current_name(self) -> str:
        return self.current.name

    def force(self, b: Behavior, now: float) -> None:
        """Run a given behaviour now (the mind's ``express``)."""
        self.begin(b, now)

    def summary(self) -> dict:
        s = self.stats.summary()
        s["current"] = self.current.name
        s["entropy"] = self.last_entropy
        s["space"] = self.space.describe()["descriptors_log10"]
        return s
