"""
Motor skills: everything the body can *do* on purpose.

The active-inference layer in :mod:`embodied_human.active_inference` chooses
among 55 fixed postures.  That is a reflex-level repertoire.  Deliberate action
-- go over there, pick that up, put it down, point, wave, say something --
needs closed-loop control towards things in the world, so it lives here, and
the mind (see :mod:`embodied_human.mind`) drives it through a small API.

======================  =======================================================
skill                   how it works
======================  =======================================================
walking                 :mod:`embodied_human.locomotion` (whole-body control);
                        ``walk_to`` closes the loop on position and heading,
                        and turns on the spot when it has to
head and eyes           ``look_at`` aims the neck, and the eyes take whatever
                        the neck cannot reach (sign convention verified)
reaching                damped-least-squares *Jacobian servo* on a point in the
                        palm: arm (6 joints) plus optionally the trunk, driven
                        by the *measured* hand pose so that sag under gravity
                        and moving targets are corrected for
fingers                 per-digit closure (thumb / index / the other three), two
                        phalanges each, closed under a torque limit
grasping                approach from the side, close until each digit touches,
                        squeeze a little, lift, and *check the object came
                        with the hand* before believing it
holding / carrying      the grip is held while the arm moves and while walking
putting down            find a free spot, lower, release, retract
gestures                wave, nod, shake the head, shrug, clap, thumbs-up,
                        point, think, scratch head, drink, bow
speech                  :mod:`embodied_human.speech` (jaw + words above head)
======================  =======================================================

Actions are written as Python generators: each ``yield`` is one control tick
(50 Hz), the body of the generator is the state machine.  A queue runs one
*physical* action at a time; speech and gaze are separate channels, because
people talk while they walk and look at things while they reach.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError("MuJoCo is required: python -m pip install mujoco") from exc

from .locomotion import fwd, left
from .speech import Speech
from .world import World, wrap

TICK = 0.02                    # skill control period (s)
DIGITS = ("thumb", "index", "fingers")
# maximal flexion of (MCP, PIP) per digit, rad (the joint limits are 1.45/1.60)
FINGER_MAX = {"thumb": (1.10, 1.25), "index": (1.35, 1.45), "fingers": (1.35, 1.50)}
# point in the hand frame, between the palm and the curled fingers (x towards
# the palm side), used as the end-effector of every reach
GRASP_SITE = np.array([0.030, 0.012, -0.060])
ARM_REACH = 0.54               # m from shoulder to the grasp site, at full stretch


class ActionFailed(Exception):
    pass


def _sx(side: str) -> float:
    return 1.0 if side == "l" else -1.0


def hand_frame(side: str, palm_normal: np.ndarray, finger_dir: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = hand axes in the world) with the palm facing
    ``palm_normal`` and the fingers pointing along ``finger_dir``.

    The palm faces -x of the hand frame on the left, +x on the right, the
    fingers point along -z, and the thumb is on the -y side.
    """
    z = -np.asarray(finger_dir, float)
    z /= np.linalg.norm(z)
    x = -_sx(side) * np.asarray(palm_normal, float)
    x = x - z * float(np.dot(x, z))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# --------------------------------------------------------------------------
# Hand
# --------------------------------------------------------------------------
@dataclass
class HandState:
    side: str
    c: dict = field(default_factory=lambda: {"thumb": 0.3, "index": 0.2, "fingers": 0.4})
    target: dict = field(default_factory=lambda: {"thumb": 0.3, "index": 0.2, "fingers": 0.4})
    rate: float = 3.0                # closure units per second
    owned: bool = False
    contact: dict = field(default_factory=dict)      # digit/palm -> normal force (N)
    contact_obj: str | None = None
    freeze: set = field(default_factory=set)         # digits that stopped on contact
    squeeze: float = 0.0

    def targets_as_joints(self) -> dict:
        sx = _sx(self.side)
        out = {}
        for dg in DIGITS:
            mc, pc = FINGER_MAX[dg]
            out[f"{dg}_{self.side}_mcp"] = sx * self.c[dg] * mc
            out[f"{dg}_{self.side}_pip"] = sx * self.c[dg] * pc
        return out


HAND_POSES = {
    "open": dict(thumb=0.0, index=0.0, fingers=0.0),
    "relaxed": dict(thumb=0.3, index=0.2, fingers=0.4),
    "fist": dict(thumb=0.9, index=1.0, fingers=1.0),
    "point": dict(thumb=0.7, index=0.0, fingers=1.0),
    "pinch": dict(thumb=0.55, index=0.55, fingers=0.3),
    "thumbs_up": dict(thumb=0.0, index=1.0, fingers=1.0),
    "ok": dict(thumb=0.6, index=0.65, fingers=0.0),
    "grip": dict(thumb=0.8, index=0.8, fingers=0.8),
}


# --------------------------------------------------------------------------
# Arm: analytic inverse kinematics + a sag-compensating servo
# --------------------------------------------------------------------------
L_UPPER = 0.255          # shoulder -> elbow (m)
L_FORE = 0.230           # elbow -> wrist


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class ArmServo:
    """Reach with one arm.

    A Jacobian-based servo was tried first and wandered: from a hanging arm it
    found a shoulder configuration (arm raised sideways, twisted) that could not
    get any closer and sat there.  A human arm is two links, so it has a closed
    form.  For a target hand pose:

    1. the wrist target is the palm site minus the hand's own offset;
    2. the elbow angle follows from the law of cosines on the shoulder-wrist
       distance;
    3. the elbow lies on a cone around the shoulder-wrist axis; the *swivel*
       angle on that cone is chosen to keep the elbow low and the joints
       inside their limits;
    4. the upper-arm direction gives shoulder flexion/abduction exactly
       (``a = (-cos f sin b, sin f, -cos f cos b)`` for the joint order used
       here), the forearm direction in the upper-arm frame gives the shoulder
       twist, and what is left of the hand orientation goes to the wrist.

    Gravity makes the real hand sag below the commanded point, so the measured
    error is integrated into the target (a small, bounded integral term).
    """

    TRUNK = ("spine_bend", "spine_twist", "chest_bend", "chest_twist")

    def __init__(self, sk: "SkillSystem", side: str):
        self.sk = sk
        self.side = side
        meta = sk.agent.meta
        self.names = [f"sh_{side}_abd", f"sh_{side}_flex", f"sh_{side}_rot",
                      f"elbow_{side}", f"wrist_{side}_flex", f"wrist_{side}_dev"]
        self.trunk = list(self.TRUNK)
        allj = self.names + self.trunk
        self.qadr = np.array([meta.qpos_addr[n] for n in allj])
        self.midx = np.array([sk.agent.motor.idx[n] for n in allj])
        self.lo = np.array([sk.agent.motor.q_lo[i] for i in self.midx])
        self.hi = np.array([sk.agent.motor.q_hi[i] for i in self.midx])
        self.hand_body = meta.body_ids[f"hand_{side}"]
        self.sh_body = meta.body_ids[f"shoulder_{side}"]
        self.chest_body = meta.body_ids["chest"]
        self.sx = _sx(side)
        self.site_local = np.array([-self.sx * GRASP_SITE[0], GRASP_SITE[1], GRASP_SITE[2]])
        self.active = False
        self.use_trunk = False
        self.w_ori = 1.0
        self.target_fn = None
        self.q_cmd = np.zeros(len(allj))
        self.err_pos = 9.9
        self.err_rot = 9.9
        self.integ = np.zeros(3)
        self.swivel = 0.0
        self.trunk_lean = 0.0
        self.retract = False

    # ------------------------------------------------------------------
    def site_world(self, d) -> tuple[np.ndarray, np.ndarray]:
        R = d.xmat[self.hand_body].reshape(3, 3)
        return d.xpos[self.hand_body] + R @ self.site_local, R

    def start(self, target_fn, *, use_trunk=False, w_ori=1.0) -> None:
        d = self.sk.agent.data
        self.target_fn = target_fn
        self.use_trunk = use_trunk
        self.w_ori = w_ori
        if not self.active:
            self.q_cmd = np.array(d.qpos[self.qadr], float)
            self.integ[:] = 0.0
            self.trunk_lean = float(d.qpos[self.qadr[6]])
        self.active = True
        self.err_pos = 9.9

    def stop(self) -> None:
        self.active = False
        self.target_fn = None
        self.retract = False

    def begin_retract(self) -> None:
        """Ease the arm back to its resting posture along a joint-space path.
        (Dropping the command at once swings the hand through whatever is in
        front of it -- it flicked a just-released apple across the room.)"""
        if self.active:
            self.retract = True
            self.target_fn = None

    def joint_targets(self) -> dict:
        out = {nm: float(self.q_cmd[i]) for i, nm in enumerate(self.names)}
        if self.use_trunk:
            out["spine_bend"] = float(self.q_cmd[6])
            out["chest_bend"] = float(self.q_cmd[8])
        return out

    # ------------------------------------------------------------------
    def solve(self, d, p_site: np.ndarray, R_t: np.ndarray) -> tuple[np.ndarray, float]:
        """Joint angles (6) for a palm-site pose.  Returns (q, shortfall) where
        shortfall is how far beyond reach the target was (m)."""
        # The wrist target depends on the hand's orientation, and the hand may
        # not be able to take the requested one (wrist limits).  Iterate: solve,
        # see what orientation was actually achieved, move the wrist target to
        # where *that* puts the palm site on the target, solve again.
        R_use = R_t
        q = None
        for _ in range(3):
            p_w = p_site - R_use @ self.site_local
            q, shortfall, R_use = self._solve_once(d, p_w, R_t)
        return q, shortfall

    def _solve_once(self, d, p_w: np.ndarray, R_t: np.ndarray):
        # NB: a MuJoCo body's frame already includes that body's own joints, so
        # xmat[shoulder] would contain the shoulder rotation we are solving
        # for.  The frame the shoulder joints are expressed in is the chest's.
        R_sh = d.xmat[self.chest_body].reshape(3, 3)
        p_sh = d.xpos[self.sh_body]
        v = R_sh.T @ (p_w - p_sh)
        D = float(np.linalg.norm(v))
        reach = L_UPPER + L_FORE
        Dc = float(np.clip(D, 0.14, 0.985 * reach))
        shortfall = max(0.0, D - 0.985 * reach)
        u = v / max(D, 1e-9)
        W = u * Dc
        cos_e = (Dc ** 2 - L_UPPER ** 2 - L_FORE ** 2) / (2 * L_UPPER * L_FORE)
        e = -math.acos(float(np.clip(cos_e, -1, 1)))
        cos_a = (L_UPPER ** 2 + Dc ** 2 - L_FORE ** 2) / (2 * L_UPPER * Dc)
        alpha = math.acos(float(np.clip(cos_a, -1, 1)))
        # elbow cone: perpendicular basis around u, hint = down and outward
        g = np.array([-self.sx * 0.35, 0.0, -1.0])
        perp0 = g - float(g @ u) * u
        if np.linalg.norm(perp0) < 1e-6:
            perp0 = np.array([1.0, 0.0, 0.0]) - u[0] * u
        perp0 /= np.linalg.norm(perp0)
        perp1 = np.cross(u, perp0)
        Rw = R_sh.T @ R_t
        lim_lo, lim_hi = self.lo[:6], self.hi[:6]
        best, best_cost = None, 1e9
        for psi in np.linspace(-1.5, 1.5, 21):
            perp = math.cos(psi) * perp0 + math.sin(psi) * perp1
            E = L_UPPER * (math.cos(alpha) * u + math.sin(alpha) * perp)
            a = E / L_UPPER
            flex = math.asin(float(np.clip(a[1], -1, 1)))
            cf = math.cos(flex)
            abd = math.atan2(-a[0], -a[2]) if cf > 1e-3 else 0.0
            Rua = _ry(abd) @ _rx(flex)
            f = (W - E) / L_FORE
            f_pre = Rua.T @ f
            rot = math.atan2(f_pre[0], -f_pre[1]) if abs(math.sin(e)) > 1e-3 else 0.0
            Rfore = Rua @ _rz(rot) @ _rx(e)
            M = Rfore.T @ Rw
            wf = math.atan2(-M[1, 2], M[2, 2])
            wd = math.asin(float(np.clip(M[0, 2], -1, 1)))
            q = np.array([abd, flex, rot, e, wf, wd])
            viol = np.maximum(0.0, np.maximum(q - lim_hi, lim_lo - q))
            wgt = np.array([1, 1, 1, 1, 0.5 * self.w_ori, 0.5 * self.w_ori])
            cost = float(np.sum(wgt * viol ** 2)) * 40.0 + 0.08 * (psi - self.swivel) ** 2
            if cost < best_cost:
                best, best_cost, best_psi = q, cost, psi
                best_Rfore = Rfore
        self.swivel += 0.4 * (best_psi - self.swivel)
        E_best = L_UPPER * (math.cos(alpha) * u + math.sin(alpha) * (
            math.cos(best_psi) * perp0 + math.sin(best_psi) * perp1))
        self.last_dbg = (D, e, alpha, best_psi, best_cost, best.copy(),
                         p_sh + R_sh @ W, p_w.copy(), p_sh + R_sh @ E_best)
        q = np.clip(best, lim_lo + 0.01, lim_hi - 0.01)
        R_ach = R_sh @ best_Rfore @ _rx(q[4]) @ _ry(q[5])
        return q, shortfall, R_ach

    # ------------------------------------------------------------------
    def update(self) -> None:
        if self.active and self.retract:
            nom = np.array([self.sk.q_nom_map[n] for n in self.names])
            diff = nom - self.q_cmd[:6]
            self.q_cmd[:6] += np.clip(diff, -0.9 * TICK, 0.9 * TICK)
            # the trunk joints simply follow where they are
            self.q_cmd[6:] = self.sk.agent.data.qpos[self.qadr[6:]]
            if float(np.abs(diff).max()) < 0.06:
                self.stop()
            return
        if not self.active or self.target_fn is None:
            return
        sk = self.sk
        d = sk.agent.data
        tgt = self.target_fn()
        if tgt is None:
            return
        p_t, R_t = tgt
        if R_t is None or self.w_ori <= 0.0:
            R_t = hand_frame(self.side, np.array([-self.sx, 0, 0]) * 0
                             + (sk._handshake_axes(self.side)[0]),
                             sk._handshake_axes(self.side)[1])
            ori_on = False
        else:
            ori_on = True
        p, R = self.site_world(d)
        e_p = np.asarray(p_t) - p
        self.err_pos = float(np.linalg.norm(e_p))
        e_r = 0.5 * (np.cross(R[:, 0], R_t[:, 0]) + np.cross(R[:, 1], R_t[:, 1])
                     + np.cross(R[:, 2], R_t[:, 2]))
        self.err_rot = float(np.linalg.norm(e_r)) if ori_on else 0.0
        # sag compensation: integrate the measured error (bounded)
        if self.err_pos < 0.10:
            self.integ = np.clip(self.integ + 0.25 * e_p, -0.07, 0.07)
        else:
            self.integ *= 0.9
        q, shortfall = self.solve(d, np.asarray(p_t) + self.integ, R_t)
        # trunk lean helps with what the arm cannot reach
        if self.use_trunk:
            # begin to lean once the arm is past a comfortable 88 % of its
            # reach, proportionally, so the lean settles where the arm just
            # reaches rather than where the target is merely "almost" in range
            p_w = np.asarray(p_t) - R_t @ self.site_local
            D = float(np.linalg.norm(p_w - d.xpos[self.sh_body]))
            want = float(np.clip((D - 0.88 * (L_UPPER + L_FORE)) / 0.16, 0.0, 0.50))
            # a deliberate lean is only allowed while the centre of mass stays
            # comfortably over the feet; past that, straighten up
            st = sk.agent.state
            if st is not None:
                fwd_margin = -float(st.com_over_support[1])        # + = COM ahead of the feet
                side_off = abs(float(st.com_over_support[0]))
                s_ok = float(np.clip(1.0 - (fwd_margin - 0.025) / 0.05, 0.0, 1.0)) * \
                    float(np.clip(1.0 - (side_off - 0.03) / 0.04, 0.0, 1.0))
                want *= s_ok
            self.trunk_lean += 0.10 * (want - self.trunk_lean)
            self.q_cmd[6] = 0.5 * self.trunk_lean + 0.0
            self.q_cmd[8] = 0.5 * self.trunk_lean
        else:
            self.q_cmd[6] = d.qpos[self.qadr[6]]
            self.q_cmd[8] = d.qpos[self.qadr[8]]
        step = np.clip(q - self.q_cmd[:6], -0.10, 0.10)
        self.q_cmd[:6] += step
        self.shortfall = shortfall

# --------------------------------------------------------------------------
# The skill system
# --------------------------------------------------------------------------
class SkillSystem:
    def __init__(self, agent, voice: bool = False):
        self.agent = agent
        self.world = World(agent)
        agent.skills = self          # World.visible_objects looks this up
        self.speech = Speech(voice=voice)
        self.hands = {s: HandState(s) for s in "lr"}
        self.arm = {s: ArmServo(self, s) for s in "lr"}
        self.held: dict[str, str | None] = {"l": None, "r": None}
        self.queue: list[tuple[str, tuple, object]] = []
        self.current: tuple[str, tuple, object] | None = None
        self.current_started = 0.0
        self._lock = threading.RLock()
        self.events: list[str] = []
        self.heard: list[str] = []            # speech from outside, for the mind
        self.social_pulse = 0.0               # decaying social stimulus, feeds affect
        self.touching = None                  # (region, side, action, phase) while self-touching
        self.gaze = None                      # None | ("point", xyz) | ("object", name)
        self.gaze_cur = np.zeros(4)           # neck yaw, neck pitch, eye yaw, eye pitch
        self.gesture_targets: dict[str, float] = {}
        self.gesture_active = False
        self.recovery_targets: dict[str, float] = {}
        self.recovery_active = False
        self._rec_hold_table = False
        self._rec_climbing = False
        self._auto_recover_t = -1e9
        self._recover_attempts = 0
        self.crouch = 0.0
        self.crouch_target = 0.0
        self.stand_height0 = None
        self.owned = np.zeros(agent.meta.n_actuators, bool)
        self._step = 0
        self.last_result = ""
        self.rng = np.random.default_rng(agent.cfg.seed + 5)
        self._idle_gaze = np.zeros(2)
        self._idle_t = 0.0
        m = agent.meta
        self.hand_geoms = {}
        for s in "lr":
            g = {"palm": [m.geom_ids[f"palm_{s}"]]}
            for dg in DIGITS:
                g[dg] = [m.geom_ids[f"{dg}_seg_{s}"], m.geom_ids[f"{dg}_dseg_{s}"]]
            self.hand_geoms[s] = g
        self.jaw_adr = m.qpos_addr.get("jaw_open")
        jid = mujoco.mj_name2id(agent.model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_open")
        self.jaw_dof = int(agent.model.jnt_dofadr[jid])
        self.jaw_act = mujoco.mj_name2id(agent.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                         "act_jaw_open")
        self.eye_adr = {(s, a): m.qpos_addr[f"eye_{s}_{a}"] for s in "lr"
                        for a in ("yaw", "pitch")}
        self.q_nom_map = {n: float(agent.motor.q_nom[i])
                          for i, n in enumerate(agent.motor.names)}
        self.motor_idx = agent.motor.idx
        # quick lookup tables
        self.neck_idx = (self.motor_idx["neck_twist"], self.motor_idx["neck_bend"])
        self.hand_joint_idx = {s: {j: self.motor_idx[j] for j in
                                   self.hands[s].targets_as_joints()} for s in "lr"}

    # ==================================================================
    # per-physics-step hooks
    # ==================================================================
    def hear(self, text: str) -> None:
        """Something is said to the person.  It is heard (audio-wise the cochlear
        model sees nothing, there is no acoustic field) and handed to the mind."""
        text = " ".join(str(text).split())[:300]
        if text:
            self.heard.append(text)
            self.social_pulse = 1.0

    def mute_jaw_actuator(self, ctrl: np.ndarray) -> None:
        """While the jaw is driven kinematically its motor must not push against
        that: the reaction of a 45 N m actuator holding the jaw shut shoves the
        head and, through the neck, topples the body."""
        beh = getattr(self.agent, "behavior", None)
        driven = beh is not None and getattr(beh, "jaw_active", False)
        if self.jaw_act >= 0 and (self.speech.speaking or self.speech.jaw > 1e-3 or driven):
            ctrl[self.jaw_act] = 0.0

    def update(self, dt: float) -> None:
        """Called every physics step by the agent."""
        self._step += 1
        self.social_pulse *= math.exp(-dt / 2.5)
        self.speech.update(dt)
        ag = self.agent
        # kinematic jaw
        if self.jaw_adr is not None and (self.speech.speaking or self.speech.jaw > 1e-3):
            ag.data.qpos[self.jaw_adr] = float(np.clip(self.speech.jaw, 0.0, 0.42))
            ag.data.qvel[self.jaw_dof] = 0.0
        n_sub = max(int(round(TICK / ag.dt)), 1)
        if self._step % n_sub == 0:
            self._tick()
        # fingers move every step (smooth)
        for s, h in self.hands.items():
            for dg in DIGITS:
                diff = h.target[dg] - h.c[dg]
                h.c[dg] += float(np.clip(diff, -h.rate * dt, h.rate * dt))

    def _tick(self) -> None:
        ag = self.agent
        m, d = ag.model, ag.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        self._scan_contacts()
        # action scheduler
        with self._lock:
            if self.current is None and self.queue:
                self.current = self.queue.pop(0)
                self.current_started = ag.t
            cur = self.current
        if cur is not None:
            name, args, gen = cur
            try:
                next(gen)
            except StopIteration:
                with self._lock:
                    if self.current is cur:
                        self.current = None
                self.last_result = f"{name}: done"
                self.events.append(f"finished {self._fmt(name, args)}")
            except ActionFailed as exc:
                self.last_result = f"{name}: failed - {exc}"
                self.events.append(f"FAILED {self._fmt(name, args)}: {exc}")
                with self._lock:
                    self.current = None
                    self.queue.clear()
                self._abort_motion()
            except ValueError:
                # the generator was cancelled from another thread mid-step
                pass
        for s in "lr":
            self.arm[s].update()
        self._update_gaze()
        # Fall recovery reflex: the body tries to get back up on its own when
        # down and idle, without waiting for the mind. The mind (or AZR) sees
        # the resulting events and verified outcomes like any other action.
        try:
            st = ag.state
            if (st is not None and st.fallen and not self.recovery_active
                    and self.current is None and not self.queue
                    and ag.t - self._auto_recover_t > 6.0):
                self._auto_recover_t = ag.t
                self._recover_attempts += 1
                self.events.append(
                    f"fallen: trying to get back up (attempt {self._recover_attempts})")
                self._enqueue("stand_up", (), self._a_stand_up())
        except Exception:
            pass

    @staticmethod
    def _fmt(name, args) -> str:
        a = ", ".join(repr(x) for x in args if x is not None)
        return f"{name}({a})"

    def _abort_motion(self) -> None:
        g = self.agent.gait
        g.stop()
        self.gesture_active = False
        self.gesture_targets = {}

    # ==================================================================
    def override_target(self, v: np.ndarray) -> np.ndarray:
        """Apply the skills' joint targets on top of the posture the
        active-inference layer chose.  Also returns the ownership mask."""
        v = v.copy()
        own = np.zeros_like(self.owned)
        for s in "lr":
            arm = self.arm[s]
            if arm.active:
                for nm, val in arm.joint_targets().items():
                    i = self.motor_idx[nm]
                    v[i] = val
                    own[i] = True
            h = self.hands[s]
            if h.owned:
                for nm, val in h.targets_as_joints().items():
                    i = self.motor_idx[nm]
                    v[i] = val
                    own[i] = True
        if self.gaze is not None or abs(self.gaze_cur[0]) > 1e-3 or abs(self.gaze_cur[1]) > 1e-3:
            v[self.neck_idx[0]] = self.gaze_cur[0]
            v[self.neck_idx[1]] = self.gaze_cur[1] + self.q_nom_map["neck_bend"]
            own[self.neck_idx[0]] = own[self.neck_idx[1]] = True
        if self.gesture_active:
            for nm, val in self.gesture_targets.items():
                i = self.motor_idx[nm]
                v[i] = val
                own[i] = True
        if self.recovery_active:
            # Fall recovery owns the whole body (legs, spine, arms): it must
            # beat the posture spring, the gait hold and any gesture.
            # Exception: while climbing furniture the arm servos own the arms
            # (they pin the hands to the table); recovery takes legs+trunk.
            # During hand-walk the servos own only the shoulders (hand
            # placement); recovery drives the elbows to coordinate the press.
            hold = getattr(self, "_rec_hold_table", False)
            table_climb = getattr(self, "_rec_climbing", False)
            for nm, val in list(self.recovery_targets.items()):
                if hold and (nm.startswith("sh_") or nm.startswith("wrist_") or
                             (table_climb and nm.startswith("elbow_"))):
                    continue
                i = self.motor_idx.get(nm)
                if i is not None:
                    v[i] = val
                    own[i] = True
        if self.crouch > 1e-3 and not self.agent.gait.hold_stance:
            c = self.crouch
            for s in "lr":
                for nm, val in ((f"knee_{s}", 0.80), (f"hip_{s}_flex", -0.60),
                                (f"ankle_{s}_flex", -0.30)):
                    i = self.motor_idx[nm]
                    v[i] = self.q_nom_map[nm] + c * (val - self.q_nom_map[nm])
                    own[i] = True
            i = self.motor_idx["spine_bend"]
            v[i] = self.q_nom_map["spine_bend"] + c * 0.20
            own[i] = True
        self.owned = own
        return v

    def lean_exempt(self) -> np.ndarray | None:
        """Trunk joints the arms are deliberately using to lean."""
        if getattr(self, "recovery_active", False):
            # Fall recovery drives the whole body through its stages; the
            # postural-priority attenuator (which fires constantly while down)
            # must not steal the legs and trunk back to nominal mid-stand-up.
            return np.ones(self.agent.meta.n_actuators, bool)
        if not any(self.arm[s].active and self.arm[s].use_trunk for s in "lr"):
            return None
        ex = np.zeros(self.agent.meta.n_actuators, bool)
        for nm in ("spine_bend", "chest_bend"):
            ex[self.motor_idx[nm]] = True
        return ex

    def arm_busy(self, side: str) -> bool:
        return self.arm[side].active or self.hands[side].owned and self.held[side] is not None \
            or self.gesture_active

    # ==================================================================
    # sensing helpers
    # ==================================================================
    def _scan_contacts(self) -> None:
        ag = self.agent
        m, d = ag.model, ag.data
        for s in "lr":
            self.hands[s].contact = {}
            self.hands[s].contact_obj = None
        gid2digit = {}
        for s in "lr":
            for key, gl in self.hand_geoms[s].items():
                for g in gl:
                    gid2digit[g] = (s, key)
        obj_gids = {self.world.geom[n]: n for n in self.world.objects}
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            for hg, og in ((a, b), (b, a)):
                if hg in gid2digit and og in obj_gids:
                    s, key = gid2digit[hg]
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, f)
                    if f[0] > 0.3:
                        h = self.hands[s]
                        h.contact[key] = max(h.contact.get(key, 0.0), float(f[0]))
                        h.contact_obj = obj_gids[og]

    # ------------------------------------------------------------------
    def _eq_id(self, side: str, obj: str) -> int:
        return mujoco.mj_name2id(self.agent.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                 f"grip_{side}_{obj}")

    def grip(self, side: str, obj: str, on: bool) -> None:
        """Switch the grip-assist weld between a hand and an object."""
        m, d = self.agent.model, self.agent.data
        eid = self._eq_id(side, obj)
        if eid < 0:
            return
        if on and not d.eq_active[eid]:
            b1 = self.agent.meta.body_ids[f"hand_{side}"]
            b2 = self.agent.meta.body_ids[f"{obj}_body"]
            R1 = d.xmat[b1].reshape(3, 3)
            rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
            q1n = np.zeros(4)
            mujoco.mju_negQuat(q1n, d.xquat[b1])
            qrel = np.zeros(4)
            mujoco.mju_mulQuat(qrel, q1n, d.xquat[b2])
            m.eq_data[eid, 0:3] = 0.0           # anchor at the object's origin
            m.eq_data[eid, 3:6] = rel
            m.eq_data[eid, 6:10] = qrel
            m.eq_data[eid, 10] = 1.0
            d.eq_active[eid] = 1
        elif not on:
            d.eq_active[eid] = 0

    def held_by(self, obj: str) -> str | None:
        for s, o in self.held.items():
            if o == obj:
                return "left hand" if s == "l" else "right hand"
        return None

    # ==================================================================
    # gaze
    # ==================================================================
    def _gaze_point(self) -> np.ndarray | None:
        if self.gaze is None:
            return None
        kind, val = self.gaze
        if kind == "point":
            return np.asarray(val, float)
        if kind == "object":
            return self.world.obj_pos(val)
        if kind == "hand":
            return self.arm[val].site_world(self.agent.data)[0]
        return None

    def _update_gaze(self) -> None:
        ag = self.agent
        d = ag.data
        w = self.world
        pt = self._gaze_point()
        if pt is None:
            # idle: the eyes wander a little, the neck settles to neutral -- unless
            # the behaviour layer is directing them
            beh = getattr(ag, "behavior", None)
            if beh is not None and beh.enabled and getattr(beh, "eye_target", None) is not None:
                want = np.array([0.0, 0.0, float(beh.eye_target[0]), float(beh.eye_target[1])])
            else:
                self._idle_t -= TICK
                if self._idle_t <= 0:
                    self._idle_gaze = self.rng.normal(0, 0.10, 2)
                    self._idle_t = float(self.rng.uniform(1.2, 3.5))
                want = np.array([0.0, 0.0, self._idle_gaze[0], self._idle_gaze[1]])
        else:
            # direction to the target in the chest frame
            R = d.xmat[w.chest].reshape(3, 3)
            v = R.T @ (pt - w.head_pos())
            yaw = math.atan2(v[0], -v[1])            # + = target on the left
            pitch = math.atan2(-v[2], math.hypot(v[0], v[1]))   # + = target below
            n_yaw = float(np.clip(0.75 * yaw, -0.85, 0.85))
            n_pit = float(np.clip(0.6 * pitch, -0.5, 0.6))
            e_yaw = float(np.clip(yaw - n_yaw, -0.7, 0.7))
            e_pit = float(np.clip(pitch - n_pit, -0.45, 0.45))
            want = np.array([n_yaw, n_pit, e_yaw, e_pit])
        k = 1.0 - math.exp(-TICK / 0.12)
        self.gaze_cur += k * (want - self.gaze_cur)
        if pt is None and ag.autonomous:
            return                      # leave the eyes to the active-inference gaze
        for s in "lr":
            d.qpos[self.eye_adr[(s, "yaw")]] = self.gaze_cur[2]
            d.qpos[self.eye_adr[(s, "pitch")]] = self.gaze_cur[3]

    # ==================================================================
    # the API the mind calls
    # ==================================================================
    def _enqueue(self, name: str, args: tuple, gen) -> None:
        with self._lock:
            self.queue.append((name, args, gen))

    def cancel_all(self) -> None:
        with self._lock:
            self.queue.clear()
            cur = self.current
            self.current = None
            # Defensive: if the generator's finally never runs (see below),
            # recovery must not own the body forever.
            self.recovery_active = False
            self._rec_hold_table = False
            self._rec_climbing = False
            if cur is not None:
                # Close the generator so try/finally blocks inside actions
                # (e.g. stand_up cleanup) actually run on cancel.
                try:
                    cur[2].close()
                except Exception:
                    pass
            self._abort_motion()
            for s in "lr":
                if self.held[s] is None:
                    self.arm[s].stop()

    @property
    def busy(self) -> bool:
        return self.current is not None or bool(self.queue)

    def current_description(self) -> str:
        if self.current is None:
            return "nothing"
        name, args, _ = self.current
        return self._fmt(name, args)

    # ---- speech / gaze (immediate channels) ---------------------------
    def api_say(self, text: str) -> None:
        self.speech.say(str(text))

    def api_look_at(self, target) -> None:
        spec = self.resolve(target)
        if spec is None:
            raise ActionFailed(f"I do not know what '{target}' is")
        kind, val = spec
        if kind == "object":
            self.gaze = ("object", val)
        elif kind == "surface":
            self.gaze = ("point", self.world.surface_point(val))
        elif kind == "hand":
            self.gaze = ("hand", val)
        elif kind == "point":
            self.gaze = ("point", np.array(val, float))
        elif kind == "ego":
            self.gaze = ("point", self.world.from_ego(val[0], val[1], val[2]))

    def api_look_forward(self) -> None:
        self.gaze = None

    def api_hand_pose(self, hand, pose) -> None:
        sides = self._sides(hand)
        if pose not in HAND_POSES:
            raise ActionFailed(f"unknown hand pose '{pose}' (try {', '.join(HAND_POSES)})")
        for s in sides:
            h = self.hands[s]
            h.owned = True
            h.freeze.clear()
            h.target.update(HAND_POSES[pose])

    # ---- queued physical actions --------------------------------------
    def api_wait(self, seconds=1.0):
        self._enqueue("wait", (seconds,), self._a_wait(float(seconds)))

    def api_walk_to(self, target, standoff=None):
        self._enqueue("walk_to", (target,), self._a_walk_to(target, standoff))

    def api_walk(self, distance):
        self._enqueue("walk", (distance,), self._a_walk(float(distance)))

    def api_turn(self, degrees):
        self._enqueue("turn", (degrees,), self._a_turn(math.radians(float(degrees))))

    def api_face(self, target):
        self._enqueue("face", (target,), self._a_face(target))

    def api_stop(self):
        self.cancel_all()

    def api_reach(self, target, hand=None):
        self._enqueue("reach", (target, hand), self._a_reach(target, hand))

    def api_grab(self, obj, hand=None):
        self._enqueue("grab", (obj, hand), self._a_grab(obj, hand))

    def api_release(self, hand=None):
        self._enqueue("release", (hand,), self._a_release(hand))

    def api_put_down(self, where="table", hand=None):
        self._enqueue("put_down", (where, hand), self._a_put_down(where, hand))

    def api_point_at(self, target, hand=None):
        self._enqueue("point_at", (target, hand), self._a_point(target, hand))

    def api_gesture(self, name, hand=None):
        if name not in GESTURES:
            raise ActionFailed(f"unknown gesture '{name}' (try {', '.join(GESTURES)})")
        self._enqueue("gesture", (name,), self._a_gesture(name, hand))

    def api_touch_self(self, region, hand=None, action="rest"):
        self._enqueue("touch_self", (region, hand, action), self._a_touch_self(region, hand, action))

    def api_express(self, **parts):
        """A whole-body expression from named parts (see behavior_space.option_names)."""
        beh = getattr(self.agent, "behavior", None)
        if beh is None:
            return
        errors = beh.express(**parts)
        for e in errors:
            self.events.append(f"(express: {e})")

    def api_rub_eyes(self, hand="right"):
        self.api_touch_self("eyes", hand, "rub")

    def api_scratch(self, part="scalp", hand="right"):
        self.api_touch_self(part, hand, "scratch")

    def api_blink(self, kind="normal"):
        oc = getattr(self.agent, "ocular", None)
        if oc is None:
            return
        if kind in ("slow",):
            oc.blink_now(1.0, 0.45)
        elif kind == "double":
            oc.blink_now(1.0, 1.3); oc.blink_now(0.9, 1.3)
        elif kind in ("wink_left", "wink"):
            oc.wink(0, 1.0)
        elif kind == "wink_right":
            oc.wink(1, 1.0)
        else:
            oc.blink_now(1.0, 1.0)

    def api_crouch(self, depth=0.7):
        self._enqueue("crouch", (depth,), self._a_crouch(float(depth)))

    def api_stand(self):
        self._enqueue("stand", (), self._a_crouch(0.0))

    def api_stand_up(self):
        """Get back up after a fall (staged tuck -> kneel -> stand)."""
        self._enqueue("stand_up", (), self._a_stand_up())

    # ==================================================================
    # target resolution
    # ==================================================================
    def resolve(self, target):
        """Turn what the mind said into ('object'|'surface'|'point'|'ego'|'hand', value)."""
        w = self.world
        if isinstance(target, str):
            t = target.strip().lower().replace(" ", "_")
            if t in w.objects:
                return ("object", t)
            for n, o in w.objects.items():
                if o.label and t.replace("_", " ") in o.label.lower():
                    return ("object", n)
            if t in w.furniture:
                return ("surface", t)
            if t in ("left_hand", "right_hand"):
                return ("hand", "l" if t.startswith("l") else "r")
            if t in ("table", "workbench", "desk"):
                return ("surface", "table")
            if t in ("shelf", "other_table", "far_table"):
                return ("surface", "shelf")
            if t in ("ball", "toy", "sphere", "small_ball"):
                return ("object", "sphere_toy")
            if t in ("coffee", "cup", "hot_mug"):
                return ("object", "mug")
            if t in ("rock",):
                return ("object", "stone")
            return None
        if isinstance(target, (tuple, list)) and len(target) == 2:
            return ("point", (float(target[0]), float(target[1]), 0.0))
        return None

    def _sides(self, hand) -> list[str]:
        if hand in (None, "", "any"):
            return ["l", "r"]
        h = str(hand).lower()
        if h.startswith("l"):
            return ["l"]
        if h.startswith("r"):
            return ["r"]
        return ["l", "r"]

    def _free_side_for(self, pos: np.ndarray, hand=None) -> str:
        if hand not in (None, "", "any"):
            return self._sides(hand)[0]
        free = [s for s in "lr" if self.held[s] is None]
        if not free:
            raise ActionFailed("both hands are full")
        if len(free) == 1:
            return free[0]
        _, l, _ = self.world.to_ego(pos)
        return "l" if l >= 0 else "r"

    # ==================================================================
    # actions (generators)
    # ==================================================================
    def _a_wait(self, seconds):
        n = int(max(seconds, 0.0) / TICK)
        for _ in range(n):
            yield

    def _fell(self) -> bool:
        st = self.agent.state
        return bool(st is not None and st.fallen)

    def _goal_for(self, spec, standoff):
        w = self.world
        kind, val = spec
        pos = w.body_pos()[:2]
        if kind == "object":
            op = w.obj_pos(val)
            on = w.surface_of(op, w.obj_radius(val))
            if on in w.furniture and standoff is None:
                # an object on furniture: stand at the near edge, in line with it
                f = w.furniture[on]
                side = 1.0 if pos[1] > f["y"] else -1.0
                goal = np.array([float(np.clip(op[0], f["x"] - f["hx"] + 0.12,
                                               f["x"] + f["hx"] - 0.12)),
                                 f["y"] + side * (f["hy"] + 0.20)])
                return goal, op[:2]
            p = op[:2]
            so = 0.50 if standoff is None else float(standoff)
        elif kind == "surface":
            f = w.furniture[val]
            p = np.array([float(np.clip(pos[0], f["x"] - f["hx"] + 0.15, f["x"] + f["hx"] - 0.15)),
                          f["y"]])
            so = (f["hy"] + 0.20) if standoff is None else float(standoff)
        elif kind == "point":
            p = np.array(val[:2], float)
            so = 0.0 if standoff is None else float(standoff)
        elif kind == "ego":
            q = w.from_ego(val[0], val[1])
            p = q[:2]
            so = 0.0
        else:
            raise ActionFailed("cannot walk to that")
        v = p - pos
        dist = float(np.linalg.norm(v))
        if dist < 1e-6:
            return pos, p
        goal = p - v / dist * min(so, dist)
        return goal, p

    def _turn_to(self, psi_des_fn, tol=0.10, timeout=8.0):
        g = self.agent.gait
        t0 = self.agent.t
        while True:
            e = wrap(psi_des_fn() - self.world.heading())
            if abs(e) < tol:
                break
            g.walk(0.0, float(np.clip(1.3 * e, -0.55, 0.55)))
            if self._fell():
                raise ActionFailed("I fell over")
            if self.agent.t - t0 > timeout:
                break
            yield
        g.stop()
        yield from self._wait_gait_idle()

    def _wait_gait_idle(self, timeout=4.0):
        g = self.agent.gait
        t0 = self.agent.t
        g.stop()
        while g.walking and self.agent.t - t0 < timeout:
            if self._fell():
                raise ActionFailed("I fell over")
            yield

    def _a_walk_to(self, target, standoff):
        spec = self.resolve(target)
        if spec is None:
            raise ActionFailed(f"I do not know where '{target}' is")
        g = self.agent.gait
        w = self.world
        goal, face_pt = self._goal_for(spec, standoff)
        d0 = float(np.linalg.norm(goal - w.body_pos()[:2]))
        t0 = self.agent.t
        tol = 0.10
        while True:
            goal, face_pt = self._goal_for(spec, standoff)
            pos = w.body_pos()[:2]
            delta = goal - pos
            dist = float(np.linalg.norm(delta))
            if dist < tol:
                break
            des = math.atan2(delta[0], -delta[1])
            e = wrap(des - w.heading())
            if abs(e) > 0.45 and dist > 0.25:
                g.walk(0.0, float(np.clip(1.3 * e, -0.55, 0.55)))
            else:
                bal = float(getattr(self.agent.motor, "balance_error", 0.0) or 0.0)
                slow = float(max(0.35, 1.0 - 8.0 * bal))
                speed = float(np.clip(0.9 * dist, 0.15, 0.36)) * slow
                g.walk(speed, float(np.clip(1.4 * e, -0.35, 0.35)) * slow)
            if self._fell():
                raise ActionFailed("I fell over")
            bal = float(getattr(self.agent.motor, "balance_error", 0.0) or 0.0)
            if bal > 0.12:
                bad = getattr(self, "_walk_bad", 0.0) + TICK
                self._walk_bad = bad
            else:
                self._walk_bad = 0.0
            if getattr(self, "_walk_bad", 0.0) > 0.6:
                g.stop()
                self._walk_bad = 0.0
                yield from self._wait_gait_idle()
                raise ActionFailed(
                    f"lost balance ({bal:.3f} m), stopped to recover")
            if self.agent.t - t0 > 12.0 + 6.0 * d0:
                g.stop()
                raise ActionFailed("could not get there in time")
            yield
        g.stop()
        yield from self._wait_gait_idle()
        yield from self._turn_to(lambda: self._heading_to(face_pt))

    def _heading_to(self, p) -> float:
        v = np.asarray(p[:2]) - self.world.body_pos()[:2]
        return math.atan2(v[0], -v[1])

    def _a_walk(self, distance):
        g = self.agent.gait
        w = self.world
        psi0 = w.heading()
        start = w.body_pos()[:2].copy()
        sgn = 1.0 if distance >= 0 else -1.0
        t0 = self.agent.t
        while True:
            travelled = float(np.dot(w.body_pos()[:2] - start, fwd(psi0)))
            remaining = (abs(distance) - sgn * travelled)
            if remaining < 0.06:
                break
            e = wrap(psi0 - w.heading())
            g.walk(sgn * float(np.clip(1.0 * remaining, 0.12, 0.34 if sgn > 0 else 0.2)),
                   float(np.clip(1.4 * e, -0.3, 0.3)))
            if self._fell():
                raise ActionFailed("I fell over")
            if self.agent.t - t0 > 10.0 + 6.0 * abs(distance):
                break
            yield
        yield from self._wait_gait_idle()

    def _a_turn(self, rad):
        psi0 = self.world.heading()
        yield from self._turn_to(lambda: psi0 + rad)

    def _a_face(self, target):
        spec = self.resolve(target)
        if spec is None:
            raise ActionFailed(f"I do not know where '{target}' is")
        kind, val = spec
        w = self.world

        def pt():
            if kind == "object":
                return w.obj_pos(val)
            if kind == "surface":
                return w.surface_point(val)
            return np.array(val if kind == "point" else w.from_ego(val[0], val[1]))
        yield from self._turn_to(lambda: self._heading_to(pt()))

    # ------------------------------------------------------------------
    def _arm_target_fn(self, side, pos_fn, normal_fn=None, finger_fn=None, w_ori=1.0):
        """Closure producing (position, rotation) each tick for the servo."""
        def fn():
            p = pos_fn()
            if normal_fn is None or w_ori <= 0:
                return p, None
            return p, hand_frame(side, normal_fn(), finger_fn())
        return fn

    def _handshake_axes(self, side):
        """Palm normal and finger direction for a sideways power grasp."""
        psi = self.world.heading()
        l = left(psi)
        n = -l if side == "l" else l
        n3 = np.array([n[0], n[1], 0.0])
        f2 = fwd(psi)
        return n3, np.array([f2[0], f2[1], 0.0])

    def _a_reach(self, target, hand):
        spec = self.resolve(target)
        if spec is None:
            raise ActionFailed(f"I do not know where '{target}' is")
        w = self.world
        kind, val = spec

        def pos():
            if kind == "object":
                return w.obj_pos(val)
            if kind == "surface":
                return w.surface_point(val) + np.array([0, 0, 0.04])
            if kind == "ego":
                return w.from_ego(*val)
            return np.array(val, float)
        p = pos()
        side = self._free_side_for(p, hand)
        self._check_reach(side, p, allow_trunk=True)
        arm = self.arm[side]
        arm.start(self._arm_target_fn(
            side, pos, lambda: self._handshake_axes(side)[0],
            lambda: self._handshake_axes(side)[1], w_ori=0.5),
            use_trunk=True, w_ori=0.5)
        t0 = self.agent.t
        while arm.err_pos > 0.025 and self.agent.t - t0 < 6.0:
            yield
        self.events.append(f"reached with the {'left' if side == 'l' else 'right'} hand "
                           f"(error {arm.err_pos * 100:.0f} cm)")
        for _ in range(int(0.4 / TICK)):
            yield

    def _shoulder_pos(self, side) -> np.ndarray:
        d = self.agent.data
        return d.xpos[self.agent.meta.body_ids[f"shoulder_{side}"]].copy()

    def _check_reach(self, side, p, allow_trunk=False):
        dist = float(np.linalg.norm(p - self._shoulder_pos(side)))
        lim = ARM_REACH + (0.18 if allow_trunk else 0.0)
        if dist > lim:
            raise ActionFailed(f"that is {dist:.2f} m from my shoulder - out of reach; "
                               f"walk closer first")

    # ------------------------------------------------------------------
    def _a_grab(self, obj, hand):
        spec = self.resolve(obj)
        if spec is None or spec[0] != "object":
            raise ActionFailed(f"I do not see anything called '{obj}'")
        name = spec[1]
        w = self.world
        if self.held_by(name):
            raise ActionFailed(f"I am already holding the {name}")
        p = w.obj_pos(name)
        side = self._free_side_for(p, hand)
        # not close enough: walk there first
        if np.linalg.norm(p - self._shoulder_pos(side)) > ARM_REACH + 0.12:
            yield from self._a_walk_to(name, None)
            p = w.obj_pos(name)
            side = self._free_side_for(p, hand)
        self._check_reach(side, p, allow_trunk=True)
        h = self.hands[side]
        arm = self.arm[side]
        r_obj = w.obj_radius(name)
        for attempt in range(2):
            h.owned = True
            h.freeze.clear()
            h.squeeze = 0.0
            h.target.update(HAND_POSES["open"])
            sgn = 1.0 if side == "l" else -1.0
            # approach from above: the hand must clear the table's edge before
            # it can come down beside the object
            hover = {"v": 0.17}
            # the hand starts low by the hip: lift it clear of the table edge
            # first, hold it back toward the body, then bring it out over the table
            stage = {"back": 0.20, "up": 0.10}
            # once the object is welded to the hand, chasing the object's position
            # would be a feedback loop (it moves with the hand): hold the hand where
            # it took hold instead
            anchor = {"p": None}

            def site_target(offset):
                def fn():
                    if anchor["p"] is not None:
                        return anchor["p"].copy()
                    o = w.obj_pos(name)
                    l2 = left(w.heading())
                    f2 = fwd(w.heading())
                    z = o[2]
                    on = w.surface_of(o, r_obj)
                    if on in w.furniture:
                        # the palm is 9 cm tall: keep it off the table top
                        z = max(z, w.furniture[on]["top"] + 0.064)
                    return np.array([o[0], o[1], z]) + \
                        np.array([l2[0], l2[1], 0.0]) * sgn * offset + \
                        np.array([0, 0, hover["v"] + stage["up"]]) - \
                        np.array([f2[0], f2[1], 0.0]) * stage["back"]
                return fn
            normal_fn = lambda: self._handshake_axes(side)[0]
            finger_fn = lambda: self._handshake_axes(side)[1]
            self.phase = 'pregrasp'
            # 1. pre-grasp, beside the object
            off = {"v": 0.09 + r_obj}
            arm.start(self._arm_target_fn(side, lambda: site_target(off["v"])(),
                                          normal_fn, finger_fn), use_trunk=True, w_ori=0.8)
            t0 = self.agent.t
            while (arm.err_pos > 0.035 or arm.err_rot > 0.55 or stage["back"] > 0.0
                   or stage["up"] > 0.0) and self.agent.t - t0 < 9.0:
                self._check_object_ok(name)
                on = w.surface_of(w.obj_pos(name), r_obj)
                clear = w.furniture[on]["top"] + 0.13 if on in w.furniture else 0.0
                if arm.err_pos < 0.10 or arm.site_world(self.agent.data)[0][2] > clear:
                    stage["back"] = max(0.0, stage["back"] - 0.22 * TICK)
                    if stage["back"] <= 0.0:
                        stage["up"] = max(0.0, stage["up"] - 0.10 * TICK)
                yield
            self.phase = 'slide'
            # 2. slide in until the object is in the palm
            t0 = self.agent.t
            # come down beside the object first, only then slide the palm in
            while (off["v"] > 0.0 or hover["v"] > 0.0) and self.agent.t - t0 < 7.0:
                if hover["v"] > 0.0:
                    hover["v"] = max(0.0, hover["v"] - 0.09 * TICK)
                else:
                    off["v"] = max(0.0, off["v"] - 0.06 * TICK)
                    if h.contact.get("palm", 0) > 1.0 and h.contact_obj == name:
                        off["v"] += 0.015            # ease off: do not press into it
                        break
                yield
            # the digits have no opposing thumb, so they would squeeze the object
            # out: take hold (assist weld) before they close, if it is in the palm
            if (h.contact and np.linalg.norm(w.obj_pos(name) - arm.site_world(
                    self.agent.data)[0]) < 0.07):
                self.grip(side, name, True)
                anchor["p"] = arm.site_world(self.agent.data)[0].copy()
            for _ in range(int(0.25 / TICK)):
                yield
            self.phase = 'close'
            # 3. close each digit until it touches, then squeeze
            t0 = self.agent.t
            while self.agent.t - t0 < 2.5:
                done = True
                for dg in DIGITS:
                    if dg in h.freeze:
                        continue
                    if h.contact.get(dg, 0.0) > 1.5:
                        h.freeze.add(dg)
                        h.target[dg] = min(1.0, h.c[dg] + 0.18)
                    else:
                        h.target[dg] = min(1.0, h.target[dg] + 0.7 * TICK)
                        done = False if h.target[dg] < 1.0 else done
                touching = sum(1 for dg in DIGITS if dg in h.freeze)
                if touching >= 2 or (touching >= 1 and h.contact.get("palm", 0) > 1.0
                                     and self.agent.t - t0 > 1.0):
                    break
                self._check_object_ok(name)
                yield
            for _ in range(int(0.4 / TICK)):
                yield
            touching = sum(1 for dg in DIGITS if h.contact.get(dg, 0.0) > 0.5)
            if h.contact_obj == name and (touching >= 2 or (touching >= 1 and
                                                            h.contact.get("palm", 0) > 0.5)):
                self.grip(side, name, True)          # grasp established
            self.phase = 'lift'
            # 4. lift and check the object comes along
            z0 = w.obj_pos(name)[2]
            t0 = self.agent.t
            lift = {"v": 0.0}
            arm.target_fn = self._arm_target_fn(
                side, lambda: site_target(0.0)() + np.array([0, 0, lift["v"]]),
                normal_fn, finger_fn)
            while lift["v"] < 0.06 and self.agent.t - t0 < 3.0:
                lift["v"] += 0.10 * TICK
                yield
            for _ in range(int(0.3 / TICK)):
                yield
            if w.obj_pos(name)[2] > z0 + 0.03 and h.contact_obj == name:
                self.held[side] = name
                self.events.append(f"I am holding the {name} in my "
                                   f"{'left' if side == 'l' else 'right'} hand")
                yield from self._carry_pose(side)
                return
            # failed: open and try again
            self.grip(side, name, False)
            h.target.update(HAND_POSES["open"])
            h.freeze.clear()
            for _ in range(int(0.5 / TICK)):
                yield
        arm.stop()
        h.owned = False
        raise ActionFailed(f"I could not get a grip on the {name}")

    def _check_object_ok(self, name):
        p = self.world.obj_pos(name)
        if p[2] < 0.2:
            raise ActionFailed(f"the {name} fell on the floor")

    def _carry_pose(self, side):
        """Bring the held object in front of the chest and keep it there."""
        w = self.world
        arm = self.arm[side]
        sgn = 1.0 if side == "l" else -1.0
        name = self.held[side]

        def pos():
            psi = w.heading()
            b = w.body_pos()
            xy = b[:2] + 0.34 * fwd(psi) + 0.12 * sgn * left(psi)
            return np.array([xy[0], xy[1], b[2] + 0.20])
        arm.target_fn = self._arm_target_fn(
            side, pos, lambda: self._handshake_axes(side)[0],
            lambda: self._handshake_axes(side)[1], w_ori=0.6)
        arm.w_ori = 0.6
        t0 = self.agent.t
        while arm.err_pos > 0.05 and self.agent.t - t0 < 3.0:
            if name and w.obj_pos(name)[2] < 0.3:
                self.held[side] = None
                raise ActionFailed(f"I dropped the {name}")
            yield

    # ------------------------------------------------------------------
    def _a_release(self, hand):
        sides = [s for s in self._sides(hand)]
        for s in sides:
            h = self.hands[s]
            h.owned = True
            h.freeze.clear()
            if self.held[s]:
                self.grip(s, self.held[s], False)
            h.target.update(HAND_POSES["open"])
        for _ in range(int(0.7 / TICK)):
            yield
        for s in sides:
            if self.held[s]:
                self.events.append(f"I let go of the {self.held[s]}")
            self.held[s] = None
        # lift the hand clear, then ease the arm back down by the side
        lifted = []
        for s in sides:
            arm = self.arm[s]
            if arm.active and arm.target_fn is not None:
                base_fn = arm.target_fn

                def up(base_fn=base_fn):
                    p, R = base_fn()
                    return p + np.array([0, 0, 0.14]), R
                arm.target_fn = up
                lifted.append(s)
        for _ in range(int((0.9 if lifted else 0.1) / TICK)):
            yield
        for s in sides:
            self.arm[s].begin_retract()
            self.hands[s].target.update(HAND_POSES["relaxed"])
        t0 = self.agent.t
        while any(self.arm[s].active for s in sides) and self.agent.t - t0 < 4.0:
            yield
        for s in sides:
            self.arm[s].stop()
            self.hands[s].owned = False

    def _a_put_down(self, where, hand):
        w = self.world
        sides = [s for s in "lr" if self.held[s]]
        if hand not in (None, "", "any"):
            sides = [s for s in self._sides(hand) if self.held[s]]
        if not sides:
            raise ActionFailed("I am not holding anything")
        side = sides[0]
        name = self.held[side]
        spec = self.resolve(where) or ("surface", "table")
        if spec[0] == "surface":
            sname = spec[1]
        elif spec[0] == "object":
            sname = w.surface_of(w.obj_pos(spec[1]), w.obj_radius(spec[1])) or "table"
        else:
            sname = "floor"
        if sname == "floor":
            raise ActionFailed("I can only put things down on the table or the shelf")
        f = w.furniture[sname]
        # must be standing at the furniture
        pos = w.body_pos()[:2]
        if abs(pos[1] - f["y"]) > f["hy"] + 0.75 or abs(pos[0] - f["x"]) > f["hx"] + 0.5:
            yield from self._a_walk_to(sname, None)
        r_obj = w.obj_radius(name)
        near = w.from_ego(0.33, 0.18 * (1 if side == "l" else -1))
        spot = w.nearest_free_spot(sname, near, r_obj)
        arm = self.arm[side]
        lift = {"z": 0.09}
        sgn = 1.0 if side == "l" else -1.0
        # the grasp site is at the object's centre
        arm.start(self._arm_target_fn(
            side, lambda: np.array([spot[0], spot[1], spot[2] + r_obj + 0.004 + lift["z"]]),
            lambda: self._handshake_axes(side)[0], lambda: self._handshake_axes(side)[1],
            w_ori=0.6), use_trunk=True, w_ori=0.6)
        t0 = self.agent.t
        while arm.err_pos > 0.03 and self.agent.t - t0 < 7.0:
            yield
        while lift["z"] > 0.0 and self.agent.t - t0 < 10.0:
            lift["z"] = max(0.0, lift["z"] - 0.06 * TICK)
            yield
        for _ in range(int(0.3 / TICK)):
            yield
        yield from self._a_release(side)
        self.events.append(f"I put the {name} down on the {sname}")

    # ------------------------------------------------------------------
    def _a_point(self, target, hand):
        spec = self.resolve(target)
        if spec is None:
            raise ActionFailed(f"I do not know where '{target}' is")
        w = self.world
        kind, val = spec

        def tpos():
            if kind == "object":
                return w.obj_pos(val)
            if kind == "surface":
                return w.surface_point(val)
            if kind == "ego":
                return w.from_ego(*val)
            return np.array(val, float)
        p = tpos()
        side = self._sides(hand)[0] if hand not in (None, "", "any") else \
            ("l" if w.to_ego(p)[1] >= 0 else "r")
        h = self.hands[side]
        h.owned = True
        h.freeze.clear()
        h.target.update(HAND_POSES["point"])
        arm = self.arm[side]

        def hand_pos():
            sh = self._shoulder_pos(side)
            v = tpos() - sh
            v = v / max(np.linalg.norm(v), 1e-6)
            return sh + v * 0.58

        def finger():
            sh = self._shoulder_pos(side)
            v = tpos() - sh
            return v / max(np.linalg.norm(v), 1e-6)

        def normal():
            f = finger()
            up = np.array([0, 0, 1.0])
            n = np.cross(finger(), np.cross(up, f))
            return -n / max(np.linalg.norm(n), 1e-6) if side == "l" else \
                n / max(np.linalg.norm(n), 1e-6)
        arm.start(self._arm_target_fn(side, hand_pos, normal, finger), use_trunk=False,
                  w_ori=0.35)
        self.gaze = ("point", p)
        t0 = self.agent.t
        while arm.err_pos > 0.04 and self.agent.t - t0 < 5.0:
            yield
        for _ in range(int(1.6 / TICK)):
            yield
        arm.stop()
        h.target.update(HAND_POSES["relaxed"])
        for _ in range(int(0.6 / TICK)):
            yield
        h.owned = False

    # ------------------------------------------------------------------
    def _a_touch_self(self, region, hand, action):
        """Bring a hand to a part of the person's own body and rest on it, rub it,
        tap it or scratch it: the hand that goes to a dry eye, an itchy nose, a
        cold upper arm."""
        from .behavior_space import SELF_ACTIONS, SELF_REGIONS
        table = {r[0]: r for r in SELF_REGIONS}
        key = str(region).lower().replace(" ", "_")
        if key not in table:
            raise ActionFailed(f"I do not know a body part called '{region}' "
                               f"(try {', '.join(table)})")
        if action not in SELF_ACTIONS:
            raise ActionFailed(f"unknown way of touching '{action}' (try {', '.join(SELF_ACTIONS)})")
        name, body, local, _ = table[key]
        h = str(hand).lower() if hand else ""
        side = "l" if h in ("l", "left") else "r" if h in ("r", "right") else "r"
        d = self.agent.data
        ids = self.agent.meta.body_ids
        loc = np.array(local, float)
        base = body.rsplit("_", 1)[0] if body.endswith(("_l", "_r")) else body
        if body.endswith(("_l", "_r")):
            opposite = name in ("upper_arm", "forearm")
            bside = ("r" if side == "l" else "l") if opposite else side
            bname = f"{base}_{bside}"
        else:
            bname = body
            if side == "r":
                loc[0] = -loc[0]               # the anchors are given for the left side
        bid = ids[bname]
        arm = self.arm[side]
        hs = self.hands[side]
        self.touching = (name, side, action, "reach")

        def anchor():
            return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ loc
        off = {"v": np.zeros(3)}
        arm.start(self._arm_target_fn(side, lambda: anchor() + off["v"], None, None,
                                      w_ori=0.0), use_trunk=name in ("thigh", "hip", "abdomen"),
                  w_ori=0.0)
        hs.owned = True
        hs.freeze.clear()
        hs.target.update(HAND_POSES["relaxed"] if action != "scratch" else HAND_POSES["pinch"])
        t0 = self.agent.t
        while arm.err_pos > 0.035 and self.agent.t - t0 < 4.5:
            yield
        self.touching = (name, side, action, "act")
        R = d.xmat[bid].reshape(3, 3)
        e1, e2 = R[:, 0], R[:, 2]
        n = R[:, 1]
        dur = {"rest": 1.6, "rub": 2.2, "tap": 1.6, "scratch": 2.4}[action]
        t1 = self.agent.t
        while self.agent.t - t1 < dur:
            tt = self.agent.t - t1
            if action == "rub":
                off["v"] = 0.012 * (e1 * math.cos(2 * math.pi * 2.8 * tt) + e2 * math.sin(2 * math.pi * 2.8 * tt))
            elif action == "tap":
                off["v"] = -0.014 * n * max(0.0, math.sin(2 * math.pi * 3.5 * tt))
            elif action == "scratch":
                off["v"] = 0.009 * e2 * math.sin(2 * math.pi * 6.0 * tt) + 0.004 * e1 * math.sin(2 * math.pi * 3.0 * tt)
            yield
        off["v"] = np.zeros(3)
        self.touching = (name, side, action, "retract")
        if self.held[side] is None:
            arm.begin_retract()
            t2 = self.agent.t
            while arm.active and self.agent.t - t2 < 3.0:
                yield
            if arm.active:
                arm.stop()
            hs.target.update(HAND_POSES["relaxed"])
            for _ in range(int(0.4 / TICK)):
                yield
            hs.owned = False
        self.touching = None

    def _a_crouch(self, depth):
        depth = float(np.clip(depth, 0.0, 1.0))
        if self.agent.gait.walking:
            yield from self._wait_gait_idle()
        g = self.agent.gait
        if g.hold_stance:
            # the whole-body controller lowers the centre of mass itself
            if self.stand_height0 is None:
                self.stand_height0 = float(g.stand_height or g.diag.com[2] or 0.86)
            g.set_stand_height(self.stand_height0 - 0.20 * depth)
            self.crouch = depth
            for _ in range(int(2.2 / TICK)):
                yield
            return
        self.crouch_target = depth
        for _ in range(int(1.8 / TICK)):
            self.crouch += float(np.clip(depth - self.crouch, -0.6 * TICK, 0.6 * TICK))
            yield
        self.crouch = depth

    def _a_stand_up(self):
        """Closed-loop fall recovery: roll to prone -> push -> tuck -> kneel
        -> half-kneel -> stand, each stage verified before advancing.

        Drives the whole body through ``recovery_targets`` (which beat every
        other controller in ``override_target`` and are exempt from postural
        prioritisation in ``lean_exempt``). Every stage has a done-condition
        on the live body (chest orientation, COM height) plus a timeout, so a
        blocked stage fails loudly instead of posing on the floor — and the
        auto-recovery reflex retries with the mirrored roll direction.
        Success is verified: COM > 0.72 m and not fallen, then the whole-body
        stance hold takes over again.
        """
        ag = self.agent
        nom = self.q_nom_map
        for s in "lr":
            if self.held[s] is not None:
                try:
                    self.grip(s, self.held[s], False)
                except Exception:
                    pass
                self.held[s] = None
            self.arm[s].stop()
        ag.gait.stop()
        self.gesture_active = False
        self.recovery_active = True
        self._rec_caught = False
        self._rec_walked = False
        try:
            # Curl into a ball first: always feasible (no ground leverage
            # needed), and a balled body rolls far easier than a sprawled one.
            yield from self._rec_hold(
                {"knee_l": 2.00, "knee_r": 2.00,
                 "hip_l_flex": -1.20, "hip_r_flex": -1.20,
                 "spine_bend": 0.40, "chest_bend": 0.25,
                 "elbow_l": -1.00, "elbow_r": -1.00},
                0.1, lambda: False, "curl", proceed=lambda: True)
            table_ok = yield from self._rec_table_assist()
            if not table_ok:
                yield from self._rec_extract()
            if table_ok:
                self._rec_caught = True
                self.events.append("stand_up: table climb worked, kneeling")
            else:
                yield from self._rec_roll()
            if self._rec_caught:
                if not table_ok:
                    # Caught a big roll onto the folded knees: straight to kneel.
                    self.events.append("stand_up: caught the roll, kneeling")
            else:
                # Cobra setup: bring the hands back BESIDE the chest with
                # bent elbows (pressing from overhead arms pushes only air).
                yield from self._rec_hold(
                    {"elbow_l": -1.30, "elbow_r": -1.30,
                     "sh_l_flex": 0.15, "sh_r_flex": 0.15,
                     "hip_l_flex": -0.10, "hip_r_flex": -0.10,
                     "knee_l": 0.20, "knee_r": 0.20,
                     "spine_bend": 0.0, "chest_bend": 0.0},
                    3.0, lambda: self._rec_com() > 0.25, "hand-plant",
                    proceed=lambda: True)
                # Hand-walk: servos own the arms from here (see flag).
                self._rec_hold_table = True
                self._rec_set({"knee_l": 0.10, "knee_r": 0.10,
                               "hip_l_flex": -0.05, "hip_r_flex": -0.05,
                               "ankle_l_flex": 0.40, "ankle_r_flex": 0.40,
                               "spine_bend": 0.50, "chest_bend": 0.30,
                               "elbow_l": -0.10, "elbow_r": -0.10})
                try:
                    yield from self._rec_hand_walk()
                    self._rec_walked = True
                except ActionFailed as exc:
                    # Fall through to the kneel chain, which verifies and
                    # fails fast if the walk bought nothing.
                    self.events.append(f"stand_up: walk failed ({exc}), trying kneel")
                    self._rec_walked = False
                finally:
                    if not self._rec_walked:
                        self._rec_hold_table = False
            if self._rec_walked:
                # Bent-over stance reached and unrolled: straight to stand.
                self.events.append("stand_up: walked up, standing")
            else:
                yield from self._rec_hold(
                    # Rise to tall kneel: extend the hips (thighs vertical) while
                    # the knees stay planted and the torso comes upright — the
                    # glutes lift the torso pivoting on the knees.
                    {"knee_l": 1.70, "knee_r": 1.70,
                     "hip_l_flex": -0.15, "hip_r_flex": -0.15,
                     "ankle_l_flex": -0.20, "ankle_r_flex": -0.20,
                     "spine_bend": 0.10, "chest_bend": 0.05,
                     "sh_l_flex": 0.30, "sh_r_flex": 0.30,
                     "elbow_l": -0.50, "elbow_r": -0.50},
                    6.0, lambda: self._rec_com() > 0.58, "kneel")
                yield from self._rec_hold(
                    {"knee_l": 0.40, "hip_l_flex": -0.30, "ankle_l_flex": -0.15,
                     "knee_r": 1.60, "hip_r_flex": -1.10, "ankle_r_flex": -0.55,
                     "spine_bend": 0.08, "chest_bend": 0.05,
                     "sh_l_flex": 0.50, "sh_r_flex": 0.50},
                    5.0, lambda: self._rec_com() > 0.68, "half-kneel")
            yield from self._rec_stand(nom)
            st = self.agent.state
            com_z = float(st.com[2]) if st is not None else 0.0
            if st is None or st.fallen or com_z <= 0.70:
                raise ActionFailed(
                    f"still down (COM {com_z:.2f} m) after trying to stand")
            ag.gait.fallen = False  # let the stance hold take over again
            self._rec_hold_table = False
            self.events.append(
                f"got back up (COM {com_z:.2f} m, attempt {self._recover_attempts})")
        finally:
            self.recovery_active = False
            self.recovery_targets = {}
            self._rec_hold_table = False
            self._rec_climbing = False
            for s in "lr":
                try:
                    self.arm[s].stop()
                except Exception:
                    pass

    # ---- recovery helpers (closed loop on the live body) -----------------
    def _rec_hand_home(self, side: str) -> np.ndarray:
        d = self.agent.data
        return d.xpos[self.agent.meta.body_ids[f"hand_{side}"]].copy()

    def _rec_foot_home(self, side: str) -> np.ndarray:
        d = self.agent.data
        return d.xpos[self.agent.meta.body_ids[f"foot_{side}"]].copy()

    def _rec_servo_hand(self, side: str, p: np.ndarray) -> None:
        q = p.copy()
        self.arm[side].start(lambda p=q: (p, None), use_trunk=False, w_ori=0.0)

    def _rec_hand_walk(self):
        """Walk the hands back toward the feet until the torso is up in a
        bent-over stance, then unroll to standing. Each step is lift (through
        the air, no drag) → shift back → press down INTO the ground (sets
        stiction so the hand anchors instead of sliding). Stops when a hand
        reaches its foot, then unrolls."""
        ag = self.agent
        sk = self
        R = ag.data.xmat[ag.meta.body_ids["chest"]].reshape(3, 3)
        fwd = -R[:, 1]
        fwd[2] = 0.0
        n = float(np.linalg.norm(fwd))
        if n < 1e-3:
            raise ActionFailed("stand_up: no facing to walk toward")
        back = -fwd / n  # toward the feet
        for s in "lr":
            h = sk.hands[s]
            h.owned = True
            sk._rec_servo_hand(s, sk._rec_hand_home(s))
        t0 = ag.t
        step_side = "l"

        def coord() -> None:
            # Coordinate the press with the fold: as the spine jackknifes,
            # the elbows extend so the arms stop pinning the chest down.
            try:
                sp = max(float(ag.state.qof("spine_bend")), 0.0)
            except Exception:
                sp = 0.0
            e = -1.3 + min(sp / 0.5, 1.0) * 1.2
            sk._rec_set({"elbow_l": e, "elbow_r": e})

        def wait_hold(side: str, timeout: float, err_ok: float) -> bool:
            t2 = ag.t
            while ag.t - t2 < timeout:
                coord()
                if sk.arm[side].err_pos < err_ok:
                    return True
                yield
            return False

        def press_anchor(side: str) -> bool:
            """Press the planted hand down and verify stiction by position:
            a hand that stays within 2 cm over 0.7 s while pressed is an
            anchor; one that drifts is slipping and gets re-pressed deeper
            (up to 3 tries) before giving up on it."""
            for depth in (0.005, -0.010, -0.025):
                cur = sk._rec_hand_home(side)
                tgt = cur.copy()
                tgt[2] = depth
                sk._rec_servo_hand(side, tgt)
                t2 = ag.t
                while ag.t - t2 < 0.7:
                    coord()
                    yield
                now = sk._rec_hand_home(side)
                if float(np.linalg.norm(now - cur)) < 0.020:
                    return True
            return False

        try:
            while ag.t - t0 < 60.0:
                coord()
                up = float(ag.data.xmat[ag.meta.body_ids["chest"]].reshape(3, 3)[2, 2])
                if up > 0.45 and sk._rec_com() > 0.55:
                    sk.events.append("stand_up: bent-over stance, unrolling")
                    break
                s = step_side
                step_side = "r" if s == "l" else "l"
                # stop when this hand is already at its foot
                if float(np.linalg.norm(sk._rec_hand_home(s)
                                       - sk._rec_foot_home(s))) < 0.30:
                    sk.events.append("stand_up: hands at feet, unrolling")
                    break
                cur = sk._rec_hand_home(s)
                over = cur.copy()
                over[2] = cur[2] + 0.10
                sk._rec_servo_hand(s, over)
                yield from wait_hold(s, 1.0, 0.05)
                shifted = (cur + np.array([back[0], back[1], 0.0]) * 0.05).copy()
                shifted[2] = cur[2] + 0.10
                sk._rec_servo_hand(s, shifted)
                yield from wait_hold(s, 1.2, 0.05)
                if not (yield from press_anchor(s)):
                    sk.events.append(f"stand_up: {s} hand keeps slipping, continuing")
                yield
            else:
                raise ActionFailed(f"stand_up: hands never got under (COM {sk._rec_com():.2f} m)")
            # Unroll bottom-up with both hands still planted: hips first
            # (pelvis over feet), then spine in two halves. Knees stay
            # locked straight throughout — flexing them collapses the strut.
            sk._rec_set({"hip_l_flex": 0.0, "hip_r_flex": 0.0})
            t1 = ag.t
            while ag.t - t1 < 2.0:
                yield
            for sp, ch in ((0.25, 0.10), (0.02, 0.0)):
                sk._rec_set({"spine_bend": sp, "chest_bend": ch,
                             "ankle_l_flex": -0.05, "ankle_r_flex": -0.05})
                t1 = ag.t
                while ag.t - t1 < 2.0:
                    if not ag.state.fallen and sk._rec_com() > 0.72:
                        break
                    yield
            # Sequential release: left hand first once high and steady with
            # the right still pinning; right hand only after the full
            # standing gate. Both at once face-plants.
            t1 = ag.t
            released_l = released_r = False
            plants = {s: sk._rec_hand_home(s).copy() for s in "lr"}
            ok = False
            while ag.t - t1 < 6.0:
                com = sk._rec_com()
                chest_up = float(ag.data.xmat[
                    ag.meta.body_ids["chest"]].reshape(3, 3)[2, 2])
                bal = float(getattr(ag.motor, "balance_error", 1.0) or 1.0)
                if not released_l and com > 0.68 and chest_up > 0.6:
                    sk.arm["l"].begin_retract()
                    released_l = True
                if released_l and not released_r and not ag.state.fallen \
                        and com > 0.72 and bal < 0.02:
                    sk.arm["r"].begin_retract()
                    released_r = True
                if released_r and (not ag.state.fallen and com > 0.72 and bal < 0.02):
                    ok = True
                    break
                if bal > 0.06 and (released_l != released_r):
                    # tipping with one hand off: re-plant the released hand
                    s = "l" if released_l and not released_r else "r"
                    sk._rec_servo_hand(s, plants[s])
                    if s == "l":
                        released_l = False
                    else:
                        released_r = False
                yield
            if not ok:
                raise ActionFailed(
                    f"stand_up: unroll never stabilized "
                    f"(COM {sk._rec_com():.2f} m)")
        finally:
            if not ok:
                for s in "lr":
                    try:
                        sk.arm[s].stop()
                    except Exception:
                        pass
    def _rec_table_assist(self) -> bool:
        """Climb the workbench leg hand-over-hand: plant both hands low on
        the nearest corner leg, alternate reaching higher rungs while the
        legs tuck-push below, and keep both hands anchored as support
        through kneel and stand. An external anchor beats any floor move.
        Returns True (hands stay anchored, ``_rec_hold_table`` set) or False.
        """
        ag = self.agent
        sk = self
        sk._rec_hold_table = False
        try:
            f = sk.world.furniture["table"]
        except Exception:
            return False
        c = sk.world.body_pos()
        # nearest bottom-corner leg of the table
        cx = f["x"] + (f["hx"] - 0.03) * (1.0 if c[0] >= f["x"] else -1.0)
        cy = f["y"] + (f["hy"] - 0.03) * (1.0 if c[1] >= f["y"] else -1.0)
        gap = float(np.hypot(c[0] - cx, c[1] - cy))
        inside = (abs(c[0] - f["x"]) < f["hx"] and abs(c[1] - f["y"]) < f["hy"])
        if inside or gap > 0.70:
            sk.events.append(f"stand_up: table leg {'under' if inside else 'too far'} "
                             f"(gap {gap:.2f} m), floor routine")
            return False
        out = np.array([c[0] - cx, c[1] - cy, 0.0])
        out /= max(np.linalg.norm(out), 1e-6)
        rungs = [0.22, 0.42, 0.62]

        def anchor(rung: int) -> np.ndarray:
            return np.array([cx, cy, rungs[rung]]) + out * 0.035

        for s in "lr":
            h = sk.hands[s]
            h.owned = True
            h.target.update({"thumb": 0.8, "index": 0.8, "fingers": 0.8})
            p = anchor(0).copy()
            sk.arm[s].start(lambda p=p: (p, None), use_trunk=False, w_ori=0.0)
        t0 = ag.t
        while ag.t - t0 < 5.0:
            if all(sk.arm[s].err_pos < 0.10 for s in "lr"):
                break
            yield
        if any(sk.arm[s].err_pos > 0.18 for s in "lr"):
            sk.events.append("stand_up: could not reach table leg")
            for s in "lr":
                sk.arm[s].stop()
            return False
        sk.events.append("stand_up: holding table leg, climbing")
        sk._rec_set({"knee_l": 1.60, "knee_r": 1.60,
                     "hip_l_flex": -1.00, "hip_r_flex": -1.00,
                     "spine_bend": 0.20, "chest_bend": 0.10,
                     "ankle_l_flex": 0.50, "ankle_r_flex": 0.50})
        rung = {"l": 0, "r": 0}
        turn = "l"
        t1 = ag.t
        while ag.t - t1 < 25.0:
            if all(rung[s] >= 2 for s in "lr") and sk._rec_com() > 0.38:
                break
            s = turn
            turn = "r" if s == "l" else "l"
            if rung[s] >= 2:
                yield
                continue
            rung[s] += 1
            p = anchor(rung[s]).copy()
            sk.arm[s].start(lambda p=p: (p, None), use_trunk=False, w_ori=0.0)
            t2 = ag.t
            while ag.t - t2 < 4.0:
                if sk.arm[s].err_pos < 0.12:
                    break
                yield
            if sk.arm[s].err_pos > 0.20:
                rung[s] -= 1
                p = anchor(rung[s]).copy()
                sk.arm[s].start(lambda p=p: (p, None), use_trunk=False, w_ori=0.0)
                sk.events.append(f"stand_up: {s} hand slipped back")
            yield
        ok = all(rung[s] >= 2 for s in "lr") and sk._rec_com() > 0.34
        if not ok:
            sk.events.append(f"stand_up: table climb slipped (COM {sk._rec_com():.2f} m)")
            for s in "lr":
                sk.arm[s].stop()
            return False
        sk._rec_hold_table = True
        sk._rec_climbing = True
        sk.events.append(f"stand_up: up the leg (COM {sk._rec_com():.2f} m), kneeling")
        return True

    def _rec_cobra(self):
        ag = self.agent
        t0 = ag.t
        prev = self._rec_com()
        peak = prev
        tucked = False
        while ag.t - t0 < 14.0:
            com = self._rec_com()
            peak = max(peak, com)
            if com > 0.40:
                self._rec_caught = bool(tucked)
                self.events.append("stand_up: cobra up")
                return
            vel = (com - prev) / max(TICK, 1e-6)
            prev = com
            if not tucked and com > 0.19 and vel > 0.04:
                tucked = True  # rocking up: fold early so the fold lands at the peak
            if tucked:
                push = max(0.0, float(np.sin(2 * np.pi * 0.55 * (ag.t - t0))))
                arch = -0.05 - 0.30 * push
                arm = 0.90 - 1.80 * push
                self._rec_set({
                    "spine_bend": arch, "chest_bend": 0.7 * arch,
                    "neck_bend": -0.30 * push,
                    "elbow_l": -0.05, "elbow_r": -0.05,
                    "sh_l_flex": arm, "sh_r_flex": arm,
                    "hip_l_flex": -1.10, "hip_r_flex": -1.10,
                    "knee_l": 1.90, "knee_r": 1.90})
            else:
                u = min((ag.t - t0) / 8.0, 1.0)
                push = max(0.0, float(np.sin(2 * np.pi * 0.55 * (ag.t - t0))))
                arch = -0.05 - 0.30 * push
                knee = 0.15 + 0.65 * u
                hip = 0.25 * push - 0.45 * u
                # full-range arm swing: overhead on release, slam down-back
                # on the push, hands pressing the ground beside the hips
                arm = 0.90 - 1.80 * push
                self._rec_set({
                    "spine_bend": arch, "chest_bend": 0.7 * arch,
                    "neck_bend": -0.30 * push,
                    "elbow_l": -0.05, "elbow_r": -0.05,
                    "sh_l_flex": arm, "sh_r_flex": arm,
                    "hip_l_flex": hip, "hip_r_flex": hip,
                    "knee_l": knee, "knee_r": knee,
                    "ankle_l_flex": 0.30, "ankle_r_flex": 0.30})
            yield
        tail_com = self._rec_com()
        self._rec_caught = bool(tucked) and (peak > 0.30 or tail_com > 0.22)
        if tail_com > 0.30 or peak > 0.30:
            self.events.append("stand_up: cobra up")
            return
        if peak > 0.24 or tail_com > 0.22:
            self.events.append(
                f"stand_up: low kneel (COM {tail_com:.2f} m), kneeling up")
            return
        raise ActionFailed(f"stand_up: cobra made no progress "
                           f"(COM {tail_com:.2f} m, peak {peak:.2f} m)")

    def _rec_com(self) -> float:
        st = self.agent.state
        return float(st.com[2]) if st is not None else 0.0

    def _rec_chest_face_z(self) -> float:
        """Chest facing direction, z-component: -1 face-down (prone), +1
        face-up (supine), 0 upright or on the side. (Chest-up alone cannot
        separate prone from side-lying — both read ~0.)"""
        try:
            ag = self.agent
            cid = ag.meta.body_ids["chest"]
            return float(-ag.data.xmat[cid].reshape(3, 3)[1, 2])
        except Exception:
            return 0.0

    def _rec_set(self, tgt: dict) -> None:
        for nm, tv in tgt.items():
            if nm in self.motor_idx:
                self.recovery_targets[nm] = float(tv)

    def _rec_hold(self, over: dict, timeout: float, done, stage: str,
                  proceed=None):
        """Blend to ``over`` (from nominal base), hold until ``done``."""
        ag = self.agent
        tgt = dict(self.q_nom_map)
        tgt.update(over)
        want = {nm: tv for nm, tv in tgt.items() if nm in ag.meta.qpos_addr}
        d = ag.data
        start = {nm: float(d.qpos[ag.meta.qpos_addr[nm]]) for nm in want}
        n = max(int(1.2 / TICK), 1)
        for i in range(n):
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            for nm, tv in want.items():
                self.recovery_targets[nm] = (1 - u) * start[nm] + u * tv
            yield
        self._rec_set(want)
        t0 = ag.t
        while ag.t - t0 < timeout:
            if done():
                return
            yield
        if proceed is not None and proceed():
            self.events.append(f"stand_up: {stage} partial, continuing")
            return
        raise ActionFailed(f"stand_up: {stage} made no progress "
                           f"(COM {self._rec_com():.2f} m)")

    def _rec_roll_pose(self, sign, pump=1.0):
        p = {"spine_twist": sign * 0.45, "chest_twist": sign * 0.40,
             "spine_side": sign * 0.20, "spine_bend": 0.30,
             "chest_bend": 0.15}
        top = "l" if sign > 0 else "r"
        bot = "r" if sign > 0 else "l"
        swing = 0.55 + 0.45 * pump
        p.update({f"hip_{top}_flex": -1.40 * swing, f"knee_{top}": 1.80 * swing,
                  f"hip_{top}_abd": 0.25,
                  f"hip_{bot}_flex": -0.80, f"knee_{bot}": 1.20,
                  f"sh_{top}_flex": 2.20, f"elbow_{top}": -0.20,
                  f"sh_{bot}_flex": 0.80, f"elbow_{bot}": -0.90})
        return p

    def _rec_furniture_gap(self) -> float:
        """Horizontal distance from the body to the nearest furniture edge
        (negative = underneath it). Rising under a table just bangs into it."""
        try:
            w = self.agent.skills.world
            c = w.body_pos()
            best = 1e9
            for f in w.furniture.values():
                dx = abs(c[0] - f["x"]) - f["hx"]
                dy = abs(c[1] - f["y"]) - f["hy"]
                best = min(best, max(dx, dy))
            return float(best)
        except Exception:
            return 1e9

    def _rec_extract(self):
        """Roll out from under furniture before trying to rise."""
        ag = self.agent
        if self._rec_furniture_gap() > 0.25:
            return
        self.events.append("stand_up: under furniture, rolling clear first")
        sign = 1.0
        t0 = ag.t
        last_flip = ag.t
        best = self._rec_furniture_gap()
        while ag.t - t0 < 14.0:
            if self._rec_furniture_gap() > 0.25:
                self.events.append("stand_up: clear of furniture")
                return
            if self._rec_chest_face_z() < -0.50 or self._rec_com() > 0.35:
                # rocked prone/high while extracting: that counts, roll on
                self.events.append("stand_up: clear of furniture")
                return
            pump = 0.55 + 0.45 * float(np.sin(2 * np.pi * 0.8 * (ag.t - t0)))
            if ag.t - last_flip > 2.5:
                if self._rec_furniture_gap() > best + 0.03:
                    best = self._rec_furniture_gap()
                else:
                    sign = -sign
                    best = self._rec_furniture_gap()
                last_flip = ag.t
            full = dict(self.q_nom_map)
            full.update(self._rec_roll_pose(sign, pump))
            self._rec_set({nm: tv for nm, tv in full.items()
                           if nm in ag.meta.qpos_addr})
            yield
        self.events.append("stand_up: still near furniture, trying anyway")

    def _rec_roll(self):
        """Roll until prone (chest-back faces up) or the COM lifts.

        Side-lying is a stable equilibrium far stronger than the spine-twist
        motors, so the roll uses the heavy levers: one leg swings over the
        body while the opposite arm reaches overhead, and the twist follows.
        Two mirror poses alternate every ~2.5 s; whichever raises the chest
        is kept (hill-climbing on the live orientation).
        """
        ag = self.agent
        if self._rec_chest_face_z() < -0.50 or self._rec_com() > 0.35:
            return

        sign = 1.0 if self._recover_attempts % 2 == 1 else -1.0
        t0 = ag.t
        last_flip = ag.t
        best = self._rec_chest_face_z()
        prev = self._rec_com()
        peak = prev
        tucked = False
        while ag.t - t0 < 16.0:
            face = self._rec_chest_face_z()
            com = self._rec_com()
            peak = max(peak, com)
            vel = (com - prev) / max(TICK, 1e-6)
            prev = com
            if face < -0.50 or com > 0.35:
                self.events.append("stand_up: rolled prone")
                return
            if tucked and com > 0.42:
                # caught a big rock onto the folded knees: skip ahead
                self._rec_caught = True
                self.events.append("stand_up: caught roll onto knees")
                return
            if not tucked and com > 0.28 and vel > 0.03:
                tucked = True  # big rock: fold knees, land on them
            if tucked and com < 0.18:
                tucked = False
            # pump the swing at ~0.8 Hz to rock over the hump; static holds
            # alone rock up part-way and fall back
            pump = 0.55 + 0.45 * float(np.sin(2 * np.pi * 0.8 * (ag.t - t0)))
            if ag.t - last_flip > 3.0:
                if self._rec_chest_face_z() < best - 0.05:
                    best = self._rec_chest_face_z()
                else:
                    sign = -sign  # wrong side: mirror the whole pose
                    best = self._rec_chest_face_z()
                last_flip = ag.t
            if tucked:
                self._rec_set({
                    "knee_l": 1.90, "knee_r": 1.90,
                    "hip_l_flex": -1.10, "hip_r_flex": -1.10,
                    "spine_bend": 0.20, "chest_bend": 0.10,
                    "sh_l_flex": 0.60, "sh_r_flex": 0.60})
            else:
                full = dict(self.q_nom_map)
                full.update(self._rec_roll_pose(sign, pump))
                self._rec_set({nm: tv for nm, tv in full.items()
                               if nm in ag.meta.qpos_addr})
            yield
        if peak > 0.30 or abs(self._rec_chest_face_z()) > 0.35:
            self.events.append(
                f"stand_up: rolling on (peak {peak:.2f} m)")
            return
        raise ActionFailed("stand_up: could not roll prone "
                           f"(face {self._rec_chest_face_z():+.2f})")

    def _rec_stand(self, nom):
        tgt = {f"knee_{s}": nom.get(f"knee_{s}", 0.10) for s in "lr"}
        for s in "lr":
            tgt.update({f"hip_{s}_flex": nom.get(f"hip_{s}_flex", 0.0),
                        f"ankle_{s}_flex": nom.get(f"ankle_{s}_flex", 0.0),
                        f"sh_{s}_flex": nom.get(f"sh_{s}_flex", 0.02),
                        f"sh_{s}_abd": nom.get(f"sh_{s}_abd", 0.0),
                        f"elbow_{s}": nom.get(f"elbow_{s}", -0.30)})
        tgt.update({"spine_bend": nom.get("spine_bend", 0.02),
                    "chest_bend": nom.get("chest_bend", 0.0)})
        ag = self.agent
        want = {nm: tv for nm, tv in tgt.items() if nm in ag.meta.qpos_addr}
        d = ag.data
        start = {nm: float(d.qpos[ag.meta.qpos_addr[nm]]) for nm in want}
        n = max(int(1.5 / TICK), 1)
        for i in range(n):
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            for nm, tv in want.items():
                self.recovery_targets[nm] = (1 - u) * start[nm] + u * tv
            st = self.agent.state
            if st is not None and not st.fallen and self._rec_com() > 0.72:
                break  # already up: stop driving, let stance hold
            yield
        self._rec_set(want)
        ok_until = None
        t0 = ag.t
        while ag.t - t0 < 7.0:
            st = self.agent.state
            com_z = self._rec_com()
            if st is not None and not st.fallen and com_z > 0.72:
                if ok_until is None:
                    ok_until = ag.t
                if ag.t - ok_until > 1.0:
                    return
            else:
                ok_until = None
            yield
        raise ActionFailed(f"stand_up: stand not reached (COM {self._rec_com():.2f} m)")

    def _a_gesture(self, name, hand):
        dur, fn, default_side = GESTURES[name]
        side = self._sides(hand)[0] if hand not in (None, "", "any") else default_side
        if side == "any":
            side = "r"
        # arms used by this gesture must be free
        n = int(dur / TICK)
        nominal = self.q_nom_map
        self.gesture_active = True
        for i in range(n):
            t = i * TICK
            env = min(1.0, t / 0.45, (dur - t) / 0.45)
            env = max(env, 0.0)
            pose = fn(t, side)
            tg = {}
            for jn, val in pose.items():
                base = nominal.get(jn, 0.0)
                tg[jn] = base + env * (val - base)
            self.gesture_targets = tg
            yield
        self.gesture_active = False
        self.gesture_targets = {}


# --------------------------------------------------------------------------
# Gesture library: name -> (duration, pose(t, side) -> joint targets, default side)
# --------------------------------------------------------------------------
def _g_wave(t, s):
    sx = _sx(s)
    # arm out to the side and up, shoulder twisted so that the elbow bends
    # *upwards*: the forearm stands vertical and the hand waves from the wrist
    return {f"sh_{s}_abd": -1.40 * sx, f"sh_{s}_flex": -0.05, f"sh_{s}_rot": 1.5 * sx,
            f"elbow_{s}": -1.55 + 0.10 * math.sin(2 * math.pi * 2.4 * t),
            f"wrist_{s}_dev": 0.40 * math.sin(2 * math.pi * 2.4 * t),
            f"index_{s}_mcp": 0.0, f"fingers_{s}_mcp": 0.0, f"thumb_{s}_mcp": 0.0,
            f"index_{s}_pip": 0.0, f"fingers_{s}_pip": 0.0, f"thumb_{s}_pip": 0.0}


def _g_nod(t, s):
    return {"neck_bend": 0.05 + 0.30 * math.sin(2 * math.pi * 1.7 * t) ** 2 * 1.0}


def _g_shake(t, s):
    return {"neck_twist": 0.55 * math.sin(2 * math.pi * 1.5 * t)}


def _g_shrug(t, s):
    out = {"neck_bend": -0.08}
    for sd in "lr":
        sx = _sx(sd)
        out.update({f"sh_{sd}_abd": -0.55 * sx, f"elbow_{sd}": -1.15,
                    f"sh_{sd}_flex": -0.35, f"wrist_{sd}_flex": -0.5})
    return out


def _g_clap(t, s):
    out = {}
    swing = 0.5 + 0.5 * math.sin(2 * math.pi * 2.6 * t)
    for sd in "lr":
        sx = _sx(sd)
        out.update({f"sh_{sd}_flex": -1.0, f"sh_{sd}_abd": (0.30 - 0.12 * swing) * sx,
                    f"elbow_{sd}": -1.30 - 0.15 * swing, f"sh_{sd}_rot": 0.2 * sx,
                    f"index_{sd}_mcp": 0.0, f"fingers_{sd}_mcp": 0.0, f"thumb_{sd}_mcp": 0.0,
                    f"index_{sd}_pip": 0.0, f"fingers_{sd}_pip": 0.0, f"thumb_{sd}_pip": 0.0})
    return out


def _g_thumbs_up(t, s):
    sx = _sx(s)
    return {f"sh_{s}_flex": -1.1, f"sh_{s}_abd": -0.15 * sx, f"elbow_{s}": -1.45,
            f"index_{s}_mcp": sx * 1.3, f"index_{s}_pip": sx * 1.4,
            f"fingers_{s}_mcp": sx * 1.3, f"fingers_{s}_pip": sx * 1.45,
            f"thumb_{s}_mcp": 0.0, f"thumb_{s}_pip": 0.0}


def _g_think(t, s):
    sx = _sx(s)
    return {f"sh_{s}_flex": -1.25, f"sh_{s}_abd": 0.35 * sx, f"elbow_{s}": -2.25,
            f"wrist_{s}_flex": 0.2, "neck_bend": 0.12}


def _g_scratch(t, s):
    sx = _sx(s)
    return {f"sh_{s}_flex": -1.35, f"sh_{s}_abd": -0.65 * sx, f"elbow_{s}": -2.25,
            f"wrist_{s}_dev": 0.35 * math.sin(2 * math.pi * 3.0 * t),
            f"wrist_{s}_flex": 0.2}


def _g_drink(t, s):
    sx = _sx(s)
    ph = min(max((t - 0.4) / 1.0, 0.0), 1.0)
    return {f"sh_{s}_flex": -1.15 - 0.2 * ph, f"sh_{s}_abd": 0.15 * sx,
            f"elbow_{s}": -2.05 - 0.2 * ph, f"wrist_{s}_flex": 0.5 * ph,
            "neck_bend": -0.12 * ph}


def _g_bow(t, s):
    return {"spine_bend": 0.55, "chest_bend": 0.25, "neck_bend": 0.35}


GESTURES = {
    "wave": (2.6, _g_wave, "r"),
    "nod": (1.7, _g_nod, "any"),
    "shake_head": (1.8, _g_shake, "any"),
    "shrug": (1.8, _g_shrug, "any"),
    "clap": (2.2, _g_clap, "any"),
    "thumbs_up": (2.2, _g_thumbs_up, "r"),
    "think": (3.0, _g_think, "r"),
    "scratch_head": (2.6, _g_scratch, "r"),
    "drink": (3.0, _g_drink, "r"),
    "bow": (2.4, _g_bow, "any"),
}
