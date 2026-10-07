"""
Bipedal walking.

The standing controller in :mod:`embodied_human.motor` holds a *point* of
balance: it has no notion of a step.  Walking needs the opposite -- the body
must keep falling and keep catching itself -- so it is a separate controller
that takes the whole body over while the gait is active and hands it back to
the posture/balance system when the body has come to rest.

It has two layers.

**The planner (this file)** decides what the body should do, in terms of
quantities a walker reasons about:

* *Divergent component of motion (capture point).*  The linear inverted
  pendulum has one unstable mode, ``xi = c + v/omega``.  For a desired step
  length ``L`` and step time ``T`` a periodic gait has the DCM sitting
  ``L / (exp(omega T) - 1)`` ahead of the foot it is about to land on and,
  laterally, ``L_y / (exp(omega T) + 1)`` inside it.  Predicting the DCM to the
  end of the step and correcting the nominal foothold by a fraction of the
  error gives velocity regulation: push the walker and the next step is simply
  placed differently.
* *Centre-of-pressure (CoP) shifts.*  The body falls *away* from the CoP, so a
  walk begins with an anticipatory CoP shift backwards and towards the swing
  side, weight transfer is a CoP that slides from one foot to the other, and
  small CoP offsets correct the DCM within a step.
* *A swing foot* that follows a minimum-jerk path with a raised mid-swing.

**The whole-body controller (:mod:`embodied_human.wbc`)** turns those into
joint torques with the real dynamics: it solves for the joint accelerations
and ground reaction wrenches that produce the commanded centre-of-mass
acceleration, keep the pelvis level, and move the swing foot, while the stance
feet stay planted and their centres of pressure stay on the soles.

A quasi-static Jacobian-transpose controller was tried first.  It reached five
or six steps and then fell: the ground force it commands is not the one the
body gets once the pelvis and leg are accelerating, and a walker amplifies that
error by ``exp(omega T)`` every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError("MuJoCo is required: python -m pip install mujoco") from exc

from .wbc import Stance, SwingTask, WholeBodyController

LEG_JOINTS = ("hip_{s}_flex", "hip_{s}_abd", "hip_{s}_rot", "knee_{s}",
              "ankle_{s}_flex", "ankle_{s}_inv")

# Foot frame: the foot box is centred 5.5 cm in front of the ankle and its sole
# is 6 cm below it (see skeleton.py).
ANKLE_TO_CENTER = 0.055
ANKLE_HEIGHT = 0.062


def _rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def fwd(psi: float) -> np.ndarray:
    """Unit forward vector (the body faces -y at heading 0)."""
    return np.array([np.sin(psi), -np.cos(psi)])


def left(psi: float) -> np.ndarray:
    return np.array([np.cos(psi), np.sin(psi)])


def _mj(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _mj1(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 30.0 * s * s * (1.0 - s) ** 2


def _mj2(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 60.0 * s * (1.0 - 3.0 * s + 2.0 * s * s)


@dataclass
class GaitParams:
    t_ss: float = 0.424            # single-support duration (s)
    t_ds: float = 0.116            # double-support (weight transfer) duration
    t_unload: float = 0.12        # load removed from the swing foot before lift-off
    step_width: float = 0.181      # foot-to-foot lateral distance (m)
    swing_height: float = 0.037
    com_height: float = 0.821     # regulated height of the whole-body COM
    height_rate: float = 0.10     # m/s, rate of change of the height setpoint
    k_dcm: float = 4.044            # CoP shift per metre of DCM error in single support
    beta_step: float = 1.06        # fraction of the predicted DCM error corrected by the step (see below)
    heading_pull: float = 0.313     # 0 = steps follow the body, 1 = follow the commanded heading
    k_path: float = 0.212           # lateral-position-to-foothold gain when walking straight
    acc_lin: float = 0.30         # m/s^2, slew rate of the speed command
    acc_ang: float = 0.50         # rad/s^2, slew rate of the turn command
    k_speed: float = 0.421          # integral gain on the average-speed error (per step)
    max_extra_step: float = 0.02  # cap on the integrated step-length correction (m)
    max_step_fwd: float = 0.283
    max_step_back: float = 0.30
    min_step_lat: float = 0.15
    max_step_lat: float = 0.42
    # vertical acceleration law
    wz: float = 6.0               # rad/s
    zeta_z: float = 1.0
    # swing foot (acceleration-level gains)
    kp_swing: float = 150.0
    kd_swing: float = 22.841
    kr_swing: float = 200.0
    kw_swing: float = 22.0
    # CoP limits inside a foot, relative to the foot centre (heading frame)
    cop_lat: float = 0.034
    cop_sag: float = 0.071
    # trunk uprightness / yaw (acceleration-level)
    wn_tilt: float = 6.279
    k_yaw: float = 29.34
    d_yaw: float = 7.0
    w_com: float = 400.0
    w_pel: float = 56.993
    # arm swing amplitude (rad) at 0.4 m/s
    arm_swing: float = 0.30
    # receding horizon (s) of the anticipatory CoP shift that starts a walk
    apa_horizon: float = 0.293
    rate_hz: float = 250.0        # the controller runs at this rate; torque is held between
    leg_knee_ref: float = 0.65
    trunk_weight: float = 0.764
    knee_barrier: float = 0.22        # rad; below this the knee posture task stiffens
    knee_barrier_gain: float = 12.0
    posture_weight: float = 3.0
    leg_weight: float = 0.03


@dataclass
class GaitDiag:
    mode: str = "off"
    stance: str = "l"
    t_in_mode: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    com_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dcm: np.ndarray = field(default_factory=lambda: np.zeros(2))
    cop: np.ndarray = field(default_factory=lambda: np.zeros(2))
    step_target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    steps: int = 0
    balance_error: float = 0.0
    distance: float = 0.0
    speed: float = 0.0
    a_cmd: np.ndarray = field(default_factory=lambda: np.zeros(2))
    a_wbc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wrench: dict = field(default_factory=dict)
    pinned: int = 0
    sat: float = 0.0              # largest |torque| / limit demanded by the controller
    sat_joint: str = ""


class Gait:
    """Footstep walking planner on top of the whole-body controller."""

    def __init__(self, model, meta, params: GaitParams | None = None):
        self.m = model
        self.meta = meta
        self.P = params or GaitParams()
        self.names = [n for _, n, _ in meta.joint_order]
        self.n = len(self.names)
        self.pel = meta.body_ids["pelvis"]
        self.chest = meta.body_ids["chest"]
        self.mass = float(model.body_subtreemass[self.pel])
        self.g = 9.81
        self.wbc = WholeBodyController(model, meta)

        self.foot_body = {s: meta.body_ids[f"foot_{s}"] for s in "lr"}
        self.foot_geom = {s: [meta.geom_ids[f"footbody_{s}"], meta.geom_ids[f"toebox_{s}"]]
                          for s in "lr"}
        self.floor_geom = meta.geom_ids["floor"]
        self.motor_idx = {s: np.array([self.names.index(j.format(s=s))
                                       for j in LEG_JOINTS]) for s in "lr"}
        # joints the walker owns: everything except the kinematically-driven
        # eyes and jaw
        # (the fingers keep the ordinary motor system's torque-limited PD: a
        # computed-torque law on a joint with the inertia of a finger produces
        # almost no grip force)
        self.mask = np.array([not nm.startswith(("eye_", "jaw", "thumb_", "index_", "fingers_"))
                              for nm in self.names])
        # When True the whole-body controller also *holds the stance* whenever the
        # gait is not walking, instead of handing the body back to the older
        # posture/balance system.  It copes far better with arm motion.
        self.hold_stance = False
        self.stand_height = None          # COM height to hold while standing (None: as found)
        self.q_nom = None
        self.leg_ref = np.zeros(self.n)

        # command: forward speed (m/s), yaw rate (rad/s)
        self.cmd_speed = 0.0
        self.cmd_turn = 0.0
        self.cmd_active = False
        # the commands the gait actually follows: slewed, because stopping a
        # walker dead in one step is a capture step it cannot make
        self.v_eff = 0.0
        self.w_eff = 0.0

        self.mode = "off"
        self.stance = "l"
        self.t_mode = 0.0
        self.heading = 0.0
        self._e_yaw = 0.0
        self.steps = 0
        self.diag = GaitDiag()
        self._com_prev = None
        self._swing0 = np.zeros(3)
        self._p_next = np.zeros(2)
        self._psi_next = 0.0
        self._touch = {"l": False, "r": False}
        self._start_pos = None
        self.fallen = False
        self.arm_phase = 0.0       # -1..1, drives arm swing
        self._arm_amp = 0.0
        self._h_des = None
        self._tick = 0
        self._tau = np.zeros(self.n)
        self._com_vel = np.zeros(3)
        self._vel_t = 0.0
        self._lam_unload0 = 0.0
        self._stand_ref = None
        self._L_extra = 0.0
        self._path = None
        self._v_acc = 0.0
        self._v_t = 0.0

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.mode != "off"

    @property
    def walking(self) -> bool:
        """True while the feet are stepping (not merely standing under control)."""
        return self.mode not in ("off", "stand")

    def set_stand_height(self, h: float | None) -> None:
        self.stand_height = None if h is None else float(fclip(h, 0.60, 0.88))

    def walk(self, speed: float = 0.4, turn: float = 0.0) -> None:
        """Command walking: forward speed (m/s, negative = backwards) and yaw
        rate (rad/s, positive = turn left)."""
        self.cmd_speed = float(fclip(speed, -0.30, 0.70))
        self.cmd_turn = float(fclip(turn, -0.8, 0.8))
        self.cmd_active = True

    def stop(self) -> None:
        self.cmd_speed = 0.0
        self.cmd_turn = 0.0
        self.cmd_active = False

    def reset(self) -> None:
        self.mode = "off"
        self.t_mode = 0.0
        self.steps = 0
        self._com_prev = None
        self.fallen = False
        self.stop()

    # ------------------------------------------------------------------
    def _foot_center(self, d, s: str) -> np.ndarray:
        return d.geom_xpos[self.foot_geom[s][0]][:2].copy()

    def _ankle(self, d, s: str) -> np.ndarray:
        return d.xpos[self.foot_body[s]].copy()

    def _contacts(self, d) -> None:
        self._touch = {"l": False, "r": False}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 != self.floor_geom and g2 != self.floor_geom:
                continue
            other = g2 if g1 == self.floor_geom else g1
            for s in "lr":
                if other in self.foot_geom[s]:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(self.m, d, i, f)
                    if f[0] > 8.0:
                        self._touch[s] = True

    def _heading_of(self, d) -> float:
        R = d.xmat[self.pel].reshape(3, 3)
        fvec = -R[:, 1]
        return float(np.arctan2(fvec[0], -fvec[1]))

    # ------------------------------------------------------------------
    def compute(self, d, dt: float, q_ref: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint torque, mask of the joints this controller owns).

        ``q_ref`` is the posture the upper body should hold (voluntary targets,
        arm swing, ...).  The controller itself runs at ``rate_hz`` and holds
        its torque in between.
        """
        P = self.P
        # ---- state that is needed every call ----------------------------
        com = d.subtree_com[self.pel].copy()
        self._tick += 1
        period = max(int(round(1.0 / (P.rate_hz * dt))), 1) if dt > 0 else 1
        due = (self._tick % period == 1) or period == 1 or not self.active
        if com[2] < 0.5 and self.mode != "off":
            self.fallen = True
            self.mode = "off"
        if self.mode == "off" and (self.cmd_active or self.hold_stance) and not self.fallen:
            self._begin(d)
        if self.mode == "off":
            self.diag.mode = "off"
            return np.zeros(self.n), np.zeros(self.n, bool)
        self.t_mode += dt
        if not due:
            return self._tau, self.mask.copy()

        dt_c = dt * period
        tgt_v = self.cmd_speed if self.cmd_active else 0.0
        tgt_w = self.cmd_turn if self.cmd_active else 0.0
        self.v_eff += float(fclip(tgt_v - self.v_eff, -P.acc_lin * dt_c, P.acc_lin * dt_c))
        self.w_eff += float(fclip(tgt_w - self.w_eff, -P.acc_ang * dt_c, P.acc_ang * dt_c))
        if self._com_prev is None:
            vel = np.zeros(3)
        else:
            raw = (com - self._com_prev) / max(dt_c, 1e-9)
            vel = 0.5 * self._com_vel + 0.5 * raw
        self._com_prev = com.copy()
        self._com_vel = vel
        h = max(float(com[2]), 0.3)
        omega = float(np.sqrt(self.g / h))
        xi = com[:2] + vel[:2] / omega
        self._contacts(d)
        for _ in range(3):
            if not self._transition(d, xi, vel, omega):
                break
        if self.mode == "off":
            self.diag.mode = "off"
            return np.zeros(self.n), np.zeros(self.n, bool)

        S = self.stance
        W = "r" if S == "l" else "l"
        pS = self._foot_center(d, S)
        pW = self._foot_center(d, W)
        psi_t = self.heading                      # where the body should point
        psi_b = self._heading_of(d)               # where it points
        # Steps are laid out relative to the body, pulled gently towards the
        # desired heading: foot yaw that disagrees with the pelvis makes every
        # swing a torsion load on the stance foot, and the error grows.
        psi = psi_b + P.heading_pull * float(fclip(_wrap(psi_t - psi_b), -0.3, 0.3))
        f, l = fwd(psi), left(psi)
        sigma = 1.0 if W == "l" else -1.0           # side the swing foot is on
        T = P.t_ss + P.t_ds
        eT = np.exp(omega * T)
        # nominal step length, plus an integral correction that makes the
        # *average* speed meet the command: the capture-point law regulates the
        # state of the pendulum but a walker with unmodelled losses settles
        # slower than commanded unless something integrates the shortfall
        L_d = self.v_eff * T + self._L_extra * np.sign(self.v_eff)
        self._v_acc += float(np.dot(vel[:2], f)) * dt_c
        self._v_t += dt_c
        Wd = P.step_width

        swing = None
        resid_des = np.zeros(2)
        lam = {S: 1.0, W: 0.0}
        step_target = pS
        stance_sides = [S, W]

        if self.mode == "init":
            target = pS + (L_d / (eT - 1.0)) * f + (sigma * Wd / (eT + 1.0)) * l
            step_target = target
            e = np.exp(omega * P.apa_horizon)
            p_apa = (target - xi * e) / (1.0 - e)
            lam, resid_des = self._split_ds(p_apa, pS, pW, S, W)
            lam[S] = max(lam[S], 0.15)
            lam[W] = 1.0 - lam[S]
            self._lam_unload0 = lam[W]
        elif self.mode == "unload":
            a = _mj(self.t_mode / P.t_unload)
            lam = {S: 1.0 - self._lam_unload0 * (1 - a), W: self._lam_unload0 * (1 - a)}
            if lam[W] < 0.02:
                lam = {S: 1.0, W: 0.0}
        elif self.mode == "ss":
            t = self.t_mode
            xi_ref = pS + ((L_d / (eT - 1.0)) * f + (sigma * Wd / (eT + 1.0)) * l) \
                * np.exp(omega * t)
            resid_des = P.k_dcm * (xi - xi_ref)
            Tr = max(P.t_ss - t, 0.0) + 0.5 * P.t_ds + 0.02
            psi_n = psi + self.w_eff * T
            fn, ln = fwd(psi_n), left(psi_n)
            err_eos = (xi - xi_ref) * np.exp(omega * Tr)
            p_next = pS + L_d * fn + sigma * Wd * ln + P.beta_step * err_eos
            # keep to the line: a COM that has drifted to one side of the path
            # gets its next foot placed on that side (the pendulum falls away
            # from the foot, i.e. back towards the line)
            if abs(self.w_eff) < 0.03 and P.k_path > 0:
                if self._path is None:
                    self._path = (com[:2].copy(), psi_t)
                p0, ps0 = self._path
                lat_err = float(np.dot(com[:2] - p0, left(ps0)))
                p_next = p_next + fclip(P.k_path * lat_err, -0.08, 0.08) * left(ps0)
            else:
                self._path = None
            rel = p_next - pS
            a = float(fclip(np.dot(rel, f), -P.max_step_back, P.max_step_fwd))
            c = float(fclip(np.dot(rel, l) * sigma, P.min_step_lat, P.max_step_lat))
            p_next = pS + a * f + c * sigma * l
            if t < 0.75 * P.t_ss:           # freeze the foothold late in the swing
                self._p_next = p_next
                self._psi_next = psi_n
            step_target = self._p_next
            swing = self._swing_path(t)
            stance_sides = [S]
        elif self.mode == "ds":
            a = _mj(self.t_mode / P.t_ds)
            lam = {S: 1.0 - a, W: a}
        elif self.mode == "settle":
            mid = 0.5 * (pS + pW)
            e = np.exp(omega * 0.35)
            p_des = (mid - xi * e) / (1.0 - e)
            lam, resid_des = self._split_ds(p_des, pS, pW, S, W)
            lam[S] = float(fclip(lam[S], 0.2, 0.8))
            lam[W] = 1.0 - lam[S]
        elif self.mode == "stand":
            # hold the COM over the middle of the feet with a PD law (the DCM
            # law of the walking modes is for *falling* on purpose)
            lam = {S: 0.5, W: 0.5}
            self._stand_ref = 0.5 * (pS + pW) if self._stand_ref is None else self._stand_ref
            if self.t_mode < 0.05:
                self._stand_ref = 0.5 * (pS + pW)

        # ---- desired accelerations --------------------------------------
        centers = {s: self._foot_center(d, s) for s in "lr"}
        resid = self._clip_in_foot(resid_des, f, l)
        cop_des = sum(lam[s] * centers[s] for s in "lr") + resid
        a_c = np.zeros(3)
        if self.mode == "stand":
            wn = 4.5
            a_c[:2] = wn ** 2 * (self._stand_ref - com[:2]) - 2 * 0.95 * wn * vel[:2]
        else:
            a_c[:2] = omega ** 2 * (com[:2] - cop_des)
        if self._h_des is None:
            self._h_des = float(com[2])
        goal_h = P.com_height
        if self.mode == "stand":
            if self.stand_height is None:
                self.stand_height = float(com[2])
            goal_h = self.stand_height
        self._h_des += float(fclip(goal_h - self._h_des,
                                     -P.height_rate * dt_c, P.height_rate * dt_c))
        a_c[2] = (P.wz ** 2 * (self._h_des - com[2])
                  - 2.0 * P.zeta_z * P.wz * vel[2])
        a_c[2] = float(fclip(a_c[2], -4.0, 3.0))

        def level_alpha(body: int, w_gain: float = 1.0):
            R = d.xmat[body].reshape(3, 3)
            axis = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))
            wv = d.cvel[body][:3]
            al = np.zeros(3)
            al[:2] = (P.wn_tilt ** 2 * axis[:2] - 2 * 0.9 * P.wn_tilt * wv[:2]) * w_gain
            return al

        yaw = self._heading_of(d)
        e_yaw = (psi_t - yaw + np.pi) % (2 * np.pi) - np.pi
        self._e_yaw = float(e_yaw)
        pel_alpha = level_alpha(self.pel)
        pel_alpha[2] = P.k_yaw * e_yaw - P.d_yaw * float(d.cvel[self.pel][2])
        chest_alpha = level_alpha(self.chest)

        swing_task = None
        if swing is not None:
            tgt, tv, ta, yaw_t = swing
            ank = self._ankle(d, W)
            vcart = np.zeros(6)
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_XBODY,
                                     self.foot_body[W], vcart, 0)
            wang, vlin = vcart[:3], vcart[3:]
            Rf = d.xmat[self.foot_body[W]].reshape(3, 3)
            Rd = _rz(yaw_t)
            rot_err = 0.5 * (np.cross(Rf[:, 0], Rd[:, 0]) + np.cross(Rf[:, 1], Rd[:, 1])
                             + np.cross(Rf[:, 2], Rd[:, 2]))
            swing_task = SwingTask(
                body=self.foot_body[W],
                acc=ta + P.kp_swing * (tgt - ank) + P.kd_swing * (tv - vlin),
                alpha=P.kr_swing * rot_err - P.kw_swing * wang)

        stances = [Stance(self.foot_body[s], None, lam[s], side=s)
                   for s in stance_sides if lam.get(s, 0.0) > 1e-3 or s == S]

        # ---- posture reference -------------------------------------------
        if self.q_nom is None:
            self.q_nom = np.array([d.qpos[self.meta.qpos_addr[nm]] for nm in self.names])
        qr = np.array(q_ref if q_ref is not None else self.q_nom, float)
        weights = np.full(self.n, P.posture_weight)
        for s in "lr":
            idx = self.motor_idx[s]
            weights[idx] = P.leg_weight
            qr[idx] = self.leg_ref[idx] if self.leg_ref[idx].any() else qr[idx]
            qr[self.names.index(f"knee_{s}")] = P.leg_knee_ref
        # knee barrier: a locked knee has no compliance left (the walker fell
        # a few steps after both knees reached their extension limit), so the
        # closer a knee gets to straight the harder the posture task pulls it back
        walking_now = self.mode != "stand"
        for s in "lr":
            ki = self.names.index(f"knee_{s}")
            qk = float(d.qpos[self.meta.qpos_addr[f"knee_{s}"]])
            if walking_now and qk < P.knee_barrier:
                weights[ki] = P.leg_weight + P.knee_barrier_gain * (P.knee_barrier - qk) / P.knee_barrier
                qr[ki] = P.knee_barrier + 0.15
        for i, nm in enumerate(self.names):
            if walking_now and nm.startswith(("spine_", "chest_")):
                # a soft trunk: it has to follow the pelvis as the swing leg
                # twists it, and a stiff chest saturates its torque doing so
                weights[i] = P.trunk_weight
        weights[~self.mask] = 0.0

        self.wbc.soft_trunk = walking_now
        res = self.wbc.solve(d, stances=stances, com_acc=a_c, pelvis_alpha=pel_alpha,
                             chest_alpha=chest_alpha, swing=swing_task, q_ref=qr,
                             posture_weight=weights,
                             load_vz=1.0 if len(stances) == 2 else None,
                             w_com=P.w_com, w_pel=P.w_pel)
        tau = res.tau.copy()
        sat = np.abs(tau) / np.maximum(self.wbc.limits, 1e-9)
        sat[~self.mask] = 0.0
        self.diag.sat = float(sat.max())
        self.diag.sat_joint = self.names[int(sat.argmax())]
        tau[~self.mask] = 0.0
        tau = fclip(tau, -self.wbc.limits, self.wbc.limits)
        self._tau = tau

        # ---- arm swing phase: arms oppose the legs ----------------------
        if self.mode == "ss":
            ph = 2.0 * _mj(self.t_mode / P.t_ss) - 1.0
            self.arm_phase = ph if S == "l" else -ph
        elif self.mode == "ds":
            self.arm_phase = (1.0 if S == "l" else -1.0) * (1.0 - 2.0 * _mj(
                self.t_mode / P.t_ds))
        self._arm_amp = P.arm_swing * min(abs(self.v_eff) / 0.4, 1.2)

        dg = self.diag
        dg.mode, dg.stance, dg.t_in_mode = self.mode, S, self.t_mode
        dg.com, dg.com_vel, dg.dcm = com, vel, xi
        dg.cop = cop_des
        dg.a_cmd = a_c[:2]
        dg.step_target = step_target
        dg.steps = self.steps
        dg.pinned = res.pinned
        dg.a_wbc = res.com_acc[:2].copy()
        dg.wrench = {k: v.copy() for k, v in res.wrench.items()}
        mid = 0.5 * (pS + pW)
        dg.balance_error = float(np.linalg.norm(xi - mid)) * 0.6
        if self._start_pos is not None:
            dg.distance = float(np.linalg.norm(com[:2] - self._start_pos))
        dg.speed = float(np.linalg.norm(vel[:2]))
        return tau, self.mask.copy()

    # ------------------------------------------------------------------
    def arm_swing_targets(self) -> dict:
        """Shoulder-flexion offsets (rad) for each arm; zero when not walking."""
        if not self.active:
            return {"l": 0.0, "r": 0.0}
        a = self._arm_amp * self.arm_phase
        return {"l": a, "r": -a}

    # ------------------------------------------------------------------
    def _swing_path(self, t: float):
        P = self.P
        s = t / P.t_ss
        T = P.t_ss
        p0 = self._swing0
        c = _rz(self._psi_next)[:2, :2] @ np.array([0.0, ANKLE_TO_CENTER])
        tgt_xy = self._p_next + c
        dxy = tgt_xy - p0[:2]
        xy = p0[:2] + _mj(s) * dxy
        vxy = _mj1(s) / T * dxy
        axy = _mj2(s) / T ** 2 * dxy
        sc = min(max(s, 0.0), 1.0)
        H = P.swing_height
        z_lift = H * 16.0 * sc ** 2 * (1.0 - sc) ** 2
        dz = H * 32.0 * sc * (1.0 - sc) * (1.0 - 2.0 * sc) / T
        ddz = H * 32.0 * (1.0 - 6.0 * sc + 6.0 * sc ** 2) / T ** 2
        z = ANKLE_HEIGHT + (p0[2] - ANKLE_HEIGHT) * (1.0 - _mj(s)) + z_lift
        if t > T:
            # still airborne: keep lowering the foot until it finds the ground
            z -= min(0.05, 0.25 * (t - T))
            vxy = vxy * 0.0
            axy = axy * 0.0
            dz, ddz = -0.25, 0.0
        return (np.array([xy[0], xy[1], z]), np.array([vxy[0], vxy[1], dz]),
                np.array([axy[0], axy[1], ddz]), self._psi_next)

    # ------------------------------------------------------------------
    def _begin(self, d) -> None:
        self.heading = self._heading_of(d)
        self.stance = "l"
        self._stand_ref = None
        moving = (self.cmd_active and abs(self.cmd_speed) + abs(self.cmd_turn) > 1e-3) \
            or abs(self.v_eff) + abs(self.w_eff) > 0.03
        self.mode = "init" if moving else "stand"
        self.t_mode = 0.0
        self._start_pos = d.subtree_com[self.pel][:2].copy()
        self._com_prev = None
        self._h_des = None
        self.fallen = False
        self._tick = 0
        # leg reference: the standing posture
        self.leg_ref = np.zeros(self.n)

    def _enter(self, mode: str, d) -> None:
        self.mode = mode
        self.t_mode = 0.0
        if mode == "ss":
            W = "r" if self.stance == "l" else "l"
            self._swing0 = self._ankle(d, W)
            self._p_next = self._foot_center(d, W)
            self._psi_next = self.heading

    def _transition(self, d, xi, vel, omega) -> bool:
        """Advance the gait state machine.  Returns True if the mode changed."""
        P = self.P
        S = self.stance
        W = "r" if S == "l" else "l"
        pS = self._foot_center(d, S)
        pW = self._foot_center(d, W)
        moving = (self.cmd_active and abs(self.cmd_speed) + abs(self.cmd_turn) > 1e-3) \
            or abs(self.v_eff) + abs(self.w_eff) > 0.03

        if self.mode == "init":
            f, l = fwd(self.heading), left(self.heading)
            sigma = 1.0 if W == "l" else -1.0
            T = P.t_ss + P.t_ds
            eT = np.exp(omega * T)
            target = pS + (self.cmd_speed * T / (eT - 1.0)) * f \
                + (sigma * P.step_width / (eT + 1.0)) * l
            err = float(np.linalg.norm(xi - target))
            crouched = (self._h_des is not None
                        and abs(self._h_des - P.com_height) < 1e-3
                        and abs(float(d.subtree_com[self.pel][2]) - P.com_height) < 0.02)
            if (err < 0.015 and self.t_mode > 0.2 and crouched) or self.t_mode > 3.0:
                self._enter("unload", d)
                return True
            if not moving and self.t_mode > 0.3:
                self._enter("settle", d)
                return True
        elif self.mode == "unload":
            if self.t_mode >= P.t_unload:
                self._enter("ss", d)
                return True
        elif self.mode == "ss":
            t = self.t_mode
            landed = (self._touch[W] and t > 0.6 * P.t_ss
                      and self._ankle(d, W)[2] < ANKLE_HEIGHT + 0.03)
            if (t >= P.t_ss and landed) or t > P.t_ss + 0.45:
                self._enter("ds", d)
                return True
        elif self.mode == "ds":
            if self.t_mode >= P.t_ds:
                self.steps += 1
                if self._v_t > 0.2 and abs(self.v_eff) > 0.02:
                    v_avg = self._v_acc / self._v_t
                    self._L_extra = float(fclip(
                        self._L_extra + P.k_speed * (abs(self.v_eff) - abs(v_avg))
                        * (P.t_ss + P.t_ds), -0.05, P.max_extra_step))
                elif abs(self.v_eff) <= 0.02:
                    self._L_extra *= 0.5
                self._v_acc = 0.0
                self._v_t = 0.0
                self.stance = W
                self.heading += self.w_eff * (P.t_ss + P.t_ds)
                nS, nW = self.stance, S
                mid = 0.5 * (self._foot_center(d, nS) + self._foot_center(d, nW))
                # (the DCM of a walker that has just stopped stepping still sits
                # a few centimetres inside the foot it landed on, so "calm" has
                # to allow for that, or the walker marches on the spot forever)
                calm = (np.linalg.norm(xi - mid) < 0.10
                        and np.linalg.norm(vel[:2]) < 0.20)
                self._enter("settle" if (not moving and calm) else "ss", d)
                return True
        elif self.mode == "settle":
            mid = 0.5 * (pS + pW)
            if (np.linalg.norm(xi - mid) < 0.02 and np.linalg.norm(vel[:2]) < 0.06
                    and self.t_mode > 0.5 and not moving):
                if self.hold_stance:
                    self._stand_ref = None
                    self._enter("stand", d)
                else:
                    self.mode = "off"
                    self.t_mode = 0.0
                return True
            if moving and self.t_mode > 0.3:
                self._enter("init", d)
                return True
        elif self.mode == "stand":
            if moving:
                self.heading = self._heading_of(d)
                self._enter("init", d)
                return True
        return False

    # ------------------------------------------------------------------
    def _clip_in_foot(self, r: np.ndarray, f: np.ndarray, l: np.ndarray) -> np.ndarray:
        a = float(fclip(np.dot(r, f), -self.P.cop_sag, self.P.cop_sag))
        b = float(fclip(np.dot(r, l), -self.P.cop_lat, self.P.cop_lat))
        return a * f + b * l

    def _split_ds(self, p_des, pS, pW, S, W):
        """Split a desired CoP across two feet.

        The part of ``p_des`` along the line between the feet is realised by
        shifting load from one foot to the other; whatever remains is realised
        inside the feet (limited by their size).
        """
        seg = pW - pS
        den = float(np.dot(seg, seg)) + 1e-9
        lam_w = float(fclip(np.dot(p_des - pS, seg) / den, 0.0, 1.0))
        base = pS + lam_w * seg
        return {S: 1.0 - lam_w, W: lam_w}, p_des - base
