"""
Whole-body inverse-dynamics controller.

A walking planner decides *what the body should do* -- accelerate the centre of
mass this way, hold the pelvis level, put the swing foot there.  This module
decides *which joint torques make that happen*, using the real dynamics of the
model rather than a quasi-static approximation.

The unknowns are the joint accelerations ``qdd`` of the human (floating base
+ every joint) and the ground reaction wrench ``lam`` on each foot that is in
contact.  They are tied together by the equations of motion::

    M qdd + h = tau + qfrc_passive + sum_i J_i^T lam_i

with ``tau`` = 0 on the six unactuated base coordinates.  That gives six
*equality* constraints (the base rows), plus one six-row equality per stance
foot (it does not move: ``J_i qdd = -Jdot_i qd``).  Everything the walker
wants is a *soft* task, ``J_task qdd = a_des - Jdot_task qd``, with a weight:
centre-of-mass acceleration, pelvis orientation, swing-foot pose, and a joint
space posture for whatever is left.  The weighted least-squares problem with
those equalities is one linear (KKT) solve.

Afterwards the contact wrenches are checked against what a foot can actually
do -- the centre of pressure must stay on the sole, the torsional moment
must stay inside what friction can supply -- and, if not, the offending
wrench component is *pinned* at its limit and the problem is solved again.
That active-set loop is the only nonlinearity.

The torque that falls out is ``tau = (M qdd + h - qfrc_passive - sum J^T lam)``
on the actuated rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError("MuJoCo is required: python -m pip install mujoco") from exc


@dataclass
class Stance:
    """A foot that is on the ground and must not move."""
    body: int                      # foot body id
    geom_center: np.ndarray        # unused here; kept for the caller
    share: float = 1.0             # desired fraction of the vertical load
    cop_target: np.ndarray | None = None   # world xy the CoP should sit at
    side: str = "l"


@dataclass
class SwingTask:
    """Desired task-space acceleration of the swing foot (already includes the
    planner's feed-forward and PD terms)."""
    body: int
    acc: np.ndarray        # linear, world frame (m/s^2)
    alpha: np.ndarray      # angular, world frame (rad/s^2)


@dataclass
class WBCResult:
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))
    qdd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wrench: dict = field(default_factory=dict)
    cop: dict = field(default_factory=dict)
    pinned: int = 0
    com_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ok: bool = True


class WholeBodyController:
    def __init__(self, model, meta, *, foot_half_len: float = 0.095,
                 foot_half_wid: float = 0.042, mu_torsion: float = 0.075):
        self.m = model
        self.meta = meta
        names = [n for _, n, _ in meta.joint_order]
        self.names = names
        self.act_dofs = np.array([meta.dof_addr[n] for n in names])
        self.nvh = int(self.act_dofs.max()) + 1          # human dofs: 0 .. nvh-1
        self.nv = model.nv
        self.pel = meta.body_ids["pelvis"]
        self.chest = meta.body_ids["chest"]
        self.d2 = mujoco.MjData(model)
        self._M = np.zeros((model.nv, model.nv))
        self.limits = meta.torque_limit.copy()
        self.foot_half_len = foot_half_len
        self.foot_half_wid = foot_half_wid
        self.mu_torsion = mu_torsion
        self.ankle_h = 0.071
        # posture task gains in acceleration space (critically damped)
        self.wn_post = 14.0
        self.wn_trunk = 6.202          # a soft trunk: its torque limit is easily exceeded
        self._wn_cache = None
        self.soft_trunk = True       # the gait switches this off while standing
        self.w_swing_p = 119.69
        self.w_swing_r = 10.0
        self.margin = 0.729          # pin the CoP before it reaches the edge of the sole
        self.yaw_ratio = 0.158
        self.mu_slip = 0.584
        self.pin_log = {"cop": 0, "tors": 0}
        self.last_cop = {}
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))
        self._eps = 1e-4

    @property
    def wn_vec(self) -> np.ndarray:
        key = (self.wn_post, self.wn_trunk, self.soft_trunk)
        if self._wn_cache is None or self._wn_cache[0] != key:
            v = np.full(len(self.names), self.wn_post)
            for i, nm in enumerate(self.names):
                if self.soft_trunk and nm.startswith(("spine_", "chest_")):
                    v[i] = self.wn_trunk
            self._wn_cache = (key, v)
        return self._wn_cache[1]

    # ------------------------------------------------------------------
    def _jacs(self, d, bodies: dict) -> dict:
        """Jacobians (restricted to human dofs) at the data's configuration."""
        out = {}
        nvh = self.nvh
        mujoco.mj_jacSubtreeCom(self.m, d, self._jp, self.pel)
        out["com"] = self._jp[:, :nvh].copy()
        for key, bid in bodies.items():
            mujoco.mj_jacBody(self.m, d, self._jp, self._jr, bid)
            out[key + "_p"] = self._jp[:, :nvh].copy()
            out[key + "_r"] = self._jr[:, :nvh].copy()
        return out

    def _jdot_qd(self, d, bodies: dict, J0: dict) -> dict:
        """Jdot * qd by differencing the Jacobians along the current velocity."""
        d2 = self.d2
        d2.qpos[:] = d.qpos
        mujoco.mj_integratePos(self.m, d2.qpos, d.qvel, self._eps)
        mujoco.mj_kinematics(self.m, d2)
        mujoco.mj_comPos(self.m, d2)
        J1 = self._jacs(d2, bodies)
        qd = d.qvel[:self.nvh]
        return {k: ((J1[k] - J0[k]) @ qd) / self._eps for k in J0}

    # ------------------------------------------------------------------
    def solve(self, d, *, stances: list[Stance], com_acc: np.ndarray,
              pelvis_alpha: np.ndarray, chest_alpha: np.ndarray | None,
              swing: SwingTask | None, q_ref: np.ndarray,
              posture_weight: np.ndarray, load_vz: float | None = None,
              w_com: float = 400.0, w_pel: float = 120.0) -> WBCResult:
        m = self.m
        nvh = self.nvh
        mujoco.mj_forward(m, d)
        qd = d.qvel[:nvh]

        mujoco.mj_fullM(m, d, self._M)
        M = self._M[:nvh, :nvh]
        h = d.qfrc_bias[:nvh] - d.qfrc_passive[:nvh]

        bodies = {"pel": self.pel, "chest": self.chest}
        for i, s in enumerate(stances):
            bodies[f"f{i}"] = s.body
        if swing is not None:
            bodies["sw"] = swing.body
        J = self._jacs(d, bodies)
        Jd = self._jdot_qd(d, bodies, J)

        nf = len(stances)
        nx = nvh + 6 * nf
        rows_A, rows_b, rows_w = [], [], []

        def task(Arow, b, w):
            rows_A.append(Arow)
            rows_b.append(b)
            rows_w.append(np.full(Arow.shape[0], w))

        def qdd_block(Jm):
            Z = np.zeros((Jm.shape[0], nx))
            Z[:, :nvh] = Jm
            return Z

        # --- tasks ------------------------------------------------------
        task(qdd_block(J["com"]), com_acc - Jd["com"], w_com)
        # pelvis orientation: roll/pitch strong, yaw softer
        Jr = J["pel_r"]
        ba = pelvis_alpha - Jd["pel_r"]
        task(qdd_block(Jr[:2]), ba[:2], w_pel)
        task(qdd_block(Jr[2:]), ba[2:], self.yaw_ratio * w_pel)
        if chest_alpha is not None:
            Jc = J["chest_r"]
            task(qdd_block(Jc[:2]), (chest_alpha - Jd["chest_r"])[:2], 0.25 * w_pel)
        if swing is not None:
            Jp, Jrr = J["sw_p"], J["sw_r"]
            task(qdd_block(Jp), swing.acc - Jd["sw_p"], self.w_swing_p)
            task(qdd_block(Jrr), swing.alpha - Jd["sw_r"], self.w_swing_r)
        # posture: q_ddot = kp (q_ref - q) - kd qd, per actuated dof
        wn = self.wn_vec
        kp = wn ** 2
        kd = 2.0 * wn
        sel = self.act_dofs
        P = np.zeros((len(sel), nx))
        P[np.arange(len(sel)), sel] = 1.0
        q_act = np.array([d.qpos[self.meta.qpos_addr[n]] for n in self.names])
        qd_act = qd[sel]
        a_post = kp * (q_ref - q_act) - kd * qd_act
        rows_A.append(P)
        rows_b.append(a_post)
        rows_w.append(posture_weight)

        # load sharing between the feet (vertical force ratio) and CoP targets
        if nf == 2 and load_vz is not None:
            s0, s1 = stances[0].share, stances[1].share
            R = np.zeros((1, nx))
            R[0, nvh + 2] = s1
            R[0, nvh + 6 + 2] = -s0
            task(R, np.zeros(1), 8.0)

        # regularisation
        reg = np.zeros((nx, nx))
        reg[:nvh, :nvh] = np.eye(nvh) * 1e-3
        for i in range(nf):
            reg[nvh + 6 * i:nvh + 6 * i + 6, nvh + 6 * i:nvh + 6 * i + 6] = np.eye(6) * 2e-5

        A = np.vstack(rows_A)
        b = np.concatenate(rows_b)
        w = np.concatenate(rows_w)
        Aw = A * w[:, None]
        H = A.T @ Aw + reg
        g = Aw.T @ b

        # --- equality constraints --------------------------------------
        C_rows, c_vals = [], []
        # unactuated base rows:  M_b qdd - sum J_i,b^T lam_i = -h_b
        Cb = np.zeros((6, nx))
        Cb[:, :nvh] = M[:6]
        for i, s in enumerate(stances):
            Jf = np.vstack([J[f"f{i}_p"], J[f"f{i}_r"]])
            Cb[:, nvh + 6 * i:nvh + 6 * i + 6] = -Jf[:, :6].T
        C_rows.append(Cb)
        c_vals.append(-h[:6])
        for i, s in enumerate(stances):
            Jf = np.vstack([J[f"f{i}_p"], J[f"f{i}_r"]])
            Cf = np.zeros((6, nx))
            Cf[:, :nvh] = Jf
            C_rows.append(Cf)
            c_vals.append(-np.concatenate([Jd[f"f{i}_p"], Jd[f"f{i}_r"]]))

        extra_rows: list[np.ndarray] = []
        extra_vals: list[float] = []
        pinned = 0
        res = None
        for it in range(4):
            C = np.vstack(C_rows + ([np.array(extra_rows)] if extra_rows else []))
            c = np.concatenate(c_vals + ([np.array(extra_vals)] if extra_vals else []))
            nc = C.shape[0]
            K = np.zeros((nx + nc, nx + nc))
            K[:nx, :nx] = H
            K[:nx, nx:] = C.T
            K[nx:, :nx] = C
            rhs = np.concatenate([g, c])
            try:
                sol = np.linalg.solve(K + np.diag(np.r_[np.zeros(nx), -1e-9 * np.ones(nc)]),
                                      rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            x = sol[:nx]
            qdd = x[:nvh]
            lams = [x[nvh + 6 * i: nvh + 6 * i + 6] for i in range(nf)]

            # --- feasibility of each contact wrench ---------------------
            new_rows = False
            for i, s in enumerate(stances):
                lam = lams[i]
                Fz = lam[2]
                if Fz < 20.0:
                    # (nearly) unloaded foot: it cannot push sideways or twist
                    if it == 0 and (abs(lam[0]) + abs(lam[1]) + abs(lam[5]) > 1.0
                                    or Fz < 0.0):
                        for comp, val in ((0, 0.0), (1, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)):
                            row = np.zeros(nx)
                            row[nvh + 6 * i + comp] = 1.0
                            extra_rows.append(row)
                            extra_vals.append(val)
                        new_rows = True
                        self.pin_log["unloaded"] = self.pin_log.get("unloaded", 0) + 1
                    continue
                # friction cone: a foot cannot push sideways harder than mu * load
                fh = float(np.hypot(lam[0], lam[1]))
                if fh > self.mu_slip * Fz:
                    sc = self.mu_slip * Fz / fh
                    for comp in (0, 1):
                        row = np.zeros(nx)
                        row[nvh + 6 * i + comp] = 1.0
                        extra_rows.append(row)
                        extra_vals.append(float(lam[comp] * sc))
                    new_rows = True
                    pinned += 1
                    self.pin_log["slip"] = self.pin_log.get("slip", 0) + 1
                Rm = d.xmat[s.body].reshape(3, 3)
                ank = d.xpos[s.body]
                fwd_ax = -Rm[:, 1][:2]
                lft_ax = Rm[:, 0][:2]
                # foot centre = ankle + 0.055 along the foot's forward axis
                ctr = ank[:2] + 0.055 * fwd_ax
                ry = (lam[3] - self.ankle_h * lam[1]) / Fz
                rx = (-lam[4] - self.ankle_h * lam[0]) / Fz
                cop = ank[:2] + np.array([rx, ry])
                off = cop - ctr
                of = float(off @ fwd_ax)
                ol = float(off @ lft_ax)
                if it == 0:
                    self.last_cop[s.side] = (of, ol, float(lam[5] / max(Fz, 1.0)), float(Fz))
                of_c = float(fclip(of, -self.margin * self.foot_half_len,
                                     self.margin * self.foot_half_len))
                ol_c = float(fclip(ol, -self.margin * self.foot_half_wid,
                                     self.margin * self.foot_half_wid))
                if abs(of - of_c) > 1e-4 or abs(ol - ol_c) > 1e-4:
                    cop_c = ctr + of_c * fwd_ax + ol_c * lft_ax
                    rx_c, ry_c = cop_c - ank[:2]
                    Mx = ry_c * Fz + self.ankle_h * lam[1]
                    My = -self.ankle_h * lam[0] - rx_c * Fz
                    for comp, val in ((3, Mx), (4, My)):
                        row = np.zeros(nx)
                        row[nvh + 6 * i + comp] = 1.0
                        extra_rows.append(row)
                        extra_vals.append(float(val))
                    new_rows = True
                    pinned += 1
                    self.pin_log["cop"] += 1
                mz_max = self.mu_torsion * Fz
                if abs(lam[5]) > mz_max:
                    row = np.zeros(nx)
                    row[nvh + 6 * i + 5] = 1.0
                    extra_rows.append(row)
                    extra_vals.append(float(np.sign(lam[5]) * mz_max))
                    new_rows = True
                    pinned += 1
                    self.pin_log["tors"] += 1
            if not new_rows:
                break

        tau_full = M @ qdd + h
        for i, s in enumerate(stances):
            Jf = np.vstack([J[f"f{i}_p"], J[f"f{i}_r"]])
            tau_full = tau_full - Jf.T @ lams[i]
        tau = tau_full[self.act_dofs]
        out = WBCResult(tau=tau, qdd=qdd, pinned=pinned,
                        com_acc=J["com"] @ qdd + Jd["com"])
        for i, s in enumerate(stances):
            out.wrench[s.side] = lams[i].copy()
        return out
