"""Diagnostic: trace exactly why a moderate push topples the body."""
import numpy as np
import mujoco
from embodied_human.config import SimConfig
from embodied_human.build_model import load_model
from embodied_human.skeleton import nominal_posture

cfg = SimConfig()
m, d0, meta = load_model(cfg)
names = [n for _, n, _ in meta.joint_order]
nominal = nominal_posture()
q_nom = np.array([nominal.get(n, 0.0) for n in names])
lim = meta.torque_limit
q_addr = np.array([meta.qpos_addr[n] for n in names])
qd_addr = np.array([meta.dof_addr[n] for n in names])
IN = {n: i for i, n in enumerate(names)}
KP_COM, KD_COM = 700.0, 60.0
kp, kd = 2.0 * lim, 0.12 * lim

d = mujoco.MjData(m)
d.qpos[:] = d0.qpos
d.qvel[:] = 0
mujoco.mj_forward(m, d)
BODY_CHEST = meta.body_ids["chest"]
F = 60.0
for s in range(int(8.0 / m.opt.timestep)):
    t = s * m.opt.timestep
    q, qd = d.qpos[q_addr], d.qvel[qd_addr]
    tau = np.clip(kp * (q_nom - q) - kd * qd, -lim, lim)
    com, comv = d.subtree_com[0], d.subtree_linvel[0]
    feet = 0.5 * (d.site_xpos[meta.landmark_site_ids["soma_foot_l"]]
                  + d.site_xpos[meta.landmark_site_ids["soma_foot_r"]])
    ex, ey = com[0] - feet[0], com[1] - feet[1]
    ty = -KP_COM * ey - KD_COM * comv[1]
    tx = -KP_COM * ex - KD_COM * comv[0]
    tau[IN["ankle_l_flex"]] += 0.5 * ty
    tau[IN["ankle_r_flex"]] += 0.5 * ty
    tau[IN["ankle_l_inv"]] += 0.5 * tx
    tau[IN["ankle_r_inv"]] += 0.5 * tx
    d.xfrc_applied[:] = 0.0
    if 3.0 <= t < 3.2:
        d.xfrc_applied[BODY_CHEST, 1] = -F
    d.ctrl[:] = np.clip(tau, -lim, lim)
    mujoco.mj_step(m, d)
    if s % 75 == 0 and 2.9 <= t <= 5.0:
        print(f"t={t:5.2f} z={d.qpos[2]:.3f} ey={ey:+.3f} comy={com[1]:+.3f} "
              f"comvy={comv[1]:+.3f} ankL={d.qpos[q_addr[IN['ankle_l_flex']]]:+.3f} "
              f"tau_ank={d.actuator_force[meta.name_to_actuator['ankle_l_flex']]:+7.1f} "
              f"toeL={d.qpos[q_addr[IN['toe_l']]]:+.3f} "
              f"hipL={d.qpos[q_addr[IN['hip_l_flex']]]:+.3f} "
              f"kneeL={d.qpos[q_addr[IN['knee_l']]]:+.3f} "
              f"ncon={d.ncon}")
    if d.qpos[2] < 0.6:
        print(f"FELL at t={t:.2f}")
        break

# how far does the COM travel before the ankle stops helping?
print("\nankle_flex range:", -0.35, "to", 0.60, " nominal -0.05")
print("CoP must reach the toe. foot half-length 0.105 m; "
      "so max COM offset before stepping is needed ~0.10 m")
