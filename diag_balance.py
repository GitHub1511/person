"""
Diagnostic: tune standing balance with physically-derived gains.

Required ankle torque to hold the body against gravity is
    tau ~= m * g * (com_xy - support_xy)
so for m=72 kg the loop gain must be of order 700 Nm/m, not 45.
"""
import numpy as np
import mujoco
from embodied_human.config import SimConfig
from embodied_human.build_model import load_model, lowest_foot_z
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
MASS = sum(m.body_mass) - sum(o.mass for o in meta.objects)
print(f"human mass {MASS:.2f} kg, weight {MASS*9.81:.0f} N "
      f"(so 1 cm of COM offset costs {MASS*9.81*0.01:.1f} Nm)")

ANK_L, ANK_R = IN["ankle_l_flex"], IN["ankle_r_flex"]
INV_L, INV_R = IN["ankle_l_inv"], IN["ankle_r_inv"]
HIP_L, HIP_R = IN["hip_l_flex"], IN["hip_r_flex"]
TOE_L, TOE_R = IN["toe_l"], IN["toe_r"]
SPINE, CHEST = IN["spine_bend"], IN["chest_bend"]


def run(kp_scale, kd_scale, kp_com, kd_com, kp_hip, kd_hip,
        horizon=10.0, push=0.0, trace=False):
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    kp = kp_scale * lim
    kd = kd_scale * lim
    steps = int(horizon / m.opt.timestep)
    minz = 9.9
    for s in range(steps):
        q = d.qpos[q_addr]
        qd = d.qvel[qd_addr]
        tau = np.clip(kp * (q_nom - q) - kd * qd, -lim, lim)

        com = d.subtree_com[0]
        comv = d.subtree_linvel[0]
        feet = 0.5 * (d.site_xpos[meta.landmark_site_ids["soma_foot_l"]]
                      + d.site_xpos[meta.landmark_site_ids["soma_foot_r"]])
        ex, ey = com[0] - feet[0], com[1] - feet[1]
        t_y = -kp_com * ey - kd_com * comv[1]
        t_x = -kp_com * ex - kd_com * comv[0]
        tau[ANK_L] += 0.5 * t_y
        tau[ANK_R] += 0.5 * t_y
        tau[INV_L] += 0.5 * t_x
        tau[INV_R] += 0.5 * t_x
        tau[HIP_L] += 0.5 * (-kp_hip * ey - kd_hip * comv[1])
        tau[HIP_R] += 0.5 * (-kp_hip * ey - kd_hip * comv[1])
        # keep the trunk upright: extra spine stiffness
        tau[SPINE] += -kp_hip * 2.0 * (q[SPINE] - q_nom[SPINE])
        tau[CHEST] += -kp_hip * 2.0 * (q[CHEST] - q_nom[CHEST])
        if push and 3.0 <= s * m.opt.timestep < 3.06:
            d.xfrc_applied[meta.body_ids["chest"], 1] = -push
        else:
            d.xfrc_applied[meta.body_ids["chest"], 1] = 0.0

        d.ctrl[:] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        minz = min(minz, d.qpos[2])
        if trace and s % 250 == 0:
            print(f"    t={s*m.opt.timestep:4.2f} z={d.qpos[2]:.3f} "
                  f"com_y={com[1]:+.3f} |tau|max={np.abs(d.actuator_force).max():6.1f} "
                  f"ank={d.actuator_force[meta.name_to_actuator['ankle_l_flex']]:+7.1f}")
        if d.qpos[2] < 0.6:
            return s * m.opt.timestep, d.qpos[2], float(np.abs(d.qpos[:2]).max())
    return horizon, d.qpos[2], float(np.abs(d.qpos[:2]).max())


print("\n--- posture stiffness sweep (kp_com=700) ---")
for kp_scale in (1.5, 3.0, 6.0, 10.0):
    for kd_scale in (0.05, 0.12):
        t, z, drift = run(kp_scale, kd_scale, 700, 90, 25, 6)
        print(f"  kp={kp_scale:4.1f}*lim kd={kd_scale:.2f}*lim -> "
              f"t={t:5.2f}s drift={drift:.3f} {'OK' if t > 9.9 else 'FALL'}")

print("\n--- COM loop sweep (kp_scale=3, kd_scale=0.05) ---")
for kp_com in (200, 400, 700, 1200, 2000):
    for kd_com in (30, 90, 200):
        t, z, drift = run(3.0, 0.05, kp_com, kd_com, 25, 6)
        print(f"  kp_com={kp_com:5.0f} kd_com={kd_com:4.0f} -> "
              f"t={t:5.2f}s drift={drift:.3f} {'OK' if t > 9.9 else 'FALL'}")
