"""Diagnostic: refine balance gains and test perturbation recovery."""
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
ANK_L, ANK_R = IN["ankle_l_flex"], IN["ankle_r_flex"]
INV_L, INV_R = IN["ankle_l_inv"], IN["ankle_r_inv"]
HIP_L, HIP_R = IN["hip_l_flex"], IN["hip_r_flex"]
KNEE_L, KNEE_R = IN["knee_l"], IN["knee_r"]
SPINE, CHEST = IN["spine_bend"], IN["chest_bend"]


def run(kp_s, kd_s, kp_com, kd_com, kp_hip, kd_hip, horizon=20.0,
        pushes=(), seed=0, trace=False):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    d.qvel[:] = 0
    d.qpos[0:2] += rng.normal(0, 1e-4, 2)
    mujoco.mj_forward(m, d)
    kp, kd = kp_s * lim, kd_s * lim
    steps = int(horizon / m.opt.timestep)
    maxdrift = 0.0
    for s in range(steps):
        t = s * m.opt.timestep
        q, qd = d.qpos[q_addr], d.qvel[qd_addr]
        tau = np.clip(kp * (q_nom - q) - kd * qd, -lim, lim)
        com, comv = d.subtree_com[0], d.subtree_linvel[0]
        feet = 0.5 * (d.site_xpos[meta.landmark_site_ids["soma_foot_l"]]
                      + d.site_xpos[meta.landmark_site_ids["soma_foot_r"]])
        ex, ey = com[0] - feet[0], com[1] - feet[1]
        ty = -kp_com * ey - kd_com * comv[1]
        tx = -kp_com * ex - kd_com * comv[0]
        tau[ANK_L] += 0.5 * ty
        tau[ANK_R] += 0.5 * ty
        tau[INV_L] += 0.5 * tx
        tau[INV_R] += 0.5 * tx
        tau[HIP_L] += 0.5 * (-kp_hip * ey - kd_hip * comv[1])
        tau[HIP_R] += 0.5 * (-kp_hip * ey - kd_hip * comv[1])
        tau[SPINE] += -kp_hip * 2.0 * (q[SPINE] - q_nom[SPINE])
        tau[CHEST] += -kp_hip * 2.0 * (q[CHEST] - q_nom[CHEST])
        # ankle must not be driven into its stops by the balance loop
        d.xfrc_applied[:] = 0.0
        for (t0, t1, fy) in pushes:
            if t0 <= t < t1:
                d.xfrc_applied[meta.body_ids["chest"], 1] = fy
        d.ctrl[:] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        maxdrift = max(maxdrift, float(np.abs(d.qpos[:2]).max()))
        if trace and s % 500 == 0:
            print(f"    t={t:5.2f} z={d.qpos[2]:.3f} com_y={com[1]:+.3f} "
                  f"|tau|max={np.abs(d.actuator_force).max():6.1f}")
        if d.qpos[2] < 0.6:
            return t, float(np.abs(d.qpos[:2]).max()), maxdrift
    return horizon, float(np.abs(d.qpos[:2]).max()), maxdrift


print("--- refined: kd_s=0.12 ---")
best = []
for kp_s in (1.0, 1.5, 2.0, 3.0):
    for kp_com in (300, 700, 1400, 2500):
        for kd_com in (60, 150):
            t, drift, mx = run(kp_s, 0.12, kp_com, kd_com, 25, 6)
            ok = t > 19.9
            if ok:
                best.append((mx, kp_s, kp_com, kd_com))
            print(f"  kp={kp_s:.1f} kp_com={kp_com:5d} kd_com={kd_com:4d} "
                  f"-> t={t:5.2f} drift={drift:.3f} max={mx:.3f} "
                  f"{'OK' if ok else 'FALL'}")

print(f"\nstable configs: {len(best)}")
best.sort()
print("best by max drift:", best[:5])

if best:
    _, kp_s, kp_com, kd_com = best[0]
    print(f"\n--- perturbation test with kp={kp_s} kp_com={kp_com} kd_com={kd_com} ---")
    for label, pushes in [
        ("gentle nudge  (60 N, 0.2 s)", [(3.0, 3.2, -60.0)]),
        ("firm shove    (150 N, 0.25 s)", [(3.0, 3.25, -150.0)]),
        ("backward push (-120 N, 0.25 s)", [(3.0, 3.25, 120.0)]),
        ("two shoves", [(2.0, 2.2, -120.0), (6.0, 6.3, 90.0)]),
    ]:
        t, drift, mx = run(kp_s, 0.12, kp_com, kd_com, 25, 6, pushes=pushes)
        print(f"  {label:32s} -> survived {t:5.2f}s  final drift={drift:.3f} "
              f"max drift={mx:.3f} {'OK' if t > 19.9 else 'FELL'}")
