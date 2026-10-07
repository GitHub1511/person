"""Diagnostic: trace what actually happens during the fall."""
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

# --- 1. do actuators actually produce force? ------------------------------
d = mujoco.MjData(m)
d.qpos[:] = d0.qpos
mujoco.mj_forward(m, d)
d.ctrl[:] = 0.0
for _ in range(200):
    mujoco.mj_step(m, d)
print("ctrl=0    -> pelz=%.4f  knee_l=%.3f  hip_lf=%.3f  actf_max=%.3f" % (
    d.qpos[2], d.qpos[q_addr[IN["knee_l"]]], d.qpos[q_addr[IN["hip_l_flex"]]],
    np.abs(d.actuator_force).max()))

# --- 2. can a single joint hold against gravity? --------------------------
d2 = mujoco.MjData(m)
d2.qpos[:] = d0.qpos
mujoco.mj_forward(m, d2)
kp = 0.55 * lim
kd = 0.055 * kp
for step in range(1500):
    q = d2.qpos[q_addr]
    qd = d2.qvel[qd_addr]
    d2.ctrl[:] = np.clip(kp * (q_nom - q) - kd * qd, -lim, lim)
    mujoco.mj_step(m, d2)
    if step % 150 == 0:
        print(f"  t={step*m.opt.timestep:4.2f} pelz={d2.qpos[2]:.3f} "
              f"knee_l={d2.qpos[q_addr[IN['knee_l']]]:+.3f} "
              f"hip_lf={d2.qpos[q_addr[IN['hip_l_flex']]]:+.3f} "
              f"ank_lf={d2.qpos[q_addr[IN['ankle_l_flex']]]:+.3f} "
              f"spineb={d2.qpos[q_addr[IN['spine_bend']]]:+.3f} "
              f"|tau|max={np.abs(d2.actuator_force).max():6.1f} "
              f"com={np.round(d2.subtree_com[0][:2],3)}")

# --- 3. what is on the ground at the end? ---------------------------------
print("\nfinal lowest bodies:")
order = np.argsort(d2.xpos[:, 2])[:6]
for b in order:
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
    print(f"   z={d2.xpos[b,2]:+.3f}  {nm}")
print("contacts at end:", d2.ncon)
from collections import Counter
cc = Counter()
for c in d2.contact:
    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
    n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
    cc[f"{n1}|{n2}"] += 1
for k, v in cc.most_common(10):
    print(f"   {v:3d}  {k}")

# --- 4. joint limit violations? ------------------------------------------
print("\njoints far from nominal at end:")
err = np.abs(d2.qpos[q_addr] - q_nom)
for i in np.argsort(-err)[:8]:
    print(f"   {names[i]:16s} q={d2.qpos[q_addr[i]]:+.3f} nom={q_nom[i]:+.3f} "
          f"err={err[i]:.3f} lim=±{lim[i]:.0f}")
