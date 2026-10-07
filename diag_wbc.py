"""Stand using only the whole-body controller (no posture/balance system)."""
import sys
import time

import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.wbc import WholeBodyController, Stance

ag = EmbodiedHuman(SimConfig(), vision=False, touch_sensors=False)
ag.autonomous = False
m, meta = ag.model, ag.meta
wbc = WholeBodyController(m, meta)
names = [n for _, n, _ in meta.joint_order]
n = len(names)
q_nom = ag.motor.q_nom.copy()
mask = np.array([not (nm.startswith(("eye_", "jaw"))) for nm in names])
weights = np.where(mask, 3.0, 0.0)
foot_l, foot_r = meta.body_ids["foot_l"], meta.body_ids["foot_r"]
state = {"tau": np.zeros(n), "k": 0}
com0 = None
hold_xy = None
sub = 4
pelvis = meta.body_ids["pelvis"]


def compute(d, dt):
    global com0, hold_xy
    state["k"] += 1
    if state["k"] % sub == 1 or state["k"] == 1:
        com = d.subtree_com[pelvis].copy()
        if hold_xy is None:
            hold_xy = com[:2].copy()
            state["h"] = com[2]
        vel = np.zeros(3)
        mujoco_vel = d.subtree_linvel[pelvis] if False else None
        # velocities from the WBC helper: use finite difference of com
        if "prev" in state:
            vel = (com - state["prev"]) / (sub * dt)
        state["prev"] = com.copy()
        a = np.zeros(3)
        wn = 5.0
        a[:2] = wn ** 2 * (hold_xy - com[:2]) - 2 * 0.9 * wn * vel[:2]
        a[2] = 40.0 * (state["h"] - com[2]) - 12.0 * vel[2]
        Rp = d.xmat[pelvis].reshape(3, 3)
        axis = np.cross(Rp[:, 2], np.array([0, 0, 1.0]))
        wv = d.cvel[pelvis][:3]
        alpha = 120.0 * axis - 20.0 * wv
        st = [Stance(foot_l, None, 0.5, side="l"), Stance(foot_r, None, 0.5, side="r")]
        r = wbc.solve(d, stances=st, com_acc=a, pelvis_alpha=alpha, chest_alpha=None,
                      swing=None, q_ref=q_nom, posture_weight=weights, load_vz=1.0)
        state["res"] = r
        state["tau"] = r.tau
    return state["tau"], mask


ag.gait.compute = compute
ag.gait.cmd_active = True
t0 = time.perf_counter()
for i in range(4000):
    ag.step()
    if i % 500 == 0:
        com = ag.state.com
        r = state.get("res")
        print(f"t={ag.t:.2f} com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:.3f}) pinned={r.pinned if r else 0} "
              f"Fz={[round(v[2]) for v in r.wrench.values()] if r else None} fallen={ag.state.fallen}")
    if ag.state.fallen:
        print("FELL", ag.t)
        break
print("wall", time.perf_counter() - t0)
