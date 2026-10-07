"""Trace hand-walk mechanics: hand pos, spine q, COM per step."""
import numpy as np
import mujoco
from pathlib import Path
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

cfg = SimConfig(out_dir=Path("out"))
cfg.rates.receptor = 100.0
cfg.rates.afferent = 100.0
ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
ag.autonomous = False
ag.gait.hold_stance = True
for _ in range(300):
    ag.step()
sk = ag.skills
sk._auto_recover_t = 1e9
ra = ag.meta.root_qpos_addr
qa = ag.meta.qpos_addr
ag.data.qpos[qa['sh_l_flex']] = 0.9
ag.data.qpos[qa['sh_r_flex']] = 0.9
ag.data.qpos[ra] = 1.2
ag.data.qpos[ra + 1] = 0.6
ag.data.qpos[ra + 2] = 0.40
ag.data.qpos[ra + 3:ra + 7] = [0.7071, 0.7071, 0, 0]
ag.data.qvel[:] = 0.0
mujoco.mj_forward(ag.model, ag.data)
for _ in range(1500):
    ag.step()
sk.recovery_active = True
sk._rec_hold_table = True
sk.recovery_targets = {"elbow_l": -1.3, "elbow_r": -1.3,
                       "knee_l": 0.1, "knee_r": 0.1,
                       "hip_l_flex": -0.05, "hip_r_flex": -0.05,
                       "spine_bend": 0.5, "chest_bend": 0.3}
for _ in range(300):
    ag.step()
R = ag.data.xmat[ag.meta.body_ids["chest"]].reshape(3, 3)
fwd = -R[:, 1]
fwd[2] = 0
fwd /= max(np.linalg.norm(fwd), 1e-6)
back = -fwd
for s in "lr":
    sk.hands[s].owned = True
    p = ag.data.xpos[ag.meta.body_ids[f"hand_{s}"]].copy()
    sk.arm[s].start(lambda p=p: (p, None), use_trunk=False, w_ori=0.0)
side = "l"
for step in range(8):
    cur = ag.data.xpos[ag.meta.body_ids[f"hand_{side}"]].copy()
    tgt = (cur + np.array([back[0], back[1], 0.0]) * 0.08).copy()
    tgt[2] = max(tgt[2], 0.03)
    sk.arm[side].start(lambda p=tgt: (p, None), use_trunk=False, w_ori=0.0)
    t0 = ag.t
    while ag.t - t0 < 2.5:
        ag.step()
        if sk.arm[side].err_pos < 0.06:
            break
    q = dict(zip(ag.motor.names, ag.state.q))
    hl = ag.data.xpos[ag.meta.body_ids["hand_l"]]
    hp = ag.data.xpos[ag.meta.body_ids["pelvis"]]
    print(f"step {step} side={side} com={ag.state.com[2]:.3f} "
          f"spine={q['spine_bend']:.2f} handL=({hl[0]:.2f},{hl[1]:.2f},{hl[2]:.2f}) "
          f"pelvis=({hp[0]:.2f},{hp[1]:.2f}) err={sk.arm[side].err_pos:.3f}")
    side = "r" if side == "l" else "l"
