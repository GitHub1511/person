"""Measure hand/foot slip during press-plant: positions + contact forces."""
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
sk.recovery_targets = {"elbow_l": -0.10, "elbow_r": -0.10,
                       "sh_l_flex": 0.10, "sh_r_flex": 0.10,
                       "knee_l": 0.1, "knee_r": 0.1,
                       "hip_l_flex": -0.05, "hip_r_flex": -0.05,
                       "ankle_l_flex": 0.4, "ankle_r_flex": 0.4,
                       "spine_bend": 0.5, "chest_bend": 0.3}
h0 = ag.data.xpos[ag.meta.body_ids["hand_l"]].copy()
f0 = ag.data.xpos[ag.meta.body_ids["foot_l"]].copy()
for i in range(15):
    for _ in range(100):
        ag.step()
    h = ag.data.xpos[ag.meta.body_ids["hand_l"]]
    f = ag.data.xpos[ag.meta.body_ids["foot_l"]]
    q = dict(zip(ag.motor.names, ag.state.q))
    hc = sk.hands["l"].contact
    print(f"t={i + 1:2d}s com={ag.state.com[2]:.3f} spine={q['spine_bend']:+.2f} "
          f"hand_slip={np.linalg.norm(h - h0):.3f} foot_slip={np.linalg.norm(f - f0):.3f} "
          f"palm_N={hc.get('palm', 0.0):.1f} digits={sorted(hc)}")
