"""Clean cobra-pump probe: prone placement, hand-plant, 9 s pump, COM trace."""
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
print("settled com=%.3f" % ag.state.com[2])
sk.recovery_active = True
sk.recovery_targets = {"elbow_l": -1.3, "elbow_r": -1.3,
                       "sh_l_flex": 0.15, "sh_r_flex": 0.15,
                       "hip_l_flex": -0.1, "hip_r_flex": -0.1,
                       "knee_l": 0.2, "knee_r": 0.2}
for _ in range(300):
    ag.step()
print("planted com=%.3f" % ag.state.com[2])
t0 = ag.t
zs = []
i = 0
while ag.t - t0 < 9.0:
    ag.step()
    u = min((ag.t - t0) / 8.0, 1.0)
    push = max(0.0, float(np.sin(2 * np.pi * 0.6 * (ag.t - t0))))
    arch = -0.05 - 0.27 * push
    knee = 0.15 + 1.65 * u
    hip = 0.15 - 1.05 * u
    sk.recovery_targets = {
        "spine_bend": arch, "chest_bend": 0.7 * arch,
        "elbow_l": -0.05, "elbow_r": -0.05,
        "sh_l_flex": 0.10, "sh_r_flex": 0.10,
        "hip_l_flex": hip, "hip_r_flex": hip,
        "knee_l": knee, "knee_r": knee,
        "ankle_l_flex": 0.30, "ankle_r_flex": 0.30}
    i += 1
    if i % 50 == 0:
        zs.append(round(float(ag.state.com[2]), 3))
print("cobra pump trace:", zs)
