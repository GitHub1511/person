"""Clean kip-pump probe mirroring SkillSystem._rec_cobra, COM trace."""
import numpy as np
import mujoco
from pathlib import Path
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.skills import TICK

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
prev = ag.state.com[2]
tucked = False
peak = 0.0
i = n = 0
while ag.t - t0 < 12.0:
    ag.step()
    com = ag.state.com[2]
    peak = max(peak, com)
    vel = (com - prev) / max(TICK, 1e-6)
    prev = com
    if not tucked and com > 0.19 and vel > 0.04:
        tucked = True
        print(f"tuck ON at t={ag.t - t0:.1f}s com={com:.3f}")
    if tucked and com < 0.16:
        tucked = False
        print(f"tuck OFF at t={ag.t - t0:.1f}s com={com:.3f}")
    if tucked:
        sk.recovery_targets = {
            "spine_bend": -0.15, "chest_bend": -0.10,
            "elbow_l": -0.05, "elbow_r": -0.05,
            "sh_l_flex": 0.10, "sh_r_flex": 0.10,
            "hip_l_flex": -1.10, "hip_r_flex": -1.10,
            "knee_l": 1.90, "knee_r": 1.90}
    else:
        u = min((ag.t - t0) / 8.0, 1.0)
        push = max(0.0, float(np.sin(2 * np.pi * 0.55 * (ag.t - t0))))
        arch = -0.05 - 0.30 * push
        knee = 0.15 + 1.65 * u
        hip = 0.25 * push - 1.05 * u
        arm = 0.10 + 1.90 * push
        sk.recovery_targets = {
            "spine_bend": arch, "chest_bend": 0.7 * arch,
            "neck_bend": -0.30 * push,
            "elbow_l": -0.05, "elbow_r": -0.05,
            "sh_l_flex": arm, "sh_r_flex": arm,
            "hip_l_flex": hip, "hip_r_flex": hip,
            "knee_l": knee, "knee_r": knee,
            "ankle_l_flex": 0.30, "ankle_r_flex": 0.30}
    i += 1
print(f"peak com={peak:.3f} end com={ag.state.com[2]:.3f}")
