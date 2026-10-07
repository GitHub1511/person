"""Render what the body actually does during tip-over and fold (one png)."""
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

cfg = SimConfig(out_dir=Path("out"))
cfg.rates.receptor = 100.0
cfg.rates.afferent = 100.0
ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
ag.autonomous = False
ag.gait.hold_stance = True
for _ in range(500):
    ag.step()
ra = ag.meta.root_qpos_addr
ag.data.qpos[ra + 2] = 0.45
ag.data.qpos[ra + 3:ra + 7] = [0.7071, 0.7071, 0, 0]
ag.data.qvel[:] = 0.0
mujoco.mj_forward(ag.model, ag.data)
for _ in range(1500):
    ag.step()

renderer = mujoco.Renderer(ag.model, 360, 480)
opt = mujoco.MjvOption()
opt.sitegroup[:] = 0
shots = []


def snap():
    renderer.update_scene(ag.data, camera="observer", scene_option=opt)
    shots.append(renderer.render().copy())


snap()
sk = ag.skills
sk.recovery_active = True
sk.recovery_targets = {'knee_l': 2.2, 'knee_r': 2.2, 'hip_l_flex': -1.4,
                       'hip_r_flex': -1.4, 'spine_bend': 0.35,
                       'chest_bend': 0.2, 'sh_l_flex': 0.8, 'sh_r_flex': 0.8,
                       'elbow_l': -0.4, 'elbow_r': -0.4}
for i in range(1800):
    ag.step()
    if i in (599, 1199, 1799):
        snap()
print("fallen:", ag.state.fallen, "com_z:", round(float(ag.state.com[2]), 3))
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for k, (fr, t) in enumerate(zip(shots, ["tipped+settled", "fold+2s", "fold+4s", "fold+6s"])):
    axes[k].imshow(fr)
    axes[k].set_title(t)
    axes[k].set_xticks([])
    axes[k].set_yticks([])
fig.savefig("out/getup_shots.png", dpi=80)
print("saved out/getup_shots.png")
