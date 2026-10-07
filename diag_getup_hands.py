"""Render prone + candidate hand-plant poses side by side."""
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

POSES = {
    "sh+0.15": {"sh_l_flex": 0.15, "sh_r_flex": 0.15, "elbow_l": -1.3, "elbow_r": -1.3},
    "sh-0.30": {"sh_l_flex": -0.30, "sh_r_flex": -0.30, "elbow_l": -1.3, "elbow_r": -1.3},
    "sh+0.60": {"sh_l_flex": 0.60, "sh_r_flex": 0.60, "elbow_l": -1.3, "elbow_r": -1.3},
}

cfg = SimConfig(out_dir=Path("out"))
cfg.rates.receptor = 100.0
cfg.rates.afferent = 100.0
renderer = None
opt = mujoco.MjvOption()
opt.sitegroup[:] = 0
shots = []
for name, arm in POSES.items():
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    ag.gait.hold_stance = True
    for _ in range(300):
        ag.step()
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
    if renderer is None:
        renderer = mujoco.Renderer(ag.model, 360, 480)
    sk = ag.skills
    sk._auto_recover_t = 1e9
    sk.recovery_active = True
    tgt = dict(arm)
    tgt.update({"hip_l_flex": -0.1, "hip_r_flex": -0.1, "knee_l": 0.2, "knee_r": 0.2})
    sk.recovery_targets = tgt
    for _ in range(400):
        ag.step()
    renderer.update_scene(ag.data, camera="observer", scene_option=opt)
    shots.append(renderer.render().copy())
    print(name, "com_z=%.2f" % ag.state.com[2])
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for k, name in enumerate(POSES):
    axes[k].imshow(shots[k])
    axes[k].set_title(name)
    axes[k].set_xticks([])
    axes[k].set_yticks([])
fig.savefig("out/getup_hands.png", dpi=80)
print("saved out/getup_hands.png")
