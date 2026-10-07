"""Render a full stand_up attempt (6 shots)."""
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
ag.data.qpos[ra] = 1.2
ag.data.qpos[ra + 1] = 0.6
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
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = ag.state.com
    cam.distance = 2.4
    cam.elevation = -8.0
    cam.azimuth = 135.0
    renderer.update_scene(ag.data, camera=cam, scene_option=opt)
    shots.append(renderer.render().copy())


snap()
sk = ag.skills
sk._auto_recover_t = 1e9  # only the explicit attempt runs
sk.api_stand_up()
labels = ["down"]
steps = [1999, 3999, 5999, 7999, 9999, 11999, 15999, 19999, 23999, 27999]
for i in range(30000):
    ag.step()
    if i in steps:
        snap()
        st = ag.state
        labels.append(f"t+{(i+1)//1000}s com{st.com[2]:.2f}")
print("fallen:", ag.state.fallen, "com_z:", round(float(ag.state.com[2]), 3))
print("events:", ag.skills.events[-6:])
fig, axes = plt.subplots(4, 3, figsize=(15, 14))
for k in range(11):
    r, c = k // 3, k % 3
    if k < len(shots):
        axes[r][c].imshow(shots[k])
        axes[r][c].set_title(labels[k])
    axes[r][c].set_xticks([])
    axes[r][c].set_yticks([])
fig.savefig("out/getup_shots.png", dpi=80)
print("saved out/getup_shots.png")
