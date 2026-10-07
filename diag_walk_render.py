"""Render hand-walk directly from prone placement."""
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

renderer = mujoco.Renderer(ag.model, 360, 480)
opt = mujoco.MjvOption()
opt.sitegroup[:] = 0
shots = []
labels = []


def snap(tag):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = ag.state.com
    cam.distance = 2.4
    cam.elevation = -8.0
    cam.azimuth = 135.0
    renderer.update_scene(ag.data, camera=cam, scene_option=opt)
    shots.append(renderer.render().copy())
    labels.append(f"{tag} com{ag.state.com[2]:.2f}")


snap("prone ")
sk.recovery_active = True
sk._rec_hold_table = True
sk.recovery_targets = {"elbow_l": -1.3, "elbow_r": -1.3,
                       "knee_l": 0.1, "knee_r": 0.1,
                       "hip_l_flex": -0.05, "hip_r_flex": -0.05,
                       "ankle_l_flex": 0.4, "ankle_r_flex": 0.4,
                       "spine_bend": 0.5, "chest_bend": 0.3}
g = sk._rec_hand_walk()
i = 0
done = False
while not done:
    for _ in range(250):
        ag.step()
        try:
            next(g)
        except StopIteration:
            done = True
            break
        except Exception as exc:
            print("walk raised:", exc)
            done = True
            break
    i += 1
    snap(f"walk+{i * 5}s ")
    if i >= 7:
        break
print("events:", [e for e in sk.events if "stand_up" in e])
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for k in range(8):
    ax = axes[k // 4][k % 4]
    ax.imshow(shots[k])
    ax.set_title(labels[k])
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig("out/getup_walk.png", dpi=80)
print("saved out/getup_walk.png")
