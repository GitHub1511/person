"""Isolate roll+lunge: side placement, drive _rec_roll only, report."""
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
ag.data.qpos[ra] = 1.2
ag.data.qpos[ra + 1] = 0.6
ag.data.qpos[ra + 2] = 0.45
ag.data.qpos[ra + 3:ra + 7] = [0.7071, 0.7071, 0, 0]
ag.data.qvel[:] = 0.0
mujoco.mj_forward(ag.model, ag.data)
for _ in range(1500):
    ag.step()
print("down: com=%.3f" % ag.state.com[2])
sk.recovery_active = True
sk._recover_attempts = 2  # even -> lunge left forward
peak = 0.0
try:
    g = sk._rec_roll()
    t0 = ag.t
    while ag.t - t0 < 40.0:
        ag.step()
        peak = max(peak, float(ag.state.com[2]))
        try:
            next(g)
        except StopIteration:
            break
except Exception as exc:
    print("roll raised:", exc)
print(f"peak={peak:.3f} end={ag.state.com[2]:.3f}")
print([e for e in sk.events if "stand_up" in e])