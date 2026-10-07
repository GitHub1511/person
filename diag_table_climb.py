"""Table-climb test: tip prone near the table, auto-recovery runs, report."""
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
ra = ag.meta.root_qpos_addr
ag.data.qpos[ra] = 0.0
ag.data.qpos[ra + 1] = -0.30
ag.data.qpos[ra + 2] = 0.45
ag.data.qpos[ra + 3:ra + 7] = [0.7071, 0.7071, 0, 0]
ag.data.qvel[:] = 0.0
mujoco.mj_forward(ag.model, ag.data)
for _ in range(1500):
    ag.step()
print("down: com=%.3f fallen=%s gap=%.2f"
      % (ag.state.com[2], ag.state.fallen, ag.skills._rec_furniture_gap()))
t0 = ag.t
peak = 0.0
while ag.t - t0 < 60.0:
    ag.step()
    peak = max(peak, float(ag.state.com[2]))
    if not ag.state.fallen and ag.state.com[2] > 0.72:
        break
print("end: com=%.3f fallen=%s peak=%.3f t=%.1f"
      % (ag.state.com[2], ag.state.fallen, peak, ag.t))
for e in ag.skills.events:
    if "stand_up" in e or "fallen" in e or "got back" in e:
        print(" ", e)
