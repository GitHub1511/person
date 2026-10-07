"""Drive the real SkillSystem._rec_cobra in isolation, report peak COM."""
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
gen = sk._a_stand_up()
# fast-forward through curl/extract/roll/hand-plant by running to cobra:
# instead run only up to cobra via direct staging: plant then cobra
sk.recovery_targets = {"elbow_l": -1.3, "elbow_r": -1.3,
                       "sh_l_flex": 0.15, "sh_r_flex": 0.15,
                       "hip_l_flex": -0.1, "hip_r_flex": -0.1,
                       "knee_l": 0.2, "knee_r": 0.2}
for _ in range(300):
    ag.step()
print("planted com=%.3f" % ag.state.com[2])
peak = 0.0
try:
    g = sk._rec_cobra()
    while True:
        ag.step()
        peak = max(peak, float(ag.state.com[2]))
        next(g)
except StopIteration:
    pass
except Exception as exc:
    print("cobra raised:", exc)
print(f"peak com={peak:.3f} end com={ag.state.com[2]:.3f}")
print("events:", [e for e in sk.events if "cobra" in e][-3:])
