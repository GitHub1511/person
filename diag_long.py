"""Trace a long episode to find when and why it falls."""
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

cfg = SimConfig(duration=12.0, log_every=10**9, seed=7)
a = EmbodiedHuman(cfg, vision=False)
n = int(12.0 / a.dt)
prev_pol = None
for i in range(n):
    a.step()
    st = a.state
    pol = a.inference.current.name
    fell_now = st.root_pos[2] < 0.55
    if i % 400 == 0 or (pol != prev_pol and i % 40 == 0) or fell_now:
        print(f"t={a.t:5.2f} z={st.root_pos[2]:6.3f} bal={a.motor.balance_error:5.3f} "
              f"prio={a.motor.postural_priority:4.2f} "
              f"ey={st.com_over_support[1]:+6.3f} vy={st.com_vel[1]:+6.3f} "
              f"pol={pol:16s} tgt_hipL={a.voluntary_target[a.motor.idx['hip_l_flex']]:+.2f} "
              f"q_hipL={st.qof('hip_l_flex'):+.2f} "
              f"q_kneeL={st.qof('knee_l'):+.2f}")
        prev_pol = pol
    if fell_now:
        print(f"\nFELL at t={a.t:.2f}s")
        break
print(f"\nfinal z={a.state.root_pos[2]:.3f}")
