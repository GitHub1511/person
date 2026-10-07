"""Diagnostic: does the body stay upright, and for how long?"""
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

cfg = SimConfig(duration=1.0, log_every=1000)
a = EmbodiedHuman(cfg, vision=False)

print(f"{'t':>5} {'pelz':>6} {'upright':>7} {'bal':>6} {'prio':>5} "
      f"{'strat(ank/hip/toe/knee)':>26} {'kneeL':>6} {'ankL':>6} {'step':>8} policy")
E = a.inference
hist = []
for i in range(6000):
    a.step()
    st = a.state
    if i % 250 == 0:
        s = a.motor_frame.strategy_weight
        print(f"{a.t:5.2f} {st.root_pos[2]:6.3f} {float(st.root_pos[2]):7.3f} "
              f"{a.motor.balance_error:6.3f} {a.motor.postural_priority:5.2f} "
              f"  {s[0]:5.2f}{s[1]:6.2f}{s[2]:6.2f}{s[3]:6.2f}          "
              f"{st.qof('knee_l'):+6.2f} {st.qof('ankle_l_flex'):+6.2f} "
              f"{a.motor.step_state:>8} {a.inference.current.name}")
    hist.append(st.root_pos[2])
    if st.root_pos[2] < 0.5:
        print(f"\nFELL at t={a.t:.2f}s  (pelvis z={st.root_pos[2]:.3f})")
        break

h = np.array(hist)
print(f"\nmin pelvis z = {h.min():.3f}  final = {h[-1]:.3f}  "
      f"survived {len(h)*a.dt:.2f}s")
print(f"steps taken (protective stepping events) = {a.motor.step_count}")
