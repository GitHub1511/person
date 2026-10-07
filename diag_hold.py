"""Isolate the motor layer: force the policy to 'hold' and see if it stands."""
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

cfg = SimConfig(duration=1.0, log_every=1000)
a = EmbodiedHuman(cfg, vision=False)

# force the posture target to the nominal configuration: no voluntary movement
a.inference.current = a.inference.policies[0]
a.inference.hold_timer = 1e9
a.voluntary_target = a.motor.q_nom.copy()

MASS = 71.68
cop_max = MASS * 9.81 * 0.155

print(f"CoP-limited ankle torque ~= {cop_max:.1f} Nm "
      f"(ankle actuator limit {a.motor.limits[a.motor.idx['ankle_l_flex']]:.0f} Nm)")
print(f"{'t':>5} {'pelz':>6} {'bal':>6} {'ey':>7} {'comvy':>7} {'ankL':>6} "
      f"{'tau_ank':>8} {'kneeL':>6} {'hipL':>6}")

for i in range(8000):
    a.step()
    # keep re-forcing hold every step
    a.inference.current = a.inference.policies[0]
    a.inference.hold_timer = 1e9
    a.voluntary_target = a.motor.q_nom.copy()
    st = a.state
    if i % 400 == 0:
        ey = st.com_over_support[1]
        print(f"{a.t:5.2f} {st.root_pos[2]:6.3f} {a.motor.balance_error:6.3f} "
              f"{ey:+7.4f} {st.com_vel[1]:+7.4f} {st.qof('ankle_l_flex'):+6.3f} "
              f"{st.tauof('ankle_l_flex'):+8.1f} {st.qof('knee_l'):+6.3f} "
              f"{st.qof('hip_l_flex'):+6.3f}")
    if st.root_pos[2] < 0.5:
        print(f"\nFELL at t={a.t:.2f}s")
        break
else:
    print(f"\nSTOOD for the full {a.t:.2f}s  final pelvis z={st.root_pos[2]:.3f}")
