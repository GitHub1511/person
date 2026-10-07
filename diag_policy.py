"""Diagnostic: why does the agent choose destabilising policies?"""
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

cfg = SimConfig(duration=2.0, log_every=1000)
a = EmbodiedHuman(cfg, vision=False)

E = a.inference
names = [p.name for p in E.policies]
cats = [p.category for p in E.policies]

print(f"{'t':>5} {'policy':16} {'bal':>6} {'risk':>8} {'bias':>7} {'G':>8} | top-4 by G")
for i in range(2000):
    a.step()
    if i % 100 == 0:
        info = a.last_inference_info
        if "risk" not in info:
            continue
        risk, bias, G = info["risk"], info["bias"], E.last_G
        order = np.argsort(G)[:4]
        tops = " ".join(f"{names[j]}({cats[j][:4]})G={G[j]:+.2f}" for j in order)
        cur = names.index(a.inference.current.name)
        print(f"{a.t:5.2f} {a.inference.current.name:16} "
              f"{a.motor.balance_error:6.3f} {risk[cur]:8.2f} {bias[cur]:+7.2f} "
              f"{G[cur]:+8.2f} | {tops}")

st = a.state
print(f"\nfinal pelvis z={st.root_pos[2]:.3f} fallen={st.fallen}")
# biggest joint deviations from nominal
dev = np.abs(st.q - a.motor.q_nom)
order = np.argsort(-dev)[:10]
print("largest joint deviations:")
for j in order:
    print(f"   {a.motor.names[j]:18s} q={st.q[j]:+.3f} nom={a.motor.q_nom[j]:+.3f} "
          f"target={a.voluntary_target[j]:+.3f} dev={dev[j]:.3f}")
print(f"postural priority = {a.motor.postural_priority:.4f}")
print(f"motor strategy    = {np.round(a.motor_frame.strategy_weight,3)}")
print(f"stepping state    = {a.motor.step_state}")
