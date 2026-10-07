"""Smoke test: build the agent, run a short episode, report everything."""
import time
import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

t0 = time.perf_counter()
cfg = SimConfig(duration=2.0, log_every=10)
a = EmbodiedHuman(cfg, vision=True)
print(f"build: {time.perf_counter()-t0:.2f} s")

d = a.describe()
print("\n=== MODEL ===")
for k, v in d["model"].items():
    print(f"  {k:22s} {v}")
print("\n=== LATENT ===")
print(f"  dim={d['latent']['dim']}")
for name, size in d["latent"]["blocks"]:
    print(f"    {name:12s} {size}")
print("\n=== SCALAR STATE CHANNELS ===")
for k, v in d["total_scalar_state"].items():
    print(f"  {k:26s} {v}")

print("\n=== SENSES ===")
s = d["senses"]
print(f"  tactile: {s['tactile']['n_taxels']} taxels x "
      f"{s['tactile']['channels_per_taxel']} = "
      f"{s['tactile']['total_scalar_channels']} channels")
print(f"  proprio: {s['proprioceptive']['n_joints']} joints x "
      f"{s['proprioceptive']['channels_per_joint']} = "
      f"{s['proprioceptive']['total_scalar_channels']} channels")
print(f"  emotions: {len(d['affect']['emotions'])}")
print(f"  neuromodulators: {len(d['affect']['neuromodulators'])}")
print(f"  policies: {d['inference']['n_policies']}")

print("\n=== RUN ===")
t1 = time.perf_counter()
steps = a.run(progress=True)
wall = time.perf_counter() - t1
sim = cfg.duration
print(f"\nwall={wall:.2f}s  sim={sim:.2f}s  speed={sim/wall:.2f}x realtime")
per_step = np.array(a.step_durations)
print(f"per physics step: mean={per_step.mean()*1e3:.3f} ms  "
      f"p95={np.percentile(per_step,95)*1e3:.3f} ms")

print("\n=== OUTCOME ===")
last = steps[-1]
print(f"  pelvis z     = {a.state.root_pos[2]:.3f}  (fell={a.state.fallen})")
print(f"  COM          = {np.round(a.state.com,3)}")
print(f"  balance err  = {a.motor.balance_error:.4f}")
print(f"  contacts     = {a.state.n_contact}  self_touch={a.state.self_touch}")
print(f"  valence      = {last.valence:+.3f}")
print(f"  arousal      = {last.arousal:+.3f}")
print(f"  dominance    = {last.dominance:+.3f}")
print(f"  stress       = {last.stress:+.3f}")
print(f"  dominant emo = {last.dominant_emotion}")
print(f"  policy       = {last.policy}")
print(f"  free energy  = {last.free_energy:.3f}")
print(f"  reward       = {last.reward:+.4f}")
print(f"  pain         = {last.pain:.3f}  touch={last.touch_intensity:.3f}")
print(f"  drives       = {a.drive_frame.as_dict()}")
print(f"  heart rate   = {a.interoception.s[16]:.1f}")

print("\n  top emotions:", [f"{n}={v:.2f}" for n, v in
                            a.affect_frame.top_emotions(6)])
print("  neuromodulators:", {n: round(float(a.affect.nm[i]), 2)
                             for i, n in enumerate(
                                 __import__("embodied_human.affect", fromlist=["x"])
                                 .NEUROMODULATORS) if a.affect.nm[i] > 0.35})

print("\n=== TACTILE SANITY ===")
tt = a.frame.tactile
print(f"  shape={tt.shape}")
print(f"  taxels with contact : {int((tt[:,0]>0.05).sum())}")
print(f"  max normal force    : {tt[:,0].max():.1f} N")
print(f"  max pressure        : {tt[:,3].max():.1f} kPa")
print(f"  max temperature     : {tt[:,23].max():.1f} C  "
      f"min {tt[:,23].min():.1f} C")
top = np.argsort(-tt[:,0])[:6]
from embodied_human import skin
for i in top:
    print(f"    {skin.TAXELS[i].region:16s} {skin.TAXELS[i].surface:22s} "
          f"F={tt[i,0]:6.2f}N p={tt[i,3]:6.1f}kPa T={tt[i,23]:.2f}C "
          f"sa1={tt[i,12]:.2f} fa2={tt[i,15]:.2f}")

print("\n=== PREDICTION ===")
p = a.pred_frame
print(f"  free energy   = {p.free_energy:.3f}")
print(f"  inaccuracy    = {p.inaccuracy:.3f}   complexity={p.complexity:.3f}")
print(f"  surprise      = {p.surprise:.2f}")
print(f"  proprio drift = {p.proprioceptive_drift:.5f}")
print(f"  empowerment   = {p.empowerment:.3f}")
print(f"  mean ownership= {p.body_ownership.mean():.3f}")
print(f"  forward params= {a.predict.forward.W.size}")
