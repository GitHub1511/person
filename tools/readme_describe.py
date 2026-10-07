"""Dump agent.describe() (and a few extra counts) for the current PERSON_COMPLEXITY to JSON.
    PERSON_COMPLEXITY=base python tools/readme_describe.py out/readme_data/describe_base.json
Used to give the README measured numbers instead of remembered ones."""
import json, sys, time
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human import complexity

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return {"__array__": list(o.shape)}
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o

t0 = time.perf_counter()
a = EmbodiedHuman(SimConfig(duration=1.0, log_every=10), vision=True)
build = time.perf_counter() - t0
for _ in range(400): a.step()
out = {"complexity": complexity.C.name, "build_s": build, "describe": clean(a.describe())}
try: out["counts"] = clean(a._count_state())
except Exception as e: out["counts_err"] = str(e)
m = a.model
out["mj"] = {k: int(getattr(m, k)) for k in ("nq","nv","nu","nbody"," njnt","ngeom","nsite","nsensor","nsensordata","nmesh","ntendon","neq","nlight","ncam")  if hasattr(m,k)}
out["mass_kg"] = float(np.sum(m.body_mass))
out["latent_dim"] = int(a.latent_spec.dim) if hasattr(a.latent_spec,"dim") else None
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print("wrote", sys.argv[1], round(build,1), "s build")
