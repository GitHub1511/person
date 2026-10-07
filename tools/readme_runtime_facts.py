"""Inventory of what the running person is made of (array sizes per subsystem), for the README.
    PYTHONPATH=. PERSON_COMPLEXITY=extreme python tools/readme_runtime_facts.py out/readme_data/runtime_extreme.json"""
import json, sys
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human import complexity

a = EmbodiedHuman(SimConfig(duration=1.0, log_every=10), vision=True)
for _ in range(300): a.step()

def inventory(obj, depth=2, seen=None, prefix=""):
    seen = seen if seen is not None else set()
    tot, items = 0, []
    if id(obj) in seen or depth < 0: return 0, items
    seen.add(id(obj))
    d = getattr(obj, "__dict__", None)
    if d is None: return 0, items
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            tot += v.size; items.append((prefix + k, list(v.shape), int(v.size)))
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], np.ndarray):
            n = sum(x.size for x in v if isinstance(x, np.ndarray)); tot += n; items.append((prefix + k + "[]", [len(v)], int(n)))
        elif isinstance(v, dict) and v and all(isinstance(x, np.ndarray) for x in list(v.values())[:3]):
            n = sum(x.size for x in v.values() if isinstance(x, np.ndarray)); tot += n; items.append((prefix + k + "{}", [len(v)], int(n)))
        elif hasattr(v, "__dict__") and not callable(v) and depth > 0 and not isinstance(v, type):
            t2, i2 = inventory(v, depth - 1, seen, prefix + k + "."); tot += t2; items += i2
    return tot, items

systems = ["receptors", "afferents", "interoception", "affect", "drives", "predict", "inference", "motor",
           "gait", "skills", "behavior", "inner", "ocular", "latent_spec", "motivation", "precision", "action"]
out = {"complexity": complexity.C.name, "systems": {}}
for s in systems:
    o = getattr(a, s, None)
    if o is None: continue
    tot, items = inventory(o, depth=3)
    items.sort(key=lambda r: -r[2])
    out["systems"][s] = {"array_values": int(tot), "top": items[:8]}
    print(f"{s:14s} {tot:>14,d} array values", [(n, sh) for n, sh, _ in items[:3]])
# specific attributes
ex = {}
def tryget(name, f):
    try: ex[name] = f()
    except Exception as e: ex[name] = f"ERR {type(e).__name__}: {e}"
tryget("n_policies", lambda: len(a.inference.policies))
tryget("policy_names", lambda: [p.name for p in a.inference.policies])
tryget("emotions", lambda: list(__import__("embodied_human.affect", fromlist=["x"]).EMOTIONS))
tryget("neuromodulators", lambda: list(__import__("embodied_human.affect", fromlist=["x"]).NEUROMODULATORS))
tryget("appraisal", lambda: list(__import__("embodied_human.affect", fromlist=["x"]).APPRAISAL))
tryget("n_appraisal_weights", lambda: list(__import__("embodied_human.affect", fromlist=["x"]).APPRAISAL_W.shape))
tryget("drives", lambda: list(__import__("embodied_human.drives", fromlist=["x"]).DRIVES))
tryget("latent_blocks", lambda: [(n, int(s)) for n, s in a.latent_spec.blocks] if hasattr(a.latent_spec, "blocks") else None)
tryget("gait_states", lambda: [s for s in dir(a.gait) if s.isupper()])
tryget("inner_rates", lambda: dict(a.inner.RATES))
tryget("inner_counts", lambda: a.inner.counts())
tryget("behavior_size", lambda: a.behavior.space.size() if hasattr(a.behavior, "space") else None)
out["extra"] = ex
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
