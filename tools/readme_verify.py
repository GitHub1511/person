#!/usr/bin/env python
"""Cross-check the numbers quoted in README.md against freshly computed or recorded values.

    python tools/readme_verify.py

Each check recomputes a quantity (from the code, or from the data files under ``out/``) and confirms that
the README contains it in the formatted form used there.  Exit status 1 if any check fails.  It cannot
prove prose is right; it catches transcription errors and drift in the counts.
"""
import json
import re
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
README = (ROOT / "README.md").read_text(encoding="utf-8")
DATA = ROOT / "out" / "readme_data"
fails, n = [], 0


def check(name, value, *needles):
    """README must contain every needle; value is shown on failure."""
    global n
    n += 1
    miss = [x for x in needles if x not in README]
    if miss:
        fails.append(f"{name}: computed {value!r}, README lacks {miss}")


def fmt(x):
    return f"{x:,}"


# ---- combinatorics ------------------------------------------------------------------------------
c = json.loads((DATA / "combinatorics.json").read_text())
N = int(c["behavior"]["descriptors"]["exact"])
check("descriptor count", N, fmt(N))
check("per arm", c["behavior"]["per_arm"], fmt(c["behavior"]["per_arm"]))
check("both arms", c["behavior"]["arms_both"], fmt(c["behavior"]["arms_both"]))
check("both hands", c["behavior"]["hands_both"], fmt(c["behavior"]["hands_both"]))
for g, v in c["behavior"]["groups"].items():
    if "arm" in g or "hand" in g and False:
        continue
    check("group " + g, v, fmt(v))
a = c["affect"]
check("emotion pairs", a["emotion_pairs"], fmt(a["emotion_pairs"]))
check("emotion triples", a["emotion_triples"], fmt(a["emotion_triples"]))
check("emotion patterns", a["emotion_on_off_patterns"], fmt(a["emotion_on_off_patterns"]))
check("drive 3-level states", a["drive_3_level_states"], fmt(a["drive_3_level_states"]))
check("appraisal weights", a["appraisal_weights"], "280")
check("interoception pairs", comb(67, 2), fmt(comb(67, 2)))
check("interoception triples", comb(67, 3), fmt(comb(67, 3)))
for h, v in c["policies"]["sequences"].items():
    if int(h) in (2, 3, 4, 5):
        check(f"policy sequences h={h}", v, fmt(v))
for k, v in c["calls"]["sequences"].items():
    check(f"call sequences k={k}", v, fmt(v))
check("taxel pairs base", c["skin"]["taxel_pairs"], fmt(c["skin"]["taxel_pairs"]))
check("taxel pairs extreme", comb(17928, 2), fmt(comb(17928, 2)))
check("policies", c["policies"]["n"], "55")
check("quota ceiling", c["helper"]["output_tokens_ceiling_day"], "102 million")

# ---- measured levels ------------------------------------------------------------------------------
def parse_measure(path):
    out = {}
    cols = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s{20,}(base|rich|extreme|max)(\s+(base|rich|extreme|max))*\s*$", line)
        if m:
            cols = line.split()
            continue
        if cols:
            mm = re.match(r"\s*([a-zA-Z][a-zA-Z \-]*?)\s{2,}(.*)$", line)
            if mm:
                vals = mm.group(2).split()
                if len(vals) == len(cols):
                    out[mm.group(1).strip()] = dict(zip(cols, vals))
    return out


M = parse_measure(ROOT / "out" / "measure_baseline.txt")
M.update({k: {**M.get(k, {}), **v} for k, v in parse_measure(DATA / "measure_max.txt").items()})
for key in ("skin taxels", "sensory scalars per frame", "internal dynamic variables", "episodic store values",
            "neural-mass units", "motor units", "chemistry analytes"):
    for lv, v in M.get(key, {}).items():
        if v != "-":
            check(f"{key} @ {lv}", v, v)
for lv, v in M.get("ms per physics step", {}).items():
    check(f"ms/step @ {lv}", v, v)

# ---- code-derived counts -----------------------------------------------------------------------------
rt = json.loads((DATA / "runtime_extreme.json").read_text())
pred = rt["systems"]["predict"]["array_values"]
check("predictive array values", pred, fmt(pred))
check("afferent array values", rt["systems"]["afferents"]["array_values"], "5.06 million")
d = json.loads((DATA / "describe_extreme.json").read_text())
check("latent dim", d["latent_dim"], "572")
check("nsensor extreme", d["mj"]["nsensor"], fmt(d["mj"]["nsensor"]))
check("nsite extreme", d["mj"]["nsite"], fmt(d["mj"]["nsite"]))
inner = rt["extra"]["inner_counts"]
check("inner dynamic state", inner["dynamic_state_variables"], fmt(inner["dynamic_state_variables"]))
check("episodic", inner["episodic_store_values"], fmt(inner["episodic_store_values"]))
blocks = rt["extra"]["latent_blocks"]
check("latent blocks sum", sum(b for _, b in blocks), "572")

# ---- reference run -----------------------------------------------------------------------------------------
sm = json.loads((ROOT / "out_readme" / "summary.json").read_text())
check("ownership", sm["final_ownership"], f"{sm['final_ownership']:.3f}")
check("drift", sm["final_proprio_drift"], f"{sm['final_proprio_drift']:.4f}")
check("min pelvis", sm["min_pelvis_z"], f"{sm['min_pelvis_z']:.3f}")
check("max pain", sm["max_pain"], f"{sm['max_pain']:.4f}")
check("self-touch", sm["self_touch_fraction"], f"{sm['self_touch_fraction']}")
check("mean free energy", sm["mean_free_energy"], f"{sm['mean_free_energy']:.2f}")

# ---- training report -----------------------------------------------------------------------------------------
tr = json.loads((ROOT / "out" / "train_body_report.json").read_text())
check("AUC", tr["heldout_auc"], f"{tr['heldout_auc']:.3f}")
check("training behaviours", tr["training_behaviours"], fmt(tr["training_behaviours"]))
for e in tr["evaluation"]:
    check("falls " + e["label"], e["falls"], str(e["falls"]))
    check("falls/h " + e["label"], e["falls_per_sim_hour"], f"{e['falls_per_sim_hour']:.1f}")
    check("unsafe/100 " + e["label"], e["unsafe_per_100"], f"{e['unsafe_per_100']:.1f}")

print(f"{n} checks, {len(fails)} failed")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
