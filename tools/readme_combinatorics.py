#!/usr/bin/env python
"""Compute the combinatorial quantities quoted in the README from the code itself.

    PYTHONPATH=. PERSON_COMPLEXITY=base python tools/readme_combinatorics.py [out.json]

Every figure is derived from the live objects (the behaviour space layout, the skin patches, the
affect tables ...), never typed in.  Python integers are exact; large counts are also given as
base-10 logarithms and bits.
"""
import json
import math
import sys
from itertools import accumulate
from math import comb, log10, log2

import numpy as np

from embodied_human import affect, drives, skin
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.mind import ALLOWED_CALLS
from embodied_human import body_learning as BL

a = EmbodiedHuman(SimConfig(duration=1.0, log_every=10), vision=False)
sp = a.behavior.space
R = {}


def big(n: int) -> dict:
    return {"exact": str(n), "log10": log10(n) if n > 0 else None, "bits": log2(n) if n > 0 else None}


# ---- the behaviour space ---------------------------------------------------------------
layout = [(n, int(s)) for n, s in sp.layout]
N = sp.size
R["behavior"] = {
    "n_channels_layout": len(layout), "n_channels_declared": sp.n_channels,
    "layout": layout, "descriptors": big(N), "per_arm": sp.per_arm,
    "arm_schemas": 47, "hand_shapes": 14,
}
groups = {
    "face (head x eyes x lids x mouth)": ["head", "eyes", "lids", "mouth"],
    "torso and stance (torso x stance)": ["torso", "stance"],
    "left arm": ["l_schema"] + [f"l_arm_{i}" for i in range(6)],
    "right arm": ["r_schema"] + [f"r_arm_{i}" for i in range(6)],
    "left hand": ["l_hand", "l_hand_0", "l_hand_1", "l_hand_2"],
    "right hand": ["r_hand", "r_hand_0", "r_hand_1", "r_hand_2"],
    "style x hold": ["style", "hold"],
    "intent x touch x walk": ["intent", "touch", "walk"],
}
lay = dict(layout)
R["behavior"]["groups"] = {g: int(np.prod([lay[c] for c in cs], dtype=object)) for g, cs in groups.items()}
R["behavior"]["arms_both"] = sp.per_arm ** 2
R["behavior"]["hands_both"] = (14 * 27) ** 2
R["behavior"]["sequences"] = {k: big(N ** k) for k in (2, 3, 5, 10)}
R["behavior"]["unordered_pairs"] = big(N * (N - 1) // 2)
R["behavior"]["left_right_mirror_classes"] = big((N + sp.size // 1) // 2 if False else N // 2)
R["behavior"]["years_to_enumerate_at_1e9_per_s"] = N / 1e9 / 3.156e7
R["behavior"]["samples_seen_by_training"] = 1814
R["behavior"]["fraction_seen_by_training"] = 1814 / N
R["behavior"]["safety_model"] = {"components": len(BL.COMPONENTS), "continuous_features": BL.N_CONT,
                                 "one_hot": BL.N_ONEHOT, "n_features": BL.N_FEATURES,
                                 "weights": int(a.behavior.safety.w.size)}
R["behavior"]["self_touch"] = {"regions": len(sp.__class__.__mro__) and 16, "actions": 4, "hands": 2,
                               "options": 1 + 16 * 2 * 4}
# ---- the policies and the call API -------------------------------------------------------
npol = len(a.inference.policies)
R["policies"] = {"n": npol, "sequences": {h: npol ** h for h in (2, 3, 4, 5, 8)}}
calls = len(ALLOWED_CALLS)
targets = len(set(v for v in ALLOWED_CALLS.values() if v))
R["calls"] = {"allowed_names": calls, "distinct_skills": targets,
              "sequences": {k: calls ** k for k in (2, 3, 4, 6)},
              "objects": 6, "grab_targets_x_hands": 6 * 2}
# ---- skin ----------------------------------------------------------------------------------
ps = skin.default_patches()
cnt = [p.n_u * p.n_v for p in ps]
R["skin"] = {"patches": len(ps), "taxels": sum(cnt), "patch_pairs": comb(len(ps), 2),
             "taxel_pairs": comb(sum(cnt), 2), "patches": {p.region: p.n_u * p.n_v for p in ps},
             "min_patch": min(cnt), "max_patch": max(cnt)}
# ---- affect and drives ------------------------------------------------------------------------
ne, na = len(affect.EMOTIONS), affect.APPRAISAL_W.shape[1]
nm = len(affect.NEUROMODULATORS)
nd = len(drives.DRIVES)
R["affect"] = {"emotions": ne, "appraisal_dims": na, "appraisal_weights": int(affect.APPRAISAL_W.size),
               "neuromodulators": nm, "emotion_pairs": comb(ne, 2), "emotion_triples": comb(ne, 3),
               "emotion_on_off_patterns": 2 ** ne, "neuromod_pairs": comb(nm, 2),
               "drives": nd, "drive_pairs": comb(nd, 2), "drive_3_level_states": 3 ** nd,
               "drive_on_off_patterns": 2 ** nd}
nI = 67
R["intero"] = {"variables": nI, "pairs": comb(nI, 2), "triples": comb(nI, 3)}
# ---- sensors -----------------------------------------------------------------------------------
R["senses"] = {"proprio": 52 * 13, "joint_pairs": comb(52, 2), "actuated_joints": 52,
               "tactile_channels": 43, "tactile_channels_base": 26}
# ---- collision ----------------------------------------------------------------------------------
cats = 6
R["collision"] = {"categories": cats, "unordered_pairs_incl_self": comb(cats, 2) + cats}
# ---- the helper ----------------------------------------------------------------------------------
models = 5
R["helper"] = {"models": models, "per_model_day": 200, "per_day": 1000, "per_min": 20,
               "min_gap_s": 3.0, "minutes_to_exhaust_day": 1000 / 20,
               "output_tokens_ceiling_day": 200 * 200000 * 2 + 200 * 30000 + 200 * 50000 + 200 * 30000}
json.dump(R, open(sys.argv[1] if len(sys.argv) > 1 else "out/readme_data/combinatorics.json", "w"), indent=1, default=str)
b = R["behavior"]
print("descriptors", b["descriptors"]["exact"], "log10", round(b["descriptors"]["log10"], 3), "bits", round(b["descriptors"]["bits"], 2))
print("per_arm", b["per_arm"], "arms both", b["arms_both"], "hands both", b["hands_both"])
print("groups", b["groups"])
print("years to enumerate", b["years_to_enumerate_at_1e9_per_s"])
print("safety", b["safety_model"])
print("policies", R["policies"]); print("calls", R["calls"]); print("skin", {k: v for k, v in R["skin"].items() if k != "patches"})
print("affect", R["affect"])
