# Embodied Human in MuJoCo

A simulated person, rather than a policy.

This work describes a humanoid with 52 actuated degrees of freedom (DoF) in MuJoCo, whose
internal organisation is intended to be rich: dense skin, populations of sensory cells,
eyes that undergo desiccation and blinking, organ systems, a circadian clock, a neural mass,
episodic memory, 28 emotions, 25 neuromodulators, 21 drives, a predictive-coding brain that
learns its own body schema, and a generative behaviour space of ~10^37 describable
behaviours. Action selection proceeds by minimising expected free energy with respect to
setpoints that are supplied by the agent's own body.

The agent is given no task and no external reward. All of its motivational content is
generated internally, from its own body.

## Contents of the repository, and the extent to which each part has been verified

| part | where | status |
|---|---|---|
| The base person: 1,872 taxels, 49,588 sensory scalars, 67 interoceptive variables, affect, drives, active inference (§2-§11) | `base` complexity | **measured** (reference run in §11); not re-run end to end since the addition of §17 |
| Whole-body control, walking, hands, speech, thought bubbles (§16) | `wbc.py`, `locomotion.py`, `skills.py`, `speech.py` | standing and grasping function in scripted tests; **walking is unreliable** |
| The "mind" interface to the Absolute Zero Reasoner (AZR) (§16.5) | `mind.py`, `run_mind.py` | prompt construction, response parsing and dispatch were tested against a mock server; **AZR itself has never been run** |
| Scalable complexity: skin density, sensory-cell populations, eyes, inner world (§17) | `complexity.py`, `senses_ext.py`, `ocular.py`, `inner_*.py` | implemented and exercised at unit level; **sizes and speed at `rich`/`extreme` have not yet been measured** (the run of `tools/scale_complexity.py --measure` has not been completed) |
| Generative behaviour space and learned body safety (§17.5-§17.6) | `behavior_*.py`, `body_learning.py`, `tools/train_body.py` | trained and evaluated at `base`: falls per simulated hour were reduced from 307 to 87 (§17.6); falls nevertheless persist |
| Unattended planner-and-coder loop using free OpenRouter models (§18) | `tools/openrouter_helper/` | offline tests pass; two live steps were executed and passed the loop's checks, but **both degraded olfaction and taste** (§18.1); **both were reverted**, and the loop is stopped |

```bash
python -m pip install mujoco numpy matplotlib
python run_sim.py                    # 10 s episode + all figures
python run_sim.py --describe         # print the whole vector space and exit
python run_sim.py --duration 20 --render
PERSON_COMPLEXITY=base python run_sim.py     # the original body (§2-§11 numbers)
```

**Complexity default.**  The default level is currently `extreme` (§17.1); consequently, the
figures reported in §2-§11 (49,588 sensory scalars, 67 interoceptive variables, ~3 ms per
step) describe the **`base`** level.  They are reproduced by setting
`PERSON_COMPLEXITY=base`.

Outputs are written to `out/`: a compressed `.npz` file containing every layer, comma-separated
values (CSV) files, a JavaScript Object Notation (JSON) manifest, six figures, and rendered
frames.

---
