# Embodied Human in MuJoCo

A simulated person, not a policy.

This is a 52-actuated-DoF humanoid in MuJoCo whose inside is meant to be
rich: dense skin, populations of sensory cells, eyes that dry out and blink,
organ systems, a circadian clock, a neural mass, episodic memory, 28 emotions,
25 neuromodulators, 21 drives, a predictive-coding brain that learns its own
body schema, and a generative behaviour space of ~10^37 describable behaviours.
It chooses what to do by minimising expected free energy against setpoints
that its own body supplies.

There is no task and no external reward. Everything the agent wants is
generated from inside its own body.

## What this repository contains, and how far each part is verified

| part | where | status |
|---|---|---|
| The base person: 1,872 taxels, 49,588 sensory scalars, 67 interoceptive variables, affect, drives, active inference (§2-§11) | `base` complexity | **measured** (reference run in §11); not re-run end to end since §17 was added |
| Whole-body control, walking, hands, speech, thought bubbles (§16) | `wbc.py`, `locomotion.py`, `skills.py`, `speech.py` | standing and grasping work in scripted tests; **walking is unreliable** |
| The "mind" hook for the Absolute Zero Reasoner (§16.5) | `mind.py`, `run_mind.py` | prompt/parse/dispatch tested against a fake server; **AZR itself has never been run** |
| Scalable complexity: skin density, sensory-cell populations, eyes, inner world (§17) | `complexity.py`, `senses_ext.py`, `ocular.py`, `inner_*.py` | built and unit-exercised; **sizes and speed at `rich`/`extreme` are not yet measured** (`tools/scale_complexity.py --measure` has not been completed) |
| Generative behaviour space + learned body safety (§17.5-§17.6) | `behavior_*.py`, `body_learning.py`, `tools/train_body.py` | trained and evaluated at `base`: falls per sim-hour 307 -> 87 (§17.6); still falls |
| Unattended planner + coder loop on free OpenRouter models (§18) | `tools/openrouter_helper/` | offline tests pass; two live steps ran and passed its checks but **both degraded olfaction/taste** (§18.1); **both were reverted**, the loop is stopped |

```bash
python -m pip install mujoco numpy matplotlib
python run_sim.py                    # 10 s episode + all figures
python run_sim.py --describe         # print the whole vector space and exit
python run_sim.py --duration 20 --render
PERSON_COMPLEXITY=base python run_sim.py     # the original body (§2-§11 numbers)
```

**Complexity default.**  The default level is now `extreme` (§17.1), so the
numbers quoted in §2-§11 (49,588 sensory scalars, 67 interoceptive variables,
~3 ms per step) describe the **`base`** level.  Use `PERSON_COMPLEXITY=base` to
reproduce them.

Outputs land in `out/`: a compressed `.npz` of every layer, CSVs, a JSON
manifest, six figures, and rendered frames.

---
