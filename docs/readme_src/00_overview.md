# Embodied Human in MuJoCo

**A simulated person, not a policy.** The project is a humanoid model in the MuJoCo physics engine
whose interior is deliberately rich: dense artificial skin, populations of sensory cells, eyes that
dry and blink, organ systems, a circadian clock, a recurrent neural mass, episodic memory, 28 emotions,
25 neuromodulators, 21 homeostatic drives, a predictive-coding brain that learns the structure of its
own body, and a generative behaviour space of 1.16 × 10³⁷ describable behaviours (123.1 bits).
Action is selected by minimising expected free energy against setpoints that the body itself supplies.
No external task or reward is defined; every motivation arises internally.

All quantities in this document were obtained by executing the code on 2026-10-07 unless a statement
says otherwise (Windows 11, Python 3.14, MuJoCo 3.15, numpy 2.3; i7-14700, 28 threads). The scripts that
produced them are listed in Appendix B and can be rerun. Where a figure is an estimate, a design target
or a count of what is *describable* rather than what has been *demonstrated*, the text says so.

## At a glance

| quantity | value | where measured |
|---|---|---|
| Python modules in `embodied_human/` | {{MODSTATS}} | static analysis (`ast`), §12 |
| Complexity levels | 6 (`base`, `rich`, `extreme`, `max`, `ultra`, `mega`) | `complexity.py` |
| Actuated degrees of freedom | 52 (qpos 94, qvel 88) | MuJoCo model |
| Bodies / joints / geoms | 44 / 58 / 68 (58 colliding, 10 visual only) | MuJoCo model, §2 |
| Skin taxels | 1,992 (`base`), 7,968 (`rich`), 17,928 (`extreme`), 31,872 (`max`) | `skin.py`, §3 |
| Sensory scalars per frame (vision off) | 52,981 / 346,406 / 778,054 / 1,384,030 at `base` / `rich` / `extreme` / `max` | `tools/scale_complexity.py --measure` |
| Interoceptive variables | 67 in 11 systems | §4 |
| Emotions / neuromodulators / drives | 28 / 25 / 21 | §5, §6 |
| Latent state of the brain | 572 dimensions in 11 blocks | §7 |
| Motor programs | 55 hand-written policies; 1.16 × 10³⁷ generative descriptors | §8, §17 |
| Internal dynamic variables | 0 (`base`), 15,066 (`rich`), 36,932 (`extreme`), 88,892 (`max`) | §17 |
| Speed | 0.54× (`base`), 0.42× (`rich`), 0.25× (`extreme`), 0.16× (`max`) real time on the reference machine | §17 |

## Verification status

The sections of this document differ in how thoroughly they have been tested. This table is the
authoritative summary; it is deliberately conservative.

| part | status |
|---|---|
| Body, sensors, interoception, affect, drives, predictive coding, active inference (§2–§11) | Implemented and run. A 20 s reference run at `base` (seed 7) completed without a fall (§11). |
| Whole-body control, bipedal walking, arm and hand skills, speech (§16.1–§16.4) | Standing and scripted reach/grasp/hold work in tests; **walking is unreliable** (§16.2); grasping uses an assist weld (§16.3). |
| Fall recovery (`stand_up`, §16.7) | Under active development in this repository; its success rate is reported only where a diagnostic was run (§16.7). |
| AZR as the live mind (§16.5–§16.6) | The 3B model in LM Studio answers in the expected format; a live diagnostic (`diag_azr_live.py`, 2026-10-07) produced a real thought, parsed two calls and left the body standing. Calls pass through a precondition gate. Its competence as an embodied agent, and any improvement in navigation, are **not established**; weights are not fine-tuned. |
| Complexity levels, eyes, inner world (§17.1–§17.5) | Implemented; sizes and speed measured at `base`, `rich`, `extreme` and `max`; end-to-end behaviour at the higher levels is less tested. |
| Learned body safety (§17.6) | Trained and evaluated at `base`: falls per simulated hour 307 → 87; the person still falls. |
| Planner/coder loop and its tools (§18) | Offline tests pass; two live coder steps were reverted because they degraded olfaction and taste. |
| Ultra tier (§19) | Framework, benchmark tools and conventions exist; **no domain subsystem has been built**: the design and build workflows returned no output. |

## Reproduction

```bash
python -m pip install mujoco numpy matplotlib
python run_sim.py                          # 10 s episode and all figures (default level: extreme)
PERSON_COMPLEXITY=base python run_sim.py --duration 20 --seed 7 --no-figures
python run_sim.py --describe               # print the whole vector space and exit
python tools/scale_complexity.py --measure base rich extreme   # sizes and speed per level
python run_mind.py --at-table --backend server --url http://127.0.0.1:1234/v1 --model-name absolute_zero_reasoner-coder-3b
```

Outputs are written to `out/`: a compressed `.npz` of every layer, CSV files, a JSON manifest, six
figures and rendered frames. The default complexity level is now `extreme`; the numbers of §2–§11
describe the `base` level unless stated, and `PERSON_COMPLEXITY=base` reproduces them.

---
