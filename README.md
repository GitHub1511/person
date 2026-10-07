# Embodied Human in MuJoCo

A simulated person, not a policy. This is a humanoid model in MuJoCo with rich internal physiology: dense skin, populations of sensory cells, eyes that dry and blink, organ systems, a circadian clock, a neural mass, episodic memory, 28 emotions, 25 neuromodulators, 21 drives, a predictive‑coding brain that learns its own body schema, and a generative behaviour space of ~10^37 describable behaviours. It chooses actions by minimising expected free energy against setpoints supplied by its own body. There is no external task or reward; all motivations arise internally.

## Module Inventory

**embodied_human/** (39 files, 18 716 lines total)
- `__init__.py`: 3 lines – Embodied human: a simulated person, not a policy.
- `_fast.py`: 16 lines – (unparseable: SyntaxError)
- `active_inference.py`: 410 lines – Active inference: choosing what to do by minimising expected free energy.
- `affect.py`: 602 lines – Affect: neuromodulators, appraisal, core affect, emotions, mood, temperament.
- `afferents.py`: 350 lines – Peripheral nerve: receptor potentials → afferent traffic.
- `agent.py`: 1093 lines – The embodied human.
- `azr_loop.py`: 164 lines – AZR control loop: reasoning proposes, physics disposes.
- `behavior_exec.py`: 641 lines – Choosing and executing behaviours from the generative space.
- `behavior_space.py`: 659 lines – The space of things the person can *do*.
- `body_learning.py`: 223 lines – Learning to use its own body: which parts of a behaviour are safe.
- `build_model.py`: 806 lines – (unparseable: SyntaxError)
- `complexity.py`: 159 lines – How complicated the person is.
- `config.py`: 350 lines – Global configuration for the Embodied Human simulation.
- `drives.py`: 213 lines – Homeostatic drives: what the body *wants*.
- `inner_brain.py`: 355 lines – The brain-side inner world.
- `inner_organs.py`: 546 lines – The organs: compartmental physiology that the original 67‑variable interoceptive
- `inner_world.py`: 483 lines – The inner world: every internal system, run at its own rate and coupled to the
- `instance_log.py`: 107 lines – Per-instance identity and thought/action transcripts.
- `interoception.py`: 425 lines – Interoception: the sense of the internal state of the body.
- `intrinsic.py`: 120 lines – Intrinsic motivation: the reward decomposition.
- `locomotion.py`: 740 lines – Bipedal walking.
- `mind.py`: 995 lines – (unparseable: SyntaxError)
- `motor.py`: 540 lines – Motor system: from intention to torque.
- `ocular.py`: 429 lines – The ocular surface: why the eyes need to blink.
- `plots.py`: 518 lines – Visualisation.
- `predictive.py`: 322 lines – Predictive coding: forward model, inverse model, body schema, precision.
- `receptors.py`: 1256 lines – Peripheral transduction: physics → receptor potentials.
- `record.py`: 294 lines – Episode recording.
- `senses_ext.py`: 363 lines – Populations of sensory cells, not single channels.
- `skeleton.py`: 465 lines – (unparseable: SyntaxError)
- `skills.py`: 2599 lines – Motor skills: everything the body can *do* on purpose.
- `skin.py`: 571 lines – Whole‑body artificial skin.
- `speech.py`: 177 lines – Speech: the mouth, and the words above the head.
- `state.py`: 113 lines – Shared state snapshot.
- `ultra.py`: 592 lines – The ultra tier: a plug‑in framework for subsystems far larger than the base person.
- `ux_example.py`: 57 lines – A deliberately tiny reference subsystem: copy its structure, not its content.
- `viewer.py`: 425 lines – (unparseable: SyntaxError)
- `wbc.py`: 352 lines – Whole‑body inverse‑dynamics controller.
- `world.py`: 183 lines – The world, as the body and the mind can query it.

**tools/** (7 files)
- `autopush.py`: 165 lines – Watch the project and push to GitHub whenever a file is saved.
- `measure_identity.py`: 134 lines – First‑person ownership score for an instance transcript file.
- `readme_check.py`: 88 lines – Check that an edited README section kept everything that is not wording.
- `scale_complexity.py`: 187 lines – Set, inspect and measure how complicated the person is.
- `simlock.py`: 119 lines – Run a heavy command only when the machine can afford it.
- `train_body.py`: 302 lines – Learn, from many bodies at once, which behaviours are safe to perform.
- `ultra_bench.py`: 119 lines – Benchmark the ultra‑tier subsystems one at a time, each in its own process.

**Root scripts**
- `run_mind.py`
- `run_sim.py`
- 32 diagnostic scripts: `diag_app.py`, `diag_azr_balance.py`, `diag_azr_live.py`, `diag_balance.py`, `diag_balance2.py`, `diag_bubbles.py`, `diag_cobra.py`, `diag_fall.py`, `diag_getup.py`, `diag_getup_hands.py`, `diag_getup_shots.py`, `diag_handwalk.py`, `diag_hold.py`, `diag_long.py`, `diag_mind_server.py`, `diag_mind_smoke.py`, `diag_policy.py`, `diag_push2.py`, `diag_robust.py`, `diag_robust2.py`, `diag_skills.py`, `diag_slip.py`, `diag_stand2.py`, `diag_sweep.py`, `diag_table_climb.py`, `diag_walk.py`, `diag_walk_eval.py`, `diag_walk_render.py`, `diag_walk_sweep.py`, `diag_walk_turn.py`, `diag_wbc.py`

**README.md** – present (5 316 bytes)

## How to Run

1. Install the required Python packages (see `requirements.txt` or the project’s setup instructions).
2. Run a simulation with the base script:
