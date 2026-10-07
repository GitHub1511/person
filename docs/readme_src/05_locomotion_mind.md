## 16. Locomotion, manipulation, speech and a mind

This part was added on top of the body of §2–§11. It is presented in terms of what was *measured*, because
several components are demonstrations rather than solved problems.

### 16.1 Whole-body control (`wbc.py`, 352 lines)

Standing is controlled by inverse dynamics instead of the equilibrium-point spring. Every 4 ms (250 Hz) the
whole-body controller (WBC) solves one least-squares problem for the 88 generalised accelerations and the
contact wrenches. It tracks a desired centre-of-mass (COM) acceleration, the pelvis and chest orientation, the
swing-foot trajectory and a posture reference, subject to the floating-base dynamics (an 88 × 88 mass matrix),
unilateral contacts, a centre-of-pressure box inside each foot, a friction cone and a torsional-friction
limit. Violated inequality constraints are handled with an active set (pins). Joint torques are clipped to the
limits of §2.1. The eyes, jaw and fingers remain under their own controllers.

### 16.2 Bipedal walking (`locomotion.py`, 741 lines)

A divergent-component-of-motion (capture-point) planner places each foot, with `beta_step` (1.06) scaling the
aggressiveness of the placement; the gait states are `stand → init → unload → single support ↔ double support →
settle → stand`. Step length is integrated to meet the commanded speed, swing feet follow minimum-jerk paths
and the heading is drawn towards the commanded heading. The gait dataclasses hold 46 float parameters, of which
about 25 were tuned by random search (parallel, eight rounds, 224 candidates).

**Measured result: partly functional.**

* For the best parameter set, over 13 s of straight walking at 0.3 m/s with four noise seeds, the walker
  **survived 2 of 4** runs; the rest fell between 8 and 9 s (about 1.5–2 m). Across the sweep only 18 of 224
  candidates survived even one seed, and the 18, re-tested under new seeds, mostly scored 0/4; the sweep
  therefore over-fits its seeds.
* Walking 5 s and then stopping, eight seeds: **2 of 8** stayed upright. Three bursts of 3.5 s with stops:
  **1 of 8**. Stopping and turning are the weakest parts.
* A bounded virtual pelvis stabiliser was tried; it did not help and is not in the code.
* Standing (including standing while the arms move) is dependable; walking should be regarded as limited to a
  few steps, and the skill layer will fall if asked to walk further. `walk_to` inherits this unreliability.
* The tuning was performed with the table moved out of the way; a long walk past furniture was not tested.

### 16.3 Arms, hands, grasping and holding (`skills.py`, 2,635 lines, 131 functions)

`SkillSystem` has 89 methods, 36 of them public (the `api_*` calls of §16.5 and a few state accessors).

* **Arm inverse kinematics (IK)** is analytic: the law of cosines for the shoulder–elbow–wrist chain (upper arm
  0.255 m, forearm 0.23 m, reach `ARM_REACH` = 0.54 m), a search over the swivel angle within the joint limits,
  the wrist from the residual orientation, a small integral term for gravity sag, and a trunk lean permitted only
  while the COM remains over the feet.
* **Hands**: rigid two-phalanx fingers in three digit groups (thumb, index, other fingers) with **8** named poses
  (`open, relaxed, fist, point, pinch, thumbs_up, ok, grip`). There is no opposable-thumb squeeze. The
  generative behaviour space (§17.5) adds 14 hand shapes × 3³ digit variations = 378 configurations per hand.
* **Grasp pipeline**: lift the hand clear of the table edge, approach beside the object, descend, slide the
  palm in, close the digits, lift.
* **The grip is assisted, and this is a shortcut**: when the palm touches the object a soft weld constraint
  between hand and object is switched on. Without it the rigid fingers squeeze the object out (an apple leaves
  at about 0.4 m/s). Release switches the weld off. Pick-up therefore demonstrates the *behaviour* of grasping,
  not a force-closure grasp.
* **Gestures** (10): `wave, nod, shake_head, shrug, clap, thumbs_up, think, scratch_head, drink, bow`. **Head and
  eyes**: neck yaw and pitch and the eyes follow `look_at` and wander when idle. Self-touch includes rubbing the
  eyes and scratching (`api_touch_self`, `api_rub_eyes`, `api_scratch`).
* Measured: grab → hold → put down succeeded for the apple and the mug in the scripted test with the person
  standing at the table. The stone (0.9 kg, wide) could not be gripped, and one mug put-down ended with the mug
  on the floor; the procedure is fragile.

### 16.4 Speech and thoughts (`speech.py`, 177 lines; `viewer.py`)

Speech opens and closes the jaw with a syllable-timed envelope (50 Hz, no fine mouth shape). The spoken
content appears in a white bubble above the head and the thoughts appear above the head in grey italics. It can
optionally be rendered aloud through the Windows Speech API (`--voice`). Heard speech is delayed by the
distance at the speed of sound (343 m/s). There is **no built-in drive to speak**; the person speaks only if the
mind decides to. The jaw is moved kinematically and its actuator is muted while speaking (an earlier version let
the actuator oppose the override and the reaction torque knocked the body over).

### 16.5 The mind (`mind.py`, 996 lines; `world.py`; `run_mind.py`)

The body exposes an application programming interface (API) of **26 call names** (24 distinct skills; `pick_up`
and `grab` are aliases, `nothing` is a no-op): `say, look_at, look_forward, hand_pose, wait, walk_to, walk, turn,
face, grab, pick_up, release, put_down, reach, point_at, gesture, crouch, stand, stand_up, stop, nothing, blink,
touch_self, express, rub_eyes, scratch`. A sequence of *k* calls drawn from these names has 26^*k* forms: 676
for *k* = 2, 17,576 for *k* = 3, 456,976 for *k* = 4 and 308,915,776 for *k* = 6, before arguments (six scene
objects, 16 self-touch regions, angles).

`world.py` converts the simulator state into what a person could know: the objects in the field of view,
described egocentrically ("0.4 m ahead, 0.2 m to the right"), what the hands hold, and what was heard.
`mind.py` builds a prompt from that, queries a language model and parses the answer:

* **AZR format.** The Absolute Zero Reasoner (AZR; Zhao et al., 2025) is a Qwen2.5-Coder model trained to reason
  by proposing and solving its own code tasks, answering as `<think>…</think><answer>…</answer>`. The prompt
  ends with `Assistant: <think>`; the `<think>` text becomes the grey italic thought bubble and the `<answer>` is
  parsed with `ast` into whitelisted calls with literal arguments. The answer is **never executed**.
* **Backends** (4): any OpenAI-compatible `/v1/completions` server (llama.cpp, vLLM, LM Studio, Ollama) with
  streaming; `llama-cpp-python` in-process; `transformers` in-process; and `StubBackend`, a scripted stand-in
  that is *not* a language model and is labelled as such. AZR sizes known to `mind.AZR_MODELS`: 3B, 7B, 14B.
* **Incentive.** The prompt states that nothing is expected and silence is normal. No reward is tied to speech.

### 16.6 AZR as the live mind

**Model in use.** `absolute_zero_reasoner-coder-3b` (publisher Mungert, GGUF, Qwen2 architecture, 3.1 B
parameters, quantisation Q4_K_S at 4 bits per weight, 1,850,110,336 bytes = 1.85 GB) is loaded in LM Studio at
`http://localhost:1234` with a context of 20,480 tokens (maximum 32,768), 4 parallel slots, flash attention and
the KV cache on the GPU. No download was needed. A direct trial with the project's prompt format (126 prompt
tokens) returned 95 completion tokens, a short reasoning trace and `walk_to("apple")`.

```bash
python run_mind.py --at-table --backend server --url http://127.0.0.1:1234/v1 --model-name absolute_zero_reasoner-coder-3b
python diag_azr_live.py        # one real thought, dispatched into the skill system
python diag_azr_balance.py     # cycles of prompt -> gate -> step -> verify; exit 0 iff no falls
```

**The control loop (`azr_loop.py`, 164 lines): reasoning proposes, physics disposes.** AZR is a code-reasoning
model, not a balance controller, and never drives torques. Every proposed call passes three stages:

1. *Assess* (`assess`): the live body state decides whether the call is allowed, rewritten or denied, and a
   denial returns a reason the model sees in its next prompt. The rules: if the body is on the floor, movement
   calls (`walk_to, walk, turn, grab, reach, put_down, crouch, face`) are denied in favour of `stand_up()`; only one
   locomotion call per turn; `walk_to, walk, grab, reach` are denied while the balance error exceeds 0.06 m;
   `walk_to, walk, turn` are denied while the body or gait is still moving; and a `walk_to` to a target already
   within `ARM_REACH + 0.18` = 0.72 m is rewritten to `face`.
2. *Execute*: the surviving call runs on the existing controllers (whole-body stance hold, capture-point gait,
   analytic arm servo).
3. *Verify* (`verify_outcome`): body snapshots before and after give a one-line outcome (fell, held, balance,
   still moving) that is appended to the mind's episodic memory and to the next prompt.

This is deliberately **not fine-tuning** of the 3B weights (no GPU budget: the 6 GB card is occupied by the model
server). It is the scaffolding that fine-tuning would need, namely constrained decoding by the whitelist, a
precondition gate and verified outcomes, and it can only make the person *better at navigation* through what
is kept in context, not through changed weights. No navigation improvement has been measured yet; a held-out
evaluation (stub versus AZR zero-shot versus AZR with accumulated experience) is the intended protocol and has
not been run.

**Instances, transcripts and the interviewer.** Every person gets an identifier `P-YYYYMMDD-<8 hex>`
(`instance_log.py`), and every completed thought, dispatched action, heard and spoken utterance is appended to
`<out>/instances/<id>.txt`, one line per event with a UTC timestamp and the simulated time. `tools/measure_identity.py`
scores such a transcript for slips out of the first person (third-person self nouns, third-person narration, and
echoes of the last heard line); silence is not a failure. `tools/interview/interview.py` runs one simulation
continuously with the local AZR as its mind and poses **at most one question per 30 minutes (at most 48 per day)**,
each generated from the person's own transcript tail by a model reached through the rate-limited gateway of §18;
a question must be 10–300 characters with exactly one `?`, academic, non-leading and non-roleplaying. None of
this trains or fine-tunes weights.

### 16.7 Fall recovery (`stand_up`, in development)

`SkillSystem._a_stand_up` is a closed-loop recovery: roll to prone → push → tuck → kneel → half-kneel → stand,
each stage with a done-condition on the live body (chest orientation, COM height) and a timeout, so that a blocked
stage fails visibly. Success means COM > 0.72 m and not fallen, after which the stance hold takes over. Variants
exist for a lunge, a cobra push-up, a hand-walk and a table climb; the arm and trunk torque limits were raised
above human norms to make the push-up feasible (§2.1). The corresponding diagnostics are `diag_getup.py`,
`diag_lunge.py`, `diag_cobra.py`, `diag_handwalk.py`, `diag_table_climb.py`, `diag_slip.py` and
`diag_walk_render.py`. {{GETUP_STATUS}}

### 16.8 Limitations of this part

* Walking is unreliable (§16.2), stopping and turning more so.
* Grasping relies on an assist weld; fingers are rigid, there is no opposable thumb, heavy or wide objects fail.
* The simulation runs at about 0.1–0.35× real time at the levels measured (whole-body control every 4 ms plus
  rendering), so the window is slow; LM Studio latency adds to every decision.
* Several fixes were tuned against one scene layout (table 0.2 m from the edge, objects where `build_model.py`
  puts them).
* The mind sees a symbolic description, not pixels.
* AZR was trained for code and mathematical reasoning, not for embodied dialogue; its competence as this
  person's mind is **not established**, and the gate of §16.6 exists because it is expected to propose unsafe
  moves.

---
