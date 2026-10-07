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
| Unattended planner + coder loop on free OpenRouter models (§18) | `tools/openrouter_helper/` | offline tests pass; live runs in progress; **its output is unreviewed machine-written code** |

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

## 1. Why this is not just a sensor dump

A machine with 48,000 touch channels is not more embodied than one with 48.
What makes the difference is that the signals arrive **late**, **adapt**,
**predict themselves**, and **matter**:

| Mechanism | Where | What it buys |
|---|---|---|
| Conduction delay (18 / 60 / 250 ms by fibre class) | `afferents.py` | the brain acts on a stale body image, so a forward model is *necessary* |
| Receptor adaptation + habituation | `receptors.py`, `afferents.py` | steady contact fades; you stop feeling your shirt |
| Reafference cancellation (corollary discharge) | `afferents.py` | self-touch feels different from being touched |
| Learned body schema (`d proprio / d action`) | `predictive.py` | "this is my body" as a measurable quantity |
| Interoception + homeostatic drives | `interoception.py`, `drives.py` | the agent has *needs* of its own |
| Affect as gain-and-setpoint control | `affect.py` | emotions change what the agent prefers, not just how it looks |
| Segmental postural prioritisation | `motor.py` | the body protects itself before it moves |

Every one of these is measured in the outputs, so the claims are checkable
rather than rhetorical.

---

## 2. The body

Generated programmatically from declarative tables in `skeleton.py` and
`skin.py` (`embodied_human/build_model.py` writes `out/models/human.xml`).

| | |
|---|---|
| DoF | **94 qpos / 88 qvel** (52 actuated joints + 6-DoF floating base + 5 free objects) |
| Mass | **71.68 kg**, every segment carrying a human mass fraction |
| Actuators | 52, each with a human isokinetic **torque limit** (knee 200 N·m, ankle 110 N·m, hip 160 N·m, eyes 0.05 N·m) |
| Sensors | **2,118** MuJoCo sensors producing **2,186** `sensordata` values |
| Skin | **1,872 taxel sites** across **40 body patches** |
| Geometry | 49 collision geoms, 1,885 sites, 38 bodies |

The body faces **−y**, up is **+z**, and every joint axis is derived from that
convention in `skeleton.build_bones()`. This matters: an earlier version had
the knee, elbow, wrist and finger axes in the *frontal* plane instead of the
sagittal one, so the legs had no knee flexion at all and no controller could
make the body stand up.

### Collision filtering (`build_model.py`)

Contact is enabled when `(contype_A & conaffinity_B) or (contype_B &
conaffinity_A)`, with bits for *world*, *objects*, *left hand*, *right hand*,
*trunk/head* and *limbs*:

```
         limb  hand_l  hand_r    body   world  object
  limb  False    True    True   False    True    True
hand_l   True   False    True    True    True    True
hand_r   True    True   False    True    True    True
  body  False    True    True   False    True    True
 world   True    True    True    True   False    True
object   True    True    True    True    True    True
```

* limbs never collide with limbs or trunk → the body cannot tear itself apart
  on self-intersection (a pelvis capsule sitting inside a thigh capsule)
* geoms of the *same hand* never collide → removes the deep finger–finger
  interpenetration that otherwise registers as **370 kPa of phantom pain** on
  the fingers
* the two hands, and either hand against trunk/head, still collide →
  **self-touch works**, which is the whole point of a body schema

---

## 3. The sensory vector space

### 3.1 Touch — 1,872 taxels × 26 receptor channels = **48,672 scalars**

Taxels are laid out on the real geometric surfaces by `skin.py`
(cylindrical wraps around limb capsules, spherical caps on the head, grids on
box faces), each with its own receptor density profile. Densities differ by
roughly an order of magnitude across the body, as they do in humans.

| idx | channel | unit | idx | channel | unit |
|---|---|---|---|---|---|
| 0 | normal force | N | 13 | SA-II (Ruffini) | a.u. |
| 1 | tangential force u | N | 14 | FA-I (Meissner) | a.u. |
| 2 | tangential force v | N | 15 | FA-II (Pacinian) | a.u. |
| 3 | contact pressure | kPa | 16 | C-tactile (affective) | a.u. |
| 4 | shear magnitude | N | 17 | mechanonociceptor | a.u. |
| 5 | indentation depth | mm | 18 | heat nociceptor | a.u. |
| 6 | contact area | cm² | 19 | cold nociceptor | a.u. |
| 7 | normal force rate | N/s | 20 | itch (MrgprA3+) | a.u. |
| 8 | vibration, broadband | a.u. | 21 | warmth receptor | a.u. |
| 9 | FA-I band (5–50 Hz) | a.u. | 22 | cold receptor | a.u. |
| 10 | FA-II band (50–400 Hz) | a.u. | 23 | local skin temperature | °C |
| 11 | SA ripple (<5 Hz) | a.u. | 24 | slip probability | a.u. |
| 12 | SA-I (Merkel) | a.u. | 25 | friction utilisation | a.u. |

**How the field is reconstructed.** MuJoCo's native `touch` sensor is gated by
`site_size` and sums whole contact forces, which blurs the image. Here each
MuJoCo contact is instead splatted onto the taxels of both contacting bodies
with a Gaussian receptive field (σ = 28 mm) and an outward-facing test, which
gives a smooth, gap-free, spatially honest field — and it is what makes a
fingertip feel different from a back. The 1,872 native `touch` sensors are
*also* generated, so both paths exist.

**Pain is thresholded on pressure, not force**, with a site-specific
threshold (`NOCI_PRESSURE_THRESHOLD_KPA`): 250 kPa on a fingertip, 700 kPa on
a sole. Thresholding on total force makes every step agonising, because a sole
taxel legitimately carries 50 N.

**Skin temperature** is a per-taxel state variable: perfused towards core
temperature, cooled by ambient, and conductively exchanged with whatever is
being touched. Standing on a 22 °C floor cools the soles to ~27.5 °C over 20 s
(visible in `fig_homunculus.png`).

### 3.2 Proprioception — 52 joints × 13 channels = **676 scalars**

`q`, `qd`, `qdd`, muscle **spindle Ia** (length *and* rate — silent during a
static hold, firing on movement), spindle **II** (length only), **Golgi tendon
organ** (force), efference copy, measured torque, torque error, joint-limit
proximity, muscle length, muscle velocity, and sense of effort.

### 3.3 Everything else

| modality | channels | notes |
|---|---|---|
| Vestibular | 18 | 3 semicircular canals **with cupula adaptation**, 3 otolith organs, gravity direction, tilt, yaw, magnetometer |
| Vision | 17 | 48×36 retina rendered from the egocentric camera at 60 Hz, foveal vs peripheral, motion energy, contrast, saccades, blinks, pupil (light **and** arousal) |
| Audition | 24 bands + 6 | log-spaced cochlear filterbank excited by contact transients (bone conduction) and self-generated motor noise |
| Olfaction | 24 | 1/r² concentration from scene objects' odour signatures |
| Gustation | 5 | sweet / salty / sour / bitter / umami on tongue contact |
| Interoception | 67 | see §4 |

**Total: 49,588 sensory scalars**, compressed into a **410-dimensional learned
latent** by `agent.LatentSpec`, which builds the *current* and the *preferred*
latent together so they can never drift out of alignment.

---

## 4. Interoception — 67 variables, 11 organ systems

The sense of the internal state of the body, integrated as ODEs so a sprint
leaves you with lactate and an oxygen debt, cold constricts the skin's blood
vessels, and pain is gated by endorphins released under stress.

| system | n | variables |
|---|---|---|
| cardiovascular | 10 | heart rate, HRV (RMSSD), stroke volume, cardiac output, systolic/diastolic BP, cutaneous & muscular perfusion, baroreflex error, venous return |
| respiratory | 7 | rate, tidal volume, minute ventilation, SpO₂, arterial CO₂/O₂, **air hunger** |
| metabolic | 8 | glucose, glycogen, lactate, ATP reserve, metabolic rate, energy expended, insulin, glucagon |
| thermoregulatory | 7 | core temp, skin temp, shivering, sweating, thermal discomfort, heat production/loss |
| gastrointestinal | 6 | gastric fullness, ghrelin, leptin, nausea, gut motility, nutrient absorption |
| renal | 6 | hydration, osmolality, bladder fullness, sodium, potassium, urine output |
| fatigue | 6 | peripheral & central fatigue, adenosine, sleepiness, alertness, recovery need |
| immune | 4 | cytokines, inflammation, sickness behaviour, tissue damage |
| nociceptive | 5 | nociceptive input, pain intensity, pain unpleasantness, **central sensitisation**, descending inhibition |
| effort | 4 | sense of effort, breathlessness, motor command magnitude, muscle tone |
| endocrine | 4 | adrenaline, noradrenaline, cortisol, insulin sensitivity |

Two details worth calling out. **Sense of effort comes from the corollary
discharge, not the muscle** — it rises with the command even when the muscle
cannot deliver. And **pain is gated**: `pain = input × (1 + sensitisation) ×
(1 − 0.75 × descending inhibition)`, where the inhibition is driven by
endorphins and low stress. That is stress-induced analgesia, and it is why a
frightened agent reports less pain from the same injury.

---

## 5. The emotional vector space

### 5.1 Appraisal (10 dimensions) → emotions (28) → neuromodulators (25)

Scherer's Component Process Model condensed to ten evaluative dimensions:
`novelty, intrinsic_pleasantness, goal_relevance, goal_congruence,
agency_self, coping_potential, certainty, norm_compatibility, urgency,
social_evaluation`.

Each emotion is a **weighted pattern over those dimensions** — this is the
theory made executable — with its own time constant, so fear persists after
the threat has gone:

| emotion | defining appraisal weights |
|---|---|
| fear | urgency ↑↑, coping ↓↓, goal_congruence ↓↓, novelty ↑ |
| anger | goal_congruence ↓↓, **agency_self ↓** (other-caused), coping ↑ |
| joy | intrinsic_pleasantness ↑↑, goal_congruence ↑↑ |
| disgust | intrinsic_pleasantness ↓↓, norm_compatibility ↓↓ |
| guilt | **agency_self ↑↑**, norm_compatibility ↓↓ |
| shame | agency_self ↑, norm_compatibility ↓↓, social_evaluation ↓↓ |
| pride | agency_self ↑↑, social_evaluation ↑↑, norm_compatibility ↑ |
| awe | novelty ↑↑, coping ↓, certainty ↓ |

The full 28: `fear, anxiety, anger, sadness, joy, surprise, disgust, contempt,
trust, anticipation, love, guilt, shame, pride, envy, jealousy, hope, awe,
gratitude, embarrassment, curiosity, boredom, calm, relief, loneliness,
empathy, amusement, satisfaction`.

**Neuromodulators** are the gain layer, and they are wired to do something:
dopamine phasic (τ = 0.4 s, reward prediction error) and tonic (τ = 45 s),
serotonin (τ = 900 s, selectively damps *negative* emotions),
norepinephrine (arousal, policy temperature), acetylcholine (learning rate),
oxytocin (social touch), vasopressin, endorphin **→ pain gate**, enkephalin,
endocannabinoid, cortisol, adrenaline, noradrenaline, testosterone
(dominance), estrogen, progesterone, prolactin, melatonin, adenosine,
histamine (itch), GABA, glutamate, substance P, BDNF, orexin.

### 5.2 Core affect, mood, temperament

* **Core affect**: valence, arousal, dominance (Mehrabian PAD) + tension and
  pleasantness, each with its own time constant.
* **Mood**: slow (τ = 900 s) moving averages — the agent has a disposition,
  not just reactions.
* **Temperament**: five damped traits (neuroticism, extraversion, openness,
  agreeableness, conscientiousness) that **bias appraisal itself**. A
  high-neuroticism agent weights urgency more and coping less.
* **Action tendencies**: approach, avoid, freeze, attack, withdraw, explore —
  the bridge from feeling to movement.

---

## 6. Homeostatic drives — what the body wants

16 drives, each with a level, an urgency (amplified by allostatic load and
stress), a setpoint, the current value, and a **slope** — because a need that
is getting worse feels more urgent than one that is merely bad:

`hunger, thirst, sleepiness, thermal_cold, thermal_heat, pain, air_hunger,
nausea, bladder, fatigue, itch, social_need, safety, curiosity, comfort,
restlessness`

Drives become the **preferred observations** in latent space (§8), so they
change behaviour without any reward function being written.

---

## 7. Predictive coding and the body schema

Three online learners in `predictive.py`, all recursive least squares with
forgetting and ridge regularisation:

| model | maps | params |
|---|---|---|
| forward `p(s_{t+1} | s_t, a_t)` | 410 + 52 → 410 | **189,420** |
| inverse `a_t = f(s_t, Δs)` | 820 → 52 | 42,640 |
| body schema `∂proprio/∂action` | 462 → 410 | 189,420 |

The forward model is **initialised as identity** ("nothing changes unless I
act"). Without that prior it predicts a zero state, every candidate policy
looks equally bad, and policy selection is vacuous until the model has learned.

The body schema's row norms define **controllability** — the operational
definition of "this is my body" — and the forward model's per-dimension R²
gives **ownership**: a sensory dimension that efference copy explains is
*mine*. In the reference run, reported body ownership reaches **0.975**, and
proprioceptive drift (the slow accumulation of unexplained proprioceptive
error, as in the vibration illusion) stays at **0.0016**.

**Precision** is the inverse error variance per channel, which is attention:
an unpredictable channel is down-weighted rather than allowed to dominate.
Free energy is `F = inaccuracy + 0.05 × complexity`, and is reported both
absolutely and **per dimension** (the raw sum over 410 dims saturates every
downstream use of it).

---

## 8. Active inference

```
G(π) = risk  +  ambiguity  −  epistemic value  +  motor cost  +  emotional bias
```

* **Risk** is the divergence between the predicted sensory consequence of a
  policy and the *preferred* observation built from drives, interoceptive
  setpoints and the current affect. A frightened agent literally prefers
  different sensations than a curious one.
* The prediction is a **belief**: the proprioceptive block is expected to move
  most of the way to the policy's equilibrium point, blended with the learned
  forward model once it has data. A body has to act *before* it has learned
  anything, so policy selection must be meaningful from the first tick.
* **Emotional action tendencies** bias whole policy *categories* and are kept
  out of the free-energy term on purpose: an action tendency is a low-level
  disposition, not a deliberative prediction, and keeping them separate is
  what lets fear produce a flinch in a fraction of a second.
* Policy precision is a softmax whose temperature rises with arousal.

**55 motor programs** across 11 categories: posture, balance shifting, reaching,
self-touch (hand-to-face, hand-to-chest, rub-forearm), hand/grasp, gaze, head,
face, defensive (flinch, freeze, brace, step-back), locomotion, exploration.

### Intrinsic motivation

Ten reward terms, all internal: **jerk, energy, stability, novelty, tactile,
comfort, affective balance, empowerment, pain, contact smoothness**. The
novelty term rewards *reducible* prediction error, not noise.

---

## 9. The motor system, and what it took to stand up

Commanded through an **equilibrium point** (a virtual posture), not torques —
the same λ-model real motor control uses. On top of that: stretch reflex with
conduction delay, Golgi tendon protection, nociceptive withdrawal, righting,
vestibulo-collic, startle, a CPG, and a protective stepping reflex.

Getting a 72 kg humanoid to stand took five separate fixes, each of which was
invisible until measured:

1. **Joint axes.** Knee/elbow/wrist/finger axes were in the frontal plane, so
   the legs had no knee flexion. No controller can fix that.
2. **The COM velocity was always zero.** MuJoCo never writes
   `subtree_linvel[0]` (the recursion starts at body 1), so the balance loop
   had *no derivative term* — an undamped spring on an inverted pendulum. The
   human's COM is `subtree_com[pelvis]`, not `subtree_com[0]` (which includes
   the scene objects).
3. **The ankle rate limit.** `max_delta = 40 × dt × (limit/60)` gave the knee
   **1.5 s to reach full torque** — longer than the entire balance response.
   Replaced with a *rise time* (50 ms), which is also physically right.
4. **Damping.** With a 45 ms actuator lag, a PD loop needs `kd/kp ≈ 2 × lag =
   0.12 s`. At 0.06 s the legs oscillate and the body folds.
5. **Controller authority.** The posture spring and the balance loop both drive
   the ankle; with equal authority the spring wins, the centre of pressure
   never travels, and the COM drifts away until the body falls. Balance gets
   priority on the ankle (×0.45).

The ankle strategy is formulated on the **centre of pressure** via an inverted
pendulum, so it saturates at the edge of the foot instead of over-driving the
joint into its stop (where the *constraint*, not the muscle, determines the
motion and the body topples about the toe):

```
ω = sqrt(g / h_com)
CoP_desired = x + Kp·x + Kd·x'          Kp = 1,  Kd = 2ζ/ω,  ζ = 0.9
CoP         = clip(CoP_desired, −foot_front, +foot_back)
τ_ankle     = −m·g·(CoP − a)            a = ankle offset from the support centre
```

Two more mechanisms came out of the debugging and are physiologically real:

* **Segmental postural prioritisation.** When balance is threatened (measured
  by the **capture point**, `x + x'/ω`, because that is what decides
  recoverability), voluntary displacement of the trunk and legs is attenuated
  towards the nominal configuration. Arms, hands, head and eyes keep full
  authority — a 4 kg arm moves the whole-body COM by a centimetre or two.
  A single global gain (the obvious implementation) leaves the agent unable to
  gesture whenever it is standing, which is always.
* **Protective stepping** that is **direction-aware**. Once the capture point
  leaves the support polygon no torque can save the body; the only correct
  action is to move the polygon. An earlier version always swung the leg
  *forwards*, which rescues a forward fall and accelerates a backward one.

The CPG is gated by stability: it oscillates the legs without a balance
coupling, so engaging it while standing is a guaranteed fall.

---

## 10. The loop

Seven nested rates, because a nervous system does not run at one clock:

```
1000 Hz   physics          MuJoCo
 250 Hz   receptors        transduction
 200 Hz   afferents        conduction delay, rate coding, reafference
 100 Hz   interoception    organ systems, autonomic
  50 Hz   affect           neuromodulators, appraisal, emotion
  10 Hz   cognition        active inference, policy selection
   1 Hz   mood             allostatic slow state
```

Each step: **feel → interpret → want → predict → choose → move → feel again.**

---

## 11. Outputs

| file | contents |
|---|---|
| `out/episode.npz` | every layer: emotions, neuromodulators, appraisal, drives, 67 interoceptive variables, per-region tactile, full taxel snapshots, reward terms, proprioception, and ~60 scalar headline channels |
| `out/episode_channels.csv` | the scalar channels, one row per logged tick |
| `out/episode_tactile_regions.csv` | per-region tactile aggregates in long format (pandas-ready) |
| `out/episode_manifest.json` | the complete description of the model and its senses |
| `out/summary.json` | episode statistics |
| `out/fig_affect.png` | core affect, all 28 emotions, all 25 neuromodulators, appraisal, action tendencies |
| `out/fig_interoception.png` | the internal milieu across all 11 organ systems |
| `out/fig_senses.png` | every exteroceptive channel, vestibular organs, balance strategies, body height |
| `out/fig_homunculus.png` | **the tactile body map** at 6 instants: force, pressure, nociception, temperature, SA-I, C-tactile |
| `out/fig_learning.png` | free energy, complexity, surprise, body schema, reward terms, RPE |
| `out/fig_drives.png` | all 16 drives and the policy timeline/histogram |
| `out/frames.png` | rendered views of the body |
| `out/models/human.xml` | the generated MJCF, for inspection |

### Reference run (20 s, seed 7, vision on)

| quantity | value |
|---|---|
| fell | **no** |
| pelvis height | 0.836 m, constant (min 0.8363) |
| mean / max balance error | 0.0152 m / 0.0155 m |
| max pain | 0.013 |
| body ownership | 0.977 |
| proprioceptive drift | 0.0004 |
| mean valence | +0.056 |
| mean arousal | 0.334 |
| mean free energy (per dim) | 0.560 |
| self-touch fraction | 0.995 |
| policies used | 12 of 55 |
| speed | ~0.37× realtime, ~3 ms per physics step |

The forward model's free energy falls from ~35 to ~0.06 within the first
second — that is the body schema forming. The agent starts from the prior
"nothing changes unless I act" and, within about ten cognitive ticks, has
learned the sensory consequences of its own commands.

---

## 12. Code map

```
embodied_human/
  config.py         all rates, time constants, thresholds, weights
  skeleton.py       bones, joint axes + limits, actuator torque limits, masses
  skin.py           1,872 taxels, receptor density profiles, body-map layout
  build_model.py    MJCF generation, collision filtering, scene, index metadata
  state.py          the per-step body snapshot shared by every subsystem
  receptors.py      tactile / proprioceptive / vestibular / visual / audio / chemo
  afferents.py      conduction delay, rate coding, adaptation, reafference
  interoception.py  67-variable internal milieu
  drives.py         homeostatic drives -> preferred observations
  affect.py         appraisal, 28 emotions, 25 neuromodulators, mood, temperament
  predictive.py     forward + inverse models, body schema, precision, Kalman
  active_inference.py  55 policies, expected free energy, selection
  motor.py          equilibrium-point control, balance strategies, reflexes, CPG
  wbc.py            whole-body inverse dynamics (QP-style least squares + active set)
  locomotion.py     capture-point footstep planner / gait state machine on top of wbc
  skills.py         arm IK, hands, grasp, gestures, gaze, speech hook, action queue
  speech.py         syllable-timed jaw, text bubbles, optional Windows SAPI voice
  world.py          egocentric symbolic view of the scene for the mind
  mind.py           prompt, parser/sandbox and backends (AZR) for the "mind"
  viewer.py         renderer with speech/thought bubbles, Tk app
  intrinsic.py      10-term intrinsic reward
  agent.py          the multi-rate loop and the latent/preferred space
  record.py         episode recording
  plots.py          figures
  complexity.py     the complexity presets (base / rich / extreme / max), read once at import
  senses_ext.py     populations of sensory cells (spindles, hair cells, olfactory, taste, retina)
  ocular.py         tear film, dryness, blink CPG, lids, corneal nerves; EyeRig (visible eyes)
  inner_organs.py   vascular beds, lungs, kidney, liver, gut + microbiome, motor units, skin thermo, immune, chemistry
  inner_brain.py    circadian clock, neural mass, episodic memory, conditioning, interoceptive prediction
  inner_world.py    runs the above at their own rates and couples them to affect, drives, skin, behaviour
  behavior_space.py the generative behaviour space (33 channels, mixed-radix encode/decode, affinity tags)
  behavior_exec.py  selector (expected free energy over sampled descriptors) and 50 Hz executor
  body_learning.py  BodySafety: learned model of which behaviours unbalance the body
  body_safety.json  the trained weights (loaded by every new person)
run_sim.py          CLI
run_mind.py         the person with a language-model mind (§16.5)
tools/
  scale_complexity.py   list / measure / set the complexity level
  train_body.py         parallel babbling + fitting of BodySafety
  autopush.py           commit (and push) whenever a file is saved
  openrouter_helper/    the planner + coder loop (§18)
```

Diagnostic scripts (`diag_*.py`) are the experiments that produced §9; each
one is runnable and reports the measurement it was written for.

---

## 13. Extending it

* **New skin region** — add a `SkinPatch` to `skin.default_patches()`.
* **New emotion** — add a profile to `affect.EMOTION_PROFILES` (weights over
  the appraisal dimensions) plus entries in `EMOTION_VALENCE`,
  `EMOTION_AROUSAL` and `EMOTION_TAU`.
* **New neuromodulator** — add to `NEUROMODULATORS`, `NM_BASELINE`, `NM_TAU`,
  and a target expression in `AffectSystem.update`.
* **New organ system** — add variables to `interoception.INTERO_NAMES` and
  ODEs in `update()`.
* **New drive** — add to `drives.DRIVES`, a setpoint, and an expression.
* **New motor program** — add a `Policy` in `active_inference.default_policies()`.
* **Different personality** — pass a 5-vector `temperament` to `AffectSystem`.

---

## 14. Honest limitations

* **Vision is a 48×36 retina with hand-designed features**, not a learned
  encoder. It preserves foveal/peripheral structure, saccades, blinks and the
  pupil reflex, but it is not a model of early visual cortex.
* **Audition is synthesised** from contact transients and self-motion, since
  MuJoCo has no acoustic field. There is no real sound propagation.
* **The forward model is linear (RLS).** It captures the body's local
  input–output structure well, but it cannot represent contact discontinuities
  or multi-step dynamics. A sequence model would be the natural upgrade.
* **Contact is rigid-body.** Skin compliance is modelled in the *receptor*
  layer (indentation, contact area, slip) rather than by soft bodies, so
  contact forces are stiffer than flesh.
* **The policy library is hand-authored**, not discovered. Active inference
  chooses among 55 programs; it does not synthesise new ones.
* **Free-energy magnitudes are not calibrated** to any physical unit; only
  relative differences drive behaviour.
* **Self-touch is spatially coarse.** Contacts are attributed to taxels by a
  Gaussian receptive field, so a hand resting on the chest produces a smooth
  blob rather than a resolved fingerprint-like pattern.
* **It walks, but not reliably** (see §16.2 for the measured numbers): the new
  whole-body-control gait takes a handful of steps and falls on a large
  fraction of longer or stop-and-go walks.  The old CPG is still gated off.
* Everything in §16 (skills, hands, the mind) is *newer and less measured*
  than the sensory/affective machinery above; its limitations are listed in
  §16.6.
* Everything in §17 (complexity, eyes, inner world, behaviour space) was
  added later still; limitations are in §17.7.  §18 is code that edits this
  repository by itself and has not been reviewed by a person.

---

## 15. References for the physiological content

Receptor classes and densities: Johansson & Vallbo (1983); Vallbo & Johansson
(1984). Affective touch: Löken et al. (2009); Olausson et al. (2010).
Nociceptor transduction and sensitisation: Julius & Basbaum (2001); Woolf
(2011). Pressure pain thresholds by site: Rolke et al. (2006). Interoception
and allostatic load: Craig (2009); Sterling (2012); Barrett & Simmons (2015).
Appraisal theory: Scherer (2001); Lazarus (1991). Core affect: Russell (1980);
Mehrabian & Russell (1974). Predictive coding and active inference: Rao &
Ballard (1999); Friston (2010); Adams, Shipp & Friston (2013). Corollary
discharge and reafference: von Holst & Mittelstaedt (1950); Blakemore, Wolpert
& Frith (2002). Body schema and ownership: Head & Holmes (1911); Botvinick &
Cohen (1998). Postural strategies: Horak & Nashner (1986); Winter (1995).
Capture point: Hof, Gazendam & Sinke (2005). Equilibrium-point control:
Feldman (1966); Bizzi et al. (1984).  Whole-body control / divergent component
of motion walking: Kajita et al. (2003); Englsberger et al. (2015); Sentis &
Khatib (2005).  Absolute Zero Reasoner: Zhao et al. (2025), "Absolute Zero:
Reinforced Self-play Reasoning with Zero Data".

---

## 16. Walking, hands, speech, and a mind

This part was added on top of the body above.  It is deliberately described
in terms of what was *measured*, because several pieces are demos, not solved
problems.

### 16.1 Whole-body control (`wbc.py`)

Standing now uses inverse dynamics instead of the equilibrium-point spring.
Every 4 ms the controller solves one least-squares problem for joint
accelerations and contact wrenches: track a desired COM acceleration, pelvis
and chest orientation, the swing-foot trajectory and a posture reference,
subject to the floating-base dynamics, unilateral contacts, a centre-of-
pressure box inside each foot, a friction cone and a torsional-friction
limit.  Violated inequalities are handled with an active set (pins).  Joint
torques are clipped to the human limits in `skeleton.py`.  The eyes, jaw and
fingers stay on their own controllers.

### 16.2 Bipedal walking (`locomotion.py`)

A divergent-component-of-motion (capture-point) planner places each foot,
`beta_step` scales how aggressively; states are
`stand → init → unload → single support ↔ double support → settle → stand`.
Step length is integrated to meet the commanded speed, swing feet follow
minimum-jerk paths, and the heading is pulled towards the commanded one.
About 25 gait/WBC parameters were tuned by random search (parallel, eight
rounds).

**Measured result: partly working.**
* Best parameter set, 13 s of straight walking at 0.3 m/s, four noise seeds:
  **survived 2 of 4**; the others fell between 8 and 9 s (≈1.5–2 m).  Across
  the whole sweep only 18 of 224 candidates survived even one seed, and the
  18 survivors re-tested under new seeds went mostly 0/4: the sweep
  over-fits its seeds.
* Walk 5 s then stop, eight seeds: **2 of 8** stayed up; three bursts of
  3.5 s with stops: **1 of 8**.  Stopping and turning are the weakest parts.
* I tried a bounded virtual pelvis stabiliser; it did not help, so it is not
  in the code.
* Standing (and standing while moving the arms) is solid; walking should be
  treated as "a few steps", and the skill layer will fall over if asked to
  walk further.  `walk_to` exists but inherits this unreliability.
* The tuning ran with the table moved out of the way; a long walk past
  furniture was not tested.

### 16.3 Arms, hands, grasping, holding (`skills.py`)

* **Arm IK** is analytic (law of cosines for shoulder–elbow–wrist, a swivel
  angle search inside joint limits, wrist from the residual orientation), with
  a small integral term for gravity sag, and a trunk lean that is allowed only
  while the COM stays over the feet.
* **Hands**: rigid two-phalanx fingers, three digit groups (thumb, index,
  other fingers) with named poses (`open, relaxed, fist, point, pinch,
  thumbs_up, ok, grip`).  There is no opposable-thumb squeeze.
* **Grasp pipeline**: lift the hand clear of the table edge, come in beside
  the object, descend, slide the palm in, close the digits, lift.
* **The grip is assisted, and this is a shortcut:** when the palm touches the
  object a soft weld constraint is switched on between hand and object.
  Without it the rigid fingers squeeze the object out (an apple shoots off at
  ~0.4 m/s).  Release switches the weld off.  Pick-up therefore shows the
  *behaviour* of grasping, not a physically earned force-closure grasp.
* **Gestures**: `wave, nod, shake_head, shrug, clap, thumbs_up, think,
  scratch_head, drink, bow`.  **Head and eyes**: neck yaw/pitch plus eyes
  follow `look_at`, with wandering when idle.
* Measured: grab → hold → put down works for the apple and mug in the
  scripted test at the table, with the person standing at the table.  The
  stone (heavy, wide) failed to be gripped, and one mug put-down run ended
  with the mug flung to the floor, so treat it as fragile.

### 16.4 Speech and thoughts (`speech.py`, `viewer.py`)

Speaking opens and closes the jaw with a syllable-timed envelope (no
fine mouth shape).  What it says appears in a white bubble above the head;
its thoughts appear above its head in grey italics.  Optionally also spoken
aloud through Windows SAPI (`--voice`).  There is **no built-in drive to
speak**; the person talks only if the mind decides to.

The jaw is moved kinematically; its actuator is muted while it speaks (an
earlier version let the actuator fight the override, and the reaction torque
knocked the whole body over).

### 16.5 The mind (`mind.py`, `world.py`, `run_mind.py`, `setup_azr.py`)

The body exposes an API of actions (`say, look_at, look_forward, hand_pose,
wait, walk_to, walk, turn, face, stop, reach, grab, release, put_down,
point_at, gesture, crouch, stand`).  `world.py` turns the simulator state into
what a person could know: objects in the field of view, described
egocentrically ("0.4 m ahead, 0.2 m to the right"), plus what the hands hold
and what was heard.  `mind.py` builds a prompt from that, asks a language
model, and parses its answer.

* **AZR format.**  Absolute Zero Reasoner is a Qwen2.5-Coder model trained
  with an R1-style `<think>…</think><answer>…</answer>` format.  The prompt
  ends with `Assistant: <think>` so the model continues in that format; the
  text in `<think>` becomes the grey italic thought bubble, and the
  `<answer>` is parsed as lines of calls such as `look_at("apple")`.  The
  answer is **never executed**: it is parsed with `ast` against a whitelist
  of the API names with literal arguments only.
* **Backends**: any OpenAI-compatible `/v1/completions` server (llama.cpp's
  `llama-server`, vLLM, Ollama) with streaming, `llama-cpp-python`, or
  `transformers`.  `setup_azr.py` explains how to get the weights; it
  downloads nothing unless you pass `--download`.
* **Incentive.**  The prompt tells the mind that it does not have to say or do
  anything.  There is no reward for speaking.

**What has and has not been run.  AZR itself has never been run here.**  No
weights are installed, so the live path is verified only against a small fake
OpenAI-compatible server returning a canned `<think>/<answer>` stream
(`diag_mind_server.py`: prompt construction, HTTP streaming, parsing,
dispatch into the body, the thought bubble, and a finished `grab('apple')`
all work).  Without any model, `run_mind.py` falls back to a **scripted
stand-in** (`StubBackend`) that reacts to a few keywords and wanders; it is
*not* a language model, and the HUD says so.  How a real AZR model behaves
as this person's mind (and whether a 3B code-reasoning model is any good at
it) is unknown.  AZR was trained for code/maths reasoning, not for
embodied dialogue.

```bash
python run_mind.py                                   # window + scripted stand-in
python run_mind.py --at-table --backend server --url http://127.0.0.1:8080/v1
python run_mind.py --headless 45 --at-table --say "4:hello" --say "14:please pick up the apple"
python setup_azr.py                                  # what to do to get AZR
```

### 16.6 Limitations of this part

* Walking is unreliable (§16.2); stopping and turning more so.
* Grasping uses an assist weld; fingers are rigid and there is no opposable
  thumb; heavy or wide objects fail.
* The simulation runs at roughly 0.1–0.3× real time on the test machine
  (whole-body control every 4 ms plus rendering), so the window is slow.
* Several fixes were tuned against one scene layout (table at 0.2 m from the
  edge, objects where `build_model.py` puts them).
* The mind sees a symbolic description, not pixels; the 48×36 retina is
  unused by it.
* Diagnostic scripts for this part: `diag_wbc.py`, `diag_walk*.py`,
  `diag_skills.py`, `diag_bubbles.py`, `diag_mind_smoke.py`,
  `diag_mind_server.py`, `diag_app.py`.
