## 17. Complexity levels, eyes, the inner world and a generative behaviour space

This part was added after §1–§16 to enlarge the interior of the person and the range of what it can do. It is
toy modelling with plausible orders of magnitude; none of it claims to match a mammal (a mouse has orders of
magnitude more receptors than any level here).

### 17.1 One dial: `complexity.py`

Every size that scales is read from one `Complexity` profile, **once at import time**, because the skin
geometry and therefore the MuJoCo model are generated from it. Choose it with
`PERSON_COMPLEXITY=base|rich|extreme|max|ultra|mega` (or a JSON file of overrides) or persistently with
`tools/scale_complexity.py --set LEVEL`. The default is `extreme`. `ultra` and `mega` have the classic body of
`max` and differ only in the plug-in subsystems of §19.

{{COMPLEXITY_TABLE}}

**Measured** with `python tools/scale_complexity.py --measure base rich extreme max` (vision off, 2026-10-07;
`out/measure_baseline.txt`, `out/readme_data/measure_max.txt`):

| quantity | `base` | `rich` | `extreme` | `max` |
|---|---|---|---|---|
| skin taxels | 1,992 | 7,968 | 17,928 | 31,872 |
| receptor channels per taxel | 26 | 43 | 43 | 43 |
| sensory scalars per frame | 52,981 | 346,406 | 778,054 | 1,384,030 |
| × the `base` body | 1.0 | 6.5 | 14.7 | 26.1 |
| internal dynamic variables | – | 15,066 | 36,932 | 88,892 |
| × the original 136 | – | 111 | 272 | 654 |
| neural-mass units | – | 4,096 | 12,288 | 32,768 |
| synaptic weights in the neural mass (units × in-degree) | – | 131,072 | 589,824 | 2,097,152 |
| motor units | – | 1,664 | 3,328 | 6,656 |
| chemistry analytes | – | 240 | 480 | 960 |
| episodic-store values | – | 137,216 | 405,504 | 1,073,152 |
| behaviour channels / configurations per arm | 33 / 734,375 | same | same | same |
| latent dimension | 572 | 572 | 572 | 572 |
| milliseconds per physics step | 1.87 | 2.37 | 3.95 | 6.34 |
| speed (× real time) | 0.54 | 0.42 | 0.25 | 0.16 |

(The "original 136" are the 67 interoceptive variables, 28 emotions, 25 neuromodulators and 16 original
drives.) From `base` to `max` the sensory vector grows 26-fold, the internal state 654-fold relative to the
original, and the physics step costs 3.4 times as much; the cost per sensory scalar falls from 35 ns to 4.6 ns
of step time, because the dense skin is transduced at 100 Hz instead of 250 Hz (§10). A simulated person needs
about 1–1.5 GB of RAM at `base` (an estimate from the helper's resource model, not a measurement at every
level).

### 17.2 Denser skin and more receptors

* `skin.default_patches()` multiplies each patch's grid by the skin density, so the taxel count grows with its
  square: ×4 at `rich`, ×9 at `extreme`, ×16 at `max` (Appendix A.2 lists all 46 patches).
* `receptors.py` adds 17 extended tactile channels per taxel (26 → 43), with sensitisation and fatigue
  modifiers fed from the inner world, and the `ReceptorFrame.ext` dictionary for the extra banks.
* `senses_ext.py` adds the cell populations of §3.3: 439 / 3,032 / 6,400 / 12,784 non-visual values per frame at
  `base` / `rich` / `extreme` / `max`, plus the retina bank (12,096 to 86,016 values).

### 17.3 Eyes that dry and blink (`ocular.py`, 429 lines)

Each eye has a tear film that thins and evaporates (96 film sectors per eye at `extreme`, 16 at `base`), a
meniscus reservoir fed by the lacrimal gland, lipid and mucin layers, and a population of corneal nerve
terminals (192 per eye at `extreme`, 384 in all, 16 per eye at `base`) of cold, polymodal-nociceptor and
lid-friction classes. The ocular system carries 1,174 state variables at `extreme`. The *sensation of dryness*
drives a blink central pattern generator and reaches the rest of the person:

* **vision**: an irregular film scatters light (blur) and a blink blanks the retina;
* **affect and pain**: ocular discomfort enters appraisal, and chronic irritation sensitises central pain;
* **drives and behaviour**: an `ocular_comfort` drive, and the urge to rub the eyes (self-touch);
* **attention**: staring (reading, concentrating) suppresses blinking, which dries the eyes further; speaking,
  arousal and tiredness change the rate; sadness produces tears, dehydration dries the eyes, lids droop with
  sleepiness and narrow in bright light.

Schematically the blink drive is `base·(1 − 0.80·attention) + 7·dryness² + 2.5·grit`. The eyeballs, irises,
pupils and four lids are the visual-only geoms of §2.3. Orders of magnitude (film thickness about 3 µm,
break-up time 10–20 s, tear osmolarity 300 rising above 320 mOsm/L when dry, about 4 blinks a minute while
reading and 15 while talking) are plausible; it is not a clinical model.

### 17.4 The inner world (`inner_organs.py`, `inner_brain.py`, `inner_world.py`)

Added on top of the original 136 numbers (all still present and still driving everything). Counts at
`extreme` (`InnerWorld.counts()`):

| system | units | notes |
|---|---|---|
| vascular beds | 96 | per-patch blood flow, sympathetic tone, demand |
| lungs | 192 alveolar units | ventilation / perfusion |
| kidney | 192 nephron groups | filtration, reabsorption, ADH |
| liver | 96 zones | glycogen, gluconeogenesis, lipid |
| gut and microbiome | 16 segments, 64 taxa | nutrient pools, enteric neurons, metabolite exchange |
| muscle bank | 104 muscles × 32 = 3,328 motor units | size principle, fibre type, fatigue, glycogen, soreness |
| skin thermoregulation | 46 patches | sweat, blood flow, temperature |
| immune network | 32 populations, 64 cytokines | signed interactions; damage and healing per patch |
| chemistry panel | 480 analytes | factor model (9 drivers), about 70 mechanistic channels |
| circadian clock | 192 oscillators | suprachiasmatic network, peripheral clocks, sleep pressure; entrained by light and meals |
| neural mass | 12,288 units in 20 populations | in-degree 48 (589,824 synapses), 160 inputs (1,966,080 input weights), adaptation, noise, neuromodulator gain |
| episodic memory | 4,096 episodes × 96 features | recall by similarity: familiarity and remembered mood (405,504 values) |
| conditioning, interoceptive prediction | about 130 and 300 states | Rescorla–Wagner associations; one-step predictor with learned precision |
| eyes | 1,174 states | §17.3 |
| **dynamic state variables** | **36,796** (+136 original = **36,932**) | excluding the episodic store |

The inner world runs on eight clocks: neural mass 50 Hz, motor units 20 Hz, circulation and skin 10 Hz,
association 10 Hz, memory recall 4 Hz, chemistry and organs 2 Hz, the circadian clock 1 Hz and memory storage
1 Hz. Its outputs close the loops into the body and the mind: per-patch blood flow, sweat, inflammation and
itch change the receptors; a threat tone (neural-mass "amygdala"), rumination, interoceptive surprise and the
mood carried by memory enter affect; five extra drives (`ocular_comfort, muscle_soreness, gut_discomfort,
mental_fatigue, shift_urge`) enter motivation; the "mind-wandering" tone drives exploration and boredom; and two
blocks of the latent (ocular 16, inner 128 numbers) are seen by active inference. The couplings have the right
*qualitative* behaviour (exertion recruits and fatigues motor units, heat opens skin beds and starts sweating,
damage drives inflammation that sensitises skin) and are otherwise toy physiology.

### 17.5 A generative behaviour space (`behavior_space.py`, `behavior_exec.py`)

The 55 hand-written policies of §8 are not the whole repertoire. A *behaviour* is a descriptor: one choice in
each of **33 factored channels**, and every descriptor compiles to something the body can execute. The factors,
their sizes and their derivations:

{{BEHAVIOR_CHANNELS}}

Derivation of the larger factors: head = 11 yaw × 9 pitch × 7 tilt = 693; eyes = 9 × 7 = 63; lids = 5 apertures
× 7 blink patterns = 35; mouth = 6 jaw levels × 9 vocalisations = 54; torso = 7 bend × 5 side × 7 twist = 245;
arm = 47 schemas × 5⁶ = 734,375 (each of six joints varied in five steps); hand = 14 shapes × 3³ = 378; stance
= 5 heights × 5 lateral × 3 forward = 75; style = 5 speeds × 4 amplitudes × 4 tremors × 4 rhythms = 320; hold
= 6 durations; intent = 1 + 4 modes × 6 objects = 25; self-touch = 1 + 16 regions × 2 hands × 4 actions = 129;
walk = 1 + 3 speeds × 5 turns = 16.

**Combinatorics (computed by `tools/readme_combinatorics.py`).**

| quantity | value |
|---|---|
| descriptors | **11,575,326,076,025,945,959,687,500,000,000,000,000** = 1.1575 × 10³⁷ (log₁₀ = 37.064; 123.12 bits) |
| groups: face | 82,515,510 |
| groups: torso × stance | 18,375 |
| groups: each arm / both arms | 734,375 / 539,306,640,625 (5.39 × 10¹¹) |
| groups: each hand / both hands | 378 / 142,884 |
| groups: style × hold | 1,920 |
| groups: intent × touch × walk | 51,600 |
| product of the groups | equals the descriptor count exactly |
| unordered pairs of distinct behaviours | about 10⁷³·⁸ |
| ordered sequences of 2 / 3 / 5 / 10 behaviours | 10⁷⁴·¹ / 10¹¹¹·² / 10¹⁸⁵·³ / 10³⁷⁰·⁶ |
| time to enumerate at 10⁹ descriptors per second | 3.7 × 10²⁰ years, 2.7 × 10¹⁰ ages of the universe |
| fraction ever sampled in training (1,814 behaviours) | 1.6 × 10⁻³⁴ |

These are counts of what is **describable**, not of distinct meaningful behaviours, and only a few thousand were
ever executed. Relative to the 55 policies the space is larger by a factor of 2.1 × 10³⁵.

* `BehaviorSelector` turns drives, affect and the state of the eyes into a *desire* over 18 affinity tags
  (`vigilance, defensive, explore, social, comfort, fatigue, ocular, pain, itch, cold, heat, hunger, boredom,
  energetic, expressive, soothe, object, relax`), samples **40 candidate descriptors** per decision from the
  structured prior (about 2 decisions per second, so 80 scored candidates per simulated second), and scores each
  by expected free energy using the learned forward model, with a repetition penalty and a balance guard.
* `BehaviorExecutor` runs the chosen behaviour at 50 Hz: joint targets with speed, amplitude, tremor and rhythm;
  gaze, lids and blink patterns; jaw and vocalisation; weight shift and crouch; and, through the skill layer,
  reaching, pointing, grasping and touching itself.
* Two modes: **autonomous** (these behaviours are everything the person does) and **ambient** (the
  language-model mind is in charge of deliberate action and this layer supplies the involuntary, expressive and
  housekeeping behaviour on the channels the mind is not using: glances, shifting, blinking, a hand that goes
  to a dry eye). The mind can also call `express(...)`.

### 17.6 Learning to use its own body (`body_learning.py`, `tools/train_body.py`)

Most descriptors are harmless; some unbalance the body. `BodySafety` is a logistic model over the *parts* of a
behaviour: **30 components** (head yaw, arm schema, stance, speed, …) one-hot encoded into 300 features plus 8
continuous summaries (for example how far the arms are from rest and how fast they are asked to move), i.e.
**308 weights** in all. It predicts the chance that the behaviour moves the COM more than 10 cm from the feet
or makes the person fall. It is fitted offline from many bodies babbling in parallel (8 workers, 270 simulated
seconds per copy, 3 rounds, **1,814** training behaviours; held-out AUC **0.777**; later rounds babble *under
the current model* so as to concentrate near the safe/unsafe boundary) and keeps adapting online from each
behaviour's outcome. The selector adds the hazard −log(1 − p) to a candidate's expected free energy, so unsafe
behaviours become unattractive rather than impossible.

Evaluation (fresh copies, same budget of 1,440 simulated seconds = 24 minutes each, at `base`, because the
balance physics does not depend on receptor counts):

| | behaviours | unsafe per 100 | falls | falls per simulated hour | distinct |
|---|---|---|---|---|---|
| unconstrained babbling | 463 | 45.1 | 123 | 307.5 | 463 (100 %) |
| with the learned safety model | 465 | 11.8 | 35 | 87.5 | 460 (99 %) |

Falls per hour fell by 71.5 % and unsafe choices by 73.8 %, with variety preserved, **but the person still falls
about 87 times per simulated hour**; a model with AUC 0.78 is a useful filter, not a solution. Weights are in
`embodied_human/body_safety.json`, the report in `out/train_body_report.json`.

### 17.7 Limitations of this part

* **Not verified end to end at the higher levels.** Sizes and step cost were measured at four levels; a long
  autonomous run at `extreme` or `max` was not completed, and the `run_sim.py` reference run (§11) was repeated
  only at `base`.
* The cost is real: about 1–1.5 GB of RAM per person at `base` (more at higher levels), and 0.16–0.54× real time.
* The behaviour space is wide but each behaviour is a motion template, not a discovered skill.
* The ocular, organ and neural-mass models are plausible toys.
* Learned safety was evaluated only at `base`, with random babbling, not in a long autonomous life.
* In the reference run of §11 only the `hold` policy was logged; why the behaviour layer shows no variety there
  has not been investigated.

---
