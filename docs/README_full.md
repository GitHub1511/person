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
| Python modules in `embodied_human/` | 39 files, 18,800 lines, 124 classes, 676 functions | static analysis (`ast`), §12 |
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
| Fall recovery (`stand_up`, §16.7) | Under active development; the single completed diagnostic trial (2026-10-07) ended still down (§16.7). |
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

## 1. Rationale: why the architecture is not merely a sensor dump

A system with 48,000 touch channels is not more embodied than one with 48. The distinguishing
property is that the signals arrive **late**, **adapt**, are **predicted by the system itself** and
**matter** functionally. Seven mechanisms implement this; each is measured in the outputs, so the
corresponding claims can be checked.

| # | mechanism | location | functional contribution | quantity |
|---|---|---|---|---|
| 1 | conduction delay by fibre class | `afferents.py` | the brain acts on a stale image of the body, so a forward model is *necessary* | 18 ms (Aβ), 60 ms (Aδ), 250 ms (C), 30 ms (efferent); 13 afferent populations |
| 2 | receptor adaptation and habituation | `receptors.py`, `afferents.py` | steady contact fades perceptually, as with the habituation to clothing | 4 vibration-state values per taxel |
| 3 | reafference cancellation (corollary discharge) | `afferents.py` | self-generated touch is perceived differently from imposed touch | weights 1,992 × 4 × 52 = 414,336 at `base`; 17,928 × 4 × 52 = 3,729,024 at `extreme` |
| 4 | learned body schema `∂proprio/∂action` | `predictive.py` | body ownership becomes a measurable quantity | 572 × 624 = 356,928 weights |
| 5 | interoception and homeostatic drives | `interoception.py`, `drives.py` | the agent has intrinsic needs | 67 variables, 21 drives |
| 6 | affect as gain-and-setpoint control | `affect.py` | emotions change what the agent prefers, not only its expression | 28 × 10 = 280 appraisal weights; 25 modulators |
| 7 | segmental postural prioritisation | `motor.py` | the body protects its balance before voluntary movement | capture-point criterion (§9) |

The reafference matrix (mechanism 3) is the single largest array of the sensory pipeline: at the
`extreme` level it holds 3.73 million weights, 74 % of the afferent system's 5.06 million array values
(measured by `tools/readme_runtime_facts.py`).

---

## 2. The body model

The model is generated programmatically from the declarative tables of `skeleton.py` and `skin.py`;
`embodied_human/build_model.py` writes `out/models/human.xml` (an MJCF file of about 18,500 lines at the
`extreme` level).

| quantity | value |
|---|---|
| Skeleton | **38** bones in a parent-before-child tree |
| MuJoCo bodies | **44** = world + 38 body segments + 5 free scene objects (mug, apple, stone, cushion, toy ball) |
| Joints | **58** = 52 actuated hinge joints + 1 free root joint + 5 free-object joints |
| Degrees of freedom | **94** generalised coordinates (`nq`), **88** velocities (`nv`): 52 + 7 + 5·7 and 52 + 6 + 5·6 |
| Actuators | **52** (one torque motor per actuated joint) |
| Mass | **71.68 kg** for the person (every segment carries a human mass fraction), 73.41 kg including the scene objects |
| Geoms | **68** = 58 that collide + 10 visual only (the eyes, irises, pupils and eyelids) |
| Sites | **2,005** at `base` (1,992 taxel sites + 13 landmarks); 17,941 at `extreme` |
| Sensors | **2,238** at `base` (2,306 `sensordata` values); 18,174 at `extreme` (18,242 values) |
| Cameras / lights / equality constraints | 3 / 3 / 10 |
| Time step | 1 ms |

The largest segments are the chest (14.0 kg), the pelvis (9.3 kg), each thigh (7.04 kg), the abdomen
(7.0 kg), the head (4.28 kg) and each shin (4.2 kg).

The body faces **−y**, the vertical axis is **+z**, and every joint axis is derived from that convention
in `skeleton.build_bones()`. The convention is consequential: in an early version the knee, elbow, wrist
and finger axes lay in the *frontal* plane rather than the sagittal plane, so the legs had no knee flexion
at all and no controller could bring the body to a standing posture.

### 2.1 Actuator torque limits

Limits follow isokinetic dynamometry for the legs. The upper body and trunk are **deliberately assisted
above human norms** so that the body can push itself back up after a fall (the push-up margin was 1.04 and
the cobra posture 1.4 before the change); the legs keep human values because standing is tuned on them.

| joint group | limit (N·m) | joint group | limit (N·m) |
|---|---|---|---|
| spine bend / side / twist | 150 / 100 / 55 | hip flexion / abduction / rotation | 160 / 110 / 55 |
| neck | 18 | knee | 200 |
| jaw | 45 | ankle flexion / inversion | 110 / 60 |
| eye | 0.05 | toe | 45 |
| shoulder flexion / abduction / rotation | 110 / 90 / 40 | wrist flexion / deviation | 14 / 10 |
| elbow | 80 | finger / thumb (per joint) | 2.6 / 3.2 |

Finger torque was lowered from an earlier 18 N·m because it crushed objects and registered as phantom
pain; a human finger flexor produces a few newton-metres at the metacarpophalangeal joint.

### 2.2 Collision filtering (`build_model.py`)

Contact between two geoms is enabled when `(contype_A & conaffinity_B) or (contype_B & conaffinity_A)`.
Six collision categories exist (*limb*, *left hand*, *right hand*, *body* (trunk and head), *world*,
*object*), which give C(6,2) + 6 = **21** unordered category pairs, of which **15** are enabled:

```
         limb  hand_l  hand_r    body   world  object
  limb  False    True    True   False    True    True
hand_l   True   False    True    True    True    True
hand_r   True    True   False    True    True    True
  body  False    True    True   False    True    True
 world   True    True    True    True   False    True
object   True    True    True    True    True    True
```

* The six disabled pairs are limb–limb, limb–body, body–body, hand–hand (each hand with itself) and
  world–world. Limbs therefore never collide with limbs or the trunk, so the body cannot disintegrate
  through self-intersection (for example a pelvis capsule lying inside a thigh capsule).
* Geoms of the *same hand* never collide. This removes the deep finger–finger interpenetration that
  previously registered as **370 kPa of phantom pain** at the fingers.
* The two hands, and either hand against the trunk or head, remain in collision, so **self-touch is
  possible**, which is the essential purpose of a body schema.

### 2.3 Visible eyes

Each eye consists of an eyeball, an iris and a pupil, and the lids are visual-only geoms that are moved at
run time by `ocular.EyeRig`; none of them collides or carries mass (they account for the 10 non-colliding
geoms), so adding them did not change the standing behaviour.

---

## 3. The sensory vector space

### 3.1 Tactile sensation

Taxels (tactile sensing elements) are laid out on the geometric surfaces of the body model by `skin.py`
(cylindrical wraps around limb capsules, spherical caps on the head, grids on box faces). There are
**46 skin patches**, and each taxel has its own receptor density profile; densities differ by roughly an
order of magnitude across the body. The number of taxels scales with the square of the skin-density knob
of the complexity level (§17.1):

| level | skin density | taxels | channels per taxel | tactile scalars |
|---|---|---|---|---|
| `base` | 1 | **1,992** | 26 | 51,792 |
| `rich` | 2 | **7,968** | 43 | 342,624 |
| `extreme` | 3 | **17,928** | 43 | 770,904 |
| `max` (also `ultra`, `mega`) | 4 | **31,872** | 43 | 1,370,496 |

The 46 patches range from 8 taxels (`base`) to 112 (the thigh), so a patch holds 0.4 % to 5.6 % of the
skin. Distinct patches form C(46,2) = **1,035** unordered pairs; distinct taxels form C(1992,2) =
**1,983,036** pairs at `base` and 160,697,628 pairs at `extreme`, which is why contact is attributed to
taxels by a local receptive field and never by pairwise search. The full patch table is Appendix A.2.

The 43 channels are the 26 original ones (below) plus 17 extended ones that exist at `rich` and above
(`hair_deflection`, `stretch_u`, `stretch_v`, `stretch_energy`, `edge_gradient`, `tickle`, `wetness`,
`blood_flow_local`, `piloerection`, `sensitisation`, `local_inflammation`, `pruritogen`,
`receptor_fatigue`, `noci_a_delta`, `noci_c_poly`, `irritant`, `ischemia`); the last five are fed by the
inner world (§17.4). In the table, a.u. denotes arbitrary units, SA slowly adapting and FA fast adapting.

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

**Reconstruction of the field.** The native `touch` sensor of MuJoCo is gated by `site_size` and sums entire
contact forces, which blurs the tactile image. Each MuJoCo contact is therefore *splatted* onto the taxels of
both contacting bodies with a Gaussian receptive field (σ = 28 mm) and an outward-facing test, which yields
a smooth, spatially faithful field. The native touch sensors are generated as well (1,992 at `base`), so that
both computational paths exist.

**Pain is thresholded on pressure, not on force**, with a site-specific threshold
(`NOCI_PRESSURE_THRESHOLD_KPA`): 250 kPa on a fingertip and 700 kPa on a sole. Thresholding on total force
would register every step as severely painful because a sole taxel legitimately carries 50 N.

**Skin temperature** is a per-taxel state variable driven towards core temperature by perfusion, cooled by the
ambient temperature, and exchanging heat by conduction with whatever is touched. Standing on a floor at 22 °C
cools the soles to about 27.5 °C over 20 s (`fig_homunculus.png`).

### 3.2 Proprioception

52 joints × 13 channels = **676 scalars** (C(52,2) = 1,326 joint pairs). The channels are `q`, `qd`, `qdd`;
the muscle **spindle Ia** afferent (length *and* rate, hence silent during a static hold and active during
movement); spindle **II** (length only); the **Golgi tendon organ** (force); efference copy; measured torque;
torque error; joint-limit proximity; muscle length; muscle velocity; and sense of effort.

### 3.3 Remaining modalities and cell populations

| modality | channels | notes |
|---|---|---|
| Vestibular | 18 | 3 semicircular canals **with cupula adaptation**, 3 otolith organs, gravity direction, tilt, yaw, magnetometer |
| Vision | 17 + retina | 17 summary channels (luminance, contrast, motion energy, foveal and peripheral terms, pupil, blink, saccade, gaze, RGB, edge density, minimum depth) plus a retina bank rendered from the egocentric camera: 48×36 cells × 7 values = 12,096 (`base`), 72×54×7 = 27,216 (`rich`), 96×72×7 = 48,384 (`extreme`), 128×96×7 = 86,016 (`max`) |
| Audition | 24 bands + 6 | log-spaced cochlear filterbank excited by contact transients (bone conduction) and by self-generated motor noise |
| Olfaction | 24 | concentration from the odour signatures of scene objects, with distance falloff and receptor adaptation |
| Gustation | 5 | sweet, salty, sour, bitter and umami on tongue contact |
| Interoception | 67 | §4 |

Beyond the single-number channels, `senses_ext.py` generates **populations of cells**, each a thresholded,
saturating, adapting unit with randomised parameters (broad overlapping tuning, push–pull between ears). The
non-visual population values per frame are **439** (`base`), **3,032** (`rich`), **6,400** (`extreme`) and
**12,784** (`max`). At `extreme` they are 4,160 muscle spindle and Golgi tendon values, 1,152 vestibular
(canal and otolith) values, 392 cochlear values, 646 olfactory values and 50 taste values.

**Total sensory scalars per frame**, measured with vision off (`tools/scale_complexity.py --measure`):

| level | sensory scalars | × base | with retina bank |
|---|---|---|---|
| `base` | 52,981 | 1.0 | 65,077 |
| `rich` | 346,406 | 6.5 | 373,622 |
| `extreme` | 778,054 | 14.7 | 826,438 |
| `max` | 1,384,030 | 26.1 | 1,470,046 |

At `base`, `agent.describe()` reports 65,248 scalar state channels in all, which adds the retina bank (12,096)
and the 84 affective and drive channels (28 emotions + 25 neuromodulators + 10 appraisal + 21 drives) to the
sensory ones. The brain compresses all of this into a **572-dimensional learned latent** (`agent.LatentSpec`),
which builds the *current* and the *preferred* latent jointly so that the two can never drift out of alignment
(§7).

---

## 4. Interoception: 67 variables across 11 organ systems

Interoception, the sensing of the internal state of the body, is integrated as a system of ordinary
differential equations (ODEs). A sprint therefore accumulates lactate and an oxygen debt, cold constricts the
cutaneous blood vessels, and pain is gated by endorphins released under stress. The 67 variables admit
C(67,2) = **2,211** pairs and C(67,3) = **47,905** triples; the ODEs couple a sparse subset of them.

| system | n | variables |
|---|---|---|
| cardiovascular | 10 | heart rate, heart rate variability (HRV; root mean square of successive differences, RMSSD), stroke volume, cardiac output, systolic and diastolic blood pressure (BP), cutaneous and muscular perfusion, baroreflex error, venous return |
| respiratory | 7 | respiratory rate, tidal volume, minute ventilation, peripheral oxygen saturation (SpO₂), arterial CO₂ and O₂, **air hunger** |
| metabolic | 8 | glucose, glycogen, lactate, adenosine triphosphate (ATP) reserve, metabolic rate, energy expended, insulin, glucagon |
| thermoregulatory | 7 | core temperature, skin temperature, shivering, sweating, thermal discomfort, heat production and loss |
| gastrointestinal | 6 | gastric fullness, ghrelin, leptin, nausea, gut motility, nutrient absorption |
| renal | 6 | hydration, osmolality, bladder fullness, sodium, potassium, urine output |
| fatigue | 6 | peripheral and central fatigue, adenosine, sleepiness, alertness, recovery need |
| immune | 4 | cytokines, inflammation, sickness behaviour, tissue damage |
| nociceptive | 5 | nociceptive input, pain intensity, pain unpleasantness, **central sensitisation**, descending inhibition |
| effort | 4 | sense of effort, breathlessness, motor command magnitude, muscle tone |
| endocrine | 4 | adrenaline, noradrenaline, cortisol, insulin sensitivity |

(10 + 7 + 8 + 7 + 6 + 6 + 6 + 4 + 5 + 4 + 4 = 67.)

Two modelling details merit emphasis. First, **the sense of effort is derived from the corollary discharge and
not from the muscle**: it increases with the motor command even when the muscle cannot deliver. Second,
**pain is gated**: `pain = input × (1 + sensitisation) × (1 − 0.75 × descending inhibition)`, where the
descending inhibition is driven by endorphins and low stress. This corresponds to stress-induced analgesia and
accounts for the lower pain reported by a frightened agent in response to the same injury; the coefficient 0.75
is a chosen parameter, not a fitted one.

---

## 5. The emotional vector space

### 5.1 Appraisal (10 dimensions) → emotions (28) → neuromodulators (25)

Scherer's Component Process Model is condensed to ten evaluative dimensions: `novelty,
intrinsic_pleasantness, goal_relevance, goal_congruence, agency_self, coping_potential, certainty,
norm_compatibility, urgency, social_evaluation`. Each emotion is a **weighted pattern over these dimensions**,
which renders the theory executable: the weight matrix has 28 × 10 = **280 entries** (`affect.APPRAISAL_W`).
Each emotion also has its own time constant, so that fear persists after the threat has ceased.

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

The 28 emotions are `fear, anxiety, anger, sadness, joy, surprise, disgust, contempt, trust, anticipation,
love, guilt, shame, pride, envy, jealousy, hope, awe, gratitude, embarrassment, curiosity, boredom, calm,
relief, loneliness, empathy, amusement, satisfaction`. They form C(28,2) = **378** pairs and C(28,3) = **3,276**
triples; if each were treated as merely on or off there would be 2²⁸ = **268,435,456** emotional patterns, and
the continuous blended state is far richer still.

The **25 neuromodulators** form the gain layer, and each is coupled to a functional effect: phasic dopamine
(τ = 0.4 s; reward prediction error) and tonic dopamine (τ = 45 s); serotonin (τ = 900 s; selectively attenuates
*negative* emotions); norepinephrine (arousal, policy temperature); acetylcholine (learning rate); oxytocin
(social touch); vasopressin; endorphin **→ pain gate**; enkephalin; endocannabinoid; cortisol; adrenaline;
noradrenaline; testosterone (dominance); estrogen; progesterone; prolactin; melatonin; adenosine; histamine
(itch); GABA (γ-aminobutyric acid); glutamate; substance P; BDNF (brain-derived neurotrophic factor); and
orexin. They admit C(25,2) = 300 pairs.

### 5.2 Core affect, mood and temperament

* **Core affect**: valence, arousal and dominance (the pleasure–arousal–dominance model of Mehrabian, PAD),
  together with tension and pleasantness, each with its own time constant.
* **Mood**: slow moving averages (τ = 900 s), such that the agent has a disposition and not merely a set of
  reactions.
* **Temperament**: five damped traits (neuroticism, extraversion, openness, agreeableness,
  conscientiousness) that **bias the appraisal process itself**; an agent high in neuroticism weights urgency
  more heavily and coping less.
* **Action tendencies**: approach, avoid, freeze, attack, withdraw and explore (6), the bridge from affective
  state to movement.

---

## 6. Homeostatic drives: the regulatory demands of the body

The model has **21 drives**: 16 original ones and 5 added with the inner world (§17.4). Each drive has a
level, an urgency (amplified by allostatic load and stress), a setpoint, the current value and a **slope**,
since a need that is deteriorating is more urgent than one that is merely poor.

`hunger, thirst, sleepiness, thermal_cold, thermal_heat, pain, air_hunger, nausea, bladder, fatigue, itch,
social_need, safety, curiosity, comfort, restlessness` (original 16) and `ocular_comfort, muscle_soreness,
gut_discomfort, mental_fatigue, shift_urge` (added 5).

The drives span C(21,2) = **210** pairs; with three coarse levels per drive (satisfied, pressing, urgent) the
motivational state space has 3²¹ = **10,460,353,203** cells, and with on/off levels 2²¹ = 2,097,152. The
drives constitute the **preferred observations** in latent space (§8); they therefore modify behaviour without
any reward function having been specified.

---

## 7. Predictive coding and the body schema

Three online learners are implemented in `predictive.py`. All are recursive least-squares (RLS)
estimators with forgetting and ridge regularisation; each stores a weight matrix `W` and an inverse-covariance
matrix `P`. The latent state has **572** dimensions (blocks below) and the action vector has 52.

| model | maps | weights `W` | covariance `P` | total |
|---|---|---|---|---|
| forward | 572 + 52 = 624 → 572 | 572 × 624 = **356,928** | 624² = 389,376 | 746,304 |
| inverse `a_t = f(s_t, Δs)` | 2 × 572 = 1,144 → 52 | 52 × 1,144 = **59,488** | 1,144² = 1,308,736 | 1,368,224 |
| body schema `∂proprio/∂action` | 624 → 572 | 572 × 624 = **356,928** | 624² = 389,376 | 746,304 |

The three learners hold 2,860,832 numbers; with precision and bookkeeping vectors the predictive system
contains **2,866,708** array values (measured). These sizes depend on the latent dimension only, not on the
complexity level: the latent is a fixed-size compression of 52,981 to 778,054 sensory scalars.

| latent block | q | qd | tactile | vestibular | visual | auditory | chem | intero | affect | ocular | inner | **total** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dimensions | 52 | 52 | 177 | 18 | 17 | 6 | 29 | 67 | 10 | 16 | 128 | **572** |

(The `ocular` and `inner` blocks exist at every level and are zero at `base`, where the inner world is off.)

The forward model, `p(s_{t+1} | s_t, a_t)`, is **initialised as the identity**, which encodes the prior that the
state remains unchanged in the absence of an action. Without it the model predicts a zero state, all candidate
policies appear equally poor, and policy selection is vacuous until the model has been learned.

The row norms of the body schema define **controllability**, the operational definition of bodily
self-attribution, and the per-dimension coefficient of determination (R²) of the forward model yields
**ownership**: a sensory dimension explained by the efference copy is attributed to the self. In the current
20 s reference run at `base` ownership reaches **0.942** and proprioceptive drift (the slow accumulation of
unexplained proprioceptive error, as in the vibration illusion) is **0.0323**; an earlier version of the
project reported 0.975 and 0.0016, before the latent was enlarged from 410 to 572 dimensions.

**Precision** is the inverse error variance per channel and corresponds to attention: an unpredictable channel
is down-weighted rather than allowed to dominate. Free energy is `F = inaccuracy + 0.05 × complexity` and is
reported both absolutely and per dimension (the raw sum over 572 dimensions saturates every downstream use).

---

## 8. Active inference

```
G(π) = risk  +  ambiguity  −  epistemic value  +  motor cost  +  emotional bias
```

* **Risk** is the divergence between the predicted sensory consequence of a policy and the *preferred*
  observation, which is constructed from drives, interoceptive setpoints and the current affective state. A
  fearful agent therefore prefers different sensations from a curious one.
* The prediction is treated as a **belief**: the proprioceptive block is expected to move most of the way
  towards the equilibrium point of the policy, and this expectation is blended with the learned forward model
  once it has accumulated data. The body must act *before* any learning has occurred, so selection has to be
  meaningful from the first tick.
* **Emotional action tendencies** bias whole policy *categories* and are deliberately excluded from the
  free-energy term: an action tendency is a low-level disposition, not a deliberative prediction, which is what
  allows fear to produce a flinch within a fraction of a second.
* Policy precision is a softmax whose temperature rises with arousal (clipped to 0.15–3.0).

The repertoire comprises **55 motor programs** (each a vector of 52 equilibrium targets, 2,860 numbers in all)
in 11 categories: posture, balance shifting, reaching (left and right), self-touch (hand-to-face, hand-to-chest,
rub-forearm), hand and grasp, gaze, head, face, defensive (flinch, freeze, brace, step-back), locomotion and
exploration. Selection runs at 10 Hz. Over a horizon of *h* decisions the library offers 55^*h* policy
sequences: 3,025 for *h* = 2, 166,375 for *h* = 3, 9,150,625 for *h* = 4 and 503,284,375 for *h* = 5; the
selector evaluates single policies, not sequences, so these are the sizes of the space it could be extended
to, not of what it searches.

**Observation (2026-10-07).** The 55-policy selector is bypassed whenever the generative behaviour layer
(§17.5) owns the body (`EmbodiedHuman._cognitive_tick`). In the current 20 s reference run at `base` the
logged policy name was `hold` for all 2,000 log ticks, against 12 of 55 policies used in an earlier
version of the project; the cause was not investigated here.

### Intrinsic motivation

The reward signal comprises ten terms, all internal: **jerk, energy, stability, novelty, tactile, comfort,
affective balance, empowerment, pain, contact smoothness**. The novelty term rewards *reducible* prediction
error, not noise.

---

## 9. The motor system and the corrections required to stand upright

Motor commands are issued as an **equilibrium point** (a virtual posture), not as torques, in accordance with
the λ-model used in biological motor control. The following mechanisms are layered upon it: a stretch reflex
with conduction delay, Golgi tendon protection, nociceptive withdrawal, righting, vestibulo-collic and startle
responses, a central pattern generator (CPG) and a protective stepping reflex.

Enabling a 72 kg humanoid to stand required **five separate corrections**, each of which remained undetected
until it was measured:

1. **Joint axes.** The knee, elbow, wrist and finger axes were defined in the frontal plane, so the legs had
   no knee flexion. No controller can compensate for this fault.
2. **The centre-of-mass (COM) velocity was always zero.** MuJoCo never writes `subtree_linvel[0]` (the
   recursion begins at body 1), so the balance loop contained *no derivative term*, equivalent to an undamped
   spring acting on an inverted pendulum. The COM of the humanoid is `subtree_com[pelvis]`, not
   `subtree_com[0]`, which includes the scene objects.
3. **The ankle rate limit.** The expression `max_delta = 40 × dt × (limit/60)` imposed a delay of **1.5 s
   before the knee reached full torque**, longer than the entire balance response. It was replaced by a rise
   time of 50 ms, which is also physically appropriate.
4. **Damping.** With an actuator lag of 45 ms, a proportional-derivative (PD) loop requires
   `kd/kp ≈ 2 × lag = 0.12 s`; at 0.06 s the legs oscillate and the body collapses.
5. **Controller authority.** The posture spring and the balance loop both act on the ankle; when their
   authority is equal the spring dominates, the centre of pressure (CoP) never travels, and the COM drifts away
   until the body falls. Balance is given priority at the ankle (×0.45).

The ankle strategy is formulated in terms of the **centre of pressure** by means of an inverted pendulum, so
that it saturates at the edge of the foot instead of over-driving the joint into its stop (where the
*constraint*, not the muscle, determines the motion and the body topples about the toe):

```
ω = sqrt(g / h_com)
CoP_desired = x + Kp·x + Kd·x'          Kp = 1,  Kd = 2ζ/ω,  ζ = 0.9
CoP         = clip(CoP_desired, −foot_front, +foot_back)
τ_ankle     = −m·g·(CoP − a)            a = ankle offset from the support centre
```

Two further mechanisms emerged from the debugging and correspond to physiological ones:

* **Segmental postural prioritisation.** When balance is threatened (measured by the **capture point**,
  `x + x'/ω`, which determines recoverability) voluntary displacement of the trunk and legs is attenuated
  towards the nominal configuration. The arms, hands, head and eyes retain full authority, since a 4 kg arm
  displaces the whole-body COM by only one to two centimetres. A single global gain would leave the agent unable
  to gesture whenever it is standing, which is always.
* **Protective stepping** that is **direction-aware**. Once the capture point leaves the support polygon no
  torque can save the body; the only correct action is to relocate the support polygon. An earlier version
  always swung the leg *forwards*, which arrests a forward fall but accelerates a backward one.

The CPG is gated by stability: it drives oscillation of the legs without a balance coupling, so engaging it
while standing invariably results in a fall.

---

## 10. The multi-rate control loop

A nervous system does not run at one clock rate; the architecture has seven nested rates (updates per
simulated second in parentheses):

```
1000 Hz   physics          MuJoCo (1 ms step)
 250 Hz   receptors        transduction (250 Hz at base; 100 Hz at rich and above, where the skin has 4-16x more taxels)
 200 Hz   afferents        conduction delay, rate coding, reafference
 100 Hz   interoception    organ systems, autonomic
  50 Hz   affect           neuromodulators, appraisal, emotion
  10 Hz   cognition        active inference, policy selection
   1 Hz   mood             allostatic slow state
```

That is 1,611 scheduled updates per simulated second at `base` (1,461 at `extreme`). The inner world (§17.4)
adds eight more clocks: neural mass 50 Hz, motor units 20 Hz, circulation and skin 10 Hz, association 10 Hz,
memory recall 4 Hz, chemistry and organs 2 Hz, circadian clock 1 Hz and memory store 1 Hz.

Each control step proceeds: **feel → interpret → want → predict → choose → move → feel again.**

---

## 11. Outputs

| file | contents |
|---|---|
| `out/episode.npz` | every layer: emotions, neuromodulators, appraisal, drives, 67 interoceptive variables, per-region tactile, full taxel snapshots, reward terms, proprioception, about 60 scalar headline channels |
| `out/episode_channels.csv` | the scalar channels, one row per logged tick |
| `out/episode_tactile_regions.csv` | per-region tactile aggregates in long format |
| `out/episode_manifest.json` | the complete description of the model and its senses |
| `out/summary.json` | episode statistics |
| `out/fig_affect.png` | core affect, all 28 emotions, all 25 neuromodulators, appraisal, action tendencies |
| `out/fig_interoception.png` | the internal milieu across the 11 organ systems |
| `out/fig_senses.png` | every exteroceptive channel, vestibular organs, balance strategies, body height |
| `out/fig_homunculus.png` | the tactile body map at 6 instants: force, pressure, nociception, temperature, SA-I, C-tactile |
| `out/fig_learning.png` | free energy, complexity, surprise, body schema, reward terms, reward prediction error |
| `out/fig_drives.png` | all drives and the policy timeline and histogram |
| `out/frames.png` | rendered views of the body |
| `out/models/human.xml` | the generated MJCF |

### Reference run (20 s, seed 7, `PERSON_COMPLEXITY=base`, vision on; 2026-10-07)

| quantity | value |
|---|---|
| fell | **no** |
| minimum pelvis height | 0.823 m |
| mean / maximum balance error | 1.34 mm / 9.0 mm |
| maximum pain | 0.0139 |
| body ownership / proprioceptive drift | 0.942 / 0.0323 |
| mean valence / arousal / stress | −0.016 / 0.342 / 0.058 |
| mean free energy (`summary.json`) | 28.54 |
| mean reward / empowerment | −0.029 / 0.116 |
| self-touch fraction | 0.0875 |
| policies used | 1 (`hold`, 2,000 of 2,000 log ticks) |
| mean emotions | joy 0.209, anger 0.192, hope 0.136, calm 0.074, satisfaction 0.070, guilt 0.061 |
| peak emotions | anger 0.375, joy 0.296, hope 0.261 |
| mean drives | comfort 0.464, thermal_cold 0.462, hunger 0.443, curiosity 0.348, sleepiness 0.147, itch 0.091 |
| speed | 57.9 s wall for 20.0 s simulated = 0.35× real time |

In the original reference run (an earlier version of the project, not re-measured here) the forward model's
prediction error fell steeply within the first second: the agent starts from the prior "nothing changes unless
it acts" and learned the sensory consequences of its own commands within about ten cognitive ticks.

---

## 12. Code map

### 12.1 Modules of `embodied_human/`

The table below is generated from the source tree by `tools/build_readme.py` (line, class and function counts
come from `ast`; the purpose is the first line of each module docstring).

| group | module | lines | classes | functions | purpose (first docstring line) |
|---|---|---|---|---|---|
| body and scene | `skeleton.py` | 466 | 3 | 15 | Skeleton specification for the embodied human. |
| body and scene | `skin.py` | 572 | 2 | 12 | Whole-body artificial skin. |
| body and scene | `build_model.py` | 807 | 2 | 25 | MuJoCo model generation. |
| body and scene | `state.py` | 114 | 2 | 4 | Shared state snapshot. |
| body and scene | `config.py` | 351 | 14 | 2 | Global configuration for the Embodied Human simulation. |
| body and scene | `complexity.py` | 160 | 1 | 4 | How complicated the person is. |
| sensing | `receptors.py` | 1,257 | 8 | 32 | Peripheral transduction: physics -> receptor potentials. |
| sensing | `senses_ext.py` | 364 | 7 | 15 | Populations of sensory cells, not single channels. |
| sensing | `afferents.py` | 351 | 3 | 14 | Peripheral nerve: receptor potentials -> afferent traffic. |
| sensing | `ocular.py` | 430 | 4 | 9 | The ocular surface: why the eyes need to blink. |
| physiology and inner world | `interoception.py` | 426 | 2 | 9 | Interoception: the sense of the internal state of the body. |
| physiology and inner world | `inner_organs.py` | 547 | 9 | 31 | The organs: compartmental physiology that the original 67-variable interoceptive |
| physiology and inner world | `inner_brain.py` | 356 | 5 | 21 | The brain-side inner world. |
| physiology and inner world | `inner_world.py` | 484 | 2 | 16 | The inner world: every internal system, run at its own rate and coupled to the |
| affect and drives | `affect.py` | 603 | 3 | 11 | Affect: neuromodulators, appraisal, core affect, emotions, mood, temperament. |
| affect and drives | `drives.py` | 214 | 2 | 6 | Homeostatic drives: what the body *wants*. |
| brain | `predictive.py` | 323 | 4 | 17 | Predictive coding: forward model, inverse model, body schema, precision. |
| brain | `active_inference.py` | 411 | 2 | 12 | Active inference: choosing what to do by minimising expected free energy. |
| brain | `intrinsic.py` | 121 | 2 | 5 | Intrinsic motivation: the reward decomposition. |
| brain | `behavior_space.py` | 660 | 3 | 29 | The space of things the person can *do*. |
| brain | `behavior_exec.py` | 642 | 3 | 22 | Choosing and executing behaviours from the generative space. |
| brain | `body_learning.py` | 224 | 1 | 11 | Learning to use its own body: which parts of a behaviour are safe. |
| motor control and skills | `motor.py` | 541 | 2 | 6 | Motor system: from intention to torque. |
| motor control and skills | `wbc.py` | 353 | 4 | 7 | Whole-body inverse-dynamics controller. |
| motor control and skills | `locomotion.py` | 741 | 3 | 27 | Bipedal walking. |
| motor control and skills | `skills.py` | 2,645 | 5 | 131 | Motor skills: everything the body can *do* on purpose. |
| motor control and skills | `speech.py` | 178 | 2 | 9 | Speech: the mouth, and the words above the head. |
| mind | `mind.py` | 996 | 9 | 37 | The mind: a language model behind the body. |
| mind | `world.py` | 184 | 1 | 18 | The world, as the body and the mind can query it. |
| mind | `azr_loop.py` | 165 | 0 | 7 | AZR control loop: reasoning proposes, physics disposes. |
| mind | `instance_log.py` | 108 | 1 | 12 | Per-instance identity and thought/action transcripts. |
| infrastructure | `agent.py` | 1,094 | 3 | 18 | The embodied human. |
| infrastructure | `record.py` | 295 | 1 | 7 | Episode recording. |
| infrastructure | `plots.py` | 519 | 0 | 12 | Visualisation. |
| infrastructure | `viewer.py` | 426 | 2 | 22 | Seeing the person: rendering, speech bubbles, thought bubbles, and the window. |
| infrastructure | `_fast.py` | 17 | 0 | 1 | Cheap replacement for np.clip on the per-step hot path (the np.clip wrapper |
| infrastructure | `ultra.py` | 593 | 6 | 34 | The ultra tier: a plug-in framework for subsystems far larger than the base person. |
| infrastructure | `ux_example.py` | 58 | 1 | 6 | A deliberately tiny reference subsystem: copy its structure, not its content. |
| infrastructure | `__init__.py` | 4 | 0 | 0 | Embodied human: a simulated person, not a policy. |
| **total** | **39 modules** | **18,800** | **124** | **676** |  |

### 12.2 Entry points, tools and diagnostics

Root scripts: `run_sim.py` (episode, figures, `--describe`), `run_mind.py` (the person with a language-model
mind, §16.5), `setup_azr.py` (how to obtain AZR), `smoke.py` (build and short run).

| script | lines | purpose |
|---|---|---|
| `tools/autopush.py` | 166 | Watch the project and push to GitHub whenever a file is saved. |
| `tools/build_readme.py` | 199 | Assemble README.md from hand-written sections plus tables generated from the code. |
| `tools/measure_identity.py` | 135 | First-person ownership score for an instance transcript file. |
| `tools/readme_check.py` | 89 | Check that an edited README section kept everything that is not wording. |
| `tools/readme_combinatorics.py` | 111 | Compute the combinatorial quantities quoted in the README from the code itself. |
| `tools/readme_describe.py` | 31 | Dump agent.describe() (and a few extra counts) for the current PERSON_COMPLEXITY to JSON. |
| `tools/readme_runtime_facts.py` | 59 | Inventory of what the running person is made of (array sizes per subsystem), for the README. |
| `tools/readme_verify.py` | 130 | Cross-check the numbers quoted in README.md against freshly computed or recorded values. |
| `tools/scale_complexity.py` | 188 | Set, inspect and measure how complicated the person is. |
| `tools/simlock.py` | 120 | Run a heavy command only when the machine can afford it. |
| `tools/train_body.py` | 303 | Learn, from many bodies at once, which behaviours are safe to perform. |
| `tools/ultra_bench.py` | 120 | Benchmark the ultra-tier subsystems one at a time, each in its own process. |

The experiments that produced §9 and §16 are the `diag_*.py` scripts in the repository root; each is runnable
and prints the measurement it was written for. In addition there are six one-off repair scripts (`fix_lunge*.py`)
left by the fall-recovery work (§16.7).

| script | lines | what it measures |
|---|---|---|
| `diag_app.py` | 37 | Open the Tk window for a few seconds, drive it programmatically, and save the |
| `diag_azr_balance.py` | 88 | AZR balance/move benchmark — real model only, no stubs. |
| `diag_azr_live.py` | 76 | Live AZR mind test — uses the real model in LM Studio, no stubs. |
| `diag_balance.py` | 95 | Diagnostic: tune standing balance with physically-derived gains. |
| `diag_balance2.py` | 98 | Diagnostic: refine balance gains and test perturbation recovery. |
| `diag_bubbles.py` | 41 | Render one frame with a speech bubble and a thought bubble over the head. |
| `diag_cobra.py` | 54 | Drive the real SkillSystem._rec_cobra in isolation, report peak COM. |
| `diag_fall.py` | 71 | Diagnostic: trace what actually happens during the fall. |
| `diag_getup.py` | 104 | Fall-recovery test: knock the body down, verify it gets back up. |
| `diag_getup_hands.py` | 69 | Render prone + candidate hand-plant poses side by side. |
| `diag_getup_shots.py` | 70 | Render a full stand_up attempt (6 shots). |
| `diag_handwalk.py` | 65 | Trace hand-walk mechanics: hand pos, spine q, COM per step. |
| `diag_hold.py` | 40 | Isolate the motor layer: force the policy to 'hold' and see if it stands. |
| `diag_long.py` | 27 | Trace a long episode to find when and why it falls. |
| `diag_lunge.py` | 43 | Isolate roll+lunge: side placement, drive _rec_roll only, report. |
| `diag_mind_server.py` | 82 | Exercise the real-backend code path (OpenAI-compatible /v1/completions with |
| `diag_mind_smoke.py` | 27 | Smoke test for the mind layer without any model: parse replies, check the |
| `diag_policy.py` | 40 | Diagnostic: why does the agent choose destabilising policies? |
| `diag_push2.py` | 61 | Diagnostic: trace exactly why a moderate push topples the body. |
| `diag_robust.py` | 43 | Sweep stabilisation parameters for robust standing across seeds. |
| `diag_robust2.py` | 38 | Sweep for stability AND expressiveness with segmental postural priority. |
| `diag_skills.py` | 113 | Exercise the skills one by one while standing (no walking): head, hands, |
| `diag_slip.py` | 49 | Measure hand/foot slip during press-plant: positions + contact forces. |
| `diag_stand2.py` | 32 | Diagnostic: does the body stay upright, and for how long? |
| `diag_sweep.py` | 59 | Sweep balance-controller variants with forced 'hold' and measure survival. |
| `diag_table_climb.py` | 38 | Table-climb test: tip prone near the table, auto-recovery runs, report. |
| `diag_walk.py` | 84 | Walking diagnostic: stand, walk forward, optionally turn, stop. |
| `diag_walk_eval.py` | 94 | Re-test the best gait parameter sets from a sweep under different noise seeds |
| `diag_walk_render.py` | 90 | Render hand-walk directly from prone placement. |
| `diag_walk_sweep.py` | 224 | Random search over gait parameters, run in parallel. |
| `diag_walk_turn.py` | 53 | Walk in a circle (turning while walking) and turn on the spot. |
| `diag_wbc.py` | 72 | Stand using only the whole-body controller (no posture/balance system). |

---

## 13. Extension procedures

Each procedure below also changes the combinatorial quantities of the corresponding section; the figure in
brackets is the size before the change.

* **Additional skin region** — add a `SkinPatch` to `skin.default_patches()` (46 patches, 1,992 taxels at `base`).
* **Additional emotion** — add a profile (weights over the 10 appraisal dimensions) to `affect.EMOTION_PROFILES`
  together with entries in `EMOTION_VALENCE`, `EMOTION_AROUSAL` and `EMOTION_TAU` (28 emotions, 280 weights).
* **Additional neuromodulator** — register it in `NEUROMODULATORS`, `NM_BASELINE` and `NM_TAU`, and specify a
  target expression in `AffectSystem.update` (25).
* **Additional organ variable** — append to `interoception.INTERO_NAMES` and add the ODE in `update()` (67; the
  latent block `intero` grows with it, and so do the forward, inverse and schema matrices of §7).
* **Additional drive** — register it in `drives.DRIVES` with a setpoint and an expression (21).
* **Additional motor program** — add a `Policy` to `active_inference.default_policies()` (55).
* **Additional behaviour channel** — add a `Channel` to `BehaviorSpace.build_channels()`; the descriptor count is
  multiplied by its number of choices (1.16 × 10³⁷ at present).
* **Additional ultra-tier subsystem** — write `embodied_human/ux_<domain>.py` with a `@register`ed
  `Subsystem` (§19); nothing else needs to change.
* **Alternative personality** — pass a 5-vector `temperament` to `AffectSystem`.

---

## 14. Statement of limitations

* **Vision is a retina bank with hand-designed features**, not a learned encoder: 48×36 cells at `base` up to
  128×96 at `max`, each cell contributing 7 values. It preserves foveal and peripheral structure, saccades,
  blinks and the pupillary reflex; it is not a model of early visual cortex.
* **Audition is synthesised** from contact transients and self-motion, because MuJoCo provides no acoustic
  field; no sound propagation is modelled.
* **Olfaction is a distance falloff** from object odour signatures (and the `ultra` plume design of §19, which
  is not built); it is not an advection–diffusion model.
* **The forward model is linear** (RLS, 356,928 weights). It captures the local input–output structure of the
  body, but it cannot represent contact discontinuities or multi-step dynamics.
* **Contact is rigid-body contact.** Skin compliance is represented in the receptor layer (indentation, contact
  area, slip) rather than by soft bodies, so contact forces are stiffer than those of tissue.
* **The policy library is hand-authored** (55 programs). The generative behaviour space (§17.5) is wide
  (1.16 × 10³⁷ descriptors) but each behaviour is a parameterised motion template, not a discovered skill.
* **Free-energy magnitudes are not calibrated** to any physical unit; only relative differences matter.
* **Self-touch is spatially coarse.** Contacts are attributed to taxels by a Gaussian receptive field (σ = 28 mm),
  so a hand resting on the chest gives a smooth blob, not a resolved pattern.
* **Walking is unreliable** (§16.2) and **fall recovery is unfinished** (§16.7).
* **The physiological models are toys** with plausible orders of magnitude and the right qualitative couplings
  (§4, §17.3, §17.4); none is a clinical model, and counts of "state variables" are counts of numbers updated by
  dynamics, not a measure of biological fidelity.
* **Counts of what is describable are not counts of what has happened.** The 1.16 × 10³⁷ descriptors were
  never enumerated or sampled beyond a few thousand behaviours (§17.6).
* **The learned safety model** has a held-out AUC of 0.777 and the person still falls about 87 times per
  simulated hour (§17.6).
* **Most of the components of §16–§19 are newer and less measured** than §2–§11, and §18 describes code that
  modifies this repository without human review of each step.

---

## 15. References for the physiological content

The sources underpinning the physiological content are organised by topic. Receptor classes and densities:
Johansson & Vallbo (1983); Vallbo & Johansson (1984). Affective touch: Löken et al. (2009); Olausson et al.
(2010). Nociceptor transduction and sensitisation: Julius & Basbaum (2001); Woolf (2011). Pressure pain
thresholds by body site: Rolke et al. (2006). Interoception and allostatic load: Craig (2009); Sterling (2012);
Barrett & Simmons (2015). Appraisal theory: Scherer (2001); Lazarus (1991). Core affect: Russell (1980);
Mehrabian & Russell (1974). Predictive coding and active inference: Rao & Ballard (1999); Friston (2010); Adams,
Shipp & Friston (2013). Corollary discharge and reafference: von Holst & Mittelstaedt (1950); Blakemore, Wolpert
& Frith (2002). Body schema and body ownership: Head & Holmes (1911); Botvinick & Cohen (1998). Postural
strategies: Horak & Nashner (1986); Winter (1995). Capture point: Hof, Gazendam & Sinke (2005).
Equilibrium-point control: Feldman (1966); Bizzi et al. (1984). Whole-body control and walking based on the
divergent component of motion: Kajita et al. (2003); Englsberger et al. (2015); Sentis & Khatib (2005). Absolute
Zero Reasoner: Zhao et al. (2025), "Absolute Zero: Reinforced Self-play Reasoning with Zero Data".

---

## 16. Locomotion, manipulation, speech and a mind

This part was added on top of the body of §2–§11. It is presented in terms of what was *measured*, because
several components are demonstrations rather than solved problems.

### 16.1 Whole-body control (`wbc.py`)

Standing is controlled by inverse dynamics instead of the equilibrium-point spring. Every 4 ms (250 Hz) the
whole-body controller (WBC) solves one least-squares problem for the 88 generalised accelerations and the
contact wrenches. It tracks a desired centre-of-mass (COM) acceleration, the pelvis and chest orientation, the
swing-foot trajectory and a posture reference, subject to the floating-base dynamics (an 88 × 88 mass matrix),
unilateral contacts, a centre-of-pressure box inside each foot, a friction cone and a torsional-friction
limit. Violated inequality constraints are handled with an active set (pins). Joint torques are clipped to the
limits of §2.1. The eyes, jaw and fingers remain under their own controllers.

### 16.2 Bipedal walking (`locomotion.py`)

A divergent-component-of-motion (capture-point) planner places each foot, with `beta_step` (1.06) scaling the
aggressiveness of the placement; the gait states are `stand → init → unload → single support ↔ double support →
settle → stand`. Step length is integrated to meet the commanded speed, swing feet follow minimum-jerk paths
and the heading is drawn towards the commanded heading. `GaitParams` holds 41 float parameters, of which
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

### 16.3 Arms, hands, grasping and holding (`skills.py`)

`skills.py` has 131 functions; `SkillSystem` has 89 methods, 36 of them public (the `api_*` calls of §16.5 and a few state accessors).

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

### 16.4 Speech and thoughts (`speech.py`, `viewer.py`)

Speech opens and closes the jaw with a syllable-timed envelope (updated on the 50 Hz skill tick, no fine mouth shape). The spoken
content appears in a white bubble above the head and the thoughts appear above the head in grey italics. It can
optionally be rendered aloud through the Windows Speech API (`--voice`). Heard speech is delayed by the
distance at the speed of sound (343 m/s). There is **no built-in drive to speak**; the person speaks only if the
mind decides to. The jaw is moved kinematically and its actuator is muted while speaking (an earlier version let
the actuator oppose the override and the reaction torque knocked the body over).

### 16.5 The mind (`mind.py`, `world.py`, `run_mind.py`)

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

**The control loop (`azr_loop.py`): reasoning proposes, physics disposes.** AZR is a code-reasoning
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
`diag_walk_render.py`. **Measured status (2026-10-07).** One trial of `diag_getup.py --trials 1` (tipped onto the side from standing, 30 s allowed, automatic recovery on) ended **still down**: COM height 0.19 m at the end against the 0.72 m success criterion (peak 0.4 m), with six recovery attempts, each logging "pushup-pos made no progress" or "cobra made no progress". A first run with two trials did not finish within 400 s and reported nothing. The success rate is therefore 0 of 1 in the single completed trial; one trial is not a rate, and the code was being edited by another session while this was measured.

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

| knob | `base` | `rich` | `extreme` | `max` (= `ultra` = `mega`) |
|---|---|---|---|---|
| `skin_density` | 1.0 | 2.0 | 3.0 | 4.0 |
| `extended_tactile` | False | True | True | True |
| `spindles_per_muscle` | 1 | 8 | 16 | 32 |
| `gto_per_muscle` | 1 | 4 | 8 | 16 |
| `canal_afferents` | 1 | 24 | 64 | 128 |
| `otolith_hair_cells` | 1 | 64 | 192 | 384 |
| `cochlear_bands` | 24 | 96 | 192 | 384 |
| `olfactory_receptors` | 24 | 160 | 320 | 640 |
| `taste_cell_types` | 5 | 24 | 48 | 96 |
| `retina_w` | 48 | 72 | 96 | 128 |
| `retina_h` | 36 | 54 | 72 | 96 |
| `corneal_units` | 16 | 96 | 192 | 384 |
| `tear_film_sectors` | 16 | 48 | 96 | 192 |
| `neural_units` | 512 | 4096 | 12288 | 32768 |
| `neural_indegree` | 16 | 32 | 48 | 64 |
| `motor_units_per_muscle` | 4 | 16 | 32 | 64 |
| `alveoli` | 16 | 96 | 192 | 384 |
| `nephron_groups` | 16 | 96 | 192 | 384 |
| `liver_zones` | 8 | 48 | 96 | 192 |
| `vascular_beds` | 12 | 48 | 96 | 192 |
| `gut_segments` | 6 | 12 | 16 | 24 |
| `microbiome_taxa` | 8 | 32 | 64 | 128 |
| `analytes` | 48 | 240 | 480 | 960 |
| `cytokines` | 8 | 32 | 64 | 128 |
| `immune_populations` | 8 | 24 | 32 | 48 |
| `circadian_oscillators` | 16 | 96 | 192 | 384 |
| `episodic_capacity` | 256 | 2048 | 4096 | 8192 |
| `episodic_dim` | 32 | 64 | 96 | 128 |
| `tactile_hz` | 250.0 | 100.0 | 100.0 | 50.0 |
| `inner_world` | False | True | True | True |

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

### 17.3 Eyes that dry and blink (`ocular.py`)

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

| channel | choices | log10 |
|---|---|---|
| `head` | 693 | 2.84 |
| `eyes` | 63 | 1.80 |
| `lids` | 35 | 1.54 |
| `mouth` | 54 | 1.73 |
| `torso` | 245 | 2.39 |
| `l_schema` | 47 | 1.67 |
| `l_arm_0` | 5 | 0.70 |
| `l_arm_1` | 5 | 0.70 |
| `l_arm_2` | 5 | 0.70 |
| `l_arm_3` | 5 | 0.70 |
| `l_arm_4` | 5 | 0.70 |
| `l_arm_5` | 5 | 0.70 |
| `r_schema` | 47 | 1.67 |
| `r_arm_0` | 5 | 0.70 |
| `r_arm_1` | 5 | 0.70 |
| `r_arm_2` | 5 | 0.70 |
| `r_arm_3` | 5 | 0.70 |
| `r_arm_4` | 5 | 0.70 |
| `r_arm_5` | 5 | 0.70 |
| `l_hand` | 14 | 1.15 |
| `l_hand_0` | 3 | 0.48 |
| `l_hand_1` | 3 | 0.48 |
| `l_hand_2` | 3 | 0.48 |
| `r_hand` | 14 | 1.15 |
| `r_hand_0` | 3 | 0.48 |
| `r_hand_1` | 3 | 0.48 |
| `r_hand_2` | 3 | 0.48 |
| `stance` | 75 | 1.88 |
| `style` | 320 | 2.51 |
| `hold` | 6 | 0.78 |
| `intent` | 25 | 1.40 |
| `touch` | 129 | 2.11 |
| `walk` | 16 | 1.20 |
| **product (all 33 layout entries)** | **11,575,326,076,025,945,959,687,500,000,000,000,000** | **37.064** |

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

## 18. The OpenRouter planner and Pi coder loop (`tools/openrouter_helper/`)

A bash helper that keeps proposing the next step on this project, has a coding agent implement it, verifies
the result and undoes it if it broke the person. It comprises about 2,200 lines (`helper.sh` 266, `hlib.py` 966,
`proxy.py` 487, `ui.py` 413, `smoke.py` 64) and four offline test suites. Its own `README.md` documents every
command; in outline:

```bash
bash tools/openrouter_helper/helper.sh run                       # plan → Pi codes it → verify → rollback on FAIL → repeat
bash tools/openrouter_helper/helper.sh run-goal "<goal>" 12     # the same, steered by a standing goal, at most 12 cycles
bash tools/openrouter_helper/helper.sh ui                        # localhost web console (port 8770): goal box and live feedback
bash tools/openrouter_helper/helper.sh stop
```

**Components.**

| component | what it does | quantities |
|---|---|---|
| planner (`hlib.py`) | reads the newest `.npz` and JSON run files, the complexity table, an `ast`-generated API outline, git state, earlier outcomes and the machine's free resources; asks a free model for one next step | context of about 32 kB per plan |
| gateway (`proxy.py`, 127.0.0.1 only) | the only path to OpenRouter: model order, quotas, retries, streaming, tool calls | 5 models in order; **200 requests per model per UTC day**; **20 per minute** (≈3.25 s gap); **1,000 per day** (5 × 200); counts persist across restarts |
| coder | [Pi](https://pi.dev) in an isolated config directory, model = the gateway only, no API key in its environment, a tool-call budget (default 400) and a time limit (default 3,600 s) | permission guard `guard.ts` logs every tool call to `state/pi_audit.jsonl` |
| verify | changed files compile; the package imports at `base` and `rich`; a 20 s smoke run has no exception, no NaN, no fall and is not >25 % slower; helper and key untouched; on FAIL the files are rolled back | |
| console (`ui.py`) | goal box, loop state, quota usage, last PASS/FAIL with reasons, step history, Pi's latest report (refresh 3 s) | stdlib only |

**Quota arithmetic.** 5 models × 200 = 1,000 requests a day; at 20 per minute the whole day's allowance could be
used in 50 minutes. If every request used the maximum output length the daily ceiling would be
2 × 200 × 200,000 + 200 × 30,000 + 200 × 50,000 + 200 × 30,000 = **102 million output tokens**, but models do not
reliably honour long output targets. Every Pi turn is one request, so a coder step with 30 tool calls uses about
30 of one model's 200, giving on the order of 10–30 coder steps a day.

**Live probe (2026-10-07).** `nemotron-3-super-120b-a12b:free` and `laguna-s-2.1:free` work;
`nemotron-3-ultra-550b-a55b:free` returned 503 (overloaded) intermittently; `inkling-small:free` returns 403
unless the caller is a coding agent (Pi identifies itself honestly and is accepted); `ling-3.1-flash:free` is no
longer free (404) and is skipped. Effectively three models share the planner's day (600 requests).

**Tests** (all offline, on throwaway copies): `test_gateway.py` (9 scenarios: order, quotas, retry counting,
skipped models, `max_tokens` ladder, restart persistence, day rollover, per-minute spacing, harness gating),
`test_safety_net.py`, `test_run_loop.py` (the whole loop with a fake coder, killed gateways and a restarted
mock) and `test_pi_coder.py` (real Pi against the real gateway with a mock model; the guard blocked `git push`,
reading the key file, writing outside the project and data-sending web requests, and allowed ordinary
commands).

### 18.1 What the first two live steps did (2026-10-07), and why they were reverted

Two coder steps ran on olfaction and taste (constant zero in the run files). Both **passed verify**
(smoke run 1.87 ms/step and 1.74 ms/step, no fall, no NaN, imports at `base` and `rich`), used `inkling-small`
through Pi (about 67 and 59 gateway requests, about 5 and 6 minutes) and tripped no guard rule. Reading the
diffs showed that both degraded the project:

* Step 1 added constants (0.02 above a height of 0.1 m, 0.01 near a fixed point) and overwrote the receptor
  output with them; nothing depended on hunger, blood chemistry or breathing, no complexity flag and no
  validation script were added.
* Step 2 added `stimuli.py` (two fixed sources at (±1, 0, 0.5) with 1/distance falloff) behind a flag enabled only
  at `rich`, and **rewrote `ChemoSystem.sense`, deleting** the original object-based smell (distance falloff,
  adaptation, novelty) and tongue-contact taste. At the default `extreme` level smell and taste became exactly
  zero.

Verify cannot see this kind of damage because the smoke run never touches smell or taste. **Both steps were
reverted** at the owner's request (the eight edited files were restored to their state before step 1 and the
new file was deleted). The lesson is recorded here because it bounds what the loop can be trusted with: it
guarantees that the simulation still runs, not that a step did its job. The helper's own history records the
step title as "?" because of a Windows path problem in `helper.sh`; this is a known bug.

### 18.2 The README helper

`tools/openrouter_helper/readme_helper/` is a separate agent that sends **one** prompt describing the whole
project and the current README to `nemotron-3-super-120b-a12b:free` and installs the reply as `README.md` (with
`--apply`; a backup is kept under `state/backups/`). In `loop --apply` mode it does this every hour. Its three
applied runs on 2026-10-07 reduced a 1,028-line README to 66–69 lines. The README you are reading is therefore
built **deterministically** by `tools/build_readme.py` from `docs/readme_src/`, and a copy is kept at
`docs/README_full.md`; if the hourly agent is running it will overwrite `README.md` again (see Appendix B for
how to rebuild it).

---

## 19. The ultra tier (framework built, subsystems not built)

Two levels, `ultra` and `mega`, sit above `max`. They keep the classic body at `max` size (31,872 taxels,
1,384,030 sensory scalars, 88,892 internal variables) and add **plug-in subsystems**
(`embodied_human/ux_*.py`) that run beside it through a shared bus.

### 19.1 What exists

| item | file | size / status |
|---|---|---|
| framework | `embodied_human/ultra.py` | 593 lines, 6 classes (`Level`, `Bus`, `Subsystem`, `UltraFrame`, `UltraWorld`, `SyntheticBus`), 34 functions |
| bus | `ultra.BUS_DOC` | **59 documented standard inputs** (tactile array, 52 × 13 proprioception, 67 interoceptive variables, 28 emotions, 25 neuromodulators, 21 drives, inner-world scalars, body state, world state, scene objects) |
| output channels | `affect.<field>`, `drive.<name>`, `intero.<name>`, `desire.<tag>` | standing levels, summed over subsystems |
| presets | `complexity.PRESETS["ultra"]`, `["mega"]` | same classic sizes as `max` |
| reference subsystem | `embodied_human/ux_example.py` | 50,000 (`ultra`) / 1,000,000 (`mega`) leaky units; excluded from real persons |
| benchmark | `tools/ultra_bench.py` | each subsystem in its own process; state variables, MB, peak MB, ms per simulated second against declared budgets |
| memory lock | `tools/simlock.py` | slots and free-RAM check before heavy runs |
| conventions | `docs/ultra/CONVENTIONS.md` | the contract every builder follows |

Benchmark of the reference subsystem (2026-10-07): `ultra` 50,000 state variables, 1.4 MB, 6.8 ms per
simulated second; `mega` 1,000,000 state variables, 28 MB, 193.8 ms per simulated second; both within their
declared budgets (20 ms and 400 ms).

A subsystem declares its sizes per level (`level.pick(extreme=…, ultra=…, mega=…)`), update rate, summary length,
the bus keys it reads and writes, and budgets for wall time and memory; the optional wall-clock **governor**
stretches update periods (by up to 8×) when the machine is busy. Rules: numpy only, deterministic given its
seed, finite, no per-step large allocations.

### 19.2 Targets and the plan that did not run

| | `extreme` (measured) | `ultra` target | `mega` target |
|---|---|---|---|
| internal dynamic state | 36,932 | ≥ 3,000,000 (81×) | ≥ 40,000,000 (1,083×) |
| new sensory/event scalars per frame | 778,054 (existing) | ≥ 5,000,000 | ≥ 50,000,000 |
| extra wall time per simulated second | – | ≤ 6 s | ≤ 40 s |
| extra memory | – | ≤ 2.5 GB | ≤ 8 GB |
| parallel instances | – | 3–4 | 1 |

A design workflow for 20 domains (somatic afferents, spinal motor, vision pathway, audition, chemosense and
plumes, multisensory body schema, cardio-respiratory, endocrine-metabolic, immune repertoire, thermo-tissue,
visceral/renal, spiking cortex, subcortical loops, learning and memory, affect and personality, behaviour
grammar, environment and social, musculoskeletal tissue, AZR navigation, infrastructure), and a second team for
10 further domains (hierarchical predictive coding, planning and a world model, language and speech, lifespan
dynamics, haptic object perception, oculomotor/vestibular, spatial cognition, visible expression, attention and
workspace, valuation and hedonics) plus an audit of the classic modules, were launched. **Every agent errored out
and returned nothing; no design file or subsystem was produced.** The targets above are therefore design
intentions, not results, and the machine limits (i7-14700, 32 GB of which 5–10 GB are usually free, a 6 GB GPU
fully used by LM Studio, numpy only) decide how much of them is attainable.

---

## Appendix A. Quantities and combinatorics

### A.1 Summary of countable quantities

| quantity | count | derivation |
|---|---|---|
| Actuated DoF | 52 | one torque motor per joint |
| Joint pairs | 1,326 | C(52,2) |
| Skin patches / patch pairs | 46 / 1,035 | C(46,2) |
| Taxels (`base` … `max`) | 1,992 / 7,968 / 17,928 / 31,872 | 1,992 × (1, 4, 9, 16) |
| Taxel pairs (`base`) | 1,983,036 | C(1992,2) |
| Tactile channels per taxel | 26 (`base`), 43 (above) | 26 + 17 extended |
| Proprioceptive scalars | 676 | 52 × 13 |
| Collision categories / pairs / enabled | 6 / 21 / 15 | C(6,2) + 6 |
| Interoceptive variables / pairs / triples | 67 / 2,211 / 47,905 | C(67,2), C(67,3) |
| Appraisal weights | 280 | 28 emotions × 10 dimensions |
| Emotions / pairs / triples / on-off patterns | 28 / 378 / 3,276 / 268,435,456 | C(28,2), C(28,3), 2²⁸ |
| Neuromodulators / pairs | 25 / 300 | C(25,2) |
| Drives / pairs / 3-level states / on-off | 21 / 210 / 10,460,353,203 / 2,097,152 | 3²¹, 2²¹ |
| Latent dimension | 572 | 52 + 52 + 177 + 18 + 17 + 6 + 29 + 67 + 10 + 16 + 128 |
| Forward model weights | 356,928 | 572 × 624 |
| Inverse model weights | 59,488 | 52 × 1,144 |
| Predictive system array values | 2,866,708 | measured |
| Reafference weights (`base` / `extreme`) | 414,336 / 3,729,024 | taxels × 4 × 52 |
| Library policies / sequences of 3 / of 5 | 55 / 166,375 / 503,284,375 | 55^h |
| Allowed mind calls / sequences of 3 / of 6 | 26 / 17,576 / 308,915,776 | 26^k |
| Behaviour channels / descriptors | 33 / 1.1575 × 10³⁷ | product of channel sizes |
| Safety-model weights | 308 | 300 one-hot + 8 continuous |
| Neural-mass synapses (`extreme`) | 589,824 | 12,288 × 48 |
| Inner dynamic variables (`rich` / `extreme` / `max`) | 15,066 / 36,932 / 88,892 | measured |
| Gateway requests per day | 1,000 | 5 × 200 |

### A.2 Skin patches

Taxels per patch at each level (each patch scales by the square of the skin density; all 46 follow the
×9 rule at `extreme`).

| patch | `base` | `rich` | `extreme` | `max` | share of skin | extreme = 9 × base |
|---|---|---|---|---|---|---|
| `upper_arm_l` | 72 | 288 | 648 | 1152 | 3.6 % | yes |
| `deltoid_l` | 50 | 200 | 450 | 800 | 2.5 % | yes |
| `forearm_l` | 72 | 288 | 648 | 1152 | 3.6 % | yes |
| `palm_l` | 24 | 96 | 216 | 384 | 1.2 % | yes |
| `hand_dorsum_l` | 24 | 96 | 216 | 384 | 1.2 % | yes |
| `thumb_l` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `thumb_tip_l` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `index_l` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `index_tip_l` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `fingers_l` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `fingers_tip_l` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `thigh_l` | 112 | 448 | 1008 | 1792 | 5.6 % | yes |
| `shin_l` | 84 | 336 | 756 | 1344 | 4.2 % | yes |
| `knee_l` | 36 | 144 | 324 | 576 | 1.8 % | yes |
| `sole_l` | 28 | 112 | 252 | 448 | 1.4 % | yes |
| `foot_dorsum_l` | 28 | 112 | 252 | 448 | 1.4 % | yes |
| `toes_l` | 12 | 48 | 108 | 192 | 0.6 % | yes |
| `hip_l` | 36 | 144 | 324 | 576 | 1.8 % | yes |
| `upper_arm_r` | 72 | 288 | 648 | 1152 | 3.6 % | yes |
| `deltoid_r` | 50 | 200 | 450 | 800 | 2.5 % | yes |
| `forearm_r` | 72 | 288 | 648 | 1152 | 3.6 % | yes |
| `palm_r` | 24 | 96 | 216 | 384 | 1.2 % | yes |
| `hand_dorsum_r` | 24 | 96 | 216 | 384 | 1.2 % | yes |
| `thumb_r` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `thumb_tip_r` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `index_r` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `index_tip_r` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `fingers_r` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `fingers_tip_r` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `thigh_r` | 112 | 448 | 1008 | 1792 | 5.6 % | yes |
| `shin_r` | 84 | 336 | 756 | 1344 | 4.2 % | yes |
| `knee_r` | 36 | 144 | 324 | 576 | 1.8 % | yes |
| `sole_r` | 28 | 112 | 252 | 448 | 1.4 % | yes |
| `foot_dorsum_r` | 28 | 112 | 252 | 448 | 1.4 % | yes |
| `toes_r` | 12 | 48 | 108 | 192 | 0.6 % | yes |
| `hip_r` | 36 | 144 | 324 | 576 | 1.8 % | yes |
| `chest` | 108 | 432 | 972 | 1728 | 5.4 % | yes |
| `abdomen` | 80 | 320 | 720 | 1280 | 4.0 % | yes |
| `pelvis` | 56 | 224 | 504 | 896 | 2.8 % | yes |
| `neck` | 30 | 120 | 270 | 480 | 1.5 % | yes |
| `scalp` | 112 | 448 | 1008 | 1792 | 5.6 % | yes |
| `face` | 24 | 96 | 216 | 384 | 1.2 % | yes |
| `nose` | 28 | 112 | 252 | 448 | 1.4 % | yes |
| `jaw` | 10 | 40 | 90 | 160 | 0.5 % | yes |
| `tongue` | 20 | 80 | 180 | 320 | 1.0 % | yes |
| `lips` | 8 | 32 | 72 | 128 | 0.4 % | yes |
| **total (46 patches)** | **1,992** | **7,968** | **17,928** | **31,872** | 100 % |  |

### A.3 The 55 library policies

|   |   |   |   |   |
|---|---|---|---|---|
| 1. `hold` | 12. `reach_down_l` | 23. `reach_across_r` | 34. `gaze_down` | 45. `freeze` |
| 2. `stand_tall` | 13. `reach_across_l` | 24. `hand_to_face_r` | 35. `gaze_hand_l` | 46. `brace` |
| 3. `crouch` | 14. `hand_to_face_l` | 25. `hand_to_chest_r` | 36. `gaze_hand_r` | 47. `step_back` |
| 4. `weight_left` | 15. `hand_to_chest_l` | 26. `rub_forearm_r` | 37. `nod` | 48. `walk_in_place` |
| 5. `weight_right` | 16. `rub_forearm_l` | 27. `grasp_r` | 38. `look_up` | 49. `walk_forward` |
| 6. `lean_forward` | 17. `grasp_l` | 28. `open_hand_r` | 39. `shake_head` | 50. `march` |
| 7. `lean_back` | 18. `open_hand_l` | 29. `tap_fingers_r` | 40. `turn_head_left` | 51. `explore_left` |
| 8. `lean_left` | 19. `tap_fingers_l` | 30. `gaze_front` | 41. `turn_head_right` | 52. `explore_right` |
| 9. `lean_right` | 20. `reach_forward_r` | 31. `gaze_left` | 42. `open_mouth` | 53. `explore_hands` |
| 10. `reach_forward_l` | 21. `reach_up_r` | 32. `gaze_right` | 43. `close_mouth` | 54. `press_palm_l` |
| 11. `reach_up_l` | 22. `reach_down_r` | 33. `gaze_up` | 44. `flinch` | 55. `press_palm_r` |

---

## Appendix B. Reproduction and data files

```bash
python tools/build_readme.py                     # rebuild this README from docs/readme_src/ (deterministic, offline)
PYTHONPATH=. PERSON_COMPLEXITY=base python tools/readme_combinatorics.py   # all combinatorial quantities → out/readme_data/combinatorics.json
PYTHONPATH=. PERSON_COMPLEXITY=extreme python tools/readme_runtime_facts.py out/readme_data/runtime_extreme.json   # array inventories
PYTHONPATH=. PERSON_COMPLEXITY=extreme python tools/readme_describe.py out/readme_data/describe_extreme.json      # agent.describe()
python tools/scale_complexity.py --measure base rich extreme max      # sizes and speed per level
PERSON_COMPLEXITY=base python run_sim.py --duration 20 --seed 7 --out out_readme --no-figures    # reference run (§11)
python tools/ultra_bench.py                      # ultra-tier subsystem benchmarks
python tools/readme_check.py <old.md> <new.md>   # numbers, code, links and caveats preserved across a rewrite
```

Data files written by these commands: `out/measure_baseline.txt`, `out/readme_data/measure_max.txt`,
`out/readme_data/{combinatorics,runtime_base,runtime_extreme,describe_base,describe_extreme}.json`,
`out/readme_data/refrun_base.txt`, `out_readme/summary.json`, `out/ultra_bench.json`.

The sections of this README are the files of `docs/readme_src/`; edit those and rebuild, rather than editing
`README.md` directly. The earlier academic rewrite of the previous text is in `docs/readme_work/new/`.
