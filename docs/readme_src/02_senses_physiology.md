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
**1,983,036** pairs at `base` and 160,709,... pairs at `extreme`, which is why contact is attributed to
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
