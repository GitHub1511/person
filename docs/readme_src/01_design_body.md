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
