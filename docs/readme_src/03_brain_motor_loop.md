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

The forward model's prediction error falls steeply within the first seconds; the agent starts from the prior
"nothing changes unless I act" and learns the sensory consequences of its own commands within about ten
cognitive ticks (1 s).

---
