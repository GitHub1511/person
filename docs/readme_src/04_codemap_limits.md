## 12. Code map

### 12.1 Modules of `embodied_human/`

The table below is generated from the source tree by `tools/build_readme.py` (line, class and function counts
come from `ast`; the purpose is the first line of each module docstring).

{{CODEMAP}}

### 12.2 Entry points, tools and diagnostics

Root scripts: `run_sim.py` (episode, figures, `--describe`), `run_mind.py` (the person with a language-model
mind, §16.5), `setup_azr.py` (how to obtain AZR), `smoke.py` (build and short run).

{{TOOLS}}

The experiments that produced §9 and §16 are the `diag_*.py` scripts in the repository root; each is runnable
and prints the measurement it was written for. In addition there are six one-off repair scripts (`fix_lunge*.py`)
left by the fall-recovery work (§16.7).

{{DIAGS}}

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
