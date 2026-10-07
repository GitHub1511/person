# The project and its thesis

`person` is a simulated human in MuJoCo (Python, numpy, MuJoCo, PIL only). It is **a person, not a
policy**: there is no task and no external reward. Everything it wants comes from inside its own body.

The thesis is that *richness and coupling of internal and external signal* is what makes a simulated body
feel like a body, and that behaviour should *emerge from that internal state* rather than from a list of
scripts. Concretely the project has these layers (all in `embodied_human/`):

| layer | modules | what it is |
|---|---|---|
| body + physics | `skeleton.py`, `build_model.py`, `skin.py` | 52-actuator humanoid, visible eyes with lids, taxels on a fine skin |
| senses | `receptors.py`, `senses_ext.py`, `afferents.py` | dense skin bank (many channels per taxel), muscle spindle / tendon populations, vestibular hair cells, two-ear cochlea, olfactory and taste receptor banks, retina bank; conduction delays, adaptation, reafference |
| eyes | `ocular.py` | tear film, corneal nerves, a dryness sensation that drives blinking and spills into vision, affect, pain, drives, behaviour |
| internal world | `inner_world.py`, `inner_organs.py`, `inner_brain.py`, `interoception.py`, `drives.py`, `affect.py` | organs with compartments (vascular, lungs, kidney, liver, gut + microbiome, ~3,000 motor units, skin thermoregulation, immune network, a ~480-channel chemistry panel), a circadian clock, a ~12,000-unit neural mass with neuromodulated populations, episodic memory, conditioning, an interoceptive prediction model; emotions, neuromodulators, drives |
| behaviour | `behavior_space.py`, `behavior_exec.py`, `body_learning.py`, `skills.py`, `locomotion.py`, `wbc.py`, `motor.py` | a generative behaviour space (33 channels, ~10^37 descriptors, ~734,000 configurations per arm), expected-free-energy selection, a learned body-safety model, skills (reach, grasp, touch self...), whole-body control |
| mind | `mind.py`, `world.py` | an optional language model (Absolute Zero Reasoner) behind the body via a whitelisted action API |
| learning | `tools/train_body.py` | many parallel bodies babble behaviours and learn which are safe |

What "better" means, in order of priority:
1. **Never break what works.** The person must still stand and not fall; `base` complexity must still run.
2. **More signal that matters**: new variables/populations are only worth adding if something *listens*
   to them (they change affect, drives, behaviour, vision, pain, the mind's percept, or the latent the
   active-inference layer sees). Disconnected state is decoration.
3. **More varied, more individual behaviour** that follows from internal state (different needs, moods,
   histories produce different behaviour), with a measured diversity (`BehaviorStats`).
4. **Learning from experience over a lifetime** and across many parallel copies (body safety, body
   schema, conditioning, memory), with a measured before/after.
5. **Honesty**: everything is a toy model; say so in docstrings; no claim without a measurement. The README
   (`README.md`, section 17 onwards) must stay truthful if you change numbers it quotes.
