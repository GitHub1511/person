"""
How complicated the person is.

Everything that scales -- how finely the skin is quantised, how many receptor
types there are, how many compartments the organs have, how large the neural
mass is -- is read from one :class:`Complexity` profile, so the whole body can
be dialled up or down in one place.

The profile is chosen, in this order, by

1. the ``PERSON_COMPLEXITY`` environment variable (``base`` / ``rich`` /
   ``extreme`` / ``max``, or a path to a JSON file of overrides),
2. ``embodied_human/complexity.json`` (written by ``tools/scale_complexity.py``),
3. the default, ``extreme``.

It is read **once, at import time**, because the skin geometry (and therefore
the MuJoCo model) is generated from it.  Use ``tools/scale_complexity.py`` to
change it persistently and to print what each level costs.

``base`` is the original body (1,872 taxels, 49,588 sensory scalars, 67
interoceptive variables).  The others keep every original mechanism and add
to it; none of them are claims about how many receptors a real mammal has
(a mouse has orders of magnitude more).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FILE = _HERE / "complexity.json"


@dataclass
class Complexity:
    name: str = "extreme"

    # ---- somatosensation ---------------------------------------------------
    skin_density: float = 3.0          # linear taxel density (x^2 taxels)
    extended_tactile: bool = True      # the extra per-taxel receptor channels
    # ---- proprioception ----------------------------------------------------
    spindles_per_muscle: int = 16      # Ia + II fibres, per muscle (2 per joint)
    gto_per_muscle: int = 8
    # ---- vestibular --------------------------------------------------------
    canal_afferents: int = 64          # per canal (3 canals x 2 ears)
    otolith_hair_cells: int = 192      # per otolith organ (2 organs x 2 ears)
    # ---- audition ----------------------------------------------------------
    cochlear_bands: int = 192          # inner hair cell channels per ear
    # ---- chemical senses ---------------------------------------------------
    olfactory_receptors: int = 320     # receptor types (glomeruli)
    taste_cell_types: int = 48
    stimuli: bool = False
    # ---- vision ------------------------------------------------------------
    retina_w: int = 96
    retina_h: int = 72
    # ---- eyes --------------------------------------------------------------
    corneal_units: int = 192           # sensory nerve terminals per eye
    tear_film_sectors: int = 96        # film thickness samples per eye
    # ---- the internal world --------------------------------------------------
    neural_units: int = 12288          # rate units in the neural mass
    neural_indegree: int = 48
    motor_units_per_muscle: int = 32
    alveoli: int = 192
    nephron_groups: int = 192
    liver_zones: int = 96
    vascular_beds: int = 96
    gut_segments: int = 16
    microbiome_taxa: int = 64
    analytes: int = 480                # blood / tissue chemistry panel
    cytokines: int = 64
    immune_populations: int = 32
    circadian_oscillators: int = 192
    episodic_capacity: int = 4096
    episodic_dim: int = 96
    # ---- speed knobs -------------------------------------------------------
    tactile_hz: float = 100.0          # how often the dense skin is transduced
    inner_world: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


PRESETS: dict[str, Complexity] = {
    "base": Complexity(
        name="base", skin_density=1.0, extended_tactile=False,
        spindles_per_muscle=1, gto_per_muscle=1, canal_afferents=1,
        otolith_hair_cells=1, cochlear_bands=24, olfactory_receptors=24,
        taste_cell_types=5, retina_w=48, retina_h=36, corneal_units=16,
        tear_film_sectors=16, neural_units=512, neural_indegree=16,
        motor_units_per_muscle=4, alveoli=16, nephron_groups=16, liver_zones=8,
        vascular_beds=12, gut_segments=6, microbiome_taxa=8, analytes=48,
        cytokines=8, immune_populations=8, circadian_oscillators=16,
        episodic_capacity=256, episodic_dim=32, tactile_hz=250.0,
        inner_world=False),
    "rich": Complexity(
        name="rich", skin_density=2.0, spindles_per_muscle=8, gto_per_muscle=4,
        canal_afferents=24, otolith_hair_cells=64, cochlear_bands=96,
        olfactory_receptors=160, taste_cell_types=24, retina_w=72, retina_h=54,
        corneal_units=96, tear_film_sectors=48, neural_units=4096,
        neural_indegree=32, motor_units_per_muscle=16, alveoli=96,
        nephron_groups=96, liver_zones=48, vascular_beds=48, gut_segments=12,
        microbiome_taxa=32, analytes=240, cytokines=32, immune_populations=24,
        circadian_oscillators=96, episodic_capacity=2048, episodic_dim=64),
    "extreme": Complexity(name="extreme"),
    "max": Complexity(
        name="max", skin_density=4.0, spindles_per_muscle=32, gto_per_muscle=16,
        canal_afferents=128, otolith_hair_cells=384, cochlear_bands=384,
        olfactory_receptors=640, taste_cell_types=96, retina_w=128, retina_h=96,
        corneal_units=384, tear_film_sectors=192, neural_units=32768,
        neural_indegree=64, motor_units_per_muscle=64, alveoli=384,
        nephron_groups=384, liver_zones=192, vascular_beds=192, gut_segments=24,
        microbiome_taxa=128, analytes=960, cytokines=128, immune_populations=48,
        circadian_oscillators=384, episodic_capacity=8192, episodic_dim=128,
        tactile_hz=50.0),
}


def _from_overrides(base: Complexity, data: dict) -> Complexity:
    valid = {f.name for f in fields(Complexity)}
    kw = {k: v for k, v in data.items() if k in valid}
    out = Complexity(**{**asdict(base), **kw})
    return out


def load() -> Complexity:
    spec = os.environ.get("PERSON_COMPLEXITY", "").strip()
    if spec:
        if spec in PRESETS:
            return PRESETS[spec]
        p = Path(spec)
        if p.exists():
            return _from_overrides(PRESETS["extreme"], json.loads(p.read_text()))
        raise ValueError(f"PERSON_COMPLEXITY={spec!r} is neither a preset "
                         f"({', '.join(PRESETS)}) nor a JSON file")
    if _FILE.exists():
        try:
            d = json.loads(_FILE.read_text())
            base = PRESETS.get(d.get("name", "extreme"), PRESETS["extreme"])
            return _from_overrides(base, d)
        except (OSError, ValueError):
            pass
    return PRESETS["extreme"]


C: Complexity = load()


def save(c: Complexity) -> Path:
    _FILE.write_text(c.to_json())
    return _FILE
