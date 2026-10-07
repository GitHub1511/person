"""
The ultra tier: a plug-in framework for subsystems far larger than the base person.

The person of ``base`` ... ``max`` (see :mod:`complexity`) is one hand-wired body.  The two
extra levels, ``ultra`` and ``mega``, add many *subsystems* -- spiking sensory afferents, a
spiking cortex, circulation, an immune repertoire, a metabolic network, odour plumes, a
hierarchical behaviour grammar ... -- each in its own module, each independent of the others
except through a shared :class:`Bus`.

Why a framework instead of more hand-wiring
-------------------------------------------
* a domain can be written, tested and benchmarked *without the simulator* (a
  :class:`SyntheticBus` provides every input with the right shape), so many can be built at
  once;
* every subsystem declares its own sizes per level, its update rate, its memory and what it
  reads and writes, so the cost of the whole is known and bounded;
* a wall-clock **governor** can stretch update periods when the machine is busy, so the person
  gets as complex as the computer allows rather than as complex as the largest setting;
* nothing outside this file needs to know a subsystem exists: modules named ``ux_*.py`` in
  this package register themselves and are discovered automatically.

Writing a subsystem
-------------------
::

    from .ultra import Subsystem, register

    @register
    class Afferents(Subsystem):
        name = "somatic_afferents"          # unique, snake_case
        domain = "somatosensation"
        rate_hz = 100.0                      # how often step() is called (sim time)
        summary_dim = 32                     # length of the vector it contributes to the latent
        reads = ("tactile", "taxel_patch")   # bus keys it needs
        writes = ("somatic_afferents.rate",) # bus keys it publishes for others

        def __init__(self, level, seed):
            super().__init__(level, seed)
            self.n = level.pick(ultra=200_000, mega=1_000_000)   # sizes per level
            self.v = np.zeros(self.n, np.float32)

        def step(self, dt, bus):
            ...                              # read bus.get("tactile"), update state
            self.out["affect.pain"] = 0.1    # standing levels pushed to the rest of the person
            self.summary[:] = ...            # fixed-length summary for the latent

        def n_state(self): return self.n
        def mem_bytes(self): return self.v.nbytes

Rules every subsystem follows
-----------------------------
* **numpy only** (no new dependencies), float32 where precision allows, vectorised; no Python
  loops over more than ~10^3 items in ``step``.
* ``step`` must be deterministic given ``seed`` and the bus, finite (no NaN/inf) and must not
  allocate large arrays per call (preallocate).
* ``out`` holds *levels* (they persist until the next step), not events.  Recognised keys:
  ``affect.<AffectInputs field>``, ``drive.<drive name>``, ``intero.<variable name>``,
  ``desire.<behaviour tag>``.  Anything else is recorded but not applied.
* the cost at ``ultra`` and ``mega`` is declared by ``wall_budget_ms_per_sim_s`` and
  ``ram_budget_mb`` and checked by ``tools/ultra_bench.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

LEVELS = ("base", "rich", "extreme", "max", "ultra", "mega")


# ==========================================================================
# Levels
# ==========================================================================
@dataclass(frozen=True)
class Level:
    """Which complexity level a subsystem is being built for."""
    name: str = "ultra"

    @property
    def index(self) -> int:
        return LEVELS.index(self.name) if self.name in LEVELS else LEVELS.index("extreme")

    def pick(self, **kw):
        """``level.pick(extreme=1, ultra=10, mega=100)``: the value for this level.  A level
        without its own entry uses the nearest *lower* entry (or the lowest one given)."""
        order = [n for n in LEVELS if n in kw]
        if not order:
            raise ValueError("pick() needs at least one level")
        best = order[0]
        for n in order:
            if LEVELS.index(n) <= self.index:
                best = n
        return kw[best]

    def at_least(self, name: str) -> bool:
        return self.index >= LEVELS.index(name)


def current_level() -> Level:
    from . import complexity
    return Level(complexity.C.name)


# ==========================================================================
# The bus
# ==========================================================================
class Bus:
    """Named arrays and scalars shared by every subsystem for one tick.

    The agent (see :func:`publish_bus`) writes the standard inputs; subsystems read them and
    publish their own results as ``"<subsystem name>.<key>"``."""

    def __init__(self):
        self.d: dict = {}

    def put(self, key: str, value) -> None:
        self.d[key] = value

    def get(self, key: str, default=None):
        return self.d.get(key, default)

    def vec(self, key: str, n: int, dtype=np.float32) -> np.ndarray:
        """The array under ``key`` flattened and padded / cut to length ``n`` (zeros if absent)."""
        v = self.d.get(key)
        out = np.zeros(n, dtype)
        if v is None:
            return out
        a = np.asarray(v, dtype).ravel()
        m = min(n, a.size)
        out[:m] = a[:m]
        return out

    def scalar(self, key: str, default: float = 0.0) -> float:
        v = self.d.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def __contains__(self, key: str) -> bool:
        return key in self.d


#: The standard inputs, with their shapes at ``extreme``.  The agent publishes these every
#: physics step (arrays are references to the live frame, do not modify them).
BUS_DOC = {
    "t": "simulated time, s", "dt": "physics step, s",
    "tactile": "(n_taxels, 43) receptor channels per taxel (extreme: 17928 x 43)",
    "taxel_patch": "(n_taxels,) int patch index, 0..45",
    "skin_temp": "(n_taxels,) deg C", "proprio": "(52, 13) per joint",
    "vestibular": "(18,)", "visual": "(17,)", "auditory": "(6,)", "olfactory": "(24,)",
    "gustatory": "(5,)", "chemo_summary": "(4,)",
    "ext.spindle": "(4160,)", "ext.vestibular_cells": "(1152,)", "ext.cochlea": "(392,)",
    "ext.olfactory": "(646,)", "ext.gustatory": "(50,)", "ext.retina": "(48384,)",
    "pain": "(4,) mech, heat, cold, total", "itch": "float", "affective_touch": "float",
    "touch_intensity": "float", "contact_count": "float",
    "intero": "(67,) interoceptive variables", "emotions": "(28,)", "neuromod": "(25,)",
    "appraisal": "(10,)", "drives": "(21,)",
    "core_affect": "(10,) valence, arousal, dominance, tension, pleasantness, stress, "
                   "allostatic_load, mood_valence, mood_arousal, mood_energy",
    "inner.clock_hour": "float", "inner.sleep_pressure": "float", "inner.cortisol": "float",
    "inner.glucose": "float", "inner.threat_tone": "float", "inner.rumination": "float",
    "inner.mind_wandering": "float", "inner.familiarity": "float",
    "inner.memory_valence": "float", "inner.muscle_fatigue": "float",
    "inner.muscle_soreness": "float", "inner.gut_discomfort": "float",
    "inner.systemic_inflammation": "float", "inner.interoceptive_surprise": "float",
    "ocular.discomfort": "float", "ocular.blur": "float",
    "body.com_vel": "(3,)", "body.torque_effort": "float", "body.fallen": "0/1",
    "body.root_pos": "(3,)", "body.n_contact": "float", "body.tau": "(52,) joint torques",
    "body.q": "(n_q,) joint angles", "body.qd": "(n_qd,) joint velocities",
    "world.luminance": "float 0..1", "world.ambient_temp": "deg C",
    "world.humidity": "0..1", "world.airflow": "m/s",
    "behavior.name": "str, name of the behaviour being executed (may be '')",
    "world.objects": "list of dicts {name, pos(3), odor(24) or None, taste(5) or None, temp} for every scene object",
}

#: The behaviour tags a ``desire.<tag>`` output can address.
def behaviour_tags() -> tuple[str, ...]:
    try:
        from .behavior_space import TAGS
        return tuple(TAGS)
    except Exception:
        return ()


def publish_bus(ag, bus: Bus) -> None:
    """Fill the standard inputs from a running :class:`EmbodiedHuman`.  Missing things are skipped."""
    g = bus.put
    g("t", float(ag.t))
    g("dt", float(ag.dt))
    f = ag.frame
    if f is not None:
        g("tactile", f.tactile)
        try:
            from . import skin
            g("taxel_patch", skin.TAXEL_PATCH_IDX)
        except Exception:
            pass
        g("skin_temp", f.skin_temperature)
        g("proprio", f.proprio)
        g("vestibular", f.vestibular)
        g("visual", f.visual)
        g("auditory", f.auditory)
        g("olfactory", f.olfactory)
        g("gustatory", f.gustatory)
        g("chemo_summary", f.chemo_summary)
        for k, v in f.ext.items():
            g("ext." + k, v)
        g("pain", np.array([f.pain_mech, f.pain_heat, f.pain_cold, f.pain_total]))
        g("itch", f.itch_total)
        g("affective_touch", f.affective_touch)
        g("touch_intensity", f.touch_intensity)
        g("contact_count", f.contact_count)
    g("intero", ag.interoception.s)
    a = ag.affect_frame
    if a is not None:
        g("emotions", a.emotions)
        g("neuromod", a.neuromodulators)
        g("appraisal", a.appraisal)
        g("core_affect", np.array([a.valence, a.arousal, a.dominance, a.tension, a.pleasantness,
                                   a.stress, a.allostatic_load, a.mood_valence, a.mood_arousal,
                                   a.mood_energy]))
    d = ag.drive_frame
    if d is not None:
        g("drives", d.level)
    inner = getattr(ag, "inner", None)
    if inner is not None:
        o = inner.out
        for k in ("clock_hour", "sleep_pressure", "cortisol", "glucose", "threat_tone", "rumination",
                  "mind_wandering", "familiarity", "memory_valence", "muscle_fatigue",
                  "muscle_soreness", "gut_discomfort", "systemic_inflammation",
                  "interoceptive_surprise"):
            g("inner." + k, getattr(o, k, 0.0))
    oc = getattr(ag, "ocular", None)
    if oc is not None:
        g("ocular.discomfort", oc.out.discomfort)
        g("ocular.blur", oc.out.blur)
    s = ag.state
    if s is not None:
        g("body.com_vel", s.com_vel)
        g("body.torque_effort", s.torque_effort)
        g("body.fallen", float(s.fallen))
        g("body.root_pos", s.root_pos)
        g("body.n_contact", float(s.n_contact))
        g("body.tau", s.tau)
        g("body.q", s.q)
        g("body.qd", s.qd)
    g("world.luminance", float(ag.luminance))
    g("world.ambient_temp", float(ag.cfg.intero.__dict__.get("_ambient", 22.0)))
    g("world.humidity", float(ag.ambient_humidity))
    g("world.airflow", float(ag.ambient_airflow))
    g("behavior.name", "")
    objs = []
    for o in getattr(ag.meta, "objects", []):
        a = ag.meta.object_qpos_addr.get(o.name)
        if a is None:
            continue
        objs.append({"name": o.name, "pos": np.array(ag.data.qpos[a:a + 3]), "odor": o.odor,
                     "taste": o.taste, "temp": float(getattr(o, "temperature", 22.0))})
    g("world.objects", objs)


# ==========================================================================
# Subsystem
# ==========================================================================
class Subsystem:
    """Base class.  See the module docstring."""
    name: str = "unnamed"
    domain: str = "misc"
    rate_hz: float = 10.0
    summary_dim: int = 16
    reads: tuple = ()
    writes: tuple = ()
    #: declared budgets at the two ultra levels (checked by tools/ultra_bench.py)
    wall_budget_ms_per_sim_s: dict = {"ultra": 250.0, "mega": 1500.0}
    ram_budget_mb: dict = {"ultra": 150.0, "mega": 600.0}
    #: first level at which this subsystem exists
    min_level: str = "ultra"

    def __init__(self, level: Level, seed: int):
        self.level = level
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.out: dict[str, float] = {}
        self.summary = np.zeros(self.summary_dim, np.float32)

    # -- to override ---------------------------------------------------------
    def step(self, dt: float, bus: Bus) -> None:
        raise NotImplementedError

    def n_state(self) -> int:
        return 0

    def mem_bytes(self) -> int:
        return 0

    def counts(self) -> dict:
        return {}

    def reset(self) -> None:
        pass


REGISTRY: dict[str, type] = {}


def register(cls):
    """Class decorator."""
    if not cls.name or cls.name == "unnamed":
        raise ValueError(f"{cls} needs a name")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"subsystem {cls.name!r} registered twice")
    REGISTRY[cls.name] = cls
    return cls


_LOADED = False


def load_all() -> list[str]:
    """Import every ``ux_*`` module of this package so its subsystems register."""
    global _LOADED
    errors = []
    pkg = importlib.import_module(__package__)
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("ux_"):
            try:
                importlib.import_module(f"{__package__}.{m.name}")
            except Exception as e:                      # a broken domain must not break the rest
                errors.append(f"{m.name}: {type(e).__name__}: {e}")
    _LOADED = True
    return errors


# ==========================================================================
# The world of subsystems
# ==========================================================================
@dataclass
class UltraFrame:
    summary: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    push: dict = field(default_factory=dict)        # summed standing outputs, by key
    unknown: dict = field(default_factory=dict)     # outputs nobody applies (still recorded)


class UltraWorld:
    """Owns the subsystems of one person, steps each at its own rate, and accounts for cost."""

    APPLIED_PREFIXES = ("affect.", "drive.", "intero.", "desire.")

    def __init__(self, level: Level | None = None, seed: int = 0, only=None, exclude=(),
                 governor: bool = False, budget_s_per_sim_s: float | None = None):
        self.level = level or current_level()
        self.seed = int(seed)
        if not _LOADED:
            self.load_errors = load_all()
        else:
            self.load_errors = []
        self.subs: list[Subsystem] = []
        for name, cls in sorted(REGISTRY.items(), key=lambda kv: (kv[1].domain, kv[0])):
            if only and name not in only:
                continue
            if cls.domain == "example" and not only:        # the template, not part of a person
                continue
            if name in exclude or not self.level.at_least(cls.min_level):
                continue
            sub_seed = self.seed * 1_000_003 + zlib.crc32(name.encode())
            self.subs.append(cls(self.level, sub_seed & 0x7FFFFFFF))
        self.acc = {s.name: 0.0 for s in self.subs}
        self.period = {s.name: 1.0 / s.rate_hz for s in self.subs}       # may be stretched
        self.base_period = dict(self.period)
        self.wall = {s.name: 0.0 for s in self.subs}                      # seconds, total
        self.calls = {s.name: 0 for s in self.subs}
        self.t = 0.0
        self.governor = governor
        self.budget = budget_s_per_sim_s if budget_s_per_sim_s is not None else 6.0
        self._gov_wall, self._gov_t0 = 0.0, 0.0
        self.frame = UltraFrame(summary=np.zeros(self.summary_dim, np.float32))

    # -- sizes -----------------------------------------------------------------
    @property
    def summary_dim(self) -> int:
        return int(sum(s.summary_dim for s in self.subs))

    @property
    def n_state(self) -> int:
        return int(sum(s.n_state() for s in self.subs))

    def mem_bytes(self) -> int:
        return int(sum(s.mem_bytes() for s in self.subs))

    def counts(self) -> dict:
        out = {"subsystems": len(self.subs), "dynamic_state_variables": self.n_state,
               "memory_mb": round(self.mem_bytes() / 1e6, 1)}
        for s in self.subs:
            for k, v in s.counts().items():
                out[f"{s.name}.{k}"] = v
        return out

    # -- stepping ----------------------------------------------------------------
    def step(self, dt: float, bus: Bus) -> None:
        self.t += dt
        for s in self.subs:
            n = s.name
            self.acc[n] += dt
            if self.acc[n] >= self.period[n]:
                h = self.acc[n]
                self.acc[n] = 0.0
                t0 = time.perf_counter()
                s.step(h, bus)
                self.wall[n] += time.perf_counter() - t0
                self.calls[n] += 1
        if self.governor and self.t - self._gov_t0 >= 1.0:
            self._govern()

    def _govern(self) -> None:
        total = sum(self.wall.values())
        used = (total - self._gov_wall) / max(self.t - self._gov_t0, 1e-9)
        self._gov_wall, self._gov_t0 = total, self.t
        if used > self.budget:
            hog = max(self.subs, key=lambda s: self.wall[s.name] / max(self.calls[s.name], 1) * s.rate_hz)
            self.period[hog.name] = min(self.period[hog.name] * 1.5, self.base_period[hog.name] * 8.0)
        elif used < 0.5 * self.budget:
            for s in self.subs:
                self.period[s.name] = max(self.period[s.name] / 1.2, self.base_period[s.name])

    def collect(self) -> UltraFrame:
        """Sum the standing outputs and concatenate the summaries."""
        push: dict[str, float] = {}
        unknown: dict[str, float] = {}
        for s in self.subs:
            for k, v in s.out.items():
                v = float(v) if np.isfinite(v) else 0.0
                (push if k.startswith(self.APPLIED_PREFIXES) else unknown)[k] = \
                    (push if k.startswith(self.APPLIED_PREFIXES) else unknown).get(k, 0.0) + v
        summ = np.concatenate([np.nan_to_num(s.summary) for s in self.subs]) if self.subs \
            else np.zeros(0, np.float32)
        self.frame = UltraFrame(summary=summ.astype(np.float32, copy=False), push=push, unknown=unknown)
        return self.frame

    def profile(self) -> list[dict]:
        """Wall time per subsystem per simulated second (ms), state variables, memory."""
        rows = []
        for s in self.subs:
            rows.append({"name": s.name, "domain": s.domain, "rate_hz": s.rate_hz,
                         "period_s": round(self.period[s.name], 4),
                         "ms_per_sim_s": 1e3 * self.wall[s.name] / max(self.t, 1e-9),
                         "n_state": s.n_state(), "mem_mb": s.mem_bytes() / 1e6,
                         "summary_dim": s.summary_dim})
        return rows

    def reset(self) -> None:
        for s in self.subs:
            s.reset()
            s.out.clear()
            s.summary[:] = 0
        self.t = 0.0
        for k in self.acc:
            self.acc[k] = 0.0


# ==========================================================================
# Synthetic inputs, so a subsystem can be built and benchmarked without the simulator
# ==========================================================================
class SyntheticBus(Bus):
    """A bus whose inputs have the right shapes and smoothly varying, plausible values.

    ``advance(dt)`` moves time forward; every array is a slowly drifting random field plus
    occasional "events" (a touch, a startle, a meal), so a subsystem sees non-trivial input."""

    def __init__(self, level: Level | None = None, seed: int = 0, n_taxels: int | None = None):
        super().__init__()
        self.level = level or Level("ultra")
        self.rng = np.random.default_rng(seed)
        self.n_taxels = n_taxels or self.level.pick(extreme=17928, max=31872)
        n = self.n_taxels
        r = self.rng
        self.d.update({
            "t": 0.0, "dt": 0.001,
            "tactile": np.zeros((n, 43), np.float32),
            "taxel_patch": np.sort(r.integers(0, 46, n)).astype(np.int32),
            "skin_temp": np.full(n, 33.0, np.float32), "proprio": np.zeros((52, 13), np.float32),
            "vestibular": np.zeros(18, np.float32), "visual": np.zeros(17, np.float32),
            "auditory": np.zeros(6, np.float32), "olfactory": np.zeros(24, np.float32),
            "gustatory": np.zeros(5, np.float32), "chemo_summary": np.zeros(4, np.float32),
            "ext.spindle": np.zeros(4160, np.float32), "ext.vestibular_cells": np.zeros(1152, np.float32),
            "ext.cochlea": np.zeros(392, np.float32), "ext.olfactory": np.zeros(646, np.float32),
            "ext.gustatory": np.zeros(50, np.float32), "ext.retina": np.zeros(48384, np.float32),
            "pain": np.zeros(4, np.float32), "itch": 0.0, "affective_touch": 0.0,
            "touch_intensity": 0.0, "contact_count": 0.0,
            "intero": np.zeros(67, np.float32), "emotions": np.zeros(28, np.float32),
            "neuromod": np.zeros(25, np.float32), "appraisal": np.zeros(10, np.float32),
            "drives": np.zeros(21, np.float32), "core_affect": np.zeros(10, np.float32),
            "body.com_vel": np.zeros(3), "body.torque_effort": 0.1, "body.fallen": 0.0,
            "body.root_pos": np.array([0.0, 0.0, 0.84]), "body.n_contact": 2.0,
            "body.tau": np.zeros(52), "body.q": np.zeros(94), "body.qd": np.zeros(88),
            "world.luminance": 0.7, "world.ambient_temp": 22.0, "world.humidity": 0.45,
            "world.airflow": 0.05, "behavior.name": "",
            "world.objects": [
                {"name": "apple", "pos": np.array([0.3, -0.6, 0.8]), "odor": r.random(24), "taste": r.random(5), "temp": 21.0},
                {"name": "mug", "pos": np.array([-0.2, -0.7, 0.8]), "odor": r.random(24), "taste": None, "temp": 45.0},
                {"name": "ball", "pos": np.array([1.0, -1.5, 0.1]), "odor": None, "taste": None, "temp": 22.0},
                {"name": "stone", "pos": np.array([-1.2, -0.4, 0.1]), "odor": None, "taste": None, "temp": 18.0},
            ],
        })
        for k in ("clock_hour", "sleep_pressure", "cortisol", "glucose", "threat_tone", "rumination",
                  "mind_wandering", "familiarity", "memory_valence", "muscle_fatigue",
                  "muscle_soreness", "gut_discomfort", "systemic_inflammation",
                  "interoceptive_surprise"):
            self.d["inner." + k] = 0.0
        self.d["inner.clock_hour"] = 10.0
        self.d["inner.glucose"] = 5.2
        self.d["ocular.discomfort"] = 0.0
        self.d["ocular.blur"] = 0.0
        self._phase = r.uniform(0, 6.28, 64)
        self.t = 0.0

    def advance(self, dt: float) -> None:
        self.t += dt
        r, d = self.rng, self.d
        d["t"], d["dt"] = self.t, dt
        w = 0.5 + 0.5 * np.sin(0.7 * self.t + self._phase[0])
        # tactile: sparse touches that move around
        T = d["tactile"]
        T *= 0.9
        k = max(1, self.n_taxels // 400)
        idx = r.integers(0, self.n_taxels, k)
        T[idx, :8] += r.random((k, 8), dtype=np.float32) * (0.5 + w)
        d["skin_temp"][:] = 33.0 + 1.5 * np.sin(0.05 * self.t + self._phase[1])
        d["proprio"][:] = 0.9 * d["proprio"] + 0.1 * r.standard_normal(d["proprio"].shape).astype(np.float32)
        for key, n in (("vestibular", 18), ("visual", 17), ("auditory", 6), ("olfactory", 24)):
            d[key][:] = 0.95 * d[key] + 0.05 * r.standard_normal(n).astype(np.float32)
        for key in ("ext.spindle", "ext.vestibular_cells", "ext.cochlea", "ext.olfactory",
                    "ext.gustatory", "ext.retina"):
            a = d[key]
            a[:] = 0.97 * a + 0.03 * r.random(a.size, dtype=np.float32)
        d["pain"][:] = np.clip(0.9 * d["pain"] + 0.02 * r.standard_normal(4), 0, 1)
        d["itch"] = float(np.clip(0.5 + 0.5 * np.sin(0.02 * self.t + self._phase[2]), 0, 1) * 0.1)
        d["intero"][:] = 0.99 * d["intero"] + 0.01 * (1.0 + 0.1 * r.standard_normal(67))
        for key, n in (("emotions", 28), ("neuromod", 25), ("appraisal", 10), ("drives", 21),
                       ("core_affect", 10)):
            d[key][:] = np.clip(0.98 * d[key] + 0.02 * r.random(n), 0, 1)
        d["inner.clock_hour"] = (10.0 + self.t / 3600.0) % 24.0
        d["inner.sleep_pressure"] = float(np.clip(0.3 + 0.02 * self.t / 60.0, 0, 1))
        d["inner.cortisol"] = 250.0 + 40.0 * np.sin(2 * np.pi * d["inner.clock_hour"] / 24.0)
        d["body.com_vel"][:] = 0.02 * r.standard_normal(3)
        d["body.tau"][:] = 5.0 * r.standard_normal(52)
        d["world.luminance"] = float(np.clip(0.6 + 0.3 * np.sin(2 * np.pi * d["inner.clock_hour"] / 24.0), 0.05, 1))


def make_bus(source=None, level: Level | None = None, seed: int = 0) -> Bus:
    """A bus for ``source`` (an agent, or None for synthetic data)."""
    return SyntheticBus(level, seed)


# ==========================================================================
# Cost accounting (used by tools/ultra_bench.py)
# ==========================================================================
def bench(sub_cls, level: Level, sim_seconds: float = 5.0, seed: int = 0, dt: float = 0.001) -> dict:
    """Build one subsystem, run it for ``sim_seconds`` on synthetic input, return cost + health."""
    import tracemalloc
    bus = SyntheticBus(level, seed)
    t0 = time.perf_counter()
    tracemalloc.start()
    w = UltraWorld(level, seed, only={sub_cls.name})
    build_s = time.perf_counter() - t0
    _, peak_build = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert w.subs and isinstance(w.subs[0], sub_cls), "subsystem is not registered at this level"
    n = int(sim_seconds / dt)
    t0 = time.perf_counter()
    bad = 0
    for _ in range(n):
        bus.advance(dt)
        w.step(dt, bus)
    wall = time.perf_counter() - t0
    fr = w.collect()
    s = w.subs[0]
    finite = bool(np.all(np.isfinite(fr.summary))) and all(np.isfinite(v) for v in fr.push.values()) \
        and all(np.isfinite(v) for v in fr.unknown.values())
    return {"name": sub_cls.name, "level": level.name, "n_state": s.n_state(),
            "mem_mb": s.mem_bytes() / 1e6, "build_s": build_s, "build_peak_mb": peak_build / 1e6,
            "ms_per_sim_s": 1e3 * w.wall[sub_cls.name] / sim_seconds,
            "total_ms_per_sim_s": 1e3 * wall / sim_seconds, "calls": w.calls[sub_cls.name],
            "finite": finite, "summary_norm": float(np.linalg.norm(fr.summary)),
            "outputs": sorted(set(fr.push) | set(fr.unknown)),
            "budget_ms": sub_cls.wall_budget_ms_per_sim_s.get(level.name),
            "budget_mb": sub_cls.ram_budget_mb.get(level.name)}
