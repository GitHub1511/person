"""
The organs: compartmental physiology that the original 67-variable interoceptive
model summarised in a handful of numbers.

Each class here is a population of coupled compartments, vectorised so that tens
of thousands of state variables cost a few milliseconds per update:

=====================  ===================================================  ==========
system                 compartments                                          states
=====================  ===================================================  ==========
VascularBeds           perfusion beds (skin patches, muscle groups, organs)  3 x N
Lungs                  alveolar units with their own ventilation/perfusion   4 x N
Kidney                 nephron groups: filtration, reabsorption, ADH         3 x N
Liver                  lobule zones: glycogen, gluconeogenesis, lipid        3 x N
GutMicrobiome          gut segments x nutrient pools, enteric neurons,       ~700
                       microbial taxa with metabolite exchange
MuscleBank             motor units of every muscle (size principle, fibre    3 x 3300
                       type, fatigue, glycogen, soreness)
SkinThermo             sweat, blood flow and temperature per skin patch      4 x 46
Immune                 cell populations and cytokines with signed            ~100 + 3 x 46
                       interactions; tissue damage and healing per patch
Chemistry              a blood / tissue chemistry panel                      ~480
=====================  ===================================================  ==========

All of it is a *toy physiology*: the right qualitative couplings (exertion
recruits motor units and fatigues them, heat opens skin beds and starts sweating,
tissue damage drives inflammation which sensitises the skin, a meal raises glucose
and insulin and feeds the microbiome) with plausible orders of magnitude, but no
claim to be a validated model of any organ.  Names in the chemistry panel are real;
only ~60 of its ~480 channels have mechanistic couplings, the rest follow
group-level factors (see ``Chemistry``).
"""

from __future__ import annotations

import numpy as np

from ._fast import fclip
from .complexity import C as COMPLEXITY


def _sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ==========================================================================
class VascularBeds:
    """Perfusion beds with sympathetic tone, local metabolic regulation and a
    baroreflex.  The first ``n_skin`` beds are the skin patches."""

    def __init__(self, n_skin: int, seed: int):
        C = COMPLEXITY
        self.n = max(int(C.vascular_beds), n_skin + 8)
        self.n_skin = n_skin
        rng = np.random.default_rng(seed)
        self.resist0 = rng.lognormal(0.0, 0.25, self.n)
        self.compl = rng.uniform(0.5, 2.0, self.n)
        self.demand = np.ones(self.n)                 # metabolic demand (1 = rest)
        self.tone = np.full(self.n, 0.5)              # 0 dilated .. 1 constricted
        self.flow = np.ones(self.n)
        self.o2_extr = np.full(self.n, 0.25)
        self.tau_tone = rng.uniform(4.0, 40.0, self.n)
        self.map_ = 90.0
        self.size = 3 * self.n

    def update(self, dt, sympathetic, core_temp, heat_load, local_demand, hr, sv):
        """local_demand: per-bed metabolic demand (n,)."""
        self.demand += 0.25 * (local_demand - self.demand)
        # skin beds respond to temperature (vasodilation when hot, constriction
        # when cold), the rest to sympathetic drive and local demand
        thermal = np.clip((core_temp - 36.8) * 1.6 + 0.5 * heat_load, -1.0, 1.5)
        target = np.full(self.n, 0.5) + 0.35 * sympathetic - 0.55 * (self.demand - 1.0)
        target[:self.n_skin] += -0.55 * thermal - 0.15 * (self.demand[:self.n_skin] - 1.0)
        target = np.clip(target, 0.02, 1.0)
        self.tone += (dt / self.tau_tone) * (target - self.tone)
        resist = self.resist0 * (0.35 + 1.6 * self.tone ** 1.5)
        co = hr * sv / 1000.0                          # L/min
        cond = 1.0 / resist
        self.map_ = 90.0 + 0.9 * (co - 5.0) * 6.0 + 18.0 * (np.mean(self.tone) - 0.5)
        self.flow = co * cond / cond.sum() * self.n / 5.0
        self.o2_extr = np.clip(0.2 + 0.25 * (self.demand - 1.0) / np.maximum(self.flow, 0.3), 0.1, 0.9)

    def state(self):
        return np.concatenate([self.tone, self.flow, self.o2_extr])


# ==========================================================================
class Lungs:
    """Alveolar units with their own ventilation / perfusion ratio."""

    def __init__(self, seed: int):
        self.n = int(COMPLEXITY.alveoli)
        rng = np.random.default_rng(seed)
        self.vq = np.exp(rng.normal(0.0, 0.35, self.n))       # V/Q ratio per unit
        self.po2 = np.full(self.n, 100.0)
        self.pco2 = np.full(self.n, 40.0)
        self.vent = np.ones(self.n)
        self.perf = np.ones(self.n)
        self.sao2 = 98.0
        self.paco2 = 40.0
        self.size = 4 * self.n

    def update(self, dt, minute_vent, cardiac_output, metabolic_rate):
        vent = minute_vent * self.vq / self.vq.sum() * self.n          # L/min per unit scale
        perf = cardiac_output / 5.0 * np.ones(self.n)
        vq = vent / np.maximum(perf, 1e-3) / 0.9
        # alveolar gas from V/Q: high V/Q -> PO2 up, PCO2 down
        po2_t = 40.0 + 100.0 * vq / (vq + 0.45)
        pco2_t = 46.0 - 8.0 * vq / (vq + 0.45) + 0.012 * (metabolic_rate - 85.0)
        self.po2 += (dt / 2.0) * (po2_t - self.po2)
        self.pco2 += (dt / 2.0) * (pco2_t - self.pco2)
        self.vent, self.perf = vent, perf
        shunt_mix = np.average(self.po2, weights=perf)
        self.sao2 = float(np.clip(100.0 * (shunt_mix ** 2.7) / (shunt_mix ** 2.7 + 26.0 ** 2.7), 60, 100))
        self.paco2 = float(np.average(self.pco2, weights=vent + 1e-6))

    def state(self):
        return np.concatenate([self.po2, self.pco2, self.vent, self.perf])


# ==========================================================================
class Kidney:
    """Nephron groups: filtration, sodium / water reabsorption under ADH."""

    def __init__(self, seed: int):
        self.n = int(COMPLEXITY.nephron_groups)
        rng = np.random.default_rng(seed)
        self.gfr = rng.uniform(0.7, 1.3, self.n)
        self.reab_na = np.full(self.n, 0.99)
        self.reab_h2o = np.full(self.n, 0.80)
        self.urine = 1.0
        self.adh = 0.3
        self.osm = 290.0
        self.size = 3 * self.n

    def update(self, dt, osm, hydration, map_, aldosterone, sympathetic):
        self.osm = osm
        self.adh += (dt / 60.0) * ((0.3 + 2.5 * max(osm - 292.0, 0) / 10.0
                                    + 1.5 * max(1.0 - hydration, 0)) - self.adh)
        filt = self.gfr * np.clip(map_ / 90.0, 0.2, 1.6) * (1.0 - 0.2 * sympathetic)
        self.reab_h2o += (dt / 30.0) * ((0.75 + 0.2 * np.tanh(self.adh - 0.6)) - self.reab_h2o)
        self.reab_na += (dt / 120.0) * ((0.985 + 0.012 * np.tanh(aldosterone)) - self.reab_na)
        self.urine = float(np.sum(filt * (1.0 - self.reab_h2o)) / self.n * 8.0)

    def state(self):
        return np.concatenate([self.gfr, self.reab_na, self.reab_h2o])


# ==========================================================================
class Liver:
    def __init__(self, seed: int):
        self.n = int(COMPLEXITY.liver_zones)
        rng = np.random.default_rng(seed)
        self.glycogen = np.full(self.n, 0.7)
        self.gng = np.full(self.n, 0.2)          # gluconeogenesis
        self.lipid = np.full(self.n, 0.3)
        self.zone = np.linspace(0, 1, self.n)    # periportal -> pericentral
        self.glucose_out = 0.0
        self.size = 3 * self.n

    def update(self, dt, glucose, insulin, glucagon, cortisol, fed):
        hi = np.tanh(max(glucose - 5.5, 0.0) * 0.6 + 0.8 * insulin)
        lo = np.tanh(max(4.8 - glucose, 0.0) * 1.2 + 0.8 * glucagon)
        self.glycogen = np.clip(self.glycogen + dt * (0.004 * hi * (1 - self.zone * 0.4)
                                                      - 0.004 * lo), 0, 1)
        self.gng += (dt / 120.0) * ((0.1 + 0.5 * lo * (0.5 + cortisol) + 0.3 * (1 - self.glycogen)) - self.gng)
        self.lipid = np.clip(self.lipid + dt * (0.002 * hi * self.glycogen - 0.0008 * lo), 0, 1)
        self.glucose_out = float(np.mean(0.35 * lo * self.glycogen + self.gng * 0.25)
                                 - 0.2 * np.mean(hi * (1 - self.glycogen)))

    def state(self):
        return np.concatenate([self.glycogen, self.gng, self.lipid])


# ==========================================================================
class GutMicrobiome:
    """Gut segments x nutrient pools with transit, enteric neurons that generate
    motility waves, and a microbial community exchanging metabolites with it."""

    POOLS = ("carbohydrate", "protein", "fat", "water", "electrolytes", "fibre", "bile",
             "microbial_metabolites")

    def __init__(self, seed: int):
        C = COMPLEXITY
        self.ns = int(C.gut_segments)
        self.np_ = len(self.POOLS)
        self.nt = int(C.microbiome_taxa)
        rng = np.random.default_rng(seed)
        self.pool = np.zeros((self.ns, self.np_))
        self.pool[0, 3] = 0.3
        self.transit = np.linspace(1.0, 0.25, self.ns)
        # enteric nervous system: two coupled rings of oscillating neurons
        self.ne = 16 * self.ns
        self.enteric_phase = rng.uniform(0, 2 * np.pi, self.ne)
        self.enteric_freq = rng.uniform(0.04, 0.09, self.ne)        # slow waves ~3/min
        self.motility = np.full(self.ns, 0.4)
        # microbiome: Lotka-Volterra with cross-feeding
        self.abund = rng.dirichlet(np.ones(self.nt) * 2.0)
        self.growth = rng.uniform(0.01, 0.05, self.nt)
        A = rng.normal(0, 0.012, (self.nt, self.nt))
        A -= np.diag(np.diag(A))
        self.inter = A
        self.sub_pref = rng.dirichlet(np.ones(3) * 0.7, self.nt)    # carbs / protein / fibre
        self.scfa = 0.5                                              # short-chain fatty acids
        self.absorbed = np.zeros(4)                                  # glucose, aa, fat, water this tick
        self.discomfort = 0.0
        self.size = self.pool.size + self.ne + self.ns + 2 * self.nt + 4

    def update(self, dt, intake, t):
        """intake: nutrient inflow into the stomach (carb, protein, fat, water)."""
        self.pool[0, :4] += np.asarray(intake) * 1.0
        # enteric waves -> segment motility
        self.enteric_phase += 2 * np.pi * self.enteric_freq * dt
        waves = 0.5 + 0.5 * np.sin(self.enteric_phase.reshape(self.ns, -1))
        self.motility += 0.3 * (waves.mean(axis=1) - self.motility)
        # transit between segments and absorption in the small intestine
        mv = dt * 0.02 * self.transit[:, None] * (0.4 + self.motility[:, None])
        flux = self.pool[:-1] * mv[:-1]
        self.pool[:-1] -= flux
        self.pool[1:] += flux
        small = np.arange(self.ns) < max(self.ns * 2 // 3, 2)
        absb = self.pool * (dt * 0.03) * small[:, None]
        self.pool -= absb
        self.absorbed = np.array([absb[:, 0].sum(), absb[:, 1].sum(), absb[:, 2].sum(),
                                  absb[:, 3].sum() + 0.3 * absb[:, 4].sum()])
        # the colon ferments the fibre and leftover carbohydrate
        colon = self.pool[-3:, :]
        food = colon[:, 0].sum() + 1.5 * colon[:, 5].sum() + 0.2 * colon[:, 1].sum()
        food_per = np.array([colon[:, 0].sum(), colon[:, 1].sum(), colon[:, 5].sum()])
        supply = self.sub_pref @ food_per
        g = self.growth * (supply / (supply + 0.05)) - (self.inter @ self.abund) \
            - 0.02 * self.abund
        self.abund = np.clip(self.abund + dt * self.abund * g, 1e-5, 1.0)
        self.abund /= self.abund.sum()
        prod = float(np.sum(self.abund * self.growth)) * food
        self.scfa += dt * (prod * 4.0 - 0.02 * (self.scfa - 0.5))
        self.pool[-1, 7] += dt * prod
        self.pool[-3:, :6] *= (1.0 - 0.0008 * dt)
        # discomfort: gas / distension when much fibre is fermented fast
        self.discomfort += dt * (0.15 * food * 3.0 - 0.04 * self.discomfort)
        self.discomfort = float(np.clip(self.discomfort, 0, 1.5))

    def state(self):
        return np.concatenate([self.pool.ravel(), np.sin(self.enteric_phase), self.motility,
                               self.abund, self.growth, self.absorbed])


# ==========================================================================
class MuscleBank:
    """Motor units of every muscle: size-principle recruitment, fibre types,
    fatigue, glycogen and delayed-onset soreness.  Two muscles per joint."""

    def __init__(self, n_joints: int, torque_limit: np.ndarray, seed: int):
        C = COMPLEXITY
        self.nj = n_joints
        self.nm = 2 * n_joints
        self.nu = int(C.motor_units_per_muscle)
        rng = np.random.default_rng(seed)
        shp = (self.nm, self.nu)
        u = np.sort(rng.random(shp), axis=1)
        self.thresh = u ** 1.6                              # recruitment threshold (size principle)
        self.slow = u < 0.45                                # fibre type: slow (I) vs fast (II)
        self.tau_fat = np.where(self.slow, 900.0, 70.0) * rng.uniform(0.7, 1.4, shp)
        self.tau_rec = np.where(self.slow, 120.0, 400.0) * rng.uniform(0.7, 1.4, shp)
        self.fatigue = np.zeros(shp)
        self.glycogen = np.ones(shp)
        self.soreness = np.zeros(self.nm)
        self.temp = np.full(self.nm, 34.0)
        self.tlim = np.maximum(torque_limit, 1.0)
        self.active_frac = np.zeros(self.nm)
        self.capacity = np.ones(self.nm)
        self.size = 2 * self.nm * self.nu + 2 * self.nm

    def update(self, dt, tau, perfusion):
        """tau: joint torques (n_joints,) ; perfusion: per-muscle blood flow (nm,)."""
        e = np.abs(tau) / self.tlim
        effort = np.zeros(self.nm)
        effort[0::2] = np.where(tau >= 0, e, 0.1 * e)
        effort[1::2] = np.where(tau < 0, e, 0.1 * e)
        effort = np.clip(effort * 1.15, 0.0, 1.0)
        rec = (effort[:, None] > self.thresh).astype(float)             # recruited units
        gl = self.glycogen
        fat_rate = rec * (1.0 - self.fatigue) / self.tau_fat * (1.0 + (1.0 - gl))
        rec_rate = (1.0 - rec) * self.fatigue / self.tau_rec * np.clip(perfusion[:, None], 0.3, 2.0)
        self.fatigue = np.clip(self.fatigue + dt * (fat_rate - rec_rate), 0.0, 1.0)
        self.glycogen = np.clip(self.glycogen + dt * (-rec * 0.0006 + (1 - rec) * 0.0004 * (1 - gl)), 0.05, 1.0)
        self.active_frac = rec.mean(axis=1)
        self.capacity = 1.0 - 0.6 * (self.fatigue * (rec + 0.3)).mean(axis=1)
        heat = effort ** 2 * 0.2
        self.temp += dt * (heat - (self.temp - 34.0) * 0.05 * np.clip(perfusion, 0.3, 2.0))
        # soreness: slow, from repeated high effort; heals over many minutes
        self.soreness = np.clip(self.soreness + dt * (0.0015 * effort ** 3 - self.soreness / 1800.0), 0, 1)

    def summary(self):
        return {"fatigue": float(self.fatigue.mean()), "soreness": float(self.soreness.mean()),
                "capacity_min": float(self.capacity.min()), "temp": float(self.temp.mean()),
                "active": float(self.active_frac.mean())}

    def state(self):
        return np.concatenate([self.fatigue.ravel(), self.glycogen.ravel(), self.soreness, self.temp])


# ==========================================================================
class SkinThermo:
    """Sweat, blood flow and temperature per skin patch (see skin.PATCH_NAMES)."""

    def __init__(self, patch_names: list[str], seed: int):
        self.n = len(patch_names)
        rng = np.random.default_rng(seed)
        glabrous = np.array([any(k in nm for k in ("palm", "sole", "thumb", "index", "fingers", "toes")) for nm in patch_names])
        self.sweat_density = np.where(glabrous, 1.4, 0.8) * rng.uniform(0.8, 1.2, self.n)
        self.sweat = np.zeros(self.n)
        self.flow = np.ones(self.n)
        self.temp = np.full(self.n, 33.0)
        self.wet = np.zeros(self.n)
        self.size = 4 * self.n

    def update(self, dt, core_temp, ambient, sympathetic, emotional, bed_flow):
        drive = np.clip((core_temp - 37.0) * 2.5, 0, 2.0) + 0.5 * emotional
        self.sweat += (dt / 20.0) * (drive * self.sweat_density - self.sweat)
        self.flow = bed_flow[: self.n]
        tgt = ambient + (core_temp - ambient) * np.clip(0.25 + 0.55 * self.flow, 0.1, 1.2) * 0.8
        self.temp += (dt / 40.0) * (tgt - 0.5 * self.sweat - self.temp)
        self.wet = np.clip(self.wet + dt * (0.02 * self.sweat - 0.004 * self.wet), 0, 1)

    def state(self):
        return np.concatenate([self.sweat, self.flow, self.temp, self.wet])


# ==========================================================================
class Immune:
    """Cell populations and cytokines with signed interactions, plus tissue damage,
    inflammation and healing in every skin patch."""

    def __init__(self, n_patches: int, seed: int):
        C = COMPLEXITY
        self.nc = int(C.immune_populations)
        self.ny = int(C.cytokines)
        self.np_ = n_patches
        rng = np.random.default_rng(seed)
        self.cells = np.ones(self.nc)
        self.cyto = np.zeros(self.ny)
        # signed interaction matrix: pro-inflammatory cytokines (first 60 %) excite
        # each other and the cells; anti-inflammatory ones suppress
        pro = np.arange(self.ny) < int(0.6 * self.ny)
        sign = np.where(pro, 1.0, -1.0)
        W = rng.random((self.ny, self.ny)) * (rng.random((self.ny, self.ny)) < 0.12)
        self.W_cc = (W * sign[None, :]) * 0.5
        self.W_cell_cy = rng.random((self.ny, self.nc)) * (rng.random((self.ny, self.nc)) < 0.2) * 0.6
        self.tau_cy = rng.uniform(300.0, 3600.0, self.ny)
        self.damage = np.zeros(n_patches)
        self.infl = np.zeros(n_patches)
        self.pruri = np.zeros(n_patches)
        self.systemic = 0.0
        self.size = self.nc + self.ny + 3 * n_patches

    def update(self, dt, patch_load, perfusion, cortisol, sleep_loss, pathogen=0.0):
        """patch_load: per-patch mechanical / noxious load (0..)."""
        # tissue damage: accumulates under sustained noxious load, heals with perfusion
        self.damage = np.clip(self.damage + dt * (0.012 * patch_load
                                                  - self.damage * 0.0006 * np.clip(perfusion, 0.3, 2.0)), 0, 1.2)
        local = float(np.mean(self.damage))
        drive = 0.4 * local + 0.5 * pathogen + 0.15 * sleep_loss
        act = np.maximum(self.W_cc @ self.cyto + self.W_cell_cy @ (self.cells - 1.0) * 0.3 + drive, 0.0)
        act = np.tanh(act) * (1.0 - 0.5 * np.tanh(cortisol))
        self.cyto += dt * ((act - self.cyto) / self.tau_cy)
        self.cells += dt * (0.002 * (self.W_cell_cy.T @ self.cyto) - 0.0005 * (self.cells - 1.0))
        self.cells = np.clip(self.cells, 0.2, 4.0)
        pro = float(np.mean(self.cyto[: int(0.6 * self.ny)]))
        self.systemic = pro
        self.infl += dt * ((0.7 * self.damage + 0.8 * pro - self.infl) / 240.0)
        self.infl = np.clip(self.infl, 0.0, 1.5)
        self.pruri += dt * ((0.4 * self.infl * (self.damage > 0.2) - self.pruri) / 120.0)

    def state(self):
        return np.concatenate([self.cells, self.cyto, self.damage, self.infl, self.pruri])


# ==========================================================================
# Chemistry panel
# ==========================================================================
_NAMED = [  # (name, group, setpoint, unit-ish scale, tau seconds)
    ("sodium", "ion", 140, 3, 600), ("potassium", "ion", 4.2, 0.3, 400),
    ("chloride", "ion", 102, 3, 600), ("calcium", "ion", 2.4, 0.1, 900),
    ("magnesium", "ion", 0.85, 0.05, 900), ("phosphate", "ion", 1.1, 0.1, 900),
    ("bicarbonate", "acidbase", 24, 2, 300), ("ph", "acidbase", 7.40, 0.03, 200),
    ("lactate", "metabolite", 1.0, 0.5, 240), ("pyruvate", "metabolite", 0.08, 0.03, 240),
    ("glucose", "glucose", 5.2, 0.8, 300), ("insulin", "hormone", 6.0, 3.0, 400),
    ("glucagon", "hormone", 60, 15, 300), ("cortisol", "hormone", 300, 80, 1800),
    ("acth", "hormone", 25, 10, 900), ("crh", "hormone", 10, 4, 600),
    ("adrenaline", "hormone", 0.3, 0.2, 60), ("noradrenaline", "hormone", 1.5, 0.6, 90),
    ("growth_hormone", "hormone", 2.0, 1.5, 1200), ("igf1", "hormone", 180, 40, 3600),
    ("tsh", "hormone", 2.0, 0.8, 3600), ("t3", "hormone", 4.5, 0.8, 7200),
    ("t4", "hormone", 100, 15, 7200), ("prolactin", "hormone", 10, 4, 1800),
    ("melatonin", "hormone", 5, 12, 1800), ("testosterone", "hormone", 18, 4, 3600),
    ("estradiol", "hormone", 120, 30, 3600), ("progesterone", "hormone", 5, 2, 3600),
    ("lh", "hormone", 6, 2, 3600), ("fsh", "hormone", 6, 2, 3600),
    ("aldosterone", "hormone", 200, 60, 900), ("renin", "hormone", 1.2, 0.5, 900),
    ("adh", "hormone", 2.5, 1.0, 300), ("anp", "hormone", 30, 10, 300),
    ("leptin", "hormone", 8, 3, 3600), ("ghrelin", "hormone", 800, 200, 1800),
    ("glp1", "hormone", 20, 8, 600), ("pyy", "hormone", 60, 20, 900),
    ("cck", "hormone", 1.0, 0.4, 600), ("oxytocin", "hormone", 3.0, 1.5, 300),
    ("urea", "nitrogen", 5.0, 1.0, 3600), ("creatinine", "nitrogen", 80, 10, 7200),
    ("uric_acid", "nitrogen", 300, 50, 3600), ("ammonia", "nitrogen", 25, 8, 600),
    ("albumin", "protein", 42, 3, 14400), ("globulin", "protein", 28, 3, 14400),
    ("crp", "inflammation", 1.0, 1.0, 3600), ("ferritin", "protein", 100, 40, 14400),
    ("hemoglobin", "protein", 145, 8, 14400), ("fibrinogen", "protein", 3.0, 0.6, 7200),
    ("total_cholesterol", "lipid", 4.8, 0.7, 7200), ("ldl", "lipid", 2.8, 0.6, 7200),
    ("hdl", "lipid", 1.4, 0.3, 7200), ("triglycerides", "lipid", 1.2, 0.5, 1800),
    ("free_fatty_acids", "lipid", 0.4, 0.2, 600), ("ketones", "metabolite", 0.1, 0.1, 900),
    ("alt", "enzyme", 25, 8, 14400), ("ast", "enzyme", 24, 7, 14400),
    ("ck", "enzyme", 120, 60, 3600), ("ldh", "enzyme", 180, 30, 7200),
    ("alp", "enzyme", 70, 15, 14400), ("amylase", "enzyme", 70, 15, 7200),
    ("lipase", "enzyme", 30, 10, 7200), ("ggt", "enzyme", 25, 8, 14400),
    ("troponin", "enzyme", 0.005, 0.004, 3600), ("bnp", "enzyme", 20, 10, 3600),
    ("osmolality", "ion", 290, 4, 400), ("anion_gap", "acidbase", 12, 2, 600),
    ("pco2", "acidbase", 40, 3, 120), ("po2", "acidbase", 95, 5, 120),
]
_AMINO = ["alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamate",
          "glutamine", "glycine", "histidine", "isoleucine", "leucine", "lysine",
          "methionine", "phenylalanine", "proline", "serine", "threonine", "tryptophan",
          "tyrosine", "valine", "taurine", "ornithine", "citrulline", "carnitine"]
_VITAMINS = ["vitamin_a", "vitamin_b1", "vitamin_b2", "vitamin_b3", "vitamin_b5",
             "vitamin_b6", "vitamin_b7", "vitamin_b9", "vitamin_b12", "vitamin_c",
             "vitamin_d", "vitamin_e", "vitamin_k", "zinc", "copper", "selenium",
             "iron", "iodine", "manganese", "chromium"]
_CYT = ["il1b", "il2", "il4", "il6", "il8", "il10", "il12", "il13", "il17", "il18",
        "ifng", "tnfa", "tgfb", "gcsf", "mcp1"]
_NT = ["dopamine", "serotonin", "gaba", "glutamate", "acetylcholine", "histamine",
       "substance_p", "bdnf", "orexin", "endorphin", "enkephalin", "anandamide",
       "adenosine", "kynurenine", "neuropeptide_y", "vip"]
_FACTORS = ("exertion", "feeding", "stress", "circadian", "dehydration", "inflammation",
            "fatigue", "heat", "sleep_debt")


class Chemistry:
    """A blood / tissue chemistry panel (~480 channels).

    About 70 channels are *named analytes with real setpoints* and group-level
    couplings; the remainder (amino acids, vitamins and minerals, neurotransmitter
    metabolites, cytokine spill-over, lipid species ...) are generated procedurally
    and follow the same nine physiological factors through random loadings.  This is
    a *factor model*, not a pathway model: a faithful pathway model of 480 analytes
    does not exist; the point is a high-dimensional, correlated, slowly-moving
    internal chemical state that responds to what the body is doing.
    """

    GROUPS = ("ion", "acidbase", "metabolite", "glucose", "hormone", "nitrogen",
              "protein", "lipid", "enzyme", "inflammation", "aminoacid", "vitamin",
              "neurochem", "other")

    def __init__(self, seed: int):
        n = int(COMPLEXITY.analytes)
        rng = np.random.default_rng(seed)
        names, groups, sp, sc, tau = [], [], [], [], []
        for nm, g, s, c, t in _NAMED:
            names.append(nm); groups.append(g); sp.append(s); sc.append(c); tau.append(t)
        for nm in _AMINO:
            names.append("aa_" + nm); groups.append("aminoacid")
            sp.append(rng.uniform(40, 400)); sc.append(sp[-1] * 0.12); tau.append(rng.uniform(900, 3600))
        for nm in _VITAMINS:
            names.append(nm); groups.append("vitamin")
            sp.append(rng.uniform(1, 80)); sc.append(sp[-1] * 0.15); tau.append(rng.uniform(7200, 43200))
        for nm in _CYT:
            names.append("blood_" + nm); groups.append("inflammation")
            sp.append(rng.uniform(1, 12)); sc.append(sp[-1] * 0.5); tau.append(rng.uniform(600, 3600))
        for nm in _NT:
            names.append("csf_" + nm); groups.append("neurochem")
            sp.append(rng.uniform(5, 200)); sc.append(sp[-1] * 0.2); tau.append(rng.uniform(300, 3600))
        k = 0
        fam = ["acylcarnitine", "phospholipid", "sphingolipid", "bile_acid", "steroid_metabolite",
               "nucleotide", "tca_intermediate", "peptide", "oxylipin", "glycan"]
        while len(names) < n:
            f = fam[k % len(fam)]
            names.append(f"{f}_{k // len(fam):03d}"); groups.append("other")
            sp.append(rng.uniform(0.5, 50)); sc.append(sp[-1] * 0.2); tau.append(rng.uniform(900, 14400))
            k += 1
        self.names = names[:n]
        self.groups = np.array(groups[:n])
        self.setpoint = np.array(sp[:n], float)
        self.scale = np.array(sc[:n], float)
        self.tau = np.array(tau[:n], float)
        self.n = n
        self.z = rng.normal(0, 0.05, n)                 # normalised deviation from setpoint
        # loadings of each analyte on the nine factors, by group
        L = np.zeros((n, len(_FACTORS)))
        gi = {g: i for i, g in enumerate(self.GROUPS)}
        prof = {
            "ion": dict(dehydration=1.0, exertion=0.3, heat=0.4), "acidbase": dict(exertion=0.8, stress=0.2),
            "metabolite": dict(exertion=1.0, feeding=0.4, fatigue=0.3), "glucose": dict(feeding=1.0, exertion=-0.4, stress=0.4),
            "hormone": dict(stress=0.8, circadian=0.7, feeding=0.3, sleep_debt=0.3, exertion=0.3),
            "nitrogen": dict(dehydration=0.7, exertion=0.3, feeding=0.3), "protein": dict(inflammation=0.5, dehydration=0.4),
            "lipid": dict(feeding=0.9, exertion=-0.3, stress=0.3), "enzyme": dict(exertion=0.8, inflammation=0.5, fatigue=0.4),
            "inflammation": dict(inflammation=1.0, stress=0.3, sleep_debt=0.4), "aminoacid": dict(feeding=0.8, exertion=0.4, fatigue=0.3),
            "vitamin": dict(feeding=0.3, dehydration=0.1), "neurochem": dict(stress=0.5, circadian=0.6, sleep_debt=0.5, fatigue=0.4),
            "other": dict(exertion=0.4, feeding=0.4, stress=0.3, circadian=0.3, fatigue=0.3),
        }
        fi = {f: i for i, f in enumerate(_FACTORS)}
        for j, g in enumerate(self.groups):
            for f, w in prof[g].items():
                L[j, fi[f]] = w * rng.uniform(0.5, 1.5) * rng.choice([1.0, 1.0, -1.0], p=[0.5, 0.2, 0.3]) \
                    if g in ("other", "vitamin", "aminoacid", "neurochem") else w * rng.uniform(0.7, 1.3)
        self.L = L * 0.6
        # sparse within-group coupling (excitation and inhibition)
        A = np.zeros((n, n))
        for g in self.GROUPS:
            idx = np.where(self.groups == g)[0]
            if len(idx) < 3:
                continue
            for j in idx:
                pre = rng.choice(idx, size=min(3, len(idx) - 1), replace=False)
                A[j, pre] = rng.normal(0, 0.12, len(pre))
        # keep the linear system stable: scale so that the spectral radius is < 0.6
        v = rng.standard_normal(n)
        rad = 1.0
        for _ in range(60):
            v = A @ v
            rad = float(np.linalg.norm(v)) + 1e-12
            v = v / rad
        self.A = A * (0.6 / rad if rad > 0.6 else 1.0)
        self.noise = 0.01
        self.rng = rng
        self.idx = {nm: i for i, nm in enumerate(self.names)}
        self.size = n

    def value(self, name: str) -> float:
        i = self.idx[name]
        return float(self.setpoint[i] + self.scale[i] * self.z[i])

    def update(self, dt, factors: np.ndarray):
        """factors: the nine physiological factors, each roughly in 0..1.5."""
        drive = self.L @ factors + self.A @ np.tanh(self.z)
        self.z += dt * ((drive - self.z) / self.tau) + np.sqrt(dt) * self.noise * self.rng.standard_normal(self.n) \
            / np.sqrt(np.maximum(self.tau / 600.0, 1.0))
        self.z = np.clip(self.z, -4.0, 4.0)

    def group_summary(self) -> np.ndarray:
        out = []
        for g in self.GROUPS:
            m = self.groups == g
            zz = self.z[m] if m.any() else np.zeros(1)
            out += [float(zz.mean()), float(zz.std()), float(np.abs(zz).max())]
        return np.array(out)

    def state(self):
        return self.z.copy()
