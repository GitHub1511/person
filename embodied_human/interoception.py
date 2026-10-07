"""
Interoception: the sense of the internal state of the body.

This is the modality that separates a machine that *has* a body from one that
*is* a body.  It tracks ~65 physiological variables across seven organ systems
and integrates them as ordinary differential equations, so a sprint leaves you
with lactate and an oxygen debt, being cold constricts the skin's blood
vessels, and pain is gated by endorphins released under stress.

Systems
-------
cardiovascular   heart rate, HRV, stroke volume, cardiac output, blood
                 pressure, cutaneous and muscular perfusion
respiratory      rate, tidal volume, minute ventilation, SpO2, arterial CO2,
                 air hunger
metabolic        glucose, glycogen, lactate, ATP reserve, metabolic rate,
                 cumulative energy expenditure
thermoregulatory core and skin temperature, shivering, sweating, discomfort
gastrointestinal gastric fullness, ghrelin (hunger), leptin (satiety), nausea
renal            hydration, osmolality, bladder fullness, electrolytes
fatigue          peripheral and central fatigue, adenosine (sleep pressure)
immune           cytokines, inflammation, sickness behaviour
nociceptive      pain intensity / unpleasantness, central sensitisation
effort           sense of effort (from corollary discharge), breathlessness
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from .config import SimConfig

# Ordered scalar outputs ----------------------------------------------------
INTERO_NAMES = (
    # cardiovascular (10)
    "heart_rate", "hrv_rmssd", "stroke_volume", "cardiac_output",
    "systolic_bp", "diastolic_bp", "perfusion_skin", "perfusion_muscle",
    "baroreflex_error", "venous_return",
    # respiratory (7)
    "resp_rate", "tidal_volume", "minute_ventilation", "spo2",
    "co2_arterial", "o2_arterial", "air_hunger",
    # metabolic (8)
    "glucose", "glycogen", "lactate", "atp_reserve", "metabolic_rate",
    "energy_expended_kj", "insulin", "glucagon",
    # thermoregulatory (7)
    "core_temp", "skin_temp_mean", "shivering", "sweating",
    "thermal_discomfort", "heat_production", "heat_loss",
    # gastrointestinal (6)
    "gastric_fullness", "ghrelin", "leptin", "nausea", "gut_motility",
    "nutrient_absorption",
    # renal (6)
    "hydration", "osmolality", "bladder_fullness", "sodium", "potassium",
    "urine_output",
    # fatigue (6)
    "peripheral_fatigue", "central_fatigue", "adenosine", "sleepiness",
    "alertness", "recovery_need",
    # immune (4)
    "cytokine", "inflammation", "sickness_behavior", "tissue_damage",
    # nociceptive (5)
    "nociceptive_input", "pain_intensity", "pain_unpleasantness",
    "central_sensitisation", "descending_inhibition",
    # effort (4)
    "sense_of_effort", "breathlessness", "motor_command_magnitude",
    "muscle_tone",
    # endocrine adjuncts (4)
    "adrenaline_h", "noradrenaline_h", "cortisol_h", "insulin_sensitivity",
)
N_INTEROCEPTION = len(INTERO_NAMES)
IDX = {n: i for i, n in enumerate(INTERO_NAMES)}


@dataclass
class InteroInputs:
    """What the rest of the agent tells the body about what it is doing."""
    exertion: float = 0.0          # 0..1 normalised motor effort
    mechanical_power: float = 0.0  # W
    arousal: float = 0.2           # from the affect system
    stress: float = 0.1            # from the affect system
    pain_afferent: float = 0.0     # from the afferent layer
    itch_afferent: float = 0.0
    affective_touch: float = 0.0
    touch_intensity: float = 0.0
    ambient_temp: float = 22.0
    contact_heat: float = 0.0
    contact_cold: float = 0.0
    taste_sweet: float = 0.0
    taste_bitter: float = 0.0
    odor_food: float = 0.0
    vestibular_discomfort: float = 0.0
    endorphin: float = 0.0         # from the affect system (analgesia)
    falling: float = 0.0
    sleep_debt: float = 0.0


class InteroceptiveSystem:
    """Integrates the internal milieu."""

    def __init__(self, cfg: SimConfig, meta=None):
        self.cfg = cfg
        self.I = cfg.intero
        self.s = np.zeros(N_INTEROCEPTION)
        self.prev = np.zeros(N_INTEROCEPTION)
        self._init_state()
        self.rng = np.random.default_rng(cfg.seed + 41)
        self.total_energy_j = 0.0
        self.chronic_stress_load = 0.0
        self.meal_timer = 0.0
        self.history: list = []

    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        I, s = self.I, self.s
        s[IDX["heart_rate"]] = I.hr_rest
        s[IDX["hrv_rmssd"]] = 55.0
        s[IDX["stroke_volume"]] = 70.0
        s[IDX["cardiac_output"]] = I.hr_rest * 70.0 / 1000.0
        s[IDX["systolic_bp"]] = 118.0
        s[IDX["diastolic_bp"]] = 74.0
        s[IDX["perfusion_skin"]] = 1.0
        s[IDX["perfusion_muscle"]] = 1.0
        s[IDX["venous_return"]] = 1.0
        s[IDX["resp_rate"]] = I.rr_rest
        s[IDX["tidal_volume"]] = I.tidal_volume
        s[IDX["minute_ventilation"]] = I.rr_rest * I.tidal_volume
        s[IDX["spo2"]] = 98.0
        s[IDX["co2_arterial"]] = 40.0
        s[IDX["o2_arterial"]] = 97.0
        s[IDX["glucose"]] = I.glucose_rest
        s[IDX["glycogen"]] = I.glycogen_rest
        s[IDX["lactate"]] = I.lactate_rest
        s[IDX["atp_reserve"]] = I.atp_rest
        s[IDX["metabolic_rate"]] = 85.0
        s[IDX["insulin"]] = 0.5
        s[IDX["glucagon"]] = 0.5
        s[IDX["core_temp"]] = I.core_temp_rest
        s[IDX["skin_temp_mean"]] = 33.0
        s[IDX["heat_production"]] = 85.0
        s[IDX["heat_loss"]] = 85.0
        s[IDX["gastric_fullness"]] = 0.45
        s[IDX["ghrelin"]] = 0.45
        s[IDX["leptin"]] = 0.55
        s[IDX["gut_motility"]] = 0.5
        s[IDX["hydration"]] = 1.0
        s[IDX["osmolality"]] = I.osmolality_rest
        s[IDX["bladder_fullness"]] = 0.15
        s[IDX["sodium"]] = 140.0
        s[IDX["potassium"]] = 4.2
        s[IDX["adenosine"]] = 0.25
        s[IDX["sleepiness"]] = 0.25
        s[IDX["alertness"]] = 0.75
        s[IDX["insulin_sensitivity"]] = 1.0
        s[IDX["muscle_tone"]] = 0.2
        s[IDX["venous_return"]] = 1.0
        self.prev = s.copy()

    # ------------------------------------------------------------------
    def update(self, dt: float, inp: InteroInputs) -> np.ndarray:
        I, s = self.I, self.s
        g = IDX

        exertion = float(fclip(inp.exertion, 0.0, 1.0))
        arousal = float(fclip(inp.arousal, 0.0, 1.0))
        stress = float(fclip(inp.stress, 0.0, 1.0))

        # ---------------- metabolic --------------------------------
        power = max(inp.mechanical_power, 0.0)
        metabolic = 85.0 + 0.22 * power + 900.0 * exertion + 260.0 * s[g["shivering"]]
        s[g["metabolic_rate"]] += (dt / 5.0) * (metabolic - s[g["metabolic_rate"]])
        self.total_energy_j += s[g["metabolic_rate"]] * dt
        s[g["energy_expended_kj"]] = self.total_energy_j / 1000.0

        intensity = s[g["metabolic_rate"]] / 1600.0
        # glucose uptake scales with intensity; glycogen buffers the shortfall
        uptake = 0.0022 * (0.4 + 5.0 * intensity) * s[g["insulin_sensitivity"]]
        glycogen_use = max(0.0, uptake - 0.004) * 1.4
        s[g["glucose"]] += dt * (-uptake + 0.010 * s[g["nutrient_absorption"]]
                                 + 0.055 * s[g["glucagon"]])
        s[g["glycogen"]] = float(fclip(
            s[g["glycogen"]] + dt * (0.00025 - glycogen_use * 0.02), 0.0, 1.2))
        # lactate appears above the lactate threshold (~55 % of max)
        production = max(0.0, exertion - 0.55) * 1.1
        clearance = 0.02 * (s[g["lactate"]] - I.lactate_rest) * (1.0 + 2.0 * s[g["perfusion_muscle"]])
        s[g["lactate"]] = float(fclip(
            s[g["lactate"]] + dt * (production - clearance), 0.2, 25.0))
        s[g["atp_reserve"]] = float(fclip(
            s[g["atp_reserve"]] + dt * ((1.0 - s[g["atp_reserve"]]) * 0.05
                                        - exertion * 0.02), 0.0, 1.0))
        s[g["insulin"]] += dt * (1.5 * (s[g["glucose"]] - I.glucose_rest) / 3.0
                                 - 0.35 * s[g["insulin"]])
        s[g["glucagon"]] += dt * (1.2 * max(0.0, I.glucose_rest - s[g["glucose"]]) / 2.0
                                  - 0.4 * s[g["glucagon"]])

        # ---------------- thermoregulation -------------------------
        heat_prod = 0.28 * s[g["metabolic_rate"]] + 320.0 * s[g["shivering"]]
        ambient = inp.ambient_temp
        skin_c = s[g["skin_temp_mean"]]
        # conductive exchange through the skin, then core<->skin transfer
        effective_skin = skin_c - 0.45 * inp.contact_cold + 0.55 * inp.contact_heat
        heat_loss = 6.5 * (skin_c - ambient) * (0.4 + s[g["perfusion_skin"]]) \
            + 22.0 * max(0.0, s[g["sweating"]])
        s[g["heat_production"]] += (dt / 4.0) * (heat_prod - s[g["heat_production"]])
        s[g["heat_loss"]] += (dt / 6.0) * (heat_loss - s[g["heat_loss"]])
        core_flow = 0.0009 * (s[g["heat_production"]] - s[g["heat_loss"]])
        s[g["core_temp"]] += dt * core_flow + self.rng.normal(0, 0.0004)
        skin_target = ambient + (s[g["core_temp"]] - ambient) * (
            0.35 + 0.45 * s[g["perfusion_skin"]]) - 0.25 * inp.contact_cold \
            + 0.35 * inp.contact_heat
        s[g["skin_temp_mean"]] += (dt / 30.0) * (skin_target - s[g["skin_temp_mean"]])

        shiver_t = float(fclip((I.shivering_threshold - s[g["core_temp"]]) * 6.0, 0, 1))
        sweat_t = float(fclip((s[g["core_temp"]] - I.sweating_threshold) * 5.0, 0, 1))
        s[g["shivering"]] += (dt / 3.0) * (shiver_t - s[g["shivering"]])
        s[g["sweating"]] += (dt / 8.0) * (sweat_t - s[g["sweating"]])
        s[g["thermal_discomfort"]] = float(fclip(
            3.0 * abs(s[g["core_temp"]] - I.core_temp_rest)
            + 0.035 * abs(s[g["skin_temp_mean"]] - 33.0)
            + 0.25 * s[g["shivering"]], 0.0, 1.5))

        # ---------------- cardiovascular ---------------------------
        target_hr = (I.hr_rest
                     + (I.hr_max - I.hr_rest) * (0.60 * exertion + 0.30 * arousal
                                                 + 0.16 * stress
                                                 + 0.10 * s[g["pain_intensity"]]))
        tau_hr = 1.6 if target_hr > s[g["heart_rate"]] else 3.2
        s[g["heart_rate"]] += (dt / tau_hr) * (target_hr - s[g["heart_rate"]])
        s[g["hrv_rmssd"]] += (dt / 8.0) * (
            55.0 / (1.0 + 2.2 * stress + 1.1 * arousal + 2.0 * exertion)
            - s[g["hrv_rmssd"]])
        s[g["stroke_volume"]] += (dt / 3.0) * (
            70.0 * (1.0 + 0.35 * exertion - 0.25 * max(0.0, 0.5 - s[g["hydration"]]))
            - s[g["stroke_volume"]])
        s[g["cardiac_output"]] = s[g["heart_rate"]] * s[g["stroke_volume"]] / 1000.0
        s[g["systolic_bp"]] += (dt / 3.0) * (
            112.0 + 0.30 * (s[g["heart_rate"]] - I.hr_rest)
            + 22.0 * exertion + 12.0 * stress - s[g["systolic_bp"]])
        s[g["diastolic_bp"]] += (dt / 4.0) * (
            72.0 + 0.16 * (s[g["heart_rate"]] - I.hr_rest)
            + 10.0 * exertion + 8.0 * stress - s[g["diastolic_bp"]])
        # cutaneous vasoconstriction with cold and stress, vasodilation with heat
        perf_t = float(fclip(
            1.0 - 1.5 * (I.core_temp_rest - s[g["core_temp"]]) * 4.0
            - 0.55 * stress + 1.1 * max(0.0, s[g["core_temp"]] - I.core_temp_rest) * 4.0
            - 0.4 * inp.contact_cold, 0.18, 1.7))
        s[g["perfusion_skin"]] += (dt / 6.0) * (perf_t - s[g["perfusion_skin"]])
        s[g["perfusion_muscle"]] += (dt / 4.0) * (
            1.0 + 2.2 * exertion - s[g["perfusion_muscle"]])
        s[g["baroreflex_error"]] = float(fclip(
            (118.0 - s[g["systolic_bp"]]) / 30.0, -1.5, 1.5))
        s[g["venous_return"]] += (dt / 5.0) * (
            1.0 + 0.8 * exertion - 0.5 * max(0.0, 1.0 - s[g["hydration"]])
            - s[g["venous_return"]])

        # ---------------- respiratory ------------------------------
        co2_drive = max(0.0, s[g["co2_arterial"]] - 40.0) / 6.0
        target_rr = I.rr_rest * (1.0 + 2.5 * exertion + 0.55 * arousal
                                 + 1.4 * co2_drive + 0.3 * stress)
        s[g["resp_rate"]] += (dt / 2.5) * (target_rr - s[g["resp_rate"]])
        s[g["tidal_volume"]] += (dt / 2.0) * (
            I.tidal_volume * (1.0 + 1.4 * exertion + 0.4 * co2_drive)
            - s[g["tidal_volume"]])
        s[g["minute_ventilation"]] = s[g["resp_rate"]] * s[g["tidal_volume"]]
        # CO2 is produced metabolically and cleared by ventilation
        clearance = 0.010 * s[g["minute_ventilation"]]
        production_co2 = 0.0085 * (s[g["metabolic_rate"]] / 85.0)
        s[g["co2_arterial"]] = float(fclip(
            s[g["co2_arterial"]] + dt * (production_co2 - clearance) * 6.0,
            25.0, 75.0))
        s[g["spo2"]] += (dt / 4.0) * (
            99.0 - 3.5 * max(0.0, s[g["co2_arterial"]] - 46.0) / 4.0
            - 1.2 * exertion - 0.8 * inp.falling - s[g["spo2"]])
        s[g["o2_arterial"]] += (dt / 4.0) * (
            s[g["spo2"]] - 2.0 - s[g["o2_arterial"]])
        # air hunger: a distinct, unpleasant sensation driven by CO2
        s[g["air_hunger"]] = float(fclip(
            (s[g["co2_arterial"]] - 42.0) / 8.0
            + 0.6 * exertion * max(0.0, 1.0 - s[g["minute_ventilation"]] / 20.0),
            0.0, 1.0))

        # ---------------- gastrointestinal -------------------------
        self.meal_timer += dt
        s[g["nutrient_absorption"]] = float(np.exp(-self.meal_timer / 1800.0)) \
            if self.meal_timer < 5400 else 0.0
        empty = (1.0 - s[g["gastric_fullness"]]) / I.gastric_emptying_tau
        s[g["gastric_fullness"]] = float(fclip(
            s[g["gastric_fullness"]] - dt * empty * 60.0
            + dt * 0.25 * inp.odor_food, 0.0, 1.2))
        s[g["ghrelin"]] += (dt / 900.0) * (
            1.0 - s[g["gastric_fullness"]] - s[g["ghrelin"]])
        s[g["leptin"]] += (dt / 1800.0) * (
            0.55 + 0.25 * (s[g["glucose"]] - I.glucose_rest) / 2.0 - s[g["leptin"]])
        naus_t = float(fclip(
            0.8 * inp.vestibular_discomfort + 0.9 * inp.taste_bitter
            + 0.004 * max(0.0, s[g["lactate"]] - 8.0) + 0.5 * inp.pain_afferent * 0.4,
            0.0, 1.2))
        s[g["nausea"]] += (dt / 12.0) * (naus_t - s[g["nausea"]])
        s[g["gut_motility"]] += (dt / 30.0) * (
            0.8 * s[g["gastric_fullness"]] - s[g["gut_motility"]])

        # ---------------- renal ------------------------------------
        sweat_loss = 0.00035 * s[g["sweating"]]
        urine = 0.00012 * (1.0 + 0.4 * s[g["bladder_fullness"]])
        s[g["hydration"]] = float(fclip(
            s[g["hydration"]] - dt * (sweat_loss + urine * 0.1), 0.5, 1.05))
        s[g["osmolality"]] = I.osmolality_rest * (
            1.0 + 3.2 * (1.0 - s[g["hydration"]]))
        s[g["bladder_fullness"]] = float(fclip(
            s[g["bladder_fullness"]] + dt * I.bladder_fill_rate, 0.0, 1.2))
        s[g["urine_output"]] = urine * 3600.0
        s[g["sodium"]] += dt * (0.02 * (1.0 - s[g["hydration"]]) - 0.002)
        s[g["potassium"]] = float(fclip(
            4.2 + 0.06 * max(0.0, s[g["lactate"]] - 4.0) - 0.02 * exertion, 3.0, 6.0))

        # ---------------- fatigue ----------------------------------
        s[g["peripheral_fatigue"]] = float(fclip(
            s[g["peripheral_fatigue"]]
            + dt * (exertion ** 1.5 * 0.020
                    - s[g["peripheral_fatigue"]] / I.tau_peripheral_fatigue),
            0.0, 1.0))
        s[g["central_fatigue"]] = float(fclip(
            s[g["central_fatigue"]]
            + dt * (0.00004 + 0.0006 * exertion ** 2
                    - s[g["central_fatigue"]] / I.tau_central_fatigue), 0.0, 1.0))
        s[g["adenosine"]] = float(fclip(
            s[g["adenosine"]] + dt * (1.0 / I.tau_adenosine) - dt * 0.0,
            0.0, 1.2))
        s[g["sleepiness"]] = float(fclip(
            0.35 * s[g["adenosine"]] / 0.6 + 0.5 * inp.sleep_debt
            + 0.25 * s[g["central_fatigue"]], 0.0, 1.2))
        s[g["alertness"]] = float(fclip(
            1.0 - s[g["sleepiness"]] + 0.35 * arousal - 0.2 * s[g["pain_intensity"]],
            0.0, 1.2))
        s[g["recovery_need"]] = float(fclip(
            0.5 * s[g["peripheral_fatigue"]] + 0.5 * s[g["central_fatigue"]], 0, 1))

        # ---------------- immune -----------------------------------
        injury = 0.35 * inp.pain_afferent + 0.5 * s[g["tissue_damage"]]
        s[g["tissue_damage"]] = float(fclip(
            s[g["tissue_damage"]] + dt * (0.02 * inp.pain_afferent - 0.0002), 0, 1))
        s[g["cytokine"]] += (dt / I.tau_cytokine) * (injury - s[g["cytokine"]])
        s[g["inflammation"]] = float(fclip(
            0.7 * s[g["cytokine"]] + 0.3 * s[g["tissue_damage"]], 0, 1))
        s[g["sickness_behavior"]] = float(fclip(
            s[g["inflammation"]] - 0.15, 0, 1))

        # ---------------- nociception ------------------------------
        s[g["nociceptive_input"]] += (dt / 0.25) * (
            inp.pain_afferent - s[g["nociceptive_input"]])
        s[g["central_sensitisation"]] = float(fclip(
            s[g["central_sensitisation"]]
            + dt * (0.004 * s[g["nociceptive_input"]] - 0.0008), 0, 1.2))
        # descending inhibition: endorphins and positive affect close the gate
        s[g["descending_inhibition"]] = float(fclip(
            1.2 * inp.endorphin + 0.28 * (1.0 - stress), 0, 1.4))
        gated = (s[g["nociceptive_input"]] * (1.0 + 0.9 * s[g["central_sensitisation"]])
                 * (1.0 - 0.75 * s[g["descending_inhibition"]]))
        s[g["pain_intensity"]] = float(fclip(gated, 0.0, 1.5))
        s[g["pain_unpleasantness"]] = float(fclip(
            s[g["pain_intensity"]] * (0.6 + 0.5 * stress + 0.3 * (1 - s[g["alertness"]])),
            0.0, 1.5))

        # ---------------- effort -----------------------------------
        s[g["motor_command_magnitude"]] = float(1.0 * exertion)
        s[g["sense_of_effort"]] += (dt / 0.8) * (
            exertion * (1.0 + 0.7 * s[g["peripheral_fatigue"]])
            + 0.3 * s[g["sense_of_effort"]] * 0.0 - s[g["sense_of_effort"]])
        s[g["breathlessness"]] = float(fclip(
            0.55 * s[g["air_hunger"]] + 0.5 * exertion * (1 - s[g["spo2"]] / 100.0) * 5,
            0, 1.2))
        s[g["muscle_tone"]] += (dt / 2.0) * (
            0.2 + 0.5 * exertion + 0.25 * arousal - s[g["muscle_tone"]])

        # ---------------- endocrine adjuncts -----------------------
        s[g["adrenaline_h"]] += (dt / 3.0) * (
            arousal * 0.7 + stress * 0.6 + 0.25 * exertion - s[g["adrenaline_h"]])
        s[g["noradrenaline_h"]] += (dt / 6.0) * (
            0.35 + 0.4 * arousal + 0.3 * stress - s[g["noradrenaline_h"]])
        s[g["cortisol_h"]] += (dt / 300.0) * (
            0.25 + 0.6 * stress + 0.3 * s[g["pain_intensity"]] - s[g["cortisol_h"]])
        s[g["insulin_sensitivity"]] = float(fclip(
            1.0 - 0.35 * s[g["inflammation"]] - 0.2 * s[g["central_fatigue"]]
            + 0.1 * exertion, 0.3, 1.3))

        # track allostatic load (chronic wear)
        self.chronic_stress_load = float(fclip(
            self.chronic_stress_load + dt * (stress * 0.001 - 0.0002), 0, 1))

        s[:] = np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)
        return s.copy()

    # ------------------------------------------------------------------
    def vector(self) -> np.ndarray:
        return self.s.copy()

    def as_dict(self) -> dict:
        return {n: float(self.s[i]) for i, n in enumerate(INTERO_NAMES)}

    def eat(self, calories: float = 1.0) -> None:
        self.s[IDX["gastric_fullness"]] = float(fclip(
            self.s[IDX["gastric_fullness"]] + 0.6 * calories, 0, 1.2))
        self.meal_timer = 0.0

    def drink(self, amount: float = 1.0) -> None:
        self.s[IDX["hydration"]] = float(fclip(
            self.s[IDX["hydration"]] + 0.2 * amount, 0, 1.05))

    def sleep(self, hours: float = 8.0) -> None:
        self.s[IDX["adenosine"]] = float(fclip(
            self.s[IDX["adenosine"]] - 0.15 * hours, 0, 1.2))
        self.s[IDX["central_fatigue"]] = float(fclip(
            self.s[IDX["central_fatigue"]] - 0.08 * hours, 0, 1))

    def describe(self) -> dict:
        return {
            "n_variables": N_INTEROCEPTION,
            "names": list(INTERO_NAMES),
            "systems": {
                "cardiovascular": 10, "respiratory": 7, "metabolic": 8,
                "thermoregulatory": 7, "gastrointestinal": 6, "renal": 6,
                "fatigue": 6, "immune": 4, "nociceptive": 5, "effort": 4,
                "endocrine": 4,
            },
        }
