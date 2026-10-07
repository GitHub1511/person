"""
Episode recording.

Captures every layer of the agent at a fixed logging rate and writes a
compressed ``.npz`` (for analysis), a human-readable ``.csv`` of the headline
channels, and a JSON manifest describing the whole vector space.

The tactile bank is far too large to store in full at every tick
(1872 x 26 values), so region aggregates are stored continuously and the full
taxel image is snapshotted at a few instants for the body-map figures.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import receptors as R
from . import skin
from .affect import APPRAISAL_DIMS, EMOTIONS, NEUROMODULATORS
from .drives import DRIVES
from .interoception import INTERO_NAMES
from .intrinsic import REWARD_TERMS
from .receptors import TACTILE_CHANNELS


@dataclass
class EpisodeRecorder:
    """Accumulates logged frames and writes them out."""

    out_dir: Path
    log_every: int = 10
    n_taxel_snapshots: int = 6
    duration: float = 10.0

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.steps: list = []
        self._tactile_full: list[np.ndarray] = []
        self._tactile_full_t: list[float] = []
        # Snapshot the full taxel bank at evenly spaced instants across the
        # episode.  Taking them eagerly at the start (the obvious way) captures
        # only the first milliseconds, before any thermal or contact dynamics
        # have developed, and every panel then looks identical.
        self._set_snap_times(self.duration)
        self.region_names: list[str] = []
        self.meta: dict = {}

    def _set_snap_times(self, duration: float) -> None:
        self.duration = float(duration)
        n = max(int(self.n_taxel_snapshots), 1)
        self._snap_times = [self.duration * (k + 1) / (n + 1) for k in range(n)]
        self._snap_next = 0

    # ------------------------------------------------------------------
    def configure(self, agent, duration: float | None = None) -> None:
        self.agent = agent
        if duration is not None:
            self._set_snap_times(duration)
        self.region_names = list(R.REGION_ORDER)
        self.meta = agent.describe()

    # ------------------------------------------------------------------
    def log(self, agent, step) -> None:
        """Record one logged frame."""
        f = agent.frame
        a = agent.affect_frame
        d = agent.drive_frame
        p = agent.pred_frame
        rw = agent.reward_frame
        mf = agent.motor_frame
        st = agent.state

        # per-region tactile aggregate: 5 informative channels out of 26
        agg = f.tactile_by_region
        region_mat = agg.get("_region5")
        if region_mat is None:
            region_mat = np.zeros((len(R.REGION_ORDER), 5))
        region_mat = np.asarray(region_mat, float).copy()

        self.rows.append(dict(
            t=agent.t,
            # body
            pelvis_z=float(st.root_pos[2]),
            com_x=float(st.com[0]), com_y=float(st.com[1]), com_z=float(st.com[2]),
            com_vx=float(st.com_vel[0]), com_vy=float(st.com_vel[1]),
            support_x=float(st.support_center[0]), support_y=float(st.support_center[1]),
            balance_error=float(agent.motor.balance_error),
            fallen=bool(st.fallen), n_contact=int(st.n_contact),
            self_touch=bool(st.self_touch),
            torque_effort=float(st.torque_effort),
            mechanical_power=float(st.mechanical_power),
            # senses
            touch_intensity=float(f.touch_intensity),
            pain_mech=float(f.pain_mech), pain_heat=float(f.pain_heat),
            pain_cold=float(f.pain_cold), pain_total=float(f.pain_total),
            itch=float(f.itch_total),
            affective_touch=float(f.affective_touch),
            contact_taxels=int(f.contact_mask.sum()) if f.contact_mask.size else 0,
            # affect
            valence=float(a.valence), arousal=float(a.arousal),
            dominance=float(a.dominance), tension=float(a.tension),
            stress=float(a.stress), mood_valence=float(a.mood_valence),
            mood_arousal=float(a.mood_arousal), mood_energy=float(a.mood_energy),
            allostatic_load=float(a.allostatic_load),
            feeling_intensity=float(a.feeling_intensity),
            dominant_emotion=a.dominant_emotion,
            # cognition
            policy=agent.inference.current.name,
            policy_prob=float(agent.inference.last_probs.max()),
            free_energy=float(p.free_energy) if p else 0.0,
            free_energy_norm=float(p.free_energy_norm) if p else 0.0,
            inaccuracy=float(p.inaccuracy) if p else 0.0,
            complexity=float(p.complexity) if p else 0.0,
            surprise=float(p.surprise) if p else 0.0,
            proprio_drift=float(p.proprioceptive_drift) if p else 0.0,
            empowerment=float(p.empowerment) if p else 0.0,
            # `body_ownership` is empty on the very first predictive tick, before
            # the forward model has produced an error to attribute
            ownership=(float(p.body_ownership.mean())
                       if p is not None and p.body_ownership.size else 0.0),
            inverse_error=float(p.inverse_error) if p else 0.0,
            reward=float(rw.total) if rw else 0.0,
            rpe=float(rw.rpe) if rw else 0.0,
            drive_pressure=float(d.total_pressure) if d else 0.0,
            most_urgent_drive=d.most_urgent if d else "",
            # motor
            strategy_ankle=float(mf.strategy_weight[0]) if mf else 0.0,
            strategy_hip=float(mf.strategy_weight[1]) if mf else 0.0,
            strategy_toe=float(mf.strategy_weight[2]) if mf else 0.0,
            strategy_knee=float(mf.strategy_weight[3]) if mf else 0.0,
            balance_priority=float(agent.motor.postural_priority),
            stepping=agent.motor.step_state,
            jerk=float(mf.jerk) if mf else 0.0,
            # vision / audio
            luminance=float(f.visual[0]), pupil=float(f.visual[7]),
            blink=float(f.visual[8]), saccade=float(f.visual[9]),
            loudness=float(f.auditory[0]),
        ))
        # dense arrays
        self.steps.append(dict(
            t=agent.t,
            emotions=a.emotions.copy() if a else np.zeros(len(EMOTIONS)),
            neuromodulators=a.neuromodulators.copy() if a else np.zeros(len(NEUROMODULATORS)),
            appraisal=a.appraisal.copy() if a else np.zeros(len(APPRAISAL_DIMS)),
            action_tendency=a.action_tendency.copy() if a else np.zeros(6),
            drive_level=d.level.copy() if d else np.zeros(len(DRIVES)),
            drive_urgency=d.urgency.copy() if d else np.zeros(len(DRIVES)),
            interoception=agent.interoception.s.copy(),
            region_tactile=region_mat,
            reward_terms=rw.terms.copy() if rw else np.zeros(len(REWARD_TERMS)),
            vestibular=f.vestibular.copy(),
            visual=f.visual.copy(),
            auditory=f.auditory.copy(),
            olfactory=f.olfactory.copy(),
            gustatory=f.gustatory.copy(),
            proprio=agent.state.q.copy(),
            proprio_vel=agent.state.qd.copy(),
        ))

    # ------------------------------------------------------------------
    def maybe_snapshot_tactile(self, agent) -> None:
        """Store the full taxel bank at the pre-scheduled instants."""
        if self._snap_next >= len(self._snap_times):
            return
        if agent.t < self._snap_times[self._snap_next]:
            return
        if agent.frame is None:
            return
        self._tactile_full.append(agent.frame.tactile.copy())
        self._tactile_full_t.append(agent.t)
        self._snap_next += 1

    # ------------------------------------------------------------------
    def save(self, prefix: str = "episode") -> dict[str, Path]:
        out = {}
        arrays = {}
        if self.steps:
            for key in self.steps[0]:
                if key == "t":
                    arrays["t"] = np.array([s["t"] for s in self.steps])
                else:
                    arrays[key] = np.stack([s[key] for s in self.steps])
        # Scalar headline channels are stored under their own names so the
        # figures and any downstream analysis can read them straight from the
        # npz instead of re-parsing the CSV.  None of these names collide with
        # the dense arrays collected above.
        if self.rows:
            for key, val in self.rows[0].items():
                if isinstance(val, (bool, np.bool_)):
                    arrays[key] = np.array([bool(r[key]) for r in self.rows])
                elif isinstance(val, str):
                    arrays[key] = np.array([str(r[key]) for r in self.rows])
                elif isinstance(val, (int, float, np.floating, np.integer)):
                    arrays[key] = np.array([float(r[key]) for r in self.rows])
            arrays["row_t"] = np.array([float(r["t"]) for r in self.rows])
        arrays["taxel_snapshot_t"] = np.array(self._tactile_full_t)
        if self._tactile_full:
            arrays["taxel_snapshot"] = np.stack(self._tactile_full)
        arrays["taxel_u"] = np.array([t.u for t in skin.TAXELS])
        arrays["taxel_v"] = np.array([t.v for t in skin.TAXELS])
        arrays["taxel_region"] = np.array([t.region for t in skin.TAXELS])
        arrays["region_names"] = np.array(self.region_names)
        arrays["emotion_names"] = np.array(EMOTIONS)
        arrays["neuromodulator_names"] = np.array(NEUROMODULATORS)
        arrays["appraisal_names"] = np.array(APPRAISAL_DIMS)
        arrays["drive_names"] = np.array(DRIVES)
        arrays["interoception_names"] = np.array(INTERO_NAMES)
        arrays["reward_term_names"] = np.array(REWARD_TERMS)
        arrays["tactile_channel_names"] = np.array(TACTILE_CHANNELS)

        npz = self.out_dir / f"{prefix}.npz"
        np.savez_compressed(npz, **arrays)
        out["npz"] = npz

        # ---- flat CSV of the scalar headline channels ------------------
        csv_path = self.out_dir / f"{prefix}_channels.csv"
        scalar_keys = [k for k, v in self.rows[0].items()
                       if isinstance(v, (int, float, bool, np.floating, np.integer))]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=scalar_keys, extrasaction="ignore")
            w.writeheader()
            for row in self.rows:
                w.writerow({k: row[k] for k in scalar_keys})
        out["csv"] = csv_path

        # ---- full per-channel CSV is too wide; write a long-format one for
        #      the region aggregates so pandas users have something usable
        if self.steps:
            import pandas as pd
            recs = []
            for i, s in enumerate(self.steps):
                mat = s["region_tactile"]
                for j, rn in enumerate(self.region_names):
                    recs.append((s["t"], rn, mat[j, 0], mat[j, 1], mat[j, 2],
                                 mat[j, 3], mat[j, 4]))
            df = pd.DataFrame(recs, columns=["t", "region", "force_N", "peak_kPa",
                                             "sa1", "ct", "temp_C"])
            long_csv = self.out_dir / f"{prefix}_tactile_regions.csv"
            df.to_csv(long_csv, index=False)
            out["tactile_csv"] = long_csv

        manifest = self.out_dir / f"{prefix}_manifest.json"
        manifest.write_text(json.dumps(self.meta, indent=2, default=str),
                            encoding="utf-8")
        out["manifest"] = manifest
        return out

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        if not self.rows:
            return {}
        r = self.rows
        ges = np.array([s["emotions"] for s in self.steps])
        nm = np.array([s["neuromodulators"] for s in self.steps])
        dr = np.array([s["drive_level"] for s in self.steps])
        val = np.array([x["valence"] for x in r])
        pol = [x["policy"] for x in r]
        return {
            "duration_s": float(r[-1]["t"]),
            "n_logged": len(r),
            "fell": bool(any(x["fallen"] for x in r)),
            "min_pelvis_z": float(min(x["pelvis_z"] for x in r)),
            "mean_balance_error": float(np.mean([x["balance_error"] for x in r])),
            "max_balance_error": float(np.max([x["balance_error"] for x in r])),
            "touch_total": float(np.mean([x["touch_intensity"] for x in r])),
            "max_pain": float(np.max([x["pain_total"] for x in r])),
            "mean_valence": float(val.mean()), "min_valence": float(val.min()),
            "max_valence": float(val.max()),
            "mean_arousal": float(np.mean([x["arousal"] for x in r])),
            "mean_stress": float(np.mean([x["stress"] for x in r])),
            "mean_free_energy": float(np.mean([x["free_energy"] for x in r])),
            "mean_reward": float(np.mean([x["reward"] for x in r])),
            "final_proprio_drift": float(r[-1]["proprio_drift"]),
            "final_ownership": float(r[-1]["ownership"]),
            "final_empowerment": float(r[-1]["empowerment"]),
            "top_emotions": [(EMOTIONS[i], float(ges[:, i].mean()))
                             for i in np.argsort(-ges.mean(axis=0))[:6]],
            "top_emotions_peak": [(EMOTIONS[i], float(ges[:, i].max()))
                                  for i in np.argsort(-ges.max(axis=0))[:6]],
            "neuromodulator_means": {n: float(nm[:, i].mean())
                                     for i, n in enumerate(NEUROMODULATORS)},
            "drive_means": {n: float(dr[:, i].mean())
                            for i, n in enumerate(DRIVES)},
            "policies_used": sorted(set(pol)),
            "policy_histogram": {p: pol.count(p) for p in sorted(set(pol))},
            "self_touch_fraction": float(np.mean([x["self_touch"] for x in r])),
        }
