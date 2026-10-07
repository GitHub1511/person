"""
Visualisation.

Produces the figures that make the vector space legible:

* ``fig_affect.png``       core affect, the full emotion set, and neuromodulators
* ``fig_interoception.png`` the internal milieu across organ systems
* ``fig_senses.png``       every exteroceptive channel over time
* ``fig_homunculus.png``   the tactile body map (a sensory homunculus) at
                           several instants, for force, pain and temperature
* ``fig_learning.png``     free energy, the body schema and the reward terms
* ``fig_motor.png``        balance strategies, joint angles and the policy timeline
* ``frames.png``           rendered views of the body
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from . import skin

CMAP_EMOTION = "magma"
CMAP_TACTILE = "inferno"


# --------------------------------------------------------------------------
def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=125, facecolor="#11151c")
    plt.close(fig)
    return path


def _style(ax, title, ylabel=None, xlabel="time (s)"):
    ax.set_facecolor("#171c25")
    ax.set_title(title, color="#e8eaf0", fontsize=9.5, pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, color="#aab2c0", fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color="#aab2c0", fontsize=8)
    ax.tick_params(colors="#8892a4", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#2c3542")
    ax.grid(alpha=0.15, color="#5a6473", linewidth=0.5)


def _fig(nrows, ncols, figsize, title=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor="#11151c", squeeze=False)
    if title:
        fig.suptitle(title, color="#f2f4f8", fontsize=12.5, y=0.995)
    return fig, axes


# ==========================================================================
# 1. Affect
# ==========================================================================
def fig_affect(npz: dict, out: Path) -> Path:
    t = npz["t"]
    emo = npz["emotions"]
    enames = [str(x) for x in npz["emotion_names"]]
    nm = npz["neuromodulators"]
    nnames = [str(x) for x in npz["neuromodulator_names"]]
    appr = npz["appraisal"]
    anames = [str(x) for x in npz["appraisal_names"]]

    fig, axes = _fig(3, 3, (16, 10), "Affect: core affect, emotions, neuromodulators")

    ax = axes[0, 0]
    for key, c in (("valence", "#4fc3f7"), ("arousal", "#ff8a65"),
                   ("dominance", "#81c784"), ("tension", "#ba68c8"),
                   ("stress", "#e57373")):
        ax.plot(t, npz[key], label=key, color=c, linewidth=1.3)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb", ncol=2)
    ax.set_ylim(-0.4, 1.3)
    _style(ax, "Core affect (PAD + tension/stress)")

    ax = axes[0, 1]
    ax.plot(t, npz["mood_valence"], color="#4fc3f7", label="mood valence")
    ax.plot(t, npz["mood_arousal"], color="#ff8a65", label="mood arousal")
    ax.plot(t, npz["mood_energy"], color="#ffd54f", label="mood energy")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Mood (slow) and allostatic load")
    ax2 = ax.twinx()
    ax2.plot(t, npz["allostatic_load"], color="#e57373", linestyle="--",
             linewidth=1.0, label="allostatic load")
    ax2.tick_params(colors="#e57373", labelsize=7)
    ax2.set_ylabel("allostatic load", color="#e57373", fontsize=8)

    ax = axes[0, 2]
    order = np.argsort(-emo.max(axis=0))[:8]
    for i in order:
        ax.plot(t, emo[:, i], label=enames[i], linewidth=1.2)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb", ncol=2)
    _style(ax, "Strongest emotions")

    ax = axes[1, 0]
    im = ax.imshow(emo.T, aspect="auto", origin="lower", cmap=CMAP_EMOTION,
                   extent=[t[0], t[-1], 0, len(enames)], vmin=0, vmax=1.0)
    ax.set_yticks(np.arange(len(enames)) + 0.5)
    ax.set_yticklabels(enames, fontsize=4.2)
    _style(ax, "All 28 emotion channels", "emotion", "time (s)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)

    ax = axes[1, 1]
    im = ax.imshow(nm.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[t[0], t[-1], 0, len(nnames)])
    ax.set_yticks(np.arange(len(nnames)) + 0.5)
    ax.set_yticklabels(nnames, fontsize=4.6)
    _style(ax, "All 25 neuromodulators", "neuromodulator", "time (s)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)

    ax = axes[1, 2]
    im = ax.imshow(appr.T, aspect="auto", origin="lower", cmap="coolwarm",
                   extent=[t[0], t[-1], 0, len(anames)], vmin=-1.2, vmax=1.2)
    ax.set_yticks(np.arange(len(anames)) + 0.5)
    ax.set_yticklabels(anames, fontsize=5.5)
    _style(ax, "Appraisal dimensions", "dimension", "time (s)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)

    ax = axes[2, 0]
    at = npz["action_tendency"]
    for i, nmv in enumerate(("approach", "avoid", "freeze", "attack",
                             "withdraw", "explore")):
        ax.plot(t, at[:, i], label=nmv, linewidth=1.2)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb", ncol=3)
    _style(ax, "Action tendencies")

    ax = axes[2, 1]
    ax.plot(t, npz["feeling_intensity"], color="#ffd54f", linewidth=1.3)
    _style(ax, "Feeling intensity (arousal + |valence| + emotion norm)")

    ax = axes[2, 2]
    top = np.argsort(-nm.mean(axis=0))[:6]
    for i in top:
        ax.plot(t, nm[:, i], label=nnames[i], linewidth=1.2)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Dominant neuromodulators (mean)")
    return _finish(fig, out)


# ==========================================================================
# 2. Interoception
# ==========================================================================
def fig_interoception(npz: dict, out: Path) -> Path:
    t = npz["t"]
    I = npz["interoception"]
    names = [str(x) for x in npz["interoception_names"]]
    idx = {n: i for i, n in enumerate(names)}

    groups = [
        ("Cardiovascular", ["heart_rate", "hrv_rmssd", "systolic_bp",
                            "diastolic_bp", "perfusion_skin"], "bpm / mmHg / a.u."),
        ("Respiratory", ["resp_rate", "tidal_volume", "minute_ventilation",
                         "spo2", "co2_arterial", "air_hunger"], "mixed"),
        ("Metabolic", ["glucose", "lactate", "glycogen", "atp_reserve",
                       "metabolic_rate"], "mmol/L / a.u. / W"),
        ("Thermoregulatory", ["core_temp", "skin_temp_mean", "shivering",
                              "sweating", "thermal_discomfort"], "degC / a.u."),
        ("Gastrointestinal & renal", ["gastric_fullness", "ghrelin", "leptin",
                                      "hydration", "osmolality",
                                      "bladder_fullness"], "a.u. / mOsm"),
        ("Fatigue & sleep", ["peripheral_fatigue", "central_fatigue",
                             "adenosine", "sleepiness", "alertness"], "a.u."),
        ("Nociception & effort", ["nociceptive_input", "pain_intensity",
                                  "pain_unpleasantness", "sense_of_effort",
                                  "breathlessness"], "a.u."),
        ("Immune & endocrine", ["cytokine", "inflammation", "sickness_behavior",
                                "adrenaline_h", "cortisol_h"], "a.u."),
    ]
    fig, axes = _fig(4, 2, (16, 12),
                     "Interoception: the internal milieu (67 channels)")
    for k, (title, keys, ylab) in enumerate(groups):
        ax = axes[k // 2][k % 2]
        for key in keys:
            if key in idx:
                ax.plot(t, I[:, idx[key]], label=key, linewidth=1.2)
        ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb", ncol=2)
        _style(ax, title, ylab)
    return _finish(fig, out)


# ==========================================================================
# 3. Senses
# ==========================================================================
def fig_senses(npz: dict, out: Path) -> Path:
    t = npz["t"]
    fig, axes = _fig(4, 3, (16, 12), "Exteroception and the afferent stream")

    ax = axes[0, 0]
    ax.plot(t, npz["touch_intensity"], color="#4fc3f7", label="touch intensity")
    ax.plot(t, npz["pain_total"], color="#e57373", label="pain")
    ax.plot(t, npz["itch"], color="#ba68c8", label="itch")
    ax.plot(t, npz["affective_touch"], color="#81c784", label="C-tactile")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Somatosensory summary (N)")

    ax = axes[0, 1]
    ax.plot(t, npz["contact_taxels"], color="#ffd54f")
    _style(ax, "Taxels in contact", "count")

    ax = axes[0, 2]
    ax.plot(t, npz["loudness"], color="#4fc3f7", label="loudness")
    ax.plot(t, npz["luminance"], color="#ffd54f", label="luminance")
    ax.plot(t, npz["pupil"] / 8.0, color="#81c784", label="pupil/8")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Vision and audition")

    ax = axes[1, 0]
    v = npz["vestibular"]
    for i, nm in enumerate(("canal_x", "canal_y", "canal_z")):
        ax.plot(t, v[:, i], label=nm, linewidth=1.2)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Semicircular canals (rad/s)")

    ax = axes[1, 1]
    for i, nm in enumerate(("otolith_x", "otolith_y", "otolith_z")):
        ax.plot(t, v[:, 6 + i], label=nm, linewidth=1.2)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Otolith organs (m/s^2)")

    ax = axes[1, 2]
    for i, nm in enumerate(("gravity_x", "gravity_y", "gravity_z")):
        ax.plot(t, v[:, 9 + i], label=nm, linewidth=1.2)
    ax.plot(t, v[:, 12], label="tilt", color="#e57373", linewidth=1.0)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Gravity direction in head frame / tilt")

    ax = axes[2, 0]
    ax.plot(t, npz["balance_error"], color="#e57373", label="|COM - support|")
    ax.plot(t, npz["postural_priority"] if "postural_priority" in npz
            else npz["balance_priority"], color="#81c784",
            label="postural priority (trunk+legs)")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Balance: capture-point error and postural priority")

    ax = axes[2, 1]
    olf = npz["olfactory"]
    if olf.ndim == 2 and olf.shape[1] > 0:
        active = np.argsort(-np.abs(olf).max(axis=0))[:6]
        for i in active:
            ax.plot(t, olf[:, i], label=f"odor {i}", linewidth=1.2)
        ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Olfactory channels (strongest 6)")

    ax = axes[2, 2]
    g = npz["gustatory"]
    if g.ndim == 2 and g.shape[1] == 5:
        for i, nm in enumerate(("sweet", "salty", "sour", "bitter", "umami")):
            ax.plot(t, g[:, i], label=nm, linewidth=1.2)
        ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Gustatory channels", "a.u.")

    ax = axes[3, 0]
    q = npz["proprio"]
    sel = np.argsort(-np.abs(q - q[0]).max(axis=0))[:6]
    for i in sel:
        ax.plot(t, q[:, i], linewidth=1.1)
    _style(ax, "Most-active joint angles (rad)")

    ax = axes[3, 1]
    st = npz["strategy_ankle"]
    ax.plot(t, npz["strategy_ankle"], label="ankle", color="#4fc3f7")
    ax.plot(t, npz["strategy_hip"], label="hip", color="#ff8a65")
    ax.plot(t, npz["strategy_toe"], label="toe", color="#ffd54f")
    ax.plot(t, npz["strategy_knee"], label="knee", color="#81c784")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb", ncol=2)
    _style(ax, "Balance strategy recruitment")

    ax = axes[3, 2]
    ax.plot(t, npz["pelvis_z"], color="#4fc3f7", label="pelvis z")
    ax.plot(t, npz["com_z"], color="#ffd54f", label="COM z")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Body height (m)")
    return _finish(fig, out)


# ==========================================================================
# 4. Sensory homunculus
# ==========================================================================
def fig_homunculus(npz: dict, out: Path) -> Path:
    """The tactile body map, drawn at the taxels' schematic body positions."""
    u = npz["taxel_u"]
    v = npz["taxel_v"]
    snaps = npz["taxel_snapshot"]           # (S, n_taxels, 26)
    ts = npz["taxel_snapshot_t"]
    ch = {str(n): i for i, n in enumerate(npz["tactile_channel_names"])}
    S = snaps.shape[0]
    # (key, label, colormap, fixed range or None, log scale?)
    panels = [("normal_force", "contact force (N)", CMAP_TACTILE, None, False),
              ("pressure", "pressure (kPa)", "inferno", None, False),
              ("noci_mech", "mechanonociceptor", "hot", (0.0, 0.6), False),
              ("temperature", "skin temperature (degC)", "coolwarm",
               (28.0, 34.0), False),
              # Colormaps must be dark at zero, otherwise "no signal" renders as
              # the brightest colour on the body and the figure reads backwards.
              ("sa1", "SA-I (sustained pressure)", "plasma", (0.0, 1.1), False),
              ("ct", "C-tactile (affective touch)", "viridis", (0.0, 0.6), False)]

    fig, axes = _fig(S, len(panels), (2.25 * len(panels), 2.6 * S),
                     "Sensory homunculus: the tactile body map "
                     "(1872 taxels, 26 receptor channels each)")
    for row in range(S):
        for col, (key, label, cmap, lim, use_log) in enumerate(panels):
            ax = axes[row][col]
            vals = snaps[row][:, ch[key]]
            if lim is not None:
                vmin, vmax = lim
            else:
                # Only a handful of taxels are ever in contact, so a
                # percentile-based upper limit collapses: force the floor to
                # zero so "no contact" is uniformly dark and any contact reads
                # as bright.
                vmin = 0.0
                vmax = max(float(np.percentile(vals, 99.5)),
                           float(vals.max()) * 0.5, 1e-3)
            sc = ax.scatter(u, v, c=vals, s=17, cmap=cmap, vmin=vmin, vmax=vmax,
                            marker="s", linewidths=0.15,
                            edgecolors="#0b0e13")
            ax.set_facecolor("#0d1117")
            ax.set_xlim(0.13, 0.87)
            ax.set_ylim(-0.03, 1.02)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#2c3542")
            if row == 0:
                ax.set_title(label, color="#e8eaf0", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"t = {ts[row]:.2f} s", color="#aab2c0", fontsize=8)
            if col == len(panels) - 1:
                cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
                cb.ax.tick_params(labelsize=5, colors="#8892a4")
    return _finish(fig, out)


# ==========================================================================
# 5. Learning / predictive coding
# ==========================================================================
def fig_learning(npz: dict, out: Path) -> Path:
    t = npz["t"]
    fig, axes = _fig(3, 3, (16, 10),
                     "Predictive coding, body schema and intrinsic reward")

    ax = axes[0, 0]
    ax.plot(t, npz["free_energy"], color="#4fc3f7", label="free energy")
    ax.plot(t, npz["inaccuracy"], color="#ff8a65", label="inaccuracy")
    ax.plot(t, npz["complexity"], color="#81c784", label="complexity")
    ax.set_yscale("symlog")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Variational free energy")

    ax = axes[0, 1]
    ax.plot(t, npz["free_energy_norm"], color="#4fc3f7")
    _style(ax, "Free energy per sensory dimension")

    ax = axes[0, 2]
    ax.plot(t, npz["proprio_drift"], color="#ba68c8", label="proprioceptive drift")
    ax.plot(t, npz["ownership"], color="#81c784", label="body ownership (R^2)")
    ax.plot(t, npz["empowerment"], color="#ffd54f", label="empowerment")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Body schema")

    ax = axes[1, 0]
    ax.plot(t, npz["surprise"], color="#e57373")
    _style(ax, "Surprise")

    ax = axes[1, 1]
    rw = npz["reward_terms"]
    names = [str(x) for x in npz["reward_term_names"]]
    for i in range(rw.shape[1] - 1):
        ax.plot(t, rw[:, i], label=names[i], linewidth=1.1)
    ax.legend(fontsize=6, facecolor="#171c25", labelcolor="#c8cfdb", ncol=2)
    _style(ax, "Intrinsic reward terms")

    ax = axes[1, 2]
    ax.plot(t, npz["reward"], color="#4fc3f7", label="total reward")
    ax.plot(t, npz["rpe"], color="#ff8a65", label="reward prediction error")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Reward and RPE")

    ax = axes[2, 0]
    ax.plot(t, npz["inverse_error"], color="#ba68c8")
    _style(ax, "Inverse-model error (a.u.)")

    ax = axes[2, 1]
    ax.plot(t, npz["drive_pressure"], color="#e57373")
    _style(ax, "Total homeostatic drive pressure")

    ax = axes[2, 2]
    ax.plot(t, npz["torque_effort"], color="#ffd54f", label="torque effort")
    ax.plot(t, npz["jerk"], color="#4fc3f7", label="jerk")
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb")
    _style(ax, "Effort and jerk")
    return _finish(fig, out)


# ==========================================================================
# 6. Drives and policies
# ==========================================================================
def fig_drives(npz: dict, out: Path) -> Path:
    t = npz["t"]
    dl = npz["drive_level"]
    dn = [str(x) for x in npz["drive_names"]]
    fig, axes = _fig(2, 2, (15, 8), "Homeostatic drives and behaviour selection")

    ax = axes[0, 0]
    im = ax.imshow(dl.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[t[0], t[-1], 0, len(dn)], vmin=0, vmax=1.0)
    ax.set_yticks(np.arange(len(dn)) + 0.5)
    ax.set_yticklabels(dn, fontsize=7)
    _style(ax, "All 16 drive levels", "drive", "time (s)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)

    ax = axes[0, 1]
    for i in np.argsort(-dl.mean(axis=0))[:7]:
        ax.plot(t, dl[:, i], label=dn[i], linewidth=1.2)
    ax.legend(fontsize=7, facecolor="#171c25", labelcolor="#c8cfdb", ncol=2)
    _style(ax, "Strongest drives")

    ax = axes[1, 0]
    pol = npz["policy"]
    uniq = sorted(set(str(p) for p in pol))
    idx = {p: i for i, p in enumerate(uniq)}
    codes = np.array([idx[str(p)] for p in pol])
    ax.plot(t, codes, drawstyle="steps-post", color="#4fc3f7", linewidth=1.2)
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels(uniq, fontsize=6)
    _style(ax, "Selected motor program", "policy", "time (s)")

    ax = axes[1, 1]
    counts = [int((codes == k).sum()) for k in range(len(uniq))]
    order = np.argsort(-np.array(counts))
    ax.barh([uniq[k] for k in order][:18],
            [counts[k] for k in order][:18], color="#81c784")
    ax.tick_params(colors="#8892a4", labelsize=6.5)
    for s in ax.spines.values():
        s.set_color("#2c3542")
    ax.set_facecolor("#171c25")
    ax.set_title("Policy histogram", color="#e8eaf0", fontsize=9.5)
    ax.set_xlabel("logged ticks", color="#aab2c0", fontsize=8)
    return _finish(fig, out)


# ==========================================================================
# 7. Rendered frames
# ==========================================================================
def render_frames(agent, out: Path, n: int = 4, duration: float = 2.0,
                  camera: str = "closeup", size=(480, 360)) -> Path:
    """Run briefly and render the body from a fixed camera."""
    import mujoco
    w, h = size
    renderer = mujoco.Renderer(agent.model, h, w)
    # Hiding site group 0 removes the rangefinder rays, which are drawn as long
    # yellow lines across the body and otherwise dominate the image.
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    agent.reset()
    frames = []
    per = max(int(duration / agent.dt / n), 1)
    for i in range(per * n):
        agent.step()
        if i % per == per - 1:
            renderer.update_scene(agent.data, camera=camera, scene_option=opt)
            frames.append(renderer.render().copy())

    fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 2.7),
                             facecolor="#11151c", squeeze=False)
    for k, fr in enumerate(frames):
        axes[0][k].imshow(fr)
        axes[0][k].set_xticks([])
        axes[0][k].set_yticks([])
        axes[0][k].set_title(f"t = {(k + 1) * duration / len(frames):.2f} s",
                             color="#c8cfdb", fontsize=9)
        for s in axes[0][k].spines.values():
            s.set_color("#2c3542")
    fig.suptitle(f"Embodied human in MuJoCo  (camera: {camera})",
                 color="#f2f4f8", fontsize=12)
    return _finish(fig, out)


# ==========================================================================
def _normalise(npz: dict) -> dict:
    """Accept either channel-name layout.

    Scalar channels were once written with a ``ch_`` prefix to keep them from
    clashing with the dense arrays; they now use plain names.  Normalising here
    means the figures work with episodes recorded by either version.
    """
    out = dict(npz)
    for k in list(out.keys()):
        if k.startswith("ch_"):
            out.setdefault(k[3:], out[k])
    return out


def build_all(npz_path: Path, out_dir: Path) -> list[Path]:
    npz = _normalise(dict(np.load(npz_path, allow_pickle=False)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn, name in ((fig_affect, "fig_affect"),
                     (fig_interoception, "fig_interoception"),
                     (fig_senses, "fig_senses"),
                     (fig_homunculus, "fig_homunculus"),
                     (fig_learning, "fig_learning"),
                     (fig_drives, "fig_drives")):
        try:
            made.append(fn(npz, out_dir / f"{name}.png"))
        except Exception as exc:      # keep going; report the failure
            print(f"  ! {name} failed: {type(exc).__name__}: {exc}")
    return made
