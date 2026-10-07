#!/usr/bin/env python
"""
Run the embodied human.

    python run_sim.py                      # 10 s episode + figures
    python run_sim.py --duration 20 --render
    python run_sim.py --no-vision          # much faster, no retina rendering
    python run_sim.py --describe           # just print the vector space

Outputs land in ``out/``:
    episode.npz                 every layer, compressed
    episode_channels.csv        headline scalar channels
    episode_tactile_regions.csv per-region tactile aggregates (long format)
    episode_manifest.json       the full description of the model and its senses
    fig_*.png                   figures
    frames.png                  rendered views
    models/human.xml            the generated MJCF, for inspection
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.record import EpisodeRecorder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate an embodied human in MuJoCo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--duration", type=float, default=10.0,
                   help="episode length in seconds")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--log-every", type=int, default=10,
                   help="log one frame every N physics steps")
    p.add_argument("--no-vision", action="store_true",
                   help="disable offscreen retina rendering (much faster)")
    p.add_argument("--no-touch-sensors", action="store_true",
                   help="omit the 1872 native MuJoCo touch sensors")
    p.add_argument("--render", action="store_true",
                   help="save rendered frames of the body")
    p.add_argument("--figures", action="store_true", default=True)
    p.add_argument("--no-figures", dest="figures", action="store_false")
    p.add_argument("--describe", action="store_true",
                   help="print the vector space and exit")
    p.add_argument("--quiet", action="store_true")
    return p


def print_description(d: dict) -> None:
    m = d["model"]
    print("=" * 78)
    print("EMBODIED HUMAN - model")
    print("=" * 78)
    for k, v in m.items():
        print(f"  {k:22s} {v}")

    print("\n" + "=" * 78)
    print("SENSORY VECTOR SPACE")
    print("=" * 78)
    print(f"  {'modality':<26}{'channels':>10}   detail")
    print("  " + "-" * 74)
    tax = d["total_scalar_state"]
    s = d["senses"]
    rows = [
        ("Touch (tactile)", tax["tactile_channels"],
         f"{s['tactile']['n_taxels']} taxels x "
         f"{s['tactile']['channels_per_taxel']} receptor channels"),
        ("Proprioception", tax["proprioceptive_channels"],
         f"{s['proprioceptive']['n_joints']} joints x "
         f"{s['proprioceptive']['channels_per_joint']} channels"),
        ("Vestibular", tax["vestibular"], "canals + otoliths + magnetometer"),
        ("Vision", tax["visual"], f"{s['visual']['retina'][1]}x"
         f"{s['visual']['retina'][0]} retina, fovea, motion, pupil"),
        ("Audition", tax["auditory"], f"{s['auditory']['bands']}-band "
         "cochlear filterbank"),
        ("Olfaction", tax["olfactory"], "odour channels"),
        ("Gustation", tax["gustatory"], "sweet/salty/sour/bitter/umami"),
        ("Interoception", tax["interoceptive"], "11 organ systems"),
    ]
    for name, n, detail in rows:
        print(f"  {name:<26}{n:>10,}   {detail}")
    print("  " + "-" * 74)
    print(f"  {'TOTAL sensory scalars':<26}{tax['total']:>10,}")
    print(f"  {'latent (learned) space':<26}{d['latent']['dim']:>10,}")

    print("\n" + "=" * 78)
    print("AFFECTIVE VECTOR SPACE")
    print("=" * 78)
    a = d["affect"]
    print(f"  appraisal dimensions  ({len(a['appraisal_dimensions'])}): "
          f"{', '.join(a['appraisal_dimensions'])}")
    print(f"  emotions              ({a['n_emotions']}): {', '.join(a['emotions'])}")
    print(f"  neuromodulators       ({a['n_neuromodulators']}): "
          f"{', '.join(a['neuromodulators'])}")
    print(f"  core affect           : {', '.join(a['core_affect'])}")
    print(f"  temperament           : {', '.join(a['temperament_traits'])}")
    print(f"  action tendencies     : {', '.join(a['action_tendencies'])}")
    print(f"  drives ({d['drives']['n_drives']}): {', '.join(d['drives']['drives'])}")

    print("\n" + "=" * 78)
    print("RATES, AFFERENTS, COGNITION")
    print("=" * 78)
    for k, v in d["rates_hz"].items():
        print(f"  {k:16s} {v:8.1f} Hz")
    print(f"  afferent delays (ms)   {d['afferents']['delays_ms']}")
    print(f"  firing range (Hz)      {d['afferents']['rate_range_hz']}")
    print(f"  policies               {d['inference']['n_policies']} "
          f"({d['inference']['categories']})")
    print(f"  RLS forward model      {d['prediction']['forward_params']:,} params")
    print(f"  RLS inverse model      {d['prediction']['inverse_params']:,} params")
    print(f"  body schema            {d['prediction']['schema_params']:,} params")
    print(f"  reward terms           {', '.join(d['motivation']['terms'])}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = SimConfig(duration=args.duration, seed=args.seed,
                    out_dir=args.out, log_every=args.log_every)
    if args.no_vision:
        cfg.vision.enabled = False

    t0 = time.perf_counter()
    agent = EmbodiedHuman(cfg, vision=not args.no_vision,
                          touch_sensors=not args.no_touch_sensors)
    d = agent.describe()
    print(f"model built in {time.perf_counter() - t0:.2f} s")
    print_description(d)

    if args.describe:
        return 0

    rec = EpisodeRecorder(out_dir=args.out, log_every=args.log_every,
                          duration=args.duration)
    rec.configure(agent, duration=args.duration)

    print("\n" + "=" * 78)
    print(f"RUNNING {args.duration:.1f} s  "
          f"({int(args.duration / agent.dt):,} physics steps)")
    print("=" * 78)
    t0 = time.perf_counter()
    n = int(round(args.duration / agent.dt))
    for i in range(n):
        agent.step()
        if i % args.log_every == 0:
            rec.log(agent, agent.snapshot())
            rec.maybe_snapshot_tactile(agent)
        if not args.quiet and i % max(n // 12, 1) == 0:
            a = agent.affect_frame
            st = agent.state
            fe = agent.pred_frame.free_energy_norm if agent.pred_frame else 0.0
            pain = agent.frame.pain_total if agent.frame else 0.0
            print(f"  t={agent.t:5.2f}s  z={st.root_pos[2]:.3f}  "
                  f"bal={agent.motor.balance_error:.3f}  "
                  f"pol={agent.inference.current.name:16s} "
                  f"val={a.valence:+.2f} aro={a.arousal:.2f} "
                  f"FE={fe:6.3f}  pain={pain:.2f}")
    wall = time.perf_counter() - t0

    files = rec.save(prefix="episode")
    summary = rec.summary()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k in ("duration_s", "fell", "min_pelvis_z", "mean_balance_error",
              "max_balance_error", "max_pain", "self_touch_fraction"):
        v = summary.get(k)
        print(f"  {k:26s} {v}")
    print(f"  {'mean valence':26s} {summary['mean_valence']:+.4f}")
    print(f"  {'mean arousal':26s} {summary['mean_arousal']:.4f}")
    print(f"  {'mean free energy':26s} {summary['mean_free_energy']:.3f}")
    print(f"  {'mean reward':26s} {summary['mean_reward']:+.5f}")
    print(f"  {'proprioceptive drift':26s} {summary['final_proprio_drift']:.5f}")
    print(f"  {'body ownership':26s} {summary['final_ownership']:.4f}")
    print(f"  {'empowerment':26s} {summary['final_empowerment']:.4f}")
    print(f"  {'emotions (mean)':26s} "
          f"{', '.join(f'{n}={v:.3f}' for n, v in summary['top_emotions'])}")
    print(f"  {'emotions (peak)':26s} "
          f"{', '.join(f'{n}={v:.3f}' for n, v in summary['top_emotions_peak'])}")
    print(f"  {'policies used':26s} {len(summary['policies_used'])}")
    print(f"  {'top policies':26s} "
          f"{sorted(summary['policy_histogram'].items(), key=lambda kv: -kv[1])[:6]}")
    print(f"\n  wall clock {wall:.1f} s for {args.duration:.1f} s of simulation "
          f"({args.duration / max(wall, 1e-9):.2f}x realtime)")

    print("\nFILES")
    for k, p in files.items():
        print(f"  {k:14s} {p}")

    if args.figures:
        from embodied_human.plots import build_all
        print("\nFIGURES")
        made = build_all(files["npz"], args.out)
        for p in made:
            print(f"  {p}")

    if args.render:
        from embodied_human.plots import render_frames
        agent.reset()
        p = render_frames(agent, args.out / "frames.png", n=5, duration=5.0,
                          camera="observer", size=(420, 560))
        print(f"  {p}")

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
