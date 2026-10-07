#!/usr/bin/env python
"""
Run the embodied human with a mind behind it.

    python run_mind.py                          # window, scripted stand-in mind
    python run_mind.py --backend server         # AZR served by llama-server / vLLM / Ollama
    python run_mind.py --backend llamacpp --model-path azr-coder-3b-Q4_K_M.gguf
    python run_mind.py --backend hf --azr 3b    # transformers (4-bit)

    python run_mind.py --headless 40 --say "5:hello" --say "15:please pick up the apple"
                                                # no window: write frames to out/mind/

Talk to the person by typing in the box under the picture (or ``--say``).
Whatever the person *thinks* appears above its head in grey italics; whatever it
chooses to *say* appears in a white bubble and moves the jaw.  Nothing makes it
speak: it is free to stay silent, and usually will.

See ``setup_azr.py`` for getting the AZR weights.  Nothing is downloaded unless you
ask for it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.mind import (AZR_MODELS, HFBackend, LlamaCppBackend, Mind,
                                 OpenAICompatBackend, StubBackend)


def make_backend(a):
    if a.backend == "stub":
        return StubBackend(seed=a.seed)
    if a.backend == "server":
        return OpenAICompatBackend(base_url=a.url, model=a.model_name,
                                   api_key=a.api_key or None)
    if a.backend == "llamacpp":
        if not a.model_path:
            sys.exit("--model-path is required for --backend llamacpp")
        return LlamaCppBackend(a.model_path)
    if a.backend == "hf":
        return HFBackend(AZR_MODELS[a.azr], load_in_4bit=not a.fp16)
    raise SystemExit(f"unknown backend {a.backend}")


def build_agent(a) -> EmbodiedHuman:
    cfg = SimConfig(seed=a.seed, out_dir=Path(a.out))
    # the live app does not need the sensory stack at its full rates; the
    # reflexes that depend on them (withdrawal, startle) are slower than this
    cfg.rates.receptor = 100.0
    cfg.rates.afferent = 100.0
    cfg.rates.interoception = 50.0
    agent = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    agent.autonomous = False           # the mind decides what to do
    agent.gait.hold_stance = True      # whole-body control holds the stance
    agent.skills.speech.voice = a.voice
    if a.at_table:
        import mujoco
        ra = agent.meta.root_qpos_addr
        agent.data.qpos[ra + 1] -= 0.32
        mujoco.mj_forward(agent.model, agent.data)
    return agent


def build_parser_for_tests(argv):
    return _parser().parse_args(argv)


def main(argv=None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    return run(a)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["stub", "server", "llamacpp", "hf"], default="stub")
    p.add_argument("--azr", choices=list(AZR_MODELS), default="3b")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model-name", default="azr")
    p.add_argument("--api-key", default="", help="bearer token, if the server wants one")
    p.add_argument("--model-path", default="")
    p.add_argument("--fp16", action="store_true", help="hf backend: no 4-bit quantisation")
    p.add_argument("--voice", action="store_true", help="also speak aloud (Windows SAPI)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", default="out")
    p.add_argument("--headless", type=float, default=0.0,
                   help="run this many simulated seconds with no window and save frames")
    p.add_argument("--say", action="append", default=[],
                   help="headless: 'seconds:text' spoken to the person")
    p.add_argument("--frame-every", type=float, default=1.0)
    p.add_argument("--at-table", action="store_true",
                   help="start standing at the workbench instead of in the middle of the room")
    return p


def run(a) -> int:
    agent = build_agent(a)
    backend = make_backend(a)
    mind = Mind(agent, backend, log=print if a.headless else None)

    if a.headless > 0:
        return headless(a, agent, mind)

    from embodied_human.viewer import App
    app = App(agent, mind, speed=a.speed)
    mind.start()
    app.run()
    return 0


def headless(a, agent, mind) -> int:
    from embodied_human.viewer import SceneRenderer
    from PIL import Image
    out = Path(a.out) / "mind"
    out.mkdir(parents=True, exist_ok=True)
    scripted = sorted((float(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in a.say)
    view = SceneRenderer(agent, (640, 420), mind)
    view.set_view("three_quarter")
    mind.start()
    frames = []
    next_frame = 0.0
    t0 = time.time()
    last_t = -1
    while agent.t < a.headless:
        # the mind thread runs in wall-clock time; keep the sim from racing ahead of it
        while scripted and agent.t >= scripted[0][0]:
            _, text = scripted.pop(0)
            agent.skills.hear(text)
            mind.poke()
            print(f"[{agent.t:6.1f}s] heard: {text}")
        agent.step()
        if agent.t >= next_frame:
            frames.append(view.render().copy())
            next_frame += a.frame_every
        if int(agent.t) != last_t and int(agent.t) % 5 == 0:
            last_t = int(agent.t)
            print(f"[{agent.t:6.1f}s] {agent.gait.diag.mode if agent.gait.active else 'standing':8s} "
                  f"doing: {agent.skills.current_description():40s} mind: {mind.status}")
        if agent.state.fallen:
            print("the person has fallen")
            break
    mind.stop()
    print(f"wall {time.time() - t0:.0f}s for {agent.t:.0f}s simulated")
    cols = 4
    rows = (len(frames) + cols - 1) // cols
    for k in range(0, len(frames), cols * 3):
        chunk = frames[k:k + cols * 3]
        r = (len(chunk) + cols - 1) // cols
        w, h = chunk[0].size
        sheet = Image.new("RGB", (cols * w, r * h))
        for i, f in enumerate(chunk):
            sheet.paste(f, ((i % cols) * w, (i // cols) * h))
        path = out / f"sheet_{k // (cols * 3):02d}.png"
        sheet.save(path)
        print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
