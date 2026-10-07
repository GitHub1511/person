"""Render one frame with a speech bubble and a thought bubble over the head."""
import sys
from pathlib import Path

import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.mind import Mind, StubBackend
from embodied_human.viewer import SceneRenderer

cfg = SimConfig(out_dir=Path("out"))
cfg.rates.receptor = 100.0
cfg.rates.afferent = 100.0
ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
ag.autonomous = False
ag.gait.hold_stance = True
mind = Mind(ag, StubBackend())
mind.thought = ("The red apple is on the table, about half a metre in front of me. "
                "I have not looked at it properly, and I am curious what it feels like. "
                "Nobody is here, so there is nothing to say.")
view = SceneRenderer(ag, (960, 600), mind)
view.set_view("three_quarter")
view.distance = 3.0
ag.skills.api_look_at("apple")
for _ in range(1500):
    ag.step()
    if ag.t > 0.8:
        mind.thought_t = ag.t - 1.0
ag.skills.api_say("Hello! I was just looking at this apple.")
shots = []
for k in range(3):
    for _ in range(int(0.55 / ag.dt)):
        ag.step()
        mind.thought_t = ag.t - 1.0
    shots.append(view.render())
out = Path("out/mind")
out.mkdir(parents=True, exist_ok=True)
shots[1].save(out / "bubbles.png")
print("saved", out / "bubbles.png", "jaw", round(ag.skills.speech.jaw, 3), "speaking", ag.skills.speech.speaking)
