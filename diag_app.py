"""Open the Tk window for a few seconds, drive it programmatically, and save the
last rendered frame.  Verifies the GUI path without anyone watching it."""
import sys
import time
from pathlib import Path

from run_mind import build_agent, make_backend, build_parser_for_tests
from embodied_human.mind import Mind
from embodied_human.viewer import App

a = build_parser_for_tests(["--at-table"])
agent = build_agent(a)
mind = Mind(agent, make_backend(a))
app = App(agent, mind, speed=2.0)
mind.start()
done = {"n": 0}


def script():
    done["n"] += 1
    if done["n"] == 1:
        agent.skills.hear("hello there")
        app._append("heard", 'you: "hello there"')
        mind.poke()
    if done["n"] == 4:
        app.view.render().save(Path("out/mind/app_frame.png"))
        print("sim time", round(agent.t, 1), "log chars", len(app.log.get("1.0", "end")))
        app.close()
        return
    app.root.after(2500, script)


app.root.after(2500, script)
t0 = time.time()
app.run()
print("window ran for", round(time.time() - t0, 1), "s; mind cycles:", mind.n_cycles)
