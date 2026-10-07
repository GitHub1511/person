"""Exercise the real-backend code path (OpenAI-compatible /v1/completions with
streaming) against a tiny fake server, since AZR weights are not installed.

The fake server returns a canned ``<think>/<answer>`` stream; everything else --
prompt construction, HTTP streaming, parsing, the sandbox, dispatch into the
skill system, the thought bubble text -- is the code that would run with AZR.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.mind import Mind, OpenAICompatBackend

SEEN = {}

REPLY = (' The apple is close and I have nothing else to do. I want to see how it '
         'feels in my hand. </think> <answer>\nlook_at("apple")\ngrab("apple")\n</answer>')


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        SEEN["prompt"] = body["prompt"]
        SEEN["auth"] = self.headers.get("Authorization")
        SEEN["stop"] = body.get("stop")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        words = REPLY.split(" ")
        for i in range(0, len(words), 3):
            piece = " ".join(words[i:i + 3]) + " "
            chunk = {"choices": [{"text": piece}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


srv = HTTPServer(("127.0.0.1", 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

cfg = SimConfig(out_dir=Path("out"))
cfg.rates.receptor = 100.0
cfg.rates.afferent = 100.0
ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
ag.autonomous = False
ag.data.qpos[ag.meta.root_qpos_addr + 1] -= 0.32
import mujoco
mujoco.mj_forward(ag.model, ag.data)
for _ in range(1200):
    ag.step()

backend = OpenAICompatBackend(base_url=f"http://127.0.0.1:{port}/v1", model="azr", api_key="k")
mind = Mind(ag, backend)
ag.skills.hear("could you pick up the apple?")
mind._think_once(False)
print("STATUS      :", mind.status)
print("THOUGHT     :", mind.thought)
print("ANSWER      :", mind.last_reply.answer.replace("\n", " ; "))
print("QUEUE       :", [q[0] for q in ag.skills.queue], "heard cleared:", ag.skills.heard == [])
print("AUTH HEADER :", SEEN["auth"], " STOP:", SEEN["stop"])
print("--- prompt sent to the model (tail) ---")
print(SEEN["prompt"][-900:])
assert mind.last_reply.calls and mind.last_reply.calls[0][0] == "look_at"
assert SEEN["prompt"].rstrip().endswith("<think>")
assert "you do not have to say anything" in SEEN["prompt"].lower().replace("\n", " ") or \
    "do not have to say anything" in SEEN["prompt"].lower().replace("\n", " ")
for _ in range(int(14 / ag.dt)):
    ag.step()
    if ag.skills.held["l"] or ag.skills.held["r"]:
        break
print("after acting: held =", ag.skills.held, " events =", ag.skills.events[-3:],
      " doing:", ag.skills.current_description(), " last:", ag.skills.last_result,
      " fallen:", ag.state.fallen, " t =", round(ag.t, 1))
