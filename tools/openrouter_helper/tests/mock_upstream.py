#!/usr/bin/env python3
"""A fake OpenRouter for testing the helper offline.

    python mock_upstream.py 9911            # canned plans, streamed
    MOCK_429_FIRST=2 python mock_upstream.py 9911   # the first two requests get HTTP 429
    MOCK_FAIL_MODEL=poolside/laguna-s-2.1:free ...  # that model returns 404
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

N = {"n": 0}
PLAN = """TITLE: Wire the circadian clock into the eyes and the drives
RATIONALE: The run files show the clock advancing but nothing reads it. Connect sleep pressure to the lids and to the sleepiness drive so the inner clock has a visible effect.
FILES_TO_READ: embodied_human/inner_world.py, embodied_human/ocular.py
RISKS: Per-step cost; keep the update at 1 Hz.
ACCEPTANCE:
- ms per step at base unchanged within 5 %
- lids droop measurably when sleep pressure > 0.7
CODER_PROMPT:
Context: the inner world has a circadian clock (inner_brain.CircadianClock) whose sleep_pressure is computed
but only read by the neural mass. Context 2: the ocular model takes `sleepiness` as an input.

1. In embodied_human/inner_world.py add the clock's sleep pressure to the extra drive `mental_fatigue` (already there)...
2. In embodied_human/agent.py pass the clock sleep pressure into OcularInputs.sleepiness as max(existing, pressure).

How to verify: bash tools/openrouter_helper/helper.sh verify

Do not: add dependencies or change public interfaces.
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        N["n"] += 1
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if N["n"] <= int(os.environ.get("MOCK_429_FIRST", "0")):
            self.send_response(429); self.send_header("Retry-After", "1")
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":{"message":"rate limited","code":429}}'); return
        if body.get("model") == os.environ.get("MOCK_FAIL_MODEL"):
            self.send_response(404); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":{"message":"no such model","code":404}}'); return
        if os.environ.get("MOCK_REJECT_BIG") and body.get("max_tokens", 0) > int(os.environ["MOCK_REJECT_BIG"]):
            self.send_response(400); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":{"message":"max_tokens is too large for this model","code":400}}'); return
        text = PLAN if "PLANNER" in json.dumps(body.get("messages", [])[:1]) else "OK"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        step = 120
        for i in range(0, len(text), step):
            chunk = {"id": "x", "model": body.get("model"), "choices": [{"delta": {"content": text[i:i + step]}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
            time.sleep(0.005)
        fin = {"id": "x", "model": body.get("model"), "choices": [{"delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": len(text) // 4}}
        self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
