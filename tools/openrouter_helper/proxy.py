#!/usr/bin/env python3
"""
A local, rate-limiting OpenRouter gateway for the helper.

Everything the helper asks a model goes through here, so the limits are enforced in one
place no matter who is asking:

* **model order** -- ``models.json`` is walked strictly in order; a model is used until it
  has made ``requests_per_model_per_day`` requests (200), then the next one takes over;
* **account limits** -- at most 20 requests in any rolling minute (and a minimum gap
  between requests), at most 1000 per UTC day;
* **persistence** -- counts live in ``state/usage.json`` so restarting the helper (or the
  machine) never resets them; they reset by themselves when the UTC date changes;
* **robustness** -- 429s honour Retry-After; a model that keeps failing, or does not
  exist, is skipped for the day; a ``max_tokens`` the provider rejects steps down the
  model's ladder (and the choice is remembered); long answers are streamed from
  OpenRouter and handed back as one complete JSON reply.

It listens only on 127.0.0.1.  The API key is read from the environment and is never
logged or returned.

    python proxy.py                   # serve on 127.0.0.1:8765
    python proxy.py --probe           # send "Reply with OK" to every model once, then exit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get("OR_STATE", HERE / "state"))
STATE.mkdir(parents=True, exist_ok=True)
USAGE = STATE / "usage.json"
REQLOG = STATE / "requests.jsonl"
UPSTREAM = os.environ.get("OR_UPSTREAM", "https://openrouter.ai/api/v1/chat/completions")
CFG = json.loads((Path(os.environ.get("OR_MODELS", HERE / "models.json"))).read_text())
MODELS = CFG["models"]
QUOTA = int(os.environ.get("OR_QUOTA", CFG.get("requests_per_model_per_day", 200)))
RPM = int(os.environ.get("OR_RPM", CFG.get("requests_per_minute", 20)))
DAILY = int(os.environ.get("OR_DAILY", CFG.get("requests_per_day", 1000)))
TIME_SCALE = float(os.environ.get("OR_TIME_SCALE", "1.0"))        # tests shrink the waits
MIN_GAP = (60.0 / RPM + 0.25) * TIME_SCALE
READ_TIMEOUT = float(os.environ.get("OR_READ_TIMEOUT", "420"))
MAX_ATTEMPTS = 4
LOCK = threading.Lock()


def today() -> str:
    f = os.environ.get("OR_FAKE_DATE_FILE")
    if f and Path(f).exists():
        return Path(f).read_text().strip()
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def seconds_to_midnight() -> int:
    n = dt.datetime.now(dt.timezone.utc)
    nxt = (n + dt.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    return int((nxt - n).total_seconds())


class Usage:
    def __init__(self):
        self.d = {"date": today(), "counts": {}, "total": 0, "skipped": {}, "gated": {}, "consec_fail": {},
                  "max_tokens_ok": {}, "ladder_idx": {}}
        if USAGE.exists():
            try:
                self.d.update(json.loads(USAGE.read_text()))
            except (OSError, ValueError):
                pass
        self.recent: deque[float] = deque()
        self.last_call = 0.0
        self.rollover()

    def save(self) -> None:
        tmp = USAGE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        tmp.replace(USAGE)

    def rollover(self) -> None:
        t = today()
        if self.d.get("date") != t:
            keep = {k: self.d[k] for k in ("max_tokens_ok", "ladder_idx") if k in self.d}
            self.d.update({"date": t, "counts": {}, "total": 0, "skipped": {}, "gated": {}, "consec_fail": {}})
            self.d.update(keep)
            self.save()

    # ---- model selection ---------------------------------------------------------------
    def current(self, harness: bool = False) -> dict | None:
        """The model to use next.  Models that only answer listed coding agents are passed over
        for ordinary callers (the planner) and used for requests that carry a harness identity."""
        self.rollover()
        if self.d["total"] >= DAILY:
            return None
        for m in MODELS:
            mid = m["id"]
            if mid in self.d["skipped"]:
                continue
            if mid in self.d.get("gated", {}) and not harness:
                continue
            if self.d["counts"].get(mid, 0) < QUOTA:
                return m
        return None

    def max_tokens_for(self, m: dict) -> int:
        i = int(self.d["ladder_idx"].get(m["id"], 0))
        ladder = m["max_tokens_try"]
        return int(ladder[min(i, len(ladder) - 1)])

    def step_down(self, m: dict) -> bool:
        i = int(self.d["ladder_idx"].get(m["id"], 0))
        if i + 1 < len(m["max_tokens_try"]):
            self.d["ladder_idx"][m["id"]] = i + 1
            self.save()
            return True
        return False

    def count(self, mid: str) -> None:
        self.d["counts"][mid] = self.d["counts"].get(mid, 0) + 1
        self.d["total"] += 1
        self.save()

    # ---- rate limiting ---------------------------------------------------------------------
    def wait_turn(self) -> float:
        waited = 0.0
        window = 60.0 * TIME_SCALE
        while True:
            now = time.time()
            while self.recent and now - self.recent[0] > window:
                self.recent.popleft()
            gap = self.last_call + MIN_GAP - now
            full = len(self.recent) >= RPM
            if not full and gap <= 0:
                break
            sleep = max(gap, (self.recent[0] + window - now) if full else 0.0, 0.05)
            time.sleep(min(sleep, 5.0))
            waited += min(sleep, 5.0)
        self.recent.append(time.time())
        self.last_call = time.time()
        return waited


U = Usage()


def log_request(rec: dict) -> None:
    with open(REQLOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


class UpstreamError(Exception):
    def __init__(self, status, body, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status, self.body, self.retry_after = status, body, retry_after


def call_upstream(model: str, messages: list, max_tokens: int, extra: dict, ident: dict | None = None) -> dict:
    """One streamed request to OpenRouter; returns an assembled chat.completion dict."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True}
    for k in ("temperature", "top_p", "stop", "seed", "tools", "tool_choice", "parallel_tool_calls"):
        if k in extra:
            payload[k] = extra[k]
    req = urllib.request.Request(UPSTREAM, data=json.dumps(payload).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        # the identity of whoever is really asking is passed through unchanged; we never claim
        # to be a harness we are not
        "HTTP-Referer": (ident or {}).get("HTTP-Referer", "http://localhost/person-helper"),
        "X-Title": (ident or {}).get("X-Title", "person-sim helper")})
    text, reasoning, usage, finish, mid = [], [], {}, None, model
    calls: dict[int, dict] = {}                   # streamed tool-call fragments, by index
    try:
        resp = urllib.request.urlopen(req, timeout=READ_TIMEOUT)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        ra = e.headers.get("Retry-After")
        raise UpstreamError(e.code, body, float(ra) if ra and ra.replace(".", "").isdigit() else None)
    except Exception as e:
        raise UpstreamError(0, f"{type(e).__name__}: {e}")
    with resp:
        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in ctype:                  # a provider that ignores stream=true
            body = json.loads(resp.read().decode("utf-8", "replace"))
            if "error" in body:
                raise UpstreamError(int(body["error"].get("code", 500) or 500), json.dumps(body["error"])[:1500])
            return body
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if "error" in j:
                raise UpstreamError(int(j["error"].get("code", 502) or 502), json.dumps(j["error"])[:1500])
            mid = j.get("model", mid)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content"):
                    text.append(d["content"])
                if d.get("reasoning"):
                    reasoning.append(d["reasoning"])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(int(tc.get("index", 0)), {"id": "", "type": "function",
                                                                     "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        c["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        c["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        c["function"]["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": "".join(text)}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        for n, c in enumerate(msg["tool_calls"]):
            c["id"] = c["id"] or f"call_{n}"
        msg["content"] = msg["content"] or None
    return {"id": "gen-proxy", "object": "chat.completion", "model": mid,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": usage, "x_reasoning_chars": sum(len(r) for r in reasoning)}


def handle_chat(body: dict, ident: dict | None = None) -> tuple[int, dict]:
    harness = bool(ident)
    messages = body.get("messages") or []
    if not messages:
        return 400, {"error": {"message": "no messages"}}
    with LOCK:                                    # one request at a time, in order
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            m = U.current(harness)
            if m is None:
                return 429, {"error": {"message": "daily quota exhausted for every model",
                                       "code": "quota_exhausted", "retry_after": seconds_to_midnight()}}
            mid = m["id"]
            mt = U.max_tokens_for(m)
            asked = body.get("max_tokens") or body.get("max_completion_tokens")
            if asked:
                mt = min(mt, int(asked))
            U.wait_turn()
            t0 = time.time()
            U.count(mid)                          # every attempt counts, conservatively
            attempts += 1
            try:
                out = call_upstream(mid, messages, mt, body, ident)
                U.d["consec_fail"][mid] = 0
                U.save()
                content = out["choices"][0]["message"]["content"] or ""
                log_request({"t": time.time(), "model": mid, "status": 200, "s": round(time.time() - t0, 1),
                             "max_tokens": mt, "out_chars": len(content), "usage": out.get("usage", {}),
                             "n_today": U.d["counts"].get(mid, 0)})
                out["x_helper"] = {"model": mid, "max_tokens": mt, "target_tokens": m["target_tokens"],
                                   "count_for_model": U.d["counts"].get(mid, 0), "total_today": U.d["total"]}
                return 200, out
            except UpstreamError as e:
                body_l = (e.body or "").lower()
                log_request({"t": time.time(), "model": mid, "status": e.status, "s": round(time.time() - t0, 1),
                             "max_tokens": mt, "err": (e.body or "")[:300]})
                if e.status == 429:
                    time.sleep(min(e.retry_after or 20.0, 120.0) * TIME_SCALE)
                    continue
                if e.status == 400 and any(w in body_l for w in ("max_tokens", "maximum", "context", "too large", "output")):
                    if U.step_down(m):
                        continue
                if e.status == 403 and "agentic harness" in body_l:
                    # a provider rule, not ours to get around: only a listed coding agent may use it
                    if harness:
                        U.d["skipped"][mid] = "403: refused even with the caller's harness identity"
                    else:
                        U.d.setdefault("gated", {})[mid] = "403: only available on agentic harnesses"
                    U.save()
                    continue
                if e.status in (401, 402, 403):
                    return e.status, {"error": {"message": f"upstream refused ({e.status}); check the key / credits",
                                                "detail": (e.body or "")[:200]}}
                if e.status == 404 or e.status == 400:
                    U.d["skipped"][mid] = f"HTTP {e.status}: {(e.body or '')[:120]}"
                    U.save()
                    continue
                U.d["consec_fail"][mid] = U.d["consec_fail"].get(mid, 0) + 1
                if U.d["consec_fail"][mid] >= 8:
                    U.d["skipped"][mid] = f"8 consecutive failures, last HTTP {e.status}"
                U.save()
                time.sleep(min(2.0 ** attempts, 20.0) * TIME_SCALE)
        return 502, {"error": {"message": "upstream kept failing", "code": "upstream_failed"}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            m = U.current()
            self._send(200, {"ok": True, "current": m["id"] if m else None, "date": U.d["date"]})
        elif self.path.startswith("/current"):
            m = U.current()
            if m is None:
                self._send(200, {"model": None, "retry_after": seconds_to_midnight(), "usage": U.d})
            else:
                self._send(200, {"model": m["id"], "target_tokens": m["target_tokens"],
                                 "max_tokens": U.max_tokens_for(m), "cap_note": m.get("cap_note", ""),
                                 "used": U.d["counts"].get(m["id"], 0), "quota": QUOTA,
                                 "total_today": U.d["total"], "daily": DAILY})
        elif self.path.startswith("/usage"):
            self._send(200, U.d)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
        except ValueError:
            return self._send(400, {"error": {"message": "bad json"}})
        ident = {k: self.headers[k] for k in ("HTTP-Referer", "X-Title") if self.headers.get(k)}
        code, obj = handle_chat(body, ident or None)
        if body.get("stream") and code == 200:
            self._send_sse(obj)
        else:
            self._send(code, obj)

    def _send_sse(self, obj: dict) -> None:
        """The reply is already complete; replay it as an event stream for clients (agents) that
        insist on streaming."""
        ch = obj["choices"][0]
        msg = ch["message"]
        delta = {"role": "assistant"}
        if msg.get("content"):
            delta["content"] = msg["content"]
        if msg.get("tool_calls"):
            delta["tool_calls"] = [dict(c, index=i) for i, c in enumerate(msg["tool_calls"])]
        base = {"id": obj.get("id", "gen-proxy"), "object": "chat.completion.chunk", "model": obj.get("model", "")}
        chunks = [dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                  dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": ch.get("finish_reason", "stop")}]),
                  dict(base, choices=[], usage=obj.get("usage") or {})]
        data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        raw = data.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def probe() -> int:
    """Send "Reply with OK" to every model once (a few requests), without touching quotas' meaning."""
    rows = []
    for m in MODELS:
        t0 = time.time()
        U.wait_turn()
        U.count(m["id"])
        try:
            out = call_upstream(m["id"], [{"role": "user", "content": "Reply with OK"}], 64, {})
            txt = out["choices"][0]["message"]["content"].strip()[:40]
            rows.append((m["id"], "ok", f"{time.time() - t0:.1f}s", repr(txt)))
        except UpstreamError as e:
            rows.append((m["id"], f"HTTP {e.status}", f"{time.time() - t0:.1f}s", (e.body or "")[:100].replace("\n", " ")))
    for r in rows:
        print("  %-42s %-9s %-7s %s" % r)
    return 0 if all(r[1] == "ok" for r in rows) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("OR_PORT", "8765")))
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return probe()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"proxy on 127.0.0.1:{a.port}  quota {QUOTA}/model, {RPM}/min, {DAILY}/day, "
          f"{len(MODELS)} models", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
