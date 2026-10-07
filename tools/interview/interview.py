#!/usr/bin/env python
"""The interviewer: one simulation running constantly, one academic question
at a time.

    python tools/interview/interview.py              # run forever (1 sim, AZR mind)
    python tools/interview/interview.py --once       # one question cycle now, then exit
    python tools/interview/interview.py --dry-run    # print the question-generator
                                                     # prompt for a fixture transcript; no network, no sim

What it does
------------
* Keeps exactly ONE headless simulation alive around the clock.  Its mind is
  the local Absolute Zero Reasoner coder 3B
  (``http://127.0.0.1:1234/v1``, model ``absolute_zero_reasoner-coder-3b``),
  so the person itself costs no OpenRouter quota at all.
* At most ONE question every 30 minutes (wall clock, persisted across
  restarts).  Each next question is built by reading the person's answers
  since the last question: the generator prompt carries the transcript tail
  (its thoughts, actions, spoken lines, silence) plus a body snapshot, and
  must reference something actually observed.
* Questions are academic and experimental only: neutral, single, about the
  person's own perception/processing/decisions.  The generator is forbidden
  from leading the witness -- no emotional baiting, no suggesting feelings,
  no praise-bait, no roleplay, no re-asking until it talks.  Silence is the
  person's normal state and counts as data.
* Question generation goes through the local OpenRouter gateway
  (``tools/openrouter_helper/proxy.py`` on 127.0.0.1:8765), which walks
  models strictly in order under the 200-requests-per-model-per-day limit
  and rotates to the next model on exhaustion.  At ~48 questions/day the
  interviewer barely sips quota; rotation is automatic when the shared
  budget runs dry.  The model actually used is recorded with the question.
* EVERYTHING lands in the instance's own transcript file
  (``<out>/instances/<id>.txt``): INTERVIEW_QUESTION lines carry the model
  and rationale; the person's THOUGHT / ACTION / SAID lines follow,
  written by the mind loop itself (see ``embodied_human/instance_log.py``).

Stdlib + numpy/mujoco/requests only (same as the project itself).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HELPER = ROOT / "tools" / "openrouter_helper"
STATE = Path(os.environ.get("INTERVIEW_STATE") or (Path(__file__).resolve().parent / "state"))
STATE.mkdir(parents=True, exist_ok=True)

AZR_URL = os.environ.get("INTERVIEW_AZR_URL", "http://127.0.0.1:1234/v1")
AZR_MODEL = os.environ.get("INTERVIEW_AZR_MODEL", "absolute_zero_reasoner-coder-3b")
PROXY = os.environ.get("OR_PROXY", "http://127.0.0.1:8765")
PROXY_PORT = int(os.environ.get("OR_PORT", "8765"))

MIN_ASK_GAP_S = 30 * 60          # NEVER more than one question per 30 minutes
GEN_TIMEOUT = 120
FAIL_RETRY_S = 5 * 60            # wait this long after a failed generation

GENERATOR_SYSTEM = """\
You write interview questions for an embodied-simulation research subject: a simulated human
(MuJoCo body, interoception, affect, drives, active inference) whose deliberating mind is a
small code-reasoning model. An interviewer asks it at most one question per 30 minutes and
studies how its thoughts, actions and speech change. Your output is exactly one question.

Rules (all mandatory):
- ACADEMIC AND EXPERIMENTAL ONLY. Ask about its own perception, body state, decisions, or
  reasoning (e.g. what it noticed, how it chose, what changed its mind). Real observational value.
- REFERENCE SOMETHING ACTUALLY OBSERVED in the transcript excerpt (a thought, action, spoken
  line, or a stretch of silence). Quote or name it briefly. Never ask about things with no
  basis in the excerpt.
- NEVER LEAD OR EGG ON. No emotional baiting, no suggesting feelings it did not report, no
  praise-bait ("you're so smart, now tell me..."), no roleplay ("pretend you are..."), no
  asking it to feel things, no second question hidden inside the first, no re-asking a
  question it already declined or ignored. Silence is its normal state and valid data:
  if it was silent, ask about that neutrally or not at all.
- ONE question only: 10-300 characters, exactly one '?' at the end. Plain wording a
  small model parses easily.

Reply with ONLY strict JSON inside ```json fences:
{"question": "<the single question>", "rationale": "<2 sentences: what observed pattern this probes and why>"}"""


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def _load_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = HELPER / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def _state() -> dict:
    f = STATE / "state.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            pass
    return {"last_asked_wall": 0.0, "n_asked": 0, "last_model": "", "last_question": ""}


def _save_state(d: dict) -> None:
    (STATE / "state.json").write_text(json.dumps(d, indent=1), encoding="utf-8")


def _proxy_up() -> bool:
    try:
        with urllib.request.urlopen(PROXY + "/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_proxy() -> bool:
    """Make sure the quota-enforcing gateway is answering (start it if not)."""
    if _proxy_up():
        return True
    key = _load_key()
    if not key:
        print("interview: no OPENROUTER_API_KEY (env or helper .env); "
              "question generation paused, sim keeps running", flush=True)
        return False
    env = dict(os.environ, OPENROUTER_API_KEY=key, OR_STATE=str(HELPER / "state"),
               OR_PORT=str(PROXY_PORT))
    logf = open(HELPER / "state" / "proxy.log", "a", encoding="utf-8", errors="replace")
    subprocess.Popen([sys.executable, str(HELPER / "proxy.py"), "--port", str(PROXY_PORT)],
                     env=env, stdout=logf, stderr=subprocess.STDOUT,
                     cwd=str(ROOT))
    for _ in range(60):
        if _proxy_up():
            return True
        time.sleep(0.5)
    print("interview: gateway did not come up; sim keeps running", flush=True)
    return False


# --------------------------------------------------------------------------
# question generation (through the rotating gateway)
# --------------------------------------------------------------------------
class QuotaDry(Exception):
    def __init__(self, retry_after: float):
        super().__init__("quota exhausted")
        self.retry_after = retry_after


def ask_proxy(messages: list[dict], max_tokens: int = 400) -> tuple[str, str]:
    """One gateway call.  Returns (model_used, text).  Raises QuotaDry."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
    req = urllib.request.Request(
        PROXY + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "HTTP-Referer": "http://localhost/person-interview",
                 "X-Title": "person-interview"})
    try:
        with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read())
        except Exception:
            raise RuntimeError(f"gateway HTTP {e.code}")
        if err.get("error", {}).get("code") == "quota_exhausted":
            raise QuotaDry(float(err["error"].get("retry_after", 3600)))
        raise RuntimeError(f"gateway refused: {str(err)[:200]}")
    try:
        content = out["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("bad gateway reply shape")
    model = (out.get("x_helper") or {}).get("model", "?")
    return model, content


def _extract_json(text: str) -> dict | None:
    import re
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    cand = m.group(1) if m else text
    start = cand.find("{")
    end = cand.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        d = json.loads(cand[start:end + 1])
        return d if isinstance(d, dict) else None
    except ValueError:
        return None


def valid_question(q: str) -> bool:
    q = (q or "").strip()
    return 10 <= len(q) <= 300 and q.count("?") == 1 and q.endswith("?")


def transcript_tail(txt_path: Path, max_chars: int = 4000) -> str:
    """Recent transcript lines (whole tail if small; last ~max_chars otherwise)."""
    try:
        data = txt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(transcript not yet written)"
    lines = [ln for ln in data.splitlines() if ln.strip()]
    if not lines:
        return "(transcript empty so far)"
    # keep everything since the previous interview question plus a short lead-in
    idx = max(i for i, ln in enumerate(lines) if "INTERVIEW_QUESTION" in ln) \
        if any("INTERVIEW_QUESTION" in ln for ln in lines) else max(0, len(lines) - 40)
    tail = lines[max(0, idx - 6):]
    text = "\n".join(tail)
    return text[-max_chars:]


def body_snapshot(mind) -> str:
    try:
        ctx = mind.percept.context()
    except Exception as exc:
        return f"(body snapshot unavailable: {exc})"
    bits = [f"posture: {ctx.get('posture')}", f"doing: {ctx.get('doing')}",
            f"emotion: {ctx.get('emotion')}",
            f"urgent drives: {ctx.get('urgent') or 'none'}"]
    heard = ctx.get("heard") or []
    if heard:
        bits.append("recently heard: " + "; ".join(f'"{h}"' for h in heard[-2:]))
    return "; ".join(bits)


def build_generator_messages(transcript: str, body: str) -> list[dict]:
    user = ("TRANSCRIPT EXCERPT (thoughts/actions/speech of the simulated person; "
            "silence between lines is normal):\n```\n" + transcript[:4000] +
            "\n```\nBODY SNAPSHOT: " + body[:600] +
            "\nWrite the next single interview question as specified.")
    return [{"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": user}]


def generate_question(mind) -> tuple[str, str, str] | None:
    """Returns (question, model, rationale) or None on any failure."""
    txt = mind.instance_log.path
    msgs = build_generator_messages(transcript_tail(txt), body_snapshot(mind))
    try:
        model, content = ask_proxy(msgs)
    except QuotaDry as q:
        print(f"interview: quota dry, retry in {q.retry_after:.0f}s", flush=True)
        st = _state()
        st["quota_wait_until"] = time.time() + q.retry_after
        _save_state(st)
        return None
    except Exception as exc:
        print(f"interview: generation failed ({exc}); retry later", flush=True)
        return None
    d = _extract_json(content)
    q = (d or {}).get("question", "")
    r = (d or {}).get("rationale", "")
    if not valid_question(q):
        print(f"interview: generator broke the rules, discarding: {q[:120]!r}", flush=True)
        return None
    return q.strip(), model, " ".join(str(r).split())[:300]


# --------------------------------------------------------------------------
# the always-on loop
# --------------------------------------------------------------------------
def due_now(st: dict, now: float | None = None) -> bool:
    """True when a new question may be asked (30-minute floor, quota waits)."""
    now = time.time() if now is None else now
    if now < float(st.get("quota_wait_until", 0)):
        return False
    return now - float(st.get("last_asked_wall", 0)) >= MIN_ASK_GAP_S


def build_sim(instance_id: str = "", out_dir: str = "out_interview"):
    from embodied_human.agent import EmbodiedHuman
    from embodied_human.config import SimConfig
    from embodied_human.mind import OpenAICompatBackend, Mind
    cfg = SimConfig(seed=7, out_dir=ROOT / out_dir, instance_id=instance_id)
    cfg.rates.receptor = 100.0
    cfg.rates.afferent = 100.0
    cfg.rates.interoception = 50.0
    agent = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    agent.autonomous = False
    agent.gait.hold_stance = True
    backend = OpenAICompatBackend(base_url=AZR_URL, model=AZR_MODEL)
    mind = Mind(agent, backend, instance_id=agent.instance_id)
    return agent, mind


def run_forever(duration_s: float = 0.0, ask_first: bool = False) -> int:
    """Keep one sim alive; ask at most one question per 30 minutes."""
    if not ensure_proxy():
        print("interview: continuing without question generation until a key/quota exists",
              flush=True)
    agent, mind = build_sim()
    print(f"interview: instance {agent.instance_id}", flush=True)
    print(f"interview: transcript at {mind.instance_log.path}", flush=True)
    mind.start()
    t_start = time.time()
    last_fail = 0.0
    try:
        while True:
            # pace the physics to roughly real time (cap steps per tick)
            target = time.time() - t_start
            for _ in range(5000):
                if agent.t >= target:
                    break
                agent.step()
            st = _state()
            if (ask_first and st["n_asked"] == 0) or due_now(st):
                if ensure_proxy():
                    got = generate_question(mind)
                    st = _state()
                    if got:
                        q, model, rationale = got
                        mind.interview_ask(q, model=model, rationale=rationale)
                        st.update(last_asked_wall=time.time(), n_asked=st.get("n_asked", 0) + 1,
                                  last_model=model, last_question=q)
                        _save_state(st)
                        print(f"interview: asked [{model}]: {q}", flush=True)
                    elif time.time() - last_fail > FAIL_RETRY_S:
                        last_fail = time.time()
                ask_first = False
            if duration_s and time.time() - t_start >= duration_s:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            mind.stop()
        except Exception:
            pass
        print("interview: stopped", flush=True)
    return 0


def selftest() -> int:
    fails = []

    def check(name, cond, detail=""):
        print(("  ok   " if cond else "  FAIL ") + name + (f" [{detail}]" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    # 30-minute gate, with injected clock
    check("fresh state is due", due_now({"last_asked_wall": 0}, now=10_000.0))
    check("asked 1s ago is NOT due",
          not due_now({"last_asked_wall": 10_000.0}, now=10_001.0))
    check("asked 29min ago is NOT due",
          not due_now({"last_asked_wall": 0}, now=29 * 60.0))
    check("asked 30min ago is due",
          due_now({"last_asked_wall": 0}, now=30 * 60.0))
    check("quota wait blocks",
          not due_now({"last_asked_wall": 0, "quota_wait_until": 99_999.0}, now=50_000.0))
    # question shape rules
    check("good question passes", valid_question("What did you notice just before you turned?"))
    check("two questions rejected", not valid_question("What did you see? What did you hear?"))
    check("no mark rejected", not valid_question("Tell me about the table"))
    check("too long rejected", not valid_question("x?" * 200))
    # generator prompt carries the rules + observation grounding
    msgs = build_generator_messages("THOUGHT: staring at the apple", "posture: standing")
    sys_txt = msgs[0]["content"]
    for needle in ("ONE question", "NEVER LEAD", "silence", "```json",
                   "REFERENCE SOMETHING ACTUALLY OBSERVED"):
        check(f"system has {needle!r}", needle in sys_txt)
    check("user msg carries transcript", "staring at the apple" in msgs[1]["content"])
    # JSON extraction tolerance
    d = _extract_json('```json\n{"question": "Q?", "rationale": "R"}```')
    check("fenced json parses", d == {"question": "Q?", "rationale": "R"}, str(d))
    d = _extract_json('noise {"question": "Q?", "rationale": "R"} tail')
    check("bare json parses", d is not None and d["question"] == "Q?")
    check("garbage gives None", _extract_json("no braces here") is None)
    print("ALL PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true",
                    help="ask one question now (bypasses the 30-min gate once), then exit")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="run the sim this many wall seconds then exit (0 = forever)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the generator prompt for a fixture transcript; no sim, no network")
    ap.add_argument("--selftest", action="store_true", help="offline unit checks")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.dry_run:
        fixture = ("[2026-10-07T12:00:09Z t=13.1s] THOUGHT: There is a mug on the table. "
                   "I have not looked at it properly.\n"
                   "[2026-10-07T12:00:11Z t=15.0s] ACTION: look_at(\"mug\") | latency 8.2s\n"
                   "[2026-10-07T12:00:40Z t=44.0s] THOUGHT: Nothing in particular needs doing.")
        for m in build_generator_messages(fixture, "posture: standing; doing: nothing"):
            print(f"== {m['role']} ==\n{m['content']}\n")
        return 0
    if a.once:
        st = _state()
        st["last_asked_wall"] = 0.0
        _save_state(st)
        return run_forever(duration_s=a.duration or 120.0, ask_first=True)
    return run_forever(duration_s=a.duration)


if __name__ == "__main__":
    sys.exit(main())
