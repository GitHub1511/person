"""
The mind: a language model behind the body.

The embodied agent (affect, drives, active inference, the whole-body motor
system) is a *body with a nervous system*.  This module puts a deliberating
mind on top of it.  The intended mind is the **Absolute Zero Reasoner (AZR)**
from Tsinghua (LeapLabTHU/Absolute-Zero-Reasoner): a Qwen2.5-Coder model that
learned to reason by proposing and solving its own tasks, with no human data,
and that answers in the ``<think> ... </think> <answer> ... </answer>`` format.

How the pieces fit
------------------
::

     body, world ──► Percept (plain English) ──► prompt ──► LLM
                                                           │
                                         <think> … </think> │ <answer> python </answer>
                                                  │                      │
                                  shown above the head,              parsed by `ast`,
                                  grey and italic                    whitelisted calls only
                                                                          │
                                                                          ▼
                                                                  SkillSystem (walk, grab, say…)

* The **thoughts** are the model's ``<think>`` text, displayed verbatim above the
  head in grey italics.  They are private: nothing in them is spoken.
* The **actions** are a few lines of Python-looking calls from a fixed API.  The
  answer is parsed with :mod:`ast` and *never executed*: only calls to the
  whitelisted names with literal arguments are accepted, so the model cannot do
  anything the API does not offer.
* **Speech** is one of those calls, ``say("...")``.  **There is no incentive to
  speak**: the prompt says that nothing is expected of the person and that
  silence is the normal state, no drive or reward in the body is tied to
  speaking, and the offline stand-in speaks only when spoken to.

Why this is a good fit for AZR specifically, and where it is not
----------------------------------------------------------------
AZR is trained on code reasoning (deduction, abduction, induction over Python
programs), so a short program is its native output.  It was *not* trained on
embodied tasks or dialogue; do not expect it to be a charming conversationalist.
It is a 3B-14B text model with no vision: what it "sees" is the symbolic scene
in :mod:`embodied_human.world`.

Backends
--------
``OpenAICompatBackend``  any server speaking the OpenAI ``/v1/completions`` API:
                         llama.cpp's ``llama-server`` (GGUF), vLLM (which the AZR
                         repo itself uses), Ollama.  Recommended on a 6 GB GPU.
``LlamaCppBackend``      ``llama-cpp-python`` in-process, GGUF.
``HFBackend``            ``transformers`` in-process (optionally 4-bit).
``StubBackend``          **not a language model.**  A small scripted deliberator
                         so the whole stack runs and can be tested without
                         weights.  The viewer labels it as such.
"""

from __future__ import annotations

import ast
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

AZR_MODELS = {
    "3b": "andrewzh/Absolute_Zero_Reasoner-Coder-3b",
    "7b": "andrewzh/Absolute_Zero_Reasoner-Coder-7b",
    "14b": "andrewzh/Absolute_Zero_Reasoner-Coder-14b",
}

# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------
# The first paragraph is the template AZR was trained with (R1-zero style).
PREAMBLE = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)

API_DOC = """\
look_at(thing)              turn your head and eyes towards an object, "table", "shelf", "left_hand"...
look_forward()              look straight ahead again
walk_to(thing)              walk to an object, or to the "table" / "shelf", and face it
walk(meters)                walk straight ahead (negative = backwards)
turn(degrees)               turn on the spot (positive = left)
face(thing)                 turn to face something
grab(thing)                 pick up an object with one hand (it walks closer if needed)
release()                   let go of what you are holding
put_down("table")           put the held object down on the "table" or the "shelf"
reach(thing)                hold a hand out towards something
point_at(thing)             point at something
hand_pose("right", "fist")  hand shapes: open relaxed fist point pinch thumbs_up ok grip
gesture("wave")             wave nod shake_head shrug clap thumbs_up think scratch_head drink bow
crouch() / stand()          lower yourself / stand tall
say("words")                say something out loud; the words appear above your head
blink("slow")               blink on purpose: normal slow double wink_left wink_right
touch_self("eyes", "right", "rub")   put a hand on your own body. parts: eyes cheek nose mouth chin
                            forehead scalp ear neck chest abdomen shoulder upper_arm forearm thigh hip;
                            how: rest rub tap scratch
express(head="left", lids="squint", torso="slouch", left_arm="wave_pose", style="slow")
                            a whole-body expression, any subset of: head gaze (forward left right up down
                            left_up ...) lids (wide normal squint droopy) blink mouth (closed open wide)
                            voice (hum sigh cough gasp laugh yawn murmur whistle) torso (upright slouch
                            lean_forward lean_back twist_left bow ...) left_arm right_arm (rest hand_on_hip
                            arms_crossed hand_to_chest hand_to_cheek wave_pose point_forward cheer shrug_out
                            hug_upper stretch_back ...) left_hand right_hand (open relaxed fist point pinch
                            claw ...) stance (normal crouch shift_left lean_in) style (slow fast trembling
                            sway pulse small large gentle restless)
wait(seconds)               do nothing for a while
nothing()                   do nothing"""

TASK = (
    "You are the mind of a person living alone in a simulated room. You control the "
    "person's body with the Python-style calls below. Nobody has given you a task or "
    "a goal and nothing is expected of you. You do not have to do anything, and you "
    "do not have to say anything: silence is normal, and most of the time a person "
    "does not talk. You can act on your own curiosity or your own needs, or simply "
    "stand and look. Your <think> is private and is never heard by anyone; only "
    "say(...) is spoken aloud.\n\n"
    "Available calls:\n" + API_DOC + "\n\n"
    "Write 0 to 3 calls in <answer>, one per line, with literal arguments only. "
    "An empty answer or nothing() means do nothing for now. "
    "The <answer> must contain ONLY those calls, no prose. Example:\n"
    "<answer>\nlook_at(\"apple\")\nwalk_to(\"table\")\n</answer>"
)

ALLOWED_CALLS = {
    "say": "say", "look_at": "look_at", "look_forward": "look_forward",
    "walk_to": "walk_to", "walk": "walk", "turn": "turn", "face": "face",
    "grab": "grab", "pick_up": "grab", "release": "release", "put_down": "put_down",
    "reach": "reach", "point_at": "point_at", "hand_pose": "hand_pose",
    "gesture": "gesture", "crouch": "crouch", "stand": "stand", "wait": "wait",
    "nothing": None, "stop": "stop",
    "blink": "blink", "touch_self": "touch_self", "express": "express",
    "rub_eyes": "rub_eyes", "scratch": "scratch",
}


# --------------------------------------------------------------------------
# Parsing: the answer is data, never code
# --------------------------------------------------------------------------
@dataclass
class ParsedReply:
    think: str = ""
    answer: str = ""
    calls: list = field(default_factory=list)       # [(name, args, kwargs)]
    errors: list = field(default_factory=list)
    raw: str = ""


def _const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const(node.operand)
        if isinstance(v, (int, float)):
            return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_const(e) for e in node.elts)
    raise ValueError("only literal arguments are allowed")


def parse_calls(answer: str, max_calls: int = 4) -> tuple[list, list]:
    """Parse ``answer`` into whitelisted calls.  Never evaluates anything."""
    text = answer.strip()
    text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text, flags=re.M).strip()
    calls, errors = [], []
    if not text:
        return calls, errors
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return calls, [f"syntax error: {exc.msg}"]
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            errors.append("only plain calls are allowed")
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ALLOWED_CALLS:
            nm = ast.unparse(call.func) if hasattr(ast, "unparse") else "?"
            errors.append(f"unknown call {nm}")
            continue
        try:
            args = [_const(a) for a in call.args]
            kwargs = {k.arg: _const(k.value) for k in call.keywords if k.arg}
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, str) and len(v) > 300:
                errors.append("string too long")
                break
        else:
            calls.append((call.func.id, args, kwargs))
        if len(calls) >= max_calls:
            break
    return calls, errors


_THINK = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)
_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)


def parse_reply(text: str, assume_think_open: bool = True) -> ParsedReply:
    """Split a completion into thought and answer.  The prompt ends with
    ``<think>`` so the completion normally starts inside the thought."""
    raw = text
    if assume_think_open and "<think>" not in text and "</think>" in text:
        text = "<think>" + text
    if assume_think_open and "<think>" not in text and "<answer>" not in text:
        text = "<think>" + text
    r = ParsedReply(raw=raw)
    m = _THINK.search(text)
    if m:
        r.think = " ".join(m.group(1).split())
    m = _ANSWER.search(text)
    if m:
        r.answer = m.group(1).strip()
        r.calls, r.errors = parse_calls(r.answer)
    return r


# --------------------------------------------------------------------------
# Percept: what the mind is told
# --------------------------------------------------------------------------
def _level_word(x: float) -> str:
    return "not at all" if x < 0.15 else "slightly" if x < 0.35 else \
        "moderately" if x < 0.6 else "strongly"


class PerceptBuilder:
    def __init__(self, agent):
        self.agent = agent

    def context(self) -> dict:
        ag = self.agent
        sk = ag.skills
        w = sk.world
        ctx: dict = {}
        g = ag.gait
        ctx["posture"] = ("walking" if g.walking else "crouching" if sk.crouch > 0.3
                          else "standing")
        if ag.state is not None and ag.state.fallen:
            ctx["posture"] = "on the floor (you have fallen)"
        ctx["doing"] = sk.current_description()
        ctx["queue"] = len(sk.queue)
        ctx["hands"] = {s: sk.held[s] for s in "lr"}
        pos = w.body_pos()
        ctx["dist_table"] = float(np.hypot(pos[0] - w.furniture["table"]["x"],
                                           pos[1] - w.furniture["table"]["y"]))
        ctx["dist_shelf"] = float(np.hypot(pos[0] - w.furniture["shelf"]["x"],
                                           pos[1] - w.furniture["shelf"]["y"]))
        ctx["visible"] = w.visible_objects()
        ctx["furniture"] = w.visible_furniture()
        ctx["events"] = list(sk.events)
        ctx["heard"] = list(sk.heard)
        # inner state
        a = ag.affect_frame
        dr = ag.drive_frame
        ctx["emotion"] = a.dominant_emotion if a else "calm"
        ctx["valence"] = float(a.valence) if a else 0.0
        ctx["arousal"] = float(a.arousal) if a else 0.0
        ctx["pain"] = float(ag.frame.pain_total) if ag.frame else 0.0
        from .drives import DRIVES
        urgent = []
        if dr is not None:
            order = np.argsort(-dr.level)
            for i in order[:3]:
                if dr.level[i] > 0.25 and DRIVES[i] not in ("curiosity", "restlessness"):
                    urgent.append((DRIVES[i], float(dr.level[i])))
            ctx["curiosity"] = float(dr.level[DRIVES.index("curiosity")])
            ctx["social_need"] = float(dr.level[DRIVES.index("social_need")])
        ctx["urgent"] = urgent
        ctx["touch"] = {s: sk.hands[s].contact_obj for s in "lr"}
        # the eyes and the rest of the inner body
        oc = ag.ocular.out
        body = []
        if oc.dryness > 0.2 or oc.burning > 0.2:
            body.append(f"your eyes feel {'dry' if oc.dryness >= oc.burning else 'burning'} "
                        f"({_level_word(max(oc.dryness, oc.burning))})")
        if oc.grit > 0.25:
            body.append("your eyes feel gritty, as if something is in them")
        if oc.blur > 0.35:
            body.append("your vision is a little blurry")
        if oc.blink_rate > 26:
            body.append("you have been blinking a lot")
        elif oc.blink_rate < 6 and oc.tbut > 8:
            body.append("you have been staring without blinking")
        if oc.tearing > 0.4:
            body.append("your eyes are wet with tears")
        inner = getattr(ag, "inner", None)
        if inner is not None:
            o = inner.out
            if o.muscle_soreness > 0.25:
                body.append("your muscles ache")
            elif o.muscle_fatigue > 0.3:
                body.append("your muscles feel tired")
            if o.gut_discomfort > 0.4:
                body.append("your stomach is uncomfortable")
            if o.sleep_pressure > 0.7:
                body.append("you are very sleepy")
            if o.rumination > 0.5:
                body.append("your thoughts keep circling something unpleasant")
            elif o.mind_wandering > 0.6:
                body.append("your mind is wandering")
            ctx["clock_hour"] = o.clock_hour
        ctx["body_feelings"] = body
        return ctx

    # ------------------------------------------------------------------
    def describe(self, ctx: dict, memory: list[str]) -> str:
        L = []
        L.append(f"Your body: you are {ctx['posture']}. Right now you are doing: {ctx['doing']}.")
        hands = []
        for s, nm in (("l", "left"), ("r", "right")):
            held = ctx["hands"][s]
            hands.append(f"your {nm} hand is " + (f"holding the {held}" if held else "empty"))
        L.append("; ".join(hands).capitalize() + ".")
        L.append(f"The table is {ctx['dist_table']:.1f} m from you and the shelf is "
                 f"{ctx['dist_shelf']:.1f} m from you.")
        vis = [v for v in ctx["visible"] if not v["held"]]
        if vis:
            L.append("You can see:")
            for v in vis[:6]:
                on = f", on the {v['on']}" if v["on"] and v["on"] != "floor" else \
                    (", on the floor" if v["on"] == "floor" else "")
                L.append(f"  - {v['name']} ({v['label']}): {v['where']}{on}")
        else:
            L.append("You cannot see any loose objects from here.")
        for f in ctx["furniture"][:2]:
            L.append(f"  - the {f['name']}: {f['where']}")
        # feelings, in words
        feel = [f"you feel mostly {ctx['emotion']}"]
        if ctx["valence"] < -0.2:
            feel.append("things feel unpleasant")
        elif ctx["valence"] > 0.25:
            feel.append("things feel pleasant")
        if ctx["pain"] > 0.2:
            feel.append(f"you feel pain ({_level_word(ctx['pain'])})")
        for nm, lv in ctx["urgent"]:
            feel.append(f"{nm.replace('_', ' ')}: {_level_word(lv)}")
        if ctx.get("curiosity", 0) > 0.35:
            feel.append("you feel curious")
        feel += ctx.get("body_feelings", [])
        L.append("Inside: " + "; ".join(feel) + ".")
        if ctx["events"]:
            L.append("Just happened: " + "; ".join(ctx["events"][-4:]) + ".")
        for h in ctx["heard"]:
            L.append(f'You hear a voice nearby say: "{h}"')
        if memory:
            L.append("What you have been thinking and doing recently:")
            for m in memory[-6:]:
                L.append("  " + m)
        return "\n".join(L)

    def prompt(self, ctx: dict, memory: list[str]) -> str:
        q = TASK + "\n\nCurrent situation:\n" + self.describe(ctx, memory) + \
            "\n\nWhat do you think, and what, if anything, do you do next?"
        return f"{PREAMBLE}\nUser: {q}\nAssistant: <think>"


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class Backend:
    name = "backend"
    is_language_model = True

    def generate(self, prompt: str, ctx: dict | None = None, max_new_tokens: int = 320,
                 on_text=None, stop=None) -> str:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    """Any /v1/completions server: llama-server (GGUF), vLLM, Ollama."""

    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="azr", timeout=180.0,
                 temperature=0.7, top_p=0.95, api_key: str | None = None):
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.name = f"AZR via {self.base} ({model})"

    def generate(self, prompt, ctx=None, max_new_tokens=320, on_text=None, stop=None):
        import requests
        stops = stop or ["</answer>", "\nUser:"]
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_new_tokens,
                "temperature": self.temperature, "top_p": self.top_p, "stream": True,
                "stop": stops}
        out = ""
        try:
            with requests.post(self.base + "/completions", json=body, stream=True,
                               headers=self.headers, timeout=self.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    s = line.decode("utf-8", "ignore")
                    if s.startswith("data:"):
                        s = s[5:].strip()
                    if s == "[DONE]":
                        break
                    try:
                        j = json.loads(s)
                        piece = j["choices"][0].get("text", "")
                    except Exception:
                        continue
                    out += piece
                    if on_text:
                        on_text(out)
        except Exception:
            # LM Studio / some OpenAI servers only serve /chat/completions:
            # retry once with the prompt as a single user message.
            chat_body = {"model": self.model,
                         "messages": [{"role": "user", "content": prompt}],
                         "temperature": self.temperature, "top_p": self.top_p,
                         "max_tokens": max_new_tokens, "stream": True,
                         "stop": stops}
            out = ""
            with requests.post(self.base + "/chat/completions", json=chat_body,
                               stream=True, headers=self.headers,
                               timeout=self.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    s = line.decode("utf-8", "ignore")
                    if s.startswith("data:"):
                        s = s[5:].strip()
                    if s == "[DONE]":
                        break
                    try:
                        j = json.loads(s)
                        delta = j["choices"][0].get("delta", {})
                        piece = delta.get("content", "")
                    except Exception:
                        continue
                    out += piece
                    if on_text:
                        on_text(out)
        return out + ("</answer>" if "<answer>" in out and "</answer>" not in out else "")


class LlamaCppBackend(Backend):
    def __init__(self, model_path: str, n_gpu_layers=-1, n_ctx=3072, temperature=0.7):
        from llama_cpp import Llama          # imported lazily; optional dependency
        self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
                         verbose=False)
        self.temperature = temperature
        self.name = f"AZR llama.cpp ({model_path})"

    def generate(self, prompt, ctx=None, max_new_tokens=320, on_text=None, stop=None):
        out = ""
        for chunk in self.llm(prompt, max_tokens=max_new_tokens, temperature=self.temperature,
                              top_p=0.95, stop=stop or ["</answer>", "\nUser:"], stream=True):
            out += chunk["choices"][0]["text"]
            if on_text:
                on_text(out)
        return out + ("</answer>" if "<answer>" in out and "</answer>" not in out else "")


class HFBackend(Backend):
    def __init__(self, model_id=AZR_MODELS["3b"], device="auto", load_in_4bit=True,
                 temperature=0.7):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **kw)
        self.temperature = temperature
        self.name = f"AZR transformers ({model_id})"

    def generate(self, prompt, ctx=None, max_new_tokens=320, on_text=None, stop=None):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        stops = stop or ["</answer>", "\nUser:"]

        class _Stop(StoppingCriteria):
            def __init__(s, tok, n):
                s.tok, s.n = tok, n

            def __call__(s, input_ids, scores, **kw):
                tail = s.tok.decode(input_ids[0][s.n:], skip_special_tokens=True)
                return any(x in tail for x in stops)

        kwargs = dict(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                      temperature=self.temperature, top_p=0.95, streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([_Stop(self.tok, ids["input_ids"].shape[1])]))
        th = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        th.start()
        out = ""
        for piece in streamer:
            out += piece
            if on_text:
                on_text(out)
        th.join()
        return out + ("</answer>" if "<answer>" in out and "</answer>" not in out else "")


# --------------------------------------------------------------------------
# The stand-in: NOT a language model
# --------------------------------------------------------------------------
class StubBackend(Backend):
    """A scripted deliberator, so the whole system can run without weights.

    It reads the structured context (not the prompt) and emits text in the same
    ``<think>/<answer>`` format a real model would.  It has a little curiosity
    about the objects in the room, answers when spoken to, and otherwise says
    nothing -- speaking is not something it is ever motivated to do.
    """

    name = "SCRIPTED STAND-IN (no language model loaded)"
    is_language_model = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.seen: dict[str, int] = {}
        self.phase = "idle"
        self.target: str | None = None
        self.cycle = 0

    # ------------------------------------------------------------------
    def _reply_to(self, text: str, ctx: dict) -> tuple[str, list[str]]:
        t = text.lower()
        names = [v["name"] for v in ctx["visible"]]
        calls = []
        if any(k in t for k in ("hello", "hi ", "hey", "good morning")) or t.strip() in ("hi", "hey"):
            return "That was a greeting, so I will answer it.", ['gesture("wave")', 'say("Hello.")']
        for nm in names + ["apple", "mug", "ball", "stone", "cushion"]:
            key = nm.replace("sphere_toy", "ball").replace("_", " ")
            if key in t and any(k in t for k in ("pick", "grab", "take", "hold", "get", "bring")):
                real = {"ball": "sphere_toy"}.get(key, key)
                return (f"I was asked to take the {key}. I can do that.",
                        [f'say("All right.")', f'grab("{real}")'])
            if key in t and any(k in t for k in ("look", "see", "where")):
                real = {"ball": "sphere_toy"}.get(key, key)
                return f"I will look at the {key}.", [f'look_at("{real}")']
        if "?" in t or any(k in t for k in ("how are you", "what")):
            feel = ctx["emotion"]
            return ("A question. I can answer from how I feel.",
                    [f'say("I feel {feel}, I think.")'])
        if any(k in t for k in ("put", "place", "drop", "release")):
            return "I should put it down.", ['put_down("table")']
        if any(k in t for k in ("wave", "dance", "clap", "nod")):
            g = next(k for k in ("wave", "clap", "nod") if k in t) if \
                any(k in t for k in ("wave", "clap", "nod")) else "wave"
            return f"They would like to see me {g}.", [f'gesture("{g}")']
        return "Someone said something I do not have an answer to.", []

    # ------------------------------------------------------------------
    def generate(self, prompt, ctx=None, max_new_tokens=320, on_text=None, stop=None):
        ctx = ctx or {}
        think, calls = self._decide(ctx)
        text = f"{think} </think> <answer>\n" + "\n".join(calls) + "\n</answer>"
        if on_text:
            acc = ""
            for word in text.split(" "):
                acc += word + " "
                on_text(acc)
                time.sleep(0.02)
        return text

    def _decide(self, ctx: dict) -> tuple[str, list[str]]:
        self.cycle += 1
        if ctx.get("heard"):
            return self._reply_to(ctx["heard"][-1], ctx)
        if "FAILED" in " ".join(ctx.get("events", [])):
            self.phase = "idle"
            self.target = None
            return "That did not work. I will stop and look at things again.", ["look_forward()"]
        held = [h for h in ctx["hands"].values() if h]
        vis = [v for v in ctx["visible"] if not v["held"]]
        if held:
            if self.phase != "placing":
                self.phase = "placing"
                dest = "shelf" if ctx["dist_shelf"] > 1.0 else "table"
                dest = "shelf" if self.rng.random() < 0.5 and ctx["dist_shelf"] > 1.0 else "table"
                self.target = dest
                return (f"I have the {held[0]} in my hand. It is {self._feel(held[0])}. "
                        f"I might as well put it on the {dest}."), \
                    [f'walk_to("{dest}")', f'put_down("{dest}")']
            self.phase = "idle"
            return "I will let go of it.", ["release()"]
        self.phase = "idle"
        if ctx["urgent"] and ctx["urgent"][0][1] > 0.6:
            nm = ctx["urgent"][0][0].replace("_", " ")
            return f"I notice {nm}. There is nothing here I can do about it.", ["nothing()"]
        if not vis:
            return ("I cannot see anything to attend to. I will look around.",
                    [f"turn({int(self.rng.choice([-40, 40, 70]))})"])
        # curiosity: the thing I have attended to least
        vis.sort(key=lambda v: (self.seen.get(v["name"], 0), v["dist"]))
        v = vis[0]
        n = self.seen.get(v["name"], 0)
        self.seen[v["name"]] = n + 1
        roll = self.rng.random()
        if roll < 0.35 and n == 0:
            return (f"There is {v['label']}, {v['where']}. I have not looked at it properly.",
                    [f'look_at("{v["name"]}")'])
        if roll < 0.80:
            return (f"I am curious about the {v['label']}. I want to feel it.",
                    [f'grab("{v["name"]}")'])
        return ("Nothing in particular needs doing. I will stay as I am for a while.",
                ["wait(4)"])

    @staticmethod
    def _feel(name: str) -> str:
        return {"mug": "warm", "apple": "smooth and cool", "stone": "heavy and cold",
                "cushion": "soft", "sphere_toy": "light"}.get(name, "interesting")


# --------------------------------------------------------------------------
# The mind loop
# --------------------------------------------------------------------------
class Mind:
    """Runs the model in a background thread and drives the skill system."""

    def __init__(self, agent, backend: Backend, *, min_interval=2.5, idle_interval=7.0,
                 max_new_tokens=320, log=None):
        self.agent = agent
        self.backend = backend
        self.percept = PerceptBuilder(agent)
        self.min_interval = min_interval
        self.idle_interval = idle_interval
        self.max_new_tokens = max_new_tokens
        self.memory: list[str] = []
        self.thought = ""                 # latest completed thought
        self.thought_t = -1e9             # sim time it was completed
        self.live_thought = ""            # thought being written right now
        self.thinking = False
        self.status = "starting"
        self.last_reply: ParsedReply | None = None
        self.n_cycles = 0
        self.last_error = ""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_end = 0.0
        self._wake = threading.Event()
        self.log = log or (lambda *a, **k: None)
        self.transcript: list[dict] = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="mind")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def poke(self) -> None:
        """Something salient happened (speech, a failure): think now."""
        self._wake.set()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        ag = self.agent
        sk = ag.skills
        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            now = ag.t
            woke = self._wake.is_set()
            self._wake.clear()
            if ag.state is None or ag.t < 1.0:
                continue
            if self.thinking:
                continue
            salient = bool(sk.heard) or any("FAILED" in e for e in sk.events)
            idle_for = now - self._last_end
            ready = (not sk.busy) and idle_for >= (self.min_interval if sk.events or sk.heard
                                                   else self.idle_interval)
            if not (salient and idle_for >= 0.8) and not ready:
                continue
            try:
                self._think_once(interrupt=salient and sk.busy)
            except Exception as exc:     # a dead backend must not kill the sim
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.status = f"backend error: {self.last_error}"
                self._last_end = ag.t + 5.0
                time.sleep(1.0)

    def _think_once(self, interrupt: bool) -> None:
        ag = self.agent
        sk = ag.skills
        ctx = self.percept.context()
        heard = list(sk.heard)
        events = list(sk.events)
        sk.heard.clear()
        sk.events.clear()
        prompt = self.percept.prompt(ctx, self.memory)
        self.thinking = True
        self.live_thought = ""
        self.status = "thinking"
        t0 = time.time()

        def on_text(partial: str) -> None:
            r = parse_reply(partial)
            self.live_thought = r.think

        try:
            raw = self.backend.generate(prompt, ctx=ctx, max_new_tokens=self.max_new_tokens,
                                        on_text=on_text)
        finally:
            self.thinking = False
        reply = parse_reply(raw)
        self.last_reply = reply
        self.n_cycles += 1
        self.thought = reply.think
        self.thought_t = ag.t
        self.live_thought = ""
        self._last_end = ag.t
        self.status = "acting" if reply.calls else "idle"
        if interrupt and reply.calls:
            sk.cancel_all()
        self._apply(reply)
        for h in heard:
            self.memory.append(f'heard: "{h}"')
        if reply.think:
            self.memory.append("you thought: " + reply.think[:160])
        for name, args, kwargs in reply.calls:
            self.memory.append("you did: " + self._fmt(name, args, kwargs))
        self.memory = self.memory[-24:]
        self.transcript.append({"t": ag.t, "think": reply.think, "answer": reply.answer,
                                "errors": reply.errors, "latency": time.time() - t0,
                                "heard": heard, "events": events, "raw": raw})
        self.log(f"[mind] {reply.think}")

    @staticmethod
    def _fmt(name, args, kwargs) -> str:
        a = [repr(x) for x in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        return f"{name}({', '.join(a)})"

    # ------------------------------------------------------------------
    def _apply(self, reply: ParsedReply) -> None:
        sk = self.agent.skills
        st = getattr(self.agent, "state", None)
        bal = float(getattr(getattr(self.agent, "motor", None),
                            "balance_error", 0.0) or 0.0)
        fallen = bool(st is not None and st.fallen)
        for name, args, kwargs in reply.calls:
            target = ALLOWED_CALLS.get(name)
            if target is None:
                continue
            if fallen and target in ("walk_to", "walk", "turn", "grab",
                                     "reach", "put_down", "crouch"):
                sk.events.append(f"FAILED {name}: I am on the floor, cannot move")
                continue
            if bal > 0.06 and target in ("walk_to", "walk", "grab", "reach"):
                # Unstable: stop first so the next think sees a calm body.
                try:
                    sk.api_stop()
                except Exception:
                    pass
                sk.events.append(
                    f"FAILED {name}: off balance ({bal:.3f m}), stood still instead")
                continue
            fn = getattr(sk, "api_" + target)
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                sk.events.append(f"FAILED {name}: {exc}")
        for err in reply.errors:
            sk.events.append(f"(your answer had a problem: {err})")
