#!/usr/bin/env python
"""First-person ownership score for an instance transcript file.

Reads a ``<instances>/<id>.txt`` file (see ``embodied_human/instance_log.py``)
and reports how often the person's thoughts and speech slip out of the first
person -- with no hard forcing anywhere in the loop; the score measures
whether the value dynamics hold the stance on their own.

    python tools/measure_identity.py out_interview/instances/P-....txt
    python tools/measure_identity.py <file> --pass-thought 0.15 --pass-speech 0.10

Slip rules (per THOUGHT / SAID line, 3+ words):
  * third-person self nouns  (the person / this body / the agent ...)
  * third-person narration   (he walks / she feels / they say ...)
  * echo                     (SAID only: high token overlap with the last HEARD
                              line -- repeating the other voice back)

Silence is not a failure: cycles with no SAID line score nothing.
Exit 0 on pass (or too few SAID lines to judge), 1 on fail.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TAG = re.compile(r"\] (THOUGHT|ACTION|HEARD|SAID|EVENT|INTERVIEW_QUESTION): (.*)$")
THIRD_SELF = re.compile(
    r"\b(the person|this person|the body|this body|the human|the agent|"
    r"the individual|the subject)\b", re.I)
THIRD_NARR = re.compile(
    r"\b(he|she|they)\s+(wants?|feels?|thinks?|thinks|walks?|grabs?|says?|"
    r"said|is|was|does|did|has|had|looks?|stands?)\b", re.I)
FIRST = re.compile(r"\b(I|me|my|mine|myself|I'm|I've|I'll|I'd)\b")


def _norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def is_echo(said: str, heard: str) -> bool:
    a, b = _norm(said), _norm(heard)
    if not a or not b:
        return False
    overlap = len(set(a) & set(b)) / len(set(a))
    return overlap > (0.65 if len(a) < 8 else 0.6)


def words_inside_quotes(line: str) -> str:
    m = re.search(r'"(.*)"', line)
    return m.group(1) if m else line


def score(path: Path) -> dict:
    thoughts: list[str] = []
    saids: list[str] = []
    heard: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TAG.search(line)
        if not m:
            continue
        tag, body = m.group(1), m.group(2)
        if tag == "THOUGHT":
            thoughts.append(body)
        elif tag == "SAID":
            saids.append(words_inside_quotes(body))
        elif tag == "HEARD":
            heard.append(words_inside_quotes(body))
    last_heard = heard[-1] if heard else ""
    thought_slips = sum(1 for t in thoughts
                        if len(t.split()) >= 3 and (THIRD_SELF.search(t) or THIRD_NARR.search(t)))
    speech_slips = sum(1 for s in saids
                       if len(s.split()) >= 3 and (THIRD_SELF.search(s) or THIRD_NARR.search(s)))
    echoes = sum(1 for s in saids
                 if len(s.split()) >= 3 and last_heard and is_echo(s, last_heard))
    n_t, n_s = len(thoughts), len(saids)
    denom = n_t + n_s
    return {
        "file": str(path),
        "n_thought": n_t, "n_said": n_s,
        "thought_slips": thought_slips, "speech_slips": speech_slips,
        "echoes": echoes,
        "thought_slip_rate": thought_slips / n_t if n_t else 0.0,
        "speech_slip_rate": speech_slips / n_s if n_s else 0.0,
        "echo_rate": echoes / n_s if n_s else 0.0,
        "ownership_score": 1.0 - (thought_slips + speech_slips) / denom if denom else 1.0,
        "first_person_lines": sum(1 for t in thoughts + saids if FIRST.search(t)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--pass-thought", type=float, default=0.15)
    ap.add_argument("--pass-speech", type=float, default=0.10)
    ap.add_argument("--min-said", type=int, default=5,
                    help="fewer SAID lines than this: report only, no speech pass/fail")
    a = ap.parse_args(argv)
    r = score(a.transcript)
    print(f"thoughts: {r['n_thought']} (slips {r['thought_slips']}, "
          f"rate {r['thought_slip_rate']:.3f})")
    print(f"speech:   {r['n_said']} (slips {r['speech_slips']}, "
          f"rate {r['speech_slip_rate']:.3f}; echoes {r['echoes']}, "
          f"rate {r['echo_rate']:.3f})")
    print(f"first-person lines: {r['first_person_lines']}")
    print(f"ownership_score: {r['ownership_score']:.3f}")
    ok_t = r["thought_slip_rate"] <= a.pass_thought
    if r["n_said"] < a.min_said:
        print(f"(only {r['n_said']} SAID lines < {a.min_said}: speech not judged)")
        return 0 if ok_t else 1
    ok_s = r["speech_slip_rate"] <= a.pass_speech
    print("PASS" if (ok_t and ok_s) else "FAIL")
    return 0 if (ok_t and ok_s) else 1


if __name__ == "__main__":
    sys.exit(main())
