"""
Speech: the mouth, and the words above the head.

There is no drive to speak anywhere in this package.  Nothing rewards an
utterance, nothing in the drives or affect systems is raised by silence, and
the mind is told in plain words that it is free to say nothing.  Speech is an
*action available to the mind*, exactly like reaching; this module only makes
the action look and sound like speaking when the mind chooses it.

* **Mouth.**  Fine articulation is not simulated.  The jaw opens while the
  person is speaking, in a rhythm that follows the syllables of what is being
  said (about 4 per second, wider for open vowels, closed on pauses), and
  closes when it stops.  The jaw joint is driven kinematically, like the eyes.
* **Words.**  The utterance is revealed word by word in step with the jaw and
  kept above the head while it is spoken, then lingers and fades.
* **Voice (optional).**  If a text-to-speech engine is available the words can
  also be spoken aloud (Windows SAPI, no extra install).  Off by default.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

_VOWEL_OPEN = {"a": 1.0, "o": 0.9, "u": 0.55, "e": 0.6, "i": 0.4, "y": 0.4}
_SYLLABLE_RATE = 4.3          # syllables per second
JAW_MAX = 0.36                # rad; the joint allows 0.42


def _syllables(word: str) -> list[float]:
    """Crude syllable nuclei: one per vowel group, carrying its openness."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return []
    groups = re.findall(r"[aeiouy]+", w)
    if not groups:
        return [0.5]
    out = []
    for g in groups:
        out.append(max(_VOWEL_OPEN.get(c, 0.5) for c in g))
    # a trailing silent 'e' is not a syllable
    if len(out) > 1 and w.endswith("e") and not w.endswith(("le", "ee")):
        out = out[:-1]
    return out


@dataclass
class Utterance:
    text: str
    words: list[str]
    word_start: list[float]            # seconds from the start of speech
    syl_times: list[tuple[float, float, float]]   # (t0, t1, openness)
    duration: float
    t0: float = 0.0
    done: bool = False


def plan_utterance(text: str) -> Utterance:
    text = " ".join(text.strip().split())
    words = text.split(" ") if text else []
    t = 0.12                                         # breath before speaking
    starts, syl = [], []
    dur_syl = 1.0 / _SYLLABLE_RATE
    for w in words:
        starts.append(t)
        for op in _syllables(w) or [0.3]:
            syl.append((t, t + dur_syl * 0.92, op))
            t += dur_syl
        t += 0.05
        if w.endswith((",", ";", ":")):
            t += 0.18
        elif w.endswith((".", "!", "?", "...")):
            t += 0.34
    return Utterance(text=text, words=words, word_start=starts, syl_times=syl,
                     duration=t + 0.10)


class Speech:
    """Queue of utterances, jaw animation and the visible text."""

    def __init__(self, voice: bool = False):
        self.queue: list[Utterance] = []
        self.current: Utterance | None = None
        self.jaw = 0.0
        self._jaw_target = 0.0
        self.t = 0.0
        self.voice = voice
        self.last_text = ""
        self.last_end = -1e9
        self.history: list[tuple[float, str]] = []
        self._tts = None
        self._clock = 0.0

    # ------------------------------------------------------------------
    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.queue.append(plan_utterance(text[:400]))

    @property
    def speaking(self) -> bool:
        return self.current is not None

    def shut_up(self) -> None:
        self.queue.clear()
        if self.current is not None:
            self.last_end = self._clock
        self.current = None

    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        self._clock += dt
        if self.current is None and self.queue:
            self.current = self.queue.pop(0)
            self.current.t0 = self._clock
            self.last_text = self.current.text
            self.history.append((self._clock, self.current.text))
            if self.voice:
                self._speak_aloud(self.current.text)
        target = 0.0
        if self.current is not None:
            u = self.current
            el = self._clock - u.t0
            for t0, t1, op in u.syl_times:
                if t0 <= el <= t1:
                    ph = (el - t0) / max(t1 - t0, 1e-6)
                    target = JAW_MAX * (0.35 + 0.65 * op) * np.sin(np.pi * ph) ** 0.8
                    break
            if el >= u.duration:
                self.last_end = self._clock
                self.current = None
                target = 0.0
        # the jaw is a heavy-ish mass: first-order lag, faster to open than close
        tau = 0.035 if target > self.jaw else 0.05
        self.jaw += (target - self.jaw) * (1.0 - np.exp(-dt / tau))

    # ------------------------------------------------------------------
    def visible_text(self) -> tuple[str, float]:
        """(text revealed so far, opacity 0..1) for the bubble above the head."""
        if self.current is not None:
            u = self.current
            el = self._clock - u.t0
            n = sum(1 for s in u.word_start if s <= el)
            return " ".join(u.words[:n]), 1.0
        if self.last_text:
            age = self._clock - self.last_end
            hold = 3.5 + 0.06 * len(self.last_text)
            if age < hold:
                return self.last_text, 1.0
            if age < hold + 1.2:
                return self.last_text, 1.0 - (age - hold) / 1.2
        return "", 0.0

    # ------------------------------------------------------------------
    def _speak_aloud(self, text: str) -> None:
        """Fire-and-forget Windows SAPI; silently does nothing elsewhere."""
        if not sys.platform.startswith("win"):
            return
        safe = text.replace("'", " ").replace('"', " ")
        cmd = ("Add-Type -AssemblyName System.Speech; "
               "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
               "$s.Rate = 0; $s.Speak('" + safe + "')")
        try:
            if self._tts is not None and self._tts.poll() is None:
                self._tts.terminate()
            self._tts = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            self._tts = None
