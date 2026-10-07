"""Per-instance identity and thought/action transcripts.

Every simulated person gets its own identifier (``P-YYYYMMDD-<hex>``), and
every thought the mind completes, every action it dispatches, everything said
to it and everything it says is appended to that instance's own ``.txt``
file.  One line per event, UTC timestamp plus sim time, so the file reads
naturally and is still trivially parseable::

    [2026-10-07T12:00:09Z t=13.1s] THOUGHT: I will look at the apple.
    [2026-10-07T12:00:09Z t=13.1s] ACTION: look_at("apple") | say("Hello.")

Stdlib only.  Thread-safe: the mind loop thread and the interview driver
may both append.
"""

from __future__ import annotations

import datetime
import re
import threading
import uuid
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


def new_instance_id() -> str:
    """Mint a fresh instance identifier, e.g. ``P-20261007-3f9a2c1e``."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    return f"P-{stamp}-{uuid.uuid4().hex[:8]}"


def sanitize_instance_id(text: str) -> str:
    """Clean a user-supplied id; mint a fresh one if it is unusable."""
    s = " ".join(str(text or "").split())
    if s and len(s) <= 64 and _ID_RE.match(s):
        return s
    return new_instance_id()


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _one_line(text: str, limit: int = 2000) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit] + " ...[truncated]"


class InstanceLog:
    """Append-only ``.txt`` transcript for one simulation instance."""

    def __init__(self, instance_id: str, log_dir: str | Path, *,
                 seed: int | None = None, backend: str = "",
                 note: str = "") -> None:
        self.instance_id = sanitize_instance_id(instance_id)
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.instance_id}.txt"
        self._lock = threading.Lock()
        header = [
            f"=== INSTANCE {self.instance_id} ===",
            f"started_utc: {_utcnow()}",
        ]
        if seed is not None:
            header.append(f"seed: {seed}")
        if backend:
            header.append(f"backend: {_one_line(backend, 160)}")
        if note:
            header.append(f"note: {_one_line(note, 300)}")
        header.append("---")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")

    # ------------------------------------------------------------------
    def append(self, tag: str, text: str, sim_t: float | None = None) -> None:
        """Append one event line.  ``tag`` is e.g. THOUGHT, ACTION, HEARD."""
        t = f" t={sim_t:.1f}s" if sim_t is not None else ""
        line = f"[{_utcnow()}{t}] {tag}: {_one_line(text)}\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    # ---- convenience shorthands --------------------------------------
    def thought(self, text: str, sim_t: float | None = None) -> None:
        if text:
            self.append("THOUGHT", text, sim_t)

    def action(self, calls: str, sim_t: float | None = None, extra: str = "") -> None:
        body = calls if not extra else f"{calls} | {extra}"
        self.append("ACTION", body or "(no calls)", sim_t)

    def heard(self, text: str, sim_t: float | None = None, source: str = "") -> None:
        src = f" (source: {source})" if source else ""
        self.append("HEARD", f'"{text}"{src}', sim_t)

    def said(self, text: str, sim_t: float | None = None) -> None:
        self.append("SAID", f'"{text}"', sim_t)

    def interview(self, text: str, model: str = "", rationale: str = "",
                  sim_t: float | None = None) -> None:
        tail = f" | rationale: {rationale}" if rationale else ""
        self.append("INTERVIEW_QUESTION", f'"{text}" (via {model}){tail}' if model
                    else f'"{text}"{tail}', sim_t)

    def event(self, text: str, sim_t: float | None = None) -> None:
        self.append("EVENT", text, sim_t)
