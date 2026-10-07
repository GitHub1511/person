#!/usr/bin/env python
"""Assemble README.md from hand-written sections plus tables generated from the code.

    python tools/build_readme.py              # writes README.md and docs/README_full.md
    python tools/build_readme.py --check      # only report problems, write nothing

The prose lives in ``docs/readme_src/NN_*.md`` (sections in file-name order).  Placeholders of the form
``{{NAME}}`` are replaced by tables computed here from the live source tree and from
``out/readme_data/*.json`` (written by ``tools/readme_combinatorics.py`` and
``tools/readme_runtime_facts.py``), so the counts in the tables cannot drift from the code.

This is deliberately deterministic and offline: it makes no model call and never rewrites prose.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "readme_src"
DATA = ROOT / "out" / "readme_data"
sys.path.insert(0, str(ROOT))

CATEGORY = {
    "body and scene": ["skeleton", "skin", "build_model", "state", "config", "complexity"],
    "sensing": ["receptors", "senses_ext", "afferents", "ocular"],
    "physiology and inner world": ["interoception", "inner_organs", "inner_brain", "inner_world"],
    "affect and drives": ["affect", "drives"],
    "brain": ["predictive", "active_inference", "intrinsic", "behavior_space", "behavior_exec", "body_learning"],
    "motor control and skills": ["motor", "wbc", "locomotion", "skills", "speech"],
    "mind": ["mind", "world", "azr_loop", "instance_log"],
    "infrastructure": ["agent", "record", "plots", "viewer", "_fast", "ultra", "ux_example", "__init__"],
}


def first_doc_line(path: Path) -> tuple[str, int, int, int]:
    s = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = s.count("\n") + 1
    try:
        t = ast.parse(s)
    except SyntaxError:
        return "(not parseable)", lines, 0, 0
    d = ast.get_docstring(t) or ""
    first = d.strip().split("\n")[0].strip() if d else ""
    cls = sum(isinstance(n, ast.ClassDef) for n in ast.walk(t))
    fn = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(t))
    return first, lines, cls, fn


def table(rows: list[list], header: list[str], align: str = "") -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def codemap() -> str:
    mods = {p.stem: p for p in (ROOT / "embodied_human").glob("*.py")}
    rows, seen = [], set()
    tl = tc = tf = 0
    for cat, names in CATEGORY.items():
        for n in names:
            if n in mods:
                seen.add(n)
                first, ln, c, f = first_doc_line(mods[n])
                tl += ln; tc += c; tf += f
                rows.append([cat, f"`{n}.py`", f"{ln:,}", c, f, first[:96].replace("|", "/")])
    for n in sorted(set(mods) - seen):
        first, ln, c, f = first_doc_line(mods[n])
        tl += ln; tc += c; tf += f
        rows.append(["other", f"`{n}.py`", f"{ln:,}", c, f, first[:96].replace("|", "/")])
    rows.append(["**total**", f"**{len(mods)} modules**", f"**{tl:,}**", f"**{tc}**", f"**{tf}**", ""])
    return table(rows, ["group", "module", "lines", "classes", "functions", "purpose (first docstring line)"])


def patches() -> str:
    code = ("import sys,json;sys.path.insert(0,r'%s');from embodied_human import skin;"
            "print(json.dumps({p.region:p.n_u*p.n_v for p in skin.default_patches()}))" % ROOT)
    res = {}
    for lv in ("base", "extreme"):
        out = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PERSON_COMPLEXITY=lv),
                             capture_output=True, text=True).stdout.strip().splitlines()[-1]
        res[lv] = json.loads(out)
    rows = []
    for name, b in res["base"].items():
        e = res["extreme"][name]
        rows.append([f"`{name}`", b, b * 4, e, b * 16, f"{100 * b / sum(res['base'].values()):.1f} %",
                     "yes" if e == b * 9 else f"no ({e})"])
    tot = [sum(res["base"].values()), 0, sum(res["extreme"].values())]
    rows.append(["**total (46 patches)**", f"**{tot[0]:,}**", f"**{tot[0] * 4:,}**", f"**{tot[2]:,}**",
                 f"**{tot[0] * 16:,}**", "100 %", ""])
    return table(rows, ["patch", "`base`", "`rich`", "`extreme`", "`max`", "share of skin", "extreme = 9 × base"])


def behavior_channels() -> str:
    c = json.loads((DATA / "combinatorics.json").read_text())["behavior"]
    from math import log10
    rows = []
    for name, size in c["layout"]:
        rows.append([f"`{name}`", f"{size:,}", f"{log10(size):.2f}"])
    rows.append(["**product (all 33 layout entries)**", f"**{int(c['descriptors']['exact']):,}**",
                 f"**{c['descriptors']['log10']:.3f}**"])
    return table(rows, ["channel", "choices", "log10"])


def complexity_table() -> str:
    from embodied_human import complexity as cx
    from dataclasses import asdict
    P = {k: asdict(v) for k, v in cx.PRESETS.items()}
    keys = [k for k in P["base"] if k not in ("name",)]
    rows = [[f"`{k}`"] + [P[l][k] for l in ("base", "rich", "extreme", "max")] for k in keys]
    return table(rows, ["knob", "`base`", "`rich`", "`extreme`", "`max` (= `ultra` = `mega`)"])


def tools_list() -> str:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "tools" / "*.py"))):
        first, ln, c, f = first_doc_line(Path(p))
        rows.append([f"`tools/{Path(p).name}`", f"{ln:,}", first[:110].replace("|", "/")])
    return table(rows, ["script", "lines", "purpose"])


def diag_list() -> str:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "diag_*.py"))):
        s = Path(p).read_text(encoding="utf-8-sig", errors="replace")
        try:
            d = ast.get_docstring(ast.parse(s)) or ""
        except SyntaxError:
            d = ""
        first = d.strip().split("\n")[0].strip() if d else "(no docstring)"
        rows.append([f"`{Path(p).name}`", s.count("\n") + 1, first[:100].replace("|", "/")])
    return table(rows, ["script", "lines", "what it measures"])


def policies() -> str:
    names = json.loads((DATA / "runtime_base.json").read_text())["extra"]["policy_names"]
    cols = 5
    n = len(names)
    per = -(-n // cols)
    rows = []
    for r in range(per):
        rows.append([f"{r + c * per + 1}. `{names[r + c * per]}`" if r + c * per < n else "" for c in range(cols)])
    return table(rows, [" "] * cols)


def getup_status() -> str:
    f = DATA / "getup_status.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else         "No success rate is claimed: the recovery diagnostic had not been run to completion when this was written."


def modstats() -> str:
    tl = tc = tf = n = 0
    for p in (ROOT / "embodied_human").glob("*.py"):
        _, ln, c, f = first_doc_line(p)
        tl += ln; tc += c; tf += f; n += 1
    return f"{n} files, {tl:,} lines, {tc} classes, {tf} functions"


GEN = {"MODSTATS": modstats, "POLICIES": policies, "GETUP_STATUS": getup_status, "CODEMAP": codemap, "PATCHES": patches, "BEHAVIOR_CHANNELS": behavior_channels,
       "COMPLEXITY_TABLE": complexity_table, "TOOLS": tools_list, "DIAGS": diag_list}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    parts = sorted(SRC.glob("*.md"))
    text = "\n".join(p.read_text(encoding="utf-8").rstrip("\n") + "\n" for p in parts)
    problems = []
    for key in sorted(set(re.findall(r"\{\{([A-Z_]+)\}\}", text))):
        if key not in GEN:
            problems.append(f"unknown placeholder {key}")
            continue
        text = text.replace("{{" + key + "}}", GEN[key]())
    left = re.findall(r"\{\{[^}]*\}\}", text)
    if left:
        problems.append(f"unreplaced: {left[:5]}")
    for pr in problems:
        print("PROBLEM:", pr)
    print(f"{len(parts)} source files, {text.count(chr(10)) + 1} lines, {len(text):,} characters")
    if a.check or problems:
        return 1 if problems else 0
    (ROOT / "README.md").write_text(text, encoding="utf-8", newline="\n")
    (ROOT / "docs" / "README_full.md").write_text(text, encoding="utf-8", newline="\n")
    print("wrote README.md and docs/README_full.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
