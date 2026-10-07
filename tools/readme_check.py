#!/usr/bin/env python
"""Check that an edited README section kept everything that is not wording.

    python tools/readme_check.py docs/readme_work/orig/05.md docs/readme_work/new/05.md

A rewrite for style may change sentences, but it must keep: every fenced code block exactly,
every inline `code span`, every number, every link target, the number of headings and their
numeric prefixes, the number of table rows, and the hedges ("not", "unverified", "toy",
"unreliable", ...) -- a caveat may be rephrased but not dropped.  Exit status 1 on any loss.
"""
import re
import sys
from collections import Counter
from pathlib import Path


def fences(t):
    return re.findall(r"```.*?```", t, flags=re.S)


def strip_fences(t):
    return re.sub(r"```.*?```", "", t, flags=re.S)


def spans(t):
    return Counter(re.findall(r"`[^`\n]+`", strip_fences(t)))


def numbers(t):
    t = strip_fences(t)
    t = re.sub(r"`[^`\n]+`", " ", t)
    t = re.sub(r"\(https?://[^)]*\)", " ", t)
    return Counter(re.findall(r"(?<![\w.])\d[\d,]*\.?\d*(?![\w])", t))


def links(t):
    return Counter(re.findall(r"\]\(([^)]+)\)", t))


def headings(t):
    return [re.match(r"(#+)\s*(\d+(?:\.\d+)*)?", l).groups() for l in strip_fences(t).split("\n") if l.startswith("#")]


def table_rows(t):
    return sum(1 for l in strip_fences(t).split("\n") if l.lstrip().startswith("|"))


HEDGES = ["not ", "no ", "never", "unverified", "unreliable", "toy", "approximate", "assist", "shortcut",
          "limitation", "fragile", "unknown", "has not", "have not", "does not", "cannot", "only", "partly"]


def hedge_count(t):
    s = strip_fences(t).lower()
    return {h: s.count(h) for h in HEDGES}


def main():
    a, b = (Path(p).read_text(encoding="utf-8") for p in sys.argv[1:3])
    bad = []
    if fences(a) != fences(b):
        bad.append("fenced code blocks changed")
    for what, f in (("inline code spans", spans), ("numbers", numbers), ("links", links)):
        ca, cb = f(a), f(b)
        lost = ca - cb
        if lost:
            bad.append(f"{what} lost: {dict(list(lost.items())[:8])}")
    if headings(a) != headings(b):
        bad.append(f"headings changed: {headings(a)} -> {headings(b)}")
    if table_rows(a) != table_rows(b):
        bad.append(f"table rows {table_rows(a)} -> {table_rows(b)}")
    ha, hb = hedge_count(a), hedge_count(b)
    total_a = sum(ha.values())
    total_b = sum(hb.values())
    if total_b < 0.7 * total_a:
        bad.append(f"caveat words fell from {total_a} to {total_b}; a limitation may have been softened")
    if re.search(r"\b(I|my|me)\b", strip_fences(b)) and not re.search(r"\b(I|my|me)\b", strip_fences(a)):
        bad.append("first person singular introduced")
    if bad:
        print("FAIL")
        for x in bad:
            print("  -", x)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
