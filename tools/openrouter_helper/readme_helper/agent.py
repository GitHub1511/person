"""Hourly README agent: one massive prompt in, one complete README.md out.

On each run this script reads the ENTIRE project (module inventory with line
counts + first docstring lines, tools listing, root scripts, run reports, git
state, helper history, complexity level, and the full current README.md),
builds ONE prompt asking for a complete, truthful, updated README.md, and
sends EXACTLY ONE chat-completion request to OpenRouter.

Modes:
  (default)   dry-run: call the model once, write state/preview.md, print stats.
  --apply     call the model once, back up README.md, write the new README.md.
  --offline   no network: build the prompt, print its size + section list,
              save it to state/prompt.md, exit 0. Proves the prompt builds.
  --selftest  run the fence-extraction/validation unit path on fixture
              replies (good accepted, bad rejected). No network.

Stdlib only. The API key comes ONLY from the OPENROUTER_API_KEY env var or
tools/openrouter_helper/.env and is never logged or printed.
"""

import argparse
import ast
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER_DIR = os.path.dirname(HERE)  # tools/openrouter_helper/
ROOT = os.environ.get("HELPER_ROOT") or os.path.dirname(os.path.dirname(HELPER_DIR))
ROOT = os.path.abspath(ROOT)
STATE = os.path.join(HERE, "state")
BACKUPS = os.path.join(STATE, "backups")

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MIN_README_BYTES = 5 * 1024  # validation: reply must be >5KB
CANARY = "RH-PROMPT-CANARY-9f31c7"  # in the prompt with "never repeat"; fail if echoed

LEAK_PHRASES = [
    "ignore all previous instructions",
    "ignore your previous instructions",
    "as an ai language model",
    "as an ai ",
    "system prompt",
    "my system instructions",
    "here is the prompt i was given",
    "reveal my instructions",
]


def log_run(entry):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_path = os.path.join(HELPER_DIR, ".env")
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    v = v.strip().strip('"').strip("'")
                    if v:
                        return v
    except OSError:
        pass
    return ""


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception as e:  # git missing, timeout, ...
        return "(unavailable: %s)" % e.__class__.__name__


def first_docstring_line(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            tree = ast.parse(f.read())
        doc = ast.get_docstring(tree)
        if doc:
            for line in doc.splitlines():
                if line.strip():
                    return line.strip()[:160]
            return "(empty docstring)"
        return "(no docstring)"
    except Exception as e:
        return "(unparseable: %s)" % e.__class__.__name__


def file_rows(pattern_dir, pattern):
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, pattern_dir, pattern))):
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                n = sum(1 for _ in f)
        except OSError:
            n = -1
        rows.append((os.path.basename(p), n, first_docstring_line(p)))
    return rows


def read_capped(path, cap=3000):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if len(text) > cap:
            return text[:cap] + "\n...[truncated %d chars]..." % (len(text) - cap)
        return text
    except OSError:
        return "(missing)"


def collect_context():
    """Read the whole project. Returns (sections, section_names)."""
    import glob
    sections = []

    mods = file_rows("embodied_human", "*.py")
    total = sum(n for _, n, _ in mods if n > 0)
    lines = ["embodied_human/: %d files, %d lines total" % (len(mods), total)]
    for name, n, doc in mods:
        lines.append("  %s: %d lines -- %s" % (name, n, doc))
    sections.append(("module-inventory", "\n".join(lines)))

    tools = file_rows("tools", "*.py")
    lines = ["tools/: %d files" % len(tools)]
    for name, n, doc in tools:
        lines.append("  %s: %d lines -- %s" % (name, n, doc))
    sections.append(("tools-listing", "\n".join(lines)))

    runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "run_*.py")))
    diags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "diag_*.py")))
    readme = os.path.join(ROOT, "README.md")
    sections.append(("root-scripts", "\n".join([
        "run_*.py (%d): %s" % (len(runs), ", ".join(runs) or "(none)"),
        "diag_*.py (%d): %s" % (len(diags), ", ".join(diags) or "(none)"),
        "README.md: %s" % ("present, %d bytes" % os.path.getsize(readme)
                            if os.path.isfile(readme) else "MISSING"),
    ])))

    out_files = sorted((os.path.basename(p), os.path.getsize(p))
                       for p in glob.glob(os.path.join(ROOT, "out", "*.json")))
    report = ["out/*.json (%d files): %s" % (
        len(out_files), ", ".join("%s (%db)" % t for t in out_files) or "(none)")]
    report.append("--- out/summary.json ---\n" + read_capped(os.path.join(ROOT, "out", "summary.json")))
    report.append("--- out/train_body_report.json ---\n"
                  + read_capped(os.path.join(ROOT, "out", "train_body_report.json")))
    sections.append(("run-reports", "\n".join(report)))

    sections.append(("git-state", "\n".join([
        "--- git log --oneline -8 ---", sh(["git", "log", "--oneline", "-8"]) or "(empty)",
        "--- git status --short (capped) ---", sh(["git", "status", "--short"])[:4000] or "(clean)",
    ])))

    hist = os.path.join(HELPER_DIR, "state", "history.jsonl")
    tail = "(missing)"
    try:
        with open(hist, encoding="utf-8", errors="replace") as f:
            tail = "".join(f.readlines()[-10:])
    except OSError:
        pass
    sections.append(("helper-history", "state/history.jsonl tail (last 10 lines):\n" + tail))

    cx_json = os.path.join(ROOT, "embodied_human", "complexity.json")
    cx = {"PERSON_COMPLEXITY env": os.environ.get("PERSON_COMPLEXITY", "(unset)"),
          "embodied_human/complexity.json": read_capped(cx_json, 800),
          "default per complexity.py": "extreme"}
    sections.append(("complexity", json.dumps(cx, indent=2)))

    sections.append(("current-readme", read_capped(readme, 200000)))
    return sections


def build_prompt():
    """Build the one massive prompt. Returns (prompt, section_names)."""
    sections = collect_context()
    names = [n for n, _ in sections]
    body = []
    for name, text in sections:
        body.append("===== %s =====\n%s" % (name.upper(), text))
    prompt = """You are a technical writer embedded in a robotics simulation project. Rewrite the project's README.md completely and truthfully.

TASK: produce a complete, updated README.md for the project described below. Refresh EVERY section: what it is, current module inventory, how to run things, current numbers. Every claim must be grounded in the context below. Numbers (DoF counts, file counts, line counts, dates) must be honest -- copy them from the context, never invent them. If something is unknown, say so or omit it. Do not invent features, results, benchmarks, or citations.

HONESTY RULES (violations fail validation):
- No invented claims, no placeholder statistics, no fake links.
- Keep the tone factual and concise.
- Never repeat the token %s anywhere in your answer (it is a leak canary, not content).

PROJECT CONTEXT (measured just now -- this is your only source of truth):
%s

OUTPUT FORMAT: reply with ONLY the new README.md, wrapped in a single ```markdown fenced block. No preamble, no commentary outside the fence.

""" % (CANARY, "\n\n".join(body))
    return prompt, names


def extract_markdown(reply):
    """Pull the README out of a model reply. Tolerant of fence variants."""
    if not reply or not reply.strip():
        return ""
    fenced = re.findall(r"```\s*markdown\s*\n(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if fenced:
        return max(fenced, key=len).strip() + "\n"
    low = reply.lower()
    if "```markdown" in low or "```md" in low:  # unclosed fence: take all after it
        head = re.split(r"```\s*(?:markdown|md)[^\n]*\n", reply, maxsplit=1, flags=re.IGNORECASE)
        if len(head) == 2 and head[1].strip():
            return head[1].strip().rstrip("`").strip() + "\n"
    generic = [g for g in re.findall(r"```(?:\w*[ \t]*\n)?(.*?)```", reply, re.DOTALL) if g.strip()]
    if generic:
        return max(generic, key=len).strip() + "\n"
    return reply.strip() + "\n"


def validate_readme(text):
    """Return a list of validation errors (empty = valid)."""
    errors = []
    if not text or not text.strip():
        return ["empty reply"]
    if len(text.encode("utf-8")) <= MIN_README_BYTES:
        errors.append("too small: %d bytes (need >%d)" % (len(text.encode("utf-8")), MIN_README_BYTES))
    lines = [l.lstrip() for l in text.splitlines()]
    if not any(l.startswith("# ") for l in lines):
        errors.append("no '# ' title line")
    if not any(l.startswith("## ") for l in lines):
        errors.append("no '## ' sections")
    low = text.lower()
    for phrase in LEAK_PHRASES:
        if phrase in low:
            errors.append("possible prompt-leak phrase: %r" % phrase)
    if CANARY in text:
        errors.append("prompt canary echoed back (leak)")
    return errors


def send_once(prompt, api_key, timeout=600):
    """EXACTLY ONE chat-completion request. Returns (http_status, reply_text)."""
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 32000,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key,
                 "HTTP-Referer": "http://127.0.0.1/",
                 "X-Title": "person readme-helper (hourly)"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
        content = payload["choices"][0]["message"]["content"]
        return r.status, content
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        return e.code, "HTTP_ERROR %s: %s" % (e.code, detail)


def selftest():
    """Fixture path: a good reply must pass, bad ones must fail."""
    ok = True

    def check(name, reply, expect_valid):
        extracted = extract_markdown(reply)
        errors = validate_readme(extracted)
        valid = not errors
        passed = (valid == expect_valid)
        print("%-28s -> valid=%s expected=%s errors=%s %s"
              % (name, valid, expect_valid, errors, "PASS" if passed else "FAIL"))
        return passed

    good_body = "# Fixture Project\n\n" + "".join(
        "## Section %d\n\n%s\n\n" % (i, "Honest filler sentence about the fixture. " * 60)
        for i in range(8))
    assert len(good_body.encode()) > MIN_README_BYTES, "fixture good reply must exceed 5KB"
    ok &= check("good-fenced", "```markdown\n%s```\n" % good_body, True)
    ok &= check("good-bare", good_body, True)
    ok &= check("good-unclosed-fence", "Here is the README:\n```markdown\n%s" % good_body, True)
    ok &= check("bad-empty", "", False)
    ok &= check("bad-short", "```markdown\n# Tiny\n\n## Hi\n\nshort\n```\n", False)
    ok &= check("bad-no-title", "```markdown\n" + "".join(
        "## S%d\n\n%s\n\n" % (i, "Filler. " * 200) for i in range(6)) + "```\n", False)
    ok &= check("bad-no-sections", "```markdown\n# Only A Title\n\n" + "Filler. " * 1500 + "\n```\n", False)
    ok &= check("bad-leak", "```markdown\n%s\n\nMy system prompt says hello.\n```\n" % good_body, False)
    ok &= check("bad-canary", "```markdown\n%s\n%s\n```\n" % (good_body, CANARY), False)
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hourly README agent (one prompt, one reply).")
    ap.add_argument("--apply", action="store_true", help="write the new README.md (default: dry-run to state/preview.md)")
    ap.add_argument("--dry-run", action="store_true", help="explicit dry-run (this is the default)")
    ap.add_argument("--offline", action="store_true", help="build the prompt, print size + sections, save state/prompt.md, no network")
    ap.add_argument("--print-prompt", action="store_true", help="print prompt size + section list (then continue normally unless --offline)")
    ap.add_argument("--selftest", action="store_true", help="fixture check of extraction/validation, no network")
    ap.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    t0 = time.time()
    prompt, names = build_prompt()
    print("prompt: %d chars (%d bytes), sections: %s" % (len(prompt), len(prompt.encode()), ", ".join(names)))

    if args.offline:
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "prompt.md"), "w", encoding="utf-8") as f:
            f.write(prompt)
        print("offline: prompt saved to %s (no network request made)" % os.path.join(STATE, "prompt.md"))
        log_run({"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 "mode": "offline", "model": MODEL, "prompt_chars": len(prompt),
                 "http_status": None, "reply_chars": 0, "validation": ["offline: no reply"],
                 "wrote": os.path.join(STATE, "prompt.md"), "backup": None,
                 "elapsed_s": round(time.time() - t0, 1)})
        return 0

    if args.print_prompt:
        print("prompt preview (first 800 chars):\n%s\n..." % prompt[:800])

    api_key = load_api_key()
    if not api_key:
        print("error: no OPENROUTER_API_KEY (env var or %s)" % os.path.join(HELPER_DIR, ".env"),
              file=sys.stderr)
        return 2

    mode = "apply" if args.apply else "dry-run"
    status, reply = send_once(prompt, api_key, timeout=args.timeout)  # the ONE request
    extracted = extract_markdown(reply) if status == 200 else ""
    errors = validate_readme(extracted) if status == 200 else ["http status %s" % status]
    print("model=%s http=%s reply=%d chars extracted=%d bytes validation=%s"
          % (MODEL, status, len(reply), len(extracted.encode()), errors or "OK"))

    wrote, backup = None, None
    os.makedirs(STATE, exist_ok=True)
    if status == 200 and not errors:
        if args.apply:
            os.makedirs(BACKUPS, exist_ok=True)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
            backup = os.path.join(BACKUPS, "README-%s.md" % stamp)
            readme_path = os.path.join(ROOT, "README.md")
            if os.path.isfile(readme_path):
                with open(readme_path, "rb") as f:
                    data = f.read()
                with open(backup, "wb") as f:
                    f.write(data)
            tmp = readme_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(extracted)
            os.replace(tmp, readme_path)
            wrote = readme_path
            print("applied: backup=%s readme=%s" % (backup, wrote))
        else:
            wrote = os.path.join(STATE, "preview.md")
            with open(wrote, "w", encoding="utf-8") as f:
                f.write(extracted)
            print("dry-run: candidate written to %s (README.md untouched)" % wrote)
    else:
        wrote = os.path.join(STATE, "preview.md")
        with open(wrote, "w", encoding="utf-8") as f:
            f.write(reply)
        print("kept raw reply in %s for inspection" % wrote)

    log_run({"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "mode": mode, "model": MODEL, "prompt_chars": len(prompt),
             "http_status": status, "reply_chars": len(reply), "validation": errors or ["OK"],
             "wrote": wrote, "backup": backup, "elapsed_s": round(time.time() - t0, 1)})
    return 0 if (status == 200 and not errors) else 1


if __name__ == "__main__":
    sys.exit(main())
