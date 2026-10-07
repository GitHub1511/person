# Hourly README agent

Keeps the project's `README.md` truthful by regenerating it from the measured
state of the tree, once an hour.

```bash
bash tools/openrouter_helper/readme_helper/run.sh preview       # offline: prompt size + sections, no network
bash tools/openrouter_helper/readme_helper/run.sh once           # dry-run: 1 request, writes state/preview.md
bash tools/openrouter_helper/readme_helper/run.sh once --apply  # 1 request, backs up + rewrites README.md
bash tools/openrouter_helper/readme_helper/run.sh loop --apply  # every hour on the hour, forever
```

## What it does

Each run, `agent.py` reads the **entire** project and only then asks the model:

* every `embodied_human/*.py` (line count + first docstring line)
* `tools/*.py` listing, root `run_*.py` / `diag_*.py` / `README.md` presence
* `out/summary.json` + `out/train_body_report.json` (when present)
* `git log --oneline -8`, `git status --short`
* `tools/openrouter_helper/state/history.jsonl` tail
* complexity level (`PERSON_COMPLEXITY` env / `complexity.json` / default)
* the full current `README.md`, so every section can be refreshed against it

## One prompt, one output

There is deliberately no loop, no retry, no multi-step plan: the script builds
**one** massive prompt asking for a complete, truthful, updated README.md
(numbers honest, no invented claims) and sends **exactly one** chat-completion
request. The reply is expected as a single ` ```markdown ` fence (extraction is
tolerant of bare or unclosed fences), validated, and either filed as
`state/preview.md` (dry-run) or installed as `README.md` (`--apply` only).
Every run is logged to `state/runs.jsonl`.

## Why Nemotron 3 Super is pinned

The model is hard-coded to `nvidia/nemotron-3-super-120b-a12b:free`, called
**directly** at `https://openrouter.ai/api/v1/chat/completions` (no local
gateway, no model rotation). The main helper's live probe (2026-10-07) found it
working with a large output ladder, and a long-form README rewrite wants a big
output window. One request per run is 24 requests/day in hourly `loop` mode --
far under that model's share of the quotas, so it never starves the planner.

## Safety

* **Dry-run by default.** `once`/`loop` without `--apply` never touch
  `README.md`; the candidate goes to `state/preview.md`.
* **Backup.** `--apply` copies the current `README.md` to
  `state/backups/README-<UTC-stamp>.md` before replacing it (atomic replace).
* **Validation.** The reply is rejected (nothing installed) unless it is
  non-empty, >5KB, has a `# ` title and `## ` sections, and shows no
  prompt-leak (known leak phrases plus a canary token the prompt forbids
  repeating).
* **Key hygiene.** The API key comes only from `OPENROUTER_API_KEY` or
  `tools/openrouter_helper/.env` and is never printed or logged.
* **Blast radius.** This helper only ever writes `README.md` (on `--apply`)
  and its own `state/` files. It never touches code, the main helper, or git.

## Scheduling

Windows Task Scheduler, hourly (runs while logged on; Git-Bash must be
installed -- adjust the path to your `bash.exe`):

```cmd
schtasks /create /tn "PersonReadmeHourly" /sc hourly /mo 1 ^
  /tr "\"C:\Program Files\Git\bin\bash.exe\" -lc \"cd /c/Users/Shivi/Downloads/person && bash tools/openrouter_helper/readme_helper/run.sh once --apply\""
```

Or keep a terminal open in `loop` mode instead (same effect, aligns itself to
each `:00`):

```bash
bash tools/openrouter_helper/readme_helper/run.sh loop --apply
```

## Wiring into the main helper (left for the owner)

This directory is intentionally **not** wired into `helper.sh` -- no existing
file was modified to add it. If you want a `readme` passthrough, add this one
line to the `case` block in `tools/openrouter_helper/helper.sh` (next to the
other subcommands):

```bash
readme)   shift; bash "$HERE/readme_helper/run.sh" "$@" ;;
```

That gives `bash tools/openrouter_helper/helper.sh readme once --apply` and
friends, with nothing else changed.

## Files

| file | role |
|---|---|
| `agent.py` | the agent: context builder, one-shot request, extract/validate, backup/apply, logging (stdlib only) |
| `run.sh` | driver: `once [--apply]`, `loop [--apply]` (hour-aligned), `preview` (offline) |
| `state/` | `prompt.md`, `preview.md`, `backups/`, `runs.jsonl` (git-ignored) |
| `.gitignore` | keeps `state/` and `__pycache__/` out of git |
