# OpenRouter planning helper

One bash command that keeps deciding **what the next step on the simulation should be**, hands it to
whatever coder you point at it, checks the result, undoes it if it broke something, and starts over.

```bash
bash tools/openrouter_helper/helper.sh run          # runs until you stop it
bash tools/openrouter_helper/helper.sh stop         # from another terminal
```

## What one step does

1. **Looks at the project.** Reads the newest `out/*.npz` files (every array: shape, range, NaN/inf,
   dead channels) and `out/*.json` reports, the module list and a generated outline of the *real* classes
   and function signatures, the complexity table, git status, what earlier steps achieved or broke, and
   **how many simulations the machine can run in parallel right now** (free CPU, free RAM, free commit
   charge, measured memory per simulated person).
2. **Asks the planner model** for the single most valuable next step, written as a complete prompt for a
   coder, sized for the model's output limit (200k / 50k / 30k tokens of work per request, as asked in
   the prompt; models cannot be forced to comply). The planner may also name public web pages, which are
   fetched read-only (public hosts only, text only, size-limited) and attached as reference material.
3. **Snapshots** the editable files and writes `state/next_task.md`.
4. **Waits for your coder.** The coder works in the project directory with its own tools and ends with
   `bash tools/openrouter_helper/helper.sh done`.
5. **Verifies**: every changed `.py` compiles; the package imports at `base` *and* `rich`; a smoke
   simulation runs (no exception, no NaN, not fallen, not >25 % slower than baseline); the helper and the
   key file were not touched. **On failure the files are rolled back** to the snapshot. The outcome is
   recorded and the next step is planned with that history in view.

If all models are at today's quota it sleeps until 00:00 UTC and carries on, so it runs indefinitely.

## The limits (enforced in `proxy.py`, whoever is asking)

| limit | value |
|---|---|
| models | used **strictly in the order of `models.json`** |
| per model | **200 requests per UTC day** |
| per minute | **20** (the gateway keeps a gap of ~3.25 s, so never more than 18-19) |
| per day | **1000** |

Counts are saved in `state/usage.json` (survive restarts, reset by themselves at 00:00 UTC). Every
upstream attempt counts, including retries. 429s honour `Retry-After`; a model that keeps failing, or
that does not exist, is skipped for the day; a `max_tokens` the provider rejects steps down the model's
ladder and the working value is remembered.

### What the live probe found (2026-10-07)

| model | result |
|---|---|
| `nvidia/nemotron-3-super-120b-a12b:free` | works |
| `poolside/laguna-s-2.1:free` | works |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 503 "temporarily overloaded" (retried; may work later) |
| `thinkingmachines/inkling-small:free` | **403: only available on agentic harnesses.** OpenRouter will not serve it to a plain API caller. The gateway marks it *gated*: ordinary callers (the planner) skip it; a request that carries a real harness's own `HTTP-Referer` / `X-Title` headers is passed through unchanged and may use it. The gateway never claims to be a harness it is not. |
| `inclusionai/ling-3.1-flash:free` | **404: no longer free**; OpenRouter points at the paid slug `inclusionai/ling-3.1-flash`. Skipped; the helper never switches to a paid model on its own. |

So for the planner, effectively three models share the day (600 requests), and `inkling-small` is only
reachable through a listed coding agent.

## The Pi coder (automatic, with a permission guard)

`helper.sh run` now starts [Pi](https://pi.dev) itself for the coder step (`AUTO_CODER=0` goes back to
waiting for a coder you start yourself; `helper.sh coder` runs just that step once).

| piece | how |
|---|---|
| model | only the local gateway (`pi_agent/models.json`), so order / 200 per model / 20 per min / 1000 per day still hold; Pi identifies itself honestly (`HTTP-Referer: https://pi.dev`, `X-Title: pi`), which is why `inkling-small` works for it |
| key | the real OpenRouter key is **not** in Pi's environment; only the gateway process has it |
| config | an isolated agent directory (`state/pi_home`), so your own `~/.pi` and its logins are untouched |
| tools | `read bash edit write grep find ls`; no MCP, skills, themes, context files or project-local extensions |
| limits | `CODER_TIMEOUT` (default 3600 s), `PI_MAX_TOOL_CALLS` (default 400) |
| guard | `pi_agent/extensions/guard.ts` blocks, and logs to `state/pi_audit.jsonl`: `git push/remote/credentials`, `gh`, recursive deletes, system/security settings, package installs, web requests that *send* data (plain GET is fine), `ssh/nc/scp…`, `sudo`, reading `.env` / `auth.json` / `.ssh`, writing outside the project or into `tools/openrouter_helper/` / `.git`, and the loop's own `run/stop/rollback` commands |
| net | after the coder, `verify` always runs; a failing step is rolled back |

**What the guard is not:** a sandbox. It matches commands by pattern, so a determined or confused model can
find a way around it (Pi's own docs say it has no built-in sandbox). What actually contains damage is: the
key is not in the process, every step is snapshotted and verified, and nothing is pushed. For real isolation
run the whole thing in a container or VM.

**Quota note:** every model turn of the coder is one gateway request. A step with 30 tool calls uses ~30 of
the 200 per model, so expect on the order of 10-30 coder steps per day across the models.

## What it deliberately does not do

* It never changes `tools/openrouter_helper/` or `.env`; `verify` fails if a coder did.
* It does not use paid models.
* It never pushes to GitHub.

## Files

| file | role |
|---|---|
| `helper.sh` | the bash driver: `run plan done verify rollback snapshot resources fetch probe serve status stop` |
| `proxy.py` | local gateway: model order, quotas, rate limit, retries, persistence (127.0.0.1 only) |
| `hlib.py` | context builder, `.npz` reader, resource allocation, planner request/parse, snapshot/rollback/verify |
| `smoke.py` | the 20 s verification simulation |
| `models.json` | the five models in order, `max_tokens` ladders, output targets |
| `prompts/` | thesis, hard constraints, planner and coder instructions |
| `.env` | `OPENROUTER_API_KEY=...` (git-ignored; permissions restricted to your user) |
| `state/` | usage, request log, snapshots, plans, history (git-ignored) |
| `tests/` | `test_gateway.py`, `test_safety_net.py`, `test_run_loop.py`, `mock_upstream.py` (all offline) |

## Resource allocation

`helper.sh resources` prints how many parallel simulations fit *now*:
`min(free CPU threads - 2, (free commit - 3 GB) / GB per instance, free RAM x 1.8 / GB per instance, 12)`.
It is re-computed each step and shown to the planner, so plans are sized to what the machine can spare at
that moment (this PC usually has only 3-5 GB of RAM free because of other applications).

## Safety notes

* The API key sits in `.env` only; `.env` and `state/` are in `.gitignore`. The key was pasted into a chat
  to set this up: **rotate it when you are done.**
* A step that fails verification is undone, but a coder that has a shell can do things no file snapshot
  can undo (delete outside the project, push to GitHub, read other secrets). Pages fetched from the web
  can contain text written to manipulate an automated reader. The helper marks them as data, not
  instructions; keep that in mind when you choose how much freedom to give your coder.

## Tests

```bash
python tools/openrouter_helper/tests/test_gateway.py       # order, quotas, rate limit, 429, 404, gating, restart, day rollover
python tools/openrouter_helper/tests/test_safety_net.py    # snapshot / verify / rollback / protection on a sandbox copy
python tools/openrouter_helper/tests/test_run_loop.py 9    # the whole `run` loop, with a fake coder, kills and restarts
python tools/openrouter_helper/tests/test_pi_coder.py      # real Pi -> gateway -> mock model; guard allows/blocks
```
