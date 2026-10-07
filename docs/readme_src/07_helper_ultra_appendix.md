## 18. The OpenRouter planner and Pi coder loop (`tools/openrouter_helper/`)

A bash helper that keeps proposing the next step on this project, has a coding agent implement it, verifies
the result and undoes it if it broke the person. It comprises about 2,200 lines (`helper.sh` 266, `hlib.py` 966,
`proxy.py` 487, `ui.py` 413, `smoke.py` 64) and four offline test suites. Its own `README.md` documents every
command; in outline:

```bash
bash tools/openrouter_helper/helper.sh run                       # plan → Pi codes it → verify → rollback on FAIL → repeat
bash tools/openrouter_helper/helper.sh run-goal "<goal>" 12     # the same, steered by a standing goal, at most 12 cycles
bash tools/openrouter_helper/helper.sh ui                        # localhost web console (port 8770): goal box and live feedback
bash tools/openrouter_helper/helper.sh stop
```

**Components.**

| component | what it does | quantities |
|---|---|---|
| planner (`hlib.py`) | reads the newest `.npz` and JSON run files, the complexity table, an `ast`-generated API outline, git state, earlier outcomes and the machine's free resources; asks a free model for one next step | context of about 32 kB per plan |
| gateway (`proxy.py`, 127.0.0.1 only) | the only path to OpenRouter: model order, quotas, retries, streaming, tool calls | 5 models in order; **200 requests per model per UTC day**; **20 per minute** (≈3.25 s gap); **1,000 per day** (5 × 200); counts persist across restarts |
| coder | [Pi](https://pi.dev) in an isolated config directory, model = the gateway only, no API key in its environment, a tool-call budget (default 400) and a time limit (default 3,600 s) | permission guard `guard.ts` logs every tool call to `state/pi_audit.jsonl` |
| verify | changed files compile; the package imports at `base` and `rich`; a 20 s smoke run has no exception, no NaN, no fall and is not >25 % slower; helper and key untouched; on FAIL the files are rolled back | |
| console (`ui.py`) | goal box, loop state, quota usage, last PASS/FAIL with reasons, step history, Pi's latest report (refresh 3 s) | stdlib only |

**Quota arithmetic.** 5 models × 200 = 1,000 requests a day; at 20 per minute the whole day's allowance could be
used in 50 minutes. If every request used the maximum output length the daily ceiling would be
2 × 200 × 200,000 + 200 × 30,000 + 200 × 50,000 + 200 × 30,000 = **102 million output tokens**, but models do not
reliably honour long output targets. Every Pi turn is one request, so a coder step with 30 tool calls uses about
30 of one model's 200, giving on the order of 10–30 coder steps a day.

**Live probe (2026-10-07).** `nemotron-3-super-120b-a12b:free` and `laguna-s-2.1:free` work;
`nemotron-3-ultra-550b-a55b:free` returned 503 (overloaded) intermittently; `inkling-small:free` returns 403
unless the caller is a coding agent (Pi identifies itself honestly and is accepted); `ling-3.1-flash:free` is no
longer free (404) and is skipped. Effectively three models share the planner's day (600 requests).

**Tests** (all offline, on throwaway copies): `test_gateway.py` (9 scenarios: order, quotas, retry counting,
skipped models, `max_tokens` ladder, restart persistence, day rollover, per-minute spacing, harness gating),
`test_safety_net.py`, `test_run_loop.py` (the whole loop with a fake coder, killed gateways and a restarted
mock) and `test_pi_coder.py` (real Pi against the real gateway with a mock model; the guard blocked `git push`,
reading the key file, writing outside the project and data-sending web requests, and allowed ordinary
commands).

### 18.1 What the first two live steps did (2026-10-07), and why they were reverted

Two coder steps ran on olfaction and taste (constant zero in the run files). Both **passed verify**
(smoke run 1.87 ms/step and 1.74 ms/step, no fall, no NaN, imports at `base` and `rich`), used `inkling-small`
through Pi (about 67 and 59 gateway requests, about 5 and 6 minutes) and tripped no guard rule. Reading the
diffs showed that both degraded the project:

* Step 1 added constants (0.02 above a height of 0.1 m, 0.01 near a fixed point) and overwrote the receptor
  output with them; nothing depended on hunger, blood chemistry or breathing, no complexity flag and no
  validation script were added.
* Step 2 added `stimuli.py` (two fixed sources at (±1, 0, 0.5) with 1/distance falloff) behind a flag enabled only
  at `rich`, and **rewrote `ChemoSystem.sense`, deleting** the original object-based smell (distance falloff,
  adaptation, novelty) and tongue-contact taste. At the default `extreme` level smell and taste became exactly
  zero.

Verify cannot see this kind of damage because the smoke run never touches smell or taste. **Both steps were
reverted** at the owner's request (the eight edited files were restored to their state before step 1 and the
new file was deleted). The lesson is recorded here because it bounds what the loop can be trusted with: it
guarantees that the simulation still runs, not that a step did its job. The helper's own history records the
step title as "?" because of a Windows path problem in `helper.sh`; this is a known bug.

### 18.2 The README helper

`tools/openrouter_helper/readme_helper/` is a separate agent that sends **one** prompt describing the whole
project and the current README to `nemotron-3-super-120b-a12b:free` and installs the reply as `README.md` (with
`--apply`; a backup is kept under `state/backups/`). In `loop --apply` mode it does this every hour. Its two
applied runs on 2026-10-07 reduced a 1,028-line README to 66–69 lines. The README you are reading is therefore
built **deterministically** by `tools/build_readme.py` from `docs/readme_src/`, and a copy is kept at
`docs/README_full.md`; if the hourly agent is running it will overwrite `README.md` again (see Appendix B for
how to rebuild it).

---

## 19. The ultra tier (framework built, subsystems not built)

Two levels, `ultra` and `mega`, sit above `max`. They keep the classic body at `max` size (31,872 taxels,
1,384,030 sensory scalars, 88,892 internal variables) and add **plug-in subsystems**
(`embodied_human/ux_*.py`) that run beside it through a shared bus.

### 19.1 What exists

| item | file | size / status |
|---|---|---|
| framework | `embodied_human/ultra.py` | 593 lines, 6 classes (`Level`, `Bus`, `Subsystem`, `UltraFrame`, `UltraWorld`, `SyntheticBus`), 34 functions |
| bus | `ultra.BUS_DOC` | **59 documented standard inputs** (tactile array, 52 × 13 proprioception, 67 interoceptive variables, 28 emotions, 25 neuromodulators, 21 drives, inner-world scalars, body state, world state, scene objects) |
| output channels | `affect.<field>`, `drive.<name>`, `intero.<name>`, `desire.<tag>` | standing levels, summed over subsystems |
| presets | `complexity.PRESETS["ultra"]`, `["mega"]` | same classic sizes as `max` |
| reference subsystem | `embodied_human/ux_example.py` | 50,000 (`ultra`) / 1,000,000 (`mega`) leaky units; excluded from real persons |
| benchmark | `tools/ultra_bench.py` | each subsystem in its own process; state variables, MB, peak MB, ms per simulated second against declared budgets |
| memory lock | `tools/simlock.py` | slots and free-RAM check before heavy runs |
| conventions | `docs/ultra/CONVENTIONS.md` | the contract every builder follows |

Benchmark of the reference subsystem (2026-10-07): `ultra` 50,000 state variables, 1.4 MB, 6.8 ms per
simulated second; `mega` 1,000,000 state variables, 28 MB, 193.8 ms per simulated second; both within their
declared budgets (20 ms and 400 ms).

A subsystem declares its sizes per level (`level.pick(extreme=…, ultra=…, mega=…)`), update rate, summary length,
the bus keys it reads and writes, and budgets for wall time and memory; the optional wall-clock **governor**
stretches update periods (by up to 8×) when the machine is busy. Rules: numpy only, deterministic given its
seed, finite, no per-step large allocations.

### 19.2 Targets and the plan that did not run

| | `extreme` (measured) | `ultra` target | `mega` target |
|---|---|---|---|
| internal dynamic state | 36,932 | ≥ 3,000,000 (81×) | ≥ 40,000,000 (1,083×) |
| new sensory/event scalars per frame | 778,054 (existing) | ≥ 5,000,000 | ≥ 50,000,000 |
| extra wall time per simulated second | – | ≤ 6 s | ≤ 40 s |
| extra memory | – | ≤ 2.5 GB | ≤ 8 GB |
| parallel instances | – | 3–4 | 1 |

A design workflow for 20 domains (somatic afferents, spinal motor, vision pathway, audition, chemosense and
plumes, multisensory body schema, cardio-respiratory, endocrine-metabolic, immune repertoire, thermo-tissue,
visceral/renal, spiking cortex, subcortical loops, learning and memory, affect and personality, behaviour
grammar, environment and social, musculoskeletal tissue, AZR navigation, infrastructure), and a second team for
10 further domains (hierarchical predictive coding, planning and a world model, language and speech, lifespan
dynamics, haptic object perception, oculomotor/vestibular, spatial cognition, visible expression, attention and
workspace, valuation and hedonics) plus an audit of the classic modules, were launched. **Every agent errored out
and returned nothing; no design file or subsystem was produced.** The targets above are therefore design
intentions, not results, and the machine limits (i7-14700, 32 GB of which 5–10 GB are usually free, a 6 GB GPU
fully used by LM Studio, numpy only) decide how much of them is attainable.

---

## Appendix A. Quantities and combinatorics

### A.1 Summary of countable quantities

| quantity | count | derivation |
|---|---|---|
| Actuated DoF | 52 | one torque motor per joint |
| Joint pairs | 1,326 | C(52,2) |
| Skin patches / patch pairs | 46 / 1,035 | C(46,2) |
| Taxels (`base` … `max`) | 1,992 / 7,968 / 17,928 / 31,872 | 1,992 × (1, 4, 9, 16) |
| Taxel pairs (`base`) | 1,983,036 | C(1992,2) |
| Tactile channels per taxel | 26 (`base`), 43 (above) | 26 + 17 extended |
| Proprioceptive scalars | 676 | 52 × 13 |
| Collision categories / pairs / enabled | 6 / 21 / 15 | C(6,2) + 6 |
| Interoceptive variables / pairs / triples | 67 / 2,211 / 47,905 | C(67,2), C(67,3) |
| Appraisal weights | 280 | 28 emotions × 10 dimensions |
| Emotions / pairs / triples / on-off patterns | 28 / 378 / 3,276 / 268,435,456 | C(28,2), C(28,3), 2²⁸ |
| Neuromodulators / pairs | 25 / 300 | C(25,2) |
| Drives / pairs / 3-level states / on-off | 21 / 210 / 10,460,353,203 / 2,097,152 | 3²¹, 2²¹ |
| Latent dimension | 572 | 52 + 52 + 177 + 18 + 17 + 6 + 29 + 67 + 10 + 16 + 128 |
| Forward model weights | 356,928 | 572 × 624 |
| Inverse model weights | 59,488 | 52 × 1,144 |
| Predictive system array values | 2,866,708 | measured |
| Reafference weights (`base` / `extreme`) | 414,336 / 3,729,024 | taxels × 4 × 52 |
| Library policies / sequences of 3 / of 5 | 55 / 166,375 / 503,284,375 | 55^h |
| Allowed mind calls / sequences of 3 / of 6 | 26 / 17,576 / 308,915,776 | 26^k |
| Behaviour channels / descriptors | 33 / 1.1575 × 10³⁷ | product of channel sizes |
| Safety-model weights | 308 | 300 one-hot + 8 continuous |
| Neural-mass synapses (`extreme`) | 589,824 | 12,288 × 48 |
| Inner dynamic variables (`rich` / `extreme` / `max`) | 15,066 / 36,932 / 88,892 | measured |
| Gateway requests per day | 1,000 | 5 × 200 |

### A.2 Skin patches

Taxels per patch at each level (each patch scales by the square of the skin density; all 46 follow the
×9 rule at `extreme`).

{{PATCHES}}

### A.3 The 55 library policies

{{POLICIES}}

---

## Appendix B. Reproduction and data files

```bash
python tools/build_readme.py                     # rebuild this README from docs/readme_src/ (deterministic, offline)
PYTHONPATH=. PERSON_COMPLEXITY=base python tools/readme_combinatorics.py   # all combinatorial quantities → out/readme_data/combinatorics.json
PYTHONPATH=. PERSON_COMPLEXITY=extreme python tools/readme_runtime_facts.py out/readme_data/runtime_extreme.json   # array inventories
PYTHONPATH=. PERSON_COMPLEXITY=extreme python tools/readme_describe.py out/readme_data/describe_extreme.json      # agent.describe()
python tools/scale_complexity.py --measure base rich extreme max      # sizes and speed per level
PERSON_COMPLEXITY=base python run_sim.py --duration 20 --seed 7 --out out_readme --no-figures    # reference run (§11)
python tools/ultra_bench.py                      # ultra-tier subsystem benchmarks
python tools/readme_check.py <old.md> <new.md>   # numbers, code, links and caveats preserved across a rewrite
```

Data files written by these commands: `out/measure_baseline.txt`, `out/readme_data/measure_max.txt`,
`out/readme_data/{combinatorics,runtime_base,runtime_extreme,describe_base,describe_extreme}.json`,
`out/readme_data/refrun_base.txt`, `out_readme/summary.json`, `out/ultra_bench.json`.

The sections of this README are the files of `docs/readme_src/`; edit those and rebuild, rather than editing
`README.md` directly. The earlier academic rewrite of the previous text is in `docs/readme_work/new/`.
