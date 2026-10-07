# Hard constraints (the helper checks the first group itself; you must respect all of them)

## The machine
- CPU: Intel i7-14700 (20 cores / 28 threads). GPU: 6 GB VRAM (the simulation does **not** use it).
- RAM: 31.7 GB installed, but **only about 3-5 GB is free and about 13 GB of commit** because Discord, Chrome,
  LM Studio (llama-server) and others are running. Treat **~6-8 GB total** as the budget for the simulation.
- The user wants **at least a few (4+) simulated people running in parallel** (parameter search, body
  learning). So one instance at the default complexity must stay at **~1-1.5 GB resident** and
  **a few milliseconds per physics step**. Anything that raises per-instance memory by more than ~10 %
  or time per step by more than ~15 % **must be behind a complexity knob** in `embodied_human/complexity.py`
  and **must be off at `base`**.
- Preallocate arrays; no per-step allocations of large arrays; vectorise with numpy; update slow systems
  at low rates (see `InnerWorld.RATES`).

## What you may and may not do
- Python 3 + numpy + mujoco + Pillow (+ stdlib) only. **No new dependencies.** (scipy is not installed.)
- You may **edit existing** `.py`/`.md` files and **create new** `.py`/`.md`/`.txt` files **only under**
  `embodied_human/`, `tools/` (not `tools/openrouter_helper/`), and the project root for `run_*.py`,
  `diag_*.py`, `README.md`. You cannot delete files. You cannot touch `.env`, `.git`, `orchestrator/`,
  `AI_LOG.txt`, or the helper itself.
- You work with a shell in the project directory, so you can and should run code and tests yourself. The helper
  will additionally compile what you changed, import the package, run a smoke simulation and **revert everything
  if a check fails**. Keep every change importable.
- Do not break public interfaces other modules use (`LatentSpec.build` argument order, `BodyState`
  fields, `ReceptorFrame` fields, `Behavior` fields, `PERSON_COMPLEXITY` levels). Add, do not rename.
- Do not add anything that asks for credentials, makes network calls, or runs subprocesses.

## What the checks do (so you can pass them)
1. every touched `.py` file must `py_compile`;
2. `import embodied_human.agent` must work at `PERSON_COMPLEXITY=base` and at `rich`;
3. a 20-second smoke run (`tools/openrouter_helper/smoke.py`) must complete with no exception, no NaN in the
   latent, the person not fallen, and no more than 25 % slower than the recorded baseline;
4. if you changed `tools/` scripts they are only compiled.
