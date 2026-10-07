# The interviewer: one simulation, always on, one question per 30 minutes

An always-on loop around a single simulated person. The person's mind is the
local AZR coder 3B (`http://127.0.0.1:1234/v1`), so the simulation itself
costs no OpenRouter quota. At most **one question every 30 minutes** is posed
to the person; each next question is generated from the person's own
transcript tail (thoughts, actions, speech, silence) plus a body snapshot.

```bash
python tools/interview/interview.py            # run forever (Ctrl+C to stop)
python tools/interview/interview.py --once    # one question cycle now, then exit
python tools/interview/interview.py --dry-run # print the generator prompt; no sim, no network
python tools/interview/interview.py --selftest
```

## The rules the question generator must obey

Baked into `GENERATOR_SYSTEM` in `interview.py`:

* academic and experimental only — about the person's own perception, body
  state, decisions, or reasoning, referencing something actually observed;
* never leading — no emotional baiting, no suggested feelings, no
  praise-bait, no roleplay, no re-asking; silence is valid data;
* exactly one question, 10–300 chars, one `?`. Anything else is discarded
  (and does NOT count as the half-hour question).

## Model rotation (the 200/day limit)

Question generation goes through the OpenRouter gateway
(`tools/openrouter_helper/proxy.py`), which walks models in order and moves
on after 200 requests per model per day. The interviewer asks at most ~48
questions a day, so it barely sips quota; when the shared budget runs dry
the gateway reports `quota_exhausted` and the interviewer waits (sim keeps
running). The model used for each question is recorded in the transcript
and in `state/state.json`.

The gateway starts itself if it is down (key from `OPENROUTER_API_KEY` or
`tools/openrouter_helper/.env`, never logged). Without a key or quota the
simulation still runs — only questioning pauses.

## Where everything is recorded

Every instance's transcript: `<out>/instances/<instance-id>.txt`
(the interview sim uses `out_interview/`). One line per event:

* `INTERVIEW_QUESTION "..." (via model) | rationale: ...`
* `HEARD`, `THOUGHT`, `ACTION`, `SAID`, `EVENT`

`state/state.json` holds `last_asked_wall` (the 30-minute floor survives
restarts), counts, and the last model/question.

## Resource note

One headless sim paced to real time plus the already-running LM Studio
server. Physics runs on CPU; the 3B model stays on the 6 GB GPU as today.
Nothing here trains or fine-tunes weights.
