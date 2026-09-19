# Does Jev's `Choice` always return the highest-probability option?

A reproducible experiment against TypeSafe's Jev. The short answer is no: in
**16.3%** of 1,000 single-question requests the `choice` field named an option
whose probability in the *same response* was lower than the maximum.

Full write-up with statistics, charts and a minimal reproduction: **[report.md](report.md)**.

A follow-up asks whether feeding Jev reference material fixes it. It does, completely:
accuracy goes from 10.5% to 100% and the mismatch rate from 16.0% to 0%, changing
nothing but `state`. See **[context_probe.md](context_probe.md)**.

A third experiment asks whether small Jev calls can be *chained* into a computation
one call cannot do. They can: `N mod 7` and weekday offsets are both 100% inside the
range a date needs, and a retrieval-fed three-stage chain takes the same 200 dates
from 9.5% to **100%**. The weak link is fact recall, not composition. See
**[chain.md](chain.md)**.

Those anchors were close enough to count from, so a second probe makes the reference
material remote. Scattered true dates do not help at all, and a single reference 365
days back scores 81% while the same reference moved one day, to 364 days back, scores
0%. See **[context_probe_hard.md](context_probe_hard.md)**.

## Run it

```bash
export TYPESAFE_API_KEY=...          # read from the environment, never logged or stored
pip install -r requirements.txt
python -m pytest tests -q            # 15 tests, no network

python -m choice_audit.collect       # ~1,263 requests, resumable, appends to data/raw_responses.jsonl
python -m choice_audit.analyze       # reads the JSONL only; writes the CSV, results.json, charts and report.md
python -m choice_audit.probe                 # context ladder: 800 requests -> context_probe.md
python -m choice_audit.probe --suite hard    # remote references: 1,400 -> context_probe_hard.md
python -m choice_audit.chain                 # sub-skills and the chain: ~1,460 -> chain.md
```

`collect` is resumable: a `record_id` already recorded as successful is skipped, so
re-running after an interruption costs nothing. `--dry-run` prints the plan without
sending, `--limit N` sends only the next N, and `--experiment` restricts to one of
`main`, `determinism` or `batch`.

`analyze` makes **no network calls**. Every number in the report can be recomputed
from the committed JSONL by anyone, with no API key.

## What is tested

| Experiment | Requests | Shape |
|---|---|---|
| `main` | 1,000 | one date per request |
| `determinism` | 250 | 50 of those dates, 5 byte-identical repeats each |
| `batch` | 13 | the same 50 dates, up to 4 questions per request |
| `probe --suite ladder` | 800 | 200 dates, four context conditions |
| `probe --suite hard` | 1,400 | the same 200 dates, seven remote-reference conditions |

The task is "What day of the week is {date} in the Gregorian calendar?" with
Monday to Sunday plus `Unknown` as the options, a constant neutral greeting as the
state, and dates drawn with seed 20260919 from 2027 to 2099. Ground truth comes
from Python's `datetime` and never enters a request.

The model cannot do this task; both `choice` and the argmax score at chance. That
is the point: it produces the near-uniform distributions in which the disagreement
between `choice` and `probabilities` shows up.

## Layout

```
choice_audit/
  config.py    every knob of the experiment
  plan.py      the deterministic request list
  client.py    HTTP, retries with exponential backoff, rate limit, counters
  store.py     append-only JSONL, resumability
  collect.py   CLI: send requests, record raw exchanges
  extract.py   raw records -> one row per question
  stats.py     Wilson intervals, binomial and Fisher tests
  analyze.py   CLI: statistics, CSV, results.json
  charts.py    the seven figures
  report.py    report.md
  probe.py     both context suites and their write-ups
  chain.py     sub-skill tests and the end-to-end chain
data/          raw_responses.jsonl, per_question_results.csv, results.json
charts/        PNGs at 1920x1080, 150 dpi
tests/         offline
```

## Notes on method

- The request schema was taken from the published OpenAPI document at
  `https://api.typesafe.ai/openapi.json`, not guessed. That document is also the
  source of the contract under test: `choice` is documented as "the name of the
  choice with the highest probability among the question's criteria".
- Probabilities are compared with a tolerance of 1e-9. A mismatch is
  `probabilities[choice] < max(probabilities)` beyond that tolerance.
- Where the argmax is tied, the tie is broken at random with a fixed seed, not by
  the order the options were supplied, which would otherwise flatter whichever
  option happens to be listed first.
- `jev-latest` is requested; the version actually served is recorded on every
  response and reported.
