# Jev vs classical ML on public ED triage data

Does a System One model (TypeSafe's **Jev**) assign five-level emergency triage
acuity as well as supervised ML trained on the same dataset, and how well
calibrated are its probabilities? This repo runs that comparison end to end on
fully public triage datasets.

```
load dataset -> bucket vitals in code -> one Jev call per patient
  (6 Nouls for the ESI decision points + 1 Score over the five levels)
-> combine in code -> confusion matrices + calibration plots vs the true acuity
-> same metrics for logistic regression, gradient boosting, random forest (out-of-fold)
```

## Quick start

```bash
cd jev-triage-eval
pip install -r requirements.txt
python -m pytest -q                                   # 12 tests, no network

# Keyless dry run: exercises everything with a heuristic stand-in for Jev
python -m triage_eval.run --dataset ktas --jev-backend mock --limit 200

# Real run (1,267 KTAS patients, ~1,267 requests)
export TYPESAFE_API_KEY=...                           # or put it in .env / your sandbox secrets
python scripts/check_key.py                           # one patient, prints the raw answers
python -m triage_eval.run --dataset ktas --jev-backend typesafe --out results/ktas
```

Outputs land in `results/<name>/`:

| File | Content |
| --- | --- |
| `metrics.md`, `metrics.json` | summary table and full per-model metrics, run metadata (model id, latency, tokens, policy) |
| `predictions.csv` | one row per patient: truth, nurse level, every model's level and 5-way probabilities, the six raw Nouls |
| `confusion_<model>.png` | row-normalised confusion matrix per model |
| `calibration.png` | reliability diagram of top-level confidence for every model, plus the confidence histogram |
| `noul_calibration.png` | each Noul against the code-defined proxy of what it should predict |
| `jev_cache.jsonl` | every raw Jev answer, keyed by prompt version, so re-running the policy costs nothing |

Rows tagged `[mock]` come from the stand-in and are never Jev results.

## Datasets

**KTAS** (default, bundled): 1,267 adult visits from two Korean EDs with the
nurse's level and a three-expert consensus level (the target). See
`data/ktas/README.md` for provenance and licence (CC BY 4.0). The nurse level
is reported as a human reference row.

**NHAMCS-ED** (CDC, public, not bundled): download a year's Stata file, for
example `ed2022-stata.zip` from
`ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/`,
unzip into `data/nhamcs/`, then

```bash
python -m triage_eval.run --dataset nhamcs --sample 1500 --adults-only --out results/nhamcs
```

The loader keeps visits with `IMMEDR` 1-5, converts `TEMPF` to Celsius and
turns the reason-for-visit codes into text with the file's own value labels.
NHAMCS records no mental status at triage and its complaint is a coded
category, so expect lower ceilings than KTAS.

**MIMIC-IV-ED** is deliberately not wired in. PhysioNet's data-use agreement
forbids sending the data to third-party APIs without a no-retention
agreement, and that covers the TypeSafe endpoint. If you get credentialed and
want a local run, add a loader that maps `triage.csv` onto `TriageRecord`;
nothing else needs to change.

## What Jev is asked

`triage_eval/esi.py` builds one request per patient. The state is a JSON
object with age, sex, arrival mode, injury flag, chief complaint, mental
status, pain, and each vital sign as `{value, unit, bucket}` where the bucket
(for example `"tachycardia"`, `"severe hypoxia"`) is computed in
`triage_eval/buckets.py`. Nothing downstream of triage (diagnosis,
disposition, nurse level) is ever in the state; a test enforces that.

Questions, all answered in parallel from the same state:

| Key | Type | ESI decision point |
| --- | --- | --- |
| `lifesaving` | Noul | A: needs an immediate life-saving intervention |
| `high_risk` | Noul | B: high-risk situation |
| `altered_mental` | Noul | B: new confusion, lethargy, disorientation |
| `severe_distress` | Noul | B: severe pain or distress |
| `many_resources` | Noul | C: two or more resources |
| `any_resources` | Noul | C: at least one resource |
| `acuity` | Score | five ordered levels, level 1 first |

Decision point D (danger-zone vitals) is computed in code.

Three ways of turning the answers into a level are reported:

- `jev_score`: argmax of the Score distribution.
- `jev_rules`: the ESI ladder with thresholds on the Nouls (`Policy`), turned
  into a distribution by chaining the Noul probabilities.
- `jev_combined` (headline): levels 1 and 2 are gated by the Nouls, the
  remaining mass is shaped by the Score over levels 3-5.

Thresholds and the danger-zone weight live in `esi.Policy`. Because raw
answers are cached, changing the policy re-runs in seconds without new
inference. Pin a versioned model id (`--jev-model jev-1.13.0`) before you tune
thresholds; the run metadata records the model the API actually served.

## Metrics

Accuracy, balanced accuracy, macro F1, quadratic weighted kappa, mean absolute
level error, within-one-level rate, under- and over-triage rates (under =
predicted less urgent than truth, the dangerous direction), sensitivity and
specificity for levels 1-2, AUROC for levels 1-2 from the probabilities, and
for calibration: top-1 expected calibration error, multiclass Brier score, NLL,
and the reliability bins behind `calibration.png`.

ML rows are stratified out-of-fold predictions, so every patient is scored by a
model that never saw them. Jev rows are zero-shot. The nurse row is the human
performing the task live.

## Reference numbers (KTAS, ML baselines, 5-fold)

| Model | Acc | Bal. acc | QWK | MAE | Under | Over | Sens (L1-2) | ECE |
|---|---|---|---|---|---|---|---|---|
| majority class | 0.384 | 0.200 | 0.000 | 0.695 | 0.194 | 0.421 | 0.000 | |
| nurse (human) | 0.853 | 0.797 | 0.875 | 0.163 | 0.103 | 0.043 | 0.862 | |
| logreg | 0.640 | 0.661 | 0.628 | 0.448 | 0.196 | 0.164 | 0.691 | 0.053 |
| hgb | 0.685 | 0.558 | 0.670 | 0.363 | 0.163 | 0.152 | 0.593 | 0.161 |
| rf | 0.704 | 0.617 | 0.686 | 0.345 | 0.158 | 0.138 | 0.598 | 0.083 |

Jev rows will appear here after the first real run.

## Layout

```
triage_eval/
  datasets/base.py    TriageRecord schema shared by every loader
  datasets/ktas.py    bundled KTAS csv -> records
  datasets/nhamcs.py  CDC Stata/SAS file -> records
  buckets.py          vital-sign buckets, danger-zone rule, shock index (code, not model)
  esi.py              state builder, the seven questions, combination policies
  jev.py              TypeSafe backend, mock backend, JSONL cache, threaded runner
  baselines.py        feature matrix + logreg / HistGradientBoosting / RandomForest, out-of-fold
  metrics.py          classification, ordinal and calibration metrics
  plots.py            confusion, calibration and Noul reliability figures
  run.py              CLI
scripts/check_key.py  one-patient smoke test of the API key
tests/                12 tests, run without network
```

## Notes on the SDK

`typesafe-sdk` reads `TYPESAFE_API_KEY`, `TYPESAFE_DEFAULT_MODEL` and
`TYPESAFE_BASE_URL` from the environment. The call is
`TypeSafeClient().system_one(state=..., questions={...})`; Nouls return
`.noul` (P(yes)), Scores return `.score` (expected level index), `.confidence`
and `.probabilities` keyed by integer level index. Retries and rate-limit
back-off are handled by the SDK's `RetryPolicy`; the runner uses eight threads
by default, well inside the published 1,200 requests/minute limit.
