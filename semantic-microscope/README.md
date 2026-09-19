# Semantic microscope

An embedding of 448 Python packages in which **every dimension is a sentence you can read**.

A normal sentence embedding has several hundred dimensions and not one of them means anything
on its own. This one has forty, and dimension 19 is literally *"It defines itself against an
existing alternative, as a replacement, drop-in, successor or improvement on something else."*
Each package's
one-line summary went to [TypeSafe's Jev](https://typesafe.ai) once, carrying all forty questions
as `Noul` primitives; the forty returned probabilities are the coordinates.

Live page: **https://claude.ai/artifact/9GTnNFTuMfeB9AaeLVto1g**

## The result

| | mean held-out AUROC |
| --- | --- |
| 40 readable lenses | **0.727** |
| TF-IDF char n-grams on the very same sentence | **0.737** |

Forty numbers a person can read carry about as much signal as several thousand nobody can.
That is a tie, not a win, and it is the honest headline: the lenses buy interpretability at
roughly zero cost in information, which is interesting, rather than beating the baseline,
which they do not. They do win on two targets of seven — *Scientific/Engineering* (0.895 vs
0.815) and *Utilities* (0.647 vs 0.586) — and lose worst on *System* (0.576 vs 0.701).

Targets are PyPI's own topic classifiers and development-status metadata, which the model never
saw: `state_for()` sends the summary and nothing else, so name, topics, maturity, rank and
release count are all clean ground truth. Five-fold stratified cross-validation, logistic
regression on both sides.

## What the axes turned out to be

The first three principal components of the forty readings, each stated in the instrument's own
words — this is the part an opaque embedding cannot give you:

| | variance | high end | low end |
| --- | --- | --- | --- |
| PC1 | 13.7% | infrastructure, low-level, reassuringly boring | superlatives, full sentences, batteries included |
| PC2 | 9.0% | hard to explain, assumes jargon, sounds early | boring, shows you something, explains its purpose |
| PC3 | 8.1% | modest, low-level, early | batteries included, vendor SDK, networked |

PC1 is the plumbing-to-product axis and it is the single largest source of variance in how
Python packages describe themselves.

## What is wrong with it

- **One dead lens.** `for_developers` answered 0.93 with sd 0.04. Everything on PyPI is for
  developers, so the question never discriminates. A lens that agrees with everything measures
  nothing, and it should be replaced rather than kept for the round number.
- **Three near-dead ones.** `is_batteries_included` (sd 0.111), `is_playful` (0.107) and
  `is_framework` (0.093) barely move either.
- **Visible redundancy.** `is_library` ~ `is_cli` correlate at −0.86: the library/tool
  distinction was asked twice. `assumes_jargon` ~ `hard_to_explain` at +0.77, `is_sdk` ~
  `names_a_vendor` at +0.72. In an opaque embedding this duplication is invisible; here you can
  read it and delete one side.
- **One sentence per package.** Summaries are 25–200 characters. Whatever a package is that its
  authors did not put in the pitch, the instrument cannot see.
- **Only the mean is validated.** Each reading is a single call; no repeat measurements, so
  per-reading stability is unmeasured.

## Running it

```sh
export TYPESAFE_API_KEY=...          # never passed in a prompt, never logged
python -m scope.corpus               # top-500 PyPI ranking -> data/corpus.json  (448 kept)
python -m scope.measure              # 448 requests, all 40 lenses each -> data/readings.jsonl
python -m scope.analyse              # validation, PCA, correlations -> data/analysis.json
python -m scope.page                 # bundle -> page/data.js
pytest tests                         # offline; no network, no key
```

`measure` is resumable: readings append to `data/readings.jsonl` keyed by package name, and a
rerun only asks for what is missing.

## Cost of the run recorded here

448 requests, 501,049 input tokens, model `jev-1.13.0`, one pass, no retries needed.
