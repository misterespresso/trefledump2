"""Held-out tuning of the code-side policy on cached Jev answers. No new inference.

Everything is cross-fitted: parameters are chosen on K-1 folds and applied to the
held-out fold, so every reported prediction comes from parameters that never saw
that patient. This is the same protocol as the ML baselines.

Variants:
  jev_rules_tuned   ESI ladder with the six Noul thresholds chosen by grid search
  jev_score_shift   round(expected level + s), s chosen on the training folds
  jev_score_cuts    ordinal cut-points on the expected level (4 parameters)
  jev_stacked_lr    multinomial logistic regression on Jev's outputs only
                    (log Score probabilities, logit Nouls, danger-zone flag);
                    the only variant that also recalibrates the probabilities

    python -m triage_eval.tune --cache results/ktas/jev_cache.jsonl --out results/ktas-tuned
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .buckets import danger_zone
from .datasets import LOADERS, SUBSETS, subset
from .esi import DEFAULT_PROMPT_VERSION, NOUL_KEYS, PROMPT_VERSIONS, Policy
from .jev import JudgmentCache
from .metrics import evaluate, summary_table
from .plots import plot_calibration, plot_confusion

LEVELS = np.arange(1, 6)


# ----------------------------------------------------------------------------- fast metrics
def qwk(y: np.ndarray, p: np.ndarray) -> float:
    """Quadratic weighted kappa on levels 1..5, vectorised."""
    k = 5
    o = np.zeros((k, k))
    np.add.at(o, (y - 1, p - 1), 1)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    denom = (w * e).sum()
    return 1.0 - (w * o).sum() / denom if denom > 0 else 0.0


def make_objectives(max_under: float) -> dict:
    """qwk_safe: QWK with a steep penalty once under-triage exceeds max_under (the dangerous direction)."""

    def qwk_safe(y, p):
        under = float((p > y).mean())
        return qwk(y, p) - 10.0 * max(0.0, under - max_under)

    return {
        "qwk": qwk,
        "qwk_safe": qwk_safe,
        "accuracy": lambda y, p: float((y == p).mean()),
        "balanced": lambda y, p: float(np.mean([(p[y == c] == c).mean() for c in LEVELS if (y == c).any()])),
    }


OBJECTIVES = make_objectives(0.10)


# ----------------------------------------------------------------------------- data
def load_judgments(cache_path: Path, records, backend: str = "typesafe", version: str = DEFAULT_PROMPT_VERSION):
    cache = JudgmentCache(cache_path, backend=backend, version=version)
    missing = [r.record_id for r in records if cache.get(r.record_id) is None]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(records)} records have no cached answer for "
                         f"{version!r}/{backend!r}.{_cache_hint(cache_path, version, backend)}")
    return [cache.get(r.record_id) for r in records]


def _cache_hint(cache_path: Path, version: str, backend: str) -> str:
    """Name the prompt versions the file does hold, since a version mismatch looks like an empty cache."""
    present: set[str] = set()
    try:
        with Path(cache_path).open() as f:
            for line in f:
                if line.strip():
                    key = json.loads(line).get("cache_key", "")
                    if key:
                        present.add(key)
    except OSError:
        return f" No readable cache at {cache_path}. Run triage_eval.run first."
    if not present:
        return f" {cache_path} is empty. Run triage_eval.run first."
    others = sorted(p.split(":")[0] for p in present if not p.startswith(f"{version}:"))
    if others:
        return (f" That file holds answers for {', '.join(sorted(set(others)))} instead."
                f" Pass --prompt-version {others[0]}, or run triage_eval.run to fetch {version} answers.")
    return (f" The file holds {version} answers under a different question hash or backend,"
            f" so the questions have changed since it was written. Re-run triage_eval.run.")


def arrays(records, judgments):
    y = np.array([r.true_acuity for r in records])
    N = np.array([[j.nouls[k] for k in NOUL_KEYS] for j in judgments])  # (n, 6)
    S = np.array([[j.score_probs.get(k, 0.0) for k in range(5)] for j in judgments])
    S = S / S.sum(1, keepdims=True)
    expected = S @ LEVELS
    dz = np.array([danger_zone(r) for r in records])
    return y, N, S, expected, dz


# ----------------------------------------------------------------------------- variants
def rules_predict(N: np.ndarray, dz: np.ndarray, t: dict) -> np.ndarray:
    ls, hr, am, sd, mr, ar = N.T
    return np.where(
        ls >= t["t_lifesaving"],
        1,
        np.where(
            (hr >= t["t_high_risk"]) | (am >= t["t_altered"]) | (sd >= t["t_distress"]),
            2,
            np.where(mr >= t["t_many"], np.where(dz, 2, 3), np.where(ar >= t["t_any"], 4, 5)),
        ),
    )


RULE_GRID = {
    "t_lifesaving": [0.3, 0.5, 0.7],
    "t_high_risk": [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)],
    "t_altered": [0.3, 0.5, 0.7],
    "t_distress": [0.3, 0.5, 0.7, 0.9],
    "t_many": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "t_any": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}


def fit_rules(N, dz, y, objective) -> dict:
    keys = list(RULE_GRID)
    best, best_t = -np.inf, None
    for combo in itertools.product(*(RULE_GRID[k] for k in keys)):
        t = dict(zip(keys, combo))
        v = objective(y, rules_predict(N, dz, t))
        if v > best:
            best, best_t = v, t
    return best_t


def fit_shift(expected, y, objective) -> float:
    grid = np.round(np.arange(-0.5, 1.01, 0.05), 2)
    scores = [objective(y, np.clip(np.round(expected + s), 1, 5).astype(int)) for s in grid]
    return float(grid[int(np.argmax(scores))])


def shift_predict(expected, s):
    return np.clip(np.round(expected + s), 1, 5).astype(int)


def cuts_predict(expected, cuts):
    return (np.searchsorted(np.asarray(cuts), expected, side="right") + 1).astype(int)


def fit_cuts(expected, y, objective) -> list[float]:
    """Coordinate ascent on four monotone cut-points, started from the rounding cuts."""
    cuts = [1.5, 2.5, 3.5, 4.5]
    grid = np.round(np.arange(1.0, 5.01, 0.05), 2)  # rounded so a stored cut reproduces the fitted decision
    for _ in range(4):
        changed = False
        for i in range(4):
            lo = cuts[i - 1] + 0.05 if i > 0 else 1.0
            hi = cuts[i + 1] - 0.05 if i < 3 else 5.0
            cands = grid[(grid >= lo) & (grid <= hi)]
            vals = []
            for c in cands:
                trial = list(cuts)
                trial[i] = float(c)
                vals.append(objective(y, cuts_predict(expected, trial)))
            new = float(cands[int(np.argmax(vals))])
            if new != cuts[i]:
                cuts[i] = new
                changed = True
        if not changed:
            break
    return [float(c) for c in cuts]


def stack_features(N, S, dz):
    eps = 1e-4
    logit = np.log(np.clip(N, eps, 1 - eps) / np.clip(1 - N, eps, 1 - eps))
    return np.hstack([np.log(np.clip(S, eps, None)), logit, dz[:, None].astype(float)])


def fit_stack(X, y, seed):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=5000, C=0.5, random_state=seed).fit(sc.transform(X), y)
    return sc, clf


def stack_proba(model, X):
    sc, clf = model
    p = clf.predict_proba(sc.transform(X))
    out = np.zeros((len(X), 5))
    for col, cls in enumerate(clf.classes_):
        out[:, int(cls) - 1] = p[:, col]
    return out


# ----------------------------------------------------------------------------- main
def run(records, judgments, folds=5, seed=0, objective="qwk", base_metrics: Path | None = None, max_under=0.10):
    obj = make_objectives(max_under)[objective]
    y, N, S, expected, dz = arrays(records, judgments)
    n = len(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    preds = {k: np.zeros(n, dtype=int) for k in ("jev_rules_tuned", "jev_score_shift", "jev_score_cuts", "jev_stacked_lr")}
    probs_stack = np.zeros((n, 5))
    chosen = []
    X = stack_features(N, S, dz)
    for f, (tr, te) in enumerate(skf.split(X, y)):
        t = fit_rules(N[tr], dz[tr], y[tr], obj)
        preds["jev_rules_tuned"][te] = rules_predict(N[te], dz[te], t)
        s = fit_shift(expected[tr], y[tr], obj)
        preds["jev_score_shift"][te] = shift_predict(expected[te], s)
        cuts = fit_cuts(expected[tr], y[tr], obj)
        preds["jev_score_cuts"][te] = cuts_predict(expected[te], cuts)
        model = fit_stack(X[tr], y[tr], seed)
        p = stack_proba(model, X[te])
        probs_stack[te] = p
        preds["jev_stacked_lr"][te] = p.argmax(1) + 1
        chosen.append({"fold": f, "n_train": int(len(tr)), **t, "shift": s, "cuts": cuts})

    results: dict[str, dict] = {}
    # untuned references, recomputed here so the table is self-contained
    results["jev_rules (default)"] = evaluate(y, rules_predict(N, dz, Policy().__dict__), None)
    results["jev_score (argmax)"] = evaluate(y, S.argmax(1) + 1, S)
    results["jev_rules_tuned"] = evaluate(y, preds["jev_rules_tuned"], None)
    results["jev_score_shift"] = evaluate(y, preds["jev_score_shift"], None)
    results["jev_score_cuts"] = evaluate(y, preds["jev_score_cuts"], None)
    results["jev_stacked_lr"] = evaluate(y, preds["jev_stacked_lr"], probs_stack)
    if base_metrics and base_metrics.exists():
        base = json.loads(base_metrics.read_text())["results"]
        for k in ("nurse", "logreg", "hgb", "rf"):
            if k in base:
                results[k] = base[k]

    # final parameters on all data, for deployment / the README
    full = {
        "rules": fit_rules(N, dz, y, obj),
        "shift": fit_shift(expected, y, obj),
        "cuts": fit_cuts(expected, y, obj),
    }
    pred_df = pd.DataFrame({"record_id": [r.record_id for r in records], "true_acuity": y, "jev_expected_level": expected})
    for k, v in preds.items():
        pred_df[f"pred_{k}"] = v
    for k in range(5):
        pred_df[f"p{k + 1}_jev_stacked_lr"] = probs_stack[:, k]
    return results, chosen, full, pred_df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-fitted tuning of the Jev combination policy")
    ap.add_argument("--dataset", choices=sorted(LOADERS), default="ktas")
    ap.add_argument("--nhamcs-path", default=None)
    ap.add_argument("--cache", required=True, help="jev_cache.jsonl from a real run")
    ap.add_argument("--base-metrics", default=None, help="metrics.json of the base run, to copy nurse/ML rows")
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="qwk")
    ap.add_argument("--max-under", type=float, default=0.10, help="under-triage ceiling used by the qwk_safe objective")
    ap.add_argument("--backend", default="typesafe", help="cache backend tag; 'mock' for dry-run caches")
    ap.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--subset", choices=SUBSETS, default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records = LOADERS[args.dataset]() if args.dataset == "ktas" else LOADERS[args.dataset](path=args.nhamcs_path)
    records = subset(records, args.subset, seed=0)
    judgments = load_judgments(Path(args.cache), records, args.backend, args.prompt_version)
    base_metrics = Path(args.base_metrics) if args.base_metrics else Path(args.cache).parent / "metrics.json"
    results, chosen, full, pred_df = run(records, judgments, args.folds, args.seed, args.objective, base_metrics, args.max_under)

    for name, m in results.items():
        if name.startswith("jev_"):
            plot_confusion(m["confusion_matrix"], f"{name} on {args.dataset} (row-normalised)", out / f"confusion_{name.split(' ')[0]}.png")
    plot_calibration({k: v for k, v in results.items() if "reliability" in v}, out / "calibration.png")
    pred_df.to_csv(out / "predictions.csv", index=False)
    table = summary_table(results)
    (out / "metrics.json").write_text(
        json.dumps({"meta": {"objective": args.objective, "max_under": args.max_under, "prompt_version": args.prompt_version, "subset": args.subset, "folds": args.folds, "seed": args.seed, "cache": str(args.cache), "elapsed_s": round(time.time() - t0, 1)}, "per_fold": chosen, "final_params": full, "results": results}, indent=2)
    )
    fold_df = pd.DataFrame(chosen)
    md = [
        f"# {args.dataset} ({args.subset}, {args.prompt_version}): held-out tuning of the Jev policy",
        "",
        f"{args.folds}-fold cross-fitted, objective = {args.objective}" + (f" (under-triage ceiling {args.max_under})" if args.objective == "qwk_safe" else "") + ". Every jev_*_tuned / shift / cuts / stacked row is out-of-fold: parameters chosen on the other folds. Rows in parentheses are the untuned references; nurse and ML rows are copied from the base run.",
        "",
        table,
        "",
        "## Parameters chosen per fold",
        "",
        fold_df.to_markdown(index=False),
        "",
        "## Parameters fitted on all data (for deployment)",
        "",
        "```json",
        json.dumps(full, indent=1),
        "```",
    ]
    (out / "metrics.md").write_text("\n".join(md) + "\n")
    print(table)
    print("\nper-fold parameters:\n" + fold_df.to_string(index=False))
    print("\nfull-data parameters:", json.dumps(full))
    print(f"\nwrote {out}/ in {round(time.time() - t0, 1)} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
