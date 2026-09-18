"""CLI: load a dataset, run Jev and the ML baselines, write metrics and figures.

    python -m triage_eval.run --dataset ktas --jev-backend typesafe --out results/ktas
    python -m triage_eval.run --dataset ktas --jev-backend mock --limit 200     # keyless dry run
    python -m triage_eval.run --dataset nhamcs --nhamcs-path data/nhamcs/ed2022.dta --sample 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .baselines import cross_val_predict_all, majority_baseline, nurse_reference
from .buckets import danger_zone
from .datasets import LOADERS
from .esi import METHODS, PROMPT_VERSION, Policy, predict
from .jev import JudgmentCache, judge_all, make_backend
from .metrics import evaluate, summary_table
from .plots import plot_calibration, plot_confusion, plot_noul_calibration


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Jev vs classical ML on public ED triage data")
    p.add_argument("--dataset", choices=sorted(LOADERS), default="ktas")
    p.add_argument("--nhamcs-path", default=None, help="NHAMCS .dta/.sas7bdat file or directory")
    p.add_argument("--limit", type=int, default=None, help="use only the first N records")
    p.add_argument("--sample", type=int, default=None, help="random sample of N records (NHAMCS)")
    p.add_argument("--adults-only", action="store_true", help="NHAMCS: drop under-18s")
    p.add_argument("--jev-backend", choices=["typesafe", "mock", "none"], default="typesafe")
    p.add_argument("--jev-model", default=None, help="model id, e.g. jev-1.13.0 to pin; default jev-latest")
    p.add_argument("--jev-methods", default="jev_combined,jev_rules,jev_score")
    p.add_argument("--ml-models", default="logreg,hgb,rf", help="comma list from logreg,hgb,rf or 'none'")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output directory (default results/<dataset>-<backend>)")
    p.add_argument("--cache", default=None, help="Jev cache path (default <out>/jev_cache.jsonl)")
    return p.parse_args(argv)


def load_records(args):
    if args.dataset == "ktas":
        return LOADERS["ktas"](limit=args.limit)
    return LOADERS["nhamcs"](path=args.nhamcs_path, limit=args.limit, sample=args.sample, seed=args.seed, adults_only=args.adults_only)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out or f"results/{args.dataset}-{args.jev_backend}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    records = load_records(args)
    y = np.array([r.true_acuity for r in records])
    print(f"[{args.dataset}] {len(records)} records; level counts {np.bincount(y, minlength=6)[1:].tolist()}", flush=True)

    pred_df = pd.DataFrame({"record_id": [r.record_id for r in records], "true_acuity": y})
    if all(r.nurse_acuity is not None for r in records):
        pred_df["nurse_acuity"] = [r.nurse_acuity for r in records]
    results: dict[str, dict] = {}

    def record(name, pred, probs):
        results[name] = evaluate(y, pred, probs)
        pred_df[f"pred_{name}"] = pred
        if probs is not None:
            for k in range(5):
                pred_df[f"p{k + 1}_{name}"] = np.asarray(probs)[:, k]

    # Reference rows
    maj = majority_baseline(records)
    record("majority", maj.pred, maj.probs)
    nurse = nurse_reference(records)
    if nurse is not None:
        record("nurse", nurse.pred, None)

    # Jev
    jev_meta = {}
    if args.jev_backend != "none":
        backend = make_backend(args.jev_backend, model=args.jev_model)
        cache = JudgmentCache(Path(args.cache) if args.cache else out / "jev_cache.jsonl", backend=backend.name)
        print(f"[jev] backend={backend.name} prompt={PROMPT_VERSION} cached={len(cache)}", flush=True)

        def progress(done, total):
            if done % 50 == 0 or done == total:
                print(f"[jev] {done}/{total}", flush=True)

        try:
            judgments = judge_all(records, backend, cache, workers=args.workers, progress=progress)
        finally:
            backend.close()
        pol = Policy()
        for method in [m.strip() for m in args.jev_methods.split(",") if m.strip()]:
            if method not in METHODS:
                raise SystemExit(f"unknown jev method {method}")
            preds, probs = [], []
            for r in records:
                lvl, pr = predict(judgments[r.record_id], danger_zone(r), method, pol)
                preds.append(lvl)
                probs.append(pr)
            name = method if backend.name == "typesafe" else f"{method}[{backend.name}]"
            record(name, np.array(preds), np.array(probs))
        for k in ("lifesaving", "high_risk", "altered_mental", "severe_distress", "many_resources", "any_resources"):
            pred_df[f"noul_{k}"] = [judgments[r.record_id].nouls[k] for r in records]
        pred_df["jev_score_expected_level"] = [judgments[r.record_id].score_expected + 1 for r in records]
        pred_df["jev_score_confidence"] = [judgments[r.record_id].score_confidence for r in records]
        lat = [j.latency_s for j in judgments.values() if j.latency_s]
        toks = [j.input_tokens for j in judgments.values() if j.input_tokens]
        jev_meta = {
            "backend": backend.name,
            "model": sorted({j.model for j in judgments.values()}),
            "prompt_version": PROMPT_VERSION,
            "requests": len(judgments),
            "median_latency_s": float(np.median(lat)) if lat else None,
            "p95_latency_s": float(np.percentile(lat, 95)) if lat else None,
            "total_input_tokens": int(sum(toks)) if toks else None,
            "policy": pol.__dict__,
        }
        # Noul sanity plot against proxies computable from the data itself
        targets = {
            "lifesaving": (y == 1).astype(float),
            "high_risk": (y <= 2).astype(float),
            "many_resources": (y <= 3).astype(float),
            "any_resources": (y <= 4).astype(float),
        }
        if all(r.mental_status is not None for r in records):
            targets["altered_mental"] = np.array([r.mental_status != "alert" for r in records], dtype=float)
        nouls = {k: pred_df[f"noul_{k}"].to_numpy() for k in targets}
        plot_noul_calibration(nouls, targets, out / "noul_calibration.png")

    # ML baselines (out-of-fold)
    ml_names = [m.strip() for m in args.ml_models.split(",") if m.strip() and m.strip() != "none"]
    if ml_names:
        print(f"[ml] {args.folds}-fold CV for {ml_names}", flush=True)
        for name, res in cross_val_predict_all(records, ml_names, folds=args.folds, seed=args.seed).items():
            record(name, res.pred, res.probs)

    # Outputs
    for name, m in results.items():
        plot_confusion(m["confusion_matrix"], f"{name} on {args.dataset} (row-normalised)", out / f"confusion_{name}.png")
    plot_calibration({k: v for k, v in results.items() if "reliability" in v and k != "majority"}, out / "calibration.png")
    pred_df.to_csv(out / "predictions.csv", index=False)
    table = summary_table(results)
    meta = {
        "version": __version__,
        "dataset": args.dataset,
        "n": len(records),
        "args": vars(args),
        "jev": jev_meta,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    md = [f"# {args.dataset}: Jev vs ML", "", f"n = {len(records)}. Under = predicted less urgent than truth. ML rows are {args.folds}-fold out-of-fold; Jev rows are zero-shot.", ""]
    if jev_meta:
        md.append(f"Jev backend `{jev_meta['backend']}`, model {jev_meta['model']}, prompt `{PROMPT_VERSION}`, median latency {jev_meta['median_latency_s']} s.")
        if jev_meta["backend"] != "typesafe":
            md.append("**Rows tagged [mock] come from the keyless heuristic stand-in and are not Jev results.**")
        md.append("")
    md += [table, "", "Figures: confusion_<model>.png, calibration.png, noul_calibration.png"]
    (out / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n" + table + "\n")
    print(f"wrote {out}/ in {meta['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
