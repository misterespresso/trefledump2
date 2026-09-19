"""Is the instrument measuring anything, and what does it see?

Validation is the part that matters. Forty readable axes are only interesting if they
carry real information, so they are tested against held-out metadata the model never
saw (PyPI topic classifiers and maturity) and compared with TF-IDF on the very same
sentence. If character n-grams win by a mile, the lenses are decoration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"
MATRIX, OUT = DATA / "matrix.json", DATA / "analysis.json"
TOP_TOPICS = 6
FOLDS = 5


def load() -> tuple[dict[str, Any], list[str], np.ndarray, list[dict[str, Any]]]:
    d = json.loads(MATRIX.read_text())
    keys = list(d["lenses"])
    pkgs = d["packages"]
    X = np.array([[p["readings"][k] for k in keys] for p in pkgs])
    return d["lenses"], keys, X, pkgs


def auroc_cv(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    if y.sum() < FOLDS or (~y.astype(bool)).sum() < FOLDS:
        return float("nan")
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=3000, C=1.0))
    return float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())


def auroc_cv_text(texts: list[str], y: np.ndarray, seed: int = 0) -> float:
    if y.sum() < FOLDS or (~y.astype(bool)).sum() < FOLDS:
        return float("nan")
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=3000, C=1.0))
    return float(cross_val_score(model, texts, y, cv=cv, scoring="roc_auc").mean())


def main() -> int:
    lenses, keys, X, pkgs = load()
    texts = [p["summary"] for p in pkgs]
    n = len(pkgs)

    # ---- targets the model never saw -------------------------------------
    from collections import Counter
    topic_counts = Counter(t for p in pkgs for t in p["topics"])
    targets: dict[str, np.ndarray] = {}
    for topic, _ in topic_counts.most_common(TOP_TOPICS):
        targets[f"topic: {topic}"] = np.array([topic in p["topics"] for p in pkgs], dtype=int)
    targets["maturity: production/stable"] = np.array(
        [p["maturity"] == "5 - Production/Stable" for p in pkgs], dtype=int)

    validation = []
    for name, y in targets.items():
        validation.append({
            "target": name, "positives": int(y.sum()), "n": n,
            "auroc_lenses": auroc_cv(X, y),
            "auroc_tfidf": auroc_cv_text(texts, y),
        })

    # ---- structure ---------------------------------------------------------
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pca = PCA(n_components=min(10, X.shape[1])).fit(Xs)
    coords = PCA(n_components=2).fit_transform(Xs)
    comps = []
    for i, (var, vec) in enumerate(zip(pca.explained_variance_ratio_, pca.components_)):
        order = np.argsort(vec)
        comps.append({
            "component": i + 1, "variance": float(var),
            "high": [{"lens": keys[j], "text": lenses[keys[j]]["text"], "weight": float(vec[j])}
                     for j in order[::-1][:4]],
            "low": [{"lens": keys[j], "text": lenses[keys[j]]["text"], "weight": float(vec[j])}
                    for j in order[:4]],
        })

    corr = np.corrcoef(X.T)
    pairs = [{"a": keys[i], "b": keys[j], "r": float(corr[i, j])}
             for i in range(len(keys)) for j in range(i + 1, len(keys))]
    pairs.sort(key=lambda d: -abs(d["r"]))

    spread = [{"lens": k, "text": lenses[k]["text"], "group": lenses[k]["group"],
               "mean": float(X[:, i].mean()), "sd": float(X[:, i].std()),
               "share_above_half": float((X[:, i] >= 0.5).mean())}
              for i, k in enumerate(keys)]
    spread.sort(key=lambda d: -d["sd"])

    out = {
        "n_packages": n, "n_lenses": len(keys),
        "served_models": sorted({p.get("served_model") for p in pkgs if p.get("served_model")}),
        "validation": validation,
        "mean_auroc_lenses": float(np.nanmean([v["auroc_lenses"] for v in validation])),
        "mean_auroc_tfidf": float(np.nanmean([v["auroc_tfidf"] for v in validation])),
        "components": comps,
        "strongest_pairs": pairs[:12],
        "lens_spread": spread,
        "coords": [{"name": p["name"], "x": float(coords[i, 0]), "y": float(coords[i, 1])}
                   for i, p in enumerate(pkgs)],
    }
    OUT.write_text(json.dumps(out, indent=1))

    print(f"{n} packages x {len(keys)} lenses, model {out['served_models']}\n")
    print(f"{'held-out target':34s} {'pos':>4}  {'lenses':>7} {'tf-idf':>7}")
    for v in validation:
        print(f"  {v['target']:32s} {v['positives']:>4}  {v['auroc_lenses']:>7.3f} {v['auroc_tfidf']:>7.3f}")
    print(f"  {'mean':32s} {'':>4}  {out['mean_auroc_lenses']:>7.3f} {out['mean_auroc_tfidf']:>7.3f}")
    print(f"\ntop 3 components explain "
          f"{sum(c['variance'] for c in comps[:3]):.1%} of variance")
    for c in comps[:3]:
        print(f"  PC{c['component']} ({c['variance']:.1%}): {c['high'][0]['lens']} / {c['high'][1]['lens']}"
              f"  <->  {c['low'][0]['lens']} / {c['low'][1]['lens']}")
    print("\nmost correlated lens pairs:")
    for p in pairs[:5]:
        print(f"  r={p['r']:+.2f}  {p['a']} ~ {p['b']}")
    print("\nleast decisive lenses (lowest spread):")
    for s in spread[-4:]:
        print(f"  sd={s['sd']:.3f} mean={s['mean']:.2f}  {s['lens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
