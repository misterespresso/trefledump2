"""Classical ML baselines trained on the same fields Jev sees.

All models produce out-of-fold predictions via stratified K-fold so every patient
has a prediction from a model that never saw them, which is the fair comparison
against Jev's zero-shot answers. The nurse's own level is reported as a human
reference where the dataset has it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .buckets import bucket_dbp, bucket_hr, bucket_pain, bucket_rr, bucket_sbp, bucket_spo2, bucket_temp, danger_zone, shock_index
from .datasets.base import ACUITY_LEVELS, TriageRecord

NUMERIC = ["age", "pain_score", "sbp", "dbp", "hr", "rr", "temp_c", "spo2", "shock_index"]
CATEGORICAL = ["sex", "arrival_mode", "mental_status", "b_sbp", "b_dbp", "b_hr", "b_rr", "b_spo2", "b_temp", "b_pain"]
BINARY = ["injury", "pain_present", "danger_zone"]
TEXT = "chief_complaint"


def feature_frame(records: list[TriageRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "record_id": r.record_id,
                "age": r.age,
                "pain_score": r.pain_score,
                "sbp": r.sbp,
                "dbp": r.dbp,
                "hr": r.hr,
                "rr": r.rr,
                "temp_c": r.temp_c,
                "spo2": r.spo2,
                "shock_index": shock_index(r),
                "sex": r.sex or "missing",
                "arrival_mode": r.arrival_mode or "missing",
                "mental_status": r.mental_status or "missing",
                "b_sbp": bucket_sbp(r.sbp),
                "b_dbp": bucket_dbp(r.dbp),
                "b_hr": bucket_hr(r.hr, r.age),
                "b_rr": bucket_rr(r.rr, r.age),
                "b_spo2": bucket_spo2(r.spo2),
                "b_temp": bucket_temp(r.temp_c),
                "b_pain": bucket_pain(r.pain_score),
                "injury": float(r.injury) if r.injury is not None else np.nan,
                "pain_present": float(r.pain_present) if r.pain_present is not None else np.nan,
                "danger_zone": float(danger_zone(r)),
                TEXT: (r.chief_complaint or "").lower(),
                "y": r.true_acuity,
            }
        )
    return pd.DataFrame(rows)


def _preprocessor(text_dims: Optional[int]) -> ColumnTransformer:
    text_pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True))
    if text_dims:
        text_pipe.steps.append(("svd", TruncatedSVD(n_components=text_dims, random_state=0)))
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
            ("bin", SimpleImputer(strategy="constant", fill_value=0.0), BINARY),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("txt", text_pipe, TEXT),
        ],
        sparse_threshold=0.3,
    )


class _Dense:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else X

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self


def build_models(seed: int = 0) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("prep", _preprocessor(text_dims=None)),
                ("clf", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=seed)),
            ]
        ),
        "hgb": Pipeline(
            [
                ("prep", _preprocessor(text_dims=48)),
                ("dense", _Dense()),
                ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)),
            ]
        ),
        "rf": Pipeline(
            [
                ("prep", _preprocessor(text_dims=48)),
                ("dense", _Dense()),
                ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
            ]
        ),
    }


@dataclass
class BaselineResult:
    name: str
    pred: np.ndarray  # levels 1..5, aligned with the input records
    probs: np.ndarray  # shape (n, 5), columns are levels 1..5


def cross_val_predict_all(
    records: list[TriageRecord], names: list[str], folds: int = 5, seed: int = 0
) -> dict[str, BaselineResult]:
    df = feature_frame(records)
    y = df["y"].to_numpy()
    X = df.drop(columns=["y", "record_id"])
    models = build_models(seed)
    counts = np.bincount(y, minlength=6)[1:]
    rarest = int(counts[counts > 0].min())
    if rarest >= folds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    elif rarest >= 2:
        skf = StratifiedKFold(n_splits=rarest, shuffle=True, random_state=seed)
    else:  # a class with a single example cannot be stratified
        skf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    out: dict[str, BaselineResult] = {}
    for name in names:
        model = models[name]
        probs = np.zeros((len(df), len(ACUITY_LEVELS)))
        for tr, te in skf.split(X, y):
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])
            for col, cls in enumerate(model.classes_):
                probs[te, int(cls) - 1] = p[:, col]
        pred = probs.argmax(axis=1) + 1
        out[name] = BaselineResult(name=name, pred=pred, probs=probs)
    return out


def majority_baseline(records: list[TriageRecord]) -> BaselineResult:
    y = np.array([r.true_acuity for r in records])
    counts = np.bincount(y, minlength=6)[1:]
    probs = np.tile(counts / counts.sum(), (len(y), 1))
    return BaselineResult("majority", probs.argmax(axis=1) + 1, probs)


def nurse_reference(records: list[TriageRecord]) -> Optional[BaselineResult]:
    if any(r.nurse_acuity is None for r in records):
        return None
    pred = np.array([r.nurse_acuity for r in records])
    probs = np.eye(5)[pred - 1]
    return BaselineResult("nurse", pred, probs)
