"""Classification, ordinal and calibration metrics for 5-level acuity."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .datasets.base import ACUITY_LEVELS


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(m.sum()),
                "mean_confidence": float(conf[m].mean()) if m.any() else float("nan"),
                "accuracy": float(correct[m].mean()) if m.any() else float("nan"),
            }
        )
    return rows


def evaluate(y_true: np.ndarray, pred: np.ndarray, probs: np.ndarray | None) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    m: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=ACUITY_LEVELS, zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, pred, weights="quadratic", labels=ACUITY_LEVELS)),
        "mean_absolute_level_error": float(np.abs(y_true - pred).mean()),
        "within_one_level": float((np.abs(y_true - pred) <= 1).mean()),
        # higher number = less urgent, so pred > true is under-triage
        "under_triage_rate": float((pred > y_true).mean()),
        "over_triage_rate": float((pred < y_true).mean()),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=ACUITY_LEVELS).tolist(),
    }
    high = y_true <= 2
    if high.any() and (~high).any():
        m["high_acuity_sensitivity"] = float((pred[high] <= 2).mean())
        m["high_acuity_specificity"] = float((pred[~high] > 2).mean())
    if probs is not None:
        probs = np.asarray(probs, dtype=float)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        conf = probs.max(axis=1)
        top = probs.argmax(axis=1) + 1
        correct = (top == y_true).astype(float)
        onehot = np.eye(5)[y_true - 1]
        m["ece_top1"] = expected_calibration_error(conf, correct)
        m["brier_multiclass"] = float(((probs - onehot) ** 2).sum(axis=1).mean())
        m["nll"] = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true - 1], 1e-12, None)).mean())
        m["mean_confidence"] = float(conf.mean())
        m["reliability"] = reliability_bins(conf, correct)
        p_high = probs[:, :2].sum(axis=1)
        if high.any() and (~high).any() and len(np.unique(p_high)) > 1:
            m["auroc_high_acuity"] = float(roc_auc_score(high.astype(int), p_high))
    return m


SUMMARY_COLUMNS = [
    ("accuracy", "Acc"),
    ("balanced_accuracy", "Bal. acc"),
    ("macro_f1", "Macro F1"),
    ("quadratic_weighted_kappa", "QWK"),
    ("mean_absolute_level_error", "MAE"),
    ("within_one_level", "±1 level"),
    ("under_triage_rate", "Under"),
    ("over_triage_rate", "Over"),
    ("high_acuity_sensitivity", "Sens (L1-2)"),
    ("auroc_high_acuity", "AUROC (L1-2)"),
    ("ece_top1", "ECE"),
    ("brier_multiclass", "Brier"),
]


def summary_table(results: dict[str, dict[str, Any]]) -> str:
    head = "| Model | " + " | ".join(label for _, label in SUMMARY_COLUMNS) + " |"
    sep = "|" + "---|" * (len(SUMMARY_COLUMNS) + 1)
    lines = [head, sep]
    for name, m in results.items():
        cells = []
        for key, _ in SUMMARY_COLUMNS:
            v = m.get(key)
            cells.append("" if v is None else f"{v:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
