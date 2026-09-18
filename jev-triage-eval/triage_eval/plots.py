"""Confusion-matrix and calibration figures (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .datasets.base import ACUITY_LEVELS  # noqa: E402

# One-hue sequential ramp (light -> dark blue) for magnitude; fixed categorical order for series.
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def plot_confusion(cm: list[list[int]], title: str, path: Path, normalize: bool = True) -> None:
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    shown = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) if normalize else cm
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.imshow(shown, cmap=SEQ, vmin=0, vmax=1 if normalize else shown.max())
    for i in range(5):
        for j in range(5):
            v = shown[i, j]
            label = f"{v:.0%}\n({int(cm[i, j])})" if normalize else f"{int(v)}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#ffffff" if v > 0.55 * (1 if normalize else shown.max()) else TEXT)
    ax.set_xticks(range(5), [f"L{k}" for k in ACUITY_LEVELS])
    ax.set_yticks(range(5), [f"L{k}" for k in ACUITY_LEVELS])
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("True level")
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(results: dict[str, dict[str, Any]], path: Path) -> None:
    """Reliability diagram of top-class confidence for every model with probabilities."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=150, gridspec_kw={"width_ratios": [1, 1]})
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--", zorder=1)
    i = 0
    for name, m in results.items():
        rel = m.get("reliability")
        if not rel:
            continue
        color = SERIES[i % len(SERIES)]
        xs = [b["mean_confidence"] for b in rel if b["count"] > 0]
        ys = [b["accuracy"] for b in rel if b["count"] > 0]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, markeredgecolor="#ffffff", markeredgewidth=1, label=f"{name} (ECE {m['ece_top1']:.3f})", zorder=3)
        counts = [b["count"] for b in rel]
        centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in rel]
        ax2.step(centers, np.array(counts) / max(1, sum(counts)), where="mid", color=color, lw=1.8, label=name)
        i += 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of the chosen level")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration of the top prediction", fontsize=11, color=TEXT, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted probability bin")
    ax2.set_ylabel("Share of patients")
    ax2.set_title("Confidence distribution", fontsize=11, color=TEXT, loc="left")
    ax2.grid(color=GRID, lw=0.6)
    ax2.legend(frameon=False, fontsize=8)
    _style(ax2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_noul_calibration(nouls: dict[str, np.ndarray], targets: dict[str, np.ndarray], path: Path) -> None:
    """Reliability of the individual Nouls against code-defined proxies of their targets."""
    keys = [k for k in nouls if k in targets]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.4), dpi=150, squeeze=False)
    for ax, k in zip(axes[0], keys):
        p = np.asarray(nouls[k], dtype=float)
        t = np.asarray(targets[k], dtype=float)
        edges = np.linspace(0, 1, 11)
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            msk = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
            if msk.sum() >= 5:
                xs.append(p[msk].mean())
                ys.append(t[msk].mean())
                ns.append(msk.sum())
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
        if xs:
            ax.scatter(xs, ys, s=np.clip(np.array(ns) * 1.5, 12, 140), color=SERIES[0], edgecolor="#ffffff", lw=0.8, zorder=3)
            ax.plot(xs, ys, color=SERIES[0], lw=1.5, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(k, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Noul P(yes)")
        ax.set_ylabel("Observed rate")
        ax.grid(color=GRID, lw=0.6)
        _style(ax)
    fig.suptitle("Noul calibration against code-defined targets (marker size = bin count)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
