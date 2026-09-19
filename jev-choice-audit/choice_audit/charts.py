"""Charts. Built from the extracted rows only; makes no API calls.

Colour roles are constant across every chart:
  blue   the option `choice` returned
  orange the highest-probability option, and the mismatch subset generally
  aqua   the correct answer from the calendar (always carries a visible label,
         since aqua sits below 3:1 on this surface)
  grey   everything else

Palette is the validated default categorical set, slots 1-3, which clears the
all-pairs CVD and normal-vision floors.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .extract import Row  # noqa: E402
from .stats import wilson  # noqa: E402

# 1920x1080 at 150 dpi
FIGSIZE = (12.8, 7.2)
DPI = 150

CHOICE = "#2a78d6"
MAXPROB = "#eb6834"
CORRECT = "#1baf7a"
GREY = "#b8b6ae"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

TITLE_FS = 25
LABEL_FS = 20
TICK_FS = 18
ANNOT_FS = 17
TITLE_WRAP = 56   # characters that fit across 12.8in at TITLE_FS bold
SUB_WRAP = 112    # characters that fit at ANNOT_FS regular


def _fig():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _style(ax, *, ygrid: bool = True) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=TICK_FS, length=0)
    ax.xaxis.label.set_size(LABEL_FS)
    ax.yaxis.label.set_size(LABEL_FS)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=1.0)
        ax.set_axisbelow(True)


def _title(ax, text: str, sub: str = "") -> None:
    """Wrap both lines to the figure width; an overflowing title is simply unreadable."""
    title = "\n".join(textwrap.wrap(text, TITLE_WRAP)) or text
    sub_lines = textwrap.wrap(sub, SUB_WRAP) if sub else []
    pad = 14 + 22 * len(sub_lines)
    ax.set_title(title, fontsize=TITLE_FS, color=INK, loc="left", pad=pad, fontweight="bold")
    for i, line in enumerate(sub_lines):
        y = 1.012 + 0.031 * (len(sub_lines) - 1 - i)
        ax.text(0, y, line, transform=ax.transAxes, fontsize=ANNOT_FS, color=MUTED, va="bottom")


def _bar_labels(ax, data: list[dict], rates: list[float], fmt: str = "{:.1%}", extra: list[str] | None = None) -> float:
    """One label per bar, placed clear of the error-bar cap. Returns the top y used."""
    top = 0.0
    for i, (d, r) in enumerate(zip(data, rates)):
        hi = d.get("ci_high", r)
        txt = fmt.format(r) + f"\n{d['k']} of {d['n']}"
        if extra:
            txt += f"\n{extra[i]}"
        y = max(hi, r) + 0.022
        ax.text(i, y, txt, ha="center", va="bottom", fontsize=ANNOT_FS + 1, color=INK, fontweight="bold",
                linespacing=1.35)
        top = max(top, y + 0.09)
    return top


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def _ci_err(ks: list[int], ns: list[int], rates: list[float]):
    los, his = [], []
    for k, n, r in zip(ks, ns, rates):
        lo, hi = wilson(k, n)
        los.append(max(0.0, r - lo))
        his.append(max(0.0, hi - r))
    return [los, his]


# ---------------------------------------------------------------- 1. the example
def chart_mismatch_example(row: Row, path: Path) -> str:
    opts = list(config.OPTIONS)
    vals = [row.probabilities.get(o, 0.0) for o in opts]
    colors = []
    for o, v in zip(opts, vals):
        if o == row.choice:
            colors.append(CHOICE)
        elif v >= row.p_max - config.TOL:
            colors.append(MAXPROB)
        else:
            colors.append(GREY)

    fig, ax = _fig()
    bars = ax.bar(opts, vals, color=colors, width=0.68, zorder=3)
    ymax = max(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=ANNOT_FS, color=INK)

    # Callouts are staggered in height so their text blocks cannot collide.
    callouts = []
    for i, o in enumerate(opts):
        tags = []
        col = INK
        if o == row.choice:
            tags.append("returned as `choice`")
            col = CHOICE
        if row.probabilities.get(o, 0.0) >= row.p_max - config.TOL:
            tags.append("highest probability")
            col = MAXPROB if o != row.choice else col
        if o == row.truth:
            tags.append("correct answer")
            col = CORRECT if len(tags) == 1 else col
        if tags:
            callouts.append((i, " + ".join(tags), col))
    levels = [1.50, 1.34, 1.18, 1.02][: len(callouts)] or [1.5]
    for (i, text, col), lv in zip(callouts, levels):
        ax.annotate(text, xy=(i, vals[i] + ymax * 0.10), xytext=(i, ymax * lv), ha="center",
                    fontsize=ANNOT_FS + 1, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, shrinkA=4, shrinkB=2))
    ax.set_ylim(0, ymax * 1.68)
    ax.set_ylabel("Reported probability")
    ax.tick_params(axis="x", labelsize=TICK_FS)
    for lab in ax.get_xticklabels():
        lab.set_rotation(18)
        lab.set_ha("right")
    _title(ax,
           f"`choice` took {row.p_choice:.2f} when {row.p_max:.2f} was available",
           f"{row.date} · one question in its own request · served model {row.served_model} · record `{row.record_id}`")
    _style(ax)
    return _save(fig, path)


# ---------------------------------------------------------------- 2. by gap bucket
def chart_mismatch_by_gap(results: dict[str, Any], path: Path) -> str:
    buckets = ["0 (exact tie)", "<=0.01", ">0.01"]
    labels = ["Top two identical\n(gap = 0)", "Near-tie\n(0 < gap \u2264 0.01)", "Clear winner\n(gap > 0.01)"]
    data = [results["gap_buckets_main"][b] for b in buckets]
    rates = [d["rate"] for d in data]
    fig, ax = _fig()
    ax.bar(labels, rates, color=MAXPROB, width=0.6, zorder=3)
    ax.errorbar(range(len(data)), rates, yerr=_ci_err([d["k"] for d in data], [d["n"] for d in data], rates),
                fmt="none", ecolor=INK, elinewidth=2.2, capsize=10, capthick=2.2, zorder=4)
    top = _bar_labels(ax, data, rates)
    ax.set_ylabel("Share where `choice` was not the highest")
    ax.set_ylim(0, max(0.5, top))
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    f = results["gap_buckets_main"]["fisher_near_vs_far"]
    _title(ax, "Mismatches occur only when the top two are within 0.01",
           f"n = {results['meta']['n_main']} single questions \u00b7 95% Wilson intervals \u00b7 "
           f"Fisher's exact, near-tie against not: p = {f['p_value']:.2g}")
    _style(ax)
    return _save(fig, path)


# ---------------------------------------------------------------- 3. accuracy
def chart_accuracy(results: dict[str, Any], path: Path) -> str:
    acc = results["accuracy_main"]
    labels = ["`choice`\n(what the API returned)", "argmax of probabilities\n(ties broken at random)"]
    data = [acc["choice"], acc["argmax_random_tiebreak"]]
    rates = [d["rate"] for d in data]
    fig, ax = _fig()
    ax.bar(labels, rates, color=[CHOICE, MAXPROB], width=0.5, zorder=3)
    ax.errorbar(range(2), rates, yerr=_ci_err([d["k"] for d in data], [d["n"] for d in data], rates),
                fmt="none", ecolor=INK, elinewidth=2.2, capsize=10, capthick=2.2, zorder=4)
    ax.axhline(acc["chance"], ls="--", lw=2.4, color=INK, zorder=2)
    ax.text(1.42, acc["chance"], f"chance = 1/7 = {acc['chance']:.1%}", va="bottom", ha="left",
            fontsize=ANNOT_FS + 1, color=INK, fontweight="bold",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=3.0))
    top = _bar_labels(ax, data, rates, extra=[f"p = {d['binom_p_vs_chance']:.2f} vs chance" for d in data])
    ax.set_ylabel("Correct day of the week")
    ax.set_ylim(0, max(top, acc["chance"] * 1.6))
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_xlim(-0.6, 2.35)
    _title(ax, "Neither selection rule beats chance on this task",
           f"n = {results['meta']['n_main']} dates \u00b7 95% Wilson intervals \u00b7 two-sided binomial test against 1/7")
    _style(ax)
    return _save(fig, path)


# ---------------------------------------------------------------- 4. gap histogram
def chart_gap_histogram(rows: list[Row], path: Path) -> str:
    main = [r for r in rows if r.experiment == "main"]
    gaps = sorted({round(r.top2_gap, 2) for r in main})
    totals = [sum(1 for r in main if abs(round(r.top2_gap, 2) - g) < 1e-9) for g in gaps]
    misses = [sum(1 for r in main if abs(round(r.top2_gap, 2) - g) < 1e-9 and r.mismatch) for g in gaps]
    x = range(len(gaps))
    fig, ax = _fig()
    ax.bar(x, totals, color=GREY, width=0.68, zorder=3, label="All questions")
    ax.bar(x, misses, color=MAXPROB, width=0.68, zorder=4, label="`choice` was not the highest")
    for i, (t, m) in enumerate(zip(totals, misses)):
        ax.text(i, t + 8, f"{t}", ha="center", va="bottom", fontsize=ANNOT_FS, color=MUTED)
        if m:
            ax.text(i, m + 8, f"{m}", ha="center", va="bottom", fontsize=ANNOT_FS, color=MAXPROB, fontweight="bold")
    ax.set_xticks(list(x), [f"{g:.2f}" for g in gaps])
    ax.set_xlabel("Gap between the top two reported probabilities")
    ax.set_ylabel("Questions")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.legend(frameon=False, fontsize=ANNOT_FS + 1, loc="upper right")
    n_mm = sum(misses)
    _title(ax, f"All {n_mm} mismatches sit at a gap of 0.01 or less",
           f"n = {len(main)} single questions · reported probabilities are quantised to 0.01, so the gap is too")
    _style(ax)
    return _save(fig, path)


# ---------------------------------------------------------------- 5. determinism
def chart_determinism(results: dict[str, Any], path: Path) -> str:
    d = results["determinism"]
    if not d.get("n_dates"):
        return ""
    labels = ["`choice`\nchanged", "Any probability\nchanged", "`confidence`\nchanged"]
    data = [d["choice_changed"], d["probabilities_changed"], d["confidence_changed"]]
    rates = [x["rate"] for x in data]
    fig, ax = _fig()
    ax.bar(labels, rates, color=[CHOICE, MAXPROB, GREY], width=0.55, zorder=3)
    ax.errorbar(range(3), rates, yerr=_ci_err([x["k"] for x in data], [x["n"] for x in data], rates),
                fmt="none", ecolor=INK, elinewidth=2.2, capsize=10, capthick=2.2, zorder=4)
    top = _bar_labels(ax, data, rates, fmt="{:.0%}")
    ax.set_ylabel("Share of dates that varied across repeats")
    ax.set_ylim(0, max(1.12, top))
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    _title(ax, "Identical requests are not deterministic in any field",
           f"{d['n_dates']} dates \u00d7 {d['repeats_per_date']:.0f} byte-identical repeats \u00b7 "
           f"largest swing in any one option {d['max_per_option_swing']:.2f}")
    _style(ax)
    return _save(fig, path)


# ---------------------------------------------------------------- 6. confidence
def chart_confidence(rows: list[Row], results: dict[str, Any], path: Path) -> str:
    from matplotlib.lines import Line2D

    main = [r for r in rows if r.experiment == "main"]
    groups = [
        ([r for r in main if not r.mismatch], GREY, "`choice` was the highest", -0.0009, 3),
        ([r for r in main if r.mismatch], MAXPROB, "`choice` was not the highest", 0.0009, 4),
    ]
    fig, ax = _fig()
    for subset, color, _label, dx, z in groups:
        counts: dict[tuple[float, float], int] = {}
        for r in subset:
            counts[(r.p_max, r.confidence)] = counts.get((r.p_max, r.confidence), 0) + 1
        if not counts:
            continue
        # Dodged left and right so the two groups sit side by side instead of nesting.
        ax.scatter([k[0] + dx for k in counts], [k[1] for k in counts],
                   s=[26 + 7 * v for v in counts.values()], color=color, alpha=0.9,
                   edgecolor=SURFACE, linewidth=1.5, zorder=z)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=13, markerfacecolor=c,
                      markeredgecolor=SURFACE, label=lab) for _, c, lab, _, _ in groups]
    ax.legend(handles=handles, frameon=False, fontsize=ANNOT_FS + 1, loc="upper left", borderpad=0.2)
    best = results["confidence_main"]["best_match"]
    cand = results["confidence_main"]["candidates"][best]
    ax.set_xlabel("Highest reported probability")
    ax.set_ylabel("Reported `confidence`")
    ax.margins(x=0.10, y=0.18)
    om = results["confidence_main"].get("on_mismatches", {})
    sub = (f"marker size \u221d number of questions, groups dodged left and right \u00b7 "
           f"closest closed form `{best}`, Pearson r = {cand['pearson']:.3f}")
    if om:
        sub += (f" \u00b7 on mismatches it matches the chosen option {om['matches_formula_on_p_choice']}/{om['n']} "
                f"against the maximum {om['matches_formula_on_p_max']}/{om['n']}")
    _title(ax, "Confidence is not a function of the top probability alone", sub)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=1.0)
    return _save(fig, path)


# ---------------------------------------------------------------- 7. batching
def chart_batching(results: dict[str, Any], path: Path) -> str:
    b = results["batching"]
    if not b.get("ran"):
        return ""
    labels = [f"Batched\n(up to {config.BATCH_SIZE} questions per request)", "Single\n(1 question per request)"]
    data = [b["batched"], b["single_same_dates"]]
    rates = [d["rate"] for d in data]
    fig, ax = _fig()
    ax.bar(labels, rates, color=MAXPROB, width=0.5, zorder=3)
    ax.errorbar(range(2), rates, yerr=_ci_err([d["k"] for d in data], [d["n"] for d in data], rates),
                fmt="none", ecolor=INK, elinewidth=2.2, capsize=10, capthick=2.2, zorder=4)
    top = _bar_labels(ax, data, rates)
    ax.set_ylabel("Share where `choice` was not the highest")
    ax.set_ylim(0, max(top, max(rates) * 1.8 if max(rates) else 0.1))
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    _title(ax, "Batching questions does not change the mismatch rate",
           f"the same {config.N_DETERMINISM} dates sent both ways \u00b7 95% Wilson intervals \u00b7 "
           f"Fisher's exact, two-sided: p = {b['fisher']['p_value']:.2g}")
    _style(ax)
    return _save(fig, path)


# ---------------------------------------------------------------- driver
def build_all(rows: list[Row], results: dict[str, Any], out_dir: Path) -> dict[str, str]:
    from .analyze import find_repro

    out_dir.mkdir(parents=True, exist_ok=True)
    main = [r for r in rows if r.experiment == "main"]
    charts: dict[str, str] = {}
    repro = find_repro(main) or find_repro(rows)
    if repro is not None:
        charts["mismatch_example"] = chart_mismatch_example(repro, out_dir / "01_mismatch_example.png")
    charts["mismatch_by_gap"] = chart_mismatch_by_gap(results, out_dir / "02_mismatch_by_gap.png")
    charts["accuracy"] = chart_accuracy(results, out_dir / "03_accuracy.png")
    charts["gap_histogram"] = chart_gap_histogram(rows, out_dir / "04_gap_histogram.png")
    d = chart_determinism(results, out_dir / "05_determinism.png")
    if d:
        charts["determinism"] = d
    charts["confidence"] = chart_confidence(rows, results, out_dir / "06_confidence.png")
    b = chart_batching(results, out_dir / "07_batched_vs_single.png")
    if b:
        charts["batching"] = b
    return charts
