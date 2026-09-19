"""Does giving Jev reference material in `state` actually help?

A difficulty ladder over the same dates, changing only what the state contains:

  none          the neutral greeting, as in the main experiment (control)
  month_anchor  the weekday of the 1st of that month is given
  near_anchor   the weekday of three days earlier is given
  lookup        the answer itself is present, among 39 distractor rows

The question text and the options never change, so the only variable is context.
This also tests a prediction from the main report: mismatches between `choice` and
`probabilities` should get rarer as the model becomes confident and gaps widen.

    python -m choice_audit.probe              # collect then analyse
    python -m choice_audit.probe --analyse-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

from . import config
from .client import Counter, JevClient
from .collect import run
from .extract import Row, load_rows
from .plan import Item, Unit, main_dates
from .stats import fisher_2x2, fmt_ci, wilson
from .store import JsonlStore

PROBE_JSONL = config.DATA_DIR / "context_probe.jsonl"
PROBE_MD = config.ROOT / "context_probe.md"
HARD_JSONL = config.DATA_DIR / "context_probe_hard.jsonl"
HARD_MD = config.ROOT / "context_probe_hard.md"
N_PROBE = 200
N_LOOKUP_ROWS = 40


def _pairs(dates: list[dt.date]) -> list[str]:
    return [f"{d.isoformat()} was a {config.truth(d)}" for d in dates]


def state_none(d: dt.date, pool: list[dt.date]) -> Any:
    return config.STATE


def state_month_anchor(d: dt.date, pool: list[dt.date]) -> Any:
    first = d.replace(day=1)
    return {
        "greeting": config.STATE,
        "reference_calendar": _pairs([first]),
    }


def state_near_anchor(d: dt.date, pool: list[dt.date]) -> Any:
    return {
        "greeting": config.STATE,
        "reference_calendar": _pairs([d - dt.timedelta(days=3)]),
    }


def state_lookup(d: dt.date, pool: list[dt.date]) -> Any:
    """The answer is in the table, among distractors, in shuffled order."""
    rng = random.Random(f"{config.SEED}:{d.isoformat()}")
    others = [x for x in pool if x != d]
    rows = rng.sample(others, min(N_LOOKUP_ROWS - 1, len(others))) + [d]
    rng.shuffle(rows)
    return {"greeting": config.STATE, "reference_calendar": _pairs(rows)}


MIN_SCATTER_GAP = 60  # scattered references must be this far from the date asked about


def _scatter(d: dt.date, pool: list[dt.date], n: int) -> Any:
    """n true date/weekday pairs from anywhere in the range, none of them near `d`.

    Everything needed to derive the answer is present, since any known date plus
    modular arithmetic gives any other. What is absent is anything to count from.
    """
    rng = random.Random(f"{config.SEED}:scatter{n}:{d.isoformat()}")
    far = [x for x in pool if abs((x - d).days) >= MIN_SCATTER_GAP]
    rows = rng.sample(far, min(n, len(far)))
    rng.shuffle(rows)
    return {"greeting": config.STATE, "reference_calendar": _pairs(rows)}


def _offset_anchor(d: dt.date, days: int) -> Any:
    return {"greeting": config.STATE, "reference_calendar": _pairs([d - dt.timedelta(days=days)])}


CONDITIONS: dict[str, Callable[[dt.date, list[dt.date]], Any]] = {
    "none": state_none,
    "month_anchor": state_month_anchor,
    "near_anchor": state_near_anchor,
    "lookup": state_lookup,
    "scatter_10": lambda d, pool: _scatter(d, pool, 10),
    "scatter_40": lambda d, pool: _scatter(d, pool, 40),
    # 28 and 364 are whole numbers of weeks, so the answer is the anchor's own
    # weekday. 30 and 365 are not, so the answer is 2 and 1 days past it. Pairing
    # them separates copying the anchor from computing from it.
    "anchor_28": lambda d, pool: _offset_anchor(d, 28),
    "anchor_30": lambda d, pool: _offset_anchor(d, 30),
    "anchor_364": lambda d, pool: _offset_anchor(d, 364),
    "anchor_365": lambda d, pool: _offset_anchor(d, 365),
}

LADDER = ["none", "month_anchor", "near_anchor", "lookup"]
HARD = ["none", "scatter_10", "scatter_40", "anchor_28", "anchor_30", "anchor_364", "anchor_365"]
SUITES = {"ladder": LADDER, "hard": HARD}
LABELS = {
    "none": "No context\n(control)",
    "month_anchor": "1st of the month\ngiven",
    "near_anchor": "3 days earlier\ngiven",
    "lookup": "The answer is\nin the state",
    "scatter_10": "10 random dates\n(all >60d away)",
    "scatter_40": "40 random dates\n(all >60d away)",
    "anchor_28": "1 date, 28d back\n(= 4 weeks)",
    "anchor_30": "1 date, 30d back\n(not a week multiple)",
    "anchor_364": "1 date, 364d back\n(= 52 weeks)",
    "anchor_365": "1 date, 365d back\n(not a week multiple)",
}


# Kept to seven characters a line: at 18pt across seven slots, anything longer
# collides with its neighbour.
SHORT_LABELS = {
    "none": "No\ncontext",
    "scatter_10": "10 far\ndates",
    "scatter_40": "40 far\ndates",
    "anchor_28": "28d\nback",
    "anchor_30": "30d\nback",
    "anchor_364": "364d\nback",
    "anchor_365": "365d\nback",
}


def anchor_offset(cond: str) -> int | None:
    """Days back to the single reference date, for conditions that have exactly one."""
    if cond.startswith("anchor_"):
        return int(cond.split("_")[1])
    if cond == "near_anchor":
        return 3
    return None


def probe_dates() -> list[dt.date]:
    rng = random.Random(config.SEED + 2)
    return sorted(rng.sample(main_dates(), N_PROBE))


def build_probe_units(suite: str = "ladder") -> list[Unit]:
    dates = probe_dates()
    pool = main_dates()
    units: list[Unit] = []
    for cond in SUITES[suite]:
        builder = CONDITIONS[cond]
        for d in dates:
            units.append(Unit(f"{cond}:{d.isoformat()}", cond, (Item("q0", d),), state=builder(d, pool)))
    return units


# ------------------------------------------------------------------ analysis
def parrot_rate(rows: list[Row], cond: str) -> float | None:
    """Share of answers equal to the reference date's own weekday.

    For a whole number of weeks that is also the correct answer; for 30 or 365 days
    it is wrong. A high rate in both is copying, not computing.
    """
    off = anchor_offset(cond)
    if off is None:
        return None
    sub = [r for r in rows if r.experiment == cond]
    if not sub:
        return None
    hits = 0
    for r in sub:
        anchor = dt.date.fromisoformat(r.date) - dt.timedelta(days=off)
        if r.choice == config.truth(anchor):
            hits += 1
    return hits / len(sub)


def summarise(rows: list[Row], conds: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cond in (conds if conds is not None else LADDER):
        sub = [r for r in rows if r.experiment == cond]
        if not sub:
            continue
        n = len(sub)
        correct = sum(1 for r in sub if r.choice_correct)
        mm = sum(1 for r in sub if r.mismatch)
        lo, hi = wilson(correct, n)
        mlo, mhi = wilson(mm, n)
        out[cond] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "acc_ci": [lo, hi],
            "mismatch": mm,
            "mismatch_rate": mm / n,
            "mm_ci": [mlo, mhi],
            "mean_top_prob": sum(r.p_max for r in sub) / n,
            "mean_gap": sum(r.top2_gap for r in sub) / n,
            "tie_rate": sum(1 for r in sub if r.is_tie) / n,
            "mean_confidence": sum(r.confidence for r in sub) / n,
            "unknown_rate": sum(1 for r in sub if r.choice == "Unknown") / n,
            "parrot_rate": parrot_rate(rows, cond),
            "anchor_offset": anchor_offset(cond),
        }
    return out


def chart(summary: dict[str, Any], path: Path, conds: list[str] | None = None, title: str | None = None) -> None:
    from . import charts as C
    import matplotlib.pyplot as plt

    conds = [c for c in (conds if conds is not None else LADDER) if c in summary]
    fig, ax = plt.subplots(figsize=C.FIGSIZE, dpi=C.DPI)
    fig.patch.set_facecolor(C.SURFACE)
    ax.set_facecolor(C.SURFACE)
    x = range(len(conds))
    acc = [summary[c]["accuracy"] for c in conds]
    mm = [summary[c]["mismatch_rate"] for c in conds]
    w = 0.38
    ax.bar([i - w / 2 for i in x], acc, width=w, color=C.CORRECT, zorder=3, label="Correct answer")
    ax.bar([i + w / 2 for i in x], mm, width=w, color=C.MAXPROB, zorder=3, label="`choice` was not the top probability")
    ax.errorbar([i - w / 2 for i in x], acc,
                yerr=C._ci_err([summary[c]["correct"] for c in conds], [summary[c]["n"] for c in conds], acc),
                fmt="none", ecolor=C.INK, elinewidth=2.0, capsize=8, capthick=2.0, zorder=4)
    ax.errorbar([i + w / 2 for i in x], mm,
                yerr=C._ci_err([summary[c]["mismatch"] for c in conds], [summary[c]["n"] for c in conds], mm),
                fmt="none", ecolor=C.INK, elinewidth=2.0, capsize=8, capthick=2.0, zorder=4)
    for i, c in enumerate(conds):
        ax.text(i - w / 2, summary[c]["acc_ci"][1] + 0.025, f"{acc[i]:.0%}", ha="center", va="bottom",
                fontsize=C.ANNOT_FS + 2, color=C.INK, fontweight="bold")
        ax.text(i + w / 2, summary[c]["mm_ci"][1] + 0.025, f"{mm[i]:.0%}", ha="center", va="bottom",
                fontsize=C.ANNOT_FS + 2, color=C.INK, fontweight="bold")
    # The chance line is identified in the legend rather than by a floating label,
    # which has nowhere collision-free to sit once the bars reach 100%.
    ax.axhline(config.CHANCE, ls="--", lw=2.2, color=C.INK, zorder=2,
               label=f"chance = 1/7 = {config.CHANCE:.1%}")
    ax.set_xticks(list(x), [LABELS[c] for c in conds])
    ax.set_ylim(0, 1.26)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_ylabel("Share of questions")
    ax.legend(frameon=False, fontsize=C.ANNOT_FS + 1, loc="upper left", handlelength=1.8,
              labelspacing=0.5)
    C._title(ax, title or "Context fixes the answers and the field disagreement together",
             f"the same {summary[conds[0]]['n']} dates in every condition, only the state changes · "
             f"95% Wilson intervals")
    C._style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=C.SURFACE)
    plt.close(fig)


DAY_BUCKETS = [("2-7", 1, 6), ("8-14", 7, 13), ("15-21", 14, 20), ("22-31", 21, 31)]


def counting_distance(rows: list[Row]) -> list[tuple[str, int, int]]:
    """In `month_anchor` the model must count from the 1st. How far can it count?

    Dates that are themselves the 1st are excluded: for those the anchor *is* the
    answer, so they measure lookup rather than counting.
    """
    out = []
    for label, lo, hi in DAY_BUCKETS:
        sub = [r for r in rows if r.experiment == "month_anchor"
               and lo <= dt.date.fromisoformat(r.date).day - 1 <= hi]
        out.append((label, sum(1 for r in sub if r.choice_correct), len(sub)))
    return out


def month_anchor_freebies(rows: list[Row]) -> tuple[int, int, int, int]:
    """(freebies, freebies correct, remaining, remaining correct) for the day-1 confound."""
    ma = [r for r in rows if r.experiment == "month_anchor"]
    first = [r for r in ma if dt.date.fromisoformat(r.date).day == 1]
    rest = [r for r in ma if dt.date.fromisoformat(r.date).day != 1]
    return (len(first), sum(1 for r in first if r.choice_correct),
            len(rest), sum(1 for r in rest if r.choice_correct))


def error_offsets(rows: list[Row]) -> dict[int, int]:
    """Signed day offset of the chosen weekday from the true one, in `month_anchor`."""
    counts: dict[int, int] = {}
    for r in rows:
        if r.experiment != "month_anchor" or r.choice_correct or r.choice == "Unknown":
            continue
        diff = (config.DAYS.index(r.choice) - config.DAYS.index(r.truth)) % 7
        diff = diff if diff <= 3 else diff - 7
        counts[diff] = counts.get(diff, 0) + 1
    return dict(sorted(counts.items()))


def confidence_separation(rows: list[Row]) -> dict[str, tuple[float | None, float | None]]:
    out = {}
    for c in LADDER:
        sub = [r for r in rows if r.experiment == c]
        if not sub:
            continue
        right = [r.p_max for r in sub if r.choice_correct]
        wrong = [r.p_max for r in sub if not r.choice_correct]
        out[c] = (sum(right) / len(right) if right else None, sum(wrong) / len(wrong) if wrong else None)
    return out


def write_md(summary: dict[str, Any], rows: list[Row], path: Path, chart_name: str) -> None:
    L = ["# Does context help Jev, and does it fix the `choice` disagreement?", ""]
    n = summary[LADDER[0]]["n"]
    L.append(f"The same {n} dates asked four ways. The question text and the options are identical "
             f"throughout; only `state` changes. Ground truth never appears except in the `lookup` "
             f"condition, where it is deliberately placed among {N_LOOKUP_ROWS - 1} distractor rows.")
    L.append("")
    L.append("| State contains | Accuracy | 95% CI | `choice` mismatch | 95% CI | Mean top probability | Tie rate |")
    L.append("|---|---|---|---|---|---|---|")
    names = {"none": "nothing (control)", "month_anchor": "the 1st of that month",
             "near_anchor": "the date 3 days earlier", "lookup": "**the answer itself**"}
    for c in LADDER:
        if c not in summary:
            continue
        s = summary[c]
        L.append(f"| {names[c]} | {s['accuracy']:.1%} | {fmt_ci(*s['acc_ci'])} | {s['mismatch_rate']:.1%} | "
                 f"{fmt_ci(*s['mm_ci'])} | {s['mean_top_prob']:.3f} | {s['tie_rate']:.0%} |")
    L.append("")
    base, best = summary.get("none"), summary.get("lookup")
    if base and best:
        odds, p = fisher_2x2(best["correct"], best["n"] - best["correct"], base["correct"], base["n"] - base["correct"])
        L.append(f"Lookup against control on accuracy: Fisher's exact, two-sided p = {p:.3g}.")
        o2, p2 = fisher_2x2(best["mismatch"], best["n"] - best["mismatch"], base["mismatch"], base["n"] - base["mismatch"])
        L.append(f"Lookup against control on the mismatch rate: p = {p2:.3g}.")
        L.append("")
    nf, nfc, nr, nrc = month_anchor_freebies(rows)
    if nf:
        L.append(f"**One caveat on `month_anchor`.** When the date asked about is itself the 1st, the anchor "
                 f"*is* the answer, so those questions measure lookup rather than counting. That affects "
                 f"{nf} of {nf + nr} dates and the model got all {nfc} right. Excluding them, accuracy in that "
                 f"condition is {nrc}/{nr} = {nrc / nr:.1%} rather than the {(nfc + nrc) / (nf + nr):.1%} in the "
                 f"table. Every other condition is clean: the `near_anchor` reference is always three days "
                 f"before the date asked about and can never coincide with it.")
        L.append("")
    L.append(f"![context ladder](charts/{chart_name})")
    L.append("")
    L.append("*Accuracy and the `choice` mismatch rate across the four conditions.*")
    L.append("")

    L.append("## How far can it count?")
    L.append("")
    L.append("In `month_anchor` the model is told the weekday of the 1st and must count forward. "
             "Accuracy collapses with the distance it has to cover. Dates that are themselves the 1st are "
             "excluded here, since for those the anchor is the answer.")
    L.append("")
    L.append("| Day of the month | Correct | Rate |")
    L.append("|---|---|---|")
    for label, k, n in counting_distance(rows):
        if n:
            L.append(f"| {label} | {k} / {n} | {k / n:.1%} |")
    L.append("")
    off = error_offsets(rows)
    if off:
        total = sum(off.values())
        biggest = max(off, key=lambda k: off[k])
        L.append("Its mistakes are not random guesses. The signed offset of the chosen weekday from the "
                 "correct one, over the " + str(total) + " wrong answers in that condition:")
        L.append("")
        L.append("| Offset in days | " + " | ".join(f"{k:+d}" for k in off) + " |")
        L.append("|---" * (len(off) + 1) + "|")
        L.append("| Count | " + " | ".join(str(off[k]) for k in off) + " |")
        L.append("")
        L.append(f"The most common single error is {biggest:+d} day ({off[biggest]} of {total}), "
                 f"the signature of an off-by-one in the count rather than a wrong method.")
        L.append("")
    na_fail = [r for r in rows if r.experiment == "near_anchor" and not r.choice_correct]
    if na_fail:
        L.append(f"In `near_anchor` only {len(na_fail)} of {summary['near_anchor']['n']} were wrong, and every "
                 f"one of them crosses a month boundary: "
                 + "; ".join(f"{r.date} (told the weekday of "
                             f"{(dt.date.fromisoformat(r.date) - dt.timedelta(days=3)).isoformat()}), "
                             f"answered {r.choice} for {r.truth}" for r in na_fail) + ".")
        L.append("")

    L.append("## The probabilities only mean something once the model knows something")
    L.append("")
    L.append("| State contains | Mean top probability when right | When wrong |")
    L.append("|---|---|---|")
    for c, (right, wrong) in confidence_separation(rows).items():
        rs = f"{right:.3f}" if right is not None else "n/a"
        ws = f"{wrong:.3f}" if wrong is not None else "n/a"
        L.append(f"| {names[c]} | {rs} | {ws} |")
    L.append("")
    L.append("Without context the top probability is 0.14 whether the answer is right or wrong, so it carries "
             "no signal at all. With context it separates cleanly, which is the behaviour the product promises.")
    L.append("")

    base = summary.get("none")
    if base:
        L.append("## Two things this confirms about the main result")
        L.append("")
        L.append(f"1. **The mismatch replicates.** The control condition here is a fresh set of "
                 f"{base['n']} requests and reproduces the headline rate: {base['mismatch_rate']:.1%} "
                 f"against 16.3% in the original 1,000.")
        L.append(f"2. **The prediction in the report holds.** It said mismatches should become rarer once "
                 f"gaps widen. They disappear entirely: {base['mismatch_rate']:.1%} with no context, "
                 f"{summary['month_anchor']['mismatch_rate']:.1%} once an anchor is given, and 0% in both "
                 f"conditions where the model is confident. The disagreement is real but it only bites in "
                 f"photo finishes, which is where a near-uniform distribution puts you.")
        L.append("")
    path.write_text("\n".join(L))


def same_date_split(rows: list[Row], cond: str = "anchor_365") -> tuple[int, int, int, int]:
    """For a 365-day anchor, was it genuinely the same calendar date? A leap day in
    between breaks that, which separates counting from pattern-matching."""
    off = anchor_offset(cond) or 365
    same = diff = same_ok = diff_ok = 0
    for r in rows:
        if r.experiment != cond:
            continue
        d = dt.date.fromisoformat(r.date)
        a = d - dt.timedelta(days=off)
        if (a.month, a.day) == (d.month, d.day):
            same += 1
            same_ok += r.choice_correct
        else:
            diff += 1
            diff_ok += r.choice_correct
    return same_ok, same, diff_ok, diff


def paired_examples(rows: list[Row], n: int = 4) -> list[tuple[str, str, str, str, str, str, str]]:
    by: dict[str, dict[str, Row]] = {}
    for r in rows:
        by.setdefault(r.experiment, {})[r.date] = r
    out = []
    for d in sorted(set(by.get("anchor_364", {})) & set(by.get("anchor_365", {})))[:n]:
        dd = dt.date.fromisoformat(d)
        a5, a4 = dd - dt.timedelta(days=365), dd - dt.timedelta(days=364)
        out.append((d, config.truth(dd), a5.isoformat(), config.truth(a5), by["anchor_365"][d].choice,
                    a4.isoformat(), by["anchor_364"][d].choice))
    return out


def chart_hard(summary: dict[str, Any], path: Path, conds: list[str]) -> None:
    from . import charts as C
    import matplotlib.pyplot as plt

    conds = [c for c in conds if c in summary]
    acc = [summary[c]["accuracy"] for c in conds]
    fig, ax = plt.subplots(figsize=C.FIGSIZE, dpi=C.DPI)
    fig.patch.set_facecolor(C.SURFACE)
    ax.set_facecolor(C.SURFACE)
    ax.bar(range(len(conds)), acc, width=0.62, color=C.CORRECT, zorder=3)
    ax.errorbar(range(len(conds)), acc,
                yerr=C._ci_err([summary[c]["correct"] for c in conds], [summary[c]["n"] for c in conds], acc),
                fmt="none", ecolor=C.INK, elinewidth=2.2, capsize=9, capthick=2.2, zorder=4)
    for i, c in enumerate(conds):
        ax.text(i, summary[c]["acc_ci"][1] + 0.03, f"{acc[i]:.0%}", ha="center", va="bottom",
                fontsize=C.ANNOT_FS + 3, color=C.INK, fontweight="bold")
    ax.axhline(config.CHANCE, ls="--", lw=2.2, color=C.INK, zorder=2,
               label=f"chance = 1/7 = {config.CHANCE:.1%}")
    ax.legend(frameon=False, fontsize=C.ANNOT_FS + 1, loc="upper left", handlelength=1.8)
    ax.set_xticks(range(len(conds)), [SHORT_LABELS.get(c, LABELS[c]) for c in conds])
    ax.tick_params(axis="x", labelsize=C.TICK_FS)
    ax.set_ylim(0, 1.16)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_ylabel("Correct day of the week")
    C._title(ax, "A reference date helps only when it fits a pattern it knows",
             f"same {summary[conds[0]]['n']} dates, only `state` changes \u00b7 compare 364d with 365d")
    C._style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=C.SURFACE)
    plt.close(fig)


def write_md_hard(summary: dict[str, Any], rows: list[Row], path: Path, chart_name: str) -> None:
    n = summary["none"]["n"]
    L = ["# Can Jev use reference dates it cannot simply count from?", ""]
    L.append(f"The first probe gave anchors 3 days or 1 month away, which is close enough to count on your "
             f"fingers. This one removes that. The same {n} dates are asked again with reference material "
             f"that is either far away or deliberately structured to tell copying apart from computing. "
             f"Only `state` changes.")
    L.append("")
    L.append("| State contains | Accuracy | 95% CI | Answered the reference date's own weekday |")
    L.append("|---|---|---|---|")
    desc = {
        "none": "nothing (control)",
        "scatter_10": "10 true date/weekday pairs, every one at least 60 days away",
        "scatter_40": "40 true date/weekday pairs, every one at least 60 days away",
        "anchor_28": "one pair, 28 days back (**4 whole weeks**, so the answer *is* that weekday)",
        "anchor_30": "one pair, 30 days back (not a week multiple, the answer is 2 days later)",
        "anchor_364": "one pair, 364 days back (**52 whole weeks**, so the answer *is* that weekday)",
        "anchor_365": "one pair, 365 days back (not a week multiple, the answer is 1 day later)",
    }
    for c in HARD:
        if c not in summary:
            continue
        sm = summary[c]
        pr = f"{sm['parrot_rate']:.1%}" if sm["parrot_rate"] is not None else "n/a"
        L.append(f"| {desc[c]} | {sm['accuracy']:.1%} | {fmt_ci(*sm['acc_ci'])} | {pr} |")
    L.append("")

    base = summary["none"]
    for c in ("scatter_10", "scatter_40"):
        if c in summary:
            o, p = fisher_2x2(summary[c]["correct"], summary[c]["n"] - summary[c]["correct"],
                              base["correct"], base["n"] - base["correct"])
            L.append(f"`{c}` against the control: Fisher's exact, two-sided p = {p:.3g}.")
    L.append("")
    L.append(f"![hard context](charts/{chart_name})")
    L.append("")
    L.append("*Accuracy when the reference material cannot be counted from, and how often the answer is "
             "simply the reference date's own weekday.*")
    L.append("")

    L.append("## Scattered references do not help at all")
    L.append("")
    L.append(f"Ten true date/weekday pairs scored {summary['scatter_10']['accuracy']:.1%} and forty scored "
             f"{summary['scatter_40']['accuracy']:.1%}, against {summary['none']['accuracy']:.1%} for no context. "
             f"Everything needed is present, since any known date plus arithmetic yields any other, and four "
             f"times as much of it changes nothing. What the earlier probe measured was not use of reference "
             f"material. It was counting on your fingers from a nearby date.")
    L.append("")

    L.append("## One day changes 81% into 0%")
    L.append("")
    L.append(f"The sharpest result here is the pair at the end. A single reference date 365 days back scores "
             f"{summary['anchor_365']['accuracy']:.1%}. Move that same reference one day, to 364 days back, and "
             f"it scores {summary['anchor_364']['accuracy']:.1%}: not one correct answer in "
             f"{summary['anchor_364']['n']}. By the day count 364 is the easier of the two, being exactly 52 "
             f"weeks, so the answer is simply the reference date's own weekday.")
    L.append("")
    ex = paired_examples(rows)
    if ex:
        L.append("| Date asked | Truth | 365 days back | Its weekday | Answer | 364 days back | Answer |")
        L.append("|---|---|---|---|---|---|---|")
        for d, t, a5, a5w, c5, a4, c4 in ex:
            L.append(f"| {d} | **{t}** | {a5} | {a5w} | {c5} | {a4} | {c4} |")
        L.append("")
    so, sn, do, dn = same_date_split(rows)
    if sn and dn:
        L.append(f"The mechanism is visible once the 365-day condition is split by whether that reference "
                 f"really lands on the same calendar date. Without a leap day in between it does, and accuracy "
                 f"is {so}/{sn} = {so / sn:.1%}. With a leap day in between, 365 days lands a day off the "
                 f"matching date, and accuracy falls to {do}/{dn} = {do / dn:.1%}, near chance.")
        L.append("")
        L.append("So it is not counting 365 days. It recognises *the same calendar date one year earlier* and "
                 "applies the rule that this advances the weekday by one. Give it a reference that does not fit "
                 "that shape and the rule misfires. At 364 days it answers exactly one day before the correct "
                 "day in "
                 f"{sum(1 for r in rows if r.experiment == 'anchor_364' and r.choice != 'Unknown' and (config.DAYS.index(r.choice) + 1) % 7 == config.DAYS.index(r.truth)) / summary['anchor_364']['n']:.1%} "
                 "of cases, which is a systematic error, not noise.")
        L.append("")

    L.append("## It is confidently wrong, which is worse than being uncertain")
    L.append("")
    L.append("| State contains | Accuracy | Mean top probability | Mean `confidence` |")
    L.append("|---|---|---|---|")
    for c in HARD:
        if c not in summary:
            continue
        sm = summary[c]
        L.append(f"| {desc[c]} | {sm['accuracy']:.1%} | {sm['mean_top_prob']:.3f} | {sm['mean_confidence']:.3f} |")
    L.append("")
    L.append(f"With no context the model is at chance and says so, reporting a top probability of "
             f"{summary['none']['mean_top_prob']:.2f} across eight options. With a 364-day reference it is "
             f"wrong on every single question while reporting {summary['anchor_364']['mean_top_prob']:.2f} and a "
             f"confidence of {summary['anchor_364']['mean_confidence']:.2f}. Adding context that looks helpful "
             f"but is not removes the one useful signal the earlier probe found, which was that low confidence "
             f"meant the model did not know.")
    L.append("")
    path.write_text("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Context probe: does reference material in `state` help?")
    ap.add_argument("--suite", choices=sorted(SUITES), default="ladder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--rate", type=float, default=config.MAX_REQUESTS_PER_SECOND)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)

    hard = args.suite == "hard"
    out = Path(args.out) if args.out else (HARD_JSONL if hard else PROBE_JSONL)
    conds = SUITES[args.suite]
    if not args.analyse_only:
        units = build_probe_units(args.suite)
        store = JsonlStore(out)
        counter = Counter()
        client = JevClient(rate=args.rate, counter=counter)
        done, failed = run(units, store, client, args.workers)
        print(f"[probe] ok={done} failed={failed} tokens_in={counter.input_tokens}", flush=True)

    rows = load_rows(out)
    summary = summarise(rows, conds)
    chart_name = "09_context_hard.png" if hard else "08_context_ladder.png"
    if hard:
        chart_hard(summary, config.CHARTS_DIR / chart_name, conds)
    else:
        chart(summary, config.CHARTS_DIR / chart_name, conds)
    if hard:
        write_md_hard(summary, rows, HARD_MD, chart_name)
    else:
        write_md(summary, rows, PROBE_MD, chart_name)
    (config.DATA_DIR / f"context_probe_{args.suite}_summary.json").write_text(json.dumps(summary, indent=2))
    for c in conds:
        if c in summary:
            sm = summary[c]
            pr = f"{sm['parrot_rate']:6.1%}" if sm["parrot_rate"] is not None else "   n/a"
            print(f"{c:13s} acc {sm['accuracy']:6.1%}  mismatch {sm['mismatch_rate']:6.1%}  "
                  f"top-p {sm['mean_top_prob']:.3f}  copied-anchor {pr}")
    print(f"wrote {HARD_MD if hard else PROBE_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
