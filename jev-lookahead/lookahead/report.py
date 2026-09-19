"""Charts and the write-up. Reads the recorded JSONL only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .maze import Maze, bfs, optimal_move  # noqa: E402

FIGSIZE, DPI = (12.8, 7.2), 150
CHOICE, BAIT, CORRECT = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED, GRID, SURFACE = "#b8b6ae", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TITLE_FS, LABEL_FS, TICK_FS, ANNOT_FS = 25, 20, 18, 17


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=TICK_FS, length=0)
    ax.xaxis.label.set_size(LABEL_FS)
    ax.yaxis.label.set_size(LABEL_FS)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)


def _title(ax, text: str, sub: str = "") -> None:
    import textwrap
    t = "\n".join(textwrap.wrap(text, 56))
    lines = textwrap.wrap(sub, 112) if sub else []
    ax.set_title(t, fontsize=TITLE_FS, color=INK, loc="left", pad=14 + 22 * len(lines), fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(0, 1.012 + 0.031 * (len(lines) - 1 - i), line, transform=ax.transAxes,
                fontsize=ANNOT_FS, color=MUTED, va="bottom")


def chart_fork(res: dict[str, Any], path: Path) -> None:
    f = res["fork"]
    labels = ["Jev", "Greedy policy\n(code)", "Random among\nthe legal moves"]
    vals = [f["correct_rate"], f["greedy_baseline"], f["legal_baseline"]]
    cols = [CHOICE, BAIT, GREY]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(labels, vals, color=cols, width=0.55, zorder=3)
    lo, hi = f["correct_ci"]
    ax.errorbar([0], [f["correct_rate"]], yerr=[[f["correct_rate"] - lo], [hi - f["correct_rate"]]],
                fmt="none", ecolor=INK, elinewidth=2.2, capsize=10, capthick=2.2, zorder=4)
    for i, v in enumerate(vals):
        extra = f"\n{f['correct']} of {f['n']}" if i == 0 else ""
        ax.text(i, (hi if i == 0 else v) + 0.03, f"{v:.0%}{extra}", ha="center", va="bottom",
                fontsize=ANNOT_FS + 3, color=INK, fontweight="bold", linespacing=1.35)
    ax.set_ylabel("Chose the branch reaching G")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    _title(ax, "It does not guess at the fork, it takes the bait",
           f"{f['n']} mazes, one question each · it chose the dead end {f['bait_rate']:.0%} of the time")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_mazes(mazes: list[Maze], res: dict[str, Any], path: Path, k: int = 3) -> None:
    paths = res.get("_paths", {})
    picks = [i for i in sorted(paths, key=lambda x: int(x))][:k]
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    axes = [axes] if len(picks) == 1 else list(axes)
    for ax, key in zip(axes, picks):
        m = mazes[int(key)]
        jev = [tuple(p) for p in paths[key]]
        for r in range(m.rows):
            for c in range(m.cols):
                if m.grid[r][c] == "#":
                    ax.add_patch(Rectangle((c, -r), 1, -1, facecolor="#3a3a38", edgecolor=SURFACE, lw=1))
        best, at = [m.start], m.start
        for _ in range(200):
            if at == m.goal:
                break
            mv = sorted(optimal_move(m, at))
            if not mv:
                break
            at = m.legal(at)[mv[0]]
            best.append(at)
        ax.plot([c + 0.5 for _, c in best], [-r - 0.5 for r, _ in best], color=MUTED, lw=6,
                ls=(0, (1, 1.6)), solid_capstyle="round", zorder=3)
        ax.plot([c + 0.5 for _, c in jev], [-r - 0.5 for r, _ in jev], color=CHOICE, lw=4,
                solid_capstyle="round", zorder=4)
        ax.scatter([m.start[1] + 0.5], [-m.start[0] - 0.5], s=170, color=CHOICE, zorder=5)
        ax.scatter([m.goal[1] + 0.5], [-m.goal[0] - 0.5], s=230, marker="*", color=CORRECT, zorder=5)
        ax.add_patch(Rectangle((m.fork[1], -m.fork[0]), 1, -1, facecolor="none",
                               edgecolor=BAIT, lw=3.5, zorder=6))
        reached = jev[-1] == m.goal
        ax.set_title(("reached the goal" if reached else "never got there")
                     + f"\n{len(jev) - 1} moves, optimal {bfs(m, m.start)[m.goal]}",
                     fontsize=ANNOT_FS + 1, color=INK, fontweight="bold")
        ax.set_xlim(0, m.cols)
        ax.set_ylim(-m.rows, 0)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Three mazes: the route Jev actually walked", fontsize=TITLE_FS, color=INK,
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.90, "blue is the route it walked, dotted grey is the shortest route, "
             "the orange square is the fork, the star is the goal", fontsize=ANNOT_FS, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_all(mazes: list[Maze], res: dict[str, Any], report: Path, charts: Path) -> None:
    charts.mkdir(parents=True, exist_ok=True)
    chart_fork(res, charts / "01_fork.png")
    chart_mazes(mazes, res, charts / "02_mazes.png")
    f, w = res["fork"], res["walk"]
    L = ["# Does Jev look further than one move ahead?", ""]
    L.append("The planner experiment in the sibling folder showed Jev choosing its next question "
             "near-optimally, but the baseline it matched was *greedy*, so nothing there ruled out a purely "
             "myopic policy. These mazes are built so that greedy is provably wrong.")
    L.append("")
    L.append("Every maze has one fork. One branch points straight at the goal and dead-ends a few cells "
             "later; the other starts by moving away from the goal and is the only route there. A policy "
             "that minimises distance-to-goal takes the bait every time and never arrives.")
    L.append("")
    L.append("```")
    m = mazes[0]
    L += m.render(m.fork)
    L.append("```")
    L.append("")
    L.append(f"The agent is at `@`, the goal at `G`. East leads two cells and stops. West, then north, "
             f"then east along the top is the only way, and its first step increases the distance to the "
             f"goal. Walls are given explicitly in the state as `blocked_by_wall`, so reading the picture "
             f"is not the thing being tested.")
    L.append("")
    L.append("## The fork, asked once per maze")
    L.append("")
    L.append("| Policy | Chose the branch that reaches the goal | 95% CI |")
    L.append("|---|---|---|")
    L.append(f"| **Jev** | **{f['correct']} / {f['n']} = {f['correct_rate']:.1%}** | "
             f"{f['correct_ci'][0]:.1%}-{f['correct_ci'][1]:.1%} |")
    L.append("| Greedy, in code | 0% by construction | |")
    L.append(f"| Random among the legal moves | {f['legal_baseline']:.1%} | |")
    L.append("")
    L.append(f"Every fork has exactly {f['mean_legal_moves']:.0f} legal moves, so chance is "
             f"{f['legal_baseline']:.1%}. At {f['correct_rate']:.1%} it is not guessing. It chose the dead "
             f"end in {f['took_the_bait']} of {f['n']} cases ({f['bait_rate']:.1%}), which is a greedy "
             f"policy with almost no noise.")
    L.append("")
    L.append(f"It never once moved into a wall, in {f['n']} fork decisions and "
             f"{w['total_decisions']} decisions overall. Legality was given in the state as a "
             f"`blocked_by_wall` flag and it used it perfectly. The failure is not carelessness.")
    L.append("")
    if f["top_p_when_right"] is not None and f["top_p_when_wrong"] is not None:
        L.append(f"It was also more confident when it was wrong: mean top probability "
                 f"{f['top_p_when_wrong']:.3f} on the bait against {f['top_p_when_right']:.3f} on the "
                 f"correct branch.")
        L.append("")
    L.append("![fork](charts/01_fork.png)")
    L.append("")
    L.append("*The fork decision against the two code-side baselines.*")
    L.append("")
    if w["n"]:
        fv, lv = w["fork_first_visit"], w["fork_later_visits"]
        L.append("## Does a record of the dead end help?")
        L.append("")
        L.append("The second phase lets it navigate from the start, one request per step, capped at "
                 f"{30} steps, on {w['n']} of the same mazes. This time the state carries the trail: which "
                 "cells it has already stood on, and an `already_visited` flag on each move. A dead end it "
                 "has walked into is therefore visible in the state, and looping becomes a choice rather "
                 "than an inevitability.")
        L.append("")
        L.append("| At the fork | Correct | Rate |")
        L.append("|---|---|---|")
        L.append(f"| First visit, dead end not yet explored | {fv['k']} / {fv['n']} | {fv['rate']:.1%} |")
        L.append(f"| Later visits, dead end already in the trail | {lv['k']} / {lv['n']} | {lv['rate']:.1%} |")
        L.append(f"| Random among the legal moves | | {f['legal_baseline']:.1%} |")
        L.append("")
        L.append("This is the most interesting number here, and it is smaller than it first looks. "
                 f"Having walked the dead end takes it from {fv['rate']:.1%} to {lv['rate']:.1%}, roughly a "
                 f"fivefold improvement, but {lv['rate']:.1%} is indistinguishable from the "
                 f"{f['legal_baseline']:.1%} you get by picking at random among the legal moves. The memory "
                 "does not teach it the route. It cancels the pull of the bait and leaves it guessing.")
        L.append("")
        L.append("Over a whole maze that is still worth something:")
        L.append("")
        L.append("| | Value |")
        L.append("|---|---|")
        L.append(f"| Reached the goal | {w['reached']} / {w['n']} = {w['reached_rate']:.1%} |")
        L.append("| Greedy policy, in code | 0% |")
        if w["mean_steps_when_reached"]:
            L.append(f"| Steps taken when it arrived | {w['mean_steps_when_reached']:.1f} |")
            L.append(f"| Shortest possible | {w['mean_optimal_steps']:.1f} |")
        L.append(f"| Moves into a wall | {w['illegal_attempts']} of {w['total_decisions']} |")
        L.append("")
        L.append("![mazes](charts/02_mazes.png)")
        L.append("")
        L.append("*Three mazes with the route it walked against the shortest route.*")
        L.append("")
    L.append("## What this settles")
    L.append("")
    L.append("The planner experiment in the sibling folder left an open question: it matched a greedy "
             "baseline, so was it planning or just being greedy? This answers it. **Just greedy.** Where "
             "greedy and correct come apart, it follows greedy off a cliff, at a rate well below chance.")
    L.append("")
    L.append("That reframes the planner result rather than overturning it. In twenty questions the greedy "
             "move usually *is* the good move, so a myopic policy looks like a planner. The skill on "
             "display there was judging one step well, which it does; the lookahead was never being tested.")
    L.append("")
    L.append("Two things it does do well. It obeys stated constraints exactly, never entering a wall in "
             f"{w['total_decisions']} decisions. And it reacts to state you put in front of it: an explicit "
             "record of failure is enough to break the loop, even though it cannot derive the same "
             "conclusion by looking ahead.")
    L.append("")
    L.append("The practical reading is the same one the chain experiment reached from the other side. Hold "
             "the plan in code, hold the memory in code, and use the model for the one-step judgment. Here "
             "a four-line breadth-first search does the whole task perfectly, and no amount of asking helps.")
    L.append("")
    report.write_text("\n".join(L))
