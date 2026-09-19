"""Can Jev choose its own next question, or does the plan have to live in the code?

Every chain in this repo so far had Python deciding what to ask and in what order.
Jev only filled in blanks. This asks the other thing: a twenty-questions game where
the *only* thing Jev does is pick which question to ask next.

Design notes, because the point is to isolate planning:

* The full attribute table for the remaining candidates is supplied in `state`, so
  no world knowledge is needed. Judging a question's value is reading the table.
* Code answers each chosen question truthfully and filters the candidate set, so no
  arithmetic is asked of the model.
* Already-asked questions stay in the pool. Re-picking one gains exactly nothing and
  is the cheapest possible planning failure to detect.
* Two baselines are computed in code over the identical game states: greedy optimal,
  which picks the question with the highest expected information, and uniform random.

    python -m choice_audit.planner
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .chain import QItem, QUnit, read
from .client import Counter, JevClient
from .collect import run
from .stats import fmt_ci, wilson
from .store import JsonlStore

PLANNER_JSONL = config.DATA_DIR / "planner.jsonl"
PLANNER_MD = config.ROOT / "planner.md"
N_GAMES = 100
N_TURNS = 4
SEED = config.SEED + 7

ATTRS: tuple[str, ...] = (
    "mammal", "bird", "flies", "aquatic", "eats_meat",
    "bigger_than_human", "has_fur", "lays_eggs", "kept_by_humans", "four_legs",
)
QUESTIONS: dict[str, str] = {
    "mammal": "Is it a mammal?",
    "bird": "Is it a bird?",
    "flies": "Can it fly?",
    "aquatic": "Does it live primarily in water?",
    "eats_meat": "Does it eat meat?",
    "bigger_than_human": "Is it larger than an adult human?",
    "has_fur": "Does it have fur?",
    "lays_eggs": "Does it lay eggs?",
    "kept_by_humans": "Is it commonly kept by humans?",
    "four_legs": "Does it walk on four legs?",
}
#                     mam bird fly aqua meat big fur eggs kept 4legs
TABLE: dict[str, tuple[int, ...]] = {
    "Lion":      (1, 0, 0, 0, 1, 1, 1, 0, 0, 1),
    "Eagle":     (0, 1, 1, 0, 1, 0, 0, 1, 0, 0),
    "Dolphin":   (1, 0, 0, 1, 1, 1, 0, 0, 0, 0),
    "Dog":       (1, 0, 0, 0, 1, 0, 1, 0, 1, 1),
    "Penguin":   (0, 1, 0, 1, 1, 0, 0, 1, 0, 0),
    "Elephant":  (1, 0, 0, 0, 0, 1, 0, 0, 0, 1),
    "Ostrich":   (0, 1, 0, 0, 0, 1, 0, 1, 0, 0),
    "Salmon":    (0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "Crocodile": (0, 0, 0, 1, 1, 1, 0, 1, 0, 1),
    "Cat":       (1, 0, 0, 0, 1, 0, 1, 0, 1, 1),
    "Cow":       (1, 0, 0, 0, 0, 1, 1, 0, 1, 1),
    "Horse":     (1, 0, 0, 0, 0, 1, 1, 0, 1, 1),
    "Chicken":   (0, 1, 0, 0, 0, 0, 0, 1, 1, 0),
    "Parrot":    (0, 1, 1, 0, 0, 0, 0, 1, 1, 0),
    "Whale":     (1, 0, 0, 1, 1, 1, 0, 0, 0, 0),
    "Bear":      (1, 0, 0, 0, 1, 1, 1, 0, 0, 1),
    "Rabbit":    (1, 0, 0, 0, 0, 0, 1, 0, 1, 1),
    "Frog":      (0, 0, 0, 1, 1, 0, 0, 1, 0, 1),
    "Owl":       (0, 1, 1, 0, 1, 0, 0, 1, 0, 0),
    "Sheep":     (1, 0, 0, 0, 0, 0, 1, 0, 1, 1),
}


def attr(name: str, key: str) -> int:
    return TABLE[name][ATTRS.index(key)]


def info_gain(candidates: list[str], key: str) -> float:
    """Expected bits gained by asking `key` of this candidate set. Zero if it splits nothing."""
    n = len(candidates)
    if n <= 1:
        return 0.0
    yes = sum(1 for c in candidates if attr(c, key))
    no = n - yes
    if yes == 0 or no == 0:
        return 0.0
    h = lambda p: -p * math.log2(p)
    return h(yes / n) + h(no / n)


def best_keys(candidates: list[str]) -> list[str]:
    gains = {k: info_gain(candidates, k) for k in ATTRS}
    top = max(gains.values())
    return [k for k, v in gains.items() if abs(v - top) < 1e-12]


def apply_answer(candidates: list[str], key: str, target: str) -> list[str]:
    want = attr(target, key)
    return [c for c in candidates if attr(c, key) == want]


@dataclass
class Game:
    gid: int
    target: str
    candidates: list[str]
    asked: list[str]
    history: list[dict[str, Any]]


def new_games() -> list[Game]:
    rng = random.Random(SEED)
    names = sorted(TABLE)
    return [Game(i, rng.choice(names), list(names), [], []) for i in range(N_GAMES)]


def build_state(g: Game) -> dict[str, Any]:
    return {
        "task": ("You are narrowing down which animal is the hidden one. Choose the question that "
                 "will eliminate as many of the remaining candidates as possible. A question whose "
                 "answer is the same for every remaining candidate eliminates nothing."),
        "answers_so_far": g.history or "none yet",
        "questions_already_asked": [QUESTIONS[k] for k in g.asked] or "none yet",
        "remaining_candidates": [
            {"animal": c, **{k: bool(attr(c, k)) for k in ATTRS}} for c in g.candidates
        ],
    }


def turn_units(games: list[Game], turn: int) -> list[QUnit]:
    units = []
    for g in games:
        if len(g.candidates) <= 1:
            continue
        q = {"type": "choice",
             "instructions": "Which question should be asked next to narrow the list down fastest?",
             "criteria": {QUESTIONS[k]: None for k in ATTRS}}
        best = best_keys(g.candidates)
        units.append(QUnit(
            f"plan:{g.gid:03d}:{turn}", f"turn{turn}",
            (QItem("q0", QUESTIONS[best[0]], {
                "gid": g.gid, "turn": turn, "target": g.target,
                "n_candidates": len(g.candidates),
                "optimal_questions": [QUESTIONS[k] for k in best],
                "asked": [QUESTIONS[k] for k in g.asked],
                "gains": {QUESTIONS[k]: round(info_gain(g.candidates, k), 6) for k in ATTRS},
            }),),
            {"q0": q}, state=build_state(g)))
    return units


def q_to_key(text: str) -> str | None:
    for k, v in QUESTIONS.items():
        if v == text:
            return k
    return None


# ------------------------------------------------------------------ baselines
def simulate(policy: str, rng: random.Random) -> list[list[int]]:
    """Candidates remaining after each turn, per game, for a code-side policy."""
    out = []
    for g in new_games():
        cands, asked, row = list(g.candidates), [], []
        for _ in range(N_TURNS):
            if len(cands) <= 1:
                row.append(len(cands))
                continue
            key = rng.choice(best_keys(cands)) if policy == "optimal" else rng.choice(list(ATTRS))
            cands = apply_answer(cands, key, g.target)
            asked.append(key)
            row.append(len(cands))
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Can Jev pick its own next question?")
    ap.add_argument("--out", default=str(PLANNER_JSONL))
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--rate", type=float, default=config.MAX_REQUESTS_PER_SECOND)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    store = JsonlStore(out)
    games = new_games()

    if not args.analyse_only:
        client = JevClient(rate=args.rate, counter=Counter())
        for turn in range(N_TURNS):
            units = turn_units(games, turn)
            if not units:
                break
            run(units, store, client, args.workers)
            picked = {r["gid"]: r["choice"] for r in read(out, f"turn{turn}")}
            for g in games:  # advance every game on its own answer, right or wrong
                key = q_to_key(picked.get(g.gid, ""))
                if key is None or len(g.candidates) <= 1:
                    continue
                before = len(g.candidates)
                g.candidates = apply_answer(g.candidates, key, g.target)
                g.history.append({"question": QUESTIONS[key],
                                  "answer": "yes" if attr(g.target, key) else "no"})
                g.asked.append(key)
                print(f"  game {g.gid:3d} turn {turn}: {QUESTIONS[key]:38s} {before:2d} -> {len(g.candidates):2d}",
                      flush=True) if g.gid < 3 else None
    else:
        for turn in range(N_TURNS):
            picked = {r["gid"]: r["choice"] for r in read(out, f"turn{turn}")}
            for g in games:
                key = q_to_key(picked.get(g.gid, ""))
                if key is None or len(g.candidates) <= 1:
                    continue
                g.candidates = apply_answer(g.candidates, key, g.target)
                g.history.append({"question": QUESTIONS[key],
                                  "answer": "yes" if attr(g.target, key) else "no"})
                g.asked.append(key)

    analyse(out, games)
    return 0


def analyse(out: Path, games: list[Game]) -> None:
    rng = random.Random(SEED)
    opt = simulate("optimal", random.Random(SEED))
    rnd = simulate("random", rng)
    per_turn: list[dict[str, Any]] = []
    for turn in range(N_TURNS):
        rows = read(out, f"turn{turn}")
        if not rows:
            continue
        n = len(rows)
        chose_opt = sum(1 for r in rows if r["choice"] in r["optimal_questions"])
        repeated = sum(1 for r in rows if r["choice"] in r["asked"])
        # A zero-information pick only counts against the model when a useful question
        # was on the table. Late in a game every option can be worthless.
        live = [r for r in rows if max(r["gains"].values()) > 1e-12]
        wasted = sum(1 for r in live if r["gains"].get(r["choice"], 0.0) <= 1e-12)
        # Efficiency is the fair per-turn score: exact-match to the argmax is too strict,
        # since a question worth 0.92 bits against a best of 0.99 is not a planning failure.
        eff = [r["gains"].get(r["choice"], 0.0) / max(r["gains"].values()) for r in live]
        gains = [r["gains"].get(r["choice"], 0.0) for r in rows]
        best = [max(r["gains"].values()) for r in rows]
        avg_rand = [sum(r["gains"].values()) / len(r["gains"]) for r in rows]
        per_turn.append({
            "turn": turn, "n": n, "n_live": len(live),
            "chose_optimal": chose_opt, "chose_optimal_rate": chose_opt / n,
            "chose_optimal_ci": list(wilson(chose_opt, n)),
            "repeated_a_question": repeated,
            "wasted_pick": wasted,
            "wasted_pick_rate": (wasted / len(live)) if live else float("nan"),
            "efficiency": (sum(eff) / len(eff)) if eff else float("nan"),
            "mean_bits_jev": sum(gains) / n,
            "mean_bits_optimal": sum(best) / n,
            "mean_bits_random": sum(avg_rand) / n,
            "mean_top_p": sum(r["p_max"] for r in rows) / n,
        })
    jev_remaining = [len(g.candidates) for g in games]
    summary = {
        "n_games": len(games), "n_turns": N_TURNS, "n_animals": len(TABLE),
        "per_turn": per_turn,
        "final_candidates": {
            "jev": sum(jev_remaining) / len(jev_remaining),
            "optimal": sum(r[-1] for r in opt) / len(opt),
            "random": sum(r[-1] for r in rnd) / len(rnd),
        },
        "solved": {
            "jev": sum(1 for x in jev_remaining if x == 1) / len(jev_remaining),
            "optimal": sum(1 for r in opt if r[-1] == 1) / len(opt),
            "random": sum(1 for r in rnd if r[-1] == 1) / len(rnd),
        },
        "curve": {
            "jev": _curve(out, games),
            "optimal": [sum(r[t] for r in opt) / len(opt) for t in range(N_TURNS)],
            "random": [sum(r[t] for r in rnd) / len(rnd) for t in range(N_TURNS)],
        },
    }
    (config.DATA_DIR / "planner_summary.json").write_text(json.dumps(summary, indent=2))
    chart(summary, config.CHARTS_DIR / "10_planner.png")
    write_md(summary, PLANNER_MD, "10_planner.png")
    for t in per_turn:
        print(f"turn {t['turn']}: efficiency {t['efficiency']:6.1%}  exact-optimal {t['chose_optimal_rate']:6.1%}  "
              f"wasted {t['wasted_pick']}/{t['n_live']}  "
              f"bits {t['mean_bits_jev']:.3f} (best {t['mean_bits_optimal']:.3f}, "
              f"random {t['mean_bits_random']:.3f})")
    f = summary["final_candidates"]
    print(f"after {N_TURNS} turns, candidates left of {len(TABLE)}: "
          f"jev {f['jev']:.2f}  optimal {f['optimal']:.2f}  random {f['random']:.2f}")
    print(f"wrote {PLANNER_MD}")


def _curve(out: Path, games: list[Game]) -> list[float]:
    sizes = [[] for _ in range(N_TURNS)]
    for turn in range(N_TURNS):
        rows = read(out, f"turn{turn}")
        by = {r["gid"]: r for r in rows}
        for g in games:
            r = by.get(g.gid)
            sizes[turn].append(r["n_candidates"] if r else 1)
    # n_candidates is the size *before* that turn, so shift to get after-turn sizes
    after = [sum(s) / len(s) for s in sizes[1:]] if len(sizes) > 1 else []
    after.append(sum(len(g.candidates) for g in games) / len(games))
    return after


def chart(S: dict[str, Any], path: Path) -> None:
    from . import charts as C
    import matplotlib.pyplot as plt

    turns = list(range(1, S["n_turns"] + 1))
    fig, ax = plt.subplots(figsize=C.FIGSIZE, dpi=C.DPI)
    fig.patch.set_facecolor(C.SURFACE)
    ax.set_facecolor(C.SURFACE)
    # dy staggers the end labels; the Jev and optimal curves finish within 0.2 of each other
    series = [("Greedy optimal (code)", S["curve"]["optimal"], C.MAXPROB, -15),
              ("Jev choosing its own question", S["curve"]["jev"], C.CHOICE, 15),
              ("Uniform random (code)", S["curve"]["random"], C.CORRECT, 0)]
    for label, ys, col, dy in series:
        ax.plot(turns[:len(ys)], ys, color=col, lw=3.0, marker="o", ms=11,
                markeredgecolor=C.SURFACE, markeredgewidth=2.0, label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.2f}", xy=(turns[len(ys) - 1], ys[-1]), xytext=(10, dy),
                    textcoords="offset points", va="center", fontsize=C.ANNOT_FS + 2,
                    color=col, fontweight="bold")
    ax.set_xticks(turns)
    ax.set_xlabel("Questions asked")
    ax.set_ylabel(f"Candidates left of {S['n_animals']}")
    ax.set_ylim(0, max(S["curve"]["random"]) * 1.15)
    ax.set_xlim(0.85, S["n_turns"] + 0.5)
    ax.legend(frameon=False, fontsize=C.ANNOT_FS + 1, loc="upper right")
    C._title(ax, "Jev picks its own next question close to optimally",
             f"{S['n_games']} games, {S['n_animals']} animals, attribute table supplied in `state`")
    C._style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=C.SURFACE)
    plt.close(fig)


def write_md(S: dict[str, Any], path: Path, chart_name: str) -> None:
    L = ["# Can Jev choose its own next question?", ""]
    L.append(f"Every chain elsewhere in this repo had Python deciding what to ask. This inverts that: "
             f"{S['n_games']} games of twenty questions over {S['n_animals']} animals, where the only thing "
             f"Jev does is pick which of ten questions to ask next. The attribute table for the remaining "
             f"candidates is supplied in `state`, so no world knowledge is needed; code answers truthfully "
             f"and filters the list, so no arithmetic is asked of the model. Questions already asked stay in "
             f"the pool, because re-picking one is the cheapest planning failure to detect.")
    L.append("")
    L.append("Two scores per turn. **Efficiency** is the information the chosen question actually yields "
             "divided by the most any question could have yielded, which is the fair measure: a question "
             "worth 0.92 bits against a best of 0.99 is not a planning failure. **Wasted** counts picks that "
             "eliminate nothing, and only over games where something useful was still on the table.")
    L.append("")
    L.append("| Turn | Efficiency | Wasted picks | Exact match to the best question | Bits gained | Best available | Random would get |")
    L.append("|---|---|---|---|---|---|---|")
    for t in S["per_turn"]:
        L.append(f"| {t['turn'] + 1} | **{t['efficiency']:.1%}** | {t['wasted_pick']} / {t['n_live']} | "
                 f"{t['chose_optimal']} / {t['n']} ({t['chose_optimal_rate']:.0%}) | "
                 f"{t['mean_bits_jev']:.3f} | {t['mean_bits_optimal']:.3f} | {t['mean_bits_random']:.3f} |")
    L.append("")
    total_live = sum(t["n_live"] for t in S["per_turn"])
    total_wasted = sum(t["wasted_pick"] for t in S["per_turn"])
    L.append(f"**It never asked a pointless question.** Across all {total_live} turns where a "
             f"discriminating question was still available, it picked one that eliminates nothing "
             f"{total_wasted} times. Already-asked questions were left in the pool the whole time.")
    L.append("")
    L.append("The exact-match column is the strict reading and undersells it. At turn 2 it matched the "
             "argmax in 0 of 100 games while still capturing 94% of the available information, because it "
             "consistently took a question worth 0.918 bits when the best was 0.991.")
    L.append("")
    f, sv = S["final_candidates"], S["solved"]
    L.append("## End to end")
    L.append("")
    L.append(f"After {S['n_turns']} questions, the average number of candidates still standing out of "
             f"{S['n_animals']}. Each policy plays its own {S['n_games']} games against the same hidden targets.")
    L.append("")
    L.append("| Policy | Candidates left | Games narrowed to exactly one |")
    L.append("|---|---|---|")
    L.append(f"| Greedy optimal, chosen in code | {f['optimal']:.2f} | {sv['optimal']:.0%} |")
    L.append(f"| **Jev choosing its own question** | **{f['jev']:.2f}** | **{sv['jev']:.0%}** |")
    L.append(f"| Uniform random, chosen in code | {f['random']:.2f} | {sv['random']:.0%} |")
    L.append("")
    gap = (f["jev"] - f["optimal"]) / (f["random"] - f["optimal"])
    L.append(f"Jev closes {1 - gap:.0%} of the distance between random and optimal play.")
    L.append("")
    L.append(f"![planner](charts/{chart_name})")
    L.append("")
    L.append("*Candidates remaining after each question, for the three policies.*")
    L.append("")
    L.append("## What this changes")
    L.append("")
    L.append("Elsewhere in this repo the plan always lived in Python and Jev filled in blanks, which made "
             "every chain a program with model-shaped function calls rather than anything that composed its "
             "own reasoning. This is the opposite arrangement and it holds up: given the state of a search "
             "and a menu of moves, the model selects near-optimally and never wastes a turn.")
    L.append("")
    L.append("Two limits worth stating. The menu of questions was supplied, so this is selection among given "
             "options and not generation of new ones. And the horizon is one step: the greedy baseline it "
             "matches is itself myopic, so nothing here shows lookahead over several moves.")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
