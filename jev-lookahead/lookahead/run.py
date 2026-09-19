"""Two experiments on bait mazes.

  fork   the agent is placed at the fork and asked once. One request per maze, and
         a clean one-bit read: greedy scores 0% by construction, uniform choice over
         the four directions scores 25%.
  walk   the agent navigates from the start, one request per step, up to a cap.
         Measures whether it reaches the goal at all and whether it escapes the bait.

    python -m lookahead.run                # both
    python -m lookahead.run --only fork
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .client import Jev, Store
from .maze import MOVES, Maze, bfs, generate, greedy_move, manhattan, optimal_move, walk

ROOT = Path(__file__).resolve().parents[1]
DATA, CHARTS = ROOT / "data", ROOT / "charts"
FORK_JSONL, WALK_JSONL = DATA / "fork.jsonl", DATA / "walk.jsonl"
REPORT = ROOT / "report.md"
SEED = 20260919
N_FORK, N_WALK, STEP_CAP = 200, 30, 30
DIRS = ("north", "south", "east", "west")


def state_for(m: Maze, at: tuple[int, int], visited: list[tuple[int, int]] | None = None) -> dict[str, Any]:
    """The fork phase sends no history. The walk phase sends the trail, so a repeated
    dead end is visible in the state and looping is a choice rather than an inevitability."""
    legal = m.legal(at)
    seen = set(visited or [])
    extra: dict[str, Any] = {}
    if visited:
        extra = {
            "cells_you_have_already_visited": [{"row": r, "col": c} for r, c in dict.fromkeys(visited)],
            "note": "Revisiting a cell you have already stood on makes no progress.",
        }
    return {
        "task": ("You are '@' in a grid maze and you want to stand on 'G'. You may move one cell "
                 "per turn. '#' is a wall you cannot enter. Choose the direction that starts the "
                 "shortest route to 'G'. A direction that looks closer to 'G' may lead into a "
                 "dead end."),
        "legend": {"#": "wall", ".": "open floor", "@": "you", "G": "the goal"},
        "maze": m.render(at),
        "your_position": {"row": at[0], "col": at[1]},
        "goal_position": {"row": m.goal[0], "col": m.goal[1]},
        "moves": {d: {"row": at[0] + MOVES[d][0], "col": at[1] + MOVES[d][1],
                      "blocked_by_wall": d not in legal,
                      **({"already_visited": (at[0] + MOVES[d][0], at[1] + MOVES[d][1]) in seen}
                         if visited else {})}
                  for d in DIRS},
        **extra,
    }


QUESTION = {"type": "choice",
            "instructions": "Which direction should you move to reach 'G' in as few moves as possible?",
            "criteria": {d: None for d in DIRS}}


def ask_many(jev: Jev, store: Store, jobs: list[tuple[str, Maze, tuple[int, int], dict[str, Any]]],
             workers: int = 6, trails: dict[int, list[tuple[int, int]]] | None = None) -> dict[str, dict[str, Any]]:
    """Ask one Choice per job, skipping anything already recorded."""
    todo = [j for j in jobs if j[0] not in store.done]
    def one(job):
        rid, m, at, meta = job
        trail = (trails or {}).get(meta.get("maze_index"))
        res = jev.ask(state_for(m, at, trail), {"q0": QUESTION})
        rec = {"record_id": rid, "at": list(at), "goal": list(m.goal), "grid": list(m.grid), **meta, **res}
        if res.get("ok"):
            a = res["response"]["answers"]["q0"]
            rec.update(choice=a["choice"], confidence=a["confidence"],
                       probabilities=a["probabilities"], served_model=res["response"].get("model"))
        store.append(rec)
        return rec
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))
    return {j[0]: store.done.get(j[0], {}) for j in jobs}


# ------------------------------------------------------------------ experiments
def run_fork(jev: Jev, mazes: list[Maze]) -> None:
    store = Store(FORK_JSONL)
    jobs = [(f"fork:{i:04d}", m, m.fork,
             {"phase": "fork", "maze_index": i,
              "optimal": sorted(optimal_move(m, m.fork)), "greedy": sorted(greedy_move(m, m.fork)),
              "legal": sorted(m.legal(m.fork))})
            for i, m in enumerate(mazes)]
    ask_many(jev, store, jobs)
    print(f"[fork] {len(store.done)} recorded", flush=True)


def run_walk(jev: Jev, mazes: list[Maze]) -> None:
    store = Store(WALK_JSONL)
    positions = {i: m.start for i, m in enumerate(mazes)}
    trails: dict[int, list[tuple[int, int]]] = {i: [m.start] for i, m in enumerate(mazes)}
    finished: set[int] = set()
    for step in range(STEP_CAP):
        jobs = []
        for i, m in enumerate(mazes):
            if i in finished:
                continue
            at = positions[i]
            if at == m.goal:
                finished.add(i)
                continue
            jobs.append((f"walk:{i:04d}:{step:02d}", m, at,
                         {"phase": "walk", "maze_index": i, "step": step,
                          "optimal": sorted(optimal_move(m, at)), "greedy": sorted(greedy_move(m, at)),
                          "legal": sorted(m.legal(at))}))
        if not jobs:
            break
        recs = ask_many(jev, store, jobs, trails=trails)
        for rid, m, at, meta in jobs:
            r = recs.get(rid) or {}
            d = r.get("choice")
            i = meta["maze_index"]
            if d in m.legal(at):                      # an illegal pick wastes the turn
                positions[i] = m.legal(at)[d]
                trails[i].append(positions[i])
        print(f"[walk] step {step + 1}: {len(jobs)} moving, {len(finished)} finished", flush=True)
    print(f"[walk] done, {len(finished)}/{len(mazes)} reached the goal", flush=True)


# ------------------------------------------------------------------ analysis
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def analyse(mazes: list[Maze]) -> dict[str, Any]:
    fork = [r for r in Store(FORK_JSONL).read() if r.get("ok")]
    n = len(fork)
    got = sum(1 for r in fork if r["choice"] in r["optimal"])
    bait = sum(1 for r in fork if r["choice"] in r["greedy"])
    illegal = sum(1 for r in fork if r["choice"] not in r["legal"])
    back = sum(1 for r in fork if r["choice"] not in r["optimal"] and r["choice"] not in r["greedy"]
               and r["choice"] in r["legal"])
    out: dict[str, Any] = {
        "n_mazes": len(mazes),
        "fork": {
            "n": n,
            "correct": got, "correct_rate": got / n if n else float("nan"),
            "correct_ci": list(wilson(got, n)),
            "took_the_bait": bait, "bait_rate": bait / n if n else float("nan"),
            "bait_ci": list(wilson(bait, n)),
            "walked_into_a_wall": illegal, "illegal_rate": illegal / n if n else float("nan"),
            "backwards": back,
            "mean_top_p": statistics.fmean(max(r["probabilities"].values()) for r in fork) if n else float("nan"),
            "mean_confidence": statistics.fmean(r["confidence"] for r in fork) if n else float("nan"),
            # It never picks a blocked direction, so the honest chance baseline is a
            # uniform choice among the legal moves, not among all four.
            "mean_legal_moves": statistics.fmean(len(r["legal"]) for r in fork) if n else float("nan"),
            "legal_baseline": statistics.fmean(1 / len(r["legal"]) for r in fork) if n else float("nan"),
            "uniform_baseline": 0.25, "greedy_baseline": 0.0,
        },
    }
    # confidence when right against when wrong
    right = [max(r["probabilities"].values()) for r in fork if r["choice"] in r["optimal"]]
    wrong = [max(r["probabilities"].values()) for r in fork if r["choice"] not in r["optimal"]]
    out["fork"]["top_p_when_right"] = statistics.fmean(right) if right else None
    out["fork"]["top_p_when_wrong"] = statistics.fmean(wrong) if wrong else None

    rows = [r for r in Store(WALK_JSONL).read() if r.get("ok")]
    by: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["maze_index"], []).append(r)
    paths, reached, steps, illegals, opt_steps = {}, 0, [], 0, []
    for i, rs in by.items():
        m = mazes[i]
        rs.sort(key=lambda r: r["step"])
        at, path = m.start, [m.start]
        for r in rs:
            illegals += 0 if r["choice"] in r["legal"] else 1
            if r["choice"] in m.legal(at):
                at = m.legal(at)[r["choice"]]
                path.append(at)
        paths[i] = path
        if at == m.goal:
            reached += 1
            steps.append(len(path) - 1)
            opt_steps.append(bfs(m, m.start)[m.goal])
    out["walk"] = {
        "n": len(by), "reached": reached,
        "reached_rate": reached / len(by) if by else float("nan"),
        "reached_ci": list(wilson(reached, len(by))) if by else [float("nan")] * 2,
        "mean_steps_when_reached": statistics.fmean(steps) if steps else None,
        "mean_optimal_steps": statistics.fmean(opt_steps) if opt_steps else None,
        "illegal_attempts": illegals,
        "total_decisions": len(rows),
        "greedy_baseline_reached": 0.0,
    }
    # The mechanism: does having already walked the dead end change the fork decision?
    first = [0, 0]
    later = [0, 0]
    for i, rs in by.items():
        m = mazes[i]
        rs.sort(key=lambda r: r["step"])
        visits = 0
        for r in rs:
            if tuple(r["at"]) != m.fork:
                continue
            visits += 1
            bucket = first if visits == 1 else later
            bucket[1] += 1
            bucket[0] += int(r["choice"] == m.correct_move)
    out["walk"]["fork_first_visit"] = {
        "k": first[0], "n": first[1],
        "rate": first[0] / first[1] if first[1] else float("nan"),
        "ci": list(wilson(first[0], first[1])) if first[1] else [float("nan")] * 2,
    }
    out["walk"]["fork_later_visits"] = {
        "k": later[0], "n": later[1],
        "rate": later[0] / later[1] if later[1] else float("nan"),
        "ci": list(wilson(later[0], later[1])) if later[1] else [float("nan")] * 2,
    }
    out["_paths"] = {str(k): [list(p) for p in v] for k, v in paths.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Does Jev look further than one move ahead?")
    ap.add_argument("--only", choices=["fork", "walk"], default=None)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--rate", type=float, default=10.0)
    args = ap.parse_args(argv)

    mazes = generate(N_FORK, SEED)
    if not args.analyse_only:
        jev = Jev(per_second=args.rate)
        if args.only in (None, "fork"):
            run_fork(jev, mazes)
        if args.only in (None, "walk"):
            run_walk(jev, mazes[:N_WALK])
        print(f"[usage] {jev.requests} requests, {jev.input_tokens} input tokens", flush=True)

    res = analyse(mazes)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "summary.json").write_text(json.dumps({k: v for k, v in res.items() if k != "_paths"}, indent=2))
    f, w = res["fork"], res["walk"]
    print(f"\nfork decision over {f['n']} mazes:")
    print(f"  correct (the non-greedy branch) : {f['correct']:4d} = {f['correct_rate']:6.1%}")
    print(f"  took the bait (greedy)          : {f['took_the_bait']:4d} = {f['bait_rate']:6.1%}")
    print(f"  walked into a wall              : {f['walked_into_a_wall']:4d} = {f['illegal_rate']:6.1%}")
    print(f"  backwards                       : {f['backwards']:4d}")
    print(f"  baselines: greedy 0.0%, uniform over the {f['mean_legal_moves']:.0f} legal moves "
          f"{f['legal_baseline']:.1%}")
    print(f"  mean top probability: {f['top_p_when_right']:.3f} when right, "
          f"{f['top_p_when_wrong']:.3f} when wrong")
    if w["n"]:
        print(f"\nfull navigation over {w['n']} mazes, cap {STEP_CAP} steps:")
        print(f"  reached the goal: {w['reached']}/{w['n']} = {w['reached_rate']:.1%} "
              f"(greedy policy: 0%)")
        if w["mean_steps_when_reached"]:
            print(f"  steps when it arrived: {w['mean_steps_when_reached']:.1f} "
                  f"against an optimal {w['mean_optimal_steps']:.1f}")
        print(f"  illegal move attempts: {w['illegal_attempts']}/{w['total_decisions']}")
        fv, lv = w["fork_first_visit"], w["fork_later_visits"]
        print(f"  at the fork, first visit : {fv['k']}/{fv['n']} = {fv['rate']:.1%}")
        print(f"  at the fork, after the dead end is in history: {lv['k']}/{lv['n']} = {lv['rate']:.1%}")
    from .report import write_all
    write_all(mazes, res, REPORT, CHARTS)
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
