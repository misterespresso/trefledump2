"""Mazes built so that the greedy move is provably wrong.

Every maze has one fork. One branch points at the goal and dead-ends a few cells
later; the other starts by moving away from the goal and is the only route there.
A policy that minimises distance-to-goal takes the bait every time, so the fork is
a clean one-bit test of whether anything is looking further than one step.

Grid convention: (row, col), row 0 at the top. '#' wall, '.' open, 'S' start,
'G' goal. Moves are north/south/east/west.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator

MOVES: dict[str, tuple[int, int]] = {
    "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1),
}


@dataclass(frozen=True)
class Maze:
    grid: tuple[str, ...]
    start: tuple[int, int]
    goal: tuple[int, int]
    fork: tuple[int, int]
    bait_move: str      # the move that reduces distance-to-goal and dead-ends
    correct_move: str   # the only move at the fork that reaches the goal

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0])

    def open_at(self, rc: tuple[int, int]) -> bool:
        r, c = rc
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != "#"

    def legal(self, rc: tuple[int, int]) -> dict[str, tuple[int, int]]:
        out = {}
        for name, (dr, dc) in MOVES.items():
            nxt = (rc[0] + dr, rc[1] + dc)
            if self.open_at(nxt):
                out[name] = nxt
        return out

    def render(self, at: tuple[int, int] | None = None) -> list[str]:
        """Only four glyphs ever reach the model: wall, floor, agent, goal."""
        rows = [list(r.replace("S", ".")) for r in self.grid]
        if at is not None and at != self.goal:
            rows[at[0]][at[1]] = "@"
        return ["".join(r) for r in rows]


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs(m: Maze, src: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {src: 0}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nxt in m.legal(cur).values():
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def optimal_move(m: Maze, at: tuple[int, int]) -> set[str]:
    """Every move that stays on a shortest path to the goal."""
    from_goal = bfs(m, m.goal)
    here = from_goal.get(at)
    if here is None:
        return set()
    return {name for name, nxt in m.legal(at).items() if from_goal.get(nxt, 10**9) == here - 1}


def greedy_move(m: Maze, at: tuple[int, int]) -> set[str]:
    """Every move that minimises straight-line distance to the goal, ignoring walls."""
    legal = m.legal(at)
    if not legal:
        return set()
    best = min(manhattan(nxt, m.goal) for nxt in legal.values())
    return {name for name, nxt in legal.items() if manhattan(nxt, m.goal) == best}


# --------------------------------------------------------------------- builder
def build(rng: random.Random) -> Maze | None:
    """One bait maze. Returns None if the random parameters produced a bad layout.

    Layout, before random flips: the agent walks up a stem to a fork. East of the
    fork is a short corridor that ends in a wall, and it points straight at the
    goal. West of the fork the corridor runs away from the goal, then north, then
    all the way east along the top to reach it.
    """
    stem = rng.randint(1, 4)          # cells below the fork
    bait = rng.randint(2, 5)          # length of the dead-end corridor
    detour = rng.randint(2, 5)        # how far west the true path goes
    climb = rng.randint(2, 4)         # how far north before turning back east
    rows = stem + climb + 3
    cols = detour + bait + 4

    grid = [["#"] * cols for _ in range(rows)]
    # The stem hangs below the fork, so the fork sits `stem` rows above the floor.
    fork_r, fork_c = rows - 2 - stem, detour + 1
    start = (fork_r + stem, fork_c)
    if not (0 < fork_r - climb and start[0] < rows - 1):
        return None

    def carve(r: int, c: int) -> None:
        if 0 < r < rows - 1 and 0 < c < cols - 1:
            grid[r][c] = "."

    for r in range(fork_r, start[0] + 1):        # the stem
        carve(r, fork_c)
    for c in range(fork_c, min(fork_c + bait + 1, cols - 1)):  # the bait, going east
        carve(fork_r, c)
    for c in range(fork_c - detour, fork_c + 1):  # the detour, going west
        carve(fork_r, c)
    west_c = fork_c - detour
    for r in range(fork_r - climb, fork_r + 1):   # climbing north
        carve(r, west_c)
    top_r = fork_r - climb
    for c in range(west_c, cols - 1):             # running east along the top
        carve(top_r, c)

    goal = (top_r, cols - 2)
    if grid[goal[0]][goal[1]] != ".":
        return None
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    m = Maze(tuple("".join(r) for r in grid), start, goal, (fork_r, fork_c), "east", "west")

    # Check the trap on the canonical orientation, where bait is east and the route west.
    if bfs(m, m.start).get(m.goal) is None:
        return None
    if greedy_move(m, m.fork) != {"east"} or optimal_move(m, m.fork) != {"west"}:
        return None

    # Only then rotate it. Left as built, the answer would be "west" in every maze and
    # a model that always replied "west" would score 100%.
    if rng.random() < 0.5:
        m = mirror_h(m)
    if rng.random() < 0.5:
        m = flip_v(m)
    if rng.random() < 0.5:
        m = transpose(m)
    return m


_H = {"east": "west", "west": "east", "north": "north", "south": "south"}
_V = {"north": "south", "south": "north", "east": "east", "west": "west"}
_T = {"north": "west", "west": "north", "south": "east", "east": "south"}


def mirror_h(m: Maze) -> Maze:
    w = len(m.grid[0]) - 1
    f = lambda rc: (rc[0], w - rc[1])
    return Maze(tuple(r[::-1] for r in m.grid), f(m.start), f(m.goal), f(m.fork),
                _H[m.bait_move], _H[m.correct_move])


def flip_v(m: Maze) -> Maze:
    h = len(m.grid) - 1
    f = lambda rc: (h - rc[0], rc[1])
    return Maze(tuple(reversed(m.grid)), f(m.start), f(m.goal), f(m.fork),
                _V[m.bait_move], _V[m.correct_move])


def transpose(m: Maze) -> Maze:
    grid = tuple("".join(m.grid[r][c] for r in range(len(m.grid))) for c in range(len(m.grid[0])))
    f = lambda rc: (rc[1], rc[0])
    return Maze(grid, f(m.start), f(m.goal), f(m.fork), _T[m.bait_move], _T[m.correct_move])


def generate(n: int, seed: int, max_attempts: int = 100_000) -> list[Maze]:
    rng = random.Random(seed)
    out: list[Maze] = []
    seen: set[tuple[str, ...]] = set()
    for _ in range(max_attempts):
        if len(out) >= n:
            return out
        m = build(rng)
        if m is None or m.grid in seen:
            continue
        if greedy_move(m, m.fork) != {m.bait_move} or optimal_move(m, m.fork) != {m.correct_move}:
            continue  # a transform that broke the trap is discarded rather than trusted
        seen.add(m.grid)
        out.append(m)
    raise RuntimeError(f"only generated {len(out)} of {n} mazes in {max_attempts} attempts")


def walk(m: Maze, policy, limit: int) -> tuple[list[tuple[int, int]], bool]:
    """Run a code-side policy. Returns the path and whether it reached the goal."""
    at, path = m.start, [m.start]
    for _ in range(limit):
        if at == m.goal:
            return path, True
        legal = m.legal(at)
        if not legal:
            break
        at = legal[policy(m, at)]
        path.append(at)
    return path, at == m.goal
