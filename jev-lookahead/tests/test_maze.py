"""Offline tests. The experiment is only meaningful if the mazes really trap a greedy walker."""

from __future__ import annotations

import random

from lookahead.maze import (MOVES, bfs, generate, greedy_move, manhattan, mirror_h,
                            optimal_move, transpose, flip_v, walk)

MAZES = generate(60, 20260919)


def test_every_maze_traps_a_greedy_walker():
    for m in MAZES:
        assert greedy_move(m, m.fork) == {m.bait_move}, m.grid
        assert optimal_move(m, m.fork) == {m.correct_move}, m.grid
        assert m.bait_move != m.correct_move


def test_every_maze_is_solvable_and_the_bait_is_a_dead_end():
    for m in MAZES:
        assert bfs(m, m.start).get(m.goal) is not None
        # Stepping onto the bait strictly lengthens the remaining route.
        from_goal = bfs(m, m.goal)
        here = from_goal[m.fork]
        bait_cell = m.legal(m.fork)[m.bait_move]
        assert from_goal[bait_cell] == here + 1


def test_the_correct_answer_is_not_always_the_same_direction():
    """Without this the whole experiment would be gameable by replying 'west' every time."""
    dirs = {m.correct_move for m in MAZES}
    assert len(dirs) == 4, dirs
    counts = {d: sum(1 for m in MAZES if m.correct_move == d) for d in dirs}
    assert min(counts.values()) >= 5, counts


def test_policies_behave_as_the_experiment_assumes():
    opt = lambda m, at: sorted(optimal_move(m, at))[0]
    grd = lambda m, at: sorted(greedy_move(m, at))[0]
    assert all(walk(m, opt, 150)[1] for m in MAZES)
    assert not any(walk(m, grd, 150)[1] for m in MAZES)


def test_transforms_preserve_structure():
    m = MAZES[0]
    for f in (mirror_h, flip_v, transpose):
        t = f(m)
        assert t.grid != m.grid or f is transpose
        assert bfs(t, t.start).get(t.goal) == bfs(m, m.start).get(m.goal)
        assert greedy_move(t, t.fork) == {t.bait_move}
        assert optimal_move(t, t.fork) == {t.correct_move}


def test_render_hides_the_start_marker_and_shows_the_agent():
    m = MAZES[0]
    rows = m.render(m.fork)
    assert "S" not in "".join(rows)
    assert "".join(rows).count("@") == 1
    assert rows[m.fork[0]][m.fork[1]] == "@"


def test_generation_is_deterministic():
    assert [m.grid for m in generate(10, 7)] == [m.grid for m in generate(10, 7)]


def test_legal_moves_never_include_a_wall():
    rng = random.Random(0)
    for m in MAZES[:20]:
        for _ in range(20):
            r, c = rng.randrange(m.rows), rng.randrange(m.cols)
            if not m.open_at((r, c)):
                continue
            for d, nxt in m.legal((r, c)).items():
                assert m.open_at(nxt)
                assert nxt == (r + MOVES[d][0], c + MOVES[d][1])
