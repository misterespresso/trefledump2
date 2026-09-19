# Does Jev look further than one move ahead?

A follow-up to the planner experiment in [`../jev-choice-audit`](../jev-choice-audit),
which left one question open. There, Jev chose its own next question in a game of
twenty questions and closed 94% of the distance between random and optimal play. But
the baseline it matched was *greedy*, and in twenty questions the greedy move is
almost always the good move, so nothing ruled out a purely myopic policy.

These mazes are built so that greedy is provably wrong.

```
##########
#####.####
#####.####
#....@..##
#.########
#.......G#
##########
```

One branch from the fork points at the goal and dead-ends. The other starts by
moving *away* and is the only route there. Wall positions are handed to the model
as an explicit `blocked_by_wall` flag on every option, so reading the picture is not
what is under test.

## Result

| | |
|---|---|
| Chose the branch that reaches the goal | **9 / 200 = 4.5%** |
| Chose the dead end | 188 / 200 = 94.0% |
| Random among the three legal moves | 33.3% |
| Greedy policy, in code | 0% |
| Moved into a wall | 0 of 1,003 decisions |

At 4.5% against a 33.3% chance baseline it is not guessing. It is greedy, with
almost no noise. It also reported a higher top probability when wrong (0.674) than
when right (0.492).

Giving it a record of where it has already been changes this, but less than it
looks: at the fork it goes from 6.7% on a first visit to 32.1% once the dead end is
in its trail, and 32.1% is indistinguishable from picking at random among the legal
moves. The memory cancels the pull of the bait rather than teaching the route. Over
a whole maze that is still enough to reach the goal 27% of the time against greedy's
0%.

Full write-up: **[report.md](report.md)**.

## Run it

```bash
export TYPESAFE_API_KEY=...
pip install matplotlib pytest
python -m pytest tests -q            # 8 tests, no network
python -m lookahead.run              # ~1,000 requests, resumable
python -m lookahead.run --analyse-only
```

Collection appends every exchange to `data/*.jsonl` and skips anything already
recorded, so the analysis and charts regenerate offline with no API key.

## Layout

```
lookahead/
  maze.py     generation, BFS, the greedy and optimal policies
  client.py   self-contained Jev client, rate limit, retries, JSONL store
  run.py      the two experiments and the metrics
  report.py   charts and report.md
tests/        8 tests, all offline
```

The tests are the load-bearing part. They assert that every generated maze really
does trap a greedy walker, that the bait is a genuine dead end, that optimal play
solves all of them and greedy solves none, and that the correct answer is spread
across all four directions. That last one matters: before it was added, the answer
was `west` in every maze and a model that always replied `west` would have scored
100%.
