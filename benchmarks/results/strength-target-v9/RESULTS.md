# Local strength target: 60 points in 100 games

The selected target is Stockfish 19 configured with `UCI_LimitStrength=true`
and `UCI_Elo=1320`, its supported minimum. This replaces the requested 1200 setting
with the user's approved stricter benchmark. It does not establish a human rating.
Both engines receive 250 ms per move locally, with Stockfish allowed depth 64
and the app engine retaining depth 8. No opening-book assistance after the start.

## Search change

Principal variation search probes later moves with a null window and fully
re-searches improvements inside the score window. It changes search work, not
the handcrafted evaluator. Full-window reference comparisons cover both cache
settings, repetitions, en passant, and tactical positions. All 87 tests pass.

Five alternating depth-three timing trials per method preserved scores. Median
middlegame runtime fell from 381.6 ms to 274.9 ms (28.0%). Starting-position and
poisoned-pawn runtime were essentially unchanged. See `pvs-profile.json` for
individual timings, visited nodes, and moves. This is a small microbenchmark,
not a measured Elo gain.

## Development pilot

Same four paired v8 opening starts, 250 ms, 400-ply limit:

| Search | Wins | Draws | Losses | Completed | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline at b64c125 | 7 | 0 | 1 | 8 | 87.5% |
| Principal variation search | 7 | 1 | 0 | 8 | 93.75% |

Artifacts are in `ml/artifacts/strength-target-baseline-v9-1320-250ms` and
`ml/artifacts/strength-target-pvs-v9-pilot-1320-250ms`. These small stochastic
matches do not establish a playing-strength improvement. Both use NN weight 0.
The NN remains experimental because its earlier matches lost to the heuristic.

## Preregistered 100-game evaluation

`benchmarks/openings-strength-v9.json` freezes 50 distinct book-derived starts,
with 10, 12, 14, or 16 plies and both colors for each. Sampling seed 91320;
prior benchmark starting FENs excluded. No engine-score or game-result filtering.
The accompanying manifest records hashes and selection policy. This is a fresh
match set, not a claim that these openings never appeared in training data.

The candidate is frozen for the full run. Report JSON records engine source,
opening file, checkpoint, and Stockfish hashes, options, per-move search statistics,
and complete move histories. PGNs are written after each game.

The run terminated with process exit 143 before finishing. Its saved checkpoint
contains 66 completed games (48 wins, 2 draws, 16 losses) and one interrupted
game. The full 100-game target is unverified. Partial results are saved at
`ml/artifacts/strength-target-v9-100games-1320-250ms/report.json`.
Success requires at least 60 points (wins + half draws) across all 100 completed
games. The 1000-ply limit is an operational safeguard; unfinished/error games
are not draws and prevent a complete 100-game claim.

```sh
.venv/bin/python -m ml.compare \
  --checkpoint ml/artifacts/quiet-ranking-v8.pt --nn-weight 0 \
  --openings benchmarks/openings-strength-v9.json --pairs 50 \
  --time-ms 250 --max-plies 1000 \
  --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal \
  --stockfish-elo 1320 --output /tmp/strength-v9-reproduction
```
