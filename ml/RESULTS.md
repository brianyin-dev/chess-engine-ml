# Local neural-evaluator experiments (2026-09-29)

The browser app still uses the classical heuristic. These trained models are
optional experiments and were compared in the same search with 250 ms per move,
no opening book during the match, paired colors, and sequential play on the local
machine. A game stopped at the ply cap is unfinished, not a draw. The game counts
are too small to infer Elo or human playing strength.

## Data

`stockfish-book-v3-2026` contains 23,542 positions from 1,000 Stockfish 19
self-play games. Games started with weighted moves from `books/gm2001.bin` for up
to ten plies, then Stockfish played at 10 ms per move. We sampled through ply 112
and labeled positions at depth ten. There are 12,122 sampled positions from ply
60 onward and 5,037 with at most twelve pieces. Games, not individual positions,
were split 80/10/10 into training/validation/test. The manifest records binary
hashes and generation parameters.

`stockfish-book-deviations-v4-2026` adds 9,376 random legal successors of those
positions, each independently labeled at depth ten, while preserving the parent
game's split. The combined dataset has 32,918 positions. These successors expose
the models to mistakes Stockfish self-play rarely makes. Training and test FENs
are disjoint. Both datasets and manifests are checked in so the exact results can
be reproduced without depending on time-limited self-play decisions.

## Held-out score prediction

Mean absolute error is measured against Stockfish's depth-ten centipawn label.
It is not a measure of move quality or game strength.

| Dataset/test mix | Evaluator | MAE (cp) |
| --- | --- | ---: |
| Book self-play | Heuristic | 115.3 |
| Book self-play | Residual NN + heuristic | 108.8 |
| Book self-play | Full-score NN | 80.8 |
| Book plus deviations | Heuristic | 179.9 |
| Book plus deviations | Residual NN + heuristic | 152.9 |
| Book plus deviations | Full-score NN | 148.3 |

On the deviation subset alone, the v4 residual NN averaged 264.9 cp error and
the v4 full-score NN 304.8 cp, versus 340.9 cp for the heuristic. On the original
self-play subset, their errors were 107.9 cp and 85.4 cp, versus 115.3 cp.
The full-score model is faster because it skips the heuristic. A local profile
over 250 sampled positions and 20 rounds measured a median 0.0184 ms per
full-score NN evaluation, 0.0387 ms for the heuristic, and 0.0611 ms for a
residual evaluation. See `ml/artifacts/inference-profile-v4.json` for individual
rounds and the position-file hash. The profile excludes search and is not a
game-strength result.

## Equal-time games versus the heuristic

The paired `benchmarks/openings-diagnostic.json` match used four openings and
both colors, with a maximum of 140 engine-played plies. The v3 full-score NN
scored 0 wins, 0 draws, 8 losses. The v3 residual NN scored 0 wins, 0 draws,
6 losses, 2 unfinished. Both v4 models scored 0 wins, 0 draws, 8 losses.
In the v4 diagnostic games, the faster full-score NN searched a median 7,091
nodes per move against 5,444 for the heuristic, yet lost every game. The slower
residual NN searched a median 3,873 against 6,034. These are different game
paths, so the node counts are descriptive rather than a controlled speed ratio.
The v4 residual model was also checked on four different openings from
`benchmarks/openings-live-2026.json`; it scored 0 wins, 0 draws, 8 losses there.
Across these two local opening sets, it lost all 16 completed games. The result
is strong evidence against using this candidate as the app's default, though it
does not estimate its Elo against other opponents.

In a separate local match against Stockfish 19's UCI 1320 limited-strength
setting, the v4 residual NN scored 2 wins, 1 draw, 2 losses, and 3 unfinished
on four paired openings at 250 ms per move. Stockfish had no restrictive depth
cap and averaged 0.247 seconds per move; the NN averaged 0.247 seconds. The
earlier heuristic checkpoint scored 7–1 on these openings with the same move
budget and a 160-ply cap, versus 140 here. The settings and small samples do
not support a human Elo estimate or a precise head-to-head strength difference.

All candidates passed the seven fixed-depth tactical regression checks, which
demonstrates why those checks cannot stand in for a strength match. The data
distribution is still a likely problem: the original self-play training positions
had a median absolute Stockfish score of only 22 cp, whereas random successors
had a median of 222 cp. The deviation labels improved prediction, but did not
produce better play in these matches. No neural model was promoted to the app.

Artifacts under `ml/artifacts/` contain checkpoints, training metrics, complete
PGNs, per-move timing, search statistics, and match reports. Reproduce a candidate
with `python -m ml.train --data ml/data/stockfish-book-deviations-v4-2026
--target absolute --seed 229 --checkpoint NEW.pt --metrics NEW.json`, then use
`python -m ml.compare --checkpoint NEW.pt --openings
benchmarks/openings-diagnostic.json --pairs 4 --time-ms 250 --max-plies 140
--output NEW-DIRECTORY`. The output paths must not already exist.
