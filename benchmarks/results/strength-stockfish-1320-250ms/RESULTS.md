# Limited-strength Stockfish checkpoint

The current engine scored **11 wins and 1 loss** in 12 completed paired games
against Stockfish 19 configured with `UCI_LimitStrength=true` and `UCI_Elo=1320`.
There were no draws, unfinished games, engine errors, or depth-zero fallback moves.

## Setup

- Six prescribed opening lines were each played twice with colors swapped.
- Both engines received 250 ms per move and ran sequentially with one thread.
- The current engine's opening book was bypassed because every game began after a
  prescribed opening; results therefore measure search and evaluation behavior.
- Games were capped at 160 engine-played plies after the opening. Every game ended
  by checkmate before the cap.
- The exact source files, opening file, environment, Stockfish executable, UCI
  options, moves, positions, and timings are hashed or recorded in `report.json`.

| Outcome for current engine | Games |
| --- | ---: |
| Wins | 11 |
| Draws | 0 |
| Losses | 1 |
| Unfinished/errors | 0 |

The current engine averaged 240.8 ms across 486 moves and completed a mean main
search depth of 3.06 plies. Stockfish averaged 235.7 ms across 481 moves. Maximum
observed overruns were 2.9 ms for the current engine and 11.5 ms for Stockfish.

## Interpretation

This is encouraging evidence that the application can compete above the intended
roughly 1200-friend target in this specific computer match. It is not a human or
FIDE Elo measurement. Stockfish's UCI rating is an engine calibration setting;
human style, hardware, move budget, prescribed openings, and the small sample all
affect the comparison. The 11/12 score has a wide 95% Wilson interval of roughly
64.6% to 98.5%, so a larger tournament is needed before claiming a stable rating.

The six openings were previously reserved but have now been consumed as a
validation set. Future tuning should use different openings and preserve a new
holdout set. A public human-play test remains valuable because the application
uses a 1.5-second move budget and an opening book, unlike this controlled run.

## Artifacts

- `report.json`: complete reproducibility metadata and per-move statistics.
- `games.pgn`: all 12 games with engine identity and timing comments.

