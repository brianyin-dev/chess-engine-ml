# v24: selective search and additional thinking time

This experiment changes search, not evaluation or NN training. The frozen opponent is exactly the pre-v24 search from commit bda71b6, with the same heuristic evaluator. No opening book is used after the prescribed common start. The app now enables LMR, retaining its existing 750ms budget and depth-8 ceiling. The library switch defaults off for reproducible baseline comparisons.

## Known mistakes at three budgets

Reconstructed the first confirmed avoidable losing-state transition from each of the 11 v22 pilot losses, preserving full history. At each position, both searches received 250ms, 750ms and 2 seconds, with a depth ceiling of 64 so the clock rather than a low ceiling limits search. Order alternates by position; Stockfish confirmation occurs after both timed searches. One thread, 32MB teacher hash, 256,000 nodes per best/forced-move search. This is a development diagnostic on known mistakes, not unseen playing-strength evidence.

| Time per move | Baseline losing transitions | LMR losing transitions | Baseline mean regret | LMR mean regret |
| --- | ---: | ---: | ---: | ---: |
| 250ms | 9/11 | 9/11 | 416.73cp | 416.73cp |
| 750ms | 8/11 | 8/11 | 414.64cp | 414.64cp |
| 2 seconds | 6/11 | 4/11 | 252.55cp | 217.64cp |

Increasing time helped the unchanged search on these cases. LMR provided additional move-quality improvement at 2 seconds, but none at the smaller budgets. Its reported average depth increased from 2.82 to 3.00 at 250ms, 3.64 to 3.91 at 750ms and 4.36 to 4.91 at 2 seconds. Selective depth is not equivalent to exhaustively searching that many plies: less promising branches receive reduced depth.

The two previously identified rook-endgame evaluation failures still occur even when LMR reaches reported depth 7 at 2 seconds. Greater depth alone does not repair every position. The -150cp mover-perspective threshold flags a serious risk, not a mathematically lost game; best continuation must avoid the threshold, with substantial deterioration, to count as a new transition.

## Reduction policy

Reduce by one ply only at depth at least 3, below the root, on the fifth or later ordered move. Exclude captures, promotions, check evasions, moves giving check, preferred moves, killer moves and moves with positive history. Probe with a narrow window. Any reduced result above alpha is verified at full depth, including a result above beta, before accepting a cutoff. This is selective search and can miss an initially unpromising quiet move; the exclusions reduce that risk without removing it.

Public search has an explicit boolean `use_lmr` switch, allowing comparisons with identical evaluation and timing. Disabled behavior is checked against the frozen source. The frozen source is retained in `search_baseline_v24.py`, with its hash saved in the experiment protocol.

## Fresh paired games

Twenty games on ten fresh weighted opening-book starts, seed 240025, frozen before the matches and excluding prior benchmark starting positions. Every start is played with both colors. Both versions receive 250ms per move and depth cap 64, with no NN, pondering or book after the prefix. Maximum 400 played plies; unfinished games are not draws. All search and teacher workloads are serialized with the exclusive CPU lock.

Finished all twenty games: **11 wins, 0 draws, 9 losses (55%)**, ten complete color-swapped pairs, no errors or unfinished games. The predeclared screen required fewer losing transitions across the three-budget diagnostic and a score above 50% over all twenty completed games, without engine errors. Both conditions passed. Enabled LMR in the app, without increasing its thinking time or changing evaluation.

This is modest, provisional evidence, not an Elo estimate or statistically established strength gain. Ten opening pairs are too few to claim a repeatable advantage, and the 250ms/depth-64 pilot is not an exact test of the app's 750ms/depth-8 configuration. The known-position screen at 750ms found identical move quality. Confirmation on a larger independent set is still needed before a résumé strength claim. LMR's 11–9 result is against our unchanged heuristic search, not Stockfish.

## Verification

All **116 tests pass**, including disabled-versus-frozen search parity, actual reduction execution, board/history restoration, deadline interruption and full-depth verification of reduced results above beta. Audited all 20 PGNs for legal moves, matching recorded move histories/final boards, and genuine terminal results. The frozen baseline matches the original source at bda71b6 byte for byte. See `results/lmr-v24/audit.json`, `mistakes.json`, `report.json` and `games.pgn` for the measurements.
