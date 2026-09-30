# Incremental hybrid evaluation without changing network weights

Status: completed. The optimization is retained as an experimental opt-in mode.
The app still uses the heuristic; the hybrid failed the heuristic game gate.

The candidate uses the unchanged v8 checkpoint, all 892 inputs including the
98 relationship features, and its actual 25% quiet-only correction. No retraining,
new label generation, network pruning, or evaluation-policy change.

## Cached feature dependencies

- Piece planes update only the squares whose piece masks changed.
- King geometry reuses a block when its piece mask and both king squares match.
- Pawn relationships reuse a block when both pawn masks and the own king match.
- Handcrafted material and piece-square terms reuse per-color/per-piece blocks;
  king piece-square terms additionally depend on game phase.
- Development depends on minor/queen masks and phase; passed pawns depend on both
  pawn masks; rook-file activity depends on rook and both pawn masks.
- Mobility, center control, and NN attack maps are recomputed from the current
  board. Per-square attack caching was dropped because overhead erased savings.

Each cache verifies its full board dependencies. No assumption about consecutive
moves or move-stack depth; undo operations, unrelated sibling positions, and new
games are safe. Storage is bounded to the last entry per feature block. Neural
encoding has separate caches for canonical White and Black perspectives. Frozen
handcrafted constants remain fixed for the lifetime of an evaluator.
Mutable caches belong to one sequential search worker; concurrent workers use
separate evaluator instances.

The optimization is opt-in (`NeuralEvaluator(..., incremental=True)` or
`ml.compare --incremental`). The ordinary evaluator and frozen heuristic opponent
retain their prior behavior. No changes to the search algorithm or app defaults.

## Verification and profiling

Exact feature parity on 3,000 move/undo/reset positions; exact handcrafted score
parity on another 3,000 such positions. Tests cover castling, en-passant,
capture-underpromotion, random undo operations, and neural search move/score/node
parity. All **99 tests pass**.

The first cache of neural relationship features alone gave no meaningful net
search-time gain. Profiling found the handcrafted baseline dominated hybrid
runtime; caching those feature blocks and reducing cache bookkeeping helped.

Across ten opening positions, five alternating depth-three search trials per
position give summed median times of **2.22066 s reference vs 1.99505 s incremental:
10.2% less complete-search time**. Moves, scores, nodes, and quiescence nodes are
identical in every trial. This measures the complete search, not just one feature
function. Search preparation builds immutable king tables before per-move clocks.

Primary profile: `ml/artifacts/incremental-v15/final-profile.json`.
Earlier exploratory profiles are preserved but are not claimed as improvements.

Additional verification on all ten fresh game starts at a 12,000-node budget
(clock disabled) gives identical moves, scores, completed depth, node counts,
quiescence node counts, and stop reasons. This includes interrupted iterations.
See `fresh-node-parity.json`.

## Fresh equal-time games

Freeze 50 new book starts before results; exclude previous benchmark starts and
canonical aliases of labeled positions/ranking endpoints. First ten starts give
20 paired-color games versus the frozen heuristic, then 20 versus the unchanged
reference v8 evaluator. Both NN sides use the same checkpoint and 25% quiet gate;
only the candidate has incremental caching. Both screens share the same fresh
starts, 250 ms cooperative clocks, no opening book after the prefix, and depth
cap eight. CPU jobs are serialized with the benchmark lock.

Extend the heuristic match to 100 games using the remaining 40 starts only if
all pilot games complete, score is at least 60% against heuristic and above 50%
against reference NN. No automatic app promotion or external Stockfish games.

Coordinator: `ml.run_incremental_v15`.
Profile reproduction: `ml.profile_incremental`.
Live status: `ml/artifacts/incremental-v15/status.json`.

## Completed game results at 250 ms per move

| Opponent | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|
| Frozen unchanged heuristic | 4 | 3 | 13 | 27.5% |
| Reference v8 NN, same weights/blend without incremental cache | 12 | 0 | 8 | 60% |

Both matches completed 20 games and 10 color pairs, with zero errors and zero
unfinished games. The direct NN screen favors the incremental implementation,
but 20 games is too small to establish a precise strength gain. This 60% is
against the reference NN, **not Stockfish 1500** and not the heuristic.

The heuristic screen failed the predeclared 60% extension criterion. No 100-game
or external Stockfish match was run. Do not compare the 27.5% to prior screens as
if the openings were identical. The optimization preserves evaluation quality
and improves complete-search timing, but this round does not establish a hybrid
that is stronger than the heuristic. All 892 neural inputs and the old checkpoint
are retained; there is no new trained model.
