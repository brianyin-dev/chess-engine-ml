# v23: consequential losses and bounded correction

The focused candidate did not qualify for another game pilot. It repaired one of two development rook-endgame decisions, but produced slightly worse searched choices on the held-out screen. The app continues to use the unchanged heuristic. No playing-strength improvement is claimed.

## Diagnosis

Reviewed all 11 losses from the v22 20-game pilot. A first pass identified significant errors; a second pass located the first confirmed avoidable losing-state transition in each loss. Stockfish scores are from the player making the move's perspective. A transition requires a best continuation above -150cp (or a winning mate), a played continuation at or below -150cp (or a losing mate), and at least 50cp deterioration or a mate failure. Already-losing positions do not count as new transitions. -150cp is a risk threshold, not a guaranteed loss.

Screening used 16,000 Stockfish nodes; confirmation used 256,000 nodes per same-root best/forced-move search. Stockfish used one thread and 32MB hash; only the last exact scored PV was consumed. Full game history was preserved. These are engine estimates rather than proofs, and finite node budgets can misjudge positions.

Compared heuristic, old v8 NN and v22 NN with unchanged search, at exact depth 3 and then isolated 250ms/depth 8, rotating model order. All three chose identical moves at depth 3 in all 11 true-transition positions. In the timed controls, the NNs sometimes finished a shallower iteration. For example, in source game 17 the heuristic reached depth 4 and chose a zero-regret continuation, while both NNs reached depth 3 and chose a move estimated to lose 1,060cp. Source games 10 and 13 were also repaired by depth 3 but not their timed depths 2 and 1. Source game 15 remained a serious mistake at depth 3, but the NNs at depth 2 lost still more value. This separates some search-depth limitations from evaluation-sensitive cases; it does not prove every remaining error has a single cause.

## What a bounded correction could change

At each original move's completed depth (clamped to 1–4), forced complete root searches traced score-bearing leaves. Quiet-position correction is still 25% of a residual bounded to 250cp: at most 62.5cp before rounding. Two forced root values cannot reverse a gap greater than 126cp on the same complete fixed-depth minimax/quiescence tree. This bound does not apply across different depths or clock-limited searches.

A diagnostic oracle labeled actual quiet evaluation calls with Stockfish at 64,000 nodes, clipped the correction to the same bound, and applied the same gate and rounding. Mate labels supplied only the bounded correction direction. The oracle demonstrated two repairs at matched depth 4:

| Pilot source game (zero-based) | Oracle move | Baseline regret | Oracle regret |
| --- | --- | ---: | ---: |
| 1 | g8f8 | 647cp | 0cp |
| 4 | a2a4 | 298cp | 0cp |

Both avoided the losing-state threshold. The oracle is a feasibility diagnostic, not a deployable fast evaluator or a strength result. Failure to repair the other positions does not prove no learned evaluator could repair them.

## Focused training

Before fitting, both demonstrated cases and their complete current opening-family groups were designated development/training, including one previously assigned pilot-validation case. Both groups were excluded from new validation and test sets. The old pilot is consequently development evidence, not a new holdout.

Generated 256 novel quiet score examples from those two cases and one eligible searched-backup ranking pair. Targets are feasible clipped oracle scores, with raw Stockfish labels retained separately; they are not calibrated Stockfish evaluations. Retained 512 prior v22 prediction targets and 512 legacy teacher ranking pairs. New canonical labels exclude previously labeled positions. Legacy opening-family provenance remains incomplete, so full historical family isolation is not claimed.

Kept architecture (892 inputs, 64/32 hidden layers), fast inference, search and heuristic fixed. Used weighted correction MSE, prediction anchoring and smooth pair loss with protection for existing rounded rankings. Trained 60 epochs; checked six predetermined checkpoints at epochs 10–60. Selection used searched validation on 16 novel roots, then retention regressions, with training repairs only as a tie-break. Selected epoch 20. It repaired a2a4 but chose g8g7 in the other development case, still losing 742cp. It also regressed 15 previously correct validation rankings; retention was insufficient.

## Unseen searched-choice gate

The 32-root final test was frozen before fitting and excluded the two development families. All evaluators completed depth 3; each distinct chosen move was confirmed with 256,000-node Stockfish analysis.

| Evaluator | Mean regret | Avoidable losing transitions | Moves losing at least 150cp |
| --- | ---: | ---: | ---: |
| Heuristic | 82.25cp | 4 | 5 |
| Old v8 NN, 25% | 82.25cp | 4 | 5 |
| v22 NN, 25% | 82.25cp | 4 | 5 |
| Focused v23 candidate | 85.125cp | 4 | 5 |

Equal aggregate baseline numbers do not imply identical moves. The candidate changed one choice relative to v22 and increased aggregate regret. The predefined gate required strictly lower regret than v22, no worse regret than the heuristic, and no additional losing transitions or 150cp mistakes. It failed. No new game pilot, Stockfish opponent match or app promotion was run.

## Verification and next implication

All 111 tests pass. Canonical overlap checks found zero training overlap with validation rows, validation roots or test roots; development families are excluded. All five frozen runtime source hashes match v22. Torch predictions and fast runtime inference agree after gate/blend/rounding on 1,283 positions. Twenty-two forced searches at actual depths agree with the original full-window search and restore full board history. See `artifacts/critical-v23/audit.json` and the saved protocols, teacher reviews and training history.

The next useful diagnostic is to learn both confirmed development decisions under their actual searched backups while preserving existing rankings, before collecting another broad dataset or running more games. The unlearned rook endgame is a concrete failure case. Search-depth fixes should be evaluated separately; these measurements show why speed can still determine a decisive move.
