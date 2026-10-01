# v21: family-isolated training with fixed inference

This round did not establish a stronger learned evaluator. Both the 5% and 25% training runs retained their original v8 weights under validation selection. The fresh game pilot therefore measures the existing 25% configuration with v20 fast inference, not a retraining improvement. The app still uses the heuristic.

## Changes

- Kept v20 inference, network architecture, search, and heuristic fixed.
- Generated 560 independent book-start Stockfish games (2,000 nodes/move, at most 72 continuation plies), covering 101 opening families. A canonical first-four-ply board defines a family; transposing starts share its split. Source families: 47 train, 27 validation, 27 test.
- Initial 240 games produced an undersized held-out label set. Before fitting, added a recorded 320-game extension balanced by source split, with at most six added games/family. This extension used source coverage, never teacher outcomes or model performance, to select games.
- Collected up to eight quiet positions per game. Teacher alternatives were settled through captures, promotions, and check evasions, then labeled at 64,000 nodes. Rankings require at least 50 cp teacher separation and an attainable integer 5 cp margin under the rounded 5% quiet blend. Excluded all previously labeled canonical board aliases.
- Added 207 useful training rankings across 28 families, 78 validation rankings across 14 families, and 67 test rankings across 15 families. Retained previous training data and correct rankings. Not every source family yields a usable near-tie ranking.
- Removed absolute score loss in this experiment; optimized rounded hybrid move rankings with protection and an anchor to original predictions. Initialization: old v8. Architecture: unchanged. Learning rate .0001, ranking weight 4, margin 5 cp, hard/protected multipliers 3, anchor .1, max 30 epochs, patience 8.
- Corrected the training reachability mask to round each complete endpoint score and include margins exactly equal to the target. Previously its continuous bound could discard attainable examples (11 + 12.5 rounds to 24, versus a gated score19: attainable margin5, not4.5).

## Former held-out error review

The rejected v20 trained model had 42 wrong held-out rankings, including nine regressions from old5. Overlapping descriptive categories included 27 low-material cases (12 pieces or fewer), 19 with substantial pawn material, five with queens, and five involving a capture-gated endpoint. These categories are not causal diagnoses of missing learned relationships. The already Stockfish-confirmed settled labels were reused for analysis. These former held-outs remained excluded from the new labels.

## Selection and position results

Validation selection maximizes runtime ranking accuracy, then breaks ties by score MAE. Both runs retained epoch zero. Retraining at 5% reached the original validation accuracy only in later epochs, with a worse tie-break; retraining at25% remained below the original accuracy. Saved checkpoints have the original learned tensors, although serialization and blend metadata produce different file hashes.

| Configuration | New training correct | Validation correct | Test correct | Test hard correct | Test regressions from heuristic |
|---|---:|---:|---:|---:|---:|
| Original5% / selected5% |136/207|51/78|44/67|3/25|1|
| Original25% / selected25% |129/207|54/78|42/67|10/25|10|
| Heuristic |133/207|48/78|42/67|0/25|0|

The 25% follow-up was recorded after the 5% validation failure, based on correction range and the established25% blend. It used the same splits and inference. Test results are descriptive and did not choose checkpoints or the follow-up. These targeted near-tie accuracies are not general board accuracy or evidence of strength.

## Fresh strength pilot

**8 wins, 2 draws, 10 losses:45% score**,20 completed games/10 paired starts; zero errors, unfinished games, or interruptions. The60% gate failed, so no100-game extension, secondary oldNN match, or Stockfish opponent match ran. Candidate: selected25%, which retained the original NN weights, v20 incremental/fast inference. Opponent: unchanged heuristic via frozen v20 inference with correction0. Equal250 ms/move, depth8 cap, sequential paired colors, no book after10 fresh prefixes, 1,000-ply limit. Opening seed212500. Openings exclude previous benchmark starts and new training/held-out label aliases.

The declared extension threshold is60% across20 complete, error-free pilot games. Only if passed:80 games on40 fresh openings, reported separately and combined; secondary oldNN20 games. A claimed repeatable success also requires confirmation score above50% and combined score at least60%. No Stockfish opponent test or automatic app promotion.

The small pilot provides no evidence of a repeatable advantage and does not establish Elo. Time limits are cooperative: mean candidate elapsed243.5ms, heuristic241.4ms; maximum overruns29.8ms and371.9ms respectively; fallback root moves2/0. The isolated heuristic overrun is recorded, not evidence of a general speed or strength difference.

## Verification

Final audit passed: no cross-split canonical positions or source families;499 held-out FENs matched explicit PyTorch and fast integer runtime inference for all three checked configurations; all20 PGNs replayed legally to their recorded terminal results. Exact tensor comparison confirms both selected checkpoints retained the old learned weights. Search/inference/heuristic source hashes remained unchanged from the frozen snapshot.

## Reproduction and artifacts

- `python -m ml.run_generalization_v21` generates initial data, automatically performs the fixed pre-fit coverage extension if needed, prepares data, and runs5% training. Existing results are preserved; use a fresh output directory to reproduce rather than overwriting these files.
- This recorded run used `python -m ml.extend_generalization_v21`, then `python -m ml.run_generalization_v21 --resume-after-coverage` after the initial coverage exception. That sequence occurred before any fitting.
- `python -m ml.compare_generalization_v21` records25% training/selection and fresh heuristic games.
- `python -m ml.audit_generalization_v21` checks family and canonical board isolation, explicit PyTorch versus fast integer inference, and legal terminal game replay.
- Data: `ml/data/generalization-v21-2026`, new labels in `generalization-v21-new`, raw extension in `generalization-v21-extension`.
- Protocols, full source games, diagnostics, training history, frozen selection, match JSON/PGN and audit: `ml/artifacts/generalization-v21`.
- Unit verification:105 tests passed in6.107s with CPU lock free. Earlier suite attempts while the experiment held the lock timed out in benchmark subprocess tests; rerunning after release passed. No timing tests were altered.
