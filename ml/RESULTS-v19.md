# Balanced supervision and a development-selected correction weight

This experiment diagnoses the failed v18 model, broadens supervision, and selects a 5%, 10% or 25% quiet-position correction on development validation before fresh games. The app stays on the heuristic. The frozen 5% configuration scored 52.5% against the heuristic and 57.5% against old v8 at 25% across 20 games each. Training selected epoch zero: no learned-weight improvement was obtained.

## Loss diagnosis

Stockfish confirmed a first >=100 cp mistake or mate deterioration in each of the 24 v18 losses. There are 19 distinct FENs, so these are correlated, selected failure cases, not general strength measurements. All three probes use incremental inference here to compare weights under the same implementation; the final old-NN game baseline retains its original nonincremental inference.

| Evaluator | Mean regret at exact depth 3 | Mean regret at 250 ms | Mean completed depth at 250 ms |
| --- | ---: | ---: | ---: |
| Heuristic | 192.7 cp | 222.0 cp | 2.33 |
| Old v8 at 25% | 191.9 cp | 221.9 cp | 2.33 |
| Failed v18 at 25% | 191.9 cp | 267.0 cp | 2.25 |

The old NN and failed v18 selected the same move in all 24 exact-depth probes. Six v18 250 ms errors dropped below 100 cp regret at depth three. All 24 scores are comparable in centipawns. At depth three, each evaluator made a confirmed costly move in 18 cases. Under 250 ms, the counts were 22, 22 and 24. This supports examining time-limited search behavior; it does not prove pure inference speed causes the regression. Fixed-depth search also includes move ordering and pruning, and the teacher scores are estimates. The recorded game history was retained in diagnostic searches.

## Protocol

Start from the unchanged v8 checkpoint rather than failed v18. Retain previous training score and ranking examples, including confirmed mistakes. Add ordinary quiet teacher-game score labels, teacher-confirmed candidate alternatives after tactical settling, and extra importance for old-correct rankings. New labels use exact 64,000-node endpoint scores; root alternatives receive 200 ms Stockfish confirmation. Tactical settling follows teacher-selected captures, promotions and check evasions; other legal captures can remain, and at least one endpoint must activate the quiet gate.

Initial collection uses 120 fresh common-book starts, whole-game splits of 80 training/20 validation/20 test games, up to eight sampled roots per source game. A fixed extra 120-start batch is allowed only if either holdout has fewer than 30 pairs, before any candidate training/scoring. All prior dataset aliases are excluded from new examples, including color mirrors; new source games own their positions within one split. The model size is unchanged.

One candidate trains with the existing rounded 25% hybrid objective, a smaller learning rate, stronger anchoring and greater protection of heuristic-correct pairs. Validation selects the epoch, then selects one of 5%, 10% and 25% by ranking accuracy, fewer heuristic regressions, then smaller weight. The selection and checkpoint hash are saved before test scoring or games. The trainer also produces its standard fixed-25% test metric; this never enters selection. New validation and test are separate source games; prior heldouts are not included in training.

Fresh games use the chosen frozen weight: 20 paired games against optimized heuristic 0%, then 20 against unchanged old v8 25%, same openings and 250 ms/depth-eight budgets, sequentially under the shared CPU lock. No app promotion or external Stockfish opponent match is automatic.

Evidence and reproduction: `python -m ml.run_balanced_v19`, protocols and reports under `ml/artifacts/balanced-v19`, source labels under `ml/data/balanced-v19-new`, and combined training under `ml/data/balanced-v19-2026`.

## Training and development results

The initial batch met coverage; no supplemental batch was needed. New data adds 437 training score rows and 193 training pairs, of which 155 rankings were already correct under old v8 and four were new confirmed-loss pairs that survived the filters. Combined training contains 13,197 score rows and 4,186 pairs. Fresh validation has 101 rows/45 pairs; test has 73 rows/31 pairs.

Validation selected **epoch zero**: the starting v8 weights. Epochs 1–5 did not improve ranking accuracy, and their score error was worse. This round therefore produced no learned-weight improvement. Calling the selected checkpoint a newly stronger trained NN would be misleading.

| Development correction | Validation correct / 45 | Heuristic-correct regressions |
| --- | ---: | ---: |
| 5% | 41 | 0 |
| 10% | 41 | 0 |
| 25% | 40 | 1 |

The predeclared tie rule chose 5%. Its checkpoint hash and weight were frozen before test scoring and games. On the fresh test, the chosen 5% configuration scored 22/31, old v8 at 25% scored 25/31, and the heuristic scored 22/31. The 5% configuration corrected none of the nine heuristic-wrong test rankings, whereas old 25% corrected three. The development gain came from avoiding a regression, not learning new corrections; it did not generalize as a ranking gain on this test. Games are still run as the requested controlled weight experiment, with no weight changes based on test results.

The initial sampling chose roots every eight plies after even-length book prefixes, so these new ranking roots are White to move. The evaluator is color consistent, but this remains a limitation of source diversity; prior retained training includes other positions. All 102 existing unit tests passed before the run.

## Fresh game screens

The heuristic screen completed 20 games: **8 wins, 5 draws, 7 losses (52.5%)**, ten complete opening pairs, zero errors or unfinished games. This is approximately even in a small sample and below the 60% target. It does not establish superiority over the heuristic, and comparisons with earlier screens use different opening sets.

The old-NN screen completed on the same ten new starts: **9 wins, 5 draws, 6 losses (57.5%)**, ten complete pairs, zero errors or unfinished games. Both configurations retain the v8 learned weights; this comparison tests 5% incremental correction against unchanged 25% nonincremental correction. It does not isolate correction weight from inference implementation and cannot demonstrate a retraining gain.

## Verification and conclusion

All 40 PGNs replay legally to their recorded terminal results. The selected checkpoint has parameters exactly equal to original v8. Incremental, NumPy and PyTorch inference agree on all 174 fresh validation/test score positions. There are zero canonical position or source-game overlaps across train/validation/test, zero new aliases shared with prior data, and zero validation/test aliases shared with the reviewed v18 games.

Mean elapsed move times against the heuristic were 246.1 ms for the candidate and 245.9 ms for the reference; maxima exceeded the nominal budget by 36/115 ms. Against old NN, means were 245.5/245.4 ms, with maximum overruns of 17/32 ms. No fallback moves occurred. These are nominal equal budgets, not hard real-time deadlines.

At 5% weight, the bounded quiet correction could potentially supply a 5 cp ranking margin for only one of four heuristic-wrong validation pairs and two of nine heuristic-wrong test pairs. At 25%, the respective upper-bound counts were three and seven. This is an upper-bound reachability check, not a promise that a network can learn those rankings. More layers cannot overcome the blend's fixed correction bounds.

The smaller correction configuration was competitive with the heuristic and scored positively against old 25% in this screen, but 20 games per opponent cannot establish a reliable advantage, and the heuristic score remains below 60%. The fresh ranking test was worse than old 25%, and training produced no better checkpoint. No 100-game extension, Stockfish opponent benchmark, or app promotion was performed.

The evidence favors profiling and reducing the cost of neural use before increasing model size, while retaining both difficult and ordinary examples for future training. It does not support claiming that broader retraining made the NN stronger. Both game comparisons test complete configurations; the old-NN comparison changes correction weight and inference implementation together.

Candidate artifact SHA256: `fbba93d3e0e8f70706413a10af60f88515c25c3363d2696535a28c4ceb87853b`. Original v8 SHA256: `32beb3b6d5231fe115b2f07ccbc6298c9d9f76a5d58fce27801f6e1d443237b0`.

Run `python -m ml.audit_balanced_v19` to verify the saved dataset, checkpoint and game records. The coordinator preserves existing experiments rather than overwriting them; frozen artifacts and commands in its source document the completed run.
