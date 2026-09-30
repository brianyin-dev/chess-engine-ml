# Targeted mistakes and runtime-aligned hybrid training

The candidate learned new training rankings but failed the fresh-position improvement gate. It was not promoted and no candidate game screens, 100-game extension, or Stockfish opponent games were run. The app retains its heuristic; v8 at 25% remains the unchanged experimental reference. This round does not establish a playing-strength improvement.

## Confirmed mistakes and search-depth diagnostics

Reviewed the eight losses/two draws of incremental v8 at 25%, plus nine losses/two draws of the old nonincremental 25% control from v16. Each game's checkpoint hash and recorded blend/gate/incremental setting were used. Stockfish screened moves at 100 ms and confirmed qualifying mistakes at 400 ms per root search, using full-strength single-thread analysis.

- 19 of 21 reviewed games had a confirmed >=100 cp mistake or mate deterioration.
- Two of four draws lost a >=150 cp evaluated advantage to <=50 cp, or missed a forced mate. Centipawn advantage alone does not prove a forced win.
- At a one-second search budget, the NN avoided a qualifying error in eight of the 19 positions. This diagnoses time sensitivity; it does not prove a single cause.
- At fixed depth three without a clock, mean Stockfish move regret was 178.3 cp for optimized 0% and 200.3 cp for the source NN configuration on these 19 selected mistakes. All were comparable in centipawns; neither probe allowed or missed a forced mate under the teacher review. This selected failure set is not a general strength benchmark.

Evidence: `ml/artifacts/targeted-v17/failures.json`, `equal-depth-summary.json`.

## Implementation and training

Training now rounds the **full baseline + correction endpoint score**, retaining gradients through a straight-through approximation. Ranking validation also rounds the full endpoints, including half-centipawn ties that depend on baseline parity. The objective uses exactly 25% residual correction and the runtime quiet gate. Architecture, bounded correction, search, and inference implementation are unchanged.

Added 800 training pairs and 120 pairs to each existing development validation/test split; 46 new training pairs came from confirmed failures after teacher-guided tactical settling. Prior data and rankings were retained, with 4x weight for consequential confirmed failures, protection of heuristic-correct pairs, and anchoring to the original checkpoint. Whole-game IDs and canonical mirrored position aliases remain separate across splits; root positions are now protected too.

Final training data: 12,497 score rows/3,848 ranking pairs; development validation: 2,094 rows/808 pairs; development test: 2,116 rows/795 pairs. These existing development splits have been used in prior experiments and are not fresh generalization evidence.

One candidate was trained with learning rate 0.0001, rank weight 4, 5 cp margin, hard-pair weight 2, protected-pair weight 3, score weight 0.1, anchor weight 0.5, maximum 25 epochs and patience six. Checkpoint selection used development validation ranking, then continuous score MAE for ties, with the initial checkpoint eligible as epoch zero. Epoch 11 was selected: development validation ranking went from 70.42% to 70.67%; test ranking was 74.21%.

| New training subset | Old 25% correct | Candidate 25% correct |
| --- | ---: | ---: |
| All 800 new pairs | 521/800 (65.1%) | 648/800 (81.0%) |
| 46 confirmed-failure pairs | 27/46 | 30/46 |
| 145 bounded-correction-attainable heuristic mistakes | 35/145 | 143/145 |

These are training-fit diagnostics, not strength measurements. Of 294 new pairs the heuristic ranked incorrectly, 149 cannot be reversed under the current correction bound and quiet gate even with an ideal network. This constraint and the failure to transfer training gains warrant investigation; neither alone establishes the cause of weak play.

Evidence: `training.json`, `targeted-learning.json`, `ml/data/confirmed-v17-2026/manifest.json`, `ml/data/aligned-v17-2026/manifest.json`.

## Fresh-position check and qualification decision

The original holdout was frozen before training from 30 independent book starts followed by teacher self-play. Its strict capture-free filter yielded only two validation/three test pairs. All models ranked those five correctly. Training completed and the pipeline scored that tiny set before the expansion safeguard was applied; preserve that initial result in `fresh-validation-small.json` and `status-small.json`.

A supplementary holdout was then frozen after training and the initial tiny check, **without changing or retraining the candidate**. It used 60 further book starts (seed 170002), fixed teacher sampling, and seeded diverse legal alternatives. At least one endpoint must activate the quiet gate; teacher continuations settle captures, promotions and check evasions. Root analysis uses 24,000 nodes and settling steps 12,000 nodes. Pairs need >=50 cp gaps. All prior and newly trained score/pair/root aliases, plus reviewed game positions, were excluded. Entire source openings were assigned to validation or test. This is supplementary evidence, not an untouched first test or proof of game strength.

| Model, exact quiet hybrid | Supplementary validation | Supplementary test |
| --- | ---: | ---: |
| Optimized heuristic, 0% | 62/68 (91.2%) | 52/60 (86.7%) |
| Unchanged v8 NN, 25% | 60/68 (88.2%) | 52/60 (86.7%) |
| Candidate NN, 25% | 59/68 (86.8%) | 52/60 (86.7%) |

The candidate corrected one of six heuristic-wrong validation pairs, but regressed on four heuristic-correct pairs versus two for the old NN. It failed the strict validation improvement/nonregression gate. The holdout contains 28 source games per split; positions from the same game are correlated. Approximate teacher labels and the predominance of heuristic-correct pairs limit what the scores establish.

The implemented next stage remains available for eligible future candidates: fresh paired openings, 20 games versus optimized 0%, then 20 versus the unchanged old nonincremental NN, both at 250 ms/depth eight. Expand to 100 total games versus 0% only for >=60% against 0% and >50% against the old NN, with all games complete and error-free. Pilot and extension would be reported separately. No app promotion is automatic.

Evidence: `fresh-validation.json`, `fresh-holdout-expanded.json`, `holdout-amendment.json`, `protocol.json`, `status.json`.

## Verification and reproduction

All 101 tests passed, including full-score rounding and gradient preservation. Integrity checks found zero training/holdout canonical alias overlap, zero validation/test aliases or game-ID overlap. All 256 fresh endpoints produced identical integer evaluations using separate Torch model and incremental NumPy engine inference. No tolerance was needed in the observed results.

Commands and settings are recorded in the status/protocol artifacts. `ml.run_targeted_v17` runs the pipeline after the loss review; fresh outputs are required. Its one-time `--resume-small-holdout` option preserves the undersized original result before supplementary validation; it does not retrain. Normal runs now expand undersized holdouts before scoring the candidate. `ml.analyze_losses`, `ml.fresh_holdout_v17`, and `ml.validate_targeted_v17` preserve existing result files.
