# v22: actual search leaves and smooth protected training

The new trainer selected a genuinely updated checkpoint (epoch1). Validation ranking correctness improved from148/212 to153/212 withzero regressions fromthe old25% NN. This is a position-level improvement, not yet evidence of stronger play. The app retains the heuristic.

## Fixed components

Architecture remains892 inputs,64/32 hidden units, bounded250cp residual. Runtime uses exactly25% correction outside check and legal captures, whole-score integer rounding, and unchanged v20 incremental/fast inference. Search and heuristic source hashes are frozen. No inference optimization or larger network was introduced.

## Search-matched data

- Sampled306 root positions across102 first-four-ply opening families. Historical heuristic/NN game roots supply actual match situations; prior560 book-source games supply broader family exposure. The latter are searched with our evaluators, rather than using their teacher positions directly as evaluation examples.
- Ran both heuristic andold25 searches to depth3 with2,000-node collection cap. Reservoir-sampled40 actual quiet static evaluator calls/search, canonical deduplicated, then selected atmost24/family. Collector depth/node cap is for sampling, not a strength or latency claim.
- Saved2,432 quiet FEN calls:600 from match roots,1,832 from book-source roots searched by our engine. Current adaptation families retain one train/validation/test assignment across colors, evaluator variants and source games.
- Generated Stockfish alternatives at24,000 nodes, settled captures/promotions/check evasions at12,000 nodes (max17 continuation plies), and obtained last-exact endpoint scores at64,000 nodes. Rankings require>=50cp teacher separation and an attainable5cp margin under the actual rounded25% correction. One quarter of processed quiet roots also supplies ordinary score labels.
- New labels:307 train rankings/769 score rows across45 usable families;212 validation rankings/482 rows across26 families;212 test rankings/481 rows across27 families. All previously labeled canonical roots/endpoints/FENs are excluded. Retained prior training examples; prior held-outs do not enter adaptation training.
- Legacy training opening-family provenance is incomplete. New-source family isolation and canonical non-overlap are verified, but this is not proof of globally unseen opening families for pretrained weights or all retained legacy data.

## Smoother learning and retention

One candidate, initialized from oldv8, unchanged architecture. Learning rate.00005, AdamW, seed42, maximum30 epochs/patience8.

- Score loss: MSE between sigmoid(predicted hybrid score/400) and sigmoid(teacher score/400), weight4. This is a monotone expected-score proxy, not calibrated Stockfish WDL probability or actual game-outcome supervision.
- Ranking loss: smooth binary cross-entropy between signed hybrid gap/100 logits and sigmoid(teacher gap/100); teacher gaps capped at500cp for this target.
- Protection: weight8 hinge against losing an oldNN-correct ranking, preserving up to10cp of its continuous hybrid margin. Nonpositive reference margins do not receive that protection; exact rounded oldNN regressions are measured separately during validation selection.
- Anchor: weight2 MSE penalizes correction drift fromoldNN on score examples.
- Loss uses continuous hybrid scores to provide smooth gradients; validation uses exact whole-endpoint integer rounding and runtime quiet gating.
- Best **updated** epoch is selected by validation correct count minus2 regressions fromoldNN, then correct count, then probability-proxy MSE. The unchanged original remains a separately measured baseline and is never silently substituted. Selected epoch1:153/212 versusoriginal148/212,0 oldNN regressions. Later epochs showed more regressions, so they were rejected.

## Equal-depth decision screen

A model-independent input-selection rule was fixed before fitting: random seed220023, novel test-family roots, exclude every known labeled canonical root/endpoint/FEN, max2 roots/family, max32 total. This initial process materialized the list after fitting without consulting candidate outputs; training did not alter the JSONL exclusion set. The current runner also materializes the same rule before fitting. The case list is saved for review.

All32 positions were nonmate-score comparable. Each evaluator completed exact depth3 with no clock/node cap and full root game history. Stockfish analyzed the same root with64,000 nodes, and separately forced each distinct chosen move at64,000 nodes, consuming the last exact scored PV. Regret is a teacher estimate, not a proved move value; this comparison does not isolate static evaluation from all search interactions.

| Evaluator | Mean regret(cp) | Moves losing>=100cp | Newly allowed mate |
|---|---:|---:|---:|
| Heuristic |82.34375|7|0|
| Old25% |83.4375|8|0|
| Trained25% |83.4375|8|0|

The candidate chose the same move asoldNN inall32 cases. Compared withthe heuristic, bothNNs improved two choices and worsened one; total regret was35cp worse across32 positions. **The original strict game-qualification gate failed:** no improvement over the heuristic.

## Exploratory fresh games

Because validation improved withoutoldNN regressions, a recorded amendment allowed one exploratory20-game pilot despite the failed strict screen. The amendment was made after the screen and before any game outcomes; it is not represented as the original gate passing. Equal-time behavior can differ from a fixeddepth3 check.

Candidate: updated epoch1,25% quiet correction, fixedv20fast inference. Opponent: unchanged heuristic through frozen inference atcorrection0.250ms/move, depth8 cap,10 fresh paired starts, no book after prefix, max1,000 plies. Seed222500; exclude prior benchmark starts and new dataset label aliases. Only20 complete, error-free games with score>=60% trigger80 new confirmation games, reported separately and combined. Noautomatic app promotion.

**6 wins,3 draws,11 losses:37.5% score against the heuristic**,20 completed games/10 paired starts. Zero errors, unfinished games or interruptions. The60% threshold failed, so no80-game confirmation,100-game combined result, secondary oldNN match, or Stockfish opponent match ran. No demonstrated playing-strength gain; app remainsheuristic. Previous pilots used different openings, so their percentages cannot establish that this candidate is better or worse thanoldNN.

Budgets were cooperative: candidate mean244.3ms/move, heuristic242.8ms; maximum overruns69.9ms/48.0ms; fallback root moves0/1. No hard real-time claim.

New test rankings: oldNN142/212, candidate143/212, heuristic139/212. Candidate fixed15 heuristic-wrong rankings but introduced11 regressions fromthe heuristic. AgainstoldNN it fixedtwo rankings and introducedone regression. The net test gain is one ranking, far too small to claim a robust improvement. Validationcorrectness153/212 is likewise a screening result, not game strength.

## Verification and files

-108 unit tests passed in5.165s, including directional smooth-loss gradients, protection behavior and disabled-endpoint gradient masking.
- Pre-game audit: no cross-split canonical positions/source IDs; new labels overlap no prior label aliases;963 held-out FENs match explicit PyTorch and fast integer inference forold andtrained25% withzero mismatches; selected weights changed; frozen architecture/search/inference/heuristic source hashes unchanged; no move-screen root overlaps training aliases.
- Final audit passed: all20 PGNs replayed legally to their recorded terminal results. Source stability, alias isolation, explicit inference parity and changed learned weights were confirmed.
- Scripts: `ml/run_search_v22.py`, `ml/train_search_v22.py`, `ml/play_search_v22.py`, `ml/audit_search_v22.py`.
- Data: `ml/data/search-smooth-v22-new`, combined retention data`ml/data/search-smooth-v22-2026`.
- Protocols, source roots, quiet calls, checkpoint, training history, decision screen, amendment, games and audit: `ml/artifacts/search-smooth-v22`.
- Run with`.venv/bin/python -m ml.run_search_v22` against a fresh output path; existing artifacts are preserved. The exploratory pilot is separately runnable with`-m ml.play_search_v22` only when its recorded validation and oldNN nonregression checks hold.
