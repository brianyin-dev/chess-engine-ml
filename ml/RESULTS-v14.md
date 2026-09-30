# Compact evaluator: cost and decision quality

Status: completed. The compact candidate failed both screens and is not promoted.
The app uses the unchanged heuristic.

## Implementation

The compact candidate uses 794 board/material/rule-state features and hidden
layers of 32 and 16 units, versus v8's 892 features and 64/32 units. It avoids
constructing the 98 attack/king/pawn relationship features at every active leaf.
Checkpoints store hidden widths; old checkpoints retain their original defaults.

Initialization projects the old checkpoint by retaining units with the largest
absolute downstream weights, then removes relationship input columns. This uses
only model parameters, not validation/test examples. The projected model is
trained on the existing game-separated Stockfish score and settled-move ranking
examples, including confirmed failures. Both score and ranking objectives apply
the exact 25% quiet-only deployment correction. Correct heuristic rankings get
extra protection; an anchor penalizes drift from the projected initialization.
The initial model remains an epoch-0 candidate. Selection uses validation integer
ranking, then validation MAE. No epoch is selected from game or test results.

The reused test set is a development holdout. This is not an untouched final
research test. Inference parity and color symmetry are checked separately.

## Diagnostic protocol

`ml.diagnose_compact` samples 32 nonterminal development-test positions with a
fixed seed. All evaluators use the same search algorithm: depth 3 without a clock,
and separately depth cap 8 with 250 ms per move. Full-strength Stockfish depth 12
scores selected moves for approximate regret. Restricted-root scores can differ
from unrestricted-root analysis, so small regret differences are not conclusive.
This teacher analysis is not a Stockfish strength match or a rating estimate.

Evaluation timing alternates seven trials on 1,024 positions including color
mirrors. Report: `ml/artifacts/compact-v14/diagnosis.json`.

## Fresh game screening

Freeze 50 new book-derived starts before games; exclude starts from previous
benchmarks and canonical aliases of labeled positions and ranking endpoints.
First ten starts supply 20 paired-color games against the frozen heuristic, then
20 against v8 with its actual 25% quiet correction. Both matches use the same
starts, 250 ms cooperative budgets, and no book after the opening prefix.
CPU jobs share the checkout's exclusive benchmark lock.

Extend the heuristic match to 100 games on the other 40 starts only if all pilot
games finish, the candidate scores at least 60% against the heuristic and above
50% against v8. No automatic app promotion or external opponent games.
Coordinator: `ml.run_compact_v14`; status: `ml/artifacts/compact-v14/status.json`.

## Measured results

Complete evaluation median time is 0.10337 ms for old v8, 0.06899 ms for compact,
and 0.02671 ms for the heuristic on the identical timing sample. Compact is
33.3% faster than old v8 but still costs 2.58 times the heuristic. Parameter count
falls from 59,265 to 25,985 (56.2%). These are evaluation microbenchmarks, not
playing-strength claims.

| Diagnostic on 32 positions | Heuristic | Old v8 | Compact |
|---|---:|---:|---:|
| Equal depth: mean teacher regret, cp | 55.2 | 50.8 | 57.2 |
| Equal time: mean teacher regret, cp | 54.8 | 69.9 | 56.0 |
| Equal time: mean completed depth | 3.94 | 3.44 | 3.62 |
| Equal time: mean visited nodes | 12,380 | 8,389 | 9,386 |

At depth 3 the old NN has slightly lower teacher regret; with equal time it has
higher regret and searches less deeply. This supports investigating inference
cost, but the sample and teacher depth are too small for a definitive attribution.
Compact visits 11.9% more nodes than v8 at equal time, recovering part of the gap.

Rounded test ranking is 522/683 (76.4%) versus v8's 512/683 (75.0%); validation is
491/694 (70.7%) versus v8's 495/694 (71.3%). The candidate reverses 9 heuristic-correct
test rankings versus v8's 14, but 10 validation rankings versus v8's 4. It learns
43/74 confirmed training pairs versus v8's 41/74. On the most recent 41 confirmed
pairs, both get 21 correct. Prediction accuracy is mixed; more data alone is not
a demonstrated solution. On 512 positions, there are zero integer inference
mismatches and zero color-symmetry error.

The compact model finished the heuristic screen at **6 wins, 4 draws, 10 losses:
40%**. All 20 games and 10 color pairs completed with no errors or unfinished
games. It failed the predeclared 60% extension gate; no 100-game or Stockfish
1500 match is justified. Direct v8 comparison finished at **8 wins, 2 draws,
10 losses: 45%**, with all 20 games and 10 pairs complete and no errors or
unfinished games. This does not establish a strength gain over v8. The checkpoint
remains experimental; smaller/faster inference alone did not meet the game gate.

All **97 tests pass**, including compact checkpoint inference parity, legacy
checkpoint compatibility, color symmetry, and projection connection preservation.
Complete PGNs and per-move reports are saved for both matches.

Projection is reproducible with `python -m ml.project_compact --source
ml/artifacts/quiet-ranking-v8.pt --output NEW.pt`. Training uses `ml.train` with
`--features board --hidden-sizes 32 16 --correction-weight .25 --quiet-only`,
retaining the same game-separated supervision. Exact hyperparameters and source
hashes are recorded in `compact-v14/training.json` and `projection.json`.

## Diagnosis of actual losses and drawn wins

Full-strength Stockfish reviewed the 10 losses and 4 draws from the heuristic
screen at 100 ms, confirming first costly moves at 400 ms. Twelve games had a
confirmed mistake, including 2 of the 4 draws with a confirmed conversion loss.
A longer one-second compact search recovered 8 of those 12 mistakes under the
same teacher criterion. The mixed evaluator still ranked the bad immediate
continuation above the better one in 9 cases; this includes the heuristic and
quiet-only gate, not necessarily an error introduced by the NN alone.

On these 12 confirmed positions, the equal-depth-3 diagnostic gives mean teacher
regret of 127.2 cp for heuristic, 93.5 cp for old v8, and 93.5 cp for compact. At
250 ms, completed depth averages 2.50, 2.17, and 2.17 respectively; mean regret is
292.1, 235.0, and 301.7 cp. The compact model's equal-depth choices are substantially
better than its equal-time choices on its known failures. This reinforces the
cost concern but does not establish speed as the only cause. Sampling the compact
model's failures biases this diagnostic; it is not independent strength evidence.
FEN reconstruction also omits repetition history. Source reports and teacher
hashes are retained in `losses-and-draws.json` and `loss-diagnosis.json`.

The next experimental target is preserving useful evaluator information while
reducing repeated feature construction, ideally with incremental updates. This
round did not produce a stronger checkpoint and did not alter the app evaluator.
