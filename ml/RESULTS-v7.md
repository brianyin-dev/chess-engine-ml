# King-aware features and targeted Stockfish distillation

This experiment follows the v6 consistency work with richer relationships, a
10,000-position training expansion, budgeted teacher labels, and separate
node-budget and clock-budget game tests. The app remains classical unless a
candidate establishes practical improvement and passes consistency checks.

## Representation and compatibility

Version-four checkpoints use 892 inputs: the existing 794 board/rule/material
features plus 98 relationship features. Each color has 49 features:

- Piece-type aggregates for proximity and offsets relative to its own and enemy king.
- Attacked, defended, and attacked-but-undefended counts for each non-king piece type.
- Isolated, doubled, and passed pawn counts, plus pawns shielding the king.

These are compact aggregates, not full pairwise king-piece embeddings. Original
board planes still describe individual squares. Attack features use pseudo-attacks,
including pinned attackers. Both colors share one side-to-move canonical network,
so color-mirrored predictions have opposite signs. Version-two and version-three
checkpoints retain their prior encodings and behavior.

The 64→32 hidden architecture is retained. Both candidates predict a bounded
±250 cp correction to the full heuristic. This restores more positional freedom
than v6's ±45 cp experiment; it does not guarantee material monotonicity.
The board-only control uses the same new data, ranking loss, correction limit,
sampling weights, seed, optimizer, and checkpoint-selection policy. The extra
input weights increase parameter count slightly, and one seed is not a complete
architecture ablation.

## Targeted data and modest compute

`ml.generate_targeted_data` selects roots from split-isolated v6 data, alternating
ordinary roots, high heuristic/teacher disagreement, and endgames. It reservoir
samples actual evaluator calls from a 1,500-visited-node search and adds teacher
candidate successors. Worse successor moves supply ranking examples; these are
not all actual moves made in lost games. Strong Stockfish self-play is not the
only position source.

New quotas are exactly 10,000 training, 1,000 validation, and 1,000 test positions.
Training has 4,057 endgame-search records, 4,336 other stand-pat records, 552 quiet
non-endgame records, and 1,055 teacher-candidate successors. Across all categories,
2,740 of the 10,000 positions have no legal capture. The combined split sizes are
36,277 / 4,989 / 5,095. There are 205 / 19 / 22 ranking pairs.

Each label first receives 500 then 1,500 Stockfish nodes. An additional 6,000 nodes
is used for unstable scores, close root candidates, or student/teacher disagreement.
Refinement is capped at 25% of each new quota. Labels are approximate searched
scores, clipped at ±1,500 cp, not static ground truth. FEN labels lose repetition
history. Canonical FEN/color mirrors and whole-game IDs are isolated across splits,
including candidate-pair endpoints. Label results are cached within generation;
outputs and provenance are saved for reuse rather than relabeling each experiment.

Generation took **278 seconds locally**, with **43.4 million requested Stockfish
nodes**, including teacher candidate analysis. Requested node limits are cooperative;
this is not a claim of exact Stockfish node consumption or a hardware-independent
runtime. No GPU or rented compute was used. Extra-budget cases exhaust their
quarter-quota partway through generation, so later ambiguous records can retain
the cheaper label.

## Training and diagnostic results

Both runs use seed 42, AdamW, Huber score loss plus mover-relative ranking loss,
2× sampling weight for search-derived rows, at most 40 epochs, and eight stale
validation epochs for stopping. Checkpoints minimize validation MAE plus 100 cp
times ranking error. Both selected epoch one; later epochs lowered training error
but did not improve the combined validation criterion. The validation ranking
set has only 19 pairs, making selection noisy.

| Evaluator | All 5,095 test positions MAE | Test ranking on 22 pairs |
| --- | ---: | ---: |
| Heuristic | 226.5 cp | 19/22 (86.4%) |
| Board-only control | 213.9 cp | 18/22 (81.8%) |
| King-aware candidate | 197.6 cp | 15/22 (68.2%) |

The new features improve this run's score prediction but not its tiny held-out
ranking result. Neither metric substitutes for games. A separate diagnostic audit
of 512 positions found exact training/inference agreement and zero color-symmetry
error for both candidates. Both passed 7/7 tactical search checks. The king-aware
candidate had four wrong-direction pawn-removal probes out of 195, versus none
for the control; all sampled knight/bishop/rook/queen probes had the expected sign.
Removing pieces also changes lines and positional relationships, so these probes
are useful diagnostics rather than universal proofs of bad moves.

The audit sampled 480 actual static calls at 24 reconstructed game roots, searching
with the king-aware model. Error was 320.4 cp versus the heuristic's 371.4 cp on
this tree. On the 72 quiet, unsaturated subset, the candidate was worse:
247.8 cp versus 217.2 cp. Exact sample count is recorded in the audit artifact.
These diagnostic roots are not a clean held-out strength test, and results from
different candidates' search trees are not a direct before/after comparison.

## Inference cost

Feature extraction dominated profiling. Precomputed king geometry and shared
attack masks reduced the median standalone candidate cost from **0.1752 ms to
0.1137 ms per position**, about **35%**, while producing identical feature vectors
on 1,019 diverse held-out boards. Final comparison: heuristic **0.0353 ms**,
board-only control **0.0707 ms**, candidate **0.1137 ms**. This measures standalone
calls, not whole-engine throughput. The model is not a full incrementally updated
NNUE implementation; feature extraction still runs at each uncached evaluation.

## Game policy

Use four fresh opening families with paired colors and no opening book, depth cap
eight and 200 played plies. Run sequentially, without labeling/training/profile
work alongside timed games. Equal-node tests allocate 2,000 visited nodes per move,
including main and quiescence visits; fallback static calls are counted separately.
No clock limit applies. Equal-time tests allocate 250 ms cooperatively per move.
Unfinished games do not count as draws, and these suites do not imply an Elo rating.

At **2,000 nodes per move**, the candidate scored **0 wins, 6 losses, 2 unfinished**.
All six completed games were checkmates. This rejects the explanation that slower
inference alone causes its weaker play. Mean recorded search time was 75 ms for
the heuristic and 226 ms for the NN; these are diagnostics for this suite, not a
standalone throughput benchmark. One heuristic move used a fallback because no
iteration completed within its visited-node budget.

## Reproduction

```sh
.venv/bin/python -m ml.generate_targeted_data \
  --data ml/data/search-v6-2026 \
  --checkpoint ml/artifacts/search-residual-v6.pt \
  --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal \
  --output ml/data/targeted-v7-2026

.venv/bin/python -m ml.train --data ml/data/targeted-v7-2026 \
  --pairs ml/data/targeted-v7-2026/pairs \
  --checkpoint ml/artifacts/relational-v7.pt --metrics ml/artifacts/relational-v7.json \
  --target residual --features relationships --color-consistent \
  --correction-limit-cp 250 --search-weight 2 --epochs 40 --patience 8

.venv/bin/python -m ml.train --data ml/data/targeted-v7-2026 \
  --pairs ml/data/targeted-v7-2026/pairs \
  --checkpoint ml/artifacts/board-control-v7.pt --metrics ml/artifacts/board-control-v7.json \
  --target residual --features board --color-consistent \
  --correction-limit-cp 250 --search-weight 2 --epochs 40 --patience 8

.venv/bin/python -m ml.compare --checkpoint ml/artifacts/relational-v7.pt \
  --openings benchmarks/openings-neural-v7.json --pairs 4 --nodes 2000 \
  --max-plies 200 --output ml/artifacts/relational-v7-vs-heuristic-2000nodes

.venv/bin/python -m ml.compare --checkpoint ml/artifacts/relational-v7.pt \
  --openings benchmarks/openings-neural-v7.json --pairs 4 --time-ms 250 \
  --max-plies 200 --output ml/artifacts/relational-v7-vs-heuristic-250ms
```

Use new output paths for reruns. Checkpoint and dataset hashes, timings, exact
labels and budgets, audit FENs, match PGNs, and partial/final game summaries are
saved alongside the corresponding artifacts.

## Final outcome and loss review

At **250 ms per move**, the candidate scored **0 wins, 8 losses**, all completed.
Neither suite supports promotion, and the app keeps the heuristic. Better average
score prediction did not translate into better chess in this experiment.

A Stockfish review screened moves at 100 ms and confirmed the first ≥100 cp loss
or mate deterioration at 400 ms in **all eight losses**. In **six** reviewed cases,
the NN's immediate static scores ranked the played successor at least as high as
the teacher's better successor. In **three**, a one-second NN search recovered
below the costly-move threshold. These categories overlap; immediate static
ranking is not a complete quiescence/search diagnosis, and extra time probes are
not definitive causal attribution. The saved analysis contains exact FENs and
better/worse pairs for focused future experiments. None of these new match-loss
pairs were used to train or select the evaluated model.

The evidence prioritizes robust move-ranking and quiet-position supervision over
another indiscriminate data expansion. Feature speed matters, but equal-node losses
show it cannot by itself explain or fix the current weakness. The training pipeline
and artifacts remain useful even though this candidate is rejected.

Verification: **81 unit tests pass**, including legacy checkpoint loading, relational
feature symmetry, training/NumPy inference parity, attacked/defended features,
node-budget validation, and board/history restoration. An independent data check
confirms disjoint game IDs and canonical positions/color mirrors across all splits,
including ranking pairs.
