# Color-consistent, search-informed neural candidates

This experiment addresses the findings in [the v5 audit](AUDIT-v5.md).
The app evaluator remains classical unless local paired matches establish an improvement.

## Changes

New checkpoints opt into a canonical side-to-move representation: Black-to-move
positions are rank-flipped and color-swapped to make the mover White. Network
outputs are converted back to White's perspective. This enforces opposite scores
for color-mirrored positions without evaluating the network twice. Training and
NumPy inference implement the same transformation. Old checkpoints keep their
original behavior; loading them does not silently change their evaluation.

Both new candidates bound learned corrections to ±45 cp. The material candidate
adds that correction to fixed material values. Thus removing an enemy pawn worth
100 cp increases its score by at least 10 cp, regardless of the network's response.
This deliberately restricts positional expressiveness; it is a conservative sanity
experiment, not a claim that all positional advantages are smaller than a pawn.
The residual candidate retains the full heuristic evaluation and adds a small NN
correction instead. The material monotonicity guarantee does not apply universally
to this candidate because the heuristic also changes when pieces are removed.

## Data and training

The reproducible `ml.generate_search_data` command samples depth-three search's
actual static evaluation calls using the previous ranked NN, then labels them
independently with Stockfish at depth ten. These are quiescence stand-pat calls,
not necessarily final quiet leaves. Roots come from the existing self-play splits,
and generated positions inherit the root game's split. Canonical FENs, including
color mirrors, cannot occur in more than one split. Preexisting aliases are removed
before sampling. FEN labels cannot reconstruct repetition history.

The dataset has 25,478 base + 799 search training positions, 3,190 + 799 validation
positions, and 3,298 + 797 test positions. Total new search labels: 2,395. Only 799
are training examples. Validation and test positions are not used for optimization.

The material model uses uniform sampling; the residual candidate gives search-derived
training rows eight times the sampling weight. Both use seed 42, at most 30 epochs,
early stopping after six stale validation epochs, and validation MAE for selection.
No candidate-ranking pairs are used in this experiment.

## Reproduction

```sh
.venv/bin/python -m ml.generate_search_data \
  --data ml/data/stockfish-book-deviations-v4-2026 \
  --checkpoint ml/artifacts/material-ranked-v5.pt \
  --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal \
  --output ml/data/search-v6-2026 --roots-per-split 40

.venv/bin/python -m ml.train --data ml/data/search-v6-2026 \
  --checkpoint ml/artifacts/search-material-v6.pt \
  --metrics ml/artifacts/search-material-v6.json --target material \
  --color-consistent --correction-limit-cp 45 --epochs 30 --patience 6

.venv/bin/python -m ml.train --data ml/data/search-v6-2026 \
  --checkpoint ml/artifacts/search-residual-v6.pt \
  --metrics ml/artifacts/search-residual-v6.json --target residual \
  --color-consistent --correction-limit-cp 45 --search-weight 8 \
  --epochs 30 --patience 6
```

Outputs must use fresh paths when rerunning. Each checkpoint records the color
consistency flag and correction limit; metrics record the dataset hash.

## Prediction and consistency results

| Candidate | All 4,095 test positions MAE | 797 held-out search positions MAE |
| --- | ---: | ---: |
| Heuristic | 213.5 cp | 345.4 cp |
| Material + bounded NN | 193.8 cp | 327.4 cp |
| Heuristic + bounded NN | 202.3 cp | 327.4 cp |

A separate repeat of the audit sampled 512 test positions and 480 actual evaluation
calls from 24 reconstructed game roots, searching with the new material model.
Both candidates had **zero color-symmetry error**, identical training/inference
rounded scores in all 512 comparisons each, positive material-removal gains in all
sampled probes, and passed **7/7 tactical search checks**. The material candidate's
minimum sampled pawn gain was 88 cp; the residual candidate's was 7 cp.

On the 480 newly sampled search calls, MAE was 283.5 cp for the material candidate,
287.0 cp for the residual candidate, and 303.8 cp for the heuristic. This is a small
prediction advantage, not a playing-strength result. These positions were sampled
from a different model's search tree than the v5 audit, so its 313/316 cp numbers
are not a direct before/after comparison. Some diagnostic audit positions may also
overlap training; the separate 797 held-out search positions are the clean data
split comparison. Exact audit outputs live in
`ml/artifacts/consistency-leaf-audit-v6/`.

## Local games and app decision

Matches use the same four opening pairs, both colors, 250 ms per move, depth cap
eight, no book, and a 200-ply limit. Matches run sequentially without training or
other engine searches running alongside them. Unfinished games do not count as
draws. These small suites screen candidates; they do not establish an Elo rating.

The heuristic-correction candidate scored **2 wins, 6 losses**, all completed,
against the heuristic. It therefore fails the local improvement gate.

The material candidate scored **1 win, 7 losses**, all completed. Neither model is
promoted to the app. Fixing symmetry and the sampled material inconsistencies did
not make either candidate beat the heuristic in this screening suite. The data
expansion is a pilot (799 extra training labels), and the conservative correction
bound trades positional expressiveness for sanity. Future work should explore a
more expressive, color-consistent representation and broader search-derived
training coverage, while keeping local games as the promotion criterion.

Full PGNs and timing summaries are in
`ml/artifacts/search-material-v6-vs-heuristic-250ms/` and
`ml/artifacts/search-residual-v6-vs-heuristic-250ms/`.

```sh
.venv/bin/python -m ml.compare --checkpoint ml/artifacts/search-material-v6.pt \
  --openings benchmarks/openings-neural-v5.json --pairs 4 --time-ms 250 \
  --max-plies 200 --output ml/artifacts/search-material-v6-vs-heuristic-250ms

.venv/bin/python -m ml.compare --checkpoint ml/artifacts/search-residual-v6.pt \
  --openings benchmarks/openings-neural-v5.json --pairs 4 --time-ms 250 \
  --max-plies 200 --output ml/artifacts/search-residual-v6-vs-heuristic-250ms
```

Verification: all 77 unit tests pass, including legacy-checkpoint compatibility,
color-mirror training/inference parity, material direction, and split-key behavior.
An independent dataset check confirms no game IDs or canonical mirrored positions
overlap across train, validation, and test splits.
