# Evaluator consistency and search-position audit

The current trained NN is not ready for a representation redesign based on
clean fundamentals: the audit found color asymmetry, material inconsistencies,
and a mismatch between training positions and positions evaluated during search.
It found no training-to-inference conversion error. The app continues to use the
classical evaluator.

The reproducible audit is `ml/audit.py`. Exact results, example FENs, source
checkpoint hashes, per-position Stockfish labels, and search roots are saved in
`ml/artifacts/consistency-leaf-audit-v5-final/`.

## Basic consistency

We sampled 512 held-out positions and evaluated the latest ranked material NN,
its score-only control, both v4 NNs, and the heuristic. Color mirroring swaps
White/Black pieces, turn, and castling rights while flipping ranks. A consistent
White-perspective evaluator should satisfy `eval(board) = -eval(board.mirror())`.

| Evaluator | Mean color-symmetry error | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Heuristic | 0 cp | 0 cp | 0 cp |
| Latest ranked material NN | 148 cp | 356 cp | 497 cp |
| Material score-only control | 148 cp | 350 cp | 500 cp |
| Previous residual NN | 214 cp | 511 cp | 737 cp |
| Previous full-score NN | 294 cp | 523 cp | 835 cp |

For the latest model, 81.8% of sampled positions had more than 25 cp of mirror
inconsistency. In one case the original score was −228 cp and the mirrored score
was −269 cp, though it should have changed sign. This is a learned inconsistency;
the current network does not enforce color symmetry.

The PyTorch training model and NumPy engine adapter returned identical rounded
centipawn scores on all 512 positions for each of the four NNs: 2,048 checks,
zero differences. The encoder, score scaling, baseline addition, and inference
path agreed on this sample.

## Material and hanging pieces

Each evaluator received the same controlled legal piece-removal perturbations.
Deleting a Black piece should normally improve the White-perspective evaluation,
and conversely for deleting White material. This is a sanity probe, not a theorem:
removing a piece can also change pins, lines, and positional factors.

The latest ranked NN valued removing an enemy queen positively in all 116 probes,
with a mean increase of 735 cp. It valued bishops and rooks positively in every
probe. However, 15 of 195 pawn probes decreased the expected advantage, and one
of 130 knight probes did too. The heuristic gave the expected positive direction
in every identical probe. The previous full-score NN was particularly deficient:
removing an enemy queen changed its score by only about 9 cp on average, and it
gave the expected direction in only 64.7% of those probes.

At fixed main-search depth three, the latest NN passed all seven tactical
regression checks, including capturing a hanging queen, avoiding a poisoned pawn,
check evasion, and promotion. The previous full-score NN passed six of seven:
it promoted to a rook instead of the expected queen in the capture-promotion
fixture. That fixture failure does not prove the resulting rook endgame loses.

Static evaluation alone does not fully recognize every hanging-piece threat.
In a controlled equal-material example, Stockfish scored the hanging White queen
position at −489 cp and its safe counterpart at +220 cp. The latest NN scored
them +266 and +322 cp: it ordered them correctly but represented only a small
part of the tactical difference. The heuristic ordered them incorrectly in
static evaluation (+411 vs +393 cp), yet passed the search-level tactical tests.
Quiescence searches captures and evasions; a static-score limitation is therefore
not by itself a search bug. The NN's mirrored example was incorrectly ordered,
reinforcing the symmetry finding.

## Actual positions evaluated by search

We reconstructed history at three stages of eight previously lost games and ran
the latest NN's search to depth three at 24 roots. Each search reservoir-sampled
20 actual static evaluator calls, yielding 480 unique sampled positions. The
wrapper preserved the evaluator's caching declaration. There was no clock limit,
so instrumentation overhead did not alter a timed strength comparison.

These calls occur at quiescence stand-pat nodes and are not all final quiet leaves.
The search does not stand pat in check. We report the full sample and separately
the 105 positions with no legal capture and an unsaturated Stockfish score.
The training-distribution comparison uses all 26,254 training records; the
prediction-error comparison uses the same 512 held-out positions from above.

| Characteristic | Training records | Sampled search calls |
| --- | ---: | ---: |
| Median absolute material imbalance | 50 cp | 420 cp |
| Mean absolute material imbalance | 92 cp | 524 cp |
| 95th-percentile material imbalance | 330 cp | 1,120 cp |
| Mean number of pieces | 19.2 | 20.4 |
| At most twelve pieces | 21.5% | 13.8% |
| A legal capture exists | 80.8% | 77.5% |

The main observed shift is material imbalance, rather than simply the presence
of captures. Search deliberately explores bad and unbalanced continuations that
strong self-play rarely reaches, even after adding single-move deviations.

Stockfish independently labeled the sampled calls at depth ten. Labels used the
same ±1,500 cp clipping as training; only 0.6% of sampled labels saturated. FEN
labels cannot reconstruct repetition history, and a searched Stockfish score is
an estimate rather than a static-evaluation ground truth.

| Evaluator | MAE on held-out training-like positions | MAE on sampled search calls |
| --- | ---: | ---: |
| Heuristic | 183 cp | 316 cp |
| Latest ranked material NN | 149 cp | 313 cp |
| Material score-only control | 145 cp | 313 cp |
| Previous residual NN | 157 cp | 275 cp |
| Previous full-score NN | 151 cp | 666 cp |

On the 105 quiet, unsaturated search calls, the latest NN's MAE was 244 cp,
versus 243 cp for the heuristic. Its score-prediction advantage on the original
held-out distribution largely disappears in this sample. These errors do not
alone prove the cause of the game losses; move ordering and rare critical errors
matter beyond averages, and the game matches remain the strength gate.

## Next priorities

Enforce color-consistent predictions and validate them on mirrored positions.
Expand training with independently labeled positions sampled from the engine's
actual search, particularly material imbalances. Recheck material sanity and
candidate rankings on held-out games before another architecture experiment.
Require improvement in local paired games before changing the app evaluator.

The checks were not clean, so this audit does not trigger automatic architecture
replacement or another training run. The existing local results still reject
promotion of the latest NN: 1–11 against the heuristic and 1 win, 4 losses,
3 unfinished against the previous NN.
