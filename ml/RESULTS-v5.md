# Loss analysis, material features, and candidate ranking

This experiment implements three changes: diagnose the previous NN's losses,
train on better/worse candidate comparisons, and supply material and game phase
explicitly. The app keeps its classical evaluator because the new candidate did
not improve local playing strength.

## Diagnosis

`artifacts/loss-analysis-v5.json` reviews 16 losses by the v4 residual evaluator.
The first move losing at least 100 centipawns (or deteriorating a mate result)
was screened with full-strength Stockfish for 100 ms and confirmed with separate
400 ms root searches for the best and played move. These are estimates, not
proofs. All 16 games had a confirmed early costly move.

Seven one-second NN probes changed the move and brought its estimated loss below
the threshold. Nine original positions had the wrong ordering under the NN's
static scores for the played and preferred successor positions. These categories
overlap. A static score cannot capture every tactical continuation; neither test
fully separates search errors from evaluation errors.

For example, in the Ruy Lopez as White, Be3 lost an estimated 208 cp relative to
Bxc6. A longer search completed depth four rather than three and avoided that
mistake. In the French Tarrasch as Black, dxe4 lost an estimated 476 cp relative
to Bb6; the NN preferred the worse successor statically, and a longer probe did
not recover. The JSON records every first mistake and both principal variations.

## Training changes

Model version three adds ten piece-count features, material balance, and material
phase to the original 782 inputs, for 794 total. Older checkpoints remain loadable.
The new material mode uses fixed centipawn piece values plus a learned positional
adjustment bounded to ±250 cp with tanh. This limits the network's ability to erase
a large material difference and avoids computing the full classical heuristic.

The score-only control and ranked candidate use the same 32,918-position v4 data,
architecture, material baseline, bound, optimizer, and seed. The ranked candidate
also uses a mover-relative margin loss on 663 candidate pairs: 452 training,
108 validation, 103 test. The confirmed mistakes are training examples only.
Other pairs come from sampled games in the corresponding base-data split and
Stockfish comparisons at 80 ms per root search. Identical successor FENs do not
cross base-data or pair splits; an audit ignoring the FEN fullmove number also
found no overlap.

Training minimizes Huber score error plus 0.2 times a ranking hinge loss with a
100 cp margin. Checkpoint selection uses validation MAE plus 100 cp times the
validation ranking error rate. Test labels and game results are not used for
checkpoint selection. The control is selected by validation score error alone.

| Evaluator | Test score MAE (cp) | Correct ordering on 103 test pairs |
| --- | ---: | ---: |
| Classical heuristic | 179.9 | 69.9% |
| Material baseline alone | — | 33.0% |
| Material NN, score-only control | 143.4 | 58.3% |
| Material NN, score and ranking loss | 146.2 | 71.8% |

Ranking accuracy improved relative to the control while score error worsened
slightly. The two-point difference from the heuristic on this small pair sample
does not establish superiority. See `artifacts/ranking-evaluation-v5.json` and
the two training-metrics JSON files.

## Local game validation

Six legal opening lines in `benchmarks/openings-neural-v5.json` were fixed before
game validation. Each opening is played with both colors, no book moves during
the match, a 250 ms per-move budget, depth cap eight, sequential execution, and
a 160-ply cap after the prescribed opening. Timeouts are cooperative; reports
include actual elapsed time and overruns. Unfinished games are not draws.

The ranked model scored **1 win and 11 losses** against the heuristic over all
six opening pairs. Against the previous v4 residual NN on the first four pairs,
it scored **1 win, 4 losses, and 3 unfinished**. These small local samples do not
estimate human Elo. They show that this recipe has not improved playing strength,
even though it improves the held-out candidate-ranking metric.

On the first four opening pairs, the score-only control scored 0 wins, 6 losses,
and 2 unfinished against the heuristic; the ranked candidate scored 1 win and
7 losses on that same subset. The incomplete games and small samples prevent
a firm claim that ranking training improved game strength relative to the control.

A standalone local profile over 250 held-out positions and 20 rounds measured
0.02393 ms per ranked-model evaluation versus 0.03697 ms for the heuristic.
This is about 35% less evaluator time; it excludes the rest of search. The
new candidate's weakness cannot be explained solely by slow evaluator calls.
Individual rounds are saved in `artifacts/inference-profile-v5.json`.

Complete PGNs, per-move search diagnostics, checkpoints, and source hashes are
under `ml/artifacts/material-*-v5*`. The previous experiments remain documented
in `ml/RESULTS.md`.

## Reproduction

Install `requirements.txt` and supply the local Stockfish binary. Every output
must be new to preserve existing experiments.

```bash
python -m ml.analyze_losses --report ml/artifacts/book-deviations-v4-residual-vs-heuristic-250ms/report.json --report ml/artifacts/book-deviations-v4-residual-vs-heuristic-live-250ms/report.json --checkpoint ml/artifacts/book-deviations-v4-residual.pt --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal --output NEW-ANALYSIS.json
python -m ml.generate_pairs --data ml/data/stockfish-book-deviations-v4-2026 --analysis NEW-ANALYSIS.json --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal --output NEW-PAIRS
python -m ml.train --data ml/data/stockfish-book-deviations-v4-2026 --target material --pairs NEW-PAIRS --seed 337 --checkpoint NEW-MODEL.pt --metrics NEW-METRICS.json
python -m ml.compare --checkpoint NEW-MODEL.pt --openings benchmarks/openings-neural-v5.json --pairs 6 --time-ms 250 --max-plies 160 --output NEW-MATCH
```

The exact checked-in labels and pairs reproduce the training experiment more
closely than regenerated time-limited Stockfish analysis, which depends on hardware.
