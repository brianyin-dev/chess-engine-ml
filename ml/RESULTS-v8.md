# Quiet-position ranking and smaller neural corrections

This experiment tests whether the current architecture can learn confirmed
mistakes, then changes supervision rather than growing the network. The app stays
classical unless local game evidence and consistency checks support a promotion.

## 1. Learnability diagnostic

The v7 network initially preferred the teacher's better immediate successor in
only 2 of 8 confirmed losses. Twelve Adam optimization steps on those same eight
pairs produced 8/8 preferences with at least a 20 cp margin. The ±250 cp correction
bound was retained. This is deliberately training-set memorization, not a held-out
result or playing-strength claim. It establishes that these distinctions are
representable with the existing inputs, network, and inference conversion.

The saved checkpoint is marked `diagnostic_only`; training initialization and match
commands reject it. It is not used to initialize the general candidate. Exact
before/after margins are in `ml/artifacts/memorization-diagnostic-v8/report.json`.

## 2–4. More rankings, settled tactical continuations, focused labeling

The new split sizes are **2,000 training / 400 validation / 400 test move pairs**,
up from 205/19/22 in v7. There are 494 training, 106 validation, and 87 test pairs
where the heuristic ranks the teacher's better settled endpoint incorrectly or
ties it. Twenty training comparisons come from confirmed v7 failure roots.

For each root, shallow 250-node heuristic and NN searches supply candidates,
alongside the teacher's alternative and a random legal anchor. Extra teacher work
is reserved for disagreement roots and confirmed failures, up to 500/100/100
refined roots. Unrefined root analysis uses 2,000 Stockfish nodes; refined analysis
uses 6,000. Candidate successors follow the teacher's preferred captures,
promotions, and check evasions until its chosen continuation is quiet. Each settling
step receives 1,200 or 4,800 nodes. Terminal endpoints, saturated scores, alias
pairs, and gaps under 50 cp are rejected. This is approximate teacher-guided
settling, not an exact independent quiescence oracle. Other legal captures can
remain when the teacher prefers a quiet move. Safety stops reject lines beyond
sixteen tactical plies.

Of the training pairs, 1,262 required at least one tactical settling move. Across
training endpoints, 2,884 tactical plies were followed. The regression dataset
reuses **6,614 / 874 / 880** existing capture-free score labels and adds settled
endpoint labels, for **9,711 / 1,481 / 1,503** score rows. No new general self-play
batch was generated. Teacher analysis caching avoided 5,352 repeated calls;
generation took **157 seconds locally**, with **38.0 million requested nodes**.
Node limits are cooperative, and timings depend on hardware. No GPU or cloud
compute was used.

Whole-game source splits and canonical FEN/color-mirror ownership prevent positions
and ranking endpoints crossing splits. Confirmed benchmark failures are training
only with separate game IDs. FEN labeling does not preserve repetition history.

## General candidate and validation policy

The candidate starts from v7's general checkpoint, keeps the 892-input 64→32
architecture, shared color-canonical representation, and ±250 cp residual bound.
Training uses AdamW at 0.0005, score Huber loss plus ranking loss with weight one,
a **10 cp ranking margin**, and **4× weight on heuristic-wrong pairs**. Loss weights
are normalized within each pair batch. Checkpoints maximize validation ranking
accuracy first; validation MAE breaks ties. With at most forty epochs and eight
stale epochs, epoch two was selected. Later training reduced training error but
worsened validation ranking, so it was not selected.

On the new quiet regression test rows, unrounded score MAE is **149.3 cp**, versus
**152.2 cp** for the heuristic. These are different positions from v7, so comparing
149.3 directly with v7's 197.6 would be misleading. Engine-rounded ranking scores
are used for the comparisons below; unrounded PyTorch ranking can differ by a
single tied pair from integer engine ranking.

## 5. Held-out correction-weight comparisons

The formula is `heuristic + weight × learned correction`. Zero skips neural
encoding entirely; match selection uses the pure heuristic path at zero weight.
The optional strict quiet gate skips NN work in check or whenever a legal capture
exists. This gate is stricter than teacher-guided settling: a settled training
endpoint can still contain an unprofitable legal capture.

Weights 0%, 10%, 25%, 50%, and 100%, with and without the gate, were predeclared.
For each small weight, the gate used in games was selected only by validation
ranking accuracy, preferring the gate for ties. Test results did not select
configurations. All comparisons are saved in `ml/artifacts/blend-validation-v8.json`.

| Evaluator / configuration | Correct on 400 test pairs | Corrected heuristic mistakes | Regressed previously correct pairs |
| --- | ---: | ---: | ---: |
| Heuristic / 0% NN | 313 (78.25%) | 0 | 0 |
| Previous v7 full NN correction | 289 (72.25%) | 40 | 64 |
| Retrained full correction | 320 (80.00%) | 45 | 38 |
| 10%, ungated | 315 (78.75%) | 10 | 8 |
| 25%, ungated | 323 (80.75%) | 22 | 12 |
| 50%, strict quiet gate | 321 (80.25%) | 14 | 6 |

The 25% comparison is a modest 2.5-percentage-point gain over the heuristic.
Pairs share roots/games and are correlated; these counts do not establish statistical
confidence or chess strength. Small corrections reduce regressions, while also
repairing fewer mistakes. The larger validation set is more useful than v7's
nineteen pairs, but validation gains remain small.

## Equal-time game policy

Each selected configuration plays the same four fresh opening families, both
colors, at 250 ms per move, depth cap eight, no opening book, and 200 played plies.
The heuristic opponent is the 0% baseline. Labeling, training, and profiling do
not run alongside timed games. Unfinished games do not count as draws. Configurations
are screening candidates; multiple comparisons and eight games per configuration
are insufficient for Elo claims or automatic app promotion.

The **10% ungated** configuration scored **1 win, 1 draw, 6 losses**, all completed.

The **25% ungated** configuration scored **1 win, 5 losses, 2 unfinished**. Its
better held-out ranking score did not translate into a win over the heuristic.

The 50% quiet-gated audit independently checked 512 held-out positions: zero mirror
error, identical rounded PyTorch/NumPy scores throughout, 7/7 tactical checks, and
positive material-removal gains in every sampled pawn/knight/bishop/rook/queen probe.
This is a sample-based sanity result, not a universal monotonicity guarantee. On
240 diagnostic search calls, score error was 286.3 cp versus the heuristic's 288.2 cp,
a very small difference. Those roots are from training-failure games and are
not a held-out strength result. Standalone inference on the quiet-heavy test rows
was 0.0887 ms versus the heuristic's 0.0327 ms; actual search contains many more
capture positions that skip NN work, so this ratio is not engine throughput.

## Reproduction

```sh
.venv/bin/python -m ml.diagnose_learning \
  --analysis ml/artifacts/relational-v7-loss-analysis.json \
  --checkpoint ml/artifacts/relational-v7.pt \
  --output ml/artifacts/memorization-diagnostic-v8

.venv/bin/python -m ml.generate_quiet_rankings \
  --data ml/data/targeted-v7-2026 --checkpoint ml/artifacts/relational-v7.pt \
  --failures ml/artifacts/relational-v7-loss-analysis.json \
  --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal \
  --output ml/data/quiet-rankings-v8-2026

.venv/bin/python -m ml.train --data ml/data/quiet-rankings-v8-2026 \
  --pairs ml/data/quiet-rankings-v8-2026/pairs \
  --checkpoint ml/artifacts/quiet-ranking-v8.pt --metrics ml/artifacts/quiet-ranking-v8.json \
  --target residual --features relationships --color-consistent \
  --correction-limit-cp 250 --initial-checkpoint ml/artifacts/relational-v7.pt \
  --rank-weight 1 --hard-pair-weight 4 --rank-margin-cp 10 --selection ranking \
  --learning-rate .0005 --epochs 40 --patience 8

.venv/bin/python -m ml.evaluate_blends --data ml/data/quiet-rankings-v8-2026 \
  --checkpoint ml/artifacts/quiet-ranking-v8.pt --previous ml/artifacts/relational-v7.pt \
  --output ml/artifacts/blend-validation-v8.json

.venv/bin/python -m ml.compare --checkpoint ml/artifacts/quiet-ranking-v8.pt \
  --openings benchmarks/openings-neural-v8.json --pairs 4 --nn-weight .1 \
  --time-ms 250 --max-plies 200 --output ml/artifacts/quiet-ranking-v8-w10-vs-heuristic-250ms

.venv/bin/python -m ml.compare --checkpoint ml/artifacts/quiet-ranking-v8.pt \
  --openings benchmarks/openings-neural-v8.json --pairs 4 --nn-weight .25 \
  --time-ms 250 --max-plies 200 --output ml/artifacts/quiet-ranking-v8-w25-vs-heuristic-250ms

.venv/bin/python -m ml.compare --checkpoint ml/artifacts/quiet-ranking-v8.pt \
  --openings benchmarks/openings-neural-v8.json --pairs 4 --nn-weight .5 --quiet-only \
  --time-ms 250 --max-plies 200 --output ml/artifacts/quiet-ranking-v8-w50-quiet-vs-heuristic-250ms
```

Use fresh output paths for reruns. Exact labels, settlement lines, source hashes,
checkpoints, game PGNs, and runtime configuration are preserved in the artifacts.

## Final game results and app decision

| Configuration | Wins | Draws | Losses | Unfinished |
| --- | ---: | ---: | ---: | ---: |
| 10% correction, ungated | 1 | 1 | 6 | 0 |
| 25% correction, ungated | 1 | 0 | 5 | 2 |
| 50% correction, strict quiet gate | 0 | 2 | 4 | 2 |

All unfinished games hit the 200-ply cap. They are not treated as draws. None of
these candidates wins its completed paired-opening comparison, so none qualifies
to replace the heuristic in the app. The current architecture can learn the
specific failures and now modestly exceeds the heuristic's held-out ranking count,
but that does not establish stronger search decisions or practical play. Small
weights and a quiet gate have not solved the playing-strength problem in this suite.

Verification: **86 unit tests pass**, including weighted corrections, zero-weight
inference bypass, quiet/check gating, color symmetry, invalid-weight rejection,
tactical-settling classification, legacy checkpoints, and search-budget behavior.
An independent validation checks game IDs and canonical mirrored positions across
all score and pair splits. The app evaluator remains unchanged.
