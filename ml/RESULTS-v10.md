# Faster neural inference and targeted mistake learning

Status: first experiment complete; shared-CPU screening only. No model promotion or verified strength gain.

## Fixed experiment

The unchanged heuristic opponent uses the search copied from commit `9e23b7c`
in `benchmarks/nn_baseline_v10/search.py`, plus the unchanged handcrafted evaluator.
Matches record both hashes. Candidate and opponent receive 250 ms per move,
depth cap 8, with no opening book after the prescribed starting moves.

Three predeclared ablations use the same correction weight (0.25), quiet-only
gate, four fresh paired starts, and 600-ply safeguard:

1. v8 checkpoint with original inference work.
2. Identical v8 checkpoint with the search-leaf fast path.
3. Targeted v10 checkpoint with that same fast path.

The remaining four starts are reserved. All match results will be reported,
including losses and unfinished games; unfinished games are not draws. These
are eight-game screening matches per ablation, not an Elo estimate.

## Inference change

Search already resolves terminal outcomes with preserved repetition history.
The NN residual baseline previously repeated mate/stalemate, automatic-draw,
and repetition checks at every evaluation. The new optional search-leaf method
uses the positional heuristic directly. Encoding, neural arithmetic, rounding,
and public evaluator behavior remain unchanged. The reference mode retains
old work for equal-clock comparison. Tests and a 1024-position parity/profile
run verify this, including matching moves, scores, and visited nodes at fixed depth.

## Targeted supervision

Review the five losses from the v8 0.25-weight match, screening possible mistakes
at 100 ms and confirming them at 400 ms with full-strength Stockfish. Follow
teacher-preferred captures, promotions, and check evasions before labeling endpoints.
Generate 1000 additional training ranking pairs and 200 per held-out split,
using extra analysis for consequential disagreement roots. Retain the prior
2000/400/400 pairs; deduplicate canonical FEN pairs within each split.

Confirmed failure games receive new IDs starting at 2000000 to avoid reusing
IDs assigned to earlier failure games. The trainer rejects whole-game and
canonical-position overlap across training, validation, and test splits.
Warm-start v8; correction bounded at 250 cp. Train with a stronger ranking
objective (weight 2, 20 cp margin, 4x weight for heuristic-wrong/tied pairs),
while retaining score regression. Training-only importance adds 4x weight for
confirmed failures costing at least 100 cp, and 2x for other consequential
pairs still ranked incorrectly by v8; validation and test remain unweighted. Select epochs by validation ranking, then MAE;
no game-result or test-set selection. The source v8 dataset has been used in
prior experiments, so it is a reused development holdout, not a new final test.

`ml/run_targeted_v10.py` runs tests, profiling, label generation, training,
and sequential games after the earlier timed Stockfish benchmark finishes.
Live phase and outcomes: `ml/artifacts/targeted-v10/status.json`.
Full log: `ml/artifacts/targeted-v10-run.log`.

## Shared-CPU limitation

The separately started 1600 benchmark in another chat overlapped these initial
NN screening games and profiling. Their results are preliminary shared-CPU
observations; they do not establish a controlled strength or latency gain.
Score parity and fixed-depth node/move equivalence remain valid. An isolated
repeat is required before promotion or a performance claim.

## Results

All **89 tests passed**. The first five alternating profiling trials observed
median search-leaf evaluation at **0.10560 ms reference vs 0.09494 ms fast**
(10.1% lower). All 1024 sampled integer scores matched. At fixed depth, moves,
scores, and visited nodes matched. The concurrent 1600 match prevents treating
this timing sample as an isolated performance claim.

Stockfish confirmed a >=100cp mistake in all five reviewed losses (814, 240,
136, 174, 107 cp); four had an incorrect raw static ranking and one recovered
with extra time. Generation requested 18.81 million Stockfish nodes in 83.36 s,
reusing labels and avoiding 3512 cached analyses. It added 1000/200/200 pairs.
After retaining old supervision and deduplication, there are 2988 training,
596 validation, and 592 test pairs; 536 training pairs receive extra importance.
Score regression uses 11230/1776/1799 rows. Training selected epoch 1 by validation.

At the deployed 0.25 quiet blend, recent confirmed settled **training** pairs
improved from **5/13 to 6/13**. Across all confirmed training pairs, 20/33 became
21/33. This is a small learnability improvement, not generalization evidence.
On the reused held-out ranking set, the same blend scored **449/592 (75.84%)
for v8 vs 448/592 (75.68%) for v10**, with the heuristic at 442/592 (74.66%).
The trained model did not improve over the prior NN at the deployed blend.
Both checkpoints matched rounded PyTorch/engine inference on 512 positions,
with zero color-symmetry error. See `learning-check.json`.

All game rows below are **against the frozen heuristic**, at 250 ms per move,
on the identical four color-paired starts. The overlapping 1600 benchmark makes
these preliminary observations. They do not establish Elo or a controlled gain.

| Candidate | Wins | Draws | Losses | Completed | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| v8 original inference | 5 | 1 | 2 | 8 | 68.75% |
| v8 fast inference | 5 | 1 | 2 | 8 | 68.75% |
| v10 targeted retraining + fast inference | 3 | 3 | 2 | 8 | 56.25% |

No errors, interrupted games, or unfinished games in these three NN matches.
The quiet 0.25 v8 setting had not been tested in the earlier v8 game series;
its different result is not evidence that v8 changed or suddenly gained Elo.
The inference fast path preserves evaluations and is retained. The v10 checkpoint
remains experimental, and the app's heuristic default remains unchanged.

A likely next issue is objective alignment: training ranks the full neural
correction, while these games use only a quarter of it and disable it on
positions with captures or check. Some incorrect rankings cannot be fixed
within this blend's maximum correction. More data alone does not resolve that.
Before any promotion, repeat the timings and game comparisons in isolation;
then align ranking supervision with the actual blend/gate and attainable margin.

Artifacts: `ml/artifacts/targeted-v10/`, `ml/artifacts/targeted-ranking-v10.json`,
`ml/data/targeted-v10-2026/`. Four additional paired starts remain reserved.
