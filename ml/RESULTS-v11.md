# Isolated NN verification and training aligned with deployed evaluation

Status: completed. The candidate failed both game comparisons and was rejected.
Stockfish matches were cancelled at the user's request. The app keeps the heuristic.

## Results at 250 ms per move

| Candidate | Opponent | Wins | Draws | Losses | Score |
|---|---|---:|---:|---:|---:|
| Old v8, 25% quiet correction | Frozen heuristic | 5 | 4 | 11 | 35% |
| Aligned v11, 25% quiet correction | Frozen heuristic | 8 | 2 | 10 | 45% |
| Aligned v11, 25% quiet correction | Old v8, 25% quiet correction | 4 | 1 | 15 | 22.5% |

All three matches completed all 20 games and 10 color pairs without errors or
unfinished games. Each match ran alone. Old-v8 verification used one set of starts;
both v11 comparisons shared a different fresh set. Comparing 35% and 45% across
different openings does not establish an improvement. The direct v11-v8 match
favored v8. The old model failed the predeclared 60% extension gate, so no 100-game
NN match was run. These small screens are not precise strength estimates.

Stockfish confirmed 18 first mistakes across 20 reviewed losses/draws, including
conversion losses in 5 of the 7 draws examined. Added supervision contains 600
training, 100 validation, and 100 test move pairs with separate game ownership.

At the exact deployed blend, rounded inference ranks 518/683 test pairs correctly
for v11 versus 512/683 for v8 and 502/683 for the heuristic. However, on validation,
v11 gets 491/694 versus v8's 495/694. It reverses 15 heuristic-correct validation
rankings, compared with 4 for v8. Better aggregate test prediction did not produce
better games. Torch and engine inference agree on 512 positions, with zero integer
score differences and zero color-symmetry error.

## Follow-up after rejecting v11

- Added optional loss weights to protect teacher rankings the heuristic already
  gets right and anchor corrections to the initial model. Reduced emphasis on
  fitting absolute Stockfish scores in the follow-up experiment.
- Fixed validation ranking to use integer centipawn rounding as search does.
- Added the starting checkpoint as epoch 0 in validation selection, allowing
  training to retain it if every trained epoch is worse.
- The protected v12 diagnostic improved test ranking to 520/683, but validation
  remained below v8 (494/694). It was rejected before spending time on games.
- Repeated training with rounded selection and the epoch-0 safeguard (v13).
  No epoch improved validation ranking; epoch 0 was selected. Tensor equality
  confirms the saved parameters are unchanged v8. This is not a new strength gain.

The retained runtime change precomputes pawn geometry as bitboard masks. Features
are exactly identical on 1,940 labeled positions. Complete hybrid evaluation on
2,048 positions including color mirrors improved from median 0.09909 ms to
0.09449 ms (4.6%) over seven alternating trials, with zero changed integer scores.
The heuristic alone takes 0.02588 ms on the same sample, so hybrid evaluation
still costs about 3.65 times as much. This is a microbenchmark, not a claimed
playing-strength gain. The earlier terminal-check optimization independently
measured about a 10% evaluation-time reduction.

Validation: 95 tests pass; randomized legal-position tests independently check
the pawn masks, and a regression test checks integer-score ranking ties.
No neural checkpoint was promoted and no Stockfish 1500 result is claimed here.

## Protocol fixed before results

- Reference NN: unchanged v8 checkpoint, 25% correction, quiet-only gate.
- Frozen handcrafted opponent: search and evaluator at `9e23b7c`, depth cap 8.
- Equal 250 ms cooperative move clocks, paired colors, no book after prescribed starts.
- Fix 50 fresh book starts. First 10 starts supply 20 games. Extend with the other
  40 starts only if all pilot games complete and score is at least 60%.
- Preserve the first attempted 20 games (6 wins, 3 draws, 11 losses) but exclude
  them from this decision: another chat's 1500 benchmark overlapped the entire
  attempt. Repeat the same fixed starts, not a result-selected replacement set.
- Wait for pre-existing benchmark reports to finish. A shared advisory CPU lock
  then serializes both benchmark CLIs and this pipeline's tests, profiling,
  Stockfish labeling, and training. Synchronous child commands share their
  parent's locked job; independent jobs wait. macOS/Linux `flock` releases on exit.

## Training alignment

Regression, ranking, validation MAE, and validation ranking now multiply the
network output by 0.25, and mask it to zero on positions with check or legal
captures, exactly as inference does. For ranking pairs this is applied to each
endpoint separately. The 250 cp raw correction bound gives only 62.5 cp per
active endpoint after mixing. Exclude training ranking margins above the
maximum attainable correction and pairs with no active NN endpoint. Validation
still includes every example, so unfixable mistakes remain visible.

Warm-start v8 rather than the regressed v10 model. Retain prior data and add
600 training/100 validation/100 test teacher-settled pairs. Failure game IDs start
at 3000000 to preserve ownership. Epoch selection uses the deployed-blend
validation ranking, with deployed-blend MAE for ties. No test or game-result
selection of epochs. The reused validation/test dataset is a development
holdout, not an untouched final research test.

## Loss and draw review

Use full-strength Stockfish, screening at 100 ms and confirming at 400 ms.
Losses require a >=100cp mistake or mate deterioration. Draws require a missed
forced mate or a best-move advantage >=150cp dropped to <=50cp with >=100cp loss.
A centipawn advantage is an estimate, not proof of a forced win. Absence of a
confirmed conversion mistake does not prove there was never a win.

Review the isolated reference games plus the prior v10 games with extra draws.
Resolve each report's actual checkpoint by SHA-256 for NN diagnostics. Teacher
labels are independent of which NN produced the game. Follow teacher-selected
captures, promotions, and check evasions before training on settled endpoints.

## Candidate and external comparisons

The aligned candidate plays 20 fresh paired games against the unchanged
heuristic and 20 against the actual old 25% quiet NN (not its default full-weight
configuration). New comparison roots exclude canonical aliases of the labeled
positions, teacher roots, and ranking endpoints. Both baselines must score
below the candidate before it is eligible for larger validation. No automatic
app promotion from small screening matches.

Planned external benchmark: Stockfish 19 `UCI_Elo=1500`, as requested. Both the heuristic
and aligned candidate would play the same 20 fresh paired games at 250 ms per move.
The user subsequently requested skipping this step if the NN failed against the
heuristic. The just-started heuristic match was stopped and excluded; the candidate
match was never started. Future runs now skip external games unless both baseline
comparisons score above 50%.
This separate app-depth-eight comparison is not identical to the other chat's
100-game depth-cap-64 heuristic benchmark, and results must not be combined.
Configured Stockfish Elo is not a measured human rating of this engine.

Live status: `ml/artifacts/aligned-v11/status.json`.
Log: `ml/artifacts/aligned-v11/run.log`.
Verification: `ml/artifacts/aligned-v11/old-v8-verified-20/`.
Full phase coordinator: `ml/run_aligned_v11.py`.
