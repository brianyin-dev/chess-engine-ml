# Exact feature sharing and capacity-aware leaf training

The opt-in inference path shares pseudo-attack maps between handcrafted mobility/center evaluation and relational neural features, canonicalizes Black-to-move feature arrays without constructing a mirrored board, and reuses NumPy hidden/output buffers. The app and default evaluator remain unchanged. Frozen inference at b962422 and the unchanged engine provide the references.

## Speed and correctness

On 4,741 saved quiet search calls, alternating-order measurements reduced median search-only evaluation time from 111.83 to 103.30 microseconds: **7.6% less evaluation time**. Across 12 representative exact-depth-three searches, summed median search time fell **12.3%**, with identical moves, scores, depths, nodes and quiescence nodes. These are local profile results, not a playing-strength claim.

The initial sampling pilot counted all evaluation calls in its quiet reservoir denominator. This was corrected before optimization or training, with the pilot preserved; subsequent comparisons use one fixed corrected quiet sample. Search sampling overhead is excluded from speed measurements. Frozen and optimized scores match on all 4,741 saved quiet leaves. The final 104-test suite passed, including new parity checks across random moves/undo, both colors, castling, en passant, underpromotion and gated/ungated correction weights.

## Capacity-aware development data

Labels come from actual quiet static search calls in the v19 games. All colors and opponent variants of an opening pair belong to one group/split: six training opening groups, two validation and two test. Canonical aliases of prior labels are excluded. Up to 80 unique roots/group were sampled initially; validation was undersized, so the predeclared next 80/group were added before fitting. This produced 1,600 roots.

Teacher continuations settle captures, promotions and check evasions at 12,000 nodes. Retained pairs have exact completed 64,000-node endpoint scores, at least a 50 cp teacher gap, and a 5 cp margin reachable under the actual bounded, rounded 5% correction, or an existing correct ranking that the correction could damage. Ordinary quiet score labels provide additional coverage. This targeted distribution is not general chess accuracy; only two opening groups per holdout limit generalization evidence.

New data contains 467 training score rows and 151 pairs (52 correctable heuristic mistakes), 157 validation rows/49 pairs (15 mistakes), and 192 test rows/68 pairs (25 mistakes). Combined training retains 13,664 score rows and 3,282 rankings. The 1,055 prior hard training rankings whose margins cannot be reached at 5% were excluded from ranking training; prior scores and correct/attainable rankings were retained. Rounded endpoint bounds are used rather than an approximate continuous correction limit.

## Training result

One unchanged v8 architecture trained at the actual 5% quiet, rounded hybrid setting. Validation selected epoch 11. The candidate did not qualify: it repaired some mistakes but damaged more previously correct rankings.

| Model | Validation correct / 49 | Validation protected regressions | Test correct / 68 | Test protected regressions |
| --- | ---: | ---: | ---: | ---: |
| Old 5% | 34 | 1 | 44 | 1 |
| Trained 5% | 34 | 2 | 41 | 6 |
| Old 25% | 33 | 6 | 42 | 9 |
| Heuristic | 34 | 0 | 43 | 0 |

The trained model improved hard-case correctness from 1 to 2 on validation and 2 to 4 on test, but this did not offset its regressions. It was rejected. The predeclared speed-only fallback uses unchanged v8 weights at 5%, with optimized inference. Its checkpoint and runtime settings were frozen before fresh games; no weight adjustment follows test results.

## Fresh game protocol

20 paired games against frozen optimized heuristic 0%, then 20 against frozen unchanged old v8 25% nonincremental inference, same fresh starts, 250 ms/depth-eight/1,000-ply budgets. All games run sequentially under the shared CPU lock. Extend the heuristic screen to 100 total games on 40 further fresh opening pairs only if the first 20 score at least 60%, the old-NN screen exceeds 50%, and all 40 games complete without errors. No app promotion or Stockfish opponent match is automatic.

Scripts and evidence are preserved under `ml/artifacts/leaf-speed-v20`; `ml.run_leaf_speed_v20` coordinates labeling/training/games, and `ml.audit_leaf_speed_v20` checks the final records and explicit PyTorch inference parity.

## Completed game results and audit

| Opponent | Wins | Draws | Losses | Score |
| --- | ---: | ---: | ---: | ---: |
| Frozen optimized heuristic 0% | 5 | 3 | 12 | 32.5% |
| Frozen unchanged old NN 25% | 10 | 2 | 8 | 55% |

All 40 games completed, with ten complete opening pairs per screen, zero errors, interruptions or unfinished games. The 100-game extension was skipped because the heuristic screen failed the predeclared 60% threshold. The app retains the heuristic. These games used unchanged v8 learned weights at 5% with the new inference implementation, not the rejected retrained checkpoint.

Mean elapsed move times were 244.6/244.1 ms against the heuristic and 243.6/243.7 ms against old NN. Maximum overruns beyond the nominal 250 ms were 43/39 ms and 65/54 ms respectively. Candidate fallback root moves numbered one in the heuristic screen and two in the old-NN screen; reference counts were zero and two. Budgets are cooperative rather than hard real-time deadlines. All PGNs replay legally to the recorded terminal results.

Both original and rejected retrained 5% models match **explicit PyTorch model inference** on all 349 held-out score positions. The audit invokes the Torch model directly: disabling the optimized search path alone still uses NumPy. The earlier v19 audit helper was corrected to make this distinction explicit and its 174-position result was reverified. The saved v18 model was also explicitly rechecked on 157 positions, with zero mismatches. New/prior canonical alias overlap, cross-split position overlap and cross-split source-opening group overlap are all zero.

The model learned 150/151 new training rankings (99.3%) versus old 5%'s 101/151. It repaired 51/52 targeted training mistakes versus 7/52, with zero training protected regressions versus five. Its worse held-out behavior therefore points to generalization rather than inability to fit these training examples. Only six opening groups supplied new training and two each supplied validation/test; increasing independent game/opening diversity is more meaningful than adding more closely related leaves from those same games.

The optimization reduces measured local evaluation and fixed-depth search time while preserving scores and trees. This round **does not establish a playing-strength gain**. The 32.5% score against the heuristic differs from v19's 52.5%, but the screens use different openings and are too small to rank inference implementations reliably. The positive old-NN score also combines the 5% weight and optimized inference changes relative to its 25% nonincremental reference. No human or Stockfish Elo is inferred.

To reproduce a saved-opening screen with a new output directory:

```bash
.venv/bin/python -m ml.compare \
  --checkpoint ml/artifacts/quiet-ranking-v8.pt --nn-weight .05 \
  --quiet-only --incremental --fast-features \
  --opponent-checkpoint ml/artifacts/quiet-ranking-v8.pt \
  --opponent-nn-weight 0 --opponent-quiet-only --opponent-incremental \
  --opponent-frozen-inference \
  --openings benchmarks/openings-leaf-speed-v20-pilot.json \
  --pairs 10 --time-ms 250 --max-plies 1000 --output /tmp/hce-v20-new-run
```

This reuses a regression set; reserve new openings for future final strength claims. Existing artifacts are preserved rather than overwritten by the coordinator.
