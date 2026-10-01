# Fresh search disagreements: prediction gain, game regression

The candidate passed fresh position qualification but failed both game screens. It is not promoted; the app retains the heuristic and v8 remains the unchanged experimental NN reference. No 100-game extension or Stockfish opponent match was run.

## Data and training

Collected quiet leaves from actual search after common book starts: 480 teacher self-play starts and 120 unchanged-student self-play starts. Compared heuristic and v8 at 25% correction using depth-three caps and equal node budgets (600 initially, 1,500 subsequently). These are bounded probes: completed depths can differ. Stockfish screened candidate moves at 100 ms, confirmed consequential disagreements at 400 ms, and settled tactical continuations before producing endpoint rankings. Existing fixed endpoints were refined at 64,000 nodes, retaining the last completed exact score.

The initial 120-start collection was too small. Fixed supplemental batches, a student source-color split correction, and endpoint refinement were all specified and completed before candidate fitting or validation. The coordinator was paused during supplementation, then resumed; saved protocol amendments and raw snapshots preserve this history. Student whole-game validation/test ownership was balanced across source colors without changing training membership. No candidate result selected examples or splits.

After refinement, new data contains 145 training pairs, 41 validation pairs and 46 test pairs, from 90/24/27 source games respectively. Canonical position aliases, including color mirrors, have zero overlap with prior data or other splits; source games do not cross splits. Prior training data was retained; prior validation/test labels were excluded from training.

One candidate retained the v8 architecture, bounded residual, quiet gate and 25% correction. Training uses rounded full hybrid scores, ranking weight 4, 5 cp margin, protected-example weight 4, hard-example weight 2, score weight 0.02, anchor weight 2 and learning rate 0.0001. Validation selected epoch 5; the untouched new test qualified it once. No tuning followed the test or game results.

## Position checks

| Evaluator | Validation correct / 41 | Test correct / 46 | Test regressions on heuristic-correct pairs |
| --- | ---: | ---: | ---: |
| Heuristic | 14 | 16 | 0 |
| Old v8 at 25% | 13 | 19 | 2 |
| Candidate at 25% | 16 | 23 | 1 |

The selected failure set is small and deliberately difficult, and examples within a source game are correlated. These numbers measure ranking on that set, not general chess strength. Candidate incremental, NumPy and PyTorch inference agree exactly on all 157 fresh validation/test score positions. All 102 unit tests passed before fitting.

## Equal-time games

Both screens used the same ten fresh book starts with colors reversed, 250 ms per move, depth-eight cap and 1,000-ply limit. The candidate uses incremental inference; the heuristic uses optimized zero correction, and the old NN retains its unchanged nonincremental inference. Matches ran sequentially under the shared CPU lock.

| Opponent | Wins | Draws | Losses | Score |
| --- | ---: | ---: | ---: | ---: |
| Optimized heuristic, 0% NN | 6 | 2 | 12 | 35% |
| Unchanged old v8, 25% NN | 7 | 1 | 12 | 37.5% |

All 40 games completed with zero errors, interrupted games or fallback moves. Mean elapsed move times were 244.4/242.1 ms against the heuristic and 244.7/244.6 ms against old v8. Budgets are nominal, not hard real-time limits: maximum overrun was 222 ms for one heuristic move, versus 23 ms for the candidate in that match; the old-NN match maxima were 42/26 ms. These small screens do not establish precise Elo, but neither supports promotion. The predeclared extension required at least 60% against the heuristic and more than 50% against old v8, so it was skipped.

The practical finding is that learning more selected rankings did not improve search decisions in games. Further work should diagnose lost-game decisions and use a broader retention/qualification suite before another candidate; adding more of the same targeted labels alone is not supported by this result.

## Reproduction and evidence

`ml.run_disagreements_v18` coordinates collection, refinement, training, qualification and conditional games. `ml/artifacts/disagreements-v18` preserves protocols, audits, checkpoint, metrics, PGNs and final status. `ml/data/disagreements-v18-2026` holds the new examples; `ml/data/aligned-v18-2026` holds training inputs. Frozen opening manifests record seeds and exclusions.

Candidate SHA256: `6ed626ac741fd38d435e0036eb9c0015a6fbadc0be9b0a20f9758e40f1d15355`.
Old v8 SHA256: `32beb3b6d5231fe115b2f07ccbc6298c9d9f76a5d58fce27801f6e1d443237b0`.
