# Controlled NN correction weights on common openings

Status: complete. All 100 screening games finished, with no errors, unfinished games, or fallback moves. No positive weight reached the predeclared 60% confirmation gate; no confirmation or Stockfish matches were run.

| Candidate | Wins | Draws | Losses | Score vs optimized 0% | Opening-pair bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original heuristic control | 9 | 3 | 8 | 52.5% | 45–60% |
| Old nonincremental NN, 25% | 9 | 2 | 9 | 50% | 32.5–70% |
| Incremental NN, 5% | 7 | 3 | 10 | 42.5% | 20–65% |
| Incremental NN, 10% | 6 | 4 | 10 | 40% | 22.5–60% |
| Incremental NN, 25% | 10 | 2 | 8 | 55% | 45–67.5% |

The 0% configuration is the shared opponent, not a separately measured self-play score. Scores count a draw as half a point. Every row contains 20 games on the same 10 starts; this is 100 screening games total, not 100 games per weight.

25% has the highest observed score among the tested weights, but its 55% does not establish an improvement. Against the common opponent, its paired score difference from the original heuristic is +2.5 percentage points (descriptive 95% interval −15 to +17.5), and from the old NN is +5 points (−12.5 to +22.5). These are common-opponent score comparisons, not direct head-to-head matches. Intervals resample whole opening pairs, retaining correlation between colors, and do not correct for choosing the best weight. All include no improvement. Smaller weights did not help in this screen. Keep the heuristic app default and 25% as the experimental setting; improving confirmed evaluation mistakes is more justified than shrinking the multiplier further.

The audit verified identical checkpoint/source hashes, opening positions and color order, opponent configuration, time budgets, quiet gate, and depth cap across all five screens. Mean search times were 242–246 ms. Clocks are cooperative, not hard deadlines: the largest overrun was 201 ms on one 0% opponent move in the 5% screen (opening pair 9, candidate White, ultimately a candidate win); other screen maxima were 24–52 ms. No games were removed or rerun based on outcomes. Sequential screen order and normal clock jitter remain limitations.

Detailed audits and bootstrap estimates: `ml/artifacts/weight-control-v16/analysis.json`. Recompute with `.venv/bin/python -m ml.summarize_weight_control_v16`.

The fixed v8 checkpoint is tested with 5%, 10%, and 25% correction against the same optimized handcrafted evaluator at 0%. All use the same incremental feature implementation and quiet-position gate. The 0% reference skips neural feature encoding and inference; this is the practical cost of adding the NN. The original heuristic and old nonincremental 25% NN are additional controls against that same reference, not direct matches against each other.

Each screen plays 20 games (10 identical opening starts with colors reversed), at 250 ms per move and depth eight. Starts are sampled from the existing common-opening book with seed 162500, excluding previous benchmark starts and canonical aliases of the existing training/validation/test positions. No book is used after the prefix. Games capped at 1,000 plies remain unfinished rather than being counted as draws. CPU-heavy jobs are serialized with the shared checkout lock.

The positive weight with the highest score, breaking ties toward the smaller weight, qualifies for confirmation only with at least 60% over 20 complete error-free games. Confirmation uses 20 fresh games, expanding to 100 separate confirmation games only if its first 20 also score at least 60%. Selection games are excluded from confirmation. No automatic app promotion or Stockfish matches.

The benchmark now preserves the incremental evaluator at 0% correction instead of silently falling back to the original heuristic, and records evaluation-source hashes. All 100 tests passed, including a real-search regression check for both players' 0% routing.

Protocol and results: `ml/artifacts/weight-control-v16/`. Reproduction: `.venv/bin/python -m ml.run_weight_control_v16` (refuses to overwrite existing status/results).
