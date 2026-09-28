# Final classical-search round: features and measured results

The changes are retained. They improve search throughput, produce a better backup
choice in the diagnosed timeout case, and scored better than the immediately
previous engine in this small paired match. They do not establish a human Elo rating.

## Features

| Feature | What it does | Why it matters |
| --- | --- | --- |
| Scored fallback | Uses a small share of the existing budget to rank legal backup moves, including a moved-piece exposure penalty and rule-based terminal results. | A slow first search no longer normally leaves only the arbitrary first legal move. |
| Direct tactical move generation | Generates legal captures and promotions directly in quiet quiescence nodes, while keeping every check evasion and checking for stalemate. | Avoids building and filtering a full quiet-move list at most leaves. |
| Positional evaluation cache | Reuses built-in evaluator results within a search, after checking terminal/history rules; avoids redundant terminal detection inside evaluation. | Repeated positions cost less to score. Custom evaluators are not assumed history-independent. |
| Killer moves and preferred-move hints | Tries quiet moves that caused earlier cutoffs sooner, and shares ordering hints across equivalent positions without sharing history-dependent search scores. | Helps alpha-beta discard unpromising branches earlier. |

Evaluation weights are unchanged. All captures/promotions are still considered by
quiescence; this round did not add speculative pruning or quiet-threat extensions.
Each auxiliary cache is capped at 20,000 entries per request; killer moves are
limited to two per ply. Timing remains cooperative, not a hard real-time guarantee.
If essentially no time is available, the fallback can still be unscored. An exposure
penalty is a heuristic, not proof that a move is safe.

## Validation

- **51 tests pass**, including equivalence with the frozen evaluator, fixed-depth
  search scores on regression positions, optimized move generation against full
  legal generation (including en passant and all promotions), fallback behavior,
  mate/stalemate handling, draw-safe caching, and custom-evaluator behavior.
- All **7 tactical regression checks** pass at main depth three.
- The immediately previous search/evaluator are frozen in `benchmarks/pre_round/`.
  Their content was checked against the previous diagnostic run's source hashes,
  allowing only the frozen evaluator import change.
- All 12 match PGNs were replayed and verified against recorded moves, final FENs,
  and rule results. Match and probe source hashes match the delivered code.

## Equal-budget match against the previous version

Six previously reserved opening lines were each played twice with colors swapped,
for 12 games. Both engines received 250 ms per move; the cap was 160 engine-played
plies after the opening. The opponent was the immediately previous engine, not
the much weaker original baseline. Both versions were fixed throughout the run.

| Outcome for new engine | Games |
| --- | ---: |
| Wins | 7 |
| Draws | 0 |
| Losses | 2 |
| Unfinished at the move limit | 3 |
| Engine errors | 0 |

The three opening pairs where **both games finished** produced 4 wins and 2 losses.
The other three pairs each contained a win and an unfinished game. Unfinished games
are not draws, and their outcomes cannot be inferred from the completed games.
This is promising small-sample evidence, not a statistically established Elo gain.
These six openings have now been used for validation and should no longer be
called unseen data in future experiments. No tuning followed this holdout result.

Mean actual move times were approximately 245 ms for each engine. Maximum observed
overruns were 16.6 ms for current and 0.7 ms for previous. The current engine had no
depth-zero returns across 552 moves; the previous engine had one across 549 moves.

## Diagnostic performance

On the six already-known mistake positions, serialized probes used identical
budgets and Stockfish reference analysis. These are diagnostic, not holdout cases.

| Thinking budget | Previous mean completed depth | New mean completed depth | Median searched-node ratio (new/previous) |
| --- | ---: | ---: | ---: |
| 250 ms | 1.50 | 1.83 | 1.75x |
| 1500 ms | 2.83 | 3.17 | 1.69x |

Node counts exclude fallback evaluations, while elapsed time includes them. More
nodes are a throughput measurement, not a direct strength score. At the same fixed
main depth, the sample middlegame regression took about 0.43 seconds with the new
engine, versus about 0.86 seconds in the earlier recorded run; those timings were
not simultaneous and should be treated as illustrative.

The strongest diagnostic improvement was the Dutch position:

- At 250 ms, previous returned unscored **Nf7** at depth zero; current returned
  scored **Bxe4** at depth zero. The reference estimated losses of 606 and 368 cp.
- At the app's normal 1.5-second budget, previous chose **Bxe4** at depth two;
  current reached depth three and chose **Nxd7**, the reference's preferred move.
- The other five diagnostic positions retained the same choices at each budget.
  Several therefore remain real weaknesses, including the French **...Qa5** and
  Scandinavian **...Nxc2** decisions. This round did not solve every tactical issue.

Reference scores are finite-search estimates from completed non-bound UCI updates.
They are not exact game-theoretic values. The gains in these selected cases are
concentrated in one position and should not be generalized to all positions.

## Using the result

Restart the backend to use the updated engine. The API now reports
`static_cache_hits` and `fallback_evaluations` alongside existing search statistics.
The original `best_move` interface and custom evaluator hook remain available.

This completes the bounded classical-search round. The engine is a better measured
baseline for an NN evaluator experiment; compare any learned evaluator under equal
thinking time and preserve the distinction between diagnostic and validation data.
No opening book or trained neural evaluator was added.

## Saved artifacts

- `holdout-match/report.json`: configuration, hashes, moves, times, and results.
- `holdout-match/games.pgn`: all 12 games, including the three unfinished games.
- `diagnostic-probes.json`: old/new decisions, depths, nodes, and reference reviews.
- `tactics-depth3.json`: tactical regression results.

Reproduction commands are in the repository README under “Final classical-search round.”
