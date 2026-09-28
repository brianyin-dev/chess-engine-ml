# Diagnostic findings: classical engine versus Stockfish

The next development target is tactical search efficiency and reliable move selection
under a deadline, followed by evaluation changes supported by individual cases.
This run does not support starting neural-network training as the next strength fix.
No heuristic weights or engine search code were changed during this diagnostic task.

## What was tested

- 16 games across eight new opening lines, with colors swapped for each pair.
- Both engines received 250 ms per move; the original validation had used 100 ms.
- Full-strength Stockfish 19, one thread and 32 MB hash, was the stronger opponent.
- Result for this engine: **0 wins, 0 draws, 16 losses**. All ended by checkmate;
  no engine errors or unfinished games. All PGNs were replayed and verified.
- Six additional legal opening lines remain reserved and unplayed in
  `benchmarks/openings-holdout.json` for future validation.
- The expanded suite passes **43 tests**.

Stockfish is deliberately far stronger. These games expose mistakes; they are not
an estimate of performance against a 1200-rated human. The opening set is small
and hand-selected, and timed results can vary with machine load.

## Search measurements

Across 438 engine moves, completed depths were:

| Depth | Moves |
| --- | ---: |
| 0 (fallback) | 1 |
| 1 | 18 |
| 2 | 208 |
| 3 | 158 |
| 4 | 44 |
| 5 | 9 |

Mean completed main-search depth was **2.58 plies**. Quiescence accounted for
**1,241,772 of 1,325,063 visited nodes (93.7%)**. That fraction is not itself a bug:
quiescence resolves exchanges. It does show where computation is concentrated.
There were 449 transposition-cache cutoffs across these searches. This is not a
cache hit rate because quiescence nodes are not cached, and preferred-move reuse
is not included in that counter.

Actual mean move times were 240 ms for this engine and 227 ms for Stockfish.
Maximum recorded overruns were about 2.8 ms and 5.5 ms respectively. The deadlines
are cooperative; Stockfish sometimes finished early, and there was no pondering.

## Move review and longer-thinking probes

All 438 engine moves were reviewed with Stockfish at 100 ms per root search.
The screen estimated at least 100 centipawns of loss on 52 moves, and reported a
new forced-mate disadvantage on 11. These are finite-search estimates, not proofs.

Six high-loss candidates from distinct games were rechecked at 500 ms per reference
search. Reference searches use the same root and mover's perspective, restrict the
played-move search to that move, and retain only completed non-bound score updates.
An earlier review incorrectly merged transient bounds; `analysis.json` is marked
superseded. **Use `analysis-exact.json`.** “Exact” here means a non-bound UCI score,
not a solved chess position.

The unchanged engine was then given 1.5 seconds at each selected position, matching
the app's normal budget. Both original and new moves were reviewed again. A loss
of 100 cp is one pawn-unit of reference evaluation, not necessarily a literal pawn.

| Game / opening / side & move | Played at 250 ms | Reference candidate | Estimated loss (cp) | Move at 1.5 s | Estimated loss (cp) |
| --- | --- | --- | ---: | --- | ---: |
| 13 / Dutch / white 11 | Nf7 (d0) | Nxd7 | 606 | Bxe4 (d2) | 368 |
| 3 / French Advance / black 9 | Qa5 (d2) | Nxb4 | 677 | Qa5 (d4) | 677 |
| 6 / Caro-Kann Classical / black 16 | Qb6 (d2) | Kf8 | 401 | Ne5 (d3) | 130 |
| 8 / Scandinavian / white 16 | Bd6 (d2) | Qe2 | 398 | Ne5 (d3) | 279 |
| 7 / Scandinavian / black 8 | Nxc2 (d2) | c6 | 405 | Nxc2 (d3) | 405 |
| 2 / Ruy Lopez / black 13 | Re8 (d1) | Bxc3 | 340 | Be6 (d2) | 466 |

Three probes reduced the estimated loss, two kept the same move/loss, and one
worsened it. None matched the reference candidate in these six cases. Giving the
engine more time helps some decisions, but is not a complete remedy. The selected
cases are biased toward mistakes and cannot estimate average benefit from more time.
Two positions crossed below -300 cp under confirmation; not all were near equality.

## Concrete weaknesses and next experiments

1. **Deadline fallback quality.** In the Dutch game, 11.Nf7 was returned at depth
   zero and met ...Rxf7. No full iteration had finished. The fallback is the first
   legal move, so it has no tactical quality guarantee. First experiment: produce
   an inexpensive scored fallback before allowing a long quiescence branch to
   consume the budget. Test tiny budgets and this real position.
2. **Quiet threats beyond the tactical horizon.** In the Scandinavian, ...Nxc2
   remained the choice at 1.5 seconds; the reference line starts Bb5+ Kd8 Ne5 and
   continues with knight forks and rook captures. In the Ruy Lopez screening case,
   Qxd4 was followed by ...c5 and ...c4. Capture-only quiescence does not extend
   every quiet threat or checking move outside check. These are useful cases for
   testing search depth and selective extensions, not a reason to add all quiet
   moves to quiescence and explode the tree further.
3. **More time is not sufficient.** ...Qa5 persisted in the French at completed
   depth four, while the reference favored ...Nxb4. The Ruy Lopez probe changed
   ...Re8 to ...Be6 and worsened the reference estimate. Inspect these variations
   before blaming a particular heuristic; horizon effects and evaluation interact.
4. **Optimize where the work is.** Profile quiescence on these actual positions,
   then test stronger move ordering and carefully validated capture pruning.
   Preserve all legal evasions while in check, promotions, and mate/draw rules.
   Compare candidates with fixed budgets, rather than declaring fewer nodes a win.
5. **Validate before tuning more.** Keep these six cases in the diagnostic set,
   compare candidate versions on diagnostic games, and finally run the untouched
   holdout openings. Evaluate win/loss results, fallback frequency, tactical error
   estimates, and completed depth. Avoid tuning solely to Stockfish's exact move.

These experiments precede NN training. A slower evaluator could worsen the current
shallow-search problem; a learned evaluator should later compete against a stable,
measured classical baseline under equal thinking time.

## Artifacts

- `report.json`: match settings, hashes, per-move diagnostics, and results.
- `games.pgn`: 16 complete games.
- `analysis-exact.json`: corrected move review, reference lines, and longer probes.
- `benchmarks/diagnostic-mistakes.json`: the six selected positions with full histories.
- `benchmarks/openings-diagnostic.json`: eight played opening lines.
- `benchmarks/openings-holdout.json`: six reserved opening lines.

Reproduction commands and methodological limits are in the repository README.
