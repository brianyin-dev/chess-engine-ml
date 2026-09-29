# Chess Engine ML

[![CI](https://github.com/brianyin-dev/chess-engine-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/brianyin-dev/chess-engine-ml/actions/workflows/ci.yml)

A browser chess app with a Python classical engine, a Polyglot opening book, and
an experimental trained neural evaluator. The app still uses the classical evaluator:
the neural model improved held-out score prediction but has not improved playing
strength in equal-time games. Human playing strength has not been rated.

The board supports either color, three difficulty settings, a move list, thinking
feedback, and game-end messages. Select Black to have the engine make the first move.
The move list uses compact piece/destination notation rather than full SAN.

## Run locally

Python 3.10+ is required. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-engine.txt
.venv/bin/python books/install_gm2001.py
.venv/bin/python -m backend.app
```

Open <http://127.0.0.1:5000>. Flask now serves both the board and API from one
origin, matching production deployment. The separate port-8000 frontend workflow
still works for development. The browser uses 200 ms on Easy, 750 ms on Medium,
and 3 seconds on Hard, with a maximum main-search depth of eight plies. Restart
the backend and refresh the page after
updating the code. These are local development servers, not a production deployment.

## Optional deployment

The previous Render demo is suspended. The deployment configuration remains in
`render.yaml` in case you want to host the app again. Automatic deploys are off.

`render.yaml` describes a single Render web service. Its build installs the small
engine requirements and downloads the externally sourced opening book with pinned
archive and book checksums. Gunicorn serves the Flask API and frontend together,
so a deployed browser never tries to call its own localhost.
The service uses one worker and two threads, with one search
slot per process. Excess concurrent search requests receive HTTP 503. A 16 KB body
limit, 300-move history limit, and 120 requests per minute per observed client IP
bound public requests; HTTP 429 includes `Retry-After`. The limit is in memory and
resets on restart. Render's proxy may group clients under one observed address.

If you resume hosting later, the service filesystem only needs the downloaded book at runtime; the
third-party `.bin` remains excluded from Git. Free hosting, cold-start behavior,
and plan availability depend on Render's current terms.

The engine-only requirements omit PyTorch and NumPy. Install `requirements.txt`
when working on the ML modules.

## Classical engine

- **Evaluation:** material, correctly oriented piece-square tables, development,
  center control, rook files, pseudo-mobility, and passed pawns. King placement
  blends from shelter in the middlegame to activity in the endgame. Development
  incentives fade as material disappears. Evaluation does not mutate the board.
- **Search:** iterative-deepening negamax with alpha-beta pruning. Captures and
  promotions are ordered ahead of quiet moves; quiet cutoff moves accumulate a
  history bonus. A per-search transposition table reuses results and preferred moves.
- **Quiescence:** at the nominal depth limit, continue captures and promotions.
  When in check, search all legal evasions; never use a stand-pat score in check.
- **Mate and draws:** terminal results are determined by rules, independently of
  the evaluator. Mate scores include distance, favoring shorter discovered mates.
  Automatic fivefold repetition and the 75-move rule are supported, with mate
  taking precedence. Threefold/50-move claims are not automatic: there is no claim
  action in this app yet, matching `python-chess`'s default game-over policy.
- **Limits:** return the last completed iteration on timeout, or a legal fallback
  if no iteration finishes. Board state and history are restored on interruption
  and evaluator exceptions. A 96-ply safety limit aborts excessively long branches.

The cache is capped at 20,000 entries and belongs to one search request. Its key
includes the halfmove clock and reversible-history position counts, avoiding
incorrect reuse across repetition contexts. This conservative choice reduces cache
hits and uses more memory than a position-only key. It is a correctness baseline,
not a claim of optimal cache design.

## Python interface

```python
import chess
from engine.search import search, best_move

board = chess.Board()
result = search(board, depth=8, time_limit=1.5)
print(result.move, result.depth, result.nodes, result.elapsed)

# Existing callers can continue to request just a move.
move = best_move(board, depth=3)
```

`eval_fn(board)` must return centipawns from White's perspective. The returned
`SearchResult.score` is instead from the root side-to-move's perspective. It is
`None` when only a fallback move was available. `nodes` includes `qnodes`.
`depth` counts completed main-search plies; quiescence can examine further plies.
The deadline is cooperative, checked between nodes, not a hard process timeout.
A supplied slow evaluator can overrun it. `stop_reason` distinguishes `time_limit`
from the safety `ply_limit`; `timed_out` only means the former.

## API

`POST /move` accepts either a move history or a FEN:

```json
{"moves": ["e2e4", "e7e5"], "depth": 8, "time_ms": 1500, "use_book": true, "max_book_ply": 20}
```

`depth` is an integer from 1 to 64 (default 8), and `time_ms` is a number from 1 to
10,000 (default 1,500). `use_book` defaults to true and `max_book_ply` defaults to
20, or ten full moves. Prefer `moves` because FEN cannot encode repetition history.
Responses retain `move` (UCI) and `eval` (White's static score after the move), and
identify `source` as `book` or `search`. Book responses include the filename,
selected weight, number of alternatives, and no search diagnostics. Search responses
include completed depth, nodes, cache hits, elapsed time, score, and timeout details.
The `game` field reports automatic termination after the engine move. Invalid
payloads/positions and finished games return HTTP 400; finished games include the
result and reason. Oversized, rate-limited, and busy requests return 413, 429, and
503 respectively.

## Opening book

The app now uses the local `books/gm2001.bin` Polyglot book before search. It makes
weighted random choices, so common grandmaster moves appear more often while games
still vary. If the current position has no legal stored move, or the 20-ply cap is
reached, normal timed search starts immediately. A missing default book also falls
back to search, allowing source checkouts to run without the ignored binary.

The selected book is 486,656 bytes and contains 30,416 position–move entries across
23,813 distinct stored keys. It contains reachable continuations through 15 full
moves, though the app uses at most 10. In a seeded 10,000-line simulation of the
book's weighted choices, 93.9% reached five full moves and 49.6% reached the
configured ten; average coverage was 17.1 plies. Opponent deviations can leave it
earlier. The starting position offers eight recorded moves, heavily favoring e4,
d4, Nf3, and c4.

The binary and its provenance are documented in `books/README.md`. It came from
the [polyglot-books v1.0 release](https://github.com/ChrisWhittington/polyglot-books/releases/tag/v1.0),
which credits Oliver Deville and describes GM games from 2001–2013 with a 2530+
threshold. The distributor says the books remain their authors' property. Downloaded
`.bin` files are ignored by Git; the reader works with any compatible Polyglot book.
Set `CHESS_BOOK_PATH=/absolute/path/book.bin` to replace it.

Disable book play with `"use_book": false` for evaluator/search benchmarks, or
give every competitor the same prescribed opening. Book moves do not count as
evidence that a classical or learned evaluator became stronger.

## Tests and reproducible benchmarks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.run --depth 1 --output benchmarks/results/depth1.json
.venv/bin/python -m benchmarks.run --depth 3 --output benchmarks/results/depth3.json
.venv/bin/python -m benchmarks.run --engine current --depth 8 --time-ms 1500 --output benchmarks/results/timed.json
```

The engine/API tests cover color symmetry, pawn progression, king phase behavior, pawn-mask
correctness, tactical choices for both colors, rule-based mates, draw history,
cache consistency, promotions/castling/en passant, timeout/exception restoration,
and API validation.

`benchmarks/legacy/` preserves the original evaluator and search, changing only
the evaluator import so it runs independently. The benchmark records source
SHA-256 hashes, Python/platform/library versions, moves, times, and available
search statistics. It exits nonzero when the current engine fails a tactical
check. Legacy has no deadline support, so timed runs only accept `--engine current`.

Observed local results (see the JSON reports for exact timings):

| Configuration | Original | Current |
| --- | ---: | ---: |
| Main depth 1 | 5/7 | 7/7 |
| Main depth 3 | 6/7 | 7/7 |
| 1.5-second budget, main depth at most 8 | Not supported | 7/7 |

These seven hand-built checks cover basic mates, avoiding a poisoned pawn,
capturing a queen, capture-promotion, escaping check without losing a queen, and
en passant. They are a development regression suite, not an independent strength
benchmark. Three other positions are performance samples and are excluded from
success counts. In particular, the quiet-promotion sample can stay winning after
a king move; an earlier version incorrectly demanded immediate promotion.

The original engine's depth-three mate failure means it chose a slower continuation
instead of immediate mate. It does not imply it would lose that position. Similarly,
the poisoned-pawn test checks avoidance of one known blunder, not optimal play.

Fixed-depth comparisons are **not equal-work or equal-time comparisons**:
quiescence adds search work, and the current engine performs iterative deepening.
In the sample middlegame at depth three, the current search took about 0.86 seconds
versus 0.41 seconds for the original; it also chose a different move. This establishes
neither an Elo rating nor a win rate against humans. The paired match runner below
adds equal-budget game testing; larger independent opening and tactical sets are
still needed for broader validation.

To profile without mixing instrumentation overhead into benchmark timings:

```bash
.venv/bin/python -m cProfile -o /tmp/chess-engine.prof -m benchmarks.run --engine current --depth 3
.venv/bin/python -c 'import pstats; pstats.Stats("/tmp/chess-engine.prof").sort_stats("cumulative").print_stats(20)'
```

## Experimental neural evaluation

The `ml/` pipeline generates positions from local Stockfish self-play starting
from varied book openings, labels them with Stockfish analysis, splits by source
game, and trains either a residual correction to the heuristic or a full-score
MLP. Model version three has 794 features, adding material counts, balance, and
game phase to the original encoding. A second generator adds labeled legal-move deviations to expose
the model to weaker play. Features cover piece placement, side to move, castling
rights, legal en passant, and the halfmove clock. NumPy runs inference; PyTorch
trains and loads checkpoints. Legal moves, checkmate, and draws remain search rules.

The latest 1,000-game dataset contains 23,542 positions, including later-game and
low-material positions. Adding legal-move deviations yields 32,918 positions.
Both new model types predict held-out Stockfish depth-ten scores more accurately
than the heuristic, but neither beat it in 250 ms local paired matches. The faster
full-score NN and the residual NN each lost all eight completed games on the
diagnostic openings after training on the expanded dataset. The browser app
therefore keeps the classical evaluator. See [the ML experiment report](ml/RESULTS.md)
for data provenance, prediction metrics, timing, PGNs, and match limitations.

The next experiment analyzed the first costly move in 16 NN losses and trained
candidate rankings with a fixed material baseline and bounded positional correction.
Ranking accuracy improved from 58.3% for the score-only control to 71.8% on 103
held-out pairs, but the ranked model scored 1–11 against the heuristic on six
fresh paired openings. See [the loss-analysis and ranking report](ml/RESULTS-v5.md).
The trained NN remains experimental.

A subsequent [consistency and search-position audit](ml/AUDIT-v5.md) found
color asymmetry and material inconsistencies despite exact agreement between
training and engine inference. Its prediction advantage also largely disappeared
on positions sampled from actual search. These checks must improve before another
architecture experiment or promotion to the app.

To reproduce or extend the experiment, install `requirements.txt` and provide a
local Stockfish UCI binary. Each command writes to a new path to preserve results:

```bash
.venv/bin/python -m ml.generate_data --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal --opening-book books/gm2001.bin --games 1000 --plies 112 --sample-every 4 --label-depth 10 --play-ms 10 --seed 227 --output ml/data/another-run
.venv/bin/python -m ml.augment_data --data ml/data/another-run --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal --fraction 0.4 --label-depth 10 --seed 229 --output ml/data/another-run-deviations
.venv/bin/python -m ml.train --data ml/data/another-run-deviations --target residual --seed 229 --checkpoint ml/artifacts/another-run.pt --metrics ml/artifacts/another-run-metrics.json
.venv/bin/python -m ml.compare --checkpoint ml/artifacts/another-run.pt --pairs 4 --time-ms 250 --output ml/artifacts/another-run-match
```

Manifests record binary hashes and generation settings. Time-limited Stockfish
self-play depends on hardware, so the checked-in data, checkpoints, and reports
identify the exact experiments. The earlier 8,259-position, depth-eight experiment
is preserved separately under `ml/data/stockfish-selfplay-v2-2026`.

## Automated paired matches

Run the current engine against the frozen original with equal thinking budgets:

```bash
.venv/bin/python -m benchmarks.match --time-ms 250 --max-plies 200 --output benchmarks/results/match-250ms
```

For a repeatable strength checkpoint, play the same paired openings against a
UCI engine in limited-strength mode:

```bash
.venv/bin/python -m benchmarks.match \
  --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal \
  --stockfish-elo 1320 --openings benchmarks/openings-holdout.json \
  --time-ms 250 --max-plies 200 --output benchmarks/results/stockfish-elo-1320
```

The requested rating must fall within the UCI engine's advertised range
(Stockfish 19 starts at 1320). Treat it as a controlled engine checkpoint rather
than a direct human/FIDE rating: hardware, time control, opening selection, and
the number of completed games all affect the result.

The first 12-game checkpoint at 250 ms per move produced **11 wins and 1 loss**
against Stockfish 19 at its 1320 setting, with all games completed and no engine
errors. See `benchmarks/results/strength-stockfish-1320-250ms/RESULTS.md` for the
method and limitations. This supports testing the engine against the intended
roughly 1200-strength friend; it does not establish a human Elo rating.

This plays eight games: four prescribed openings, each with the engines swapping
colors. Games run sequentially so engines do not compete for CPU time. The order
of the two colors alternates between opening pairs. `--pairs 2` selects the first
two openings for four games. `--depth-cap` defaults to 64 for both engines; the
time budget usually limits search first. `--max-plies` counts engine-played half
moves after the prescribed opening, not full moves.

The original engine has no clock support. The match adapter leaves its frozen
source unchanged and adds iterative deepening, checking its deadline before and
after each leaf evaluation. Interrupted iterations use disposable board copies;
only the last completed iteration's move is accepted. Both engines use a legal
fallback if depth one cannot finish. This compares the updated engine against the
**original search/evaluator with a timing adapter**, not the original fixed-depth
application. The current engine checks time between nodes. Neither limit is a
hard process timeout; actual elapsed time, overruns, and fallback counts are
recorded so timing differences remain visible. PGNs use `MoveTimeMs` for the
fractional-second budget; the standard `TimeControl` tag is unknown (`?`).

Each output directory must be new, protecting earlier results. It contains:

- `games.pgn`: every game, including unfinished games, with opening moves, engine
  identities, results, termination reasons, and per-move depth/time comments.
- `report.json`: configuration, opening definitions, source hashes, environment
  versions, per-move diagnostics, final positions, and current-engine W/D/L totals.

Reports are saved after each game. Ctrl-C during search saves the partial game and
exits with status 130. Engine failures are saved separately and give exit status 1;
they do not count as opponent wins. There is no score-based adjudication or clock
forfeit. Automatic rule draws use the same policy as the app. Games reaching the
ply cap have result `*` and count as unfinished, **never as draws**. Scores exclude
unfinished/error/interrupted games; a separate score includes only opening pairs
where both games reached a result. Check the counts before interpreting either.

Custom opening JSON is supported through `--openings path/to/openings.json`:

```json
[
  {"name": "King's pawn", "moves": ["e2e4", "e7e5"]},
  {"name": "Endgame", "fen": "7k/8/8/8/8/8/P7/K7 w - - 0 1", "moves": []}
]
```

All openings are validated before play. Move histories are retained for repetition
rules, and FEN starts are exported with PGN setup headers. Timed runs may choose
different moves across machines or runs, even with identical inputs. Repeating
these four openings does not create a broad independent sample or establish Elo.
Use more held-out openings and longer budgets before making strength claims.

The original match-runner milestone had 34 tests, including the timing adapter, paired colors,
automatic draw history, legal PGN replay, result accounting, interrupted/error
handling, and CLI protection against overwriting results.

The saved `benchmarks/results/match-validation-final/` run used 100 ms per move,
all four opening pairs, and a 200-ply cap. Current won 8, drew 0, and lost 0;
all games ended by checkmate, with no unfinished games or engine errors. PGNs were
replayed and checked against JSON final positions and rule results. Mean observed
move times were about 94 ms for current and 100 ms for legacy. Current used one
depth-zero fallback; maximum recorded overrun was about 3.1 ms. This small local
validation supports improvement over this baseline at this budget, not a 1200 Elo
claim or a statistically broad strength estimate. Earlier validation attempts are
retained separately; use the `match-validation-final` artifacts for this result.

## Stronger-opponent diagnostics and loss review

Use an external Stockfish executable as the opponent, without changing the app's
own engine:

```bash
.venv/bin/python -m benchmarks.match --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal --openings benchmarks/openings-diagnostic.json --time-ms 250 --max-plies 160 --output benchmarks/results/stockfish-new-run
.venv/bin/python -m benchmarks.analyze --report benchmarks/results/stockfish-new-run/report.json --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal --output benchmarks/results/stockfish-new-run/analysis.json
```

The opponent uses full-strength Stockfish, one thread, 32 MB hash, no pondering,
and a fresh game marker between games. Its UCI ID, binary checksum and settings
are saved in the report. Equal time does not mean equal nodes or playing strength;
Stockfish is deliberately much stronger to expose weaknesses. It is not a 1200
human proxy. Local binary provenance is documented in `tools/README.md`.

The eight diagnostic openings are separate from the original four. Six further
openings in `benchmarks/openings-holdout.json` are reserved for evaluating future
changes; they have been syntax/legality checked but not played or tuned against.
These are hand-selected legal opening lines, not a large independent published
benchmark. Preserve the distinction between diagnostic and holdout sets.

The analysis CLI reviews every current-engine move. It compares Stockfish's best
root search with a search restricted to the played move, both scored from the
mover's perspective. Default reference time is 100 ms per search. It then selects
up to six significant mistakes from distinct games, prioritizing positions not
already clearly lost, and confirms them at 500 ms per search. The same positions
are probed with the unchanged current engine at 1,500 ms, and those new choices
are reviewed too. Set `--time-ms`, `--confirm-count`, `--confirm-ms`, or `--probe-ms`
to change these budgets. Each reference search starts with a fresh game marker.

The reviewer consumes raw UCI updates and retains the last completed, non-bound
score. Transient aspiration bounds and merged/stale bound flags are excluded.
Centipawn losses are approximate search estimates, not exact truths. Mate scores
are retained as mate distances and never converted into giant centipawn losses.
Longer probes indicate sensitivity to thinking time; they do not isolate search
from evaluation as a causal experiment. Full histories, FENs, principal variations,
depths, source hashes, and reference timings are saved for investigation. Early
candidate mistakes can disappear under confirmation and should not be treated as
confirmed simply because they were selected. Neither engine weights nor the
search algorithm are automatically changed by this workflow.

The stronger-opponent diagnostic run is complete: 16 games on eight new opening
lines at 250 ms per move, with 0 wins, 0 draws, and 16 losses against Stockfish 19.
All games and source hashes were verified. The suite now passes 43 tests. See
[`FINDINGS.md`](benchmarks/results/stockfish-diagnostic-250ms/FINDINGS.md) for the
corrected review of all 438 engine moves and six longer-thinking probes. Use
`analysis-exact.json`; the preliminary `analysis.json` is marked superseded.
The selected cases are saved in `benchmarks/diagnostic-mistakes.json` for diagnosis,
not as held-out evaluation data. The six holdout opening lines remain unplayed.

## Final classical-search round

The current engine now adds four search features:

1. **Scored emergency fallback.** Before deep search, a small part of the same time
   budget scores legal moves with the classical evaluator and a conservative
   penalty for exposing the moved piece. Immediate terminal results take priority.
   If the first iteration times out, it retains that scored choice. This is a
   heuristic fallback, not guaranteed tactical safety. An already-expired/tiny
   budget may still return an unscored legal move. Its score remains `None` until
   a real iteration completes; `fallback_evaluations` reports the preparatory work.
2. **Cheaper quiescence move generation.** At quiet nodes, generate captures and
   all promotions directly. Find only one legal move to exclude stalemate instead
   of building every quiet move. In check, continue to search every legal evasion.
   No capture pruning, quiet-threat extension, or changed evaluation weight was
   introduced in this round.
3. **Cached positional evaluation.** Avoid repeating the built-in evaluator on
   the same board, and avoid repeating terminal tests already done by search.
   Draw/history rules are checked before this cache is used. The cache is bounded
   at 20,000 entries per request and is disabled for custom evaluators whose
   history dependence is unknown. `static_cache_hits` counts these savings.
4. **Improved move ordering.** Remember up to two quiet moves at each search ply
   that previously caused a cutoff (commonly called killer moves). Reuse preferred
   moves from matching positions even when their histories differ. Such hints
   affect ordering only; cached search scores still require the full draw context.

The app automatically uses these features after backend restart. The API's search
statistics now include `static_cache_hits` and `fallback_evaluations`. Normal node
counts exclude fallback evaluations; elapsed time includes the whole operation.
Existing mate, legal-move, draw, and board-restoration behavior is retained.

The engine immediately before this round is frozen in `benchmarks/pre_round/`,
with only its evaluation import redirected to its frozen copy. Compare against
that baseline, rather than the much weaker original engine:

```bash
.venv/bin/python -m benchmarks.match --baseline previous --openings benchmarks/openings-holdout.json --time-ms 250 --max-plies 160 --output benchmarks/results/new-round-comparison
.venv/bin/python -m benchmarks.round_probe --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal --output benchmarks/results/new-round-probes.json
```

The probe command compares both versions at 250 ms and 1.5 seconds on the six
known diagnostic mistakes, with 500 ms Stockfish reference searches. These cases
are not holdout data. The paired match command is the separate validation of game
outcomes. Once used for a version comparison, those opening lines are no longer
unseen for subsequent rounds; reserve a new set before further tuning.

This round is complete: **51 tests pass**, and the equal-budget comparison against
the immediately previous engine finished with **7 wins, 2 losses, and 3 unfinished
games**, with no engine errors. Among fully completed opening pairs the score was
4–2. Diagnostic probes searched roughly 1.7 times as many nodes per time budget,
but retained several known tactical mistakes. See the [feature and results report](benchmarks/results/final-search-round/RESULTS.md)
for the measurements, limits, and artifact links. The six reserved openings have
now been used; no engine tuning followed this validation run.

The full suite now passes **75 tests**, including optional neural-path checks.

## Deployment evaluation and profiling

The `benchmarks/match.py` runner can send current-engine moves to the deployed API
while Stockfish runs locally. Both get the same per-move thinking budget; network
round-trip time is recorded separately from server search time. Book play is disabled
for these matches. `benchmarks/openings-live-2026.json` supplies eight fresh legal
opening lines, paired by color. For example:

```bash
.venv/bin/python -m benchmarks.match \
  --remote-url https://YOUR-SERVICE.onrender.com \
  --stockfish tools/stockfish-sf19/stockfish/stockfish-macos-universal \
  --stockfish-elo 1320 --openings benchmarks/openings-live-2026.json \
  --time-ms 250 --max-plies 140 --output benchmarks/results/live-1320
```

To profile the search without network effects, run
`.venv/bin/python -m benchmarks.profile_search --output /tmp/profile.json`.
The first optimization pass reduced median depth-three runtime on three fixed
positions from 33.9 to 32.1 ms, 31.7 to 29.5 ms, and 411.7 to 379.0 ms on the
development machine. It iterates piece bitmasks directly in the evaluator, avoiding
extra square-set objects. Moves, scores, and node counts were unchanged in the
saved before/after reports. These timings are local microbenchmarks, not evidence
of a specific gain on Render or a higher rating.

The [live evaluation report](benchmarks/results/live-2026-09/RESULTS.md) records
the final paired checkpoints: Render scored 2–6 against Stockfish's 1320 setting
on four fresh openings, versus 7–1 locally on those same openings at 250 ms.
Smaller Render checkpoints covered 1500 and 1800. Render searched a median of
463 nodes per move in the 1320 games, versus 5,864 locally. These are limited
engine settings, not human ratings; the deployed 250 ms configuration is not a
reliable 1200-strength claim. Hard uses a longer 3-second budget, which reached
roughly the laptop's 250 ms depth/node count in one sample position, but has not
been rated by a full match.
