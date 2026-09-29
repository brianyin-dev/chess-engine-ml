# Live deployment, strength, and profiling checkpoint

The engine was tested on Render's free web service at
`https://chess-engine-ml.onrender.com` against a local Stockfish 19 UCI process.
Both engines were allotted 250 ms of thinking time per move; the network round
trip was recorded separately and was not charged to the engine's search budget.
The engine's Polyglot book was disabled for these matches. The eight new opening
lines in `benchmarks/openings-live-2026.json` were legal-checked before play;
the first four were used against 1320 and the first two against 1500 and 1800.
Each opening was played twice with colors reversed. Games ran sequentially.

| Run | Games | Wins | Draws | Losses | Unfinished | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Render vs Stockfish UCI_Elo 1320 | 8 | 2 | 0 | 6 | 0 | 0 |
| Render vs Stockfish UCI_Elo 1500 | 4 | 1 | 0 | 2 | 1 | 0 |
| Render vs Stockfish UCI_Elo 1800 | 4 | 1 | 0 | 3 | 0 | 0 |
| Laptop vs 1320, same first four openings | 8 | 7 | 0 | 1 | 0 | 0 |

The unfinished 1500 game hit the 160-ply cap and is excluded from its W/D/L.
The final reports and PGNs are in the sibling directories named
`live-retry-1320-250ms`, `live-retry-1500-250ms`, `live-retry-1800-250ms`, and
`local-same-openings-1320-250ms`. Each report records source hashes, opening
hash, Stockfish binary hash/settings, per-move times, depths, and nodes. None of
these final runs had an API error or needed a network retry. Earlier exploratory
runs encountered HTTPS certificate and network read errors; those were excluded
and replaced by these fresh, complete runs. The benchmark adapter now uses the
system CA bundle on this Mac (`SSL_CERT_FILE=/etc/ssl/cert.pem`) and can retry a
transient timeout or HTTP 502–504 response twice, recording any retries.

The 1320 same-opening comparison suggests hosting CPU matters substantially:
median search depth was **1 on Render versus 3 locally**, and median nodes per
engine move were **463 versus 5,864**. Render used a depth-zero fallback on
42/341 moves; the laptop used none on 256 moves. Game outcomes also vary because
both the engines' timed search and Stockfish's limited-strength play can be
nondeterministic. Eight games are too few to estimate Elo or isolate CPU as the
only cause of the W/D/L difference. In particular, the earlier local 11–1 run
used different openings and should not be compared as a controlled pair.

The browser's Hard setting now gives Render 3 seconds per searched move. On the
QGD Exchange sample position at the end of the prescribed opening, a 3-second
Render probe reached depth 3 and 5,275 nodes, choosing `g5f6`; the same position
locally reached depth 3 and 5,368 nodes in 250 ms, also choosing `g5f6`. This
is one position, not a strength result for the Hard setting. Browser play can
also use the opening book, unlike these benchmark games.

The fixed-depth profiling pass used five repetitions on three positions. A
cProfile run identified position evaluation as a leading cost. Iterating piece
bitmasks directly in the evaluator reduced median depth-three search time from
33.9 to 32.1 ms (poisoned pawn), 31.7 to 29.5 ms (start), and 411.7 to 379.0 ms
(middlegame). The move, score, and node count stayed identical in each case.
See `profile-2026-before.json` and `profile-2026-after.json`; rerun with
`python -m benchmarks.profile_search --output /tmp/profile.json`. This is a
local microbenchmark, not a demonstrated Elo gain.

The résumé-safe claim is a measured 5–8% fixed-depth runtime reduction and a
repeatable deployed-versus-local strength evaluation, alongside CI-gated
deployment, bounded API requests, and the improved playing interface. Do not
claim the deployed engine is rated 1200 or that the optimization improved Elo
without a larger controlled match.
