# v25: bishop trace and queen confinement diagnosis

Material accounting is correct. The original position gives White a 230cp material advantage; after Bxc4 Qxc4, material is equal. The evaluator's +77cp leaf is the sum of mobility (+18), piece placement (+65), development (+12), center control (+10) and rook files (-28). It does not fail to subtract the captured bishop.

## The missed continuation

Stockfish at two million nodes prefers Qc1, estimating White's position around +4.6. Its continuation includes **Qc1 gxf4 Ra3 Qb4 Ra4 Bh6 Rxb4 cxb4**: White gives up the other bishop, pursues the confined queen and then exchanges a rook for it. This is a finite-analysis principal variation, not a proof that every defense loses the queen.

After Bxc4 Qxc4, the queen instead reaches c4 and has more escape squares. Stockfish estimates this resulting position around -5.6 and gives Ra4 Qa6 Qe3 gxf4 as the beginning of its continuation. The original description of a simple unnoticed one-move bishop loss was incomplete: the engine sees Qxc4 and correctly counts material, but misses the stronger quiet continuation and the consequences of letting the queen escape.

The evaluator's mobility term counts attacked squares without checking their safety. Its queen has 10 pseudo-mobility destinations on b3 but only one destination unattacked by White's pieces, compared with four such destinations on c4. This geometric count is not legal safe mobility or proof of a trap: pinned attackers, favorable trades, counterplay and checks can invalidate it. King danger and future forcing threats also contribute to Stockfish's judgment; confinement is a demonstrated missing relationship, not an exclusive causal explanation of the whole score difference.

## Targeted candidate

Tested a fixed general 220cp confinement penalty for advanced queens with at most one geometrically unattacked destination. Queens on the first two ranks from their own side are exempt, so normally undeveloped queens are not penalized. No parameter sweep or NN training. The original material, positional terms and v24 LMR search are unchanged. The candidate is a separate evaluator, not installed in the app.

| Original position search | Baseline move / regret | Candidate move / regret |
| --- | --- | --- |
| Depth 2 | Bxc4 / 943cp | Nd5 / 808cp |
| Depth 3 | Qe3 / 481cp | Qc1 / 0cp |
| Depth 4 | Qe3 / 481cp | Qe2 / 386cp |
| 250ms | Bxc4 / 943cp | Nd5 / 808cp |
| 750ms | Qe3 / 481cp | Qc1 / 0cp |
| 2 seconds | Qe3 / 481cp | Qe2 / 386cp |

Regret is estimated by Stockfish at 256,000 nodes per same-root best/forced-move analysis. Full game history is preserved. At two seconds the baseline completed depth 5 while the more expensive candidate completed depth 4. This development result does not establish general strength. In particular, the candidate still makes a serious mistake at depth 2 and does not preserve the best move at depth 4.

## Independent decision screen

The acceptance rule was saved before screening: on the same 32 prior held-out roots, require no worse mean regret, no additional moves losing at least 150cp, and no additional avoidable -150cp transitions, at exact depth 3 and isolated 750ms. The development position must also improve by at least 50cp at 750ms. These roots are held out from this feature's design, not globally unused positions: prior experiments evaluated them. No tuning followed their results.

| Screen | Baseline mean regret | Candidate mean regret | Baseline >=150cp mistakes | Candidate |
| --- | ---: | ---: | ---: | ---: |
| Depth 3, 32 positions | 82.25cp | 85.84375cp | 5 | 5 |
| 750ms, 32 positions | 48.65625cp | 71.0625cp | 3 | 5 |

Both gates failed. At equal depth, one changed choice lost an additional 115cp. In the timed screen, two regressions coincided with shallower completed searches (depth 3 to 2 and 4 to 3); the latter choice lost 624cp versus the baseline's 131cp. Another changed timed move improved by 13cp. Added feature cost and evaluation changes can both affect timed choices, so the precise fraction attributable to each is not proved by these observations.

No game pilot was run, the candidate was rejected, and the app retains its v24 heuristic plus LMR. All **121 tests pass**, including the captured bishop's material accounting, exact score decomposition, color symmetry, queen confinement before/after the capture, exemption for undeveloped queens, and zero-weight baseline parity.

## Next implication

Do not globally increase material penalties or install this flat confinement term. A more selective follow-up would search supported quiet attacks on an already confined queen, including the queen's replies, with a bounded extension. That would verify an actual forcing sequence rather than awarding a large static bonus for a geometric guess. It requires separate searched-choice and equal-time validation; it is not implemented in this experiment.

Saved evidence: `results/bishop-v25/component-diagnosis.json`, `target-probes.json`, `heldout.json`, `heldout-summary.json`, `protocol.json` and `decision.json`. Reproduce with `python -m benchmarks.experiment_bishop_v25` in a fresh result directory; the runner refuses to overwrite existing evidence.
