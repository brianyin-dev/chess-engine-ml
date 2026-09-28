# Frozen pre-round baseline

This is the engine immediately before the final classical-search improvement
round, after the Stockfish diagnostics. Search and evaluation are preserved;
the only source adjustment is the search module's import of this frozen evaluator.

Use `python -m benchmarks.match --baseline previous ...` to compare it with the
current engine. This differs from `benchmarks/legacy/`, which preserves the much
weaker original project engine. Do not change this baseline to match later code.
Match/probe reports record the relevant source hashes.
